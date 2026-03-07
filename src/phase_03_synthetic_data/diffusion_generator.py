"""
GBM-Aware Diffusion-Based Synthetic Price Path Generator
==========================================================
Combines Geometric Brownian Motion (GBM) calibration with a noise-schedule
diffusion process to generate diverse but realistic synthetic OHLCV daily data.

This is NOT a neural diffusion model. It is a statistical diffusion that uses
the noise-schedule concept (forward corruption + reverse denoising) to produce
paths whose drift and volatility statistics match the training data.

Process:
  1. **Forward process**: Add calibrated noise to real log-return paths at
     multiple noise levels controlled by a linear beta schedule.
  2. **Reverse process**: Start from pure noise and iteratively denoise using
     mean-reverting dynamics calibrated from the training data.
  3. **GBM bridge**: Post-process so generated paths respect calibrated
     GBM drift (mu) and volatility (sigma) statistics.

Usage::

    from src.phase_03_synthetic_data.diffusion_generator import DiffusionSyntheticGenerator

    gen = DiffusionSyntheticGenerator(n_steps=50, seed=42)
    gen.fit(spy_daily_df)                        # calibrate from real data
    paths = gen.generate(n_paths=10, n_days=252) # list of DataFrames
    bear_paths = gen.generate_regime_conditioned("bear", n_paths=5)
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Regime drift / vol multipliers
# ---------------------------------------------------------------------------
_REGIME_PARAMS: Dict[str, Dict[str, float]] = {
    "bear": {"drift_mult": -2.0, "vol_mult": 1.3},
    "bull": {"drift_mult": 1.5, "vol_mult": 0.9},
    "sideways": {"drift_mult": 0.0, "vol_mult": 0.7},
    "crisis": {"drift_mult": -3.0, "vol_mult": 2.0},
}


class DiffusionSyntheticGenerator:
    """GBM-aware diffusion-based synthetic price path generator.

    Parameters
    ----------
    n_steps : int
        Number of diffusion (noise) steps in the forward / reverse process.
    beta_start : float
        Starting value of the linear noise schedule.
    beta_end : float
        Ending value of the linear noise schedule.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_steps: int = 50,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        seed: int = 42,
    ) -> None:
        self.n_steps = n_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.seed = seed

        # Noise schedule
        self.betas = np.linspace(beta_start, beta_end, n_steps)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = np.cumprod(self.alphas)

        # Calibrated parameters (populated by fit)
        self._fitted = False
        self.mu: float = 0.0          # annualised drift
        self.sigma: float = 0.0       # annualised volatility
        self.daily_mu: float = 0.0
        self.daily_sigma: float = 0.0

        # OHLC relationship ratios (calibrated from data)
        self.high_ratio_mean: float = 0.0
        self.high_ratio_std: float = 0.0
        self.low_ratio_mean: float = 0.0
        self.low_ratio_std: float = 0.0

        # Volume distribution (log-normal parameters)
        self.vol_log_mean: float = 0.0
        self.vol_log_std: float = 0.0

        # Return autocorrelation (for mean-reversion strength)
        self.return_autocorr: float = 0.0

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "DiffusionSyntheticGenerator":
        """Calibrate GBM and OHLC parameters from real daily data.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain at minimum ``close`` column.  Optionally ``open``,
            ``high``, ``low``, ``volume`` for full OHLC calibration.

        Returns
        -------
        self
        """
        if df is None or df.empty:
            raise ValueError("Cannot fit on empty DataFrame")

        close = df["close"].values.astype(np.float64)
        if len(close) < 10:
            raise ValueError("Need at least 10 data points to calibrate")

        # Daily log-returns
        log_returns = np.diff(np.log(close))
        log_returns = log_returns[np.isfinite(log_returns)]

        self.daily_mu = float(np.mean(log_returns))
        self.daily_sigma = float(np.std(log_returns, ddof=1))

        # Annualise (252 trading days)
        self.mu = self.daily_mu * 252
        self.sigma = self.daily_sigma * np.sqrt(252)

        # Return autocorrelation (lag-1) — used in reverse denoising
        if len(log_returns) > 2:
            self.return_autocorr = float(
                np.corrcoef(log_returns[:-1], log_returns[1:])[0, 1]
            )
            if not np.isfinite(self.return_autocorr):
                self.return_autocorr = 0.0
        else:
            self.return_autocorr = 0.0

        # OHLC ratios (calibrate high/low relative to close)
        if "high" in df.columns and "low" in df.columns:
            oc_max = df[["open", "close"]].max(axis=1).values.astype(np.float64)
            oc_min = df[["open", "close"]].min(axis=1).values.astype(np.float64)
            high_vals = df["high"].values.astype(np.float64)
            low_vals = df["low"].values.astype(np.float64)

            # Ratio of (high - max(O,C)) / close  — how much high exceeds body
            high_excess = (high_vals - oc_max) / close
            high_excess = high_excess[np.isfinite(high_excess)]
            self.high_ratio_mean = float(np.mean(np.abs(high_excess)))
            self.high_ratio_std = float(np.std(np.abs(high_excess), ddof=1))

            # Ratio of (min(O,C) - low) / close  — how much low undershoots body
            low_excess = (oc_min - low_vals) / close
            low_excess = low_excess[np.isfinite(low_excess)]
            self.low_ratio_mean = float(np.mean(np.abs(low_excess)))
            self.low_ratio_std = float(np.std(np.abs(low_excess), ddof=1))
        else:
            # Defaults based on typical SPY daily wicks
            self.high_ratio_mean = 0.004
            self.high_ratio_std = 0.003
            self.low_ratio_mean = 0.004
            self.low_ratio_std = 0.003

        # Volume (log-normal calibration)
        if "volume" in df.columns:
            vol = df["volume"].values.astype(np.float64)
            vol = vol[vol > 0]
            if len(vol) > 0:
                log_vol = np.log(vol)
                self.vol_log_mean = float(np.mean(log_vol))
                self.vol_log_std = float(np.std(log_vol, ddof=1))
            else:
                self.vol_log_mean = np.log(50_000_000)
                self.vol_log_std = 0.3
        else:
            self.vol_log_mean = np.log(50_000_000)
            self.vol_log_std = 0.3

        self._fitted = True
        logger.info(
            "DiffusionSyntheticGenerator fitted: mu=%.4f sigma=%.4f "
            "(daily mu=%.6f sigma=%.4f)",
            self.mu, self.sigma, self.daily_mu, self.daily_sigma,
        )
        return self

    # ------------------------------------------------------------------
    # Forward diffusion
    # ------------------------------------------------------------------

    def _forward_diffuse(
        self, x0: np.ndarray, t: int, rng: np.random.RandomState
    ) -> np.ndarray:
        """Apply forward diffusion to log-return series at noise level *t*.

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        """
        t = min(t, self.n_steps - 1)
        ab = self.alpha_bar[t]
        noise = rng.randn(*x0.shape) * self.daily_sigma
        return np.sqrt(ab) * x0 + np.sqrt(1.0 - ab) * noise

    # ------------------------------------------------------------------
    # Reverse denoising
    # ------------------------------------------------------------------

    def _reverse_denoise(
        self,
        n_days: int,
        drift: float,
        vol: float,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Run the reverse diffusion process to generate a log-return series.

        Starting from pure noise, iteratively denoise using mean-reverting
        dynamics calibrated from data.  Each reverse step blends the current
        noisy sample toward the GBM-implied mean while reducing noise
        according to the schedule.
        """
        # Start from pure noise scaled by calibrated vol
        x = rng.randn(n_days) * vol

        # Mean-reversion strength (stronger autocorrelation → weaker reversion)
        mr_strength = max(0.01, 1.0 - abs(self.return_autocorr))

        for t in reversed(range(self.n_steps)):
            ab = self.alpha_bar[t]
            ab_prev = self.alpha_bar[t - 1] if t > 0 else 1.0
            beta_t = self.betas[t]

            # Predicted clean signal: pull toward drift
            x0_pred = (x - np.sqrt(1.0 - ab) * rng.randn(n_days) * vol) / max(
                np.sqrt(ab), 1e-8
            )

            # Mean-revert predicted signal toward calibrated drift
            x0_pred = (1.0 - mr_strength) * x0_pred + mr_strength * drift

            # Posterior mean
            coeff1 = np.sqrt(ab_prev) * beta_t / max(1.0 - ab, 1e-8)
            coeff2 = np.sqrt(1.0 - beta_t) * (1.0 - ab_prev) / max(1.0 - ab, 1e-8)
            # Normalise coefficients to avoid blow-up
            total = coeff1 + coeff2
            if total > 0:
                coeff1 /= total
                coeff2 /= total

            mean = coeff1 * x0_pred + coeff2 * x

            # Add noise (except at t=0)
            if t > 0:
                noise_scale = np.sqrt(beta_t) * vol * 0.5
                x = mean + rng.randn(n_days) * noise_scale
            else:
                x = mean

        return x

    # ------------------------------------------------------------------
    # OHLCV construction
    # ------------------------------------------------------------------

    def _build_ohlcv(
        self,
        log_returns: np.ndarray,
        start_price: float,
        rng: np.random.RandomState,
    ) -> pd.DataFrame:
        """Convert a log-return series into a full OHLCV DataFrame.

        Guarantees OHLC consistency:
          - High >= max(Open, Close)
          - Low  <= min(Open, Close)
        """
        n = len(log_returns)
        dates = pd.bdate_range("2020-01-02", periods=n)

        # Close prices from cumulative log-returns
        close = start_price * np.exp(np.cumsum(log_returns))

        # Open = previous close (with small gap noise)
        gap_noise = rng.randn(n) * self.daily_sigma * 0.1
        open_prices = np.empty(n)
        open_prices[0] = start_price * np.exp(gap_noise[0])
        open_prices[1:] = close[:-1] * np.exp(gap_noise[1:])

        # High and low with calibrated wick ratios
        high_excess = np.abs(rng.normal(self.high_ratio_mean, self.high_ratio_std, n))
        low_excess = np.abs(rng.normal(self.low_ratio_mean, self.low_ratio_std, n))

        oc_max = np.maximum(open_prices, close)
        oc_min = np.minimum(open_prices, close)

        high = oc_max + high_excess * close
        low = oc_min - low_excess * close

        # Enforce OHLC consistency (belt and suspenders)
        high = np.maximum(high, oc_max)
        low = np.minimum(low, oc_min)
        low = np.maximum(low, 1e-4)  # price must be positive

        # Volume (log-normal)
        volume = np.exp(
            rng.normal(self.vol_log_mean, self.vol_log_std, n)
        ).astype(np.int64)
        volume = np.maximum(volume, 1)

        # Daily return
        daily_return = np.zeros(n)
        daily_return[0] = (close[0] - start_price) / start_price
        daily_return[1:] = np.diff(close) / close[:-1]

        df = pd.DataFrame(
            {
                "date": dates,
                "open": open_prices,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume,
                "daily_return": daily_return,
            }
        )
        return df

    # ------------------------------------------------------------------
    # GBM constraint enforcement
    # ------------------------------------------------------------------

    def _enforce_gbm_stats(
        self,
        log_returns: np.ndarray,
        target_mu: float,
        target_sigma: float,
    ) -> np.ndarray:
        """Rescale log-returns so their mean and std match target GBM params."""
        current_mu = np.mean(log_returns)
        current_sigma = np.std(log_returns, ddof=1)

        if current_sigma < 1e-12:
            # Degenerate — replace with GBM draws
            return np.random.RandomState(0).normal(target_mu, target_sigma, len(log_returns))

        # Standardise then rescale
        standardised = (log_returns - current_mu) / current_sigma
        return standardised * target_sigma + target_mu

    # ------------------------------------------------------------------
    # Public: generate
    # ------------------------------------------------------------------

    def generate(
        self,
        n_paths: int = 10,
        n_days: int = 252,
        start_price: float = 450.0,
    ) -> List[pd.DataFrame]:
        """Generate *n_paths* synthetic daily OHLCV DataFrames.

        Parameters
        ----------
        n_paths : int
            Number of independent paths to generate.
        n_days : int
            Trading days per path.
        start_price : float
            Starting close price for all paths.

        Returns
        -------
        list[pd.DataFrame]
            Each DataFrame has columns:
            ``date, open, high, low, close, volume, daily_return``.
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before generate()")

        rng = np.random.RandomState(self.seed)
        paths: List[pd.DataFrame] = []

        for i in range(n_paths):
            # Reverse-diffuse to get raw log-returns
            raw_lr = self._reverse_denoise(
                n_days, self.daily_mu, self.daily_sigma, rng
            )

            # Enforce GBM statistics
            lr = self._enforce_gbm_stats(raw_lr, self.daily_mu, self.daily_sigma)

            # Build OHLCV DataFrame
            df = self._build_ohlcv(lr, start_price, rng)
            paths.append(df)

        logger.info(
            "Generated %d diffusion paths (%d days each)", n_paths, n_days
        )
        return paths

    # ------------------------------------------------------------------
    # Public: regime-conditioned generation
    # ------------------------------------------------------------------

    def generate_regime_conditioned(
        self,
        regime: str = "bear",
        n_paths: int = 5,
        n_days: int = 252,
        start_price: float = 450.0,
    ) -> List[pd.DataFrame]:
        """Generate paths conditioned on a market regime.

        Parameters
        ----------
        regime : str
            One of ``"bear"``, ``"bull"``, ``"sideways"``, ``"crisis"``.
        n_paths : int
            Number of paths.
        n_days : int
            Trading days per path.
        start_price : float
            Starting close price.

        Returns
        -------
        list[pd.DataFrame]
        """
        if not self._fitted:
            raise RuntimeError("Must call fit() before generate_regime_conditioned()")

        regime = regime.lower()
        if regime not in _REGIME_PARAMS:
            raise ValueError(
                f"Unknown regime '{regime}'. Choose from {list(_REGIME_PARAMS.keys())}"
            )

        params = _REGIME_PARAMS[regime]
        drift_mult = params["drift_mult"]
        vol_mult = params["vol_mult"]

        # Adjust daily drift and vol for the regime
        regime_daily_mu = self.daily_mu * drift_mult
        regime_daily_sigma = self.daily_sigma * vol_mult

        rng = np.random.RandomState(self.seed + hash(regime) % (2**31))
        paths: List[pd.DataFrame] = []

        for _ in range(n_paths):
            raw_lr = self._reverse_denoise(
                n_days, regime_daily_mu, regime_daily_sigma, rng
            )
            lr = self._enforce_gbm_stats(
                raw_lr, regime_daily_mu, regime_daily_sigma
            )
            df = self._build_ohlcv(lr, start_price, rng)
            paths.append(df)

        logger.info(
            "Generated %d regime-conditioned (%s) diffusion paths", n_paths, regime
        )
        return paths
