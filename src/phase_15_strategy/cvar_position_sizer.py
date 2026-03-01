"""
CVaR (Conditional Value-at-Risk) Position Sizer — scales position inversely
with tail risk (Expected Shortfall).

When tail risk is elevated (CVaR exceeds target), position size is reduced
proportionally. Also computes CVaR-related features for ML consumption.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class CVaRPositionSizer:
    """
    CVaR-based position sizing — scales position inversely with tail risk.

    Parameters
    ----------
    alpha : float
        VaR confidence level (default 0.05 = 95th percentile).
        CVaR is the expected loss in the worst ``alpha`` fraction of outcomes.
    lookback : int
        Rolling window for CVaR estimation (default 60 trading days).
    max_position : float
        Maximum position size as fraction of portfolio (default 0.25).
    min_position : float
        Minimum position size as fraction of portfolio (default 0.02).
    target_cvar : float
        Target CVaR for full position sizing (default 0.02 = 2%).
        When observed CVaR equals this value, the base position is unchanged.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        lookback: int = 60,
        max_position: float = 0.25,
        min_position: float = 0.02,
        target_cvar: float = 0.02,
    ):
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        if lookback < 2:
            raise ValueError(f"lookback must be >= 2, got {lookback}")
        if not 0.0 < min_position <= max_position <= 1.0:
            raise ValueError(
                f"Need 0 < min_position <= max_position <= 1, "
                f"got min_position={min_position}, max_position={max_position}"
            )
        if target_cvar <= 0.0:
            raise ValueError(f"target_cvar must be > 0, got {target_cvar}")

        self.alpha = alpha
        self.lookback = lookback
        self.max_position = max_position
        self.min_position = min_position
        self.target_cvar = target_cvar

        self._fitted = False
        self._returns: Optional[np.ndarray] = None
        self._cvar: Optional[float] = None

    def fit(self, returns: np.ndarray) -> "CVaRPositionSizer":
        """
        Estimate CVaR from a historical returns array.

        Parameters
        ----------
        returns : np.ndarray
            1-D array of historical daily returns (e.g. log or simple returns).

        Returns
        -------
        self
            For method chaining.
        """
        returns = np.asarray(returns, dtype=float).ravel()
        if len(returns) < 2:
            logger.warning(
                "CVaRPositionSizer.fit: fewer than 2 returns provided — "
                "cannot estimate CVaR."
            )
            self._fitted = False
            return self

        # Remove NaNs
        clean = returns[~np.isnan(returns)]
        if len(clean) < 2:
            logger.warning(
                "CVaRPositionSizer.fit: fewer than 2 non-NaN returns."
            )
            self._fitted = False
            return self

        self._returns = clean
        self._cvar = self._compute_cvar(clean)
        self._fitted = True

        logger.info(
            "CVaRPositionSizer fitted: alpha=%.3f, n_returns=%d, CVaR=%.6f",
            self.alpha,
            len(clean),
            self._cvar,
        )
        return self

    def _compute_cvar(self, returns: np.ndarray) -> float:
        """
        Compute CVaR (Expected Shortfall) at the configured alpha level.

        CVaR = mean of returns that fall at or below the alpha-quantile.
        Returns the absolute value (positive number representing loss magnitude).
        """
        var_threshold = np.percentile(returns, self.alpha * 100)
        tail = returns[returns <= var_threshold]
        if len(tail) == 0:
            # Edge case: no returns at or below VaR (e.g. constant series)
            return 0.0
        return float(abs(np.mean(tail)))

    def compute_rolling_cvar(self, returns: pd.Series) -> pd.Series:
        """
        Compute rolling CVaR over the lookback window.

        Parameters
        ----------
        returns : pd.Series
            Series of daily returns.

        Returns
        -------
        pd.Series
            Rolling CVaR (absolute value), NaN for positions before the
            lookback window is filled.
        """
        returns = returns.copy()
        result = pd.Series(np.nan, index=returns.index, dtype=float)

        for i in range(self.lookback, len(returns)):
            window = returns.iloc[i - self.lookback : i].values
            clean = window[~np.isnan(window)]
            if len(clean) >= 2:
                result.iloc[i] = self._compute_cvar(clean)
            # else: stays NaN

        return result

    def size(self, base_position: float, current_cvar: float) -> float:
        """
        Scale position based on current CVaR relative to target.

        If current CVaR <= target_cvar: full base position (capped at max).
        If current CVaR > target_cvar: position = base * (target / current).
        Always clipped to [min_position, max_position].

        Parameters
        ----------
        base_position : float
            Desired position size before tail-risk adjustment.
        current_cvar : float
            Current CVaR estimate (positive number representing loss magnitude).

        Returns
        -------
        float
            Adjusted position size in [min_position, max_position].
        """
        if current_cvar <= 0.0:
            # No tail risk detected — use full base position
            scaled = base_position
        elif current_cvar <= self.target_cvar:
            # Below target — full position
            scaled = base_position
        else:
            # Above target — scale down proportionally
            scaled = base_position * (self.target_cvar / current_cvar)

        clipped = float(np.clip(scaled, self.min_position, self.max_position))
        return clipped

    def get_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Compute CVaR-related features for ML use.

        Expects ``df_daily`` to have a ``close`` column. Returns a copy of the
        dataframe with the following columns added:

        - ``cvar_5pct_20d``: 20-day rolling CVaR at 5% alpha
        - ``cvar_5pct_60d``: 60-day rolling CVaR at 5% alpha
        - ``cvar_ratio``: cvar_20d / cvar_60d (short-term vs long-term tail risk)
        - ``cvar_regime``: 1 if cvar_20d > 1.5x cvar_60d (elevated tail risk), else 0

        Parameters
        ----------
        df_daily : pd.DataFrame
            Daily OHLCV data with a ``close`` column.

        Returns
        -------
        pd.DataFrame
            Copy of input with CVaR feature columns added.
        """
        df = df_daily.copy()

        if "close" not in df.columns:
            raise ValueError("df_daily must contain a 'close' column")

        returns = df["close"].pct_change()

        # Compute rolling CVaR at two horizons using alpha=0.05
        short_sizer = CVaRPositionSizer(alpha=0.05, lookback=20)
        long_sizer = CVaRPositionSizer(alpha=0.05, lookback=60)

        df["cvar_5pct_20d"] = short_sizer.compute_rolling_cvar(returns)
        df["cvar_5pct_60d"] = long_sizer.compute_rolling_cvar(returns)

        # Ratio: short-term tail risk relative to long-term
        # Guard against division by zero
        df["cvar_ratio"] = np.where(
            df["cvar_5pct_60d"] > 0,
            df["cvar_5pct_20d"] / df["cvar_5pct_60d"],
            np.nan,
        )

        # Regime flag: elevated tail risk when short-term CVaR exceeds
        # 1.5x the long-term baseline
        df["cvar_regime"] = np.where(
            df["cvar_5pct_20d"] > 1.5 * df["cvar_5pct_60d"],
            1,
            0,
        ).astype(int)

        # Where either input is NaN, regime should also be 0 (safe default)
        nan_mask = df["cvar_5pct_20d"].isna() | df["cvar_5pct_60d"].isna()
        df.loc[nan_mask, "cvar_regime"] = 0

        logger.info(
            "CVaR features computed: %d rows, elevated regime days: %d",
            len(df),
            df["cvar_regime"].sum(),
        )

        return df

    @property
    def fitted_cvar(self) -> Optional[float]:
        """Return the CVaR estimated during fit(), or None if not fitted."""
        return self._cvar if self._fitted else None

    def __repr__(self) -> str:
        status = "fitted" if self._fitted else "not fitted"
        cvar_str = f", CVaR={self._cvar:.6f}" if self._fitted else ""
        return (
            f"CVaRPositionSizer(alpha={self.alpha}, lookback={self.lookback}, "
            f"target_cvar={self.target_cvar}{cvar_str}, {status})"
        )
