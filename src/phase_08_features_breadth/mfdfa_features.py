"""
GIGA TRADER - Multifractal Detrended Fluctuation Analysis (MFDFA) Features
===========================================================================
Pure-numpy implementation of MFDFA features for measuring market complexity
and long-range memory in price time series.

DFA (Detrended Fluctuation Analysis) extends R/S analysis by explicitly
detrending local segments before computing the fluctuation function.  The
generalised form (q ≠ 2) captures different moments of the fluctuation
distribution, revealing the full multifractal spectrum.

Features (4, prefix mfdfa_):
  mfdfa_alpha    — DFA scaling exponent (mono-fractal alpha, q=2).
                   Estimated over rolling 100-day window.
                   ~0.5 = random walk, >0.5 = persistent, <0.5 = anti-persistent.
  mfdfa_width    — Multifractal spectrum width = |alpha(q=2) - alpha(q=-2)|.
                   Wider → more multifractal / heterogeneous dynamics.
  mfdfa_asymmetry — Spectrum asymmetry = alpha(q=-2) - 2*alpha(q=0) + alpha(q=2).
                   Positive → left-skewed spectrum (large fluctuations dominate).
  mfdfa_z        — 60-day rolling z-score of mfdfa_width (clipped ±4).

Implementation Notes:
  DFA Algorithm:
    1. Integrate: Y(i) = cumsum(x - mean(x)) where x = log returns
    2. Divide Y into non-overlapping segments of length s
    3. Linear detrend each segment (fit & subtract a degree-1 polynomial)
    4. Compute RMS of residuals = F(s) for q=2, generalised for other q
    5. Repeat for multiple scales s (8, 16, 32)
    6. Slope of log(F(s)) vs log(s) = alpha(q)

  For generalised q ≠ 2:
    F_q(s) = (mean_segments(|F_seg(s)|^q))^(1/q)
    (Special case q=2 is the standard RMS formula)

  For q=0:
    F_0(s) = exp(0.5 * mean_segments(log(F_seg(s)^2)))
    (Geometric mean, avoids singularity at q=0)
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from src.core.feature_base import FeatureModuleBase

logger = logging.getLogger(__name__)

# Scales used for DFA (must be >= 4 to fit a line; min segment needed = 8)
_DFA_SCALES = [8, 16, 32]

# q values: [-2, 0, 2] give us width and asymmetry with 3 points
_Q_VALUES = [-2.0, 0.0, 2.0]


# ---------------------------------------------------------------------------
# Internal DFA helpers
# ---------------------------------------------------------------------------


def _integrate_profile(x: np.ndarray) -> np.ndarray:
    """Return integrated random-walk profile: Y(i) = cumsum(x - mean(x))."""
    return np.cumsum(x - np.mean(x))


def _segment_fluctuation(Y: np.ndarray, s: int) -> np.ndarray:
    """
    Compute per-segment fluctuation magnitudes for scale s.

    Divides Y into floor(len(Y)/s) non-overlapping segments of size s,
    fits a degree-1 polynomial (linear detrend) to each, and returns the
    RMS of residuals for every segment.

    Returns
    -------
    np.ndarray
        1-D array of per-segment F values (length = n_segs).
        Returns empty array if no complete segment fits.
    """
    n = len(Y)
    n_segs = n // s
    if n_segs < 1:
        return np.array([], dtype=float)

    t = np.arange(s, dtype=float)
    f_segs = np.empty(n_segs, dtype=float)

    for seg_idx in range(n_segs):
        seg = Y[seg_idx * s : (seg_idx + 1) * s]
        # Linear detrend: fit y = a*t + b, compute residuals
        coeffs = np.polyfit(t, seg, 1)
        trend = np.polyval(coeffs, t)
        residuals = seg - trend
        f_segs[seg_idx] = np.sqrt(np.mean(residuals ** 2))

    return f_segs


def _generalised_fluctuation(f_segs: np.ndarray, q: float) -> float:
    """
    Aggregate per-segment fluctuations into a single generalised F_q value.

    Parameters
    ----------
    f_segs : np.ndarray
        Per-segment fluctuation magnitudes (must all be >= 0).
    q : float
        Moment order.

    Returns
    -------
    float
        Generalised fluctuation F_q, or NaN if computation fails.
    """
    if len(f_segs) == 0:
        return np.nan

    # Remove zero segments to avoid log/power issues
    f_pos = f_segs[f_segs > 0]
    if len(f_pos) == 0:
        return np.nan

    if abs(q) < 1e-10:
        # q ≈ 0: geometric mean
        return float(np.exp(0.5 * np.mean(np.log(f_pos ** 2))))
    else:
        moments = f_pos ** q
        mean_moment = np.mean(moments)
        if mean_moment <= 0:
            return np.nan
        return float(mean_moment ** (1.0 / q))


def _dfa_alpha(x: np.ndarray, q: float, scales: List[int]) -> float:
    """
    Compute the DFA scaling exponent alpha(q) for a 1-D array of returns.

    Parameters
    ----------
    x : np.ndarray
        Log returns (mean-centred internally via integration).
    q : float
        Moment order.
    scales : list of int
        List of segment sizes to use.

    Returns
    -------
    float
        Scaling exponent alpha(q), clipped to [0.0, 1.5].
        Returns 0.5 (random-walk default) if data are insufficient.
    """
    n = len(x)
    if n < max(scales) * 2:
        return 0.5

    Y = _integrate_profile(x)

    log_s_vals: List[float] = []
    log_fq_vals: List[float] = []

    for s in scales:
        if s >= n:
            continue
        f_segs = _segment_fluctuation(Y, s)
        fq = _generalised_fluctuation(f_segs, q)
        if np.isfinite(fq) and fq > 0:
            log_s_vals.append(np.log(s))
            log_fq_vals.append(np.log(fq))

    if len(log_s_vals) < 2:
        return 0.5

    # Linear fit: slope = alpha(q)
    slope = np.polyfit(log_s_vals, log_fq_vals, 1)[0]
    return float(np.clip(slope, 0.0, 1.5))


def _compute_mfdfa_triplet(
    log_returns: np.ndarray,
    scales: List[int],
) -> tuple:
    """
    Compute (alpha_q2, alpha_q0, alpha_qm2) for a window of log returns.

    Returns
    -------
    tuple of 3 floats
        (alpha at q=2, alpha at q=0, alpha at q=-2)
    """
    alpha_q2 = _dfa_alpha(log_returns, q=2.0, scales=scales)
    alpha_q0 = _dfa_alpha(log_returns, q=0.0, scales=scales)
    alpha_qm2 = _dfa_alpha(log_returns, q=-2.0, scales=scales)
    return alpha_q2, alpha_q0, alpha_qm2


# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class MFDFAFeatures(FeatureModuleBase):
    """
    Compute Multifractal Detrended Fluctuation Analysis (MFDFA) features
    from daily close prices.

    Parameters
    ----------
    window : int
        Rolling window (in trading days) for MFDFA estimation. Default 100.
    z_window : int
        Rolling window for z-score of mfdfa_width. Default 60.
    scales : list of int, optional
        DFA segment sizes. Defaults to [8, 16, 32].
    """
    FEATURE_NAMES = ["mfdfa_alpha", "mfdfa_width", "mfdfa_asymmetry", "mfdfa_z"]


    REQUIRED_COLS = {"close"}

    def __init__(
        self,
        window: int = 100,
        z_window: int = 60,
        scales: Optional[List[int]] = None,
    ) -> None:
        self.window = window
        self.z_window = z_window
        self.scales = scales if scales is not None else list(_DFA_SCALES)

    # ------------------------------------------------------------------
    # Data download (no external data needed — purely price-based)
    # ------------------------------------------------------------------

    def download_mfdfa_data(
        self,
        start_date,  # noqa: ANN001
        end_date,    # noqa: ANN001
    ) -> pd.DataFrame:
        """
        No external data required for MFDFA computation.

        This method exists for API consistency with other feature classes
        (e.g. EconomicFeatures, CrossAssetFeatures).

        Returns
        -------
        pd.DataFrame
            Always returns an empty DataFrame.
        """
        return pd.DataFrame()

    # ------------------------------------------------------------------
    # Main feature creation
    # ------------------------------------------------------------------

    def create_mfdfa_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add 4 MFDFA features to df.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain a 'close' column (daily prices).

        Returns
        -------
        pd.DataFrame
            Original df with 4 new mfdfa_ columns appended:
            mfdfa_alpha, mfdfa_width, mfdfa_asymmetry, mfdfa_z.
        """
        df = df.copy()

        if "close" not in df.columns:
            logger.warning("MFDFAFeatures: 'close' column missing — skipping")
            return df

        n = len(df)
        close = df["close"].values.astype(float)

        # Log returns (NaN at index 0 by design)
        log_ret = np.full(n, np.nan)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = close[1:] / close[:-1]
            log_ret[1:] = np.where(ratio > 0, np.log(ratio), np.nan)

        # Output arrays — initialise with sensible defaults
        alpha_arr = np.full(n, 0.5)        # random-walk default
        width_arr = np.zeros(n)
        asym_arr = np.zeros(n)

        min_obs = max(self.scales) * 2     # need enough data for DFA scales

        for i in range(n):
            if i < self.window - 1:
                # Not enough history yet — keep defaults
                continue

            window_data = log_ret[i - self.window + 1 : i + 1]
            valid = window_data[~np.isnan(window_data)]

            if len(valid) < min_obs:
                # Still insufficient after NaN removal — keep defaults
                continue

            aq2, aq0, aqm2 = _compute_mfdfa_triplet(valid, self.scales)

            alpha_arr[i] = aq2
            width_arr[i] = abs(aq2 - aqm2)
            asym_arr[i] = aqm2 - 2.0 * aq0 + aq2

        # Assign raw features
        df["mfdfa_alpha"] = alpha_arr
        df["mfdfa_width"] = width_arr
        df["mfdfa_asymmetry"] = asym_arr

        # --- mfdfa_z: rolling z-score of mfdfa_width ---
        width_series = pd.Series(width_arr, index=df.index)
        roll_mean = width_series.rolling(self.z_window, min_periods=max(10, self.z_window // 4)).mean()
        roll_std = width_series.rolling(self.z_window, min_periods=max(10, self.z_window // 4)).std()
        z = ((width_series - roll_mean) / (roll_std + 1e-10)).clip(-4.0, 4.0)
        df["mfdfa_z"] = z.values

        # --- Cleanup: fill NaN, clip extremes, handle infinities ---
        df["mfdfa_alpha"] = (
            df["mfdfa_alpha"]
            .fillna(0.5)
            .replace([np.inf, -np.inf], 0.5)
            .clip(0.0, 1.5)
        )
        df["mfdfa_width"] = (
            df["mfdfa_width"]
            .fillna(0.0)
            .replace([np.inf, -np.inf], 0.0)
            .clip(0.0, 2.0)
        )
        df["mfdfa_asymmetry"] = (
            df["mfdfa_asymmetry"]
            .fillna(0.0)
            .replace([np.inf, -np.inf], 0.0)
            .clip(-2.0, 2.0)
        )
        df["mfdfa_z"] = (
            df["mfdfa_z"]
            .fillna(0.0)
            .replace([np.inf, -np.inf], 0.0)
        )

        n_features = sum(1 for c in df.columns if c.startswith("mfdfa_"))
        logger.info("MFDFAFeatures: added %d features", n_features)
        return df

    # ------------------------------------------------------------------
    # Dashboard / analysis helper
    # ------------------------------------------------------------------

    def analyze_current_mfdfa(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Summarise the current MFDFA regime for dashboard display.

        Parameters
        ----------
        df : pd.DataFrame
            Must already have mfdfa_alpha, mfdfa_width, mfdfa_z, mfdfa_asymmetry
            columns (i.e. create_mfdfa_features has been called).

        Returns
        -------
        dict or None
            Dictionary with:
              - fractal_regime : str
                  PERSISTENT / RANDOM_WALK / ANTI_PERSISTENT / MULTIFRACTAL
              - mfdfa_alpha : float
              - mfdfa_width : float
              - mfdfa_z : float
              - mfdfa_asymmetry : float
            Returns None if features are absent or df is empty.
        """
        if "mfdfa_alpha" not in df.columns or len(df) < 2:
            return None

        last = df.iloc[-1]
        alpha = float(last.get("mfdfa_alpha", 0.5))
        width = float(last.get("mfdfa_width", 0.0))
        z = float(last.get("mfdfa_z", 0.0))
        asym = float(last.get("mfdfa_asymmetry", 0.0))

        # Classify regime
        if width > 0.3:
            regime = "MULTIFRACTAL"
        elif alpha > 0.6:
            regime = "PERSISTENT"
        elif alpha < 0.4:
            regime = "ANTI_PERSISTENT"
        else:
            regime = "RANDOM_WALK"

        return {
            "fractal_regime": regime,
            "mfdfa_alpha": round(alpha, 4),
            "mfdfa_width": round(width, 4),
            "mfdfa_z": round(z, 3),
            "mfdfa_asymmetry": round(asym, 4),
        }

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _all_feature_names() -> List[str]:
        """Return canonical list of all mfdfa_ feature column names."""
        return [
            "mfdfa_alpha",
            "mfdfa_width",
            "mfdfa_asymmetry",
            "mfdfa_z",
        ]
