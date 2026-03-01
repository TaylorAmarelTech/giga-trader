"""
Wasserstein Regime Detector -- distribution-distance regime change detection.

Uses the Wasserstein (Earth Mover's) distance between adjacent rolling
windows of returns to detect when the market regime has fundamentally
shifted. Unlike simple threshold-based detection, this captures changes
in the ENTIRE distribution shape (mean, variance, skewness, kurtosis).

Features (8, prefix wreg_):
  wreg_distance_20d     -- Wasserstein distance between [t-40:t-20] and [t-20:t] windows
  wreg_distance_60d     -- Same with 60-day half-windows
  wreg_distance_z       -- Z-score of wreg_distance_20d vs 120-day history
  wreg_regime_change    -- Binary: 1 if distance_z > 2.0 (significant shift)
  wreg_regime_duration  -- Days since last regime change
  wreg_stability        -- 1 / (1 + wreg_distance_20d) -- higher = more stable
  wreg_mean_shift       -- Change in rolling mean between windows
  wreg_vol_shift        -- Change in rolling std between windows
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

logger = logging.getLogger(__name__)


class WassersteinRegimeDetector:
    """
    Detect regime changes via Wasserstein distance between rolling distributions.

    Parameters
    ----------
    short_window : int
        Short comparison window (default 20).
    long_window : int
        Long comparison window (default 60).
    z_threshold : float
        Z-score threshold for regime change detection (default 2.0).
    z_lookback : int
        Lookback period for z-score calculation (default 120).
    """

    REQUIRED_COLS = {"close"}

    FEATURE_NAMES = [
        "wreg_distance_20d",
        "wreg_distance_60d",
        "wreg_distance_z",
        "wreg_regime_change",
        "wreg_regime_duration",
        "wreg_stability",
        "wreg_mean_shift",
        "wreg_vol_shift",
    ]

    def __init__(
        self,
        short_window: int = 20,
        long_window: int = 60,
        z_threshold: float = 2.0,
        z_lookback: int = 120,
    ):
        self.short_window = short_window
        self.long_window = long_window
        self.z_threshold = z_threshold
        self.z_lookback = z_lookback

    def create_wasserstein_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Add Wasserstein regime features to df_daily.

        Parameters
        ----------
        df_daily : pd.DataFrame
            Must have 'close' column.

        Returns
        -------
        pd.DataFrame
            Original df_daily with 8 new wreg_ columns appended.
        """
        missing = self.REQUIRED_COLS - set(df_daily.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        df = df_daily.copy()
        n = len(df)

        # Compute simple returns from close prices
        close = df["close"].values.astype(np.float64)
        returns = np.empty(n, dtype=np.float64)
        returns[0] = np.nan
        returns[1:] = (close[1:] - close[:-1]) / close[:-1]

        # -- Wasserstein distances for short and long windows --
        dist_short = self._rolling_wasserstein(returns, self.short_window)
        dist_long = self._rolling_wasserstein(returns, self.long_window)

        df["wreg_distance_20d"] = dist_short
        df["wreg_distance_60d"] = dist_long

        # -- Z-score of short distance vs its own trailing history --
        dist_z = np.full(n, np.nan, dtype=np.float64)
        for t in range(n):
            if np.isnan(dist_short[t]):
                continue
            # Need at least z_lookback valid distance observations before t
            lookback_start = max(0, t - self.z_lookback + 1)
            window_vals = dist_short[lookback_start : t + 1]
            valid = window_vals[~np.isnan(window_vals)]
            if len(valid) < 10:
                # Not enough history for a meaningful z-score
                continue
            mean_d = np.mean(valid)
            std_d = np.std(valid, ddof=1)
            if std_d < 1e-12:
                dist_z[t] = 0.0
            else:
                dist_z[t] = (dist_short[t] - mean_d) / std_d

        df["wreg_distance_z"] = dist_z

        # -- Regime change flag: 1 when z-score exceeds threshold --
        regime_change = np.where(
            np.isnan(dist_z), np.nan, np.where(dist_z > self.z_threshold, 1.0, 0.0)
        )
        df["wreg_regime_change"] = regime_change

        # -- Regime duration: days since last regime_change == 1 --
        regime_duration = np.full(n, np.nan, dtype=np.float64)
        last_change_idx: Optional[int] = None
        for t in range(n):
            if np.isnan(regime_change[t]):
                continue
            if regime_change[t] == 1.0:
                last_change_idx = t
                regime_duration[t] = 0.0
            elif last_change_idx is not None:
                regime_duration[t] = float(t - last_change_idx)
            else:
                # No regime change observed yet -- report days since first
                # valid observation
                regime_duration[t] = np.nan

        df["wreg_regime_duration"] = regime_duration

        # -- Stability: inverse of distance, bounded [0, 1] --
        stability = np.where(
            np.isnan(dist_short),
            np.nan,
            1.0 / (1.0 + dist_short),
        )
        df["wreg_stability"] = stability

        # -- Mean shift and vol shift between adjacent windows --
        mean_shift = np.full(n, np.nan, dtype=np.float64)
        vol_shift = np.full(n, np.nan, dtype=np.float64)
        w = self.short_window
        min_t = 2 * w  # need 2*window returns (plus 1 for the first NaN return)
        for t in range(min_t, n):
            prev_window = returns[t - 2 * w : t - w]
            curr_window = returns[t - w : t]
            valid_prev = prev_window[~np.isnan(prev_window)]
            valid_curr = curr_window[~np.isnan(curr_window)]
            if len(valid_prev) < 2 or len(valid_curr) < 2:
                continue
            mean_shift[t] = np.mean(valid_curr) - np.mean(valid_prev)
            vol_shift[t] = np.std(valid_curr, ddof=1) - np.std(valid_prev, ddof=1)

        df["wreg_mean_shift"] = mean_shift
        df["wreg_vol_shift"] = vol_shift

        # Ensure all feature columns are float64
        for col in self.FEATURE_NAMES:
            df[col] = df[col].astype(np.float64)

        logger.info(
            "WassersteinRegimeDetector: added %d features, %d rows",
            len(self.FEATURE_NAMES),
            n,
        )
        return df

    def _rolling_wasserstein(self, returns: np.ndarray, window: int) -> np.ndarray:
        """
        Compute rolling Wasserstein distance between adjacent windows.

        At each point t, compare the distribution of returns in
        [t - 2*window : t - window] vs [t - window : t].

        Parameters
        ----------
        returns : np.ndarray
            1-D array of daily returns (first element is NaN).
        window : int
            Half-window size.

        Returns
        -------
        np.ndarray
            Array of Wasserstein distances, same length as returns.
            Early positions (before 2*window valid data) are NaN.
        """
        n = len(returns)
        result = np.full(n, np.nan, dtype=np.float64)
        min_t = 2 * window  # need at least 2*window data points behind us

        for t in range(min_t, n):
            prev_window = returns[t - 2 * window : t - window]
            curr_window = returns[t - window : t]

            # Skip if either window has NaN (only possible near start)
            valid_prev = prev_window[~np.isnan(prev_window)]
            valid_curr = curr_window[~np.isnan(curr_window)]

            if len(valid_prev) < 2 or len(valid_curr) < 2:
                continue

            result[t] = wasserstein_distance(valid_prev, valid_curr)

        return result
