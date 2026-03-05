"""
Drift Detection Features — CUSUM-based distribution change detection.

Detects distributional shifts in return series using a simple CUSUM approach.
Useful for identifying regime transitions and adapting model behavior.

Features (3, prefix drift_):
  drift_detected     — Binary (1.0/0.0) whether distribution change detected
  drift_days_since   — Days since last detected drift (capped at 252)
  drift_window_size  — Adaptive window size relative to reference (50)
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from src.core.feature_base import FeatureModuleBase

logger = logging.getLogger(__name__)


class DriftFeatures(FeatureModuleBase):
    """Compute CUSUM-based drift detection features from daily OHLCV data."""
    FEATURE_NAMES = ["drift_detected", "drift_days_since", "drift_window_size"]


    REQUIRED_COLS = {"close"}

    def create_drift_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Add drift detection features to df_daily.

        Parameters
        ----------
        df_daily : pd.DataFrame
            Must have 'close' column.

        Returns
        -------
        pd.DataFrame
            Original df_daily with 3 new drift_ columns added.
        """
        df = df_daily.copy()

        if "close" not in df.columns:
            logger.warning("DriftFeatures: 'close' column missing, skipping")
            return df

        # Compute returns
        close = df["close"].values.astype(float)
        n = len(close)

        if n < 2:
            for col in self._all_feature_names():
                df[col] = 0.0
            return df

        returns = np.empty(n)
        returns[0] = 0.0
        returns[1:] = close[1:] / close[:-1] - 1.0

        # Run CUSUM-based drift detection
        detected, days_since, window_size = _detect_drift(returns)

        df["drift_detected"] = detected
        df["drift_days_since"] = days_since
        df["drift_window_size"] = window_size

        # Cleanup: NaN -> 0.0, inf -> 0.0
        for col in self._all_feature_names():
            df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        n_features = sum(1 for c in df.columns if c.startswith("drift_"))
        logger.info(f"DriftFeatures: added {n_features} features")
        return df

    def analyze_current_drift(self, df_daily: pd.DataFrame) -> Optional[Dict]:
        """Analyze current drift regime for dashboard display."""
        if "drift_days_since" not in df_daily.columns or len(df_daily) < 2:
            return None

        last = df_daily.iloc[-1]
        days_since = float(last.get("drift_days_since", 252))
        detected = float(last.get("drift_detected", 0.0))
        window_sz = float(last.get("drift_window_size", 1.0))

        if detected == 1.0 or days_since <= 5:
            regime = "DRIFTING"
        elif days_since <= 20:
            regime = "TRANSITIONING"
        else:
            regime = "STABLE"

        return {
            "drift_regime": regime,
            "drift_days_since": round(days_since),
            "drift_window_size": round(window_sz, 3),
        }

    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "drift_detected",
            "drift_days_since",
            "drift_window_size",
        ]


# ─── Internal helpers ─────────────────────────────────────────────────────


def _detect_drift(
    returns: np.ndarray, threshold: float = 2.0, min_window: int = 20
) -> tuple:
    """
    Simple CUSUM-based drift detection on return distribution.

    Parameters
    ----------
    returns : np.ndarray
        Array of daily returns.
    threshold : float
        Z-score threshold for drift detection.
    min_window : int
        Minimum lookback window for statistics.

    Returns
    -------
    tuple of (detected, days_since, window_size)
        Each is a np.ndarray of length len(returns).
    """
    n = len(returns)
    detected = np.zeros(n)
    days_since = np.zeros(n)
    window_size = np.zeros(n)

    last_drift_idx = -min_window * 2  # Initialize far back

    # Fill pre-loop indices with bounded defaults
    for i in range(min(min_window, n)):
        days_since[i] = min(i - last_drift_idx, 252)
        window_size[i] = min((i - last_drift_idx) / 50.0, 5.0)

    for i in range(min_window, n):
        # Rolling mean and std
        window = returns[max(0, i - min_window) : i]
        mu = np.mean(window)
        sigma = np.std(window)
        if sigma < 1e-10:
            days_since[i] = i - last_drift_idx
            continue

        # Compare current return to recent distribution
        z = abs(returns[i] - mu) / sigma

        # Also check if rolling mean has shifted significantly
        if i >= min_window * 2:
            old_window = returns[max(0, i - min_window * 2) : i - min_window]
            old_mu = np.mean(old_window)
            mean_shift = abs(mu - old_mu) / (sigma + 1e-10)
            z = max(z, mean_shift)

        if z > threshold:
            detected[i] = 1.0
            last_drift_idx = i

        days_since[i] = min(i - last_drift_idx, 252)
        # Window size = distance to last drift, normalized
        window_size[i] = min((i - last_drift_idx) / 50.0, 5.0)

    return detected, days_since, window_size
