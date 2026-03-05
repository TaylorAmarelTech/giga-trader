"""
Hurst Exponent Features — persistence/anti-persistence detection via R/S analysis.

The Hurst exponent H measures long-term memory in a time series:
  H < 0.4  → mean-reverting (anti-persistent)
  0.4 <= H <= 0.6 → random walk (no memory)
  H > 0.6  → trending (persistent)

Features (4, prefix hurst_):
  hurst_50d     — Rolling R/S Hurst exponent over 50-day window
  hurst_100d    — Rolling R/S Hurst exponent over 100-day window
  hurst_z       — Z-score of hurst_50d over rolling 100-day window, clipped [-4, 4]
  hurst_regime  — Regime: 0=mean-reverting (H<0.4), 1=random walk, 2=trending (H>0.6)
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from src.core.feature_base import FeatureModuleBase

logger = logging.getLogger(__name__)


def _rs_hurst(series: np.ndarray) -> float:
    """Compute Hurst exponent via R/S analysis (pure numpy)."""
    n = len(series)
    if n < 20:
        return 0.5  # default to random walk

    # Use multiple sub-series lengths
    sizes = []
    rs_values = []

    for size in [int(n / 4), int(n / 3), int(n / 2)]:
        if size < 10:
            continue
        n_subseries = n // size
        rs_list = []
        for i in range(n_subseries):
            subseries = series[i * size : (i + 1) * size]
            mean_val = np.mean(subseries)
            deviations = np.cumsum(subseries - mean_val)
            R = np.max(deviations) - np.min(deviations)
            S = np.std(subseries, ddof=1)
            if S > 0:
                rs_list.append(R / S)
        if rs_list:
            sizes.append(size)
            rs_values.append(np.mean(rs_list))

    if len(sizes) < 2:
        return 0.5

    # Linear regression of log(R/S) vs log(n)
    log_sizes = np.log(sizes)
    log_rs = np.log(rs_values)
    slope = np.polyfit(log_sizes, log_rs, 1)[0]
    return float(np.clip(slope, 0.0, 1.0))


class HurstFeatures(FeatureModuleBase):
    """Compute Hurst exponent features from daily OHLCV data."""
    FEATURE_NAMES = ["hurst_50d", "hurst_100d", "hurst_z", "hurst_regime"]


    REQUIRED_COLS = {"close"}

    def create_hurst_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Add Hurst exponent features to df_daily.

        Parameters
        ----------
        df_daily : pd.DataFrame
            Must have 'close' column.

        Returns
        -------
        pd.DataFrame
            Original df_daily with 4 new hurst_ columns added.
        """
        df = df_daily.copy()

        if "close" not in df.columns:
            logger.warning("HurstFeatures: 'close' column missing, skipping")
            return df

        # Log returns for R/S computation
        log_returns = np.log(df["close"] / df["close"].shift(1)).values

        # --- Rolling Hurst exponents ---
        hurst_50 = np.full(len(df), 0.5)
        hurst_100 = np.full(len(df), 0.5)

        for i in range(len(df)):
            # 50-day window
            if i >= 49:
                window = log_returns[i - 49 : i + 1]
                valid = window[~np.isnan(window)]
                if len(valid) >= 20:
                    hurst_50[i] = _rs_hurst(valid)

            # 100-day window
            if i >= 99:
                window = log_returns[i - 99 : i + 1]
                valid = window[~np.isnan(window)]
                if len(valid) >= 20:
                    hurst_100[i] = _rs_hurst(valid)

        df["hurst_50d"] = hurst_50
        df["hurst_100d"] = hurst_100

        # Z-score of hurst_50d over rolling 100-day window
        h50 = pd.Series(hurst_50, index=df.index)
        rolling_mean = h50.rolling(100, min_periods=20).mean()
        rolling_std = h50.rolling(100, min_periods=20).std()
        z = ((h50 - rolling_mean) / (rolling_std + 1e-10)).clip(-4, 4)
        df["hurst_z"] = z

        # Regime classification
        regime = np.where(
            hurst_50 < 0.4,
            0.0,  # mean-reverting
            np.where(hurst_50 > 0.6, 2.0, 1.0),  # trending vs random walk
        )
        df["hurst_regime"] = regime

        # Cleanup: fill NaN with appropriate defaults
        df["hurst_50d"] = df["hurst_50d"].fillna(0.5).replace([np.inf, -np.inf], 0.5)
        df["hurst_100d"] = df["hurst_100d"].fillna(0.5).replace([np.inf, -np.inf], 0.5)
        df["hurst_z"] = df["hurst_z"].fillna(0.0).replace([np.inf, -np.inf], 0.0)
        df["hurst_regime"] = df["hurst_regime"].fillna(1.0).replace([np.inf, -np.inf], 1.0)

        n_features = sum(1 for c in df.columns if c.startswith("hurst_"))
        logger.info(f"HurstFeatures: added {n_features} features")
        return df

    def analyze_current_hurst(self, df_daily: pd.DataFrame) -> Optional[Dict]:
        """Analyze current Hurst regime for dashboard display."""
        if "hurst_50d" not in df_daily.columns or len(df_daily) < 2:
            return None

        last = df_daily.iloc[-1]
        h = last.get("hurst_50d", 0.5)
        regime_val = last.get("hurst_regime", 1.0)

        if regime_val == 0.0:
            regime = "MEAN_REVERTING"
        elif regime_val == 2.0:
            regime = "TRENDING"
        else:
            regime = "RANDOM_WALK"

        return {
            "hurst_regime": regime,
            "hurst_50d": round(float(h), 4),
            "hurst_100d": round(float(last.get("hurst_100d", 0.5)), 4),
            "hurst_z": round(float(last.get("hurst_z", 0.0)), 3),
        }

    @staticmethod
    def _all_feature_names():
        return [
            "hurst_50d",
            "hurst_100d",
            "hurst_z",
            "hurst_regime",
        ]
