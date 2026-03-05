"""
Range-Based Volatility Features — OHLC-efficient vol estimators.

Garman-Klass (1980): Uses OHLC, up to 7.4x more efficient than close-to-close.
Yang-Zhang (2000): Combines overnight + open-close + Rogers-Satchell, up to 14x efficient.
Rogers-Satchell (1991): Zero-drift estimator using OHLC.

Features (8, prefix rvol_):
  rvol_gk_5d           — Garman-Klass 5-day rolling vol
  rvol_gk_20d          — Garman-Klass 20-day rolling vol
  rvol_yz_5d           — Yang-Zhang 5-day rolling vol
  rvol_yz_20d          — Yang-Zhang 20-day rolling vol
  rvol_rs_5d           — Rogers-Satchell 5-day rolling vol
  rvol_rs_20d          — Rogers-Satchell 20-day rolling vol
  rvol_ratio_gk_cc     — GK vol / close-to-close vol (efficiency measure)
  rvol_vol_surprise    — Today's range / 20-day expected range
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from src.core.feature_base import FeatureModuleBase

logger = logging.getLogger(__name__)


class RangeVolFeatures(FeatureModuleBase):
    """Compute range-based volatility estimator features from daily OHLCV."""
    FEATURE_NAMES = ["rvol_gk_5d", "rvol_gk_20d", "rvol_yz_5d", "rvol_yz_20d", "rvol_rs_5d", "rvol_rs_20d", "rvol_ratio_gk_cc", "rvol_vol_surprise"]


    REQUIRED_COLS = {"close", "open", "high", "low"}

    def create_range_vol_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Add range-based volatility features to df_daily.

        Parameters
        ----------
        df_daily : pd.DataFrame
            Must have 'close' column. 'open', 'high', 'low' optional (features default to 0.0).

        Returns
        -------
        pd.DataFrame
            Original df_daily with 8 new rvol_ columns added.
        """
        df = df_daily.copy()

        if "close" not in df.columns:
            logger.warning("RangeVolFeatures: 'close' column missing, skipping")
            return df

        has_ohlc = all(c in df.columns for c in ["open", "high", "low"])

        if not has_ohlc:
            logger.info("RangeVolFeatures: missing OHLC columns, defaulting to 0.0")
            for col in self._all_feature_names():
                df[col] = 0.0
            return df

        o = df["open"]
        h = df["high"]
        l = df["low"]  # noqa: E741
        c = df["close"]

        # Prevent log(0) — use small floor
        eps = 1e-10

        # --- Rogers-Satchell variance (per-bar) ---
        # RS = log(H/C)*log(H/O) + log(L/C)*log(L/O)
        rs_var = (
            np.log(h / (c + eps) + eps) * np.log(h / (o + eps) + eps)
            + np.log(l / (c + eps) + eps) * np.log(l / (o + eps) + eps)
        )
        rs_var = rs_var.clip(lower=0)  # RS variance should be non-negative

        # --- Garman-Klass variance (per-bar) ---
        # GK = 0.5*log(H/L)^2 - (2*ln2-1)*log(C/O)^2
        gk_var = (
            0.5 * np.log(h / (l + eps) + eps) ** 2
            - (2 * np.log(2) - 1) * np.log(c / (o + eps) + eps) ** 2
        )
        gk_var = gk_var.clip(lower=0)

        # --- Yang-Zhang variance (per-bar) ---
        # YZ = overnight_var + open_close_var + k * RS_var
        # overnight = log(O_t / C_{t-1})
        # open_close = log(C_t / O_t)
        c_prev = c.shift(1)
        overnight = np.log(o / (c_prev + eps) + eps)
        oc = np.log(c / (o + eps) + eps)

        k = 0.34 / (1.34 + 2 / 5)  # Yang-Zhang optimal constant for n=5

        # Rolling volatilities (annualized as sqrt(252 * mean_variance))
        for window in [5, 20]:
            min_p = max(3, window // 2)

            gk_rolling = gk_var.rolling(window, min_periods=min_p).mean()
            df[f"rvol_gk_{window}d"] = np.sqrt(252 * gk_rolling.clip(lower=0))

            rs_rolling = rs_var.rolling(window, min_periods=min_p).mean()
            df[f"rvol_rs_{window}d"] = np.sqrt(252 * rs_rolling.clip(lower=0))

            # Yang-Zhang: need rolling components
            on_var = overnight.rolling(window, min_periods=min_p).var()
            oc_var = oc.rolling(window, min_periods=min_p).var()
            yz_var = on_var + k * rs_rolling + (1 - k) * oc_var
            df[f"rvol_yz_{window}d"] = np.sqrt(252 * yz_var.clip(lower=0))

        # --- Ratio: GK / close-to-close vol ---
        cc_var = df["close"].pct_change().rolling(20, min_periods=5).var()
        cc_vol = np.sqrt(252 * cc_var.clip(lower=0))
        df["rvol_ratio_gk_cc"] = (df["rvol_gk_20d"] / (cc_vol + 1e-10)).clip(0, 5)

        # --- Volatility surprise: today's range / 20d expected range ---
        daily_range = (h - l) / (c + eps)
        expected_range = daily_range.rolling(20, min_periods=5).mean()
        df["rvol_vol_surprise"] = (daily_range / (expected_range + 1e-10)).clip(0, 5)

        # Cleanup
        for col in self._all_feature_names():
            if col in df.columns:
                df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        n_features = sum(1 for c in df.columns if c.startswith("rvol_"))
        logger.info(f"RangeVolFeatures: added {n_features} features")
        return df

    def analyze_current_volatility(self, df_daily: pd.DataFrame) -> Optional[Dict]:
        """Analyze current volatility regime for dashboard display."""
        if "rvol_yz_20d" not in df_daily.columns or len(df_daily) < 2:
            return None

        last = df_daily.iloc[-1]
        yz_vol = last.get("rvol_yz_20d", 0.0)
        surprise = last.get("rvol_vol_surprise", 1.0)

        if yz_vol > 0.25:
            regime = "HIGH_VOL"
        elif yz_vol < 0.10:
            regime = "LOW_VOL"
        else:
            regime = "NORMAL_VOL"

        return {
            "vol_regime": regime,
            "yz_vol_20d": round(float(yz_vol), 4),
            "vol_surprise": round(float(surprise), 3),
        }

    @staticmethod
    def _all_feature_names():
        return [
            "rvol_gk_5d", "rvol_gk_20d",
            "rvol_yz_5d", "rvol_yz_20d",
            "rvol_rs_5d", "rvol_rs_20d",
            "rvol_ratio_gk_cc", "rvol_vol_surprise",
        ]
