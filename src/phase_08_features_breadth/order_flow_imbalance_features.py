"""
Order Flow Imbalance Features -- Bulk Volume Classification (BVC) proxy.

BVC estimates buy/sell volume from OHLCV data without tick-level data.
The key insight: close price position within the high-low range indicates
buying vs selling pressure.

  buy_volume ~ volume * (close - low) / (high - low)
  sell_volume ~ volume * (high - close) / (high - low)

Features (8, prefix ofi_):
  ofi_buy_volume_proxy     -- BVC estimated buy volume fraction
  ofi_sell_volume_proxy    -- BVC estimated sell volume fraction
  ofi_imbalance            -- (buy - sell) / total, in [-1, 1]
  ofi_cumulative_5d        -- 5-day cumulative imbalance
  ofi_cumulative_20d       -- 20-day cumulative imbalance
  ofi_normalized_20d       -- 20-day z-score of daily imbalance
  ofi_imbalance_z          -- 60-day z-score of ofi_imbalance
  ofi_regime               -- 1.0 ACCUMULATION (z>1.0), -1.0 DISTRIBUTION (z<-1.0), 0.0 NEUTRAL
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from src.core.feature_base import FeatureModuleBase

logger = logging.getLogger(__name__)


class OrderFlowImbalanceFeatures(FeatureModuleBase):
    """Compute order flow imbalance features from OHLCV data."""
    FEATURE_NAMES = ["ofi_buy_volume_proxy", "ofi_sell_volume_proxy", "ofi_imbalance", "ofi_cumulative_5d", "ofi_cumulative_20d", "ofi_normalized_20d", "ofi_imbalance_z", "ofi_regime"]


    REQUIRED_COLS = {"open", "high", "low", "close", "volume"}

    def create_order_flow_imbalance_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """Add 8 ofi_ features to df_daily."""
        df = df_daily.copy()
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            logger.warning("OrderFlowImbalanceFeatures: missing columns %s, skipping", missing)
            return df

        high = df["high"].astype(float).values
        low = df["low"].astype(float).values
        close = df["close"].astype(float).values
        volume = df["volume"].astype(float).values

        # BVC: Bulk Volume Classification
        hl_range = high - low
        # Avoid division by zero for flat bars
        hl_range = np.where(hl_range < 1e-10, 1e-10, hl_range)

        buy_frac = (close - low) / hl_range
        sell_frac = (high - close) / hl_range

        buy_vol = volume * buy_frac
        sell_vol = volume * sell_frac
        total_vol = buy_vol + sell_vol
        total_vol = np.where(total_vol < 1e-10, 1e-10, total_vol)

        df["ofi_buy_volume_proxy"] = buy_frac
        df["ofi_sell_volume_proxy"] = sell_frac

        # Imbalance: (buy - sell) / total
        imbalance = (buy_vol - sell_vol) / total_vol
        df["ofi_imbalance"] = imbalance

        # Cumulative imbalance
        imb_series = pd.Series(imbalance, index=df.index)
        df["ofi_cumulative_5d"] = imb_series.rolling(5, min_periods=1).sum()
        df["ofi_cumulative_20d"] = imb_series.rolling(20, min_periods=5).sum()

        # Normalized 20d z-score
        imb_mean_20 = imb_series.rolling(20, min_periods=5).mean()
        imb_std_20 = imb_series.rolling(20, min_periods=5).std()
        df["ofi_normalized_20d"] = ((imb_series - imb_mean_20) / (imb_std_20 + 1e-10)).clip(-4, 4)

        # 60-day z-score
        imb_mean_60 = imb_series.rolling(60, min_periods=10).mean()
        imb_std_60 = imb_series.rolling(60, min_periods=10).std()
        df["ofi_imbalance_z"] = ((imb_series - imb_mean_60) / (imb_std_60 + 1e-10)).clip(-4, 4)

        # Regime
        z = df["ofi_imbalance_z"]
        df["ofi_regime"] = np.where(z > 1.0, 1.0, np.where(z < -1.0, -1.0, 0.0))

        # Cleanup
        for col in self._all_feature_names():
            df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        n_features = sum(1 for c in df.columns if c.startswith("ofi_"))
        logger.info("OrderFlowImbalanceFeatures: added %d features", n_features)
        return df

    def analyze_current_order_flow(self, df_daily: pd.DataFrame) -> Optional[Dict]:
        """Return current order flow regime."""
        if "ofi_imbalance_z" not in df_daily.columns or len(df_daily) < 2:
            return None
        last = df_daily.iloc[-1]
        z = float(last.get("ofi_imbalance_z", 0.0))
        if z > 1.0:
            regime = "ACCUMULATION"
        elif z < -1.0:
            regime = "DISTRIBUTION"
        else:
            regime = "NEUTRAL"
        return {
            "flow_regime": regime,
            "imbalance_z": round(z, 3),
            "imbalance": round(float(last.get("ofi_imbalance", 0.0)), 4),
            "cumulative_20d": round(float(last.get("ofi_cumulative_20d", 0.0)), 4),
        }

    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "ofi_buy_volume_proxy",
            "ofi_sell_volume_proxy",
            "ofi_imbalance",
            "ofi_cumulative_5d",
            "ofi_cumulative_20d",
            "ofi_normalized_20d",
            "ofi_imbalance_z",
            "ofi_regime",
        ]
