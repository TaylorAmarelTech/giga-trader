"""
GIGA TRADER - Cross-Asset Features
====================================
Add features from correlated assets: TLT, QQQ, GLD, VIX, etc.

These provide:
  - Risk-on/risk-off signals
  - Interest rate sensitivity
  - Sector rotation hints
  - Volatility regime context
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from src.core.feature_base import FeatureModuleBase
from src.phase_01_data_acquisition.alpaca_data_helper import get_alpaca_helper


class CrossAssetFeatures(FeatureModuleBase):
    """
    Add features from correlated assets: TLT, QQQ, GLD, VIX, etc.

    These provide:
      - Risk-on/risk-off signals
      - Interest rate sensitivity
      - Sector rotation hints
      - Volatility regime context
    """

    FEATURE_NAMES = []  # Dynamic: features created per asset at runtime

    ASSETS = {
        "TLT": "Treasury bonds (20+ year)",
        "QQQ": "NASDAQ 100 (tech-heavy)",
        "GLD": "Gold ETF",
        "IWM": "Russell 2000 (small caps)",
        "EEM": "Emerging markets",
        "VXX": "VIX short-term futures",
        "UUP": "US Dollar index ETF",  # Changed from DXY (not a stock)
        "HYG": "High yield bonds",
    }

    def __init__(self, assets: List[str] = None):
        self.assets = assets or list(self.ASSETS.keys())

    def download_cross_assets(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Download cross-asset data via Alpaca."""
        print("\n[CROSS-ASSETS] Downloading correlated asset data via Alpaca...")

        try:
            helper = get_alpaca_helper()
            close_prices = helper.download_close_prices(self.assets, start_date, end_date)

            if close_prices.empty:
                print("  [WARN] No data returned from Alpaca")
                return pd.DataFrame()

            print(f"  Downloaded {len(close_prices.columns)} assets, {len(close_prices)} days")
            return close_prices

        except Exception as e:
            print(f"  [ERROR] Failed to download via Alpaca: {e}")
            return pd.DataFrame()

    def create_cross_asset_features(
        self,
        cross_data: pd.DataFrame,
        spy_daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create features from cross-asset data.

        Features:
          - Daily returns
          - Relative strength vs SPY
          - Correlation rolling
          - Regime indicators
        """
        if cross_data.empty:
            return spy_daily

        print("\n[CROSS-ASSETS] Engineering features...")

        features = spy_daily.copy()
        returns = cross_data.pct_change()

        for asset in returns.columns:
            asset_ret = returns[asset]

            # Match dates
            asset_features = pd.DataFrame(index=asset_ret.index)

            # 1. Daily return
            asset_features[f"{asset}_return"] = asset_ret

            # 2. 5-day return
            asset_features[f"{asset}_return_5d"] = asset_ret.rolling(5).sum()

            # 3. 20-day volatility
            asset_features[f"{asset}_vol_20d"] = asset_ret.rolling(20).std()

            # 4. 20-day momentum
            asset_features[f"{asset}_mom_20d"] = cross_data[asset].pct_change(20)

            # 5. RSI-like (simplified)
            delta = asset_ret.copy()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            asset_features[f"{asset}_rsi"] = 100 - (100 / (1 + gain / (loss + 1e-10)))

            # Convert index to date (as Timestamp for consistent merge)
            asset_features["date"] = pd.to_datetime(asset_features.index.date)

            # Merge with spy_daily
            features = features.merge(
                asset_features.reset_index(drop=True),
                on="date",
                how="left"
            )

        # Fill NaN with 0 for cross-asset features
        cross_cols = [c for c in features.columns if any(a in c for a in self.assets)]
        features[cross_cols] = features[cross_cols].fillna(0)

        print(f"  Added {len(cross_cols)} cross-asset features")
        return features
