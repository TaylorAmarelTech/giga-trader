"""
GIGA TRADER - Sector Breadth Features (S&P 500 Sectors)
=========================================================
Market breadth features by S&P 500 sector for validation.

Tracks sector rotation, leadership, and divergence signals using sector ETFs.
This provides additional validation dimensions beyond individual stocks.

Sector ETFs:
  XLK - Technology
  XLF - Financials
  XLV - Healthcare
  XLE - Energy
  XLI - Industrials
  XLY - Consumer Discretionary
  XLP - Consumer Staples
  XLU - Utilities
  XLB - Materials
  XLRE - Real Estate
  XLC - Communication Services
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

from src.phase_01_data_acquisition.alpaca_data_helper import get_alpaca_helper
from src.core.feature_base import FeatureModuleBase


class SectorBreadthFeatures(FeatureModuleBase):
    """
    Market breadth features by S&P 500 sector for validation.

    Tracks sector rotation, leadership, and divergence signals using sector ETFs.
    This provides additional validation dimensions beyond individual stocks.

    Sector ETFs:
      XLK - Technology
      XLF - Financials
      XLV - Healthcare
      XLE - Energy
      XLI - Industrials
      XLY - Consumer Discretionary
      XLP - Consumer Staples
      XLU - Utilities
      XLB - Materials
      XLRE - Real Estate
      XLC - Communication Services

    FEATURE_NAMES is empty because the exact feature set varies dynamically
    based on which sectors have data available.
    """

    # Dynamic feature set — varies by available sector ETFs at runtime
    FEATURE_NAMES = []

    SECTOR_ETFS = {
        "XLK": "Technology",
        "XLF": "Financials",
        "XLV": "Healthcare",
        "XLE": "Energy",
        "XLI": "Industrials",
        "XLY": "Consumer Discretionary",
        "XLP": "Consumer Staples",
        "XLU": "Utilities",
        "XLB": "Materials",
        "XLRE": "Real Estate",
        "XLC": "Communication Services",
    }

    # Defensive vs Cyclical classification
    DEFENSIVE_SECTORS = ["XLV", "XLP", "XLU", "XLRE"]
    CYCLICAL_SECTORS = ["XLK", "XLF", "XLE", "XLI", "XLY", "XLB", "XLC"]

    # Risk-on vs Risk-off classification
    RISK_ON_SECTORS = ["XLK", "XLY", "XLF", "XLE", "XLI"]
    RISK_OFF_SECTORS = ["XLV", "XLP", "XLU", "XLRE"]

    def __init__(self):
        self.data_cache = {}

    def download_sector_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Download sector ETF data via Alpaca."""
        print("\n[SECTORS] Downloading sector ETF data via Alpaca...")

        try:
            helper = get_alpaca_helper()
            tickers = list(self.SECTOR_ETFS.keys())

            data = helper.download_daily_bars(tickers, start_date, end_date)

            if isinstance(data, dict):
                close = data.get("close", pd.DataFrame())
                volume = data.get("volume", pd.DataFrame())
            else:
                print("  [WARN] Unexpected data format from Alpaca")
                return pd.DataFrame()

            if close.empty:
                print("  [WARN] No sector data returned from Alpaca")
                return pd.DataFrame()

            print(f"  Downloaded {len(close.columns)} sector ETFs, {len(close)} days")

            self.data_cache = {
                "close": close,
                "volume": volume,
            }
            return close

        except Exception as e:
            print(f"  [ERROR] Failed to download sector data: {e}")
            return pd.DataFrame()

    def create_sector_features(
        self,
        spy_daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create sector breadth features for validation."""
        if not self.data_cache:
            print("  [WARN] No sector data cached. Call download_sector_data first.")
            return spy_daily

        close = self.data_cache.get("close", pd.DataFrame())
        volume = self.data_cache.get("volume", pd.DataFrame())

        if close.empty:
            return spy_daily

        print("\n[SECTORS] Engineering sector breadth features...")

        features = spy_daily.copy()
        returns = close.pct_change()

        # Get available sectors
        available_sectors = [c for c in self.SECTOR_ETFS.keys() if c in returns.columns]
        defensive_cols = [c for c in self.DEFENSIVE_SECTORS if c in returns.columns]
        cyclical_cols = [c for c in self.CYCLICAL_SECTORS if c in returns.columns]
        risk_on_cols = [c for c in self.RISK_ON_SECTORS if c in returns.columns]
        risk_off_cols = [c for c in self.RISK_OFF_SECTORS if c in returns.columns]

        sector_features = pd.DataFrame(index=returns.index)

        # 1. Overall sector breadth
        sector_features["sector_pct_advancing"] = (returns[available_sectors] > 0).mean(axis=1)
        sector_features["sector_avg_return"] = returns[available_sectors].mean(axis=1)

        # 2. Defensive vs Cyclical rotation
        if len(defensive_cols) > 0 and len(cyclical_cols) > 0:
            defensive_ret = returns[defensive_cols].mean(axis=1)
            cyclical_ret = returns[cyclical_cols].mean(axis=1)
            sector_features["sector_cyclical_vs_defensive"] = cyclical_ret - defensive_ret
            sector_features["sector_cyclical_vs_defensive_5d"] = sector_features["sector_cyclical_vs_defensive"].rolling(5).sum()
            sector_features["sector_cyclical_vs_defensive_20d"] = sector_features["sector_cyclical_vs_defensive"].rolling(20).sum()

        # 3. Risk-on vs Risk-off rotation
        if len(risk_on_cols) > 0 and len(risk_off_cols) > 0:
            risk_on_ret = returns[risk_on_cols].mean(axis=1)
            risk_off_ret = returns[risk_off_cols].mean(axis=1)
            sector_features["sector_risk_on_vs_off"] = risk_on_ret - risk_off_ret
            sector_features["sector_risk_appetite_5d"] = sector_features["sector_risk_on_vs_off"].rolling(5).sum()
            sector_features["sector_risk_appetite_20d"] = sector_features["sector_risk_on_vs_off"].rolling(20).sum()

        # 4. Leading/Lagging sectors
        sector_features["sector_best_return"] = returns[available_sectors].max(axis=1)
        sector_features["sector_worst_return"] = returns[available_sectors].min(axis=1)
        sector_features["sector_dispersion"] = sector_features["sector_best_return"] - sector_features["sector_worst_return"]

        # 5. Momentum for key sectors
        for sector in ["XLK", "XLF", "XLE", "XLV"]:
            if sector in returns.columns:
                sector_features[f"{sector.lower()}_momentum_5d"] = returns[sector].rolling(5).sum()
                sector_features[f"{sector.lower()}_momentum_20d"] = returns[sector].rolling(20).sum()
                sector_features[f"{sector.lower()}_rel_strength"] = (
                    returns[sector].rolling(20).sum() - sector_features["sector_avg_return"].rolling(20).sum()
                )

        # 6. Sector breadth divergence
        sector_features["sector_breadth_strong"] = (sector_features["sector_pct_advancing"] >= 0.7).astype(int)
        sector_features["sector_breadth_weak"] = (sector_features["sector_pct_advancing"] <= 0.3).astype(int)

        # 7. Volume-weighted sector strength (if volume available)
        if not volume.empty:
            vol_cols = [c for c in available_sectors if c in volume.columns]
            if len(vol_cols) > 0:
                vol_weights = volume[vol_cols].div(volume[vol_cols].sum(axis=1), axis=0)
                sector_features["sector_vol_wtd_return"] = (returns[vol_cols] * vol_weights).sum(axis=1)

        # Add date for merge
        sector_features["date"] = pd.to_datetime(sector_features.index.date)

        # Merge with SPY daily
        features = features.merge(
            sector_features.reset_index(drop=True),
            on="date",
            how="left"
        )

        # Compute divergence after merge
        if "day_return" in features.columns:
            spy_direction = (features["day_return"] > 0).astype(int)
            sector_direction = (features["sector_pct_advancing"] > 0.5).astype(int)
            features["sector_breadth_divergence"] = sector_direction - spy_direction

        # Fill NaN
        sector_cols = [c for c in features.columns if "sector_" in c.lower() or c.startswith("xl")]
        features[sector_cols] = features[sector_cols].fillna(0)

        print(f"  Added {len(sector_cols)} sector breadth features")
        return features

    def analyze_sector_rotation(self, features: pd.DataFrame) -> Dict:
        """Analyze current sector rotation signals."""
        if len(features) == 0:
            return {}

        latest = features.iloc[-1].to_dict()

        signal = {
            "date": latest.get("date"),
            "sector_advancing": latest.get("sector_pct_advancing", 0),
            "cyclical_vs_defensive": latest.get("sector_cyclical_vs_defensive_5d", 0),
            "risk_appetite": latest.get("sector_risk_appetite_5d", 0),
            "dispersion": latest.get("sector_dispersion", 0),
        }

        # Rotation interpretation
        if signal["cyclical_vs_defensive"] > 0.02:
            signal["rotation"] = "CYCLICAL_LEADING"
            signal["market_phase"] = "EXPANSION"
        elif signal["cyclical_vs_defensive"] < -0.02:
            signal["rotation"] = "DEFENSIVE_LEADING"
            signal["market_phase"] = "CONTRACTION"
        else:
            signal["rotation"] = "NEUTRAL"
            signal["market_phase"] = "TRANSITION"

        # Risk appetite
        if signal["risk_appetite"] > 0.02:
            signal["risk_sentiment"] = "RISK_ON"
        elif signal["risk_appetite"] < -0.02:
            signal["risk_sentiment"] = "RISK_OFF"
        else:
            signal["risk_sentiment"] = "NEUTRAL"

        return signal
