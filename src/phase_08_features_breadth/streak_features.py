"""
GIGA TRADER - Component Streak Breadth Features
=================================================
Track % of SPY components and market-cap weighted % that have been green
for consecutive days (2, 3, 4, 5+ days in a row).

These breadth features indicate:
  - Momentum strength/exhaustion
  - Breadth divergences (SPY up but fewer stocks participating)
  - Potential reversal signals
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


class ComponentStreakFeatures(FeatureModuleBase):
    """
    Track % of SPY components and market-cap weighted % that have been green
    for consecutive days (2, 3, 4, 5+ days in a row).

    These breadth features indicate:
      - Momentum strength/exhaustion
      - Breadth divergences (SPY up but fewer stocks participating)
      - Potential reversal signals
    """
    FEATURE_NAMES = ["pct_green_3d", "pct_red_3d", "wtd_net_green_3d"]


    # Approximate market cap weights for top SPY components (as of 2024)
    COMPONENT_WEIGHTS = {
        "AAPL": 0.072, "MSFT": 0.071, "AMZN": 0.036, "NVDA": 0.031, "GOOGL": 0.021,
        "META": 0.020, "TSLA": 0.018, "BRK.B": 0.017, "UNH": 0.013, "XOM": 0.013,
        "JNJ": 0.012, "JPM": 0.012, "V": 0.011, "PG": 0.011, "MA": 0.010,
        "HD": 0.010, "CVX": 0.010, "MRK": 0.009, "ABBV": 0.009, "LLY": 0.009,
        "PEP": 0.008, "KO": 0.008, "COST": 0.008, "AVGO": 0.008, "WMT": 0.007,
        "MCD": 0.007, "CSCO": 0.007, "ACN": 0.007, "TMO": 0.007, "ABT": 0.006,
        "DHR": 0.006, "VZ": 0.006, "ADBE": 0.006, "CRM": 0.006, "NKE": 0.005,
        "CMCSA": 0.005, "PFE": 0.005, "INTC": 0.005, "TXN": 0.005, "AMD": 0.005,
        "NEE": 0.005, "PM": 0.005, "RTX": 0.005, "HON": 0.005, "UNP": 0.004,
        "IBM": 0.004, "LOW": 0.004, "SPGI": 0.004, "BA": 0.004, "CAT": 0.004,
    }

    def __init__(self, max_streak: int = 10):
        self.max_streak = max_streak
        self.components = list(self.COMPONENT_WEIGHTS.keys())

    def download_component_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Download daily close prices for SPY components via Alpaca."""
        print("\n[BREADTH] Downloading component data for streak analysis via Alpaca...")

        try:
            helper = get_alpaca_helper()
            close_prices = helper.download_close_prices(self.components, start_date, end_date)

            if close_prices.empty:
                print("  [WARN] No data returned from Alpaca")
                return pd.DataFrame()

            print(f"  Downloaded {len(close_prices.columns)} components, {len(close_prices)} days")
            return close_prices

        except Exception as e:
            print(f"  [ERROR] Failed to download via Alpaca: {e}")
            return pd.DataFrame()

    def compute_streak_features(
        self,
        component_prices: pd.DataFrame,
        spy_daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute breadth features based on component streaks.

        Features:
          - pct_green_2d: % of components green 2+ days in a row
          - pct_green_3d: % of components green 3+ days in a row
          - pct_green_4d: % of components green 4+ days in a row
          - pct_green_5d: % of components green 5+ days in a row
          - wtd_pct_green_Nd: Same but weighted by market cap
          - pct_red_Nd: % of components red N+ days in a row
          - wtd_pct_red_Nd: Weighted version
          - breadth_divergence: SPY return vs breadth change
        """
        if component_prices.empty:
            return spy_daily

        print("\n[BREADTH] Computing component streak features...")

        # Calculate daily returns for each component
        returns = component_prices.pct_change()

        # Compute streaks for each component
        streaks = self._compute_all_streaks(returns)

        # Create features dataframe
        features_list = []

        for date in returns.index:
            if date not in streaks.index:
                continue

            day_streaks = streaks.loc[date]
            # Keep as timestamp for consistent merge with spy_daily
            record = {"date": pd.Timestamp(date.date())}

            # For each streak length
            for n in range(2, self.max_streak + 1):
                # Unweighted percentages
                green_count = (day_streaks >= n).sum()
                red_count = (day_streaks <= -n).sum()
                total = len(day_streaks.dropna())

                record[f"pct_green_{n}d"] = green_count / total if total > 0 else 0
                record[f"pct_red_{n}d"] = red_count / total if total > 0 else 0

                # Weighted percentages
                wtd_green = 0
                wtd_red = 0
                total_weight = 0

                for ticker in day_streaks.index:
                    streak = day_streaks[ticker]
                    weight = self.COMPONENT_WEIGHTS.get(ticker, 0.01)

                    if pd.notna(streak):
                        total_weight += weight
                        if streak >= n:
                            wtd_green += weight
                        elif streak <= -n:
                            wtd_red += weight

                record[f"wtd_pct_green_{n}d"] = wtd_green / total_weight if total_weight > 0 else 0
                record[f"wtd_pct_red_{n}d"] = wtd_red / total_weight if total_weight > 0 else 0

            # Summary features
            record["avg_streak_length"] = day_streaks.mean()
            record["max_green_streak"] = day_streaks.max()
            record["max_red_streak"] = day_streaks.min()
            record["streak_dispersion"] = day_streaks.std()

            # Net green (green - red) for various lengths
            for n in [2, 3, 5]:
                record[f"net_green_{n}d"] = record[f"pct_green_{n}d"] - record[f"pct_red_{n}d"]
                record[f"wtd_net_green_{n}d"] = record[f"wtd_pct_green_{n}d"] - record[f"wtd_pct_red_{n}d"]

            features_list.append(record)

        streak_features = pd.DataFrame(features_list)

        # Merge with spy_daily
        spy_daily = spy_daily.copy()
        result = spy_daily.merge(streak_features, on="date", how="left")

        # Fill NaN with 0 for streak features
        streak_cols = [c for c in result.columns if "green" in c or "red" in c or "streak" in c]
        result[streak_cols] = result[streak_cols].fillna(0)

        # Add lagged streak features (previous day's breadth)
        for col in ["pct_green_3d", "pct_red_3d", "wtd_net_green_3d"]:
            if col in result.columns:
                result[f"{col}_lag1"] = result[col].shift(1)
                result[f"{col}_change"] = result[col] - result[col].shift(1)

        # Breadth divergence: Is SPY going up while breadth is declining?
        if "day_return" in result.columns:
            result["breadth_divergence"] = (
                result["day_return"].rolling(3).sum() *
                result["net_green_3d"].diff().rolling(3).sum()
            )
            # Negative = divergence (SPY up, breadth down or vice versa)

        print(f"  Added {len(streak_cols)} streak-based breadth features")
        return result

    def _compute_all_streaks(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute consecutive day streaks for all components.

        Returns DataFrame where positive = green streak, negative = red streak.
        """
        streaks = pd.DataFrame(index=returns.index, columns=returns.columns)

        for ticker in returns.columns:
            ticker_returns = returns[ticker]
            streak = 0

            for i, (date, ret) in enumerate(ticker_returns.items()):
                if pd.isna(ret):
                    streak = 0
                elif ret > 0:
                    if streak >= 0:
                        streak += 1
                    else:
                        streak = 1
                elif ret < 0:
                    if streak <= 0:
                        streak -= 1
                    else:
                        streak = -1
                else:
                    streak = 0

                streaks.loc[date, ticker] = streak

        return streaks

    def analyze_breadth_signal(self, features: pd.DataFrame) -> Dict:
        """
        Analyze current breadth conditions.

        Returns signal interpretation.
        """
        if len(features) == 0:
            return {}

        latest = features.iloc[-1]

        signal = {
            "date": latest.get("date"),
            "pct_green_3d": latest.get("pct_green_3d", 0),
            "pct_red_3d": latest.get("pct_red_3d", 0),
            "wtd_net_green_3d": latest.get("wtd_net_green_3d", 0),
            "breadth_divergence": latest.get("breadth_divergence", 0),
        }

        # Interpretation
        if signal["wtd_net_green_3d"] > 0.3:
            signal["interpretation"] = "STRONG_BULLISH_BREADTH"
        elif signal["wtd_net_green_3d"] > 0.1:
            signal["interpretation"] = "BULLISH_BREADTH"
        elif signal["wtd_net_green_3d"] < -0.3:
            signal["interpretation"] = "STRONG_BEARISH_BREADTH"
        elif signal["wtd_net_green_3d"] < -0.1:
            signal["interpretation"] = "BEARISH_BREADTH"
        else:
            signal["interpretation"] = "NEUTRAL_BREADTH"

        if signal["breadth_divergence"] < -0.01:
            signal["warning"] = "POTENTIAL_DIVERGENCE"

        return signal
