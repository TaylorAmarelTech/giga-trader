"""
GIGA TRADER - MAG Market Breadth Features (MAG3, MAG5, MAG6, MAG7, MAG10, MAG15)
==================================================================================
Market breadth features for various MAG (Magnificent) groupings.

MAG3:  AAPL, MSFT, NVDA (Big 3 tech leaders, ~20% of S&P)
MAG5:  AAPL, MSFT, NVDA, GOOGL, AMZN (Top 5 by market cap)
MAG6:  MAG5 + META (Core tech mega-caps)
MAG7:  MAG6 + TSLA (Magnificent 7)
MAG10: MAG7 + BRK.B, UNH, XOM (Top 10 S&P weights)
MAG15: MAG10 + JNJ, V, JPM, PG, MA (Top 15 S&P weights)

These mega-caps drive ~35-40% of S&P 500 movement, so tracking them
specifically provides valuable market leadership signals.
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


class Mag7BreadthFeatures:
    """
    Market breadth features for various MAG (Magnificent) groupings.

    MAG3:  AAPL, MSFT, NVDA (Big 3 tech leaders, ~20% of S&P)
    MAG5:  AAPL, MSFT, NVDA, GOOGL, AMZN (Top 5 by market cap)
    MAG6:  MAG5 + META (Core tech mega-caps)
    MAG7:  MAG6 + TSLA (Magnificent 7)
    MAG10: MAG7 + BRK.B, UNH, XOM (Top 10 S&P weights)
    MAG15: MAG10 + JNJ, V, JPM, PG, MA (Top 15 S&P weights)

    These mega-caps drive ~35-40% of S&P 500 movement, so tracking them
    specifically provides valuable market leadership signals.

    Features for each group:
      - % advancing (breadth)
      - % at 52-week high/low
      - Average and weighted momentum
      - Breadth divergence from SPY
      - Relative strength vs SPY
      - Sector rotation signals
    """

    # MAG groupings (ordered by market cap, largest first)
    MAG3 = ["AAPL", "MSFT", "NVDA"]
    MAG5 = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
    MAG6 = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"]
    MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
    MAG10 = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK.B", "UNH", "XOM"]
    MAG15 = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK.B", "UNH", "XOM",
             "JNJ", "V", "JPM", "PG", "MA"]

    # All unique tickers to download
    ALL_MAG_TICKERS = list(set(MAG15))

    # Approximate market cap weights (billions, for weighting) - Updated 2026
    MAG_WEIGHTS = {
        "AAPL": 3200, "MSFT": 3000, "NVDA": 2000, "GOOGL": 1900, "AMZN": 1800,
        "META": 1200, "TSLA": 900, "BRK.B": 800, "UNH": 550, "XOM": 500,
        "JNJ": 400, "V": 480, "JPM": 550, "PG": 380, "MA": 420,
    }

    # Tech vs Non-Tech classification for rotation analysis
    TECH_MAGS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]
    NON_TECH_MAGS = ["BRK.B", "UNH", "XOM", "JNJ", "V", "JPM", "PG", "MA"]

    def __init__(self):
        self.data_cache = {}

    def download_mag_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Download MAG15 price data via Alpaca (covers all MAG groupings)."""
        print("\n[MAG] Downloading MAG3/5/6/7/10/15 data via Alpaca...")

        try:
            helper = get_alpaca_helper()

            # Download all MAG15 tickers (covers all smaller groups)
            tickers = self.ALL_MAG_TICKERS.copy()

            data = helper.download_daily_bars(tickers, start_date, end_date)

            if isinstance(data, dict):
                close = data.get("close", pd.DataFrame())
                high = data.get("high", pd.DataFrame())
                low = data.get("low", pd.DataFrame())
            else:
                print("  [WARN] Unexpected data format from Alpaca")
                return pd.DataFrame()

            if close.empty:
                print("  [WARN] No data returned from Alpaca")
                return pd.DataFrame()

            print(f"  Downloaded {len(close.columns)} MAG stocks, {len(close)} days")

            self.data_cache = {
                "close": close,
                "high": high,
                "low": low,
            }
            return close

        except Exception as e:
            print(f"  [ERROR] Failed to download MAG data via Alpaca: {e}")
            return pd.DataFrame()

    def create_mag_features(
        self,
        spy_daily: pd.DataFrame,
        lookback_52w: int = 252,
    ) -> pd.DataFrame:
        """
        Create MAG7/MAG10 breadth features.

        Features created:
          - mag7_pct_advancing: % of MAG7 green today
          - mag10_pct_advancing: % of MAG10 green today
          - mag7_avg_return: Average return across MAG7
          - mag10_avg_return: Average return across MAG10
          - mag7_wtd_return: Market-cap weighted return
          - mag10_wtd_return: Market-cap weighted return
          - mag7_at_52w_high: % near 52-week high
          - mag7_at_52w_low: % near 52-week low
          - mag7_momentum_5d: 5-day momentum
          - mag7_momentum_20d: 20-day momentum
          - mag7_breadth_divergence: Divergence from SPY
          - mag_tech_vs_nontch: Tech MAG vs non-tech MAG performance
        """
        if not self.data_cache:
            print("  [WARN] No MAG data cached. Call download_mag_data first.")
            return spy_daily

        close = self.data_cache.get("close", pd.DataFrame())
        high = self.data_cache.get("high", pd.DataFrame())
        low = self.data_cache.get("low", pd.DataFrame())

        if close.empty:
            return spy_daily

        print("\n[MAG] Engineering MAG3/5/6/7/10/15 breadth features...")

        features = spy_daily.copy()
        returns = close.pct_change()

        # Define all MAG groups with their tickers
        mag_groups = {
            "mag3": self.MAG3,
            "mag5": self.MAG5,
            "mag6": self.MAG6,
            "mag7": self.MAG7,
            "mag10": self.MAG10,
            "mag15": self.MAG15,
        }

        # Get available columns for each group
        mag_cols_map = {name: [c for c in tickers if c in returns.columns]
                        for name, tickers in mag_groups.items()}

        # Compute normalized weights for each group
        mag_norm_weights = {}
        for name, cols in mag_cols_map.items():
            weights = {k: v for k, v in self.MAG_WEIGHTS.items() if k in cols}
            total = sum(weights.values()) if weights else 1
            mag_norm_weights[name] = {k: v / total for k, v in weights.items()}

        mag_features = pd.DataFrame(index=returns.index)

        # 52-week high/low calculations (shared)
        rolling_high = high.rolling(lookback_52w).max()
        rolling_low = low.rolling(lookback_52w).min()
        pct_from_high = (close - rolling_high) / rolling_high
        pct_from_low = (close - rolling_low) / rolling_low
        near_high = pct_from_high > -0.05  # Within 5% of 52w high
        near_low = pct_from_low < 0.05     # Within 5% of 52w low

        # Generate features for each MAG group
        for name, cols in mag_cols_map.items():
            if len(cols) == 0:
                continue

            prefix = name  # e.g., "mag3", "mag5", etc.

            # 1. % Advancing (breadth)
            mag_features[f"{prefix}_pct_advancing"] = (returns[cols] > 0).mean(axis=1)

            # 2. Average return (equal weight)
            mag_features[f"{prefix}_avg_return"] = returns[cols].mean(axis=1)

            # 3. Market-cap weighted return
            norm_w = mag_norm_weights[name]
            wtd_ret = sum(returns.get(t, 0) * norm_w.get(t, 0) for t in cols)
            mag_features[f"{prefix}_wtd_return"] = wtd_ret

            # 4. 52-week high/low proximity
            mag_features[f"{prefix}_pct_near_52w_high"] = near_high[cols].mean(axis=1)
            mag_features[f"{prefix}_pct_near_52w_low"] = near_low[cols].mean(axis=1)

            # 5. Momentum (5-day and 20-day)
            mag_features[f"{prefix}_momentum_5d"] = returns[cols].rolling(5).sum().mean(axis=1)
            mag_features[f"{prefix}_momentum_20d"] = returns[cols].rolling(20).sum().mean(axis=1)

            # 6. Volatility (20-day)
            mag_features[f"{prefix}_volatility_20d"] = returns[cols].rolling(20).std().mean(axis=1)

            # 7. Relative strength (cumulative 20d return)
            mag_features[f"{prefix}_rel_strength_20d"] = returns[cols].rolling(20).sum().mean(axis=1)

            # 8. Streak features (consecutive advancing/declining days)
            advancing = (returns[cols] > 0).all(axis=1)
            declining = (returns[cols] < 0).all(axis=1)
            mag_features[f"{prefix}_all_advancing"] = advancing.astype(int)
            mag_features[f"{prefix}_all_declining"] = declining.astype(int)

        # Tech vs Non-Tech MAG rotation analysis
        tech_cols = [c for c in self.TECH_MAGS if c in returns.columns]
        non_tech_cols = [c for c in self.NON_TECH_MAGS if c in returns.columns]

        if len(tech_cols) > 0 and len(non_tech_cols) > 0:
            tech_return = returns[tech_cols].mean(axis=1)
            non_tech_return = returns[non_tech_cols].mean(axis=1)
            mag_features["mag_tech_vs_nontech"] = tech_return - non_tech_return
            mag_features["mag_tech_vs_nontech_5d"] = mag_features["mag_tech_vs_nontech"].rolling(5).sum()
            mag_features["mag_tech_vs_nontech_20d"] = mag_features["mag_tech_vs_nontech"].rolling(20).sum()

            # Tech leadership indicator
            mag_features["mag_tech_leading"] = (mag_features["mag_tech_vs_nontech_5d"] > 0).astype(int)
        else:
            mag_features["mag_tech_vs_nontech"] = 0
            mag_features["mag_tech_vs_nontech_5d"] = 0
            mag_features["mag_tech_vs_nontech_20d"] = 0
            mag_features["mag_tech_leading"] = 0

        # Cross-MAG breadth comparison (larger group vs smaller group)
        if "mag7_pct_advancing" in mag_features.columns and "mag3_pct_advancing" in mag_features.columns:
            # MAG3 vs MAG7 divergence (big 3 leading/lagging the MAG7)
            mag_features["mag3_vs_mag7_breadth"] = (
                mag_features["mag3_pct_advancing"] - mag_features["mag7_pct_advancing"]
            )

        if "mag15_pct_advancing" in mag_features.columns and "mag7_pct_advancing" in mag_features.columns:
            # MAG7 vs MAG15 divergence (core tech vs broader mega-caps)
            mag_features["mag7_vs_mag15_breadth"] = (
                mag_features["mag7_pct_advancing"] - mag_features["mag15_pct_advancing"]
            )

        # Concentration risk indicator (top 3 driving all returns)
        if "mag3_wtd_return" in mag_features.columns and "mag7_wtd_return" in mag_features.columns:
            mag3_wt = mag_features["mag3_wtd_return"]
            mag7_wt = mag_features["mag7_wtd_return"]
            # If MAG3 return > 80% of MAG7 return, concentration is high
            mag_features["mag_concentration_risk"] = (
                (mag3_wt.abs() > 0.8 * mag7_wt.abs()) & (mag7_wt.abs() > 0.001)
            ).astype(int)

        # SPY correlation (rolling 20-day) for key groups
        if "day_return" in features.columns:
            spy_ret = features.set_index("date")["day_return"] if "date" in features.columns else features["day_return"]
            try:
                for name in ["mag3", "mag7", "mag10", "mag15"]:
                    if f"{name}_avg_return" in mag_features.columns:
                        mag_features[f"{name}_spy_corr_20d"] = (
                            mag_features[f"{name}_avg_return"].rolling(20).corr(spy_ret)
                        )
            except (KeyError, ValueError):
                pass  # Correlation calc may fail on misaligned indices

        # Breadth divergence (MAG advancing but SPY flat/down)
        mag_features["date"] = pd.to_datetime(mag_features.index.date)

        # Merge with SPY daily
        features = features.merge(
            mag_features.reset_index(drop=True),
            on="date",
            how="left"
        )

        # Compute divergence after merge
        if "day_return" in features.columns:
            spy_direction = (features["day_return"] > 0).astype(int)
            for name in ["mag3", "mag7", "mag10", "mag15"]:
                col = f"{name}_pct_advancing"
                if col in features.columns:
                    mag_direction = (features[col] > 0.5).astype(int)
                    features[f"{name}_breadth_divergence"] = mag_direction - spy_direction

        # Fill NaN
        all_mag_cols = [c for c in features.columns if "mag" in c.lower()]
        features[all_mag_cols] = features[all_mag_cols].fillna(0)

        print(f"  Added {len(all_mag_cols)} MAG3/5/6/7/10/15 breadth features")
        return features

    def analyze_mag_leadership(self, features: pd.DataFrame) -> Dict:
        """
        Analyze current MAG7/MAG10 leadership conditions.

        Returns signal interpretation.
        """
        if len(features) == 0:
            return {}

        latest = features.iloc[-1]

        signal = {
            "date": latest.get("date"),
            # All MAG group breadths
            "mag3_advancing": latest.get("mag3_pct_advancing", 0),
            "mag5_advancing": latest.get("mag5_pct_advancing", 0),
            "mag6_advancing": latest.get("mag6_pct_advancing", 0),
            "mag7_advancing": latest.get("mag7_pct_advancing", 0),
            "mag10_advancing": latest.get("mag10_pct_advancing", 0),
            "mag15_advancing": latest.get("mag15_pct_advancing", 0),
            # Momentum
            "mag3_momentum_5d": latest.get("mag3_momentum_5d", 0),
            "mag7_momentum_5d": latest.get("mag7_momentum_5d", 0),
            "mag15_momentum_5d": latest.get("mag15_momentum_5d", 0),
            # 52-week position
            "mag7_near_52w_high": latest.get("mag7_pct_near_52w_high", 0),
            # Rotation
            "tech_vs_nontech": latest.get("mag_tech_vs_nontech_20d", 0),
            "tech_leading": latest.get("mag_tech_leading", 0),
            # Concentration
            "concentration_risk": latest.get("mag_concentration_risk", 0),
            # Cross-group divergence
            "mag3_vs_mag7_breadth": latest.get("mag3_vs_mag7_breadth", 0),
            "mag7_vs_mag15_breadth": latest.get("mag7_vs_mag15_breadth", 0),
        }

        # Interpretation based on MAG7 (core indicator)
        if signal["mag7_advancing"] >= 0.7 and signal["mag7_momentum_5d"] > 0.02:
            signal["interpretation"] = "STRONG_MAG_LEADERSHIP"
            signal["bias"] = "BULLISH"
        elif signal["mag7_advancing"] >= 0.5 and signal["mag7_momentum_5d"] > 0:
            signal["interpretation"] = "MODERATE_MAG_LEADERSHIP"
            signal["bias"] = "SLIGHTLY_BULLISH"
        elif signal["mag7_advancing"] <= 0.3 and signal["mag7_momentum_5d"] < -0.02:
            signal["interpretation"] = "MAG_WEAKNESS"
            signal["bias"] = "BEARISH"
        elif signal["mag7_advancing"] <= 0.5 and signal["mag7_momentum_5d"] < 0:
            signal["interpretation"] = "MODERATE_MAG_WEAKNESS"
            signal["bias"] = "SLIGHTLY_BEARISH"
        else:
            signal["interpretation"] = "NEUTRAL_MAG"
            signal["bias"] = "NEUTRAL"

        # Tech rotation signal
        if signal["tech_vs_nontech"] > 0.05:
            signal["rotation"] = "TECH_OUTPERFORMING"
        elif signal["tech_vs_nontech"] < -0.05:
            signal["rotation"] = "TECH_UNDERPERFORMING"
        else:
            signal["rotation"] = "NO_CLEAR_ROTATION"

        # Breadth divergence warning (big 3 leading but rest lagging)
        if signal["mag3_vs_mag7_breadth"] > 0.3:
            signal["breadth_warning"] = "BIG3_LEADING_NARROWLY"
        elif signal["mag7_vs_mag15_breadth"] > 0.3:
            signal["breadth_warning"] = "CORE_TECH_LEADING_NARROWLY"
        else:
            signal["breadth_warning"] = "NONE"

        # Concentration warning
        if signal["concentration_risk"] == 1:
            signal["concentration_warning"] = "HIGH_CONCENTRATION_IN_TOP3"
        else:
            signal["concentration_warning"] = "NONE"

        return signal
