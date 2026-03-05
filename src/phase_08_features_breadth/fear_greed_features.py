"""
GIGA TRADER - CNN Fear & Greed Index Features
===============================================
Download and engineer features from the CNN Fear & Greed Index.

The CNN Fear & Greed Index (0-100) is a composite of 7 sub-indicators:
  - Market Momentum (S&P 500 vs 125-day MA)
  - Stock Price Strength (52-week highs vs lows)
  - Stock Price Breadth (advancing vs declining volume)
  - Put/Call Options ratio
  - Junk Bond Demand (yield spread vs investment grade)
  - Market Volatility (VIX vs 50-day MA)
  - Safe Haven Demand (stock vs bond returns)

Data source: CNN API endpoint (free, no API key required).
Fallback: `fear-and-greed` Python package if installed.

Features generated (prefix: fg_):
  - fg_index: Raw index value (0-100)
  - fg_index_z: 60-day z-score
  - fg_index_pctile: 252-day percentile rank
  - fg_chg_1d: 1-day change
  - fg_chg_5d: 5-day change
  - fg_regime: Categorical (0=extreme_fear, 1=fear, 2=neutral, 3=greed, 4=extreme_greed)
  - fg_extreme_signal: Binary contrarian signal (1 if <20 or >80)
  - fg_momentum_5d: 5-day momentum of index
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.core.feature_base import FeatureModuleBase

warnings.filterwarnings("ignore")

logger = logging.getLogger("FEAR_GREED_FEATURES")


class FearGreedFeatures(FeatureModuleBase):
    """
    Download CNN Fear & Greed Index data and create predictive features.

    Pattern: download → compute → merge (same as EconomicFeatures).
    """

    FEATURE_NAMES = [
        "fg_index",
        "fg_index_z",
        "fg_index_pctile",
        "fg_chg_1d",
        "fg_chg_5d",
        "fg_regime",
        "fg_extreme_signal",
        "fg_momentum_5d",
    ]

    def __init__(self):
        self.data: pd.DataFrame = pd.DataFrame()

    def download_fear_greed_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Download Fear & Greed Index historical data.

        Tries CNN API endpoint first, falls back to fear-and-greed package.
        Returns empty DataFrame on failure (graceful degradation).
        """
        print("\n[FEAR_GREED] Downloading CNN Fear & Greed Index data...")

        # Try CNN API endpoint first
        df = self._download_from_cnn_api(start_date, end_date)

        # Fallback to fear-and-greed package
        if df.empty:
            df = self._download_from_package()

        if df.empty:
            print("  [WARN] No Fear & Greed data available")
            return pd.DataFrame()

        # Filter to requested date range
        df["date"] = pd.to_datetime(df["date"])
        mask = (df["date"] >= pd.to_datetime(start_date)) & (
            df["date"] <= pd.to_datetime(end_date)
        )
        df = df[mask].reset_index(drop=True)

        self.data = df
        print(f"  [FEAR_GREED] {len(df)} days of Fear & Greed data loaded")
        return df

    def _download_from_cnn_api(
        self, start_date: datetime, end_date: datetime
    ) -> pd.DataFrame:
        """Try downloading from CNN's public API endpoint."""
        try:
            import requests
        except ImportError:
            return pd.DataFrame()

        try:
            url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            }
            resp = requests.get(url, headers=headers, timeout=30)

            if resp.status_code != 200:
                logger.info(f"  CNN API returned status {resp.status_code}")
                return pd.DataFrame()

            data = resp.json()

            # CNN API returns data in fear_and_greed_historical.data
            historical = data.get("fear_and_greed_historical", {}).get("data", [])
            if not historical:
                # Try alternative structure
                historical = data.get("fear_and_greed", {}).get("data", [])

            if not historical:
                logger.info("  CNN API: no historical data found in response")
                return pd.DataFrame()

            records = []
            for point in historical:
                try:
                    # CNN provides timestamp in milliseconds
                    ts = point.get("x", 0)
                    score = point.get("y", None)
                    if ts and score is not None:
                        dt = datetime.fromtimestamp(ts / 1000)
                        records.append({"date": dt.date(), "fg_score": float(score)})
                except (ValueError, TypeError, OSError):
                    continue

            if records:
                df = pd.DataFrame(records)
                df["date"] = pd.to_datetime(df["date"])
                # Deduplicate by date (keep last reading)
                df = df.drop_duplicates(subset="date", keep="last")
                df = df.sort_values("date").reset_index(drop=True)
                print(f"  CNN API: {len(df)} days of historical data")
                return df

        except Exception as e:
            logger.info(f"  CNN API failed: {e}")

        return pd.DataFrame()

    def _download_from_package(self) -> pd.DataFrame:
        """Try downloading via the fear-and-greed Python package."""
        try:
            import fear_and_greed

            resp = fear_and_greed.get()
            if resp and hasattr(resp, "value"):
                # Package only provides current value, not historical
                today = datetime.now().date()
                return pd.DataFrame(
                    [{"date": pd.Timestamp(today), "fg_score": float(resp.value)}]
                )
        except ImportError:
            pass
        except Exception as e:
            logger.info(f"  fear-and-greed package failed: {e}")

        return pd.DataFrame()

    def create_fear_greed_features(
        self,
        spy_daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create Fear & Greed features and merge into spy_daily.

        Produces 8 features with fg_ prefix.
        Returns original spy_daily unchanged if no data available.
        """
        if self.data.empty:
            return spy_daily

        print("\n[FEAR_GREED] Engineering features...")

        features = spy_daily.copy()
        fg = self.data.copy()
        fg["date"] = pd.to_datetime(fg["date"])

        # Ensure fg_score is numeric
        fg["fg_score"] = pd.to_numeric(fg["fg_score"], errors="coerce")
        fg = fg.dropna(subset=["fg_score"])

        if fg.empty:
            return spy_daily

        # Sort by date for rolling calculations
        fg = fg.sort_values("date").reset_index(drop=True)

        # 1. Raw index
        fg["fg_index"] = fg["fg_score"]

        # 2. 60-day z-score
        fg["fg_index_z"] = (
            (fg["fg_score"] - fg["fg_score"].rolling(60, min_periods=10).mean())
            / (fg["fg_score"].rolling(60, min_periods=10).std() + 1e-10)
        )

        # 3. 252-day percentile rank
        fg["fg_index_pctile"] = fg["fg_score"].rolling(252, min_periods=20).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1], raw=False
        )

        # 4. 1-day change
        fg["fg_chg_1d"] = fg["fg_score"].diff(1)

        # 5. 5-day change
        fg["fg_chg_5d"] = fg["fg_score"].diff(5)

        # 6. Regime classification
        def _classify_regime(score):
            if score < 20:
                return 0  # extreme_fear
            elif score < 40:
                return 1  # fear
            elif score <= 60:
                return 2  # neutral
            elif score <= 80:
                return 3  # greed
            else:
                return 4  # extreme_greed

        fg["fg_regime"] = fg["fg_score"].apply(_classify_regime)

        # 7. Extreme signal (contrarian indicator)
        fg["fg_extreme_signal"] = ((fg["fg_score"] < 20) | (fg["fg_score"] > 80)).astype(int)

        # 8. 5-day momentum
        fg["fg_momentum_5d"] = fg["fg_score"].rolling(5, min_periods=1).mean().diff(5)

        # Select feature columns for merge
        fg_feature_cols = [
            "fg_index", "fg_index_z", "fg_index_pctile",
            "fg_chg_1d", "fg_chg_5d", "fg_regime",
            "fg_extreme_signal", "fg_momentum_5d",
        ]

        merge_df = fg[["date"] + fg_feature_cols].copy()

        # Merge into spy_daily
        features = features.merge(merge_df, on="date", how="left")

        # Fill NaN with 0
        for col in fg_feature_cols:
            if col in features.columns:
                features[col] = features[col].fillna(0)

        print(f"  [FEAR_GREED] Added {len(fg_feature_cols)} features")
        return features

    def analyze_current_fear_greed(self, spy_daily: pd.DataFrame) -> Optional[dict]:
        """Analyze current Fear & Greed conditions for dashboard display."""
        if self.data.empty:
            return None

        fg = self.data.sort_values("date")
        if fg.empty:
            return None

        latest = fg.iloc[-1]
        score = float(latest["fg_score"])

        regimes = {0: "EXTREME_FEAR", 1: "FEAR", 2: "NEUTRAL", 3: "GREED", 4: "EXTREME_GREED"}
        regime_val = 0 if score < 20 else (1 if score < 40 else (2 if score <= 60 else (3 if score <= 80 else 4)))

        conditions = {
            "fear_greed_score": score,
            "fear_greed_regime": regimes[regime_val],
            "is_extreme": score < 20 or score > 80,
            "date": str(latest["date"]),
        }

        # Add historical context
        if len(fg) >= 5:
            conditions["5d_change"] = float(score - fg.iloc[-5]["fg_score"]) if len(fg) >= 5 else 0

        return conditions
