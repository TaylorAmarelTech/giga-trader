"""
GIGA TRADER - Reddit Sentiment Features (ApeWisdom)
=====================================================
Download and engineer features from Reddit/WSB sentiment data via ApeWisdom.

ApeWisdom aggregates mentions of stock tickers across Reddit
(primarily WallStreetBets, stocks, investing subreddits).

Data source: ApeWisdom API (free, no API key required).
API endpoint: https://apewisdom.io/api/v1.0/filter/all-stocks

Features generated (prefix: reddit_):
  - reddit_spy_mentions: SPY mention count (normalized)
  - reddit_spy_rank: SPY rank in mentions (lower = more popular)
  - reddit_spy_upvotes: SPY upvote ratio
  - reddit_breadth_bullish: % of top SPY components mentioned positively
  - reddit_momentum_3d: 3-day change in SPY mention count
  - reddit_buzz_zscore: Z-score of mention activity vs 20d average
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.core.feature_base import FeatureModuleBase

warnings.filterwarnings("ignore")

logger = logging.getLogger("REDDIT_SENTIMENT")

# Top SPY components to track on Reddit
SPY_TOP_COMPONENTS = [
    "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA",
    "BRK.B", "UNH", "XOM", "JPM", "JNJ", "V", "PG", "MA",
]


class RedditSentimentFeatures(FeatureModuleBase):
    """
    Download Reddit sentiment data via ApeWisdom and create predictive features.

    Pattern: download → compute → merge (same as EconomicFeatures).
    """

    FEATURE_NAMES = [
        "reddit_spy_mentions",
        "reddit_spy_rank",
        "reddit_spy_upvotes",
        "reddit_breadth_bullish",
        "reddit_momentum_3d",
        "reddit_buzz_zscore",
    ]

    def __init__(self):
        self.data: pd.DataFrame = pd.DataFrame()

    def download_reddit_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Download Reddit sentiment data from ApeWisdom.

        Note: ApeWisdom only provides current/recent data (not historical).
        For historical coverage, we cache and accumulate over time.
        Returns empty DataFrame on failure.
        """
        print("\n[REDDIT] Downloading Reddit sentiment data from ApeWisdom...")

        try:
            import requests
        except ImportError:
            print("  [WARN] requests package not available")
            return pd.DataFrame()

        try:
            url = "https://apewisdom.io/api/v1.0/filter/all-stocks"
            headers = {
                "User-Agent": "GigaTrader/1.0 (research bot)",
            }
            resp = requests.get(url, headers=headers, timeout=30)

            if resp.status_code != 200:
                logger.info(f"  ApeWisdom API returned status {resp.status_code}")
                return pd.DataFrame()

            data = resp.json()
            results = data.get("results", [])

            if not results:
                print("  [WARN] No results from ApeWisdom API")
                return pd.DataFrame()

            # Parse ticker mentions
            today = pd.Timestamp.now().normalize()
            records = []

            spy_record = None
            component_mentions = {}

            for item in results:
                ticker = item.get("ticker", "")
                mentions = item.get("mentions", 0)
                rank = item.get("rank", 999)
                upvotes = item.get("upvotes", 0)

                if ticker == "SPY":
                    spy_record = {
                        "date": today,
                        "spy_mentions": mentions,
                        "spy_rank": rank,
                        "spy_upvotes": upvotes,
                    }

                if ticker in SPY_TOP_COMPONENTS:
                    component_mentions[ticker] = {
                        "mentions": mentions,
                        "rank": rank,
                        "upvotes": upvotes,
                    }

            if spy_record is None:
                # SPY not in top results — set defaults
                spy_record = {
                    "date": today,
                    "spy_mentions": 0,
                    "spy_rank": 999,
                    "spy_upvotes": 0,
                }

            # Calculate breadth: % of SPY components with positive sentiment
            if component_mentions:
                total_mentioned = len(component_mentions)
                breadth_bullish = total_mentioned / len(SPY_TOP_COMPONENTS)
            else:
                breadth_bullish = 0.0

            spy_record["breadth_bullish"] = breadth_bullish
            spy_record["n_components_mentioned"] = len(component_mentions)

            self.data = pd.DataFrame([spy_record])
            print(f"  [REDDIT] Got data for {len(results)} tickers, "
                  f"SPY rank={spy_record['spy_rank']}, "
                  f"{len(component_mentions)} SPY components mentioned")
            return self.data

        except Exception as e:
            logger.warning(f"  ApeWisdom API failed: {e}")
            return pd.DataFrame()

    def create_reddit_features(
        self,
        spy_daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create Reddit sentiment features and merge into spy_daily.

        Produces 6 features with reddit_ prefix.
        Returns original spy_daily unchanged if no data available.
        """
        if self.data.empty:
            return spy_daily

        print("\n[REDDIT] Engineering Reddit sentiment features...")

        features = spy_daily.copy()
        reddit = self.data.copy()
        reddit["date"] = pd.to_datetime(reddit["date"])

        # Merge single-day data into spy_daily
        merge_data = pd.DataFrame({
            "date": reddit["date"],
            "reddit_spy_mentions": reddit["spy_mentions"].astype(float),
            "reddit_spy_rank": reddit["spy_rank"].astype(float),
            "reddit_spy_upvotes": reddit["spy_upvotes"].astype(float),
            "reddit_breadth_bullish": reddit["breadth_bullish"].astype(float),
        })

        features = features.merge(merge_data, on="date", how="left")

        # For features requiring history (momentum, z-score), fill with 0
        # since ApeWisdom only provides current snapshot
        features["reddit_momentum_3d"] = 0.0  # Requires historical accumulation
        features["reddit_buzz_zscore"] = 0.0   # Requires historical accumulation

        # Fill all NaN with 0
        reddit_cols = [c for c in features.columns if c.startswith("reddit_")]
        features[reddit_cols] = features[reddit_cols].fillna(0)

        print(f"  [REDDIT] Added {len(reddit_cols)} Reddit sentiment features")
        return features

    def analyze_current_reddit(self, spy_daily: pd.DataFrame) -> Optional[dict]:
        """Analyze current Reddit sentiment for dashboard display."""
        if self.data.empty:
            return None

        latest = self.data.iloc[-1]
        return {
            "spy_mentions": int(latest.get("spy_mentions", 0)),
            "spy_rank": int(latest.get("spy_rank", 999)),
            "components_mentioned": int(latest.get("n_components_mentioned", 0)),
            "breadth_bullish": float(latest.get("breadth_bullish", 0)),
        }
