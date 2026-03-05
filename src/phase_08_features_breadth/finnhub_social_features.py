"""
GIGA TRADER - Finnhub Social Sentiment Features
==================================================
Download and engineer features from Finnhub social sentiment API.

Finnhub provides aggregated social media sentiment metrics for tickers
from Reddit and Twitter/X, including mention counts, sentiment scores,
and trend data.

Requires: FINNHUB_API_KEY environment variable.
Gracefully returns empty DataFrame if key is missing.

Data source: Finnhub API (free tier, rate limited).
API endpoint: https://finnhub.io/api/v1/stock/social-sentiment

Features generated (prefix: finnhub_social_):
  - finnhub_social_reddit_mentions: Reddit mention count (normalized)
  - finnhub_social_twitter_mentions: Twitter mention count (normalized)
  - finnhub_social_total_mentions: Combined mention count
  - finnhub_social_positive_pct: Percentage of positive mentions
  - finnhub_social_score: Composite sentiment score (-1 to +1)
  - finnhub_social_buzz_zscore: Z-score of total mentions vs 20d average
"""

import logging
import os
import warnings
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.core.feature_base import FeatureModuleBase

warnings.filterwarnings("ignore")

logger = logging.getLogger("FINNHUB_SOCIAL")


class FinnhubSocialFeatures(FeatureModuleBase):
    """
    Download Finnhub social sentiment data and create predictive features.

    Pattern: download → compute → merge (same as EconomicFeatures).
    """

    FEATURE_NAMES = [
        "finnhub_social_reddit_mentions",
        "finnhub_social_twitter_mentions",
        "finnhub_social_total_mentions",
        "finnhub_social_positive_pct",
        "finnhub_social_score",
        "finnhub_social_buzz_zscore",
    ]

    def __init__(self):
        self.data: pd.DataFrame = pd.DataFrame()
        self._api_key: Optional[str] = None

    def _get_api_key(self) -> Optional[str]:
        """Get Finnhub API key from environment."""
        if self._api_key:
            return self._api_key

        # Try .env file first
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        self._api_key = os.environ.get("FINNHUB_API_KEY")
        return self._api_key

    def download_finnhub_social_data(
        self,
        start_date: datetime,
        end_date: datetime,
        symbol: str = "SPY",
    ) -> pd.DataFrame:
        """
        Download social sentiment data from Finnhub API.

        Note: Finnhub free tier provides recent data only.
        For historical coverage, we cache and accumulate over time.
        Returns empty DataFrame if API key missing or on failure.
        """
        print("\n[FINNHUB_SOCIAL] Downloading social sentiment data...")

        api_key = self._get_api_key()
        if not api_key:
            print("  [WARN] FINNHUB_API_KEY not set - skipping Finnhub social sentiment")
            return pd.DataFrame()

        try:
            import requests
        except ImportError:
            print("  [WARN] requests package not available")
            return pd.DataFrame()

        try:
            # Fetch social sentiment for SPY
            url = "https://finnhub.io/api/v1/stock/social-sentiment"
            params = {
                "symbol": symbol,
                "from": pd.Timestamp(start_date).strftime("%Y-%m-%d"),
                "to": pd.Timestamp(end_date).strftime("%Y-%m-%d"),
                "token": api_key,
            }

            resp = requests.get(url, params=params, timeout=30)

            if resp.status_code == 401:
                print("  [WARN] Finnhub API key is invalid")
                return pd.DataFrame()

            if resp.status_code == 429:
                print("  [WARN] Finnhub API rate limit exceeded")
                return pd.DataFrame()

            if resp.status_code != 200:
                logger.info(f"  Finnhub API returned status {resp.status_code}")
                return pd.DataFrame()

            data = resp.json()
            reddit_data = data.get("reddit", [])
            twitter_data = data.get("twitter", [])

            if not reddit_data and not twitter_data:
                print("  [WARN] No social sentiment data returned")
                return pd.DataFrame()

            # Parse Reddit data
            records = {}
            for item in reddit_data:
                dt = item.get("atTime", "")
                if not dt:
                    continue
                date_key = pd.Timestamp(dt).normalize()
                if date_key not in records:
                    records[date_key] = {
                        "date": date_key,
                        "reddit_mentions": 0,
                        "reddit_positive": 0,
                        "reddit_negative": 0,
                        "twitter_mentions": 0,
                        "twitter_positive": 0,
                        "twitter_negative": 0,
                    }
                records[date_key]["reddit_mentions"] += item.get("mention", 0)
                records[date_key]["reddit_positive"] += item.get("positiveScore", 0)
                records[date_key]["reddit_negative"] += item.get("negativeScore", 0)

            # Parse Twitter data
            for item in twitter_data:
                dt = item.get("atTime", "")
                if not dt:
                    continue
                date_key = pd.Timestamp(dt).normalize()
                if date_key not in records:
                    records[date_key] = {
                        "date": date_key,
                        "reddit_mentions": 0,
                        "reddit_positive": 0,
                        "reddit_negative": 0,
                        "twitter_mentions": 0,
                        "twitter_positive": 0,
                        "twitter_negative": 0,
                    }
                records[date_key]["twitter_mentions"] += item.get("mention", 0)
                records[date_key]["twitter_positive"] += item.get("positiveScore", 0)
                records[date_key]["twitter_negative"] += item.get("negativeScore", 0)

            if not records:
                print("  [WARN] No parseable social data")
                return pd.DataFrame()

            self.data = pd.DataFrame(list(records.values()))
            self.data = self.data.sort_values("date").reset_index(drop=True)
            print(f"  [FINNHUB_SOCIAL] Got {len(self.data)} days of social sentiment data")
            return self.data

        except Exception as e:
            logger.warning(f"  Finnhub social API failed: {e}")
            return pd.DataFrame()

    def create_finnhub_social_features(
        self,
        spy_daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create Finnhub social sentiment features and merge into spy_daily.

        Produces 6 features with finnhub_social_ prefix.
        Returns original spy_daily unchanged if no data available.
        """
        if self.data.empty:
            return spy_daily

        print("\n[FINNHUB_SOCIAL] Engineering Finnhub social sentiment features...")

        features = spy_daily.copy()
        social = self.data.copy()
        social["date"] = pd.to_datetime(social["date"])

        # Compute aggregated metrics
        social["finnhub_social_reddit_mentions"] = social["reddit_mentions"].astype(float)
        social["finnhub_social_twitter_mentions"] = social["twitter_mentions"].astype(float)
        social["finnhub_social_total_mentions"] = (
            social["reddit_mentions"] + social["twitter_mentions"]
        ).astype(float)

        # Positive percentage
        total_sentiment = (
            social["reddit_positive"] + social["reddit_negative"]
            + social["twitter_positive"] + social["twitter_negative"]
        )
        social["finnhub_social_positive_pct"] = np.where(
            total_sentiment > 0,
            (social["reddit_positive"] + social["twitter_positive"]) / total_sentiment,
            0.5,  # Neutral when no data
        )

        # Composite score: (positive - negative) / total, range [-1, +1]
        pos_total = social["reddit_positive"] + social["twitter_positive"]
        neg_total = social["reddit_negative"] + social["twitter_negative"]
        total = pos_total + neg_total
        social["finnhub_social_score"] = np.where(
            total > 0,
            (pos_total - neg_total) / total,
            0.0,
        )

        # Buzz z-score (mentions vs 20-day rolling average)
        rolling_mean = social["finnhub_social_total_mentions"].rolling(20, min_periods=1).mean()
        rolling_std = social["finnhub_social_total_mentions"].rolling(20, min_periods=1).std()
        social["finnhub_social_buzz_zscore"] = np.where(
            rolling_std > 0,
            (social["finnhub_social_total_mentions"] - rolling_mean) / rolling_std,
            0.0,
        )
        social["finnhub_social_buzz_zscore"] = social["finnhub_social_buzz_zscore"].clip(-3, 3)

        # Merge into spy_daily
        merge_cols = [
            "date", "finnhub_social_reddit_mentions", "finnhub_social_twitter_mentions",
            "finnhub_social_total_mentions", "finnhub_social_positive_pct",
            "finnhub_social_score", "finnhub_social_buzz_zscore",
        ]
        merge_data = social[merge_cols].copy()

        features = features.merge(merge_data, on="date", how="left")

        # Fill NaN with neutral values
        fh_cols = [c for c in features.columns if c.startswith("finnhub_social_")]
        features[fh_cols] = features[fh_cols].fillna(0)

        print(f"  [FINNHUB_SOCIAL] Added {len(fh_cols)} Finnhub social features")
        return features

    def analyze_current_finnhub_social(self, spy_daily: pd.DataFrame) -> Optional[dict]:
        """Analyze current Finnhub social sentiment for dashboard display."""
        if self.data.empty:
            return None

        fh_cols = [c for c in spy_daily.columns if c.startswith("finnhub_social_")]
        if not fh_cols or spy_daily.empty:
            return None

        latest = spy_daily.iloc[-1]
        score = float(latest.get("finnhub_social_score", 0))
        sentiment = "bullish" if score > 0.2 else "bearish" if score < -0.2 else "neutral"

        return {
            "social_score": score,
            "social_sentiment": sentiment,
            "total_mentions": int(latest.get("finnhub_social_total_mentions", 0)),
            "positive_pct": float(latest.get("finnhub_social_positive_pct", 0.5)),
            "buzz_zscore": float(latest.get("finnhub_social_buzz_zscore", 0)),
        }
