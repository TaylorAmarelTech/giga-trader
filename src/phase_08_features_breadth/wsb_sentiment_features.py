"""
Wave N6: Reddit WSB PRAW Sentiment Features — Direct Reddit API access.

Uses PRAW to fetch posts from r/wallstreetbets and r/stocks,
scores them with VADER, and computes mention velocity + sentiment.

Data source chain:
  L1: praw (Reddit OAuth) — requires REDDIT_CLIENT_ID, REDDIT_CLIENT_SECRET
  L2: ApeWisdom data via existing RedditSentimentFeatures module
  L3: SPY volume/return proxy

Prefix: wsb_
Default: OFF (requires PRAW + Reddit OAuth credentials)
"""

import logging
import os
import re
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from src.core.feature_base import FeatureModuleBase

logger = logging.getLogger(__name__)

MIN_POSTS = 5
SUBREDDITS = ["wallstreetbets", "stocks", "investing"]
TICKER_RE = re.compile(r"\b([A-Z]{2,5})\b")
COMMON_WORDS = {
    "I", "A", "THE", "AND", "OR", "FOR", "IN", "ON", "IS", "IT", "AT",
    "TO", "OF", "UP", "MY", "BE", "DO", "AS", "BY", "IF", "SO", "NO",
    "DD", "WSB", "CEO", "CFO", "SEC", "ETF", "IPO", "OTC", "GDP", "CPI",
    "API", "AI", "ML", "EV", "AR", "VR", "PE", "ER", "PR", "PB", "EPS",
    "LOL", "IMO", "YOLO", "FD", "IRA", "LLC", "USD", "NFP", "AM", "PM",
    "US", "UK", "EU", "FD", "OI", "IV", "ATH", "ATL", "FOMO", "HODL",
    "TL", "DR", "TLDR", "OP", "EDIT", "PSA", "FYI", "AMA", "PT",
}


def _extract_tickers(text: str) -> List[str]:
    """Extract stock tickers from text, filtering common words."""
    return [t for t in TICKER_RE.findall(text) if t not in COMMON_WORDS]


class WSBSentimentFeatures(FeatureModuleBase):
    """Reddit WSB sentiment features via PRAW."""
    FEATURE_NAMES = ["wsb_mention_velocity", "wsb_vader_compound", "wsb_bullish_pct", "wsb_post_score_weighted", "wsb_volume_zscore", "wsb_sentiment_5d_ma", "wsb_sentiment_divergence", "wsb_regime"]


    REQUIRED_COLS = {"close"}

    def __init__(self, target_ticker: str = "SPY"):
        self._target = target_ticker
        self._data: Optional[pd.DataFrame] = None
        self._data_source: str = "none"

    # ------------------------------------------------------------------
    def download_wsb_data(
        self, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """Download recent Reddit posts from financial subreddits."""
        client_id = os.environ.get("REDDIT_CLIENT_ID", "")
        client_secret = os.environ.get("REDDIT_CLIENT_SECRET", "")
        user_agent = os.environ.get(
            "REDDIT_USER_AGENT", "GigaTrader/1.0"
        )

        if not client_id or not client_secret:
            logger.info("[WSB] No Reddit credentials found — trying ApeWisdom fallback")
            return self._try_apewisdom_fallback()

        try:
            import praw

            reddit = praw.Reddit(
                client_id=client_id,
                client_secret=client_secret,
                user_agent=user_agent,
            )

            # Load VADER for scoring
            vader = self._load_vader()

            records = []
            for sub_name in SUBREDDITS:
                try:
                    sub = reddit.subreddit(sub_name)
                    for post in sub.hot(limit=50):
                        text = f"{post.title} {post.selftext or ''}"
                        tickers = _extract_tickers(text)

                        # Check if target ticker is mentioned
                        has_target = self._target in tickers

                        # Score with VADER
                        if vader:
                            vader_scores = vader.polarity_scores(text)
                            compound = vader_scores["compound"]
                        else:
                            compound = 0.0

                        records.append({
                            "subreddit": sub_name,
                            "title": post.title[:200],
                            "score": post.score,
                            "num_comments": post.num_comments,
                            "vader_compound": compound,
                            "has_target": has_target,
                            "tickers_mentioned": tickers,
                            "created_utc": post.created_utc,
                        })
                except Exception as e:
                    logger.debug(f"[WSB] Error reading r/{sub_name}: {e}")
                    continue

            if len(records) < MIN_POSTS:
                logger.info(f"[WSB] Only {len(records)} posts — trying ApeWisdom")
                return self._try_apewisdom_fallback()

            result = pd.DataFrame(records)

            # Filter for target-relevant posts
            target_posts = result[result["has_target"]]
            if len(target_posts) < 2:
                # Use all posts as general market sentiment
                target_posts = result

            today = pd.Timestamp.now().normalize()
            agg = {
                "date": today,
                "mention_count": len(target_posts),
                "vader_mean": target_posts["vader_compound"].mean(),
                "bullish_pct": (target_posts["vader_compound"] > 0.05).mean(),
                "avg_post_score": target_posts["score"].mean(),
                "total_posts": len(result),
            }

            self._data = pd.DataFrame([agg])
            self._data_source = "praw"
            logger.info(
                f"[WSB] Fetched {len(result)} posts, "
                f"{len(target_posts)} mention {self._target}"
            )
            return self._data

        except ImportError:
            logger.info("[WSB] praw not installed — trying ApeWisdom fallback")
            return self._try_apewisdom_fallback()
        except Exception as e:
            logger.warning(f"[WSB] PRAW failed: {e} — trying ApeWisdom")
            return self._try_apewisdom_fallback()

    # ------------------------------------------------------------------
    def _try_apewisdom_fallback(self) -> Optional[pd.DataFrame]:
        """L2 fallback: try existing ApeWisdom-based Reddit data."""
        try:
            from src.phase_08_features_breadth.reddit_sentiment_features import (
                RedditSentimentFeatures,
            )

            ape = RedditSentimentFeatures()
            ape.download_reddit_data()

            if not ape.data.empty:
                self._data_source = "apewisdom"
                # Convert ApeWisdom data to our format
                today = pd.Timestamp.now().normalize()
                row = ape.data.iloc[-1] if not ape.data.empty else {}
                self._data = pd.DataFrame([{
                    "date": today,
                    "mention_count": float(row.get("spy_mentions", 0)),
                    "vader_mean": 0.0,  # ApeWisdom doesn't provide sentiment
                    "bullish_pct": 0.5,
                    "avg_post_score": float(row.get("spy_upvotes", 0)),
                    "total_posts": float(row.get("spy_mentions", 0)),
                }])
                logger.info("[WSB] Using ApeWisdom fallback data")
                return self._data
        except Exception as e:
            logger.debug(f"[WSB] ApeWisdom fallback failed: {e}")

        self._data_source = "proxy"
        return None

    # ------------------------------------------------------------------
    @staticmethod
    def _load_vader():
        """Load VADER sentiment analyzer."""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            return SentimentIntensityAnalyzer()
        except ImportError:
            try:
                import nltk
                nltk.download("vader_lexicon", quiet=True)
                from nltk.sentiment.vader import SentimentIntensityAnalyzer
                return SentimentIntensityAnalyzer()
            except Exception:
                return None

    # ------------------------------------------------------------------
    def create_wsb_features(
        self, df_daily: pd.DataFrame
    ) -> pd.DataFrame:
        """Create Reddit WSB sentiment features. Routes to full or proxy."""
        df = df_daily.copy()

        if self._data is not None and len(self._data) > 0:
            df = self._create_full_features(df)
        else:
            df = self._create_proxy_features(df)

        # NaN/Inf cleanup
        for col in self._all_feature_names():
            if col in df.columns:
                df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)
            else:
                df[col] = 0.0

        return df

    # ------------------------------------------------------------------
    def _create_full_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from actual Reddit/ApeWisdom data."""
        latest = self._data.iloc[-1]

        # Build proxy first, then override last row
        df = self._create_proxy_features(df)

        if not df.empty:
            idx = df.index[-1]
            mention_count = float(latest.get("mention_count", 0))
            vader_mean = float(latest.get("vader_mean", 0.0))
            bullish_pct = float(latest.get("bullish_pct", 0.5))
            avg_score = float(latest.get("avg_post_score", 0))

            df.loc[idx, "wsb_mention_velocity"] = mention_count / 50.0  # Normalized
            df.loc[idx, "wsb_vader_compound"] = vader_mean
            df.loc[idx, "wsb_bullish_pct"] = bullish_pct

            # Weight by post score
            if avg_score > 0:
                df.loc[idx, "wsb_post_score_weighted"] = (
                    vader_mean * min(avg_score / 100.0, 5.0)
                )
            else:
                df.loc[idx, "wsb_post_score_weighted"] = vader_mean

            # Z-score would need history; use raw for snapshot
            df.loc[idx, "wsb_volume_zscore"] = (mention_count - 50) / 30.0

            # Regime
            if vader_mean > 0.2:
                df.loc[idx, "wsb_regime"] = 1.0
            elif vader_mean < -0.2:
                df.loc[idx, "wsb_regime"] = -1.0

        return df

    # ------------------------------------------------------------------
    def _create_proxy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Proxy: use SPY volume and return as sentiment proxy."""
        if "close" in df.columns:
            ret = df["close"].pct_change().fillna(0.0)
            mom_5d = ret.rolling(5, min_periods=1).mean()

            df["wsb_mention_velocity"] = 0.5  # Neutral
            df["wsb_vader_compound"] = (mom_5d * 10).clip(-1, 1)
            df["wsb_bullish_pct"] = 0.5 + (mom_5d * 5).clip(-0.4, 0.4)
            df["wsb_post_score_weighted"] = df["wsb_vader_compound"] * 0.5
        else:
            df["wsb_mention_velocity"] = 0.5
            df["wsb_vader_compound"] = 0.0
            df["wsb_bullish_pct"] = 0.5
            df["wsb_post_score_weighted"] = 0.0

        df["wsb_volume_zscore"] = 0.0
        df["wsb_sentiment_5d_ma"] = (
            df["wsb_vader_compound"].rolling(5, min_periods=1).mean()
        )

        # Divergence: sentiment vs price direction
        if "close" in df.columns:
            ret = df["close"].pct_change().fillna(0.0)
            price_dir = np.sign(ret.rolling(5, min_periods=1).mean())
            sent_dir = np.sign(df["wsb_vader_compound"])
            df["wsb_sentiment_divergence"] = (sent_dir - price_dir).fillna(0.0)
        else:
            df["wsb_sentiment_divergence"] = 0.0

        df["wsb_regime"] = 0.0

        return df

    # ------------------------------------------------------------------
    def analyze_current_wsb(
        self, df_daily: pd.DataFrame
    ) -> Optional[Dict]:
        """Analyze current Reddit WSB sentiment."""
        if df_daily.empty:
            return None

        last = df_daily.iloc[-1]
        vader = float(last.get("wsb_vader_compound", 0.0))
        velocity = float(last.get("wsb_mention_velocity", 0.0))

        if vader > 0.3:
            regime = "EUPHORIC"
        elif vader > 0.1:
            regime = "BULLISH"
        elif vader < -0.3:
            regime = "PANIC"
        elif vader < -0.1:
            regime = "BEARISH"
        else:
            regime = "NEUTRAL"

        return {
            "regime": regime,
            "vader_compound": vader,
            "mention_velocity": velocity,
            "source": self._data_source,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _all_feature_names() -> List[str]:
        """Return all feature column names."""
        return [
            "wsb_mention_velocity",
            "wsb_vader_compound",
            "wsb_bullish_pct",
            "wsb_post_score_weighted",
            "wsb_volume_zscore",
            "wsb_sentiment_5d_ma",
            "wsb_sentiment_divergence",
            "wsb_regime",
        ]
