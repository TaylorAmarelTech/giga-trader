"""
Wave N3: Alpaca News Sentiment Features — Benzinga headlines via Alpaca.

Uses the existing Alpaca API key to fetch Benzinga financial news headlines
and scores them with keyword-based sentiment (or FinBERT if available).

Data source chain:
  L1: alpaca-py NewsClient (Benzinga via Alpaca) — existing key
  L2: Keyword-based financial headline scoring
  L3: Zero-fill

Prefix: anews_
Default: ON
"""

import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from src.core.feature_base import FeatureModuleBase

logger = logging.getLogger(__name__)

MIN_DATA_POINTS = 5

# Financial sentiment keyword lists
POSITIVE_WORDS = {
    "beat", "beats", "surpass", "surge", "gain", "gains", "rally", "bullish",
    "upgrade", "upgraded", "record", "growth", "strong", "strength", "soar",
    "outperform", "profit", "profitable", "recovery", "optimism", "optimistic",
    "exceed", "exceeded", "positive", "boost", "boosted", "dividend", "buyback",
    "innovation", "breakthrough", "milestone", "expansion", "accelerate",
}

NEGATIVE_WORDS = {
    "miss", "misses", "fall", "falls", "decline", "drop", "plunge", "bearish",
    "downgrade", "downgraded", "loss", "weak", "weakness", "crash", "sell-off",
    "selloff", "underperform", "layoff", "layoffs", "recession", "default",
    "bankruptcy", "warning", "concern", "risk", "volatile", "volatility",
    "inflation", "tariff", "tariffs", "sanctions", "investigation", "fraud",
    "negative", "lawsuit", "shortage", "crisis", "slump", "contraction",
}


def _keyword_score(text: str) -> float:
    """Score text using financial keyword matching. Returns [-1, 1]."""
    if not text:
        return 0.0
    words = set(text.lower().split())
    pos = len(words & POSITIVE_WORDS)
    neg = len(words & NEGATIVE_WORDS)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


class AlpacaNewsFeatures(FeatureModuleBase):
    """Alpaca/Benzinga news headline sentiment features."""
    FEATURE_NAMES = ["anews_sentiment", "anews_confidence", "anews_article_count", "anews_negative_pct", "anews_positive_pct", "anews_sentiment_5d_ma", "anews_momentum", "anews_regime"]


    REQUIRED_COLS = {"close"}

    def __init__(self, symbol: str = "SPY"):
        self._symbol = symbol
        self._data: Optional[pd.DataFrame] = None
        self._data_source: str = "none"

    # ------------------------------------------------------------------
    def download_alpaca_news(
        self, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """Download news headlines from Alpaca (Benzinga) API."""
        api_key = os.environ.get("APCA_API_KEY_ID", "")
        api_secret = os.environ.get("APCA_API_SECRET_KEY", "")

        if not api_key or not api_secret:
            logger.info("[ANEWS] No Alpaca API keys found — using proxy")
            self._data_source = "proxy"
            return None

        try:
            from alpaca.data.historical.news import NewsClient
            from alpaca.data.requests import NewsRequest

            client = NewsClient(api_key=api_key, secret_key=api_secret)

            start_dt = pd.to_datetime(start)
            end_dt = pd.to_datetime(end)

            request = NewsRequest(
                symbols=[self._symbol],
                start=start_dt.isoformat(),
                end=end_dt.isoformat(),
                limit=50,
                sort="desc",
            )
            news_set = client.get_news(request)

            if not news_set or not hasattr(news_set, "news") or not news_set.news:
                logger.info("[ANEWS] No articles returned — using proxy")
                self._data_source = "proxy"
                return None

            records = []
            for article in news_set.news:
                headline = getattr(article, "headline", "") or ""
                summary = getattr(article, "summary", "") or ""
                text = f"{headline} {summary}".strip()
                created = getattr(article, "created_at", None)

                if not text or not created:
                    continue

                score = _keyword_score(text)
                records.append({
                    "date": pd.to_datetime(created).normalize(),
                    "headline": headline,
                    "score": score,
                    "confidence": abs(score),
                })

            if len(records) < MIN_DATA_POINTS:
                logger.info(
                    f"[ANEWS] Only {len(records)} articles — insufficient, using proxy"
                )
                self._data_source = "proxy"
                return None

            result = pd.DataFrame(records)
            self._data = result
            self._data_source = "alpaca_benzinga"
            logger.info(f"[ANEWS] Downloaded {len(records)} articles for {self._symbol}")
            return result

        except ImportError:
            logger.info("[ANEWS] alpaca-py not installed — using proxy")
            self._data_source = "proxy"
            return None
        except Exception as e:
            logger.warning(f"[ANEWS] Download failed: {e} — using proxy")
            self._data_source = "proxy"
            return None

    # ------------------------------------------------------------------
    def create_alpaca_news_features(
        self, df_daily: pd.DataFrame
    ) -> pd.DataFrame:
        """Create Alpaca news sentiment features. Routes to full or proxy."""
        df = df_daily.copy()

        if self._data is not None and len(self._data) >= MIN_DATA_POINTS:
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
        """Create features from actual Alpaca news data."""
        news = self._data.copy()
        news["date"] = pd.to_datetime(news["date"]).dt.normalize()

        # Aggregate per day
        daily_agg = news.groupby("date").agg(
            sentiment=("score", "mean"),
            confidence=("confidence", "mean"),
            article_count=("score", "count"),
            negative_pct=("score", lambda x: (x < -0.1).mean()),
            positive_pct=("score", lambda x: (x > 0.1).mean()),
        ).reset_index()

        # Merge with daily data
        df["_merge_date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.merge(
            daily_agg.rename(columns={"date": "_merge_date"}),
            on="_merge_date",
            how="left",
        )
        df.drop(columns=["_merge_date"], inplace=True, errors="ignore")

        # Forward-fill for days without news
        df["sentiment"] = df["sentiment"].ffill().fillna(0.0)
        df["confidence"] = df["confidence"].ffill().fillna(0.0)
        df["article_count"] = df["article_count"].fillna(0.0)
        df["negative_pct"] = df["negative_pct"].ffill().fillna(0.0)
        df["positive_pct"] = df["positive_pct"].ffill().fillna(0.0)

        # Rename to feature names
        df["anews_sentiment"] = df["sentiment"]
        df["anews_confidence"] = df["confidence"]

        # Z-score article count
        roll_mean = df["article_count"].rolling(20, min_periods=5).mean()
        roll_std = df["article_count"].rolling(20, min_periods=5).std().replace(0, np.nan)
        df["anews_article_count"] = (
            (df["article_count"] - roll_mean) / roll_std
        ).fillna(0.0)

        df["anews_negative_pct"] = df["negative_pct"]
        df["anews_positive_pct"] = df["positive_pct"]

        df["anews_sentiment_5d_ma"] = (
            df["anews_sentiment"].rolling(5, min_periods=1).mean()
        )
        df["anews_momentum"] = df["anews_sentiment"].diff(5).fillna(0.0)

        df["anews_regime"] = 0.0
        df.loc[df["anews_sentiment"] > 0.3, "anews_regime"] = 1.0
        df.loc[df["anews_sentiment"] < -0.3, "anews_regime"] = -1.0

        # Clean up temp columns
        for col in ["sentiment", "confidence", "article_count",
                     "negative_pct", "positive_pct"]:
            df.drop(columns=[col], inplace=True, errors="ignore")

        return df

    # ------------------------------------------------------------------
    def _create_proxy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Proxy: use SPY return momentum as news sentiment proxy."""
        if "close" in df.columns:
            ret = df["close"].pct_change().fillna(0.0)
            # 5d momentum as sentiment proxy
            mom = ret.rolling(5, min_periods=1).mean() * 100  # Scale up
            df["anews_sentiment"] = mom.clip(-1, 1)
            df["anews_confidence"] = abs(df["anews_sentiment"]) * 0.5
        else:
            df["anews_sentiment"] = 0.0
            df["anews_confidence"] = 0.0

        df["anews_article_count"] = 0.0
        df["anews_negative_pct"] = 0.0
        df["anews_positive_pct"] = 0.0
        df["anews_sentiment_5d_ma"] = (
            df["anews_sentiment"].rolling(5, min_periods=1).mean()
        )
        df["anews_momentum"] = df["anews_sentiment"].diff(5).fillna(0.0)
        df["anews_regime"] = 0.0

        return df

    # ------------------------------------------------------------------
    def analyze_current_news(
        self, df_daily: pd.DataFrame
    ) -> Optional[Dict]:
        """Analyze current news sentiment conditions."""
        if df_daily.empty:
            return None

        last = df_daily.iloc[-1]
        sent = float(last.get("anews_sentiment", 0.0))
        mom = float(last.get("anews_momentum", 0.0))

        if sent > 0.3:
            regime = "POSITIVE"
        elif sent < -0.3:
            regime = "NEGATIVE"
        else:
            regime = "NEUTRAL"

        return {
            "regime": regime,
            "sentiment": sent,
            "momentum": mom,
            "source": self._data_source,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _all_feature_names() -> List[str]:
        """Return all feature column names."""
        return [
            "anews_sentiment",
            "anews_confidence",
            "anews_article_count",
            "anews_negative_pct",
            "anews_positive_pct",
            "anews_sentiment_5d_ma",
            "anews_momentum",
            "anews_regime",
        ]
