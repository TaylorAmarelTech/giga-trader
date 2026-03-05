"""
Wave N4: Google News Headlines Features — Financial headline sentiment.

Uses the gnews package to fetch recent headlines about SPY/market,
then scores them with keyword-based sentiment (or FinBERT if available).

Data source chain:
  L1: gnews package (Google News RSS) — free, no key
  L2: Keyword-based scoring on cached headlines
  L3: Zero-fill

Prefix: gnews_
Default: ON
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from src.core.feature_base import FeatureModuleBase

logger = logging.getLogger(__name__)

MIN_HEADLINES = 3

# Financial sentiment keyword lists (shared with alpaca_news)
_POSITIVE = {
    "beat", "beats", "surpass", "surge", "gain", "gains", "rally", "bullish",
    "upgrade", "record", "growth", "strong", "soar", "outperform", "profit",
    "recovery", "optimism", "exceed", "positive", "boost", "dividend",
    "innovation", "breakthrough", "expansion", "accelerate", "high",
}

_NEGATIVE = {
    "miss", "misses", "fall", "falls", "decline", "drop", "plunge", "bearish",
    "downgrade", "loss", "weak", "crash", "selloff", "sell-off", "underperform",
    "layoff", "recession", "default", "bankruptcy", "warning", "concern",
    "risk", "volatile", "inflation", "tariff", "sanctions", "fraud",
    "negative", "lawsuit", "crisis", "slump", "contraction", "low",
}


def _keyword_score(text: str) -> float:
    """Score text using financial keyword matching. Returns [-1, 1]."""
    if not text:
        return 0.0
    words = set(text.lower().split())
    pos = len(words & _POSITIVE)
    neg = len(words & _NEGATIVE)
    total = pos + neg
    if total == 0:
        return 0.0
    return (pos - neg) / total


class GNewsHeadlineFeatures(FeatureModuleBase):
    """Google News headline sentiment features via gnews package."""
    FEATURE_NAMES = ["gnews_headline_sentiment", "gnews_headline_count", "gnews_negative_pct", "gnews_positive_pct", "gnews_sentiment_5d_ma", "gnews_momentum", "gnews_vol_sentiment", "gnews_regime"]


    REQUIRED_COLS = {"close"}

    def __init__(self):
        self._data: Optional[pd.DataFrame] = None
        self._data_source: str = "none"

    # ------------------------------------------------------------------
    def download_gnews_data(
        self, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """Download recent financial headlines from Google News."""
        try:
            from gnews import GNews

            gn = GNews(language="en", country="US", max_results=30)

            # Search for SPY-related headlines
            search_terms = [
                "SPY stock market",
                "S&P 500 today",
            ]

            all_articles = []
            for term in search_terms:
                try:
                    articles = gn.get_news(term)
                    if articles:
                        all_articles.extend(articles)
                except Exception as e:
                    logger.debug(f"[GNEWS] Search '{term}' failed: {e}")
                    continue

            if len(all_articles) < MIN_HEADLINES:
                logger.info(
                    f"[GNEWS] Only {len(all_articles)} headlines found — using proxy"
                )
                self._data_source = "proxy"
                return None

            # Extract and score headlines
            records = []
            seen_titles = set()
            for article in all_articles:
                title = article.get("title", "")
                if not title or title in seen_titles:
                    continue
                seen_titles.add(title)

                pub_date = article.get("published date")
                try:
                    date = pd.to_datetime(pub_date).normalize()
                except Exception:
                    date = pd.Timestamp.now().normalize()

                # Try to get full text via newspaper4k
                full_text = title
                url = article.get("url", "")
                if url:
                    full_text = self._extract_article_text(url) or title

                score = _keyword_score(full_text)
                records.append({
                    "date": date,
                    "headline": title,
                    "score": score,
                    "confidence": abs(score),
                })

            if len(records) < MIN_HEADLINES:
                logger.info("[GNEWS] Insufficient scored headlines — using proxy")
                self._data_source = "proxy"
                return None

            result = pd.DataFrame(records)
            self._data = result
            self._data_source = "gnews"
            logger.info(f"[GNEWS] Scored {len(records)} headlines from Google News")
            return result

        except ImportError:
            logger.info("[GNEWS] gnews package not installed — using proxy")
            self._data_source = "proxy"
            return None
        except Exception as e:
            logger.warning(f"[GNEWS] Download failed: {e} — using proxy")
            self._data_source = "proxy"
            return None

    # ------------------------------------------------------------------
    @staticmethod
    def _extract_article_text(url: str) -> Optional[str]:
        """Try to extract full article text using newspaper4k."""
        try:
            from newspaper import Article

            article = Article(url)
            article.download()
            article.parse()
            text = article.text
            if text and len(text) > 50:
                # Truncate to first 500 chars for scoring efficiency
                return text[:500]
            return None
        except Exception:
            return None

    # ------------------------------------------------------------------
    def create_gnews_features(
        self, df_daily: pd.DataFrame
    ) -> pd.DataFrame:
        """Create Google News headline features. Routes to full or proxy."""
        df = df_daily.copy()

        if self._data is not None and len(self._data) >= MIN_HEADLINES:
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
        """Create features from actual Google News headlines."""
        news = self._data.copy()
        news["date"] = pd.to_datetime(news["date"]).dt.normalize()

        # Aggregate per day
        daily_agg = news.groupby("date").agg(
            sentiment=("score", "mean"),
            count=("score", "count"),
            neg_pct=("score", lambda x: (x < -0.1).mean()),
            pos_pct=("score", lambda x: (x > 0.1).mean()),
        ).reset_index()

        # Merge with daily data
        df["_merge_date"] = pd.to_datetime(df["date"]).dt.normalize()
        df = df.merge(
            daily_agg.rename(columns={"date": "_merge_date"}),
            on="_merge_date",
            how="left",
        )
        df.drop(columns=["_merge_date"], inplace=True, errors="ignore")

        # Forward-fill
        df["sentiment"] = df["sentiment"].ffill().fillna(0.0)
        df["count"] = df["count"].fillna(0.0)
        df["neg_pct"] = df["neg_pct"].ffill().fillna(0.0)
        df["pos_pct"] = df["pos_pct"].ffill().fillna(0.0)

        # Assign feature names
        df["gnews_headline_sentiment"] = df["sentiment"]
        df["gnews_headline_count"] = df["count"]
        df["gnews_negative_pct"] = df["neg_pct"]
        df["gnews_positive_pct"] = df["pos_pct"]

        df["gnews_sentiment_5d_ma"] = (
            df["gnews_headline_sentiment"].rolling(5, min_periods=1).mean()
        )
        df["gnews_momentum"] = df["gnews_headline_sentiment"].diff(5).fillna(0.0)

        # Interaction: high volume + strong sentiment
        count_z = (df["count"] - df["count"].rolling(20, min_periods=5).mean()) / (
            df["count"].rolling(20, min_periods=5).std().replace(0, np.nan)
        )
        df["gnews_vol_sentiment"] = (
            count_z.fillna(0.0) * df["gnews_headline_sentiment"]
        )

        df["gnews_regime"] = 0.0
        df.loc[df["gnews_headline_sentiment"] > 0.3, "gnews_regime"] = 1.0
        df.loc[df["gnews_headline_sentiment"] < -0.3, "gnews_regime"] = -1.0

        # Cleanup temp columns
        for col in ["sentiment", "count", "neg_pct", "pos_pct"]:
            df.drop(columns=[col], inplace=True, errors="ignore")

        return df

    # ------------------------------------------------------------------
    def _create_proxy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Proxy: zero-fill all features (no good proxy for news)."""
        for col in self._all_feature_names():
            df[col] = 0.0
        return df

    # ------------------------------------------------------------------
    def analyze_current_gnews(
        self, df_daily: pd.DataFrame
    ) -> Optional[Dict]:
        """Analyze current Google News sentiment."""
        if df_daily.empty:
            return None

        last = df_daily.iloc[-1]
        sent = float(last.get("gnews_headline_sentiment", 0.0))

        if sent > 0.3:
            regime = "POSITIVE"
        elif sent < -0.3:
            regime = "NEGATIVE"
        else:
            regime = "NEUTRAL"

        return {
            "regime": regime,
            "sentiment": sent,
            "headline_count": float(last.get("gnews_headline_count", 0)),
            "source": self._data_source,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _all_feature_names() -> List[str]:
        """Return all feature column names."""
        return [
            "gnews_headline_sentiment",
            "gnews_headline_count",
            "gnews_negative_pct",
            "gnews_positive_pct",
            "gnews_sentiment_5d_ma",
            "gnews_momentum",
            "gnews_vol_sentiment",
            "gnews_regime",
        ]
