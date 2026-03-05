"""
Wave N2: StockTwits Sentiment Features — Free social sentiment from StockTwits.

Uses the free StockTwits API (no key required, 200 req/hr) to get
bullish/bearish sentiment from the 30 most recent SPY messages.

Data source chain:
  L1: StockTwits API (api.stocktwits.com) — free, no key, 200 req/hr
  L2: SPY volume/return proxy (up-volume days = bullish proxy)
  L3: Zero-fill

Prefix: stwit_
Default: ON
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from src.core.feature_base import FeatureModuleBase

logger = logging.getLogger(__name__)

STOCKTWITS_API_URL = "https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
MIN_DATA_POINTS = 10


class StockTwitsSentimentFeatures(FeatureModuleBase):
    """StockTwits social sentiment features from free API."""
    FEATURE_NAMES = ["stwit_bull_ratio", "stwit_bear_ratio", "stwit_volume", "stwit_sentiment_score", "stwit_sentiment_5d_ma", "stwit_momentum", "stwit_vol_sent_interaction", "stwit_regime"]


    REQUIRED_COLS = {"close"}

    def __init__(self, symbol: str = "SPY"):
        self._symbol = symbol
        self._data: Optional[pd.DataFrame] = None
        self._data_source: str = "none"

    # ------------------------------------------------------------------
    def download_stocktwits_data(
        self, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """Fetch recent StockTwits messages for sentiment aggregation.

        Note: The free API only returns the 30 most recent messages (no history).
        For training, this will mostly use proxy. Real value is inference-time.
        """
        try:
            import requests

            url = STOCKTWITS_API_URL.format(symbol=self._symbol)
            resp = requests.get(url, timeout=10)

            if resp.status_code == 429:
                logger.warning("[STWIT] Rate limited (200 req/hr) — using proxy")
                self._data_source = "proxy"
                return None

            if resp.status_code != 200:
                logger.info(f"[STWIT] API returned {resp.status_code} — using proxy")
                self._data_source = "proxy"
                return None

            data = resp.json()
            messages = data.get("messages", [])

            if not messages or len(messages) < 3:
                logger.info("[STWIT] Insufficient messages — using proxy")
                self._data_source = "proxy"
                return None

            # Aggregate sentiment from messages
            bullish = 0
            bearish = 0
            total = len(messages)
            follower_weighted_score = 0.0
            total_followers = 0

            for msg in messages:
                sent_obj = msg.get("entities", {}).get("sentiment")
                if sent_obj is None:
                    continue
                label = sent_obj.get("basic")
                followers = msg.get("user", {}).get("followers", 1)
                followers = max(followers, 1)

                if label == "Bullish":
                    bullish += 1
                    follower_weighted_score += followers
                    total_followers += followers
                elif label == "Bearish":
                    bearish += 1
                    follower_weighted_score -= followers
                    total_followers += followers

            tagged = bullish + bearish
            if tagged == 0:
                logger.info("[STWIT] No tagged messages — using proxy")
                self._data_source = "proxy"
                return None

            today = pd.Timestamp.now().normalize()
            result = pd.DataFrame([{
                "date": today,
                "bullish": bullish,
                "bearish": bearish,
                "total": total,
                "tagged": tagged,
                "weighted_score": (
                    follower_weighted_score / total_followers
                    if total_followers > 0 else 0.0
                ),
            }])

            self._data = result
            self._data_source = "stocktwits"
            logger.info(
                f"[STWIT] Fetched {total} msgs: {bullish} bullish, "
                f"{bearish} bearish, {total - tagged} neutral"
            )
            return result

        except Exception as e:
            logger.warning(f"[STWIT] Download failed: {e} — using proxy")
            self._data_source = "proxy"
            return None

    # ------------------------------------------------------------------
    def create_stocktwits_features(
        self, df_daily: pd.DataFrame
    ) -> pd.DataFrame:
        """Create StockTwits sentiment features. Routes to full or proxy."""
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
        """Create features from actual StockTwits data.

        Since StockTwits only gives a snapshot, we apply the latest
        sentiment to the most recent date and proxy the rest.
        """
        latest = self._data.iloc[-1]
        bullish = float(latest.get("bullish", 0))
        bearish = float(latest.get("bearish", 0))
        total = float(latest.get("total", 1))
        tagged = float(latest.get("tagged", 1))
        weighted = float(latest.get("weighted_score", 0.0))

        bull_ratio = bullish / max(tagged, 1)
        bear_ratio = bearish / max(tagged, 1)
        sentiment = (bullish - bearish) / max(tagged, 1)

        # Apply to most recent row, proxy the rest
        df = self._create_proxy_features(df)

        # Override last row with real data
        if not df.empty:
            idx = df.index[-1]
            df.loc[idx, "stwit_bull_ratio"] = bull_ratio
            df.loc[idx, "stwit_bear_ratio"] = bear_ratio
            df.loc[idx, "stwit_volume"] = total / 30.0  # Normalized (30 is max per call)
            df.loc[idx, "stwit_sentiment_score"] = sentiment
            df.loc[idx, "stwit_vol_sent_interaction"] = (total / 30.0) * sentiment

            if sentiment > 0.3:
                df.loc[idx, "stwit_regime"] = 1.0
            elif sentiment < -0.3:
                df.loc[idx, "stwit_regime"] = -1.0
            else:
                df.loc[idx, "stwit_regime"] = 0.0

        return df

    # ------------------------------------------------------------------
    def _create_proxy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Proxy: use SPY up-volume ratio as sentiment proxy."""
        if "close" in df.columns and "volume" in df.columns:
            ret = df["close"].pct_change().fillna(0.0)
            vol = df["volume"].fillna(df["volume"].median())

            # Up-volume ratio as bullish proxy
            up_vol = (ret > 0).astype(float) * vol
            total_vol = vol.replace(0, np.nan)
            up_ratio = (
                up_vol.rolling(5, min_periods=1).sum()
                / total_vol.rolling(5, min_periods=1).sum()
            ).fillna(0.5)

            df["stwit_bull_ratio"] = up_ratio
            df["stwit_bear_ratio"] = 1.0 - up_ratio
            df["stwit_sentiment_score"] = (up_ratio - 0.5) * 2.0  # Scale to [-1, 1]
        elif "close" in df.columns:
            ret = df["close"].pct_change().fillna(0.0)
            # Simple momentum proxy
            mom_5d = ret.rolling(5, min_periods=1).mean()
            df["stwit_bull_ratio"] = 0.5 + mom_5d.clip(-0.5, 0.5)
            df["stwit_bear_ratio"] = 1.0 - df["stwit_bull_ratio"]
            df["stwit_sentiment_score"] = (df["stwit_bull_ratio"] - 0.5) * 2.0
        else:
            df["stwit_bull_ratio"] = 0.5
            df["stwit_bear_ratio"] = 0.5
            df["stwit_sentiment_score"] = 0.0

        df["stwit_volume"] = 0.5  # Neutral volume
        df["stwit_sentiment_5d_ma"] = (
            df["stwit_sentiment_score"].rolling(5, min_periods=1).mean()
        )
        df["stwit_momentum"] = df["stwit_sentiment_score"].diff(5).fillna(0.0)
        df["stwit_vol_sent_interaction"] = (
            df["stwit_volume"] * df["stwit_sentiment_score"]
        )
        df["stwit_regime"] = 0.0

        return df

    # ------------------------------------------------------------------
    def analyze_current_stocktwits(
        self, df_daily: pd.DataFrame
    ) -> Optional[Dict]:
        """Analyze current StockTwits sentiment."""
        if df_daily.empty:
            return None

        last = df_daily.iloc[-1]
        score = float(last.get("stwit_sentiment_score", 0.0))
        bull = float(last.get("stwit_bull_ratio", 0.5))

        if score > 0.4:
            regime = "STRONG_BULLISH"
        elif score > 0.15:
            regime = "BULLISH"
        elif score < -0.4:
            regime = "STRONG_BEARISH"
        elif score < -0.15:
            regime = "BEARISH"
        else:
            regime = "NEUTRAL"

        return {
            "regime": regime,
            "sentiment_score": score,
            "bull_ratio": bull,
            "source": self._data_source,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _all_feature_names() -> List[str]:
        """Return all feature column names."""
        return [
            "stwit_bull_ratio",
            "stwit_bear_ratio",
            "stwit_volume",
            "stwit_sentiment_score",
            "stwit_sentiment_5d_ma",
            "stwit_momentum",
            "stwit_vol_sent_interaction",
            "stwit_regime",
        ]
