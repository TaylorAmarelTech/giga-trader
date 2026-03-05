"""
Wave N5: FinBERT Local NLP Engine — Transformer-based financial sentiment.

Provides local NLP scoring using FinBERT (distilroberta for news,
FinTwitBERT for social media). Also serves as a utility that other
modules (N3 Alpaca News, N4 GNews) can call for text scoring.

Data source chain:
  L1: transformers pipeline (distilroberta + FinTwitBERT) — offline model
  L2: VADER sentiment (vaderSentiment) — lightweight lexicon
  L3: Zero-fill

Prefix: nlp_
Default: OFF (requires transformers + torch, ~2GB download)
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional
from src.core.feature_base import FeatureModuleBase

logger = logging.getLogger(__name__)

MIN_TEXTS = 3

# Singleton model caches (class-level to avoid reloading)
_NEWS_PIPELINE = None
_SOCIAL_PIPELINE = None
_VADER_ANALYZER = None
_BACKEND = "none"  # "transformers", "vader", "none"


def _load_news_pipeline():
    """Load distilroberta financial sentiment pipeline (lazy singleton)."""
    global _NEWS_PIPELINE, _BACKEND
    if _NEWS_PIPELINE is not None:
        return _NEWS_PIPELINE
    try:
        from transformers import pipeline
        _NEWS_PIPELINE = pipeline(
            "text-classification",
            model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
            device=-1,  # CPU
            top_k=None,
        )
        _BACKEND = "transformers"
        logger.info("[NLP] Loaded distilroberta financial news pipeline")
        return _NEWS_PIPELINE
    except Exception as e:
        logger.info(f"[NLP] transformers not available: {e}")
        return None


def _load_social_pipeline():
    """Load FinTwitBERT social sentiment pipeline (lazy singleton)."""
    global _SOCIAL_PIPELINE
    if _SOCIAL_PIPELINE is not None:
        return _SOCIAL_PIPELINE
    try:
        from transformers import pipeline
        _SOCIAL_PIPELINE = pipeline(
            "sentiment-analysis",
            model="StephanAkkerman/FinTwitBERT-sentiment",
            device=-1,
            top_k=None,
        )
        logger.info("[NLP] Loaded FinTwitBERT social sentiment pipeline")
        return _SOCIAL_PIPELINE
    except Exception as e:
        logger.debug(f"[NLP] FinTwitBERT not available: {e}")
        return None


def _load_vader():
    """Load VADER sentiment analyzer (lazy singleton)."""
    global _VADER_ANALYZER, _BACKEND
    if _VADER_ANALYZER is not None:
        return _VADER_ANALYZER
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _VADER_ANALYZER = SentimentIntensityAnalyzer()
        if _BACKEND == "none":
            _BACKEND = "vader"
        logger.info("[NLP] Loaded VADER sentiment analyzer")
        return _VADER_ANALYZER
    except ImportError:
        try:
            import nltk
            nltk.download("vader_lexicon", quiet=True)
            from nltk.sentiment.vader import SentimentIntensityAnalyzer
            _VADER_ANALYZER = SentimentIntensityAnalyzer()
            if _BACKEND == "none":
                _BACKEND = "vader"
            logger.info("[NLP] Loaded NLTK VADER sentiment analyzer")
            return _VADER_ANALYZER
        except Exception:
            logger.info("[NLP] VADER not available")
            return None


def _transformers_score(texts: List[str], pipe) -> List[Dict]:
    """Score texts with a transformers pipeline. Returns list of {score, confidence}."""
    results = []
    try:
        raw_results = pipe(
            texts, batch_size=16, truncation=True, max_length=512
        )
        for result_list in raw_results:
            if isinstance(result_list, dict):
                result_list = [result_list]
            # Map labels to numeric score
            label_map = {"positive": 1.0, "neutral": 0.0, "negative": -1.0}
            score = 0.0
            confidence = 0.0
            for r in result_list:
                label = r.get("label", "neutral").lower()
                prob = r.get("score", 0.0)
                mapped = label_map.get(label, 0.0)
                score += mapped * prob
                if abs(mapped) > 0:
                    confidence = max(confidence, prob)
            results.append({"score": score, "confidence": confidence})
    except Exception as e:
        logger.warning(f"[NLP] Transformers scoring failed: {e}")
        results = [{"score": 0.0, "confidence": 0.0}] * len(texts)
    return results


def _vader_score(texts: List[str], analyzer) -> List[Dict]:
    """Score texts with VADER. Returns list of {score, confidence}."""
    results = []
    for text in texts:
        try:
            scores = analyzer.polarity_scores(text)
            compound = scores["compound"]
            results.append({
                "score": compound,
                "confidence": abs(compound),
            })
        except Exception:
            results.append({"score": 0.0, "confidence": 0.0})
    return results


def score_texts(
    texts: List[str], source_type: str = "news"
) -> List[Dict]:
    """Public utility: score texts with best available NLP backend.

    Args:
        texts: List of text strings to score
        source_type: "news" (use distilroberta) or "social" (use FinTwitBERT)

    Returns:
        List of {"score": float, "confidence": float} dicts
    """
    if not texts:
        return []

    # Try transformers first
    if source_type == "social":
        pipe = _load_social_pipeline()
    else:
        pipe = _load_news_pipeline()

    if pipe is not None:
        return _transformers_score(texts, pipe)

    # Fallback to VADER
    vader = _load_vader()
    if vader is not None:
        return _vader_score(texts, vader)

    # No NLP available
    return [{"score": 0.0, "confidence": 0.0}] * len(texts)


class FinBERTNLPFeatures(FeatureModuleBase):
    """FinBERT-based local NLP sentiment features."""
    FEATURE_NAMES = ["nlp_news_sentiment", "nlp_news_confidence", "nlp_social_sentiment", "nlp_social_confidence", "nlp_combined_score", "nlp_dispersion", "nlp_extreme_pct", "nlp_regime"]


    REQUIRED_COLS = {"close"}

    def __init__(self):
        self._news_texts: List[str] = []
        self._social_texts: List[str] = []
        self._news_scores: Optional[List[Dict]] = None
        self._social_scores: Optional[List[Dict]] = None
        self._data_source: str = "none"

    # ------------------------------------------------------------------
    def download_nlp_data(
        self, start: str, end: str
    ) -> Optional[pd.DataFrame]:
        """No-op download — this module scores text provided by others.

        Call set_texts() before create_finbert_features() to provide input.
        """
        # Probe which backend is available
        _load_news_pipeline()
        if _BACKEND == "none":
            _load_vader()
        self._data_source = _BACKEND
        if _BACKEND == "none":
            logger.info("[NLP] No NLP backend available — features will be zero-filled")
        else:
            logger.info(f"[NLP] Using backend: {_BACKEND}")
        return None

    # ------------------------------------------------------------------
    def set_texts(
        self,
        news_texts: Optional[List[str]] = None,
        social_texts: Optional[List[str]] = None,
    ):
        """Provide texts for scoring. Called by other modules (N3, N4)."""
        if news_texts:
            self._news_texts = [t[:512] for t in news_texts if t]
        if social_texts:
            self._social_texts = [t[:512] for t in social_texts if t]

    # ------------------------------------------------------------------
    def create_finbert_features(
        self, df_daily: pd.DataFrame
    ) -> pd.DataFrame:
        """Create FinBERT NLP features from provided texts."""
        df = df_daily.copy()

        if _BACKEND == "none" or (
            len(self._news_texts) < MIN_TEXTS
            and len(self._social_texts) < MIN_TEXTS
        ):
            return self._create_proxy_features(df)

        # Score texts
        if self._news_texts:
            self._news_scores = score_texts(self._news_texts, "news")
        if self._social_texts:
            self._social_scores = score_texts(self._social_texts, "social")

        # Compute aggregate features
        news_scores = [s["score"] for s in (self._news_scores or [])]
        news_confs = [s["confidence"] for s in (self._news_scores or [])]
        social_scores = [s["score"] for s in (self._social_scores or [])]
        social_confs = [s["confidence"] for s in (self._social_scores or [])]

        all_scores = news_scores + social_scores

        news_sent = float(np.mean(news_scores)) if news_scores else 0.0
        news_conf = float(np.mean(news_confs)) if news_confs else 0.0
        social_sent = float(np.mean(social_scores)) if social_scores else 0.0
        social_conf = float(np.mean(social_confs)) if social_confs else 0.0

        combined = 0.6 * news_sent + 0.4 * social_sent if news_scores else social_sent
        dispersion = float(np.std(all_scores)) if len(all_scores) > 1 else 0.0
        extreme_pct = (
            sum(1 for s in all_scores if abs(s) > 0.7) / max(len(all_scores), 1)
        )

        regime = 0.0
        if combined > 0.3:
            regime = 1.0
        elif combined < -0.3:
            regime = -1.0

        # Apply to last row (snapshot), rest get proxy
        df = self._create_proxy_features(df)
        if not df.empty:
            idx = df.index[-1]
            df.loc[idx, "nlp_news_sentiment"] = news_sent
            df.loc[idx, "nlp_news_confidence"] = news_conf
            df.loc[idx, "nlp_social_sentiment"] = social_sent
            df.loc[idx, "nlp_social_confidence"] = social_conf
            df.loc[idx, "nlp_combined_score"] = combined
            df.loc[idx, "nlp_dispersion"] = dispersion
            df.loc[idx, "nlp_extreme_pct"] = extreme_pct
            df.loc[idx, "nlp_regime"] = regime

        # NaN/Inf cleanup
        for col in self._all_feature_names():
            if col in df.columns:
                df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)
            else:
                df[col] = 0.0

        return df

    # ------------------------------------------------------------------
    def _create_proxy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Zero-fill all NLP features (no meaningful proxy)."""
        for col in self._all_feature_names():
            df[col] = 0.0
        return df

    # ------------------------------------------------------------------
    def analyze_current_nlp(
        self, df_daily: pd.DataFrame
    ) -> Optional[Dict]:
        """Analyze current NLP sentiment conditions."""
        if df_daily.empty:
            return None

        last = df_daily.iloc[-1]
        combined = float(last.get("nlp_combined_score", 0.0))

        if combined > 0.3:
            regime = "BULLISH"
        elif combined < -0.3:
            regime = "BEARISH"
        else:
            regime = "NEUTRAL"

        return {
            "regime": regime,
            "combined_score": combined,
            "backend": _BACKEND,
            "source": self._data_source,
        }

    # ------------------------------------------------------------------
    @staticmethod
    def _all_feature_names() -> List[str]:
        """Return all feature column names."""
        return [
            "nlp_news_sentiment",
            "nlp_news_confidence",
            "nlp_social_sentiment",
            "nlp_social_confidence",
            "nlp_combined_score",
            "nlp_dispersion",
            "nlp_extreme_pct",
            "nlp_regime",
        ]
