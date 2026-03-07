"""Tests for FinBERTNLPFeatures class."""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.finbert_nlp_features import (
    FinBERTNLPFeatures,
    score_texts,
)


@pytest.fixture
def spy_daily():
    np.random.seed(42)
    n = 50
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 450 * np.exp(np.cumsum(np.random.normal(0.0004, 0.01, n)))
    return pd.DataFrame({
        "date": pd.to_datetime(dates.date),
        "close": close,
    })


class TestFinBERTInit:
    def test_default_constructor(self):
        nlp = FinBERTNLPFeatures()
        assert nlp._data_source == "none"
        assert len(nlp._news_texts) == 0
        assert len(nlp._social_texts) == 0


class TestFinBERTFeatures:
    def test_proxy_features_all_present(self, spy_daily):
        nlp = FinBERTNLPFeatures()
        df = nlp.create_finbert_features(spy_daily)
        for col in FinBERTNLPFeatures._all_feature_names():
            assert col in df.columns, f"Missing: {col}"

    def test_no_nan_or_inf(self, spy_daily):
        nlp = FinBERTNLPFeatures()
        df = nlp.create_finbert_features(spy_daily)
        for col in FinBERTNLPFeatures._all_feature_names():
            assert not df[col].isna().any(), f"NaN in {col}"
            assert not np.isinf(df[col]).any(), f"Inf in {col}"

    def test_feature_count(self):
        assert len(FinBERTNLPFeatures._all_feature_names()) == 8

    def test_proxy_is_zero_filled(self, spy_daily):
        nlp = FinBERTNLPFeatures()
        df = nlp.create_finbert_features(spy_daily)
        for col in FinBERTNLPFeatures._all_feature_names():
            assert (df[col] == 0.0).all(), f"{col} not zero in proxy"

    def test_set_texts(self):
        nlp = FinBERTNLPFeatures()
        nlp.set_texts(
            news_texts=["Revenue beat expectations", "Market is down"],
            social_texts=["SPY to the moon!"],
        )
        assert len(nlp._news_texts) == 2
        assert len(nlp._social_texts) == 1

    def test_set_texts_truncation(self):
        nlp = FinBERTNLPFeatures()
        long_text = "x" * 1000
        nlp.set_texts(news_texts=[long_text])
        assert len(nlp._news_texts[0]) == 512


class TestScoreTexts:
    def test_empty_texts(self):
        result = score_texts([])
        assert result == []

    def test_returns_list_of_dicts(self):
        result = score_texts(["Market is up today"])
        assert isinstance(result, list)
        assert len(result) == 1
        assert "score" in result[0]
        assert "confidence" in result[0]


class TestFinBERTAnalysis:
    def test_analyze_current(self, spy_daily):
        nlp = FinBERTNLPFeatures()
        df = nlp.create_finbert_features(spy_daily)
        analysis = nlp.analyze_current_nlp(df)
        assert analysis is not None
        assert "regime" in analysis
        assert "backend" in analysis

    def test_analyze_empty(self):
        nlp = FinBERTNLPFeatures()
        assert nlp.analyze_current_nlp(pd.DataFrame()) is None
