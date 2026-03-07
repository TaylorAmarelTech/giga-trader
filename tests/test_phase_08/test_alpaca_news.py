"""Tests for AlpacaNewsFeatures class."""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.alpaca_news_features import (
    AlpacaNewsFeatures,
    _keyword_score,
)


@pytest.fixture
def spy_daily():
    np.random.seed(42)
    n = 100
    dates = pd.bdate_range("2024-01-01", periods=n)
    close = 450 * np.exp(np.cumsum(np.random.normal(0.0004, 0.01, n)))
    return pd.DataFrame({
        "date": pd.to_datetime(dates.date),
        "close": close,
    })


@pytest.fixture
def mock_news_data():
    dates = pd.bdate_range("2024-03-01", periods=10)
    records = []
    for d in dates:
        records.append({"date": d, "headline": "SPY rally strong gains", "score": 0.5, "confidence": 0.5})
        records.append({"date": d, "headline": "Market decline risk warning", "score": -0.4, "confidence": 0.4})
    return pd.DataFrame(records)


class TestAlpacaNewsInit:
    def test_default_constructor(self):
        an = AlpacaNewsFeatures()
        assert an._data is None
        assert an._data_source == "none"


class TestKeywordScoring:
    def test_positive_text(self):
        score = _keyword_score("Market rally strong gains record growth")
        assert score > 0

    def test_negative_text(self):
        score = _keyword_score("Market crash decline loss weak recession")
        assert score < 0

    def test_neutral_text(self):
        score = _keyword_score("The market traded today as expected")
        assert score == 0.0

    def test_empty_text(self):
        assert _keyword_score("") == 0.0
        assert _keyword_score(None) == 0.0


class TestAlpacaNewsFeatures:
    def test_proxy_features_all_present(self, spy_daily):
        an = AlpacaNewsFeatures()
        df = an.create_alpaca_news_features(spy_daily)
        for col in AlpacaNewsFeatures._all_feature_names():
            assert col in df.columns, f"Missing: {col}"

    def test_full_features_with_data(self, spy_daily, mock_news_data):
        an = AlpacaNewsFeatures()
        an._data = mock_news_data
        an._data_source = "alpaca_benzinga"
        df = an.create_alpaca_news_features(spy_daily)
        for col in AlpacaNewsFeatures._all_feature_names():
            assert col in df.columns

    def test_no_nan_or_inf(self, spy_daily):
        an = AlpacaNewsFeatures()
        df = an.create_alpaca_news_features(spy_daily)
        for col in AlpacaNewsFeatures._all_feature_names():
            assert not df[col].isna().any(), f"NaN in {col}"
            assert not np.isinf(df[col]).any(), f"Inf in {col}"

    def test_feature_count(self):
        assert len(AlpacaNewsFeatures._all_feature_names()) == 8


class TestAlpacaNewsAnalysis:
    def test_analyze_current(self, spy_daily):
        an = AlpacaNewsFeatures()
        df = an.create_alpaca_news_features(spy_daily)
        analysis = an.analyze_current_news(df)
        assert analysis is not None
        assert "regime" in analysis

    def test_analyze_empty(self):
        an = AlpacaNewsFeatures()
        assert an.analyze_current_news(pd.DataFrame()) is None
