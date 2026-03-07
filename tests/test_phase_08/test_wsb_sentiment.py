"""Tests for WSBSentimentFeatures class."""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.wsb_sentiment_features import (
    WSBSentimentFeatures,
    _extract_tickers,
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
        "volume": np.random.randint(50_000_000, 200_000_000, n),
    })


@pytest.fixture
def mock_wsb_data():
    today = pd.Timestamp.now().normalize()
    return pd.DataFrame([{
        "date": today,
        "mention_count": 85,
        "vader_mean": 0.25,
        "bullish_pct": 0.62,
        "avg_post_score": 150,
        "total_posts": 120,
    }])


class TestWSBInit:
    def test_default_constructor(self):
        wsb = WSBSentimentFeatures()
        assert wsb._data is None
        assert wsb._data_source == "none"
        assert wsb._target == "SPY"

    def test_custom_target(self):
        wsb = WSBSentimentFeatures(target_ticker="NVDA")
        assert wsb._target == "NVDA"


class TestTickerExtraction:
    def test_extract_tickers(self):
        tickers = _extract_tickers("I bought SPY and AAPL today")
        assert "SPY" in tickers
        assert "AAPL" in tickers

    def test_filter_common_words(self):
        tickers = _extract_tickers("I THE AND FOR DD")
        assert len(tickers) == 0

    def test_short_tickers_filtered(self):
        # Single-char tickers should be filtered (min length 2)
        tickers = _extract_tickers("I bought A")
        assert "A" not in tickers


class TestWSBFeatures:
    def test_proxy_features_all_present(self, spy_daily):
        wsb = WSBSentimentFeatures()
        df = wsb.create_wsb_features(spy_daily)
        for col in WSBSentimentFeatures._all_feature_names():
            assert col in df.columns, f"Missing: {col}"

    def test_full_features_with_data(self, spy_daily, mock_wsb_data):
        wsb = WSBSentimentFeatures()
        wsb._data = mock_wsb_data
        wsb._data_source = "praw"
        df = wsb.create_wsb_features(spy_daily)
        for col in WSBSentimentFeatures._all_feature_names():
            assert col in df.columns

    def test_no_nan_or_inf(self, spy_daily):
        wsb = WSBSentimentFeatures()
        df = wsb.create_wsb_features(spy_daily)
        for col in WSBSentimentFeatures._all_feature_names():
            assert not df[col].isna().any(), f"NaN in {col}"
            assert not np.isinf(df[col]).any(), f"Inf in {col}"

    def test_feature_count(self):
        assert len(WSBSentimentFeatures._all_feature_names()) == 8

    def test_proxy_without_volume(self):
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=20),
            "close": np.linspace(450, 460, 20),
        })
        wsb = WSBSentimentFeatures()
        result = wsb.create_wsb_features(df)
        for col in WSBSentimentFeatures._all_feature_names():
            assert col in result.columns


class TestWSBAnalysis:
    def test_analyze_current(self, spy_daily):
        wsb = WSBSentimentFeatures()
        df = wsb.create_wsb_features(spy_daily)
        analysis = wsb.analyze_current_wsb(df)
        assert analysis is not None
        assert "regime" in analysis
        assert "source" in analysis

    def test_analyze_empty(self):
        wsb = WSBSentimentFeatures()
        assert wsb.analyze_current_wsb(pd.DataFrame()) is None
