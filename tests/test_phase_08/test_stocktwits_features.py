"""Tests for StockTwitsSentimentFeatures class."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.phase_08_features_breadth.stocktwits_features import StockTwitsSentimentFeatures


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
def mock_stwit_data():
    today = pd.Timestamp.now().normalize()
    return pd.DataFrame([{
        "date": today,
        "bullish": 18,
        "bearish": 7,
        "total": 30,
        "tagged": 25,
        "weighted_score": 0.35,
    }])


class TestStockTwitsInit:
    def test_default_constructor(self):
        st = StockTwitsSentimentFeatures()
        assert st._data is None
        assert st._data_source == "none"
        assert st._symbol == "SPY"

    def test_custom_symbol(self):
        st = StockTwitsSentimentFeatures(symbol="QQQ")
        assert st._symbol == "QQQ"


class TestStockTwitsFeatures:
    def test_proxy_features_all_present(self, spy_daily):
        st = StockTwitsSentimentFeatures()
        df = st.create_stocktwits_features(spy_daily)
        for col in StockTwitsSentimentFeatures._all_feature_names():
            assert col in df.columns, f"Missing: {col}"

    def test_full_features_with_data(self, spy_daily, mock_stwit_data):
        st = StockTwitsSentimentFeatures()
        st._data = mock_stwit_data
        st._data_source = "stocktwits"
        df = st.create_stocktwits_features(spy_daily)
        for col in StockTwitsSentimentFeatures._all_feature_names():
            assert col in df.columns

    def test_no_nan_or_inf(self, spy_daily):
        st = StockTwitsSentimentFeatures()
        df = st.create_stocktwits_features(spy_daily)
        for col in StockTwitsSentimentFeatures._all_feature_names():
            assert not df[col].isna().any(), f"NaN in {col}"
            assert not np.isinf(df[col]).any(), f"Inf in {col}"

    def test_feature_count(self):
        assert len(StockTwitsSentimentFeatures._all_feature_names()) == 8

    def test_proxy_without_volume(self):
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=20),
            "close": np.linspace(450, 460, 20),
        })
        st = StockTwitsSentimentFeatures()
        result = st.create_stocktwits_features(df)
        for col in StockTwitsSentimentFeatures._all_feature_names():
            assert col in result.columns

    def test_bull_ratio_range(self, spy_daily, mock_stwit_data):
        st = StockTwitsSentimentFeatures()
        st._data = mock_stwit_data
        st._data_source = "stocktwits"
        df = st.create_stocktwits_features(spy_daily)
        last = df.iloc[-1]
        assert 0.0 <= last["stwit_bull_ratio"] <= 1.0


class TestStockTwitsAnalysis:
    def test_analyze_current(self, spy_daily):
        st = StockTwitsSentimentFeatures()
        df = st.create_stocktwits_features(spy_daily)
        analysis = st.analyze_current_stocktwits(df)
        assert analysis is not None
        assert "regime" in analysis
        assert "source" in analysis

    def test_analyze_empty(self):
        st = StockTwitsSentimentFeatures()
        assert st.analyze_current_stocktwits(pd.DataFrame()) is None
