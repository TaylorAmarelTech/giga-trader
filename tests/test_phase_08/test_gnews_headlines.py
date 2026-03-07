"""Tests for GNewsHeadlineFeatures class."""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.gnews_headline_features import (
    GNewsHeadlineFeatures,
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
def mock_gnews_data():
    today = pd.Timestamp.now().normalize()
    return pd.DataFrame([
        {"date": today, "headline": "SPY hits record high", "score": 0.6, "confidence": 0.6},
        {"date": today, "headline": "Market gains strong rally", "score": 0.5, "confidence": 0.5},
        {"date": today, "headline": "Investors concerned about recession", "score": -0.3, "confidence": 0.3},
        {"date": today, "headline": "Fed signals rate cuts", "score": 0.1, "confidence": 0.1},
    ])


class TestGNewsInit:
    def test_default_constructor(self):
        gn = GNewsHeadlineFeatures()
        assert gn._data is None
        assert gn._data_source == "none"


class TestGNewsKeywordScoring:
    def test_positive_headline(self):
        assert _keyword_score("Markets rally on strong growth") > 0

    def test_negative_headline(self):
        assert _keyword_score("Stocks crash amid recession fears") < 0

    def test_neutral_headline(self):
        assert _keyword_score("Markets open for trading today") == 0.0


class TestGNewsFeatures:
    def test_proxy_features_all_present(self, spy_daily):
        gn = GNewsHeadlineFeatures()
        df = gn.create_gnews_features(spy_daily)
        for col in GNewsHeadlineFeatures._all_feature_names():
            assert col in df.columns, f"Missing: {col}"

    def test_full_features_with_data(self, spy_daily, mock_gnews_data):
        gn = GNewsHeadlineFeatures()
        gn._data = mock_gnews_data
        gn._data_source = "gnews"
        df = gn.create_gnews_features(spy_daily)
        for col in GNewsHeadlineFeatures._all_feature_names():
            assert col in df.columns

    def test_no_nan_or_inf(self, spy_daily):
        gn = GNewsHeadlineFeatures()
        df = gn.create_gnews_features(spy_daily)
        for col in GNewsHeadlineFeatures._all_feature_names():
            assert not df[col].isna().any(), f"NaN in {col}"
            assert not np.isinf(df[col]).any(), f"Inf in {col}"

    def test_feature_count(self):
        assert len(GNewsHeadlineFeatures._all_feature_names()) == 8

    def test_proxy_is_zero_filled(self, spy_daily):
        gn = GNewsHeadlineFeatures()
        df = gn.create_gnews_features(spy_daily)
        for col in GNewsHeadlineFeatures._all_feature_names():
            assert (df[col] == 0.0).all(), f"{col} not zero-filled in proxy"


class TestGNewsAnalysis:
    def test_analyze_current(self, spy_daily):
        gn = GNewsHeadlineFeatures()
        df = gn.create_gnews_features(spy_daily)
        analysis = gn.analyze_current_gnews(df)
        assert analysis is not None
        assert "regime" in analysis
        assert "source" in analysis

    def test_analyze_empty(self):
        gn = GNewsHeadlineFeatures()
        assert gn.analyze_current_gnews(pd.DataFrame()) is None
