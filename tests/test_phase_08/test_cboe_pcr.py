"""Tests for CBOEPutCallFeatures class."""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.phase_08_features_breadth.cboe_pcr_features import CBOEPutCallFeatures


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
def mock_cboe_data():
    dates = pd.bdate_range("2024-01-01", periods=60)
    return pd.DataFrame({
        "date": dates,
        "pcr": 0.85 + np.random.normal(0, 0.15, 60),
    })


class TestCBOEInit:
    def test_default_constructor(self):
        cboe = CBOEPutCallFeatures()
        assert cboe._data is None
        assert cboe._data_source == "none"


class TestCBOEFeatureCreation:
    def test_proxy_features_all_present(self, spy_daily):
        cboe = CBOEPutCallFeatures()
        df = cboe.create_cboe_pcr_features(spy_daily)
        for col in CBOEPutCallFeatures._all_feature_names():
            assert col in df.columns, f"Missing feature: {col}"

    def test_full_features_with_data(self, spy_daily, mock_cboe_data):
        cboe = CBOEPutCallFeatures()
        cboe._data = mock_cboe_data
        cboe._data_source = "cboe_csv"
        df = cboe.create_cboe_pcr_features(spy_daily)
        for col in CBOEPutCallFeatures._all_feature_names():
            assert col in df.columns

    def test_no_nan_or_inf(self, spy_daily):
        cboe = CBOEPutCallFeatures()
        df = cboe.create_cboe_pcr_features(spy_daily)
        for col in CBOEPutCallFeatures._all_feature_names():
            assert not df[col].isna().any(), f"NaN in {col}"
            assert not np.isinf(df[col]).any(), f"Inf in {col}"

    def test_feature_count(self):
        assert len(CBOEPutCallFeatures._all_feature_names()) == 8

    def test_proxy_without_close(self):
        df = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=10)})
        cboe = CBOEPutCallFeatures()
        result = cboe.create_cboe_pcr_features(df)
        for col in CBOEPutCallFeatures._all_feature_names():
            assert col in result.columns


class TestCBOEAnalysis:
    def test_analyze_current(self, spy_daily):
        cboe = CBOEPutCallFeatures()
        df = cboe.create_cboe_pcr_features(spy_daily)
        analysis = cboe.analyze_current_cboe(df)
        assert analysis is not None
        assert "regime" in analysis
        assert "source" in analysis

    def test_analyze_empty(self):
        cboe = CBOEPutCallFeatures()
        assert cboe.analyze_current_cboe(pd.DataFrame()) is None
