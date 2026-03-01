"""Tests for HARRVFeatures (harv_ prefix, 4 features)."""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.har_rv_features import HARRVFeatures
from src.phase_10_feature_processing.group_aware_processor import FEATURE_GROUPS


@pytest.fixture
def ohlcv_df():
    np.random.seed(42)
    n = 300
    close = 400.0 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "close": close,
        "high": close + np.abs(np.random.randn(n) * 0.8),
        "low": close - np.abs(np.random.randn(n) * 0.8),
    })


@pytest.fixture
def short_df():
    return pd.DataFrame({"close": [400.0 + i * 0.1 for i in range(15)]})


@pytest.fixture
def harv():
    return HARRVFeatures()


class TestInvariants:
    def test_all_4_features_created(self, harv, ohlcv_df):
        result = harv.create_har_rv_features(ohlcv_df)
        cols = [c for c in result.columns if c.startswith("harv_")]
        assert len(cols) == 4

    def test_no_nan(self, harv, ohlcv_df):
        result = harv.create_har_rv_features(ohlcv_df)
        for col in [c for c in result.columns if c.startswith("harv_")]:
            assert not result[col].isna().any(), f"NaN in {col}"

    def test_no_inf(self, harv, ohlcv_df):
        result = harv.create_har_rv_features(ohlcv_df)
        for col in [c for c in result.columns if c.startswith("harv_")]:
            assert not np.isinf(result[col].values).any(), f"Inf in {col}"

    def test_preserves_rows(self, harv, ohlcv_df):
        result = harv.create_har_rv_features(ohlcv_df)
        assert len(result) == len(ohlcv_df)

    def test_short_data_no_crash(self, harv, short_df):
        result = harv.create_har_rv_features(short_df)
        assert len(result) == len(short_df)


class TestHARRV:
    def test_predicted_non_negative(self, harv, ohlcv_df):
        result = harv.create_har_rv_features(ohlcv_df)
        assert np.all(result["harv_predicted"].values >= 0.0)

    def test_residual_z_clipped(self, harv, ohlcv_df):
        result = harv.create_har_rv_features(ohlcv_df)
        vals = result["harv_residual_z"].values
        assert np.all(vals >= -5.0) and np.all(vals <= 5.0)

    def test_component_ratio_bounded(self, harv, ohlcv_df):
        result = harv.create_har_rv_features(ohlcv_df)
        vals = result["harv_component_ratio"].values
        assert np.all(vals >= 0.0) and np.all(vals <= 10.0)


class TestAnalyze:
    def test_returns_dict(self, harv, ohlcv_df):
        result = harv.create_har_rv_features(ohlcv_df)
        analysis = harv.analyze_current_harv(result)
        assert isinstance(analysis, dict)

    def test_has_keys(self, harv, ohlcv_df):
        result = harv.create_har_rv_features(ohlcv_df)
        analysis = harv.analyze_current_harv(result)
        for key in ["vol_regime", "predicted_rv", "residual_z", "component_ratio"]:
            assert key in analysis

    def test_none_on_empty(self, harv):
        assert harv.analyze_current_harv(pd.DataFrame()) is None


class TestFeatureGroup:
    def test_group_exists(self):
        assert "har_rv" in FEATURE_GROUPS

    def test_group_prefix(self):
        assert FEATURE_GROUPS["har_rv"] == ["harv_"]
