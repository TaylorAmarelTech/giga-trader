"""Tests for MultiscaleEntropyFeatures (mse_ prefix, 3 features)."""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.multiscale_entropy_features import MultiscaleEntropyFeatures
from src.phase_10_feature_processing.group_aware_processor import FEATURE_GROUPS


@pytest.fixture
def ohlcv_df():
    np.random.seed(42)
    n = 300
    close = 400.0 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({"close": close})


@pytest.fixture
def short_df():
    return pd.DataFrame({"close": [400.0 + i * 0.1 for i in range(20)]})


@pytest.fixture
def mse():
    return MultiscaleEntropyFeatures()


class TestInvariants:
    def test_all_3_features_created(self, mse, ohlcv_df):
        result = mse.create_multiscale_entropy_features(ohlcv_df)
        cols = [c for c in result.columns if c.startswith("mse_")]
        assert len(cols) == 3

    def test_no_nan(self, mse, ohlcv_df):
        result = mse.create_multiscale_entropy_features(ohlcv_df)
        for col in [c for c in result.columns if c.startswith("mse_")]:
            assert not result[col].isna().any(), f"NaN in {col}"

    def test_no_inf(self, mse, ohlcv_df):
        result = mse.create_multiscale_entropy_features(ohlcv_df)
        for col in [c for c in result.columns if c.startswith("mse_")]:
            assert not np.isinf(result[col].values).any(), f"Inf in {col}"

    def test_preserves_rows(self, mse, ohlcv_df):
        result = mse.create_multiscale_entropy_features(ohlcv_df)
        assert len(result) == len(ohlcv_df)

    def test_short_data_no_crash(self, mse, short_df):
        result = mse.create_multiscale_entropy_features(short_df)
        assert len(result) == len(short_df)


class TestMSE:
    def test_slope_clipped(self, mse, ohlcv_df):
        result = mse.create_multiscale_entropy_features(ohlcv_df)
        vals = result["mse_slope"].values
        assert np.all(vals >= -5.0) and np.all(vals <= 5.0)

    def test_area_non_negative(self, mse, ohlcv_df):
        result = mse.create_multiscale_entropy_features(ohlcv_df)
        vals = result["mse_area"].values
        assert np.all(vals >= 0.0)

    def test_complexity_bounded(self, mse, ohlcv_df):
        result = mse.create_multiscale_entropy_features(ohlcv_df)
        vals = result["mse_complexity_index"].values
        assert np.all(vals >= 0.0) and np.all(vals <= 5.0)


class TestAnalyze:
    def test_returns_dict(self, mse, ohlcv_df):
        result = mse.create_multiscale_entropy_features(ohlcv_df)
        analysis = mse.analyze_current_entropy(result)
        assert isinstance(analysis, dict)

    def test_has_keys(self, mse, ohlcv_df):
        result = mse.create_multiscale_entropy_features(ohlcv_df)
        analysis = mse.analyze_current_entropy(result)
        for key in ["entropy_regime", "mse_slope", "complexity"]:
            assert key in analysis


class TestFeatureGroup:
    def test_group_exists(self):
        assert "multiscale_entropy" in FEATURE_GROUPS

    def test_group_prefix(self):
        assert FEATURE_GROUPS["multiscale_entropy"] == ["mse_"]
