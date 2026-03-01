"""Tests for RVSignaturePlotFeatures (rvsp_ prefix, 3 features)."""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.rv_signature_features import RVSignaturePlotFeatures
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
def rvsp():
    return RVSignaturePlotFeatures()


class TestInvariants:
    def test_all_3_features_created(self, rvsp, ohlcv_df):
        result = rvsp.create_rv_signature_features(ohlcv_df)
        cols = [c for c in result.columns if c.startswith("rvsp_")]
        assert len(cols) == 3

    def test_no_nan(self, rvsp, ohlcv_df):
        result = rvsp.create_rv_signature_features(ohlcv_df)
        for col in [c for c in result.columns if c.startswith("rvsp_")]:
            assert not result[col].isna().any(), f"NaN in {col}"

    def test_no_inf(self, rvsp, ohlcv_df):
        result = rvsp.create_rv_signature_features(ohlcv_df)
        for col in [c for c in result.columns if c.startswith("rvsp_")]:
            assert not np.isinf(result[col].values).any(), f"Inf in {col}"

    def test_preserves_rows(self, rvsp, ohlcv_df):
        result = rvsp.create_rv_signature_features(ohlcv_df)
        assert len(result) == len(ohlcv_df)

    def test_short_data_no_crash(self, rvsp, short_df):
        result = rvsp.create_rv_signature_features(short_df)
        assert len(result) == len(short_df)


class TestRVSP:
    def test_slope_clipped(self, rvsp, ohlcv_df):
        result = rvsp.create_rv_signature_features(ohlcv_df)
        vals = result["rvsp_slope"].values
        assert np.all(vals >= -5.0) and np.all(vals <= 5.0)

    def test_noise_ratio_bounded(self, rvsp, ohlcv_df):
        result = rvsp.create_rv_signature_features(ohlcv_df)
        vals = result["rvsp_noise_ratio"].values
        assert np.all(vals >= 0.0) and np.all(vals <= 10.0)

    def test_flatness_bounded(self, rvsp, ohlcv_df):
        result = rvsp.create_rv_signature_features(ohlcv_df)
        vals = result["rvsp_flatness"].values
        assert np.all(vals >= 0.0) and np.all(vals <= 1.0)


class TestFeatureGroup:
    def test_group_exists(self):
        assert "rv_signature" in FEATURE_GROUPS

    def test_group_prefix(self):
        assert FEATURE_GROUPS["rv_signature"] == ["rvsp_"]
