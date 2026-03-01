"""Tests for LMomentsFeatures (lmom_ prefix, 4 features)."""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.l_moments_features import LMomentsFeatures
from src.phase_10_feature_processing.group_aware_processor import FEATURE_GROUPS


@pytest.fixture
def ohlcv_df():
    np.random.seed(42)
    n = 300
    close = 400.0 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({"close": close})


@pytest.fixture
def short_df():
    return pd.DataFrame({"close": [400.0 + i * 0.1 for i in range(10)]})


@pytest.fixture
def lmom():
    return LMomentsFeatures()


class TestInvariants:
    def test_all_4_features_created(self, lmom, ohlcv_df):
        result = lmom.create_l_moments_features(ohlcv_df)
        cols = [c for c in result.columns if c.startswith("lmom_")]
        assert len(cols) == 4

    def test_no_nan(self, lmom, ohlcv_df):
        result = lmom.create_l_moments_features(ohlcv_df)
        for col in [c for c in result.columns if c.startswith("lmom_")]:
            assert not result[col].isna().any(), f"NaN in {col}"

    def test_no_inf(self, lmom, ohlcv_df):
        result = lmom.create_l_moments_features(ohlcv_df)
        for col in [c for c in result.columns if c.startswith("lmom_")]:
            assert not np.isinf(result[col].values).any(), f"Inf in {col}"

    def test_preserves_rows(self, lmom, ohlcv_df):
        result = lmom.create_l_moments_features(ohlcv_df)
        assert len(result) == len(ohlcv_df)

    def test_short_data_no_crash(self, lmom, short_df):
        result = lmom.create_l_moments_features(short_df)
        assert len(result) == len(short_df)


class TestLMoments:
    def test_lskew_bounded(self, lmom, ohlcv_df):
        result = lmom.create_l_moments_features(ohlcv_df)
        vals = result["lmom_lskew_20d"].values
        assert np.all(vals >= -1.0) and np.all(vals <= 1.0)

    def test_lkurt_bounded(self, lmom, ohlcv_df):
        result = lmom.create_l_moments_features(ohlcv_df)
        vals = result["lmom_lkurt_20d"].values
        assert np.all(vals >= -1.0) and np.all(vals <= 1.0)

    def test_lcv_non_negative(self, lmom, ohlcv_df):
        result = lmom.create_l_moments_features(ohlcv_df)
        vals = result["lmom_lcv_20d"].values
        assert np.all(vals >= 0.0)

    def test_lskew_z_clipped(self, lmom, ohlcv_df):
        result = lmom.create_l_moments_features(ohlcv_df)
        vals = result["lmom_lskew_z"].values
        assert np.all(vals >= -5.0) and np.all(vals <= 5.0)


class TestAnalyze:
    def test_returns_dict(self, lmom, ohlcv_df):
        result = lmom.create_l_moments_features(ohlcv_df)
        analysis = lmom.analyze_current_lmoments(result)
        assert isinstance(analysis, dict)

    def test_has_keys(self, lmom, ohlcv_df):
        result = lmom.create_l_moments_features(ohlcv_df)
        analysis = lmom.analyze_current_lmoments(result)
        for key in ["distribution_regime", "l_skewness", "l_kurtosis"]:
            assert key in analysis


class TestFeatureGroup:
    def test_group_exists(self):
        assert "l_moments" in FEATURE_GROUPS

    def test_group_prefix(self):
        assert FEATURE_GROUPS["l_moments"] == ["lmom_"]
