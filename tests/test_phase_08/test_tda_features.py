"""Tests for TDAHomologyFeatures (tda_ prefix, 5 features)."""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.tda_features import TDAHomologyFeatures
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
def tda():
    return TDAHomologyFeatures()


class TestInvariants:
    def test_all_5_features_created(self, tda, ohlcv_df):
        result = tda.create_tda_features(ohlcv_df)
        cols = [c for c in result.columns if c.startswith("tda_")]
        assert len(cols) == 5

    def test_no_nan(self, tda, ohlcv_df):
        result = tda.create_tda_features(ohlcv_df)
        for col in [c for c in result.columns if c.startswith("tda_")]:
            assert not result[col].isna().any(), f"NaN in {col}"

    def test_no_inf(self, tda, ohlcv_df):
        result = tda.create_tda_features(ohlcv_df)
        for col in [c for c in result.columns if c.startswith("tda_")]:
            assert not np.isinf(result[col].values).any(), f"Inf in {col}"

    def test_preserves_rows(self, tda, ohlcv_df):
        result = tda.create_tda_features(ohlcv_df)
        assert len(result) == len(ohlcv_df)

    def test_short_data_no_crash(self, tda, short_df):
        result = tda.create_tda_features(short_df)
        assert len(result) == len(short_df)


class TestGraceful:
    def test_features_zero_when_gtda_unavailable(self, ohlcv_df):
        tda = TDAHomologyFeatures()
        result = tda.create_tda_features(ohlcv_df)
        if not tda._gtda_available:
            for col in [c for c in result.columns if c.startswith("tda_")]:
                assert (result[col] == 0.0).all(), f"{col} should be 0.0"

    def test_missing_close(self, tda):
        df = pd.DataFrame({"volume": [1e8] * 50})
        result = tda.create_tda_features(df)
        assert len(result) == 50


class TestAnalyze:
    def test_returns_dict(self, tda, ohlcv_df):
        result = tda.create_tda_features(ohlcv_df)
        analysis = tda.analyze_current_topology(result)
        assert isinstance(analysis, dict)

    def test_has_keys(self, tda, ohlcv_df):
        result = tda.create_tda_features(ohlcv_df)
        analysis = tda.analyze_current_topology(result)
        for key in ["topology", "h0_persistence", "h1_persistence", "gtda_available"]:
            assert key in analysis


class TestFeatureGroup:
    def test_group_exists(self):
        assert "tda_homology" in FEATURE_GROUPS

    def test_group_prefix(self):
        assert FEATURE_GROUPS["tda_homology"] == ["tda_"]
