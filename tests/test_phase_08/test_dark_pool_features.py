"""
Tests for DarkPoolFeatures (FINRA short sale volume proxy).
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from src.phase_08_features_breadth.dark_pool_features import DarkPoolFeatures


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def dp_engine():
    return DarkPoolFeatures()


@pytest.fixture
def mock_dp_data():
    """Mock pre-processed dark pool data."""
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    np.random.seed(42)
    return pd.DataFrame({
        "date": dates,
        "short_volume_ratio": 0.45 + np.random.randn(120) * 0.05,
    })


@pytest.fixture
def spy_daily():
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    np.random.seed(42)
    return pd.DataFrame({
        "date": dates,
        "close": 450 + np.cumsum(np.random.randn(120) * 0.5),
        "volume": np.random.randint(50_000_000, 200_000_000, 120),
    })


# ─── Init Tests ──────────────────────────────────────────────────────────────

class TestDarkPoolInit:

    def test_default_constructor(self, dp_engine):
        assert isinstance(dp_engine, DarkPoolFeatures)
        assert dp_engine.data.empty


# ─── Feature Engineering Tests ───────────────────────────────────────────────

class TestDarkPoolFeatures:

    def test_creates_features(self, dp_engine, mock_dp_data, spy_daily):
        dp_engine.data = mock_dp_data
        result = dp_engine.create_dark_pool_features(spy_daily)
        dp_cols = [c for c in result.columns if c.startswith("dp_")]
        assert len(dp_cols) == 4

    def test_feature_names(self, dp_engine, mock_dp_data, spy_daily):
        dp_engine.data = mock_dp_data
        result = dp_engine.create_dark_pool_features(spy_daily)
        expected = [
            "dp_short_volume_ratio", "dp_short_ratio_zscore",
            "dp_short_ratio_chg_5d", "dp_short_ratio_extreme",
        ]
        for name in expected:
            assert name in result.columns, f"Missing: {name}"

    def test_preserves_original(self, dp_engine, mock_dp_data, spy_daily):
        dp_engine.data = mock_dp_data
        result = dp_engine.create_dark_pool_features(spy_daily)
        assert "close" in result.columns

    def test_same_row_count(self, dp_engine, mock_dp_data, spy_daily):
        dp_engine.data = mock_dp_data
        result = dp_engine.create_dark_pool_features(spy_daily)
        assert len(result) == len(spy_daily)

    def test_no_nans(self, dp_engine, mock_dp_data, spy_daily):
        dp_engine.data = mock_dp_data
        result = dp_engine.create_dark_pool_features(spy_daily)
        dp_cols = [c for c in result.columns if c.startswith("dp_")]
        for col in dp_cols:
            assert result[col].isna().sum() == 0, f"NaN in {col}"

    def test_zscore_clamped(self, dp_engine, mock_dp_data, spy_daily):
        dp_engine.data = mock_dp_data
        result = dp_engine.create_dark_pool_features(spy_daily)
        assert result["dp_short_ratio_zscore"].min() >= -3.0
        assert result["dp_short_ratio_zscore"].max() <= 3.0

    def test_extreme_binary(self, dp_engine, mock_dp_data, spy_daily):
        dp_engine.data = mock_dp_data
        result = dp_engine.create_dark_pool_features(spy_daily)
        vals = result["dp_short_ratio_extreme"].unique()
        assert set(vals).issubset({0.0, 1.0})

    def test_empty_data_returns_original(self, dp_engine, spy_daily):
        result = dp_engine.create_dark_pool_features(spy_daily)
        assert list(result.columns) == list(spy_daily.columns)

    def test_prefix_consistency(self, dp_engine, mock_dp_data, spy_daily):
        dp_engine.data = mock_dp_data
        result = dp_engine.create_dark_pool_features(spy_daily)
        new_cols = set(result.columns) - set(spy_daily.columns)
        for col in new_cols:
            assert col.startswith("dp_"), f"{col} missing dp_ prefix"


# ─── Condition Analysis Tests ────────────────────────────────────────────────

class TestDarkPoolAnalysis:

    def test_returns_dict(self, dp_engine, mock_dp_data, spy_daily):
        dp_engine.data = mock_dp_data
        result_df = dp_engine.create_dark_pool_features(spy_daily)
        analysis = dp_engine.analyze_current_dark_pool(result_df)
        assert isinstance(analysis, dict)

    def test_sentiment_present(self, dp_engine, mock_dp_data, spy_daily):
        dp_engine.data = mock_dp_data
        result_df = dp_engine.create_dark_pool_features(spy_daily)
        analysis = dp_engine.analyze_current_dark_pool(result_df)
        assert analysis["sentiment"] in ["bullish", "bearish", "neutral"]

    def test_none_when_no_data(self, dp_engine, spy_daily):
        analysis = dp_engine.analyze_current_dark_pool(spy_daily)
        assert analysis is None


# ─── Config Integration Tests ────────────────────────────────────────────────

class TestDarkPoolConfig:

    def test_config_has_flag(self):
        from src.experiment_config import AntiOverfitConfig
        config = AntiOverfitConfig()
        assert hasattr(config, "use_dark_pool")
        assert config.use_dark_pool is True

    def test_feature_group_registered(self):
        from src.phase_10_feature_processing.group_aware_processor import FEATURE_GROUPS
        assert "dark_pool" in FEATURE_GROUPS
        assert "dp_" in FEATURE_GROUPS["dark_pool"]
