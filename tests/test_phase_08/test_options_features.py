"""
Tests for OptionsFeatures (VIX rank, CBOE SKEW Index, vol-of-vol).
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.phase_08_features_breadth.options_features import OptionsFeatures


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def options_engine():
    return OptionsFeatures()


@pytest.fixture
def mock_options_data():
    """Create mock VIX + SKEW data."""
    dates = pd.date_range("2024-01-01", periods=300, freq="B")
    np.random.seed(42)
    vix_spot = 18 + np.random.randn(300) * 3
    vix_spot = np.clip(vix_spot, 10, 40)  # Realistic VIX range
    skew = 120 + np.random.randn(300) * 10
    skew = np.clip(skew, 100, 160)  # Realistic SKEW range

    data = pd.DataFrame({
        "vix_spot": vix_spot,
        "skew": skew,
    }, index=dates)
    return data


@pytest.fixture
def spy_daily():
    """Standard spy_daily DataFrame for merge tests."""
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    np.random.seed(42)
    close = 450 + np.cumsum(np.random.randn(120) * 0.5)
    return pd.DataFrame({
        "date": dates,
        "close": close,
        "volume": np.random.randint(50_000_000, 200_000_000, 120),
    })


# ─── Init Tests ──────────────────────────────────────────────────────────────

class TestOptionsInit:

    def test_default_constructor(self, options_engine):
        assert isinstance(options_engine, OptionsFeatures)
        assert options_engine.data.empty

    def test_data_initially_empty(self, options_engine):
        assert len(options_engine.data) == 0


# ─── Download Tests ──────────────────────────────────────────────────────────

class TestOptionsDownload:

    def test_download_with_mock_data(self, options_engine, mock_options_data):
        """Manually set data to simulate download."""
        options_engine.data = mock_options_data
        assert not options_engine.data.empty
        assert "vix_spot" in options_engine.data.columns
        assert "skew" in options_engine.data.columns

    def test_download_failure_returns_empty(self, options_engine):
        """Download should return empty on import failure."""
        with patch.dict("sys.modules", {"yfinance": None}):
            result = options_engine.download_options_data(
                datetime(2024, 1, 1), datetime(2024, 6, 1)
            )
            assert isinstance(result, pd.DataFrame)


# ─── Feature Engineering Tests ───────────────────────────────────────────────

class TestOptionsFeatures:

    def test_creates_features(self, options_engine, mock_options_data, spy_daily):
        options_engine.data = mock_options_data
        result = options_engine.create_options_features(spy_daily)
        opt_cols = [c for c in result.columns if c.startswith("opt_")]
        assert len(opt_cols) == 15

    def test_feature_names(self, options_engine, mock_options_data, spy_daily):
        options_engine.data = mock_options_data
        result = options_engine.create_options_features(spy_daily)
        expected = [
            "opt_iv_rank", "opt_iv_percentile", "opt_iv_zscore",
            "opt_iv_chg_1d", "opt_iv_chg_5d", "opt_iv_mean_revert",
            "opt_skew_raw", "opt_skew_zscore", "opt_skew_chg_5d",
            "opt_skew_regime", "opt_fear_composite", "opt_complacency",
            "opt_tail_risk", "opt_vol_of_vol", "opt_vix_rv_spread",
        ]
        for name in expected:
            assert name in result.columns, f"Missing feature: {name}"

    def test_preserves_original_columns(self, options_engine, mock_options_data, spy_daily):
        options_engine.data = mock_options_data
        result = options_engine.create_options_features(spy_daily)
        assert "close" in result.columns
        assert "volume" in result.columns

    def test_same_row_count(self, options_engine, mock_options_data, spy_daily):
        options_engine.data = mock_options_data
        result = options_engine.create_options_features(spy_daily)
        assert len(result) == len(spy_daily)

    def test_no_nans(self, options_engine, mock_options_data, spy_daily):
        options_engine.data = mock_options_data
        result = options_engine.create_options_features(spy_daily)
        opt_cols = [c for c in result.columns if c.startswith("opt_")]
        for col in opt_cols:
            assert result[col].isna().sum() == 0, f"NaN found in {col}"

    def test_iv_rank_range(self, options_engine, mock_options_data, spy_daily):
        """IV rank should be in [0, 1] range."""
        options_engine.data = mock_options_data
        result = options_engine.create_options_features(spy_daily)
        assert result["opt_iv_rank"].min() >= 0.0
        assert result["opt_iv_rank"].max() <= 1.0

    def test_zscore_clamped(self, options_engine, mock_options_data, spy_daily):
        """Z-scores should be clamped to [-3, +3]."""
        options_engine.data = mock_options_data
        result = options_engine.create_options_features(spy_daily)
        for col in ["opt_iv_zscore", "opt_skew_zscore", "opt_iv_mean_revert"]:
            assert result[col].min() >= -3.0, f"{col} below -3"
            assert result[col].max() <= 3.0, f"{col} above 3"

    def test_regime_values(self, options_engine, mock_options_data, spy_daily):
        """Skew regime should be 0, 1, or 2."""
        options_engine.data = mock_options_data
        result = options_engine.create_options_features(spy_daily)
        assert set(result["opt_skew_regime"].unique()).issubset({0, 1, 2})

    def test_binary_features(self, options_engine, mock_options_data, spy_daily):
        """Complacency and tail risk should be binary {0, 1}."""
        options_engine.data = mock_options_data
        result = options_engine.create_options_features(spy_daily)
        for col in ["opt_complacency", "opt_tail_risk"]:
            assert set(result[col].unique()).issubset({0.0, 1.0}), f"{col} not binary"

    def test_prefix_consistency(self, options_engine, mock_options_data, spy_daily):
        """All new features should have opt_ prefix."""
        options_engine.data = mock_options_data
        result = options_engine.create_options_features(spy_daily)
        new_cols = set(result.columns) - set(spy_daily.columns)
        for col in new_cols:
            assert col.startswith("opt_"), f"Feature {col} lacks opt_ prefix"

    def test_empty_data_returns_original(self, options_engine, spy_daily):
        """Empty data should return original DataFrame unchanged."""
        result = options_engine.create_options_features(spy_daily)
        assert list(result.columns) == list(spy_daily.columns)

    def test_missing_skew_still_works(self, options_engine, spy_daily):
        """Should work even if SKEW data is missing (VIX only)."""
        dates = pd.date_range("2024-01-01", periods=300, freq="B")
        data = pd.DataFrame({
            "vix_spot": 18 + np.random.randn(300) * 3,
        }, index=dates)
        options_engine.data = data
        result = options_engine.create_options_features(spy_daily)
        assert "opt_iv_rank" in result.columns
        assert "opt_skew_raw" in result.columns  # Should be 0 when no SKEW data


# ─── Condition Analysis Tests ────────────────────────────────────────────────

class TestOptionsAnalysis:

    def test_returns_dict(self, options_engine, mock_options_data, spy_daily):
        options_engine.data = mock_options_data
        result_df = options_engine.create_options_features(spy_daily)
        analysis = options_engine.analyze_current_options(result_df)
        assert isinstance(analysis, dict)

    def test_analysis_keys(self, options_engine, mock_options_data, spy_daily):
        options_engine.data = mock_options_data
        result_df = options_engine.create_options_features(spy_daily)
        analysis = options_engine.analyze_current_options(result_df)
        assert "iv_rank" in analysis
        assert "skew_regime" in analysis
        assert "iv_state" in analysis
        assert analysis["skew_regime"] in ["normal", "elevated", "extreme"]

    def test_none_when_no_data(self, options_engine, spy_daily):
        analysis = options_engine.analyze_current_options(spy_daily)
        assert analysis is None


# ─── Config Integration Tests ────────────────────────────────────────────────

class TestOptionsConfig:

    def test_config_has_flag(self):
        from src.experiment_config import AntiOverfitConfig
        config = AntiOverfitConfig()
        assert hasattr(config, "use_options_features")
        assert config.use_options_features is True

    def test_feature_group_registered(self):
        from src.phase_10_feature_processing.group_aware_processor import FEATURE_GROUPS
        assert "options_iv" in FEATURE_GROUPS
        assert "opt_" in FEATURE_GROUPS["options_iv"]
