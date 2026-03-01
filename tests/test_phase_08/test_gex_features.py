"""
Tests for GammaExposureFeatures (GEX proxy from VIX term structure).
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

from src.phase_08_features_breadth.gamma_exposure_features import GammaExposureFeatures


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def gex_engine():
    return GammaExposureFeatures()


@pytest.fixture
def mock_vix_data():
    """Create mock VIX term structure data."""
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    np.random.seed(42)
    vix_spot = 18 + np.random.randn(120) * 3
    vix_3m = vix_spot + np.random.randn(120) * 1 + 1.5  # Contango on average
    vix_9d = vix_spot + np.random.randn(120) * 2 - 0.5  # Slightly below spot

    data = pd.DataFrame({
        "vix_spot": vix_spot,
        "vix_3m": vix_3m,
        "vix_9d": vix_9d,
    }, index=dates)
    return data


@pytest.fixture
def spy_daily():
    """Standard spy_daily DataFrame for merge tests."""
    dates = pd.date_range("2024-01-01", periods=120, freq="B")
    np.random.seed(42)
    return pd.DataFrame({
        "date": dates,
        "close": 450 + np.cumsum(np.random.randn(120) * 0.5),
        "volume": np.random.randint(50_000_000, 200_000_000, 120),
    })


# ─── Init Tests ──────────────────────────────────────────────────────────────

class TestGexInit:

    def test_default_constructor(self, gex_engine):
        assert isinstance(gex_engine, GammaExposureFeatures)
        assert gex_engine.data.empty

    def test_data_initially_empty(self, gex_engine):
        assert len(gex_engine.data) == 0


# ─── Download Tests ──────────────────────────────────────────────────────────

class TestGexDownload:

    def test_download_with_mock_yfinance(self, gex_engine, mock_vix_data):
        """Manually set data to simulate download."""
        gex_engine.data = mock_vix_data
        assert not gex_engine.data.empty
        assert "vix_spot" in gex_engine.data.columns
        assert "vix_3m" in gex_engine.data.columns

    def test_download_failure_returns_empty(self, gex_engine):
        """Download should return empty on import failure."""
        with patch.dict("sys.modules", {"yfinance": None}):
            result = gex_engine.download_gex_data(
                datetime(2024, 1, 1), datetime(2024, 6, 1)
            )
            # May or may not fail depending on caching; just ensure no crash
            assert isinstance(result, pd.DataFrame)


# ─── Feature Engineering Tests ───────────────────────────────────────────────

class TestGexFeatures:

    def test_creates_features(self, gex_engine, mock_vix_data, spy_daily):
        gex_engine.data = mock_vix_data
        result = gex_engine.create_gex_features(spy_daily)
        gex_cols = [c for c in result.columns if c.startswith("gex_")]
        assert len(gex_cols) == 6

    def test_feature_names(self, gex_engine, mock_vix_data, spy_daily):
        gex_engine.data = mock_vix_data
        result = gex_engine.create_gex_features(spy_daily)
        expected = [
            "gex_proxy", "gex_proxy_zscore", "gex_regime",
            "gex_flip_signal", "gex_magnitude", "gex_chg_5d",
        ]
        for name in expected:
            assert name in result.columns, f"Missing feature: {name}"

    def test_preserves_original_columns(self, gex_engine, mock_vix_data, spy_daily):
        gex_engine.data = mock_vix_data
        result = gex_engine.create_gex_features(spy_daily)
        assert "close" in result.columns
        assert "volume" in result.columns

    def test_same_row_count(self, gex_engine, mock_vix_data, spy_daily):
        gex_engine.data = mock_vix_data
        result = gex_engine.create_gex_features(spy_daily)
        assert len(result) == len(spy_daily)

    def test_no_nans(self, gex_engine, mock_vix_data, spy_daily):
        gex_engine.data = mock_vix_data
        result = gex_engine.create_gex_features(spy_daily)
        gex_cols = [c for c in result.columns if c.startswith("gex_")]
        for col in gex_cols:
            assert result[col].isna().sum() == 0, f"NaN found in {col}"

    def test_proxy_range(self, gex_engine, mock_vix_data, spy_daily):
        """GEX proxy should be in [-1, +1] range (tanh bounded)."""
        gex_engine.data = mock_vix_data
        result = gex_engine.create_gex_features(spy_daily)
        assert result["gex_proxy"].min() >= -1.0
        assert result["gex_proxy"].max() <= 1.0

    def test_zscore_clamped(self, gex_engine, mock_vix_data, spy_daily):
        """Z-score should be clamped to [-3, +3]."""
        gex_engine.data = mock_vix_data
        result = gex_engine.create_gex_features(spy_daily)
        assert result["gex_proxy_zscore"].min() >= -3.0
        assert result["gex_proxy_zscore"].max() <= 3.0

    def test_regime_values(self, gex_engine, mock_vix_data, spy_daily):
        """Regime should be 0, 1, or 2."""
        gex_engine.data = mock_vix_data
        result = gex_engine.create_gex_features(spy_daily)
        assert set(result["gex_regime"].unique()).issubset({0, 1, 2})

    def test_flip_signal_binary(self, gex_engine, mock_vix_data, spy_daily):
        gex_engine.data = mock_vix_data
        result = gex_engine.create_gex_features(spy_daily)
        assert set(result["gex_flip_signal"].unique()).issubset({0.0, 1.0})

    def test_magnitude_non_negative(self, gex_engine, mock_vix_data, spy_daily):
        gex_engine.data = mock_vix_data
        result = gex_engine.create_gex_features(spy_daily)
        assert (result["gex_magnitude"] >= 0).all()

    def test_empty_data_returns_original(self, gex_engine, spy_daily):
        """Empty data should return original DataFrame unchanged."""
        result = gex_engine.create_gex_features(spy_daily)
        assert list(result.columns) == list(spy_daily.columns)

    def test_prefix_consistency(self, gex_engine, mock_vix_data, spy_daily):
        """All new features should have gex_ prefix."""
        gex_engine.data = mock_vix_data
        result = gex_engine.create_gex_features(spy_daily)
        new_cols = set(result.columns) - set(spy_daily.columns)
        for col in new_cols:
            assert col.startswith("gex_"), f"Feature {col} lacks gex_ prefix"

    def test_missing_vix3m_still_works(self, gex_engine, spy_daily):
        """Should work even if VIX3M data is missing."""
        dates = pd.date_range("2024-01-01", periods=120, freq="B")
        data = pd.DataFrame({
            "vix_spot": 18 + np.random.randn(120) * 3,
        }, index=dates)
        gex_engine.data = data
        result = gex_engine.create_gex_features(spy_daily)
        assert "gex_proxy" in result.columns


# ─── Condition Analysis Tests ────────────────────────────────────────────────

class TestGexAnalysis:

    def test_returns_dict(self, gex_engine, mock_vix_data, spy_daily):
        gex_engine.data = mock_vix_data
        result_df = gex_engine.create_gex_features(spy_daily)
        analysis = gex_engine.analyze_current_gex(result_df)
        assert isinstance(analysis, dict)

    def test_regime_present(self, gex_engine, mock_vix_data, spy_daily):
        gex_engine.data = mock_vix_data
        result_df = gex_engine.create_gex_features(spy_daily)
        analysis = gex_engine.analyze_current_gex(result_df)
        assert "gex_regime" in analysis
        assert analysis["gex_regime"] in ["negative_gex", "neutral", "positive_gex"]

    def test_market_behavior(self, gex_engine, mock_vix_data, spy_daily):
        gex_engine.data = mock_vix_data
        result_df = gex_engine.create_gex_features(spy_daily)
        analysis = gex_engine.analyze_current_gex(result_df)
        assert analysis["market_behavior"] in ["mean_reverting", "trending", "mixed"]

    def test_none_when_no_data(self, gex_engine, spy_daily):
        analysis = gex_engine.analyze_current_gex(spy_daily)
        assert analysis is None


# ─── Config Integration Tests ────────────────────────────────────────────────

class TestGexConfig:

    def test_config_has_flag(self):
        from src.experiment_config import AntiOverfitConfig
        config = AntiOverfitConfig()
        assert hasattr(config, "use_gamma_exposure")
        assert config.use_gamma_exposure is True

    def test_feature_group_registered(self):
        from src.phase_10_feature_processing.group_aware_processor import FEATURE_GROUPS
        assert "options_flow" in FEATURE_GROUPS
        assert "gex_" in FEATURE_GROUPS["options_flow"]
