"""
Tests for EconomicFeatures class.

Validates feature engineering from economic indicator data
(yields, VIX, credit spreads) without requiring live API calls.
"""

import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock

from src.phase_08_features_breadth.economic_features import EconomicFeatures


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_prices():
    """Create realistic mock price data for all sources."""
    np.random.seed(42)
    n_days = 500
    dates = pd.bdate_range("2022-01-03", periods=n_days)

    prices = pd.DataFrame(index=dates)

    # VIX: ~20 with mean reversion
    vix = 20.0 + np.cumsum(np.random.normal(0, 0.5, n_days))
    vix = np.clip(vix, 10, 50)
    prices["^VIX"] = vix

    # Treasury yields: levels around 3-5%
    prices["^TNX"] = 4.0 + np.cumsum(np.random.normal(0, 0.02, n_days))
    prices["^TYX"] = 4.5 + np.cumsum(np.random.normal(0, 0.02, n_days))
    prices["^FVX"] = 3.8 + np.cumsum(np.random.normal(0, 0.02, n_days))
    prices["^IRX"] = 2.5 + np.cumsum(np.random.normal(0, 0.01, n_days))

    # Bond ETFs: start at 100, random walk
    for etf in ["SHY", "LQD", "JNK", "TIP", "AGG"]:
        prices[etf] = 100 * np.exp(np.cumsum(np.random.normal(0.0001, 0.003, n_days)))

    # Volatility term structure
    prices["^VXV"] = 22.0 + np.cumsum(np.random.normal(0, 0.4, n_days))
    prices["^VXV"] = np.clip(prices["^VXV"], 12, 55)

    # Other ETFs
    prices["USO"] = 60 * np.exp(np.cumsum(np.random.normal(0, 0.01, n_days)))
    prices["DBC"] = 20 * np.exp(np.cumsum(np.random.normal(0, 0.007, n_days)))
    prices["GLD"] = 180 * np.exp(np.cumsum(np.random.normal(0.0002, 0.005, n_days)))
    prices["XLF"] = 35 * np.exp(np.cumsum(np.random.normal(0.0003, 0.008, n_days)))

    return prices


@pytest.fixture
def spy_daily(mock_prices):
    """Create a mock SPY daily DataFrame aligned with mock prices."""
    dates = mock_prices.index
    n = len(dates)
    np.random.seed(123)
    df = pd.DataFrame({
        "date": pd.to_datetime(dates.date),
        "day_return": np.random.normal(0.0004, 0.01, n),
        "close": 450 * np.exp(np.cumsum(np.random.normal(0.0004, 0.01, n))),
    })
    return df


@pytest.fixture
def econ_features(mock_prices):
    """Create an EconomicFeatures instance with pre-loaded prices."""
    ef = EconomicFeatures()
    ef._prices = mock_prices
    return ef


# ─── Constructor Tests ───────────────────────────────────────────────────────

class TestEconomicFeaturesInit:

    def test_default_sources(self):
        ef = EconomicFeatures()
        assert len(ef.sources) == 15

    def test_custom_sources(self):
        ef = EconomicFeatures(sources=["^VIX", "^TNX"])
        assert len(ef.sources) == 2
        assert "^VIX" in ef.sources
        assert "^TNX" in ef.sources

    def test_unknown_source_ignored(self):
        ef = EconomicFeatures(sources=["^VIX", "FAKE123"])
        assert len(ef.sources) == 1


# ─── Feature Engineering Tests ───────────────────────────────────────────────

class TestFeatureEngineering:

    def test_creates_features(self, econ_features, spy_daily):
        result = econ_features.create_economic_features(spy_daily)
        econ_cols = [c for c in result.columns if c.startswith("econ_")]
        assert len(econ_cols) > 0

    def test_feature_count_per_source(self, econ_features, spy_daily):
        """Each source should produce 6 base features."""
        result = econ_features.create_economic_features(spy_daily)
        # Check VIX features as example
        vix_cols = [c for c in result.columns if c.startswith("econ_vix_")]
        # 6 base + 1 derived (vix_regime) = 7
        assert len(vix_cols) >= 6

    def test_preserves_original_columns(self, econ_features, spy_daily):
        original_cols = set(spy_daily.columns)
        result = econ_features.create_economic_features(spy_daily)
        assert original_cols.issubset(set(result.columns))

    def test_same_row_count(self, econ_features, spy_daily):
        result = econ_features.create_economic_features(spy_daily)
        assert len(result) == len(spy_daily)

    def test_no_nans_in_econ_cols(self, econ_features, spy_daily):
        """Economic columns should be filled with 0 (not NaN)."""
        result = econ_features.create_economic_features(spy_daily)
        econ_cols = [c for c in result.columns if c.startswith("econ_")]
        nan_count = result[econ_cols].isna().sum().sum()
        assert nan_count == 0

    def test_feature_prefixes(self, econ_features, spy_daily):
        result = econ_features.create_economic_features(spy_daily)
        econ_cols = [c for c in result.columns if c.startswith("econ_")]
        # Should have features from multiple sources
        prefixes = set(c.split("_")[1] for c in econ_cols)
        assert len(prefixes) >= 5  # At least 5 different source prefixes

    def test_empty_prices_returns_original(self, spy_daily):
        ef = EconomicFeatures()
        ef._prices = pd.DataFrame()
        result = ef.create_economic_features(spy_daily)
        assert list(result.columns) == list(spy_daily.columns)

    def test_none_prices_returns_original(self, spy_daily):
        ef = EconomicFeatures()
        result = ef.create_economic_features(spy_daily)
        assert list(result.columns) == list(spy_daily.columns)


# ─── Derived Features Tests ─────────────────────────────────────────────────

class TestDerivedFeatures:

    def test_yield_curve_created(self, econ_features, spy_daily):
        result = econ_features.create_economic_features(spy_daily)
        assert "econ_yield_curve_10_5" in result.columns

    def test_yield_curve_steep_created(self, econ_features, spy_daily):
        result = econ_features.create_economic_features(spy_daily)
        assert "econ_yield_curve_10_13w" in result.columns

    def test_credit_spread_created(self, econ_features, spy_daily):
        result = econ_features.create_economic_features(spy_daily)
        assert "econ_credit_spread" in result.columns

    def test_real_yield_proxy_created(self, econ_features, spy_daily):
        result = econ_features.create_economic_features(spy_daily)
        assert "econ_real_yield_proxy" in result.columns

    def test_vix_regime_created(self, econ_features, spy_daily):
        result = econ_features.create_economic_features(spy_daily)
        assert "econ_vix_regime" in result.columns

    def test_oil_fin_divergence_created(self, econ_features, spy_daily):
        result = econ_features.create_economic_features(spy_daily)
        assert "econ_oil_fin_diverge" in result.columns

    def test_vix_term_structure_created(self, econ_features, spy_daily):
        result = econ_features.create_economic_features(spy_daily)
        assert "econ_vix_term_ratio" in result.columns
        assert "econ_vix_term_zscore" in result.columns

    def test_gold_equity_signal_created(self, econ_features, spy_daily):
        result = econ_features.create_economic_features(spy_daily)
        assert "econ_gold_equity_signal" in result.columns

    def test_bond_equity_rotation_created(self, econ_features, spy_daily):
        result = econ_features.create_economic_features(spy_daily)
        assert "econ_bond_equity_rotation" in result.columns

    def test_commodity_breadth_created(self, econ_features, spy_daily):
        result = econ_features.create_economic_features(spy_daily)
        assert "econ_commodity_breadth" in result.columns


# ─── Condition Analysis Tests ────────────────────────────────────────────────

class TestConditionAnalysis:

    def test_returns_dict(self, econ_features, spy_daily):
        signal = econ_features.analyze_current_conditions(spy_daily)
        assert isinstance(signal, dict)

    def test_vix_regime_present(self, econ_features, spy_daily):
        signal = econ_features.analyze_current_conditions(spy_daily)
        assert "vix_regime" in signal
        assert signal["vix_regime"] in ("HIGH_VOL", "LOW_VOL", "NORMAL")

    def test_yield_curve_signal_present(self, econ_features, spy_daily):
        signal = econ_features.analyze_current_conditions(spy_daily)
        assert "yield_curve_signal" in signal
        assert signal["yield_curve_signal"] in ("INVERTED", "NORMAL")

    def test_credit_signal_present(self, econ_features, spy_daily):
        signal = econ_features.analyze_current_conditions(spy_daily)
        assert "credit_signal" in signal
        assert signal["credit_signal"] in ("RISK_ON", "RISK_OFF")

    def test_vix_term_structure_present(self, econ_features, spy_daily):
        signal = econ_features.analyze_current_conditions(spy_daily)
        assert "vix_term_structure" in signal
        assert signal["vix_term_structure"] in ("BACKWARDATION", "CONTANGO", "STEEP_CONTANGO")

    def test_none_prices_returns_none(self, spy_daily):
        ef = EconomicFeatures()
        assert ef.analyze_current_conditions(spy_daily) is None


# ─── Config Integration Tests ───────────────────────────────────────────────

class TestConfigIntegration:

    def test_config_has_flag(self):
        from src.experiment_config import AntiOverfitConfig
        config = AntiOverfitConfig()
        assert hasattr(config, "use_economic_features")
        assert config.use_economic_features is True

    def test_can_disable(self):
        from src.experiment_config import AntiOverfitConfig
        config = AntiOverfitConfig(use_economic_features=False)
        assert config.use_economic_features is False
