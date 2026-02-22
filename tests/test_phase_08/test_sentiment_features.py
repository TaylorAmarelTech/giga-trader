"""
Tests for SentimentFeatures class.

Validates sentiment feature engineering from VIX, cross-asset flows,
and optional news APIs without requiring live API calls.
"""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.sentiment_features import SentimentFeatures


# ─── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_prices():
    """Create realistic mock price data for all sentiment sources."""
    np.random.seed(42)
    n_days = 500
    dates = pd.bdate_range("2022-01-03", periods=n_days)

    prices = pd.DataFrame(index=dates)

    # VIX: ~20 with mean reversion
    vix = 20.0 + np.cumsum(np.random.normal(0, 0.5, n_days))
    prices["^VIX"] = np.clip(vix, 10, 50)

    # SPY
    prices["SPY"] = 450 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, n_days)))

    # Bond ETFs
    prices["JNK"] = 95 * np.exp(np.cumsum(np.random.normal(0.0001, 0.005, n_days)))
    prices["SHY"] = 85 * np.exp(np.cumsum(np.random.normal(0.00005, 0.001, n_days)))

    # Safe haven
    prices["GLD"] = 180 * np.exp(np.cumsum(np.random.normal(0.0002, 0.005, n_days)))
    prices["TIP"] = 110 * np.exp(np.cumsum(np.random.normal(0.0001, 0.003, n_days)))

    # Risk-on
    prices["XLF"] = 35 * np.exp(np.cumsum(np.random.normal(0.0003, 0.008, n_days)))
    prices["USO"] = 60 * np.exp(np.cumsum(np.random.normal(0, 0.01, n_days)))

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
def sent_features(mock_prices):
    """Create a SentimentFeatures instance with pre-loaded prices."""
    sf = SentimentFeatures()
    sf._prices = mock_prices
    return sf


# ─── Constructor Tests ───────────────────────────────────────────────────────

class TestSentimentFeaturesInit:

    def test_required_sources(self):
        sf = SentimentFeatures()
        assert "^VIX" in sf.REQUIRED_SOURCES
        assert "SPY" in sf.REQUIRED_SOURCES
        assert len(sf.REQUIRED_SOURCES) == 8

    def test_prices_initially_none(self):
        sf = SentimentFeatures()
        assert sf._prices is None


# ─── Feature Engineering Tests ───────────────────────────────────────────────

class TestFeatureEngineering:

    def test_creates_features(self, sent_features, spy_daily):
        result = sent_features.create_sentiment_features(spy_daily)
        sent_cols = [c for c in result.columns if c.startswith("sent_")]
        assert len(sent_cols) > 0

    def test_vix_derived_features(self, sent_features, spy_daily):
        """VIX-derived features should be created when VIX and SPY are available."""
        result = sent_features.create_sentiment_features(spy_daily)
        expected = [
            "sent_fear_greed",
            "sent_fear_greed_z",
            "sent_vix_mean_revert",
            "sent_vix_acceleration",
            "sent_equity_put_call_proxy",
        ]
        for col in expected:
            assert col in result.columns, f"Missing VIX feature: {col}"

    def test_credit_risk_features(self, sent_features, spy_daily):
        """Credit risk features should be created when JNK and SHY are available."""
        result = sent_features.create_sentiment_features(spy_daily)
        assert "sent_risk_appetite" in result.columns
        assert "sent_risk_appetite_z" in result.columns

    def test_cross_asset_flow_features(self, sent_features, spy_daily):
        """Cross-asset flow features should be created."""
        result = sent_features.create_sentiment_features(spy_daily)
        expected = [
            "sent_flight_to_safety",
            "sent_safe_haven_flow",
            "sent_risk_on_flow",
            "sent_risk_rotation",
            "sent_risk_rotation_z",
        ]
        for col in expected:
            assert col in result.columns, f"Missing cross-asset feature: {col}"

    def test_total_feature_count(self, sent_features, spy_daily):
        """Should create exactly 12 features (no news API in test)."""
        result = sent_features.create_sentiment_features(spy_daily)
        sent_cols = [c for c in result.columns if c.startswith("sent_")]
        assert len(sent_cols) == 12

    def test_preserves_original_columns(self, sent_features, spy_daily):
        original_cols = set(spy_daily.columns)
        result = sent_features.create_sentiment_features(spy_daily)
        assert original_cols.issubset(set(result.columns))

    def test_same_row_count(self, sent_features, spy_daily):
        result = sent_features.create_sentiment_features(spy_daily)
        assert len(result) == len(spy_daily)

    def test_no_nans_in_sent_cols(self, sent_features, spy_daily):
        """Sentiment columns should be filled with 0 (not NaN)."""
        result = sent_features.create_sentiment_features(spy_daily)
        sent_cols = [c for c in result.columns if c.startswith("sent_")]
        nan_count = result[sent_cols].isna().sum().sum()
        assert nan_count == 0

    def test_empty_prices_returns_original(self, spy_daily):
        sf = SentimentFeatures()
        sf._prices = pd.DataFrame()
        result = sf.create_sentiment_features(spy_daily)
        assert list(result.columns) == list(spy_daily.columns)

    def test_none_prices_returns_original(self, spy_daily):
        sf = SentimentFeatures()
        result = sf.create_sentiment_features(spy_daily)
        assert list(result.columns) == list(spy_daily.columns)


# ─── Partial Data Tests ──────────────────────────────────────────────────────

class TestPartialData:

    def test_vix_only(self, spy_daily):
        """With only VIX + SPY, should still create 5 VIX-derived features."""
        np.random.seed(42)
        n = 500
        dates = pd.bdate_range("2022-01-03", periods=n)
        prices = pd.DataFrame({
            "^VIX": np.clip(20 + np.cumsum(np.random.normal(0, 0.5, n)), 10, 50),
            "SPY": 450 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, n))),
        }, index=dates)

        sf = SentimentFeatures()
        sf._prices = prices
        result = sf.create_sentiment_features(spy_daily)
        sent_cols = [c for c in result.columns if c.startswith("sent_")]
        assert len(sent_cols) == 5  # Only VIX-derived, no credit or cross-asset

    def test_no_vix_still_works(self, spy_daily):
        """Without VIX, should still create credit + cross-asset features."""
        np.random.seed(42)
        n = 500
        dates = pd.bdate_range("2022-01-03", periods=n)
        prices = pd.DataFrame({
            "SPY": 450 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, n))),
            "JNK": 95 * np.exp(np.cumsum(np.random.normal(0.0001, 0.005, n))),
            "SHY": 85 * np.exp(np.cumsum(np.random.normal(0.00005, 0.001, n))),
            "GLD": 180 * np.exp(np.cumsum(np.random.normal(0.0002, 0.005, n))),
            "TIP": 110 * np.exp(np.cumsum(np.random.normal(0.0001, 0.003, n))),
            "XLF": 35 * np.exp(np.cumsum(np.random.normal(0.0003, 0.008, n))),
            "USO": 60 * np.exp(np.cumsum(np.random.normal(0, 0.01, n))),
        }, index=dates)

        sf = SentimentFeatures()
        sf._prices = prices
        result = sf.create_sentiment_features(spy_daily)
        sent_cols = [c for c in result.columns if c.startswith("sent_")]
        # Credit (2) + cross-asset (5) = 7
        assert len(sent_cols) == 7


# ─── Condition Analysis Tests ────────────────────────────────────────────────

class TestConditionAnalysis:

    def test_returns_dict(self, sent_features, spy_daily):
        conditions = sent_features.analyze_current_sentiment(spy_daily)
        assert isinstance(conditions, dict)

    def test_fear_greed_ratio_present(self, sent_features, spy_daily):
        conditions = sent_features.analyze_current_sentiment(spy_daily)
        assert "fear_greed_ratio" in conditions

    def test_sentiment_regime_present(self, sent_features, spy_daily):
        conditions = sent_features.analyze_current_sentiment(spy_daily)
        assert "sentiment_regime" in conditions
        assert conditions["sentiment_regime"] in ("FEAR", "GREED", "NEUTRAL")

    def test_risk_rotation_present(self, sent_features, spy_daily):
        conditions = sent_features.analyze_current_sentiment(spy_daily)
        assert "risk_rotation" in conditions

    def test_risk_appetite_present(self, sent_features, spy_daily):
        conditions = sent_features.analyze_current_sentiment(spy_daily)
        assert "risk_appetite" in conditions
        assert conditions["risk_appetite"] in ("RISK_ON", "RISK_OFF", "NEUTRAL")

    def test_none_prices_returns_none(self, spy_daily):
        sf = SentimentFeatures()
        assert sf.analyze_current_sentiment(spy_daily) is None


# ─── Config + Pipeline Integration Tests ─────────────────────────────────────

class TestConfigIntegration:

    def test_config_has_flag(self):
        from src.experiment_config import AntiOverfitConfig
        config = AntiOverfitConfig()
        assert hasattr(config, "use_sentiment_features")
        assert config.use_sentiment_features is True

    def test_can_disable(self):
        from src.experiment_config import AntiOverfitConfig
        config = AntiOverfitConfig(use_sentiment_features=False)
        assert config.use_sentiment_features is False

    def test_integrate_anti_overfit_accepts_param(self):
        import inspect
        from src.phase_13_validation.anti_overfit_integration import integrate_anti_overfit
        sig = inspect.signature(integrate_anti_overfit)
        assert "use_sentiment_features" in sig.parameters

    def test_feature_group_mapping(self):
        """Sentiment features should map to 'sentiment' group."""
        from src.phase_10_feature_processing.group_aware_processor import (
            FEATURE_GROUPS,
            assign_feature_groups,
        )
        assert "sentiment" in FEATURE_GROUPS
        assert "sent_" in FEATURE_GROUPS["sentiment"]

        test_features = [
            "sent_fear_greed", "sent_fear_greed_z", "sent_vix_mean_revert",
            "sent_risk_appetite", "sent_risk_rotation", "other_feature",
        ]
        groups = assign_feature_groups(test_features)
        assert "sentiment" in groups
        assert len(groups["sentiment"]) == 5  # All sent_ features
        assert 5 in groups.get("other", [])   # "other_feature" index
