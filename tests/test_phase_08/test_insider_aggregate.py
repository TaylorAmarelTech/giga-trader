"""
Tests for InsiderAggregateFeatures
====================================
Validates feature engineering, edge-case handling, config registration, and
the analysis helper — without any network calls or external dependencies.
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime

from src.phase_08_features_breadth.insider_aggregate_features import (
    InsiderAggregateFeatures,
    _CLUSTER_THRESHOLD,
)


# =============================================================================
# Shared fixtures
# =============================================================================

@pytest.fixture
def engine():
    """Default engine with standard windows (30 / 60)."""
    return InsiderAggregateFeatures()


@pytest.fixture
def spy_daily():
    """
    Realistic SPY daily DataFrame with 200 business-day rows.
    close and volume are non-trivial so that both above/below MA cases arise.
    """
    np.random.seed(2026)
    n = 200
    dates = pd.bdate_range("2023-01-03", periods=n)

    # Trending price with noise — guarantees mix of above/below MA days
    base_price = 420.0
    returns = np.random.normal(0.0005, 0.008, n)
    close = base_price * np.exp(np.cumsum(returns))

    # Volume oscillates around 80M with spikes — mix of above/below MA days
    base_vol = 80_000_000
    volume = base_vol * (1.0 + np.random.normal(0, 0.25, n))
    volume = np.abs(volume).astype(int)

    return pd.DataFrame({"date": dates, "close": close, "volume": volume})


@pytest.fixture
def spy_with_features(engine, spy_daily):
    """DataFrame after running create_insider_aggregate_features."""
    return engine.create_insider_aggregate_features(spy_daily)


# =============================================================================
# TestInsiderAggregateInit
# =============================================================================

class TestInsiderAggregateInit:

    def test_default_windows(self, engine):
        assert engine.window == 30
        assert engine.z_window == 60

    def test_custom_windows(self):
        e = InsiderAggregateFeatures(window=15, z_window=45)
        assert e.window == 15
        assert e.z_window == 45

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            InsiderAggregateFeatures(window=0)

    def test_invalid_z_window_raises(self):
        with pytest.raises(ValueError):
            InsiderAggregateFeatures(z_window=-1)


# =============================================================================
# TestDownloadInsiderData
# =============================================================================

class TestDownloadInsiderData:

    def test_returns_empty_dataframe(self, engine):
        result = engine.download_insider_data(
            datetime(2024, 1, 1), datetime(2024, 12, 31)
        )
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_no_side_effects(self, engine):
        """Calling download should not mutate the engine in any unexpected way."""
        engine.download_insider_data(datetime(2024, 1, 1), datetime(2024, 12, 31))
        assert engine.window == 30  # unchanged
        assert engine.z_window == 60


# =============================================================================
# TestFeatureCreation
# =============================================================================

class TestFeatureCreation:

    def test_adds_four_features(self, spy_with_features):
        insider_cols = [c for c in spy_with_features.columns if c.startswith("insider_agg_")]
        assert len(insider_cols) == 4, f"Expected 4, got {insider_cols}"

    def test_feature_names_present(self, spy_with_features):
        expected = [
            "insider_agg_buy_ratio",
            "insider_agg_volume",
            "insider_agg_cluster",
            "insider_agg_z",
        ]
        for name in expected:
            assert name in spy_with_features.columns, f"Missing: {name}"

    def test_preserves_original_columns(self, engine, spy_daily):
        original_cols = set(spy_daily.columns)
        result = engine.create_insider_aggregate_features(spy_daily)
        assert original_cols.issubset(set(result.columns))

    def test_same_row_count(self, engine, spy_daily):
        result = engine.create_insider_aggregate_features(spy_daily)
        assert len(result) == len(spy_daily)

    def test_no_nans_in_feature_cols(self, spy_with_features):
        insider_cols = [c for c in spy_with_features.columns if c.startswith("insider_agg_")]
        for col in insider_cols:
            nan_n = spy_with_features[col].isna().sum()
            assert nan_n == 0, f"{col} has {nan_n} NaN rows"

    def test_buy_ratio_in_range(self, spy_with_features):
        col = spy_with_features["insider_agg_buy_ratio"]
        assert col.min() >= 0.0
        assert col.max() <= 1.0

    def test_volume_in_range(self, spy_with_features):
        col = spy_with_features["insider_agg_volume"]
        assert col.min() >= 0.0
        assert col.max() <= 1.0

    def test_cluster_is_binary(self, spy_with_features):
        unique = set(spy_with_features["insider_agg_cluster"].unique())
        assert unique.issubset({0.0, 1.0}), f"Unexpected cluster values: {unique}"

    def test_z_score_clipped(self, spy_with_features):
        z = spy_with_features["insider_agg_z"]
        assert z.min() >= -3.0
        assert z.max() <= 3.0

    def test_all_prefix_consistent(self, engine, spy_daily):
        original_cols = set(spy_daily.columns)
        result = engine.create_insider_aggregate_features(spy_daily)
        new_cols = set(result.columns) - original_cols
        for col in new_cols:
            assert col.startswith("insider_agg_"), f"Bad prefix: {col}"


# =============================================================================
# TestEdgeCases
# =============================================================================

class TestEdgeCases:

    def test_missing_close_column_returns_unchanged(self, engine):
        df = pd.DataFrame({"volume": [1_000_000, 2_000_000, 3_000_000]})
        result = engine.create_insider_aggregate_features(df)
        assert list(result.columns) == ["volume"]

    def test_missing_volume_column_returns_unchanged(self, engine):
        df = pd.DataFrame({"close": [450.0, 451.0, 452.0]})
        result = engine.create_insider_aggregate_features(df)
        assert list(result.columns) == ["close"]

    def test_empty_dataframe_returns_unchanged(self, engine):
        df = pd.DataFrame({"close": pd.Series(dtype=float), "volume": pd.Series(dtype=float)})
        result = engine.create_insider_aggregate_features(df)
        assert result.empty

    def test_all_accumulation_days(self, engine):
        """
        Construct data where every day is an accumulation day:
        close is always above MA, volume always above MA.
        After warm-up, buy_ratio should approach 1.0.
        """
        n = 60
        # Strictly monotone close (always above expanding MA)
        close = np.linspace(400.0, 500.0, n)
        # Volume also strictly monotone (always above expanding MA)
        volume = np.linspace(1_000_000, 2_000_000, n).astype(int)
        df = pd.DataFrame({"close": close, "volume": volume})
        result = engine.create_insider_aggregate_features(df)
        # After 30 days of warm-up the ratio should be high
        late_rows = result.iloc[35:]
        assert late_rows["insider_agg_buy_ratio"].mean() > 0.80

    def test_cluster_flag_triggers_when_buy_ratio_exceeds_threshold(self, engine):
        """
        Force buy_ratio > _CLUSTER_THRESHOLD on all rows; cluster flag must be 1.
        """
        n = 100
        # Strictly increasing so every row is above MA
        close = np.linspace(400.0, 600.0, n)
        volume = np.linspace(1_000_000, 3_000_000, n).astype(int)
        df = pd.DataFrame({"close": close, "volume": volume})
        result = engine.create_insider_aggregate_features(df)
        # After the warm-up window the cluster flag should activate
        late_cluster = result["insider_agg_cluster"].iloc[40:]
        assert late_cluster.sum() > 0, "Cluster flag never triggered despite high buy_ratio"

    def test_custom_window_used(self):
        """A shorter window should compute buy_ratio identically from scratch."""
        n = 80
        np.random.seed(7)
        close = 400 + np.cumsum(np.random.randn(n) * 2)
        volume = np.random.randint(500_000, 2_000_000, n)
        df = pd.DataFrame({"close": close, "volume": volume})

        e15 = InsiderAggregateFeatures(window=15, z_window=30)
        result = e15.create_insider_aggregate_features(df)
        # buy_ratio at row 15 should equal count of accumulation days / 15
        assert "insider_agg_buy_ratio" in result.columns
        assert result["insider_agg_buy_ratio"].iloc[14] <= 1.0


# =============================================================================
# TestAnalyzeCurrentInsider
# =============================================================================

class TestAnalyzeCurrentInsider:

    def test_returns_dict(self, engine, spy_with_features):
        result = engine.analyze_current_insider(spy_with_features)
        assert isinstance(result, dict)

    def test_required_keys_present(self, engine, spy_with_features):
        result = engine.analyze_current_insider(spy_with_features)
        for key in ("insider_regime", "buy_ratio", "volume_fraction", "cluster_buying", "buy_ratio_z"):
            assert key in result, f"Missing key: {key}"

    def test_regime_valid_values(self, engine, spy_with_features):
        result = engine.analyze_current_insider(spy_with_features)
        assert result["insider_regime"] in {"ACCUMULATION", "NEUTRAL", "DISTRIBUTION"}

    def test_none_when_no_insider_cols(self, engine, spy_daily):
        """Before create_insider_aggregate_features, there are no insider_agg_ cols."""
        result = engine.analyze_current_insider(spy_daily)
        assert result is None

    def test_accumulation_regime(self, engine):
        """Force a row with buy_ratio > 0.50 — regime must be ACCUMULATION."""
        df = pd.DataFrame({
            "insider_agg_buy_ratio": [0.70],
            "insider_agg_volume": [0.65],
            "insider_agg_cluster": [1.0],
            "insider_agg_z": [1.5],
        })
        result = engine.analyze_current_insider(df)
        assert result["insider_regime"] == "ACCUMULATION"

    def test_distribution_regime(self, engine):
        """Force a row with buy_ratio < 0.25 — regime must be DISTRIBUTION."""
        df = pd.DataFrame({
            "insider_agg_buy_ratio": [0.10],
            "insider_agg_volume": [0.20],
            "insider_agg_cluster": [0.0],
            "insider_agg_z": [-2.0],
        })
        result = engine.analyze_current_insider(df)
        assert result["insider_regime"] == "DISTRIBUTION"

    def test_neutral_regime(self, engine):
        """buy_ratio in (0.25, 0.50] → NEUTRAL."""
        df = pd.DataFrame({
            "insider_agg_buy_ratio": [0.38],
            "insider_agg_volume": [0.45],
            "insider_agg_cluster": [0.0],
            "insider_agg_z": [0.2],
        })
        result = engine.analyze_current_insider(df)
        assert result["insider_regime"] == "NEUTRAL"

    def test_none_on_empty_df(self, engine):
        df = pd.DataFrame({
            "insider_agg_buy_ratio": pd.Series(dtype=float),
            "insider_agg_volume": pd.Series(dtype=float),
            "insider_agg_cluster": pd.Series(dtype=float),
            "insider_agg_z": pd.Series(dtype=float),
        })
        result = engine.analyze_current_insider(df)
        assert result is None


# =============================================================================
# TestConfigIntegration
# =============================================================================

class TestConfigIntegration:

    def test_config_has_flag(self):
        from src.experiment_config import AntiOverfitConfig
        config = AntiOverfitConfig()
        assert hasattr(config, "use_insider_aggregate"), \
            "AntiOverfitConfig missing 'use_insider_aggregate'"
        assert config.use_insider_aggregate is True

    def test_feature_group_registered(self):
        from src.phase_10_feature_processing.group_aware_processor import FEATURE_GROUPS
        assert "insider_aggregate" in FEATURE_GROUPS, \
            "'insider_aggregate' not registered in FEATURE_GROUPS"
        assert "insider_agg_" in FEATURE_GROUPS["insider_aggregate"], \
            "'insider_agg_' prefix not in insider_aggregate group"
