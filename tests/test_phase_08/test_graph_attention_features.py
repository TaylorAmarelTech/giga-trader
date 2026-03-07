"""
Tests for GraphAttentionFeatures - Cross-asset Graph Attention Network features.

Validates:
  - Default and custom construction
  - Feature names (all prefixed with gat_)
  - No NaN or Inf in output
  - Missing columns zero-fill
  - Short data handling
  - Feature value ranges (entropy >= 0, flows in [0,1], density in [0,1])
  - Row and column preservation
  - Multi-head attention produces valid distributions
  - Regime clustering produces valid labels
  - Analyze method returns correct structure
  - Correlated vs uncorrelated data produces different features
"""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.graph_attention_features import (
    GraphAttentionFeatures,
    _softmax,
    _kmeans_1d,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_spy_daily(n_days: int = 200, seed: int = 42) -> pd.DataFrame:
    """Minimal SPY daily DataFrame: close + daily_return."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    returns = rng.normal(0.0003, 0.012, n_days)
    close = 450.0 * np.cumprod(1 + returns)
    return pd.DataFrame({"date": dates, "close": close, "daily_return": returns})


def _make_full_daily(n_days: int = 200, seed: int = 7) -> pd.DataFrame:
    """Daily DataFrame with SPY + all 6 cross-asset return columns."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    spy_ret = rng.normal(0.0003, 0.012, n_days)
    close = 450.0 * np.cumprod(1 + spy_ret)
    df = pd.DataFrame({"date": dates, "close": close, "daily_return": spy_ret})
    for col in ["TLT_return", "GLD_return", "QQQ_return",
                 "IWM_return", "VXX_return", "HYG_return"]:
        df[col] = rng.normal(0.0001, 0.01, n_days)
    return df


def _make_correlated_daily(n_days: int = 200, seed: int = 0) -> pd.DataFrame:
    """All cross-asset returns are strongly correlated with SPY."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    spy_ret = rng.normal(0.0003, 0.012, n_days)
    close = 450.0 * np.cumprod(1 + spy_ret)
    df = pd.DataFrame({"date": dates, "close": close, "daily_return": spy_ret})
    # All assets = SPY + small noise -> high correlation
    for col in ["TLT_return", "GLD_return", "QQQ_return",
                 "IWM_return", "VXX_return", "HYG_return"]:
        df[col] = spy_ret + rng.normal(0, 0.001, n_days)
    return df


# ── Constructor Tests ─────────────────────────────────────────────────────────


class TestGraphAttentionFeaturesInit:

    def test_default_construction(self):
        gaf = GraphAttentionFeatures()
        assert gaf.window == 20
        assert gaf.n_heads == 4
        assert gaf.density_threshold == 0.3

    def test_custom_window(self):
        gaf = GraphAttentionFeatures(window=40)
        assert gaf.window == 40

    def test_custom_n_heads(self):
        gaf = GraphAttentionFeatures(n_heads=8)
        assert gaf.n_heads == 8

    def test_custom_density_threshold(self):
        gaf = GraphAttentionFeatures(density_threshold=0.5)
        assert gaf.density_threshold == 0.5

    def test_minimum_window_clamped(self):
        """Window should be at least 2."""
        gaf = GraphAttentionFeatures(window=1)
        assert gaf.window >= 2

    def test_minimum_heads_clamped(self):
        """At least 1 attention head."""
        gaf = GraphAttentionFeatures(n_heads=0)
        assert gaf.n_heads >= 1


# ── Feature Names Validation ─────────────────────────────────────────────────


class TestFeatureNames:

    def test_all_feature_names_start_with_gat(self):
        for name in GraphAttentionFeatures.FEATURE_NAMES:
            assert name.startswith("gat_"), (
                f"Feature {name!r} does not start with gat_"
            )

    def test_exactly_8_features_defined(self):
        assert len(GraphAttentionFeatures.FEATURE_NAMES) == 8

    def test_all_8_features_in_output(self):
        gaf = GraphAttentionFeatures()
        df = _make_full_daily(100)
        result = gaf.create_graph_attention_features(df)
        for name in GraphAttentionFeatures.FEATURE_NAMES:
            assert name in result.columns, f"Missing feature: {name}"

    def test_new_columns_all_have_gat_prefix(self):
        gaf = GraphAttentionFeatures()
        df = _make_full_daily(100)
        original_cols = set(df.columns)
        result = gaf.create_graph_attention_features(df)
        new_cols = set(result.columns) - original_cols
        for col in new_cols:
            assert col.startswith("gat_"), f"New column {col!r} missing gat_ prefix"


# ── No NaN / Inf Tests ───────────────────────────────────────────────────────


class TestNoNanInf:

    def test_no_nans_spy_only(self):
        gaf = GraphAttentionFeatures()
        df = _make_spy_daily(100)
        result = gaf.create_graph_attention_features(df)
        gat_cols = [c for c in result.columns if c.startswith("gat_")]
        nan_count = result[gat_cols].isna().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values"

    def test_no_nans_with_cross_assets(self):
        gaf = GraphAttentionFeatures()
        df = _make_full_daily(200)
        result = gaf.create_graph_attention_features(df)
        gat_cols = [c for c in result.columns if c.startswith("gat_")]
        nan_count = result[gat_cols].isna().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values"

    def test_no_inf_values(self):
        gaf = GraphAttentionFeatures()
        df = _make_full_daily(200)
        result = gaf.create_graph_attention_features(df)
        gat_cols = [c for c in result.columns if c.startswith("gat_")]
        inf_count = np.isinf(result[gat_cols].values).sum()
        assert inf_count == 0, f"Found {inf_count} Inf values"


# ── Missing Columns / Zero-Fill ──────────────────────────────────────────────


class TestMissingColumnsZeroFill:

    def test_missing_close_all_zeros(self):
        """Without 'close', all features should be zero-filled."""
        gaf = GraphAttentionFeatures()
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-01", periods=50),
            "volume": range(50),
        })
        result = gaf.create_graph_attention_features(df)
        gat_cols = [c for c in result.columns if c.startswith("gat_")]
        assert len(gat_cols) == 8
        assert (result[gat_cols] == 0.0).all().all()

    def test_single_row_no_crash(self):
        """Single-row DataFrame should not crash."""
        gaf = GraphAttentionFeatures()
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-06-01"]),
            "close": [450.0],
        })
        result = gaf.create_graph_attention_features(df)
        gat_cols = [c for c in result.columns if c.startswith("gat_")]
        assert len(gat_cols) == 8
        assert len(result) == 1


# ── Short Data Handling ──────────────────────────────────────────────────────


class TestShortData:

    def test_shorter_than_window(self):
        """Data shorter than window should produce features (mostly zero)."""
        gaf = GraphAttentionFeatures(window=60)
        df = _make_spy_daily(30)
        result = gaf.create_graph_attention_features(df)
        assert len(result) == 30
        gat_cols = [c for c in result.columns if c.startswith("gat_")]
        assert len(gat_cols) == 8
        # No NaN
        assert result[gat_cols].isna().sum().sum() == 0

    def test_two_rows_no_crash(self):
        gaf = GraphAttentionFeatures()
        df = pd.DataFrame({
            "close": [450.0, 451.0],
        })
        result = gaf.create_graph_attention_features(df)
        assert len(result) == 2
        gat_cols = [c for c in result.columns if c.startswith("gat_")]
        assert len(gat_cols) == 8

    def test_all_nan_close(self):
        """All-NaN close should produce zero-filled features."""
        gaf = GraphAttentionFeatures()
        dates = pd.bdate_range("2024-01-02", periods=50)
        df = pd.DataFrame({"date": dates, "close": np.full(50, np.nan)})
        result = gaf.create_graph_attention_features(df)
        gat_cols = [c for c in result.columns if c.startswith("gat_")]
        assert len(gat_cols) == 8
        assert result[gat_cols].isna().sum().sum() == 0


# ── Feature Value Ranges ─────────────────────────────────────────────────────


class TestFeatureValueRanges:

    def test_entropy_non_negative(self):
        """Attention entropy must be >= 0."""
        gaf = GraphAttentionFeatures()
        df = _make_full_daily(200)
        result = gaf.create_graph_attention_features(df)
        assert (result["gat_spy_attention_entropy"] >= 0).all(), (
            f"Min entropy: {result['gat_spy_attention_entropy'].min()}"
        )

    def test_safe_haven_flow_in_zero_one(self):
        gaf = GraphAttentionFeatures()
        df = _make_full_daily(200)
        result = gaf.create_graph_attention_features(df)
        col = result["gat_spy_safe_haven_flow"]
        assert (col >= -1e-9).all() and (col <= 1.0 + 1e-9).all(), (
            f"safe_haven_flow range: [{col.min()}, {col.max()}]"
        )

    def test_risk_on_flow_in_zero_one(self):
        gaf = GraphAttentionFeatures()
        df = _make_full_daily(200)
        result = gaf.create_graph_attention_features(df)
        col = result["gat_spy_risk_on_flow"]
        assert (col >= -1e-9).all() and (col <= 1.0 + 1e-9).all(), (
            f"risk_on_flow range: [{col.min()}, {col.max()}]"
        )

    def test_fear_attention_in_zero_one(self):
        gaf = GraphAttentionFeatures()
        df = _make_full_daily(200)
        result = gaf.create_graph_attention_features(df)
        col = result["gat_spy_fear_attention"]
        assert (col >= -1e-9).all() and (col <= 1.0 + 1e-9).all()

    def test_credit_attention_in_zero_one(self):
        gaf = GraphAttentionFeatures()
        df = _make_full_daily(200)
        result = gaf.create_graph_attention_features(df)
        col = result["gat_spy_credit_attention"]
        assert (col >= -1e-9).all() and (col <= 1.0 + 1e-9).all()

    def test_density_in_zero_one(self):
        gaf = GraphAttentionFeatures()
        df = _make_full_daily(200)
        result = gaf.create_graph_attention_features(df)
        col = result["gat_graph_density"]
        assert (col >= 0.0).all() and (col <= 1.0).all(), (
            f"density range: [{col.min()}, {col.max()}]"
        )

    def test_regime_cluster_valid_labels(self):
        """Cluster labels should be in {0, 1, 2}."""
        gaf = GraphAttentionFeatures()
        df = _make_full_daily(200)
        result = gaf.create_graph_attention_features(df)
        unique = set(result["gat_regime_cluster"].unique())
        assert unique.issubset({0.0, 1.0, 2.0}), f"Unexpected cluster labels: {unique}"


# ── Row and Column Preservation ──────────────────────────────────────────────


class TestRowColumnPreservation:

    def test_row_count_unchanged(self):
        gaf = GraphAttentionFeatures()
        df = _make_full_daily(150)
        result = gaf.create_graph_attention_features(df)
        assert len(result) == len(df)

    def test_original_columns_preserved(self):
        gaf = GraphAttentionFeatures()
        df = _make_full_daily(100)
        original_cols = set(df.columns)
        result = gaf.create_graph_attention_features(df)
        assert original_cols.issubset(set(result.columns))

    def test_original_close_unchanged(self):
        gaf = GraphAttentionFeatures()
        df = _make_full_daily(100)
        result = gaf.create_graph_attention_features(df)
        pd.testing.assert_series_equal(
            result["close"].reset_index(drop=True),
            df["close"].reset_index(drop=True),
        )


# ── Softmax Helper ───────────────────────────────────────────────────────────


class TestSoftmax:

    def test_1d_sums_to_one(self):
        x = np.array([1.0, 2.0, 3.0])
        result = _softmax(x)
        assert abs(result.sum() - 1.0) < 1e-8

    def test_1d_all_positive(self):
        x = np.array([-10.0, 0.0, 10.0])
        result = _softmax(x)
        assert (result >= 0).all()

    def test_2d_rows_sum_to_one(self):
        x = np.array([[1.0, 2.0], [3.0, 4.0]])
        result = _softmax(x)
        row_sums = result.sum(axis=1)
        np.testing.assert_allclose(row_sums, [1.0, 1.0], atol=1e-8)

    def test_uniform_input(self):
        """Equal inputs should produce uniform distribution."""
        x = np.array([5.0, 5.0, 5.0, 5.0])
        result = _softmax(x)
        np.testing.assert_allclose(result, 0.25, atol=1e-8)


# ── K-Means Helper ───────────────────────────────────────────────────────────


class TestKMeans:

    def test_returns_correct_length(self):
        data = np.random.RandomState(0).randn(50, 3)
        labels = _kmeans_1d(data, k=3)
        assert len(labels) == 50

    def test_labels_in_valid_range(self):
        data = np.random.RandomState(0).randn(50, 3)
        labels = _kmeans_1d(data, k=3)
        assert set(labels).issubset({0, 1, 2})

    def test_empty_input(self):
        labels = _kmeans_1d(np.zeros((0, 3)), k=3)
        assert len(labels) == 0

    def test_fewer_points_than_k(self):
        data = np.array([[1.0, 0.0], [0.0, 1.0]])
        labels = _kmeans_1d(data, k=3)
        assert len(labels) == 2


# ── Correlated vs Uncorrelated Behavior ──────────────────────────────────────


class TestCorrelatedBehavior:

    def test_high_correlation_gives_high_density(self):
        """Strongly correlated assets should produce higher graph density."""
        gaf = GraphAttentionFeatures(window=20, density_threshold=0.3)

        corr_df = _make_correlated_daily(200)
        result_corr = gaf.create_graph_attention_features(corr_df)

        # Random (low-correlation) data
        uncorr_df = _make_full_daily(200, seed=99)
        result_uncorr = gaf.create_graph_attention_features(uncorr_df)

        # After warmup, correlated data should have higher mean density
        tail_corr = result_corr["gat_graph_density"].iloc[30:].mean()
        tail_uncorr = result_uncorr["gat_graph_density"].iloc[30:].mean()
        assert tail_corr >= tail_uncorr, (
            f"Correlated density ({tail_corr:.3f}) should be >= "
            f"uncorrelated density ({tail_uncorr:.3f})"
        )


# ── Analyze Method ───────────────────────────────────────────────────────────


class TestAnalyzeCurrentGraph:

    def test_returns_dict(self):
        gaf = GraphAttentionFeatures()
        df = _make_full_daily(100)
        result = gaf.create_graph_attention_features(df)
        analysis = gaf.analyze_current_graph(result)
        assert isinstance(analysis, dict)

    def test_regime_key_present(self):
        gaf = GraphAttentionFeatures()
        df = _make_full_daily(100)
        result = gaf.create_graph_attention_features(df)
        analysis = gaf.analyze_current_graph(result)
        assert "graph_regime" in analysis

    def test_valid_regime_values(self):
        gaf = GraphAttentionFeatures()
        df = _make_full_daily(100)
        result = gaf.create_graph_attention_features(df)
        analysis = gaf.analyze_current_graph(result)
        valid = {"RISK_OFF", "RISK_ON", "SYSTEMIC", "NEUTRAL"}
        assert analysis["graph_regime"] in valid

    def test_returns_none_without_features(self):
        gaf = GraphAttentionFeatures()
        df = _make_spy_daily(50)
        # Features not computed yet
        analysis = gaf.analyze_current_graph(df)
        assert analysis is None

    def test_returns_none_for_empty_df(self):
        gaf = GraphAttentionFeatures()
        analysis = gaf.analyze_current_graph(pd.DataFrame())
        assert analysis is None

    def test_numeric_values_are_floats(self):
        gaf = GraphAttentionFeatures()
        df = _make_full_daily(100)
        result = gaf.create_graph_attention_features(df)
        analysis = gaf.analyze_current_graph(result)
        for key in ["gat_spy_attention_entropy", "gat_spy_safe_haven_flow",
                     "gat_spy_risk_on_flow", "gat_spy_fear_attention",
                     "gat_graph_density"]:
            assert isinstance(analysis[key], float), (
                f"{key} is {type(analysis[key])}, expected float"
            )


# ── FeatureModuleBase Integration ────────────────────────────────────────────


class TestFeatureModuleBaseIntegration:

    def test_inherits_from_base(self):
        from src.core.feature_base import FeatureModuleBase
        assert issubclass(GraphAttentionFeatures, FeatureModuleBase)

    def test_required_cols_defined(self):
        assert "close" in GraphAttentionFeatures.REQUIRED_COLS

    def test_all_feature_names_classmethod(self):
        names = GraphAttentionFeatures._all_feature_names()
        assert len(names) == 8
        assert all(n.startswith("gat_") for n in names)

    def test_cleanup_removes_nan(self):
        """The _cleanup_features method should replace NaN with 0."""
        gaf = GraphAttentionFeatures()
        df = pd.DataFrame({"gat_spy_attention_entropy": [np.nan, 1.0, np.inf]})
        result = gaf._cleanup_features(df)
        assert result["gat_spy_attention_entropy"].isna().sum() == 0
        assert not np.isinf(result["gat_spy_attention_entropy"].values).any()
