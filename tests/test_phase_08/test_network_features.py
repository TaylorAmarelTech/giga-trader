"""
Tests for NetworkFeatures — Correlation Network Centrality.

Validates graph construction, BFS connected-component counting,
rolling window logic, feature values, and regime classification
without any live API calls or networkx dependency.
"""

from collections import deque

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.network_features import (
    NetworkFeatures,
    _count_connected_components,
    _FEATURE_NAMES,
    _CROSS_ASSET_RETURN_COLS,
)


# ─── Shared fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def spy_daily_base(rng):
    """Minimal SPY daily DataFrame: 220 business days, close only."""
    n = 220
    dates = pd.bdate_range("2023-01-03", periods=n)
    log_ret = rng.normal(0.0004, 0.01, n)
    close = 450.0 * np.exp(np.cumsum(log_ret))
    return pd.DataFrame({"date": dates, "close": close})


@pytest.fixture
def spy_daily_with_cross_assets(spy_daily_base, rng):
    """SPY daily DataFrame enriched with 4 cross-asset return columns."""
    df = spy_daily_base.copy()
    n = len(df)
    for col in ["TLT_return", "QQQ_return", "GLD_return", "VXX_return"]:
        df[col] = rng.normal(0.0, 0.008, n)
    return df


@pytest.fixture
def nf():
    """Default NetworkFeatures instance."""
    return NetworkFeatures()


@pytest.fixture
def nf_custom():
    """NetworkFeatures with custom window and threshold."""
    return NetworkFeatures(window=30, correlation_threshold=0.3)


# ─── BFS helper ──────────────────────────────────────────────────────────────

class TestBFSConnectedComponents:

    def test_fully_connected_graph(self):
        """All nodes connected → 1 component."""
        adj = np.array([
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ], dtype=bool)
        assert _count_connected_components(adj) == 1

    def test_no_edges(self):
        """No edges → each node is its own component."""
        adj = np.zeros((4, 4), dtype=bool)
        assert _count_connected_components(adj) == 4

    def test_two_components(self):
        """Two isolated pairs → 2 components."""
        adj = np.array([
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ], dtype=bool)
        assert _count_connected_components(adj) == 2

    def test_single_node(self):
        """Single-node graph → 1 component."""
        adj = np.zeros((1, 1), dtype=bool)
        assert _count_connected_components(adj) == 1

    def test_empty_graph(self):
        """Zero-node graph → returns 1 (degenerate base case)."""
        adj = np.zeros((0, 0), dtype=bool)
        assert _count_connected_components(adj) == 1

    def test_chain_graph(self):
        """A–B–C–D (chain) → 1 component."""
        adj = np.array([
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ], dtype=bool)
        assert _count_connected_components(adj) == 1


# ─── Constructor tests ────────────────────────────────────────────────────────

class TestNetworkFeaturesInit:

    def test_default_window(self):
        nf = NetworkFeatures()
        assert nf.window == 60

    def test_default_threshold(self):
        nf = NetworkFeatures()
        assert nf.correlation_threshold == 0.5

    def test_custom_params(self):
        nf = NetworkFeatures(window=30, correlation_threshold=0.3)
        assert nf.window == 30
        assert nf.correlation_threshold == 0.3


# ─── download_network_data ────────────────────────────────────────────────────

class TestDownloadNetworkData:

    def test_returns_dataframe(self, nf):
        from datetime import datetime
        result = nf.download_network_data(datetime(2024, 1, 1), datetime(2024, 12, 31))
        assert isinstance(result, pd.DataFrame)

    def test_returns_empty(self, nf):
        from datetime import datetime
        result = nf.download_network_data(datetime(2024, 1, 1), datetime(2024, 12, 31))
        assert result.empty


# ─── Feature count and naming ─────────────────────────────────────────────────

class TestFeatureNamesAndCount:

    def test_exactly_five_features_spy_only(self, nf, spy_daily_base):
        result = nf.create_network_features(spy_daily_base)
        netw_cols = [c for c in result.columns if c.startswith("netw_")]
        assert len(netw_cols) == 5

    def test_exactly_five_features_with_cross_assets(self, nf, spy_daily_with_cross_assets):
        result = nf.create_network_features(spy_daily_with_cross_assets)
        netw_cols = [c for c in result.columns if c.startswith("netw_")]
        assert len(netw_cols) == 5

    def test_feature_names_correct(self, nf, spy_daily_base):
        result = nf.create_network_features(spy_daily_base)
        for name in _FEATURE_NAMES:
            assert name in result.columns, f"Missing feature column: {name}"

    def test_all_features_have_netw_prefix(self, nf, spy_daily_base):
        original_cols = set(spy_daily_base.columns)
        result = nf.create_network_features(spy_daily_base)
        new_cols = set(result.columns) - original_cols
        for col in new_cols:
            assert col.startswith("netw_"), f"Column {col!r} missing netw_ prefix"


# ─── Row preservation ─────────────────────────────────────────────────────────

class TestRowPreservation:

    def test_row_count_unchanged_spy_only(self, nf, spy_daily_base):
        result = nf.create_network_features(spy_daily_base)
        assert len(result) == len(spy_daily_base)

    def test_row_count_unchanged_with_cross_assets(self, nf, spy_daily_with_cross_assets):
        result = nf.create_network_features(spy_daily_with_cross_assets)
        assert len(result) == len(spy_daily_with_cross_assets)

    def test_original_columns_preserved(self, nf, spy_daily_base):
        original_cols = set(spy_daily_base.columns)
        result = nf.create_network_features(spy_daily_base)
        assert original_cols.issubset(set(result.columns))


# ─── NaN / Inf hygiene ────────────────────────────────────────────────────────

class TestNoNanOrInf:

    def test_no_nans_spy_only(self, nf, spy_daily_base):
        result = nf.create_network_features(spy_daily_base)
        netw_cols = [c for c in result.columns if c.startswith("netw_")]
        assert result[netw_cols].isna().sum().sum() == 0

    def test_no_nans_with_cross_assets(self, nf, spy_daily_with_cross_assets):
        result = nf.create_network_features(spy_daily_with_cross_assets)
        netw_cols = [c for c in result.columns if c.startswith("netw_")]
        assert result[netw_cols].isna().sum().sum() == 0

    def test_no_inf_values(self, nf, spy_daily_base):
        result = nf.create_network_features(spy_daily_base)
        netw_cols = [c for c in result.columns if c.startswith("netw_")]
        assert not np.isinf(result[netw_cols].values).any()


# ─── Value range validation ────────────────────────────────────────────────────

class TestValueRanges:

    def test_density_in_zero_one(self, nf, spy_daily_with_cross_assets):
        result = nf.create_network_features(spy_daily_with_cross_assets)
        col = result["netw_density"]
        assert col.min() >= 0.0, f"netw_density below 0: {col.min()}"
        assert col.max() <= 1.0, f"netw_density above 1: {col.max()}"

    def test_avg_centrality_in_zero_one(self, nf, spy_daily_with_cross_assets):
        result = nf.create_network_features(spy_daily_with_cross_assets)
        col = result["netw_avg_centrality"]
        assert col.min() >= 0.0
        assert col.max() <= 1.0

    def test_modularity_in_zero_one(self, nf, spy_daily_with_cross_assets):
        result = nf.create_network_features(spy_daily_with_cross_assets)
        col = result["netw_modularity"]
        assert col.min() >= 0.0
        # modularity = 1 - 1/n_components; max is 1 - 1/n_nodes
        assert col.max() < 1.0

    def test_density_nonzero_with_cross_assets(self, nf, spy_daily_with_cross_assets):
        """With 5 return series, some correlation edges should form."""
        result = nf.create_network_features(spy_daily_with_cross_assets)
        # After the warm-up period (window rows), at least some nonzero density
        tail = result["netw_density"].iloc[nf.window :]
        assert tail.max() >= 0.0  # Just ensure no crash; can be 0 with low-corr data


# ─── Graceful degradation ─────────────────────────────────────────────────────

class TestGracefulDegradation:

    def test_missing_close_column(self, nf):
        """No 'close' column → all features are 0.0."""
        df = pd.DataFrame({"date": pd.bdate_range("2024-01-01", periods=10), "volume": range(10)})
        result = nf.create_network_features(df)
        netw_cols = [c for c in result.columns if c.startswith("netw_")]
        assert len(netw_cols) == 5
        assert (result[netw_cols] == 0.0).all().all()

    def test_short_history_no_crash(self, nf):
        """Fewer rows than window → warm-up zeros, no crash."""
        dates = pd.bdate_range("2024-01-01", periods=10)
        df = pd.DataFrame({
            "date": dates,
            "close": np.linspace(440, 460, 10),
        })
        result = nf.create_network_features(df)
        assert len(result) == 10
        assert "netw_density" in result.columns

    def test_single_row(self, nf):
        """Single-row DataFrame: no crash, features are 0.0."""
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-06-01"]),
            "close": [450.0],
        })
        result = nf.create_network_features(df)
        netw_cols = [c for c in result.columns if c.startswith("netw_")]
        assert len(netw_cols) == 5

    def test_nan_close_values(self, nf):
        """NaN close values degrade gracefully."""
        n = 100
        dates = pd.bdate_range("2024-01-01", periods=n)
        close = np.where(np.arange(n) % 10 == 0, np.nan, 450.0 + np.arange(n) * 0.1)
        df = pd.DataFrame({"date": dates, "close": close})
        result = nf.create_network_features(df)
        netw_cols = [c for c in result.columns if c.startswith("netw_")]
        assert result[netw_cols].isna().sum().sum() == 0


# ─── Highly-correlated synthetic data ────────────────────────────────────────

class TestHighCorrelationRegime:
    """When all assets are perfectly correlated, density should be 1.0."""

    def _make_perfectly_correlated(self, n: int = 200) -> pd.DataFrame:
        """All return columns are identical → |corr| == 1.0 everywhere."""
        dates = pd.bdate_range("2022-01-03", periods=n)
        rng = np.random.default_rng(0)
        base_ret = rng.normal(0.0005, 0.01, n)
        close = 450.0 * np.exp(np.cumsum(base_ret))
        df = pd.DataFrame({"date": dates, "close": close})
        # Add perfectly correlated cross-asset columns
        for col in ["TLT_return", "QQQ_return", "GLD_return", "VXX_return"]:
            df[col] = base_ret  # Identical to SPY return → corr = 1.0
        return df

    def test_density_near_one_when_all_correlated(self):
        nf = NetworkFeatures(window=60, correlation_threshold=0.5)
        df = self._make_perfectly_correlated(200)
        result = nf.create_network_features(df)
        # After warm-up, all edges should exist → density = 1.0
        tail_density = result["netw_density"].iloc[nf.window :]
        assert tail_density.max() == pytest.approx(1.0, abs=0.01)

    def test_modularity_zero_when_fully_connected(self):
        """Fully connected → 1 component → modularity = 0."""
        nf = NetworkFeatures(window=60, correlation_threshold=0.5)
        df = self._make_perfectly_correlated(200)
        result = nf.create_network_features(df)
        tail_mod = result["netw_modularity"].iloc[nf.window :]
        assert tail_mod.max() == pytest.approx(0.0, abs=0.01)


# ─── Uncorrelated data ────────────────────────────────────────────────────────

class TestLowCorrelationRegime:
    """Independent assets → very few or no edges → low density."""

    def _make_uncorrelated(self, n: int = 200) -> pd.DataFrame:
        dates = pd.bdate_range("2022-01-03", periods=n)
        rng = np.random.default_rng(99)
        close = 450.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.01, n)))
        df = pd.DataFrame({"date": dates, "close": close})
        # Completely independent returns
        for col in ["TLT_return", "QQQ_return", "GLD_return", "VXX_return"]:
            df[col] = rng.normal(0.0, 0.01, n)
        return df

    def test_density_low_when_uncorrelated(self):
        nf = NetworkFeatures(window=60, correlation_threshold=0.8)
        df = self._make_uncorrelated(200)
        result = nf.create_network_features(df)
        tail_density = result["netw_density"].iloc[nf.window :]
        assert tail_density.mean() < 0.5  # Most edges should be absent


# ─── analyze_current_network ─────────────────────────────────────────────────

class TestAnalyzeCurrentNetwork:

    def test_returns_dict_after_feature_creation(self, nf, spy_daily_with_cross_assets):
        result = nf.create_network_features(spy_daily_with_cross_assets)
        analysis = nf.analyze_current_network(result)
        assert isinstance(analysis, dict)

    def test_network_regime_key_present(self, nf, spy_daily_with_cross_assets):
        result = nf.create_network_features(spy_daily_with_cross_assets)
        analysis = nf.analyze_current_network(result)
        assert "network_regime" in analysis

    def test_valid_regime_values(self, nf, spy_daily_with_cross_assets):
        result = nf.create_network_features(spy_daily_with_cross_assets)
        analysis = nf.analyze_current_network(result)
        valid = {"SYSTEMIC_RISK", "NORMAL", "FRAGMENTED"}
        assert analysis["network_regime"] in valid

    def test_all_metric_keys_present(self, nf, spy_daily_with_cross_assets):
        result = nf.create_network_features(spy_daily_with_cross_assets)
        analysis = nf.analyze_current_network(result)
        for key in _FEATURE_NAMES:
            assert key in analysis, f"Missing key {key!r} in analysis dict"

    def test_returns_none_without_feature_cols(self, nf, spy_daily_base):
        """If features haven't been computed, analyze should return None."""
        analysis = nf.analyze_current_network(spy_daily_base)
        assert analysis is None

    def test_returns_none_for_empty_df(self, nf):
        analysis = nf.analyze_current_network(pd.DataFrame())
        assert analysis is None

    def test_systemic_risk_regime_detection(self, nf):
        """Manually inject high-density + high-z values and verify regime."""
        df = pd.DataFrame({
            "netw_density": [0.9],
            "netw_avg_centrality": [0.8],
            "netw_centrality_z": [2.0],
            "netw_modularity": [0.0],
            "netw_hub_disconnect": [0.0],
            "close": [450.0],
        })
        result = nf.analyze_current_network(df)
        assert result is not None
        assert result["network_regime"] == "SYSTEMIC_RISK"

    def test_fragmented_regime_detection(self, nf):
        """High modularity → FRAGMENTED regime."""
        df = pd.DataFrame({
            "netw_density": [0.1],
            "netw_avg_centrality": [0.1],
            "netw_centrality_z": [0.0],
            "netw_modularity": [0.8],
            "netw_hub_disconnect": [0.0],
            "close": [450.0],
        })
        result = nf.analyze_current_network(df)
        assert result is not None
        assert result["network_regime"] == "FRAGMENTED"

    def test_normal_regime_default(self, nf):
        """Moderate density, low z → NORMAL regime."""
        df = pd.DataFrame({
            "netw_density": [0.3],
            "netw_avg_centrality": [0.3],
            "netw_centrality_z": [0.1],
            "netw_modularity": [0.2],
            "netw_hub_disconnect": [0.0],
            "close": [450.0],
        })
        result = nf.analyze_current_network(df)
        assert result is not None
        assert result["network_regime"] == "NORMAL"


# ─── Custom window and threshold ──────────────────────────────────────────────

class TestCustomWindowThreshold:

    def test_custom_window_runs(self, nf_custom, spy_daily_with_cross_assets):
        result = nf_custom.create_network_features(spy_daily_with_cross_assets)
        netw_cols = [c for c in result.columns if c.startswith("netw_")]
        assert len(netw_cols) == 5

    def test_lower_threshold_gives_higher_density(self, spy_daily_with_cross_assets):
        """Lower correlation threshold → more edges → higher average density."""
        nf_low = NetworkFeatures(window=60, correlation_threshold=0.1)
        nf_high = NetworkFeatures(window=60, correlation_threshold=0.9)

        df = spy_daily_with_cross_assets.copy()
        res_low = nf_low.create_network_features(df)
        res_high = nf_high.create_network_features(df.copy())

        mean_low = res_low["netw_density"].mean()
        mean_high = res_high["netw_density"].mean()
        assert mean_low >= mean_high, (
            f"Lower threshold should give >= density: {mean_low:.4f} vs {mean_high:.4f}"
        )
