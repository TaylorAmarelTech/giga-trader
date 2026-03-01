"""Tests for OrderFlowImbalanceFeatures -- BVC order flow features (8 total)."""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.order_flow_imbalance_features import (
    OrderFlowImbalanceFeatures,
)


# ---- Helpers ----------------------------------------------------------------

def _make_spy_daily(n_days: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    returns = rng.normal(0.0003, 0.012, n_days)
    close = 450.0 * np.cumprod(1 + returns)
    volume = rng.randint(50_000_000, 150_000_000, n_days).astype(float)
    return pd.DataFrame({
        "date": dates, "close": close, "volume": volume,
        "open": close * (1 + rng.normal(0, 0.003, n_days)),
        "high": close * 1.005, "low": close * 0.995,
    })


ALL_8 = {
    "ofi_buy_volume_proxy",
    "ofi_sell_volume_proxy",
    "ofi_imbalance",
    "ofi_cumulative_5d",
    "ofi_cumulative_20d",
    "ofi_normalized_20d",
    "ofi_imbalance_z",
    "ofi_regime",
}


# ---- Invariant Tests --------------------------------------------------------

class TestOrderFlowImbalanceInvariants:
    @pytest.fixture
    def feat(self):
        return OrderFlowImbalanceFeatures()

    @pytest.fixture
    def spy(self):
        return _make_spy_daily(200)

    def test_all_8_features_created(self, feat, spy):
        result = feat.create_order_flow_imbalance_features(spy)
        ofi_cols = {c for c in result.columns if c.startswith("ofi_")}
        assert ofi_cols == ALL_8

    def test_no_nans(self, feat, spy):
        result = feat.create_order_flow_imbalance_features(spy)
        for col in ALL_8:
            assert result[col].isna().sum() == 0, f"NaN in {col}"

    def test_no_infinities(self, feat, spy):
        result = feat.create_order_flow_imbalance_features(spy)
        for col in ALL_8:
            assert not np.isinf(result[col]).any(), f"Inf in {col}"

    def test_preserves_original_columns(self, feat, spy):
        original = set(spy.columns)
        result = feat.create_order_flow_imbalance_features(spy)
        assert original.issubset(set(result.columns))

    def test_preserves_row_count(self, feat, spy):
        result = feat.create_order_flow_imbalance_features(spy)
        assert len(result) == len(spy)

    def test_no_required_column(self, feat):
        """Missing OHLCV columns -> returns df unchanged."""
        df = pd.DataFrame({"date": [1, 2], "price": [100, 101]})
        result = feat.create_order_flow_imbalance_features(df)
        assert len(result.columns) == len(df.columns)

    def test_works_with_flat_data(self, feat):
        """Flat prices (high == low == close) should not crash."""
        n = 50
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=n),
            "open": np.full(n, 450.0),
            "high": np.full(n, 450.0),
            "low": np.full(n, 450.0),
            "close": np.full(n, 450.0),
            "volume": np.full(n, 100_000_000.0),
        })
        result = feat.create_order_flow_imbalance_features(df)
        for col in ALL_8:
            assert col in result.columns
            assert result[col].isna().sum() == 0, f"NaN in {col} with flat data"


# ---- Feature Logic Tests ----------------------------------------------------

class TestOrderFlowImbalanceLogic:
    @pytest.fixture
    def feat(self):
        return OrderFlowImbalanceFeatures()

    def test_buy_sell_frac_sum_equals_one(self, feat):
        """Buy fraction + sell fraction should equal 1.0 (source stores fractions)."""
        df = _make_spy_daily(100)
        result = feat.create_order_flow_imbalance_features(df)
        total = result["ofi_buy_volume_proxy"] + result["ofi_sell_volume_proxy"]
        np.testing.assert_allclose(total.values, np.ones(len(df)), rtol=1e-5)

    def test_close_at_high_means_all_buying(self, feat):
        """When close == high, buy fraction should be 1.0."""
        n = 50
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=n),
            "open": np.full(n, 448.0),
            "high": np.full(n, 452.0),
            "low": np.full(n, 446.0),
            "close": np.full(n, 452.0),  # close at high
            "volume": np.full(n, 100_000_000.0),
        })
        result = feat.create_order_flow_imbalance_features(df)
        np.testing.assert_allclose(
            result["ofi_buy_volume_proxy"].values,
            np.ones(n),
            rtol=1e-4,
        )
        # Imbalance should be close to 1.0
        assert result["ofi_imbalance"].iloc[-1] > 0.9

    def test_imbalance_z_bounded(self, feat):
        """Z-score should be clipped to [-4, 4]."""
        df = _make_spy_daily(200)
        result = feat.create_order_flow_imbalance_features(df)
        assert result["ofi_imbalance_z"].max() <= 4.0
        assert result["ofi_imbalance_z"].min() >= -4.0

    def test_regime_values(self, feat):
        """Regime should only be -1.0, 0.0, or 1.0."""
        df = _make_spy_daily(200)
        result = feat.create_order_flow_imbalance_features(df)
        vals = set(result["ofi_regime"].unique())
        assert vals.issubset({-1.0, 0.0, 1.0})


# ---- Analyze Tests -----------------------------------------------------------

class TestAnalyzeCurrentOrderFlow:
    def test_returns_dict(self):
        feat = OrderFlowImbalanceFeatures()
        df = _make_spy_daily(100)
        df = feat.create_order_flow_imbalance_features(df)
        result = feat.analyze_current_order_flow(df)
        assert isinstance(result, dict)
        assert "flow_regime" in result

    def test_regime_values(self):
        feat = OrderFlowImbalanceFeatures()
        df = _make_spy_daily(200)
        df = feat.create_order_flow_imbalance_features(df)
        result = feat.analyze_current_order_flow(df)
        assert result["flow_regime"] in {"ACCUMULATION", "DISTRIBUTION", "NEUTRAL"}

    def test_returns_none_without_features(self):
        feat = OrderFlowImbalanceFeatures()
        df = pd.DataFrame({"close": [100]})
        assert feat.analyze_current_order_flow(df) is None


# ---- Feature Count Test ------------------------------------------------------

class TestFeatureCounts:
    def test_total_count(self):
        assert len(ALL_8) == 8
