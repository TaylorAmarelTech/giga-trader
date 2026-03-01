"""Tests for NMIFeatures — Normalized Mutual Information features (3 total)."""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.nmi_features import NMIFeatures


# ─── Helpers ────────────────────────────────────────────────────────────

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


ALL_3 = {"nmi_lag1_50d", "nmi_lag5_50d", "nmi_efficiency"}


# ─── Invariant Tests ────────────────────────────────────────────────────

class TestNMIInvariants:
    @pytest.fixture
    def feat(self):
        return NMIFeatures()

    @pytest.fixture
    def spy(self):
        return _make_spy_daily(200)

    def test_all_3_features_created(self, feat, spy):
        result = feat.create_nmi_features(spy)
        nmi_cols = {c for c in result.columns if c.startswith("nmi_")}
        assert nmi_cols == ALL_3

    def test_no_nans(self, feat, spy):
        result = feat.create_nmi_features(spy)
        for col in ALL_3:
            assert result[col].isna().sum() == 0, f"NaN in {col}"

    def test_no_infinities(self, feat, spy):
        result = feat.create_nmi_features(spy)
        for col in ALL_3:
            assert not np.isinf(result[col]).any(), f"Inf in {col}"

    def test_preserves_original_columns(self, feat, spy):
        original = set(spy.columns)
        result = feat.create_nmi_features(spy)
        assert original.issubset(set(result.columns))

    def test_preserves_row_count(self, feat, spy):
        result = feat.create_nmi_features(spy)
        assert len(result) == len(spy)

    def test_no_close_column(self, feat):
        df = pd.DataFrame({"date": [1, 2], "price": [100, 101]})
        result = feat.create_nmi_features(df)
        assert len(result.columns) == len(df.columns)

    def test_short_data(self, feat):
        """With fewer than 56 rows, features should default to 0.0."""
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=30),
            "close": np.linspace(400, 420, 30),
        })
        result = feat.create_nmi_features(df)
        for col in ALL_3:
            assert col in result.columns
            assert (result[col] == 0.0).all()


# ─── Feature Logic Tests ───────────────────────────────────────────────

class TestNMILogic:
    @pytest.fixture
    def feat(self):
        return NMIFeatures()

    def test_nmi_bounded_0_1(self, feat):
        df = _make_spy_daily(200)
        result = feat.create_nmi_features(df)
        for col in ["nmi_lag1_50d", "nmi_lag5_50d"]:
            assert result[col].min() >= 0.0, f"{col} below 0"
            assert result[col].max() <= 1.0, f"{col} above 1"

    def test_efficiency_bounded_0_1(self, feat):
        df = _make_spy_daily(200)
        result = feat.create_nmi_features(df)
        assert result["nmi_efficiency"].min() >= 0.0
        assert result["nmi_efficiency"].max() <= 1.0

    def test_efficiency_equals_1_minus_max(self, feat):
        df = _make_spy_daily(200)
        result = feat.create_nmi_features(df)
        expected = 1.0 - np.maximum(
            result["nmi_lag1_50d"].values,
            result["nmi_lag5_50d"].values,
        )
        # Clip expected the same way the implementation does
        expected = np.clip(expected, 0.0, 1.0)
        np.testing.assert_allclose(
            result["nmi_efficiency"].values, expected, atol=1e-10,
        )

    def test_lag1_positive_for_autocorrelated_data(self, feat):
        """Strongly autocorrelated series should produce higher NMI at lag 1."""
        rng = np.random.RandomState(99)
        n = 200
        # Build AR(1) process with phi=0.8 for strong autocorrelation
        returns = np.zeros(n)
        for i in range(1, n):
            returns[i] = 0.8 * returns[i - 1] + rng.normal(0, 0.01)
        close = 450.0 * np.cumprod(1 + returns)
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=n, freq="B"),
            "close": close,
        })
        result = feat.create_nmi_features(df)
        # Tail values (where rolling window is populated) should show nonzero NMI
        tail_lag1 = result["nmi_lag1_50d"].iloc[-20:].mean()
        assert tail_lag1 > 0.05, f"Expected positive NMI for autocorrelated data, got {tail_lag1}"


# ─── Analyze Tests ──────────────────────────────────────────────────────

class TestAnalyzeCurrentEfficiency:
    def test_returns_dict(self):
        feat = NMIFeatures()
        df = _make_spy_daily(200)
        df = feat.create_nmi_features(df)
        result = feat.analyze_current_efficiency(df)
        assert isinstance(result, dict)
        assert "efficiency_regime" in result

    def test_regime_values(self):
        feat = NMIFeatures()
        df = _make_spy_daily(200)
        df = feat.create_nmi_features(df)
        result = feat.analyze_current_efficiency(df)
        assert result["efficiency_regime"] in {"EFFICIENT", "PREDICTABLE", "MODERATE"}

    def test_returns_none_without_features(self):
        feat = NMIFeatures()
        df = pd.DataFrame({"close": [100]})
        assert feat.analyze_current_efficiency(df) is None


# ─── Feature Count Test ─────────────────────────────────────────────────

class TestFeatureCounts:
    def test_total_count(self):
        assert len(ALL_3) == 3
