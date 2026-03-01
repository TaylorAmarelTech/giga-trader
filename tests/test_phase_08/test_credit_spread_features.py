"""Tests for CreditSpreadFeatures -- 8 credit-spread features (prefix cred_)."""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.credit_spread_features import CreditSpreadFeatures


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


ALL_8 = {
    "cred_hy_ig_spread",
    "cred_spread_20d",
    "cred_spread_z",
    "cred_spread_momentum",
    "cred_spread_accel",
    "cred_hy_return_20d",
    "cred_ig_return_20d",
    "cred_spread_regime",
}


# ─── Invariant Tests ────────────────────────────────────────────────────

class TestCreditSpreadInvariants:
    @pytest.fixture
    def feat(self):
        return CreditSpreadFeatures()

    @pytest.fixture
    def spy(self):
        return _make_spy_daily(200)

    def test_all_8_features_created(self, feat, spy):
        result = feat.create_credit_spread_features(spy)
        cred_cols = {c for c in result.columns if c.startswith("cred_")}
        assert cred_cols == ALL_8

    def test_no_nans(self, feat, spy):
        result = feat.create_credit_spread_features(spy)
        for col in ALL_8:
            assert result[col].isna().sum() == 0, f"NaN in {col}"

    def test_no_infinities(self, feat, spy):
        result = feat.create_credit_spread_features(spy)
        for col in ALL_8:
            assert not np.isinf(result[col]).any(), f"Inf in {col}"

    def test_preserves_original_columns(self, feat, spy):
        original = set(spy.columns)
        result = feat.create_credit_spread_features(spy)
        assert original.issubset(set(result.columns))

    def test_preserves_row_count(self, feat, spy):
        result = feat.create_credit_spread_features(spy)
        assert len(result) == len(spy)

    def test_no_required_column(self, feat):
        """Missing 'close' should return df unchanged, not crash."""
        df = pd.DataFrame({"date": [1, 2], "price": [100, 101]})
        result = feat.create_credit_spread_features(df)
        assert len(result.columns) == len(df.columns)

    def test_works_with_flat_data(self, feat):
        """Flat close prices (zero variance) should not crash."""
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=50),
            "close": np.full(50, 450.0),
        })
        result = feat.create_credit_spread_features(df)
        for col in ALL_8:
            assert col in result.columns
            assert result[col].isna().sum() == 0, f"NaN in {col} with flat data"


# ─── Feature Logic Tests ───────────────────────────────────────────────

class TestCreditSpreadLogic:
    @pytest.fixture
    def feat(self):
        return CreditSpreadFeatures()

    def test_spread_z_bounded(self, feat):
        """Z-score should be clipped to [-4, 4]."""
        df = _make_spy_daily(200)
        result = feat.create_credit_spread_features(df)
        assert result["cred_spread_z"].max() <= 4.0
        assert result["cred_spread_z"].min() >= -4.0

    def test_regime_values(self, feat):
        """Regime should only be -1.0, 0.0, or 1.0."""
        df = _make_spy_daily(200)
        result = feat.create_credit_spread_features(df)
        unique = set(result["cred_spread_regime"].unique())
        assert unique.issubset({-1.0, 0.0, 1.0})

    def test_momentum_is_diff_of_spread_20d(self, feat):
        """Momentum should be the 5-day change of the 20d spread."""
        df = _make_spy_daily(200)
        result = feat.create_credit_spread_features(df)
        expected = result["cred_spread_20d"].diff(5)
        # Compare where both are not zero (first rows are filled to 0)
        mask = (expected != 0) & (result["cred_spread_momentum"] != 0)
        if mask.any():
            np.testing.assert_allclose(
                result.loc[mask, "cred_spread_momentum"].values,
                expected.loc[mask].values,
                atol=1e-10,
            )


# ─── Analyze Tests ──────────────────────────────────────────────────────

class TestAnalyzeCurrentCreditSpread:
    def test_returns_dict(self):
        feat = CreditSpreadFeatures()
        df = _make_spy_daily(100)
        df = feat.create_credit_spread_features(df)
        result = feat.analyze_current_credit_spread(df)
        assert isinstance(result, dict)
        assert "credit_regime" in result

    def test_regime_values(self):
        feat = CreditSpreadFeatures()
        df = _make_spy_daily(200)
        df = feat.create_credit_spread_features(df)
        result = feat.analyze_current_credit_spread(df)
        assert result["credit_regime"] in {"STRESS", "CALM", "NORMAL"}

    def test_returns_none_without_features(self):
        feat = CreditSpreadFeatures()
        df = pd.DataFrame({"close": [100]})
        assert feat.analyze_current_credit_spread(df) is None


# ─── Feature Count Test ─────────────────────────────────────────────────

class TestFeatureCounts:
    def test_total_count(self):
        assert len(ALL_8) == 8
