"""Tests for EntropyFeatures — information-theoretic entropy features (6 total)."""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.entropy_features import EntropyFeatures


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


ALL_6 = {
    "ent_shannon_20d", "ent_permutation_20d", "ent_sample_20d",
    "ent_shannon_z", "ent_regime_change", "ent_predictability",
}


# ─── Invariant Tests ────────────────────────────────────────────────────

class TestEntropyInvariants:
    @pytest.fixture
    def feat(self):
        return EntropyFeatures()

    @pytest.fixture
    def spy(self):
        return _make_spy_daily(200)

    def test_all_6_features_created(self, feat, spy):
        result = feat.create_entropy_features(spy)
        ent_cols = {c for c in result.columns if c.startswith("ent_")}
        assert ent_cols == ALL_6

    def test_no_nans(self, feat, spy):
        result = feat.create_entropy_features(spy)
        for col in ALL_6:
            assert result[col].isna().sum() == 0, f"NaN in {col}"

    def test_no_infinities(self, feat, spy):
        result = feat.create_entropy_features(spy)
        for col in ALL_6:
            assert not np.isinf(result[col]).any(), f"Inf in {col}"

    def test_preserves_original_columns(self, feat, spy):
        original = set(spy.columns)
        result = feat.create_entropy_features(spy)
        assert original.issubset(set(result.columns))

    def test_preserves_row_count(self, feat, spy):
        result = feat.create_entropy_features(spy)
        assert len(result) == len(spy)

    def test_no_close_column(self, feat):
        df = pd.DataFrame({"date": [1, 2], "price": [100, 101]})
        result = feat.create_entropy_features(df)
        assert len(result.columns) == len(df.columns)

    def test_works_with_flat_data(self, feat):
        """Flat close prices (zero variance) should not crash."""
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=50),
            "close": np.full(50, 450.0),
        })
        result = feat.create_entropy_features(df)
        for col in ALL_6:
            assert col in result.columns
            assert result[col].isna().sum() == 0, f"NaN in {col} with flat data"


# ─── Feature Logic Tests ───────────────────────────────────────────────

class TestEntropyLogic:
    @pytest.fixture
    def feat(self):
        return EntropyFeatures()

    def test_entropy_non_negative(self, feat):
        df = _make_spy_daily(100)
        result = feat.create_entropy_features(df)
        assert (result["ent_shannon_20d"] >= 0).all()
        assert (result["ent_permutation_20d"] >= 0).all()

    def test_high_vol_higher_sample_entropy(self, feat):
        """Higher volatility returns should produce higher sample entropy proxy."""
        rng = np.random.RandomState(99)
        dates = pd.bdate_range("2024-01-02", periods=100, freq="B")

        # Low vol — very tight returns
        low_ret = rng.normal(0, 0.001, 100)
        low_close = 450.0 * np.cumprod(1 + low_ret)
        df_low = pd.DataFrame({"date": dates, "close": low_close})
        r_low = feat.create_entropy_features(df_low)

        # High vol — much wider returns
        high_ret = np.random.RandomState(77).normal(0, 0.05, 100)
        high_close = 450.0 * np.cumprod(1 + high_ret)
        df_high = pd.DataFrame({"date": dates, "close": high_close})
        r_high = feat.create_entropy_features(df_high)

        # Sample entropy proxy (std of return diffs) should be higher for high-vol
        low_mean = r_low["ent_sample_20d"].iloc[30:].mean()
        high_mean = r_high["ent_sample_20d"].iloc[30:].mean()
        assert high_mean > low_mean

    def test_shannon_z_bounded(self, feat):
        df = _make_spy_daily(200)
        result = feat.create_entropy_features(df)
        assert result["ent_shannon_z"].max() <= 4.0
        assert result["ent_shannon_z"].min() >= -4.0

    def test_regime_change_non_negative(self, feat):
        df = _make_spy_daily(200)
        result = feat.create_entropy_features(df)
        assert (result["ent_regime_change"] >= 0).all()

    def test_predictability_bounded_0_1(self, feat):
        df = _make_spy_daily(200)
        result = feat.create_entropy_features(df)
        assert (result["ent_predictability"] >= 0).all()
        assert (result["ent_predictability"] <= 1).all()


# ─── Analyze Tests ──────────────────────────────────────────────────────

class TestAnalyzeCurrentEntropy:
    def test_returns_dict(self):
        feat = EntropyFeatures()
        df = _make_spy_daily(100)
        df = feat.create_entropy_features(df)
        result = feat.analyze_current_entropy(df)
        assert isinstance(result, dict)
        assert "entropy_regime" in result

    def test_regime_values(self):
        feat = EntropyFeatures()
        df = _make_spy_daily(200)
        df = feat.create_entropy_features(df)
        result = feat.analyze_current_entropy(df)
        assert result["entropy_regime"] in {"HIGH_ENTROPY", "LOW_ENTROPY", "NORMAL"}

    def test_returns_none_without_features(self):
        feat = EntropyFeatures()
        df = pd.DataFrame({"close": [100]})
        assert feat.analyze_current_entropy(df) is None


# ─── Feature Count Test ─────────────────────────────────────────────────

class TestFeatureCounts:
    def test_total_count(self):
        assert len(ALL_6) == 6
