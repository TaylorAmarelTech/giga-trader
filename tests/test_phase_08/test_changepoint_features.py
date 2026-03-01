"""Tests for ChangepointFeatures — Bayesian Online Changepoint Detection (3 total)."""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.changepoint_features import ChangepointFeatures


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


ALL_3 = {"cpd_run_length", "cpd_prob_change", "cpd_regime_id"}


# ─── Invariant Tests ────────────────────────────────────────────────────

class TestChangepointInvariants:
    @pytest.fixture
    def feat(self):
        return ChangepointFeatures()

    @pytest.fixture
    def spy(self):
        return _make_spy_daily(200)

    def test_all_3_features_created(self, feat, spy):
        result = feat.create_changepoint_features(spy)
        cpd_cols = {c for c in result.columns if c.startswith("cpd_")}
        assert cpd_cols == ALL_3

    def test_no_nans(self, feat, spy):
        result = feat.create_changepoint_features(spy)
        for col in ALL_3:
            assert result[col].isna().sum() == 0, f"NaN in {col}"

    def test_no_infinities(self, feat, spy):
        result = feat.create_changepoint_features(spy)
        for col in ALL_3:
            assert not np.isinf(result[col]).any(), f"Inf in {col}"

    def test_preserves_original_columns(self, feat, spy):
        original = set(spy.columns)
        result = feat.create_changepoint_features(spy)
        assert original.issubset(set(result.columns))

    def test_preserves_row_count(self, feat, spy):
        result = feat.create_changepoint_features(spy)
        assert len(result) == len(spy)

    def test_no_close_column(self, feat):
        df = pd.DataFrame({"date": [1, 2], "price": [100, 101]})
        result = feat.create_changepoint_features(df)
        assert len(result.columns) == len(df.columns)


# ─── Logic Tests ─────────────────────────────────────────────────────────

class TestChangepointLogic:
    @pytest.fixture
    def feat(self):
        return ChangepointFeatures()

    def test_run_length_non_negative(self, feat):
        df = _make_spy_daily(200)
        result = feat.create_changepoint_features(df)
        assert (result["cpd_run_length"] >= 0).all()

    def test_run_length_capped_at_252(self, feat):
        df = _make_spy_daily(500)
        result = feat.create_changepoint_features(df)
        assert result["cpd_run_length"].max() <= 252

    def test_prob_change_bounded(self, feat):
        df = _make_spy_daily(200)
        result = feat.create_changepoint_features(df)
        assert (result["cpd_prob_change"] >= 0.0).all()
        assert (result["cpd_prob_change"] <= 1.0).all()

    def test_regime_id_bounded(self, feat):
        df = _make_spy_daily(500, seed=99)
        result = feat.create_changepoint_features(df)
        assert (result["cpd_regime_id"] >= 0).all()
        assert (result["cpd_regime_id"] <= 9).all()


# ─── Analyze Tests ──────────────────────────────────────────────────────

class TestAnalyze:
    def test_returns_dict(self):
        feat = ChangepointFeatures()
        df = _make_spy_daily(100)
        df = feat.create_changepoint_features(df)
        result = feat.analyze_current_changepoint(df)
        assert isinstance(result, dict)

    def test_keys_present(self):
        feat = ChangepointFeatures()
        df = _make_spy_daily(100)
        df = feat.create_changepoint_features(df)
        result = feat.analyze_current_changepoint(df)
        assert "run_length" in result
        assert "prob_change" in result
        assert "regime_id" in result

    def test_returns_none_without_features(self):
        feat = ChangepointFeatures()
        df = pd.DataFrame({"close": [100]})
        assert feat.analyze_current_changepoint(df) is None


# ─── Feature Count Test ─────────────────────────────────────────────────

class TestFeatureCounts:
    def test_total_count(self):
        assert len(ALL_3) == 3
