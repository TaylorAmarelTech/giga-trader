"""Tests for DriftFeatures — CUSUM-based drift detection features (3 total)."""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.drift_features import DriftFeatures


# ─── Helpers ────────────────────────────────────────────────────────────

def _make_spy_daily(n_days: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    returns = rng.normal(0.0003, 0.012, n_days)
    close = 450.0 * np.cumprod(1 + returns)
    return pd.DataFrame({
        "date": dates, "close": close,
        "open": close * (1 + rng.normal(0, 0.003, n_days)),
        "high": close * 1.005, "low": close * 0.995,
    })


ALL_3 = {"drift_detected", "drift_days_since", "drift_window_size"}


# ─── Invariant Tests ────────────────────────────────────────────────────

class TestDriftInvariants:
    @pytest.fixture
    def feat(self):
        return DriftFeatures()

    @pytest.fixture
    def spy(self):
        return _make_spy_daily(200)

    def test_all_3_features_created(self, feat, spy):
        result = feat.create_drift_features(spy)
        drift_cols = {c for c in result.columns if c.startswith("drift_")}
        assert drift_cols == ALL_3

    def test_no_nans(self, feat, spy):
        result = feat.create_drift_features(spy)
        for col in ALL_3:
            assert result[col].isna().sum() == 0, f"NaN in {col}"

    def test_no_infinities(self, feat, spy):
        result = feat.create_drift_features(spy)
        for col in ALL_3:
            assert not np.isinf(result[col]).any(), f"Inf in {col}"

    def test_preserves_original_columns(self, feat, spy):
        original = set(spy.columns)
        result = feat.create_drift_features(spy)
        assert original.issubset(set(result.columns))

    def test_preserves_row_count(self, feat, spy):
        result = feat.create_drift_features(spy)
        assert len(result) == len(spy)

    def test_no_close_column(self, feat):
        df = pd.DataFrame({"date": [1, 2], "price": [100, 101]})
        result = feat.create_drift_features(df)
        assert len(result.columns) == len(df.columns)


# ─── Feature Logic Tests ───────────────────────────────────────────────

class TestDriftLogic:
    @pytest.fixture
    def feat(self):
        return DriftFeatures()

    def test_detected_is_binary(self, feat):
        df = _make_spy_daily(200)
        result = feat.create_drift_features(df)
        vals = set(result["drift_detected"].unique())
        assert vals.issubset({0.0, 1.0})

    def test_days_since_non_negative(self, feat):
        df = _make_spy_daily(200)
        result = feat.create_drift_features(df)
        assert (result["drift_days_since"] >= 0).all()

    def test_days_since_capped_at_252(self, feat):
        df = _make_spy_daily(300)
        result = feat.create_drift_features(df)
        assert result["drift_days_since"].max() <= 252

    def test_window_size_bounded(self, feat):
        df = _make_spy_daily(300)
        result = feat.create_drift_features(df)
        assert result["drift_window_size"].min() >= 0.0
        assert result["drift_window_size"].max() <= 5.0


# ─── Analyze Tests ──────────────────────────────────────────────────────

class TestAnalyze:
    def test_returns_dict(self):
        feat = DriftFeatures()
        df = _make_spy_daily(100)
        df = feat.create_drift_features(df)
        result = feat.analyze_current_drift(df)
        assert isinstance(result, dict)
        assert "drift_regime" in result

    def test_regime_values(self):
        feat = DriftFeatures()
        df = _make_spy_daily(200)
        df = feat.create_drift_features(df)
        result = feat.analyze_current_drift(df)
        assert result["drift_regime"] in {"STABLE", "TRANSITIONING", "DRIFTING"}

    def test_returns_none_without_features(self):
        feat = DriftFeatures()
        df = pd.DataFrame({"close": [100]})
        assert feat.analyze_current_drift(df) is None


# ─── Feature Count Test ─────────────────────────────────────────────────

class TestFeatureCounts:
    def test_total_count(self):
        assert len(ALL_3) == 3
