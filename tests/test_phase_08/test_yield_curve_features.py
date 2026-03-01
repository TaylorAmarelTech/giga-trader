"""Tests for YieldCurveFeatures -- 10 yield-curve features (prefix yc_)."""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.yield_curve_features import YieldCurveFeatures


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


ALL_10 = {
    "yc_2s10s_slope",
    "yc_3m10y_slope",
    "yc_curvature",
    "yc_slope_momentum_5d",
    "yc_slope_momentum_20d",
    "yc_slope_z",
    "yc_real_yield_proxy",
    "yc_inversion_flag",
    "yc_steepening_speed",
    "yc_regime",
}


# ─── Invariant Tests ────────────────────────────────────────────────────

class TestYieldCurveInvariants:
    @pytest.fixture
    def feat(self):
        return YieldCurveFeatures()

    @pytest.fixture
    def spy(self):
        return _make_spy_daily(200)

    def test_all_10_features_created(self, feat, spy):
        result = feat.create_yield_curve_features(spy)
        yc_cols = {c for c in result.columns if c.startswith("yc_")}
        assert yc_cols == ALL_10

    def test_no_nans(self, feat, spy):
        result = feat.create_yield_curve_features(spy)
        for col in ALL_10:
            assert result[col].isna().sum() == 0, f"NaN in {col}"

    def test_no_infinities(self, feat, spy):
        result = feat.create_yield_curve_features(spy)
        for col in ALL_10:
            assert not np.isinf(result[col]).any(), f"Inf in {col}"

    def test_preserves_original_columns(self, feat, spy):
        original = set(spy.columns)
        result = feat.create_yield_curve_features(spy)
        assert original.issubset(set(result.columns))

    def test_preserves_row_count(self, feat, spy):
        result = feat.create_yield_curve_features(spy)
        assert len(result) == len(spy)

    def test_no_required_column(self, feat):
        """Missing 'close' should return df unchanged, not crash."""
        df = pd.DataFrame({"date": [1, 2], "price": [100, 101]})
        result = feat.create_yield_curve_features(df)
        assert len(result.columns) == len(df.columns)

    def test_works_with_flat_data(self, feat):
        """Flat close prices (zero variance) should not crash."""
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=50),
            "close": np.full(50, 450.0),
        })
        result = feat.create_yield_curve_features(df)
        for col in ALL_10:
            assert col in result.columns
            assert result[col].isna().sum() == 0, f"NaN in {col} with flat data"


# ─── Feature Logic Tests ───────────────────────────────────────────────

class TestYieldCurveLogic:
    @pytest.fixture
    def feat(self):
        return YieldCurveFeatures()

    def test_slope_z_bounded(self, feat):
        """Z-score should be clipped to [-4, 4]."""
        df = _make_spy_daily(200)
        result = feat.create_yield_curve_features(df)
        assert result["yc_slope_z"].max() <= 4.0
        assert result["yc_slope_z"].min() >= -4.0

    def test_real_yield_proxy_bounded(self, feat):
        """Real yield proxy z-score should be clipped to [-4, 4]."""
        df = _make_spy_daily(200)
        result = feat.create_yield_curve_features(df)
        assert result["yc_real_yield_proxy"].max() <= 4.0
        assert result["yc_real_yield_proxy"].min() >= -4.0

    def test_inversion_flag_binary(self, feat):
        """Inversion flag should only be 0.0 or 1.0."""
        df = _make_spy_daily(200)
        result = feat.create_yield_curve_features(df)
        unique = set(result["yc_inversion_flag"].unique())
        assert unique.issubset({0.0, 1.0})

    def test_regime_valid_values(self, feat):
        """Regime should only be -1.0, 0.0, 1.0, or 2.0."""
        df = _make_spy_daily(200)
        result = feat.create_yield_curve_features(df)
        unique = set(result["yc_regime"].unique())
        assert unique.issubset({-1.0, 0.0, 1.0, 2.0})


# ─── Analyze Tests ──────────────────────────────────────────────────────

class TestAnalyzeCurrentYieldCurve:
    def test_returns_dict(self):
        feat = YieldCurveFeatures()
        df = _make_spy_daily(100)
        df = feat.create_yield_curve_features(df)
        result = feat.analyze_current_yield_curve(df)
        assert isinstance(result, dict)
        assert "yield_regime" in result

    def test_regime_values(self):
        feat = YieldCurveFeatures()
        df = _make_spy_daily(200)
        df = feat.create_yield_curve_features(df)
        result = feat.analyze_current_yield_curve(df)
        assert result["yield_regime"] in {"STEEP", "NORMAL", "FLAT", "INVERTED"}

    def test_returns_none_without_features(self):
        feat = YieldCurveFeatures()
        df = pd.DataFrame({"close": [100]})
        assert feat.analyze_current_yield_curve(df) is None


# ─── Feature Count Test ─────────────────────────────────────────────────

class TestFeatureCounts:
    def test_total_count(self):
        assert len(ALL_10) == 10
