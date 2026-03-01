"""Tests for MacroSurpriseFeatures -- 10 macro-surprise features (prefix msurp_)."""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.macro_surprise_features import MacroSurpriseFeatures


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
    "msurp_citi_surprise_proxy",
    "msurp_nfp_surprise_proxy",
    "msurp_cpi_surprise_proxy",
    "msurp_ism_surprise_proxy",
    "msurp_composite_z",
    "msurp_positive_surprise_streak",
    "msurp_surprise_momentum_5d",
    "msurp_surprise_momentum_20d",
    "msurp_surprise_vol",
    "msurp_regime",
}


# ─── Invariant Tests ────────────────────────────────────────────────────

class TestMacroSurpriseInvariants:
    @pytest.fixture
    def feat(self):
        return MacroSurpriseFeatures()

    @pytest.fixture
    def spy(self):
        return _make_spy_daily(200)

    def test_all_10_features_created(self, feat, spy):
        result = feat.create_macro_surprise_features(spy)
        msurp_cols = {c for c in result.columns if c.startswith("msurp_")}
        assert msurp_cols == ALL_10

    def test_no_nans(self, feat, spy):
        result = feat.create_macro_surprise_features(spy)
        for col in ALL_10:
            assert result[col].isna().sum() == 0, f"NaN in {col}"

    def test_no_infinities(self, feat, spy):
        result = feat.create_macro_surprise_features(spy)
        for col in ALL_10:
            assert not np.isinf(result[col]).any(), f"Inf in {col}"

    def test_preserves_original_columns(self, feat, spy):
        original = set(spy.columns)
        result = feat.create_macro_surprise_features(spy)
        assert original.issubset(set(result.columns))

    def test_preserves_row_count(self, feat, spy):
        result = feat.create_macro_surprise_features(spy)
        assert len(result) == len(spy)

    def test_no_required_column(self, feat):
        """Missing 'close' should return df unchanged, not crash."""
        df = pd.DataFrame({"date": [1, 2], "price": [100, 101]})
        result = feat.create_macro_surprise_features(df)
        assert len(result.columns) == len(df.columns)

    def test_works_with_flat_data(self, feat):
        """Flat close prices (zero variance) should not crash."""
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=50),
            "close": np.full(50, 450.0),
            "volume": np.full(50, 100_000_000.0),
        })
        result = feat.create_macro_surprise_features(df)
        for col in ALL_10:
            assert col in result.columns
            assert result[col].isna().sum() == 0, f"NaN in {col} with flat data"


# ─── Feature Logic Tests ───────────────────────────────────────────────

class TestMacroSurpriseLogic:
    @pytest.fixture
    def feat(self):
        return MacroSurpriseFeatures()

    def test_composite_z_bounded(self, feat):
        """Composite z-score should be clipped to [-4, 4]."""
        df = _make_spy_daily(200)
        result = feat.create_macro_surprise_features(df)
        assert result["msurp_composite_z"].max() <= 4.0
        assert result["msurp_composite_z"].min() >= -4.0

    def test_regime_values(self, feat):
        """Regime should only be -1.0, 0.0, or 1.0."""
        df = _make_spy_daily(200)
        result = feat.create_macro_surprise_features(df)
        unique = set(result["msurp_regime"].unique())
        assert unique.issubset({-1.0, 0.0, 1.0})

    def test_streak_non_negative(self, feat):
        """Positive surprise streak should be >= 0."""
        df = _make_spy_daily(200)
        result = feat.create_macro_surprise_features(df)
        assert (result["msurp_positive_surprise_streak"] >= 0).all()

    def test_nfp_surprise_bounded(self, feat):
        """NFP surprise z-score should be clipped to [-4, 4]."""
        df = _make_spy_daily(200)
        result = feat.create_macro_surprise_features(df)
        assert result["msurp_nfp_surprise_proxy"].max() <= 4.0
        assert result["msurp_nfp_surprise_proxy"].min() >= -4.0

    def test_works_without_volume(self, feat):
        """Should work even without volume column (uses abs return fallback)."""
        df = _make_spy_daily(200).drop(columns=["volume"])
        result = feat.create_macro_surprise_features(df)
        msurp_cols = {c for c in result.columns if c.startswith("msurp_")}
        assert msurp_cols == ALL_10


# ─── Analyze Tests ──────────────────────────────────────────────────────

class TestAnalyzeCurrentMacroSurprise:
    def test_returns_dict(self):
        feat = MacroSurpriseFeatures()
        df = _make_spy_daily(100)
        df = feat.create_macro_surprise_features(df)
        result = feat.analyze_current_macro_surprise(df)
        assert isinstance(result, dict)
        assert "surprise_regime" in result

    def test_regime_values(self):
        feat = MacroSurpriseFeatures()
        df = _make_spy_daily(200)
        df = feat.create_macro_surprise_features(df)
        result = feat.analyze_current_macro_surprise(df)
        assert result["surprise_regime"] in {"POSITIVE", "NEGATIVE", "NEUTRAL"}

    def test_returns_none_without_features(self):
        feat = MacroSurpriseFeatures()
        df = pd.DataFrame({"close": [100]})
        assert feat.analyze_current_macro_surprise(df) is None


# ─── Feature Count Test ─────────────────────────────────────────────────

class TestFeatureCounts:
    def test_total_count(self):
        assert len(ALL_10) == 10
