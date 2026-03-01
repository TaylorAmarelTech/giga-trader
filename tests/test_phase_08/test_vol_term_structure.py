"""Tests for VolTermStructureFeatures -- 8 VIX term-structure features (prefix vts_)."""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.vol_term_structure_features import VolTermStructureFeatures


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
    "vts_vix_vix3m_ratio",
    "vts_contango",
    "vts_backwardation",
    "vts_term_slope",
    "vts_term_slope_z",
    "vts_roll_yield_proxy",
    "vts_ratio_momentum_5d",
    "vts_regime",
}


# ─── Invariant Tests ────────────────────────────────────────────────────

class TestVolTermStructureInvariants:
    @pytest.fixture
    def feat(self):
        return VolTermStructureFeatures()

    @pytest.fixture
    def spy(self):
        return _make_spy_daily(200)

    def test_all_8_features_created(self, feat, spy):
        result = feat.create_vol_term_structure_features(spy)
        vts_cols = {c for c in result.columns if c.startswith("vts_")}
        assert vts_cols == ALL_8

    def test_no_nans(self, feat, spy):
        result = feat.create_vol_term_structure_features(spy)
        for col in ALL_8:
            assert result[col].isna().sum() == 0, f"NaN in {col}"

    def test_no_infinities(self, feat, spy):
        result = feat.create_vol_term_structure_features(spy)
        for col in ALL_8:
            assert not np.isinf(result[col]).any(), f"Inf in {col}"

    def test_preserves_original_columns(self, feat, spy):
        original = set(spy.columns)
        result = feat.create_vol_term_structure_features(spy)
        assert original.issubset(set(result.columns))

    def test_preserves_row_count(self, feat, spy):
        result = feat.create_vol_term_structure_features(spy)
        assert len(result) == len(spy)

    def test_no_required_column(self, feat):
        """Missing 'close' should return df unchanged, not crash."""
        df = pd.DataFrame({"date": [1, 2], "price": [100, 101]})
        result = feat.create_vol_term_structure_features(df)
        assert len(result.columns) == len(df.columns)

    def test_works_with_flat_data(self, feat):
        """Flat close prices (zero variance) should not crash."""
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=50),
            "close": np.full(50, 450.0),
        })
        result = feat.create_vol_term_structure_features(df)
        for col in ALL_8:
            assert col in result.columns
            assert result[col].isna().sum() == 0, f"NaN in {col} with flat data"


# ─── Feature Logic Tests ───────────────────────────────────────────────

class TestVolTermStructureLogic:
    @pytest.fixture
    def feat(self):
        return VolTermStructureFeatures()

    def test_term_slope_z_bounded(self, feat):
        """Z-score should be clipped to [-4, 4]."""
        df = _make_spy_daily(200)
        result = feat.create_vol_term_structure_features(df)
        assert result["vts_term_slope_z"].max() <= 4.0
        assert result["vts_term_slope_z"].min() >= -4.0

    def test_contango_backwardation_exclusive(self, feat):
        """Contango and backwardation flags should never both be 1.0 on the same row."""
        df = _make_spy_daily(200)
        result = feat.create_vol_term_structure_features(df)
        both_set = (result["vts_contango"] == 1.0) & (result["vts_backwardation"] == 1.0)
        assert not both_set.any(), "Contango and backwardation are both 1.0"

    def test_regime_valid_values(self, feat):
        """Regime should only contain valid encoded values."""
        df = _make_spy_daily(200)
        result = feat.create_vol_term_structure_features(df)
        unique = set(result["vts_regime"].unique())
        assert unique.issubset({-2.0, -1.0, 0.0, 1.0, 2.0})


# ─── Analyze Tests ──────────────────────────────────────────────────────

class TestAnalyzeCurrentVolTermStructure:
    def test_returns_dict(self):
        feat = VolTermStructureFeatures()
        df = _make_spy_daily(100)
        df = feat.create_vol_term_structure_features(df)
        result = feat.analyze_current_vol_term_structure(df)
        assert isinstance(result, dict)
        assert "vol_term_regime" in result

    def test_regime_values(self):
        feat = VolTermStructureFeatures()
        df = _make_spy_daily(200)
        df = feat.create_vol_term_structure_features(df)
        result = feat.analyze_current_vol_term_structure(df)
        assert result["vol_term_regime"] in {
            "CONTANGO", "NORMAL", "BACKWARDATION",
        }

    def test_returns_none_without_features(self):
        feat = VolTermStructureFeatures()
        df = pd.DataFrame({"close": [100]})
        assert feat.analyze_current_vol_term_structure(df) is None


# ─── Feature Count Test ─────────────────────────────────────────────────

class TestFeatureCounts:
    def test_total_count(self):
        assert len(ALL_8) == 8
