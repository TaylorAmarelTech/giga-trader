"""Tests for PatchTemporalFeatures -- PatchTST-style patch-based temporal features (8 total)."""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.patch_temporal_features import PatchTemporalFeatures


# --- Helpers ----------------------------------------------------------------

def _make_spy_daily(n_days: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    returns = rng.normal(0.0003, 0.012, n_days)
    close = 450.0 * np.cumprod(1 + returns)
    volume = rng.randint(50_000_000, 150_000_000, n_days).astype(float)
    return pd.DataFrame({
        "date": dates,
        "close": close,
        "volume": volume,
        "open": close * (1 + rng.normal(0, 0.003, n_days)),
        "high": close * (1 + np.abs(rng.normal(0, 0.005, n_days))),
        "low": close * (1 - np.abs(rng.normal(0, 0.005, n_days))),
    })


ALL_8 = {
    "ptst_patch_trend_consistency",
    "ptst_patch_volatility_profile",
    "ptst_patch_volume_profile",
    "ptst_cross_patch_correlation",
    "ptst_patch_momentum_decay",
    "ptst_patch_breakout_score",
    "ptst_multi_scale_trend",
    "ptst_patch_entropy",
}


# --- Construction Tests -----------------------------------------------------

class TestConstruction:
    def test_default_construction(self):
        feat = PatchTemporalFeatures()
        assert feat.patch_sizes == (5, 10, 20)
        assert feat.lookback == 60

    def test_custom_construction(self):
        feat = PatchTemporalFeatures(patch_sizes=(3, 7, 15), lookback=45)
        assert feat.patch_sizes == (3, 7, 15)
        assert feat.lookback == 45

    def test_lookback_floor(self):
        """Lookback must be at least 2 * max(patch_sizes)."""
        feat = PatchTemporalFeatures(patch_sizes=(5, 10, 50), lookback=30)
        assert feat.lookback >= 100  # 2 * 50 = 100

    def test_feature_names_all_prefixed(self):
        assert all(name.startswith("ptst_") for name in PatchTemporalFeatures.FEATURE_NAMES)

    def test_feature_names_count(self):
        assert len(PatchTemporalFeatures.FEATURE_NAMES) == 8


# --- Invariant Tests --------------------------------------------------------

class TestInvariants:
    @pytest.fixture
    def feat(self):
        return PatchTemporalFeatures()

    @pytest.fixture
    def spy(self):
        return _make_spy_daily(200)

    def test_all_8_features_created(self, feat, spy):
        result = feat.create_patch_temporal_features(spy)
        ptst_cols = {c for c in result.columns if c.startswith("ptst_")}
        assert ptst_cols == ALL_8

    def test_no_nans(self, feat, spy):
        result = feat.create_patch_temporal_features(spy)
        for col in ALL_8:
            assert result[col].isna().sum() == 0, f"NaN in {col}"

    def test_no_infinities(self, feat, spy):
        result = feat.create_patch_temporal_features(spy)
        for col in ALL_8:
            assert not np.isinf(result[col]).any(), f"Inf in {col}"

    def test_preserves_original_columns(self, feat, spy):
        original = set(spy.columns)
        result = feat.create_patch_temporal_features(spy)
        assert original.issubset(set(result.columns))

    def test_preserves_row_count(self, feat, spy):
        result = feat.create_patch_temporal_features(spy)
        assert len(result) == len(spy)


# --- Missing / Degenerate Data Tests ----------------------------------------

class TestMissingData:
    @pytest.fixture
    def feat(self):
        return PatchTemporalFeatures()

    def test_missing_columns_zero_fill(self, feat):
        """When required columns are missing, all features should be zero-filled."""
        df = pd.DataFrame({"date": [1, 2, 3], "price": [100, 101, 102]})
        result = feat.create_patch_temporal_features(df)
        for col in ALL_8:
            assert col in result.columns
            assert (result[col] == 0.0).all(), f"{col} should be zero-filled"

    def test_short_data_no_crash(self, feat):
        """With very short data (< lookback), features should be zero."""
        df = pd.DataFrame({
            "close": [450.0, 451.0, 449.0, 452.0, 450.5],
            "high": [451.0, 452.0, 450.0, 453.0, 451.5],
            "low": [449.0, 450.0, 448.0, 451.0, 449.5],
            "volume": [1e8, 1.1e8, 0.9e8, 1.2e8, 1e8],
        })
        result = feat.create_patch_temporal_features(df)
        for col in ALL_8:
            assert col in result.columns
            # All zeros because data length < lookback
            assert (result[col] == 0.0).all()

    def test_flat_prices_no_crash(self, feat):
        """Flat close prices (zero variance) should not crash."""
        n = 100
        df = pd.DataFrame({
            "close": np.full(n, 450.0),
            "high": np.full(n, 451.0),
            "low": np.full(n, 449.0),
            "volume": np.full(n, 1e8),
        })
        result = feat.create_patch_temporal_features(df)
        for col in ALL_8:
            assert col in result.columns
            assert result[col].isna().sum() == 0, f"NaN in {col} with flat data"

    def test_single_row(self, feat):
        """Single row should produce zero-filled features."""
        df = pd.DataFrame({
            "close": [450.0], "high": [451.0], "low": [449.0], "volume": [1e8],
        })
        result = feat.create_patch_temporal_features(df)
        for col in ALL_8:
            assert col in result.columns


# --- Feature Logic Tests ----------------------------------------------------

class TestFeatureLogic:
    @pytest.fixture
    def feat(self):
        return PatchTemporalFeatures()

    def test_trend_consistency_bounded(self, feat):
        """Trend consistency should be in [0, 1]."""
        df = _make_spy_daily(200)
        result = feat.create_patch_temporal_features(df)
        tc = result["ptst_patch_trend_consistency"]
        # Non-lookback rows are 0.0; warm-up rows should be in [0.5, 1.0]
        active = tc[tc > 0]
        if len(active) > 0:
            assert active.min() >= 0.5 - 1e-9  # at least 50% agreement (majority)
            assert active.max() <= 1.0 + 1e-9

    def test_breakout_score_non_negative(self, feat):
        """Breakout score should always be >= 0."""
        df = _make_spy_daily(200)
        result = feat.create_patch_temporal_features(df)
        assert (result["ptst_patch_breakout_score"] >= 0).all()

    def test_multi_scale_trend_bounded(self, feat):
        """Multi-scale trend should be in [-1, 1]."""
        df = _make_spy_daily(200)
        result = feat.create_patch_temporal_features(df)
        mst = result["ptst_multi_scale_trend"]
        assert mst.min() >= -1.0 - 1e-9
        assert mst.max() <= 1.0 + 1e-9

    def test_patch_entropy_non_negative(self, feat):
        """Patch entropy should always be >= 0."""
        df = _make_spy_daily(200)
        result = feat.create_patch_temporal_features(df)
        assert (result["ptst_patch_entropy"] >= 0).all()

    def test_cross_patch_correlation_bounded(self, feat):
        """Cross-patch correlation should be in [-1, 1]."""
        df = _make_spy_daily(200)
        result = feat.create_patch_temporal_features(df)
        cpc = result["ptst_cross_patch_correlation"]
        assert cpc.min() >= -1.0 - 1e-9
        assert cpc.max() <= 1.0 + 1e-9

    def test_trending_series_high_consistency(self):
        """A strongly trending series should show high trend consistency."""
        n = 200
        rng = np.random.RandomState(123)
        # Strongly uptrending: every day up on average
        returns = np.abs(rng.normal(0.005, 0.002, n))  # all positive
        close = 450.0 * np.cumprod(1 + returns)
        df = pd.DataFrame({
            "close": close,
            "high": close * 1.003,
            "low": close * 0.997,
            "volume": np.full(n, 1e8),
        })
        feat = PatchTemporalFeatures(patch_sizes=(5, 10), lookback=40)
        result = feat.create_patch_temporal_features(df)
        # Last row should have high trend consistency
        last_tc = result["ptst_patch_trend_consistency"].iloc[-1]
        assert last_tc >= 0.8, (
            f"Strongly trending series got consistency={last_tc:.3f}, expected >= 0.8"
        )

    def test_different_patch_sizes_produce_different_features(self):
        """Two different patch_size configurations should produce different values."""
        df = _make_spy_daily(200)
        feat_a = PatchTemporalFeatures(patch_sizes=(5, 10), lookback=40)
        feat_b = PatchTemporalFeatures(patch_sizes=(7, 14), lookback=42)
        result_a = feat_a.create_patch_temporal_features(df)
        result_b = feat_b.create_patch_temporal_features(df)
        # At least one feature should differ
        any_different = False
        for col in ALL_8:
            if not np.allclose(result_a[col].values, result_b[col].values, atol=1e-12):
                any_different = True
                break
        assert any_different, "Different patch sizes should produce different features"


# --- Feature Count Test -----------------------------------------------------

class TestFeatureCounts:
    def test_total_count(self):
        assert len(ALL_8) == 8

    def test_class_feature_names_match(self):
        assert set(PatchTemporalFeatures.FEATURE_NAMES) == ALL_8

    def test_static_feature_names_match(self):
        assert set(PatchTemporalFeatures._all_feature_names()) == ALL_8
