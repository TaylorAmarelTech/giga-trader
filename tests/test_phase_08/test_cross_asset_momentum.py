"""Tests for CrossAssetMomentumFeatures -- cross-asset leading signals (12 total)."""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.cross_asset_momentum_features import (
    CrossAssetMomentumFeatures,
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


ALL_12 = {
    "xmom_tlt_lead_5d",
    "xmom_gld_lead_5d",
    "xmom_hyg_lead_5d",
    "xmom_tlt_lead_tstat",
    "xmom_gld_lead_tstat",
    "xmom_hyg_lead_tstat",
    "xmom_cross_momentum_composite",
    "xmom_cross_divergence",
    "xmom_tlt_spy_corr_20d",
    "xmom_gld_spy_corr_20d",
    "xmom_corr_regime_change",
    "xmom_momentum_regime",
}


# ---- Invariant Tests --------------------------------------------------------

class TestCrossAssetMomentumInvariants:
    @pytest.fixture
    def feat(self):
        return CrossAssetMomentumFeatures()

    @pytest.fixture
    def spy(self):
        return _make_spy_daily(200)

    def test_all_12_features_created(self, feat, spy):
        result = feat.create_cross_asset_momentum_features(spy)
        xmom_cols = {c for c in result.columns if c.startswith("xmom_")}
        assert xmom_cols == ALL_12

    def test_no_nans(self, feat, spy):
        result = feat.create_cross_asset_momentum_features(spy)
        for col in ALL_12:
            assert result[col].isna().sum() == 0, f"NaN in {col}"

    def test_no_infinities(self, feat, spy):
        result = feat.create_cross_asset_momentum_features(spy)
        for col in ALL_12:
            assert not np.isinf(result[col]).any(), f"Inf in {col}"

    def test_preserves_original_columns(self, feat, spy):
        original = set(spy.columns)
        result = feat.create_cross_asset_momentum_features(spy)
        assert original.issubset(set(result.columns))

    def test_preserves_row_count(self, feat, spy):
        result = feat.create_cross_asset_momentum_features(spy)
        assert len(result) == len(spy)

    def test_no_required_column(self, feat):
        df = pd.DataFrame({"date": [1, 2], "price": [100, 101]})
        result = feat.create_cross_asset_momentum_features(df)
        assert len(result.columns) == len(df.columns)

    def test_works_with_flat_data(self, feat):
        """Flat close prices should not crash."""
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=50),
            "close": np.full(50, 450.0),
        })
        result = feat.create_cross_asset_momentum_features(df)
        for col in ALL_12:
            assert col in result.columns
            assert result[col].isna().sum() == 0, f"NaN in {col} with flat data"


# ---- Feature Logic Tests ----------------------------------------------------

class TestCrossAssetMomentumLogic:
    @pytest.fixture
    def feat(self):
        return CrossAssetMomentumFeatures()

    def test_correlations_bounded(self, feat):
        """Correlation features should be bounded in [-1, 1]."""
        df = _make_spy_daily(200)
        result = feat.create_cross_asset_momentum_features(df)
        for col in ["xmom_tlt_lead_5d", "xmom_gld_lead_5d", "xmom_hyg_lead_5d",
                     "xmom_tlt_spy_corr_20d", "xmom_gld_spy_corr_20d"]:
            assert result[col].min() >= -1.01, f"{col} below -1"
            assert result[col].max() <= 1.01, f"{col} above 1"

    def test_momentum_regime_values(self, feat):
        """Regime should only be -1.0, 0.0, or 1.0."""
        df = _make_spy_daily(200)
        result = feat.create_cross_asset_momentum_features(df)
        vals = set(result["xmom_momentum_regime"].unique())
        assert vals.issubset({-1.0, 0.0, 1.0})

    def test_composite_is_mean_of_tstats(self, feat):
        """Composite should equal mean of the 3 t-stat columns."""
        df = _make_spy_daily(200)
        result = feat.create_cross_asset_momentum_features(df)
        expected = (
            result["xmom_tlt_lead_tstat"]
            + result["xmom_gld_lead_tstat"]
            + result["xmom_hyg_lead_tstat"]
        ) / 3.0
        np.testing.assert_allclose(
            result["xmom_cross_momentum_composite"].values,
            expected.values,
            atol=1e-10,
        )


# ---- Analyze Tests -----------------------------------------------------------

class TestAnalyzeCurrentCrossAssetMomentum:
    def test_returns_dict(self):
        feat = CrossAssetMomentumFeatures()
        df = _make_spy_daily(100)
        df = feat.create_cross_asset_momentum_features(df)
        result = feat.analyze_current_cross_asset_momentum(df)
        assert isinstance(result, dict)
        assert "momentum_regime" in result

    def test_regime_values(self):
        feat = CrossAssetMomentumFeatures()
        df = _make_spy_daily(200)
        df = feat.create_cross_asset_momentum_features(df)
        result = feat.analyze_current_cross_asset_momentum(df)
        assert result["momentum_regime"] in {"RISK_ON", "RISK_OFF", "NEUTRAL"}

    def test_returns_none_without_features(self):
        feat = CrossAssetMomentumFeatures()
        df = pd.DataFrame({"close": [100]})
        assert feat.analyze_current_cross_asset_momentum(df) is None


# ---- Feature Count Test ------------------------------------------------------

class TestFeatureCounts:
    def test_total_count(self):
        assert len(ALL_12) == 12
