"""Tests for AbsorptionRatioFeatures — systemic risk via PCA eigenvalue concentration (3 features)."""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.absorption_ratio_features import AbsorptionRatioFeatures


# --- Helpers -----------------------------------------------------------------

ALL_3 = {"ar_ratio", "ar_change_20d", "ar_z"}


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


def _make_cross_asset_daily(n_days: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create df with cross-asset return columns (the preferred path)."""
    rng = np.random.RandomState(seed)
    df = _make_spy_daily(n_days, seed)
    for col in ["TLT_return", "QQQ_return", "GLD_return",
                 "IWM_return", "EEM_return", "HYG_return", "VXX_return"]:
        df[col] = rng.normal(0.0001, 0.01, n_days)
    return df


# --- Invariant Tests ---------------------------------------------------------

class TestAbsorptionRatioInvariants:
    @pytest.fixture
    def feat(self):
        return AbsorptionRatioFeatures()

    @pytest.fixture
    def spy(self):
        return _make_spy_daily(200)

    def test_all_3_features_created(self, feat, spy):
        result = feat.create_absorption_ratio_features(spy)
        ar_cols = {c for c in result.columns if c.startswith("ar_")}
        assert ar_cols == ALL_3

    def test_no_nans(self, feat, spy):
        result = feat.create_absorption_ratio_features(spy)
        for col in ALL_3:
            assert result[col].isna().sum() == 0, f"NaN in {col}"

    def test_no_infinities(self, feat, spy):
        result = feat.create_absorption_ratio_features(spy)
        for col in ALL_3:
            assert not np.isinf(result[col]).any(), f"Inf in {col}"

    def test_preserves_original_columns(self, feat, spy):
        original = set(spy.columns)
        result = feat.create_absorption_ratio_features(spy)
        assert original.issubset(set(result.columns))

    def test_preserves_row_count(self, feat, spy):
        result = feat.create_absorption_ratio_features(spy)
        assert len(result) == len(spy)

    def test_no_close_column(self, feat):
        df = pd.DataFrame({"date": [1, 2], "price": [100, 101]})
        result = feat.create_absorption_ratio_features(df)
        # Should return unchanged (no ar_ columns added)
        assert len(result.columns) == len(df.columns)


# --- Logic Tests -------------------------------------------------------------

class TestAbsorptionRatioLogic:
    @pytest.fixture
    def feat(self):
        return AbsorptionRatioFeatures()

    def test_ar_ratio_bounded_0_1(self, feat):
        df = _make_cross_asset_daily(200)
        result = feat.create_absorption_ratio_features(df)
        nonzero = result["ar_ratio"][result["ar_ratio"] != 0.0]
        if len(nonzero) > 0:
            assert nonzero.min() >= 0.0, "ar_ratio should be >= 0"
            assert nonzero.max() <= 1.0, "ar_ratio should be <= 1"

    def test_ar_z_bounded(self, feat):
        df = _make_cross_asset_daily(200)
        result = feat.create_absorption_ratio_features(df)
        assert result["ar_z"].max() <= 4.0
        assert result["ar_z"].min() >= -4.0

    def test_change_20d_exists(self, feat):
        df = _make_cross_asset_daily(200)
        result = feat.create_absorption_ratio_features(df)
        # After initial warmup, change should have nonzero values
        nonzero = result["ar_change_20d"][result["ar_change_20d"] != 0.0]
        assert len(nonzero) > 0, "ar_change_20d should have nonzero values"


# --- Analyze Tests -----------------------------------------------------------

class TestAnalyze:
    def test_returns_dict(self):
        feat = AbsorptionRatioFeatures()
        df = _make_cross_asset_daily(200)
        df = feat.create_absorption_ratio_features(df)
        result = feat.analyze_current_absorption_ratio(df)
        assert isinstance(result, dict)
        assert "absorption_regime" in result

    def test_regime_values(self):
        feat = AbsorptionRatioFeatures()
        df = _make_cross_asset_daily(200)
        df = feat.create_absorption_ratio_features(df)
        result = feat.analyze_current_absorption_ratio(df)
        assert result["absorption_regime"] in {"HIGH_RISK", "LOW_RISK", "MODERATE"}

    def test_returns_none_without_features(self):
        feat = AbsorptionRatioFeatures()
        df = pd.DataFrame({"close": [100]})
        assert feat.analyze_current_absorption_ratio(df) is None


# --- Feature Count Test ------------------------------------------------------

class TestFeatureCounts:
    def test_total_count(self):
        assert len(ALL_3) == 3
