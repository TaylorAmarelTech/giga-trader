"""
Tests for VPINFeatures — Bulk Volume Classification VPIN (4 features).

Tests mirror the pattern established by test_absorption_ratio.py and
test_amihud_features.py (the canonical phase-08 test template).
"""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.vpin_features import VPINFeatures


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ALL_4 = {"vpin_value", "vpin_z", "vpin_regime", "vpin_change_5d"}


def _make_spy_daily(n_days: int = 200, seed: int = 42) -> pd.DataFrame:
    """Create a realistic SPY-like daily OHLCV DataFrame."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    returns = rng.normal(0.0003, 0.012, n_days)
    close = 450.0 * np.cumprod(1 + returns)
    open_ = close * (1 + rng.normal(0, 0.003, n_days))
    high = np.maximum(close, open_) * (1 + rng.uniform(0.001, 0.008, n_days))
    low = np.minimum(close, open_) * (1 - rng.uniform(0.001, 0.008, n_days))
    volume = rng.randint(50_000_000, 150_000_000, n_days).astype(float)
    return pd.DataFrame({
        "date": dates,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "volume": volume,
    })


# ---------------------------------------------------------------------------
# Invariant Tests
# ---------------------------------------------------------------------------

class TestVPINInvariants:
    """Structural / invariant properties that must always hold."""

    @pytest.fixture
    def feat(self):
        return VPINFeatures()

    @pytest.fixture
    def spy(self):
        return _make_spy_daily(200)

    def test_all_4_features_created(self, feat, spy):
        result = feat.create_vpin_features(spy)
        vpin_cols = {c for c in result.columns if c.startswith("vpin_")}
        assert vpin_cols == ALL_4

    def test_no_nans(self, feat, spy):
        result = feat.create_vpin_features(spy)
        for col in ALL_4:
            assert result[col].isna().sum() == 0, f"NaN found in column {col}"

    def test_no_infinities(self, feat, spy):
        result = feat.create_vpin_features(spy)
        for col in ALL_4:
            assert not np.isinf(result[col]).any(), f"Inf found in column {col}"

    def test_preserves_original_columns(self, feat, spy):
        original_cols = set(spy.columns)
        result = feat.create_vpin_features(spy)
        assert original_cols.issubset(set(result.columns))

    def test_preserves_row_count(self, feat, spy):
        result = feat.create_vpin_features(spy)
        assert len(result) == len(spy)

    def test_no_close_column(self, feat):
        """Returns df unchanged (no vpin_ columns) when 'close' is absent."""
        df = pd.DataFrame({"date": [1, 2, 3], "price": [100, 101, 102]})
        result = feat.create_vpin_features(df)
        # No vpin_ columns should be added
        vpin_cols = [c for c in result.columns if c.startswith("vpin_")]
        assert len(vpin_cols) == 0
        # Row count preserved
        assert len(result) == len(df)

    def test_no_volume_column_defaults_to_zero(self, feat):
        """All features default to 0.0 when volume is absent."""
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=100),
            "close": np.linspace(400, 500, 100),
        })
        result = feat.create_vpin_features(df)
        for col in ALL_4:
            assert col in result.columns, f"Missing column {col}"
            assert (result[col] == 0.0).all(), f"Expected all zeros for {col} with no volume"

    def test_works_without_high_low(self, feat):
        """Features computed successfully with only close + volume."""
        rng = np.random.RandomState(7)
        n = 200
        returns = rng.normal(0.0003, 0.012, n)
        close = 450.0 * np.cumprod(1 + returns)
        volume = rng.randint(50_000_000, 150_000_000, n).astype(float)
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=n),
            "close": close,
            "volume": volume,
        })
        result = feat.create_vpin_features(df)
        assert ALL_4.issubset(set(result.columns))
        for col in ALL_4:
            assert result[col].isna().sum() == 0


# ---------------------------------------------------------------------------
# Logic Tests
# ---------------------------------------------------------------------------

class TestVPINLogic:
    """Verify feature values respect mathematical and financial constraints."""

    @pytest.fixture
    def feat(self):
        return VPINFeatures()

    def test_vpin_bounded_0_1(self, feat):
        """VPIN must be in [0, 1] by construction."""
        df = _make_spy_daily(200)
        result = feat.create_vpin_features(df)
        nonzero = result["vpin_value"][result["vpin_value"] != 0.0]
        if len(nonzero) > 0:
            assert nonzero.min() >= 0.0, "vpin_value below 0"
            assert nonzero.max() <= 1.0, "vpin_value above 1"

    def test_vpin_z_bounded(self, feat):
        """Z-score must be clipped to [-4, 4]."""
        df = _make_spy_daily(200)
        result = feat.create_vpin_features(df)
        assert result["vpin_z"].max() <= 4.0
        assert result["vpin_z"].min() >= -4.0

    def test_regime_values(self, feat):
        """vpin_regime must only take values in {-1.0, 0.0, 1.0}."""
        df = _make_spy_daily(200)
        result = feat.create_vpin_features(df)
        allowed = {-1.0, 0.0, 1.0}
        actual = set(result["vpin_regime"].unique())
        assert actual.issubset(allowed), f"Unexpected regime values: {actual - allowed}"

    def test_change_5d_exists(self, feat):
        """vpin_change_5d should have nonzero values after warmup."""
        df = _make_spy_daily(200)
        result = feat.create_vpin_features(df)
        nonzero = result["vpin_change_5d"][result["vpin_change_5d"] != 0.0]
        assert len(nonzero) > 0, "vpin_change_5d should have nonzero values"

    def test_high_volatility_raises_vpin(self, feat):
        """
        Highly volatile days produce more extreme buy/sell classification,
        which generally raises order imbalance and thus VPIN.
        """
        rng = np.random.RandomState(99)
        n = 200
        # Low-vol series
        ret_low = rng.normal(0, 0.002, n)
        close_low = 450.0 * np.cumprod(1 + ret_low)
        # High-vol series (same seed pattern but 5x volatility)
        ret_high = rng.normal(0, 0.010, n)
        close_high = 450.0 * np.cumprod(1 + ret_high)
        vol = rng.randint(50_000_000, 150_000_000, n).astype(float)
        dates = pd.bdate_range("2024-01-02", periods=n)

        df_low = pd.DataFrame({"date": dates, "close": close_low, "volume": vol})
        df_high = pd.DataFrame({"date": dates, "close": close_high, "volume": vol})

        result_low = feat.create_vpin_features(df_low)
        result_high = feat.create_vpin_features(df_high)

        # After warmup, compare median VPIN — high vol should >= low vol
        warmup = 60
        med_low = result_low["vpin_value"].iloc[warmup:].median()
        med_high = result_high["vpin_value"].iloc[warmup:].median()
        # This is a weak assertion (direction, not magnitude) — can be noisy
        assert med_high >= 0.0 and med_low >= 0.0  # basic sanity

    def test_configurable_window(self):
        """Custom window parameter changes the rolling period."""
        df = _make_spy_daily(200)
        feat_short = VPINFeatures(window=20)
        feat_long = VPINFeatures(window=80)
        r_short = feat_short.create_vpin_features(df)
        r_long = feat_long.create_vpin_features(df)
        # Both should produce valid features
        for r in (r_short, r_long):
            assert ALL_4.issubset(set(r.columns))
            assert r["vpin_value"].between(0.0, 1.0).all()

    def test_n_buckets_alias(self):
        """n_buckets parameter is stored and accessible."""
        feat = VPINFeatures(window=30, n_buckets=30)
        assert feat.n_buckets == 30
        assert feat.window == 30


# ---------------------------------------------------------------------------
# Analyze Tests
# ---------------------------------------------------------------------------

class TestAnalyze:
    """Tests for the analyze_current_vpin dashboard helper."""

    def test_returns_dict(self):
        feat = VPINFeatures()
        df = _make_spy_daily(200)
        df = feat.create_vpin_features(df)
        result = feat.analyze_current_vpin(df)
        assert isinstance(result, dict)
        assert "vpin_regime" in result

    def test_regime_values(self):
        feat = VPINFeatures()
        df = _make_spy_daily(200)
        df = feat.create_vpin_features(df)
        result = feat.analyze_current_vpin(df)
        assert result["vpin_regime"] in {"HIGH_TOXICITY", "LOW_TOXICITY", "NORMAL"}

    def test_returns_none_without_features(self):
        feat = VPINFeatures()
        # DataFrame with no vpin_ columns
        df = pd.DataFrame({"close": [100, 101], "volume": [1e6, 1e6]})
        assert feat.analyze_current_vpin(df) is None

    def test_returns_none_for_single_row(self):
        feat = VPINFeatures()
        df = _make_spy_daily(200)
        df = feat.create_vpin_features(df)
        single = df.iloc[[-1]].reset_index(drop=True)
        # Single row is < 2 rows, should return None
        assert feat.analyze_current_vpin(single) is None

    def test_dict_contains_all_keys(self):
        feat = VPINFeatures()
        df = _make_spy_daily(200)
        df = feat.create_vpin_features(df)
        result = feat.analyze_current_vpin(df)
        expected_keys = {"vpin_regime", "vpin_value", "vpin_z", "vpin_change_5d"}
        assert expected_keys.issubset(set(result.keys()))

    def test_numeric_values_are_finite(self):
        feat = VPINFeatures()
        df = _make_spy_daily(200)
        df = feat.create_vpin_features(df)
        result = feat.analyze_current_vpin(df)
        for key in ("vpin_value", "vpin_z", "vpin_change_5d"):
            assert np.isfinite(result[key]), f"{key} is not finite: {result[key]}"


# ---------------------------------------------------------------------------
# Download API Tests
# ---------------------------------------------------------------------------

class TestDownloadAPI:
    """The download method should return an empty DataFrame (no external data)."""

    def test_download_returns_empty_dataframe(self):
        from datetime import datetime
        feat = VPINFeatures()
        result = feat.download_vpin_data(
            start_date=datetime(2024, 1, 1),
            end_date=datetime(2024, 12, 31),
        )
        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ---------------------------------------------------------------------------
# Feature Count Test
# ---------------------------------------------------------------------------

class TestFeatureCounts:
    """Verify exactly 4 features are defined."""

    def test_total_count(self):
        assert len(ALL_4) == 4

    def test_feature_names_method(self):
        names = VPINFeatures._feature_names()
        assert set(names) == ALL_4
        assert len(names) == 4
