"""Tests for HurstFeatures — Hurst exponent features via R/S analysis (4 total)."""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.hurst_features import HurstFeatures


# --- Helpers ----------------------------------------------------------------

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


def _make_trending(n_days: int = 200) -> pd.DataFrame:
    """Create a series with persistent (autocorrelated) returns -> high Hurst."""
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    # Autocorrelated returns: each return is 0.7 * previous + noise
    # This creates momentum/persistence in the return series itself
    rng = np.random.RandomState(99)
    returns = np.zeros(n_days)
    returns[0] = 0.005
    for i in range(1, n_days):
        returns[i] = 0.7 * returns[i - 1] + 0.3 * rng.normal(0.002, 0.003)
    close = 450.0 * np.cumprod(1 + returns)
    return pd.DataFrame({
        "date": dates, "close": close,
        "open": close * 0.999, "high": close * 1.002, "low": close * 0.998,
        "volume": np.full(n_days, 100_000_000.0),
    })


ALL_4 = {"hurst_50d", "hurst_100d", "hurst_z", "hurst_regime"}


# --- Invariant Tests --------------------------------------------------------

class TestHurstInvariants:
    @pytest.fixture
    def feat(self):
        return HurstFeatures()

    @pytest.fixture
    def spy(self):
        return _make_spy_daily(200)

    def test_all_4_features_created(self, feat, spy):
        result = feat.create_hurst_features(spy)
        hurst_cols = {c for c in result.columns if c.startswith("hurst_")}
        assert hurst_cols == ALL_4

    def test_no_nans(self, feat, spy):
        result = feat.create_hurst_features(spy)
        for col in ALL_4:
            assert result[col].isna().sum() == 0, f"NaN in {col}"

    def test_no_infinities(self, feat, spy):
        result = feat.create_hurst_features(spy)
        for col in ALL_4:
            assert not np.isinf(result[col]).any(), f"Inf in {col}"

    def test_preserves_original_columns(self, feat, spy):
        original = set(spy.columns)
        result = feat.create_hurst_features(spy)
        assert original.issubset(set(result.columns))

    def test_preserves_row_count(self, feat, spy):
        result = feat.create_hurst_features(spy)
        assert len(result) == len(spy)

    def test_no_close_column(self, feat):
        df = pd.DataFrame({"date": [1, 2], "price": [100, 101]})
        result = feat.create_hurst_features(df)
        assert len(result.columns) == len(df.columns)

    def test_short_data_defaults(self, feat):
        """With very short data, hurst values should default to 0.5."""
        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=10),
            "close": np.linspace(400, 410, 10),
        })
        result = feat.create_hurst_features(df)
        assert "hurst_50d" in result.columns
        # All values should be 0.5 (default) since window > data length
        assert (result["hurst_50d"] == 0.5).all()
        assert (result["hurst_100d"] == 0.5).all()


# --- Feature Logic Tests ---------------------------------------------------

class TestHurstLogic:
    @pytest.fixture
    def feat(self):
        return HurstFeatures()

    def test_hurst_bounded_0_1(self, feat):
        df = _make_spy_daily(200)
        result = feat.create_hurst_features(df)
        assert result["hurst_50d"].min() >= 0.0
        assert result["hurst_50d"].max() <= 1.0
        assert result["hurst_100d"].min() >= 0.0
        assert result["hurst_100d"].max() <= 1.0

    def test_hurst_z_bounded(self, feat):
        df = _make_spy_daily(200)
        result = feat.create_hurst_features(df)
        assert result["hurst_z"].max() <= 4.0
        assert result["hurst_z"].min() >= -4.0

    def test_regime_values(self, feat):
        df = _make_spy_daily(200)
        result = feat.create_hurst_features(df)
        vals = set(result["hurst_regime"].unique())
        assert vals.issubset({0.0, 1.0, 2.0})

    def test_trending_series_high_hurst(self, feat):
        """A strongly trending series should have Hurst > 0.5."""
        df = _make_trending(200)
        result = feat.create_hurst_features(df)
        # Check last value where we have a full 50-day window
        last_hurst = result["hurst_50d"].iloc[-1]
        assert last_hurst > 0.5, f"Trending series got Hurst={last_hurst}, expected > 0.5"

    def test_50d_noisier_than_100d(self, feat):
        """50-day window estimates should be more volatile than 100-day."""
        df = _make_spy_daily(200)
        result = feat.create_hurst_features(df)
        # Compare standard deviations of the two series (after warm-up)
        std_50 = result["hurst_50d"].iloc[100:].std()
        std_100 = result["hurst_100d"].iloc[100:].std()
        assert std_50 >= std_100, (
            f"50d std ({std_50:.4f}) should be >= 100d std ({std_100:.4f})"
        )


# --- Analyze Tests ----------------------------------------------------------

class TestAnalyzeCurrentHurst:
    def test_returns_dict(self):
        feat = HurstFeatures()
        df = _make_spy_daily(200)
        df = feat.create_hurst_features(df)
        result = feat.analyze_current_hurst(df)
        assert isinstance(result, dict)
        assert "hurst_regime" in result

    def test_regime_values(self):
        feat = HurstFeatures()
        df = _make_spy_daily(200)
        df = feat.create_hurst_features(df)
        result = feat.analyze_current_hurst(df)
        assert result["hurst_regime"] in {"MEAN_REVERTING", "RANDOM_WALK", "TRENDING"}

    def test_returns_none_without_features(self):
        feat = HurstFeatures()
        df = pd.DataFrame({"close": [100]})
        assert feat.analyze_current_hurst(df) is None


# --- Feature Count Test -----------------------------------------------------

class TestFeatureCounts:
    def test_total_count(self):
        assert len(ALL_4) == 4
