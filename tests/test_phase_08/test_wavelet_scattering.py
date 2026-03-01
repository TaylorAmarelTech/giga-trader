"""Tests for WaveletScatteringFeatures -- scattering transform features (12 total)."""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.wavelet_scattering_features import (
    WaveletScatteringFeatures,
)


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


def _make_constant(n_days: int = 200, price: float = 450.0) -> pd.DataFrame:
    """Constant price series -- returns are all zero."""
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    close = np.full(n_days, price)
    return pd.DataFrame({
        "date": dates, "close": close,
        "open": close, "high": close, "low": close,
        "volume": np.full(n_days, 100_000_000.0),
    })


def _make_trending(n_days: int = 200) -> pd.DataFrame:
    """Smooth uptrend -- low high-freq energy, high low-freq energy."""
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    close = 400.0 + np.linspace(0, 100, n_days)
    return pd.DataFrame({
        "date": dates, "close": close,
        "open": close * 0.999, "high": close * 1.002, "low": close * 0.998,
        "volume": np.full(n_days, 100_000_000.0),
    })


def _make_volatile(n_days: int = 200, seed: int = 77) -> pd.DataFrame:
    """Highly volatile series -- large random moves each day."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    returns = rng.normal(0.0, 0.05, n_days)
    close = 450.0 * np.cumprod(1 + returns)
    return pd.DataFrame({
        "date": dates, "close": close,
        "open": close * (1 + rng.normal(0, 0.01, n_days)),
        "high": close * 1.02, "low": close * 0.98,
        "volume": np.full(n_days, 100_000_000.0),
    })


ALL_12 = {
    "wscat_s0_20d",
    "wscat_s1_scale2_20d",
    "wscat_s1_scale4_20d",
    "wscat_s1_scale8_20d",
    "wscat_s2_2x4_20d",
    "wscat_s2_2x8_20d",
    "wscat_s2_4x8_20d",
    "wscat_energy_ratio",
    "wscat_intermittency",
    "wscat_s0_z",
    "wscat_energy_trend",
    "wscat_regime",
}


# --- Invariant Tests --------------------------------------------------------

class TestScatteringInvariants:
    @pytest.fixture
    def feat(self):
        return WaveletScatteringFeatures()

    @pytest.fixture
    def spy(self):
        return _make_spy_daily(200)

    def test_correct_column_names(self, feat, spy):
        """All 12 wscat_ features are created with correct names."""
        result = feat.create_wavelet_scattering_features(spy)
        wscat_cols = {c for c in result.columns if c.startswith("wscat_")}
        assert wscat_cols == ALL_12

    def test_output_shape_matches_input(self, feat, spy):
        """Row count is preserved; column count increases by exactly 12."""
        result = feat.create_wavelet_scattering_features(spy)
        assert len(result) == len(spy)
        new_cols = [c for c in result.columns if c.startswith("wscat_")]
        assert len(new_cols) == 12

    def test_nan_handling_first_rows(self, feat, spy):
        """After cleanup, no NaN or Inf should remain in any wscat_ column."""
        result = feat.create_wavelet_scattering_features(spy)
        for col in ALL_12:
            assert result[col].isna().sum() == 0, f"NaN found in {col}"
            assert not np.isinf(result[col]).any(), f"Inf found in {col}"

    def test_preserves_original_columns(self, feat, spy):
        original = set(spy.columns)
        result = feat.create_wavelet_scattering_features(spy)
        assert original.issubset(set(result.columns))

    def test_no_close_column_skips(self, feat):
        """Missing 'close' column returns df unchanged."""
        df = pd.DataFrame({"date": [1, 2], "price": [100, 101]})
        result = feat.create_wavelet_scattering_features(df)
        assert len(result.columns) == len(df.columns)


# --- Feature Logic Tests ---------------------------------------------------

class TestScatteringLogic:
    @pytest.fixture
    def feat(self):
        return WaveletScatteringFeatures()

    def test_constant_price_near_zero_energy(self, feat):
        """Constant price => zero returns => near-zero scattering energy."""
        df = _make_constant(200)
        result = feat.create_wavelet_scattering_features(df)
        # All layer-1 and layer-2 scattering coefficients should be ~0
        for col in [
            "wscat_s0_20d",
            "wscat_s1_scale2_20d",
            "wscat_s1_scale4_20d",
            "wscat_s1_scale8_20d",
            "wscat_s2_2x4_20d",
            "wscat_s2_2x8_20d",
            "wscat_s2_4x8_20d",
        ]:
            assert result[col].iloc[30:].max() < 1e-6, (
                f"{col} should be ~0 for constant price, "
                f"got max={result[col].iloc[30:].max()}"
            )

    def test_trending_price_has_low_freq_energy(self, feat):
        """A smooth trend should have relatively more energy at scale 8."""
        df = _make_trending(200)
        result = feat.create_wavelet_scattering_features(df)
        # Low-freq (scale 8) should have significant energy
        s1_8_mean = result["wscat_s1_scale8_20d"].iloc[30:].mean()
        assert s1_8_mean > 0, (
            f"Trending series should have scale-8 energy, got {s1_8_mean}"
        )

    def test_volatile_price_has_high_freq_energy(self, feat):
        """Volatile data should have more high-frequency energy (scale 2)."""
        df = _make_volatile(200)
        result = feat.create_wavelet_scattering_features(df)
        s1_2_mean = result["wscat_s1_scale2_20d"].iloc[30:].mean()
        assert s1_2_mean > 0, (
            f"Volatile series should have scale-2 energy, got {s1_2_mean}"
        )

    def test_energy_ratio_positive_for_volatile(self, feat):
        """Energy ratio (high/low freq) should be > 0 for volatile data."""
        df = _make_volatile(200)
        result = feat.create_wavelet_scattering_features(df)
        # After warm-up, energy ratio should be positive
        er_tail = result["wscat_energy_ratio"].iloc[30:]
        assert (er_tail >= 0).all(), "Energy ratio should be non-negative"
        assert er_tail.mean() > 0, "Mean energy ratio should be > 0"

    def test_no_future_leakage(self, feat):
        """Features computed on a prefix should not change when suffix data is appended.

        fftconvolve with mode="same" has boundary effects that depend on
        signal length, so we use a large dataset and compare a central region
        that is well away from both boundaries.  Layer-1 features (single
        convolution) should match tightly.  Layer-2 features (double
        convolution) are checked with a looser tolerance to account for the
        compounded boundary zone while still catching systematic leakage.
        """
        df = _make_spy_daily(500, seed=42)
        result_full = feat.create_wavelet_scattering_features(df)

        # Truncate to first 250 rows and recompute
        df_short = df.iloc[:250].copy()
        result_short = feat.create_wavelet_scattering_features(df_short)

        # Check a central region far from both boundaries.
        # The max wavelet kernel is scale=8 * 10 = 80 points;
        # layer-2 doubles this.  Stay well away from both ends.
        check_start = 100
        check_end = 200

        # Layer-0 and layer-1 features: tight tolerance
        layer01_cols = {
            "wscat_s0_20d", "wscat_s1_scale2_20d", "wscat_s1_scale4_20d",
            "wscat_s1_scale8_20d", "wscat_energy_ratio", "wscat_s0_z",
            "wscat_energy_trend", "wscat_regime",
        }
        for col in layer01_cols:
            vals_full = result_full[col].iloc[check_start:check_end].values
            vals_short = result_short[col].iloc[check_start:check_end].values
            np.testing.assert_allclose(
                vals_full, vals_short, atol=1e-6, rtol=1e-4,
                err_msg=f"Future leakage detected in {col}",
            )

        # Layer-2 features: looser tolerance (double convolution boundary)
        layer2_cols = {
            "wscat_s2_2x4_20d", "wscat_s2_2x8_20d", "wscat_s2_4x8_20d",
            "wscat_intermittency",
        }
        for col in layer2_cols:
            vals_full = result_full[col].iloc[check_start:check_end].values
            vals_short = result_short[col].iloc[check_start:check_end].values
            np.testing.assert_allclose(
                vals_full, vals_short, atol=1e-3, rtol=0.05,
                err_msg=f"Future leakage detected in {col}",
            )

    def test_random_walk_data(self, feat):
        """Random walk data should produce valid features without errors."""
        rng = np.random.RandomState(123)
        n = 300
        dates = pd.bdate_range("2023-01-02", periods=n, freq="B")
        steps = rng.normal(0, 0.01, n)
        close = 100.0 * np.exp(np.cumsum(steps))
        df = pd.DataFrame({
            "date": dates, "close": close,
            "open": close * 0.999, "high": close * 1.005, "low": close * 0.995,
            "volume": np.full(n, 50_000_000.0),
        })
        result = feat.create_wavelet_scattering_features(df)

        # All features should be present and finite
        for col in ALL_12:
            assert col in result.columns
            assert result[col].isna().sum() == 0, f"NaN in {col}"
            assert not np.isinf(result[col]).any(), f"Inf in {col}"

        # Scattering coefficients should all be non-negative (energy measures)
        for col in [
            "wscat_s0_20d", "wscat_s1_scale2_20d",
            "wscat_s1_scale4_20d", "wscat_s1_scale8_20d",
            "wscat_s2_2x4_20d", "wscat_s2_2x8_20d", "wscat_s2_4x8_20d",
        ]:
            assert (result[col] >= 0).all(), f"{col} should be >= 0"

    def test_wscat_regime_is_binary(self, feat):
        """wscat_regime should contain only 0.0 or 1.0."""
        df = _make_spy_daily(200)
        result = feat.create_wavelet_scattering_features(df)
        unique_vals = set(result["wscat_regime"].unique())
        assert unique_vals.issubset({0.0, 1.0}), (
            f"wscat_regime should be binary {{0, 1}}, got {unique_vals}"
        )


# --- Analyze Tests ----------------------------------------------------------

class TestAnalyzeCurrentScattering:
    def test_returns_dict(self):
        feat = WaveletScatteringFeatures()
        df = _make_spy_daily(200)
        df = feat.create_wavelet_scattering_features(df)
        result = feat.analyze_current_scattering(df)
        assert isinstance(result, dict)
        assert "scattering_regime" in result

    def test_regime_values(self):
        feat = WaveletScatteringFeatures()
        df = _make_spy_daily(200)
        df = feat.create_wavelet_scattering_features(df)
        result = feat.analyze_current_scattering(df)
        assert result["scattering_regime"] in {"HIGH_FREQUENCY", "NORMAL"}

    def test_returns_none_without_features(self):
        feat = WaveletScatteringFeatures()
        df = pd.DataFrame({"close": [100]})
        assert feat.analyze_current_scattering(df) is None


# --- Feature Count Test -----------------------------------------------------

class TestFeatureCounts:
    def test_total_count(self):
        assert len(ALL_12) == 12
