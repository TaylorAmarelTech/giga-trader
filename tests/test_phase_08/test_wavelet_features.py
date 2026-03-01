"""
Tests for WaveletFeatures class.

Validates multi-resolution wavelet-like price decomposition using pure
numpy/pandas (no PyWavelets dependency).

All tests use synthetic data with 200+ rows so rolling-window features
have sufficient history to produce meaningful values.
"""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.wavelet_features import WaveletFeatures


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def daily_close():
    """Synthetic daily close prices — 252 business days, realistic price path."""
    np.random.seed(7)
    n = 252
    dates = pd.bdate_range("2023-01-03", periods=n)
    close = 450.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, n)))
    return pd.Series(close, index=dates, name="close")


@pytest.fixture
def daily_df(daily_close):
    """Daily DataFrame with 'close' column plus extra columns."""
    return pd.DataFrame({
        "close": daily_close.values,
        "volume": np.random.randint(50_000_000, 150_000_000, len(daily_close)),
        "open": daily_close.values * np.random.uniform(0.998, 1.002, len(daily_close)),
    }, index=daily_close.index)


@pytest.fixture
def wf():
    """Default WaveletFeatures instance."""
    return WaveletFeatures()


@pytest.fixture
def computed_df(wf, daily_df):
    """DataFrame with wavelet features already computed."""
    return wf.create_wavelet_features(daily_df)


# =============================================================================
# 1. Constructor tests
# =============================================================================

class TestConstructor:

    def test_default_windows(self):
        wf = WaveletFeatures()
        assert wf.short_window == 3
        assert wf.long_window == 5

    def test_custom_windows(self):
        wf = WaveletFeatures(short_window=2, long_window=8)
        assert wf.short_window == 2
        assert wf.long_window == 8

    def test_prefix_constant(self):
        assert WaveletFeatures.FEATURE_PREFIX == "wav_"

    def test_feature_names_count(self):
        assert len(WaveletFeatures.FEATURE_NAMES) == 10


# =============================================================================
# 2. download_wavelet_data returns empty DataFrame
# =============================================================================

class TestDownloadWaveletData:

    def test_returns_empty_dataframe(self, wf):
        from datetime import datetime
        result = wf.download_wavelet_data(
            datetime(2023, 1, 1), datetime(2023, 12, 31)
        )
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_accepts_datetime_objects(self, wf):
        from datetime import datetime
        # Should not raise
        result = wf.download_wavelet_data(
            datetime(2020, 1, 1), datetime(2020, 12, 31)
        )
        assert isinstance(result, pd.DataFrame)


# =============================================================================
# 3. Feature creation — column presence
# =============================================================================

class TestFeatureColumns:

    def test_all_ten_features_present(self, computed_df):
        for col in WaveletFeatures.FEATURE_NAMES:
            assert col in computed_df.columns, f"Missing column: {col}"

    def test_exactly_ten_wav_columns(self, computed_df):
        wav_cols = [c for c in computed_df.columns if c.startswith("wav_")]
        assert len(wav_cols) == 10

    def test_original_columns_preserved(self, computed_df, daily_df):
        for col in daily_df.columns:
            assert col in computed_df.columns

    def test_row_count_unchanged(self, computed_df, daily_df):
        assert len(computed_df) == len(daily_df)

    def test_index_unchanged(self, computed_df, daily_df):
        assert computed_df.index.equals(daily_df.index)


# =============================================================================
# 4. No NaN / Inf in output features
# =============================================================================

class TestNoNaNsOrInf:

    def test_no_nans_in_wav_columns(self, computed_df):
        wav_cols = [c for c in computed_df.columns if c.startswith("wav_")]
        nan_count = computed_df[wav_cols].isna().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values in wav columns"

    def test_no_infs_in_wav_columns(self, computed_df):
        wav_cols = [c for c in computed_df.columns if c.startswith("wav_")]
        inf_count = np.isinf(computed_df[wav_cols].values).sum()
        assert inf_count == 0, f"Found {inf_count} Inf values in wav columns"


# =============================================================================
# 5. Value range / clipping tests
# =============================================================================

class TestValueRanges:

    def test_wav_regime_values_are_minus1_0_or_1(self, computed_df):
        valid_values = {-1.0, 0.0, 1.0}
        actual_values = set(computed_df["wav_regime"].unique())
        assert actual_values.issubset(valid_values), (
            f"wav_regime contains unexpected values: {actual_values - valid_values}"
        )

    def test_wav_energy_ratio_non_negative(self, computed_df):
        assert (computed_df["wav_energy_ratio"] >= 0).all()

    def test_wav_energy_ratio_clipped_at_20(self, computed_df):
        assert (computed_df["wav_energy_ratio"] <= 20.0).all()

    def test_continuous_features_clipped_at_10(self, computed_df):
        clipped_features = [
            "wav_trend_5d", "wav_trend_3d", "wav_detail_1d", "wav_detail_2d",
            "wav_trend_momentum", "wav_noise_level", "wav_denoised_return",
            "wav_cross_scale",
        ]
        for col in clipped_features:
            assert (computed_df[col] >= -10.0).all(), f"{col} below -10"
            assert (computed_df[col] <= 10.0).all(), f"{col} above +10"

    def test_wav_cross_scale_bounded_by_clip(self, computed_df):
        """Correlation is naturally in [-1, 1] before clipping at [-10, 10]."""
        # After clipping to [-10, 10], values should still be within [-1, 1]
        # because correlations don't exceed that range
        assert (computed_df["wav_cross_scale"] >= -1.0).all()
        assert (computed_df["wav_cross_scale"] <= 1.0).all()


# =============================================================================
# 6. Feature semantics
# =============================================================================

class TestFeatureSemantics:

    def test_wav_detail_1d_is_pct_change(self, wf, daily_df):
        result = wf.create_wavelet_features(daily_df)
        close = daily_df["close"]
        expected = close.pct_change(1).fillna(0).clip(-10, 10)
        pd.testing.assert_series_equal(
            result["wav_detail_1d"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
            atol=1e-9,
        )

    def test_wav_trend_5d_and_3d_differ(self, computed_df):
        """5-day and 3-day trend residuals should not be identical."""
        assert not np.allclose(
            computed_df["wav_trend_5d"].values,
            computed_df["wav_trend_3d"].values,
        )

    def test_wav_noise_level_non_negative(self, computed_df):
        """Noise level is a std, so must be >= 0."""
        assert (computed_df["wav_noise_level"] >= 0).all()

    def test_wav_energy_ratio_increases_with_noise(self):
        """High-noise prices → high energy ratio; smooth prices → low energy ratio."""
        n = 252
        dates = pd.bdate_range("2023-01-03", periods=n)

        # Very smooth (trending) series
        close_smooth = pd.Series(
            np.linspace(400, 500, n), index=dates
        )
        df_smooth = pd.DataFrame({"close": close_smooth})

        # Very noisy series (white noise around constant)
        np.random.seed(99)
        close_noisy = pd.Series(
            450.0 + np.random.normal(0, 10, n), index=dates
        )
        df_noisy = pd.DataFrame({"close": close_noisy})

        wf = WaveletFeatures()
        smooth_result = wf.create_wavelet_features(df_smooth)
        noisy_result = wf.create_wavelet_features(df_noisy)

        # Use mean over later half to let rolling windows warm up
        half = n // 2
        smooth_ratio = smooth_result["wav_energy_ratio"].iloc[half:].mean()
        noisy_ratio = noisy_result["wav_energy_ratio"].iloc[half:].mean()

        assert noisy_ratio > smooth_ratio, (
            f"Expected noisy_ratio ({noisy_ratio:.3f}) > smooth_ratio ({smooth_ratio:.3f})"
        )

    def test_wav_regime_trending_on_smooth_data(self):
        """A perfectly smooth trending series should produce mostly -1 (trending) regimes.

        For a linear price series the energy ratio converges to ~0.5 (well below the
        TRENDING threshold of 0.7), so after the 10-row rolling warm-up period the
        regime should be classified as -1 (trending).
        """
        n = 252
        dates = pd.bdate_range("2023-01-03", periods=n)
        close = pd.Series(np.linspace(400, 500, n), index=dates)
        df = pd.DataFrame({"close": close})

        wf = WaveletFeatures()
        result = wf.create_wavelet_features(df)

        # After warm-up (skip first 15 rows for rolling(10) to settle),
        # regime should be mostly trending (-1).
        later_regimes = result["wav_regime"].iloc[15:]
        n_trending = (later_regimes == -1).sum()
        n_total = len(later_regimes)
        assert n_trending / n_total > 0.5, (
            f"Expected mostly trending on smooth data, got {n_trending}/{n_total} trending"
        )

    def test_wav_denoised_return_smoother_than_raw(self, computed_df):
        """EMA-smoothed returns should have lower std than raw daily returns."""
        raw_std = computed_df["wav_detail_1d"].std()
        smooth_std = computed_df["wav_denoised_return"].std()
        assert smooth_std < raw_std, (
            f"Denoised return (std={smooth_std:.5f}) should be < raw (std={raw_std:.5f})"
        )


# =============================================================================
# 7. Missing / invalid input handling
# =============================================================================

class TestMissingInputHandling:

    def test_no_close_column_returns_unchanged(self, wf):
        df = pd.DataFrame({"open": [100, 101, 102], "volume": [1e6, 1e6, 1e6]})
        result = wf.create_wavelet_features(df)
        assert list(result.columns) == list(df.columns)

    def test_close_column_case_insensitive_upper(self, wf):
        """Should detect 'Close' (title case)."""
        n = 50
        dates = pd.bdate_range("2023-01-03", periods=n)
        df = pd.DataFrame({"Close": np.linspace(400, 450, n)}, index=dates)
        result = wf.create_wavelet_features(df)
        wav_cols = [c for c in result.columns if c.startswith("wav_")]
        assert len(wav_cols) == 10

    def test_insufficient_rows_returns_zeros(self, wf):
        """With fewer rows than needed, features should be 0-filled not errored."""
        df = pd.DataFrame({"close": [100.0, 101.0, 102.0]})
        result = wf.create_wavelet_features(df)
        wav_cols = [c for c in result.columns if c.startswith("wav_")]
        assert len(wav_cols) == 10
        # All zero-filled
        assert (result[wav_cols] == 0.0).all().all()

    def test_close_with_zeros_does_not_raise(self, wf):
        """A close column containing zeros should not raise ZeroDivisionError."""
        n = 252
        dates = pd.bdate_range("2023-01-03", periods=n)
        close = np.zeros(n)
        df = pd.DataFrame({"close": close}, index=dates)
        result = wf.create_wavelet_features(df)  # should not raise
        wav_cols = [c for c in result.columns if c.startswith("wav_")]
        assert len(wav_cols) == 10

    def test_empty_dataframe_returns_original(self, wf):
        df = pd.DataFrame()
        result = wf.create_wavelet_features(df)
        assert result.empty

    def test_does_not_mutate_input(self, wf, daily_df):
        original_cols = list(daily_df.columns)
        _ = wf.create_wavelet_features(daily_df)
        assert list(daily_df.columns) == original_cols


# =============================================================================
# 8. analyze_current_wavelet
# =============================================================================

class TestAnalyzeCurrentWavelet:

    def test_returns_dict(self, wf, computed_df):
        result = wf.analyze_current_wavelet(computed_df)
        assert isinstance(result, dict)

    def test_wavelet_regime_key_present(self, wf, computed_df):
        result = wf.analyze_current_wavelet(computed_df)
        assert "wavelet_regime" in result

    def test_wavelet_regime_valid_values(self, wf, computed_df):
        result = wf.analyze_current_wavelet(computed_df)
        assert result["wavelet_regime"] in ("TRENDING", "NOISY", "MIXED")

    def test_all_expected_keys_present(self, wf, computed_df):
        result = wf.analyze_current_wavelet(computed_df)
        expected_keys = {
            "wavelet_regime", "energy_ratio", "noise_level",
            "trend_momentum", "cross_scale", "trend_5d", "denoised_return",
        }
        assert expected_keys.issubset(set(result.keys()))

    def test_computes_from_raw_df(self, wf, daily_df):
        """analyze_current_wavelet should work on a df without wav_ columns."""
        result = wf.analyze_current_wavelet(daily_df)
        assert result is not None
        assert "wavelet_regime" in result

    def test_no_close_returns_none(self, wf):
        df = pd.DataFrame({"open": [100, 101, 102]})
        result = wf.analyze_current_wavelet(df)
        assert result is None

    def test_trending_regime_on_smooth_data(self):
        """Smooth linear price → should report TRENDING."""
        n = 252
        dates = pd.bdate_range("2023-01-03", periods=n)
        close = pd.Series(np.linspace(400, 500, n), index=dates)
        df = pd.DataFrame({"close": close})

        wf = WaveletFeatures()
        result = wf.analyze_current_wavelet(df)
        assert result is not None
        assert result["wavelet_regime"] == "TRENDING"

    def test_noisy_regime_on_white_noise_data(self):
        """Pure white-noise prices → should report NOISY."""
        np.random.seed(42)
        n = 252
        dates = pd.bdate_range("2023-01-03", periods=n)
        close = pd.Series(450.0 + np.random.normal(0, 15, n), index=dates)
        df = pd.DataFrame({"close": close})

        wf = WaveletFeatures()
        result = wf.analyze_current_wavelet(df)
        assert result is not None
        assert result["wavelet_regime"] == "NOISY"


# =============================================================================
# 9. Custom window sizes
# =============================================================================

class TestCustomWindows:

    def test_custom_windows_produce_features(self):
        n = 252
        dates = pd.bdate_range("2023-01-03", periods=n)
        np.random.seed(1)
        close = 400.0 * np.exp(np.cumsum(np.random.normal(0.0002, 0.009, n)))
        df = pd.DataFrame({"close": close}, index=dates)

        wf = WaveletFeatures(short_window=2, long_window=10)
        result = wf.create_wavelet_features(df)
        wav_cols = [c for c in result.columns if c.startswith("wav_")]
        assert len(wav_cols) == 10

    def test_different_windows_produce_different_trends(self):
        n = 252
        dates = pd.bdate_range("2023-01-03", periods=n)
        np.random.seed(5)
        close = 400.0 * np.exp(np.cumsum(np.random.normal(0.0003, 0.01, n)))
        df = pd.DataFrame({"close": close}, index=dates)

        wf_default = WaveletFeatures(short_window=3, long_window=5)
        wf_wide = WaveletFeatures(short_window=5, long_window=10)

        r1 = wf_default.create_wavelet_features(df.copy())
        r2 = wf_wide.create_wavelet_features(df.copy())

        # wav_trend_5d should differ between the two
        assert not np.allclose(r1["wav_trend_5d"].values, r2["wav_trend_5d"].values)
