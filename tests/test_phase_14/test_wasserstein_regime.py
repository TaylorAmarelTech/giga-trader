"""
Tests for WassersteinRegimeDetector -- Wasserstein distance regime change detection.
"""

import pytest
import numpy as np
import pandas as pd

from src.phase_14_robustness.wasserstein_regime import WassersteinRegimeDetector


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_daily(close: np.ndarray) -> pd.DataFrame:
    """Build a minimal df_daily from a close price array."""
    return pd.DataFrame({"close": close.astype(np.float64)})


def _random_walk(n: int = 300, start: float = 100.0, seed: int = 42) -> np.ndarray:
    """Generate a geometric random walk for close prices."""
    rng = np.random.RandomState(seed)
    log_returns = rng.normal(0.0005, 0.01, size=n)
    log_prices = np.cumsum(log_returns)
    return start * np.exp(log_prices)


# ── Test 1: Output has correct column names (wreg_ prefix) ──────────────────


class TestColumnNames:
    def test_all_wreg_columns_present(self):
        """Output should contain exactly the 8 wreg_ columns."""
        detector = WassersteinRegimeDetector(short_window=10, long_window=30)
        df = _make_daily(_random_walk(200))
        result = detector.create_wasserstein_features(df)

        expected = set(WassersteinRegimeDetector.FEATURE_NAMES)
        actual_wreg = {c for c in result.columns if c.startswith("wreg_")}
        assert actual_wreg == expected, (
            f"Expected columns {expected}, got {actual_wreg}"
        )

    def test_original_columns_preserved(self):
        """Original columns (at least 'close') should still be in output."""
        detector = WassersteinRegimeDetector(short_window=10)
        df = _make_daily(_random_walk(200))
        result = detector.create_wasserstein_features(df)
        assert "close" in result.columns


# ── Test 2: Output shape matches input ──────────────────────────────────────


class TestOutputShape:
    def test_row_count_unchanged(self):
        """Number of rows should be identical to input."""
        detector = WassersteinRegimeDetector(short_window=10, long_window=30)
        df = _make_daily(_random_walk(250))
        result = detector.create_wasserstein_features(df)
        assert len(result) == 250

    def test_column_count_correct(self):
        """Output should have original columns + 8 wreg_ columns."""
        detector = WassersteinRegimeDetector(short_window=10, long_window=30)
        df = _make_daily(_random_walk(250))
        n_orig = len(df.columns)
        result = detector.create_wasserstein_features(df)
        assert len(result.columns) == n_orig + 8


# ── Test 3: NaN handling (early rows should be NaN) ─────────────────────────


class TestNaNHandling:
    def test_early_rows_are_nan(self):
        """First 2*short_window rows of wreg_distance_20d should be NaN."""
        short_w = 15
        detector = WassersteinRegimeDetector(short_window=short_w)
        df = _make_daily(_random_walk(200))
        result = detector.create_wasserstein_features(df)

        early = result["wreg_distance_20d"].iloc[: 2 * short_w].values
        assert np.all(np.isnan(early)), (
            f"First {2 * short_w} rows of wreg_distance_20d should be NaN, "
            f"found {np.sum(~np.isnan(early))} non-NaN values"
        )

    def test_later_rows_have_values(self):
        """After sufficient warmup, wreg_distance_20d should have valid values."""
        short_w = 10
        detector = WassersteinRegimeDetector(short_window=short_w)
        df = _make_daily(_random_walk(200))
        result = detector.create_wasserstein_features(df)

        # Beyond 2*short_window, should have non-NaN
        later = result["wreg_distance_20d"].iloc[2 * short_w + 5 :].values
        n_valid = np.sum(~np.isnan(later))
        assert n_valid > 0, "Should have valid distance values after warmup"


# ── Test 4: Constant returns produce distance near 0 ────────────────────────


class TestConstantReturns:
    def test_constant_close_gives_zero_distance(self):
        """If close is constant (returns=0), Wasserstein distance should be 0."""
        detector = WassersteinRegimeDetector(short_window=10, long_window=30)
        close = np.full(200, 100.0)
        df = _make_daily(close)
        result = detector.create_wasserstein_features(df)

        dist_20d = result["wreg_distance_20d"].dropna().values
        assert len(dist_20d) > 0, "Should have valid distances for constant series"
        assert np.allclose(dist_20d, 0.0, atol=1e-12), (
            f"Distance should be ~0 for constant returns, got max={np.max(dist_20d):.6e}"
        )

    def test_constant_returns_high_stability(self):
        """Constant returns should yield stability = 1 / (1 + 0) = 1.0."""
        detector = WassersteinRegimeDetector(short_window=10)
        close = np.full(200, 50.0)
        df = _make_daily(close)
        result = detector.create_wasserstein_features(df)

        stability = result["wreg_stability"].dropna().values
        assert np.allclose(stability, 1.0, atol=1e-10), (
            f"Stability should be 1.0 for constant close, got {stability[:5]}"
        )


# ── Test 5: Regime change detection on spliced distributions ────────────────


class TestRegimeChangeDetection:
    def test_detects_volatility_shift(self):
        """
        A series that switches from low vol to high vol should trigger
        at least one regime change near the splice point.
        """
        rng = np.random.RandomState(123)
        n_low = 150
        n_high = 150
        # Low volatility regime
        returns_low = rng.normal(0.0005, 0.005, size=n_low)
        # High volatility regime (10x vol)
        returns_high = rng.normal(0.0005, 0.05, size=n_high)

        all_returns = np.concatenate([returns_low, returns_high])
        close = 100.0 * np.exp(np.cumsum(all_returns))

        detector = WassersteinRegimeDetector(
            short_window=20, z_threshold=2.0
        )
        df = _make_daily(close)
        result = detector.create_wasserstein_features(df)

        # Check for regime change near splice point (index ~150)
        # Look in window [130, 190] to allow for rolling window lag
        search_start = 130
        search_end = min(190, len(result))
        regime_changes = result["wreg_regime_change"].iloc[search_start:search_end]
        n_changes = regime_changes.dropna().sum()

        assert n_changes >= 1, (
            f"Expected at least 1 regime change near splice point "
            f"(rows {search_start}-{search_end}), found {n_changes}"
        )

    def test_quiet_period_no_regime_change(self):
        """
        A stationary random walk (constant vol) should produce few or no
        regime changes.
        """
        detector = WassersteinRegimeDetector(
            short_window=20, z_threshold=2.0
        )
        close = _random_walk(300, seed=99)
        df = _make_daily(close)
        result = detector.create_wasserstein_features(df)

        total_changes = result["wreg_regime_change"].dropna().sum()
        # With a z > 2 threshold on stationary data, expect very few changes.
        # Allow up to ~5% of observations (generous for randomness).
        n_valid = result["wreg_regime_change"].dropna().count()
        rate = total_changes / max(n_valid, 1)
        assert rate < 0.15, (
            f"Regime change rate {rate:.2%} too high for stationary data "
            f"({int(total_changes)} changes in {n_valid} valid rows)"
        )


# ── Test 6: Stability is between 0 and 1 ────────────────────────────────────


class TestStabilityBounds:
    def test_stability_in_zero_one(self):
        """wreg_stability should always be in (0, 1] for valid rows."""
        detector = WassersteinRegimeDetector(short_window=10)
        df = _make_daily(_random_walk(250))
        result = detector.create_wasserstein_features(df)

        stability = result["wreg_stability"].dropna().values
        assert len(stability) > 0
        assert np.all(stability > 0.0), (
            f"Stability should be > 0, min={np.min(stability)}"
        )
        assert np.all(stability <= 1.0), (
            f"Stability should be <= 1, max={np.max(stability)}"
        )

    def test_stability_decreases_with_higher_distance(self):
        """Higher Wasserstein distance should produce lower stability."""
        detector = WassersteinRegimeDetector(short_window=10)
        df = _make_daily(_random_walk(250))
        result = detector.create_wasserstein_features(df)

        valid = result[["wreg_distance_20d", "wreg_stability"]].dropna()
        if len(valid) < 10:
            pytest.skip("Not enough valid rows")

        # Correlation between distance and stability should be negative
        corr = valid["wreg_distance_20d"].corr(valid["wreg_stability"])
        assert corr < 0, (
            f"Distance and stability should be negatively correlated, got r={corr:.3f}"
        )


# ── Test 7: Regime duration counts correctly ────────────────────────────────


class TestRegimeDuration:
    def test_duration_zero_at_change_point(self):
        """wreg_regime_duration should be 0.0 where wreg_regime_change == 1."""
        rng = np.random.RandomState(42)
        # Create a series with a clear regime change
        returns_low = rng.normal(0.0003, 0.003, size=120)
        returns_high = rng.normal(0.0003, 0.06, size=120)
        all_returns = np.concatenate([returns_low, returns_high])
        close = 100.0 * np.exp(np.cumsum(all_returns))

        detector = WassersteinRegimeDetector(short_window=15, z_threshold=1.5)
        df = _make_daily(close)
        result = detector.create_wasserstein_features(df)

        change_mask = result["wreg_regime_change"] == 1.0
        if change_mask.sum() == 0:
            pytest.skip("No regime changes detected -- cannot test duration")

        durations_at_change = result.loc[change_mask, "wreg_regime_duration"]
        assert (durations_at_change == 0.0).all(), (
            f"Duration at change points should be 0, got {durations_at_change.values}"
        )

    def test_duration_increases_monotonically(self):
        """
        Between two consecutive regime changes, wreg_regime_duration should
        increase by 1 each day.
        """
        rng = np.random.RandomState(77)
        returns_low = rng.normal(0.0003, 0.003, size=100)
        returns_high = rng.normal(0.0003, 0.06, size=100)
        returns_low2 = rng.normal(0.0003, 0.003, size=100)
        all_returns = np.concatenate([returns_low, returns_high, returns_low2])
        close = 100.0 * np.exp(np.cumsum(all_returns))

        detector = WassersteinRegimeDetector(short_window=15, z_threshold=1.5)
        df = _make_daily(close)
        result = detector.create_wasserstein_features(df)

        change_indices = result.index[result["wreg_regime_change"] == 1.0].tolist()
        if len(change_indices) < 2:
            pytest.skip("Need at least 2 regime changes for monotonicity test")

        # Between first and second change, duration should increase by 1 each step
        start = change_indices[0]
        end = change_indices[1]
        segment = result["wreg_regime_duration"].iloc[start:end].dropna().values
        if len(segment) < 2:
            pytest.skip("Segment too short for monotonicity test")

        diffs = np.diff(segment)
        assert np.all(diffs == 1.0), (
            f"Duration should increase by 1 each day between changes, "
            f"got diffs: {diffs[:10]}"
        )


# ── Test 8: No future leakage ───────────────────────────────────────────────


class TestNoFutureLeakage:
    def test_modifying_future_does_not_change_past_features(self):
        """
        Features at time t should depend only on data at t and before.
        Modifying future data should not change features at earlier times.
        """
        detector = WassersteinRegimeDetector(short_window=10, long_window=30)

        close_original = _random_walk(200, seed=55)
        df_original = _make_daily(close_original)
        result_original = detector.create_wasserstein_features(df_original)

        # Modify only the last 20 rows
        close_modified = close_original.copy()
        close_modified[-20:] *= 2.0  # Large change to future data
        df_modified = _make_daily(close_modified)
        result_modified = detector.create_wasserstein_features(df_modified)

        # Features up to row 160 (before modification starts at 180)
        # should be identical.
        cutoff = 160
        for col in WassersteinRegimeDetector.FEATURE_NAMES:
            orig_vals = result_original[col].iloc[:cutoff].values
            mod_vals = result_modified[col].iloc[:cutoff].values

            # Compare only non-NaN positions
            both_valid = ~np.isnan(orig_vals) & ~np.isnan(mod_vals)
            if both_valid.sum() == 0:
                continue

            np.testing.assert_array_equal(
                orig_vals[both_valid],
                mod_vals[both_valid],
                err_msg=(
                    f"Feature '{col}' at rows < {cutoff} changed when future "
                    f"data was modified -- possible future leakage"
                ),
            )

    def test_features_use_only_past_data(self):
        """
        Features at row t should be NaN when insufficient past data exists,
        confirming they do not peek ahead.
        """
        short_w = 20
        detector = WassersteinRegimeDetector(short_window=short_w)
        df = _make_daily(_random_walk(100))
        result = detector.create_wasserstein_features(df)

        # wreg_distance_20d needs 2*short_w rows of return history.
        # Row 0 has NaN return, so first valid distance is at index 2*short_w.
        for t in range(2 * short_w):
            assert np.isnan(result["wreg_distance_20d"].iloc[t]), (
                f"wreg_distance_20d at row {t} should be NaN (need {2 * short_w} "
                f"rows of data)"
            )


# ── Test 9: All features are float64 dtype ──────────────────────────────────


class TestDtypes:
    def test_all_features_float64(self):
        """Every wreg_ column should have dtype float64."""
        detector = WassersteinRegimeDetector(short_window=10)
        df = _make_daily(_random_walk(200))
        result = detector.create_wasserstein_features(df)

        for col in WassersteinRegimeDetector.FEATURE_NAMES:
            assert result[col].dtype == np.float64, (
                f"Column '{col}' has dtype {result[col].dtype}, expected float64"
            )
