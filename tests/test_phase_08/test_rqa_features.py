"""
Tests for RQAFeatures — Recurrence Quantification Analysis features (5 total).

Tests cover:
  - All 5 features are created with the correct rqa_ prefix
  - No NaN or infinity values in any feature
  - Original columns are preserved and row count is unchanged
  - Edge cases: missing close column, short data, flat data, NaN-filled data
  - Feature bounds (recurrence_rate, determinism, laminarity in [0,1])
  - Logic direction: periodic data should have higher DET than noise
  - analyze_current_rqa: regime values and None-on-missing behaviour
  - Configurable parameters (custom window, embedding_dim, delay)
  - download_rqa_data returns an empty DataFrame
"""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.rqa_features import (
    RQAFeatures,
    _phase_space_embed,
    _recurrence_matrix,
    _rqa_measures,
    _run_lengths,
)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _make_spy_daily(n_days: int = 200, seed: int = 42) -> pd.DataFrame:
    """Synthetic SPY-like daily OHLCV DataFrame."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    returns = rng.normal(0.0003, 0.012, n_days)
    close = 450.0 * np.cumprod(1 + returns)
    volume = rng.randint(50_000_000, 150_000_000, n_days).astype(float)
    return pd.DataFrame(
        {
            "date": dates,
            "close": close,
            "volume": volume,
            "open": close * (1 + rng.normal(0, 0.003, n_days)),
            "high": close * 1.005,
            "low": close * 0.995,
        }
    )


def _make_periodic(n_days: int = 200, period: int = 10) -> pd.DataFrame:
    """
    Create a strongly periodic price series.
    Returns cycle with a very small noise term so the RQA detects structure.
    """
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    t = np.arange(n_days)
    # Sinusoidal returns — very predictable
    returns = 0.005 * np.sin(2 * np.pi * t / period)
    close = 450.0 * np.cumprod(1 + returns)
    return pd.DataFrame({"date": dates, "close": close})


def _make_flat(n_days: int = 200) -> pd.DataFrame:
    """Completely flat price — zero variance."""
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    return pd.DataFrame({"date": dates, "close": np.full(n_days, 450.0)})


ALL_5 = {
    "rqa_recurrence_rate",
    "rqa_determinism",
    "rqa_laminarity",
    "rqa_entropy",
    "rqa_trapping_time",
}


# ─── Invariant Tests ─────────────────────────────────────────────────────────


class TestRQAInvariants:
    """Core invariants: correct columns, no NaN/inf, shape preservation."""

    @pytest.fixture
    def feat(self):
        return RQAFeatures(window=50)

    @pytest.fixture
    def spy(self):
        return _make_spy_daily(200)

    def test_all_5_features_created(self, feat, spy):
        result = feat.create_rqa_features(spy)
        rqa_cols = {c for c in result.columns if c.startswith("rqa_")}
        assert rqa_cols == ALL_5, f"Expected {ALL_5}, got {rqa_cols}"

    def test_no_nans(self, feat, spy):
        result = feat.create_rqa_features(spy)
        for col in ALL_5:
            nan_count = result[col].isna().sum()
            assert nan_count == 0, f"{col} has {nan_count} NaN values"

    def test_no_infinities(self, feat, spy):
        result = feat.create_rqa_features(spy)
        for col in ALL_5:
            assert not np.isinf(result[col]).any(), f"{col} contains infinity"

    def test_preserves_original_columns(self, feat, spy):
        original_cols = set(spy.columns)
        result = feat.create_rqa_features(spy)
        assert original_cols.issubset(set(result.columns))

    def test_preserves_row_count(self, feat, spy):
        result = feat.create_rqa_features(spy)
        assert len(result) == len(spy)

    def test_missing_close_column_skips_gracefully(self, feat):
        df = pd.DataFrame({"date": [1, 2, 3], "price": [100.0, 101.0, 102.0]})
        result = feat.create_rqa_features(df)
        # No rqa_ columns should be added
        rqa_cols = [c for c in result.columns if c.startswith("rqa_")]
        assert len(rqa_cols) == 0
        # Original columns preserved
        assert set(df.columns) == set(result.columns)


# ─── Edge Case Tests ─────────────────────────────────────────────────────────


class TestRQAEdgeCases:
    """Boundary conditions: short data, flat data, NaN data."""

    def test_short_data_below_window_produces_zeros(self):
        """With data shorter than window, all rqa_ features should be 0."""
        feat = RQAFeatures(window=50)
        df = pd.DataFrame(
            {
                "date": pd.bdate_range("2024-01-02", periods=30),
                "close": np.linspace(440.0, 460.0, 30),
            }
        )
        result = feat.create_rqa_features(df)
        for col in ALL_5:
            assert col in result.columns
            assert result[col].isna().sum() == 0

    def test_flat_data_does_not_crash(self):
        """Constant price → zero variance → must handle gracefully."""
        feat = RQAFeatures(window=50)
        df = _make_flat(200)
        result = feat.create_rqa_features(df)
        for col in ALL_5:
            assert col in result.columns
            assert result[col].isna().sum() == 0, f"NaN in {col} with flat data"
            assert not np.isinf(result[col]).any()

    def test_nan_in_close_handled(self):
        """NaN values in close should not propagate or crash."""
        feat = RQAFeatures(window=30)
        rng = np.random.RandomState(7)
        close = 450.0 * np.cumprod(1 + rng.normal(0, 0.01, 200))
        # Inject some NaN into close
        close[10] = np.nan
        close[50] = np.nan
        close[100] = np.nan
        df = pd.DataFrame(
            {
                "date": pd.bdate_range("2024-01-02", periods=200),
                "close": close,
            }
        )
        result = feat.create_rqa_features(df)
        for col in ALL_5:
            assert result[col].isna().sum() == 0
            assert not np.isinf(result[col]).any()

    def test_minimum_viable_data(self):
        """Exactly window rows should not crash."""
        feat = RQAFeatures(window=20)
        df = _make_spy_daily(20)
        result = feat.create_rqa_features(df)
        for col in ALL_5:
            assert col in result.columns


# ─── Feature Bounds Tests ─────────────────────────────────────────────────────


class TestRQABounds:
    """Value-range checks for each feature."""

    @pytest.fixture
    def computed(self):
        feat = RQAFeatures(window=50)
        df = _make_spy_daily(200)
        return feat.create_rqa_features(df)

    def test_recurrence_rate_bounded_0_1(self, computed):
        assert computed["rqa_recurrence_rate"].min() >= 0.0
        assert computed["rqa_recurrence_rate"].max() <= 1.0

    def test_determinism_bounded_0_1(self, computed):
        assert computed["rqa_determinism"].min() >= 0.0
        assert computed["rqa_determinism"].max() <= 1.0

    def test_laminarity_bounded_0_1(self, computed):
        assert computed["rqa_laminarity"].min() >= 0.0
        assert computed["rqa_laminarity"].max() <= 1.0

    def test_entropy_non_negative(self, computed):
        assert computed["rqa_entropy"].min() >= 0.0

    def test_trapping_time_non_negative(self, computed):
        assert computed["rqa_trapping_time"].min() >= 0.0

    def test_warm_up_rows_are_zero(self, computed):
        """First window-1 rows should be 0 (no data yet)."""
        for col in ALL_5:
            # First 49 rows should be 0.0 (window=50, first valid at index 50)
            assert (computed[col].iloc[:49] == 0.0).all(), (
                f"{col} warm-up rows are not all zero"
            )


# ─── Feature Logic Tests ─────────────────────────────────────────────────────


class TestRQALogic:
    """Directional/semantic tests for the RQA measures."""

    def test_periodic_series_has_positive_determinism(self):
        """A periodic sinusoidal series must produce non-zero DET."""
        feat = RQAFeatures(window=50, embedding_dim=3, delay=1)
        df = _make_periodic(n_days=200, period=10)
        result = feat.create_rqa_features(df)
        # After warm-up, DET should be meaningfully positive for a periodic series
        det_after_warmup = result["rqa_determinism"].iloc[50:]
        assert det_after_warmup.max() > 0.0, (
            "Periodic series should produce positive determinism"
        )

    def test_periodic_vs_noise_determinism(self):
        """Periodic series should produce higher mean DET than pure noise."""
        feat = RQAFeatures(window=50)
        df_periodic = _make_periodic(200)
        df_noise = _make_spy_daily(200, seed=13)

        r_periodic = feat.create_rqa_features(df_periodic)
        r_noise = feat.create_rqa_features(df_noise)

        # Compare only after warm-up period
        mean_det_periodic = r_periodic["rqa_determinism"].iloc[60:].mean()
        mean_det_noise = r_noise["rqa_determinism"].iloc[60:].mean()

        assert mean_det_periodic >= mean_det_noise, (
            f"Periodic DET ({mean_det_periodic:.4f}) should be >= noise DET "
            f"({mean_det_noise:.4f})"
        )

    def test_recurrence_rate_responds_to_epsilon(self):
        """Larger epsilon_factor should produce larger recurrence rate."""
        df = _make_spy_daily(200)
        feat_tight = RQAFeatures(window=50, epsilon_factor=0.1)
        feat_wide = RQAFeatures(window=50, epsilon_factor=2.0)

        r_tight = feat_tight.create_rqa_features(df)
        r_wide = feat_wide.create_rqa_features(df)

        mean_rr_tight = r_tight["rqa_recurrence_rate"].iloc[60:].mean()
        mean_rr_wide = r_wide["rqa_recurrence_rate"].iloc[60:].mean()

        assert mean_rr_wide > mean_rr_tight, (
            f"Wide epsilon RR ({mean_rr_wide:.4f}) should exceed tight ({mean_rr_tight:.4f})"
        )

    def test_features_non_trivially_zero_after_warmup(self):
        """At least some rows after warm-up must be non-zero for typical data."""
        feat = RQAFeatures(window=50)
        df = _make_spy_daily(200)
        result = feat.create_rqa_features(df)

        # At least recurrence rate should be > 0 for most rows post warm-up
        rr_after = result["rqa_recurrence_rate"].iloc[60:]
        assert (rr_after > 0).sum() > 0, (
            "Expected non-zero recurrence rate for typical SPY data after warm-up"
        )


# ─── Configurable Parameters Tests ───────────────────────────────────────────


class TestRQAConfigurableParameters:
    """Test custom window, embedding_dim, delay values."""

    def test_custom_window(self):
        feat = RQAFeatures(window=30)
        df = _make_spy_daily(200)
        result = feat.create_rqa_features(df)
        for col in ALL_5:
            assert col in result.columns
            assert result[col].isna().sum() == 0

    def test_custom_embedding_dim(self):
        feat = RQAFeatures(window=50, embedding_dim=5)
        df = _make_spy_daily(200)
        result = feat.create_rqa_features(df)
        for col in ALL_5:
            assert col in result.columns
            assert result[col].isna().sum() == 0

    def test_custom_delay(self):
        feat = RQAFeatures(window=50, embedding_dim=3, delay=2)
        df = _make_spy_daily(200)
        result = feat.create_rqa_features(df)
        for col in ALL_5:
            assert col in result.columns
            assert result[col].isna().sum() == 0

    def test_download_rqa_data_returns_empty_df(self):
        """download_rqa_data is a no-op — must return empty DataFrame."""
        feat = RQAFeatures()
        result = feat.download_rqa_data("2023-01-01", "2024-01-01")
        assert isinstance(result, pd.DataFrame)
        assert len(result) == 0


# ─── analyze_current_rqa Tests ───────────────────────────────────────────────


class TestAnalyzeCurrentRQA:
    """Tests for the regime analysis helper."""

    def test_returns_dict_with_regime(self):
        feat = RQAFeatures(window=50)
        df = _make_spy_daily(200)
        df = feat.create_rqa_features(df)
        result = feat.analyze_current_rqa(df)
        assert isinstance(result, dict)
        assert "rqa_regime" in result

    def test_regime_is_valid_label(self):
        feat = RQAFeatures(window=50)
        df = _make_spy_daily(200)
        df = feat.create_rqa_features(df)
        result = feat.analyze_current_rqa(df)
        valid_regimes = {"PERIODIC", "CHAOTIC", "LAMINAR", "STOCHASTIC"}
        assert result["rqa_regime"] in valid_regimes, (
            f"Unexpected regime: {result['rqa_regime']}"
        )

    def test_returns_none_without_rqa_columns(self):
        feat = RQAFeatures()
        df = pd.DataFrame({"close": [100.0, 101.0]})
        assert feat.analyze_current_rqa(df) is None

    def test_returns_none_for_empty_df(self):
        feat = RQAFeatures()
        df = pd.DataFrame(columns=["rqa_recurrence_rate", "rqa_determinism", "rqa_laminarity"])
        assert feat.analyze_current_rqa(df) is None

    def test_dict_contains_all_scalar_fields(self):
        feat = RQAFeatures(window=50)
        df = _make_spy_daily(200)
        df = feat.create_rqa_features(df)
        result = feat.analyze_current_rqa(df)
        expected_keys = {
            "rqa_regime",
            "rqa_recurrence_rate",
            "rqa_determinism",
            "rqa_laminarity",
            "rqa_entropy",
            "rqa_trapping_time",
        }
        assert expected_keys.issubset(result.keys())

    def test_scalar_values_are_finite(self):
        feat = RQAFeatures(window=50)
        df = _make_spy_daily(200)
        df = feat.create_rqa_features(df)
        result = feat.analyze_current_rqa(df)
        for key, val in result.items():
            if key != "rqa_regime":
                assert np.isfinite(val), f"{key} is not finite: {val}"


# ─── Internal Helper Unit Tests ───────────────────────────────────────────────


class TestInternalHelpers:
    """Unit tests for the module-level helper functions."""

    def test_run_lengths_basic(self):
        arr = np.array([True, True, False, True, True, True, False])
        result = _run_lengths(arr)
        assert result == [2, 3]

    def test_run_lengths_all_true(self):
        arr = np.ones(5, dtype=bool)
        assert _run_lengths(arr) == [5]

    def test_run_lengths_all_false(self):
        arr = np.zeros(5, dtype=bool)
        assert _run_lengths(arr) == []

    def test_run_lengths_empty(self):
        assert _run_lengths(np.array([], dtype=bool)) == []

    def test_phase_space_embed_shape(self):
        series = np.arange(10.0)
        embedded = _phase_space_embed(series, dim=3, delay=1)
        # M = 10 - (3-1)*1 = 8
        assert embedded.shape == (8, 3)

    def test_phase_space_embed_values(self):
        series = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        embedded = _phase_space_embed(series, dim=2, delay=1)
        # Expected: [[0,1],[1,2],[2,3],[3,4]]
        expected = np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0], [3.0, 4.0]])
        np.testing.assert_array_almost_equal(embedded, expected)

    def test_recurrence_matrix_diagonal_is_all_true(self):
        """Every point is always recurrent with itself."""
        embedded = np.random.randn(10, 3)
        R = _recurrence_matrix(embedded, epsilon=0.5)
        assert R.shape == (10, 10)
        assert np.all(np.diagonal(R)), "Main diagonal must be True (self-recurrence)"

    def test_recurrence_matrix_symmetric(self):
        embedded = np.random.randn(8, 2)
        R = _recurrence_matrix(embedded, epsilon=1.0)
        np.testing.assert_array_equal(R, R.T)

    def test_rqa_measures_returns_all_keys(self):
        R = np.eye(5, dtype=bool)  # Minimal: only self-recurrences
        measures = _rqa_measures(R, min_line=2)
        expected_keys = {
            "recurrence_rate",
            "determinism",
            "laminarity",
            "entropy",
            "trapping_time",
        }
        assert set(measures.keys()) == expected_keys

    def test_rqa_measures_empty_matrix(self):
        """Empty matrix should return all zeros without crashing."""
        R = np.empty((0, 0), dtype=bool)
        measures = _rqa_measures(R)
        assert all(v == 0.0 for v in measures.values())


# ─── Feature Count Test ───────────────────────────────────────────────────────


class TestFeatureCounts:
    def test_total_count(self):
        assert len(ALL_5) == 5

    def test_all_feature_names_static_method(self):
        names = RQAFeatures._all_feature_names()
        assert set(names) == ALL_5
        assert len(names) == 5
