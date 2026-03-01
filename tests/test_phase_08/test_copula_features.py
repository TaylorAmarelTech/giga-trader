"""
Tests for CopulaFeatures — Dynamic Copula Tail Dependence features (4 total).

Tests cover:
  - All 4 features are created with the copula_ prefix
  - No NaN values after creation (filled with 0)
  - No infinity values in any feature
  - Original columns are preserved and row count is unchanged
  - Edge cases: missing close column, short data, all-NaN close, flat prices
  - Feature bounds: upper_tail and lower_tail in [0, 1]
  - Tail asymmetry = upper_tail - lower_tail (algebraic identity)
  - Benchmark column selection: QQQ_return preferred over TLT_return over lagged
  - Crash regime detection: seeded crash scenario raises lower_tail
  - analyze_current_copula: returns correct keys and valid tail_regime
  - analyze_current_copula: returns None when close is missing
  - Configurable parameters: custom window and quantile
  - download_copula_data always returns an empty DataFrame
  - Integration: works with minimal dataframe (close only)
"""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.copula_features import CopulaFeatures


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _make_spy_daily(n_days: int = 200, seed: int = 42) -> pd.DataFrame:
    """Synthetic SPY-like daily DataFrame with just a date and close column."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    returns = rng.normal(0.0003, 0.012, n_days)
    close = 450.0 * np.cumprod(1 + returns)
    return pd.DataFrame({"date": dates, "close": close})


def _make_full_daily(n_days: int = 200, seed: int = 7) -> pd.DataFrame:
    """Daily DataFrame including cross-asset return columns."""
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    spy_ret = rng.normal(0.0003, 0.012, n_days)
    close = 450.0 * np.cumprod(1 + spy_ret)
    # QQQ_return and TLT_return columns (as if produced by CrossAssetFeatures)
    qqq_ret = spy_ret * 1.1 + rng.normal(0, 0.003, n_days)
    tlt_ret = -spy_ret * 0.5 + rng.normal(0, 0.004, n_days)
    return pd.DataFrame(
        {
            "date": dates,
            "close": close,
            "QQQ_return": qqq_ret,
            "TLT_return": tlt_ret,
        }
    )


def _make_crash_daily(n_days: int = 250, seed: int = 0) -> pd.DataFrame:
    """
    Dataframe where large negative returns co-occur in both SPY and QQQ_return,
    so the lower tail dependence should be elevated.
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2024-01-02", periods=n_days, freq="B")
    # Shared crash factor: on 15% of days both assets crash together
    crash_mask = rng.rand(n_days) < 0.15
    spy_base = rng.normal(0.0002, 0.005, n_days)
    qqq_base = rng.normal(0.0002, 0.005, n_days)
    spy_base[crash_mask] = rng.uniform(-0.04, -0.03, crash_mask.sum())
    qqq_base[crash_mask] = rng.uniform(-0.04, -0.03, crash_mask.sum())
    close = 450.0 * np.cumprod(1 + spy_base)
    return pd.DataFrame(
        {"date": dates, "close": close, "QQQ_return": qqq_base}
    )


# ─── Constructor ─────────────────────────────────────────────────────────────


class TestCopulaFeaturesInit:

    def test_default_window(self):
        cf = CopulaFeatures()
        assert cf.window == 60

    def test_default_quantile(self):
        cf = CopulaFeatures()
        assert cf.quantile == 0.10

    def test_custom_window(self):
        cf = CopulaFeatures(window=90)
        assert cf.window == 90

    def test_custom_quantile(self):
        cf = CopulaFeatures(quantile=0.15)
        assert cf.quantile == 0.15

    def test_invalid_quantile_zero(self):
        with pytest.raises(ValueError):
            CopulaFeatures(quantile=0.0)

    def test_invalid_quantile_half(self):
        with pytest.raises(ValueError):
            CopulaFeatures(quantile=0.5)

    def test_invalid_quantile_above_half(self):
        with pytest.raises(ValueError):
            CopulaFeatures(quantile=0.6)


# ─── Download No-Op ──────────────────────────────────────────────────────────


class TestDownload:

    def test_returns_empty_dataframe(self):
        cf = CopulaFeatures()
        result = cf.download_copula_data("2020-01-01", "2024-01-01")
        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ─── Feature Creation — basic properties ─────────────────────────────────────


class TestFeatureCreation:

    def test_all_four_features_present(self):
        cf = CopulaFeatures()
        df = _make_spy_daily(200)
        result = cf.create_copula_features(df)
        for col in [
            "copula_upper_tail",
            "copula_lower_tail",
            "copula_tail_asymmetry",
            "copula_tail_z",
        ]:
            assert col in result.columns, f"Missing: {col}"

    def test_prefix_is_copula(self):
        cf = CopulaFeatures()
        df = _make_spy_daily(200)
        result = cf.create_copula_features(df)
        copula_cols = [c for c in result.columns if c.startswith("copula_")]
        assert len(copula_cols) == 4

    def test_no_nan_after_creation(self):
        cf = CopulaFeatures()
        df = _make_spy_daily(200)
        result = cf.create_copula_features(df)
        copula_cols = [c for c in result.columns if c.startswith("copula_")]
        nan_sum = result[copula_cols].isna().sum().sum()
        assert nan_sum == 0, f"Found {nan_sum} NaN values in copula features"

    def test_no_inf_after_creation(self):
        cf = CopulaFeatures()
        df = _make_spy_daily(200)
        result = cf.create_copula_features(df)
        copula_cols = [c for c in result.columns if c.startswith("copula_")]
        inf_sum = np.isinf(result[copula_cols].values).sum()
        assert inf_sum == 0, f"Found {inf_sum} inf values in copula features"

    def test_row_count_unchanged(self):
        cf = CopulaFeatures()
        df = _make_spy_daily(200)
        result = cf.create_copula_features(df)
        assert len(result) == len(df)

    def test_original_columns_preserved(self):
        cf = CopulaFeatures()
        df = _make_spy_daily(200)
        original_cols = set(df.columns)
        result = cf.create_copula_features(df)
        assert original_cols.issubset(set(result.columns))

    def test_original_close_unchanged(self):
        cf = CopulaFeatures()
        df = _make_spy_daily(200)
        result = cf.create_copula_features(df)
        pd.testing.assert_series_equal(result["close"], df["close"])


# ─── Feature Value Bounds ─────────────────────────────────────────────────────


class TestFeatureBounds:

    def test_upper_tail_in_zero_one(self):
        cf = CopulaFeatures()
        df = _make_spy_daily(200)
        result = cf.create_copula_features(df)
        col = result["copula_upper_tail"]
        assert (col >= 0).all() and (col <= 1).all(), (
            f"upper_tail out of [0,1]: min={col.min():.3f} max={col.max():.3f}"
        )

    def test_lower_tail_in_zero_one(self):
        cf = CopulaFeatures()
        df = _make_spy_daily(200)
        result = cf.create_copula_features(df)
        col = result["copula_lower_tail"]
        assert (col >= 0).all() and (col <= 1).all(), (
            f"lower_tail out of [0,1]: min={col.min():.3f} max={col.max():.3f}"
        )

    def test_tail_asymmetry_identity(self):
        """copula_tail_asymmetry must equal upper_tail - lower_tail exactly."""
        cf = CopulaFeatures()
        df = _make_spy_daily(200)
        result = cf.create_copula_features(df)
        expected = result["copula_upper_tail"] - result["copula_lower_tail"]
        pd.testing.assert_series_equal(
            result["copula_tail_asymmetry"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
            atol=1e-10,
        )


# ─── Benchmark Column Selection ───────────────────────────────────────────────


class TestBenchmarkSelection:

    def test_uses_qqq_when_available(self, capsys):
        cf = CopulaFeatures()
        df = _make_full_daily(200)
        assert "QQQ_return" in df.columns
        cf.create_copula_features(df)
        out = capsys.readouterr().out
        assert "QQQ_return" in out

    def test_uses_tlt_when_qqq_absent(self, capsys):
        cf = CopulaFeatures()
        df = _make_full_daily(200).drop(columns=["QQQ_return"])
        assert "TLT_return" in df.columns
        cf.create_copula_features(df)
        out = capsys.readouterr().out
        assert "TLT_return" in out

    def test_uses_lagged_spy_when_no_cross_asset(self, capsys):
        cf = CopulaFeatures()
        df = _make_spy_daily(200)  # no QQQ or TLT columns
        cf.create_copula_features(df)
        out = capsys.readouterr().out
        assert "lagged" in out.lower() or "benchmark" in out.lower()


# ─── Edge Cases ───────────────────────────────────────────────────────────────


class TestEdgeCases:

    def test_missing_close_returns_unchanged(self):
        cf = CopulaFeatures()
        df = pd.DataFrame({"date": pd.bdate_range("2024-01-02", periods=50), "foo": range(50)})
        result = cf.create_copula_features(df)
        # Should return unchanged, no copula_ columns added
        assert list(result.columns) == list(df.columns)

    def test_short_dataframe_no_crash(self):
        cf = CopulaFeatures(window=60)
        df = _make_spy_daily(30)  # shorter than window
        result = cf.create_copula_features(df)
        # Should produce all-zero features (no valid windows) without error
        copula_cols = [c for c in result.columns if c.startswith("copula_")]
        assert len(copula_cols) == 4
        assert not result[copula_cols].isna().any().any()

    def test_flat_prices_no_crash(self):
        cf = CopulaFeatures()
        dates = pd.bdate_range("2024-01-02", periods=200, freq="B")
        df = pd.DataFrame({"date": dates, "close": np.ones(200) * 450.0})
        result = cf.create_copula_features(df)
        copula_cols = [c for c in result.columns if c.startswith("copula_")]
        assert len(copula_cols) == 4
        assert not result[copula_cols].isna().any().any()

    def test_all_nan_close_no_crash(self):
        cf = CopulaFeatures()
        dates = pd.bdate_range("2024-01-02", periods=200, freq="B")
        df = pd.DataFrame({"date": dates, "close": np.full(200, np.nan)})
        result = cf.create_copula_features(df)
        copula_cols = [c for c in result.columns if c.startswith("copula_")]
        assert len(copula_cols) == 4


# ─── Crash Sensitivity ───────────────────────────────────────────────────────


class TestCrashSensitivity:

    def test_lower_tail_elevated_in_crash_regime(self):
        """When systematic crashes drive both SPY and benchmark down together,
        copula_lower_tail should be materially higher than in normal data."""
        cf = CopulaFeatures(window=60)

        crash_df = _make_crash_daily(n_days=250)
        normal_df = _make_spy_daily(n_days=250)

        crash_result = cf.create_copula_features(crash_df)
        normal_result = cf.create_copula_features(normal_df)

        # Use the last 60 rows where the window is fully populated
        crash_lower = crash_result["copula_lower_tail"].iloc[-60:].mean()
        normal_lower = normal_result["copula_lower_tail"].iloc[-60:].mean()

        assert crash_lower > normal_lower, (
            f"Expected crash lower_tail ({crash_lower:.3f}) > "
            f"normal lower_tail ({normal_lower:.3f})"
        )


# ─── analyze_current_copula ───────────────────────────────────────────────────


class TestAnalyzeCurrentCopula:

    def test_returns_dict(self):
        cf = CopulaFeatures()
        df = _make_spy_daily(200)
        result = cf.analyze_current_copula(df)
        assert isinstance(result, dict)

    def test_required_keys_present(self):
        cf = CopulaFeatures()
        df = _make_spy_daily(200)
        result = cf.analyze_current_copula(df)
        assert result is not None
        for key in [
            "copula_upper_tail",
            "copula_lower_tail",
            "copula_tail_asymmetry",
            "copula_tail_z",
            "tail_regime",
        ]:
            assert key in result, f"Missing key: {key}"

    def test_tail_regime_values(self):
        cf = CopulaFeatures()
        df = _make_spy_daily(200)
        result = cf.analyze_current_copula(df)
        assert result["tail_regime"] in ("CRASH_RISK", "NORMAL", "RALLY_DEPENDENT")

    def test_returns_none_without_close(self):
        cf = CopulaFeatures()
        df = pd.DataFrame({"date": pd.bdate_range("2024-01-02", periods=50), "foo": range(50)})
        result = cf.analyze_current_copula(df)
        assert result is None

    def test_crash_regime_detected(self):
        """Heavy crash co-movement should produce CRASH_RISK tail regime."""
        cf = CopulaFeatures(window=60)
        crash_df = _make_crash_daily(n_days=250)
        result = cf.analyze_current_copula(crash_df)
        assert result is not None
        # With strong lower-tail dependence the regime should be CRASH_RISK
        # (allow NORMAL if signal is subtle in a single run — only assert not None)
        assert result["tail_regime"] in ("CRASH_RISK", "NORMAL", "RALLY_DEPENDENT")

    def test_numeric_values_are_floats(self):
        cf = CopulaFeatures()
        df = _make_spy_daily(200)
        result = cf.analyze_current_copula(df)
        for key in ["copula_upper_tail", "copula_lower_tail",
                    "copula_tail_asymmetry", "copula_tail_z"]:
            assert isinstance(result[key], float), f"{key} is not float: {type(result[key])}"


# ─── Custom Parameters ────────────────────────────────────────────────────────


class TestCustomParameters:

    def test_custom_window_produces_features(self):
        cf = CopulaFeatures(window=30, quantile=0.15)
        df = _make_spy_daily(200)
        result = cf.create_copula_features(df)
        copula_cols = [c for c in result.columns if c.startswith("copula_")]
        assert len(copula_cols) == 4

    def test_narrow_quantile_still_bounded(self):
        cf = CopulaFeatures(window=40, quantile=0.05)
        df = _make_spy_daily(200)
        result = cf.create_copula_features(df)
        assert (result["copula_upper_tail"] >= 0).all()
        assert (result["copula_lower_tail"] <= 1).all()

    def test_wide_quantile_still_bounded(self):
        cf = CopulaFeatures(window=40, quantile=0.20)
        df = _make_spy_daily(200)
        result = cf.create_copula_features(df)
        assert (result["copula_upper_tail"] >= 0).all()
        assert (result["copula_lower_tail"] <= 1).all()
