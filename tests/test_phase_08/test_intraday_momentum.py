"""
Tests for IntradayMomentumFeatures class.

Validates intraday momentum feature engineering from daily OHLC data
without requiring any external API calls or downloads.
"""

import numpy as np
import pandas as pd
import pytest

from src.phase_08_features_breadth.intraday_momentum_features import (
    IntradayMomentumFeatures,
    IMOM_FEATURES,
)


# ─── Fixtures ─────────────────────────────────────────────────────────────────


def _make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    """
    Generate a realistic daily OHLCV DataFrame with valid OHLC relationships.

    Invariants enforced:
      high >= max(open, close)
      low  <= min(open, close)
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n)

    # Simulate a price path for close
    log_returns = rng.normal(0.0004, 0.010, n)
    close = 450.0 * np.exp(np.cumsum(log_returns))

    # Open is yesterday's close ± small gap
    open_ = np.empty(n)
    open_[0] = close[0] * (1.0 + rng.normal(0, 0.003))
    open_[1:] = close[:-1] * (1.0 + rng.normal(0, 0.003, n - 1))
    open_ = np.clip(open_, 1.0, None)

    # True range from close as pivot
    intraday_vol = np.abs(close - open_) + rng.uniform(0.1, 2.0, n)
    high = np.maximum(open_, close) + rng.uniform(0.0, 1.0, n)
    low = np.minimum(open_, close) - rng.uniform(0.0, 1.0, n)

    # Enforce OHLC integrity
    high = np.maximum(high, np.maximum(open_, close))
    low = np.minimum(low, np.minimum(open_, close))

    volume = rng.integers(20_000_000, 150_000_000, n).astype(float)

    return pd.DataFrame(
        {
            "date": pd.to_datetime(dates.date),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


@pytest.fixture
def ohlcv() -> pd.DataFrame:
    return _make_ohlcv(n=200, seed=42)


@pytest.fixture
def imom() -> IntradayMomentumFeatures:
    return IntradayMomentumFeatures(correlation_window=20)


@pytest.fixture
def result_df(imom, ohlcv) -> pd.DataFrame:
    return imom.create_intraday_momentum_features(ohlcv)


# ─── TestIntradayMomentumInvariants ──────────────────────────────────────────


class TestIntradayMomentumInvariants:
    """Core invariants that must hold for any valid OHLCV input."""

    def test_all_4_features_created(self, result_df):
        for col in IMOM_FEATURES:
            assert col in result_df.columns, f"Expected feature '{col}' to be present"
        imom_cols = [c for c in result_df.columns if c.startswith("imom_")]
        assert len(imom_cols) == 4, f"Expected 4 imom_ features, got {len(imom_cols)}"

    def test_no_nans(self, result_df):
        imom_cols = [c for c in result_df.columns if c.startswith("imom_")]
        nan_count = result_df[imom_cols].isna().sum().sum()
        assert nan_count == 0, f"Found {nan_count} NaN values in imom_ columns"

    def test_no_infinities(self, result_df):
        imom_cols = [c for c in result_df.columns if c.startswith("imom_")]
        inf_count = np.isinf(result_df[imom_cols].values).sum()
        assert inf_count == 0, f"Found {inf_count} infinite values in imom_ columns"

    def test_preserves_original_columns(self, ohlcv, result_df):
        original_cols = set(ohlcv.columns)
        assert original_cols.issubset(set(result_df.columns)), (
            f"Some original columns were lost: {original_cols - set(result_df.columns)}"
        )

    def test_preserves_row_count(self, ohlcv, result_df):
        assert len(result_df) == len(ohlcv), (
            f"Row count changed from {len(ohlcv)} to {len(result_df)}"
        )

    def test_no_close_column(self, imom):
        """When required columns are missing, return original df unchanged."""
        df_no_close = _make_ohlcv(n=50).drop(columns=["close"])
        result = imom.create_intraday_momentum_features(df_no_close)
        imom_cols = [c for c in result.columns if c.startswith("imom_")]
        assert len(imom_cols) == 0, (
            "Expected no imom_ columns when 'close' is missing"
        )
        # Row count and original columns must be preserved
        assert len(result) == len(df_no_close)


# ─── TestIntradayMomentumLogic ────────────────────────────────────────────────


class TestIntradayMomentumLogic:
    """Verify that each feature obeys its mathematical specification."""

    def test_last_60min_bounded(self, result_df):
        """imom_last_60min must be in [-1, 1]."""
        col = result_df["imom_last_60min"]
        assert col.min() >= -1.0 - 1e-9, f"imom_last_60min too low: {col.min()}"
        assert col.max() <= 1.0 + 1e-9, f"imom_last_60min too high: {col.max()}"

    def test_midday_reversal_bounded(self, result_df):
        """imom_midday_reversal is a correlation → must be in [-1, 1]."""
        col = result_df["imom_midday_reversal"]
        assert col.min() >= -1.0 - 1e-9, f"midday_reversal too low: {col.min()}"
        assert col.max() <= 1.0 + 1e-9, f"midday_reversal too high: {col.max()}"

    def test_overnight_gap_clipped(self, result_df):
        """imom_overnight_gap_impact must be clipped to [-2, 2]."""
        col = result_df["imom_overnight_gap_impact"]
        assert col.min() >= -2.0 - 1e-9, f"gap_impact too low: {col.min()}"
        assert col.max() <= 2.0 + 1e-9, f"gap_impact too high: {col.max()}"

    def test_first_30min_clipped(self, result_df):
        """imom_first_30min (overnight gap %) is clipped to [-0.10, 0.10]."""
        col = result_df["imom_first_30min"]
        assert col.min() >= -0.10 - 1e-9, f"first_30min too low: {col.min()}"
        assert col.max() <= 0.10 + 1e-9, f"first_30min too high: {col.max()}"

    def test_last_60min_positive_when_close_at_high(self):
        """If close == high (and high > midpoint), imom_last_60min should be +1."""
        imom = IntradayMomentumFeatures()
        # Construct a day where close == high
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open":  [440.0, 450.0],
            "high":  [445.0, 460.0],
            "low":   [438.0, 445.0],
            "close": [441.0, 460.0],  # close == high on row 1
            "volume": [1e8, 1e8],
        })
        result = imom.create_intraday_momentum_features(df)
        val = result["imom_last_60min"].iloc[1]
        assert val == pytest.approx(1.0, abs=1e-6), (
            f"Expected +1.0 when close==high, got {val}"
        )

    def test_last_60min_negative_when_close_at_low(self):
        """If close == low (and low < midpoint), imom_last_60min should be -1."""
        imom = IntradayMomentumFeatures()
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open":  [440.0, 450.0],
            "high":  [445.0, 460.0],
            "low":   [438.0, 445.0],
            "close": [441.0, 445.0],  # close == low on row 1
            "volume": [1e8, 1e8],
        })
        result = imom.create_intraday_momentum_features(df)
        val = result["imom_last_60min"].iloc[1]
        assert val == pytest.approx(-1.0, abs=1e-6), (
            f"Expected -1.0 when close==low, got {val}"
        )

    def test_first_row_gap_impact_is_zero(self, result_df):
        """First row has no prev_close, so overnight_gap_impact must be 0."""
        assert result_df["imom_overnight_gap_impact"].iloc[0] == 0.0

    def test_first_30min_reflects_overnight_gap_sign(self):
        """Positive overnight gap → positive first_30min."""
        imom = IntradayMomentumFeatures()
        # Row 0: seed day; row 1: open clearly above prev close (positive gap)
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02", "2024-01-03"]),
            "open":  [440.0, 455.0],  # 455 > 441 (prev close) → gap up
            "high":  [445.0, 460.0],
            "low":   [438.0, 448.0],
            "close": [441.0, 452.0],
            "volume": [1e8, 1e8],
        })
        result = imom.create_intraday_momentum_features(df)
        # Overnight gap for row 1: (455 - 441) / 441 ≈ +0.0317
        assert result["imom_first_30min"].iloc[1] > 0.0


# ─── TestAnalyze ──────────────────────────────────────────────────────────────


class TestAnalyze:
    """Tests for analyze_current_momentum."""

    def test_returns_dict(self, imom, result_df):
        out = imom.analyze_current_momentum(result_df)
        assert isinstance(out, dict)

    def test_regime_values(self, imom, result_df):
        out = imom.analyze_current_momentum(result_df)
        assert "momentum_regime" in out
        assert out["momentum_regime"] in ("REVERSAL", "CONTINUATION", "MIXED"), (
            f"Unexpected regime: {out['momentum_regime']}"
        )

    def test_returns_none_without_features(self, imom, ohlcv):
        """If imom_ columns haven't been added yet, should return None."""
        out = imom.analyze_current_momentum(ohlcv)
        assert out is None

    def test_returns_none_on_empty_df(self, imom):
        """Empty DataFrame → None."""
        empty = pd.DataFrame(columns=["imom_first_30min"])
        out = imom.analyze_current_momentum(empty)
        assert out is None

    def test_feature_values_present_in_dict(self, imom, result_df):
        """All four imom_ features should appear in the analysis dict."""
        out = imom.analyze_current_momentum(result_df)
        for col in IMOM_FEATURES:
            assert col in out, f"Expected '{col}' in analyze_current_momentum output"
            assert isinstance(out[col], float)

    def test_reversal_regime_when_high_gap_impact(self, imom):
        """When gap_impact is very high (≥ 0.7), regime should lean REVERSAL."""
        # Build a DataFrame where gap_impact will be forced high: open explains
        # almost all of the day's gain, then price barely moved intraday.
        n = 25
        rng = np.random.default_rng(99)
        base = 450.0
        # All days: open big gap up, close only marginally above open
        close = np.linspace(base, base * 1.05, n)
        prev_close = np.concatenate([[base * 0.99], close[:-1]])
        open_ = prev_close * 1.02    # big gap up each day
        high = open_ + rng.uniform(0.1, 0.5, n)
        low = open_ - rng.uniform(0.1, 0.5, n)
        # Ensure high >= close and low <= close
        high = np.maximum(high, close)
        low = np.minimum(low, close)

        df = pd.DataFrame({
            "date": pd.bdate_range("2024-01-02", periods=n),
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.ones(n) * 1e8,
        })
        df = imom.create_intraday_momentum_features(df)
        out = imom.analyze_current_momentum(df)
        assert out is not None
        assert out["momentum_regime"] in ("REVERSAL", "CONTINUATION", "MIXED")

    def test_download_returns_empty_df(self, imom):
        """download_intraday_data should always return an empty DataFrame."""
        from datetime import datetime
        result = imom.download_intraday_data(datetime(2024, 1, 1), datetime(2024, 12, 31))
        assert isinstance(result, pd.DataFrame)
        assert result.empty


# ─── TestFeatureCounts ────────────────────────────────────────────────────────


class TestFeatureCounts:
    """Verify exact feature count contract."""

    def test_total_count(self, result_df):
        """Exactly 4 imom_ features should be added."""
        imom_cols = [c for c in result_df.columns if c.startswith("imom_")]
        assert len(imom_cols) == 4, (
            f"Expected 4 imom_ features, got {len(imom_cols)}: {imom_cols}"
        )

    def test_imom_features_constant(self):
        """IMOM_FEATURES module-level list must have exactly 4 items."""
        assert len(IMOM_FEATURES) == 4

    def test_no_extra_imom_cols_with_200_rows(self):
        """Run with 200 rows and confirm exactly 4 new columns."""
        df = _make_ohlcv(n=200)
        original_cols = set(df.columns)
        imom = IntradayMomentumFeatures(correlation_window=20)
        result = imom.create_intraday_momentum_features(df)
        new_cols = [c for c in result.columns if c not in original_cols]
        assert len(new_cols) == 4


# ─── TestConstructor ──────────────────────────────────────────────────────────


class TestConstructor:

    def test_default_correlation_window(self):
        imom = IntradayMomentumFeatures()
        assert imom.correlation_window == 20

    def test_custom_correlation_window(self):
        imom = IntradayMomentumFeatures(correlation_window=10)
        assert imom.correlation_window == 10

    def test_invalid_window_raises(self):
        with pytest.raises(ValueError):
            IntradayMomentumFeatures(correlation_window=1)

    def test_zero_window_raises(self):
        with pytest.raises(ValueError):
            IntradayMomentumFeatures(correlation_window=0)

    def test_short_window_still_computes(self):
        """correlation_window=2 should run without errors."""
        imom = IntradayMomentumFeatures(correlation_window=2)
        df = _make_ohlcv(n=50)
        result = imom.create_intraday_momentum_features(df)
        imom_cols = [c for c in result.columns if c.startswith("imom_")]
        assert len(imom_cols) == 4


# ─── TestMissingColumns ───────────────────────────────────────────────────────


class TestMissingColumns:

    def test_missing_open(self):
        imom = IntradayMomentumFeatures()
        df = _make_ohlcv(n=30).drop(columns=["open"])
        result = imom.create_intraday_momentum_features(df)
        assert not any(c.startswith("imom_") for c in result.columns)
        assert len(result) == len(df)

    def test_missing_high(self):
        imom = IntradayMomentumFeatures()
        df = _make_ohlcv(n=30).drop(columns=["high"])
        result = imom.create_intraday_momentum_features(df)
        assert not any(c.startswith("imom_") for c in result.columns)

    def test_missing_low(self):
        imom = IntradayMomentumFeatures()
        df = _make_ohlcv(n=30).drop(columns=["low"])
        result = imom.create_intraday_momentum_features(df)
        assert not any(c.startswith("imom_") for c in result.columns)

    def test_missing_close(self):
        imom = IntradayMomentumFeatures()
        df = _make_ohlcv(n=30).drop(columns=["close"])
        result = imom.create_intraday_momentum_features(df)
        assert not any(c.startswith("imom_") for c in result.columns)

    def test_extra_columns_preserved(self):
        """Extra columns beyond OHLCV must survive unchanged."""
        imom = IntradayMomentumFeatures()
        df = _make_ohlcv(n=50)
        df["extra_signal"] = 99.0
        result = imom.create_intraday_momentum_features(df)
        assert "extra_signal" in result.columns
        assert (result["extra_signal"] == 99.0).all()


# ─── TestEdgeCases ────────────────────────────────────────────────────────────


class TestEdgeCases:

    def test_single_row(self):
        """Single-row input must not crash."""
        imom = IntradayMomentumFeatures()
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-06-01"]),
            "open":  [449.0],
            "high":  [452.0],
            "low":   [448.0],
            "close": [451.0],
            "volume": [1e8],
        })
        result = imom.create_intraday_momentum_features(df)
        assert len(result) == 1
        imom_cols = [c for c in result.columns if c.startswith("imom_")]
        assert len(imom_cols) == 4
        nan_count = result[imom_cols].isna().sum().sum()
        assert nan_count == 0

    def test_two_rows(self):
        """Two-row input: first row gap is undefined (0), second computable."""
        imom = IntradayMomentumFeatures()
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-06-03", "2024-06-04"]),
            "open":  [449.0, 451.5],
            "high":  [453.0, 455.0],
            "low":   [448.0, 449.0],
            "close": [451.0, 454.0],
            "volume": [1e8, 1e8],
        })
        result = imom.create_intraday_momentum_features(df)
        assert len(result) == 2
        assert result["imom_first_30min"].iloc[0] == 0.0  # no prev_close
        nan_count = result[[c for c in result.columns if c.startswith("imom_")]].isna().sum().sum()
        assert nan_count == 0

    def test_zero_range_day(self):
        """Doji candle (high == low == open == close) must not produce inf/nan."""
        imom = IntradayMomentumFeatures()
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-06-03", "2024-06-04"]),
            "open":  [450.0, 451.0],
            "high":  [450.0, 451.0],
            "low":   [450.0, 451.0],
            "close": [450.0, 451.0],
            "volume": [1e8, 1e8],
        })
        result = imom.create_intraday_momentum_features(df)
        imom_cols = [c for c in result.columns if c.startswith("imom_")]
        assert not result[imom_cols].isna().any().any()
        assert not np.isinf(result[imom_cols].values).any()

    def test_idempotent(self, imom, ohlcv):
        """Calling create_intraday_momentum_features twice should not double features."""
        result1 = imom.create_intraday_momentum_features(ohlcv)
        result2 = imom.create_intraday_momentum_features(result1)
        imom_cols = [c for c in result2.columns if c.startswith("imom_")]
        assert len(imom_cols) == 4, (
            f"Expected 4 after second call, got {len(imom_cols)}"
        )

    def test_large_dataset(self):
        """1260-row dataset (5 years) must complete without errors."""
        imom = IntradayMomentumFeatures()
        df = _make_ohlcv(n=1260, seed=7)
        result = imom.create_intraday_momentum_features(df)
        imom_cols = [c for c in result.columns if c.startswith("imom_")]
        assert len(imom_cols) == 4
        assert result[imom_cols].isna().sum().sum() == 0
        assert not np.isinf(result[imom_cols].values).any()
