"""
Tests for OHLCValidator.

Validates that the OHLC data validator correctly identifies and handles
corrupt, malformed, and suspicious market data bars.
"""

import numpy as np
import pandas as pd
import pytest

from src.phase_02_preprocessing.ohlc_validator import OHLCValidator


# ─── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def validator():
    """Default validator with auto_fix=True and 50% max daily move."""
    return OHLCValidator(max_daily_pct_change=0.50, auto_fix=True)


@pytest.fixture
def strict_validator():
    """Strict validator with auto_fix=False -- drops instead of fixing."""
    return OHLCValidator(max_daily_pct_change=0.50, auto_fix=False)


@pytest.fixture
def clean_df():
    """A well-formed 10-row OHLCV DataFrame that should pass all checks."""
    np.random.seed(42)
    dates = pd.bdate_range("2024-01-02", periods=10)
    close = 450.0 + np.cumsum(np.random.normal(0, 1.5, 10))
    opens = close + np.random.normal(0, 0.3, 10)
    highs = np.maximum(opens, close) + np.abs(np.random.normal(0.5, 0.3, 10))
    lows = np.minimum(opens, close) - np.abs(np.random.normal(0.5, 0.3, 10))
    volume = np.random.randint(1_000_000, 5_000_000, 10)

    df = pd.DataFrame(
        {
            "open": opens,
            "high": highs,
            "low": lows,
            "close": close,
            "volume": volume,
        },
        index=dates,
    )
    return df


# ─── Test: Clean data passes through unchanged ──────────────────────────────


def test_clean_data_unchanged(validator, clean_df):
    """Clean data should pass through with zero violations and identical values."""
    result, stats = validator.validate(clean_df)

    assert stats["rows_input"] == 10
    assert stats["rows_output"] == 10
    assert stats["rows_dropped"] == 0
    assert stats["rows_fixed"] == 0
    assert stats["violations"] == {}

    pd.testing.assert_frame_equal(result, clean_df)


# ─── Test: High < Close gets auto-fixed ─────────────────────────────────────


def test_high_below_close_autofix(validator, clean_df):
    """When High < Close, auto_fix should set High = max(O, H, C)."""
    df = clean_df.copy()
    # Corrupt row 3: set high below close
    df.iloc[3, df.columns.get_loc("high")] = df.iloc[3]["close"] - 1.0

    result, stats = validator.validate(df)

    assert stats["rows_fixed"] >= 1
    assert "high_below_oc_max" in stats["violations"]
    assert stats["rows_output"] == 10  # row kept, not dropped

    # The fixed high should be >= max(open, close) for that row
    row = result.iloc[3]
    assert row["high"] >= max(row["open"], row["close"])


# ─── Test: Low > Open gets auto-fixed ───────────────────────────────────────


def test_low_above_open_autofix(validator, clean_df):
    """When Low > min(O, C), auto_fix should set Low = min(O, L, C)."""
    df = clean_df.copy()
    # Corrupt row 5: set low above open
    df.iloc[5, df.columns.get_loc("low")] = df.iloc[5]["open"] + 2.0

    result, stats = validator.validate(df)

    assert stats["rows_fixed"] >= 1
    assert "low_above_oc_min" in stats["violations"]
    assert stats["rows_output"] == 10

    row = result.iloc[5]
    assert row["low"] <= min(row["open"], row["close"])


# ─── Test: High < Low gets swapped ──────────────────────────────────────────


def test_high_below_low_swap(validator):
    """When High < Low (inverted bar), auto_fix should swap them."""
    dates = pd.bdate_range("2024-01-02", periods=3)
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [95.0, 105.0, 105.0],   # row 0: high < low
            "low": [105.0, 95.0, 95.0],     # row 0: low > high
            "close": [100.0, 100.0, 100.0],
            "volume": [1000, 1000, 1000],
        },
        index=dates,
    )

    result, stats = validator.validate(df)

    # After the previous checks fix high/low relative to O/C, the H<L swap
    # handles any remaining inversions.
    assert stats["rows_output"] == 3
    # Every row should now satisfy High >= Low
    assert (result["high"] >= result["low"]).all()


# ─── Test: Negative prices get dropped ──────────────────────────────────────


def test_negative_prices_dropped(validator, clean_df):
    """Rows with any negative price should be dropped."""
    df = clean_df.copy()
    df.iloc[2, df.columns.get_loc("open")] = -10.0
    df.iloc[7, df.columns.get_loc("close")] = -5.0

    result, stats = validator.validate(df)

    assert stats["rows_dropped"] == 2
    assert stats["violations"]["negative_prices"] == 2
    assert stats["rows_output"] == 8


# ─── Test: Zero volume gets dropped ─────────────────────────────────────────


def test_zero_volume_dropped(validator, clean_df):
    """Rows with volume <= 0 should be dropped."""
    df = clean_df.copy()
    df.iloc[0, df.columns.get_loc("volume")] = 0
    df.iloc[4, df.columns.get_loc("volume")] = -100

    result, stats = validator.validate(df)

    assert stats["violations"]["zero_or_negative_volume"] == 2
    assert stats["rows_dropped"] == 2
    assert stats["rows_output"] == 8


# ─── Test: Extreme daily move dropped ───────────────────────────────────────


def test_extreme_daily_move_dropped(validator, clean_df):
    """Rows where close changes > 50% from previous close are dropped."""
    df = clean_df.copy()
    prev_close = df.iloc[4]["close"]
    # Make row 5 have a +80% jump
    extreme_close = prev_close * 1.80
    df.iloc[5, df.columns.get_loc("close")] = extreme_close
    df.iloc[5, df.columns.get_loc("high")] = extreme_close + 1.0

    result, stats = validator.validate(df)

    assert "extreme_daily_move" in stats["violations"]
    assert stats["rows_dropped"] >= 1


# ─── Test: Duplicate index deduplicated ─────────────────────────────────────


def test_duplicate_index_deduplicated(validator, clean_df):
    """Duplicate index entries should be removed (keep first)."""
    df = clean_df.copy()
    # Duplicate the index of row 2 onto row 3
    idx = df.index.tolist()
    idx[3] = idx[2]
    df.index = pd.DatetimeIndex(idx)

    result, stats = validator.validate(df)

    assert "duplicate_index" in stats["violations"]
    assert stats["violations"]["duplicate_index"] == 1
    assert stats["rows_output"] == 9
    assert not result.index.duplicated().any()


# ─── Test: Empty DataFrame ──────────────────────────────────────────────────


def test_empty_dataframe(validator):
    """Empty DataFrame should return empty with zero stats."""
    df = pd.DataFrame()
    result, stats = validator.validate(df)

    assert result.empty
    assert stats["rows_input"] == 0
    assert stats["rows_output"] == 0
    assert stats["rows_dropped"] == 0
    assert stats["rows_fixed"] == 0
    assert stats["violations"] == {}


# ─── Test: Missing required columns raises ValueError ────────────────────────


def test_missing_columns_raises(validator):
    """If required columns are absent, validate should raise ValueError."""
    df = pd.DataFrame({"open": [100], "high": [101], "close": [100.5]})

    with pytest.raises(ValueError, match="Missing required OHLCV columns"):
        validator.validate(df)


# ─── Test: NaN prices get dropped ────────────────────────────────────────────


def test_nan_prices_dropped(validator, clean_df):
    """Rows with NaN in price columns should be dropped."""
    df = clean_df.copy()
    df.iloc[1, df.columns.get_loc("high")] = np.nan
    df.iloc[6, df.columns.get_loc("low")] = np.nan

    result, stats = validator.validate(df)

    assert stats["violations"]["nan_or_inf_prices"] == 2
    assert stats["rows_dropped"] == 2
    assert stats["rows_output"] == 8


# ─── Test: Inf prices get dropped ────────────────────────────────────────────


def test_inf_prices_dropped(validator, clean_df):
    """Rows with inf in price columns should be dropped."""
    df = clean_df.copy()
    df.iloc[0, df.columns.get_loc("open")] = np.inf

    result, stats = validator.validate(df)

    assert "nan_or_inf_prices" in stats["violations"]
    assert stats["rows_dropped"] >= 1


# ─── Test: Case-insensitive column matching ──────────────────────────────────


def test_case_insensitive_columns(validator):
    """Column names like 'Open', 'HIGH', 'Volume' should be accepted."""
    dates = pd.bdate_range("2024-01-02", periods=3)
    df = pd.DataFrame(
        {
            "Open": [100.0, 101.0, 102.0],
            "HIGH": [103.0, 104.0, 105.0],
            "Low": [98.0, 99.0, 100.0],
            "CLOSE": [101.0, 102.0, 103.0],
            "Volume": [1_000_000, 2_000_000, 3_000_000],
        },
        index=dates,
    )

    result, stats = validator.validate(df)

    assert stats["rows_input"] == 3
    assert stats["rows_output"] == 3
    # Columns should be normalised to lowercase
    assert "open" in result.columns
    assert "high" in result.columns


# ─── Test: auto_fix=False drops violating rows ──────────────────────────────


def test_autofix_false_drops_rows(strict_validator, clean_df):
    """With auto_fix=False, rows with H < max(O,C) should be dropped, not fixed."""
    df = clean_df.copy()
    df.iloc[3, df.columns.get_loc("high")] = df.iloc[3]["close"] - 1.0

    result, stats = strict_validator.validate(df)

    assert stats["rows_fixed"] == 0
    assert "high_below_oc_max" in stats["violations"]
    assert stats["rows_dropped"] >= 1
    assert stats["rows_output"] < 10


# ─── Test: Stats dict has correct counts ─────────────────────────────────────


def test_stats_dict_correct(validator, clean_df):
    """Stats dict should correctly tally input, output, dropped, fixed."""
    df = clean_df.copy()
    # Introduce 2 negative-price rows and 1 high<close row
    df.iloc[0, df.columns.get_loc("open")] = -1.0
    df.iloc[1, df.columns.get_loc("close")] = -2.0
    df.iloc[4, df.columns.get_loc("high")] = df.iloc[4]["close"] - 5.0

    result, stats = validator.validate(df)

    assert stats["rows_input"] == 10
    # 2 rows dropped for negative prices, row 4 fixed
    assert stats["rows_output"] == 8
    assert stats["rows_dropped"] == 2
    assert stats["rows_fixed"] >= 1
    assert isinstance(stats["violations"], dict)
    assert "negative_prices" in stats["violations"]


# ─── Test: Original DataFrame not modified ───────────────────────────────────


def test_original_not_modified(validator, clean_df):
    """The validator must not modify the original DataFrame (copy semantics)."""
    df = clean_df.copy()

    # Corrupt a row so the validator has something to fix
    df.iloc[2, df.columns.get_loc("high")] = df.iloc[2]["close"] - 1.0

    # Capture values AFTER corruption but BEFORE validate
    corrupted_values = df.values.copy()

    _result, _stats = validator.validate(df)

    # The original df should still have the corrupted value (not auto-fixed)
    np.testing.assert_array_equal(df.values, corrupted_values)


# ─── Test: validate_bar single bar ──────────────────────────────────────────


def test_validate_bar_valid(validator):
    """A valid bar should return (True, [])."""
    bar = pd.Series({"open": 100.0, "high": 105.0, "low": 98.0, "close": 103.0, "volume": 50000})
    is_valid, issues = validator.validate_bar(bar)
    assert is_valid is True
    assert issues == []


def test_validate_bar_invalid(validator):
    """An invalid bar should return (False, [...violations...])."""
    bar = pd.Series({"open": 100.0, "high": 95.0, "low": 98.0, "close": 103.0, "volume": 50000})
    is_valid, issues = validator.validate_bar(bar)
    assert is_valid is False
    assert "high_below_oc_max" in issues
