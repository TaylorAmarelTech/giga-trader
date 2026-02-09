"""
Test calendar feature generators: FOMC, Options Expiration, Economic Events,
and the unified CalendarFeatureGenerator.
"""

import sys
from pathlib import Path
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase_09_features_calendar.calendar_features import (
    FOMCFeatures,
    OptionsExpirationFeatures,
    EconomicEventFeatures,
    CalendarFeatureGenerator,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def daily_df():
    """Create a daily DataFrame spanning 2024 for calendar feature testing."""
    dates = pd.bdate_range(start="2024-01-02", end="2024-12-31", freq="B")
    np.random.seed(77)
    n = len(dates)
    close = 450.0 + np.cumsum(np.random.normal(0, 1, n))
    df = pd.DataFrame({
        "open": close + np.random.normal(0, 0.5, n),
        "high": close + abs(np.random.normal(0, 1, n)),
        "low": close - abs(np.random.normal(0, 1, n)),
        "close": close,
        "volume": np.random.randint(50_000_000, 150_000_000, n).astype(float),
    }, index=dates)
    df.index.name = "date"
    return df


# ---------------------------------------------------------------------------
# FOMCFeatures tests
# ---------------------------------------------------------------------------

def test_fomc_features_initialization():
    """FOMCFeatures should initialize and load FOMC dates."""
    fomc = FOMCFeatures()
    assert len(fomc._fomc_dates) > 50  # Multiple years of meetings


def test_fomc_features_creates_columns(daily_df):
    """FOMCFeatures.create_features should add expected columns."""
    fomc = FOMCFeatures()
    result = fomc.create_features(daily_df)

    expected_cols = [
        "fomc_is_meeting_day",
        "fomc_is_day_before",
        "fomc_is_day_after",
        "fomc_days_until_next",
        "fomc_days_since_last",
        "fomc_is_meeting_week",
        "fomc_cycle_position",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_fomc_meeting_day_flags(daily_df):
    """FOMC meeting days should be flagged correctly."""
    fomc = FOMCFeatures()
    result = fomc.create_features(daily_df)

    # 2024 FOMC dates include 2024-01-31
    meeting_days = result[result["fomc_is_meeting_day"] == 1]
    assert len(meeting_days) >= 1, "Should have at least 1 FOMC meeting day in 2024"

    # fomc_days_until_next should always be >= 0
    assert (result["fomc_days_until_next"] >= 0).all()

    # fomc_cycle_position should be between 0 and 1
    assert (result["fomc_cycle_position"] >= 0).all()
    assert (result["fomc_cycle_position"] <= 1).all()


# ---------------------------------------------------------------------------
# OptionsExpirationFeatures tests
# ---------------------------------------------------------------------------

def test_opex_features_initialization():
    """OptionsExpirationFeatures should initialize."""
    opex = OptionsExpirationFeatures()
    assert isinstance(opex, OptionsExpirationFeatures)


def test_opex_third_friday_detection():
    """_get_third_friday should return the correct date."""
    opex = OptionsExpirationFeatures()
    # January 2024: 3rd Friday is Jan 19
    third_friday = opex._get_third_friday(2024, 1)
    assert third_friday == date(2024, 1, 19)
    assert third_friday.weekday() == 4  # Friday

    # March 2024: 3rd Friday is Mar 15
    third_friday_mar = opex._get_third_friday(2024, 3)
    assert third_friday_mar.weekday() == 4


def test_opex_creates_columns(daily_df):
    """OptionsExpirationFeatures should add expected columns."""
    opex = OptionsExpirationFeatures()
    result = opex.create_features(daily_df)

    expected_cols = [
        "opex_is_monthly",
        "opex_is_quad_witching",
        "opex_days_until_next",
        "opex_is_expiration_week",
        "opex_is_day_before",
    ]
    for col in expected_cols:
        assert col in result.columns, f"Missing column: {col}"


def test_opex_quad_witching_months(daily_df):
    """Quad witching should only flag March, June, September, December."""
    opex = OptionsExpirationFeatures()
    result = opex.create_features(daily_df)

    quad_days = result[result["opex_is_quad_witching"] == 1]
    if len(quad_days) > 0:
        # All quad witching days should be in months 3, 6, 9, 12
        for idx in quad_days.index:
            assert idx.month in (3, 6, 9, 12), (
                f"Quad witching flagged in month {idx.month}"
            )


# ---------------------------------------------------------------------------
# EconomicEventFeatures tests
# ---------------------------------------------------------------------------

def test_economic_event_features_initialization():
    """EconomicEventFeatures should initialize."""
    econ = EconomicEventFeatures()
    assert isinstance(econ, EconomicEventFeatures)


def test_economic_event_creates_features(daily_df):
    """EconomicEventFeatures should add columns to the DataFrame."""
    econ = EconomicEventFeatures()
    result = econ.create_features(daily_df)

    # Should have more columns than input
    assert len(result.columns) > len(daily_df.columns)


# ---------------------------------------------------------------------------
# CalendarFeatureGenerator tests (unified)
# ---------------------------------------------------------------------------

def test_calendar_generator_initialization():
    """CalendarFeatureGenerator should initialize with sub-generators."""
    gen = CalendarFeatureGenerator()
    assert gen.fomc is not None
    assert gen.opex is not None
    assert gen.economic is not None


def test_calendar_generator_creates_all_features(daily_df):
    """CalendarFeatureGenerator.create_all_features should combine all features."""
    gen = CalendarFeatureGenerator()
    result = gen.create_all_features(daily_df)

    initial_cols = len(daily_df.columns)
    new_cols = len(result.columns) - initial_cols

    # Should add approximately 25-35 calendar features
    # 7 FOMC + 5 opex + economic + calendar basics
    assert new_cols >= 15, f"Expected at least 15 new columns, got {new_cols}"


def test_calendar_generator_selective_features(daily_df):
    """CalendarFeatureGenerator should respect include flags."""
    gen = CalendarFeatureGenerator(
        include_fomc=True,
        include_opex=False,
        include_economic=False,
        include_calendar_basics=False,
    )
    result = gen.create_all_features(daily_df)

    # Should have FOMC columns but not opex
    assert "fomc_is_meeting_day" in result.columns
    assert "opex_is_monthly" not in result.columns


def test_calendar_features_with_date_column():
    """Features should work with a 'date' column instead of DatetimeIndex."""
    dates = pd.bdate_range(start="2024-06-01", end="2024-09-30", freq="B")
    np.random.seed(99)
    n = len(dates)
    df = pd.DataFrame({
        "date": dates,
        "close": 450.0 + np.cumsum(np.random.normal(0, 0.5, n)),
        "volume": np.random.randint(50_000_000, 150_000_000, n).astype(float),
    })

    fomc = FOMCFeatures()
    result = fomc.create_features(df)
    assert "fomc_is_meeting_day" in result.columns


def test_calendar_features_preserve_original_data(daily_df):
    """Original columns should remain unchanged after adding features."""
    gen = CalendarFeatureGenerator()
    original_close = daily_df["close"].copy()
    result = gen.create_all_features(daily_df)

    pd.testing.assert_series_equal(result["close"], original_close, check_names=True)
