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


# ---------------------------------------------------------------------------
# Pipeline Integration tests
# ---------------------------------------------------------------------------

def test_calendar_features_config_flag():
    """AntiOverfitConfig should have use_calendar_features flag."""
    from src.experiment_config import AntiOverfitConfig
    config = AntiOverfitConfig()
    assert hasattr(config, "use_calendar_features")
    assert config.use_calendar_features is True


def test_calendar_features_config_disable():
    """use_calendar_features should be disableable."""
    from src.experiment_config import AntiOverfitConfig
    config = AntiOverfitConfig(use_calendar_features=False)
    assert config.use_calendar_features is False


def test_integrate_anti_overfit_accepts_calendar_param():
    """integrate_anti_overfit should accept use_calendar_features parameter."""
    import inspect
    from src.phase_13_validation.anti_overfit_integration import integrate_anti_overfit
    sig = inspect.signature(integrate_anti_overfit)
    assert "use_calendar_features" in sig.parameters


def test_calendar_features_in_feature_groups():
    """Calendar features should map to the 'calendar' group in GroupAwareProcessor."""
    from src.phase_10_feature_processing.group_aware_processor import (
        FEATURE_GROUPS,
        assign_feature_groups,
    )

    # The calendar group should include our prefixes
    assert "calendar" in FEATURE_GROUPS
    cal_prefixes = FEATURE_GROUPS["calendar"]
    assert "fomc_" in cal_prefixes
    assert "opex_" in cal_prefixes
    assert "cal_" in cal_prefixes

    # Test that actual calendar feature names map to the group
    test_features = [
        "fomc_is_meeting_day", "fomc_days_until_next", "fomc_cycle_position",
        "opex_is_monthly", "opex_is_quad_witching", "opex_days_until_next",
        "cal_day_of_week", "cal_is_monday", "cal_is_friday",
        "econ_is_nfp_day", "econ_is_cpi_day",
        "some_other_feature",  # should go to "other"
    ]
    groups = assign_feature_groups(test_features)
    assert "calendar" in groups
    assert len(groups["calendar"]) >= 10  # At least 10 of 11 should map to calendar
    assert 11 in groups.get("other", [])  # "some_other_feature" index


def test_calendar_generator_feature_count(daily_df):
    """CalendarFeatureGenerator should produce exactly 29 features."""
    gen = CalendarFeatureGenerator()
    result = gen.create_all_features(daily_df)

    initial_cols = len(daily_df.columns)
    new_cols = len(result.columns) - initial_cols

    # 11 calendar basics + 7 FOMC + 5 opex + 6 economic events = 29
    assert new_cols == 29, f"Expected 29 calendar features, got {new_cols}"


def test_calendar_feature_names_match():
    """get_feature_names() should match actual columns added."""
    gen = CalendarFeatureGenerator()
    expected_names = gen.get_feature_names()

    dates = pd.bdate_range(start="2024-01-02", end="2024-06-30", freq="B")
    np.random.seed(42)
    n = len(dates)
    df = pd.DataFrame({
        "close": 450.0 + np.cumsum(np.random.normal(0, 0.5, n)),
    }, index=dates)
    df.index.name = "date"

    result = gen.create_all_features(df)
    actual_new = [c for c in result.columns if c not in df.columns]

    for name in expected_names:
        assert name in actual_new, f"Expected feature '{name}' not found in output"
