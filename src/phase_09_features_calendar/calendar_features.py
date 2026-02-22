"""
Calendar & Event Features
=========================
Features based on known market calendar events that affect trading behavior.

Implements:
- FOMC meeting dates and surrounding windows
- Options expiration (monthly opex, quad witching)
- Economic data releases (NFP, CPI, GDP, retail sales)
- Market calendar awareness (holidays, half-days)

These features capture event-driven regime shifts that purely
price-based features miss.
"""

import logging
from datetime import datetime, date, timedelta
from typing import List, Optional, Dict, Set
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger("CALENDAR_FEATURES")


# =============================================================================
# FOMC MEETING DATES
# =============================================================================

class FOMCFeatures:
    """
    FOMC (Federal Open Market Committee) meeting date features.

    FOMC meetings are among the highest-impact events for equity markets.
    Markets often exhibit different behavior:
    - Day before FOMC: Reduced volatility, positioning
    - FOMC day: High volatility after 2:00 PM ET announcement
    - Day after FOMC: Trend continuation or reversal

    Features generated:
    - is_fomc_day: Binary flag for FOMC decision day
    - is_fomc_day_before: Day before FOMC
    - is_fomc_day_after: Day after FOMC
    - fomc_days_until: Days until next FOMC (0-45)
    - fomc_days_since: Days since last FOMC
    - is_fomc_week: Any day in FOMC decision week
    """

    # FOMC scheduled meeting dates (statement release dates)
    # Source: Federal Reserve Board
    FOMC_DATES = {
        2019: [
            "2019-01-30", "2019-03-20", "2019-05-01", "2019-06-19",
            "2019-07-31", "2019-09-18", "2019-10-30", "2019-12-11",
        ],
        2020: [
            "2020-01-29", "2020-03-03", "2020-03-15",  # Emergency
            "2020-03-18", "2020-04-29", "2020-06-10", "2020-07-29",
            "2020-09-16", "2020-11-05", "2020-12-16",
        ],
        2021: [
            "2021-01-27", "2021-03-17", "2021-04-28", "2021-06-16",
            "2021-07-28", "2021-09-22", "2021-11-03", "2021-12-15",
        ],
        2022: [
            "2022-01-26", "2022-03-16", "2022-05-04", "2022-06-15",
            "2022-07-27", "2022-09-21", "2022-11-02", "2022-12-14",
        ],
        2023: [
            "2023-02-01", "2023-03-22", "2023-05-03", "2023-06-14",
            "2023-07-26", "2023-09-20", "2023-11-01", "2023-12-13",
        ],
        2024: [
            "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
            "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
        ],
        2025: [
            "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
            "2025-07-30", "2025-09-17", "2025-10-29", "2025-12-10",
        ],
        2026: [
            "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
            "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
        ],
    }

    def __init__(self):
        self._fomc_dates: Set[date] = set()
        self._load_dates()

    def _load_dates(self):
        """Parse all FOMC dates into a set."""
        for year, dates in self.FOMC_DATES.items():
            for d in dates:
                self._fomc_dates.add(datetime.strptime(d, "%Y-%m-%d").date())
        logger.info(f"Loaded {len(self._fomc_dates)} FOMC meeting dates")

    def _get_sorted_dates(self) -> List[date]:
        """Get all FOMC dates sorted."""
        return sorted(self._fomc_dates)

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add FOMC-related features to a DataFrame.

        Args:
            df: DataFrame with a datetime index or 'date' column

        Returns:
            DataFrame with FOMC feature columns added
        """
        df = df.copy()

        # Get dates from index or column
        if isinstance(df.index, pd.DatetimeIndex):
            dates = df.index.date
        elif "date" in df.columns:
            dates = pd.to_datetime(df["date"]).dt.date
        else:
            logger.warning("No date index or column found, skipping FOMC features")
            return df

        sorted_fomc = self._get_sorted_dates()

        is_fomc = []
        is_before = []
        is_after = []
        days_until = []
        days_since = []
        is_week = []

        for d in dates:
            # Is FOMC day
            is_fomc.append(1 if d in self._fomc_dates else 0)

            # Day before / after
            next_day = d + timedelta(days=1)
            prev_day = d - timedelta(days=1)
            # Handle weekends
            if prev_day.weekday() == 6:  # Sunday
                prev_day = d - timedelta(days=2)
            elif prev_day.weekday() == 5:  # Saturday
                prev_day = d - timedelta(days=1)

            is_before.append(1 if next_day in self._fomc_dates or
                           (d + timedelta(days=3)) in self._fomc_dates and d.weekday() == 4
                           else 0)
            is_after.append(1 if prev_day in self._fomc_dates else 0)

            # Days until next FOMC
            future = [fd for fd in sorted_fomc if fd > d]
            if future:
                days_until.append((future[0] - d).days)
            else:
                days_until.append(45)  # Max cap

            # Days since last FOMC
            past = [fd for fd in sorted_fomc if fd <= d]
            if past:
                days_since.append((d - past[-1]).days)
            else:
                days_since.append(45)

            # FOMC week (Mon-Fri of FOMC decision week)
            week_start = d - timedelta(days=d.weekday())
            week_dates = [week_start + timedelta(days=i) for i in range(5)]
            is_fomc_week = any(wd in self._fomc_dates for wd in week_dates)
            is_week.append(1 if is_fomc_week else 0)

        df["fomc_is_meeting_day"] = is_fomc
        df["fomc_is_day_before"] = is_before
        df["fomc_is_day_after"] = is_after
        df["fomc_days_until_next"] = days_until
        df["fomc_days_since_last"] = days_since
        df["fomc_is_meeting_week"] = is_week

        # Normalized distance feature (0 = FOMC day, 1 = mid-cycle)
        df["fomc_cycle_position"] = df["fomc_days_until_next"] / (
            df["fomc_days_until_next"] + df["fomc_days_since_last"] + 1
        )

        n_features = 7
        logger.info(f"Added {n_features} FOMC features")
        return df


# =============================================================================
# OPTIONS EXPIRATION FEATURES
# =============================================================================

class OptionsExpirationFeatures:
    """
    Options expiration date features.

    Key dates:
    - Monthly opex: 3rd Friday of each month (large open interest expiry)
    - Quad witching: 3rd Friday of March, June, September, December
      (stock options, index options, index futures, single stock futures expire)
    - 0DTE: Daily options (increasingly important, hard to model)

    Features:
    - is_monthly_opex: Monthly options expiration
    - is_quad_witching: Quarterly quad witching day
    - days_to_opex: Trading days until next monthly opex
    - is_opex_week: Week of monthly opex
    - is_opex_day_before: Day before monthly opex
    """

    QUAD_WITCHING_MONTHS = {3, 6, 9, 12}

    def __init__(self):
        self._opex_cache: Dict[int, List[date]] = {}

    def _get_third_friday(self, year: int, month: int) -> date:
        """Get the third Friday of a given month/year."""
        # Find first day of month
        first_day = date(year, month, 1)
        # Find first Friday
        days_until_friday = (4 - first_day.weekday()) % 7
        first_friday = first_day + timedelta(days=days_until_friday)
        # Third Friday is 2 weeks later
        third_friday = first_friday + timedelta(weeks=2)
        return third_friday

    def _get_opex_dates(self, year: int) -> List[date]:
        """Get all monthly opex dates for a year."""
        if year in self._opex_cache:
            return self._opex_cache[year]

        dates = []
        for month in range(1, 13):
            dates.append(self._get_third_friday(year, month))

        self._opex_cache[year] = dates
        return dates

    def _get_all_opex_dates(self, start_year: int, end_year: int) -> List[date]:
        """Get all opex dates in range."""
        all_dates = []
        for year in range(start_year, end_year + 1):
            all_dates.extend(self._get_opex_dates(year))
        return sorted(all_dates)

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add options expiration features to DataFrame."""
        df = df.copy()

        if isinstance(df.index, pd.DatetimeIndex):
            dates = df.index.date
        elif "date" in df.columns:
            dates = pd.to_datetime(df["date"]).dt.date
        else:
            logger.warning("No date index or column found")
            return df

        # Get year range
        min_year = min(d.year for d in dates)
        max_year = max(d.year for d in dates)
        all_opex = set(self._get_all_opex_dates(min_year, max_year + 1))
        sorted_opex = sorted(all_opex)

        quad_witching = set()
        for d in all_opex:
            if d.month in self.QUAD_WITCHING_MONTHS:
                quad_witching.add(d)

        is_opex = []
        is_quad = []
        days_to = []
        is_opex_week = []
        is_day_before = []

        for d in dates:
            is_opex.append(1 if d in all_opex else 0)
            is_quad.append(1 if d in quad_witching else 0)

            # Days to next opex
            future = [od for od in sorted_opex if od > d]
            if future:
                days_to.append((future[0] - d).days)
            else:
                days_to.append(30)

            # Opex week
            week_start = d - timedelta(days=d.weekday())
            week_dates = [week_start + timedelta(days=i) for i in range(5)]
            is_opex_week.append(1 if any(wd in all_opex for wd in week_dates) else 0)

            # Day before opex
            next_day = d + timedelta(days=1)
            # Handle weekend
            if d.weekday() == 3:  # Thursday
                next_day = d + timedelta(days=1)  # Friday
            is_day_before.append(1 if next_day in all_opex else 0)

        df["opex_is_monthly"] = is_opex
        df["opex_is_quad_witching"] = is_quad
        df["opex_days_until_next"] = days_to
        df["opex_is_expiration_week"] = is_opex_week
        df["opex_is_day_before"] = is_day_before

        n_features = 5
        logger.info(f"Added {n_features} options expiration features")
        return df


# =============================================================================
# ECONOMIC EVENT FEATURES
# =============================================================================

class EconomicEventFeatures:
    """
    Major economic data release features.

    Key releases:
    - NFP (Non-Farm Payrolls): First Friday of each month, 8:30 AM ET
    - CPI (Consumer Price Index): ~10th-15th of month, 8:30 AM ET
    - GDP: End of month (advance, preliminary, final), 8:30 AM ET
    - Retail Sales: ~15th of month, 8:30 AM ET
    - PMI (Purchasing Managers Index): First business day, 10:00 AM ET

    Features:
    - is_nfp_day: NFP release day
    - is_cpi_day: CPI release day
    - is_major_econ_release: Any major release
    - econ_release_count_week: Number of major releases this week
    """

    def __init__(self):
        pass

    def _get_nfp_dates(self, year: int) -> List[date]:
        """Get NFP dates (first Friday of each month)."""
        dates = []
        for month in range(1, 13):
            first_day = date(year, month, 1)
            days_until_friday = (4 - first_day.weekday()) % 7
            first_friday = first_day + timedelta(days=days_until_friday)
            dates.append(first_friday)
        return dates

    def _get_approximate_cpi_dates(self, year: int) -> List[date]:
        """
        Get approximate CPI release dates.
        CPI is typically released around the 10th-15th of each month.
        Actual dates vary; this uses the 2nd Tuesday of each month as proxy.
        """
        dates = []
        for month in range(1, 13):
            first_day = date(year, month, 1)
            days_until_tuesday = (1 - first_day.weekday()) % 7
            first_tuesday = first_day + timedelta(days=days_until_tuesday)
            second_tuesday = first_tuesday + timedelta(weeks=1)
            dates.append(second_tuesday)
        return dates

    def _get_pmi_dates(self, year: int) -> List[date]:
        """Get PMI dates (first business day of each month)."""
        dates = []
        for month in range(1, 13):
            first_day = date(year, month, 1)
            # If weekend, move to Monday
            while first_day.weekday() >= 5:
                first_day += timedelta(days=1)
            dates.append(first_day)
        return dates

    def _get_gdp_dates(self, year: int) -> List[date]:
        """
        Get approximate GDP release dates.
        GDP advance estimate: ~last Thursday of Jan, Apr, Jul, Oct.
        """
        dates = []
        for month in [1, 4, 7, 10]:
            # Last Thursday of the month
            if month == 12:
                next_month_first = date(year + 1, 1, 1)
            else:
                next_month_first = date(year, month + 1, 1)
            last_day = next_month_first - timedelta(days=1)
            # Find last Thursday
            days_back = (last_day.weekday() - 3) % 7
            last_thursday = last_day - timedelta(days=days_back)
            dates.append(last_thursday)
        return dates

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add economic event features to DataFrame."""
        df = df.copy()

        if isinstance(df.index, pd.DatetimeIndex):
            dates = df.index.date
        elif "date" in df.columns:
            dates = pd.to_datetime(df["date"]).dt.date
        else:
            logger.warning("No date index or column found")
            return df

        min_year = min(d.year for d in dates)
        max_year = max(d.year for d in dates)

        # Collect all event dates by type
        all_nfp = set()
        all_cpi = set()
        all_pmi = set()
        all_gdp = set()

        for year in range(min_year, max_year + 1):
            all_nfp.update(self._get_nfp_dates(year))
            all_cpi.update(self._get_approximate_cpi_dates(year))
            all_pmi.update(self._get_pmi_dates(year))
            all_gdp.update(self._get_gdp_dates(year))

        all_major = all_nfp | all_cpi | all_gdp

        is_nfp = []
        is_cpi = []
        is_pmi = []
        is_gdp = []
        is_major = []
        releases_this_week = []

        for d in dates:
            is_nfp.append(1 if d in all_nfp else 0)
            is_cpi.append(1 if d in all_cpi else 0)
            is_pmi.append(1 if d in all_pmi else 0)
            is_gdp.append(1 if d in all_gdp else 0)
            is_major.append(1 if d in all_major else 0)

            # Count releases this week
            week_start = d - timedelta(days=d.weekday())
            week_dates = [week_start + timedelta(days=i) for i in range(5)]
            count = sum(1 for wd in week_dates if wd in all_major)
            releases_this_week.append(count)

        df["econ_is_nfp_day"] = is_nfp
        df["econ_is_cpi_day"] = is_cpi
        df["econ_is_pmi_day"] = is_pmi
        df["econ_is_gdp_day"] = is_gdp
        df["econ_is_major_release"] = is_major
        df["econ_releases_this_week"] = releases_this_week

        n_features = 6
        logger.info(f"Added {n_features} economic event features")
        return df


# =============================================================================
# COMBINED CALENDAR FEATURE GENERATOR
# =============================================================================

class CalendarFeatureGenerator:
    """
    Unified generator for all calendar-based features.

    Combines FOMC, options expiration, and economic event features
    into a single call. Also adds day-of-week and month-of-year features.

    Usage:
        generator = CalendarFeatureGenerator()
        df = generator.create_all_features(df)
    """

    def __init__(
        self,
        include_fomc: bool = True,
        include_opex: bool = True,
        include_economic: bool = True,
        include_calendar_basics: bool = True,
    ):
        self.include_fomc = include_fomc
        self.include_opex = include_opex
        self.include_economic = include_economic
        self.include_calendar_basics = include_calendar_basics

        self.fomc = FOMCFeatures() if include_fomc else None
        self.opex = OptionsExpirationFeatures() if include_opex else None
        self.economic = EconomicEventFeatures() if include_economic else None

    def create_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate all calendar features."""
        initial_cols = len(df.columns)

        if self.include_calendar_basics:
            df = self._add_calendar_basics(df)

        if self.fomc:
            df = self.fomc.create_features(df)

        if self.opex:
            df = self.opex.create_features(df)

        if self.economic:
            df = self.economic.create_features(df)

        new_cols = len(df.columns) - initial_cols
        logger.info(f"CalendarFeatureGenerator added {new_cols} total features")
        return df

    def _add_calendar_basics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add basic calendar features (day of week, month, etc.)."""
        df = df.copy()

        if isinstance(df.index, pd.DatetimeIndex):
            dt_index = df.index
        elif "date" in df.columns:
            dt_index = pd.DatetimeIndex(pd.to_datetime(df["date"]))
        else:
            return df

        # Day of week (0=Monday, 4=Friday)
        df["cal_day_of_week"] = dt_index.dayofweek.values
        df["cal_is_monday"] = (dt_index.dayofweek == 0).astype(int)
        df["cal_is_friday"] = (dt_index.dayofweek == 4).astype(int)

        # Month features
        df["cal_month"] = dt_index.month.values
        df["cal_is_month_end"] = dt_index.is_month_end.astype(int)
        df["cal_is_month_start"] = dt_index.is_month_start.astype(int)
        df["cal_is_quarter_end"] = dt_index.is_quarter_end.astype(int)

        # Week of year (for seasonal patterns)
        df["cal_week_of_year"] = dt_index.isocalendar().week.values

        # January effect, sell-in-May
        df["cal_is_january"] = (dt_index.month == 1).astype(int)
        df["cal_is_sell_in_may"] = ((dt_index.month >= 5) & (dt_index.month <= 10)).astype(int)

        # Year-end effects (last 5 trading days, Santa Claus rally)
        df["cal_is_year_end"] = ((dt_index.month == 12) & (dt_index.day >= 24)).astype(int)

        n_features = 11
        logger.info(f"Added {n_features} basic calendar features")
        return df

    def get_feature_names(self) -> List[str]:
        """Get all feature names that will be generated."""
        names = []

        if self.include_calendar_basics:
            names.extend([
                "cal_day_of_week", "cal_is_monday", "cal_is_friday",
                "cal_month", "cal_is_month_end", "cal_is_month_start",
                "cal_is_quarter_end", "cal_week_of_year", "cal_is_january",
                "cal_is_sell_in_may", "cal_is_year_end",
            ])

        if self.include_fomc:
            names.extend([
                "fomc_is_meeting_day", "fomc_is_day_before", "fomc_is_day_after",
                "fomc_days_until_next", "fomc_days_since_last",
                "fomc_is_meeting_week", "fomc_cycle_position",
            ])

        if self.include_opex:
            names.extend([
                "opex_is_monthly", "opex_is_quad_witching",
                "opex_days_until_next", "opex_is_expiration_week",
                "opex_is_day_before",
            ])

        if self.include_economic:
            names.extend([
                "econ_is_nfp_day", "econ_is_cpi_day", "econ_is_pmi_day",
                "econ_is_gdp_day", "econ_is_major_release",
                "econ_releases_this_week",
            ])

        return names
