"""
Seasonality Features -- well-documented equity calendar anomalies.

Academic research has identified several persistent calendar effects
in equity markets. While most have weakened over time, they remain
useful as conditioning variables for ML models.

Features (8, prefix seas_):
  seas_turn_of_month    -- 1.0 if last 2 or first 3 trading days of month
  seas_january_effect   -- 1.0 if January
  seas_pre_fomc_drift   -- 1.0 if day before or day of FOMC (known dates)
  seas_holiday_drift    -- 1.0 if day before major US holiday
  seas_santa_claus      -- 1.0 if Dec 25 to Jan 2 window
  seas_quad_witching    -- 1.0 if third Friday of Mar/Jun/Sep/Dec
  seas_sell_in_may      -- 1.0 if May through October
  seas_day_of_week      -- Ordinal day of week (0=Mon to 4=Fri)
"""

import logging
from datetime import date, timedelta
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from src.core.feature_base import FeatureModuleBase

logger = logging.getLogger(__name__)

# Known FOMC meeting decision dates (same set as macro_calendar_gate)
_FOMC_DATES: Set[date] = {
    # 2024
    date(2024, 1, 31), date(2024, 3, 20), date(2024, 5, 1), date(2024, 6, 12),
    date(2024, 7, 31), date(2024, 9, 18), date(2024, 11, 7), date(2024, 12, 18),
    # 2025
    date(2025, 1, 29), date(2025, 3, 19), date(2025, 5, 7), date(2025, 6, 18),
    date(2025, 7, 30), date(2025, 9, 17), date(2025, 10, 29), date(2025, 12, 17),
    # 2026
    date(2026, 1, 28), date(2026, 3, 18), date(2026, 4, 29), date(2026, 6, 17),
    date(2026, 7, 29), date(2026, 9, 16), date(2026, 10, 28), date(2026, 12, 16),
    # 2027
    date(2027, 1, 27), date(2027, 3, 17), date(2027, 4, 28), date(2027, 6, 16),
    date(2027, 7, 28), date(2027, 9, 15), date(2027, 10, 27), date(2027, 12, 15),
}


def _third_friday(year: int, month: int) -> date:
    """Return the third Friday of the given month."""
    first_day = date(year, month, 1)
    # Find first Friday
    days_until_friday = (4 - first_day.weekday()) % 7
    first_friday = first_day + timedelta(days=days_until_friday)
    return first_friday + timedelta(weeks=2)  # Third Friday


class SeasonalityFeatures(FeatureModuleBase):
    """Compute equity calendar anomaly features from dates."""
    FEATURE_NAMES = ["seas_turn_of_month", "seas_january_effect", "seas_pre_fomc_drift", "seas_holiday_drift", "seas_santa_claus", "seas_quad_witching", "seas_sell_in_may", "seas_day_of_week"]


    REQUIRED_COLS = {"close"}  # Need close for df validation; date used if available

    def create_seasonality_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """Add 8 seas_ features to df_daily."""
        df = df_daily.copy()

        if "close" not in df.columns:
            logger.warning("SeasonalityFeatures: 'close' column missing, skipping")
            return df

        # Determine date source
        if "date" in df.columns:
            dates = pd.to_datetime(df["date"])
        elif isinstance(df.index, pd.DatetimeIndex):
            dates = df.index.to_series().reset_index(drop=True)
        else:
            # Generate synthetic business dates
            dates = pd.bdate_range("2020-01-01", periods=len(df), freq="B")
            dates = pd.Series(dates)

        n = len(df)
        seas_tom = np.zeros(n)
        seas_jan = np.zeros(n)
        seas_fomc = np.zeros(n)
        seas_holiday = np.zeros(n)
        seas_santa = np.zeros(n)
        seas_quad = np.zeros(n)
        seas_sim = np.zeros(n)
        seas_dow = np.zeros(n)

        for i in range(n):
            dt = dates.iloc[i]
            if hasattr(dt, 'date'):
                d = dt.date() if callable(getattr(dt, 'date', None)) else dt
            else:
                d = dt
            if not isinstance(d, date):
                try:
                    d = pd.Timestamp(d).date()
                except Exception:
                    continue

            year, month, day = d.year, d.month, d.day
            weekday = d.weekday()  # 0=Mon, 4=Fri

            # Turn of month: last 2 or first 3 business days
            if day <= 3 or day >= 27:
                seas_tom[i] = 1.0

            # January effect
            if month == 1:
                seas_jan[i] = 1.0

            # Pre-FOMC drift: day before or day of FOMC
            if d in _FOMC_DATES:
                seas_fomc[i] = 1.0
            elif any((fomc - d).days == 1 for fomc in _FOMC_DATES if fomc >= d):
                seas_fomc[i] = 1.0

            # Holiday drift: day before major holidays (simplified)
            # Major US holidays that affect markets
            if (month == 7 and day in (3, 2)) or \
               (month == 11 and day >= 22 and weekday == 2) or \
               (month == 12 and day in (23, 24, 30, 31)) or \
               (month == 1 and day == 1) or \
               (month == 5 and day >= 25 and weekday == 4):
                seas_holiday[i] = 1.0

            # Santa Claus rally: Dec 25 to Jan 2
            if (month == 12 and day >= 25) or (month == 1 and day <= 2):
                seas_santa[i] = 1.0

            # Quad witching: third Friday of Mar, Jun, Sep, Dec
            if month in (3, 6, 9, 12):
                tf = _third_friday(year, month)
                if d == tf:
                    seas_quad[i] = 1.0

            # Sell in May: May through October
            if 5 <= month <= 10:
                seas_sim[i] = 1.0

            # Day of week
            seas_dow[i] = min(weekday, 4)  # Cap at 4 (Fri)

        df["seas_turn_of_month"] = seas_tom
        df["seas_january_effect"] = seas_jan
        df["seas_pre_fomc_drift"] = seas_fomc
        df["seas_holiday_drift"] = seas_holiday
        df["seas_santa_claus"] = seas_santa
        df["seas_quad_witching"] = seas_quad
        df["seas_sell_in_may"] = seas_sim
        df["seas_day_of_week"] = seas_dow

        # Cleanup
        for col in self._all_feature_names():
            df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        n_features = sum(1 for c in df.columns if c.startswith("seas_"))
        logger.info("SeasonalityFeatures: added %d features", n_features)
        return df

    def analyze_current_seasonality(self, df_daily: pd.DataFrame) -> Optional[Dict]:
        """Return current seasonality flags."""
        if "seas_turn_of_month" not in df_daily.columns or len(df_daily) < 2:
            return None
        last = df_daily.iloc[-1]
        active = []
        for col in self._all_feature_names():
            if col == "seas_day_of_week":
                continue
            if float(last.get(col, 0.0)) > 0.5:
                active.append(col.replace("seas_", ""))
        return {
            "active_effects": active,
            "day_of_week": int(last.get("seas_day_of_week", 0)),
            "n_active": len(active),
        }

    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "seas_turn_of_month",
            "seas_january_effect",
            "seas_pre_fomc_drift",
            "seas_holiday_drift",
            "seas_santa_claus",
            "seas_quad_witching",
            "seas_sell_in_may",
            "seas_day_of_week",
        ]
