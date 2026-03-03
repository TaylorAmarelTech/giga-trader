"""
GIGA TRADER - Earnings Calendar Features (Wave L9)
=====================================================
Earnings season density and anticipation signals.

Source: Finnhub earnings calendar API (free tier).
Fallback: seasonality proxy (Q1-Q4 earnings season pattern).

Features (6, prefix ecal_):
  ecal_density_this_week   -- Count of S&P 500 earnings this week (normalized)
  ecal_density_next_week   -- Count of earnings next week (normalized)
  ecal_season_flag         -- 1.0 if in peak earnings season
  ecal_tech_density        -- Tech sector earnings density this week
  ecal_spy_component_pct   -- % of SPY components reporting this week
  ecal_pre_earnings_vol    -- Implied vol proxy before heavy earnings
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class EarningsCalendarFeatures:
    """Compute earnings calendar density features."""

    REQUIRED_COLS = {"close"}

    # Peak earnings weeks: ~3-5 weeks after quarter end
    # Q4 earnings: mid-Jan to mid-Feb
    # Q1 earnings: mid-Apr to mid-May
    # Q2 earnings: mid-Jul to mid-Aug
    # Q3 earnings: mid-Oct to mid-Nov
    PEAK_MONTHS_WEEKS = {
        (1, 3): True, (1, 4): True, (2, 1): True, (2, 2): True,
        (4, 3): True, (4, 4): True, (5, 1): True, (5, 2): True,
        (7, 3): True, (7, 4): True, (8, 1): True, (8, 2): True,
        (10, 3): True, (10, 4): True, (11, 1): True, (11, 2): True,
    }

    def __init__(self) -> None:
        self._calendar_data: Optional[pd.DataFrame] = None
        self._data_source: str = "none"

    def download_calendar_data(
        self, start_date, end_date
    ) -> Optional[pd.DataFrame]:
        """Download earnings calendar from Finnhub."""
        api_key = os.environ.get("FINNHUB_API_KEY", "")
        if not api_key:
            logger.info("EarningsCalendarFeatures: no FINNHUB_API_KEY, will use proxy")
            return None

        try:
            import requests
        except ImportError:
            logger.info("EarningsCalendarFeatures: requests not installed")
            return None

        try:
            url = "https://finnhub.io/api/v1/calendar/earnings"
            # Finnhub only allows short date ranges, fetch in chunks
            start = pd.to_datetime(str(start_date)[:10])
            end = pd.to_datetime(str(end_date)[:10])

            all_records = []
            current = start
            while current < end:
                chunk_end = min(current + pd.Timedelta(days=30), end)
                params = {
                    "from": current.strftime("%Y-%m-%d"),
                    "to": chunk_end.strftime("%Y-%m-%d"),
                    "token": api_key,
                }
                resp = requests.get(url, params=params, timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    earnings = data.get("earningsCalendar", [])
                    for item in earnings:
                        all_records.append({
                            "date": item.get("date", ""),
                            "symbol": item.get("symbol", ""),
                        })
                current = chunk_end + pd.Timedelta(days=1)

            if len(all_records) < 10:
                logger.info("EarningsCalendarFeatures: insufficient calendar data")
                return None

            self._calendar_data = pd.DataFrame(all_records)
            self._calendar_data["date"] = pd.to_datetime(self._calendar_data["date"])
            self._data_source = "finnhub"
            logger.info(
                f"EarningsCalendarFeatures: loaded {len(all_records)} earnings events"
            )
            return self._calendar_data

        except Exception as e:
            logger.warning(f"EarningsCalendarFeatures: download failed: {e}")
            return None

    def create_earnings_calendar_features(
        self, df_daily: pd.DataFrame
    ) -> pd.DataFrame:
        """Create earnings calendar features."""
        df = df_daily.copy()

        if self._calendar_data is not None and not self._calendar_data.empty:
            df = self._create_full_features(df)
        else:
            df = self._create_proxy_features(df)

        # NaN/Inf cleanup
        for col in self._all_feature_names():
            if col in df.columns:
                df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        return df

    def _create_full_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features from real calendar data."""
        df["_date"] = pd.to_datetime(df["date"]).dt.normalize()

        # Count earnings per day
        daily_counts = (
            self._calendar_data.groupby(
                self._calendar_data["date"].dt.normalize()
            )
            .size()
            .reset_index(name="count")
        )
        daily_counts.columns = ["_date", "daily_count"]

        df = df.merge(daily_counts, on="_date", how="left")
        df["daily_count"] = df["daily_count"].fillna(0)

        # This week density (rolling 5-day sum)
        df["ecal_density_this_week"] = (
            df["daily_count"].rolling(5, min_periods=1).sum() / 500.0
        )  # Normalized by ~S&P 500

        # Next week density (shift forward)
        df["ecal_density_next_week"] = df["ecal_density_this_week"].shift(-5).fillna(
            df["ecal_density_this_week"]
        )

        # Season flag from actual density
        density_threshold = df["ecal_density_this_week"].quantile(0.75)
        df["ecal_season_flag"] = (
            df["ecal_density_this_week"] >= density_threshold
        ).astype(float)

        # Tech density: count symbols with tech-like tickers (rough proxy)
        # Without a real sector map, use overall density as proxy
        df["ecal_tech_density"] = df["ecal_density_this_week"] * 0.3

        # SPY component percentage
        df["ecal_spy_component_pct"] = df["ecal_density_this_week"]

        # Pre-earnings vol: vol expansion before heavy weeks
        spy_vol = df["close"].pct_change().rolling(5).std()
        df["ecal_pre_earnings_vol"] = spy_vol * df["ecal_density_this_week"]

        df.drop(columns=["_date", "daily_count"], inplace=True, errors="ignore")
        return df

    def _create_proxy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Proxy: use known earnings season calendar pattern."""
        logger.info("EarningsCalendarFeatures: using seasonal proxy features")
        dates = pd.to_datetime(df["date"])
        months = dates.dt.month
        weeks = ((dates.dt.day - 1) // 7) + 1

        # Peak season flag
        is_peak = pd.Series(0.0, index=df.index)
        for i in range(len(df)):
            key = (int(months.iloc[i]), int(weeks.iloc[i]))
            if key in self.PEAK_MONTHS_WEEKS:
                is_peak.iloc[i] = 1.0

        df["ecal_density_this_week"] = is_peak * 0.1  # Normalized estimate
        df["ecal_density_next_week"] = is_peak.shift(-5).fillna(is_peak) * 0.1
        df["ecal_season_flag"] = is_peak
        df["ecal_tech_density"] = is_peak * 0.03
        df["ecal_spy_component_pct"] = is_peak * 0.1
        spy_vol = df["close"].pct_change().rolling(5).std()
        df["ecal_pre_earnings_vol"] = spy_vol * is_peak

        return df

    def analyze_current_calendar(
        self, df_daily: pd.DataFrame
    ) -> Optional[Dict]:
        """Analyze current earnings calendar state."""
        if df_daily.empty or "ecal_season_flag" not in df_daily.columns:
            return None

        last = df_daily.iloc[-1]
        return {
            "in_earnings_season": bool(last.get("ecal_season_flag", 0) > 0.5),
            "density_this_week": float(last.get("ecal_density_this_week", 0)),
            "density_next_week": float(last.get("ecal_density_next_week", 0)),
            "source": self._data_source,
        }

    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "ecal_density_this_week",
            "ecal_density_next_week",
            "ecal_season_flag",
            "ecal_tech_density",
            "ecal_spy_component_pct",
            "ecal_pre_earnings_vol",
        ]
