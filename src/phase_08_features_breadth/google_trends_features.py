"""
GIGA TRADER - Google Trends Features (Wave L5)
=================================================
Retail investor attention and panic signals from Google Trends.

Source: pytrends library (free, rate-limited).
Fallback: VIX-volume composite as attention proxy.
DEFAULT OFF: rate limits make automated use fragile.

Features (8, prefix gtrend_):
  gtrend_crash_interest     -- Interest in "stock market crash"
  gtrend_buy_interest       -- Interest in "buy stocks"
  gtrend_recession_interest -- Interest in "recession"
  gtrend_bull_interest      -- Interest in "bull market"
  gtrend_panic_idx          -- crash + recession combined
  gtrend_euphoria_idx       -- buy + bull combined
  gtrend_momentum           -- 5d change in panic_idx
  gtrend_contrarian         -- euphoria - panic (contrarian signal)
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class GoogleTrendsFeatures:
    """Compute Google Trends sentiment features."""

    REQUIRED_COLS = {"close", "volume"}

    SEARCH_TERMS = {
        "crash": "stock market crash",
        "buy": "buy stocks",
        "recession": "recession",
        "bull": "bull market",
    }

    def __init__(self) -> None:
        self._trends_data: Optional[pd.DataFrame] = None
        self._data_source: str = "none"

    def download_trends_data(
        self, start_date, end_date
    ) -> Optional[pd.DataFrame]:
        """Download Google Trends data via pytrends.

        Level 1: pytrends (Google Trends)
        Level 2: None (will use VIX/volume proxy)
        """
        try:
            from pytrends.request import TrendReq
        except ImportError:
            logger.info("GoogleTrendsFeatures: pytrends not installed, will use proxy")
            return None

        try:
            pytrends = TrendReq(hl="en-US", tz=300)  # Eastern time
            start = str(start_date)[:10]
            end = str(end_date)[:10]
            timeframe = f"{start} {end}"

            kw_list = list(self.SEARCH_TERMS.values())
            # Google Trends allows max 5 keywords at once
            pytrends.build_payload(kw_list, timeframe=timeframe, geo="US")
            data = pytrends.interest_over_time()

            if data is None or data.empty:
                logger.info("GoogleTrendsFeatures: no data from Google Trends")
                return None

            # Rename columns to short names
            rename_map = {v: k for k, v in self.SEARCH_TERMS.items()}
            data = data.rename(columns=rename_map)
            if "isPartial" in data.columns:
                data.drop(columns=["isPartial"], inplace=True)

            self._trends_data = data
            self._data_source = "google_trends"
            logger.info(f"GoogleTrendsFeatures: loaded {len(data)} data points")
            return self._trends_data

        except Exception as e:
            logger.warning(f"GoogleTrendsFeatures: pytrends failed: {e}")
            return None

    def create_google_trends_features(
        self, df_daily: pd.DataFrame
    ) -> pd.DataFrame:
        """Create Google Trends features."""
        df = df_daily.copy()

        if self._trends_data is not None and not self._trends_data.empty:
            df = self._create_full_features(df)
        else:
            df = self._create_proxy_features(df)

        for col in self._all_feature_names():
            if col in df.columns:
                df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        return df

    def _create_full_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute from real Google Trends data."""
        df["_date"] = pd.to_datetime(df["date"]).dt.normalize()
        trends = self._trends_data.copy()
        trends["_date"] = trends.index.normalize()

        # Google Trends is weekly, forward-fill to daily
        df = df.merge(trends, on="_date", how="left")
        for term in self.SEARCH_TERMS:
            if term in df.columns:
                df[term] = df[term].ffill().fillna(0)

        # Normalize to 0-1
        for term in self.SEARCH_TERMS:
            if term in df.columns:
                max_val = df[term].max()
                if max_val > 0:
                    df[term] = df[term] / max_val

        df["gtrend_crash_interest"] = df.get("crash", pd.Series(0.0, index=df.index))
        df["gtrend_buy_interest"] = df.get("buy", pd.Series(0.0, index=df.index))
        df["gtrend_recession_interest"] = df.get("recession", pd.Series(0.0, index=df.index))
        df["gtrend_bull_interest"] = df.get("bull", pd.Series(0.0, index=df.index))

        df["gtrend_panic_idx"] = (
            df["gtrend_crash_interest"] + df["gtrend_recession_interest"]
        ) / 2.0
        df["gtrend_euphoria_idx"] = (
            df["gtrend_buy_interest"] + df["gtrend_bull_interest"]
        ) / 2.0
        df["gtrend_momentum"] = df["gtrend_panic_idx"].diff(5)
        df["gtrend_contrarian"] = df["gtrend_euphoria_idx"] - df["gtrend_panic_idx"]

        # Cleanup temp columns
        for term in self.SEARCH_TERMS:
            if term in df.columns:
                df.drop(columns=[term], inplace=True)
        df.drop(columns=["_date"], inplace=True, errors="ignore")

        return df

    def _create_proxy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Proxy: VIX-volume composite as attention proxy."""
        logger.info("GoogleTrendsFeatures: using VIX-volume proxy features")
        spy_ret = df["close"].pct_change()
        vol = df["volume"].astype(float)
        vol_ma = vol.rolling(20).mean()
        vol_spike = vol / (vol_ma + 1e-8) - 1.0

        # Crash interest proxy: high volume + negative returns
        crash_proxy = vol_spike * (-spy_ret).clip(lower=0)
        df["gtrend_crash_interest"] = crash_proxy.rolling(5).mean()

        # Buy interest proxy: high volume + positive returns
        buy_proxy = vol_spike * spy_ret.clip(lower=0)
        df["gtrend_buy_interest"] = buy_proxy.rolling(5).mean()

        # Recession proxy: sustained negative momentum
        df["gtrend_recession_interest"] = (-spy_ret.rolling(20).mean()).clip(lower=0) * 10

        # Bull proxy: sustained positive momentum
        df["gtrend_bull_interest"] = spy_ret.rolling(20).mean().clip(lower=0) * 10

        df["gtrend_panic_idx"] = (
            df["gtrend_crash_interest"] + df["gtrend_recession_interest"]
        ) / 2.0
        df["gtrend_euphoria_idx"] = (
            df["gtrend_buy_interest"] + df["gtrend_bull_interest"]
        ) / 2.0
        df["gtrend_momentum"] = df["gtrend_panic_idx"].diff(5)
        df["gtrend_contrarian"] = df["gtrend_euphoria_idx"] - df["gtrend_panic_idx"]

        return df

    def analyze_current_trends(
        self, df_daily: pd.DataFrame
    ) -> Optional[Dict]:
        """Analyze current search trends."""
        if df_daily.empty or "gtrend_panic_idx" not in df_daily.columns:
            return None

        last = df_daily.iloc[-1]
        panic = float(last.get("gtrend_panic_idx", 0))
        euphoria = float(last.get("gtrend_euphoria_idx", 0))

        if panic > euphoria * 1.5:
            mood = "FEAR"
        elif euphoria > panic * 1.5:
            mood = "GREED"
        else:
            mood = "NEUTRAL"

        return {
            "mood": mood,
            "panic_idx": panic,
            "euphoria_idx": euphoria,
            "contrarian": float(last.get("gtrend_contrarian", 0)),
            "source": self._data_source,
        }

    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "gtrend_crash_interest",
            "gtrend_buy_interest",
            "gtrend_recession_interest",
            "gtrend_bull_interest",
            "gtrend_panic_idx",
            "gtrend_euphoria_idx",
            "gtrend_momentum",
            "gtrend_contrarian",
        ]
