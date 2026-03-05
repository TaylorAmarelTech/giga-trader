"""
GIGA TRADER - Earnings Revision Features (Wave L1)
=====================================================
Earnings estimate revision momentum from Finnhub.

Source: Finnhub earnings estimates API (free tier).
Fallback: price/volume momentum proxy.

Features (8, prefix ern_):
  ern_revision_momentum    -- 20d rolling slope of estimate changes (proxy: price momentum)
  ern_surprise_history     -- Rolling mean of past earnings surprise z-scores
  ern_revision_breadth     -- Fraction of analysts revising up (proxy: up-days fraction)
  ern_estimate_dispersion  -- Std dev of analyst estimates (proxy: realized vol)
  ern_surprise_z           -- Z-score of last surprise vs rolling window
  ern_revision_accel       -- 5d change in revision momentum
  ern_consensus_drift      -- 20d vs 60d estimate trend difference
  ern_revision_std         -- Rolling std of revision momentum (stability)
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from src.core.feature_base import FeatureModuleBase

logger = logging.getLogger(__name__)


class EarningsRevisionFeatures(FeatureModuleBase):
    """Compute earnings revision momentum features."""
    FEATURE_NAMES = ["ern_revision_momentum", "ern_surprise_history", "ern_revision_breadth", "ern_estimate_dispersion", "ern_surprise_z", "ern_revision_accel", "ern_consensus_drift", "ern_revision_std"]


    REQUIRED_COLS = {"close", "volume"}

    def __init__(self) -> None:
        self._earnings_data: Optional[pd.DataFrame] = None
        self._data_source: str = "none"

    def download_earnings_data(
        self, start_date, end_date
    ) -> Optional[pd.DataFrame]:
        """Download earnings estimates from Finnhub.

        Level 1: Finnhub earnings estimates API
        Level 2: None (will use price/volume proxy)
        """
        api_key = os.environ.get("FINNHUB_API_KEY", "")
        if not api_key:
            logger.info("EarningsRevisionFeatures: no FINNHUB_API_KEY, will use proxy")
            return None

        try:
            import requests
        except ImportError:
            logger.info("EarningsRevisionFeatures: requests not installed, will use proxy")
            return None

        try:
            # Fetch earnings surprises for SPY
            url = "https://finnhub.io/api/v1/stock/earnings"
            params = {"symbol": "SPY", "token": api_key}
            resp = requests.get(url, params=params, timeout=15)

            if resp.status_code != 200:
                logger.warning(
                    f"EarningsRevisionFeatures: Finnhub returned {resp.status_code}"
                )
                return None

            data = resp.json()
            if not data or not isinstance(data, list):
                logger.info("EarningsRevisionFeatures: no earnings data from Finnhub")
                return None

            # Parse earnings data into DataFrame
            records = []
            for item in data:
                period = item.get("period", "")
                actual = item.get("actual")
                estimate = item.get("estimate")
                surprise_pct = item.get("surprisePercent")
                if period and actual is not None and estimate is not None:
                    records.append({
                        "date": pd.to_datetime(period),
                        "actual": float(actual),
                        "estimate": float(estimate),
                        "surprise_pct": float(surprise_pct) if surprise_pct else 0.0,
                    })

            if len(records) < 4:
                logger.info("EarningsRevisionFeatures: insufficient earnings data")
                return None

            self._earnings_data = pd.DataFrame(records).sort_values("date")
            self._data_source = "finnhub"
            logger.info(
                f"EarningsRevisionFeatures: loaded {len(records)} earnings records"
            )
            return self._earnings_data

        except Exception as e:
            logger.warning(f"EarningsRevisionFeatures: Finnhub download failed: {e}")
            return None

    def create_earnings_revision_features(
        self, df_daily: pd.DataFrame
    ) -> pd.DataFrame:
        """Create earnings revision features."""
        df = df_daily.copy()

        if self._earnings_data is not None and not self._earnings_data.empty:
            df = self._create_full_features(df)
        else:
            df = self._create_proxy_features(df)

        # NaN/Inf cleanup
        for col in self._all_feature_names():
            if col in df.columns:
                df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        return df

    def _create_full_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features from real earnings data."""
        df["_date"] = pd.to_datetime(df["date"]).dt.normalize()
        earn = self._earnings_data.copy()
        earn["_date"] = earn["date"].dt.normalize()

        # For each day, assign latest known earnings metrics
        # (point-in-time: only use earnings data known before that date)
        surprise_series = pd.Series(np.nan, index=df.index)
        estimate_series = pd.Series(np.nan, index=df.index)

        for idx, row in df.iterrows():
            known = earn[earn["_date"] <= row["_date"]]
            if len(known) > 0:
                surprise_series.iloc[idx] = known["surprise_pct"].iloc[-1]
                estimate_series.iloc[idx] = known["estimate"].iloc[-1]

        surprise_series = surprise_series.ffill()
        estimate_series = estimate_series.ffill()

        # Revision momentum: rolling change in estimate
        est_change = estimate_series.diff()
        df["ern_revision_momentum"] = est_change.rolling(20, min_periods=1).mean()
        df["ern_surprise_history"] = surprise_series.rolling(8, min_periods=1).mean()
        # Breadth: sign of recent estimate changes
        pos_changes = (est_change > 0).astype(float).rolling(20, min_periods=1).mean()
        df["ern_revision_breadth"] = pos_changes
        df["ern_estimate_dispersion"] = estimate_series.rolling(8, min_periods=1).std()
        # Surprise z-score
        mu_s = surprise_series.rolling(8, min_periods=1).mean()
        std_s = surprise_series.rolling(8, min_periods=1).std()
        df["ern_surprise_z"] = (surprise_series - mu_s) / (std_s + 1e-8)
        df["ern_revision_accel"] = df["ern_revision_momentum"].diff(5)
        mom_20 = est_change.rolling(20, min_periods=1).mean()
        mom_60 = est_change.rolling(60, min_periods=1).mean()
        df["ern_consensus_drift"] = mom_20 - mom_60
        df["ern_revision_std"] = df["ern_revision_momentum"].rolling(20, min_periods=1).std()

        df.drop(columns=["_date"], inplace=True, errors="ignore")
        return df

    def _create_proxy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Proxy: use price/volume momentum as earnings revision proxy."""
        logger.info("EarningsRevisionFeatures: using price/volume proxy features")
        spy_ret = df["close"].pct_change()
        vol_ratio = df["volume"].astype(float) / (
            df["volume"].astype(float).rolling(20).mean() + 1e-8
        )

        # Price momentum as revision momentum proxy
        df["ern_revision_momentum"] = spy_ret.rolling(20).mean()
        df["ern_surprise_history"] = spy_ret.rolling(60).mean()
        df["ern_revision_breadth"] = (spy_ret > 0).astype(float).rolling(20).mean()
        df["ern_estimate_dispersion"] = spy_ret.rolling(20).std()
        df["ern_surprise_z"] = (spy_ret - spy_ret.rolling(60).mean()) / (
            spy_ret.rolling(60).std() + 1e-8
        )
        df["ern_revision_accel"] = df["ern_revision_momentum"].diff(5)
        df["ern_consensus_drift"] = (
            spy_ret.rolling(20).mean() - spy_ret.rolling(60).mean()
        )
        df["ern_revision_std"] = df["ern_revision_momentum"].rolling(20).std()

        return df

    def analyze_current_earnings_revision(
        self, df_daily: pd.DataFrame
    ) -> Optional[Dict]:
        """Analyze current earnings revision regime."""
        if df_daily.empty or "ern_revision_momentum" not in df_daily.columns:
            return None

        last = df_daily.iloc[-1]
        mom = float(last.get("ern_revision_momentum", 0))
        breadth = float(last.get("ern_revision_breadth", 0.5))

        if mom > 0 and breadth > 0.6:
            regime = "POSITIVE_REVISIONS"
        elif mom < 0 and breadth < 0.4:
            regime = "NEGATIVE_REVISIONS"
        else:
            regime = "MIXED"

        return {
            "revision_regime": regime,
            "revision_momentum": mom,
            "revision_breadth": breadth,
            "surprise_z": float(last.get("ern_surprise_z", 0)),
            "source": self._data_source,
        }

    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "ern_revision_momentum",
            "ern_surprise_history",
            "ern_revision_breadth",
            "ern_estimate_dispersion",
            "ern_surprise_z",
            "ern_revision_accel",
            "ern_consensus_drift",
            "ern_revision_std",
        ]
