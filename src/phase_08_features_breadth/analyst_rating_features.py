"""
GIGA TRADER - Analyst Rating Features (Wave L10)
===================================================
Consensus rating drift and revision signals from Finnhub.

Source: Finnhub recommendation trends API (free tier).
Fallback: price momentum as consensus proxy.

Features (8, prefix anlst_):
  anlst_buy_pct           -- Fraction of analysts with Buy rating
  anlst_sell_pct          -- Fraction of analysts with Sell rating
  anlst_strong_buy_pct    -- Fraction with Strong Buy
  anlst_consensus_drift   -- Month-over-month change in buy fraction
  anlst_upgrade_count     -- Recent upgrade proxy (positive momentum days)
  anlst_downgrade_count   -- Recent downgrade proxy (negative momentum days)
  anlst_net_revisions     -- Upgrades minus downgrades
  anlst_consensus_z       -- Z-score of consensus drift
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from src.core.feature_base import FeatureModuleBase

logger = logging.getLogger(__name__)


class AnalystRatingFeatures(FeatureModuleBase):
    """Compute analyst rating consensus features."""
    FEATURE_NAMES = ["anlst_buy_pct", "anlst_sell_pct", "anlst_strong_buy_pct", "anlst_consensus_drift", "anlst_upgrade_count", "anlst_downgrade_count", "anlst_net_revisions", "anlst_consensus_z"]


    REQUIRED_COLS = {"close"}

    def __init__(self) -> None:
        self._rating_data: Optional[pd.DataFrame] = None
        self._data_source: str = "none"

    def download_rating_data(
        self, start_date, end_date
    ) -> Optional[pd.DataFrame]:
        """Download analyst recommendations from Finnhub."""
        api_key = os.environ.get("FINNHUB_API_KEY", "")
        if not api_key:
            logger.info("AnalystRatingFeatures: no FINNHUB_API_KEY, will use proxy")
            return None

        try:
            import requests
        except ImportError:
            logger.info("AnalystRatingFeatures: requests not installed")
            return None

        try:
            url = "https://finnhub.io/api/v1/stock/recommendation"
            params = {"symbol": "AAPL", "token": api_key}  # Use AAPL as SPY proxy
            resp = requests.get(url, params=params, timeout=15)

            if resp.status_code != 200:
                logger.warning(
                    f"AnalystRatingFeatures: Finnhub returned {resp.status_code}"
                )
                return None

            data = resp.json()
            if not data or not isinstance(data, list):
                logger.info("AnalystRatingFeatures: no rating data")
                return None

            records = []
            for item in data:
                period = item.get("period", "")
                if period:
                    total = (
                        item.get("strongBuy", 0)
                        + item.get("buy", 0)
                        + item.get("hold", 0)
                        + item.get("sell", 0)
                        + item.get("strongSell", 0)
                    )
                    if total > 0:
                        records.append({
                            "date": pd.to_datetime(period),
                            "strong_buy": item.get("strongBuy", 0),
                            "buy": item.get("buy", 0),
                            "hold": item.get("hold", 0),
                            "sell": item.get("sell", 0),
                            "strong_sell": item.get("strongSell", 0),
                            "total": total,
                        })

            if len(records) < 3:
                logger.info("AnalystRatingFeatures: insufficient data")
                return None

            self._rating_data = pd.DataFrame(records).sort_values("date")
            self._data_source = "finnhub"
            logger.info(f"AnalystRatingFeatures: loaded {len(records)} periods")
            return self._rating_data

        except Exception as e:
            logger.warning(f"AnalystRatingFeatures: download failed: {e}")
            return None

    def create_analyst_rating_features(
        self, df_daily: pd.DataFrame
    ) -> pd.DataFrame:
        """Create analyst rating features."""
        df = df_daily.copy()

        if self._rating_data is not None and not self._rating_data.empty:
            df = self._create_full_features(df)
        else:
            df = self._create_proxy_features(df)

        # NaN/Inf cleanup
        for col in self._all_feature_names():
            if col in df.columns:
                df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        return df

    def _create_full_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features from real analyst rating data."""
        df["_date"] = pd.to_datetime(df["date"]).dt.normalize()
        ratings = self._rating_data.copy()
        ratings["_date"] = ratings["date"].dt.normalize()

        # Compute rating percentages
        ratings["buy_pct"] = (ratings["strong_buy"] + ratings["buy"]) / ratings["total"]
        ratings["sell_pct"] = (ratings["sell"] + ratings["strong_sell"]) / ratings["total"]
        ratings["strong_buy_pct"] = ratings["strong_buy"] / ratings["total"]

        # Merge with point-in-time: for each trading day, use latest known rating
        # This is a forward-fill merge
        ratings_slim = ratings[["_date", "buy_pct", "sell_pct", "strong_buy_pct"]].copy()
        df = df.merge(ratings_slim, on="_date", how="left")

        for col in ["buy_pct", "sell_pct", "strong_buy_pct"]:
            df[col] = df[col].ffill()

        df["anlst_buy_pct"] = df["buy_pct"].fillna(0.5)
        df["anlst_sell_pct"] = df["sell_pct"].fillna(0.1)
        df["anlst_strong_buy_pct"] = df["strong_buy_pct"].fillna(0.2)

        # Consensus drift (change in buy fraction)
        df["anlst_consensus_drift"] = df["anlst_buy_pct"].diff(20)

        # Upgrade/downgrade counts from buy_pct changes
        buy_change = df["anlst_buy_pct"].diff()
        df["anlst_upgrade_count"] = (buy_change > 0).astype(float).rolling(20).sum()
        df["anlst_downgrade_count"] = (buy_change < 0).astype(float).rolling(20).sum()
        df["anlst_net_revisions"] = df["anlst_upgrade_count"] - df["anlst_downgrade_count"]

        # Consensus z-score
        drift = df["anlst_consensus_drift"]
        mu = drift.rolling(60).mean()
        std = drift.rolling(60).std()
        df["anlst_consensus_z"] = (drift - mu) / (std + 1e-8)

        df.drop(
            columns=["_date", "buy_pct", "sell_pct", "strong_buy_pct"],
            inplace=True,
            errors="ignore",
        )
        return df

    def _create_proxy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Proxy: use price momentum as consensus proxy."""
        logger.info("AnalystRatingFeatures: using price momentum proxy")
        spy_ret = df["close"].pct_change()

        # Positive momentum ≈ analyst upgrades
        mom_20 = spy_ret.rolling(20).mean()
        pos_days = (spy_ret > 0).astype(float).rolling(20).mean()

        df["anlst_buy_pct"] = pos_days
        df["anlst_sell_pct"] = 1.0 - pos_days
        df["anlst_strong_buy_pct"] = (spy_ret > spy_ret.quantile(0.8)).astype(
            float
        ).rolling(20).mean()
        df["anlst_consensus_drift"] = pos_days.diff(20)
        df["anlst_upgrade_count"] = (spy_ret > 0).astype(float).rolling(20).sum()
        df["anlst_downgrade_count"] = (spy_ret < 0).astype(float).rolling(20).sum()
        df["anlst_net_revisions"] = (
            df["anlst_upgrade_count"] - df["anlst_downgrade_count"]
        )
        drift = df["anlst_consensus_drift"]
        mu = drift.rolling(60).mean()
        std = drift.rolling(60).std()
        df["anlst_consensus_z"] = (drift - mu) / (std + 1e-8)

        return df

    def analyze_current_ratings(
        self, df_daily: pd.DataFrame
    ) -> Optional[Dict]:
        """Analyze current analyst consensus."""
        if df_daily.empty or "anlst_buy_pct" not in df_daily.columns:
            return None

        last = df_daily.iloc[-1]
        buy = float(last.get("anlst_buy_pct", 0.5))
        sell = float(last.get("anlst_sell_pct", 0.1))

        if buy > 0.7:
            consensus = "STRONG_BUY"
        elif buy > 0.5:
            consensus = "BUY"
        elif sell > 0.3:
            consensus = "SELL"
        else:
            consensus = "HOLD"

        return {
            "consensus": consensus,
            "buy_pct": buy,
            "sell_pct": sell,
            "net_revisions": float(last.get("anlst_net_revisions", 0)),
            "consensus_z": float(last.get("anlst_consensus_z", 0)),
            "source": self._data_source,
        }

    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "anlst_buy_pct",
            "anlst_sell_pct",
            "anlst_strong_buy_pct",
            "anlst_consensus_drift",
            "anlst_upgrade_count",
            "anlst_downgrade_count",
            "anlst_net_revisions",
            "anlst_consensus_z",
        ]
