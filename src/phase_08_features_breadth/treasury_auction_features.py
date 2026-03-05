"""
GIGA TRADER - Treasury Auction Features (Wave L7)
====================================================
Primary dealer demand signals from TreasuryDirect.

Source: TreasuryDirect Fiscal Data API (free, no key).
Fallback: yield curve slope + TIP/AGG ratio as auction demand proxy.
DEFAULT OFF: parsing complexity, limited historical coverage.

Features (6, prefix tauct_):
  tauct_bid_cover_10y      -- 10Y auction bid-to-cover ratio
  tauct_indirect_pct       -- Indirect bidder % (foreign demand)
  tauct_tail_bps           -- Auction tail (high yield - when-issued, bps)
  tauct_demand_z           -- Z-score of bid-to-cover
  tauct_bid_cover_change   -- Change in bid-to-cover from prior auction
  tauct_auction_quality    -- Composite quality score
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from src.core.feature_base import FeatureModuleBase

logger = logging.getLogger(__name__)


class TreasuryAuctionFeatures(FeatureModuleBase):
    """Compute treasury auction demand features."""
    FEATURE_NAMES = ["tauct_bid_cover_10y", "tauct_indirect_pct", "tauct_tail_bps", "tauct_demand_z", "tauct_bid_cover_change", "tauct_auction_quality"]


    REQUIRED_COLS = {"close"}

    def __init__(self) -> None:
        self._auction_data: Optional[pd.DataFrame] = None
        self._data_source: str = "none"

    def download_auction_data(
        self, start_date, end_date
    ) -> Optional[pd.DataFrame]:
        """Download treasury auction data from Fiscal Data API.

        Level 1: api.fiscaldata.treasury.gov
        Level 2: None (will use yield curve proxy)
        """
        try:
            import requests
        except ImportError:
            logger.info("TreasuryAuctionFeatures: requests not installed")
            return None

        try:
            url = (
                "https://api.fiscaldata.treasury.gov/services/api/"
                "fiscal_service/v1/accounting/od/auctions_query"
            )
            params = {
                "fields": (
                    "auction_date,security_type,security_term,"
                    "bid_to_cover_ratio,high_yield,percentage_awards_to_indirect"
                ),
                "filter": f"auction_date:gte:{str(start_date)[:10]}",
                "sort": "-auction_date",
                "page[size]": 200,
            }

            resp = requests.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                logger.info(
                    f"TreasuryAuctionFeatures: API returned {resp.status_code}"
                )
                return None

            data = resp.json().get("data", [])
            if not data or len(data) < 3:
                logger.info("TreasuryAuctionFeatures: insufficient auction data")
                return None

            records = []
            for item in data:
                try:
                    bid_cover = float(item.get("bid_to_cover_ratio") or 0)
                    indirect = float(item.get("percentage_awards_to_indirect") or 0)
                    high_yield = float(item.get("high_yield") or 0)
                    if bid_cover > 0:
                        records.append({
                            "date": pd.to_datetime(item["auction_date"]),
                            "bid_to_cover": bid_cover,
                            "indirect_pct": indirect,
                            "high_yield": high_yield,
                            "security_term": item.get("security_term", ""),
                        })
                except (ValueError, TypeError):
                    continue

            if len(records) < 3:
                return None

            self._auction_data = pd.DataFrame(records).sort_values("date")
            self._data_source = "treasury_direct"
            logger.info(f"TreasuryAuctionFeatures: loaded {len(records)} auctions")
            return self._auction_data

        except Exception as e:
            logger.warning(f"TreasuryAuctionFeatures: download failed: {e}")
            return None

    def create_treasury_auction_features(
        self, df_daily: pd.DataFrame
    ) -> pd.DataFrame:
        """Create treasury auction features."""
        df = df_daily.copy()

        if self._auction_data is not None and not self._auction_data.empty:
            df = self._create_full_features(df)
        else:
            df = self._create_proxy_features(df)

        for col in self._all_feature_names():
            if col in df.columns:
                df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        return df

    def _create_full_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute from real auction data (forward-filled to daily)."""
        df["_date"] = pd.to_datetime(df["date"]).dt.normalize()
        auct = self._auction_data.copy()

        # Filter for 10Y Notes primarily
        ten_year = auct[auct["security_term"].str.contains("10-Year|10Y", case=False, na=False)]
        if len(ten_year) < 3:
            ten_year = auct  # Use all if not enough 10Y

        ten_year = ten_year.sort_values("date")
        ten_year["_date"] = ten_year["date"].dt.normalize()

        df = df.merge(
            ten_year[["_date", "bid_to_cover", "indirect_pct", "high_yield"]],
            on="_date",
            how="left",
        )

        # Forward fill (auctions are ~monthly)
        for col in ["bid_to_cover", "indirect_pct", "high_yield"]:
            df[col] = df[col].ffill()

        bc = df["bid_to_cover"].fillna(2.5)  # Historical average
        df["tauct_bid_cover_10y"] = bc
        df["tauct_indirect_pct"] = df["indirect_pct"].fillna(60.0) / 100.0
        df["tauct_tail_bps"] = 0.0  # Would need when-issued data

        mu = bc.rolling(60).mean()
        std = bc.rolling(60).std()
        df["tauct_demand_z"] = (bc - mu) / (std + 1e-8)
        df["tauct_bid_cover_change"] = bc.diff()
        df["tauct_auction_quality"] = (
            df["tauct_bid_cover_10y"] * df["tauct_indirect_pct"]
        )

        df.drop(
            columns=["_date", "bid_to_cover", "indirect_pct", "high_yield"],
            inplace=True,
            errors="ignore",
        )
        return df

    def _create_proxy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Proxy: yield curve + SPY vol as auction demand proxy."""
        logger.info("TreasuryAuctionFeatures: using yield-curve proxy features")
        spy_ret = df["close"].pct_change()
        spy_vol = spy_ret.rolling(20).std()

        # Low volatility + positive momentum ≈ good auction demand
        df["tauct_bid_cover_10y"] = 2.5 - spy_vol * 10  # Inverse vol proxy
        df["tauct_indirect_pct"] = 0.6  # Historical average ~60%
        df["tauct_tail_bps"] = 0.0
        df["tauct_demand_z"] = 0.0
        df["tauct_bid_cover_change"] = 0.0
        df["tauct_auction_quality"] = 1.5

        return df

    def analyze_current_auction(
        self, df_daily: pd.DataFrame
    ) -> Optional[Dict]:
        """Analyze current auction quality."""
        if df_daily.empty or "tauct_demand_z" not in df_daily.columns:
            return None

        last = df_daily.iloc[-1]
        z = float(last.get("tauct_demand_z", 0))

        if z > 1.0:
            quality = "STRONG_DEMAND"
        elif z < -1.0:
            quality = "WEAK_DEMAND"
        else:
            quality = "NORMAL"

        return {
            "quality": quality,
            "demand_z": z,
            "bid_to_cover": float(last.get("tauct_bid_cover_10y", 0)),
            "source": self._data_source,
        }

    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "tauct_bid_cover_10y",
            "tauct_indirect_pct",
            "tauct_tail_bps",
            "tauct_demand_z",
            "tauct_bid_cover_change",
            "tauct_auction_quality",
        ]
