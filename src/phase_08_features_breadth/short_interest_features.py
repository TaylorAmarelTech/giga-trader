"""
GIGA TRADER - Short Interest Features (Wave L2)
==================================================
FINRA short interest data signals.

Source: FINRA API (bi-monthly short interest, free).
Fallback: volume/price proxy (high volume + decline approximates SI).
DEFAULT OFF: bi-monthly data is sparse.

Features (8, prefix si_):
  si_ratio             -- Short interest as ratio of avg volume
  si_change            -- Period-over-period change in SI ratio
  si_days_to_cover     -- Short interest / avg daily volume
  si_velocity          -- Rate of change of si_ratio
  si_z_score           -- 60d z-score of si_ratio
  si_pct_float         -- SI normalized (proxy: volume anomaly)
  si_change_5d         -- 5-day change in si_ratio
  si_squeeze_signal    -- SI high + price rising = squeeze indicator
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ShortInterestFeatures:
    """Compute short interest features."""

    REQUIRED_COLS = {"close", "volume"}

    def __init__(self) -> None:
        self._si_data: Optional[pd.DataFrame] = None
        self._data_source: str = "none"

    def download_short_interest_data(
        self, start_date, end_date
    ) -> Optional[pd.DataFrame]:
        """Download short interest from FINRA API.

        Level 1: FINRA short interest API
        Level 2: None (will use volume proxy)
        """
        try:
            import requests
        except ImportError:
            logger.info("ShortInterestFeatures: requests not installed")
            return None

        try:
            url = "https://api.finra.org/data/group/otcMarket/name/shortInterest"
            headers = {"Accept": "application/json"}
            params = {
                "symbol": "SPY",
                "limit": 100,
            }
            resp = requests.get(url, headers=headers, params=params, timeout=15)

            if resp.status_code != 200:
                logger.info(
                    f"ShortInterestFeatures: FINRA returned {resp.status_code}, "
                    "will use proxy"
                )
                return None

            data = resp.json()
            if not data or not isinstance(data, list) or len(data) < 3:
                logger.info("ShortInterestFeatures: insufficient FINRA data")
                return None

            records = []
            for item in data:
                records.append({
                    "date": pd.to_datetime(item.get("settlementDate", "")),
                    "short_interest": float(item.get("shortInterest", 0)),
                    "avg_volume": float(item.get("avgDailyVolume", 1)),
                })

            self._si_data = pd.DataFrame(records).sort_values("date")
            self._data_source = "finra"
            logger.info(f"ShortInterestFeatures: loaded {len(records)} SI records")
            return self._si_data

        except Exception as e:
            logger.warning(f"ShortInterestFeatures: FINRA download failed: {e}")
            return None

    def create_short_interest_features(
        self, df_daily: pd.DataFrame
    ) -> pd.DataFrame:
        """Create short interest features."""
        df = df_daily.copy()

        if self._si_data is not None and not self._si_data.empty:
            df = self._create_full_features(df)
        else:
            df = self._create_proxy_features(df)

        # NaN/Inf cleanup
        for col in self._all_feature_names():
            if col in df.columns:
                df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        return df

    def _create_full_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute from real SI data (forward-filled bi-monthly to daily)."""
        df["_date"] = pd.to_datetime(df["date"]).dt.normalize()
        si = self._si_data.copy()
        si["_date"] = si["date"].dt.normalize()
        si["si_ratio_raw"] = si["short_interest"] / (si["avg_volume"] + 1e-8)

        df = df.merge(si[["_date", "si_ratio_raw"]], on="_date", how="left")
        df["si_ratio_raw"] = df["si_ratio_raw"].ffill()

        si_r = df["si_ratio_raw"].fillna(0.0)
        df["si_ratio"] = si_r
        df["si_change"] = si_r.diff()
        df["si_days_to_cover"] = si_r  # SI ratio ≈ days to cover
        df["si_velocity"] = si_r.diff().rolling(5).mean()
        mu = si_r.rolling(60).mean()
        std = si_r.rolling(60).std()
        df["si_z_score"] = (si_r - mu) / (std + 1e-8)
        df["si_pct_float"] = si_r / (si_r.rolling(252).max() + 1e-8)
        df["si_change_5d"] = si_r.diff(5)

        # Squeeze signal: high SI + price rising
        price_rising = (df["close"].pct_change(5) > 0.01).astype(float)
        si_high = (df["si_z_score"] > 1.0).astype(float)
        df["si_squeeze_signal"] = price_rising * si_high

        df.drop(columns=["_date", "si_ratio_raw"], inplace=True, errors="ignore")
        return df

    def _create_proxy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Proxy: high-volume down days as SI indicator."""
        logger.info("ShortInterestFeatures: using volume/price proxy features")
        spy_ret = df["close"].pct_change()
        vol = df["volume"].astype(float)
        vol_ma = vol.rolling(20).mean()
        vol_ratio = vol / (vol_ma + 1e-8)

        # High volume on down days suggests short activity
        down_vol = vol_ratio * (spy_ret < 0).astype(float)
        si_proxy = down_vol.rolling(10).mean()

        df["si_ratio"] = si_proxy
        df["si_change"] = si_proxy.diff()
        df["si_days_to_cover"] = si_proxy * 3.0  # Rough scaling
        df["si_velocity"] = si_proxy.diff().rolling(5).mean()
        mu = si_proxy.rolling(60).mean()
        std = si_proxy.rolling(60).std()
        df["si_z_score"] = (si_proxy - mu) / (std + 1e-8)
        df["si_pct_float"] = si_proxy / (si_proxy.rolling(252).max() + 1e-8)
        df["si_change_5d"] = si_proxy.diff(5)

        price_rising = (spy_ret.rolling(5).sum() > 0.01).astype(float)
        si_high = (df["si_z_score"] > 1.0).astype(float)
        df["si_squeeze_signal"] = price_rising * si_high

        return df

    def analyze_current_short_interest(
        self, df_daily: pd.DataFrame
    ) -> Optional[Dict]:
        """Analyze current short interest state."""
        if df_daily.empty or "si_z_score" not in df_daily.columns:
            return None

        last = df_daily.iloc[-1]
        z = float(last.get("si_z_score", 0))

        if z > 2.0:
            regime = "VERY_HIGH_SI"
        elif z > 1.0:
            regime = "HIGH_SI"
        elif z < -1.0:
            regime = "LOW_SI"
        else:
            regime = "NORMAL"

        return {
            "regime": regime,
            "si_z_score": z,
            "squeeze_signal": float(last.get("si_squeeze_signal", 0)),
            "source": self._data_source,
        }

    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "si_ratio",
            "si_change",
            "si_days_to_cover",
            "si_velocity",
            "si_z_score",
            "si_pct_float",
            "si_change_5d",
            "si_squeeze_signal",
        ]
