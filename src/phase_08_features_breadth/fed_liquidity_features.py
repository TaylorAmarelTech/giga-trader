"""
GIGA TRADER - Fed Liquidity Features (Wave L8)
=================================================
Federal Reserve balance sheet and facility metrics.

Source: FRED API for WALCL (Fed Balance Sheet), RRPONTSYD (Reverse Repo),
        WRESBAL (Bank Reserves), WTREGEN (Treasury General Account).
Fallback: M2SL + DFF (already in FRED series) as liquidity proxy.

Features (8, prefix fedliq_):
  fedliq_balance_sheet_chg  -- Weekly change in Fed balance sheet (WALCL)
  fedliq_rrp_level          -- Reverse repo outstanding (normalized z-score)
  fedliq_rrp_change         -- Weekly change in reverse repo
  fedliq_net_liquidity      -- Fed BS - RRP - TGA (net liquidity injection)
  fedliq_liquidity_z        -- 60d z-score of net liquidity
  fedliq_tga_change         -- Weekly change in Treasury General Account
  fedliq_reserves_chg       -- Weekly change in bank reserves
  fedliq_liquidity_regime   -- 1.0 easing, -1.0 tightening, 0.0 neutral
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from src.core.feature_base import FeatureModuleBase

logger = logging.getLogger(__name__)


class FedLiquidityFeatures(FeatureModuleBase):
    """Compute Federal Reserve liquidity features."""
    FEATURE_NAMES = ["fedliq_balance_sheet_chg", "fedliq_rrp_level", "fedliq_rrp_change", "fedliq_net_liquidity", "fedliq_liquidity_z", "fedliq_tga_change", "fedliq_reserves_chg", "fedliq_liquidity_regime"]


    REQUIRED_COLS = {"close"}

    FRED_SERIES = {
        "WALCL": "Fed Total Assets (Balance Sheet)",
        "RRPONTSYD": "Reverse Repo (ON RRP)",
        "WRESBAL": "Reserve Balances",
        "WTREGEN": "Treasury General Account",
    }

    def __init__(self) -> None:
        self._liquidity_data: Optional[pd.DataFrame] = None
        self._data_source: str = "none"

    def download_liquidity_data(
        self, start_date, end_date
    ) -> Optional[pd.DataFrame]:
        """Download FRED liquidity data with graceful degradation."""
        api_key = os.environ.get("FRED_API_KEY", "")
        if not api_key:
            logger.info("FedLiquidityFeatures: no FRED_API_KEY, will use proxy")
            return None

        try:
            from fredapi import Fred
        except ImportError:
            logger.info("FedLiquidityFeatures: fredapi not installed, will use proxy")
            return None

        try:
            fred = Fred(api_key=api_key)
        except Exception as e:
            logger.warning(f"FedLiquidityFeatures: FRED init failed: {e}")
            return None

        results = {}
        for series_id, description in self.FRED_SERIES.items():
            try:
                data = fred.get_series(
                    series_id,
                    observation_start=str(start_date)[:10],
                    observation_end=str(end_date)[:10],
                )
                if data is not None and len(data) > 5:
                    results[series_id] = data
                    logger.info(f"  FRED {series_id}: {len(data)} observations")
                else:
                    logger.info(f"  FRED {series_id}: no data returned")
            except Exception as e:
                logger.warning(f"  FRED {series_id}: failed ({e})")

        if not results:
            logger.info("FedLiquidityFeatures: no FRED data available, will use proxy")
            return None

        self._liquidity_data = pd.DataFrame(results)
        self._liquidity_data.index = pd.to_datetime(self._liquidity_data.index)
        # Forward-fill weekly data to daily
        full_idx = pd.date_range(
            self._liquidity_data.index.min(),
            self._liquidity_data.index.max(),
            freq="B",
        )
        self._liquidity_data = self._liquidity_data.reindex(full_idx).ffill()
        self._data_source = "fred"
        logger.info(
            f"FedLiquidityFeatures: loaded {len(results)} series, "
            f"{len(self._liquidity_data)} daily rows"
        )
        return self._liquidity_data

    def create_fed_liquidity_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """Create Fed liquidity features."""
        df = df_daily.copy()

        if self._liquidity_data is not None and not self._liquidity_data.empty:
            df = self._create_full_features(df)
        else:
            df = self._create_proxy_features(df)

        # NaN/Inf cleanup
        for col in self._all_feature_names():
            if col in df.columns:
                df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        return df

    def _create_full_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features from real FRED data."""
        df["_date"] = pd.to_datetime(df["date"]).dt.normalize()
        liq = self._liquidity_data.copy()
        liq["_date"] = liq.index.normalize()

        df = df.merge(liq, on="_date", how="left")
        # Forward-fill any gaps
        for col in self.FRED_SERIES:
            if col in df.columns:
                df[col] = df[col].ffill(limit=5)
        df.drop(columns=["_date"], inplace=True)

        # Balance sheet change (weekly pct change)
        if "WALCL" in df.columns and df["WALCL"].notna().sum() > 10:
            walcl = df["WALCL"].fillna(method="ffill")
            df["fedliq_balance_sheet_chg"] = walcl.pct_change(5)  # ~weekly
        else:
            df["fedliq_balance_sheet_chg"] = 0.0

        # Reverse repo level and change
        if "RRPONTSYD" in df.columns and df["RRPONTSYD"].notna().sum() > 10:
            rrp = df["RRPONTSYD"].fillna(method="ffill")
            mu = rrp.rolling(60).mean()
            std = rrp.rolling(60).std()
            df["fedliq_rrp_level"] = (rrp - mu) / (std + 1e-8)
            df["fedliq_rrp_change"] = rrp.pct_change(5)
        else:
            df["fedliq_rrp_level"] = 0.0
            df["fedliq_rrp_change"] = 0.0

        # Net liquidity: Fed BS - RRP - TGA
        walcl = df.get("WALCL", pd.Series(0.0, index=df.index))
        rrp = df.get("RRPONTSYD", pd.Series(0.0, index=df.index))
        tga = df.get("WTREGEN", pd.Series(0.0, index=df.index))
        walcl = walcl.fillna(0.0)
        rrp = rrp.fillna(0.0)
        tga = tga.fillna(0.0)

        net_liq = walcl - rrp - tga
        if net_liq.abs().sum() > 0:
            df["fedliq_net_liquidity"] = net_liq.pct_change(5)
            mu_nl = net_liq.rolling(60).mean()
            std_nl = net_liq.rolling(60).std()
            df["fedliq_liquidity_z"] = (net_liq - mu_nl) / (std_nl + 1e-8)
        else:
            df["fedliq_net_liquidity"] = 0.0
            df["fedliq_liquidity_z"] = 0.0

        # TGA change
        if "WTREGEN" in df.columns and df["WTREGEN"].notna().sum() > 10:
            tga_series = df["WTREGEN"].fillna(method="ffill")
            df["fedliq_tga_change"] = tga_series.pct_change(5)
        else:
            df["fedliq_tga_change"] = 0.0

        # Reserves change
        if "WRESBAL" in df.columns and df["WRESBAL"].notna().sum() > 10:
            res = df["WRESBAL"].fillna(method="ffill")
            df["fedliq_reserves_chg"] = res.pct_change(5)
        else:
            df["fedliq_reserves_chg"] = 0.0

        # Liquidity regime
        z = df["fedliq_liquidity_z"]
        df["fedliq_liquidity_regime"] = np.where(
            z > 1.0, 1.0, np.where(z < -1.0, -1.0, 0.0)
        )

        # Drop temp FRED columns
        for col in self.FRED_SERIES:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        return df

    def _create_proxy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Proxy: use SPY volume trend as liquidity indicator."""
        logger.info("FedLiquidityFeatures: using volume-based proxy features")
        vol = df["volume"].astype(float)
        vol_ma = vol.rolling(20).mean()
        vol_ratio = vol / (vol_ma + 1e-8)

        df["fedliq_balance_sheet_chg"] = 0.0
        df["fedliq_rrp_level"] = 0.0
        df["fedliq_rrp_change"] = 0.0
        df["fedliq_net_liquidity"] = (vol_ratio - 1.0) * 0.1  # Volume expansion proxy
        mu = df["fedliq_net_liquidity"].rolling(60).mean()
        std = df["fedliq_net_liquidity"].rolling(60).std()
        df["fedliq_liquidity_z"] = (df["fedliq_net_liquidity"] - mu) / (std + 1e-8)
        df["fedliq_tga_change"] = 0.0
        df["fedliq_reserves_chg"] = 0.0
        df["fedliq_liquidity_regime"] = 0.0

        return df

    def analyze_current_liquidity(self, df_daily: pd.DataFrame) -> Optional[Dict]:
        """Analyze current Fed liquidity regime."""
        if df_daily.empty or "fedliq_liquidity_z" not in df_daily.columns:
            return None

        last = df_daily.iloc[-1]
        z = float(last.get("fedliq_liquidity_z", 0))
        regime_val = float(last.get("fedliq_liquidity_regime", 0))

        if regime_val > 0:
            regime = "EASING"
        elif regime_val < 0:
            regime = "TIGHTENING"
        else:
            regime = "NEUTRAL"

        return {
            "regime": regime,
            "liquidity_z": z,
            "balance_sheet_chg": float(last.get("fedliq_balance_sheet_chg", 0)),
            "rrp_level": float(last.get("fedliq_rrp_level", 0)),
            "source": self._data_source,
        }

    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "fedliq_balance_sheet_chg",
            "fedliq_rrp_level",
            "fedliq_rrp_change",
            "fedliq_net_liquidity",
            "fedliq_liquidity_z",
            "fedliq_tga_change",
            "fedliq_reserves_chg",
            "fedliq_liquidity_regime",
        ]
