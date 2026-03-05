"""
GIGA TRADER - Dollar Index Features (Wave L3)
===============================================
US Dollar strength signals for equity market prediction.

Source: yfinance DX-Y.NYB (Dollar Index futures).
Fallback: UUP ETF or inverse EUR/USD proxy.

Features (8, prefix dxy_):
  dxy_level            -- Dollar index level (normalized)
  dxy_return           -- Daily dollar index return
  dxy_momentum_20d     -- 20d momentum (cumulative return)
  dxy_vol_20d          -- 20d realized volatility
  dxy_z_score          -- 60d z-score of level
  dxy_spy_corr         -- Rolling 20d correlation with SPY
  dxy_roc_5d           -- 5-day rate of change
  dxy_regime           -- 1.0 strong dollar, -1.0 weak, 0.0 neutral
"""

import logging
import os
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from src.core.feature_base import FeatureModuleBase

logger = logging.getLogger(__name__)


class DollarIndexFeatures(FeatureModuleBase):
    """Compute US Dollar index features for equity prediction."""
    FEATURE_NAMES = ["dxy_level", "dxy_return", "dxy_momentum_20d", "dxy_vol_20d", "dxy_z_score", "dxy_spy_corr", "dxy_roc_5d", "dxy_regime"]


    REQUIRED_COLS = {"close"}

    def __init__(self) -> None:
        self._dollar_data: Optional[pd.DataFrame] = None
        self._data_source: str = "none"

    def download_dollar_data(
        self, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Download dollar index data with multi-level fallback.

        Level 1: DX-Y.NYB (Dollar Index futures) via yfinance
        Level 2: UUP (Dollar Bull ETF) via yfinance
        Level 3: None (will use SPY-based proxy)
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.info("DollarIndexFeatures: yfinance not installed, will use proxy")
            return None

        # Level 1: Dollar Index futures
        for ticker in ["DX-Y.NYB", "DX=F"]:
            try:
                data = yf.download(
                    ticker, start=start_date, end=end_date,
                    progress=False, auto_adjust=True
                )
                if data is not None and not data.empty and len(data) > 20:
                    if hasattr(data.columns, "levels"):
                        data.columns = data.columns.get_level_values(0)
                    self._dollar_data = data[["Close"]].rename(columns={"Close": "dxy_close"})
                    self._dollar_data.index = pd.to_datetime(self._dollar_data.index)
                    self._data_source = "dxy_futures"
                    logger.info(f"DollarIndexFeatures: downloaded {ticker} ({len(data)} rows)")
                    return self._dollar_data
            except Exception as e:
                logger.debug(f"DollarIndexFeatures: {ticker} failed: {e}")

        # Level 2: UUP ETF
        try:
            data = yf.download(
                "UUP", start=start_date, end=end_date,
                progress=False, auto_adjust=True
            )
            if data is not None and not data.empty and len(data) > 20:
                if hasattr(data.columns, "levels"):
                    data.columns = data.columns.get_level_values(0)
                self._dollar_data = data[["Close"]].rename(columns={"Close": "dxy_close"})
                self._dollar_data.index = pd.to_datetime(self._dollar_data.index)
                self._data_source = "uup_etf"
                logger.info(f"DollarIndexFeatures: using UUP ETF proxy ({len(data)} rows)")
                return self._dollar_data
        except Exception as e:
            logger.warning(f"DollarIndexFeatures: UUP download failed: {e}")

        logger.info("DollarIndexFeatures: no dollar data available, will use proxy")
        return None

    def create_dollar_index_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """Create dollar index features."""
        df = df_daily.copy()

        if self._dollar_data is not None and not self._dollar_data.empty:
            df = self._create_full_features(df)
        else:
            df = self._create_proxy_features(df)

        # NaN/Inf cleanup
        for col in self._all_feature_names():
            if col in df.columns:
                df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        return df

    def _create_full_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features from real dollar index data."""
        # Merge dollar data by date
        df["_date"] = pd.to_datetime(df["date"]).dt.normalize()
        dxy = self._dollar_data.copy()
        dxy["_date"] = dxy.index.normalize()

        df = df.merge(dxy[["_date", "dxy_close"]], on="_date", how="left")
        df["dxy_close"] = df["dxy_close"].ffill(limit=3)
        df.drop(columns=["_date"], inplace=True)

        # If merge produced no data, fall back to proxy
        if df["dxy_close"].isna().sum() > len(df) * 0.5:
            df.drop(columns=["dxy_close"], inplace=True)
            return self._create_proxy_features(df)

        dxy_close = df["dxy_close"].fillna(method="ffill").fillna(df["dxy_close"].mean())

        # Feature engineering
        df["dxy_level"] = (dxy_close - dxy_close.rolling(60).mean()) / (
            dxy_close.rolling(60).std() + 1e-8
        )
        df["dxy_return"] = dxy_close.pct_change()
        df["dxy_momentum_20d"] = dxy_close.pct_change(20)
        df["dxy_vol_20d"] = dxy_close.pct_change().rolling(20).std() * np.sqrt(252)
        df["dxy_z_score"] = (dxy_close - dxy_close.rolling(60).mean()) / (
            dxy_close.rolling(60).std() + 1e-8
        )
        # SPY-DXY correlation
        spy_ret = df["close"].pct_change()
        dxy_ret = dxy_close.pct_change()
        df["dxy_spy_corr"] = spy_ret.rolling(20).corr(dxy_ret)
        df["dxy_roc_5d"] = dxy_close.pct_change(5)

        # Regime: strong dollar (z>1) = -1 for equities, weak dollar (z<-1) = +1
        z = df["dxy_z_score"]
        df["dxy_regime"] = np.where(z > 1.0, 1.0, np.where(z < -1.0, -1.0, 0.0))

        # Drop temp column
        if "dxy_close" in df.columns:
            df.drop(columns=["dxy_close"], inplace=True)

        return df

    def _create_proxy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Proxy: derive dollar-like features from SPY inverse momentum."""
        logger.info("DollarIndexFeatures: using SPY-inverse proxy features")
        spy_ret = df["close"].pct_change()

        # Dollar tends to move inversely to equities
        df["dxy_level"] = 0.0
        df["dxy_return"] = -spy_ret  # Inverse proxy
        df["dxy_momentum_20d"] = -spy_ret.rolling(20).sum()
        df["dxy_vol_20d"] = spy_ret.rolling(20).std() * np.sqrt(252)
        df["dxy_z_score"] = 0.0
        df["dxy_spy_corr"] = -0.5  # Typical negative correlation
        df["dxy_roc_5d"] = -spy_ret.rolling(5).sum()
        df["dxy_regime"] = 0.0

        return df

    def analyze_current_dollar(self, df_daily: pd.DataFrame) -> Optional[Dict]:
        """Analyze current dollar regime."""
        if df_daily.empty or "dxy_z_score" not in df_daily.columns:
            return None

        last = df_daily.iloc[-1]
        z = float(last.get("dxy_z_score", 0))
        regime_val = float(last.get("dxy_regime", 0))

        if regime_val > 0:
            regime = "STRONG_DOLLAR"
        elif regime_val < 0:
            regime = "WEAK_DOLLAR"
        else:
            regime = "NEUTRAL"

        return {
            "regime": regime,
            "z_score": z,
            "momentum_20d": float(last.get("dxy_momentum_20d", 0)),
            "spy_corr": float(last.get("dxy_spy_corr", 0)),
            "source": self._data_source,
        }

    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "dxy_level",
            "dxy_return",
            "dxy_momentum_20d",
            "dxy_vol_20d",
            "dxy_z_score",
            "dxy_spy_corr",
            "dxy_roc_5d",
            "dxy_regime",
        ]
