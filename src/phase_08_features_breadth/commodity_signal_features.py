"""
GIGA TRADER - Commodity Signal Features (Wave L6)
====================================================
Cross-commodity regime signals: copper, oil, silver, gold ratios.

Source: yfinance for copper (HG=F), silver (SI=F), oil (CL=F).
Fallback: USO/GLD ETFs already downloaded, derive proxy signals.

Features (10, prefix cmdty_):
  cmdty_copper_momentum       -- 20d copper return (Dr. Copper signal)
  cmdty_copper_gold_ratio     -- Copper/gold ratio (risk appetite)
  cmdty_oil_copper_div        -- Oil-copper divergence (supply vs demand)
  cmdty_oil_vol               -- 20d oil volatility
  cmdty_gold_silver_ratio     -- Gold/silver ratio (safe haven vs industrial)
  cmdty_dr_copper_z           -- 60d z-score of copper momentum
  cmdty_energy_metals_ratio   -- Energy vs metals momentum divergence
  cmdty_industrial_signal     -- Composite industrial demand signal
  cmdty_safe_haven_flow       -- Gold momentum relative to copper (flight to safety)
  cmdty_commodity_regime      -- 1.0 expansion, -1.0 contraction, 0.0 neutral
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from src.core.feature_base import FeatureModuleBase

logger = logging.getLogger(__name__)


class CommoditySignalFeatures(FeatureModuleBase):
    """Compute cross-commodity signals for equity prediction."""
    FEATURE_NAMES = ["cmdty_copper_momentum", "cmdty_copper_gold_ratio", "cmdty_oil_copper_div", "cmdty_oil_vol", "cmdty_gold_silver_ratio", "cmdty_dr_copper_z", "cmdty_energy_metals_ratio", "cmdty_industrial_signal", "cmdty_safe_haven_flow", "cmdty_commodity_regime"]


    REQUIRED_COLS = {"close"}

    # Futures symbols with ETF fallbacks
    COMMODITY_MAP = {
        "copper": {"futures": ["HG=F"], "etf": ["COPX", "XLB"]},
        "oil": {"futures": ["CL=F"], "etf": ["USO"]},
        "gold": {"futures": ["GC=F"], "etf": ["GLD"]},
        "silver": {"futures": ["SI=F"], "etf": ["SLV"]},
    }

    def __init__(self) -> None:
        self._commodity_data: Optional[pd.DataFrame] = None
        self._data_source: str = "none"
        self._available: Dict[str, bool] = {}

    def download_commodity_data(
        self, start_date: str, end_date: str
    ) -> Optional[pd.DataFrame]:
        """Download commodity data with futures → ETF fallback chain."""
        try:
            import yfinance as yf
        except ImportError:
            logger.info("CommoditySignalFeatures: yfinance not installed, will use proxy")
            return None

        results = {}
        for commodity, tickers in self.COMMODITY_MAP.items():
            downloaded = False
            # Try futures first, then ETFs
            for ticker in tickers["futures"] + tickers["etf"]:
                try:
                    data = yf.download(
                        ticker, start=start_date, end=end_date,
                        progress=False, auto_adjust=True
                    )
                    if data is not None and not data.empty and len(data) > 20:
                        if hasattr(data.columns, "levels"):
                            data.columns = data.columns.get_level_values(0)
                        results[commodity] = data["Close"].copy()
                        self._available[commodity] = True
                        downloaded = True
                        logger.info(f"  {commodity}: {ticker} ({len(data)} rows)")
                        break
                except Exception as e:
                    logger.debug(f"  {commodity}: {ticker} failed: {e}")

            if not downloaded:
                self._available[commodity] = False
                logger.info(f"  {commodity}: no data available")

        if not results:
            logger.info("CommoditySignalFeatures: no commodity data, will use proxy")
            return None

        self._commodity_data = pd.DataFrame(results)
        self._commodity_data.index = pd.to_datetime(self._commodity_data.index)
        self._data_source = "yfinance"
        n_avail = sum(1 for v in self._available.values() if v)
        logger.info(f"CommoditySignalFeatures: downloaded {n_avail}/4 commodities")
        return self._commodity_data

    def create_commodity_signal_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """Create commodity signal features."""
        df = df_daily.copy()

        if self._commodity_data is not None and not self._commodity_data.empty:
            df = self._create_full_features(df)
        else:
            df = self._create_proxy_features(df)

        # NaN/Inf cleanup
        for col in self._all_feature_names():
            if col in df.columns:
                df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        return df

    def _create_full_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features from real commodity data."""
        df["_date"] = pd.to_datetime(df["date"]).dt.normalize()
        cmdty = self._commodity_data.copy()
        cmdty["_date"] = cmdty.index.normalize()

        df = df.merge(cmdty, on="_date", how="left")
        # Forward-fill commodity data
        for col in ["copper", "oil", "gold", "silver"]:
            if col in df.columns:
                df[col] = df[col].ffill(limit=3)
        df.drop(columns=["_date"], inplace=True)

        # Copper momentum (Dr. Copper signal)
        if "copper" in df.columns and df["copper"].notna().sum() > 20:
            cu = df["copper"].fillna(method="ffill")
            df["cmdty_copper_momentum"] = cu.pct_change(20)
            mu = cu.pct_change(20).rolling(60).mean()
            std = cu.pct_change(20).rolling(60).std()
            df["cmdty_dr_copper_z"] = (cu.pct_change(20) - mu) / (std + 1e-8)
        else:
            df["cmdty_copper_momentum"] = 0.0
            df["cmdty_dr_copper_z"] = 0.0

        # Copper/Gold ratio
        if (
            "copper" in df.columns
            and "gold" in df.columns
            and df["copper"].notna().sum() > 20
            and df["gold"].notna().sum() > 20
        ):
            cu = df["copper"].fillna(method="ffill")
            au = df["gold"].fillna(method="ffill")
            ratio = cu / (au + 1e-8)
            df["cmdty_copper_gold_ratio"] = (ratio - ratio.rolling(60).mean()) / (
                ratio.rolling(60).std() + 1e-8
            )
        else:
            df["cmdty_copper_gold_ratio"] = 0.0

        # Oil-copper divergence
        if (
            "oil" in df.columns
            and "copper" in df.columns
            and df["oil"].notna().sum() > 20
            and df["copper"].notna().sum() > 20
        ):
            oil_mom = df["oil"].fillna(method="ffill").pct_change(20)
            cu_mom = df["copper"].fillna(method="ffill").pct_change(20)
            df["cmdty_oil_copper_div"] = oil_mom - cu_mom
        else:
            df["cmdty_oil_copper_div"] = 0.0

        # Oil volatility
        if "oil" in df.columns and df["oil"].notna().sum() > 20:
            df["cmdty_oil_vol"] = (
                df["oil"].fillna(method="ffill").pct_change().rolling(20).std()
                * np.sqrt(252)
            )
        else:
            df["cmdty_oil_vol"] = 0.0

        # Gold/Silver ratio
        if (
            "gold" in df.columns
            and "silver" in df.columns
            and df["gold"].notna().sum() > 20
            and df["silver"].notna().sum() > 20
        ):
            au = df["gold"].fillna(method="ffill")
            ag = df["silver"].fillna(method="ffill")
            gs_ratio = au / (ag + 1e-8)
            df["cmdty_gold_silver_ratio"] = (
                gs_ratio - gs_ratio.rolling(60).mean()
            ) / (gs_ratio.rolling(60).std() + 1e-8)
        else:
            df["cmdty_gold_silver_ratio"] = 0.0

        # Energy vs metals momentum divergence
        oil_mom = 0.0
        cu_mom_val = df["cmdty_copper_momentum"]
        if "oil" in df.columns and df["oil"].notna().sum() > 20:
            oil_mom = df["oil"].fillna(method="ffill").pct_change(20)
        df["cmdty_energy_metals_ratio"] = oil_mom - cu_mom_val

        # Industrial signal (copper + oil momentum average)
        df["cmdty_industrial_signal"] = (
            df["cmdty_copper_momentum"] + (oil_mom if isinstance(oil_mom, pd.Series) else 0.0)
        ) / 2.0

        # Safe haven flow (gold vs copper)
        if "gold" in df.columns and df["gold"].notna().sum() > 20:
            gold_mom = df["gold"].fillna(method="ffill").pct_change(20)
            df["cmdty_safe_haven_flow"] = gold_mom - df["cmdty_copper_momentum"]
        else:
            df["cmdty_safe_haven_flow"] = -df["cmdty_copper_momentum"]

        # Commodity regime
        industrial = df["cmdty_industrial_signal"]
        df["cmdty_commodity_regime"] = np.where(
            industrial > 0.02, 1.0, np.where(industrial < -0.02, -1.0, 0.0)
        )

        # Clean up temp columns
        for col in ["copper", "oil", "gold", "silver"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)

        return df

    def _create_proxy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Proxy: derive commodity-like signals from SPY volume/volatility."""
        logger.info("CommoditySignalFeatures: using SPY-based proxy features")
        spy_ret = df["close"].pct_change()
        spy_vol = spy_ret.rolling(20).std()

        df["cmdty_copper_momentum"] = spy_ret.rolling(20).sum() * 0.5
        df["cmdty_copper_gold_ratio"] = 0.0
        df["cmdty_oil_copper_div"] = 0.0
        df["cmdty_oil_vol"] = spy_vol * np.sqrt(252)
        df["cmdty_gold_silver_ratio"] = 0.0
        df["cmdty_dr_copper_z"] = 0.0
        df["cmdty_energy_metals_ratio"] = 0.0
        df["cmdty_industrial_signal"] = spy_ret.rolling(20).sum() * 0.3
        df["cmdty_safe_haven_flow"] = 0.0
        df["cmdty_commodity_regime"] = 0.0

        return df

    def analyze_current_commodity(self, df_daily: pd.DataFrame) -> Optional[Dict]:
        """Analyze current commodity regime."""
        if df_daily.empty or "cmdty_commodity_regime" not in df_daily.columns:
            return None

        last = df_daily.iloc[-1]
        regime_val = float(last.get("cmdty_commodity_regime", 0))

        if regime_val > 0:
            regime = "EXPANSION"
        elif regime_val < 0:
            regime = "CONTRACTION"
        else:
            regime = "NEUTRAL"

        return {
            "regime": regime,
            "dr_copper_z": float(last.get("cmdty_dr_copper_z", 0)),
            "copper_gold_ratio": float(last.get("cmdty_copper_gold_ratio", 0)),
            "safe_haven_flow": float(last.get("cmdty_safe_haven_flow", 0)),
            "source": self._data_source,
        }

    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "cmdty_copper_momentum",
            "cmdty_copper_gold_ratio",
            "cmdty_oil_copper_div",
            "cmdty_oil_vol",
            "cmdty_gold_silver_ratio",
            "cmdty_dr_copper_z",
            "cmdty_energy_metals_ratio",
            "cmdty_industrial_signal",
            "cmdty_safe_haven_flow",
            "cmdty_commodity_regime",
        ]
