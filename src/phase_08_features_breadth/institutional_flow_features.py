"""
GIGA TRADER - Institutional Flow Features (Wave L4)
======================================================
SEC EDGAR 13F-based institutional positioning signals.

Source: SEC EDGAR API (free, no key). 13F filings quarterly.
Fallback: SPY/IVV/VOO ETF volume ratios as institutional flow proxy.
DEFAULT OFF: quarterly data, 45-day delay = very sparse signal.

Features (8, prefix inst_):
  inst_net_flow             -- Net institutional flow (proxy: ETF volume divergence)
  inst_concentration        -- Ownership concentration proxy
  inst_momentum             -- Quarter-over-quarter flow change
  inst_breadth              -- Fraction of volume going to SPY-linked ETFs
  inst_flow_z               -- Z-score of net flow
  inst_new_positions        -- Volume anomaly as new position proxy
  inst_position_changes     -- Magnitude of volume changes
  inst_top10_weight_change  -- Relative weight shift of large-cap ETFs
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from src.core.feature_base import FeatureModuleBase

logger = logging.getLogger(__name__)


class InstitutionalFlowFeatures(FeatureModuleBase):
    """Compute institutional flow features from ETF volume proxy."""
    FEATURE_NAMES = ["inst_net_flow", "inst_concentration", "inst_momentum", "inst_breadth", "inst_flow_z", "inst_new_positions", "inst_position_changes", "inst_top10_weight_change"]


    REQUIRED_COLS = {"close", "volume"}

    # SPY-linked ETFs for institutional flow proxy
    ETF_TICKERS = ["SPY", "IVV", "VOO"]

    def __init__(self) -> None:
        self._etf_data: Optional[pd.DataFrame] = None
        self._data_source: str = "none"

    def download_institutional_data(
        self, start_date, end_date
    ) -> Optional[pd.DataFrame]:
        """Download ETF volume data as institutional flow proxy."""
        try:
            import yfinance as yf
        except ImportError:
            logger.info("InstitutionalFlowFeatures: yfinance not installed")
            return None

        try:
            data = yf.download(
                self.ETF_TICKERS,
                start=str(start_date)[:10],
                end=str(end_date)[:10],
                progress=False,
                auto_adjust=True,
            )
            if data is None or data.empty:
                logger.info("InstitutionalFlowFeatures: no ETF data")
                return None

            # Extract volume for each ETF
            results = {}
            if hasattr(data.columns, "levels") and data.columns.nlevels > 1:
                for ticker in self.ETF_TICKERS:
                    try:
                        results[f"{ticker}_vol"] = data["Volume"][ticker]
                        results[f"{ticker}_close"] = data["Close"][ticker]
                    except (KeyError, TypeError):
                        pass
            else:
                # Single ticker case
                results["SPY_vol"] = data.get("Volume", pd.Series())
                results["SPY_close"] = data.get("Close", pd.Series())

            if not results:
                return None

            self._etf_data = pd.DataFrame(results)
            self._etf_data.index = pd.to_datetime(self._etf_data.index)
            self._data_source = "etf_volume"
            logger.info(f"InstitutionalFlowFeatures: loaded ETF data ({len(self._etf_data)} rows)")
            return self._etf_data

        except Exception as e:
            logger.warning(f"InstitutionalFlowFeatures: download failed: {e}")
            return None

    def create_institutional_flow_features(
        self, df_daily: pd.DataFrame
    ) -> pd.DataFrame:
        """Create institutional flow features."""
        df = df_daily.copy()

        if self._etf_data is not None and not self._etf_data.empty:
            df = self._create_full_features(df)
        else:
            df = self._create_proxy_features(df)

        for col in self._all_feature_names():
            if col in df.columns:
                df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        return df

    def _create_full_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute from ETF volume data."""
        df["_date"] = pd.to_datetime(df["date"]).dt.normalize()
        etf = self._etf_data.copy()
        etf["_date"] = etf.index.normalize()

        df = df.merge(etf, on="_date", how="left")
        for col in etf.columns:
            if col != "_date" and col in df.columns:
                df[col] = df[col].ffill(limit=3)

        # Total ETF volume as institutional flow proxy
        vol_cols = [c for c in df.columns if c.endswith("_vol")]
        if vol_cols:
            total_vol = df[vol_cols].sum(axis=1).fillna(0)
            vol_ma = total_vol.rolling(20).mean()

            df["inst_net_flow"] = (total_vol - vol_ma) / (vol_ma + 1e-8)
            df["inst_momentum"] = total_vol.pct_change(20)
            df["inst_breadth"] = total_vol / (df["volume"].astype(float) + 1e-8)

            # Volume concentration (HHI of ETF volumes)
            if len(vol_cols) > 1:
                shares = df[vol_cols].div(total_vol + 1e-8, axis=0)
                df["inst_concentration"] = (shares ** 2).sum(axis=1)
            else:
                df["inst_concentration"] = 1.0

            mu = df["inst_net_flow"].rolling(60).mean()
            std = df["inst_net_flow"].rolling(60).std()
            df["inst_flow_z"] = (df["inst_net_flow"] - mu) / (std + 1e-8)

            vol_change = total_vol.diff().abs()
            df["inst_new_positions"] = vol_change / (vol_ma + 1e-8)
            df["inst_position_changes"] = vol_change.rolling(5).mean() / (vol_ma + 1e-8)

            # Top ETF weight change
            if "SPY_vol" in df.columns:
                spy_share = df["SPY_vol"].fillna(0) / (total_vol + 1e-8)
                df["inst_top10_weight_change"] = spy_share.diff(20)
            else:
                df["inst_top10_weight_change"] = 0.0
        else:
            return self._create_proxy_features(df)

        # Clean up temp columns
        drop_cols = [c for c in df.columns if c.endswith("_vol") or c.endswith("_close")]
        drop_cols.append("_date")
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

        return df

    def _create_proxy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Proxy: use SPY volume characteristics."""
        logger.info("InstitutionalFlowFeatures: using SPY volume proxy")
        vol = df["volume"].astype(float)
        vol_ma = vol.rolling(20).mean()
        vol_ratio = vol / (vol_ma + 1e-8)

        df["inst_net_flow"] = vol_ratio - 1.0
        df["inst_concentration"] = 0.5
        df["inst_momentum"] = vol.pct_change(20)
        df["inst_breadth"] = 0.5
        mu = df["inst_net_flow"].rolling(60).mean()
        std = df["inst_net_flow"].rolling(60).std()
        df["inst_flow_z"] = (df["inst_net_flow"] - mu) / (std + 1e-8)
        df["inst_new_positions"] = vol.diff().abs() / (vol_ma + 1e-8)
        df["inst_position_changes"] = df["inst_new_positions"].rolling(5).mean()
        df["inst_top10_weight_change"] = 0.0

        return df

    def analyze_current_flow(
        self, df_daily: pd.DataFrame
    ) -> Optional[Dict]:
        """Analyze current institutional flow."""
        if df_daily.empty or "inst_flow_z" not in df_daily.columns:
            return None

        last = df_daily.iloc[-1]
        z = float(last.get("inst_flow_z", 0))

        if z > 1.5:
            regime = "HEAVY_INFLOW"
        elif z > 0.5:
            regime = "MILD_INFLOW"
        elif z < -1.5:
            regime = "HEAVY_OUTFLOW"
        elif z < -0.5:
            regime = "MILD_OUTFLOW"
        else:
            regime = "NEUTRAL"

        return {
            "regime": regime,
            "flow_z": z,
            "net_flow": float(last.get("inst_net_flow", 0)),
            "source": self._data_source,
        }

    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "inst_net_flow",
            "inst_concentration",
            "inst_momentum",
            "inst_breadth",
            "inst_flow_z",
            "inst_new_positions",
            "inst_position_changes",
            "inst_top10_weight_change",
        ]
