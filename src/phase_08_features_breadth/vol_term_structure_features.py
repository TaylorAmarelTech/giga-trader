"""
Volatility Term Structure Features -- VIX vs VIX3M contango/backwardation signals.

VIX measures 30-day implied vol; VIX3M measures 3-month implied vol.
When VIX > VIX3M (backwardation), near-term stress exceeds longer-term
expectations -- historically a bearish signal. Contango is the normal
state and positive for equities.

Features (8, prefix vts_):
  vts_vix_vix3m_ratio       -- VIX / VIX3M ratio (>1 = backwardation)
  vts_contango              -- max(0, 1 - ratio) i.e. contango magnitude
  vts_backwardation         -- max(0, ratio - 1) i.e. backwardation magnitude
  vts_term_slope            -- VIX3M - VIX (positive = contango)
  vts_term_slope_z          -- 60-day z-score of vts_term_slope, clipped [-4, 4]
  vts_roll_yield_proxy      -- 5-day change in vts_vix_vix3m_ratio
  vts_ratio_momentum_5d     -- 5-day change in vts_vix_vix3m_ratio
  vts_regime                -- 1.0 CONTANGO (ratio<0.9), 0.0 NORMAL, -1.0 BACKWARDATION (ratio>1.05)
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VolTermStructureFeatures:
    """Compute VIX term structure features from VIX/VIX3M data or proxy."""

    REQUIRED_COLS = {"close"}

    def __init__(self) -> None:
        self._vix_data: Optional[pd.DataFrame] = None

    def download_vix_data(self, start_date, end_date) -> None:
        """Download ^VIX and ^VIX3M daily closes via yfinance."""
        try:
            import yfinance as yf
            tickers = yf.download(
                ["^VIX", "^VIX3M"],
                start=start_date, end=end_date, progress=False,
            )
            if tickers.empty:
                self._vix_data = None
                return

            adj = tickers["Close"] if "Close" in tickers.columns.get_level_values(0) else tickers["Adj Close"]
            df = pd.DataFrame({
                "vix": adj["^VIX"].values,
                "vix3m": adj["^VIX3M"].values,
            }, index=adj.index)
            df = df.dropna()
            df["date"] = pd.to_datetime(df.index.date)
            df = df.reset_index(drop=True)

            if df.empty:
                self._vix_data = None
            else:
                self._vix_data = df
                logger.info("VolTermStructureFeatures: downloaded %d rows of VIX data", len(df))
        except Exception as exc:
            logger.warning("VolTermStructureFeatures: download failed (%s), will use proxy", exc)
            self._vix_data = None

    def create_vol_term_structure_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """Add 8 vts_ features to df_daily."""
        df = df_daily.copy()
        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            logger.warning("VolTermStructureFeatures: missing columns %s, skipping", missing)
            return df

        close = df["close"].astype(float)
        returns = close.pct_change()

        # Determine source
        if self._vix_data is not None and not self._vix_data.empty and "date" in df.columns:
            vix_df = self._vix_data.copy()
            vix_df["date"] = pd.to_datetime(vix_df["date"])
            df["date"] = pd.to_datetime(df["date"])
            df = df.merge(vix_df[["date", "vix", "vix3m"]], on="date", how="left")
            for c in ["vix", "vix3m"]:
                df[c] = df[c].fillna(method="ffill", limit=3).fillna(0.0)
            vix_series = df["vix"]
            vix3m_series = df["vix3m"]
            df.drop(columns=["vix", "vix3m"], inplace=True)
        else:
            # Proxy: use rolling vol at different horizons
            vix_series = returns.rolling(20, min_periods=5).std() * np.sqrt(252) * 100
            vix3m_series = returns.rolling(60, min_periods=10).std() * np.sqrt(252) * 100
            vix_series = vix_series.fillna(20.0)
            vix3m_series = vix3m_series.fillna(20.0)

        # Compute features
        ratio = vix_series / (vix3m_series + 1e-10)
        df["vts_vix_vix3m_ratio"] = ratio.values
        df["vts_contango"] = np.maximum(0.0, 1.0 - ratio.values)
        df["vts_backwardation"] = np.maximum(0.0, ratio.values - 1.0)
        df["vts_term_slope"] = (vix3m_series - vix_series).values

        # Z-score of term slope
        slope = df["vts_term_slope"]
        rmean = slope.rolling(60, min_periods=10).mean()
        rstd = slope.rolling(60, min_periods=10).std()
        df["vts_term_slope_z"] = ((slope - rmean) / (rstd + 1e-10)).clip(-4, 4)

        # Roll yield proxy and momentum
        ratio_series = df["vts_vix_vix3m_ratio"]
        df["vts_roll_yield_proxy"] = ratio_series.diff(5)
        df["vts_ratio_momentum_5d"] = ratio_series.diff(5)

        # Regime
        r = df["vts_vix_vix3m_ratio"]
        df["vts_regime"] = np.where(r < 0.9, 1.0, np.where(r > 1.05, -1.0, 0.0))

        # Cleanup
        for col in self._all_feature_names():
            df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        n_features = sum(1 for c in df.columns if c.startswith("vts_"))
        logger.info("VolTermStructureFeatures: added %d features", n_features)
        return df

    def analyze_current_vol_term_structure(self, df_daily: pd.DataFrame) -> Optional[Dict]:
        """Return current vol term structure regime info."""
        if "vts_vix_vix3m_ratio" not in df_daily.columns or len(df_daily) < 2:
            return None
        last = df_daily.iloc[-1]
        ratio = float(last.get("vts_vix_vix3m_ratio", 1.0))
        if ratio < 0.9:
            regime = "CONTANGO"
        elif ratio > 1.05:
            regime = "BACKWARDATION"
        else:
            regime = "NORMAL"
        return {
            "vol_term_regime": regime,
            "vix_vix3m_ratio": round(ratio, 4),
            "term_slope": round(float(last.get("vts_term_slope", 0.0)), 4),
        }

    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "vts_vix_vix3m_ratio",
            "vts_contango",
            "vts_backwardation",
            "vts_term_slope",
            "vts_term_slope_z",
            "vts_roll_yield_proxy",
            "vts_ratio_momentum_5d",
            "vts_regime",
        ]
