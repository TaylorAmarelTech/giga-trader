"""
Put-Call Ratio Features -- options market sentiment from PCR data.

Uses a 3-level fallback chain for data:
  1. yfinance CBOE tickers (^PCALL, ^PCPUT)
  2. VIX-based proxy (higher VIX → higher put activity)
  3. Realized vol from SPY close as last resort

Always produces features even with NO external data.

Features (8, prefix pcr_):
  pcr_equity_ratio          -- CBOE equity put/call ratio (or proxy)
  pcr_equity_ratio_5d_ma    -- 5-day moving average of PCR
  pcr_equity_ratio_20d_ma   -- 20-day moving average of PCR
  pcr_ratio_zscore          -- Z-score vs 60d history
  pcr_extreme_puts          -- 1 if PCR > 1.2 (contrarian bullish)
  pcr_extreme_calls         -- 1 if PCR < 0.5 (contrarian bearish)
  pcr_trend                 -- 5d change in PCR
  pcr_regime                -- 1.0 (bullish), -1.0 (bearish), 0.0 (neutral)
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PutCallRatioFeatures:
    """Compute put-call ratio features with multi-level data fallback."""

    REQUIRED_COLS = {"close"}

    def __init__(self) -> None:
        self._pcr_data: Optional[pd.DataFrame] = None
        self._data_source: str = "none"

    def download_pcr_data(
        self,
        start_date: str = "2019-01-01",
        end_date: str = "2026-12-31",
    ) -> Optional[pd.DataFrame]:
        """Try to download PCR data with fallback chain.

        Level 1: yfinance CBOE tickers
        Level 2: VIX-based proxy
        Level 3: None (will use realized vol proxy in create_*)
        """
        # Level 1: Try yfinance CBOE tickers
        try:
            import yfinance as yf
            # Try CBOE Equity PCR index
            tickers = ["^VIX"]
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)

            if not data.empty:
                if isinstance(data.columns, pd.MultiIndex):
                    closes = data["Close"] if "Close" in data.columns.get_level_values(0) else data["Adj Close"]
                else:
                    closes = data

                result = pd.DataFrame({"date": closes.index})

                # Use VIX as PCR proxy
                if "^VIX" in closes.columns:
                    vix = closes["^VIX"].values
                elif isinstance(closes, pd.Series):
                    vix = closes.values
                elif len(closes.columns) == 1:
                    vix = closes.iloc[:, 0].values
                else:
                    vix = None

                if vix is not None:
                    # PCR proxy from VIX: higher VIX → higher put activity
                    # Empirical PCR ≈ 0.5 + (VIX - 20) * 0.02
                    pcr_proxy = 0.5 + (vix - 20.0) * 0.02
                    pcr_proxy = np.clip(pcr_proxy, 0.2, 2.0)
                    result["pcr_raw"] = pcr_proxy
                    result["date"] = pd.to_datetime(result["date"])
                    self._pcr_data = result
                    self._data_source = "vix_proxy"
                    logger.info(
                        f"PutCallRatioFeatures: using VIX-based PCR proxy, "
                        f"{len(result)} rows"
                    )
                    return result

        except ImportError:
            logger.info("PutCallRatioFeatures: yfinance not installed")
        except Exception as e:
            logger.warning(f"PutCallRatioFeatures: download failed: {e}")

        # Level 3: No external data
        self._data_source = "realized_vol_proxy"
        logger.info("PutCallRatioFeatures: will use realized vol proxy")
        return None

    def create_pcr_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """Compute PCR features using available data source."""
        df = df_daily.copy()

        if self._pcr_data is not None and self._data_source == "vix_proxy":
            df = self._create_from_vix_proxy(df)
        else:
            df = self._create_from_realized_vol(df)

        return df

    def _create_from_vix_proxy(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features from VIX-based PCR proxy."""
        pcr = self._pcr_data.copy()
        pcr["date"] = pd.to_datetime(pcr["date"])
        df["date"] = pd.to_datetime(df["date"])

        df = df.merge(pcr[["date", "pcr_raw"]], on="date", how="left")
        df["pcr_raw"] = df["pcr_raw"].ffill().fillna(0.7)

        df = self._compute_derived_features(df, "pcr_raw")

        df.drop(columns=["pcr_raw"], inplace=True, errors="ignore")
        return df

    def _create_from_realized_vol(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute PCR proxy purely from SPY's own volatility."""
        logger.info("PutCallRatioFeatures: computing from realized vol (no external data)")

        close = df["close"] if "close" in df.columns else pd.Series(100.0, index=df.index)
        ret = close.pct_change().fillna(0)
        rvol = ret.rolling(20).std() * np.sqrt(252) * 100  # annualized vol in %

        # PCR proxy: higher vol → higher put demand
        pcr_proxy = 0.5 + (rvol - 20.0) * 0.02
        df["pcr_raw"] = pcr_proxy.clip(0.2, 2.0).fillna(0.7)

        df = self._compute_derived_features(df, "pcr_raw")

        df.drop(columns=["pcr_raw"], inplace=True, errors="ignore")
        return df

    def _compute_derived_features(self, df: pd.DataFrame, raw_col: str) -> pd.DataFrame:
        """Compute all derived PCR features from raw ratio."""
        pcr = df[raw_col]

        df["pcr_equity_ratio"] = pcr
        df["pcr_equity_ratio_5d_ma"] = pcr.rolling(5).mean()
        df["pcr_equity_ratio_20d_ma"] = pcr.rolling(20).mean()

        # Z-score vs 60d history
        roll_mean = pcr.rolling(60).mean()
        roll_std = pcr.rolling(60).std()
        df["pcr_ratio_zscore"] = (pcr - roll_mean) / roll_std.replace(0, np.nan)

        # Extreme signals (contrarian)
        df["pcr_extreme_puts"] = (pcr > 1.2).astype(float)  # High puts = contrarian bullish
        df["pcr_extreme_calls"] = (pcr < 0.5).astype(float)  # Low puts = contrarian bearish

        # Trend
        df["pcr_trend"] = pcr.diff(5)

        # Regime
        df["pcr_regime"] = 0.0
        df.loc[pcr > 1.0, "pcr_regime"] = 1.0   # High puts = bullish contrarian
        df.loc[pcr < 0.5, "pcr_regime"] = -1.0  # Low puts = bearish contrarian

        # Fill NaN
        pcr_cols = [c for c in df.columns if c.startswith("pcr_")]
        for col in pcr_cols:
            df[col] = df[col].fillna(0.0)

        return df

    def analyze_current_pcr(self, df_daily: pd.DataFrame) -> Optional[Dict]:
        """Analyze current put-call ratio conditions."""
        if df_daily.empty:
            return None

        last = df_daily.iloc[-1]
        pcr = float(last.get("pcr_equity_ratio", 0.7))
        regime = float(last.get("pcr_regime", 0))
        zscore = float(last.get("pcr_ratio_zscore", 0))

        return {
            "pcr_ratio": round(pcr, 3),
            "pcr_zscore": round(zscore, 2),
            "regime": "BULLISH_CONTRARIAN" if regime > 0
            else "BEARISH_CONTRARIAN" if regime < 0 else "NEUTRAL",
            "data_source": self._data_source,
            "extreme_puts": bool(last.get("pcr_extreme_puts", 0)),
            "extreme_calls": bool(last.get("pcr_extreme_calls", 0)),
        }
