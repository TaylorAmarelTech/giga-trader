"""
Fama-French Factor Exposure Features -- rolling factor betas for SPY.

Approximates factor exposures using ETF proxies:
  - Market: SPY itself
  - SMB (size): IWM - SPY spread
  - HML (value): IWD - IWF spread (value minus growth)
  - Momentum: MTUM ETF

Falls back to zero-filled features if ETF data is unavailable.

Features (8, prefix ff_):
  ff_mkt_beta_60d           -- Rolling 60d beta to market excess return
  ff_smb_beta_60d           -- Rolling 60d beta to SMB proxy (IWM-SPY)
  ff_hml_beta_60d           -- Rolling 60d beta to HML proxy (IWD-IWF)
  ff_momentum_beta_60d      -- Rolling 60d beta to momentum proxy (MTUM)
  ff_factor_momentum        -- Composite factor momentum signal
  ff_value_growth_spread    -- IWD/IWF spread z-score
  ff_size_spread            -- IWM/SPY spread z-score
  ff_factor_regime          -- 1.0 (value), -1.0 (growth), 0.0 (neutral)
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class FamaFrenchFeatures:
    """Compute Fama-French factor exposure features from ETF proxies."""

    REQUIRED_COLS = {"close"}

    def __init__(self) -> None:
        self._factor_data: Optional[pd.DataFrame] = None

    def download_factor_data(
        self,
        start_date: str = "2019-01-01",
        end_date: str = "2026-12-31",
    ) -> Optional[pd.DataFrame]:
        """Download factor proxy ETFs via yfinance.

        Tickers: IWM (small cap), IWD (value), IWF (growth), MTUM (momentum).
        Returns DataFrame or None on failure.
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.info("FamaFrenchFeatures: yfinance not installed, will use zero proxies")
            return None

        try:
            tickers = ["SPY", "IWM", "IWD", "IWF", "MTUM"]
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)

            if data.empty:
                logger.warning("FamaFrenchFeatures: yfinance returned empty data")
                return None

            if isinstance(data.columns, pd.MultiIndex):
                closes = data["Close"] if "Close" in data.columns.get_level_values(0) else data["Adj Close"]
            else:
                closes = data

            result = pd.DataFrame({"date": closes.index})
            available = []
            for tkr in tickers:
                col = f"{tkr.lower()}_close"
                if tkr in closes.columns:
                    result[col] = closes[tkr].values
                    available.append(tkr)

            if len(available) < 2:
                logger.warning(f"FamaFrenchFeatures: only {available} available, insufficient")
                return None

            result["date"] = pd.to_datetime(result["date"])
            self._factor_data = result
            logger.info(f"FamaFrenchFeatures: downloaded {len(result)} rows, tickers: {available}")
            return result

        except Exception as e:
            logger.warning(f"FamaFrenchFeatures: download failed: {e}")
            return None

    def create_fama_french_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """Compute factor exposure features.

        Falls back to zero-filled columns if external data unavailable.
        """
        df = df_daily.copy()

        if self._factor_data is not None and len(self._factor_data) > 60:
            df = self._create_full_features(df)
        else:
            df = self._create_proxy_features(df)

        return df

    def _create_full_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features from real ETF data."""
        fd = self._factor_data.copy()
        fd["date"] = pd.to_datetime(fd["date"])

        # Compute returns for available tickers
        for col in ["spy_close", "iwm_close", "iwd_close", "iwf_close", "mtum_close"]:
            ret_col = col.replace("_close", "_ret")
            if col in fd.columns:
                fd[ret_col] = fd[col].pct_change()

        # Factor proxies
        if "iwm_ret" in fd.columns and "spy_ret" in fd.columns:
            fd["smb_proxy"] = fd["iwm_ret"] - fd["spy_ret"]
        else:
            fd["smb_proxy"] = 0.0

        if "iwd_ret" in fd.columns and "iwf_ret" in fd.columns:
            fd["hml_proxy"] = fd["iwd_ret"] - fd["iwf_ret"]
        else:
            fd["hml_proxy"] = 0.0

        fd["mom_proxy"] = fd.get("mtum_ret", pd.Series(0.0, index=fd.index))

        # Merge with daily
        df["date"] = pd.to_datetime(df["date"])
        merge_cols = ["date", "spy_ret", "smb_proxy", "hml_proxy", "mom_proxy"]
        # Add spread data for z-scores
        if "iwm_close" in fd.columns and "spy_close" in fd.columns:
            fd["size_spread"] = (fd["iwm_close"] / fd["spy_close"]).pct_change()
            merge_cols.append("size_spread")
        if "iwd_close" in fd.columns and "iwf_close" in fd.columns:
            fd["vg_spread"] = (fd["iwd_close"] / fd["iwf_close"]).pct_change()
            merge_cols.append("vg_spread")

        available_merge = [c for c in merge_cols if c in fd.columns]
        df = df.merge(fd[available_merge], on="date", how="left")

        for col in available_merge:
            if col in df.columns and col != "date":
                df[col] = df[col].ffill().fillna(0)

        spy_ret = df.get("spy_ret", pd.Series(0.0, index=df.index)).fillna(0)
        smb = df.get("smb_proxy", pd.Series(0.0, index=df.index)).fillna(0)
        hml = df.get("hml_proxy", pd.Series(0.0, index=df.index)).fillna(0)
        mom = df.get("mom_proxy", pd.Series(0.0, index=df.index)).fillna(0)

        # Rolling betas (60d OLS slope proxy via correlation * vol ratio)
        df["ff_mkt_beta_60d"] = self._rolling_beta(spy_ret, spy_ret, 60)
        df["ff_smb_beta_60d"] = self._rolling_beta(spy_ret, smb, 60)
        df["ff_hml_beta_60d"] = self._rolling_beta(spy_ret, hml, 60)
        df["ff_momentum_beta_60d"] = self._rolling_beta(spy_ret, mom, 60)

        # Factor momentum composite
        df["ff_factor_momentum"] = (
            df["ff_smb_beta_60d"] * smb.rolling(20).mean()
            + df["ff_hml_beta_60d"] * hml.rolling(20).mean()
            + df["ff_momentum_beta_60d"] * mom.rolling(20).mean()
        )

        # Value/growth spread z-score
        if "vg_spread" in df.columns:
            vg_cum = df["vg_spread"].rolling(20).sum()
            df["ff_value_growth_spread"] = (
                (vg_cum - vg_cum.rolling(252).mean())
                / vg_cum.rolling(252).std().replace(0, np.nan)
            )
        else:
            df["ff_value_growth_spread"] = 0.0

        # Size spread z-score
        if "size_spread" in df.columns:
            sz_cum = df["size_spread"].rolling(20).sum()
            df["ff_size_spread"] = (
                (sz_cum - sz_cum.rolling(252).mean())
                / sz_cum.rolling(252).std().replace(0, np.nan)
            )
        else:
            df["ff_size_spread"] = 0.0

        # Factor regime
        df["ff_factor_regime"] = 0.0
        df.loc[df["ff_value_growth_spread"] > 1.0, "ff_factor_regime"] = 1.0  # value
        df.loc[df["ff_value_growth_spread"] < -1.0, "ff_factor_regime"] = -1.0  # growth

        # Clean up temp columns
        for col in ["spy_ret", "smb_proxy", "hml_proxy", "mom_proxy", "size_spread", "vg_spread"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True, errors="ignore")

        # Fill NaN
        ff_cols = [c for c in df.columns if c.startswith("ff_")]
        for col in ff_cols:
            df[col] = df[col].fillna(0.0)

        return df

    def _create_proxy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Zero-fill features when external data unavailable."""
        logger.info("FamaFrenchFeatures: using zero-fill proxy (no factor ETF data)")
        for col in [
            "ff_mkt_beta_60d", "ff_smb_beta_60d", "ff_hml_beta_60d",
            "ff_momentum_beta_60d", "ff_factor_momentum",
            "ff_value_growth_spread", "ff_size_spread", "ff_factor_regime",
        ]:
            df[col] = 0.0
        return df

    @staticmethod
    def _rolling_beta(y: pd.Series, x: pd.Series, window: int) -> pd.Series:
        """Rolling OLS beta = cov(y,x) / var(x)."""
        cov = y.rolling(window).cov(x)
        var = x.rolling(window).var()
        return (cov / var.replace(0, np.nan)).fillna(0.0)

    def analyze_current_factors(self, df_daily: pd.DataFrame) -> Optional[Dict]:
        """Analyze current factor regime."""
        if df_daily.empty:
            return None

        last = df_daily.iloc[-1]
        return {
            "mkt_beta": round(float(last.get("ff_mkt_beta_60d", 0)), 3),
            "smb_beta": round(float(last.get("ff_smb_beta_60d", 0)), 3),
            "hml_beta": round(float(last.get("ff_hml_beta_60d", 0)), 3),
            "momentum_beta": round(float(last.get("ff_momentum_beta_60d", 0)), 3),
            "factor_regime": "VALUE" if last.get("ff_factor_regime", 0) > 0
            else "GROWTH" if last.get("ff_factor_regime", 0) < 0 else "NEUTRAL",
        }
