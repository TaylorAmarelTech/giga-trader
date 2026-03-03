"""
Correlation Regime Features -- rolling cross-asset correlation matrix for regime detection.

Downloads SPY, TLT, GLD daily data via yfinance and computes rolling
pairwise correlations, eigenvalue ratios, and regime shift indicators.
Falls back to proxy features derived solely from SPY close if external
data is unavailable.

Features (12, prefix corr_):
  corr_spy_tlt_20d          -- Rolling 20d correlation SPY vs TLT returns
  corr_spy_gld_20d          -- Rolling 20d correlation SPY vs GLD returns
  corr_spy_vix_20d          -- Rolling 20d correlation SPY vs realized vol proxy
  corr_tlt_gld_20d          -- Rolling 20d TLT vs GLD correlation
  corr_mean_abs_20d         -- Mean absolute correlation across all pairs
  corr_spy_tlt_60d          -- Rolling 60d SPY-TLT correlation (slower regime)
  corr_regime_shift         -- |20d - 60d| SPY-TLT correlation (transition speed)
  corr_dispersion           -- Std dev of all pairwise correlations
  corr_risk_on_off          -- Composite risk-on/off signal from correlation structure
  corr_eigen_ratio          -- Ratio of 1st/2nd eigenvalue of correlation matrix
  corr_stability_20d        -- Rolling 20d std of daily correlation changes
  corr_spy_tlt_zscore       -- Z-score of 20d SPY-TLT corr vs 252d history
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CorrelationRegimeFeatures:
    """Compute rolling cross-asset correlation regime features."""

    REQUIRED_COLS = {"close"}

    def __init__(self) -> None:
        self._cross_data: Optional[pd.DataFrame] = None

    def download_correlation_data(
        self,
        start_date: str = "2019-01-01",
        end_date: str = "2026-12-31",
    ) -> Optional[pd.DataFrame]:
        """Download SPY, TLT, GLD daily closes via yfinance.

        Returns DataFrame with [date, spy_close, tlt_close, gld_close]
        or None on failure.
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.info("CorrelationRegimeFeatures: yfinance not installed, will use proxies")
            return None

        try:
            tickers = ["SPY", "TLT", "GLD"]
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)

            if data.empty:
                logger.warning("CorrelationRegimeFeatures: yfinance returned empty data")
                return None

            if isinstance(data.columns, pd.MultiIndex):
                closes = data["Close"] if "Close" in data.columns.get_level_values(0) else data["Adj Close"]
            else:
                closes = data

            result = pd.DataFrame({"date": closes.index})
            for tkr, col in [("SPY", "spy_close"), ("TLT", "tlt_close"), ("GLD", "gld_close")]:
                if tkr in closes.columns:
                    result[col] = closes[tkr].values

            if "tlt_close" not in result.columns or "gld_close" not in result.columns:
                logger.warning("CorrelationRegimeFeatures: missing TLT/GLD data")
                return None

            result["date"] = pd.to_datetime(result["date"])
            self._cross_data = result
            logger.info(f"CorrelationRegimeFeatures: downloaded {len(result)} rows")
            return result

        except Exception as e:
            logger.warning(f"CorrelationRegimeFeatures: download failed: {e}")
            return None

    def create_correlation_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """Compute correlation regime features.

        If cross-asset data is unavailable, computes proxy features
        from SPY close only (realized vol autocorrelations).
        """
        df = df_daily.copy()

        if self._cross_data is not None and len(self._cross_data) > 20:
            df = self._create_full_features(df)
        else:
            df = self._create_proxy_features(df)

        return df

    def _create_full_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute features using real cross-asset data."""
        cross = self._cross_data.copy()
        cross["date"] = pd.to_datetime(cross["date"])

        # Compute returns
        for col in ["spy_close", "tlt_close", "gld_close"]:
            if col in cross.columns:
                cross[col.replace("_close", "_ret")] = cross[col].pct_change()

        # Realized vol as VIX proxy
        cross["rvol_proxy"] = cross["spy_ret"].rolling(20).std() * np.sqrt(252)

        df["date"] = pd.to_datetime(df["date"])
        df = df.merge(
            cross[["date", "spy_ret", "tlt_ret", "gld_ret", "rvol_proxy"]],
            on="date",
            how="left",
        )

        # Fill forward cross data for missing dates
        for col in ["spy_ret", "tlt_ret", "gld_ret", "rvol_proxy"]:
            if col in df.columns:
                df[col] = df[col].ffill().fillna(0)

        # Use SPY return from df if available
        spy_ret = df["spy_ret"] if "spy_ret" in df.columns else df.get("close", pd.Series(dtype=float)).pct_change()
        tlt_ret = df.get("tlt_ret", pd.Series(0.0, index=df.index))
        gld_ret = df.get("gld_ret", pd.Series(0.0, index=df.index))
        rvol = df.get("rvol_proxy", pd.Series(0.0, index=df.index))

        # Rolling correlations
        df["corr_spy_tlt_20d"] = spy_ret.rolling(20).corr(tlt_ret)
        df["corr_spy_gld_20d"] = spy_ret.rolling(20).corr(gld_ret)
        df["corr_spy_vix_20d"] = spy_ret.rolling(20).corr(rvol)
        df["corr_tlt_gld_20d"] = tlt_ret.rolling(20).corr(gld_ret)
        df["corr_spy_tlt_60d"] = spy_ret.rolling(60).corr(tlt_ret)

        # Mean absolute correlation
        pair_corrs = df[["corr_spy_tlt_20d", "corr_spy_gld_20d", "corr_tlt_gld_20d"]]
        df["corr_mean_abs_20d"] = pair_corrs.abs().mean(axis=1)

        # Regime shift speed
        df["corr_regime_shift"] = (df["corr_spy_tlt_20d"] - df["corr_spy_tlt_60d"]).abs()

        # Dispersion
        df["corr_dispersion"] = pair_corrs.std(axis=1)

        # Risk on/off
        df["corr_risk_on_off"] = 0.0
        df.loc[df["corr_spy_tlt_20d"] < -0.3, "corr_risk_on_off"] = 1.0  # risk-on
        df.loc[df["corr_spy_tlt_20d"] > 0.3, "corr_risk_on_off"] = -1.0  # risk-off

        # Eigenvalue ratio (rolling 20d)
        df["corr_eigen_ratio"] = self._rolling_eigen_ratio(spy_ret, tlt_ret, gld_ret, window=20)

        # Stability
        df["corr_stability_20d"] = df["corr_spy_tlt_20d"].diff().rolling(20).std()

        # Z-score
        roll_mean = df["corr_spy_tlt_20d"].rolling(252).mean()
        roll_std = df["corr_spy_tlt_20d"].rolling(252).std()
        df["corr_spy_tlt_zscore"] = (df["corr_spy_tlt_20d"] - roll_mean) / roll_std.replace(0, np.nan)

        # Clean up temp columns
        for col in ["spy_ret", "tlt_ret", "gld_ret", "rvol_proxy"]:
            if col in df.columns:
                df.drop(columns=[col], inplace=True, errors="ignore")

        # Fill NaN
        corr_cols = [c for c in df.columns if c.startswith("corr_")]
        for col in corr_cols:
            df[col] = df[col].fillna(0.0)

        return df

    def _create_proxy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute proxy features from SPY close only."""
        logger.info("CorrelationRegimeFeatures: using proxy features (no cross-asset data)")

        close = df["close"] if "close" in df.columns else pd.Series(0.0, index=df.index)
        ret = close.pct_change().fillna(0)
        rvol = ret.rolling(20).std() * np.sqrt(252)

        # Self-correlation proxies
        df["corr_spy_tlt_20d"] = ret.rolling(20).corr(ret.shift(1))  # lag-1 autocorrelation
        df["corr_spy_gld_20d"] = ret.rolling(20).corr(ret.shift(2))  # lag-2
        df["corr_spy_vix_20d"] = ret.rolling(20).corr(rvol)
        df["corr_tlt_gld_20d"] = 0.0  # no proxy available
        df["corr_mean_abs_20d"] = df[["corr_spy_tlt_20d", "corr_spy_gld_20d"]].abs().mean(axis=1)
        df["corr_spy_tlt_60d"] = ret.rolling(60).corr(ret.shift(1))
        df["corr_regime_shift"] = (df["corr_spy_tlt_20d"] - df["corr_spy_tlt_60d"]).abs()
        df["corr_dispersion"] = 0.0
        df["corr_risk_on_off"] = 0.0
        df["corr_eigen_ratio"] = 1.0  # single asset = dominated by 1 factor
        df["corr_stability_20d"] = df["corr_spy_tlt_20d"].diff().rolling(20).std()

        roll_mean = df["corr_spy_tlt_20d"].rolling(252).mean()
        roll_std = df["corr_spy_tlt_20d"].rolling(252).std()
        df["corr_spy_tlt_zscore"] = (df["corr_spy_tlt_20d"] - roll_mean) / roll_std.replace(0, np.nan)

        corr_cols = [c for c in df.columns if c.startswith("corr_")]
        for col in corr_cols:
            df[col] = df[col].fillna(0.0)

        return df

    def _rolling_eigen_ratio(
        self,
        s1: pd.Series,
        s2: pd.Series,
        s3: pd.Series,
        window: int = 20,
    ) -> pd.Series:
        """Rolling ratio of 1st/2nd eigenvalue of 3x3 correlation matrix."""
        result = pd.Series(np.nan, index=s1.index)

        arr = np.column_stack([
            s1.fillna(0).values,
            s2.fillna(0).values,
            s3.fillna(0).values,
        ])

        for i in range(window, len(arr)):
            block = arr[i - window : i]
            if np.std(block, axis=0).min() == 0:
                result.iloc[i] = 1.0
                continue
            try:
                corr_matrix = np.corrcoef(block.T)
                eigenvalues = np.sort(np.linalg.eigvalsh(corr_matrix))[::-1]
                if eigenvalues[1] > 1e-10:
                    result.iloc[i] = eigenvalues[0] / eigenvalues[1]
                else:
                    result.iloc[i] = eigenvalues[0]
            except Exception:
                result.iloc[i] = 1.0

        return result

    def analyze_current_regime(self, df_daily: pd.DataFrame) -> Optional[Dict]:
        """Analyze current correlation regime."""
        if df_daily.empty:
            return None

        last = df_daily.iloc[-1]
        result = {}

        spy_tlt = last.get("corr_spy_tlt_20d", 0.0)
        risk_on_off = last.get("corr_risk_on_off", 0.0)
        eigen_ratio = last.get("corr_eigen_ratio", 1.0)
        dispersion = last.get("corr_dispersion", 0.0)

        if risk_on_off > 0:
            result["regime"] = "RISK_ON"
        elif risk_on_off < 0:
            result["regime"] = "RISK_OFF"
        else:
            result["regime"] = "NEUTRAL"

        result["spy_tlt_corr"] = round(float(spy_tlt), 3)
        result["eigen_dominance"] = round(float(eigen_ratio), 2)
        result["correlation_dispersion"] = round(float(dispersion), 3)

        if eigen_ratio > 3.0:
            result["factor_structure"] = "SINGLE_FACTOR_DOMINATED"
        else:
            result["factor_structure"] = "DIVERSIFIED"

        return result
