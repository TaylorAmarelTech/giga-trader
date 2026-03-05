"""
Cross-Asset Momentum Features -- leading cross-asset signals for SPY.

Downloads TLT, GLD, HYG daily data via yfinance and computes lagged
correlations, lead-lag t-statistics, and cross-momentum composites.
Falls back to proxy features derived solely from SPY close if external
data is unavailable.

Features (12, prefix xmom_):
  xmom_tlt_lead_5d          -- Rolling 20d correlation between TLT_ret(t-5) and SPY_ret(t)
  xmom_gld_lead_5d          -- Same for GLD
  xmom_hyg_lead_5d          -- Same for HYG
  xmom_tlt_lead_tstat       -- Rolling 60d OLS t-stat of TLT lagged return predicting SPY
  xmom_gld_lead_tstat       -- Same for GLD
  xmom_hyg_lead_tstat       -- Same for HYG
  xmom_cross_momentum_composite -- Equal-weighted mean of the 3 t-stats
  xmom_cross_divergence     -- SPY 5d return minus mean(TLT, GLD, HYG 5d returns)
  xmom_tlt_spy_corr_20d     -- Rolling 20d Pearson correlation, TLT vs SPY returns
  xmom_gld_spy_corr_20d     -- Same for GLD
  xmom_corr_regime_change   -- |20d change in xmom_tlt_spy_corr_20d|
  xmom_momentum_regime      -- 1.0 (RISK_ON) / -1.0 (RISK_OFF) / 0.0 (NEUTRAL)
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from src.core.feature_base import FeatureModuleBase

logger = logging.getLogger(__name__)


class CrossAssetMomentumFeatures(FeatureModuleBase):
    """Compute cross-asset leading-momentum features from daily OHLCV data."""
    FEATURE_NAMES = ["xmom_tlt_lead_5d", "xmom_gld_lead_5d", "xmom_hyg_lead_5d", "xmom_tlt_lead_tstat", "xmom_gld_lead_tstat", "xmom_hyg_lead_tstat", "xmom_cross_momentum_composite", "xmom_cross_divergence", "xmom_tlt_spy_corr_20d", "xmom_gld_spy_corr_20d", "xmom_corr_regime_change", "xmom_momentum_regime"]


    REQUIRED_COLS = {"close"}

    def __init__(self) -> None:
        self._cross_data: Optional[pd.DataFrame] = None  # TLT, GLD, HYG closes

    # ------------------------------------------------------------------
    # Data download
    # ------------------------------------------------------------------

    def download_cross_asset_data(
        self,
        start_date: str = "2019-01-01",
        end_date: str = "2026-12-31",
    ) -> Optional[pd.DataFrame]:
        """
        Download TLT, GLD, HYG daily closes via yfinance.

        Stores result internally for later use by create_* method.
        Returns DataFrame with columns [date, tlt_close, gld_close, hyg_close]
        or None on failure.
        """
        try:
            import yfinance as yf
        except ImportError:
            logger.info("CrossAssetMomentumFeatures: yfinance not installed, will use proxies")
            return None

        try:
            tickers = ["TLT", "GLD", "HYG"]
            data = yf.download(tickers, start=start_date, end=end_date, progress=False)

            if data.empty:
                logger.warning("CrossAssetMomentumFeatures: yfinance returned empty data")
                return None

            # Handle multi-level columns from yf.download
            if isinstance(data.columns, pd.MultiIndex):
                closes = data["Close"] if "Close" in data.columns.get_level_values(0) else data["Adj Close"]
            else:
                closes = data

            result = pd.DataFrame({"date": closes.index})
            for tkr, col_name in [("TLT", "tlt_close"), ("GLD", "gld_close"), ("HYG", "hyg_close")]:
                if tkr in closes.columns:
                    result[col_name] = closes[tkr].values
                else:
                    logger.warning(f"CrossAssetMomentumFeatures: {tkr} not in download")

            result["date"] = pd.to_datetime(result["date"]).dt.tz_localize(None)
            result = result.dropna().reset_index(drop=True)
            self._cross_data = result
            logger.info(f"CrossAssetMomentumFeatures: downloaded {len(result)} days of cross-asset data")
            return result

        except Exception as exc:
            logger.warning(f"CrossAssetMomentumFeatures: download failed: {exc}")
            return None

    # ------------------------------------------------------------------
    # Feature creation
    # ------------------------------------------------------------------

    def create_cross_asset_momentum_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Add 12 xmom_ features to df_daily.

        Parameters
        ----------
        df_daily : pd.DataFrame
            Must have 'close' column.

        Returns
        -------
        pd.DataFrame
            Original df_daily with 12 new xmom_ columns added.
        """
        df = df_daily.copy()

        if "close" not in df.columns:
            logger.warning("CrossAssetMomentumFeatures: 'close' column missing, skipping")
            return df

        spy_ret = df["close"].pct_change()
        spy_ret_5d = df["close"].pct_change(5)

        # Determine whether we have real cross-asset data
        have_cross = self._cross_data is not None and len(self._cross_data) > 0
        if have_cross:
            df = self._compute_with_cross_data(df, spy_ret, spy_ret_5d)
        else:
            df = self._compute_proxy_features(df, spy_ret, spy_ret_5d)

        # Cleanup individual t-stat columns FIRST, then recompute composite
        # so the composite equals the mean of cleaned values.
        tstat_cols = ["xmom_tlt_lead_tstat", "xmom_gld_lead_tstat", "xmom_hyg_lead_tstat"]
        for col in tstat_cols:
            if col in df.columns:
                df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        # Recompute composite from cleaned t-stat values
        if all(c in df.columns for c in tstat_cols):
            df["xmom_cross_momentum_composite"] = (
                df["xmom_tlt_lead_tstat"]
                + df["xmom_gld_lead_tstat"]
                + df["xmom_hyg_lead_tstat"]
            ) / 3.0

        # Recompute regime from updated composite
        if "xmom_cross_momentum_composite" in df.columns:
            comp = df["xmom_cross_momentum_composite"]
            df["xmom_momentum_regime"] = np.where(
                comp > 0.5, 1.0, np.where(comp < -0.5, -1.0, 0.0)
            )

        # Cleanup: fill NaN with 0.0, remove infinities
        for col in self._all_feature_names():
            if col in df.columns:
                df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        n_features = sum(1 for c in df.columns if c.startswith("xmom_"))
        logger.info(f"CrossAssetMomentumFeatures: added {n_features} features")
        return df

    # ------------------------------------------------------------------
    # Real cross-asset computation
    # ------------------------------------------------------------------

    def _compute_with_cross_data(
        self,
        df: pd.DataFrame,
        spy_ret: pd.Series,
        spy_ret_5d: pd.Series,
    ) -> pd.DataFrame:
        """Compute features using real TLT/GLD/HYG data."""
        cross = self._cross_data.copy()

        # Merge cross-asset data by date
        if "date" in df.columns:
            merge_key = pd.to_datetime(df["date"]).dt.normalize()
        elif isinstance(df.index, pd.DatetimeIndex):
            merge_key = df.index.normalize()
        else:
            # Cannot merge, fall back to proxies
            return self._compute_proxy_features(df, spy_ret, spy_ret_5d)

        cross["date"] = pd.to_datetime(cross["date"]).dt.normalize()

        # Build a temporary aligned frame
        tmp = pd.DataFrame({"date": merge_key.values, "spy_ret": spy_ret.values})
        tmp = tmp.merge(cross, on="date", how="left")

        assets = {}
        for name, col in [("tlt", "tlt_close"), ("gld", "gld_close"), ("hyg", "hyg_close")]:
            if col in tmp.columns:
                assets[name] = pd.Series(tmp[col].values, index=df.index).pct_change()
            else:
                assets[name] = spy_ret.copy()  # fallback to spy itself

        for name in ["tlt", "gld", "hyg"]:
            asset_ret = assets[name]
            lagged_ret = asset_ret.shift(5)

            # xmom_{name}_lead_5d: rolling 20d correlation between lagged asset and SPY
            df[f"xmom_{name}_lead_5d"] = spy_ret.rolling(20, min_periods=10).corr(lagged_ret)

            # xmom_{name}_lead_tstat: rolling 60d OLS t-stat
            df[f"xmom_{name}_lead_tstat"] = self._rolling_tstat(
                spy_ret.values, lagged_ret.values, window=60
            )

        # Composite: mean of 3 t-stats
        df["xmom_cross_momentum_composite"] = (
            df["xmom_tlt_lead_tstat"]
            + df["xmom_gld_lead_tstat"]
            + df["xmom_hyg_lead_tstat"]
        ) / 3.0

        # Divergence: SPY 5d return - mean of cross-asset 5d returns
        cross_5d = pd.DataFrame()
        for name, col in [("tlt", "tlt_close"), ("gld", "gld_close"), ("hyg", "hyg_close")]:
            if col in tmp.columns:
                s = pd.Series(tmp[col].values, index=df.index)
                cross_5d[name] = s.pct_change(5)
        if len(cross_5d.columns) > 0:
            df["xmom_cross_divergence"] = spy_ret_5d - cross_5d.mean(axis=1)
        else:
            df["xmom_cross_divergence"] = spy_ret_5d - spy_ret.rolling(20).mean() * 20

        # Contemporaneous correlations
        df["xmom_tlt_spy_corr_20d"] = spy_ret.rolling(20, min_periods=10).corr(assets["tlt"])
        df["xmom_gld_spy_corr_20d"] = spy_ret.rolling(20, min_periods=10).corr(assets["gld"])

        # Correlation regime change
        df["xmom_corr_regime_change"] = df["xmom_tlt_spy_corr_20d"].diff(20).abs()

        # Momentum regime
        comp = df["xmom_cross_momentum_composite"]
        df["xmom_momentum_regime"] = np.where(
            comp > 0.5, 1.0, np.where(comp < -0.5, -1.0, 0.0)
        )

        return df

    # ------------------------------------------------------------------
    # Proxy computation (no external data)
    # ------------------------------------------------------------------

    def _compute_proxy_features(
        self,
        df: pd.DataFrame,
        spy_ret: pd.Series,
        spy_ret_5d: pd.Series,
    ) -> pd.DataFrame:
        """Compute proxy features from SPY close alone."""
        lagged_5 = spy_ret.shift(5)

        # Lagged autocorrelation proxies
        df["xmom_tlt_lead_5d"] = spy_ret.rolling(20, min_periods=10).corr(lagged_5)
        df["xmom_gld_lead_5d"] = spy_ret.rolling(20, min_periods=10).corr(spy_ret.shift(10))
        df["xmom_hyg_lead_5d"] = spy_ret.rolling(20, min_periods=10).corr(spy_ret.shift(3))

        # T-stat proxies: z-score of corresponding lead feature
        for src, dst in [
            ("xmom_tlt_lead_5d", "xmom_tlt_lead_tstat"),
            ("xmom_gld_lead_5d", "xmom_gld_lead_tstat"),
            ("xmom_hyg_lead_5d", "xmom_hyg_lead_tstat"),
        ]:
            s = df[src]
            rmean = s.rolling(60, min_periods=20).mean()
            rstd = s.rolling(60, min_periods=20).std()
            df[dst] = ((s - rmean) / (rstd + 1e-10)).clip(-4, 4)

        # Composite
        df["xmom_cross_momentum_composite"] = (
            df["xmom_tlt_lead_tstat"]
            + df["xmom_gld_lead_tstat"]
            + df["xmom_hyg_lead_tstat"]
        ) / 3.0

        # Divergence proxy: SPY 5d return - SPY 20d return
        spy_ret_20d = df["close"].pct_change(20)
        df["xmom_cross_divergence"] = spy_ret_5d - spy_ret_20d

        # Autocorrelation proxies for contemporaneous correlation
        df["xmom_tlt_spy_corr_20d"] = spy_ret.rolling(20, min_periods=10).corr(spy_ret.shift(1))
        df["xmom_gld_spy_corr_20d"] = spy_ret.rolling(20, min_periods=10).corr(spy_ret.shift(2))

        # Correlation regime change
        df["xmom_corr_regime_change"] = df["xmom_tlt_spy_corr_20d"].diff(20).abs()

        # Momentum regime
        comp = df["xmom_cross_momentum_composite"]
        df["xmom_momentum_regime"] = np.where(
            comp > 0.5, 1.0, np.where(comp < -0.5, -1.0, 0.0)
        )

        return df

    # ------------------------------------------------------------------
    # Analyze
    # ------------------------------------------------------------------

    def analyze_current_cross_asset_momentum(
        self, df_daily: pd.DataFrame
    ) -> Optional[Dict]:
        """Analyze current cross-asset momentum regime for dashboard display."""
        if "xmom_cross_momentum_composite" not in df_daily.columns or len(df_daily) < 2:
            return None

        last = df_daily.iloc[-1]
        comp = float(last.get("xmom_cross_momentum_composite", 0.0))
        regime_val = float(last.get("xmom_momentum_regime", 0.0))

        if regime_val > 0.5:
            regime = "RISK_ON"
        elif regime_val < -0.5:
            regime = "RISK_OFF"
        else:
            regime = "NEUTRAL"

        return {
            "momentum_regime": regime,
            "composite": round(comp, 4),
            "tlt_corr": round(float(last.get("xmom_tlt_spy_corr_20d", 0.0)), 4),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "xmom_tlt_lead_5d",
            "xmom_gld_lead_5d",
            "xmom_hyg_lead_5d",
            "xmom_tlt_lead_tstat",
            "xmom_gld_lead_tstat",
            "xmom_hyg_lead_tstat",
            "xmom_cross_momentum_composite",
            "xmom_cross_divergence",
            "xmom_tlt_spy_corr_20d",
            "xmom_gld_spy_corr_20d",
            "xmom_corr_regime_change",
            "xmom_momentum_regime",
        ]

    @staticmethod
    def _rolling_tstat(y: np.ndarray, x: np.ndarray, window: int = 60) -> np.ndarray:
        """
        Rolling OLS t-statistic of x predicting y.

        For each window, fits y = a + b*x and returns the t-stat of b.
        """
        n = len(y)
        result = np.full(n, 0.0)

        for i in range(window, n):
            y_w = y[i - window : i]
            x_w = x[i - window : i]

            # Skip if NaN
            valid = ~(np.isnan(y_w) | np.isnan(x_w))
            if valid.sum() < 10:
                continue

            y_v = y_w[valid]
            x_v = x_w[valid]
            n_v = len(y_v)

            x_mean = np.mean(x_v)
            y_mean = np.mean(y_v)
            ss_xx = np.sum((x_v - x_mean) ** 2)
            if ss_xx < 1e-20:
                continue

            beta = np.sum((x_v - x_mean) * (y_v - y_mean)) / ss_xx
            residuals = y_v - (y_mean + beta * (x_v - x_mean))
            mse = np.sum(residuals ** 2) / max(n_v - 2, 1)
            se_beta = np.sqrt(mse / (ss_xx + 1e-20))
            if se_beta > 1e-15:
                result[i] = np.clip(beta / se_beta, -10, 10)

        return result
