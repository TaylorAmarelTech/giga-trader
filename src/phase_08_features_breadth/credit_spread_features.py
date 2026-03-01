"""
Credit Spread Features -- high-yield vs investment-grade credit stress signals.

HY-IG spread widens during risk-off episodes and tightens in risk-on.
This module downloads HYG (high-yield ETF) and LQD (investment-grade ETF)
via yfinance to compute the spread.  When no external data is available a
volatility-based proxy is used instead.

Features (8, prefix cred_):
  cred_hy_ig_spread       -- HY minus IG return spread (or vol proxy)
  cred_spread_20d         -- 20-day rolling mean of cred_hy_ig_spread
  cred_spread_z           -- 60-day z-score of cred_spread_20d, clipped [-4, 4]
  cred_spread_momentum    -- 5-day change of cred_spread_20d
  cred_spread_accel       -- 5-day change of cred_spread_momentum
  cred_hy_return_20d      -- 20-day rolling HYG return (or proxy)
  cred_ig_return_20d      -- 20-day rolling LQD return (or proxy)
  cred_spread_regime      -- 1.0 stress (z>1.5), -1.0 calm (z<-1.5), 0.0 normal
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class CreditSpreadFeatures:
    """Compute credit-spread features from HYG/LQD data or close-price proxy."""

    REQUIRED_COLS = {"close"}

    def __init__(self) -> None:
        self._credit_data: Optional[pd.DataFrame] = None

    # ─── Download ──────────────────────────────────────────────────────

    def download_credit_data(self, start_date, end_date) -> None:
        """
        Download HYG and LQD daily closes via yfinance.

        Stores result as self._credit_data with columns
        ``date``, ``hyg_close``, ``lqd_close``.  On any failure the
        attribute is set to ``None`` so the feature builder falls back
        to a volatility proxy.
        """
        try:
            import yfinance as yf

            tickers = yf.download(
                ["HYG", "LQD"],
                start=start_date,
                end=end_date,
                progress=False,
            )

            if tickers.empty:
                self._credit_data = None
                return

            adj = tickers["Close"] if "Close" in tickers.columns.get_level_values(0) else tickers["Adj Close"]
            df = pd.DataFrame({
                "hyg_close": adj["HYG"].values,
                "lqd_close": adj["LQD"].values,
            }, index=adj.index)
            df = df.dropna()
            df["date"] = pd.to_datetime(df.index.date)
            df = df.reset_index(drop=True)

            if df.empty:
                self._credit_data = None
            else:
                self._credit_data = df
                logger.info("CreditSpreadFeatures: downloaded %d rows of HYG/LQD data", len(df))

        except Exception as exc:
            logger.warning("CreditSpreadFeatures: download failed (%s), will use proxy", exc)
            self._credit_data = None

    # ─── Feature builder ──────────────────────────────────────────────

    def create_credit_spread_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Add 8 credit-spread features (prefix ``cred_``) to *df_daily*.

        When downloaded HYG/LQD data is available it is merged by date.
        Otherwise a volatility-based proxy is computed from the ``close``
        column.
        """
        df = df_daily.copy()

        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            logger.warning("CreditSpreadFeatures: missing columns %s, skipping", missing)
            return df

        close = df["close"].astype(float)
        returns = close.pct_change()

        # ------ Determine spread source ------
        if self._credit_data is not None and not self._credit_data.empty and "date" in df.columns:
            # Merge real data
            credit = self._credit_data.copy()
            credit["date"] = pd.to_datetime(credit["date"])
            df["date"] = pd.to_datetime(df["date"])

            hyg_ret = credit["hyg_close"].pct_change()
            lqd_ret = credit["lqd_close"].pct_change()

            credit["_hyg_ret"] = hyg_ret
            credit["_lqd_ret"] = lqd_ret
            credit["_spread"] = hyg_ret - lqd_ret

            merge_cols = ["date", "_hyg_ret", "_lqd_ret", "_spread"]
            df = df.merge(credit[merge_cols], on="date", how="left")

            # Forward-fill small gaps
            for c in ["_hyg_ret", "_lqd_ret", "_spread"]:
                df[c] = df[c].fillna(method="ffill", limit=3)

            spread_series = df["_spread"].fillna(0.0)
            hyg_ret_series = df["_hyg_ret"].fillna(0.0)
            lqd_ret_series = df["_lqd_ret"].fillna(0.0)

            df.drop(columns=["_hyg_ret", "_lqd_ret", "_spread"], inplace=True)
        else:
            # Proxy: rolling 20d vol of close returns (higher vol ~ wider spreads)
            spread_series = returns.rolling(20, min_periods=5).std().fillna(0.0)
            hyg_ret_series = returns.fillna(0.0)
            lqd_ret_series = pd.Series(np.zeros(len(df)), index=df.index)

        # ------ Compute features ------
        df["cred_hy_ig_spread"] = spread_series.values
        df["cred_spread_20d"] = pd.Series(spread_series.values).rolling(20, min_periods=5).mean().values
        df["cred_spread_20d"] = df["cred_spread_20d"].fillna(0.0)

        # Z-score
        s20 = df["cred_spread_20d"]
        roll_mean = s20.rolling(60, min_periods=10).mean()
        roll_std = s20.rolling(60, min_periods=10).std()
        df["cred_spread_z"] = ((s20 - roll_mean) / (roll_std + 1e-10)).clip(-4, 4)

        # Momentum / acceleration
        df["cred_spread_momentum"] = df["cred_spread_20d"].diff(5)
        df["cred_spread_accel"] = df["cred_spread_momentum"].diff(5)

        # Rolling returns
        df["cred_hy_return_20d"] = pd.Series(hyg_ret_series.values).rolling(20, min_periods=5).sum().values
        df["cred_ig_return_20d"] = pd.Series(lqd_ret_series.values).rolling(20, min_periods=5).sum().values

        # Regime
        z = df["cred_spread_z"]
        df["cred_spread_regime"] = np.where(z > 1.5, 1.0, np.where(z < -1.5, -1.0, 0.0))

        # ------ NaN / Inf cleanup ------
        for col in self._all_feature_names():
            df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        n_features = sum(1 for c in df.columns if c.startswith("cred_"))
        logger.info("CreditSpreadFeatures: added %d features", n_features)
        return df

    # ─── Analysis ─────────────────────────────────────────────────────

    def analyze_current_credit_spread(self, df_daily: pd.DataFrame) -> Optional[Dict]:
        """Return current credit-spread regime info, or *None* if features absent."""
        if "cred_spread_z" not in df_daily.columns or len(df_daily) < 2:
            return None

        last = df_daily.iloc[-1]
        z = float(last.get("cred_spread_z", 0.0))

        if z > 1.5:
            regime = "STRESS"
        elif z < -1.5:
            regime = "CALM"
        else:
            regime = "NORMAL"

        return {
            "credit_regime": regime,
            "spread_z": round(z, 3),
            "spread_20d": round(float(last.get("cred_spread_20d", 0.0)), 6),
            "spread_momentum": round(float(last.get("cred_spread_momentum", 0.0)), 6),
        }

    # ─── Metadata ─────────────────────────────────────────────────────

    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "cred_hy_ig_spread",
            "cred_spread_20d",
            "cred_spread_z",
            "cred_spread_momentum",
            "cred_spread_accel",
            "cred_hy_return_20d",
            "cred_ig_return_20d",
            "cred_spread_regime",
        ]
