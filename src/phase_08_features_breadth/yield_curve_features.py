"""
Yield Curve Features -- Treasury yield curve shape and dynamics.

The yield curve (2s10s, 3m10y) is one of the most reliable macro signals.
Inversion (short rates > long rates) has preceded every US recession since
1960.  This module downloads ^TNX (10Y), ^FVX (5Y), ^IRX (3M) via yfinance
and computes slope, curvature, and regime features.  When no external data
is available a proxy from the close price is used.

Features (10, prefix yc_):
  yc_2s10s_slope         -- ^TNX - ^FVX (10Y minus 5Y proxy for 2s10s)
  yc_3m10y_slope         -- ^TNX - ^IRX (classic recession signal)
  yc_curvature           -- ^TNX + ^IRX - 2*^FVX (butterfly / belly)
  yc_slope_momentum_5d   -- 5-day change in yc_3m10y_slope
  yc_slope_momentum_20d  -- 20-day change in yc_3m10y_slope
  yc_slope_z             -- 60-day z-score of yc_3m10y_slope, clipped [-4, 4]
  yc_real_yield_proxy    -- ^TNX z-score (positive = tight, negative = loose)
  yc_inversion_flag      -- 1.0 if yc_3m10y_slope < 0, else 0.0
  yc_steepening_speed    -- 20-day change in yc_curvature
  yc_regime              -- 2.0 STEEP (slope>1.5), 1.0 NORMAL, 0.0 FLAT (<0.5), -1.0 INVERTED (<0)
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class YieldCurveFeatures:
    """Compute yield-curve features from Treasury data or close-price proxy."""

    REQUIRED_COLS = {"close"}

    def __init__(self) -> None:
        self._yield_data: Optional[pd.DataFrame] = None

    # ─── Download ──────────────────────────────────────────────────────

    def download_yield_data(self, start_date, end_date) -> None:
        """
        Download ^TNX, ^FVX, ^IRX daily closes via yfinance.

        Stores as self._yield_data with columns
        ``date``, ``tnx``, ``fvx``, ``irx``.
        On failure sets self._yield_data = None.
        """
        try:
            import yfinance as yf

            tickers = yf.download(
                ["^TNX", "^FVX", "^IRX"],
                start=start_date,
                end=end_date,
                progress=False,
            )

            if tickers.empty:
                self._yield_data = None
                return

            adj = tickers["Close"] if "Close" in tickers.columns.get_level_values(0) else tickers["Adj Close"]
            df = pd.DataFrame({
                "tnx": adj["^TNX"].values,
                "fvx": adj["^FVX"].values,
                "irx": adj["^IRX"].values,
            }, index=adj.index)
            df = df.dropna()
            df["date"] = pd.to_datetime(df.index.date)
            df = df.reset_index(drop=True)

            if df.empty:
                self._yield_data = None
            else:
                self._yield_data = df
                logger.info("YieldCurveFeatures: downloaded %d rows of yield data", len(df))

        except Exception as exc:
            logger.warning("YieldCurveFeatures: download failed (%s), will use proxy", exc)
            self._yield_data = None

    # ─── Feature builder ──────────────────────────────────────────────

    def create_yield_curve_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Add 10 yield-curve features (prefix ``yc_``) to *df_daily*.

        Uses downloaded Treasury data when available, otherwise computes
        a proxy from the ``close`` column (ratio of 5d vs 20d MA as
        slope proxy).
        """
        df = df_daily.copy()

        missing = self.REQUIRED_COLS - set(df.columns)
        if missing:
            logger.warning("YieldCurveFeatures: missing columns %s, skipping", missing)
            return df

        close = df["close"].astype(float)

        # ------ Determine yield source ------
        if self._yield_data is not None and not self._yield_data.empty and "date" in df.columns:
            yld = self._yield_data.copy()
            yld["date"] = pd.to_datetime(yld["date"])
            df["date"] = pd.to_datetime(df["date"])

            df = df.merge(yld[["date", "tnx", "fvx", "irx"]], on="date", how="left")

            # Forward-fill small gaps
            for c in ["tnx", "fvx", "irx"]:
                df[c] = df[c].fillna(method="ffill", limit=3).fillna(0.0)

            tnx = df["tnx"]
            fvx = df["fvx"]
            irx = df["irx"]

            df.drop(columns=["tnx", "fvx", "irx"], inplace=True)
        else:
            # Proxy: use moving-average ratios as yield-curve proxies
            ma_5 = close.rolling(5, min_periods=1).mean()
            ma_20 = close.rolling(20, min_periods=1).mean()
            ma_60 = close.rolling(60, min_periods=1).mean()

            # Scale to plausible yield range (0-6%)
            # "tnx" proxy ~ 20d/60d MA ratio deviation, scaled
            tnx = ((ma_20 / (ma_60 + 1e-10)) - 1.0) * 100.0
            fvx = ((ma_5 / (ma_20 + 1e-10)) - 1.0) * 100.0
            irx = ((close / (ma_5 + 1e-10)) - 1.0) * 100.0

        # ------ Compute features ------
        df["yc_2s10s_slope"] = (tnx - fvx).values
        df["yc_3m10y_slope"] = (tnx - irx).values
        df["yc_curvature"] = (tnx + irx - 2.0 * fvx).values

        # Slope momentum
        slope_3m10y = df["yc_3m10y_slope"]
        df["yc_slope_momentum_5d"] = slope_3m10y.diff(5)
        df["yc_slope_momentum_20d"] = slope_3m10y.diff(20)

        # Z-score
        roll_mean = slope_3m10y.rolling(60, min_periods=10).mean()
        roll_std = slope_3m10y.rolling(60, min_periods=10).std()
        df["yc_slope_z"] = ((slope_3m10y - roll_mean) / (roll_std + 1e-10)).clip(-4, 4)

        # Real yield proxy: z-score of tnx
        tnx_series = pd.Series(tnx.values if hasattr(tnx, "values") else tnx, index=df.index)
        tnx_mean = tnx_series.rolling(60, min_periods=10).mean()
        tnx_std = tnx_series.rolling(60, min_periods=10).std()
        df["yc_real_yield_proxy"] = ((tnx_series - tnx_mean) / (tnx_std + 1e-10)).clip(-4, 4)

        # Inversion flag
        df["yc_inversion_flag"] = np.where(df["yc_3m10y_slope"] < 0, 1.0, 0.0)

        # Steepening speed
        df["yc_steepening_speed"] = df["yc_curvature"].diff(20)

        # Regime
        slope = df["yc_3m10y_slope"]
        df["yc_regime"] = np.select(
            [slope > 1.5, slope >= 0.5, slope >= 0, slope < 0],
            [2.0, 1.0, 0.0, -1.0],
            default=0.0,
        )

        # ------ NaN / Inf cleanup ------
        for col in self._all_feature_names():
            df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        n_features = sum(1 for c in df.columns if c.startswith("yc_"))
        logger.info("YieldCurveFeatures: added %d features", n_features)
        return df

    # ─── Analysis ─────────────────────────────────────────────────────

    def analyze_current_yield_curve(self, df_daily: pd.DataFrame) -> Optional[Dict]:
        """Return current yield-curve regime info, or *None* if features absent."""
        if "yc_slope_z" not in df_daily.columns or len(df_daily) < 2:
            return None

        last = df_daily.iloc[-1]
        z = float(last.get("yc_slope_z", 0.0))
        slope = float(last.get("yc_3m10y_slope", 0.0))

        if slope > 1.5:
            regime = "STEEP"
        elif slope >= 0.5:
            regime = "NORMAL"
        elif slope >= 0:
            regime = "FLAT"
        else:
            regime = "INVERTED"

        return {
            "yield_regime": regime,
            "slope_z": round(z, 3),
            "inversion": bool(last.get("yc_inversion_flag", 0.0) > 0.5),
            "slope_3m10y": round(slope, 4),
        }

    # ─── Metadata ─────────────────────────────────────────────────────

    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "yc_2s10s_slope",
            "yc_3m10y_slope",
            "yc_curvature",
            "yc_slope_momentum_5d",
            "yc_slope_momentum_20d",
            "yc_slope_z",
            "yc_real_yield_proxy",
            "yc_inversion_flag",
            "yc_steepening_speed",
            "yc_regime",
        ]
