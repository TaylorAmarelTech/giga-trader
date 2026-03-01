"""
GIGA TRADER - Futures-Spot Basis Features
==========================================
Derive features from the basis spread between ES futures and the SPY spot price.

The futures-spot basis reflects cost-of-carry, supply/demand imbalances,
and institutional positioning in index futures. A positive basis (contango)
means futures trade at a premium to spot; a negative basis (backwardation)
signals near-term bearish pressure.

Primary source: E-mini S&P 500 futures (ES=F) via yfinance.
Fallback: Return-acceleration proxy using SPY close prices only
  (second difference of daily returns, which correlates with basis shifts).

Features generated (prefix: basis_):
  - basis_spread: Normalized futures-spot spread (or proxy return acceleration)
  - basis_spread_z: 60-day z-score of basis_spread
  - basis_change_5d: 5-day change in basis_spread
  - basis_regime: Categorical (-1=backwardation, 0=normal, 1=contango)
                  based on z-score thresholds of ±1.0
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger("FUTURES_BASIS")


class FuturesBasisFeatures:
    """
    Download ES futures data and create futures-spot basis features.

    Attempts to download ES=F (E-mini S&P 500 futures) via yfinance.
    Falls back to a return-acceleration proxy if download fails or data
    is unavailable — ensuring features are ALWAYS produced.

    Pattern: download → compute → merge (same as EconomicFeatures).
    """

    #: yfinance ticker for E-mini S&P 500 continuous futures
    FUTURES_TICKER = "ES=F"

    def __init__(self, z_window: int = 60) -> None:
        """
        Initialise FuturesBasisFeatures.

        Parameters
        ----------
        z_window:
            Rolling window (trading days) used for z-score normalisation.
            Default 60 (~3 months).
        """
        self.z_window = z_window
        self._futures_data: pd.DataFrame = pd.DataFrame()

    # ─── Download ─────────────────────────────────────────────────────────────

    def download_futures_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Attempt to download ES=F continuous futures data from yfinance.

        Stores result in ``self._futures_data``.
        Returns an empty DataFrame on any failure (graceful degradation).

        Parameters
        ----------
        start_date:
            Start of the desired date range.
        end_date:
            End of the desired date range.
        """
        print(f"\n[BASIS] Downloading {self.FUTURES_TICKER} futures data...")

        try:
            import yfinance as yf
        except ImportError:
            logger.warning("  [BASIS] yfinance not installed; using proxy")
            return pd.DataFrame()

        try:
            # Pull a bit of extra history so rolling windows warm up properly.
            dl_start = (pd.Timestamp(start_date) - pd.Timedelta(days=90)).strftime(
                "%Y-%m-%d"
            )
            dl_end = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).strftime(
                "%Y-%m-%d"
            )

            raw = yf.download(
                self.FUTURES_TICKER,
                start=dl_start,
                end=dl_end,
                auto_adjust=True,
                progress=False,
            )

            if raw.empty:
                logger.info("  [BASIS] No futures data returned by yfinance")
                return pd.DataFrame()

            # Flatten MultiIndex columns if present (yfinance ≥0.2)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)

            close_col = "Close" if "Close" in raw.columns else raw.columns[0]
            futures_close = raw[close_col].copy()
            futures_close.index = pd.to_datetime(futures_close.index).tz_localize(None)
            futures_close = futures_close.dropna()

            self._futures_data = pd.DataFrame(
                {"date": pd.to_datetime(futures_close.index.date), "es_close": futures_close.values}
            ).reset_index(drop=True)

            print(
                f"  [BASIS] {len(self._futures_data)} days of {self.FUTURES_TICKER} data loaded"
            )
            return self._futures_data

        except Exception as exc:
            logger.warning(f"  [BASIS] Futures download failed: {exc}")
            return pd.DataFrame()

    # ─── Feature creation ─────────────────────────────────────────────────────

    def create_futures_basis_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create the four futures-spot basis features and merge into *df*.

        If ES futures data was successfully downloaded and can be merged on
        the ``date`` column, the true basis spread is computed:

            basis_spread = (es_close - spot_close) / spot_close

        Otherwise the return-acceleration proxy is used:

            basis_spread = ret_today - ret_yesterday
                         = (close/close.shift(1) - 1) - (close.shift(1)/close.shift(2) - 1)

        Parameters
        ----------
        df:
            SPY daily DataFrame.  Must contain a ``close`` column.
            May optionally contain a ``date`` column (Timestamp) used for
            merging futures data.
        """
        if "close" not in df.columns:
            logger.warning(
                "  [BASIS] 'close' column not found — returning df unchanged"
            )
            return df

        features = df.copy()
        basis_series = self._compute_basis_spread(features)

        # ── Derived features ──────────────────────────────────────────────────

        # 60-day z-score (min_periods capped at z_window to allow custom small windows)
        min_p = min(10, self.z_window)
        roll_mean = basis_series.rolling(self.z_window, min_periods=min_p).mean()
        roll_std = basis_series.rolling(self.z_window, min_periods=min_p).std()
        z_series = np.where(
            roll_std > 1e-10,
            (basis_series - roll_mean) / roll_std,
            0.0,
        )
        z_series = pd.Series(z_series, index=basis_series.index)

        # 5-day change
        change_5d = basis_series.diff(5)

        # Regime: contango (+1), backwardation (-1), normal (0)
        regime = pd.Series(0, index=z_series.index, dtype=float)
        regime[z_series > 1.0] = 1.0
        regime[z_series < -1.0] = -1.0

        # ── Assign to output ──────────────────────────────────────────────────
        features["basis_spread"] = basis_series.values
        features["basis_spread_z"] = z_series.values
        features["basis_change_5d"] = change_5d.values
        features["basis_regime"] = regime.values

        # Fill NaN with 0
        basis_cols = ["basis_spread", "basis_spread_z", "basis_change_5d", "basis_regime"]
        features[basis_cols] = features[basis_cols].fillna(0)

        print(f"  [BASIS] Added {len(basis_cols)} futures-spot basis features")
        return features

    # ─── Current-state analysis ────────────────────────────────────────────────

    def analyze_current_basis(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Return a summary of the current futures-spot basis conditions.

        Parameters
        ----------
        df:
            A DataFrame that has already been processed by
            :meth:`create_futures_basis_features` (i.e., contains the
            ``basis_*`` columns).

        Returns
        -------
        dict or None
            ``None`` if the required columns are absent or the DataFrame is empty.
        """
        required = {"basis_spread", "basis_spread_z", "basis_regime"}
        if not required.issubset(df.columns) or df.empty:
            return None

        latest = df.iloc[-1]
        regime_val = float(latest["basis_regime"])

        if regime_val > 0:
            regime_label = "CONTANGO"
        elif regime_val < 0:
            regime_label = "BACKWARDATION"
        else:
            regime_label = "NORMAL"

        return {
            "basis_spread": float(latest["basis_spread"]),
            "basis_spread_z": float(latest["basis_spread_z"]),
            "basis_change_5d": float(latest.get("basis_change_5d", 0.0)),
            "basis_regime": regime_label,
            "is_extreme": abs(float(latest["basis_spread_z"])) > 1.0,
        }

    # ─── Private helpers ──────────────────────────────────────────────────────

    def _compute_basis_spread(self, df: pd.DataFrame) -> pd.Series:
        """
        Compute the basis spread time series for *df*.

        Uses real ES futures data if available and mergeable on ``date``;
        otherwise falls back to the return-acceleration proxy.
        """
        close = df["close"].copy().reset_index(drop=True)

        if not self._futures_data.empty and "date" in df.columns:
            basis = self._merge_futures_basis(df, close)
            if basis is not None:
                return basis

        # Proxy: second difference of returns (return acceleration)
        ret = close / close.shift(1) - 1
        proxy = ret - ret.shift(1)
        return proxy.reset_index(drop=True)

    def _merge_futures_basis(
        self, df: pd.DataFrame, spot_close: pd.Series
    ) -> Optional[pd.Series]:
        """
        Attempt to compute true basis by merging ES futures closes with spot.

        Returns ``None`` if the merge yields fewer than 30 valid rows.
        """
        try:
            futures = self._futures_data.copy()
            futures["date"] = pd.to_datetime(futures["date"])

            spy_dates = pd.to_datetime(df["date"]).reset_index(drop=True)
            tmp = pd.DataFrame({"date": spy_dates, "spot_close": spot_close})
            merged = tmp.merge(futures, on="date", how="left")

            valid = merged["es_close"].notna().sum()
            if valid < 30:
                logger.info(
                    f"  [BASIS] Only {valid} matching dates with futures data; using proxy"
                )
                return None

            basis = (merged["es_close"] - merged["spot_close"]) / (
                merged["spot_close"] + 1e-10
            )
            basis = basis.reset_index(drop=True)
            logger.info(
                f"  [BASIS] Using real futures basis ({valid} matching dates)"
            )
            return basis

        except Exception as exc:
            logger.warning(f"  [BASIS] Futures merge failed: {exc}; using proxy")
            return None
