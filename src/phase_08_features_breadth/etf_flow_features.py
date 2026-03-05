"""
GIGA TRADER - ETF Fund Flow Features
======================================
Volume-price divergence proxies for ETF creation/redemption activity.

Since direct ETF flow data (from Bloomberg or similar) is expensive, we derive
flow proxies entirely from SPY's own close/volume columns — no external API needed.

Theory:
  - Large ETF creation/redemption units (50,000-share blocks) drive volume.
  - These institutional transactions leave a characteristic footprint:
      high volume with small price movement (APs arbitrage the NAV spread).
  - Conversely, genuine directional demand shows large price movement relative
    to volume.  The divergence between volume change and price movement is
    therefore an indirect flow signal.

Features generated (prefix: etf_flow_):
  - etf_flow_spy_20d         : 20-day rolling mean of the volume-price divergence
                               proxy.  High = elevated flow activity (creation or
                               redemption); low = price-driven, thin-flow market.
  - etf_flow_spy_z           : 60-day z-score of the proxy (standardises the signal
                               across different vol regimes).
  - etf_flow_creation_redemption : Rolling 20-day fraction of days where volume is
                               unusually high (>1.5× 20d avg) AND price barely moved
                               (<0.3%).  These are the hallmark creation/redemption
                               days in the SPY basket.
  - etf_flow_short_interest_ratio : Rolling 20-day fraction of heavy-selling days
                               (return < -0.5% AND volume > 20d MA), used as a
                               proxy for growing short interest / redemption pressure.

Required columns in the input DataFrame: close, volume (at minimum).
"""

import logging
import warnings
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.core.feature_base import FeatureModuleBase

warnings.filterwarnings("ignore")

logger = logging.getLogger("ETF_FLOW_FEATURES")


class ETFFlowFeatures(FeatureModuleBase):
    """
    Engineer ETF fund-flow proxy features from SPY close and volume data.

    All four features are computed purely from columns already present in the
    training DataFrame — no yfinance downloads or API keys required.

    Pattern: construct → create_etf_flow_features → merge (same as DarkPoolFeatures).
    """

    FEATURE_NAMES = [
        "etf_flow_spy_20d",
        "etf_flow_spy_z",
        "etf_flow_creation_redemption",
        "etf_flow_short_interest_ratio",
    ]

    def __init__(
        self,
        flow_window: int = 20,
        z_window: int = 60,
    ) -> None:
        """
        Parameters
        ----------
        flow_window : int
            Rolling window (business days) for the flow proxy mean and the
            creation/redemption and short-interest fraction.  Default 20.
        z_window : int
            Rolling window (business days) for the z-score normalisation of the
            flow proxy.  Default 60.
        """
        self.flow_window = flow_window
        self.z_window = z_window

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_etf_flow_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Placeholder: direct ETF flow data is not freely available.

        All features are derived from close/volume already present in the
        training DataFrame.  This method exists to satisfy the standard
        phase_08 interface and always returns an empty DataFrame.

        Parameters
        ----------
        start_date, end_date : datetime
            Date range (ignored).

        Returns
        -------
        pd.DataFrame
            Always empty.
        """
        logger.info(
            "[ETF_FLOW] No external download required — features computed from "
            "close/volume columns."
        )
        return pd.DataFrame()

    def create_etf_flow_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute all four ETF flow proxy features and append them to *df*.

        Operates entirely on the ``close`` and ``volume`` columns already in
        *df*.  Missing columns are handled gracefully — the original DataFrame
        is returned unchanged.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain ``close`` and ``volume`` columns.

        Returns
        -------
        pd.DataFrame
            Original DataFrame with four new ``etf_flow_*`` columns appended.
            NaN values in new columns are filled with 0.
        """
        required = {"close", "volume"}
        missing = required - set(df.columns)
        if missing:
            logger.warning(
                "[ETF_FLOW] Missing required columns %s — returning df unchanged.", missing
            )
            return df

        if df.empty:
            return df

        print("\n[ETF_FLOW] Engineering ETF fund-flow proxy features...")

        out = df.copy()

        close = out["close"].astype(float)
        volume = out["volume"].astype(float)

        # ── Daily return ────────────────────────────────────────────────────
        daily_return = close.pct_change()

        # ── Volume change % ─────────────────────────────────────────────────
        volume_change_pct = volume.pct_change()

        # ── 20-day volume moving average ─────────────────────────────────────
        vol_ma_20 = volume.rolling(self.flow_window, min_periods=max(1, self.flow_window // 2)).mean()

        # ─────────────────────────────────────────────────────────────────────
        # Feature 1: etf_flow_spy_20d
        #   Flow proxy = (volume_change_pct - |return|) / (|return| + 0.001)
        #   High positive values → large volume change relative to price move
        #   (consistent with creation/redemption flows).
        # ─────────────────────────────────────────────────────────────────────
        abs_return = daily_return.abs()
        flow_proxy_raw = (volume_change_pct - abs_return) / (abs_return + 0.001)

        out["etf_flow_spy_20d"] = (
            flow_proxy_raw
            .rolling(self.flow_window, min_periods=max(1, self.flow_window // 2))
            .mean()
        )

        # ─────────────────────────────────────────────────────────────────────
        # Feature 2: etf_flow_spy_z
        #   60-day z-score of the raw flow proxy (not the 20d mean), so the
        #   z-score captures the instantaneous signal relative to recent history.
        # ─────────────────────────────────────────────────────────────────────
        roll_mean = flow_proxy_raw.rolling(
            self.z_window, min_periods=max(1, self.z_window // 4)
        ).mean()
        roll_std = flow_proxy_raw.rolling(
            self.z_window, min_periods=max(1, self.z_window // 4)
        ).std()
        out["etf_flow_spy_z"] = np.where(
            roll_std > 1e-8,
            (flow_proxy_raw - roll_mean) / roll_std,
            0.0,
        )

        # ─────────────────────────────────────────────────────────────────────
        # Feature 3: etf_flow_creation_redemption
        #   Binary flag: 1 if volume > 1.5× 20d MA AND |return| < 0.3%.
        #   This day pattern strongly suggests creation/redemption basket activity.
        #   The feature is the 20-day rolling mean of these binary flags,
        #   i.e. fraction of recent days with such activity.
        # ─────────────────────────────────────────────────────────────────────
        high_volume = volume > (1.5 * vol_ma_20)
        small_move = abs_return < 0.003  # 0.3% threshold

        cr_flag = (high_volume & small_move).astype(float)
        out["etf_flow_creation_redemption"] = (
            cr_flag
            .rolling(self.flow_window, min_periods=max(1, self.flow_window // 2))
            .mean()
        )

        # ─────────────────────────────────────────────────────────────────────
        # Feature 4: etf_flow_short_interest_ratio
        #   Fraction of recent days with heavy selling pressure:
        #   return < -0.5% AND volume > 20d MA volume.
        #   Rolling 20-day count / 20 → range [0, 1].
        # ─────────────────────────────────────────────────────────────────────
        heavy_sell = (daily_return < -0.005) & (volume > vol_ma_20)
        out["etf_flow_short_interest_ratio"] = (
            heavy_sell.astype(float)
            .rolling(self.flow_window, min_periods=max(1, self.flow_window // 2))
            .mean()
        )

        # ── Fill NaN with 0 ─────────────────────────────────────────────────
        etf_cols = [c for c in out.columns if c.startswith("etf_flow_")]
        out[etf_cols] = out[etf_cols].fillna(0)

        n_added = len(etf_cols)
        print(f"  [ETF_FLOW] Added {n_added} ETF flow proxy features")
        return out

    def analyze_current_flows(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Summarise the most recent ETF flow conditions for dashboard display.

        Parameters
        ----------
        df : pd.DataFrame
            Output of :meth:`create_etf_flow_features` (must contain the
            four ``etf_flow_*`` columns).

        Returns
        -------
        dict or None
            Dictionary with keys:
              - ``flow_proxy``       : latest etf_flow_spy_20d value
              - ``flow_z``           : latest etf_flow_spy_z value
              - ``cr_fraction``      : latest creation/redemption fraction
              - ``short_fraction``   : latest short interest fraction
              - ``flow_regime``      : "INFLOW" | "NEUTRAL" | "OUTFLOW"
            Returns None if required columns are absent or df is empty.
        """
        etf_cols = [c for c in df.columns if c.startswith("etf_flow_")]
        if not etf_cols or df.empty:
            return None

        latest = df.iloc[-1]

        flow_proxy = float(latest.get("etf_flow_spy_20d", 0.0))
        flow_z = float(latest.get("etf_flow_spy_z", 0.0))
        cr_frac = float(latest.get("etf_flow_creation_redemption", 0.0))
        short_frac = float(latest.get("etf_flow_short_interest_ratio", 0.0))

        # Regime classification:
        #  INFLOW  : high flow proxy z-score AND low short pressure
        #             → creation activity dominates (net buying of SPY basket)
        #  OUTFLOW : high short pressure AND/OR negative flow proxy
        #             → redemption / short selling dominates
        #  NEUTRAL : mixed or ambiguous signals
        if flow_z > 0.5 and short_frac < 0.15:
            flow_regime = "INFLOW"
        elif flow_z < -0.5 or short_frac > 0.25:
            flow_regime = "OUTFLOW"
        else:
            flow_regime = "NEUTRAL"

        return {
            "flow_proxy": flow_proxy,
            "flow_z": flow_z,
            "cr_fraction": cr_frac,
            "short_fraction": short_frac,
            "flow_regime": flow_regime,
        }
