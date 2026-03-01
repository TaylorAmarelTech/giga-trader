"""
GIGA TRADER - SEC EDGAR Insider Aggregate Features
====================================================
Compute proxy insider-aggregate features from price and volume data.

SEC EDGAR Form 4 filings (insider transactions) are publicly available
but require parsing XML from full-text search. As a practical proxy, we
use price-and-volume signals that correlate with institutional accumulation:

  - Price above 20-day MA with above-average volume = "informed buying"
  - Rolling density of such days = insider-like accumulation pressure
  - Cluster threshold (>60%) = intense, concentrated buying period
  - Z-score = normalised deviation from recent baseline

No external API required.  All computations use `close` and `volume`.

Features generated (prefix: insider_agg_):
  - insider_agg_buy_ratio : rolling 30-day fraction of "accumulation" days
  - insider_agg_volume    : rolling 30-day fraction of above-average-volume days
  - insider_agg_cluster   : binary flag: buy_ratio > 0.60 → cluster buying
  - insider_agg_z         : 60-day z-score of buy_ratio (normalised pressure)
"""

import logging
import warnings
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger("INSIDER_AGG_FEATURES")

# ─── Thresholds ───────────────────────────────────────────────────────────────

_CLUSTER_THRESHOLD: float = 0.60   # buy_ratio above this → cluster flag
_MA_WINDOW: int = 20               # moving average for price + volume baseline
_Z_CLIP: float = 3.0               # symmetric clip on z-score


class InsiderAggregateFeatures:
    """
    Compute insider-aggregate proxy features from close and volume data.

    No external downloads are required.  The ``download_insider_data``
    method exists to satisfy the standard module interface but always
    returns an empty DataFrame (data is derived directly from OHLCV).

    Usage
    -----
    >>> engine = InsiderAggregateFeatures()
    >>> engine.download_insider_data(start, end)   # no-op, returns empty df
    >>> result = engine.create_insider_aggregate_features(spy_daily_df)
    """

    def __init__(self, window: int = 30, z_window: int = 60) -> None:
        """
        Parameters
        ----------
        window : int
            Rolling window (trading days) used for buy_ratio and volume.
            Default 30.
        z_window : int
            Rolling window used for the z-score of buy_ratio.  Default 60.
        """
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}")
        if z_window < 1:
            raise ValueError(f"z_window must be >= 1, got {z_window}")

        self.window: int = window
        self.z_window: int = z_window

    # ──────────────────────────────────────────────────────────────────────────
    # Download interface (no-op for this module)
    # ──────────────────────────────────────────────────────────────────────────

    def download_insider_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Stub download method.

        SEC EDGAR XML parsing is out-of-scope; features are derived from
        price/volume data that is already available in the pipeline.

        Returns
        -------
        pd.DataFrame
            Always empty.
        """
        logger.debug(
            "[INSIDER_AGG] download_insider_data is a no-op — "
            "features computed from close/volume directly"
        )
        return pd.DataFrame()

    # ──────────────────────────────────────────────────────────────────────────
    # Feature engineering
    # ──────────────────────────────────────────────────────────────────────────

    def create_insider_aggregate_features(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add four insider_agg_ features to *df* and return the result.

        Required columns: ``close``, ``volume``.
        If either column is absent the input DataFrame is returned unchanged.

        Parameters
        ----------
        df : pd.DataFrame
            Daily bar DataFrame.  Must contain 'close' and 'volume'.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with four additional columns (or *df* unchanged if
            the required columns are missing).
        """
        # Guard: required columns
        required = {"close", "volume"}
        missing = required - set(df.columns)
        if missing:
            logger.warning(
                f"[INSIDER_AGG] Missing required columns: {missing}. "
                "Returning df unchanged."
            )
            return df

        if df.empty:
            return df

        print("\n[INSIDER_AGG] Engineering insider aggregate features...")

        out = df.copy()

        close = pd.to_numeric(out["close"], errors="coerce")
        volume = pd.to_numeric(out["volume"], errors="coerce")

        # ── Step 1: rolling baselines ──────────────────────────────────────
        close_ma20 = close.rolling(_MA_WINDOW, min_periods=1).mean()
        volume_ma20 = volume.rolling(_MA_WINDOW, min_periods=1).mean()

        # ── Step 2: accumulation flag ──────────────────────────────────────
        # 1 when price is above its 20-day MA AND volume is above average
        accumulation_day = (
            (close > close_ma20) & (volume > volume_ma20)
        ).astype(float)

        # above-average-volume flag (used independently for insider_agg_volume)
        high_volume_day = (volume > volume_ma20).astype(float)

        # ── Step 3: rolling 30-day counts → fractions ─────────────────────
        buy_ratio = (
            accumulation_day
            .rolling(self.window, min_periods=1)
            .sum()
            .div(self.window)
        )

        volume_frac = (
            high_volume_day
            .rolling(self.window, min_periods=1)
            .sum()
            .div(self.window)
        )

        # ── Step 4: cluster flag ───────────────────────────────────────────
        cluster = (buy_ratio > _CLUSTER_THRESHOLD).astype(float)

        # ── Step 5: z-score of buy_ratio over z_window ────────────────────
        roll_mean = buy_ratio.rolling(self.z_window, min_periods=2).mean()
        roll_std = buy_ratio.rolling(self.z_window, min_periods=2).std()

        z_score = np.where(
            roll_std > 1e-9,
            (buy_ratio - roll_mean) / roll_std,
            0.0,
        )
        z_score = np.clip(z_score, -_Z_CLIP, _Z_CLIP)

        # ── Assign ────────────────────────────────────────────────────────
        out["insider_agg_buy_ratio"] = buy_ratio.fillna(0.0).values
        out["insider_agg_volume"] = volume_frac.fillna(0.0).values
        out["insider_agg_cluster"] = cluster.fillna(0.0).values
        out["insider_agg_z"] = pd.Series(z_score, index=out.index).fillna(0.0)

        n_features = 4
        print(f"  [INSIDER_AGG] Added {n_features} insider aggregate features")
        return out

    # ──────────────────────────────────────────────────────────────────────────
    # Analysis helper
    # ──────────────────────────────────────────────────────────────────────────

    def analyze_current_insider(
        self, df: pd.DataFrame
    ) -> Optional[Dict]:
        """
        Return a summary dict characterising the latest insider-aggregate
        regime, suitable for dashboard display.

        Parameters
        ----------
        df : pd.DataFrame
            Output of :meth:`create_insider_aggregate_features`.

        Returns
        -------
        dict or None
            ``None`` when the expected feature columns are absent.

        Regime labels
        -------------
        ACCUMULATION  buy_ratio > 0.50 (majority of days are accumulation days)
        DISTRIBUTION  buy_ratio < 0.25 (concentration of selling / weakness)
        NEUTRAL       otherwise
        """
        insider_cols = [c for c in df.columns if c.startswith("insider_agg_")]
        if not insider_cols or df.empty:
            return None

        latest = df.iloc[-1]

        buy_ratio = float(latest.get("insider_agg_buy_ratio", 0.0))
        volume_frac = float(latest.get("insider_agg_volume", 0.0))
        cluster = bool(latest.get("insider_agg_cluster", 0.0))
        z_score = float(latest.get("insider_agg_z", 0.0))

        if buy_ratio > 0.50:
            regime = "ACCUMULATION"
        elif buy_ratio < 0.25:
            regime = "DISTRIBUTION"
        else:
            regime = "NEUTRAL"

        return {
            "insider_regime": regime,
            "buy_ratio": buy_ratio,
            "volume_fraction": volume_frac,
            "cluster_buying": cluster,
            "buy_ratio_z": z_score,
        }
