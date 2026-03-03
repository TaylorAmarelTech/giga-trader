"""
GIGA TRADER - Multi-Horizon Ensemble Filter
=============================================
Requires agreement across 1d/3d/5d prediction horizons before
allowing a signal through. Prevents trading against the longer
timeframe trend.

Features (11, prefix mh_):
  mh_return_1d              -- Lagged 1-day return
  mh_return_3d              -- Lagged 3-day return
  mh_return_5d              -- Lagged 5-day return
  mh_direction_1d           -- 1 if positive, -1 if negative (lagged)
  mh_direction_3d           -- Same for 3d
  mh_direction_5d           -- Same for 5d
  mh_momentum_1d            -- Rolling 1d momentum strength
  mh_momentum_3d            -- Rolling 3d momentum strength
  mh_momentum_5d            -- Rolling 5d momentum strength
  mh_agreement              -- Fraction of horizons agreeing (-1 to 1)
  mh_strength               -- Average absolute momentum across horizons
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MultiHorizonFilter:
    """Filter signals that lack multi-horizon agreement.

    Computes returns and momentum across multiple horizons (default 1/3/5 days)
    and blocks signals where shorter and longer timeframes disagree.

    All features are LAGGED to avoid look-ahead bias.

    Parameters
    ----------
    horizons : list of int, optional
        Prediction horizons in days (default [1, 3, 5]).
    agreement_threshold : float
        Minimum agreement fraction to pass filter (default 0.6).
    min_horizons : int
        Minimum number of horizons that must agree (default 2).
    """

    def __init__(
        self,
        horizons: Optional[List[int]] = None,
        agreement_threshold: float = 0.6,
        min_horizons: int = 2,
    ):
        self.horizons = horizons or [1, 3, 5]
        self.agreement_threshold = agreement_threshold
        self.min_horizons = min_horizons

    def compute_horizon_signals(
        self,
        df_daily: pd.DataFrame,
        close_col: str = "close",
    ) -> pd.DataFrame:
        """Compute multi-horizon features.

        All returns/directions are LAGGED (looking backward, not forward)
        to avoid look-ahead bias.

        Parameters
        ----------
        df_daily : pd.DataFrame
            Daily data with close prices.
        close_col : str
            Column name for close price.

        Returns
        -------
        DataFrame with added mh_ features.
        """
        df = df_daily.copy()

        if close_col not in df.columns:
            logger.warning(f"MultiHorizonFilter: '{close_col}' not found, zero-filling")
            for h in self.horizons:
                df[f"mh_return_{h}d"] = 0.0
                df[f"mh_direction_{h}d"] = 0.0
                df[f"mh_momentum_{h}d"] = 0.0
            df["mh_agreement"] = 0.0
            df["mh_strength"] = 0.0
            return df

        close = df[close_col].astype(float)

        direction_cols = []
        momentum_cols = []

        for h in self.horizons:
            # Lagged return (backward-looking)
            ret_col = f"mh_return_{h}d"
            df[ret_col] = close.pct_change(h)

            # Direction
            dir_col = f"mh_direction_{h}d"
            df[dir_col] = np.sign(df[ret_col])
            direction_cols.append(dir_col)

            # Momentum strength (rolling mean of absolute returns)
            mom_col = f"mh_momentum_{h}d"
            df[mom_col] = df[ret_col].abs().rolling(h * 2).mean()
            momentum_cols.append(mom_col)

        # Agreement: mean of directions (-1 to 1)
        # +1 means all horizons agree on up, -1 all agree on down
        df["mh_agreement"] = df[direction_cols].mean(axis=1)

        # Strength: average absolute momentum
        df["mh_strength"] = df[momentum_cols].mean(axis=1)

        # Fill NaN
        mh_cols = [c for c in df.columns if c.startswith("mh_")]
        for col in mh_cols:
            df[col] = df[col].fillna(0.0)

        return df

    def should_filter(
        self,
        agreement: float,
        direction: str = "LONG",
    ) -> bool:
        """Check if a signal should be filtered (blocked).

        Parameters
        ----------
        agreement : float
            Multi-horizon agreement value (-1 to 1).
        direction : str
            Proposed trade direction ("LONG" or "SHORT").

        Returns
        -------
        bool
            True if signal should be BLOCKED (horizons disagree).
        """
        if direction.upper() == "LONG":
            # For LONG, need positive agreement
            return agreement < self.agreement_threshold
        else:
            # For SHORT, need negative agreement
            return agreement > -self.agreement_threshold

    def evaluate(
        self,
        df_daily: pd.DataFrame,
        direction: str = "LONG",
    ) -> Dict:
        """Full evaluation: compute features and check filter.

        Parameters
        ----------
        df_daily : pd.DataFrame
            Daily data.
        direction : str
            Proposed direction.

        Returns
        -------
        dict with agreement, strength, should_block, horizon_details.
        """
        if df_daily.empty:
            return {
                "agreement": 0.0,
                "strength": 0.0,
                "should_block": True,
                "reason": "empty data",
                "horizon_details": {},
            }

        last = df_daily.iloc[-1]
        agreement = float(last.get("mh_agreement", 0.0))
        strength = float(last.get("mh_strength", 0.0))
        should_block = self.should_filter(agreement, direction)

        details = {}
        for h in self.horizons:
            details[f"{h}d"] = {
                "return": round(float(last.get(f"mh_return_{h}d", 0)), 4),
                "direction": float(last.get(f"mh_direction_{h}d", 0)),
                "momentum": round(float(last.get(f"mh_momentum_{h}d", 0)), 4),
            }

        reason = None
        if should_block:
            reason = (
                f"Multi-horizon disagreement for {direction}: "
                f"agreement={agreement:.2f} (threshold={self.agreement_threshold})"
            )
            logger.info(f"MultiHorizonFilter: BLOCKED - {reason}")

        return {
            "agreement": round(agreement, 3),
            "strength": round(strength, 4),
            "should_block": should_block,
            "reason": reason,
            "horizon_details": details,
        }
