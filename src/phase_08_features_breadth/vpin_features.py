"""
VPIN (Volume-Synchronized Probability of Informed Trading) Features
====================================================================
Easley et al. (2012) VPIN measures order-flow toxicity via Bulk Volume
Classification (BVC). High VPIN signals informed trading activity, which
often precedes large price moves and liquidity crises.

BVC approach (no tick data required):
  - Classify each day's volume as buy/sell based on price-change direction
    and magnitude relative to recent return volatility (via normal CDF).
  - Compute order imbalance (|buy_vol - sell_vol| / total_vol) over a
    rolling window of n_buckets bars.

Features generated (4, prefix: vpin_):
  vpin_value      — VPIN estimate (0-1); higher = more informed trading
  vpin_z          — 60-day z-score of vpin_value, clipped to [-4, 4]
  vpin_regime     — Categorical (-1=low toxicity, 0=normal, 1=high toxicity)
                    based on z-score thresholds of ±1.5
  vpin_change_5d  — 5-day change in vpin_value
"""

import logging
from typing import Dict, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)

# Default parameters
_DEFAULT_WINDOW = 50         # rolling window for VPIN (n_buckets)
_DEFAULT_VOL_WINDOW = 20     # window for return-std estimation (BVC denominator)
_VOL_MIN_PERIODS = 5
_Z_WINDOW = 60
_Z_MIN_PERIODS = 20
_Z_CLIP = 4.0
_REGIME_THRESHOLD = 1.5      # z-score boundary for regime classification


class VPINFeatures:
    """
    Compute VPIN (Bulk Volume Classification) features from daily OHLCV data.

    Parameters
    ----------
    window : int
        Rolling bucket window for VPIN calculation (default 50).
    n_buckets : int
        Alias for window; the number of trading days over which order
        imbalance is averaged.  If both are provided, ``window`` wins.
    """

    REQUIRED_COLS = {"close", "volume"}

    def __init__(self, window: int = 50, n_buckets: int = 50) -> None:
        self.window = window
        self.n_buckets = n_buckets  # kept for API symmetry

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def download_vpin_data(
        self,
        start_date,
        end_date,
    ) -> pd.DataFrame:
        """
        No external download needed — VPIN is computed from price/volume
        data that is already present in the pipeline.

        Returns an empty DataFrame (consistent interface with other feature
        classes that do download external data).
        """
        logger.info("VPINFeatures: no external download required — uses existing OHLCV data")
        return pd.DataFrame()

    def create_vpin_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add VPIN features to *df* in-place (returns a copy).

        Parameters
        ----------
        df : pd.DataFrame
            Daily OHLCV data.  Must contain at least ``close`` and
            ``volume``.  If ``high`` / ``low`` are also present they are
            used to refine the BVC classification.

        Returns
        -------
        pd.DataFrame
            Original *df* with 4 new ``vpin_`` columns appended.
            Returns *df* unchanged if required columns are missing.
        """
        result = df.copy()

        # Guard: require close + volume
        if "close" not in result.columns:
            logger.warning("VPINFeatures: 'close' column missing — skipping")
            return result

        if "volume" not in result.columns or result["volume"].sum() == 0:
            logger.info("VPINFeatures: no volume data — defaulting to 0.0")
            for col in self._feature_names():
                result[col] = 0.0
            return result

        # --- Step 1: Estimate daily return std (BVC denominator) ----------
        close = result["close"].astype(float)
        volume = result["volume"].astype(float)

        daily_return = close.pct_change()

        # Use high/low range for a better volatility estimate when available
        if "high" in result.columns and "low" in result.columns:
            high = result["high"].astype(float)
            low = result["low"].astype(float)
            # Parkinson estimator: range / (2 * sqrt(ln2))
            hl_range = (np.log(high / (low + 1e-10))).abs() / (2.0 * np.sqrt(np.log(2.0)))
            rolling_std = hl_range.rolling(_DEFAULT_VOL_WINDOW, min_periods=_VOL_MIN_PERIODS).mean()
        else:
            rolling_std = daily_return.rolling(
                _DEFAULT_VOL_WINDOW, min_periods=_VOL_MIN_PERIODS
            ).std()

        # Replace zeros / NaNs with a global std fallback
        global_std = daily_return.std()
        if pd.isna(global_std) or global_std < 1e-10:
            global_std = 0.01
        rolling_std = rolling_std.fillna(global_std).clip(lower=1e-10)

        # --- Step 2: BVC — classify volume as buy vs sell ----------------
        # buy_volume = volume * Phi(delta_price / sigma)
        # sell_volume = volume * (1 - Phi(delta_price / sigma))
        # where delta_price = (close_t - close_{t-1}) / close_{t-1}
        z_score_bvc = daily_return / rolling_std
        phi = pd.Series(norm.cdf(z_score_bvc.fillna(0.0).values), index=result.index)

        buy_vol = volume * phi
        sell_vol = volume * (1.0 - phi)

        # --- Step 3: VPIN over rolling n_buckets window ------------------
        imbalance = (buy_vol - sell_vol).abs()
        total_vol = volume

        # Rolling sum of |buy - sell| / rolling sum of total
        roll_imbalance = imbalance.rolling(self.window, min_periods=max(5, self.window // 5)).sum()
        roll_total = total_vol.rolling(self.window, min_periods=max(5, self.window // 5)).sum()

        vpin_value = roll_imbalance / (roll_total + 1e-10)
        vpin_value = vpin_value.clip(0.0, 1.0)

        result["vpin_value"] = vpin_value

        # --- Step 4: Derived features ------------------------------------
        # 5-day change
        result["vpin_change_5d"] = result["vpin_value"].diff(5)

        # 60-day z-score
        roll_mean = result["vpin_value"].rolling(_Z_WINDOW, min_periods=_Z_MIN_PERIODS).mean()
        roll_std = result["vpin_value"].rolling(_Z_WINDOW, min_periods=_Z_MIN_PERIODS).std()
        result["vpin_z"] = (
            (result["vpin_value"] - roll_mean) / (roll_std + 1e-10)
        ).clip(-_Z_CLIP, _Z_CLIP)

        # Regime: high toxicity (+1), normal (0), low toxicity (-1)
        z = result["vpin_z"]
        result["vpin_regime"] = np.where(
            z > _REGIME_THRESHOLD, 1.0,
            np.where(z < -_REGIME_THRESHOLD, -1.0, 0.0),
        )

        # --- Cleanup: NaN -> 0.0, no infinities -------------------------
        for col in self._feature_names():
            result[col] = result[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        n_features = sum(1 for c in result.columns if c.startswith("vpin_"))
        logger.info(f"VPINFeatures: added {n_features} features")
        return result

    def analyze_current_vpin(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Return a summary of the latest VPIN reading for dashboard display.

        Parameters
        ----------
        df : pd.DataFrame
            Daily data that already has ``vpin_`` columns (produced by
            :meth:`create_vpin_features`).

        Returns
        -------
        dict or None
            None if features are missing or the DataFrame has fewer than
            two rows.
        """
        if "vpin_value" not in df.columns or len(df) < 2:
            return None

        last = df.iloc[-1]
        z = float(last.get("vpin_z", 0.0))
        vpin = float(last.get("vpin_value", 0.0))
        regime_val = float(last.get("vpin_regime", 0.0))

        if regime_val > 0:
            regime_label = "HIGH_TOXICITY"
        elif regime_val < 0:
            regime_label = "LOW_TOXICITY"
        else:
            regime_label = "NORMAL"

        return {
            "vpin_regime": regime_label,
            "vpin_value": round(vpin, 4),
            "vpin_z": round(z, 3),
            "vpin_change_5d": round(float(last.get("vpin_change_5d", 0.0)), 4),
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _feature_names():
        return ["vpin_value", "vpin_z", "vpin_regime", "vpin_change_5d"]
