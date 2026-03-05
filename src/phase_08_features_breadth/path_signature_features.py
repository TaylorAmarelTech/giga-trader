"""
Path Signature Features — iterated integrals of price/volume paths.

Path signatures capture the order and interaction of events in a
multivariate time series via iterated integrals.  Level-1 signatures
are net increments; level-2 signatures encode lead-lag (cross-area)
between return and volume channels.  All computations use pure numpy.

Features (17, prefix psig_):
  psig_sig1_5d          — Level-1 signature (net return) over 5-day window
  psig_sig1_10d         — Level-1 signature (net return) over 10-day window
  psig_sig1_20d         — Level-1 signature (net return) over 20-day window
  psig_sig2_5d          — Level-2 lead-lag area (return x volume_ratio) over 5-day window
  psig_sig2_10d         — Level-2 lead-lag area over 10-day window
  psig_sig2_20d         — Level-2 lead-lag area over 20-day window
  psig_sig2_anti_5d     — Anti-symmetric level-2 (asymmetry) over 5-day window
  psig_sig2_anti_10d    — Anti-symmetric level-2 over 10-day window
  psig_sig2_anti_20d    — Anti-symmetric level-2 over 20-day window
  psig_logsig1_5d       — Log-signature level 1 over 5-day window
  psig_logsig1_10d      — Log-signature level 1 over 10-day window
  psig_logsig1_20d      — Log-signature level 1 over 20-day window
  psig_path_length_5d   — Total variation (sum |increments|) over 5-day window
  psig_path_length_10d  — Total variation over 10-day window
  psig_path_length_20d  — Total variation over 20-day window
  psig_sig_ratio        — sig2_20d / (sig2_5d + 1e-10) — multi-scale ratio
  psig_momentum_asym    — sig2_anti_20d z-scored vs 60-day mean/std
"""

import logging
from typing import List

import numpy as np
import pandas as pd
from src.core.feature_base import FeatureModuleBase

logger = logging.getLogger(__name__)


class PathSignatureFeatures(FeatureModuleBase):
    """Compute path signature features from daily OHLCV data."""
    FEATURE_NAMES = ["psig_sig_ratio", "psig_momentum_asym"]


    REQUIRED_COLS = {"close"}

    def create_path_signature_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Add path signature features to df_daily.

        Parameters
        ----------
        df_daily : pd.DataFrame
            Must have 'close' column.  If 'volume' is present, 2D
            path (return, volume_ratio) is used; otherwise 1D (return only).

        Returns
        -------
        pd.DataFrame
            Original df_daily with 17 new psig_ columns added.
        """
        df = df_daily.copy()

        if "close" not in df.columns:
            logger.warning("PathSignatureFeatures: 'close' column missing, skipping")
            return df

        close = df["close"].values.astype(float)
        returns = np.empty(len(close), dtype=float)
        returns[0] = 0.0
        returns[1:] = close[1:] / close[:-1] - 1.0

        # Build volume_ratio channel if volume is available
        has_volume = "volume" in df.columns
        if has_volume:
            volume = df["volume"].values.astype(float)
            vol_ma20 = self._rolling_mean(volume, 20)
            vol_ratio = np.where(vol_ma20 > 0, volume / vol_ma20, 1.0)
            vol_ratio_inc = np.empty(len(vol_ratio), dtype=float)
            vol_ratio_inc[0] = 0.0
            vol_ratio_inc[1:] = vol_ratio[1:] - vol_ratio[:-1]
        else:
            vol_ratio_inc = np.zeros(len(close), dtype=float)

        windows = [5, 10, 20]

        # --- Rolling signature features per window ---
        for w in windows:
            sig1 = self._rolling_sig1(returns, w)
            sig2 = self._rolling_sig2(returns, vol_ratio_inc, w)
            sig2_rev = self._rolling_sig2(vol_ratio_inc, returns, w)
            sig2_anti = sig2 - sig2_rev
            logsig1 = np.sign(sig1) * np.log1p(np.abs(sig1))
            path_len = self._rolling_path_length(returns, w)

            df[f"psig_sig1_{w}d"] = sig1
            df[f"psig_sig2_{w}d"] = sig2
            df[f"psig_sig2_anti_{w}d"] = sig2_anti
            df[f"psig_logsig1_{w}d"] = logsig1
            df[f"psig_path_length_{w}d"] = path_len

        # --- Derived features ---
        # Multi-scale signature ratio
        df["psig_sig_ratio"] = df["psig_sig2_20d"] / (df["psig_sig2_5d"].abs() + 1e-10)

        # Momentum asymmetry: z-score of sig2_anti_20d over 60-day window
        anti_20 = df["psig_sig2_anti_20d"]
        roll_mean = anti_20.rolling(60, min_periods=20).mean()
        roll_std = anti_20.rolling(60, min_periods=20).std()
        df["psig_momentum_asym"] = ((anti_20 - roll_mean) / (roll_std + 1e-10)).clip(-4, 4)

        # Cleanup: fill NaN with 0.0, remove infinities
        for col in self._all_feature_names():
            df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        n_features = sum(1 for c in df.columns if c.startswith("psig_"))
        logger.info(f"PathSignatureFeatures: added {n_features} features")
        return df

    @staticmethod
    def _all_feature_names() -> List[str]:
        names: List[str] = []
        for w in [5, 10, 20]:
            names.append(f"psig_sig1_{w}d")
            names.append(f"psig_sig2_{w}d")
            names.append(f"psig_sig2_anti_{w}d")
            names.append(f"psig_logsig1_{w}d")
            names.append(f"psig_path_length_{w}d")
        names.append("psig_sig_ratio")
        names.append("psig_momentum_asym")
        return names

    # ─── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
        """Simple rolling mean with NaN for early rows."""
        n = len(arr)
        result = np.full(n, np.nan, dtype=float)
        cumsum = np.nancumsum(arr)
        result[window - 1] = cumsum[window - 1] / window
        for i in range(window, n):
            result[i] = (cumsum[i] - cumsum[i - window]) / window
        return result

    @staticmethod
    def _rolling_sig1(increments: np.ndarray, window: int) -> np.ndarray:
        """
        Level-1 signature of a 1D path over a rolling window.

        S^1 = sum(dx_i) = x[-1] - x[0]  (net increment over window).
        Here *increments* are already the dx (returns), so we just sum them.
        """
        n = len(increments)
        result = np.full(n, np.nan, dtype=float)
        # Use cumulative sum trick for O(n) computation
        cumsum = np.nancumsum(increments)
        for i in range(window - 1, n):
            start = i - window + 1
            if start == 0:
                result[i] = cumsum[i]
            else:
                result[i] = cumsum[i] - cumsum[start - 1]
        return result

    @staticmethod
    def _rolling_sig2(
        dx: np.ndarray, dy: np.ndarray, window: int
    ) -> np.ndarray:
        """
        Level-2 cross-signature of a 2D path (x, y) over a rolling window.

        S^{1,2} = sum_{i} sum_{j>i} dx_j * dy_i

        This is the "lead-lag" area: how much x-increments follow y-increments.
        Equivalent to sum_{i < j} dx[j] * dy[i] within the window.
        """
        n = len(dx)
        result = np.full(n, np.nan, dtype=float)

        for i in range(window - 1, n):
            start = i - window + 1
            w_dx = dx[start: i + 1]
            w_dy = dy[start: i + 1]
            w_len = len(w_dx)

            # Efficient computation: for each j, dx[j] * cumsum(dy[0:j])
            cum_dy = np.cumsum(w_dy)
            # S^{1,2} = sum_{j=1..w_len-1} dx[j] * cum_dy[j-1]
            if w_len < 2:
                result[i] = 0.0
            else:
                result[i] = np.sum(w_dx[1:] * cum_dy[:-1])

        return result

    @staticmethod
    def _rolling_path_length(increments: np.ndarray, window: int) -> np.ndarray:
        """
        Total variation (path length) over rolling window.

        L = sum |dx_i|
        """
        n = len(increments)
        result = np.full(n, np.nan, dtype=float)
        abs_inc = np.abs(increments)
        cumsum = np.nancumsum(abs_inc)
        for i in range(window - 1, n):
            start = i - window + 1
            if start == 0:
                result[i] = cumsum[i]
            else:
                result[i] = cumsum[i] - cumsum[start - 1]
        return result
