"""
GIGA TRADER - L-Moments Features
==================================
Robust distributional shape features using L-moments (probability-weighted
moments of order statistics).

L-moments are far more robust to outliers than conventional moments, and
uniquely defined for heavy-tailed distributions where conventional kurtosis
may not exist.  L-skewness is bounded [-1, 1], L-kurtosis is bounded,
making them ideal for characterizing financial return distributions.

4 features generated (prefix: lmom_).

L1 = mean, L2 = scale (analogous to std), L3/L2 = L-skewness, L4/L2 = L-kurtosis.
"""

import logging
import warnings
from typing import Optional, Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger("L_MOMENTS")


class LMomentsFeatures:
    """
    Compute L-moment features from daily returns.

    All features use the lmom_ prefix.  Pure numpy implementation
    using probability-weighted moments (PWM) formula.
    """

    REQUIRED_COLS = {"close"}

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def create_l_moments_features(
        self,
        spy_daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create L-moment features and merge into spy_daily.

        Returns spy_daily with new lmom_* columns added.
        """
        df = spy_daily.copy()

        print("\n[LMOM] Engineering L-moment features...")

        if "close" not in df.columns:
            print("  [WARN] No 'close' column — skipping L-moments")
            return df

        close = df["close"].values.astype(np.float64)
        n = len(close)

        if n < 25:
            print("  [WARN] Insufficient data (<25 rows) — skipping")
            for name in self._all_feature_names():
                df[name] = 0.0
            return df

        # Daily returns
        returns = np.empty(n)
        returns[0] = 0.0
        returns[1:] = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-10)

        window = 20
        lcv = np.zeros(n)
        lskew = np.zeros(n)
        lkurt = np.zeros(n)

        for i in range(window, n):
            seg = returns[i - window:i]
            l1, l2, l3, l4 = self._compute_l_moments(seg)

            # L-CV: L2 / |L1| (robust coefficient of variation)
            if abs(l1) > 1e-10:
                lcv[i] = l2 / abs(l1)
            else:
                lcv[i] = 0.0

            # L-skewness: L3 / L2
            if l2 > 1e-10:
                lskew[i] = l3 / l2
                lkurt[i] = l4 / l2
            else:
                lskew[i] = 0.0
                lkurt[i] = 0.0

        # Feature 1: L-CV (robust dispersion)
        df["lmom_lcv_20d"] = np.clip(lcv, 0.0, 50.0)

        # Feature 2: L-skewness (bounded [-1, 1])
        df["lmom_lskew_20d"] = np.clip(lskew, -1.0, 1.0)

        # Feature 3: L-kurtosis
        df["lmom_lkurt_20d"] = np.clip(lkurt, -1.0, 1.0)

        # Feature 4: Z-scored L-skewness (regime change detector)
        lskew_series = pd.Series(lskew)
        lskew_mean = lskew_series.rolling(60, min_periods=30).mean()
        lskew_std = lskew_series.rolling(60, min_periods=30).std()
        lskew_std = lskew_std.replace(0.0, 1e-10)
        df["lmom_lskew_z"] = np.clip(
            ((lskew_series - lskew_mean) / lskew_std).to_numpy(copy=True), -5.0, 5.0
        )

        # Cleanup
        for col in self._all_feature_names():
            df[col] = df[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)

        print(f"  [LMOM] Total: {len(self._all_feature_names())} L-moment features added")
        return df

    def analyze_current_lmoments(
        self,
        spy_daily: pd.DataFrame,
    ) -> Optional[Dict]:
        """Return snapshot of current L-moment state."""
        if "lmom_lskew_20d" not in spy_daily.columns or len(spy_daily) < 2:
            return None

        last = spy_daily.iloc[-1]
        lskew = float(last.get("lmom_lskew_20d", 0.0))
        lkurt = float(last.get("lmom_lkurt_20d", 0.0))
        lskew_z = float(last.get("lmom_lskew_z", 0.0))

        if lskew < -0.3:
            dist_regime = "LEFT_SKEWED"
        elif lskew > 0.3:
            dist_regime = "RIGHT_SKEWED"
        else:
            dist_regime = "SYMMETRIC"

        return {
            "distribution_regime": dist_regime,
            "l_skewness": round(lskew, 4),
            "l_kurtosis": round(lkurt, 4),
            "l_skewness_z": round(lskew_z, 3),
        }

    # ------------------------------------------------------------------ #
    #  Private helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _all_feature_names():
        return ["lmom_lcv_20d", "lmom_lskew_20d", "lmom_lkurt_20d", "lmom_lskew_z"]

    @staticmethod
    def _compute_l_moments(x: np.ndarray):
        """
        Compute L-moments L1, L2, L3, L4 from a sample using PWM (probability
        weighted moments) method.

        PWM formula: β_r = (1/n) Σ_{j=1}^{n} C(j-1, r) / C(n-1, r) * x_{(j)}
        where x_{(j)} are the sorted observations.

        L1 = β_0
        L2 = 2*β_1 - β_0
        L3 = 6*β_2 - 6*β_1 + β_0
        L4 = 20*β_3 - 30*β_2 + 12*β_1 - β_0
        """
        xs = np.sort(x)
        n = len(xs)

        if n < 4:
            return 0.0, 0.0, 0.0, 0.0

        # Compute PWMs β_0, β_1, β_2, β_3
        j = np.arange(1, n + 1, dtype=np.float64)  # 1-indexed

        b0 = np.mean(xs)
        b1 = np.sum((j - 1) / (n - 1) * xs) / n if n > 1 else 0.0
        b2 = np.sum((j - 1) * (j - 2) / ((n - 1) * max(n - 2, 1)) * xs) / n if n > 2 else 0.0
        b3 = np.sum((j - 1) * (j - 2) * (j - 3) / ((n - 1) * max(n - 2, 1) * max(n - 3, 1)) * xs) / n if n > 3 else 0.0

        l1 = b0
        l2 = 2 * b1 - b0
        l3 = 6 * b2 - 6 * b1 + b0
        l4 = 20 * b3 - 30 * b2 + 12 * b1 - b0

        return l1, l2, l3, l4
