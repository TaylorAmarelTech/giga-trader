"""
GIGA TRADER - Multiscale Sample Entropy Features
==================================================
Extends single-scale sample entropy by computing SampEn at multiple
coarse-graining scales.  The MSE profile reveals whether a system is
complex-deterministic (entropy maintained across scales) or purely random
(entropy decays at coarser scales).

3 features generated (prefix: mse_).

Coarse-graining: at scale τ, average every τ consecutive points to create
a shorter, smoother time series, then compute sample entropy on it.
Scales used: [1, 2, 3, 5, 10].

The slope of SampEn across scales is the key feature: negative slope
means "more random than complex", flat means "genuine complexity at
multiple timescales".
"""

import logging
import warnings
from typing import Optional, Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger("MSE")


class MultiscaleEntropyFeatures:
    """
    Compute multiscale sample entropy features from daily returns.

    All features use the mse_ prefix.  Pure numpy implementation
    of coarse-graining + template-matching sample entropy.
    """

    REQUIRED_COLS = {"close"}
    SCALES = [1, 2, 3, 5, 10]

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def create_multiscale_entropy_features(
        self,
        spy_daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create multiscale entropy features and merge into spy_daily.

        Returns spy_daily with new mse_* columns added.
        """
        df = spy_daily.copy()

        print("\n[MSE] Engineering multiscale entropy features...")

        if "close" not in df.columns:
            print("  [WARN] No 'close' column — skipping MSE")
            return df

        close = df["close"].values.astype(np.float64)
        n = len(close)

        if n < 50:
            print("  [WARN] Insufficient data (<50 rows) — skipping")
            for name in self._all_feature_names():
                df[name] = 0.0
            return df

        # Daily returns
        returns = np.empty(n)
        returns[0] = 0.0
        returns[1:] = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-10)

        window = 60
        step = 5  # Compute every 5th row for efficiency
        m = 2  # Template length for SampEn
        r_mult = 0.2  # Tolerance = r_mult * std

        slope_arr = np.zeros(n)
        area_arr = np.zeros(n)
        complexity_arr = np.zeros(n)

        for i in range(window, n, step):
            seg = returns[i - window:i]
            seg_std = np.std(seg)

            if seg_std < 1e-10:
                # Constant series — fill with defaults
                fill_end = min(i + step, n)
                for j in range(i, fill_end):
                    slope_arr[j] = 0.0
                    area_arr[j] = 0.0
                    complexity_arr[j] = 0.5
                continue

            r = r_mult * seg_std

            # Compute SampEn at each coarse-graining scale
            entropies = []
            for scale in self.SCALES:
                cg = self._coarse_grain(seg, scale)
                if len(cg) < m + 2:
                    entropies.append(0.0)
                    continue
                se = self._sample_entropy(cg, m, r)
                entropies.append(se)

            entropies = np.array(entropies)

            # Feature 1: Slope of SampEn vs log(scale)
            if len(entropies) >= 2 and np.std(entropies) > 1e-10:
                log_scales = np.log(np.array(self.SCALES, dtype=float))
                slope = self._simple_slope(log_scales, entropies)
            else:
                slope = 0.0

            # Feature 2: Area under MSE curve (trapezoidal)
            area = float(np.sum(0.5 * (entropies[:-1] + entropies[1:]) * np.diff(np.array(self.SCALES, dtype=float))))

            # Feature 3: Complexity index (coarse/fine ratio)
            if entropies[0] > 1e-10:
                complexity = entropies[-1] / entropies[0]
            else:
                complexity = 0.5

            # Forward-fill for skipped rows
            fill_end = min(i + step, n)
            for j in range(i, fill_end):
                slope_arr[j] = slope
                area_arr[j] = area
                complexity_arr[j] = complexity

        df["mse_slope"] = np.clip(slope_arr, -5.0, 5.0)
        df["mse_area"] = np.clip(area_arr, 0.0, 50.0)
        df["mse_complexity_index"] = np.clip(complexity_arr, 0.0, 5.0)

        # Cleanup
        for col in self._all_feature_names():
            df[col] = df[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)

        print(f"  [MSE] Total: {len(self._all_feature_names())} multiscale entropy features added")
        return df

    def analyze_current_entropy(
        self,
        spy_daily: pd.DataFrame,
    ) -> Optional[Dict]:
        """Return snapshot of current multiscale entropy state."""
        if "mse_slope" not in spy_daily.columns or len(spy_daily) < 2:
            return None

        last = spy_daily.iloc[-1]
        slope = float(last.get("mse_slope", 0.0))
        complexity = float(last.get("mse_complexity_index", 0.5))

        if complexity > 0.8:
            regime = "COMPLEX"  # Genuine multi-scale structure
        elif complexity < 0.3:
            regime = "RANDOM"  # Entropy collapses at coarser scales
        else:
            regime = "MIXED"

        return {
            "entropy_regime": regime,
            "mse_slope": round(slope, 4),
            "complexity": round(complexity, 4),
            "area": round(float(last.get("mse_area", 0.0)), 4),
        }

    # ------------------------------------------------------------------ #
    #  Private helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _all_feature_names():
        return ["mse_slope", "mse_area", "mse_complexity_index"]

    @staticmethod
    def _coarse_grain(x: np.ndarray, scale: int) -> np.ndarray:
        """Coarse-grain a time series by averaging consecutive scale-length blocks."""
        if scale == 1:
            return x
        n = len(x)
        n_blocks = n // scale
        if n_blocks < 1:
            return x[:1]
        trimmed = x[:n_blocks * scale].reshape(n_blocks, scale)
        return trimmed.mean(axis=1)

    @staticmethod
    def _sample_entropy(x: np.ndarray, m: int, r: float) -> float:
        """
        Compute sample entropy using Chebyshev distance template matching.

        SampEn = -ln(A/B) where:
        - B = number of template matches of length m within tolerance r
        - A = number of template matches of length m+1 within tolerance r
        """
        n = len(x)
        if n < m + 2:
            return 0.0

        # Count matches of length m and m+1
        count_m = 0
        count_m1 = 0

        for i in range(n - m):
            for j in range(i + 1, n - m):
                # Chebyshev distance for templates of length m
                dist_m = np.max(np.abs(x[i:i + m] - x[j:j + m]))
                if dist_m <= r:
                    count_m += 1
                    # Check if extending to m+1 also matches
                    if i + m < n and j + m < n:
                        dist_m1 = max(dist_m, abs(x[i + m] - x[j + m]))
                        if dist_m1 <= r:
                            count_m1 += 1

        if count_m == 0 or count_m1 == 0:
            return 0.0

        return -np.log(count_m1 / count_m)

    @staticmethod
    def _simple_slope(x: np.ndarray, y: np.ndarray) -> float:
        """Simple linear regression slope."""
        n = len(x)
        if n < 2:
            return 0.0
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        denom = np.sum((x - x_mean) ** 2)
        if denom < 1e-15:
            return 0.0
        return float(np.sum((x - x_mean) * (y - y_mean)) / denom)
