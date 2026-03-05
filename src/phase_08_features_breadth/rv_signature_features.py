"""
GIGA TRADER - Realized Volatility Signature Plot Features
===========================================================
Cross-frequency volatility analysis.  Computes realized volatility at
multiple return sampling frequencies and extracts features from how RV
changes across frequencies.

When the RV signature plot is flat (same RV regardless of frequency),
the market is "clean".  When it rises at high frequencies, there is
heavy microstructure noise (typically stressed or illiquid periods).

3 features generated (prefix: rvsp_).

Uses subsampled returns at frequencies: 1d, 2d, 5d, 10d, 20d to
construct a daily-resolution signature plot from daily OHLCV data.
"""

import logging
import warnings
from typing import Optional, Dict

import numpy as np
import pandas as pd
from src.core.feature_base import FeatureModuleBase

warnings.filterwarnings("ignore")

logger = logging.getLogger("RV_SIGNATURE")


class RVSignaturePlotFeatures(FeatureModuleBase):
    """
    Compute RV signature plot features from daily close prices.

    All features use the rvsp_ prefix.  Pure numpy implementation.
    """
    FEATURE_NAMES = ["rvsp_slope", "rvsp_noise_ratio", "rvsp_flatness"]


    REQUIRED_COLS = {"close"}
    FREQUENCIES = [1, 2, 5, 10, 20]  # Subsampling periods in days

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def create_rv_signature_features(
        self,
        spy_daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create RV signature plot features and merge into spy_daily.

        Returns spy_daily with new rvsp_* columns added.
        """
        df = spy_daily.copy()

        print("\n[RVSP] Engineering RV signature plot features...")

        if "close" not in df.columns:
            print("  [WARN] No 'close' column — skipping RV signature")
            return df

        close = df["close"].values.astype(np.float64)
        n = len(close)

        if n < 60:
            print("  [WARN] Insufficient data (<60 rows) — skipping")
            for name in self._all_feature_names():
                df[name] = 0.0
            return df

        # Compute returns at each frequency
        window = 60
        step = 5

        slope_arr = np.zeros(n)
        noise_ratio_arr = np.zeros(n)
        flatness_arr = np.zeros(n)

        for i in range(window, n, step):
            seg = close[i - window:i]

            # Compute RV at each subsampling frequency
            rvs = []
            for freq in self.FREQUENCIES:
                # Subsample: take every freq-th price
                sub_prices = seg[::freq]
                if len(sub_prices) < 3:
                    rvs.append(np.nan)
                    continue
                sub_returns = np.diff(np.log(np.maximum(sub_prices, 1e-10)))
                # Annualize: multiply by (252 / freq) to make comparable
                rv = np.sum(sub_returns ** 2) * (252.0 / freq)
                rvs.append(rv)

            rvs = np.array(rvs)
            valid = ~np.isnan(rvs)

            if np.sum(valid) < 3:
                fill_end = min(i + step, n)
                for j in range(i, fill_end):
                    slope_arr[j] = 0.0
                    noise_ratio_arr[j] = 1.0
                    flatness_arr[j] = 1.0
                continue

            # Feature 1: Log-log slope of RV vs frequency
            log_freq = np.log(np.array(self.FREQUENCIES, dtype=float)[valid])
            log_rv = np.log(np.maximum(rvs[valid], 1e-15))
            slope = self._simple_slope(log_freq, log_rv)

            # Feature 2: Noise ratio (high-freq RV / low-freq RV)
            rv_high = rvs[0] if not np.isnan(rvs[0]) else 1e-10
            rv_low = rvs[-1] if not np.isnan(rvs[-1]) else 1e-10
            noise_ratio = rv_high / max(rv_low, 1e-15)

            # Feature 3: Flatness (1 - normalized variance of RVs)
            valid_rvs = rvs[valid]
            rv_mean = np.mean(valid_rvs)
            if rv_mean > 1e-15:
                cv = np.std(valid_rvs) / rv_mean
                flatness = max(0.0, 1.0 - cv)
            else:
                flatness = 1.0

            fill_end = min(i + step, n)
            for j in range(i, fill_end):
                slope_arr[j] = slope
                noise_ratio_arr[j] = noise_ratio
                flatness_arr[j] = flatness

        df["rvsp_slope"] = np.clip(slope_arr, -5.0, 5.0)
        df["rvsp_noise_ratio"] = np.clip(noise_ratio_arr, 0.0, 10.0)
        df["rvsp_flatness"] = np.clip(flatness_arr, 0.0, 1.0)

        # Cleanup
        for col in self._all_feature_names():
            df[col] = df[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)

        print(f"  [RVSP] Total: {len(self._all_feature_names())} RV signature features added")
        return df

    def analyze_current_rv_signature(
        self,
        spy_daily: pd.DataFrame,
    ) -> Optional[Dict]:
        """Return snapshot of current RV signature state."""
        if "rvsp_flatness" not in spy_daily.columns or len(spy_daily) < 2:
            return None

        last = spy_daily.iloc[-1]
        flatness = float(last.get("rvsp_flatness", 1.0))
        noise = float(last.get("rvsp_noise_ratio", 1.0))

        if flatness > 0.8:
            microstructure = "CLEAN"
        elif flatness < 0.4:
            microstructure = "NOISY"
        else:
            microstructure = "MODERATE"

        return {
            "microstructure": microstructure,
            "flatness": round(flatness, 3),
            "noise_ratio": round(noise, 3),
            "slope": round(float(last.get("rvsp_slope", 0.0)), 4),
        }

    # ------------------------------------------------------------------ #
    #  Private helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _all_feature_names():
        return ["rvsp_slope", "rvsp_noise_ratio", "rvsp_flatness"]

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
