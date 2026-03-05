"""
Entropy Features — information-theoretic measures of price dynamics.

Shannon entropy measures randomness of return distributions.
Permutation entropy captures ordinal pattern complexity.
High entropy = unpredictable market; low entropy = trending/mean-reverting.

Features (6, prefix ent_):
  ent_shannon_20d       — Shannon entropy of binned 20-day returns (10 bins)
  ent_permutation_20d   — Permutation entropy of rolling 20-day close prices (order-3)
  ent_sample_20d        — Sample entropy proxy: rolling 20d std of first-differences of returns
  ent_shannon_z         — 60-day z-score of ent_shannon_20d, clipped [-4, 4]
  ent_regime_change     — |delta(ent_permutation_20d)| — regime shift detector
  ent_predictability    — 1 - normalized permutation entropy (from ent_permutation_20d)
"""

import logging
from math import factorial, log2
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from src.core.feature_base import FeatureModuleBase

logger = logging.getLogger(__name__)


class EntropyFeatures(FeatureModuleBase):
    """Compute information-theoretic entropy features from daily OHLCV data."""
    FEATURE_NAMES = ["ent_shannon_20d", "ent_permutation_20d", "ent_sample_20d", "ent_shannon_z", "ent_regime_change", "ent_predictability"]


    REQUIRED_COLS = {"close"}

    def create_entropy_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Add entropy features to df_daily.

        Parameters
        ----------
        df_daily : pd.DataFrame
            Must have 'close' column.

        Returns
        -------
        pd.DataFrame
            Original df_daily with 6 new ent_ columns added.
        """
        df = df_daily.copy()

        if "close" not in df.columns:
            logger.warning("EntropyFeatures: 'close' column missing, skipping")
            return df

        close = df["close"].values.astype(float)
        returns = pd.Series(close).pct_change().values

        # --- ent_shannon_20d: Shannon entropy of binned 20-day returns ---
        df["ent_shannon_20d"] = self._rolling_shannon_entropy(returns, window=20, n_bins=10)

        # --- ent_permutation_20d: Permutation entropy of rolling 20-day close ---
        perm_ent = self._rolling_permutation_entropy(close, window=20, order=3)
        df["ent_permutation_20d"] = perm_ent

        # --- ent_sample_20d: Sample entropy proxy ---
        # Rolling 20d std of first-differences of returns
        ret_series = pd.Series(returns)
        ret_diff = ret_series.diff()
        df["ent_sample_20d"] = ret_diff.rolling(20, min_periods=5).std().values

        # --- ent_shannon_z: 60-day z-score of shannon entropy ---
        shannon = df["ent_shannon_20d"]
        rolling_mean = shannon.rolling(60, min_periods=20).mean()
        rolling_std = shannon.rolling(60, min_periods=20).std()
        df["ent_shannon_z"] = ((shannon - rolling_mean) / (rolling_std + 1e-10)).clip(-4, 4)

        # --- ent_regime_change: |delta(permutation_entropy)| ---
        perm_series = df["ent_permutation_20d"]
        df["ent_regime_change"] = perm_series.diff().abs()

        # --- ent_predictability: 1 - normalized permutation entropy ---
        # ent_permutation_20d is already normalized by log2(order!), so in [0, 1]
        df["ent_predictability"] = (1.0 - df["ent_permutation_20d"]).clip(0, 1)

        # Cleanup: fill NaN with 0.0, remove infinities
        for col in self._all_feature_names():
            df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        n_features = sum(1 for c in df.columns if c.startswith("ent_"))
        logger.info(f"EntropyFeatures: added {n_features} features")
        return df

    def analyze_current_entropy(self, df_daily: pd.DataFrame) -> Optional[Dict]:
        """Analyze current entropy regime for dashboard display."""
        if "ent_shannon_z" not in df_daily.columns or len(df_daily) < 2:
            return None

        last = df_daily.iloc[-1]
        z = last.get("ent_shannon_z", 0.0)

        if z > 1.5:
            regime = "HIGH_ENTROPY"
        elif z < -1.5:
            regime = "LOW_ENTROPY"
        else:
            regime = "NORMAL"

        return {
            "entropy_regime": regime,
            "shannon_z": round(float(z), 3),
            "permutation_entropy": round(float(last.get("ent_permutation_20d", 0.0)), 4),
            "predictability": round(float(last.get("ent_predictability", 0.0)), 4),
        }

    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "ent_shannon_20d",
            "ent_permutation_20d",
            "ent_sample_20d",
            "ent_shannon_z",
            "ent_regime_change",
            "ent_predictability",
        ]

    # ─── Internal helpers ─────────────────────────────────────────────────

    @staticmethod
    def _rolling_shannon_entropy(
        returns: np.ndarray, window: int = 20, n_bins: int = 10
    ) -> np.ndarray:
        """
        Compute Shannon entropy of binned returns over a rolling window.

        For each window, discretize returns into *n_bins* equal-width bins
        spanning the window's min to max, compute relative frequencies,
        and return -sum(p * log2(p)).
        """
        n = len(returns)
        result = np.full(n, np.nan)

        for i in range(window, n):
            w = returns[i - window : i]
            # Skip if all NaN or constant
            valid = w[~np.isnan(w)]
            if len(valid) < 3:
                continue

            w_min, w_max = valid.min(), valid.max()
            if w_max - w_min < 1e-15:
                result[i] = 0.0
                continue

            # Digitize into n_bins
            bin_edges = np.linspace(w_min, w_max, n_bins + 1)
            bin_edges[-1] += 1e-10  # ensure max value falls in last bin
            indices = np.digitize(valid, bin_edges) - 1
            indices = np.clip(indices, 0, n_bins - 1)

            counts = np.bincount(indices, minlength=n_bins).astype(float)
            probs = counts / counts.sum()
            probs = probs[probs > 0]

            result[i] = -np.sum(probs * np.log2(probs))

        return result

    @staticmethod
    def _rolling_permutation_entropy(
        values: np.ndarray, window: int = 20, order: int = 3
    ) -> np.ndarray:
        """
        Compute normalized permutation entropy over a rolling window.

        For each window of *values*, extract all overlapping ordinal patterns
        of length *order*, count their frequencies, compute Shannon entropy
        of the pattern distribution, and normalize by log2(order!).

        Result is in [0, 1]: 0 = perfectly predictable, 1 = maximum complexity.
        """
        n = len(values)
        n_perms = factorial(order)
        max_entropy = log2(n_perms) if n_perms > 1 else 1.0
        result = np.full(n, np.nan)

        for i in range(window, n):
            w = values[i - window : i]

            # Skip if any NaN in window
            if np.any(np.isnan(w)):
                continue

            # Extract ordinal patterns
            n_patterns = len(w) - order + 1
            if n_patterns < 2:
                continue

            # Count pattern frequencies using a dict for speed
            pattern_counts: Dict[tuple, int] = {}
            for j in range(n_patterns):
                segment = w[j : j + order]
                # Ordinal pattern: rank order of the segment
                pattern = tuple(np.argsort(segment).tolist())
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

            total = sum(pattern_counts.values())
            probs = np.array(list(pattern_counts.values()), dtype=float) / total

            entropy = -np.sum(probs * np.log2(probs))
            result[i] = entropy / max_entropy  # Normalize to [0, 1]

        return result
