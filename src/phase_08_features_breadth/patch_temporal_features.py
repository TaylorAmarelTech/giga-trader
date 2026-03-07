"""
PatchTST-Style Patch-Based Temporal Features — multi-scale temporal pattern extraction.

Inspired by PatchTST (Nie et al., 2023), this module divides time series into
non-overlapping patches (segments) of multiple sizes and extracts statistics
from each patch. Cross-patch aggregation captures multi-scale temporal dynamics
without recurrence or attention mechanisms.

Key insight: by examining how statistical properties (trend, volatility, volume)
evolve across consecutive patches, we detect regime shifts, momentum decay,
and breakout conditions that single-scale indicators miss.

Features (8, prefix ptst_):
  ptst_patch_trend_consistency  — Fraction of patches with same trend sign (across scales)
  ptst_patch_volatility_profile — Slope of patch volatility over time (rising = risk increasing)
  ptst_patch_volume_profile     — Slope of patch volume ratio over time
  ptst_cross_patch_correlation  — Mean correlation between adjacent patch return vectors
  ptst_patch_momentum_decay     — Ratio of most-recent patch momentum to oldest patch
  ptst_patch_breakout_score     — Max absolute return at any patch boundary
  ptst_multi_scale_trend        — Weighted average of trend signals across patch sizes
  ptst_patch_entropy            — Shannon entropy of patch-level return distribution
"""

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from src.core.feature_base import FeatureModuleBase

logger = logging.getLogger(__name__)


class PatchTemporalFeatures(FeatureModuleBase):
    """Compute PatchTST-style patch-based temporal features from daily OHLCV data."""

    REQUIRED_COLS = {"close", "high", "low", "volume"}
    FEATURE_NAMES = [
        "ptst_patch_trend_consistency",
        "ptst_patch_volatility_profile",
        "ptst_patch_volume_profile",
        "ptst_cross_patch_correlation",
        "ptst_patch_momentum_decay",
        "ptst_patch_breakout_score",
        "ptst_multi_scale_trend",
        "ptst_patch_entropy",
    ]

    def __init__(
        self,
        patch_sizes: Tuple[int, ...] = (5, 10, 20),
        lookback: int = 60,
    ):
        """
        Parameters
        ----------
        patch_sizes : tuple of int
            Multiple patch sizes for multi-scale analysis. Each divides the
            lookback window into non-overlapping segments.
        lookback : int
            Total lookback window in rows (trading days).
        """
        self.patch_sizes = tuple(sorted(patch_sizes))
        self.lookback = max(lookback, max(patch_sizes) * 2)  # ensure at least 2 patches

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_patch_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add 8 ptst_ features to *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Must contain columns: close, high, low, volume.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with 8 new ptst_ columns appended.
        """
        df = df.copy()

        if not self._validate_input(df, min_rows=2):
            return self._zero_fill_all(df)

        n = len(df)
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        volume = df["volume"].values.astype(float)

        # Pre-compute returns (percentage change)
        returns = np.empty(n)
        returns[0] = 0.0
        returns[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-15)

        # Allocate output arrays
        trend_consistency = np.zeros(n)
        volatility_profile = np.zeros(n)
        volume_profile = np.zeros(n)
        cross_patch_corr = np.zeros(n)
        momentum_decay = np.zeros(n)
        breakout_score = np.zeros(n)
        multi_scale_trend = np.zeros(n)
        patch_entropy = np.zeros(n)

        for i in range(self.lookback, n):
            # Extract the lookback window ending at index i (inclusive)
            start = i - self.lookback + 1
            win_returns = returns[start: i + 1]
            win_close = close[start: i + 1]
            win_high = high[start: i + 1]
            win_low = low[start: i + 1]
            win_volume = volume[start: i + 1]
            win_len = len(win_returns)

            # Accumulate across patch sizes
            all_trend_fracs: List[float] = []
            all_vol_slopes: List[float] = []
            all_volm_slopes: List[float] = []
            all_cross_corrs: List[float] = []
            all_mom_decays: List[float] = []
            all_breakouts: List[float] = []
            all_trend_signals: List[float] = []
            all_entropies: List[float] = []

            for ps in self.patch_sizes:
                n_patches = win_len // ps
                if n_patches < 2:
                    continue

                # Trim to exact patch coverage (drop leftover at the start)
                usable = n_patches * ps
                offset = win_len - usable

                # Per-patch statistics
                patch_means = np.empty(n_patches)
                patch_vols = np.empty(n_patches)
                patch_vol_ratios = np.empty(n_patches)
                patch_directions = np.empty(n_patches)
                patch_return_vecs: List[np.ndarray] = []
                patch_boundary_returns: List[float] = []

                mean_volume = np.mean(win_volume) + 1e-15

                for p in range(n_patches):
                    p_start = offset + p * ps
                    p_end = p_start + ps
                    p_ret = win_returns[p_start:p_end]
                    p_close = win_close[p_start:p_end]
                    p_high = win_high[p_start:p_end]
                    p_low = win_low[p_start:p_end]
                    p_vol = win_volume[p_start:p_end]

                    patch_means[p] = np.mean(p_ret)
                    patch_vols[p] = np.std(p_ret) if len(p_ret) > 1 else 0.0
                    patch_vol_ratios[p] = np.mean(p_vol) / mean_volume
                    patch_directions[p] = 1.0 if (p_close[-1] - p_close[0]) > 0 else -1.0

                    patch_return_vecs.append(p_ret.copy())

                    # Boundary return: return at the start of this patch
                    if p_start > 0:
                        prev_close = win_close[p_start - 1]
                        if abs(prev_close) > 1e-15:
                            boundary_ret = (win_close[p_start] - prev_close) / prev_close
                        else:
                            boundary_ret = 0.0
                        patch_boundary_returns.append(abs(boundary_ret))

                # --- trend_consistency: fraction of patches with majority trend sign ---
                n_pos = np.sum(patch_directions > 0)
                n_neg = n_patches - n_pos
                majority = max(n_pos, n_neg)
                all_trend_fracs.append(majority / n_patches)

                # --- volatility_profile: slope of volatility across patches ---
                if n_patches >= 2:
                    x = np.arange(n_patches, dtype=float)
                    all_vol_slopes.append(self._simple_slope(x, patch_vols))
                else:
                    all_vol_slopes.append(0.0)

                # --- volume_profile: slope of volume ratio across patches ---
                if n_patches >= 2:
                    all_volm_slopes.append(self._simple_slope(x, patch_vol_ratios))
                else:
                    all_volm_slopes.append(0.0)

                # --- cross_patch_correlation: mean corr between adjacent patches ---
                corrs: List[float] = []
                for p in range(n_patches - 1):
                    v1 = patch_return_vecs[p]
                    v2 = patch_return_vecs[p + 1]
                    c = self._safe_correlation(v1, v2)
                    corrs.append(c)
                if corrs:
                    all_cross_corrs.append(float(np.mean(corrs)))
                else:
                    all_cross_corrs.append(0.0)

                # --- momentum_decay: ratio of newest patch mean to oldest ---
                oldest_mom = patch_means[0]
                newest_mom = patch_means[-1]
                if abs(oldest_mom) > 1e-10:
                    all_mom_decays.append(newest_mom / oldest_mom)
                else:
                    # If oldest is ~0, use sign comparison instead
                    all_mom_decays.append(np.sign(newest_mom))

                # --- breakout_score: max boundary return ---
                if patch_boundary_returns:
                    all_breakouts.append(max(patch_boundary_returns))
                else:
                    all_breakouts.append(0.0)

                # --- trend signal for this patch size ---
                # Weighted mean of patch directions (more recent patches weighted higher)
                weights = np.arange(1, n_patches + 1, dtype=float)
                weights /= weights.sum()
                all_trend_signals.append(float(np.dot(weights, patch_directions)))

                # --- patch_entropy: Shannon entropy of patch return distribution ---
                all_entropies.append(self._patch_shannon_entropy(patch_means))

            # Aggregate across patch sizes (weighted by 1/patch_size for diversity)
            if all_trend_fracs:
                trend_consistency[i] = float(np.mean(all_trend_fracs))
                volatility_profile[i] = float(np.mean(all_vol_slopes))
                volume_profile[i] = float(np.mean(all_volm_slopes))
                cross_patch_corr[i] = float(np.mean(all_cross_corrs))

                # Clip momentum_decay to [-5, 5] to avoid extreme ratios
                clipped_decays = np.clip(all_mom_decays, -5.0, 5.0)
                momentum_decay[i] = float(np.mean(clipped_decays))

                breakout_score[i] = float(np.max(all_breakouts))

                # multi_scale_trend: weighted average by inverse patch size
                inv_sizes = np.array([1.0 / ps for ps in self.patch_sizes[:len(all_trend_signals)]])
                inv_sizes /= inv_sizes.sum() + 1e-15
                multi_scale_trend[i] = float(np.dot(inv_sizes, all_trend_signals))

                patch_entropy[i] = float(np.mean(all_entropies))

        # Assign to DataFrame
        df["ptst_patch_trend_consistency"] = trend_consistency
        df["ptst_patch_volatility_profile"] = volatility_profile
        df["ptst_patch_volume_profile"] = volume_profile
        df["ptst_cross_patch_correlation"] = cross_patch_corr
        df["ptst_patch_momentum_decay"] = momentum_decay
        df["ptst_patch_breakout_score"] = breakout_score
        df["ptst_multi_scale_trend"] = multi_scale_trend
        df["ptst_patch_entropy"] = patch_entropy

        self._cleanup_features(df)

        n_features = sum(1 for c in df.columns if c.startswith("ptst_"))
        logger.info(f"PatchTemporalFeatures: added {n_features} features")
        return df

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _simple_slope(x: np.ndarray, y: np.ndarray) -> float:
        """Compute slope of simple linear regression y = a + b*x.

        Uses the standard formula: b = cov(x,y) / var(x).
        Returns 0.0 if x is constant or arrays are too short.
        """
        if len(x) < 2:
            return 0.0
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        var_x = np.mean((x - x_mean) ** 2)
        if var_x < 1e-15:
            return 0.0
        cov_xy = np.mean((x - x_mean) * (y - y_mean))
        return float(cov_xy / var_x)

    @staticmethod
    def _safe_correlation(a: np.ndarray, b: np.ndarray) -> float:
        """Pearson correlation that returns 0.0 on degenerate inputs."""
        if len(a) < 2 or len(b) < 2:
            return 0.0
        min_len = min(len(a), len(b))
        a = a[:min_len]
        b = b[:min_len]
        std_a = np.std(a)
        std_b = np.std(b)
        if std_a < 1e-15 or std_b < 1e-15:
            return 0.0
        cov = np.mean((a - np.mean(a)) * (b - np.mean(b)))
        return float(np.clip(cov / (std_a * std_b), -1.0, 1.0))

    @staticmethod
    def _patch_shannon_entropy(patch_means: np.ndarray, n_bins: int = 8) -> float:
        """Shannon entropy of patch-level mean-return distribution.

        Discretizes patch means into *n_bins* equal-width bins and computes
        H = -sum(p * log2(p)).  Returns 0.0 if data is degenerate.
        """
        valid = patch_means[~np.isnan(patch_means)]
        if len(valid) < 2:
            return 0.0
        v_min, v_max = valid.min(), valid.max()
        if v_max - v_min < 1e-15:
            return 0.0

        bin_edges = np.linspace(v_min, v_max, n_bins + 1)
        bin_edges[-1] += 1e-10  # ensure max falls in last bin
        indices = np.digitize(valid, bin_edges) - 1
        indices = np.clip(indices, 0, n_bins - 1)

        counts = np.bincount(indices, minlength=n_bins).astype(float)
        probs = counts / counts.sum()
        probs = probs[probs > 0]

        return float(-np.sum(probs * np.log2(probs)))

    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "ptst_patch_trend_consistency",
            "ptst_patch_volatility_profile",
            "ptst_patch_volume_profile",
            "ptst_cross_patch_correlation",
            "ptst_patch_momentum_decay",
            "ptst_patch_breakout_score",
            "ptst_multi_scale_trend",
            "ptst_patch_entropy",
        ]
