"""
Kronos Features -- foundation-model-style feature extractor via random projection.

A lightweight autoencoder-inspired approach that compresses rolling OHLCV windows
into low-dimensional embeddings using a fixed random projection matrix.  No deep
learning dependencies -- pure numpy.

Algorithm:
  1. Take a rolling window of normalised OHLCV data (default 20 bars).
  2. Flatten each window into a vector of (window * n_input_features).
  3. Project through a fixed random matrix (the "encoder") -> embedding.
  4. Compute reconstruction via pseudo-inverse -> reconstruction error.
  5. Derive regime/volatility/trend features from the embedding.

Features (12, prefix kron_):
  kron_embed_0 .. kron_embed_7  -- 8-dim random projection embedding
  kron_recon_error              -- reconstruction error (anomaly score)
  kron_volatility_mode          -- std of embedding values (volatility regime proxy)
  kron_trend_strength           -- autocorrelation of embedding with shifted version
  kron_regime_proxy             -- sign of dominant component of embedding history
"""

import logging
from typing import List, Set

import numpy as np
import pandas as pd

from src.core.feature_base import FeatureModuleBase

logger = logging.getLogger(__name__)


class KronosFeatures(FeatureModuleBase):
    """Compress rolling OHLCV windows into learned representations via random projection.

    This follows EDGE 1 (regularization-first): the random projection is fixed and
    low-dimensional, preventing overfitting by design.  The pseudo-inverse
    reconstruction provides an anomaly score without any gradient-based training.
    """

    REQUIRED_COLS: Set[str] = {"close", "high", "low", "volume"}
    FEATURE_NAMES: List[str] = [
        "kron_embed_0",
        "kron_embed_1",
        "kron_embed_2",
        "kron_embed_3",
        "kron_embed_4",
        "kron_embed_5",
        "kron_embed_6",
        "kron_embed_7",
        "kron_recon_error",
        "kron_volatility_mode",
        "kron_trend_strength",
        "kron_regime_proxy",
    ]

    # Input columns used for the flattened window vector (order matters).
    _INPUT_COLS = ["close", "high", "low", "volume"]

    def __init__(self, window: int = 20, n_embed: int = 8, seed: int = 42) -> None:
        self.window = max(window, 2)
        self.n_embed = max(n_embed, 1)
        self.seed = seed

        # Build fixed random projection matrix ("encoder weights").
        n_input = self.window * len(self._INPUT_COLS)
        rng = np.random.RandomState(self.seed)
        # Orthogonalise via QR for a better-conditioned projection.
        raw = rng.randn(n_input, self.n_embed)
        if n_input >= self.n_embed:
            Q, _ = np.linalg.qr(raw, mode="reduced")
            self._proj = Q  # shape (n_input, n_embed), columns are orthonormal
        else:
            self._proj = raw / (np.linalg.norm(raw, axis=0, keepdims=True) + 1e-10)

        # Pseudo-inverse of the projection for reconstruction.
        self._proj_pinv = np.linalg.pinv(self._proj)  # shape (n_embed, n_input)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def create_kronos_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Kronos random-projection features to *df*.

        Parameters
        ----------
        df : pd.DataFrame
            OHLCV-style frame.  Must contain ``close``, ``high``, ``low``,
            ``volume``.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with 12 new ``kron_`` columns appended.
        """
        df = df.copy()

        if not self._validate_input(df, min_rows=self.window):
            return self._zero_fill_all(df)

        n = len(df)
        n_embed = self.n_embed

        # --- Normalise input columns (z-score per column) ---------------
        normed = {}
        for col in self._INPUT_COLS:
            vals = df[col].values.astype(np.float64)
            vals = np.where(np.isfinite(vals), vals, 0.0)
            mu = np.mean(vals)
            sigma = np.std(vals)
            if sigma < 1e-12:
                normed[col] = np.zeros(n)
            else:
                normed[col] = (vals - mu) / sigma

        # --- Pre-allocate output arrays ---------------------------------
        embeddings = np.zeros((n, n_embed))
        recon_error = np.zeros(n)

        # --- Sliding-window projection ----------------------------------
        n_input = self.window * len(self._INPUT_COLS)

        for i in range(self.window - 1, n):
            start = i - self.window + 1
            # Flatten the window: [close_t-w+1, ..., close_t, high_t-w+1, ...]
            vec = np.empty(n_input)
            offset = 0
            for col in self._INPUT_COLS:
                vec[offset : offset + self.window] = normed[col][start : i + 1]
                offset += self.window

            # Project to embedding.
            embed = vec @ self._proj  # (n_embed,)
            embeddings[i] = embed

            # Reconstruct and measure error.
            recon = embed @ self._proj_pinv  # (n_input,)
            recon_error[i] = float(np.sqrt(np.mean((vec - recon) ** 2)))

        # --- Write embedding columns ------------------------------------
        for j in range(min(n_embed, 8)):
            df[f"kron_embed_{j}"] = embeddings[:, j]

        # Pad any remaining embed columns if n_embed < 8
        for j in range(n_embed, 8):
            df[f"kron_embed_{j}"] = 0.0

        df["kron_recon_error"] = recon_error

        # --- Derived features -------------------------------------------
        # kron_volatility_mode: std of embedding values per row
        df["kron_volatility_mode"] = np.std(embeddings[:, :min(n_embed, 8)], axis=1)

        # kron_trend_strength: correlation between current embedding
        # and embedding shifted by 1 row.  Computed as a rolling metric.
        trend = np.zeros(n)
        for i in range(1, n):
            cur = embeddings[i, :min(n_embed, 8)]
            prev = embeddings[i - 1, :min(n_embed, 8)]
            denom = (np.linalg.norm(cur) * np.linalg.norm(prev))
            if denom > 1e-12:
                trend[i] = float(np.dot(cur, prev) / denom)
            # else stays 0.0
        df["kron_trend_strength"] = trend

        # kron_regime_proxy: sign of mean of first embedding component
        # over a trailing window (same length as self.window).
        regime = np.zeros(n)
        embed0 = embeddings[:, 0]
        for i in range(self.window - 1, n):
            mean_val = np.mean(embed0[i - self.window + 1 : i + 1])
            regime[i] = float(np.sign(mean_val)) if abs(mean_val) > 1e-12 else 0.0
        df["kron_regime_proxy"] = regime

        df = self._cleanup_features(df)

        n_feat = sum(1 for c in df.columns if c.startswith("kron_"))
        logger.info("KronosFeatures: added %d features", n_feat)
        return df

    # ------------------------------------------------------------------
    # Backward-compatible class method
    # ------------------------------------------------------------------

    @staticmethod
    def _all_feature_names() -> List[str]:
        return list(KronosFeatures.FEATURE_NAMES)
