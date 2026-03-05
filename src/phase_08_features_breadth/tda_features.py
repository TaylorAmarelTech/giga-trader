"""
GIGA TRADER - TDA Persistent Homology Features
================================================
Topological Data Analysis features using persistent homology.

Embeds the price time series into a point cloud via Takens' delay embedding,
computes persistent homology (tracks birth/death of topological features
like connected components and loops), and extracts summary statistics.

5 features generated (prefix: tda_).

Requires giotto-tda (optional).  Gracefully degrades to 0.0 when absent.
Best suited as a slow-moving regime/stress indicator (meaningful on 60-120d
windows, not useful for daily direction prediction).
"""

import logging
import warnings
from typing import Optional, Dict

import numpy as np
import pandas as pd
from src.core.feature_base import FeatureModuleBase

warnings.filterwarnings("ignore")

logger = logging.getLogger("TDA")


class TDAHomologyFeatures(FeatureModuleBase):
    """
    Compute persistent homology features from daily returns.

    All features use the tda_ prefix.  Requires giotto-tda (optional).
    Falls back to 0.0 when giotto-tda is not installed.
    """
    FEATURE_NAMES = ["tda_h0_max_persistence", "tda_h1_max_persistence", "tda_persistence_entropy", "tda_betti_1_count", "tda_amplitude"]


    REQUIRED_COLS = {"close"}

    def __init__(self, max_dim: int = 1, delay: int = 1, embedding_dim: int = 3):
        self._max_dim = max_dim
        self._delay = delay
        self._embedding_dim = embedding_dim
        self._gtda_available = False

        try:
            from gtda.homology import VietorisRipsPersistence  # noqa: F401
            from gtda.time_series import SingleTakensEmbedding  # noqa: F401
            self._gtda_available = True
        except ImportError:
            self._gtda_available = False

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def create_tda_features(
        self,
        spy_daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create TDA persistent homology features and merge into spy_daily.

        Returns spy_daily with new tda_* columns added.
        """
        df = spy_daily.copy()

        print(f"\n[TDA] Engineering persistent homology features "
              f"(available={self._gtda_available})...")

        # Initialize all features to defaults
        for name in self._all_feature_names():
            df[name] = 0.0

        if "close" not in df.columns:
            print("  [WARN] No 'close' column — skipping TDA")
            return df

        close = df["close"].values.astype(np.float64)
        n = len(close)

        if n < 60:
            print("  [WARN] Insufficient data (<60 rows) — skipping")
            return df

        if not self._gtda_available:
            print("  [INFO] giotto-tda not installed — features set to 0.0")
            return df

        # Compute with giotto-tda
        try:
            self._compute_tda_features(df, close, n)
        except Exception as e:
            print(f"  [WARN] TDA feature computation failed: {e}")

        # Cleanup
        for col in self._all_feature_names():
            df[col] = df[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)

        n_added = len(self._all_feature_names())
        print(f"  [TDA] Total: {n_added} persistent homology features added")
        return df

    def analyze_current_topology(
        self,
        spy_daily: pd.DataFrame,
    ) -> Optional[Dict]:
        """Return snapshot of current topological state."""
        if "tda_h0_max_persistence" not in spy_daily.columns or len(spy_daily) < 2:
            return None

        last = spy_daily.iloc[-1]
        h0 = float(last.get("tda_h0_max_persistence", 0.0))
        h1 = float(last.get("tda_h1_max_persistence", 0.0))
        pe = float(last.get("tda_persistence_entropy", 0.0))

        if h1 > 0.5:
            topology = "COMPLEX"  # Significant loops in phase space
        elif h0 > 2.0:
            topology = "FRAGMENTED"  # Many disconnected components
        else:
            topology = "SIMPLE"

        return {
            "topology": topology,
            "h0_persistence": round(h0, 4),
            "h1_persistence": round(h1, 4),
            "persistence_entropy": round(pe, 4),
            "gtda_available": self._gtda_available,
        }

    # ------------------------------------------------------------------ #
    #  Private helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _all_feature_names():
        return [
            "tda_h0_max_persistence",
            "tda_h1_max_persistence",
            "tda_persistence_entropy",
            "tda_betti_1_count",
            "tda_amplitude",
        ]

    def _compute_tda_features(self, df, close, n):
        """Compute TDA features using giotto-tda."""
        from gtda.homology import VietorisRipsPersistence

        returns = np.empty(n)
        returns[0] = 0.0
        returns[1:] = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-10)

        window = 60
        step = 10  # Recompute every 10 rows (TDA is expensive)

        h0_max = np.zeros(n)
        h1_max = np.zeros(n)
        pers_entropy = np.zeros(n)
        betti_1 = np.zeros(n)
        amplitude = np.zeros(n)

        vr = VietorisRipsPersistence(
            homology_dimensions=[0, self._max_dim],
            n_jobs=1,
        )

        for i in range(window, n, step):
            try:
                seg = returns[i - window:i]

                # Takens delay embedding (manual — avoid SingleTakensEmbedding overhead)
                point_cloud = self._takens_embed(seg, self._embedding_dim, self._delay)

                if len(point_cloud) < 5:
                    continue

                # Compute persistence diagram
                # VR expects (n_samples, n_points, n_dims)
                diagrams = vr.fit_transform(point_cloud[np.newaxis, :, :])
                diag = diagrams[0]  # (n_features, 3) — [birth, death, dimension]

                # Extract H0 features (connected components)
                h0_mask = diag[:, 2] == 0
                h0_lifetimes = diag[h0_mask, 1] - diag[h0_mask, 0]
                h0_finite = h0_lifetimes[np.isfinite(h0_lifetimes)]

                if len(h0_finite) > 0:
                    h0_max_val = np.max(h0_finite)
                else:
                    h0_max_val = 0.0

                # Extract H1 features (loops)
                h1_mask = diag[:, 2] == self._max_dim
                h1_lifetimes = diag[h1_mask, 1] - diag[h1_mask, 0]
                h1_finite = h1_lifetimes[np.isfinite(h1_lifetimes)]

                if len(h1_finite) > 0:
                    h1_max_val = np.max(h1_finite)
                    betti_count = len(h1_finite)
                else:
                    h1_max_val = 0.0
                    betti_count = 0

                # Persistence entropy
                all_finite = np.concatenate([h0_finite, h1_finite]) if len(h1_finite) > 0 else h0_finite
                if len(all_finite) > 1:
                    total = np.sum(all_finite)
                    if total > 1e-10:
                        probs = all_finite / total
                        probs = probs[probs > 0]
                        pe = -np.sum(probs * np.log(probs))
                    else:
                        pe = 0.0
                else:
                    pe = 0.0

                # Amplitude (average persistence)
                amp = np.mean(all_finite) if len(all_finite) > 0 else 0.0

                # Forward-fill
                fill_end = min(i + step, n)
                for j in range(i, fill_end):
                    h0_max[j] = h0_max_val
                    h1_max[j] = h1_max_val
                    pers_entropy[j] = pe
                    betti_1[j] = float(betti_count)
                    amplitude[j] = amp

            except Exception:
                continue

        df["tda_h0_max_persistence"] = h0_max
        df["tda_h1_max_persistence"] = h1_max
        df["tda_persistence_entropy"] = pers_entropy
        df["tda_betti_1_count"] = betti_1
        df["tda_amplitude"] = amplitude

    @staticmethod
    def _takens_embed(x: np.ndarray, dim: int, delay: int) -> np.ndarray:
        """Create Takens delay embedding point cloud."""
        n = len(x)
        n_points = n - (dim - 1) * delay
        if n_points < 1:
            return np.empty((0, dim))
        cloud = np.empty((n_points, dim))
        for d in range(dim):
            cloud[:, d] = x[d * delay:d * delay + n_points]
        return cloud
