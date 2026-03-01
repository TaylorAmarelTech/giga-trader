"""
GIGA TRADER - Interaction Discovery
=====================================
Systematically discovers informative pairwise feature products and ratios.

Uses mutual information to select top individual features, then tests
pairwise combinations and keeps those with MI > threshold relative to
individual features.
"""

import logging
from itertools import combinations
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

logger = logging.getLogger(__name__)


class InteractionDiscovery:
    """
    Discovers informative pairwise feature interactions (products and ratios).

    Uses mutual information to select top individual features, then tests
    pairwise combinations and keeps those with MI > threshold relative to
    individual features.
    """

    def __init__(
        self,
        max_interactions: int = 20,
        mi_threshold_multiplier: float = 1.5,
        top_k_features: int = 15,
        operations: Optional[List[str]] = None,
        random_state: int = 42,
    ):
        """
        Args:
            max_interactions: Maximum number of interactions to discover.
            mi_threshold_multiplier: Keep interactions with MI > multiplier * max(MI_a, MI_b).
            top_k_features: Number of top-MI features to consider for interactions.
            operations: List of operations to try. Default: ["product", "ratio"].
            random_state: Random seed for MI estimation.
        """
        self.max_interactions = max_interactions
        self.mi_threshold_multiplier = mi_threshold_multiplier
        self.top_k_features = top_k_features
        self.operations = operations or ["product", "ratio"]
        self.random_state = random_state
        self._interactions: List[Tuple[str, str, str]] = []  # (feat_a, feat_b, operation)
        self._mi_scores: List[float] = []  # MI score for each discovered interaction
        self._fitted = False

    def discover(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> List[Tuple[str, str, str]]:
        """
        Discover informative feature interactions.

        Args:
            X: Feature DataFrame with named columns.
            y: Binary target labels.

        Returns:
            List of (feature_a_name, feature_b_name, operation) tuples.
            Sorted by MI score descending.

        Algorithm:
            1. Compute MI for each individual feature.
            2. Select top_k_features by MI score.
            3. For each pair (a, b) from top-k, for each operation:
               - Compute interaction feature.
               - Compute MI of interaction with target.
               - If MI > mi_threshold_multiplier * max(MI_a, MI_b): keep it.
            4. Sort by MI descending, keep top max_interactions.
        """
        y_arr = np.asarray(y).ravel()
        feature_names = list(X.columns)
        n_features = len(feature_names)

        if n_features == 0:
            logger.warning("No features provided; returning empty interactions.")
            self._interactions = []
            self._mi_scores = []
            self._fitted = True
            return self._interactions.copy()

        # Step 1: Compute MI for each individual feature
        X_safe = X.copy()
        X_safe = X_safe.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        individual_mi = mutual_info_classif(
            X_safe.values, y_arr, random_state=self.random_state,
        )
        mi_by_name: Dict[str, float] = {
            name: score for name, score in zip(feature_names, individual_mi)
        }

        logger.info(
            "Individual MI scores computed for %d features (max=%.4f, min=%.4f).",
            n_features,
            float(np.max(individual_mi)),
            float(np.min(individual_mi)),
        )

        # Step 2: Select top-k features (use all if fewer than 5)
        if n_features <= 5:
            top_k = feature_names
        else:
            k = min(self.top_k_features, n_features)
            sorted_indices = np.argsort(individual_mi)[::-1][:k]
            top_k = [feature_names[i] for i in sorted_indices]

        logger.info("Selected %d top features for interaction search.", len(top_k))

        # Step 3: Test pairwise interactions
        candidates: List[Tuple[str, str, str, float]] = []  # (a, b, op, mi_score)
        seen_pairs: set = set()

        for feat_a, feat_b in combinations(sorted(top_k), 2):
            # Deterministic ordering: sorted pair names
            pair_key_base = (feat_a, feat_b) if feat_a <= feat_b else (feat_b, feat_a)

            for op in self.operations:
                pair_key = (pair_key_base[0], pair_key_base[1], op)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                # Compute interaction feature
                a_vals = X_safe[feat_a].values
                b_vals = X_safe[feat_b].values

                if op == "product":
                    interaction = self._compute_product(a_vals, b_vals)
                elif op == "ratio":
                    interaction = self._compute_ratio(a_vals, b_vals)
                else:
                    logger.warning("Unknown operation '%s', skipping.", op)
                    continue

                # Clean inf/nan
                interaction = np.where(np.isfinite(interaction), interaction, 0.0)

                # Compute MI of interaction with target
                interaction_2d = interaction.reshape(-1, 1)
                try:
                    mi_score = mutual_info_classif(
                        interaction_2d, y_arr, random_state=self.random_state,
                    )[0]
                except Exception:
                    continue

                # Check threshold: MI > multiplier * max(MI_a, MI_b)
                max_individual = max(
                    mi_by_name.get(pair_key_base[0], 0.0),
                    mi_by_name.get(pair_key_base[1], 0.0),
                )
                threshold = self.mi_threshold_multiplier * max_individual

                if mi_score > threshold:
                    candidates.append((pair_key_base[0], pair_key_base[1], op, mi_score))

        # Step 4: Sort by MI descending, keep top max_interactions
        candidates.sort(key=lambda x: x[3], reverse=True)
        candidates = candidates[: self.max_interactions]

        self._interactions = [(a, b, op) for a, b, op, _ in candidates]
        self._mi_scores = [score for _, _, _, score in candidates]
        self._fitted = True

        logger.info(
            "Discovered %d informative interactions (tested %d pairs).",
            len(self._interactions),
            len(seen_pairs),
        )

        return self._interactions.copy()

    def transform(
        self,
        X: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add discovered interaction features to DataFrame.

        Args:
            X: Feature DataFrame.

        Returns:
            X augmented with interaction columns named like "ix_{feat_a}_{op}_{feat_b}".

        Raises:
            RuntimeError: If discover() has not been called yet.
        """
        if not self._fitted:
            raise RuntimeError(
                "InteractionDiscovery.discover() must be called before transform()."
            )

        if not self._interactions:
            return X.copy()

        result = X.copy()

        for feat_a, feat_b, op in self._interactions:
            col_name = self._make_column_name(feat_a, feat_b, op)

            if feat_a not in X.columns or feat_b not in X.columns:
                logger.warning(
                    "Feature '%s' or '%s' not in DataFrame; skipping interaction.",
                    feat_a, feat_b,
                )
                continue

            a_vals = X[feat_a].values
            b_vals = X[feat_b].values

            if op == "product":
                interaction = self._compute_product(a_vals, b_vals)
            elif op == "ratio":
                interaction = self._compute_ratio(a_vals, b_vals)
            else:
                continue

            # Clean inf/nan to 0.0
            interaction = np.where(np.isfinite(interaction), interaction, 0.0)
            result[col_name] = interaction

        return result

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
    ) -> pd.DataFrame:
        """Discover interactions and transform in one step."""
        self.discover(X, y)
        return self.transform(X)

    @staticmethod
    def _compute_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute element-wise product."""
        return a * b

    @staticmethod
    def _compute_ratio(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """Compute element-wise ratio with safe division (avoid div by zero)."""
        return a / np.where(np.abs(b) > 1e-10, b, 1e-10)

    @staticmethod
    def _make_column_name(feat_a: str, feat_b: str, op: str) -> str:
        """
        Create deterministic interaction column name.

        Truncates long feature names to keep column names manageable.
        Format: ix_{feat_a}_{op}_{feat_b}
        """
        max_feat_len = 30
        a_trunc = feat_a[:max_feat_len]
        b_trunc = feat_b[:max_feat_len]
        return f"ix_{a_trunc}_{op}_{b_trunc}"

    @property
    def interactions(self) -> List[Tuple[str, str, str]]:
        """Return discovered interactions."""
        return self._interactions.copy()

    @property
    def n_interactions(self) -> int:
        """Number of discovered interactions."""
        return len(self._interactions)

    @property
    def mi_scores(self) -> List[float]:
        """MI scores for discovered interactions (same order as interactions)."""
        return self._mi_scores.copy()
