"""
Diversity Selector for Mega Ensemble.

Selects diverse models from registry for ensemble using:
- Pairwise disagreement (1 - |correlation|)
- Anti-correlation bonus (right when others wrong)
- Greedy selection maximizing AUC + diversity
"""

import sys
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import numpy as np
import joblib

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model_registry_v2 import ModelRegistryV2, ModelEntry

logger = logging.getLogger(__name__)


@dataclass
class DiversityConfig:
    """Configuration for diversity-based model selection."""

    # Selection criteria
    min_auc_threshold: float = 0.55      # Minimum AUC to consider
    max_gap_threshold: float = 0.20      # Maximum train-test gap
    diversity_weight: float = 0.3        # Weight for diversity vs AUC (0-1)
    anti_corr_weight: float = 0.1        # Weight for anti-correlation bonus

    # Selection parameters
    n_models_to_select: int = 10         # Number of models to select
    max_correlation: float = 0.95        # Max allowed prediction correlation

    # Selection method
    selection_method: str = "greedy"      # "greedy", "pareto"


class DiversitySelector:
    """
    Selects diverse models from registry for ensemble.

    Key insight: Models that DISAGREE on predictions are more valuable
    in an ensemble than models that agree (even if all are accurate).

    Algorithm:
    1. Get predictions from all candidate models on validation set
    2. Compute pairwise disagreement matrix (1 - |correlation|)
    3. Compute anti-correlation bonus (right when others wrong)
    4. Greedy selection: maximize (AUC + diversity + anti-corr bonus)
    """

    def __init__(self, config: DiversityConfig = None):
        self.config = config or DiversityConfig()

    def compute_prediction_matrix(
        self,
        models: List[ModelEntry],
        X_val: np.ndarray,
        feature_cols: List[str] = None,
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Get predictions from all models on validation set.

        Args:
            models: List of ModelEntry objects
            X_val: Validation features (n_samples, n_features)
            feature_cols: Feature column names (optional)

        Returns:
            Tuple of:
                - predictions: np.ndarray of shape (n_samples, n_valid_models)
                - valid_indices: List of indices of models that predicted successfully
        """
        predictions = []
        valid_indices = []

        for i, entry in enumerate(models):
            try:
                # Check if model artifact exists
                if not entry.artifacts or not entry.artifacts.model_path:
                    logger.warning(f"Model {entry.model_id} has no artifact path")
                    continue

                model_path = Path(entry.artifacts.model_path)
                if not model_path.exists():
                    logger.warning(f"Model file not found: {model_path}")
                    continue

                # Load model artifacts
                artifacts = joblib.load(model_path)

                # Get pipeline components
                model = artifacts.get('model')
                scaler = artifacts.get('scaler')
                selector = artifacts.get('feature_selector')
                reducer = artifacts.get('dim_reducer')

                if model is None:
                    logger.warning(f"Model {entry.model_id} has no 'model' key")
                    continue

                # Apply pipeline
                X = X_val.copy()

                if selector is not None:
                    X = selector.transform(X)

                if reducer is not None:
                    X = reducer.transform(X)

                if scaler is not None:
                    X = scaler.transform(X)

                # Get predictions
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)[:, 1]
                else:
                    pred = model.predict(X)

                predictions.append(pred)
                valid_indices.append(i)

            except Exception as e:
                logger.warning(f"Failed to get predictions from {entry.model_id}: {e}")
                continue

        if not predictions:
            return np.array([]), []

        return np.column_stack(predictions), valid_indices

    def compute_disagreement_matrix(
        self,
        pred_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Compute pairwise disagreement between models.

        Disagreement = 1 - |correlation of predictions|
        High disagreement = more valuable for ensemble.

        Args:
            pred_matrix: Predictions, shape (n_samples, n_models)

        Returns:
            np.ndarray of shape (n_models, n_models)
        """
        if pred_matrix.shape[1] < 2:
            return np.array([[0.0]])

        # Compute correlation matrix
        corr = np.corrcoef(pred_matrix.T)

        # Handle NaN correlations (constant predictions)
        corr = np.nan_to_num(corr, nan=0.0)

        # Disagreement = 1 - |correlation|
        disagreement = 1.0 - np.abs(corr)

        return disagreement

    def compute_anti_correlation_bonus(
        self,
        pred_matrix: np.ndarray,
        y_true: np.ndarray,
    ) -> np.ndarray:
        """
        Compute anti-correlation bonus: reward for being right when others are wrong.

        For each model: avg(this_correct AND others_wrong)

        Args:
            pred_matrix: Predictions, shape (n_samples, n_models)
            y_true: True labels, shape (n_samples,)

        Returns:
            np.ndarray of shape (n_models,) with bonus scores
        """
        n_models = pred_matrix.shape[1]
        bonuses = np.zeros(n_models)

        if n_models < 2:
            return bonuses

        for i in range(n_models):
            pred_i = (pred_matrix[:, i] > 0.5).astype(int)
            correct_i = (pred_i == y_true)

            anti_corr_sum = 0.0
            for j in range(n_models):
                if i == j:
                    continue

                pred_j = (pred_matrix[:, j] > 0.5).astype(int)
                wrong_j = (pred_j != y_true)

                # Bonus when i is right and j is wrong
                anti_corr = np.mean(correct_i & wrong_j)
                anti_corr_sum += anti_corr

            bonuses[i] = anti_corr_sum / (n_models - 1)

        return bonuses

    def select_diverse_models_greedy(
        self,
        models: List[ModelEntry],
        auc_scores: np.ndarray,
        disagreement_matrix: np.ndarray,
        anti_corr_bonus: np.ndarray = None,
    ) -> List[int]:
        """
        Greedy selection: iteratively add model that maximizes
        (AUC contribution + diversity contribution + anti-correlation bonus).

        Args:
            models: List of ModelEntry
            auc_scores: AUC scores for each model
            disagreement_matrix: Pairwise disagreement
            anti_corr_bonus: Anti-correlation bonus scores

        Returns:
            List of indices of selected models
        """
        n_models = len(models)
        n_select = min(self.config.n_models_to_select, n_models)

        if n_models == 0:
            return []

        selected = []
        remaining = list(range(n_models))

        # Filter by minimum AUC
        remaining = [i for i in remaining if auc_scores[i] >= self.config.min_auc_threshold]

        if not remaining:
            logger.warning(f"No models meet AUC threshold {self.config.min_auc_threshold}")
            return []

        for _ in range(n_select):
            if not remaining:
                break

            best_score = -np.inf
            best_idx = None

            for i in remaining:
                # AUC contribution (normalized to 0-1 assuming AUC in [0.5, 1])
                auc_contrib = (auc_scores[i] - 0.5) * 2

                # Diversity contribution (avg disagreement with selected models)
                if selected:
                    div_contrib = np.mean([disagreement_matrix[i, j] for j in selected])
                else:
                    div_contrib = 1.0  # First model gets full diversity credit

                # Anti-correlation bonus
                acb = anti_corr_bonus[i] if anti_corr_bonus is not None else 0

                # Combined score
                score = (
                    (1 - self.config.diversity_weight - self.config.anti_corr_weight) * auc_contrib +
                    self.config.diversity_weight * div_contrib +
                    self.config.anti_corr_weight * acb
                )

                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx is not None:
                selected.append(best_idx)
                remaining.remove(best_idx)

        return selected

    def select(
        self,
        registry: ModelRegistryV2,
        X_val: np.ndarray,
        y_val: np.ndarray,
        target_type: str = "swing",
    ) -> Tuple[List[ModelEntry], Dict[str, Any]]:
        """
        Main selection interface.

        Args:
            registry: Model registry to select from
            X_val: Validation features
            y_val: Validation labels
            target_type: Target type to filter by

        Returns:
            Tuple of (selected_models, selection_metrics)
        """
        logger.info("=" * 60)
        logger.info("DIVERSITY SELECTOR - Selecting ensemble members")
        logger.info("=" * 60)

        # Get all trained models above threshold
        all_models = registry.query(
            target_type=target_type,
            status="trained",
            min_cv_auc=self.config.min_auc_threshold,
        )

        logger.info(f"Found {len(all_models)} candidate models (AUC >= {self.config.min_auc_threshold})")

        if len(all_models) < 2:
            logger.warning("Insufficient models for diversity selection")
            return all_models, {"status": "insufficient_models", "n_candidates": len(all_models)}

        # Filter by max gap
        filtered_models = [
            m for m in all_models
            if m.metrics.train_test_gap <= self.config.max_gap_threshold
        ]
        logger.info(f"After gap filter (<= {self.config.max_gap_threshold}): {len(filtered_models)} models")

        if len(filtered_models) < 2:
            logger.warning("Insufficient models after gap filter, using all candidates")
            filtered_models = all_models

        # Get predictions
        logger.info("Computing predictions on validation set...")
        pred_matrix, valid_indices = self.compute_prediction_matrix(filtered_models, X_val)

        if len(valid_indices) < 2:
            logger.warning(f"Only {len(valid_indices)} models produced valid predictions")
            return [filtered_models[i] for i in valid_indices], {
                "status": "insufficient_predictions",
                "n_valid": len(valid_indices),
            }

        # Get valid models and scores
        valid_models = [filtered_models[i] for i in valid_indices]
        auc_scores = np.array([m.metrics.cv_auc for m in valid_models])

        logger.info(f"Computing disagreement matrix for {len(valid_models)} models...")

        # Compute disagreement
        disagreement = self.compute_disagreement_matrix(pred_matrix)

        # Compute anti-correlation bonus
        logger.info("Computing anti-correlation bonuses...")
        anti_corr = self.compute_anti_correlation_bonus(pred_matrix, y_val)

        # Select
        logger.info(f"Selecting {self.config.n_models_to_select} diverse models...")
        selected_indices = self.select_diverse_models_greedy(
            valid_models, auc_scores, disagreement, anti_corr
        )

        selected_models = [valid_models[i] for i in selected_indices]

        # Compute metrics
        if selected_indices:
            selected_disagreement = disagreement[np.ix_(selected_indices, selected_indices)]
            # Average off-diagonal elements
            mask = ~np.eye(len(selected_indices), dtype=bool)
            avg_disagreement = np.mean(selected_disagreement[mask]) if mask.sum() > 0 else 0.0
        else:
            avg_disagreement = 0.0

        metrics = {
            "status": "success",
            "n_candidates": len(all_models),
            "n_after_gap_filter": len(filtered_models),
            "n_valid_predictions": len(valid_indices),
            "n_selected": len(selected_models),
            "avg_auc": float(np.mean([m.metrics.cv_auc for m in selected_models])) if selected_models else 0,
            "min_auc": float(min([m.metrics.cv_auc for m in selected_models])) if selected_models else 0,
            "max_auc": float(max([m.metrics.cv_auc for m in selected_models])) if selected_models else 0,
            "avg_pairwise_disagreement": float(avg_disagreement),
            "selected_model_ids": [m.model_id for m in selected_models],
        }

        # Log results
        logger.info("")
        logger.info(f"Selected {len(selected_models)} diverse models:")
        for i, m in enumerate(selected_models):
            logger.info(f"  {i+1}. {m.model_id[:20]}... | {m.model_config.model_type} | "
                       f"AUC={m.metrics.cv_auc:.4f} | Gap={m.metrics.train_test_gap:.4f}")
        logger.info("")
        logger.info(f"Average pairwise disagreement: {avg_disagreement:.4f}")
        logger.info("=" * 60)

        return selected_models, metrics

    def select_from_entries(
        self,
        models: List[ModelEntry],
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[List[ModelEntry], Dict[str, Any]]:
        """
        Select diverse models from a provided list (without registry query).

        Args:
            models: List of ModelEntry to select from
            X_val: Validation features
            y_val: Validation labels

        Returns:
            Tuple of (selected_models, selection_metrics)
        """
        if len(models) < 2:
            return models, {"status": "insufficient_models", "n_candidates": len(models)}

        # Get predictions
        pred_matrix, valid_indices = self.compute_prediction_matrix(models, X_val)

        if len(valid_indices) < 2:
            return [models[i] for i in valid_indices], {
                "status": "insufficient_predictions",
                "n_valid": len(valid_indices),
            }

        valid_models = [models[i] for i in valid_indices]
        auc_scores = np.array([m.metrics.cv_auc for m in valid_models])

        disagreement = self.compute_disagreement_matrix(pred_matrix)
        anti_corr = self.compute_anti_correlation_bonus(pred_matrix, y_val)

        selected_indices = self.select_diverse_models_greedy(
            valid_models, auc_scores, disagreement, anti_corr
        )

        selected_models = [valid_models[i] for i in selected_indices]

        metrics = {
            "status": "success",
            "n_candidates": len(models),
            "n_selected": len(selected_models),
            "avg_auc": float(np.mean([m.metrics.cv_auc for m in selected_models])) if selected_models else 0,
        }

        return selected_models, metrics


if __name__ == "__main__":
    # Test basic functionality
    config = DiversityConfig(
        min_auc_threshold=0.55,
        n_models_to_select=5,
        diversity_weight=0.3,
    )
    selector = DiversitySelector(config)
    print(f"Diversity selector initialized with config: {config}")
