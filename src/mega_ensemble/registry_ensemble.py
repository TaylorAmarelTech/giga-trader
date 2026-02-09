"""
Registry Ensemble for Mega Ensemble.

Combines diverse models from registry using:
- Soft/weighted voting
- Stacking with meta-learner
"""

import sys
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import numpy as np
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model_registry_v2 import ModelEntry

logger = logging.getLogger(__name__)


@dataclass
class RegistryEnsembleConfig:
    """Configuration for registry-based ensemble."""

    # Voting settings
    voting_method: str = "soft"           # "hard", "soft"
    use_auc_weights: bool = True          # Weight by CV AUC

    # Stacking settings
    stacking_enabled: bool = True
    stacking_meta_model: str = "logistic"  # "logistic", "gradient_boosting"
    stacking_cv_folds: int = 3
    stacking_meta_C: float = 1.0           # Regularization for logistic meta

    # Calibration
    calibrate_probabilities: bool = False


class RegistryEnsemble:
    """
    Ensemble layer that combines diverse models from the registry.

    Supports:
    - Soft/hard voting with optional AUC weighting
    - Stacking with meta-learner (logistic or gradient boosting)

    The stacking approach uses out-of-fold predictions to avoid leakage.
    """

    def __init__(
        self,
        selected_models: List[ModelEntry],
        config: RegistryEnsembleConfig = None,
    ):
        self.selected_models = selected_models
        self.config = config or RegistryEnsembleConfig()

        # Computed during fit
        self.model_weights: Optional[np.ndarray] = None
        self.meta_model: Optional[Any] = None
        self.is_fitted: bool = False

        # Cache for loaded models
        self._model_cache: Dict[str, Any] = {}

    def _load_model_pipeline(self, entry: ModelEntry) -> Dict[str, Any]:
        """Load model and preprocessing pipeline from artifacts."""
        if entry.model_id in self._model_cache:
            return self._model_cache[entry.model_id]

        if not entry.artifacts or not entry.artifacts.model_path:
            raise ValueError(f"Model {entry.model_id} has no artifact path")

        model_path = Path(entry.artifacts.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        artifacts = joblib.load(model_path)
        self._model_cache[entry.model_id] = artifacts
        return artifacts

    def _apply_pipeline(self, X: np.ndarray, artifacts: Dict) -> np.ndarray:
        """Apply preprocessing pipeline to features."""
        X_out = X.copy()

        selector = artifacts.get('feature_selector')
        reducer = artifacts.get('dim_reducer')
        scaler = artifacts.get('scaler')

        if selector is not None:
            X_out = selector.transform(X_out)

        if reducer is not None:
            X_out = reducer.transform(X_out)

        if scaler is not None:
            X_out = scaler.transform(X_out)

        return X_out

    def _get_model_predictions(
        self,
        X: np.ndarray,
        entries: List[ModelEntry] = None,
    ) -> np.ndarray:
        """Get predictions from all models."""
        if entries is None:
            entries = self.selected_models

        predictions = []
        for entry in entries:
            try:
                artifacts = self._load_model_pipeline(entry)
                model = artifacts.get('model')

                X_processed = self._apply_pipeline(X, artifacts)

                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X_processed)[:, 1]
                else:
                    pred = model.predict(X_processed).astype(float)

                predictions.append(pred)

            except Exception as e:
                logger.warning(f"Failed to get predictions from {entry.model_id}: {e}")
                # Use neutral prediction (0.5) for failed models
                predictions.append(np.full(len(X), 0.5))

        return np.column_stack(predictions)

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
    ) -> Dict[str, Any]:
        """
        Fit the ensemble.

        For voting: compute weights from AUC scores.
        For stacking: generate OOF predictions and fit meta-model.

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional, for evaluation)
            y_val: Validation labels (optional, for evaluation)

        Returns:
            Dict with fitting metrics
        """
        logger.info("=" * 60)
        logger.info("REGISTRY ENSEMBLE - Fitting")
        logger.info("=" * 60)

        metrics = {}

        # Compute voting weights from AUC
        if self.config.use_auc_weights:
            aucs = np.array([m.metrics.cv_auc for m in self.selected_models])
            # Square to emphasize differences, shift so AUC=0.5 gets zero weight
            weights = np.maximum(aucs - 0.5, 0) ** 2
            if weights.sum() > 0:
                self.model_weights = weights / weights.sum()
            else:
                self.model_weights = np.ones(len(self.selected_models)) / len(self.selected_models)
        else:
            self.model_weights = np.ones(len(self.selected_models)) / len(self.selected_models)

        logger.info(f"Model weights (AUC-based): {self.model_weights.round(3)}")
        metrics["model_weights"] = self.model_weights.tolist()

        # Stacking: generate OOF predictions and fit meta-model
        if self.config.stacking_enabled and len(self.selected_models) >= 2:
            logger.info(f"Training stacking meta-model ({self.config.stacking_meta_model})...")

            oof_preds = self._generate_oof_predictions(X_train, y_train)

            # Fit meta-model
            if self.config.stacking_meta_model == "logistic":
                self.meta_model = LogisticRegression(
                    C=self.config.stacking_meta_C,
                    max_iter=1000,
                    solver='lbfgs',
                )
            else:
                self.meta_model = GradientBoostingClassifier(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                )

            self.meta_model.fit(oof_preds, y_train)
            metrics["stacking_fitted"] = True
            logger.info("  Stacking meta-model fitted")

            # Evaluate on validation if provided
            if X_val is not None and y_val is not None:
                val_preds = self._get_model_predictions(X_val)
                stacking_pred = self.meta_model.predict_proba(val_preds)[:, 1]
                voting_pred = np.average(val_preds, axis=1, weights=self.model_weights)

                from sklearn.metrics import roc_auc_score
                stacking_auc = roc_auc_score(y_val, stacking_pred)
                voting_auc = roc_auc_score(y_val, voting_pred)

                metrics["val_stacking_auc"] = float(stacking_auc)
                metrics["val_voting_auc"] = float(voting_auc)

                logger.info(f"  Validation Voting AUC: {voting_auc:.4f}")
                logger.info(f"  Validation Stacking AUC: {stacking_auc:.4f}")

        self.is_fitted = True
        logger.info("=" * 60)

        return metrics

    def _generate_oof_predictions(
        self,
        X: np.ndarray,
        y: np.ndarray,
    ) -> np.ndarray:
        """
        Generate out-of-fold predictions for stacking.

        Uses existing trained models (doesn't retrain them),
        just organizes predictions in OOF manner.
        """
        n_samples = len(X)
        n_models = len(self.selected_models)
        oof_preds = np.zeros((n_samples, n_models))

        kf = KFold(n_splits=self.config.stacking_cv_folds, shuffle=False)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_val_fold = X[val_idx]

            # Get predictions from each model on this fold's validation set
            for model_idx, entry in enumerate(self.selected_models):
                try:
                    artifacts = self._load_model_pipeline(entry)
                    model = artifacts.get('model')
                    X_processed = self._apply_pipeline(X_val_fold, artifacts)

                    if hasattr(model, 'predict_proba'):
                        pred = model.predict_proba(X_processed)[:, 1]
                    else:
                        pred = model.predict(X_processed).astype(float)

                    oof_preds[val_idx, model_idx] = pred

                except Exception as e:
                    logger.warning(f"OOF prediction failed for {entry.model_id}: {e}")
                    oof_preds[val_idx, model_idx] = 0.5

        return oof_preds

    def predict_voting(self, X: np.ndarray) -> np.ndarray:
        """
        Get predictions using weighted voting.

        Args:
            X: Features to predict

        Returns:
            Weighted average of model predictions
        """
        predictions = self._get_model_predictions(X)

        if self.config.voting_method == "soft":
            return np.average(predictions, axis=1, weights=self.model_weights)
        else:  # hard voting
            hard_preds = (predictions > 0.5).astype(float)
            return np.average(hard_preds, axis=1, weights=self.model_weights)

    def predict_stacking(self, X: np.ndarray) -> np.ndarray:
        """
        Get predictions using stacking meta-model.

        Args:
            X: Features to predict

        Returns:
            Meta-model predictions based on base model outputs
        """
        if self.meta_model is None:
            raise ValueError("Stacking meta-model not fitted. Call fit() first.")

        # Get base model predictions
        base_preds = self._get_model_predictions(X)
        return self.meta_model.predict_proba(base_preds)[:, 1]

    def predict(
        self,
        X: np.ndarray,
        method: str = "stacking",
    ) -> np.ndarray:
        """
        Get ensemble predictions.

        Args:
            X: Features to predict
            method: "voting" or "stacking"

        Returns:
            Ensemble predictions
        """
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted. Call fit() first.")

        if method == "voting":
            return self.predict_voting(X)
        else:
            if self.meta_model is not None:
                return self.predict_stacking(X)
            # Fall back to voting if stacking not available
            return self.predict_voting(X)

    def get_model_contributions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get individual model contributions for interpretability.

        Args:
            X: Features to predict

        Returns:
            Dict mapping model_id to predictions
        """
        contributions = {}
        predictions = self._get_model_predictions(X)

        for i, entry in enumerate(self.selected_models):
            contributions[entry.model_id] = predictions[:, i]

        return contributions

    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get summary information about the ensemble."""
        return {
            "n_models": len(self.selected_models),
            "model_types": [m.model_config.model_type for m in self.selected_models],
            "model_ids": [m.model_id for m in self.selected_models],
            "model_aucs": [m.metrics.cv_auc for m in self.selected_models],
            "model_weights": self.model_weights.tolist() if self.model_weights is not None else [],
            "stacking_enabled": self.config.stacking_enabled and self.meta_model is not None,
            "voting_method": self.config.voting_method,
        }


if __name__ == "__main__":
    # Test basic functionality
    config = RegistryEnsembleConfig(
        voting_method="soft",
        use_auc_weights=True,
        stacking_enabled=True,
        stacking_meta_model="logistic",
    )
    print(f"Registry ensemble config: {config}")
