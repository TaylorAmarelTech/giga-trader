"""
Mega Ensemble - Multi-layer ensemble for robust predictions.

Architecture:
    Layer 1: Extended Grid Search (base models) - training only
    Layer 2: Diversity-selected registry ensemble (voting + stacking)
    Layer 3: Interpolated config fabric (trained models)
    Layer 4: Temporal cascade ensemble (time-slice predictions)
    Layer 5: Final meta-learner combining all layers
"""

import sys
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model_registry_v2 import ModelRegistryV2, ModelEntry
from src.mega_ensemble.diversity_selector import DiversitySelector, DiversityConfig
from src.mega_ensemble.registry_ensemble import RegistryEnsemble, RegistryEnsembleConfig
from src.mega_ensemble.config_interpolator import FabricOfPoints, InterpolationConfig

logger = logging.getLogger(__name__)


@dataclass
class MegaEnsembleConfig:
    """Configuration for the mega ensemble."""

    # Layer weights for final combination
    layer_2_weight: float = 0.35    # Registry ensemble
    layer_3_weight: float = 0.25    # Interpolated fabric
    layer_4_weight: float = 0.25    # Temporal cascade
    cross_layer_weight: float = 0.15  # Cross-layer disagreement features

    # Final meta-learner settings
    final_meta_model: str = "logistic"  # "logistic", "gradient_boosting"
    final_meta_C: float = 1.0           # Regularization for logistic
    use_cross_layer_features: bool = True

    # Diversity selection settings
    diversity_n_models: int = 10
    diversity_weight: float = 0.3
    min_auc_threshold: float = 0.55

    # Fabric settings
    fabric_max_models: int = 30
    fabric_n_interpolation_points: int = 2

    # Output options
    output_cascade_lineage: bool = True
    output_confidence_decomposition: bool = True

    # Temporal cascade (Layer 4)
    use_temporal_cascade: bool = False  # Disabled by default (requires 1min data)


@dataclass
class CascadeLineage:
    """Tracks the full cascade path of a prediction."""

    timestamp: str

    # Layer 2 (Registry Ensemble)
    layer2_voting_pred: float
    layer2_stacking_pred: float
    layer2_model_contributions: Dict[str, float]

    # Layer 3 (Interpolated Fabric)
    layer3_predictions: Dict[str, float]
    layer3_avg_pred: float
    layer3_std_pred: float

    # Layer 4 (Temporal Cascade)
    layer4_temporal_predictions: Dict[str, float]
    layer4_agreement_score: float

    # Cross-layer metrics
    cross_layer_disagreement: float
    layer_agreement_vector: List[float]

    # Final output
    final_prediction: float
    final_confidence: float

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "timestamp": self.timestamp,
            "layer2": {
                "voting": self.layer2_voting_pred,
                "stacking": self.layer2_stacking_pred,
                "n_models": len(self.layer2_model_contributions),
            },
            "layer3": {
                "avg": self.layer3_avg_pred,
                "std": self.layer3_std_pred,
                "n_models": len(self.layer3_predictions),
            },
            "layer4": {
                "predictions": self.layer4_temporal_predictions,
                "agreement": self.layer4_agreement_score,
            },
            "cross_layer": {
                "disagreement": self.cross_layer_disagreement,
                "agreement_vector": self.layer_agreement_vector,
            },
            "final": {
                "prediction": self.final_prediction,
                "confidence": self.final_confidence,
            },
        }


class MegaEnsemble:
    """
    The final mega ensemble that combines all layers.

    Architecture:
        Layer 1: Extended grid search (base models) - training only, populates registry
        Layer 2: Diversity-selected registry ensemble (voting + stacking)
        Layer 3: Interpolated config fabric (trained models)
        Layer 4: Temporal cascade ensemble (optional, requires 1min data)
        Layer 5: Final meta-learner combining all layers

    Usage:
        mega = MegaEnsemble(config)
        mega.fit(registry, X_train, y_train, X_val, y_val)
        predictions, lineage = mega.predict(X_test, return_lineage=True)
    """

    def __init__(self, config: MegaEnsembleConfig = None):
        self.config = config or MegaEnsembleConfig()

        # Layer components (set during fit)
        self.registry_ensemble: Optional[RegistryEnsemble] = None
        self.fabric_models: List[ModelEntry] = []
        self.temporal_cascade: Optional[Any] = None

        # Final meta-learner
        self.final_meta_model: Optional[Any] = None

        # Cache for fabric model predictions
        self._fabric_model_cache: Dict[str, Any] = {}

        # Fit state
        self.is_fitted: bool = False
        self.fit_metrics: Dict[str, Any] = {}

    def fit(
        self,
        registry: ModelRegistryV2,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        trained_fabric_models: List[ModelEntry] = None,
        df_1min: pd.DataFrame = None,
    ) -> Dict[str, Any]:
        """
        Fit the mega ensemble.

        Steps:
        1. Select diverse models from registry (Layer 2)
        2. Fit registry ensemble (voting + stacking)
        3. Use provided fabric models or skip (Layer 3)
        4. Optionally train temporal cascade (Layer 4)
        5. Train final meta-learner (Layer 5)

        Args:
            registry: Model registry with trained base models
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            trained_fabric_models: Pre-trained fabric models (optional)
            df_1min: 1-minute data for temporal cascade (optional)

        Returns:
            Dict with fitting metrics
        """
        logger.info("")
        logger.info("=" * 70)
        logger.info("MEGA ENSEMBLE - Training")
        logger.info("=" * 70)
        logger.info("")

        metrics = {}

        # === Layer 2: Diversity Selection + Registry Ensemble ===
        logger.info("=" * 60)
        logger.info("LAYER 2: Diversity-Selected Registry Ensemble")
        logger.info("=" * 60)

        diversity_config = DiversityConfig(
            min_auc_threshold=self.config.min_auc_threshold,
            n_models_to_select=self.config.diversity_n_models,
            diversity_weight=self.config.diversity_weight,
        )
        diversity_selector = DiversitySelector(diversity_config)

        selected_models, selection_metrics = diversity_selector.select(
            registry, X_val, y_val, target_type="swing"
        )
        metrics["layer2_selection"] = selection_metrics

        if len(selected_models) < 2:
            logger.warning("Insufficient models for ensemble, using all available")
            selected_models = list(registry.models.values())[:10]

        # Fit registry ensemble
        ensemble_config = RegistryEnsembleConfig(
            voting_method="soft",
            use_auc_weights=True,
            stacking_enabled=True,
            stacking_meta_model="logistic",
        )
        self.registry_ensemble = RegistryEnsemble(selected_models, ensemble_config)
        ensemble_metrics = self.registry_ensemble.fit(X_train, y_train, X_val, y_val)
        metrics["layer2_ensemble"] = ensemble_metrics

        # === Layer 3: Interpolated Fabric ===
        logger.info("")
        logger.info("=" * 60)
        logger.info("LAYER 3: Interpolated Config Fabric")
        logger.info("=" * 60)

        if trained_fabric_models:
            self.fabric_models = [m for m in trained_fabric_models if m.status == "trained"]
            metrics["layer3_fabric"] = {
                "n_provided": len(trained_fabric_models),
                "n_trained": len(self.fabric_models),
            }
            logger.info(f"Using {len(self.fabric_models)} pre-trained fabric models")
        else:
            logger.info("No fabric models provided - Layer 3 will be skipped")
            metrics["layer3_fabric"] = {"status": "skipped"}

        # === Layer 4: Temporal Cascade (optional) ===
        if self.config.use_temporal_cascade and df_1min is not None:
            logger.info("")
            logger.info("=" * 60)
            logger.info("LAYER 4: Temporal Cascade")
            logger.info("=" * 60)

            try:
                from src.temporal_cascade_trainer import train_temporal_cascade

                cascade_result = train_temporal_cascade(df_1min=df_1min)
                if cascade_result.success:
                    self.temporal_cascade = joblib.load(cascade_result.model_path)
                    metrics["layer4_cascade"] = {"status": "trained"}
                    logger.info("Temporal cascade trained successfully")
                else:
                    metrics["layer4_cascade"] = {"status": "failed", "error": cascade_result.error}
                    logger.warning(f"Temporal cascade training failed: {cascade_result.error}")

            except ImportError:
                logger.warning("Temporal cascade trainer not available - skipping Layer 4")
                metrics["layer4_cascade"] = {"status": "import_error"}
            except Exception as e:
                logger.warning(f"Temporal cascade error: {e}")
                metrics["layer4_cascade"] = {"status": "error", "error": str(e)}
        else:
            logger.info("")
            logger.info("Layer 4 (Temporal Cascade): Skipped")
            metrics["layer4_cascade"] = {"status": "disabled"}

        # === Layer 5: Final Meta-Learner ===
        logger.info("")
        logger.info("=" * 60)
        logger.info("LAYER 5: Final Meta-Learner")
        logger.info("=" * 60)

        # Generate meta-features
        meta_features_train = self._generate_meta_features(X_train)
        meta_features_val = self._generate_meta_features(X_val)

        logger.info(f"Meta-feature dimensions: {meta_features_train.shape[1]}")

        # Fit final meta-model
        if self.config.final_meta_model == "logistic":
            self.final_meta_model = LogisticRegression(
                C=self.config.final_meta_C,
                max_iter=1000,
                solver='lbfgs',
            )
        else:
            self.final_meta_model = GradientBoostingClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
            )

        self.final_meta_model.fit(meta_features_train, y_train)

        # Evaluate
        val_pred = self.final_meta_model.predict_proba(meta_features_val)[:, 1]
        final_auc = roc_auc_score(y_val, val_pred)

        metrics["layer5_meta"] = {
            "meta_model": self.config.final_meta_model,
            "n_features": meta_features_train.shape[1],
            "val_auc": float(final_auc),
        }

        logger.info(f"Final meta-learner validation AUC: {final_auc:.4f}")

        self.is_fitted = True
        self.fit_metrics = metrics

        # Print summary
        self._print_fit_summary(metrics)

        return metrics

    def _generate_meta_features(self, X: np.ndarray) -> np.ndarray:
        """Generate meta-features from all layers for the final meta-learner."""
        features = []
        feature_names = []

        # Layer 2 predictions
        if self.registry_ensemble is not None and self.registry_ensemble.is_fitted:
            layer2_voting = self.registry_ensemble.predict_voting(X)
            layer2_stacking = self.registry_ensemble.predict_stacking(X)
            features.extend([layer2_voting, layer2_stacking])
            feature_names.extend(["l2_voting", "l2_stacking"])

        # Layer 3 predictions (average and std of fabric models)
        if self.fabric_models:
            fabric_preds = []
            for entry in self.fabric_models:
                try:
                    pred = self._get_fabric_model_prediction(entry, X)
                    fabric_preds.append(pred)
                except Exception as e:
                    logger.warning(f"Fabric model {entry.model_id} prediction failed: {e}")
                    continue

            if fabric_preds:
                fabric_preds = np.column_stack(fabric_preds)
                layer3_avg = np.mean(fabric_preds, axis=1)
                layer3_std = np.std(fabric_preds, axis=1)
                features.extend([layer3_avg, layer3_std])
                feature_names.extend(["l3_avg", "l3_std"])

        # Layer 4 predictions (if temporal cascade available)
        if self.temporal_cascade is not None:
            try:
                # Use a default time point (T60 = 60 minutes from open)
                layer4_preds = []
                for i in range(len(X)):
                    pred = self.temporal_cascade.predict_at_time({}, 60)
                    layer4_preds.append(pred.swing_direction if hasattr(pred, 'swing_direction') else 0.5)
                features.append(np.array(layer4_preds))
                feature_names.append("l4_t60")
            except Exception as e:
                logger.warning(f"Temporal cascade prediction failed: {e}")

        # Cross-layer disagreement features
        if self.config.use_cross_layer_features and len(features) >= 2:
            # Pairwise disagreement between layers
            for i in range(len(features)):
                for j in range(i + 1, len(features)):
                    disagree = np.abs(features[i] - features[j])
                    features.append(disagree)
                    feature_names.append(f"disagree_{feature_names[i]}_{feature_names[j]}")

        if not features:
            # Fallback: just use Layer 2 voting
            if self.registry_ensemble is not None:
                return self.registry_ensemble.predict_voting(X).reshape(-1, 1)
            raise ValueError("No features available for meta-learner")

        return np.column_stack(features)

    def _get_fabric_model_prediction(self, entry: ModelEntry, X: np.ndarray) -> np.ndarray:
        """Get predictions from a fabric model."""
        if entry.model_id in self._fabric_model_cache:
            artifacts = self._fabric_model_cache[entry.model_id]
        else:
            if not entry.artifacts or not entry.artifacts.model_path:
                raise ValueError(f"No artifact path for {entry.model_id}")

            model_path = Path(entry.artifacts.model_path)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found: {model_path}")

            artifacts = joblib.load(model_path)
            self._fabric_model_cache[entry.model_id] = artifacts

        model = artifacts.get('model')
        scaler = artifacts.get('scaler')
        selector = artifacts.get('feature_selector')
        reducer = artifacts.get('dim_reducer')

        X_out = X.copy()
        if selector is not None:
            X_out = selector.transform(X_out)
        if reducer is not None:
            X_out = reducer.transform(X_out)
        if scaler is not None:
            X_out = scaler.transform(X_out)

        if hasattr(model, 'predict_proba'):
            return model.predict_proba(X_out)[:, 1]
        return model.predict(X_out).astype(float)

    def predict(
        self,
        X: np.ndarray,
        return_lineage: bool = False,
    ) -> Tuple[np.ndarray, Optional[List[CascadeLineage]]]:
        """
        Make predictions with the mega ensemble.

        Args:
            X: Features to predict
            return_lineage: Whether to return cascade lineage

        Returns:
            Tuple of (predictions, optional cascade lineage)
        """
        if not self.is_fitted:
            raise ValueError("Mega ensemble not fitted. Call fit() first.")

        meta_features = self._generate_meta_features(X)
        predictions = self.final_meta_model.predict_proba(meta_features)[:, 1]

        lineage = None
        if return_lineage and self.config.output_cascade_lineage:
            lineage = self._generate_lineage(X, predictions)

        return predictions, lineage

    def _generate_lineage(
        self,
        X: np.ndarray,
        final_predictions: np.ndarray,
    ) -> List[CascadeLineage]:
        """Generate cascade lineage for each prediction."""
        lineages = []

        # Get layer predictions
        layer2_voting = self.registry_ensemble.predict_voting(X) if self.registry_ensemble else np.full(len(X), 0.5)
        layer2_stacking = self.registry_ensemble.predict_stacking(X) if self.registry_ensemble and self.registry_ensemble.meta_model else layer2_voting

        for i in range(len(X)):
            # Layer 3 info
            l3_preds = {}
            l3_values = []
            for entry in self.fabric_models[:5]:  # Limit to first 5 for lineage
                try:
                    pred = self._get_fabric_model_prediction(entry, X[i:i+1])[0]
                    l3_preds[entry.model_id[:12]] = float(pred)
                    l3_values.append(pred)
                except:
                    continue

            l3_avg = np.mean(l3_values) if l3_values else 0.5
            l3_std = np.std(l3_values) if l3_values else 0.0

            # Cross-layer disagreement
            layer_preds = [layer2_voting[i], layer2_stacking[i], l3_avg]
            cross_disagree = np.std(layer_preds)

            lineage = CascadeLineage(
                timestamp=datetime.now().isoformat(),
                layer2_voting_pred=float(layer2_voting[i]),
                layer2_stacking_pred=float(layer2_stacking[i]),
                layer2_model_contributions={
                    m.model_id[:12]: float(m.metrics.cv_auc)
                    for m in (self.registry_ensemble.selected_models[:5] if self.registry_ensemble else [])
                },
                layer3_predictions=l3_preds,
                layer3_avg_pred=float(l3_avg),
                layer3_std_pred=float(l3_std),
                layer4_temporal_predictions={},
                layer4_agreement_score=0.0,
                cross_layer_disagreement=float(cross_disagree),
                layer_agreement_vector=[float(p) for p in layer_preds],
                final_prediction=float(final_predictions[i]),
                final_confidence=float(abs(final_predictions[i] - 0.5) * 2),
            )
            lineages.append(lineage)

        return lineages

    def _print_fit_summary(self, metrics: Dict[str, Any]) -> None:
        """Print a summary of the fitting process."""
        logger.info("")
        logger.info("=" * 70)
        logger.info("MEGA ENSEMBLE - Training Summary")
        logger.info("=" * 70)
        logger.info("")

        # Layer 2
        l2 = metrics.get("layer2_selection", {})
        logger.info(f"Layer 2 (Registry Ensemble):")
        logger.info(f"  Models selected: {l2.get('n_selected', 0)}")
        logger.info(f"  Avg AUC: {l2.get('avg_auc', 0):.4f}")
        logger.info(f"  Avg disagreement: {l2.get('avg_pairwise_disagreement', 0):.4f}")

        l2e = metrics.get("layer2_ensemble", {})
        if "val_voting_auc" in l2e:
            logger.info(f"  Voting AUC: {l2e['val_voting_auc']:.4f}")
        if "val_stacking_auc" in l2e:
            logger.info(f"  Stacking AUC: {l2e['val_stacking_auc']:.4f}")

        # Layer 3
        l3 = metrics.get("layer3_fabric", {})
        if l3.get("status") != "skipped":
            logger.info(f"\nLayer 3 (Interpolated Fabric):")
            logger.info(f"  Fabric models: {l3.get('n_trained', 0)}")
        else:
            logger.info(f"\nLayer 3 (Interpolated Fabric): Skipped")

        # Layer 4
        l4 = metrics.get("layer4_cascade", {})
        status = l4.get("status", "unknown")
        logger.info(f"\nLayer 4 (Temporal Cascade): {status}")

        # Layer 5
        l5 = metrics.get("layer5_meta", {})
        logger.info(f"\nLayer 5 (Meta-Learner):")
        logger.info(f"  Model: {l5.get('meta_model', 'unknown')}")
        logger.info(f"  Features: {l5.get('n_features', 0)}")
        logger.info(f"  Validation AUC: {l5.get('val_auc', 0):.4f}")

        logger.info("")
        logger.info("=" * 70)

    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get summary information about the ensemble."""
        info = {
            "is_fitted": self.is_fitted,
            "config": {
                "diversity_n_models": self.config.diversity_n_models,
                "fabric_max_models": self.config.fabric_max_models,
                "final_meta_model": self.config.final_meta_model,
            },
        }

        if self.registry_ensemble:
            info["layer2"] = self.registry_ensemble.get_ensemble_info()

        info["layer3"] = {
            "n_fabric_models": len(self.fabric_models),
        }

        info["layer4"] = {
            "temporal_cascade_available": self.temporal_cascade is not None,
        }

        if self.fit_metrics:
            info["fit_metrics"] = self.fit_metrics

        return info


if __name__ == "__main__":
    # Test basic functionality
    config = MegaEnsembleConfig(
        diversity_n_models=5,
        fabric_max_models=20,
        final_meta_model="logistic",
        use_temporal_cascade=False,
    )
    print(f"Mega ensemble config: {config}")

    mega = MegaEnsemble(config)
    print("Mega ensemble initialized")
