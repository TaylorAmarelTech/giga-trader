"""
Config Interpolator for Mega Ensemble.

Creates interpolated configurations between successful models:
- Geometric interpolation for continuous params (C, learning_rate)
- Linear interpolation for discrete params (n_estimators, max_depth)
- Hybrid dim reduction combinations

This creates a "fabric" of model configurations in hyperparameter space.
"""

import sys
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
from pathlib import Path
from itertools import combinations
from copy import deepcopy
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model_registry_v2 import ModelEntry, ModelConfig, DimReductionConfig

logger = logging.getLogger(__name__)


@dataclass
class InterpolationConfig:
    """Configuration for config interpolation."""

    # Interpolation settings
    n_interpolation_points: int = 3      # Number of points between two configs
    interpolation_method: str = "mixed"   # "linear", "geometric", "mixed"

    # Which parameters to interpolate
    interpolate_continuous: bool = True   # C, learning_rate, etc.
    interpolate_discrete: bool = True     # n_estimators, max_depth
    create_hybrid_dim_reduction: bool = True  # Combine dim reduction methods

    # Mesh density limits
    max_fabric_models: int = 50           # Max total interpolated models
    max_pairs_to_consider: int = 45       # Max pairs (C(10,2)=45)


class ConfigInterpolator:
    """
    Creates interpolated configurations between successful models.

    Key innovation: Instead of just grid search, explore the "space between"
    known good configurations, creating a dense "fabric" of models.

    Example:
        Model A: lr_C=0.1, max_depth=2, n_estimators=50
        Model B: lr_C=10.0, max_depth=5, n_estimators=200

        Interpolated configs:
        - lr_C=1.0 (geometric mean), max_depth=3, n_estimators=100
        - lr_C=1.0, max_depth=4, n_estimators=125
        - lr_C=3.16 (geometric mean), max_depth=3
        etc.
    """

    # Continuous parameters that can be interpolated (use geometric/log)
    CONTINUOUS_PARAMS = {
        'model_config.lr_C': {'method': 'log', 'min': 0.001, 'max': 100.0},
        'model_config.gb_learning_rate': {'method': 'log', 'min': 0.001, 'max': 0.5},
        'model_config.xgb_learning_rate': {'method': 'log', 'min': 0.001, 'max': 0.5},
        'model_config.lgb_learning_rate': {'method': 'log', 'min': 0.001, 'max': 0.5},
        'model_config.lr_l1_ratio': {'method': 'linear', 'min': 0.0, 'max': 1.0},
        'model_config.en_l1_ratio': {'method': 'linear', 'min': 0.0, 'max': 1.0},
        'model_config.gb_subsample': {'method': 'linear', 'min': 0.5, 'max': 1.0},
        'dim_reduction_config.n_components': {'method': 'linear', 'min': 5, 'max': 100},
        'dim_reduction_config.kpca_gamma': {'method': 'log', 'min': 0.001, 'max': 1.0},
        'dim_reduction_config.umap_n_neighbors': {'method': 'linear', 'min': 5, 'max': 50},
    }

    # Discrete parameters that can be interpolated (use linear with rounding)
    DISCRETE_PARAMS = {
        'model_config.gb_n_estimators': {'min': 10, 'max': 500},
        'model_config.rf_n_estimators': {'min': 10, 'max': 500},
        'model_config.xgb_n_estimators': {'min': 10, 'max': 500},
        'model_config.lgb_n_estimators': {'min': 10, 'max': 500},
        'model_config.gb_max_depth': {'min': 2, 'max': 5},  # EDGE 1: max 5
        'model_config.rf_max_depth': {'min': 2, 'max': 5},
        'model_config.xgb_max_depth': {'min': 2, 'max': 5},
        'model_config.gb_min_samples_leaf': {'min': 5, 'max': 100},
        'model_config.rf_min_samples_leaf': {'min': 5, 'max': 100},
    }

    def __init__(self, config: InterpolationConfig = None):
        self.config = config or InterpolationConfig()

    def _get_nested_attr(self, obj: Any, path: str) -> Any:
        """Get nested attribute from object using dot notation."""
        parts = path.split('.')
        for part in parts:
            if obj is None:
                return None
            obj = getattr(obj, part, None)
        return obj

    def _set_nested_attr(self, obj: Any, path: str, value: Any) -> None:
        """Set nested attribute on object using dot notation."""
        parts = path.split('.')
        for part in parts[:-1]:
            obj = getattr(obj, part)
        setattr(obj, parts[-1], value)

    def extract_config_values(self, entry: ModelEntry) -> Dict[str, Any]:
        """Extract interpolatable parameter values from a ModelEntry."""
        values = {}

        all_params = list(self.CONTINUOUS_PARAMS.keys()) + list(self.DISCRETE_PARAMS.keys())

        for param in all_params:
            value = self._get_nested_attr(entry, param)
            if value is not None:
                values[param] = value

        return values

    def interpolate_value(
        self,
        v1: float,
        v2: float,
        param_name: str,
        n_points: int = 3,
    ) -> List[float]:
        """
        Interpolate between two values.

        For continuous params: geometric interpolation (better for learning rates, C)
        For linear-type params: linear interpolation

        Returns list of intermediate values (excluding endpoints).
        """
        if v1 == v2:
            return []

        # Determine interpolation method
        if param_name in self.CONTINUOUS_PARAMS:
            method = self.CONTINUOUS_PARAMS[param_name].get('method', 'linear')
        else:
            method = 'linear'

        if method == 'log':
            # Geometric interpolation
            log_v1 = np.log10(max(v1, 1e-10))
            log_v2 = np.log10(max(v2, 1e-10))
            log_interp = np.linspace(log_v1, log_v2, n_points + 2)[1:-1]
            return [10 ** x for x in log_interp]
        else:
            # Linear interpolation
            return list(np.linspace(v1, v2, n_points + 2)[1:-1])

    def interpolate_discrete(
        self,
        v1: int,
        v2: int,
        n_points: int = 3,
    ) -> List[int]:
        """Interpolate discrete values with rounding."""
        if v1 == v2:
            return []

        interp = np.linspace(v1, v2, n_points + 2)[1:-1]
        # Round and remove duplicates while preserving order
        rounded = [int(round(x)) for x in interp]
        seen = set()
        unique = []
        for x in rounded:
            if x not in seen and x != v1 and x != v2:
                seen.add(x)
                unique.append(x)
        return unique

    def _clone_entry(self, entry: ModelEntry) -> ModelEntry:
        """Clone a ModelEntry (deep copy without metrics/artifacts)."""
        # Use deep copy to avoid reference issues
        new_entry = deepcopy(entry)

        # Reset status and metrics
        new_entry.model_id = ""  # Will be regenerated
        new_entry.status = "queued"
        new_entry.metrics = None
        new_entry.artifacts = None
        new_entry.training_time_seconds = None
        new_entry.training_started_at = None
        new_entry.trained_at = None

        # Clear tags and add interpolated marker
        if new_entry.tags is None:
            new_entry.tags = []
        else:
            new_entry.tags = list(new_entry.tags)

        return new_entry

    def create_interpolated_configs(
        self,
        models: List[ModelEntry],
    ) -> Tuple[List[ModelEntry], Dict[str, Any]]:
        """
        Create interpolated configurations between pairs of best models.

        Algorithm:
        1. For each pair of models
        2. For each interpolatable parameter where values differ
        3. Generate interpolation points
        4. Create new ModelEntry with interpolated values

        Returns:
            Tuple of (interpolated_configs, generation_metrics)
        """
        logger.info("=" * 60)
        logger.info("CONFIG INTERPOLATOR - Generating fabric")
        logger.info("=" * 60)

        interpolated_configs = []
        params_interpolated = {}

        # Get all pairs (limited by config)
        n_models = len(models)
        all_pairs = list(combinations(range(n_models), 2))

        if len(all_pairs) > self.config.max_pairs_to_consider:
            # Prioritize pairs with higher average AUC
            pair_scores = []
            for i, j in all_pairs:
                avg_auc = (models[i].metrics.cv_auc + models[j].metrics.cv_auc) / 2
                pair_scores.append((avg_auc, (i, j)))
            pair_scores.sort(reverse=True)
            pairs = [p[1] for p in pair_scores[:self.config.max_pairs_to_consider]]
        else:
            pairs = all_pairs

        logger.info(f"Considering {len(pairs)} model pairs for interpolation")

        for pair_idx, (i, j) in enumerate(pairs):
            if len(interpolated_configs) >= self.config.max_fabric_models:
                break

            model_a = models[i]
            model_b = models[j]

            values_a = self.extract_config_values(model_a)
            values_b = self.extract_config_values(model_b)

            # Find parameters that differ
            differing_params = []
            for param in values_a:
                if param in values_b and values_a[param] != values_b[param]:
                    differing_params.append(param)

            if not differing_params:
                continue

            # Generate interpolations for each differing parameter
            for param in differing_params:
                if len(interpolated_configs) >= self.config.max_fabric_models:
                    break

                v1 = values_a[param]
                v2 = values_b[param]

                # Skip if values are None or invalid
                if v1 is None or v2 is None:
                    continue

                # Interpolate based on parameter type
                if param in self.DISCRETE_PARAMS:
                    if not self.config.interpolate_discrete:
                        continue
                    try:
                        interp_values = self.interpolate_discrete(
                            int(v1), int(v2), self.config.n_interpolation_points
                        )
                    except (ValueError, TypeError):
                        continue
                else:
                    if not self.config.interpolate_continuous:
                        continue
                    try:
                        interp_values = self.interpolate_value(
                            float(v1), float(v2), param, self.config.n_interpolation_points
                        )
                    except (ValueError, TypeError):
                        continue

                if not interp_values:
                    continue

                # Track which params we interpolated
                if param not in params_interpolated:
                    params_interpolated[param] = 0

                # Create configs with interpolated values
                for value in interp_values:
                    if len(interpolated_configs) >= self.config.max_fabric_models:
                        break

                    # Clone model_a config
                    new_entry = self._clone_entry(model_a)
                    new_entry.parent_model_id = model_a.model_id
                    new_entry.tags.append("interpolated")
                    new_entry.tags.append(f"interp_{param.split('.')[-1]}")

                    # Set interpolated value
                    try:
                        if param in self.DISCRETE_PARAMS:
                            self._set_nested_attr(new_entry, param, int(value))
                        else:
                            self._set_nested_attr(new_entry, param, float(value))

                        interpolated_configs.append(new_entry)
                        params_interpolated[param] += 1

                    except Exception as e:
                        logger.warning(f"Failed to set {param}={value}: {e}")
                        continue

        metrics = {
            "n_pairs_considered": len(pairs),
            "n_interpolated_configs": len(interpolated_configs),
            "params_interpolated": params_interpolated,
        }

        logger.info(f"Generated {len(interpolated_configs)} interpolated configs")
        for param, count in params_interpolated.items():
            logger.info(f"  - {param}: {count} configs")

        return interpolated_configs, metrics

    def create_hybrid_dim_reduction_configs(
        self,
        models: List[ModelEntry],
    ) -> Tuple[List[ModelEntry], Dict[str, Any]]:
        """
        Create hybrid configs that combine dim reduction methods.

        E.g., if model A uses PCA and model B uses ICA,
        create a config using ensemble_plus with averaged n_components.
        """
        if not self.config.create_hybrid_dim_reduction:
            return [], {"status": "disabled"}

        logger.info("Creating hybrid dim reduction configs...")

        # Group models by dim reduction method
        by_method = {}
        for m in models:
            method = m.dim_reduction_config.method if m.dim_reduction_config else "none"
            if method not in by_method:
                by_method[method] = []
            by_method[method].append(m)

        hybrid_configs = []
        methods = list(by_method.keys())

        for i, j in combinations(range(len(methods)), 2):
            method_a = methods[i]
            method_b = methods[j]

            # Skip if either is 'none' or they're the same
            if 'none' in [method_a, method_b]:
                continue

            # Get representative models
            model_a = by_method[method_a][0]
            model_b = by_method[method_b][0]

            # Calculate average n_components
            n_comp_a = model_a.dim_reduction_config.n_components if model_a.dim_reduction_config else 20
            n_comp_b = model_b.dim_reduction_config.n_components if model_b.dim_reduction_config else 20
            avg_n_components = (n_comp_a + n_comp_b) // 2

            # Create hybrid using ensemble_plus
            new_entry = self._clone_entry(model_a)
            new_entry.dim_reduction_config.method = "ensemble_plus"
            new_entry.dim_reduction_config.n_components = avg_n_components
            new_entry.tags.append("hybrid_dim_reduction")
            new_entry.tags.append(f"hybrid_{method_a}_{method_b}")
            new_entry.notes = f"Hybrid: {method_a} + {method_b}"

            hybrid_configs.append(new_entry)

            if len(hybrid_configs) >= 10:  # Limit hybrids
                break

        metrics = {
            "n_methods_found": len(methods),
            "n_hybrid_configs": len(hybrid_configs),
            "method_combinations": [f"{methods[i]}+{methods[j]}"
                                    for i, j in combinations(range(min(len(methods), 5)), 2)],
        }

        logger.info(f"Generated {len(hybrid_configs)} hybrid dim reduction configs")

        return hybrid_configs, metrics


class FabricOfPoints:
    """
    Creates a dense "fabric" of model configurations in hyperparameter space.

    Combines:
    1. Original best models (anchor points)
    2. Interpolated configs (edges between anchors)
    3. Hybrid configs (dim reduction combinations)
    """

    def __init__(
        self,
        best_models: List[ModelEntry],
        interpolation_config: InterpolationConfig = None,
    ):
        self.best_models = best_models
        self.interpolator = ConfigInterpolator(interpolation_config)

    def generate_fabric(self) -> Tuple[List[ModelEntry], Dict[str, Any]]:
        """
        Generate the full fabric of configurations.

        Returns:
            Tuple of (fabric_configs, generation_metrics)
        """
        logger.info("=" * 60)
        logger.info("FABRIC OF POINTS - Generating configuration mesh")
        logger.info("=" * 60)

        fabric = []  # Don't include original anchors - they're already trained

        # Add interpolated configs
        interpolated, interp_metrics = self.interpolator.create_interpolated_configs(
            self.best_models
        )
        fabric.extend(interpolated)

        # Add hybrid dim reduction configs
        hybrids, hybrid_metrics = self.interpolator.create_hybrid_dim_reduction_configs(
            self.best_models
        )
        fabric.extend(hybrids)

        metrics = {
            "n_anchors": len(self.best_models),
            "n_interpolated": len(interpolated),
            "n_hybrids": len(hybrids),
            "total_fabric_size": len(fabric),
            "interpolation_details": interp_metrics,
            "hybrid_details": hybrid_metrics,
        }

        logger.info("")
        logger.info(f"Fabric generation complete:")
        logger.info(f"  Anchor models: {len(self.best_models)}")
        logger.info(f"  Interpolated configs: {len(interpolated)}")
        logger.info(f"  Hybrid configs: {len(hybrids)}")
        logger.info(f"  Total fabric size: {len(fabric)}")
        logger.info("=" * 60)

        return fabric, metrics


if __name__ == "__main__":
    # Test basic functionality
    config = InterpolationConfig(
        n_interpolation_points=3,
        max_fabric_models=50,
        interpolate_continuous=True,
        interpolate_discrete=True,
    )
    interpolator = ConfigInterpolator(config)

    # Test interpolation
    print("Testing continuous interpolation (log scale):")
    vals = interpolator.interpolate_value(0.01, 10.0, 'model_config.lr_C', 3)
    print(f"  0.01 -> 10.0: {[f'{v:.4f}' for v in vals]}")

    print("\nTesting discrete interpolation:")
    vals = interpolator.interpolate_discrete(50, 200, 3)
    print(f"  50 -> 200: {vals}")

    print("\nConfig interpolator initialized successfully")
