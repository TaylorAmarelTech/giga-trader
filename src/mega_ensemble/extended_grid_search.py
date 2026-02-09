"""
Extended Grid Search Generator for Mega Ensemble.

Generates configurations covering:
- 11 model types (logistic, trees, boosting, neural)
- 6 dimensionality reduction methods
- 3 scaling methods
- Full hyperparameter ranges
"""

import sys
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model_registry_v2 import (
    GridSearchConfigGenerator,
    ModelEntry,
    ModelType,
    DimReductionMethod,
    ScalingMethod,
    CascadeType,
)


@dataclass
class ExtendedGridConfig:
    """Configuration for extended grid search."""

    # Model type families to include
    include_logistic: bool = True       # L1, L2, ElasticNet
    include_trees: bool = True          # RF, ExtraTrees, DecisionTree
    include_boosting: bool = True       # GB, XGB, LGB, HistGB
    include_svm: bool = False           # SVM variants (slow, disabled by default)
    include_neural: bool = False        # MLP variants (can be slow)

    # Hyperparameter ranges
    logistic_C_range: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0, 10.0])
    tree_depth_range: List[int] = field(default_factory=lambda: [2, 3, 4, 5])  # EDGE 1: max 5
    boosting_n_estimators: List[int] = field(default_factory=lambda: [50, 100, 150])
    boosting_learning_rate: List[float] = field(default_factory=lambda: [0.01, 0.05, 0.1])

    # Dim reduction methods to try
    dim_reduction_methods: List[str] = field(default_factory=lambda: [
        "none", "pca", "kernel_pca_rbf", "ica", "umap", "ensemble_plus"
    ])
    n_components_range: List[int] = field(default_factory=lambda: [15, 20, 30])

    # Scaling methods
    scaling_methods: List[str] = field(default_factory=lambda: [
        "standard", "robust", "quantile_normal"
    ])

    # Cascade types to include
    cascade_types: List[str] = field(default_factory=lambda: [
        "base", "attention", "masked", "multi_resolution"
    ])

    # Grid limits
    max_configs: int = 500
    random_sample: bool = True


class ExtendedGridSearchGenerator:
    """
    Generates extended grid search configurations with more model types.

    Extends GridSearchConfigGenerator from model_registry_v2.py with:
    - More model types per family
    - Full hyperparameter ranges
    - Diversity tracking
    """

    def __init__(self, config: ExtendedGridConfig = None):
        self.config = config or ExtendedGridConfig()

    def _get_model_types(self) -> List[str]:
        """Get list of model types based on config."""
        model_types = []

        if self.config.include_logistic:
            model_types.extend([
                ModelType.LOGISTIC_L1.value,
                ModelType.LOGISTIC_L2.value,
                ModelType.ELASTIC_NET.value,
            ])

        if self.config.include_trees:
            model_types.extend([
                ModelType.RANDOM_FOREST.value,
                ModelType.EXTRA_TREES.value,
                ModelType.DECISION_TREE.value,
            ])

        if self.config.include_boosting:
            model_types.extend([
                ModelType.GRADIENT_BOOSTING.value,
                ModelType.HIST_GRADIENT_BOOSTING.value,
                ModelType.XGBOOST.value,
                ModelType.LIGHTGBM.value,
            ])

        if self.config.include_svm:
            model_types.extend([
                ModelType.SVM_LINEAR.value,
                ModelType.SVM_RBF.value,
            ])

        if self.config.include_neural:
            model_types.extend([
                ModelType.MLP.value,
            ])

        return model_types

    def create_extended_grid(self) -> GridSearchConfigGenerator:
        """
        Create extended grid with all model families.

        Returns GridSearchConfigGenerator ready to generate configs.
        """
        gen = GridSearchConfigGenerator()

        # Model types
        model_types = self._get_model_types()
        gen.add_dimension('model_config.model_type', model_types)

        # Logistic regularization
        gen.add_dimension('model_config.lr_C', self.config.logistic_C_range)

        # Tree/boosting hyperparameters
        gen.add_dimension('model_config.gb_max_depth', self.config.tree_depth_range)
        gen.add_dimension('model_config.rf_max_depth', self.config.tree_depth_range)
        gen.add_dimension('model_config.gb_n_estimators', self.config.boosting_n_estimators)
        gen.add_dimension('model_config.rf_n_estimators', self.config.boosting_n_estimators)
        gen.add_dimension('model_config.gb_learning_rate', self.config.boosting_learning_rate)

        # Dim reduction
        gen.add_dimension('dim_reduction_config.method', self.config.dim_reduction_methods)
        gen.add_dimension('dim_reduction_config.n_components', self.config.n_components_range)

        # Scaling
        gen.add_dimension('preprocess_config.scaling_method', self.config.scaling_methods)

        # Cascade types
        gen.add_dimension('cascade_config.cascade_type', self.config.cascade_types)

        # Add constraints to avoid invalid combinations
        # Linear models don't use tree hyperparameters
        for linear_model in ['logistic_l1', 'logistic_l2', 'elastic_net', 'svm_linear', 'svm_rbf', 'mlp']:
            gen.add_constraint(
                'model_config.model_type', linear_model,
                'model_config.gb_max_depth', self.config.tree_depth_range[:1]  # Use first only
            )
            gen.add_constraint(
                'model_config.model_type', linear_model,
                'model_config.gb_n_estimators', self.config.boosting_n_estimators[:1]
            )

        return gen

    def estimate_grid_size(self) -> Dict[str, Any]:
        """Estimate total configurations and resources."""
        gen = self.create_extended_grid()
        n_combos = gen.count_combinations()

        return {
            "total_combinations": n_combos,
            "after_max_configs": min(n_combos, self.config.max_configs),
            "estimated_time_minutes": min(n_combos, self.config.max_configs) * 3,
            "model_types": len(self._get_model_types()),
            "dim_methods": len(self.config.dim_reduction_methods),
            "scaling_methods": len(self.config.scaling_methods),
        }

    def generate_configs(
        self,
        target_type: str = "swing",
        experiment_id: str = None,
    ) -> List[ModelEntry]:
        """
        Generate model entry configurations for grid search.

        Args:
            target_type: Target variable type (swing, timing)
            experiment_id: Optional experiment ID for grouping

        Returns:
            List of ModelEntry objects ready for training
        """
        gen = self.create_extended_grid()

        configs = gen.generate_configs(
            target_type=target_type,
            experiment_id=experiment_id,
            max_configs=self.config.max_configs,
            random_sample=self.config.random_sample,
        )

        return configs

    def get_summary(self) -> str:
        """Get human-readable summary of grid configuration."""
        est = self.estimate_grid_size()

        lines = [
            "=" * 60,
            "EXTENDED GRID SEARCH CONFIGURATION",
            "=" * 60,
            "",
            f"Model Types ({est['model_types']}):",
        ]

        if self.config.include_logistic:
            lines.append("  - Logistic: L1, L2, ElasticNet")
        if self.config.include_trees:
            lines.append("  - Trees: RandomForest, ExtraTrees, DecisionTree")
        if self.config.include_boosting:
            lines.append("  - Boosting: GB, HistGB, XGBoost, LightGBM")
        if self.config.include_svm:
            lines.append("  - SVM: Linear, RBF")
        if self.config.include_neural:
            lines.append("  - Neural: MLP")

        lines.extend([
            "",
            f"Dim Reduction Methods ({est['dim_methods']}): {self.config.dim_reduction_methods}",
            f"Scaling Methods ({est['scaling_methods']}): {self.config.scaling_methods}",
            "",
            "Hyperparameter Ranges:",
            f"  - lr_C: {self.config.logistic_C_range}",
            f"  - tree_depth: {self.config.tree_depth_range}",
            f"  - n_estimators: {self.config.boosting_n_estimators}",
            f"  - learning_rate: {self.config.boosting_learning_rate}",
            f"  - n_components: {self.config.n_components_range}",
            "",
            f"Total Combinations: {est['total_combinations']:,}",
            f"Max Configs to Train: {est['after_max_configs']:,}",
            f"Estimated Time: {est['estimated_time_minutes']} minutes",
            "=" * 60,
        ])

        return "\n".join(lines)


# Convenience functions
def create_quick_extended_grid(n_configs: int = 50) -> List[ModelEntry]:
    """Create a quick extended grid for testing."""
    config = ExtendedGridConfig(
        include_svm=False,
        include_neural=False,
        logistic_C_range=[0.1, 1.0, 10.0],
        tree_depth_range=[3, 4, 5],
        boosting_n_estimators=[50, 100],
        boosting_learning_rate=[0.05, 0.1],
        dim_reduction_methods=["none", "pca", "ica"],
        n_components_range=[20, 30],
        scaling_methods=["standard", "robust"],
        cascade_types=["base"],
        max_configs=n_configs,
        random_sample=True,
    )

    gen = ExtendedGridSearchGenerator(config)
    return gen.generate_configs(target_type="swing")


def create_full_extended_grid(n_configs: int = 500) -> List[ModelEntry]:
    """Create a full extended grid for production."""
    config = ExtendedGridConfig(
        include_logistic=True,
        include_trees=True,
        include_boosting=True,
        include_svm=False,  # Still skip SVM for speed
        include_neural=True,
        max_configs=n_configs,
        random_sample=True,
    )

    gen = ExtendedGridSearchGenerator(config)
    return gen.generate_configs(target_type="swing")


if __name__ == "__main__":
    # Test the generator
    gen = ExtendedGridSearchGenerator()
    print(gen.get_summary())

    # Generate a few configs
    configs = gen.generate_configs(target_type="swing")[:5]
    print(f"\nGenerated {len(configs)} sample configs:")
    for c in configs:
        print(f"  - {c.model_config.model_type} / {c.dim_reduction_config.method}")
