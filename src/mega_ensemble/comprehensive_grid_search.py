"""
Comprehensive Grid Search Generator for Mega Ensemble.

This module creates grid searches that cover ALL 13 pipeline steps:
1. DataConfig - source, period, resolution, market hours
2. SyntheticDataConfig - 21 synthetic data methods
3. AugmentationConfig - data augmentation methods
4. PreprocessConfig - outlier methods, missing value methods, scaling, transforms
5. FeatureConfig - feature groups, rolling windows, lags
6. TargetConfig - target definition, soft targets, smoothing
7. FeatureSelectionConfig - 17 feature selection methods
8. DimReductionConfig - 23 dim reduction methods with hyperparameters
9. ModelConfig - 33 model types with 170+ hyperparameters
10. CascadeConfig - 12 cascade types, temporal settings, attention
11. SampleWeightConfig - time decay, return magnitude, class balancing
12. TrainingConfig - CV methods, Optuna, robustness testing
13. EvaluationConfig - calibration, threshold optimization

Total possible combinations: 10^15+ (quintillions)
"""

import sys
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from itertools import product
import random

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.model_registry_v2 import (
    GridSearchConfigGenerator,
    ModelEntry,
    # Data enums
    DataSource, DataPeriod, TimeResolution, MarketHours,
    # Synthetic & Augmentation
    SyntheticDataMethod, AugmentationMethod,
    # Preprocessing
    OutlierMethod, MissingValueMethod, ScalingMethod, TransformMethod,
    # Feature selection & dim reduction
    FeatureSelectionMethod, DimReductionMethod,
    # Models
    ModelType, EnsembleMethod,
    # Cascade
    CascadeType, TemporalEncoding,
    # Target
    TargetType, TargetDefinition, LabelSmoothingMethod,
    # Training
    CVMethod, ScoringMetric, SampleWeightMethod,
    # Status
    ModelStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION OPTIONS FOR EACH PIPELINE STEP
# =============================================================================

# Step 1: Data Configuration
DATA_OPTIONS = {
    "source": [DataSource.ALPACA.value],
    "period": [DataPeriod.YEARS_3.value, DataPeriod.YEARS_5.value],
    "market_hours": [MarketHours.ALL_HOURS.value, MarketHours.REGULAR_ONLY.value],
}

# Step 2: Synthetic Data Configuration
SYNTHETIC_OPTIONS = {
    "enabled": [False, True],
    "methods": [
        [SyntheticDataMethod.NONE.value],
        [SyntheticDataMethod.FILTER_EXTREMES_10.value],
        [SyntheticDataMethod.BOOTSTRAP_70.value],
        [SyntheticDataMethod.FILTER_EXTREMES_10.value, SyntheticDataMethod.BOOTSTRAP_70.value],
    ],
    "synthetic_weight": [0.2, 0.3, 0.4],
}

# Step 3: Augmentation Configuration
AUGMENTATION_OPTIONS = {
    "enabled": [False, True],
    "methods": [
        [],
        [AugmentationMethod.JITTER.value],
        [AugmentationMethod.SMOTE.value],
        [AugmentationMethod.JITTER.value, AugmentationMethod.NOISE_INJECTION.value],
    ],
}

# Step 4: Preprocessing Configuration
PREPROCESS_OPTIONS = {
    "outlier_method": [
        OutlierMethod.NONE.value,
        OutlierMethod.WINSORIZE_1.value,
        OutlierMethod.WINSORIZE_2_5.value,
        OutlierMethod.CLIP_3STD.value,
        OutlierMethod.IQR_1_5.value,
        OutlierMethod.ISOLATION_FOREST.value,
    ],
    "missing_method": [
        MissingValueMethod.FORWARD_FILL.value,
        MissingValueMethod.MEAN_FILL.value,
        MissingValueMethod.KNN_IMPUTE.value,
    ],
    "scaling_method": [
        ScalingMethod.NONE.value,
        ScalingMethod.STANDARD.value,
        ScalingMethod.ROBUST.value,
        ScalingMethod.QUANTILE_NORMAL.value,
        ScalingMethod.MINMAX.value,
    ],
    "transform_method": [
        TransformMethod.NONE.value,
        TransformMethod.LOG1P.value,
        TransformMethod.RANK.value,
    ],
}

# Step 5: Feature Configuration (groups to include)
FEATURE_OPTIONS = {
    "include_price_features": [True],
    "include_volume_features": [True, False],
    "include_momentum_features": [True],
    "include_volatility_features": [True],
    "include_extended_hours": [True, False],
    "include_cross_asset": [False, True],
    "include_breadth": [False, True],
    "rolling_windows": [[5, 10, 20], [5, 10, 20, 50]],
    "lag_periods": [[1, 2, 3, 5], [1, 2, 3, 5, 10]],
}

# Step 6: Target Configuration
TARGET_OPTIONS = {
    "target_type": [TargetType.SWING.value],
    "target_definition": [
        TargetDefinition.RETURN_THRESHOLD.value,
        TargetDefinition.CLOSE_VS_OPEN.value,
    ],
    "swing_threshold": [0.002, 0.0025, 0.003],
    "use_soft_targets": [True, False],
    "smoothing_method": [
        LabelSmoothingMethod.NONE.value,
        LabelSmoothingMethod.EPSILON_SMOOTHING.value,
        LabelSmoothingMethod.SIGMOID_TRANSFORM.value,
    ],
    "smoothing_epsilon": [0.05, 0.1],
}

# Step 7: Feature Selection Configuration
FEATURE_SELECTION_OPTIONS = {
    "method": [
        FeatureSelectionMethod.NONE.value,
        FeatureSelectionMethod.MUTUAL_INFO.value,
        FeatureSelectionMethod.ANOVA_F.value,
        FeatureSelectionMethod.VARIANCE_THRESHOLD.value,
        FeatureSelectionMethod.CORRELATION_FILTER.value,
        FeatureSelectionMethod.RFE.value,
        FeatureSelectionMethod.TREE_IMPORTANCE.value,
        FeatureSelectionMethod.BORUTA.value,
        FeatureSelectionMethod.SHAP_IMPORTANCE.value,
    ],
    "n_features": [20, 30, 40, 50],
    "variance_threshold": [0.01, 0.05],
    "correlation_threshold": [0.90, 0.95],
}

# Step 8: Dimensionality Reduction Configuration
DIM_REDUCTION_OPTIONS = {
    "method": [
        DimReductionMethod.NONE.value,
        DimReductionMethod.PCA.value,
        DimReductionMethod.KERNEL_PCA_RBF.value,
        DimReductionMethod.ICA.value,
        DimReductionMethod.UMAP.value,
        DimReductionMethod.TRUNCATED_SVD.value,
        DimReductionMethod.FACTOR_ANALYSIS.value,
        DimReductionMethod.AGGLOMERATION.value,
        DimReductionMethod.ENSEMBLE_PLUS.value,
    ],
    "n_components": [15, 20, 25, 30, 40],
    "kpca_gamma": [0.01, 0.05, 0.1],
    "kpca_kernel": ["rbf", "poly", "sigmoid"],
    "umap_n_neighbors": [10, 15, 20],
    "umap_min_dist": [0.0, 0.1, 0.25],
}

# Step 9: Model Configuration (COMPREHENSIVE)
MODEL_OPTIONS = {
    "model_type": [
        # Linear models
        ModelType.LOGISTIC_L1.value,
        ModelType.LOGISTIC_L2.value,
        ModelType.ELASTIC_NET.value,
        ModelType.RIDGE.value,
        ModelType.SGD_CLASSIFIER.value,
        # Tree models
        ModelType.DECISION_TREE.value,
        ModelType.RANDOM_FOREST.value,
        ModelType.EXTRA_TREES.value,
        # Boosting models
        ModelType.GRADIENT_BOOSTING.value,
        ModelType.HIST_GRADIENT_BOOSTING.value,
        ModelType.XGBOOST.value,
        ModelType.LIGHTGBM.value,
        # SVM
        ModelType.SVM_LINEAR.value,
        ModelType.SVM_RBF.value,
        # Neural networks
        ModelType.MLP.value,
        ModelType.MLP_DEEP.value,
        # Naive Bayes
        ModelType.GAUSSIAN_NB.value,
        # Neighbors
        ModelType.KNN.value,
        # Ensemble
        ModelType.ADABOOST.value,
        ModelType.BAGGING.value,
    ],
    # Logistic regression hyperparameters
    "lr_C": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
    "lr_l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9],  # For elastic net
    # Tree hyperparameters (EDGE 1: max_depth <= 5)
    "tree_max_depth": [2, 3, 4, 5],
    "tree_n_estimators": [50, 100, 150, 200],
    "tree_min_samples_leaf": [10, 20, 50],
    # Boosting hyperparameters
    "gb_learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "gb_subsample": [0.7, 0.8, 0.9, 1.0],
    # SVM hyperparameters
    "svm_C": [0.1, 1.0, 10.0],
    "svm_gamma": ["scale", "auto", 0.01, 0.1],
    # KNN hyperparameters
    "knn_n_neighbors": [3, 5, 7, 11],
    "knn_weights": ["uniform", "distance"],
    # MLP hyperparameters
    "mlp_hidden_layers": [[50], [100], [100, 50], [100, 50, 25]],
    "mlp_alpha": [0.0001, 0.001, 0.01],
}

# Step 10: Cascade Configuration
CASCADE_OPTIONS = {
    "cascade_type": [
        CascadeType.BASE.value,
        CascadeType.MASKED.value,
        CascadeType.ATTENTION.value,
        CascadeType.STOCHASTIC_DEPTH.value,
        CascadeType.MULTI_RESOLUTION.value,
        CascadeType.MIXTURE_OF_EXPERTS.value,
    ],
    "temporal_slices": [
        [0, 30, 60, 90, 120, 180],  # Standard
        [0, 15, 30, 60, 120, 240],  # Fine-grained early
        [0, 60, 120, 180, 240, 300, 390],  # Full day
    ],
    "mask_probability": [0.1, 0.2, 0.3],
    "attention_heads": [1, 2, 4],
    "temporal_encoding": [
        TemporalEncoding.NONE.value,
        TemporalEncoding.SINUSOIDAL.value,
        TemporalEncoding.LEARNED.value,
    ],
}

# Step 11: Sample Weight Configuration
SAMPLE_WEIGHT_OPTIONS = {
    "method": [
        SampleWeightMethod.NONE.value,
        SampleWeightMethod.BALANCED.value,
        SampleWeightMethod.TIME_DECAY_EXPONENTIAL.value,
        SampleWeightMethod.RETURN_MAGNITUDE.value,
        SampleWeightMethod.LOSS_EMPHASIS.value,
    ],
    "decay_rate": [0.99, 0.995, 0.999],
    "loss_emphasis_factor": [1.5, 2.0, 3.0],
}

# Step 12: Training Configuration
TRAINING_OPTIONS = {
    "cv_method": [
        CVMethod.PURGED_KFOLD.value,
        CVMethod.WALK_FORWARD.value,
        CVMethod.EXPANDING_WINDOW.value,
    ],
    "cv_folds": [3, 5],
    "purge_days": [3, 5, 7],
    "embargo_days": [1, 2, 3],
    "use_optuna": [False, True],
    "optuna_n_trials": [20, 50],
    "use_robustness_testing": [False, True],
    "robustness_noise_level": [0.01, 0.05],
}

# Step 13: Evaluation Configuration
EVALUATION_OPTIONS = {
    "compute_calibration_curve": [False, True],
    "compute_trading_metrics": [False, True],
    "compute_robustness_metrics": [False, True],
}


# =============================================================================
# GRID SEARCH CONFIGURATION LEVELS
# =============================================================================

@dataclass
class GridSearchLevel:
    """Defines what's included at each grid search level."""
    name: str
    description: str

    # Which steps to vary
    vary_data: bool = False
    vary_synthetic: bool = False
    vary_augmentation: bool = False
    vary_preprocess: bool = False
    vary_features: bool = False
    vary_target: bool = False
    vary_feature_selection: bool = False
    vary_dim_reduction: bool = False
    vary_model: bool = True  # Always vary models
    vary_model_hyperparams: bool = False
    vary_cascade: bool = False
    vary_sample_weights: bool = False
    vary_training: bool = False
    vary_evaluation: bool = False


# Predefined levels
GRID_LEVELS = {
    "minimal": GridSearchLevel(
        name="minimal",
        description="Just model types and basic dim reduction",
        vary_dim_reduction=True,
        vary_model=True,
    ),
    "standard": GridSearchLevel(
        name="standard",
        description="Model types, dim reduction, preprocessing, cascade",
        vary_preprocess=True,
        vary_dim_reduction=True,
        vary_model=True,
        vary_model_hyperparams=True,
        vary_cascade=True,
    ),
    "comprehensive": GridSearchLevel(
        name="comprehensive",
        description="All pipeline steps except data/synthetic",
        vary_preprocess=True,
        vary_features=True,
        vary_target=True,
        vary_feature_selection=True,
        vary_dim_reduction=True,
        vary_model=True,
        vary_model_hyperparams=True,
        vary_cascade=True,
        vary_sample_weights=True,
        vary_training=True,
    ),
    "full": GridSearchLevel(
        name="full",
        description="Every single pipeline step",
        vary_data=True,
        vary_synthetic=True,
        vary_augmentation=True,
        vary_preprocess=True,
        vary_features=True,
        vary_target=True,
        vary_feature_selection=True,
        vary_dim_reduction=True,
        vary_model=True,
        vary_model_hyperparams=True,
        vary_cascade=True,
        vary_sample_weights=True,
        vary_training=True,
        vary_evaluation=True,
    ),
}


# =============================================================================
# COMPREHENSIVE GRID SEARCH GENERATOR
# =============================================================================

class ComprehensiveGridSearchGenerator:
    """
    Generates grid search configurations covering ALL 13 pipeline steps.

    This is the FULL grid search that explores the entire configuration space.
    """

    def __init__(
        self,
        level: str = "comprehensive",
        max_configs: int = 1000,
        random_sample: bool = True,
        seed: int = 42,
    ):
        if level not in GRID_LEVELS:
            raise ValueError(f"Unknown level: {level}. Choose from {list(GRID_LEVELS.keys())}")

        self.level_config = GRID_LEVELS[level]
        self.max_configs = max_configs
        self.random_sample = random_sample
        self.seed = seed

        random.seed(seed)

    def _get_options_for_step(self, step_name: str, options_dict: Dict) -> Dict:
        """Get options to include for a step based on level config."""
        attr_name = f"vary_{step_name}"
        if hasattr(self.level_config, attr_name) and getattr(self.level_config, attr_name):
            return options_dict
        return {}  # Return empty if not varying this step

    def estimate_total_combinations(self) -> Dict[str, Any]:
        """Estimate total number of combinations for each step."""
        estimates = {}
        total = 1

        if self.level_config.vary_data:
            n = len(DATA_OPTIONS.get("period", [1])) * len(DATA_OPTIONS.get("market_hours", [1]))
            estimates["data"] = n
            total *= max(n, 1)

        if self.level_config.vary_synthetic:
            n = len(SYNTHETIC_OPTIONS.get("enabled", [1])) * len(SYNTHETIC_OPTIONS.get("methods", [1]))
            estimates["synthetic"] = n
            total *= max(n, 1)

        if self.level_config.vary_augmentation:
            n = len(AUGMENTATION_OPTIONS.get("enabled", [1])) * len(AUGMENTATION_OPTIONS.get("methods", [1]))
            estimates["augmentation"] = n
            total *= max(n, 1)

        if self.level_config.vary_preprocess:
            n = (len(PREPROCESS_OPTIONS.get("outlier_method", [1])) *
                 len(PREPROCESS_OPTIONS.get("scaling_method", [1])))
            estimates["preprocess"] = n
            total *= max(n, 1)

        if self.level_config.vary_feature_selection:
            n = (len(FEATURE_SELECTION_OPTIONS.get("method", [1])) *
                 len(FEATURE_SELECTION_OPTIONS.get("n_features", [1])))
            estimates["feature_selection"] = n
            total *= max(n, 1)

        if self.level_config.vary_dim_reduction:
            n = (len(DIM_REDUCTION_OPTIONS.get("method", [1])) *
                 len(DIM_REDUCTION_OPTIONS.get("n_components", [1])))
            estimates["dim_reduction"] = n
            total *= max(n, 1)

        if self.level_config.vary_model:
            n = len(MODEL_OPTIONS.get("model_type", [1]))
            if self.level_config.vary_model_hyperparams:
                n *= (len(MODEL_OPTIONS.get("lr_C", [1])) *
                      len(MODEL_OPTIONS.get("tree_max_depth", [1])) *
                      len(MODEL_OPTIONS.get("tree_n_estimators", [1])))
            estimates["model"] = n
            total *= max(n, 1)

        if self.level_config.vary_cascade:
            n = len(CASCADE_OPTIONS.get("cascade_type", [1]))
            estimates["cascade"] = n
            total *= max(n, 1)

        if self.level_config.vary_sample_weights:
            n = len(SAMPLE_WEIGHT_OPTIONS.get("method", [1]))
            estimates["sample_weights"] = n
            total *= max(n, 1)

        if self.level_config.vary_training:
            n = (len(TRAINING_OPTIONS.get("cv_method", [1])) *
                 len(TRAINING_OPTIONS.get("cv_folds", [1])))
            estimates["training"] = n
            total *= max(n, 1)

        return {
            "step_estimates": estimates,
            "total_combinations": total,
            "max_configs": self.max_configs,
            "sampling_ratio": self.max_configs / total if total > 0 else 1.0,
        }

    def create_grid(self) -> GridSearchConfigGenerator:
        """Create the comprehensive grid search configuration."""
        gen = GridSearchConfigGenerator()

        # Step 1: Data Config
        if self.level_config.vary_data:
            gen.add_dimension('data_config.period', DATA_OPTIONS["period"])
            gen.add_dimension('data_config.market_hours', DATA_OPTIONS["market_hours"])

        # Step 2: Synthetic Data Config
        if self.level_config.vary_synthetic:
            gen.add_dimension('synthetic_config.enabled', SYNTHETIC_OPTIONS["enabled"])
            gen.add_dimension('synthetic_config.synthetic_weight', SYNTHETIC_OPTIONS["synthetic_weight"])

        # Step 3: Augmentation Config
        if self.level_config.vary_augmentation:
            gen.add_dimension('augmentation_config.enabled', AUGMENTATION_OPTIONS["enabled"])

        # Step 4: Preprocess Config
        if self.level_config.vary_preprocess:
            gen.add_dimension('preprocess_config.outlier_method', PREPROCESS_OPTIONS["outlier_method"])
            gen.add_dimension('preprocess_config.missing_method', PREPROCESS_OPTIONS["missing_method"])
            gen.add_dimension('preprocess_config.scaling_method', PREPROCESS_OPTIONS["scaling_method"])

        # Step 5: Feature Config
        if self.level_config.vary_features:
            gen.add_dimension('feature_config.include_volume_features', FEATURE_OPTIONS["include_volume_features"])
            gen.add_dimension('feature_config.include_extended_hours', FEATURE_OPTIONS["include_extended_hours"])
            gen.add_dimension('feature_config.include_cross_asset', FEATURE_OPTIONS["include_cross_asset"])

        # Step 6: Target Config
        if self.level_config.vary_target:
            gen.add_dimension('target_config.swing_threshold', TARGET_OPTIONS["swing_threshold"])
            gen.add_dimension('target_config.use_soft_targets', TARGET_OPTIONS["use_soft_targets"])
            gen.add_dimension('target_config.smoothing_method', TARGET_OPTIONS["smoothing_method"])

        # Step 7: Feature Selection Config
        if self.level_config.vary_feature_selection:
            gen.add_dimension('feature_selection_config.method', FEATURE_SELECTION_OPTIONS["method"])
            gen.add_dimension('feature_selection_config.n_features', FEATURE_SELECTION_OPTIONS["n_features"])
            gen.add_dimension('feature_selection_config.correlation_threshold', FEATURE_SELECTION_OPTIONS["correlation_threshold"])

        # Step 8: Dim Reduction Config
        if self.level_config.vary_dim_reduction:
            gen.add_dimension('dim_reduction_config.method', DIM_REDUCTION_OPTIONS["method"])
            gen.add_dimension('dim_reduction_config.n_components', DIM_REDUCTION_OPTIONS["n_components"])

        # Step 9: Model Config
        if self.level_config.vary_model:
            gen.add_dimension('model_config.model_type', MODEL_OPTIONS["model_type"])

            if self.level_config.vary_model_hyperparams:
                # Logistic
                gen.add_dimension('model_config.lr_C', MODEL_OPTIONS["lr_C"])
                gen.add_dimension('model_config.lr_l1_ratio', MODEL_OPTIONS["lr_l1_ratio"])
                # Trees
                gen.add_dimension('model_config.gb_max_depth', MODEL_OPTIONS["tree_max_depth"])
                gen.add_dimension('model_config.rf_max_depth', MODEL_OPTIONS["tree_max_depth"])
                gen.add_dimension('model_config.gb_n_estimators', MODEL_OPTIONS["tree_n_estimators"])
                gen.add_dimension('model_config.rf_n_estimators', MODEL_OPTIONS["tree_n_estimators"])
                gen.add_dimension('model_config.gb_learning_rate', MODEL_OPTIONS["gb_learning_rate"])
                gen.add_dimension('model_config.gb_subsample', MODEL_OPTIONS["gb_subsample"])

        # Step 10: Cascade Config
        if self.level_config.vary_cascade:
            gen.add_dimension('cascade_config.cascade_type', CASCADE_OPTIONS["cascade_type"])
            gen.add_dimension('cascade_config.mask_probability', CASCADE_OPTIONS["mask_probability"])
            gen.add_dimension('cascade_config.temporal_encoding', CASCADE_OPTIONS["temporal_encoding"])

        # Step 11: Sample Weight Config
        if self.level_config.vary_sample_weights:
            gen.add_dimension('sample_weight_config.method', SAMPLE_WEIGHT_OPTIONS["method"])

        # Step 12: Training Config
        if self.level_config.vary_training:
            gen.add_dimension('training_config.cv_method', TRAINING_OPTIONS["cv_method"])
            gen.add_dimension('training_config.cv_folds', TRAINING_OPTIONS["cv_folds"])
            gen.add_dimension('training_config.purge_days', TRAINING_OPTIONS["purge_days"])

        # Step 13: Evaluation Config
        if self.level_config.vary_evaluation:
            gen.add_dimension('evaluation_config.compute_calibration_curve', EVALUATION_OPTIONS["compute_calibration_curve"])
            gen.add_dimension('evaluation_config.compute_trading_metrics', EVALUATION_OPTIONS["compute_trading_metrics"])

        # Add constraints
        self._add_constraints(gen)

        return gen

    def _add_constraints(self, gen: GridSearchConfigGenerator):
        """Add constraints to filter invalid combinations."""
        # Linear models don't use tree hyperparameters
        linear_models = [
            ModelType.LOGISTIC_L1.value, ModelType.LOGISTIC_L2.value,
            ModelType.ELASTIC_NET.value, ModelType.RIDGE.value,
            ModelType.SVM_LINEAR.value, ModelType.SVM_RBF.value,
            ModelType.MLP.value, ModelType.MLP_DEEP.value,
            ModelType.GAUSSIAN_NB.value, ModelType.KNN.value,
        ]

        for model in linear_models:
            if self.level_config.vary_model_hyperparams:
                gen.add_constraint(
                    'model_config.model_type', model,
                    'model_config.gb_max_depth', MODEL_OPTIONS["tree_max_depth"][:1]
                )
                gen.add_constraint(
                    'model_config.model_type', model,
                    'model_config.gb_n_estimators', MODEL_OPTIONS["tree_n_estimators"][:1]
                )

        # Boruta requires simpler dim reduction
        if self.level_config.vary_feature_selection and self.level_config.vary_dim_reduction:
            gen.add_constraint(
                'feature_selection_config.method', FeatureSelectionMethod.BORUTA.value,
                'dim_reduction_config.method', [DimReductionMethod.NONE.value, DimReductionMethod.PCA.value]
            )

    def generate_configs(
        self,
        target_type: str = "swing",
        experiment_id: str = None,
    ) -> List[ModelEntry]:
        """Generate model configurations for grid search."""
        gen = self.create_grid()

        configs = gen.generate_configs(
            target_type=target_type,
            experiment_id=experiment_id,
            max_configs=self.max_configs,
            random_sample=self.random_sample,
        )

        return configs

    def get_summary(self) -> str:
        """Get human-readable summary of grid configuration."""
        est = self.estimate_total_combinations()

        lines = [
            "=" * 70,
            f"COMPREHENSIVE GRID SEARCH - Level: {self.level_config.name.upper()}",
            "=" * 70,
            "",
            f"Description: {self.level_config.description}",
            "",
            "Pipeline Steps Varied:",
        ]

        step_names = [
            ("vary_data", "1. Data Config"),
            ("vary_synthetic", "2. Synthetic Data"),
            ("vary_augmentation", "3. Augmentation"),
            ("vary_preprocess", "4. Preprocessing"),
            ("vary_features", "5. Feature Config"),
            ("vary_target", "6. Target Config"),
            ("vary_feature_selection", "7. Feature Selection"),
            ("vary_dim_reduction", "8. Dim Reduction"),
            ("vary_model", "9. Model Type"),
            ("vary_model_hyperparams", "   - Model Hyperparams"),
            ("vary_cascade", "10. Cascade Config"),
            ("vary_sample_weights", "11. Sample Weights"),
            ("vary_training", "12. Training Config"),
            ("vary_evaluation", "13. Evaluation Config"),
        ]

        for attr, name in step_names:
            status = "YES" if getattr(self.level_config, attr) else "no"
            lines.append(f"  {name}: {status}")

        lines.extend([
            "",
            "Combination Estimates:",
        ])

        for step, count in est["step_estimates"].items():
            lines.append(f"  {step}: {count:,}")

        lines.extend([
            "",
            f"Total Possible Combinations: {est['total_combinations']:,}",
            f"Max Configs to Train: {self.max_configs:,}",
            f"Sampling Ratio: {est['sampling_ratio']:.6%}",
            "=" * 70,
        ])

        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def create_minimal_grid(n_configs: int = 50) -> List[ModelEntry]:
    """Create minimal grid (model types + dim reduction only)."""
    gen = ComprehensiveGridSearchGenerator(level="minimal", max_configs=n_configs)
    return gen.generate_configs(target_type="swing")


def create_standard_grid(n_configs: int = 200) -> List[ModelEntry]:
    """Create standard grid (preprocessing + model + cascade)."""
    gen = ComprehensiveGridSearchGenerator(level="standard", max_configs=n_configs)
    return gen.generate_configs(target_type="swing")


def create_comprehensive_grid(n_configs: int = 500) -> List[ModelEntry]:
    """Create comprehensive grid (most steps except data/synthetic)."""
    gen = ComprehensiveGridSearchGenerator(level="comprehensive", max_configs=n_configs)
    return gen.generate_configs(target_type="swing")


def create_full_grid(n_configs: int = 1000) -> List[ModelEntry]:
    """Create full grid (ALL 13 pipeline steps)."""
    gen = ComprehensiveGridSearchGenerator(level="full", max_configs=n_configs)
    return gen.generate_configs(target_type="swing")


if __name__ == "__main__":
    # Test each level
    for level in ["minimal", "standard", "comprehensive", "full"]:
        print(f"\n{'='*70}")
        print(f"Testing level: {level}")
        print('='*70)

        gen = ComprehensiveGridSearchGenerator(level=level, max_configs=10)
        print(gen.get_summary())

        configs = gen.generate_configs(target_type="swing")
        print(f"\nGenerated {len(configs)} configs")

        # Show first config details
        if configs:
            c = configs[0]
            print(f"\nFirst config details:")
            print(f"  Model: {c.model_config.model_type}")
            print(f"  Dim Reduction: {c.dim_reduction_config.method}")
            print(f"  Cascade: {c.cascade_config.cascade_type}")
            print(f"  Preprocessing Outlier: {c.preprocess_config.outlier_method}")
            print(f"  Feature Selection: {c.feature_selection_config.method}")
            print(f"  Sample Weighting: {c.sample_weight_config.method}")
