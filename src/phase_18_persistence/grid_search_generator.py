"""
GIGA TRADER - Model Registry v2: Grid Search Generator
========================================================
GridSearchConfigGenerator class and utility functions.
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple

from src.phase_18_persistence.registry_enums import (
    DataSource,
    TimeResolution,
    DataPeriod,
    MarketHours,
    SyntheticDataMethod,
    AugmentationMethod,
    OutlierMethod,
    MissingValueMethod,
    ScalingMethod,
    TransformMethod,
    FeatureSelectionMethod,
    FeatureGroupMode,
    DimReductionMethod,
    ModelType,
    EnsembleMethod,
    CascadeType,
    TemporalEncoding,
    TargetType,
    TargetDefinition,
    LabelSmoothingMethod,
    CVMethod,
    ScoringMetric,
    SampleWeightMethod,
    ModelStatus,
)
from src.phase_18_persistence.registry_configs import (
    ModelEntry,
)
from src.phase_18_persistence.model_registry import ModelRegistryV2
from src.core.registry_db import get_registry_db

logger = logging.getLogger("MODEL_REGISTRY_V2")


# =============================================================================
# GRID SEARCH CONFIGURATION GENERATOR - MASSIVE GRID SUPPORT
# =============================================================================

class GridSearchConfigGenerator:
    """
    Generates configuration combinations for massive grid search.

    Supports grid search across ALL pipeline steps:
    1. Data source and period
    2. Synthetic data methods
    3. Data augmentation
    4. Outlier handling
    5. Missing value handling
    6. Scaling methods
    7. Feature groups
    8. Target definitions
    9. Feature selection methods
    10. Dimensionality reduction
    11. Model types
    12. Cascade types
    13. Sample weighting
    14. CV methods

    Each dimension can be a list of options.
    The generator produces all combinations (Cartesian product).
    """

    def __init__(self):
        self.dimensions = {}
        self.constraints = []  # List of (dimension1, value1, dimension2, allowed_values2) tuples

    def add_dimension(self, name: str, options: List[Any]):
        """Add a dimension to the grid."""
        self.dimensions[name] = options
        return self

    def add_constraint(self, dim1: str, val1: Any, dim2: str, allowed_vals2: List[Any]):
        """
        Add a constraint: when dim1=val1, dim2 must be in allowed_vals2.
        Useful for incompatible combinations.
        """
        self.constraints.append((dim1, val1, dim2, allowed_vals2))
        return self

    def _check_constraints(self, config: Dict) -> bool:
        """Check if a configuration satisfies all constraints."""
        for dim1, val1, dim2, allowed_vals2 in self.constraints:
            if config.get(dim1) == val1:
                if config.get(dim2) not in allowed_vals2:
                    return False
        return True

    def _sample_random_combinations(
        self,
        dim_names: List[str],
        dim_options: List[List[Any]],
        n_samples: int,
    ) -> List[Tuple]:
        """
        Sample random combinations without enumerating all possibilities.

        This uses direct indexing into the combination space, treating it
        like a mixed-radix number system where each digit corresponds to
        a dimension.
        """
        import random

        # Calculate the sizes of each dimension
        sizes = [len(opts) for opts in dim_options]
        total = 1
        for s in sizes:
            total *= s

        # If n_samples >= total, just enumerate all (small space)
        if n_samples >= total:
            from itertools import product
            return list(product(*dim_options))

        # Sample random indices into the combination space
        sampled_indices = set()
        combinations = []

        # Keep sampling until we have enough unique combinations
        max_attempts = n_samples * 10
        attempts = 0

        while len(combinations) < n_samples and attempts < max_attempts:
            # Generate a random index in [0, total)
            idx = random.randint(0, total - 1)

            if idx not in sampled_indices:
                sampled_indices.add(idx)

                # Convert index to combination (like mixed-radix decoding)
                combo = []
                remaining = idx
                for i, size in enumerate(sizes):
                    combo.append(dim_options[i][remaining % size])
                    remaining //= size

                combinations.append(tuple(combo))

            attempts += 1

        return combinations

    def count_combinations(self) -> int:
        """Count total combinations without generating them."""
        from functools import reduce
        return reduce(lambda x, y: x * y, [len(v) for v in self.dimensions.values()], 1)

    def generate_configs(
        self,
        target_type: str,
        experiment_id: str = None,
        max_configs: int = None,
        random_sample: bool = False,
    ) -> List[ModelEntry]:
        """
        Generate all configuration combinations.

        Args:
            target_type: Type of prediction target
            experiment_id: Optional experiment identifier
            max_configs: Maximum number of configs to generate (for sampling)
            random_sample: If True and max_configs set, randomly sample instead of first N

        Returns:
            List of ModelEntry objects (not yet trained)
        """
        from itertools import product
        import random

        # Get all dimension names and their options
        dim_names = list(self.dimensions.keys())
        dim_options = [self.dimensions[name] for name in dim_names]

        total_combinations = self.count_combinations()
        logger.info(f"Total possible combinations: {total_combinations}")

        configs = []
        grid_search_id = experiment_id or datetime.now().strftime('%Y%m%d_%H%M%S')

        # For large combination spaces, use random sampling without full enumeration
        # This prevents memory errors with quintillions of combinations
        if random_sample and max_configs and total_combinations > max_configs * 10:
            logger.info(f"Using direct random sampling (space too large for enumeration)")
            valid_combinations = self._sample_random_combinations(
                dim_names, dim_options, max_configs * 3  # Over-sample to account for constraint filtering
            )
            # Filter by constraints
            valid_combinations = [
                combo for combo in valid_combinations
                if self._check_constraints(dict(zip(dim_names, combo)))
            ][:max_configs]
            logger.info(f"Random sampling produced {len(valid_combinations)} valid configs")
        else:
            # Generate all combinations (for smaller spaces)
            all_combinations = list(product(*dim_options))

            # Filter by constraints
            valid_combinations = []
            for combo in all_combinations:
                config = dict(zip(dim_names, combo))
                if self._check_constraints(config):
                    valid_combinations.append(combo)

            logger.info(f"Valid combinations after constraints: {len(valid_combinations)}")

            # Sample if needed
            if max_configs and len(valid_combinations) > max_configs:
                if random_sample:
                    valid_combinations = random.sample(valid_combinations, max_configs)
                else:
                    valid_combinations = valid_combinations[:max_configs]
                logger.info(f"Sampled down to: {len(valid_combinations)}")

        for idx, combination in enumerate(valid_combinations):
            config = dict(zip(dim_names, combination))

            # Create entry
            entry = ModelEntry(
                target_type=target_type,
                experiment_id=grid_search_id,
                grid_search_id=grid_search_id,
                grid_position=idx,
                grid_total=len(valid_combinations),
            )

            # Apply grid search values
            for key, value in config.items():
                self._set_nested_attr(entry, key, value)

            # Generate hash for deduplication
            entry.hyperparameters_hash = entry.get_config_hash()

            configs.append(entry)

        logger.info(f"Generated {len(configs)} configurations")
        return configs

    def _set_nested_attr(self, obj: Any, key: str, value: Any):
        """Set a nested attribute like 'model_config.model_type'."""
        parts = key.split('.')
        if len(parts) == 1:
            # Try common config objects
            for config_name in ['model_config', 'cascade_config', 'training_config',
                               'dim_reduction_config', 'preprocess_config', 'feature_config']:
                config_obj = getattr(obj, config_name, None)
                if config_obj and hasattr(config_obj, key):
                    setattr(config_obj, key, value)
                    return
        elif len(parts) == 2:
            config_name, param = parts
            config_obj = getattr(obj, config_name, None)
            if config_obj and hasattr(config_obj, param):
                setattr(config_obj, param, value)
        elif len(parts) == 3:
            # For deeper nesting if needed
            config_name, sub_config, param = parts
            config_obj = getattr(obj, config_name, None)
            if config_obj:
                sub_obj = getattr(config_obj, sub_config, None)
                if sub_obj and hasattr(sub_obj, param):
                    setattr(sub_obj, param, value)

    @classmethod
    def create_minimal_grid(cls) -> 'GridSearchConfigGenerator':
        """Create minimal grid for quick testing."""
        gen = cls()

        # 2 resolutions x 5 dim reduction x 2 group modes x 2 models x 4 cascades = 160
        gen.add_dimension('data_config.primary_resolution', [
            TimeResolution.MINUTE_1.value,
            TimeResolution.MINUTE_5.value,
        ])

        gen.add_dimension('dim_reduction_config.method', [
            DimReductionMethod.NONE.value,
            DimReductionMethod.PCA.value,
            DimReductionMethod.KERNEL_PCA_RBF.value,
            DimReductionMethod.ICA.value,
            DimReductionMethod.ENSEMBLE_PLUS.value,
        ])

        gen.add_dimension('feature_group_config.group_mode', [
            FeatureGroupMode.FLAT.value,
            FeatureGroupMode.GROUPED_PROTECTED.value,
        ])

        gen.add_dimension('model_config.model_type', [
            ModelType.LOGISTIC_L2.value,
            ModelType.GRADIENT_BOOSTING.value,
        ])

        gen.add_dimension('cascade_config.cascade_type', [
            CascadeType.BASE.value,
            CascadeType.MASKED.value,
            CascadeType.INTERMITTENT_MASKED.value,
            CascadeType.ATTENTION.value,
        ])

        return gen

    @classmethod
    def create_standard_grid(cls) -> 'GridSearchConfigGenerator':
        """Create standard grid search (500+ configs)."""
        gen = cls()

        # Data period (4)
        gen.add_dimension('data_config.period', [
            DataPeriod.YEARS_3.value,
            DataPeriod.YEARS_5.value,
            DataPeriod.YEARS_7.value,
            DataPeriod.YEARS_10.value,
        ])

        # Bar resolution (4)
        gen.add_dimension('data_config.primary_resolution', [
            TimeResolution.MINUTE_1.value,
            TimeResolution.MINUTE_3.value,
            TimeResolution.MINUTE_5.value,
            TimeResolution.MINUTE_15.value,
        ])

        # Outlier methods (4)
        gen.add_dimension('preprocess_config.outlier_method', [
            OutlierMethod.NONE.value,
            OutlierMethod.WINSORIZE_1.value,
            OutlierMethod.CLIP_3STD.value,
            OutlierMethod.ISOLATION_FOREST.value,
        ])

        # Scaling methods (3)
        gen.add_dimension('preprocess_config.scaling_method', [
            ScalingMethod.STANDARD.value,
            ScalingMethod.ROBUST.value,
            ScalingMethod.QUANTILE_NORMAL.value,
        ])

        # Feature selection (3)
        gen.add_dimension('feature_selection_config.method', [
            FeatureSelectionMethod.NONE.value,
            FeatureSelectionMethod.MUTUAL_INFO.value,
            FeatureSelectionMethod.TREE_IMPORTANCE.value,
        ])

        # Dimensionality reduction (5)
        gen.add_dimension('dim_reduction_config.method', [
            DimReductionMethod.NONE.value,
            DimReductionMethod.PCA.value,
            DimReductionMethod.KERNEL_PCA_RBF.value,
            DimReductionMethod.UMAP.value,
            DimReductionMethod.ENSEMBLE_PLUS.value,
        ])

        # Feature group mode (2)
        gen.add_dimension('feature_group_config.group_mode', [
            FeatureGroupMode.FLAT.value,
            FeatureGroupMode.GROUPED_PROTECTED.value,
        ])

        # Model types (4)
        gen.add_dimension('model_config.model_type', [
            ModelType.LOGISTIC_L2.value,
            ModelType.GRADIENT_BOOSTING.value,
            ModelType.XGBOOST.value,
            ModelType.LIGHTGBM.value,
        ])

        # Cascade types (4)
        gen.add_dimension('cascade_config.cascade_type', [
            CascadeType.BASE.value,
            CascadeType.MASKED.value,
            CascadeType.ATTENTION.value,
            CascadeType.MULTI_RESOLUTION.value,
        ])

        # Total: 3 x 4 x 4 x 3 x 3 x 5 x 2 x 4 x 4 = 69,120 configs
        return gen

    @classmethod
    def create_full_grid(cls) -> 'GridSearchConfigGenerator':
        """
        Create full/massive grid search covering ALL pipeline steps.
        WARNING: This generates 100,000+ configurations!
        """
        gen = cls()

        # ===== STEP 1: Data Configuration =====
        gen.add_dimension('data_config.source', [
            DataSource.ALPACA.value,
            DataSource.YFINANCE.value,
        ])

        gen.add_dimension('data_config.period', [
            DataPeriod.YEARS_3.value,
            DataPeriod.YEARS_5.value,
            DataPeriod.YEARS_7.value,
            DataPeriod.YEARS_10.value,
        ])

        gen.add_dimension('data_config.primary_resolution', [
            TimeResolution.MINUTE_1.value,
            TimeResolution.MINUTE_2.value,
            TimeResolution.MINUTE_3.value,
            TimeResolution.MINUTE_5.value,
            TimeResolution.MINUTE_10.value,
            TimeResolution.MINUTE_15.value,
            TimeResolution.MINUTE_30.value,
        ])

        gen.add_dimension('data_config.market_hours', [
            MarketHours.REGULAR_ONLY.value,
            MarketHours.ALL_HOURS.value,
        ])

        # ===== STEP 2: Synthetic Data =====
        gen.add_dimension('synthetic_config.enabled', [False, True])

        gen.add_dimension('synthetic_config.synthetic_weight', [0.0, 0.2, 0.3])

        # ===== STEP 3: Augmentation =====
        gen.add_dimension('augmentation_config.enabled', [False, True])

        # ===== STEP 4: Preprocessing =====
        gen.add_dimension('preprocess_config.outlier_method', [
            OutlierMethod.NONE.value,
            OutlierMethod.WINSORIZE_1.value,
            OutlierMethod.WINSORIZE_5.value,
            OutlierMethod.CLIP_3STD.value,
            OutlierMethod.IQR_1_5.value,
            OutlierMethod.ISOLATION_FOREST.value,
        ])

        gen.add_dimension('preprocess_config.missing_method', [
            MissingValueMethod.FORWARD_FILL.value,
            MissingValueMethod.LINEAR_INTERPOLATE.value,
            MissingValueMethod.KNN_IMPUTE.value,
        ])

        gen.add_dimension('preprocess_config.scaling_method', [
            ScalingMethod.STANDARD.value,
            ScalingMethod.MINMAX.value,
            ScalingMethod.ROBUST.value,
            ScalingMethod.QUANTILE_NORMAL.value,
            ScalingMethod.POWER_YEOBJOHNSON.value,
        ])

        # ===== STEP 5: Feature Engineering =====
        gen.add_dimension('feature_config.include_extended_hours', [True, False])
        gen.add_dimension('feature_config.include_cross_asset_features', [False, True])
        gen.add_dimension('feature_config.include_breadth_features', [False, True])

        # ===== STEP 6: Target Definition =====
        gen.add_dimension('target_config.swing_threshold', [0.002, 0.0025, 0.003])
        gen.add_dimension('target_config.use_soft_targets', [False, True])

        # ===== STEP 7: Feature Selection =====
        gen.add_dimension('feature_selection_config.method', [
            FeatureSelectionMethod.NONE.value,
            FeatureSelectionMethod.VARIANCE_THRESHOLD.value,
            FeatureSelectionMethod.CORRELATION_FILTER.value,
            FeatureSelectionMethod.MUTUAL_INFO.value,
            FeatureSelectionMethod.RFE.value,
            FeatureSelectionMethod.TREE_IMPORTANCE.value,
            FeatureSelectionMethod.BORUTA.value,
        ])

        # ===== STEP 8: Dimensionality Reduction =====
        gen.add_dimension('dim_reduction_config.method', [
            DimReductionMethod.NONE.value,
            DimReductionMethod.PCA.value,
            DimReductionMethod.SPARSE_PCA.value,
            DimReductionMethod.KERNEL_PCA_RBF.value,
            DimReductionMethod.KERNEL_PCA_POLY.value,
            DimReductionMethod.UMAP.value,
            DimReductionMethod.ICA.value,
            DimReductionMethod.FACTOR_ANALYSIS.value,
            DimReductionMethod.AGGLOMERATION.value,
            DimReductionMethod.ENSEMBLE_PLUS.value,
        ])

        gen.add_dimension('dim_reduction_config.n_components', [20, 30, 40, 50])

        # ===== STEP 9: Model Types =====
        gen.add_dimension('model_config.model_type', [
            ModelType.LOGISTIC_L1.value,
            ModelType.LOGISTIC_L2.value,
            ModelType.ELASTIC_NET.value,
            ModelType.GRADIENT_BOOSTING.value,
            ModelType.HIST_GRADIENT_BOOSTING.value,
            ModelType.XGBOOST.value,
            ModelType.LIGHTGBM.value,
            ModelType.RANDOM_FOREST.value,
            ModelType.EXTRA_TREES.value,
            ModelType.SVM_RBF.value,
            ModelType.MLP.value,
        ])

        # ===== STEP 10: Cascade Types =====
        gen.add_dimension('cascade_config.cascade_type', [
            CascadeType.BASE.value,
            CascadeType.MASKED.value,
            CascadeType.INTERMITTENT_MASKED.value,
            CascadeType.ATTENTION.value,
            CascadeType.STOCHASTIC_DEPTH.value,
            CascadeType.MULTI_RESOLUTION.value,
            CascadeType.MIXTURE_OF_EXPERTS.value,
        ])

        # ===== STEP 11: Sample Weighting =====
        gen.add_dimension('sample_weight_config.method', [
            SampleWeightMethod.NONE.value,
            SampleWeightMethod.BALANCED.value,
            SampleWeightMethod.TIME_DECAY_EXPONENTIAL.value,
        ])

        # ===== STEP 12: Cross-Validation =====
        gen.add_dimension('training_config.cv_method', [
            CVMethod.TIMESERIES_SPLIT.value,
            CVMethod.PURGED_KFOLD.value,
            CVMethod.WALK_FORWARD.value,
        ])

        gen.add_dimension('training_config.cv_folds', [3, 5])

        # Add constraint: no dim reduction when feature selection is strong
        gen.add_constraint(
            'feature_selection_config.method', FeatureSelectionMethod.BORUTA.value,
            'dim_reduction_config.method', [DimReductionMethod.NONE.value, DimReductionMethod.PCA.value]
        )

        return gen

    @classmethod
    def create_focused_grid(
        cls,
        focus_area: str = "model",
        base_config: Dict = None
    ) -> 'GridSearchConfigGenerator':
        """
        Create a focused grid search on a specific area.

        Args:
            focus_area: One of "data", "preprocess", "features", "dim_reduction",
                       "model", "cascade", "training"
            base_config: Base configuration to use for non-focused dimensions
        """
        gen = cls()

        if focus_area == "data":
            gen.add_dimension('data_config.source', [
                DataSource.ALPACA.value, DataSource.YFINANCE.value
            ])
            gen.add_dimension('data_config.period', [
                DataPeriod.YEARS_2.value, DataPeriod.YEARS_3.value,
                DataPeriod.YEARS_5.value, DataPeriod.YEARS_10.value
            ])
            gen.add_dimension('data_config.primary_resolution', [
                TimeResolution.MINUTE_1.value, TimeResolution.MINUTE_5.value,
                TimeResolution.MINUTE_15.value
            ])
            gen.add_dimension('data_config.market_hours', [
                MarketHours.REGULAR_ONLY.value, MarketHours.ALL_HOURS.value
            ])

        elif focus_area == "preprocess":
            gen.add_dimension('preprocess_config.outlier_method', [
                OutlierMethod.NONE.value, OutlierMethod.WINSORIZE_0_5.value,
                OutlierMethod.WINSORIZE_1.value, OutlierMethod.WINSORIZE_5.value,
                OutlierMethod.CLIP_3STD.value, OutlierMethod.IQR_1_5.value,
                OutlierMethod.ISOLATION_FOREST.value
            ])
            gen.add_dimension('preprocess_config.scaling_method', [
                ScalingMethod.STANDARD.value, ScalingMethod.MINMAX.value,
                ScalingMethod.ROBUST.value, ScalingMethod.QUANTILE_NORMAL.value,
                ScalingMethod.POWER_YEOBJOHNSON.value
            ])
            gen.add_dimension('preprocess_config.missing_method', [
                MissingValueMethod.FORWARD_FILL.value,
                MissingValueMethod.LINEAR_INTERPOLATE.value,
                MissingValueMethod.KNN_IMPUTE.value
            ])

        elif focus_area == "dim_reduction":
            gen.add_dimension('dim_reduction_config.method', [
                DimReductionMethod.NONE.value, DimReductionMethod.PCA.value,
                DimReductionMethod.SPARSE_PCA.value, DimReductionMethod.KERNEL_PCA_RBF.value,
                DimReductionMethod.KERNEL_PCA_POLY.value, DimReductionMethod.UMAP.value,
                DimReductionMethod.TSNE.value, DimReductionMethod.ICA.value,
                DimReductionMethod.FACTOR_ANALYSIS.value, DimReductionMethod.NMF.value,
                DimReductionMethod.AGGLOMERATION.value, DimReductionMethod.ENSEMBLE_PLUS.value
            ])
            gen.add_dimension('dim_reduction_config.n_components', [
                10, 15, 20, 25, 30, 40, 50
            ])

        elif focus_area == "model":
            gen.add_dimension('model_config.model_type', [
                ModelType.LOGISTIC_L1.value, ModelType.LOGISTIC_L2.value,
                ModelType.ELASTIC_NET.value, ModelType.GRADIENT_BOOSTING.value,
                ModelType.XGBOOST.value, ModelType.LIGHTGBM.value,
                ModelType.CATBOOST.value, ModelType.RANDOM_FOREST.value,
                ModelType.EXTRA_TREES.value, ModelType.SVM_RBF.value,
                ModelType.MLP.value, ModelType.KNN.value
            ])
            # Add model-specific hyperparameters
            gen.add_dimension('model_config.gb_max_depth', [2, 3, 4, 5])
            gen.add_dimension('model_config.gb_n_estimators', [50, 100, 200])
            gen.add_dimension('model_config.gb_learning_rate', [0.01, 0.05, 0.1])

        elif focus_area == "cascade":
            gen.add_dimension('cascade_config.cascade_type', [
                CascadeType.BASE.value, CascadeType.MASKED.value,
                CascadeType.INTERMITTENT_MASKED.value, CascadeType.ATTENTION.value,
                CascadeType.SELF_ATTENTION.value, CascadeType.STOCHASTIC_DEPTH.value,
                CascadeType.MULTI_RESOLUTION.value, CascadeType.HIERARCHICAL.value,
                CascadeType.MIXTURE_OF_EXPERTS.value
            ])
            gen.add_dimension('cascade_config.attention_type', [
                "softmax", "sigmoid", "linear"
            ])
            gen.add_dimension('cascade_config.temporal_encoding', [
                TemporalEncoding.NONE.value, TemporalEncoding.SINUSOIDAL.value,
                TemporalEncoding.LEARNED.value
            ])

        elif focus_area == "training":
            gen.add_dimension('training_config.cv_method', [
                CVMethod.KFOLD.value, CVMethod.STRATIFIED_KFOLD.value,
                CVMethod.TIMESERIES_SPLIT.value, CVMethod.PURGED_KFOLD.value,
                CVMethod.WALK_FORWARD.value
            ])
            gen.add_dimension('training_config.cv_folds', [3, 5, 10])
            gen.add_dimension('training_config.early_stopping', [True, False])
            gen.add_dimension('sample_weight_config.method', [
                SampleWeightMethod.NONE.value, SampleWeightMethod.BALANCED.value,
                SampleWeightMethod.TIME_DECAY_EXPONENTIAL.value
            ])

        return gen

    def get_summary(self) -> str:
        """Get a summary of the grid configuration."""
        lines = [
            "=" * 60,
            "GRID SEARCH CONFIGURATION SUMMARY",
            "=" * 60,
            f"Total dimensions: {len(self.dimensions)}",
            f"Total combinations: {self.count_combinations():,}",
            f"Constraints: {len(self.constraints)}",
            "",
            "Dimensions:",
        ]

        for name, options in self.dimensions.items():
            lines.append(f"  {name}: {len(options)} options")
            if len(options) <= 5:
                for opt in options:
                    lines.append(f"    - {opt}")
            else:
                for opt in options[:3]:
                    lines.append(f"    - {opt}")
                lines.append(f"    ... and {len(options) - 3} more")

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_model_entry(
    target_type: str,
    cascade_type: str = CascadeType.BASE.value,
    model_type: str = ModelType.GRADIENT_BOOSTING.value,
    **kwargs
) -> ModelEntry:
    """Create a new model entry with specified configuration."""
    entry = ModelEntry(target_type=target_type)
    entry.cascade_config.cascade_type = cascade_type
    entry.model_config.model_type = model_type

    for key, value in kwargs.items():
        if hasattr(entry, key):
            setattr(entry, key, value)

    return entry


def list_all_options() -> Dict[str, List[str]]:
    """
    List all available options for each configuration dimension.

    Returns:
        Dictionary mapping dimension name to list of available options
    """
    return {
        # Data
        "data_source": [e.value for e in DataSource],
        "time_resolution": [e.value for e in TimeResolution],
        "data_period": [e.value for e in DataPeriod],
        "market_hours": [e.value for e in MarketHours],

        # Synthetic
        "synthetic_data_method": [e.value for e in SyntheticDataMethod],
        "augmentation_method": [e.value for e in AugmentationMethod],

        # Preprocessing
        "outlier_method": [e.value for e in OutlierMethod],
        "missing_value_method": [e.value for e in MissingValueMethod],
        "scaling_method": [e.value for e in ScalingMethod],
        "transform_method": [e.value for e in TransformMethod],

        # Feature Selection
        "feature_selection_method": [e.value for e in FeatureSelectionMethod],

        # Dimensionality Reduction
        "dim_reduction_method": [e.value for e in DimReductionMethod],

        # Model
        "model_type": [e.value for e in ModelType],
        "ensemble_method": [e.value for e in EnsembleMethod],

        # Cascade
        "cascade_type": [e.value for e in CascadeType],
        "temporal_encoding": [e.value for e in TemporalEncoding],

        # Target
        "target_type": [e.value for e in TargetType],
        "target_definition": [e.value for e in TargetDefinition],
        "label_smoothing_method": [e.value for e in LabelSmoothingMethod],

        # Training
        "cv_method": [e.value for e in CVMethod],
        "scoring_metric": [e.value for e in ScoringMetric],
        "sample_weight_method": [e.value for e in SampleWeightMethod],

        # Status
        "model_status": [e.value for e in ModelStatus],
    }


def print_all_options():
    """Print all available configuration options."""
    options = list_all_options()
    print("=" * 70)
    print("AVAILABLE CONFIGURATION OPTIONS")
    print("=" * 70)

    for category, values in options.items():
        print(f"\n{category.upper().replace('_', ' ')}:")
        for v in values:
            print(f"  - {v}")


def estimate_grid_size(grid_config: Dict[str, List]) -> Dict[str, Any]:
    """
    Estimate the size and resource requirements of a grid search.

    Args:
        grid_config: Dictionary mapping dimension names to lists of options

    Returns:
        Dictionary with size estimates and recommendations
    """
    from functools import reduce

    n_combinations = reduce(lambda x, y: x * y, [len(v) for v in grid_config.values()], 1)

    # Estimate time per model (rough averages)
    time_estimates = {
        "logistic": 30,  # seconds
        "gradient_boosting": 120,
        "xgboost": 90,
        "lightgbm": 60,
        "random_forest": 180,
        "svm": 300,
        "mlp": 240,
    }

    avg_time_per_model = 120  # seconds
    total_time_hours = (n_combinations * avg_time_per_model) / 3600

    # Estimate storage
    avg_model_size_mb = 5
    total_storage_gb = (n_combinations * avg_model_size_mb) / 1024

    return {
        "n_combinations": n_combinations,
        "dimensions": {k: len(v) for k, v in grid_config.items()},
        "estimated_time_hours": round(total_time_hours, 1),
        "estimated_storage_gb": round(total_storage_gb, 1),
        "recommendation": (
            "manageable" if n_combinations < 1000
            else "large - consider sampling" if n_combinations < 10000
            else "very large - use focused grid or random search"
        ),
    }


def create_quick_experiment(
    target_type: str = TargetType.SWING.value,
    n_configs: int = 10,
) -> List[ModelEntry]:
    """
    Create a quick experiment with a small number of diverse configurations.

    Useful for rapid prototyping and sanity checks.
    """
    configs = []

    # Predefined diverse configurations
    presets = [
        # Baseline
        {"model_type": ModelType.LOGISTIC_L2.value, "cascade_type": CascadeType.BASE.value,
         "dim_method": DimReductionMethod.NONE.value},
        # Simple with PCA
        {"model_type": ModelType.LOGISTIC_L2.value, "cascade_type": CascadeType.BASE.value,
         "dim_method": DimReductionMethod.PCA.value},
        # Gradient Boosting
        {"model_type": ModelType.GRADIENT_BOOSTING.value, "cascade_type": CascadeType.BASE.value,
         "dim_method": DimReductionMethod.PCA.value},
        # XGBoost with Ensemble+
        {"model_type": ModelType.XGBOOST.value, "cascade_type": CascadeType.BASE.value,
         "dim_method": DimReductionMethod.ENSEMBLE_PLUS.value},
        # LightGBM with attention cascade
        {"model_type": ModelType.LIGHTGBM.value, "cascade_type": CascadeType.ATTENTION.value,
         "dim_method": DimReductionMethod.PCA.value},
        # Random Forest with masked cascade
        {"model_type": ModelType.RANDOM_FOREST.value, "cascade_type": CascadeType.MASKED.value,
         "dim_method": DimReductionMethod.UMAP.value},
        # Elastic Net with ICA
        {"model_type": ModelType.ELASTIC_NET.value, "cascade_type": CascadeType.BASE.value,
         "dim_method": DimReductionMethod.ICA.value},
        # SVM with Kernel PCA
        {"model_type": ModelType.SVM_RBF.value, "cascade_type": CascadeType.BASE.value,
         "dim_method": DimReductionMethod.KERNEL_PCA_RBF.value},
        # MLP with mixture of experts
        {"model_type": ModelType.MLP.value, "cascade_type": CascadeType.MIXTURE_OF_EXPERTS.value,
         "dim_method": DimReductionMethod.PCA.value},
        # Extra Trees with multi-resolution
        {"model_type": ModelType.EXTRA_TREES.value, "cascade_type": CascadeType.MULTI_RESOLUTION.value,
         "dim_method": DimReductionMethod.AGGLOMERATION.value},
    ]

    for i, preset in enumerate(presets[:n_configs]):
        entry = ModelEntry(target_type=target_type)
        entry.model_config.model_type = preset["model_type"]
        entry.cascade_config.cascade_type = preset["cascade_type"]
        entry.dim_reduction_config.method = preset["dim_method"]
        entry.tags = ["quick_experiment"]
        configs.append(entry)

    return configs


# =============================================================================
# PIPELINE STEP DESCRIPTIONS
# =============================================================================

PIPELINE_STEPS = {
    1: {
        "name": "Data Loading",
        "config_class": "DataConfig",
        "description": "Load market data from various sources",
        "key_parameters": ["source", "period", "primary_resolution", "market_hours"],
    },
    2: {
        "name": "Synthetic Data",
        "config_class": "SyntheticDataConfig",
        "description": "Generate 'What SPY could have been' synthetic data",
        "key_parameters": ["enabled", "methods", "synthetic_weight"],
    },
    3: {
        "name": "Data Augmentation",
        "config_class": "AugmentationConfig",
        "description": "Augment training data with various techniques",
        "key_parameters": ["enabled", "methods", "oversampling_strategy"],
    },
    4: {
        "name": "Preprocessing",
        "config_class": "PreprocessConfig",
        "description": "Handle outliers, missing values, and scale features",
        "key_parameters": ["outlier_method", "missing_method", "scaling_method"],
    },
    5: {
        "name": "Feature Engineering",
        "config_class": "FeatureConfig",
        "description": "Engineer features from raw data",
        "key_parameters": ["include_*", "rolling_windows", "lag_periods"],
    },
    6: {
        "name": "Target Definition",
        "config_class": "TargetConfig",
        "description": "Define prediction target variable",
        "key_parameters": ["target_type", "target_definition", "use_soft_targets"],
    },
    7: {
        "name": "Feature Selection",
        "config_class": "FeatureSelectionConfig",
        "description": "Select most relevant features",
        "key_parameters": ["method", "n_features", "correlation_threshold"],
    },
    8: {
        "name": "Dimensionality Reduction",
        "config_class": "DimReductionConfig",
        "description": "Reduce feature space dimensionality",
        "key_parameters": ["method", "n_components"],
    },
    9: {
        "name": "Model Architecture",
        "config_class": "ModelConfig",
        "description": "Select and configure ML model",
        "key_parameters": ["model_type", "hyperparameters"],
    },
    10: {
        "name": "Temporal Cascade",
        "config_class": "CascadeConfig",
        "description": "Configure temporal cascade structure",
        "key_parameters": ["cascade_type", "temporal_slices", "attention_type"],
    },
    11: {
        "name": "Sample Weighting",
        "config_class": "SampleWeightConfig",
        "description": "Weight training samples",
        "key_parameters": ["method", "time_decay", "volatility_weighting"],
    },
    12: {
        "name": "Training Procedure",
        "config_class": "TrainingConfig",
        "description": "Configure training and cross-validation",
        "key_parameters": ["cv_method", "cv_folds", "early_stopping", "use_optuna"],
    },
    13: {
        "name": "Evaluation",
        "config_class": "EvaluationConfig",
        "description": "Configure model evaluation metrics",
        "key_parameters": ["scoring_metric", "compute_trading_metrics"],
    },
}


def print_pipeline_steps():
    """Print all pipeline steps and their configurations."""
    print("=" * 70)
    print("GIGA TRADER ML PIPELINE STEPS")
    print("=" * 70)

    for step_num, info in PIPELINE_STEPS.items():
        print(f"\nStep {step_num}: {info['name']}")
        print(f"  Config Class: {info['config_class']}")
        print(f"  Description: {info['description']}")
        print(f"  Key Parameters: {', '.join(info['key_parameters'])}")


# =============================================================================
# MAIN - Testing
# =============================================================================

if __name__ == "__main__":
    # Print available options
    print_all_options()
    print()

    # Print pipeline steps
    print_pipeline_steps()
    print()

    # Test grid generators
    print("\n" + "=" * 70)
    print("GRID SEARCH CONFIGURATIONS")
    print("=" * 70)

    # Minimal grid
    minimal = GridSearchConfigGenerator.create_minimal_grid()
    print(f"\nMinimal Grid: {minimal.count_combinations()} combinations")

    # Standard grid
    standard = GridSearchConfigGenerator.create_standard_grid()
    print(f"Standard Grid: {standard.count_combinations():,} combinations")

    # Full grid
    full = GridSearchConfigGenerator.create_full_grid()
    print(f"Full Grid: {full.count_combinations():,} combinations")

    # Test generating configs
    print("\n" + "=" * 70)
    print("SAMPLE CONFIGURATIONS")
    print("=" * 70)

    configs = minimal.generate_configs(
        target_type=TargetType.SWING.value,
        max_configs=5,
    )

    for i, config in enumerate(configs):
        print(f"\nConfig {i+1}:")
        print(f"  ID: {config.model_id}")
        print(f"  Model: {config.model_config.model_type}")
        print(f"  Cascade: {config.cascade_config.cascade_type}")
        print(f"  Dim Reduction: {config.dim_reduction_config.method}")

    # Test registry
    print("\n" + "=" * 70)
    print("MODEL REGISTRY TEST")
    print("=" * 70)

    registry = ModelRegistryV2(db=get_registry_db())
    print(f"\nRegistry location: {registry.registry_dir}")
    print(f"Current models: {len(registry.models)}")

    # Register a test model
    test_entry = create_model_entry(
        target_type=TargetType.SWING.value,
        model_type=ModelType.GRADIENT_BOOSTING.value,
        cascade_type=CascadeType.ATTENTION.value,
    )
    test_entry.metrics.cv_auc = 0.75
    test_entry.status = ModelStatus.TRAINED.value

    model_id = registry.register(test_entry)
    print(f"Registered model: {model_id}")

    print(registry.summary())
