"""
GIGA TRADER - Comprehensive Experiment Configuration Schema
============================================================
Defines the complete JSON schema for experiment configurations.
Every experiment is fully specified by this config, enabling:
  - Exact reproducibility
  - Dashboard visualization
  - Config-based model recreation

Usage:
    from src.experiment_config import (
        ExperimentConfig,
        create_default_config,
        create_experiment_variant,
        validate_config,
    )
"""

import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import copy


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT CONFIGURATION SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class DataConfig:
    """Data source and preprocessing configuration."""
    # Data sources
    symbol: str = "SPY"
    years_to_download: int = 5
    chunk_days: int = 30

    # Quality thresholds
    min_bars_per_day: int = 200
    min_premarket_bars: int = 10
    min_afterhours_bars: int = 10

    # Missing data handling
    fill_missing_bars: bool = True
    max_gap_minutes: int = 15
    flag_incomplete_extended: bool = True


@dataclass
class FeatureEngineeringConfig:
    """Feature engineering configuration."""
    # Swing threshold for target labeling
    swing_threshold: float = 0.0025
    swing_thresholds_to_test: List[float] = field(
        default_factory=lambda: [0.002, 0.0025, 0.003, 0.0035, 0.004]
    )

    # Premarket features (EDGE 2)
    use_premarket_features: bool = True
    premarket_time_points: List[str] = field(
        default_factory=lambda: ["08:00", "08:30", "09:00", "09:15", "09:25"]
    )

    # Afterhours features (EDGE 2)
    use_afterhours_features: bool = True
    afterhours_lag_days: int = 5
    afterhours_time_points: List[str] = field(
        default_factory=lambda: ["16:30", "17:00", "18:00", "19:00"]
    )

    # Intraday features (EDGE 3)
    use_intraday_features: bool = True
    intraday_check_times: List[str] = field(
        default_factory=lambda: ["09:45", "10:00", "10:30", "11:00", "12:00",
                                  "13:00", "14:00", "14:30", "15:00", "15:30"]
    )

    # Technical indicators
    use_rsi: bool = True
    use_macd: bool = True
    use_bollinger: bool = True
    use_stochastic: bool = True
    use_atr: bool = True
    use_volume_profile: bool = True
    use_momentum: bool = True

    # Pattern recognition
    use_pattern_recognition: bool = True
    pattern_types: List[str] = field(
        default_factory=lambda: ["gap_up", "gap_down", "inside_day", "outside_day",
                                  "higher_high", "lower_low", "volume_surge",
                                  "momentum_divergence", "trend_continuation"]
    )

    # Feature interactions
    use_feature_interactions: bool = True
    interaction_depth: int = 2  # Pairwise interactions

    # Rolling window features
    rolling_windows: List[int] = field(
        default_factory=lambda: [5, 10, 20, 50]
    )


@dataclass
class DimensionalityReductionConfig:
    """Dimensionality reduction configuration."""
    # Method: "pca", "kernel_pca", "ica", "umap", "mutual_info",
    #         "agglomeration", "kmedoids", "ensemble", "ensemble_plus"
    method: str = "ensemble_plus"

    # Pre-filtering
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.95

    # Target output dimensions
    target_dimensions: int = 50

    # PCA params
    pca_n_components: int = 30

    # UMAP params
    umap_n_components: int = 20
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "euclidean"

    # Kernel PCA params
    kpca_n_components: int = 25
    kpca_kernel: str = "rbf"
    kpca_gamma: float = 0.01

    # ICA params
    ica_n_components: int = 20
    ica_max_iter: int = 500

    # Mutual Information params
    mi_n_features: int = 30
    mi_n_neighbors: int = 5

    # Feature Agglomeration params
    agglom_n_clusters: int = 25

    # K-Medoids params
    kmedoids_n_clusters: int = 20
    kmedoids_metric: str = "euclidean"
    kmedoids_max_iter: int = 300


@dataclass
class ModelConfig:
    """Model selection and training configuration."""
    # Model type: "logistic", "gradient_boosting", "ensemble", "stacking", "diverse_ensemble"
    # "diverse_ensemble" uses multiple models with DIFFERENT regularization strengths
    # to reduce overfitting (models that disagree on noise but agree on signal)
    model_type: str = "ensemble"

    # Regularization (EDGE 1) - AGGRESSIVE to reduce overfitting
    regularization: str = "l2"  # "l1", "l2", "elastic_net"
    l2_C: float = 0.1  # Lower C = stronger regularization (was 1.0)
    l1_alpha: float = 0.01
    elastic_l1_ratio: float = 0.7  # Higher L1 ratio for more sparsity (was 0.5)

    # Gradient Boosting params - CONSERVATIVE to reduce overfitting
    gb_n_estimators: int = 75  # Fewer trees (was 100)
    gb_max_depth: int = 3  # Shallower trees (was 4)
    gb_learning_rate: float = 0.08  # Lower LR (was 0.1)
    gb_min_samples_leaf: int = 75  # Higher min leaf (was 50)
    gb_subsample: float = 0.75  # More aggressive subsampling (was 0.8)

    # Random Forest params
    rf_n_estimators: int = 100
    rf_max_depth: int = 5
    rf_min_samples_leaf: int = 50

    # Ensemble configuration
    ensemble_models: List[str] = field(
        default_factory=lambda: ["logistic_l2", "gradient_boosting"]
    )
    ensemble_voting: str = "soft"  # "soft", "hard"
    ensemble_weights: Optional[List[float]] = None


@dataclass
class CrossValidationConfig:
    """Cross-validation and evaluation configuration."""
    # CV settings
    n_cv_folds: int = 5
    purge_days: int = 5
    embargo_days: int = 2

    # Soft targets (EDGE 4)
    use_soft_targets: bool = True
    soft_target_k: int = 50
    label_smoothing_epsilon: float = 0.1

    # Scoring
    primary_metric: str = "roc_auc"
    secondary_metrics: List[str] = field(
        default_factory=lambda: ["accuracy", "precision", "recall", "f1"]
    )


@dataclass
class HyperparameterOptimizationConfig:
    """Hyperparameter optimization configuration."""
    # Optuna settings
    use_optuna: bool = True
    optuna_n_trials: int = 50
    optuna_timeout: int = 300
    optuna_sampler: str = "tpe"  # "tpe", "cmaes", "random"

    # Search spaces (min, max tuples)
    # AGGRESSIVE REGULARIZATION: Smaller C ranges, shallower trees
    hp_search_space: Dict[str, Tuple[float, float]] = field(
        default_factory=lambda: {
            "swing_threshold": (0.001, 0.006),
            "l2_C": (0.001, 0.5),  # Much stronger regularization (was 0.01-10.0)
            "gb_n_estimators": (30, 100),  # Fewer trees (was 30-150)
            "gb_max_depth": (2, 4),  # Shallower, max 4 (was 2-5)
            "gb_learning_rate": (0.03, 0.15),  # Lower LR (was 0.01-0.3)
            "gb_min_samples_leaf": (50, 150),  # Larger leaves (was 20-100)
            "gb_subsample": (0.6, 0.85),  # More aggressive (was 0.6-1.0)
        }
    )


@dataclass
class AntiOverfitConfig:
    """Anti-overfitting measures configuration."""
    # Master switch
    use_anti_overfit: bool = True

    # Synthetic universes ("what SPY could have been")
    use_synthetic_universes: bool = True
    n_synthetic_universes: int = 10
    synthetic_weight: float = 0.3  # Weight for synthetic samples

    # SPY-minus-component modifiers
    use_spy_minus_component: bool = True
    component_tickers: List[str] = field(
        default_factory=lambda: ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA"]
    )

    # Cross-asset features
    use_cross_assets: bool = True
    cross_asset_tickers: List[str] = field(
        default_factory=lambda: ["TLT", "GLD", "QQQ", "IWM", "EEM", "HYG", "VXX"]
    )

    # Breadth features
    use_breadth_streaks: bool = True
    n_breadth_components: int = 50

    # MAG7/MAG10 features
    use_mag_breadth: bool = True
    mag_tickers: List[str] = field(
        default_factory=lambda: ["AAPL", "MSFT", "GOOGL", "AMZN", "META",
                                  "NVDA", "TSLA", "BRK-B", "UNH", "XOM"]
    )

    # Evaluation thresholds
    wmes_threshold: float = 0.55
    stability_threshold: float = 0.5


@dataclass
class RobustnessEnsembleConfig:
    """Robustness ensemble configuration."""
    use_robustness_ensemble: bool = True
    n_dimension_variants: int = 2  # +/- N dimensions
    n_param_variants: int = 2
    param_noise_pct: float = 0.05
    ensemble_center_weight: float = 0.5
    fragility_threshold: float = 0.35


@dataclass
class EntryExitConfig:
    """Entry/exit timing model configuration."""
    train_entry_exit_model: bool = True
    model_type: str = "gradient_boosting"  # "gradient_boosting", "random_forest", "ridge"
    entry_window: Tuple[int, int] = (0, 120)  # Minutes from open
    exit_window: Tuple[int, int] = (180, 385)
    min_position_pct: float = 0.05
    max_position_pct: float = 0.25


@dataclass
class TradingConfig:
    """Trading strategy configuration."""
    entry_threshold: float = 0.6
    exit_threshold: float = 0.4
    stop_loss_pct: float = 0.01
    take_profit_pct: float = 0.02
    max_daily_trades: int = 5
    position_sizing: str = "fixed"  # "fixed", "kelly", "volatility_scaled"


@dataclass
class ExperimentConfig:
    """
    Complete experiment configuration.

    This config fully specifies an experiment and can be used to:
      - Reproduce the exact same model
      - Compare experiments
      - Visualize in dashboard
    """
    # Metadata
    experiment_id: str = ""
    experiment_name: str = ""
    experiment_type: str = "full"  # "full", "hyperparameter", "feature_subset", etc.
    description: str = ""
    created_at: str = ""

    # All configuration sections
    data: DataConfig = field(default_factory=DataConfig)
    feature_engineering: FeatureEngineeringConfig = field(default_factory=FeatureEngineeringConfig)
    dim_reduction: DimensionalityReductionConfig = field(default_factory=DimensionalityReductionConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    cross_validation: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    hp_optimization: HyperparameterOptimizationConfig = field(default_factory=HyperparameterOptimizationConfig)
    anti_overfit: AntiOverfitConfig = field(default_factory=AntiOverfitConfig)
    robustness_ensemble: RobustnessEnsembleConfig = field(default_factory=RobustnessEnsembleConfig)
    entry_exit: EntryExitConfig = field(default_factory=EntryExitConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)

    def __post_init__(self):
        if not self.experiment_id:
            self.experiment_id = self._generate_id()
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def _generate_id(self) -> str:
        """Generate unique experiment ID from config hash."""
        config_str = json.dumps(self.to_dict(), sort_keys=True, default=str)
        hash_val = hashlib.md5(config_str.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"exp_{timestamp}_{hash_val}"

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent, default=str)

    def save(self, path: Path):
        """Save config to JSON file."""
        with open(path, 'w') as f:
            f.write(self.to_json())

    @classmethod
    def from_dict(cls, d: Dict) -> "ExperimentConfig":
        """Create from dictionary."""
        # Convert nested dicts to dataclasses
        config = cls(
            experiment_id=d.get("experiment_id", ""),
            experiment_name=d.get("experiment_name", ""),
            experiment_type=d.get("experiment_type", "full"),
            description=d.get("description", ""),
            created_at=d.get("created_at", ""),
            data=DataConfig(**d.get("data", {})),
            feature_engineering=FeatureEngineeringConfig(**d.get("feature_engineering", {})),
            dim_reduction=DimensionalityReductionConfig(**d.get("dim_reduction", {})),
            model=ModelConfig(**d.get("model", {})),
            cross_validation=CrossValidationConfig(**d.get("cross_validation", {})),
            hp_optimization=HyperparameterOptimizationConfig(**d.get("hp_optimization", {})),
            anti_overfit=AntiOverfitConfig(**d.get("anti_overfit", {})),
            robustness_ensemble=RobustnessEnsembleConfig(**d.get("robustness_ensemble", {})),
            entry_exit=EntryExitConfig(**d.get("entry_exit", {})),
            trading=TradingConfig(**d.get("trading", {})),
        )
        return config

    @classmethod
    def from_json(cls, json_str: str) -> "ExperimentConfig":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    @classmethod
    def load(cls, path: Path) -> "ExperimentConfig":
        """Load config from JSON file."""
        with open(path) as f:
            return cls.from_json(f.read())


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════
def create_default_config(name: str = "default") -> ExperimentConfig:
    """Create default experiment configuration."""
    return ExperimentConfig(
        experiment_name=name,
        description="Default full pipeline configuration"
    )


def create_experiment_variant(
    base_config: ExperimentConfig,
    variant_type: str,
    **overrides
) -> ExperimentConfig:
    """
    Create a variant of an experiment configuration.

    Variant types:
      - "hyperparameter": Vary model hyperparameters
      - "feature_subset": Vary feature selection
      - "dim_reduction": Vary dimensionality reduction method
      - "regularization": Vary regularization strength
      - "ensemble": Different ensemble configurations
      - "threshold": Vary trading thresholds
    """
    config = copy.deepcopy(base_config)
    config.experiment_id = ""  # Will be regenerated
    config.experiment_type = variant_type

    if variant_type == "hyperparameter":
        # Vary model hyperparameters
        if "l2_C" in overrides:
            config.model.l2_C = overrides["l2_C"]
        if "gb_n_estimators" in overrides:
            config.model.gb_n_estimators = overrides["gb_n_estimators"]
        if "gb_max_depth" in overrides:
            config.model.gb_max_depth = overrides["gb_max_depth"]
        if "gb_learning_rate" in overrides:
            config.model.gb_learning_rate = overrides["gb_learning_rate"]

    elif variant_type == "feature_subset":
        # Vary feature selection
        if "use_premarket_features" in overrides:
            config.feature_engineering.use_premarket_features = overrides["use_premarket_features"]
        if "use_afterhours_features" in overrides:
            config.feature_engineering.use_afterhours_features = overrides["use_afterhours_features"]
        if "use_pattern_recognition" in overrides:
            config.feature_engineering.use_pattern_recognition = overrides["use_pattern_recognition"]
        if "use_feature_interactions" in overrides:
            config.feature_engineering.use_feature_interactions = overrides["use_feature_interactions"]

    elif variant_type == "dim_reduction":
        # Vary dimensionality reduction
        if "method" in overrides:
            config.dim_reduction.method = overrides["method"]
        if "target_dimensions" in overrides:
            config.dim_reduction.target_dimensions = overrides["target_dimensions"]

    elif variant_type == "regularization":
        # Vary regularization
        if "regularization" in overrides:
            config.model.regularization = overrides["regularization"]
        if "l2_C" in overrides:
            config.model.l2_C = overrides["l2_C"]
        if "l1_alpha" in overrides:
            config.model.l1_alpha = overrides["l1_alpha"]

    elif variant_type == "ensemble":
        # Vary ensemble configuration
        if "model_type" in overrides:
            config.model.model_type = overrides["model_type"]
        if "ensemble_models" in overrides:
            config.model.ensemble_models = overrides["ensemble_models"]
        if "ensemble_voting" in overrides:
            config.model.ensemble_voting = overrides["ensemble_voting"]

    elif variant_type == "threshold":
        # Vary trading thresholds
        if "entry_threshold" in overrides:
            config.trading.entry_threshold = overrides["entry_threshold"]
        if "exit_threshold" in overrides:
            config.trading.exit_threshold = overrides["exit_threshold"]
        if "stop_loss_pct" in overrides:
            config.trading.stop_loss_pct = overrides["stop_loss_pct"]

    # Apply any remaining overrides
    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    # Generate new ID
    config.experiment_id = config._generate_id()

    return config


def validate_config(config: ExperimentConfig) -> Tuple[bool, List[str]]:
    """
    Validate experiment configuration.

    Returns (is_valid, list_of_errors).
    """
    errors = []

    # Data validation
    if config.data.years_to_download < 1:
        errors.append("years_to_download must be >= 1")
    if config.data.min_bars_per_day < 100:
        errors.append("min_bars_per_day should be >= 100")

    # Feature engineering validation
    if config.feature_engineering.swing_threshold <= 0:
        errors.append("swing_threshold must be > 0")
    if config.feature_engineering.swing_threshold > 0.02:
        errors.append("swing_threshold > 0.02 is unusually high")

    # Dimensionality reduction validation
    valid_methods = ["pca", "kernel_pca", "ica", "umap", "mutual_info",
                     "agglomeration", "kmedoids", "ensemble", "ensemble_plus"]
    if config.dim_reduction.method not in valid_methods:
        errors.append(f"dim_reduction.method must be one of {valid_methods}")

    # Model validation
    if config.model.gb_max_depth > 10:
        errors.append("gb_max_depth > 10 risks overfitting")
    if config.model.l2_C <= 0:
        errors.append("l2_C must be > 0")

    # Cross-validation validation
    if config.cross_validation.n_cv_folds < 3:
        errors.append("n_cv_folds should be >= 3")
    if config.cross_validation.purge_days < 1:
        errors.append("purge_days should be >= 1")

    # Trading validation
    if config.trading.entry_threshold <= 0.5:
        errors.append("entry_threshold <= 0.5 may generate too many signals")
    if config.trading.stop_loss_pct > 0.05:
        errors.append("stop_loss_pct > 5% is very high")

    return len(errors) == 0, errors


def convert_legacy_config(old_config: Dict) -> ExperimentConfig:
    """Convert old CONFIG dict format to new ExperimentConfig."""
    config = ExperimentConfig()

    # Map old keys to new structure
    if "years_to_download" in old_config:
        config.data.years_to_download = old_config["years_to_download"]
    if "swing_threshold" in old_config:
        config.feature_engineering.swing_threshold = old_config["swing_threshold"]
    if "dim_reduction_method" in old_config:
        config.dim_reduction.method = old_config["dim_reduction_method"]
    if "l2_C" in old_config:
        config.model.l2_C = old_config["l2_C"]
    if "n_cv_folds" in old_config:
        config.cross_validation.n_cv_folds = old_config["n_cv_folds"]
    if "use_anti_overfit" in old_config:
        config.anti_overfit.use_anti_overfit = old_config["use_anti_overfit"]
    if "use_synthetic_universes" in old_config:
        config.anti_overfit.use_synthetic_universes = old_config["use_synthetic_universes"]
    if "synthetic_weight" in old_config:
        config.anti_overfit.synthetic_weight = old_config["synthetic_weight"]
    if "use_optuna" in old_config:
        config.hp_optimization.use_optuna = old_config["use_optuna"]

    return config


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG TO TRAIN_ROBUST_MODEL FORMAT
# ═══════════════════════════════════════════════════════════════════════════════
def to_training_config(config: ExperimentConfig) -> Dict:
    """
    Convert ExperimentConfig to the CONFIG dict format used by train_robust_model.py.

    This enables using the EXACT same training pipeline with experiment configs.
    """
    return {
        # Data settings
        "years_to_download": config.data.years_to_download,
        "chunk_days": config.data.chunk_days,
        "min_bars_per_day": config.data.min_bars_per_day,
        "min_premarket_bars": config.data.min_premarket_bars,
        "min_afterhours_bars": config.data.min_afterhours_bars,
        "fill_missing_bars": config.data.fill_missing_bars,
        "max_gap_minutes": config.data.max_gap_minutes,
        "flag_incomplete_extended": config.data.flag_incomplete_extended,

        # Feature engineering
        "swing_threshold": config.feature_engineering.swing_threshold,
        "swing_thresholds_to_test": config.feature_engineering.swing_thresholds_to_test,

        # Dimensionality reduction
        "dim_reduction_method": config.dim_reduction.method,
        "variance_threshold": config.dim_reduction.variance_threshold,
        "correlation_threshold": config.dim_reduction.correlation_threshold,
        "umap_n_components": config.dim_reduction.umap_n_components,
        "umap_n_neighbors": config.dim_reduction.umap_n_neighbors,
        "umap_min_dist": config.dim_reduction.umap_min_dist,
        "umap_metric": config.dim_reduction.umap_metric,
        "kpca_n_components": config.dim_reduction.kpca_n_components,
        "kpca_kernel": config.dim_reduction.kpca_kernel,
        "kpca_gamma": config.dim_reduction.kpca_gamma,
        "ica_n_components": config.dim_reduction.ica_n_components,
        "ica_max_iter": config.dim_reduction.ica_max_iter,
        "mi_n_features": config.dim_reduction.mi_n_features,
        "mi_n_neighbors": config.dim_reduction.mi_n_neighbors,
        "agglom_n_clusters": config.dim_reduction.agglom_n_clusters,
        "kmedoids_n_clusters": config.dim_reduction.kmedoids_n_clusters,
        "kmedoids_metric": config.dim_reduction.kmedoids_metric,
        "kmedoids_max_iter": config.dim_reduction.kmedoids_max_iter,

        # Hyperparameter optimization
        "use_optuna": config.hp_optimization.use_optuna,
        "optuna_n_trials": config.hp_optimization.optuna_n_trials,
        "optuna_timeout": config.hp_optimization.optuna_timeout,
        "optuna_sampler": config.hp_optimization.optuna_sampler,
        "hp_search_space": config.hp_optimization.hp_search_space,

        # Cross-validation
        "n_cv_folds": config.cross_validation.n_cv_folds,
        "purge_days": config.cross_validation.purge_days,
        "embargo_days": config.cross_validation.embargo_days,

        # Soft targets
        "soft_target_k": config.cross_validation.soft_target_k,
        "label_smoothing_epsilon": config.cross_validation.label_smoothing_epsilon,

        # Model constraints
        "max_tree_depth": config.model.gb_max_depth,
        "min_samples_leaf": config.model.gb_min_samples_leaf,
        "l2_C": config.model.l2_C,
        "gb_n_estimators": config.model.gb_n_estimators,
        "gb_max_depth": config.model.gb_max_depth,
        "gb_learning_rate": config.model.gb_learning_rate,
        "gb_min_samples_leaf": config.model.gb_min_samples_leaf,
        "gb_subsample": config.model.gb_subsample,

        # Anti-overfitting
        "use_anti_overfit": config.anti_overfit.use_anti_overfit,
        "use_synthetic_universes": config.anti_overfit.use_synthetic_universes,
        "use_cross_assets": config.anti_overfit.use_cross_assets,
        "use_breadth_streaks": config.anti_overfit.use_breadth_streaks,
        "use_mag_breadth": config.anti_overfit.use_mag_breadth,
        "synthetic_weight": config.anti_overfit.synthetic_weight,
        "wmes_threshold": config.anti_overfit.wmes_threshold,
        "stability_threshold": config.anti_overfit.stability_threshold,

        # Robustness ensemble
        "use_robustness_ensemble": config.robustness_ensemble.use_robustness_ensemble,
        "n_dimension_variants": config.robustness_ensemble.n_dimension_variants,
        "n_param_variants": config.robustness_ensemble.n_param_variants,
        "param_noise_pct": config.robustness_ensemble.param_noise_pct,
        "ensemble_center_weight": config.robustness_ensemble.ensemble_center_weight,
        "fragility_threshold": config.robustness_ensemble.fragility_threshold,

        # Entry/exit timing
        "train_entry_exit_model": config.entry_exit.train_entry_exit_model,
        "entry_exit_model_type": config.entry_exit.model_type,
        "entry_window": config.entry_exit.entry_window,
        "exit_window": config.entry_exit.exit_window,
        "min_position_pct": config.entry_exit.min_position_pct,
        "max_position_pct": config.entry_exit.max_position_pct,

        # Trading
        "entry_threshold": config.trading.entry_threshold,
        "exit_threshold": config.trading.exit_threshold,
        "stop_loss_pct": config.trading.stop_loss_pct,
        "take_profit_pct": config.trading.take_profit_pct,
    }


if __name__ == "__main__":
    # Demo: Create and display default config
    config = create_default_config("demo")
    print("=" * 70)
    print("GIGA TRADER - Experiment Configuration Schema")
    print("=" * 70)
    print(f"\nExperiment ID: {config.experiment_id}")
    print(f"Created: {config.created_at}")
    print("\nConfiguration sections:")
    print(f"  - Data: {config.data.symbol}, {config.data.years_to_download} years")
    print(f"  - Features: swing_threshold={config.feature_engineering.swing_threshold}")
    print(f"  - Dim Reduction: {config.dim_reduction.method}")
    print(f"  - Model: {config.model.model_type}")
    print(f"  - Anti-Overfit: {config.anti_overfit.use_anti_overfit}")

    # Validate
    is_valid, errors = validate_config(config)
    print(f"\nValidation: {'PASS' if is_valid else 'FAIL'}")
    if errors:
        for err in errors:
            print(f"  - {err}")

    # Show JSON preview
    print("\nJSON preview (first 500 chars):")
    print(config.to_json()[:500] + "...")
