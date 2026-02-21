"""
GIGA TRADER - Model Registry v2: Configuration Dataclasses
============================================================
All config dataclasses for pipeline configuration.
"""

import json
import hashlib
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, field, asdict

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


# =============================================================================
# CONFIGURATION DATACLASSES - COMPREHENSIVE PIPELINE STEPS
# =============================================================================

@dataclass
class DataConfig:
    """Data loading and preparation configuration."""
    # Source configuration
    source: str = DataSource.ALPACA.value
    fallback_source: Optional[str] = DataSource.YFINANCE.value
    symbol: str = "SPY"
    additional_symbols: List[str] = field(default_factory=list)  # For cross-asset features

    # Time period
    period: str = DataPeriod.YEARS_10.value
    years: float = 10.0
    start_date: Optional[str] = None
    end_date: Optional[str] = None

    # Resolution
    primary_resolution: str = TimeResolution.MINUTE_1.value
    additional_resolutions: List[str] = field(default_factory=lambda: [
        TimeResolution.MINUTE_5.value,
        TimeResolution.MINUTE_15.value,
        TimeResolution.HOUR_1.value,
        TimeResolution.DAILY.value
    ])

    # Market hours
    market_hours: str = MarketHours.ALL_HOURS.value
    include_extended_hours: bool = True
    include_premarket: bool = True
    include_afterhours: bool = True

    # Train/test/validation split
    train_ratio: float = 0.7
    validation_ratio: float = 0.15
    test_ratio: float = 0.15
    purge_days: int = 5
    embargo_days: int = 2

    # Data quality
    min_volume_filter: int = 0
    remove_zero_volume: bool = True
    validate_ohlc: bool = True
    fill_gaps: bool = True

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'DataConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SyntheticDataConfig:
    """Synthetic data / "What SPY could have been" configuration."""
    # Enable/disable
    enabled: bool = False

    # Method selection
    methods: List[str] = field(default_factory=lambda: [
        SyntheticDataMethod.FILTER_EXTREMES_10.value,
        SyntheticDataMethod.BOOTSTRAP_70.value,
    ])

    # Weighting
    synthetic_weight: float = 0.3  # Weight of synthetic vs real data
    real_weight: float = 0.7
    method_weights: Dict[str, float] = field(default_factory=dict)  # Per-method weights

    # Component selection for filtering methods
    n_components_filter: int = 50  # How many SPY components to use
    filter_threshold_pct: float = 10.0  # % to filter for extremes

    # Bootstrap parameters
    bootstrap_sample_pct: float = 70.0
    bootstrap_n_iterations: int = 3
    bootstrap_replace: bool = True

    # Sector-based
    excluded_sectors: List[str] = field(default_factory=list)
    sector_rotation_window: int = 20

    # Regime-based
    vix_high_threshold: float = 25.0
    vix_low_threshold: float = 15.0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'SyntheticDataConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    enabled: bool = False

    # Methods
    methods: List[str] = field(default_factory=list)

    # Jitter
    jitter_sigma: float = 0.03

    # Time warp
    time_warp_sigma: float = 0.2
    time_warp_knot: int = 4

    # Magnitude warp
    magnitude_warp_sigma: float = 0.2
    magnitude_warp_knot: int = 4

    # Window operations
    window_slice_ratio: float = 0.9
    window_warp_ratio: float = 0.1

    # Mixup
    mixup_alpha: float = 0.2

    # Oversampling
    oversampling_strategy: str = "auto"
    smote_k_neighbors: int = 5

    # Noise
    noise_std: float = 0.01

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'AugmentationConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class PreprocessConfig:
    """Data preprocessing configuration."""
    # Outlier handling
    outlier_method: str = OutlierMethod.WINSORIZE_1.value
    outlier_threshold: float = 0.01
    outlier_contamination: float = 0.05  # For ML-based methods
    outlier_n_neighbors: int = 20  # For LOF

    # Missing value handling
    missing_method: str = MissingValueMethod.FORWARD_FILL.value
    missing_threshold: float = 0.1  # Drop columns with > 10% missing
    missing_knn_neighbors: int = 5
    missing_max_iter: int = 10  # For iterative imputer

    # Scaling
    scaling_method: str = ScalingMethod.STANDARD.value
    scale_per_feature: bool = True
    scale_with_mean: bool = True
    scale_with_std: bool = True

    # Transformation
    transform_method: str = TransformMethod.NONE.value
    transform_columns: List[str] = field(default_factory=list)  # Empty = all

    # Feature filtering (pre-selection)
    variance_threshold: float = 0.0
    correlation_threshold: float = 0.95
    constant_filter: bool = True
    quasi_constant_threshold: float = 0.99

    # Duplicate handling
    remove_duplicate_features: bool = True
    remove_duplicate_rows: bool = False

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'PreprocessConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class FeatureConfig:
    """Feature engineering configuration."""
    # ===== Feature Groups =====
    # Price-based
    include_price_features: bool = True
    include_returns: bool = True
    include_log_returns: bool = True

    # Volume
    include_volume_features: bool = True
    include_dollar_volume: bool = True
    include_volume_profile: bool = False

    # Momentum
    include_momentum_features: bool = True
    momentum_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])

    # Volatility
    include_volatility_features: bool = True
    volatility_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50])
    include_parkinson_vol: bool = True
    include_garman_klass: bool = True
    include_yang_zhang: bool = True

    # Pattern recognition
    include_pattern_features: bool = True
    include_candlestick_patterns: bool = True
    include_chart_patterns: bool = False

    # Technical indicators
    include_rsi: bool = True
    rsi_periods: List[int] = field(default_factory=lambda: [7, 14, 21])
    include_macd: bool = True
    include_bollinger: bool = True
    bollinger_periods: List[int] = field(default_factory=lambda: [10, 20, 50])
    include_atr: bool = True
    include_adx: bool = True
    include_cci: bool = True
    include_stochastic: bool = True
    include_williams_r: bool = True
    include_obv: bool = True
    include_mfi: bool = True
    include_vwap: bool = True

    # Moving averages
    include_sma: bool = True
    include_ema: bool = True
    include_wma: bool = False
    include_dema: bool = False
    include_tema: bool = False
    ma_periods: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100, 200])

    # Extended hours
    include_extended_hours: bool = True
    include_premarket_features: bool = True
    include_afterhours_features: bool = True
    include_overnight_features: bool = True

    # Intraday
    include_intraday_features: bool = True
    intraday_intervals: List[int] = field(default_factory=lambda: [30, 60, 90, 120, 180, 240])

    # Cross-asset
    include_cross_asset_features: bool = False
    cross_asset_symbols: List[str] = field(default_factory=lambda: ["TLT", "QQQ", "GLD", "IWM", "VXX"])

    # Breadth
    include_breadth_features: bool = False
    breadth_streak_days: List[int] = field(default_factory=lambda: [2, 3, 4, 5, 6, 7])

    # Sentiment
    include_sentiment_features: bool = False

    # ===== Rolling/Lag Configuration =====
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 50, 100])
    lag_periods: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10])

    # ===== Interaction Features =====
    include_interaction_features: bool = False
    interaction_top_n: int = 10
    polynomial_degree: int = 2

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'FeatureConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TargetConfig:
    """Target variable configuration."""
    # Target type
    target_type: str = TargetType.SWING.value
    target_definition: str = TargetDefinition.RETURN_THRESHOLD.value

    # Thresholds
    swing_threshold: float = 0.0025  # 0.25% for swing classification
    timing_threshold: float = 0.0  # For low-before-high

    # Multi-class
    n_classes: int = 2
    class_boundaries: List[float] = field(default_factory=lambda: [-0.005, 0.005])

    # Forward looking
    forward_periods: int = 1  # Days to look forward
    forward_return_type: str = "close_to_close"  # or "open_to_close", "high_to_low"

    # Label smoothing
    use_soft_targets: bool = True
    smoothing_method: str = LabelSmoothingMethod.EPSILON_SMOOTHING.value
    smoothing_epsilon: float = 0.1
    sigmoid_steepness: float = 50.0

    # Triple barrier
    use_triple_barrier: bool = False
    take_profit: float = 0.01
    stop_loss: float = 0.01
    max_holding_period: int = 5

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'TargetConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class FeatureSelectionConfig:
    """Feature selection configuration (before dim reduction)."""
    enabled: bool = True
    method: str = FeatureSelectionMethod.MUTUAL_INFO.value

    # Number of features
    n_features: Optional[int] = None  # None = auto
    n_features_pct: float = 0.5  # Select top 50% if n_features is None

    # Variance threshold
    variance_threshold: float = 0.01

    # Correlation filter
    correlation_threshold: float = 0.95
    correlation_method: str = "pearson"  # or "spearman", "kendall"

    # Mutual information
    mi_n_neighbors: int = 5
    mi_discrete_features: bool = False

    # RFE
    rfe_estimator: str = ModelType.RANDOM_FOREST.value
    rfe_step: float = 0.1
    rfe_cv: int = 3

    # Boruta
    boruta_max_iter: int = 100
    boruta_alpha: float = 0.05

    # Tree importance
    tree_importance_type: str = "impurity"  # or "permutation"
    tree_n_estimators: int = 100

    # SHAP
    shap_max_samples: int = 1000

    # Stability selection
    stability_n_bootstrap: int = 100
    stability_threshold: float = 0.6

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'FeatureSelectionConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class FeatureGroupConfig:
    """Feature group configuration for protection and hierarchical dim reduction."""
    enabled: bool = False
    group_mode: str = FeatureGroupMode.FLAT.value

    # Groups to protect (pass through selection + reduction untouched)
    protected_groups: List[str] = field(default_factory=list)

    # Per-group component budget mode
    budget_mode: str = "proportional"  # "equal", "proportional"

    # Total output components (across all non-protected groups)
    total_components: int = 40

    # Min components per group (so no group gets zero)
    min_components_per_group: int = 2

    # Selection method per group
    per_group_selection_method: str = "mutual_info"

    # Reduction method per group
    per_group_reduction_method: str = "pca"

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'FeatureGroupConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class DimReductionConfig:
    """Dimensionality reduction configuration."""
    enabled: bool = True
    method: str = DimReductionMethod.ENSEMBLE_PLUS.value
    n_components: Optional[int] = 30  # None = auto

    # Auto component selection
    explained_variance_target: float = 0.95  # For PCA-based methods

    # ===== PCA Variants =====
    pca_whiten: bool = False
    pca_svd_solver: str = "auto"  # "auto", "full", "arpack", "randomized"

    sparse_pca_alpha: float = 1.0
    sparse_pca_max_iter: int = 1000

    incremental_pca_batch_size: Optional[int] = None

    truncated_svd_algorithm: str = "randomized"  # "randomized", "arpack"

    # ===== Kernel PCA =====
    kpca_kernel: str = "rbf"  # "linear", "poly", "rbf", "sigmoid", "cosine"
    kpca_gamma: float = 0.01
    kpca_degree: int = 3  # For poly kernel
    kpca_coef0: float = 1.0
    kpca_alpha: int = 1  # Hyperparameter for inverse transform
    kpca_fit_inverse_transform: bool = False

    # ===== UMAP =====
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "euclidean"  # Many options: "manhattan", "cosine", "correlation"
    umap_spread: float = 1.0
    umap_random_state: int = 42
    umap_transform_seed: int = 42

    # ===== t-SNE =====
    tsne_perplexity: float = 30.0
    tsne_learning_rate: float = 200.0
    tsne_n_iter: int = 1000
    tsne_metric: str = "euclidean"
    tsne_init: str = "pca"  # "random", "pca"

    # ===== ICA =====
    ica_algorithm: str = "parallel"  # "parallel", "deflation"
    ica_whiten: str = "unit-variance"
    ica_fun: str = "logcosh"  # "logcosh", "exp", "cube"
    ica_max_iter: int = 500
    ica_tol: float = 0.0001

    # ===== Manifold Learning =====
    isomap_n_neighbors: int = 5
    lle_n_neighbors: int = 5
    lle_method: str = "standard"  # "standard", "hessian", "modified", "ltsa"

    spectral_affinity: str = "nearest_neighbors"  # "nearest_neighbors", "rbf"
    spectral_n_neighbors: int = 10
    spectral_gamma: Optional[float] = None

    # ===== Factor Analysis =====
    fa_max_iter: int = 1000
    fa_tol: float = 0.01
    fa_rotation: Optional[str] = "varimax"  # None, "varimax", "quartimax"

    # ===== NMF =====
    nmf_init: str = "nndsvda"  # "random", "nndsvd", "nndsvda", "nndsvdar"
    nmf_solver: str = "cd"  # "cd", "mu"
    nmf_beta_loss: str = "frobenius"  # "frobenius", "kullback-leibler", "itakura-saito"
    nmf_max_iter: int = 200
    nmf_alpha: float = 0.0  # Regularization
    nmf_l1_ratio: float = 0.0

    # ===== Agglomeration =====
    agglom_linkage: str = "ward"  # "ward", "complete", "average", "single"
    agglom_distance_threshold: Optional[float] = None
    agglom_compute_distances: bool = False

    # ===== K-Medoids =====
    kmedoids_metric: str = "euclidean"  # "manhattan", "cosine"
    kmedoids_init: str = "heuristic"  # "random", "heuristic", "k-medoids++"
    kmedoids_max_iter: int = 300

    # ===== Autoencoder =====
    ae_hidden_layers: List[int] = field(default_factory=lambda: [64, 32, 16])
    ae_activation: str = "relu"
    ae_dropout: float = 0.1
    ae_epochs: int = 100
    ae_batch_size: int = 32
    ae_learning_rate: float = 0.001

    # ===== Ensemble+ =====
    ensemble_mi_features: int = 20
    ensemble_kpca_components: int = 12
    ensemble_ica_components: int = 8
    ensemble_kmedoids_clusters: int = 10
    ensemble_pca_components: int = 10
    ensemble_voting_method: str = "concatenate"  # "concatenate", "average", "weighted"
    ensemble_weights: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'DimReductionConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    model_type: str = ModelType.GRADIENT_BOOSTING.value

    # ===== Logistic Regression =====
    lr_penalty: str = "l2"  # "l1", "l2", "elasticnet", "none"
    lr_C: float = 1.0
    lr_solver: str = "lbfgs"  # "newton-cg", "lbfgs", "liblinear", "sag", "saga"
    lr_max_iter: int = 1000
    lr_tol: float = 0.0001
    lr_l1_ratio: float = 0.5  # For elasticnet

    # ===== Elastic Net =====
    en_alpha: float = 1.0
    en_l1_ratio: float = 0.5
    en_max_iter: int = 1000

    # ===== Ridge =====
    ridge_alpha: float = 1.0
    ridge_solver: str = "auto"

    # ===== SGD Classifier =====
    sgd_loss: str = "log_loss"  # "hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"
    sgd_penalty: str = "l2"
    sgd_alpha: float = 0.0001
    sgd_learning_rate: str = "optimal"  # "constant", "optimal", "invscaling", "adaptive"
    sgd_eta0: float = 0.01
    sgd_max_iter: int = 1000

    # ===== Decision Tree =====
    dt_criterion: str = "gini"  # "gini", "entropy", "log_loss"
    dt_splitter: str = "best"  # "best", "random"
    dt_max_depth: Optional[int] = 5
    dt_min_samples_split: int = 2
    dt_min_samples_leaf: int = 1
    dt_max_features: Optional[str] = "sqrt"  # "sqrt", "log2", None

    # ===== Random Forest =====
    rf_n_estimators: int = 100
    rf_criterion: str = "gini"
    rf_max_depth: Optional[int] = 5
    rf_min_samples_split: int = 2
    rf_min_samples_leaf: int = 20
    rf_max_features: str = "sqrt"
    rf_bootstrap: bool = True
    rf_oob_score: bool = False
    rf_max_samples: Optional[float] = None

    # ===== Extra Trees =====
    et_n_estimators: int = 100
    et_max_depth: Optional[int] = 5
    et_min_samples_leaf: int = 20

    # ===== Gradient Boosting =====
    gb_loss: str = "log_loss"  # "log_loss", "exponential"
    gb_n_estimators: int = 100
    gb_max_depth: int = 3
    gb_learning_rate: float = 0.1
    gb_subsample: float = 0.8
    gb_min_samples_split: int = 2
    gb_min_samples_leaf: int = 20
    gb_max_features: Optional[str] = "sqrt"
    gb_validation_fraction: float = 0.1
    gb_n_iter_no_change: Optional[int] = 10
    gb_tol: float = 0.0001

    # ===== Hist Gradient Boosting =====
    hgb_max_iter: int = 100
    hgb_max_depth: Optional[int] = None
    hgb_learning_rate: float = 0.1
    hgb_max_leaf_nodes: int = 31
    hgb_min_samples_leaf: int = 20
    hgb_l2_regularization: float = 0.0
    hgb_max_bins: int = 255
    hgb_early_stopping: bool = True

    # ===== XGBoost =====
    xgb_n_estimators: int = 100
    xgb_max_depth: int = 3
    xgb_learning_rate: float = 0.1
    xgb_booster: str = "gbtree"  # "gbtree", "gblinear", "dart"
    xgb_tree_method: str = "auto"  # "auto", "exact", "approx", "hist", "gpu_hist"
    xgb_gamma: float = 0.0
    xgb_min_child_weight: int = 1
    xgb_subsample: float = 0.8
    xgb_colsample_bytree: float = 0.8
    xgb_colsample_bylevel: float = 1.0
    xgb_colsample_bynode: float = 1.0
    xgb_reg_alpha: float = 0.0
    xgb_reg_lambda: float = 1.0
    xgb_scale_pos_weight: float = 1.0
    xgb_max_bin: int = 256

    # ===== LightGBM =====
    lgb_n_estimators: int = 100
    lgb_max_depth: int = -1  # -1 = no limit
    lgb_num_leaves: int = 31
    lgb_learning_rate: float = 0.1
    lgb_subsample: float = 0.8
    lgb_colsample_bytree: float = 0.8
    lgb_reg_alpha: float = 0.0
    lgb_reg_lambda: float = 0.0
    lgb_min_child_samples: int = 20
    lgb_min_split_gain: float = 0.0
    lgb_boosting_type: str = "gbdt"  # "gbdt", "dart", "goss", "rf"
    lgb_subsample_freq: int = 0
    lgb_is_unbalance: bool = False

    # ===== CatBoost =====
    cb_iterations: int = 100
    cb_depth: int = 6
    cb_learning_rate: float = 0.03
    cb_l2_leaf_reg: float = 3.0
    cb_bagging_temperature: float = 1.0
    cb_random_strength: float = 1.0
    cb_border_count: int = 254
    cb_bootstrap_type: str = "Bayesian"  # "Bayesian", "Bernoulli", "MVS", "No"

    # ===== SVM =====
    svm_C: float = 1.0
    svm_kernel: str = "rbf"  # "linear", "poly", "rbf", "sigmoid"
    svm_degree: int = 3
    svm_gamma: str = "scale"  # "scale", "auto" or float
    svm_coef0: float = 0.0
    svm_shrinking: bool = True
    svm_probability: bool = True
    svm_max_iter: int = -1  # -1 = no limit

    # ===== Linear SVC =====
    lsvc_C: float = 1.0
    lsvc_loss: str = "squared_hinge"  # "hinge", "squared_hinge"
    lsvc_penalty: str = "l2"
    lsvc_dual: bool = True
    lsvc_max_iter: int = 1000

    # ===== MLP =====
    mlp_hidden_layer_sizes: List[int] = field(default_factory=lambda: [100, 50])
    mlp_activation: str = "relu"  # "identity", "logistic", "tanh", "relu"
    mlp_solver: str = "adam"  # "lbfgs", "sgd", "adam"
    mlp_alpha: float = 0.0001  # L2 penalty
    mlp_batch_size: str = "auto"
    mlp_learning_rate: str = "constant"  # "constant", "invscaling", "adaptive"
    mlp_learning_rate_init: float = 0.001
    mlp_max_iter: int = 200
    mlp_early_stopping: bool = True
    mlp_validation_fraction: float = 0.1
    mlp_beta_1: float = 0.9
    mlp_beta_2: float = 0.999
    mlp_epsilon: float = 1e-8
    mlp_n_iter_no_change: int = 10
    mlp_momentum: float = 0.9
    mlp_nesterovs_momentum: bool = True

    # ===== KNN =====
    knn_n_neighbors: int = 5
    knn_weights: str = "uniform"  # "uniform", "distance"
    knn_algorithm: str = "auto"  # "auto", "ball_tree", "kd_tree", "brute"
    knn_leaf_size: int = 30
    knn_p: int = 2  # Power parameter for Minkowski metric
    knn_metric: str = "minkowski"

    # ===== Naive Bayes =====
    nb_var_smoothing: float = 1e-9  # For GaussianNB
    nb_alpha: float = 1.0  # For MultinomialNB

    # ===== AdaBoost =====
    ada_n_estimators: int = 50
    ada_learning_rate: float = 1.0
    ada_algorithm: str = "SAMME.R"  # "SAMME", "SAMME.R"

    # ===== Bagging =====
    bag_n_estimators: int = 10
    bag_max_samples: float = 1.0
    bag_max_features: float = 1.0
    bag_bootstrap: bool = True
    bag_bootstrap_features: bool = False
    bag_oob_score: bool = False

    # ===== Voting =====
    voting_type: str = "soft"  # "hard", "soft"
    voting_weights: Optional[List[float]] = None
    voting_estimators: List[str] = field(default_factory=lambda: [
        ModelType.LOGISTIC_L2.value,
        ModelType.GRADIENT_BOOSTING.value,
        ModelType.RANDOM_FOREST.value,
    ])

    # ===== Stacking =====
    stacking_final_estimator: str = ModelType.LOGISTIC_L2.value
    stacking_cv: int = 5
    stacking_stack_method: str = "auto"  # "auto", "predict_proba", "decision_function", "predict"
    stacking_passthrough: bool = False

    # ===== Calibration =====
    calibration_method: str = "sigmoid"  # "sigmoid", "isotonic"
    calibration_cv: int = 5

    # ===== General =====
    random_state: int = 42
    class_weight: Optional[str] = "balanced"  # None, "balanced", or dict
    n_jobs: int = -1  # -1 = use all cores

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'ModelConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class CascadeConfig:
    """Temporal cascade configuration."""
    enabled: bool = True
    cascade_type: str = CascadeType.BASE.value

    # ===== Temporal Slices =====
    # Minutes from market open when slices are taken
    temporal_slices: List[int] = field(default_factory=lambda: [0, 30, 60, 90, 120, 180, 240, 300])
    use_market_close_slice: bool = True  # Add 390 min (6.5 hours = market close)

    # ===== Multi-Resolution =====
    resolutions_minutes: List[int] = field(default_factory=lambda: [1, 2, 3, 5, 10, 15, 30])
    resolution_aggregation: str = "concatenate"  # "concatenate", "stack", "hierarchical"

    # ===== Masked Cascade =====
    n_masked_models: int = 5
    mask_probability: float = 0.2
    mask_strategy: str = "random"  # "random", "structured", "learned"
    mask_feature_groups: bool = False  # Mask entire feature groups vs individual features

    # ===== Attention Cascade =====
    attention_type: str = "softmax"  # "softmax", "sigmoid", "linear", "sparsemax"
    attention_temperature: float = 1.0
    attention_heads: int = 1  # For multi-head attention
    attention_dropout: float = 0.1
    attention_normalize: bool = True

    # ===== Cross-Temporal Attention =====
    cross_attention_enabled: bool = False
    cross_attention_window: int = 3  # Look at +/- 3 slices
    cross_attention_bidirectional: bool = True

    # ===== Stochastic Depth =====
    drop_probability: float = 0.2
    drop_schedule: str = "linear"  # "constant", "linear", "exponential"
    drop_mode: str = "batch"  # "batch", "sample"

    # ===== Intermittent Masking =====
    intermittent_mask_prob: float = 0.15
    intermittent_mask_schedule: str = "constant"  # "constant", "random", "learned"

    # ===== Temporal Encoding =====
    temporal_encoding: str = TemporalEncoding.SINUSOIDAL.value
    temporal_encoding_dim: int = 16
    learned_encoding_trainable: bool = True

    # ===== Mixture of Experts =====
    moe_n_experts: int = 4
    moe_top_k: int = 2
    moe_gating: str = "softmax"  # "softmax", "sparsemax", "topk"
    moe_load_balancing: bool = True
    moe_jitter: float = 0.1

    # ===== Hierarchical Cascade =====
    hierarchy_levels: int = 3
    hierarchy_pooling: str = "attention"  # "mean", "max", "attention"

    # ===== Backward Looking =====
    lookback_slices: int = 3
    lookback_weights: str = "exponential"  # "uniform", "linear", "exponential"

    # ===== Bidirectional =====
    bidirectional_merge: str = "concat"  # "concat", "add", "attention"

    # ===== Output Aggregation =====
    slice_aggregation: str = "weighted_average"  # "average", "weighted_average", "attention", "last"
    learn_aggregation_weights: bool = True
    aggregation_regularization: float = 0.01

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'CascadeConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class SampleWeightConfig:
    """Sample weighting configuration."""
    enabled: bool = True
    method: str = SampleWeightMethod.BALANCED.value

    # ===== Class Weighting =====
    balance_classes: bool = True
    class_weight_type: str = "balanced"  # "balanced", "sqrt_balanced", "custom"
    custom_class_weights: Dict[int, float] = field(default_factory=dict)

    # ===== Time Decay =====
    time_decay_enabled: bool = False
    decay_type: str = "exponential"  # "linear", "exponential", "half_life"
    decay_rate: float = 0.01
    half_life_days: int = 252  # 1 year
    recent_weight_multiplier: float = 2.0

    # ===== Volatility Weighting =====
    volatility_weighting_enabled: bool = False
    volatility_inverse: bool = True  # Lower vol = higher weight
    volatility_lookback: int = 20

    # ===== Return Magnitude =====
    return_magnitude_enabled: bool = False
    magnitude_type: str = "absolute"  # "absolute", "squared"
    emphasize_losses: bool = True
    loss_emphasis_factor: float = 1.5

    # ===== Sample Uniqueness =====
    uniqueness_enabled: bool = False
    uniqueness_lookback: int = 5
    uniqueness_threshold: float = 0.5

    # ===== Combination =====
    combination_method: str = "product"  # "product", "sum", "max"
    normalize_weights: bool = True
    clip_min: float = 0.1
    clip_max: float = 10.0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'SampleWeightConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingConfig:
    """Training configuration."""
    # ===== Cross-Validation =====
    cv_method: str = CVMethod.PURGED_KFOLD.value
    cv_folds: int = 5
    cv_shuffle: bool = False  # Usually False for time series
    cv_gap: int = 0  # Gap between train and test

    # Purged CV specific
    purge_days: int = 5
    embargo_pct: float = 0.01

    # Walk-forward
    walk_forward_train_size: int = 252  # 1 year
    walk_forward_test_size: int = 21  # 1 month
    walk_forward_step: int = 21

    # ===== Scoring =====
    scoring_metric: str = ScoringMetric.ROC_AUC.value
    secondary_metrics: List[str] = field(default_factory=lambda: [
        ScoringMetric.PR_AUC.value,
        ScoringMetric.F1.value,
        ScoringMetric.BRIER_SCORE.value,
    ])

    # ===== Early Stopping =====
    early_stopping: bool = True
    early_stopping_rounds: int = 10
    early_stopping_min_delta: float = 0.001
    early_stopping_restore_best: bool = True

    # ===== Optuna Hyperparameter Optimization =====
    use_optuna: bool = False
    optuna_n_trials: int = 50
    optuna_timeout: int = 3600  # 1 hour
    optuna_sampler: str = "tpe"  # "tpe", "random", "grid", "cmaes"
    optuna_pruner: str = "median"  # "median", "hyperband", "successive_halving"
    optuna_n_startup_trials: int = 10
    optuna_n_warmup_steps: int = 5
    optuna_cv_folds: int = 3

    # ===== Robustness Testing =====
    robustness_enabled: bool = True
    robustness_n_tests: int = 10
    robustness_noise_level: float = 0.05
    robustness_perturbation_pct: float = 0.05

    # ===== Ensemble Training =====
    train_ensemble: bool = False
    ensemble_method: str = EnsembleMethod.STACKING_LR.value
    ensemble_n_models: int = 5
    ensemble_diversity_regularization: float = 0.1

    # ===== Batch Processing =====
    batch_size: Optional[int] = None  # None = full batch
    mini_batch_enabled: bool = False
    accumulate_gradients: int = 1

    # ===== Checkpointing =====
    checkpoint_enabled: bool = True
    checkpoint_frequency: int = 10  # Every N trials/epochs
    checkpoint_keep_n: int = 3

    # ===== Reproducibility =====
    seed: int = 42
    deterministic: bool = True

    # ===== Resource Limits =====
    max_training_time: int = 7200  # 2 hours
    max_memory_gb: Optional[float] = None
    n_jobs: int = -1

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'TrainingConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class EvaluationConfig:
    """Model evaluation configuration."""
    # ===== Classification Metrics =====
    compute_roc_auc: bool = True
    compute_pr_auc: bool = True
    compute_confusion_matrix: bool = True
    compute_classification_report: bool = True
    compute_calibration_curve: bool = True

    # ===== Trading Metrics =====
    compute_trading_metrics: bool = True
    compute_win_rate: bool = True
    compute_profit_factor: bool = True
    compute_sharpe_ratio: bool = True
    compute_sortino_ratio: bool = True
    compute_max_drawdown: bool = True
    compute_calmar_ratio: bool = True

    # ===== Robustness Metrics =====
    compute_train_test_gap: bool = True
    train_test_gap_threshold: float = 0.10  # Warn if > 10%
    compute_stability_score: bool = True
    compute_noise_tolerance: bool = True

    # ===== Feature Analysis =====
    compute_feature_importance: bool = True
    importance_method: str = "permutation"  # "permutation", "impurity", "shap"
    compute_shap_values: bool = False
    shap_max_samples: int = 1000

    # ===== Probability Calibration =====
    evaluate_calibration: bool = True
    calibration_n_bins: int = 10
    calibration_strategy: str = "uniform"  # "uniform", "quantile"

    # ===== Threshold Optimization =====
    optimize_threshold: bool = True
    threshold_metric: str = "f1"  # "f1", "precision", "recall", "profit"
    threshold_search_range: List[float] = field(default_factory=lambda: [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7])

    # ===== Statistical Tests =====
    run_permutation_test: bool = True
    permutation_n_iterations: int = 100
    permutation_alpha: float = 0.05

    run_bootstrap_test: bool = True
    bootstrap_n_iterations: int = 1000
    bootstrap_ci: float = 0.95

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'EvaluationConfig':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelMetrics:
    """Model performance metrics - comprehensive tracking."""
    # ===== Classification Metrics =====
    cv_auc: float = 0.0
    cv_auc_std: float = 0.0
    cv_auc_scores: List[float] = field(default_factory=list)  # Per-fold scores
    test_auc: float = 0.0
    train_auc: float = 0.0
    pr_auc: float = 0.0

    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    precision: float = 0.0
    precision_class_0: float = 0.0
    precision_class_1: float = 0.0
    recall: float = 0.0
    recall_class_0: float = 0.0
    recall_class_1: float = 0.0
    f1_score: float = 0.0
    f1_class_0: float = 0.0
    f1_class_1: float = 0.0

    log_loss: float = 0.0
    brier_score: float = 0.0
    matthews_corrcoef: float = 0.0
    cohen_kappa: float = 0.0

    # ===== Confusion Matrix =====
    true_positives: int = 0
    true_negatives: int = 0
    false_positives: int = 0
    false_negatives: int = 0

    # ===== Trading Metrics =====
    win_rate: float = 0.0
    win_rate_long: float = 0.0
    win_rate_short: float = 0.0
    profit_factor: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    avg_win_loss_ratio: float = 0.0
    total_return: float = 0.0
    annualized_return: float = 0.0

    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    omega_ratio: float = 0.0

    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0  # Days
    avg_drawdown: float = 0.0
    ulcer_index: float = 0.0

    n_trades: int = 0
    n_winning_trades: int = 0
    n_losing_trades: int = 0
    avg_trade_duration: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0

    # ===== Robustness Metrics =====
    train_test_gap: float = 0.0
    train_val_gap: float = 0.0
    cv_score_variance: float = 0.0
    stability_score: float = 0.0
    noise_tolerance: float = 0.0
    fragility_score: float = 0.0

    permutation_test_pvalue: float = 1.0
    permutation_test_passed: bool = False
    bootstrap_ci_lower: float = 0.0
    bootstrap_ci_upper: float = 0.0

    # ===== Calibration Metrics =====
    calibration_ece: float = 0.0  # Expected Calibration Error
    calibration_mce: float = 0.0  # Maximum Calibration Error
    calibration_slope: float = 0.0
    calibration_intercept: float = 0.0

    # ===== Cascade-Specific =====
    cascade_agreement: float = 0.0
    temporal_consistency: float = 0.0
    slice_auc_scores: Dict[int, float] = field(default_factory=dict)  # AUC per temporal slice
    attention_entropy: float = 0.0

    # ===== Feature Importance =====
    top_features: List[str] = field(default_factory=list)
    top_feature_importances: List[float] = field(default_factory=list)

    # ===== Threshold Metrics =====
    optimal_threshold: float = 0.5
    precision_at_threshold: float = 0.0
    recall_at_threshold: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'ModelMetrics':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class ModelArtifacts:
    """Paths to model artifacts - complete pipeline components."""
    # ===== Model Files =====
    model_path: Optional[str] = None
    model_checkpoints: List[str] = field(default_factory=list)

    # ===== Preprocessing Artifacts =====
    scaler_path: Optional[str] = None
    transformer_path: Optional[str] = None
    imputer_path: Optional[str] = None
    outlier_detector_path: Optional[str] = None

    # ===== Feature Engineering Artifacts =====
    feature_selector_path: Optional[str] = None
    dim_reducer_path: Optional[str] = None
    feature_pipeline_path: Optional[str] = None

    # ===== Cascade Artifacts =====
    cascade_models_path: Optional[str] = None
    attention_weights_path: Optional[str] = None
    temporal_encodings_path: Optional[str] = None

    # ===== Calibration Artifacts =====
    calibrator_path: Optional[str] = None

    # ===== Feature Information =====
    raw_feature_cols: List[str] = field(default_factory=list)
    input_feature_cols: List[str] = field(default_factory=list)  # After selection
    output_feature_cols: List[str] = field(default_factory=list)  # After dim reduction
    n_raw_features: int = 0
    n_input_features: int = 0
    n_output_features: int = 0

    # ===== Feature Metadata =====
    feature_dtypes: Dict[str, str] = field(default_factory=dict)
    feature_ranges: Dict[str, List[float]] = field(default_factory=dict)  # [min, max]
    feature_means: Dict[str, float] = field(default_factory=dict)
    feature_stds: Dict[str, float] = field(default_factory=dict)

    # ===== Training Data Info =====
    train_date_range: List[str] = field(default_factory=list)  # [start, end]
    test_date_range: List[str] = field(default_factory=list)
    data_hash: str = ""  # Hash of training data for reproducibility

    # ===== Log Files =====
    training_log_path: Optional[str] = None
    optuna_study_path: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> 'ModelArtifacts':
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


# =============================================================================
# MODEL ENTRY - COMPLETE PIPELINE CONFIGURATION
# =============================================================================

@dataclass
class ModelEntry:
    """
    Complete model entry with full pipeline configuration.

    This represents a single model in the grid search, tracking:
    - All pipeline step configurations (data -> features -> model -> cascade)
    - Training results and metrics
    - Artifact paths for reproducibility
    - Lifecycle status

    Grid Search Dimensions Covered:
    1. Data loading (source, period, resolution)
    2. Synthetic data ("What SPY could have been")
    3. Data augmentation (oversampling, noise injection)
    4. Preprocessing (outliers, missing, scaling)
    5. Feature engineering (which features to include)
    6. Target variable definition
    7. Feature selection (before dim reduction)
    8. Dimensionality reduction
    9. Model type and hyperparameters
    10. Cascade/temporal configuration
    11. Sample weighting
    12. Training procedure (CV, early stopping, Optuna)
    13. Evaluation metrics
    """
    # ===== Identification =====
    model_id: str = ""
    experiment_id: str = ""  # Group models from same experiment
    parent_model_id: Optional[str] = None  # For model lineage tracking
    target_type: str = TargetType.SWING.value
    status: str = ModelStatus.QUEUED.value
    version: str = "2.0"

    # ===== Timestamps =====
    created_at: str = ""
    updated_at: str = ""
    queued_at: str = ""
    training_started_at: str = ""
    trained_at: str = ""
    validated_at: str = ""
    promoted_at: str = ""
    deprecated_at: str = ""

    # ===== Pipeline Configuration - All Steps =====
    # Step 1: Data Loading
    data_config: DataConfig = field(default_factory=DataConfig)

    # Step 2: Synthetic Data Generation
    synthetic_config: SyntheticDataConfig = field(default_factory=SyntheticDataConfig)

    # Step 3: Data Augmentation
    augmentation_config: AugmentationConfig = field(default_factory=AugmentationConfig)

    # Step 4: Preprocessing
    preprocess_config: PreprocessConfig = field(default_factory=PreprocessConfig)

    # Step 5: Feature Engineering
    feature_config: FeatureConfig = field(default_factory=FeatureConfig)

    # Step 6: Target Variable Definition
    target_config: TargetConfig = field(default_factory=TargetConfig)

    # Step 7: Feature Selection
    feature_selection_config: FeatureSelectionConfig = field(default_factory=FeatureSelectionConfig)

    # Step 7b: Feature Group Configuration (protection + hierarchical reduction)
    feature_group_config: FeatureGroupConfig = field(default_factory=FeatureGroupConfig)

    # Step 8: Dimensionality Reduction
    dim_reduction_config: DimReductionConfig = field(default_factory=DimReductionConfig)

    # Step 9: Model Architecture
    model_config: ModelConfig = field(default_factory=ModelConfig)

    # Step 10: Temporal Cascade
    cascade_config: CascadeConfig = field(default_factory=CascadeConfig)

    # Step 11: Sample Weighting
    sample_weight_config: SampleWeightConfig = field(default_factory=SampleWeightConfig)

    # Step 12: Training Procedure
    training_config: TrainingConfig = field(default_factory=TrainingConfig)

    # Step 13: Evaluation
    evaluation_config: EvaluationConfig = field(default_factory=EvaluationConfig)

    # ===== Results =====
    metrics: ModelMetrics = field(default_factory=ModelMetrics)
    artifacts: ModelArtifacts = field(default_factory=ModelArtifacts)

    # ===== Training Metadata =====
    training_time_seconds: float = 0.0
    total_pipeline_time_seconds: float = 0.0
    training_samples: int = 0
    validation_samples: int = 0
    test_samples: int = 0
    n_features_raw: int = 0
    n_features_selected: int = 0
    n_features_final: int = 0

    # ===== Grid Search Metadata =====
    grid_search_id: str = ""
    grid_position: int = 0  # Position in grid (for tracking progress)
    grid_total: int = 0
    hyperparameters_hash: str = ""  # For deduplication

    # ===== Resource Usage =====
    peak_memory_mb: float = 0.0
    gpu_used: bool = False
    n_cpus_used: int = 1

    # ===== Tags and Notes =====
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    failure_reason: str = ""

    def __post_init__(self):
        if not self.model_id:
            self.model_id = self._generate_id()
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.experiment_id:
            self.experiment_id = datetime.now().strftime('%Y%m%d')

    def _generate_id(self) -> str:
        """Generate unique model ID from configuration."""
        config_str = json.dumps({
            'target': self.target_type,
            'cascade': self.cascade_config.cascade_type,
            'model': self.model_config.model_type,
            'dim_red': self.dim_reduction_config.method,
            'feature_sel': self.feature_selection_config.method,
        }, sort_keys=True)
        hash_prefix = hashlib.md5(config_str.encode()).hexdigest()[:8]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{self.target_type}_{self.cascade_config.cascade_type}_{timestamp}_{hash_prefix}"

    def get_config_hash(self) -> str:
        """Get hash of full configuration for deduplication."""
        config_dict = {
            'data': self.data_config.to_dict(),
            'synthetic': self.synthetic_config.to_dict(),
            'augmentation': self.augmentation_config.to_dict(),
            'preprocess': self.preprocess_config.to_dict(),
            'feature': self.feature_config.to_dict(),
            'target': self.target_config.to_dict(),
            'feature_selection': self.feature_selection_config.to_dict(),
            'dim_reduction': self.dim_reduction_config.to_dict(),
            'model': self.model_config.to_dict(),
            'cascade': self.cascade_config.to_dict(),
            'sample_weight': self.sample_weight_config.to_dict(),
            'training': self.training_config.to_dict(),
        }
        return hashlib.md5(json.dumps(config_dict, sort_keys=True).encode()).hexdigest()

    def get_pipeline_summary(self) -> str:
        """Get human-readable pipeline summary."""
        return (
            f"Data: {self.data_config.source}/{self.data_config.period} | "
            f"Synthetic: {self.synthetic_config.enabled} | "
            f"Preprocess: {self.preprocess_config.outlier_method}/{self.preprocess_config.scaling_method} | "
            f"Features: {self.n_features_raw}->{self.n_features_selected}->{self.n_features_final} | "
            f"Model: {self.model_config.model_type} | "
            f"Cascade: {self.cascade_config.cascade_type}"
        )

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'model_id': self.model_id,
            'experiment_id': self.experiment_id,
            'parent_model_id': self.parent_model_id,
            'target_type': self.target_type,
            'status': self.status,
            'version': self.version,
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'queued_at': self.queued_at,
            'training_started_at': self.training_started_at,
            'trained_at': self.trained_at,
            'validated_at': self.validated_at,
            'promoted_at': self.promoted_at,
            'deprecated_at': self.deprecated_at,
            # All pipeline configs
            'data_config': self.data_config.to_dict(),
            'synthetic_config': self.synthetic_config.to_dict(),
            'augmentation_config': self.augmentation_config.to_dict(),
            'preprocess_config': self.preprocess_config.to_dict(),
            'feature_config': self.feature_config.to_dict(),
            'target_config': self.target_config.to_dict(),
            'feature_selection_config': self.feature_selection_config.to_dict(),
            'dim_reduction_config': self.dim_reduction_config.to_dict(),
            'model_config': self.model_config.to_dict(),
            'cascade_config': self.cascade_config.to_dict(),
            'sample_weight_config': self.sample_weight_config.to_dict(),
            'training_config': self.training_config.to_dict(),
            'evaluation_config': self.evaluation_config.to_dict(),
            # Results
            'metrics': self.metrics.to_dict(),
            'artifacts': self.artifacts.to_dict(),
            # Metadata
            'training_time_seconds': self.training_time_seconds,
            'total_pipeline_time_seconds': self.total_pipeline_time_seconds,
            'training_samples': self.training_samples,
            'validation_samples': self.validation_samples,
            'test_samples': self.test_samples,
            'n_features_raw': self.n_features_raw,
            'n_features_selected': self.n_features_selected,
            'n_features_final': self.n_features_final,
            'grid_search_id': self.grid_search_id,
            'grid_position': self.grid_position,
            'grid_total': self.grid_total,
            'hyperparameters_hash': self.hyperparameters_hash,
            'peak_memory_mb': self.peak_memory_mb,
            'gpu_used': self.gpu_used,
            'n_cpus_used': self.n_cpus_used,
            'tags': self.tags,
            'notes': self.notes,
            'failure_reason': self.failure_reason,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'ModelEntry':
        """Create from dictionary."""
        return cls(
            model_id=d.get('model_id', ''),
            experiment_id=d.get('experiment_id', ''),
            parent_model_id=d.get('parent_model_id'),
            target_type=d.get('target_type', TargetType.SWING.value),
            status=d.get('status', ModelStatus.QUEUED.value),
            version=d.get('version', '2.0'),
            created_at=d.get('created_at', ''),
            updated_at=d.get('updated_at', ''),
            queued_at=d.get('queued_at', ''),
            training_started_at=d.get('training_started_at', ''),
            trained_at=d.get('trained_at', ''),
            validated_at=d.get('validated_at', ''),
            promoted_at=d.get('promoted_at', ''),
            deprecated_at=d.get('deprecated_at', ''),
            # All pipeline configs
            data_config=DataConfig.from_dict(d.get('data_config', {})),
            synthetic_config=SyntheticDataConfig.from_dict(d.get('synthetic_config', {})),
            augmentation_config=AugmentationConfig.from_dict(d.get('augmentation_config', {})),
            preprocess_config=PreprocessConfig.from_dict(d.get('preprocess_config', {})),
            feature_config=FeatureConfig.from_dict(d.get('feature_config', {})),
            target_config=TargetConfig.from_dict(d.get('target_config', {})),
            feature_selection_config=FeatureSelectionConfig.from_dict(d.get('feature_selection_config', {})),
            dim_reduction_config=DimReductionConfig.from_dict(d.get('dim_reduction_config', {})),
            model_config=ModelConfig.from_dict(d.get('model_config', {})),
            cascade_config=CascadeConfig.from_dict(d.get('cascade_config', {})),
            sample_weight_config=SampleWeightConfig.from_dict(d.get('sample_weight_config', {})),
            training_config=TrainingConfig.from_dict(d.get('training_config', {})),
            evaluation_config=EvaluationConfig.from_dict(d.get('evaluation_config', {})),
            # Results
            metrics=ModelMetrics.from_dict(d.get('metrics', {})),
            artifacts=ModelArtifacts.from_dict(d.get('artifacts', {})),
            # Metadata
            training_time_seconds=d.get('training_time_seconds', 0.0),
            total_pipeline_time_seconds=d.get('total_pipeline_time_seconds', 0.0),
            training_samples=d.get('training_samples', 0),
            validation_samples=d.get('validation_samples', 0),
            test_samples=d.get('test_samples', 0),
            n_features_raw=d.get('n_features_raw', 0),
            n_features_selected=d.get('n_features_selected', 0),
            n_features_final=d.get('n_features_final', 0),
            grid_search_id=d.get('grid_search_id', ''),
            grid_position=d.get('grid_position', 0),
            grid_total=d.get('grid_total', 0),
            hyperparameters_hash=d.get('hyperparameters_hash', ''),
            peak_memory_mb=d.get('peak_memory_mb', 0.0),
            gpu_used=d.get('gpu_used', False),
            n_cpus_used=d.get('n_cpus_used', 1),
            tags=d.get('tags', []),
            notes=d.get('notes', ''),
            failure_reason=d.get('failure_reason', ''),
        )
