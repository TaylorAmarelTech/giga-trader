"""
GIGA TRADER - Model Registry v2: Enum Definitions
====================================================
All enum classes for pipeline configuration options.
"""

from enum import Enum


# -----------------------------------------------------------------------------
# DATA SOURCE & ACQUISITION
# -----------------------------------------------------------------------------

class DataSource(str, Enum):
    """Data source options."""
    ALPACA = "alpaca"
    YFINANCE = "yfinance"
    POLYGON = "polygon"
    CSV = "csv"
    ALPHA_VANTAGE = "alpha_vantage"
    QUANDL = "quandl"
    IEX = "iex"


class TimeResolution(str, Enum):
    """Data time resolution."""
    TICK = "tick"
    SECOND_1 = "1s"
    SECOND_5 = "5s"
    SECOND_15 = "15s"
    SECOND_30 = "30s"
    MINUTE_1 = "1min"
    MINUTE_2 = "2min"
    MINUTE_3 = "3min"
    MINUTE_5 = "5min"
    MINUTE_10 = "10min"
    MINUTE_15 = "15min"
    MINUTE_30 = "30min"
    HOUR_1 = "1h"
    HOUR_4 = "4h"
    DAILY = "1d"
    WEEKLY = "1w"


class DataPeriod(str, Enum):
    """Data lookback period."""
    MONTHS_6 = "6M"
    YEAR_1 = "1Y"
    YEARS_2 = "2Y"
    YEARS_3 = "3Y"
    YEARS_5 = "5Y"
    YEARS_7 = "7Y"
    YEARS_10 = "10Y"
    MAX = "max"


class MarketHours(str, Enum):
    """Market hours inclusion."""
    REGULAR_ONLY = "regular_only"
    EXTENDED_ONLY = "extended_only"
    ALL_HOURS = "all_hours"
    PREMARKET_ONLY = "premarket_only"
    AFTERHOURS_ONLY = "afterhours_only"


# -----------------------------------------------------------------------------
# SYNTHETIC DATA - "WHAT SPY COULD HAVE BEEN"
# -----------------------------------------------------------------------------

class SyntheticDataMethod(str, Enum):
    """Synthetic data generation methods."""
    NONE = "none"
    # Filter-based methods
    FILTER_EXTREMES_5 = "filter_extremes_5"
    FILTER_EXTREMES_10 = "filter_extremes_10"
    FILTER_EXTREMES_20 = "filter_extremes_20"
    FILTER_MIDDLE_10 = "filter_middle_10"
    FILTER_MIDDLE_20 = "filter_middle_20"
    FILTER_MIDDLE_30 = "filter_middle_30"
    # Volatility-based
    FILTER_HIGH_VOL = "filter_high_vol"
    FILTER_LOW_VOL = "filter_low_vol"
    VOLATILITY_QUARTILE_1 = "vol_q1"
    VOLATILITY_QUARTILE_4 = "vol_q4"
    # Momentum-based
    FILTER_WINNERS = "filter_winners"
    FILTER_LOSERS = "filter_losers"
    MOMENTUM_TOP_20 = "momentum_top_20"
    MOMENTUM_BOTTOM_20 = "momentum_bottom_20"
    # Bootstrap methods
    BOOTSTRAP_50 = "bootstrap_50"
    BOOTSTRAP_70 = "bootstrap_70"
    BOOTSTRAP_90 = "bootstrap_90"
    # Weighted combinations
    COMPONENT_WEIGHTED = "component_weighted"
    EQUAL_WEIGHTED = "equal_weighted"
    INVERSE_VOL_WEIGHTED = "inverse_vol_weighted"
    # Sector-based
    SECTOR_ROTATION = "sector_rotation"
    EXCLUDE_TECH = "exclude_tech"
    EXCLUDE_FINANCE = "exclude_finance"
    # Time-based
    RECESSION_PERIODS = "recession_periods"
    EXPANSION_PERIODS = "expansion_periods"
    HIGH_VIX_PERIODS = "high_vix_periods"
    LOW_VIX_PERIODS = "low_vix_periods"


class AugmentationMethod(str, Enum):
    """Data augmentation methods."""
    NONE = "none"
    JITTER = "jitter"
    TIME_WARP = "time_warp"
    MAGNITUDE_WARP = "magnitude_warp"
    WINDOW_SLICE = "window_slice"
    WINDOW_WARP = "window_warp"
    MIXUP = "mixup"
    SMOTE = "smote"
    ADASYN = "adasyn"
    RANDOM_OVERSAMPLING = "random_oversampling"
    NOISE_INJECTION = "noise_injection"


# -----------------------------------------------------------------------------
# OUTLIER & MISSING DATA HANDLING
# -----------------------------------------------------------------------------

class OutlierMethod(str, Enum):
    """Outlier handling methods."""
    NONE = "none"
    # Winsorization
    WINSORIZE_0_5 = "winsorize_0.5"
    WINSORIZE_1 = "winsorize_1"
    WINSORIZE_2_5 = "winsorize_2.5"
    WINSORIZE_5 = "winsorize_5"
    # Clipping
    CLIP_2STD = "clip_2std"
    CLIP_3STD = "clip_3std"
    CLIP_4STD = "clip_4std"
    CLIP_5STD = "clip_5std"
    # Statistical methods
    IQR_1_5 = "iqr_1.5"
    IQR_3 = "iqr_3"
    ZSCORE = "zscore"
    MODIFIED_ZSCORE = "modified_zscore"
    MEDIAN_ABSOLUTE_DEVIATION = "mad"
    # ML-based
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "lof"
    ONE_CLASS_SVM = "one_class_svm"
    ELLIPTIC_ENVELOPE = "elliptic_envelope"
    DBSCAN = "dbscan"


class MissingValueMethod(str, Enum):
    """Missing value handling methods."""
    DROP_ROWS = "drop_rows"
    DROP_COLS = "drop_cols"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"
    LINEAR_INTERPOLATE = "linear_interpolate"
    SPLINE_INTERPOLATE = "spline_interpolate"
    TIME_INTERPOLATE = "time_interpolate"
    MEAN_FILL = "mean_fill"
    MEDIAN_FILL = "median_fill"
    MODE_FILL = "mode_fill"
    KNN_IMPUTE = "knn_impute"
    ITERATIVE_IMPUTE = "iterative_impute"
    ROLLING_MEAN = "rolling_mean"
    ZERO_FILL = "zero_fill"


# -----------------------------------------------------------------------------
# SCALING & TRANSFORMATION
# -----------------------------------------------------------------------------

class ScalingMethod(str, Enum):
    """Feature scaling methods."""
    NONE = "none"
    STANDARD = "standard"
    MINMAX = "minmax"
    MINMAX_NEG1_1 = "minmax_neg1_1"
    ROBUST = "robust"
    ROBUST_IQR = "robust_iqr"
    QUANTILE_UNIFORM = "quantile_uniform"
    QUANTILE_NORMAL = "quantile_normal"
    POWER_YEOBJOHNSON = "power_yeojohnson"
    POWER_BOXCOX = "power_boxcox"
    MAX_ABS = "max_abs"
    L1_NORMALIZE = "l1_normalize"
    L2_NORMALIZE = "l2_normalize"


class TransformMethod(str, Enum):
    """Feature transformation methods."""
    NONE = "none"
    LOG = "log"
    LOG1P = "log1p"
    SQRT = "sqrt"
    CBRT = "cbrt"
    SQUARE = "square"
    RECIPROCAL = "reciprocal"
    BOXCOX = "boxcox"
    YEOJOHNSON = "yeojohnson"
    RANK = "rank"
    PERCENTILE = "percentile"
    SIGMOID = "sigmoid"
    TANH = "tanh"


# -----------------------------------------------------------------------------
# FEATURE SELECTION
# -----------------------------------------------------------------------------

class FeatureSelectionMethod(str, Enum):
    """Feature selection methods (before dim reduction)."""
    NONE = "none"
    # Filter methods
    VARIANCE_THRESHOLD = "variance_threshold"
    CORRELATION_FILTER = "correlation_filter"
    MUTUAL_INFO = "mutual_info"
    CHI2 = "chi2"
    ANOVA_F = "anova_f"
    RELIEF_F = "relief_f"
    # Wrapper methods
    RFE = "rfe"
    RFECV = "rfecv"
    SEQUENTIAL_FORWARD = "sequential_forward"
    SEQUENTIAL_BACKWARD = "sequential_backward"
    BORUTA = "boruta"
    # Embedded methods
    LASSO_PATH = "lasso_path"
    ELASTIC_NET_PATH = "elastic_net_path"
    TREE_IMPORTANCE = "tree_importance"
    PERMUTATION_IMPORTANCE = "permutation_importance"
    SHAP_IMPORTANCE = "shap_importance"
    # Hybrid
    STABILITY_SELECTION = "stability_selection"
    MULTI_SURP = "multi_surp"


class FeatureGroupMode(str, Enum):
    """Feature group processing modes for hierarchical dim reduction."""
    FLAT = "flat"                        # Current behavior - no grouping
    PROTECTED = "protected"              # Protected groups pass through untouched
    GROUPED = "grouped"                  # Per-group dim reduction
    GROUPED_PROTECTED = "grouped_protected"  # Both: protect + per-group reduce


# -----------------------------------------------------------------------------
# DIMENSIONALITY REDUCTION
# -----------------------------------------------------------------------------

class DimReductionMethod(str, Enum):
    """Dimensionality reduction methods."""
    NONE = "none"
    # Linear methods
    PCA = "pca"
    INCREMENTAL_PCA = "incremental_pca"
    SPARSE_PCA = "sparse_pca"
    TRUNCATED_SVD = "truncated_svd"
    FACTOR_ANALYSIS = "factor_analysis"
    NMF = "nmf"  # Non-negative Matrix Factorization
    LDA = "lda"  # Linear Discriminant Analysis
    # Non-linear kernel methods
    KERNEL_PCA_RBF = "kernel_pca_rbf"
    KERNEL_PCA_POLY = "kernel_pca_poly"
    KERNEL_PCA_SIGMOID = "kernel_pca_sigmoid"
    KERNEL_PCA_COSINE = "kernel_pca_cosine"
    # Manifold learning
    UMAP = "umap"
    TSNE = "tsne"
    ISOMAP = "isomap"
    LLE = "lle"  # Locally Linear Embedding
    SPECTRAL_EMBEDDING = "spectral_embedding"
    MDS = "mds"  # Multidimensional Scaling
    # ICA variants
    ICA = "ica"
    FAST_ICA = "fast_ica"
    # Clustering-based
    AGGLOMERATION = "agglomeration"
    KMEDOIDS = "kmedoids"
    KMEANS_FEATURES = "kmeans_features"
    # Autoencoder-based
    AUTOENCODER = "autoencoder"
    VARIATIONAL_AE = "variational_ae"
    # Ensemble methods
    ENSEMBLE_PLUS = "ensemble_plus"
    STACKED_DIM_REDUCTION = "stacked_dim_reduction"
    VOTING_DIM_REDUCTION = "voting_dim_reduction"


# -----------------------------------------------------------------------------
# MODEL TYPES
# -----------------------------------------------------------------------------

class ModelType(str, Enum):
    """Base model types."""
    # Linear models
    LOGISTIC_L1 = "logistic_l1"
    LOGISTIC_L2 = "logistic_l2"
    ELASTIC_NET = "elastic_net"
    RIDGE = "ridge"
    SGD_CLASSIFIER = "sgd_classifier"
    PASSIVE_AGGRESSIVE = "passive_aggressive"
    # Tree-based
    DECISION_TREE = "decision_tree"
    RANDOM_FOREST = "random_forest"
    EXTRA_TREES = "extra_trees"
    GRADIENT_BOOSTING = "gradient_boosting"
    HIST_GRADIENT_BOOSTING = "hist_gradient_boosting"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    CATBOOST = "catboost"
    # Support Vector Machines
    SVM_LINEAR = "svm_linear"
    SVM_RBF = "svm_rbf"
    SVM_POLY = "svm_poly"
    NU_SVC = "nu_svc"
    LINEAR_SVC = "linear_svc"
    # Neural Networks
    MLP = "mlp"
    MLP_DEEP = "mlp_deep"
    MLP_WIDE = "mlp_wide"
    # Naive Bayes
    GAUSSIAN_NB = "gaussian_nb"
    MULTINOMIAL_NB = "multinomial_nb"
    # Neighbors
    KNN = "knn"
    RADIUS_NEIGHBORS = "radius_neighbors"
    # Ensemble meta-learners
    VOTING_HARD = "voting_hard"
    VOTING_SOFT = "voting_soft"
    BAGGING = "bagging"
    ADABOOST = "adaboost"
    STACKING = "stacking"
    # Calibrated models
    CALIBRATED_SIGMOID = "calibrated_sigmoid"
    CALIBRATED_ISOTONIC = "calibrated_isotonic"
    # Quantile models (Wave E3)
    QUANTILE_FOREST = "quantile_forest"
    # Wave F2: Purge-aware stacking ensemble
    STACKING_ENSEMBLE = "stacking_ensemble"


class EnsembleMethod(str, Enum):
    """Meta-ensemble methods for combining models."""
    NONE = "none"
    SIMPLE_AVERAGE = "simple_average"
    WEIGHTED_AVERAGE = "weighted_average"
    VOTING_HARD = "voting_hard"
    VOTING_SOFT = "voting_soft"
    STACKING_LR = "stacking_lr"
    STACKING_GB = "stacking_gb"
    BLENDING = "blending"
    BAYESIAN_AVERAGING = "bayesian_averaging"


# -----------------------------------------------------------------------------
# CASCADE & TEMPORAL CONFIGURATIONS
# -----------------------------------------------------------------------------

class CascadeType(str, Enum):
    """Cascade/ensemble types."""
    BASE = "base"
    MASKED = "masked"
    INTERMITTENT_MASKED = "intermittent_masked"
    ATTENTION = "attention"
    SELF_ATTENTION = "self_attention"
    CROSS_ATTENTION = "cross_attention"
    STOCHASTIC_DEPTH = "stochastic_depth"
    MULTI_RESOLUTION = "multi_resolution"
    BACKWARD_LOOKING = "backward_looking"
    BIDIRECTIONAL = "bidirectional"
    HIERARCHICAL = "hierarchical"
    RECURSIVE = "recursive"
    MIXTURE_OF_EXPERTS = "mixture_of_experts"


class TemporalEncoding(str, Enum):
    """Temporal encoding methods for cascade models."""
    NONE = "none"
    POSITIONAL = "positional"
    SINUSOIDAL = "sinusoidal"
    LEARNED = "learned"
    RELATIVE = "relative"
    ROTARY = "rotary"


# -----------------------------------------------------------------------------
# TARGET VARIABLE GENERATION
# -----------------------------------------------------------------------------

class TargetType(str, Enum):
    """Prediction target types."""
    SWING = "swing"
    TIMING = "timing"
    ENTRY_EXIT = "entry_exit"
    POSITION_SIZE = "position_size"
    DIRECTION = "direction"
    MAGNITUDE = "magnitude"
    VOLATILITY = "volatility"
    REGIME = "regime"


class TargetDefinition(str, Enum):
    """How target variable is defined."""
    # Binary classification
    RETURN_THRESHOLD = "return_threshold"
    HIGH_LOW_TIMING = "high_low_timing"
    CLOSE_VS_OPEN = "close_vs_open"
    NEXT_DAY_DIRECTION = "next_day_direction"
    # Multi-class
    RETURN_BUCKETS_3 = "return_buckets_3"
    RETURN_BUCKETS_5 = "return_buckets_5"
    RETURN_PERCENTILES = "return_percentiles"
    REGIME_CLASSIFICATION = "regime_classification"
    # Regression
    FORWARD_RETURN_1D = "forward_return_1d"
    FORWARD_RETURN_5D = "forward_return_5d"
    MAX_FAVORABLE_EXCURSION = "max_favorable_excursion"
    MAX_ADVERSE_EXCURSION = "max_adverse_excursion"


class LabelSmoothingMethod(str, Enum):
    """Label smoothing / soft target methods."""
    NONE = "none"
    EPSILON_SMOOTHING = "epsilon_smoothing"
    SIGMOID_TRANSFORM = "sigmoid_transform"
    RETURN_PERCENTILE = "return_percentile"
    CONFIDENCE_BASED = "confidence_based"
    TEMPERATURE_SCALING = "temperature_scaling"


# -----------------------------------------------------------------------------
# CROSS-VALIDATION STRATEGIES
# -----------------------------------------------------------------------------

class CVMethod(str, Enum):
    """Cross-validation methods."""
    # Standard
    KFOLD = "kfold"
    STRATIFIED_KFOLD = "stratified_kfold"
    REPEATED_KFOLD = "repeated_kfold"
    LEAVE_ONE_OUT = "leave_one_out"
    # Time-series specific
    TIMESERIES_SPLIT = "timeseries_split"
    BLOCKED_TIMESERIES = "blocked_timeseries"
    PURGED_KFOLD = "purged_kfold"
    COMBINATORIAL_PURGED = "combinatorial_purged"
    WALK_FORWARD = "walk_forward"
    EXPANDING_WINDOW = "expanding_window"
    SLIDING_WINDOW = "sliding_window"
    # Financial specific
    PURGED_GROUP_TIMESERIES = "purged_group_timeseries"
    EMBARGO_CV = "embargo_cv"


class ScoringMetric(str, Enum):
    """Scoring metrics for model evaluation."""
    # Classification
    ACCURACY = "accuracy"
    BALANCED_ACCURACY = "balanced_accuracy"
    ROC_AUC = "roc_auc"
    ROC_AUC_OVR = "roc_auc_ovr"
    PR_AUC = "pr_auc"
    F1 = "f1"
    F1_WEIGHTED = "f1_weighted"
    PRECISION = "precision"
    RECALL = "recall"
    LOG_LOSS = "log_loss"
    BRIER_SCORE = "brier_score"
    MATTHEWS_CORRCOEF = "matthews_corrcoef"
    COHEN_KAPPA = "cohen_kappa"
    # Trading-specific
    PROFIT_FACTOR = "profit_factor"
    SHARPE_RATIO = "sharpe_ratio"
    SORTINO_RATIO = "sortino_ratio"
    CALMAR_RATIO = "calmar_ratio"
    WIN_RATE = "win_rate"


# -----------------------------------------------------------------------------
# SAMPLE WEIGHTING
# -----------------------------------------------------------------------------

class SampleWeightMethod(str, Enum):
    """Sample weighting methods."""
    NONE = "none"
    BALANCED = "balanced"
    INVERSE_FREQUENCY = "inverse_frequency"
    SQRT_INVERSE_FREQUENCY = "sqrt_inverse_frequency"
    # Time-based
    TIME_DECAY_LINEAR = "time_decay_linear"
    TIME_DECAY_EXPONENTIAL = "time_decay_exponential"
    RECENT_EMPHASIS = "recent_emphasis"
    # Volatility-based
    INVERSE_VOLATILITY = "inverse_volatility"
    HIGH_VOL_EMPHASIS = "high_vol_emphasis"
    # Return-based
    RETURN_MAGNITUDE = "return_magnitude"
    LOSS_EMPHASIS = "loss_emphasis"
    # Uniqueness
    SAMPLE_UNIQUENESS = "sample_uniqueness"
    TRIPLE_BARRIER_OVERLAP = "triple_barrier_overlap"


# -----------------------------------------------------------------------------
# MODEL LIFECYCLE STATUS
# -----------------------------------------------------------------------------

class ModelStatus(str, Enum):
    """Model lifecycle status."""
    QUEUED = "queued"
    TRAINING = "training"
    TRAINED = "trained"
    VALIDATING = "validating"
    VALIDATED = "validated"
    BACKTESTING = "backtesting"
    BACKTESTED = "backtested"
    PAPER_TRADING = "paper_trading"
    PRODUCTION = "production"
    MONITORING = "monitoring"
    DEGRADED = "degraded"
    DEPRECATED = "deprecated"
    FAILED = "failed"
