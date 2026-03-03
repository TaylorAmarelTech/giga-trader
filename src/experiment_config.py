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

from src.core.system_resources import ResourceConfig, create_resource_config


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT CONFIGURATION SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class DataConfig:
    """Data source and preprocessing configuration."""
    # Data sources
    symbol: str = "SPY"
    years_to_download: int = 10
    chunk_days: int = 30

    # Quality thresholds
    min_bars_per_day: int = 200
    min_premarket_bars: int = 10
    min_afterhours_bars: int = 10

    # Missing data handling
    fill_missing_bars: bool = True
    max_gap_minutes: int = 15
    flag_incomplete_extended: bool = True

    # Regime filtering (Wave 26: regime-specific training)
    regime_filter: str = ""  # "", "low_vol", "high_vol"

    # OHLC data validation (Wave F1.1)
    validate_ohlc: bool = True

    # Wave G4: Information-driven bars (dollar/volume/tick bars)
    use_information_bars: bool = False  # Default OFF: alternative to time bars
    information_bar_type: str = "dollar"  # "dollar", "volume", "tick"


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

    # CUSUM event filter (Wave E4) — filter training data to significant-move dates
    use_cusum_filter: bool = False  # Default OFF: opt-in via grid search
    cusum_threshold: float = 0.01   # 1% cumulative return triggers event

    # Triple barrier labeling (Wave F5.1) — first-touch-wins: TP, SL, or time expiry
    use_triple_barrier: bool = False  # Default OFF: alternative to standard swing labeling
    tp_pct: float = 0.01           # Take-profit threshold
    sl_pct: float = 0.01           # Stop-loss threshold
    max_holding_days: int = 5      # Maximum holding period

    # Feature interaction discovery (Wave F5.2) — pairwise products/ratios
    use_interaction_discovery: bool = False  # Default OFF: expensive
    max_interactions: int = 20


@dataclass
class DimensionalityReductionConfig:
    """Dimensionality reduction configuration."""
    # Method: "pca", "kernel_pca", "ica", "umap", "mutual_info",
    #         "agglomeration", "kmedoids", "ensemble", "ensemble_plus"
    # Wave 31: Default changed to ica — but ExperimentGenerator randomizes
    # across all methods for diversity. This default only affects manual runs.
    method: str = "ica"

    # Pre-filtering
    variance_threshold: float = 0.01
    correlation_threshold: float = 0.95

    # Feature neutralization (Wave E1) — remove market beta from features
    neutralize_features: bool = False  # Default OFF: opt-in via full grid

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

    # Feature selection method: "mutual_info" or "f_classif"
    feature_selection_method: str = "mutual_info"

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

    # Wave F2: Class weight balancing for imbalanced data
    use_class_weights: bool = False

    # Wave F2: Stacking ensemble base models
    stacking_base_models: List[str] = field(
        default_factory=lambda: ["logistic_l2", "gradient_boosting", "hist_gradient_boosting"]
    )

    # Wave F4.1: Regime-conditional model routing
    use_regime_router: bool = False
    regime_split_method: str = "vix_quartile"  # "vix_quartile", "vix_fixed", "binary_vol"

    # Wave F6.2: Bayesian model averaging
    use_bma: bool = False


@dataclass
class CrossValidationConfig:
    """Cross-validation and evaluation configuration."""
    # CV settings
    n_cv_folds: int = 5
    purge_days: int = 10  # Increased from 5 — SPY features have 10-20 day autocorrelation
    embargo_days: int = 3  # Increased from 2 — extra safety margin

    # Soft targets (EDGE 4)
    use_soft_targets: bool = True
    soft_target_k: int = 50
    label_smoothing_epsilon: float = 0.1

    # Scoring
    primary_metric: str = "roc_auc"
    secondary_metrics: List[str] = field(
        default_factory=lambda: ["accuracy", "precision", "recall", "f1"]
    )

    # Wave F4.3: Isotonic probability recalibration
    use_isotonic_calibration: bool = False


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

    # Bear market synthetic universes (bull bias correction)
    use_bear_universes: bool = True
    bear_mean_shift_bps: List[int] = field(default_factory=lambda: [5, 10])
    bear_vol_amplify_factor: float = 1.5
    bear_vol_dampen_factor: float = 0.7

    # Multi-timescale regime bootstrap (weekly/monthly/magnitude/vol resampling)
    use_multiscale_bootstrap: bool = True

    # Economic indicator features (yields, VIX, credit spreads via yfinance)
    use_economic_features: bool = True

    # Calendar & event features (FOMC, opex, NFP, CPI, PMI, GDP, seasonality)
    use_calendar_features: bool = True

    # Sentiment features (VIX-derived fear/greed, cross-asset flows, optional news)
    use_sentiment_features: bool = True

    # CNN Fear & Greed Index features (contrarian indicator)
    use_fear_greed: bool = True

    # Reddit sentiment features (ApeWisdom)
    use_reddit_sentiment: bool = True

    # Congressional trading proxy features (smart-money volume signal)
    use_congressional_features: bool = True

    # Crypto Fear & Greed Index (risk-on/off proxy)
    use_crypto_sentiment: bool = True

    # Gamma Exposure (GEX) proxy from VIX term structure
    use_gamma_exposure: bool = True

    # Finnhub Social sentiment (requires FINNHUB_API_KEY)
    use_finnhub_social: bool = True

    # FINRA Dark Pool short sale volume features (2-4 week lag)
    use_dark_pool: bool = True

    # Options IV/Skew features (VIX rank, CBOE SKEW Index, vol-of-vol)
    use_options_features: bool = True

    # Event recency features (days since last -1%/-2% drop, reversal, etc.)
    use_event_recency: bool = True

    # Block structure features (multi-day 3d/5d blocks, cascades, texture)
    use_block_structure: bool = True

    # Amihud illiquidity ratio features (|return|/dollar_volume)
    use_amihud_features: bool = True

    # Range-based volatility features (Garman-Klass, Yang-Zhang, Rogers-Satchell)
    use_range_vol_features: bool = True

    # Entropy features (Shannon, permutation, sample entropy)
    use_entropy_features: bool = True

    # Hurst exponent features (R/S analysis, trending/mean-reverting regimes)
    use_hurst_features: bool = True

    # NMI market efficiency features (mutual information between returns and lags)
    use_nmi_features: bool = True

    # Absorption ratio features (PCA-based systemic risk)
    use_absorption_ratio: bool = True

    # ADWIN drift detection features (distribution change detection)
    use_drift_features: bool = True

    # Changepoint detection features (Bayesian online changepoint detection)
    use_changepoint_features: bool = True

    # HMM regime features (rolling hidden Markov model states)
    use_hmm_features: bool = True

    # VPIN order flow toxicity features (bulk volume classification)
    use_vpin_features: bool = True

    # Intraday momentum features (overnight gap, close location, reversal)
    use_intraday_momentum: bool = True

    # Futures-spot basis features (ES=F vs SPY spread proxy)
    use_futures_basis: bool = True

    # Insider aggregate proxy features (price+volume accumulation signals)
    use_insider_aggregate: bool = True

    # ETF fund flow proxy features (volume-price divergence, creation/redemption)
    use_etf_flow: bool = True

    # Wavelet decomposition features (multi-resolution price decomposition)
    use_wavelet_features: bool = True

    # SAX pattern features (symbolic aggregate approximation)
    use_sax_features: bool = True

    # Transfer entropy features (cross-asset information flow)
    use_transfer_entropy: bool = True

    # MFDFA features (multifractal detrended fluctuation analysis)
    use_mfdfa_features: bool = True

    # RQA features (recurrence quantification analysis)
    use_rqa_features: bool = True

    # Copula tail dependence features (empirical copula lower/upper lambda)
    use_copula_features: bool = True

    # Correlation network centrality features (cross-asset graph metrics)
    use_network_centrality: bool = True

    # Path signature features (iterated integrals of price/volume paths)
    use_path_signatures: bool = True

    # Wavelet scattering transform features (multi-scale time-frequency)
    use_wavelet_scattering: bool = True

    # Wasserstein regime detection features (distribution-distance regime changes)
    use_wasserstein_regime: bool = True

    # Market structure features (compression/squeeze, attractors, inflection zones)
    use_market_structure: bool = True

    # Time series model features (ARIMA residuals, optional Chronos/catch22)
    use_time_series_models: bool = False   # Needs statsmodels; Chronos/catch22 optional

    # catch22 canonical time series features (requires pycatch22)
    use_catch22: bool = False

    # HAR-RV features (multi-horizon volatility cascade)
    use_har_rv: bool = True

    # L-Moments features (robust distributional shape)
    use_l_moments: bool = True

    # Multiscale sample entropy features (cross-scale complexity)
    use_multiscale_entropy: bool = True

    # RV signature plot features (cross-frequency microstructure noise)
    use_rv_signature_plot: bool = False

    # TDA persistent homology features (topological crash detection, requires giotto-tda)
    use_tda_homology: bool = False

    # Wave J: Credit spread features (HYG-LQD spread, credit stress)
    use_credit_spread_features: bool = True

    # Wave J: Yield curve features (2s10s, 3m10y, curvature, inversion)
    use_yield_curve_features: bool = True

    # Wave J: Volatility term structure features (VIX/VIX3M contango/backwardation)
    use_vol_term_structure_features: bool = True

    # Wave J: Macro surprise features (deviation from trend proxies)
    use_macro_surprise_features: bool = True

    # Wave J: Cross-asset momentum features (TLT/GLD/HYG lead-lag signals)
    use_cross_asset_momentum: bool = True

    # Wave J: Skew/kurtosis features (higher-order moment analysis)
    use_skew_kurtosis_features: bool = True

    # Wave J: Seasonality features (calendar anomalies: TOM, January, FOMC drift)
    use_seasonality_features: bool = True

    # Wave J: Order flow imbalance features (BVC buy/sell pressure)
    use_order_flow_imbalance: bool = True

    # Wave K: Correlation regime features (cross-asset correlation matrix)
    use_correlation_regime: bool = True

    # Wave K: Fama-French factor exposure features (ETF proxies)
    use_fama_french: bool = True

    # Wave K: Put-call ratio features (VIX/vol proxy fallback)
    use_put_call_ratio: bool = True

    # Wave K: Multi-horizon ensemble filter features (1d/3d/5d)
    use_multi_horizon: bool = True

    # Meta-labeling: secondary classifier predicting signal profitability
    use_meta_labeling: bool = True

    # Synthetic weight penalty (prevents overfitting to synthetic data)
    synthetic_weight_penalty: float = 0.5   # Synthetic rows get 50% of confidence weight
    synthetic_weight_floor: float = 0.10    # Min weight for any synthetic sample
    synthetic_weight_ceiling: float = 0.60  # Max weight for any synthetic sample

    # Wave F3.1: Label noise robustness test
    use_label_noise_test: bool = True

    # Wave F3.2: Adversarial validation hard gate threshold
    adversarial_gate_threshold: float = 0.65

    # Wave F3.3: Feature importance stability gate
    use_fi_stability_gate: bool = True
    fi_stability_threshold: float = 0.5

    # Wave F3.4: Knockoff feature hard gate (expensive, default OFF)
    use_knockoff_gate: bool = False

    # Wave F4.4: Purge/embargo at synthetic data boundary
    purge_synthetic_boundary: bool = True

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
class TrainingAugmentationConfig:
    """Wave 35: Training augmentation configuration for anti-overfitting."""
    # Temporal decay weighting — emphasize recent samples
    use_temporal_decay: bool = False
    temporal_decay_lambda: float = 0.5  # Higher = more recent emphasis

    # Noise injection — add Gaussian noise to features during training
    use_noise_injection: bool = False
    noise_sigma: float = 0.1  # Std dev multiplier per feature

    # Nested CV — double cross-validation for honest evaluation
    use_nested_cv: bool = False
    nested_outer_folds: int = 3
    nested_inner_folds: int = 3

    # Calibrated ensemble distillation — detect memorization
    use_distillation: bool = True  # Always on by default (cheap)


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
class TemporalCascadeConfig:
    """Temporal cascade model configuration for anti-overfitting."""
    # Master switch
    use_temporal_cascade: bool = True

    # Temporal slices to train (minutes from market open)
    temporal_slices: List[int] = field(
        default_factory=lambda: [0, 30, 60, 90, 120, 180]
    )

    # Model settings
    model_type: str = "gradient_boosting"  # "gradient_boosting", "logistic", "ensemble"
    regularization_strength: float = 0.1

    # Training settings
    cv_folds: int = 5
    purge_days: int = 10
    embargo_days: int = 3

    # Thresholds for model registration
    min_cv_auc: float = 0.55
    min_slices_passing: int = 3

    # Prediction weighting
    weight_by_recency: bool = True  # More recent temporal slices get higher weight
    min_agreement_for_signal: float = 0.6  # Minimum model agreement to generate signal


@dataclass
class TradingConfig:
    """Trading strategy configuration."""
    entry_threshold: float = 0.6
    exit_threshold: float = 0.4
    stop_loss_pct: float = 0.01
    take_profit_pct: float = 0.02
    max_daily_trades: int = 5
    position_sizing: str = "fixed"  # "fixed", "kelly", "volatility_scaled"

    # Wave F1.2: Circuit breaker enforcement
    use_circuit_breaker: bool = True
    max_daily_loss_pct: float = 0.02
    max_drawdown_pct: float = 0.10
    max_consecutive_losses: int = 5

    # Wave F1.3: Model staleness check
    max_model_age_days: int = 30

    # Wave F2.2: Dynamic ensemble weighting
    use_dynamic_weights: bool = False
    dynamic_weight_lookback: int = 20

    # Wave F4.2: Conformal prediction sizing
    use_conformal_sizing: bool = False
    conformal_alpha: float = 0.1

    # Wave F6.1: Feature drift monitoring
    use_drift_monitor: bool = True
    drift_psi_threshold: float = 0.2

    # Wave F6.3: Online learning updates
    use_online_learning: bool = False
    online_buffer_days: int = 5

    # Wave G2: CVaR-based position sizing (tail-risk-aware)
    use_cvar_sizing: bool = False
    cvar_alpha: float = 0.05
    cvar_lookback: int = 60
    cvar_target: float = 0.02

    # Wave G3: Thompson Sampling model selection (bandit-based)
    use_thompson_selector: bool = False
    thompson_decay: float = 0.995
    thompson_min_weight: float = 0.1

    # Wave J: Dynamic Kelly position sizing (VIX-conditioned Kelly criterion)
    use_dynamic_kelly: bool = False

    # Wave J: Drawdown-adaptive position sizing (power-law decay)
    use_drawdown_adaptive_sizing: bool = False


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
    temporal_cascade: TemporalCascadeConfig = field(default_factory=TemporalCascadeConfig)
    training_augmentation: TrainingAugmentationConfig = field(default_factory=TrainingAugmentationConfig)
    trading: TradingConfig = field(default_factory=TradingConfig)

    # System resource detection and adaptive scaling
    resources: ResourceConfig = field(default_factory=create_resource_config)

    # Wave 32: Arbitrary metadata for research experiments, feature candidates, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)

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
            temporal_cascade=TemporalCascadeConfig(**d.get("temporal_cascade", {})),
            training_augmentation=TrainingAugmentationConfig(**d.get("training_augmentation", {})),
            trading=TradingConfig(**d.get("trading", {})),
            resources=(
                ResourceConfig(**d["resources"]) if d.get("resources")
                else create_resource_config()
            ),
            metadata=d.get("metadata", {}),
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
