"""
GIGA TRADER - Grid Configuration & Dimension Definitions
=========================================================
Defines ALL configurable parameters across EVERY pipeline step for exhaustive grid search.

Grid Dimensions:
  1. Data Handling - Missing bar detection, gap filling, quality thresholds
  2. Feature Engineering - Which features to include, window sizes, thresholds
  3. Dimensionality Reduction - Methods, parameters, combinations
  4. Model Training - Algorithms, hyperparameters, regularization
  5. Entry/Exit Strategy - Timing windows, position sizing, batching, guardrails
  6. Anti-Overfitting - Synthetic data weights, ensemble configurations

Usage:
    from src.phase_23_analytics.grid_config import GridDimensions, GridConfig
"""

import json
from typing import Dict, Any


# =============================================================================
# 1. GRID DIMENSION DEFINITIONS
# =============================================================================

class GridDimensions:
    """
    All configurable grid dimensions across the pipeline.

    Each dimension is a dict of {param_name: [possible_values]}
    """

    # ─────────────────────────────────────────────────────────────────────────
    # DATA HANDLING GRID
    # ─────────────────────────────────────────────────────────────────────────
    DATA_HANDLING = {
        # Missing bar detection
        "min_bars_per_day": [100, 200, 300],  # Minimum bars to consider valid day
        "min_premarket_bars": [5, 10, 20],  # Minimum premarket bars
        "min_afterhours_bars": [5, 10, 15],  # Minimum afterhours bars

        # Gap handling strategies
        "fill_missing_bars": [True, False],
        "max_gap_minutes": [5, 10, 15, 30],  # Max gap to forward-fill
        "gap_fill_method": ["ffill", "interpolate", "zero", "drop"],

        # Quality thresholds
        "quality_score_min": [0.5, 0.6, 0.75],  # Min quality score to include day
        "flag_incomplete_extended": [True, False],

        # Data range
        "years_to_download": [3, 5, 7, 10],
    }

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE ENGINEERING GRID
    # ─────────────────────────────────────────────────────────────────────────
    FEATURE_ENGINEERING = {
        # Technical indicator windows
        "rsi_period": [7, 14, 21],
        "macd_fast": [8, 12, 16],
        "macd_slow": [20, 26, 30],
        "macd_signal": [5, 9, 12],
        "bb_period": [10, 20, 30],
        "bb_std": [1.5, 2.0, 2.5],

        # Momentum windows
        "momentum_windows": [
            [5, 10, 20],
            [3, 7, 14, 21],
            [5, 10, 20, 40],
        ],

        # Volume features
        "volume_ma_window": [10, 20, 50],
        "include_volume_profile": [True, False],

        # Extended hours features
        "include_premarket": [True, False],
        "include_afterhours": [True, False],
        "pm_lookback_days": [1, 3, 5],
        "ah_lookback_days": [1, 3, 5],

        # Calendar features
        "include_day_of_week": [True, False],
        "include_month": [True, False],
        "include_quarter_end": [True, False],

        # Swing threshold
        "swing_threshold": [0.002, 0.0025, 0.003, 0.0035, 0.004],
    }

    # ─────────────────────────────────────────────────────────────────────────
    # DIMENSIONALITY REDUCTION GRID
    # ─────────────────────────────────────────────────────────────────────────
    DIMENSIONALITY_REDUCTION = {
        # Method selection (which methods to include in ensemble)
        "use_variance_filter": [True, False],
        "use_correlation_filter": [True, False],
        "use_mutual_info": [True, False],
        "use_kernel_pca": [True, False],
        "use_ica": [True, False],
        "use_umap": [True, False],
        "use_kmedoids": [True, False],
        "use_agglomeration": [True, False],

        # Variance filter params
        "variance_threshold": [0.005, 0.01, 0.02, 0.05],

        # Correlation filter params
        "correlation_threshold": [0.85, 0.90, 0.95, 0.98],

        # Mutual Information params
        "mi_n_features": [15, 20, 25, 30, 40],
        "mi_n_neighbors": [3, 5, 7],

        # Kernel PCA params
        "kpca_n_components": [8, 12, 15, 20, 25],
        "kpca_kernel": ["rbf", "poly", "sigmoid", "cosine"],
        "kpca_gamma": [0.001, 0.01, 0.05, 0.1],

        # ICA params
        "ica_n_components": [5, 8, 10, 15, 20],
        "ica_max_iter": [200, 500, 1000],
        "ica_algorithm": ["parallel", "deflation"],

        # UMAP params
        "umap_n_components": [10, 15, 20, 30],
        "umap_n_neighbors": [5, 10, 15, 30],
        "umap_min_dist": [0.0, 0.1, 0.25, 0.5],
        "umap_metric": ["euclidean", "manhattan", "cosine"],

        # K-Medoids params
        "kmedoids_n_clusters": [10, 15, 20, 30],
        "kmedoids_metric": ["euclidean", "manhattan", "cosine"],
        "kmedoids_max_iter": [100, 300, 500],

        # Feature Agglomeration params
        "agglom_n_clusters": [15, 20, 25, 30],
        "agglom_linkage": ["ward", "complete", "average"],

        # Overall reduction target
        "target_dimensions": [20, 30, 40, 50],
    }

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL TRAINING GRID
    # ─────────────────────────────────────────────────────────────────────────
    MODEL_TRAINING = {
        # Model types
        "swing_model_type": ["logistic", "gradient_boosting", "ensemble"],
        "timing_model_type": ["logistic", "gradient_boosting", "ensemble"],

        # Logistic Regression params
        "lr_C": [0.01, 0.1, 1.0, 10.0],
        "lr_penalty": ["l1", "l2", "elasticnet"],
        "lr_solver": ["lbfgs", "saga"],

        # Gradient Boosting params
        "gb_n_estimators": [30, 50, 100, 150],
        "gb_max_depth": [2, 3, 4, 5],  # NEVER > 5
        "gb_learning_rate": [0.01, 0.05, 0.1, 0.2],
        "gb_min_samples_leaf": [20, 50, 100],
        "gb_subsample": [0.6, 0.8, 1.0],

        # Cross-validation
        "n_cv_folds": [3, 5, 7],
        "purge_days": [3, 5, 7],
        "embargo_days": [1, 2, 3],

        # Regularization (EDGE 1)
        "use_lasso_selection": [True, False],
        "lasso_alpha": [0.001, 0.01, 0.1],

        # Soft targets (EDGE 4)
        "use_soft_targets": [True, False],
        "soft_target_k": [30, 50, 100],
        "label_smoothing_epsilon": [0.05, 0.1, 0.15],
    }

    # ─────────────────────────────────────────────────────────────────────────
    # ENTRY/EXIT STRATEGY GRID (NEW)
    # ─────────────────────────────────────────────────────────────────────────
    ENTRY_EXIT_STRATEGY = {
        # Entry timing windows (minutes from market open)
        "entry_window_start": [0, 15, 30, 60],  # Start of entry window (min from open)
        "entry_window_end": [30, 60, 120, 180],  # End of entry window

        # Exit timing windows (minutes from market open)
        "exit_window_start": [180, 240, 300],  # Start of exit window
        "exit_window_end": [360, 390],  # End of exit window (390 = market close)

        # Position direction rules
        "long_only": [True, False],  # Only allow long positions
        "short_only": [False],  # Only allow short positions
        "allow_both": [True, False],  # Allow both long and short

        # Position sizing
        "base_position_pct": [0.05, 0.10, 0.15, 0.20],  # % of portfolio per trade
        "max_position_pct": [0.20, 0.30, 0.40],  # Max % of portfolio in one position
        "scale_by_confidence": [True, False],  # Scale position by model confidence
        "confidence_scale_factor": [0.5, 1.0, 1.5],  # How much to scale

        # Trade batching (EDGE 5: don't enter all at once)
        "batch_entry": [True, False],  # Split entry across multiple trades
        "n_entry_batches": [1, 2, 3, 5],  # Number of entry tranches
        "batch_interval_minutes": [5, 10, 15, 30],  # Time between batches
        "batch_size_method": ["equal", "pyramid", "reverse_pyramid"],

        # Exit batching
        "batch_exit": [True, False],
        "n_exit_batches": [1, 2, 3],
        "exit_interval_minutes": [5, 10, 15],

        # Guardrails
        "use_stop_loss": [True, False],
        "stop_loss_pct": [0.005, 0.01, 0.015, 0.02],  # Stop loss %
        "use_trailing_stop": [True, False],
        "trailing_stop_pct": [0.005, 0.01, 0.015],

        "use_take_profit": [True, False],
        "take_profit_pct": [0.01, 0.015, 0.02, 0.03],

        # Emergency exits
        "emergency_exit_loss_pct": [0.03, 0.05],  # Hard stop
        "emergency_exit_drawdown_pct": [0.10, 0.15],  # Portfolio drawdown trigger
        "max_daily_loss_pct": [0.02, 0.03, 0.05],  # Max loss per day

        # Time-based exits
        "force_exit_before_close_minutes": [5, 10, 15],  # Force exit N min before close
        "hold_overnight": [False],  # Never hold overnight for SPY swing

        # Minimum confidence thresholds
        "min_swing_confidence": [0.55, 0.60, 0.65, 0.70],
        "min_timing_confidence": [0.50, 0.55, 0.60],
        "require_both_models_agree": [True, False],
    }

    # ─────────────────────────────────────────────────────────────────────────
    # ANTI-OVERFITTING GRID
    # ─────────────────────────────────────────────────────────────────────────
    ANTI_OVERFIT = {
        # Synthetic universes
        "use_synthetic_universes": [True, False],
        "n_synthetic_universes": [5, 10, 15],
        "synthetic_weight": [0.2, 0.3, 0.4],

        # Cross-asset features
        "use_cross_assets": [True, False],
        "cross_asset_lookback": [5, 10, 20],

        # Breadth features
        "use_breadth_streaks": [True, False],
        "max_streak_days": [5, 7, 10],

        # Robustness ensemble
        "use_robustness_ensemble": [True, False],
        "n_dimension_variants": [1, 2, 3],
        "n_param_variants": [1, 2, 3],
        "param_noise_pct": [0.03, 0.05, 0.10],

        # WMES thresholds
        "wmes_threshold": [0.50, 0.55, 0.60],
        "fragility_threshold": [0.25, 0.35, 0.45],
    }


# =============================================================================
# 2. GRID CONFIGURATION CLASS
# =============================================================================

class GridConfig:
    """
    A single configuration from the grid.

    Holds all parameter values for one complete pipeline run.
    """

    def __init__(self, config_dict: Dict[str, Any], config_id: str = None):
        self.config = config_dict
        self.config_id = config_id or self._generate_id()
        self.results = None
        self.metrics = {}

    def _generate_id(self) -> str:
        """Generate unique config ID from hash."""
        import hashlib
        config_str = json.dumps(self.config, sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:12]

    def __getitem__(self, key: str) -> Any:
        return self.config.get(key)

    def __setitem__(self, key: str, value: Any):
        self.config[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def to_dict(self) -> Dict:
        return {
            "config_id": self.config_id,
            "config": self.config,
            "results": self.results,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "GridConfig":
        config = cls(d["config"], d.get("config_id"))
        config.results = d.get("results")
        config.metrics = d.get("metrics", {})
        return config
