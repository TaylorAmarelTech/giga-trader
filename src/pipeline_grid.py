"""
GIGA TRADER - Comprehensive Pipeline Grid Configuration
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
    from src.pipeline_grid import PipelineGridSearch, GridConfig

    grid = PipelineGridSearch()
    for config in grid.iterate_all_configs():
        results = train_with_config(config)
        grid.record_results(config, results)
"""

import os
import sys
from datetime import datetime, time
from pathlib import Path
from itertools import product
from typing import Dict, List, Tuple, Optional, Any, Generator
import json

import numpy as np
import pandas as pd

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


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


# =============================================================================
# 3. GRID SEARCH ENGINE
# =============================================================================

class PipelineGridSearch:
    """
    Exhaustive grid search across all pipeline configurations.

    Supports:
      - Full grid search (all combinations)
      - Random search (sample N configs)
      - Smart search (prioritize based on prior results)
      - Incremental search (resume from checkpoint)
    """

    def __init__(
        self,
        dimensions: Dict[str, Dict] = None,
        search_mode: str = "random",  # "full", "random", "smart"
        max_configs: int = 1000,
        random_seed: int = 42,
    ):
        self.dimensions = dimensions or self._get_default_dimensions()
        self.search_mode = search_mode
        self.max_configs = max_configs
        self.random_seed = random_seed

        self.results = []
        self.best_config = None
        self.best_score = -np.inf

        np.random.seed(random_seed)

    def _get_default_dimensions(self) -> Dict[str, Dict]:
        """Get default grid dimensions from GridDimensions class."""
        return {
            "data": GridDimensions.DATA_HANDLING,
            "features": GridDimensions.FEATURE_ENGINEERING,
            "dim_reduction": GridDimensions.DIMENSIONALITY_REDUCTION,
            "training": GridDimensions.MODEL_TRAINING,
            "entry_exit": GridDimensions.ENTRY_EXIT_STRATEGY,
            "anti_overfit": GridDimensions.ANTI_OVERFIT,
        }

    def count_total_configs(self) -> int:
        """Count total number of possible configurations."""
        total = 1
        for category, params in self.dimensions.items():
            for param_name, values in params.items():
                total *= len(values)
        return total

    def iterate_all_configs(self) -> Generator[GridConfig, None, None]:
        """
        Iterate through configurations based on search mode.

        Yields GridConfig objects.
        """
        if self.search_mode == "full":
            yield from self._full_grid_search()
        elif self.search_mode == "random":
            yield from self._random_search()
        elif self.search_mode == "smart":
            yield from self._smart_search()
        else:
            raise ValueError(f"Unknown search mode: {self.search_mode}")

    def _full_grid_search(self) -> Generator[GridConfig, None, None]:
        """Exhaustive search through all combinations."""
        # Flatten all parameters
        all_params = {}
        for category, params in self.dimensions.items():
            for param_name, values in params.items():
                all_params[f"{category}__{param_name}"] = values

        # Generate all combinations
        param_names = list(all_params.keys())
        param_values = list(all_params.values())

        count = 0
        for values in product(*param_values):
            config_dict = dict(zip(param_names, values))
            # Unflatten
            unflattened = self._unflatten_config(config_dict)

            yield GridConfig(unflattened)

            count += 1
            if count >= self.max_configs:
                print(f"[WARN] Reached max_configs limit ({self.max_configs})")
                return

    def _random_search(self) -> Generator[GridConfig, None, None]:
        """Random sampling from the grid."""
        for _ in range(self.max_configs):
            config_dict = {}

            for category, params in self.dimensions.items():
                config_dict[category] = {}
                for param_name, values in params.items():
                    # Handle list-type values (like momentum_windows)
                    idx = np.random.randint(0, len(values))
                    config_dict[category][param_name] = values[idx]

            yield GridConfig(config_dict)

    def _smart_search(self) -> Generator[GridConfig, None, None]:
        """
        Smart search that prioritizes promising regions.

        Uses Thompson Sampling / Bayesian optimization concepts.
        """
        # Start with random exploration
        exploration_fraction = 0.3
        n_explore = int(self.max_configs * exploration_fraction)

        # Exploration phase
        for config in self._random_search():
            yield config
            n_explore -= 1
            if n_explore <= 0:
                break

        # Exploitation phase - sample near best configs
        n_exploit = self.max_configs - int(self.max_configs * exploration_fraction)

        for _ in range(n_exploit):
            if self.best_config is None:
                # No best config yet, keep exploring
                config_dict = {}
                for category, params in self.dimensions.items():
                    config_dict[category] = {}
                    for param_name, values in params.items():
                        config_dict[category][param_name] = np.random.choice(values)
            else:
                # Perturb best config
                config_dict = self._perturb_config(self.best_config.config)

            yield GridConfig(config_dict)

    def _perturb_config(self, base_config: Dict) -> Dict:
        """Perturb a config by randomly changing some parameters."""
        config_dict = {}
        perturbation_prob = 0.2  # Probability of changing each param

        for category, params in base_config.items():
            config_dict[category] = {}
            for param_name, value in params.items():
                if np.random.random() < perturbation_prob:
                    # Change this parameter
                    possible_values = self.dimensions.get(category, {}).get(param_name, [value])
                    idx = np.random.randint(0, len(possible_values))
                    config_dict[category][param_name] = possible_values[idx]
                else:
                    # Keep same value
                    config_dict[category][param_name] = value

        return config_dict

    def _unflatten_config(self, flat_config: Dict) -> Dict:
        """Convert flattened config back to nested structure."""
        unflattened = {}
        for key, value in flat_config.items():
            category, param_name = key.split("__", 1)
            if category not in unflattened:
                unflattened[category] = {}
            unflattened[category][param_name] = value
        return unflattened

    def record_result(
        self,
        config: GridConfig,
        metrics: Dict[str, float],
        primary_metric: str = "wmes",
    ):
        """Record results for a configuration."""
        config.metrics = metrics
        config.results = metrics.get(primary_metric, 0)

        self.results.append(config.to_dict())

        # Update best
        if config.results > self.best_score:
            self.best_score = config.results
            self.best_config = config
            print(f"[NEW BEST] Config {config.config_id}: {primary_metric}={config.results:.4f}")

    def get_top_configs(self, n: int = 10) -> List[GridConfig]:
        """Get top N configurations by score."""
        sorted_results = sorted(
            self.results,
            key=lambda x: x.get("results", 0),
            reverse=True
        )
        return [GridConfig.from_dict(r) for r in sorted_results[:n]]

    def save_results(self, path: str):
        """Save all results to JSON."""
        with open(path, "w") as f:
            json.dump({
                "search_mode": self.search_mode,
                "max_configs": self.max_configs,
                "random_seed": self.random_seed,
                "n_completed": len(self.results),
                "best_config_id": self.best_config.config_id if self.best_config else None,
                "best_score": self.best_score,
                "results": self.results,
            }, f, indent=2, default=str)
        print(f"[SAVED] Results to {path}")

    def load_results(self, path: str):
        """Load results from JSON to resume search."""
        with open(path, "r") as f:
            data = json.load(f)

        self.results = data.get("results", [])
        self.best_score = data.get("best_score", -np.inf)

        if data.get("best_config_id"):
            for r in self.results:
                if r.get("config_id") == data["best_config_id"]:
                    self.best_config = GridConfig.from_dict(r)
                    break

        print(f"[LOADED] {len(self.results)} results from {path}")


# =============================================================================
# 4. QUICK CONFIG PRESETS
# =============================================================================

class QuickPresets:
    """Pre-defined configuration presets for common scenarios."""

    @staticmethod
    def conservative() -> Dict:
        """Conservative settings - high regularization, low complexity."""
        return {
            "data": {
                "min_bars_per_day": 200,
                "fill_missing_bars": True,
                "max_gap_minutes": 10,
                "quality_score_min": 0.75,
            },
            "dim_reduction": {
                "use_variance_filter": True,
                "use_correlation_filter": True,
                "use_mutual_info": True,
                "mi_n_features": 20,
                "use_kernel_pca": True,
                "kpca_n_components": 10,
                "use_ica": False,
                "use_umap": False,
                "use_kmedoids": False,
                "target_dimensions": 30,
            },
            "training": {
                "swing_model_type": "logistic",
                "lr_C": 0.1,  # High regularization
                "gb_max_depth": 3,
                "use_soft_targets": True,
            },
            "entry_exit": {
                "long_only": True,
                "base_position_pct": 0.10,
                "batch_entry": True,
                "n_entry_batches": 3,
                "use_stop_loss": True,
                "stop_loss_pct": 0.01,
                "min_swing_confidence": 0.65,
            },
            "anti_overfit": {
                "use_synthetic_universes": True,
                "synthetic_weight": 0.3,
                "use_robustness_ensemble": True,
                "wmes_threshold": 0.55,
            },
        }

    @staticmethod
    def aggressive() -> Dict:
        """Aggressive settings - more features, lower thresholds."""
        return {
            "data": {
                "min_bars_per_day": 100,
                "fill_missing_bars": True,
                "max_gap_minutes": 30,
                "quality_score_min": 0.5,
            },
            "dim_reduction": {
                "use_variance_filter": True,
                "use_correlation_filter": True,
                "use_mutual_info": True,
                "mi_n_features": 30,
                "use_kernel_pca": True,
                "kpca_n_components": 15,
                "use_ica": True,
                "ica_n_components": 10,
                "use_umap": True,
                "umap_n_components": 15,
                "use_kmedoids": True,
                "kmedoids_n_clusters": 15,
                "target_dimensions": 50,
            },
            "training": {
                "swing_model_type": "ensemble",
                "lr_C": 1.0,
                "gb_max_depth": 4,
                "gb_n_estimators": 100,
                "use_soft_targets": True,
            },
            "entry_exit": {
                "long_only": False,
                "allow_both": True,
                "base_position_pct": 0.15,
                "scale_by_confidence": True,
                "batch_entry": True,
                "n_entry_batches": 2,
                "use_stop_loss": True,
                "stop_loss_pct": 0.015,
                "min_swing_confidence": 0.55,
            },
            "anti_overfit": {
                "use_synthetic_universes": True,
                "synthetic_weight": 0.2,
                "use_robustness_ensemble": True,
                "wmes_threshold": 0.50,
            },
        }

    @staticmethod
    def minimal() -> Dict:
        """Minimal settings for fast iteration."""
        return {
            "data": {
                "years_to_download": 3,
                "min_bars_per_day": 200,
            },
            "dim_reduction": {
                "use_variance_filter": True,
                "use_correlation_filter": True,
                "use_mutual_info": True,
                "mi_n_features": 25,
                "use_kernel_pca": False,
                "use_ica": False,
                "use_umap": False,
                "use_kmedoids": False,
                "target_dimensions": 25,
            },
            "training": {
                "swing_model_type": "logistic",
                "n_cv_folds": 3,
            },
            "entry_exit": {
                "long_only": True,
                "batch_entry": False,
                "min_swing_confidence": 0.55,
            },
            "anti_overfit": {
                "use_synthetic_universes": False,
                "use_robustness_ensemble": False,
            },
        }


# =============================================================================
# 5. ENTRY/EXIT PREDICTOR (What's Missing)
# =============================================================================

class EntryExitPredictor:
    """
    Predicts optimal entry and exit parameters based on model outputs.

    This bridges the gap between model predictions and actual trading decisions.

    Inputs:
      - swing_probability: P(up day)
      - timing_probability: P(low before high)
      - confidence scores
      - current market conditions

    Outputs:
      - Entry time window
      - Exit time window
      - Position direction (long/short)
      - Position size
      - Stop loss level
      - Take profit level
      - Batch schedule
    """

    def __init__(self, config: Dict):
        self.config = config
        self.entry_exit_config = config.get("entry_exit", {})

    def predict(
        self,
        swing_proba: float,
        timing_proba: float,
        current_price: float,
        volatility: float,
        hour: int,
        minute: int,
    ) -> Dict:
        """
        Generate entry/exit decision based on model predictions.

        Returns:
            Dict with entry/exit parameters
        """
        decision = {
            "action": "HOLD",  # LONG, SHORT, HOLD
            "confidence": 0.0,
            "entry_window": None,
            "exit_window": None,
            "position_size_pct": 0.0,
            "stop_loss": None,
            "take_profit": None,
            "batch_schedule": [],
            "guardrails": {},
        }

        # Check minimum confidence thresholds
        min_swing_conf = self.entry_exit_config.get("min_swing_confidence", 0.60)
        min_timing_conf = self.entry_exit_config.get("min_timing_confidence", 0.55)
        require_both = self.entry_exit_config.get("require_both_models_agree", True)

        # Determine direction
        is_bullish = swing_proba > min_swing_conf
        is_bearish = swing_proba < (1 - min_swing_conf)
        timing_valid = timing_proba > min_timing_conf or timing_proba < (1 - min_timing_conf)

        if require_both and not timing_valid:
            return decision  # HOLD

        # Determine action
        long_only = self.entry_exit_config.get("long_only", True)

        if is_bullish:
            decision["action"] = "LONG"
            decision["confidence"] = swing_proba
        elif is_bearish and not long_only:
            decision["action"] = "SHORT"
            decision["confidence"] = 1 - swing_proba
        else:
            return decision  # HOLD

        # Calculate position size
        base_size = self.entry_exit_config.get("base_position_pct", 0.10)
        max_size = self.entry_exit_config.get("max_position_pct", 0.20)

        if self.entry_exit_config.get("scale_by_confidence", False):
            scale_factor = self.entry_exit_config.get("confidence_scale_factor", 1.0)
            position_size = base_size * (0.5 + decision["confidence"] * scale_factor)
        else:
            position_size = base_size

        decision["position_size_pct"] = min(position_size, max_size)

        # Entry window
        entry_start = self.entry_exit_config.get("entry_window_start", 30)
        entry_end = self.entry_exit_config.get("entry_window_end", 120)
        decision["entry_window"] = (entry_start, entry_end)

        # Exit window
        exit_start = self.entry_exit_config.get("exit_window_start", 300)
        exit_end = self.entry_exit_config.get("exit_window_end", 385)
        decision["exit_window"] = (exit_start, exit_end)

        # Stop loss
        if self.entry_exit_config.get("use_stop_loss", True):
            stop_pct = self.entry_exit_config.get("stop_loss_pct", 0.01)
            if decision["action"] == "LONG":
                decision["stop_loss"] = current_price * (1 - stop_pct)
            else:
                decision["stop_loss"] = current_price * (1 + stop_pct)

        # Take profit
        if self.entry_exit_config.get("use_take_profit", False):
            tp_pct = self.entry_exit_config.get("take_profit_pct", 0.02)
            if decision["action"] == "LONG":
                decision["take_profit"] = current_price * (1 + tp_pct)
            else:
                decision["take_profit"] = current_price * (1 - tp_pct)

        # Batch schedule
        if self.entry_exit_config.get("batch_entry", False):
            n_batches = self.entry_exit_config.get("n_entry_batches", 3)
            interval = self.entry_exit_config.get("batch_interval_minutes", 10)
            method = self.entry_exit_config.get("batch_size_method", "equal")

            decision["batch_schedule"] = self._create_batch_schedule(
                n_batches, interval, method, decision["position_size_pct"]
            )
        else:
            decision["batch_schedule"] = [{
                "time_offset": 0,
                "size_pct": decision["position_size_pct"],
            }]

        # Guardrails
        decision["guardrails"] = {
            "emergency_exit_loss": self.entry_exit_config.get("emergency_exit_loss_pct", 0.05),
            "max_daily_loss": self.entry_exit_config.get("max_daily_loss_pct", 0.03),
            "force_exit_minutes_before_close": self.entry_exit_config.get(
                "force_exit_before_close_minutes", 10
            ),
            "hold_overnight": self.entry_exit_config.get("hold_overnight", False),
        }

        return decision

    def _create_batch_schedule(
        self,
        n_batches: int,
        interval: int,
        method: str,
        total_size: float,
    ) -> List[Dict]:
        """Create batch entry schedule."""
        schedule = []

        if method == "equal":
            size_per_batch = total_size / n_batches
            sizes = [size_per_batch] * n_batches
        elif method == "pyramid":
            # Start small, increase
            weights = list(range(1, n_batches + 1))
            total_weight = sum(weights)
            sizes = [w / total_weight * total_size for w in weights]
        elif method == "reverse_pyramid":
            # Start large, decrease
            weights = list(range(n_batches, 0, -1))
            total_weight = sum(weights)
            sizes = [w / total_weight * total_size for w in weights]
        else:
            sizes = [total_size / n_batches] * n_batches

        for i, size in enumerate(sizes):
            schedule.append({
                "batch_num": i + 1,
                "time_offset_minutes": i * interval,
                "size_pct": size,
            })

        return schedule


# =============================================================================
# 6. MULTI-OBJECTIVE OPTIMIZER
# =============================================================================

class MultiObjectiveOptimizer:
    """
    Multi-objective optimization for finding Pareto-optimal configurations.

    Objectives (to maximize unless specified):
      - sharpe_ratio: Risk-adjusted return
      - win_rate: Percentage of winning trades
      - profit_factor: Gross profit / gross loss
      - total_return: Total return percentage
      - negative_max_drawdown: Max drawdown (negated, so lower is better)

    Methods:
      - NSGA-II style dominance sorting
      - Scalarization (weighted sum)
      - Thompson Sampling with multi-armed bandits
    """

    def __init__(
        self,
        objectives: List[str] = None,
        objective_weights: Dict[str, float] = None,
        pareto_archive_size: int = 100,
    ):
        self.objectives = objectives or [
            "sharpe_ratio",
            "win_rate",
            "profit_factor",
            "neg_max_drawdown",  # Negated so we maximize it
        ]

        self.objective_weights = objective_weights or {
            "sharpe_ratio": 0.35,
            "win_rate": 0.25,
            "profit_factor": 0.20,
            "neg_max_drawdown": 0.20,
        }

        self.pareto_archive_size = pareto_archive_size
        self.pareto_front: List[Dict] = []
        self.all_results: List[Dict] = []

    def evaluate_config(
        self,
        config: GridConfig,
        backtest_results: Dict,
    ) -> Dict[str, float]:
        """Extract objective values from backtest results."""
        metrics = {
            "sharpe_ratio": backtest_results.get("sharpe_ratio", 0),
            "win_rate": backtest_results.get("win_rate", 0),
            "profit_factor": backtest_results.get("profit_factor", 0),
            "total_return": backtest_results.get("total_return_pct", 0) / 100,
            "neg_max_drawdown": -backtest_results.get("max_drawdown_pct", 100) / 100,
            "sortino_ratio": backtest_results.get("sortino_ratio", 0),
            "n_trades": backtest_results.get("n_trades", 0),
        }

        # Only include requested objectives
        return {k: metrics.get(k, 0) for k in self.objectives}

    def dominates(self, a: Dict[str, float], b: Dict[str, float]) -> bool:
        """
        Check if solution a dominates solution b.

        a dominates b if:
          - a is at least as good as b in all objectives
          - a is strictly better in at least one objective
        """
        at_least_as_good = all(a[obj] >= b[obj] for obj in self.objectives)
        strictly_better = any(a[obj] > b[obj] for obj in self.objectives)
        return at_least_as_good and strictly_better

    def update_pareto_front(
        self,
        config: GridConfig,
        objectives: Dict[str, float],
    ):
        """Update Pareto front with new solution."""
        result = {
            "config_id": config.config_id,
            "config": config.config,
            "objectives": objectives,
        }

        # Check if dominated by any current Pareto member
        for pf_member in self.pareto_front:
            if self.dominates(pf_member["objectives"], objectives):
                # New solution is dominated, don't add
                return False

        # Remove any solutions dominated by the new one
        self.pareto_front = [
            pf for pf in self.pareto_front
            if not self.dominates(objectives, pf["objectives"])
        ]

        # Add new solution
        self.pareto_front.append(result)

        # Maintain archive size using crowding distance
        if len(self.pareto_front) > self.pareto_archive_size:
            self._prune_pareto_front()

        return True

    def _prune_pareto_front(self):
        """Prune Pareto front using crowding distance."""
        if len(self.pareto_front) <= self.pareto_archive_size:
            return

        # Calculate crowding distance for each solution
        crowding_distances = self._calculate_crowding_distances()

        # Sort by crowding distance and keep top solutions
        sorted_indices = np.argsort(crowding_distances)[::-1]
        self.pareto_front = [self.pareto_front[i] for i in sorted_indices[:self.pareto_archive_size]]

    def _calculate_crowding_distances(self) -> np.ndarray:
        """Calculate crowding distance for diversity preservation."""
        n = len(self.pareto_front)
        distances = np.zeros(n)

        for obj in self.objectives:
            # Sort by this objective
            values = np.array([pf["objectives"][obj] for pf in self.pareto_front])
            sorted_indices = np.argsort(values)

            # Boundary points get infinite distance
            distances[sorted_indices[0]] = np.inf
            distances[sorted_indices[-1]] = np.inf

            # Interior points
            obj_range = values.max() - values.min()
            if obj_range > 0:
                for i in range(1, n - 1):
                    idx = sorted_indices[i]
                    prev_idx = sorted_indices[i - 1]
                    next_idx = sorted_indices[i + 1]
                    distances[idx] += (values[next_idx] - values[prev_idx]) / obj_range

        return distances

    def scalarize(self, objectives: Dict[str, float]) -> float:
        """Convert multi-objective to single score using weighted sum."""
        return sum(
            self.objective_weights.get(obj, 0) * objectives.get(obj, 0)
            for obj in self.objectives
        )

    def get_best_by_scalarization(self) -> Optional[Dict]:
        """Get best solution using weighted scalarization."""
        if not self.pareto_front:
            return None

        best = max(
            self.pareto_front,
            key=lambda x: self.scalarize(x["objectives"])
        )
        return best

    def get_best_by_objective(self, objective: str) -> Optional[Dict]:
        """Get best solution for a specific objective."""
        if not self.pareto_front:
            return None

        return max(self.pareto_front, key=lambda x: x["objectives"].get(objective, 0))

    def suggest_next_config(
        self,
        grid: PipelineGridSearch,
        exploration_rate: float = 0.3,
    ) -> GridConfig:
        """
        Suggest next configuration to try using Thompson Sampling.

        Balances exploitation (near best configs) and exploration.
        """
        if len(self.pareto_front) == 0 or np.random.random() < exploration_rate:
            # Exploration: random config
            for config in grid._random_search():
                return config

        # Exploitation: perturb a Pareto-optimal config
        # Select from Pareto front proportional to scalarized score
        scores = [self.scalarize(pf["objectives"]) for pf in self.pareto_front]
        scores = np.array(scores)
        scores = scores - scores.min() + 1e-8  # Make positive
        probs = scores / scores.sum()

        selected_idx = np.random.choice(len(self.pareto_front), p=probs)
        base_config = self.pareto_front[selected_idx]["config"]

        # Perturb the config
        perturbed = grid._perturb_config(base_config)
        return GridConfig(perturbed)

    def get_pareto_summary(self) -> pd.DataFrame:
        """Get summary DataFrame of Pareto front."""
        if not self.pareto_front:
            return pd.DataFrame()

        records = []
        for pf in self.pareto_front:
            record = {"config_id": pf["config_id"]}
            record.update(pf["objectives"])
            record["scalarized_score"] = self.scalarize(pf["objectives"])
            records.append(record)

        df = pd.DataFrame(records)
        return df.sort_values("scalarized_score", ascending=False)

    def visualize_pareto_front(self, obj_x: str, obj_y: str) -> Dict:
        """
        Get data for visualizing 2D projection of Pareto front.

        Returns dict with x, y coordinates for plotting.
        """
        if not self.pareto_front:
            return {"x": [], "y": [], "labels": []}

        x = [pf["objectives"].get(obj_x, 0) for pf in self.pareto_front]
        y = [pf["objectives"].get(obj_y, 0) for pf in self.pareto_front]
        labels = [pf["config_id"] for pf in self.pareto_front]

        return {"x": x, "y": y, "labels": labels, "x_label": obj_x, "y_label": obj_y}


# =============================================================================
# 7. INTEGRATED GRID SEARCH WITH BACKTEST
# =============================================================================

class IntegratedGridSearch:
    """
    Combines grid search with backtesting and multi-objective optimization.

    This is the main class for finding optimal trading configurations.
    """

    def __init__(
        self,
        grid_search: PipelineGridSearch = None,
        optimizer: MultiObjectiveOptimizer = None,
        max_configs: int = 100,
        early_stopping_patience: int = 20,
    ):
        self.grid_search = grid_search or PipelineGridSearch(
            search_mode="smart",
            max_configs=max_configs,
        )
        self.optimizer = optimizer or MultiObjectiveOptimizer()
        self.max_configs = max_configs
        self.early_stopping_patience = early_stopping_patience

        self.best_scalarized_score = -np.inf
        self.no_improvement_count = 0
        self.search_history: List[Dict] = []

    def run_search(
        self,
        train_fn,  # Function(config) -> (models, scaler, feature_cols)
        backtest_fn,  # Function(models, scaler, feature_cols, config) -> backtest_results
        progress_callback=None,
    ) -> Dict:
        """
        Run integrated grid search with backtesting.

        Args:
            train_fn: Function that trains models given a config
            backtest_fn: Function that runs backtest given models and config
            progress_callback: Optional callback(iteration, config, results)

        Returns:
            Dict with search results
        """
        print("[INTEGRATED SEARCH] Starting multi-objective optimization...")
        print(f"  Max configs: {self.max_configs}")
        print(f"  Objectives: {self.optimizer.objectives}")

        for i, config in enumerate(self.grid_search.iterate_all_configs()):
            if i >= self.max_configs:
                break

            print(f"\n  Config {i+1}/{self.max_configs} [{config.config_id}]")

            try:
                # Train models
                models, scaler, feature_cols = train_fn(config.config)

                # Run backtest
                backtest_results = backtest_fn(models, scaler, feature_cols, config.config)

                # Evaluate objectives
                objectives = self.optimizer.evaluate_config(config, backtest_results)

                # Update Pareto front
                is_pareto = self.optimizer.update_pareto_front(config, objectives)

                # Record result
                self.grid_search.record_result(
                    config,
                    {**backtest_results, "objectives": objectives},
                    primary_metric="sharpe_ratio",
                )

                # Check for improvement
                scalarized = self.optimizer.scalarize(objectives)
                if scalarized > self.best_scalarized_score:
                    self.best_scalarized_score = scalarized
                    self.no_improvement_count = 0
                    print(f"    NEW BEST: Scalarized={scalarized:.4f}")
                else:
                    self.no_improvement_count += 1

                # Log
                self.search_history.append({
                    "iteration": i,
                    "config_id": config.config_id,
                    "objectives": objectives,
                    "scalarized": scalarized,
                    "is_pareto": is_pareto,
                })

                # Progress callback
                if progress_callback:
                    progress_callback(i, config, backtest_results)

                # Early stopping
                if self.no_improvement_count >= self.early_stopping_patience:
                    print(f"\n  Early stopping after {self.early_stopping_patience} iterations without improvement")
                    break

            except Exception as e:
                print(f"    FAILED: {e}")
                continue

        # Final results
        return self._compile_results()

    def _compile_results(self) -> Dict:
        """Compile search results."""
        pareto_summary = self.optimizer.get_pareto_summary()
        best = self.optimizer.get_best_by_scalarization()

        return {
            "n_evaluated": len(self.search_history),
            "pareto_front_size": len(self.optimizer.pareto_front),
            "pareto_summary": pareto_summary.to_dict() if not pareto_summary.empty else {},
            "best_config": best,
            "best_scalarized_score": self.best_scalarized_score,
            "search_history": self.search_history,
            "objective_weights": self.optimizer.objective_weights,
        }


# =============================================================================
# MAIN - Testing
# =============================================================================

if __name__ == "__main__":
    print("Pipeline Grid Configuration System")
    print("=" * 60)

    # Count total configurations
    grid = PipelineGridSearch(search_mode="random", max_configs=10)
    total = grid.count_total_configs()
    print(f"\nTotal possible configurations: {total:,}")

    # Show dimension counts
    print("\nGrid dimensions:")
    for category, params in grid.dimensions.items():
        n_params = len(params)
        n_values = sum(len(v) for v in params.values())
        combos = 1
        for v in params.values():
            combos *= len(v)
        print(f"  {category}: {n_params} params, {n_values} values, {combos:,} combos")

    # Generate sample configs
    print("\nSample configurations:")
    for i, config in enumerate(grid.iterate_all_configs()):
        if i >= 3:
            break
        print(f"\n  Config {config.config_id}:")
        print(f"    Swing model: {config.get('training', {}).get('swing_model_type', 'N/A')}")
        print(f"    Dim reduction MI features: {config.get('dim_reduction', {}).get('mi_n_features', 'N/A')}")
        print(f"    Entry batches: {config.get('entry_exit', {}).get('n_entry_batches', 'N/A')}")

    # Test entry/exit predictor
    print("\n" + "=" * 60)
    print("Entry/Exit Predictor Test")

    preset = QuickPresets.conservative()
    predictor = EntryExitPredictor(preset)

    decision = predictor.predict(
        swing_proba=0.72,
        timing_proba=0.65,
        current_price=450.0,
        volatility=0.015,
        hour=10,
        minute=15,
    )

    print(f"\nSwing: 72%, Timing: 65%")
    print(f"  Action: {decision['action']}")
    print(f"  Position Size: {decision['position_size_pct']*100:.1f}%")
    print(f"  Entry Window: {decision['entry_window']}")
    print(f"  Stop Loss: ${decision['stop_loss']:.2f}" if decision['stop_loss'] else "  Stop Loss: None")
    print(f"  Batch Schedule: {len(decision['batch_schedule'])} batches")
    for batch in decision['batch_schedule']:
        print(f"    Batch {batch['batch_num']}: +{batch['time_offset_minutes']}min, {batch['size_pct']*100:.1f}%")

    print("\nPipeline Grid module loaded successfully!")
