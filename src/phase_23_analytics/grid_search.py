"""
GIGA TRADER - Pipeline Grid Search Engine
==========================================
Exhaustive grid search across all pipeline configurations.

Supports:
  - Full grid search (all combinations)
  - Random search (sample N configs)
  - Smart search (prioritize based on prior results)
  - Incremental search (resume from checkpoint)

Usage:
    from src.phase_23_analytics.grid_search import PipelineGridSearch, QuickPresets
"""

import json
from itertools import product
from typing import Dict, List, Generator

import numpy as np

from src.phase_23_analytics.grid_config import GridDimensions, GridConfig


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
