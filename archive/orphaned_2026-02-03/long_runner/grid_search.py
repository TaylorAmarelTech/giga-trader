"""
Grid Search Controller
======================
Manages the massive hyperparameter grid search with:
- Intelligent parameter space exploration
- Priority-based experiment scheduling
- Result tracking and analysis
- Bayesian-inspired next-point selection
"""

import json
import logging
import hashlib
import random
import math
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Iterator, Tuple
from enum import Enum
from collections import defaultdict
import itertools

logger = logging.getLogger("GigaTrader.GridSearch")


class ExperimentStatus(Enum):
    """Status of an experiment configuration."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ParameterSpec:
    """Specification for a single parameter."""
    name: str
    param_type: str  # "continuous", "discrete", "categorical"
    values: List[Any] = field(default_factory=list)
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    step: Optional[float] = None
    log_scale: bool = False
    description: str = ""

    def get_values(self) -> List[Any]:
        """Get all possible values for this parameter."""
        if self.values:
            return self.values

        if self.param_type == "continuous":
            # Generate grid points
            if self.log_scale:
                # Logarithmic scale
                log_min = math.log10(self.min_val)
                log_max = math.log10(self.max_val)
                n_points = int((log_max - log_min) / (self.step or 0.1)) + 1
                return [10 ** (log_min + i * (self.step or 0.1)) for i in range(n_points)]
            else:
                # Linear scale
                n_points = int((self.max_val - self.min_val) / (self.step or 1)) + 1
                return [self.min_val + i * (self.step or 1) for i in range(n_points)]

        if self.param_type == "discrete":
            return list(range(int(self.min_val), int(self.max_val) + 1, int(self.step or 1)))

        return self.values

    def sample_random(self) -> Any:
        """Sample a random value from this parameter."""
        if self.values:
            return random.choice(self.values)

        if self.param_type == "continuous":
            if self.log_scale:
                log_val = random.uniform(math.log10(self.min_val), math.log10(self.max_val))
                return 10 ** log_val
            return random.uniform(self.min_val, self.max_val)

        if self.param_type == "discrete":
            return random.randint(int(self.min_val), int(self.max_val))

        return random.choice(self.get_values())


@dataclass
class ExperimentConfig:
    """A single experiment configuration."""
    config_id: str
    parameters: Dict[str, Any]
    status: ExperimentStatus = ExperimentStatus.PENDING
    priority: float = 0.0  # Higher = more important
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    results: Dict[str, Any] = field(default_factory=dict)
    error_message: str = ""
    parent_config_id: Optional[str] = None  # For Bayesian-inspired refinement

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["created_at"] = self.created_at.isoformat()
        d["started_at"] = self.started_at.isoformat() if self.started_at else None
        d["completed_at"] = self.completed_at.isoformat() if self.completed_at else None
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "ExperimentConfig":
        d = d.copy()
        d["status"] = ExperimentStatus(d["status"])
        d["created_at"] = datetime.fromisoformat(d["created_at"])
        d["started_at"] = datetime.fromisoformat(d["started_at"]) if d.get("started_at") else None
        d["completed_at"] = datetime.fromisoformat(d["completed_at"]) if d.get("completed_at") else None
        return cls(**d)


class ParameterGrid:
    """
    Defines the parameter search space for experiments.
    """

    # Default trading model parameter grid
    DEFAULT_GRID = {
        # Regularization parameters
        "l2_C": ParameterSpec(
            name="l2_C",
            param_type="continuous",
            min_val=0.001,
            max_val=100.0,
            step=0.1,
            log_scale=True,
            description="L2 regularization strength (inverse)",
        ),
        "l1_ratio": ParameterSpec(
            name="l1_ratio",
            param_type="continuous",
            min_val=0.0,
            max_val=1.0,
            step=0.1,
            description="L1/L2 ratio for Elastic Net",
        ),

        # Gradient Boosting parameters
        "gb_n_estimators": ParameterSpec(
            name="gb_n_estimators",
            param_type="discrete",
            min_val=30,
            max_val=200,
            step=10,
            description="Number of boosting iterations",
        ),
        "gb_max_depth": ParameterSpec(
            name="gb_max_depth",
            param_type="discrete",
            min_val=2,
            max_val=5,  # EDGE 1: max_depth <= 5
            step=1,
            description="Maximum tree depth",
        ),
        "gb_learning_rate": ParameterSpec(
            name="gb_learning_rate",
            param_type="continuous",
            min_val=0.01,
            max_val=0.3,
            step=0.01,
            log_scale=True,
            description="Boosting learning rate",
        ),
        "gb_min_samples_leaf": ParameterSpec(
            name="gb_min_samples_leaf",
            param_type="discrete",
            min_val=10,
            max_val=100,
            step=10,
            description="Minimum samples per leaf",
        ),
        "gb_subsample": ParameterSpec(
            name="gb_subsample",
            param_type="continuous",
            min_val=0.5,
            max_val=1.0,
            step=0.1,
            description="Subsample ratio for stochastic GB",
        ),

        # Dimensionality reduction
        "dim_reduction_method": ParameterSpec(
            name="dim_reduction_method",
            param_type="categorical",
            values=["ensemble_plus", "mutual_info", "kernel_pca", "umap", "ica"],
            description="Dimensionality reduction method",
        ),
        "n_components": ParameterSpec(
            name="n_components",
            param_type="discrete",
            min_val=20,
            max_val=60,
            step=5,
            description="Number of components to keep",
        ),

        # Anti-overfit settings
        "synthetic_weight": ParameterSpec(
            name="synthetic_weight",
            param_type="continuous",
            min_val=0.1,
            max_val=0.5,
            step=0.05,
            description="Weight for synthetic data",
        ),
        "cv_folds": ParameterSpec(
            name="cv_folds",
            param_type="discrete",
            min_val=3,
            max_val=7,
            step=1,
            description="Number of CV folds",
        ),

        # Signal thresholds
        "swing_threshold": ParameterSpec(
            name="swing_threshold",
            param_type="continuous",
            min_val=0.45,
            max_val=0.70,
            step=0.025,
            description="Swing signal threshold",
        ),
        "timing_threshold": ParameterSpec(
            name="timing_threshold",
            param_type="continuous",
            min_val=0.45,
            max_val=0.70,
            step=0.025,
            description="Timing signal threshold",
        ),

        # Target creation
        "target_threshold": ParameterSpec(
            name="target_threshold",
            param_type="continuous",
            min_val=0.001,
            max_val=0.010,
            step=0.001,
            description="Return threshold for positive label",
        ),
        "soft_label_k": ParameterSpec(
            name="soft_label_k",
            param_type="discrete",
            min_val=20,
            max_val=100,
            step=10,
            description="Soft label sigmoid steepness",
        ),

        # Feature engineering
        "use_extended_hours": ParameterSpec(
            name="use_extended_hours",
            param_type="categorical",
            values=[True, False],
            description="Include premarket/afterhours features",
        ),
        "use_cross_assets": ParameterSpec(
            name="use_cross_assets",
            param_type="categorical",
            values=[True, False],
            description="Include cross-asset features",
        ),
        "use_breadth_streaks": ParameterSpec(
            name="use_breadth_streaks",
            param_type="categorical",
            values=[True, False],
            description="Include breadth streak features",
        ),
    }

    def __init__(self, custom_specs: Optional[Dict[str, ParameterSpec]] = None):
        self.specs = self.DEFAULT_GRID.copy()
        if custom_specs:
            self.specs.update(custom_specs)

    def get_total_combinations(self) -> int:
        """Calculate total number of grid combinations."""
        total = 1
        for spec in self.specs.values():
            total *= len(spec.get_values())
        return total

    def generate_all_combinations(self) -> Iterator[Dict[str, Any]]:
        """Generate all parameter combinations (use with caution for large grids!)."""
        keys = list(self.specs.keys())
        value_lists = [self.specs[k].get_values() for k in keys]

        for values in itertools.product(*value_lists):
            yield dict(zip(keys, values))

    def generate_random_sample(self, n: int = 100) -> List[Dict[str, Any]]:
        """Generate n random parameter combinations."""
        samples = []
        for _ in range(n):
            sample = {
                name: spec.sample_random()
                for name, spec in self.specs.items()
            }
            samples.append(sample)
        return samples

    def generate_around_point(
        self,
        center: Dict[str, Any],
        n_samples: int = 10,
        radius: float = 0.2,
    ) -> List[Dict[str, Any]]:
        """
        Generate samples around a center point (for Bayesian-style refinement).

        Args:
            center: The center point parameters
            n_samples: Number of samples to generate
            radius: Radius as fraction of parameter range

        Returns:
            List of parameter combinations near the center
        """
        samples = []

        for _ in range(n_samples):
            sample = {}
            for name, spec in self.specs.items():
                center_val = center.get(name)

                if spec.param_type == "categorical":
                    # Small chance to change categorical
                    if random.random() < 0.2:
                        sample[name] = spec.sample_random()
                    else:
                        sample[name] = center_val

                elif spec.param_type in ["continuous", "discrete"]:
                    # Perturb within radius
                    range_size = spec.max_val - spec.min_val
                    perturbation = random.uniform(-radius, radius) * range_size

                    new_val = center_val + perturbation
                    new_val = max(spec.min_val, min(spec.max_val, new_val))

                    if spec.param_type == "discrete":
                        new_val = round(new_val)

                    sample[name] = new_val
                else:
                    sample[name] = center_val

            samples.append(sample)

        return samples


class GridSearchController:
    """
    Controls the grid search process with intelligent exploration.

    Features:
    - Tracks completed experiments
    - Prioritizes promising regions
    - Supports resumption from saved state
    - Bayesian-inspired next-point selection
    """

    def __init__(
        self,
        grid: ParameterGrid,
        state_file: Path,
        results_dir: Path,
        exploration_ratio: float = 0.3,  # Fraction of random exploration
    ):
        self.grid = grid
        self.state_file = state_file
        self.results_dir = results_dir
        self.exploration_ratio = exploration_ratio

        # Experiment tracking
        self.experiments: Dict[str, ExperimentConfig] = {}
        self.pending_queue: List[str] = []
        self.completed_ids: set = set()
        self.failed_ids: set = set()

        # Performance tracking
        self.best_experiments: List[Tuple[float, str]] = []  # (score, config_id)
        self.parameter_performance: Dict[str, Dict[Any, List[float]]] = defaultdict(lambda: defaultdict(list))

        # Load existing state
        self._load_state()

        # Ensure directories exist
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def get_next_experiment(self) -> Optional[ExperimentConfig]:
        """
        Get the next experiment to run.

        Uses a combination of:
        - Random exploration (exploration_ratio)
        - Exploitation around best points
        - Priority-based selection from queue
        """
        # Check if we have pending experiments
        if self.pending_queue:
            config_id = self.pending_queue.pop(0)
            exp = self.experiments.get(config_id)
            if exp and exp.status == ExperimentStatus.PENDING:
                exp.status = ExperimentStatus.RUNNING
                exp.started_at = datetime.now()
                self._save_state()
                return exp

        # Decide: explore or exploit
        if random.random() < self.exploration_ratio or not self.best_experiments:
            # EXPLORE: Random sample
            params = self.grid.generate_random_sample(1)[0]
            logger.info("Generating random exploration experiment")
        else:
            # EXPLOIT: Sample around best point
            _, best_id = max(self.best_experiments[-10:])  # Top 10
            best_exp = self.experiments.get(best_id)
            if best_exp:
                params = self.grid.generate_around_point(best_exp.parameters, n_samples=1)[0]
                logger.info(f"Generating exploitation experiment around {best_id}")
            else:
                params = self.grid.generate_random_sample(1)[0]

        # Create experiment
        exp = self._create_experiment(params)
        exp.status = ExperimentStatus.RUNNING
        exp.started_at = datetime.now()
        self._save_state()
        return exp

    def add_experiments(self, configs: List[Dict[str, Any]], priority: float = 0.0):
        """Add multiple experiment configurations to the queue."""
        for params in configs:
            exp = self._create_experiment(params, priority=priority)
            self.pending_queue.append(exp.config_id)
        self._save_state()
        logger.info(f"Added {len(configs)} experiments to queue")

    def mark_completed(
        self,
        config_id: str,
        results: Dict[str, Any],
        success: bool = True,
    ):
        """Mark an experiment as completed."""
        exp = self.experiments.get(config_id)
        if not exp:
            logger.warning(f"Unknown experiment: {config_id}")
            return

        exp.completed_at = datetime.now()
        exp.results = results

        if success:
            exp.status = ExperimentStatus.COMPLETED
            self.completed_ids.add(config_id)

            # Track performance
            score = self._calculate_score(results)
            self.best_experiments.append((score, config_id))
            self.best_experiments.sort(reverse=True)
            self.best_experiments = self.best_experiments[:100]  # Keep top 100

            # Track parameter performance
            for param_name, param_value in exp.parameters.items():
                self.parameter_performance[param_name][param_value].append(score)

            logger.info(f"Experiment {config_id} completed with score {score:.4f}")
        else:
            exp.status = ExperimentStatus.FAILED
            exp.error_message = results.get("error", "Unknown error")
            self.failed_ids.add(config_id)
            logger.warning(f"Experiment {config_id} failed: {exp.error_message}")

        # Save results
        self._save_experiment_result(exp)
        self._save_state()

    def get_stats(self) -> Dict[str, Any]:
        """Get grid search statistics."""
        total = len(self.experiments)
        completed = len(self.completed_ids)
        failed = len(self.failed_ids)
        pending = len(self.pending_queue)
        running = total - completed - failed - pending

        # Calculate estimated total combinations
        total_combinations = self.grid.get_total_combinations()

        # Get best score
        best_score = self.best_experiments[0][0] if self.best_experiments else 0.0

        # Parameter analysis
        best_params = {}
        for param_name, value_scores in self.parameter_performance.items():
            if value_scores:
                best_value = max(value_scores.keys(), key=lambda v: sum(value_scores[v]) / len(value_scores[v]) if value_scores[v] else 0)
                best_params[param_name] = {
                    "best_value": best_value,
                    "avg_score": sum(value_scores[best_value]) / len(value_scores[best_value]) if value_scores[best_value] else 0,
                }

        return {
            "total_experiments": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "running": running,
            "success_rate": completed / total if total > 0 else 0.0,
            "total_possible_combinations": total_combinations,
            "coverage": completed / total_combinations if total_combinations > 0 else 0.0,
            "best_score": best_score,
            "best_experiment_id": self.best_experiments[0][1] if self.best_experiments else None,
            "best_parameters": best_params,
            "exploration_ratio": self.exploration_ratio,
        }

    def get_best_experiment(self) -> Optional[ExperimentConfig]:
        """Get the best performing experiment."""
        if not self.best_experiments:
            return None
        _, best_id = self.best_experiments[0]
        return self.experiments.get(best_id)

    def _create_experiment(
        self,
        params: Dict[str, Any],
        priority: float = 0.0,
        parent_id: Optional[str] = None,
    ) -> ExperimentConfig:
        """Create a new experiment configuration."""
        # Generate unique ID from parameters
        param_str = json.dumps(params, sort_keys=True)
        config_id = f"exp_{hashlib.md5(param_str.encode()).hexdigest()[:12]}"

        # Check if already exists
        if config_id in self.experiments:
            return self.experiments[config_id]

        exp = ExperimentConfig(
            config_id=config_id,
            parameters=params,
            priority=priority,
            parent_config_id=parent_id,
        )

        self.experiments[config_id] = exp
        return exp

    def _calculate_score(self, results: Dict[str, Any]) -> float:
        """
        Calculate a composite score from experiment results.

        Weights:
        - Test AUC: 30%
        - Backtest Sharpe: 25%
        - Win Rate: 20%
        - WMES Score: 15%
        - Low Train-Test Gap: 10%
        """
        weights = {
            "test_auc": 0.30,
            "backtest_sharpe": 0.25,
            "win_rate": 0.20,
            "wmes_score": 0.15,
            "gap_penalty": 0.10,
        }

        # Normalize metrics to 0-1 scale
        test_auc = min(1.0, max(0.0, results.get("test_auc", 0.5)))
        sharpe = min(1.0, max(0.0, results.get("backtest_sharpe", 0.0) / 3.0))  # Normalize to ~1.0 at Sharpe 3
        win_rate = min(1.0, max(0.0, results.get("win_rate", 0.5)))
        wmes = min(1.0, max(0.0, results.get("wmes_score", 0.5)))

        # Gap penalty (lower gap is better)
        gap = abs(results.get("train_auc", 0.5) - results.get("test_auc", 0.5))
        gap_score = max(0.0, 1.0 - gap * 10)  # 10% gap = 0 score

        score = (
            weights["test_auc"] * test_auc +
            weights["backtest_sharpe"] * sharpe +
            weights["win_rate"] * win_rate +
            weights["wmes_score"] * wmes +
            weights["gap_penalty"] * gap_score
        )

        return score

    def _save_experiment_result(self, exp: ExperimentConfig):
        """Save individual experiment result to file."""
        result_file = self.results_dir / f"{exp.config_id}.json"
        with open(result_file, "w") as f:
            json.dump(exp.to_dict(), f, indent=2, default=str)

    def _save_state(self):
        """Save grid search state to file."""
        state = {
            "timestamp": datetime.now().isoformat(),
            "pending_queue": self.pending_queue,
            "completed_ids": list(self.completed_ids),
            "failed_ids": list(self.failed_ids),
            "best_experiments": self.best_experiments[:20],
            "exploration_ratio": self.exploration_ratio,
            "stats": self.get_stats(),
        }

        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2, default=str)

    def _load_state(self):
        """Load grid search state from file."""
        if not self.state_file.exists():
            return

        try:
            with open(self.state_file) as f:
                state = json.load(f)

            self.pending_queue = state.get("pending_queue", [])
            self.completed_ids = set(state.get("completed_ids", []))
            self.failed_ids = set(state.get("failed_ids", []))
            self.best_experiments = state.get("best_experiments", [])
            self.exploration_ratio = state.get("exploration_ratio", 0.3)

            # Load experiment results
            for result_file in self.results_dir.glob("exp_*.json"):
                with open(result_file) as f:
                    exp_data = json.load(f)
                    exp = ExperimentConfig.from_dict(exp_data)
                    self.experiments[exp.config_id] = exp

            logger.info(f"Loaded state: {len(self.experiments)} experiments, {len(self.completed_ids)} completed")

        except Exception as e:
            logger.error(f"Failed to load state: {e}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Display grid search controller status.

    IMPORTANT: This function does NOT run experiments with fake results.
    To run actual grid search experiments, use:
        python scripts/run_grid_search.py

    This main() only shows current grid search statistics.
    """
    from pathlib import Path

    state_file = Path(__file__).parent.parent.parent / "logs" / "grid_search_state.json"
    results_dir = Path(__file__).parent.parent.parent / "experiments" / "grid_results"

    grid = ParameterGrid()
    controller = GridSearchController(
        grid=grid,
        state_file=state_file,
        results_dir=results_dir,
    )

    print("=" * 70)
    print("GRID SEARCH CONTROLLER STATUS")
    print("=" * 70)
    print(f"\nTotal possible combinations: {grid.get_total_combinations():,}")
    print(f"\nCurrent stats:")
    print(json.dumps(controller.get_stats(), indent=2))

    # Show best experiment if any
    best = controller.get_best_experiment()
    if best:
        print(f"\nBest experiment: {best.config_id}")
        print(f"  Test AUC: {best.results.get('test_auc', 'N/A')}")
        print(f"  Backtest Sharpe: {best.results.get('backtest_sharpe', 'N/A')}")
    else:
        print("\nNo completed experiments yet.")

    print("\n" + "=" * 70)
    print("To run grid search experiments, use:")
    print("  python scripts/run_grid_search.py")
    print("=" * 70)


if __name__ == "__main__":
    main()
