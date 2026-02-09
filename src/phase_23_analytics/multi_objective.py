"""
GIGA TRADER - Multi-Objective Optimization & Integrated Grid Search
====================================================================
Multi-objective optimization for finding Pareto-optimal configurations,
entry/exit prediction, and integrated grid search with backtesting.

Usage:
    from src.phase_23_analytics.multi_objective import (
        MultiObjectiveOptimizer,
        EntryExitPredictor,
        IntegratedGridSearch,
    )
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

from src.phase_23_analytics.grid_config import GridConfig
from src.phase_23_analytics.grid_search import PipelineGridSearch, QuickPresets


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
