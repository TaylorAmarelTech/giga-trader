"""
GIGA TRADER - Experiment Tracking
===================================
Data structures and utilities for tracking experiments, models, and their performance.

Includes:
  - ExperimentStatus enum
  - ExperimentResult dataclass
  - ModelRecord dataclass
  - ExperimentGenerator class
  - ExperimentHistory class
  - ModelRegistry class
  - compute_realistic_backtest_metrics() function
  - calibrate_probabilities() function
"""

import json
import random
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict, fields
from enum import Enum

import numpy as np

from src.experiment_config import (
    ExperimentConfig,
    create_default_config,
    create_experiment_variant,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
project_root = Path(__file__).parent.parent.parent

ENGINE_CONFIG = {
    "results_dir": project_root / "experiments",
    "models_dir": project_root / "models",
    "max_experiments_per_hour": 20,
    "experiment_history_file": project_root / "experiments" / "experiment_history.json",
    "model_registry_file": project_root / "experiments" / "model_registry.json",
    # Transaction cost settings for realistic backtesting
    "slippage_bps": 5,      # 5 basis points per trade
    "commission_bps": 1,    # 1 basis point per trade
    "min_trade_return": 0.001,  # 0.1% minimum to cover costs
    # Walk-forward validation settings
    "purge_days": 5,        # Days to purge between train/test
    "embargo_days": 2,      # Days to embargo after test
    # Leak-proof CV (recommended - fixes data leakage)
    "use_leak_proof_cv": True,  # Use leak-proof pipeline
    "use_model_ensemble": True,  # Ensemble models for reduced overfitting
}


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_realistic_backtest_metrics(
    signals: np.ndarray,
    returns: np.ndarray,
    dates: np.ndarray = None,
    slippage_bps: float = 5,
    commission_bps: float = 1,
) -> Dict[str, float]:
    """
    Compute backtest metrics with realistic transaction costs.

    Args:
        signals: Binary trading signals (0=no trade, 1=trade)
        returns: Daily returns for each position
        dates: Dates for each position
        slippage_bps: Slippage in basis points per trade
        commission_bps: Commission in basis points per trade

    Returns:
        Dict with realistic metrics
    """
    if len(signals) == 0 or signals.sum() == 0:
        return {
            "win_rate": 0.0,
            "win_rate_net": 0.0,
            "total_return": 0.0,
            "total_return_net": 0.0,
            "sharpe": 0.0,
            "sharpe_net": 0.0,
            "max_drawdown": 0.0,
            "n_trades": 0,
            "avg_trade_return": 0.0,
            "avg_trade_return_net": 0.0,
        }

    # Transaction costs per trade (entry + exit = 2x)
    total_cost_per_trade = 2 * (slippage_bps + commission_bps) / 10000

    # Gross returns (no costs)
    gross_returns = signals * returns
    n_trades = (signals > 0).sum()

    # Net returns (after costs)
    # Subtract transaction costs for each trade
    net_returns = gross_returns.copy()
    trade_mask = signals > 0
    net_returns[trade_mask] = net_returns[trade_mask] - total_cost_per_trade

    # Win rate calculations
    wins_gross = (gross_returns > 0).sum()
    wins_net = (net_returns > 0).sum()  # After costs, fewer trades are profitable

    win_rate_gross = wins_gross / n_trades if n_trades > 0 else 0
    win_rate_net = wins_net / n_trades if n_trades > 0 else 0

    # Return calculations
    total_return_gross = np.sum(gross_returns)
    total_return_net = np.sum(net_returns)

    avg_trade_return_gross = total_return_gross / n_trades if n_trades > 0 else 0
    avg_trade_return_net = total_return_net / n_trades if n_trades > 0 else 0

    # Sharpe calculations
    sharpe_gross = 0.0
    sharpe_net = 0.0
    if np.std(gross_returns) > 0:
        sharpe_gross = np.mean(gross_returns) / np.std(gross_returns) * np.sqrt(252)
    if np.std(net_returns) > 0:
        sharpe_net = np.mean(net_returns) / np.std(net_returns) * np.sqrt(252)

    # Max drawdown (on net returns)
    cumulative = np.cumsum(net_returns)
    peak = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - peak) / (peak + 1e-10)
    max_drawdown = float(np.min(drawdown)) if len(drawdown) > 0 else 0

    return {
        "win_rate": float(win_rate_gross),
        "win_rate_net": float(win_rate_net),
        "total_return": float(total_return_gross),
        "total_return_net": float(total_return_net),
        "sharpe": float(sharpe_gross),
        "sharpe_net": float(sharpe_net),
        "max_drawdown": float(max_drawdown),
        "n_trades": int(n_trades),
        "avg_trade_return": float(avg_trade_return_gross),
        "avg_trade_return_net": float(avg_trade_return_net),
        "transaction_cost_per_trade": float(total_cost_per_trade),
    }


def calibrate_probabilities(
    probabilities: np.ndarray,
    min_prob: float = 0.05,
    max_prob: float = 0.95,
    confidence_penalty: float = 0.1,
) -> np.ndarray:
    """
    Calibrate probabilities to prevent overconfident predictions.

    ANTI-OVERFITTING MEASURE:
    - Clips extreme probabilities (overconfidence indicator)
    - Applies confidence penalty that shrinks predictions toward 0.5
    - Prevents models from being "too sure" about noisy predictions

    Args:
        probabilities: Raw probability predictions from model
        min_prob: Minimum allowed probability (default 0.05)
        max_prob: Maximum allowed probability (default 0.95)
        confidence_penalty: Shrinkage toward 0.5 (0.0 = no shrink, 1.0 = all 0.5)

    Returns:
        Calibrated probabilities

    Example:
        If confidence_penalty = 0.1:
        - prob 0.90 -> 0.90 * 0.9 + 0.5 * 0.1 = 0.86
        - prob 0.99 -> clipped to 0.95 first, then 0.95 * 0.9 + 0.5 * 0.1 = 0.905
    """
    proba = np.asarray(probabilities).copy()

    # Step 1: Clip extreme values (overconfidence)
    proba = np.clip(proba, min_prob, max_prob)

    # Step 2: Apply confidence penalty (shrink toward 0.5)
    if confidence_penalty > 0:
        proba = proba * (1 - confidence_penalty) + 0.5 * confidence_penalty

    return proba


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════
class ExperimentStatus(Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ExperimentResult:
    """Results from running an experiment."""
    experiment_id: str
    config: ExperimentConfig
    status: ExperimentStatus = ExperimentStatus.QUEUED
    started_at: str = ""
    completed_at: str = ""

    # Training results
    cv_scores: List[float] = field(default_factory=list)
    cv_auc_mean: float = 0.0
    cv_auc_std: float = 0.0
    train_auc: float = 0.0
    test_auc: float = 0.0

    # Backtest results (NET - after transaction costs)
    backtest_sharpe: float = 0.0           # Net Sharpe (after costs)
    backtest_win_rate: float = 0.0         # Net win rate (after costs)
    backtest_total_return: float = 0.0     # Net return (after costs)
    backtest_max_drawdown: float = 0.0
    # Backtest results (GROSS - before transaction costs)
    backtest_sharpe_gross: float = 0.0
    backtest_win_rate_gross: float = 0.0
    backtest_total_return_gross: float = 0.0
    n_trades: int = 0
    transaction_cost_per_trade: float = 0.0

    # Anti-overfit metrics
    wmes_score: float = 0.0
    stability_score: float = 0.0
    fragility_score: float = 0.0

    # Metadata
    n_features_initial: int = 0
    n_features_final: int = 0
    n_samples_real: int = 0
    n_samples_synthetic: int = 0
    duration_seconds: float = 0.0
    model_path: str = ""
    error_message: str = ""

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["config"] = self.config.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "ExperimentResult":
        d["status"] = ExperimentStatus(d["status"])
        d["config"] = ExperimentConfig.from_dict(d["config"])
        return cls(**d)


@dataclass
class ModelRecord:
    """Record of a trained model and its performance."""
    model_id: str
    experiment_id: str
    created_at: str
    model_path: str
    config: Dict

    # Performance metrics
    cv_auc: float = 0.0
    test_auc: float = 0.0
    backtest_sharpe: float = 0.0
    backtest_win_rate: float = 0.0
    backtest_total_return: float = 0.0
    wmes_score: float = 0.0

    # Robustness metrics (from multi-tier gates)
    stability_score: float = 0.0   # 0-1, higher = more stable (HP plateau)
    fragility_score: float = 1.0   # 0-1, lower = more robust (dim/param perturbation)
    train_test_gap: float = 0.0    # train_auc - test_auc
    tier: int = 1                  # 1=registry, 2=paper-eligible, 3=live-eligible

    # Live performance (from paper trading)
    live_trades: int = 0
    live_win_rate: float = 0.0
    live_total_return: float = 0.0
    live_sharpe: float = 0.0

    def __post_init__(self):
        """Handle unknown fields from future JSON versions."""
        pass

    def score(self, weights: Dict = None) -> float:
        """Calculate weighted score for ranking, with stability bonus."""
        weights = weights or {
            "cv_auc": 0.15,
            "backtest_sharpe": 0.25,
            "wmes_score": 0.20,
            "live_sharpe": 0.25,
            "stability_score": 0.15,
        }
        base = sum(
            weights.get(k, 0) * getattr(self, k, 0)
            for k in weights
        )
        # Penalize fragile models (fragility 0=robust, 1=fragile)
        fragility_penalty = self.fragility_score * 0.1
        return base - fragility_penalty


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════
class ExperimentGenerator:
    """
    Generates new experiment configurations.

    Creates variants of the base config for systematic exploration.
    """

    def __init__(self):
        self.base_config = create_default_config("base")
        self.best_configs: List[ExperimentConfig] = []

        # Experiment type weights
        # ANTI-OVERFITTING: Weight regularization and ensemble higher
        self.experiment_weights = {
            "hyperparameter": 0.20,  # Reduced from 0.30
            "feature_subset": 0.15,  # Reduced from 0.20
            "dim_reduction": 0.15,
            "regularization": 0.25,  # Increased from 0.15 - key for anti-overfit
            "ensemble": 0.15,  # Increased from 0.10 - diverse ensemble helps
            "threshold": 0.10,
        }

    def generate_next(self) -> ExperimentConfig:
        """Generate next experiment configuration."""
        # Choose experiment type
        exp_type = random.choices(
            list(self.experiment_weights.keys()),
            weights=list(self.experiment_weights.values())
        )[0]

        # Use best config as base if available
        if self.best_configs and random.random() < 0.7:
            base = random.choice(self.best_configs)
        else:
            base = self.base_config

        # Generate variant based on type
        # ANTI-OVERFITTING: Use aggressive regularization across all experiment types
        if exp_type == "hyperparameter":
            # AGGRESSIVE: C range 0.001-0.5 (was 0.01-10.0), fewer/shallower trees
            config = create_experiment_variant(base, exp_type,
                l2_C=random.uniform(0.001, 0.5),  # 10x stronger regularization
                gb_n_estimators=random.randint(30, 100),  # Fewer trees
                gb_max_depth=random.randint(2, 4),  # Shallower (max 4)
                gb_learning_rate=random.uniform(0.03, 0.15),  # Lower LR
            )

        elif exp_type == "feature_subset":
            config = create_experiment_variant(base, exp_type,
                use_premarket_features=random.choice([True, False]),
                use_afterhours_features=random.choice([True, False]),
                use_pattern_recognition=random.choice([True, False]),
                use_feature_interactions=random.choice([True, False]),
            )
            # Ensure at least premarket OR afterhours features
            if not config.feature_engineering.use_premarket_features and not config.feature_engineering.use_afterhours_features:
                config.feature_engineering.use_premarket_features = True

        elif exp_type == "dim_reduction":
            methods = ["pca", "kernel_pca", "ica", "umap", "ensemble_plus"]
            config = create_experiment_variant(base, exp_type,
                method=random.choice(methods),
                target_dimensions=random.randint(25, 50),  # Fewer dimensions
            )

        elif exp_type == "regularization":
            # AGGRESSIVE: Much stronger regularization
            config = create_experiment_variant(base, exp_type,
                regularization=random.choice(["l1", "l2", "elastic_net"]),
                l2_C=random.uniform(0.001, 0.1),  # 100x stronger than before
            )

        elif exp_type == "ensemble":
            # Add diverse_ensemble option - combines models with DIFFERENT regularization
            ensemble_options = [
                ["logistic_l2", "gradient_boosting"],
                ["logistic_l2"],
                ["gradient_boosting"],
            ]
            # 40% chance to use diverse_ensemble for better generalization
            if random.random() < 0.4:
                config = create_experiment_variant(base, exp_type,
                    model_type="diverse_ensemble",
                )
            else:
                config = create_experiment_variant(base, exp_type,
                    ensemble_models=random.choice(ensemble_options),
                )

        elif exp_type == "threshold":
            # Higher entry thresholds = more selective = less overfit signals
            config = create_experiment_variant(base, exp_type,
                entry_threshold=random.uniform(0.60, 0.80),  # Higher (was 0.55-0.75)
                exit_threshold=random.uniform(0.35, 0.50),
                stop_loss_pct=random.uniform(0.005, 0.02),
            )

        else:
            config = create_experiment_variant(base, "hyperparameter")

        # Generate new experiment name
        config.experiment_name = f"{exp_type}_{datetime.now().strftime('%H%M%S')}"
        config.description = f"Auto-generated {exp_type} experiment"

        return config

    def add_best_config(self, config: ExperimentConfig, score: float):
        """Add a well-performing config to the pool."""
        self.best_configs.append(config)
        if len(self.best_configs) > 20:
            self.best_configs = self.best_configs[-20:]


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT HISTORY
# ═══════════════════════════════════════════════════════════════════════════════
class ExperimentHistory:
    """Stores and queries experiment history."""

    def __init__(self, history_path: Path = None):
        self.history_path = history_path or ENGINE_CONFIG["experiment_history_file"]
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        self.results: List[ExperimentResult] = []
        self._load()

    def _load(self):
        """Load history from disk."""
        if self.history_path.exists():
            try:
                with open(self.history_path) as f:
                    data = json.load(f)
                    self.results = [ExperimentResult.from_dict(e) for e in data]
            except Exception as e:
                logging.warning(f"Could not load experiment history: {e}")
                self.results = []

    def _save(self):
        """Save history to disk."""
        data = [r.to_dict() for r in self.results]
        with open(self.history_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def add(self, result: ExperimentResult):
        """Add experiment result to history."""
        self.results.append(result)
        self._save()

    def get_recent(self, n: int = 50) -> List[ExperimentResult]:
        """Get recent experiments."""
        return self.results[-n:]

    def get_by_status(self, status: ExperimentStatus) -> List[ExperimentResult]:
        """Filter by status."""
        return [r for r in self.results if r.status == status]

    def get_statistics(self) -> Dict:
        """Get experiment statistics."""
        if not self.results:
            return {"total": 0}

        completed = [r for r in self.results if r.status == ExperimentStatus.COMPLETED]
        failed = [r for r in self.results if r.status == ExperimentStatus.FAILED]

        return {
            "total": len(self.results),
            "completed": len(completed),
            "failed": len(failed),
            "success_rate": len(completed) / len(self.results) if self.results else 0,
            "avg_duration": np.mean([r.duration_seconds for r in completed]) if completed else 0,
            "avg_test_auc": np.mean([r.test_auc for r in completed]) if completed else 0,
            "avg_wmes": np.mean([r.wmes_score for r in completed]) if completed else 0,
            "best_test_auc": max([r.test_auc for r in completed]) if completed else 0,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════
class ModelRegistry:
    """Tracks all trained models and their performance."""

    def __init__(self, registry_path: Path = None):
        self.registry_path = registry_path or ENGINE_CONFIG["model_registry_file"]
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.models: Dict[str, ModelRecord] = {}
        self._load()

    def _load(self):
        """Load registry from disk (backward-compatible with old JSON)."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path) as f:
                    data = json.load(f)
                    # Filter to only known ModelRecord fields for backward compat
                    known_fields = {f.name for f in fields(ModelRecord)}
                    self.models = {}
                    for k, v in data.items():
                        filtered = {fk: fv for fk, fv in v.items() if fk in known_fields}
                        self.models[k] = ModelRecord(**filtered)
            except Exception as e:
                logging.warning(f"Could not load model registry: {e}")
                self.models = {}

    def _save(self):
        """Save registry to disk."""
        data = {k: asdict(v) for k, v in self.models.items()}
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def _compute_tier(self, result: ExperimentResult) -> int:
        """Compute quality tier based on robustness metrics.

        Tier 1: Registry (AUC + gap + WMES)
        Tier 2: Paper-eligible (Tier 1 + stability_score >= 0.50)
        Tier 3: Live-eligible (Tier 2 + fragility < 0.35 + AUC >= 0.60)
        """
        tier = 1  # Passed registration gate to get here

        # Tier 2: stability verified
        if result.stability_score >= 0.50:
            tier = 2

            # Tier 3: fragility verified + higher AUC
            if result.fragility_score < 0.35 and result.test_auc >= 0.60:
                tier = 3

        return tier

    def register_model(self, result: ExperimentResult) -> str:
        """Register a new model from experiment result."""
        model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        train_test_gap = result.train_auc - result.test_auc if result.train_auc > 0 else 0.0
        tier = self._compute_tier(result)

        record = ModelRecord(
            model_id=model_id,
            experiment_id=result.experiment_id,
            created_at=datetime.now().isoformat(),
            model_path=result.model_path,
            config=result.config.to_dict(),
            cv_auc=result.cv_auc_mean,
            test_auc=result.test_auc,
            backtest_sharpe=result.backtest_sharpe,
            backtest_win_rate=result.backtest_win_rate,
            backtest_total_return=result.backtest_total_return,
            wmes_score=result.wmes_score,
            stability_score=result.stability_score,
            fragility_score=result.fragility_score,
            train_test_gap=train_test_gap,
            tier=tier,
        )

        self.models[model_id] = record
        self._save()

        return model_id

    def get_top_models(self, n: int = 10) -> List[ModelRecord]:
        """Get top N models by score."""
        models = list(self.models.values())
        models.sort(key=lambda m: m.score(), reverse=True)
        return models[:n]

    def get_statistics(self) -> Dict:
        """Get registry statistics."""
        if not self.models:
            return {"total_models": 0}

        records = list(self.models.values())
        return {
            "total_models": len(records),
            "avg_cv_auc": np.mean([r.cv_auc for r in records]),
            "avg_backtest_sharpe": np.mean([r.backtest_sharpe for r in records]),
            "best_cv_auc": max(r.cv_auc for r in records),
            "best_backtest_sharpe": max(r.backtest_sharpe for r in records),
            "best_wmes": max(r.wmes_score for r in records),
        }
