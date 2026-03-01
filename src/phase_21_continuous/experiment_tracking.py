"""
GIGA TRADER - Experiment Tracking
===================================
Data structures and utilities for tracking experiments, models, and their performance.

Includes:
  - ExperimentStatus enum
  - ExperimentResult dataclass
  - ExperimentGenerator class
  - ExperimentHistory class (SQLite-backed via RegistryDB)
  - compute_realistic_backtest_metrics() function
  - calibrate_probabilities() function
"""

import random
import hashlib
import logging
import threading
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
    # Transaction cost settings for realistic backtesting
    "slippage_bps": 5,      # 5 basis points per trade
    "commission_bps": 1,    # 1 basis point per trade
    "min_trade_return": 0.001,  # 0.1% minimum to cover costs
    # Walk-forward validation settings
    "purge_days": 10,       # Days to purge between train/test (Wave 25: SPY autocorrelation)
    "embargo_days": 3,      # Days to embargo after test (Wave 25: extra safety margin)
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
            "transaction_cost_per_trade": 0.0,
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

    # Sharpe calculations — computed on TRADING DAYS ONLY to avoid inflation
    # from zero-return non-trading days in the denominator.
    # Subtract daily risk-free rate (~4% annual / 252 trading days) per standard Sharpe definition.
    sharpe_gross = 0.0
    sharpe_net = 0.0
    risk_free_daily = 0.04 / 252  # ~0.000159 per day
    trade_gross = gross_returns[trade_mask]
    trade_net = net_returns[trade_mask]
    if len(trade_gross) > 1 and np.std(trade_gross) > 0:
        # Annualize by estimated trades per year, not calendar days
        trades_per_year = n_trades / max(len(signals), 1) * 252
        excess_gross = np.mean(trade_gross) - risk_free_daily
        sharpe_gross = excess_gross / np.std(trade_gross) * np.sqrt(max(trades_per_year, 1))
    if len(trade_net) > 1 and np.std(trade_net) > 0:
        trades_per_year = n_trades / max(len(signals), 1) * 252
        excess_net = np.mean(trade_net) - risk_free_daily
        sharpe_net = excess_net / np.std(trade_net) * np.sqrt(max(trades_per_year, 1))

    # Max drawdown (on net returns, equity-curve based)
    # Convert cumulative log-returns to equity curve starting at 1.0
    equity = 1.0 + np.cumsum(net_returns)
    peak = np.maximum.accumulate(equity)
    drawdown = (equity - peak) / peak  # Percentage drawdown from peak
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
    permutation_auc_mean: float = 0.0  # Mean AUC on shuffled targets (should be ~0.50)
    permutation_passed: bool = True     # False if permutation test detected leakage

    # Walk-forward validation
    walk_forward_aucs: List[float] = field(default_factory=list)
    worst_window_auc: float = 0.0
    walk_forward_passed: bool = False

    # Regime-specific evaluation
    regime_auc_low_vol: float = 0.0
    regime_auc_high_vol: float = 0.0
    regime_sensitive: bool = False

    # Multi-faceted stability (Wave 29)
    stability_bootstrap: float = 0.0
    stability_feature_dropout: float = 0.0
    stability_seed: float = 0.0
    stability_prediction: float = 0.0
    stability_composite: float = 0.0

    # Advanced stability metrics (Wave 30) — flexible dict for all methods
    # Stores PSI, CSI, adversarial, ECE, DSR, SFI, meta_label, knockoff,
    # ADWIN, CPCV, stability_selection, rashomon, SHAP, conformal, composite
    stability_advanced: Dict[str, Any] = field(default_factory=dict)

    # Wave 35: Training augmentation metrics
    nested_cv_auc: float = 0.0           # Honest nested CV estimate
    distillation_gap: float = 0.0        # AUC gap: teacher - distilled student

    # Wave 40: Meta-labeling metrics
    meta_label_auc: float = 0.0          # Meta-model cross-validated AUC
    meta_label_fitted: bool = False       # Whether meta-model was successfully trained
    meta_sharpe: float = 0.0             # Sharpe ratio after meta-label filtering
    meta_win_rate: float = 0.0           # Win rate after meta-label filtering
    meta_improvement: float = 0.0        # Sharpe improvement vs unfiltered

    # Metadata
    n_features_initial: int = 0
    n_features_final: int = 0
    n_samples_real: int = 0
    n_samples_synthetic: int = 0
    duration_seconds: float = 0.0
    model_path: str = ""
    error_message: str = ""

    # Scoring version (Wave 36: tracks measurement methodology)
    scoring_version: str = ""

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["status"] = self.status.value
        d["config"] = self.config.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: Dict) -> "ExperimentResult":
        d = dict(d)  # Copy to avoid mutating caller
        d["status"] = ExperimentStatus(d["status"])
        d["config"] = ExperimentConfig.from_dict(d["config"])
        # Filter unknown fields for forward compatibility
        valid_keys = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**d)


# ═══════════════════════════════════════════════════════════════════════════════
# EXPERIMENT GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════
class ExperimentGenerator:
    """
    Generates new experiment configurations.

    Creates variants of the base config for systematic exploration.

    Wave 31: Configurable min/max weight bounds prevent any single method,
    model type, or experiment type from monopolizing the search space.
    """

    # ── Configurable weight bounds (Wave 31) ──
    # These prevent overfitting to any single option. After raw weights are
    # set (by hand or by adaptive logic), _clamp_weights() enforces these
    # bounds and re-normalizes so weights sum to 1.0.
    MIN_EXPERIMENT_TYPE_WEIGHT = 0.03   # Every experiment type gets at least 3%
    MAX_EXPERIMENT_TYPE_WEIGHT = 0.25   # No type gets more than 25%
    MIN_DIM_REDUCTION_WEIGHT = 0.05     # Every dim method gets at least 5%
    MAX_DIM_REDUCTION_WEIGHT = 0.25     # No dim method gets more than 25%
    MIN_MODEL_TYPE_WEIGHT = 0.03        # Every model type gets at least 3%
    MAX_MODEL_TYPE_WEIGHT = 0.15        # No model type gets more than 15%

    def __init__(self, history: "ExperimentHistory" = None):
        self.base_config = create_default_config("base")
        self.best_configs: List[ExperimentConfig] = []
        self.history = history  # Used for failure-pattern avoidance

        # Experiment type weights
        # Wave 31: Regime experiments increased — they have
        # 0% failure rate, 100% WF pass rate, and highest stability scores.
        # ensemble type reduced (worst WF pass + WMES corruption source).
        self.experiment_weights = self._clamp_weights({
            "hyperparameter": 0.13,
            "feature_subset": 0.08,
            "dim_reduction": 0.11,
            "regularization": 0.08,
            "ensemble": 0.05,
            "threshold": 0.06,
            "regime_lowvol": 0.15,
            "regime_highvol": 0.15,
            "feature_research": 0.06,   # Wave 32: iterative feature discovery
            "augmented_training": 0.07, # Wave 35: training augmentations
            "data_source": 0.08,        # Wave 38: explore data source combinations
        }, self.MIN_EXPERIMENT_TYPE_WEIGHT, self.MAX_EXPERIMENT_TYPE_WEIGHT)

    @staticmethod
    def _clamp_weights(
        weights: Dict[str, float],
        min_weight: float,
        max_weight: float,
    ) -> Dict[str, float]:
        """Clamp all weights to [min_weight, max_weight] and re-normalize to sum=1.0.

        Ensures no single option can monopolize or be starved out of the search.
        Normalizes first, then iteratively clamps and redistributes until stable.
        """
        # Normalize to sum=1.0 first (handles raw integer weights like 16, 13, etc.)
        w = dict(weights)
        total = sum(w.values())
        if total > 0:
            w = {k: v / total for k, v in w.items()}

        # Iteratively clamp and re-normalize until stable
        for _ in range(10):
            changed = False
            for k in w:
                if w[k] < min_weight:
                    w[k] = min_weight
                    changed = True
                elif w[k] > max_weight:
                    w[k] = max_weight
                    changed = True
            # Re-normalize
            total = sum(w.values())
            if total > 0:
                w = {k: v / total for k, v in w.items()}
            if not changed:
                break
        return w

    def generate_next(self) -> ExperimentConfig:
        """Generate next experiment configuration (with dedup).

        Uses ExperimentHistory._config_hash() for consistent hashing
        across both history storage and novelty checking.
        """
        # Get previously tried configs to avoid duplicates
        tried_hashes = set()
        if self.history is not None:
            try:
                tried_hashes = self.history.get_config_hashes()
            except Exception:
                pass

        # Try up to 10 times to generate a novel config
        for _attempt in range(10):
            config = self._generate_config()
            if not tried_hashes:
                return config
            h = ExperimentHistory._config_hash(config)
            if h is None or h not in tried_hashes:
                return config

        return config  # After 10 attempts, return whatever we have

    def _generate_config(self) -> ExperimentConfig:
        """Generate a single experiment configuration."""
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

        # ── ALWAYS vary foundational dimensions (base layer) ──
        # This ensures ALL experiment types explore diverse dim_reduction
        # methods and model types, not just the 15% that explicitly vary them.
        # Wave 31: All weights are clamped by MIN/MAX bounds to guarantee
        # diversity — no method can monopolize or be starved out.
        _dim_methods = ["pca", "kernel_pca", "ica", "umap", "ensemble_plus", "mutual_info", "agglomeration"]
        _dim_raw = dict(zip(_dim_methods, [16, 13, 17, 11, 11, 16, 16]))
        _dim_clamped = self._clamp_weights(_dim_raw, self.MIN_DIM_REDUCTION_WEIGHT, self.MAX_DIM_REDUCTION_WEIGHT)
        base.dim_reduction.method = random.choices(
            list(_dim_clamped.keys()), weights=list(_dim_clamped.values()))[0]
        base.dim_reduction.feature_selection_method = random.choice(["mutual_info", "f_classif"])
        base.dim_reduction.mi_n_features = random.choice([20, 25, 30, 35, 40])
        base.dim_reduction.target_dimensions = random.randint(15, 50)
        _model_types = [
            "logistic_l2", "logistic_l1", "logistic_elastic",
            "gradient_boosting", "hist_gradient_boosting",
            "ensemble", "diverse_ensemble",
            "mlp_small", "mlp_medium", "sgd_linear",
            "ridge", "extra_trees", "bagged_linear",
            # Wave 35: 5 new complexity baselines
            "lda", "gaussian_nb", "svc_linear", "bayesian_ridge", "quantile_gb",
            # Wave E3: Quantile random forest
            "quantile_forest",
            # Wave F2: CatBoost + Stacking
            "catboost", "stacking_ensemble",
        ]
        _model_raw = dict(zip(_model_types, [
            8, 9, 8,       # logistic variants
            5, 7,           # tree boosting
            6, 7,           # ensembles
            9, 8, 8,        # MLP + SGD
            8, 5, 8,        # ridge, extra_trees, bagged_linear
            7, 7, 7, 7, 8,  # Wave 35: lda, nb, svc, bayesian, quantile_gb
            6,               # Wave E3: quantile_forest
            6, 7,            # Wave F2: catboost, stacking_ensemble
        ]))
        _model_clamped = self._clamp_weights(_model_raw, self.MIN_MODEL_TYPE_WEIGHT, self.MAX_MODEL_TYPE_WEIGHT)
        base.model.model_type = random.choices(
            list(_model_clamped.keys()), weights=list(_model_clamped.values()))[0]

        # Generate variant based on type — applies experiment-specific overrides on top
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
            # Avoid dim_reduction methods that fail >60% of the time (Wave 17)
            if self.history is not None:
                try:
                    patterns = self.history.get_failure_patterns()
                    bad_methods = patterns.get("dim_methods_failing", {})
                    n_fail = patterns.get("n_failures", 0)
                    if n_fail > 10:
                        methods = [
                            m for m in methods
                            if bad_methods.get(m, 0) / max(n_fail, 1) < 0.6
                        ] or methods  # Fallback to all if everything fails
                except Exception:
                    pass
            config = create_experiment_variant(base, exp_type,
                method=random.choice(methods),
                target_dimensions=random.randint(25, 50),  # Fewer dimensions
            )
            # Also vary feature selection method and n_features
            config.dim_reduction.feature_selection_method = random.choice(["mutual_info", "f_classif"])
            config.dim_reduction.mi_n_features = random.choice([20, 25, 30, 35, 40])

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
                ["logistic_l1", "gradient_boosting"],
                ["logistic_l2", "logistic_l1"],
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
                    model_type="ensemble",  # Explicit: ensemble type needs ensemble model_type
                    ensemble_models=random.choice(ensemble_options),
                )

        elif exp_type == "threshold":
            # Higher entry thresholds = more selective = less overfit signals
            config = create_experiment_variant(base, exp_type,
                entry_threshold=random.uniform(0.60, 0.80),  # Higher (was 0.55-0.75)
                exit_threshold=random.uniform(0.35, 0.50),
                stop_loss_pct=random.uniform(0.005, 0.02),
            )

        elif exp_type == "regime_lowvol":
            config = create_experiment_variant(base, exp_type)
            config.data.regime_filter = "low_vol"
            config.description = "Train on low-volatility periods only"

        elif exp_type == "regime_highvol":
            config = create_experiment_variant(base, exp_type)
            config.data.regime_filter = "high_vol"
            config.description = "Train on high-volatility periods only"

        elif exp_type == "feature_research":
            # Wave 32: Feature research — generate candidate features to test
            config = create_experiment_variant(base, exp_type)
            config.description = "Feature research: testing candidate features"
            try:
                from src.phase_09_features_calendar.feature_researcher import FeatureResearchAgent
                agent = FeatureResearchAgent()
                config.metadata["candidates"] = agent.get_candidates_for_config(n=3)
            except Exception as fr_err:
                logging.warning(f"[FEATURE_RESEARCH] Could not generate candidates: {fr_err}")
                config.metadata["candidates"] = []

        elif exp_type == "augmented_training":
            # Wave 35: Training augmentations — test anti-overfitting training mods
            config = create_experiment_variant(base, exp_type)
            aug = config.training_augmentation
            r = random.random()
            if r < 0.40:
                aug.use_temporal_decay = True
                aug.temporal_decay_lambda = random.uniform(0.3, 1.0)
                config.description = f"Augmented: temporal decay (lambda={aug.temporal_decay_lambda:.2f})"
            elif r < 0.70:
                aug.use_noise_injection = True
                aug.noise_sigma = random.uniform(0.05, 0.20)
                config.description = f"Augmented: noise injection (sigma={aug.noise_sigma:.2f})"
            elif r < 0.90:
                aug.use_nested_cv = True
                config.description = "Augmented: nested CV (honest evaluation)"
            else:
                config.description = "Augmented: distillation baseline"
            # Distillation always on by default

        elif exp_type == "data_source":
            # Wave 38: Explore data source combinations — randomly toggle 2-4 flags
            config = create_experiment_variant(base, exp_type)
            source_flags = [
                "use_fear_greed", "use_reddit_sentiment", "use_crypto_sentiment",
                "use_gamma_exposure", "use_finnhub_social", "use_dark_pool",
                "use_options_features", "use_amihud_features", "use_range_vol_features",
                "use_entropy_features", "use_hurst_features", "use_nmi_features",
                "use_absorption_ratio", "use_drift_features",
                "use_changepoint_features", "use_hmm_features",
                "use_vpin_features", "use_intraday_momentum", "use_futures_basis",
                "use_congressional_features", "use_insider_aggregate", "use_etf_flow",
                "use_wavelet_features", "use_sax_features", "use_transfer_entropy",
                "use_mfdfa_features", "use_rqa_features",
                "use_copula_features", "use_network_centrality",
                "use_path_signatures", "use_wavelet_scattering", "use_wasserstein_regime",
                "use_market_structure", "use_time_series_models",
                "use_har_rv", "use_l_moments", "use_multiscale_entropy",
                "use_rv_signature_plot", "use_tda_homology",
            ]
            # Randomly pick 2-4 flags to disable
            n_disable = random.randint(2, 4)
            to_disable = random.sample(source_flags, min(n_disable, len(source_flags) - 1))
            enabled = []
            for flag in source_flags:
                val = flag not in to_disable
                setattr(config.anti_overfit, flag, val)
                if val:
                    enabled.append(flag.replace("use_", ""))
            config.description = f"Data source: {', '.join(enabled)}"

        else:
            config = create_experiment_variant(base, "hyperparameter")

        # Generate new experiment name
        config.experiment_name = f"{exp_type}_{datetime.now().strftime('%H%M%S')}"
        if not config.description.startswith(("Train on", "Augmented:", "Data source:", "Feature research:")):
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
    """Stores and queries experiment history via SQLite (RegistryDB)."""

    def __init__(self, db, **_kwargs):
        self._db = db
        self._lock = threading.Lock()
        self.results: List[ExperimentResult] = []
        self._load()

    def _load(self):
        """Load history from SQLite."""
        try:
            data = self._db.get_experiments()
            self.results = [ExperimentResult.from_dict(e) for e in data]
        except Exception as e:
            logging.warning(f"Could not load experiment history: {e}")
            self.results = []

    def add(self, result: ExperimentResult):
        """Add experiment result to history (thread-safe)."""
        with self._lock:
            self.results.append(result)
            self._db.add_experiment(
                result.to_dict(),
                self._config_hash(result.config),
            )

    @staticmethod
    def _config_hash(cfg) -> Optional[str]:
        """Compute hash of an ExperimentConfig's key hyperparameters.

        Covers dim_reduction, model, feature engineering, and CV settings
        that define unique experiment configurations.
        """
        try:
            key_parts = [
                # Experiment type
                str(cfg.experiment_type),
                # Dimensionality reduction (all significant params)
                str(getattr(cfg.dim_reduction, 'method', '')),
                str(getattr(cfg.dim_reduction, 'target_dimensions', 0)),
                str(getattr(cfg.dim_reduction, 'feature_selection_method', '')),
                str(getattr(cfg.dim_reduction, 'mi_n_features', 0)),
                str(getattr(cfg.dim_reduction, 'kpca_n_components', 0)),
                str(getattr(cfg.dim_reduction, 'kpca_kernel', '')),
                str(getattr(cfg.dim_reduction, 'ica_n_components', 0)),
                str(getattr(cfg.dim_reduction, 'umap_n_components', 0)),
                # Model (type + key hyperparameters)
                str(getattr(cfg.model, 'model_type', '')),
                str(getattr(cfg.model, 'regularization', '')),
                str(getattr(cfg.model, 'l2_C', 0)),
                str(getattr(cfg.model, 'gb_n_estimators', 0)),
                str(getattr(cfg.model, 'gb_max_depth', 0)),
                str(getattr(cfg.model, 'gb_learning_rate', 0)),
                str(getattr(cfg.model, 'gb_min_samples_leaf', 0)),
                str(getattr(cfg.model, 'gb_subsample', 0)),
                # Feature engineering flags
                str(getattr(cfg.feature_engineering, 'use_premarket_features', True)),
                str(getattr(cfg.feature_engineering, 'use_afterhours_features', True)),
                str(getattr(cfg.feature_engineering, 'use_pattern_recognition', True)),
                str(getattr(cfg.feature_engineering, 'swing_threshold', 0)),
                # CV settings
                str(getattr(cfg.cross_validation, 'n_cv_folds', 5)),
                str(getattr(cfg.cross_validation, 'use_soft_targets', True)),
                # Data source flags (Wave 38)
                str(getattr(cfg.anti_overfit, 'use_fear_greed', True)),
                str(getattr(cfg.anti_overfit, 'use_reddit_sentiment', True)),
                str(getattr(cfg.anti_overfit, 'use_crypto_sentiment', True)),
                str(getattr(cfg.anti_overfit, 'use_gamma_exposure', True)),
                str(getattr(cfg.anti_overfit, 'use_finnhub_social', True)),
                str(getattr(cfg.anti_overfit, 'use_dark_pool', True)),
                str(getattr(cfg.anti_overfit, 'use_options_features', True)),
                str(getattr(cfg.anti_overfit, 'use_amihud_features', True)),
                str(getattr(cfg.anti_overfit, 'use_range_vol_features', True)),
                str(getattr(cfg.anti_overfit, 'use_entropy_features', True)),
                str(getattr(cfg.anti_overfit, 'use_hurst_features', True)),
                str(getattr(cfg.anti_overfit, 'use_nmi_features', True)),
                str(getattr(cfg.anti_overfit, 'use_absorption_ratio', True)),
                str(getattr(cfg.anti_overfit, 'use_drift_features', True)),
                str(getattr(cfg.anti_overfit, 'use_changepoint_features', True)),
                str(getattr(cfg.anti_overfit, 'use_hmm_features', True)),
                str(getattr(cfg.anti_overfit, 'use_vpin_features', True)),
                str(getattr(cfg.anti_overfit, 'use_intraday_momentum', True)),
                str(getattr(cfg.anti_overfit, 'use_futures_basis', True)),
                str(getattr(cfg.anti_overfit, 'use_congressional_features', True)),
                str(getattr(cfg.anti_overfit, 'use_insider_aggregate', True)),
                str(getattr(cfg.anti_overfit, 'use_etf_flow', True)),
                str(getattr(cfg.anti_overfit, 'use_wavelet_features', True)),
                str(getattr(cfg.anti_overfit, 'use_sax_features', True)),
                str(getattr(cfg.anti_overfit, 'use_transfer_entropy', True)),
                str(getattr(cfg.anti_overfit, 'use_mfdfa_features', True)),
                str(getattr(cfg.anti_overfit, 'use_rqa_features', True)),
                str(getattr(cfg.anti_overfit, 'use_copula_features', True)),
                str(getattr(cfg.anti_overfit, 'use_network_centrality', True)),
                str(getattr(cfg.anti_overfit, 'use_path_signatures', True)),
                str(getattr(cfg.anti_overfit, 'use_wavelet_scattering', True)),
                str(getattr(cfg.anti_overfit, 'use_wasserstein_regime', True)),
                str(getattr(cfg.anti_overfit, 'use_market_structure', True)),
                str(getattr(cfg.anti_overfit, 'use_time_series_models', False)),
                str(getattr(cfg.anti_overfit, 'use_har_rv', True)),
                str(getattr(cfg.anti_overfit, 'use_l_moments', True)),
                str(getattr(cfg.anti_overfit, 'use_multiscale_entropy', True)),
                str(getattr(cfg.anti_overfit, 'use_rv_signature_plot', False)),
                str(getattr(cfg.anti_overfit, 'use_tda_homology', False)),
            ]
            return hashlib.md5("|".join(key_parts).encode()).hexdigest()[:12]
        except Exception:
            return None

    def get_config_hashes(self) -> set:
        """Return set of config hashes for all previously attempted experiments.

        Used by ExperimentGenerator to avoid re-running identical configs.
        """
        return self._db.get_config_hashes()

    def get_recent(self, n: int = 50) -> List[ExperimentResult]:
        """Get recent experiments."""
        with self._lock:
            return list(self.results[-n:])

    def get_by_status(self, status: ExperimentStatus) -> List[ExperimentResult]:
        """Filter by status."""
        with self._lock:
            return [r for r in self.results if r.status == status]

    def get_statistics(self) -> Dict:
        """Get experiment statistics (thread-safe).

        Only includes completed experiments with AUC > 0 in averaging
        to avoid contamination from partial failures.
        """
        with self._lock:
            if not self.results:
                return {"total": 0}

            completed = [r for r in self.results if r.status == ExperimentStatus.COMPLETED]
            failed = [r for r in self.results if r.status == ExperimentStatus.FAILED]

            # Filter out completed-but-zero-AUC (partial failures) from averaging
            scored = [r for r in completed if r.test_auc > 0]

            # Wave 16: separate realistic best_auc (excluding leaky pre-Wave-14 results)
            realistic = [r for r in scored if r.test_auc < 0.85]
            return {
                "total": len(self.results),
                "completed": len(completed),
                "failed": len(failed),
                "scored": len(scored),
                "success_rate": len(completed) / len(self.results) if self.results else 0,
                "avg_duration": np.mean([r.duration_seconds for r in completed]) if completed else 0,
                "avg_test_auc": np.mean([r.test_auc for r in scored]) if scored else 0,
                "avg_wmes": np.mean([r.wmes_score for r in scored]) if scored else 0,
                "best_test_auc": max([r.test_auc for r in scored]) if scored else 0,
                "best_realistic_auc": max([r.test_auc for r in realistic]) if realistic else 0,
            }

    def get_failure_patterns(self) -> Dict:
        """Analyze which experiment configurations tend to fail.

        Returns dict with failure counts by dim_reduction method and model type,
        plus overall failure rate for the last 20 experiments.
        """
        failed = [r for r in self.results if r.status == ExperimentStatus.FAILED]
        if not failed:
            return {"n_failures": 0, "dim_methods_failing": {}, "model_types_failing": {}}

        dim_methods: Dict[str, int] = {}
        model_types: Dict[str, int] = {}
        for r in failed:
            try:
                dm = r.config.dim_reduction.method if hasattr(r.config, "dim_reduction") else "unknown"
                dim_methods[dm] = dim_methods.get(dm, 0) + 1
            except Exception:
                pass
            try:
                mt = r.config.model.model_type if hasattr(r.config, "model") else "unknown"
                model_types[mt] = model_types.get(mt, 0) + 1
            except Exception:
                pass

        recent = self.results[-20:] if len(self.results) >= 20 else self.results
        recent_failure_rate = sum(
            1 for r in recent if r.status == ExperimentStatus.FAILED
        ) / max(len(recent), 1)

        return {
            "n_failures": len(failed),
            "dim_methods_failing": dim_methods,
            "model_types_failing": model_types,
            "recent_failure_rate": round(recent_failure_rate, 3),
        }

    def get_recent_trend(self, n: int = 20) -> Dict:
        """Detect if recent experiment quality is improving or degrading.

        Compares the first half of the last N completed experiments to the
        second half. A decline of >0.02 AUC flags the trend as 'declining'.
        """
        completed = [r for r in self.results if r.status == ExperimentStatus.COMPLETED
                     and r.test_auc < 0.85]
        recent = completed[-n:] if len(completed) >= n else completed

        if len(recent) < 6:
            return {"trend": "insufficient_data", "n_completed": len(recent)}

        mid = len(recent) // 2
        first_half_auc = float(np.mean([r.test_auc for r in recent[:mid]]))
        second_half_auc = float(np.mean([r.test_auc for r in recent[mid:]]))
        delta = second_half_auc - first_half_auc

        if delta < -0.02:
            trend = "declining"
        elif delta > 0.02:
            trend = "improving"
        else:
            trend = "stable"

        return {
            "trend": trend,
            "first_half_auc": round(first_half_auc, 4),
            "second_half_auc": round(second_half_auc, 4),
            "auc_delta": round(delta, 4),
            "n_completed": len(recent),
        }


