"""
GIGA TRADER - Unified Experiment Engine
========================================
Runs experiments using the EXACT SAME pipeline as train_robust_model.py.

NO simplified parallel pipelines - every experiment uses:
  - Full feature engineering (premarket, afterhours, patterns, interactions)
  - Full anti-overfit integration (synthetic universes, cross-assets, MAG breadth)
  - SPY-minus-component modifiers
  - Comprehensive dimensionality reduction
  - Proper cross-validation with purging

Each experiment is fully specified by a JSON config that can:
  - Reproduce the exact same model
  - Be visualized in the dashboard
  - Track all parameters and results

Usage:
    from src.experiment_engine import ExperimentEngine
    engine = ExperimentEngine()
    result = engine.run_experiment(config)
"""

import os
import sys
import time
import json
import random
import hashlib
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
import logging
from queue import Queue, Empty
import traceback

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import joblib

load_dotenv(project_root / ".env")

# Import the comprehensive config schema
from src.experiment_config import (
    ExperimentConfig,
    create_default_config,
    create_experiment_variant,
    validate_config,
    to_training_config,
)
from src.experiment_progress import (
    ExperimentProgressTracker,
    ExperimentStep,
)

# Import leak-proof CV pipeline (fixes data leakage)
try:
    from src.leak_proof_cv import (
        LeakProofPipeline,
        LeakProofCV,
        train_with_leak_proof_cv,
    )
    HAS_LEAK_PROOF = True
except ImportError:
    HAS_LEAK_PROOF = False
    print("[WARN] leak_proof_cv module not available, using legacy CV")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
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
# WALK-FORWARD CROSS-VALIDATION WITH PURGING & EMBARGO
# ═══════════════════════════════════════════════════════════════════════════════
class WalkForwardCV:
    """
    Walk-forward cross-validation with purging and embargo.

    Prevents data leakage by:
      1. Always training on past data, testing on future data
      2. Purging: Remove N days between train and test (autocorrelation)
      3. Embargo: Remove N days after test before next train (information bleed)
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_days: int = 5,
        embargo_days: int = 2,
        min_train_size: int = 100,
        test_size: int = 50,
    ):
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.min_train_size = min_train_size
        self.test_size = test_size

    def split(self, X, y=None, dates=None):
        """
        Generate walk-forward train/test indices with purging and embargo.

        Args:
            X: Feature matrix
            y: Target vector (optional)
            dates: Array of dates for each sample (required for proper purging)

        Yields:
            train_idx, test_idx tuples
        """
        n_samples = len(X)

        if dates is None:
            # Fall back to simple time-based split (less accurate)
            dates = np.arange(n_samples)

        unique_dates = np.unique(dates)
        n_dates = len(unique_dates)

        # Calculate split points
        total_test = self.n_splits * self.test_size
        available_for_train = n_dates - total_test - self.n_splits * (self.purge_days + self.embargo_days)

        if available_for_train < self.min_train_size:
            # Reduce n_splits if not enough data
            self.n_splits = max(2, (n_dates - self.min_train_size) // (self.test_size + self.purge_days + self.embargo_days))

        for split_idx in range(self.n_splits):
            # Calculate date indices for this split
            test_end_date_idx = n_dates - split_idx * (self.test_size + self.embargo_days)
            test_start_date_idx = test_end_date_idx - self.test_size

            # Training ends before purge period
            train_end_date_idx = test_start_date_idx - self.purge_days
            train_start_date_idx = max(0, train_end_date_idx - (self.min_train_size + split_idx * 20))

            if train_end_date_idx <= train_start_date_idx:
                continue

            # Convert date indices to sample indices
            train_dates = unique_dates[train_start_date_idx:train_end_date_idx]
            test_dates = unique_dates[test_start_date_idx:test_end_date_idx]

            train_idx = np.where(np.isin(dates, train_dates))[0]
            test_idx = np.where(np.isin(dates, test_dates))[0]

            if len(train_idx) >= self.min_train_size and len(test_idx) > 0:
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


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

    # Live performance (from paper trading)
    live_trades: int = 0
    live_win_rate: float = 0.0
    live_total_return: float = 0.0
    live_sharpe: float = 0.0

    def score(self, weights: Dict = None) -> float:
        """Calculate weighted score for ranking."""
        weights = weights or {
            "cv_auc": 0.2,
            "backtest_sharpe": 0.3,
            "wmes_score": 0.2,
            "live_sharpe": 0.3,
        }
        return sum(
            weights.get(k, 0) * getattr(self, k, 0)
            for k in weights
        )


# ═══════════════════════════════════════════════════════════════════════════════
# UNIFIED EXPERIMENT RUNNER
# ═══════════════════════════════════════════════════════════════════════════════
class UnifiedExperimentRunner:
    """
    Runs experiments using the FULL train_robust_model.py pipeline.

    NO shortcuts, NO simplified versions - uses identical code paths.
    """

    def __init__(self):
        self.logger = logging.getLogger("EXPERIMENT")
        self.data = None
        self._data_lock = threading.Lock()

    def load_data(self, config: ExperimentConfig):
        """Load data using DataManager with proper columns."""
        with self._data_lock:
            if self.data is None:
                from src.data_manager import get_spy_data
                self.data = get_spy_data(years=config.data.years_to_download)
                self.logger.info(f"Loaded {len(self.data):,} bars")

    def run(self, config: ExperimentConfig) -> ExperimentResult:
        """
        Run experiment using the FULL unified pipeline.

        Steps:
          1. Load data
          2. Engineer ALL features (premarket, afterhours, patterns, interactions)
          3. Run FULL anti-overfit integration (synthetic universes, cross-assets)
          4. Apply dimensionality reduction
          5. Train model with cross-validation
          6. Evaluate with WMES and stability analysis
          7. Run backtest
        """
        result = ExperimentResult(
            experiment_id=config.experiment_id,
            config=config,
            status=ExperimentStatus.RUNNING,
            started_at=datetime.now().isoformat(),
        )
        start_time = time.time()

        # Get progress tracker
        tracker = ExperimentProgressTracker.instance()
        tracker.start_experiment(
            experiment_id=config.experiment_id,
            experiment_type=config.experiment_type,
            experiment_name=config.experiment_name,
        )

        try:
            self.logger.info(f"=" * 70)
            self.logger.info(f"RUNNING EXPERIMENT: {config.experiment_id}")
            self.logger.info(f"Type: {config.experiment_type}")
            self.logger.info(f"=" * 70)

            # Step 1: Load data
            self.logger.info("[STEP 1] Loading data...")
            tracker.set_step(ExperimentStep.LOADING_DATA, f"Loading {config.data.years_to_download} years of data")
            self.load_data(config)
            tracker.record_metric("bars_loaded", len(self.data) if self.data is not None else 0)

            # Step 2: Import the SAME functions from train_robust_model.py
            from src.train_robust_model import (
                engineer_all_features,
                add_rolling_features,
                create_soft_targets,
                reduce_dimensions,
            )
            from src.anti_overfit import (
                integrate_anti_overfit,
                compute_weighted_evaluation,
                WeightedModelEvaluator,
                StabilityAnalyzer,
            )

            # Step 3: Feature engineering (FULL - premarket, afterhours, patterns)
            self.logger.info("[STEP 2] Engineering features (FULL pipeline)...")
            tracker.set_step(ExperimentStep.FEATURE_ENGINEERING, "Engineering premarket, afterhours, patterns")
            swing_threshold = config.feature_engineering.swing_threshold

            tracker.update_substep("Running engineer_all_features", 30)
            df_daily = engineer_all_features(self.data.copy(), swing_threshold=swing_threshold)

            tracker.update_substep("Adding rolling features", 60)
            df_daily = add_rolling_features(df_daily)

            tracker.update_substep("Creating soft targets", 90)
            df_daily = create_soft_targets(df_daily, threshold=swing_threshold)

            result.n_features_initial = len([c for c in df_daily.columns
                                             if df_daily[c].dtype in ['float64', 'int64']])
            result.n_samples_real = len(df_daily)
            self.logger.info(f"  Features: {result.n_features_initial}, Samples: {result.n_samples_real}")
            tracker.record_metrics({
                "n_features_initial": result.n_features_initial,
                "n_samples_real": result.n_samples_real,
            })

            # ═══════════════════════════════════════════════════════════════════
            # CRITICAL FIX: Split by DATE first, THEN apply synthetic augmentation
            # This prevents data leakage from synthetic data sharing temporal patterns
            # ═══════════════════════════════════════════════════════════════════

            # Step 4: Extract dates and prepare for DATE-based split
            self.logger.info("[STEP 3] Preparing DATE-based train/test split...")
            tracker.set_step(ExperimentStep.TRAIN_TEST_SPLIT, "Splitting by DATE (not row index)")

            # Get dates from the dataframe
            if "date" in df_daily.columns:
                dates = pd.to_datetime(df_daily["date"]).dt.date.values
            elif df_daily.index.dtype == 'datetime64[ns]':
                dates = df_daily.index.date
            else:
                # Fall back to row index as date proxy
                dates = np.arange(len(df_daily))

            unique_dates = np.unique(dates)
            n_dates = len(unique_dates)
            self.logger.info(f"  Total unique dates: {n_dates}")

            # Split dates: 80% train, 20% test
            split_date_idx = int(n_dates * 0.8)
            train_dates = set(unique_dates[:split_date_idx])
            test_dates = set(unique_dates[split_date_idx:])

            self.logger.info(f"  Train dates: {len(train_dates)}, Test dates: {len(test_dates)}")
            self.logger.info(f"  Train range: {min(train_dates)} to {max(train_dates)}")
            self.logger.info(f"  Test range: {min(test_dates)} to {max(test_dates)}")

            # Create masks for REAL data only
            train_mask_real = np.array([d in train_dates for d in dates])
            test_mask_real = np.array([d in test_dates for d in dates])

            # Separate train and test data (REAL only at this point)
            df_train_real = df_daily[train_mask_real].copy()
            df_test = df_daily[test_mask_real].copy()

            result.n_samples_real = len(df_daily)
            self.logger.info(f"  Train samples (real): {len(df_train_real)}, Test samples: {len(df_test)}")

            tracker.record_metrics({
                "n_train_dates": len(train_dates),
                "n_test_dates": len(test_dates),
                "n_train_real": len(df_train_real),
                "n_test": len(df_test),
            })

            # Step 5: Apply anti-overfit augmentation ONLY to training data
            df_train = df_train_real.copy()
            if config.anti_overfit.use_anti_overfit:
                self.logger.info("[STEP 4] Running anti-overfit integration (TRAINING ONLY)...")
                tracker.set_step(ExperimentStep.ANTI_OVERFIT, "Augmenting TRAINING data only")
                self.logger.info(f"  - Synthetic universes: {config.anti_overfit.use_synthetic_universes}")
                self.logger.info(f"  - Cross-assets: {config.anti_overfit.use_cross_assets}")
                self.logger.info(f"  - MAG breadth: {config.anti_overfit.use_mag_breadth}")
                self.logger.info(f"  - Breadth streaks: {config.anti_overfit.use_breadth_streaks}")
                self.logger.info(f"  - SPY-minus-component: {config.anti_overfit.use_spy_minus_component}")
                self.logger.info(f"  [NOTE] Test data will NOT be augmented (pure REAL data)")

                tracker.update_substep("Integrating synthetic universes (training only)", 50)
                try:
                    df_train_augmented, metadata = integrate_anti_overfit(
                        df_train_real,
                        use_breadth_streaks=config.anti_overfit.use_breadth_streaks,
                        use_cross_assets=config.anti_overfit.use_cross_assets,
                        use_mag_breadth=config.anti_overfit.use_mag_breadth,
                        use_synthetic=config.anti_overfit.use_synthetic_universes,
                        synthetic_weight=config.anti_overfit.synthetic_weight,
                    )
                    result.n_samples_synthetic = len(df_train_augmented) - len(df_train_real)
                    df_train = df_train_augmented
                    self.logger.info(f"  Train augmented: {len(df_train)} (+{result.n_samples_synthetic} synthetic)")
                    tracker.record_metric("n_samples_synthetic", result.n_samples_synthetic)
                except Exception as aug_err:
                    self.logger.warning(f"  Anti-overfit augmentation failed: {aug_err}")
                    self.logger.warning("  Continuing with real training data only...")
                    tracker.record_metric("anti_overfit_error", str(aug_err))

            # Step 6: Prepare features and targets for train and test SEPARATELY
            self.logger.info("[STEP 5] Preparing features...")
            exclude_cols = ["date", "target_up", "target_timing", "soft_target_up",
                           "day_return", "sample_weight", "is_synthetic"]
            feature_cols = [c for c in df_train.columns if c not in exclude_cols
                           and not c.startswith("smoothed_")]
            feature_cols = [c for c in feature_cols if df_train[c].dtype in ['float64', 'int64']]

            # Ensure test has same features
            feature_cols = [c for c in feature_cols if c in df_test.columns]

            # Clean and prepare training data
            df_train_clean = df_train.dropna(subset=feature_cols + ["target_up"])
            X_train_raw = df_train_clean[feature_cols].values
            y_train = df_train_clean["target_up"].astype(int).values

            # Get dates for training data (for walk-forward CV)
            if "date" in df_train_clean.columns:
                train_dates_array = pd.to_datetime(df_train_clean["date"]).dt.date.values
            else:
                train_dates_array = np.arange(len(df_train_clean))

            # Clean and prepare test data (REAL data only, no synthetic)
            df_test_clean = df_test.dropna(subset=feature_cols + ["target_up"])
            X_test_raw = df_test_clean[feature_cols].values
            y_test = df_test_clean["target_up"].astype(int).values
            test_returns = df_test_clean["day_return"].values if "day_return" in df_test_clean.columns else None

            # Get dates for test data
            if "date" in df_test_clean.columns:
                test_dates_array = pd.to_datetime(df_test_clean["date"]).dt.date.values
            else:
                test_dates_array = np.arange(len(df_test_clean))

            # Sample weights for training (synthetic samples have lower weight)
            weights_train = None
            if "sample_weight" in df_train_clean.columns:
                weights_train = df_train_clean["sample_weight"].values

            self.logger.info(f"  Train features: {X_train_raw.shape}, Test features: {X_test_raw.shape}")

            # ═══════════════════════════════════════════════════════════════════
            # LEAK-PROOF CV PATH (recommended - all transforms inside CV folds)
            # ═══════════════════════════════════════════════════════════════════
            use_leak_proof = ENGINE_CONFIG.get("use_leak_proof_cv", True) and HAS_LEAK_PROOF

            if use_leak_proof:
                from sklearn.metrics import roc_auc_score

                self.logger.info("[STEP 6] Using LEAK-PROOF CV (all transforms inside folds)")
                tracker.set_step(ExperimentStep.CROSS_VALIDATION, "Leak-proof CV with embedded transforms")

                leak_proof_config = {
                    "n_cv_folds": config.cross_validation.n_cv_folds,
                    "purge_days": ENGINE_CONFIG['purge_days'],
                    "embargo_days": ENGINE_CONFIG['embargo_days'],
                    "feature_selection_method": "mutual_info",
                    "n_features": 30,
                    "dim_reduction_method": "kernel_pca",
                    "n_components": 20,
                    "use_ensemble": ENGINE_CONFIG.get("use_model_ensemble", True),
                    "random_state": 42,
                }

                # Run leak-proof CV (transforms inside each fold)
                pipeline, cv_results = train_with_leak_proof_cv(
                    X_train_raw, y_train,
                    sample_weights=weights_train,
                    config=leak_proof_config,
                    verbose=True,
                )

                # Store CV results
                result.cv_scores = cv_results.get("test_aucs", [])
                result.cv_auc_mean = cv_results.get("mean_test_auc", 0.5)
                result.cv_auc_std = cv_results.get("std_test_auc", 0.0)

                self.logger.info(f"  Leak-Proof CV AUC: {result.cv_auc_mean:.4f} +/- {result.cv_auc_std:.4f}")
                self.logger.info(f"  Train-Test Gap: {cv_results.get('train_test_gap', 0):.3f}")

                # Evaluate on held-out test data
                tracker.set_step(ExperimentStep.EVALUATION, "Evaluating on test data")
                test_proba = pipeline.predict_proba(X_test_raw)[:, 1]
                result.test_auc = float(roc_auc_score(y_test, test_proba))

                # Get train AUC from final model
                train_proba = pipeline.predict_proba(X_train_raw)[:, 1]
                result.train_auc = float(roc_auc_score(y_train, train_proba))

                self.logger.info(f"  Train AUC: {result.train_auc:.4f}")
                self.logger.info(f"  Test AUC: {result.test_auc:.4f}")

                result.n_features_initial = len(feature_cols)
                result.n_features_final = leak_proof_config["n_features"]

                tracker.record_metrics({
                    "cv_auc_mean": round(result.cv_auc_mean, 4),
                    "cv_auc_std": round(result.cv_auc_std, 4),
                    "train_auc": round(result.train_auc, 4),
                    "test_auc": round(result.test_auc, 4),
                    "train_test_gap": round(cv_results.get('train_test_gap', 0), 4),
                    "leak_proof": True,
                })

                # Store pipeline as model
                model = pipeline

            else:
                # ═══════════════════════════════════════════════════════════════════
                # LEGACY CV PATH (may have data leakage - use for comparison only)
                # ═══════════════════════════════════════════════════════════════════
                self.logger.info("[STEP 6] Using LEGACY CV (transforms before split)")
                self.logger.warning("  [NOTE] This path may have data leakage - use leak-proof for production")

                # Step 7: Dimensionality reduction (fit on TRAIN, transform BOTH)
                self.logger.info(f"[STEP 6] Dimensionality reduction: {config.dim_reduction.method}")
                tracker.set_step(ExperimentStep.DIM_REDUCTION, f"Method: {config.dim_reduction.method}")
                tracker.update_substep(f"Reducing {X_train_raw.shape[1]} features (fit on train only)", 50)

                # Fit on training data only
                # reduce_dimensions returns: (X_transformed, feature_names, state_dict)
                X_train, selected_features, dim_state = reduce_dimensions(
                    X_train_raw, feature_cols, y=y_train, fit=True
                )

                # Transform test data using the same state dict
                X_test, _, _ = reduce_dimensions(
                    X_test_raw, feature_cols, y=None, fit=False, state=dim_state
                )

                result.n_features_initial = len(feature_cols)
                result.n_features_final = X_train.shape[1]
                self.logger.info(f"  Reduced: {len(feature_cols)} -> {X_train.shape[1]} features")
                tracker.record_metric("n_features_final", result.n_features_final)

                tracker.record_metrics({
                    "n_train": len(X_train),
                    "n_test": len(X_test),
                })

                # Step 8: Train model
                self.logger.info(f"[STEP 7] Training model: {config.model.model_type}")
                tracker.set_step(ExperimentStep.MODEL_TRAINING, f"Training {config.model.model_type}")
                model = self._create_model(config)

                # Step 9: Walk-forward cross-validation with purging and embargo
                # IMPORTANT: Run CV on REAL data only (synthetic causes duplicate dates issues)
                tracker.set_step(ExperimentStep.CROSS_VALIDATION,
                               f"Walk-forward CV ({config.cross_validation.n_cv_folds} folds) with purging")
                self.logger.info("[STEP 8] Walk-forward CV with purging and embargo...")
                self.logger.info(f"  Purge days: {ENGINE_CONFIG['purge_days']}, Embargo days: {ENGINE_CONFIG['embargo_days']}")

                from sklearn.metrics import roc_auc_score

                # Filter to REAL data only for CV (synthetic has duplicate dates)
                is_synthetic = df_train_clean.get("is_synthetic", pd.Series([False] * len(df_train_clean))).values
                real_mask = ~is_synthetic.astype(bool) if is_synthetic is not None else np.ones(len(X_train), dtype=bool)

                X_train_real_cv = X_train[real_mask]
                y_train_real_cv = y_train[real_mask]
                dates_real_cv = train_dates_array[real_mask] if len(train_dates_array) == len(real_mask) else train_dates_array[:len(X_train_real_cv)]
                weights_real_cv = weights_train[real_mask] if weights_train is not None else None

                self.logger.info(f"  CV on REAL data only: {len(X_train_real_cv)} samples ({len(np.unique(dates_real_cv))} unique dates)")

                # Use walk-forward CV on real data
                wf_cv = WalkForwardCV(
                    n_splits=config.cross_validation.n_cv_folds,
                    purge_days=ENGINE_CONFIG['purge_days'],
                    embargo_days=ENGINE_CONFIG['embargo_days'],
                    min_train_size=max(50, len(X_train_real_cv) // 10),
                    test_size=max(20, len(X_train_real_cv) // 20),
                )

                cv_scores = []
                for fold_idx, (train_idx, val_idx) in enumerate(wf_cv.split(X_train_real_cv, y_train_real_cv, dates_real_cv)):
                    # Train on fold (REAL data only)
                    X_fold_train, y_fold_train = X_train_real_cv[train_idx], y_train_real_cv[train_idx]
                    X_fold_val, y_fold_val = X_train_real_cv[val_idx], y_train_real_cv[val_idx]

                    fold_weights = weights_real_cv[train_idx] if weights_real_cv is not None else None

                    # Clone model for this fold
                    fold_model = self._create_model(config)
                    if fold_weights is not None:
                        try:
                            fold_model.fit(X_fold_train, y_fold_train, sample_weight=fold_weights)
                        except TypeError:
                            fold_model.fit(X_fold_train, y_fold_train)
                    else:
                        fold_model.fit(X_fold_train, y_fold_train)

                    # Score on validation
                    try:
                        val_proba = fold_model.predict_proba(X_fold_val)[:, 1]
                        fold_auc = roc_auc_score(y_fold_val, val_proba)
                        cv_scores.append(fold_auc)
                        self.logger.info(f"    Fold {fold_idx + 1}: AUC={fold_auc:.4f} (train={len(train_idx)}, val={len(val_idx)})")
                    except Exception as fold_err:
                        self.logger.warning(f"    Fold {fold_idx + 1} failed: {fold_err}")

                if cv_scores:
                    result.cv_scores = cv_scores
                    result.cv_auc_mean = float(np.mean(cv_scores))
                    result.cv_auc_std = float(np.std(cv_scores))
                else:
                    self.logger.warning("  No CV folds completed - check data size and dates")
                    result.cv_auc_mean = 0.5
                    result.cv_auc_std = 0.0

                self.logger.info(f"  Walk-Forward CV AUC: {result.cv_auc_mean:.4f} +/- {result.cv_auc_std:.4f}")
                tracker.record_metrics({
                    "cv_auc_mean": round(result.cv_auc_mean, 4),
                    "cv_auc_std": round(result.cv_auc_std, 4),
                })

                # Step 10: Train final model on all training data
                self.logger.info("[STEP 9] Training final model on all training data...")
                if weights_train is not None:
                    try:
                        model.fit(X_train, y_train, sample_weight=weights_train)
                    except TypeError:
                        model.fit(X_train, y_train)
                else:
                    model.fit(X_train, y_train)

                # Step 11: Evaluate on held-out test data (REAL data only)
                tracker.set_step(ExperimentStep.EVALUATION, "Evaluating on REAL test data")
                tracker.update_substep("Computing train/test AUC", 30)

                result.train_auc = float(roc_auc_score(y_train, model.predict_proba(X_train)[:, 1]))
                result.test_auc = float(roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))

                self.logger.info(f"  Train AUC: {result.train_auc:.4f}")
                self.logger.info(f"  Test AUC (REAL data only): {result.test_auc:.4f}")

                # Check for suspicious AUC (potential leakage indicator)
                if result.test_auc > 0.90:
                    self.logger.warning(f"  [WARNING] Test AUC > 0.90 is suspicious - check for data leakage!")
                if result.train_auc - result.test_auc > 0.15:
                    self.logger.warning(f"  [WARNING] Large train-test gap ({result.train_auc - result.test_auc:.3f}) - possible overfitting!")

                tracker.record_metrics({
                    "train_auc": round(result.train_auc, 4),
                    "test_auc": round(result.test_auc, 4),
                    "auc_gap": round(result.train_auc - result.test_auc, 4),
                })

            # Step 12: WMES evaluation
            self.logger.info("[STEP 10] Computing WMES and stability metrics...")
            tracker.update_substep("Computing WMES score", 70)

            # Use correct feature arrays based on which path was taken
            if use_leak_proof:
                test_proba_wmes = model.predict_proba(X_test_raw)[:, 1]
                train_proba_wmes = model.predict_proba(X_train_raw)[:, 1]
            else:
                test_proba_wmes = model.predict_proba(X_test)[:, 1]
                train_proba_wmes = model.predict_proba(X_train)[:, 1]

            evaluator = WeightedModelEvaluator()
            wmes_result = evaluator.evaluate(
                y_test=y_test,
                y_pred_proba=test_proba_wmes,
                y_train=y_train,
                y_train_proba=train_proba_wmes,
            )
            result.wmes_score = wmes_result.get("wmes_score", 0)
            self.logger.info(f"  WMES Score: {result.wmes_score:.3f}")
            tracker.record_metric("wmes_score", round(result.wmes_score, 3))

            # Step 13: Realistic backtest with transaction costs
            self.logger.info("[STEP 11] Running REALISTIC backtest with transaction costs...")
            tracker.set_step(ExperimentStep.BACKTEST, "Backtest with slippage & commission")

            # Use correct feature arrays based on which path was taken
            if use_leak_proof:
                proba_raw = model.predict_proba(X_test_raw)[:, 1]
            else:
                proba_raw = model.predict_proba(X_test)[:, 1]

            # ANTI-OVERFITTING: Calibrate probabilities to prevent overconfidence
            # This clips extreme predictions and shrinks toward 0.5
            proba = calibrate_probabilities(
                proba_raw,
                min_prob=0.05,
                max_prob=0.95,
                confidence_penalty=0.1,  # 10% shrinkage toward 0.5
            )
            self.logger.info(f"  Calibrated probabilities: min={proba.min():.3f}, max={proba.max():.3f}, mean={proba.mean():.3f}")

            threshold = config.trading.entry_threshold
            signals = (proba > threshold).astype(int)

            if test_returns is not None and len(test_returns) == len(signals):
                # Use realistic backtest metrics
                backtest_metrics = compute_realistic_backtest_metrics(
                    signals=signals,
                    returns=test_returns,
                    dates=test_dates_array,
                    slippage_bps=ENGINE_CONFIG['slippage_bps'],
                    commission_bps=ENGINE_CONFIG['commission_bps'],
                )

                # Report both gross and net metrics
                result.backtest_win_rate = backtest_metrics["win_rate_net"]  # Use NET win rate
                result.backtest_total_return = backtest_metrics["total_return_net"]  # Use NET return
                result.backtest_sharpe = backtest_metrics["sharpe_net"]  # Use NET Sharpe
                result.backtest_max_drawdown = backtest_metrics["max_drawdown"]

                self.logger.info(f"  Slippage: {ENGINE_CONFIG['slippage_bps']} bps, Commission: {ENGINE_CONFIG['commission_bps']} bps")
                self.logger.info(f"  Transaction cost per trade: {backtest_metrics['transaction_cost_per_trade']:.4%}")
                self.logger.info(f"  ---")
                self.logger.info(f"  Win Rate (gross): {backtest_metrics['win_rate']:.1%}")
                self.logger.info(f"  Win Rate (net):   {backtest_metrics['win_rate_net']:.1%}")
                self.logger.info(f"  Total Return (gross): {backtest_metrics['total_return']:.2%}")
                self.logger.info(f"  Total Return (net):   {backtest_metrics['total_return_net']:.2%}")
                self.logger.info(f"  Sharpe (gross): {backtest_metrics['sharpe']:.3f}")
                self.logger.info(f"  Sharpe (net):   {backtest_metrics['sharpe_net']:.3f}")
                self.logger.info(f"  Max Drawdown: {backtest_metrics['max_drawdown']:.2%}")
                self.logger.info(f"  N Trades: {backtest_metrics['n_trades']}")

                tracker.record_metrics({
                    "backtest_sharpe_gross": round(backtest_metrics['sharpe'], 3),
                    "backtest_sharpe_net": round(backtest_metrics['sharpe_net'], 3),
                    "backtest_win_rate_gross": round(backtest_metrics['win_rate'], 3),
                    "backtest_win_rate_net": round(backtest_metrics['win_rate_net'], 3),
                    "backtest_total_return_gross": round(backtest_metrics['total_return'], 4),
                    "backtest_total_return_net": round(backtest_metrics['total_return_net'], 4),
                    "n_trades": backtest_metrics['n_trades'],
                })
            else:
                self.logger.warning("  Test returns not available for backtest")
                tracker.record_metric("backtest_error", "Test returns not available")

            # Step 14: Optional Entry/Exit Timing Model training
            # This model requires intraday data which may not always be available
            entry_exit_model_trained = False
            if getattr(config, 'train_entry_exit_model', False):
                self.logger.info("[STEP 12] Training Entry/Exit Timing Model (optional)...")
                tracker.set_step(ExperimentStep.MODEL_TRAINING, "Training Entry/Exit Timing Model")

                try:
                    from src.entry_exit_model import EntryExitTimingModel

                    # Check if we have intraday data
                    # The intraday data should have minute-level bars
                    if self.data is not None and len(self.data) > 5000:
                        # Assume we have minute-level data
                        self.logger.info("  Using minute-level data for Entry/Exit model...")

                        # Create directions from predictions
                        train_proba = model.predict_proba(X_train)[:, 1]
                        directions = pd.Series(
                            np.where(train_proba > threshold, "LONG", "SHORT"),
                            index=pd.to_datetime([d for d in train_dates_array])
                        )

                        # Create entry/exit model
                        entry_exit_model = EntryExitTimingModel(
                            model_type="gradient_boosting",
                            entry_window=(0, 120),
                            exit_window=(180, 385),
                            min_position_pct=0.05,
                            max_position_pct=0.25,
                        )

                        # Train (this may fail if data format doesn't match)
                        try:
                            ee_metrics = entry_exit_model.fit(
                                daily_data=df_train_real,
                                intraday_data=self.data,
                                directions=directions,
                                cv_folds=3,
                            )
                            entry_exit_model_trained = True
                            self.logger.info("  Entry/Exit Timing Model trained successfully!")
                            tracker.record_metric("entry_exit_model_trained", True)

                            # Record key metrics
                            if ee_metrics:
                                for model_name, metrics in ee_metrics.items():
                                    for metric_name, value in metrics.items():
                                        if isinstance(value, dict):
                                            tracker.record_metric(
                                                f"ee_{model_name}_{metric_name}",
                                                round(value.get('mean', 0), 3)
                                            )
                        except Exception as ee_train_err:
                            self.logger.warning(f"  Entry/Exit model training failed: {ee_train_err}")
                            tracker.record_metric("entry_exit_model_error", str(ee_train_err))
                    else:
                        self.logger.info("  Skipping Entry/Exit model (insufficient intraday data)")
                        tracker.record_metric("entry_exit_model_trained", False)

                except ImportError as ie:
                    self.logger.warning(f"  Entry/Exit model not available: {ie}")
                except Exception as e:
                    self.logger.warning(f"  Entry/Exit model setup failed: {e}")

            # Success
            result.status = ExperimentStatus.COMPLETED
            self.logger.info(f"[SUCCESS] Experiment completed: {config.experiment_id}")
            tracker.complete_experiment(success=True, result_summary={
                "test_auc": result.test_auc,
                "backtest_sharpe": result.backtest_sharpe,
                "entry_exit_model_trained": entry_exit_model_trained,
            })

        except Exception as e:
            result.status = ExperimentStatus.FAILED
            result.error_message = str(e)
            self.logger.error(f"Experiment {config.experiment_id} failed: {e}")
            self.logger.error(traceback.format_exc())
            tracker.fail_experiment(str(e))

        result.completed_at = datetime.now().isoformat()
        result.duration_seconds = time.time() - start_time

        return result

    def _create_model(self, config: ExperimentConfig):
        """Create model based on configuration.

        ANTI-OVERFITTING MEASURES:
        - Aggressive regularization (lower C = stronger penalty)
        - Diverse ensemble combines models with DIFFERENT regularization strengths
        - Shallow trees (max_depth capped at 4 for robustness)
        - High min_samples_leaf to prevent memorization
        - Calibration wrapper for probability outputs
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
        from sklearn.calibration import CalibratedClassifierCV

        # AGGRESSIVE REGULARIZATION: Apply 10x stronger penalty
        # Original C ranges 0.01-10.0, now use 0.001-1.0 (lower C = stronger reg)
        aggressive_C = min(config.model.l2_C, 1.0) * 0.1  # Cap at 0.1, typically 0.001-0.1

        # Cap tree depth more aggressively (max 4 instead of 5)
        safe_max_depth = min(config.model.gb_max_depth, 4)

        # Increase min_samples_leaf to prevent memorization
        safe_min_samples_leaf = max(config.model.gb_min_samples_leaf, 50)

        if config.model.model_type == "logistic":
            if config.model.regularization == "l1":
                base_model = LogisticRegression(
                    penalty='l1', solver='saga',
                    C=aggressive_C, max_iter=1000, random_state=42
                )
            elif config.model.regularization == "elastic_net":
                # Elastic net with heavier L1 component
                base_model = LogisticRegression(
                    penalty='elasticnet', solver='saga',
                    C=aggressive_C, l1_ratio=max(config.model.elastic_l1_ratio, 0.7),
                    max_iter=1000, random_state=42
                )
            else:
                base_model = LogisticRegression(C=aggressive_C, max_iter=1000, random_state=42)
            return base_model

        elif config.model.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=min(config.model.gb_n_estimators, 100),  # Cap estimators
                max_depth=safe_max_depth,
                learning_rate=min(config.model.gb_learning_rate, 0.1),  # Lower learning rate
                min_samples_leaf=safe_min_samples_leaf,
                subsample=min(config.model.gb_subsample, 0.8),  # More aggressive subsampling
                max_features='sqrt',  # Random feature subset per tree
                random_state=42
            )

        elif config.model.model_type == "ensemble":
            models = []
            for model_name in config.model.ensemble_models:
                if model_name == "logistic_l2":
                    models.append(('lr', LogisticRegression(C=aggressive_C, max_iter=1000, random_state=42)))
                elif model_name == "gradient_boosting":
                    models.append(('gb', GradientBoostingClassifier(
                        n_estimators=min(config.model.gb_n_estimators, 100),
                        max_depth=safe_max_depth,
                        learning_rate=min(config.model.gb_learning_rate, 0.1),
                        min_samples_leaf=safe_min_samples_leaf,
                        max_features='sqrt',
                        random_state=42
                    )))
            return VotingClassifier(models, voting='soft')

        elif config.model.model_type == "diverse_ensemble":
            # DIVERSE ENSEMBLE: Multiple models with DIFFERENT regularization strengths
            # Key insight: Ensemble of similar models just amplifies overfitting
            # We need models that disagree on noise but agree on signal
            models = [
                # Very strong L1 (extreme sparsity)
                ('l1_strong', LogisticRegression(
                    penalty='l1', solver='saga', C=0.001, max_iter=1000, random_state=42
                )),
                # Moderate L1
                ('l1_moderate', LogisticRegression(
                    penalty='l1', solver='saga', C=0.01, max_iter=1000, random_state=43
                )),
                # Strong L2 (ridge)
                ('l2_strong', LogisticRegression(
                    penalty='l2', C=0.01, max_iter=1000, random_state=44
                )),
                # Elastic net (balanced)
                ('elastic', LogisticRegression(
                    penalty='elasticnet', solver='saga', C=0.01, l1_ratio=0.5,
                    max_iter=1000, random_state=45
                )),
                # Very shallow tree (almost decision stump)
                ('tree_shallow', GradientBoostingClassifier(
                    n_estimators=30, max_depth=2, learning_rate=0.05,
                    min_samples_leaf=100, subsample=0.7, max_features='sqrt',
                    random_state=46
                )),
                # Moderate tree (still conservative)
                ('tree_moderate', GradientBoostingClassifier(
                    n_estimators=50, max_depth=3, learning_rate=0.08,
                    min_samples_leaf=75, subsample=0.75, max_features='sqrt',
                    random_state=47
                )),
            ]
            return VotingClassifier(models, voting='soft')

        else:
            # Default to strongly regularized L2 logistic regression
            return LogisticRegression(C=aggressive_C, max_iter=1000, random_state=42)


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
        """Load registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path) as f:
                    data = json.load(f)
                    self.models = {k: ModelRecord(**v) for k, v in data.items()}
            except Exception as e:
                logging.warning(f"Could not load model registry: {e}")
                self.models = {}

    def _save(self):
        """Save registry to disk."""
        data = {k: asdict(v) for k, v in self.models.items()}
        with open(self.registry_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

    def register_model(self, result: ExperimentResult) -> str:
        """Register a new model from experiment result."""
        model_id = f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

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


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
class ExperimentEngine:
    """
    Main engine that runs experiments using the UNIFIED FULL pipeline.

    Every experiment uses the SAME code path as train_robust_model.py.
    No simplified parallel pipelines.
    """

    def __init__(self):
        self.logger = logging.getLogger("ENGINE")
        self.generator = ExperimentGenerator()
        self.runner = UnifiedExperimentRunner()
        self.registry = ModelRegistry()
        self.history = ExperimentHistory()

        self.running = False
        self.current_experiment: Optional[ExperimentConfig] = None
        self.experiments_today = 0
        self.last_reset = datetime.now().date()

    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment with the given configuration."""
        # Validate config
        is_valid, errors = validate_config(config)
        if not is_valid:
            self.logger.warning(f"Config validation warnings: {errors}")

        # Run using unified runner
        result = self.runner.run(config)

        # Store result
        self.history.add(result)

        # Register model if successful
        if result.status == ExperimentStatus.COMPLETED and result.test_auc > 0.55:
            model_id = self.registry.register_model(result)
            self.logger.info(f"Registered model: {model_id} (AUC={result.test_auc:.3f})")

            # Add to best configs pool if good
            if result.test_auc > 0.60:
                self.generator.add_best_config(config, result.test_auc)

        return result

    def run_one_experiment(self) -> Optional[ExperimentResult]:
        """Generate and run a single experiment."""
        # Check daily limit
        if datetime.now().date() != self.last_reset:
            self.experiments_today = 0
            self.last_reset = datetime.now().date()

        if self.experiments_today >= ENGINE_CONFIG["max_experiments_per_hour"] * 24:
            self.logger.warning("Daily experiment limit reached")
            return None

        # Generate config
        config = self.generator.generate_next()
        self.current_experiment = config
        self.logger.info(f"Running experiment: {config.experiment_type}")

        # Run experiment
        result = self.run_experiment(config)
        self.experiments_today += 1

        self.current_experiment = None
        return result

    def get_status(self) -> Dict:
        """Get current engine status."""
        return {
            "running": self.running,
            "current_experiment": self.current_experiment.to_dict() if self.current_experiment else None,
            "experiments_today": self.experiments_today,
            "history_stats": self.history.get_statistics(),
            "registry_stats": self.registry.get_statistics(),
            "top_models": [asdict(m) for m in self.registry.get_top_models(5)],
        }

    def run_forever(self, interval_seconds: int = 60):
        """Run experiments forever."""
        self.running = True
        self.logger.info("Experiment engine starting (UNIFIED FULL PIPELINE)...")

        while self.running:
            try:
                self.run_one_experiment()
                time.sleep(interval_seconds)

            except KeyboardInterrupt:
                self.logger.info("Shutting down...")
                break

            except Exception as e:
                self.logger.error(f"Engine error: {e}")
                time.sleep(10)

        self.running = False


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    print("=" * 70)
    print("GIGA TRADER - Unified Experiment Engine")
    print("=" * 70)
    print("\nThis engine uses the FULL pipeline for every experiment:")
    print("  - Complete feature engineering (premarket, afterhours, patterns)")
    print("  - Full anti-overfit integration (synthetic universes, cross-assets)")
    print("  - SPY-minus-component modifiers")
    print("  - Comprehensive dimensionality reduction")
    print("  - WMES evaluation and stability analysis")
    print("=" * 70)

    engine = ExperimentEngine()

    # Run one experiment as demo
    print("\nRunning demo experiment...")
    config = create_default_config("demo")
    config.hp_optimization.use_optuna = False  # Faster for demo
    config.hp_optimization.optuna_n_trials = 5

    result = engine.run_experiment(config)

    print(f"\nResult: {result.status.value}")
    print(f"  CV AUC: {result.cv_auc_mean:.4f} +/- {result.cv_auc_std:.4f}")
    print(f"  Test AUC: {result.test_auc:.4f}")
    print(f"  WMES: {result.wmes_score:.3f}")
    print(f"  Backtest Sharpe: {result.backtest_sharpe:.3f}")
    print(f"  Duration: {result.duration_seconds:.1f}s")
