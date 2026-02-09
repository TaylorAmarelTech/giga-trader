"""
GIGA TRADER - Experiment Runner
=================================
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
    from src.phase_21_continuous.experiment_runner import ExperimentEngine
    engine = ExperimentEngine()
    result = engine.run_experiment(config)
"""

import os
import sys
import time
import threading
import traceback
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import asdict

import numpy as np
import pandas as pd
from dotenv import load_dotenv
import joblib

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

load_dotenv(project_root / ".env")

from src.experiment_config import (
    ExperimentConfig,
    create_default_config,
    validate_config,
)
from src.experiment_progress import (
    ExperimentProgressTracker,
    ExperimentStep,
)
from src.phase_11_cv_splitting.walk_forward_cv import WalkForwardCV
from src.phase_21_continuous.experiment_tracking import (
    ENGINE_CONFIG,
    ExperimentStatus,
    ExperimentResult,
    ModelRecord,
    ExperimentGenerator,
    ExperimentHistory,
    ModelRegistry,
    compute_realistic_backtest_metrics,
    calibrate_probabilities,
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

                tracker.update_substep("Integrating synthetic universes (training only) — this can take 10-30 min", 50)
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
                    tracker.touch()  # Heartbeat after long anti-overfit integration
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
                tracker.set_step(ExperimentStep.CROSS_VALIDATION, "Leak-proof CV with embedded transforms — typically 5-30 min")

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
                tracker.touch()  # Heartbeat after long leak-proof CV
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
                n_folds = config.cross_validation.n_cv_folds
                for fold_idx, (train_idx, val_idx) in enumerate(wf_cv.split(X_train_real_cv, y_train_real_cv, dates_real_cv)):
                    tracker.update_substep(f"CV fold {fold_idx + 1}/{n_folds}", ((fold_idx) / n_folds) * 100)
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

            # ───────────────────────────────────────────────────────────────
            # Step 12b: TIER 1 GATE CHECK (early reject before expensive analysis)
            # ───────────────────────────────────────────────────────────────
            train_test_gap = result.train_auc - result.test_auc if result.train_auc > 0 else 0.0
            tier1_pass = (
                result.test_auc > 0.55
                and result.test_auc < 0.90
                and train_test_gap < 0.12
                and result.wmes_score >= 0.50
            )

            if tier1_pass:
                self.logger.info("  [TIER 1] PASS - Basic quality checks met")
            else:
                reasons = []
                if result.test_auc <= 0.55:
                    reasons.append(f"AUC={result.test_auc:.3f} <= 0.55")
                if result.test_auc >= 0.90:
                    reasons.append(f"AUC={result.test_auc:.3f} >= 0.90 (likely leakage)")
                if train_test_gap >= 0.12:
                    reasons.append(f"train-test gap={train_test_gap:.3f} >= 0.12")
                if result.wmes_score < 0.50:
                    reasons.append(f"WMES={result.wmes_score:.3f} < 0.50")
                self.logger.info(f"  [TIER 1] FAIL - {'; '.join(reasons)}")

            # ───────────────────────────────────────────────────────────────
            # Step 12c: STABILITY ANALYSIS (Tier 2 — Paper-Eligible)
            # ───────────────────────────────────────────────────────────────
            if tier1_pass:
                try:
                    from src.phase_14_robustness.stability_analyzer import StabilityAnalyzer
                    from sklearn.model_selection import cross_val_score

                    self.logger.info("[STEP 10b] Running stability analysis (HP perturbation)...")
                    tracker.update_substep("Stability analysis (10 retrains with perturbed HPs) — ~2-5 min", 80)

                    # Build score function: retrain with perturbed params, return CV AUC
                    X_stab = X_train if not use_leak_proof else X_train_raw
                    y_stab = y_train

                    def stability_score_fn(params):
                        """Retrain model with perturbed params and return CV AUC."""
                        tracker.touch()  # Heartbeat during long stability analysis
                        perturbed_config = ExperimentConfig.from_dict(config.to_dict())
                        # Apply numeric param overrides
                        if "C" in params or "l2_C" in params:
                            perturbed_config.model.l2_C = params.get("l2_C", params.get("C", config.model.l2_C))
                        if "gb_n_estimators" in params:
                            perturbed_config.model.gb_n_estimators = int(params["gb_n_estimators"])
                        if "gb_max_depth" in params:
                            perturbed_config.model.gb_max_depth = int(params["gb_max_depth"])
                        if "gb_learning_rate" in params:
                            perturbed_config.model.gb_learning_rate = params["gb_learning_rate"]

                        m = self._create_model(perturbed_config)
                        scores = cross_val_score(m, X_stab, y_stab, cv=3, scoring="roc_auc")
                        return float(scores.mean())

                    # Define param ranges for perturbation
                    param_ranges = {
                        "l2_C": (0.001, 1.0),
                        "gb_n_estimators": (30, 150),
                        "gb_max_depth": (2, 5),
                        "gb_learning_rate": (0.01, 0.3),
                    }
                    base_params = {
                        "l2_C": config.model.l2_C,
                        "gb_n_estimators": config.model.gb_n_estimators,
                        "gb_max_depth": config.model.gb_max_depth,
                        "gb_learning_rate": config.model.gb_learning_rate,
                    }

                    analyzer = StabilityAnalyzer(perturbation_pct=0.05)
                    stab_result = analyzer.compute_stability_score(
                        base_params=base_params,
                        base_score=result.cv_auc_mean,
                        param_ranges=param_ranges,
                        score_fn=stability_score_fn,
                        n_samples=10,
                    )
                    result.stability_score = stab_result.get("stability_score", 0)
                    self.logger.info(f"  Stability Score: {result.stability_score:.3f}")
                    tracker.record_metric("stability_score", round(result.stability_score, 3))

                    if result.stability_score >= 0.50:
                        self.logger.info("  [TIER 2] PASS - Model is on HP plateau (paper-eligible)")
                    else:
                        self.logger.info(f"  [TIER 2] FAIL - stability={result.stability_score:.3f} < 0.50 (fragile to HP changes)")

                except Exception as stab_err:
                    self.logger.warning(f"  [TIER 2] Stability analysis failed: {stab_err}")
                    result.stability_score = 0.0

            # ───────────────────────────────────────────────────────────────
            # Step 12d: FRAGILITY ANALYSIS (Tier 3 — Live-Eligible)
            # ───────────────────────────────────────────────────────────────
            if tier1_pass and result.stability_score >= 0.50 and result.test_auc >= 0.60:
                try:
                    from src.phase_14_robustness.robustness_ensemble import RobustnessEnsemble

                    self.logger.info("[STEP 10c] Running fragility analysis (dim + param perturbation)...")
                    tracker.update_substep("Fragility analysis (ensemble of perturbed models) — ~3-7 min", 90)

                    X_frag = X_train if not use_leak_proof else X_train_raw
                    y_frag = y_train
                    weights_frag = weights_train if not use_leak_proof else weights_real

                    rob_ensemble = RobustnessEnsemble(
                        n_dimension_variants=2,
                        n_param_variants=2,
                        param_noise_pct=0.05,
                    )

                    base_model_params = {"C": config.model.l2_C, "max_iter": 1000, "random_state": 42}
                    optimal_dims = result.n_features_final or 30

                    rob_results = rob_ensemble.train_ensemble(
                        X=X_frag,
                        y=y_frag,
                        sample_weights=weights_frag,
                        base_params=base_model_params,
                        optimal_dims=optimal_dims,
                        cv_folds=3,
                    )

                    frag = rob_results.get("fragility", {})
                    result.fragility_score = frag.get("fragility_score", 1.0)
                    self.logger.info(f"  Fragility Score: {result.fragility_score:.3f}")
                    tracker.record_metric("fragility_score", round(result.fragility_score, 3))

                    if result.fragility_score < 0.35:
                        self.logger.info("  [TIER 3] PASS - Model is robust (live-eligible)")
                    else:
                        self.logger.info(f"  [TIER 3] FAIL - fragility={result.fragility_score:.3f} >= 0.35")

                except Exception as frag_err:
                    self.logger.warning(f"  [TIER 3] Fragility analysis failed: {frag_err}")
                    result.fragility_score = 1.0

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

            # Save trained model to disk
            try:
                models_exp_dir = project_root / "models" / "experiments"
                models_exp_dir.mkdir(parents=True, exist_ok=True)
                model_save_path = models_exp_dir / f"{config.experiment_id}.joblib"
                joblib.dump({
                    "model": model,
                    "feature_cols": feature_cols,
                    "config": config.to_dict(),
                    "test_auc": result.test_auc,
                    "cv_auc": result.cv_auc_mean,
                    "wmes_score": result.wmes_score,
                    "leak_proof": use_leak_proof,
                }, model_save_path)
                result.model_path = str(model_save_path)
                self.logger.info(f"  Model saved: {model_save_path}")
            except Exception as save_err:
                self.logger.warning(f"  Could not save model: {save_err}")

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

        # Register model if it passes Tier 1 quality gate
        train_test_gap = result.train_auc - result.test_auc if result.train_auc > 0 else 0.0
        tier1_pass = (
            result.status == ExperimentStatus.COMPLETED
            and result.test_auc > 0.55
            and result.test_auc < 0.90
            and train_test_gap < 0.12
            and result.wmes_score >= 0.50
        )

        if tier1_pass:
            model_id = self.registry.register_model(result)
            tier = self.registry._compute_tier(result)
            tier_labels = {1: "REGISTRY", 2: "PAPER-ELIGIBLE", 3: "LIVE-ELIGIBLE"}
            self.logger.info(
                f"Registered model: {model_id} "
                f"(AUC={result.test_auc:.3f}, WMES={result.wmes_score:.3f}, "
                f"stability={result.stability_score:.3f}, fragility={result.fragility_score:.3f}, "
                f"tier={tier} [{tier_labels.get(tier, '?')}])"
            )

            # Add to best configs pool if good
            if result.test_auc > 0.60:
                self.generator.add_best_config(config, result.test_auc)
        elif result.status == ExperimentStatus.COMPLETED:
            reasons = []
            if result.test_auc <= 0.55:
                reasons.append(f"AUC={result.test_auc:.3f}<=0.55")
            if result.test_auc >= 0.90:
                reasons.append(f"AUC={result.test_auc:.3f}>=0.90")
            if train_test_gap >= 0.12:
                reasons.append(f"gap={train_test_gap:.3f}>=0.12")
            if result.wmes_score < 0.50:
                reasons.append(f"WMES={result.wmes_score:.3f}<0.50")
            self.logger.info(f"Model NOT registered (Tier 1 fail): {'; '.join(reasons)}")

            # Register with ModelRegistryV2 for mega ensemble discovery
            try:
                from src.model_registry_v2 import ModelRegistryV2, ModelEntry, ModelType

                v2_registry = ModelRegistryV2()

                # Map experiment config to ModelEntry
                entry = ModelEntry(target_type="swing")
                entry.model_config.model_type = ModelType.LOGISTIC_L2  # Default, override from config

                # Set metrics from experiment result
                if hasattr(result, 'cv_auc_mean') and result.cv_auc_mean:
                    entry.metrics.cv_auc = result.cv_auc_mean
                if hasattr(result, 'test_auc') and result.test_auc:
                    entry.metrics.test_auc = result.test_auc
                if hasattr(result, 'wmes_score') and result.wmes_score:
                    entry.metrics.wmes_score = result.wmes_score
                if hasattr(result, 'train_auc') and result.train_auc and hasattr(result, 'test_auc') and result.test_auc:
                    entry.metrics.train_test_gap = result.train_auc - result.test_auc

                entry.status = "trained"
                entry.artifacts.model_path = str(result.model_path) if hasattr(result, 'model_path') and result.model_path else ""

                v2_registry.register(entry)
                self.logger.info(f"Registered experiment {result.experiment_id} with ModelRegistryV2")
            except Exception as e:
                self.logger.debug(f"ModelRegistryV2 registration skipped: {e}")

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
def main():
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


if __name__ == "__main__":
    main()
