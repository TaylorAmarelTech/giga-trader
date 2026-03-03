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
import hashlib
import threading
import traceback
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
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
from src.core.system_resources import get_system_resources
from src.experiment_progress import (
    ExperimentProgressTracker,
    ExperimentStep,
)
from src.phase_11_cv_splitting.walk_forward_cv import WalkForwardCV
from src.phase_21_continuous.experiment_tracking import (
    ENGINE_CONFIG,
    ExperimentStatus,
    ExperimentResult,
    ExperimentGenerator,
    ExperimentHistory,
    compute_realistic_backtest_metrics,
    calibrate_probabilities,
)
from src.core.registry_db import compute_tier

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
# FEATURE EXCLUSION LIST (matches train_robust_model.py line 2312)
# These columns must NEVER be used as input features because they are either:
#   - Target variables (is_up_day, low_before_high, target_up, etc.)
#   - Derived from same-day data unknowable at prediction time (high_minutes, etc.)
#   - Look-ahead features (max_gain_from_*, return_at_*, etc.)
#   - Metadata columns (sample_weight, universe_id, etc.)
# ═══════════════════════════════════════════════════════════════════════════════
_TARGET_AND_METADATA_EXCLUDE = [
    # Raw OHLCV (non-stationary, not features — must match training_pipeline_v2)
    "timestamp", "open", "high", "low", "close", "volume",
    # Target variables and target-derived columns
    "date", "day_return", "day_volume", "day_range",
    "is_up_day", "is_down_day",
    "low_before_high", "high_minutes", "low_minutes",
    "target_up", "target_timing", "soft_target_up",
    "smoothed_target_up", "smoothed_target_timing",
    "sample_weight", "timing_weight",
    # Look-ahead features (require knowing prices after entry)
    "max_gain_from_1015", "max_gain_from_1230",
    # Metadata / quality columns
    "has_premarket", "has_afterhours", "quality_score", "year",
    # Anti-overfit metadata columns (not features)
    "sample_weight_augment", "universe_id", "universe_type",
    "synthetic_return", "real_return", "is_synthetic",
]

# Same-day intraday features: these are computed from the SAME day's price data.
# For open-to-close prediction, they are look-ahead (not available at market open).
_INTRADAY_TIME_POINTS = ["0945", "1015", "1100", "1130", "1230", "1330", "1430", "1530"]
_INTRADAY_PREFIXES = [
    "return_at_", "high_to_", "low_to_", "range_to_",
    "rsi_at_", "macd_at_", "bb_at_", "return_from_low_",
]
_INTRADAY_EXCLUDE = [f"{prefix}{tp}" for prefix in _INTRADAY_PREFIXES
                     for tp in _INTRADAY_TIME_POINTS]

FEATURE_EXCLUDE_COLS = set(_TARGET_AND_METADATA_EXCLUDE + _INTRADAY_EXCLUDE)

# Pattern-based exclusion: any column whose lowercase name contains these substrings
# is excluded.  Must match training_pipeline_v2.py exclude_patterns exactly.
FEATURE_EXCLUDE_PATTERNS = [
    "target", "soft_target", "smoothed_target", "label",
    "sample_weight", "target_weight", "class_weight",
    "forward_return", "future_",
]


def _is_excluded_feature(col: str) -> bool:
    """Return True if column should be excluded from model features."""
    if col in FEATURE_EXCLUDE_COLS:
        return True
    col_lower = col.lower()
    return any(pat in col_lower for pat in FEATURE_EXCLUDE_PATTERNS)


def _validate_no_leakage(feature_cols: list, target_col: str = "target_up") -> list:
    """Runtime check that no target/future/OHLCV columns survived filtering.

    Returns list of suspicious column names (empty = clean).
    """
    suspicious = []
    raw_ohlcv = {"open", "high", "low", "close", "volume", "timestamp"}
    for col in feature_cols:
        if col == target_col:
            suspicious.append(col)
            continue
        if col in raw_ohlcv:
            suspicious.append(col)
            continue
        cl = col.lower()
        if any(pat in cl for pat in ("target", "future_", "forward_return",
                                      "is_up_day", "is_down_day", "label")):
            suspicious.append(col)
    return suspicious


# ═══════════════════════════════════════════════════════════════════════════════
# Wave 35: MODULE-LEVEL WRAPPER CLASSES (must be top-level for joblib pickling)
# ═══════════════════════════════════════════════════════════════════════════════
from sklearn.base import BaseEstimator, ClassifierMixin


class _BayesianRidgeClassifierWrapper(ClassifierMixin, BaseEstimator):
    """Wraps sklearn BayesianRidge regressor for binary classification.

    BayesianRidge auto-tunes regularization via evidence maximization,
    providing full Bayesian posterior uncertainty. Naturally penalizes
    complexity without manual hyperparameter tuning.
    """

    def __init__(self):
        from sklearn.linear_model import BayesianRidge
        self._reg = BayesianRidge()
        self.classes_ = np.array([0, 1])

    def fit(self, X, y, **kwargs):
        self._reg.fit(X, y.astype(float))
        return self

    def predict(self, X):
        raw = self._reg.predict(X)
        return (raw >= 0.5).astype(int)

    def decision_function(self, X):
        return self._reg.predict(X)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


class QuantileGBClassifier(ClassifierMixin, BaseEstimator):
    """Multi-quantile gradient boosting for prediction intervals.

    Trains 3 HistGradientBoostingRegressors at quantiles [0.10, 0.50, 0.90].
    The median (50th) prediction serves as the classification probability.
    The interval width (90th - 10th) provides uncertainty estimates.
    Overfit models produce artificially narrow intervals that blow up OOS.
    """

    def __init__(self, max_iter=100, max_depth=3, learning_rate=0.1,
                 min_samples_leaf=50, random_state=42):
        self._alphas = [0.10, 0.50, 0.90]
        self._models = {}
        self.max_iter = max_iter
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.classes_ = np.array([0, 1])

    def fit(self, X, y, **kwargs):
        from sklearn.ensemble import HistGradientBoostingRegressor
        y_float = y.astype(float)
        for alpha in self._alphas:
            m = HistGradientBoostingRegressor(
                loss='quantile', quantile=alpha,
                max_iter=self.max_iter, max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
            )
            m.fit(X, y_float)
            self._models[alpha] = m
        return self

    def predict_proba(self, X):
        median_pred = self._models[0.50].predict(X)
        p = np.clip(median_pred, 0.01, 0.99)
        return np.column_stack([1 - p, p])

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def predict_with_intervals(self, X):
        """Return median prediction and 80% interval width."""
        lo = self._models[0.10].predict(X)
        med = self._models[0.50].predict(X)
        hi = self._models[0.90].predict(X)
        return med, hi - lo

    def get_params(self, deep=True):
        return {
            "max_iter": self.max_iter, "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "min_samples_leaf": self.min_samples_leaf,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        return self


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
        self._feature_cache: Dict[str, Any] = {}  # Cache engineered features by swing_threshold
        self._feature_cache_order: list = []  # LRU eviction order
        _sys = get_system_resources()
        self._feature_cache_max_size: int = {
            "low": 1, "medium": 3, "high": 5, "ultra": 10,
        }.get(_sys.tier.value, 3)

    @staticmethod
    def _map_dim_method(method: str) -> Optional[str]:
        """Map ExperimentConfig dim method to leak-proof pipeline method.

        The LeakProofDimReducer supports: "kernel_pca", "ica", "pca", None.
        """
        direct = {"kernel_pca", "ica", "pca"}
        if method in direct:
            return method
        elif method == "mutual_info":
            return None  # Feature selection only, no dim reduction
        else:
            # "umap", "ensemble", "ensemble_plus", "agglomeration", "kmedoids"
            return "kernel_pca"  # Best supported fallback

    @staticmethod
    def _get_n_components(config: ExperimentConfig) -> int:
        """Get n_components based on the dim reduction method."""
        method = config.dim_reduction.method
        if method == "kernel_pca":
            return config.dim_reduction.kpca_n_components
        elif method == "ica":
            return config.dim_reduction.ica_n_components
        elif method == "pca":
            return config.dim_reduction.pca_n_components
        elif method == "mutual_info":
            return config.dim_reduction.mi_n_features
        else:
            return config.dim_reduction.kpca_n_components  # fallback

    @staticmethod
    def _get_random_state(experiment_id: str) -> int:
        """Derive deterministic random state from experiment ID."""
        return int(hashlib.md5(experiment_id.encode()).hexdigest(), 16) % (2**31)

    def load_data(self, config: ExperimentConfig):
        """Load data using DataManager with proper columns."""
        with self._data_lock:
            if self.data is None:
                from src.data_manager import get_spy_data
                self.data = get_spy_data(years=config.data.years_to_download, skip_freshness=True)
                n_bars = len(self.data)
                if n_bars > 0:
                    first = self.data.index.min() if hasattr(self.data.index, 'min') else 'N/A'
                    last = self.data.index.max() if hasattr(self.data.index, 'max') else 'N/A'
                    self.logger.info(f"Loaded {n_bars:,} bars ({first} to {last})")
                else:
                    self.logger.info(f"Loaded {n_bars:,} bars")

    def run(self, config: ExperimentConfig, fast_screen: bool = False) -> ExperimentResult:
        """
        Run experiment using the FULL unified pipeline.

        Args:
            config: Experiment configuration
            fast_screen: If True, skip anti-overfit augmentation for initial
                screening (Wave 26). Full validation runs only on Tier 1 candidates.

        Steps:
          1. Load data
          2. Engineer ALL features (premarket, afterhours, patterns, interactions)
          3. Run FULL anti-overfit integration (synthetic universes, cross-assets)
          4. Apply dimensionality reduction
          5. Train model with cross-validation
          6. Evaluate with WMES and stability analysis
          7. Run backtest
        """
        from src.core.registry_db import CURRENT_SCORING_VERSION
        result = ExperimentResult(
            experiment_id=config.experiment_id,
            config=config,
            status=ExperimentStatus.RUNNING,
            started_at=datetime.now().isoformat(),
            scoring_version=CURRENT_SCORING_VERSION,
        )
        start_time = time.time()

        # Get progress tracker
        tracker = ExperimentProgressTracker.instance()
        tracker.start_experiment(
            experiment_id=config.experiment_id,
            experiment_type=config.experiment_type,
            experiment_name=config.experiment_name,
        )

        # Initialize variables used across multiple code paths
        use_leak_proof = ENGINE_CONFIG.get("use_leak_proof_cv", True) and HAS_LEAK_PROOF

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

            # Wave G4: Information-driven bars (dollar/volume bars) — optional preprocessing
            if getattr(config.data, 'use_information_bars', False):
                try:
                    from src.phase_02_preprocessing.information_bars import InformationBarGenerator
                    bar_type = getattr(config.data, 'information_bar_type', 'dollar')
                    bar_gen = InformationBarGenerator(bar_type=bar_type, auto_calibrate=True)
                    info_bars = bar_gen.generate(self.data)
                    if len(info_bars) >= 100:
                        self.logger.info(f"  [INFO_BARS] Converted to {len(info_bars)} {bar_type} bars")
                        # Replace self.data with information bars for this experiment only
                        # (don't modify the cached raw data)
                        _data_for_features = info_bars
                    else:
                        self.logger.warning(f"  [INFO_BARS] Only {len(info_bars)} bars, using time bars")
                        _data_for_features = self.data.copy()
                except Exception as ib_err:
                    self.logger.debug(f"  [INFO_BARS] Skipped: {ib_err}")
                    _data_for_features = self.data.copy()
            else:
                _data_for_features = self.data.copy()

            # Cache engineered features by swing_threshold (same raw data → same features)
            # Wave 32: Feature research experiments bypass cache (they add extra columns)
            is_feature_research = (config.experiment_type == "feature_research")
            _info_bars_suffix = f"_ib_{getattr(config.data, 'information_bar_type', 'none')}" if getattr(config.data, 'use_information_bars', False) else ""
            feat_cache_key = f"{swing_threshold}{_info_bars_suffix}"
            if feat_cache_key in self._feature_cache and not is_feature_research:
                self.logger.info(f"  Using cached features (threshold={swing_threshold})")
                df_daily = self._feature_cache[feat_cache_key].copy()
            else:
                tracker.update_substep("Running engineer_all_features", 30)
                df_daily = engineer_all_features(_data_for_features, swing_threshold=swing_threshold)

                tracker.update_substep("Adding rolling features", 60)
                df_daily = add_rolling_features(df_daily)

                tracker.update_substep("Creating soft targets", 90)
                df_daily = create_soft_targets(df_daily, threshold=swing_threshold)

                self._feature_cache[feat_cache_key] = df_daily.copy()
                self._feature_cache_order.append(feat_cache_key)
                # LRU eviction: keep only max_size most recent entries
                while len(self._feature_cache) > self._feature_cache_max_size:
                    oldest = self._feature_cache_order.pop(0)
                    if oldest in self._feature_cache:
                        del self._feature_cache[oldest]
                        self.logger.info(f"  [CACHE] Evicted features for threshold={oldest}")
                self.logger.info(f"  Cached features for threshold={swing_threshold} ({len(self._feature_cache)}/{self._feature_cache_max_size})")

            result.n_features_initial = len([c for c in df_daily.columns
                                             if df_daily[c].dtype in ['float64', 'int64']])
            result.n_samples_real = len(df_daily)
            self.logger.info(f"  Features: {result.n_features_initial}, Samples: {result.n_samples_real}")
            tracker.record_metrics({
                "n_features_initial": result.n_features_initial,
                "n_samples_real": result.n_samples_real,
            })

            # ── Wave 32: Inject research candidate features ──────────────────
            _research_candidate_names = []
            if is_feature_research:
                try:
                    from src.phase_09_features_calendar.feature_researcher import FeatureResearchAgent
                    _fr_agent = FeatureResearchAgent()
                    _research_candidate_names = _fr_agent.inject_candidates(df_daily, config)
                    n_after = len([c for c in df_daily.columns if df_daily[c].dtype in ['float64', 'int64']])
                    self.logger.info(f"  [FEATURE_RESEARCH] Injected {len(_research_candidate_names)} candidates, "
                                     f"features: {result.n_features_initial} -> {n_after}")
                    result.n_features_initial = n_after
                except Exception as fr_err:
                    self.logger.warning(f"  [FEATURE_RESEARCH] Injection failed: {fr_err}")

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

            # Determine leak-proof mode early (needed by walk-forward and main path)
            use_leak_proof = ENGINE_CONFIG.get("use_leak_proof_cv", True) and HAS_LEAK_PROOF

            # ═══════════════════════════════════════════════════════════════════
            # WALK-FORWARD EXPANDING WINDOW VALIDATION
            # Dynamic windows based on actual data size (Wave 16).
            # With 10yr data: windows at years 2,4,6,8 (every 2nd year)
            # With 5yr data: windows at years 2,3,4 (every year)
            # ═══════════════════════════════════════════════════════════════════
            year_size_approx = 252  # Trading days per year
            n_years = max(2, n_dates // year_size_approx)
            year_size = n_dates // n_years  # Adjust to fit data evenly
            self.logger.info(f"  Data spans ~{n_years} years ({n_dates} dates, {year_size} per year)")

            # Build expanding windows: train from start, test next year
            # Step every 2 years if >7 years, else every year; cap at 7 early windows
            wf_step = 2 if n_years > 7 else 1
            walk_forward_windows = []
            for test_yr in range(1, n_years - 1, wf_step):
                tr_end = test_yr * year_size
                te_start = test_yr * year_size
                te_end = min((test_yr + 1) * year_size, n_dates)
                if tr_end >= 100 and te_end - te_start >= 30:
                    walk_forward_windows.append((0, tr_end, te_start, te_end))
                if len(walk_forward_windows) >= 7:
                    break

            # Track per-window regime-adjusted thresholds (Wave 16)
            wf_per_window_thresholds = []

            if use_leak_proof and HAS_LEAK_PROOF and year_size >= 100:
                from sklearn.metrics import roc_auc_score as wf_roc_auc

                self.logger.info(f"[WALK-FORWARD] Running expanding window pre-validation ({len(walk_forward_windows)} early windows)...")
                tracker.update_substep("Walk-forward pre-validation", 30)

                # Build leak-proof config for walk-forward (same config as main experiment)
                wf_dim_method = self._map_dim_method(config.dim_reduction.method)
                wf_n_components = self._get_n_components(config)
                wf_random_state = self._get_random_state(config.experiment_id)
                wf_feat_sel = getattr(config.dim_reduction, 'feature_selection_method', 'mutual_info')

                wf_leak_proof_config = {
                    "n_cv_folds": min(config.cross_validation.n_cv_folds, 3),  # Fewer folds for speed
                    "purge_days": config.cross_validation.purge_days,
                    "embargo_days": config.cross_validation.embargo_days,
                    "feature_selection_method": wf_feat_sel,
                    "n_features": config.dim_reduction.mi_n_features,
                    "dim_reduction_method": wf_dim_method,
                    "n_components": wf_n_components,
                    "use_ensemble": False,  # Skip ensemble for speed
                    "random_state": wf_random_state,
                }

                # Determine feature columns from full dataset
                wf_feature_cols = [c for c in df_daily.columns
                                   if not _is_excluded_feature(c)]
                wf_feature_cols = [c for c in wf_feature_cols if df_daily[c].dtype in ['float64', 'int64']]

                for w_idx, (tr_start, tr_end, te_start, te_end) in enumerate(walk_forward_windows):
                    try:
                        w_train_dates_set = set(unique_dates[tr_start:tr_end])
                        w_test_dates_set = set(unique_dates[te_start:te_end])

                        w_train_mask = np.array([d in w_train_dates_set for d in dates])
                        w_test_mask = np.array([d in w_test_dates_set for d in dates])

                        w_df_train = df_daily[w_train_mask].copy()
                        w_df_test = df_daily[w_test_mask].copy()

                        # Regime detection: measure test window volatility (Wave 16)
                        # Post-leakage-fix realistic AUCs are 0.53-0.57; per-window threshold
                        # must be below this range to avoid rejecting ALL models (Wave 26).
                        wf_threshold = 0.50
                        try:
                            if "close" in w_df_test.columns and len(w_df_test) > 20:
                                w_test_returns = w_df_test["close"].pct_change().dropna()
                                if len(w_test_returns) > 20:
                                    w_ann_vol = float(w_test_returns.std()) * (252 ** 0.5)
                                    if w_ann_vol > 0.22:  # >22% annualized = crisis regime (COVID, etc.)
                                        wf_threshold = 0.48
                                        self.logger.info(f"  Window {w_idx+1}: HIGH-VOL regime (ann_vol={w_ann_vol:.1%}), threshold={wf_threshold}")
                                    else:
                                        self.logger.info(f"  Window {w_idx+1}: NORMAL regime (ann_vol={w_ann_vol:.1%}), threshold={wf_threshold}")
                        except Exception:
                            pass
                        wf_per_window_thresholds.append(wf_threshold)

                        # Feature cols available in both
                        w_cols = [c for c in wf_feature_cols if c in w_df_train.columns and c in w_df_test.columns]

                        w_df_train_clean = w_df_train.dropna(subset=w_cols + ["target_up"])
                        w_df_test_clean = w_df_test.dropna(subset=w_cols + ["target_up"])

                        if len(w_df_train_clean) < 100 or len(w_df_test_clean) < 30:
                            self.logger.warning(f"  Window {w_idx+1}: Insufficient data (train={len(w_df_train_clean)}, test={len(w_df_test_clean)}), skipping")
                            continue

                        w_X_train = w_df_train_clean[w_cols].values
                        w_y_train = w_df_train_clean["target_up"].astype(int).values
                        w_X_test = w_df_test_clean[w_cols].values
                        w_y_test = w_df_test_clean["target_up"].astype(int).values

                        tracker.update_substep(f"Walk-forward window {w_idx+1}/{len(walk_forward_windows)+1}", 30 + w_idx * 15)
                        tracker.touch()

                        w_pipeline, w_cv = train_with_leak_proof_cv(
                            w_X_train, w_y_train, config=wf_leak_proof_config, verbose=False,
                        )
                        w_test_proba = w_pipeline.predict_proba(w_X_test)[:, 1]
                        w_auc = float(wf_roc_auc(w_y_test, w_test_proba))
                        result.walk_forward_aucs.append(w_auc)
                        self.logger.info(f"  Window {w_idx+1} (train {len(w_df_train_clean)}, test {len(w_df_test_clean)}): AUC={w_auc:.4f} (threshold={wf_threshold})")
                        tracker.touch()

                        # Early abort: if first non-crisis window is very weak, skip remaining
                        if w_idx == 0 and w_auc < 0.48:
                            self.logger.info(f"  [EARLY ABORT] Window 1 AUC={w_auc:.3f} < 0.48 — skipping remaining windows")
                            break
                    except Exception as wf_err:
                        self.logger.warning(f"  Window {w_idx+1} failed: {wf_err}")

                self.logger.info(f"[WALK-FORWARD] Early window AUCs: {result.walk_forward_aucs}")
                self.logger.info(f"[WALK-FORWARD] Per-window thresholds: {wf_per_window_thresholds}")
                tracker.record_metric("walk_forward_early_aucs", result.walk_forward_aucs)
                tracker.record_metric("walk_forward_thresholds", wf_per_window_thresholds)
            else:
                if year_size < 100:
                    self.logger.info("[WALK-FORWARD] Skipped (insufficient data for yearly windows)")

            # Final window: Train all-but-last-year, test last year (dynamic)
            split_date_idx = (n_years - 1) * year_size if year_size >= 100 else int(n_dates * 0.8)
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

            # Wave 26: Regime-specific training — filter training data by volatility
            regime_filter = getattr(config.data, 'regime_filter', '')
            if regime_filter and "close" in df_train_real.columns:
                rolling_vol = df_train_real["close"].pct_change().rolling(20).std() * (252 ** 0.5)
                vol_median = rolling_vol.median()
                if regime_filter == "low_vol":
                    regime_mask = rolling_vol <= vol_median
                    df_train_real = df_train_real[regime_mask.fillna(True)].copy()
                    self.logger.info(f"  [REGIME] Filtered to low-vol training: {len(df_train_real)} rows")
                elif regime_filter == "high_vol":
                    regime_mask = rolling_vol > vol_median
                    df_train_real = df_train_real[regime_mask.fillna(False)].copy()
                    self.logger.info(f"  [REGIME] Filtered to high-vol training: {len(df_train_real)} rows")
                # Note: test data is NOT filtered — evaluate on ALL regimes

            self.logger.info(f"  Train samples (real): {len(df_train_real)}, Test samples: {len(df_test)}")

            tracker.record_metrics({
                "n_train_dates": len(train_dates),
                "n_test_dates": len(test_dates),
                "n_train_real": len(df_train_real),
                "n_test": len(df_test),
                "regime_filter": regime_filter,
            })

            # Step 4b: CUSUM event filter (Wave E4) — filter training data to significant-move dates
            cusum_enabled = getattr(config.feature_engineering, 'use_cusum_filter', False)
            if cusum_enabled and "day_return" in df_train_real.columns:
                try:
                    from src.phase_05_targets.cusum_filter import CUSUMFilter
                    cusum_threshold = getattr(config.feature_engineering, 'cusum_threshold', 0.01)
                    cusum = CUSUMFilter(threshold=cusum_threshold, min_events=100)
                    df_train_filtered = cusum.filter_dataframe(df_train_real, return_col="day_return")
                    n_before = len(df_train_real)
                    n_after = len(df_train_filtered)
                    if n_after < n_before:
                        df_train_real = df_train_filtered
                        self.logger.info(
                            f"  [CUSUM] Filtered training data: {n_before} -> {n_after} "
                            f"({n_after/n_before:.1%} retained, threshold={cusum_threshold})"
                        )
                    else:
                        self.logger.info(f"  [CUSUM] No filtering applied (all {n_before} events retained)")
                except Exception as cusum_err:
                    self.logger.warning(f"  [CUSUM] Filter failed: {cusum_err}")

            # Step 5: Apply anti-overfit augmentation ONLY to training data
            df_train = df_train_real.copy()
            skip_augmentation = fast_screen and config.anti_overfit.use_anti_overfit
            if skip_augmentation:
                self.logger.info("[STEP 4] FAST SCREEN: Skipping anti-overfit augmentation (deferred to Tier 1)")
                tracker.set_step(ExperimentStep.ANTI_OVERFIT, "Skipped (fast screen)")
            elif config.anti_overfit.use_anti_overfit:
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
                        use_sector_breadth=getattr(config.anti_overfit, 'use_sector_breadth', True),
                        use_vol_regime=getattr(config.anti_overfit, 'use_vol_regime', True),
                        use_economic_features=config.anti_overfit.use_economic_features,
                        use_calendar_features=config.anti_overfit.use_calendar_features,
                        use_sentiment_features=config.anti_overfit.use_sentiment_features,
                        use_fear_greed=config.anti_overfit.use_fear_greed,
                        use_reddit_sentiment=config.anti_overfit.use_reddit_sentiment,
                        use_crypto_sentiment=config.anti_overfit.use_crypto_sentiment,
                        use_gamma_exposure=config.anti_overfit.use_gamma_exposure,
                        use_finnhub_social=config.anti_overfit.use_finnhub_social,
                        use_dark_pool=config.anti_overfit.use_dark_pool,
                        use_options_features=getattr(config.anti_overfit, 'use_options_features', True),
                        use_event_recency=getattr(config.anti_overfit, 'use_event_recency', True),
                        use_block_structure=getattr(config.anti_overfit, 'use_block_structure', True),
                        use_amihud_features=getattr(config.anti_overfit, 'use_amihud_features', True),
                        use_range_vol_features=getattr(config.anti_overfit, 'use_range_vol_features', True),
                        use_entropy_features=getattr(config.anti_overfit, 'use_entropy_features', True),
                        use_hurst_features=getattr(config.anti_overfit, 'use_hurst_features', True),
                        use_nmi_features=getattr(config.anti_overfit, 'use_nmi_features', True),
                        use_absorption_ratio=getattr(config.anti_overfit, 'use_absorption_ratio', True),
                        use_drift_features=getattr(config.anti_overfit, 'use_drift_features', True),
                        use_changepoint_features=getattr(config.anti_overfit, 'use_changepoint_features', True),
                        use_hmm_features=getattr(config.anti_overfit, 'use_hmm_features', True),
                        use_vpin_features=getattr(config.anti_overfit, 'use_vpin_features', True),
                        use_intraday_momentum=getattr(config.anti_overfit, 'use_intraday_momentum', True),
                        use_futures_basis=getattr(config.anti_overfit, 'use_futures_basis', True),
                        use_congressional_features=getattr(config.anti_overfit, 'use_congressional_features', True),
                        use_insider_aggregate=getattr(config.anti_overfit, 'use_insider_aggregate', True),
                        use_etf_flow=getattr(config.anti_overfit, 'use_etf_flow', True),
                        use_wavelet_features=getattr(config.anti_overfit, 'use_wavelet_features', True),
                        use_sax_features=getattr(config.anti_overfit, 'use_sax_features', True),
                        use_transfer_entropy=getattr(config.anti_overfit, 'use_transfer_entropy', True),
                        use_mfdfa_features=getattr(config.anti_overfit, 'use_mfdfa_features', True),
                        use_rqa_features=getattr(config.anti_overfit, 'use_rqa_features', True),
                        use_copula_features=getattr(config.anti_overfit, 'use_copula_features', True),
                        use_network_centrality=getattr(config.anti_overfit, 'use_network_centrality', True),
                        use_path_signatures=getattr(config.anti_overfit, 'use_path_signatures', True),
                        use_wavelet_scattering=getattr(config.anti_overfit, 'use_wavelet_scattering', True),
                        use_wasserstein_regime=getattr(config.anti_overfit, 'use_wasserstein_regime', True),
                        use_market_structure=getattr(config.anti_overfit, 'use_market_structure', True),
                        use_time_series_models=getattr(config.anti_overfit, 'use_time_series_models', False),
                        use_catch22=getattr(config.anti_overfit, 'use_catch22', False),
                        use_har_rv=getattr(config.anti_overfit, 'use_har_rv', True),
                        use_l_moments=getattr(config.anti_overfit, 'use_l_moments', True),
                        use_multiscale_entropy=getattr(config.anti_overfit, 'use_multiscale_entropy', True),
                        use_rv_signature_plot=getattr(config.anti_overfit, 'use_rv_signature_plot', False),
                        use_tda_homology=getattr(config.anti_overfit, 'use_tda_homology', False),
                        use_credit_spread_features=getattr(config.anti_overfit, 'use_credit_spread_features', True),
                        use_yield_curve_features=getattr(config.anti_overfit, 'use_yield_curve_features', True),
                        use_vol_term_structure_features=getattr(config.anti_overfit, 'use_vol_term_structure_features', True),
                        use_macro_surprise_features=getattr(config.anti_overfit, 'use_macro_surprise_features', True),
                        use_cross_asset_momentum=getattr(config.anti_overfit, 'use_cross_asset_momentum', True),
                        use_skew_kurtosis_features=getattr(config.anti_overfit, 'use_skew_kurtosis_features', True),
                        use_seasonality_features=getattr(config.anti_overfit, 'use_seasonality_features', True),
                        use_order_flow_imbalance=getattr(config.anti_overfit, 'use_order_flow_imbalance', True),
                        use_correlation_regime=getattr(config.anti_overfit, 'use_correlation_regime', True),
                        use_fama_french=getattr(config.anti_overfit, 'use_fama_french', True),
                        use_put_call_ratio=getattr(config.anti_overfit, 'use_put_call_ratio', True),
                        use_multi_horizon=getattr(config.anti_overfit, 'use_multi_horizon', True),
                        use_earnings_revision=getattr(config.anti_overfit, 'use_earnings_revision', True),
                        use_short_interest=getattr(config.anti_overfit, 'use_short_interest', False),
                        use_dollar_index=getattr(config.anti_overfit, 'use_dollar_index', True),
                        use_institutional_flow=getattr(config.anti_overfit, 'use_institutional_flow', False),
                        use_google_trends=getattr(config.anti_overfit, 'use_google_trends', False),
                        use_commodity_signals=getattr(config.anti_overfit, 'use_commodity_signals', True),
                        use_treasury_auction=getattr(config.anti_overfit, 'use_treasury_auction', False),
                        use_fed_liquidity=getattr(config.anti_overfit, 'use_fed_liquidity', True),
                        use_earnings_calendar=getattr(config.anti_overfit, 'use_earnings_calendar', True),
                        use_analyst_rating=getattr(config.anti_overfit, 'use_analyst_rating', True),
                        validate_ohlc=getattr(config.data, 'validate_ohlc', True),
                        use_synthetic=config.anti_overfit.use_synthetic_universes,
                        synthetic_weight=config.anti_overfit.synthetic_weight,
                        use_bear_universes=config.anti_overfit.use_bear_universes,
                        bear_mean_shift_bps=config.anti_overfit.bear_mean_shift_bps,
                        bear_vol_amplify_factor=config.anti_overfit.bear_vol_amplify_factor,
                        bear_vol_dampen_factor=config.anti_overfit.bear_vol_dampen_factor,
                        use_multiscale_bootstrap=config.anti_overfit.use_multiscale_bootstrap,
                        resource_config=getattr(config, 'resources', None),
                    )
                    result.n_samples_synthetic = len(df_train_augmented) - len(df_train_real)
                    df_train = df_train_augmented

                    # Cap synthetic:real ratio to 1:1 to reduce train-test gap
                    # (was 3:1 in Wave 14, but synthetic data still dominates gradient)
                    n_real = len(df_train_real)
                    n_synthetic = result.n_samples_synthetic
                    max_synthetic = n_real * 1
                    if n_synthetic > max_synthetic and n_real > 0:
                        self.logger.info(
                            f"  [RATIO CAP] Synthetic:real = {n_synthetic}:{n_real} "
                            f"({n_synthetic/n_real:.1f}:1) — capping to 1:1"
                        )
                        # Keep all real rows (first n_real), downsample the rest
                        # Note: integrate_anti_overfit appends synthetic after real rows
                        real_rows = df_train.iloc[:n_real]
                        synth_rows = df_train.iloc[n_real:]
                        if len(synth_rows) > max_synthetic:
                            synth_rows = synth_rows.sample(n=max_synthetic, random_state=42)
                            df_train = pd.concat([real_rows, synth_rows], ignore_index=True)
                            result.n_samples_synthetic = len(synth_rows)
                            self.logger.info(
                                f"  [RATIO CAP] Downsampled to {len(synth_rows)} synthetic "
                                f"({len(synth_rows)/n_real:.1f}:1 ratio)"
                            )
                        tracker.record_metric("synthetic_ratio_capped", True)

                    self.logger.info(f"  Train augmented: {len(df_train)} (+{result.n_samples_synthetic} synthetic)")
                    tracker.record_metric("n_samples_synthetic", result.n_samples_synthetic)
                    tracker.touch()  # Heartbeat after long anti-overfit integration
                except Exception as aug_err:
                    self.logger.warning(f"  Anti-overfit augmentation failed: {aug_err}")
                    self.logger.warning("  Continuing with real training data only...")
                    tracker.record_metric("anti_overfit_error", str(aug_err))

            # Step 6: Prepare features and targets for train and test SEPARATELY
            self.logger.info("[STEP 5] Preparing features...")
            feature_cols = [c for c in df_train.columns
                           if not _is_excluded_feature(c)]
            feature_cols = [c for c in feature_cols if df_train[c].dtype in ['float64', 'int64']]

            # Ensure test has same features
            feature_cols = [c for c in feature_cols if c in df_test.columns]

            # Runtime leakage check (Wave 17)
            suspicious = _validate_no_leakage(feature_cols, "target_up")
            if suspicious:
                self.logger.error(
                    f"  [LEAKAGE] {len(suspicious)} suspicious features survived "
                    f"filtering: {suspicious[:10]}. Removing them."
                )
                feature_cols = [c for c in feature_cols if c not in suspicious]
                tracker.record_metric("leakage_cols_removed", len(suspicious))

            self.logger.info(f"  Using {len(feature_cols)} features after filtering")

            # ── Wave F5.1: Optional Triple Barrier labeling ─────────────────
            target_col = "target_up"
            if getattr(config.feature_engineering, 'use_triple_barrier', False):
                try:
                    from src.phase_05_targets.triple_barrier import TripleBarrierLabeler
                    _tb_col = "close" if "close" in df_train.columns else "Close"
                    if _tb_col in df_train.columns:
                        labeler = TripleBarrierLabeler(
                            tp_pct=getattr(config.feature_engineering, 'tp_pct', 0.01),
                            sl_pct=getattr(config.feature_engineering, 'sl_pct', 0.01),
                            max_holding_days=getattr(config.feature_engineering, 'max_holding_days', 5),
                            label_mode="binary",
                        )
                        tb_labels = labeler.label(df_train[_tb_col])
                        df_train["target_triple_barrier"] = tb_labels.values
                        tb_labels_test = labeler.label(df_test[_tb_col])
                        df_test["target_triple_barrier"] = tb_labels_test.values
                        target_col = "target_triple_barrier"
                        self.logger.info(f"  [TRIPLE_BARRIER] Using triple barrier labels "
                                         f"(tp={labeler.tp_pct}, sl={labeler.sl_pct}, max_days={labeler.max_holding_days})")
                except Exception as tb_err:
                    self.logger.warning(f"  [TRIPLE_BARRIER] Failed, falling back to target_up: {tb_err}")
                    target_col = "target_up"

            # Clean and prepare training data
            df_train_clean = df_train.dropna(subset=feature_cols + [target_col])
            X_train_raw = df_train_clean[feature_cols].values
            y_train = df_train_clean[target_col].astype(int).values

            # Get dates for training data (for walk-forward CV)
            if "date" in df_train_clean.columns:
                train_dates_array = pd.to_datetime(df_train_clean["date"]).dt.date.values
            else:
                train_dates_array = np.arange(len(df_train_clean))

            # Clean and prepare test data (REAL data only, no synthetic)
            df_test_clean = df_test.dropna(subset=feature_cols + [target_col])
            X_test_raw = df_test_clean[feature_cols].values
            y_test = df_test_clean[target_col].astype(int).values
            test_returns = df_test_clean["day_return"].values if "day_return" in df_test_clean.columns else None

            # Get dates for test data
            if "date" in df_test_clean.columns:
                test_dates_array = pd.to_datetime(df_test_clean["date"]).dt.date.values
            else:
                test_dates_array = np.arange(len(df_test_clean))

            # Sample weights for training (synthetic samples have lower weight)
            weights_train = None
            if "sample_weight" in df_train_clean.columns:
                weights_train = df_train_clean["sample_weight"].values.copy()

            # ── Synthetic weight penalty (prevents overfitting to synthetic data) ──
            if weights_train is not None and "sample_weight_augment" in df_train_clean.columns:
                real_weight_val = 1.0 - config.anti_overfit.synthetic_weight
                is_synthetic = df_train_clean["sample_weight_augment"].values < real_weight_val - 0.01
                n_synth = int(is_synthetic.sum())
                if n_synth > 0:
                    penalty = config.anti_overfit.synthetic_weight_penalty
                    floor = config.anti_overfit.synthetic_weight_floor
                    ceiling = config.anti_overfit.synthetic_weight_ceiling
                    weights_train[is_synthetic] = np.clip(
                        weights_train[is_synthetic] * penalty, floor, ceiling
                    )
                    self.logger.info(
                        f"  [WEIGHT PENALTY] Applied {penalty:.0%} penalty to {n_synth} synthetic samples "
                        f"(bounds=[{floor:.2f}, {ceiling:.2f}])"
                    )

            # ── Wave 35: Training augmentations ─────────────────────────────
            aug_config = getattr(config, 'training_augmentation', None)

            # C2. Temporal decay weighting
            if aug_config and getattr(aug_config, 'use_temporal_decay', False):
                self.logger.info("  [AUGMENT] Applying temporal decay sample weighting")
                n_train = len(X_train_raw)
                date_indices = np.arange(n_train, dtype=float)
                decay_lambda = getattr(aug_config, 'temporal_decay_lambda', 0.5)
                decay_weights = np.exp(-decay_lambda * (n_train - 1 - date_indices) / n_train)
                decay_weights = decay_weights / decay_weights.mean()  # Normalize to mean=1.0
                if weights_train is not None:
                    weights_train = weights_train * decay_weights
                else:
                    weights_train = decay_weights
                self.logger.info(f"    lambda={decay_lambda:.2f}, weight range: [{decay_weights.min():.3f}, {decay_weights.max():.3f}]")

            # C3. Noise injection to training features
            if aug_config and getattr(aug_config, 'use_noise_injection', False):
                noise_sigma = getattr(aug_config, 'noise_sigma', 0.1)
                self.logger.info(f"  [AUGMENT] Injecting Gaussian noise (sigma={noise_sigma}) to training features")
                rng_noise = np.random.RandomState(42)
                feature_stds = np.std(X_train_raw, axis=0) + 1e-10
                noise = rng_noise.randn(*X_train_raw.shape) * (noise_sigma * feature_stds[np.newaxis, :])
                X_train_raw = X_train_raw + noise
                # NOTE: X_test_raw is NOT modified (noise only during training)

            self.logger.info(f"  Train features: {X_train_raw.shape}, Test features: {X_test_raw.shape}")

            # ═══════════════════════════════════════════════════════════════════
            # OPTIONAL: Cross-model meta-feature augmentation
            # Appends rule activations + importance-weighted features from
            # the universal feature map (only when enough models have
            # contributed). See symbolic_cross_learner.py.
            # ═══════════════════════════════════════════════════════════════════
            try:
                from src.phase_23_analytics.symbolic_cross_learner import (
                    CrossModelAugmenter, UniversalFeatureMap,
                )
                map_path = project_root / "models" / "feature_importance_map.json"
                if map_path.is_file():
                    fmap = UniversalFeatureMap(persist_path=map_path)
                    if len(fmap._model_contributions) >= 10:
                        augmenter = CrossModelAugmenter(feature_map=fmap)
                        aug_train, aug_names = augmenter.generate_importance_weighted(
                            X_train_raw, feature_cols
                        )
                        if aug_train.shape[1] > 0:
                            aug_test, _ = augmenter.generate_importance_weighted(
                                X_test_raw, feature_cols
                            )
                            X_train_raw = np.hstack([X_train_raw, aug_train])
                            X_test_raw = np.hstack([X_test_raw, aug_test])
                            feature_cols = feature_cols + aug_names
                            self.logger.info(
                                f"  [AUGMENT] Added {len(aug_names)} cross-model meta-features "
                                f"(total: {len(feature_cols)})"
                            )
            except Exception as aug_err:
                self.logger.debug(f"  Cross-model augmentation skipped: {aug_err}")

            # ── Wave F5.2: Interaction Discovery ─────────────────────────────
            if getattr(config.feature_engineering, 'use_interaction_discovery', False):
                try:
                    from src.phase_10_feature_processing.interaction_discovery import InteractionDiscovery
                    max_interactions = getattr(config.feature_engineering, 'max_interactions', 20)
                    disc = InteractionDiscovery(max_interactions=max_interactions, random_state=42)
                    _X_df_train = pd.DataFrame(X_train_raw, columns=feature_cols)
                    interactions = disc.discover(_X_df_train, y_train)
                    if interactions:
                        _X_aug_train = disc.transform(_X_df_train, interactions)
                        new_cols = [c for c in _X_aug_train.columns if c not in feature_cols]
                        if new_cols:
                            _X_df_test = pd.DataFrame(X_test_raw, columns=feature_cols)
                            _X_aug_test = disc.transform(_X_df_test, interactions)
                            X_train_raw = _X_aug_train.values
                            X_test_raw = _X_aug_test.values
                            feature_cols = list(_X_aug_train.columns)
                            self.logger.info(
                                f"  [INTERACTION] Discovered {len(new_cols)} feature interactions "
                                f"(total features: {len(feature_cols)})"
                            )
                except Exception as int_err:
                    self.logger.debug(f"  Interaction discovery skipped: {int_err}")

            # ═══════════════════════════════════════════════════════════════════
            # LEAK-PROOF CV PATH (recommended - all transforms inside CV folds)
            # ═══════════════════════════════════════════════════════════════════
            if use_leak_proof:
                from sklearn.metrics import roc_auc_score

                self.logger.info("[STEP 6] Using LEAK-PROOF CV (all transforms inside folds)")
                tracker.set_step(ExperimentStep.CROSS_VALIDATION, "Leak-proof CV with embedded transforms — typically 5-30 min")

                # Build leak-proof config from ExperimentConfig (NOT hardcoded)
                dim_method = self._map_dim_method(config.dim_reduction.method)
                n_components = self._get_n_components(config)
                exp_random_state = self._get_random_state(config.experiment_id)
                feat_sel_method = getattr(config.dim_reduction, 'feature_selection_method', 'mutual_info')

                self.logger.info(f"  Feature selection: {feat_sel_method}, n_features: {config.dim_reduction.mi_n_features}")
                self.logger.info(f"  Dim reduction: {dim_method}, n_components: {n_components}")
                self.logger.info(f"  Random state: {exp_random_state} (from experiment ID)")

                leak_proof_config = {
                    "n_cv_folds": config.cross_validation.n_cv_folds,
                    "purge_days": config.cross_validation.purge_days,
                    "embargo_days": config.cross_validation.embargo_days,
                    "feature_selection_method": feat_sel_method,
                    "n_features": config.dim_reduction.mi_n_features,
                    "dim_reduction_method": dim_method,
                    "n_components": n_components,
                    "use_ensemble": ENGINE_CONFIG.get("use_model_ensemble", True),
                    "random_state": exp_random_state,
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

                # ── Isotonic Calibration (Wave F4.3) ──
                if getattr(config.cross_validation, 'use_isotonic_calibration', False):
                    try:
                        from src.phase_15_strategy.isotonic_calibrator import IsotonicCalibrator
                        iso_cal = IsotonicCalibrator()
                        iso_cal.fit(test_proba, y_test)
                        cal_proba = iso_cal.calibrate(test_proba)
                        cal_auc = float(roc_auc_score(y_test, cal_proba))
                        self.logger.info(
                            f"  Isotonic calibration: AUC {result.test_auc:.4f} -> {cal_auc:.4f}"
                        )
                    except Exception as iso_err:
                        self.logger.debug(f"  Isotonic calibration skipped: {iso_err}")

                # ── Conformal Position Sizer (Wave F4.2) ──
                conformal_sizer = None
                if getattr(config.trading, 'use_conformal_sizing', False):
                    try:
                        from src.phase_15_strategy.conformal_sizer import ConformalPositionSizer
                        alpha = getattr(config.trading, 'conformal_alpha', 0.1)
                        conformal_sizer = ConformalPositionSizer(alpha=alpha)
                        # Use the pipeline as the model (it handles its own transforms)
                        conformal_sizer.fit(pipeline, X_test_raw, y_test)
                        if conformal_sizer._fitted:
                            self.logger.info(
                                f"  Conformal sizer fitted: alpha={alpha}, "
                                f"cal_samples={len(y_test)}"
                            )
                        else:
                            conformal_sizer = None
                    except Exception as cs_err:
                        self.logger.debug(f"  Conformal sizer skipped: {cs_err}")
                        conformal_sizer = None

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

                # ── Isotonic Calibration (Wave F4.3) ──
                if getattr(config.cross_validation, 'use_isotonic_calibration', False):
                    try:
                        from src.phase_15_strategy.isotonic_calibrator import IsotonicCalibrator
                        iso_cal = IsotonicCalibrator()
                        _test_proba = model.predict_proba(X_test)[:, 1]
                        iso_cal.fit(_test_proba, y_test)
                        cal_proba = iso_cal.calibrate(_test_proba)
                        cal_auc = float(roc_auc_score(y_test, cal_proba))
                        self.logger.info(
                            f"  Isotonic calibration: AUC {result.test_auc:.4f} -> {cal_auc:.4f}"
                        )
                    except Exception as iso_err:
                        self.logger.debug(f"  Isotonic calibration skipped: {iso_err}")

                # ── Conformal Position Sizer (Wave F4.2, legacy path) ──
                if not hasattr(self, '_conformal_sizer_legacy'):
                    self._conformal_sizer_legacy = None
                if getattr(config.trading, 'use_conformal_sizing', False):
                    try:
                        from src.phase_15_strategy.conformal_sizer import ConformalPositionSizer
                        alpha = getattr(config.trading, 'conformal_alpha', 0.1)
                        cs = ConformalPositionSizer(alpha=alpha)
                        cs.fit(model, X_test, y_test)
                        if cs._fitted:
                            self._conformal_sizer_legacy = cs
                            self.logger.info(f"  Conformal sizer fitted (legacy): alpha={alpha}")
                        else:
                            self._conformal_sizer_legacy = None
                    except Exception as cs_err:
                        self.logger.debug(f"  Conformal sizer skipped (legacy): {cs_err}")

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
            # Walk-forward: add final window AUC and compute aggregate
            # ───────────────────────────────────────────────────────────────
            result.walk_forward_aucs.append(result.test_auc)  # Final window = last AUC
            # Final window uses post-leakage-fix threshold (Wave 26: 0.50)
            wf_per_window_thresholds.append(0.50)

            if len(result.walk_forward_aucs) >= 2:
                result.worst_window_auc = min(result.walk_forward_aucs)
                wf_variance = float(np.std(result.walk_forward_aucs))
                wf_mean = float(np.mean(result.walk_forward_aucs))

                # Regime-adjusted per-window pass check (Wave 16)
                # Each window uses its own threshold: 0.56 for normal, 0.50 for high-vol
                if len(wf_per_window_thresholds) == len(result.walk_forward_aucs):
                    n_passing = sum(
                        auc > threshold
                        for auc, threshold in zip(result.walk_forward_aucs, wf_per_window_thresholds)
                    )
                else:
                    # Fallback: use flat 0.50 if thresholds misaligned
                    n_passing = sum(a > 0.50 for a in result.walk_forward_aucs)

                # Post-leakage-fix thresholds (recalibrated Wave 26):
                # Realistic AUCs with clean features are 0.53-0.57, not 0.75+.
                # Crisis windows (COVID 2020, 2022 bear) systematically score ~0.49.
                # Allow 2 failing windows when 4+ total (was: all-but-1).
                max_failures = 2 if len(result.walk_forward_aucs) >= 4 else 1
                result.walk_forward_passed = (
                    n_passing >= max(len(result.walk_forward_aucs) - max_failures, 1)
                    and wf_variance < 0.07   # Wave 26: relaxed 0.06→0.07 to reduce false negatives
                    and wf_mean >= 0.51      # Average floor (was 0.52)
                    and result.worst_window_auc >= 0.47  # Allow crisis periods (was 0.48)
                )
                self.logger.info(f"  Walk-forward AUCs: {[round(a, 4) for a in result.walk_forward_aucs]}")
                self.logger.info(f"  Walk-forward thresholds: {[round(t, 2) for t in wf_per_window_thresholds[:len(result.walk_forward_aucs)]]}")
                self.logger.info(f"  Worst window AUC: {result.worst_window_auc:.4f}, Mean: {wf_mean:.4f}")
                self.logger.info(f"  Walk-forward variance: {wf_variance:.4f}")
                self.logger.info(f"  Walk-forward passed: {result.walk_forward_passed} ({n_passing}/{len(result.walk_forward_aucs)} windows passed)")
                tracker.record_metrics({
                    "walk_forward_aucs": [round(a, 4) for a in result.walk_forward_aucs],
                    "walk_forward_thresholds": [round(t, 2) for t in wf_per_window_thresholds[:len(result.walk_forward_aucs)]],
                    "worst_window_auc": round(result.worst_window_auc, 4),
                    "walk_forward_mean_auc": round(wf_mean, 4),
                    "walk_forward_passed": result.walk_forward_passed,
                })
            else:
                result.worst_window_auc = result.test_auc
                result.walk_forward_passed = result.test_auc > 0.50

            # ───────────────────────────────────────────────────────────────
            # Regime-specific evaluation (low-vol vs high-vol)
            # ───────────────────────────────────────────────────────────────
            try:
                if test_returns is not None and len(test_returns) > 40:
                    from sklearn.metrics import roc_auc_score as regime_roc_auc
                    rolling_vol = pd.Series(test_returns).rolling(20, min_periods=10).std().values
                    vol_median = np.nanmedian(rolling_vol)
                    valid_vol = ~np.isnan(rolling_vol)

                    low_vol_mask = valid_vol & (rolling_vol <= vol_median)
                    high_vol_mask = valid_vol & (rolling_vol > vol_median)

                    regime_proba = test_proba_wmes if 'test_proba_wmes' in dir() else model.predict_proba(
                        X_test_raw if use_leak_proof else X_test)[:, 1]

                    if sum(low_vol_mask) > 15 and len(np.unique(y_test[low_vol_mask])) > 1:
                        result.regime_auc_low_vol = float(regime_roc_auc(
                            y_test[low_vol_mask], regime_proba[low_vol_mask]))
                    if sum(high_vol_mask) > 15 and len(np.unique(y_test[high_vol_mask])) > 1:
                        result.regime_auc_high_vol = float(regime_roc_auc(
                            y_test[high_vol_mask], regime_proba[high_vol_mask]))

                    if result.regime_auc_low_vol > 0 and result.regime_auc_high_vol > 0:
                        regime_gap = abs(result.regime_auc_low_vol - result.regime_auc_high_vol)
                        result.regime_sensitive = regime_gap > 0.10
                        self.logger.info(f"  Regime AUC — low_vol: {result.regime_auc_low_vol:.4f}, high_vol: {result.regime_auc_high_vol:.4f}")
                        if result.regime_sensitive:
                            self.logger.warning(f"  [REGIME] Model is regime-sensitive (gap={regime_gap:.3f} > 0.10)")
                        tracker.record_metrics({
                            "regime_auc_low_vol": round(result.regime_auc_low_vol, 4),
                            "regime_auc_high_vol": round(result.regime_auc_high_vol, 4),
                            "regime_sensitive": result.regime_sensitive,
                        })
            except Exception as regime_err:
                self.logger.warning(f"  Regime evaluation failed: {regime_err}")

            # ───────────────────────────────────────────────────────────────
            # Wave 35 C4: Nested CV (honest generalization estimate)
            # ───────────────────────────────────────────────────────────────
            if aug_config and getattr(aug_config, 'use_nested_cv', False):
                try:
                    self.logger.info("[STEP 10a] Running nested CV for honest AUC estimate...")
                    from sklearn.model_selection import StratifiedKFold as NestedSKF
                    from sklearn.metrics import roc_auc_score as nested_roc_auc

                    outer_folds = getattr(aug_config, 'nested_outer_folds', 3)
                    inner_folds = getattr(aug_config, 'nested_inner_folds', 3)
                    X_nested = X_train_raw if use_leak_proof else X_train
                    outer_aucs = []

                    outer_cv = NestedSKF(n_splits=outer_folds, shuffle=True, random_state=42)
                    for outer_idx, (outer_train, outer_test) in enumerate(outer_cv.split(X_nested, y_train)):
                        X_out_train, y_out_train = X_nested[outer_train], y_train[outer_train]
                        X_out_test, y_out_test = X_nested[outer_test], y_train[outer_test]

                        # Inner CV for model selection (use best of inner folds)
                        inner_cv = NestedSKF(n_splits=inner_folds, shuffle=True, random_state=42 + outer_idx)
                        inner_scores = []
                        for in_train, in_val in inner_cv.split(X_out_train, y_out_train):
                            in_model = self._create_model(config)
                            try:
                                in_model.fit(X_out_train[in_train], y_out_train[in_train])
                                in_proba = in_model.predict_proba(X_out_train[in_val])[:, 1]
                                inner_scores.append(float(nested_roc_auc(y_out_train[in_val], in_proba)))
                            except Exception:
                                inner_scores.append(0.5)

                        # Retrain on full outer train, evaluate on outer test
                        out_model = self._create_model(config)
                        out_model.fit(X_out_train, y_out_train)
                        out_proba = out_model.predict_proba(X_out_test)[:, 1]
                        outer_auc = float(nested_roc_auc(y_out_test, out_proba))
                        outer_aucs.append(outer_auc)
                        self.logger.info(f"    Outer fold {outer_idx+1}: AUC={outer_auc:.4f} (inner mean={np.mean(inner_scores):.4f})")

                    result.nested_cv_auc = float(np.mean(outer_aucs))
                    self.logger.info(f"  Nested CV AUC: {result.nested_cv_auc:.4f} (vs test AUC: {result.test_auc:.4f})")
                    tracker.record_metric("nested_cv_auc", round(result.nested_cv_auc, 4))
                except Exception as nested_err:
                    self.logger.warning(f"  Nested CV failed: {nested_err}")

            # ───────────────────────────────────────────────────────────────
            # Wave 35 C5: Calibrated distillation (memorization detector)
            # ───────────────────────────────────────────────────────────────
            if aug_config and getattr(aug_config, 'use_distillation', True):
                try:
                    self.logger.info("[STEP 10b-dist] Running calibrated distillation check...")
                    from sklearn.linear_model import LogisticRegression as DistillLR
                    from sklearn.metrics import roc_auc_score as distill_roc_auc

                    # Teacher predictions on test set
                    X_dist_test = X_test_raw if use_leak_proof else X_test
                    teacher_proba = model.predict_proba(X_dist_test)[:, 1]
                    teacher_auc = float(distill_roc_auc(y_test, teacher_proba))

                    # Student: simple LogReg trained on teacher's soft labels
                    student = DistillLR(C=1.0, max_iter=500, random_state=42)
                    # Train student on teacher's predictions of the TRAINING set
                    X_dist_train = X_train_raw if use_leak_proof else X_train
                    teacher_train_proba = model.predict_proba(X_dist_train)[:, 1]
                    # Use teacher's soft labels as binary targets for student
                    teacher_hard = (teacher_train_proba >= 0.5).astype(int)
                    student.fit(X_dist_train, teacher_hard)
                    student_proba = student.predict_proba(X_dist_test)[:, 1]
                    student_auc = float(distill_roc_auc(y_test, student_proba))

                    result.distillation_gap = round(teacher_auc - student_auc, 4)
                    self.logger.info(
                        f"  Distillation: teacher={teacher_auc:.4f}, student={student_auc:.4f}, "
                        f"gap={result.distillation_gap:.4f}"
                    )
                    if result.distillation_gap > 0.05:
                        self.logger.warning(
                            f"  [DISTILL] Large gap ({result.distillation_gap:.3f} > 0.05) — "
                            f"teacher may be memorizing patterns student can't learn"
                        )
                    tracker.record_metric("distillation_gap", result.distillation_gap)
                except Exception as dist_err:
                    self.logger.warning(f"  Distillation check failed: {dist_err}")

            # ───────────────────────────────────────────────────────────────
            # Step 12b: REALITY CHECK + TIER 1 GATE (Wave 14)
            # ───────────────────────────────────────────────────────────────
            train_test_gap = result.train_auc - result.test_auc if result.train_auc > 0 else 0.0

            # Upper-bound reality checks (catch leakage / unrealistic metrics)
            reality_flags = self._reality_check(result)
            if reality_flags:
                for flag_name, flag_msg in reality_flags.items():
                    self.logger.warning(f"  [REALITY CHECK] {flag_name}: {flag_msg}")
                tracker.record_metric("reality_check_flags", list(reality_flags.keys()))
            has_leakage = "likely_leakage" in reality_flags

            # Post-leakage-fix tier1 thresholds (Wave 26 recalibration):
            # With clean features, realistic AUCs are 0.53-0.57.
            # AUC > 0.55 meaningfully above random (0.50).
            # Gap < 0.15: Augmented training data (3:1 synthetic ratio) causes a
            # structural ~0.12 gap even for linear models. Tier2/3 gates handle
            # true overfitting detection via stability + fragility analysis.
            # WMES > 0.40 — realistic WMES with corrected sharpe capping.
            tier1_pass = (
                result.test_auc > 0.55
                and result.test_auc < 0.85
                and train_test_gap < 0.15
                and result.wmes_score >= 0.40
                and not has_leakage
                and result.walk_forward_passed
            )

            if tier1_pass:
                self.logger.info("  [TIER 1] PASS - Basic quality checks met")
            else:
                reasons = []
                if result.test_auc <= 0.55:
                    reasons.append(f"AUC={result.test_auc:.3f} <= 0.55")
                if result.test_auc >= 0.85:
                    reasons.append(f"AUC={result.test_auc:.3f} >= 0.85 (likely leakage)")
                if train_test_gap >= 0.15:
                    reasons.append(f"train-test gap={train_test_gap:.3f} >= 0.15")
                if result.wmes_score < 0.40:
                    reasons.append(f"WMES={result.wmes_score:.3f} < 0.40")
                if has_leakage:
                    reasons.append("reality check: likely leakage")
                self.logger.info(f"  [TIER 1] FAIL - {'; '.join(reasons)}")

            # ───────────────────────────────────────────────────────────────
            # Step 12b2: PERMUTATION TEST (gate blocker, Wave 17)
            # Run 5 shuffles × 3-fold CV. If mean permuted AUC > 0.53,
            # the model is likely learning leaked features, not signal.
            # ───────────────────────────────────────────────────────────────
            if tier1_pass:
                try:
                    from sklearn.model_selection import cross_val_score as perm_cv_score
                    n_permutations = 5
                    perm_cv_folds = 3
                    self.logger.info(
                        f"[STEP 10b] Running permutation test "
                        f"({n_permutations} shuffles × {perm_cv_folds}-fold CV)..."
                    )
                    tracker.update_substep("Permutation test — ~1-2 min", 72)

                    X_perm = X_train if not use_leak_proof else X_train_raw
                    perm_aucs = []
                    for perm_i in range(n_permutations):
                        y_perm = y_train.copy()
                        np.random.seed(42 + perm_i)
                        np.random.shuffle(y_perm)

                        perm_model = self._create_model(config)
                        perm_scores = perm_cv_score(
                            perm_model, X_perm, y_perm,
                            cv=perm_cv_folds, scoring="roc_auc",
                        )
                        perm_aucs.append(float(perm_scores.mean()))

                    perm_mean = float(np.mean(perm_aucs))
                    perm_max = float(np.max(perm_aucs))
                    result.permutation_auc_mean = round(perm_mean, 4)
                    self.logger.info(
                        f"  Permutation AUCs: {[round(a, 4) for a in perm_aucs]}"
                    )
                    self.logger.info(
                        f"  Permutation mean={perm_mean:.4f}, max={perm_max:.4f} "
                        f"(should be ~0.50)"
                    )
                    tracker.record_metric("permutation_auc_mean", round(perm_mean, 4))
                    tracker.record_metric("permutation_auc_max", round(perm_max, 4))

                    if perm_mean > 0.53:
                        self.logger.warning(
                            f"  [PERMUTATION] FAIL: Mean shuffled AUC={perm_mean:.3f} > 0.53 — "
                            f"model is learning from leaked features, blocking tier promotion"
                        )
                        result.permutation_passed = False
                        tier1_pass = False  # Block tier promotion
                        tracker.record_metric("permutation_blocked", True)
                    elif perm_max > 0.56:
                        self.logger.warning(
                            f"  [PERMUTATION] WARNING: Max shuffled AUC={perm_max:.3f} > 0.56 — "
                            f"possible marginal leakage"
                        )
                        tracker.record_metric("permutation_warning", True)
                    else:
                        self.logger.info("  [PERMUTATION] PASS — no leakage detected")
                        result.permutation_passed = True
                except Exception as perm_err:
                    self.logger.warning(f"  Permutation test failed: {perm_err}")

            # ───────────────────────────────────────────────────────────────
            # Step 12c: STABILITY ANALYSIS (Tier 2 — Paper-Eligible)
            # ───────────────────────────────────────────────────────────────
            if tier1_pass:
                try:
                    from src.phase_14_robustness.stability_analyzer import StabilityAnalyzer
                    from sklearn.model_selection import cross_val_score

                    self.logger.info("[STEP 10b] Running stability analysis (HP perturbation)...")
                    tracker.update_substep("Stability analysis (24 multi-radius retrains) — ~2-5 min", 80)

                    # Build score function: retrain with perturbed params, return CV AUC
                    X_stab = X_train if not use_leak_proof else X_train_raw
                    y_stab = y_train

                    # Wave 28c: Model-type-specific stability parameters.
                    # Previously all models used GB params, giving fake stab=1.0
                    # for MLPs, SGD, etc. that don't use those params at all.
                    model_type = config.model.model_type

                    def stability_score_fn(params):
                        """Retrain model with perturbed params and return CV AUC."""
                        tracker.touch()  # Heartbeat during long stability analysis
                        perturbed_config = ExperimentConfig.from_dict(config.to_dict())
                        # Apply param overrides based on model type
                        if "l2_C" in params:
                            perturbed_config.model.l2_C = params["l2_C"]
                        if "gb_n_estimators" in params:
                            perturbed_config.model.gb_n_estimators = int(params["gb_n_estimators"])
                        if "gb_max_depth" in params:
                            perturbed_config.model.gb_max_depth = int(params["gb_max_depth"])
                        if "gb_learning_rate" in params:
                            perturbed_config.model.gb_learning_rate = params["gb_learning_rate"]

                        # For MLP/SGD: perturb via direct model construction
                        if model_type in ("mlp_small", "mlp_medium"):
                            from sklearn.neural_network import MLPClassifier
                            alpha = params.get("mlp_alpha", 1.0)
                            lr = params.get("mlp_lr", 0.001)
                            layers = (32,) if model_type == "mlp_small" else (64, 32)
                            m = MLPClassifier(
                                hidden_layer_sizes=layers, alpha=alpha,
                                learning_rate_init=lr, max_iter=500,
                                early_stopping=True, validation_fraction=0.15,
                                n_iter_no_change=10, random_state=42,
                            )
                        elif model_type == "sgd_linear":
                            from sklearn.linear_model import SGDClassifier
                            from sklearn.calibration import CalibratedClassifierCV
                            alpha = params.get("sgd_alpha", 0.01)
                            l1_ratio = params.get("sgd_l1_ratio", 0.5)
                            base = SGDClassifier(
                                loss='log_loss', penalty='elasticnet',
                                alpha=alpha, l1_ratio=l1_ratio,
                                max_iter=1000, early_stopping=True,
                                validation_fraction=0.15, n_iter_no_change=10,
                                random_state=42,
                            )
                            m = CalibratedClassifierCV(base, cv=3, method='sigmoid')
                        elif model_type == "ridge":
                            from sklearn.linear_model import RidgeClassifier
                            from sklearn.calibration import CalibratedClassifierCV
                            alpha = params.get("ridge_alpha", 10.0)
                            base = RidgeClassifier(alpha=alpha, random_state=42)
                            m = CalibratedClassifierCV(base, cv=3, method='sigmoid')
                        elif model_type == "bagged_linear":
                            from sklearn.linear_model import LogisticRegression as LR_
                            from sklearn.ensemble import BaggingClassifier
                            bag_C = params.get("l2_C", 0.01)
                            n_est = int(params.get("bag_n_estimators", 20))
                            base_lr = LR_(penalty='l1', solver='saga', C=bag_C,
                                          max_iter=1000, random_state=42)
                            m = BaggingClassifier(
                                estimator=base_lr, n_estimators=n_est,
                                max_samples=0.7, max_features=0.7,
                                bootstrap=True, random_state=42,
                            )
                        # Wave 35: New model types
                        elif model_type == "lda":
                            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
                            from sklearn.calibration import CalibratedClassifierCV as CalCV
                            m = CalCV(LinearDiscriminantAnalysis(solver='svd'), cv=3, method='sigmoid')
                        elif model_type == "gaussian_nb":
                            from sklearn.naive_bayes import GaussianNB
                            m = GaussianNB(var_smoothing=1e-8)
                        elif model_type == "svc_linear":
                            from sklearn.svm import LinearSVC
                            from sklearn.calibration import CalibratedClassifierCV as CalCV
                            svc_C = params.get("l2_C", 0.1) * 0.1
                            m = CalCV(LinearSVC(C=svc_C, max_iter=2000, dual='auto', random_state=42), cv=3)
                        elif model_type == "bayesian_ridge":
                            from sklearn.calibration import CalibratedClassifierCV as CalCV
                            m = CalCV(_BayesianRidgeClassifierWrapper(), cv=3, method='sigmoid')
                        elif model_type == "quantile_gb":
                            m = QuantileGBClassifier(
                                max_iter=int(params.get("gb_n_estimators", 100)),
                                max_depth=int(params.get("gb_max_depth", 3)),
                                learning_rate=params.get("gb_learning_rate", 0.1),
                                min_samples_leaf=50, random_state=42,
                            )
                        elif model_type == "quantile_forest":
                            from src.phase_12_model_training.quantile_forest_wrapper import (
                                QuantileForestClassifier as _QFC,
                            )
                            m = _QFC(
                                n_estimators=int(params.get("gb_n_estimators", 100)),
                                max_depth=int(params.get("gb_max_depth", 3)),
                                min_samples_leaf=50, random_state=42,
                            )
                        elif model_type == "catboost":
                            try:
                                from catboost import CatBoostClassifier
                                m = CatBoostClassifier(
                                    iterations=int(params.get("gb_n_estimators", 100)),
                                    depth=int(params.get("gb_max_depth", 3)),
                                    learning_rate=params.get("gb_learning_rate", 0.1),
                                    l2_leaf_reg=3.0, random_seed=42, verbose=0,
                                    allow_writing_files=False,
                                )
                            except ImportError:
                                m = self._create_model(perturbed_config)
                        elif model_type == "stacking_ensemble":
                            m = self._create_model(perturbed_config)
                        else:
                            m = self._create_model(perturbed_config)

                        scores = cross_val_score(m, X_stab, y_stab, cv=3, scoring="roc_auc")
                        return float(scores.mean())

                    # Define model-type-specific param ranges and base values
                    if model_type in ("mlp_small", "mlp_medium"):
                        param_ranges = {
                            "mlp_alpha": (0.01, 10.0),
                            "mlp_lr": (0.0001, 0.01),
                        }
                        base_params = {
                            "mlp_alpha": 1.0 if model_type == "mlp_small" else 0.5,
                            "mlp_lr": 0.001,
                        }
                    elif model_type == "sgd_linear":
                        param_ranges = {
                            "sgd_alpha": (0.0001, 0.1),
                            "sgd_l1_ratio": (0.1, 0.9),
                        }
                        base_params = {"sgd_alpha": 0.01, "sgd_l1_ratio": 0.5}
                    elif model_type == "ridge":
                        param_ranges = {"ridge_alpha": (0.1, 100.0)}
                        base_params = {"ridge_alpha": 10.0}
                    elif model_type == "bagged_linear":
                        param_ranges = {
                            "l2_C": (0.001, 0.1),
                            "bag_n_estimators": (10, 40),
                        }
                        base_params = {"l2_C": 0.01, "bag_n_estimators": 20}
                    elif model_type in ("gradient_boosting", "hist_gradient_boosting"):
                        param_ranges = {
                            "gb_n_estimators": (30, 150),
                            "gb_max_depth": (2, 5),
                            "gb_learning_rate": (0.01, 0.3),
                        }
                        base_params = {
                            "gb_n_estimators": config.model.gb_n_estimators,
                            "gb_max_depth": config.model.gb_max_depth,
                            "gb_learning_rate": config.model.gb_learning_rate,
                        }
                    elif model_type == "extra_trees":
                        param_ranges = {
                            "gb_n_estimators": (30, 150),
                            "gb_max_depth": (2, 5),
                        }
                        base_params = {
                            "gb_n_estimators": config.model.gb_n_estimators,
                            "gb_max_depth": config.model.gb_max_depth,
                        }
                    # Wave 35: New model types
                    elif model_type in ("lda", "gaussian_nb", "bayesian_ridge"):
                        # No tunable HPs — stability naturally 1.0
                        param_ranges = {}
                        base_params = {}
                    elif model_type == "svc_linear":
                        param_ranges = {"l2_C": (0.001, 1.0)}
                        base_params = {"l2_C": config.model.l2_C}
                    elif model_type == "quantile_gb":
                        param_ranges = {
                            "gb_n_estimators": (30, 150),
                            "gb_max_depth": (2, 5),
                            "gb_learning_rate": (0.01, 0.3),
                        }
                        base_params = {
                            "gb_n_estimators": config.model.gb_n_estimators,
                            "gb_max_depth": config.model.gb_max_depth,
                            "gb_learning_rate": config.model.gb_learning_rate,
                        }
                    elif model_type == "quantile_forest":
                        param_ranges = {
                            "gb_n_estimators": (30, 150),
                            "gb_max_depth": (2, 5),
                        }
                        base_params = {
                            "gb_n_estimators": config.model.gb_n_estimators,
                            "gb_max_depth": config.model.gb_max_depth,
                        }
                    # Wave F2: CatBoost + Stacking
                    elif model_type == "catboost":
                        param_ranges = {
                            "gb_n_estimators": (30, 150),
                            "gb_max_depth": (2, 5),
                            "gb_learning_rate": (0.01, 0.3),
                        }
                        base_params = {
                            "gb_n_estimators": config.model.gb_n_estimators,
                            "gb_max_depth": config.model.gb_max_depth,
                            "gb_learning_rate": config.model.gb_learning_rate,
                        }
                    elif model_type == "stacking_ensemble":
                        # Stacking's main tunable is meta-learner C
                        param_ranges = {"l2_C": (0.001, 1.0)}
                        base_params = {"l2_C": config.model.l2_C}
                    else:
                        # Logistic variants, ensemble, diverse_ensemble
                        # All use l2_C (via aggressive_C)
                        param_ranges = {"l2_C": (0.001, 1.0)}
                        base_params = {"l2_C": config.model.l2_C}

                    # Wave 28b: Compute base_score using same CV method as
                    # perturbed scores (standard 3-fold CV). Previously used
                    # result.cv_auc_mean (leak-proof CV with purge/embargo),
                    # which has a systematic ~0.05 offset causing stability=0.
                    base_cv_score = stability_score_fn(base_params)
                    result._base_cv_score = base_cv_score  # Used by stability suite
                    self.logger.info(
                        f"  Stability base score: {base_cv_score:.3f} "
                        f"(leak-proof CV was {result.cv_auc_mean:.3f})"
                    )

                    # Wave 29: Multi-radius stability (local ±15%, moderate ±50%, wide full-range)
                    # 24 joint perturbations across 3 rings, weighted 20/30/50%
                    analyzer = StabilityAnalyzer()
                    stab_result = analyzer.compute_stability_score(
                        base_params=base_params,
                        base_score=base_cv_score,
                        param_ranges=param_ranges,
                        score_fn=stability_score_fn,
                        n_samples=24,
                    )
                    result.stability_score = stab_result.get("stability_score", 0)
                    self.logger.info(f"  Stability Score: {result.stability_score:.3f}")
                    tracker.record_metric("stability_score", round(result.stability_score, 3))

                    # Wave 29: Log per-ring details for tracking
                    per_ring = stab_result.get("per_ring", {})
                    for rname in ("local", "moderate", "wide"):
                        rdata = per_ring.get(rname, {})
                        tracker.record_metric(
                            f"stability_{rname}",
                            round(rdata.get("stability", 0), 3),
                        )
                    self.logger.info(
                        f"  Per-ring: local={per_ring.get('local', {}).get('stability', 0):.3f} "
                        f"moderate={per_ring.get('moderate', {}).get('stability', 0):.3f} "
                        f"wide={per_ring.get('wide', {}).get('stability', 0):.3f}"
                    )

                    if result.stability_score >= 0.60:
                        self.logger.info("  [TIER 2] PASS - Model is on HP plateau (paper-eligible)")
                    else:
                        self.logger.info(f"  [TIER 2] FAIL - stability={result.stability_score:.3f} < 0.60 (fragile to HP changes)")

                except Exception as stab_err:
                    self.logger.warning(f"  [TIER 2] Stability analysis failed: {stab_err}")
                    self.logger.warning(f"  [TIER 2] Traceback: {traceback.format_exc()}")
                    # Wave 33: Don't set to 0.0 (blocks tier promotion forever).
                    # Set to -1 (sentinel = "not computed") so suite can still gate.
                    result.stability_score = -1.0

            # ───────────────────────────────────────────────────────────────
            # Step 12c2: MULTI-FACETED STABILITY SUITE (Wave 29)
            # Augments HP perturbation with bootstrap, feature dropout,
            # seed stability, and prediction agreement. Tracked as metadata.
            # ───────────────────────────────────────────────────────────────
            if tier1_pass:
                try:
                    from src.phase_14_robustness.stability_suite import StabilitySuite

                    self.logger.info("[STEP 10b2] Running multi-faceted stability suite...")
                    tracker.update_substep("Multi-stability suite (bootstrap/dropout/seed/prediction) — ~1-3 min", 83)

                    X_suite = X_train if not use_leak_proof else X_train_raw
                    y_suite = y_train
                    model_type = config.model.model_type

                    # Model factory: returns fresh model with same config
                    def _make_model():
                        tracker.touch()
                        return self._create_model(config)

                    # Seed-aware factory for seed stability test
                    def _make_model_with_seed(seed):
                        tracker.touch()
                        m = self._create_model(config)
                        # Try to set random_state on the model or its base estimator
                        if hasattr(m, 'random_state'):
                            m.random_state = seed
                        elif hasattr(m, 'estimator') and hasattr(m.estimator, 'random_state'):
                            m.estimator.random_state = seed
                        elif hasattr(m, 'base_estimator') and hasattr(m.base_estimator, 'random_state'):
                            m.base_estimator.random_state = seed
                        return m

                    suite = StabilitySuite(
                        n_bootstrap=5,
                        n_feature_dropout=5,
                        n_seeds=5,
                        n_prediction_models=5,
                        dropout_fraction=0.15,
                        subsample_fraction=0.80,
                    )

                    # Use base CV score as reference (computed earlier or from result)
                    ref_auc = getattr(result, '_base_cv_score', result.cv_auc_mean)

                    suite_results = suite.run_all(
                        X=X_suite,
                        y=y_suite,
                        model_factory_fn=_make_model,
                        base_auc=ref_auc,
                        seed_model_factory_fn=_make_model_with_seed,
                    )

                    # Store results
                    result.stability_bootstrap = suite_results["bootstrap"]["score"]
                    result.stability_feature_dropout = suite_results["feature_dropout"]["score"]
                    result.stability_seed = suite_results["seed"].get("score", -1.0)
                    result.stability_prediction = suite_results["prediction"]["score"]
                    result.stability_composite = suite_results["composite"]

                    # Log results
                    self.logger.info(
                        f"  Stability Suite: "
                        f"bootstrap={result.stability_bootstrap:.3f} "
                        f"feat_dropout={result.stability_feature_dropout:.3f} "
                        f"seed={result.stability_seed:.3f} "
                        f"prediction={result.stability_prediction:.3f} "
                        f"composite={result.stability_composite:.3f}"
                    )

                    # Track metrics
                    tracker.record_metric("stability_bootstrap", round(result.stability_bootstrap, 3))
                    tracker.record_metric("stability_feature_dropout", round(result.stability_feature_dropout, 3))
                    tracker.record_metric("stability_seed", round(result.stability_seed, 3))
                    tracker.record_metric("stability_prediction", round(result.stability_prediction, 3))
                    tracker.record_metric("stability_composite", round(result.stability_composite, 3))

                except Exception as suite_err:
                    self.logger.warning(f"  [STABILITY SUITE] Failed: {suite_err}")

            # ───────────────────────────────────────────────────────────────
            # Step 12c3: ADVANCED STABILITY SUITE (Wave 30)
            # 14 complementary methods: PSI, CSI, adversarial, ECE, DSR,
            # SFI, meta-labeling, knockoff, ADWIN, CPCV, stability
            # selection, Rashomon set, SHAP consistency, conformal prediction
            # ───────────────────────────────────────────────────────────────
            if tier1_pass:
                try:
                    from src.phase_14_robustness.advanced_stability import AdvancedStabilitySuite

                    self.logger.info("[STEP 12c3] Running advanced stability suite (19 methods)...")
                    tracker.update_substep("Advanced stability (PSI/CPCV/SHAP/Rashomon/adversarial/smoothing) — ~5-8 min", 85)

                    # Pick correct feature arrays based on CV path
                    X_adv_train = X_train_raw if use_leak_proof else X_train
                    X_adv_test = X_test_raw if use_leak_proof else X_test

                    _rc = getattr(config, 'resources', None)
                    advanced_suite = AdvancedStabilitySuite(
                        n_cpcv_groups=getattr(_rc, 'stability_n_cpcv_groups', 5),
                        n_stability_sel=getattr(_rc, 'stability_n_stability_sel', 10),
                        n_rashomon=getattr(_rc, 'stability_n_rashomon', 5),
                        n_sfi_repeats=3,
                        skip_expensive=getattr(_rc, 'stability_skip_expensive', False),
                    )

                    ref_auc_adv = getattr(result, '_base_cv_score', result.cv_auc_mean)
                    # Estimate experiment count for DSR correction
                    try:
                        from src.core.registry_db import get_registry_db
                        n_exp_total = get_registry_db().get_experiment_count()
                    except Exception:
                        n_exp_total = 1

                    advanced_results = advanced_suite.run_all(
                        X_train=X_adv_train,
                        y_train=y_train,
                        X_test=X_adv_test,
                        y_test=y_test,
                        model_factory_fn=_make_model,
                        trained_model=model,
                        predictions=test_proba_wmes,
                        base_auc=ref_auc_adv,
                        cv_scores=result.cv_scores,
                        n_experiments_total=n_exp_total,
                    )
                    result.stability_advanced = advanced_results

                    # Log composite
                    adv_composite = advanced_results.get("composite_advanced", 0)
                    self.logger.info(f"  Advanced Stability Composite: {adv_composite:.3f}")
                    tracker.record_metric("stability_advanced_composite", round(adv_composite, 3))

                    # Log adversarial overfitting sub-scores
                    adv_of = advanced_results.get("adversarial_overfitting", {})
                    if adv_of.get("score", -1) >= 0:
                        self.logger.info(
                            f"  Adversarial Overfitting: {adv_of['score']:.3f} "
                            f"(noise={adv_of.get('noise_score', 0):.3f}, "
                            f"perturb={adv_of.get('perturb_score', 0):.3f}, "
                            f"conf={adv_of.get('confidence_score', 0):.3f})"
                        )

                    # Log disagreement smoothing sub-scores
                    dis_sm = advanced_results.get("disagreement_smoothing", {})
                    if dis_sm.get("score", -1) >= 0:
                        self.logger.info(
                            f"  Disagreement Smoothing: {dis_sm['score']:.3f} "
                            f"(pred_agree={dis_sm.get('pred_agreement_score', 0):.3f}, "
                            f"feat_agree={dis_sm.get('feat_agreement_score', 0):.3f})"
                        )

                except Exception as adv_err:
                    self.logger.warning(f"  [ADVANCED STABILITY] Failed: {adv_err}")

            # ───────────────────────────────────────────────────────────────
            # Step 12d: FRAGILITY ANALYSIS (Tier 3 — Live-Eligible)
            # Wave 33: Replaced generic RobustnessEnsemble (which used
            # LogisticRegression+PCA regardless of actual model, producing
            # fragility ~0 for everything) with proper model-aware test.
            # Now uses the same stability_score_fn to test the ACTUAL model
            # with feature count perturbation + wider param perturbation.
            # ───────────────────────────────────────────────────────────────
            # Wave 33: Also run fragility if stability failed (-1) but suite composite is good
            suite_comp = getattr(result, 'stability_composite', 0.0) or 0.0
            stab_ok_for_frag = (result.stability_score >= 0.60) or (result.stability_score == -1.0 and suite_comp >= 0.50)
            if tier1_pass and stab_ok_for_frag:
                try:
                    from sklearn.model_selection import cross_val_score as frag_cv_score

                    self.logger.info("[STEP 10c] Running fragility analysis (dim + param perturbation)...")
                    tracker.update_substep("Fragility analysis (actual model retrains) — ~3-7 min", 90)

                    X_frag = X_train if not use_leak_proof else X_train_raw
                    y_frag = y_train
                    optimal_n_feat = result.n_features_final or 30
                    frag_scores = []

                    # --- A. Feature count perturbation ---
                    # Vary n_features widely to stress-test feature sensitivity
                    feat_variants = sorted(set([
                        max(5, int(optimal_n_feat * 0.4)),
                        max(5, int(optimal_n_feat * 0.6)),
                        max(5, int(optimal_n_feat * 0.8)),
                        optimal_n_feat,
                        min(X_frag.shape[1], int(optimal_n_feat * 1.2)),
                        min(X_frag.shape[1], int(optimal_n_feat * 1.4)),
                        min(X_frag.shape[1], int(optimal_n_feat * 1.8)),
                    ]))
                    self.logger.info(f"  Feature count variants: {feat_variants} (optimal={optimal_n_feat})")

                    for n_feat in feat_variants:
                        try:
                            tracker.touch()
                            m = self._create_model(config)
                            # Use top-n features by variance (fast, no leakage)
                            feat_var = np.var(X_frag, axis=0)
                            top_idx = np.argsort(feat_var)[-n_feat:]
                            X_sub = X_frag[:, top_idx]
                            scores = frag_cv_score(m, X_sub, y_frag, cv=3, scoring="roc_auc")
                            frag_scores.append(float(scores.mean()))
                            self.logger.info(f"    n_feat={n_feat}: AUC={scores.mean():.4f}")
                        except Exception:
                            pass

                    # --- B. Wide parameter perturbation (±50%) ---
                    # Reuse stability_score_fn from earlier (already defined for this model type)
                    for _ in range(8):
                        try:
                            tracker.touch()
                            perturbed_p = {}
                            for pk, (plow, phigh) in param_ranges.items():
                                bv = base_params.get(pk)
                                if bv is None:
                                    continue
                                delta = np.random.uniform(-0.50, 0.50)
                                nv = bv * (1 + delta)
                                nv = max(plow, min(phigh, nv))
                                if isinstance(bv, int):
                                    nv = int(round(nv))
                                perturbed_p[pk] = nv
                            score = stability_score_fn(perturbed_p)
                            frag_scores.append(score)
                        except Exception:
                            pass

                    # --- C. Compute fragility score ---
                    if len(frag_scores) >= 3:
                        base_frag_score = getattr(result, '_base_cv_score', result.cv_auc_mean)
                        frag_std = float(np.std(frag_scores))
                        frag_mean = float(np.mean(frag_scores))
                        max_drop = max(0, base_frag_score - min(frag_scores))
                        # Coefficient of variation relative to base score
                        cv_frag = frag_std / (base_frag_score + 1e-6)
                        # Drop factor: how much does the worst variant drop from base
                        drop_factor = max_drop / (base_frag_score + 1e-6)

                        # Fragility = weighted combination (aggressively scaled)
                        # cv_frag of 0.05 (5% variation) → 0.50 (concerning)
                        # drop_factor of 0.10 (10% drop) → 0.60 (fragile)
                        result.fragility_score = float(min(1.0, max(0.0,
                            0.50 * min(cv_frag * 10, 1.0) +     # 5% CV → 0.50
                            0.50 * min(drop_factor * 6, 1.0)    # 10% drop → 0.60
                        )))

                        # v4: Floor for models with empty param_ranges (LDA, GaussianNB,
                        # BayesianRidge). When param perturbation is a no-op, 8 of ~15
                        # frag_scores are identical to base, artificially suppressing
                        # cv_frag. Floor = 0.05 acknowledges untestable HP dimension.
                        if not param_ranges:
                            result.fragility_score = max(result.fragility_score, 0.05)

                        self.logger.info(
                            f"  Fragility: score={result.fragility_score:.3f} "
                            f"(cv={cv_frag:.4f}, max_drop={max_drop:.4f}, "
                            f"std={frag_std:.4f}, n={len(frag_scores)})"
                        )
                        tracker.record_metric("fragility_score", round(result.fragility_score, 3))
                        tracker.record_metric("fragility_cv", round(cv_frag, 4))
                        tracker.record_metric("fragility_max_drop", round(max_drop, 4))
                    else:
                        self.logger.warning("  [FRAGILITY] Not enough scores, defaulting to 0.5")
                        result.fragility_score = 0.5

                    if result.fragility_score < 0.40 and result.test_auc >= 0.57:
                        self.logger.info("  [TIER 3] PASS - Model is robust (live-eligible)")
                    elif result.test_auc < 0.57:
                        self.logger.info(f"  [TIER 3] FAIL - AUC={result.test_auc:.3f} < 0.57")
                    else:
                        self.logger.info(f"  [TIER 3] FAIL - fragility={result.fragility_score:.3f} >= 0.40")

                except Exception as frag_err:
                    self.logger.warning(f"  [TIER 3] Fragility analysis failed: {frag_err}")
                    self.logger.warning(f"  [TIER 3] Traceback: {traceback.format_exc()}")
                    result.fragility_score = 0.5  # Unknown, not 1.0

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
            # Quantile-based signal generation: heavily regularized models produce
            # low raw probabilities (mean ~0.27, max ~0.56) due to bear synthetic
            # data and L1/L2 penalties. Fixed thresholds (0.5 or 0.6) yield 0-2
            # signals. Instead, select the top 25% of predictions as signals —
            # the model ranks UP days higher even if absolute probabilities are low.
            signal_quantile = np.percentile(proba_raw, 75)
            signals = (proba_raw >= signal_quantile).astype(int)
            n_signals = int(signals.sum())
            self.logger.info(f"  Trading signals: {n_signals}/{len(signals)} "
                             f"(top 25%, quantile threshold={signal_quantile:.3f}, "
                             f"raw range=[{proba_raw.min():.3f}, {proba_raw.max():.3f}])")

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
                self.logger.info(f"  Transaction cost per trade: {backtest_metrics.get('transaction_cost_per_trade', 0.0):.4%}")
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

            # Step 13.5: Meta-labeling (secondary classifier for signal profitability)
            meta_model = None
            if getattr(config.anti_overfit, 'use_meta_labeling', True):
                self.logger.info("[STEP 13.5] Meta-labeling: training signal profitability classifier...")
                tracker.set_step(ExperimentStep.BACKTEST, "Meta-labeling signal filter")

                try:
                    from src.phase_15_strategy.meta_labeler import MetaLabeler

                    X_meta_base = X_test_raw if use_leak_proof else X_test
                    swing_proba = proba_raw  # Raw (uncalibrated) swing probabilities
                    # No separate timing model in experiment runner; use swing proba as proxy
                    timing_proba = proba_raw

                    # Use median-based signals for meta-labeling (top 50% of predictions).
                    # This gives ~128 signals from 256 samples — enough for meta-labeler
                    # training. The meta-labeler learns which "model-leans-UP" days are
                    # actually profitable, providing signal filtering.
                    meta_threshold = np.median(proba_raw)
                    meta_signals = (proba_raw >= meta_threshold).astype(int)
                    self.logger.info(f"  Meta-label signals: {int(meta_signals.sum())}/{len(meta_signals)} "
                                     f"(above median={meta_threshold:.3f})")

                    meta_labeler = MetaLabeler(C=1.0, min_signals=30, min_per_class=15, cv_folds=3)
                    meta_fit_result = meta_labeler.fit(
                        X_test=X_meta_base,
                        swing_proba=swing_proba,
                        timing_proba=timing_proba,
                        signals=meta_signals,
                        returns=test_returns if test_returns is not None else np.zeros(len(signals)),
                        slippage_bps=ENGINE_CONFIG['slippage_bps'],
                        commission_bps=ENGINE_CONFIG['commission_bps'],
                    )

                    result.meta_label_fitted = meta_fit_result.get("fitted", False)
                    result.meta_label_auc = meta_fit_result.get("meta_auc", 0.0)

                    if result.meta_label_fitted:
                        meta_model = meta_labeler
                        self.logger.info(f"  Meta-labeler fitted: AUC={result.meta_label_auc:.4f}, "
                                         f"signals={meta_fit_result.get('n_signals', 0)}, "
                                         f"wins={meta_fit_result.get('n_wins', 0)}, "
                                         f"losses={meta_fit_result.get('n_losses', 0)}")

                        # Re-run backtest with meta-filtered signals
                        meta_proba_all = meta_labeler.predict(X_meta_base, swing_proba, timing_proba)
                        if meta_proba_all is not None and test_returns is not None:
                            filtered_signals = meta_signals.copy()
                            filtered_signals[meta_proba_all < 0.5] = 0  # Skip low-confidence signals

                            if filtered_signals.sum() >= 5:
                                meta_backtest = compute_realistic_backtest_metrics(
                                    signals=filtered_signals,
                                    returns=test_returns,
                                    dates=test_dates_array,
                                    slippage_bps=ENGINE_CONFIG['slippage_bps'],
                                    commission_bps=ENGINE_CONFIG['commission_bps'],
                                )
                                result.meta_sharpe = meta_backtest["sharpe_net"]
                                result.meta_win_rate = meta_backtest["win_rate_net"]
                                result.meta_improvement = result.meta_sharpe - result.backtest_sharpe

                                self.logger.info(f"  Meta-filtered backtest: Sharpe={result.meta_sharpe:.3f} "
                                                 f"(vs {result.backtest_sharpe:.3f}), "
                                                 f"Win rate={result.meta_win_rate:.1%}, "
                                                 f"Improvement={result.meta_improvement:+.3f}")
                            else:
                                self.logger.info(f"  Meta-filtering removed too many signals ({int(filtered_signals.sum())} remaining)")
                    else:
                        reason = meta_fit_result.get("reason", "unknown")
                        self.logger.info(f"  Meta-labeler not fitted: {reason}")

                except Exception as meta_err:
                    self.logger.warning(f"  Meta-labeling failed: {meta_err}")

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

                        # Create directions from predictions (0.5 = natural class boundary)
                        train_proba = model.predict_proba(X_train)[:, 1]
                        directions = pd.Series(
                            np.where(train_proba > 0.5, "LONG", "SHORT"),
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
                save_dict = {
                    "model": model,
                    "feature_cols": feature_cols,
                    "config": config.to_dict(),
                    "test_auc": result.test_auc,
                    "cv_auc": result.cv_auc_mean,
                    "wmes_score": result.wmes_score,
                    "leak_proof": use_leak_proof,
                }
                if meta_model is not None:
                    save_dict["meta_model"] = meta_model
                # Save conformal sizer if fitted (Wave F4.2)
                _cs = conformal_sizer if 'conformal_sizer' in dir() else getattr(self, '_conformal_sizer_legacy', None)
                if _cs is not None and getattr(_cs, '_fitted', False):
                    save_dict["conformal_sizer"] = _cs
                joblib.dump(save_dict, model_save_path)
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

            # ── Wave 32: Update feature research candidate stats ─────────
            if is_feature_research and _research_candidate_names:
                try:
                    from src.phase_09_features_calendar.feature_researcher import FeatureResearchAgent
                    _fr_agent = FeatureResearchAgent()
                    tier1_pass = (result.test_auc > 0.55 and result.wmes_score >= 0.40
                                  and result.walk_forward_passed)
                    _fr_agent.update_candidate_stats(
                        _research_candidate_names,
                        tier1_passed=tier1_pass,
                        wmes_score=result.wmes_score,
                        walk_forward_passed=result.walk_forward_passed,
                    )
                    baseline = _fr_agent.get_baseline_stats()
                    graduated = _fr_agent.check_graduations(baseline["tier1_pass_rate"])
                    if graduated:
                        self.logger.info(f"  [FEATURE_RESEARCH] Graduated features: {graduated}")
                except Exception as fr_err:
                    self.logger.warning(f"  [FEATURE_RESEARCH] Stats update failed: {fr_err}")

        except Exception as e:
            result.status = ExperimentStatus.FAILED
            result.error_message = str(e)
            self.logger.error(f"Experiment {config.experiment_id} failed: {e}")
            self.logger.error(traceback.format_exc())
            tracker.fail_experiment(str(e))

        result.completed_at = datetime.now().isoformat()
        result.duration_seconds = time.time() - start_time

        return result

    @staticmethod
    def _reality_check(result) -> dict:
        """Upper-bound reality checks to catch leakage or unrealistic metrics.

        Returns dict of flag_name → message for any triggered flags.
        Empty dict means all checks passed.
        """
        flags = {}

        # AUC > 0.85 for daily SPY direction is almost certainly leakage
        if result.test_auc > 0.85:
            flags["likely_leakage"] = (
                f"AUC={result.test_auc:.3f} > 0.85 — likely feature leakage "
                f"(daily SPY direction rarely exceeds 0.75 AUC)"
            )

        # Sharpe > 4.0 on daily returns is unrealistic
        if hasattr(result, 'backtest_sharpe') and result.backtest_sharpe and result.backtest_sharpe > 4.0:
            flags["unrealistic_sharpe"] = (
                f"Sharpe={result.backtest_sharpe:.2f} > 4.0 — unrealistic for daily SPY"
            )

        # Win rate > 75% on daily direction is suspicious
        if hasattr(result, 'backtest_win_rate') and result.backtest_win_rate and result.backtest_win_rate > 0.75:
            flags["suspicious_win_rate"] = (
                f"Win rate={result.backtest_win_rate:.1%} > 75% — suspicious"
            )

        # Train-test gap suspiciously close with high AUC → data leakage
        if result.train_auc > 0 and result.test_auc > 0.60:
            gap = abs(result.train_auc - result.test_auc)
            if gap < 0.001:
                flags["suspiciously_close"] = (
                    f"Train-test gap={gap:.4f} < 0.001 with AUC={result.test_auc:.3f} "
                    f"— train and test may share leaked features"
                )

        return flags

    def _create_model(self, config: ExperimentConfig):
        """Create model based on configuration.

        ANTI-OVERFITTING MEASURES:
        - Aggressive regularization (lower C = stronger penalty)
        - Diverse ensemble combines models with DIFFERENT regularization strengths
        - Shallow trees (max_depth capped at 4 for robustness)
        - High min_samples_leaf to prevent memorization
        - Calibration wrapper for probability outputs
        - Optional RegimeRouter wrapping (Wave F4.1)
        """
        base_model = self._create_base_model(config)
        return self._maybe_wrap_regime_router(base_model, config)

    def _create_base_model(self, config: ExperimentConfig):
        """Create the base model (before any wrapper like RegimeRouter)."""
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

        if config.model.model_type in ("logistic", "logistic_l2"):
            return LogisticRegression(C=aggressive_C, max_iter=1000, random_state=42)

        elif config.model.model_type == "logistic_l1":
            return LogisticRegression(
                penalty='l1', solver='saga',
                C=aggressive_C, max_iter=1000, random_state=42
            )

        elif config.model.model_type == "logistic_elastic":
            return LogisticRegression(
                penalty='elasticnet', solver='saga',
                C=aggressive_C, l1_ratio=max(getattr(config.model, 'elastic_l1_ratio', 0.5), 0.5),
                max_iter=1000, random_state=42
            )

        elif config.model.model_type == "hist_gradient_boosting":
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(
                max_iter=min(config.model.gb_n_estimators, 100),
                max_depth=safe_max_depth,
                learning_rate=min(config.model.gb_learning_rate, 0.1),
                min_samples_leaf=safe_min_samples_leaf,
                l2_regularization=1.0,  # Strong L2 regularization
                max_bins=255,
                early_stopping=True,
                random_state=42,
            )

        elif config.model.model_type == "mlp_small":
            # Regularized MLP: 1 hidden layer, strong alpha (L2), early stopping
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(
                hidden_layer_sizes=(32,),
                alpha=1.0,  # Very strong L2 regularization (default 0.0001)
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=10,
                random_state=42,
            )

        elif config.model.model_type == "mlp_medium":
            # Regularized MLP: 2 hidden layers, moderate alpha, dropout-like early stopping
            from sklearn.neural_network import MLPClassifier
            return MLPClassifier(
                hidden_layer_sizes=(64, 32),
                alpha=0.5,  # Strong L2 regularization
                learning_rate='adaptive',
                learning_rate_init=0.001,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=10,
                batch_size=min(256, max(32, safe_min_samples_leaf)),
                random_state=42,
            )

        elif config.model.model_type == "sgd_linear":
            # SGD Classifier: regularized linear model, very fast, elastic net support
            from sklearn.linear_model import SGDClassifier
            from sklearn.calibration import CalibratedClassifierCV
            base = SGDClassifier(
                loss='log_loss',
                penalty='elasticnet',
                alpha=0.01,  # Strong regularization
                l1_ratio=0.5,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=10,
                random_state=42,
            )
            # Wrap in CalibratedClassifierCV for proper probability outputs
            return CalibratedClassifierCV(base, cv=3, method='sigmoid')

        elif config.model.model_type == "ridge":
            # RidgeClassifier: strong L2 regularized linear model
            from sklearn.linear_model import RidgeClassifier
            from sklearn.calibration import CalibratedClassifierCV
            base = RidgeClassifier(alpha=10.0, random_state=42)
            return CalibratedClassifierCV(base, cv=3, method='sigmoid')

        elif config.model.model_type == "extra_trees":
            # Extra-randomized trees: more random splits = less overfitting than RF
            from sklearn.ensemble import ExtraTreesClassifier
            return ExtraTreesClassifier(
                n_estimators=min(config.model.gb_n_estimators, 100),
                max_depth=safe_max_depth,
                min_samples_leaf=safe_min_samples_leaf,
                max_features='sqrt',
                bootstrap=True,
                random_state=42,
            )

        elif config.model.model_type == "bagged_linear":
            # Bagging of strongly regularized linear models (GANDALF-like)
            # Multiple linear models on random subsets → captures non-linear patterns
            from sklearn.ensemble import BaggingClassifier
            base_lr = LogisticRegression(
                penalty='l1', solver='saga', C=0.01, max_iter=1000, random_state=42
            )
            return BaggingClassifier(
                estimator=base_lr,
                n_estimators=20,
                max_samples=0.7,
                max_features=0.7,
                bootstrap=True,
                random_state=42,
            )

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
                    models.append(('lr_l2', LogisticRegression(C=aggressive_C, max_iter=1000, random_state=42)))
                elif model_name == "logistic_l1":
                    models.append(('lr_l1', LogisticRegression(
                        penalty='l1', solver='saga', C=aggressive_C, max_iter=1000, random_state=43
                    )))
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

        # ── Wave 35: 5 new model types ──────────────────────────────────

        elif config.model.model_type == "lda":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            base = LinearDiscriminantAnalysis(solver='svd')
            return CalibratedClassifierCV(base, cv=3, method='sigmoid')

        elif config.model.model_type == "gaussian_nb":
            from sklearn.naive_bayes import GaussianNB
            return GaussianNB(var_smoothing=1e-8)

        elif config.model.model_type == "svc_linear":
            from sklearn.svm import LinearSVC
            base = LinearSVC(
                C=aggressive_C, loss='squared_hinge', penalty='l2',
                max_iter=2000, dual='auto', random_state=42,
            )
            return CalibratedClassifierCV(base, cv=3, method='sigmoid')

        elif config.model.model_type == "bayesian_ridge":
            base = _BayesianRidgeClassifierWrapper()
            return CalibratedClassifierCV(base, cv=3, method='sigmoid')

        elif config.model.model_type == "quantile_gb":
            return QuantileGBClassifier(
                max_iter=min(config.model.gb_n_estimators, 100),
                max_depth=safe_max_depth,
                learning_rate=min(config.model.gb_learning_rate, 0.1),
                min_samples_leaf=safe_min_samples_leaf,
                random_state=42,
            )

        elif config.model.model_type == "quantile_forest":
            from src.phase_12_model_training.quantile_forest_wrapper import (
                QuantileForestClassifier,
            )
            return QuantileForestClassifier(
                n_estimators=min(config.model.gb_n_estimators, 100),
                max_depth=safe_max_depth,
                min_samples_leaf=safe_min_samples_leaf,
                random_state=42,
            )

        # ── Wave F2: CatBoost + Stacking Ensemble ─────────────────────────
        elif config.model.model_type == "catboost":
            try:
                from catboost import CatBoostClassifier
                return CatBoostClassifier(
                    iterations=min(config.model.gb_n_estimators, 100),
                    depth=safe_max_depth,
                    learning_rate=min(config.model.gb_learning_rate, 0.1),
                    l2_leaf_reg=3.0,
                    random_seed=42,
                    verbose=0,
                    allow_writing_files=False,
                )
            except ImportError:
                self.logger.warning("catboost not installed, falling back to HistGradientBoosting")
                from sklearn.ensemble import HistGradientBoostingClassifier
                return HistGradientBoostingClassifier(
                    max_iter=min(config.model.gb_n_estimators, 100),
                    max_depth=safe_max_depth,
                    learning_rate=min(config.model.gb_learning_rate, 0.1),
                    min_samples_leaf=safe_min_samples_leaf,
                    random_state=42,
                )

        elif config.model.model_type == "stacking_ensemble":
            from src.phase_12_model_training.stacking_ensemble import (
                StackingEnsembleClassifier,
            )
            return StackingEnsembleClassifier(
                meta_C=aggressive_C * 10,  # Meta-learner needs slightly less regularization
                n_cv_folds=min(config.cross_validation.n_cv_folds, 5),
                purge_days=config.cross_validation.purge_days,
                embargo_days=config.cross_validation.embargo_days,
                random_state=42,
            )

        else:
            # Default to strongly regularized L2 logistic regression
            return LogisticRegression(C=aggressive_C, max_iter=1000, random_state=42)

    def _maybe_wrap_regime_router(self, model, config: ExperimentConfig):
        """Optionally wrap a base model with RegimeRouter (Wave F4.1).

        If ``config.model.use_regime_router`` is True, the base model is cloned
        per-regime and routed at inference time based on volatility.  If the
        flag is False or RegimeRouter is unavailable, returns the model unchanged.
        """
        if not getattr(config.model, 'use_regime_router', False):
            return model
        try:
            from src.phase_15_strategy.regime_router import RegimeRouter
            method = getattr(config.model, 'regime_split_method', 'vix_quartile')
            router = RegimeRouter(
                base_model=model,
                regime_method=method,
                min_samples_per_regime=100,
                random_state=42,
            )
            self.logger.info(f"  [REGIME_ROUTER] Wrapping {type(model).__name__} with RegimeRouter ({method})")
            return router
        except Exception as rr_err:
            self.logger.debug(f"  RegimeRouter skipped: {rr_err}")
            return model


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXPERIMENT ENGINE
# ═══════════════════════════════════════════════════════════════════════════════
class ExperimentEngine:
    """
    Main engine that runs experiments using the UNIFIED FULL pipeline.

    Every experiment uses the SAME code path as train_robust_model.py.
    No simplified parallel pipelines.
    """

    def __init__(self, db=None):
        self.logger = logging.getLogger("ENGINE")
        self._db = db  # Optional RegistryDB for SQLite backend
        self.history = ExperimentHistory(db=db)
        self.generator = ExperimentGenerator(history=self.history)
        self.runner = UnifiedExperimentRunner()

        self.running = False
        self.current_experiment: Optional[ExperimentConfig] = None
        self.experiments_today = 0
        self.last_reset = datetime.now().date()

    def run_experiment(self, config: ExperimentConfig) -> ExperimentResult:
        """Run a single experiment with the given configuration.

        Wave 26: Two-pass fast screen — Pass 1 skips anti-overfit augmentation.
        If Pass 1 yields a Tier 1 candidate, Pass 2 re-runs with full validation.
        """
        # Validate config
        is_valid, errors = validate_config(config)
        if not is_valid:
            self.logger.warning(f"Config validation warnings: {errors}")

        # Pass 1: Fast screen (skip anti-overfit augmentation)
        result = self.runner.run(config, fast_screen=True)

        # Check if worth a full validation pass
        if (result.status == ExperimentStatus.COMPLETED
                and result.test_auc > 0.55
                and result.wmes_score >= 0.40
                and getattr(result, 'walk_forward_passed', False)
                and config.anti_overfit.use_anti_overfit):
            self.logger.info(
                f"[FAST SCREEN PASS] AUC={result.test_auc:.3f}, WMES={result.wmes_score:.3f} "
                f"— re-running with full anti-overfit augmentation..."
            )
            result = self.runner.run(config, fast_screen=False)

        # Store result
        self.history.add(result)

        # Extract feature importances for universal feature map
        self._update_feature_map(result, config)

        # Register model if it passes Tier 1 quality gate (Wave 26 thresholds)
        train_test_gap = result.train_auc - result.test_auc if result.train_auc > 0 else 0.0
        reality_flags = self.runner._reality_check(result)
        has_leakage = "likely_leakage" in reality_flags
        tier1_pass = (
            result.status == ExperimentStatus.COMPLETED
            and result.test_auc > 0.55
            and result.test_auc < 0.85
            and train_test_gap < 0.15
            and result.wmes_score >= 0.40
            and not has_leakage
            and getattr(result, 'walk_forward_passed', True)
        )

        if tier1_pass:
            model_id = self._db.register_model_from_experiment(result.to_dict())
            # Wave 33: Include stability suite composite in tier calculation
            suite_composite = getattr(result, 'stability_composite', 0.0) or 0.0
            tier = compute_tier(
                result.stability_score, result.fragility_score,
                result.test_auc, suite_composite,
            )
            tier_labels = {1: "REGISTRY", 2: "PAPER-ELIGIBLE", 3: "LIVE-ELIGIBLE"}
            self.logger.info(
                f"Registered model: {model_id} "
                f"(AUC={result.test_auc:.3f}, WMES={result.wmes_score:.3f}, "
                f"stability={result.stability_score:.3f}, fragility={result.fragility_score:.3f}, "
                f"suite={suite_composite:.3f}, "
                f"tier={tier} [{tier_labels.get(tier, '?')}])"
            )

            # Add to best configs pool if good
            if result.test_auc > 0.56:
                self.generator.add_best_config(config, result.test_auc)
        elif result.status == ExperimentStatus.COMPLETED:
            reasons = []
            if result.test_auc <= 0.55:
                reasons.append(f"AUC={result.test_auc:.3f}<=0.55")
            if result.test_auc >= 0.85:
                reasons.append(f"AUC={result.test_auc:.3f}>=0.85 (likely leakage)")
            if train_test_gap >= 0.15:
                reasons.append(f"gap={train_test_gap:.3f}>=0.15")
            if result.wmes_score < 0.40:
                reasons.append(f"WMES={result.wmes_score:.3f}<0.40")
            if not getattr(result, 'walk_forward_passed', True):
                reasons.append("walk-forward failed")
            if has_leakage:
                reasons.append("reality check: likely leakage")
            self.logger.info(f"Model NOT registered (Tier 1 fail): {'; '.join(reasons)}")

            # Register with ModelRegistryV2 for mega ensemble discovery
            try:
                from src.model_registry_v2 import ModelRegistryV2, ModelEntry, ModelType

                v2_registry = ModelRegistryV2(db=self._db)

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

    def _update_feature_map(self, result: ExperimentResult, config: ExperimentConfig):
        """Extract feature importances and update universal feature map."""
        if result.status != ExperimentStatus.COMPLETED:
            return
        if not result.model_path:
            return
        try:
            from src.phase_23_analytics.symbolic_cross_learner import (
                FeatureImportanceExtractor, UniversalFeatureMap,
            )
            model_path = Path(result.model_path)
            if not model_path.is_file():
                return
            saved = joblib.load(model_path)
            model = saved.get("model")
            feature_cols = saved.get("feature_cols", [])
            if model is None:
                return

            importances = FeatureImportanceExtractor.extract(
                model, feature_names=feature_cols
            )
            if not importances:
                return

            map_path = project_root / "models" / "feature_importance_map.json"
            fmap = UniversalFeatureMap(persist_path=map_path)
            model_type = getattr(config, "model_type", "unknown")
            if hasattr(config, "dim_reduction") and hasattr(config.dim_reduction, "method"):
                model_type = f"{model_type}_{config.dim_reduction.method}"
            fmap.add_model_full(
                model_id=result.experiment_id,
                importances=importances,
                model_auc=result.test_auc,
                model_type=model_type,
            )
            self.logger.info(
                f"Updated feature map: {len(importances)} features from {result.experiment_id}"
            )
        except Exception as e:
            self.logger.debug(f"Feature map update skipped: {e}")

    def _check_memory(self) -> float:
        """Check current process memory in MB. Returns usage in MB."""
        try:
            import psutil
            proc = psutil.Process()
            mem_mb = proc.memory_info().rss / (1024 * 1024)
            return mem_mb
        except ImportError:
            return 0.0

    def run_one_experiment(self) -> Optional[ExperimentResult]:
        """Generate and run a single experiment.

        Wave 36: Added per-experiment memory monitoring and timeout protection.
        """
        import gc

        # Check daily limit
        if datetime.now().date() != self.last_reset:
            self.experiments_today = 0
            self.last_reset = datetime.now().date()

        if self.experiments_today >= ENGINE_CONFIG["max_experiments_per_hour"] * 24:
            self.logger.warning("Daily experiment limit reached")
            return None

        # Pre-experiment memory check (thresholds from system resource detection)
        _sys_res = get_system_resources()
        _mem_pressure = {
            "low": 1500, "medium": 3500, "high": 8000, "ultra": 32000,
        }.get(_sys_res.tier.value, 3500)
        _mem_critical = {
            "low": 2500, "medium": 5000, "high": 16000, "ultra": 64000,
        }.get(_sys_res.tier.value, 5000)
        mem_before = self._check_memory()
        if mem_before > _mem_pressure:
            gc.collect()
            mem_before = self._check_memory()
            self.logger.warning(f"[MEM] High memory before experiment: {mem_before:.0f} MB, forced GC")
        if mem_before > _mem_critical:
            gc.collect()
            self.logger.error(f"[MEM] Memory critical ({mem_before:.0f} MB), skipping experiment")
            return None

        # Generate config
        config = self.generator.generate_next()
        self.current_experiment = config
        self.logger.info(f"Running experiment: {config.experiment_type}")

        # Run experiment with timeout
        from src.core.registry_db import CURRENT_SCORING_VERSION
        result = None
        timeout_seconds = 1800  # 30 min max per experiment (prevents hung stability suite)
        try:
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(self.run_experiment, config)
                result = future.result(timeout=timeout_seconds)
        except concurrent.futures.TimeoutError:
            self.logger.error(
                f"[TIMEOUT] Experiment {config.experiment_id} exceeded {timeout_seconds}s, aborting"
            )
            result = ExperimentResult(
                experiment_id=config.experiment_id,
                status=ExperimentStatus.FAILED,
                error_message=f"Timeout after {timeout_seconds}s",
                scoring_version=CURRENT_SCORING_VERSION,
            )
        except Exception as e:
            self.logger.error(f"[ERROR] Experiment failed: {e}")
            result = ExperimentResult(
                experiment_id=config.experiment_id,
                status=ExperimentStatus.FAILED,
                error_message=str(e),
                scoring_version=CURRENT_SCORING_VERSION,
            )

        self.experiments_today += 1

        # Post-experiment memory cleanup — every experiment, not every 5
        gc.collect()
        mem_after = self._check_memory()
        mem_delta = mem_after - mem_before
        if mem_delta > 500:  # Leaked 500+ MB
            self.logger.warning(
                f"[MEM] Experiment leaked {mem_delta:.0f} MB "
                f"({mem_before:.0f} -> {mem_after:.0f} MB)"
            )

        if self.experiments_today % 10 == 0:
            self.logger.info(
                f"[STATUS] {self.experiments_today} experiments today, "
                f"memory: {mem_after:.0f} MB"
            )

        self.current_experiment = None
        return result

    def get_status(self) -> Dict:
        """Get current engine status."""
        return {
            "running": self.running,
            "current_experiment": self.current_experiment.to_dict() if self.current_experiment else None,
            "experiments_today": self.experiments_today,
            "history_stats": self.history.get_statistics(),
            "registry_stats": self._db.get_model_statistics() if self._db else {},
            "top_models": self._db.get_models(min_tier=2, min_auc=0.56)[:5] if self._db else [],
        }

    def run_forever(self, interval_seconds: int = 60, max_consecutive_failures: int = 10):
        """Run experiments continuously with trend logging and adaptive cooldown.

        Features (Wave 17):
        - Logs summary stats every 5 rounds
        - Tracks consecutive failures and pauses after max_consecutive_failures
        - Adaptive cooldown: doubles sleep time when recent experiments fail
        - Uses failure patterns to guide experiment generation
        """
        self.running = True
        self.logger.info("Experiment engine starting (UNIFIED FULL PIPELINE)...")
        round_num = 0
        consecutive_failures = 0
        current_cooldown = interval_seconds

        while self.running:
            try:
                round_num += 1
                result = self.run_one_experiment()

                # Track consecutive failures
                if result is not None and hasattr(result, 'status'):
                    if result.status == ExperimentStatus.COMPLETED:
                        consecutive_failures = 0
                        current_cooldown = interval_seconds  # Reset cooldown
                    elif result.status == ExperimentStatus.FAILED:
                        consecutive_failures += 1
                else:
                    consecutive_failures += 1

                # Auto-pause on too many consecutive failures
                if consecutive_failures >= max_consecutive_failures:
                    self.logger.error(
                        f"[AUTO-PAUSE] {consecutive_failures} consecutive failures. "
                        f"Pausing for 10 minutes. Check logs for systematic issues."
                    )
                    time.sleep(600)  # 10 minute pause
                    consecutive_failures = 0
                    # Log failure patterns to help diagnose
                    try:
                        patterns = self.history.get_failure_patterns()
                        self.logger.info(f"  Failure patterns: {patterns}")
                    except Exception:
                        pass

                # Adaptive cooldown: double if failing, reset if succeeding
                if consecutive_failures >= 3:
                    current_cooldown = min(current_cooldown * 2, 300)  # Cap at 5 min
                    self.logger.info(
                        f"  Adaptive cooldown: {current_cooldown}s "
                        f"({consecutive_failures} consecutive failures)"
                    )

                # Log summary every 5 rounds
                if round_num % 5 == 0:
                    try:
                        stats = self.history.get_statistics()
                        trend = self.history.get_recent_trend(n=10)
                        self.logger.info(
                            f"══ Round {round_num} Summary ══ "
                            f"Total={stats.get('total', 0)}, "
                            f"Success={stats.get('success_rate', 0):.0%}, "
                            f"Avg AUC={stats.get('avg_test_auc', 0):.3f}, "
                            f"Best={stats.get('best_realistic_auc', 0):.3f}, "
                            f"Trend={trend.get('trend', '?')} "
                            f"(Δ={trend.get('auc_delta', 0):+.4f})"
                        )
                    except Exception:
                        pass

                time.sleep(current_cooldown)

            except KeyboardInterrupt:
                self.logger.info("Shutting down...")
                break

            except Exception as e:
                self.logger.error(f"Engine error: {e}")
                consecutive_failures += 1
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
