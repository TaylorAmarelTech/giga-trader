"""
GIGA TRADER - Dynamic Model Selector
=====================================
Intelligent model selection and ensembling based on registry performance.

Instead of loading a single hardcoded model, this module:
1. Queries the model registry for trained models
2. Filters by entry/exit window parameters
3. Selects or ensembles the best models for current conditions
4. Provides dynamic predictions with confidence aggregation

Usage:
    from src.dynamic_model_selector import DynamicModelSelector

    selector = DynamicModelSelector()
    selector.load_from_registry()

    prediction = selector.predict(features, entry_window=(30, 120), exit_window=(300, 385))
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import logging
import json

import numpy as np
import pandas as pd
import joblib

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger("GigaTrader.DynamicModelSelector")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ModelCandidate:
    """A candidate model from the registry."""
    model_id: str
    model_path: str
    config: Dict

    # Performance metrics
    cv_auc: float = 0.0
    test_auc: float = 0.0
    backtest_sharpe: float = 0.0
    backtest_win_rate: float = 0.0
    wmes_score: float = 0.0

    # Entry/Exit configuration
    entry_window_start: int = 0
    entry_window_end: int = 120
    exit_window_start: int = 300
    exit_window_end: int = 385

    # Model object (loaded on demand)
    model: Any = None
    feature_cols: List[str] = field(default_factory=list)

    def score(self, weights: Dict[str, float] = None) -> float:
        """Calculate weighted score for ranking."""
        weights = weights or {
            "cv_auc": 0.2,
            "test_auc": 0.25,
            "backtest_sharpe": 0.3,
            "wmes_score": 0.25,
        }
        return sum(
            weights.get(k, 0) * getattr(self, k, 0)
            for k in weights
        )

    def matches_window(
        self,
        entry_window: Tuple[int, int] = None,
        exit_window: Tuple[int, int] = None,
        tolerance: int = 30,
    ) -> bool:
        """Check if model's windows match requested windows within tolerance."""
        if entry_window:
            if abs(self.entry_window_start - entry_window[0]) > tolerance:
                return False
            if abs(self.entry_window_end - entry_window[1]) > tolerance:
                return False
        if exit_window:
            if abs(self.exit_window_start - exit_window[0]) > tolerance:
                return False
            if abs(self.exit_window_end - exit_window[1]) > tolerance:
                return False
        return True


@dataclass
class EnsemblePrediction:
    """Prediction from an ensemble of models."""
    swing_probability: float
    timing_probability: float
    confidence: float
    direction: str  # "LONG", "SHORT", "HOLD"

    # Individual model predictions
    model_predictions: List[Dict] = field(default_factory=list)

    # Ensemble metadata
    n_models: int = 0
    agreement_ratio: float = 0.0  # Fraction of models agreeing on direction
    ensemble_method: str = "weighted_average"

    # Entry/Exit windows from best model
    entry_window: Tuple[int, int] = (30, 120)
    exit_window: Tuple[int, int] = (300, 385)

    # Position sizing suggestions
    suggested_position_pct: float = 0.10
    confidence_adjusted_position_pct: float = 0.10


# =============================================================================
# ENSEMBLE STRATEGIES
# =============================================================================

class EnsembleStrategy:
    """Base class for ensemble strategies."""

    def combine(
        self,
        predictions: List[Dict],
        weights: List[float],
    ) -> Tuple[float, float]:
        """Combine predictions into final probabilities."""
        raise NotImplementedError


class WeightedAverageEnsemble(EnsembleStrategy):
    """Weighted average of model predictions."""

    def combine(
        self,
        predictions: List[Dict],
        weights: List[float],
    ) -> Tuple[float, float]:
        if not predictions:
            return 0.5, 0.5

        weights = np.array(weights)
        weights = weights / weights.sum()

        swing_probas = np.array([p["swing_proba"] for p in predictions])
        timing_probas = np.array([p["timing_proba"] for p in predictions])

        swing_combined = np.sum(swing_probas * weights)
        timing_combined = np.sum(timing_probas * weights)

        return float(swing_combined), float(timing_combined)


class MedianEnsemble(EnsembleStrategy):
    """Median of model predictions (robust to outliers)."""

    def combine(
        self,
        predictions: List[Dict],
        weights: List[float],
    ) -> Tuple[float, float]:
        if not predictions:
            return 0.5, 0.5

        swing_probas = np.array([p["swing_proba"] for p in predictions])
        timing_probas = np.array([p["timing_proba"] for p in predictions])

        return float(np.median(swing_probas)), float(np.median(timing_probas))


class VotingEnsemble(EnsembleStrategy):
    """Voting ensemble with weighted votes."""

    def __init__(self, threshold: float = 0.55):
        self.threshold = threshold

    def combine(
        self,
        predictions: List[Dict],
        weights: List[float],
    ) -> Tuple[float, float]:
        if not predictions:
            return 0.5, 0.5

        weights = np.array(weights)
        weights = weights / weights.sum()

        # Convert to votes
        swing_votes_up = np.array([
            1 if p["swing_proba"] > self.threshold else 0
            for p in predictions
        ])
        timing_votes_up = np.array([
            1 if p["timing_proba"] > self.threshold else 0
            for p in predictions
        ])

        # Weighted votes
        swing_vote_pct = np.sum(swing_votes_up * weights)
        timing_vote_pct = np.sum(timing_votes_up * weights)

        # Convert back to probability-like score
        swing_combined = 0.5 + (swing_vote_pct - 0.5) * 0.4  # Scale to 0.3-0.7 range
        timing_combined = 0.5 + (timing_vote_pct - 0.5) * 0.4

        return float(swing_combined), float(timing_combined)


class StackingEnsemble(EnsembleStrategy):
    """Stacking ensemble using meta-learner (if available)."""

    def __init__(self, meta_model=None):
        self.meta_model = meta_model

    def combine(
        self,
        predictions: List[Dict],
        weights: List[float],
    ) -> Tuple[float, float]:
        if not predictions:
            return 0.5, 0.5

        if self.meta_model is None:
            # Fall back to weighted average
            return WeightedAverageEnsemble().combine(predictions, weights)

        # Stack predictions as features for meta-model
        meta_features = np.array([
            [p["swing_proba"], p["timing_proba"]] for p in predictions
        ]).flatten()

        # Meta-model predicts final probabilities
        meta_pred = self.meta_model.predict_proba(meta_features.reshape(1, -1))[0]

        return float(meta_pred[1]), float(meta_pred[1])  # Assuming binary classification


# =============================================================================
# DYNAMIC MODEL SELECTOR
# =============================================================================

class DynamicModelSelector:
    """
    Dynamically selects and ensembles models based on registry performance.

    Key features:
    - Loads models from registry based on performance metrics
    - Filters models by entry/exit window parameters
    - Supports multiple ensemble strategies
    - Provides confidence-weighted predictions
    - Automatically selects best strategy based on conditions
    """

    def __init__(
        self,
        registry_path: Path = None,
        models_dir: Path = None,
        min_test_auc: float = 0.55,
        min_wmes: float = 0.45,
        max_models_to_load: int = 10,
        ensemble_method: str = "weighted_average",
    ):
        self.registry_path = registry_path or (project_root / "experiments" / "model_registry.json")
        self.models_dir = models_dir or (project_root / "models")
        self.min_test_auc = min_test_auc
        self.min_wmes = min_wmes
        self.max_models_to_load = max_models_to_load
        self.ensemble_method = ensemble_method

        self.candidates: List[ModelCandidate] = []
        self.loaded_models: Dict[str, ModelCandidate] = {}

        # Ensemble strategies
        self.ensemble_strategies = {
            "weighted_average": WeightedAverageEnsemble(),
            "median": MedianEnsemble(),
            "voting": VotingEnsemble(threshold=0.55),
            "stacking": StackingEnsemble(),
        }

        # Default thresholds
        self.swing_threshold = 0.55
        self.timing_threshold = 0.50

    def load_from_registry(self) -> int:
        """Load model candidates from registry."""
        if not self.registry_path.exists():
            logger.warning(f"Registry not found: {self.registry_path}")
            return 0

        try:
            with open(self.registry_path) as f:
                registry_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load registry: {e}")
            return 0

        self.candidates = []

        for model_id, record in registry_data.items():
            # Extract entry/exit windows from config
            config = record.get("config", {})
            entry_exit = config.get("entry_exit", config.get("trading", {}))

            candidate = ModelCandidate(
                model_id=model_id,
                model_path=record.get("model_path", ""),
                config=config,
                cv_auc=record.get("cv_auc", 0),
                test_auc=record.get("test_auc", 0),
                backtest_sharpe=record.get("backtest_sharpe", 0),
                backtest_win_rate=record.get("backtest_win_rate", 0),
                wmes_score=record.get("wmes_score", 0),
                entry_window_start=entry_exit.get("entry_window_start", 0),
                entry_window_end=entry_exit.get("entry_window_end", 120),
                exit_window_start=entry_exit.get("exit_window_start", 300),
                exit_window_end=entry_exit.get("exit_window_end", 385),
            )

            # Filter by minimum requirements
            if candidate.test_auc >= self.min_test_auc:
                if candidate.wmes_score >= self.min_wmes or candidate.wmes_score == 0:
                    self.candidates.append(candidate)

        # Sort by score
        self.candidates.sort(key=lambda c: c.score(), reverse=True)

        logger.info(f"Loaded {len(self.candidates)} model candidates from registry")
        return len(self.candidates)

    def load_model(self, candidate: ModelCandidate) -> bool:
        """Load a model from disk."""
        if candidate.model_id in self.loaded_models:
            return True

        model_path = Path(candidate.model_path)
        if not model_path.exists():
            # Try alternative paths
            alt_paths = [
                self.models_dir / "production" / f"{candidate.model_id}.joblib",
                self.models_dir / f"{candidate.model_id}.joblib",
                self.models_dir / "production" / "spy_leak_proof_models.joblib",
            ]
            for alt_path in alt_paths:
                if alt_path.exists():
                    model_path = alt_path
                    break

        if not model_path.exists():
            logger.warning(f"Model file not found: {candidate.model_path}")
            return False

        try:
            data = joblib.load(model_path)

            # Handle different model storage formats
            if "swing_pipeline" in data:
                candidate.model = {
                    "swing": data["swing_pipeline"],
                    "timing": data.get("timing_pipeline"),
                }
                candidate.feature_cols = data.get("feature_columns", [])
            elif "swing_model" in data:
                candidate.model = {
                    "swing": data["swing_model"],
                    "timing": data.get("timing_model"),
                }
                candidate.feature_cols = data.get("feature_cols", [])
            else:
                candidate.model = data
                candidate.feature_cols = []

            self.loaded_models[candidate.model_id] = candidate
            logger.info(f"Loaded model: {candidate.model_id}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model {candidate.model_id}: {e}")
            return False

    def get_best_models(
        self,
        n: int = 5,
        entry_window: Tuple[int, int] = None,
        exit_window: Tuple[int, int] = None,
    ) -> List[ModelCandidate]:
        """Get top N models matching the specified windows."""
        matching = [
            c for c in self.candidates
            if c.matches_window(entry_window, exit_window)
        ]

        if not matching:
            # Fall back to all candidates if no exact match
            matching = self.candidates

        return matching[:n]

    def predict(
        self,
        features: np.ndarray,
        feature_names: List[str] = None,
        entry_window: Tuple[int, int] = None,
        exit_window: Tuple[int, int] = None,
        n_models: int = 5,
    ) -> EnsemblePrediction:
        """
        Make prediction using ensemble of best models.

        Args:
            features: Feature array (1D or 2D)
            feature_names: Names of features
            entry_window: Desired entry window (start, end) in minutes
            exit_window: Desired exit window (start, end) in minutes
            n_models: Number of models to use in ensemble

        Returns:
            EnsemblePrediction with combined probabilities
        """
        # Ensure 2D features
        if features.ndim == 1:
            features = features.reshape(1, -1)

        # Get best models for this window configuration
        best_candidates = self.get_best_models(
            n=min(n_models, self.max_models_to_load),
            entry_window=entry_window,
            exit_window=exit_window,
        )

        if not best_candidates:
            logger.warning("No suitable models found")
            return EnsemblePrediction(
                swing_probability=0.5,
                timing_probability=0.5,
                confidence=0.0,
                direction="HOLD",
                entry_window=entry_window or (30, 120),
                exit_window=exit_window or (300, 385),
            )

        # Load models and make predictions
        predictions = []
        weights = []

        for candidate in best_candidates:
            if not self.load_model(candidate):
                continue

            try:
                pred = self._predict_single(candidate, features, feature_names)
                if pred is not None:
                    predictions.append(pred)
                    weights.append(candidate.score())
            except Exception as e:
                logger.warning(f"Prediction failed for {candidate.model_id}: {e}")

        if not predictions:
            logger.warning("All model predictions failed")
            return EnsemblePrediction(
                swing_probability=0.5,
                timing_probability=0.5,
                confidence=0.0,
                direction="HOLD",
                entry_window=entry_window or (30, 120),
                exit_window=exit_window or (300, 385),
            )

        # Combine predictions
        strategy = self.ensemble_strategies.get(
            self.ensemble_method,
            self.ensemble_strategies["weighted_average"]
        )
        swing_proba, timing_proba = strategy.combine(predictions, weights)

        # Calculate confidence and agreement
        swing_probas = [p["swing_proba"] for p in predictions]
        confidence = 1 - np.std(swing_probas) * 2  # Lower std = higher confidence
        confidence = max(0.0, min(1.0, confidence))

        # Determine direction
        direction = "HOLD"
        if swing_proba > self.swing_threshold:
            if timing_proba > self.timing_threshold:
                direction = "LONG"
        elif swing_proba < (1 - self.swing_threshold):
            if timing_proba < (1 - self.timing_threshold):
                direction = "SHORT"

        # Calculate agreement
        n_bullish = sum(1 for p in swing_probas if p > self.swing_threshold)
        n_bearish = sum(1 for p in swing_probas if p < (1 - self.swing_threshold))
        agreement = max(n_bullish, n_bearish) / len(predictions)

        # Position sizing based on confidence
        base_position = 0.10
        if direction != "HOLD":
            adjusted_position = base_position * (0.5 + confidence * 0.5) * agreement
        else:
            adjusted_position = 0.0

        # Use windows from best candidate
        best = best_candidates[0]

        return EnsemblePrediction(
            swing_probability=swing_proba,
            timing_probability=timing_proba,
            confidence=confidence,
            direction=direction,
            model_predictions=predictions,
            n_models=len(predictions),
            agreement_ratio=agreement,
            ensemble_method=self.ensemble_method,
            entry_window=(best.entry_window_start, best.entry_window_end),
            exit_window=(best.exit_window_start, best.exit_window_end),
            suggested_position_pct=base_position,
            confidence_adjusted_position_pct=adjusted_position,
        )

    def _predict_single(
        self,
        candidate: ModelCandidate,
        features: np.ndarray,
        feature_names: List[str] = None,
    ) -> Optional[Dict]:
        """Make prediction with a single model."""
        if candidate.model is None:
            return None

        model = candidate.model

        # Handle dict format with swing/timing models
        if isinstance(model, dict):
            swing_model = model.get("swing")
            timing_model = model.get("timing")
        else:
            swing_model = model
            timing_model = None

        # Reorder features if needed
        if feature_names and candidate.feature_cols:
            features = self._reorder_features(features, feature_names, candidate.feature_cols)

        # Predict
        # CRITICAL: Both swing AND timing models are REQUIRED
        # No fallback to 0.5 - this would bypass proper signal validation
        try:
            # Swing model prediction - REQUIRED
            if swing_model is None:
                logger.warning(f"Model {candidate.model_id}: No swing model available")
                return None

            if not hasattr(swing_model, "predict_proba"):
                logger.warning(f"Model {candidate.model_id}: Swing model lacks predict_proba (invalid model)")
                return None

            swing_proba = swing_model.predict_proba(features)[0, 1]

            # Timing model prediction - REQUIRED
            if timing_model is None:
                logger.warning(f"Model {candidate.model_id}: No timing model available - signal rejected")
                return None

            if not hasattr(timing_model, "predict_proba"):
                logger.warning(f"Model {candidate.model_id}: Timing model lacks predict_proba (invalid model)")
                return None

            timing_proba = timing_model.predict_proba(features)[0, 1]

            return {
                "model_id": candidate.model_id,
                "swing_proba": float(swing_proba),
                "timing_proba": float(timing_proba),
            }

        except Exception as e:
            logger.warning(f"Prediction error for model {candidate.model_id}: {e}")
            return None

    def _reorder_features(
        self,
        features: np.ndarray,
        input_names: List[str],
        expected_names: List[str],
    ) -> np.ndarray:
        """Reorder features to match expected column order."""
        if not expected_names:
            return features

        # Create mapping
        input_to_idx = {name: i for i, name in enumerate(input_names)}

        # Reorder
        reordered = np.zeros((features.shape[0], len(expected_names)))
        for i, name in enumerate(expected_names):
            if name in input_to_idx:
                reordered[:, i] = features[:, input_to_idx[name]]
            else:
                # Missing feature - use 0
                reordered[:, i] = 0

        return reordered

    def get_available_windows(self) -> List[Dict]:
        """Get list of available entry/exit window configurations."""
        windows = []
        seen = set()

        for c in self.candidates:
            key = (c.entry_window_start, c.entry_window_end,
                   c.exit_window_start, c.exit_window_end)
            if key not in seen:
                seen.add(key)
                windows.append({
                    "entry_window": (c.entry_window_start, c.entry_window_end),
                    "exit_window": (c.exit_window_start, c.exit_window_end),
                    "n_models": sum(1 for x in self.candidates
                                   if x.entry_window_start == c.entry_window_start
                                   and x.entry_window_end == c.entry_window_end),
                    "best_score": c.score(),
                })

        return sorted(windows, key=lambda w: w["best_score"], reverse=True)

    def get_status(self) -> Dict:
        """Get selector status."""
        return {
            "n_candidates": len(self.candidates),
            "n_loaded": len(self.loaded_models),
            "min_test_auc": self.min_test_auc,
            "min_wmes": self.min_wmes,
            "ensemble_method": self.ensemble_method,
            "top_models": [
                {
                    "model_id": c.model_id,
                    "score": c.score(),
                    "test_auc": c.test_auc,
                    "wmes": c.wmes_score,
                    "entry_window": (c.entry_window_start, c.entry_window_end),
                    "exit_window": (c.exit_window_start, c.exit_window_end),
                }
                for c in self.candidates[:5]
            ],
            "available_windows": self.get_available_windows()[:5],
        }


# =============================================================================
# GRID SEARCH RUNNER
# =============================================================================

class EntryExitGridSearchRunner:
    """
    Runs systematic grid search across entry/exit window combinations.

    Integrates with ExperimentEngine and ModelRegistry for full tracking.
    """

    # All entry/exit window combinations to test
    ENTRY_WINDOWS = [
        (0, 30),     # First 30 min
        (0, 60),     # First hour
        (15, 60),    # 15 min to 1 hour
        (30, 90),    # 30 min to 1.5 hours
        (30, 120),   # 30 min to 2 hours (default)
        (60, 120),   # 1 to 2 hours
        (60, 180),   # 1 to 3 hours
    ]

    EXIT_WINDOWS = [
        (240, 360),   # 4-6 hours from open
        (240, 385),   # 4 hours to close
        (300, 360),   # 5-6 hours
        (300, 385),   # 5 hours to close (default)
        (330, 385),   # Last hour
        (180, 300),   # 3-5 hours (early exit)
        (180, 385),   # 3 hours to close
    ]

    def __init__(self, experiment_engine=None):
        from src.experiment_engine import ExperimentEngine
        self.engine = experiment_engine or ExperimentEngine()
        self.results = []

    def generate_all_configs(self):
        """Generate configs for all entry/exit window combinations."""
        from src.experiment_config import create_default_config

        configs = []

        for entry_window in self.ENTRY_WINDOWS:
            for exit_window in self.EXIT_WINDOWS:
                # Skip invalid combinations (exit before entry ends)
                if exit_window[0] < entry_window[1]:
                    continue

                config = create_default_config(
                    f"grid_e{entry_window[0]}_{entry_window[1]}_x{exit_window[0]}_{exit_window[1]}"
                )

                # Set entry/exit windows
                if hasattr(config, 'trading'):
                    config.trading.entry_window_start = entry_window[0]
                    config.trading.entry_window_end = entry_window[1]
                    config.trading.exit_window_start = exit_window[0]
                    config.trading.exit_window_end = exit_window[1]

                # Also store in config for registry
                config.entry_exit = {
                    "entry_window_start": entry_window[0],
                    "entry_window_end": entry_window[1],
                    "exit_window_start": exit_window[0],
                    "exit_window_end": exit_window[1],
                }

                config.experiment_type = "entry_exit_grid"
                config.description = (
                    f"Grid search: entry {entry_window[0]}-{entry_window[1]}min, "
                    f"exit {exit_window[0]}-{exit_window[1]}min"
                )

                configs.append(config)

        logger.info(f"Generated {len(configs)} entry/exit window combinations")
        return configs

    def run_grid_search(self, max_configs: int = None) -> List[Dict]:
        """
        Run grid search across all configurations.

        Args:
            max_configs: Maximum number of configs to run (None = all)

        Returns:
            List of results
        """
        configs = self.generate_all_configs()

        if max_configs:
            configs = configs[:max_configs]

        logger.info(f"Starting grid search with {len(configs)} configurations")

        for i, config in enumerate(configs):
            logger.info(f"Running config {i+1}/{len(configs)}: {config.experiment_name}")

            try:
                result = self.engine.run_experiment(config)

                self.results.append({
                    "config_id": config.experiment_id,
                    "entry_window": (
                        config.entry_exit.get("entry_window_start", 30),
                        config.entry_exit.get("entry_window_end", 120),
                    ),
                    "exit_window": (
                        config.entry_exit.get("exit_window_start", 300),
                        config.entry_exit.get("exit_window_end", 385),
                    ),
                    "test_auc": result.test_auc,
                    "cv_auc": result.cv_auc_mean,
                    "backtest_sharpe": result.backtest_sharpe,
                    "wmes_score": result.wmes_score,
                    "status": result.status.value,
                })

            except Exception as e:
                logger.error(f"Config {config.experiment_name} failed: {e}")
                self.results.append({
                    "config_id": config.experiment_id,
                    "error": str(e),
                    "status": "failed",
                })

        # Save results
        results_path = project_root / "experiments" / "grid_search_results.json"
        with open(results_path, "w") as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Grid search complete. Results saved to {results_path}")
        return self.results

    def get_best_configs(self, n: int = 5) -> List[Dict]:
        """Get top N configurations by test AUC."""
        successful = [r for r in self.results if r.get("status") == "completed"]
        sorted_results = sorted(
            successful,
            key=lambda x: x.get("test_auc", 0),
            reverse=True
        )
        return sorted_results[:n]


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    print("=" * 70)
    print("GIGA TRADER - Dynamic Model Selector")
    print("=" * 70)

    # Initialize selector
    selector = DynamicModelSelector()
    n_loaded = selector.load_from_registry()

    print(f"\nLoaded {n_loaded} model candidates from registry")

    if n_loaded > 0:
        status = selector.get_status()
        print(f"\nTop models:")
        for m in status["top_models"]:
            print(f"  {m['model_id']}: score={m['score']:.3f}, AUC={m['test_auc']:.3f}")

        print(f"\nAvailable entry/exit windows:")
        for w in status["available_windows"]:
            print(f"  Entry: {w['entry_window']}, Exit: {w['exit_window']} ({w['n_models']} models)")
    else:
        print("\nNo models in registry. Run grid search first:")
        print("  from src.dynamic_model_selector import EntryExitGridSearchRunner")
        print("  runner = EntryExitGridSearchRunner()")
        print("  runner.run_grid_search(max_configs=10)")
