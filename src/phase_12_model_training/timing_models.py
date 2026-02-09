"""
GIGA TRADER - Entry/Exit Timing Models
========================================
ML models that LEARN optimal entry/exit times, position sizes, stop/take profit levels,
batch schedules, and dynamic guardrails from historical data.

This addresses the critical gap:
  - Current model predicts: Direction (swing), Timing (low before high)
  - This model predicts: SPECIFIC entry time, exit time, position size, stops, batches

Architecture:
  1. EntryTimeModel - Predicts optimal entry time (minutes from open)
  2. ExitTimeModel - Predicts optimal exit time (minutes from open)
  3. PositionSizeModel - Predicts optimal position size based on conditions
  4. StopTakeProfitModel - Predicts dynamic stop/take profit levels
  5. BatchScheduleModel - Predicts optimal batch entry/exit schedules
  6. GuardrailModel - Predicts when to apply emergency exits

Combined: EntryExitTimingModel orchestrates all sub-models.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import (
    GradientBoostingRegressor,
    GradientBoostingClassifier,
    RandomForestRegressor,
    RandomForestClassifier,
)
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import joblib

from src.phase_05_targets.timing_targets import TargetLabeler
from src.phase_06_features_intraday.timing_features import TimingFeatureEngineer


# =============================================================================
# SUB-MODELS
# =============================================================================

class EntryTimeModel:
    """Predicts optimal entry time (minutes from market open)."""

    def __init__(
        self,
        model_type: str = "gradient_boosting",
        entry_window: Tuple[int, int] = (0, 120),
    ):
        self.model_type = model_type
        self.entry_window = entry_window
        self.model = None
        self._setup_model()

    def _setup_model(self):
        if self.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                min_samples_leaf=20,
                subsample=0.8,
                random_state=42,
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=50,
                max_depth=5,
                min_samples_leaf=20,
                random_state=42,
            )
        else:  # ridge
            self.model = Ridge(alpha=1.0)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        """Train the model."""
        # Clip targets to valid window
        y_clipped = np.clip(y, self.entry_window[0], self.entry_window[1])

        if sample_weight is not None and hasattr(self.model, 'fit'):
            if 'sample_weight' in self.model.fit.__code__.co_varnames:
                self.model.fit(X, y_clipped, sample_weight=sample_weight)
            else:
                self.model.fit(X, y_clipped)
        else:
            self.model.fit(X, y_clipped)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict entry time."""
        pred = self.model.predict(X)
        return np.clip(pred, self.entry_window[0], self.entry_window[1])

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        pred = self.predict(X)
        return {
            "mae_minutes": mean_absolute_error(y, pred),
            "rmse_minutes": np.sqrt(mean_squared_error(y, pred)),
            "r2": r2_score(y, pred),
            "within_15min": np.mean(np.abs(y - pred) <= 15),
            "within_30min": np.mean(np.abs(y - pred) <= 30),
        }


class ExitTimeModel:
    """Predicts optimal exit time (minutes from market open)."""

    def __init__(
        self,
        model_type: str = "gradient_boosting",
        exit_window: Tuple[int, int] = (180, 385),
    ):
        self.model_type = model_type
        self.exit_window = exit_window
        self.model = None
        self._setup_model()

    def _setup_model(self):
        if self.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                min_samples_leaf=20,
                subsample=0.8,
                random_state=42,
            )
        elif self.model_type == "random_forest":
            self.model = RandomForestRegressor(
                n_estimators=50,
                max_depth=5,
                min_samples_leaf=20,
                random_state=42,
            )
        else:
            self.model = Ridge(alpha=1.0)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        """Train the model."""
        y_clipped = np.clip(y, self.exit_window[0], self.exit_window[1])

        if sample_weight is not None and hasattr(self.model, 'fit'):
            if 'sample_weight' in self.model.fit.__code__.co_varnames:
                self.model.fit(X, y_clipped, sample_weight=sample_weight)
            else:
                self.model.fit(X, y_clipped)
        else:
            self.model.fit(X, y_clipped)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict exit time."""
        pred = self.model.predict(X)
        return np.clip(pred, self.exit_window[0], self.exit_window[1])

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        pred = self.predict(X)
        return {
            "mae_minutes": mean_absolute_error(y, pred),
            "rmse_minutes": np.sqrt(mean_squared_error(y, pred)),
            "r2": r2_score(y, pred),
            "within_15min": np.mean(np.abs(y - pred) <= 15),
            "within_30min": np.mean(np.abs(y - pred) <= 30),
        }


class PositionSizeModel:
    """Predicts optimal position size (% of portfolio)."""

    def __init__(
        self,
        model_type: str = "gradient_boosting",
        min_position: float = 0.05,
        max_position: float = 0.25,
    ):
        self.model_type = model_type
        self.min_position = min_position
        self.max_position = max_position
        self.model = None
        self._setup_model()

    def _setup_model(self):
        if self.model_type == "gradient_boosting":
            self.model = GradientBoostingRegressor(
                n_estimators=30,
                max_depth=3,
                learning_rate=0.1,
                min_samples_leaf=20,
                random_state=42,
            )
        else:
            self.model = Ridge(alpha=1.0)

    def fit(self, X: np.ndarray, y: np.ndarray, sample_weight: np.ndarray = None):
        """Train the model."""
        y_clipped = np.clip(y, self.min_position, self.max_position)

        if sample_weight is not None and hasattr(self.model, 'fit'):
            if 'sample_weight' in self.model.fit.__code__.co_varnames:
                self.model.fit(X, y_clipped, sample_weight=sample_weight)
            else:
                self.model.fit(X, y_clipped)
        else:
            self.model.fit(X, y_clipped)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict position size."""
        pred = self.model.predict(X)
        return np.clip(pred, self.min_position, self.max_position)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        pred = self.predict(X)
        return {
            "mae_pct": mean_absolute_error(y, pred) * 100,
            "rmse_pct": np.sqrt(mean_squared_error(y, pred)) * 100,
            "r2": r2_score(y, pred),
        }


class StopTakeProfitModel:
    """Predicts optimal stop loss and take profit levels (%)."""

    def __init__(
        self,
        model_type: str = "gradient_boosting",
        min_stop: float = 0.005,
        max_stop: float = 0.05,
        min_tp: float = 0.005,
        max_tp: float = 0.05,
    ):
        self.model_type = model_type
        self.min_stop = min_stop
        self.max_stop = max_stop
        self.min_tp = min_tp
        self.max_tp = max_tp

        self.stop_model = None
        self.tp_model = None
        self._setup_models()

    def _setup_models(self):
        if self.model_type == "gradient_boosting":
            self.stop_model = GradientBoostingRegressor(
                n_estimators=30,
                max_depth=3,
                learning_rate=0.1,
                min_samples_leaf=20,
                random_state=42,
            )
            self.tp_model = GradientBoostingRegressor(
                n_estimators=30,
                max_depth=3,
                learning_rate=0.1,
                min_samples_leaf=20,
                random_state=43,
            )
        else:
            self.stop_model = Ridge(alpha=1.0)
            self.tp_model = Ridge(alpha=1.0)

    def fit(
        self,
        X: np.ndarray,
        y_stop: np.ndarray,
        y_tp: np.ndarray,
        sample_weight: np.ndarray = None,
    ):
        """Train both models."""
        y_stop_clipped = np.clip(y_stop, self.min_stop, self.max_stop)
        y_tp_clipped = np.clip(y_tp, self.min_tp, self.max_tp)

        self.stop_model.fit(X, y_stop_clipped)
        self.tp_model.fit(X, y_tp_clipped)

        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict stop loss and take profit."""
        stop_pred = np.clip(self.stop_model.predict(X), self.min_stop, self.max_stop)
        tp_pred = np.clip(self.tp_model.predict(X), self.min_tp, self.max_tp)
        return stop_pred, tp_pred

    def evaluate(
        self,
        X: np.ndarray,
        y_stop: np.ndarray,
        y_tp: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        stop_pred, tp_pred = self.predict(X)
        return {
            "stop_mae_bps": mean_absolute_error(y_stop, stop_pred) * 10000,
            "stop_rmse_bps": np.sqrt(mean_squared_error(y_stop, stop_pred)) * 10000,
            "tp_mae_bps": mean_absolute_error(y_tp, tp_pred) * 10000,
            "tp_rmse_bps": np.sqrt(mean_squared_error(y_tp, tp_pred)) * 10000,
        }


class BatchScheduleModel:
    """Predicts whether to batch and how many batches."""

    def __init__(self, model_type: str = "gradient_boosting"):
        self.model_type = model_type
        self.should_batch_model = None
        self.n_batches_model = None
        self._setup_models()

    def _setup_models(self):
        if self.model_type == "gradient_boosting":
            self.should_batch_model = GradientBoostingClassifier(
                n_estimators=30,
                max_depth=3,
                learning_rate=0.1,
                min_samples_leaf=20,
                random_state=42,
            )
            self.n_batches_model = GradientBoostingClassifier(
                n_estimators=30,
                max_depth=3,
                learning_rate=0.1,
                min_samples_leaf=20,
                random_state=43,
            )
        else:
            self.should_batch_model = LogisticRegression(C=0.1, max_iter=1000)
            self.n_batches_model = LogisticRegression(C=0.1, max_iter=1000)

    def fit(
        self,
        X: np.ndarray,
        y_should_batch: np.ndarray,
        y_n_batches: np.ndarray,
        sample_weight: np.ndarray = None,
    ):
        """Train both models."""
        self.should_batch_model.fit(X, y_should_batch.astype(int))

        # Only train n_batches on samples where batching is optimal
        batch_mask = y_should_batch.astype(bool)
        if batch_mask.sum() > 10:
            self.n_batches_model.fit(X[batch_mask], y_n_batches[batch_mask])
        else:
            # Not enough samples, use all data
            self.n_batches_model.fit(X, y_n_batches)

        return self

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict batching decisions."""
        should_batch = self.should_batch_model.predict(X)
        n_batches = self.n_batches_model.predict(X)

        # If not batching, set n_batches to 1
        n_batches = np.where(should_batch, n_batches, 1)

        return should_batch, n_batches

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of batching being beneficial."""
        return self.should_batch_model.predict_proba(X)[:, 1]

    def evaluate(
        self,
        X: np.ndarray,
        y_should_batch: np.ndarray,
        y_n_batches: np.ndarray,
    ) -> Dict[str, float]:
        """Evaluate model performance."""
        should_batch_pred, n_batches_pred = self.predict(X)

        return {
            "should_batch_accuracy": accuracy_score(y_should_batch, should_batch_pred),
            "should_batch_precision": precision_score(y_should_batch, should_batch_pred, zero_division=0),
            "n_batches_accuracy": accuracy_score(y_n_batches, n_batches_pred),
        }


class GuardrailModel:
    """Predicts when emergency exits or guardrails should trigger."""

    def __init__(self, model_type: str = "gradient_boosting"):
        self.model_type = model_type
        self.model = None
        self._setup_model()

    def _setup_model(self):
        if self.model_type == "gradient_boosting":
            self.model = GradientBoostingClassifier(
                n_estimators=30,
                max_depth=3,
                learning_rate=0.1,
                min_samples_leaf=20,
                random_state=42,
            )
        else:
            self.model = LogisticRegression(C=0.1, max_iter=1000)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray = None,
    ):
        """Train the model. y = 1 if guardrail should have triggered."""
        self.model.fit(X, y.astype(int))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict whether guardrail should trigger."""
        return self.model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probability of guardrail trigger."""
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance."""
        pred = self.predict(X)
        return {
            "accuracy": accuracy_score(y, pred),
            "precision": precision_score(y, pred, zero_division=0),
            "recall": recall_score(y, pred, zero_division=0),
            "f1": f1_score(y, pred, zero_division=0),
        }


# =============================================================================
# COMBINED ENTRY/EXIT TIMING MODEL
# =============================================================================

class EntryExitTimingModel:
    """
    Combined model that orchestrates all sub-models for complete
    entry/exit decision making.

    This is the main model to use - it coordinates:
      - Entry time prediction
      - Exit time prediction
      - Position sizing
      - Stop/Take profit levels
      - Batch scheduling
      - Guardrail triggers
    """

    def __init__(
        self,
        model_type: str = "gradient_boosting",
        entry_window: Tuple[int, int] = (0, 120),
        exit_window: Tuple[int, int] = (180, 385),
        min_position_pct: float = 0.05,
        max_position_pct: float = 0.25,
    ):
        self.model_type = model_type
        self.entry_window = entry_window
        self.exit_window = exit_window
        self.min_position_pct = min_position_pct
        self.max_position_pct = max_position_pct

        # Sub-models
        self.entry_model = EntryTimeModel(model_type, entry_window)
        self.exit_model = ExitTimeModel(model_type, exit_window)
        self.position_model = PositionSizeModel(model_type, min_position_pct, max_position_pct)
        self.stop_tp_model = StopTakeProfitModel(model_type)
        self.batch_model = BatchScheduleModel(model_type)
        self.guardrail_model = GuardrailModel(model_type)

        # Feature engineering
        self.feature_engineer = TimingFeatureEngineer()
        self.target_labeler = TargetLabeler(
            entry_window=entry_window,
            exit_window=exit_window,
        )

        # Training state
        self.is_fitted = False
        self.training_metrics = {}

    def fit(
        self,
        daily_data: pd.DataFrame,
        intraday_data: pd.DataFrame,
        directions: pd.Series,
        sample_weight: np.ndarray = None,
        cv_folds: int = 3,
    ) -> Dict[str, Any]:
        """
        Train all sub-models on historical data.

        Args:
            daily_data: Daily OHLCV data
            intraday_data: Intraday (minute) OHLCV data
            directions: Series of trade directions indexed by date
            sample_weight: Optional sample weights
            cv_folds: Number of cross-validation folds

        Returns:
            Dict with training metrics
        """
        print("[EntryExitTimingModel] Starting training...")

        # Step 1: Create targets
        print("  Creating targets from historical data...")
        targets = self.target_labeler.create_training_targets(intraday_data, directions)

        if len(targets) < 50:
            print(f"  WARNING: Only {len(targets)} valid training samples")

        # Step 2: Create features
        print("  Engineering timing features...")
        features = self.feature_engineer.create_features(daily_data, intraday_data)

        # Align features and targets
        common_idx = features.index.intersection(targets.index)
        X = features.loc[common_idx]
        y = targets.loc[common_idx]

        print(f"  Training samples: {len(X)}")

        # Step 3: Scale features
        X_scaled = self.feature_engineer.fit_transform(X)

        # Step 4: Train each sub-model with time series CV
        tscv = TimeSeriesSplit(n_splits=cv_folds)

        all_metrics = {
            "entry": [],
            "exit": [],
            "position": [],
            "stop_tp": [],
            "batch": [],
        }

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
            print(f"  Training fold {fold + 1}/{cv_folds}...")

            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            # Train entry model
            self.entry_model.fit(X_train, y_train['optimal_entry_minute'].values)
            entry_metrics = self.entry_model.evaluate(X_val, y_val['optimal_entry_minute'].values)
            all_metrics["entry"].append(entry_metrics)

            # Train exit model
            self.exit_model.fit(X_train, y_train['optimal_exit_minute'].values)
            exit_metrics = self.exit_model.evaluate(X_val, y_val['optimal_exit_minute'].values)
            all_metrics["exit"].append(exit_metrics)

            # Train position size model
            self.position_model.fit(X_train, y_train['optimal_position_pct'].values)
            position_metrics = self.position_model.evaluate(X_val, y_val['optimal_position_pct'].values)
            all_metrics["position"].append(position_metrics)

            # Train stop/TP model
            self.stop_tp_model.fit(
                X_train,
                y_train['optimal_stop_pct'].values,
                y_train['optimal_take_profit_pct'].values,
            )
            stop_tp_metrics = self.stop_tp_model.evaluate(
                X_val,
                y_val['optimal_stop_pct'].values,
                y_val['optimal_take_profit_pct'].values,
            )
            all_metrics["stop_tp"].append(stop_tp_metrics)

            # Train batch model
            self.batch_model.fit(
                X_train,
                y_train['should_batch'].values,
                y_train['optimal_batches'].values,
            )
            batch_metrics = self.batch_model.evaluate(
                X_val,
                y_val['should_batch'].values,
                y_val['optimal_batches'].values,
            )
            all_metrics["batch"].append(batch_metrics)

        # Final fit on all data
        print("  Final fit on all data...")
        self.entry_model.fit(X_scaled, y['optimal_entry_minute'].values)
        self.exit_model.fit(X_scaled, y['optimal_exit_minute'].values)
        self.position_model.fit(X_scaled, y['optimal_position_pct'].values)
        self.stop_tp_model.fit(
            X_scaled,
            y['optimal_stop_pct'].values,
            y['optimal_take_profit_pct'].values,
        )
        self.batch_model.fit(
            X_scaled,
            y['should_batch'].values,
            y['optimal_batches'].values,
        )

        # Create guardrail labels (did the trade lose > 3%?)
        guardrail_labels = (y['optimal_return'] < -0.03).astype(int).values
        self.guardrail_model.fit(X_scaled, guardrail_labels)

        self.is_fitted = True

        # Aggregate metrics
        self.training_metrics = self._aggregate_metrics(all_metrics)

        print("\n  Training complete!")
        self._print_metrics()

        return self.training_metrics

    def _aggregate_metrics(self, all_metrics: Dict) -> Dict:
        """Aggregate metrics across CV folds."""
        aggregated = {}

        for model_name, fold_metrics in all_metrics.items():
            aggregated[model_name] = {}
            if not fold_metrics:
                continue

            for metric_name in fold_metrics[0].keys():
                values = [m[metric_name] for m in fold_metrics]
                aggregated[model_name][metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                }

        return aggregated

    def _print_metrics(self):
        """Print training metrics summary."""
        print("\n  ═══════════════════════════════════════════════")
        print("  ENTRY/EXIT TIMING MODEL - Training Summary")
        print("  ═══════════════════════════════════════════════")

        if "entry" in self.training_metrics:
            entry = self.training_metrics["entry"]
            print(f"\n  Entry Time Model:")
            print(f"    MAE: {entry.get('mae_minutes', {}).get('mean', 0):.1f} ± {entry.get('mae_minutes', {}).get('std', 0):.1f} min")
            print(f"    Within 15 min: {entry.get('within_15min', {}).get('mean', 0)*100:.1f}%")
            print(f"    Within 30 min: {entry.get('within_30min', {}).get('mean', 0)*100:.1f}%")

        if "exit" in self.training_metrics:
            exit_m = self.training_metrics["exit"]
            print(f"\n  Exit Time Model:")
            print(f"    MAE: {exit_m.get('mae_minutes', {}).get('mean', 0):.1f} ± {exit_m.get('mae_minutes', {}).get('std', 0):.1f} min")
            print(f"    Within 15 min: {exit_m.get('within_15min', {}).get('mean', 0)*100:.1f}%")
            print(f"    Within 30 min: {exit_m.get('within_30min', {}).get('mean', 0)*100:.1f}%")

        if "position" in self.training_metrics:
            pos = self.training_metrics["position"]
            print(f"\n  Position Size Model:")
            print(f"    MAE: {pos.get('mae_pct', {}).get('mean', 0):.2f}%")
            print(f"    R²: {pos.get('r2', {}).get('mean', 0):.3f}")

        if "stop_tp" in self.training_metrics:
            stp = self.training_metrics["stop_tp"]
            print(f"\n  Stop/Take Profit Model:")
            print(f"    Stop MAE: {stp.get('stop_mae_bps', {}).get('mean', 0):.1f} bps")
            print(f"    TP MAE: {stp.get('tp_mae_bps', {}).get('mean', 0):.1f} bps")

        if "batch" in self.training_metrics:
            batch = self.training_metrics["batch"]
            print(f"\n  Batch Schedule Model:")
            print(f"    Should Batch Accuracy: {batch.get('should_batch_accuracy', {}).get('mean', 0)*100:.1f}%")
            print(f"    N Batches Accuracy: {batch.get('n_batches_accuracy', {}).get('mean', 0)*100:.1f}%")

        print("  ═══════════════════════════════════════════════")

    def predict(
        self,
        features: pd.DataFrame,
        swing_proba: float = None,
        timing_proba: float = None,
        current_price: float = None,
    ) -> Dict[str, Any]:
        """
        Make complete entry/exit predictions.

        Args:
            features: Feature DataFrame (or single row)
            swing_proba: Swing model probability (optional, for confidence scaling)
            timing_proba: Timing model probability (optional)
            current_price: Current price for stop/TP calculation

        Returns:
            Dict with all predictions
        """
        if not self.is_fitted:
            raise RuntimeError("Model not fitted. Call fit() first.")

        # Scale features
        X = self.feature_engineer.transform(features)

        # Get predictions from all sub-models
        entry_time = self.entry_model.predict(X)
        exit_time = self.exit_model.predict(X)
        position_size = self.position_model.predict(X)
        stop_pct, tp_pct = self.stop_tp_model.predict(X)
        should_batch, n_batches = self.batch_model.predict(X)
        guardrail_prob = self.guardrail_model.predict_proba(X)

        # Adjust position size by swing confidence if provided
        if swing_proba is not None:
            confidence_scale = 0.5 + swing_proba  # Scale 0.5-1.5x
            position_size = position_size * confidence_scale
            position_size = np.clip(position_size, self.min_position_pct, self.max_position_pct)

        # Calculate actual stop/TP prices if current price provided
        stop_price = None
        tp_price = None
        if current_price is not None:
            stop_price = current_price * (1 - stop_pct)
            tp_price = current_price * (1 + tp_pct)

        # Create batch schedule
        batch_schedule = self._create_batch_schedule(
            int(entry_time[0]) if len(entry_time) == 1 else int(entry_time.mean()),
            int(n_batches[0]) if len(n_batches) == 1 else int(n_batches.mean()),
            float(position_size[0]) if len(position_size) == 1 else float(position_size.mean()),
            bool(should_batch[0]) if len(should_batch) == 1 else bool(should_batch.mean() > 0.5),
        )

        # Determine guardrail recommendations
        guardrails = {
            "risk_level": "HIGH" if guardrail_prob.mean() > 0.5 else "NORMAL",
            "guardrail_probability": float(guardrail_prob.mean()),
            "recommended_actions": [],
        }

        if guardrail_prob.mean() > 0.7:
            guardrails["recommended_actions"].append("Reduce position size")
            guardrails["recommended_actions"].append("Tighten stop loss")
        elif guardrail_prob.mean() > 0.5:
            guardrails["recommended_actions"].append("Use tighter stop loss")

        return {
            # Core timing
            "entry_time_minutes": int(entry_time[0]) if len(entry_time) == 1 else entry_time.tolist(),
            "exit_time_minutes": int(exit_time[0]) if len(exit_time) == 1 else exit_time.tolist(),

            # Position sizing
            "position_size_pct": float(position_size[0]) if len(position_size) == 1 else position_size.tolist(),

            # Stop/Take profit
            "stop_loss_pct": float(stop_pct[0]) if len(stop_pct) == 1 else stop_pct.tolist(),
            "take_profit_pct": float(tp_pct[0]) if len(tp_pct) == 1 else tp_pct.tolist(),
            "stop_loss_price": float(stop_price[0]) if stop_price is not None and len(stop_price) == 1 else None,
            "take_profit_price": float(tp_price[0]) if tp_price is not None and len(tp_price) == 1 else None,

            # Batching
            "should_batch": bool(should_batch[0]) if len(should_batch) == 1 else should_batch.tolist(),
            "n_batches": int(n_batches[0]) if len(n_batches) == 1 else n_batches.tolist(),
            "batch_schedule": batch_schedule,

            # Guardrails
            "guardrails": guardrails,

            # Confidence
            "swing_confidence": swing_proba,
            "timing_confidence": timing_proba,
        }

    def _create_batch_schedule(
        self,
        entry_minute: int,
        n_batches: int,
        total_position: float,
        should_batch: bool,
    ) -> List[Dict]:
        """Create batch entry schedule."""
        if not should_batch or n_batches <= 1:
            return [{
                "batch_num": 1,
                "entry_minute": entry_minute,
                "position_pct": total_position,
            }]

        schedule = []
        batch_interval = 10  # 10 minutes between batches

        # Pyramid sizing (start small, increase)
        weights = list(range(1, n_batches + 1))
        total_weight = sum(weights)

        for i in range(n_batches):
            schedule.append({
                "batch_num": i + 1,
                "entry_minute": entry_minute + (i * batch_interval),
                "position_pct": total_position * weights[i] / total_weight,
            })

        return schedule

    def save(self, path: str):
        """Save all models to disk."""
        save_dict = {
            "entry_model": self.entry_model.model,
            "exit_model": self.exit_model.model,
            "position_model": self.position_model.model,
            "stop_model": self.stop_tp_model.stop_model,
            "tp_model": self.stop_tp_model.tp_model,
            "should_batch_model": self.batch_model.should_batch_model,
            "n_batches_model": self.batch_model.n_batches_model,
            "guardrail_model": self.guardrail_model.model,
            "feature_scaler": self.feature_engineer.scaler,
            "feature_names": self.feature_engineer.feature_names,
            "training_metrics": self.training_metrics,
            "config": {
                "model_type": self.model_type,
                "entry_window": self.entry_window,
                "exit_window": self.exit_window,
                "min_position_pct": self.min_position_pct,
                "max_position_pct": self.max_position_pct,
            },
        }

        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(save_dict, path)
        print(f"[SAVED] EntryExitTimingModel to {path}")

    @classmethod
    def load(cls, path: str) -> "EntryExitTimingModel":
        """Load models from disk."""
        save_dict = joblib.load(path)

        config = save_dict.get("config", {})
        model = cls(
            model_type=config.get("model_type", "gradient_boosting"),
            entry_window=config.get("entry_window", (0, 120)),
            exit_window=config.get("exit_window", (180, 385)),
            min_position_pct=config.get("min_position_pct", 0.05),
            max_position_pct=config.get("max_position_pct", 0.25),
        )

        # Restore models
        model.entry_model.model = save_dict["entry_model"]
        model.exit_model.model = save_dict["exit_model"]
        model.position_model.model = save_dict["position_model"]
        model.stop_tp_model.stop_model = save_dict["stop_model"]
        model.stop_tp_model.tp_model = save_dict["tp_model"]
        model.batch_model.should_batch_model = save_dict["should_batch_model"]
        model.batch_model.n_batches_model = save_dict["n_batches_model"]
        model.guardrail_model.model = save_dict["guardrail_model"]
        model.feature_engineer.scaler = save_dict["feature_scaler"]
        model.feature_engineer.feature_names = save_dict["feature_names"]
        model.training_metrics = save_dict.get("training_metrics", {})
        model.is_fitted = True

        print(f"[LOADED] EntryExitTimingModel from {path}")
        return model


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def create_entry_exit_model(
    daily_data: pd.DataFrame,
    intraday_data: pd.DataFrame,
    directions: pd.Series = None,
    model_type: str = "gradient_boosting",
    save_path: str = None,
) -> EntryExitTimingModel:
    """
    Convenience function to create and train an EntryExitTimingModel.

    Args:
        daily_data: Daily OHLCV data
        intraday_data: Intraday OHLCV data
        directions: Trade directions by date (defaults to all LONG)
        model_type: Type of model to use
        save_path: Optional path to save the model

    Returns:
        Trained EntryExitTimingModel
    """
    if directions is None:
        # Default to LONG for all days
        directions = pd.Series("LONG", index=daily_data.index)

    model = EntryExitTimingModel(model_type=model_type)
    model.fit(daily_data, intraday_data, directions)

    if save_path:
        model.save(save_path)

    return model


# =============================================================================
# MAIN - Testing
# =============================================================================

if __name__ == "__main__":
    print("Entry/Exit Timing Model")
    print("=" * 60)

    # Create synthetic test data
    np.random.seed(42)
    n_days = 200

    # Daily data
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    daily_data = pd.DataFrame({
        "open": 450 + np.cumsum(np.random.randn(n_days) * 2),
        "high": 0,
        "low": 0,
        "close": 0,
        "volume": np.random.randint(50000000, 100000000, n_days),
    }, index=dates)

    daily_data["high"] = daily_data["open"] + np.abs(np.random.randn(n_days)) * 3
    daily_data["low"] = daily_data["open"] - np.abs(np.random.randn(n_days)) * 3
    daily_data["close"] = daily_data["open"] + np.random.randn(n_days) * 2

    # Intraday data (simplified)
    intraday_records = []
    for date in dates:
        base_price = daily_data.loc[date, "open"]
        for minute in range(0, 391, 5):  # 5-minute bars
            ts = pd.Timestamp(date) + pd.Timedelta(hours=9, minutes=30) + pd.Timedelta(minutes=minute)
            price = base_price + np.random.randn() * 0.5 + minute * 0.001  # Slight upward drift
            intraday_records.append({
                "datetime": ts,
                "open": price,
                "high": price + abs(np.random.randn()) * 0.3,
                "low": price - abs(np.random.randn()) * 0.3,
                "close": price + np.random.randn() * 0.2,
                "volume": np.random.randint(100000, 500000),
            })

    intraday_data = pd.DataFrame(intraday_records).set_index("datetime")

    # Directions (random long/short)
    directions = pd.Series(
        np.where(np.random.random(n_days) > 0.3, "LONG", "SHORT"),
        index=dates,
    )

    print(f"\nTest data: {n_days} days, {len(intraday_data)} intraday bars")

    # Create and train model
    print("\n" + "=" * 60)
    print("Training EntryExitTimingModel...")
    print("=" * 60)

    model = EntryExitTimingModel(model_type="gradient_boosting")
    metrics = model.fit(daily_data, intraday_data, directions, cv_folds=3)

    # Test prediction
    print("\n" + "=" * 60)
    print("Testing Prediction...")
    print("=" * 60)

    # Create features for last day
    test_features = model.feature_engineer.create_features(daily_data.tail(5), None)
    test_features = test_features.tail(1)  # Last day

    prediction = model.predict(
        test_features,
        swing_proba=0.72,
        timing_proba=0.65,
        current_price=460.0,
    )

    print(f"\nPrediction for test day:")
    print(f"  Entry Time: {prediction['entry_time_minutes']} minutes from open")
    print(f"  Exit Time: {prediction['exit_time_minutes']} minutes from open")
    print(f"  Position Size: {prediction['position_size_pct']*100:.1f}%")
    print(f"  Stop Loss: {prediction['stop_loss_pct']*100:.2f}% (${prediction['stop_loss_price']:.2f})")
    print(f"  Take Profit: {prediction['take_profit_pct']*100:.2f}% (${prediction['take_profit_price']:.2f})")
    print(f"  Should Batch: {prediction['should_batch']} ({prediction['n_batches']} batches)")
    print(f"  Risk Level: {prediction['guardrails']['risk_level']}")

    if prediction['batch_schedule']:
        print(f"\n  Batch Schedule:")
        for batch in prediction['batch_schedule']:
            print(f"    Batch {batch['batch_num']}: minute {batch['entry_minute']}, {batch['position_pct']*100:.1f}%")

    print("\n" + "=" * 60)
    print("Entry/Exit Timing Model loaded successfully!")
    print("=" * 60)
