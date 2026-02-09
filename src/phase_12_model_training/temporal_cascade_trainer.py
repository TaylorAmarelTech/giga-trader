"""
GIGA TRADER - Temporal Cascade Model Trainer
=============================================
Integrates temporal cascade training with the existing experiment engine.

This module:
  1. Trains temporal cascade models using the full data pipeline
  2. Registers models with the experiment engine
  3. Provides integration with signal generation

Usage:
    from src.temporal_cascade_trainer import train_temporal_cascade

    # Train cascade ensemble
    result = train_temporal_cascade(
        df_daily=daily_data,
        df_1min_dict=intraday_data,
        config=experiment_config
    )
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import warnings

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
import joblib

# Import temporal cascade models
from src.phase_26_temporal.temporal_cascade_models import (
    TemporalCascadeEnsemble,
    TemporalFeatureEngineer,
    TemporalSliceModel,
    TemporalTargetLabeler,
    TemporalPrediction,
    CascadeEnsemblePrediction,
)

logger = logging.getLogger("TEMPORAL_CASCADE_TRAINER")


# =============================================================================
# TRAINING CONFIG
# =============================================================================
TEMPORAL_CASCADE_CONFIG = {
    # Temporal slices to train (minutes from market open)
    "temporal_slices": [0, 30, 60, 90, 120, 180],

    # Model settings
    "model_type": "gradient_boosting",  # gradient_boosting, logistic, ensemble
    "regularization_strength": 0.1,

    # Training settings
    "cv_folds": 5,
    "purge_days": 5,
    "embargo_days": 2,

    # Required for registration
    "min_cv_auc": 0.55,  # Minimum CV AUC to register model
    "min_models_passing": 3,  # At least 3 temporal slices must pass

    # Save paths
    "model_dir": project_root / "models" / "temporal_cascade",
    "model_filename": "temporal_cascade_ensemble.joblib",
}


# =============================================================================
# DATA PREPARATION
# =============================================================================
def prepare_intraday_data_dict(
    df_1min: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Convert minute-level DataFrame to dictionary keyed by date.

    Args:
        df_1min: DataFrame with timestamp column/index and OHLCV

    Returns:
        Dict mapping date string -> minute data for that date
    """
    # Ensure we have a datetime index or column
    if 'timestamp' in df_1min.columns:
        df = df_1min.copy()
        df['date'] = pd.to_datetime(df['timestamp']).dt.date.astype(str)
    elif isinstance(df_1min.index, pd.DatetimeIndex):
        df = df_1min.copy()
        df['date'] = df.index.date.astype(str)
    else:
        logger.warning("Cannot extract dates from intraday data")
        return {}

    # Group by date
    result = {}
    for date, group in df.groupby('date'):
        result[date] = group.copy().reset_index(drop=True)

    return result


def aggregate_to_daily(df_1min: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate minute data to daily OHLCV.

    Args:
        df_1min: Minute-level DataFrame

    Returns:
        Daily OHLCV DataFrame
    """
    # Ensure we have a date
    if 'timestamp' in df_1min.columns:
        df = df_1min.copy()
        df['date'] = pd.to_datetime(df['timestamp']).dt.date
    elif isinstance(df_1min.index, pd.DatetimeIndex):
        df = df_1min.copy()
        df['date'] = df.index.date
    else:
        raise ValueError("Cannot extract dates from intraday data")

    # Aggregate
    daily = df.groupby('date').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).reset_index()

    daily['date'] = pd.to_datetime(daily['date'])

    return daily


# =============================================================================
# TRAINING RESULT
# =============================================================================
@dataclass
class TemporalCascadeTrainResult:
    """Result of training a temporal cascade ensemble."""
    success: bool
    model_path: str = ""
    training_time_seconds: float = 0.0
    timestamp: str = ""

    # Per-slice metrics
    slice_metrics: Dict[int, Dict[str, float]] = field(default_factory=dict)

    # Ensemble metrics
    ensemble_cv_auc: float = 0.0
    n_slices_trained: int = 0
    n_slices_passing: int = 0

    # Error info
    error_message: str = ""

    def to_dict(self) -> Dict:
        return asdict(self)

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 60,
            "TEMPORAL CASCADE TRAINING RESULT",
            "=" * 60,
            f"Success: {self.success}",
            f"Training time: {self.training_time_seconds:.1f}s",
            f"Slices trained: {self.n_slices_trained}",
            f"Slices passing: {self.n_slices_passing}",
            f"Ensemble CV AUC: {self.ensemble_cv_auc:.4f}",
            "",
            "Per-Slice Metrics:",
        ]

        for ts, metrics in sorted(self.slice_metrics.items()):
            cv_auc = metrics.get('swing_cv_auc', 0)
            passing = "✓" if cv_auc >= TEMPORAL_CASCADE_CONFIG['min_cv_auc'] else "✗"
            lines.append(f"  T{ts:3d}: CV AUC = {cv_auc:.4f} {passing}")

        if self.model_path:
            lines.append(f"\nModel saved to: {self.model_path}")

        if self.error_message:
            lines.append(f"\nError: {self.error_message}")

        lines.append("=" * 60)
        return "\n".join(lines)


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================
def train_temporal_cascade(
    df_daily: pd.DataFrame = None,
    df_1min: pd.DataFrame = None,
    df_1min_dict: Dict[str, pd.DataFrame] = None,
    config: Dict = None,
    save_model: bool = True,
) -> TemporalCascadeTrainResult:
    """
    Train temporal cascade ensemble.

    This function:
      1. Prepares data for temporal training
      2. Trains models for each temporal slice (T0, T30, T60, T90, T120, T180)
      3. Evaluates each slice with cross-validation
      4. Saves the ensemble if enough slices pass the threshold

    Args:
        df_daily: Daily OHLCV data (optional if df_1min provided)
        df_1min: Minute-level OHLCV data (optional if df_daily provided)
        df_1min_dict: Pre-prepared dict mapping date -> minute data
        config: Optional config overrides
        save_model: Whether to save the trained ensemble

    Returns:
        TemporalCascadeTrainResult with training metrics
    """
    start_time = time.time()
    cfg = {**TEMPORAL_CASCADE_CONFIG, **(config or {})}

    logger.info("=" * 70)
    logger.info("TEMPORAL CASCADE TRAINING")
    logger.info("=" * 70)

    result = TemporalCascadeTrainResult(
        success=False,
        timestamp=datetime.now().isoformat(),
    )

    try:
        # Step 1: Prepare data
        logger.info("[STEP 1] Preparing data...")

        if df_daily is None and df_1min is not None:
            logger.info("  Aggregating minute data to daily...")
            df_daily = aggregate_to_daily(df_1min)

        if df_daily is None:
            raise ValueError("Must provide either df_daily or df_1min")

        if df_1min_dict is None and df_1min is not None:
            logger.info("  Preparing intraday data dictionary...")
            df_1min_dict = prepare_intraday_data_dict(df_1min)

        logger.info(f"  Daily samples: {len(df_daily)}")
        logger.info(f"  Intraday dates: {len(df_1min_dict) if df_1min_dict else 0}")

        # Step 2: Create ensemble
        logger.info("[STEP 2] Creating temporal cascade ensemble...")
        ensemble = TemporalCascadeEnsemble(
            model_type=cfg['model_type'],
            regularization_strength=cfg['regularization_strength'],
        )

        # Step 3: Train
        logger.info("[STEP 3] Training temporal slice models...")
        metrics = ensemble.fit(
            df_daily=df_daily,
            df_1min_dict=df_1min_dict,
        )

        # Step 4: Evaluate
        logger.info("[STEP 4] Evaluating slice performance...")

        n_trained = 0
        n_passing = 0
        cv_aucs = []

        for ts in cfg['temporal_slices']:
            model = ensemble.models.get(ts)
            if model and model.is_fitted:
                n_trained += 1
                cv_auc = model.cv_auc
                cv_aucs.append(cv_auc)

                result.slice_metrics[ts] = {
                    'swing_cv_auc': cv_auc,
                    'is_fitted': True,
                }

                if cv_auc >= cfg['min_cv_auc']:
                    n_passing += 1
                    logger.info(f"  T{ts:3d}: CV AUC = {cv_auc:.4f} [PASS]")
                else:
                    logger.info(f"  T{ts:3d}: CV AUC = {cv_auc:.4f} [FAIL]")
            else:
                result.slice_metrics[ts] = {
                    'swing_cv_auc': 0.0,
                    'is_fitted': False,
                }
                logger.warning(f"  T{ts:3d}: Not trained")

        result.n_slices_trained = n_trained
        result.n_slices_passing = n_passing
        result.ensemble_cv_auc = np.mean(cv_aucs) if cv_aucs else 0.0

        # Step 5: Check if enough slices pass
        if n_passing >= cfg['min_models_passing']:
            logger.info(f"[PASS] {n_passing}/{n_trained} slices meet threshold")
            result.success = True

            # Save model if requested
            if save_model:
                model_dir = Path(cfg['model_dir'])
                model_dir.mkdir(parents=True, exist_ok=True)

                model_path = model_dir / cfg['model_filename']
                ensemble.save(model_path)

                result.model_path = str(model_path)
                logger.info(f"  Saved ensemble to: {model_path}")
        else:
            logger.warning(f"[FAIL] Only {n_passing}/{n_trained} slices meet threshold (need {cfg['min_models_passing']})")
            result.error_message = f"Only {n_passing} slices passed (need {cfg['min_models_passing']})"

    except Exception as e:
        import traceback
        result.error_message = str(e)
        logger.error(f"Training failed: {e}")
        logger.error(traceback.format_exc())

    result.training_time_seconds = time.time() - start_time

    logger.info(result.summary())

    return result


def load_temporal_cascade(
    model_path: Path = None
) -> Optional[TemporalCascadeEnsemble]:
    """
    Load a trained temporal cascade ensemble.

    Args:
        model_path: Path to saved model (uses default if not specified)

    Returns:
        TemporalCascadeEnsemble or None if not found
    """
    if model_path is None:
        model_path = TEMPORAL_CASCADE_CONFIG['model_dir'] / TEMPORAL_CASCADE_CONFIG['model_filename']

    model_path = Path(model_path)

    if not model_path.exists():
        logger.warning(f"Model not found at: {model_path}")
        return None

    try:
        ensemble = TemporalCascadeEnsemble.load(model_path)
        logger.info(f"Loaded temporal cascade from: {model_path}")
        return ensemble
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return None


# =============================================================================
# EXPERIMENT ENGINE INTEGRATION
# =============================================================================
def register_temporal_cascade_experiment(
    experiment_engine,
    result: TemporalCascadeTrainResult,
) -> bool:
    """
    Register temporal cascade training as an experiment.

    Args:
        experiment_engine: The ExperimentEngine instance
        result: Training result to register

    Returns:
        True if registered successfully
    """
    if not result.success:
        logger.warning("Cannot register failed training")
        return False

    try:
        # Create experiment record
        record = {
            "experiment_id": f"temporal_cascade_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "experiment_type": "temporal_cascade",
            "timestamp": result.timestamp,
            "metrics": {
                "ensemble_cv_auc": result.ensemble_cv_auc,
                "n_slices_trained": result.n_slices_trained,
                "n_slices_passing": result.n_slices_passing,
                "training_time_seconds": result.training_time_seconds,
            },
            "slice_metrics": result.slice_metrics,
            "model_path": result.model_path,
        }

        # Save to experiments directory
        exp_dir = project_root / "experiments"
        exp_dir.mkdir(parents=True, exist_ok=True)

        record_path = exp_dir / f"{record['experiment_id']}.json"
        with open(record_path, 'w') as f:
            json.dump(record, f, indent=2)

        logger.info(f"Registered experiment: {record['experiment_id']}")
        return True

    except Exception as e:
        logger.error(f"Failed to register experiment: {e}")
        return False


# =============================================================================
# SIGNAL GENERATION INTEGRATION
# =============================================================================
class TemporalCascadeSignalGenerator:
    """
    Generates trading signals using temporal cascade predictions.

    Integrates with the existing SignalGenerator to provide:
      1. Pre-market predictions (T0)
      2. Real-time updates as trading day progresses
      3. Ensemble predictions with agreement scoring
    """

    def __init__(
        self,
        ensemble: TemporalCascadeEnsemble = None,
        model_path: Path = None,
    ):
        """
        Initialize signal generator.

        Args:
            ensemble: Pre-loaded ensemble (optional)
            model_path: Path to load ensemble from (optional)
        """
        self.ensemble = ensemble

        if self.ensemble is None and model_path:
            self.ensemble = load_temporal_cascade(model_path)
        elif self.ensemble is None:
            self.ensemble = load_temporal_cascade()

        self.feature_engineer = TemporalFeatureEngineer()
        self.logger = logging.getLogger("TEMPORAL_SIGNAL")

    @property
    def is_available(self) -> bool:
        """Check if ensemble is loaded and ready."""
        return self.ensemble is not None and self.ensemble.is_fitted

    def generate_premarket_signal(
        self,
        historical_features: np.ndarray,
    ) -> TemporalPrediction:
        """
        Generate pre-market signal using T0 model (historical features only).

        Args:
            historical_features: Feature vector from historical data

        Returns:
            TemporalPrediction from T0 model
        """
        if not self.is_available:
            raise ValueError("Ensemble not loaded")

        intraday_feat = self.feature_engineer._get_empty_intraday_features()

        return self.ensemble.predict_at_time(
            historical_features=historical_features,
            intraday_features=intraday_feat,
            minutes_since_open=0,
        )

    def generate_realtime_signal(
        self,
        historical_features: np.ndarray,
        df_1min_today: pd.DataFrame,
        minutes_since_open: int,
    ) -> CascadeEnsemblePrediction:
        """
        Generate real-time signal using all available temporal models.

        Args:
            historical_features: Feature vector from historical data
            df_1min_today: Today's minute-level data so far
            minutes_since_open: Current time (minutes from market open)

        Returns:
            CascadeEnsemblePrediction with ensemble prediction
        """
        if not self.is_available:
            raise ValueError("Ensemble not loaded")

        # Engineer intraday features for each available slice
        intraday_features_by_slice = {}

        for ts in self.ensemble.TEMPORAL_SLICES:
            if ts <= minutes_since_open:
                intraday_feat = self.feature_engineer.engineer_intraday_features(
                    df_1min_today,
                    ts
                )
                intraday_features_by_slice[ts] = intraday_feat

        return self.ensemble.get_ensemble_prediction(
            historical_features=historical_features,
            intraday_features_by_slice=intraday_features_by_slice,
            minutes_since_open=minutes_since_open,
        )

    def get_trading_recommendation(
        self,
        historical_features: np.ndarray,
        df_1min_today: pd.DataFrame,
        minutes_since_open: int,
        current_position: str = "FLAT",
        swing_model_prediction: float = None,
        timing_model_prediction: float = None,
    ) -> Dict[str, Any]:
        """
        Get complete trading recommendation combining temporal cascade with existing models.

        Args:
            historical_features: Feature vector
            df_1min_today: Today's minute data
            minutes_since_open: Current time
            current_position: Current position (FLAT/LONG/SHORT)
            swing_model_prediction: Optional swing model probability
            timing_model_prediction: Optional timing model probability

        Returns:
            Trading recommendation dict
        """
        # Get temporal cascade prediction
        cascade_pred = self.generate_realtime_signal(
            historical_features=historical_features,
            df_1min_today=df_1min_today,
            minutes_since_open=minutes_since_open,
        )

        # Combine with existing models if available
        combined_swing = cascade_pred.swing_direction
        if swing_model_prediction is not None:
            # Weight: 40% cascade, 60% existing swing model
            combined_swing = 0.4 * cascade_pred.swing_direction + 0.6 * swing_model_prediction

        # Determine signal
        signal = "HOLD"
        if current_position == "FLAT":
            if combined_swing > 0.65 and cascade_pred.model_agreement > 0.6:
                signal = "BUY"
            elif combined_swing < 0.35 and cascade_pred.model_agreement > 0.6:
                signal = "SELL"
        else:
            # Already in position - check for exit
            if current_position == "LONG" and combined_swing < 0.45:
                signal = "EXIT_LONG"
            elif current_position == "SHORT" and combined_swing > 0.55:
                signal = "EXIT_SHORT"

        return {
            "signal": signal,
            "combined_swing_probability": float(combined_swing),
            "cascade_prediction": cascade_pred.to_dict(),
            "model_agreement": float(cascade_pred.model_agreement),
            "prediction_stability": float(cascade_pred.prediction_stability),
            "recommended_position_size": float(cascade_pred.recommended_position_size),
            "recommended_entry_time": cascade_pred.recommended_entry_time,
            "recommended_exit_time": cascade_pred.recommended_exit_time,
            "expected_magnitude": cascade_pred.expected_magnitude,
            "predicted_regime": cascade_pred.predicted_regime,
            "predicted_pattern": cascade_pred.predicted_pattern,
            "optimal_entry_window": cascade_pred.optimal_entry_window,
            "optimal_exit_window": cascade_pred.optimal_exit_window,
            "should_trade": cascade_pred.should_trade,
        }


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    print("=" * 70)
    print("TEMPORAL CASCADE TRAINER")
    print("=" * 70)
    print("""
This module trains temporal cascade models for anti-overfitting.

Usage:
    from src.temporal_cascade_trainer import train_temporal_cascade

    # Load your data
    df_daily = ...  # Daily OHLCV
    df_1min = ...   # Minute-level data

    # Train cascade
    result = train_temporal_cascade(
        df_daily=df_daily,
        df_1min=df_1min,
    )

    if result.success:
        print(f"Trained ensemble with {result.n_slices_passing} passing slices")
        print(f"Ensemble CV AUC: {result.ensemble_cv_auc:.4f}")

To train with existing data pipeline:
    python src/temporal_cascade_trainer.py --train
""")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train temporal cascade with existing data")
    args = parser.parse_args()

    if args.train:
        print("\n[INFO] Loading data from data manager...")
        try:
            from src.data_manager import get_spy_data

            df_1min = get_spy_data(years=3)  # Use 3 years for faster training

            if df_1min is not None and len(df_1min) > 0:
                print(f"[INFO] Loaded {len(df_1min):,} bars")

                result = train_temporal_cascade(
                    df_1min=df_1min,
                    save_model=True,
                )

                print(result.summary())
            else:
                print("[ERROR] No data loaded")
        except Exception as e:
            print(f"[ERROR] {e}")
            import traceback
            traceback.print_exc()
