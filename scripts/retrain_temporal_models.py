"""
GIGA TRADER - Retrain All Models with Temporal Integration
===========================================================
Resets the model registry and retrains all models with temporal variations.

This script:
  1. Resets the model registry (backs up old one)
  2. Downloads/loads data
  3. Engineers features
  4. Trains all temporal model variants:
     - Base models
     - Masked models (5 variants each)
     - Temporal cascade (T0, T30, T60, T90, T120, T180)
     - Intermittent masked cascade
     - Attention-weighted cascade
  5. Registers all models in the new temporal registry
  6. Selects best models for production

Usage:
    python scripts/retrain_temporal_models.py

    # With specific options
    python scripts/retrain_temporal_models.py --years 3 --reset-only
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

import numpy as np
import pandas as pd

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            project_root / "logs" / f"retrain_temporal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
    ]
)
logger = logging.getLogger("RETRAIN")


def main():
    parser = argparse.ArgumentParser(description="Retrain models with temporal integration")
    parser.add_argument("--years", type=int, default=3, help="Years of data to use")
    parser.add_argument("--reset-only", action="store_true", help="Only reset registry, don't train")
    parser.add_argument("--skip-cascade", action="store_true", help="Skip temporal cascade (faster)")
    args = parser.parse_args()

    print("=" * 70)
    print("TEMPORAL INTEGRATED RETRAINING")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")
    print(f"Years of data: {args.years}")
    print()

    # Import modules
    from src.temporal_integrated_training import (
        TemporalIntegratedTrainer,
        TemporalModelRegistry,
        reset_model_registry,
        train_all_temporal_models,
    )

    # Step 1: Reset registry
    logger.info("=" * 60)
    logger.info("STEP 1: RESET MODEL REGISTRY")
    logger.info("=" * 60)

    registry = reset_model_registry()
    logger.info("Registry reset complete")

    if args.reset_only:
        logger.info("--reset-only specified, exiting")
        return 0

    # Step 2: Load data
    logger.info("\n" + "=" * 60)
    logger.info("STEP 2: LOAD DATA")
    logger.info("=" * 60)

    try:
        from src.data_manager import get_spy_data
        df_1min = get_spy_data(years=args.years)

        if df_1min is None or len(df_1min) == 0:
            logger.error("No data loaded")
            return 1

        logger.info(f"Loaded {len(df_1min):,} bars")
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 3: Engineer features (reuse existing pipeline)
    logger.info("\n" + "=" * 60)
    logger.info("STEP 3: ENGINEER FEATURES")
    logger.info("=" * 60)

    try:
        from src.train_robust_model import (
            engineer_all_features,
            add_rolling_features,
            create_soft_targets,
        )

        # Engineer features from minute data (engineer_all_features aggregates internally)
        logger.info("Engineering features from minute data...")
        df_features = engineer_all_features(df_1min.copy(), swing_threshold=0.0025)
        logger.info(f"Daily features: {len(df_features)} days")

        # Add rolling features
        df_features = add_rolling_features(df_features)

        # Create targets (soft targets include target_up and target_timing)
        logger.info("Creating targets...")
        df_features = create_soft_targets(df_features, threshold=0.0025)

        # Get feature columns
        exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume',
                       'target_up', 'target_timing', 'day_return', 'timestamp',
                       'hour', 'minute', 'time', 'session', 'year']
        feature_cols = [c for c in df_features.columns if c not in exclude_cols]

        logger.info(f"Features: {len(feature_cols)}")

        # Clean data
        df_clean = df_features.dropna(subset=feature_cols + ['target_up', 'target_timing']).copy()
        logger.info(f"Clean samples: {len(df_clean)}")

    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 4: Prepare train/test split
    logger.info("\n" + "=" * 60)
    logger.info("STEP 4: PREPARE DATA SPLITS")
    logger.info("=" * 60)

    n_samples = len(df_clean)
    split_idx = int(n_samples * 0.8)
    purge_days = 5

    train_idx = list(range(0, split_idx - purge_days))
    test_idx = list(range(split_idx, n_samples))

    X = df_clean[feature_cols].values
    y_swing = df_clean['target_up'].values.astype(int)
    y_timing = df_clean['target_timing'].values.astype(int)

    X_train, X_test = X[train_idx], X[test_idx]
    y_swing_train, y_swing_test = y_swing[train_idx], y_swing[test_idx]
    y_timing_train, y_timing_test = y_timing[train_idx], y_timing[test_idx]

    logger.info(f"Train samples: {len(train_idx)}")
    logger.info(f"Test samples: {len(test_idx)}")

    # Prepare intraday data dict and daily OHLCV for temporal cascade
    df_1min_dict = None
    df_daily_ohlcv = None
    if not args.skip_cascade:
        logger.info("Preparing intraday data dictionary...")
        df_1min_dict = {}
        for date, group in df_1min.groupby(df_1min['timestamp'].dt.date):
            df_1min_dict[str(date)] = group.copy().reset_index(drop=True)
        logger.info(f"Intraday data for {len(df_1min_dict)} days")

        # Create daily OHLCV DataFrame for temporal cascade (it needs close, high, low, volume)
        logger.info("Creating daily OHLCV DataFrame for temporal cascade...")
        daily_ohlcv_records = []
        for date_str, day_df in df_1min_dict.items():
            # Filter to regular market hours for OHLCV aggregation
            if 'session' in day_df.columns:
                reg_df = day_df[day_df['session'] == 'regular']
            else:
                reg_df = day_df

            if len(reg_df) > 0:
                daily_ohlcv_records.append({
                    'date': date_str,
                    'open': reg_df.iloc[0]['open'],
                    'high': reg_df['high'].max(),
                    'low': reg_df['low'].min(),
                    'close': reg_df.iloc[-1]['close'],
                    'volume': reg_df['volume'].sum(),
                })

        df_daily_ohlcv = pd.DataFrame(daily_ohlcv_records)
        df_daily_ohlcv['date'] = pd.to_datetime(df_daily_ohlcv['date'])
        df_daily_ohlcv = df_daily_ohlcv.sort_values('date').reset_index(drop=True)
        logger.info(f"Daily OHLCV: {len(df_daily_ohlcv)} days")

    # Step 5: Train all temporal models
    logger.info("\n" + "=" * 60)
    logger.info("STEP 5: TRAIN TEMPORAL MODELS")
    logger.info("=" * 60)

    try:
        results = train_all_temporal_models(
            X_train=X_train,
            y_swing_train=y_swing_train,
            y_timing_train=y_timing_train,
            X_test=X_test,
            y_swing_test=y_swing_test,
            y_timing_test=y_timing_test,
            df_daily=df_daily_ohlcv,  # Pass daily OHLCV for temporal cascade features
            df_1min_dict=df_1min_dict,
            feature_cols=feature_cols,  # Save feature names with models for inference
        )

        logger.info("\n" + "=" * 60)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 60)

        # Print final summary
        registry = results['registry']
        print(registry.summary())

    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Step 6: Save status
    logger.info("\n" + "=" * 60)
    logger.info("STEP 6: UPDATE STATUS")
    logger.info("=" * 60)

    try:
        status_file = project_root / "logs" / "status.json"
        if status_file.exists():
            import json
            with open(status_file) as f:
                status = json.load(f)

            status['model']['last_train'] = datetime.now().isoformat()
            status['model']['accuracy'] = 0.0  # Will be updated during validation

            with open(status_file, 'w') as f:
                json.dump(status, f, indent=2)

            logger.info("Status updated")
    except Exception as e:
        logger.warning(f"Failed to update status: {e}")

    print("\n" + "=" * 70)
    print("RETRAINING COMPLETE")
    print("=" * 70)
    print(f"End time: {datetime.now()}")
    print(f"Models trained: {len(registry.models)}")
    print(f"Production models: {len(registry.get_production_models())}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
