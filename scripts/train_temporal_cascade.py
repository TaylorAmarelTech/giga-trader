"""
GIGA TRADER - Train Temporal Cascade Models
============================================
Manual script to train temporal cascade ensemble.

Usage:
    python scripts/train_temporal_cascade.py

This will:
  1. Load 3 years of SPY minute data
  2. Train temporal cascade models (T0, T30, T60, T90, T120, T180)
  3. Save the ensemble to models/temporal_cascade/
"""

import os
import sys
from pathlib import Path
import logging

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger("TRAIN_CASCADE")


def main():
    """Train temporal cascade models."""
    print("=" * 70)
    print("TEMPORAL CASCADE TRAINING")
    print("=" * 70)

    try:
        # Import trainer
        from src.temporal_cascade_trainer import train_temporal_cascade
        from src.data_manager import get_spy_data

        # Load data
        logger.info("Loading 3 years of SPY minute data...")
        df_1min = get_spy_data(years=3)

        if df_1min is None or len(df_1min) == 0:
            logger.error("No data available")
            return 1

        logger.info(f"Loaded {len(df_1min):,} bars")

        # Train
        logger.info("Starting temporal cascade training...")
        result = train_temporal_cascade(
            df_1min=df_1min,
            save_model=True,
        )

        # Report results
        print("\n" + result.summary())

        if result.success:
            logger.info("Training successful!")
            return 0
        else:
            logger.warning("Training completed but did not meet thresholds")
            return 1

    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error("Make sure all dependencies are installed")
        return 1
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
