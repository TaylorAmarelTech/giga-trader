"""Phase: Data Preprocessing.

Core preprocessing functions live in src/train_robust_model.py.
This phase provides programmatic access to the preprocessing pipeline.

Key functions (imported lazily to avoid circular imports):
  - detect_and_handle_missing_bars: Fix missing bars in OHLCV data
  - DataManager.preprocess: Full preprocessing pipeline
"""


def detect_and_handle_missing_bars(df):
    """Re-export from train_robust_model.py."""
    from src.train_robust_model import detect_and_handle_missing_bars as _fn
    return _fn(df)


from src.phase_02_preprocessing.bar_resampler import (
    BarResampler,
    resolution_to_minutes,
    RESOLUTION_MAP,
)


__all__ = [
    "detect_and_handle_missing_bars",
    "BarResampler",
    "resolution_to_minutes",
    "RESOLUTION_MAP",
]
