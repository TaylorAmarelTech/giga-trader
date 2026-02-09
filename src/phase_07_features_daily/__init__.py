"""Phase: Daily Features.

Core feature engineering functions live in src/train_robust_model.py.
This phase provides programmatic access to feature engineering.

Key functions (imported lazily to avoid circular imports):
  - engineer_all_features: Generate 200+ features from 1-min OHLCV
  - add_rolling_features: Add rolling window features
  - create_soft_targets: Generate soft target labels (EDGE 4)
"""


def engineer_all_features(df, swing_threshold=0.003):
    """Re-export from train_robust_model.py."""
    from src.train_robust_model import engineer_all_features as _fn
    return _fn(df, swing_threshold=swing_threshold)


def add_rolling_features(df):
    """Re-export from train_robust_model.py."""
    from src.train_robust_model import add_rolling_features as _fn
    return _fn(df)


def create_soft_targets(df, threshold=0.003):
    """Re-export from train_robust_model.py."""
    from src.train_robust_model import create_soft_targets as _fn
    return _fn(df, threshold=threshold)


__all__ = [
    "engineer_all_features",
    "add_rolling_features",
    "create_soft_targets",
]
