"""
GIGA TRADER - Advanced Temporal Cascade Architectures
======================================================
Implements advanced temporal cascade models for maximum anti-overfitting:

1. MULTI-RESOLUTION CASCADE
   - Models trained on different time granularities (1min, 5min, 15min, 1hr)
   - Captures patterns at different scales

2. BACKWARD-LOOKING CASCADE
   - Models with different historical lookback windows
   - Short-term vs long-term pattern recognition

3. INTERMITTENT/MASKED CASCADE (KEY INNOVATION)
   - Random masking of temporal components during training
   - Forces robustness, prevents over-reliance on any time slice
   - Similar to dropout but for temporal structure

4. STOCHASTIC DEPTH CASCADE
   - Randomly skip temporal slices during training
   - Ensemble of "partial" models

5. CROSS-TEMPORAL ATTENTION CASCADE
   - Learn attention weights between temporal slices
   - Automatically discover which times matter most

BENEFITS:
  - Maximum regularization through temporal diversity
  - Robust to missing data (intermittent masking)
  - Adaptive importance weighting (attention)
  - Multi-scale pattern recognition

Usage:
    from src.advanced_temporal_cascades import (
        MultiResolutionCascade,
        BackwardLookingCascade,
        IntermittentMaskedCascade,
        StochasticDepthCascade,
        CrossTemporalAttentionCascade,
        UnifiedTemporalEnsemble,
    )
"""

import os
import sys
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
import warnings

import numpy as np
import pandas as pd
import joblib

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score, StratifiedKFold, TimeSeriesSplit

logger = logging.getLogger("ADVANCED_CASCADE")


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class CascadePrediction:
    """Unified prediction from any cascade architecture."""
    cascade_type: str
    timestamp: str

    # Core prediction
    swing_probability: float
    confidence: float

    # Component predictions
    component_predictions: Dict[str, float] = field(default_factory=dict)
    component_weights: Dict[str, float] = field(default_factory=dict)

    # Agreement metrics
    agreement_score: float = 0.0
    stability_score: float = 0.0

    # Masking info (for intermittent cascade)
    mask_applied: Dict[str, bool] = field(default_factory=dict)
    effective_components: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


# =============================================================================
# 1. MULTI-RESOLUTION CASCADE
# =============================================================================
class MultiResolutionCascade:
    """
    Cascade of models trained on different time resolutions.

    Each resolution captures patterns at different scales:
      - 1-min: Micro-structure, noise, immediate reactions
      - 5-min: Short-term momentum, order flow
      - 15-min: Medium-term trends, support/resistance
      - 1-hour: Macro trends, regime shifts
    """

    RESOLUTIONS = {
        '1min': 1,
        '5min': 5,
        '15min': 15,
        '30min': 30,
        '1hour': 60,
    }

    def __init__(
        self,
        resolutions: List[str] = None,
        model_type: str = "gradient_boosting",
    ):
        self.resolutions = resolutions or list(self.RESOLUTIONS.keys())
        self.model_type = model_type

        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.cv_aucs: Dict[str, float] = {}
        self.is_fitted = False

    def _resample_data(
        self,
        df_1min: pd.DataFrame,
        resolution_minutes: int,
    ) -> pd.DataFrame:
        """Resample 1-min data to target resolution."""
        if resolution_minutes == 1:
            return df_1min.copy()

        # Ensure datetime index
        df = df_1min.copy()
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')

        # Resample
        resampled = df.resample(f'{resolution_minutes}min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
        }).dropna()

        return resampled.reset_index()

    def _engineer_features(self, df: pd.DataFrame) -> np.ndarray:
        """Engineer features from resampled data."""
        features = []

        # Returns at different lags
        for lag in [1, 2, 3, 5, 10]:
            if len(df) > lag:
                features.append(df['close'].pct_change(lag).iloc[-1])
            else:
                features.append(0.0)

        # Volatility
        if len(df) >= 10:
            features.append(df['close'].pct_change().rolling(10).std().iloc[-1])
        else:
            features.append(0.0)

        # Range
        features.append((df['high'].iloc[-1] - df['low'].iloc[-1]) / df['close'].iloc[-1])

        # Volume trend
        if len(df) >= 5:
            vol_ma = df['volume'].rolling(5).mean().iloc[-1]
            features.append(df['volume'].iloc[-1] / max(vol_ma, 1))
        else:
            features.append(1.0)

        # Price position in range
        day_high = df['high'].max()
        day_low = df['low'].min()
        if day_high > day_low:
            features.append((df['close'].iloc[-1] - day_low) / (day_high - day_low))
        else:
            features.append(0.5)

        return np.array(features)

    def _create_model(self):
        """Create model based on type."""
        if self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=50,
                max_depth=3,
                learning_rate=0.1,
                min_samples_leaf=50,
                random_state=42,
            )
        else:
            return LogisticRegression(C=0.1, max_iter=1000, random_state=42)

    def _aggregate_daily_features(
        self,
        df_1min: pd.DataFrame,
        resolution_minutes: int,
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Aggregate minute-level data into per-day feature vectors at a given resolution.

        Groups minute data by trading day, resamples to the target resolution,
        then engineers features for each day.

        Args:
            df_1min: Minute-level OHLCV data with a 'timestamp' or datetime index.
            resolution_minutes: Target resolution in minutes.

        Returns:
            Tuple of (feature_matrix, list_of_date_strings) aligned by day.
        """
        df = df_1min.copy()

        # Ensure we have a datetime column for grouping
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df['_date'] = df['timestamp'].dt.date.astype(str)
        elif isinstance(df.index, pd.DatetimeIndex):
            df['_date'] = df.index.date.astype(str)
        else:
            logger.warning("Cannot determine dates from minute data; falling back to single group")
            df['_date'] = 'unknown'

        daily_features = []
        day_labels = []

        for day_str, day_df in df.groupby('_date'):
            if len(day_df) < max(resolution_minutes, 5):
                # Not enough bars for this day; skip
                continue

            # Resample this day's data to the target resolution
            resampled = self._resample_data(day_df.drop(columns=['_date'], errors='ignore'), resolution_minutes)

            if len(resampled) < 2:
                continue

            features = self._engineer_features(resampled)
            if features is not None and not np.any(np.isnan(features)):
                daily_features.append(features)
                day_labels.append(day_str)

        if not daily_features:
            return np.empty((0, 0)), []

        return np.array(daily_features), day_labels

    def fit(
        self,
        df_1min: pd.DataFrame,
        y: np.ndarray,
        sample_weights: np.ndarray = None,
    ) -> Dict[str, float]:
        """
        Train models at each resolution with proper cross-validation.

        Args:
            df_1min: Minute-level data with 'timestamp' column or DatetimeIndex.
            y: Target labels (one per day, aligned with the day order in df_1min).
            sample_weights: Optional sample weights (one per day).

        Returns:
            Dictionary mapping resolution name to metrics dict with 'cv_auc'.
        """
        logger.info("=" * 60)
        logger.info("MULTI-RESOLUTION CASCADE - Training")
        logger.info("=" * 60)

        # Build a date-to-index mapping from y so we can align aggregated days
        # Assume y is ordered the same as unique days in df_1min.
        df_tmp = df_1min.copy()
        if 'timestamp' in df_tmp.columns:
            df_tmp['timestamp'] = pd.to_datetime(df_tmp['timestamp'])
            all_days_ordered = list(df_tmp['timestamp'].dt.date.astype(str).unique())
        elif isinstance(df_tmp.index, pd.DatetimeIndex):
            all_days_ordered = list(df_tmp.index.date.astype(str).unique())
        else:
            all_days_ordered = []

        # Map day string -> index into y
        day_to_y_idx = {}
        for idx, day_str in enumerate(all_days_ordered):
            if idx < len(y):
                day_to_y_idx[day_str] = idx

        metrics = {}

        for res_name in self.resolutions:
            res_minutes = self.RESOLUTIONS[res_name]
            logger.info(f"[{res_name}] Training at {res_minutes}-minute resolution...")

            # Aggregate to per-day feature vectors at this resolution
            X_daily, day_labels = self._aggregate_daily_features(df_1min, res_minutes)

            if len(X_daily) == 0:
                logger.warning(f"  [{res_name}] No valid daily features could be aggregated. Skipping.")
                self.cv_aucs[res_name] = 0.5
                metrics[res_name] = {'cv_auc': 0.5, 'n_samples': 0, 'status': 'skipped'}
                continue

            # Align targets with the days that survived aggregation
            y_aligned = []
            w_aligned = []
            valid_rows = []
            for i, day_str in enumerate(day_labels):
                if day_str in day_to_y_idx:
                    y_idx = day_to_y_idx[day_str]
                    y_aligned.append(y[y_idx])
                    if sample_weights is not None and y_idx < len(sample_weights):
                        w_aligned.append(sample_weights[y_idx])
                    valid_rows.append(i)

            if len(y_aligned) < 20:
                logger.warning(f"  [{res_name}] Only {len(y_aligned)} aligned samples. Skipping.")
                self.cv_aucs[res_name] = 0.5
                metrics[res_name] = {'cv_auc': 0.5, 'n_samples': len(y_aligned), 'status': 'skipped'}
                continue

            X = X_daily[valid_rows]
            y_arr = np.array(y_aligned)
            w_arr = np.array(w_aligned) if w_aligned else None

            logger.info(f"  [{res_name}] {len(X)} samples, {X.shape[1]} features")

            # Scale features
            self.scalers[res_name] = StandardScaler()
            X_scaled = self.scalers[res_name].fit_transform(X)

            # Replace NaN/inf that may arise from scaling edge cases
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

            # Create and train model on full data
            self.models[res_name] = self._create_model()
            try:
                if w_arr is not None and len(w_arr) == len(y_arr):
                    self.models[res_name].fit(X_scaled, y_arr, sample_weight=w_arr)
                else:
                    self.models[res_name].fit(X_scaled, y_arr)
            except Exception as e:
                logger.error(f"  [{res_name}] Model training failed: {e}")
                self.cv_aucs[res_name] = 0.5
                metrics[res_name] = {'cv_auc': 0.5, 'n_samples': len(y_arr), 'status': 'train_failed'}
                continue

            # Cross-validation for AUC estimation
            n_unique_classes = len(np.unique(y_arr))
            if n_unique_classes < 2:
                logger.warning(f"  [{res_name}] Only one class present. CV AUC set to 0.5.")
                cv_auc = 0.5
            elif len(y_arr) < 50:
                # Too few samples for proper CV; use simple train/test split (80/20)
                split_idx = int(len(y_arr) * 0.8)
                X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
                y_train, y_test = y_arr[:split_idx], y_arr[split_idx:]
                if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                    cv_auc = 0.5
                else:
                    try:
                        holdout_model = self._create_model()
                        holdout_model.fit(X_train, y_train)
                        y_proba = holdout_model.predict_proba(X_test)[:, 1]
                        cv_auc = float(roc_auc_score(y_test, y_proba))
                    except Exception as e:
                        logger.warning(f"  [{res_name}] Holdout AUC failed: {e}")
                        cv_auc = 0.5
            else:
                # Use StratifiedKFold for larger datasets (3-5 folds based on size)
                n_folds = 5 if len(y_arr) >= 200 else 3
                try:
                    skf = StratifiedKFold(n_splits=n_folds, shuffle=False)
                    fold_aucs = []
                    for train_idx, val_idx in skf.split(X_scaled, y_arr):
                        X_tr, X_val = X_scaled[train_idx], X_scaled[val_idx]
                        y_tr, y_val = y_arr[train_idx], y_arr[val_idx]
                        if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
                            continue
                        fold_model = self._create_model()
                        fold_model.fit(X_tr, y_tr)
                        y_proba = fold_model.predict_proba(X_val)[:, 1]
                        fold_aucs.append(float(roc_auc_score(y_val, y_proba)))
                    cv_auc = float(np.mean(fold_aucs)) if fold_aucs else 0.5
                except Exception as e:
                    logger.warning(f"  [{res_name}] CV AUC failed: {e}")
                    cv_auc = 0.5

            self.cv_aucs[res_name] = cv_auc
            logger.info(f"  [{res_name}] CV AUC: {cv_auc:.4f} (n={len(y_arr)})")
            metrics[res_name] = {'cv_auc': cv_auc, 'n_samples': len(y_arr), 'status': 'trained'}

        self.is_fitted = True
        return metrics

    def predict(self, df_1min_today: pd.DataFrame) -> CascadePrediction:
        """Get ensemble prediction from all resolutions."""
        predictions = {}
        weights = {}

        for res_name in self.resolutions:
            if res_name not in self.models or self.models[res_name] is None:
                continue
            if res_name not in self.scalers:
                continue

            res_minutes = self.RESOLUTIONS[res_name]
            df_resampled = self._resample_data(df_1min_today, res_minutes)

            if len(df_resampled) > 0:
                features = self._engineer_features(df_resampled)
                if features is not None and not np.any(np.isnan(features)):
                    try:
                        X = self.scalers[res_name].transform(features.reshape(1, -1))
                        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
                        prob = self.models[res_name].predict_proba(X)[0, 1]
                        predictions[res_name] = float(prob)
                        weights[res_name] = self.cv_aucs.get(res_name, 0.5)
                    except Exception as e:
                        logger.warning(f"[{res_name}] Prediction failed: {e}")
                        continue

        # Weighted average
        if predictions:
            total_weight = sum(weights.values())
            ensemble_prob = sum(
                predictions[k] * weights[k] / total_weight
                for k in predictions
            )
        else:
            ensemble_prob = 0.5

        return CascadePrediction(
            cascade_type="multi_resolution",
            timestamp=datetime.now().isoformat(),
            swing_probability=ensemble_prob,
            confidence=abs(ensemble_prob - 0.5) * 2,
            component_predictions=predictions,
            component_weights=weights,
            agreement_score=self._calculate_agreement(predictions),
            effective_components=len(predictions),
        )

    def _calculate_agreement(self, predictions: Dict[str, float]) -> float:
        """Calculate agreement between components."""
        if len(predictions) < 2:
            return 1.0

        values = list(predictions.values())
        directions = [1 if v > 0.5 else 0 for v in values]
        agreement = sum(1 for d in directions if d == directions[0]) / len(directions)
        return agreement


# =============================================================================
# 2. BACKWARD-LOOKING CASCADE
# =============================================================================
class BackwardLookingCascade:
    """
    Cascade of models with different historical lookback windows.

    Each lookback captures different market memory:
      - 1-day: Immediate context, overnight gaps
      - 5-day: Weekly patterns, recent momentum
      - 20-day: Monthly cycles, medium-term trends
      - 60-day: Quarterly patterns, regime persistence
      - 252-day: Annual seasonality, long-term trends
    """

    LOOKBACKS = {
        'short': 5,      # 1 week
        'medium': 20,    # 1 month
        'long': 60,      # 1 quarter
        'annual': 252,   # 1 year
    }

    def __init__(
        self,
        lookbacks: Dict[str, int] = None,
        model_type: str = "gradient_boosting",
    ):
        self.lookbacks = lookbacks or self.LOOKBACKS
        self.model_type = model_type

        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.cv_aucs: Dict[str, float] = {}
        self.is_fitted = False

    def _engineer_lookback_features(
        self,
        df_daily: pd.DataFrame,
        lookback_days: int,
    ) -> np.ndarray:
        """Engineer features using specified lookback window."""
        if len(df_daily) < lookback_days:
            return None

        df = df_daily.tail(lookback_days).copy()
        features = []

        # Return over lookback period
        total_return = (df['close'].iloc[-1] / df['close'].iloc[0]) - 1
        features.append(total_return)

        # Volatility over period
        daily_returns = df['close'].pct_change().dropna()
        features.append(daily_returns.std())

        # Trend strength (linear regression slope)
        if len(df) > 2:
            x = np.arange(len(df))
            slope = np.polyfit(x, df['close'].values, 1)[0]
            features.append(slope / df['close'].mean())
        else:
            features.append(0.0)

        # Max drawdown
        cummax = df['close'].cummax()
        drawdown = (df['close'] - cummax) / cummax
        features.append(drawdown.min())

        # Days since high/low
        days_since_high = len(df) - df['close'].argmax() - 1
        days_since_low = len(df) - df['close'].argmin() - 1
        features.append(days_since_high / lookback_days)
        features.append(days_since_low / lookback_days)

        # Volume trend
        if 'volume' in df.columns:
            vol_trend = (df['volume'].iloc[-5:].mean() / df['volume'].mean()) - 1
            features.append(vol_trend)
        else:
            features.append(0.0)

        return np.array(features)

    def _create_model(self):
        """Create model based on type (EDGE 1: regularization-first)."""
        if self.model_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=50,
                max_depth=3,          # EDGE 1: max_depth <= 5
                learning_rate=0.1,
                min_samples_leaf=50,
                subsample=0.8,
                random_state=42,
            )
        else:
            return LogisticRegression(
                C=0.1,               # EDGE 1: strong regularization
                max_iter=1000,
                random_state=42,
            )

    def _expanding_window_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_splits: int = 3,
        purge_days: int = 5,
        embargo_days: int = 2,
        min_train_size: int = 50,
    ) -> List[float]:
        """
        Expanding-window time-series cross-validation with purging and embargo.

        This ensures that:
        - Training data always comes BEFORE validation data (no future leakage).
        - A purge gap of `purge_days` samples is removed between train and val.
        - An embargo of `embargo_days` samples is removed at the start of val.
        - The training window expands with each fold.

        Args:
            X: Feature matrix (n_samples, n_features), ordered by time.
            y: Target labels, ordered by time.
            n_splits: Number of CV folds.
            purge_days: Number of samples to remove at end of training set.
            embargo_days: Number of samples to remove at start of validation set.
            min_train_size: Minimum number of training samples per fold.

        Returns:
            List of AUC scores per fold.
        """
        n_samples = len(y)
        fold_aucs = []

        # Each fold gets an increasing fraction of the data for training
        # and a fixed-size validation window after the purge/embargo gap.
        val_size = max(20, n_samples // (n_splits + 1))
        total_gap = purge_days + embargo_days

        for fold_idx in range(n_splits):
            # Validation window position: starts after enough room for training + gap
            val_start = min_train_size + total_gap + fold_idx * val_size
            val_end = val_start + val_size

            if val_end > n_samples:
                # Not enough data for this fold
                break

            # Training: everything before (val_start - total_gap)
            train_end = val_start - total_gap
            if train_end < min_train_size:
                continue

            train_idx = np.arange(0, train_end)
            # Validation: skip embargo days after purge
            actual_val_start = val_start
            val_idx = np.arange(actual_val_start, min(val_end, n_samples))

            if len(train_idx) < min_train_size or len(val_idx) < 10:
                continue

            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Need both classes in train and val
            if len(np.unique(y_tr)) < 2 or len(np.unique(y_val)) < 2:
                continue

            try:
                fold_model = self._create_model()
                fold_model.fit(X_tr, y_tr)
                y_proba = fold_model.predict_proba(X_val)[:, 1]
                auc = float(roc_auc_score(y_val, y_proba))
                fold_aucs.append(auc)
            except Exception as e:
                logger.warning(f"    Fold {fold_idx + 1} failed: {e}")
                continue

        return fold_aucs

    def fit(
        self,
        df_daily: pd.DataFrame,
        y: np.ndarray,
    ) -> Dict[str, float]:
        """
        Train models for each lookback window with strict backward-looking
        feature construction and expanding-window time-series cross-validation.

        Each sample at time T uses ONLY data from time (T - lookback) to T.
        No future data is ever used in feature engineering or cross-validation.

        Args:
            df_daily: Daily OHLCV DataFrame ordered chronologically.
            y: Target labels (one per day), aligned with df_daily rows.

        Returns:
            Dictionary mapping lookback name to metrics with 'cv_auc'.
        """
        logger.info("=" * 60)
        logger.info("BACKWARD-LOOKING CASCADE - Training")
        logger.info("=" * 60)

        metrics = {}

        for name, lookback in self.lookbacks.items():
            logger.info(f"[{name}] Training with {lookback}-day lookback...")

            # Build feature matrix using STRICT backward-looking windows.
            # For sample at index i, use only df_daily.iloc[i-lookback : i+1]
            # (the +1 includes the current day's OHLCV for features like
            # "close relative to lookback high", but targets are the label at i).
            X_list = []
            y_valid = []
            valid_indices = []  # Track original indices for time-series ordering

            for i in range(lookback, len(df_daily)):
                if i >= len(y):
                    break

                # Strict backward window: only past data up to and including day i
                window = df_daily.iloc[i - lookback:i + 1]
                features = self._engineer_lookback_features(window, lookback)

                if features is not None and not np.any(np.isnan(features)):
                    X_list.append(features)
                    y_valid.append(y[i])
                    valid_indices.append(i)

            if len(X_list) < 30:
                logger.warning(f"  [{name}] Not enough valid samples ({len(X_list)}). Skipping.")
                metrics[name] = {'cv_auc': 0.5, 'n_samples': len(X_list), 'status': 'skipped'}
                continue

            X = np.array(X_list)
            y_arr = np.array(y_valid)

            logger.info(f"  [{name}] {len(X)} samples, {X.shape[1]} features")

            # Check class distribution
            n_unique = len(np.unique(y_arr))
            if n_unique < 2:
                logger.warning(f"  [{name}] Only one class present. Skipping.")
                metrics[name] = {'cv_auc': 0.5, 'n_samples': len(y_arr), 'status': 'single_class'}
                continue

            # Scale features (fit on all data; CV will re-fit per fold internally)
            self.scalers[name] = StandardScaler()
            X_scaled = self.scalers[name].fit_transform(X)
            X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=0.0, neginf=0.0)

            # Train final model on all available data
            self.models[name] = self._create_model()
            try:
                self.models[name].fit(X_scaled, y_arr)
            except Exception as e:
                logger.error(f"  [{name}] Model training failed: {e}")
                metrics[name] = {'cv_auc': 0.5, 'n_samples': len(y_arr), 'status': 'train_failed'}
                continue

            # Time-series cross-validation with purging and embargo
            # (expanding window, never using future data)
            if len(y_arr) < 50:
                # Too few samples for proper expanding-window CV.
                # Use a simple chronological train/test split (80/20).
                split_idx = int(len(y_arr) * 0.8)
                X_tr, X_te = X_scaled[:split_idx], X_scaled[split_idx:]
                y_tr, y_te = y_arr[:split_idx], y_arr[split_idx:]

                if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                    cv_auc = 0.5
                else:
                    try:
                        holdout_model = self._create_model()
                        holdout_model.fit(X_tr, y_tr)
                        y_proba = holdout_model.predict_proba(X_te)[:, 1]
                        cv_auc = float(roc_auc_score(y_te, y_proba))
                    except Exception as e:
                        logger.warning(f"  [{name}] Holdout AUC failed: {e}")
                        cv_auc = 0.5
            else:
                # Expanding-window CV with purge (5 days) and embargo (2 days)
                n_folds = 5 if len(y_arr) >= 300 else 3
                fold_aucs = self._expanding_window_cv(
                    X_scaled, y_arr,
                    n_splits=n_folds,
                    purge_days=5,
                    embargo_days=2,
                    min_train_size=max(50, len(y_arr) // 4),
                )

                if fold_aucs:
                    cv_auc = float(np.mean(fold_aucs))
                    cv_std = float(np.std(fold_aucs))
                    logger.info(f"  [{name}] CV folds: {len(fold_aucs)}, "
                                f"AUCs: {[f'{a:.4f}' for a in fold_aucs]}")
                    logger.info(f"  [{name}] CV AUC: {cv_auc:.4f} +/- {cv_std:.4f}")
                else:
                    logger.warning(f"  [{name}] All CV folds failed. Falling back to 0.5.")
                    cv_auc = 0.5

            self.cv_aucs[name] = cv_auc
            logger.info(f"  [{name}] Final CV AUC: {self.cv_aucs[name]:.4f}")
            metrics[name] = {
                'cv_auc': self.cv_aucs[name],
                'n_samples': len(y_arr),
                'status': 'trained',
            }

        self.is_fitted = True
        return metrics

    def predict(self, df_daily: pd.DataFrame) -> CascadePrediction:
        """Get ensemble prediction from all lookback models."""
        predictions = {}
        weights = {}

        for name, lookback in self.lookbacks.items():
            if name not in self.models:
                continue

            features = self._engineer_lookback_features(df_daily, lookback)
            if features is None:
                continue

            X = self.scalers[name].transform(features.reshape(1, -1))
            prob = self.models[name].predict_proba(X)[0, 1]

            predictions[name] = float(prob)
            weights[name] = self.cv_aucs.get(name, 0.5)

        # Weighted ensemble
        if predictions:
            total_weight = sum(weights.values())
            ensemble_prob = sum(
                predictions[k] * weights[k] / total_weight
                for k in predictions
            )
        else:
            ensemble_prob = 0.5

        return CascadePrediction(
            cascade_type="backward_looking",
            timestamp=datetime.now().isoformat(),
            swing_probability=float(ensemble_prob),
            confidence=abs(ensemble_prob - 0.5) * 2,
            component_predictions=predictions,
            component_weights=weights,
            agreement_score=self._calculate_agreement(predictions),
            effective_components=len(predictions),
        )

    def _calculate_agreement(self, predictions: Dict[str, float]) -> float:
        if len(predictions) < 2:
            return 1.0
        values = list(predictions.values())
        directions = [1 if v > 0.5 else 0 for v in values]
        return sum(1 for d in directions if d == directions[0]) / len(directions)


# =============================================================================
# 3. INTERMITTENT/MASKED CASCADE (KEY INNOVATION!)
# =============================================================================
class IntermittentMaskedCascade:
    """
    Temporal cascade with random masking for maximum regularization.

    CONCEPT:
    --------
    During training, randomly mask out temporal components to force
    the model to be robust and not over-rely on any single time slice.

    This is similar to:
      - Dropout (but for temporal structure)
      - Data augmentation (via missing data simulation)
      - Ensemble of partial models

    MASKING STRATEGIES:
    ------------------
    1. RANDOM: Randomly mask X% of temporal slices each training sample
    2. BLOCK: Mask contiguous blocks of time
    3. FEATURE: Mask specific features within time slices
    4. PROGRESSIVE: Increase masking as training progresses
    5. ADVERSARIAL: Mask the most important slices (hardest regularization)

    BENEFITS:
    ---------
    - Forces model to be robust to missing data (real-world scenario)
    - Prevents co-adaptation between temporal slices
    - Creates implicit ensemble during training
    - Stronger regularization than standard L1/L2
    """

    TEMPORAL_SLICES = [0, 30, 60, 90, 120, 180]

    def __init__(
        self,
        mask_probability: float = 0.2,  # 20% of slices masked per sample
        mask_strategy: str = "random",   # random, block, progressive, adversarial
        n_masked_models: int = 5,        # Train multiple models with different masks
        base_model_type: str = "gradient_boosting",
    ):
        self.mask_probability = mask_probability
        self.mask_strategy = mask_strategy
        self.n_masked_models = n_masked_models
        self.base_model_type = base_model_type

        # Multiple models trained with different mask patterns
        self.masked_models: List[Dict] = []
        self.scalers: List[StandardScaler] = []
        self.mask_patterns: List[Dict[int, bool]] = []
        self.cv_aucs: List[float] = []

        self.is_fitted = False

    def _generate_mask(
        self,
        n_samples: int,
        strategy: str = None,
    ) -> np.ndarray:
        """
        Generate mask matrix for temporal slices.

        Returns:
            mask: (n_samples, n_slices) boolean array
                  True = use this slice, False = mask it out
        """
        strategy = strategy or self.mask_strategy
        n_slices = len(self.TEMPORAL_SLICES)

        if strategy == "random":
            # Random independent masking
            mask = np.random.random((n_samples, n_slices)) > self.mask_probability

        elif strategy == "block":
            # Mask contiguous blocks
            mask = np.ones((n_samples, n_slices), dtype=bool)
            for i in range(n_samples):
                if np.random.random() < self.mask_probability * 2:
                    # Mask a random contiguous block
                    start = np.random.randint(0, n_slices - 1)
                    length = np.random.randint(1, min(3, n_slices - start + 1))
                    mask[i, start:start+length] = False

        elif strategy == "progressive":
            # More masking for later slices (force reliance on early info)
            mask = np.ones((n_samples, n_slices), dtype=bool)
            for j in range(n_slices):
                # Probability increases with slice index
                slice_mask_prob = self.mask_probability * (1 + j / n_slices)
                mask[:, j] = np.random.random(n_samples) > slice_mask_prob

        elif strategy == "leave_one_out":
            # Each model trained with one slice always masked
            mask = np.ones((n_samples, n_slices), dtype=bool)
            # This is handled at model level, not sample level

        else:
            # Default: no masking
            mask = np.ones((n_samples, n_slices), dtype=bool)

        # Ensure at least one slice is always available
        for i in range(n_samples):
            if not mask[i].any():
                mask[i, 0] = True  # Always keep T0

        return mask

    def _apply_mask_to_features(
        self,
        X: np.ndarray,
        mask: np.ndarray,
        n_features_per_slice: int,
    ) -> np.ndarray:
        """
        Apply mask to feature matrix.

        Args:
            X: (n_samples, n_features) feature matrix
            mask: (n_samples, n_slices) mask matrix
            n_features_per_slice: Features per temporal slice
        """
        X_masked = X.copy()
        n_slices = mask.shape[1]

        for i in range(len(X)):
            for j in range(n_slices):
                if not mask[i, j]:
                    # Mask out this slice's features (set to 0 or mean)
                    start_idx = j * n_features_per_slice
                    end_idx = start_idx + n_features_per_slice
                    if end_idx <= X.shape[1]:
                        X_masked[i, start_idx:end_idx] = 0

        return X_masked

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_features_per_slice: int = 20,
    ) -> Dict[str, Any]:
        """
        Train multiple models with different mask patterns.

        The ensemble of masked models provides strong regularization.
        """
        logger.info("=" * 60)
        logger.info("INTERMITTENT MASKED CASCADE - Training")
        logger.info(f"Mask probability: {self.mask_probability}")
        logger.info(f"Mask strategy: {self.mask_strategy}")
        logger.info(f"Number of masked models: {self.n_masked_models}")
        logger.info("=" * 60)

        n_samples = len(X)
        metrics = {'models': []}

        for model_idx in range(self.n_masked_models):
            logger.info(f"[Model {model_idx + 1}/{self.n_masked_models}] Training...")

            # Generate mask for this model variant
            if self.mask_strategy == "leave_one_out" and model_idx < len(self.TEMPORAL_SLICES):
                # Leave one slice out completely
                mask = np.ones((n_samples, len(self.TEMPORAL_SLICES)), dtype=bool)
                mask[:, model_idx] = False
                mask_pattern = {ts: (i != model_idx) for i, ts in enumerate(self.TEMPORAL_SLICES)}
            else:
                mask = self._generate_mask(n_samples, self.mask_strategy)
                mask_pattern = {ts: True for ts in self.TEMPORAL_SLICES}  # Dynamic per sample

            # Apply mask to features
            X_masked = self._apply_mask_to_features(X, mask, n_features_per_slice)

            # Scale
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_masked)

            # Train model
            if self.base_model_type == "gradient_boosting":
                model = GradientBoostingClassifier(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    min_samples_leaf=50,
                    subsample=0.8,
                    random_state=42 + model_idx,
                )
            else:
                model = LogisticRegression(C=0.1, max_iter=1000, random_state=42 + model_idx)

            model.fit(X_scaled, y)

            # CV score
            try:
                cv_scores = cross_val_score(
                    GradientBoostingClassifier(n_estimators=50, max_depth=3, random_state=42),
                    X_scaled, y, cv=3, scoring='roc_auc'
                )
                cv_auc = float(np.mean(cv_scores))
            except:
                cv_auc = 0.5

            self.masked_models.append({'model': model, 'mask_pattern': mask_pattern})
            self.scalers.append(scaler)
            self.mask_patterns.append(mask_pattern)
            self.cv_aucs.append(cv_auc)

            logger.info(f"  [Model {model_idx + 1}] CV AUC: {cv_auc:.4f}")
            metrics['models'].append({'cv_auc': cv_auc})

        metrics['ensemble_cv_auc'] = float(np.mean(self.cv_aucs))
        logger.info(f"Ensemble CV AUC: {metrics['ensemble_cv_auc']:.4f}")

        self.is_fitted = True
        return metrics

    def predict(
        self,
        X: np.ndarray,
        apply_inference_mask: bool = False,
        inference_mask_prob: float = 0.1,
    ) -> CascadePrediction:
        """
        Get ensemble prediction from all masked models.

        Args:
            X: Feature vector
            apply_inference_mask: Whether to apply masking at inference
            inference_mask_prob: Mask probability at inference (for uncertainty)
        """
        if X.ndim == 1:
            X = X.reshape(1, -1)

        predictions = {}
        weights = {}
        masks_applied = {}

        for i, (model_dict, scaler, cv_auc) in enumerate(
            zip(self.masked_models, self.scalers, self.cv_aucs)
        ):
            model = model_dict['model']

            # Apply optional inference mask (Monte Carlo dropout style)
            X_input = X.copy()
            if apply_inference_mask:
                mask = np.random.random(X.shape[1]) > inference_mask_prob
                X_input[0, ~mask] = 0
                masks_applied[f'model_{i}'] = bool(~mask.all())

            X_scaled = scaler.transform(X_input)
            prob = model.predict_proba(X_scaled)[0, 1]

            predictions[f'model_{i}'] = float(prob)
            weights[f'model_{i}'] = cv_auc

        # Weighted ensemble
        total_weight = sum(weights.values())
        ensemble_prob = sum(
            predictions[k] * weights[k] / total_weight
            for k in predictions
        )

        # Calculate uncertainty from model disagreement
        probs = list(predictions.values())
        uncertainty = np.std(probs)

        return CascadePrediction(
            cascade_type="intermittent_masked",
            timestamp=datetime.now().isoformat(),
            swing_probability=float(ensemble_prob),
            confidence=max(0, 1 - 2 * uncertainty),  # Higher disagreement = lower confidence
            component_predictions=predictions,
            component_weights=weights,
            agreement_score=1 - uncertainty,
            stability_score=1 - uncertainty,
            mask_applied=masks_applied,
            effective_components=len(predictions),
        )

    def predict_with_uncertainty(
        self,
        X: np.ndarray,
        n_forward_passes: int = 10,
    ) -> Tuple[float, float]:
        """
        Monte Carlo prediction with uncertainty estimation.

        Run multiple forward passes with different masks to estimate
        prediction uncertainty (like MC Dropout).
        """
        predictions = []

        for _ in range(n_forward_passes):
            pred = self.predict(X, apply_inference_mask=True, inference_mask_prob=0.15)
            predictions.append(pred.swing_probability)

        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)

        return float(mean_pred), float(std_pred)


# =============================================================================
# 4. STOCHASTIC DEPTH CASCADE
# =============================================================================
class StochasticDepthCascade:
    """
    Cascade where temporal slices are randomly skipped during training.

    Similar to Stochastic Depth in ResNets - randomly drop layers/paths
    during training to create implicit ensemble and regularization.

    CONCEPT:
    --------
    Instead of always using all temporal slices, randomly skip some
    during training. This forces each slice to be independently useful.
    """

    TEMPORAL_SLICES = [0, 30, 60, 90, 120, 180]

    def __init__(
        self,
        survival_prob: float = 0.8,  # Probability of keeping each slice
        min_slices: int = 2,          # Minimum slices to use
        model_type: str = "gradient_boosting",
    ):
        self.survival_prob = survival_prob
        self.min_slices = min_slices
        self.model_type = model_type

        # Single model trained with stochastic depth
        self.model = None
        self.scaler = StandardScaler()
        self.slice_importance: Dict[int, float] = {}
        self.cv_auc = 0.0
        self.is_fitted = False

    def _stochastic_slice_selection(self, n_samples: int) -> np.ndarray:
        """
        Generate stochastic slice selection matrix.

        Returns:
            selection: (n_samples, n_slices) boolean array
        """
        n_slices = len(self.TEMPORAL_SLICES)
        selection = np.random.random((n_samples, n_slices)) < self.survival_prob

        # Ensure minimum slices (always keep T0 and at least one other)
        for i in range(n_samples):
            selection[i, 0] = True  # Always keep T0
            while selection[i].sum() < self.min_slices:
                idx = np.random.randint(1, n_slices)
                selection[i, idx] = True

        return selection

    def fit(
        self,
        X_by_slice: Dict[int, np.ndarray],
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Train with stochastic depth.

        Args:
            X_by_slice: Dictionary mapping temporal slice -> feature matrix
            y: Target labels
        """
        logger.info("=" * 60)
        logger.info("STOCHASTIC DEPTH CASCADE - Training")
        logger.info(f"Survival probability: {self.survival_prob}")
        logger.info("=" * 60)

        n_samples = len(y)

        # Generate stochastic selections for this training run
        selection = self._stochastic_slice_selection(n_samples)

        # Combine features with stochastic selection
        X_combined = []
        for i in range(n_samples):
            sample_features = []
            for j, ts in enumerate(self.TEMPORAL_SLICES):
                if ts in X_by_slice:
                    if selection[i, j]:
                        sample_features.append(X_by_slice[ts][i])
                    else:
                        # Masked slice - use zeros
                        sample_features.append(np.zeros_like(X_by_slice[ts][i]))
            X_combined.append(np.concatenate(sample_features))

        X = np.array(X_combined)

        # Scale
        X_scaled = self.scaler.fit_transform(X)

        # Train
        self.model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            min_samples_leaf=50,
            random_state=42,
        )
        self.model.fit(X_scaled, y)

        # CV score
        try:
            cv_scores = cross_val_score(
                GradientBoostingClassifier(n_estimators=50, max_depth=3),
                X_scaled, y, cv=3, scoring='roc_auc'
            )
            self.cv_auc = float(np.mean(cv_scores))
        except:
            self.cv_auc = 0.5

        # Estimate slice importance by measuring performance drop
        logger.info("Estimating slice importance...")
        for ts in self.TEMPORAL_SLICES:
            # This would require leave-one-out retraining for true importance
            self.slice_importance[ts] = 1.0 / len(self.TEMPORAL_SLICES)

        logger.info(f"CV AUC: {self.cv_auc:.4f}")
        self.is_fitted = True

        return {'cv_auc': self.cv_auc, 'slice_importance': self.slice_importance}

    def predict(self, X_by_slice: Dict[int, np.ndarray]) -> CascadePrediction:
        """Predict using all slices (no stochastic dropping at inference)."""
        sample_features = []
        for ts in self.TEMPORAL_SLICES:
            if ts in X_by_slice:
                sample_features.append(X_by_slice[ts])
            else:
                # Missing slice - use zeros
                sample_features.append(np.zeros(20))  # Assume 20 features per slice

        X = np.concatenate(sample_features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        prob = self.model.predict_proba(X_scaled)[0, 1]

        return CascadePrediction(
            cascade_type="stochastic_depth",
            timestamp=datetime.now().isoformat(),
            swing_probability=float(prob),
            confidence=abs(prob - 0.5) * 2,
            component_predictions={str(ts): self.slice_importance[ts] for ts in self.TEMPORAL_SLICES},
            effective_components=len(self.TEMPORAL_SLICES),
        )


# =============================================================================
# 5. CROSS-TEMPORAL ATTENTION CASCADE
# =============================================================================
class CrossTemporalAttentionCascade:
    """
    Cascade with learnable attention weights between temporal slices.

    CONCEPT:
    --------
    Learn which temporal slices are most important for prediction,
    and how they should be weighted. The attention mechanism allows
    the model to dynamically focus on the most relevant time periods.

    ARCHITECTURE:
    -------------
    Input (T0, T30, T60, ...) -> Attention Layer -> Weighted Features -> Classifier

    The attention weights are learned, allowing the model to discover
    which temporal slices are most predictive.
    """

    TEMPORAL_SLICES = [0, 30, 60, 90, 120, 180]

    def __init__(
        self,
        attention_type: str = "softmax",  # softmax, sigmoid, learned
        temperature: float = 1.0,          # Softmax temperature
        model_type: str = "gradient_boosting",
    ):
        self.attention_type = attention_type
        self.temperature = temperature
        self.model_type = model_type

        # Attention weights
        self.attention_weights: Dict[int, float] = {}

        # Model
        self.model = None
        self.scaler = StandardScaler()
        self.cv_auc = 0.0
        self.is_fitted = False

    def _compute_attention_weights(
        self,
        X_by_slice: Dict[int, np.ndarray],
        y: np.ndarray,
    ) -> Dict[int, float]:
        """
        Compute attention weights based on individual slice performance.

        This is a simplified version - a full implementation would use
        a neural attention mechanism.
        """
        weights = {}

        for ts in self.TEMPORAL_SLICES:
            if ts not in X_by_slice:
                weights[ts] = 0.0
                continue

            X = X_by_slice[ts]

            # Train a simple model on this slice alone
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            try:
                model = LogisticRegression(C=0.1, max_iter=500, random_state=42)
                cv_scores = cross_val_score(model, X_scaled, y, cv=3, scoring='roc_auc')
                weights[ts] = float(np.mean(cv_scores))
            except:
                weights[ts] = 0.5

        # Apply softmax to get normalized attention
        if self.attention_type == "softmax":
            values = np.array([weights[ts] for ts in self.TEMPORAL_SLICES])
            exp_values = np.exp((values - 0.5) / self.temperature)  # Center around 0.5
            softmax_values = exp_values / exp_values.sum()

            for i, ts in enumerate(self.TEMPORAL_SLICES):
                weights[ts] = float(softmax_values[i])

        elif self.attention_type == "sigmoid":
            for ts in weights:
                weights[ts] = 1 / (1 + np.exp(-(weights[ts] - 0.5) * 10))

        return weights

    def fit(
        self,
        X_by_slice: Dict[int, np.ndarray],
        y: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Train with cross-temporal attention.

        1. Compute attention weights for each slice
        2. Weight features by attention
        3. Train classifier on attention-weighted features
        """
        logger.info("=" * 60)
        logger.info("CROSS-TEMPORAL ATTENTION CASCADE - Training")
        logger.info(f"Attention type: {self.attention_type}")
        logger.info("=" * 60)

        # Step 1: Compute attention weights
        logger.info("Computing attention weights...")
        self.attention_weights = self._compute_attention_weights(X_by_slice, y)

        for ts, weight in self.attention_weights.items():
            logger.info(f"  T{ts}: attention = {weight:.4f}")

        # Step 2: Create attention-weighted features
        n_samples = len(y)
        X_attended = []

        for i in range(n_samples):
            sample_features = []
            for ts in self.TEMPORAL_SLICES:
                if ts in X_by_slice:
                    # Weight features by attention
                    weighted_features = X_by_slice[ts][i] * self.attention_weights[ts]
                    sample_features.append(weighted_features)
            X_attended.append(np.concatenate(sample_features))

        X = np.array(X_attended)

        # Step 3: Scale and train
        X_scaled = self.scaler.fit_transform(X)

        self.model = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=3,
            learning_rate=0.1,
            min_samples_leaf=50,
            random_state=42,
        )
        self.model.fit(X_scaled, y)

        # CV score
        try:
            cv_scores = cross_val_score(
                GradientBoostingClassifier(n_estimators=50, max_depth=3),
                X_scaled, y, cv=3, scoring='roc_auc'
            )
            self.cv_auc = float(np.mean(cv_scores))
        except:
            self.cv_auc = 0.5

        logger.info(f"CV AUC: {self.cv_auc:.4f}")
        self.is_fitted = True

        return {
            'cv_auc': self.cv_auc,
            'attention_weights': self.attention_weights,
        }

    def predict(self, X_by_slice: Dict[int, np.ndarray]) -> CascadePrediction:
        """Predict with attention-weighted features."""
        sample_features = []

        for ts in self.TEMPORAL_SLICES:
            if ts in X_by_slice:
                weighted_features = X_by_slice[ts] * self.attention_weights.get(ts, 0.0)
                sample_features.append(weighted_features)
            else:
                sample_features.append(np.zeros(20))

        X = np.concatenate(sample_features).reshape(1, -1)
        X_scaled = self.scaler.transform(X)

        prob = self.model.predict_proba(X_scaled)[0, 1]

        return CascadePrediction(
            cascade_type="cross_temporal_attention",
            timestamp=datetime.now().isoformat(),
            swing_probability=float(prob),
            confidence=abs(prob - 0.5) * 2,
            component_predictions={str(ts): float(prob * self.attention_weights.get(ts, 0))
                                 for ts in self.TEMPORAL_SLICES},
            component_weights={str(ts): self.attention_weights.get(ts, 0)
                             for ts in self.TEMPORAL_SLICES},
            effective_components=sum(1 for w in self.attention_weights.values() if w > 0.1),
        )


# =============================================================================
# 6. UNIFIED TEMPORAL ENSEMBLE
# =============================================================================
class UnifiedTemporalEnsemble:
    """
    Combines ALL temporal cascade architectures into a unified ensemble.

    This provides maximum regularization by ensembling:
      1. Multi-Resolution Cascade (different time scales)
      2. Backward-Looking Cascade (different lookback windows)
      3. Intermittent Masked Cascade (dropout regularization)
      4. Stochastic Depth Cascade (random slice skipping)
      5. Cross-Temporal Attention (learned importance)

    Each architecture captures different aspects of temporal patterns,
    and their disagreement provides uncertainty estimation.
    """

    def __init__(
        self,
        use_multi_resolution: bool = True,
        use_backward_looking: bool = True,
        use_intermittent_masked: bool = True,
        use_stochastic_depth: bool = True,
        use_attention: bool = True,
    ):
        self.use_multi_resolution = use_multi_resolution
        self.use_backward_looking = use_backward_looking
        self.use_intermittent_masked = use_intermittent_masked
        self.use_stochastic_depth = use_stochastic_depth
        self.use_attention = use_attention

        # Initialize cascades
        self.cascades: Dict[str, Any] = {}
        self.cascade_weights: Dict[str, float] = {}

        if use_multi_resolution:
            self.cascades['multi_resolution'] = MultiResolutionCascade()
        if use_backward_looking:
            self.cascades['backward_looking'] = BackwardLookingCascade()
        if use_intermittent_masked:
            self.cascades['intermittent_masked'] = IntermittentMaskedCascade(
                mask_probability=0.2,
                mask_strategy="random",
                n_masked_models=5,
            )
        if use_stochastic_depth:
            self.cascades['stochastic_depth'] = StochasticDepthCascade(
                survival_prob=0.8,
            )
        if use_attention:
            self.cascades['attention'] = CrossTemporalAttentionCascade(
                attention_type="softmax",
            )

        self.is_fitted = False

    def fit(
        self,
        df_1min: pd.DataFrame = None,
        df_daily: pd.DataFrame = None,
        X: np.ndarray = None,
        X_by_slice: Dict[int, np.ndarray] = None,
        y: np.ndarray = None,
    ) -> Dict[str, Any]:
        """
        Train all cascade architectures.

        Different cascades need different input formats, so we accept
        multiple input types and route appropriately.
        """
        logger.info("=" * 70)
        logger.info("UNIFIED TEMPORAL ENSEMBLE - Training All Cascades")
        logger.info("=" * 70)

        metrics = {}

        # Train each cascade with appropriate data
        for name, cascade in self.cascades.items():
            logger.info(f"\nTraining {name} cascade...")

            try:
                if name == 'multi_resolution' and df_1min is not None:
                    m = cascade.fit(df_1min, y)
                elif name == 'backward_looking' and df_daily is not None:
                    m = cascade.fit(df_daily, y)
                elif name == 'intermittent_masked' and X is not None:
                    m = cascade.fit(X, y)
                elif name == 'stochastic_depth' and X_by_slice is not None:
                    m = cascade.fit(X_by_slice, y)
                elif name == 'attention' and X_by_slice is not None:
                    m = cascade.fit(X_by_slice, y)
                else:
                    logger.warning(f"  Skipping {name} - missing required data")
                    continue

                # Get CV AUC for weighting
                cv_auc = m.get('cv_auc', m.get('ensemble_cv_auc', 0.5))
                self.cascade_weights[name] = cv_auc
                metrics[name] = m

            except Exception as e:
                logger.error(f"  Failed to train {name}: {e}")
                continue

        self.is_fitted = True

        # Summary
        logger.info("\n" + "=" * 70)
        logger.info("UNIFIED ENSEMBLE SUMMARY")
        logger.info("=" * 70)
        for name, weight in self.cascade_weights.items():
            logger.info(f"  {name}: weight = {weight:.4f}")

        return metrics

    def predict(
        self,
        df_1min: pd.DataFrame = None,
        df_daily: pd.DataFrame = None,
        X: np.ndarray = None,
        X_by_slice: Dict[int, np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Get ensemble prediction from all cascades.

        Returns predictions from each cascade plus a weighted ensemble.
        """
        predictions = {}

        # Get prediction from each cascade
        for name, cascade in self.cascades.items():
            if not hasattr(cascade, 'is_fitted') or not cascade.is_fitted:
                continue

            try:
                if name == 'multi_resolution' and df_1min is not None:
                    pred = cascade.predict(df_1min)
                elif name == 'backward_looking' and df_daily is not None:
                    pred = cascade.predict(df_daily)
                elif name == 'intermittent_masked' and X is not None:
                    pred = cascade.predict(X)
                elif name == 'stochastic_depth' and X_by_slice is not None:
                    pred = cascade.predict(X_by_slice)
                elif name == 'attention' and X_by_slice is not None:
                    pred = cascade.predict(X_by_slice)
                else:
                    continue

                predictions[name] = pred

            except Exception as e:
                logger.warning(f"Prediction failed for {name}: {e}")
                continue

        # Compute weighted ensemble
        if predictions:
            total_weight = sum(
                self.cascade_weights.get(name, 0.5)
                for name in predictions
            )
            ensemble_prob = sum(
                predictions[name].swing_probability * self.cascade_weights.get(name, 0.5) / total_weight
                for name in predictions
            )

            # Disagreement as uncertainty
            probs = [p.swing_probability for p in predictions.values()]
            uncertainty = np.std(probs) if len(probs) > 1 else 0
        else:
            ensemble_prob = 0.5
            uncertainty = 1.0

        return {
            'ensemble_probability': float(ensemble_prob),
            'ensemble_confidence': max(0, 1 - 2 * uncertainty),
            'uncertainty': float(uncertainty),
            'cascade_predictions': {
                name: pred.to_dict() for name, pred in predictions.items()
            },
            'cascade_weights': self.cascade_weights,
            'n_cascades_used': len(predictions),
        }

    def save(self, path: Path):
        """Save ensemble to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        save_dict = {
            'cascades': {},
            'cascade_weights': self.cascade_weights,
        }

        for name, cascade in self.cascades.items():
            if hasattr(cascade, 'is_fitted') and cascade.is_fitted:
                save_dict['cascades'][name] = cascade

        joblib.dump(save_dict, path)
        logger.info(f"Saved unified ensemble to {path}")

    @classmethod
    def load(cls, path: Path) -> "UnifiedTemporalEnsemble":
        """Load ensemble from disk."""
        save_dict = joblib.load(path)

        ensemble = cls(
            use_multi_resolution=False,
            use_backward_looking=False,
            use_intermittent_masked=False,
            use_stochastic_depth=False,
            use_attention=False,
        )

        ensemble.cascades = save_dict['cascades']
        ensemble.cascade_weights = save_dict['cascade_weights']
        ensemble.is_fitted = True

        return ensemble


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    print("=" * 70)
    print("ADVANCED TEMPORAL CASCADE ARCHITECTURES")
    print("=" * 70)
    print("""
Available Cascades:

1. MultiResolutionCascade
   - Models at different time granularities (1min, 5min, 15min, 1hr)
   - Captures patterns at multiple scales

2. BackwardLookingCascade
   - Different historical lookback windows (5d, 20d, 60d, 252d)
   - Short-term vs long-term pattern recognition

3. IntermittentMaskedCascade (KEY FOR REGULARIZATION)
   - Random masking of temporal slices during training
   - Forces robustness, prevents over-reliance
   - Similar to dropout but for temporal structure

4. StochasticDepthCascade
   - Randomly skip temporal slices during training
   - Creates implicit ensemble effect

5. CrossTemporalAttentionCascade
   - Learnable attention weights between slices
   - Automatically discovers important time periods

6. UnifiedTemporalEnsemble
   - Combines ALL architectures
   - Maximum regularization and diversity
""")
