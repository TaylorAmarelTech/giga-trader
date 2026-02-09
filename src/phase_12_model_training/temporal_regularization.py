"""
GIGA TRADER - Temporal Regularization for All Models
=====================================================
Applies temporal complexity and masking techniques to ALL models in the system,
not just the temporal cascade. This provides consistent anti-overfitting
across the entire pipeline.

TECHNIQUES:
-----------
1. TEMPORAL MASKING WRAPPER
   - Wraps any sklearn-compatible model
   - Applies random temporal masking during training
   - Creates implicit ensemble of robust models

2. TEMPORAL FEATURE AUGMENTATION
   - Augments features with temporal noise
   - Creates time-shifted copies of data
   - Forces models to be robust to timing variations

3. TEMPORAL CROSS-VALIDATION
   - Walk-forward CV with random time gaps
   - Simulates realistic deployment conditions

4. TEMPORAL DROPOUT
   - During training, randomly drop recent features
   - Forces model to work with incomplete information

INTEGRATION:
-----------
These techniques can be applied to:
  - Swing direction model
  - Timing model (low-before-high)
  - Entry/exit timing model
  - Position sizing model
  - Signal generator

Usage:
    from src.temporal_regularization import (
        TemporalMaskingWrapper,
        TemporalFeatureAugmenter,
        TemporalDropoutCV,
        apply_temporal_regularization,
    )

    # Wrap any model with temporal masking
    base_model = GradientBoostingClassifier()
    regularized_model = TemporalMaskingWrapper(
        base_model,
        mask_prob=0.2,
        n_ensemble=5,
    )
"""

import os
import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd
from scipy import stats

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import BaseCrossValidator
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

logger = logging.getLogger("TEMPORAL_REG")


# =============================================================================
# 1. TEMPORAL MASKING WRAPPER
# =============================================================================
class TemporalMaskingWrapper(BaseEstimator, ClassifierMixin):
    """
    Wraps any sklearn classifier with temporal masking regularization.

    During training:
      - Randomly mask features (set to 0 or mean)
      - Train multiple models with different masks
      - Ensemble predictions at inference

    This creates an implicit ensemble of models that are robust to
    missing or corrupted features, similar to dropout regularization.

    Parameters:
    -----------
    base_estimator : sklearn estimator
        The base model to wrap

    mask_prob : float
        Probability of masking each feature (default 0.2 = 20%)

    n_ensemble : int
        Number of models to train with different masks (default 5)

    mask_strategy : str
        'random' - Random independent masking
        'block' - Mask contiguous blocks of features
        'temporal' - Mask entire temporal slices (requires feature grouping)

    feature_groups : dict, optional
        Mapping of feature indices to temporal groups
        e.g., {0: 'T0', 1: 'T0', 2: 'T30', 3: 'T30', ...}
    """

    def __init__(
        self,
        base_estimator=None,
        mask_prob: float = 0.2,
        n_ensemble: int = 5,
        mask_strategy: str = 'random',
        feature_groups: Dict[int, str] = None,
        random_state: int = 42,
    ):
        # Use 'is None' instead of 'or' to avoid sklearn estimator bool evaluation issues
        if base_estimator is None:
            self.base_estimator = GradientBoostingClassifier(
                n_estimators=50, max_depth=3, min_samples_leaf=50
            )
        else:
            self.base_estimator = base_estimator
        self.mask_prob = mask_prob
        self.n_ensemble = n_ensemble
        self.mask_strategy = mask_strategy
        self.feature_groups = feature_groups
        self.random_state = random_state

        # Ensemble components
        self.estimators_: List[Any] = []
        self.scalers_: List[StandardScaler] = []
        self.feature_means_: np.ndarray = None

    def _generate_mask(self, n_samples: int, n_features: int, seed: int) -> np.ndarray:
        """Generate feature mask matrix."""
        rng = np.random.RandomState(seed)

        if self.mask_strategy == 'random':
            return rng.random((n_samples, n_features)) > self.mask_prob

        elif self.mask_strategy == 'block':
            mask = np.ones((n_samples, n_features), dtype=bool)
            for i in range(n_samples):
                if rng.random() < self.mask_prob * 3:
                    start = rng.randint(0, n_features - 1)
                    length = rng.randint(1, min(5, n_features - start + 1))
                    mask[i, start:start+length] = False
            return mask

        elif self.mask_strategy == 'temporal' and self.feature_groups:
            mask = np.ones((n_samples, n_features), dtype=bool)
            groups = list(set(self.feature_groups.values()))

            for i in range(n_samples):
                # Randomly mask entire temporal groups
                for group in groups:
                    if rng.random() < self.mask_prob:
                        group_indices = [
                            idx for idx, g in self.feature_groups.items()
                            if g == group
                        ]
                        mask[i, group_indices] = False
            return mask

        else:
            return rng.random((n_samples, n_features)) > self.mask_prob

    def _apply_mask(self, X: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply mask to features, replacing masked values with mean."""
        X_masked = X.copy()
        X_masked[~mask] = 0  # Or could use self.feature_means_

        return X_masked

    def fit(self, X, y, sample_weight=None):
        """
        Fit ensemble of masked models.
        """
        X = np.asarray(X)
        y = np.asarray(y)

        # Store feature means for masking
        self.feature_means_ = np.mean(X, axis=0)

        logger.info(f"Training {self.n_ensemble} masked models...")

        for i in range(self.n_ensemble):
            # Generate mask for this model
            mask = self._generate_mask(
                len(X), X.shape[1], self.random_state + i
            )

            # Apply mask
            X_masked = self._apply_mask(X, mask)

            # Scale
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X_masked)

            # Clone and fit
            estimator = clone(self.base_estimator)
            estimator.set_params(random_state=self.random_state + i)

            if sample_weight is not None:
                try:
                    estimator.fit(X_scaled, y, sample_weight=sample_weight)
                except TypeError:
                    estimator.fit(X_scaled, y)
            else:
                estimator.fit(X_scaled, y)

            self.estimators_.append(estimator)
            self.scalers_.append(scaler)

        return self

    def predict_proba(self, X):
        """
        Ensemble prediction from all masked models.
        """
        X = np.asarray(X)
        probas = []

        for estimator, scaler in zip(self.estimators_, self.scalers_):
            X_scaled = scaler.transform(X)
            proba = estimator.predict_proba(X_scaled)
            probas.append(proba)

        # Average probabilities
        return np.mean(probas, axis=0)

    def predict(self, X):
        """Predict class labels."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def get_prediction_uncertainty(self, X) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get mean prediction and uncertainty (std across ensemble).
        """
        X = np.asarray(X)
        probas = []

        for estimator, scaler in zip(self.estimators_, self.scalers_):
            X_scaled = scaler.transform(X)
            proba = estimator.predict_proba(X_scaled)[:, 1]
            probas.append(proba)

        probas = np.array(probas)
        mean_proba = np.mean(probas, axis=0)
        std_proba = np.std(probas, axis=0)

        return mean_proba, std_proba


# =============================================================================
# 2. TEMPORAL FEATURE AUGMENTER
# =============================================================================
class TemporalFeatureAugmenter:
    """
    Augments training data with temporal perturbations for robustness.

    AUGMENTATION TECHNIQUES:
    -----------------------
    1. TIME SHIFT: Shift features by small amounts (simulate timing errors)
    2. NOISE INJECTION: Add Gaussian noise to features
    3. DROPOUT: Randomly zero out features
    4. MIXUP: Interpolate between similar samples
    5. TIME STRETCH: Simulate faster/slower market conditions

    This forces models to be robust to:
      - Timing variations in data collection
      - Noise in real-time data
      - Missing values
      - Different market speeds
    """

    def __init__(
        self,
        augmentation_factor: float = 2.0,  # Generate 2x original samples
        noise_std: float = 0.05,            # 5% noise
        dropout_prob: float = 0.1,          # 10% feature dropout
        time_shift_max: int = 2,            # Max samples to shift
        use_mixup: bool = True,
        mixup_alpha: float = 0.2,
    ):
        self.augmentation_factor = augmentation_factor
        self.noise_std = noise_std
        self.dropout_prob = dropout_prob
        self.time_shift_max = time_shift_max
        self.use_mixup = use_mixup
        self.mixup_alpha = mixup_alpha

    def augment(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Augment data with temporal perturbations.

        Returns:
            X_augmented, y_augmented, weights_augmented
        """
        n_original = len(X)
        n_augmented = int(n_original * self.augmentation_factor)

        X_aug_list = [X]
        y_aug_list = [y]
        w_aug_list = [sample_weights if sample_weights is not None else np.ones(n_original)]

        n_to_generate = n_augmented - n_original
        rng = np.random.RandomState(42)

        for _ in range(n_to_generate):
            # Randomly select augmentation type
            aug_type = rng.choice(['noise', 'dropout', 'shift', 'mixup'])

            # Select random sample
            idx = rng.randint(0, n_original)
            x_new = X[idx].copy()
            y_new = y[idx]
            w_new = w_aug_list[0][idx] * 0.5  # Augmented samples get lower weight

            if aug_type == 'noise':
                # Add Gaussian noise
                noise = rng.normal(0, self.noise_std, x_new.shape)
                x_new = x_new + noise * np.abs(x_new)

            elif aug_type == 'dropout':
                # Random feature dropout
                mask = rng.random(x_new.shape) > self.dropout_prob
                x_new = x_new * mask

            elif aug_type == 'shift':
                # Circular shift features (simulate time offset)
                shift = rng.randint(-self.time_shift_max, self.time_shift_max + 1)
                x_new = np.roll(x_new, shift)

            elif aug_type == 'mixup' and self.use_mixup:
                # Mixup with another sample of same class
                same_class = np.where(y == y_new)[0]
                if len(same_class) > 1:
                    idx2 = rng.choice(same_class)
                    lam = rng.beta(self.mixup_alpha, self.mixup_alpha)
                    x_new = lam * x_new + (1 - lam) * X[idx2]

            X_aug_list.append(x_new.reshape(1, -1))
            y_aug_list.append(np.array([y_new]))
            w_aug_list.append(np.array([w_new]))

        X_augmented = np.vstack(X_aug_list)
        y_augmented = np.concatenate(y_aug_list)
        w_augmented = np.concatenate(w_aug_list)

        return X_augmented, y_augmented, w_augmented


# =============================================================================
# 3. TEMPORAL DROPOUT CROSS-VALIDATION
# =============================================================================
class TemporalDropoutCV(BaseCrossValidator):
    """
    Cross-validation with temporal dropout and gaps.

    Unlike standard time-series CV, this introduces:
      1. Random gaps between train and test (simulate delayed deployment)
      2. Random dropout of training samples (simulate missing data)
      3. Purging and embargo for autocorrelation

    This creates more realistic validation that accounts for:
      - Deployment delays
      - Data quality issues
      - Real-world gaps in trading
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_days: int = 5,
        embargo_days: int = 2,
        random_gap_days: int = 3,     # Random gap (0 to N days)
        train_dropout: float = 0.1,   # 10% of training samples dropped
        test_size: int = 50,
        min_train_size: int = 100,
    ):
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.random_gap_days = random_gap_days
        self.train_dropout = train_dropout
        self.test_size = test_size
        self.min_train_size = min_train_size

    def split(self, X, y=None, groups=None):
        """
        Generate train/test splits with temporal dropout.
        """
        n_samples = len(X)
        rng = np.random.RandomState(42)

        for split_idx in range(self.n_splits):
            # Random gap for this split
            random_gap = rng.randint(0, self.random_gap_days + 1)

            # Calculate split points
            total_embargo = self.purge_days + self.embargo_days + random_gap
            test_end = n_samples - split_idx * (self.test_size + self.embargo_days)
            test_start = test_end - self.test_size
            train_end = test_start - total_embargo

            if train_end < self.min_train_size:
                continue

            # Generate train indices with dropout
            train_idx = np.arange(0, train_end)
            if self.train_dropout > 0:
                dropout_mask = rng.random(len(train_idx)) > self.train_dropout
                train_idx = train_idx[dropout_mask]

            test_idx = np.arange(test_start, test_end)

            if len(train_idx) >= self.min_train_size and len(test_idx) > 0:
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits


# =============================================================================
# 4. TEMPORAL CONSISTENCY REGULARIZER
# =============================================================================
class TemporalConsistencyRegularizer:
    """
    Enforces temporal consistency in predictions.

    CONCEPT:
    --------
    Predictions should be temporally smooth - sudden large changes
    in prediction without corresponding market changes indicate overfitting.

    This regularizer:
      1. Penalizes large prediction changes between consecutive time steps
      2. Encourages agreement between nearby temporal slices
      3. Detects and flags suspicious prediction jumps

    Can be used during training (as loss component) or inference (as filter).
    """

    def __init__(
        self,
        max_prediction_change: float = 0.2,  # Max allowed change per time step
        smoothing_weight: float = 0.1,        # Weight for smoothing penalty
        agreement_threshold: float = 0.6,     # Min agreement between slices
    ):
        self.max_prediction_change = max_prediction_change
        self.smoothing_weight = smoothing_weight
        self.agreement_threshold = agreement_threshold

        self.prediction_history: List[float] = []

    def smooth_prediction(
        self,
        current_pred: float,
        n_history: int = 5,
    ) -> Tuple[float, bool]:
        """
        Smooth current prediction using history.

        Returns:
            smoothed_prediction, is_suspicious (large jump detected)
        """
        self.prediction_history.append(current_pred)

        if len(self.prediction_history) < 2:
            return current_pred, False

        # Get recent history
        history = self.prediction_history[-n_history:]

        # Check for suspicious jump
        prev_pred = history[-2]
        change = abs(current_pred - prev_pred)
        is_suspicious = change > self.max_prediction_change

        # Apply exponential smoothing
        if is_suspicious:
            # Dampen the jump
            alpha = 0.3  # Lower alpha = more smoothing
            smoothed = alpha * current_pred + (1 - alpha) * prev_pred
        else:
            # Normal smoothing
            alpha = 0.7
            smoothed = alpha * current_pred + (1 - alpha) * np.mean(history[:-1])

        return smoothed, is_suspicious

    def compute_consistency_loss(
        self,
        predictions: np.ndarray,  # (n_samples, n_time_steps)
    ) -> float:
        """
        Compute consistency loss for training.

        Penalizes:
          1. Large changes between consecutive predictions
          2. Disagreement between nearby temporal slices
        """
        n_samples, n_steps = predictions.shape

        # Temporal smoothness penalty
        diffs = np.diff(predictions, axis=1)
        smoothness_loss = np.mean(diffs ** 2)

        # Agreement penalty (variance across time steps)
        agreement_loss = np.mean(np.var(predictions, axis=1))

        total_loss = (
            self.smoothing_weight * smoothness_loss +
            (1 - self.smoothing_weight) * agreement_loss
        )

        return float(total_loss)

    def reset_history(self):
        """Reset prediction history."""
        self.prediction_history = []


# =============================================================================
# 5. APPLY TEMPORAL REGULARIZATION TO ANY MODEL
# =============================================================================
def apply_temporal_regularization(
    base_model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_groups: Dict[int, str] = None,
    use_masking: bool = True,
    use_augmentation: bool = True,
    use_dropout_cv: bool = True,
    mask_prob: float = 0.2,
    n_ensemble: int = 5,
    augmentation_factor: float = 1.5,
) -> Tuple[Any, Dict[str, Any]]:
    """
    Apply all temporal regularization techniques to a model.

    This is a convenience function that combines:
      1. Temporal masking wrapper
      2. Feature augmentation
      3. Temporal dropout CV for validation

    Args:
        base_model: sklearn-compatible model
        X_train: Training features
        y_train: Training labels
        feature_groups: Optional mapping of features to temporal groups
        use_masking: Apply temporal masking
        use_augmentation: Apply feature augmentation
        use_dropout_cv: Use temporal dropout CV
        mask_prob: Masking probability
        n_ensemble: Number of masked models
        augmentation_factor: Data augmentation multiplier

    Returns:
        trained_model, metrics_dict
    """
    logger.info("=" * 60)
    logger.info("APPLYING TEMPORAL REGULARIZATION")
    logger.info("=" * 60)

    metrics = {}

    # Step 1: Augment data
    if use_augmentation:
        logger.info("[1] Augmenting data...")
        augmenter = TemporalFeatureAugmenter(
            augmentation_factor=augmentation_factor,
        )
        X_aug, y_aug, w_aug = augmenter.augment(X_train, y_train)
        logger.info(f"  Original: {len(X_train)} samples")
        logger.info(f"  Augmented: {len(X_aug)} samples")
        metrics['n_original'] = len(X_train)
        metrics['n_augmented'] = len(X_aug)
    else:
        X_aug, y_aug, w_aug = X_train, y_train, None

    # Step 2: Wrap with masking
    if use_masking:
        logger.info("[2] Wrapping with temporal masking...")
        model = TemporalMaskingWrapper(
            base_estimator=base_model,
            mask_prob=mask_prob,
            n_ensemble=n_ensemble,
            mask_strategy='temporal' if feature_groups else 'random',
            feature_groups=feature_groups,
        )
    else:
        model = clone(base_model)

    # Step 3: Fit with temporal dropout CV for metrics
    if use_dropout_cv:
        logger.info("[3] Evaluating with temporal dropout CV...")
        cv = TemporalDropoutCV(
            n_splits=5,
            train_dropout=0.1,
        )

        cv_scores = []
        for train_idx, test_idx in cv.split(X_aug, y_aug):
            # Create temporary model for CV
            if use_masking:
                temp_model = TemporalMaskingWrapper(
                    base_estimator=clone(base_model),
                    mask_prob=mask_prob,
                    n_ensemble=3,  # Fewer for CV speed
                )
            else:
                temp_model = clone(base_model)

            temp_model.fit(X_aug[train_idx], y_aug[train_idx])

            if hasattr(temp_model, 'predict_proba'):
                y_pred = temp_model.predict_proba(X_aug[test_idx])[:, 1]
            else:
                y_pred = temp_model.predict(X_aug[test_idx])

            try:
                score = roc_auc_score(y_aug[test_idx], y_pred)
                cv_scores.append(score)
            except:
                pass

        metrics['cv_auc_mean'] = float(np.mean(cv_scores)) if cv_scores else 0.5
        metrics['cv_auc_std'] = float(np.std(cv_scores)) if cv_scores else 0.0
        logger.info(f"  CV AUC: {metrics['cv_auc_mean']:.4f} ± {metrics['cv_auc_std']:.4f}")

    # Step 4: Final fit on all data
    logger.info("[4] Final fit on all data...")
    if w_aug is not None:
        try:
            model.fit(X_aug, y_aug, sample_weight=w_aug)
        except TypeError:
            model.fit(X_aug, y_aug)
    else:
        model.fit(X_aug, y_aug)

    logger.info("Temporal regularization complete!")

    return model, metrics


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================
def create_temporally_regularized_swing_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[Any, Dict]:
    """Create swing direction model with temporal regularization."""
    base_model = GradientBoostingClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        min_samples_leaf=50,
        subsample=0.8,
    )

    return apply_temporal_regularization(
        base_model,
        X_train,
        y_train,
        use_masking=True,
        use_augmentation=True,
        mask_prob=0.2,
        n_ensemble=5,
    )


def create_temporally_regularized_timing_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
) -> Tuple[Any, Dict]:
    """Create timing model (low-before-high) with temporal regularization."""
    base_model = LogisticRegression(C=0.1, max_iter=1000)

    return apply_temporal_regularization(
        base_model,
        X_train,
        y_train,
        use_masking=True,
        use_augmentation=True,
        mask_prob=0.15,  # Lower mask prob for timing
        n_ensemble=5,
    )


def create_temporally_regularized_entry_exit_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_groups: Dict[int, str],
) -> Tuple[Any, Dict]:
    """Create entry/exit model with temporal group-aware masking."""
    base_model = GradientBoostingClassifier(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        min_samples_leaf=50,
    )

    return apply_temporal_regularization(
        base_model,
        X_train,
        y_train,
        feature_groups=feature_groups,
        use_masking=True,
        use_augmentation=True,
        mask_prob=0.2,
        n_ensemble=5,
    )


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    print("=" * 70)
    print("TEMPORAL REGULARIZATION MODULE")
    print("=" * 70)
    print("""
This module provides temporal regularization techniques that can be applied
to ANY model in the system:

1. TemporalMaskingWrapper
   - Wraps any sklearn model with temporal masking
   - Creates ensemble of robust models

2. TemporalFeatureAugmenter
   - Augments data with temporal perturbations
   - Noise, dropout, time shifts, mixup

3. TemporalDropoutCV
   - Cross-validation with temporal gaps and dropout
   - More realistic validation

4. TemporalConsistencyRegularizer
   - Enforces smooth predictions
   - Detects suspicious jumps

5. apply_temporal_regularization()
   - Convenience function to apply all techniques

Integration Example:
    from src.temporal_regularization import apply_temporal_regularization

    # Your base model
    base = GradientBoostingClassifier()

    # Apply temporal regularization
    model, metrics = apply_temporal_regularization(
        base, X_train, y_train,
        use_masking=True,
        use_augmentation=True,
    )
""")
