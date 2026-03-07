"""
GIGA TRADER - Test-Time Adaptation (TTA) for Production Inference
==================================================================
Wraps any sklearn-compatible classifier to adapt predictions at inference
time using recent unlabeled data.  The adaptation is conservative and
bounded so that it cannot flip a prediction or introduce large shifts.

Three complementary mechanisms are applied:

1. **Feature distribution alignment** -- detects column-level drift
   between training data and recent test observations and applies an
   affine correction to the base model's probability output.

2. **Entropy minimisation** -- nudges already-confident predictions
   slightly further from the 0.5 boundary, sharpening the decision.

3. **Exponential moving average (EMA)** -- smoothly blends the drift
   correction signal over time so that a single outlier sample cannot
   dominate the adaptation.

All corrections are clamped by *max_shift* so the adapter can never
move a probability by more than that amount, and can never flip a
prediction across the 0.5 boundary.

Constraints (per EDGE 1 -- Regularization First):
  - Adaptation rate is intentionally low (default 0.01)
  - Maximum shift is bounded (default 0.05)
  - No learnable parameters -- pure statistics
  - No deep learning dependencies

Usage:
    from src.phase_14_robustness.test_time_adapter import TestTimeAdapter

    adapter = TestTimeAdapter(base_model=fitted_clf)
    adapter.fit_reference(X_train)

    # At inference time:
    proba = adapter.predict_proba(X_new)
    adapter.update(X_new)

    diag = adapter.get_adaptation_diagnostics()
"""

import logging
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_EPS = 1e-8  # Numerical floor to avoid division by zero


class TestTimeAdapter:
    """Adapt a fitted sklearn classifier at test time using unlabeled data.

    Parameters
    ----------
    base_model : sklearn-compatible classifier
        A fitted model that exposes ``predict_proba`` (and optionally
        ``predict``).  The adapter does **not** modify the model's
        parameters -- it post-processes the probability output.
    adaptation_rate : float
        EMA blending coefficient for the drift correction signal.
        Lower values make adaptation slower and more conservative
        (default 0.01).
    n_recent : int
        Number of most-recent samples retained in the sliding window
        for running-statistics computation (default 50).
    entropy_weight : float
        Strength of entropy-minimisation sharpening.  0 disables
        sharpening entirely (default 0.1).
    max_shift : float
        Hard upper bound on the absolute probability shift that the
        adapter may apply to any single prediction (default 0.05).
    """

    def __init__(
        self,
        base_model: Any,
        adaptation_rate: float = 0.01,
        n_recent: int = 50,
        entropy_weight: float = 0.1,
        max_shift: float = 0.05,
    ) -> None:
        if adaptation_rate < 0.0 or adaptation_rate > 1.0:
            raise ValueError(
                f"adaptation_rate must be in [0, 1], got {adaptation_rate}"
            )
        if n_recent < 1:
            raise ValueError(f"n_recent must be >= 1, got {n_recent}")
        if entropy_weight < 0.0:
            raise ValueError(
                f"entropy_weight must be >= 0, got {entropy_weight}"
            )
        if max_shift < 0.0 or max_shift > 0.5:
            raise ValueError(
                f"max_shift must be in [0, 0.5], got {max_shift}"
            )

        self.base_model = base_model
        self.adaptation_rate = adaptation_rate
        self.n_recent = n_recent
        self.entropy_weight = entropy_weight
        self.max_shift = max_shift

        # Reference distribution (from training data)
        self._ref_mean: Optional[np.ndarray] = None  # (n_features,)
        self._ref_std: Optional[np.ndarray] = None  # (n_features,)
        self._ref_n: int = 0

        # Running statistics (Welford's online algorithm)
        self._run_n: int = 0
        self._run_mean: Optional[np.ndarray] = None  # (n_features,)
        self._run_m2: Optional[np.ndarray] = None  # (n_features,)

        # EMA drift correction signal (scalar shift applied to proba)
        self._ema_shift: float = 0.0

        # Sliding window of recent samples for diagnostics / drift test
        self._recent_buffer: Optional[np.ndarray] = None  # (<=n_recent, n_features)

        # Counters
        self._n_updates: int = 0
        self._n_predictions: int = 0
        self._total_abs_shift: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit_reference(self, X_train: np.ndarray) -> "TestTimeAdapter":
        """Store training-distribution statistics for drift detection.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training feature matrix used to fit ``base_model``.

        Returns
        -------
        self
        """
        X = np.asarray(X_train, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        self._ref_mean = np.mean(X, axis=0)
        self._ref_std = np.std(X, axis=0)
        self._ref_std = np.where(self._ref_std < _EPS, 1.0, self._ref_std)
        self._ref_n = X.shape[0]

        logger.info(
            "TestTimeAdapter: reference fitted on %d samples, %d features",
            X.shape[0], X.shape[1],
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Produce adapted probability predictions.

        Steps:
        1. Get base-model probabilities.
        2. Compute per-feature drift statistics (if reference is set).
        3. Derive a scalar correction from aggregate drift.
        4. Apply entropy-minimisation sharpening.
        5. Clamp total shift to ``max_shift`` and prevent class flips.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples, 2)
            Adapted class probabilities ``[P(class=0), P(class=1)]``.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # --- Step 1: base predictions ---
        base_proba = self._base_predict_proba(X)  # (n, 2)
        p1 = base_proba[:, 1].copy()  # P(class=1)

        # --- Step 2-3: drift correction ---
        drift_shift = self._compute_drift_shift(X)

        # --- Step 4: entropy minimisation ---
        entropy_shift = self._entropy_minimise(p1)

        # --- Combine shifts ---
        total_shift = drift_shift + self.entropy_weight * entropy_shift

        # --- Step 5: clamp and apply ---
        adapted_p1 = self._apply_shift(p1, total_shift)

        self._n_predictions += X.shape[0]
        self._total_abs_shift += float(np.sum(np.abs(adapted_p1 - p1)))

        out = np.column_stack([1.0 - adapted_p1, adapted_p1])
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hard class labels (0 or 1) from adapted probabilities.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        labels : ndarray of shape (n_samples,)
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)

    def update(
        self,
        X: np.ndarray,
        y_pseudo: Optional[np.ndarray] = None,
    ) -> None:
        """Incorporate new observations into running statistics.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Recent unlabeled (or pseudo-labeled) observations.
        y_pseudo : array-like or None
            Optional pseudo-labels from confident predictions.  Currently
            reserved for future use; the adapter only uses feature
            statistics from *X*.
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Welford online update
        for row in X:
            self._welford_update(row)

        # Update sliding window
        if self._recent_buffer is None:
            self._recent_buffer = X.copy()
        else:
            self._recent_buffer = np.concatenate(
                [self._recent_buffer, X], axis=0
            )
        # Keep only the most recent n_recent rows
        if self._recent_buffer.shape[0] > self.n_recent:
            self._recent_buffer = self._recent_buffer[-self.n_recent:]

        self._n_updates += 1

    def get_adaptation_diagnostics(self) -> Dict[str, Any]:
        """Return a summary of the adapter's current state.

        Returns
        -------
        dict
            Keys include ``n_predictions``, ``n_updates``,
            ``ema_shift``, ``mean_abs_shift``, ``drift_detected``,
            ``reference_fitted``, ``recent_buffer_size``,
            ``running_n``.
        """
        mean_abs = (
            self._total_abs_shift / max(self._n_predictions, 1)
        )
        drift = self._detect_drift()

        return {
            "n_predictions": self._n_predictions,
            "n_updates": self._n_updates,
            "ema_shift": self._ema_shift,
            "mean_abs_shift": mean_abs,
            "drift_detected": drift,
            "reference_fitted": self._ref_mean is not None,
            "recent_buffer_size": (
                self._recent_buffer.shape[0]
                if self._recent_buffer is not None
                else 0
            ),
            "running_n": self._run_n,
        }

    def reset(self) -> None:
        """Clear all adaptation state, keeping the base model and reference."""
        self._run_n = 0
        self._run_mean = None
        self._run_m2 = None
        self._ema_shift = 0.0
        self._recent_buffer = None
        self._n_updates = 0
        self._n_predictions = 0
        self._total_abs_shift = 0.0
        logger.info("TestTimeAdapter: adaptation state reset")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _base_predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get base model probabilities as (n, 2) array."""
        raw = self.base_model.predict_proba(X)
        raw = np.asarray(raw, dtype=np.float64)
        if raw.ndim == 1:
            # Model returned 1-D array -- treat as P(class=1)
            return np.column_stack([1.0 - raw, raw])
        if raw.shape[1] == 1:
            return np.column_stack([1.0 - raw[:, 0], raw[:, 0]])
        return raw

    def _compute_drift_shift(self, X: np.ndarray) -> float:
        """Compute a scalar probability correction from feature drift.

        Drift magnitude is estimated as the average normalised mean
        difference across features:  ``|mean_recent - mean_ref| / std_ref``.

        The correction direction is derived from the sign of the
        aggregate mean shift (positive drift => nudge probability up,
        under the assumption that the model under-estimates when
        feature means increase -- this is a simple linear proxy).

        The correction is blended into an EMA so that it changes
        smoothly across batches.
        """
        if self._ref_mean is None:
            return 0.0
        if self._recent_buffer is None or self._recent_buffer.shape[0] < 2:
            # Use the current batch itself when no history exists
            if X.shape[0] < 2:
                return self._ema_shift
            batch_mean = np.mean(X, axis=0)
        else:
            batch_mean = np.mean(self._recent_buffer, axis=0)

        # Normalised mean difference per feature
        diff = (batch_mean - self._ref_mean) / self._ref_std
        # Aggregate: mean signed diff (preserves direction)
        aggregate_signed = float(np.mean(diff))
        # Magnitude: mean |diff|
        aggregate_mag = float(np.mean(np.abs(diff)))

        # Only correct if drift is meaningful (> 0.1 std on average)
        if aggregate_mag < 0.1:
            raw_shift = 0.0
        else:
            # Scale: 1 std of average drift => adaptation_rate shift
            raw_shift = self.adaptation_rate * aggregate_signed

        # Smooth with EMA
        alpha = self.adaptation_rate
        self._ema_shift = (1.0 - alpha) * self._ema_shift + alpha * raw_shift

        return self._ema_shift

    def _entropy_minimise(self, p1: np.ndarray) -> np.ndarray:
        """Compute per-sample sharpening shift via entropy minimisation.

        If a prediction is already confident (``p1 > 0.6`` or ``p1 < 0.4``),
        nudge it further from 0.5 proportionally to its distance from
        the boundary.  Uncertain predictions (near 0.5) are left alone.

        Parameters
        ----------
        p1 : ndarray of shape (n_samples,)
            P(class=1) from the base model.

        Returns
        -------
        shift : ndarray of shape (n_samples,)
            Signed per-sample shift (positive = push toward 1,
            negative = push toward 0).
        """
        shift = np.zeros_like(p1)
        # Confident toward class 1
        mask_high = p1 > 0.6
        shift[mask_high] = (p1[mask_high] - 0.5) * self.adaptation_rate

        # Confident toward class 0
        mask_low = p1 < 0.4
        shift[mask_low] = (p1[mask_low] - 0.5) * self.adaptation_rate

        return shift

    def _apply_shift(
        self, p1: np.ndarray, shift: np.ndarray | float
    ) -> np.ndarray:
        """Apply a bounded shift to probabilities without flipping labels.

        Parameters
        ----------
        p1 : ndarray of shape (n_samples,)
            Original P(class=1).
        shift : float or ndarray
            Signed correction to add.

        Returns
        -------
        adapted : ndarray of shape (n_samples,)
            Clamped, adapted P(class=1).
        """
        shift = np.asarray(shift, dtype=np.float64)

        # Clamp magnitude
        shift = np.clip(shift, -self.max_shift, self.max_shift)

        adapted = p1 + shift

        # Prevent class flips: if original >= 0.5 keep >= 0.5 and vice versa
        flip_up = (p1 >= 0.5) & (adapted < 0.5)
        flip_down = (p1 < 0.5) & (adapted >= 0.5)
        adapted[flip_up] = 0.5
        adapted[flip_down] = 0.5 - _EPS

        # Clamp to valid probability range
        adapted = np.clip(adapted, 0.0, 1.0)

        return adapted

    def _detect_drift(self) -> bool:
        """Return True if significant feature drift is detected.

        Uses a simple proxy: average |z-score of mean shift| > 0.5
        across features.
        """
        if self._ref_mean is None:
            return False
        if self._recent_buffer is None or self._recent_buffer.shape[0] < 2:
            return False

        recent_mean = np.mean(self._recent_buffer, axis=0)
        z = np.abs((recent_mean - self._ref_mean) / self._ref_std)
        return bool(np.mean(z) > 0.5)

    # ------------------------------------------------------------------
    # Welford's online algorithm for mean / variance tracking
    # ------------------------------------------------------------------

    def _welford_update(self, x: np.ndarray) -> None:
        """Incorporate a single observation into running statistics.

        Uses Welford's numerically stable online algorithm for computing
        mean and variance in a single pass.
        """
        x = np.asarray(x, dtype=np.float64)
        if self._run_mean is None:
            self._run_mean = np.zeros_like(x)
            self._run_m2 = np.zeros_like(x)

        self._run_n += 1
        delta = x - self._run_mean
        self._run_mean += delta / self._run_n
        delta2 = x - self._run_mean
        self._run_m2 += delta * delta2

    def _welford_variance(self) -> Optional[np.ndarray]:
        """Return per-feature sample variance, or None if < 2 samples."""
        if self._run_n < 2 or self._run_m2 is None:
            return None
        return self._run_m2 / (self._run_n - 1)
