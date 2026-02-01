"""
Model Health Monitor - Detects when models return suspicious values.

Current Problem: L2 model returning 0.000 probability
This is SUSPICIOUS and indicates one of:
- Input features are all zeros (feature engineering failed)
- Model weights are corrupted
- Scaling/normalization issue
- Model is stuck

This module detects these patterns and provides diagnostics.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Deque
from collections import deque
import numpy as np
import logging

logger = logging.getLogger("GigaTrader.ModelHealth")


@dataclass
class ModelPrediction:
    """Record of a single model prediction."""
    timestamp: datetime
    model_name: str
    probability: float
    features_hash: str = ""  # Hash of input features for debugging


@dataclass
class HealthStatus:
    """Health status for a model."""
    model_name: str
    is_healthy: bool
    issues: List[str] = field(default_factory=list)
    recent_predictions: List[float] = field(default_factory=list)
    stuck_detection: bool = False
    extreme_detection: bool = False
    distribution_anomaly: bool = False
    confidence_score: float = 1.0  # 0-1, how much to trust predictions


class ModelHealthMonitor:
    """
    Monitors model outputs for suspicious patterns.

    Detects:
    1. Stuck models (returning same value repeatedly)
    2. Extreme outputs (always 0.000 or 1.000)
    3. Distribution shift (predictions far from training distribution)
    4. NaN/Inf outputs
    5. Sudden changes in prediction patterns
    """

    def __init__(
        self,
        history_size: int = 100,
        stuck_threshold: int = 10,
        extreme_threshold: float = 0.001,
        distribution_window: int = 50,
    ):
        """
        Initialize ModelHealthMonitor.

        Args:
            history_size: Number of predictions to track per model
            stuck_threshold: Alert if same value appears this many times in a row
            extreme_threshold: Values within this of 0/1 are considered "extreme"
            distribution_window: Window size for distribution analysis
        """
        self.history_size = history_size
        self.stuck_threshold = stuck_threshold
        self.extreme_threshold = extreme_threshold
        self.distribution_window = distribution_window

        self._prediction_history: Dict[str, Deque[ModelPrediction]] = {}
        self._baseline_distributions: Dict[str, Dict] = {}
        self._alert_cooldown: Dict[str, datetime] = {}

    def record_prediction(
        self,
        model_name: str,
        probability: float,
        features: Optional[np.ndarray] = None,
    ) -> HealthStatus:
        """
        Record a prediction and check for health issues.

        Args:
            model_name: Name of the model (e.g., "swing_l2", "swing_gb")
            probability: Predicted probability
            features: Input features (optional, for diagnostics)

        Returns:
            Current health status for the model
        """
        # Initialize history if needed
        if model_name not in self._prediction_history:
            self._prediction_history[model_name] = deque(maxlen=self.history_size)

        # Create prediction record
        features_hash = ""
        if features is not None:
            features_hash = self._hash_features(features)

        prediction = ModelPrediction(
            timestamp=datetime.now(),
            model_name=model_name,
            probability=probability,
            features_hash=features_hash,
        )

        self._prediction_history[model_name].append(prediction)

        # Check health
        return self._check_health(model_name, probability, features)

    def _check_health(
        self,
        model_name: str,
        probability: float,
        features: Optional[np.ndarray],
    ) -> HealthStatus:
        """Check health of model based on recent predictions."""
        issues = []
        is_healthy = True
        confidence = 1.0

        # Check for NaN/Inf
        if np.isnan(probability) or np.isinf(probability):
            issues.append(f"Invalid probability: {probability}")
            is_healthy = False
            confidence = 0.0
            logger.error(f"Model {model_name} returned invalid probability: {probability}")

        # Check for stuck model
        is_stuck, stuck_value = self.check_stuck(model_name)
        if is_stuck:
            issues.append(f"Model appears stuck at {stuck_value:.6f}")
            is_healthy = False
            confidence *= 0.3
            logger.warning(f"Model {model_name} stuck at {stuck_value:.6f}")

        # Check for extreme values
        is_extreme, extreme_desc = self.check_extreme(model_name)
        if is_extreme:
            issues.append(extreme_desc)
            confidence *= 0.5
            logger.warning(f"Model {model_name}: {extreme_desc}")

        # Check distribution shift
        has_shift, shift_info = self.check_distribution(model_name)
        if has_shift:
            issues.append(f"Distribution anomaly: {shift_info}")
            confidence *= 0.7

        # Special case: probability exactly 0.000 or 1.000
        if probability < 0.001:
            issues.append(f"Suspicious near-zero probability: {probability:.6f}")
            confidence *= 0.4
            logger.warning(f"Model {model_name} returned near-zero: {probability:.6f}")

            if features is not None:
                diag = self.diagnose_zero_probability(model_name, features)
                if diag.get("all_zeros"):
                    issues.append("Input features are all zeros!")
                elif diag.get("mostly_zeros"):
                    issues.append(f"Input features {diag['zero_pct']:.1%} zeros")

        elif probability > 0.999:
            issues.append(f"Suspicious near-one probability: {probability:.6f}")
            confidence *= 0.4
            logger.warning(f"Model {model_name} returned near-one: {probability:.6f}")

        # Get recent predictions for status
        recent = [p.probability for p in list(self._prediction_history[model_name])[-10:]]

        return HealthStatus(
            model_name=model_name,
            is_healthy=is_healthy,
            issues=issues,
            recent_predictions=recent,
            stuck_detection=is_stuck,
            extreme_detection=is_extreme,
            distribution_anomaly=has_shift,
            confidence_score=max(0.0, min(1.0, confidence)),
        )

    def check_stuck(self, model_name: str) -> Tuple[bool, Optional[float]]:
        """
        Check if model is stuck returning the same value.

        Returns:
            (is_stuck, stuck_value)
        """
        history = self._prediction_history.get(model_name, [])
        if len(history) < self.stuck_threshold:
            return False, None

        recent = [p.probability for p in list(history)[-self.stuck_threshold:]]

        # Check if all values are nearly identical
        if len(set(round(v, 6) for v in recent)) == 1:
            return True, recent[-1]

        # Check if variance is extremely low
        if np.std(recent) < 1e-6:
            return True, np.mean(recent)

        return False, None

    def check_extreme(self, model_name: str) -> Tuple[bool, str]:
        """
        Check if model is returning extreme values too often.

        Returns:
            (is_extreme, description)
        """
        history = self._prediction_history.get(model_name, [])
        if len(history) < 10:
            return False, ""

        recent = [p.probability for p in list(history)[-20:]]

        near_zero = sum(1 for p in recent if p < self.extreme_threshold)
        near_one = sum(1 for p in recent if p > 1 - self.extreme_threshold)

        total = len(recent)
        zero_pct = near_zero / total
        one_pct = near_one / total

        if zero_pct > 0.5:
            return True, f"{zero_pct:.0%} of predictions near zero"
        if one_pct > 0.5:
            return True, f"{one_pct:.0%} of predictions near one"
        if zero_pct + one_pct > 0.7:
            return True, f"{zero_pct + one_pct:.0%} of predictions are extreme (0 or 1)"

        return False, ""

    def check_distribution(self, model_name: str) -> Tuple[bool, Dict]:
        """
        Check if prediction distribution has shifted from baseline.

        Returns:
            (has_shifted, distribution_stats)
        """
        history = self._prediction_history.get(model_name, [])
        if len(history) < self.distribution_window:
            return False, {}

        recent = [p.probability for p in list(history)[-self.distribution_window:]]
        baseline = self._baseline_distributions.get(model_name, {})

        if not baseline:
            return False, {}

        # Compare current distribution to baseline
        current_mean = np.mean(recent)
        current_std = np.std(recent)

        baseline_mean = baseline.get("mean", 0.5)
        baseline_std = baseline.get("std", 0.2)

        # Check for significant shift
        mean_shift = abs(current_mean - baseline_mean)
        std_shift = abs(current_std - baseline_std)

        stats = {
            "current_mean": current_mean,
            "current_std": current_std,
            "baseline_mean": baseline_mean,
            "baseline_std": baseline_std,
            "mean_shift": mean_shift,
            "std_shift": std_shift,
        }

        # Alert if mean shifted by more than 1 std or std changed significantly
        if mean_shift > baseline_std or std_shift > baseline_std * 0.5:
            return True, stats

        return False, stats

    def set_baseline_distribution(
        self,
        model_name: str,
        mean: float,
        std: float,
        percentiles: Optional[Dict[int, float]] = None,
    ):
        """
        Set expected distribution from training data.

        Args:
            model_name: Name of the model
            mean: Expected mean probability
            std: Expected standard deviation
            percentiles: Optional dict of percentile -> value (e.g., {25: 0.35, 50: 0.50})
        """
        self._baseline_distributions[model_name] = {
            "mean": mean,
            "std": std,
            "percentiles": percentiles or {},
        }

    def diagnose_zero_probability(
        self,
        model_name: str,
        features: np.ndarray,
    ) -> Dict:
        """
        Diagnose why model returned 0.000 probability.

        Args:
            model_name: Name of the model
            features: Input features

        Returns:
            Diagnostic info
        """
        if features is None:
            return {"error": "No features provided"}

        features = np.atleast_2d(features)

        # Check for all zeros
        all_zeros = np.all(features == 0)
        zero_count = np.sum(features == 0)
        zero_pct = zero_count / features.size

        # Check for NaN
        nan_count = np.sum(np.isnan(features))

        # Check for extreme values
        finite_mask = np.isfinite(features)
        if np.any(finite_mask):
            min_val = np.min(features[finite_mask])
            max_val = np.max(features[finite_mask])
            mean_val = np.mean(features[finite_mask])
            std_val = np.std(features[finite_mask])
        else:
            min_val = max_val = mean_val = std_val = float('nan')

        diag = {
            "all_zeros": all_zeros,
            "mostly_zeros": zero_pct > 0.8,
            "zero_count": int(zero_count),
            "zero_pct": zero_pct,
            "nan_count": int(nan_count),
            "min_value": float(min_val),
            "max_value": float(max_val),
            "mean_value": float(mean_val),
            "std_value": float(std_val),
            "feature_shape": features.shape,
        }

        # Log diagnosis
        if all_zeros:
            logger.error(f"Model {model_name}: All input features are ZERO!")
        elif zero_pct > 0.8:
            logger.warning(f"Model {model_name}: {zero_pct:.1%} of features are zero")
        if nan_count > 0:
            logger.warning(f"Model {model_name}: {nan_count} NaN values in features")

        return diag

    def should_trust_prediction(
        self,
        model_name: str,
        probability: float,
    ) -> Tuple[bool, List[str]]:
        """
        Determine if a prediction should be trusted.

        Args:
            model_name: Name of the model
            probability: Predicted probability

        Returns:
            (should_trust, reasons_if_not)
        """
        reasons = []

        # Check for invalid values
        if np.isnan(probability) or np.isinf(probability):
            reasons.append("Invalid probability value")
            return False, reasons

        # Check for extreme values
        if probability < 0.001 or probability > 0.999:
            reasons.append(f"Extreme probability: {probability:.6f}")

        # Check if model is stuck
        is_stuck, _ = self.check_stuck(model_name)
        if is_stuck:
            reasons.append("Model appears stuck")
            return False, reasons

        # Check recent health
        history = self._prediction_history.get(model_name, [])
        if history:
            recent_issues = sum(1 for p in list(history)[-5:]
                              if p.probability < 0.001 or p.probability > 0.999)
            if recent_issues >= 3:
                reasons.append("Recent predictions all extreme")
                return False, reasons

        return len(reasons) == 0, reasons

    def get_health_report(self) -> Dict[str, HealthStatus]:
        """
        Get health status for all monitored models.

        Returns:
            Dict of model_name -> HealthStatus
        """
        report = {}

        for model_name in self._prediction_history:
            history = self._prediction_history[model_name]
            if not history:
                continue

            last_pred = history[-1]
            status = self._check_health(model_name, last_pred.probability, None)
            report[model_name] = status

        return report

    def _hash_features(self, features: np.ndarray) -> str:
        """Create a hash of features for debugging."""
        try:
            return str(hash(features.tobytes()))[:8]
        except Exception:
            return ""

    def clear_history(self, model_name: Optional[str] = None):
        """
        Clear prediction history.

        Args:
            model_name: If provided, clear only this model. Otherwise clear all.
        """
        if model_name:
            if model_name in self._prediction_history:
                self._prediction_history[model_name].clear()
        else:
            self._prediction_history.clear()
