"""
GIGA TRADER - Signal Detectors
================================
Drift detection and confidence calibration for the signal generator.

Contains:
- ADWINDriftDetector class
- ConfidenceCalibrator class
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import deque


# =============================================================================
# DRIFT DETECTION (ADWIN Algorithm)
# =============================================================================

class ADWINDriftDetector:
    """
    ADWIN (ADaptive WINdowing) drift detection.

    Detects when the distribution of prediction errors changes significantly.
    When drift is detected, we DON'T aggressively adapt - we go conservative.
    """

    def __init__(self, delta: float = 0.002, min_window: int = 30):
        self.delta = delta
        self.min_window = min_window
        self.window = deque(maxlen=1000)
        self.n_detections = 0
        self.last_detection_time: Optional[datetime] = None

    def add_element(self, value: float) -> bool:
        """
        Add element and check for drift.

        Args:
            value: Prediction error or similar metric

        Returns:
            True if drift detected
        """
        self.window.append(value)

        if len(self.window) < self.min_window * 2:
            return False

        # Check for drift using statistical test
        drift_detected = self._check_drift()

        if drift_detected:
            self.n_detections += 1
            self.last_detection_time = datetime.now()
            # Shrink window to forget old data
            while len(self.window) > self.min_window:
                self.window.popleft()

        return drift_detected

    def _check_drift(self) -> bool:
        """Check for statistically significant drift."""
        window_list = list(self.window)
        n = len(window_list)

        # Test multiple split points
        for split in range(self.min_window, n - self.min_window):
            left = window_list[:split]
            right = window_list[split:]

            # Calculate means
            mean_left = np.mean(left)
            mean_right = np.mean(right)

            # Calculate bound for difference
            n_left, n_right = len(left), len(right)
            m = 1.0 / (1.0/n_left + 1.0/n_right)
            eps_cut = np.sqrt(
                (1.0 / (2.0 * m)) * np.log(4.0 / self.delta)
            )

            if abs(mean_left - mean_right) > eps_cut:
                return True

        return False

    def get_severity(self) -> float:
        """Get drift severity (0 = no drift, 1 = severe)."""
        if not self.last_detection_time:
            return 0.0

        # Severity based on recency and frequency
        time_since = (datetime.now() - self.last_detection_time).total_seconds()
        recency_factor = max(0, 1.0 - time_since / 3600)  # Decay over 1 hour
        frequency_factor = min(1.0, self.n_detections / 10)

        return 0.5 * recency_factor + 0.5 * frequency_factor


# =============================================================================
# CONFIDENCE CALIBRATION
# =============================================================================

class ConfidenceCalibrator:
    """
    Calibrate raw confidence to match historical accuracy.

    If a strategy says 70% confidence, it should be right ~70% of the time.
    Uses isotonic regression on historical data.
    """

    def __init__(self, n_bins: int = 10, min_samples: int = 20):
        self.n_bins = n_bins
        self.min_samples = min_samples
        self.calibration_map: Dict[int, float] = {}
        self.history: List[Tuple[float, bool]] = []  # (confidence, was_correct)

    def add_outcome(self, confidence: float, was_correct: bool):
        """Record an outcome for calibration."""
        self.history.append((confidence, was_correct))

        # Recalibrate periodically
        if len(self.history) % 50 == 0:
            self._recalibrate()

    def _recalibrate(self):
        """Recalibrate based on historical data."""
        if len(self.history) < self.min_samples:
            return

        # Bin by confidence
        bins = {}
        for conf, correct in self.history[-500:]:  # Last 500 samples
            bin_idx = min(self.n_bins - 1, int(conf * self.n_bins))
            if bin_idx not in bins:
                bins[bin_idx] = []
            bins[bin_idx].append(1.0 if correct else 0.0)

        # Calculate actual accuracy per bin
        for bin_idx, outcomes in bins.items():
            if len(outcomes) >= 5:  # Min samples per bin
                self.calibration_map[bin_idx] = np.mean(outcomes)

    def calibrate(self, raw_confidence: float) -> float:
        """
        Convert raw confidence to calibrated confidence.

        Args:
            raw_confidence: Strategy's raw confidence (0-1)

        Returns:
            Calibrated confidence based on historical accuracy
        """
        if not self.calibration_map:
            return raw_confidence

        bin_idx = min(self.n_bins - 1, int(raw_confidence * self.n_bins))

        if bin_idx in self.calibration_map:
            return self.calibration_map[bin_idx]

        # Interpolate from nearby bins
        lower = max(k for k in self.calibration_map.keys() if k <= bin_idx) if any(k <= bin_idx for k in self.calibration_map.keys()) else None
        upper = min(k for k in self.calibration_map.keys() if k >= bin_idx) if any(k >= bin_idx for k in self.calibration_map.keys()) else None

        if lower is not None and upper is not None and lower != upper:
            # Linear interpolation
            ratio = (bin_idx - lower) / (upper - lower)
            return (1 - ratio) * self.calibration_map[lower] + ratio * self.calibration_map[upper]
        elif lower is not None:
            return self.calibration_map[lower]
        elif upper is not None:
            return self.calibration_map[upper]

        return raw_confidence
