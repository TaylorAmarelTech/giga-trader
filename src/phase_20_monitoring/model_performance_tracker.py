"""
GIGA TRADER - Model Performance Tracker
=========================================
Tracks rolling AUC by comparing predictions vs actual outcomes.
Triggers alerts when model performance decays beyond threshold.
"""

import json
import logging
import os
import tempfile
import threading
from collections import deque
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ModelPerformanceTracker:
    """Track rolling AUC and trigger retrain alerts.

    Parameters
    ----------
    window_size : int
        Number of recent predictions for rolling AUC (default 50).
    decay_threshold : float
        AUC decay from baseline before DEGRADING alert (default 0.05).
    alert_threshold : float
        AUC decay from baseline before CRITICAL alert (default 0.10).
    state_file : str
        Path for persistent state.
    auto_save_interval : int
        Save state every N predictions (default 10).
    """

    def __init__(
        self,
        window_size: int = 50,
        decay_threshold: float = 0.05,
        alert_threshold: float = 0.10,
        state_file: str = "data/model_performance_state.json",
        auto_save_interval: int = 10,
    ):
        self.window_size = window_size
        self.decay_threshold = decay_threshold
        self.alert_threshold = alert_threshold
        self.state_file = state_file
        self.auto_save_interval = auto_save_interval
        self._lock = threading.Lock()
        self._predictions: deque = deque(maxlen=window_size * 2)
        self._baseline_auc: Optional[float] = None
        self._unsaved_count = 0
        self.load_state()

    def set_baseline(self, baseline_auc: float) -> None:
        """Set training AUC as reference point."""
        with self._lock:
            self._baseline_auc = baseline_auc
            logger.info(f"ModelPerformanceTracker: baseline AUC set to {baseline_auc:.4f}")
            self._save_state_locked()

    def record_prediction(
        self,
        prediction: float,
        actual: int,
        timestamp: Optional[str] = None,
    ) -> None:
        """Record a single prediction/actual pair.

        Parameters
        ----------
        prediction : float
            Model's predicted probability (0-1).
        actual : int
            Actual outcome (0 or 1).
        timestamp : str, optional
            ISO format timestamp.
        """
        with self._lock:
            self._predictions.append({
                "prediction": float(prediction),
                "actual": int(actual),
                "timestamp": timestamp or "",
            })
            self._unsaved_count += 1

            if self._unsaved_count >= self.auto_save_interval:
                self._save_state_locked()
                self._unsaved_count = 0

    def compute_rolling_auc(self) -> Dict:
        """Compute rolling AUC and generate alerts.

        Returns
        -------
        dict with rolling_auc, baseline_auc, decay, n_predictions,
        alert, should_retrain.
        """
        with self._lock:
            n = len(self._predictions)
            if n < 10:
                return {
                    "rolling_auc": None,
                    "baseline_auc": self._baseline_auc,
                    "decay": 0.0,
                    "n_predictions": n,
                    "alert": "INSUFFICIENT_DATA",
                    "should_retrain": False,
                }

            # Get recent predictions
            recent = list(self._predictions)[-self.window_size :]
            preds = np.array([r["prediction"] for r in recent])
            actuals = np.array([r["actual"] for r in recent])

            # Check for all-same class
            unique_actuals = np.unique(actuals)
            if len(unique_actuals) < 2:
                return {
                    "rolling_auc": None,
                    "baseline_auc": self._baseline_auc,
                    "decay": 0.0,
                    "n_predictions": n,
                    "alert": "SINGLE_CLASS",
                    "should_retrain": False,
                }

            # Compute AUC
            rolling_auc = self._compute_auc(preds, actuals)

            if rolling_auc is None:
                return {
                    "rolling_auc": None,
                    "baseline_auc": self._baseline_auc,
                    "decay": 0.0,
                    "n_predictions": n,
                    "alert": "COMPUTATION_ERROR",
                    "should_retrain": False,
                }

            # Compute decay
            decay = 0.0
            if self._baseline_auc is not None:
                decay = self._baseline_auc - rolling_auc

            # Alert level
            if decay >= self.alert_threshold:
                alert = "CRITICAL"
                should_retrain = True
                logger.warning(
                    f"ModelPerformanceTracker [CRITICAL]: AUC decayed by {decay:.4f} "
                    f"(rolling={rolling_auc:.4f} vs baseline={self._baseline_auc})"
                )
            elif decay >= self.decay_threshold:
                alert = "DEGRADING"
                should_retrain = False
                logger.info(
                    f"ModelPerformanceTracker [DEGRADING]: AUC decayed by {decay:.4f}"
                )
            else:
                alert = "NONE"
                should_retrain = False

            return {
                "rolling_auc": round(rolling_auc, 4),
                "baseline_auc": self._baseline_auc,
                "decay": round(decay, 4),
                "n_predictions": n,
                "alert": alert,
                "should_retrain": should_retrain,
            }

    @staticmethod
    def _compute_auc(predictions: np.ndarray, actuals: np.ndarray) -> Optional[float]:
        """Compute AUC-ROC, handling edge cases."""
        try:
            from sklearn.metrics import roc_auc_score
            return float(roc_auc_score(actuals, predictions))
        except ImportError:
            pass
        except ValueError:
            return None

        # Manual AUC if sklearn unavailable (Wilcoxon-Mann-Whitney)
        try:
            pos = predictions[actuals == 1]
            neg = predictions[actuals == 0]
            if len(pos) == 0 or len(neg) == 0:
                return None
            # Count concordant pairs
            auc = 0.0
            for p in pos:
                auc += np.sum(p > neg) + 0.5 * np.sum(p == neg)
            auc /= len(pos) * len(neg)
            return float(auc)
        except Exception:
            return None

    def save_state(self) -> None:
        """Persist state to disk atomically."""
        with self._lock:
            self._save_state_locked()

    def _save_state_locked(self) -> None:
        """Internal save (caller must hold lock)."""
        try:
            os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
            data = {
                "baseline_auc": self._baseline_auc,
                "predictions": list(self._predictions),
            }
            fd, tmp_path = tempfile.mkstemp(
                dir=os.path.dirname(self.state_file) or ".",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f)
                os.replace(tmp_path, self.state_file)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as e:
            logger.warning(f"ModelPerformanceTracker: save failed: {e}")

    def load_state(self) -> None:
        """Load state from disk."""
        try:
            if os.path.isfile(self.state_file):
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._baseline_auc = data.get("baseline_auc")
                preds = data.get("predictions", [])
                self._predictions = deque(preds, maxlen=self.window_size * 2)
                logger.debug(
                    f"ModelPerformanceTracker: loaded {len(self._predictions)} predictions"
                )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"ModelPerformanceTracker: load failed, starting fresh: {e}")
