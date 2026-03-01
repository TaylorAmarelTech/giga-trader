"""Online learning updater for incremental model updates.

Applies partial_fit to models that support it (SGDClassifier,
PassiveAggressiveClassifier, GaussianNB).  For models without
partial_fit, stores observations in a buffer and signals when
a full retrain is warranted.
"""

import logging
import numpy as np
from collections import deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class OnlineUpdater:
    """
    Online learning updates.  Applies partial_fit to models that support it
    (SGDClassifier, PassiveAggressiveClassifier, GaussianNB).
    For non-partial_fit models, stores a buffer and tracks when retraining
    is needed.
    """

    def __init__(
        self,
        buffer_size: int = 5,
        retrain_threshold: int = 20,
    ):
        """
        Args:
            buffer_size: Number of recent observations to buffer for partial_fit.
            retrain_threshold: Number of buffered samples that triggers a full
                               retrain suggestion.
        """
        self.buffer_size = buffer_size
        self.retrain_threshold = retrain_threshold
        self._buffer_X: deque = deque(maxlen=retrain_threshold)
        self._buffer_y: deque = deque(maxlen=retrain_threshold)
        self._update_history: List[Dict] = []
        self._n_updates: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def update(
        self,
        model: Any,
        X_new: np.ndarray,
        y_new: np.ndarray,
    ) -> Dict:
        """Update model with new data.

        Args:
            model: Fitted sklearn model.
            X_new: New feature data (1 or more samples).
            y_new: New labels.

        Returns:
            Dict with:
              - "method": "partial_fit" or "buffered"
              - "n_samples": number of samples ingested
              - "model_updated": whether model weights were changed
              - "needs_retrain": True when buffer reaches *retrain_threshold*
                                 (only for non-partial_fit models)
              - "buffer_size": current buffer length
              - "timestamp": ISO timestamp of the update
        """
        X_new = np.atleast_2d(np.asarray(X_new))
        y_new = np.asarray(y_new).ravel()
        n_samples = len(y_new)
        ts = datetime.now(timezone.utc).isoformat()

        if self.supports_partial_fit(model):
            model.partial_fit(X_new, y_new, classes=np.array([0, 1]))
            result = {
                "method": "partial_fit",
                "n_samples": n_samples,
                "model_updated": True,
                "needs_retrain": False,
                "buffer_size": len(self._buffer_X),
                "timestamp": ts,
            }
            logger.info(
                "partial_fit update: %d sample(s)", n_samples,
            )
        else:
            # Buffer the samples
            for i in range(n_samples):
                self._buffer_X.append(X_new[i])
                self._buffer_y.append(y_new[i])

            needs_retrain = len(self._buffer_X) >= self.retrain_threshold
            result = {
                "method": "buffered",
                "n_samples": n_samples,
                "model_updated": False,
                "needs_retrain": needs_retrain,
                "buffer_size": len(self._buffer_X),
                "timestamp": ts,
            }
            if needs_retrain:
                logger.warning(
                    "Buffer reached retrain threshold (%d). Full retrain recommended.",
                    self.retrain_threshold,
                )
            else:
                logger.info(
                    "Buffered %d sample(s). Buffer: %d / %d",
                    n_samples,
                    len(self._buffer_X),
                    self.retrain_threshold,
                )

        self._n_updates += 1
        self._update_history.append(result)
        return result

    def get_buffer(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return current buffer contents as (X, y) arrays.

        Returns:
            Tuple of (X, y) numpy arrays.  If the buffer is empty both
            arrays have shape (0,).
        """
        if len(self._buffer_X) == 0:
            return np.array([]).reshape(0, 0), np.array([])
        X = np.array(list(self._buffer_X))
        y = np.array(list(self._buffer_y))
        return X, y

    def clear_buffer(self) -> None:
        """Clear the buffer (e.g., after a full retrain)."""
        self._buffer_X.clear()
        self._buffer_y.clear()
        logger.info("Buffer cleared.")

    def supports_partial_fit(self, model: Any) -> bool:
        """Check if a model supports partial_fit."""
        return hasattr(model, "partial_fit") and callable(model.partial_fit)

    @property
    def n_updates(self) -> int:
        """Number of updates performed."""
        return self._n_updates

    @property
    def update_history(self) -> List[Dict]:
        """Return update history."""
        return self._update_history.copy()
