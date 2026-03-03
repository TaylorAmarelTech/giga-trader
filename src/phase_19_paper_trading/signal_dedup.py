"""
GIGA TRADER - Signal Deduplication
====================================
Prevents duplicate signals on bot restart by persisting the last
signal timestamp to a JSON state file.
"""

import json
import logging
import os
import tempfile
import threading
from datetime import datetime
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class SignalDeduplicator:
    """Prevent duplicate signal generation across bot restarts.

    Persists the last signal to a JSON file using atomic writes
    (temp file + os.replace) to avoid corruption.

    Parameters
    ----------
    state_file : str
        Path to the JSON state file.
    """

    def __init__(self, state_file: str = "data/last_signal_state.json"):
        self.state_file = state_file
        self._lock = threading.Lock()
        self._state: Optional[Dict] = None
        self._load_state()

    def _load_state(self) -> None:
        """Load state from disk, handling missing/corrupt files."""
        try:
            if os.path.isfile(self.state_file):
                with open(self.state_file, "r", encoding="utf-8") as f:
                    self._state = json.load(f)
                logger.debug(f"SignalDedup: loaded state from {self.state_file}")
            else:
                self._state = None
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"SignalDedup: corrupt state file, resetting: {e}")
            self._state = None

    def _save_state(self) -> None:
        """Persist state atomically (temp + rename)."""
        if self._state is None:
            return
        try:
            os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
            fd, tmp_path = tempfile.mkstemp(
                dir=os.path.dirname(self.state_file) or ".",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(self._state, f, indent=2)
                os.replace(tmp_path, self.state_file)
            except Exception:
                # Clean up temp file on failure
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as e:
            logger.warning(f"SignalDedup: failed to save state: {e}")

    def record_signal(
        self,
        signal_type: str,
        timestamp: str,
        price: float,
        metadata: Optional[Dict] = None,
    ) -> None:
        """Record a generated signal.

        Parameters
        ----------
        signal_type : str
            Signal type (e.g. "BUY", "SELL", "HOLD").
        timestamp : str
            ISO format timestamp.
        price : float
            Current price at signal time.
        metadata : dict, optional
            Additional signal metadata.
        """
        with self._lock:
            self._state = {
                "signal_type": signal_type,
                "timestamp": timestamp,
                "price": price,
                "metadata": metadata or {},
            }
            self._save_state()
            logger.debug(f"SignalDedup: recorded {signal_type} at {timestamp}")

    def is_duplicate(
        self,
        signal_type: str,
        min_interval_seconds: int = 3600,
    ) -> bool:
        """Check if a signal would be a duplicate.

        Parameters
        ----------
        signal_type : str
            Proposed signal type.
        min_interval_seconds : int
            Minimum seconds between same-type signals (default 3600 = 1hr).

        Returns
        -------
        bool
            True if same signal_type was generated within min_interval_seconds.
        """
        with self._lock:
            if self._state is None:
                return False

            if self._state.get("signal_type") != signal_type:
                return False

            try:
                last_ts = datetime.fromisoformat(self._state["timestamp"])
                now = datetime.now()
                elapsed = (now - last_ts).total_seconds()
                if elapsed < min_interval_seconds:
                    logger.info(
                        f"SignalDedup: duplicate {signal_type} suppressed "
                        f"({elapsed:.0f}s < {min_interval_seconds}s)"
                    )
                    return True
            except (ValueError, KeyError):
                return False

            return False

    def get_last_signal(self) -> Optional[Dict]:
        """Return the last recorded signal."""
        with self._lock:
            return self._state.copy() if self._state else None

    def clear(self) -> None:
        """Remove state file and reset."""
        with self._lock:
            self._state = None
            try:
                if os.path.isfile(self.state_file):
                    os.unlink(self.state_file)
            except OSError as e:
                logger.warning(f"SignalDedup: failed to remove state file: {e}")
