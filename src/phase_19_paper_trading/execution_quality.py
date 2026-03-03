"""
GIGA TRADER - Execution Quality Tracker
=========================================
Compares expected vs actual fill prices from Alpaca to measure
real slippage and execution quality over time.
"""

import json
import logging
import os
import tempfile
import threading
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


class ExecutionQualityTracker:
    """Track execution quality by comparing expected vs actual fill prices.

    Slippage is measured in basis points (bps):
        slippage_bps = (actual - expected) / expected * 10000

    For LONG orders: positive slippage = paid more (unfavorable)
    For SHORT orders: negative slippage = received less (unfavorable)

    Parameters
    ----------
    state_file : str
        Path to persist execution records.
    max_records : int
        Maximum records to keep in history (default 500).
    """

    def __init__(
        self,
        state_file: str = "data/execution_quality.json",
        max_records: int = 500,
    ):
        self.state_file = state_file
        self.max_records = max_records
        self._lock = threading.Lock()
        self._expectations: Dict[str, Dict] = {}  # order_id -> {expected, direction, ts}
        self._history: List[Dict] = []
        self._load_state()

    def _load_state(self) -> None:
        """Load history from disk."""
        try:
            if os.path.isfile(self.state_file):
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._history = data.get("history", [])
                self._expectations = data.get("pending", {})
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"ExecutionQuality: corrupt state, resetting: {e}")
            self._history = []
            self._expectations = {}

    def _save_state(self) -> None:
        """Persist state atomically."""
        try:
            os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
            data = {
                "history": self._history[-self.max_records :],
                "pending": self._expectations,
            }
            fd, tmp_path = tempfile.mkstemp(
                dir=os.path.dirname(self.state_file) or ".",
                suffix=".tmp",
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
                os.replace(tmp_path, self.state_file)
            except Exception:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except OSError as e:
            logger.warning(f"ExecutionQuality: failed to save: {e}")

    def record_expectation(
        self,
        order_id: str,
        expected_price: float,
        direction: str,
        timestamp: str,
    ) -> None:
        """Record expected fill price before order submission.

        Parameters
        ----------
        order_id : str
            Unique order identifier.
        expected_price : float
            Expected fill price at order time.
        direction : str
            "LONG" or "SHORT".
        timestamp : str
            ISO format timestamp.
        """
        with self._lock:
            self._expectations[order_id] = {
                "expected_price": expected_price,
                "direction": direction.upper(),
                "timestamp": timestamp,
            }
            self._save_state()

    def record_fill(
        self,
        order_id: str,
        actual_price: float,
        fill_timestamp: str,
    ) -> Optional[Dict]:
        """Record actual fill price and compute slippage.

        Parameters
        ----------
        order_id : str
            Order identifier matching a previous expectation.
        actual_price : float
            Actual fill price from broker.
        fill_timestamp : str
            ISO format fill timestamp.

        Returns
        -------
        dict or None
            Slippage record if expectation was found.
        """
        with self._lock:
            expectation = self._expectations.pop(order_id, None)
            if expectation is None:
                logger.debug(f"ExecutionQuality: no expectation for order {order_id}")
                return None

            expected = expectation["expected_price"]
            direction = expectation["direction"]

            if expected == 0:
                slippage_bps = 0.0
            else:
                slippage_bps = (actual_price - expected) / expected * 10000

            # Favorable determination
            if direction == "LONG":
                favorable = slippage_bps <= 0  # Paid less than expected
            else:
                favorable = slippage_bps >= 0  # Received more than expected

            record = {
                "order_id": order_id,
                "expected": expected,
                "actual": actual_price,
                "slippage_bps": round(slippage_bps, 2),
                "direction": direction,
                "favorable": favorable,
                "order_timestamp": expectation["timestamp"],
                "fill_timestamp": fill_timestamp,
            }

            self._history.append(record)
            # Trim history
            if len(self._history) > self.max_records:
                self._history = self._history[-self.max_records :]

            self._save_state()
            logger.info(
                f"ExecutionQuality: {order_id} slippage={slippage_bps:+.1f}bps "
                f"({'favorable' if favorable else 'unfavorable'})"
            )
            return record

    def compute_slippage(self, order_id: str) -> Optional[Dict]:
        """Look up slippage for a completed order."""
        with self._lock:
            for record in reversed(self._history):
                if record["order_id"] == order_id:
                    return record
            return None

    def get_summary(self, last_n: int = 50) -> Dict:
        """Get execution quality summary.

        Parameters
        ----------
        last_n : int
            Number of recent orders to analyze.

        Returns
        -------
        dict with avg_slippage_bps, median_slippage_bps, pct_favorable,
        total_orders, worst_slippage_bps.
        """
        with self._lock:
            records = self._history[-last_n:] if self._history else []

        if not records:
            return {
                "avg_slippage_bps": 0.0,
                "median_slippage_bps": 0.0,
                "pct_favorable": 0.0,
                "total_orders": 0,
                "worst_slippage_bps": 0.0,
            }

        slippages = [r["slippage_bps"] for r in records]
        favorable_count = sum(1 for r in records if r["favorable"])

        return {
            "avg_slippage_bps": round(float(np.mean(slippages)), 2),
            "median_slippage_bps": round(float(np.median(slippages)), 2),
            "pct_favorable": round(favorable_count / len(records), 3),
            "total_orders": len(records),
            "worst_slippage_bps": round(float(max(slippages, key=abs)), 2),
        }
