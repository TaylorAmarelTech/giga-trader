"""
GIGA TRADER - End-of-Day Reconciliation
==========================================
Compares internal trading state vs Alpaca account positions to
detect discrepancies before they compound.
"""

import json
import logging
import os
import tempfile
import threading
from datetime import datetime
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class EODReconciliation:
    """Compare internal state vs broker positions at end of day.

    Parameters
    ----------
    state_file : str
        Path to persist reconciliation history.
    max_history : int
        Maximum reconciliation records to keep.
    """

    def __init__(
        self,
        state_file: str = "data/eod_reconciliation.json",
        max_history: int = 90,
    ):
        self.state_file = state_file
        self.max_history = max_history
        self._lock = threading.Lock()
        self._history: List[Dict] = []
        self._load_history()

    def _load_history(self) -> None:
        """Load reconciliation history from disk."""
        try:
            if os.path.isfile(self.state_file):
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._history = data.get("history", [])
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"EODReconciliation: load failed: {e}")
            self._history = []

    def _save_history(self) -> None:
        """Persist history atomically."""
        try:
            os.makedirs(os.path.dirname(self.state_file) or ".", exist_ok=True)
            data = {"history": self._history[-self.max_history :]}
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
            logger.warning(f"EODReconciliation: save failed: {e}")

    def reconcile(
        self,
        internal_positions: Dict[str, float],
        alpaca_positions: Dict[str, float],
    ) -> Dict:
        """Compare internal vs Alpaca positions.

        Parameters
        ----------
        internal_positions : dict
            {symbol: quantity} from internal state.
        alpaca_positions : dict
            {symbol: quantity} from Alpaca API.

        Returns
        -------
        dict with matched, discrepancies, internal_only, alpaca_only,
        timestamp, severity.
        """
        with self._lock:
            timestamp = datetime.now().isoformat()
            all_symbols = set(internal_positions.keys()) | set(alpaca_positions.keys())

            discrepancies = []
            internal_only = []
            alpaca_only = []

            for symbol in sorted(all_symbols):
                int_qty = internal_positions.get(symbol, 0)
                alp_qty = alpaca_positions.get(symbol, 0)
                diff = int_qty - alp_qty

                if symbol not in alpaca_positions and int_qty != 0:
                    internal_only.append(symbol)
                    discrepancies.append({
                        "symbol": symbol,
                        "internal_qty": int_qty,
                        "alpaca_qty": 0,
                        "diff": int_qty,
                    })
                elif symbol not in internal_positions and alp_qty != 0:
                    alpaca_only.append(symbol)
                    discrepancies.append({
                        "symbol": symbol,
                        "internal_qty": 0,
                        "alpaca_qty": alp_qty,
                        "diff": -alp_qty,
                    })
                elif abs(diff) > 1e-6:
                    pct_diff = abs(diff / alp_qty) if alp_qty != 0 else 1.0
                    discrepancies.append({
                        "symbol": symbol,
                        "internal_qty": int_qty,
                        "alpaca_qty": alp_qty,
                        "diff": diff,
                        "pct_diff": round(pct_diff, 4),
                    })

            # Severity
            if not discrepancies:
                severity = "OK"
            elif internal_only or alpaca_only:
                severity = "CRITICAL"
            else:
                # Check if any discrepancy exceeds 5%
                max_pct = max(
                    (d.get("pct_diff", 1.0) for d in discrepancies),
                    default=0,
                )
                severity = "CRITICAL" if max_pct > 0.05 else "WARNING"

            result = {
                "matched": len(discrepancies) == 0,
                "discrepancies": discrepancies,
                "internal_only": internal_only,
                "alpaca_only": alpaca_only,
                "timestamp": timestamp,
                "severity": severity,
            }

            if severity != "OK":
                logger.warning(
                    f"EODReconciliation [{severity}]: "
                    f"{len(discrepancies)} discrepancies, "
                    f"internal_only={internal_only}, alpaca_only={alpaca_only}"
                )

            self._history.append(result)
            self._save_history()
            return result

    def save_reconciliation(self, result: Dict) -> None:
        """Explicitly save a reconciliation result."""
        with self._lock:
            self._history.append(result)
            self._save_history()

    def get_history(self, last_n: int = 30) -> List[Dict]:
        """Return recent reconciliation results."""
        with self._lock:
            return self._history[-last_n:]
