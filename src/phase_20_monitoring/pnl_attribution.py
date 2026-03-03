"""
GIGA TRADER - P&L Attribution
===============================
Tracks which gates, models, and features drove each signal and
its outcome. Persists to SQLite for analysis.
"""

import json
import logging
import os
import sqlite3
import threading
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class PnLAttribution:
    """P&L attribution tracker using SQLite.

    Records signal metadata (gates passed/blocked, model weights,
    top features, sizing chain) and later links to actual P&L outcomes.

    Parameters
    ----------
    db_path : str
        Path to SQLite database.
    """

    def __init__(self, db_path: str = "data/giga_trader.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
        self._ensure_table()

    def _get_conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path, check_same_thread=False)

    def _ensure_table(self) -> None:
        """Create pnl_attribution table if not exists."""
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS pnl_attribution (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        signal_type TEXT,
                        entry_price REAL,
                        exit_price REAL,
                        pnl_pct REAL,
                        swing_proba REAL,
                        timing_proba REAL,
                        position_size REAL,
                        gates_passed TEXT,
                        gates_blocked TEXT,
                        top_features TEXT,
                        model_weights TEXT,
                        sizing_chain TEXT,
                        regime TEXT,
                        notes TEXT
                    )
                """)
        except sqlite3.Error as e:
            logger.warning(f"PnLAttribution: table creation failed: {e}")

    def record_signal(self, signal_data: Dict) -> Optional[int]:
        """Record a new signal with its attribution data.

        Parameters
        ----------
        signal_data : dict
            Keys: timestamp, signal_type, entry_price, swing_proba,
            timing_proba, position_size, gates_passed (list),
            gates_blocked (list), top_features (dict),
            model_weights (dict), sizing_chain (list), regime, notes.

        Returns
        -------
        int or None
            Row ID of inserted record.
        """
        with self._lock:
            try:
                with self._get_conn() as conn:
                    cursor = conn.execute(
                        """
                        INSERT INTO pnl_attribution
                            (timestamp, signal_type, entry_price, swing_proba,
                             timing_proba, position_size, gates_passed,
                             gates_blocked, top_features, model_weights,
                             sizing_chain, regime, notes)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            signal_data.get("timestamp", ""),
                            signal_data.get("signal_type", ""),
                            signal_data.get("entry_price"),
                            signal_data.get("swing_proba"),
                            signal_data.get("timing_proba"),
                            signal_data.get("position_size"),
                            json.dumps(signal_data.get("gates_passed", [])),
                            json.dumps(signal_data.get("gates_blocked", [])),
                            json.dumps(signal_data.get("top_features", {})),
                            json.dumps(signal_data.get("model_weights", {})),
                            json.dumps(signal_data.get("sizing_chain", [])),
                            signal_data.get("regime", ""),
                            signal_data.get("notes", ""),
                        ),
                    )
                    return cursor.lastrowid
            except sqlite3.Error as e:
                logger.warning(f"PnLAttribution: record failed: {e}")
                return None

    def record_outcome(
        self,
        signal_id: int,
        exit_price: float,
        pnl_pct: float,
    ) -> None:
        """Update a signal record with its outcome.

        Parameters
        ----------
        signal_id : int
            Row ID from record_signal.
        exit_price : float
            Exit price.
        pnl_pct : float
            Realized P&L percentage.
        """
        with self._lock:
            try:
                with self._get_conn() as conn:
                    conn.execute(
                        "UPDATE pnl_attribution SET exit_price=?, pnl_pct=? WHERE id=?",
                        (exit_price, pnl_pct, signal_id),
                    )
            except sqlite3.Error as e:
                logger.warning(f"PnLAttribution: outcome update failed: {e}")

    def get_attribution_summary(self, last_n: int = 30) -> Dict:
        """Get summary of recent signals and their outcomes.

        Parameters
        ----------
        last_n : int
            Number of recent signals to analyze.

        Returns
        -------
        dict with total_signals, win_rate, avg_pnl_pct,
        top_contributing_features, regime_performance.
        """
        with self._lock:
            try:
                with self._get_conn() as conn:
                    conn.row_factory = sqlite3.Row
                    rows = conn.execute(
                        """
                        SELECT * FROM pnl_attribution
                        WHERE pnl_pct IS NOT NULL
                        ORDER BY id DESC LIMIT ?
                        """,
                        (last_n,),
                    ).fetchall()

                if not rows:
                    return {
                        "total_signals": 0,
                        "win_rate": 0.0,
                        "avg_pnl_pct": 0.0,
                        "top_contributing_features": [],
                        "regime_performance": {},
                    }

                pnls = [r["pnl_pct"] for r in rows if r["pnl_pct"] is not None]
                wins = sum(1 for p in pnls if p > 0)

                # Aggregate regime performance
                regime_perf: Dict[str, List[float]] = {}
                for r in rows:
                    regime = r["regime"] or "UNKNOWN"
                    pnl = r["pnl_pct"]
                    if pnl is not None:
                        regime_perf.setdefault(regime, []).append(pnl)

                return {
                    "total_signals": len(rows),
                    "win_rate": round(wins / len(pnls), 3) if pnls else 0.0,
                    "avg_pnl_pct": round(sum(pnls) / len(pnls), 4) if pnls else 0.0,
                    "top_contributing_features": [],
                    "regime_performance": {
                        k: round(sum(v) / len(v), 4) for k, v in regime_perf.items()
                    },
                }

            except sqlite3.Error as e:
                logger.warning(f"PnLAttribution: summary failed: {e}")
                return {"total_signals": 0, "win_rate": 0.0, "avg_pnl_pct": 0.0}
