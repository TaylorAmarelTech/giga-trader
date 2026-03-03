"""
GIGA TRADER - Gate Audit Log
==============================
Persistent SQLite log of every trading gate evaluation for
analysis and debugging.
"""

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class GateAuditLog:
    """Log every trading gate evaluation to SQLite.

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
        """Create gate_audit_log table if not exists."""
        try:
            with self._get_conn() as conn:
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS gate_audit_log (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        gate_name TEXT NOT NULL,
                        result TEXT NOT NULL,
                        reason TEXT,
                        details TEXT,
                        signal_type TEXT,
                        market_conditions TEXT
                    )
                """)
                # Index for common queries
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_gate_audit_timestamp
                    ON gate_audit_log(timestamp)
                """)
                conn.execute("""
                    CREATE INDEX IF NOT EXISTS idx_gate_audit_name
                    ON gate_audit_log(gate_name)
                """)
        except sqlite3.Error as e:
            logger.warning(f"GateAuditLog: table creation failed: {e}")

    def log_evaluation(
        self,
        gate_name: str,
        result: str,
        reason: Optional[str] = None,
        details: Optional[Dict] = None,
        signal_type: Optional[str] = None,
        market_conditions: Optional[Dict] = None,
    ) -> None:
        """Log a gate evaluation.

        Parameters
        ----------
        gate_name : str
            Name of the gate (e.g. "macro_calendar", "vol_regime").
        result : str
            "PASS", "BLOCK", or "REDUCE".
        reason : str, optional
            Human-readable reason.
        details : dict, optional
            Gate-specific data (serialized to JSON).
        signal_type : str, optional
            "BUY" or "SELL".
        market_conditions : dict, optional
            Snapshot of market state.
        """
        with self._lock:
            try:
                with self._get_conn() as conn:
                    conn.execute(
                        """
                        INSERT INTO gate_audit_log
                            (timestamp, gate_name, result, reason, details,
                             signal_type, market_conditions)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            datetime.now().isoformat(),
                            gate_name,
                            result,
                            reason,
                            json.dumps(details) if details else None,
                            signal_type,
                            json.dumps(market_conditions) if market_conditions else None,
                        ),
                    )
            except sqlite3.Error as e:
                logger.warning(f"GateAuditLog: log failed: {e}")

    def get_gate_stats(
        self,
        gate_name: Optional[str] = None,
        last_n_days: int = 30,
    ) -> Dict:
        """Get gate evaluation statistics.

        Parameters
        ----------
        gate_name : str, optional
            Specific gate to analyze (None = all gates).
        last_n_days : int
            Look back period.

        Returns
        -------
        dict mapping gate_name -> {pass_count, block_count, reduce_count,
        block_rate, total}.
        """
        with self._lock:
            try:
                cutoff = (datetime.now() - timedelta(days=last_n_days)).isoformat()
                with self._get_conn() as conn:
                    conn.row_factory = sqlite3.Row

                    if gate_name:
                        rows = conn.execute(
                            """
                            SELECT gate_name, result, COUNT(*) as cnt
                            FROM gate_audit_log
                            WHERE timestamp > ? AND gate_name = ?
                            GROUP BY gate_name, result
                            """,
                            (cutoff, gate_name),
                        ).fetchall()
                    else:
                        rows = conn.execute(
                            """
                            SELECT gate_name, result, COUNT(*) as cnt
                            FROM gate_audit_log
                            WHERE timestamp > ?
                            GROUP BY gate_name, result
                            """,
                            (cutoff,),
                        ).fetchall()

                stats: Dict[str, Dict] = {}
                for row in rows:
                    gn = row["gate_name"]
                    if gn not in stats:
                        stats[gn] = {"pass_count": 0, "block_count": 0, "reduce_count": 0, "total": 0}
                    result = row["result"]
                    cnt = row["cnt"]
                    stats[gn]["total"] += cnt
                    if result == "PASS":
                        stats[gn]["pass_count"] += cnt
                    elif result == "BLOCK":
                        stats[gn]["block_count"] += cnt
                    elif result == "REDUCE":
                        stats[gn]["reduce_count"] += cnt

                for gn in stats:
                    total = stats[gn]["total"]
                    stats[gn]["block_rate"] = (
                        round(stats[gn]["block_count"] / total, 3) if total > 0 else 0.0
                    )

                return stats

            except sqlite3.Error as e:
                logger.warning(f"GateAuditLog: stats failed: {e}")
                return {}

    def get_block_history(self, last_n: int = 50) -> List[Dict]:
        """Get recent gate blocks.

        Parameters
        ----------
        last_n : int
            Number of recent blocks to return.

        Returns
        -------
        list of dicts with gate_name, result, reason, timestamp, signal_type.
        """
        with self._lock:
            try:
                with self._get_conn() as conn:
                    conn.row_factory = sqlite3.Row
                    rows = conn.execute(
                        """
                        SELECT timestamp, gate_name, result, reason, signal_type
                        FROM gate_audit_log
                        WHERE result = 'BLOCK'
                        ORDER BY id DESC LIMIT ?
                        """,
                        (last_n,),
                    ).fetchall()

                return [dict(row) for row in rows]

            except sqlite3.Error as e:
                logger.warning(f"GateAuditLog: block history failed: {e}")
                return []
