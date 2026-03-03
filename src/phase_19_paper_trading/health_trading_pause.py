"""
GIGA TRADER - Health-Driven Trading Pause
==========================================
Auto-pauses trading when HealthChecker reports degraded/unhealthy status.
Tracks consecutive unhealthy checks with escalation.
"""

import logging
import threading
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class HealthDrivenPause:
    """Auto-pause trading based on system health status.

    Parameters
    ----------
    pause_on_status : list of str, optional
        Health statuses that trigger a pause (default ["UNHEALTHY", "DEGRADED"]).
    escalation_threshold : int
        Consecutive unhealthy checks before escalating to CRITICAL (default 3).
    """

    def __init__(
        self,
        pause_on_status: Optional[List[str]] = None,
        escalation_threshold: int = 3,
    ):
        self.pause_on_status = pause_on_status or ["UNHEALTHY", "DEGRADED"]
        self.escalation_threshold = escalation_threshold
        self._lock = threading.Lock()
        self._consecutive_unhealthy = 0
        self._total_checks = 0
        self._total_pauses = 0
        self._last_status: Optional[str] = None

    def should_pause(self, health_status: Dict) -> Dict:
        """Evaluate whether trading should be paused.

        Parameters
        ----------
        health_status : dict
            Health check result with at least a "status" key.

        Returns
        -------
        dict with paused, reason, status, severity, consecutive_unhealthy.
        """
        with self._lock:
            self._total_checks += 1
            status = health_status.get("status", "UNKNOWN")
            self._last_status = status

            if status in self.pause_on_status:
                self._consecutive_unhealthy += 1
                self._total_pauses += 1

                if self._consecutive_unhealthy >= self.escalation_threshold:
                    severity = "CRITICAL"
                    reason = (
                        f"System {status} for {self._consecutive_unhealthy} "
                        f"consecutive checks (>={self.escalation_threshold})"
                    )
                elif status == "UNHEALTHY":
                    severity = "HIGH"
                    reason = f"System UNHEALTHY: {health_status.get('reason', 'unknown')}"
                else:
                    severity = "MEDIUM"
                    reason = f"System DEGRADED: {health_status.get('reason', 'unknown')}"

                logger.warning(f"HealthDrivenPause [{severity}]: {reason}")
                return {
                    "paused": True,
                    "reason": reason,
                    "status": status,
                    "severity": severity,
                    "consecutive_unhealthy": self._consecutive_unhealthy,
                }
            else:
                if self._consecutive_unhealthy > 0:
                    logger.info(
                        f"HealthDrivenPause: recovered after "
                        f"{self._consecutive_unhealthy} unhealthy checks"
                    )
                self._consecutive_unhealthy = 0
                return {
                    "paused": False,
                    "reason": None,
                    "status": status,
                    "severity": "NONE",
                    "consecutive_unhealthy": 0,
                }

    def get_pause_summary(self) -> str:
        """Human-readable summary of pause state."""
        with self._lock:
            return (
                f"HealthPause: {self._total_pauses}/{self._total_checks} checks paused, "
                f"consecutive_unhealthy={self._consecutive_unhealthy}, "
                f"last_status={self._last_status}"
            )
