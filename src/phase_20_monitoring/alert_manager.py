"""
GIGA TRADER - Alert Manager
==============================
Centralized alerting system for monitoring events.

Supports multiple notification channels:
  - Log-based alerts (always active)
  - File-based alerts (JSON log for dashboard consumption)
  - Webhook alerts (optional, for Slack/Discord/email integrations)

Usage:
    from src.phase_20_monitoring.alert_manager import (
        AlertManager,
        Alert,
        AlertLevel,
        AlertChannel,
    )
"""

import json
import logging
import threading
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Callable, Dict, List, Optional

logger = logging.getLogger("GigaTrader.AlertManager")


# =============================================================================
# ENUMS & DATA STRUCTURES
# =============================================================================

class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels."""
    LOG = "log"
    FILE = "file"
    WEBHOOK = "webhook"
    CALLBACK = "callback"


@dataclass
class Alert:
    """A single alert event."""
    level: AlertLevel
    source: str  # e.g., "health_check", "model_drift", "drawdown"
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)
    acknowledged: bool = False
    alert_id: str = ""

    def __post_init__(self):
        if not self.alert_id:
            self.alert_id = f"{self.source}_{self.timestamp.strftime('%Y%m%d_%H%M%S_%f')}"

    def to_dict(self) -> Dict:
        """Serialize to dict."""
        return {
            "alert_id": self.alert_id,
            "level": self.level.value,
            "source": self.source,
            "message": self.message,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "acknowledged": self.acknowledged,
        }


@dataclass
class AlertRule:
    """Defines when to trigger an alert."""
    name: str
    condition: Callable[[Dict], bool]  # Takes metrics dict, returns True to alert
    level: AlertLevel = AlertLevel.WARNING
    message_template: str = ""
    cooldown_seconds: int = 300  # Min seconds between repeated alerts
    last_triggered: Optional[datetime] = None

    def should_trigger(self, metrics: Dict) -> bool:
        """Check condition and cooldown."""
        if not self.condition(metrics):
            return False
        if self.last_triggered:
            elapsed = (datetime.now() - self.last_triggered).total_seconds()
            if elapsed < self.cooldown_seconds:
                return False
        return True


# =============================================================================
# ALERT MANAGER
# =============================================================================

class RuleBasedAlertManager:
    """
    Rule-based alert management with deduplication.

    Complements the existing AlertManager in health_checker.py by adding:
      - Declarative alert rules with condition functions
      - Deduplication within configurable windows
      - File-based JSON-lines logging for dashboard consumption
      - Alert acknowledgment tracking
    """

    def __init__(
        self,
        alert_file: Optional[Path] = None,
        max_alerts: int = 5000,
        dedup_window_seconds: int = 60,
    ):
        """
        Args:
            alert_file: Path for JSON alert log (None = no file output).
            max_alerts: Maximum alerts to keep in memory.
            dedup_window_seconds: Suppress duplicate alerts within this window.
        """
        self.alert_file = alert_file
        self.max_alerts = max_alerts
        self.dedup_window_seconds = dedup_window_seconds

        self.alerts: List[Alert] = []
        self.rules: Dict[str, AlertRule] = {}
        self.callbacks: List[Callable[[Alert], None]] = []
        self.webhook_urls: List[str] = []
        self._lock = threading.Lock()
        self._suppressed_count = 0

    # ── Alert dispatch ───────────────────────────────────────────────────

    def send_alert(self, alert: Alert) -> bool:
        """
        Dispatch an alert through all configured channels.

        Returns True if alert was dispatched (not suppressed).
        """
        # Dedup check
        if self._is_duplicate(alert):
            self._suppressed_count += 1
            return False

        with self._lock:
            self.alerts.append(alert)
            if len(self.alerts) > self.max_alerts:
                self.alerts = self.alerts[-self.max_alerts:]

        # Channel: Log
        log_fn = {
            AlertLevel.INFO: logger.info,
            AlertLevel.WARNING: logger.warning,
            AlertLevel.ERROR: logger.error,
            AlertLevel.CRITICAL: logger.critical,
        }.get(alert.level, logger.warning)
        log_fn(f"[ALERT:{alert.source}] {alert.message}")

        # Channel: File
        if self.alert_file:
            self._write_to_file(alert)

        # Channel: Callbacks
        for cb in self.callbacks:
            try:
                cb(alert)
            except Exception as e:
                logger.debug(f"Alert callback error: {e}")

        # Channel: Webhooks (fire-and-forget)
        for url in self.webhook_urls:
            self._send_webhook(url, alert)

        return True

    def alert(
        self,
        level: AlertLevel,
        source: str,
        message: str,
        **metadata,
    ) -> bool:
        """Convenience method to create and send an alert."""
        a = Alert(level=level, source=source, message=message, metadata=metadata)
        return self.send_alert(a)

    # ── Rules engine ─────────────────────────────────────────────────────

    def add_rule(self, rule: AlertRule) -> None:
        """Register an alert rule."""
        self.rules[rule.name] = rule

    def remove_rule(self, name: str) -> bool:
        """Remove an alert rule."""
        return self.rules.pop(name, None) is not None

    def evaluate_rules(self, metrics: Dict) -> List[Alert]:
        """Evaluate all rules against current metrics, fire matching alerts."""
        fired = []
        for rule in self.rules.values():
            if rule.should_trigger(metrics):
                msg = rule.message_template.format(**metrics) if rule.message_template else f"Rule '{rule.name}' triggered"
                alert = Alert(level=rule.level, source=f"rule:{rule.name}", message=msg, metadata=metrics)
                rule.last_triggered = datetime.now()
                if self.send_alert(alert):
                    fired.append(alert)
        return fired

    # ── Channel configuration ────────────────────────────────────────────

    def add_callback(self, callback: Callable[[Alert], None]) -> None:
        """Add a callback function for alerts."""
        self.callbacks.append(callback)

    def add_webhook(self, url: str) -> None:
        """Add a webhook URL for alerts."""
        self.webhook_urls.append(url)

    # ── Query & management ───────────────────────────────────────────────

    def get_alerts(
        self,
        level: Optional[AlertLevel] = None,
        source: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Alert]:
        """Query alerts with optional filters."""
        with self._lock:
            result = list(self.alerts)
        if level:
            result = [a for a in result if a.level == level]
        if source:
            result = [a for a in result if a.source == source]
        if since:
            result = [a for a in result if a.timestamp >= since]
        return result[-limit:]

    def get_unacknowledged(self, level: Optional[AlertLevel] = None) -> List[Alert]:
        """Get all unacknowledged alerts."""
        with self._lock:
            result = [a for a in self.alerts if not a.acknowledged]
        if level:
            result = [a for a in result if a.level == level]
        return result

    def acknowledge(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged."""
        with self._lock:
            for a in self.alerts:
                if a.alert_id == alert_id:
                    a.acknowledged = True
                    return True
        return False

    def acknowledge_all(self, source: Optional[str] = None) -> int:
        """Acknowledge all (or source-filtered) alerts. Returns count."""
        count = 0
        with self._lock:
            for a in self.alerts:
                if not a.acknowledged and (source is None or a.source == source):
                    a.acknowledged = True
                    count += 1
        return count

    def get_summary(self) -> Dict:
        """Get alert summary statistics."""
        with self._lock:
            total = len(self.alerts)
            by_level = {}
            for level in AlertLevel:
                by_level[level.value] = sum(1 for a in self.alerts if a.level == level)
            unacked = sum(1 for a in self.alerts if not a.acknowledged)
            recent = self.alerts[-5:] if self.alerts else []
        return {
            "total_alerts": total,
            "by_level": by_level,
            "unacknowledged": unacked,
            "suppressed": self._suppressed_count,
            "n_rules": len(self.rules),
            "recent": [a.to_dict() for a in recent],
        }

    # ── Internal helpers ─────────────────────────────────────────────────

    def _is_duplicate(self, alert: Alert) -> bool:
        """Check if a near-identical alert was sent recently."""
        cutoff = datetime.now() - timedelta(seconds=self.dedup_window_seconds)
        with self._lock:
            for existing in reversed(self.alerts):
                if existing.timestamp < cutoff:
                    break
                if existing.source == alert.source and existing.message == alert.message:
                    return True
        return False

    def _write_to_file(self, alert: Alert) -> None:
        """Append alert to JSON-lines file."""
        try:
            self.alert_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.alert_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(alert.to_dict()) + "\n")
        except Exception as e:
            logger.debug(f"Failed to write alert to file: {e}")

    def _send_webhook(self, url: str, alert: Alert) -> None:
        """Send alert to a webhook URL (best-effort, non-blocking)."""
        try:
            import urllib.request
            data = json.dumps({"text": f"[{alert.level.value.upper()}] {alert.source}: {alert.message}"}).encode()
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            urllib.request.urlopen(req, timeout=5)
        except Exception as e:
            logger.debug(f"Webhook send failed: {e}")

    def __repr__(self) -> str:
        return f"AlertManager(alerts={len(self.alerts)}, rules={len(self.rules)})"


# =============================================================================
# PRESET ALERT RULES
# =============================================================================

def create_drawdown_rule(threshold: float = 0.05) -> AlertRule:
    """Alert when drawdown exceeds threshold."""
    return AlertRule(
        name="max_drawdown",
        condition=lambda m: m.get("drawdown_pct", 0) >= threshold,
        level=AlertLevel.CRITICAL,
        message_template="Drawdown {drawdown_pct:.2%} exceeds limit",
        cooldown_seconds=600,
    )


def create_model_drift_rule(threshold: float = 0.10) -> AlertRule:
    """Alert when model predictions drift significantly."""
    return AlertRule(
        name="model_drift",
        condition=lambda m: m.get("prediction_drift", 0) >= threshold,
        level=AlertLevel.WARNING,
        message_template="Model prediction drift: {prediction_drift:.3f}",
        cooldown_seconds=1800,
    )


def create_data_stale_rule(max_minutes: int = 30) -> AlertRule:
    """Alert when data feed appears stale."""
    return AlertRule(
        name="data_staleness",
        condition=lambda m: m.get("data_age_minutes", 0) >= max_minutes,
        level=AlertLevel.ERROR,
        message_template="Data feed stale: {data_age_minutes:.0f} minutes old",
        cooldown_seconds=300,
    )


def create_consecutive_loss_rule(max_losses: int = 5) -> AlertRule:
    """Alert on consecutive losing trades."""
    return AlertRule(
        name="consecutive_losses",
        condition=lambda m: m.get("consecutive_losses", 0) >= max_losses,
        level=AlertLevel.WARNING,
        message_template="Consecutive losses: {consecutive_losses}",
        cooldown_seconds=3600,
    )


def create_high_volatility_rule(threshold: float = 0.03) -> AlertRule:
    """Alert when volatility spikes."""
    return AlertRule(
        name="high_volatility",
        condition=lambda m: m.get("current_volatility", 0) >= threshold,
        level=AlertLevel.WARNING,
        message_template="High volatility: {current_volatility:.4f}",
        cooldown_seconds=1800,
    )
