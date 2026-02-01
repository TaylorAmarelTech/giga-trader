"""
Alert Dispatcher - Centralized alert dispatching to logs.

This module provides a unified interface for sending alerts from
all supervision components. Currently supports log-based alerts only.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable
from enum import Enum
import logging

logger = logging.getLogger("GigaTrader.Alerts")


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


@dataclass
class Alert:
    """An alert to be dispatched."""
    timestamp: datetime
    severity: AlertSeverity
    title: str
    message: str
    context: Dict = field(default_factory=dict)
    source: str = "supervision"


class AlertDispatcher:
    """
    Dispatches alerts to configured channels.

    Currently supports:
    - Log files (default, always enabled)

    Future support planned for:
    - Email
    - Slack
    - Discord
    - SMS
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        custom_handler: Optional[Callable[[Alert], None]] = None,
    ):
        """
        Initialize AlertDispatcher.

        Args:
            config_path: Path to monitoring.yaml (for future channel config)
            custom_handler: Optional custom handler for alerts
        """
        self._alert_history: List[Alert] = []
        self._max_history = 1000
        self._custom_handler = custom_handler

        # Alert counts for rate limiting
        self._alert_counts: Dict[str, int] = {}
        self._last_alert_time: Dict[str, datetime] = {}
        self._rate_limit_seconds = 60  # Min seconds between identical alerts

    def send(
        self,
        alert: Alert,
    ):
        """
        Send alert to all configured channels.

        Args:
            alert: Alert to send
        """
        # Check rate limiting
        alert_key = f"{alert.severity.value}:{alert.title}"
        if not self._should_send(alert_key):
            return

        # Log the alert
        self._log_alert(alert)

        # Store in history
        self._store_alert(alert)

        # Call custom handler if provided
        if self._custom_handler:
            try:
                self._custom_handler(alert)
            except Exception as e:
                logger.error(f"Custom alert handler failed: {e}")

    def send_simple(
        self,
        severity: str,
        message: str,
        context: Optional[Dict] = None,
    ):
        """
        Send a simple alert (convenience method).

        Args:
            severity: "info", "warning", "critical", or "emergency"
            message: Alert message
            context: Optional context dict
        """
        sev = AlertSeverity(severity) if severity in [s.value for s in AlertSeverity] else AlertSeverity.WARNING

        alert = Alert(
            timestamp=datetime.now(),
            severity=sev,
            title=message[:50],  # First 50 chars as title
            message=message,
            context=context or {},
        )

        self.send(alert)

    def _should_send(self, alert_key: str) -> bool:
        """Check if alert should be sent (rate limiting)."""
        now = datetime.now()

        if alert_key in self._last_alert_time:
            elapsed = (now - self._last_alert_time[alert_key]).total_seconds()
            if elapsed < self._rate_limit_seconds:
                return False

        self._last_alert_time[alert_key] = now
        self._alert_counts[alert_key] = self._alert_counts.get(alert_key, 0) + 1
        return True

    def _log_alert(self, alert: Alert):
        """Log alert to appropriate log level."""
        log_msg = f"[ALERT] [{alert.severity.value.upper()}] {alert.title}: {alert.message}"

        if alert.context:
            log_msg += f" | Context: {alert.context}"

        if alert.severity == AlertSeverity.EMERGENCY:
            logger.critical(log_msg)
        elif alert.severity == AlertSeverity.CRITICAL:
            logger.error(log_msg)
        elif alert.severity == AlertSeverity.WARNING:
            logger.warning(log_msg)
        else:
            logger.info(log_msg)

    def _store_alert(self, alert: Alert):
        """Store alert in history."""
        self._alert_history.append(alert)

        # Trim history if needed
        if len(self._alert_history) > self._max_history:
            self._alert_history = self._alert_history[-self._max_history:]

    def send_position_mismatch(
        self,
        symbol: str,
        alpaca_qty: float,
        internal_qty: float,
        status: str,
    ):
        """Convenience method for position mismatch alerts."""
        alert = Alert(
            timestamp=datetime.now(),
            severity=AlertSeverity.WARNING if abs(alpaca_qty - internal_qty) < 5 else AlertSeverity.CRITICAL,
            title=f"Position Mismatch: {symbol}",
            message=f"Alpaca qty={alpaca_qty}, Internal qty={internal_qty}, Status={status}",
            context={
                "symbol": symbol,
                "alpaca_qty": alpaca_qty,
                "internal_qty": internal_qty,
                "status": status,
            },
            source="position_reconciler",
        )
        self.send(alert)

    def send_circuit_breaker_triggered(
        self,
        breaker_type: str,
        trigger_value: float,
        action: str,
    ):
        """Convenience method for circuit breaker alerts."""
        alert = Alert(
            timestamp=datetime.now(),
            severity=AlertSeverity.CRITICAL,
            title=f"Circuit Breaker: {breaker_type}",
            message=f"Triggered at {trigger_value:.4f}, action: {action}",
            context={
                "breaker_type": breaker_type,
                "trigger_value": trigger_value,
                "action": action,
            },
            source="circuit_breaker",
        )
        self.send(alert)

    def send_force_close_warning(
        self,
        urgency: str,
        positions: List[str],
        minutes_to_deadline: float,
    ):
        """Convenience method for force close warnings."""
        severity = AlertSeverity.CRITICAL if urgency in ["critical", "emergency"] else AlertSeverity.WARNING

        alert = Alert(
            timestamp=datetime.now(),
            severity=severity,
            title=f"Force Close: {urgency.upper()}",
            message=f"{len(positions)} positions to close, {minutes_to_deadline:.1f} min to deadline",
            context={
                "urgency": urgency,
                "positions": positions,
                "minutes_to_deadline": minutes_to_deadline,
            },
            source="force_close",
        )
        self.send(alert)

    def send_model_health_alert(
        self,
        model_name: str,
        issues: List[str],
        confidence: float,
    ):
        """Convenience method for model health alerts."""
        severity = AlertSeverity.CRITICAL if confidence < 0.3 else AlertSeverity.WARNING

        alert = Alert(
            timestamp=datetime.now(),
            severity=severity,
            title=f"Model Health: {model_name}",
            message=f"Issues: {', '.join(issues[:3])}, Confidence: {confidence:.2f}",
            context={
                "model_name": model_name,
                "issues": issues,
                "confidence": confidence,
            },
            source="model_health",
        )
        self.send(alert)

    def send_feature_validation_alert(
        self,
        nan_count: int,
        feature_mismatch: bool,
        missing_features: List[str],
    ):
        """Convenience method for feature validation alerts."""
        alert = Alert(
            timestamp=datetime.now(),
            severity=AlertSeverity.WARNING if not feature_mismatch else AlertSeverity.CRITICAL,
            title="Feature Validation Issue",
            message=f"NaN count={nan_count}, Mismatch={feature_mismatch}, Missing={len(missing_features)}",
            context={
                "nan_count": nan_count,
                "feature_mismatch": feature_mismatch,
                "missing_features": missing_features[:5],
            },
            source="feature_validator",
        )
        self.send(alert)

    def get_recent_alerts(
        self,
        count: int = 10,
        severity: Optional[AlertSeverity] = None,
    ) -> List[Alert]:
        """
        Get recent alerts.

        Args:
            count: Number of alerts to return
            severity: Filter by severity (optional)

        Returns:
            List of recent alerts
        """
        alerts = self._alert_history

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        return alerts[-count:]

    def get_alert_counts(self) -> Dict[str, int]:
        """Get count of alerts by severity."""
        counts = {s.value: 0 for s in AlertSeverity}

        for alert in self._alert_history:
            counts[alert.severity.value] += 1

        return counts

    def clear_history(self):
        """Clear alert history."""
        self._alert_history.clear()
        self._alert_counts.clear()
        self._last_alert_time.clear()
