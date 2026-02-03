"""
Monitoring Daemon
=================
Health monitoring and alerting for long-running experiments.

Features:
- System health monitoring (CPU, memory, disk)
- Process health monitoring
- Alert management (email, webhook, file)
- Metrics collection and logging
- Dashboard data generation
"""

import os
import sys
import json
import time
import logging
import threading
import smtplib
import requests
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from collections import deque

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None

logger = logging.getLogger("GigaTrader.Monitoring")


class AlertSeverity(Enum):
    """Severity levels for alerts."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert delivery channels."""
    FILE = "file"
    EMAIL = "email"
    WEBHOOK = "webhook"
    CONSOLE = "console"


@dataclass
class Alert:
    """An alert notification."""
    alert_id: str
    severity: AlertSeverity
    title: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)
    acknowledged: bool = False

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["severity"] = self.severity.value
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    timestamp: datetime = field(default_factory=datetime.now)
    tags: Dict[str, str] = field(default_factory=dict)


@dataclass
class HealthStatus:
    """System health status."""
    healthy: bool = True
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    disk_percent: float = 0.0
    process_count: int = 0
    uptime_seconds: float = 0.0
    last_check: datetime = field(default_factory=datetime.now)
    issues: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["last_check"] = self.last_check.isoformat()
        return d


class AlertManager:
    """
    Manages alert creation and delivery.

    Supports multiple channels:
    - File: Write alerts to JSON file
    - Email: Send via SMTP
    - Webhook: POST to URL
    - Console: Print to stdout
    """

    def __init__(
        self,
        alert_file: Optional[Path] = None,
        email_config: Optional[Dict] = None,
        webhook_url: Optional[str] = None,
        channels: Optional[List[AlertChannel]] = None,
    ):
        self.alert_file = alert_file
        self.email_config = email_config or {}
        self.webhook_url = webhook_url
        self.channels = channels or [AlertChannel.FILE, AlertChannel.CONSOLE]

        self.alerts: deque = deque(maxlen=1000)  # Keep last 1000 alerts
        self._lock = threading.Lock()
        self._alert_counter = 0

        # Alert throttling (prevent spam)
        self._last_alerts: Dict[str, datetime] = {}
        self.throttle_seconds = 300  # 5 minute throttle per alert type

    def send_alert(
        self,
        severity: AlertSeverity,
        title: str,
        message: str,
        source: str = "system",
        metadata: Optional[Dict] = None,
        throttle_key: Optional[str] = None,
    ) -> Optional[Alert]:
        """
        Send an alert through configured channels.

        Args:
            severity: Alert severity level
            title: Short alert title
            message: Detailed message
            source: Source component
            metadata: Additional data
            throttle_key: Key for throttling duplicate alerts

        Returns:
            The created Alert, or None if throttled
        """
        # Check throttling
        throttle_key = throttle_key or f"{source}:{title}"
        if throttle_key in self._last_alerts:
            elapsed = (datetime.now() - self._last_alerts[throttle_key]).total_seconds()
            if elapsed < self.throttle_seconds:
                logger.debug(f"Alert throttled: {throttle_key}")
                return None

        with self._lock:
            self._alert_counter += 1
            alert = Alert(
                alert_id=f"alert_{self._alert_counter:06d}",
                severity=severity,
                title=title,
                message=message,
                source=source,
                metadata=metadata or {},
            )

            self.alerts.append(alert)
            self._last_alerts[throttle_key] = datetime.now()

        # Deliver to channels
        for channel in self.channels:
            try:
                if channel == AlertChannel.FILE:
                    self._send_to_file(alert)
                elif channel == AlertChannel.EMAIL:
                    self._send_to_email(alert)
                elif channel == AlertChannel.WEBHOOK:
                    self._send_to_webhook(alert)
                elif channel == AlertChannel.CONSOLE:
                    self._send_to_console(alert)
            except Exception as e:
                logger.error(f"Failed to send alert to {channel.value}: {e}")

        return alert

    def get_recent_alerts(
        self,
        count: int = 50,
        severity: Optional[AlertSeverity] = None,
        source: Optional[str] = None,
    ) -> List[Alert]:
        """Get recent alerts with optional filtering."""
        with self._lock:
            alerts = list(self.alerts)

        if severity:
            alerts = [a for a in alerts if a.severity == severity]

        if source:
            alerts = [a for a in alerts if a.source == source]

        return alerts[-count:]

    def acknowledge_alert(self, alert_id: str) -> bool:
        """Mark an alert as acknowledged."""
        with self._lock:
            for alert in self.alerts:
                if alert.alert_id == alert_id:
                    alert.acknowledged = True
                    return True
        return False

    def _send_to_file(self, alert: Alert):
        """Write alert to JSON file."""
        if not self.alert_file:
            return

        self.alert_file.parent.mkdir(parents=True, exist_ok=True)

        # Append to alerts file
        alerts_data = []
        if self.alert_file.exists():
            try:
                with open(self.alert_file, "r") as f:
                    alerts_data = json.load(f)
            except (json.JSONDecodeError, IOError, OSError):
                alerts_data = []

        alerts_data.append(alert.to_dict())

        # Keep last 1000 alerts
        alerts_data = alerts_data[-1000:]

        with open(self.alert_file, "w") as f:
            json.dump(alerts_data, f, indent=2)

    def _send_to_email(self, alert: Alert):
        """Send alert via email."""
        if not self.email_config.get("smtp_host"):
            return

        # Only send email for ERROR and CRITICAL
        if alert.severity not in [AlertSeverity.ERROR, AlertSeverity.CRITICAL]:
            return

        try:
            msg = MIMEMultipart()
            msg["From"] = self.email_config.get("from_addr", "alerts@gigatrader.local")
            msg["To"] = self.email_config.get("to_addr")
            msg["Subject"] = f"[{alert.severity.value.upper()}] {alert.title}"

            body = f"""
            Alert: {alert.title}
            Severity: {alert.severity.value}
            Source: {alert.source}
            Time: {alert.timestamp.isoformat()}

            Message:
            {alert.message}

            Metadata:
            {json.dumps(alert.metadata, indent=2)}
            """

            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(
                self.email_config.get("smtp_host"),
                self.email_config.get("smtp_port", 587),
            ) as server:
                if self.email_config.get("smtp_tls", True):
                    server.starttls()
                if self.email_config.get("smtp_user"):
                    server.login(
                        self.email_config["smtp_user"],
                        self.email_config["smtp_password"],
                    )
                server.send_message(msg)

            logger.info(f"Alert email sent: {alert.alert_id}")

        except Exception as e:
            logger.error(f"Failed to send email: {e}")

    def _send_to_webhook(self, alert: Alert):
        """Send alert to webhook URL."""
        if not self.webhook_url:
            return

        try:
            response = requests.post(
                self.webhook_url,
                json=alert.to_dict(),
                timeout=10,
            )
            response.raise_for_status()
            logger.debug(f"Alert webhook sent: {alert.alert_id}")

        except Exception as e:
            logger.error(f"Failed to send webhook: {e}")

    def _send_to_console(self, alert: Alert):
        """Print alert to console."""
        severity_colors = {
            AlertSeverity.INFO: "",
            AlertSeverity.WARNING: "[WARN]",
            AlertSeverity.ERROR: "[ERROR]",
            AlertSeverity.CRITICAL: "[CRITICAL]",
        }
        prefix = severity_colors.get(alert.severity, "")
        print(f"{prefix} [{alert.source}] {alert.title}: {alert.message}")


class MonitoringDaemon:
    """
    Background daemon for system monitoring.

    Monitors:
    - System resources (CPU, memory, disk)
    - Process health
    - Experiment progress
    - Custom metrics
    """

    def __init__(
        self,
        alert_manager: AlertManager,
        metrics_file: Optional[Path] = None,
        check_interval: int = 60,  # seconds
        thresholds: Optional[Dict] = None,
    ):
        self.alert_manager = alert_manager
        self.metrics_file = metrics_file
        self.check_interval = check_interval

        # Thresholds for alerts
        self.thresholds = thresholds or {
            "cpu_warning": 80,
            "cpu_critical": 95,
            "memory_warning": 80,
            "memory_critical": 95,
            "disk_warning": 85,
            "disk_critical": 95,
            "experiment_stuck_minutes": 60,
            "no_progress_hours": 6,
        }

        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._start_time: Optional[datetime] = None
        self._last_health: HealthStatus = HealthStatus()
        self._metrics: deque = deque(maxlen=10000)
        self._lock = threading.Lock()

        # Custom health checks
        self._custom_checks: List[Callable[[], Dict]] = []

        # External state references
        self._process_manager = None
        self._grid_controller = None
        self._state_manager = None

    def set_components(
        self,
        process_manager=None,
        grid_controller=None,
        state_manager=None,
    ):
        """Set references to monitored components."""
        self._process_manager = process_manager
        self._grid_controller = grid_controller
        self._state_manager = state_manager

    def add_custom_check(self, check_func: Callable[[], Dict]):
        """Add a custom health check function."""
        self._custom_checks.append(check_func)

    def start(self):
        """Start the monitoring daemon."""
        if self._running:
            return

        self._running = True
        self._start_time = datetime.now()

        self._thread = threading.Thread(
            target=self._monitor_loop,
            daemon=True,
        )
        self._thread.start()

        logger.info("Monitoring daemon started")
        self.alert_manager.send_alert(
            AlertSeverity.INFO,
            "Monitoring Started",
            "The monitoring daemon has started successfully.",
            source="monitor",
        )

    def stop(self):
        """Stop the monitoring daemon."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)

        logger.info("Monitoring daemon stopped")

    def get_health(self) -> HealthStatus:
        """Get current health status."""
        return self._last_health

    def record_metric(self, name: str, value: float, tags: Optional[Dict] = None):
        """Record a custom metric."""
        with self._lock:
            metric = MetricPoint(
                name=name,
                value=value,
                tags=tags or {},
            )
            self._metrics.append(metric)

    def get_metrics(
        self,
        name: Optional[str] = None,
        since: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Get recorded metrics."""
        with self._lock:
            metrics = list(self._metrics)

        if name:
            metrics = [m for m in metrics if m.name == name]

        if since:
            metrics = [m for m in metrics if m.timestamp >= since]

        return [asdict(m) for m in metrics[-limit:]]

    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get data for dashboard display."""
        uptime = 0
        if self._start_time:
            uptime = (datetime.now() - self._start_time).total_seconds()

        # Collect component stats
        process_stats = {}
        if self._process_manager:
            process_stats = self._process_manager.get_stats()

        grid_stats = {}
        if self._grid_controller:
            grid_stats = self._grid_controller.get_stats()

        state_stats = {}
        if self._state_manager:
            state_stats = self._state_manager.get_stats()

        return {
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": uptime,
            "uptime_formatted": self._format_duration(uptime),
            "health": self._last_health.to_dict(),
            "process_manager": process_stats,
            "grid_search": grid_stats,
            "state_manager": state_stats,
            "recent_alerts": [a.to_dict() for a in self.alert_manager.get_recent_alerts(20)],
            "metrics_summary": self._get_metrics_summary(),
        }

    def _monitor_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                # Check system health
                health = self._check_system_health()
                self._last_health = health

                # Check for issues
                self._check_thresholds(health)

                # Check component health
                self._check_component_health()

                # Run custom checks
                for check in self._custom_checks:
                    try:
                        result = check()
                        if result.get("alert"):
                            self.alert_manager.send_alert(
                                AlertSeverity[result.get("severity", "WARNING")],
                                result.get("title", "Custom Check Alert"),
                                result.get("message", ""),
                                source="custom_check",
                            )
                    except Exception as e:
                        logger.error(f"Custom check error: {e}")

                # Save metrics
                self._save_metrics()

                # Log status
                if not health.healthy:
                    logger.warning(f"System unhealthy: {health.issues}")

            except Exception as e:
                logger.error(f"Monitor loop error: {e}")

            # Sleep in increments for responsive shutdown
            for _ in range(self.check_interval):
                if not self._running:
                    break
                time.sleep(1)

    def _check_system_health(self) -> HealthStatus:
        """Check system resource health."""
        health = HealthStatus()

        if HAS_PSUTIL:
            try:
                health.cpu_percent = psutil.cpu_percent(interval=1)
                health.memory_percent = psutil.virtual_memory().percent
                health.disk_percent = psutil.disk_usage("/").percent
                health.process_count = len(psutil.pids())
            except Exception as e:
                logger.error(f"Failed to get system stats: {e}")
                health.issues.append(f"System stats unavailable: {e}")

        if self._start_time:
            health.uptime_seconds = (datetime.now() - self._start_time).total_seconds()

        # Determine overall health
        if health.cpu_percent > self.thresholds["cpu_critical"]:
            health.healthy = False
            health.issues.append(f"CPU critical: {health.cpu_percent}%")

        if health.memory_percent > self.thresholds["memory_critical"]:
            health.healthy = False
            health.issues.append(f"Memory critical: {health.memory_percent}%")

        if health.disk_percent > self.thresholds["disk_critical"]:
            health.healthy = False
            health.issues.append(f"Disk critical: {health.disk_percent}%")

        # Record metrics
        self.record_metric("cpu_percent", health.cpu_percent)
        self.record_metric("memory_percent", health.memory_percent)
        self.record_metric("disk_percent", health.disk_percent)

        return health

    def _check_thresholds(self, health: HealthStatus):
        """Check thresholds and send alerts."""
        # CPU alerts
        if health.cpu_percent > self.thresholds["cpu_critical"]:
            self.alert_manager.send_alert(
                AlertSeverity.CRITICAL,
                "CPU Critical",
                f"CPU usage is at {health.cpu_percent}%",
                source="monitor",
                throttle_key="cpu_critical",
            )
        elif health.cpu_percent > self.thresholds["cpu_warning"]:
            self.alert_manager.send_alert(
                AlertSeverity.WARNING,
                "CPU Warning",
                f"CPU usage is at {health.cpu_percent}%",
                source="monitor",
                throttle_key="cpu_warning",
            )

        # Memory alerts
        if health.memory_percent > self.thresholds["memory_critical"]:
            self.alert_manager.send_alert(
                AlertSeverity.CRITICAL,
                "Memory Critical",
                f"Memory usage is at {health.memory_percent}%",
                source="monitor",
                throttle_key="memory_critical",
            )
        elif health.memory_percent > self.thresholds["memory_warning"]:
            self.alert_manager.send_alert(
                AlertSeverity.WARNING,
                "Memory Warning",
                f"Memory usage is at {health.memory_percent}%",
                source="monitor",
                throttle_key="memory_warning",
            )

        # Disk alerts
        if health.disk_percent > self.thresholds["disk_critical"]:
            self.alert_manager.send_alert(
                AlertSeverity.CRITICAL,
                "Disk Critical",
                f"Disk usage is at {health.disk_percent}%",
                source="monitor",
                throttle_key="disk_critical",
            )
        elif health.disk_percent > self.thresholds["disk_warning"]:
            self.alert_manager.send_alert(
                AlertSeverity.WARNING,
                "Disk Warning",
                f"Disk usage is at {health.disk_percent}%",
                source="monitor",
                throttle_key="disk_warning",
            )

    def _check_component_health(self):
        """Check health of monitored components."""
        # Check process manager
        if self._process_manager:
            stats = self._process_manager.get_stats()
            if stats["manager"]["consecutive_crashes"] >= 3:
                self.alert_manager.send_alert(
                    AlertSeverity.ERROR,
                    "Process Crashes",
                    f"Multiple consecutive process crashes: {stats['manager']['consecutive_crashes']}",
                    source="process_manager",
                )

        # Check grid search progress
        if self._grid_controller:
            stats = self._grid_controller.get_stats()
            # Record metrics
            self.record_metric("experiments_completed", stats["completed"])
            self.record_metric("best_score", stats["best_score"])

    def _save_metrics(self):
        """Save metrics to file."""
        if not self.metrics_file:
            return

        try:
            self.metrics_file.parent.mkdir(parents=True, exist_ok=True)

            with self._lock:
                metrics_data = [
                    {
                        "name": m.name,
                        "value": m.value,
                        "timestamp": m.timestamp.isoformat(),
                        "tags": m.tags,
                    }
                    for m in list(self._metrics)[-1000:]
                ]

            with open(self.metrics_file, "w") as f:
                json.dump(metrics_data, f, indent=2)

        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")

    def _get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary statistics for metrics."""
        with self._lock:
            metrics = list(self._metrics)

        if not metrics:
            return {}

        # Group by name
        by_name = {}
        for m in metrics[-100:]:  # Last 100 points
            if m.name not in by_name:
                by_name[m.name] = []
            by_name[m.name].append(m.value)

        summary = {}
        for name, values in by_name.items():
            summary[name] = {
                "min": min(values),
                "max": max(values),
                "avg": sum(values) / len(values),
                "count": len(values),
            }

        return summary

    @staticmethod
    def _format_duration(seconds: float) -> str:
        """Format duration as human-readable string."""
        days = int(seconds // 86400)
        hours = int((seconds % 86400) // 3600)
        minutes = int((seconds % 3600) // 60)

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")

        return " ".join(parts) if parts else "0m"


# =============================================================================
# MAIN ENTRY POINT FOR TESTING
# =============================================================================

def main():
    """Test the monitoring daemon."""
    from pathlib import Path

    logs_dir = Path(__file__).parent.parent.parent / "logs"

    alert_manager = AlertManager(
        alert_file=logs_dir / "alerts.json",
        channels=[AlertChannel.FILE, AlertChannel.CONSOLE],
    )

    monitor = MonitoringDaemon(
        alert_manager=alert_manager,
        metrics_file=logs_dir / "metrics.json",
        check_interval=10,
    )

    monitor.start()

    print("Monitoring started. Press Ctrl+C to stop.")

    try:
        while True:
            time.sleep(5)
            print(f"\nHealth: {json.dumps(monitor.get_health().to_dict(), indent=2)}")
    except KeyboardInterrupt:
        monitor.stop()


if __name__ == "__main__":
    main()
