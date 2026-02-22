"""
Health Checker & Alert Manager
==============================
Production monitoring for the giga_trader system.

Implements health checks from config/monitoring.yaml:
- Data freshness checks
- Model staleness detection
- API connectivity verification
- Disk space monitoring
- Performance metric thresholds

Alert channels: logging (always), email, Slack webhook, Discord webhook.
"""

import os
import json
import time
import shutil
import logging
import smtplib
import threading
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from enum import Enum

import numpy as np

from src.core.state_manager import atomic_write_json

logger = logging.getLogger("HEALTH_CHECKER")

project_root = Path(__file__).parent.parent.parent


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class HealthStatus(str, Enum):
    """Component health status."""
    HEALTHY = "HEALTHY"
    DEGRADED = "DEGRADED"
    UNHEALTHY = "UNHEALTHY"
    UNKNOWN = "UNKNOWN"


@dataclass
class HealthCheckResult:
    """Result of a single health check."""
    name: str
    status: HealthStatus
    message: str
    value: Optional[float] = None
    threshold: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Alert:
    """A triggered alert."""
    severity: AlertSeverity
    title: str
    message: str
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    sent_channels: List[str] = field(default_factory=list)


class AlertManager:
    """
    Centralized alert routing with rate limiting.

    Supports channels:
    - Logging (always active)
    - Email (SMTP)
    - Slack webhook
    - Discord webhook

    Rate limiting prevents alert storms (configurable cooldown per alert type).
    """

    def __init__(
        self,
        email_config: Optional[Dict] = None,
        slack_webhook_url: Optional[str] = None,
        discord_webhook_url: Optional[str] = None,
        cooldown_seconds: int = 300,
        max_alerts_per_hour: int = 20,
    ):
        self.email_config = email_config or {}
        self.slack_webhook_url = slack_webhook_url or os.environ.get("SLACK_WEBHOOK_URL")
        self.discord_webhook_url = discord_webhook_url or os.environ.get("DISCORD_WEBHOOK_URL")
        self.cooldown_seconds = cooldown_seconds
        self.max_alerts_per_hour = max_alerts_per_hour

        self._last_alert_time: Dict[str, datetime] = {}
        self._alerts_this_hour: int = 0
        self._hour_start: datetime = datetime.now()
        self._alert_history: List[Alert] = []
        self._lock = threading.Lock()

    def send_alert(self, alert: Alert) -> bool:
        """
        Send an alert through configured channels with rate limiting.

        Returns True if alert was sent, False if rate-limited.
        """
        with self._lock:
            # Rate limit check
            now = datetime.now()

            # Reset hourly counter
            if (now - self._hour_start).total_seconds() > 3600:
                self._alerts_this_hour = 0
                self._hour_start = now

            if self._alerts_this_hour >= self.max_alerts_per_hour:
                logger.warning(f"Alert rate limit reached ({self.max_alerts_per_hour}/hour)")
                return False

            # Cooldown check for same alert type
            alert_key = f"{alert.source}:{alert.title}"
            if alert_key in self._last_alert_time:
                elapsed = (now - self._last_alert_time[alert_key]).total_seconds()
                if elapsed < self.cooldown_seconds:
                    logger.debug(f"Alert {alert_key} in cooldown ({elapsed:.0f}s < {self.cooldown_seconds}s)")
                    return False

            # Send to all channels
            self._send_to_log(alert)

            if alert.severity in (AlertSeverity.CRITICAL, AlertSeverity.EMERGENCY):
                self._send_to_email(alert)
                self._send_to_slack(alert)
                self._send_to_discord(alert)
            elif alert.severity == AlertSeverity.WARNING:
                self._send_to_slack(alert)
                self._send_to_discord(alert)

            # Update tracking
            self._last_alert_time[alert_key] = now
            self._alerts_this_hour += 1
            self._alert_history.append(alert)

            # Trim history
            if len(self._alert_history) > 1000:
                self._alert_history = self._alert_history[-500:]

            return True

    def _send_to_log(self, alert: Alert):
        """Always log alerts."""
        log_method = {
            AlertSeverity.INFO: logger.info,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.CRITICAL: logger.critical,
            AlertSeverity.EMERGENCY: logger.critical,
        }.get(alert.severity, logger.info)

        log_method(f"[ALERT:{alert.severity.value}] {alert.title}: {alert.message}")
        alert.sent_channels.append("log")

    def _send_to_email(self, alert: Alert):
        """Send alert via email."""
        if not self.email_config.get("smtp_host"):
            return

        try:
            msg = MIMEMultipart()
            msg["From"] = self.email_config.get("from_email", "giga_trader@localhost")
            msg["To"] = self.email_config.get("to_email", os.environ.get("ALERT_EMAIL", ""))
            msg["Subject"] = f"[GigaTrader {alert.severity.value}] {alert.title}"

            body = f"""
Giga Trader Alert
=================
Severity: {alert.severity.value}
Time: {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
Source: {alert.source}

{alert.message}

Metadata: {json.dumps(alert.metadata, indent=2, default=str)}
"""
            msg.attach(MIMEText(body, "plain"))

            with smtplib.SMTP(
                self.email_config["smtp_host"],
                self.email_config.get("smtp_port", 587),
            ) as server:
                server.starttls()
                if self.email_config.get("smtp_user"):
                    server.login(
                        self.email_config["smtp_user"],
                        self.email_config.get("smtp_password", ""),
                    )
                server.send_message(msg)

            alert.sent_channels.append("email")
            logger.debug(f"Email alert sent: {alert.title}")
        except Exception as e:
            logger.error(f"Failed to send email alert: {e}")

    def _send_to_slack(self, alert: Alert):
        """Send alert via Slack webhook."""
        if not self.slack_webhook_url:
            return

        try:
            import urllib.request

            emoji = {
                AlertSeverity.INFO: ":information_source:",
                AlertSeverity.WARNING: ":warning:",
                AlertSeverity.CRITICAL: ":rotating_light:",
                AlertSeverity.EMERGENCY: ":fire:",
            }.get(alert.severity, ":bell:")

            payload = {
                "text": f"{emoji} *[{alert.severity.value}] {alert.title}*\n{alert.message}",
                "username": "GigaTrader Bot",
            }

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.slack_webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
            alert.sent_channels.append("slack")
            logger.debug(f"Slack alert sent: {alert.title}")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")

    def _send_to_discord(self, alert: Alert):
        """Send alert via Discord webhook."""
        if not self.discord_webhook_url:
            return

        try:
            import urllib.request

            color = {
                AlertSeverity.INFO: 3447003,       # Blue
                AlertSeverity.WARNING: 16776960,    # Yellow
                AlertSeverity.CRITICAL: 15158332,   # Red
                AlertSeverity.EMERGENCY: 10038562,  # Dark red
            }.get(alert.severity, 3447003)

            payload = {
                "embeds": [{
                    "title": f"[{alert.severity.value}] {alert.title}",
                    "description": alert.message,
                    "color": color,
                    "timestamp": alert.timestamp.isoformat(),
                    "footer": {"text": f"Source: {alert.source}"},
                }]
            }

            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                self.discord_webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
            )
            urllib.request.urlopen(req, timeout=10)
            alert.sent_channels.append("discord")
            logger.debug(f"Discord alert sent: {alert.title}")
        except Exception as e:
            logger.error(f"Failed to send Discord alert: {e}")

    def get_recent_alerts(self, count: int = 50) -> List[Alert]:
        """Get recent alert history."""
        return self._alert_history[-count:]

    def get_alert_stats(self) -> Dict:
        """Get alert statistics."""
        now = datetime.now()
        last_hour = [a for a in self._alert_history if (now - a.timestamp).total_seconds() < 3600]
        last_day = [a for a in self._alert_history if (now - a.timestamp).total_seconds() < 86400]

        return {
            "total_alerts": len(self._alert_history),
            "alerts_last_hour": len(last_hour),
            "alerts_last_day": len(last_day),
            "by_severity": {
                s.value: sum(1 for a in last_day if a.severity == s)
                for s in AlertSeverity
            },
        }


class HealthChecker:
    """
    System health monitoring with configurable checks.

    Runs periodic health checks and triggers alerts when thresholds are breached.

    Checks:
    - Data freshness: Is cached data up-to-date?
    - Model staleness: When were models last trained?
    - API connectivity: Can we reach Alpaca?
    - Disk space: Enough storage for logs/models?
    - Performance metrics: Are model metrics degrading?
    - Process health: Is the trading bot running?
    """

    def __init__(
        self,
        alert_manager: Optional[AlertManager] = None,
        check_interval_seconds: int = 60,
        config: Optional[Dict] = None,
    ):
        self.alert_manager = alert_manager or AlertManager()
        self.check_interval_seconds = check_interval_seconds
        self.config = config or self._default_config()

        self._checks: Dict[str, Callable] = {}
        self._results: Dict[str, HealthCheckResult] = {}
        self._results_lock = threading.Lock()
        self._running = False
        self._thread: Optional[threading.Thread] = None

        # Register default checks
        self._register_default_checks()

    def _default_config(self) -> Dict:
        """Default health check configuration."""
        return {
            "data_freshness_max_age_hours": 24,
            "model_staleness_max_days": 7,
            "min_disk_space_gb": 1.0,
            "min_model_auc": 0.55,
            "auc_warning_threshold": 0.58,
            "max_daily_loss_pct": 2.0,
            "max_drawdown_pct": 10.0,
            "status_file": str(project_root / "logs" / "status.json"),
            "models_dir": str(project_root / "models" / "production"),
            "data_cache_dir": str(project_root / "data" / "cache"),
        }

    def _register_default_checks(self):
        """Register all default health checks."""
        self.register_check("data_freshness", self._check_data_freshness)
        self.register_check("model_staleness", self._check_model_staleness)
        self.register_check("disk_space", self._check_disk_space)
        self.register_check("status_file", self._check_status_file)
        self.register_check("model_performance", self._check_model_performance)
        self.register_check("api_connectivity", self._check_api_connectivity)
        self.register_check("process_memory", self._check_process_memory)

    def register_check(self, name: str, check_fn: Callable[[], HealthCheckResult]):
        """Register a custom health check."""
        self._checks[name] = check_fn
        logger.debug(f"Registered health check: {name}")

    def _check_data_freshness(self) -> HealthCheckResult:
        """Check if cached data is fresh enough."""
        cache_dir = Path(self.config["data_cache_dir"])
        metadata_path = cache_dir / "data_metadata.json"

        if not metadata_path.exists():
            return HealthCheckResult(
                name="data_freshness",
                status=HealthStatus.UNKNOWN,
                message="No data metadata found",
            )

        try:
            with open(metadata_path) as f:
                metadata = json.load(f)

            last_update = metadata.get("last_download_date", "")
            if last_update:
                last_dt = datetime.fromisoformat(last_update)
                age_hours = (datetime.now() - last_dt).total_seconds() / 3600
                max_age = self.config["data_freshness_max_age_hours"]

                if age_hours > max_age * 2:
                    return HealthCheckResult(
                        name="data_freshness",
                        status=HealthStatus.UNHEALTHY,
                        message=f"Data is {age_hours:.1f}h old (limit: {max_age}h)",
                        value=age_hours,
                        threshold=max_age,
                    )
                elif age_hours > max_age:
                    return HealthCheckResult(
                        name="data_freshness",
                        status=HealthStatus.DEGRADED,
                        message=f"Data is {age_hours:.1f}h old (limit: {max_age}h)",
                        value=age_hours,
                        threshold=max_age,
                    )
                else:
                    return HealthCheckResult(
                        name="data_freshness",
                        status=HealthStatus.HEALTHY,
                        message=f"Data is {age_hours:.1f}h old",
                        value=age_hours,
                        threshold=max_age,
                    )
        except Exception as e:
            return HealthCheckResult(
                name="data_freshness",
                status=HealthStatus.UNKNOWN,
                message=f"Error checking data freshness: {e}",
            )

        return HealthCheckResult(
            name="data_freshness",
            status=HealthStatus.UNKNOWN,
            message="Could not determine data freshness",
        )

    def _check_model_staleness(self) -> HealthCheckResult:
        """Check if production models are recent enough."""
        models_dir = Path(self.config["models_dir"])

        if not models_dir.exists():
            return HealthCheckResult(
                name="model_staleness",
                status=HealthStatus.UNHEALTHY,
                message="Models directory does not exist",
            )

        model_files = list(models_dir.glob("*.joblib"))
        if not model_files:
            return HealthCheckResult(
                name="model_staleness",
                status=HealthStatus.UNHEALTHY,
                message="No model files found",
            )

        # Find newest model
        newest = max(model_files, key=lambda f: f.stat().st_mtime)
        age_days = (time.time() - newest.stat().st_mtime) / 86400
        max_days = self.config["model_staleness_max_days"]

        if age_days > max_days * 2:
            status = HealthStatus.UNHEALTHY
        elif age_days > max_days:
            status = HealthStatus.DEGRADED
        else:
            status = HealthStatus.HEALTHY

        return HealthCheckResult(
            name="model_staleness",
            status=status,
            message=f"Newest model is {age_days:.1f} days old ({newest.name})",
            value=age_days,
            threshold=max_days,
            metadata={"newest_model": newest.name, "model_count": len(model_files)},
        )

    def _check_disk_space(self) -> HealthCheckResult:
        """Check available disk space."""
        try:
            usage = shutil.disk_usage(str(project_root))
            free_gb = usage.free / (1024 ** 3)
            total_gb = usage.total / (1024 ** 3)
            min_gb = self.config["min_disk_space_gb"]

            if free_gb < min_gb:
                status = HealthStatus.UNHEALTHY
            elif free_gb < min_gb * 3:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY

            return HealthCheckResult(
                name="disk_space",
                status=status,
                message=f"{free_gb:.1f}GB free of {total_gb:.1f}GB total",
                value=free_gb,
                threshold=min_gb,
            )
        except Exception as e:
            return HealthCheckResult(
                name="disk_space",
                status=HealthStatus.UNKNOWN,
                message=f"Error checking disk space: {e}",
            )

    def _check_status_file(self) -> HealthCheckResult:
        """Check the system status file for issues."""
        status_path = Path(self.config["status_file"])

        if not status_path.exists():
            return HealthCheckResult(
                name="status_file",
                status=HealthStatus.DEGRADED,
                message="Status file not found",
            )

        try:
            with open(status_path) as f:
                status = json.load(f)

            # Check for error states
            mode = status.get("mode", "UNKNOWN")
            errors = status.get("errors", [])

            if errors:
                return HealthCheckResult(
                    name="status_file",
                    status=HealthStatus.DEGRADED,
                    message=f"Mode: {mode}, {len(errors)} errors recorded",
                    metadata={"mode": mode, "error_count": len(errors)},
                )

            return HealthCheckResult(
                name="status_file",
                status=HealthStatus.HEALTHY,
                message=f"Mode: {mode}, no errors",
                metadata={"mode": mode},
            )
        except Exception as e:
            return HealthCheckResult(
                name="status_file",
                status=HealthStatus.UNKNOWN,
                message=f"Error reading status file: {e}",
            )

    def _check_model_performance(self) -> HealthCheckResult:
        """Check model performance metrics from SQLite registry."""
        try:
            from src.core.registry_db import get_registry_db
            db = get_registry_db()
            stats = db.get_model_statistics()

            total = stats.get("total_models", 0)
            if total == 0:
                return HealthCheckResult(
                    name="model_performance",
                    status=HealthStatus.UNKNOWN,
                    message="No models in registry",
                )

            best_auc = stats.get("best_cv_auc", 0)
            avg_auc = stats.get("avg_cv_auc", 0)
            by_tier = stats.get("by_tier", {})
            min_auc = self.config["min_model_auc"]
            warn_auc = self.config["auc_warning_threshold"]

            if best_auc < min_auc:
                status = HealthStatus.UNHEALTHY
            elif best_auc < warn_auc:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY

            tier_summary = ", ".join(f"T{t}:{c}" for t, c in sorted(by_tier.items()))
            return HealthCheckResult(
                name="model_performance",
                status=status,
                message=f"Best AUC: {best_auc:.3f}, Avg: {avg_auc:.3f} ({total} models, {tier_summary})",
                value=best_auc,
                threshold=min_auc,
                metadata={"model_count": total, "avg_auc": avg_auc, "by_tier": by_tier},
            )
        except Exception as e:
            return HealthCheckResult(
                name="model_performance",
                status=HealthStatus.UNKNOWN,
                message=f"Error checking model performance: {e}",
            )

    def _check_process_memory(self) -> HealthCheckResult:
        """Check process memory usage.

        Wave 28: Monitors RAM usage to detect memory leaks during long runs.
        Alerts if process exceeds 2GB.
        """
        try:
            import psutil
            process = psutil.Process(os.getpid())
            mem_info = process.memory_info()
            rss_mb = mem_info.rss / (1024 * 1024)

            # Also check all child processes
            children = process.children(recursive=True)
            total_mb = rss_mb + sum(
                c.memory_info().rss / (1024 * 1024) for c in children
            )

            max_mb = self.config.get("max_process_memory_mb", 4096)

            if total_mb > max_mb:
                status = HealthStatus.UNHEALTHY
            elif total_mb > max_mb * 0.75:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY

            return HealthCheckResult(
                name="process_memory",
                status=status,
                message=f"Main: {rss_mb:.0f}MB, Total (+ {len(children)} children): {total_mb:.0f}MB",
                value=total_mb,
                threshold=max_mb,
                metadata={
                    "rss_mb": rss_mb,
                    "total_mb": total_mb,
                    "child_count": len(children),
                },
            )
        except ImportError:
            return HealthCheckResult(
                name="process_memory",
                status=HealthStatus.UNKNOWN,
                message="psutil not installed - memory monitoring unavailable",
            )
        except Exception as e:
            return HealthCheckResult(
                name="process_memory",
                status=HealthStatus.UNKNOWN,
                message=f"Error checking memory: {e}",
            )

    def _check_api_connectivity(self) -> HealthCheckResult:
        """Check Alpaca API connectivity."""
        try:
            from dotenv import load_dotenv
            load_dotenv(project_root / ".env")

            api_key = os.environ.get("ALPACA_API_KEY", "")
            if not api_key:
                return HealthCheckResult(
                    name="api_connectivity",
                    status=HealthStatus.DEGRADED,
                    message="No Alpaca API key configured",
                )

            # Quick HTTP check to Alpaca
            import urllib.request
            base_url = os.environ.get("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
            req = urllib.request.Request(
                f"{base_url}/v2/account",
                headers={
                    "APCA-API-KEY-ID": api_key,
                    "APCA-API-SECRET-KEY": os.environ.get("ALPACA_SECRET_KEY", ""),
                },
            )

            start = time.time()
            response = urllib.request.urlopen(req, timeout=10)
            latency_ms = (time.time() - start) * 1000

            if response.status == 200:
                return HealthCheckResult(
                    name="api_connectivity",
                    status=HealthStatus.HEALTHY,
                    message=f"Alpaca API connected ({latency_ms:.0f}ms)",
                    value=latency_ms,
                    threshold=5000,
                )
            else:
                return HealthCheckResult(
                    name="api_connectivity",
                    status=HealthStatus.DEGRADED,
                    message=f"Alpaca API returned status {response.status}",
                )
        except Exception as e:
            return HealthCheckResult(
                name="api_connectivity",
                status=HealthStatus.UNHEALTHY,
                message=f"Alpaca API unreachable: {e}",
            )

    def run_all_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}
        for name, check_fn in self._checks.items():
            try:
                result = check_fn()
                results[name] = result

                # Trigger alerts for unhealthy checks
                if result.status == HealthStatus.UNHEALTHY:
                    self.alert_manager.send_alert(Alert(
                        severity=AlertSeverity.CRITICAL,
                        title=f"Health Check Failed: {name}",
                        message=result.message,
                        source="health_checker",
                        metadata=result.metadata,
                    ))
                elif result.status == HealthStatus.DEGRADED:
                    self.alert_manager.send_alert(Alert(
                        severity=AlertSeverity.WARNING,
                        title=f"Health Check Degraded: {name}",
                        message=result.message,
                        source="health_checker",
                        metadata=result.metadata,
                    ))

            except Exception as e:
                results[name] = HealthCheckResult(
                    name=name,
                    status=HealthStatus.UNKNOWN,
                    message=f"Check raised exception: {e}",
                )
                logger.error(f"Health check {name} failed: {e}")

        with self._results_lock:
            self._results.update(results)

        return results

    def get_overall_status(self) -> HealthStatus:
        """Get overall system health status."""
        with self._results_lock:
            if not self._results:
                return HealthStatus.UNKNOWN

            statuses = [r.status for r in self._results.values()]

        if any(s == HealthStatus.UNHEALTHY for s in statuses):
            return HealthStatus.UNHEALTHY
        if any(s == HealthStatus.DEGRADED for s in statuses):
            return HealthStatus.DEGRADED
        if all(s == HealthStatus.HEALTHY for s in statuses):
            return HealthStatus.HEALTHY
        return HealthStatus.UNKNOWN

    def get_status_report(self) -> Dict:
        """Get a full status report."""
        with self._results_lock:
            checks_snapshot = dict(self._results)

        return {
            "timestamp": datetime.now().isoformat(),
            "overall_status": self.get_overall_status().value,
            "checks": {
                name: {
                    "status": r.status.value,
                    "message": r.message,
                    "value": r.value,
                    "threshold": r.threshold,
                    "timestamp": r.timestamp.isoformat(),
                }
                for name, r in checks_snapshot.items()
            },
            "alert_stats": self.alert_manager.get_alert_stats(),
        }

    def start_background(self):
        """Start health checks in a background thread."""
        if self._running:
            logger.warning("Health checker already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"Health checker started (interval: {self.check_interval_seconds}s)")

    def stop_background(self):
        """Stop background health checks."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("Health checker stopped")

    def _run_loop(self):
        """Background loop for periodic health checks."""
        while self._running:
            try:
                self.run_all_checks()

                # Save status to file
                status_report = self.get_status_report()
                health_path = project_root / "logs" / "health_status.json"
                atomic_write_json(health_path, status_report)

            except Exception as e:
                logger.error(f"Health check loop error: {e}")

            time.sleep(self.check_interval_seconds)
