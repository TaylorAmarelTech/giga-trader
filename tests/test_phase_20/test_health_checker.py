"""
Test HealthChecker, AlertManager, AlertSeverity, and related structures.
"""

import sys
from pathlib import Path
from datetime import datetime

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase_20_monitoring.health_checker import (
    HealthChecker,
    AlertManager,
    AlertSeverity,
    HealthStatus,
    HealthCheckResult,
    Alert,
)


# ---------------------------------------------------------------------------
# AlertSeverity enum tests
# ---------------------------------------------------------------------------

def test_alert_severity_values():
    """AlertSeverity should have INFO, WARNING, CRITICAL, EMERGENCY."""
    assert AlertSeverity.INFO.value == "INFO"
    assert AlertSeverity.WARNING.value == "WARNING"
    assert AlertSeverity.CRITICAL.value == "CRITICAL"
    assert AlertSeverity.EMERGENCY.value == "EMERGENCY"
    assert len(list(AlertSeverity)) == 4


# ---------------------------------------------------------------------------
# HealthStatus enum tests
# ---------------------------------------------------------------------------

def test_health_status_values():
    """HealthStatus should have HEALTHY, DEGRADED, UNHEALTHY, UNKNOWN."""
    assert HealthStatus.HEALTHY.value == "HEALTHY"
    assert HealthStatus.DEGRADED.value == "DEGRADED"
    assert HealthStatus.UNHEALTHY.value == "UNHEALTHY"
    assert HealthStatus.UNKNOWN.value == "UNKNOWN"
    assert len(list(HealthStatus)) == 4


# ---------------------------------------------------------------------------
# HealthCheckResult tests
# ---------------------------------------------------------------------------

def test_health_check_result_creation():
    """HealthCheckResult should initialize with required fields."""
    result = HealthCheckResult(
        name="data_freshness",
        status=HealthStatus.HEALTHY,
        message="Data is fresh",
    )
    assert result.name == "data_freshness"
    assert result.status == HealthStatus.HEALTHY
    assert result.message == "Data is fresh"
    assert result.value is None
    assert result.threshold is None
    assert isinstance(result.timestamp, datetime)
    assert isinstance(result.metadata, dict)


def test_health_check_result_with_values():
    """HealthCheckResult should accept optional value and threshold."""
    result = HealthCheckResult(
        name="model_auc",
        status=HealthStatus.DEGRADED,
        message="AUC below warning threshold",
        value=0.56,
        threshold=0.58,
        metadata={"model_id": "swing_001"},
    )
    assert result.value == 0.56
    assert result.threshold == 0.58
    assert result.metadata["model_id"] == "swing_001"


# ---------------------------------------------------------------------------
# Alert tests
# ---------------------------------------------------------------------------

def test_alert_creation():
    """Alert should initialize with required fields."""
    alert = Alert(
        severity=AlertSeverity.WARNING,
        title="Model Degradation",
        message="Swing model AUC dropped below 0.58",
        source="health_checker",
    )
    assert alert.severity == AlertSeverity.WARNING
    assert alert.title == "Model Degradation"
    assert alert.source == "health_checker"
    assert isinstance(alert.timestamp, datetime)
    assert isinstance(alert.sent_channels, list)
    assert len(alert.sent_channels) == 0


# ---------------------------------------------------------------------------
# AlertManager tests
# ---------------------------------------------------------------------------

def test_alert_manager_creation():
    """AlertManager should initialize with defaults."""
    manager = AlertManager()
    assert isinstance(manager, AlertManager)
    assert manager.cooldown_seconds == 300
    assert manager.max_alerts_per_hour == 20


def test_alert_manager_custom_config():
    """AlertManager should accept custom configuration."""
    manager = AlertManager(
        cooldown_seconds=60,
        max_alerts_per_hour=50,
    )
    assert manager.cooldown_seconds == 60
    assert manager.max_alerts_per_hour == 50


def test_alert_manager_send_alert():
    """AlertManager.send_alert should process an alert (at least to log)."""
    manager = AlertManager()
    alert = Alert(
        severity=AlertSeverity.INFO,
        title="Test Alert",
        message="This is a test",
        source="test",
    )
    result = manager.send_alert(alert)
    assert result is True
    assert "log" in alert.sent_channels


def test_alert_manager_rate_limiting():
    """AlertManager should rate-limit duplicate alerts."""
    manager = AlertManager(cooldown_seconds=300)

    alert1 = Alert(
        severity=AlertSeverity.INFO,
        title="Same Alert",
        message="First",
        source="test",
    )
    alert2 = Alert(
        severity=AlertSeverity.INFO,
        title="Same Alert",
        message="Second",
        source="test",
    )

    result1 = manager.send_alert(alert1)
    result2 = manager.send_alert(alert2)

    assert result1 is True
    assert result2 is False  # Should be rate-limited


def test_alert_manager_different_alerts_not_rate_limited():
    """Different alert types should not rate-limit each other."""
    manager = AlertManager(cooldown_seconds=300)

    alert1 = Alert(
        severity=AlertSeverity.INFO,
        title="Alert Type A",
        message="First",
        source="test_a",
    )
    alert2 = Alert(
        severity=AlertSeverity.WARNING,
        title="Alert Type B",
        message="Second",
        source="test_b",
    )

    result1 = manager.send_alert(alert1)
    result2 = manager.send_alert(alert2)

    assert result1 is True
    assert result2 is True


# ---------------------------------------------------------------------------
# HealthChecker tests
# ---------------------------------------------------------------------------

def test_health_checker_creation():
    """HealthChecker should initialize with default config."""
    checker = HealthChecker()
    assert isinstance(checker, HealthChecker)
    assert checker.check_interval_seconds == 60
    assert checker.alert_manager is not None
    assert isinstance(checker.alert_manager, AlertManager)


def test_health_checker_with_custom_config():
    """HealthChecker should accept custom alert_manager and config."""
    custom_alert = AlertManager(cooldown_seconds=30)
    custom_config = {
        "data_freshness_max_age_hours": 12,
        "model_staleness_max_days": 3,
    }
    checker = HealthChecker(
        alert_manager=custom_alert,
        check_interval_seconds=120,
        config=custom_config,
    )
    assert checker.check_interval_seconds == 120
    assert checker.alert_manager is custom_alert
    assert checker.config["data_freshness_max_age_hours"] == 12


def test_health_checker_default_config():
    """Default config should have expected keys."""
    checker = HealthChecker()
    config = checker.config
    assert isinstance(config, dict)
    assert "data_freshness_max_age_hours" in config
    assert "model_staleness_max_days" in config
    assert "min_disk_space_gb" in config
    assert "min_model_auc" in config


def test_health_checker_results_dict():
    """HealthChecker should have a _results dict."""
    checker = HealthChecker()
    assert isinstance(checker._results, dict)
