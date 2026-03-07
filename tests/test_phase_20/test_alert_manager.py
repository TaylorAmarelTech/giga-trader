"""Tests for AlertManager (Phase 20 Monitoring)."""

from datetime import datetime, timedelta
from pathlib import Path

import pytest

from src.phase_20_monitoring.alert_manager import (
    RuleBasedAlertManager,
    Alert,
    AlertLevel,
    AlertRule,
    create_drawdown_rule,
    create_model_drift_rule,
    create_data_stale_rule,
    create_consecutive_loss_rule,
    create_high_volatility_rule,
)


# ---------------------------------------------------------------------------
# Alert dataclass
# ---------------------------------------------------------------------------

class TestAlert:
    def test_construction(self):
        a = Alert(level=AlertLevel.WARNING, source="test", message="hello")
        assert a.level == AlertLevel.WARNING
        assert a.source == "test"
        assert not a.acknowledged

    def test_auto_id(self):
        a = Alert(level=AlertLevel.INFO, source="x", message="y")
        assert a.alert_id.startswith("x_")

    def test_to_dict(self):
        a = Alert(level=AlertLevel.ERROR, source="src", message="msg")
        d = a.to_dict()
        assert d["level"] == "error"
        assert d["source"] == "src"
        assert d["message"] == "msg"
        assert "timestamp" in d


# ---------------------------------------------------------------------------
# AlertRule
# ---------------------------------------------------------------------------

class TestAlertRule:
    def test_should_trigger(self):
        rule = AlertRule(
            name="test",
            condition=lambda m: m.get("val", 0) > 10,
        )
        assert rule.should_trigger({"val": 15})
        assert not rule.should_trigger({"val": 5})

    def test_cooldown_respected(self):
        rule = AlertRule(
            name="test",
            condition=lambda m: True,
            cooldown_seconds=300,
        )
        rule.last_triggered = datetime.now()
        assert not rule.should_trigger({})

    def test_cooldown_expired(self):
        rule = AlertRule(
            name="test",
            condition=lambda m: True,
            cooldown_seconds=10,
        )
        rule.last_triggered = datetime.now() - timedelta(seconds=60)
        assert rule.should_trigger({})


# ---------------------------------------------------------------------------
# AlertManager
# ---------------------------------------------------------------------------

class TestRuleBasedAlertManager:
    def test_construction(self):
        am = RuleBasedAlertManager()
        assert len(am.alerts) == 0
        assert len(am.rules) == 0

    def test_send_alert(self):
        am = RuleBasedAlertManager()
        a = Alert(level=AlertLevel.INFO, source="test", message="hello")
        result = am.send_alert(a)
        assert result is True
        assert len(am.alerts) == 1

    def test_alert_convenience(self):
        am = RuleBasedAlertManager()
        am.alert(AlertLevel.WARNING, "src", "msg", extra="data")
        assert len(am.alerts) == 1
        assert am.alerts[0].metadata["extra"] == "data"

    def test_dedup_suppresses_duplicate(self):
        am = RuleBasedAlertManager(dedup_window_seconds=60)
        am.alert(AlertLevel.INFO, "src", "same message")
        am.alert(AlertLevel.INFO, "src", "same message")
        assert len(am.alerts) == 1
        assert am._suppressed_count == 1

    def test_dedup_allows_different_messages(self):
        am = RuleBasedAlertManager(dedup_window_seconds=60)
        am.alert(AlertLevel.INFO, "src", "message 1")
        am.alert(AlertLevel.INFO, "src", "message 2")
        assert len(am.alerts) == 2

    def test_max_alerts_trimmed(self):
        am = RuleBasedAlertManager(max_alerts=5, dedup_window_seconds=0)
        for i in range(10):
            am.alert(AlertLevel.INFO, "src", f"msg {i}")
        assert len(am.alerts) == 5

    def test_callback_channel(self):
        received = []
        am = RuleBasedAlertManager()
        am.add_callback(lambda a: received.append(a))
        am.alert(AlertLevel.INFO, "test", "hello")
        assert len(received) == 1
        assert received[0].message == "hello"

    def test_file_channel(self, tmp_path):
        alert_file = tmp_path / "alerts.jsonl"
        am = RuleBasedAlertManager(alert_file=alert_file)
        am.alert(AlertLevel.ERROR, "test", "file alert")
        assert alert_file.exists()
        content = alert_file.read_text()
        assert "file alert" in content

    def test_add_remove_rule(self):
        am = RuleBasedAlertManager()
        rule = AlertRule(name="r1", condition=lambda m: True)
        am.add_rule(rule)
        assert "r1" in am.rules
        assert am.remove_rule("r1")
        assert "r1" not in am.rules

    def test_evaluate_rules_fires(self):
        am = RuleBasedAlertManager()
        am.add_rule(AlertRule(
            name="high_dd",
            condition=lambda m: m.get("dd") > 0.05,
            level=AlertLevel.CRITICAL,
            message_template="Drawdown: {dd}",
        ))
        fired = am.evaluate_rules({"dd": 0.08})
        assert len(fired) == 1
        assert fired[0].level == AlertLevel.CRITICAL

    def test_evaluate_rules_not_fired(self):
        am = RuleBasedAlertManager()
        am.add_rule(AlertRule(
            name="high_dd",
            condition=lambda m: m.get("dd", 0) > 0.05,
        ))
        fired = am.evaluate_rules({"dd": 0.02})
        assert len(fired) == 0

    def test_get_alerts_by_level(self):
        am = RuleBasedAlertManager(dedup_window_seconds=0)
        am.alert(AlertLevel.INFO, "a", "info msg")
        am.alert(AlertLevel.ERROR, "b", "error msg")
        errors = am.get_alerts(level=AlertLevel.ERROR)
        assert len(errors) == 1
        assert errors[0].source == "b"

    def test_get_alerts_by_source(self):
        am = RuleBasedAlertManager(dedup_window_seconds=0)
        am.alert(AlertLevel.INFO, "source_a", "msg1")
        am.alert(AlertLevel.INFO, "source_b", "msg2")
        result = am.get_alerts(source="source_a")
        assert len(result) == 1

    def test_acknowledge(self):
        am = RuleBasedAlertManager()
        am.alert(AlertLevel.INFO, "test", "ack me")
        alert_id = am.alerts[0].alert_id
        assert am.acknowledge(alert_id)
        assert am.alerts[0].acknowledged
        assert not am.acknowledge("nonexistent")

    def test_acknowledge_all(self):
        am = RuleBasedAlertManager(dedup_window_seconds=0)
        am.alert(AlertLevel.INFO, "a", "msg1")
        am.alert(AlertLevel.INFO, "b", "msg2")
        count = am.acknowledge_all()
        assert count == 2
        assert all(a.acknowledged for a in am.alerts)

    def test_get_unacknowledged(self):
        am = RuleBasedAlertManager(dedup_window_seconds=0)
        am.alert(AlertLevel.INFO, "a", "msg1")
        am.alert(AlertLevel.INFO, "b", "msg2")
        am.acknowledge(am.alerts[0].alert_id)
        unacked = am.get_unacknowledged()
        assert len(unacked) == 1

    def test_get_summary(self):
        am = RuleBasedAlertManager(dedup_window_seconds=0)
        am.alert(AlertLevel.INFO, "a", "info")
        am.alert(AlertLevel.ERROR, "b", "error")
        summary = am.get_summary()
        assert summary["total_alerts"] == 2
        assert summary["by_level"]["info"] == 1
        assert summary["by_level"]["error"] == 1

    def test_repr(self):
        am = RuleBasedAlertManager()
        assert "AlertManager" in repr(am)


# ---------------------------------------------------------------------------
# Preset rules
# ---------------------------------------------------------------------------

class TestPresetRules:
    def test_drawdown_rule(self):
        rule = create_drawdown_rule(0.05)
        assert rule.should_trigger({"drawdown_pct": 0.08})
        assert not rule.should_trigger({"drawdown_pct": 0.02})

    def test_model_drift_rule(self):
        rule = create_model_drift_rule(0.10)
        assert rule.should_trigger({"prediction_drift": 0.15})
        assert not rule.should_trigger({"prediction_drift": 0.05})

    def test_data_stale_rule(self):
        rule = create_data_stale_rule(30)
        assert rule.should_trigger({"data_age_minutes": 45})
        assert not rule.should_trigger({"data_age_minutes": 10})

    def test_consecutive_loss_rule(self):
        rule = create_consecutive_loss_rule(5)
        assert rule.should_trigger({"consecutive_losses": 6})
        assert not rule.should_trigger({"consecutive_losses": 3})

    def test_high_volatility_rule(self):
        rule = create_high_volatility_rule(0.03)
        assert rule.should_trigger({"current_volatility": 0.04})
        assert not rule.should_trigger({"current_volatility": 0.01})
