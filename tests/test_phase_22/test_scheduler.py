"""Tests for TaskScheduler (Phase 22 Automation)."""

import time
from datetime import datetime, time as dt_time, timedelta

import pytest

from src.phase_22_automation.scheduler import (
    TaskScheduler,
    ScheduledTask,
    TaskPriority,
    TaskStatus,
    create_health_check_task,
    create_signal_generation_task,
    create_retraining_task,
    create_data_download_task,
    create_reconciliation_task,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _noop():
    """No-op callback."""
    return {"status": "ok"}


def _failing():
    """Always-failing callback."""
    raise RuntimeError("task failed")


def _counter_factory():
    """Return a callback that counts invocations."""
    state = {"count": 0}

    def _cb():
        state["count"] += 1
        return {"count": state["count"]}

    return _cb, state


# ---------------------------------------------------------------------------
# ScheduledTask
# ---------------------------------------------------------------------------

class TestScheduledTask:
    def test_default_construction(self):
        task = ScheduledTask(name="test", callback=_noop)
        assert task.name == "test"
        assert task.interval_seconds == 3600
        assert task.priority == TaskPriority.NORMAL
        assert task.enabled is True

    def test_is_due_when_never_run(self):
        task = ScheduledTask(name="t", callback=_noop, interval_seconds=60)
        assert task.is_due()

    def test_is_due_respects_interval(self):
        task = ScheduledTask(name="t", callback=_noop, interval_seconds=60)
        task.last_run = datetime.now()
        assert not task.is_due()

    def test_is_due_after_interval(self):
        task = ScheduledTask(name="t", callback=_noop, interval_seconds=60)
        task.last_run = datetime.now() - timedelta(seconds=120)
        assert task.is_due()

    def test_disabled_task_not_due(self):
        task = ScheduledTask(name="t", callback=_noop, enabled=False)
        assert not task.is_due()

    def test_weekday_restriction(self):
        task = ScheduledTask(name="t", callback=_noop, weekdays_only=True)
        # Saturday
        saturday = datetime(2026, 3, 7, 12, 0, 0)
        assert not task.is_due(saturday)
        # Friday
        friday = datetime(2026, 3, 6, 12, 0, 0)
        assert task.is_due(friday)

    def test_time_window_restriction(self):
        task = ScheduledTask(
            name="t", callback=_noop,
            earliest=dt_time(9, 30), latest=dt_time(16, 0),
        )
        morning = datetime(2026, 3, 6, 7, 0, 0)
        assert not task.is_due(morning)
        midday = datetime(2026, 3, 6, 12, 0, 0)
        assert task.is_due(midday)
        evening = datetime(2026, 3, 6, 20, 0, 0)
        assert not task.is_due(evening)


# ---------------------------------------------------------------------------
# TaskScheduler
# ---------------------------------------------------------------------------

class TestTaskScheduler:
    def test_construction(self):
        s = TaskScheduler()
        assert s.is_running is False
        assert len(s.tasks) == 0

    def test_add_remove_task(self):
        s = TaskScheduler()
        s.add_task(ScheduledTask(name="t1", callback=_noop))
        assert "t1" in s.tasks
        assert s.remove_task("t1")
        assert "t1" not in s.tasks
        assert not s.remove_task("nonexistent")

    def test_enable_disable(self):
        s = TaskScheduler()
        s.add_task(ScheduledTask(name="t1", callback=_noop))
        s.disable_task("t1")
        assert not s.tasks["t1"].enabled
        s.enable_task("t1")
        assert s.tasks["t1"].enabled

    def test_tick_executes_due_tasks(self):
        cb, state = _counter_factory()
        s = TaskScheduler()
        s.add_task(ScheduledTask(name="t1", callback=cb, interval_seconds=0))
        results = s.tick()
        assert len(results) == 1
        assert results[0].status == TaskStatus.COMPLETED
        assert state["count"] == 1

    def test_tick_skips_not_due(self):
        s = TaskScheduler()
        task = ScheduledTask(name="t1", callback=_noop, interval_seconds=9999)
        task.last_run = datetime.now()
        s.add_task(task)
        results = s.tick()
        assert len(results) == 0

    def test_failing_task_retries_and_fails(self):
        s = TaskScheduler()
        s.add_task(ScheduledTask(
            name="fail", callback=_failing,
            interval_seconds=0, max_retries=2, retry_delay_seconds=0,
        ))
        results = s.tick()
        assert len(results) == 1
        assert results[0].status == TaskStatus.FAILED
        assert s.tasks["fail"].fail_count == 1

    def test_priority_ordering(self):
        s = TaskScheduler()
        order = []
        s.add_task(ScheduledTask(
            name="low", callback=lambda: order.append("low"),
            interval_seconds=0, priority=TaskPriority.LOW,
        ))
        s.add_task(ScheduledTask(
            name="critical", callback=lambda: order.append("critical"),
            interval_seconds=0, priority=TaskPriority.CRITICAL,
        ))
        s.add_task(ScheduledTask(
            name="normal", callback=lambda: order.append("normal"),
            interval_seconds=0, priority=TaskPriority.NORMAL,
        ))
        s.tick()
        assert order == ["critical", "normal", "low"]

    def test_dependency_check(self):
        s = TaskScheduler()
        s.add_task(ScheduledTask(
            name="parent", callback=_noop, interval_seconds=9999,
        ))
        s.add_task(ScheduledTask(
            name="child", callback=_noop, interval_seconds=0,
            depends_on=["parent"],
        ))
        # Parent not completed yet → child should not run
        results = s.tick()
        assert all(r.task_name != "child" for r in results)

    def test_dependency_met(self):
        cb, state = _counter_factory()
        s = TaskScheduler()
        parent = ScheduledTask(name="parent", callback=_noop, interval_seconds=9999)
        parent.last_status = TaskStatus.COMPLETED
        s.add_task(parent)
        s.add_task(ScheduledTask(
            name="child", callback=cb, interval_seconds=0,
            depends_on=["parent"],
        ))
        results = s.tick()
        assert any(r.task_name == "child" for r in results)

    def test_get_status(self):
        s = TaskScheduler()
        s.add_task(ScheduledTask(name="t1", callback=_noop))
        status = s.get_status()
        assert "tasks" in status
        assert "t1" in status["tasks"]
        assert status["running"] is False

    def test_get_history(self):
        s = TaskScheduler()
        s.add_task(ScheduledTask(name="t1", callback=_noop, interval_seconds=0))
        s.tick()
        history = s.get_history()
        assert len(history) == 1
        assert history[0].task_name == "t1"

    def test_get_history_filtered(self):
        s = TaskScheduler()
        s.add_task(ScheduledTask(name="a", callback=_noop, interval_seconds=0))
        s.add_task(ScheduledTask(name="b", callback=_noop, interval_seconds=0))
        s.tick()
        history_a = s.get_history(task_name="a")
        assert len(history_a) == 1
        assert history_a[0].task_name == "a"

    def test_repr(self):
        s = TaskScheduler()
        s.add_task(ScheduledTask(name="t1", callback=_noop))
        assert "TaskScheduler" in repr(s)

    def test_start_stop(self):
        s = TaskScheduler(check_interval=1)
        s.start()
        assert s.is_running
        s.stop()
        assert not s.is_running


# ---------------------------------------------------------------------------
# Preset factories
# ---------------------------------------------------------------------------

class TestPresetFactories:
    def test_health_check_task(self):
        t = create_health_check_task(_noop)
        assert t.name == "health_check"
        assert t.priority == TaskPriority.CRITICAL
        assert t.interval_seconds == 60

    def test_signal_generation_task(self):
        t = create_signal_generation_task(_noop)
        assert t.name == "signal_generation"
        assert t.weekdays_only is True

    def test_retraining_task(self):
        t = create_retraining_task(_noop)
        assert t.name == "model_retraining"
        assert t.interval_seconds == 86400

    def test_data_download_task(self):
        t = create_data_download_task(_noop)
        assert t.priority == TaskPriority.HIGH

    def test_reconciliation_depends_on_download(self):
        t = create_reconciliation_task(_noop)
        assert "data_download" in t.depends_on
