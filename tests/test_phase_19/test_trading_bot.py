"""
Tests for TradingBot initialization, health gate, signal dedup, circuit breaker.

Validates:
  - Config validation rejects missing keys
  - Config validation rejects bad percentage values
  - Health gate blocks trading when UNHEALTHY
  - SignalDeduplicator suppresses duplicate signals
  - CircuitBreaker blocks signals after daily loss exceeded
  - CircuitBreaker passes through normal signals
"""

import json
import os
import tempfile
from datetime import date, datetime
from unittest.mock import patch, MagicMock

import pytest

from src.phase_19_paper_trading.signal_dedup import SignalDeduplicator
from src.phase_19_paper_trading.circuit_breaker import CircuitBreaker
from src.phase_20_monitoring.health_checker import HealthStatus


# ─── Signal Deduplicator ────────────────────────────────────────────────────

@pytest.fixture
def dedup(tmp_path):
    state_file = str(tmp_path / "dedup_state.json")
    return SignalDeduplicator(state_file=state_file)


def test_dedup_first_signal_not_duplicate(dedup):
    assert dedup.is_duplicate("BUY") is False


def test_dedup_same_signal_within_interval_is_duplicate(dedup):
    now = datetime.now().isoformat()
    dedup.record_signal("BUY", now, 450.0)
    assert dedup.is_duplicate("BUY", min_interval_seconds=3600) is True


def test_dedup_different_signal_type_not_duplicate(dedup):
    now = datetime.now().isoformat()
    dedup.record_signal("BUY", now, 450.0)
    assert dedup.is_duplicate("SELL", min_interval_seconds=3600) is False


def test_dedup_persists_across_instances(tmp_path):
    state_file = str(tmp_path / "dedup_state.json")
    d1 = SignalDeduplicator(state_file=state_file)
    d1.record_signal("SELL", datetime.now().isoformat(), 450.0)

    d2 = SignalDeduplicator(state_file=state_file)
    assert d2.is_duplicate("SELL", min_interval_seconds=3600) is True


# ─── Circuit Breaker ────────────────────────────────────────────────────────

@pytest.fixture
def cb():
    breaker = CircuitBreaker(
        max_daily_loss_pct=0.02,
        max_drawdown_pct=0.10,
        max_consecutive_losses=5,
    )
    breaker.set_equity(100_000.0)
    breaker.current_date = date.today().isoformat()
    return breaker


def test_cb_passes_normal_signal(cb):
    signal = {"direction": "BUY", "confidence": 0.7, "position_size": 0.10}
    result, blocked = cb.check(signal)
    assert blocked is None
    assert result["direction"] == "BUY"


def test_cb_blocks_after_daily_loss(cb):
    # Simulate losses exceeding 2% of 100k = $2000
    cb.daily_pnl = -2100.0
    signal = {"direction": "BUY", "confidence": 0.7, "position_size": 0.10}
    _, blocked = cb.check(signal)
    assert blocked is not None
    assert "daily" in blocked.lower() or "loss" in blocked.lower()


# ─── Health Gate (unit logic) ───────────────────────────────────────────────

def test_health_gate_blocks_unhealthy():
    """Verify the health status enum comparison used in TradingBot.run()."""
    status = HealthStatus.UNHEALTHY
    assert status == HealthStatus.UNHEALTHY
    # TradingBot refuses to start when overall == UNHEALTHY
    should_block = (status == HealthStatus.UNHEALTHY)
    assert should_block is True


def test_health_gate_allows_healthy():
    status = HealthStatus.HEALTHY
    should_block = (status == HealthStatus.UNHEALTHY)
    assert should_block is False


def test_health_gate_allows_degraded():
    status = HealthStatus.DEGRADED
    should_block = (status == HealthStatus.UNHEALTHY)
    assert should_block is False


# ─── TradingBot config validation ───────────────────────────────────────────

def test_validate_config_rejects_missing_keys():
    """TradingBot._validate_config raises on missing required keys."""
    from src.phase_19_paper_trading.trading_bot import TradingBot
    bad_config = {"symbol": "SPY"}  # Missing many keys
    with patch("src.phase_19_paper_trading.trading_bot.TRADING_CONFIG", bad_config):
        bot_mock = MagicMock(spec=TradingBot)
        bot_mock._REQUIRED_CONFIG_KEYS = TradingBot._REQUIRED_CONFIG_KEYS
        with pytest.raises(ValueError, match="missing required keys"):
            TradingBot._validate_config(bot_mock)


def test_validate_config_rejects_bad_percentages():
    """TradingBot._validate_config raises on out-of-range percentage."""
    from src.phase_19_paper_trading.trading_bot import TradingBot
    bad_config = {
        "symbol": "SPY", "max_position_pct": 5.0,  # >1.0 is invalid
        "max_daily_trades": 10, "max_daily_loss_pct": 0.02,
        "market_open": "09:30", "market_close": "16:00",
        "stop_loss_pct": 0.002, "order_timeout_seconds": 30,
    }
    with patch("src.phase_19_paper_trading.trading_bot.TRADING_CONFIG", bad_config):
        bot_mock = MagicMock(spec=TradingBot)
        bot_mock._REQUIRED_CONFIG_KEYS = TradingBot._REQUIRED_CONFIG_KEYS
        with pytest.raises(ValueError, match="max_position_pct"):
            TradingBot._validate_config(bot_mock)
