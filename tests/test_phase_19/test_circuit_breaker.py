"""
Tests for CircuitBreaker signal-level risk enforcement.

Validates:
  - Signal pass-through when no limits hit
  - HOLD signals pass unchanged
  - Daily loss blocking
  - Drawdown blocking
  - Consecutive loss position reduction
  - Win resets consecutive loss counter
  - Daily reset behavior
  - JSON state persistence and loading
  - get_status() correctness
  - record_trade() equity updates
  - Auto date change triggers daily reset
  - Edge case: zero position_size signal
"""

import json
from datetime import date

import pytest

from src.phase_19_paper_trading.circuit_breaker import CircuitBreaker


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def cb():
    """In-memory circuit breaker with default limits."""
    breaker = CircuitBreaker(
        max_daily_loss_pct=0.02,
        max_drawdown_pct=0.10,
        max_consecutive_losses=5,
    )
    breaker.set_equity(100_000.0)
    breaker.current_date = date.today().isoformat()
    return breaker


@pytest.fixture
def buy_signal():
    """Standard BUY signal."""
    return {
        "direction": "BUY",
        "confidence": 0.72,
        "position_size": 0.10,
    }


@pytest.fixture
def sell_signal():
    """Standard SELL signal."""
    return {
        "direction": "SELL",
        "confidence": 0.65,
        "position_size": 0.08,
    }


# ─── Test 1: Signal passes through when no limits hit ───────────────────────


class TestPassThrough:

    def test_buy_signal_passes_when_no_limits_hit(self, cb, buy_signal):
        result, blocked = cb.check(buy_signal)
        assert blocked is None
        assert result["direction"] == "BUY"
        assert result["confidence"] == 0.72
        assert result["position_size"] == 0.10

    def test_sell_signal_passes_when_no_limits_hit(self, cb, sell_signal):
        result, blocked = cb.check(sell_signal)
        assert blocked is None
        assert result["direction"] == "SELL"
        assert result["confidence"] == 0.65
        assert result["position_size"] == 0.08

    def test_original_signal_not_mutated(self, cb, buy_signal):
        original_direction = buy_signal["direction"]
        original_size = buy_signal["position_size"]
        cb.check(buy_signal)
        assert buy_signal["direction"] == original_direction
        assert buy_signal["position_size"] == original_size


# ─── Test 2: HOLD signals always pass through unchanged ─────────────────────


class TestHoldSignal:

    def test_hold_passes_when_tripped(self, cb):
        cb.is_tripped = True
        cb.trip_reason = "Daily loss limit exceeded"
        signal = {"direction": "HOLD", "confidence": 0.0, "position_size": 0.0}
        result, blocked = cb.check(signal)
        assert blocked is None
        assert result["direction"] == "HOLD"

    def test_hold_passes_when_healthy(self, cb):
        signal = {"direction": "HOLD", "confidence": 0.0, "position_size": 0.0}
        result, blocked = cb.check(signal)
        assert blocked is None
        assert result["direction"] == "HOLD"


# ─── Test 3: Daily loss limit blocks further signals ─────────────────────────


class TestDailyLossLimit:

    def test_blocks_when_daily_loss_exceeded(self, cb, buy_signal):
        # Simulate daily loss of 2.5% (exceeds 2% limit)
        cb.daily_pnl = -2500.0  # -2.5% of 100k
        result, blocked = cb.check(buy_signal)
        assert blocked is not None
        assert "Daily loss" in blocked
        assert result["direction"] == "HOLD"
        assert cb.is_tripped is True

    def test_blocks_at_exact_limit(self, cb, buy_signal):
        # Exactly at the 2% limit
        cb.daily_pnl = -2000.0  # -2.0% of 100k
        result, blocked = cb.check(buy_signal)
        assert blocked is not None
        assert result["direction"] == "HOLD"

    def test_allows_just_under_limit(self, cb, buy_signal):
        # Just under 2% limit
        cb.daily_pnl = -1999.0  # -1.999% of 100k
        result, blocked = cb.check(buy_signal)
        assert blocked is None
        assert result["direction"] == "BUY"

    def test_subsequent_signals_also_blocked(self, cb, buy_signal, sell_signal):
        cb.daily_pnl = -2500.0
        _, blocked1 = cb.check(buy_signal)
        assert blocked1 is not None
        _, blocked2 = cb.check(sell_signal)
        assert blocked2 is not None
        assert "Circuit breaker tripped" in blocked2


# ─── Test 4: Drawdown limit blocks signals ───────────────────────────────────


class TestDrawdownLimit:

    def test_blocks_when_drawdown_exceeded(self, cb, buy_signal):
        # Peak at 100k, current at 89k = 11% drawdown (exceeds 10%)
        cb.current_equity = 89_000.0
        result, blocked = cb.check(buy_signal)
        assert blocked is not None
        assert "drawdown" in blocked.lower()
        assert result["direction"] == "HOLD"
        assert cb.is_tripped is True

    def test_allows_within_drawdown_limit(self, cb, buy_signal):
        # Peak at 100k, current at 91k = 9% drawdown (within 10%)
        cb.current_equity = 91_000.0
        result, blocked = cb.check(buy_signal)
        assert blocked is None
        assert result["direction"] == "BUY"

    def test_drawdown_persists_across_days(self, cb, buy_signal):
        # Set drawdown beyond limit
        cb.current_equity = 85_000.0
        _, blocked = cb.check(buy_signal)
        assert blocked is not None
        # Reset daily -- drawdown trip should persist
        cb.reset_daily("2026-02-24")
        assert cb.is_tripped is True


# ─── Test 5: Consecutive losses reduces position size by 50% ─────────────────


class TestConsecutiveLosses:

    def test_reduces_position_at_limit(self, cb, buy_signal):
        cb.consecutive_losses = 5  # At limit
        result, blocked = cb.check(buy_signal)
        assert blocked is None  # Not blocked, just reduced
        assert result["position_size"] == pytest.approx(0.05)  # 0.10 * 0.5

    def test_reduces_position_above_limit(self, cb, buy_signal):
        cb.consecutive_losses = 7  # Above limit
        result, blocked = cb.check(buy_signal)
        assert blocked is None
        assert result["position_size"] == pytest.approx(0.05)

    def test_no_reduction_below_limit(self, cb, buy_signal):
        cb.consecutive_losses = 4  # Below limit of 5
        result, blocked = cb.check(buy_signal)
        assert blocked is None
        assert result["position_size"] == pytest.approx(0.10)


# ─── Test 6: Winning trade resets consecutive loss counter ───────────────────


class TestWinResetsLosses:

    def test_win_resets_consecutive_losses(self, cb):
        cb.consecutive_losses = 4
        cb.record_trade(0.005, date="2026-02-23")  # Winning trade
        assert cb.consecutive_losses == 0

    def test_loss_increments_counter(self, cb):
        cb.consecutive_losses = 2
        cb.record_trade(-0.003, date="2026-02-23")  # Losing trade
        assert cb.consecutive_losses == 3

    def test_zero_pnl_resets_counter(self, cb):
        cb.consecutive_losses = 3
        cb.record_trade(0.0, date="2026-02-23")  # Break-even
        assert cb.consecutive_losses == 0


# ─── Test 7: Daily reset clears daily_pnl and trades_today ──────────────────


class TestDailyReset:

    def test_resets_daily_counters(self, cb):
        cb.daily_pnl = -1500.0
        cb.trades_today = [{"pnl": -0.01}, {"pnl": 0.005}]
        cb.reset_daily("2026-02-24")
        assert cb.daily_pnl == 0.0
        assert cb.trades_today == []
        assert cb.current_date == "2026-02-24"

    def test_clears_daily_loss_trip(self, cb, buy_signal):
        # Trip on daily loss
        cb.daily_pnl = -3000.0
        cb.check(buy_signal)
        assert cb.is_tripped is True
        # Reset should clear the trip
        cb.reset_daily("2026-02-24")
        assert cb.is_tripped is False
        assert cb.trip_reason == ""

    def test_preserves_consecutive_losses(self, cb):
        cb.consecutive_losses = 3
        cb.reset_daily("2026-02-24")
        assert cb.consecutive_losses == 3  # Not reset on new day


# ─── Test 8: State persists to and loads from JSON file ──────────────────────


class TestStatePersistence:

    def test_save_and_load(self, tmp_path):
        state_file = tmp_path / "cb_state.json"

        # Create, configure, and save
        cb1 = CircuitBreaker(state_file=state_file)
        cb1.set_equity(50_000.0)
        cb1.daily_pnl = -500.0
        cb1.consecutive_losses = 3
        cb1.current_date = "2026-02-23"
        cb1.is_tripped = True
        cb1.trip_reason = "test trip"
        cb1.trades_today = [{"pnl": -0.01}]
        cb1._save_state()

        assert state_file.exists()

        # Load into new instance
        cb2 = CircuitBreaker(state_file=state_file)
        assert cb2.daily_pnl == -500.0
        assert cb2.consecutive_losses == 3
        assert cb2.current_date == "2026-02-23"
        assert cb2.is_tripped is True
        assert cb2.trip_reason == "test trip"
        assert cb2.peak_equity == 50_000.0
        assert cb2.current_equity == 50_000.0
        assert len(cb2.trades_today) == 1

    def test_no_crash_on_missing_file(self, tmp_path):
        state_file = tmp_path / "nonexistent.json"
        cb = CircuitBreaker(state_file=state_file)
        assert cb.daily_pnl == 0.0  # Defaults preserved

    def test_no_crash_on_corrupt_file(self, tmp_path):
        state_file = tmp_path / "corrupt.json"
        state_file.write_text("not valid json {{{")
        cb = CircuitBreaker(state_file=state_file)
        assert cb.daily_pnl == 0.0  # Defaults preserved

    def test_atomic_write_leaves_no_tmp(self, tmp_path):
        state_file = tmp_path / "cb_state.json"
        cb = CircuitBreaker(state_file=state_file)
        cb.set_equity(10_000.0)
        cb._save_state()
        tmp_file = state_file.with_suffix(".tmp")
        assert not tmp_file.exists()
        assert state_file.exists()


# ─── Test 9: get_status() returns correct values ────────────────────────────


class TestGetStatus:

    def test_status_fields(self, cb):
        cb.daily_pnl = -1000.0
        cb.consecutive_losses = 2
        cb.trades_today = [{"pnl": -0.01}, {"pnl": 0.005}]

        status = cb.get_status()
        assert status["is_tripped"] is False
        assert status["trip_reason"] == ""
        assert status["daily_pnl"] == -1000.0
        assert status["consecutive_losses"] == 2
        assert status["trades_today"] == 2
        assert status["peak_equity"] == 100_000.0
        assert status["current_equity"] == 100_000.0
        assert status["current_date"] == date.today().isoformat()

    def test_status_drawdown_calculation(self, cb):
        cb.current_equity = 95_000.0
        status = cb.get_status()
        assert status["drawdown_pct"] == pytest.approx(0.05)

    def test_status_when_tripped(self, cb):
        cb.is_tripped = True
        cb.trip_reason = "Max drawdown exceeded"
        status = cb.get_status()
        assert status["is_tripped"] is True
        assert "drawdown" in status["trip_reason"].lower()


# ─── Test 10: record_trade() updates equity correctly ────────────────────────


class TestRecordTradeEquity:

    def test_positive_pnl_increases_equity(self, cb):
        cb.record_trade(0.01, date="2026-02-23")  # +1%
        # 100k + (0.01 * 100k) = 101k
        assert cb.current_equity == pytest.approx(101_000.0)
        assert cb.peak_equity == pytest.approx(101_000.0)

    def test_negative_pnl_decreases_equity(self, cb):
        cb.record_trade(-0.005, date="2026-02-23")  # -0.5%
        assert cb.current_equity == pytest.approx(99_500.0)
        assert cb.peak_equity == pytest.approx(100_000.0)  # Peak unchanged

    def test_daily_pnl_accumulates(self, cb):
        cb.record_trade(0.005, date="2026-02-23")
        cb.record_trade(-0.003, date="2026-02-23")
        assert cb.daily_pnl == pytest.approx(0.002)


# ─── Test 11: Auto date change triggers daily reset ─────────────────────────


class TestAutoDateChange:

    def test_check_resets_on_date_change(self, cb, buy_signal):
        cb.daily_pnl = -1500.0
        cb.current_date = "2026-02-22"  # Yesterday
        # On check, today is 2026-02-23 (or actual today), which differs
        # Force today to be different from current_date
        import unittest.mock
        with unittest.mock.patch(
            "src.phase_19_paper_trading.circuit_breaker.date"
        ) as mock_date:
            mock_date.today.return_value = type("D", (), {"isoformat": lambda s: "2026-02-23"})()
            result, blocked = cb.check(buy_signal)
        # After auto-reset, daily_pnl should be 0
        assert cb.daily_pnl == 0.0
        assert cb.current_date == "2026-02-23"

    def test_record_trade_resets_on_date_change(self, cb):
        cb.daily_pnl = -1000.0
        cb.current_date = "2026-02-22"
        cb.record_trade(0.005, date="2026-02-23")
        # After auto-reset, daily_pnl should be just the new trade
        assert cb.daily_pnl == pytest.approx(0.005)
        assert cb.current_date == "2026-02-23"


# ─── Test 12: Edge case - zero position_size signal ─────────────────────────


class TestZeroPositionSize:

    def test_zero_position_passes_through(self, cb):
        signal = {"direction": "BUY", "confidence": 0.5, "position_size": 0.0}
        result, blocked = cb.check(signal)
        assert blocked is None
        assert result["position_size"] == 0.0

    def test_zero_position_reduced_to_zero(self, cb):
        cb.consecutive_losses = 5
        signal = {"direction": "SELL", "confidence": 0.5, "position_size": 0.0}
        result, blocked = cb.check(signal)
        assert blocked is None
        assert result["position_size"] == 0.0  # 0.0 * 0.5 = 0.0
