"""
GIGA TRADER - Circuit Breaker for Signal-Level Risk Enforcement
================================================================
Enforces hard trading limits at signal generation time:
  - Daily loss limit (default 2%)
  - Max drawdown from peak equity (default 10%)
  - Consecutive loss throttling (default 5 losses -> 50% position reduction)

Persists state to JSON for crash recovery using atomic writes.

Usage:
    from src.phase_19_paper_trading.circuit_breaker import CircuitBreaker

    cb = CircuitBreaker(state_file=Path("data/circuit_breaker_state.json"))
    signal, blocked = cb.check({"direction": "BUY", "confidence": 0.72, "position_size": 0.10})
    if blocked:
        logger.warning(f"Signal blocked: {blocked}")
"""

import json
import logging
import os
from copy import deepcopy
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("GigaTrader.CircuitBreaker")


class CircuitBreaker:
    """
    Enforces hard trading limits: daily loss, max drawdown, consecutive losses.
    Persists state to JSON for crash recovery.
    """

    def __init__(
        self,
        max_daily_loss_pct: float = 0.02,
        max_drawdown_pct: float = 0.10,
        max_consecutive_losses: int = 5,
        state_file: Optional[Path] = None,
    ):
        """
        Args:
            max_daily_loss_pct: Maximum allowed daily loss as fraction (0.02 = 2%).
            max_drawdown_pct: Maximum drawdown from peak equity (0.10 = 10%).
            max_consecutive_losses: Max consecutive losing trades before reducing
                                    position size by 50%.
            state_file: Path to persist state. If None, state is in-memory only.
        """
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_drawdown_pct = max_drawdown_pct
        self.max_consecutive_losses = max_consecutive_losses
        self.state_file = Path(state_file) if state_file else None

        # State
        self.daily_pnl: float = 0.0
        self.peak_equity: float = 0.0
        self.current_equity: float = 0.0
        self.consecutive_losses: int = 0
        self.current_date: str = ""
        self.trades_today: List[Dict] = []
        self.is_tripped: bool = False
        self.trip_reason: str = ""

        # Load persisted state if available
        if self.state_file and self.state_file.exists():
            self._load_state()

    def check(self, signal: Dict) -> Tuple[Dict, Optional[str]]:
        """
        Check if a trading signal should be allowed, blocked, or reduced.

        Args:
            signal: Dict with keys like:
                {
                    "direction": "BUY"/"SELL"/"HOLD",
                    "confidence": float,
                    "position_size": float,
                    ...
                }

        Returns:
            Tuple of (possibly_modified_signal, blocked_reason_or_None).
            If blocked: signal["direction"] is set to "HOLD", blocked_reason explains why.
            If reduced: signal["position_size"] is halved, blocked_reason is None.
            If allowed: signal is returned unchanged, blocked_reason is None.
        """
        result = deepcopy(signal)

        # HOLD signals always pass through unchanged
        if result.get("direction", "").upper() == "HOLD":
            return result, None

        # Auto-detect date change and reset daily counters
        today_str = date.today().isoformat()
        if self.current_date and self.current_date != today_str:
            logger.info(
                f"Date changed from {self.current_date} to {today_str}, "
                f"resetting daily counters"
            )
            self.reset_daily(today_str)

        # Check 1: Circuit breaker already tripped
        if self.is_tripped:
            result["direction"] = "HOLD"
            reason = f"Circuit breaker tripped: {self.trip_reason}"
            logger.warning(f"Signal blocked — {reason}")
            return result, reason

        # Check 2: Daily loss limit
        if self.peak_equity > 0:
            daily_loss_pct = -self.daily_pnl / self.peak_equity
            if daily_loss_pct >= self.max_daily_loss_pct:
                self.is_tripped = True
                self.trip_reason = (
                    f"Daily loss limit exceeded: {daily_loss_pct:.2%} "
                    f">= {self.max_daily_loss_pct:.2%}"
                )
                result["direction"] = "HOLD"
                logger.warning(f"Signal blocked — {self.trip_reason}")
                self._save_state()
                return result, self.trip_reason

        # Check 3: Drawdown from peak
        drawdown = self._current_drawdown()
        if drawdown >= self.max_drawdown_pct:
            self.is_tripped = True
            self.trip_reason = (
                f"Max drawdown exceeded: {drawdown:.2%} "
                f">= {self.max_drawdown_pct:.2%}"
            )
            result["direction"] = "HOLD"
            logger.warning(f"Signal blocked — {self.trip_reason}")
            self._save_state()
            return result, self.trip_reason

        # Check 4: Consecutive losses -> reduce position size by 50%
        if self.consecutive_losses >= self.max_consecutive_losses:
            original_size = result.get("position_size", 0.0)
            result["position_size"] = original_size * 0.5
            logger.info(
                f"Consecutive losses ({self.consecutive_losses}) >= limit "
                f"({self.max_consecutive_losses}): position size reduced "
                f"{original_size:.4f} -> {result['position_size']:.4f}"
            )
            # Not blocked, just reduced — return None for blocked_reason
            return result, None

        # All checks passed
        return result, None

    def record_trade(self, pnl: float, date: str = ""):
        """
        Record a completed trade result.

        Args:
            pnl: Profit/loss as fraction of equity (e.g., 0.005 = +0.5%).
            date: Trade date string (YYYY-MM-DD). If empty, uses today.
        """
        trade_date = date or datetime.now().strftime("%Y-%m-%d")

        # Auto-detect date change
        if self.current_date and self.current_date != trade_date:
            logger.info(
                f"Date changed in record_trade: {self.current_date} -> {trade_date}"
            )
            self.reset_daily(trade_date)

        if not self.current_date:
            self.current_date = trade_date

        # Update daily P&L
        self.daily_pnl += pnl

        # Update equity
        self.current_equity += pnl * self.current_equity if self.current_equity > 0 else pnl

        # Update peak equity
        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity

        # Update consecutive losses
        if pnl < 0:
            self.consecutive_losses += 1
            logger.info(
                f"Losing trade recorded: pnl={pnl:+.4f}, "
                f"consecutive_losses={self.consecutive_losses}"
            )
        else:
            if self.consecutive_losses > 0:
                logger.info(
                    f"Winning trade resets consecutive losses "
                    f"(was {self.consecutive_losses})"
                )
            self.consecutive_losses = 0

        # Record trade details
        self.trades_today.append({
            "pnl": pnl,
            "date": trade_date,
            "timestamp": datetime.now().isoformat(),
        })

        self._save_state()

    def reset_daily(self, date: str = ""):
        """
        Reset daily counters. Called at start of each trading day.

        Args:
            date: New trading date (YYYY-MM-DD). If empty, uses today.
        """
        new_date = date or datetime.now().strftime("%Y-%m-%d")
        logger.info(f"Daily reset for {new_date}")

        self.daily_pnl = 0.0
        self.trades_today = []
        self.current_date = new_date

        # Only clear trip if it was a daily-loss trip (drawdown persists across days)
        if self.is_tripped and "Daily loss" in self.trip_reason:
            self.is_tripped = False
            self.trip_reason = ""
            logger.info("Daily loss trip cleared on new day")

        self._save_state()

    def set_equity(self, equity: float):
        """
        Initialize or update equity tracking.

        Args:
            equity: Current account equity value.
        """
        self.current_equity = equity
        if equity > self.peak_equity:
            self.peak_equity = equity
            logger.info(f"Peak equity updated: {self.peak_equity:.2f}")
        self._save_state()

    def get_status(self) -> Dict:
        """Return current circuit breaker status."""
        return {
            "is_tripped": self.is_tripped,
            "trip_reason": self.trip_reason,
            "daily_pnl": self.daily_pnl,
            "consecutive_losses": self.consecutive_losses,
            "drawdown_pct": self._current_drawdown(),
            "trades_today": len(self.trades_today),
            "peak_equity": self.peak_equity,
            "current_equity": self.current_equity,
            "current_date": self.current_date,
        }

    def _current_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self.peak_equity <= 0:
            return 0.0
        return (self.peak_equity - self.current_equity) / self.peak_equity

    def _save_state(self):
        """Persist state to JSON file using atomic write (write to .tmp then rename)."""
        if not self.state_file:
            return
        try:
            data = {
                "daily_pnl": self.daily_pnl,
                "peak_equity": self.peak_equity,
                "current_equity": self.current_equity,
                "consecutive_losses": self.consecutive_losses,
                "current_date": self.current_date,
                "trades_today": self.trades_today,
                "is_tripped": self.is_tripped,
                "trip_reason": self.trip_reason,
                "updated_at": datetime.now().isoformat(),
            }
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.state_file.with_suffix(".tmp")
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, default=str)
            os.replace(tmp_path, self.state_file)
        except Exception as e:
            logger.warning(f"Could not persist circuit breaker state: {e}")

    def _load_state(self):
        """Load state from JSON file."""
        if not self.state_file:
            return
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.daily_pnl = float(data.get("daily_pnl", 0.0))
            self.peak_equity = float(data.get("peak_equity", 0.0))
            self.current_equity = float(data.get("current_equity", 0.0))
            self.consecutive_losses = int(data.get("consecutive_losses", 0))
            self.current_date = str(data.get("current_date", ""))
            self.trades_today = list(data.get("trades_today", []))
            self.is_tripped = bool(data.get("is_tripped", False))
            self.trip_reason = str(data.get("trip_reason", ""))

            logger.info(
                f"Circuit breaker state loaded: date={self.current_date}, "
                f"daily_pnl={self.daily_pnl:.4f}, "
                f"consecutive_losses={self.consecutive_losses}, "
                f"tripped={self.is_tripped}"
            )
        except (json.JSONDecodeError, ValueError, TypeError) as e:
            logger.warning(f"Invalid circuit breaker state format: {e}")
        except IOError as e:
            logger.warning(f"Could not read circuit breaker state file: {e}")
