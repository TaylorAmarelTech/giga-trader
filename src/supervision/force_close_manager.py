"""
Force Close Manager - Ensures all positions are closed by 3:55 PM ET.

CRITICAL: Never hold positions overnight. This module enforces time-based
position closure with escalating urgency levels.

Escalation Timeline:
- 15:45 - WARNING: Start limit order close attempts
- 15:50 - ELEVATED: Tighten limit prices, increase frequency
- 15:53 - CRITICAL: Switch to market orders
- 15:55 - DEADLINE: Must be flat
- 15:56 - EMERGENCY: Force close via Alpaca API
"""

from dataclasses import dataclass, field
from datetime import datetime, time as dt_time, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import logging
from zoneinfo import ZoneInfo

logger = logging.getLogger("GigaTrader.ForceClose")


class CloseUrgency(Enum):
    """Urgency levels for position closure."""
    NORMAL = "normal"           # > 10 min to deadline
    ELEVATED = "elevated"       # 5-10 min to deadline
    HIGH = "high"               # 2-5 min to deadline
    CRITICAL = "critical"       # < 2 min to deadline
    EMERGENCY = "emergency"     # Past deadline


@dataclass
class ForceCloseConfig:
    """Configuration for force close timing."""
    force_close_time: dt_time = field(default_factory=lambda: dt_time(15, 55))
    warning_time: dt_time = field(default_factory=lambda: dt_time(15, 45))
    elevated_time: dt_time = field(default_factory=lambda: dt_time(15, 50))
    critical_time: dt_time = field(default_factory=lambda: dt_time(15, 53))
    emergency_time: dt_time = field(default_factory=lambda: dt_time(15, 56))
    timezone: str = "America/New_York"
    use_limit_orders_until: dt_time = field(default_factory=lambda: dt_time(15, 53))
    limit_offset_pct: float = 0.002  # 0.2% worse than market for quick fill
    block_new_trades_time: dt_time = field(default_factory=lambda: dt_time(15, 50))


@dataclass
class CloseAttempt:
    """Record of a close attempt."""
    timestamp: datetime
    symbol: str
    urgency: CloseUrgency
    order_type: str  # "limit" or "market"
    order_id: Optional[str]
    success: bool
    error: Optional[str] = None


class ForceCloseManager:
    """
    Manages time-based position closure with escalating urgency.

    CRITICAL RULE: NEVER hold positions after 3:55 PM ET.

    This class should be called at the START of every trading loop iteration,
    BEFORE any other trading logic.
    """

    def __init__(
        self,
        alpaca_client: Any,  # AlpacaPaperClient
        config: Optional[ForceCloseConfig] = None,
        alert_callback: Optional[Callable] = None,
    ):
        """
        Initialize ForceCloseManager.

        Args:
            alpaca_client: Alpaca trading client
            config: Force close timing configuration
            alert_callback: Function to call for alerts (severity, message)
        """
        self.client = alpaca_client
        self.config = config or ForceCloseConfig()
        self.alert_callback = alert_callback
        self.tz = ZoneInfo(self.config.timezone)
        self.close_attempts: List[CloseAttempt] = []
        self._last_urgency_log: Optional[CloseUrgency] = None
        self._emergency_close_attempted_today: bool = False
        self._last_emergency_date: Optional[datetime] = None

    def get_et_now(self) -> datetime:
        """Get current time in Eastern Time."""
        return datetime.now(self.tz)

    def get_current_urgency(self) -> CloseUrgency:
        """
        Determine current urgency level based on time.

        Returns:
            Current CloseUrgency level
        """
        now = self.get_et_now()
        current_time = now.time()

        if current_time >= self.config.emergency_time:
            return CloseUrgency.EMERGENCY
        elif current_time >= self.config.force_close_time:
            return CloseUrgency.EMERGENCY  # Past deadline = emergency
        elif current_time >= self.config.critical_time:
            return CloseUrgency.CRITICAL
        elif current_time >= self.config.elevated_time:
            return CloseUrgency.HIGH
        elif current_time >= self.config.warning_time:
            return CloseUrgency.ELEVATED
        else:
            return CloseUrgency.NORMAL

    def minutes_to_deadline(self) -> float:
        """
        Get minutes remaining until force close deadline.

        Returns:
            Minutes until deadline (negative if past)
        """
        now = self.get_et_now()
        deadline = now.replace(
            hour=self.config.force_close_time.hour,
            minute=self.config.force_close_time.minute,
            second=0,
            microsecond=0
        )
        delta = deadline - now
        return delta.total_seconds() / 60

    def is_past_deadline(self) -> bool:
        """Check if we're past the force close deadline."""
        return self.minutes_to_deadline() < 0

    def is_market_closed(self) -> bool:
        """Check if regular market hours have ended (past 4 PM ET)."""
        now = self.get_et_now()
        current_time = now.time()
        market_close = dt_time(16, 0)  # 4:00 PM ET
        return current_time > market_close or current_time < dt_time(9, 30)

    def reset_daily_flags(self):
        """Reset daily flags (call at start of each trading day)."""
        today = self.get_et_now().date()
        if self._last_emergency_date != today:
            self._emergency_close_attempted_today = False
            self._last_emergency_date = today
            self._last_urgency_log = None
            logger.info("ForceCloseManager daily flags reset")

    def should_block_new_trades(self) -> Tuple[bool, str]:
        """
        Check if new trades should be blocked due to approaching close.

        Returns:
            (should_block, reason)
        """
        now = self.get_et_now()
        current_time = now.time()

        if current_time >= self.config.block_new_trades_time:
            mins = self.minutes_to_deadline()
            return True, f"Market close approaching ({mins:.1f} min to deadline)"
        return False, ""

    def check_and_close(
        self,
        positions: List[Any],  # List of Position objects
    ) -> Dict[str, bool]:
        """
        Check time and close positions if needed.

        This should be called at the START of every trading loop iteration.

        Args:
            positions: List of current positions

        Returns:
            Dict of {symbol: success} for any close attempts
        """
        # Reset daily flags at start of new trading day
        self.reset_daily_flags()

        urgency = self.get_current_urgency()
        results: Dict[str, bool] = {}

        # Log urgency change (but don't spam)
        if urgency != self._last_urgency_log and urgency != CloseUrgency.NORMAL:
            mins = self.minutes_to_deadline()
            logger.warning(f"Force close urgency: {urgency.value} ({mins:.1f} min to deadline)")
            self._last_urgency_log = urgency

            if self.alert_callback:
                self.alert_callback(
                    "warning" if urgency in [CloseUrgency.ELEVATED, CloseUrgency.HIGH] else "critical",
                    f"Force close urgency: {urgency.value}"
                )

        # No action needed if NORMAL
        if urgency == CloseUrgency.NORMAL:
            return results

        # No positions to close
        if not positions:
            return results

        # If market is closed AND we already attempted emergency close today,
        # don't keep spamming - orders will execute on next market open
        if self.is_market_closed() and self._emergency_close_attempted_today:
            # Log once per session that we're waiting for market open
            if self._last_urgency_log != "waiting_market_open":
                logger.info("Market closed. Emergency close already attempted - orders will execute on market open.")
                self._last_urgency_log = "waiting_market_open"
            return results

        # Close all positions based on urgency
        for position in positions:
            symbol = getattr(position, 'symbol', str(position))

            if urgency == CloseUrgency.EMERGENCY:
                # Emergency: use Alpaca's close_all_positions (but only once per day)
                if not self._emergency_close_attempted_today:
                    success = self._emergency_close_all()
                    results[symbol] = success
                    self._emergency_close_attempted_today = True
                    if not success:
                        logger.critical(f"EMERGENCY CLOSE FAILED for {symbol}")
                break  # close_all handles everything

            elif urgency == CloseUrgency.CRITICAL:
                # Critical: market orders
                success = self._close_with_market(position)
                results[symbol] = success

            else:
                # Elevated/High: limit orders
                success = self._close_with_limit(position, urgency)
                results[symbol] = success

        return results

    def _close_with_limit(
        self,
        position: Any,
        urgency: CloseUrgency,
    ) -> bool:
        """
        Attempt to close position with limit order.

        Limit price gets more aggressive with higher urgency.

        Args:
            position: Position to close
            urgency: Current urgency level

        Returns:
            True if order submitted successfully
        """
        symbol = getattr(position, 'symbol', 'SPY')
        qty = abs(getattr(position, 'qty', getattr(position, 'quantity', 0)))
        side = getattr(position, 'side', 'long')

        if qty == 0:
            return True  # Nothing to close

        try:
            # Get current price
            current_price = self.client.get_latest_price(symbol)
            if not current_price:
                logger.error(f"Cannot get price for {symbol}, falling back to market order")
                return self._close_with_market(position)

            # Calculate limit price (worse for us = better fill chance)
            offset_mult = {
                CloseUrgency.ELEVATED: 1.0,
                CloseUrgency.HIGH: 1.5,
            }.get(urgency, 1.0)

            offset = current_price * self.config.limit_offset_pct * offset_mult

            if side == 'long':
                # Selling: limit below market
                limit_price = round(current_price - offset, 2)
                order_side = 'sell'
            else:
                # Buying to cover: limit above market
                limit_price = round(current_price + offset, 2)
                order_side = 'buy'

            # Submit limit order
            order_id = self.client.submit_limit_order(
                symbol=symbol,
                qty=qty,
                side=order_side,
                limit_price=limit_price,
            )

            logger.info(f"Force close limit order: {symbol} {qty} @ {limit_price} (urgency={urgency.value})")

            self.close_attempts.append(CloseAttempt(
                timestamp=self.get_et_now(),
                symbol=symbol,
                urgency=urgency,
                order_type="limit",
                order_id=order_id,
                success=True,
            ))

            return True

        except Exception as e:
            logger.error(f"Limit close failed for {symbol}: {e}")
            self.close_attempts.append(CloseAttempt(
                timestamp=self.get_et_now(),
                symbol=symbol,
                urgency=urgency,
                order_type="limit",
                order_id=None,
                success=False,
                error=str(e),
            ))
            return False

    def _close_with_market(self, position: Any) -> bool:
        """
        Force close with market order.

        Used for CRITICAL+ urgency when we need immediate fill.

        Args:
            position: Position to close

        Returns:
            True if order submitted successfully
        """
        symbol = getattr(position, 'symbol', 'SPY')
        qty = abs(getattr(position, 'qty', getattr(position, 'quantity', 0)))
        side = getattr(position, 'side', 'long')

        if qty == 0:
            return True

        try:
            order_side = 'sell' if side == 'long' else 'buy'

            order_id = self.client.submit_market_order(
                symbol=symbol,
                qty=qty,
                side=order_side,
            )

            logger.warning(f"Force close MARKET order: {symbol} {qty} (CRITICAL urgency)")

            self.close_attempts.append(CloseAttempt(
                timestamp=self.get_et_now(),
                symbol=symbol,
                urgency=CloseUrgency.CRITICAL,
                order_type="market",
                order_id=order_id,
                success=True,
            ))

            return True

        except Exception as e:
            logger.error(f"Market close failed for {symbol}: {e}")
            self.close_attempts.append(CloseAttempt(
                timestamp=self.get_et_now(),
                symbol=symbol,
                urgency=CloseUrgency.CRITICAL,
                order_type="market",
                order_id=None,
                success=False,
                error=str(e),
            ))
            return False

    def _emergency_close_all(self) -> bool:
        """
        EMERGENCY: Force close all positions via Alpaca's close_all_positions.

        This is the nuclear option when past deadline.

        Returns:
            True if successful
        """
        logger.critical("EMERGENCY: Force closing ALL positions via Alpaca API")

        if self.alert_callback:
            self.alert_callback("critical", "EMERGENCY: Force closing all positions - past deadline")

        try:
            # Use Alpaca's close_all_positions API
            self.client.close_all_positions()

            logger.critical("EMERGENCY close_all_positions executed")

            self.close_attempts.append(CloseAttempt(
                timestamp=self.get_et_now(),
                symbol="ALL",
                urgency=CloseUrgency.EMERGENCY,
                order_type="close_all",
                order_id=None,
                success=True,
            ))

            return True

        except Exception as e:
            logger.critical(f"EMERGENCY close_all_positions FAILED: {e}")

            self.close_attempts.append(CloseAttempt(
                timestamp=self.get_et_now(),
                symbol="ALL",
                urgency=CloseUrgency.EMERGENCY,
                order_type="close_all",
                order_id=None,
                success=False,
                error=str(e),
            ))

            if self.alert_callback:
                self.alert_callback("critical", f"EMERGENCY CLOSE FAILED: {e}")

            return False

    def verify_flat(self) -> Tuple[bool, List[str]]:
        """
        Verify that we have no open positions.

        Returns:
            (is_flat, list_of_open_symbols)
        """
        try:
            positions = self.client.get_all_positions()
            if not positions:
                return True, []

            open_symbols = [
                getattr(p, 'symbol', str(p))
                for p in positions
            ]
            return False, open_symbols

        except Exception as e:
            logger.error(f"Failed to verify flat: {e}")
            return False, ["ERROR_CHECKING"]

    def get_close_summary(self) -> Dict:
        """
        Get summary of close attempts for this session.

        Returns:
            Summary dict with counts and details
        """
        if not self.close_attempts:
            return {"total_attempts": 0}

        return {
            "total_attempts": len(self.close_attempts),
            "successful": sum(1 for a in self.close_attempts if a.success),
            "failed": sum(1 for a in self.close_attempts if not a.success),
            "by_urgency": {
                u.value: sum(1 for a in self.close_attempts if a.urgency == u)
                for u in CloseUrgency
            },
            "by_type": {
                "limit": sum(1 for a in self.close_attempts if a.order_type == "limit"),
                "market": sum(1 for a in self.close_attempts if a.order_type == "market"),
                "close_all": sum(1 for a in self.close_attempts if a.order_type == "close_all"),
            },
            "last_attempt": self.close_attempts[-1] if self.close_attempts else None,
        }
