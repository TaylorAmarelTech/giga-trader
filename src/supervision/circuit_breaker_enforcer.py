"""
Circuit Breaker Enforcer - Enforces risk rules from config/risk_management.yaml.

Circuit breakers that halt or reduce trading:
- Daily loss > 2% -> Halt new trades
- Drawdown > 10% -> Close all positions
- 5 consecutive losses -> Halt trading
- VIX > 40 -> Reduce positions 50%
- 3 API failures -> Halt trading
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import logging
import yaml
from pathlib import Path

logger = logging.getLogger("GigaTrader.CircuitBreaker")


class CircuitBreakerType(Enum):
    """Types of circuit breakers."""
    DAILY_LOSS = "daily_loss"
    DRAWDOWN = "drawdown"
    CONSECUTIVE_LOSSES = "consecutive_losses"
    VOLATILITY_SPIKE = "volatility_spike"
    API_FAILURE = "api_failure"


class CircuitBreakerAction(Enum):
    """Actions to take when breaker triggers."""
    HALT_NEW_TRADES = "halt_new_trades"
    CLOSE_ALL_POSITIONS = "close_all_positions"
    REDUCE_POSITIONS_50PCT = "reduce_positions_50pct"
    CANCEL_OPEN_ORDERS = "cancel_open_orders"
    HALT_TRADING = "halt_trading"


@dataclass
class CircuitBreakerState:
    """State of a single circuit breaker."""
    breaker_type: CircuitBreakerType
    triggered: bool
    triggered_at: Optional[datetime]
    trigger_value: Optional[float]
    trigger_threshold: float
    action_taken: Optional[CircuitBreakerAction]
    cooldown_until: Optional[datetime]
    requires_manual_reset: bool


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breakers."""
    # Daily loss
    daily_loss_enabled: bool = True
    daily_loss_threshold: float = 0.02  # 2%
    daily_loss_action: CircuitBreakerAction = CircuitBreakerAction.HALT_NEW_TRADES
    daily_loss_cooldown_hours: int = 0  # Resets at midnight

    # Drawdown
    drawdown_enabled: bool = True
    drawdown_threshold: float = 0.10  # 10%
    drawdown_action: CircuitBreakerAction = CircuitBreakerAction.CLOSE_ALL_POSITIONS
    drawdown_cooldown_hours: int = 24
    drawdown_requires_manual_reset: bool = True

    # Consecutive losses
    consecutive_losses_enabled: bool = True
    consecutive_losses_threshold: int = 5
    consecutive_losses_action: CircuitBreakerAction = CircuitBreakerAction.HALT_TRADING
    consecutive_losses_cooldown_hours: int = 4

    # Volatility spike (VIX)
    volatility_enabled: bool = True
    volatility_threshold: float = 40.0  # VIX level
    volatility_action: CircuitBreakerAction = CircuitBreakerAction.REDUCE_POSITIONS_50PCT

    # API failure
    api_failure_enabled: bool = True
    api_failure_threshold: int = 3
    api_failure_action: CircuitBreakerAction = CircuitBreakerAction.HALT_TRADING
    api_failure_cooldown_hours: int = 1


class CircuitBreakerEnforcer:
    """
    Enforces circuit breaker rules from configuration.

    This class monitors trading conditions and triggers protective actions
    when thresholds are exceeded.
    """

    def __init__(
        self,
        config_path: Optional[str] = None,
        config: Optional[CircuitBreakerConfig] = None,
        alpaca_client: Any = None,
        alert_callback: Optional[Callable] = None,
    ):
        """
        Initialize CircuitBreakerEnforcer.

        Args:
            config_path: Path to risk_management.yaml
            config: Direct configuration object (overrides config_path)
            alpaca_client: Alpaca trading client for executing actions
            alert_callback: Function to call for alerts
        """
        self.client = alpaca_client
        self.alert_callback = alert_callback

        # Load configuration
        if config:
            self.config = config
        elif config_path:
            self.config = self._load_config(config_path)
        else:
            self.config = CircuitBreakerConfig()

        # Initialize breaker states
        self._states: Dict[CircuitBreakerType, CircuitBreakerState] = {}
        self._initialize_states()

        # Track metrics
        self._consecutive_losses = 0
        self._api_failures = 0
        self._daily_pnl = 0.0
        self._peak_equity = 0.0
        self._current_equity = 0.0

    def _load_config(self, config_path: str) -> CircuitBreakerConfig:
        """Load configuration from YAML file."""
        try:
            path = Path(config_path)
            if not path.exists():
                logger.warning(f"Config file not found: {config_path}, using defaults")
                return CircuitBreakerConfig()

            with open(path, 'r') as f:
                yaml_config = yaml.safe_load(f)

            cb_config = yaml_config.get("circuit_breakers", {})

            return CircuitBreakerConfig(
                daily_loss_threshold=cb_config.get("daily_loss", {}).get("threshold", 0.02),
                drawdown_threshold=cb_config.get("drawdown", {}).get("threshold", 0.10),
                consecutive_losses_threshold=cb_config.get("consecutive_losses", {}).get("threshold", 5),
                volatility_threshold=cb_config.get("volatility_spike", {}).get("threshold", 40.0),
                api_failure_threshold=cb_config.get("api_failure", {}).get("threshold", 3),
            )

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return CircuitBreakerConfig()

    def _initialize_states(self):
        """Initialize circuit breaker states."""
        for breaker_type in CircuitBreakerType:
            self._states[breaker_type] = CircuitBreakerState(
                breaker_type=breaker_type,
                triggered=False,
                triggered_at=None,
                trigger_value=None,
                trigger_threshold=self._get_threshold(breaker_type),
                action_taken=None,
                cooldown_until=None,
                requires_manual_reset=breaker_type == CircuitBreakerType.DRAWDOWN,
            )

    def _get_threshold(self, breaker_type: CircuitBreakerType) -> float:
        """Get threshold for a breaker type."""
        return {
            CircuitBreakerType.DAILY_LOSS: self.config.daily_loss_threshold,
            CircuitBreakerType.DRAWDOWN: self.config.drawdown_threshold,
            CircuitBreakerType.CONSECUTIVE_LOSSES: self.config.consecutive_losses_threshold,
            CircuitBreakerType.VOLATILITY_SPIKE: self.config.volatility_threshold,
            CircuitBreakerType.API_FAILURE: self.config.api_failure_threshold,
        }.get(breaker_type, 0)

    def check_all_breakers(
        self,
        daily_pnl_pct: float,
        drawdown_pct: float,
        consecutive_losses: int,
        current_vix: Optional[float] = None,
        api_failures: int = 0,
    ) -> List[CircuitBreakerState]:
        """
        Check all circuit breakers and return their states.

        Args:
            daily_pnl_pct: Today's P&L as percentage (negative = loss)
            drawdown_pct: Current drawdown from peak as percentage
            consecutive_losses: Number of consecutive losing trades
            current_vix: Current VIX level (optional)
            api_failures: Number of recent API failures

        Returns:
            List of all breaker states
        """
        # Update tracking
        self._daily_pnl = daily_pnl_pct
        self._consecutive_losses = consecutive_losses
        self._api_failures = api_failures

        # Check each breaker
        self._check_daily_loss(daily_pnl_pct)
        self._check_drawdown(drawdown_pct)
        self._check_consecutive_losses(consecutive_losses)

        if current_vix is not None:
            self._check_volatility(current_vix)

        self._check_api_health(api_failures)

        return list(self._states.values())

    def _check_daily_loss(self, daily_pnl_pct: float):
        """Check daily loss circuit breaker."""
        if not self.config.daily_loss_enabled:
            return

        state = self._states[CircuitBreakerType.DAILY_LOSS]

        # Check if in cooldown (resets at midnight)
        if state.cooldown_until and datetime.now() < state.cooldown_until:
            return

        # Trigger if loss exceeds threshold
        if daily_pnl_pct < -self.config.daily_loss_threshold:
            if not state.triggered:
                self._trigger_breaker(
                    CircuitBreakerType.DAILY_LOSS,
                    daily_pnl_pct,
                    self.config.daily_loss_action,
                )

    def _check_drawdown(self, drawdown_pct: float):
        """Check drawdown circuit breaker."""
        if not self.config.drawdown_enabled:
            return

        state = self._states[CircuitBreakerType.DRAWDOWN]

        # Skip if already triggered and requires manual reset
        if state.triggered and state.requires_manual_reset:
            return

        # Trigger if drawdown exceeds threshold
        if drawdown_pct > self.config.drawdown_threshold:
            if not state.triggered:
                self._trigger_breaker(
                    CircuitBreakerType.DRAWDOWN,
                    drawdown_pct,
                    self.config.drawdown_action,
                )

    def _check_consecutive_losses(self, loss_count: int):
        """Check consecutive loss circuit breaker."""
        if not self.config.consecutive_losses_enabled:
            return

        state = self._states[CircuitBreakerType.CONSECUTIVE_LOSSES]

        # Check cooldown
        if state.cooldown_until and datetime.now() < state.cooldown_until:
            return

        # Trigger if losses exceed threshold
        if loss_count >= self.config.consecutive_losses_threshold:
            if not state.triggered:
                self._trigger_breaker(
                    CircuitBreakerType.CONSECUTIVE_LOSSES,
                    float(loss_count),
                    self.config.consecutive_losses_action,
                )

    def _check_volatility(self, vix: float):
        """Check volatility spike circuit breaker."""
        if not self.config.volatility_enabled:
            return

        state = self._states[CircuitBreakerType.VOLATILITY_SPIKE]

        # Trigger if VIX exceeds threshold
        if vix > self.config.volatility_threshold:
            if not state.triggered:
                self._trigger_breaker(
                    CircuitBreakerType.VOLATILITY_SPIKE,
                    vix,
                    self.config.volatility_action,
                )
        else:
            # Auto-reset when VIX drops
            if state.triggered:
                self._reset_breaker(CircuitBreakerType.VOLATILITY_SPIKE)

    def _check_api_health(self, failure_count: int):
        """Check API failure circuit breaker."""
        if not self.config.api_failure_enabled:
            return

        state = self._states[CircuitBreakerType.API_FAILURE]

        # Check cooldown
        if state.cooldown_until and datetime.now() < state.cooldown_until:
            return

        # Trigger if failures exceed threshold
        if failure_count >= self.config.api_failure_threshold:
            if not state.triggered:
                self._trigger_breaker(
                    CircuitBreakerType.API_FAILURE,
                    float(failure_count),
                    self.config.api_failure_action,
                )

    def _trigger_breaker(
        self,
        breaker_type: CircuitBreakerType,
        value: float,
        action: CircuitBreakerAction,
    ):
        """Trigger a circuit breaker."""
        state = self._states[breaker_type]
        now = datetime.now()

        state.triggered = True
        state.triggered_at = now
        state.trigger_value = value
        state.action_taken = action

        # Set cooldown
        cooldown_hours = {
            CircuitBreakerType.DAILY_LOSS: self.config.daily_loss_cooldown_hours,
            CircuitBreakerType.DRAWDOWN: self.config.drawdown_cooldown_hours,
            CircuitBreakerType.CONSECUTIVE_LOSSES: self.config.consecutive_losses_cooldown_hours,
            CircuitBreakerType.VOLATILITY_SPIKE: 0,  # Auto-resets
            CircuitBreakerType.API_FAILURE: self.config.api_failure_cooldown_hours,
        }.get(breaker_type, 0)

        if cooldown_hours > 0:
            state.cooldown_until = now + timedelta(hours=cooldown_hours)

        # Log and alert
        logger.warning(f"CIRCUIT BREAKER TRIGGERED: {breaker_type.value} at {value:.4f}")

        if self.alert_callback:
            self.alert_callback(
                "critical",
                f"Circuit breaker triggered: {breaker_type.value} (value={value:.4f}, action={action.value})",
            )

        # Execute action
        self.execute_action(action)

    def execute_action(self, action: CircuitBreakerAction) -> bool:
        """
        Execute the circuit breaker action.

        Args:
            action: Action to execute

        Returns:
            True if action executed successfully
        """
        logger.info(f"Executing circuit breaker action: {action.value}")

        try:
            if action == CircuitBreakerAction.HALT_NEW_TRADES:
                # This is just a flag - trading loop checks is_trading_allowed()
                logger.warning("New trades halted by circuit breaker")
                return True

            elif action == CircuitBreakerAction.HALT_TRADING:
                # Complete halt
                logger.critical("TRADING HALTED by circuit breaker")
                return True

            elif action == CircuitBreakerAction.CLOSE_ALL_POSITIONS:
                if self.client:
                    self.client.close_all_positions()
                    logger.critical("All positions closed by circuit breaker")
                return True

            elif action == CircuitBreakerAction.REDUCE_POSITIONS_50PCT:
                # Would need to implement position reduction logic
                logger.warning("Position reduction requested - not yet implemented")
                return False

            elif action == CircuitBreakerAction.CANCEL_OPEN_ORDERS:
                if self.client:
                    # Cancel all open orders
                    try:
                        self.client.trading_client.cancel_orders()
                        logger.warning("All open orders cancelled by circuit breaker")
                    except Exception as e:
                        logger.error(f"Failed to cancel orders: {e}")
                return True

        except Exception as e:
            logger.error(f"Failed to execute action {action.value}: {e}")
            return False

        return False

    def is_trading_allowed(self) -> Tuple[bool, List[str]]:
        """
        Check if trading is allowed based on all breaker states.

        Returns:
            (is_allowed, list_of_blocking_reasons)
        """
        blocking_reasons = []

        for breaker_type, state in self._states.items():
            if not state.triggered:
                continue

            # Check if action blocks trading
            if state.action_taken in [
                CircuitBreakerAction.HALT_NEW_TRADES,
                CircuitBreakerAction.HALT_TRADING,
            ]:
                blocking_reasons.append(
                    f"{breaker_type.value}: triggered at {state.trigger_value:.4f}"
                )

        return len(blocking_reasons) == 0, blocking_reasons

    def reset_breaker(
        self,
        breaker_type: CircuitBreakerType,
        force: bool = False,
    ) -> bool:
        """
        Reset a circuit breaker.

        Args:
            breaker_type: Which breaker to reset
            force: If True, reset even if requires_manual_reset

        Returns:
            True if reset successful
        """
        state = self._states[breaker_type]

        if state.requires_manual_reset and not force:
            logger.warning(f"Cannot reset {breaker_type.value} without force=True (requires manual reset)")
            return False

        return self._reset_breaker(breaker_type)

    def _reset_breaker(self, breaker_type: CircuitBreakerType) -> bool:
        """Internal breaker reset."""
        state = self._states[breaker_type]

        state.triggered = False
        state.triggered_at = None
        state.trigger_value = None
        state.action_taken = None
        state.cooldown_until = None

        logger.info(f"Circuit breaker reset: {breaker_type.value}")
        return True

    def get_active_breakers(self) -> List[CircuitBreakerState]:
        """Get list of currently triggered breakers."""
        return [s for s in self._states.values() if s.triggered]

    def get_status_summary(self) -> Dict:
        """Get summary of all breaker states."""
        return {
            "trading_allowed": self.is_trading_allowed()[0],
            "active_breakers": [s.breaker_type.value for s in self.get_active_breakers()],
            "states": {
                bt.value: {
                    "triggered": s.triggered,
                    "trigger_value": s.trigger_value,
                    "threshold": s.trigger_threshold,
                }
                for bt, s in self._states.items()
            },
        }

    def record_trade_result(self, is_win: bool):
        """
        Record a trade result for consecutive loss tracking.

        Args:
            is_win: True if trade was profitable
        """
        if is_win:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1

    def record_api_failure(self):
        """Record an API failure."""
        self._api_failures += 1

    def record_api_success(self):
        """Record a successful API call (resets failure counter)."""
        self._api_failures = 0
