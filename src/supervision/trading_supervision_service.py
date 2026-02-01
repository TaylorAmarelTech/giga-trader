"""
Trading Supervision Service - Central orchestration for all supervision components.

Coordinates:
- ForceCloseManager: EOD position closure
- FeatureValidator: Pre-model validation
- ModelHealthMonitor: Output sanity checks
- PositionReconciler: Alpaca vs internal state
- CircuitBreakerEnforcer: Risk rule enforcement
- AlertDispatcher: Notifications
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import logging
import numpy as np

from .force_close_manager import ForceCloseManager, ForceCloseConfig, CloseUrgency
from .feature_validator import FeatureValidator, ValidationResult
from .model_health_monitor import ModelHealthMonitor, HealthStatus
from .position_reconciler import PositionReconciler, ReconciliationResult
from .circuit_breaker_enforcer import CircuitBreakerEnforcer, CircuitBreakerConfig
from .alerts import AlertDispatcher, Alert, AlertSeverity

logger = logging.getLogger("GigaTrader.Supervision")


class SupervisionLevel(Enum):
    """Level of supervision strictness."""
    MINIMAL = "minimal"      # Only critical checks
    STANDARD = "standard"    # Recommended for paper trading
    STRICT = "strict"        # Recommended for live trading
    PARANOID = "paranoid"    # Maximum supervision


@dataclass
class SupervisionConfig:
    """Configuration for supervision service."""
    level: SupervisionLevel = SupervisionLevel.STANDARD
    reconcile_interval_seconds: int = 30
    health_check_interval_seconds: int = 60

    # Feature validation
    feature_validation_enabled: bool = True
    max_nan_pct: float = 0.05
    strict_feature_count: bool = True

    # Model health
    model_health_enabled: bool = True

    # Circuit breakers
    circuit_breakers_enabled: bool = True

    # Force close
    force_close_enabled: bool = True

    # Reconciliation
    reconciliation_enabled: bool = True
    auto_correct_positions: bool = False

    # Alerts
    alert_on_mismatch: bool = True
    halt_on_critical: bool = True


@dataclass
class SupervisionStatus:
    """Current status of the supervision system."""
    timestamp: datetime
    is_healthy: bool
    trading_allowed: bool
    blocking_reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    positions_reconciled: bool = False
    last_reconciliation_time: Optional[datetime] = None
    active_circuit_breakers: List[str] = field(default_factory=list)
    model_health: Dict[str, bool] = field(default_factory=dict)
    time_to_force_close: Optional[float] = None
    force_close_urgency: Optional[str] = None


class TradingSupervisionService:
    """
    Central orchestration for all trading supervision components.

    Usage:
        supervisor = TradingSupervisionService(alpaca_client, config)

        # Before any trading logic
        status = supervisor.pre_trade_check()
        if not status.trading_allowed:
            logger.warning(f"Trading blocked: {status.blocking_reasons}")
            return

        # Validate features before model
        validation = supervisor.validate_features(features, feature_names)
        if not validation.is_valid:
            logger.error(f"Feature validation failed: {validation.errors}")
            return

        # After model prediction
        health = supervisor.record_model_prediction("swing_l2", probability, features)
        if not health.is_healthy:
            logger.warning(f"Model health issue: {health.issues}")

        # After trade execution
        supervisor.post_trade_update(is_win=True, pnl=0.5)
    """

    def __init__(
        self,
        alpaca_client: Any,
        config: Optional[SupervisionConfig] = None,
        feature_names: Optional[List[str]] = None,
        feature_medians: Optional[Dict[str, float]] = None,
    ):
        """
        Initialize TradingSupervisionService.

        Args:
            alpaca_client: Alpaca trading client
            config: Supervision configuration
            feature_names: Expected feature names for validation
            feature_medians: Feature medians for NaN imputation
        """
        self.client = alpaca_client
        self.config = config or SupervisionConfig()

        # Initialize alert dispatcher
        self.alerts = AlertDispatcher()

        # Create alert callback
        def alert_callback(severity: str, message: str):
            self.alerts.send_simple(severity, message)

        # Initialize components
        self.force_close_mgr = ForceCloseManager(
            alpaca_client=alpaca_client,
            config=ForceCloseConfig(),
            alert_callback=alert_callback,
        ) if self.config.force_close_enabled else None

        self.feature_validator = FeatureValidator(
            expected_feature_names=feature_names or [],
            feature_medians=feature_medians or {},
            max_nan_pct=self.config.max_nan_pct,
            strict_feature_count=self.config.strict_feature_count,
            auto_correct=True,
        ) if self.config.feature_validation_enabled and feature_names else None

        self.model_monitor = ModelHealthMonitor() if self.config.model_health_enabled else None

        self.reconciler = PositionReconciler(
            alpaca_client=alpaca_client,
            reconcile_interval_seconds=self.config.reconcile_interval_seconds,
            alert_callback=alert_callback,
        ) if self.config.reconciliation_enabled else None

        self.circuit_breakers = CircuitBreakerEnforcer(
            config=CircuitBreakerConfig(),
            alpaca_client=alpaca_client,
            alert_callback=alert_callback,
        ) if self.config.circuit_breakers_enabled else None

        # Tracking
        self._last_reconciliation: Optional[datetime] = None
        self._last_health_check: Optional[datetime] = None
        self._daily_pnl_pct: float = 0.0
        self._drawdown_pct: float = 0.0
        self._consecutive_losses: int = 0
        self._api_failures: int = 0

    def pre_trade_check(
        self,
        positions: Optional[List[Any]] = None,
    ) -> SupervisionStatus:
        """
        Run all pre-trade checks.

        This should be called at the START of every trading loop iteration,
        BEFORE any other trading logic.

        Args:
            positions: Current positions (optional, will fetch from Alpaca if not provided)

        Returns:
            SupervisionStatus indicating if trading is allowed
        """
        status = SupervisionStatus(
            timestamp=datetime.now(),
            is_healthy=True,
            trading_allowed=True,
        )

        # 1. Force close check (highest priority)
        if self.force_close_mgr:
            urgency = self.force_close_mgr.get_current_urgency()
            status.force_close_urgency = urgency.value
            status.time_to_force_close = self.force_close_mgr.minutes_to_deadline()

            if urgency != CloseUrgency.NORMAL:
                # Get positions if not provided
                if positions is None:
                    try:
                        positions = self.client.get_all_positions()
                    except Exception as e:
                        logger.error(f"Failed to get positions: {e}")
                        positions = []

                if positions:
                    # Execute force close
                    results = self.force_close_mgr.check_and_close(positions)
                    if results:
                        status.warnings.append(f"Force close executed: {results}")

                # Block new trades if approaching close
                should_block, reason = self.force_close_mgr.should_block_new_trades()
                if should_block:
                    status.trading_allowed = False
                    status.blocking_reasons.append(reason)

            if self.force_close_mgr.is_past_deadline():
                status.trading_allowed = False
                status.blocking_reasons.append("Past force close deadline")

        # 2. Circuit breaker check
        if self.circuit_breakers:
            breakers = self.circuit_breakers.check_all_breakers(
                daily_pnl_pct=self._daily_pnl_pct,
                drawdown_pct=self._drawdown_pct,
                consecutive_losses=self._consecutive_losses,
                api_failures=self._api_failures,
            )

            active = [b for b in breakers if b.triggered]
            status.active_circuit_breakers = [b.breaker_type.value for b in active]

            is_allowed, reasons = self.circuit_breakers.is_trading_allowed()
            if not is_allowed:
                status.trading_allowed = False
                status.blocking_reasons.extend(reasons)

        # 3. Position reconciliation (if interval elapsed)
        if self.reconciler and self._should_reconcile():
            try:
                results = self.reconciler.reconcile()
                status.positions_reconciled = True
                status.last_reconciliation_time = datetime.now()
                self._last_reconciliation = datetime.now()

                if self.reconciler.has_critical_mismatch():
                    status.warnings.append("Critical position mismatch detected")
                    if self.config.halt_on_critical:
                        status.trading_allowed = False
                        status.blocking_reasons.append("Critical position mismatch")

            except Exception as e:
                logger.error(f"Reconciliation failed: {e}")
                status.warnings.append(f"Reconciliation error: {e}")

        # 4. Model health summary
        if self.model_monitor:
            report = self.model_monitor.get_health_report()
            status.model_health = {
                name: health.is_healthy
                for name, health in report.items()
            }

            unhealthy = [name for name, health in report.items() if not health.is_healthy]
            if unhealthy:
                status.warnings.append(f"Unhealthy models: {unhealthy}")

        status.is_healthy = status.trading_allowed and len(status.warnings) == 0

        return status

    def validate_features(
        self,
        features: np.ndarray,
        feature_names: List[str],
    ) -> ValidationResult:
        """
        Validate features before model inference.

        Args:
            features: Feature array
            feature_names: Names of features

        Returns:
            ValidationResult with detailed diagnostics
        """
        if not self.feature_validator:
            # Return a "valid" result if validation disabled
            return ValidationResult(
                is_valid=True,
                nan_count=0,
                inf_count=0,
                feature_count=len(feature_names),
                expected_count=len(feature_names),
            )

        result = self.feature_validator.validate(features, feature_names)

        # Alert on issues
        if not result.is_valid or result.nan_count > 0 or result.missing_features:
            self.alerts.send_feature_validation_alert(
                nan_count=result.nan_count,
                feature_mismatch=bool(result.missing_features or result.extra_features),
                missing_features=result.missing_features,
            )

        return result

    def record_model_prediction(
        self,
        model_name: str,
        probability: float,
        features: Optional[np.ndarray] = None,
    ) -> HealthStatus:
        """
        Record a model prediction and check health.

        Args:
            model_name: Name of the model
            probability: Predicted probability
            features: Input features (optional)

        Returns:
            HealthStatus for the model
        """
        if not self.model_monitor:
            return HealthStatus(
                model_name=model_name,
                is_healthy=True,
            )

        status = self.model_monitor.record_prediction(model_name, probability, features)

        # Alert on issues
        if not status.is_healthy:
            self.alerts.send_model_health_alert(
                model_name=model_name,
                issues=status.issues,
                confidence=status.confidence_score,
            )

        return status

    def post_trade_update(
        self,
        is_win: bool,
        pnl_pct: float,
        current_equity: Optional[float] = None,
        peak_equity: Optional[float] = None,
    ):
        """
        Update supervision state after a trade.

        Args:
            is_win: Whether the trade was profitable
            pnl_pct: P&L as percentage
            current_equity: Current account equity (optional)
            peak_equity: Peak account equity (optional)
        """
        # Update consecutive losses
        if is_win:
            self._consecutive_losses = 0
        else:
            self._consecutive_losses += 1

        # Update daily P&L
        self._daily_pnl_pct += pnl_pct

        # Update drawdown
        if current_equity and peak_equity:
            self._drawdown_pct = (peak_equity - current_equity) / peak_equity

        # Record trade result for circuit breakers
        if self.circuit_breakers:
            self.circuit_breakers.record_trade_result(is_win)

        logger.info(f"Trade recorded: win={is_win}, pnl={pnl_pct:.2%}, consecutive_losses={self._consecutive_losses}")

    def record_api_failure(self):
        """Record an API failure."""
        self._api_failures += 1
        if self.circuit_breakers:
            self.circuit_breakers.record_api_failure()

    def record_api_success(self):
        """Record a successful API call."""
        self._api_failures = 0
        if self.circuit_breakers:
            self.circuit_breakers.record_api_success()

    def force_reconcile(self) -> Optional[List[ReconciliationResult]]:
        """Force immediate reconciliation."""
        if not self.reconciler:
            return None
        return self.reconciler.reconcile()

    def get_status(self) -> SupervisionStatus:
        """Get current supervision status without executing checks."""
        return SupervisionStatus(
            timestamp=datetime.now(),
            is_healthy=True,
            trading_allowed=True,
            last_reconciliation_time=self._last_reconciliation,
            active_circuit_breakers=[
                b.breaker_type.value
                for b in (self.circuit_breakers.get_active_breakers() if self.circuit_breakers else [])
            ],
            model_health={
                name: health.is_healthy
                for name, health in (self.model_monitor.get_health_report() if self.model_monitor else {}).items()
            },
            time_to_force_close=self.force_close_mgr.minutes_to_deadline() if self.force_close_mgr else None,
        )

    def emergency_shutdown(self, reason: str):
        """
        Emergency shutdown: close all positions and halt trading.

        Args:
            reason: Reason for shutdown
        """
        logger.critical(f"EMERGENCY SHUTDOWN: {reason}")

        self.alerts.send(Alert(
            timestamp=datetime.now(),
            severity=AlertSeverity.EMERGENCY,
            title="Emergency Shutdown",
            message=reason,
        ))

        # Close all positions
        try:
            self.client.close_all_positions()
            logger.critical("All positions closed in emergency shutdown")
        except Exception as e:
            logger.critical(f"Failed to close positions in emergency: {e}")

    def reset_daily_stats(self):
        """Reset daily statistics (call at start of trading day)."""
        self._daily_pnl_pct = 0.0
        self._consecutive_losses = 0
        self._api_failures = 0
        logger.info("Daily supervision stats reset")

    def _should_reconcile(self) -> bool:
        """Check if it's time for reconciliation."""
        if not self._last_reconciliation:
            return True

        elapsed = (datetime.now() - self._last_reconciliation).total_seconds()
        return elapsed >= self.config.reconcile_interval_seconds

    def update_feature_validator(
        self,
        expected_names: List[str],
        medians: Optional[Dict[str, float]] = None,
    ):
        """
        Update feature validator with new expected features.

        Args:
            expected_names: List of expected feature names
            medians: Feature medians for imputation
        """
        self.feature_validator = FeatureValidator(
            expected_feature_names=expected_names,
            feature_medians=medians or {},
            max_nan_pct=self.config.max_nan_pct,
            strict_feature_count=self.config.strict_feature_count,
            auto_correct=True,
        )
        logger.info(f"Feature validator updated with {len(expected_names)} features")
