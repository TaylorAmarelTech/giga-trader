"""
Trading Supervision Module

Provides tightly supervised trading with:
- Position reconciliation (Alpaca vs internal state)
- Force close enforcement (3:55 PM ET deadline)
- Circuit breaker enforcement
- Feature validation
- Model health monitoring
"""

from .force_close_manager import ForceCloseManager, CloseUrgency, ForceCloseConfig
from .feature_validator import FeatureValidator, ValidationResult
from .model_health_monitor import ModelHealthMonitor, HealthStatus
from .position_reconciler import PositionReconciler, ReconciliationResult, ReconciliationStatus
from .circuit_breaker_enforcer import CircuitBreakerEnforcer, CircuitBreakerType, CircuitBreakerAction
from .alerts import AlertDispatcher, Alert, AlertSeverity
from .trading_supervision_service import TradingSupervisionService, SupervisionConfig, SupervisionLevel

__all__ = [
    # Force Close
    "ForceCloseManager",
    "CloseUrgency",
    "ForceCloseConfig",
    # Feature Validation
    "FeatureValidator",
    "ValidationResult",
    # Model Health
    "ModelHealthMonitor",
    "HealthStatus",
    # Position Reconciliation
    "PositionReconciler",
    "ReconciliationResult",
    "ReconciliationStatus",
    # Circuit Breakers
    "CircuitBreakerEnforcer",
    "CircuitBreakerType",
    "CircuitBreakerAction",
    # Alerts
    "AlertDispatcher",
    "Alert",
    "AlertSeverity",
    # Orchestration
    "TradingSupervisionService",
    "SupervisionConfig",
    "SupervisionLevel",
]
