"""
GIGA TRADER - Enhanced Web Monitoring Dashboard (SHIM)
================================================
This module has been moved to:
  - src.phase_20_monitoring.web_monitor

This file re-exports all public names for backward compatibility.
"""

from src.phase_20_monitoring.web_monitor import (
    app,
    WEB_CONFIG,
    FLASK_AVAILABLE,
    DASHBOARD_HTML,
    LOGS_HTML,
    EXPERIMENTS_HTML,
    MODELS_HTML,
    BACKTESTS_HTML,
    BASE_CSS,
    NAV_HTML,
    main,
)

__all__ = [
    "app",
    "WEB_CONFIG",
    "FLASK_AVAILABLE",
    "DASHBOARD_HTML",
    "LOGS_HTML",
    "EXPERIMENTS_HTML",
    "MODELS_HTML",
    "BACKTESTS_HTML",
    "BASE_CSS",
    "NAV_HTML",
    "main",
]

if __name__ == "__main__":
    import sys
    sys.exit(main())
