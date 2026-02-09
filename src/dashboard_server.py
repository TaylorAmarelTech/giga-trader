"""
GIGA TRADER - Live Dashboard Server (SHIM)
====================================
This module has been moved to:
  - src.phase_20_monitoring.dashboard_server

This file re-exports all public names for backward compatibility.
"""

from src.phase_20_monitoring.dashboard_server import (
    app,
    RequestTracker,
    request_tracker,
    server_start_time,
    last_heartbeat,
    DASHBOARD_HTML,
    main,
)

__all__ = [
    "app",
    "RequestTracker",
    "request_tracker",
    "server_start_time",
    "last_heartbeat",
    "DASHBOARD_HTML",
    "main",
]

if __name__ == "__main__":
    main()
