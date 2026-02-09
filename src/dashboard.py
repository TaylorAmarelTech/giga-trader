"""
GIGA TRADER - Interactive Dashboard (SHIM)
====================================
This module has been moved to:
  - src.phase_20_monitoring.dashboard

This file re-exports all public names for backward compatibility.
"""

from src.phase_20_monitoring.dashboard import (
    ModelInfo,
    BacktestSummary,
    ModelAnalyzer,
    BacktestAnalyzer,
    get_position_history,
    get_equity_chart_data,
    get_pnl_chart_data,
    get_position_chart_data,
    get_trade_history,
    get_system_status,
    generate_html_dashboard,
    print_console_dashboard,
    main,
)

__all__ = [
    "ModelInfo",
    "BacktestSummary",
    "ModelAnalyzer",
    "BacktestAnalyzer",
    "get_position_history",
    "get_equity_chart_data",
    "get_pnl_chart_data",
    "get_position_chart_data",
    "get_trade_history",
    "get_system_status",
    "generate_html_dashboard",
    "print_console_dashboard",
    "main",
]

if __name__ == "__main__":
    import sys
    sys.exit(main())
