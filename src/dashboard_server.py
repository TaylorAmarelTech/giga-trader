"""
GIGA TRADER - Live Dashboard Server
====================================
Real-time dashboard with auto-refresh, live logs, and heartbeat monitoring.

Usage:
    python src/dashboard_server.py              # Start server on http://localhost:8050
    python src/dashboard_server.py --port 8080  # Custom port
"""

import os
import sys
import json
import logging
import time
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
from collections import deque
from threading import Lock
from flask import Flask, render_template_string, jsonify, Response, request

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import dashboard components
from src.dashboard import (
    ModelAnalyzer,
    BacktestAnalyzer,
    get_system_status,
    get_equity_chart_data,
    get_pnl_chart_data,
    get_position_chart_data,
    get_trade_history,
)
from src.experiment_progress import get_experiment_progress

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DashboardServer")

app = Flask(__name__)

# =============================================================================
# REQUEST TRACKING
# =============================================================================
class RequestTracker:
    """Track API requests for monitoring."""

    def __init__(self, max_size: int = 100):
        self.requests: deque = deque(maxlen=max_size)
        self.lock = Lock()
        self.total_requests = 0
        self.start_time = datetime.now()

    def record(self, path: str, method: str, status: int, duration_ms: float):
        with self.lock:
            self.total_requests += 1
            self.requests.append({
                "timestamp": datetime.now().isoformat(),
                "path": path,
                "method": method,
                "status": status,
                "duration_ms": round(duration_ms, 2),
            })

    def get_recent(self, limit: int = 20) -> List[Dict]:
        with self.lock:
            return list(self.requests)[-limit:][::-1]

    def get_stats(self) -> Dict:
        with self.lock:
            uptime = (datetime.now() - self.start_time).total_seconds()
            return {
                "total_requests": self.total_requests,
                "requests_per_minute": round(self.total_requests / max(uptime / 60, 1), 2),
                "uptime_seconds": round(uptime),
            }


request_tracker = RequestTracker()


@app.before_request
def before_request():
    request.start_time = time.time()


@app.after_request
def after_request(response):
    if hasattr(request, 'start_time'):
        duration_ms = (time.time() - request.start_time) * 1000
        request_tracker.record(
            path=request.path,
            method=request.method,
            status=response.status_code,
            duration_ms=duration_ms,
        )
    return response


# =============================================================================
# HEARTBEAT & HEALTH
# =============================================================================
server_start_time = datetime.now()
last_heartbeat = datetime.now()


@app.route("/api/heartbeat")
def api_heartbeat():
    """Heartbeat endpoint for connection checking."""
    global last_heartbeat
    last_heartbeat = datetime.now()

    return jsonify({
        "status": "alive",
        "timestamp": datetime.now().isoformat(),
        "server_uptime_seconds": (datetime.now() - server_start_time).total_seconds(),
    })


@app.route("/api/health")
def api_health():
    """Comprehensive health check."""
    status = get_system_status()

    # Check various components
    checks = {
        "dashboard_server": True,
        "status_file": status.get("mode") != "UNKNOWN",
        "models_loaded": status.get("model", {}).get("loaded", False),
        "healthy": status.get("health", {}).get("status") == "HEALTHY",
    }

    all_healthy = all(checks.values())

    return jsonify({
        "status": "healthy" if all_healthy else "degraded",
        "checks": checks,
        "timestamp": datetime.now().isoformat(),
        "system_status": status,
    })


# =============================================================================
# LOGS ENDPOINT
# =============================================================================
def read_log_file(log_path: Path, lines: int = 100, filter_pattern: str = None) -> List[Dict]:
    """Read and parse log file."""
    if not log_path.exists():
        return []

    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()

        # Get last N lines
        recent_lines = all_lines[-lines:]

        # Parse log entries
        log_pattern = re.compile(
            r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}) \[(\w+)\] (?:\[([^\]]+)\] )?(.+)"
        )

        entries = []
        for line in recent_lines:
            line = line.strip()
            if not line:
                continue

            match = log_pattern.match(line)
            if match:
                timestamp, level, component, message = match.groups()
                entry = {
                    "timestamp": timestamp,
                    "level": level,
                    "component": component or "MAIN",
                    "message": message[:200],  # Truncate long messages
                }

                # Apply filter if provided
                if filter_pattern:
                    if filter_pattern.lower() not in line.lower():
                        continue

                entries.append(entry)
            else:
                # Non-matching line - append to previous if exists
                if entries:
                    entries[-1]["message"] += " " + line[:100]

        return entries[-lines:]

    except Exception as e:
        logger.error(f"Error reading log file: {e}")
        return []


@app.route("/api/logs")
def api_logs():
    """Get recent log entries."""
    lines = request.args.get("lines", 50, type=int)
    filter_pattern = request.args.get("filter", None)
    log_type = request.args.get("type", "trading")

    # Determine log file
    today = datetime.now().strftime("%Y%m%d")
    if log_type == "trading":
        log_path = project_root / "logs" / f"trading_{today}.log"
    elif log_type == "orchestrator":
        log_path = project_root / "logs" / f"orchestrator_{today}.log"
    else:
        log_path = project_root / "logs" / f"trading_{today}.log"

    # Fallback to any recent log file
    if not log_path.exists():
        log_files = list((project_root / "logs").glob("*.log"))
        if log_files:
            log_path = max(log_files, key=lambda p: p.stat().st_mtime)
        else:
            return jsonify({"entries": [], "log_file": None})

    entries = read_log_file(log_path, lines, filter_pattern)

    return jsonify({
        "entries": entries,
        "log_file": log_path.name,
        "total_entries": len(entries),
    })


@app.route("/api/logs/stream")
def api_logs_stream():
    """Server-Sent Events stream for live logs."""
    def generate():
        last_size = 0
        today = datetime.now().strftime("%Y%m%d")
        log_path = project_root / "logs" / f"trading_{today}.log"

        while True:
            try:
                if log_path.exists():
                    current_size = log_path.stat().st_size
                    if current_size > last_size:
                        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
                            f.seek(last_size)
                            new_content = f.read()
                            last_size = current_size

                            for line in new_content.strip().split("\n"):
                                if line:
                                    yield f"data: {json.dumps({'line': line})}\n\n"

                time.sleep(1)
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"
                time.sleep(5)

    return Response(generate(), mimetype="text/event-stream")


@app.route("/api/requests")
def api_requests():
    """Get recent API requests."""
    return jsonify({
        "requests": request_tracker.get_recent(30),
        "stats": request_tracker.get_stats(),
    })


# =============================================================================
# EXISTING API ENDPOINTS
# =============================================================================
@app.route("/api/status")
def api_status():
    """Get current system status."""
    return jsonify(get_system_status())


@app.route("/api/equity")
def api_equity():
    """Get equity history for charts."""
    data = get_equity_chart_data()
    return jsonify([
        {"x": e.get("timestamp", "")[:19], "y": e.get("equity", 0), "pnl": e.get("daily_pnl", 0)}
        for e in data[-200:]
    ])


@app.route("/api/positions")
def api_positions():
    """Get position history."""
    data = get_position_chart_data()
    return jsonify([
        {"x": p.get("timestamp", "")[:19], "y": p.get("unrealized_pnl", 0), "qty": p.get("quantity", 0)}
        for p in data[-200:]
    ])


@app.route("/api/trades")
def api_trades():
    """Get trade history."""
    return jsonify(get_trade_history()[-50:])


@app.route("/api/models")
def api_models():
    """Get model information."""
    analyzer = ModelAnalyzer()
    models = analyzer.scan_models()
    return jsonify([
        {
            "name": m.name,
            "size_mb": round(m.file_size_mb, 1),
            "features": m.features_count,
            "swing_type": m.swing_model_type,
            "timing_type": m.timing_model_type,
            "modified": m.modified_date.isoformat(),
        }
        for m in models
    ])


@app.route("/api/backtests")
def api_backtests():
    """Get backtest results."""
    analyzer = BacktestAnalyzer()
    backtests = analyzer.scan_backtests()
    return jsonify([
        {
            "id": b.run_id,
            "date": b.run_date.isoformat(),
            "return": b.total_return,
            "sharpe": b.sharpe_ratio,
            "max_dd": b.max_drawdown,
            "win_rate": b.win_rate,
            "trades": b.total_trades,
            "robustness": b.robustness_score,
        }
        for b in backtests[:10]
    ])


@app.route("/api/experiment")
def api_experiment():
    """Get current experiment progress for real-time monitoring."""
    try:
        progress = get_experiment_progress()
        return jsonify(progress)
    except Exception as e:
        return jsonify({
            "is_running": False,
            "step": "idle",
            "error": str(e),
        })


@app.route("/api/experiments/history")
def api_experiments_history():
    """Get experiment history for browsing."""
    try:
        history_file = project_root / "experiments" / "experiment_history.json"
        if not history_file.exists():
            return jsonify([])

        with open(history_file) as f:
            experiments = json.load(f)

        # Extract key fields for display (full config is too large)
        limit = request.args.get("limit", 50, type=int)
        offset = request.args.get("offset", 0, type=int)
        sort_by = request.args.get("sort", "started_at")
        sort_order = request.args.get("order", "desc")

        # Build summary list
        summaries = []
        for exp in experiments:
            summary = {
                "experiment_id": exp.get("experiment_id", ""),
                "experiment_type": exp.get("config", {}).get("experiment_type", ""),
                "experiment_name": exp.get("config", {}).get("experiment_name", ""),
                "status": exp.get("status", ""),
                "started_at": exp.get("started_at", ""),
                "completed_at": exp.get("completed_at", ""),
                "duration_seconds": exp.get("duration_seconds", 0),
                "cv_auc_mean": exp.get("cv_auc_mean", 0),
                "cv_auc_std": exp.get("cv_auc_std", 0),
                "test_auc": exp.get("test_auc", 0),
                "train_auc": exp.get("train_auc", 0),
                "wmes_score": exp.get("wmes_score", 0),
                # NET metrics (after transaction costs) - PRIMARY
                "backtest_sharpe": exp.get("backtest_sharpe", 0),
                "backtest_win_rate": exp.get("backtest_win_rate", 0),
                "backtest_total_return": exp.get("backtest_total_return", 0),
                # GROSS metrics (before costs) - for comparison
                "backtest_sharpe_gross": exp.get("backtest_sharpe_gross", 0),
                "backtest_win_rate_gross": exp.get("backtest_win_rate_gross", 0),
                "backtest_total_return_gross": exp.get("backtest_total_return_gross", 0),
                "n_trades": exp.get("n_trades", 0),
                "n_features_final": exp.get("n_features_final", 0),
                "n_samples_real": exp.get("n_samples_real", 0),
                "n_samples_synthetic": exp.get("n_samples_synthetic", 0),
            }
            summaries.append(summary)

        # Sort
        reverse = sort_order == "desc"
        if sort_by in summaries[0] if summaries else {}:
            summaries.sort(key=lambda x: x.get(sort_by, 0) or 0, reverse=reverse)

        # Paginate
        total = len(summaries)
        summaries = summaries[offset:offset + limit]

        return jsonify({
            "experiments": summaries,
            "total": total,
            "limit": limit,
            "offset": offset,
        })
    except Exception as e:
        logger.error(f"Failed to load experiment history: {e}")
        return jsonify({"experiments": [], "total": 0, "error": str(e)})


@app.route("/api/models/registry")
def api_models_registry():
    """Get model registry for browsing."""
    try:
        registry_file = project_root / "experiments" / "model_registry.json"
        if not registry_file.exists():
            return jsonify([])

        with open(registry_file) as f:
            registry = json.load(f)

        # Extract key fields for display
        limit = request.args.get("limit", 50, type=int)
        offset = request.args.get("offset", 0, type=int)
        sort_by = request.args.get("sort", "created_at")
        sort_order = request.args.get("order", "desc")

        # Build summary list
        summaries = []
        for model_id, model in registry.items():
            config = model.get("config", {})
            summary = {
                "model_id": model_id,
                "experiment_id": model.get("experiment_id", ""),
                "experiment_type": config.get("experiment_type", ""),
                "created_at": model.get("created_at", ""),
                "cv_auc": model.get("cv_auc", 0),
                "test_auc": model.get("test_auc", 0),
                "backtest_sharpe": model.get("backtest_sharpe", 0),
                "backtest_win_rate": model.get("backtest_win_rate", 0),
                "backtest_total_return": model.get("backtest_total_return", 0),
                "wmes_score": model.get("wmes_score", 0),
                "is_promoted": model.get("is_promoted", False),
                "model_type": config.get("model", {}).get("model_type", ""),
                "dim_reduction": config.get("dim_reduction", {}).get("method", ""),
            }
            summaries.append(summary)

        # Sort
        reverse = sort_order == "desc"
        if summaries and sort_by in summaries[0]:
            summaries.sort(key=lambda x: x.get(sort_by, 0) or 0, reverse=reverse)

        # Paginate
        total = len(summaries)
        summaries = summaries[offset:offset + limit]

        return jsonify({
            "models": summaries,
            "total": total,
            "limit": limit,
            "offset": offset,
        })
    except Exception as e:
        logger.error(f"Failed to load model registry: {e}")
        return jsonify({"models": [], "total": 0, "error": str(e)})


# =============================================================================
# DASHBOARD HTML
# =============================================================================
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GIGA TRADER - Live Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Consolas', 'Monaco', monospace;
            background: #0a0a0f;
            color: #e0e0e0;
            min-height: 100vh;
            padding: 15px;
        }
        .container { max-width: 1600px; margin: 0 auto; }
        .header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
            padding-bottom: 15px;
            border-bottom: 1px solid #333;
        }
        h1 {
            font-size: 1.8rem;
            color: #00d4ff;
            font-weight: 400;
        }
        .pulse-container {
            display: flex;
            align-items: center;
            gap: 15px;
        }
        .pulse {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            background: #00ff88;
            animation: pulse 1s infinite;
        }
        .pulse.disconnected {
            background: #ff4444;
            animation: none;
        }
        .pulse.warning {
            background: #ffaa00;
        }
        @keyframes pulse {
            0%, 100% { opacity: 1; transform: scale(1); }
            50% { opacity: 0.5; transform: scale(0.9); }
        }
        .pulse-info {
            font-size: 0.85rem;
            color: #888;
        }
        .pulse-info .latency { color: #00ff88; }
        .pulse-info .latency.slow { color: #ffaa00; }
        .pulse-info .latency.error { color: #ff4444; }

        .grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 20px; }
        .grid-3 { grid-template-columns: repeat(3, 1fr); }
        .grid-2 { grid-template-columns: 1fr 1fr; }
        @media (max-width: 1200px) { .grid { grid-template-columns: repeat(2, 1fr); } }
        @media (max-width: 768px) { .grid, .grid-3, .grid-2 { grid-template-columns: 1fr; } }

        .card {
            background: #12121a;
            border: 1px solid #2a2a3a;
            border-radius: 8px;
            padding: 15px;
        }
        .card h2 {
            font-size: 0.9rem;
            color: #00d4ff;
            margin-bottom: 12px;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .metric {
            display: flex;
            justify-content: space-between;
            padding: 6px 0;
            border-bottom: 1px solid #1a1a2a;
            font-size: 0.85rem;
        }
        .metric:last-child { border-bottom: none; }
        .metric-label { color: #666; }
        .metric-value { font-weight: 500; }
        .metric-value.positive { color: #00ff88; }
        .metric-value.negative { color: #ff4444; }

        .status-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 3px;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
        }
        .status-healthy, .status-running { background: #00ff8822; color: #00ff88; }
        .status-experimenting, .status-idle { background: #7b2cbf22; color: #b388ff; }
        .status-stopped { background: #ff444422; color: #ff4444; }
        .status-trading { background: #00d4ff22; color: #00d4ff; }

        .chart-container { height: 200px; margin: 10px 0; }

        /* Logs Section */
        .logs-container {
            background: #0a0a0f;
            border: 1px solid #2a2a3a;
            border-radius: 4px;
            height: 300px;
            overflow-y: auto;
            font-family: 'Consolas', monospace;
            font-size: 0.75rem;
            padding: 10px;
        }
        .log-entry {
            padding: 3px 0;
            border-bottom: 1px solid #1a1a2a;
            display: flex;
            gap: 10px;
        }
        .log-entry:hover { background: #1a1a2a; }
        .log-time { color: #666; min-width: 80px; }
        .log-level { min-width: 70px; font-weight: 600; }
        .log-level.INFO { color: #00d4ff; }
        .log-level.WARNING { color: #ffaa00; }
        .log-level.ERROR, .log-level.CRITICAL { color: #ff4444; }
        .log-level.DEBUG { color: #888; }
        .log-component { color: #7b2cbf; min-width: 120px; }
        .log-message { color: #ccc; flex: 1; word-break: break-word; }

        /* Request Monitor */
        .request-entry {
            display: flex;
            gap: 10px;
            padding: 4px 0;
            border-bottom: 1px solid #1a1a2a;
            font-size: 0.75rem;
        }
        .request-method { min-width: 40px; color: #00d4ff; }
        .request-path { flex: 1; color: #ccc; }
        .request-status { min-width: 30px; }
        .request-status.success { color: #00ff88; }
        .request-status.error { color: #ff4444; }
        .request-duration { min-width: 60px; color: #666; text-align: right; }

        table { width: 100%; border-collapse: collapse; font-size: 0.8rem; }
        th, td { padding: 8px; text-align: left; border-bottom: 1px solid #1a1a2a; }
        th { color: #00d4ff; font-weight: 500; text-transform: uppercase; font-size: 0.7rem; }

        .trade-row.profit { background: #00ff8808; }
        .trade-row.loss { background: #ff444408; }

        .tabs {
            display: flex;
            gap: 5px;
            margin-bottom: 10px;
        }
        .tab {
            padding: 6px 12px;
            background: #1a1a2a;
            border: 1px solid #2a2a3a;
            border-radius: 4px;
            color: #888;
            cursor: pointer;
            font-size: 0.75rem;
        }
        .tab:hover { background: #2a2a3a; }
        .tab.active { background: #00d4ff22; color: #00d4ff; border-color: #00d4ff; }

        .filter-input {
            background: #1a1a2a;
            border: 1px solid #2a2a3a;
            border-radius: 4px;
            padding: 6px 10px;
            color: #ccc;
            font-size: 0.8rem;
            width: 200px;
        }
        .filter-input:focus { outline: none; border-color: #00d4ff; }

        .no-data { text-align: center; color: #444; padding: 30px; font-style: italic; }
        .footer { text-align: center; color: #444; font-size: 0.75rem; margin-top: 20px; padding-top: 15px; border-top: 1px solid #2a2a3a; }

        /* Experiment Progress Styles */
        .progress-container {
            background: #1a1a2a;
            border-radius: 4px;
            height: 8px;
            overflow: hidden;
        }
        .progress-bar {
            height: 100%;
            background: linear-gradient(90deg, #7b2cbf, #00d4ff);
            border-radius: 4px;
            transition: width 0.5s ease;
        }
        .activity-indicator {
            display: flex;
            align-items: center;
            gap: 8px;
            font-size: 0.8rem;
        }
        .activity-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #666;
        }
        .activity-indicator.active .activity-dot {
            background: #00ff88;
            animation: pulse 1s infinite;
        }
        .activity-indicator.warning .activity-dot {
            background: #ffaa00;
            animation: pulse 0.5s infinite;
        }
        .activity-indicator.stuck .activity-dot {
            background: #ff4444;
        }
        .activity-text { color: #888; }
        .activity-indicator.active .activity-text { color: #00ff88; }
        .activity-indicator.warning .activity-text { color: #ffaa00; }
        .activity-indicator.stuck .activity-text { color: #ff4444; }

        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
            gap: 10px;
        }
        .metric-box {
            background: #1a1a2a;
            border-radius: 4px;
            padding: 8px 10px;
            text-align: center;
        }
        .metric-box-label {
            font-size: 0.65rem;
            color: #666;
            text-transform: uppercase;
            margin-bottom: 4px;
        }
        .metric-box-value {
            font-size: 0.9rem;
            font-weight: 500;
            color: #00d4ff;
        }
        .metric-box-value.good { color: #00ff88; }
        .metric-box-value.warning { color: #ffaa00; }
        .metric-box-value.bad { color: #ff4444; }

        .experiment-entry {
            display: flex;
            justify-content: space-between;
            padding: 6px 8px;
            border-bottom: 1px solid #1a1a2a;
            font-size: 0.75rem;
        }
        .experiment-entry:hover { background: #1a1a2a; }
        .experiment-entry.success { border-left: 2px solid #00ff88; }
        .experiment-entry.failed { border-left: 2px solid #ff4444; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>GIGA TRADER</h1>
            <div class="pulse-container">
                <div id="pulse" class="pulse"></div>
                <div class="pulse-info">
                    <span id="connectionStatus">Connecting...</span>
                    <span id="latency" class="latency"></span>
                </div>
            </div>
        </div>

        <!-- Status Cards -->
        <div class="grid">
            <div class="card">
                <h2>System</h2>
                <div class="metric">
                    <span class="metric-label">Mode</span>
                    <span id="systemMode" class="status-badge">--</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Health</span>
                    <span id="systemHealth" class="status-badge status-healthy">--</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Uptime</span>
                    <span id="uptime" class="metric-value">--</span>
                </div>
            </div>

            <div class="card">
                <h2>Trading</h2>
                <div class="metric">
                    <span class="metric-label">Position</span>
                    <span id="position" class="metric-value">FLAT</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Position P&L</span>
                    <span id="positionPnl" class="metric-value">$0.00</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Daily P&L</span>
                    <span id="dailyPnl" class="metric-value">$0.00</span>
                </div>
            </div>

            <div class="card">
                <h2>API Stats</h2>
                <div class="metric">
                    <span class="metric-label">Total Requests</span>
                    <span id="totalRequests" class="metric-value">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Req/min</span>
                    <span id="reqPerMin" class="metric-value">0</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Server Uptime</span>
                    <span id="serverUptime" class="metric-value">--</span>
                </div>
            </div>

            <div class="card">
                <h2>Components</h2>
                <div id="componentsContainer">
                    <div class="no-data">Loading...</div>
                </div>
            </div>
        </div>

        <!-- Experiment Progress Section -->
        <div class="card" style="margin-bottom: 20px;" id="experimentCard">
            <div style="display: flex; justify-content: space-between; align-items: center;">
                <h2 style="margin: 0;">Experiment Progress</h2>
                <div id="experimentActivity" class="activity-indicator">
                    <span class="activity-dot"></span>
                    <span class="activity-text">Idle</span>
                </div>
            </div>

            <div id="experimentContent" style="margin-top: 15px;">
                <!-- Progress bar -->
                <div class="progress-container">
                    <div class="progress-bar" id="experimentProgress" style="width: 0%"></div>
                </div>
                <div style="display: flex; justify-content: space-between; font-size: 0.75rem; margin-top: 5px;">
                    <span id="experimentStep">Step 0/10</span>
                    <span id="experimentElapsed">--</span>
                </div>

                <!-- Current step info -->
                <div class="experiment-info" style="margin-top: 15px;">
                    <div class="metric">
                        <span class="metric-label">Current Step</span>
                        <span id="currentStepName" class="metric-value">Idle</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Experiment Type</span>
                        <span id="experimentType" class="metric-value">--</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Substep</span>
                        <span id="experimentSubstep" class="metric-value" style="font-size: 0.8rem; color: #888;">--</span>
                    </div>
                </div>

                <!-- Live metrics -->
                <div style="margin-top: 15px;">
                    <h3 style="font-size: 0.8rem; color: #00d4ff; margin-bottom: 10px;">LIVE METRICS</h3>
                    <div id="liveMetrics" class="metrics-grid">
                        <div class="no-data">No metrics yet</div>
                    </div>
                </div>

                <!-- Recent experiments -->
                <div style="margin-top: 15px;">
                    <h3 style="font-size: 0.8rem; color: #00d4ff; margin-bottom: 10px;">RECENT EXPERIMENTS</h3>
                    <div id="recentExperiments" style="max-height: 120px; overflow-y: auto;">
                        <div class="no-data">No experiments yet</div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Experiments & Models Browser -->
        <div class="card" style="margin-bottom: 20px;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
                <h2 style="margin: 0;">Experiments & Models Browser</h2>
                <div class="tabs" id="browserTabs" style="margin: 0;">
                    <div class="tab active" data-tab="experiments">Experiments</div>
                    <div class="tab" data-tab="models">Models</div>
                </div>
            </div>

            <!-- Experiments Tab -->
            <div id="experimentsTab" class="browser-tab">
                <div style="display: flex; gap: 10px; margin-bottom: 10px; flex-wrap: wrap;">
                    <select id="expSortBy" class="filter-input" style="width: auto;">
                        <option value="started_at">Sort by Date</option>
                        <option value="test_auc">Sort by Test AUC</option>
                        <option value="backtest_sharpe">Sort by Sharpe</option>
                        <option value="wmes_score">Sort by WMES</option>
                        <option value="backtest_win_rate">Sort by Win Rate</option>
                    </select>
                    <select id="expSortOrder" class="filter-input" style="width: auto;">
                        <option value="desc">Descending</option>
                        <option value="asc">Ascending</option>
                    </select>
                    <span id="expCount" style="color: #888; font-size: 0.8rem; align-self: center;">Loading...</span>
                </div>
                <div id="experimentsTable" style="max-height: 400px; overflow-y: auto;">
                    <table style="width: 100%; border-collapse: collapse; font-size: 0.75rem;">
                        <thead style="position: sticky; top: 0; background: #12121a;">
                            <tr>
                                <th style="text-align: left; padding: 8px; border-bottom: 1px solid #2a2a3a;">Type</th>
                                <th style="text-align: left; padding: 8px; border-bottom: 1px solid #2a2a3a;">Date</th>
                                <th style="text-align: right; padding: 8px; border-bottom: 1px solid #2a2a3a;">Test AUC</th>
                                <th style="text-align: right; padding: 8px; border-bottom: 1px solid #2a2a3a;" title="NET Sharpe (after slippage & commission)">Sharpe*</th>
                                <th style="text-align: right; padding: 8px; border-bottom: 1px solid #2a2a3a;" title="NET Win Rate (after transaction costs)">Win %*</th>
                                <th style="text-align: right; padding: 8px; border-bottom: 1px solid #2a2a3a;">WMES</th>
                                <th style="text-align: right; padding: 8px; border-bottom: 1px solid #2a2a3a;" title="NET Return (after transaction costs)">Return*</th>
                                <th style="text-align: center; padding: 8px; border-bottom: 1px solid #2a2a3a;">Status</th>
                            </tr>
                            <tr style="font-size: 0.6rem; color: #666;">
                                <td colspan="8" style="padding: 2px 8px; border-bottom: 1px solid #1a1a2a;">* NET metrics shown (after 5bps slippage + 1bps commission per trade)</td>
                            </tr>
                        </thead>
                        <tbody id="expTableBody">
                            <tr><td colspan="8" class="no-data">Loading experiments...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <!-- Models Tab -->
            <div id="modelsTab" class="browser-tab" style="display: none;">
                <div style="display: flex; gap: 10px; margin-bottom: 10px; flex-wrap: wrap;">
                    <select id="modelSortBy" class="filter-input" style="width: auto;">
                        <option value="created_at">Sort by Date</option>
                        <option value="test_auc">Sort by Test AUC</option>
                        <option value="backtest_sharpe">Sort by Sharpe</option>
                        <option value="wmes_score">Sort by WMES</option>
                    </select>
                    <select id="modelSortOrder" class="filter-input" style="width: auto;">
                        <option value="desc">Descending</option>
                        <option value="asc">Ascending</option>
                    </select>
                    <span id="modelCount" style="color: #888; font-size: 0.8rem; align-self: center;">Loading...</span>
                </div>
                <div id="modelsTableContainer" style="max-height: 400px; overflow-y: auto;">
                    <table style="width: 100%; border-collapse: collapse; font-size: 0.75rem;">
                        <thead style="position: sticky; top: 0; background: #12121a;">
                            <tr>
                                <th style="text-align: left; padding: 8px; border-bottom: 1px solid #2a2a3a;">Model ID</th>
                                <th style="text-align: left; padding: 8px; border-bottom: 1px solid #2a2a3a;">Type</th>
                                <th style="text-align: left; padding: 8px; border-bottom: 1px solid #2a2a3a;">Date</th>
                                <th style="text-align: right; padding: 8px; border-bottom: 1px solid #2a2a3a;">Test AUC</th>
                                <th style="text-align: right; padding: 8px; border-bottom: 1px solid #2a2a3a;" title="NET Sharpe (after costs)">Sharpe*</th>
                                <th style="text-align: right; padding: 8px; border-bottom: 1px solid #2a2a3a;" title="NET Win Rate (after costs)">Win %*</th>
                                <th style="text-align: right; padding: 8px; border-bottom: 1px solid #2a2a3a;" title="NET Return (after costs)">Return*</th>
                                <th style="text-align: center; padding: 8px; border-bottom: 1px solid #2a2a3a;">Promoted</th>
                            </tr>
                            <tr style="font-size: 0.6rem; color: #666;">
                                <td colspan="8" style="padding: 2px 8px; border-bottom: 1px solid #1a1a2a;">* NET metrics (after transaction costs)</td>
                            </tr>
                        </thead>
                        <tbody id="modelTableBody">
                            <tr><td colspan="8" class="no-data">Loading models...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <!-- Charts Row -->
        <div class="grid grid-3" style="margin-bottom: 20px;">
            <div class="card">
                <h2>Equity</h2>
                <div class="chart-container">
                    <canvas id="equityChart"></canvas>
                </div>
            </div>
            <div class="card">
                <h2>Daily P&L</h2>
                <div class="chart-container">
                    <canvas id="pnlChart"></canvas>
                </div>
            </div>
            <div class="card">
                <h2>Position</h2>
                <div class="chart-container">
                    <canvas id="positionChart"></canvas>
                </div>
            </div>
        </div>

        <!-- Logs and Requests Row -->
        <div class="grid grid-2" style="margin-bottom: 20px;">
            <div class="card">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
                    <h2 style="margin: 0;">Live Logs</h2>
                    <input type="text" id="logFilter" class="filter-input" placeholder="Filter logs...">
                </div>
                <div class="tabs">
                    <div class="tab active" data-level="all">All</div>
                    <div class="tab" data-level="WARNING">Warnings</div>
                    <div class="tab" data-level="ERROR">Errors</div>
                    <div class="tab" data-level="INFO">Info</div>
                </div>
                <div id="logsContainer" class="logs-container">
                    <div class="no-data">Loading logs...</div>
                </div>
            </div>

            <div class="card">
                <h2>API Requests</h2>
                <div id="requestsContainer" style="max-height: 350px; overflow-y: auto;">
                    <div class="no-data">Loading...</div>
                </div>
            </div>
        </div>

        <!-- Trades and Models -->
        <div class="grid grid-2">
            <div class="card">
                <h2>Recent Trades</h2>
                <div style="max-height: 250px; overflow-y: auto;">
                    <table>
                        <thead>
                            <tr><th>Time</th><th>Side</th><th>Qty</th><th>Price</th><th>P&L</th></tr>
                        </thead>
                        <tbody id="tradesTable">
                            <tr><td colspan="5" class="no-data">No trades</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>

            <div class="card">
                <h2>Models</h2>
                <div style="max-height: 250px; overflow-y: auto;">
                    <table>
                        <thead>
                            <tr><th>Name</th><th>Type</th><th>Features</th></tr>
                        </thead>
                        <tbody id="modelsTable">
                            <tr><td colspan="3" class="no-data">Loading...</td></tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>

        <div class="footer">
            GIGA TRADER Dashboard | Auto-refresh: 3s | <span id="lastUpdate">--</span>
        </div>
    </div>

    <script>
        // Chart instances
        let equityChart = null;
        let pnlChart = null;
        let positionChart = null;
        let currentLogLevel = 'all';
        let logFilter = '';

        // Chart.js defaults
        Chart.defaults.color = '#666';
        Chart.defaults.borderColor = '#2a2a3a';

        const chartOptions = {
            responsive: true,
            maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { display: false },
                y: { grid: { color: '#1a1a2a' }, ticks: { callback: v => '$' + v.toLocaleString() } }
            },
            elements: { point: { radius: 0 } }
        };

        function initCharts() {
            equityChart = new Chart(document.getElementById('equityChart'), {
                type: 'line',
                data: { labels: [], datasets: [{ data: [], borderColor: '#00d4ff', backgroundColor: 'rgba(0,212,255,0.1)', fill: true, tension: 0.3 }] },
                options: chartOptions
            });
            pnlChart = new Chart(document.getElementById('pnlChart'), {
                type: 'bar',
                data: { labels: [], datasets: [{ data: [], backgroundColor: [] }] },
                options: { ...chartOptions, scales: { ...chartOptions.scales, y: { ...chartOptions.scales.y } } }
            });
            positionChart = new Chart(document.getElementById('positionChart'), {
                type: 'line',
                data: { labels: [], datasets: [{ data: [], borderColor: '#7b2cbf', backgroundColor: 'rgba(123,44,191,0.1)', fill: true, tension: 0.3 }] },
                options: chartOptions
            });
        }

        // Heartbeat
        let lastHeartbeat = Date.now();
        let heartbeatLatency = 0;

        async function checkHeartbeat() {
            const start = Date.now();
            try {
                const response = await fetch('/api/heartbeat');
                if (response.ok) {
                    heartbeatLatency = Date.now() - start;
                    lastHeartbeat = Date.now();

                    document.getElementById('pulse').className = 'pulse';
                    document.getElementById('connectionStatus').textContent = 'Connected';
                    document.getElementById('latency').textContent = `(${heartbeatLatency}ms)`;
                    document.getElementById('latency').className = heartbeatLatency > 500 ? 'latency slow' : 'latency';
                }
            } catch (e) {
                document.getElementById('pulse').className = 'pulse disconnected';
                document.getElementById('connectionStatus').textContent = 'Disconnected';
                document.getElementById('latency').textContent = '';
            }
        }

        async function updateStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();

                document.getElementById('systemMode').textContent = data.mode || '--';
                document.getElementById('systemMode').className = 'status-badge status-' + (data.mode || '').toLowerCase();
                document.getElementById('systemHealth').textContent = data.health?.status || '--';

                const uptime = data.uptime_seconds || 0;
                document.getElementById('uptime').textContent = Math.floor(uptime/3600) + 'h ' + Math.floor((uptime%3600)/60) + 'm';

                document.getElementById('position').textContent = data.trading?.position || 'FLAT';
                const positionPnl = data.trading?.position_pnl || 0;
                document.getElementById('positionPnl').textContent = '$' + positionPnl.toFixed(2);
                document.getElementById('positionPnl').className = 'metric-value ' + (positionPnl >= 0 ? 'positive' : 'negative');

                const dailyPnl = data.trading?.daily_pnl || 0;
                document.getElementById('dailyPnl').textContent = '$' + dailyPnl.toFixed(2);
                document.getElementById('dailyPnl').className = 'metric-value ' + (dailyPnl >= 0 ? 'positive' : 'negative');

                let componentsHtml = '';
                for (const [name, state] of Object.entries(data.components || {})) {
                    const shortName = name.replace('_engine', '').replace('_', ' ');
                    const statusClass = ['RUNNING','IDLE'].includes(state) ? 'running' : 'stopped';
                    componentsHtml += `<div class="metric"><span class="metric-label">${shortName}</span><span class="status-badge status-${statusClass}">${state}</span></div>`;
                }
                document.getElementById('componentsContainer').innerHTML = componentsHtml || '<div class="no-data">No components</div>';

                document.getElementById('lastUpdate').textContent = new Date().toLocaleTimeString();
            } catch (e) {
                console.error('Status update failed:', e);
            }
        }

        async function updateCharts() {
            try {
                const equityResponse = await fetch('/api/equity');
                const equityData = await equityResponse.json();

                if (equityData.length > 0) {
                    equityChart.data.labels = equityData.map(d => d.x.substring(11,19));
                    equityChart.data.datasets[0].data = equityData.map(d => d.y);
                    equityChart.update('none');

                    pnlChart.data.labels = equityData.map(d => d.x.substring(11,19));
                    pnlChart.data.datasets[0].data = equityData.map(d => d.pnl);
                    pnlChart.data.datasets[0].backgroundColor = equityData.map(d => d.pnl >= 0 ? '#00ff8888' : '#ff444488');
                    pnlChart.update('none');
                }

                const posResponse = await fetch('/api/positions');
                const posData = await posResponse.json();
                if (posData.length > 0) {
                    positionChart.data.labels = posData.map(d => d.x.substring(11,19));
                    positionChart.data.datasets[0].data = posData.map(d => d.y);
                    positionChart.update('none');
                }
            } catch (e) {
                console.error('Chart update failed:', e);
            }
        }

        async function updateLogs() {
            try {
                let url = '/api/logs?lines=100';
                if (logFilter) url += '&filter=' + encodeURIComponent(logFilter);

                const response = await fetch(url);
                const data = await response.json();

                let entries = data.entries || [];

                // Filter by level
                if (currentLogLevel !== 'all') {
                    entries = entries.filter(e => e.level === currentLogLevel);
                }

                if (entries.length === 0) {
                    document.getElementById('logsContainer').innerHTML = '<div class="no-data">No log entries</div>';
                    return;
                }

                let html = '';
                for (const e of entries.slice(-50).reverse()) {
                    html += `<div class="log-entry">
                        <span class="log-time">${(e.timestamp || '').substring(11,19)}</span>
                        <span class="log-level ${e.level}">${e.level}</span>
                        <span class="log-component">${e.component || ''}</span>
                        <span class="log-message">${e.message || ''}</span>
                    </div>`;
                }
                document.getElementById('logsContainer').innerHTML = html;
            } catch (e) {
                console.error('Logs update failed:', e);
            }
        }

        async function updateRequests() {
            try {
                const response = await fetch('/api/requests');
                const data = await response.json();

                document.getElementById('totalRequests').textContent = data.stats?.total_requests || 0;
                document.getElementById('reqPerMin').textContent = data.stats?.requests_per_minute || 0;

                const uptime = data.stats?.uptime_seconds || 0;
                document.getElementById('serverUptime').textContent = Math.floor(uptime/60) + 'm ' + (uptime%60) + 's';

                const requests = data.requests || [];
                if (requests.length === 0) {
                    document.getElementById('requestsContainer').innerHTML = '<div class="no-data">No requests</div>';
                    return;
                }

                let html = '';
                for (const r of requests.slice(0, 20)) {
                    const statusClass = r.status < 400 ? 'success' : 'error';
                    html += `<div class="request-entry">
                        <span class="request-method">${r.method}</span>
                        <span class="request-path">${r.path}</span>
                        <span class="request-status ${statusClass}">${r.status}</span>
                        <span class="request-duration">${r.duration_ms}ms</span>
                    </div>`;
                }
                document.getElementById('requestsContainer').innerHTML = html;
            } catch (e) {
                console.error('Requests update failed:', e);
            }
        }

        async function updateTrades() {
            try {
                const response = await fetch('/api/trades');
                const trades = await response.json();

                if (trades.length === 0) {
                    document.getElementById('tradesTable').innerHTML = '<tr><td colspan="5" class="no-data">No trades</td></tr>';
                    return;
                }

                let html = '';
                for (const t of trades.slice(-10).reverse()) {
                    const pnl = t.pnl || 0;
                    const rowClass = pnl > 0 ? 'profit' : (pnl < 0 ? 'loss' : '');
                    html += `<tr class="trade-row ${rowClass}">
                        <td>${(t.timestamp || '').substring(11,19)}</td>
                        <td style="color:${t.side==='buy'?'#00ff88':'#ff4444'}">${(t.side||'').toUpperCase()}</td>
                        <td>${t.quantity || 0}</td>
                        <td>$${(t.price || 0).toFixed(2)}</td>
                        <td class="${pnl>=0?'positive':'negative'}">$${pnl.toFixed(2)}</td>
                    </tr>`;
                }
                document.getElementById('tradesTable').innerHTML = html;
            } catch (e) {
                console.error('Trades update failed:', e);
            }
        }

        async function updateModels() {
            try {
                const response = await fetch('/api/models');
                const models = await response.json();

                if (models.length === 0) {
                    document.getElementById('modelsTable').innerHTML = '<tr><td colspan="3" class="no-data">No models</td></tr>';
                    return;
                }

                let html = '';
                for (const m of models) {
                    html += `<tr>
                        <td>${m.name}</td>
                        <td>${m.swing_type}</td>
                        <td>${m.features}</td>
                    </tr>`;
                }
                document.getElementById('modelsTable').innerHTML = html;
            } catch (e) {
                console.error('Models update failed:', e);
            }
        }

        async function updateExperiment() {
            try {
                const response = await fetch('/api/experiment');
                const data = await response.json();

                const activityEl = document.getElementById('experimentActivity');
                const progressBar = document.getElementById('experimentProgress');
                const stepEl = document.getElementById('experimentStep');
                const elapsedEl = document.getElementById('experimentElapsed');
                const stepNameEl = document.getElementById('currentStepName');
                const typeEl = document.getElementById('experimentType');
                const substepEl = document.getElementById('experimentSubstep');

                // Update activity indicator
                const secsSinceActivity = data.seconds_since_activity || 0;
                if (data.is_running) {
                    if (data.stuck_critical) {
                        activityEl.className = 'activity-indicator stuck';
                        activityEl.querySelector('.activity-text').textContent = `STUCK (${Math.floor(secsSinceActivity)}s)`;
                    } else if (data.stuck_warning) {
                        activityEl.className = 'activity-indicator warning';
                        activityEl.querySelector('.activity-text').textContent = `Slow (${Math.floor(secsSinceActivity)}s)`;
                    } else {
                        activityEl.className = 'activity-indicator active';
                        activityEl.querySelector('.activity-text').textContent = 'Running';
                    }
                } else {
                    activityEl.className = 'activity-indicator';
                    activityEl.querySelector('.activity-text').textContent = 'Idle';
                }

                // Update progress bar
                const progressPct = (data.step_number / data.total_steps) * 100;
                progressBar.style.width = progressPct + '%';

                // Update step info
                stepEl.textContent = `Step ${data.step_number}/${data.total_steps}`;

                // Format elapsed time
                const elapsed = Math.floor(data.elapsed_seconds || 0);
                const mins = Math.floor(elapsed / 60);
                const secs = elapsed % 60;
                elapsedEl.textContent = data.is_running ? `${mins}m ${secs}s` : '--';

                // Update step details
                stepNameEl.textContent = data.step_description || 'Idle';
                typeEl.textContent = data.experiment_type || '--';
                substepEl.textContent = data.substep || '--';

                // Update live metrics
                const metricsEl = document.getElementById('liveMetrics');
                const metrics = data.live_metrics || {};
                if (Object.keys(metrics).length > 0) {
                    let html = '';
                    for (const [key, value] of Object.entries(metrics)) {
                        const displayKey = key.replace(/_/g, ' ');
                        let displayValue = typeof value === 'number' ? value.toFixed(4) : value;
                        let valueClass = '';

                        // Add color coding for known metrics
                        if (key.includes('auc') && typeof value === 'number') {
                            valueClass = value > 0.7 ? 'good' : (value < 0.55 ? 'bad' : 'warning');
                            displayValue = (value * 100).toFixed(1) + '%';
                        } else if (key.includes('sharpe') && typeof value === 'number') {
                            valueClass = value > 1.0 ? 'good' : (value < 0.5 ? 'warning' : '');
                        } else if (key.includes('win_rate') && typeof value === 'number') {
                            displayValue = (value * 100).toFixed(1) + '%';
                            valueClass = value > 0.6 ? 'good' : (value < 0.45 ? 'bad' : '');
                        } else if (typeof value === 'number' && value > 1000) {
                            displayValue = value.toLocaleString();
                        }

                        html += `<div class="metric-box">
                            <div class="metric-box-label">${displayKey}</div>
                            <div class="metric-box-value ${valueClass}">${displayValue}</div>
                        </div>`;
                    }
                    metricsEl.innerHTML = html;
                } else {
                    metricsEl.innerHTML = '<div class="no-data" style="padding: 10px;">No metrics yet</div>';
                }

                // Update recent experiments
                const recentEl = document.getElementById('recentExperiments');
                const recent = data.recent_experiments || [];
                if (recent.length > 0) {
                    let html = '';
                    for (const exp of recent.slice(-5).reverse()) {
                        const statusClass = exp.success ? 'success' : 'failed';
                        const duration = Math.floor(exp.duration_seconds || 0);
                        const testAuc = exp.test_auc ? (exp.test_auc * 100).toFixed(1) + '%' : '--';
                        html += `<div class="experiment-entry ${statusClass}">
                            <span style="color: #00d4ff;">${exp.experiment_type || 'experiment'}</span>
                            <span style="color: #888;">${duration}s</span>
                            <span style="color: ${exp.success ? '#00ff88' : '#ff4444'};">${testAuc}</span>
                        </div>`;
                    }
                    recentEl.innerHTML = html;
                } else {
                    recentEl.innerHTML = '<div class="no-data" style="padding: 10px;">No experiments yet</div>';
                }

            } catch (e) {
                console.error('Experiment update failed:', e);
            }
        }

        // Experiments & Models Browser
        async function updateExperimentsBrowser() {
            try {
                const sortBy = document.getElementById('expSortBy').value;
                const sortOrder = document.getElementById('expSortOrder').value;
                const response = await fetch(`/api/experiments/history?sort=${sortBy}&order=${sortOrder}&limit=100`);
                const data = await response.json();

                document.getElementById('expCount').textContent = `${data.total} experiments`;

                const tbody = document.getElementById('expTableBody');
                if (data.experiments && data.experiments.length > 0) {
                    let html = '';
                    for (const exp of data.experiments) {
                        const date = exp.started_at ? new Date(exp.started_at).toLocaleString() : '--';
                        const testAuc = exp.test_auc ? (exp.test_auc * 100).toFixed(1) + '%' : '--';
                        const sharpe = exp.backtest_sharpe ? exp.backtest_sharpe.toFixed(2) : '--';
                        const winRate = exp.backtest_win_rate ? (exp.backtest_win_rate * 100).toFixed(1) + '%' : '--';
                        const wmes = exp.wmes_score ? exp.wmes_score.toFixed(3) : '--';
                        const totalReturn = exp.backtest_total_return ? (exp.backtest_total_return * 100).toFixed(1) + '%' : '--';
                        const status = exp.status || 'unknown';
                        const statusColor = status === 'COMPLETED' ? '#00ff88' : (status === 'FAILED' ? '#ff4444' : '#888');

                        // Color coding
                        const aucColor = exp.test_auc > 0.7 ? '#00ff88' : (exp.test_auc < 0.55 ? '#ff4444' : '#ffaa00');
                        const sharpeColor = exp.backtest_sharpe > 1 ? '#00ff88' : (exp.backtest_sharpe < 0.5 ? '#ffaa00' : '#ccc');

                        html += `<tr style="border-bottom: 1px solid #1a1a2a;">
                            <td style="padding: 8px; color: #00d4ff;">${exp.experiment_type || '--'}</td>
                            <td style="padding: 8px; color: #888; font-size: 0.7rem;">${date}</td>
                            <td style="padding: 8px; text-align: right; color: ${aucColor};">${testAuc}</td>
                            <td style="padding: 8px; text-align: right; color: ${sharpeColor};">${sharpe}</td>
                            <td style="padding: 8px; text-align: right;">${winRate}</td>
                            <td style="padding: 8px; text-align: right;">${wmes}</td>
                            <td style="padding: 8px; text-align: right; color: ${exp.backtest_total_return > 0 ? '#00ff88' : '#ff4444'};">${totalReturn}</td>
                            <td style="padding: 8px; text-align: center; color: ${statusColor};">${status}</td>
                        </tr>`;
                    }
                    tbody.innerHTML = html;
                } else {
                    tbody.innerHTML = '<tr><td colspan="8" class="no-data">No experiments found</td></tr>';
                }
            } catch (e) {
                console.error('Experiments browser update failed:', e);
            }
        }

        async function updateModelsBrowser() {
            try {
                const sortBy = document.getElementById('modelSortBy').value;
                const sortOrder = document.getElementById('modelSortOrder').value;
                const response = await fetch(`/api/models/registry?sort=${sortBy}&order=${sortOrder}&limit=100`);
                const data = await response.json();

                document.getElementById('modelCount').textContent = `${data.total} models`;

                const tbody = document.getElementById('modelTableBody');
                if (data.models && data.models.length > 0) {
                    let html = '';
                    for (const model of data.models) {
                        const date = model.created_at ? new Date(model.created_at).toLocaleString() : '--';
                        const testAuc = model.test_auc ? (model.test_auc * 100).toFixed(1) + '%' : '--';
                        const sharpe = model.backtest_sharpe ? model.backtest_sharpe.toFixed(2) : '--';
                        const winRate = model.backtest_win_rate ? (model.backtest_win_rate * 100).toFixed(1) + '%' : '--';
                        const totalReturn = model.backtest_total_return ? (model.backtest_total_return * 100).toFixed(1) + '%' : '--';
                        const promoted = model.is_promoted ? '✓' : '';

                        // Color coding
                        const aucColor = model.test_auc > 0.7 ? '#00ff88' : (model.test_auc < 0.55 ? '#ff4444' : '#ffaa00');
                        const sharpeColor = model.backtest_sharpe > 1 ? '#00ff88' : (model.backtest_sharpe < 0.5 ? '#ffaa00' : '#ccc');

                        html += `<tr style="border-bottom: 1px solid #1a1a2a;">
                            <td style="padding: 8px; color: #7b2cbf; font-size: 0.7rem;">${model.model_id || '--'}</td>
                            <td style="padding: 8px; color: #00d4ff;">${model.experiment_type || '--'}</td>
                            <td style="padding: 8px; color: #888; font-size: 0.7rem;">${date}</td>
                            <td style="padding: 8px; text-align: right; color: ${aucColor};">${testAuc}</td>
                            <td style="padding: 8px; text-align: right; color: ${sharpeColor};">${sharpe}</td>
                            <td style="padding: 8px; text-align: right;">${winRate}</td>
                            <td style="padding: 8px; text-align: right; color: ${model.backtest_total_return > 0 ? '#00ff88' : '#ff4444'};">${totalReturn}</td>
                            <td style="padding: 8px; text-align: center; color: #00ff88;">${promoted}</td>
                        </tr>`;
                    }
                    tbody.innerHTML = html;
                } else {
                    tbody.innerHTML = '<tr><td colspan="8" class="no-data">No models found</td></tr>';
                }
            } catch (e) {
                console.error('Models browser update failed:', e);
            }
        }

        // Browser tab handlers
        document.querySelectorAll('#browserTabs .tab').forEach(tab => {
            tab.addEventListener('click', () => {
                document.querySelectorAll('#browserTabs .tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                const tabName = tab.dataset.tab;
                document.getElementById('experimentsTab').style.display = tabName === 'experiments' ? 'block' : 'none';
                document.getElementById('modelsTab').style.display = tabName === 'models' ? 'block' : 'none';
            });
        });

        // Sort change handlers
        document.getElementById('expSortBy').addEventListener('change', updateExperimentsBrowser);
        document.getElementById('expSortOrder').addEventListener('change', updateExperimentsBrowser);
        document.getElementById('modelSortBy').addEventListener('change', updateModelsBrowser);
        document.getElementById('modelSortOrder').addEventListener('change', updateModelsBrowser);

        // Log Tab handlers
        document.querySelectorAll('.tabs:not(#browserTabs) .tab').forEach(tab => {
            tab.addEventListener('click', () => {
                const parent = tab.parentElement;
                parent.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                if (tab.dataset.level) {
                    currentLogLevel = tab.dataset.level;
                    updateLogs();
                }
            });
        });

        // Filter handler
        document.getElementById('logFilter').addEventListener('input', (e) => {
            logFilter = e.target.value;
            updateLogs();
        });

        // Initialize
        document.addEventListener('DOMContentLoaded', () => {
            initCharts();
            checkHeartbeat();
            updateStatus();
            updateCharts();
            updateLogs();
            updateRequests();
            updateTrades();
            updateModels();
            updateExperiment();
            updateExperimentsBrowser();
            updateModelsBrowser();

            // Fast refresh for critical data
            setInterval(checkHeartbeat, 2000);
            setInterval(() => {
                updateStatus();
                updateLogs();
                updateRequests();
            }, 3000);

            // Experiment progress (refresh every 2 seconds for responsiveness)
            setInterval(updateExperiment, 2000);

            // Slower refresh for charts and models
            setInterval(updateCharts, 5000);
            setInterval(updateTrades, 5000);
            setInterval(updateModels, 30000);

            // Browser refresh (every 30 seconds - data doesn't change frequently)
            setInterval(updateExperimentsBrowser, 30000);
            setInterval(updateModelsBrowser, 30000);
        });
    </script>
</body>
</html>
"""


@app.route("/")
def dashboard():
    """Serve the live dashboard."""
    return render_template_string(DASHBOARD_HTML)


# =============================================================================
# MAIN
# =============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="GIGA TRADER Live Dashboard Server")
    parser.add_argument("--port", type=int, default=8050, help="Server port (default: 8050)")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host (default: 127.0.0.1)")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("GIGA TRADER - Live Dashboard Server")
    print("=" * 60)
    print(f"\n  Dashboard URL: http://{args.host}:{args.port}")
    print(f"\n  API Endpoints:")
    print(f"    GET /api/heartbeat  - Connection health check")
    print(f"    GET /api/health     - Full system health")
    print(f"    GET /api/status     - System status")
    print(f"    GET /api/logs       - Recent log entries")
    print(f"    GET /api/requests   - API request tracking")
    print(f"    GET /api/experiment - Experiment progress (real-time)")
    print(f"    GET /api/equity     - Equity history")
    print(f"    GET /api/positions  - Position history")
    print(f"    GET /api/trades     - Trade history")
    print(f"    GET /api/models     - Model information")
    print(f"\n  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
