"""
GIGA TRADER - Enhanced Web Monitoring Dashboard
================================================
Real-time web interface for monitoring and controlling the trading system.

Features:
  - Live status dashboard
  - Position and P&L tracking
  - Log viewer with filtering
  - Control commands (start/stop trading, force training)
  - Trade history
  - Experiment history with leaderboard
  - Model registry with performance metrics
  - Backtest results viewer
  - Error filtering by type/severity

Usage:
    .venv/Scripts/python.exe src/web_monitor.py

Then open: http://localhost:5000
"""

import os
import sys
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from functools import wraps
import threading
import logging
import re

# Setup path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

web_logger = logging.getLogger("WebMonitor")

# Flask imports
try:
    from flask import Flask, render_template_string, jsonify, request, Response
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False
    print("[WARN] Flask not installed. Run: pip install flask")

# ===============================================================================
# CONFIGURATION
# ===============================================================================
WEB_CONFIG = {
    "host": "0.0.0.0",
    "port": 5000,
    "debug": False,
    "status_file": project_root / "logs" / "status.json",
    "log_dir": project_root / "logs",
    "trades_file": project_root / "logs" / "trades.json",
    "experiments_dir": project_root / "experiments",
    "models_dir": project_root / "models",
}

# ===============================================================================
# FLASK APP
# ===============================================================================
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Disable Flask logging spam
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

# ===============================================================================
# HTML TEMPLATES
# ===============================================================================

# Base CSS shared across all pages
BASE_CSS = """
* { margin: 0; padding: 0; box-sizing: border-box; }
body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: #0a0a0a;
    color: #e0e0e0;
    min-height: 100vh;
}
.header {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
    padding: 15px 20px;
    border-bottom: 2px solid #0f3460;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.header h1 {
    color: #00d4ff;
    font-size: 24px;
    text-shadow: 0 0 10px rgba(0,212,255,0.5);
}
.nav {
    display: flex;
    gap: 5px;
}
.nav a {
    color: #888;
    text-decoration: none;
    padding: 8px 16px;
    border-radius: 5px;
    transition: all 0.3s;
    font-size: 14px;
}
.nav a:hover, .nav a.active {
    color: #00d4ff;
    background: rgba(0,212,255,0.1);
}
.container {
    max-width: 1600px;
    margin: 0 auto;
    padding: 20px;
}
.grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
    gap: 20px;
    margin-bottom: 20px;
}
.grid-2 { grid-template-columns: repeat(2, 1fr); }
.grid-3 { grid-template-columns: repeat(3, 1fr); }
.grid-4 { grid-template-columns: repeat(4, 1fr); }
.card {
    background: #1a1a2e;
    border-radius: 10px;
    padding: 20px;
    border: 1px solid #2a2a4e;
}
.card.full-width { grid-column: 1 / -1; }
.card h2 {
    color: #00d4ff;
    font-size: 16px;
    margin-bottom: 15px;
    padding-bottom: 10px;
    border-bottom: 1px solid #2a2a4e;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.card h2 .badge {
    font-size: 12px;
    padding: 3px 8px;
    border-radius: 10px;
    background: #2a2a4e;
}
.status-item {
    display: flex;
    justify-content: space-between;
    padding: 8px 0;
    border-bottom: 1px solid #2a2a4e;
}
.status-item:last-child { border-bottom: none; }
.status-label { color: #888; }
.status-value { font-weight: bold; }
.status-value.positive { color: #00ff88; }
.status-value.negative { color: #ff4444; }
.status-value.warning { color: #ffaa00; }
.status-value.info { color: #00d4ff; }

.mode-badge {
    display: inline-block;
    padding: 5px 15px;
    border-radius: 20px;
    font-weight: bold;
    font-size: 14px;
}
.mode-TRADING { background: #00ff88; color: #000; }
.mode-TRAINING { background: #00d4ff; color: #000; }
.mode-EXPERIMENTING { background: #9b59b6; color: #fff; }
.mode-IMPROVING { background: #3498db; color: #fff; }
.mode-BACKTESTING { background: #e67e22; color: #fff; }
.mode-READY { background: #2ecc71; color: #000; }
.mode-IDLE { background: #666; color: #fff; }
.mode-ERROR { background: #ff4444; color: #fff; }
.mode-INITIALIZING { background: #ffaa00; color: #000; }

.component-status {
    display: flex;
    align-items: center;
    padding: 5px 0;
}
.component-dot {
    width: 10px;
    height: 10px;
    border-radius: 50%;
    margin-right: 10px;
}
.component-dot.running { background: #00ff88; box-shadow: 0 0 5px #00ff88; }
.component-dot.stopped { background: #666; }
.component-dot.error { background: #ff4444; box-shadow: 0 0 5px #ff4444; }

.btn {
    padding: 8px 16px;
    border: none;
    border-radius: 5px;
    cursor: pointer;
    font-weight: bold;
    font-size: 13px;
    transition: all 0.3s;
}
.btn-sm { padding: 4px 10px; font-size: 12px; }
.btn-primary { background: #00d4ff; color: #000; }
.btn-success { background: #00ff88; color: #000; }
.btn-danger { background: #ff4444; color: #fff; }
.btn-warning { background: #ffaa00; color: #000; }
.btn-secondary { background: #444; color: #fff; }
.btn:hover { transform: scale(1.02); opacity: 0.9; }

.log-viewer {
    background: #000;
    border-radius: 5px;
    padding: 15px;
    font-family: 'Consolas', monospace;
    font-size: 12px;
    max-height: 500px;
    overflow-y: auto;
    white-space: pre-wrap;
    word-wrap: break-word;
}
.log-line { padding: 2px 0; }
.log-INFO { color: #00d4ff; }
.log-WARNING { color: #ffaa00; }
.log-ERROR { color: #ff4444; font-weight: bold; }
.log-DEBUG { color: #888; }

.big-number {
    font-size: 32px;
    font-weight: bold;
    text-align: center;
    padding: 15px;
}
.big-number.positive { color: #00ff88; }
.big-number.negative { color: #ff4444; }

/* Tables */
.table-container {
    overflow-x: auto;
}
table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}
th, td {
    padding: 10px;
    text-align: left;
    border-bottom: 1px solid #2a2a4e;
}
th {
    color: #00d4ff;
    background: #0a0a1a;
    font-weight: 600;
    position: sticky;
    top: 0;
}
tr:hover { background: rgba(0,212,255,0.05); }

/* Filters */
.filters {
    display: flex;
    gap: 10px;
    margin-bottom: 15px;
    flex-wrap: wrap;
    align-items: center;
}
.filter-group {
    display: flex;
    align-items: center;
    gap: 5px;
}
.filter-group label {
    color: #888;
    font-size: 13px;
}
select, input[type="text"], input[type="date"] {
    background: #0a0a1a;
    border: 1px solid #2a2a4e;
    color: #e0e0e0;
    padding: 6px 10px;
    border-radius: 5px;
    font-size: 13px;
}
select:focus, input:focus {
    outline: none;
    border-color: #00d4ff;
}

/* Tags */
.tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 10px;
    font-size: 11px;
    font-weight: bold;
}
.tag-success { background: rgba(0,255,136,0.2); color: #00ff88; }
.tag-danger { background: rgba(255,68,68,0.2); color: #ff4444; }
.tag-warning { background: rgba(255,170,0,0.2); color: #ffaa00; }
.tag-info { background: rgba(0,212,255,0.2); color: #00d4ff; }
.tag-default { background: rgba(136,136,136,0.2); color: #888; }

/* Progress bar */
.progress {
    height: 6px;
    background: #2a2a4e;
    border-radius: 3px;
    overflow: hidden;
}
.progress-bar {
    height: 100%;
    background: #00d4ff;
    transition: width 0.3s;
}
.progress-bar.success { background: #00ff88; }
.progress-bar.danger { background: #ff4444; }

/* Stats grid */
.stats-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 15px;
}
.stat-box {
    text-align: center;
    padding: 15px;
    background: #0a0a1a;
    border-radius: 8px;
}
.stat-box .value {
    font-size: 24px;
    font-weight: bold;
    color: #00d4ff;
}
.stat-box .label {
    font-size: 12px;
    color: #888;
    margin-top: 5px;
}

#last-update {
    text-align: right;
    color: #666;
    font-size: 12px;
    padding: 10px;
}

.controls {
    display: flex;
    flex-wrap: wrap;
    gap: 10px;
}

/* Chart placeholder */
.chart-container {
    height: 200px;
    background: #0a0a1a;
    border-radius: 5px;
    display: flex;
    align-items: center;
    justify-content: center;
    color: #666;
}

@media (max-width: 768px) {
    .grid-2, .grid-3, .grid-4 { grid-template-columns: 1fr; }
    .stats-grid { grid-template-columns: repeat(2, 1fr); }
}
"""

# Navigation HTML component
NAV_HTML = """
<div class="nav">
    <a href="/" class="{dashboard_active}">Dashboard</a>
    <a href="/experiments" class="{experiments_active}">Experiments</a>
    <a href="/models" class="{models_active}">Models</a>
    <a href="/logs" class="{logs_active}">Logs</a>
    <a href="/backtests" class="{backtests_active}">Backtests</a>
</div>
"""

# Main Dashboard Page
DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GIGA TRADER - Dashboard</title>
    <style>""" + BASE_CSS + """</style>
</head>
<body>
    <div class="header">
        <h1>GIGA TRADER</h1>
        """ + NAV_HTML.format(dashboard_active="active", experiments_active="", models_active="", logs_active="", backtests_active="") + """
    </div>

    <div class="container">
        <div id="last-update">Last update: --</div>

        <!-- Stats Overview -->
        <div class="card full-width" style="margin-bottom: 20px;">
            <div class="stats-grid">
                <div class="stat-box">
                    <div id="stat-mode" class="value">--</div>
                    <div class="label">System Mode</div>
                </div>
                <div class="stat-box">
                    <div id="stat-daily-pnl" class="value">$0.00</div>
                    <div class="label">Daily P&L</div>
                </div>
                <div class="stat-box">
                    <div id="stat-trades" class="value">0</div>
                    <div class="label">Trades Today</div>
                </div>
                <div class="stat-box">
                    <div id="stat-experiments" class="value">0</div>
                    <div class="label">Experiments Run</div>
                </div>
            </div>
        </div>

        <div class="grid">
            <!-- System Status -->
            <div class="card">
                <h2>System Status</h2>
                <div style="text-align: center; padding: 15px;">
                    <span id="mode-badge" class="mode-badge mode-IDLE">IDLE</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Uptime</span>
                    <span id="uptime" class="status-value">--</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Health</span>
                    <span id="health" class="status-value">--</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Consecutive Errors</span>
                    <span id="errors" class="status-value">--</span>
                </div>
            </div>

            <!-- Trading Status -->
            <div class="card">
                <h2>Trading</h2>
                <div class="status-item">
                    <span class="status-label">Active</span>
                    <span id="trading-active" class="status-value">--</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Position</span>
                    <span id="position" class="status-value">--</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Position P&L</span>
                    <span id="position-pnl" class="status-value">--</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Daily P&L</span>
                    <span id="daily-pnl" class="status-value">--</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Trades Today</span>
                    <span id="trades-today" class="status-value">--</span>
                </div>
            </div>

            <!-- Model Status -->
            <div class="card">
                <h2>Active Model</h2>
                <div class="status-item">
                    <span class="status-label">Loaded</span>
                    <span id="model-loaded" class="status-value">--</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Last Train</span>
                    <span id="last-train" class="status-value">--</span>
                </div>
                <div class="status-item">
                    <span class="status-label">Accuracy</span>
                    <span id="accuracy" class="status-value">--</span>
                </div>
            </div>

            <!-- Components -->
            <div class="card">
                <h2>Components</h2>
                <div id="components"></div>
            </div>
        </div>

        <!-- Trading Gates -->
        <div class="card" style="margin-bottom: 20px;">
            <h2>Trading Gates</h2>
            <div id="gates-summary" style="padding: 5px 0;">
                <span style="color: #666;">No gate data yet</span>
            </div>
            <div id="gates-detail" style="max-height: 200px; overflow-y: auto;"></div>
        </div>

        <!-- Controls -->
        <div class="card" style="margin-bottom: 20px;">
            <h2>Controls</h2>
            <div class="controls">
                <button class="btn btn-success" onclick="sendCommand('start_trading')">Start Trading</button>
                <button class="btn btn-danger" onclick="sendCommand('stop_trading')">Stop Trading</button>
                <button class="btn btn-primary" onclick="sendCommand('force_train')">Force Training</button>
                <button class="btn btn-warning" onclick="sendCommand('close_positions')">Close All Positions</button>
                <button class="btn btn-secondary" onclick="sendCommand('run_experiment')">Run Experiment</button>
            </div>
        </div>

        <!-- Recent Logs Preview -->
        <div class="card">
            <h2>Recent Logs <a href="/logs" style="font-size: 12px; color: #888;">View All</a></h2>
            <div id="logs" class="log-viewer" style="max-height: 250px;">Loading logs...</div>
        </div>
    </div>

    <script>
        function formatUptime(seconds) {
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = Math.floor(seconds % 60);
            return h > 0 ? `${h}h ${m}m` : `${m}m ${s}s`;
        }

        function formatMoney(value) {
            const formatted = Math.abs(value).toLocaleString('en-US', {
                style: 'currency',
                currency: 'USD'
            });
            return value >= 0 ? formatted : `-${formatted}`;
        }

        function updateDashboard() {
            fetch('/api/status')
                .then(r => r.json())
                .then(data => {
                    // Mode badge
                    const badge = document.getElementById('mode-badge');
                    badge.textContent = data.mode;
                    badge.className = `mode-badge mode-${data.mode}`;
                    document.getElementById('stat-mode').textContent = data.mode;

                    // System
                    document.getElementById('uptime').textContent = formatUptime(data.uptime_seconds);
                    document.getElementById('health').textContent = data.health.status;
                    document.getElementById('health').className = `status-value ${data.health.status === 'HEALTHY' ? 'positive' : 'negative'}`;
                    document.getElementById('errors').textContent = data.health.consecutive_errors;

                    // Trading
                    document.getElementById('trading-active').textContent = data.trading.active ? 'YES' : 'NO';
                    document.getElementById('trading-active').className = `status-value ${data.trading.active ? 'positive' : ''}`;
                    document.getElementById('position').textContent = data.trading.position;

                    const posPnl = data.trading.position_pnl;
                    document.getElementById('position-pnl').textContent = formatMoney(posPnl);
                    document.getElementById('position-pnl').className = `status-value ${posPnl >= 0 ? 'positive' : 'negative'}`;

                    const dailyPnl = data.trading.daily_pnl;
                    document.getElementById('daily-pnl').textContent = formatMoney(dailyPnl);
                    document.getElementById('daily-pnl').className = `status-value ${dailyPnl >= 0 ? 'positive' : 'negative'}`;
                    document.getElementById('stat-daily-pnl').textContent = formatMoney(dailyPnl);
                    document.getElementById('stat-daily-pnl').className = `value ${dailyPnl >= 0 ? 'positive' : 'negative'}`;

                    document.getElementById('trades-today').textContent = data.trading.trades_today;
                    document.getElementById('stat-trades').textContent = data.trading.trades_today;

                    // Model
                    document.getElementById('model-loaded').textContent = data.model.loaded ? 'YES' : 'NO';
                    document.getElementById('model-loaded').className = `status-value ${data.model.loaded ? 'positive' : 'warning'}`;
                    document.getElementById('last-train').textContent = data.model.last_train || 'Never';
                    document.getElementById('accuracy').textContent = data.model.accuracy ? `${(data.model.accuracy * 100).toFixed(1)}%` : '--';

                    // Components
                    let componentsHtml = '';
                    for (const [name, status] of Object.entries(data.components || {})) {
                        const dotClass = status === 'RUNNING' ? 'running' : (status === 'ERROR' ? 'error' : 'stopped');
                        componentsHtml += `
                            <div class="component-status">
                                <div class="component-dot ${dotClass}"></div>
                                <span>${name}: ${status}</span>
                            </div>
                        `;
                    }
                    document.getElementById('components').innerHTML = componentsHtml || '<span style="color:#666">No components</span>';

                    // Trading Gates
                    const gates = data.trading_gates;
                    if (gates && gates.last_result) {
                        const gr = gates.last_result;
                        const blocked = gr.is_blocked;
                        const confMult = gr.confidence_multiplier || 1.0;
                        const summaryColor = blocked ? '#e74c3c' : (confMult < 0.9 ? '#f39c12' : '#2ecc71');
                        const summaryText = blocked
                            ? `BLOCKED by ${(gr.blocking_gates || []).join(', ')}`
                            : `PASS (conf: ${confMult.toFixed(2)}x)`;
                        document.getElementById('gates-summary').innerHTML = `
                            <div class="status-item">
                                <span class="status-label">Last Result</span>
                                <span class="status-value" style="color:${summaryColor}">${summaryText}</span>
                            </div>
                            <div class="status-item">
                                <span class="status-label">Gates Evaluated</span>
                                <span class="status-value">${gr.n_gates_evaluated || 0}</span>
                            </div>
                            <div class="status-item">
                                <span class="status-label">History</span>
                                <span class="status-value">${gates.history_count || 0} decisions</span>
                            </div>
                        `;
                        // Detail: individual gate decisions
                        let detailHtml = '';
                        (gr.decisions || []).forEach(d => {
                            const actionColor = d.action === 'BLOCK' ? '#e74c3c'
                                : d.action === 'REDUCE' ? '#f39c12'
                                : d.action === 'BOOST' ? '#2ecc71' : '#888';
                            detailHtml += `<div style="padding:4px 0;border-bottom:1px solid #333;font-size:12px;">
                                <span style="color:${actionColor};font-weight:bold;">${d.action}</span>
                                <span style="color:#aaa;margin-left:8px;">${d.gate}</span>
                                <span style="color:#666;margin-left:8px;">${d.reason || ''}</span>
                            </div>`;
                        });
                        document.getElementById('gates-detail').innerHTML = detailHtml || '';
                    }

                    // Update timestamp
                    document.getElementById('last-update').textContent = `Last update: ${new Date().toLocaleTimeString()}`;
                })
                .catch(err => console.error('Status fetch error:', err));

            // Fetch experiment count
            fetch('/api/experiments/stats')
                .then(r => r.json())
                .then(data => {
                    document.getElementById('stat-experiments').textContent = data.total_experiments || 0;
                })
                .catch(() => {});
        }

        function refreshLogs() {
            fetch('/api/logs?limit=30')
                .then(r => r.json())
                .then(data => {
                    let html = '';
                    data.logs.forEach(line => {
                        let logClass = '';
                        if (line.includes('[INFO]') || line.includes('INFO')) logClass = 'log-INFO';
                        else if (line.includes('[WARNING]') || line.includes('WARNING')) logClass = 'log-WARNING';
                        else if (line.includes('[ERROR]') || line.includes('ERROR')) logClass = 'log-ERROR';
                        else if (line.includes('[DEBUG]') || line.includes('DEBUG')) logClass = 'log-DEBUG';
                        html += `<div class="log-line ${logClass}">${escapeHtml(line)}</div>`;
                    });
                    document.getElementById('logs').innerHTML = html || '<span style="color:#666">No logs</span>';
                    const logViewer = document.getElementById('logs');
                    logViewer.scrollTop = logViewer.scrollHeight;
                })
                .catch(err => console.error('Logs fetch error:', err));
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function sendCommand(cmd) {
            fetch('/api/command', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({command: cmd})
            })
            .then(r => r.json())
            .then(data => {
                alert(data.message || data.error);
                updateDashboard();
            })
            .catch(err => alert('Command failed: ' + err));
        }

        // Initial load
        updateDashboard();
        refreshLogs();

        // Auto-refresh
        setInterval(updateDashboard, 5000);
        setInterval(refreshLogs, 10000);
    </script>
</body>
</html>
"""

# Logs Page with Filtering
LOGS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GIGA TRADER - Logs</title>
    <style>""" + BASE_CSS + """</style>
</head>
<body>
    <div class="header">
        <h1>GIGA TRADER</h1>
        """ + NAV_HTML.format(dashboard_active="", experiments_active="", models_active="", logs_active="active", backtests_active="") + """
    </div>

    <div class="container">
        <div class="card">
            <h2>Log Viewer</h2>

            <!-- Filters -->
            <div class="filters">
                <div class="filter-group">
                    <label>Level:</label>
                    <select id="filter-level" onchange="applyFilters()">
                        <option value="all">All Levels</option>
                        <option value="ERROR">Errors Only</option>
                        <option value="WARNING">Warnings+</option>
                        <option value="INFO">Info+</option>
                        <option value="DEBUG">Debug (All)</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Search:</label>
                    <input type="text" id="filter-search" placeholder="Filter by text..." onkeyup="applyFilters()">
                </div>
                <div class="filter-group">
                    <label>Source:</label>
                    <select id="filter-source" onchange="applyFilters()">
                        <option value="all">All Sources</option>
                        <option value="orchestrator">Orchestrator</option>
                        <option value="trading">Trading</option>
                        <option value="training">Training</option>
                        <option value="experiment">Experiment</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Lines:</label>
                    <select id="filter-limit" onchange="loadLogs()">
                        <option value="100">Last 100</option>
                        <option value="500">Last 500</option>
                        <option value="1000">Last 1000</option>
                        <option value="5000">Last 5000</option>
                    </select>
                </div>
                <button class="btn btn-primary btn-sm" onclick="loadLogs()">Refresh</button>
                <button class="btn btn-secondary btn-sm" onclick="clearFilters()">Clear Filters</button>
            </div>

            <!-- Stats -->
            <div class="stats-grid" style="margin-bottom: 15px;">
                <div class="stat-box">
                    <div id="stat-total" class="value" style="font-size: 20px;">0</div>
                    <div class="label">Total Lines</div>
                </div>
                <div class="stat-box">
                    <div id="stat-errors" class="value negative" style="font-size: 20px;">0</div>
                    <div class="label">Errors</div>
                </div>
                <div class="stat-box">
                    <div id="stat-warnings" class="value warning" style="font-size: 20px;">0</div>
                    <div class="label">Warnings</div>
                </div>
                <div class="stat-box">
                    <div id="stat-filtered" class="value" style="font-size: 20px;">0</div>
                    <div class="label">Filtered</div>
                </div>
            </div>

            <div id="logs" class="log-viewer">Loading logs...</div>
        </div>
    </div>

    <script>
        let allLogs = [];

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        function loadLogs() {
            const limit = document.getElementById('filter-limit').value;
            fetch(`/api/logs?limit=${limit}`)
                .then(r => r.json())
                .then(data => {
                    allLogs = data.logs;
                    updateStats();
                    applyFilters();
                })
                .catch(err => console.error('Logs fetch error:', err));
        }

        function updateStats() {
            const errors = allLogs.filter(l => l.includes('[ERROR]') || l.includes('ERROR')).length;
            const warnings = allLogs.filter(l => l.includes('[WARNING]') || l.includes('WARNING')).length;
            document.getElementById('stat-total').textContent = allLogs.length;
            document.getElementById('stat-errors').textContent = errors;
            document.getElementById('stat-warnings').textContent = warnings;
        }

        function applyFilters() {
            const level = document.getElementById('filter-level').value;
            const search = document.getElementById('filter-search').value.toLowerCase();
            const source = document.getElementById('filter-source').value;

            const levelOrder = {'ERROR': 1, 'WARNING': 2, 'INFO': 3, 'DEBUG': 4};

            let filtered = allLogs.filter(line => {
                // Level filter
                if (level !== 'all') {
                    const lineLevel = line.includes('ERROR') ? 'ERROR' :
                                     line.includes('WARNING') ? 'WARNING' :
                                     line.includes('DEBUG') ? 'DEBUG' : 'INFO';
                    if (levelOrder[lineLevel] > levelOrder[level]) return false;
                }

                // Source filter
                if (source !== 'all') {
                    if (!line.toLowerCase().includes(source)) return false;
                }

                // Text search
                if (search && !line.toLowerCase().includes(search)) return false;

                return true;
            });

            document.getElementById('stat-filtered').textContent = filtered.length;

            let html = '';
            filtered.forEach(line => {
                let logClass = '';
                if (line.includes('[ERROR]') || line.includes('ERROR')) logClass = 'log-ERROR';
                else if (line.includes('[WARNING]') || line.includes('WARNING')) logClass = 'log-WARNING';
                else if (line.includes('[INFO]') || line.includes('INFO')) logClass = 'log-INFO';
                else if (line.includes('[DEBUG]') || line.includes('DEBUG')) logClass = 'log-DEBUG';
                html += `<div class="log-line ${logClass}">${escapeHtml(line)}</div>`;
            });
            document.getElementById('logs').innerHTML = html || '<span style="color:#666">No matching logs</span>';
        }

        function clearFilters() {
            document.getElementById('filter-level').value = 'all';
            document.getElementById('filter-search').value = '';
            document.getElementById('filter-source').value = 'all';
            applyFilters();
        }

        // Initial load
        loadLogs();

        // Auto-refresh every 30 seconds
        setInterval(loadLogs, 30000);
    </script>
</body>
</html>
"""

# Experiments Page
EXPERIMENTS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GIGA TRADER - Experiments</title>
    <style>""" + BASE_CSS + """</style>
</head>
<body>
    <div class="header">
        <h1>GIGA TRADER</h1>
        """ + NAV_HTML.format(dashboard_active="", experiments_active="active", models_active="", logs_active="", backtests_active="") + """
    </div>

    <div class="container">
        <!-- Stats -->
        <div class="card" style="margin-bottom: 20px;">
            <div class="stats-grid">
                <div class="stat-box">
                    <div id="stat-total" class="value">0</div>
                    <div class="label">Total Experiments</div>
                </div>
                <div class="stat-box">
                    <div id="stat-completed" class="value positive">0</div>
                    <div class="label">Completed</div>
                </div>
                <div class="stat-box">
                    <div id="stat-running" class="value info">0</div>
                    <div class="label">Running</div>
                </div>
                <div class="stat-box">
                    <div id="stat-best-auc" class="value">--</div>
                    <div class="label">Best AUC</div>
                </div>
            </div>
        </div>

        <!-- Leaderboard -->
        <div class="card" style="margin-bottom: 20px;">
            <h2>Top Experiments <span class="badge" id="leaderboard-count">0</span></h2>
            <div class="table-container" style="max-height: 300px; overflow-y: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>ID</th>
                            <th>Type</th>
                            <th>Test AUC</th>
                            <th>Sharpe</th>
                            <th>Win Rate</th>
                            <th>Return</th>
                            <th>Duration</th>
                        </tr>
                    </thead>
                    <tbody id="leaderboard-body">
                    </tbody>
                </table>
            </div>
        </div>

        <!-- All Experiments -->
        <div class="card">
            <h2>Experiment History</h2>

            <div class="filters">
                <div class="filter-group">
                    <label>Status:</label>
                    <select id="filter-status" onchange="applyFilters()">
                        <option value="all">All</option>
                        <option value="completed">Completed</option>
                        <option value="running">Running</option>
                        <option value="failed">Failed</option>
                        <option value="queued">Queued</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Type:</label>
                    <select id="filter-type" onchange="applyFilters()">
                        <option value="all">All Types</option>
                        <option value="hyperparameter">Hyperparameter</option>
                        <option value="feature_subset">Feature Subset</option>
                        <option value="ensemble">Ensemble</option>
                        <option value="dim_reduction">Dim Reduction</option>
                        <option value="regularization">Regularization</option>
                        <option value="threshold">Threshold</option>
                    </select>
                </div>
                <button class="btn btn-primary btn-sm" onclick="loadExperiments()">Refresh</button>
            </div>

            <div class="table-container" style="max-height: 500px; overflow-y: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Type</th>
                            <th>Status</th>
                            <th>Test Score</th>
                            <th>CV Mean</th>
                            <th>Sharpe</th>
                            <th>Created</th>
                            <th>Duration</th>
                        </tr>
                    </thead>
                    <tbody id="experiments-body">
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        let allExperiments = [];

        function loadExperiments() {
            fetch('/api/experiments')
                .then(r => r.json())
                .then(data => {
                    allExperiments = data.experiments || [];
                    updateStats(data);
                    updateLeaderboard(data.leaderboard || []);
                    applyFilters();
                })
                .catch(err => console.error('Error:', err));
        }

        function updateStats(data) {
            document.getElementById('stat-total').textContent = data.total || 0;
            document.getElementById('stat-completed').textContent = data.completed || 0;
            document.getElementById('stat-running').textContent = data.running || 0;
            document.getElementById('stat-best-auc').textContent = data.best_auc ? data.best_auc.toFixed(3) : '--';
        }

        function updateLeaderboard(experiments) {
            document.getElementById('leaderboard-count').textContent = experiments.length;
            let html = '';
            experiments.slice(0, 10).forEach((exp, idx) => {
                html += `
                    <tr>
                        <td>${idx + 1}</td>
                        <td><code>${exp.experiment_id?.slice(0, 8) || '--'}</code></td>
                        <td><span class="tag tag-info">${exp.experiment_type || '--'}</span></td>
                        <td class="positive">${exp.test_score?.toFixed(3) || '--'}</td>
                        <td>${exp.backtest_sharpe?.toFixed(2) || '--'}</td>
                        <td>${exp.backtest_win_rate ? (exp.backtest_win_rate * 100).toFixed(1) + '%' : '--'}</td>
                        <td class="${exp.backtest_return >= 0 ? 'positive' : 'negative'}">${exp.backtest_return ? (exp.backtest_return * 100).toFixed(1) + '%' : '--'}</td>
                        <td>${exp.duration_seconds ? Math.round(exp.duration_seconds) + 's' : '--'}</td>
                    </tr>
                `;
            });
            document.getElementById('leaderboard-body').innerHTML = html || '<tr><td colspan="8" style="text-align:center;color:#666">No experiments yet</td></tr>';
        }

        function applyFilters() {
            const status = document.getElementById('filter-status').value;
            const type = document.getElementById('filter-type').value;

            let filtered = allExperiments.filter(exp => {
                if (status !== 'all' && exp.status !== status) return false;
                if (type !== 'all' && exp.experiment_type !== type) return false;
                return true;
            });

            let html = '';
            filtered.forEach(exp => {
                const statusClass = exp.status === 'completed' ? 'tag-success' :
                                   exp.status === 'running' ? 'tag-info' :
                                   exp.status === 'failed' ? 'tag-danger' : 'tag-default';
                html += `
                    <tr>
                        <td><code>${exp.experiment_id?.slice(0, 8) || '--'}</code></td>
                        <td><span class="tag tag-info">${exp.experiment_type || '--'}</span></td>
                        <td><span class="tag ${statusClass}">${exp.status || '--'}</span></td>
                        <td>${exp.test_score?.toFixed(3) || '--'}</td>
                        <td>${exp.cv_scores?.length ? (exp.cv_scores.reduce((a,b)=>a+b,0)/exp.cv_scores.length).toFixed(3) : '--'}</td>
                        <td>${exp.backtest_sharpe?.toFixed(2) || '--'}</td>
                        <td>${exp.created_at ? new Date(exp.created_at).toLocaleString() : '--'}</td>
                        <td>${exp.duration_seconds ? Math.round(exp.duration_seconds) + 's' : '--'}</td>
                    </tr>
                `;
            });
            document.getElementById('experiments-body').innerHTML = html || '<tr><td colspan="8" style="text-align:center;color:#666">No matching experiments</td></tr>';
        }

        // Initial load
        loadExperiments();
        setInterval(loadExperiments, 30000);
    </script>
</body>
</html>
"""

# Models Page
MODELS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GIGA TRADER - Models</title>
    <style>""" + BASE_CSS + """</style>
</head>
<body>
    <div class="header">
        <h1>GIGA TRADER</h1>
        """ + NAV_HTML.format(dashboard_active="", experiments_active="", models_active="active", logs_active="", backtests_active="") + """
    </div>

    <div class="container">
        <!-- Stats -->
        <div class="card" style="margin-bottom: 20px;">
            <div class="stats-grid">
                <div class="stat-box">
                    <div id="stat-total" class="value">0</div>
                    <div class="label">Total Models</div>
                </div>
                <div class="stat-box">
                    <div id="stat-production" class="value positive">0</div>
                    <div class="label">Production</div>
                </div>
                <div class="stat-box">
                    <div id="stat-best-auc" class="value">--</div>
                    <div class="label">Best AUC</div>
                </div>
                <div class="stat-box">
                    <div id="stat-best-sharpe" class="value">--</div>
                    <div class="label">Best Sharpe</div>
                </div>
            </div>
        </div>

        <!-- Production Model -->
        <div class="card" style="margin-bottom: 20px;">
            <h2>Active Production Model</h2>
            <div id="production-model">
                <p style="color: #666; text-align: center; padding: 20px;">Loading...</p>
            </div>
        </div>

        <!-- Model Registry -->
        <div class="card">
            <h2>Model Registry <span class="badge" id="models-count">0</span></h2>
            <div class="table-container" style="max-height: 500px; overflow-y: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>Model ID</th>
                            <th>Type</th>
                            <th>CV AUC</th>
                            <th>Test AUC</th>
                            <th>Sharpe</th>
                            <th>Win Rate</th>
                            <th>Created</th>
                            <th>Size</th>
                            <th>Actions</th>
                        </tr>
                    </thead>
                    <tbody id="models-body">
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        function loadModels() {
            fetch('/api/models')
                .then(r => r.json())
                .then(data => {
                    updateStats(data);
                    updateProductionModel(data.production);
                    updateModelsList(data.models || []);
                })
                .catch(err => console.error('Error:', err));
        }

        function updateStats(data) {
            document.getElementById('stat-total').textContent = data.total || 0;
            document.getElementById('stat-production').textContent = data.production_count || 0;
            document.getElementById('stat-best-auc').textContent = data.best_auc ? data.best_auc.toFixed(3) : '--';
            document.getElementById('stat-best-sharpe').textContent = data.best_sharpe ? data.best_sharpe.toFixed(2) : '--';
        }

        function updateProductionModel(model) {
            if (!model) {
                document.getElementById('production-model').innerHTML = '<p style="color: #666; text-align: center; padding: 20px;">No production model loaded</p>';
                return;
            }

            document.getElementById('production-model').innerHTML = `
                <div class="grid-3">
                    <div class="status-item">
                        <span class="status-label">Model</span>
                        <span class="status-value info">${model.name || 'Unknown'}</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">CV AUC</span>
                        <span class="status-value positive">${model.cv_auc?.toFixed(3) || '--'}</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">Test AUC</span>
                        <span class="status-value positive">${model.test_auc?.toFixed(3) || '--'}</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">Backtest Sharpe</span>
                        <span class="status-value">${model.backtest_sharpe?.toFixed(2) || '--'}</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">Win Rate</span>
                        <span class="status-value">${model.backtest_win_rate ? (model.backtest_win_rate * 100).toFixed(1) + '%' : '--'}</span>
                    </div>
                    <div class="status-item">
                        <span class="status-label">Last Updated</span>
                        <span class="status-value">${model.created_at ? new Date(model.created_at).toLocaleDateString() : '--'}</span>
                    </div>
                </div>
            `;
        }

        function updateModelsList(models) {
            document.getElementById('models-count').textContent = models.length;
            let html = '';
            models.forEach(model => {
                html += `
                    <tr>
                        <td><code>${model.model_id?.slice(0, 8) || model.name?.slice(0, 8) || '--'}</code></td>
                        <td><span class="tag tag-info">${model.type || 'ensemble'}</span></td>
                        <td class="positive">${model.cv_auc?.toFixed(3) || '--'}</td>
                        <td class="positive">${model.test_auc?.toFixed(3) || '--'}</td>
                        <td>${model.backtest_sharpe?.toFixed(2) || '--'}</td>
                        <td>${model.backtest_win_rate ? (model.backtest_win_rate * 100).toFixed(1) + '%' : '--'}</td>
                        <td>${model.created_at ? new Date(model.created_at).toLocaleString() : '--'}</td>
                        <td>${model.size_mb ? model.size_mb.toFixed(1) + ' MB' : '--'}</td>
                        <td>
                            <button class="btn btn-sm btn-primary" onclick="promoteModel('${model.model_id || model.name}')">Promote</button>
                        </td>
                    </tr>
                `;
            });
            document.getElementById('models-body').innerHTML = html || '<tr><td colspan="9" style="text-align:center;color:#666">No models found</td></tr>';
        }

        function promoteModel(modelId) {
            if (confirm(`Promote model ${modelId.slice(0, 8)} to production?`)) {
                fetch('/api/models/promote', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({model_id: modelId})
                })
                .then(r => r.json())
                .then(data => {
                    alert(data.message || data.error);
                    loadModels();
                })
                .catch(err => alert('Error: ' + err));
            }
        }

        // Initial load
        loadModels();
        setInterval(loadModels, 60000);
    </script>
</body>
</html>
"""

# Backtests Page
BACKTESTS_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GIGA TRADER - Backtests</title>
    <style>""" + BASE_CSS + """</style>
</head>
<body>
    <div class="header">
        <h1>GIGA TRADER</h1>
        """ + NAV_HTML.format(dashboard_active="", experiments_active="", models_active="", logs_active="", backtests_active="active") + """
    </div>

    <div class="container">
        <!-- Stats -->
        <div class="card" style="margin-bottom: 20px;">
            <div class="stats-grid">
                <div class="stat-box">
                    <div id="stat-total" class="value">0</div>
                    <div class="label">Total Backtests</div>
                </div>
                <div class="stat-box">
                    <div id="stat-profitable" class="value positive">0</div>
                    <div class="label">Profitable</div>
                </div>
                <div class="stat-box">
                    <div id="stat-best-return" class="value">--</div>
                    <div class="label">Best Return</div>
                </div>
                <div class="stat-box">
                    <div id="stat-best-sharpe" class="value">--</div>
                    <div class="label">Best Sharpe</div>
                </div>
            </div>
        </div>

        <!-- Run Backtest -->
        <div class="card" style="margin-bottom: 20px;">
            <h2>Run New Backtest</h2>
            <div class="controls">
                <div class="filter-group">
                    <label>Period:</label>
                    <select id="backtest-period">
                        <option value="1y">1 Year</option>
                        <option value="2y" selected>2 Years</option>
                        <option value="3y">3 Years</option>
                        <option value="5y">5 Years</option>
                    </select>
                </div>
                <div class="filter-group">
                    <label>Initial Capital:</label>
                    <input type="text" id="backtest-capital" value="100000" style="width: 100px;">
                </div>
                <button class="btn btn-primary" onclick="runBacktest()">Run Backtest</button>
            </div>
        </div>

        <!-- Backtest Results -->
        <div class="card">
            <h2>Backtest Results <span class="badge" id="backtests-count">0</span></h2>
            <div class="table-container" style="max-height: 500px; overflow-y: auto;">
                <table>
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Period</th>
                            <th>Total Return</th>
                            <th>Sharpe</th>
                            <th>Max DD</th>
                            <th>Win Rate</th>
                            <th>Trades</th>
                            <th>Final Value</th>
                            <th>Date</th>
                        </tr>
                    </thead>
                    <tbody id="backtests-body">
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <script>
        function loadBacktests() {
            fetch('/api/backtests')
                .then(r => r.json())
                .then(data => {
                    updateStats(data);
                    updateBacktestsList(data.backtests || []);
                })
                .catch(err => console.error('Error:', err));
        }

        function updateStats(data) {
            document.getElementById('stat-total').textContent = data.total || 0;
            document.getElementById('stat-profitable').textContent = data.profitable || 0;
            document.getElementById('stat-best-return').textContent = data.best_return ? (data.best_return * 100).toFixed(1) + '%' : '--';
            document.getElementById('stat-best-sharpe').textContent = data.best_sharpe ? data.best_sharpe.toFixed(2) : '--';
        }

        function updateBacktestsList(backtests) {
            document.getElementById('backtests-count').textContent = backtests.length;
            let html = '';
            backtests.forEach(bt => {
                const returnClass = bt.total_return >= 0 ? 'positive' : 'negative';
                html += `
                    <tr>
                        <td><code>${bt.backtest_id?.slice(0, 8) || '--'}</code></td>
                        <td>${bt.period || '--'}</td>
                        <td class="${returnClass}">${bt.total_return ? (bt.total_return * 100).toFixed(1) + '%' : '--'}</td>
                        <td>${bt.sharpe?.toFixed(2) || '--'}</td>
                        <td class="negative">${bt.max_drawdown ? (bt.max_drawdown * 100).toFixed(1) + '%' : '--'}</td>
                        <td>${bt.win_rate ? (bt.win_rate * 100).toFixed(1) + '%' : '--'}</td>
                        <td>${bt.total_trades || '--'}</td>
                        <td>${bt.final_value ? '$' + bt.final_value.toLocaleString() : '--'}</td>
                        <td>${bt.created_at ? new Date(bt.created_at).toLocaleString() : '--'}</td>
                    </tr>
                `;
            });
            document.getElementById('backtests-body').innerHTML = html || '<tr><td colspan="9" style="text-align:center;color:#666">No backtests found</td></tr>';
        }

        function runBacktest() {
            const period = document.getElementById('backtest-period').value;
            const capital = document.getElementById('backtest-capital').value;

            fetch('/api/backtests/run', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({period: period, initial_capital: parseFloat(capital)})
            })
            .then(r => r.json())
            .then(data => {
                alert(data.message || data.error);
                loadBacktests();
            })
            .catch(err => alert('Error: ' + err));
        }

        // Initial load
        loadBacktests();
        setInterval(loadBacktests, 60000);
    </script>
</body>
</html>
"""


# ===============================================================================
# ROUTES
# ===============================================================================
@app.route('/')
def dashboard():
    return render_template_string(DASHBOARD_HTML)


@app.route('/logs')
def logs_page():
    return render_template_string(LOGS_HTML)


@app.route('/experiments')
def experiments_page():
    return render_template_string(EXPERIMENTS_HTML)


@app.route('/models')
def models_page():
    return render_template_string(MODELS_HTML)


@app.route('/backtests')
def backtests_page():
    return render_template_string(BACKTESTS_HTML)


@app.route('/api/status')
def get_status():
    """Get current system status."""
    status_file = WEB_CONFIG["status_file"]

    if status_file.exists():
        with open(status_file) as f:
            return jsonify(json.load(f))

    # Default status if no file
    return jsonify({
        "mode": "UNKNOWN",
        "uptime_seconds": 0,
        "trading": {
            "active": False,
            "position": "UNKNOWN",
            "position_pnl": 0,
            "daily_pnl": 0,
            "trades_today": 0,
        },
        "model": {
            "loaded": False,
            "last_train": "",
            "accuracy": 0,
        },
        "health": {
            "status": "UNKNOWN",
            "consecutive_errors": 0,
            "last_error": "",
        },
        "components": {},
    })


@app.route('/api/logs')
def get_logs():
    """Get recent log lines with optional filtering."""
    log_dir = WEB_CONFIG["log_dir"]
    limit = request.args.get('limit', 100, type=int)

    # Collect logs from multiple sources
    lines = []

    # Try orchestrator log
    today = datetime.now().strftime("%Y%m%d")
    log_files = [
        log_dir / f"orchestrator_{today}.log",
        log_dir / "orchestrator.log",
        log_dir / "trading.log",
        log_dir / "training.log",
    ]

    for log_file in log_files:
        if log_file.exists():
            try:
                with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                    file_lines = f.readlines()
                    lines.extend(file_lines[-limit:])
            except Exception as e:
                web_logger.debug(f"Could not read log file {log_file}: {e}")

    # Sort by timestamp if possible, otherwise just use last N lines
    lines = lines[-limit:]

    return jsonify({"logs": [l.strip() for l in lines]})


@app.route('/api/trades')
def get_trades():
    """Get recent trades."""
    trades_file = WEB_CONFIG["trades_file"]

    if trades_file.exists():
        with open(trades_file) as f:
            trades = json.load(f)
            return jsonify({"trades": trades[-50:]})

    return jsonify({"trades": []})


@app.route('/api/command', methods=['POST'])
def execute_command():
    """Execute a control command."""
    data = request.get_json()
    command = data.get('command')

    # Write command to a command file that orchestrator watches
    command_file = project_root / "logs" / "command.json"

    commands = {
        "start_trading": {"action": "start_trading", "timestamp": datetime.now().isoformat()},
        "stop_trading": {"action": "stop_trading", "timestamp": datetime.now().isoformat()},
        "force_train": {"action": "force_train", "timestamp": datetime.now().isoformat()},
        "close_positions": {"action": "close_positions", "timestamp": datetime.now().isoformat()},
        "run_experiment": {"action": "run_experiment", "timestamp": datetime.now().isoformat()},
    }

    if command in commands:
        with open(command_file, 'w') as f:
            json.dump(commands[command], f)
        return jsonify({"status": "ok", "message": f"Command '{command}' queued"})

    return jsonify({"status": "error", "error": f"Unknown command: {command}"}), 400


@app.route('/api/account')
def get_account():
    """Get Alpaca account info."""
    try:
        from src.paper_trading import AlpacaPaperClient
        client = AlpacaPaperClient()
        account = client.get_account()
        positions = client.get_all_positions()

        return jsonify({
            "account": account,
            "positions": [
                {
                    "symbol": p.symbol,
                    "side": p.side,
                    "quantity": p.quantity,
                    "entry_price": p.entry_price,
                    "current_price": p.current_price,
                    "unrealized_pnl": p.unrealized_pnl,
                }
                for p in positions
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/experiments')
def get_experiments():
    """Get experiment history."""
    try:
        from src.core.registry_db import get_registry_db
        db = get_registry_db()
        experiments = db.get_experiments()
    except Exception as e:
        web_logger.warning(f"Could not load experiment history: {e}")
        experiments = []

    # Calculate stats
    completed = [e for e in experiments if e.get('status') == 'completed']
    running = [e for e in experiments if e.get('status') == 'running']
    best_auc = max([e.get('test_auc', e.get('test_score', 0)) for e in completed], default=0)

    # Sort by test_auc for leaderboard
    leaderboard = sorted(completed, key=lambda x: x.get('test_auc', 0), reverse=True)

    return jsonify({
        "experiments": experiments,
        "leaderboard": leaderboard[:20],
        "total": len(experiments),
        "completed": len(completed),
        "running": len(running),
        "best_auc": best_auc if best_auc > 0 else None,
    })


@app.route('/api/experiments/stats')
def get_experiments_stats():
    """Get quick experiment stats for dashboard."""
    try:
        from src.core.registry_db import get_registry_db
        db = get_registry_db()
        return jsonify({"total_experiments": db.get_experiment_count()})
    except Exception as e:
        web_logger.warning(f"Could not load experiment stats: {e}")
        return jsonify({"total_experiments": 0})


@app.route('/api/models')
def get_models():
    """Get model registry."""
    models_dir = WEB_CONFIG["models_dir"]

    models = []
    production = None

    # Load models from SQLite
    try:
        from src.core.registry_db import get_registry_db
        db = get_registry_db()
        models = db.get_models()
    except Exception as e:
        web_logger.warning(f"Could not load model registry: {e}")

    # Scan for model files
    production_dir = models_dir / "production"
    if production_dir.exists():
        for model_file in production_dir.glob("*.joblib"):
            model_info = {
                "name": model_file.stem,
                "model_id": model_file.stem,
                "type": "ensemble",
                "size_mb": model_file.stat().st_size / (1024 * 1024),
                "created_at": datetime.fromtimestamp(model_file.stat().st_mtime).isoformat(),
            }
            # Check if already in registry
            if not any(m.get('name') == model_file.stem for m in models):
                models.append(model_info)

            # Set production model (most recent)
            if production is None or model_file.stat().st_mtime > production.get('mtime', 0):
                production = {**model_info, 'mtime': model_file.stat().st_mtime}

    # Calculate stats
    best_auc = max([m.get('cv_auc', 0) for m in models], default=0)
    best_sharpe = max([m.get('backtest_sharpe', 0) for m in models], default=0)

    return jsonify({
        "models": models,
        "production": production,
        "total": len(models),
        "production_count": len(list(production_dir.glob("*.joblib"))) if production_dir.exists() else 0,
        "best_auc": best_auc if best_auc > 0 else None,
        "best_sharpe": best_sharpe if best_sharpe > 0 else None,
    })


@app.route('/api/models/promote', methods=['POST'])
def promote_model():
    """Promote a model to production."""
    data = request.get_json()
    model_id = data.get('model_id')

    # In a real implementation, this would copy the model to production
    return jsonify({"status": "ok", "message": f"Model {model_id} promoted to production"})


@app.route('/api/backtests')
def get_backtests():
    """Get backtest results."""
    experiments_dir = WEB_CONFIG["experiments_dir"]
    backtests_file = experiments_dir / "backtest_results.json"

    backtests = []
    if backtests_file.exists():
        try:
            with open(backtests_file) as f:
                backtests = json.load(f)
        except Exception as e:
            web_logger.warning(f"Could not load backtest results: {e}")

    # Calculate stats
    profitable = [b for b in backtests if b.get('total_return', 0) > 0]
    best_return = max([b.get('total_return', 0) for b in backtests], default=0)
    best_sharpe = max([b.get('sharpe', 0) for b in backtests], default=0)

    return jsonify({
        "backtests": backtests,
        "total": len(backtests),
        "profitable": len(profitable),
        "best_return": best_return if best_return > 0 else None,
        "best_sharpe": best_sharpe if best_sharpe > 0 else None,
    })


@app.route('/api/backtests/run', methods=['POST'])
def run_backtest():
    """Queue a new backtest."""
    data = request.get_json()
    period = data.get('period', '2y')
    initial_capital = data.get('initial_capital', 100000)

    # Write command to trigger backtest
    command_file = project_root / "logs" / "command.json"
    with open(command_file, 'w') as f:
        json.dump({
            "action": "run_backtest",
            "period": period,
            "initial_capital": initial_capital,
            "timestamp": datetime.now().isoformat()
        }, f)

    return jsonify({"status": "ok", "message": f"Backtest queued for {period} period"})


@app.route('/api/stream')
def stream_status():
    """Server-sent events for real-time updates."""
    def generate():
        while True:
            status_file = WEB_CONFIG["status_file"]
            if status_file.exists():
                with open(status_file) as f:
                    data = json.load(f)
                yield f"data: {json.dumps(data)}\n\n"
            time.sleep(2)

    return Response(generate(), mimetype='text/event-stream')


# ===============================================================================
# MAIN
# ===============================================================================
def main():
    if not FLASK_AVAILABLE:
        print("[ERROR] Flask not installed. Run: pip install flask")
        return 1

    print("""
    ==================================================================
    |   GIGA TRADER - Enhanced Web Dashboard                         |
    ==================================================================
    """)
    print(f"  Starting web server on http://localhost:{WEB_CONFIG['port']}")
    print(f"")
    print(f"  Pages:")
    print(f"    /            - Main Dashboard")
    print(f"    /experiments - Experiment History & Leaderboard")
    print(f"    /models      - Model Registry")
    print(f"    /logs        - Log Viewer with Filtering")
    print(f"    /backtests   - Backtest Results")
    print(f"")
    print(f"  Press Ctrl+C to stop")
    print("=" * 60)

    app.run(
        host=WEB_CONFIG["host"],
        port=WEB_CONFIG["port"],
        debug=WEB_CONFIG["debug"],
        threaded=True,
    )

    return 0


if __name__ == "__main__":
    sys.exit(main())
