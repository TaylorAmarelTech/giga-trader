"""
GIGA TRADER - Interactive Dashboard
====================================
View models, backtest performance, and entry/exit analysis.

Usage:
    python src/dashboard.py              # Generate HTML dashboard
    python src/dashboard.py --serve      # Start live server on http://localhost:8050
    python src/dashboard.py --console    # Console-only output
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# DATA CLASSES
# =============================================================================
@dataclass
class ModelInfo:
    """Information about a trained model."""
    name: str
    path: Path
    file_size_mb: float
    modified_date: datetime
    model_type: str
    features_count: int
    has_scaler: bool
    has_dim_reduction: bool
    swing_model_type: str = "Unknown"
    timing_model_type: str = "Unknown"
    test_auc: float = 0.0
    train_auc: float = 0.0


@dataclass
class BacktestSummary:
    """Summary of a backtest run."""
    run_id: str
    run_date: datetime
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    total_trades: int
    robustness_score: float
    regime_results: Dict[str, Dict]


# =============================================================================
# MODEL LOADER
# =============================================================================
class ModelAnalyzer:
    """Analyze trained models."""

    def __init__(self, model_dir: Path = None):
        self.model_dir = model_dir or (project_root / "models" / "production")
        self.models: List[ModelInfo] = []

    def scan_models(self) -> List[ModelInfo]:
        """Scan and analyze all models in the production directory."""
        self.models = []

        for model_file in self.model_dir.glob("*.joblib"):
            try:
                info = self._analyze_model(model_file)
                self.models.append(info)
            except Exception as e:
                print(f"Error analyzing {model_file.name}: {e}")

        # Sort by modification date (newest first)
        self.models.sort(key=lambda m: m.modified_date, reverse=True)
        return self.models

    def _analyze_model(self, path: Path) -> ModelInfo:
        """Analyze a single model file."""
        stat = path.stat()

        # Load model state
        state = joblib.load(path)

        # Determine model types
        swing_type = "Unknown"
        timing_type = "Unknown"
        features_count = 0
        has_scaler = False
        has_dim_reduction = False
        test_auc = 0.0
        train_auc = 0.0

        if isinstance(state, dict):
            # Check for swing model - multiple possible locations
            swing_model = (
                state.get("swing_model") or
                state.get("swing_l2") or
                state.get("models", {}).get("swing_l2") or
                state.get("models", {}).get("swing_model") or
                state.get("models", {}).get("swing", {}).get("l2")  # Nested structure
            )
            if swing_model is not None:
                swing_type = type(swing_model).__name__

            # Also check for GB model
            swing_gb = state.get("models", {}).get("swing", {}).get("gb")
            if swing_gb is not None:
                swing_type = f"{swing_type}+{type(swing_gb).__name__}"

            # Check for timing model - multiple possible locations
            timing_model = (
                state.get("timing_model") or
                state.get("timing_l2") or
                state.get("models", {}).get("timing_l2") or
                state.get("models", {}).get("timing_model") or
                state.get("models", {}).get("timing", {}).get("l2")  # Nested structure
            )
            if timing_model is not None:
                timing_type = type(timing_model).__name__

            # Also check for GB model
            timing_gb = state.get("models", {}).get("timing", {}).get("gb")
            if timing_gb is not None:
                timing_type = f"{timing_type}+{type(timing_gb).__name__}"

            # Feature count - multiple possible keys
            feature_cols = (
                state.get("feature_columns") or
                state.get("feature_cols") or
                []
            )
            features_count = len(feature_cols)

            # Scaler
            has_scaler = "scaler" in state and state["scaler"] is not None

            # Dimensionality reduction - multiple possible keys
            has_dim_reduction = (
                ("dim_state" in state and state["dim_state"] is not None) or
                ("dim_reduction_state" in state and state["dim_reduction_state"] is not None)
            )

            # AUC scores - check results dict
            results = state.get("results", {})
            if isinstance(results, dict):
                test_auc = results.get("test_auc", results.get("swing_test_auc", 0.0))
                train_auc = results.get("train_auc", results.get("swing_train_auc", 0.0))
            else:
                test_auc = state.get("test_auc", 0.0)
                train_auc = state.get("train_auc", 0.0)

        return ModelInfo(
            name=path.stem,
            path=path,
            file_size_mb=stat.st_size / (1024 * 1024),
            modified_date=datetime.fromtimestamp(stat.st_mtime),
            model_type="Ensemble" if swing_model and timing_model else "Single",
            features_count=features_count,
            has_scaler=has_scaler,
            has_dim_reduction=has_dim_reduction,
            swing_model_type=swing_type,
            timing_model_type=timing_type,
            test_auc=test_auc if isinstance(test_auc, (int, float)) else 0.0,
            train_auc=train_auc if isinstance(train_auc, (int, float)) else 0.0,
        )

    def get_model_details(self, model_name: str) -> Dict:
        """Get detailed information about a specific model."""
        model_path = self.model_dir / f"{model_name}.joblib"
        if not model_path.exists():
            return {"error": f"Model {model_name} not found"}

        state = joblib.load(model_path)

        details = {
            "name": model_name,
            "keys": list(state.keys()) if isinstance(state, dict) else ["model"],
        }

        if isinstance(state, dict):
            # Feature columns - multiple possible keys
            feature_cols = state.get("feature_columns") or state.get("feature_cols") or []
            details["feature_columns"] = feature_cols[:20]  # First 20
            details["total_features"] = len(feature_cols)

            # Model coefficients (for logistic regression)
            swing_model = (
                state.get("swing_model") or
                state.get("swing_l2") or
                state.get("models", {}).get("swing_l2") or
                state.get("models", {}).get("swing_model") or
                state.get("models", {}).get("swing", {}).get("l2")  # Nested structure
            )
            if swing_model is not None and hasattr(swing_model, "coef_"):
                coefs = swing_model.coef_.flatten()
                feature_names = feature_cols if feature_cols else [f"f{i}" for i in range(len(coefs))]

                # Ensure we have enough feature names
                if len(feature_names) >= len(coefs):
                    # Top 10 positive and negative coefficients
                    indices = np.argsort(coefs)
                    top_positive = [(feature_names[i], float(coefs[i])) for i in indices[-10:] if i < len(feature_names)][::-1]
                    top_negative = [(feature_names[i], float(coefs[i])) for i in indices[:10] if i < len(feature_names)]

                    details["top_positive_features"] = top_positive
                    details["top_negative_features"] = top_negative

            # Dimensionality reduction info - multiple possible keys
            dim_state = state.get("dim_state") or state.get("dim_reduction_state") or {}
            if dim_state:
                details["dim_reduction"] = {
                    "method": dim_state.get("method", "unknown"),
                    "n_components": dim_state.get("n_components", 0),
                }

            # Results info
            results = state.get("results", {})
            if results:
                details["results"] = {
                    "swing_test_auc": results.get("swing_test_auc", 0),
                    "timing_test_auc": results.get("timing_test_auc", 0),
                    "swing_train_auc": results.get("swing_train_auc", 0),
                }

        return details


# =============================================================================
# BACKTEST ANALYZER
# =============================================================================
class BacktestAnalyzer:
    """Analyze backtest results."""

    def __init__(self, reports_dir: Path = None):
        self.reports_dir = reports_dir or (project_root / "reports" / "backtests")
        self.backtests: List[BacktestSummary] = []

    def scan_backtests(self) -> List[BacktestSummary]:
        """Scan and load all backtest results."""
        self.backtests = []

        if not self.reports_dir.exists():
            return self.backtests

        for result_file in self.reports_dir.glob("backtest_results_*.json"):
            try:
                summary = self._load_backtest(result_file)
                if summary:
                    self.backtests.append(summary)
            except Exception as e:
                print(f"Error loading {result_file.name}: {e}")

        # Sort by date (newest first)
        self.backtests.sort(key=lambda b: b.run_date, reverse=True)
        return self.backtests

    def _load_backtest(self, path: Path) -> Optional[BacktestSummary]:
        """Load a single backtest result."""
        with open(path) as f:
            data = json.load(f)

        # Extract full historical results
        full = data.get("results", {}).get("full_historical", {})

        # Even if full_historical has error, create a summary with available data
        return BacktestSummary(
            run_id=path.stem,
            run_date=datetime.fromisoformat(data.get("run_date", datetime.now().isoformat())),
            total_return=full.get("total_return", 0) if "error" not in full else 0,
            sharpe_ratio=full.get("sharpe_ratio", 0) if "error" not in full else 0,
            max_drawdown=full.get("max_drawdown", 0) if "error" not in full else 0,
            win_rate=full.get("win_rate", 0) if "error" not in full else 0,
            total_trades=full.get("total_trades", 0) if "error" not in full else 0,
            robustness_score=data.get("overall_robustness_score", 0),
            regime_results=data.get("results", {}).get("regime_analysis", {}),
        )

    def get_entry_exit_analysis(self, backtest_id: str = None) -> Dict:
        """Get entry/exit timing analysis from backtest data."""
        # If no specific backtest, use most recent
        if backtest_id is None and self.backtests:
            backtest_id = self.backtests[0].run_id

        result_file = self.reports_dir / f"{backtest_id}.json"
        if not result_file.exists():
            return {"error": f"Backtest {backtest_id} not found"}

        with open(result_file) as f:
            data = json.load(f)

        analysis = {
            "backtest_id": backtest_id,
            "full_historical": data.get("results", {}).get("full_historical", {}),
            "walk_forward": data.get("results", {}).get("walk_forward", {}),
            "regime_analysis": data.get("results", {}).get("regime_analysis", {}),
            "robustness": data.get("results", {}).get("robustness", {}),
            "monte_carlo": data.get("results", {}).get("monte_carlo", {}),
        }

        return analysis


# =============================================================================
# POSITION HISTORY
# =============================================================================
def get_position_history() -> Dict:
    """Get position and equity history for charts."""
    from src.position_tracker import get_tracker

    tracker = get_tracker()

    return {
        "positions": tracker.get_position_history(hours=48),
        "trades": tracker.get_trade_history(days=7),
        "equity": tracker.get_equity_history(hours=48),
        "stats": tracker.get_stats(),
    }


def get_equity_chart_data() -> List[Dict]:
    """Get equity data formatted for Chart.js."""
    equity_file = project_root / "logs" / "equity_history.json"

    if not equity_file.exists():
        return []

    try:
        with open(equity_file) as f:
            data = json.load(f)
        return data[-500:]  # Last 500 points
    except Exception:
        return []


def get_pnl_chart_data() -> List[Dict]:
    """Get P&L data formatted for Chart.js."""
    equity = get_equity_chart_data()
    return [{"x": e.get("timestamp", ""), "y": e.get("daily_pnl", 0)} for e in equity]


def get_position_chart_data() -> List[Dict]:
    """Get position history data for Chart.js."""
    pos_file = project_root / "logs" / "position_history.json"

    if not pos_file.exists():
        return []

    try:
        with open(pos_file) as f:
            data = json.load(f)
        return data[-500:]  # Last 500 points
    except Exception:
        return []


def get_trade_history() -> List[Dict]:
    """Get trade history."""
    trade_file = project_root / "logs" / "trade_history.json"

    if not trade_file.exists():
        return []

    try:
        with open(trade_file) as f:
            return json.load(f)
    except Exception:
        return []


# =============================================================================
# SYSTEM STATUS
# =============================================================================
def get_system_status() -> Dict:
    """Get current system status."""
    status_file = project_root / "logs" / "status.json"

    if not status_file.exists():
        return {"mode": "UNKNOWN", "error": "Status file not found"}

    with open(status_file) as f:
        return json.load(f)


# =============================================================================
# HTML DASHBOARD GENERATOR
# =============================================================================
def generate_html_dashboard() -> str:
    """Generate an HTML dashboard."""
    model_analyzer = ModelAnalyzer()
    models = model_analyzer.scan_models()

    backtest_analyzer = BacktestAnalyzer()
    backtests = backtest_analyzer.scan_backtests()

    status = get_system_status()

    # Get detailed info for the most recent model
    model_details = {}
    if models:
        model_details = model_analyzer.get_model_details(models[0].name)

    # Get entry/exit analysis
    entry_exit = {}
    if backtests:
        entry_exit = backtest_analyzer.get_entry_exit_analysis()

    # Get position/equity history for charts
    equity_data = get_equity_chart_data()
    pnl_data = get_pnl_chart_data()
    position_data = get_position_chart_data()
    trade_history = get_trade_history()

    # Prepare chart data as JSON
    equity_chart_json = json.dumps([
        {"x": e.get("timestamp", "")[:19], "y": e.get("equity", 0)}
        for e in equity_data[-200:]  # Last 200 points
    ])
    pnl_chart_json = json.dumps([
        {"x": e.get("timestamp", "")[:19], "y": e.get("daily_pnl", 0)}
        for e in equity_data[-200:]
    ])
    position_chart_json = json.dumps([
        {"x": p.get("timestamp", "")[:19], "y": p.get("unrealized_pnl", 0), "qty": p.get("quantity", 0)}
        for p in position_data[-200:]
    ])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GIGA TRADER Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }}
        .container {{ max-width: 1400px; margin: 0 auto; }}
        h1 {{
            text-align: center;
            font-size: 2.5rem;
            margin-bottom: 10px;
            background: linear-gradient(90deg, #00d4ff, #7b2cbf);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .subtitle {{ text-align: center; color: #888; margin-bottom: 30px; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .card {{
            background: rgba(255,255,255,0.05);
            border-radius: 15px;
            padding: 20px;
            border: 1px solid rgba(255,255,255,0.1);
            backdrop-filter: blur(10px);
        }}
        .card h2 {{
            font-size: 1.1rem;
            color: #00d4ff;
            margin-bottom: 15px;
            padding-bottom: 10px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .metric {{
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }}
        .metric-label {{ color: #888; }}
        .metric-value {{ font-weight: bold; }}
        .metric-value.positive {{ color: #00ff88; }}
        .metric-value.negative {{ color: #ff4444; }}
        .status-badge {{
            display: inline-block;
            padding: 4px 12px;
            border-radius: 20px;
            font-size: 0.8rem;
            font-weight: bold;
        }}
        .status-healthy {{ background: #00ff8833; color: #00ff88; }}
        .status-trading {{ background: #00d4ff33; color: #00d4ff; }}
        .status-experimenting {{ background: #7b2cbf33; color: #b388ff; }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 10px;
        }}
        th, td {{
            padding: 10px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        th {{ color: #00d4ff; font-weight: 600; }}
        .progress-bar {{
            background: rgba(255,255,255,0.1);
            border-radius: 10px;
            height: 8px;
            overflow: hidden;
        }}
        .progress-fill {{
            height: 100%;
            border-radius: 10px;
            transition: width 0.3s ease;
        }}
        .fill-green {{ background: linear-gradient(90deg, #00ff88, #00d4ff); }}
        .fill-yellow {{ background: linear-gradient(90deg, #ffaa00, #ff6600); }}
        .fill-red {{ background: linear-gradient(90deg, #ff4444, #ff0000); }}
        .feature-tag {{
            display: inline-block;
            background: rgba(0, 212, 255, 0.2);
            color: #00d4ff;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 0.75rem;
            margin: 2px;
        }}
        .regime-card {{
            background: rgba(255,255,255,0.03);
            border-radius: 8px;
            padding: 10px;
            margin: 5px 0;
        }}
        .regime-name {{ font-weight: bold; color: #00d4ff; }}
        .timestamp {{ color: #666; font-size: 0.8rem; }}
        .chart-container {{
            position: relative;
            height: 300px;
            margin: 15px 0;
        }}
        .chart-container canvas {{
            background: rgba(0,0,0,0.2);
            border-radius: 8px;
        }}
        .trade-table {{
            max-height: 300px;
            overflow-y: auto;
        }}
        .trade-row.profit {{ background: rgba(0, 255, 136, 0.05); }}
        .trade-row.loss {{ background: rgba(255, 68, 68, 0.05); }}
        .no-data {{
            text-align: center;
            color: #666;
            padding: 40px;
            font-style: italic;
        }}
        .chart-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        @media (max-width: 900px) {{
            .chart-grid {{ grid-template-columns: 1fr; }}
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🚀 GIGA TRADER Dashboard</h1>
        <p class="subtitle">ML Trading System Status & Analytics</p>

        <!-- System Status -->
        <div class="grid">
            <div class="card">
                <h2>📊 System Status</h2>
                <div class="metric">
                    <span class="metric-label">Mode</span>
                    <span class="status-badge status-{status.get('mode', 'unknown').lower()}">{status.get('mode', 'UNKNOWN')}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Health</span>
                    <span class="status-badge status-healthy">{status.get('health', {}).get('status', 'UNKNOWN')}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Uptime</span>
                    <span class="metric-value">{status.get('uptime_seconds', 0) // 3600}h {(status.get('uptime_seconds', 0) % 3600) // 60}m</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Last Update</span>
                    <span class="timestamp">{status.get('last_update', 'N/A')[:19]}</span>
                </div>
            </div>

            <div class="card">
                <h2>💰 Trading Performance</h2>
                <div class="metric">
                    <span class="metric-label">Position</span>
                    <span class="metric-value">{status.get('trading', {}).get('position', 'FLAT')}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Position P&L</span>
                    <span class="metric-value {'positive' if status.get('trading', {}).get('position_pnl', 0) >= 0 else 'negative'}">${status.get('trading', {}).get('position_pnl', 0):,.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Daily P&L</span>
                    <span class="metric-value {'positive' if status.get('trading', {}).get('daily_pnl', 0) >= 0 else 'negative'}">${status.get('trading', {}).get('daily_pnl', 0):,.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Trades Today</span>
                    <span class="metric-value">{status.get('trading', {}).get('trades_today', 0)}</span>
                </div>
            </div>

            <div class="card">
                <h2>🔧 Components</h2>
                {''.join(f'''<div class="metric">
                    <span class="metric-label">{name.replace('_', ' ').title()}</span>
                    <span class="status-badge status-{'healthy' if state in ['RUNNING', 'IDLE'] else 'experimenting'}">{state}</span>
                </div>''' for name, state in status.get('components', {}).items())}
            </div>
        </div>

        <!-- Time Series Charts -->
        <div class="card" style="margin-bottom: 30px;">
            <h2>📈 Equity & P&L Time Series</h2>
            <div class="chart-grid">
                <div>
                    <h3 style="color: #888; font-size: 0.9rem; margin-bottom: 10px;">Account Equity</h3>
                    <div class="chart-container">
                        {'<canvas id="equityChart"></canvas>' if equity_data else '<div class="no-data">No equity data available yet. Start trading to see equity history.</div>'}
                    </div>
                </div>
                <div>
                    <h3 style="color: #888; font-size: 0.9rem; margin-bottom: 10px;">Daily P&L</h3>
                    <div class="chart-container">
                        {'<canvas id="pnlChart"></canvas>' if equity_data else '<div class="no-data">No P&L data available yet.</div>'}
                    </div>
                </div>
            </div>
        </div>

        <!-- Position History Chart -->
        <div class="card" style="margin-bottom: 30px;">
            <h2>📊 Position History</h2>
            <div class="chart-container" style="height: 250px;">
                {'<canvas id="positionChart"></canvas>' if position_data else '<div class="no-data">No position history available yet. Open positions will be tracked here.</div>'}
            </div>
        </div>

        <!-- Recent Trades -->
        <div class="card" style="margin-bottom: 30px;">
            <h2>📝 Recent Trades ({len(trade_history)} total)</h2>
            {f'''<div class="trade-table">
                <table>
                    <tr>
                        <th>Time</th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Qty</th>
                        <th>Price</th>
                        <th>P&L</th>
                        <th>Signal</th>
                    </tr>
                    {''.join(f"""<tr class="trade-row {'profit' if t.get('pnl', 0) > 0 else 'loss' if t.get('pnl', 0) < 0 else ''}">
                        <td class="timestamp">{t.get('timestamp', '')[:19]}</td>
                        <td><strong>{t.get('symbol', 'SPY')}</strong></td>
                        <td style="color: {'#00ff88' if t.get('side') == 'buy' else '#ff4444'};">{t.get('side', '').upper()}</td>
                        <td>{t.get('quantity', 0)}</td>
                        <td>${t.get('price', 0):.2f}</td>
                        <td class="metric-value {'positive' if t.get('pnl', 0) >= 0 else 'negative'}">${t.get('pnl', 0):,.2f}</td>
                        <td><span class="feature-tag">{t.get('signal_type', 'N/A')}</span></td>
                    </tr>""" for t in trade_history[-20:][::-1])}
                </table>
            </div>''' if trade_history else '<div class="no-data">No trades recorded yet.</div>'}
        </div>

        <!-- Models Section -->
        <div class="card" style="margin-bottom: 30px;">
            <h2>🤖 Trained Models ({len(models)} total)</h2>
            <table>
                <tr>
                    <th>Model Name</th>
                    <th>Size</th>
                    <th>Features</th>
                    <th>Swing Type</th>
                    <th>Timing Type</th>
                    <th>Modified</th>
                </tr>
                {''.join(f'''<tr>
                    <td><strong>{m.name}</strong></td>
                    <td>{m.file_size_mb:.1f} MB</td>
                    <td>{m.features_count}</td>
                    <td>{m.swing_model_type}</td>
                    <td>{m.timing_model_type}</td>
                    <td class="timestamp">{m.modified_date.strftime('%Y-%m-%d %H:%M')}</td>
                </tr>''' for m in models)}
            </table>
        </div>

        <!-- Top Features Section -->
        {f'''<div class="card" style="margin-bottom: 30px;">
            <h2>📈 Top Features (from {models[0].name if models else 'N/A'})</h2>
            <div class="grid" style="grid-template-columns: 1fr 1fr;">
                <div>
                    <h3 style="color: #00ff88; font-size: 0.9rem; margin-bottom: 10px;">Top Bullish Features</h3>
                    {''.join(f'<div class="metric"><span class="metric-label">{f[0][:30]}</span><span class="metric-value positive">+{f[1]:.3f}</span></div>' for f in model_details.get('top_positive_features', [])[:5])}
                </div>
                <div>
                    <h3 style="color: #ff4444; font-size: 0.9rem; margin-bottom: 10px;">Top Bearish Features</h3>
                    {''.join(f'<div class="metric"><span class="metric-label">{f[0][:30]}</span><span class="metric-value negative">{f[1]:.3f}</span></div>' for f in model_details.get('top_negative_features', [])[:5])}
                </div>
            </div>
        </div>''' if model_details.get('top_positive_features') else ''}

        <!-- Backtest Results -->
        <div class="card" style="margin-bottom: 30px;">
            <h2>📊 Recent Backtests ({len(backtests)} total)</h2>
            {f'''<table>
                <tr>
                    <th>Date</th>
                    <th>Return</th>
                    <th>Sharpe</th>
                    <th>Max DD</th>
                    <th>Win Rate</th>
                    <th>Robustness</th>
                </tr>
                {''.join(f"""<tr>
                    <td class="timestamp">{b.run_date.strftime('%Y-%m-%d %H:%M')}</td>
                    <td class="metric-value {'positive' if b.total_return >= 0 else 'negative'}">{b.total_return:.1%}</td>
                    <td>{b.sharpe_ratio:.2f}</td>
                    <td class="metric-value negative">{b.max_drawdown:.1%}</td>
                    <td>{b.win_rate:.1%}</td>
                    <td>
                        <div class="progress-bar">
                            <div class="progress-fill {'fill-green' if b.robustness_score >= 0.6 else 'fill-yellow' if b.robustness_score >= 0.4 else 'fill-red'}" style="width: {b.robustness_score * 100}%;"></div>
                        </div>
                        <span style="font-size: 0.8rem;">{b.robustness_score:.2f}</span>
                    </td>
                </tr>""" for b in backtests[:5])}
            </table>''' if backtests else '<p style="color: #888;">No backtest results found</p>'}
        </div>

        <!-- Regime Analysis -->
        {f'''<div class="card" style="margin-bottom: 30px;">
            <h2>🌍 Regime Performance Analysis</h2>
            <div class="grid" style="grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));">
                {''.join(f"""<div class="regime-card">
                    <div class="regime-name">{regime}</div>
                    <div class="metric">
                        <span class="metric-label">Trend</span>
                        <span class="metric-value">{data.get('trend', 'N/A')}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">SPY Return</span>
                        <span class="metric-value {'positive' if data.get('spy_return', 0) >= 0 else 'negative'}">{data.get('spy_return', 0):.1%}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Strategy Return</span>
                        <span class="metric-value {'positive' if data.get('strategy_return', 0) >= 0 else 'negative'}">{data.get('strategy_return', 0):.1%}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Accuracy</span>
                        <span class="metric-value">{data.get('accuracy', 0):.1%}</span>
                    </div>
                    <div class="metric">
                        <span class="metric-label">Outperformed</span>
                        <span class="status-badge {'status-healthy' if data.get('outperformed_spy', False) else 'status-experimenting'}">{'Yes ✓' if data.get('outperformed_spy', False) else 'No'}</span>
                    </div>
                </div>""" for regime, data in entry_exit.get('regime_analysis', {}).items() if isinstance(data, dict) and 'trend' in data)}
            </div>
        </div>''' if entry_exit.get('regime_analysis') else ''}

        <!-- Monte Carlo Results -->
        {f'''<div class="card" style="margin-bottom: 30px;">
            <h2>🎲 Monte Carlo Stress Test</h2>
            <div class="grid">
                <div class="metric">
                    <span class="metric-label">Simulations</span>
                    <span class="metric-value">{entry_exit.get('monte_carlo', {}).get('n_simulations', 0)}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Median Sharpe</span>
                    <span class="metric-value">{entry_exit.get('monte_carlo', {}).get('median_sharpe', 0):.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">5th Percentile Sharpe</span>
                    <span class="metric-value">{entry_exit.get('monte_carlo', {}).get('sharpe_5th_percentile', 0):.2f}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Prob. Profitable</span>
                    <span class="metric-value positive">{entry_exit.get('monte_carlo', {}).get('probability_profitable', 0):.1%}</span>
                </div>
                <div class="metric">
                    <span class="metric-label">Worst Case DD</span>
                    <span class="metric-value negative">{entry_exit.get('monte_carlo', {}).get('worst_case_drawdown', 0):.1%}</span>
                </div>
            </div>
        </div>''' if entry_exit.get('monte_carlo') else ''}

        <!-- Robustness Tests -->
        {f'''<div class="card">
            <h2>🛡️ Robustness Tests</h2>
            <h3 style="color: #888; font-size: 0.9rem; margin: 10px 0;">Noise Tolerance</h3>
            {''.join(f"""<div class="metric">
                <span class="metric-label">Noise {r.get('noise_level', 0):.0%}</span>
                <span class="metric-value">Accuracy: {r.get('accuracy', 0):.1%} (Drop: {r.get('accuracy_drop', 0):.1%})</span>
            </div>""" for r in entry_exit.get('robustness', {}).get('noise_tolerance', []))}

            <h3 style="color: #888; font-size: 0.9rem; margin: 10px 0;">Feature Subset Stability</h3>
            {''.join(f"""<div class="metric">
                <span class="metric-label">Drop {r.get('features_dropped_pct', 0):.0%} features</span>
                <span class="metric-value">Accuracy: {r.get('mean_accuracy', 0):.1%} (±{r.get('std_accuracy', 0):.1%})</span>
            </div>""" for r in entry_exit.get('robustness', {}).get('feature_subset_stability', []))}

            <div class="metric" style="margin-top: 15px; padding-top: 15px; border-top: 2px solid rgba(255,255,255,0.1);">
                <span class="metric-label" style="font-weight: bold;">Overall Robustness Score</span>
                <span class="metric-value positive" style="font-size: 1.2rem;">{entry_exit.get('robustness', {}).get('overall_score', 0):.2f}</span>
            </div>
        </div>''' if entry_exit.get('robustness') else ''}

        <p class="timestamp" style="text-align: center; margin-top: 30px;">
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | GIGA TRADER Dashboard v1.0
        </p>
    </div>

    <script>
        // Chart.js configuration
        Chart.defaults.color = '#888';
        Chart.defaults.borderColor = 'rgba(255,255,255,0.1)';

        const equityData = {equity_chart_json};
        const pnlData = {pnl_chart_json};
        const positionData = {position_chart_json};

        // Equity Chart
        if (document.getElementById('equityChart') && equityData.length > 0) {{
            new Chart(document.getElementById('equityChart'), {{
                type: 'line',
                data: {{
                    labels: equityData.map(d => d.x),
                    datasets: [{{
                        label: 'Account Equity',
                        data: equityData.map(d => d.y),
                        borderColor: '#00d4ff',
                        backgroundColor: 'rgba(0, 212, 255, 0.1)',
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0,
                        pointHoverRadius: 5,
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }},
                        tooltip: {{
                            mode: 'index',
                            intersect: false,
                            callbacks: {{
                                label: (ctx) => `$${'{'}ctx.parsed.y.toLocaleString(){'}'}`
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            display: true,
                            ticks: {{ maxTicksLimit: 8, maxRotation: 0 }}
                        }},
                        y: {{
                            display: true,
                            ticks: {{
                                callback: (value) => '$' + value.toLocaleString()
                            }}
                        }}
                    }},
                    interaction: {{ mode: 'nearest', axis: 'x', intersect: false }}
                }}
            }});
        }}

        // P&L Chart
        if (document.getElementById('pnlChart') && pnlData.length > 0) {{
            new Chart(document.getElementById('pnlChart'), {{
                type: 'bar',
                data: {{
                    labels: pnlData.map(d => d.x),
                    datasets: [{{
                        label: 'Daily P&L',
                        data: pnlData.map(d => d.y),
                        backgroundColor: pnlData.map(d => d.y >= 0 ? 'rgba(0, 255, 136, 0.6)' : 'rgba(255, 68, 68, 0.6)'),
                        borderColor: pnlData.map(d => d.y >= 0 ? '#00ff88' : '#ff4444'),
                        borderWidth: 1,
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ display: false }},
                        tooltip: {{
                            callbacks: {{
                                label: (ctx) => `$${'{'}ctx.parsed.y.toFixed(2){'}'}`
                            }}
                        }}
                    }},
                    scales: {{
                        x: {{
                            display: true,
                            ticks: {{ maxTicksLimit: 8, maxRotation: 0 }}
                        }},
                        y: {{
                            display: true,
                            ticks: {{
                                callback: (value) => '$' + value.toFixed(0)
                            }}
                        }}
                    }}
                }}
            }});
        }}

        // Position Chart
        if (document.getElementById('positionChart') && positionData.length > 0) {{
            new Chart(document.getElementById('positionChart'), {{
                type: 'line',
                data: {{
                    labels: positionData.map(d => d.x),
                    datasets: [{{
                        label: 'Unrealized P&L',
                        data: positionData.map(d => d.y),
                        borderColor: '#7b2cbf',
                        backgroundColor: 'rgba(123, 44, 191, 0.1)',
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0,
                        yAxisID: 'y',
                    }}, {{
                        label: 'Position Size',
                        data: positionData.map(d => d.qty),
                        borderColor: '#00d4ff',
                        borderDash: [5, 5],
                        fill: false,
                        tension: 0.1,
                        pointRadius: 0,
                        yAxisID: 'y1',
                    }}]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{
                        legend: {{ position: 'top' }},
                        tooltip: {{ mode: 'index', intersect: false }}
                    }},
                    scales: {{
                        x: {{
                            display: true,
                            ticks: {{ maxTicksLimit: 10, maxRotation: 0 }}
                        }},
                        y: {{
                            type: 'linear',
                            display: true,
                            position: 'left',
                            title: {{ display: true, text: 'Unrealized P&L ($)' }},
                            ticks: {{ callback: (v) => '$' + v.toFixed(0) }}
                        }},
                        y1: {{
                            type: 'linear',
                            display: true,
                            position: 'right',
                            title: {{ display: true, text: 'Quantity' }},
                            grid: {{ drawOnChartArea: false }}
                        }}
                    }},
                    interaction: {{ mode: 'nearest', axis: 'x', intersect: false }}
                }}
            }});
        }}
    </script>
</body>
</html>"""

    return html


# =============================================================================
# CONSOLE OUTPUT
# =============================================================================
def print_console_dashboard():
    """Print dashboard to console."""
    model_analyzer = ModelAnalyzer()
    models = model_analyzer.scan_models()

    backtest_analyzer = BacktestAnalyzer()
    backtests = backtest_analyzer.scan_backtests()

    status = get_system_status()

    print("\n" + "=" * 70)
    print("GIGA TRADER DASHBOARD")
    print("=" * 70)

    # System Status
    print(f"\n[SYSTEM STATUS]")
    print(f"   Mode: {status.get('mode', 'UNKNOWN')}")
    print(f"   Health: {status.get('health', {}).get('status', 'UNKNOWN')}")
    print(f"   Uptime: {status.get('uptime_seconds', 0) // 3600}h {(status.get('uptime_seconds', 0) % 3600) // 60}m")

    # Trading
    print(f"\n[TRADING]")
    print(f"   Position: {status.get('trading', {}).get('position', 'FLAT')}")
    print(f"   Position P&L: ${status.get('trading', {}).get('position_pnl', 0):,.2f}")
    print(f"   Daily P&L: ${status.get('trading', {}).get('daily_pnl', 0):,.2f}")

    # Models
    print(f"\n[MODELS] ({len(models)} total)")
    for m in models[:3]:
        print(f"   {m.name}: {m.features_count} features, {m.swing_model_type}/{m.timing_model_type}")

    # Backtests
    print(f"\n[RECENT BACKTESTS] ({len(backtests)} total)")
    for b in backtests[:3]:
        print(f"   {b.run_date.strftime('%Y-%m-%d')}: Return={b.total_return:.1%}, Sharpe={b.sharpe_ratio:.2f}, Robustness={b.robustness_score:.2f}")

    print("\n" + "=" * 70)


# =============================================================================
# MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="GIGA TRADER Dashboard")
    parser.add_argument("--serve", action="store_true", help="Start live server")
    parser.add_argument("--console", action="store_true", help="Console output only")
    parser.add_argument("--output", type=str, default="reports/dashboard.html", help="Output HTML file")
    args = parser.parse_args()

    if args.console:
        print_console_dashboard()
        return 0

    if args.serve:
        try:
            import http.server
            import socketserver

            PORT = 8050
            output_path = project_root / args.output
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Generate dashboard
            html = generate_html_dashboard()
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(html)

            os.chdir(output_path.parent)

            Handler = http.server.SimpleHTTPServer
            with socketserver.TCPServer(("", PORT), Handler) as httpd:
                print(f"[SERVER] Dashboard running at http://localhost:{PORT}")
                print(f"   Open {output_path} in browser")
                print("   Press Ctrl+C to stop")
                httpd.serve_forever()

        except ImportError:
            print("http.server not available. Generating static HTML instead.")
            args.serve = False

    # Generate static HTML
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    html = generate_html_dashboard()
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[OK] Dashboard generated: {output_path}")
    print(f"     Open in browser: file:///{output_path.as_posix()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
