"""
GIGA TRADER - Health Check Script
==================================
Run this to verify the entire pipeline is working correctly.

Usage:
    .venv/Scripts/python.exe scripts/health_check.py
"""

import os
import sys
from pathlib import Path
from datetime import datetime

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def print_header(title: str):
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)

def print_result(name: str, passed: bool, details: str = ""):
    status = "[PASS]" if passed else "[FAIL]"
    print(f"  {status} {name}")
    if details:
        print(f"         {details}")

def check_python_version():
    """Check Python version."""
    import platform
    version = platform.python_version()
    passed = version.startswith("3.12")
    return passed, f"Python {version}"

def check_env_file():
    """Check .env file exists and has API keys."""
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    passed = bool(api_key and secret_key)
    details = f"API Key: {'Found' if api_key else 'MISSING'}, Secret: {'Found' if secret_key else 'MISSING'}"
    return passed, details

def check_imports():
    """Check all modules import correctly."""
    results = []

    modules = [
        ("train_robust_model", "src.train_robust_model"),
        ("anti_overfit", "src.anti_overfit"),
        ("entry_exit_model", "src.entry_exit_model"),
        ("pipeline_grid", "src.pipeline_grid"),
        ("backtest_engine", "src.backtest_engine"),
    ]

    for name, module_path in modules:
        try:
            __import__(module_path)
            results.append((name, True, ""))
        except Exception as e:
            results.append((name, False, str(e)[:50]))

    return results

def check_dependencies():
    """Check required packages are installed."""
    results = []

    packages = [
        "numpy",
        "pandas",
        "sklearn",
        "joblib",
        "alpaca",
        "dotenv",
    ]

    for pkg in packages:
        try:
            if pkg == "sklearn":
                __import__("sklearn")
            elif pkg == "dotenv":
                __import__("dotenv")
            elif pkg == "alpaca":
                __import__("alpaca.data")
            else:
                __import__(pkg)
            results.append((pkg, True, ""))
        except ImportError as e:
            results.append((pkg, False, str(e)[:50]))

    return results

def check_models_directory():
    """Check models directory and saved models."""
    models_dir = project_root / "models" / "production"

    if not models_dir.exists():
        return False, "models/production/ does not exist"

    model_files = list(models_dir.glob("*.joblib"))
    if not model_files:
        return False, "No .joblib files found"

    return True, f"Found {len(model_files)} model files"

def check_config_files():
    """Check configuration files exist."""
    results = []

    configs = [
        "config/moltbot_workflow.yaml",
        "config/moltbot_full_pipeline.md",
        "CLAUDE.md",
        ".env",
    ]

    for config in configs:
        path = project_root / config
        exists = path.exists()
        results.append((config, exists, f"{'Found' if exists else 'MISSING'}"))

    return results

def check_data_directory():
    """Check data directory for cached data."""
    data_dir = project_root / "data"

    if not data_dir.exists():
        return False, "data/ directory does not exist"

    parquet_files = list(data_dir.glob("*.parquet"))
    return True, f"Found {len(parquet_files)} parquet files"

def run_quick_syntax_checks():
    """Run syntax checks on main files."""
    import ast
    results = []

    files = [
        "src/train_robust_model.py",
        "src/anti_overfit.py",
        "src/entry_exit_model.py",
        "src/pipeline_grid.py",
        "src/backtest_engine.py",
    ]

    for file in files:
        path = project_root / file
        if not path.exists():
            results.append((file, False, "File not found"))
            continue

        try:
            with open(path, encoding='utf-8') as f:
                ast.parse(f.read())
            results.append((file, True, ""))
        except SyntaxError as e:
            results.append((file, False, f"Line {e.lineno}: {e.msg}"))

    return results

def check_alpaca_connection():
    """Test Alpaca API connection."""
    try:
        from dotenv import load_dotenv
        load_dotenv(project_root / ".env")

        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from datetime import datetime, timedelta

        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            return False, "No API keys"

        client = StockHistoricalDataClient(api_key, secret_key)

        # Try to fetch 1 day of data
        end = datetime.now() - timedelta(days=1)
        start = end - timedelta(days=1)

        request = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )

        bars = client.get_stock_bars(request)
        n_bars = len(bars.df) if hasattr(bars, 'df') else 0

        return True, f"Connected, fetched {n_bars} bars"
    except Exception as e:
        return False, str(e)[:50]

def main():
    print("\n" + "="*60)
    print(" GIGA TRADER - HEALTH CHECK")
    print(f" {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    all_passed = True

    # Python Version
    print_header("Python Environment")
    passed, details = check_python_version()
    print_result("Python Version", passed, details)
    if not passed:
        all_passed = False
        print("\n  [CRITICAL] Use .venv/Scripts/python.exe instead of system Python!")

    # Environment
    print_header("Environment Configuration")
    passed, details = check_env_file()
    print_result(".env API Keys", passed, details)
    all_passed = all_passed and passed

    # Dependencies
    print_header("Package Dependencies")
    for name, passed, details in check_dependencies():
        print_result(name, passed, details)
        all_passed = all_passed and passed

    # Syntax Checks
    print_header("Syntax Checks")
    for name, passed, details in run_quick_syntax_checks():
        print_result(name, passed, details)
        all_passed = all_passed and passed

    # Module Imports
    print_header("Module Imports")
    for name, passed, details in check_imports():
        print_result(name, passed, details)
        all_passed = all_passed and passed

    # Config Files
    print_header("Configuration Files")
    for name, passed, details in check_config_files():
        print_result(name, passed, details)
        # Don't fail overall for missing non-critical configs

    # Models Directory
    print_header("Models Directory")
    passed, details = check_models_directory()
    print_result("Saved Models", passed, details)
    # Don't fail for missing models (can be trained)

    # Data Directory
    print_header("Data Directory")
    passed, details = check_data_directory()
    print_result("Cached Data", passed, details)
    # Don't fail for missing data (can be downloaded)

    # Alpaca Connection
    print_header("Alpaca API Connection")
    passed, details = check_alpaca_connection()
    print_result("API Connection", passed, details)
    if not passed:
        print("  [WARN] API connection failed - check keys and internet")

    # Summary
    print_header("SUMMARY")
    if all_passed:
        print("  [PASS] All critical checks passed!")
        print("  Ready to run: .venv/Scripts/python.exe src/train_robust_model.py")
    else:
        print("  [FAIL] Some checks failed - fix issues before running pipeline")

    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
