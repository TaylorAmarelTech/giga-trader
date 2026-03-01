"""
End-to-end paper trade validation script.

Tests the full inference pipeline:
  1. Load best models from registry
  2. Download latest SPY 1-min data from Alpaca
  3. Generate a trading signal
  4. Validate signal parameters
  5. Optionally place a small paper order (--execute flag)

Usage:
    python scripts/test_paper_trade.py          # Dry run (signal only)
    python scripts/test_paper_trade.py --execute # Place 1-share paper order
"""

import sys
import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("PaperTradeTest")


def check_registry():
    """Check registry for available models."""
    print("\n" + "=" * 60)
    print("STEP 1: CHECK MODEL REGISTRY")
    print("=" * 60)

    from src.core.registry_db import get_registry_db
    db = get_registry_db()
    stats = db.get_model_statistics()

    total = stats.get("total_models", 0)
    by_tier = stats.get("by_tier", {})
    best_auc = stats.get("best_cv_auc", 0)

    print(f"  Total models: {total}")
    for tier, count in sorted(by_tier.items()):
        print(f"  Tier {tier}: {count} models")
    print(f"  Best CV AUC: {best_auc:.4f}")

    if total == 0:
        print("\n  [FAIL] No models in registry. Run experiments first.")
        return False

    # Get best model details
    import sqlite3
    conn = sqlite3.connect(str(db.db_path))
    cursor = conn.cursor()
    cursor.execute(
        "SELECT model_id, test_auc, tier, wmes_score, stability_score "
        "FROM models ORDER BY test_auc DESC LIMIT 1"
    )
    best = cursor.fetchone()
    conn.close()

    if best:
        print(f"\n  Best model: {best[0]}")
        print(f"    Test AUC: {best[1]:.4f}")
        print(f"    Tier: {best[2]}")
        print(f"    WMES: {best[3]:.4f}")
        print(f"    Stability: {best[4]:.4f}")

    print("\n  [PASS] Registry has models")
    return True


def check_models_on_disk():
    """Verify production model files exist."""
    print("\n" + "=" * 60)
    print("STEP 2: CHECK MODEL FILES")
    print("=" * 60)

    model_dir = project_root / "models" / "production"
    if not model_dir.exists():
        print(f"  [FAIL] Model directory not found: {model_dir}")
        return False

    model_files = list(model_dir.glob("*.joblib"))
    print(f"  Model directory: {model_dir}")
    print(f"  Model files found: {len(model_files)}")

    for f in model_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        age_hours = (datetime.now().timestamp() - f.stat().st_mtime) / 3600
        print(f"    {f.name}: {size_mb:.1f} MB, {age_hours:.1f}h old")

    # Check for leak-proof model specifically
    leak_proof = model_dir / "spy_leak_proof_models.joblib"
    if leak_proof.exists():
        print(f"\n  [PASS] Leak-proof model found ({leak_proof.stat().st_size / (1024*1024):.1f} MB)")

        import joblib
        data = joblib.load(leak_proof)
        keys = list(data.keys())
        print(f"    Model keys: {keys}")
        if "feature_columns" in data:
            print(f"    Feature columns: {len(data['feature_columns'])}")
        if "swing_pipeline" in data:
            print(f"    Swing pipeline: {type(data['swing_pipeline']).__name__}")
        if "timing_pipeline" in data:
            print(f"    Timing pipeline: {type(data['timing_pipeline']).__name__}")
        return True
    else:
        legacy = model_dir / "spy_robust_models.joblib"
        if legacy.exists():
            print(f"\n  [PASS] Legacy model found")
            return True

    print("\n  [FAIL] No recognized model files found")
    return False


def download_data():
    """Download latest SPY 1-min data from Alpaca."""
    print("\n" + "=" * 60)
    print("STEP 3: DOWNLOAD LATEST DATA")
    print("=" * 60)

    try:
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        from src.phase_19_paper_trading.alpaca_client import AlpacaPaperClient

        client = AlpacaPaperClient()

        # Check account
        account = client.get_account()
        print(f"  Account equity: ${account['equity']:,.2f}")
        print(f"  Buying power: ${account['buying_power']:,.2f}")

        # Download 30 days of 1-min data
        end = datetime.now()
        start = end - timedelta(days=30)

        print(f"  Fetching SPY 1-min data: {start.date()} to {end.date()}...")
        request = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
        )
        bars = client.data_client.get_stock_bars(request)
        df = bars.df.reset_index()

        print(f"  Downloaded {len(df)} bars")
        print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
        print(f"  Columns: {list(df.columns)}")

        # Get current price
        current_price = client.get_latest_price("SPY")
        print(f"  Current SPY price: ${current_price:.2f}")

        print(f"\n  [PASS] Data downloaded successfully")
        return df, current_price, client

    except Exception as e:
        print(f"\n  [FAIL] Data download failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def generate_signal(df, current_price):
    """Generate a trading signal from the data."""
    print("\n" + "=" * 60)
    print("STEP 4: GENERATE SIGNAL")
    print("=" * 60)

    try:
        from src.phase_19_paper_trading.signal_generator import SignalGenerator

        print("  Initializing signal generator...")
        gen = SignalGenerator(use_dynamic_selector=True)

        print(f"  Models loaded: {list(gen.models.keys())}")
        print(f"  Leak-proof mode: {gen.use_leak_proof}")
        print(f"  Feature columns: {len(gen.feature_cols) if gen.feature_cols else 'N/A'}")
        print(f"  Dynamic selector: {gen.dynamic_selector is not None}")

        # Prepare data format (match what TradingBot.fetch_latest_data produces)
        df_prep = df.copy()
        df_prep = df_prep.rename(columns={
            "open": "open", "high": "high", "low": "low",
            "close": "close", "volume": "volume",
        })
        if "timestamp" not in df_prep.columns:
            for col in df_prep.columns:
                if pd.api.types.is_datetime64_any_dtype(df_prep[col]):
                    df_prep = df_prep.rename(columns={col: "timestamp"})
                    break

        df_prep["timestamp"] = pd.to_datetime(df_prep["timestamp"])
        df_prep = df_prep.set_index("timestamp")

        print(f"\n  Generating signal (this may take 1-2 minutes)...")
        import time
        t0 = time.time()

        signal = gen.generate_signal(df_prep, current_price)
        elapsed = time.time() - t0

        print(f"\n  Signal generated in {elapsed:.1f}s:")
        print(f"    Type: {signal.signal_type.value}")
        print(f"    Probability: {signal.probability:.4f}")
        print(f"    Confidence: {signal.confidence:.4f}")
        print(f"    Entry Price: ${signal.entry_price:.2f}" if signal.entry_price else "    Entry Price: N/A")
        print(f"    Stop Loss: ${signal.stop_loss:.2f}" if signal.stop_loss else "    Stop Loss: N/A")
        print(f"    Take Profit: ${signal.take_profit:.2f}" if signal.take_profit else "    Take Profit: N/A")
        print(f"    Position Size: {signal.position_size_pct:.2%}" if signal.position_size_pct else "    Position Size: N/A")

        if signal.metadata:
            print(f"    Metadata:")
            for k, v in signal.metadata.items():
                if k != "model_predictions":
                    print(f"      {k}: {v}")

        # Validate signal sanity
        issues = []
        if signal.probability < 0 or signal.probability > 1:
            issues.append(f"probability out of range: {signal.probability}")
        if signal.confidence < 0 or signal.confidence > 1:
            issues.append(f"confidence out of range: {signal.confidence}")
        if signal.stop_loss and signal.stop_loss <= 0:
            issues.append(f"invalid stop loss: {signal.stop_loss}")
        if signal.take_profit and signal.take_profit <= 0:
            issues.append(f"invalid take profit: {signal.take_profit}")

        if issues:
            print(f"\n  [WARN] Signal has issues: {issues}")
        else:
            print(f"\n  [PASS] Signal is valid")

        return signal

    except Exception as e:
        print(f"\n  [FAIL] Signal generation failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def execute_paper_order(signal, current_price, client):
    """Place a small paper order on Alpaca."""
    print("\n" + "=" * 60)
    print("STEP 5: EXECUTE PAPER ORDER (1 share)")
    print("=" * 60)

    from src.phase_19_paper_trading.alpaca_client import SignalType

    if signal.signal_type == SignalType.HOLD:
        print("  Signal is HOLD — skipping order")
        print("  [SKIP] No order to place")
        return True

    try:
        side = "buy" if signal.signal_type == SignalType.BUY else "sell"
        qty = 1

        print(f"  Placing {side.upper()} order for {qty} share of SPY @ ~${current_price:.2f}")

        # Use Alpaca API directly for a market order
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        order_request = MarketOrderRequest(
            symbol="SPY",
            qty=qty,
            side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )

        order = client.trading_client.submit_order(order_request)
        print(f"  Order submitted: {order.id}")
        print(f"    Status: {order.status}")
        print(f"    Side: {order.side}")
        print(f"    Qty: {order.qty}")

        # Wait for fill (up to 30 seconds)
        import time
        for i in range(30):
            time.sleep(1)
            order = client.trading_client.get_order_by_id(order.id)
            if order.status in ("filled", "partially_filled"):
                break
            if order.status in ("cancelled", "expired", "rejected"):
                print(f"  Order {order.status}: {getattr(order, 'rejected_reason', 'unknown')}")
                return False

        if order.status == "filled":
            fill_price = float(order.filled_avg_price)
            print(f"\n  Order FILLED:")
            print(f"    Fill price: ${fill_price:.2f}")
            print(f"    Slippage: {abs(fill_price - current_price) / current_price:.4%}")

            # Verify position
            position = client.get_position("SPY")
            if position:
                print(f"\n  Position confirmed:")
                print(f"    Qty: {position.quantity}")
                print(f"    Entry: ${position.entry_price:.2f}")

                # Close position immediately
                print(f"\n  Closing position...")
                from alpaca.trading.requests import ClosePositionRequest
                client.trading_client.close_position("SPY")
                time.sleep(3)

                # Verify closed
                try:
                    position = client.get_position("SPY")
                    if position:
                        print(f"  [WARN] Position still open")
                    else:
                        print(f"  Position closed successfully")
                except Exception:
                    print(f"  Position closed successfully")

            print(f"\n  [PASS] Paper order executed and closed")
            return True
        else:
            print(f"  Order not filled after 30s (status: {order.status})")
            # Cancel if still pending
            try:
                client.trading_client.cancel_order_by_id(order.id)
                print(f"  Cancelled pending order")
            except Exception:
                pass
            return False

    except Exception as e:
        print(f"\n  [FAIL] Paper order failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_health_check():
    """Run system health checks."""
    print("\n" + "=" * 60)
    print("STEP 6: SYSTEM HEALTH CHECK")
    print("=" * 60)

    try:
        from src.phase_20_monitoring.health_checker import HealthChecker

        checker = HealthChecker()
        results = checker.run_all_checks()
        overall = checker.get_overall_status()

        print(f"  Overall: {overall.value}")
        for name, result in results.items():
            status_icon = {"HEALTHY": "+", "DEGRADED": "~", "UNHEALTHY": "!", "UNKNOWN": "?"}
            icon = status_icon.get(result.status.value, "?")
            print(f"  [{icon}] {name}: {result.status.value} - {result.message}")

        if overall.value in ("HEALTHY", "DEGRADED"):
            print(f"\n  [PASS] System health: {overall.value}")
        else:
            print(f"\n  [WARN] System health: {overall.value}")

        return True
    except Exception as e:
        print(f"\n  [FAIL] Health check failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="End-to-end paper trade test")
    parser.add_argument("--execute", action="store_true",
                       help="Actually place a 1-share paper order")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("GIGA TRADER - END-TO-END PAPER TRADE TEST")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'EXECUTE (real paper order)' if args.execute else 'DRY RUN (signal only)'}")
    print("=" * 60)

    results = {}

    # Step 1: Check registry
    results["registry"] = check_registry()

    # Step 2: Check model files
    results["models_on_disk"] = check_models_on_disk()

    if not results["models_on_disk"]:
        print("\n[ABORT] No model files found. Run train_robust_model.py first.")
        return 1

    # Step 3: Download data
    df, current_price, client = download_data()
    results["data_download"] = df is not None

    if df is None:
        print("\n[ABORT] Could not download data.")
        return 1

    # Step 4: Generate signal
    signal = generate_signal(df, current_price)
    results["signal_generation"] = signal is not None

    # Step 5: Execute paper order (if --execute)
    if args.execute and signal is not None:
        results["paper_order"] = execute_paper_order(signal, current_price, client)
    else:
        results["paper_order"] = None  # Skipped

    # Step 6: Health check
    results["health_check"] = run_health_check()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    all_pass = True
    for step, passed in results.items():
        if passed is None:
            icon = "SKIP"
        elif passed:
            icon = "PASS"
        else:
            icon = "FAIL"
            all_pass = False
        print(f"  [{icon}] {step}")

    if all_pass:
        print(f"\n  All checks passed!")
    else:
        print(f"\n  Some checks failed — review output above")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
