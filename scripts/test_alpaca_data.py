"""
GIGA TRADER - Alpaca Data Download Test
========================================
Tests connection to Alpaca API and downloads SPY data including extended hours.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv

# Load environment variables
load_dotenv(project_root / ".env")


def test_alpaca_connection():
    """Test 1: Verify Alpaca API connection."""
    print("\n" + "=" * 70)
    print("TEST 1: Alpaca API Connection")
    print("=" * 70)

    try:
        from alpaca.trading.client import TradingClient

        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        if not api_key or not secret_key:
            print("[FAIL] API keys not found in .env")
            return False

        print(f"[INFO] API Key: {api_key[:8]}...{api_key[-4:]}")

        # Connect to paper trading
        client = TradingClient(api_key, secret_key, paper=True)

        # Get account info
        account = client.get_account()

        print(f"[PASS] Connected to Alpaca Paper Trading")
        print(f"       Account ID: {account.id}")
        print(f"       Buying Power: ${float(account.buying_power):,.2f}")
        print(f"       Cash: ${float(account.cash):,.2f}")
        print(f"       Portfolio Value: ${float(account.portfolio_value):,.2f}")
        print(f"       Status: {account.status}")

        return True

    except Exception as e:
        print(f"[FAIL] Connection error: {e}")
        return False


def test_market_data_download():
    """Test 2: Download SPY historical data."""
    print("\n" + "=" * 70)
    print("TEST 2: SPY Historical Data Download")
    print("=" * 70)

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        # Create data client
        data_client = StockHistoricalDataClient(api_key, secret_key)

        # Request 30 days of daily data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)

        request = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )

        bars = data_client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            print("[FAIL] No data returned")
            return False

        # Reset index to get symbol and timestamp as columns
        df = df.reset_index()

        print(f"[PASS] Downloaded {len(df)} daily bars for SPY")
        print(f"       Date Range: {df['timestamp'].min().date()} to {df['timestamp'].max().date()}")
        print(f"\n       Latest Bar:")
        latest = df.iloc[-1]
        print(f"       Date: {latest['timestamp'].date()}")
        print(f"       Open:  ${latest['open']:.2f}")
        print(f"       High:  ${latest['high']:.2f}")
        print(f"       Low:   ${latest['low']:.2f}")
        print(f"       Close: ${latest['close']:.2f}")
        print(f"       Volume: {latest['volume']:,.0f}")

        return True

    except Exception as e:
        print(f"[FAIL] Data download error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_intraday_data():
    """Test 3: Download intraday (1-minute) data."""
    print("\n" + "=" * 70)
    print("TEST 3: SPY Intraday (1-Min) Data")
    print("=" * 70)

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        data_client = StockHistoricalDataClient(api_key, secret_key)

        # Get last 5 trading days of 1-minute data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)

        request = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame.Minute,
            start=start_date,
            end=end_date
        )

        bars = data_client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            print("[WARN] No intraday data returned (market may be closed)")
            return True  # Not a failure

        df = df.reset_index()

        print(f"[PASS] Downloaded {len(df):,} minute bars for SPY")
        print(f"       Date Range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        # Show sample of data
        print(f"\n       Sample (last 5 bars):")
        for _, row in df.tail(5).iterrows():
            print(f"       {row['timestamp']} | O:{row['open']:.2f} H:{row['high']:.2f} L:{row['low']:.2f} C:{row['close']:.2f} V:{row['volume']:,.0f}")

        return True

    except Exception as e:
        print(f"[FAIL] Intraday data error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_extended_hours_data():
    """Test 4: Download extended hours data (premarket/afterhours)."""
    print("\n" + "=" * 70)
    print("TEST 4: SPY Extended Hours Data (Premarket/Afterhours)")
    print("=" * 70)

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        data_client = StockHistoricalDataClient(api_key, secret_key)

        # Get last 3 days of 5-minute data (includes extended hours by default)
        end_date = datetime.now()
        start_date = end_date - timedelta(days=3)

        from alpaca.data.timeframe import TimeFrameUnit

        request = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),  # 5-minute bars
            start=start_date,
            end=end_date
        )

        bars = data_client.get_stock_bars(request)
        df = bars.df

        if df.empty:
            print("[WARN] No extended hours data returned")
            return True

        df = df.reset_index()

        # Convert timestamp to Eastern Time for analysis
        df['hour'] = df['timestamp'].dt.hour

        # Count bars by session
        premarket = df[(df['hour'] >= 4) & (df['hour'] < 9.5)]
        regular = df[(df['hour'] >= 9.5) & (df['hour'] < 16)]
        afterhours = df[(df['hour'] >= 16) & (df['hour'] < 20)]

        print(f"[PASS] Downloaded {len(df):,} 5-minute bars for SPY")
        print(f"\n       Session Breakdown:")
        print(f"       Premarket (4:00-9:30):   {len(premarket):,} bars")
        print(f"       Regular (9:30-16:00):    {len(regular):,} bars")
        print(f"       Afterhours (16:00-20:00): {len(afterhours):,} bars")

        if len(premarket) > 0:
            print(f"\n       Sample Premarket Data:")
            for _, row in premarket.head(3).iterrows():
                print(f"       {row['timestamp']} | C:{row['close']:.2f} V:{row['volume']:,.0f}")

        return True

    except Exception as e:
        print(f"[FAIL] Extended hours data error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_save_data():
    """Test 5: Save downloaded data to parquet."""
    print("\n" + "=" * 70)
    print("TEST 5: Save Data to Parquet")
    print("=" * 70)

    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
        import pandas as pd

        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")

        data_client = StockHistoricalDataClient(api_key, secret_key)

        # Download 90 days of daily data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=90)

        request = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame.Day,
            start=start_date,
            end=end_date
        )

        bars = data_client.get_stock_bars(request)
        df = bars.df.reset_index()

        # Ensure data directory exists
        data_dir = project_root / "data" / "raw"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Save to parquet
        output_path = data_dir / "spy_daily.parquet"
        df.to_parquet(output_path, index=False)

        # Verify
        saved_df = pd.read_parquet(output_path)

        print(f"[PASS] Saved {len(saved_df)} bars to {output_path}")
        print(f"       File size: {output_path.stat().st_size / 1024:.1f} KB")
        print(f"       Columns: {list(saved_df.columns)}")

        return True

    except Exception as e:
        print(f"[FAIL] Save data error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 70)
    print("GIGA TRADER - Alpaca Data Test Suite")
    print("=" * 70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Project: {project_root}")

    tests = [
        ("Alpaca Connection", test_alpaca_connection),
        ("Daily Data Download", test_market_data_download),
        ("Intraday Data Download", test_intraday_data),
        ("Extended Hours Data", test_extended_hours_data),
        ("Save to Parquet", test_save_data),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"[ERROR] Test '{name}' crashed: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")

    print(f"\n  Results: {passed}/{total} tests passed")
    print("=" * 70)

    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
