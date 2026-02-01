"""
GIGA TRADER - Optimized Dual Model with Grid Search
====================================================
- Downloads MAXIMUM historical 1-minute data
- Adds comprehensive intraday technical features
- Grid searches swing thresholds dynamically
- Trains optimized dual models

Usage:
    python src/train_optimized_model.py
"""

import os
import sys
from datetime import datetime, timedelta, time
from pathlib import Path
from itertools import product
import warnings

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(project_root / ".env")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: DOWNLOAD MAXIMUM HISTORICAL DATA
# ═══════════════════════════════════════════════════════════════════════════════
def download_max_historical_data() -> pd.DataFrame:
    """
    Download maximum available 1-minute SPY data from Alpaca.
    Alpaca free tier: ~2 years of 1-min data for stocks.
    Downloads in chunks to avoid timeouts.
    """
    print("\n" + "=" * 70)
    print("STEP 1: DOWNLOAD MAXIMUM HISTORICAL 1-MINUTE DATA")
    print("=" * 70)

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    client = StockHistoricalDataClient(api_key, secret_key)

    # Try to get as much data as possible (download in chunks)
    end_date = datetime.now()
    all_data = []

    # Download in 30-day chunks to avoid timeouts
    chunk_days = 30
    total_days_requested = 365  # Try for 1 year first

    print(f"[INFO] Requesting {total_days_requested} days of 1-minute data in {chunk_days}-day chunks...")

    current_end = end_date
    days_downloaded = 0

    while days_downloaded < total_days_requested:
        chunk_start = current_end - timedelta(days=chunk_days)

        try:
            request = StockBarsRequest(
                symbol_or_symbols="SPY",
                timeframe=TimeFrame.Minute,
                start=chunk_start,
                end=current_end
            )

            bars = client.get_stock_bars(request)
            df_chunk = bars.df.reset_index()

            if len(df_chunk) > 0:
                all_data.append(df_chunk)
                print(f"       Downloaded {len(df_chunk):,} bars from {chunk_start.date()} to {current_end.date()}")

            current_end = chunk_start
            days_downloaded += chunk_days

        except Exception as e:
            print(f"[WARN] Error downloading chunk: {e}")
            break

    if not all_data:
        raise ValueError("No data downloaded!")

    # Combine all chunks
    df = pd.concat(all_data, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Convert to Eastern Time
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date"] = df["timestamp"].dt.date
    df["time"] = df["timestamp"].dt.time
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute

    # Classify session
    def get_session(row):
        h, m = row["hour"], row["minute"]
        if h < 9 or (h == 9 and m < 30):
            return "premarket"
        elif h < 16:
            return "regular"
        else:
            return "afterhours"

    df["session"] = df.apply(get_session, axis=1)

    print(f"\n[PASS] Downloaded {len(df):,} total 1-minute bars")
    print(f"       Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"       Trading days: {df['date'].nunique()}")
    print(f"\n       Session breakdown:")
    for session in ["premarket", "regular", "afterhours"]:
        count = len(df[df["session"] == session])
        print(f"       - {session}: {count:,} bars ({100*count/len(df):.1f}%)")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: COMPUTE INTRADAY TECHNICAL INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════
def compute_intraday_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators computed on 1-minute data.
    These will be aggregated per day later.
    """
    print("\n" + "=" * 70)
    print("STEP 2: COMPUTE INTRADAY TECHNICAL INDICATORS")
    print("=" * 70)

    df = df.copy()

    # RSI (14-period on 1-min bars)
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # MACD
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_histogram"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands (20-period)
    df["bb_mid"] = df["close"].rolling(20).mean()
    df["bb_std"] = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * df["bb_std"]
    df["bb_lower"] = df["bb_mid"] - 2 * df["bb_std"]
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)

    # Stochastic
    low_14 = df["low"].rolling(14).min()
    high_14 = df["high"].rolling(14).max()
    df["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14 + 1e-10)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # ATR
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1))
        )
    )
    df["atr_14"] = df["tr"].rolling(14).mean()

    # Volume indicators
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_sma_20"] + 1)

    # Momentum
    df["momentum_5"] = df["close"].pct_change(5)
    df["momentum_15"] = df["close"].pct_change(15)

    print(f"[PASS] Added technical indicators: RSI, MACD, Bollinger Bands, Stochastic, ATR")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: BUILD DAILY FEATURES WITH TECHNICALS
# ═══════════════════════════════════════════════════════════════════════════════
def build_daily_features(df: pd.DataFrame, swing_threshold: float = 0.003) -> pd.DataFrame:
    """
    Build daily features including extended hours and technical indicators.

    Args:
        swing_threshold: Threshold for detecting swings (default 0.3%)
    """
    trading_days = sorted(df["date"].unique())
    daily_records = []

    for i, day in enumerate(trading_days):
        day_data = df[df["date"] == day].sort_values("timestamp")
        prev_day = trading_days[i - 1] if i > 0 else None
        prev_day_data = df[df["date"] == prev_day] if prev_day else None

        # ─────────────────────────────────────────────────────────────────────
        # PREMARKET FEATURES
        # ─────────────────────────────────────────────────────────────────────
        pm_data = day_data[day_data["session"] == "premarket"]

        if len(pm_data) >= 5:
            pm_open = pm_data.iloc[0]["open"]
            pm_close = pm_data.iloc[-1]["close"]
            pm_high = pm_data["high"].max()
            pm_low = pm_data["low"].min()
            pm_volume = pm_data["volume"].sum()

            pm_return = (pm_close - pm_open) / pm_open if pm_open > 0 else 0
            pm_range = (pm_high - pm_low) / pm_open if pm_open > 0 else 0
            pm_direction = 1 if pm_return > 0.001 else (-1 if pm_return < -0.001 else 0)

            # Technical indicators at end of premarket
            pm_rsi = pm_data.iloc[-1]["rsi_14"] if "rsi_14" in pm_data.columns else 50
            pm_macd_hist = pm_data.iloc[-1]["macd_histogram"] if "macd_histogram" in pm_data.columns else 0
            pm_bb_position = pm_data.iloc[-1]["bb_position"] if "bb_position" in pm_data.columns else 0.5
            pm_stoch_k = pm_data.iloc[-1]["stoch_k"] if "stoch_k" in pm_data.columns else 50

            # Last 30/60 min momentum
            pm_last_30 = pm_data[pm_data["timestamp"] >= pm_data["timestamp"].max() - timedelta(minutes=30)]
            pm_momentum_30 = (pm_last_30.iloc[-1]["close"] - pm_last_30.iloc[0]["open"]) / pm_last_30.iloc[0]["open"] if len(pm_last_30) > 1 else 0
        else:
            pm_return = pm_range = pm_direction = 0
            pm_volume = 0
            pm_rsi = 50
            pm_macd_hist = 0
            pm_bb_position = 0.5
            pm_stoch_k = 50
            pm_momentum_30 = 0
            pm_close = None

        # ─────────────────────────────────────────────────────────────────────
        # AFTERHOURS FEATURES (previous day)
        # ─────────────────────────────────────────────────────────────────────
        if prev_day_data is not None:
            ah_data = prev_day_data[prev_day_data["session"] == "afterhours"]

            if len(ah_data) >= 5:
                ah_open = ah_data.iloc[0]["open"]
                ah_close = ah_data.iloc[-1]["close"]
                ah_high = ah_data["high"].max()
                ah_low = ah_data["low"].min()

                ah_return = (ah_close - ah_open) / ah_open if ah_open > 0 else 0
                ah_range = (ah_high - ah_low) / ah_open if ah_open > 0 else 0
                ah_direction = 1 if ah_return > 0.001 else (-1 if ah_return < -0.001 else 0)
            else:
                ah_return = ah_range = ah_direction = 0
                ah_close = None

            prev_regular = prev_day_data[prev_day_data["session"] == "regular"]
            prev_close = prev_regular.iloc[-1]["close"] if len(prev_regular) > 0 else None
        else:
            ah_return = ah_range = ah_direction = 0
            prev_close = None
            ah_close = None

        # Overnight return
        if pm_close is not None and prev_close is not None:
            overnight_return = (pm_close - prev_close) / prev_close
        else:
            overnight_return = 0

        # ─────────────────────────────────────────────────────────────────────
        # REGULAR SESSION
        # ─────────────────────────────────────────────────────────────────────
        regular_data = day_data[day_data["session"] == "regular"]

        if len(regular_data) < 50:
            continue

        reg_open = regular_data.iloc[0]["open"]
        reg_close = regular_data.iloc[-1]["close"]
        reg_high = regular_data["high"].max()
        reg_low = regular_data["low"].min()

        day_return = (reg_close - reg_open) / reg_open
        day_range = (reg_high - reg_low) / reg_open

        # Technical indicators at key times
        # At 10:15
        morning_data = regular_data[regular_data["time"] <= time(10, 15)]
        if len(morning_data) > 0:
            rsi_at_1015 = morning_data.iloc[-1]["rsi_14"] if "rsi_14" in morning_data.columns else 50
            macd_at_1015 = morning_data.iloc[-1]["macd_histogram"] if "macd_histogram" in morning_data.columns else 0
            bb_at_1015 = morning_data.iloc[-1]["bb_position"] if "bb_position" in morning_data.columns else 0.5
            stoch_at_1015 = morning_data.iloc[-1]["stoch_k"] if "stoch_k" in morning_data.columns else 50
            return_at_1015 = (morning_data.iloc[-1]["close"] - reg_open) / reg_open
            morning_high = morning_data["high"].max()
            morning_low = morning_data["low"].min()
        else:
            rsi_at_1015 = 50
            macd_at_1015 = 0
            bb_at_1015 = 0.5
            stoch_at_1015 = 50
            return_at_1015 = 0
            morning_high = reg_open
            morning_low = reg_open

        # At 12:30
        midday_data = regular_data[regular_data["time"] <= time(12, 30)]
        if len(midday_data) > 0:
            rsi_at_1230 = midday_data.iloc[-1]["rsi_14"] if "rsi_14" in midday_data.columns else 50
            macd_at_1230 = midday_data.iloc[-1]["macd_histogram"] if "macd_histogram" in midday_data.columns else 0
            bb_at_1230 = midday_data.iloc[-1]["bb_position"] if "bb_position" in midday_data.columns else 0.5
            return_at_1230 = (midday_data.iloc[-1]["close"] - reg_open) / reg_open
            low_to_1230 = midday_data["low"].min()
            high_to_1230 = midday_data["high"].max()
            return_from_low = (midday_data.iloc[-1]["close"] - low_to_1230) / low_to_1230 if low_to_1230 > 0 else 0
        else:
            rsi_at_1230 = 50
            macd_at_1230 = 0
            bb_at_1230 = 0.5
            return_at_1230 = 0
            low_to_1230 = reg_low
            high_to_1230 = reg_high
            return_from_low = 0

        # ─────────────────────────────────────────────────────────────────────
        # SWING DETECTION (dynamic threshold)
        # ─────────────────────────────────────────────────────────────────────
        morning_swing_up = morning_high >= reg_open * (1 + swing_threshold)
        morning_swing_down = morning_low <= reg_open * (1 - swing_threshold)
        has_morning_swing = morning_swing_up or morning_swing_down

        is_up_day = day_return > swing_threshold
        is_down_day = day_return < -swing_threshold

        # ─────────────────────────────────────────────────────────────────────
        # TIMING (when did high/low occur)
        # ─────────────────────────────────────────────────────────────────────
        high_idx = regular_data["high"].idxmax()
        low_idx = regular_data["low"].idxmin()
        high_time = regular_data.loc[high_idx, "timestamp"]
        low_time = regular_data.loc[low_idx, "timestamp"]
        open_time = regular_data.iloc[0]["timestamp"]

        high_minutes = (high_time - open_time).total_seconds() / 60
        low_minutes = (low_time - open_time).total_seconds() / 60
        low_before_high = low_time < high_time

        # Max gains from entry points
        post_1015 = regular_data[regular_data["time"] >= time(10, 15)]
        max_gain_from_1015 = (post_1015["high"].max() - morning_data.iloc[-1]["close"]) / morning_data.iloc[-1]["close"] if len(morning_data) > 0 and len(post_1015) > 0 else 0

        post_1230 = regular_data[regular_data["time"] >= time(12, 30)]
        max_gain_from_1230 = (post_1230["high"].max() - midday_data.iloc[-1]["close"]) / midday_data.iloc[-1]["close"] if len(midday_data) > 0 and len(post_1230) > 0 else 0

        daily_records.append({
            "date": day,

            # Premarket
            "pm_return": pm_return,
            "pm_range": pm_range,
            "pm_direction": pm_direction,
            "pm_momentum_30": pm_momentum_30,
            "pm_rsi": pm_rsi,
            "pm_macd_hist": pm_macd_hist,
            "pm_bb_position": pm_bb_position,
            "pm_stoch_k": pm_stoch_k,

            # Afterhours
            "ah_return_lag1": ah_return,
            "ah_range_lag1": ah_range,
            "ah_direction_lag1": ah_direction,

            # Overnight
            "overnight_return": overnight_return,
            "direction_agreement": 1 if pm_direction == ah_direction and pm_direction != 0 else 0,

            # Regular session
            "day_return": day_return,
            "day_range": day_range,

            # At 10:15
            "return_at_1015": return_at_1015,
            "rsi_at_1015": rsi_at_1015,
            "macd_at_1015": macd_at_1015,
            "bb_at_1015": bb_at_1015,
            "stoch_at_1015": stoch_at_1015,

            # At 12:30
            "return_at_1230": return_at_1230,
            "return_from_low": return_from_low,
            "rsi_at_1230": rsi_at_1230,
            "macd_at_1230": macd_at_1230,
            "bb_at_1230": bb_at_1230,

            # Swings
            "has_morning_swing": has_morning_swing,
            "is_up_day": is_up_day,
            "is_down_day": is_down_day,

            # Timing
            "high_minutes": high_minutes,
            "low_minutes": low_minutes,
            "low_before_high": low_before_high,

            # Outcomes
            "max_gain_from_1015": max_gain_from_1015,
            "max_gain_from_1230": max_gain_from_1230,
        })

    result_df = pd.DataFrame(daily_records)
    result_df["date"] = pd.to_datetime(result_df["date"])

    return result_df


def add_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged and rolling features."""
    df = df.sort_values("date").copy()

    for lag in [1, 2, 3, 5]:
        df[f"day_return_lag{lag}"] = df["day_return"].shift(lag)
        df[f"day_range_lag{lag}"] = df["day_range"].shift(lag)
        df[f"pm_return_lag{lag}"] = df["pm_return"].shift(lag)

    df["day_return_ma5"] = df["day_return"].shift(1).rolling(5).mean()
    df["volatility_5d"] = df["day_return"].shift(1).rolling(5).std()
    df["pm_return_ma3"] = df["pm_return"].shift(1).rolling(3).mean()

    df["consecutive_up_days"] = (df["is_up_day"].shift(1).rolling(3).sum())
    df["consecutive_down_days"] = (df["is_down_day"].shift(1).rolling(3).sum())

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: GRID SEARCH FOR OPTIMAL THRESHOLDS
# ═══════════════════════════════════════════════════════════════════════════════
def grid_search_thresholds(df_1min: pd.DataFrame):
    """
    Grid search over swing thresholds to find optimal parameters.

    Thresholds to search:
      - Swing threshold: 0.2%, 0.25%, 0.3%, 0.35%, 0.4%, 0.5%
    """
    print("\n" + "=" * 70)
    print("STEP 4: GRID SEARCH FOR OPTIMAL THRESHOLDS")
    print("=" * 70)

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    # Thresholds to test
    swing_thresholds = [0.002, 0.0025, 0.003, 0.0035, 0.004, 0.005]

    results = []

    print(f"[INFO] Testing {len(swing_thresholds)} swing thresholds...")
    print(f"       Thresholds: {[f'{t*100:.2f}%' for t in swing_thresholds]}")

    for threshold in swing_thresholds:
        print(f"\n[TESTING] Swing threshold = {threshold*100:.2f}%")

        # Build features with this threshold
        df_daily = build_daily_features(df_1min, swing_threshold=threshold)
        df_daily = add_lagged_features(df_daily)

        # Feature columns
        feature_cols = [
            "pm_return", "pm_range", "pm_direction", "pm_momentum_30",
            "pm_rsi", "pm_macd_hist", "pm_bb_position", "pm_stoch_k",
            "ah_return_lag1", "ah_range_lag1", "ah_direction_lag1",
            "overnight_return", "direction_agreement",
            "return_at_1015", "rsi_at_1015", "macd_at_1015", "bb_at_1015", "stoch_at_1015",
            "return_at_1230", "return_from_low", "rsi_at_1230", "macd_at_1230", "bb_at_1230",
            "day_return_lag1", "day_return_lag2", "day_return_lag3",
            "day_range_lag1", "day_range_lag2",
            "pm_return_lag1", "pm_return_lag2",
            "day_return_ma5", "volatility_5d", "pm_return_ma3",
            "consecutive_up_days", "consecutive_down_days"
        ]

        feature_cols = [c for c in feature_cols if c in df_daily.columns]

        # Clean data
        df_clean = df_daily.dropna(subset=feature_cols + ["is_up_day", "low_before_high"]).copy()

        if len(df_clean) < 50:
            print(f"         [SKIP] Not enough data ({len(df_clean)} samples)")
            continue

        # Split
        split_idx = int(len(df_clean) * 0.8)
        train_df = df_clean.iloc[:split_idx - 3]
        test_df = df_clean.iloc[split_idx:]

        if len(train_df) < 20 or len(test_df) < 5:
            print(f"         [SKIP] Not enough train/test data")
            continue

        X_train = train_df[feature_cols].values
        X_test = test_df[feature_cols].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Model 1: Swing direction
        y_train_swing = train_df["is_up_day"].astype(int).values
        y_test_swing = test_df["is_up_day"].astype(int).values

        model_swing = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)
        model_swing.fit(X_train_scaled, y_train_swing)

        y_proba_swing = model_swing.predict_proba(X_test_scaled)[:, 1]
        auc_swing = roc_auc_score(y_test_swing, y_proba_swing) if len(np.unique(y_test_swing)) > 1 else 0.5

        # Model 2: Timing
        y_train_timing = train_df["low_before_high"].astype(int).values
        y_test_timing = test_df["low_before_high"].astype(int).values

        model_timing = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)
        model_timing.fit(X_train_scaled, y_train_timing)

        y_proba_timing = model_timing.predict_proba(X_test_scaled)[:, 1]
        auc_timing = roc_auc_score(y_test_timing, y_proba_timing) if len(np.unique(y_test_timing)) > 1 else 0.5

        # Combined AUC
        combined_auc = (auc_swing + auc_timing) / 2

        # Signal analysis
        swing_pred = (y_proba_swing > 0.5).astype(int)
        timing_pred = (y_proba_timing > 0.5).astype(int)

        buy_signals = (swing_pred == 1) & (timing_pred == 1)
        sell_signals = (swing_pred == 0) & (timing_pred == 0)

        buy_win_rate = (test_df["day_return"].values[buy_signals] > 0).mean() if buy_signals.sum() > 0 else 0
        sell_win_rate = (test_df["day_return"].values[sell_signals] < 0).mean() if sell_signals.sum() > 0 else 0

        print(f"         Swing AUC: {auc_swing:.3f}, Timing AUC: {auc_timing:.3f}")
        print(f"         Buy signals: {buy_signals.sum()}, Win rate: {buy_win_rate*100:.1f}%")
        print(f"         Sell signals: {sell_signals.sum()}, Win rate: {sell_win_rate*100:.1f}%")

        results.append({
            "threshold": threshold,
            "auc_swing": auc_swing,
            "auc_timing": auc_timing,
            "combined_auc": combined_auc,
            "buy_signals": buy_signals.sum(),
            "buy_win_rate": buy_win_rate,
            "sell_signals": sell_signals.sum(),
            "sell_win_rate": sell_win_rate,
            "train_samples": len(train_df),
            "test_samples": len(test_df)
        })

    results_df = pd.DataFrame(results)

    # Find best threshold
    if len(results_df) > 0:
        # Score by combined AUC + win rates
        results_df["score"] = results_df["combined_auc"] + 0.3 * (results_df["buy_win_rate"] + results_df["sell_win_rate"])
        best_idx = results_df["score"].idxmax()
        best_threshold = results_df.loc[best_idx, "threshold"]

        print(f"\n[BEST THRESHOLD] {best_threshold*100:.2f}%")
        print(f"  Swing AUC: {results_df.loc[best_idx, 'auc_swing']:.3f}")
        print(f"  Timing AUC: {results_df.loc[best_idx, 'auc_timing']:.3f}")
        print(f"  Buy win rate: {results_df.loc[best_idx, 'buy_win_rate']*100:.1f}%")
        print(f"  Sell win rate: {results_df.loc[best_idx, 'sell_win_rate']*100:.1f}%")
    else:
        best_threshold = 0.003
        print(f"\n[DEFAULT] Using threshold {best_threshold*100:.2f}%")

    return results_df, best_threshold


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: TRAIN FINAL MODELS WITH BEST THRESHOLD
# ═══════════════════════════════════════════════════════════════════════════════
def train_final_models(df_1min: pd.DataFrame, best_threshold: float):
    """Train final models with the optimal threshold."""
    print("\n" + "=" * 70)
    print(f"STEP 5: TRAIN FINAL MODELS (threshold = {best_threshold*100:.2f}%)")
    print("=" * 70)

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report

    # Build features
    print("[INFO] Building features with optimal threshold...")
    df_daily = build_daily_features(df_1min, swing_threshold=best_threshold)
    df_daily = add_lagged_features(df_daily)

    print(f"[INFO] Built features for {len(df_daily)} trading days")

    # Feature columns
    feature_cols = [
        "pm_return", "pm_range", "pm_direction", "pm_momentum_30",
        "pm_rsi", "pm_macd_hist", "pm_bb_position", "pm_stoch_k",
        "ah_return_lag1", "ah_range_lag1", "ah_direction_lag1",
        "overnight_return", "direction_agreement",
        "return_at_1015", "rsi_at_1015", "macd_at_1015", "bb_at_1015", "stoch_at_1015",
        "return_at_1230", "return_from_low", "rsi_at_1230", "macd_at_1230", "bb_at_1230",
        "day_return_lag1", "day_return_lag2", "day_return_lag3",
        "day_range_lag1", "day_range_lag2",
        "pm_return_lag1", "pm_return_lag2",
        "day_return_ma5", "volatility_5d", "pm_return_ma3",
        "consecutive_up_days", "consecutive_down_days"
    ]

    feature_cols = [c for c in feature_cols if c in df_daily.columns]
    print(f"[INFO] Using {len(feature_cols)} features")

    df_clean = df_daily.dropna(subset=feature_cols + ["is_up_day", "low_before_high"]).copy()

    # Split
    split_idx = int(len(df_clean) * 0.8)
    train_df = df_clean.iloc[:split_idx - 3]
    test_df = df_clean.iloc[split_idx:]

    print(f"[INFO] Train: {len(train_df)} days, Test: {len(test_df)} days")

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {}
    results = {}

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL 1: SWING DIRECTION (L2 + Gradient Boosting ensemble)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[MODEL 1: SWING DIRECTION]")

    y_train = train_df["is_up_day"].astype(int).values
    y_test = test_df["is_up_day"].astype(int).values

    # L2 Logistic Regression
    model_l2 = LogisticRegression(penalty="l2", C=0.5, max_iter=1000, random_state=42)
    model_l2.fit(X_train_scaled, y_train)

    # Shallow Gradient Boosting
    model_gb = GradientBoostingClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    model_gb.fit(X_train_scaled, y_train)

    # Ensemble predictions (average)
    y_proba_l2 = model_l2.predict_proba(X_test_scaled)[:, 1]
    y_proba_gb = model_gb.predict_proba(X_test_scaled)[:, 1]
    y_proba_swing = (y_proba_l2 + y_proba_gb) / 2
    y_pred_swing = (y_proba_swing > 0.5).astype(int)

    auc_swing = roc_auc_score(y_test, y_proba_swing) if len(np.unique(y_test)) > 1 else 0.5
    acc_swing = accuracy_score(y_test, y_pred_swing)

    print(f"  AUC: {auc_swing:.3f}, Accuracy: {acc_swing:.3f}")
    print(classification_report(y_test, y_pred_swing, target_names=["Down/Flat", "Up"]))

    models["swing"] = {"l2": model_l2, "gb": model_gb}
    results["swing"] = {"auc": auc_swing, "accuracy": acc_swing}

    # Top features from GB
    importances = model_gb.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    print("  Top 10 features (Gradient Boosting):")
    for i, idx in enumerate(top_idx):
        print(f"    {i+1}. {feature_cols[idx]}: {importances[idx]:.3f}")

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL 2: ENTRY/EXIT TIMING
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[MODEL 2: ENTRY/EXIT TIMING]")

    y_train = train_df["low_before_high"].astype(int).values
    y_test = test_df["low_before_high"].astype(int).values

    model_timing_l2 = LogisticRegression(penalty="l2", C=0.5, max_iter=1000, random_state=42)
    model_timing_l2.fit(X_train_scaled, y_train)

    model_timing_gb = GradientBoostingClassifier(
        n_estimators=50, max_depth=3, learning_rate=0.1,
        subsample=0.8, random_state=42
    )
    model_timing_gb.fit(X_train_scaled, y_train)

    y_proba_timing_l2 = model_timing_l2.predict_proba(X_test_scaled)[:, 1]
    y_proba_timing_gb = model_timing_gb.predict_proba(X_test_scaled)[:, 1]
    y_proba_timing = (y_proba_timing_l2 + y_proba_timing_gb) / 2
    y_pred_timing = (y_proba_timing > 0.5).astype(int)

    auc_timing = roc_auc_score(y_test, y_proba_timing) if len(np.unique(y_test)) > 1 else 0.5
    acc_timing = accuracy_score(y_test, y_pred_timing)

    print(f"  AUC: {auc_timing:.3f}, Accuracy: {acc_timing:.3f}")
    print(classification_report(y_test, y_pred_timing, target_names=["High First", "Low First"]))

    models["timing"] = {"l2": model_timing_l2, "gb": model_timing_gb}
    results["timing"] = {"auc": auc_timing, "accuracy": acc_timing}

    # ─────────────────────────────────────────────────────────────────────────
    # COMBINED SIGNALS
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[COMBINED SIGNAL ANALYSIS]")

    buy_signals = (y_pred_swing == 1) & (y_pred_timing == 1)
    sell_signals = (y_pred_swing == 0) & (y_pred_timing == 0)

    test_returns = test_df["day_return"].values

    print(f"  Total test days: {len(test_df)}")
    print(f"  Buy signals:  {buy_signals.sum()} ({100*buy_signals.mean():.1f}%)")
    print(f"  Sell signals: {sell_signals.sum()} ({100*sell_signals.mean():.1f}%)")

    if buy_signals.sum() > 0:
        buy_returns = test_returns[buy_signals]
        print(f"\n  BUY PERFORMANCE:")
        print(f"    Win rate: {100*(buy_returns > 0).mean():.1f}%")
        print(f"    Avg return: {100*buy_returns.mean():.2f}%")
        print(f"    Total return: {100*buy_returns.sum():.2f}%")

    if sell_signals.sum() > 0:
        sell_returns = test_returns[sell_signals]
        print(f"\n  SELL PERFORMANCE (shorting would profit from negative returns):")
        print(f"    Win rate: {100*(sell_returns < 0).mean():.1f}%")
        print(f"    Avg return: {100*sell_returns.mean():.2f}%")

    return models, results, scaler, feature_cols, df_daily, test_df, y_proba_swing, y_proba_timing


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: SAVE MODELS
# ═══════════════════════════════════════════════════════════════════════════════
def save_models(models, results, scaler, feature_cols, best_threshold, grid_results):
    """Save all models and results."""
    print("\n" + "=" * 70)
    print("STEP 6: SAVE MODELS")
    print("=" * 70)

    import joblib

    models_dir = project_root / "models" / "production"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "spy_optimized_models.joblib"
    joblib.dump({
        "models": models,
        "results": results,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "best_threshold": best_threshold,
        "grid_search_results": grid_results.to_dict() if grid_results is not None else None,
        "trained_at": datetime.now().isoformat(),
        "description": "Optimized dual model with grid-searched thresholds"
    }, model_path)

    print(f"[PASS] Saved to {model_path}")
    return model_path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 70)
    print("GIGA TRADER - OPTIMIZED MODEL TRAINING")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Features:")
    print("  - Maximum historical 1-minute data")
    print("  - Extended hours (premarket + afterhours)")
    print("  - Technical indicators (RSI, MACD, Bollinger, Stochastic)")
    print("  - Grid search for optimal swing thresholds")
    print("  - Ensemble models (L2 + Gradient Boosting)")

    # Step 1: Download data
    df_1min = download_max_historical_data()

    # Step 2: Compute technicals
    df_1min = compute_intraday_technicals(df_1min)

    # Step 3-4: Grid search
    grid_results, best_threshold = grid_search_thresholds(df_1min)

    # Step 5: Train final models
    models, results, scaler, feature_cols, df_daily, test_df, y_proba_swing, y_proba_timing = \
        train_final_models(df_1min, best_threshold)

    # Step 6: Save
    model_path = save_models(models, results, scaler, feature_cols, best_threshold, grid_results)

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nBest swing threshold: {best_threshold*100:.2f}%")
    print(f"Swing Model AUC: {results['swing']['auc']:.3f}")
    print(f"Timing Model AUC: {results['timing']['auc']:.3f}")
    print(f"\nModels saved to: {model_path}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
