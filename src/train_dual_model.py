"""
GIGA TRADER - Dual Model Training System
=========================================
TWO MODELS THAT MUST AGREE:
  1. SWING DIRECTION MODEL: Predicts if price will swing up/down in time window
  2. ENTRY/EXIT TIMING MODEL: Predicts intraday tops and bottoms

DATA REQUIREMENTS:
  - Always 1-minute bars
  - Include premarket (4:00-9:30 ET) and afterhours (16:00-20:00 ET)
  - Extract EDGE 2 features from extended hours

Usage:
    python src/train_dual_model.py
"""

import os
import sys
from datetime import datetime, timedelta, time
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv(project_root / ".env")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: DOWNLOAD 1-MINUTE DATA WITH EXTENDED HOURS
# ═══════════════════════════════════════════════════════════════════════════════
def download_1min_extended_hours(days: int = 60) -> pd.DataFrame:
    """
    Download 1-minute SPY data INCLUDING premarket and afterhours.
    Alpaca provides extended hours data by default.
    """
    print("\n" + "=" * 70)
    print("STEP 1: DOWNLOAD 1-MINUTE DATA (Premarket + Regular + Afterhours)")
    print("=" * 70)

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    client = StockHistoricalDataClient(api_key, secret_key)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    print(f"[INFO] Requesting {days} days of 1-minute data...")
    print(f"       Date range: {start_date.date()} to {end_date.date()}")
    print(f"       Including: Premarket (4:00-9:30) + Regular (9:30-16:00) + Afterhours (16:00-20:00)")

    request = StockBarsRequest(
        symbol_or_symbols="SPY",
        timeframe=TimeFrame.Minute,
        start=start_date,
        end=end_date
    )

    bars = client.get_stock_bars(request)
    df = bars.df.reset_index()

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

    print(f"\n[PASS] Downloaded {len(df):,} 1-minute bars")
    print(f"       Trading days: {df['date'].nunique()}")
    print(f"\n       Session breakdown:")
    for session in ["premarket", "regular", "afterhours"]:
        count = len(df[df["session"] == session])
        print(f"       - {session}: {count:,} bars ({100*count/len(df):.1f}%)")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: BUILD EXTENDED HOURS FEATURES (EDGE 2)
# ═══════════════════════════════════════════════════════════════════════════════
def build_extended_hours_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build features from premarket and afterhours data per EDGE 2.

    For each trading day, extract:
      - Premarket features (today's PM before regular session)
      - Afterhours features (previous day's AH)
      - Combined overnight features
    """
    print("\n" + "=" * 70)
    print("STEP 2: BUILD EXTENDED HOURS FEATURES (EDGE 2)")
    print("=" * 70)

    trading_days = sorted(df["date"].unique())
    print(f"[INFO] Processing {len(trading_days)} trading days...")

    daily_records = []

    for i, day in enumerate(trading_days):
        day_data = df[df["date"] == day].sort_values("timestamp")

        # Get previous trading day
        prev_day = trading_days[i - 1] if i > 0 else None
        prev_day_data = df[df["date"] == prev_day] if prev_day else None

        # ─────────────────────────────────────────────────────────────────────
        # PREMARKET FEATURES (today, 4:00-9:30)
        # ─────────────────────────────────────────────────────────────────────
        pm_data = day_data[day_data["session"] == "premarket"]

        if len(pm_data) >= 5:
            pm_open = pm_data.iloc[0]["open"]
            pm_close = pm_data.iloc[-1]["close"]
            pm_high = pm_data["high"].max()
            pm_low = pm_data["low"].min()
            pm_volume = pm_data["volume"].sum()
            pm_vwap = (pm_data["close"] * pm_data["volume"]).sum() / pm_volume if pm_volume > 0 else pm_close

            pm_return = (pm_close - pm_open) / pm_open if pm_open > 0 else 0
            pm_range = (pm_high - pm_low) / pm_open if pm_open > 0 else 0
            pm_direction = 1 if pm_return > 0.001 else (-1 if pm_return < -0.001 else 0)
            pm_vwap_deviation = (pm_close - pm_vwap) / pm_vwap if pm_vwap > 0 else 0

            # Last 30/60 min momentum
            pm_last_30 = pm_data[pm_data["timestamp"] >= pm_data["timestamp"].max() - timedelta(minutes=30)]
            pm_last_60 = pm_data[pm_data["timestamp"] >= pm_data["timestamp"].max() - timedelta(minutes=60)]

            pm_momentum_30 = (pm_last_30.iloc[-1]["close"] - pm_last_30.iloc[0]["open"]) / pm_last_30.iloc[0]["open"] if len(pm_last_30) > 0 else 0
            pm_momentum_60 = (pm_last_60.iloc[-1]["close"] - pm_last_60.iloc[0]["open"]) / pm_last_60.iloc[0]["open"] if len(pm_last_60) > 0 else 0
        else:
            pm_return = pm_range = pm_direction = pm_vwap_deviation = 0
            pm_momentum_30 = pm_momentum_60 = 0
            pm_volume = 0
            pm_close = None

        # ─────────────────────────────────────────────────────────────────────
        # AFTERHOURS FEATURES (previous day, 16:00-20:00)
        # ─────────────────────────────────────────────────────────────────────
        if prev_day_data is not None:
            ah_data = prev_day_data[prev_day_data["session"] == "afterhours"]

            if len(ah_data) >= 5:
                ah_open = ah_data.iloc[0]["open"]
                ah_close = ah_data.iloc[-1]["close"]
                ah_high = ah_data["high"].max()
                ah_low = ah_data["low"].min()
                ah_volume = ah_data["volume"].sum()

                ah_return = (ah_close - ah_open) / ah_open if ah_open > 0 else 0
                ah_range = (ah_high - ah_low) / ah_open if ah_open > 0 else 0
                ah_direction = 1 if ah_return > 0.001 else (-1 if ah_return < -0.001 else 0)
            else:
                ah_return = ah_range = ah_direction = 0
                ah_volume = 0
                ah_close = None
        else:
            ah_return = ah_range = ah_direction = 0
            ah_volume = 0
            ah_close = None

        # ─────────────────────────────────────────────────────────────────────
        # OVERNIGHT COMBINED FEATURES
        # ─────────────────────────────────────────────────────────────────────
        # Get previous day's close
        if prev_day_data is not None:
            prev_regular = prev_day_data[prev_day_data["session"] == "regular"]
            prev_close = prev_regular.iloc[-1]["close"] if len(prev_regular) > 0 else None
        else:
            prev_close = None

        if pm_close is not None and prev_close is not None:
            overnight_return = (pm_close - prev_close) / prev_close
        else:
            overnight_return = 0

        # Direction agreement (PM and AH same direction = stronger signal)
        direction_agreement = 1 if pm_direction == ah_direction and pm_direction != 0 else 0

        # ─────────────────────────────────────────────────────────────────────
        # REGULAR SESSION DATA
        # ─────────────────────────────────────────────────────────────────────
        regular_data = day_data[day_data["session"] == "regular"]

        if len(regular_data) < 50:
            continue

        reg_open = regular_data.iloc[0]["open"]
        reg_close = regular_data.iloc[-1]["close"]
        reg_high = regular_data["high"].max()
        reg_low = regular_data["low"].min()
        reg_volume = regular_data["volume"].sum()

        day_return = (reg_close - reg_open) / reg_open
        day_range = (reg_high - reg_low) / reg_open

        # ─────────────────────────────────────────────────────────────────────
        # TIME WINDOW ANALYSIS (for swing detection)
        # ─────────────────────────────────────────────────────────────────────
        # Morning window (9:30 - 12:30)
        morning_data = regular_data[(regular_data["hour"] < 12) | ((regular_data["hour"] == 12) & (regular_data["minute"] <= 30))]
        # Afternoon window (12:30 - 16:00)
        afternoon_data = regular_data[(regular_data["hour"] >= 12) & ~((regular_data["hour"] == 12) & (regular_data["minute"] <= 30))]

        if len(morning_data) > 0:
            morning_high = morning_data["high"].max()
            morning_low = morning_data["low"].min()
            morning_close = morning_data.iloc[-1]["close"]
            morning_return = (morning_close - reg_open) / reg_open
            morning_range = (morning_high - morning_low) / reg_open
        else:
            morning_return = morning_range = 0
            morning_high = morning_low = reg_open

        if len(afternoon_data) > 0:
            afternoon_high = afternoon_data["high"].max()
            afternoon_low = afternoon_data["low"].min()
            afternoon_open = afternoon_data.iloc[0]["open"]
            afternoon_close = afternoon_data.iloc[-1]["close"]
            afternoon_return = (afternoon_close - afternoon_open) / afternoon_open
            afternoon_range = (afternoon_high - afternoon_low) / afternoon_open
        else:
            afternoon_return = afternoon_range = 0
            afternoon_high = afternoon_low = reg_open

        # ─────────────────────────────────────────────────────────────────────
        # SWING DETECTION (targets)
        # ─────────────────────────────────────────────────────────────────────
        # Morning swing: Did price move 0.3%+ from open in morning window?
        morning_swing_up = morning_high >= reg_open * 1.003
        morning_swing_down = morning_low <= reg_open * 0.997
        has_morning_swing = morning_swing_up or morning_swing_down

        # Afternoon swing: Did price swing 0.3%+ in afternoon?
        afternoon_swing_up = afternoon_high >= afternoon_low * 1.003
        afternoon_swing_down = afternoon_low <= afternoon_high * 0.997

        # Overall day direction
        is_up_day = day_return > 0.002
        is_down_day = day_return < -0.002

        # ─────────────────────────────────────────────────────────────────────
        # INTRADAY TOP/BOTTOM TIMING
        # ─────────────────────────────────────────────────────────────────────
        # When did the high and low occur?
        high_idx = regular_data["high"].idxmax()
        low_idx = regular_data["low"].idxmin()

        high_time = regular_data.loc[high_idx, "timestamp"]
        low_time = regular_data.loc[low_idx, "timestamp"]

        # Convert to minutes from open
        open_time = regular_data.iloc[0]["timestamp"]
        high_minutes_from_open = (high_time - open_time).total_seconds() / 60
        low_minutes_from_open = (low_time - open_time).total_seconds() / 60

        # Did low come before high? (up day pattern)
        low_before_high = low_time < high_time

        # ─────────────────────────────────────────────────────────────────────
        # RECORD
        # ─────────────────────────────────────────────────────────────────────
        daily_records.append({
            "date": day,

            # Premarket features (EDGE 2)
            "pm_return": pm_return,
            "pm_range": pm_range,
            "pm_direction": pm_direction,
            "pm_vwap_deviation": pm_vwap_deviation,
            "pm_momentum_30": pm_momentum_30,
            "pm_momentum_60": pm_momentum_60,
            "pm_volume": pm_volume,

            # Afterhours features (previous day)
            "ah_return_lag1": ah_return,
            "ah_range_lag1": ah_range,
            "ah_direction_lag1": ah_direction,
            "ah_volume_lag1": ah_volume,

            # Overnight combined
            "overnight_return": overnight_return,
            "direction_agreement": direction_agreement,

            # Regular session
            "open_price": reg_open,
            "close_price": reg_close,
            "day_return": day_return,
            "day_range": day_range,
            "day_volume": reg_volume,

            # Time windows
            "morning_return": morning_return,
            "morning_range": morning_range,
            "afternoon_return": afternoon_return,
            "afternoon_range": afternoon_range,

            # Swing targets
            "morning_swing_up": morning_swing_up,
            "morning_swing_down": morning_swing_down,
            "has_morning_swing": has_morning_swing,
            "is_up_day": is_up_day,
            "is_down_day": is_down_day,

            # Timing targets
            "high_minutes_from_open": high_minutes_from_open,
            "low_minutes_from_open": low_minutes_from_open,
            "low_before_high": low_before_high,
        })

    result_df = pd.DataFrame(daily_records)
    result_df["date"] = pd.to_datetime(result_df["date"])

    print(f"\n[PASS] Built features for {len(result_df)} trading days")
    print(f"\n       Swing Statistics:")
    print(f"       - Morning swing days: {result_df['has_morning_swing'].sum()} ({100*result_df['has_morning_swing'].mean():.1f}%)")
    print(f"       - Up days (>0.2%):    {result_df['is_up_day'].sum()} ({100*result_df['is_up_day'].mean():.1f}%)")
    print(f"       - Down days (<-0.2%): {result_df['is_down_day'].sum()} ({100*result_df['is_down_day'].mean():.1f}%)")
    print(f"\n       Timing Statistics:")
    print(f"       - Avg high time: {result_df['high_minutes_from_open'].mean():.0f} min from open")
    print(f"       - Avg low time:  {result_df['low_minutes_from_open'].mean():.0f} min from open")
    print(f"       - Low before high: {100*result_df['low_before_high'].mean():.1f}%")

    return result_df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: ADD LAGGED FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
def add_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged features from previous days."""
    print("\n" + "=" * 70)
    print("STEP 3: ADD LAGGED FEATURES")
    print("=" * 70)

    df = df.sort_values("date").copy()

    # Lag day returns
    for lag in [1, 2, 3, 5]:
        df[f"day_return_lag{lag}"] = df["day_return"].shift(lag)
        df[f"day_range_lag{lag}"] = df["day_range"].shift(lag)

    # Lag premarket features
    for lag in [1, 2, 3]:
        df[f"pm_return_lag{lag}"] = df["pm_return"].shift(lag)
        df[f"pm_direction_lag{lag}"] = df["pm_direction"].shift(lag)

    # Rolling stats
    df["day_return_ma5"] = df["day_return"].shift(1).rolling(5).mean()
    df["volatility_5d"] = df["day_return"].shift(1).rolling(5).std()
    df["pm_return_ma3"] = df["pm_return"].shift(1).rolling(3).mean()

    # Consecutive patterns
    df["consecutive_up_pm"] = (df["pm_direction"].shift(1) == 1).rolling(3).sum()
    df["consecutive_down_pm"] = (df["pm_direction"].shift(1) == -1).rolling(3).sum()

    print(f"[PASS] Added lagged features")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: TRAIN MODEL 1 - SWING DIRECTION PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
def train_swing_model(df: pd.DataFrame, feature_cols: list):
    """
    MODEL 1: Predict if today will be a swing day (up or down).

    Uses premarket + afterhours features to predict regular session movement.
    """
    print("\n" + "=" * 70)
    print("MODEL 1: SWING DIRECTION PREDICTOR")
    print("=" * 70)
    print("Predicts: Will today have a significant swing (up or down)?")

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report

    df_clean = df.dropna(subset=feature_cols + ["is_up_day"]).copy()

    # Time-series split
    split_idx = int(len(df_clean) * 0.8)
    train_df = df_clean.iloc[:split_idx - 3]
    test_df = df_clean.iloc[split_idx:]

    print(f"\n[INFO] Train: {len(train_df)} days, Test: {len(test_df)} days")

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    # Target: 1 = up day, 0 = down day (exclude flat days for clearer signal)
    # For training, use is_up_day
    y_train = train_df["is_up_day"].astype(int).values
    y_test = test_df["is_up_day"].astype(int).values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # L2 Logistic Regression (EDGE 1)
    model = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5

    print(f"\n[RESULTS]")
    print(f"  Accuracy: {acc:.3f}")
    print(f"  AUC:      {auc:.3f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['Down/Flat', 'Up'])}")

    # Top features
    coef = model.coef_[0]
    top_idx = np.argsort(np.abs(coef))[::-1][:10]
    print("\n  Top 10 predictive features:")
    for i, idx in enumerate(top_idx):
        direction = "+" if coef[idx] > 0 else "-"
        print(f"    {i+1}. {feature_cols[idx]}: {direction}{abs(coef[idx]):.4f}")

    return {
        "model": model,
        "scaler": scaler,
        "accuracy": acc,
        "auc": auc,
        "test_predictions": y_pred,
        "test_probabilities": y_proba,
        "test_dates": test_df["date"].values
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: TRAIN MODEL 2 - ENTRY/EXIT TIMING PREDICTOR
# ═══════════════════════════════════════════════════════════════════════════════
def train_timing_model(df: pd.DataFrame, feature_cols: list):
    """
    MODEL 2: Predict intraday timing (when is the low/entry point?).

    Predicts: Will the low occur before the high? (low_before_high)
    If true → buy early, sell late (up day pattern)
    If false → sell early, buy late (down day pattern)
    """
    print("\n" + "=" * 70)
    print("MODEL 2: ENTRY/EXIT TIMING PREDICTOR")
    print("=" * 70)
    print("Predicts: Will the intraday low occur BEFORE the high?")
    print("  - If YES -> Entry early in day, exit late (up day pattern)")
    print("  - If NO  -> Entry late in day after high (down day pattern)")

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

    df_clean = df.dropna(subset=feature_cols + ["low_before_high"]).copy()

    split_idx = int(len(df_clean) * 0.8)
    train_df = df_clean.iloc[:split_idx - 3]
    test_df = df_clean.iloc[split_idx:]

    print(f"\n[INFO] Train: {len(train_df)} days, Test: {len(test_df)} days")

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    y_train = train_df["low_before_high"].astype(int).values
    y_test = test_df["low_before_high"].astype(int).values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5

    print(f"\n[RESULTS]")
    print(f"  Accuracy: {acc:.3f}")
    print(f"  AUC:      {auc:.3f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=['High First', 'Low First'])}")

    # Top features
    coef = model.coef_[0]
    top_idx = np.argsort(np.abs(coef))[::-1][:10]
    print("\n  Top 10 predictive features:")
    for i, idx in enumerate(top_idx):
        direction = "+" if coef[idx] > 0 else "-"
        print(f"    {i+1}. {feature_cols[idx]}: {direction}{abs(coef[idx]):.4f}")

    return {
        "model": model,
        "scaler": scaler,
        "accuracy": acc,
        "auc": auc,
        "test_predictions": y_pred,
        "test_probabilities": y_proba,
        "test_dates": test_df["date"].values
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: COMBINED SIGNAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def analyze_combined_signals(swing_results: dict, timing_results: dict, df: pd.DataFrame):
    """
    Analyze when BOTH models agree.

    Trade rules:
      - BUY signal when: swing_model predicts UP + timing_model predicts LOW_FIRST
      - SELL signal when: swing_model predicts DOWN + timing_model predicts HIGH_FIRST
    """
    print("\n" + "=" * 70)
    print("COMBINED SIGNAL ANALYSIS (Both Models Must Agree)")
    print("=" * 70)

    # Align predictions by date
    swing_dates = swing_results["test_dates"]
    timing_dates = timing_results["test_dates"]

    # Find common dates
    common_dates = set(swing_dates) & set(timing_dates)
    print(f"[INFO] Common test dates: {len(common_dates)}")

    # Get predictions for common dates
    signals = []
    for date in sorted(common_dates):
        swing_idx = np.where(swing_dates == date)[0][0]
        timing_idx = np.where(timing_dates == date)[0][0]

        swing_pred = swing_results["test_predictions"][swing_idx]  # 1=up, 0=down
        swing_prob = swing_results["test_probabilities"][swing_idx]
        timing_pred = timing_results["test_predictions"][timing_idx]  # 1=low first, 0=high first
        timing_prob = timing_results["test_probabilities"][timing_idx]

        # BUY signal: up day predicted + low comes first (entry early)
        buy_signal = (swing_pred == 1) and (timing_pred == 1)

        # SELL signal: down day predicted + high comes first
        sell_signal = (swing_pred == 0) and (timing_pred == 0)

        # Combined confidence
        combined_confidence = (swing_prob + timing_prob) / 2 if buy_signal else ((1 - swing_prob) + (1 - timing_prob)) / 2 if sell_signal else 0.5

        # Get actual outcome
        day_data = df[df["date"] == pd.Timestamp(date)]
        if len(day_data) > 0:
            actual_return = day_data.iloc[0]["day_return"]
        else:
            actual_return = 0

        signals.append({
            "date": date,
            "swing_pred": swing_pred,
            "swing_prob": swing_prob,
            "timing_pred": timing_pred,
            "timing_prob": timing_prob,
            "buy_signal": buy_signal,
            "sell_signal": sell_signal,
            "combined_confidence": combined_confidence,
            "actual_return": actual_return
        })

    signals_df = pd.DataFrame(signals)

    # Analyze signal quality
    buy_signals = signals_df[signals_df["buy_signal"] == True]
    sell_signals = signals_df[signals_df["sell_signal"] == True]
    no_signal = signals_df[(signals_df["buy_signal"] == False) & (signals_df["sell_signal"] == False)]

    print(f"\n[SIGNAL DISTRIBUTION]")
    print(f"  BUY signals:  {len(buy_signals)} ({100*len(buy_signals)/len(signals_df):.1f}%)")
    print(f"  SELL signals: {len(sell_signals)} ({100*len(sell_signals)/len(signals_df):.1f}%)")
    print(f"  No signal:    {len(no_signal)} ({100*len(no_signal)/len(signals_df):.1f}%)")

    if len(buy_signals) > 0:
        buy_win_rate = (buy_signals["actual_return"] > 0).mean()
        buy_avg_return = buy_signals["actual_return"].mean()
        print(f"\n[BUY SIGNAL PERFORMANCE]")
        print(f"  Win rate:    {100*buy_win_rate:.1f}%")
        print(f"  Avg return:  {100*buy_avg_return:.2f}%")

    if len(sell_signals) > 0:
        sell_win_rate = (sell_signals["actual_return"] < 0).mean()
        sell_avg_return = sell_signals["actual_return"].mean()
        print(f"\n[SELL SIGNAL PERFORMANCE]")
        print(f"  Win rate:    {100*sell_win_rate:.1f}%")
        print(f"  Avg return:  {100*sell_avg_return:.2f}%")

    return signals_df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: SAVE MODELS
# ═══════════════════════════════════════════════════════════════════════════════
def save_models(swing_results, timing_results, feature_cols, signals_df):
    """Save both models and combined signal analysis."""
    print("\n" + "=" * 70)
    print("STEP 7: SAVE MODELS")
    print("=" * 70)

    import joblib

    models_dir = project_root / "models" / "production"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "spy_dual_models.joblib"
    joblib.dump({
        "swing_model": {
            "model": swing_results["model"],
            "scaler": swing_results["scaler"],
            "metrics": {"accuracy": swing_results["accuracy"], "auc": swing_results["auc"]}
        },
        "timing_model": {
            "model": timing_results["model"],
            "scaler": timing_results["scaler"],
            "metrics": {"accuracy": timing_results["accuracy"], "auc": timing_results["auc"]}
        },
        "feature_cols": feature_cols,
        "signals_analysis": signals_df.to_dict(),
        "trained_at": datetime.now().isoformat(),
        "description": "Dual model system: Swing Direction + Entry/Exit Timing"
    }, model_path)

    print(f"[PASS] Saved models to {model_path}")
    return model_path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 70)
    print("GIGA TRADER - DUAL MODEL TRAINING SYSTEM")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("TWO MODELS THAT MUST AGREE:")
    print("  1. SWING DIRECTION: Predicts up/down day")
    print("  2. ENTRY/EXIT TIMING: Predicts when low/high occurs")
    print()
    print("TRADE RULES:")
    print("  BUY  when: Up day predicted + Low comes first")
    print("  SELL when: Down day predicted + High comes first")

    # Step 1: Download data
    df_1min = download_1min_extended_hours(days=90)

    # Step 2: Build extended hours features
    df_daily = build_extended_hours_features(df_1min)

    # Step 3: Add lagged features
    df_daily = add_lagged_features(df_daily)

    # Feature columns (available at prediction time)
    feature_cols = [
        # Premarket features (today)
        "pm_return", "pm_range", "pm_direction", "pm_vwap_deviation",
        "pm_momentum_30", "pm_momentum_60",

        # Afterhours features (yesterday)
        "ah_return_lag1", "ah_range_lag1", "ah_direction_lag1",

        # Overnight combined
        "overnight_return", "direction_agreement",

        # Lagged daily features
        "day_return_lag1", "day_return_lag2", "day_return_lag3",
        "day_range_lag1", "day_range_lag2",
        "pm_return_lag1", "pm_return_lag2",
        "pm_direction_lag1", "pm_direction_lag2",

        # Rolling stats
        "day_return_ma5", "volatility_5d", "pm_return_ma3",
        "consecutive_up_pm", "consecutive_down_pm"
    ]

    # Filter to available columns
    feature_cols = [c for c in feature_cols if c in df_daily.columns]
    print(f"\n[INFO] Using {len(feature_cols)} features")

    # Step 4: Train swing model
    swing_results = train_swing_model(df_daily, feature_cols)

    # Step 5: Train timing model
    timing_results = train_timing_model(df_daily, feature_cols)

    # Step 6: Combined signal analysis
    signals_df = analyze_combined_signals(swing_results, timing_results, df_daily)

    # Step 7: Save
    model_path = save_models(swing_results, timing_results, feature_cols, signals_df)

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nModel 1 (Swing Direction): AUC = {swing_results['auc']:.3f}")
    print(f"Model 2 (Entry/Exit Timing): AUC = {timing_results['auc']:.3f}")
    print(f"\nModels saved to: {model_path}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
