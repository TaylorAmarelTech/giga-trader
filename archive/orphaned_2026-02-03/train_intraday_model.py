"""
GIGA TRADER - Intraday Swing Opportunity Model
===============================================
Trains on 1-minute SPY data to predict intraday tradable opportunities
following EDGE 3: Intraday Tradable Opportunities.

Target patterns:
  1. Morning Dip (10:15-12:30): Buy after price drops ≥0.3% from open
  2. Afternoon Swing (12:30-15:30): Buy swing up ≥0.4% from intraday low

Usage:
    python src/train_intraday_model.py
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
# STEP 1: DOWNLOAD INTRADAY DATA
# ═══════════════════════════════════════════════════════════════════════════════
def download_intraday_data(days: int = 60) -> pd.DataFrame:
    """
    Download 1-minute SPY data from Alpaca.
    Note: Free tier may have limits on historical intraday data.
    """
    print("\n" + "=" * 70)
    print("STEP 1: DOWNLOAD INTRADAY DATA (1-minute bars)")
    print("=" * 70)

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    client = StockHistoricalDataClient(api_key, secret_key)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    print(f"[INFO] Downloading {days} days of 1-minute SPY data...")
    print(f"       Date range: {start_date.date()} to {end_date.date()}")

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

    print(f"[PASS] Downloaded {len(df):,} 1-minute bars")
    print(f"       Trading days: {df['date'].nunique()}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: BUILD DAILY INTRADAY FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
def build_intraday_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each trading day, calculate intraday features at key decision points.

    Decision points per EDGE 3:
      - 10:15 ET: Morning dip entry check
      - 12:30 ET: Afternoon swing entry check
      - 15:55 ET: End of day exit

    Features calculated AT decision time (no look-ahead):
      - Return from open
      - Intraday high/low
      - Volume patterns
      - Momentum indicators
    """
    print("\n" + "=" * 70)
    print("STEP 2: BUILD INTRADAY FEATURES (per trading day)")
    print("=" * 70)

    # Filter to regular trading hours only (9:30 - 16:00)
    df_regular = df[
        (df["hour"] >= 9) &
        ((df["hour"] < 16) | ((df["hour"] == 9) & (df["minute"] >= 30)))
    ].copy()

    trading_days = df_regular["date"].unique()
    print(f"[INFO] Processing {len(trading_days)} trading days...")

    daily_records = []

    for day in trading_days:
        day_data = df_regular[df_regular["date"] == day].sort_values("timestamp")

        if len(day_data) < 100:  # Need sufficient data
            continue

        # ─────────────────────────────────────────────────────────────────────
        # Key timestamps
        # ─────────────────────────────────────────────────────────────────────
        open_time = time(9, 30)
        morning_check_time = time(10, 15)
        afternoon_check_time = time(12, 30)
        close_time = time(15, 55)

        # Get prices at key times
        open_bar = day_data[day_data["time"] >= open_time].iloc[0] if len(day_data[day_data["time"] >= open_time]) > 0 else None
        morning_bars = day_data[day_data["time"] <= morning_check_time]
        afternoon_bars = day_data[(day_data["time"] >= morning_check_time) & (day_data["time"] <= afternoon_check_time)]
        eod_bars = day_data[day_data["time"] <= close_time]

        if open_bar is None or len(morning_bars) < 10:
            continue

        open_price = open_bar["open"]

        # ─────────────────────────────────────────────────────────────────────
        # MORNING FEATURES (calculated AT 10:15)
        # ─────────────────────────────────────────────────────────────────────
        morning_high = morning_bars["high"].max()
        morning_low = morning_bars["low"].min()
        morning_close = morning_bars.iloc[-1]["close"]
        morning_volume = morning_bars["volume"].sum()

        # Return from open at 10:15
        return_at_1015 = (morning_close - open_price) / open_price

        # Morning range
        morning_range = (morning_high - morning_low) / open_price

        # Distance from morning low (how much has it bounced?)
        dist_from_morning_low = (morning_close - morning_low) / open_price

        # Morning momentum (last 15 min vs first 15 min)
        first_15 = morning_bars.head(15)
        last_15 = morning_bars.tail(15)
        if len(first_15) > 0 and len(last_15) > 0:
            morning_momentum = (last_15["close"].mean() - first_15["close"].mean()) / first_15["close"].mean()
        else:
            morning_momentum = 0

        # ─────────────────────────────────────────────────────────────────────
        # AFTERNOON FEATURES (calculated AT 12:30)
        # ─────────────────────────────────────────────────────────────────────
        if len(afternoon_bars) > 0:
            afternoon_open = day_data[day_data["time"] >= morning_check_time].iloc[0]["open"]
            afternoon_close = afternoon_bars.iloc[-1]["close"]

            # Intraday low up to 12:30
            intraday_low_to_1230 = eod_bars[eod_bars["time"] <= afternoon_check_time]["low"].min()
            intraday_high_to_1230 = eod_bars[eod_bars["time"] <= afternoon_check_time]["high"].max()

            return_at_1230 = (afternoon_close - open_price) / open_price
            return_from_low = (afternoon_close - intraday_low_to_1230) / intraday_low_to_1230

            # Afternoon momentum
            afternoon_momentum = (afternoon_close - afternoon_open) / afternoon_open
        else:
            return_at_1230 = return_at_1015
            return_from_low = 0
            afternoon_momentum = 0
            intraday_low_to_1230 = morning_low
            intraday_high_to_1230 = morning_high

        # ─────────────────────────────────────────────────────────────────────
        # END OF DAY OUTCOMES (for labeling)
        # ─────────────────────────────────────────────────────────────────────
        close_bar = eod_bars.iloc[-1]
        close_price = close_bar["close"]
        day_return = (close_price - open_price) / open_price

        # Intraday extremes
        day_high = eod_bars["high"].max()
        day_low = eod_bars["low"].min()

        # Max gain from 10:15 entry
        post_1015_high = day_data[day_data["time"] >= morning_check_time]["high"].max()
        max_gain_from_1015 = (post_1015_high - morning_close) / morning_close

        # Max gain from 12:30 entry
        post_1230_data = day_data[day_data["time"] >= afternoon_check_time]
        if len(post_1230_data) > 0:
            post_1230_high = post_1230_data["high"].max()
            post_1230_close = post_1230_data.iloc[-1]["close"]
            max_gain_from_1230 = (post_1230_high - afternoon_close) / afternoon_close if afternoon_close > 0 else 0
            return_1230_to_close = (post_1230_close - afternoon_close) / afternoon_close if afternoon_close > 0 else 0
        else:
            max_gain_from_1230 = 0
            return_1230_to_close = 0

        # ─────────────────────────────────────────────────────────────────────
        # PATTERN DETECTION (targets)
        # ─────────────────────────────────────────────────────────────────────
        # Morning Dip Pattern: Price dropped from open, then recovered
        is_morning_dip = return_at_1015 < -0.003  # Dropped 0.3%+

        # Did morning dip trade work? (gained 0.2%+ after entry)
        morning_dip_profitable = max_gain_from_1015 >= 0.002 if is_morning_dip else False

        # Afternoon Swing Pattern: Good entry at 12:30
        is_afternoon_swing_setup = return_from_low >= 0.002  # Bounced 0.2% from low

        # Did afternoon swing work?
        afternoon_swing_profitable = max_gain_from_1230 >= 0.002

        # ─────────────────────────────────────────────────────────────────────
        # RECORD
        # ─────────────────────────────────────────────────────────────────────
        daily_records.append({
            "date": day,
            "open_price": open_price,
            "close_price": close_price,

            # Morning features (at 10:15)
            "return_at_1015": return_at_1015,
            "morning_range": morning_range,
            "dist_from_morning_low": dist_from_morning_low,
            "morning_momentum": morning_momentum,
            "morning_volume": morning_volume,

            # Afternoon features (at 12:30)
            "return_at_1230": return_at_1230,
            "return_from_low": return_from_low,
            "afternoon_momentum": afternoon_momentum,

            # Intraday stats
            "day_return": day_return,
            "day_high": day_high,
            "day_low": day_low,
            "day_range": (day_high - day_low) / open_price,

            # Outcomes
            "max_gain_from_1015": max_gain_from_1015,
            "max_gain_from_1230": max_gain_from_1230,
            "return_1230_to_close": return_1230_to_close,

            # Pattern flags
            "is_morning_dip": is_morning_dip,
            "is_afternoon_swing_setup": is_afternoon_swing_setup,

            # Targets (what we predict)
            "morning_dip_profitable": morning_dip_profitable,
            "afternoon_swing_profitable": afternoon_swing_profitable,
        })

    result_df = pd.DataFrame(daily_records)
    result_df["date"] = pd.to_datetime(result_df["date"])

    print(f"[PASS] Built features for {len(result_df)} trading days")
    print(f"\n       Pattern Statistics:")
    print(f"       Morning dip days:      {result_df['is_morning_dip'].sum()} ({100*result_df['is_morning_dip'].mean():.1f}%)")
    print(f"       - Profitable:          {result_df['morning_dip_profitable'].sum()} ({100*result_df[result_df['is_morning_dip']]['morning_dip_profitable'].mean():.1f}% win rate)")
    print(f"       Afternoon swing setup: {result_df['is_afternoon_swing_setup'].sum()} ({100*result_df['is_afternoon_swing_setup'].mean():.1f}%)")
    print(f"       - Profitable:          {result_df['afternoon_swing_profitable'].sum()} ({100*result_df['afternoon_swing_profitable'].mean():.1f}%)")

    return result_df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: ADD LAGGED FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
def add_lagged_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add lagged features from previous days (no look-ahead)."""
    print("\n" + "=" * 70)
    print("STEP 3: ADD LAGGED FEATURES")
    print("=" * 70)

    df = df.sort_values("date").copy()

    # Previous day features
    for lag in [1, 2, 3, 5]:
        df[f"day_return_lag{lag}"] = df["day_return"].shift(lag)
        df[f"day_range_lag{lag}"] = df["day_range"].shift(lag)
        df[f"morning_dip_lag{lag}"] = df["is_morning_dip"].shift(lag).astype(float)

    # Rolling averages
    df["day_return_ma5"] = df["day_return"].shift(1).rolling(5).mean()
    df["day_range_ma5"] = df["day_range"].shift(1).rolling(5).mean()
    df["volatility_5d"] = df["day_return"].shift(1).rolling(5).std()

    # Consecutive patterns
    df["consecutive_dip_days"] = df["is_morning_dip"].shift(1).rolling(3).sum()

    print(f"[PASS] Added lagged features")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: TRAIN MODELS FOR EACH PATTERN
# ═══════════════════════════════════════════════════════════════════════════════
def train_pattern_models(df: pd.DataFrame):
    """
    Train separate models for:
    1. Morning Dip: Given a dip occurred, will it be profitable?
    2. Afternoon Swing: Will 12:30 entry be profitable?
    """
    print("\n" + "=" * 70)
    print("STEP 4: TRAIN PATTERN MODELS")
    print("=" * 70)

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

    # Drop NaN
    df_clean = df.dropna().copy()
    print(f"[INFO] Clean samples: {len(df_clean)}")

    # Time-series split
    split_idx = int(len(df_clean) * 0.8)
    train_df = df_clean.iloc[:split_idx - 3]  # 3-day embargo
    test_df = df_clean.iloc[split_idx:]

    print(f"[INFO] Train: {len(train_df)} days, Test: {len(test_df)} days")

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE COLUMNS
    # ─────────────────────────────────────────────────────────────────────────
    feature_cols = [
        # Intraday features (known at decision time)
        "return_at_1015", "morning_range", "dist_from_morning_low", "morning_momentum",
        "return_at_1230", "return_from_low", "afternoon_momentum",

        # Lagged features (from previous days)
        "day_return_lag1", "day_return_lag2", "day_return_lag3",
        "day_range_lag1", "day_range_lag2",
        "morning_dip_lag1", "morning_dip_lag2",
        "day_return_ma5", "day_range_ma5", "volatility_5d",
        "consecutive_dip_days"
    ]

    # Filter to available columns
    feature_cols = [c for c in feature_cols if c in df_clean.columns]
    print(f"[INFO] Using {len(feature_cols)} features")

    models = {}
    results = {}

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL 1: MORNING DIP PROFITABILITY
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[MODEL 1] Morning Dip Profitability Predictor")
    print("-" * 50)

    # Only train on days with morning dips
    train_dip = train_df[train_df["is_morning_dip"] == True]
    test_dip = test_df[test_df["is_morning_dip"] == True]

    if len(train_dip) >= 10 and len(test_dip) >= 3:
        X_train = train_dip[feature_cols].values
        y_train = train_dip["morning_dip_profitable"].astype(int).values
        X_test = test_dip[feature_cols].values
        y_test = test_dip["morning_dip_profitable"].astype(int).values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # L2 Logistic Regression (per EDGE 1)
        model = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)
        model.fit(X_train_scaled, y_train)

        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5

        print(f"  Train dip days: {len(train_dip)}, Test dip days: {len(test_dip)}")
        print(f"  Accuracy:  {acc:.3f}")
        print(f"  Precision: {prec:.3f}")
        print(f"  Recall:    {rec:.3f}")
        print(f"  AUC:       {auc:.3f}")

        models["morning_dip"] = {"model": model, "scaler": scaler}
        results["morning_dip"] = {"accuracy": acc, "precision": prec, "recall": rec, "auc": auc}
    else:
        print(f"  [SKIP] Not enough morning dip samples (train={len(train_dip)}, test={len(test_dip)})")

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL 2: AFTERNOON SWING PROFITABILITY
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[MODEL 2] Afternoon Swing Profitability Predictor")
    print("-" * 50)

    X_train = train_df[feature_cols].values
    y_train = train_df["afternoon_swing_profitable"].astype(int).values
    X_test = test_df[feature_cols].values
    y_test = test_df["afternoon_swing_profitable"].astype(int).values

    scaler2 = StandardScaler()
    X_train_scaled = scaler2.fit_transform(X_train)
    X_test_scaled = scaler2.transform(X_test)

    # L2 Logistic Regression
    model2 = LogisticRegression(penalty="l2", C=1.0, max_iter=1000, random_state=42)
    model2.fit(X_train_scaled, y_train)

    y_pred = model2.predict(X_test_scaled)
    y_proba = model2.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5

    print(f"  Train days: {len(train_df)}, Test days: {len(test_df)}")
    print(f"  Accuracy:  {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  AUC:       {auc:.3f}")

    models["afternoon_swing"] = {"model": model2, "scaler": scaler2}
    results["afternoon_swing"] = {"accuracy": acc, "precision": prec, "recall": rec, "auc": auc}

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL 3: SHALLOW GRADIENT BOOSTING (for comparison)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[MODEL 3] Gradient Boosting (max_depth=3) for Afternoon Swing")
    print("-" * 50)

    model3 = GradientBoostingClassifier(
        n_estimators=50,
        max_depth=3,  # NEVER > 5
        learning_rate=0.1,
        subsample=0.8,
        random_state=42
    )
    model3.fit(X_train_scaled, y_train)

    y_pred = model3.predict(X_test_scaled)
    y_proba = model3.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if len(np.unique(y_test)) > 1 else 0.5

    print(f"  Accuracy:  {acc:.3f}")
    print(f"  Precision: {prec:.3f}")
    print(f"  Recall:    {rec:.3f}")
    print(f"  AUC:       {auc:.3f}")

    # Feature importance
    importances = model3.feature_importances_
    top_idx = np.argsort(importances)[::-1][:5]
    print("\n  Top 5 features:")
    for i, idx in enumerate(top_idx):
        print(f"    {i+1}. {feature_cols[idx]}: {importances[idx]:.3f}")

    models["afternoon_swing_gb"] = {"model": model3, "scaler": scaler2}
    results["afternoon_swing_gb"] = {"accuracy": acc, "precision": prec, "recall": rec, "auc": auc}

    return models, results, feature_cols


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: SAVE MODELS
# ═══════════════════════════════════════════════════════════════════════════════
def save_models(models, results, feature_cols):
    """Save trained models."""
    print("\n" + "=" * 70)
    print("STEP 5: SAVE MODELS")
    print("=" * 70)

    import joblib

    models_dir = project_root / "models" / "production"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "spy_intraday_models.joblib"
    joblib.dump({
        "models": models,
        "results": results,
        "feature_cols": feature_cols,
        "trained_at": datetime.now().isoformat(),
        "description": "Intraday swing opportunity models (EDGE 3)"
    }, model_path)

    print(f"[PASS] Saved models to {model_path}")
    return model_path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 70)
    print("GIGA TRADER - Intraday Swing Opportunity Model")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Following EDGE 3: Intraday Tradable Opportunities")
    print()
    print("Patterns to predict:")
    print("  1. Morning Dip (10:15): Buy after ≥0.3% drop from open")
    print("  2. Afternoon Swing (12:30): Buy on bounce from intraday low")

    # Step 1: Download intraday data
    df_intraday = download_intraday_data(days=90)  # More days for better training

    # Step 2: Build daily features
    df_daily = build_intraday_features(df_intraday)

    # Step 3: Add lagged features
    df_daily = add_lagged_features(df_daily)

    # Step 4: Train models
    models, results, feature_cols = train_pattern_models(df_daily)

    # Step 5: Save
    model_path = save_models(models, results, feature_cols)

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nResults Summary:")
    for name, res in results.items():
        print(f"  {name}: AUC={res['auc']:.3f}, Precision={res['precision']:.3f}")
    print(f"\nModels saved to: {model_path}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
