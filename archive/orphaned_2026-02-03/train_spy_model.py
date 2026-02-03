"""
GIGA TRADER - SPY Model Training Pipeline
==========================================
Downloads 1 year of SPY data, engineers features, and trains a regularized model
following the edge strategies defined in config/edge_strategies.yaml.

Usage:
    python src/train_spy_model.py
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import pandas as pd
from dotenv import load_dotenv

# Load environment
load_dotenv(project_root / ".env")


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: DATA ACQUISITION
# ═══════════════════════════════════════════════════════════════════════════════
def download_spy_data(days: int = 365) -> pd.DataFrame:
    """Download SPY historical data from Alpaca."""
    print("\n" + "=" * 70)
    print("STEP 1: DATA ACQUISITION")
    print("=" * 70)

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")

    client = StockHistoricalDataClient(api_key, secret_key)

    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    print(f"[INFO] Downloading {days} days of SPY daily data...")
    print(f"       Date range: {start_date.date()} to {end_date.date()}")

    request = StockBarsRequest(
        symbol_or_symbols="SPY",
        timeframe=TimeFrame.Day,
        start=start_date,
        end=end_date
    )

    bars = client.get_stock_bars(request)
    df = bars.df.reset_index()

    # Rename columns for consistency
    df = df.rename(columns={
        "timestamp": "date",
        "trade_count": "trades",
    })

    # Convert to Eastern Time and extract date
    df["date"] = pd.to_datetime(df["date"]).dt.tz_convert("America/New_York").dt.date
    df["date"] = pd.to_datetime(df["date"])

    print(f"[PASS] Downloaded {len(df)} bars")
    print(f"       Columns: {list(df.columns)}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Engineer features following the regularization-first philosophy."""
    print("\n" + "=" * 70)
    print("STEP 2: FEATURE ENGINEERING")
    print("=" * 70)

    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    # ─────────────────────────────────────────────────────────────────────────────
    # PRICE-BASED FEATURES
    # ─────────────────────────────────────────────────────────────────────────────
    print("[INFO] Engineering price features...")

    # Returns
    df["return_1d"] = df["close"].pct_change(1)
    df["return_2d"] = df["close"].pct_change(2)
    df["return_3d"] = df["close"].pct_change(3)
    df["return_5d"] = df["close"].pct_change(5)
    df["return_10d"] = df["close"].pct_change(10)
    df["return_20d"] = df["close"].pct_change(20)

    # Lagged returns (for avoiding look-ahead bias)
    df["return_1d_lag1"] = df["return_1d"].shift(1)
    df["return_1d_lag2"] = df["return_1d"].shift(2)
    df["return_1d_lag3"] = df["return_1d"].shift(3)
    df["return_1d_lag5"] = df["return_1d"].shift(5)

    # Moving averages
    df["sma_5"] = df["close"].rolling(5).mean()
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_20"] = df["close"].rolling(20).mean()
    df["sma_50"] = df["close"].rolling(50).mean()

    # Price vs moving averages
    df["close_vs_sma5"] = (df["close"] - df["sma_5"]) / df["sma_5"]
    df["close_vs_sma10"] = (df["close"] - df["sma_10"]) / df["sma_10"]
    df["close_vs_sma20"] = (df["close"] - df["sma_20"]) / df["sma_20"]
    df["close_vs_sma50"] = (df["close"] - df["sma_50"]) / df["sma_50"]

    # SMA crossovers
    df["sma5_vs_sma20"] = (df["sma_5"] - df["sma_20"]) / df["sma_20"]
    df["sma10_vs_sma50"] = (df["sma_10"] - df["sma_50"]) / df["sma_50"]

    # ─────────────────────────────────────────────────────────────────────────────
    # VOLATILITY FEATURES
    # ─────────────────────────────────────────────────────────────────────────────
    print("[INFO] Engineering volatility features...")

    # Daily range
    df["daily_range"] = (df["high"] - df["low"]) / df["open"]
    df["daily_range_ma5"] = df["daily_range"].rolling(5).mean()

    # Historical volatility
    df["volatility_5d"] = df["return_1d"].rolling(5).std()
    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["volatility_20d"] = df["return_1d"].rolling(20).std()

    # Volatility ratio (current vs longer term)
    df["vol_ratio_5_20"] = df["volatility_5d"] / df["volatility_20d"]

    # ATR (Average True Range)
    df["tr"] = np.maximum(
        df["high"] - df["low"],
        np.maximum(
            abs(df["high"] - df["close"].shift(1)),
            abs(df["low"] - df["close"].shift(1))
        )
    )
    df["atr_14"] = df["tr"].rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / df["close"]

    # ─────────────────────────────────────────────────────────────────────────────
    # VOLUME FEATURES
    # ─────────────────────────────────────────────────────────────────────────────
    print("[INFO] Engineering volume features...")

    df["volume_ma5"] = df["volume"].rolling(5).mean()
    df["volume_ma20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_ma20"]
    df["volume_ratio_lag1"] = df["volume_ratio"].shift(1)

    # ─────────────────────────────────────────────────────────────────────────────
    # MOMENTUM FEATURES
    # ─────────────────────────────────────────────────────────────────────────────
    print("[INFO] Engineering momentum features...")

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # RSI lagged
    df["rsi_14_lag1"] = df["rsi_14"].shift(1)

    # Momentum
    df["momentum_5d"] = df["close"] / df["close"].shift(5) - 1
    df["momentum_10d"] = df["close"] / df["close"].shift(10) - 1
    df["momentum_20d"] = df["close"] / df["close"].shift(20) - 1

    # Rate of change
    df["roc_5"] = (df["close"] - df["close"].shift(5)) / df["close"].shift(5)
    df["roc_10"] = (df["close"] - df["close"].shift(10)) / df["close"].shift(10)

    # ─────────────────────────────────────────────────────────────────────────────
    # STREAK FEATURES
    # ─────────────────────────────────────────────────────────────────────────────
    print("[INFO] Engineering streak features...")

    # Consecutive up/down days
    df["up_day"] = (df["return_1d"] > 0).astype(int)
    df["down_day"] = (df["return_1d"] < 0).astype(int)

    # Count consecutive up days
    df["up_streak"] = df["up_day"].groupby(
        (df["up_day"] != df["up_day"].shift()).cumsum()
    ).cumsum() * df["up_day"]

    # Count consecutive down days
    df["down_streak"] = df["down_day"].groupby(
        (df["down_day"] != df["down_day"].shift()).cumsum()
    ).cumsum() * df["down_day"]

    # Lagged streaks
    df["up_streak_lag1"] = df["up_streak"].shift(1)
    df["down_streak_lag1"] = df["down_streak"].shift(1)

    # ─────────────────────────────────────────────────────────────────────────────
    # DISTANCE FEATURES
    # ─────────────────────────────────────────────────────────────────────────────
    print("[INFO] Engineering distance features...")

    # Distance from recent high/low
    df["high_20d"] = df["high"].rolling(20).max()
    df["low_20d"] = df["low"].rolling(20).min()
    df["dist_from_high_20d"] = (df["close"] - df["high_20d"]) / df["high_20d"]
    df["dist_from_low_20d"] = (df["close"] - df["low_20d"]) / df["low_20d"]

    # Position in range
    df["position_in_range_20d"] = (df["close"] - df["low_20d"]) / (df["high_20d"] - df["low_20d"] + 1e-8)

    # ─────────────────────────────────────────────────────────────────────────────
    # DAY OF WEEK FEATURES
    # ─────────────────────────────────────────────────────────────────────────────
    print("[INFO] Engineering calendar features...")

    df["day_of_week"] = df["date"].dt.dayofweek
    df["is_monday"] = (df["day_of_week"] == 0).astype(int)
    df["is_friday"] = (df["day_of_week"] == 4).astype(int)

    # Month features
    df["month"] = df["date"].dt.month
    df["is_month_end"] = (df["date"].dt.is_month_end).astype(int)

    # Drop intermediate columns
    df = df.drop(columns=["up_day", "down_day", "tr", "day_of_week", "month"], errors="ignore")

    # List all feature columns
    feature_cols = [c for c in df.columns if c not in ["symbol", "date", "open", "high", "low", "close", "volume", "trades", "vwap"]]
    print(f"[PASS] Engineered {len(feature_cols)} features")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: CREATE TARGET LABELS
# ═══════════════════════════════════════════════════════════════════════════════
def create_targets(df: pd.DataFrame, threshold: float = 0.004) -> pd.DataFrame:
    """
    Create soft target labels following EDGE 4: Confidence Soft Targets.

    Uses sigmoid transform to convert future returns into soft probabilities.
    """
    print("\n" + "=" * 70)
    print("STEP 3: CREATE TARGET LABELS (Soft Targets)")
    print("=" * 70)

    df = df.copy()

    # Future return (what we're trying to predict)
    df["future_return_1d"] = df["close"].shift(-1) / df["close"] - 1

    # Hard binary label
    df["target_hard"] = (df["future_return_1d"] > threshold).astype(int)

    # Soft target using sigmoid transform
    # Formula: 1 / (1 + exp(-k * (return - threshold)))
    k = 50  # Steepness parameter
    df["target_soft"] = 1 / (1 + np.exp(-k * (df["future_return_1d"] - threshold)))

    # Label smoothing (EDGE 4)
    epsilon = 0.1
    df["target_smoothed"] = (1 - epsilon) * df["target_hard"] + epsilon / 2

    # Confidence weight (higher for samples far from threshold)
    df["sample_weight"] = np.clip(np.abs(df["future_return_1d"] - threshold) * 100, 0.3, 1.0)

    print(f"[INFO] Threshold: {threshold:.2%}")
    print(f"[INFO] Hard labels - Class 1: {df['target_hard'].sum()}, Class 0: {(1 - df['target_hard']).sum()}")
    print(f"[INFO] Soft target range: [{df['target_soft'].min():.3f}, {df['target_soft'].max():.3f}]")
    print(f"[PASS] Created soft targets with label smoothing (epsilon={epsilon})")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: PREPARE TRAINING DATA
# ═══════════════════════════════════════════════════════════════════════════════
def prepare_training_data(df: pd.DataFrame, test_size: float = 0.2):
    """
    Prepare training and test data with time-series aware splitting.

    CRITICAL: Uses purging and embargo to prevent look-ahead bias.
    """
    print("\n" + "=" * 70)
    print("STEP 4: PREPARE TRAINING DATA")
    print("=" * 70)

    # Drop rows with NaN
    df_clean = df.dropna().copy()
    print(f"[INFO] Rows after dropping NaN: {len(df_clean)}")

    # Feature columns (exclude target and metadata)
    exclude_cols = [
        "symbol", "date", "open", "high", "low", "close", "volume", "trades", "vwap",
        "future_return_1d", "target_hard", "target_soft", "target_smoothed", "sample_weight",
        "sma_5", "sma_10", "sma_20", "sma_50", "volume_ma5", "volume_ma20",
        "high_20d", "low_20d"
    ]
    feature_cols = [c for c in df_clean.columns if c not in exclude_cols]

    print(f"[INFO] Using {len(feature_cols)} features")

    # Time-series split (no shuffling!)
    # CRITICAL: Train on past, test on future
    split_idx = int(len(df_clean) * (1 - test_size))

    # Add embargo period (5 days) to prevent leakage
    embargo_days = 5
    train_df = df_clean.iloc[:split_idx - embargo_days].copy()
    test_df = df_clean.iloc[split_idx:].copy()

    print(f"[INFO] Train period: {train_df['date'].min().date()} to {train_df['date'].max().date()} ({len(train_df)} samples)")
    print(f"[INFO] Embargo: {embargo_days} days")
    print(f"[INFO] Test period:  {test_df['date'].min().date()} to {test_df['date'].max().date()} ({len(test_df)} samples)")

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    # Use soft targets for training (EDGE 4)
    y_train = train_df["target_soft"].values
    y_test = test_df["target_hard"].values  # Evaluate on hard labels

    # Sample weights
    weights_train = train_df["sample_weight"].values

    print(f"[PASS] Prepared train/test split with embargo")

    return X_train, X_test, y_train, y_test, weights_train, feature_cols, train_df, test_df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: TRAIN MODEL
# ═══════════════════════════════════════════════════════════════════════════════
def train_model(X_train, y_train, weights_train, feature_cols):
    """
    Train model following EDGE 1: Regularization-First Philosophy.

    Priority: L1 Logistic Regression > L2 > Elastic Net > Shallow Trees
    """
    print("\n" + "=" * 70)
    print("STEP 5: TRAIN MODEL (Regularization-First)")
    print("=" * 70)

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression, ElasticNet
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.pipeline import Pipeline

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    results = {}

    # ─────────────────────────────────────────────────────────────────────────────
    # MODEL 1: L1 Logistic Regression (Feature Discovery)
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[1/3] Training L1 Logistic Regression (Lasso)...")

    # Convert soft targets to binary for logistic regression
    y_train_binary = (y_train > 0.5).astype(int)

    model_l1 = LogisticRegression(
        penalty="l1",
        C=0.1,  # Strong regularization
        solver="saga",
        max_iter=5000,
        random_state=42
    )
    model_l1.fit(X_train_scaled, y_train_binary, sample_weight=weights_train)

    # Count non-zero coefficients (surviving features)
    n_features_l1 = np.sum(model_l1.coef_[0] != 0)
    print(f"       Non-zero coefficients: {n_features_l1}/{len(feature_cols)}")

    # Get top features
    coef_abs = np.abs(model_l1.coef_[0])
    top_idx = np.argsort(coef_abs)[::-1][:10]
    print("       Top 10 features by L1 coefficient:")
    for i, idx in enumerate(top_idx):
        if coef_abs[idx] > 0:
            print(f"         {i+1}. {feature_cols[idx]}: {model_l1.coef_[0][idx]:.4f}")

    results["l1"] = model_l1

    # ─────────────────────────────────────────────────────────────────────────────
    # MODEL 2: L2 Logistic Regression (Baseline)
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[2/3] Training L2 Logistic Regression (Ridge)...")

    model_l2 = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="saga",
        max_iter=5000,
        random_state=42
    )
    model_l2.fit(X_train_scaled, y_train_binary, sample_weight=weights_train)
    results["l2"] = model_l2

    print(f"       All {len(feature_cols)} features used (L2 doesn't zero out)")

    # ─────────────────────────────────────────────────────────────────────────────
    # MODEL 3: Shallow Gradient Boosting (max_depth=3)
    # ─────────────────────────────────────────────────────────────────────────────
    print("\n[3/3] Training Shallow Gradient Boosting (max_depth=3)...")

    model_gb = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,  # NEVER > 5 per EDGE 1
        learning_rate=0.05,
        subsample=0.7,
        min_samples_leaf=50,
        random_state=42
    )
    model_gb.fit(X_train_scaled, y_train_binary, sample_weight=weights_train)
    results["gradient_boosting"] = model_gb

    # Feature importance
    importances = model_gb.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    print("       Top 10 features by importance:")
    for i, idx in enumerate(top_idx):
        print(f"         {i+1}. {feature_cols[idx]}: {importances[idx]:.4f}")

    print("\n[PASS] Trained 3 models following regularization-first philosophy")

    return results, scaler


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: EVALUATE MODELS
# ═══════════════════════════════════════════════════════════════════════════════
def evaluate_models(models, scaler, X_test, y_test, feature_cols):
    """Evaluate all models on test set."""
    print("\n" + "=" * 70)
    print("STEP 6: EVALUATE MODELS")
    print("=" * 70)

    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        roc_auc_score, confusion_matrix, classification_report
    )

    X_test_scaled = scaler.transform(X_test)

    results = {}

    for name, model in models.items():
        print(f"\n[{name.upper()}]")

        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)

        print(f"  Accuracy:  {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall:    {rec:.4f}")
        print(f"  F1 Score:  {f1:.4f}")
        print(f"  AUC-ROC:   {auc:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        print(f"  Confusion Matrix:")
        print(f"    TN={cm[0,0]:3d}  FP={cm[0,1]:3d}")
        print(f"    FN={cm[1,0]:3d}  TP={cm[1,1]:3d}")

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "auc": auc,
            "predictions": y_pred,
            "probabilities": y_proba
        }

    # Find best model by AUC
    best_model = max(results.keys(), key=lambda k: results[k]["auc"])
    print(f"\n[BEST] {best_model.upper()} with AUC = {results[best_model]['auc']:.4f}")

    return results, best_model


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: SAVE MODEL
# ═══════════════════════════════════════════════════════════════════════════════
def save_model(models, scaler, feature_cols, best_model_name, results):
    """Save the best model and metadata."""
    print("\n" + "=" * 70)
    print("STEP 7: SAVE MODEL")
    print("=" * 70)

    import joblib

    # Create models directory
    models_dir = project_root / "models" / "production"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Save best model
    model_path = models_dir / "spy_model.joblib"
    joblib.dump({
        "model": models[best_model_name],
        "scaler": scaler,
        "feature_cols": feature_cols,
        "model_type": best_model_name,
        "metrics": results[best_model_name],
        "trained_at": datetime.now().isoformat()
    }, model_path)

    print(f"[PASS] Saved best model ({best_model_name}) to {model_path}")

    # Save all models for comparison
    all_models_path = models_dir / "spy_models_all.joblib"
    joblib.dump({
        "models": models,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "results": results
    }, all_models_path)

    print(f"[PASS] Saved all models to {all_models_path}")

    return model_path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    """Run the full training pipeline."""
    print("\n" + "=" * 70)
    print("GIGA TRADER - SPY Model Training Pipeline")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Following EDGE strategies from config/edge_strategies.yaml")

    # Step 1: Download data
    df = download_spy_data(days=365)

    # Step 2: Engineer features
    df = engineer_features(df)

    # Step 3: Create targets
    df = create_targets(df, threshold=0.004)  # 0.4% threshold

    # Step 4: Prepare training data
    X_train, X_test, y_train, y_test, weights_train, feature_cols, train_df, test_df = \
        prepare_training_data(df, test_size=0.2)

    # Step 5: Train models
    models, scaler = train_model(X_train, y_train, weights_train, feature_cols)

    # Step 6: Evaluate
    results, best_model = evaluate_models(models, scaler, X_test, y_test, feature_cols)

    # Step 7: Save
    model_path = save_model(models, scaler, feature_cols, best_model, results)

    # Final summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"Best model: {best_model}")
    print(f"Test AUC:   {results[best_model]['auc']:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return results, best_model


if __name__ == "__main__":
    main()
