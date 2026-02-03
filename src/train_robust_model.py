"""
GIGA TRADER - Robust Model Training with Full Edge Strategies
==============================================================
Implements ALL edge strategies:
  - EDGE 1: Regularization-first with feature robustness scoring
  - EDGE 2: Comprehensive extended hours features
  - EDGE 3: Grid-searched intraday opportunities
  - EDGE 4: Soft targets with confidence weighting
  - EDGE 5: Batch position scaling signals

Anti-overfitting measures:
  - Purged K-fold cross-validation with embargo
  - Dimensionality reduction (PCA, variance threshold, correlation filter)
  - Feature robustness scoring via LASSO path
  - Heavy regularization (L1/L2/ElasticNet)
  - Soft targets and label smoothing

Usage:
    python src/train_robust_model.py
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

# Import anti-overfitting module
from src.anti_overfit import (
    integrate_anti_overfit,
    compute_weighted_evaluation,
    WeightedModelEvaluator,
    StabilityAnalyzer,
    RobustnessEnsemble,
    create_robustness_ensemble,
)

# Import leak-proof CV (fixes data leakage issues)
from src.leak_proof_cv import (
    LeakProofPipeline,
    LeakProofCV,
    LeakProofFeatureSelector,
    LeakProofDimReducer,
    EnsembleReducer,
    train_with_leak_proof_cv,
)

# Import entry/exit timing model
from src.entry_exit_model import (
    EntryExitTimingModel,
    TargetLabeler,
    TimingFeatureEngineer,
    create_entry_exit_model,
)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════
CONFIG = {
    # ─────────────────────────────────────────────────────────────────────────
    # DATA SETTINGS (5-10 years of 1-min data)
    # ─────────────────────────────────────────────────────────────────────────
    "years_to_download": 5,  # 5-10 years of historical data
    "chunk_days": 30,  # Download in 30-day chunks (API limit)
    "min_bars_per_day": 200,  # Minimum bars to consider a valid trading day
    "min_premarket_bars": 10,  # Minimum PM bars (some early years have sparse data)
    "min_afterhours_bars": 10,  # Minimum AH bars

    # Missing data handling
    "fill_missing_bars": True,  # Forward-fill small gaps
    "max_gap_minutes": 15,  # Max gap to fill (larger gaps flagged as missing)
    "flag_incomplete_extended": True,  # Flag days with sparse PM/AH data

    # Feature engineering
    "swing_thresholds_to_test": [0.002, 0.0025, 0.003, 0.0035, 0.004],

    # ─────────────────────────────────────────────────────────────────────────
    # ADVANCED DIMENSIONALITY REDUCTION
    # Methods: "umap", "kernel_pca", "ica", "mutual_info", "agglomeration",
    #          "kmedoids", "ensemble", "ensemble_plus"
    # ─────────────────────────────────────────────────────────────────────────
    "dim_reduction_method": "ensemble_plus",  # Now includes K-Medoids
    "variance_threshold": 0.01,
    "correlation_threshold": 0.95,

    # UMAP params
    "umap_n_components": 20,
    "umap_n_neighbors": 15,
    "umap_min_dist": 0.1,
    "umap_metric": "euclidean",

    # Kernel PCA params
    "kpca_n_components": 25,
    "kpca_kernel": "rbf",
    "kpca_gamma": 0.01,

    # ICA params
    "ica_n_components": 20,
    "ica_max_iter": 500,

    # Mutual Information params
    "mi_n_features": 30,
    "mi_n_neighbors": 5,

    # Feature Agglomeration params
    "agglom_n_clusters": 25,

    # K-Medoids params (more robust to outliers than K-Means)
    "kmedoids_n_clusters": 20,
    "kmedoids_metric": "euclidean",  # euclidean, manhattan, cosine
    "kmedoids_max_iter": 300,

    # ─────────────────────────────────────────────────────────────────────────
    # INTELLIGENT HYPERPARAMETER OPTIMIZATION (Optuna)
    # ─────────────────────────────────────────────────────────────────────────
    "use_optuna": True,  # Use Bayesian optimization instead of grid search
    "optuna_n_trials": 50,  # Number of optimization trials
    "optuna_timeout": 300,  # Max seconds for optimization
    "optuna_sampler": "tpe",  # tpe (default), cmaes, random

    # Hyperparameter search spaces (for Optuna)
    "hp_search_space": {
        "swing_threshold": (0.001, 0.006),  # Min/max for continuous
        "l2_C": (0.01, 10.0),  # Regularization strength (log scale)
        "gb_n_estimators": (30, 150),
        "gb_max_depth": (2, 5),
        "gb_learning_rate": (0.01, 0.3),
        "gb_min_samples_leaf": (20, 100),
        "gb_subsample": (0.6, 1.0),
    },

    # Anti-overfitting
    "n_cv_folds": 7,  # Increased from 5 for more robust validation
    "purge_days": 5,
    "embargo_days": 2,

    # Regularization (EDGE 1)
    "l1_alphas": [0.001, 0.01, 0.1, 1.0],
    "l2_alphas": [0.1, 1.0, 10.0],
    "elastic_l1_ratios": [0.5, 0.7, 0.9],

    # Soft targets (EDGE 4)
    "soft_target_k": 50,
    "label_smoothing_epsilon": 0.1,

    # Model constraints (EDGE 1)
    "max_tree_depth": 5,  # NEVER > 5
    "min_samples_leaf": 50,

    # ─────────────────────────────────────────────────────────────────────────
    # ANTI-OVERFITTING MEASURES
    # ─────────────────────────────────────────────────────────────────────────
    "use_anti_overfit": True,  # Enable anti-overfitting features
    "use_leak_proof_cv": True,  # Use leak-proof CV (fixes data leakage issues)
    "use_model_ensemble": True,  # Ensemble multiple models (reduces overfitting)
    "use_synthetic_universes": True,  # Generate "what SPY could have been"
    "use_cross_assets": True,  # Add TLT, QQQ, GLD, etc.
    "use_breadth_streaks": True,  # Add component streak breadth features
    "use_mag_breadth": True,  # Add MAG3/5/6/7/10/15 breadth features
    "use_sector_breadth": True,  # Add sector rotation/breadth features (11 sectors)
    "use_vol_regime": True,  # Add volatility regime features (VXX-based)
    "synthetic_weight": 0.3,  # 30% weight for synthetic data (conservative)
    "wmes_threshold": 0.55,  # Minimum weighted model evaluation score
    "stability_threshold": 0.5,  # Minimum stability score for hyperparameters

    # ─────────────────────────────────────────────────────────────────────────
    # ROBUSTNESS ENSEMBLE (dimension + parameter perturbation)
    # ─────────────────────────────────────────────────────────────────────────
    "use_robustness_ensemble": True,  # Train adjacent models for robustness
    "n_dimension_variants": 2,  # +/- this many dimensions (e.g., n-2, n-1, n, n+1, n+2)
    "n_param_variants": 2,  # Number of parameter perturbations to try
    "param_noise_pct": 0.05,  # +/- 5% noise on parameters
    "ensemble_center_weight": 0.5,  # Weight for optimal model (others split remaining)
    "fragility_threshold": 0.35,  # Warn if fragility > this (0=robust, 1=fragile)

    # ─────────────────────────────────────────────────────────────────────────
    # ENTRY/EXIT TIMING MODEL (predicts specific times, sizes, stops)
    # ─────────────────────────────────────────────────────────────────────────
    "train_entry_exit_model": True,  # Train ML model for entry/exit timing
    "entry_exit_model_type": "gradient_boosting",  # gradient_boosting, random_forest, ridge
    "entry_window": (0, 120),  # Entry window (minutes from open)
    "exit_window": (180, 385),  # Exit window (minutes from open)
    "min_position_pct": 0.05,  # Minimum position size (5%)
    "max_position_pct": 0.25,  # Maximum position size (25%)
}


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 1: DATA DOWNLOAD (5-10 years with missing bar handling)
# ═══════════════════════════════════════════════════════════════════════════════
def download_data() -> pd.DataFrame:
    """
    Download 5-10 years of 1-minute data with extended hours.

    Handles:
      - Chunked downloads (API rate limits)
      - Missing bar detection and flagging
      - Sparse premarket/afterhours in earlier years
      - Data quality validation
    """
    print("\n" + "=" * 70)
    print("STEP 1: DOWNLOAD HISTORICAL DATA")
    print("=" * 70)

    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    client = StockHistoricalDataClient(api_key, secret_key)

    years = CONFIG["years_to_download"]
    total_days = years * 365
    end_date = datetime.now()
    all_data = []
    current_end = end_date
    days_downloaded = 0
    chunk_count = 0

    print(f"[INFO] Downloading {years} years ({total_days} days) of 1-min data...")
    print(f"[INFO] This may take several minutes for large date ranges...")

    while days_downloaded < total_days:
        chunk_start = current_end - timedelta(days=CONFIG["chunk_days"])
        chunk_count += 1

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
                # Print progress every 6 chunks (~6 months)
                if chunk_count % 6 == 0 or chunk_count <= 3:
                    print(f"       {chunk_start.date()} to {current_end.date()}: {len(df_chunk):,} bars")

            current_end = chunk_start
            days_downloaded += CONFIG["chunk_days"]

        except Exception as e:
            print(f"[WARN] Error at {chunk_start.date()}: {e}")
            # Continue downloading even if one chunk fails
            current_end = chunk_start
            days_downloaded += CONFIG["chunk_days"]
            continue

    if len(all_data) == 0:
        raise ValueError("No data downloaded from Alpaca API")

    df = pd.concat(all_data, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    # Convert timezone
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York")
    df["date"] = df["timestamp"].dt.date
    df["time"] = df["timestamp"].dt.time
    df["hour"] = df["timestamp"].dt.hour
    df["minute"] = df["timestamp"].dt.minute

    # Session classification
    def get_session(row):
        h, m = row["hour"], row["minute"]
        if h < 9 or (h == 9 and m < 30):
            return "premarket"
        elif h < 16:
            return "regular"
        else:
            return "afterhours"

    df["session"] = df.apply(get_session, axis=1)

    print(f"\n[INFO] Raw download: {len(df):,} bars, {df['date'].nunique()} trading days")
    print(f"[INFO] Date range: {df['date'].min()} to {df['date'].max()}")

    # ─────────────────────────────────────────────────────────────────────────
    # MISSING BAR DETECTION AND HANDLING
    # ─────────────────────────────────────────────────────────────────────────
    df = detect_and_handle_missing_bars(df)

    return df


def detect_and_handle_missing_bars(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect and handle missing bars, especially in extended hours.

    Earlier years (pre-2020) often have sparse premarket/afterhours data.
    This function:
      1. Detects gaps in minute data
      2. Flags days with insufficient extended hours coverage
      3. Optionally forward-fills small gaps
      4. Creates quality flags for feature engineering
    """
    print("\n[MISSING DATA ANALYSIS]")

    df = df.copy()

    # Analyze by year
    df["year"] = pd.to_datetime(df["date"]).dt.year
    year_stats = []

    for year in sorted(df["year"].unique()):
        year_data = df[df["year"] == year]
        n_days = year_data["date"].nunique()

        pm_data = year_data[year_data["session"] == "premarket"]
        ah_data = year_data[year_data["session"] == "afterhours"]
        reg_data = year_data[year_data["session"] == "regular"]

        pm_bars_per_day = len(pm_data) / max(n_days, 1)
        ah_bars_per_day = len(ah_data) / max(n_days, 1)
        reg_bars_per_day = len(reg_data) / max(n_days, 1)

        year_stats.append({
            "year": year,
            "days": n_days,
            "pm_avg": pm_bars_per_day,
            "ah_avg": ah_bars_per_day,
            "reg_avg": reg_bars_per_day,
            "total_bars": len(year_data)
        })

        # Flag if extended hours data is sparse
        pm_flag = "OK" if pm_bars_per_day >= 50 else ("SPARSE" if pm_bars_per_day >= 10 else "MISSING")
        ah_flag = "OK" if ah_bars_per_day >= 30 else ("SPARSE" if ah_bars_per_day >= 5 else "MISSING")

        print(f"  {year}: {n_days} days | PM: {pm_bars_per_day:.0f}/day ({pm_flag}) | "
              f"AH: {ah_bars_per_day:.0f}/day ({ah_flag}) | Reg: {reg_bars_per_day:.0f}/day")

    # ─────────────────────────────────────────────────────────────────────────
    # CREATE QUALITY FLAGS PER DAY
    # ─────────────────────────────────────────────────────────────────────────
    day_quality = []

    for date in df["date"].unique():
        day_data = df[df["date"] == date]

        pm_bars = len(day_data[day_data["session"] == "premarket"])
        ah_bars = len(day_data[day_data["session"] == "afterhours"])
        reg_bars = len(day_data[day_data["session"] == "regular"])

        # Quality flags
        has_premarket = pm_bars >= CONFIG["min_premarket_bars"]
        has_afterhours = ah_bars >= CONFIG["min_afterhours_bars"]
        has_full_regular = reg_bars >= CONFIG["min_bars_per_day"]

        # Detect gaps in regular session
        reg_data = day_data[day_data["session"] == "regular"].sort_values("timestamp")
        if len(reg_data) > 1:
            time_diffs = reg_data["timestamp"].diff().dt.total_seconds() / 60
            max_gap = time_diffs.max() if len(time_diffs) > 0 else 0
            has_gaps = max_gap > CONFIG["max_gap_minutes"]
        else:
            has_gaps = True

        day_quality.append({
            "date": date,
            "pm_bars": pm_bars,
            "ah_bars": ah_bars,
            "reg_bars": reg_bars,
            "has_premarket": has_premarket,
            "has_afterhours": has_afterhours,
            "has_full_regular": has_full_regular,
            "has_gaps": has_gaps,
            "quality_score": (has_premarket + has_afterhours + has_full_regular + (not has_gaps)) / 4
        })

    quality_df = pd.DataFrame(day_quality)

    # Merge quality flags back
    df = df.merge(quality_df[["date", "has_premarket", "has_afterhours", "quality_score"]],
                  on="date", how="left")

    # Summary
    good_days = (quality_df["quality_score"] >= 0.75).sum()
    sparse_days = ((quality_df["quality_score"] >= 0.5) & (quality_df["quality_score"] < 0.75)).sum()
    poor_days = (quality_df["quality_score"] < 0.5).sum()

    print(f"\n  Data Quality Summary:")
    print(f"    Good quality (>=75%): {good_days} days")
    print(f"    Sparse data (50-75%): {sparse_days} days")
    print(f"    Poor quality (<50%):  {poor_days} days")

    # ─────────────────────────────────────────────────────────────────────────
    # OPTIONAL: FILL SMALL GAPS
    # ─────────────────────────────────────────────────────────────────────────
    if CONFIG["fill_missing_bars"]:
        # Forward-fill prices for gaps <= max_gap_minutes
        df = df.sort_values("timestamp")
        df["close"] = df["close"].ffill(limit=CONFIG["max_gap_minutes"])
        df["open"] = df["open"].ffill(limit=CONFIG["max_gap_minutes"])
        df["high"] = df["high"].ffill(limit=CONFIG["max_gap_minutes"])
        df["low"] = df["low"].ffill(limit=CONFIG["max_gap_minutes"])
        df["volume"] = df["volume"].fillna(0)

    print(f"\n[PASS] Processed {len(df):,} bars, {df['date'].nunique()} trading days")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 2: COMPREHENSIVE FEATURE ENGINEERING (EDGE 2 + EDGE 3)
# ═══════════════════════════════════════════════════════════════════════════════
def engineer_all_features(df: pd.DataFrame, swing_threshold: float = 0.003) -> pd.DataFrame:
    """
    Engineer comprehensive features following EDGE 2 and EDGE 3.

    Feature categories:
      1. Premarket features (today)
      2. Afterhours features (lagged 1-5 days)
      3. Overnight combined features
      4. Intraday technical indicators
      5. Price/volume patterns
      6. Momentum/trend features
      7. Volatility features
      8. Calendar features
    """
    print("\n" + "=" * 70)
    print("STEP 2: COMPREHENSIVE FEATURE ENGINEERING")
    print("=" * 70)

    # First compute 1-min technicals
    df = df.copy()

    # RSI
    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df["rsi_14"] = 100 - (100 / (1 + gain / (loss + 1e-10)))

    # MACD
    ema_12 = df["close"].ewm(span=12, adjust=False).mean()
    ema_26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = ema_12 - ema_26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    # Bollinger Bands
    df["bb_mid"] = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    df["bb_upper"] = df["bb_mid"] + 2 * bb_std
    df["bb_lower"] = df["bb_mid"] - 2 * bb_std
    df["bb_position"] = (df["close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"] + 1e-10)
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]

    # Stochastic
    low_14 = df["low"].rolling(14).min()
    high_14 = df["high"].rolling(14).max()
    df["stoch_k"] = 100 * (df["close"] - low_14) / (high_14 - low_14 + 1e-10)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()

    # ATR
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum(abs(df["high"] - df["close"].shift(1)),
                               abs(df["low"] - df["close"].shift(1))))
    df["atr_14"] = tr.rolling(14).mean()
    df["atr_pct"] = df["atr_14"] / df["close"]

    # Volume
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_sma_20"] + 1)

    # Momentum
    df["mom_5"] = df["close"].pct_change(5)
    df["mom_15"] = df["close"].pct_change(15)
    df["mom_30"] = df["close"].pct_change(30)

    # Now build daily features
    trading_days = sorted(df["date"].unique())
    daily_records = []

    print(f"[INFO] Processing {len(trading_days)} trading days...")

    for i, day in enumerate(trading_days):
        day_data = df[df["date"] == day].sort_values("timestamp")

        # Previous days for lagged features
        prev_days = [trading_days[i-j] if i >= j else None for j in range(1, 6)]
        prev_data = {j: df[df["date"] == prev_days[j-1]] if prev_days[j-1] else None for j in range(1, 6)}

        record = {"date": day}

        # ─────────────────────────────────────────────────────────────────────
        # PREMARKET FEATURES (EDGE 2) - Today
        # ─────────────────────────────────────────────────────────────────────
        pm_data = day_data[day_data["session"] == "premarket"]

        if len(pm_data) >= 5:
            pm_open = pm_data.iloc[0]["open"]
            pm_close = pm_data.iloc[-1]["close"]
            pm_high = pm_data["high"].max()
            pm_low = pm_data["low"].min()
            pm_volume = pm_data["volume"].sum()
            pm_vwap = (pm_data["close"] * pm_data["volume"]).sum() / (pm_volume + 1)

            record["pm_return"] = (pm_close - pm_open) / pm_open
            record["pm_range"] = (pm_high - pm_low) / pm_open
            record["pm_direction"] = 1 if record["pm_return"] > 0.001 else (-1 if record["pm_return"] < -0.001 else 0)
            record["pm_vwap_dev"] = (pm_close - pm_vwap) / pm_vwap
            record["pm_high_low_ratio"] = pm_high / (pm_low + 0.01)
            record["pm_volume"] = pm_volume
            record["pm_bar_count"] = len(pm_data)

            # Momentum within premarket
            pm_last_30 = pm_data.tail(30)
            pm_last_60 = pm_data.tail(60)
            record["pm_mom_30"] = (pm_last_30.iloc[-1]["close"] - pm_last_30.iloc[0]["open"]) / pm_last_30.iloc[0]["open"] if len(pm_last_30) > 1 else 0
            record["pm_mom_60"] = (pm_last_60.iloc[-1]["close"] - pm_last_60.iloc[0]["open"]) / pm_last_60.iloc[0]["open"] if len(pm_last_60) > 1 else 0

            # Technicals at PM close
            record["pm_rsi"] = pm_data.iloc[-1]["rsi_14"] if "rsi_14" in pm_data.columns else 50
            record["pm_macd_hist"] = pm_data.iloc[-1]["macd_hist"] if "macd_hist" in pm_data.columns else 0
            record["pm_bb_pos"] = pm_data.iloc[-1]["bb_position"] if "bb_position" in pm_data.columns else 0.5
            record["pm_stoch_k"] = pm_data.iloc[-1]["stoch_k"] if "stoch_k" in pm_data.columns else 50

            # Trend consistency
            pm_returns = pm_data["close"].pct_change().dropna()
            record["pm_trend_consistency"] = (pm_returns > 0).mean() if len(pm_returns) > 0 else 0.5
        else:
            for key in ["pm_return", "pm_range", "pm_direction", "pm_vwap_dev", "pm_high_low_ratio",
                        "pm_volume", "pm_bar_count", "pm_mom_30", "pm_mom_60", "pm_rsi",
                        "pm_macd_hist", "pm_bb_pos", "pm_stoch_k", "pm_trend_consistency"]:
                record[key] = 0
            pm_close = None

        # ─────────────────────────────────────────────────────────────────────
        # AFTERHOURS FEATURES (EDGE 2) - Lagged 1-5 days
        # ─────────────────────────────────────────────────────────────────────
        for lag in [1, 2, 3, 5]:
            pd_data = prev_data.get(lag)
            if pd_data is not None and len(pd_data) > 0:
                ah_data = pd_data[pd_data["session"] == "afterhours"]
                if len(ah_data) >= 5:
                    ah_open = ah_data.iloc[0]["open"]
                    ah_close = ah_data.iloc[-1]["close"]
                    ah_high = ah_data["high"].max()
                    ah_low = ah_data["low"].min()
                    ah_volume = ah_data["volume"].sum()

                    record[f"ah_return_lag{lag}"] = (ah_close - ah_open) / ah_open
                    record[f"ah_range_lag{lag}"] = (ah_high - ah_low) / ah_open
                    record[f"ah_direction_lag{lag}"] = 1 if record[f"ah_return_lag{lag}"] > 0.001 else (-1 if record[f"ah_return_lag{lag}"] < -0.001 else 0)
                    record[f"ah_volume_lag{lag}"] = ah_volume
                else:
                    record[f"ah_return_lag{lag}"] = 0
                    record[f"ah_range_lag{lag}"] = 0
                    record[f"ah_direction_lag{lag}"] = 0
                    record[f"ah_volume_lag{lag}"] = 0
            else:
                record[f"ah_return_lag{lag}"] = 0
                record[f"ah_range_lag{lag}"] = 0
                record[f"ah_direction_lag{lag}"] = 0
                record[f"ah_volume_lag{lag}"] = 0

        # ─────────────────────────────────────────────────────────────────────
        # PREMARKET LAGGED (EDGE 2)
        # ─────────────────────────────────────────────────────────────────────
        for lag in [1, 2, 3, 5]:
            pd_data = prev_data.get(lag)
            if pd_data is not None and len(pd_data) > 0:
                pm_prev = pd_data[pd_data["session"] == "premarket"]
                if len(pm_prev) >= 5:
                    pm_prev_open = pm_prev.iloc[0]["open"]
                    pm_prev_close = pm_prev.iloc[-1]["close"]
                    record[f"pm_return_lag{lag}"] = (pm_prev_close - pm_prev_open) / pm_prev_open
                    record[f"pm_direction_lag{lag}"] = 1 if record[f"pm_return_lag{lag}"] > 0.001 else (-1 if record[f"pm_return_lag{lag}"] < -0.001 else 0)
                else:
                    record[f"pm_return_lag{lag}"] = 0
                    record[f"pm_direction_lag{lag}"] = 0
            else:
                record[f"pm_return_lag{lag}"] = 0
                record[f"pm_direction_lag{lag}"] = 0

        # ─────────────────────────────────────────────────────────────────────
        # OVERNIGHT COMBINED (EDGE 2)
        # ─────────────────────────────────────────────────────────────────────
        prev_regular = prev_data.get(1)
        if prev_regular is not None:
            prev_reg = prev_regular[prev_regular["session"] == "regular"]
            if len(prev_reg) > 0:
                prev_close = prev_reg.iloc[-1]["close"]
                if pm_close is not None:
                    record["overnight_return"] = (pm_close - prev_close) / prev_close
                else:
                    record["overnight_return"] = 0
            else:
                record["overnight_return"] = 0
                prev_close = None
        else:
            record["overnight_return"] = 0
            prev_close = None

        # Direction agreement
        record["pm_ah_agree"] = 1 if (record.get("pm_direction", 0) == record.get("ah_direction_lag1", 0) and record.get("pm_direction", 0) != 0) else 0

        # Extended hours streak
        pm_dirs = [record.get(f"pm_direction_lag{lag}", 0) for lag in [1, 2, 3]]
        record["pm_direction_streak"] = sum(1 for d in pm_dirs if d == record.get("pm_direction", 0) and d != 0)

        # ─────────────────────────────────────────────────────────────────────
        # REGULAR SESSION FEATURES
        # ─────────────────────────────────────────────────────────────────────
        regular_data = day_data[day_data["session"] == "regular"]

        if len(regular_data) < 50:
            continue

        reg_open = regular_data.iloc[0]["open"]
        reg_close = regular_data.iloc[-1]["close"]
        reg_high = regular_data["high"].max()
        reg_low = regular_data["low"].min()
        reg_volume = regular_data["volume"].sum()

        record["day_return"] = (reg_close - reg_open) / reg_open
        record["day_range"] = (reg_high - reg_low) / reg_open
        record["day_volume"] = reg_volume

        # Gap from previous close
        if prev_close is not None:
            record["gap_pct"] = (reg_open - prev_close) / prev_close
        else:
            record["gap_pct"] = 0

        # ─────────────────────────────────────────────────────────────────────
        # INTRADAY FEATURES AT KEY TIMES (EDGE 3)
        # ─────────────────────────────────────────────────────────────────────
        key_times = {
            "0945": time(9, 45),
            "1015": time(10, 15),
            "1100": time(11, 0),
            "1130": time(11, 30),
            "1230": time(12, 30),
            "1330": time(13, 30),
            "1430": time(14, 30),
            "1530": time(15, 30),
        }

        for time_label, t in key_times.items():
            data_to_time = regular_data[regular_data["time"] <= t]
            if len(data_to_time) > 0:
                record[f"return_at_{time_label}"] = (data_to_time.iloc[-1]["close"] - reg_open) / reg_open
                record[f"high_to_{time_label}"] = (data_to_time["high"].max() - reg_open) / reg_open
                record[f"low_to_{time_label}"] = (data_to_time["low"].min() - reg_open) / reg_open
                record[f"range_to_{time_label}"] = record[f"high_to_{time_label}"] - record[f"low_to_{time_label}"]

                # Technicals at time
                record[f"rsi_at_{time_label}"] = data_to_time.iloc[-1]["rsi_14"] if "rsi_14" in data_to_time.columns else 50
                record[f"macd_at_{time_label}"] = data_to_time.iloc[-1]["macd_hist"] if "macd_hist" in data_to_time.columns else 0
                record[f"bb_at_{time_label}"] = data_to_time.iloc[-1]["bb_position"] if "bb_position" in data_to_time.columns else 0.5

                # Return from intraday low
                low_to_time = data_to_time["low"].min()
                record[f"return_from_low_{time_label}"] = (data_to_time.iloc[-1]["close"] - low_to_time) / low_to_time if low_to_time > 0 else 0
            else:
                for suffix in ["return_at_", "high_to_", "low_to_", "range_to_", "rsi_at_", "macd_at_", "bb_at_", "return_from_low_"]:
                    record[f"{suffix}{time_label}"] = 0

        # ─────────────────────────────────────────────────────────────────────
        # DAILY LAGGED FEATURES
        # ─────────────────────────────────────────────────────────────────────
        for lag in [1, 2, 3, 5, 10]:
            pd_data = prev_data.get(min(lag, 5))  # We only have up to 5 days
            if pd_data is not None:
                prev_reg = pd_data[pd_data["session"] == "regular"]
                if len(prev_reg) > 0:
                    pr_open = prev_reg.iloc[0]["open"]
                    pr_close = prev_reg.iloc[-1]["close"]
                    pr_high = prev_reg["high"].max()
                    pr_low = prev_reg["low"].min()
                    record[f"day_return_lag{lag}"] = (pr_close - pr_open) / pr_open
                    record[f"day_range_lag{lag}"] = (pr_high - pr_low) / pr_open
                else:
                    record[f"day_return_lag{lag}"] = 0
                    record[f"day_range_lag{lag}"] = 0
            else:
                record[f"day_return_lag{lag}"] = 0
                record[f"day_range_lag{lag}"] = 0

        # ─────────────────────────────────────────────────────────────────────
        # SWING DETECTION (TARGETS)
        # ─────────────────────────────────────────────────────────────────────
        record["is_up_day"] = record["day_return"] > swing_threshold
        record["is_down_day"] = record["day_return"] < -swing_threshold

        # Timing: when did high/low occur?
        high_idx = regular_data["high"].idxmax()
        low_idx = regular_data["low"].idxmin()
        high_time = regular_data.loc[high_idx, "timestamp"]
        low_time = regular_data.loc[low_idx, "timestamp"]
        open_time = regular_data.iloc[0]["timestamp"]

        record["high_minutes"] = (high_time - open_time).total_seconds() / 60
        record["low_minutes"] = (low_time - open_time).total_seconds() / 60
        record["low_before_high"] = low_time < high_time

        # Max gains from different entry points
        for time_label, t in [("1015", time(10, 15)), ("1230", time(12, 30))]:
            data_from = regular_data[regular_data["time"] >= t]
            data_to = regular_data[regular_data["time"] <= t]
            if len(data_from) > 0 and len(data_to) > 0:
                entry_price = data_to.iloc[-1]["close"]
                max_high = data_from["high"].max()
                record[f"max_gain_from_{time_label}"] = (max_high - entry_price) / entry_price
            else:
                record[f"max_gain_from_{time_label}"] = 0

        daily_records.append(record)

    result_df = pd.DataFrame(daily_records)
    result_df["date"] = pd.to_datetime(result_df["date"])

    print(f"[PASS] Engineered {len(result_df.columns)} features for {len(result_df)} days")

    return result_df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 3: ROLLING WINDOW FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling window statistics."""
    print("\n" + "=" * 70)
    print("STEP 3: ADD ROLLING WINDOW FEATURES")
    print("=" * 70)

    df = df.sort_values("date").copy()

    # Rolling returns
    for window in [3, 5, 10, 20]:
        df[f"return_ma{window}"] = df["day_return"].shift(1).rolling(window).mean()
        df[f"return_std{window}"] = df["day_return"].shift(1).rolling(window).std()

    # Rolling range
    for window in [5, 10]:
        df[f"range_ma{window}"] = df["day_range"].shift(1).rolling(window).mean()

    # Rolling premarket
    df["pm_return_ma3"] = df["pm_return"].shift(1).rolling(3).mean()
    df["pm_return_ma5"] = df["pm_return"].shift(1).rolling(5).mean()

    # Consecutive patterns
    df["up_streak"] = df["is_up_day"].shift(1).rolling(5).sum()
    df["down_streak"] = df["is_down_day"].shift(1).rolling(5).sum()

    # Volatility ratio
    df["vol_ratio_5_20"] = df["return_std5"] / (df["return_std20"] + 1e-10)

    # Mean reversion indicator
    df["return_zscore_20"] = (df["day_return"].shift(1) - df["return_ma20"]) / (df["return_std20"] + 1e-10)

    print(f"[PASS] Added rolling features. Total features: {len(df.columns)}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 4: ADVANCED DIMENSIONALITY REDUCTION
# ═══════════════════════════════════════════════════════════════════════════════
def reduce_dimensions(X: np.ndarray, feature_names: list, y: np.ndarray = None,
                      fit: bool = True, state: dict = None):
    """
    Advanced dimensionality reduction for complex financial features.

    Methods (selected via CONFIG["dim_reduction_method"]):
      - "umap": Uniform Manifold Approximation (preserves non-linear structure)
      - "kernel_pca": Kernel PCA with RBF (captures non-linear relationships)
      - "ica": Independent Component Analysis (finds independent signals)
      - "mutual_info": Mutual Information feature selection (non-linear relevance)
      - "agglomeration": Feature Agglomeration (hierarchical clustering)
      - "ensemble": Combines multiple methods (recommended for trading)

    Pipeline:
      1. Remove low variance features
      2. Remove highly correlated features
      3. Apply selected advanced method
    """
    from sklearn.feature_selection import VarianceThreshold, mutual_info_classif
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import KernelPCA, FastICA
    from sklearn.cluster import FeatureAgglomeration

    if state is None:
        state = {}

    method = CONFIG["dim_reduction_method"]

    if fit:
        print("\n" + "=" * 70)
        print("STEP 4: ADVANCED DIMENSIONALITY REDUCTION")
        print("=" * 70)
        print(f"[INFO] Method: {method.upper()}")
        original_features = X.shape[1]

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 1: Variance Threshold (remove near-constant features)
        # ─────────────────────────────────────────────────────────────────────
        var_selector = VarianceThreshold(threshold=CONFIG["variance_threshold"])
        X = var_selector.fit_transform(X)
        kept_mask = var_selector.get_support()
        feature_names = [f for f, k in zip(feature_names, kept_mask) if k]
        print(f"  [Stage 1] Variance threshold: {X.shape[1]} features (removed {original_features - X.shape[1]})")

        state["var_selector"] = var_selector
        state["var_kept_mask"] = kept_mask
        stage1_features = X.shape[1]

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 2: Correlation Filter (remove redundant features)
        # ─────────────────────────────────────────────────────────────────────
        corr_matrix = np.corrcoef(X.T)
        to_drop = set()
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                if abs(corr_matrix[i, j]) > CONFIG["correlation_threshold"]:
                    to_drop.add(j)

        keep_idx = [i for i in range(X.shape[1]) if i not in to_drop]
        X = X[:, keep_idx]
        feature_names = [feature_names[i] for i in keep_idx]
        print(f"  [Stage 2] Correlation filter: {X.shape[1]} features (removed {len(to_drop)})")

        state["corr_keep_idx"] = keep_idx
        state["feature_names_pre_transform"] = feature_names.copy()

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 3: Standardize before advanced methods
        # ─────────────────────────────────────────────────────────────────────
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        state["pre_transform_scaler"] = scaler

        # ─────────────────────────────────────────────────────────────────────
        # STAGE 4: Advanced Dimensionality Reduction
        # ─────────────────────────────────────────────────────────────────────
        print(f"\n  [Stage 3] Applying {method.upper()} transformation...")

        if method == "umap":
            X_transformed, feature_names, state = _apply_umap(X_scaled, feature_names, state)

        elif method == "kernel_pca":
            X_transformed, feature_names, state = _apply_kernel_pca(X_scaled, feature_names, state)

        elif method == "ica":
            X_transformed, feature_names, state = _apply_ica(X_scaled, feature_names, state)

        elif method == "mutual_info":
            if y is None:
                print("  [WARN] mutual_info requires target y, falling back to correlation filter only")
                X_transformed = X_scaled
            else:
                X_transformed, feature_names, state = _apply_mutual_info(X_scaled, y, feature_names, state)

        elif method == "agglomeration":
            X_transformed, feature_names, state = _apply_agglomeration(X_scaled, feature_names, state)

        elif method == "ensemble":
            # Ensemble combines multiple methods for robustness
            X_transformed, feature_names, state = _apply_ensemble(X_scaled, y, feature_names, state)

        elif method == "kmedoids":
            X_transformed, feature_names, state = _apply_kmedoids(X_scaled, feature_names, state)

        elif method == "ensemble_plus":
            # Enhanced ensemble with K-Medoids
            X_transformed, feature_names, state = _apply_ensemble_plus(X_scaled, y, feature_names, state)

        else:
            print(f"  [WARN] Unknown method '{method}', using raw filtered features")
            X_transformed = X_scaled
            state["transform_method"] = "none"

        print(f"  [FINAL] Output dimensions: {X_transformed.shape[1]}")
        state["feature_names"] = feature_names
        state["method"] = method

        return X_transformed, feature_names, state

    else:
        # Transform only (for inference)
        # Safely get required transformers with helpful error messages
        var_selector = state.get("var_selector")
        if var_selector is None:
            raise ValueError("dim_state missing 'var_selector' - was model trained correctly?")
        X = var_selector.transform(X)

        corr_keep_idx = state.get("corr_keep_idx")
        if corr_keep_idx is None:
            # Fallback: keep all features if correlation filter wasn't applied
            corr_keep_idx = list(range(X.shape[1]))
        X = X[:, corr_keep_idx]

        pre_transform_scaler = state.get("pre_transform_scaler")
        if pre_transform_scaler is None:
            raise ValueError("dim_state missing 'pre_transform_scaler' - was model trained correctly?")
        X_scaled = pre_transform_scaler.transform(X)

        method = state.get("method", "none")

        if method == "umap":
            X_transformed = state["umap_reducer"].transform(X_scaled)
        elif method == "kernel_pca" or method == "pca_fallback":
            if state.get("kpca_use_nystroem", False):
                # Nystroem approximation path
                X_approx = state["nystroem_transformer"].transform(X_scaled)
                X_transformed = state["kpca_pca"].transform(X_approx)
            else:
                X_transformed = state["kpca_transformer"].transform(X_scaled)
        elif method == "ica":
            X_transformed = state["ica_transformer"].transform(X_scaled)
        elif method == "mutual_info":
            X_transformed = X_scaled[:, state["mi_selected_idx"]]
        elif method == "agglomeration":
            X_transformed = state["agglom_transformer"].transform(X_scaled)
        elif method == "ensemble":
            X_transformed = _apply_ensemble_transform(X_scaled, state)
        elif method == "ensemble_plus":
            # ensemble_plus uses the same transform function (includes K-Medoids)
            X_transformed = _apply_ensemble_transform(X_scaled, state)
        elif method == "kmedoids":
            # For kmedoids, assign to nearest medoid
            components = state.get("ensemble_components", {})
            if "kmedoids_medoids" in components:
                medoids = components["kmedoids_medoids"]
                distances = np.array([[np.linalg.norm(x - m) for m in medoids] for x in X_scaled])
                X_transformed = distances
            else:
                X_transformed = X_scaled
        else:
            X_transformed = X_scaled

        return X_transformed, state.get("feature_names", []), state


def _apply_umap(X: np.ndarray, feature_names: list, state: dict):
    """
    UMAP - Uniform Manifold Approximation and Projection.

    Why UMAP for trading:
      - Preserves BOTH local and global structure (unlike t-SNE)
      - Handles non-linear manifolds well
      - Much faster than t-SNE for large datasets
      - Better at preserving inter-cluster distances
    """
    try:
        import umap
    except ImportError:
        print("  [ERROR] umap-learn not installed. Run: pip install umap-learn")
        state["transform_method"] = "none"
        return X, feature_names, state

    n_components = min(CONFIG["umap_n_components"], X.shape[1] - 1, X.shape[0] - 1)

    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=CONFIG["umap_n_neighbors"],
        min_dist=CONFIG["umap_min_dist"],
        metric=CONFIG["umap_metric"],
        random_state=42,
        n_jobs=-1
    )

    X_transformed = reducer.fit_transform(X)
    state["umap_reducer"] = reducer
    state["transform_method"] = "umap"

    new_feature_names = [f"umap_{i}" for i in range(X_transformed.shape[1])]
    print(f"    UMAP: {X.shape[1]} -> {X_transformed.shape[1]} components")
    print(f"    (n_neighbors={CONFIG['umap_n_neighbors']}, min_dist={CONFIG['umap_min_dist']})")

    return X_transformed, new_feature_names, state


def _apply_kernel_pca(X: np.ndarray, feature_names: list, state: dict):
    """
    Kernel PCA - Non-linear PCA using kernel trick.

    Why Kernel PCA for trading:
      - Captures non-linear relationships that PCA misses
      - RBF kernel is good for smooth, complex manifolds
      - Polynomial kernel captures interaction effects
      - More interpretable than deep methods

    For large datasets (>5000 samples), uses Nystroem approximation to avoid memory issues.
    """
    from sklearn.decomposition import KernelPCA, PCA
    from sklearn.kernel_approximation import Nystroem

    n_components = min(CONFIG["kpca_n_components"], X.shape[1] - 1)
    n_samples = X.shape[0]

    # For large datasets, use Nystroem approximation to avoid O(n^2) memory
    MAX_SAMPLES_FOR_EXACT = 5000

    if n_samples > MAX_SAMPLES_FOR_EXACT:
        print(f"    Using Nystroem approximation ({n_samples} samples > {MAX_SAMPLES_FOR_EXACT})")
        try:
            # Use Nystroem to approximate the kernel map, then PCA
            n_nystroem_components = min(1000, n_samples // 5, X.shape[1] * 3)
            nystroem = Nystroem(
                kernel=CONFIG["kpca_kernel"],
                gamma=CONFIG["kpca_gamma"],
                n_components=n_nystroem_components,
                random_state=42,
                n_jobs=-1
            )
            X_approx = nystroem.fit_transform(X)

            # Then apply regular PCA
            pca = PCA(n_components=n_components, random_state=42)
            X_transformed = pca.fit_transform(X_approx)

            state["nystroem_transformer"] = nystroem
            state["kpca_pca"] = pca
            state["kpca_use_nystroem"] = True
            state["transform_method"] = "kernel_pca"

            new_feature_names = [f"kpca_{i}" for i in range(X_transformed.shape[1])]
            print(f"    Nystroem + PCA: {X.shape[1]} -> {n_nystroem_components} -> {X_transformed.shape[1]} components")

            return X_transformed, new_feature_names, state

        except Exception as nystroem_error:
            print(f"    Nystroem failed: {nystroem_error}, falling back to PCA")
            pca = PCA(n_components=n_components, random_state=42)
            X_transformed = pca.fit_transform(X)
            state["kpca_transformer"] = pca
            state["kpca_use_nystroem"] = False
            state["transform_method"] = "pca_fallback"

            new_feature_names = [f"kpca_{i}" for i in range(X_transformed.shape[1])]
            return X_transformed, new_feature_names, state
    else:
        # Standard Kernel PCA for smaller datasets
        kpca = KernelPCA(
            n_components=n_components,
            kernel=CONFIG["kpca_kernel"],
            gamma=CONFIG["kpca_gamma"],
            fit_inverse_transform=False,
            random_state=42,
            n_jobs=-1
        )

        X_transformed = kpca.fit_transform(X)
        state["kpca_transformer"] = kpca
        state["kpca_use_nystroem"] = False
        state["transform_method"] = "kernel_pca"

        new_feature_names = [f"kpca_{i}" for i in range(X_transformed.shape[1])]
        print(f"    Kernel PCA ({CONFIG['kpca_kernel']}): {X.shape[1]} -> {X_transformed.shape[1]} components")
        print(f"    (gamma={CONFIG['kpca_gamma']})")

        return X_transformed, new_feature_names, state


def _apply_ica(X: np.ndarray, feature_names: list, state: dict):
    """
    ICA - Independent Component Analysis.

    Why ICA for trading:
      - Finds statistically INDEPENDENT signals (unlike PCA which finds uncorrelated)
      - Good for separating mixed signals (market, sector, idiosyncratic)
      - Robust to non-Gaussian distributions common in finance
      - Captures higher-order statistics beyond correlation
    """
    from sklearn.decomposition import FastICA

    n_components = min(CONFIG["ica_n_components"], X.shape[1] - 1)

    ica = FastICA(
        n_components=n_components,
        max_iter=CONFIG["ica_max_iter"],
        random_state=42,
        whiten="unit-variance"
    )

    try:
        X_transformed = ica.fit_transform(X)
        state["ica_transformer"] = ica
        state["transform_method"] = "ica"
        new_feature_names = [f"ica_{i}" for i in range(X_transformed.shape[1])]
        print(f"    ICA: {X.shape[1]} -> {X_transformed.shape[1]} independent components")
    except Exception as e:
        print(f"    [WARN] ICA failed ({e}), using original features")
        X_transformed = X
        new_feature_names = feature_names
        state["transform_method"] = "ica_failed"

    return X_transformed, new_feature_names, state


def _apply_mutual_info(X: np.ndarray, y: np.ndarray, feature_names: list, state: dict):
    """
    Mutual Information Feature Selection.

    Why MI for trading:
      - Captures NON-LINEAR dependencies (unlike correlation)
      - No assumptions about distribution shape
      - Directly measures predictive relevance to target
      - Works well with mixed feature types
    """
    from sklearn.feature_selection import mutual_info_classif

    n_features = min(CONFIG["mi_n_features"], X.shape[1])

    # Compute mutual information scores
    mi_scores = mutual_info_classif(
        X, y,
        n_neighbors=CONFIG["mi_n_neighbors"],
        random_state=42
    )

    # Select top features
    top_idx = np.argsort(mi_scores)[::-1][:n_features]
    X_transformed = X[:, top_idx]
    selected_features = [feature_names[i] for i in top_idx]

    state["mi_scores"] = dict(zip(feature_names, mi_scores))
    state["mi_selected_idx"] = top_idx
    state["transform_method"] = "mutual_info"

    print(f"    Mutual Info: {X.shape[1]} -> {X_transformed.shape[1]} features")
    print(f"    Top 5 by MI: {selected_features[:5]}")

    return X_transformed, selected_features, state


def _apply_agglomeration(X: np.ndarray, feature_names: list, state: dict):
    """
    Feature Agglomeration - Hierarchical clustering of features.

    Why Agglomeration for trading:
      - Groups similar/correlated features into clusters
      - Reduces redundancy while preserving information
      - Creates interpretable "super-features"
      - Works well when features have hierarchical relationships
    """
    from sklearn.cluster import FeatureAgglomeration

    n_clusters = min(CONFIG["agglom_n_clusters"], X.shape[1])

    agglom = FeatureAgglomeration(
        n_clusters=n_clusters,
        linkage="ward"
    )

    X_transformed = agglom.fit_transform(X)
    state["agglom_transformer"] = agglom
    state["transform_method"] = "agglomeration"

    new_feature_names = [f"cluster_{i}" for i in range(X_transformed.shape[1])]
    print(f"    Feature Agglomeration: {X.shape[1]} -> {X_transformed.shape[1]} clusters")

    # Show which features went into which cluster
    labels = agglom.labels_
    for c in range(min(5, n_clusters)):
        cluster_features = [f for f, l in zip(feature_names, labels) if l == c]
        print(f"      Cluster {c}: {cluster_features[:3]}...")

    return X_transformed, new_feature_names, state


def _apply_ensemble(X: np.ndarray, y: np.ndarray, feature_names: list, state: dict):
    """
    Ensemble Dimensionality Reduction - Combines multiple methods.

    Strategy:
      1. Use Mutual Info to select most predictive features (non-linear relevance)
      2. Apply Kernel PCA on selected features (capture non-linear structure)
      3. Add ICA components (independent signals)
      4. Concatenate for robust representation

    Why Ensemble:
      - Different methods capture different aspects of the data
      - More robust than single method
      - MI ensures relevance, KernelPCA/ICA capture structure
    """
    print("    [ENSEMBLE] Combining multiple methods for robustness...")

    state["ensemble_components"] = {}

    # Component 1: Mutual Info selected features (if y available)
    mi_features = min(15, X.shape[1] // 2) if y is not None else 0
    kpca_components = 10
    ica_components = 5

    ensemble_parts = []
    ensemble_names = []

    # Part 1: MI-selected original features
    if y is not None and mi_features > 0:
        from sklearn.feature_selection import mutual_info_classif
        mi_scores = mutual_info_classif(X, y, n_neighbors=5, random_state=42)
        mi_top_idx = np.argsort(mi_scores)[::-1][:mi_features]
        X_mi = X[:, mi_top_idx]
        ensemble_parts.append(X_mi)
        ensemble_names.extend([f"mi_{feature_names[i]}" for i in mi_top_idx])
        state["ensemble_components"]["mi_idx"] = mi_top_idx
        state["ensemble_components"]["mi_scores"] = dict(zip(feature_names, mi_scores))
        print(f"      Part 1: MI selection -> {X_mi.shape[1]} features")

    # Part 2: Kernel PCA components (with Nystroem for large datasets)
    if X.shape[1] >= kpca_components and X.shape[0] > kpca_components:
        MAX_SAMPLES_FOR_EXACT_KPCA = 5000

        try:
            if X.shape[0] > MAX_SAMPLES_FOR_EXACT_KPCA:
                # Use Nystroem approximation for large datasets to avoid O(n^2) memory
                from sklearn.kernel_approximation import Nystroem
                from sklearn.decomposition import PCA

                n_nystroem = min(500, X.shape[0] // 10)
                nystroem = Nystroem(kernel="rbf", gamma=0.01, n_components=n_nystroem, random_state=42)
                X_approx = nystroem.fit_transform(X)

                pca = PCA(n_components=kpca_components, random_state=42)
                X_kpca = pca.fit_transform(X_approx)

                ensemble_parts.append(X_kpca)
                ensemble_names.extend([f"kpca_{i}" for i in range(X_kpca.shape[1])])
                state["ensemble_components"]["kpca_nystroem"] = nystroem
                state["ensemble_components"]["kpca_pca"] = pca
                state["ensemble_components"]["kpca_use_nystroem"] = True
                print(f"      Part 2: Nystroem+PCA -> {X_kpca.shape[1]} components (approx)")
            else:
                # Standard Kernel PCA for smaller datasets
                from sklearn.decomposition import KernelPCA
                kpca = KernelPCA(n_components=kpca_components, kernel="rbf", gamma=0.01, random_state=42)
                X_kpca = kpca.fit_transform(X)
                ensemble_parts.append(X_kpca)
                ensemble_names.extend([f"kpca_{i}" for i in range(X_kpca.shape[1])])
                state["ensemble_components"]["kpca"] = kpca
                state["ensemble_components"]["kpca_use_nystroem"] = False
                print(f"      Part 2: Kernel PCA -> {X_kpca.shape[1]} components")
        except Exception as e:
            print(f"      Part 2: Kernel PCA failed ({e})")

    # Part 3: ICA components
    if X.shape[1] >= ica_components:
        from sklearn.decomposition import FastICA
        ica = FastICA(n_components=ica_components, max_iter=500, random_state=42, whiten="unit-variance")
        try:
            X_ica = ica.fit_transform(X)
            ensemble_parts.append(X_ica)
            ensemble_names.extend([f"ica_{i}" for i in range(X_ica.shape[1])])
            state["ensemble_components"]["ica"] = ica
            print(f"      Part 3: ICA -> {X_ica.shape[1]} components")
        except Exception as e:
            print(f"      Part 3: ICA failed ({e})")

    if len(ensemble_parts) == 0:
        print("      [WARN] No ensemble components created, using original")
        X_transformed = X
        ensemble_names = feature_names
    else:
        X_transformed = np.hstack(ensemble_parts)
        print(f"    [ENSEMBLE] Combined: {X_transformed.shape[1]} total dimensions")

    state["transform_method"] = "ensemble"
    return X_transformed, ensemble_names, state


def _apply_ensemble_transform(X: np.ndarray, state: dict) -> np.ndarray:
    """Apply ensemble transform for inference."""
    components = state.get("ensemble_components", {})
    parts = []

    # Part 1: MI-selected features
    if "mi_idx" in components:
        parts.append(X[:, components["mi_idx"]])

    # Part 2: Kernel PCA (handle both standard and Nystroem)
    if components.get("kpca_use_nystroem", False):
        try:
            X_approx = components["kpca_nystroem"].transform(X)
            parts.append(components["kpca_pca"].transform(X_approx))
        except (ValueError, RuntimeError) as e:
            print(f"  [WARN] Nystroem KPCA transform failed: {e}")
    elif "kpca" in components:
        try:
            parts.append(components["kpca"].transform(X))
        except (ValueError, RuntimeError) as e:
            print(f"  [WARN] KPCA transform failed: {e}")

    # Part 3: ICA
    if "ica" in components:
        try:
            parts.append(components["ica"].transform(X))
        except (ValueError, RuntimeError) as e:
            print(f"  [WARN] ICA transform failed: {e}")

    # Part 4: K-Medoids (if present in ensemble_plus)
    if "kmedoids_labels" in components:
        # For inference, assign to nearest medoid
        medoids = components["kmedoids_medoids"]
        distances = np.array([[np.linalg.norm(x - m) for m in medoids] for x in X])
        parts.append(distances)

    if len(parts) == 0:
        return X

    return np.hstack(parts)


def _apply_kmedoids(X: np.ndarray, feature_names: list, state: dict):
    """
    K-Medoids Clustering for feature selection/reduction.

    Why K-Medoids for trading:
      - More robust to outliers than K-Means (uses actual data points as centers)
      - Better for non-spherical clusters
      - Medoids are interpretable (actual feature vectors)
      - Works well with various distance metrics
    """
    try:
        from sklearn_extra.cluster import KMedoids
    except ImportError:
        print("  [ERROR] scikit-learn-extra not installed. Run: pip install scikit-learn-extra")
        state["transform_method"] = "none"
        return X, feature_names, state

    n_clusters = min(CONFIG["kmedoids_n_clusters"], X.shape[1] - 1, X.shape[0] // 5)

    kmedoids = KMedoids(
        n_clusters=n_clusters,
        metric=CONFIG["kmedoids_metric"],
        max_iter=CONFIG["kmedoids_max_iter"],
        random_state=42
    )

    # Cluster the samples (rows), then use distances to medoids as features
    kmedoids.fit(X)

    # Transform: distance from each sample to each medoid
    medoid_indices = kmedoids.medoid_indices_
    medoids = X[medoid_indices]
    X_transformed = np.array([[np.linalg.norm(x - m) for m in medoids] for x in X])

    state["kmedoids_model"] = kmedoids
    state["kmedoids_medoids"] = medoids
    state["kmedoids_labels"] = kmedoids.labels_
    state["transform_method"] = "kmedoids"

    new_feature_names = [f"medoid_dist_{i}" for i in range(n_clusters)]
    print(f"    K-Medoids: {X.shape[1]} features -> {n_clusters} medoid distances")
    print(f"    (metric={CONFIG['kmedoids_metric']}, iterations={CONFIG['kmedoids_max_iter']})")

    return X_transformed, new_feature_names, state


def _apply_ensemble_plus(X: np.ndarray, y: np.ndarray, feature_names: list, state: dict):
    """
    Enhanced Ensemble - Combines MI + Kernel PCA + ICA + K-Medoids.

    Improvements over basic ensemble:
      1. K-Medoids for outlier-robust clustering
      2. More components for richer representation
      3. Quality-weighted combination
    """
    print("    [ENSEMBLE+] Advanced multi-method combination...")

    state["ensemble_components"] = {}

    # Component counts (adjusted for larger datasets)
    mi_features = min(20, X.shape[1] // 3) if y is not None else 0
    kpca_components = 12
    ica_components = 8
    kmedoids_clusters = min(10, X.shape[0] // 20)

    ensemble_parts = []
    ensemble_names = []

    # Part 1: MI-selected original features
    if y is not None and mi_features > 0:
        from sklearn.feature_selection import mutual_info_classif
        mi_scores = mutual_info_classif(X, y, n_neighbors=5, random_state=42)
        mi_top_idx = np.argsort(mi_scores)[::-1][:mi_features]
        X_mi = X[:, mi_top_idx]
        ensemble_parts.append(X_mi)
        ensemble_names.extend([f"mi_{feature_names[i]}" for i in mi_top_idx])
        state["ensemble_components"]["mi_idx"] = mi_top_idx
        state["ensemble_components"]["mi_scores"] = dict(zip(feature_names, mi_scores))
        print(f"      Part 1: MI selection -> {X_mi.shape[1]} features")

    # Part 2: Kernel PCA components (with Nystroem for large datasets)
    if X.shape[1] >= kpca_components and X.shape[0] > kpca_components:
        MAX_SAMPLES_FOR_EXACT_KPCA = 5000

        try:
            if X.shape[0] > MAX_SAMPLES_FOR_EXACT_KPCA:
                # Use Nystroem approximation for large datasets to avoid O(n^2) memory
                from sklearn.kernel_approximation import Nystroem
                from sklearn.decomposition import PCA

                n_nystroem = min(500, X.shape[0] // 10)
                nystroem = Nystroem(kernel="rbf", gamma=0.01, n_components=n_nystroem, random_state=42)
                X_approx = nystroem.fit_transform(X)

                pca = PCA(n_components=kpca_components, random_state=42)
                X_kpca = pca.fit_transform(X_approx)

                ensemble_parts.append(X_kpca)
                ensemble_names.extend([f"kpca_{i}" for i in range(X_kpca.shape[1])])
                state["ensemble_components"]["kpca_nystroem"] = nystroem
                state["ensemble_components"]["kpca_pca"] = pca
                state["ensemble_components"]["kpca_use_nystroem"] = True
                print(f"      Part 2: Nystroem+PCA -> {X_kpca.shape[1]} components (approx)")
            else:
                # Standard Kernel PCA for smaller datasets
                from sklearn.decomposition import KernelPCA
                kpca = KernelPCA(n_components=kpca_components, kernel="rbf", gamma=0.01, random_state=42)
                X_kpca = kpca.fit_transform(X)
                ensemble_parts.append(X_kpca)
                ensemble_names.extend([f"kpca_{i}" for i in range(X_kpca.shape[1])])
                state["ensemble_components"]["kpca"] = kpca
                state["ensemble_components"]["kpca_use_nystroem"] = False
                print(f"      Part 2: Kernel PCA -> {X_kpca.shape[1]} components")
        except Exception as e:
            print(f"      Part 2: Kernel PCA failed ({e})")

    # Part 3: ICA components
    if X.shape[1] >= ica_components:
        from sklearn.decomposition import FastICA
        ica = FastICA(n_components=ica_components, max_iter=500, random_state=42, whiten="unit-variance")
        try:
            X_ica = ica.fit_transform(X)
            ensemble_parts.append(X_ica)
            ensemble_names.extend([f"ica_{i}" for i in range(X_ica.shape[1])])
            state["ensemble_components"]["ica"] = ica
            print(f"      Part 3: ICA -> {X_ica.shape[1]} components")
        except Exception as e:
            print(f"      Part 3: ICA failed ({e})")

    # Part 4: K-Medoids distances (NEW)
    if X.shape[0] >= kmedoids_clusters * 5:
        try:
            from sklearn_extra.cluster import KMedoids
            kmedoids = KMedoids(n_clusters=kmedoids_clusters, metric="euclidean",
                               max_iter=300, random_state=42)
            kmedoids.fit(X)
            medoids = X[kmedoids.medoid_indices_]
            X_kmed = np.array([[np.linalg.norm(x - m) for m in medoids] for x in X])
            ensemble_parts.append(X_kmed)
            ensemble_names.extend([f"medoid_{i}" for i in range(kmedoids_clusters)])
            state["ensemble_components"]["kmedoids_medoids"] = medoids
            state["ensemble_components"]["kmedoids_labels"] = kmedoids.labels_
            print(f"      Part 4: K-Medoids -> {kmedoids_clusters} medoid distances")
        except Exception as e:
            print(f"      Part 4: K-Medoids failed ({e})")

    if len(ensemble_parts) == 0:
        print("      [WARN] No ensemble components created, using original")
        X_transformed = X
        ensemble_names = feature_names
    else:
        X_transformed = np.hstack(ensemble_parts)
        print(f"    [ENSEMBLE+] Combined: {X_transformed.shape[1]} total dimensions")

    state["transform_method"] = "ensemble_plus"
    return X_transformed, ensemble_names, state


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 5: CREATE SOFT TARGETS (EDGE 4)
# ═══════════════════════════════════════════════════════════════════════════════
def create_soft_targets(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """
    Create soft targets following EDGE 4.

    Methods:
      1. Sigmoid transform
      2. Label smoothing
      3. Confidence weighting
    """
    print("\n" + "=" * 70)
    print("STEP 5: CREATE SOFT TARGETS (EDGE 4)")
    print("=" * 70)

    df = df.copy()

    # Hard targets
    df["target_up"] = df["day_return"] > threshold
    df["target_timing"] = df["low_before_high"]

    # Soft targets via sigmoid
    k = CONFIG["soft_target_k"]
    df["soft_target_up"] = 1 / (1 + np.exp(-k * (df["day_return"] - threshold)))

    # Label smoothing
    epsilon = CONFIG["label_smoothing_epsilon"]
    df["smoothed_target_up"] = (1 - epsilon) * df["target_up"].astype(float) + epsilon / 2
    df["smoothed_target_timing"] = (1 - epsilon) * df["target_timing"].astype(float) + epsilon / 2

    # Confidence weights (higher for samples far from threshold)
    df["sample_weight"] = np.clip(np.abs(df["day_return"] - threshold) * 100, 0.3, 1.0)

    # Confidence for timing (based on time difference between high and low)
    time_diff = np.abs(df["high_minutes"] - df["low_minutes"])
    df["timing_weight"] = np.clip(time_diff / 200, 0.3, 1.0)  # More confident when high/low far apart

    print(f"[PASS] Created soft targets")
    print(f"       Soft target range: [{df['soft_target_up'].min():.3f}, {df['soft_target_up'].max():.3f}]")
    print(f"       Avg sample weight: {df['sample_weight'].mean():.3f}")

    return df


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 6: PURGED K-FOLD CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
def purged_kfold_cv(X, y, weights, n_folds=5, purge_days=5, embargo_days=2):
    """
    Time-series cross-validation with purging and embargo.

    Purge: Remove N days before test set to prevent leakage
    Embargo: Remove N days after test set
    """
    n_samples = len(X)
    fold_size = n_samples // n_folds
    folds = []

    for i in range(n_folds):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size if i < n_folds - 1 else n_samples

        # Purge and embargo
        train_end = test_start - purge_days
        train_start_after_embargo = test_end + embargo_days

        if train_end > 0:
            train_idx = list(range(0, train_end))
        else:
            train_idx = []

        if train_start_after_embargo < n_samples and i < n_folds - 1:
            train_idx.extend(list(range(train_start_after_embargo, n_samples)))

        test_idx = list(range(test_start, test_end))

        if len(train_idx) > 0 and len(test_idx) > 0:
            folds.append((train_idx, test_idx))

    return folds


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 7: FEATURE ROBUSTNESS SCORING (EDGE 1)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_feature_robustness(X, y, feature_names):
    """
    Score features by their robustness across regularization strengths.

    A robust feature:
      - Has non-zero coefficient at strong regularization
      - Has consistent sign across different C values
      - Has stable magnitude
    """
    print("\n" + "=" * 70)
    print("STEP 7: FEATURE ROBUSTNESS SCORING (EDGE 1)")
    print("=" * 70)

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Test across multiple regularization strengths
    C_values = [0.001, 0.01, 0.1, 1.0, 10.0]
    coef_matrix = []

    for C in C_values:
        model = LogisticRegression(penalty="l1", C=C, solver="saga", max_iter=2000, random_state=42)
        model.fit(X_scaled, y)
        coef_matrix.append(model.coef_[0])

    coef_matrix = np.array(coef_matrix)

    # Compute robustness scores
    robustness_scores = []

    for j, fname in enumerate(feature_names):
        coefs = coef_matrix[:, j]

        # 1. Survival rate (non-zero at strong regularization)
        survival_rate = np.mean(coefs != 0)

        # 2. Sign consistency
        signs = np.sign(coefs[coefs != 0]) if np.any(coefs != 0) else [0]
        sign_consistency = np.abs(np.mean(signs)) if len(signs) > 0 else 0

        # 3. Magnitude stability (lower std = more stable)
        nonzero_coefs = coefs[coefs != 0]
        if len(nonzero_coefs) > 1:
            magnitude_stability = 1 - min(np.std(np.abs(nonzero_coefs)) / (np.mean(np.abs(nonzero_coefs)) + 1e-10), 1)
        else:
            magnitude_stability = 1 if len(nonzero_coefs) == 1 else 0

        # Combined robustness score
        robustness = 0.4 * survival_rate + 0.3 * sign_consistency + 0.3 * magnitude_stability
        robustness_scores.append({
            "feature": fname,
            "robustness": robustness,
            "survival_rate": survival_rate,
            "sign_consistency": sign_consistency,
            "magnitude_stability": magnitude_stability,
            "mean_coef": np.mean(np.abs(coefs))
        })

    robustness_df = pd.DataFrame(robustness_scores).sort_values("robustness", ascending=False)

    print(f"\n  Top 15 robust features:")
    for i, row in robustness_df.head(15).iterrows():
        print(f"    {row['feature']}: {row['robustness']:.3f} (surv={row['survival_rate']:.2f}, sign={row['sign_consistency']:.2f})")

    # Filter to robust features only
    robust_features = robustness_df[robustness_df["robustness"] >= 0.3]["feature"].tolist()
    print(f"\n  Robust features (score >= 0.3): {len(robust_features)}")

    return robustness_df, robust_features


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8: TRAIN MODELS WITH CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
def train_with_cv(X, y, weights, feature_names, model_name="swing"):
    """Train models using purged CV and multiple regularization methods."""
    print(f"\n" + "=" * 70)
    print(f"TRAINING {model_name.upper()} MODEL WITH PURGED CV")
    print("=" * 70)

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression, ElasticNet
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    # Get CV folds
    folds = purged_kfold_cv(X, y, weights,
                            n_folds=CONFIG["n_cv_folds"],
                            purge_days=CONFIG["purge_days"],
                            embargo_days=CONFIG["embargo_days"])

    print(f"[INFO] {len(folds)} CV folds with purge={CONFIG['purge_days']}, embargo={CONFIG['embargo_days']}")

    results = {"l1": [], "l2": [], "elastic": [], "gb": []}

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        w_train = weights[train_idx]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # L1 (Lasso)
        model_l1 = LogisticRegression(penalty="l1", C=0.1, solver="saga", max_iter=2000, random_state=42)
        model_l1.fit(X_train_scaled, y_train, sample_weight=w_train)
        auc_l1 = roc_auc_score(y_test, model_l1.predict_proba(X_test_scaled)[:, 1]) if len(np.unique(y_test)) > 1 else 0.5
        results["l1"].append(auc_l1)

        # L2 (Ridge)
        model_l2 = LogisticRegression(penalty="l2", C=1.0, max_iter=2000, random_state=42)
        model_l2.fit(X_train_scaled, y_train, sample_weight=w_train)
        auc_l2 = roc_auc_score(y_test, model_l2.predict_proba(X_test_scaled)[:, 1]) if len(np.unique(y_test)) > 1 else 0.5
        results["l2"].append(auc_l2)

        # Elastic Net (via L1 ratio)
        model_en = LogisticRegression(penalty="elasticnet", C=0.5, l1_ratio=0.5, solver="saga", max_iter=2000, random_state=42)
        model_en.fit(X_train_scaled, y_train, sample_weight=w_train)
        auc_en = roc_auc_score(y_test, model_en.predict_proba(X_test_scaled)[:, 1]) if len(np.unique(y_test)) > 1 else 0.5
        results["elastic"].append(auc_en)

        # Gradient Boosting (shallow)
        model_gb = GradientBoostingClassifier(
            n_estimators=50,
            max_depth=CONFIG["max_tree_depth"],
            min_samples_leaf=CONFIG["min_samples_leaf"],
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        model_gb.fit(X_train_scaled, y_train, sample_weight=w_train)
        auc_gb = roc_auc_score(y_test, model_gb.predict_proba(X_test_scaled)[:, 1]) if len(np.unique(y_test)) > 1 else 0.5
        results["gb"].append(auc_gb)

    # Print CV results
    print(f"\n  Cross-validation AUC (mean +/- std):")
    for method, aucs in results.items():
        print(f"    {method.upper():10s}: {np.mean(aucs):.3f} +/- {np.std(aucs):.3f}")

    # Select best method
    if not results:
        print("  [WARN] No CV results available, defaulting to L2")
        return {"l2": [0.5], "gb": [0.5]}, "l2"

    best_method = max(results.keys(), key=lambda k: np.mean(results[k]))
    print(f"\n  Best method: {best_method.upper()} (AUC = {np.mean(results[best_method]):.3f})")

    return results, best_method


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 8B: OPTUNA HYPERPARAMETER OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════
def optimize_hyperparameters_optuna(X, y, weights, feature_names):
    """
    Use Optuna for Bayesian hyperparameter optimization.

    Advantages over grid search:
      - Intelligent exploration (Tree-Parzen Estimator)
      - Early stopping of bad trials
      - Much more efficient for large search spaces
      - Handles continuous and categorical params
    """
    if not CONFIG["use_optuna"]:
        print("\n[INFO] Optuna disabled in config, using default hyperparameters")
        return get_default_hyperparameters()

    print("\n" + "=" * 70)
    print("STEP 8B: INTELLIGENT HYPERPARAMETER OPTIMIZATION (OPTUNA)")
    print("=" * 70)

    try:
        import optuna
        from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
    except ImportError:
        print("[WARN] Optuna not installed, using default hyperparameters")
        return get_default_hyperparameters()

    # Suppress Optuna logging
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Get CV folds for evaluation
    folds = purged_kfold_cv(X, y, weights,
                            n_folds=CONFIG["n_cv_folds"],
                            purge_days=CONFIG["purge_days"],
                            embargo_days=CONFIG["embargo_days"])

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    def objective(trial):
        """Optuna objective function - maximize AUC."""
        # Sample hyperparameters
        hp = CONFIG["hp_search_space"]

        # Logistic Regression params
        l2_C = trial.suggest_float("l2_C", hp["l2_C"][0], hp["l2_C"][1], log=True)

        # Gradient Boosting params
        gb_n_estimators = trial.suggest_int("gb_n_estimators", hp["gb_n_estimators"][0], hp["gb_n_estimators"][1])
        gb_max_depth = trial.suggest_int("gb_max_depth", hp["gb_max_depth"][0], hp["gb_max_depth"][1])
        gb_learning_rate = trial.suggest_float("gb_learning_rate", hp["gb_learning_rate"][0], hp["gb_learning_rate"][1], log=True)
        gb_min_samples_leaf = trial.suggest_int("gb_min_samples_leaf", hp["gb_min_samples_leaf"][0], hp["gb_min_samples_leaf"][1])
        gb_subsample = trial.suggest_float("gb_subsample", hp["gb_subsample"][0], hp["gb_subsample"][1])

        # Evaluate with cross-validation
        aucs = []

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            w_train = weights[train_idx]

            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Train L2 Logistic Regression
            model_l2 = LogisticRegression(penalty="l2", C=l2_C, max_iter=2000, random_state=42)
            model_l2.fit(X_train_scaled, y_train, sample_weight=w_train)

            # Train Gradient Boosting
            model_gb = GradientBoostingClassifier(
                n_estimators=gb_n_estimators,
                max_depth=gb_max_depth,
                learning_rate=gb_learning_rate,
                min_samples_leaf=gb_min_samples_leaf,
                subsample=gb_subsample,
                random_state=42
            )
            model_gb.fit(X_train_scaled, y_train, sample_weight=w_train)

            # Ensemble prediction
            proba_l2 = model_l2.predict_proba(X_test_scaled)[:, 1]
            proba_gb = model_gb.predict_proba(X_test_scaled)[:, 1]
            proba_ensemble = (proba_l2 + proba_gb) / 2

            if len(np.unique(y_test)) > 1:
                auc = roc_auc_score(y_test, proba_ensemble)
            else:
                auc = 0.5

            aucs.append(auc)

            # Pruning: stop early if this trial is clearly worse
            trial.report(np.mean(aucs), fold_idx)
            if trial.should_prune():
                raise optuna.TrialPruned()

        return np.mean(aucs)

    # Select sampler
    sampler_name = CONFIG.get("optuna_sampler", "tpe")
    if sampler_name == "tpe":
        sampler = TPESampler(seed=42)
    elif sampler_name == "cmaes":
        sampler = CmaEsSampler(seed=42)
    else:
        sampler = RandomSampler(seed=42)

    # Create study
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5)
    )

    # Optimize
    n_trials = CONFIG["optuna_n_trials"]
    timeout = CONFIG.get("optuna_timeout", 300)

    print(f"[INFO] Running {n_trials} Optuna trials (timeout: {timeout}s)...")
    print(f"[INFO] Sampler: {sampler_name.upper()}")

    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
        n_jobs=1  # Sequential for reproducibility
    )

    # Results
    best_params = study.best_params
    best_value = study.best_value

    print(f"\n[OPTUNA RESULTS]")
    print(f"  Best AUC: {best_value:.4f}")
    print(f"  Best parameters:")
    for param, value in best_params.items():
        print(f"    {param}: {value:.4f}" if isinstance(value, float) else f"    {param}: {value}")

    # Show optimization history
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    if len(completed_trials) > 0:
        trial_aucs = [t.value for t in completed_trials]
        print(f"\n  Trial statistics:")
        print(f"    Completed: {len(completed_trials)}/{n_trials}")
        print(f"    AUC range: [{min(trial_aucs):.4f}, {max(trial_aucs):.4f}]")
        print(f"    Improvement: {max(trial_aucs) - min(trial_aucs):.4f}")

    return best_params


def get_default_hyperparameters():
    """Return default hyperparameters when Optuna is disabled."""
    return {
        "l2_C": 0.5,
        "gb_n_estimators": 50,
        "gb_max_depth": 3,
        "gb_learning_rate": 0.1,
        "gb_min_samples_leaf": 50,
        "gb_subsample": 0.8,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 9: TRAIN FINAL MODELS (FROM TRANSFORMED ARRAYS)
# ═══════════════════════════════════════════════════════════════════════════════
def train_final_models_from_arrays(
    X: np.ndarray, y_swing: np.ndarray, y_timing: np.ndarray,
    swing_weights: np.ndarray, timing_weights: np.ndarray,
    feature_names: list, df_clean: pd.DataFrame, threshold: float,
    best_params: dict = None
):
    """
    Train final production models on pre-transformed feature arrays.

    This function works with features that have already been through
    dimensionality reduction (UMAP, Kernel PCA, ICA, MI selection).

    Uses Optuna-optimized hyperparameters if provided.
    """
    print("\n" + "=" * 70)
    print("STEP 9: TRAIN FINAL PRODUCTION MODELS")
    print("=" * 70)

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

    # Use optimized or default hyperparameters
    if best_params is None:
        best_params = get_default_hyperparameters()

    print(f"[INFO] Using hyperparameters:")
    print(f"       L2 C: {best_params.get('l2_C', 0.5):.4f}")
    print(f"       GB n_estimators: {best_params.get('gb_n_estimators', 50)}")
    print(f"       GB max_depth: {best_params.get('gb_max_depth', 3)}")
    print(f"       GB learning_rate: {best_params.get('gb_learning_rate', 0.1):.4f}")

    n_samples = len(X)
    print(f"[INFO] Total samples: {n_samples}")
    print(f"[INFO] Features: {X.shape[1]} (transformed)")

    # Time-series split with purge
    split_idx = int(n_samples * 0.8)
    purge = CONFIG["purge_days"]

    train_idx = list(range(0, split_idx - purge))
    test_idx = list(range(split_idx, n_samples))

    X_train, X_test = X[train_idx], X[test_idx]
    y_swing_train, y_swing_test = y_swing[train_idx], y_swing[test_idx]
    y_timing_train, y_timing_test = y_timing[train_idx], y_timing[test_idx]
    w_swing_train = swing_weights[train_idx]
    w_timing_train = timing_weights[train_idx]

    print(f"[INFO] Train: {len(train_idx)}, Test: {len(test_idx)}")

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {}
    results = {}

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL 1: SWING DIRECTION (using optimized hyperparameters)
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[MODEL 1: SWING DIRECTION]")

    # Ensemble of L2 + GB with optimized params
    model_l2 = LogisticRegression(
        penalty="l2",
        C=best_params.get("l2_C", 0.5),
        max_iter=2000,
        random_state=42
    )
    model_l2.fit(X_train_scaled, y_swing_train, sample_weight=w_swing_train)

    model_gb = GradientBoostingClassifier(
        n_estimators=best_params.get("gb_n_estimators", 50),
        max_depth=best_params.get("gb_max_depth", 3),
        min_samples_leaf=best_params.get("gb_min_samples_leaf", 50),
        learning_rate=best_params.get("gb_learning_rate", 0.1),
        subsample=best_params.get("gb_subsample", 0.8),
        random_state=42
    )
    model_gb.fit(X_train_scaled, y_swing_train, sample_weight=w_swing_train)

    # Ensemble prediction
    proba_l2 = model_l2.predict_proba(X_test_scaled)[:, 1]
    proba_gb = model_gb.predict_proba(X_test_scaled)[:, 1]
    proba_swing = (proba_l2 + proba_gb) / 2
    pred_swing = (proba_swing > 0.5).astype(int)

    auc_swing = roc_auc_score(y_swing_test, proba_swing) if len(np.unique(y_swing_test)) > 1 else 0.5
    acc_swing = accuracy_score(y_swing_test, pred_swing)

    print(f"  AUC: {auc_swing:.3f}, Accuracy: {acc_swing:.3f}")
    print(classification_report(y_swing_test, pred_swing, target_names=["Down/Flat", "Up"]))

    models["swing"] = {"l2": model_l2, "gb": model_gb}
    results["swing"] = {"auc": auc_swing, "accuracy": acc_swing}

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL 2: TIMING
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[MODEL 2: ENTRY/EXIT TIMING]")

    model_timing_l2 = LogisticRegression(penalty="l2", C=0.5, max_iter=2000, random_state=42)
    model_timing_l2.fit(X_train_scaled, y_timing_train, sample_weight=w_timing_train)

    model_timing_gb = GradientBoostingClassifier(
        n_estimators=50, max_depth=3, min_samples_leaf=50,
        learning_rate=0.1, subsample=0.8, random_state=42
    )
    model_timing_gb.fit(X_train_scaled, y_timing_train, sample_weight=w_timing_train)

    proba_timing_l2 = model_timing_l2.predict_proba(X_test_scaled)[:, 1]
    proba_timing_gb = model_timing_gb.predict_proba(X_test_scaled)[:, 1]
    proba_timing = (proba_timing_l2 + proba_timing_gb) / 2
    pred_timing = (proba_timing > 0.5).astype(int)

    auc_timing = roc_auc_score(y_timing_test, proba_timing) if len(np.unique(y_timing_test)) > 1 else 0.5
    acc_timing = accuracy_score(y_timing_test, pred_timing)

    print(f"  AUC: {auc_timing:.3f}, Accuracy: {acc_timing:.3f}")
    print(classification_report(y_timing_test, pred_timing, target_names=["High First", "Low First"]))

    models["timing"] = {"l2": model_timing_l2, "gb": model_timing_gb}
    results["timing"] = {"auc": auc_timing, "accuracy": acc_timing}

    # ─────────────────────────────────────────────────────────────────────────
    # COMBINED SIGNALS
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[COMBINED SIGNAL ANALYSIS]")

    buy_signals = (pred_swing == 1) & (pred_timing == 1)
    sell_signals = (pred_swing == 0) & (pred_timing == 0)

    # Get actual returns from df_clean for test period
    test_df = df_clean.iloc[test_idx]
    test_returns = test_df["day_return"].values

    print(f"  Test days: {len(test_idx)}")
    print(f"  Buy signals:  {buy_signals.sum()} ({100*buy_signals.mean():.1f}%)")
    print(f"  Sell signals: {sell_signals.sum()} ({100*sell_signals.mean():.1f}%)")

    if buy_signals.sum() > 0:
        buy_returns = test_returns[buy_signals]
        print(f"\n  BUY PERFORMANCE:")
        print(f"    Win rate: {100*(buy_returns > 0).mean():.1f}%")
        print(f"    Avg return: {100*buy_returns.mean():.3f}%")
        print(f"    Total return: {100*buy_returns.sum():.2f}%")
        results["buy_win_rate"] = (buy_returns > 0).mean()
        results["buy_avg_return"] = buy_returns.mean()

    if sell_signals.sum() > 0:
        sell_returns = test_returns[sell_signals]
        print(f"\n  SELL PERFORMANCE:")
        print(f"    Win rate (for shorts): {100*(sell_returns < 0).mean():.1f}%")
        print(f"    Avg return: {100*sell_returns.mean():.3f}%")
        results["sell_win_rate"] = (sell_returns < 0).mean()

    return models, results, scaler, test_idx, proba_swing, proba_timing


# Legacy function for backward compatibility
def train_final_models(df: pd.DataFrame, feature_cols: list, threshold: float):
    """Train final production models."""
    print("\n" + "=" * 70)
    print("STEP 9: TRAIN FINAL PRODUCTION MODELS")
    print("=" * 70)

    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score, accuracy_score, classification_report

    df_clean = df.dropna(subset=feature_cols + ["target_up", "target_timing"]).copy()
    print(f"[INFO] Clean samples: {len(df_clean)}")

    # Time-series split
    split_idx = int(len(df_clean) * 0.8)
    train_df = df_clean.iloc[:split_idx - CONFIG["purge_days"]]
    test_df = df_clean.iloc[split_idx:]

    print(f"[INFO] Train: {len(train_df)}, Test: {len(test_df)}")

    X_train = train_df[feature_cols].values
    X_test = test_df[feature_cols].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    models = {}
    results = {}

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL 1: SWING DIRECTION
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[MODEL 1: SWING DIRECTION]")

    y_train = train_df["target_up"].astype(int).values
    y_test = test_df["target_up"].astype(int).values
    w_train = train_df["sample_weight"].values

    # Ensemble of L2 + GB
    model_l2 = LogisticRegression(penalty="l2", C=0.5, max_iter=2000, random_state=42)
    model_l2.fit(X_train_scaled, y_train, sample_weight=w_train)

    model_gb = GradientBoostingClassifier(
        n_estimators=50, max_depth=3, min_samples_leaf=50,
        learning_rate=0.1, subsample=0.8, random_state=42
    )
    model_gb.fit(X_train_scaled, y_train, sample_weight=w_train)

    # Ensemble prediction
    proba_l2 = model_l2.predict_proba(X_test_scaled)[:, 1]
    proba_gb = model_gb.predict_proba(X_test_scaled)[:, 1]
    proba_swing = (proba_l2 + proba_gb) / 2
    pred_swing = (proba_swing > 0.5).astype(int)

    auc_swing = roc_auc_score(y_test, proba_swing) if len(np.unique(y_test)) > 1 else 0.5
    acc_swing = accuracy_score(y_test, pred_swing)

    print(f"  AUC: {auc_swing:.3f}, Accuracy: {acc_swing:.3f}")
    print(classification_report(y_test, pred_swing, target_names=["Down/Flat", "Up"]))

    models["swing"] = {"l2": model_l2, "gb": model_gb}
    results["swing"] = {"auc": auc_swing, "accuracy": acc_swing}

    # ─────────────────────────────────────────────────────────────────────────
    # MODEL 2: TIMING
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[MODEL 2: ENTRY/EXIT TIMING]")

    y_train = train_df["target_timing"].astype(int).values
    y_test = test_df["target_timing"].astype(int).values
    w_train = train_df["timing_weight"].values

    model_timing_l2 = LogisticRegression(penalty="l2", C=0.5, max_iter=2000, random_state=42)
    model_timing_l2.fit(X_train_scaled, y_train, sample_weight=w_train)

    model_timing_gb = GradientBoostingClassifier(
        n_estimators=50, max_depth=3, min_samples_leaf=50,
        learning_rate=0.1, subsample=0.8, random_state=42
    )
    model_timing_gb.fit(X_train_scaled, y_train, sample_weight=w_train)

    proba_timing_l2 = model_timing_l2.predict_proba(X_test_scaled)[:, 1]
    proba_timing_gb = model_timing_gb.predict_proba(X_test_scaled)[:, 1]
    proba_timing = (proba_timing_l2 + proba_timing_gb) / 2
    pred_timing = (proba_timing > 0.5).astype(int)

    auc_timing = roc_auc_score(y_test, proba_timing) if len(np.unique(y_test)) > 1 else 0.5
    acc_timing = accuracy_score(y_test, pred_timing)

    print(f"  AUC: {auc_timing:.3f}, Accuracy: {acc_timing:.3f}")
    print(classification_report(y_test, pred_timing, target_names=["High First", "Low First"]))

    models["timing"] = {"l2": model_timing_l2, "gb": model_timing_gb}
    results["timing"] = {"auc": auc_timing, "accuracy": acc_timing}

    # ─────────────────────────────────────────────────────────────────────────
    # COMBINED SIGNALS
    # ─────────────────────────────────────────────────────────────────────────
    print("\n[COMBINED SIGNAL ANALYSIS]")

    buy_signals = (pred_swing == 1) & (pred_timing == 1)
    sell_signals = (pred_swing == 0) & (pred_timing == 0)

    test_returns = test_df["day_return"].values

    print(f"  Test days: {len(test_df)}")
    print(f"  Buy signals:  {buy_signals.sum()} ({100*buy_signals.mean():.1f}%)")
    print(f"  Sell signals: {sell_signals.sum()} ({100*sell_signals.mean():.1f}%)")

    if buy_signals.sum() > 0:
        buy_returns = test_returns[buy_signals]
        print(f"\n  BUY PERFORMANCE:")
        print(f"    Win rate: {100*(buy_returns > 0).mean():.1f}%")
        print(f"    Avg return: {100*buy_returns.mean():.3f}%")
        print(f"    Total return: {100*buy_returns.sum():.2f}%")
        results["buy_win_rate"] = (buy_returns > 0).mean()
        results["buy_avg_return"] = buy_returns.mean()

    if sell_signals.sum() > 0:
        sell_returns = test_returns[sell_signals]
        print(f"\n  SELL PERFORMANCE:")
        print(f"    Win rate (for shorts): {100*(sell_returns < 0).mean():.1f}%")
        print(f"    Avg return: {100*sell_returns.mean():.3f}%")
        results["sell_win_rate"] = (sell_returns < 0).mean()

    return models, results, scaler, test_df, proba_swing, proba_timing


# ═══════════════════════════════════════════════════════════════════════════════
# STEP 10: SAVE MODELS
# ═══════════════════════════════════════════════════════════════════════════════
def save_models(models, results, scaler, feature_cols, best_threshold, robustness_df, dim_state, entry_exit_model=None):
    """Save all models and metadata."""
    print("\n" + "=" * 70)
    print("STEP 10: SAVE MODELS")
    print("=" * 70)

    import joblib

    models_dir = project_root / "models" / "production"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "spy_robust_models.joblib"

    # Build save dict
    save_dict = {
        "models": models,
        "results": results,
        "scaler": scaler,
        "feature_cols": feature_cols,
        "best_threshold": best_threshold,
        "robustness_scores": robustness_df.to_dict() if robustness_df is not None else None,
        "dim_reduction_state": dim_state,
        "config": CONFIG,
        "trained_at": datetime.now().isoformat(),
        "description": "Robust dual model with all edge strategies + entry/exit timing"
    }

    # Save main model bundle
    joblib.dump(save_dict, model_path)
    print(f"[PASS] Main models saved to {model_path}")

    # Save entry/exit timing model separately (it's more complex)
    if entry_exit_model is not None:
        entry_exit_path = models_dir / "entry_exit_timing_model.joblib"
        entry_exit_model.save(str(entry_exit_path))
        print(f"[PASS] Entry/Exit Timing Model saved to {entry_exit_path}")

    return model_path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    print("\n" + "=" * 70)
    print("GIGA TRADER - ROBUST MODEL TRAINING v2.0")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Configuration:")
    print(f"  Data: {CONFIG['years_to_download']} years of 1-minute data")
    print(f"  Dim Reduction: {CONFIG['dim_reduction_method'].upper()}")
    print(f"  Optuna HP Optimization: {'ENABLED' if CONFIG['use_optuna'] else 'DISABLED'}")
    print()
    print("Edge Strategies Implemented:")
    print("  EDGE 1: Regularization-first + feature robustness scoring")
    print("  EDGE 2: Comprehensive extended hours features")
    print("  EDGE 3: Grid-searched intraday opportunities")
    print("  EDGE 4: Soft targets + confidence weighting")
    print("  EDGE 5: Combined signals for batch scaling")
    print()
    print("Advanced Methods:")
    print("  - K-Medoids clustering (outlier-robust)")
    print("  - Optuna Bayesian hyperparameter optimization")
    print("  - Missing bar detection and handling")
    print("  - Multi-year extended hours analysis")

    # Step 1: Download data (5-10 years)
    df_1min = download_data()

    # Step 2: Engineer features
    best_threshold = 0.0025  # Start with best from previous run
    df_daily = engineer_all_features(df_1min, swing_threshold=best_threshold)

    # Step 3: Add rolling features
    df_daily = add_rolling_features(df_daily)

    # Step 4: Create soft targets
    df_daily = create_soft_targets(df_daily, threshold=best_threshold)

    # Step 4B: Anti-Overfitting Feature Augmentation
    if CONFIG.get("use_anti_overfit", False):
        print("\n" + "=" * 70)
        print("ANTI-OVERFITTING FEATURE AUGMENTATION")
        print("=" * 70)
        df_daily, anti_overfit_metadata = integrate_anti_overfit(
            df_daily,
            spy_1min=df_1min,
            use_synthetic=CONFIG.get("use_synthetic_universes", True),
            use_cross_assets=CONFIG.get("use_cross_assets", True),
            use_breadth_streaks=CONFIG.get("use_breadth_streaks", True),
            use_mag_breadth=CONFIG.get("use_mag_breadth", True),
            use_sector_breadth=CONFIG.get("use_sector_breadth", True),
            use_vol_regime=CONFIG.get("use_vol_regime", True),
            synthetic_weight=CONFIG.get("synthetic_weight", 0.3),
        )
        print(f"[INFO] Anti-overfit features added: {anti_overfit_metadata}")
    else:
        anti_overfit_metadata = {}

    # Get feature columns (exclude quality flags and anti-overfit metadata)
    exclude_cols = ["date", "day_return", "day_volume", "is_up_day", "is_down_day",
                    "low_before_high", "high_minutes", "low_minutes",
                    "target_up", "target_timing", "soft_target_up",
                    "smoothed_target_up", "smoothed_target_timing",
                    "sample_weight", "timing_weight",
                    "max_gain_from_1015", "max_gain_from_1230",
                    "has_premarket", "has_afterhours", "quality_score", "year",
                    # Anti-overfit metadata columns (not features)
                    "sample_weight_augment", "universe_id", "universe_type",
                    "synthetic_return", "real_return"]
    feature_cols = [c for c in df_daily.columns if c not in exclude_cols]

    print(f"\n[INFO] Initial features: {len(feature_cols)}")

    # Clean data
    df_clean = df_daily.dropna(subset=feature_cols + ["target_up"]).copy()
    print(f"[INFO] Clean samples: {len(df_clean)}")

    X = df_clean[feature_cols].values
    y = df_clean["target_up"].astype(int).values
    weights = df_clean["sample_weight"].values

    # ─────────────────────────────────────────────────────────────────────────
    # LEAK-PROOF TRAINING PATH (recommended - fixes data leakage issues)
    # ─────────────────────────────────────────────────────────────────────────
    if CONFIG.get("use_leak_proof_cv", True):
        print("\n" + "=" * 70)
        print("LEAK-PROOF CROSS-VALIDATION (prevents data leakage)")
        print("=" * 70)
        print("[INFO] This approach fits all transformations INSIDE each CV fold")
        print("[INFO] Expected: More realistic (lower) AUC, but more reliable for production")

        # Configure leak-proof pipeline
        leak_proof_config = {
            "n_cv_folds": CONFIG["n_cv_folds"],
            "purge_days": CONFIG["purge_days"],
            "embargo_days": CONFIG["embargo_days"],
            "feature_selection_method": "mutual_info",
            "n_features": CONFIG.get("mi_n_features", 30),
            "dim_reduction_method": "kernel_pca",
            "n_components": CONFIG.get("kpca_n_components", 20),
            "use_ensemble": CONFIG.get("use_model_ensemble", True),
            "random_state": 42,
        }

        # Train swing model with leak-proof CV
        print("\n[SWING MODEL - Leak-Proof Training]")
        swing_pipeline, swing_cv_results = train_with_leak_proof_cv(
            X, y,
            sample_weights=weights,
            config=leak_proof_config,
            verbose=True,
        )

        # Store results
        leak_proof_results = {
            "swing": {
                "mean_train_auc": swing_cv_results["mean_train_auc"],
                "mean_test_auc": swing_cv_results["mean_test_auc"],
                "train_test_gap": swing_cv_results["train_test_gap"],
            }
        }

        # Train timing model with leak-proof CV
        y_timing = df_clean["target_timing"].astype(int).values
        timing_weights = df_clean["timing_weight"].values

        print("\n[TIMING MODEL - Leak-Proof Training]")
        timing_pipeline, timing_cv_results = train_with_leak_proof_cv(
            X, y_timing,
            sample_weights=timing_weights,
            config=leak_proof_config,
            verbose=True,
        )

        leak_proof_results["timing"] = {
            "mean_train_auc": timing_cv_results["mean_train_auc"],
            "mean_test_auc": timing_cv_results["mean_test_auc"],
            "train_test_gap": timing_cv_results["train_test_gap"],
        }

        # Summary
        print("\n" + "=" * 70)
        print("LEAK-PROOF TRAINING SUMMARY")
        print("=" * 70)
        print(f"  Swing Model:")
        print(f"    CV Test AUC: {swing_cv_results['mean_test_auc']:.3f} +/- {swing_cv_results['std_test_auc']:.3f}")
        print(f"    Train-Test Gap: {swing_cv_results['train_test_gap']:.3f}")

        print(f"\n  Timing Model:")
        print(f"    CV Test AUC: {timing_cv_results['mean_test_auc']:.3f} +/- {timing_cv_results['std_test_auc']:.3f}")
        print(f"    Train-Test Gap: {timing_cv_results['train_test_gap']:.3f}")

        # Check for overfitting
        if swing_cv_results["train_test_gap"] > 0.10 or timing_cv_results["train_test_gap"] > 0.10:
            print("\n  [WARNING] High train-test gap detected - potential overfitting remains")
        else:
            print("\n  [GOOD] Low train-test gap - models appear robust")

        # Save models
        import joblib
        models_dir = Path("models/production")
        models_dir.mkdir(parents=True, exist_ok=True)

        model_path = models_dir / "spy_leak_proof_models.joblib"
        joblib.dump({
            "swing_pipeline": swing_pipeline,
            "timing_pipeline": timing_pipeline,
            "feature_columns": feature_cols,
            "config": CONFIG,
            "cv_results": leak_proof_results,
        }, model_path)

        print(f"\n[SAVED] Leak-proof models saved to: {model_path}")

        # Skip the legacy leaky training path
        print("\n[INFO] Skipping legacy training path (using leak-proof results)")
        print("\n" + "=" * 70)
        print(f"Training completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        return

    # ─────────────────────────────────────────────────────────────────────────
    # LEGACY TRAINING PATH - DISABLED (data leakage issues)
    # ─────────────────────────────────────────────────────────────────────────
    # The legacy path performed dimensionality reduction BEFORE cross-validation,
    # which causes data leakage because the test set information leaks into the
    # feature transformation. This produces unreliable metrics.
    #
    # To maintain pipeline integrity, the legacy path is now DISABLED.
    # All training MUST use leak-proof CV which fits transformations inside each fold.
    raise ValueError(
        "CRITICAL: Legacy training path is DISABLED due to data leakage issues.\n"
        "The legacy path performed dimensionality reduction BEFORE cross-validation,\n"
        "causing test set information to leak into feature transformations.\n"
        "\n"
        "Solution: Set CONFIG['use_leak_proof_cv'] = True (default)\n"
        "\n"
        "The leak-proof path fits all transformations INSIDE each CV fold,\n"
        "ensuring proper validation and reliable performance metrics."
    )

    # ─────────────────────────────────────────────────────────────────────────
    # LEGACY CODE BELOW (kept for reference but never executes)
    # ─────────────────────────────────────────────────────────────────────────
    # NOTE: The code below will NEVER execute due to the raise above.
    # It is kept temporarily for reference during the migration period.
    # TODO: Remove this dead code in a future cleanup.
    print("\n[LEGACY PATH] This code should never execute")

    # Step 5: Advanced dimensionality reduction (passes y for MI-based methods)
    X_reduced, reduced_features, dim_state = reduce_dimensions(X, feature_cols, y=y, fit=True)
    print(f"[INFO] Features after reduction: {len(reduced_features)}")

    # Step 6: Feature robustness scoring
    robustness_df, robust_features = compute_feature_robustness(X_reduced, y, reduced_features)

    # Use robust features only
    robust_idx = [i for i, f in enumerate(reduced_features) if f in robust_features]
    if len(robust_idx) >= 10:
        X_robust = X_reduced[:, robust_idx]
        robust_feature_names = [reduced_features[i] for i in robust_idx]
    else:
        print("[WARN] Not enough robust features, using all reduced features")
        X_robust = X_reduced
        robust_feature_names = reduced_features

    # Step 7: Cross-validation (quick check)
    cv_results, best_method = train_with_cv(X_robust, y, weights, robust_feature_names, "swing")

    # Step 8: Optuna hyperparameter optimization
    best_params = optimize_hyperparameters_optuna(X_robust, y, weights, robust_feature_names)

    # Step 8B: Hyperparameter Stability Analysis
    stability_score = 0.5  # Default if not analyzed
    if CONFIG.get("use_anti_overfit", False) and best_params:
        print("\n[STABILITY ANALYSIS]")
        try:
            stability_analyzer = StabilityAnalyzer(perturbation_pct=0.05)

            # Define a simple scoring function for stability test
            from sklearn.model_selection import cross_val_score
            from sklearn.linear_model import LogisticRegression

            def hp_score_fn(params):
                C = params.get("l2_C", 1.0)
                model = LogisticRegression(C=C, max_iter=500, random_state=42)
                scores = cross_val_score(model, X_robust, y, cv=3, scoring="roc_auc")
                return scores.mean()

            param_ranges = {
                "l2_C": CONFIG["hp_search_space"].get("l2_C", (0.01, 10.0)),
            }

            stability_results = stability_analyzer.compute_stability_score(
                base_params={"l2_C": best_params.get("l2_C", 1.0)},
                base_score=cv_results.get("mean_auc", 0.7) if isinstance(cv_results, dict) else 0.7,
                param_ranges=param_ranges,
                score_fn=hp_score_fn,
                n_samples=10,
            )
            stability_score = stability_results.get("stability_score", 0.5)

            if stability_score < CONFIG.get("stability_threshold", 0.5):
                print(f"  [WARN] Low stability score ({stability_score:.3f}). Solution may be fragile.")
            else:
                print(f"  [GOOD] Stability score: {stability_score:.3f}")
        except Exception as e:
            print(f"  [WARN] Stability analysis failed: {e}")

    # Step 9: Train final models using optimized hyperparameters
    print(f"\n[INFO] Final feature count: {len(robust_feature_names)}")

    # Get timing targets and weights from df_clean
    y_timing = df_clean["target_timing"].astype(int).values
    timing_weights = df_clean["timing_weight"].values

    models, results, scaler, test_indices, proba_swing, proba_timing = \
        train_final_models_from_arrays(
            X_robust, y, y_timing, weights, timing_weights,
            robust_feature_names, df_clean, best_threshold,
            best_params=best_params
        )

    # Step 9B: Weighted Model Evaluation Score (WMES)
    if CONFIG.get("use_anti_overfit", False):
        print("\n[WEIGHTED MODEL EVALUATION]")
        try:
            # Get test set data for evaluation
            # Note: proba_swing already contains only test set predictions
            # test_indices are the indices into the original arrays
            test_mask = np.zeros(len(y), dtype=bool)
            test_mask[test_indices] = True

            y_test = y[test_mask]
            # proba_swing is already the test set predictions (not full dataset)
            y_pred = (proba_swing > 0.5).astype(int)
            y_proba = proba_swing
            returns = df_clean["day_return"].values[test_mask]

            # Get CV scores from results
            cv_scores = results.get("cv_scores", [results["swing"]["auc"]])
            if not isinstance(cv_scores, list):
                cv_scores = [cv_scores]

            # Compute WMES
            wmes_results = compute_weighted_evaluation(
                y_true=y_test,
                y_pred=y_pred,
                y_proba=y_proba,
                returns=returns,
                cv_scores=cv_scores,
                n_features=len(robust_feature_names),
                hp_sensitivity=1 - stability_score,  # Convert to sensitivity
                noise_scores=None,  # Could add noise testing later
            )

            # Add to results
            results["wmes"] = wmes_results["wmes"]
            results["wmes_components"] = wmes_results

            print(f"  WMES Score: {wmes_results['wmes']:.3f}")
            print(f"    Win Rate Component: {wmes_results['win_rate']:.3f}")
            print(f"    Robustness Component: {wmes_results['robustness']:.3f}")
            print(f"    Profit Potential: {wmes_results['profit_potential']:.3f}")
            print(f"    Plateau Stability: {wmes_results['plateau_stability']:.3f}")
            print(f"    Complexity Penalty: {wmes_results['complexity_penalty']:.3f}")

            if wmes_results["wmes"] < CONFIG.get("wmes_threshold", 0.55):
                print(f"\n  [WARN] WMES below threshold ({CONFIG.get('wmes_threshold', 0.55):.2f})")
                print("         Model may be overfit or fragile. Consider:")
                print("         - Reducing model complexity")
                print("         - Adding more regularization")
                print("         - Using more synthetic data")
            else:
                print(f"\n  [GOOD] WMES above threshold - model appears robust")

        except Exception as e:
            print(f"  [WARN] WMES computation failed: {e}")
            results["wmes"] = None

    # Step 9C: Robustness Ensemble (train adjacent dimension/parameter models)
    ensemble_results = None
    if CONFIG.get("use_robustness_ensemble", False) and CONFIG.get("use_anti_overfit", False):
        print("\n[ROBUSTNESS ENSEMBLE]")
        try:
            # Create the ensemble with dimension and parameter perturbations
            robustness_ensemble = RobustnessEnsemble(
                n_dimension_variants=CONFIG.get("n_dimension_variants", 2),
                n_param_variants=CONFIG.get("n_param_variants", 2),
                param_noise_pct=CONFIG.get("param_noise_pct", 0.05),
                center_weight=CONFIG.get("ensemble_center_weight", 0.5),
            )

            # Get optimal dimensions from current feature count
            optimal_dims = len(robust_feature_names)

            # Train the ensemble
            ensemble_results = robustness_ensemble.train_ensemble(
                X=X_robust,
                y=y,
                sample_weights=weights,
                base_params=best_params if best_params else {"C": 1.0, "max_iter": 500},
                optimal_dims=optimal_dims,
                cv_folds=3,
            )

            # Store results
            results["robustness_ensemble"] = {
                "fragility_score": ensemble_results["fragility"]["fragility_score"],
                "interpretation": ensemble_results["fragility"]["interpretation"],
                "n_models": len(ensemble_results["models"]),
                "dim_variants": ensemble_results["dim_variants"],
            }

            # Check fragility threshold
            fragility = ensemble_results["fragility"]["fragility_score"]
            threshold = CONFIG.get("fragility_threshold", 0.35)

            if fragility > threshold:
                print(f"\n  [WARN] Fragility ({fragility:.3f}) exceeds threshold ({threshold:.2f})")
                print(f"         {ensemble_results['fragility']['interpretation']}")
                print("         Consider: more regularization, simpler model, more data")
            else:
                print(f"\n  [GOOD] Fragility ({fragility:.3f}) below threshold - solution is robust")

            # Optional: Evaluate ensemble on test set
            if len(test_indices) > 0:
                X_test = X_robust[test_indices]
                y_test = y[test_indices]
                ensemble_eval = robustness_ensemble.evaluate_ensemble(X_test, y_test)
                results["robustness_ensemble"]["test_auc"] = ensemble_eval["ensemble"]["auc"]

        except Exception as e:
            print(f"  [WARN] Robustness ensemble failed: {e}")
            import traceback
            traceback.print_exc()

    # Step 9D: Entry/Exit Timing Model (ML-based specific timing predictions)
    entry_exit_model = None
    if CONFIG.get("train_entry_exit_model", False):
        print("\n" + "=" * 70)
        print("STEP 9D: ENTRY/EXIT TIMING MODEL")
        print("=" * 70)
        try:
            # Need intraday data for this - check if df_1min is available
            # (it's passed from main() through the training flow)
            print("[INFO] Training ML model for entry/exit timing predictions")
            print("  This model predicts:")
            print("    - Optimal entry time (minutes from open)")
            print("    - Optimal exit time (minutes from open)")
            print("    - Position size based on conditions")
            print("    - Dynamic stop loss / take profit levels")
            print("    - Batch entry schedules")
            print("    - Guardrail triggers")

            # Entry/Exit timing model should use only REAL data (not synthetic)
            # Filter out synthetic augmented samples
            if "universe_id" in df_clean.columns:
                # Real data has universe_id == NaN or 0
                real_mask = df_clean["universe_id"].isna() | (df_clean["universe_id"] == 0)
                real_mask_array = real_mask.values
                df_real = df_clean[real_mask].copy()
            else:
                real_mask_array = np.ones(len(df_clean), dtype=bool)
                df_real = df_clean.copy()

            print(f"  Training on {len(df_real)} real samples (excluding synthetic)")

            # Use the already-transformed X_robust data (which matches scaler expectations)
            # X_robust has shape (n_samples, n_transformed_features) = (13630, 50)
            X_real_robust = X_robust[real_mask_array]

            # Apply scaler (expects 50 features - the transformed feature count)
            X_real_scaled = scaler.transform(X_real_robust)

            # Get swing predictions for real data
            proba_real = (models["swing"]["l2"].predict_proba(X_real_scaled)[:, 1] +
                         models["swing"]["gb"].predict_proba(X_real_scaled)[:, 1]) / 2

            # Create directions based on swing predictions
            directions = pd.Series(
                np.where(proba_real > 0.5, "LONG", "SHORT"),
                index=df_real.index,
            )

            # Create the timing model
            entry_exit_model = EntryExitTimingModel(
                model_type=CONFIG.get("entry_exit_model_type", "gradient_boosting"),
                entry_window=CONFIG.get("entry_window", (0, 120)),
                exit_window=CONFIG.get("exit_window", (180, 385)),
                min_position_pct=CONFIG.get("min_position_pct", 0.05),
                max_position_pct=CONFIG.get("max_position_pct", 0.25),
            )

            # Train the timing model using daily and intraday data
            # df_1min is available from Step 1 (download_data)
            timing_metrics = entry_exit_model.fit(
                daily_data=df_real,
                intraday_data=df_1min,
                directions=directions,
                cv_folds=3,
            )

            # Store results
            results["entry_exit_timing"] = {
                "metrics": timing_metrics,
                "entry_window": CONFIG.get("entry_window"),
                "exit_window": CONFIG.get("exit_window"),
            }

            # Store model for saving
            models["entry_exit_timing"] = entry_exit_model

            print("\n  [PASS] Entry/Exit Timing Model trained successfully")

        except Exception as e:
            print(f"  [WARN] Entry/Exit Timing Model training failed: {e}")
            import traceback
            traceback.print_exc()
            entry_exit_model = None

    # Step 10: Save (use robust_feature_names which are the transformed feature names)
    model_path = save_models(models, results, scaler, robust_feature_names,
                             best_threshold, robustness_df, dim_state, entry_exit_model)

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nSwing Model AUC: {results['swing']['auc']:.3f}")
    print(f"Timing Model AUC: {results['timing']['auc']:.3f}")
    if "buy_win_rate" in results:
        print(f"Buy Signal Win Rate: {results['buy_win_rate']*100:.1f}%")

    # Entry/Exit Timing Model summary
    if entry_exit_model is not None and "entry_exit_timing" in results:
        print("\nEntry/Exit Timing Model:")
        ee_metrics = results.get("entry_exit_timing", {}).get("metrics", {})
        if "entry" in ee_metrics:
            entry_mae = ee_metrics["entry"].get("mae_minutes", {}).get("mean", "N/A")
            print(f"  Entry Time MAE: {entry_mae:.1f} minutes" if isinstance(entry_mae, float) else f"  Entry Time MAE: {entry_mae}")
        if "exit" in ee_metrics:
            exit_mae = ee_metrics["exit"].get("mae_minutes", {}).get("mean", "N/A")
            print(f"  Exit Time MAE: {exit_mae:.1f} minutes" if isinstance(exit_mae, float) else f"  Exit Time MAE: {exit_mae}")
        if "position" in ee_metrics:
            pos_mae = ee_metrics["position"].get("mae_pct", {}).get("mean", "N/A")
            print(f"  Position Size MAE: {pos_mae:.2f}%" if isinstance(pos_mae, float) else f"  Position Size MAE: {pos_mae}")

    print(f"\nModels saved to: {model_path}")
    print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
