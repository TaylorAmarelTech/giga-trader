"""
GIGA TRADER - Alpaca Paper Trading: Signal Generator
=====================================================
ML-based trading signal generation using trained models.

Components:
  - SignalGenerator class (static and dynamic model selection)
"""

import sys
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import joblib

# Setup path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Local imports
from src.phase_19_paper_trading.alpaca_client import (
    TRADING_CONFIG,
    DynamicThresholds,
    dynamic_thresholds,
    SignalType,
    TradingSignal,
)

# Dynamic model selector
try:
    from src.dynamic_model_selector import DynamicModelSelector, EnsemblePrediction
    DYNAMIC_SELECTOR_AVAILABLE = True
except ImportError:
    DYNAMIC_SELECTOR_AVAILABLE = False
    print("[INFO] Dynamic model selector not available")

# Temporal cascade signal generator
try:
    from src.phase_12_model_training.temporal_cascade_trainer import TemporalCascadeSignalGenerator
    TEMPORAL_CASCADE_AVAILABLE = True
except ImportError:
    TEMPORAL_CASCADE_AVAILABLE = False

logger = logging.getLogger("GigaTrader")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SIGNAL GENERATOR
# ═══════════════════════════════════════════════════════════════════════════════
class SignalGenerator:
    """
    Generate trading signals from trained ML models.

    Supports two modes:
    1. Static mode: Load a single hardcoded model (legacy)
    2. Dynamic mode: Use DynamicModelSelector to select/ensemble from registry

    Dynamic mode provides:
    - Automatic model selection based on performance
    - Intelligent ensembling of multiple models
    - Adaptive entry/exit window matching
    """

    def __init__(self, model_dir: Path = None, use_dynamic_selector: bool = True,
                 cascade_blend_weight: float = 0.15):
        self.model_dir = model_dir or TRADING_CONFIG["model_dir"]
        self.models = {}
        self.scaler = None
        self.dim_state = None
        self.feature_cols = None
        self.use_leak_proof = False  # Initialize here - will be updated in _load_models() if leak-proof model loaded
        self.model_config = None

        # Temporal cascade blend weight (0.0 = ignore cascade, 1.0 = cascade only)
        self.cascade_blend_weight = max(0.0, min(1.0, cascade_blend_weight))

        # Temporal cascade integration
        self.temporal_cascade = None
        if TEMPORAL_CASCADE_AVAILABLE:
            self._init_temporal_cascade()

        # Confidence calibrator — learns from actual trade outcomes
        try:
            from src.phase_15_strategy.signal_detectors import ConfidenceCalibrator
            self.calibrator = ConfidenceCalibrator()
        except ImportError:
            self.calibrator = None

        # Dynamic model selector - MANDATORY for proper signal generation
        # No fallback to static models to ensure all signals use properly trained/validated models
        self.use_dynamic_selector = use_dynamic_selector and DYNAMIC_SELECTOR_AVAILABLE
        self.dynamic_selector = None

        # Always load models to get feature_cols and use_leak_proof metadata
        # (needed by prepare_features even when using dynamic selector)
        self._load_models()

        if self.use_dynamic_selector:
            self._init_dynamic_selector()

        # If dynamic selector is not available or empty, warn about degraded mode
        if not self.dynamic_selector or not self.dynamic_selector.candidates:
            logger.warning("=" * 60)
            logger.warning("DEGRADED MODE: Dynamic model selector not available")
            logger.warning("Signals may not use properly validated models")
            logger.warning("Run 'python scripts/run_grid_search.py' to populate registry")
            logger.warning("=" * 60)

            # Verify we have BOTH swing AND timing models
            has_swing = "swing_pipeline" in self.models or "swing_l2" in self.models or "swing" in self.models
            has_timing = "timing_pipeline" in self.models or "timing_l2" in self.models or "timing" in self.models

            if not has_swing or not has_timing:
                raise ValueError(
                    "CRITICAL: No valid models available. Both swing AND timing models are required. "
                    "Run grid search to train and validate models: python scripts/run_grid_search.py"
                )

    def _init_dynamic_selector(self):
        """Initialize the dynamic model selector from registry.

        REQUIREMENTS (per CLAUDE.md):
        - min_test_auc > 0.58 (target, allow 0.55 minimum)
        - min_wmes > 0.55 (target, allow 0.50 minimum)
        - Models must have both swing AND timing predictions
        """
        try:
            self.dynamic_selector = DynamicModelSelector(
                min_test_auc=0.55,  # Minimum required AUC (0.58 is target)
                min_wmes=0.50,      # Minimum WMES (0.55 is target)
                max_models_to_load=10,
                ensemble_method="weighted_average",
            )
            n_loaded = self.dynamic_selector.load_from_registry()

            if n_loaded > 0:
                logger.info(f"Dynamic selector loaded {n_loaded} qualified model candidates")

                # Verify models meet quality standards
                status = self.dynamic_selector.get_status()
                top_models = status.get("top_models", [])

                if top_models:
                    best_auc = max(m.get("test_auc", 0) for m in top_models)
                    best_wmes = max(m.get("wmes", 0) for m in top_models)

                    if best_auc < 0.58:
                        logger.warning(f"Best model AUC ({best_auc:.3f}) below target (0.58)")
                    if best_wmes < 0.55:
                        logger.warning(f"Best model WMES ({best_wmes:.3f}) below target (0.55)")

                # Get available entry/exit windows
                windows = self.dynamic_selector.get_available_windows()
                if windows:
                    logger.info(f"Available entry/exit windows: {len(windows)}")
                    for w in windows[:3]:
                        logger.info(f"  Entry: {w['entry_window']}, Exit: {w['exit_window']}, Score: {w['best_score']:.3f}")
            else:
                logger.warning("No models in registry meet minimum quality requirements")
                logger.warning("Run grid search to train validated models: python scripts/run_grid_search.py")
                self.dynamic_selector = None

        except Exception as e:
            logger.error(f"Failed to initialize dynamic selector: {e}")
            self.dynamic_selector = None

    def _init_temporal_cascade(self):
        """Initialize temporal cascade signal source if trained models exist."""
        try:
            cascade_dir = self.model_dir / "temporal_cascade"
            if cascade_dir.exists():
                self.temporal_cascade = TemporalCascadeSignalGenerator(model_path=cascade_dir)
                logger.info("Temporal cascade signal source initialized")
            else:
                logger.info("No temporal cascade models found (train with scripts/train_temporal_cascade.py)")
        except Exception as e:
            logger.warning(f"Temporal cascade initialization failed: {e}")
            self.temporal_cascade = None

    def _load_models(self):
        """Load trained models from disk."""
        self.use_leak_proof = False  # Track which model type is loaded

        try:
            # PREFER leak-proof model (better model with correct feature handling)
            leak_proof_path = self.model_dir / "spy_leak_proof_models.joblib"
            if leak_proof_path.exists():
                data = joblib.load(leak_proof_path)

                # Leak-proof model uses sklearn Pipelines that handle transformation internally
                if "swing_pipeline" in data:
                    self.models["swing_pipeline"] = data["swing_pipeline"]
                    logger.info("Loaded swing pipeline (leak-proof)")

                if "timing_pipeline" in data:
                    self.models["timing_pipeline"] = data["timing_pipeline"]
                    logger.info("Loaded timing pipeline (leak-proof)")

                # Feature columns are the RAW feature names (before transformation)
                if "feature_columns" in data:
                    self.feature_cols = data["feature_columns"]
                    logger.info(f"Loaded {len(self.feature_cols)} raw feature columns (leak-proof)")

                if "config" in data:
                    self.model_config = data["config"]

                self.use_leak_proof = True
                logger.info(f"Models loaded from {leak_proof_path} (LEAK-PROOF)")
                return

            # Fallback: Try loading from combined model file (legacy format)
            combined_path = self.model_dir / "spy_robust_models.joblib"
            if combined_path.exists():
                data = joblib.load(combined_path)

                # Extract models
                if "models" in data:
                    models_data = data["models"]
                    # Swing models (ensemble of L2 + GB)
                    if "swing" in models_data:
                        self.models["swing_l2"] = models_data["swing"]["l2"]
                        self.models["swing_gb"] = models_data["swing"]["gb"]
                        logger.info("Loaded swing models (L2 + GB ensemble)")

                    # Timing models
                    if "timing" in models_data:
                        self.models["timing_l2"] = models_data["timing"]["l2"]
                        self.models["timing_gb"] = models_data["timing"]["gb"]
                        logger.info("Loaded timing models (L2 + GB ensemble)")

                    # Entry/exit model
                    if "entry_exit_timing" in models_data:
                        self.models["entry_exit"] = models_data["entry_exit_timing"]
                        logger.info("Loaded entry/exit timing model")

                # Extract scaler
                if "scaler" in data:
                    self.scaler = data["scaler"]
                    logger.info("Loaded scaler")

                # Extract dim reduction state
                if "dim_reduction_state" in data:
                    self.dim_state = data["dim_reduction_state"]
                    logger.info("Loaded dimensionality reduction state")

                # Extract feature columns (NOTE: These are TRANSFORMED names in legacy model)
                if "feature_cols" in data:
                    self.feature_cols = data["feature_cols"]
                    logger.info(f"Loaded {len(self.feature_cols)} feature columns (legacy - transformed)")

                # Try to get original raw feature names from dim_state
                if self.dim_state and "var_selector" in self.dim_state:
                    # The var_selector knows the expected input dimension
                    expected_n = self.dim_state["var_selector"].n_features_in_
                    logger.info(f"Legacy model expects {expected_n} raw input features")

                # Store config for reference
                if "config" in data:
                    self.model_config = data["config"]

                logger.info(f"Models loaded from {combined_path} (LEGACY)")
                return

            # Fallback: try loading individual files (legacy format)
            swing_path = self.model_dir / "swing_model.joblib"
            timing_path = self.model_dir / "timing_model.joblib"
            scaler_path = self.model_dir / "scaler.joblib"
            dim_path = self.model_dir / "dim_reduction_state.joblib"
            features_path = self.model_dir / "feature_cols.joblib"

            if swing_path.exists():
                self.models["swing"] = joblib.load(swing_path)
                logger.info("Loaded swing model (legacy)")

            if timing_path.exists():
                self.models["timing"] = joblib.load(timing_path)
                logger.info("Loaded timing model (legacy)")

            if scaler_path.exists():
                self.scaler = joblib.load(scaler_path)
                logger.info("Loaded scaler (legacy)")

            if dim_path.exists():
                self.dim_state = joblib.load(dim_path)
                logger.info("Loaded dim reduction state (legacy)")

            if features_path.exists():
                self.feature_cols = joblib.load(features_path)
                logger.info(f"Loaded {len(self.feature_cols)} feature columns (legacy)")

            # Load entry/exit model if available
            entry_exit_path = self.model_dir / "entry_exit_model.joblib"
            if entry_exit_path.exists():
                self.models["entry_exit"] = joblib.load(entry_exit_path)
                logger.info("Loaded entry/exit timing model (legacy)")

        except Exception as e:
            logger.error(f"Failed to load models: {e}")

    # Features that are systematically biased when computed intraday
    # (training uses complete daily bars; inference sees partial-day data)
    PARTIAL_DAY_FEATURES = frozenset({
        "day_range", "day_range_pct", "day_volume", "day_return",
        "volume_ratio", "volume_20d_avg", "intraday_range",
        "vwap_deviation", "close_to_vwap",
    })

    def _validate_inference_features(
        self, df_daily: pd.DataFrame, feature_cols: list
    ) -> pd.DataFrame:
        """Validate and fix features that degrade during partial trading days.

        During market hours, today's daily OHLCV is incomplete — day_range,
        day_volume, RSI-on-close, etc. are systematically smaller/different
        than training data (computed on full-day bars).

        Strategy: for the LAST row (today), substitute prior-day value for
        known partial-day features if we're currently mid-session.
        """
        now = datetime.now()
        # Only apply during market hours (9:30 AM - 4:00 PM ET)
        if not (9 <= now.hour < 16 or (now.hour == 9 and now.minute >= 30)):
            return df_daily

        if len(df_daily) < 2:
            return df_daily

        affected = []
        for feat in feature_cols:
            # Match by exact name or prefix
            base = feat.split("_lag")[0] if "_lag" not in feat else feat
            if base in self.PARTIAL_DAY_FEATURES:
                # Substitute today's value with yesterday's
                if feat in df_daily.columns and len(df_daily) >= 2:
                    yesterday_val = df_daily[feat].iloc[-2]
                    today_val = df_daily[feat].iloc[-1]
                    if pd.notna(yesterday_val) and pd.notna(today_val):
                        # Only substitute if today's value looks truncated
                        # (e.g., volume much lower than yesterday, range much smaller)
                        if abs(today_val) < abs(yesterday_val) * 0.3 or today_val == 0:
                            df_daily.loc[df_daily.index[-1], feat] = yesterday_val
                            affected.append(feat)

        if affected:
            logger.info(
                f"[INFERENCE] Substituted {len(affected)} partial-day features "
                f"with prior-day values: {affected[:5]}"
                + ("..." if len(affected) > 5 else "")
            )

        return df_daily

    def prepare_features(self, df_1min: pd.DataFrame) -> np.ndarray:
        """
        Prepare features from 1-minute data.

        This mirrors the feature engineering in train_robust_model.py
        """
        # Import feature engineering functions
        try:
            from src.train_robust_model import engineer_all_features, add_rolling_features

            # Prepare the data by adding required columns
            df = df_1min.copy()

            # Ensure timestamp is a column, not index
            if "timestamp" not in df.columns:
                # Reset index to make it a column
                df = df.reset_index()
                # If the index was named something else (like the Alpaca timestamp column)
                # find the datetime column and rename it
                if "timestamp" not in df.columns:
                    for col in df.columns:
                        if pd.api.types.is_datetime64_any_dtype(df[col]):
                            df = df.rename(columns={col: "timestamp"})
                            break

            # Add "date" column (required by engineer_all_features)
            if "timestamp" not in df.columns:
                logger.error("No timestamp column found in data")
                return np.array([])
            df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Convert to EST for session detection (Alpaca returns UTC)
            if df["timestamp"].dt.tz is not None:
                df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")
            else:
                # Assume UTC if no timezone, convert to EST
                df["timestamp"] = df["timestamp"].dt.tz_localize("UTC").dt.tz_convert("America/New_York")

            # Create the columns expected by engineer_all_features
            df["date"] = df["timestamp"].dt.date
            df["time"] = df["timestamp"].dt.time
            df["hour"] = df["timestamp"].dt.hour
            df["minute"] = df["timestamp"].dt.minute

            # Add "session" column based on time of day (EST)
            # Premarket: 4:00 AM - 9:30 AM
            # Regular: 9:30 AM - 4:00 PM
            # Afterhours: 4:00 PM - 8:00 PM
            def get_session(row):
                h, m = row["hour"], row["minute"]
                if h < 4:
                    return "closed"
                elif h < 9 or (h == 9 and m < 30):
                    return "premarket"
                elif h < 16:
                    return "regular"
                elif h < 20:
                    return "afterhours"
                else:
                    return "closed"

            df["session"] = df.apply(get_session, axis=1)

            # Engineer daily features
            df_daily = engineer_all_features(df, swing_threshold=0.0025)
            df_daily = add_rolling_features(df_daily)

            if df_daily.empty:
                logger.error("Feature engineering returned empty dataframe")
                return np.array([])

            # Add anti-overfit features (MAG10, cross-assets, component streaks)
            # These are required because the model was trained with them
            try:
                from src.anti_overfit import integrate_anti_overfit
                df_daily, _ = integrate_anti_overfit(
                    df_daily,
                    spy_1min=df,  # Pass the 1min data for context
                    use_synthetic=False,  # Skip synthetic during inference
                    use_cross_assets=True,
                    use_breadth_streaks=True,
                    use_mag_breadth=True,
                )
            except Exception as e:
                logger.warning(f"Anti-overfit integration failed: {e}")

            # Get all numeric feature columns (the raw features)
            numeric_cols = df_daily.select_dtypes(include=[np.number]).columns
            exclude = ["target_up", "target_timing", "day_return", "sample_weight",
                       "is_up_day", "is_down_day", "low_before_high", "high_minutes", "low_minutes"]
            all_feature_cols = [c for c in numeric_cols if c not in exclude]

            if len(all_feature_cols) == 0:
                logger.error("No numeric features found in df_daily")
                return np.array([])

            # Validate and fix partial-day feature drift
            df_daily = self._validate_inference_features(df_daily, all_feature_cols)

            # ─────────────────────────────────────────────────────────────────────
            # LEAK-PROOF MODEL: Use saved feature columns and return raw features
            # The pipeline handles transformation internally
            # ─────────────────────────────────────────────────────────────────────
            if self.use_leak_proof and self.feature_cols:
                # Select only the features that were used during training, in the same order
                available_features = set(all_feature_cols)
                missing_features = [f for f in self.feature_cols if f not in available_features]
                if missing_features:
                    logger.warning(f"Missing {len(missing_features)} features: {missing_features[:5]}...")

                # Build feature array in the correct order, filling missing with 0
                X_list = []
                for feat in self.feature_cols:
                    if feat in df_daily.columns:
                        X_list.append(df_daily[feat].iloc[-1])
                    else:
                        X_list.append(0.0)  # Fill missing features with 0

                X = np.array(X_list).reshape(1, -1)

                # Handle NaN values
                if np.isnan(X).any():
                    nan_count = np.isnan(X).sum()
                    nan_indices = np.where(np.isnan(X[0]))[0]
                    nan_features = [self.feature_cols[i] for i in nan_indices[:5]]
                    logger.warning(f"Found {nan_count} NaN values in features: {nan_features}")
                    X = np.nan_to_num(X, nan=0.0)

                logger.debug(f"Prepared {X.shape[1]} features for leak-proof model")
                return X

            # ─────────────────────────────────────────────────────────────────────
            # LEGACY MODEL: Apply dimensionality reduction manually
            # ─────────────────────────────────────────────────────────────────────
            feature_cols = all_feature_cols
            X = df_daily[feature_cols].iloc[-1:].values

            # Handle NaN values - fill with 0
            if np.isnan(X).any():
                nan_count = np.isnan(X).sum()
                logger.warning(f"Found {nan_count} NaN values in features, filling with 0")
                X = np.nan_to_num(X, nan=0.0)

            # Apply dimensionality reduction (handles scaling internally via pre_transform_scaler)
            if self.dim_state:
                from src.train_robust_model import reduce_dimensions
                try:
                    # Check if we need to adjust feature count
                    var_selector = self.dim_state.get("var_selector")
                    expected_features = var_selector.n_features_in_ if var_selector else None

                    if expected_features and X.shape[1] != expected_features:
                        n_actual = X.shape[1]
                        n_diff = abs(n_actual - expected_features)
                        pct_diff = n_diff / expected_features

                        if pct_diff > 0.20:
                            # >20% feature mismatch — too risky for reliable predictions
                            logger.error(
                                f"Feature count mismatch too large: {n_actual} vs "
                                f"{expected_features} ({pct_diff:.0%} difference) — "
                                f"rejecting signal"
                            )
                            return np.array([])
                        elif n_actual > expected_features:
                            # Extra features — truncate to expected count
                            logger.warning(
                                f"Feature mismatch: {n_actual} > {expected_features} "
                                f"(+{n_diff} extra), truncating"
                            )
                            X = X[:, :expected_features]
                            feature_cols = feature_cols[:expected_features]
                        else:
                            # Missing features — pad with column medians from var_selector
                            # (better than zeros, which could be far from training distribution)
                            logger.warning(
                                f"Feature mismatch: {n_actual} < {expected_features} "
                                f"({n_diff} missing), padding with zeros"
                            )
                            padding = np.zeros((X.shape[0], expected_features - n_actual))
                            X = np.hstack([X, padding])

                    X, _, _ = reduce_dimensions(X, feature_cols, fit=False, state=self.dim_state)
                except Exception as e:
                    logger.error(f"Dimension reduction failed: {e}")
                    return np.array([])
            elif self.scaler:
                # Fallback: just scale if no dim_state
                X = self.scaler.transform(X)

            return X

        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            return np.array([])

    def generate_signal(
        self,
        df_1min: pd.DataFrame,
        current_price: float,
        entry_window: Tuple[int, int] = None,
        exit_window: Tuple[int, int] = None,
    ) -> TradingSignal:
        """
        Generate trading signal from current data.

        Args:
            df_1min: Recent 1-minute OHLCV data
            current_price: Current market price
            entry_window: Optional entry window override (start_min, end_min)
            exit_window: Optional exit window override (start_min, end_min)

        Returns:
            TradingSignal with recommendation
        """
        timestamp = datetime.now()
        symbol = TRADING_CONFIG["symbol"]

        # Default hold signal
        default_signal = TradingSignal(
            timestamp=timestamp,
            symbol=symbol,
            signal_type=SignalType.HOLD,
            probability=0.5,
            confidence=0.0,
        )

        # ═══════════════════════════════════════════════════════════════════
        # DYNAMIC MODEL SELECTOR PATH (REQUIRED)
        # Uses registry to select/ensemble best validated models
        # This is the ONLY proper path for production signal generation
        # ═══════════════════════════════════════════════════════════════════
        if self.dynamic_selector and self.dynamic_selector.candidates:
            signal = self._generate_signal_dynamic(
                df_1min, current_price, entry_window, exit_window, default_signal
            )
            # If dynamic path returned a real signal, use it
            if signal.signal_type != SignalType.HOLD or signal.confidence > 0:
                return signal
            # Otherwise fall through to static path as backup
            logger.warning("Dynamic selector returned HOLD with zero confidence — trying static fallback")

        # ═══════════════════════════════════════════════════════════════════
        # STATIC MODEL PATH (DEGRADED MODE - NOT RECOMMENDED)
        # Uses single model without proper registry validation
        # WARNING: This path should only be used during initial setup
        # Run grid search to populate registry: python scripts/run_grid_search.py
        # ═══════════════════════════════════════════════════════════════════
        logger.warning("DEGRADED MODE: Using static model without registry validation")

        # Check if we have BOTH swing AND timing models (required)
        has_swing = "swing_pipeline" in self.models or "swing_l2" in self.models or "swing" in self.models
        has_timing = "timing_pipeline" in self.models or "timing_l2" in self.models or "timing" in self.models

        if not has_swing:
            logger.error("No swing model available - cannot generate signal")
            return default_signal

        if not has_timing:
            logger.error("No timing model available - cannot generate valid signal")
            return default_signal

        try:
            # Calculate and update volatility for dynamic thresholds
            if len(df_1min) >= 14:
                df_vol = df_1min.copy()
                if "high" in df_vol.columns and "low" in df_vol.columns and "close" in df_vol.columns:
                    tr = np.maximum(
                        df_vol["high"] - df_vol["low"],
                        np.maximum(
                            abs(df_vol["high"] - df_vol["close"].shift(1)),
                            abs(df_vol["low"] - df_vol["close"].shift(1))
                        )
                    )
                    atr = tr.rolling(14).mean().iloc[-1]
                    atr_pct = atr / current_price if current_price > 0 else 0
                    dynamic_thresholds.update_volatility(atr_pct)

            # Prepare features
            X = self.prepare_features(df_1min)

            if len(X) == 0:
                return default_signal

            # ─────────────────────────────────────────────────────────────────────
            # LEAK-PROOF MODEL: Use sklearn Pipeline (handles transformation internally)
            # ─────────────────────────────────────────────────────────────────────
            if self.use_leak_proof and "swing_pipeline" in self.models:
                swing_proba = self.models["swing_pipeline"].predict_proba(X)[0, 1]
                confidence_penalty = 1.0

                # Timing model - REQUIRED for proper signal generation
                if "timing_pipeline" not in self.models:
                    logger.warning("Timing model not available - cannot generate valid signal")
                    return default_signal  # Reject signal without timing validation

                timing_proba = self.models["timing_pipeline"].predict_proba(X)[0, 1]
                timing_disagreement = 0.0

                logger.debug(f"Leak-proof: swing={swing_proba:.3f}, timing={timing_proba:.3f}")

            # ─────────────────────────────────────────────────────────────────────
            # LEGACY MODEL: Manual ensemble of L2 + GB
            # ─────────────────────────────────────────────────────────────────────
            elif "swing_l2" in self.models and "swing_gb" in self.models:
                proba_l2 = self.models["swing_l2"].predict_proba(X)[0, 1]
                proba_gb = self.models["swing_gb"].predict_proba(X)[0, 1]

                # Calculate model disagreement
                disagreement = abs(proba_l2 - proba_gb)

                # Weighted ensemble with disagreement penalty
                # When models agree strongly, use simple average
                # When models disagree, weight towards the more confident model
                if disagreement > 0.2:
                    # Significant disagreement - use the more extreme (confident) prediction
                    # but reduce overall confidence
                    dist_l2 = abs(proba_l2 - 0.5)
                    dist_gb = abs(proba_gb - 0.5)
                    if dist_l2 > dist_gb:
                        swing_proba = 0.6 * proba_l2 + 0.4 * proba_gb
                    else:
                        swing_proba = 0.4 * proba_l2 + 0.6 * proba_gb
                    # Apply disagreement penalty to confidence later
                    confidence_penalty = 1 - (disagreement * 0.5)
                else:
                    swing_proba = (proba_l2 + proba_gb) / 2
                    confidence_penalty = 1.0

                # Log ensemble details for debugging
                logger.debug(f"Ensemble: L2={proba_l2:.3f}, GB={proba_gb:.3f}, disagree={disagreement:.3f}, final={swing_proba:.3f}")

                # Legacy timing models - REQUIRED
                timing_proba = None  # Will be set below or signal rejected
                timing_disagreement = 0.0
            else:
                swing_proba = self.models["swing"].predict_proba(X)[0, 1]
                confidence_penalty = 1.0
                timing_proba = None  # Will be set below or signal rejected
                timing_disagreement = 0.0

            # Legacy timing ensemble - REQUIRED for valid signal
            if not self.use_leak_proof and "timing_l2" in self.models and "timing_gb" in self.models:
                proba_l2 = self.models["timing_l2"].predict_proba(X)[0, 1]
                proba_gb = self.models["timing_gb"].predict_proba(X)[0, 1]
                timing_disagreement = abs(proba_l2 - proba_gb)

                if timing_disagreement > 0.2:
                    dist_l2 = abs(proba_l2 - 0.5)
                    dist_gb = abs(proba_gb - 0.5)
                    if dist_l2 > dist_gb:
                        timing_proba = 0.6 * proba_l2 + 0.4 * proba_gb
                    else:
                        timing_proba = 0.4 * proba_l2 + 0.6 * proba_gb
                else:
                    timing_proba = (proba_l2 + proba_gb) / 2
            elif "timing" in self.models:
                timing_proba = self.models["timing"].predict_proba(X)[0, 1]

            # REQUIRE timing model - reject signal if no timing validation
            if timing_proba is None:
                logger.warning("No timing model available - cannot generate valid signal")
                return default_signal

            # Temporal cascade integration
            if self.temporal_cascade is not None:
                try:
                    now = datetime.now()
                    market_open_hour = 9
                    market_open_min = 30
                    minutes_since_open = (now.hour - market_open_hour) * 60 + (now.minute - market_open_min)

                    if 0 <= minutes_since_open <= 390:  # During market hours
                        cascade_signal = self.temporal_cascade.generate_realtime_signal(
                            historical_features=X,
                            df_1min_today=df_1min,
                            minutes_since_open=minutes_since_open,
                        )
                        if cascade_signal is not None:
                            cascade_proba = cascade_signal.swing_direction
                            cw = self.cascade_blend_weight
                            swing_proba = (1 - cw) * swing_proba + cw * cascade_proba
                            logger.debug(f"Temporal cascade blended (w={cw:.2f}): cascade_p={cascade_proba:.3f}, final_swing_p={swing_proba:.3f}")
                except Exception as e:
                    logger.debug(f"Temporal cascade signal not available: {e}")

            # Get dynamic thresholds based on current market conditions
            dyn_thresholds = dynamic_thresholds.get_adjusted_thresholds(current_price)
            entry_threshold = dyn_thresholds["entry_threshold"]
            stop_loss_pct = dyn_thresholds["stop_loss_pct"]
            take_profit_pct = dyn_thresholds["take_profit_pct"]
            max_position_pct = dyn_thresholds["max_position_pct"]

            # Determine signal type using dynamic thresholds
            if swing_proba >= entry_threshold:
                # Bullish signal
                if timing_proba >= 0.5:
                    # Good timing (low before high expected)
                    signal_type = SignalType.BUY
                else:
                    # Wait for better entry
                    signal_type = SignalType.HOLD
            elif swing_proba <= (1 - entry_threshold):
                # Bearish signal (could short or avoid)
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD

            # Calculate position size based on confidence with dynamic limits
            # Apply disagreement penalty from ensemble
            base_confidence = abs(swing_proba - 0.5) * 2  # 0-1 scale
            confidence = base_confidence * confidence_penalty  # Reduce if models disagree
            strong_threshold = TRADING_CONFIG["strong_signal_threshold"]
            min_position_pct = TRADING_CONFIG["min_position_pct"]

            if swing_proba >= strong_threshold and confidence_penalty > 0.8:
                # Only use max position if models agree
                position_size = max_position_pct
            else:
                # Scale position size with confidence, capped by dynamic max
                position_size = min_position_pct + \
                    (max_position_pct - min_position_pct) * confidence

            # Calculate stop loss and take profit using dynamic thresholds
            if signal_type == SignalType.BUY:
                stop_loss = current_price * (1 - stop_loss_pct)
                take_profit = current_price * (1 + take_profit_pct)
            elif signal_type == SignalType.SELL:
                stop_loss = current_price * (1 + stop_loss_pct)
                take_profit = current_price * (1 - take_profit_pct)
            else:
                stop_loss = None
                take_profit = None

            # Log dynamic adjustments if any were applied
            if dyn_thresholds.get("adjustments_applied"):
                logger.debug(f"Dynamic adjustments: {dyn_thresholds['adjustments_applied']}")

            # Use entry/exit model for refined timing if available
            entry_exit_decision = {}
            if "entry_exit" in self.models and signal_type in [SignalType.BUY, SignalType.SELL]:
                try:
                    entry_exit_decision = self.models["entry_exit"].predict(
                        X,
                        swing_proba=swing_proba,
                        timing_proba=timing_proba,
                        current_price=current_price
                    )
                    if entry_exit_decision.get("stop_loss"):
                        stop_loss = entry_exit_decision["stop_loss"]
                    if entry_exit_decision.get("take_profit"):
                        take_profit = entry_exit_decision["take_profit"]
                    if entry_exit_decision.get("position_size_pct"):
                        position_size = entry_exit_decision["position_size_pct"]
                except Exception as e:
                    logger.warning(f"Entry/exit model prediction failed: {e}")

            return TradingSignal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=signal_type,
                probability=swing_proba,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size_pct=position_size,
                metadata={
                    "timing_proba": timing_proba,
                    "entry_exit_decision": entry_exit_decision,
                }
            )

        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return default_signal

    def _generate_signal_dynamic(
        self,
        df_1min: pd.DataFrame,
        current_price: float,
        entry_window: Tuple[int, int],
        exit_window: Tuple[int, int],
        default_signal: TradingSignal,
    ) -> TradingSignal:
        """
        Generate signal using dynamic model selector.

        Uses the registry to select/ensemble the best models for current conditions.
        """
        timestamp = datetime.now()
        symbol = TRADING_CONFIG["symbol"]

        try:
            # Calculate volatility for dynamic thresholds
            if len(df_1min) >= 14:
                df_vol = df_1min.copy()
                if "high" in df_vol.columns and "low" in df_vol.columns and "close" in df_vol.columns:
                    tr = np.maximum(
                        df_vol["high"] - df_vol["low"],
                        np.maximum(
                            abs(df_vol["high"] - df_vol["close"].shift(1)),
                            abs(df_vol["low"] - df_vol["close"].shift(1))
                        )
                    )
                    atr = tr.rolling(14).mean().iloc[-1]
                    atr_pct = atr / current_price if current_price > 0 else 0
                    dynamic_thresholds.update_volatility(atr_pct)

            # Prepare features
            X = self.prepare_features(df_1min)

            if len(X) == 0:
                return default_signal

            # Get feature names for the dynamic selector
            feature_names = self.feature_cols if self.feature_cols else []

            # Get prediction from dynamic selector (ensembles best models)
            prediction = self.dynamic_selector.predict(
                features=X,
                feature_names=feature_names,
                entry_window=entry_window,
                exit_window=exit_window,
                n_models=5,
            )

            swing_proba = prediction.swing_probability
            timing_proba = prediction.timing_probability
            confidence = prediction.confidence
            direction = prediction.direction

            # Bounds check: clamp probabilities to [0, 1], reject NaN
            if any(np.isnan(v) for v in (swing_proba, timing_proba, confidence)):
                logger.warning(
                    f"NaN detected in predictions: swing={swing_proba}, "
                    f"timing={timing_proba}, conf={confidence} — returning HOLD"
                )
                return default_signal
            swing_proba = float(np.clip(swing_proba, 0.0, 1.0))
            timing_proba = float(np.clip(timing_proba, 0.0, 1.0))
            confidence = float(np.clip(confidence, 0.0, 1.0))

            # Log ensemble details
            logger.info(
                f"Dynamic ensemble: direction={direction}, swing={swing_proba:.3f}, "
                f"timing={timing_proba:.3f}, confidence={confidence:.3f}, "
                f"n_models={prediction.n_models}, agreement={prediction.agreement_ratio:.2f}"
            )

            # Temporal cascade integration (dynamic path)
            if self.temporal_cascade is not None:
                try:
                    now = datetime.now()
                    minutes_since_open = (now.hour - 9) * 60 + (now.minute - 30)

                    if 0 <= minutes_since_open <= 390:  # During market hours
                        cascade_signal = self.temporal_cascade.generate_realtime_signal(
                            historical_features=X,
                            df_1min_today=df_1min,
                            minutes_since_open=minutes_since_open,
                        )
                        if cascade_signal is not None:
                            cascade_proba = cascade_signal.swing_direction
                            cw = self.cascade_blend_weight
                            swing_proba = (1 - cw) * swing_proba + cw * cascade_proba
                            logger.debug(f"Temporal cascade blended dynamic (w={cw:.2f}): cascade_p={cascade_proba:.3f}, final_swing_p={swing_proba:.3f}")
                except Exception as e:
                    logger.debug(f"Temporal cascade signal not available (dynamic): {e}")

            # Get dynamic thresholds
            dyn_thresholds = dynamic_thresholds.get_adjusted_thresholds(current_price)
            entry_threshold = dyn_thresholds["entry_threshold"]
            stop_loss_pct = dyn_thresholds["stop_loss_pct"]
            take_profit_pct = dyn_thresholds["take_profit_pct"]
            max_position_pct = dyn_thresholds["max_position_pct"]

            # Convert direction to signal type
            if direction == "LONG" and swing_proba >= entry_threshold:
                signal_type = SignalType.BUY
            elif direction == "SHORT" and swing_proba <= (1 - entry_threshold):
                signal_type = SignalType.SELL
            else:
                signal_type = SignalType.HOLD

            # Use ensemble's position sizing suggestion, capped by risk limits
            position_size = min(
                prediction.confidence_adjusted_position_pct,
                max_position_pct
            )
            position_size = max(position_size, TRADING_CONFIG["min_position_pct"])

            # Calculate stop loss and take profit
            if signal_type == SignalType.BUY:
                stop_loss = current_price * (1 - stop_loss_pct)
                take_profit = current_price * (1 + take_profit_pct)
            elif signal_type == SignalType.SELL:
                stop_loss = current_price * (1 + stop_loss_pct)
                take_profit = current_price * (1 - take_profit_pct)
            else:
                stop_loss = None
                take_profit = None

            return TradingSignal(
                timestamp=timestamp,
                symbol=symbol,
                signal_type=signal_type,
                probability=swing_proba,
                confidence=confidence,
                entry_price=current_price,
                stop_loss=stop_loss,
                take_profit=take_profit,
                position_size_pct=position_size,
                metadata={
                    "timing_proba": timing_proba,
                    "ensemble_method": prediction.ensemble_method,
                    "n_models": prediction.n_models,
                    "agreement_ratio": prediction.agreement_ratio,
                    "entry_window": prediction.entry_window,
                    "exit_window": prediction.exit_window,
                    "model_predictions": prediction.model_predictions[:3],  # Top 3
                    "dynamic_selector": True,
                }
            )

        except Exception as e:
            logger.error(f"Dynamic signal generation failed: {e}")
            return default_signal

    def get_selector_status(self) -> Dict:
        """Get status of the dynamic model selector."""
        if self.dynamic_selector:
            return self.dynamic_selector.get_status()
        return {"dynamic_selector": False, "mode": "static"}
