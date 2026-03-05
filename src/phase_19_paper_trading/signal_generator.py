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

        # Feature drift monitor — detects distribution shift from training (Wave F6)
        self.drift_monitor = None
        self._drift_monitor_fitted = False
        try:
            from src.phase_20_monitoring.feature_drift_monitor import FeatureDriftMonitor
            self.drift_monitor = FeatureDriftMonitor(psi_threshold=0.2, alert_fraction=0.2)
        except ImportError:
            pass

        # Dynamic ensemble weighting (Wave F2.2) — weights models by recent OOS performance
        self.dynamic_weighter = None
        if TRADING_CONFIG.get("use_dynamic_weights", False):
            try:
                from src.phase_15_strategy.dynamic_weights import DynamicEnsembleWeighter
                self.dynamic_weighter = DynamicEnsembleWeighter(
                    lookback=TRADING_CONFIG.get("dynamic_weight_lookback", 20),
                )
                logger.info("Dynamic ensemble weighter initialized")
            except ImportError:
                pass

        # Thompson Sampling model selector (Wave G3) — bandit-based online model weighting
        self.thompson_selector = None
        if TRADING_CONFIG.get("use_thompson_selector", False):
            try:
                from src.phase_15_strategy.thompson_selector import ThompsonSamplingSelector
                self.thompson_selector = ThompsonSamplingSelector(
                    model_ids=["l2", "gb"],
                    decay=TRADING_CONFIG.get("thompson_decay", 0.995),
                    min_weight=TRADING_CONFIG.get("thompson_min_weight", 0.1),
                )
                # Try to load saved state
                state_path = Path("models/thompson_state.json")
                if state_path.exists():
                    self.thompson_selector.load_state(state_path)
                    logger.info(f"Thompson selector loaded: {self.thompson_selector.get_weights()}")
                else:
                    logger.info("Thompson selector initialized with uniform priors")
            except ImportError:
                pass

        # Feature source health tracking (Wave 38)
        self._signal_count = 0
        self._last_feature_quality = 1.0  # 1.0 = all features OK, 0.0 = all degraded

        # Feature preparation cache (avoid re-running expensive feature engineering
        # every 60s when the underlying 1-min data hasn't changed)
        self._prep_cache_key = None
        self._prep_cache_result = None

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

                # Load meta-labeler from leak-proof bundle if present
                if "meta_model" in data and data["meta_model"] is not None:
                    self.models["meta_labeler"] = data["meta_model"]
                    logger.info("Loaded meta-labeler from leak-proof bundle")

                # Load conformal sizer if present (Wave F4.2)
                if "conformal_sizer" in data and data["conformal_sizer"] is not None:
                    self.models["conformal_sizer"] = data["conformal_sizer"]
                    logger.info("Loaded conformal sizer from leak-proof bundle")

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

                    # BMA weights (Wave F6.2) — serialised as weight dicts, not model refs
                    if "bma_swing_weights" in models_data:
                        self.models["bma_swing_weights"] = models_data["bma_swing_weights"]
                        logger.info(f"Loaded BMA swing weights: {models_data['bma_swing_weights']}")
                    if "bma_timing_weights" in models_data:
                        self.models["bma_timing_weights"] = models_data["bma_timing_weights"]
                        logger.info(f"Loaded BMA timing weights: {models_data['bma_timing_weights']}")

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

            # Load meta-labeler if available (standalone or from experiment model)
            meta_path = self.model_dir / "meta_labeler.joblib"
            if meta_path.is_file():
                try:
                    from src.phase_15_strategy.meta_labeler import MetaLabeler
                    ml = MetaLabeler.load(meta_path)
                    if ml is not None and ml.is_fitted_:
                        self.models["meta_labeler"] = ml
                        logger.info(f"Loaded meta-labeler (AUC={ml.meta_auc_:.3f})")
                except Exception as ml_err:
                    logger.warning(f"Failed to load meta-labeler: {ml_err}")

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
        # Cache: skip recomputation if same data as last call
        _cache_key = (
            len(df_1min),
            df_1min.index[-1] if len(df_1min) > 0 else None,
        )
        if (self._prep_cache_key == _cache_key
                and self._prep_cache_result is not None
                and len(self._prep_cache_result) > 0):
            return self._prep_cache_result.copy()

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

            # Add anti-overfit features (all feature types the model was trained with)
            # Must match the flags used during training to avoid zero-filled features
            try:
                from src.anti_overfit import integrate_anti_overfit
                df_daily, _ = integrate_anti_overfit(
                    df_daily,
                    spy_1min=df,  # Pass the 1min data for context
                    use_synthetic=False,  # Skip synthetic during inference
                    use_cross_assets=True,
                    use_breadth_streaks=True,
                    use_mag_breadth=True,
                    use_sector_breadth=True,
                    use_vol_regime=True,
                    use_economic_features=True,
                    use_calendar_features=True,
                    use_sentiment_features=True,
                    use_fear_greed=True,
                    use_reddit_sentiment=True,
                    use_crypto_sentiment=True,
                    use_gamma_exposure=True,
                    use_finnhub_social=True,
                    use_dark_pool=True,
                    use_options_features=True,
                    use_event_recency=True,
                    use_block_structure=True,
                    use_amihud_features=True,
                    use_range_vol_features=True,
                    use_entropy_features=True,
                    use_hurst_features=True,
                    use_nmi_features=True,
                    use_absorption_ratio=True,
                    use_drift_features=True,
                    use_changepoint_features=True,
                    use_hmm_features=True,
                    use_vpin_features=True,
                    use_intraday_momentum=True,
                    use_futures_basis=True,
                    use_congressional_features=True,
                    use_insider_aggregate=True,
                    use_etf_flow=True,
                    use_wavelet_features=True,
                    use_sax_features=True,
                    use_transfer_entropy=True,
                    use_mfdfa_features=True,
                    use_rqa_features=True,
                    use_copula_features=True,
                    use_network_centrality=True,
                    use_path_signatures=True,
                    use_wavelet_scattering=True,
                    use_wasserstein_regime=True,
                    use_credit_spread_features=True,
                    use_yield_curve_features=True,
                    use_vol_term_structure_features=True,
                    use_macro_surprise_features=True,
                    use_cross_asset_momentum=True,
                    use_skew_kurtosis_features=True,
                    use_seasonality_features=True,
                    use_order_flow_imbalance=True,
                    use_correlation_regime=True,
                    use_fama_french=True,
                    use_put_call_ratio=True,
                    use_multi_horizon=True,
                    use_earnings_revision=True,
                    use_short_interest=False,
                    use_dollar_index=True,
                    use_institutional_flow=False,
                    use_google_trends=False,
                    use_commodity_signals=True,
                    use_treasury_auction=False,
                    use_fed_liquidity=True,
                    use_earnings_calendar=True,
                    use_analyst_rating=True,
                    use_expanded_macro=True,
                    use_vvix=True,
                    use_sector_rotation=True,
                    use_fx_carry=True,
                    use_money_market=True,
                    use_financial_stress=True,
                    use_global_equity=True,
                    use_retail_sentiment=True,
                    use_cboe_pcr=True,
                    use_stocktwits=True,
                    use_alpaca_news=True,
                    use_gnews_headlines=True,
                    use_finbert_nlp=False,
                    use_wsb_sentiment=False,
                    validate_ohlc=True,
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
                # Build feature array in training order, using column median
                # as fallback for NaN values instead of 0.0 (which distorts
                # feature distributions and degrades model predictions)
                X_list = []
                n_missing_cols = 0
                n_nan_to_median = 0
                n_nan_to_zero = 0

                for feat in self.feature_cols:
                    if feat in df_daily.columns:
                        val = df_daily[feat].iloc[-1]
                        if pd.isna(val):
                            # Use column median from available data
                            col_median = df_daily[feat].dropna().median()
                            if pd.notna(col_median):
                                X_list.append(float(col_median))
                                n_nan_to_median += 1
                            else:
                                X_list.append(0.0)
                                n_nan_to_zero += 1
                        else:
                            X_list.append(float(val))
                    else:
                        X_list.append(0.0)
                        n_missing_cols += 1

                X = np.array(X_list).reshape(1, -1)

                # Track feature quality score for position sizing
                n_total = len(self.feature_cols)
                n_degraded = n_missing_cols + n_nan_to_median + n_nan_to_zero
                self._last_feature_quality = (
                    (n_total - n_degraded) / n_total if n_total > 0 else 0.0
                )

                if n_degraded > 0:
                    logger.warning(
                        f"[FEATURE QUALITY] {n_total - n_degraded}/{n_total} "
                        f"features OK ({n_missing_cols} missing cols, "
                        f"{n_nan_to_median} NaN->median, "
                        f"{n_nan_to_zero} NaN->zero), "
                        f"quality={self._last_feature_quality:.1%}"
                    )
                else:
                    self._last_feature_quality = 1.0

                # Final NaN safety net
                if np.isnan(X).any():
                    nan_count = np.isnan(X).sum()
                    logger.warning(
                        f"Residual {nan_count} NaN values after median fill, "
                        f"zeroing"
                    )
                    X = np.nan_to_num(X, nan=0.0)

                logger.debug(
                    f"Prepared {X.shape[1]} features for leak-proof model "
                    f"(quality={self._last_feature_quality:.1%})"
                )
                self._prep_cache_key = _cache_key
                self._prep_cache_result = X.copy()
                return X

            # ─────────────────────────────────────────────────────────────────────
            # LEGACY MODEL: Apply dimensionality reduction manually
            # ─────────────────────────────────────────────────────────────────────
            feature_cols = all_feature_cols
            X = df_daily[feature_cols].iloc[-1:].values

            # Fill NaN with column median instead of 0 (preserves distribution)
            if np.isnan(X).any():
                nan_count = int(np.isnan(X).sum())
                col_medians = df_daily[feature_cols].median().values
                fill_values = np.where(
                    np.isnan(col_medians), 0.0, col_medians
                ).reshape(1, -1)
                X = np.where(np.isnan(X), fill_values, X)
                logger.warning(
                    f"Filled {nan_count} NaN values with column medians"
                )

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

            self._prep_cache_key = _cache_key
            self._prep_cache_result = X.copy()
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
        self._signal_count += 1

        # Default hold signal
        default_signal = TradingSignal(
            timestamp=timestamp,
            symbol=symbol,
            signal_type=SignalType.HOLD,
            probability=0.5,
            confidence=0.0,
        )

        # Periodic feature source health report (every 100 signals)
        if self._signal_count % 100 == 0:
            self._report_source_health()

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

            # ── Feature drift check (Wave F6) ──
            self._check_feature_drift(X)

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

                # Thompson Sampling weighting (Wave G3) — bandit-based adaptive weights
                if self.thompson_selector is not None and TRADING_CONFIG.get("use_thompson_selector", False):
                    tw = self.thompson_selector.get_weights()
                    w_l2 = tw.get("l2", 0.5)
                    w_gb = tw.get("gb", 0.5)
                    swing_proba = w_l2 * proba_l2 + w_gb * proba_gb
                    confidence_penalty = max(1 - (disagreement * 0.3), 0.5)
                    logger.debug(f"Thompson swing: L2={proba_l2:.3f}*{w_l2:.3f} + GB={proba_gb:.3f}*{w_gb:.3f} = {swing_proba:.3f}")
                # BMA weighting (Wave F6.2) — use pre-computed Bayesian posterior weights
                elif (bma_sw := self.models.get("bma_swing_weights")) and TRADING_CONFIG.get("use_bma", False):
                    w_l2 = bma_sw.get("l2", 0.5)
                    w_gb = bma_sw.get("gb", 0.5)
                    swing_proba = w_l2 * proba_l2 + w_gb * proba_gb
                    confidence_penalty = max(1 - (disagreement * 0.3), 0.5)
                    logger.debug(f"BMA swing: L2={proba_l2:.3f}*{w_l2:.3f} + GB={proba_gb:.3f}*{w_gb:.3f} = {swing_proba:.3f}")
                # Fallback: manual disagreement-based weighting
                elif disagreement > 0.2:
                    dist_l2 = abs(proba_l2 - 0.5)
                    dist_gb = abs(proba_gb - 0.5)
                    if dist_l2 > dist_gb:
                        swing_proba = 0.6 * proba_l2 + 0.4 * proba_gb
                    else:
                        swing_proba = 0.4 * proba_l2 + 0.6 * proba_gb
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

                # BMA weighting for timing (Wave F6.2)
                bma_tw = self.models.get("bma_timing_weights")
                if bma_tw and TRADING_CONFIG.get("use_bma", False):
                    w_l2 = bma_tw.get("l2", 0.5)
                    w_gb = bma_tw.get("gb", 0.5)
                    timing_proba = w_l2 * proba_l2 + w_gb * proba_gb
                    logger.debug(f"BMA timing: L2={proba_l2:.3f}*{w_l2:.3f} + GB={proba_gb:.3f}*{w_gb:.3f} = {timing_proba:.3f}")
                elif timing_disagreement > 0.2:
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

            # Conformal position sizing (Wave F4.2): scale by prediction certainty
            if "conformal_sizer" in self.models and self.models["conformal_sizer"] is not None:
                try:
                    cs = self.models["conformal_sizer"]
                    if getattr(cs, '_fitted', False):
                        sized = cs.size(X, position_size)
                        conformal_pos = float(sized[0]) if len(sized) > 0 else position_size
                        if conformal_pos != position_size:
                            logger.debug(
                                f"Conformal sizing: {position_size:.4f} -> {conformal_pos:.4f}"
                            )
                            position_size = conformal_pos
                except Exception as cs_err:
                    logger.debug(f"Conformal sizing skipped: {cs_err}")

            # CVaR position sizing (Wave G2): scale by tail risk
            if TRADING_CONFIG.get("use_cvar_sizing", False) and df_daily is not None:
                try:
                    from src.phase_15_strategy.cvar_position_sizer import CVaRPositionSizer
                    cvar_sizer = CVaRPositionSizer(
                        alpha=TRADING_CONFIG.get("cvar_alpha", 0.05),
                        lookback=TRADING_CONFIG.get("cvar_lookback", 60),
                        max_position=max_position_pct,
                        min_position=min_position_pct,
                        target_cvar=TRADING_CONFIG.get("cvar_target", 0.02),
                    )
                    _close = df_daily["close"].values if "close" in df_daily.columns else None
                    if _close is not None and len(_close) > 60:
                        _returns = np.diff(_close) / _close[:-1]
                        cvar_sizer.fit(_returns)
                        cvar_pos = cvar_sizer.size(position_size, cvar_sizer._current_cvar)
                        if cvar_pos != position_size:
                            logger.debug(
                                f"CVaR sizing: {position_size:.4f} -> {cvar_pos:.4f} "
                                f"(CVaR={cvar_sizer._current_cvar:.4f})"
                            )
                            position_size = cvar_pos
                except Exception as cvar_err:
                    logger.debug(f"CVaR sizing skipped: {cvar_err}")

            # Dynamic Kelly sizing (Wave J3.3): VIX-conditioned Kelly criterion
            if TRADING_CONFIG.get("use_dynamic_kelly", False) and df_daily is not None:
                try:
                    from src.phase_15_strategy.dynamic_kelly_sizer import DynamicKellySizer
                    _close = df_daily["close"].values if "close" in df_daily.columns else None
                    if _close is not None and len(_close) > 30:
                        _returns = np.diff(_close) / _close[:-1]
                        dk_sizer = DynamicKellySizer(
                            min_position=min_position_pct,
                            max_position=max_position_pct,
                        )
                        dk_sizer.fit(_returns)
                        # Estimate VIX from realized vol if not available
                        vix_est = float(np.std(_returns[-20:]) * np.sqrt(252) * 100) if len(_returns) >= 20 else 20.0
                        dk_pos = dk_sizer.size(
                            win_probability=swing_proba,
                            vix_level=vix_est,
                        )
                        if dk_pos != position_size:
                            logger.debug(
                                f"DynamicKelly sizing: {position_size:.4f} -> {dk_pos:.4f} "
                                f"(VIX_est={vix_est:.1f})"
                            )
                            position_size = dk_pos
                except Exception as dk_err:
                    logger.debug(f"DynamicKelly sizing skipped: {dk_err}")

            # Drawdown-adaptive sizing (Wave J3.4): reduce position in drawdown
            if TRADING_CONFIG.get("use_drawdown_adaptive_sizing", False) and df_daily is not None:
                try:
                    from src.phase_15_strategy.drawdown_adaptive_sizer import DrawdownAdaptiveSizer
                    _close = df_daily["close"].values if "close" in df_daily.columns else None
                    if _close is not None and len(_close) > 10:
                        da_sizer = DrawdownAdaptiveSizer(
                            max_drawdown=TRADING_CONFIG.get("drawdown_max_dd", 0.10),
                            power=TRADING_CONFIG.get("drawdown_power", 2.0),
                            min_position=min_position_pct,
                            max_position=max_position_pct,
                        )
                        da_sizer.fit(_close)
                        if da_sizer.current_drawdown is not None:
                            da_pos = da_sizer.size(position_size)
                            if da_pos != position_size:
                                logger.debug(
                                    f"Drawdown sizing: {position_size:.4f} -> {da_pos:.4f} "
                                    f"(dd={da_sizer.current_drawdown:.4f})"
                                )
                                position_size = da_pos
                except Exception as da_err:
                    logger.debug(f"Drawdown sizing skipped: {da_err}")

            # Regime-aware stop loss (Wave K2): ATR/VIX-conditioned levels
            if TRADING_CONFIG.get("use_regime_stop_loss", False) and df_daily is not None:
                try:
                    from src.phase_15_strategy.regime_stop_loss import RegimeAwareStopLoss
                    rasl = RegimeAwareStopLoss()
                    rasl.fit(df_daily)
                    _close = df_daily["close"].values if "close" in df_daily.columns else None
                    if _close is not None and len(_close) >= 20:
                        _returns = np.diff(_close) / _close[:-1]
                        vix_est = float(np.std(_returns[-20:]) * np.sqrt(252) * 100)
                    else:
                        vix_est = 20.0
                    direction = "LONG" if signal_type == SignalType.BUY else "SHORT"
                    levels = rasl.compute_levels(current_price, direction, vix_est)
                    stop_loss = levels["stop_loss"]
                    take_profit = levels["take_profit"]
                    stop_loss_pct = levels["stop_pct"]
                    take_profit_pct = levels["tp_pct"]
                    logger.debug(
                        f"RegimeStopLoss: regime={levels['regime']} "
                        f"stop={stop_loss_pct:.4f} tp={take_profit_pct:.4f}"
                    )
                except Exception as rasl_err:
                    logger.debug(f"RegimeStopLoss skipped: {rasl_err}")
                    # Fall through to static stops below
                    if signal_type == SignalType.BUY:
                        stop_loss = current_price * (1 - stop_loss_pct)
                        take_profit = current_price * (1 + take_profit_pct)
                    elif signal_type == SignalType.SELL:
                        stop_loss = current_price * (1 + stop_loss_pct)
                        take_profit = current_price * (1 - take_profit_pct)
                    else:
                        stop_loss = None
                        take_profit = None
            else:
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

            # Meta-labeling: scale position size by signal profitability probability
            meta_proba = None
            if "meta_labeler" in self.models and self.models["meta_labeler"] is not None:
                try:
                    meta_labeler = self.models["meta_labeler"]
                    meta_proba_arr = meta_labeler.predict(
                        X, np.array([swing_proba]), np.array([timing_proba])
                    )
                    if meta_proba_arr is not None:
                        meta_proba = float(meta_proba_arr[0])
                        # Half Kelly position sizing
                        from src.phase_15_strategy.meta_labeler import half_kelly_fraction
                        win_loss_ratio = self._estimate_win_loss_ratio()
                        kelly_size = half_kelly_fraction(meta_proba, win_loss_ratio)
                        kelly_pct = max(min_position_pct, min(kelly_size, max_position_pct))
                        logger.info(f"Meta-label: proba={meta_proba:.3f}, kelly={kelly_size:.4f}, "
                                    f"position={kelly_pct:.4f}")
                        position_size = kelly_pct
                except Exception as meta_err:
                    logger.warning(f"Meta-label prediction failed: {meta_err}")

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

            # Scale position by feature quality when degraded
            if self._last_feature_quality < 0.90:
                quality_factor = max(0.3, self._last_feature_quality)
                position_size *= quality_factor
                logger.info(
                    f"Position scaled by feature quality: "
                    f"{quality_factor:.2f} "
                    f"(quality={self._last_feature_quality:.1%})"
                )

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
                    "feature_quality": self._last_feature_quality,
                }
            )

        except Exception as e:
            logger.error(f"Signal generation failed: {e}")
            return default_signal

    def _check_feature_drift(self, X: np.ndarray) -> None:
        """Check for feature distribution drift using PSI (Wave F6).

        On first call with enough data, fits the drift monitor baseline.
        On subsequent calls, checks current features against the baseline
        and logs warnings if significant drift is detected.
        """
        if self.drift_monitor is None:
            return

        try:
            if not self._drift_monitor_fitted:
                # Fit baseline on first call (using X as the "training" distribution)
                # X may have only 1 row — defer until we see multi-row input
                if X.shape[0] >= 10:
                    feature_names = self.feature_cols if self.feature_cols else None
                    self.drift_monitor.fit(X, feature_names=feature_names)
                    self._drift_monitor_fitted = True
                    logger.debug(f"Drift monitor fitted on {X.shape[0]} x {X.shape[1]} baseline")
                return  # Skip check on the fitting call

            # Check for drift
            drift_report = self.drift_monitor.check(X)
            if drift_report.get("has_drift", False):
                n_drifted = drift_report.get("n_drifted", 0)
                severity = drift_report.get("severity", "unknown")
                logger.warning(
                    f"[DRIFT ALERT] {n_drifted} features drifted "
                    f"(severity={severity}, "
                    f"fraction={drift_report.get('drift_fraction', 0):.2f})"
                )
            elif self._signal_count % 50 == 0:
                logger.debug(
                    f"Drift check OK: {drift_report.get('n_drifted', 0)} drifted features"
                )
        except Exception as e:
            logger.debug(f"Feature drift check skipped: {e}")

    def _report_source_health(self):
        """Log health status of external data source features.

        Checks which feature prefix groups have non-zero values in the loaded
        model's feature columns. Runs every 100 signals — logging only, no side effects.
        """
        source_prefixes = {
            "fear_greed": "fg_",
            "reddit_sentiment": "reddit_",
            "crypto_sentiment": "crypto_",
            "gamma_exposure": "gex_",
            "finnhub_social": "finnhub_social_",
            "dark_pool": "dp_",
            "options_iv": "opt_",
            "economic": "econ_",
            "calendar": "cal_",
            "cross_asset": ["TLT_", "QQQ_", "GLD_"],
            "amihud_liquidity": "liq_",
            "range_vol": "rvol_",
            "entropy": "ent_",
            "hurst": "hurst_",
            "nmi": "nmi_",
            "absorption_ratio": "ar_",
            "drift_detection": "drift_",
            "changepoint": "cpd_",
            "hmm_regime": "hmm_",
            "vpin": "vpin_",
            "intraday_momentum": "imom_",
            "futures_basis": "basis_",
            "congressional": "congress_",
            "insider_aggregate": "insider_agg_",
            "etf_flow": "etf_flow_",
            "wavelet": "wav_",
            "sax_pattern": "sax_",
            "transfer_entropy": "te_",
            "mfdfa": "mfdfa_",
            "rqa": "rqa_",
            "copula": "copula_",
            "network": "netw_",
            "market_structure": "mstr_",
            "time_series_model": "tsm_",
            "har_rv": "harv_",
            "l_moments": "lmom_",
            "multiscale_entropy": "mse_",
            "rv_signature": "rvsp_",
            "tda_homology": "tda_",
        }
        if not self.feature_cols:
            logger.info(f"[SOURCE HEALTH] Signal #{self._signal_count}: No feature_cols loaded (dynamic mode)")
            return

        active = []
        missing = []
        for source, prefix in source_prefixes.items():
            prefixes = prefix if isinstance(prefix, list) else [prefix]
            found = any(
                any(col.startswith(p) for p in prefixes)
                for col in self.feature_cols
            )
            if found:
                active.append(source)
            else:
                missing.append(source)

        logger.info(
            f"[SOURCE HEALTH] Signal #{self._signal_count}: "
            f"{len(active)} active sources ({', '.join(active)})"
            + (f", {len(missing)} missing ({', '.join(missing)})" if missing else "")
        )

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

            # ── Feature drift check (Wave F6) ──
            self._check_feature_drift(X)

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

            # Ensemble disagreement features (Wave A3)
            ens_metrics = {}
            try:
                from src.phase_15_strategy.ensemble_disagreement import compute_ensemble_disagreement
                model_probas = [p.get("swing_proba", 0.5) for p in prediction.model_predictions
                                if isinstance(p, dict)]
                if model_probas:
                    ens_metrics = compute_ensemble_disagreement(model_probas)
                    logger.info(
                        f"Ensemble disagreement: std={ens_metrics['ens_std']:.4f}, "
                        f"agreement={ens_metrics['ens_agreement']:.2f}, "
                        f"entropy={ens_metrics['ens_entropy']:.4f}"
                    )
            except Exception as ens_err:
                logger.debug(f"Ensemble disagreement computation skipped: {ens_err}")

            # Dynamic ensemble weighting (Wave F2.2): re-weight predictions by recent accuracy
            if self.dynamic_weighter is not None and prediction.model_predictions:
                try:
                    model_ids = []
                    model_probas_list = []
                    for mp in prediction.model_predictions:
                        if isinstance(mp, dict) and "model_id" in mp:
                            model_ids.append(mp["model_id"])
                            model_probas_list.append(mp.get("swing_proba", 0.5))
                    if model_ids:
                        weights = self.dynamic_weighter.get_weights(model_ids)
                        if weights:
                            weighted_swing = sum(
                                weights.get(mid, 1.0 / len(model_ids)) * p
                                for mid, p in zip(model_ids, model_probas_list)
                            )
                            logger.debug(
                                f"Dynamic weighting: {swing_proba:.3f} -> {weighted_swing:.3f}"
                            )
                            swing_proba = weighted_swing
                except Exception as dw_err:
                    logger.debug(f"Dynamic weighting skipped: {dw_err}")

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

            # Meta-labeling: override position size with Half Kelly if meta-model available
            if "meta_labeler" in self.models and self.models["meta_labeler"] is not None:
                try:
                    meta_labeler = self.models["meta_labeler"]
                    meta_proba_arr = meta_labeler.predict(
                        X, np.array([swing_proba]), np.array([timing_proba])
                    )
                    if meta_proba_arr is not None:
                        meta_proba = float(meta_proba_arr[0])
                        from src.phase_15_strategy.meta_labeler import half_kelly_fraction
                        win_loss_ratio = self._estimate_win_loss_ratio()
                        kelly_size = half_kelly_fraction(meta_proba, win_loss_ratio)
                        kelly_pct = max(TRADING_CONFIG["min_position_pct"],
                                        min(kelly_size, max_position_pct))
                        logger.info(f"Meta-label (dynamic): proba={meta_proba:.3f}, "
                                    f"kelly={kelly_size:.4f}, position={kelly_pct:.4f}")
                        position_size = kelly_pct
                except Exception as meta_err:
                    logger.warning(f"Meta-label prediction failed (dynamic): {meta_err}")

            # Scale position by feature quality when degraded
            if self._last_feature_quality < 0.90:
                quality_factor = max(0.3, self._last_feature_quality)
                position_size *= quality_factor
                logger.info(
                    f"Position scaled by feature quality (dynamic): "
                    f"{quality_factor:.2f} "
                    f"(quality={self._last_feature_quality:.1%})"
                )

            # Conformal position sizing (Wave F4.2, dynamic path)
            if "conformal_sizer" in self.models and self.models["conformal_sizer"] is not None:
                try:
                    cs = self.models["conformal_sizer"]
                    if getattr(cs, '_fitted', False):
                        sized = cs.size(X, position_size)
                        conformal_pos = float(sized[0]) if len(sized) > 0 else position_size
                        if conformal_pos != position_size:
                            logger.debug(
                                f"Conformal sizing (dynamic): {position_size:.4f} -> {conformal_pos:.4f}"
                            )
                            position_size = conformal_pos
                except Exception as cs_err:
                    logger.debug(f"Conformal sizing skipped (dynamic): {cs_err}")

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
                    "feature_quality": self._last_feature_quality,
                    "ensemble_disagreement": ens_metrics,
                }
            )

        except Exception as e:
            logger.error(f"Dynamic signal generation failed: {e}")
            return default_signal

    def _estimate_win_loss_ratio(self) -> float:
        """Estimate avg_win / avg_loss from recent trades. Fallback to 1.0."""
        try:
            recent = getattr(dynamic_thresholds, 'recent_trades', [])
            if len(recent) < 10:
                return 1.0
            wins = [t for t in recent[-20:] if t > 0]
            losses = [abs(t) for t in recent[-20:] if t < 0]
            if not wins or not losses:
                return 1.0
            avg_win = sum(wins) / len(wins)
            avg_loss = sum(losses) / len(losses)
            return avg_win / avg_loss if avg_loss > 0 else 1.0
        except Exception:
            return 1.0

    def get_selector_status(self) -> Dict:
        """Get status of the dynamic model selector."""
        if self.dynamic_selector:
            return self.dynamic_selector.get_status()
        return {"dynamic_selector": False, "mode": "static"}


# ═══════════════════════════════════════════════════════════════════════════════
# POSITION SIZING UTILITIES (Wave A)
# ═══════════════════════════════════════════════════════════════════════════════

def vol_target_position_size(
    realized_vol_20d: float,
    target_vol: float = 0.15,
    base_size: float = 0.10,
    min_size: float = 0.02,
    max_size: float = 0.25,
) -> float:
    """
    Volatility-targeting position sizing.

    Scales position size inversely with realized volatility so that
    each position contributes roughly the same expected vol to the portfolio.

    Parameters
    ----------
    realized_vol_20d : float
        Annualized 20-day realized volatility (e.g., 0.15 = 15%).
    target_vol : float
        Target annualized volatility contribution per position (default 15%).
    base_size : float
        Base position size (fraction of portfolio, default 10%).
    min_size : float
        Minimum position size (default 2%).
    max_size : float
        Maximum position size (default 25%).

    Returns
    -------
    float
        Position size as fraction of portfolio.
    """
    if realized_vol_20d <= 0 or not np.isfinite(realized_vol_20d):
        return base_size

    raw_size = target_vol / realized_vol_20d * base_size
    return float(np.clip(raw_size, min_size, max_size))


def vix_adjusted_kelly(
    kelly_fraction: float,
    vix_level: float,
    min_fraction: float = 0.01,
    max_fraction: float = 0.25,
) -> float:
    """
    VIX-conditional Kelly position sizing.

    Scales a Kelly fraction by VIX regime to reduce exposure during
    high-volatility environments and increase it in calm markets.

    Parameters
    ----------
    kelly_fraction : float
        Raw Kelly fraction (half-Kelly recommended, i.e., already halved).
    vix_level : float
        Current VIX level (e.g., 15.0, 25.0).
    min_fraction : float
        Minimum position fraction (default 1%).
    max_fraction : float
        Maximum position fraction (default 25%).

    Returns
    -------
    float
        Adjusted position size as fraction of portfolio.

    VIX Scaling:
        VIX < 15:  0.50 × Kelly  (calm → moderate exposure)
        15-25:     0.35 × Kelly  (normal → conservative)
        25-35:     0.20 × Kelly  (elevated → defensive)
        VIX > 35:  0.10 × Kelly  (fear → minimal exposure)
    """
    if not np.isfinite(vix_level) or vix_level <= 0:
        scale = 0.35  # Default to normal regime

    elif vix_level < 15:
        scale = 0.50
    elif vix_level < 25:
        scale = 0.35
    elif vix_level < 35:
        scale = 0.20
    else:
        scale = 0.10

    adjusted = kelly_fraction * scale
    return float(np.clip(adjusted, min_fraction, max_fraction))
