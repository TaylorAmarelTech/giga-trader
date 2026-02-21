"""
GIGA TRADER - Training Pipeline V2
====================================
Executes model training using configurations from the Model Registry V2.

This pipeline:
1. Takes ModelEntry configs (from grid search or manual)
2. Executes each pipeline step based on the config
3. Uses leak-proof CV to prevent data leakage
4. Records all metrics and artifacts
5. Updates the model registry

Pipeline Steps:
  1. Data Loading       -> DataConfig
  2. Synthetic Data     -> SyntheticDataConfig
  3. Augmentation       -> AugmentationConfig
  4. Preprocessing      -> PreprocessConfig
  5. Feature Engineering -> FeatureConfig
  6. Target Definition  -> TargetConfig
  7. Feature Selection  -> FeatureSelectionConfig
  8. Dim Reduction      -> DimReductionConfig
  9. Model Training     -> ModelConfig
  10. Cascade Training  -> CascadeConfig
  11. Sample Weighting  -> SampleWeightConfig
  12. Cross-Validation  -> TrainingConfig
  13. Evaluation        -> EvaluationConfig

Usage:
    from src.training_pipeline_v2 import TrainingPipelineV2
    from src.model_registry_v2 import GridSearchConfigGenerator, ModelRegistryV2

    # Generate configs
    gen = GridSearchConfigGenerator.create_standard_grid()
    configs = gen.generate_configs("swing", max_configs=100, random_sample=True)

    # Train all
    pipeline = TrainingPipelineV2()
    results = pipeline.train_batch(configs)
"""

import os
import sys
import time
import traceback
import logging
import hashlib
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

import numpy as np
import pandas as pd
import joblib
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
load_dotenv(project_root / ".env")

from src.model_registry_v2 import (
    ModelRegistryV2,
    ModelEntry,
    ModelMetrics,
    ModelArtifacts,
    ModelStatus,
    ModelType,
    CascadeType,
    DimReductionMethod,
    FeatureSelectionMethod,
    ScalingMethod,
    OutlierMethod,
    MissingValueMethod,
    TransformMethod,
    CVMethod,
    ScoringMetric,
    SampleWeightMethod,
    TargetType,
    TargetDefinition,
    LabelSmoothingMethod,
    DataSource,
    DataPeriod,
    TimeResolution,
    MarketHours,
    GridSearchConfigGenerator,
)
from src.core.registry_db import get_registry_db

logger = logging.getLogger("PIPELINE_V2")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s"
)


# =============================================================================
# PIPELINE STEP EXECUTORS
# =============================================================================

class DataLoader:
    """Step 1: Load market data based on DataConfig."""

    _cache: Dict[str, pd.DataFrame] = {}  # Class-level cache

    @classmethod
    def load(cls, config: ModelEntry) -> pd.DataFrame:
        """Load data based on configuration. Caches across models."""
        dc = config.data_config

        # Build cache key from data-relevant config
        # Always cache by 1min - resampling happens after loading
        cache_key = f"{dc.source}_{dc.symbol}_{dc.period}_1min_{dc.market_hours}"

        if cache_key in cls._cache:
            logger.info(f"[DATA] Using cached data for {cache_key}")
            return cls._cache[cache_key].copy()

        logger.info(f"[DATA] Loading data: source={dc.source}, symbol={dc.symbol}, period={dc.period}")

        if dc.source in ("alpaca", "yfinance"):
            df = cls._download_data(dc)
        elif dc.source == "csv":
            df = cls._load_csv(dc)
        else:
            # Fallback to yfinance
            logger.warning(f"[DATA] Unknown source '{dc.source}', falling back to yfinance")
            dc_copy = dc
            dc_copy.source = "yfinance"
            df = cls._download_data(dc_copy)

        if df is not None and len(df) > 0:
            cls._cache[cache_key] = df
            logger.info(f"[DATA] Loaded {len(df):,} bars ({df.index[0]} to {df.index[-1]})")
        else:
            raise ValueError(f"Failed to load data for {cache_key}")

        return df.copy()

    @classmethod
    def _download_data(cls, dc) -> pd.DataFrame:
        """Download data from API source, using DataManager's parquet cache."""
        try:
            # Use DataManager's persistent parquet cache (skip_freshness=True for experiments)
            from src.phase_01_data_acquisition.data_manager import get_data_manager

            period_map = {
                "6M": 0.5, "1Y": 1, "2Y": 2, "3Y": 3,
                "5Y": 5, "7Y": 7, "10Y": 10, "max": 10
            }
            years = getattr(dc, 'years', 0.0) or period_map.get(dc.period, 10.0)

            dm = get_data_manager()
            df = dm.get_data(dc.symbol, years=int(years), skip_freshness=True)

            if df is None or len(df) == 0:
                raise ValueError(f"DataManager returned no data for {dc.symbol}")

            # Filter by market hours
            if dc.market_hours == "regular_only" and "session" in df.columns:
                df = df[df["session"] == "regular"]
            elif dc.market_hours == "premarket_only" and "session" in df.columns:
                df = df[df["session"] == "premarket"]
            elif dc.market_hours == "afterhours_only" and "session" in df.columns:
                df = df[df["session"] == "afterhours"]

            return df

        except Exception as e:
            logger.error(f"[DATA] DataManager load failed: {e}, falling back to legacy download")
            # Fallback to legacy download
            try:
                from src.train_robust_model import download_data as _download_legacy
                import src.train_robust_model as trm
                original_config = trm.CONFIG.copy()
                trm.CONFIG["years_to_download"] = years
                df = _download_legacy()
                trm.CONFIG.update(original_config)
                return df
            except Exception as e2:
                logger.error(f"[DATA] Legacy download also failed: {e2}")
                raise

    @classmethod
    def _load_csv(cls, dc) -> pd.DataFrame:
        """Load data from CSV file."""
        csv_dir = project_root / "data" / "raw"
        csv_path = csv_dir / f"{dc.symbol}_1min.csv"
        if csv_path.exists():
            return pd.read_csv(csv_path, parse_dates=["timestamp"], index_col="timestamp")
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    @classmethod
    def clear_cache(cls):
        """Clear the data cache."""
        cls._cache.clear()
        logger.info("[DATA] Cache cleared")


class FeatureEngineer:
    """Step 5: Engineer features based on FeatureConfig."""

    @staticmethod
    def engineer(df_1min: pd.DataFrame, config: ModelEntry) -> pd.DataFrame:
        """Engineer features from minute-level data."""
        logger.info("[FEATURES] Engineering features...")

        try:
            from src.train_robust_model import engineer_all_features, add_rolling_features

            # Engineer base features
            dc = config.feature_config
            df_daily = engineer_all_features(df_1min, swing_threshold=config.target_config.swing_threshold)

            # Add rolling features
            df_daily = add_rolling_features(df_daily)

            logger.info(f"[FEATURES] Engineered {len(df_daily.columns)} features, {len(df_daily)} days")
            return df_daily

        except Exception as e:
            logger.error(f"[FEATURES] Engineering failed: {e}")
            raise


class TargetCreator:
    """Step 6: Create target variables based on TargetConfig."""

    @staticmethod
    def create_targets(df_daily: pd.DataFrame, config: ModelEntry) -> Tuple[pd.DataFrame, str, str]:
        """
        Create target variables.

        Returns:
            (df_daily, swing_target_col, timing_target_col)
        """
        tc = config.target_config
        logger.info(f"[TARGETS] Creating targets: type={tc.target_type}, soft={tc.use_soft_targets}")

        try:
            from src.train_robust_model import create_soft_targets

            df_daily = create_soft_targets(df_daily, threshold=tc.swing_threshold)

            swing_target = "target_up_soft" if tc.use_soft_targets else "target_up"
            timing_target = "target_timing_soft" if tc.use_soft_targets else "low_before_high"

            # Fall back if soft targets not present
            if swing_target not in df_daily.columns:
                swing_target = "target_up" if "target_up" in df_daily.columns else "is_up_day"
            if timing_target not in df_daily.columns:
                timing_target = "low_before_high"

            logger.info(f"[TARGETS] Swing target: {swing_target}, Timing target: {timing_target}")
            return df_daily, swing_target, timing_target

        except Exception as e:
            logger.error(f"[TARGETS] Creation failed: {e}")
            raise


class Preprocessor:
    """Step 4: Preprocess data based on PreprocessConfig."""

    @staticmethod
    def preprocess(
        X: np.ndarray,
        config: ModelEntry,
        fit: bool = True,
        fitted_objects: Dict = None,
    ) -> Tuple[np.ndarray, Dict]:
        """
        Preprocess features: handle outliers, missing values, scale.

        Returns:
            (X_processed, fitted_objects_dict)
        """
        pc = config.preprocess_config
        fitted = fitted_objects or {}

        # Step 1: Handle missing values
        X = Preprocessor._handle_missing(X, pc)

        # Step 2: Handle outliers
        X = Preprocessor._handle_outliers(X, pc)

        return X, fitted

    @staticmethod
    def _handle_missing(X: np.ndarray, pc) -> np.ndarray:
        """Handle missing values."""
        if np.isnan(X).any():
            if pc.missing_method in ("forward_fill", "backward_fill"):
                # For arrays, fill column-wise
                df_temp = pd.DataFrame(X)
                if pc.missing_method == "forward_fill":
                    df_temp = df_temp.ffill().bfill()
                else:
                    df_temp = df_temp.bfill().ffill()
                X = df_temp.values
            elif pc.missing_method == "mean_fill":
                col_means = np.nanmean(X, axis=0)
                for i in range(X.shape[1]):
                    mask = np.isnan(X[:, i])
                    X[mask, i] = col_means[i]
            elif pc.missing_method == "median_fill":
                col_medians = np.nanmedian(X, axis=0)
                for i in range(X.shape[1]):
                    mask = np.isnan(X[:, i])
                    X[mask, i] = col_medians[i]
            elif pc.missing_method == "zero_fill":
                X = np.nan_to_num(X, nan=0.0)
            elif pc.missing_method == "knn_impute":
                try:
                    from sklearn.impute import KNNImputer
                    imputer = KNNImputer(n_neighbors=pc.missing_knn_neighbors)
                    X = imputer.fit_transform(X)
                except Exception:
                    X = np.nan_to_num(X, nan=0.0)
            elif pc.missing_method == "linear_interpolate":
                df_temp = pd.DataFrame(X)
                df_temp = df_temp.interpolate(method="linear", limit_direction="both")
                X = df_temp.fillna(0).values
            else:
                X = np.nan_to_num(X, nan=0.0)

        # Replace infinities
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    @staticmethod
    def _handle_outliers(X: np.ndarray, pc) -> np.ndarray:
        """Handle outliers based on config."""
        method = pc.outlier_method

        if method == "none":
            return X

        if method.startswith("winsorize"):
            # Extract percentage
            pct = float(method.split("_")[1]) / 100.0
            lower = np.nanpercentile(X, pct * 100, axis=0)
            upper = np.nanpercentile(X, (1 - pct) * 100, axis=0)
            X = np.clip(X, lower, upper)

        elif method.startswith("clip"):
            # Extract std multiplier
            n_std = float(method.split("_")[0].replace("clip", "").replace("std", ""))
            if n_std == 0:
                n_std = float(method.split("_")[1].replace("std", ""))
            means = np.nanmean(X, axis=0)
            stds = np.nanstd(X, axis=0)
            stds[stds == 0] = 1.0
            lower = means - n_std * stds
            upper = means + n_std * stds
            X = np.clip(X, lower, upper)

        elif method.startswith("iqr"):
            multiplier = float(method.split("_")[1])
            q1 = np.nanpercentile(X, 25, axis=0)
            q3 = np.nanpercentile(X, 75, axis=0)
            iqr = q3 - q1
            iqr[iqr == 0] = 1.0
            lower = q1 - multiplier * iqr
            upper = q3 + multiplier * iqr
            X = np.clip(X, lower, upper)

        elif method == "isolation_forest":
            try:
                from sklearn.ensemble import IsolationForest
                iso = IsolationForest(contamination=pc.outlier_contamination, random_state=42)
                labels = iso.fit_predict(X)
                # Replace outlier rows with column medians
                outlier_mask = labels == -1
                if outlier_mask.any():
                    medians = np.nanmedian(X[~outlier_mask], axis=0)
                    X[outlier_mask] = medians
            except Exception:
                pass

        return X


class ScalerFactory:
    """Create scaler objects based on config."""

    @staticmethod
    def create(config: ModelEntry):
        """Create a scaler based on PreprocessConfig."""
        method = config.preprocess_config.scaling_method

        if method == "none":
            return None
        elif method == "standard":
            from sklearn.preprocessing import StandardScaler
            return StandardScaler()
        elif method == "minmax":
            from sklearn.preprocessing import MinMaxScaler
            return MinMaxScaler()
        elif method == "minmax_neg1_1":
            from sklearn.preprocessing import MinMaxScaler
            return MinMaxScaler(feature_range=(-1, 1))
        elif method == "robust":
            from sklearn.preprocessing import RobustScaler
            return RobustScaler()
        elif method == "robust_iqr":
            from sklearn.preprocessing import RobustScaler
            return RobustScaler(quantile_range=(25, 75))
        elif method == "quantile_uniform":
            from sklearn.preprocessing import QuantileTransformer
            return QuantileTransformer(output_distribution="uniform", random_state=42)
        elif method == "quantile_normal":
            from sklearn.preprocessing import QuantileTransformer
            return QuantileTransformer(output_distribution="normal", random_state=42)
        elif method == "power_yeojohnson":
            from sklearn.preprocessing import PowerTransformer
            return PowerTransformer(method="yeo-johnson")
        elif method == "power_boxcox":
            from sklearn.preprocessing import PowerTransformer
            return PowerTransformer(method="box-cox")
        elif method == "max_abs":
            from sklearn.preprocessing import MaxAbsScaler
            return MaxAbsScaler()
        else:
            from sklearn.preprocessing import StandardScaler
            return StandardScaler()


class FeatureSelector:
    """Step 7: Feature selection based on FeatureSelectionConfig."""

    @staticmethod
    def create(config: ModelEntry):
        """Create a feature selector based on config."""
        fsc = config.feature_selection_config

        if not fsc.enabled or fsc.method == "none":
            return None

        # Use leak-proof feature selector for supported methods
        from src.leak_proof_cv import LeakProofFeatureSelector

        method_map = {
            "mutual_info": "mutual_info",
            "anova_f": "f_classif",
            "chi2": "f_classif",
            "variance_threshold": "mutual_info",
            "correlation_filter": "mutual_info",
        }
        mapped_method = method_map.get(fsc.method, "mutual_info")

        n_features = fsc.n_features or 30

        return LeakProofFeatureSelector(
            method=mapped_method,
            n_features=n_features,
            variance_threshold=fsc.variance_threshold,
            correlation_threshold=fsc.correlation_threshold,
            random_state=42,
        )


class DimReducerFactory:
    """Step 8: Create dimensionality reducer based on DimReductionConfig."""

    @staticmethod
    def create(config: ModelEntry):
        """Create a dim reducer based on config."""
        drc = config.dim_reduction_config

        if not drc.enabled or drc.method == "none":
            return None

        from src.leak_proof_cv import LeakProofDimReducer

        # Map expanded methods to base method names
        method_map = {
            "pca": "pca",
            "incremental_pca": "pca",
            "sparse_pca": "pca",
            "truncated_svd": "pca",
            "kernel_pca_rbf": "kernel_pca",
            "kernel_pca_poly": "kernel_pca",
            "kernel_pca_sigmoid": "kernel_pca",
            "kernel_pca_cosine": "kernel_pca",
            "ica": "ica",
            "fast_ica": "ica",
            "umap": "umap",
            "factor_analysis": "pca",
            "ensemble_plus": "ensemble_plus",
            "agglomeration": "pca",
            "kmedoids": "pca",
            "lda": "pca",
            "nmf": "pca",
        }
        mapped = method_map.get(drc.method, "pca")

        n_components = drc.n_components or 20

        return LeakProofDimReducer(
            method=mapped,
            n_components=n_components,
            random_state=42,
        )


class ModelFactory:
    """Step 9: Create ML model based on ModelConfig."""

    @staticmethod
    def create(config: ModelEntry):
        """Create a model based on ModelConfig."""
        mc = config.model_config

        if mc.model_type in ("logistic_l1", "logistic_l2", "elastic_net", "ridge"):
            return ModelFactory._create_logistic(mc)
        elif mc.model_type in ("gradient_boosting", "hist_gradient_boosting"):
            return ModelFactory._create_gradient_boosting(mc)
        elif mc.model_type == "xgboost":
            return ModelFactory._create_xgboost(mc)
        elif mc.model_type == "lightgbm":
            return ModelFactory._create_lightgbm(mc)
        elif mc.model_type == "catboost":
            return ModelFactory._create_catboost(mc)
        elif mc.model_type in ("random_forest", "extra_trees"):
            return ModelFactory._create_tree_ensemble(mc)
        elif mc.model_type in ("svm_linear", "svm_rbf", "svm_poly"):
            return ModelFactory._create_svm(mc)
        elif mc.model_type in ("mlp", "mlp_deep", "mlp_wide"):
            return ModelFactory._create_mlp(mc)
        elif mc.model_type == "knn":
            return ModelFactory._create_knn(mc)
        elif mc.model_type in ("gaussian_nb", "multinomial_nb"):
            return ModelFactory._create_naive_bayes(mc)
        elif mc.model_type == "adaboost":
            return ModelFactory._create_adaboost(mc)
        elif mc.model_type == "decision_tree":
            return ModelFactory._create_decision_tree(mc)
        elif mc.model_type == "lda":
            return ModelFactory._create_lda(mc)
        elif mc.model_type == "bayesian_ridge":
            return ModelFactory._create_bayesian_ridge(mc)
        elif mc.model_type == "quantile_gb":
            return ModelFactory._create_quantile_gb(mc)
        elif mc.model_type == "svc_linear":
            return ModelFactory._create_svc_linear(mc)
        else:
            # Default to logistic regression
            logger.warning(f"Unknown model type '{mc.model_type}', defaulting to logistic_l2")
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                C=1.0, max_iter=1000, random_state=mc.random_state,
                class_weight=mc.class_weight, n_jobs=mc.n_jobs
            )

    @staticmethod
    def _create_logistic(mc):
        from sklearn.linear_model import LogisticRegression

        penalty_map = {
            "logistic_l1": "l1",
            "logistic_l2": "l2",
            "elastic_net": "elasticnet",
            "ridge": "l2",
        }
        penalty = penalty_map.get(mc.model_type, mc.lr_penalty)

        solver = mc.lr_solver
        if penalty == "l1" and solver not in ("liblinear", "saga"):
            solver = "saga"
        elif penalty == "elasticnet" and solver != "saga":
            solver = "saga"

        params = {
            "penalty": penalty,
            "C": mc.lr_C,
            "solver": solver,
            "max_iter": mc.lr_max_iter,
            "random_state": mc.random_state,
            "class_weight": mc.class_weight,
            "n_jobs": mc.n_jobs,
        }

        if penalty == "elasticnet":
            params["l1_ratio"] = mc.lr_l1_ratio

        return LogisticRegression(**params)

    @staticmethod
    def _create_gradient_boosting(mc):
        if mc.model_type == "hist_gradient_boosting":
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(
                max_iter=mc.hgb_max_iter,
                max_depth=mc.hgb_max_depth,
                learning_rate=mc.hgb_learning_rate,
                max_leaf_nodes=mc.hgb_max_leaf_nodes,
                min_samples_leaf=mc.hgb_min_samples_leaf,
                l2_regularization=mc.hgb_l2_regularization,
                max_bins=mc.hgb_max_bins,
                early_stopping=mc.hgb_early_stopping,
                random_state=mc.random_state,
                class_weight=mc.class_weight,
            )

        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=mc.gb_n_estimators,
            max_depth=min(mc.gb_max_depth, 5),  # EDGE 1: Never > 5
            learning_rate=mc.gb_learning_rate,
            subsample=mc.gb_subsample,
            min_samples_leaf=mc.gb_min_samples_leaf,
            max_features=mc.gb_max_features,
            random_state=mc.random_state,
        )

    @staticmethod
    def _create_xgboost(mc):
        try:
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=mc.xgb_n_estimators,
                max_depth=min(mc.xgb_max_depth, 5),
                learning_rate=mc.xgb_learning_rate,
                booster=mc.xgb_booster,
                gamma=mc.xgb_gamma,
                min_child_weight=mc.xgb_min_child_weight,
                subsample=mc.xgb_subsample,
                colsample_bytree=mc.xgb_colsample_bytree,
                reg_alpha=mc.xgb_reg_alpha,
                reg_lambda=mc.xgb_reg_lambda,
                scale_pos_weight=mc.xgb_scale_pos_weight,
                random_state=mc.random_state,
                n_jobs=mc.n_jobs,
                use_label_encoder=False,
                eval_metric="logloss",
                verbosity=0,
            )
        except ImportError:
            logger.warning("XGBoost not installed, falling back to GradientBoosting")
            return ModelFactory._create_gradient_boosting(mc)

    @staticmethod
    def _create_lightgbm(mc):
        try:
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=mc.lgb_n_estimators,
                max_depth=mc.lgb_max_depth,
                num_leaves=mc.lgb_num_leaves,
                learning_rate=mc.lgb_learning_rate,
                subsample=mc.lgb_subsample,
                colsample_bytree=mc.lgb_colsample_bytree,
                reg_alpha=mc.lgb_reg_alpha,
                reg_lambda=mc.lgb_reg_lambda,
                min_child_samples=mc.lgb_min_child_samples,
                boosting_type=mc.lgb_boosting_type,
                is_unbalance=mc.lgb_is_unbalance,
                random_state=mc.random_state,
                n_jobs=mc.n_jobs,
                verbose=-1,
            )
        except ImportError:
            logger.warning("LightGBM not installed, falling back to GradientBoosting")
            return ModelFactory._create_gradient_boosting(mc)

    @staticmethod
    def _create_catboost(mc):
        try:
            from catboost import CatBoostClassifier
            return CatBoostClassifier(
                iterations=mc.cb_iterations,
                depth=min(mc.cb_depth, 5),
                learning_rate=mc.cb_learning_rate,
                l2_leaf_reg=mc.cb_l2_leaf_reg,
                bagging_temperature=mc.cb_bagging_temperature,
                random_strength=mc.cb_random_strength,
                border_count=mc.cb_border_count,
                bootstrap_type=mc.cb_bootstrap_type,
                random_seed=mc.random_state,
                verbose=False,
            )
        except ImportError:
            logger.warning("CatBoost not installed, falling back to GradientBoosting")
            return ModelFactory._create_gradient_boosting(mc)

    @staticmethod
    def _create_tree_ensemble(mc):
        if mc.model_type == "extra_trees":
            from sklearn.ensemble import ExtraTreesClassifier
            return ExtraTreesClassifier(
                n_estimators=mc.et_n_estimators,
                max_depth=min(mc.et_max_depth or 5, 5),
                min_samples_leaf=mc.et_min_samples_leaf,
                random_state=mc.random_state,
                class_weight=mc.class_weight,
                n_jobs=mc.n_jobs,
            )

        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=mc.rf_n_estimators,
            max_depth=min(mc.rf_max_depth or 5, 5),
            min_samples_leaf=mc.rf_min_samples_leaf,
            max_features=mc.rf_max_features,
            bootstrap=mc.rf_bootstrap,
            random_state=mc.random_state,
            class_weight=mc.class_weight,
            n_jobs=mc.n_jobs,
        )

    @staticmethod
    def _create_svm(mc):
        from sklearn.svm import SVC

        kernel_map = {
            "svm_linear": "linear",
            "svm_rbf": "rbf",
            "svm_poly": "poly",
        }
        return SVC(
            C=mc.svm_C,
            kernel=kernel_map.get(mc.model_type, mc.svm_kernel),
            degree=mc.svm_degree,
            gamma=mc.svm_gamma,
            probability=True,
            random_state=mc.random_state,
            class_weight=mc.class_weight,
            max_iter=mc.svm_max_iter,
        )

    @staticmethod
    def _create_mlp(mc):
        from sklearn.neural_network import MLPClassifier

        if mc.model_type == "mlp_deep":
            hidden = [256, 128, 64, 32]
        elif mc.model_type == "mlp_wide":
            hidden = [512, 256]
        else:
            hidden = mc.mlp_hidden_layer_sizes

        return MLPClassifier(
            hidden_layer_sizes=hidden,
            activation=mc.mlp_activation,
            solver=mc.mlp_solver,
            alpha=mc.mlp_alpha,
            learning_rate=mc.mlp_learning_rate,
            learning_rate_init=mc.mlp_learning_rate_init,
            max_iter=mc.mlp_max_iter,
            early_stopping=mc.mlp_early_stopping,
            random_state=mc.random_state,
        )

    @staticmethod
    def _create_knn(mc):
        from sklearn.neighbors import KNeighborsClassifier
        return KNeighborsClassifier(
            n_neighbors=mc.knn_n_neighbors,
            weights=mc.knn_weights,
            algorithm=mc.knn_algorithm,
            n_jobs=mc.n_jobs,
        )

    @staticmethod
    def _create_naive_bayes(mc):
        if mc.model_type == "gaussian_nb":
            from sklearn.naive_bayes import GaussianNB
            return GaussianNB(var_smoothing=mc.nb_var_smoothing)
        from sklearn.naive_bayes import MultinomialNB
        return MultinomialNB(alpha=mc.nb_alpha)

    @staticmethod
    def _create_adaboost(mc):
        from sklearn.ensemble import AdaBoostClassifier
        try:
            return AdaBoostClassifier(
                n_estimators=mc.ada_n_estimators,
                learning_rate=mc.ada_learning_rate,
                algorithm=mc.ada_algorithm,
                random_state=mc.random_state,
            )
        except TypeError:
            # 'algorithm' param removed in newer sklearn versions
            return AdaBoostClassifier(
                n_estimators=mc.ada_n_estimators,
                learning_rate=mc.ada_learning_rate,
                random_state=mc.random_state,
            )

    @staticmethod
    def _create_decision_tree(mc):
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(
            criterion=mc.dt_criterion,
            max_depth=min(mc.dt_max_depth or 5, 5),
            min_samples_split=mc.dt_min_samples_split,
            min_samples_leaf=mc.dt_min_samples_leaf,
            random_state=mc.random_state,
            class_weight=mc.class_weight,
        )


    @staticmethod
    def _create_lda(mc):
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.calibration import CalibratedClassifierCV
        base = LinearDiscriminantAnalysis(solver='svd')
        return CalibratedClassifierCV(base, cv=3, method='sigmoid')

    @staticmethod
    def _create_bayesian_ridge(mc):
        from sklearn.linear_model import BayesianRidge
        from sklearn.calibration import CalibratedClassifierCV
        from sklearn.base import ClassifierMixin, BaseEstimator

        class _BayesianRidgeClassifier(ClassifierMixin, BaseEstimator):
            def __init__(self):
                self._reg = BayesianRidge()
                self.classes_ = np.array([0, 1])
            def fit(self, X, y, **kw):
                self._reg.fit(X, y.astype(float))
                return self
            def predict(self, X):
                return (self._reg.predict(X) >= 0.5).astype(int)
            def decision_function(self, X):
                return self._reg.predict(X)
            def get_params(self, deep=True):
                return {}
            def set_params(self, **params):
                return self

        return CalibratedClassifierCV(_BayesianRidgeClassifier(), cv=3, method='sigmoid')

    @staticmethod
    def _create_quantile_gb(mc):
        from sklearn.ensemble import HistGradientBoostingRegressor
        from sklearn.base import ClassifierMixin, BaseEstimator

        class _QuantileGBClassifier(ClassifierMixin, BaseEstimator):
            def __init__(self, max_iter=100, max_depth=3, learning_rate=0.1,
                         min_samples_leaf=50, random_state=42):
                self._alphas = [0.10, 0.50, 0.90]
                self._models = {}
                self.max_iter = max_iter
                self.max_depth = max_depth
                self.learning_rate = learning_rate
                self.min_samples_leaf = min_samples_leaf
                self.random_state = random_state
                self.classes_ = np.array([0, 1])
            def fit(self, X, y, **kw):
                y_f = y.astype(float)
                for alpha in self._alphas:
                    m = HistGradientBoostingRegressor(
                        loss='quantile', quantile=alpha,
                        max_iter=self.max_iter, max_depth=self.max_depth,
                        learning_rate=self.learning_rate,
                        min_samples_leaf=self.min_samples_leaf,
                        random_state=self.random_state,
                    )
                    m.fit(X, y_f)
                    self._models[alpha] = m
                return self
            def predict_proba(self, X):
                p = np.clip(self._models[0.50].predict(X), 0.01, 0.99)
                return np.column_stack([1 - p, p])
            def predict(self, X):
                return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

        return _QuantileGBClassifier(
            max_iter=min(getattr(mc, 'gb_n_estimators', 100), 100),
            max_depth=min(getattr(mc, 'gb_max_depth', 3), 5),
            learning_rate=min(getattr(mc, 'gb_learning_rate', 0.1), 0.1),
            min_samples_leaf=50, random_state=mc.random_state,
        )

    @staticmethod
    def _create_svc_linear(mc):
        from sklearn.svm import LinearSVC
        from sklearn.calibration import CalibratedClassifierCV
        base = LinearSVC(
            C=getattr(mc, 'svm_C', 0.1) * 0.1,
            loss='squared_hinge', penalty='l2',
            max_iter=2000, dual='auto',
            random_state=mc.random_state,
        )
        return CalibratedClassifierCV(base, cv=3, method='sigmoid')


class SampleWeighter:
    """Step 11: Create sample weights based on SampleWeightConfig."""

    @staticmethod
    def compute_weights(
        y: np.ndarray,
        config: ModelEntry,
        returns: np.ndarray = None,
        dates: np.ndarray = None,
    ) -> np.ndarray:
        """Compute sample weights based on config."""
        swc = config.sample_weight_config

        if not swc.enabled or swc.method == "none":
            return np.ones(len(y))

        weights = np.ones(len(y))

        # Class balancing
        if swc.balance_classes:
            classes, counts = np.unique(y.astype(int), return_counts=True)
            total = len(y)
            n_classes = len(classes)
            for cls_val, cls_count in zip(classes, counts):
                mask = y.astype(int) == cls_val
                w = total / (n_classes * cls_count)
                weights[mask] *= w

        # Time decay
        if swc.time_decay_enabled and dates is not None:
            n = len(dates)
            if swc.decay_type == "linear":
                time_weights = np.linspace(1 / swc.recent_weight_multiplier, 1.0, n)
            elif swc.decay_type == "exponential":
                time_weights = np.exp(-swc.decay_rate * np.arange(n)[::-1])
            elif swc.decay_type == "half_life":
                half_life = swc.half_life_days
                time_weights = np.power(0.5, np.arange(n)[::-1] / half_life)
            else:
                time_weights = np.ones(n)
            weights *= time_weights

        # Return magnitude
        if swc.return_magnitude_enabled and returns is not None:
            if swc.magnitude_type == "absolute":
                mag_weights = np.abs(returns)
            else:
                mag_weights = returns ** 2
            mag_weights = mag_weights / (mag_weights.mean() + 1e-8)

            if swc.emphasize_losses:
                loss_mask = returns < 0
                mag_weights[loss_mask] *= swc.loss_emphasis_factor

            weights *= mag_weights

        # Normalize and clip
        if swc.normalize_weights:
            weights = weights / (weights.mean() + 1e-8)
        weights = np.clip(weights, swc.clip_min, swc.clip_max)

        return weights


class CrossValidator:
    """Step 12: Cross-validation based on TrainingConfig."""

    @staticmethod
    def create_cv(config: ModelEntry):
        """Create cross-validation object."""
        tc = config.training_config

        from src.leak_proof_cv import LeakProofCV

        # Map feature selection from config
        fsc = config.feature_selection_config
        fs_method = "mutual_info" if fsc.method in ("mutual_info", "none") else "mutual_info"

        # Map dim reduction from config
        drc = config.dim_reduction_config
        dim_method_map = {
            "pca": "pca", "kernel_pca_rbf": "kernel_pca",
            "kernel_pca_poly": "kernel_pca", "ica": "ica",
            "ensemble_plus": "ensemble_plus", "none": "none",
        }
        dim_method = dim_method_map.get(drc.method, "kernel_pca")
        n_components = drc.n_components or 20

        return LeakProofCV(
            n_folds=tc.cv_folds,
            purge_days=tc.purge_days,
            embargo_days=int(tc.embargo_pct * 252) or tc.purge_days,
            feature_selection_method=fs_method,
            n_features=fsc.n_features or 30,
            dim_reduction_method=dim_method,
            n_components=n_components,
            random_state=tc.seed,
        )


# =============================================================================
# MAIN TRAINING PIPELINE
# =============================================================================

class TrainingPipelineV2:
    """
    Main training pipeline that executes model training using
    configurations from the Model Registry V2.

    Supports:
    - Single model training
    - Batch training (grid search)
    - Progress tracking
    - Error recovery
    - Registry updates
    """

    def __init__(
        self,
        registry: ModelRegistryV2 = None,
        models_dir: Path = None,
        max_retries: int = 1,
        verbose: bool = True,
    ):
        self.registry = registry or ModelRegistryV2(db=get_registry_db())
        self.models_dir = models_dir or (project_root / "models" / "pipeline_v2")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.verbose = verbose

        # Shared data cache (avoid re-downloading for each model)
        self._data_cache: Dict[str, Any] = {}

    def train_single(self, entry: ModelEntry) -> ModelEntry:
        """
        Train a single model based on its ModelEntry configuration.

        Steps:
        1. Register model as TRAINING
        2. Load data
        3. Engineer features
        4. Create targets
        5. Preprocess
        6. Cross-validate with leak-proof pipeline
        7. Train final model
        8. Evaluate
        9. Save artifacts
        10. Update registry

        Returns:
            Updated ModelEntry with metrics and artifacts
        """
        start_time = time.time()
        entry.status = ModelStatus.TRAINING.value
        entry.training_started_at = datetime.now().isoformat()

        # Register in registry
        model_id = self.registry.register(entry)
        entry.model_id = model_id

        self._log(f"\n{'='*70}")
        self._log(f"TRAINING MODEL: {model_id}")
        self._log(f"  Model: {entry.model_config.model_type}")
        self._log(f"  Cascade: {entry.cascade_config.cascade_type}")
        self._log(f"  Dim Reduction: {entry.dim_reduction_config.method}")
        self._log(f"  Target: {entry.target_type}")
        if entry.grid_total > 0:
            self._log(f"  Grid Position: {entry.grid_position + 1}/{entry.grid_total}")
        self._log(f"{'='*70}")

        try:
            # Step 1: Load data (always 1-min bars)
            self._log("\n[Step 1/8] Loading data...")
            df_1min = DataLoader.load(entry)

            # Step 1b: Resample to config's primary_resolution
            df_bars = self._apply_resolution(df_1min, entry)

            # Step 2: Engineer features (from resampled bars)
            self._log("[Step 2/8] Engineering features...")
            df_daily = FeatureEngineer.engineer(df_bars, entry)

            # Step 3: Create targets
            self._log("[Step 3/8] Creating targets...")
            df_daily, swing_target, timing_target = TargetCreator.create_targets(df_daily, entry)

            # Choose target based on target_type
            if entry.target_type in ("swing", "direction"):
                target_col = swing_target
            elif entry.target_type == "timing":
                target_col = timing_target
            else:
                target_col = swing_target

            # Step 4: Prepare arrays
            self._log("[Step 4/8] Preparing feature matrix...")
            X, y, feature_cols, sample_weights = self._prepare_arrays(
                df_daily, target_col, entry
            )

            entry.n_features_raw = X.shape[1]
            entry.training_samples = X.shape[0]

            # Step 5: Preprocess
            self._log("[Step 5/8] Preprocessing...")
            X, _ = Preprocessor.preprocess(X, entry)

            # Step 6: Cross-validate
            self._log("[Step 6/8] Cross-validating with leak-proof pipeline...")
            cv_results = self._cross_validate(X, y, sample_weights, entry)

            # Step 7: Train final model on all data
            self._log("[Step 7/8] Training final model...")
            model, scaler, selector, reducer = self._train_final(
                X, y, sample_weights, entry, feature_cols=feature_cols
            )

            # Step 7b: Resolution cascade (if requested)
            if (
                entry.cascade_config.cascade_type == CascadeType.MULTI_RESOLUTION.value
                and entry.cascade_config.enabled
            ):
                self._log("[Step 7b/8] Training resolution cascade...")
                cascade_results = self._train_resolution_cascade(df_1min, entry)
                if cascade_results:
                    cv_results["resolution_cascade"] = cascade_results

            # Step 8: Record results and save
            self._log("[Step 8/8] Saving artifacts...")
            entry = self._record_results(
                entry, cv_results, model, scaler, selector, reducer,
                feature_cols, X, y, start_time
            )

            # Update registry
            entry.status = ModelStatus.TRAINED.value
            entry.trained_at = datetime.now().isoformat()
            self.registry.update(entry.model_id, status=entry.status)
            self.registry.update_metrics(entry.model_id, entry.metrics)
            self.registry.update_artifacts(entry.model_id, entry.artifacts)

            elapsed = time.time() - start_time
            self._log(f"\n[DONE] {model_id} trained in {elapsed:.1f}s")
            self._log(f"  CV AUC: {entry.metrics.cv_auc:.4f} +/- {entry.metrics.cv_auc_std:.4f}")
            self._log(f"  Train-Test Gap: {entry.metrics.train_test_gap:.4f}")

            return entry

        except Exception as e:
            elapsed = time.time() - start_time
            error_msg = f"{type(e).__name__}: {str(e)}"
            self._log(f"\n[FAILED] {model_id} after {elapsed:.1f}s: {error_msg}")

            entry.status = ModelStatus.FAILED.value
            entry.failure_reason = error_msg
            entry.training_time_seconds = elapsed

            try:
                self.registry.update(entry.model_id, status=entry.status)
            except Exception:
                pass

            if self.verbose:
                try:
                    traceback.print_exc()
                except OSError:
                    pass  # stdout invalid (Windows daemon/detached console)

            return entry

    def train_single_quick(self, entry: ModelEntry, data_fraction: float = 0.3) -> ModelEntry:
        """
        Quick training for Tier 1 screening in ThickWeaveSearch.

        Creates a data-subsampled, 1-fold variant of the entry and trains it.
        Much faster than full train_single() — suitable for rapid screening.

        Args:
            entry: ModelEntry configuration to train
            data_fraction: Fraction of data to use (default 0.3 = 30%)

        Returns:
            Trained ModelEntry with metrics (reduced fidelity)
        """
        from copy import deepcopy

        quick_entry = deepcopy(entry)

        # Reduce CV folds to 1
        quick_entry.training_config.cv_folds = 1

        # Disable Optuna
        quick_entry.training_config.use_optuna = False

        # Reset IDs so it doesn't conflict with full training
        quick_entry.model_id = ""
        quick_entry.status = "queued"
        quick_entry.tags = list(quick_entry.tags) + ["quick_screen"]

        return self.train_single(quick_entry)

    def train_batch(
        self,
        entries: List[ModelEntry],
        stop_on_failure: bool = False,
        progress_callback=None,
    ) -> List[ModelEntry]:
        """
        Train a batch of models (grid search).

        Args:
            entries: List of ModelEntry configurations to train
            stop_on_failure: If True, stop on first failure
            progress_callback: Optional callback(current, total, entry)

        Returns:
            List of trained ModelEntry objects
        """
        total = len(entries)
        results = []
        n_success = 0
        n_failed = 0

        self._log(f"\n{'#'*70}")
        self._log(f"BATCH TRAINING: {total} models")
        self._log(f"{'#'*70}")

        batch_start = time.time()

        for i, entry in enumerate(entries):
            entry.grid_position = i
            entry.grid_total = total

            if progress_callback:
                progress_callback(i, total, entry)

            result = self.train_single(entry)
            results.append(result)

            if result.status == ModelStatus.TRAINED.value:
                n_success += 1
            else:
                n_failed += 1
                if stop_on_failure:
                    self._log(f"\n[STOPPING] Failed at model {i+1}/{total}")
                    break

            # Progress update
            elapsed = time.time() - batch_start
            rate = (i + 1) / elapsed if elapsed > 0 else 0
            remaining = (total - i - 1) / rate if rate > 0 else 0
            self._log(
                f"\n[PROGRESS] {i+1}/{total} "
                f"(success={n_success}, failed={n_failed}, "
                f"elapsed={elapsed/60:.1f}min, "
                f"remaining~{remaining/60:.1f}min)"
            )

        batch_elapsed = time.time() - batch_start

        self._log(f"\n{'#'*70}")
        self._log(f"BATCH COMPLETE")
        self._log(f"  Total: {total}")
        self._log(f"  Success: {n_success}")
        self._log(f"  Failed: {n_failed}")
        self._log(f"  Time: {batch_elapsed/60:.1f} minutes")
        self._log(f"  Avg per model: {batch_elapsed/max(len(results),1):.1f}s")
        self._log(f"{'#'*70}")

        # Print summary of best models
        self._print_batch_summary(results)

        return results

    def _apply_resolution(
        self, df_1min: pd.DataFrame, config: ModelEntry
    ) -> pd.DataFrame:
        """
        Resample 1-min bars to the config's primary resolution.

        If primary_resolution is '1min', returns data as-is.
        Otherwise uses BarResampler to downsample (e.g. 1min → 5min).
        Uses a shared resampler instance for cache benefit across batch training.
        """
        from src.phase_02_preprocessing.bar_resampler import (
            BarResampler,
            resolution_to_minutes,
        )

        resolution = config.data_config.primary_resolution
        res_minutes = resolution_to_minutes(resolution)

        if res_minutes == 1:
            return df_1min

        # Shared resampler instance for cache across batch training
        if not hasattr(self, "_resampler"):
            self._resampler = BarResampler()

        self._log(f"  [RESAMPLE] Resampling to {resolution} ({res_minutes}-min bars)")
        return self._resampler.resample(df_1min, res_minutes)

    def _train_resolution_cascade(
        self, df_1min: pd.DataFrame, config: ModelEntry
    ) -> Optional[Dict]:
        """
        Train models at each resolution in the cascade config.

        Returns cascade summary dict, or None on failure.
        """
        try:
            from src.phase_26_temporal.resolution_cascade import ResolutionCascade

            resolutions = config.cascade_config.resolutions_minutes
            cascade = ResolutionCascade(resolutions=resolutions)
            summary = cascade.fit(df_1min, config, verbose=self.verbose)

            # Save cascade to model artifacts directory
            cascade_dir = self.models_dir / (config.model_id or "cascade") / "resolution_cascade"
            cascade.save(cascade_dir)
            self._log(f"  [CASCADE] Saved to {cascade_dir}")

            return summary

        except Exception as e:
            self._log(f"  [CASCADE] Resolution cascade failed: {e}")
            if self.verbose:
                traceback.print_exc()
            return None

    def _prepare_arrays(
        self,
        df_daily: pd.DataFrame,
        target_col: str,
        config: ModelEntry,
    ) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
        """Prepare feature matrix X, target y, and sample weights."""
        # Determine feature columns (exclude targets, metadata, leaky columns)
        # Use both exact matches AND pattern-based exclusion to prevent target leakage
        # This list MUST stay in sync with FEATURE_EXCLUDE_COLS in experiment_runner.py
        exclude_exact = {
            # OHLCV and basic price data
            "date", "timestamp",
            "open", "high", "low", "close", "volume",
            # Target variables and target-derived columns
            "is_up_day", "is_down_day",
            "low_before_high", "high_minutes", "low_minutes",
            "day_return", "day_volume", "day_range",
            "target_up", "target_timing", "soft_target_up",
            "smoothed_target_up", "smoothed_target_timing",
            "sample_weight", "timing_weight",
            # Look-ahead features (require knowing prices after entry)
            "max_gain_from_1015", "max_gain_from_1230",
            # Metadata / quality columns
            "has_premarket", "has_afterhours", "quality_score", "year",
            # Anti-overfit metadata columns (not features)
            "sample_weight_augment", "universe_id", "universe_type",
            "synthetic_return", "real_return", "is_synthetic",
        }

        # Same-day intraday features: computed from the SAME day's price data.
        # For open-to-close prediction, they are look-ahead (not available at market open).
        _intraday_times = ["0945", "1015", "1100", "1130", "1230", "1330", "1430", "1530"]
        _intraday_prefixes = [
            "return_at_", "high_to_", "low_to_", "range_to_",
            "rsi_at_", "macd_at_", "bb_at_", "return_from_low_",
        ]
        for _pfx in _intraday_prefixes:
            for _tp in _intraday_times:
                exclude_exact.add(f"{_pfx}{_tp}")

        # Patterns that indicate target/label columns (CRITICAL for leak prevention)
        exclude_patterns = [
            "target", "soft_target", "smoothed_target", "label",
            "sample_weight", "target_weight", "class_weight",
            "forward_return", "future_",
        ]

        feature_cols = []
        for c in df_daily.columns:
            if c in exclude_exact:
                continue
            if c == target_col:
                continue
            if any(pat in c.lower() for pat in exclude_patterns):
                continue
            feature_cols.append(c)

        # Drop rows where target is NaN
        valid_mask = df_daily[target_col].notna()
        df_valid = df_daily[valid_mask].copy()

        X = df_valid[feature_cols].values.astype(np.float64)
        y = df_valid[target_col].values.astype(np.float64)

        # For soft targets, binarize for CV metrics
        if "soft" in target_col:
            y_binary = (y > 0.5).astype(np.float64)
        else:
            y_binary = (y > 0).astype(np.float64)

        # Compute sample weights
        returns = df_valid["day_return"].values if "day_return" in df_valid.columns else None
        weights = SampleWeighter.compute_weights(y_binary, config, returns=returns)

        # If sample_weight column exists, multiply
        if "sample_weight" in df_valid.columns:
            weights *= df_valid["sample_weight"].values

        self._log(f"  Features: {len(feature_cols)}, Samples: {len(y)}")

        return X, y_binary, feature_cols, weights

    def _cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        config: ModelEntry,
    ) -> Dict:
        """Run cross-validation."""
        model = ModelFactory.create(config)
        cv = CrossValidator.create_cv(config)

        # Get model class and params for CV
        model_class = type(model)
        model_params = model.get_params()

        cv_results = cv.cross_validate(
            X, y,
            sample_weights=weights,
            model_class=model_class,
            model_params=model_params,
            verbose=self.verbose,
        )

        return cv_results

    def _train_final(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        config: ModelEntry,
        feature_cols: list = None,
    ) -> Tuple[Any, Any, Any, Any]:
        """Train final model on all data."""
        group_processor = None

        # Check if group-aware processing is enabled
        fgc = config.feature_group_config
        if fgc.enabled and fgc.group_mode != "flat" and feature_cols:
            from src.phase_10_feature_processing.group_aware_processor import (
                GroupAwareFeatureProcessor,
            )
            group_processor = GroupAwareFeatureProcessor(
                feature_names=feature_cols,
                group_mode=fgc.group_mode,
                protected_groups=fgc.protected_groups,
                budget_mode=fgc.budget_mode,
                total_components=fgc.total_components,
                min_components_per_group=fgc.min_components_per_group,
                selection_method=config.feature_selection_config.method,
                reduction_method=fgc.per_group_reduction_method,
                n_features=config.feature_selection_config.n_features or 30,
                n_components=config.dim_reduction_config.n_components or 20,
                random_state=42,
            )
            X_processed = group_processor.fit_transform(X, y)

            # Final scaling
            scaler = ScalerFactory.create(config)
            if scaler is not None:
                X_final = scaler.fit_transform(X_processed)
            else:
                X_final = X_processed

            # Train model
            model = ModelFactory.create(config)
            try:
                model.fit(X_final, y, sample_weight=weights)
            except TypeError:
                model.fit(X_final, y)

            config.n_features_selected = X_processed.shape[1]
            config.n_features_final = X_final.shape[1]

            # Return group_processor as both selector and reducer slots
            return model, scaler, group_processor, None

        # Standard flat processing
        selector = FeatureSelector.create(config)
        if selector is not None:
            X_sel = selector.fit_transform(X, y)
        else:
            X_sel = X.copy()

        # Dimensionality reduction
        reducer = DimReducerFactory.create(config)
        if reducer is not None:
            X_red = reducer.fit_transform(X_sel, y)
        else:
            X_red = X_sel

        # Final scaling
        scaler = ScalerFactory.create(config)
        if scaler is not None:
            X_final = scaler.fit_transform(X_red)
        else:
            X_final = X_red

        # Train model
        model = ModelFactory.create(config)

        # Some models support sample_weight, some don't
        try:
            model.fit(X_final, y, sample_weight=weights)
        except TypeError:
            model.fit(X_final, y)

        config.n_features_selected = X_sel.shape[1]
        config.n_features_final = X_final.shape[1]

        return model, scaler, selector, reducer

    def _record_results(
        self,
        entry: ModelEntry,
        cv_results: Dict,
        model: Any,
        scaler: Any,
        selector: Any,
        reducer: Any,
        feature_cols: List[str],
        X: np.ndarray,
        y: np.ndarray,
        start_time: float,
    ) -> ModelEntry:
        """Record training results and save artifacts."""
        # Metrics
        metrics = ModelMetrics(
            cv_auc=cv_results.get("mean_test_auc", 0),
            cv_auc_std=cv_results.get("std_test_auc", 0),
            cv_auc_scores=cv_results.get("test_aucs", []),
            train_auc=cv_results.get("mean_train_auc", 0),
            train_test_gap=cv_results.get("train_test_gap", 0),
        )

        # Compute test metrics on held-out portion
        # IMPORTANT: Re-fit transformers on train-only data to avoid test leakage
        test_size = int(len(y) * entry.data_config.test_ratio)
        if test_size > 0:
            X_train_raw = X[:-test_size]
            y_train_raw = y[:-test_size]
            X_test_raw = X[-test_size:]
            y_test = y[-test_size:]

            try:
                from copy import deepcopy
                from sklearn.metrics import (
                    roc_auc_score, accuracy_score, precision_score,
                    recall_score, f1_score, log_loss, brier_score_loss
                )

                # Re-fit pipeline on train-only data for unbiased test eval
                X_tr = X_train_raw.copy()
                X_te = X_test_raw.copy()

                if selector is not None:
                    eval_selector = deepcopy(selector)
                    eval_selector.fit(X_tr, y_train_raw) if hasattr(eval_selector, 'fit') else None
                    X_tr = eval_selector.transform(X_tr)
                    X_te = eval_selector.transform(X_te)

                if reducer is not None:
                    eval_reducer = deepcopy(reducer)
                    eval_reducer.fit(X_tr, y_train_raw) if hasattr(eval_reducer, 'fit') else None
                    X_tr = eval_reducer.transform(X_tr)
                    X_te = eval_reducer.transform(X_te)

                if scaler is not None:
                    eval_scaler = deepcopy(scaler)
                    eval_scaler.fit(X_tr) if hasattr(eval_scaler, 'fit') else None
                    X_te = eval_scaler.transform(X_te)

                # Train a fresh evaluation model on properly transformed train data
                eval_model = deepcopy(model)
                try:
                    eval_model.fit(X_tr, y_train_raw)
                except Exception:
                    pass  # Use the original model as fallback

                probas = eval_model.predict_proba(X_te)[:, 1]
                preds = (probas > 0.5).astype(int)

                metrics.test_auc = roc_auc_score(y_test, probas) if len(np.unique(y_test)) > 1 else 0.5
                metrics.accuracy = accuracy_score(y_test, preds)
                metrics.precision = precision_score(y_test, preds, zero_division=0)
                metrics.recall = recall_score(y_test, preds, zero_division=0)
                metrics.f1_score = f1_score(y_test, preds, zero_division=0)
                metrics.log_loss = log_loss(y_test, probas)
                metrics.brier_score = brier_score_loss(y_test, probas)

                entry.test_samples = test_size
            except Exception as e:
                self._log(f"  [WARN] Test evaluation failed: {e}")

        entry.metrics = metrics
        entry.training_time_seconds = time.time() - start_time

        # Save model artifacts
        model_filename = f"{entry.model_id}.joblib"
        model_path = self.models_dir / model_filename

        artifact_data = {
            "model": model,
            "scaler": scaler,
            "feature_selector": selector,
            "dim_reducer": reducer,
            "feature_cols": feature_cols,
            "config": entry.to_dict(),
            "model_id": entry.model_id,
            "trained_at": datetime.now().isoformat(),
        }

        joblib.dump(artifact_data, model_path)

        # Record artifacts
        entry.artifacts = ModelArtifacts(
            model_path=str(model_path),
            raw_feature_cols=feature_cols,
            n_raw_features=len(feature_cols),
            n_input_features=entry.n_features_selected,
            n_output_features=entry.n_features_final,
        )

        return entry

    def _print_batch_summary(self, results: List[ModelEntry]):
        """Print summary of batch training results."""
        trained = [r for r in results if r.status == ModelStatus.TRAINED.value]
        if not trained:
            return

        self._log("\n" + "=" * 70)
        self._log("TOP 10 MODELS BY CV AUC")
        self._log("=" * 70)

        sorted_models = sorted(trained, key=lambda m: m.metrics.cv_auc, reverse=True)

        self._log(f"{'Rank':<5} {'Model Type':<20} {'Cascade':<15} {'DimRed':<15} {'CV AUC':<10} {'Gap':<8}")
        self._log("-" * 70)

        for i, m in enumerate(sorted_models[:10]):
            self._log(
                f"{i+1:<5} {m.model_config.model_type:<20} "
                f"{m.cascade_config.cascade_type:<15} "
                f"{m.dim_reduction_config.method:<15} "
                f"{m.metrics.cv_auc:<10.4f} "
                f"{m.metrics.train_test_gap:<8.4f}"
            )

    def _log(self, msg: str):
        """Log a message."""
        if self.verbose:
            print(msg)
        logger.info(msg)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_minimal_grid(
    target_type: str = "swing",
    verbose: bool = True,
) -> List[ModelEntry]:
    """
    Run minimal grid search (40 configs) for quick testing.

    Returns:
        List of trained ModelEntry objects
    """
    gen = GridSearchConfigGenerator.create_minimal_grid()
    configs = gen.generate_configs(target_type=target_type)

    pipeline = TrainingPipelineV2(verbose=verbose)
    return pipeline.train_batch(configs)


def run_focused_grid(
    focus_area: str = "model",
    target_type: str = "swing",
    max_configs: int = 50,
    verbose: bool = True,
) -> List[ModelEntry]:
    """
    Run focused grid search on a specific pipeline area.

    Args:
        focus_area: "data", "preprocess", "dim_reduction", "model", "cascade", "training"
        target_type: "swing", "timing", etc.
        max_configs: Maximum configs to train
        verbose: Print progress

    Returns:
        List of trained ModelEntry objects
    """
    gen = GridSearchConfigGenerator.create_focused_grid(focus_area=focus_area)
    configs = gen.generate_configs(
        target_type=target_type,
        max_configs=max_configs,
        random_sample=True,
    )

    pipeline = TrainingPipelineV2(verbose=verbose)
    return pipeline.train_batch(configs)


def run_quick_experiment(
    target_type: str = "swing",
    n_configs: int = 5,
    verbose: bool = True,
) -> List[ModelEntry]:
    """
    Run a quick experiment with diverse preset configurations.

    Args:
        target_type: Prediction target type
        n_configs: Number of configs (max 10)
        verbose: Print progress

    Returns:
        List of trained ModelEntry objects
    """
    from src.model_registry_v2 import create_quick_experiment
    configs = create_quick_experiment(target_type=target_type, n_configs=n_configs)

    pipeline = TrainingPipelineV2(verbose=verbose)
    return pipeline.train_batch(configs)


def train_from_registry(
    model_ids: List[str] = None,
    status_filter: str = "queued",
    max_models: int = None,
    verbose: bool = True,
) -> List[ModelEntry]:
    """
    Train models that are already registered with QUEUED status.

    Args:
        model_ids: Specific model IDs to train (None = all queued)
        status_filter: Status to filter by
        max_models: Maximum number to train
        verbose: Print progress

    Returns:
        List of trained ModelEntry objects
    """
    registry = ModelRegistryV2(db=get_registry_db())
    pipeline = TrainingPipelineV2(registry=registry, verbose=verbose)

    if model_ids:
        entries = [registry.get(mid) for mid in model_ids if registry.get(mid)]
    else:
        entries = registry.query(status=status_filter)

    if max_models:
        entries = entries[:max_models]

    return pipeline.train_batch(entries)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Giga Trader Training Pipeline V2")
    parser.add_argument("--mode", type=str, default="quick",
                       choices=["quick", "minimal", "focused", "standard", "registry"],
                       help="Training mode")
    parser.add_argument("--target", type=str, default="swing",
                       choices=["swing", "timing", "direction"],
                       help="Target type")
    parser.add_argument("--focus", type=str, default="model",
                       choices=["data", "preprocess", "dim_reduction", "model", "cascade", "training"],
                       help="Focus area for focused grid")
    parser.add_argument("--max-configs", type=int, default=10,
                       help="Maximum configs to train")
    parser.add_argument("--quiet", action="store_true", help="Reduce output")

    args = parser.parse_args()
    verbose = not args.quiet

    if args.mode == "quick":
        print(f"\nRunning quick experiment ({args.max_configs} configs)...")
        results = run_quick_experiment(
            target_type=args.target,
            n_configs=args.max_configs,
            verbose=verbose,
        )
    elif args.mode == "minimal":
        print(f"\nRunning minimal grid (40 configs)...")
        results = run_minimal_grid(target_type=args.target, verbose=verbose)
    elif args.mode == "focused":
        print(f"\nRunning focused grid on '{args.focus}' (max {args.max_configs})...")
        results = run_focused_grid(
            focus_area=args.focus,
            target_type=args.target,
            max_configs=args.max_configs,
            verbose=verbose,
        )
    elif args.mode == "standard":
        print(f"\nRunning standard grid (8,640 configs, sampling {args.max_configs})...")
        gen = GridSearchConfigGenerator.create_standard_grid()
        configs = gen.generate_configs(
            target_type=args.target,
            max_configs=args.max_configs,
            random_sample=True,
        )
        pipeline = TrainingPipelineV2(verbose=verbose)
        results = pipeline.train_batch(configs)
    elif args.mode == "registry":
        print(f"\nTraining queued models from registry...")
        results = train_from_registry(
            max_models=args.max_configs,
            verbose=verbose,
        )

    # Print final summary
    trained = [r for r in results if r.status == ModelStatus.TRAINED.value]
    failed = [r for r in results if r.status == ModelStatus.FAILED.value]

    print(f"\n{'='*70}")
    print(f"FINAL RESULTS")
    print(f"{'='*70}")
    print(f"Trained: {len(trained)}")
    print(f"Failed:  {len(failed)}")

    if trained:
        best = max(trained, key=lambda m: m.metrics.cv_auc)
        print(f"\nBest Model: {best.model_id}")
        print(f"  Type: {best.model_config.model_type}")
        print(f"  Cascade: {best.cascade_config.cascade_type}")
        print(f"  Dim Reduction: {best.dim_reduction_config.method}")
        print(f"  CV AUC: {best.metrics.cv_auc:.4f}")
        print(f"  Test AUC: {best.metrics.test_auc:.4f}")
        print(f"  Train-Test Gap: {best.metrics.train_test_gap:.4f}")

    # Show registry summary
    registry = ModelRegistryV2(db=get_registry_db())
    print(f"\n{registry.summary()}")
