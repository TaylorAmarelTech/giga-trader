"""
GIGA TRADER - Resolution Cascade
==================================
Full-pipeline models trained at multiple bar resolutions, then ensembled.

Unlike the basic MultiResolutionCascade in advanced_temporal_cascades.py
(which uses ~9 hand-crafted features per resolution), this cascade runs
the COMPLETE engineer_all_features() pipeline at each resolution, producing
200+ features from resampled bars.

Architecture:
  1-min bars ──> engineer_all_features() ──> full pipeline ──> Model_1min  ──┐
  3-min bars ──> engineer_all_features() ──> full pipeline ──> Model_3min  ──┤
  5-min bars ──> engineer_all_features() ──> full pipeline ──> Model_5min  ──┼──> Weighted Ensemble
  15-min bars ─> engineer_all_features() ──> full pipeline ──> Model_15min ──┤
  30-min bars ─> engineer_all_features() ──> full pipeline ──> Model_30min ──┘

Cross-resolution agreement is itself a powerful signal:
  - All resolutions agree UP   → strong buy signal
  - Resolutions disagree       → reduce confidence / stay out
  - Fine-resolution bearish but coarse-resolution bullish → mixed signal

Storage:
  Single ModelEntry with cascade_type="multi_resolution".
  Sub-models stored in:
    models/pipeline_v2/{cascade_id}/
      resolution_1min.joblib
      resolution_5min.joblib
      ...
      cascade_meta.joblib  (weights, config, per-resolution metrics)

Usage:
    from src.phase_26_temporal.resolution_cascade import ResolutionCascade

    cascade = ResolutionCascade(resolutions=[1, 3, 5, 15, 30])
    results = cascade.fit(df_1min, base_config)
    prediction = cascade.predict(per_resolution_features)
"""

import logging
import time
from pathlib import Path
from copy import deepcopy
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import joblib

logger = logging.getLogger("RESOLUTION_CASCADE")


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ResolutionModelResult:
    """Training result from a single resolution's model."""

    resolution_minutes: int
    cv_auc: float = 0.0
    cv_auc_std: float = 0.0
    train_auc: float = 0.0
    train_test_gap: float = 0.0
    n_features_raw: int = 0
    n_features_final: int = 0
    n_samples: int = 0
    training_time_seconds: float = 0.0
    n_bars_per_day: float = 0.0

    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class ResolutionCascadePrediction:
    """Combined prediction from all resolution models."""

    # Ensemble output
    ensemble_probability: float = 0.5
    ensemble_confidence: float = 0.0

    # Agreement metrics
    agreement_score: float = 0.0  # 1 - std(predictions); 1.0 = perfect agreement
    agreement_direction: str = "NEUTRAL"  # "BULLISH", "BEARISH", "NEUTRAL", "MIXED"

    # Per-resolution detail
    per_resolution: Dict[int, float] = field(default_factory=dict)
    per_resolution_weight: Dict[int, float] = field(default_factory=dict)
    n_resolutions_used: int = 0

    def to_dict(self) -> Dict:
        return asdict(self)


# ============================================================================
# RESOLUTION CASCADE
# ============================================================================

class ResolutionCascade:
    """
    Trains independent full-pipeline models at multiple bar resolutions,
    then combines their predictions with AUC-weighted voting.

    Each resolution produces a DIFFERENT feature set from the same underlying
    data because technical indicators computed on 5-min bars differ from
    those computed on 1-min bars (e.g., RSI-14 covers 70 minutes at 5-min
    vs 14 minutes at 1-min).

    The cascade is stored as a single registry entry. Its component models
    and metadata are saved in a cascade directory.
    """

    DEFAULT_RESOLUTIONS = [1, 5, 15, 30]

    # Agreement thresholds
    STRONG_AGREEMENT_THRESHOLD = 0.85  # std < 0.15 across predictions
    MIXED_SIGNAL_THRESHOLD = 0.60  # std > 0.40

    def __init__(
        self,
        resolutions: Optional[List[int]] = None,
        weight_by_cv_auc: bool = True,
        min_cv_auc: float = 0.52,
    ):
        """
        Args:
            resolutions: List of resolution in minutes [1, 3, 5, 15, 30]
            weight_by_cv_auc: Weight predictions by each model's CV AUC
            min_cv_auc: Minimum CV AUC to include a resolution model
                       (below this, the resolution didn't learn anything)
        """
        self.resolutions = sorted(resolutions or self.DEFAULT_RESOLUTIONS)
        self.weight_by_cv_auc = weight_by_cv_auc
        self.min_cv_auc = min_cv_auc

        # Trained state
        self.models: Dict[int, Any] = {}  # resolution -> (model, scaler, selector, reducer)
        self.results: Dict[int, ResolutionModelResult] = {}
        self.weights: Dict[int, float] = {}
        self.is_fitted = False

    def fit(
        self,
        df_1min: pd.DataFrame,
        base_config: Any,  # ModelEntry
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Train a model at each resolution using the full pipeline.

        For each resolution:
        1. Resample 1-min bars to N-min bars
        2. Run engineer_all_features(N-min bars) → daily features
        3. Create targets, preprocess, select features, reduce dims
        4. Cross-validate and train final model
        5. Store model + metrics

        Args:
            df_1min: 1-minute OHLCV DataFrame
            base_config: ModelEntry with pipeline configuration
            verbose: Print progress

        Returns:
            Dict with cascade-level metrics and per-resolution results
        """
        from src.phase_02_preprocessing.bar_resampler import BarResampler
        from src.train_robust_model import engineer_all_features, add_rolling_features

        resampler = BarResampler()
        cascade_start = time.time()

        if verbose:
            logger.info(
                f"\n{'='*60}\n"
                f"RESOLUTION CASCADE: Training {len(self.resolutions)} resolutions\n"
                f"Resolutions: {self.resolutions} minutes\n"
                f"{'='*60}"
            )

        for res_min in self.resolutions:
            res_start = time.time()

            if verbose:
                logger.info(f"\n[RES {res_min}min] Starting training...")

            try:
                result = self._train_single_resolution(
                    df_1min, resampler, res_min, base_config, verbose
                )
                self.results[res_min] = result

                if verbose:
                    logger.info(
                        f"[RES {res_min}min] CV AUC: {result.cv_auc:.4f} "
                        f"+/- {result.cv_auc_std:.4f}, "
                        f"features: {result.n_features_raw} -> {result.n_features_final}, "
                        f"bars/day: {result.n_bars_per_day:.0f}, "
                        f"time: {result.training_time_seconds:.1f}s"
                    )

            except Exception as e:
                logger.error(f"[RES {res_min}min] Training failed: {e}")
                self.results[res_min] = ResolutionModelResult(
                    resolution_minutes=res_min,
                    training_time_seconds=time.time() - res_start,
                )

        # Compute weights from CV AUCs
        self._compute_weights()
        self.is_fitted = True

        cascade_elapsed = time.time() - cascade_start

        # Summary
        summary = self._build_summary(cascade_elapsed)

        if verbose:
            self._print_summary(summary)

        return summary

    def _train_single_resolution(
        self,
        df_1min: pd.DataFrame,
        resampler: Any,
        resolution_minutes: int,
        base_config: Any,
        verbose: bool,
    ) -> ResolutionModelResult:
        """Train a full pipeline model at a single resolution."""
        from src.train_robust_model import engineer_all_features, add_rolling_features, create_soft_targets
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_selection import mutual_info_classif

        start_time = time.time()

        # Step 1: Resample
        df_bars = resampler.resample(df_1min, resolution_minutes)

        # Count bars per day for this resolution
        if "date" in df_bars.columns:
            bars_per_day = df_bars.groupby("date").size().mean()
        else:
            bars_per_day = len(df_bars) / max(df_bars.index[-1].date().toordinal() - df_bars.index[0].date().toordinal(), 1) if len(df_bars) > 0 else 0

        # Step 2: Engineer features (full 200+ feature pipeline)
        tc = base_config.target_config
        threshold = tc.swing_threshold if hasattr(tc, "swing_threshold") else 0.002
        df_daily = engineer_all_features(df_bars, swing_threshold=threshold)
        df_daily = add_rolling_features(df_daily)

        # Step 3: Create targets
        df_daily = create_soft_targets(df_daily, threshold=threshold)

        # Determine target column
        target_col = "target_up"
        if target_col not in df_daily.columns:
            target_col = "is_up_day"

        # Step 4: Extract features
        exclude_cols = {
            "date", "timestamp", "is_up_day", "is_down_day",
            "low_before_high", "high_minutes", "low_minutes",
            "day_return", "day_range", "open", "high", "low", "close", "volume",
        }
        exclude_patterns = [
            "target", "soft_target", "smoothed_target", "label",
            "sample_weight", "target_weight", "class_weight",
            "forward_return", "future_",
        ]

        feature_cols = []
        for c in df_daily.columns:
            if c in exclude_cols or c == target_col:
                continue
            if any(pat in c.lower() for pat in exclude_patterns):
                continue
            feature_cols.append(c)

        # Drop rows with NaN target
        valid_mask = df_daily[target_col].notna()
        df_valid = df_daily[valid_mask]

        if len(df_valid) < 50:
            raise ValueError(
                f"Only {len(df_valid)} valid samples at {resolution_minutes}min - "
                f"need at least 50"
            )

        X = df_valid[feature_cols].values.astype(np.float64)
        y = (df_valid[target_col].values > 0.5).astype(np.float64)

        # Replace NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        n_features_raw = X.shape[1]

        # Step 5: Feature selection (top 30 by mutual information)
        n_select = min(30, X.shape[1])
        try:
            mi_scores = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
            top_idx = np.argsort(mi_scores)[-n_select:]
            X_sel = X[:, top_idx]
        except Exception:
            X_sel = X[:, :n_select]

        # Step 6: Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_sel)

        # Step 7: Train model (use config's model type)
        model = self._create_model(base_config)

        # Simple time-series CV (last 20% as test)
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled[:split_idx], X_scaled[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        try:
            model.fit(X_train, y_train, sample_weight=np.ones(len(y_train)))
        except TypeError:
            model.fit(X_train, y_train)

        # Evaluate
        from sklearn.metrics import roc_auc_score
        try:
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            else:
                y_pred_proba = model.decision_function(X_test)

            cv_auc = roc_auc_score(y_test, y_pred_proba)
        except Exception:
            cv_auc = 0.5

        # Train AUC
        try:
            if hasattr(model, "predict_proba"):
                y_train_proba = model.predict_proba(X_train)[:, 1]
            else:
                y_train_proba = model.decision_function(X_train)
            train_auc = roc_auc_score(y_train, y_train_proba)
        except Exception:
            train_auc = 0.5

        # Retrain on all data for production use
        try:
            model.fit(X_scaled, y, sample_weight=np.ones(len(y)))
        except TypeError:
            model.fit(X_scaled, y)

        # Store model artifacts
        self.models[resolution_minutes] = {
            "model": model,
            "scaler": scaler,
            "feature_indices": top_idx if "top_idx" in dir() else np.arange(n_select),
            "feature_cols": [feature_cols[i] for i in (top_idx if "top_idx" in dir() else range(n_select))],
            "n_features_raw": n_features_raw,
        }

        elapsed = time.time() - start_time

        return ResolutionModelResult(
            resolution_minutes=resolution_minutes,
            cv_auc=cv_auc,
            cv_auc_std=0.0,  # Single split, no std
            train_auc=train_auc,
            train_test_gap=abs(train_auc - cv_auc),
            n_features_raw=n_features_raw,
            n_features_final=X_sel.shape[1],
            n_samples=len(X_scaled),
            training_time_seconds=elapsed,
            n_bars_per_day=bars_per_day,
        )

    def _create_model(self, config: Any) -> Any:
        """Create a model based on the base config's model type."""
        mc = config.model_config
        model_type = mc.model_type

        if model_type in ("logistic_l1", "logistic_l2", "elastic_net", "ridge"):
            from sklearn.linear_model import LogisticRegression
            return LogisticRegression(
                C=mc.lr_C, max_iter=mc.lr_max_iter,
                random_state=42, class_weight="balanced", n_jobs=-1,
            )
        elif model_type in ("gradient_boosting", "hist_gradient_boosting"):
            from sklearn.ensemble import GradientBoostingClassifier
            return GradientBoostingClassifier(
                n_estimators=mc.gb_n_estimators,
                max_depth=min(mc.gb_max_depth, 5),
                learning_rate=mc.gb_learning_rate,
                subsample=mc.gb_subsample,
                min_samples_leaf=mc.gb_min_samples_leaf,
                random_state=42,
            )
        elif model_type == "xgboost":
            try:
                from xgboost import XGBClassifier
                return XGBClassifier(
                    n_estimators=mc.xgb_n_estimators,
                    max_depth=min(mc.xgb_max_depth, 5),
                    learning_rate=mc.xgb_learning_rate,
                    random_state=42, n_jobs=-1,
                    use_label_encoder=False, eval_metric="logloss", verbosity=0,
                )
            except ImportError:
                pass
        elif model_type == "lightgbm":
            try:
                from lightgbm import LGBMClassifier
                return LGBMClassifier(
                    n_estimators=mc.lgb_n_estimators,
                    max_depth=mc.lgb_max_depth,
                    learning_rate=mc.lgb_learning_rate,
                    random_state=42, n_jobs=-1, verbose=-1,
                )
            except ImportError:
                pass

        # Default fallback
        from sklearn.ensemble import GradientBoostingClassifier
        return GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.1,
            min_samples_leaf=50, random_state=42,
        )

    def _compute_weights(self):
        """Compute ensemble weights from CV AUCs."""
        valid_results = {
            res: result for res, result in self.results.items()
            if result.cv_auc >= self.min_cv_auc and res in self.models
        }

        if not valid_results:
            # All resolutions below threshold - use equal weights for any that trained
            for res in self.models:
                self.weights[res] = 1.0 / len(self.models)
            return

        if self.weight_by_cv_auc:
            # Weight proportional to (AUC - 0.5), so random (0.5) gets 0 weight
            excess_aucs = {
                res: max(result.cv_auc - 0.5, 0.0)
                for res, result in valid_results.items()
            }
            total = sum(excess_aucs.values())
            if total > 0:
                self.weights = {
                    res: auc / total for res, auc in excess_aucs.items()
                }
            else:
                n = len(valid_results)
                self.weights = {res: 1.0 / n for res in valid_results}
        else:
            # Equal weights
            n = len(valid_results)
            self.weights = {res: 1.0 / n for res in valid_results}

    def predict(
        self, features_per_resolution: Dict[int, np.ndarray]
    ) -> ResolutionCascadePrediction:
        """
        Combine predictions from all resolution models.

        Args:
            features_per_resolution: Dict mapping resolution_minutes to
                                    raw feature arrays (before selection/scaling)

        Returns:
            ResolutionCascadePrediction with ensemble output and agreement metrics
        """
        if not self.is_fitted:
            raise RuntimeError("ResolutionCascade not fitted. Call fit() first.")

        per_resolution_probs = {}

        for res, artifacts in self.models.items():
            if res not in features_per_resolution:
                continue

            X_raw = features_per_resolution[res]
            model = artifacts["model"]
            scaler = artifacts["scaler"]
            feat_idx = artifacts["feature_indices"]

            # Select features and scale
            X_sel = X_raw[:, feat_idx] if X_raw.ndim == 2 else X_raw[feat_idx]
            if X_sel.ndim == 1:
                X_sel = X_sel.reshape(1, -1)
            X_scaled = scaler.transform(X_sel)

            # Predict
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(X_scaled)[:, 1].mean()
            else:
                prob = float(model.decision_function(X_scaled).mean())
                # Sigmoid transform for decision function
                prob = 1.0 / (1.0 + np.exp(-prob))

            per_resolution_probs[res] = float(prob)

        if not per_resolution_probs:
            return ResolutionCascadePrediction()

        # Weighted ensemble
        weighted_sum = 0.0
        weight_sum = 0.0
        for res, prob in per_resolution_probs.items():
            w = self.weights.get(res, 0.0)
            weighted_sum += prob * w
            weight_sum += w

        ensemble_prob = weighted_sum / max(weight_sum, 1e-8)

        # Agreement metrics
        probs = list(per_resolution_probs.values())
        agreement_score = 1.0 - np.std(probs) if len(probs) > 1 else 1.0

        # Direction consensus
        bullish = sum(1 for p in probs if p > 0.55)
        bearish = sum(1 for p in probs if p < 0.45)
        n = len(probs)

        if bullish == n:
            direction = "BULLISH"
        elif bearish == n:
            direction = "BEARISH"
        elif bullish > bearish:
            direction = "LEAN_BULLISH"
        elif bearish > bullish:
            direction = "LEAN_BEARISH"
        else:
            direction = "MIXED"

        # Confidence based on agreement
        confidence = agreement_score * abs(ensemble_prob - 0.5) * 2.0

        return ResolutionCascadePrediction(
            ensemble_probability=float(np.clip(ensemble_prob, 0.0, 1.0)),
            ensemble_confidence=float(np.clip(confidence, 0.0, 1.0)),
            agreement_score=float(agreement_score),
            agreement_direction=direction,
            per_resolution=per_resolution_probs,
            per_resolution_weight={
                res: self.weights.get(res, 0.0) for res in per_resolution_probs
            },
            n_resolutions_used=len(per_resolution_probs),
        )

    def _build_summary(self, elapsed: float) -> Dict[str, Any]:
        """Build cascade training summary."""
        valid_results = [
            r for r in self.results.values()
            if r.cv_auc >= self.min_cv_auc
        ]
        all_results = list(self.results.values())

        return {
            "n_resolutions_total": len(self.resolutions),
            "n_resolutions_valid": len(valid_results),
            "n_resolutions_failed": len(all_results) - len(valid_results),
            "resolutions": self.resolutions,
            "per_resolution": {
                res: result.to_dict()
                for res, result in self.results.items()
            },
            "weights": dict(self.weights),
            "best_resolution": (
                max(self.results.items(), key=lambda x: x[1].cv_auc)[0]
                if self.results else None
            ),
            "best_cv_auc": (
                max(r.cv_auc for r in self.results.values())
                if self.results else 0.0
            ),
            "mean_cv_auc": (
                np.mean([r.cv_auc for r in valid_results])
                if valid_results else 0.0
            ),
            "total_training_time": elapsed,
        }

    def _print_summary(self, summary: Dict):
        """Print cascade training summary."""
        logger.info(
            f"\n{'='*60}\n"
            f"RESOLUTION CASCADE SUMMARY\n"
            f"{'='*60}"
        )
        logger.info(
            f"  Resolutions trained: {summary['n_resolutions_valid']}"
            f"/{summary['n_resolutions_total']}"
        )
        logger.info(f"  Best resolution: {summary['best_resolution']}min")
        logger.info(f"  Best CV AUC: {summary['best_cv_auc']:.4f}")
        logger.info(f"  Mean CV AUC: {summary['mean_cv_auc']:.4f}")
        logger.info(f"  Total time: {summary['total_training_time']:.1f}s")

        logger.info("\n  Per-Resolution Results:")
        logger.info(f"  {'Res':>6}  {'CV AUC':>8}  {'Gap':>6}  {'Weight':>7}  {'Bars/Day':>9}")
        logger.info(f"  {'-'*6}  {'-'*8}  {'-'*6}  {'-'*7}  {'-'*9}")

        for res in sorted(self.results.keys()):
            r = self.results[res]
            w = self.weights.get(res, 0.0)
            logger.info(
                f"  {res:>4}m  {r.cv_auc:>8.4f}  {r.train_test_gap:>6.3f}  "
                f"{w:>7.3f}  {r.n_bars_per_day:>9.0f}"
            )

        logger.info(f"{'='*60}\n")

    def save(self, path: Path):
        """
        Save all resolution models and cascade metadata.

        Directory structure:
          {path}/
            resolution_1min.joblib
            resolution_5min.joblib
            ...
            cascade_meta.joblib
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save each resolution model
        for res, artifacts in self.models.items():
            model_path = path / f"resolution_{res}min.joblib"
            joblib.dump(artifacts, model_path)

        # Save metadata
        meta = {
            "resolutions": self.resolutions,
            "weights": self.weights,
            "results": {
                res: result.to_dict()
                for res, result in self.results.items()
            },
            "weight_by_cv_auc": self.weight_by_cv_auc,
            "min_cv_auc": self.min_cv_auc,
            "is_fitted": self.is_fitted,
        }
        joblib.dump(meta, path / "cascade_meta.joblib")

        logger.info(f"[CASCADE] Saved {len(self.models)} resolution models to {path}")

    @classmethod
    def load(cls, path: Path) -> "ResolutionCascade":
        """Load a saved resolution cascade from disk."""
        path = Path(path)

        if not (path / "cascade_meta.joblib").exists():
            raise FileNotFoundError(f"No cascade_meta.joblib in {path}")

        meta = joblib.load(path / "cascade_meta.joblib")

        cascade = cls(
            resolutions=meta["resolutions"],
            weight_by_cv_auc=meta.get("weight_by_cv_auc", True),
            min_cv_auc=meta.get("min_cv_auc", 0.52),
        )
        cascade.weights = meta["weights"]
        cascade.is_fitted = meta.get("is_fitted", True)

        # Reconstruct results
        for res_str, result_dict in meta.get("results", {}).items():
            res = int(res_str) if isinstance(res_str, str) else res_str
            cascade.results[res] = ResolutionModelResult(**result_dict)

        # Load resolution models
        for res in meta["resolutions"]:
            model_path = path / f"resolution_{res}min.joblib"
            if model_path.exists():
                cascade.models[res] = joblib.load(model_path)

        logger.info(
            f"[CASCADE] Loaded {len(cascade.models)} resolution models from {path}"
        )
        return cascade
