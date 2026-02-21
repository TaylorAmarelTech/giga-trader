"""
GIGA TRADER - Advanced Stability Suite (Wave 30 + 34 + 35)
============================================================
19 complementary stability and robustness measures covering:

Fast (post-hoc, no retraining):
  1. PSI  — Population Stability Index (train/test distribution shift)
  2. CSI  — Characteristic Stability Index (per-feature drill-down)
  3. Adversarial Validation — classifier-based shift detection
  4. ECE  — Expected Calibration Error (probability quality)
  5. DSR  — Deflated Sharpe Ratio (multiple-testing correction)
  6. Distribution-Robust Scoring — worst temporal slice AUC (Wave 35)

Medium (1-5 retrains):
  7. SFI  — Single Feature Importance (permutation-based)
  8. Meta-Labeling — secondary model predicting primary correctness
  9. Knockoff Features — FDR-controlled feature discovery
  10. ADWIN — Adaptive windowing drift detection
  11. Disagreement Smoothing — cross-model weighted smoothing (Wave 34)
  12. Feature Causality — Granger-like F-test (Wave 35)
  13. Prediction Interval Coverage — bootstrap prediction intervals (Wave 35)

Expensive (5-10 retrains):
  14. CPCV — Combinatorial Purged Cross-Validation
  15. Stability Selection — bootstrap feature selection robustness
  16. Rashomon Set — near-optimal model prediction disagreement
  17. Adversarial Overfitting — memorization detection (Wave 34)

Optional (external packages):
  18. SHAP Consistency — feature ranking stability across folds
  19. Conformal Prediction — distribution-free prediction intervals

All methods return a dict with at minimum {"score": float, ...metadata}.
Scores are 0-1 (1 = maximally stable/good). Skipped methods return
{"score": -1.0, "skipped": True, "reason": "..."}.

Results are stored in a flexible dict (not fixed dataclass fields) so
weights and methods can be adjusted without schema changes.
"""

import warnings
warnings.filterwarnings("ignore")

import logging
from itertools import combinations
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy import stats as sp_stats
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)

# Optional imports — graceful fallback
try:
    from sklearn.inspection import permutation_importance as sklearn_perm_imp
    _HAS_PERM_IMP = True
except ImportError:
    _HAS_PERM_IMP = False

try:
    import shap
    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False

try:
    from mapie.classification import SplitConformalClassifier, CrossConformalClassifier
    from mapie.metrics.classification import (
        classification_coverage_score,
        classification_mean_width_score,
    )
    _HAS_MAPIE = True
except ImportError:
    _HAS_MAPIE = False


# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT WEIGHTS — adjustable without code changes
# ═══════════════════════════════════════════════════════════════════════════════
DEFAULT_WEIGHTS: Dict[str, float] = {
    "psi": 0.06,
    "csi": 0.04,
    "adversarial": 0.06,
    "ece": 0.05,
    "dsr": 0.05,
    "sfi": 0.05,
    "meta_label": 0.05,
    "knockoff": 0.05,
    "adwin": 0.05,
    "cpcv": 0.07,
    "stability_selection": 0.06,
    "rashomon": 0.05,
    "shap": 0.05,
    "conformal": 0.05,
    "adversarial_overfitting": 0.06,    # Wave 34: memorization detection
    "disagreement_smoothing": 0.05,     # Wave 34: cross-model smoothing
    "feature_causality": 0.05,          # Wave 35: Granger-like F-test
    "prediction_interval_coverage": 0.05,  # Wave 35: bootstrap prediction intervals
    "distribution_robust": 0.05,        # Wave 35: worst-slice robustness
}


class AdvancedStabilitySuite:
    """
    19 advanced stability and robustness measures.

    Parameters
    ----------
    n_cpcv_groups : int
        Number of groups for combinatorial purged CV (default 5 → 10 paths).
    n_stability_sel : int
        Number of bootstrap resamples for stability selection (default 10).
    n_rashomon : int
        Number of near-optimal models for Rashomon set analysis (default 5).
    n_sfi_repeats : int
        Repeats for permutation importance (default 3).
    purge_days : int
        Days to purge between CPCV folds (default 10).
    enable_shap : bool
        Whether to run SHAP consistency (requires shap package).
    enable_conformal : bool
        Whether to run conformal prediction (requires mapie package).
    weights : dict, optional
        Custom weights for composite score. Defaults to DEFAULT_WEIGHTS.
    """

    def __init__(
        self,
        n_cpcv_groups: int = 5,
        n_stability_sel: int = 10,
        n_rashomon: int = 5,
        n_sfi_repeats: int = 3,
        purge_days: int = 10,
        enable_shap: bool = True,
        enable_conformal: bool = True,
        n_noise_levels: int = 3,
        n_disagreement_folds: int = 5,
        weights: Optional[Dict[str, float]] = None,
    ):
        self.n_cpcv_groups = n_cpcv_groups
        self.n_stability_sel = n_stability_sel
        self.n_rashomon = n_rashomon
        self.n_sfi_repeats = n_sfi_repeats
        self.purge_days = purge_days
        self.enable_shap = enable_shap and _HAS_SHAP
        self.enable_conformal = enable_conformal and _HAS_MAPIE
        self.n_noise_levels = n_noise_levels
        self.n_disagreement_folds = n_disagreement_folds
        self.weights = weights or DEFAULT_WEIGHTS.copy()

    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN ENTRY POINT
    # ═══════════════════════════════════════════════════════════════════════════

    def run_all(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_factory_fn: Callable[[], object],
        trained_model: object,
        predictions: np.ndarray,
        base_auc: float,
        cv_scores: Optional[List[float]] = None,
        n_experiments_total: int = 1,
    ) -> Dict[str, Any]:
        """
        Run all applicable stability tests.

        Parameters
        ----------
        X_train, y_train : arrays
            Training data.
        X_test, y_test : arrays
            Held-out test data.
        model_factory_fn : callable
            Returns a fresh (unfitted) model with the same config.
        trained_model : object
            The already-fitted model (for SFI, SHAP, etc).
        predictions : array
            Predicted probabilities on X_test from trained_model.
        base_auc : float
            The base model's CV AUC for reference.
        cv_scores : list of float, optional
            Per-fold CV AUC scores (for ADWIN, DSR).
        n_experiments_total : int
            Total experiments run so far (for DSR multiple-testing correction).

        Returns
        -------
        Dict keyed by method name, each containing {"score": float, ...metadata}.
        Also contains "composite_advanced" weighted average and "weights_used".
        """
        results: Dict[str, Any] = {}

        logger.info("[ADVANCED STABILITY] Running 19-method suite...")

        # ── Fast group (post-hoc, no retraining) ──
        results["psi"] = self._population_stability_index(X_train, X_test)
        results["csi"] = self._characteristic_stability_index(X_train, X_test)
        results["adversarial"] = self._adversarial_validation(X_train, X_test)
        results["ece"] = self._expected_calibration_error(predictions, y_test)
        results["dsr"] = self._deflated_sharpe_ratio(cv_scores, n_experiments_total)
        results["distribution_robust"] = self._distribution_robust_scoring(
            X_test, y_test, trained_model, predictions
        )

        # ── Medium group (1-5 retrains) ──
        results["sfi"] = self._single_feature_importance(
            trained_model, X_test, y_test
        )
        results["meta_label"] = self._meta_labeling(X_test, predictions, y_test)
        results["knockoff"] = self._knockoff_features(
            X_train, y_train, model_factory_fn
        )
        results["adwin"] = self._adwin_drift(cv_scores)
        results["disagreement_smoothing"] = self._disagreement_weighted_smoothing(
            X_train, y_train, X_test, y_test, model_factory_fn
        )
        results["feature_causality"] = self._feature_causality_scoring(
            X_train, y_train
        )
        results["prediction_interval_coverage"] = self._prediction_interval_coverage(
            X_train, y_train, X_test, y_test, model_factory_fn
        )

        # ── Expensive group (5-10 retrains) ──
        results["cpcv"] = self._combinatorial_purged_cv(
            X_train, y_train, model_factory_fn
        )
        results["stability_selection"] = self._stability_selection(
            X_train, y_train, model_factory_fn
        )
        results["rashomon"] = self._rashomon_set(
            X_train, y_train, X_test, y_test, model_factory_fn, base_auc
        )
        results["adversarial_overfitting"] = self._adversarial_overfitting_detection(
            X_train, y_train, X_test, y_test,
            model_factory_fn, trained_model, predictions, base_auc
        )

        # ── Optional group (external packages) ──
        results["shap"] = self._shap_consistency(
            trained_model, X_train, X_test
        )
        results["conformal"] = self._conformal_prediction(
            trained_model, X_train, y_train, X_test, y_test
        )

        # ── Composite score ──
        composite = self._compute_composite(results)
        results["composite_advanced"] = composite
        results["weights_used"] = self.weights.copy()

        # Log summary
        scored = {k: v["score"] for k, v in results.items()
                  if isinstance(v, dict) and "score" in v and v.get("score", -1) >= 0}
        logger.info(f"  Advanced Stability Composite: {composite:.3f}")
        for name, score in sorted(scored.items()):
            logger.info(f"    {name:>22}: {score:.3f}")

        return results

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. PSI — Population Stability Index
    # ═══════════════════════════════════════════════════════════════════════════

    def _population_stability_index(
        self, X_train: np.ndarray, X_test: np.ndarray, n_bins: int = 10
    ) -> Dict[str, Any]:
        """Measure overall distribution shift between train and test."""
        try:
            n_features = X_train.shape[1]
            psi_values = []

            for j in range(n_features):
                psi_j = self._compute_psi_single(
                    X_train[:, j], X_test[:, j], n_bins
                )
                psi_values.append(psi_j)

            mean_psi = float(np.mean(psi_values))
            max_psi = float(np.max(psi_values))
            # PSI > 0.2 = significant shift
            score = float(max(0.0, 1.0 - mean_psi * 5))

            return {
                "score": score,
                "mean_psi": mean_psi,
                "max_psi": max_psi,
                "n_features": n_features,
                "n_high_psi": int(sum(1 for p in psi_values if p > 0.2)),
            }
        except Exception as e:
            logger.warning(f"  PSI failed: {e}")
            return {"score": 0.5, "error": str(e)}

    @staticmethod
    def _compute_psi_single(
        expected: np.ndarray, actual: np.ndarray, n_bins: int = 10
    ) -> float:
        """Compute PSI for a single feature."""
        eps = 1e-6
        # Use quantile-based binning from expected (train)
        try:
            bin_edges = np.percentile(
                expected, np.linspace(0, 100, n_bins + 1)
            )
            bin_edges = np.unique(bin_edges)
            if len(bin_edges) < 3:
                return 0.0

            expected_counts = np.histogram(expected, bins=bin_edges)[0]
            actual_counts = np.histogram(actual, bins=bin_edges)[0]

            expected_pct = expected_counts / (expected_counts.sum() + eps)
            actual_pct = actual_counts / (actual_counts.sum() + eps)

            # Avoid log(0)
            expected_pct = np.clip(expected_pct, eps, 1.0)
            actual_pct = np.clip(actual_pct, eps, 1.0)

            psi = float(np.sum(
                (actual_pct - expected_pct) * np.log(actual_pct / expected_pct)
            ))
            return max(0.0, psi)
        except Exception:
            return 0.0

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. CSI — Characteristic Stability Index
    # ═══════════════════════════════════════════════════════════════════════════

    def _characteristic_stability_index(
        self, X_train: np.ndarray, X_test: np.ndarray,
        instability_threshold: float = 0.25, n_bins: int = 10,
    ) -> Dict[str, Any]:
        """Per-feature distribution stability analysis."""
        try:
            n_features = X_train.shape[1]
            csi_values = []

            for j in range(n_features):
                csi_j = self._compute_psi_single(
                    X_train[:, j], X_test[:, j], n_bins
                )
                csi_values.append(csi_j)

            n_unstable = int(sum(1 for c in csi_values if c > instability_threshold))
            score = 1.0 - (n_unstable / max(n_features, 1))
            score = float(max(0.0, min(1.0, score)))

            # Rank features by instability
            sorted_indices = np.argsort(csi_values)[::-1]
            worst_5 = [(int(i), float(csi_values[i])) for i in sorted_indices[:5]]

            return {
                "score": score,
                "n_features": n_features,
                "n_unstable": n_unstable,
                "n_stable": n_features - n_unstable,
                "mean_csi": float(np.mean(csi_values)),
                "worst_features": worst_5,
            }
        except Exception as e:
            logger.warning(f"  CSI failed: {e}")
            return {"score": 0.5, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. Adversarial Validation
    # ═══════════════════════════════════════════════════════════════════════════

    def _adversarial_validation(
        self, X_train: np.ndarray, X_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Train a classifier to distinguish train from test data."""
        try:
            n_train = len(X_train)
            n_test = len(X_test)

            X_combined = np.vstack([X_train, X_test])
            y_combined = np.concatenate([
                np.zeros(n_train), np.ones(n_test)
            ])

            clf = LogisticRegression(
                C=1.0, max_iter=300, random_state=42, solver="lbfgs"
            )
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_val_score(
                clf, X_combined, y_combined, cv=cv, scoring="roc_auc"
            )
            adv_auc = float(np.mean(scores))

            # AUC=0.50 → can't distinguish → no shift → score=1.0
            # AUC=0.70 → clear shift → score=0.0
            score = float(max(0.0, 1.0 - (adv_auc - 0.50) * 5))

            # Feature importances for diagnostics
            clf.fit(X_combined, y_combined)
            importances = np.abs(clf.coef_[0])
            top_5_indices = np.argsort(importances)[::-1][:5]
            top_5 = [(int(i), float(importances[i])) for i in top_5_indices]

            return {
                "score": score,
                "adversarial_auc": adv_auc,
                "shift_detected": adv_auc > 0.55,
                "top_shifting_features": top_5,
            }
        except Exception as e:
            logger.warning(f"  Adversarial validation failed: {e}")
            return {"score": 0.5, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. ECE — Expected Calibration Error
    # ═══════════════════════════════════════════════════════════════════════════

    def _expected_calibration_error(
        self, predictions: np.ndarray, y_true: np.ndarray, n_bins: int = 10,
    ) -> Dict[str, Any]:
        """Measure probability calibration quality."""
        try:
            predictions = np.asarray(predictions)
            y_true = np.asarray(y_true)

            bin_edges = np.linspace(0, 1, n_bins + 1)
            bin_data = []
            ece = 0.0
            total = len(y_true)

            for i in range(n_bins):
                mask = (predictions >= bin_edges[i]) & (predictions < bin_edges[i + 1])
                if i == n_bins - 1:
                    mask = mask | (predictions == bin_edges[i + 1])

                n_in_bin = mask.sum()
                if n_in_bin == 0:
                    bin_data.append({
                        "bin": i, "n": 0,
                        "avg_pred": 0.0, "avg_actual": 0.0,
                    })
                    continue

                avg_pred = float(predictions[mask].mean())
                avg_actual = float(y_true[mask].mean())
                bin_ece = abs(avg_pred - avg_actual) * (n_in_bin / total)
                ece += bin_ece

                bin_data.append({
                    "bin": i,
                    "n": int(n_in_bin),
                    "avg_pred": round(avg_pred, 4),
                    "avg_actual": round(avg_actual, 4),
                    "gap": round(abs(avg_pred - avg_actual), 4),
                })

            ece = float(ece)
            # ECE=0 → perfect calibration → score=1.0
            # ECE=0.10 → poor → score=0.0
            score = float(max(0.0, 1.0 - ece * 10))

            return {
                "score": score,
                "ece": ece,
                "n_bins": n_bins,
                "bin_details": bin_data,
            }
        except Exception as e:
            logger.warning(f"  ECE failed: {e}")
            return {"score": 0.5, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. DSR — Deflated Sharpe Ratio
    # ═══════════════════════════════════════════════════════════════════════════

    def _deflated_sharpe_ratio(
        self,
        cv_scores: Optional[List[float]],
        n_experiments_total: int,
    ) -> Dict[str, Any]:
        """
        Adjust Sharpe-like metric for multiple testing.

        Uses the Bailey & de Prado (2014) haircut formula.
        """
        try:
            if not cv_scores or len(cv_scores) < 2:
                return {"score": 0.5, "skipped": True, "reason": "insufficient_cv_scores"}

            scores = np.array(cv_scores)
            sr_observed = float((scores.mean() - 0.50) / (scores.std() + 1e-8))
            n_trials = max(1, n_experiments_total)
            T = len(scores)

            # Skewness and kurtosis of CV scores
            skew = float(sp_stats.skew(scores))
            kurt = float(sp_stats.kurtosis(scores))

            # Expected max Sharpe under null (Bailey & de Prado)
            # E[max(SR)] ≈ sqrt(2 * ln(n_trials)) for n_trials independent tests
            e_max_sr = float(np.sqrt(2 * np.log(max(n_trials, 2))))

            # DSR: probability that observed SR is real, not luck
            # Simplified: apply a haircut based on number of trials
            haircut = min(0.90, np.log(n_trials) / 20.0)  # Cap at 90%
            dsr = sr_observed * (1.0 - haircut)

            # Also compute p-value using t-distribution
            if scores.std() > 1e-8:
                t_stat = sr_observed * np.sqrt(T)
                # Adjust for skew/kurtosis
                t_stat_adj = t_stat * (
                    1 - skew * sr_observed / 3.0
                    + (kurt - 3) * sr_observed ** 2 / 12.0
                )
                p_value = float(1 - sp_stats.t.cdf(abs(t_stat_adj), df=T - 1))
            else:
                p_value = 1.0

            # Score: DSR of 0.5+ is decent, 1.0+ is good, 2.0+ is excellent
            score = float(min(1.0, max(0.0, dsr / 2.0)))

            return {
                "score": score,
                "sharpe_observed": float(sr_observed),
                "sharpe_deflated": float(dsr),
                "haircut_pct": round(haircut * 100, 1),
                "n_trials": n_trials,
                "p_value": round(p_value, 4),
                "skewness": round(skew, 4),
                "kurtosis": round(kurt, 4),
                "e_max_sharpe": round(e_max_sr, 3),
            }
        except Exception as e:
            logger.warning(f"  DSR failed: {e}")
            return {"score": 0.5, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════════════════
    # 6. SFI — Single Feature Importance
    # ═══════════════════════════════════════════════════════════════════════════

    def _single_feature_importance(
        self, trained_model: object,
        X_test: np.ndarray, y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Permutation-based feature importance (no substitution effects)."""
        try:
            if not _HAS_PERM_IMP:
                return {"score": -1.0, "skipped": True, "reason": "sklearn too old"}

            # Handle pipeline objects (LeakProofPipeline) that don't expose fit()
            # by wrapping them with a predict_proba-only scorer
            model_for_perm = trained_model
            if not hasattr(trained_model, "fit") and hasattr(trained_model, "predict_proba"):
                # Create a wrapper that has fit() but doesn't do anything
                from sklearn.base import BaseEstimator, ClassifierMixin
                class _PredictOnlyWrapper(ClassifierMixin, BaseEstimator):
                    def __init__(self, pipeline):
                        self._pipeline = pipeline
                    def fit(self, X, y):
                        return self
                    def predict_proba(self, X):
                        return self._pipeline.predict_proba(X)
                    def predict(self, X):
                        return self._pipeline.predict(X)
                model_for_perm = _PredictOnlyWrapper(trained_model)

            perm_result = sklearn_perm_imp(
                model_for_perm, X_test, y_test,
                n_repeats=self.n_sfi_repeats,
                scoring="roc_auc",
                random_state=42,
                n_jobs=1,
            )

            importances = perm_result.importances_mean
            max_drop = float(np.max(importances))
            mean_drop = float(np.mean(importances))

            # If any single feature drops AUC by >0.10, model over-relies on it
            score = float(max(0.0, 1.0 - max_drop * 10))

            top_5_indices = np.argsort(importances)[::-1][:5]
            top_5 = [(int(i), round(float(importances[i]), 4)) for i in top_5_indices]

            return {
                "score": score,
                "max_drop": round(max_drop, 4),
                "mean_drop": round(mean_drop, 4),
                "n_features": len(importances),
                "top_features": top_5,
            }
        except Exception as e:
            logger.warning(f"  SFI failed: {e}")
            return {"score": 0.5, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════════════════
    # 7. Meta-Labeling
    # ═══════════════════════════════════════════════════════════════════════════

    def _meta_labeling(
        self, X_test: np.ndarray, predictions: np.ndarray, y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Train secondary model to predict primary model's correctness."""
        try:
            predictions = np.asarray(predictions)
            y_test = np.asarray(y_test)

            # Binary: was the primary prediction correct?
            predicted_class = (predictions >= 0.5).astype(int)
            meta_target = (predicted_class == y_test).astype(int)

            # Need both classes
            if len(np.unique(meta_target)) < 2:
                return {
                    "score": 0.5, "skipped": True,
                    "reason": "single_class_meta_target",
                }

            # Use predictions + features as meta-model input
            X_meta = np.column_stack([X_test, predictions.reshape(-1, 1)])

            clf = LogisticRegression(C=1.0, max_iter=300, random_state=42)
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            scores = cross_val_score(
                clf, X_meta, meta_target, cv=cv, scoring="roc_auc"
            )
            meta_auc = float(np.mean(scores))

            # Higher meta-AUC = model failures are more predictable
            # This is USEFUL info (for actual meta-labeling in production)
            # Score: meta_auc itself (higher = better meta-labeling potential)
            score = float(min(1.0, max(0.0, meta_auc)))

            # Accuracy of primary model
            primary_acc = float(np.mean(predicted_class == y_test))

            return {
                "score": score,
                "meta_auc": round(meta_auc, 4),
                "primary_accuracy": round(primary_acc, 4),
                "n_correct": int(meta_target.sum()),
                "n_incorrect": int(len(meta_target) - meta_target.sum()),
            }
        except Exception as e:
            logger.warning(f"  Meta-labeling failed: {e}")
            return {"score": 0.5, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════════════════
    # 8. Knockoff Features
    # ═══════════════════════════════════════════════════════════════════════════

    def _knockoff_features(
        self, X_train: np.ndarray, y_train: np.ndarray,
        model_factory_fn: Callable,
    ) -> Dict[str, Any]:
        """FDR-controlled feature discovery via knockoff filter."""
        try:
            n_features = X_train.shape[1]
            rng = np.random.RandomState(42)

            # Generate knockoffs by permuting each column independently
            X_knockoff = np.empty_like(X_train)
            for j in range(n_features):
                perm = rng.permutation(len(X_train))
                X_knockoff[:, j] = X_train[perm, j]

            # Stack original + knockoff
            X_augmented = np.hstack([X_train, X_knockoff])

            # Fit model on augmented data
            model = model_factory_fn()
            model.fit(X_augmented, y_train)

            # Extract feature importances
            if hasattr(model, "coef_"):
                importances = np.abs(model.coef_.ravel())
            elif hasattr(model, "feature_importances_"):
                importances = model.feature_importances_
            else:
                return {"score": 0.5, "skipped": True, "reason": "no_importances"}

            if len(importances) != 2 * n_features:
                return {"score": 0.5, "skipped": True, "reason": "importance_shape_mismatch"}

            orig_imp = importances[:n_features]
            knock_imp = importances[n_features:]

            # Feature survives if original importance > knockoff importance
            surviving = orig_imp > knock_imp
            n_surviving = int(surviving.sum())

            # Score: fraction of genuine features
            score = float(n_surviving / max(n_features, 1))

            return {
                "score": score,
                "n_features": n_features,
                "n_surviving": n_surviving,
                "n_knocked_out": n_features - n_surviving,
                "survival_rate": round(score, 3),
            }
        except Exception as e:
            logger.warning(f"  Knockoff features failed: {e}")
            return {"score": 0.5, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════════════════
    # 9. ADWIN — Adaptive Windowing Drift Detection
    # ═══════════════════════════════════════════════════════════════════════════

    def _adwin_drift(
        self, cv_scores: Optional[List[float]],
    ) -> Dict[str, Any]:
        """Detect concept drift in CV fold performance."""
        try:
            if not cv_scores or len(cv_scores) < 4:
                return {"score": 0.5, "skipped": True, "reason": "too_few_cv_scores"}

            scores = np.array(cv_scores, dtype=float)
            n = len(scores)
            mid = n // 2

            first_half = scores[:mid]
            second_half = scores[mid:]

            mean_diff = abs(float(first_half.mean() - second_half.mean()))
            std_pooled = float(np.sqrt(
                (first_half.var() + second_half.var()) / 2 + 1e-10
            ))

            # Normalized drift magnitude
            drift_magnitude = mean_diff / (std_pooled + 1e-6)
            drift_detected = drift_magnitude > 1.5  # >1.5 std shift

            # Also check monotonic trend (declining performance)
            if n >= 3:
                rank_corr, _ = sp_stats.spearmanr(np.arange(n), scores)
                trend_declining = rank_corr < -0.5
            else:
                rank_corr = 0.0
                trend_declining = False

            # Score: no drift → 1.0, strong drift → 0.0
            score = float(max(0.0, 1.0 - drift_magnitude * 0.33))

            return {
                "score": score,
                "drift_detected": bool(drift_detected),
                "drift_magnitude": round(drift_magnitude, 4),
                "mean_first_half": round(float(first_half.mean()), 4),
                "mean_second_half": round(float(second_half.mean()), 4),
                "trend_declining": bool(trend_declining),
                "rank_correlation": round(float(rank_corr), 4),
            }
        except Exception as e:
            logger.warning(f"  ADWIN failed: {e}")
            return {"score": 0.5, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════════════════
    # 10. CPCV — Combinatorial Purged Cross-Validation
    # ═══════════════════════════════════════════════════════════════════════════

    def _combinatorial_purged_cv(
        self, X_train: np.ndarray, y_train: np.ndarray,
        model_factory_fn: Callable,
    ) -> Dict[str, Any]:
        """
        Exhaustive combinatorial CV with purging between adjacent groups.

        With K=5 groups, generates C(5,2)=10 train/test paths.
        Adjacent groups have `purge_days` samples removed between them.
        """
        try:
            n_samples = len(y_train)
            K = self.n_cpcv_groups
            group_size = n_samples // K
            if group_size < 20:
                return {"score": 0.5, "skipped": True, "reason": "too_few_samples"}

            # Create group assignments (sequential — time-ordered)
            groups = np.zeros(n_samples, dtype=int)
            for g in range(K):
                start = g * group_size
                end = (g + 1) * group_size if g < K - 1 else n_samples
                groups[start:end] = g

            # Generate all C(K, 2) test combinations
            aucs = []
            path_details = []

            for test_groups in combinations(range(K), 2):
                test_mask = np.isin(groups, test_groups)
                train_mask = ~test_mask

                # Purge: remove samples near group boundaries
                purge_size = min(self.purge_days, group_size // 4)
                if purge_size > 0:
                    for tg in test_groups:
                        tg_start = tg * group_size
                        tg_end = (tg + 1) * group_size if tg < K - 1 else n_samples
                        # Purge before test group
                        purge_before_start = max(0, tg_start - purge_size)
                        train_mask[purge_before_start:tg_start] = False
                        # Purge after test group
                        purge_after_end = min(n_samples, tg_end + purge_size)
                        train_mask[tg_end:purge_after_end] = False

                X_tr, y_tr = X_train[train_mask], y_train[train_mask]
                X_te, y_te = X_train[test_mask], y_train[test_mask]

                if len(np.unique(y_tr)) < 2 or len(np.unique(y_te)) < 2:
                    continue

                try:
                    model = model_factory_fn()
                    model.fit(X_tr, y_tr)
                    proba = model.predict_proba(X_te)[:, 1]
                    auc = float(roc_auc_score(y_te, proba))
                    aucs.append(auc)
                    path_details.append({
                        "test_groups": test_groups,
                        "auc": round(auc, 4),
                        "n_train": int(train_mask.sum()),
                        "n_test": int(test_mask.sum()),
                    })
                except Exception:
                    pass

            if len(aucs) < 3:
                return {"score": 0.5, "n_paths": len(aucs), "aucs": aucs}

            auc_std = float(np.std(aucs))
            auc_mean = float(np.mean(aucs))
            # Low std = stable across all data partitions
            score = float(max(0.0, 1.0 - auc_std * 15))

            return {
                "score": score,
                "n_paths": len(aucs),
                "auc_mean": round(auc_mean, 4),
                "auc_std": round(auc_std, 4),
                "auc_min": round(float(min(aucs)), 4),
                "auc_max": round(float(max(aucs)), 4),
                "auc_range": round(float(max(aucs) - min(aucs)), 4),
                "paths": path_details,
            }
        except Exception as e:
            logger.warning(f"  CPCV failed: {e}")
            return {"score": 0.5, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════════════════
    # 11. Stability Selection
    # ═══════════════════════════════════════════════════════════════════════════

    def _stability_selection(
        self, X_train: np.ndarray, y_train: np.ndarray,
        model_factory_fn: Callable,
    ) -> Dict[str, Any]:
        """
        Bootstrap feature selection robustness.

        Run feature selection on multiple bootstrap resamples and track
        how consistently each feature is selected.
        """
        try:
            n_samples, n_features = X_train.shape
            selection_counts = np.zeros(n_features)

            for i in range(self.n_stability_sel):
                rng = np.random.RandomState(42 + i)
                indices = rng.choice(n_samples, size=n_samples, replace=True)
                X_boot = X_train[indices]
                y_boot = y_train[indices]

                if len(np.unique(y_boot)) < 2:
                    continue

                try:
                    model = model_factory_fn()
                    model.fit(X_boot, y_boot)

                    # Extract importances
                    if hasattr(model, "coef_"):
                        imp = np.abs(model.coef_.ravel())
                    elif hasattr(model, "feature_importances_"):
                        imp = model.feature_importances_
                    else:
                        continue

                    if len(imp) != n_features:
                        continue

                    # Select top 50% of features by importance
                    threshold = np.median(imp)
                    selected = imp >= threshold
                    selection_counts += selected.astype(float)
                except Exception:
                    pass

            if self.n_stability_sel == 0:
                return {"score": 0.5, "skipped": True, "reason": "zero_iterations"}

            selection_freq = selection_counts / self.n_stability_sel

            # Stable features: selected in >50% of bootstraps
            n_stable = int((selection_freq > 0.5).sum())
            # Very stable: selected in >80%
            n_very_stable = int((selection_freq > 0.8).sum())

            # Score: fraction of features that are at least moderately stable
            score = float(n_stable / max(n_features, 1))

            return {
                "score": score,
                "n_features": n_features,
                "n_stable_50pct": n_stable,
                "n_very_stable_80pct": n_very_stable,
                "mean_selection_freq": round(float(selection_freq.mean()), 3),
                "std_selection_freq": round(float(selection_freq.std()), 3),
            }
        except Exception as e:
            logger.warning(f"  Stability selection failed: {e}")
            return {"score": 0.5, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════════════════
    # 12. Rashomon Set Analysis
    # ═══════════════════════════════════════════════════════════════════════════

    def _rashomon_set(
        self, X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray,
        model_factory_fn: Callable, base_auc: float,
    ) -> Dict[str, Any]:
        """
        Train multiple near-optimal models and measure prediction disagreement.

        Models are trained on different bootstrap samples. Those within 2% of
        base_auc are kept. If they disagree on predictions → less reliable.
        """
        try:
            n_samples = len(y_train)
            predictions_list = []
            model_aucs = []

            for i in range(self.n_rashomon):
                rng = np.random.RandomState(42 + i * 17)
                indices = rng.choice(n_samples, size=n_samples, replace=True)
                X_boot = X_train[indices]
                y_boot = y_train[indices]

                if len(np.unique(y_boot)) < 2:
                    continue

                try:
                    model = model_factory_fn()
                    model.fit(X_boot, y_boot)

                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X_test)[:, 1]
                    elif hasattr(model, "decision_function"):
                        proba = model.decision_function(X_test)
                    else:
                        proba = model.predict(X_test).astype(float)

                    auc = float(roc_auc_score(y_test, proba))

                    # Keep if within 5% of base AUC (broader than 2% for small samples)
                    if auc >= base_auc - 0.05:
                        predictions_list.append(proba)
                        model_aucs.append(auc)
                except Exception:
                    pass

            if len(predictions_list) < 2:
                return {"score": 0.5, "n_models": len(predictions_list)}

            # Pairwise prediction correlation
            pred_matrix = np.array(predictions_list)
            correlations = []
            for i in range(len(predictions_list)):
                for j in range(i + 1, len(predictions_list)):
                    std_i = np.std(pred_matrix[i])
                    std_j = np.std(pred_matrix[j])
                    if std_i < 1e-10 or std_j < 1e-10:
                        corr = 1.0 if std_i < 1e-10 and std_j < 1e-10 else 0.5
                    else:
                        corr = float(np.corrcoef(pred_matrix[i], pred_matrix[j])[0, 1])
                        if np.isnan(corr):
                            corr = 0.5
                    correlations.append(corr)

            mean_corr = float(np.mean(correlations))
            # Map correlation to score
            score = float(max(0.0, min(1.0, (mean_corr - 0.2) / 0.8)))

            # Prediction variance per sample
            pred_std = float(np.mean(np.std(pred_matrix, axis=0)))

            return {
                "score": score,
                "n_models": len(predictions_list),
                "mean_correlation": round(mean_corr, 4),
                "min_correlation": round(float(min(correlations)), 4),
                "pred_std_mean": round(pred_std, 4),
                "model_aucs": [round(a, 4) for a in model_aucs],
                "auc_spread": round(float(max(model_aucs) - min(model_aucs)), 4),
            }
        except Exception as e:
            logger.warning(f"  Rashomon set failed: {e}")
            return {"score": 0.5, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════════════════
    # 15. Adversarial Overfitting Detection (Wave 34)
    # ═══════════════════════════════════════════════════════════════════════════

    def _adversarial_overfitting_detection(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_factory_fn: Callable,
        trained_model: object,
        predictions: np.ndarray,
        base_auc: float,
    ) -> Dict[str, Any]:
        """
        Detect whether a model memorized training data vs learned patterns.

        NOT the same as _adversarial_validation (which detects distribution shift).
        This method uses three sub-tests:
          1. Label noise tolerance: corrupt labels, retrain, measure degradation
          2. Feature perturbation: add noise to features, retrain, measure sensitivity
          3. Confidence vs correctness: check overconfidence on wrong predictions
        """
        try:
            predictions = np.asarray(predictions)
            y_test = np.asarray(y_test)

            # --- Sub-test 1: Label noise tolerance (3 retrains) ---
            noise_rates = [0.10, 0.20, 0.30]
            noise_aucs = []
            for rate in noise_rates:
                rng = np.random.RandomState(42 + int(rate * 100))
                y_noisy = y_train.copy()
                n_flip = int(len(y_train) * rate)
                flip_idx = rng.choice(len(y_train), size=n_flip, replace=False)
                y_noisy[flip_idx] = 1 - y_noisy[flip_idx]

                try:
                    model = model_factory_fn()
                    model.fit(X_train, y_noisy)
                    proba = model.predict_proba(X_test)[:, 1]
                    auc = float(roc_auc_score(y_test, proba))
                except Exception:
                    auc = 0.5
                noise_aucs.append(auc)

            degradations = [base_auc - a for a in noise_aucs]

            if max(abs(d) for d in degradations) < 0.01:
                noise_score = 0.2  # barely any degradation = memorized
            elif degradations[-1] > 0.25:
                noise_score = 0.3  # catastrophic collapse = fragile
            else:
                corr = float(np.corrcoef(noise_rates, degradations)[0, 1])
                linearity = max(0.0, corr)
                noise_score = 0.5 + 0.5 * linearity
                if 0.02 <= degradations[-1] <= 0.15:
                    noise_score = min(1.0, noise_score + 0.1)

            # --- Sub-test 2: Feature perturbation sensitivity (3 retrains) ---
            sigmas = [0.1, 0.3, 0.5]
            perturb_aucs = []
            for sigma in sigmas:
                rng = np.random.RandomState(42 + int(sigma * 100))
                noise = rng.randn(*X_train.shape) * sigma
                X_noisy = X_train + noise

                try:
                    model = model_factory_fn()
                    model.fit(X_noisy, y_train)
                    proba = model.predict_proba(X_test)[:, 1]
                    auc = float(roc_auc_score(y_test, proba))
                except Exception:
                    auc = 0.5
                perturb_aucs.append(auc)

            perturb_drops = [base_auc - a for a in perturb_aucs]
            max_drop = max(perturb_drops) if perturb_drops else 0.0
            # 1.0 if max_drop < 0.02, 0.0 if max_drop > 0.15
            perturb_score = float(max(0.0, min(1.0, 1.0 - (max_drop - 0.02) / 0.13)))

            # --- Sub-test 3: Confidence vs correctness (no retrain) ---
            predicted_class = (predictions >= 0.5).astype(int)
            correct_mask = (predicted_class == y_test)

            if correct_mask.sum() > 0 and (~correct_mask).sum() > 0:
                mean_conf_correct = float(predictions[correct_mask].mean())
                mean_conf_incorrect = float(predictions[~correct_mask].mean())
                overconfidence = abs(mean_conf_incorrect - 0.5)
                conf_gap = mean_conf_correct - overconfidence - 0.5
                conf_score = float(max(0.0, min(1.0, 0.5 + conf_gap * 2)))
                if overconfidence > 0.3:
                    conf_score = max(0.0, conf_score - 0.2)
            else:
                mean_conf_correct = float(predictions.mean())
                mean_conf_incorrect = 0.0
                overconfidence = 0.0
                conf_score = 0.5

            # --- Combined score ---
            score = 0.40 * noise_score + 0.35 * perturb_score + 0.25 * conf_score
            score = float(max(0.0, min(1.0, score)))

            return {
                "score": score,
                "noise_score": round(noise_score, 4),
                "perturb_score": round(perturb_score, 4),
                "confidence_score": round(conf_score, 4),
                "noise_aucs": {str(r): round(a, 4) for r, a in zip(noise_rates, noise_aucs)},
                "noise_degradations": [round(d, 4) for d in degradations],
                "perturb_aucs": {str(s): round(a, 4) for s, a in zip(sigmas, perturb_aucs)},
                "perturb_drops": [round(d, 4) for d in perturb_drops],
                "mean_conf_correct": round(mean_conf_correct, 4),
                "mean_conf_incorrect": round(mean_conf_incorrect, 4),
                "overconfidence": round(overconfidence, 4),
                "n_retrains": 6,
            }
        except Exception as e:
            logger.warning(f"  Adversarial overfitting detection failed: {e}")
            return {"score": 0.5, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════════════════
    # 16. Disagreement-Weighted Smoothing (Wave 34)
    # ═══════════════════════════════════════════════════════════════════════════

    def _disagreement_weighted_smoothing(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_factory_fn: Callable,
    ) -> Dict[str, Any]:
        """
        Disagreement-weighted smoothing across bootstrapped model variants.

        Trains K models on bootstrap subsets, measures:
        A. Feature importance disagreement across models
        B. Prediction disagreement per test sample
        C. Uncertainty-weighted smoothed predictions

        Score = 1.0 if models largely agree, 0.0 if massive disagreement.
        """
        try:
            K = self.n_disagreement_folds
            n_samples = len(y_train)
            n_features = X_train.shape[1]

            all_predictions = []
            all_importances = []
            model_aucs = []

            for k in range(K):
                rng = np.random.RandomState(42 + k * 13)
                indices = rng.choice(n_samples, size=int(n_samples * 0.8), replace=True)
                X_boot = X_train[indices]
                y_boot = y_train[indices]

                if len(np.unique(y_boot)) < 2:
                    continue

                try:
                    model = model_factory_fn()
                    model.fit(X_boot, y_boot)

                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(X_test)[:, 1]
                    elif hasattr(model, "decision_function"):
                        proba = model.decision_function(X_test)
                    else:
                        proba = model.predict(X_test).astype(float)
                    all_predictions.append(proba)

                    if hasattr(model, "coef_"):
                        imp = np.abs(model.coef_.ravel())
                    elif hasattr(model, "feature_importances_"):
                        imp = model.feature_importances_
                    else:
                        imp = np.ones(n_features) / n_features

                    if len(imp) == n_features:
                        imp_sum = imp.sum()
                        if imp_sum > 0:
                            imp = imp / imp_sum
                        all_importances.append(imp)

                    try:
                        auc = float(roc_auc_score(y_test, proba))
                    except Exception:
                        auc = 0.5
                    model_aucs.append(auc)
                except Exception:
                    pass

            if len(all_predictions) < 2:
                return {"score": 0.5, "n_models": len(all_predictions),
                        "reason": "too_few_successful_models"}

            pred_matrix = np.array(all_predictions)  # (K, n_test)

            # --- B. Prediction disagreement per sample ---
            pred_std_per_sample = np.std(pred_matrix, axis=0)
            pred_mean_per_sample = np.mean(pred_matrix, axis=0)
            mean_pred_std = float(np.mean(pred_std_per_sample))
            max_pred_std = float(np.max(pred_std_per_sample))

            # Smoothed predictions: uncertainty-weighted average
            model_variances = np.var(pred_matrix, axis=1)
            model_weights = 1.0 / (model_variances + 1e-8)
            model_weights = model_weights / model_weights.sum()
            smoothed_predictions = np.average(pred_matrix, axis=0, weights=model_weights)

            try:
                smoothed_auc = float(roc_auc_score(y_test, smoothed_predictions))
            except Exception:
                smoothed_auc = 0.5

            # --- A. Feature importance disagreement ---
            feature_imp_std = np.zeros(n_features)
            if len(all_importances) >= 2:
                imp_matrix = np.array(all_importances)
                feature_imp_std = np.std(imp_matrix, axis=0)
                feature_imp_mean = np.mean(imp_matrix, axis=0)

                # Penalize high-disagreement features
                disagreement_ratio = feature_imp_std / (feature_imp_mean + 1e-8)
                feature_weights = 1.0 / (1.0 + disagreement_ratio)
                smoothed_importances = feature_imp_mean * feature_weights
                sm_sum = smoothed_importances.sum()
                if sm_sum > 0:
                    smoothed_importances = smoothed_importances / sm_sum

            mean_feature_disagreement = float(np.mean(feature_imp_std))

            # --- Disagreement map ---
            n_high_disagree = int((pred_std_per_sample > 0.15).sum())
            n_moderate_disagree = int(
                ((pred_std_per_sample > 0.05) & (pred_std_per_sample <= 0.15)).sum()
            )
            n_low_disagree = int((pred_std_per_sample <= 0.05).sum())

            # --- Final score ---
            # mean_pred_std 0.0->1.0, 0.20->0.0
            pred_agreement_score = float(max(0.0, min(1.0, 1.0 - mean_pred_std / 0.20)))
            # mean_feature_disagreement 0.0->1.0, 0.05->0.0
            feat_agreement_score = float(max(0.0, min(1.0, 1.0 - mean_feature_disagreement / 0.05)))

            mean_individual_auc = float(np.mean(model_aucs)) if model_aucs else 0.5
            auc_improvement = smoothed_auc - mean_individual_auc

            score = (0.60 * pred_agreement_score
                     + 0.30 * feat_agreement_score
                     + 0.10 * min(1.0, max(0.0, 0.5 + auc_improvement * 10)))
            score = float(max(0.0, min(1.0, score)))

            return {
                "score": score,
                "pred_agreement_score": round(pred_agreement_score, 4),
                "feat_agreement_score": round(feat_agreement_score, 4),
                "mean_pred_std": round(mean_pred_std, 4),
                "max_pred_std": round(max_pred_std, 4),
                "mean_feature_disagreement": round(mean_feature_disagreement, 4),
                "n_models": len(all_predictions),
                "n_high_disagreement_samples": n_high_disagree,
                "n_moderate_disagreement_samples": n_moderate_disagree,
                "n_low_disagreement_samples": n_low_disagree,
                "smoothed_auc": round(smoothed_auc, 4),
                "mean_individual_auc": round(mean_individual_auc, 4),
                "auc_improvement": round(auc_improvement, 4),
                "model_aucs": [round(a, 4) for a in model_aucs],
                "top_disagreement_features": [
                    int(i) for i in np.argsort(feature_imp_std)[::-1][:5]
                ] if len(all_importances) >= 2 else [],
            }
        except Exception as e:
            logger.warning(f"  Disagreement smoothing failed: {e}")
            return {"score": 0.5, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════════════════
    # 13. SHAP Consistency (optional — requires shap package)
    # ═══════════════════════════════════════════════════════════════════════════

    def _shap_consistency(
        self, trained_model: object,
        X_train: np.ndarray, X_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Feature ranking stability via SHAP values across subsets."""
        if not self.enable_shap:
            return {"score": -1.0, "skipped": True, "reason": "shap not available"}

        try:
            # Use a sample to keep computation fast
            n_sample = min(100, len(X_test))
            rng = np.random.RandomState(42)

            # Split test data into 3 subsets for consistency check
            indices = rng.permutation(len(X_test))
            subset_size = n_sample // 3
            if subset_size < 10:
                return {"score": 0.5, "skipped": True, "reason": "too_few_test_samples"}

            subsets = [
                indices[i * subset_size : (i + 1) * subset_size]
                for i in range(3)
            ]

            rankings = []
            for sub_idx in subsets:
                X_sub = X_test[sub_idx]

                # Use appropriate explainer
                try:
                    if hasattr(trained_model, "predict_proba"):
                        explainer = shap.Explainer(
                            trained_model.predict_proba, X_train[:100],
                            algorithm="auto",
                        )
                        shap_values = explainer(X_sub)
                        # For binary classification, take class 1
                        if len(shap_values.shape) == 3:
                            vals = np.abs(shap_values.values[:, :, 1]).mean(axis=0)
                        else:
                            vals = np.abs(shap_values.values).mean(axis=0)
                    else:
                        explainer = shap.Explainer(
                            trained_model, X_train[:100],
                        )
                        shap_values = explainer(X_sub)
                        vals = np.abs(shap_values.values).mean(axis=0)

                    # Rank features by mean |SHAP|
                    ranking = np.argsort(vals)[::-1]
                    rankings.append(ranking)
                except Exception:
                    pass

            if len(rankings) < 2:
                return {"score": 0.5, "n_subsets": len(rankings)}

            # Compute pairwise Kendall tau on top-10 feature rankings
            top_k = min(10, len(rankings[0]))
            taus = []
            for i in range(len(rankings)):
                for j in range(i + 1, len(rankings)):
                    r1 = rankings[i][:top_k]
                    r2 = rankings[j][:top_k]
                    # Compute rank correlation on positions
                    # Create position arrays
                    pos1 = np.zeros(top_k)
                    pos2 = np.zeros(top_k)
                    for rank, feat in enumerate(r1):
                        if feat in r2:
                            pos1[rank] = rank
                            pos2[rank] = float(np.where(r2 == feat)[0][0])
                        else:
                            pos1[rank] = rank
                            pos2[rank] = top_k  # penalty for not appearing

                    tau, _ = sp_stats.kendalltau(pos1, pos2)
                    if not np.isnan(tau):
                        taus.append(float(tau))

            if not taus:
                return {"score": 0.5, "n_subsets": len(rankings)}

            mean_tau = float(np.mean(taus))
            # Map: tau=1.0 → score=1.0, tau=0.0 → score=0.5, tau=-1.0 → score=0.0
            score = float(max(0.0, min(1.0, (mean_tau + 1) / 2)))

            return {
                "score": score,
                "mean_kendall_tau": round(mean_tau, 4),
                "n_subsets": len(rankings),
                "top_k": top_k,
            }
        except Exception as e:
            logger.warning(f"  SHAP consistency failed: {e}")
            return {"score": 0.5, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════════════════
    # 14. Conformal Prediction (optional — requires mapie package)
    # ═══════════════════════════════════════════════════════════════════════════

    def _conformal_prediction(
        self, trained_model: object,
        X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray,
    ) -> Dict[str, Any]:
        """Distribution-free prediction sets via conformal prediction (mapie 1.3+).

        Uses SplitConformalClassifier with LAC conformity score at multiple
        confidence levels [0.80, 0.90, 0.95]. Last 30% of training data is
        held out as calibration set (temporal ordering preserved).
        """
        if not self.enable_conformal:
            return {"score": -1.0, "skipped": True, "reason": "mapie not available"}

        try:
            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                return {"score": 0.5, "skipped": True, "reason": "single_class"}

            confidence_levels = [0.80, 0.90, 0.95]

            # Use last 30% of training data as calibration (preserves temporal order)
            n_cal = max(30, int(len(X_train) * 0.3))
            X_cal = X_train[-n_cal:]
            y_cal = y_train[-n_cal:]

            if len(np.unique(y_cal)) < 2:
                return {"score": 0.5, "skipped": True, "reason": "single_class_cal"}

            # Wrap pipeline models for mapie compatibility
            model_for_conformal = trained_model
            if not hasattr(trained_model, "fit"):
                from sklearn.base import BaseEstimator, ClassifierMixin
                class _PredictOnlyWrapper(ClassifierMixin, BaseEstimator):
                    def __init__(self, pipeline):
                        self._pipeline = pipeline
                        self.classes_ = np.array([0, 1])
                    def fit(self, X, y):
                        return self
                    def predict_proba(self, X):
                        return self._pipeline.predict_proba(X)
                    def predict(self, X):
                        return self._pipeline.predict(X)
                model_for_conformal = _PredictOnlyWrapper(trained_model)

            # Ensure classes_ attribute exists
            if not hasattr(model_for_conformal, "classes_"):
                model_for_conformal.classes_ = np.array([0, 1])

            scc = SplitConformalClassifier(
                estimator=model_for_conformal,
                confidence_level=confidence_levels,
                conformity_score="lac",
                prefit=True,
                random_state=42,
            )
            scc.conformalize(X_cal, y_cal)
            _, y_sets = scc.predict_set(X_test)
            # y_sets shape: (n_samples, n_classes, n_confidence_levels)

            level_results = []
            coverage_deviations = []
            for i, target_cov in enumerate(confidence_levels):
                sets_i = y_sets[:, :, i]
                cov_raw = classification_coverage_score(y_test, sets_i)
                coverage = float(np.asarray(cov_raw).item())
                wid_raw = classification_mean_width_score(sets_i)
                width = float(np.asarray(wid_raw).item())
                deviation = abs(coverage - target_cov)
                coverage_deviations.append(deviation)
                level_results.append({
                    "target": target_cov,
                    "coverage": round(coverage, 4),
                    "width": round(width, 4),
                    "deviation": round(deviation, 4),
                })

            mean_deviation = float(np.mean(coverage_deviations))
            mean_width = float(np.mean([lr["width"] for lr in level_results]))

            # coverage_component: 1.0 if deviation=0, 0.0 if deviation>=0.15
            coverage_component = max(0.0, 1.0 - mean_deviation / 0.15)
            # efficiency_component: 1.0 if width=1.0 (singleton), 0.0 if width=2.0
            efficiency_component = max(0.0, min(1.0, 2.0 - mean_width))

            score = 0.60 * coverage_component + 0.40 * efficiency_component
            score = float(max(0.0, min(1.0, score)))

            return {
                "score": score,
                "coverage_component": round(coverage_component, 4),
                "efficiency_component": round(efficiency_component, 4),
                "levels": level_results,
                "mean_deviation": round(mean_deviation, 4),
                "mean_width": round(mean_width, 4),
                "n_cal": n_cal,
                "n_test": len(y_test),
            }
        except Exception as e:
            logger.warning(f"  Conformal prediction failed: {e}")
            return {"score": 0.5, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════════════════
    # 17. FEATURE CAUSALITY SCORING — Granger-like F-test
    # ═══════════════════════════════════════════════════════════════════════════

    def _feature_causality_scoring(
        self, X_train: np.ndarray, y_train: np.ndarray,
        max_features: int = 50, significance: float = 0.05,
    ) -> Dict[str, Any]:
        """
        Granger-like F-test per feature: restricted (y ~ y_lag) vs unrestricted
        (y ~ y_lag + feature_lag). Score = proportion significant * 2.5, capped at 1.0.
        """
        try:
            n_samples, n_features = X_train.shape
            if n_samples < 30:
                return {"score": -1.0, "skipped": True, "reason": "Too few samples"}

            # Subsample features if too many
            feat_indices = list(range(n_features))
            if n_features > max_features:
                rng = np.random.RandomState(42)
                feat_indices = sorted(rng.choice(n_features, max_features, replace=False))

            # Create lagged version of target (shift by 1)
            y_float = y_train.astype(float)
            y_lag = np.roll(y_float, 1)
            y_lag[0] = y_float.mean()  # Fill first lag with mean

            # Restricted model: y ~ y_lag (baseline)
            y_curr = y_float[1:]  # Skip first sample
            y_lag_curr = y_lag[1:].reshape(-1, 1)
            n = len(y_curr)

            # Restricted RSS
            from numpy.linalg import lstsq
            coef_r, rss_r_arr, _, _ = lstsq(
                np.column_stack([y_lag_curr, np.ones(n)]),
                y_curr, rcond=None
            )
            resid_r = y_curr - np.column_stack([y_lag_curr, np.ones(n)]) @ coef_r
            rss_r = float(np.sum(resid_r ** 2))

            n_tested = 0
            n_significant = 0
            p_values = []

            for fi in feat_indices:
                feat_lag = np.roll(X_train[:, fi], 1)
                feat_lag[0] = np.mean(X_train[:, fi])
                feat_lag_curr = feat_lag[1:].reshape(-1, 1)

                # Unrestricted: y ~ y_lag + feat_lag
                X_unr = np.column_stack([y_lag_curr, feat_lag_curr, np.ones(n)])
                try:
                    coef_u, _, _, _ = lstsq(X_unr, y_curr, rcond=None)
                    resid_u = y_curr - X_unr @ coef_u
                    rss_u = float(np.sum(resid_u ** 2))
                except Exception:
                    continue

                # F-statistic: ((RSS_r - RSS_u) / q) / (RSS_u / (n - k))
                q = 1  # one additional regressor
                k = 3  # intercept + y_lag + feature_lag
                if rss_u <= 0 or n <= k:
                    continue

                f_stat = ((rss_r - rss_u) / q) / (rss_u / (n - k))
                if f_stat < 0:
                    f_stat = 0.0

                p_val = 1.0 - sp_stats.f.cdf(f_stat, q, n - k)
                p_values.append(p_val)
                n_tested += 1
                if p_val < significance:
                    n_significant += 1

            if n_tested == 0:
                return {"score": -1.0, "skipped": True, "reason": "No features tested"}

            proportion = n_significant / n_tested
            score = min(1.0, proportion * 2.5)

            return {
                "score": round(score, 4),
                "n_tested": n_tested,
                "n_significant": n_significant,
                "proportion_significant": round(proportion, 4),
                "mean_p_value": round(float(np.mean(p_values)), 4),
                "median_p_value": round(float(np.median(p_values)), 4),
            }
        except Exception as e:
            logger.warning(f"  Feature causality scoring failed: {e}")
            return {"score": 0.5, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════════════════
    # 18. PREDICTION INTERVAL COVERAGE — bootstrap prediction intervals
    # ═══════════════════════════════════════════════════════════════════════════

    def _prediction_interval_coverage(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray,
        model_factory_fn: Callable[[], object],
        n_bootstraps: int = 5,
        target_coverage: float = 0.80,
    ) -> Dict[str, Any]:
        """
        Train n bootstrap models, compute 10th/90th percentile prediction bounds.
        Score = 1.0 - |actual_coverage - target| / 0.15.
        """
        try:
            n_train = X_train.shape[0]
            if n_train < 30:
                return {"score": -1.0, "skipped": True, "reason": "Too few samples"}

            rng = np.random.RandomState(42)
            all_predictions = []

            for b in range(n_bootstraps):
                boot_idx = rng.choice(n_train, size=n_train, replace=True)
                X_boot, y_boot = X_train[boot_idx], y_train[boot_idx]

                try:
                    m = model_factory_fn()
                    m.fit(X_boot, y_boot)
                    preds = m.predict_proba(X_test)[:, 1]
                    all_predictions.append(preds)
                except Exception:
                    continue

            if len(all_predictions) < 3:
                return {"score": -1.0, "skipped": True,
                        "reason": f"Only {len(all_predictions)} bootstraps succeeded"}

            pred_array = np.array(all_predictions)  # (n_bootstraps, n_test)
            lo = np.percentile(pred_array, 10, axis=0)
            hi = np.percentile(pred_array, 90, axis=0)
            median_pred = np.median(pred_array, axis=0)

            # Coverage: fraction of test samples where y_test falls in predicted interval
            # For binary classification, "covered" = predicted interval includes the class
            # y=1 covered if hi >= 0.5, y=0 covered if lo <= 0.5
            covered = np.where(
                y_test == 1,
                hi >= 0.5,
                lo <= 0.5
            )
            actual_coverage = float(np.mean(covered))

            # Interval width
            interval_width = float(np.mean(hi - lo))

            # Score: penalize deviation from target
            deviation = abs(actual_coverage - target_coverage)
            score = max(0.0, 1.0 - deviation / 0.15)

            return {
                "score": round(float(score), 4),
                "actual_coverage": round(actual_coverage, 4),
                "target_coverage": target_coverage,
                "mean_interval_width": round(interval_width, 4),
                "n_bootstraps_used": len(all_predictions),
                "undercoverage": actual_coverage < target_coverage,
            }
        except Exception as e:
            logger.warning(f"  Prediction interval coverage failed: {e}")
            return {"score": 0.5, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════════════════
    # 19. DISTRIBUTION-ROBUST SCORING — worst-slice robustness
    # ═══════════════════════════════════════════════════════════════════════════

    def _distribution_robust_scoring(
        self,
        X_test: np.ndarray, y_test: np.ndarray,
        trained_model: object, predictions: np.ndarray,
        n_slices: int = 5,
    ) -> Dict[str, Any]:
        """
        Split test set into temporal slices, find worst AUC.
        Score = (worst_auc/mean_auc - 0.70) / 0.30, capped to [0, 1].
        """
        try:
            n_test = len(y_test)
            if n_test < 50:
                return {"score": -1.0, "skipped": True, "reason": "Too few test samples"}

            # Split into temporal slices
            slice_size = n_test // n_slices
            if slice_size < 10:
                n_slices = max(2, n_test // 10)
                slice_size = n_test // n_slices

            slice_aucs = []
            slice_details = []

            for s in range(n_slices):
                start = s * slice_size
                end = start + slice_size if s < n_slices - 1 else n_test
                y_slice = y_test[start:end]
                pred_slice = predictions[start:end]

                # Need both classes for AUC
                if len(np.unique(y_slice)) < 2:
                    continue

                try:
                    s_auc = float(roc_auc_score(y_slice, pred_slice))
                    slice_aucs.append(s_auc)
                    slice_details.append({
                        "slice": s,
                        "start": start,
                        "end": end,
                        "n_samples": end - start,
                        "auc": round(s_auc, 4),
                    })
                except Exception:
                    continue

            if len(slice_aucs) < 2:
                return {"score": -1.0, "skipped": True,
                        "reason": f"Only {len(slice_aucs)} valid slices"}

            worst_auc = min(slice_aucs)
            mean_auc = float(np.mean(slice_aucs))
            robustness_ratio = worst_auc / mean_auc if mean_auc > 0 else 0.0

            # Score: (ratio - 0.70) / 0.30, capped [0, 1]
            score = max(0.0, min(1.0, (robustness_ratio - 0.70) / 0.30))

            return {
                "score": round(float(score), 4),
                "worst_auc": round(worst_auc, 4),
                "mean_auc": round(mean_auc, 4),
                "robustness_ratio": round(robustness_ratio, 4),
                "n_slices": len(slice_aucs),
                "slice_details": slice_details,
            }
        except Exception as e:
            logger.warning(f"  Distribution-robust scoring failed: {e}")
            return {"score": 0.5, "error": str(e)}

    # ═══════════════════════════════════════════════════════════════════════════
    # COMPOSITE SCORE
    # ═══════════════════════════════════════════════════════════════════════════

    def _compute_composite(self, results: Dict[str, Any]) -> float:
        """
        Weighted average of all method scores.

        Skipped methods (score == -1.0) have their weight redistributed
        proportionally among active methods.
        """
        active_scores = {}
        active_weights = {}

        for method, weight in self.weights.items():
            if method not in results:
                continue
            r = results[method]
            if not isinstance(r, dict):
                continue
            score = r.get("score", -1.0)
            if score >= 0:
                active_scores[method] = score
                active_weights[method] = weight

        if not active_scores:
            return 0.0

        # Renormalize weights to sum to 1.0
        total_weight = sum(active_weights.values())
        if total_weight <= 0:
            return 0.0

        composite = sum(
            active_scores[m] * (active_weights[m] / total_weight)
            for m in active_scores
        )
        return float(min(1.0, max(0.0, composite)))
