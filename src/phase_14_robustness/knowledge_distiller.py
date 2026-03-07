"""
GIGA TRADER - Knowledge Distillation for Model Compression
===========================================================
Compresses a teacher ensemble of sklearn classifiers into a single
lightweight student model suitable for production inference.

The distillation process trains the student to mimic the teacher
ensemble's soft probability outputs, blending soft-target regression
with hard-label classification to preserve both confidence calibration
and decision-boundary accuracy.

Algorithm:
  1. Collect soft predictions (predict_proba) from each teacher
  2. Compute weighted-average ensemble soft targets
  3. Apply temperature scaling to soften probability distributions
  4. Train a classification student on hard labels, weighted by
     teacher confidence (samples the teacher is sure about matter more)
  5. Train a regression student on soft targets directly
  6. Final prediction blends classification and regression outputs

Constraints (per EDGE 1):
  - Student must be simpler than the teacher ensemble
  - Gradient boosting student max_depth <= 3
  - Distillation fails if fidelity < 0.95

Usage:
    from src.phase_14_robustness.knowledge_distiller import KnowledgeDistiller

    distiller = KnowledgeDistiller(student_type="logistic", temperature=3.0)
    student, metrics = distiller.distill(
        teachers=fitted_models,
        X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val,
    )
    report = distiller.get_compression_report()
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_MIN_FIDELITY = 0.95
_SUPPORTED_STUDENT_TYPES = ("logistic", "gradient_boosting", "ridge")


class KnowledgeDistiller:
    """Compress an ensemble of teacher classifiers into one student model.

    Parameters
    ----------
    student_type : str
        Student architecture: ``"logistic"`` (default), ``"gradient_boosting"``,
        or ``"ridge"``.
    temperature : float
        Softmax temperature applied to teacher logits. Higher values produce
        softer probability distributions (default 3.0).
    alpha : float
        Weight for soft-target loss vs hard-target loss (default 0.7).
        ``loss = alpha * soft_loss + (1 - alpha) * hard_loss``.
    teacher_weights : list[float] or None
        Per-teacher weighting for the ensemble average. If *None*, teachers
        are weighted equally.
    max_student_depth : int
        Maximum tree depth for gradient-boosting student (default 3,
        per EDGE 1 — NEVER > 5).
    """

    def __init__(
        self,
        student_type: str = "logistic",
        temperature: float = 3.0,
        alpha: float = 0.7,
        teacher_weights: Optional[List[float]] = None,
        max_student_depth: int = 3,
    ) -> None:
        if student_type not in _SUPPORTED_STUDENT_TYPES:
            raise ValueError(
                f"student_type must be one of {_SUPPORTED_STUDENT_TYPES}, "
                f"got '{student_type}'"
            )
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if max_student_depth > 5:
            raise ValueError(
                f"max_student_depth must be <= 5 (EDGE 1), got {max_student_depth}"
            )

        self.student_type = student_type
        self.temperature = temperature
        self.alpha = alpha
        self.teacher_weights = teacher_weights
        self.max_student_depth = max_student_depth

        # Populated after distill()
        self._student_clf: Optional[Any] = None
        self._student_reg: Optional[Any] = None
        self._metrics: Optional[Dict] = None
        self._n_teachers: int = 0
        self._teacher_param_count: int = 0
        self._student_param_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def distill(
        self,
        teachers: List[Any],
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> Tuple[Any, Dict]:
        """Run the full distillation pipeline.

        Parameters
        ----------
        teachers : list
            Fitted sklearn classifiers that expose ``predict_proba``.
        X_train, y_train : array-like
            Training data and binary labels.
        X_val, y_val : array-like
            Validation data and binary labels.

        Returns
        -------
        student : fitted sklearn estimator
            The classification student (blends clf + reg internally via
            :meth:`predict_distilled`).
        metrics : dict
            Distillation quality metrics including fidelity, AUC retention,
            size reduction, and probability correlation.

        Raises
        ------
        ValueError
            If no teachers are provided or none expose ``predict_proba``.
        RuntimeError
            If fidelity on the validation set falls below 0.95.
        """
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train).ravel()
        X_val = np.asarray(X_val, dtype=np.float64)
        y_val = np.asarray(y_val).ravel()

        if len(teachers) == 0:
            raise ValueError("At least one teacher model is required")

        # Validate teachers expose predict_proba
        valid_teachers: List[Any] = []
        for i, t in enumerate(teachers):
            if hasattr(t, "predict_proba"):
                valid_teachers.append(t)
            else:
                logger.warning("Teacher %d has no predict_proba, skipping", i)
        if len(valid_teachers) == 0:
            raise ValueError("No teacher with predict_proba found")

        self._n_teachers = len(valid_teachers)
        logger.info(
            "Knowledge distillation: %d teachers, student=%s, T=%.1f, alpha=%.2f",
            self._n_teachers, self.student_type, self.temperature, self.alpha,
        )

        # Resolve teacher weights
        weights = self._resolve_teacher_weights(valid_teachers)

        # Step 1-3: Compute soft targets
        soft_train = self._compute_soft_targets(valid_teachers, X_train, weights)
        soft_val = self._compute_soft_targets(valid_teachers, X_val, weights)

        # Step 4-5: Train student (classification + regression heads)
        self._student_clf, self._student_reg = self._train_student(
            X_train, soft_train, y_train,
        )

        # Step 6: Evaluate
        metrics = self.evaluate(
            self._student_clf, self._student_reg,
            valid_teachers, weights,
            X_val, y_val, soft_val,
        )
        self._metrics = metrics

        # Param counts for compression report
        self._teacher_param_count = self._count_params_ensemble(valid_teachers)
        self._student_param_count = self._count_params_model(self._student_clf)

        logger.info(
            "Distillation complete: fidelity=%.4f, AUC_retention=%.4f, "
            "size_reduction=%.1fx",
            metrics["fidelity"], metrics["auc_retention"],
            metrics["size_reduction"],
        )

        if metrics["fidelity"] < _MIN_FIDELITY:
            raise RuntimeError(
                f"Distillation failed: fidelity {metrics['fidelity']:.4f} "
                f"< required {_MIN_FIDELITY}. Student cannot replicate "
                f"teacher ensemble adequately."
            )

        return self._student_clf, metrics

    def evaluate(
        self,
        student_clf: Any,
        student_reg: Optional[Any],
        teachers: List[Any],
        teacher_weights: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        soft_test: Optional[np.ndarray] = None,
    ) -> Dict:
        """Compare student against the teacher ensemble.

        Parameters
        ----------
        student_clf : fitted estimator
            Classification student.
        student_reg : fitted estimator or None
            Regression student (soft-target head).
        teachers : list
            Fitted teacher models.
        teacher_weights : array
            Per-teacher weights.
        X_test, y_test : array-like
            Test data and labels.
        soft_test : array or None
            Pre-computed teacher soft targets on X_test.

        Returns
        -------
        dict
            Keys: fidelity, student_auc, teacher_auc, auc_retention,
            probability_correlation, size_reduction.
        """
        X_test = np.asarray(X_test, dtype=np.float64)
        y_test = np.asarray(y_test).ravel()

        # Teacher ensemble predictions
        if soft_test is None:
            soft_test = self._compute_soft_targets(teachers, X_test, teacher_weights)
        teacher_preds = (soft_test >= 0.5).astype(int)

        # Teacher AUC
        try:
            teacher_auc = float(roc_auc_score(y_test, soft_test))
        except ValueError:
            teacher_auc = 0.5

        # Student predictions (blended)
        student_proba = self._predict_blended(student_clf, student_reg, X_test)
        student_preds = (student_proba >= 0.5).astype(int)

        # Student AUC
        try:
            student_auc = float(roc_auc_score(y_test, student_proba))
        except ValueError:
            student_auc = 0.5

        # Fidelity: agreement between student and teacher ensemble labels
        fidelity = float(accuracy_score(teacher_preds, student_preds))

        # AUC retention
        auc_retention = (
            student_auc / teacher_auc if teacher_auc > 0 else 0.0
        )

        # Pearson correlation of probability outputs
        if np.std(soft_test) > 1e-8 and np.std(student_proba) > 1e-8:
            prob_corr = float(np.corrcoef(soft_test, student_proba)[0, 1])
        else:
            prob_corr = 0.0

        # Size reduction (approximate)
        t_params = self._count_params_ensemble(teachers)
        s_params = self._count_params_model(student_clf)
        size_reduction = t_params / max(s_params, 1)

        return {
            "fidelity": fidelity,
            "student_auc": student_auc,
            "teacher_auc": teacher_auc,
            "auc_retention": auc_retention,
            "probability_correlation": prob_corr,
            "size_reduction": size_reduction,
            "student_accuracy": float(accuracy_score(y_test, student_preds)),
            "teacher_accuracy": float(accuracy_score(y_test, teacher_preds)),
            "n_teachers": len(teachers),
            "student_type": self.student_type,
        }

    def get_compression_report(self) -> Dict:
        """Return a summary of the most recent distillation.

        Returns
        -------
        dict
            Compression statistics or a stub if :meth:`distill` has not
            been called yet.
        """
        if self._metrics is None:
            return {"status": "no distillation performed yet"}

        return {
            "status": "complete",
            "n_teachers": self._n_teachers,
            "student_type": self.student_type,
            "temperature": self.temperature,
            "alpha": self.alpha,
            "teacher_approx_params": self._teacher_param_count,
            "student_approx_params": self._student_param_count,
            "size_reduction_factor": (
                self._teacher_param_count / max(self._student_param_count, 1)
            ),
            "fidelity": self._metrics["fidelity"],
            "auc_retention": self._metrics["auc_retention"],
            "probability_correlation": self._metrics["probability_correlation"],
            "student_auc": self._metrics["student_auc"],
            "teacher_auc": self._metrics["teacher_auc"],
        }

    def predict_distilled(self, X: np.ndarray) -> np.ndarray:
        """Produce blended probability predictions from the fitted student.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        proba : ndarray of shape (n_samples,)
            Predicted probability of the positive class.

        Raises
        ------
        RuntimeError
            If :meth:`distill` has not been called.
        """
        if self._student_clf is None:
            raise RuntimeError("No student model — call distill() first")
        return self._predict_blended(self._student_clf, self._student_reg,
                                     np.asarray(X, dtype=np.float64))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _resolve_teacher_weights(
        self, teachers: List[Any],
    ) -> np.ndarray:
        """Return normalised teacher weight vector."""
        n = len(teachers)
        if self.teacher_weights is not None:
            w = np.asarray(self.teacher_weights[:n], dtype=np.float64)
            if len(w) < n:
                # Pad with equal share of remaining weight
                pad = np.full(n - len(w), 1.0 / n)
                w = np.concatenate([w, pad])
        else:
            w = np.ones(n, dtype=np.float64)
        w = w / w.sum()
        return w

    def _compute_soft_targets(
        self,
        teachers: List[Any],
        X: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Weighted-average teacher probabilities with temperature scaling.

        Parameters
        ----------
        teachers : list of fitted classifiers
        X : feature matrix
        weights : per-teacher weights (summing to 1)

        Returns
        -------
        soft : ndarray of shape (n_samples,)
            Ensemble soft probability for the positive class.
        """
        n_samples = X.shape[0]
        soft = np.zeros(n_samples, dtype=np.float64)

        for teacher, w in zip(teachers, weights):
            proba = teacher.predict_proba(X)
            # Handle both binary and multi-class output
            if proba.ndim == 2 and proba.shape[1] >= 2:
                p = proba[:, 1]
            else:
                p = proba.ravel()

            # Temperature scaling: convert to logits, scale, convert back
            p_scaled = self._temperature_scaling(p, self.temperature)
            soft += w * p_scaled

        return soft

    @staticmethod
    def _temperature_scaling(proba: np.ndarray, T: float) -> np.ndarray:
        """Apply temperature scaling to probability vector.

        Converts probabilities to logits, divides by *T*, then applies
        sigmoid to get softened probabilities.

        Parameters
        ----------
        proba : ndarray
            Probabilities in (0, 1).
        T : float
            Temperature (> 0). T = 1 is identity; T > 1 softens.

        Returns
        -------
        ndarray
            Temperature-scaled probabilities.
        """
        # Clip to avoid log(0)
        eps = 1e-7
        p = np.clip(proba, eps, 1.0 - eps)
        logits = np.log(p / (1.0 - p))  # inverse sigmoid
        scaled_logits = logits / T
        return 1.0 / (1.0 + np.exp(-scaled_logits))  # sigmoid

    def _train_student(
        self,
        X: np.ndarray,
        soft_targets: np.ndarray,
        hard_targets: np.ndarray,
    ) -> Tuple[Any, Optional[Any]]:
        """Train classification and regression student heads.

        The classification head learns hard labels weighted by teacher
        confidence.  The regression head learns the soft targets directly.

        Returns
        -------
        (student_clf, student_reg)
        """
        # ---- Classification head ----
        # Weight samples by teacher confidence: samples the ensemble is
        # certain about (soft near 0 or 1) get higher weight.
        confidence = np.abs(soft_targets - 0.5) * 2.0  # 0=uncertain, 1=certain
        # Floor so uncertain samples still contribute
        sample_weight = np.clip(confidence, 0.1, 1.0)

        student_clf = self._make_student_classifier()
        logger.info("Training classification student (%s)...", self.student_type)
        student_clf.fit(X, hard_targets, sample_weight=sample_weight)

        # ---- Regression head (on soft targets) ----
        student_reg: Optional[Any] = None
        try:
            student_reg = Ridge(alpha=1.0)
            logger.info("Training regression student (Ridge on soft targets)...")
            student_reg.fit(X, soft_targets)
        except Exception as e:
            logger.warning("Regression student training failed: %s", e)
            student_reg = None

        return student_clf, student_reg

    def _make_student_classifier(self) -> Any:
        """Instantiate an unfitted student classifier."""
        if self.student_type == "logistic":
            return LogisticRegression(
                C=1.0, max_iter=1000, solver="lbfgs", random_state=42,
            )
        elif self.student_type == "gradient_boosting":
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=self.max_student_depth,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=20,
                random_state=42,
            )
        else:  # ridge — wrap as classifier via threshold
            return LogisticRegression(
                C=0.1, penalty="l2", max_iter=1000,
                solver="lbfgs", random_state=42,
            )

    def _predict_blended(
        self,
        student_clf: Any,
        student_reg: Optional[Any],
        X: np.ndarray,
    ) -> np.ndarray:
        """Blend classification and regression student outputs.

        Final probability = alpha * reg_prediction + (1-alpha) * clf_proba.
        Falls back to clf_proba alone if the regression head is unavailable.
        """
        # Classification head probability
        if hasattr(student_clf, "predict_proba"):
            clf_proba = student_clf.predict_proba(X)
            if clf_proba.ndim == 2 and clf_proba.shape[1] >= 2:
                clf_p = clf_proba[:, 1]
            else:
                clf_p = clf_proba.ravel()
        else:
            clf_p = student_clf.predict(X).astype(np.float64)

        if student_reg is None:
            return clf_p

        # Regression head prediction (clipped to valid probability range)
        reg_p = np.clip(student_reg.predict(X), 0.0, 1.0)

        # Blend: alpha weight on soft (regression) head
        blended = self.alpha * reg_p + (1.0 - self.alpha) * clf_p
        return blended

    # ------------------------------------------------------------------
    # Parameter counting (approximate)
    # ------------------------------------------------------------------

    @staticmethod
    def _count_params_model(model: Any) -> int:
        """Approximate parameter count for a single sklearn model."""
        count = 0
        # Linear models
        if hasattr(model, "coef_"):
            coef = np.asarray(model.coef_)
            count += coef.size
        if hasattr(model, "intercept_"):
            intercept = np.asarray(model.intercept_)
            count += intercept.size
        # Tree-based models
        if hasattr(model, "estimators_"):
            # GradientBoosting stores a 2-D array of DecisionTreeRegressors
            estimators = model.estimators_
            try:
                flat = np.asarray(estimators).ravel()
                for tree in flat:
                    if hasattr(tree, "tree_"):
                        count += tree.tree_.node_count * 3  # threshold, value, feature
            except Exception:
                count += len(estimators) * 50  # rough fallback
        elif hasattr(model, "tree_"):
            count += model.tree_.node_count * 3
        # Minimum baseline
        return max(count, 1)

    @classmethod
    def _count_params_ensemble(cls, teachers: List[Any]) -> int:
        """Sum approximate parameter counts across all teachers."""
        total = sum(cls._count_params_model(t) for t in teachers)
        return max(total, 1)
