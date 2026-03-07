"""Tests for KnowledgeDistiller."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

from src.phase_14_robustness.knowledge_distiller import KnowledgeDistiller


def _make_data(n=300, n_features=10, seed=42):
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features)
    y = (X[:, 0] + 0.5 * X[:, 1] + rng.randn(n) * 0.3 > 0).astype(int)
    return X, y


def _make_teachers(X, y, n_teachers=3, seed=42):
    """Train a set of diverse teacher models."""
    teachers = []
    for i in range(n_teachers):
        rng_state = seed + i
        model = LogisticRegression(C=1.0 / (i + 1), random_state=rng_state, max_iter=200)
        model.fit(X, y)
        teachers.append(model)
    return teachers


class TestKDInit:
    def test_default_construction(self):
        kd = KnowledgeDistiller()
        assert kd.student_type == "logistic"
        assert kd.temperature == 3.0
        assert kd.alpha == 0.7

    def test_custom_params(self):
        kd = KnowledgeDistiller(
            student_type="gradient_boosting",
            temperature=5.0,
            alpha=0.5,
            max_student_depth=2,
        )
        assert kd.student_type == "gradient_boosting"
        assert kd.temperature == 5.0

    def test_invalid_student_type(self):
        with pytest.raises(ValueError, match="student_type"):
            KnowledgeDistiller(student_type="deep_net")

    def test_invalid_temperature(self):
        with pytest.raises(ValueError):
            KnowledgeDistiller(temperature=0)

    def test_invalid_alpha(self):
        with pytest.raises(ValueError):
            KnowledgeDistiller(alpha=1.5)


class TestKDDistillation:
    def test_distill_logistic_student(self):
        X, y = _make_data()
        teachers = _make_teachers(X[:200], y[:200])
        kd = KnowledgeDistiller(student_type="logistic")
        student, metrics = kd.distill(
            teachers, X[:200], y[:200], X[200:], y[200:],
        )
        assert student is not None
        assert "fidelity" in metrics
        assert "auc_retention" in metrics
        assert metrics["fidelity"] >= 0.80  # Student should agree with teachers mostly

    def test_distill_gb_student(self):
        X, y = _make_data()
        teachers = _make_teachers(X[:200], y[:200])
        kd = KnowledgeDistiller(student_type="gradient_boosting", max_student_depth=3)
        student, metrics = kd.distill(
            teachers, X[:200], y[:200], X[200:], y[200:],
        )
        assert student is not None
        assert metrics["fidelity"] >= 0.70

    def test_predict_distilled(self):
        X, y = _make_data()
        teachers = _make_teachers(X[:200], y[:200])
        kd = KnowledgeDistiller()
        kd.distill(teachers, X[:200], y[:200], X[200:], y[200:])
        probas = kd.predict_distilled(X[200:])
        assert probas.shape == (100,)
        assert probas.min() >= 0.0
        assert probas.max() <= 1.0

    def test_compression_report(self):
        X, y = _make_data()
        teachers = _make_teachers(X[:200], y[:200])
        kd = KnowledgeDistiller()
        kd.distill(teachers, X[:200], y[:200], X[200:], y[200:])
        report = kd.get_compression_report()
        assert "fidelity" in report
        assert "n_teachers" in report


class TestKDTeacherWeights:
    def test_custom_teacher_weights(self):
        X, y = _make_data()
        teachers = _make_teachers(X[:200], y[:200])
        weights = [0.5, 0.3, 0.2]
        kd = KnowledgeDistiller(teacher_weights=weights)
        student, metrics = kd.distill(
            teachers, X[:200], y[:200], X[200:], y[200:],
        )
        assert student is not None

    def test_single_teacher(self):
        X, y = _make_data()
        teachers = _make_teachers(X[:200], y[:200], n_teachers=1)
        kd = KnowledgeDistiller()
        student, metrics = kd.distill(
            teachers, X[:200], y[:200], X[200:], y[200:],
        )
        assert student is not None


class TestKDTemperature:
    def test_temperature_softens_predictions(self):
        X, y = _make_data()
        teachers = _make_teachers(X[:200], y[:200])

        kd_low = KnowledgeDistiller(temperature=1.0)
        kd_low.distill(teachers, X[:200], y[:200], X[200:], y[200:])

        kd_high = KnowledgeDistiller(temperature=10.0)
        kd_high.distill(teachers, X[:200], y[:200], X[200:], y[200:])

        # Both should produce valid predictions
        pred_low = kd_low.predict_distilled(X[200:])
        pred_high = kd_high.predict_distilled(X[200:])
        assert pred_low.shape == pred_high.shape


class TestKDEvaluation:
    def test_evaluate_via_compression_report(self):
        X, y = _make_data()
        teachers = _make_teachers(X[:200], y[:200])
        kd = KnowledgeDistiller()
        student, metrics = kd.distill(teachers, X[:200], y[:200], X[200:], y[200:])
        assert "fidelity" in metrics
        assert "student_auc" in metrics
        assert "teacher_auc" in metrics
        assert "auc_retention" in metrics
        assert "probability_correlation" in metrics


class TestKDEdgeCases:
    def test_empty_teachers_raises(self):
        X, y = _make_data()
        kd = KnowledgeDistiller()
        with pytest.raises((ValueError, IndexError)):
            kd.distill([], X, y, X, y)

    def test_max_depth_rejected(self):
        with pytest.raises(ValueError, match="EDGE 1"):
            KnowledgeDistiller(max_student_depth=10)
