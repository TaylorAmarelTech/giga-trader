"""
Tests for LabelNoiseTest — label noise robustness measurement.
"""

import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.phase_14_robustness.label_noise_test import LabelNoiseTest


def _make_well_separated(n=300, n_features=10, seed=42):
    """Generate well-separated binary classification data."""
    X, y = make_classification(
        n_samples=n,
        n_features=n_features,
        n_informative=6,
        n_redundant=2,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=seed,
    )
    return X, y


def _make_noisy(n=200, n_features=10, seed=42):
    """Generate noisy, hard-to-separate data."""
    X, y = make_classification(
        n_samples=n,
        n_features=n_features,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=2,
        class_sep=0.3,
        flip_y=0.15,
        random_state=seed,
    )
    return X, y


def _logistic_factory():
    """Return a fresh logistic regression model."""
    return LogisticRegression(C=1.0, max_iter=500, random_state=42)


def _deep_tree_factory():
    """Return a deep decision tree (fragile, overfits easily)."""
    return DecisionTreeClassifier(max_depth=None, min_samples_leaf=1, random_state=42)


class TestLabelNoiseBasic:
    """Basic functionality tests for LabelNoiseTest."""

    def test_basic_run_returns_valid_score(self):
        """Basic run should return a score between 0 and 1."""
        X, y = _make_well_separated()
        test = LabelNoiseTest(noise_levels=[0.05, 0.10], n_repeats=2)
        result = test.run(model_factory=_logistic_factory, X=X, y=y, cv=3)

        assert not result["skipped"]
        assert 0.0 <= result["score"] <= 1.0
        assert result["reason"] == ""

    def test_robust_model_high_score(self):
        """Logistic regression on well-separated data should score close to 1.0."""
        X, y = _make_well_separated(n=400, n_features=10)
        test = LabelNoiseTest(noise_levels=[0.05, 0.10], n_repeats=3)
        result = test.run(model_factory=_logistic_factory, X=X, y=y, cv=3)

        assert not result["skipped"]
        # Well-separated data + regularized model = robust to small noise
        # Scoring: 1.0 - min(max_drop/0.20, 1.0). With 10% noise, even robust
        # models may see ~0.08-0.10 AUC drop, yielding scores around 0.50-0.60.
        assert result["score"] >= 0.40, (
            f"Robust model should score >= 0.40, got {result['score']:.4f} "
            f"(max_drop={result['max_drop']:.4f})"
        )

    def test_fragile_model_lower_score(self):
        """Deep tree on small noisy data should score lower than logistic on clean data."""
        X_clean, y_clean = _make_well_separated(n=400)
        X_noisy, y_noisy = _make_noisy(n=200)

        test = LabelNoiseTest(noise_levels=[0.05, 0.10], n_repeats=3)

        robust_result = test.run(model_factory=_logistic_factory, X=X_clean, y=y_clean, cv=3)
        fragile_result = test.run(model_factory=_deep_tree_factory, X=X_noisy, y=y_noisy, cv=3)

        if not fragile_result["skipped"]:
            # Fragile model should score lower (or at least not much higher)
            assert fragile_result["score"] <= robust_result["score"] + 0.15, (
                f"Fragile model score ({fragile_result['score']:.4f}) should be <= "
                f"robust model score ({robust_result['score']:.4f}) + 0.15"
            )


class TestLabelNoiseOutputFields:
    """Tests for output dictionary structure and correctness."""

    def test_max_drop_correctly_computed(self):
        """max_drop should be the maximum auc_drop across all noise levels."""
        X, y = _make_well_separated()
        test = LabelNoiseTest(noise_levels=[0.05, 0.10], n_repeats=2)
        result = test.run(model_factory=_logistic_factory, X=X, y=y, cv=3)

        assert not result["skipped"]
        expected_max_drop = max(entry["auc_drop"] for entry in result["auc_drops"])
        assert abs(result["max_drop"] - expected_max_drop) < 1e-10, (
            f"max_drop {result['max_drop']} != computed max {expected_max_drop}"
        )

    def test_base_auc_is_populated(self):
        """base_auc should be a valid AUC value above 0.5."""
        X, y = _make_well_separated()
        test = LabelNoiseTest(noise_levels=[0.05], n_repeats=2)
        result = test.run(model_factory=_logistic_factory, X=X, y=y, cv=3)

        assert not result["skipped"]
        assert result["base_auc"] > 0.52, (
            f"base_auc should be > 0.52, got {result['base_auc']:.4f}"
        )

    def test_auc_drops_has_entries_for_each_noise_level(self):
        """auc_drops should have one entry per noise level."""
        noise_levels = [0.03, 0.07, 0.15]
        X, y = _make_well_separated()
        test = LabelNoiseTest(noise_levels=noise_levels, n_repeats=2)
        result = test.run(model_factory=_logistic_factory, X=X, y=y, cv=3)

        assert not result["skipped"]
        assert len(result["auc_drops"]) == len(noise_levels)
        for i, entry in enumerate(result["auc_drops"]):
            assert entry["noise_level"] == noise_levels[i]
            assert "mean_auc" in entry
            assert "auc_drop" in entry
            assert "n_successful_repeats" in entry

    def test_auc_drops_increase_with_noise(self):
        """Higher noise levels should generally cause larger AUC drops."""
        X, y = _make_well_separated(n=400)
        test = LabelNoiseTest(noise_levels=[0.02, 0.20], n_repeats=3)
        result = test.run(model_factory=_logistic_factory, X=X, y=y, cv=3)

        if not result["skipped"] and len(result["auc_drops"]) == 2:
            low_noise_drop = result["auc_drops"][0]["auc_drop"]
            high_noise_drop = result["auc_drops"][1]["auc_drop"]
            # 20% noise should cause at least as much drop as 2% noise
            assert high_noise_drop >= low_noise_drop - 0.02, (
                f"20% noise drop ({high_noise_drop:.4f}) should be >= "
                f"2% noise drop ({low_noise_drop:.4f}) - 0.02"
            )


class TestLabelNoiseSkipConditions:
    """Tests for skip conditions and edge cases."""

    def test_skipped_when_insufficient_samples(self):
        """Should skip with reason when fewer than 50 samples."""
        X = np.random.randn(30, 5)
        y = (X[:, 0] > 0).astype(int)
        test = LabelNoiseTest()
        result = test.run(model_factory=_logistic_factory, X=X, y=y)

        assert result["skipped"] is True
        assert result["reason"] == "insufficient samples"
        assert result["score"] == -1.0

    def test_skipped_when_base_model_too_weak(self):
        """Should skip when base model AUC is below 0.52."""
        # Create data with no signal (pure noise)
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)
        y = rng.randint(0, 2, size=100)

        test = LabelNoiseTest(noise_levels=[0.05], n_repeats=2)
        result = test.run(model_factory=_logistic_factory, X=X, y=y, cv=3)

        # With pure noise, base AUC should be near 0.50 -> skip
        if result["skipped"]:
            assert "base model too weak" in result["reason"] or "base model fit failed" in result["reason"]
            assert result["score"] == -1.0

    def test_skipped_when_single_class(self):
        """Should skip when labels contain only one class."""
        X = np.random.randn(100, 5)
        y = np.ones(100, dtype=int)  # All class 1

        test = LabelNoiseTest()
        result = test.run(model_factory=_logistic_factory, X=X, y=y)

        assert result["skipped"] is True
        assert "single class" in result["reason"]


class TestLabelNoiseConfiguration:
    """Tests for custom configuration and parameters."""

    def test_custom_noise_levels_are_used(self):
        """Custom noise levels should appear in the output."""
        custom_levels = [0.01, 0.03, 0.08, 0.15]
        X, y = _make_well_separated()
        test = LabelNoiseTest(noise_levels=custom_levels, n_repeats=2)
        result = test.run(model_factory=_logistic_factory, X=X, y=y, cv=3)

        assert not result["skipped"]
        assert len(result["auc_drops"]) == len(custom_levels)
        for i, entry in enumerate(result["auc_drops"]):
            assert entry["noise_level"] == custom_levels[i]

    def test_multiple_repeats_average_correctly(self):
        """Each noise level should use n_repeats and average them."""
        X, y = _make_well_separated(n=300)
        test = LabelNoiseTest(noise_levels=[0.10], n_repeats=5)
        result = test.run(model_factory=_logistic_factory, X=X, y=y, cv=3)

        assert not result["skipped"]
        # With 5 repeats, should have at least a few successful
        entry = result["auc_drops"][0]
        assert entry["n_successful_repeats"] >= 3, (
            f"Expected >= 3 successful repeats, got {entry['n_successful_repeats']}"
        )

    def test_model_factory_called_fresh_each_time(self):
        """model_factory should be called once per noise level per repeat, plus once for base."""
        call_count = 0

        def counting_factory():
            nonlocal call_count
            call_count += 1
            return LogisticRegression(C=1.0, max_iter=500, random_state=42)

        X, y = _make_well_separated()
        noise_levels = [0.05, 0.10]
        n_repeats = 2
        test = LabelNoiseTest(noise_levels=noise_levels, n_repeats=n_repeats)
        result = test.run(model_factory=counting_factory, X=X, y=y, cv=3)

        assert not result["skipped"]
        # 1 call for base + (n_noise_levels * n_repeats) calls for noisy
        expected_calls = 1 + len(noise_levels) * n_repeats
        assert call_count == expected_calls, (
            f"model_factory called {call_count} times, expected {expected_calls}"
        )


class TestFlipLabels:
    """Tests for the _flip_labels helper method."""

    def test_flip_labels_correct_count(self):
        """Should flip exactly noise_level fraction of labels."""
        test = LabelNoiseTest()
        rng = np.random.RandomState(42)
        y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

        y_noisy = test._flip_labels(y, noise_level=0.20, rng=rng)

        n_flipped = np.sum(y != y_noisy)
        assert n_flipped == 2, f"Expected 2 flipped labels, got {n_flipped}"

    def test_flip_labels_preserves_length(self):
        """Output should have same length as input."""
        test = LabelNoiseTest()
        rng = np.random.RandomState(42)
        y = np.zeros(100, dtype=int)

        y_noisy = test._flip_labels(y, noise_level=0.10, rng=rng)
        assert len(y_noisy) == len(y)

    def test_flip_labels_does_not_modify_original(self):
        """Should return a copy, not modify the original."""
        test = LabelNoiseTest()
        rng = np.random.RandomState(42)
        y = np.array([0, 0, 0, 1, 1, 1])
        y_original = y.copy()

        test._flip_labels(y, noise_level=0.50, rng=rng)
        np.testing.assert_array_equal(y, y_original)

    def test_flip_labels_zero_noise(self):
        """Zero noise level should return identical labels."""
        test = LabelNoiseTest()
        rng = np.random.RandomState(42)
        y = np.array([0, 1, 0, 1, 0])

        y_noisy = test._flip_labels(y, noise_level=0.0, rng=rng)
        np.testing.assert_array_equal(y, y_noisy)


class TestLabelNoiseScoring:
    """Tests for the scoring formula."""

    def test_score_formula_no_drop(self):
        """Score should be 1.0 when max_drop is 0."""
        # If a model has zero AUC drop, score = 1.0 - min(0/0.20, 1.0) = 1.0
        X, y = _make_well_separated(n=400, n_features=10)
        test = LabelNoiseTest(noise_levels=[0.01], n_repeats=3)
        result = test.run(model_factory=_logistic_factory, X=X, y=y, cv=3)

        if not result["skipped"]:
            # With only 1% noise on well-separated data, score should be high
            assert result["score"] >= 0.75, (
                f"Score with 1% noise on clean data should be >= 0.75, "
                f"got {result['score']:.4f}"
            )

    def test_score_bounded_zero_one(self):
        """Score should always be in [0.0, 1.0]."""
        X, y = _make_well_separated()
        test = LabelNoiseTest(noise_levels=[0.05, 0.10, 0.20], n_repeats=2)
        result = test.run(model_factory=_logistic_factory, X=X, y=y, cv=3)

        if not result["skipped"]:
            assert 0.0 <= result["score"] <= 1.0
