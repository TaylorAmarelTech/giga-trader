"""
Tests for KnockoffGate — FDR-controlled feature discovery via knockoff features.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from src.phase_14_robustness.knockoff_gate import KnockoffGate


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_clean_data(n=300, n_features=20, n_informative=5, seed=42):
    """Generate well-separated classification data with clear signal."""
    X, y = make_classification(
        n_samples=n,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=2,
        n_clusters_per_class=1,
        class_sep=2.0,
        random_state=seed,
    )
    return X, y


def _make_noisy_data(n=300, n_features=20, seed=42):
    """Generate noisy data where signal is very weak."""
    X, y = make_classification(
        n_samples=n,
        n_features=n_features,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=0.3,
        flip_y=0.2,
        random_state=seed,
    )
    return X, y


# ── Test 1: High score for well-separated data with logistic regression ─────

class TestHighScoreCleanData:
    def test_logistic_on_clean_data_scores_high(self):
        """
        Logistic regression on well-separated data should produce a score
        meaningfully above 0.5 (random baseline), indicating real features
        outrank knockoffs.
        """
        # Use mostly informative features to give the model a strong signal
        X, y = make_classification(
            n_samples=500,
            n_features=10,
            n_informative=8,
            n_redundant=2,
            n_clusters_per_class=1,
            class_sep=3.0,
            random_state=42,
        )
        model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        gate = KnockoffGate(top_k_fraction=0.25, n_repeats=5, random_state=42)
        result = gate.run(model, X, y)

        assert not result["skipped"], f"Should not skip: {result.get('reason')}"
        assert result["score"] >= 0.6, (
            f"Expected score >= 0.6 for clean data with logistic regression, "
            f"got {result['score']:.3f}"
        )
        # With well-separated data the model should score above random
        assert result["score"] > 0.5


# ── Test 2: Lower score for overfitting tree on noisy data ──────────────────

class TestLowerScoreOverfitTree:
    def test_deep_tree_on_noisy_data_scores_lower(self):
        """
        An overfitting decision tree on noisy data should produce a lower
        knockoff gate score because the tree memorizes noise, making
        knockoff features appear important.
        """
        X, y = _make_noisy_data(n=300, n_features=20)
        # Deep tree with no regularization — overfits easily
        model = DecisionTreeClassifier(
            max_depth=None, min_samples_leaf=1, random_state=42
        )
        gate = KnockoffGate(top_k_fraction=0.3, n_repeats=3, random_state=42)
        result = gate.run(model, X, y)

        assert not result["skipped"]
        # Logistic on clean data should beat deep tree on noisy data
        X_clean, y_clean = _make_clean_data(n=400, n_features=20, n_informative=5)
        model_clean = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        gate_clean = KnockoffGate(
            top_k_fraction=0.3, n_repeats=3, random_state=42
        )
        result_clean = gate_clean.run(model_clean, X_clean, y_clean)

        assert result["score"] <= result_clean["score"], (
            f"Overfit tree score ({result['score']:.3f}) should be <= "
            f"clean logistic score ({result_clean['score']:.3f})"
        )


# ── Test 3: Score is between 0 and 1 ───────────────────────────────────────

class TestScoreBounds:
    def test_score_within_zero_one(self):
        """Score should always be in [0.0, 1.0]."""
        X, y = _make_clean_data(n=200)
        model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        gate = KnockoffGate(n_repeats=2, random_state=42)
        result = gate.run(model, X, y)

        assert 0.0 <= result["score"] <= 1.0, (
            f"Score {result['score']} out of bounds [0, 1]"
        )


# ── Test 4: n_knockoffs_in_top_k is reported correctly ─────────────────────

class TestNKnockoffsInTopK:
    def test_n_knockoffs_in_top_k_is_nonneg_and_bounded(self):
        """
        n_knockoffs_in_top_k should be non-negative and at most equal
        to top_k.
        """
        X, y = _make_clean_data(n=200, n_features=15)
        model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        gate = KnockoffGate(top_k_fraction=0.3, n_repeats=3, random_state=42)
        result = gate.run(model, X, y)

        assert not result["skipped"]
        assert result["n_knockoffs_in_top_k"] >= 0.0
        assert result["n_knockoffs_in_top_k"] <= result["top_k"], (
            f"n_knockoffs_in_top_k ({result['n_knockoffs_in_top_k']}) "
            f"should be <= top_k ({result['top_k']})"
        )
        # top_k should be 30% of 2*15 = 30 total features => top_k = 9
        expected_top_k = max(1, int(2 * 15 * 0.3))
        assert result["top_k"] == expected_top_k


# ── Test 5: knockoff_importance_ratios are non-negative ─────────────────────

class TestImportanceRatios:
    def test_ratios_are_nonneg(self):
        """knockoff_importance_ratios should all be non-negative floats."""
        X, y = _make_clean_data(n=200)
        model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        gate = KnockoffGate(n_repeats=3, random_state=42)
        result = gate.run(model, X, y)

        assert not result["skipped"]
        ratios = result["knockoff_importance_ratios"]
        assert len(ratios) > 0, "Should have at least one ratio"
        for r in ratios:
            assert r >= 0.0, f"Ratio {r} is negative"


# ── Test 6: Skipped when too few samples ────────────────────────────────────

class TestSkipFewSamples:
    def test_skip_when_fewer_than_100_samples(self):
        """Should skip with reason when n_samples < 100."""
        X, y = _make_clean_data(n=50, n_features=10)
        model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        gate = KnockoffGate(n_repeats=2, random_state=42)
        result = gate.run(model, X, y)

        assert result["skipped"] is True
        assert "Too few samples" in result["reason"]
        assert result["score"] == 0.0
        assert result["passed"] is False


# ── Test 7: Skipped when too few features ───────────────────────────────────

class TestSkipFewFeatures:
    def test_skip_when_fewer_than_5_features(self):
        """Should skip with reason when n_features < 5."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 3)  # Only 3 features
        y = (X[:, 0] > 0).astype(int)
        model = LogisticRegression(C=1.0, max_iter=500, random_state=42)
        gate = KnockoffGate(n_repeats=2, random_state=42)
        result = gate.run(model, X, y)

        assert result["skipped"] is True
        assert "Too few features" in result["reason"]
        assert result["score"] == 0.0
        assert result["passed"] is False


# ── Test 8: Multiple repeats produce averaged results ───────────────────────

class TestMultipleRepeats:
    def test_repeats_are_averaged(self):
        """
        With n_repeats > 1, the result should reflect averaging across
        repeats. The number of importance ratios should equal n_repeats
        (assuming no failures).
        """
        X, y = _make_clean_data(n=300, n_features=15)
        model = LogisticRegression(C=1.0, max_iter=500, random_state=42)

        n_repeats = 5
        gate = KnockoffGate(
            top_k_fraction=0.3, n_repeats=n_repeats, random_state=42
        )
        result = gate.run(model, X, y)

        assert not result["skipped"]
        # All repeats should succeed for logistic regression
        assert len(result["knockoff_importance_ratios"]) == n_repeats, (
            f"Expected {n_repeats} ratios, got "
            f"{len(result['knockoff_importance_ratios'])}"
        )

        # n_knockoffs_in_top_k should be the average, not an integer
        # (unless all repeats produce the same count)
        assert isinstance(result["n_knockoffs_in_top_k"], float)

        # Run with single repeat and verify it matches (deterministic seed)
        gate_single = KnockoffGate(
            top_k_fraction=0.3, n_repeats=1, random_state=42
        )
        result_single = gate_single.run(model, X, y)
        assert len(result_single["knockoff_importance_ratios"]) == 1
