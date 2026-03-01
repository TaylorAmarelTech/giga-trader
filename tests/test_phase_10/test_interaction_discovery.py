"""
Tests for InteractionDiscovery.
================================
Comprehensive suite covering discovery, transform, edge cases,
and correctness of product/ratio operations.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase_10_feature_processing.interaction_discovery import InteractionDiscovery


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def rng():
    """Reproducible random state."""
    return np.random.RandomState(42)


@pytest.fixture
def sample_data(rng):
    """
    10-feature DataFrame with 200 samples and a binary target.
    Uses make_classification to ensure some MI between features and target.
    """
    X_arr, y = make_classification(
        n_samples=200,
        n_features=10,
        n_informative=5,
        n_redundant=2,
        n_classes=2,
        random_state=42,
    )
    columns = [f"feat_{i}" for i in range(10)]
    X = pd.DataFrame(X_arr, columns=columns)
    return X, y


@pytest.fixture
def small_data(rng):
    """
    5-feature DataFrame with 100 samples — fewer than top_k threshold.
    """
    X_arr, y = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42,
    )
    columns = [f"f_{i}" for i in range(5)]
    X = pd.DataFrame(X_arr, columns=columns)
    return X, y


# =============================================================================
# TEST DISCOVER
# =============================================================================

class TestDiscover:
    """Tests for the discover() method."""

    def test_discover_returns_list_of_tuples(self, sample_data):
        """discover() returns list of (feat_a, feat_b, operation) tuples."""
        X, y = sample_data
        disc = InteractionDiscovery(
            max_interactions=20,
            mi_threshold_multiplier=0.5,  # low threshold to get results
            random_state=42,
        )
        result = disc.discover(X, y)

        assert isinstance(result, list)
        for item in result:
            assert isinstance(item, tuple)
            assert len(item) == 3
            feat_a, feat_b, op = item
            assert isinstance(feat_a, str)
            assert isinstance(feat_b, str)
            assert op in ("product", "ratio")

    def test_discover_sets_fitted_flag(self, sample_data):
        """After discover(), _fitted should be True."""
        X, y = sample_data
        disc = InteractionDiscovery(random_state=42)
        assert not disc._fitted
        disc.discover(X, y)
        assert disc._fitted

    def test_max_interactions_limits_output(self, sample_data):
        """Number of discovered interactions should not exceed max_interactions."""
        X, y = sample_data
        max_ix = 3
        disc = InteractionDiscovery(
            max_interactions=max_ix,
            mi_threshold_multiplier=0.0,  # accept everything
            random_state=42,
        )
        result = disc.discover(X, y)
        assert len(result) <= max_ix

    def test_interactions_sorted_by_mi_descending(self, sample_data):
        """Discovered interactions should be sorted by MI score descending."""
        X, y = sample_data
        disc = InteractionDiscovery(
            max_interactions=50,
            mi_threshold_multiplier=0.0,  # accept everything
            random_state=42,
        )
        disc.discover(X, y)

        scores = disc.mi_scores
        if len(scores) > 1:
            for i in range(len(scores) - 1):
                assert scores[i] >= scores[i + 1], (
                    f"MI scores not sorted descending at index {i}: "
                    f"{scores[i]} < {scores[i + 1]}"
                )

    def test_discover_no_duplicate_pairs(self, sample_data):
        """Should not have both (a, b, op) and (b, a, op)."""
        X, y = sample_data
        disc = InteractionDiscovery(
            max_interactions=100,
            mi_threshold_multiplier=0.0,
            random_state=42,
        )
        result = disc.discover(X, y)

        seen = set()
        for feat_a, feat_b, op in result:
            # Normalize pair order
            pair = (min(feat_a, feat_b), max(feat_a, feat_b), op)
            assert pair not in seen, f"Duplicate pair found: {pair}"
            seen.add(pair)

    def test_discover_deterministic_naming(self, sample_data):
        """Pair names should be sorted (deterministic)."""
        X, y = sample_data
        disc = InteractionDiscovery(
            max_interactions=100,
            mi_threshold_multiplier=0.0,
            random_state=42,
        )
        result = disc.discover(X, y)

        for feat_a, feat_b, _ in result:
            assert feat_a <= feat_b, (
                f"Pair ({feat_a}, {feat_b}) not in sorted order"
            )


# =============================================================================
# TEST TRANSFORM
# =============================================================================

class TestTransform:
    """Tests for the transform() method."""

    def test_transform_adds_ix_prefix_columns(self, sample_data):
        """transform() adds columns with 'ix_' prefix."""
        X, y = sample_data
        disc = InteractionDiscovery(
            max_interactions=10,
            mi_threshold_multiplier=0.0,
            random_state=42,
        )
        disc.discover(X, y)

        if disc.n_interactions == 0:
            pytest.skip("No interactions discovered; cannot test transform columns.")

        X_out = disc.transform(X)

        # Original columns should still be present
        for col in X.columns:
            assert col in X_out.columns

        # New columns should start with "ix_"
        new_cols = [c for c in X_out.columns if c not in X.columns]
        assert len(new_cols) > 0, "Expected new interaction columns."
        for col in new_cols:
            assert col.startswith("ix_"), f"Interaction column '{col}' missing 'ix_' prefix"

    def test_transform_column_count(self, sample_data):
        """Output should have original + interaction columns."""
        X, y = sample_data
        disc = InteractionDiscovery(
            max_interactions=5,
            mi_threshold_multiplier=0.0,
            random_state=42,
        )
        disc.discover(X, y)
        X_out = disc.transform(X)

        expected = len(X.columns) + disc.n_interactions
        assert len(X_out.columns) == expected

    def test_transform_preserves_row_count(self, sample_data):
        """Row count should be unchanged after transform."""
        X, y = sample_data
        disc = InteractionDiscovery(
            max_interactions=5,
            mi_threshold_multiplier=0.0,
            random_state=42,
        )
        disc.discover(X, y)
        X_out = disc.transform(X)
        assert len(X_out) == len(X)

    def test_transform_before_discover_raises(self, sample_data):
        """Calling transform() before discover() should raise RuntimeError."""
        X, _ = sample_data
        disc = InteractionDiscovery()
        with pytest.raises(RuntimeError, match="discover"):
            disc.transform(X)


# =============================================================================
# TEST FIT_TRANSFORM
# =============================================================================

class TestFitTransform:
    """Tests for the fit_transform() shortcut."""

    def test_fit_transform_equivalent(self, sample_data):
        """fit_transform() should equal discover() + transform()."""
        X, y = sample_data

        disc1 = InteractionDiscovery(
            max_interactions=10,
            mi_threshold_multiplier=0.0,
            random_state=42,
        )
        X_out1 = disc1.fit_transform(X, y)

        disc2 = InteractionDiscovery(
            max_interactions=10,
            mi_threshold_multiplier=0.0,
            random_state=42,
        )
        disc2.discover(X, y)
        X_out2 = disc2.transform(X)

        pd.testing.assert_frame_equal(X_out1, X_out2)


# =============================================================================
# TEST OPERATIONS
# =============================================================================

class TestOperations:
    """Tests for product and ratio computation correctness."""

    def test_product_computed_correctly(self):
        """Product operation should be element-wise multiplication."""
        a = np.array([1.0, 2.0, 3.0, -1.0])
        b = np.array([4.0, 5.0, 6.0, -2.0])
        result = InteractionDiscovery._compute_product(a, b)
        expected = np.array([4.0, 10.0, 18.0, 2.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_ratio_computed_correctly(self):
        """Ratio operation should be element-wise division."""
        a = np.array([10.0, 20.0, 30.0])
        b = np.array([2.0, 5.0, 10.0])
        result = InteractionDiscovery._compute_ratio(a, b)
        expected = np.array([5.0, 4.0, 3.0])
        np.testing.assert_array_almost_equal(result, expected)

    def test_ratio_handles_zero_denominator(self):
        """Ratio should handle zero denominator without inf."""
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([0.0, 0.0, 1e-15])
        result = InteractionDiscovery._compute_ratio(a, b)
        # All denominators are effectively zero, so use 1e-10 floor
        assert np.all(np.isfinite(result))

    def test_ratio_near_zero_floor(self):
        """Ratio with near-zero denominator should use 1e-10 floor."""
        a = np.array([5.0])
        b = np.array([1e-12])  # below 1e-10 threshold
        result = InteractionDiscovery._compute_ratio(a, b)
        expected = np.array([5.0 / 1e-10])
        np.testing.assert_array_almost_equal(result, expected)


# =============================================================================
# TEST PROPERTIES
# =============================================================================

class TestProperties:
    """Tests for property accessors."""

    def test_interactions_property_returns_copy(self, sample_data):
        """interactions property should return a copy, not a reference."""
        X, y = sample_data
        disc = InteractionDiscovery(
            max_interactions=5,
            mi_threshold_multiplier=0.0,
            random_state=42,
        )
        disc.discover(X, y)

        interactions1 = disc.interactions
        interactions2 = disc.interactions

        # Should be equal content
        assert interactions1 == interactions2

        # But not the same object
        assert interactions1 is not interactions2

        # Modifying the copy should not affect the internal state
        if len(interactions1) > 0:
            interactions1.clear()
            assert disc.n_interactions > 0

    def test_n_interactions_matches_list_length(self, sample_data):
        """n_interactions should match len(interactions)."""
        X, y = sample_data
        disc = InteractionDiscovery(
            max_interactions=10,
            mi_threshold_multiplier=0.0,
            random_state=42,
        )
        disc.discover(X, y)
        assert disc.n_interactions == len(disc.interactions)

    def test_mi_scores_length_matches_interactions(self, sample_data):
        """mi_scores should have same length as interactions."""
        X, y = sample_data
        disc = InteractionDiscovery(
            max_interactions=10,
            mi_threshold_multiplier=0.0,
            random_state=42,
        )
        disc.discover(X, y)
        assert len(disc.mi_scores) == disc.n_interactions


# =============================================================================
# TEST EDGE CASES
# =============================================================================

class TestEdgeCases:
    """Tests for boundary conditions and unusual inputs."""

    def test_works_with_small_dataframe(self, small_data):
        """Should work with 5 features and 100 samples (all features used)."""
        X, y = small_data
        disc = InteractionDiscovery(
            max_interactions=10,
            mi_threshold_multiplier=0.0,
            top_k_features=15,  # larger than n_features
            random_state=42,
        )
        result = disc.discover(X, y)

        assert isinstance(result, list)
        # With 5 features and C(5,2)=10 pairs and 2 ops = 20 candidates
        # Some should pass the threshold=0.0
        X_out = disc.transform(X)
        assert len(X_out) == len(X)

    def test_empty_dataframe(self):
        """Empty DataFrame should return empty interactions."""
        X = pd.DataFrame()
        y = np.array([])
        disc = InteractionDiscovery(random_state=42)
        result = disc.discover(X, y)
        assert result == []
        assert disc._fitted

    def test_single_feature_no_interactions(self, rng):
        """With a single feature, no pairs can be formed."""
        X = pd.DataFrame({"feat_0": rng.randn(100)})
        y = (rng.randn(100) > 0).astype(int)
        disc = InteractionDiscovery(random_state=42)
        result = disc.discover(X, y)
        assert result == []

    def test_high_threshold_filters_all(self, sample_data):
        """Very high mi_threshold_multiplier should filter out most/all interactions."""
        X, y = sample_data
        disc = InteractionDiscovery(
            max_interactions=50,
            mi_threshold_multiplier=100.0,  # extremely high
            random_state=42,
        )
        result = disc.discover(X, y)
        # With such a high threshold, very few (if any) should pass
        assert len(result) <= 5  # generous upper bound

    def test_product_only_operations(self, sample_data):
        """Restricting to product-only should work."""
        X, y = sample_data
        disc = InteractionDiscovery(
            max_interactions=10,
            mi_threshold_multiplier=0.0,
            operations=["product"],
            random_state=42,
        )
        result = disc.discover(X, y)

        for _, _, op in result:
            assert op == "product"

    def test_ratio_only_operations(self, sample_data):
        """Restricting to ratio-only should work."""
        X, y = sample_data
        disc = InteractionDiscovery(
            max_interactions=10,
            mi_threshold_multiplier=0.0,
            operations=["ratio"],
            random_state=42,
        )
        result = disc.discover(X, y)

        for _, _, op in result:
            assert op == "ratio"

    def test_inf_values_in_features(self, rng):
        """Features with inf values should not crash."""
        X_arr = rng.randn(100, 5)
        X_arr[0, 0] = np.inf
        X_arr[1, 1] = -np.inf
        X = pd.DataFrame(X_arr, columns=[f"f_{i}" for i in range(5)])
        y = (rng.randn(100) > 0).astype(int)

        disc = InteractionDiscovery(
            max_interactions=5,
            mi_threshold_multiplier=0.0,
            random_state=42,
        )
        result = disc.discover(X, y)
        assert isinstance(result, list)

    def test_nan_values_in_features(self, rng):
        """Features with NaN values should not crash."""
        X_arr = rng.randn(100, 5)
        X_arr[5, 2] = np.nan
        X_arr[10, 3] = np.nan
        X = pd.DataFrame(X_arr, columns=[f"f_{i}" for i in range(5)])
        y = (rng.randn(100) > 0).astype(int)

        disc = InteractionDiscovery(
            max_interactions=5,
            mi_threshold_multiplier=0.0,
            random_state=42,
        )
        result = disc.discover(X, y)
        assert isinstance(result, list)

    def test_constant_feature_handled(self, rng):
        """Constant features (MI=0) should be handled gracefully."""
        X_arr, y = make_classification(
            n_samples=100, n_features=5, n_informative=3,
            random_state=42,
        )
        X = pd.DataFrame(X_arr, columns=[f"f_{i}" for i in range(5)])
        X["f_const"] = 7.0  # constant feature

        disc = InteractionDiscovery(
            max_interactions=10,
            mi_threshold_multiplier=0.0,
            random_state=42,
        )
        result = disc.discover(X, y)
        assert isinstance(result, list)
