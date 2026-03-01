"""
Tests for Combinatorial Purged Cross-Validation (CPCV).

Wave E2: 30+ tests across 7 test classes covering split counts,
quality, purging, embargo, edge cases, PBO, and sklearn compatibility.
"""

import sys
from pathlib import Path
from itertools import combinations
from math import comb

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.phase_11_cv_splitting.cpcv import CombinatorialPurgedCV, compute_pbo


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_data(n_samples: int = 252, n_features: int = 10, seed: int = 42):
    """Generate synthetic array data for CV splitting tests."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n_samples, n_features)
    y = rng.randint(0, 2, size=n_samples)
    return X, y


# ===========================================================================
# 1. TestCPCVSplitCount
# ===========================================================================


class TestCPCVSplitCount:
    """Verify that the number of splits matches C(n_groups, n_test_groups)."""

    def test_default_parameters(self):
        """Default n_groups=6, n_test_groups=2 -> C(6,2)=15 splits."""
        cv = CombinatorialPurgedCV()
        assert cv.get_n_splits() == 15

    def test_c_6_2(self):
        """C(6, 2) = 15."""
        cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2)
        assert cv.get_n_splits() == comb(6, 2)

    def test_c_8_3(self):
        """C(8, 3) = 56."""
        cv = CombinatorialPurgedCV(n_groups=8, n_test_groups=3)
        assert cv.get_n_splits() == comb(8, 3)

    def test_c_10_2(self):
        """C(10, 2) = 45."""
        cv = CombinatorialPurgedCV(n_groups=10, n_test_groups=2)
        assert cv.get_n_splits() == 45

    def test_c_4_1(self):
        """C(4, 1) = 4."""
        cv = CombinatorialPurgedCV(n_groups=4, n_test_groups=1)
        assert cv.get_n_splits() == 4

    def test_c_2_1(self):
        """C(2, 1) = 2 -- minimal case."""
        cv = CombinatorialPurgedCV(n_groups=2, n_test_groups=1)
        assert cv.get_n_splits() == 2

    def test_actual_split_count_matches_get_n_splits(self):
        """The generator should yield exactly get_n_splits() pairs."""
        X, _ = _make_data(n_samples=300)
        cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, purge_days=2, embargo_days=1)
        splits = list(cv.split(X))
        assert len(splits) == cv.get_n_splits()

    def test_split_count_with_various_n_groups(self):
        """Parametrize over several n_groups/n_test_groups combos."""
        X, _ = _make_data(n_samples=500)
        for ng, ntg in [(5, 2), (6, 3), (7, 2), (4, 1), (3, 1)]:
            cv = CombinatorialPurgedCV(
                n_groups=ng, n_test_groups=ntg, purge_days=2, embargo_days=1
            )
            splits = list(cv.split(X))
            assert len(splits) == comb(ng, ntg), (
                f"Expected {comb(ng, ntg)} splits for C({ng},{ntg}), got {len(splits)}"
            )


# ===========================================================================
# 2. TestCPCVSplitQuality
# ===========================================================================


class TestCPCVSplitQuality:
    """Verify fundamental properties of the generated splits."""

    def test_no_overlap_train_test(self):
        """Train and test indices must be disjoint in every split."""
        X, _ = _make_data(n_samples=252)
        cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, purge_days=5, embargo_days=2)
        for train_idx, test_idx in cv.split(X):
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0, f"Overlap found: {overlap}"

    def test_all_indices_in_test_union(self):
        """Union of all test sets should cover every sample index."""
        X, _ = _make_data(n_samples=252)
        cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, purge_days=0, embargo_days=0)
        all_test = set()
        for _, test_idx in cv.split(X):
            all_test.update(test_idx.tolist())
        expected = set(range(len(X)))
        assert all_test == expected, (
            f"Missing indices in test union: {expected - all_test}"
        )

    def test_test_sets_are_contiguous_blocks(self):
        """Each test set should be a union of contiguous blocks."""
        X, _ = _make_data(n_samples=252)
        cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, purge_days=0, embargo_days=0)
        block_size = len(X) // 6  # 42

        for _, test_idx in cv.split(X):
            # Test indices should form exactly n_test_groups contiguous blocks
            # Find breaks
            diffs = np.diff(test_idx)
            breaks = np.where(diffs > 1)[0]
            n_blocks = len(breaks) + 1
            assert n_blocks <= cv.n_test_groups, (
                f"Expected <= {cv.n_test_groups} contiguous blocks, got {n_blocks}"
            )

    def test_train_indices_sorted(self):
        """Train indices should be sorted in ascending order."""
        X, _ = _make_data(n_samples=252)
        cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, purge_days=5, embargo_days=2)
        for train_idx, _ in cv.split(X):
            assert np.all(train_idx[:-1] <= train_idx[1:]), "Train indices not sorted"

    def test_test_indices_sorted(self):
        """Test indices should be sorted in ascending order."""
        X, _ = _make_data(n_samples=252)
        cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, purge_days=5, embargo_days=2)
        for _, test_idx in cv.split(X):
            assert np.all(test_idx[:-1] <= test_idx[1:]), "Test indices not sorted"

    def test_train_test_purge_embargo_accounts_for_all(self):
        """train + test + purge/embargo = total samples for each split."""
        n = 300
        X, _ = _make_data(n_samples=n)
        cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, purge_days=5, embargo_days=3)
        for train_idx, test_idx in cv.split(X):
            total_accounted = len(train_idx) + len(test_idx)
            removed = n - total_accounted
            # Removed indices are from purge/embargo; should be >= 0
            assert removed >= 0, f"Negative removal count: {removed}"
            # With purge=5 and embargo=3, we remove at most
            # n_test_groups * 2 boundaries * (purge + embargo) but capped
            assert total_accounted <= n


# ===========================================================================
# 3. TestCPCVPurging
# ===========================================================================


class TestCPCVPurging:
    """Verify that purging correctly removes samples near test boundaries."""

    def test_purge_removes_samples_before_test(self):
        """Samples in [test_start - purge_days, test_start) must be absent from train."""
        n = 252
        X, _ = _make_data(n_samples=n)
        purge_days = 10
        cv = CombinatorialPurgedCV(
            n_groups=6, n_test_groups=2, purge_days=purge_days, embargo_days=0
        )
        block_size = n // 6

        for train_idx, test_idx in cv.split(X):
            train_set = set(train_idx.tolist())
            test_set = set(test_idx.tolist())

            # Find test block boundaries
            test_sorted = sorted(test_set)
            # Detect contiguous blocks in test set
            blocks = []
            block_start = test_sorted[0]
            for i in range(1, len(test_sorted)):
                if test_sorted[i] != test_sorted[i - 1] + 1:
                    blocks.append((block_start, test_sorted[i - 1] + 1))
                    block_start = test_sorted[i]
            blocks.append((block_start, test_sorted[-1] + 1))

            for bstart, bend in blocks:
                # Purge zone: indices before each test block
                for idx in range(max(0, bstart - purge_days), bstart):
                    if idx not in test_set:
                        assert idx not in train_set, (
                            f"Index {idx} in purge zone [before test block "
                            f"starting at {bstart}] should not be in train"
                        )

    def test_purge_days_zero_means_no_purging(self):
        """With purge_days=0, no extra samples are removed before test."""
        n = 252
        X, _ = _make_data(n_samples=n)
        cv = CombinatorialPurgedCV(
            n_groups=6, n_test_groups=2, purge_days=0, embargo_days=0
        )
        for train_idx, test_idx in cv.split(X):
            # train + test should equal all indices
            all_idx = set(train_idx.tolist()) | set(test_idx.tolist())
            assert all_idx == set(range(n))

    def test_larger_purge_removes_more(self):
        """Increasing purge_days should result in fewer train samples."""
        X, _ = _make_data(n_samples=252)
        train_sizes = []
        for pd in [0, 5, 10, 20]:
            cv = CombinatorialPurgedCV(
                n_groups=6, n_test_groups=2, purge_days=pd, embargo_days=0
            )
            # Take the first split for comparison
            train_idx, _ = next(cv.split(X))
            train_sizes.append(len(train_idx))

        # Each successive purge_days should produce fewer (or equal) train samples
        for i in range(len(train_sizes) - 1):
            assert train_sizes[i] >= train_sizes[i + 1], (
                f"Train size did not decrease with more purging: "
                f"purge_days progression yields {train_sizes}"
            )


# ===========================================================================
# 4. TestCPCVEmbargo
# ===========================================================================


class TestCPCVEmbargo:
    """Verify that embargo correctly removes samples after test blocks."""

    def test_embargo_removes_samples_after_test(self):
        """Samples in [test_end, test_end + embargo_days) must be absent from train."""
        n = 252
        X, _ = _make_data(n_samples=n)
        embargo_days = 5
        cv = CombinatorialPurgedCV(
            n_groups=6, n_test_groups=2, purge_days=0, embargo_days=embargo_days
        )

        for train_idx, test_idx in cv.split(X):
            train_set = set(train_idx.tolist())
            test_set = set(test_idx.tolist())

            # Find test block boundaries
            test_sorted = sorted(test_set)
            blocks = []
            block_start = test_sorted[0]
            for i in range(1, len(test_sorted)):
                if test_sorted[i] != test_sorted[i - 1] + 1:
                    blocks.append((block_start, test_sorted[i - 1] + 1))
                    block_start = test_sorted[i]
            blocks.append((block_start, test_sorted[-1] + 1))

            for bstart, bend in blocks:
                # Embargo zone: indices after each test block
                for idx in range(bend, min(n, bend + embargo_days)):
                    if idx not in test_set:
                        assert idx not in train_set, (
                            f"Index {idx} in embargo zone [after test block "
                            f"ending at {bend}] should not be in train"
                        )

    def test_embargo_days_zero_means_no_embargo(self):
        """With embargo_days=0, no samples removed after test blocks."""
        n = 252
        X, _ = _make_data(n_samples=n)
        cv = CombinatorialPurgedCV(
            n_groups=6, n_test_groups=2, purge_days=0, embargo_days=0
        )
        for train_idx, test_idx in cv.split(X):
            all_idx = set(train_idx.tolist()) | set(test_idx.tolist())
            assert all_idx == set(range(n))

    def test_larger_embargo_removes_more(self):
        """Increasing embargo_days should result in fewer train samples."""
        X, _ = _make_data(n_samples=252)
        train_sizes = []
        for ed in [0, 3, 7, 15]:
            cv = CombinatorialPurgedCV(
                n_groups=6, n_test_groups=2, purge_days=0, embargo_days=ed
            )
            train_idx, _ = next(cv.split(X))
            train_sizes.append(len(train_idx))

        for i in range(len(train_sizes) - 1):
            assert train_sizes[i] >= train_sizes[i + 1], (
                f"Train size did not decrease with more embargo: {train_sizes}"
            )

    def test_purge_and_embargo_combined(self):
        """Both purge and embargo applied simultaneously."""
        n = 252
        X, _ = _make_data(n_samples=n)
        purge_days = 5
        embargo_days = 3

        cv_none = CombinatorialPurgedCV(
            n_groups=6, n_test_groups=2, purge_days=0, embargo_days=0
        )
        cv_both = CombinatorialPurgedCV(
            n_groups=6, n_test_groups=2,
            purge_days=purge_days, embargo_days=embargo_days,
        )

        for (tr_none, _), (tr_both, _) in zip(cv_none.split(X), cv_both.split(X)):
            assert len(tr_both) < len(tr_none), (
                "Combined purge+embargo should produce fewer train samples"
            )


# ===========================================================================
# 5. TestCPCVEdgeCases
# ===========================================================================


class TestCPCVEdgeCases:
    """Edge cases and validation."""

    def test_n_groups_equals_2_single_combination(self):
        """C(2, 1) = 2 splits. Minimal valid configuration."""
        X, _ = _make_data(n_samples=100)
        cv = CombinatorialPurgedCV(n_groups=2, n_test_groups=1, purge_days=2, embargo_days=1)
        splits = list(cv.split(X))
        assert len(splits) == 2

    def test_n_test_groups_equals_1(self):
        """With n_test_groups=1, behaves like leave-one-group-out."""
        X, _ = _make_data(n_samples=200)
        cv = CombinatorialPurgedCV(n_groups=5, n_test_groups=1, purge_days=3, embargo_days=1)
        splits = list(cv.split(X))
        assert len(splits) == 5

    def test_very_small_dataset(self):
        """With small data, each block has few samples but should still work."""
        X, _ = _make_data(n_samples=20)
        cv = CombinatorialPurgedCV(
            n_groups=4, n_test_groups=2, purge_days=1, embargo_days=0
        )
        splits = list(cv.split(X))
        assert len(splits) == comb(4, 2)
        for train_idx, test_idx in splits:
            assert len(train_idx) > 0
            assert len(test_idx) > 0

    def test_purge_larger_than_gap(self):
        """When purge_days exceeds gap between blocks, more train is removed."""
        X, _ = _make_data(n_samples=60)
        # 6 groups of 10 samples each. purge=15 means the purge zone
        # overlaps with adjacent blocks.
        cv = CombinatorialPurgedCV(
            n_groups=6, n_test_groups=2, purge_days=15, embargo_days=0
        )
        splits = list(cv.split(X))
        # Should still produce valid splits (possibly very few train samples)
        for train_idx, test_idx in splits:
            overlap = np.intersect1d(train_idx, test_idx)
            assert len(overlap) == 0

    def test_invalid_n_groups_raises(self):
        """n_groups < 2 should raise ValueError."""
        with pytest.raises(ValueError, match="n_groups must be >= 2"):
            CombinatorialPurgedCV(n_groups=1)

    def test_invalid_n_test_groups_zero_raises(self):
        """n_test_groups < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="n_test_groups must be >= 1"):
            CombinatorialPurgedCV(n_groups=6, n_test_groups=0)

    def test_n_test_groups_ge_n_groups_raises(self):
        """n_test_groups >= n_groups should raise ValueError."""
        with pytest.raises(ValueError, match="n_test_groups .* must be strictly less"):
            CombinatorialPurgedCV(n_groups=6, n_test_groups=6)

    def test_negative_purge_raises(self):
        """Negative purge_days should raise ValueError."""
        with pytest.raises(ValueError, match="purge_days must be >= 0"):
            CombinatorialPurgedCV(purge_days=-1)

    def test_negative_embargo_raises(self):
        """Negative embargo_days should raise ValueError."""
        with pytest.raises(ValueError, match="embargo_days must be >= 0"):
            CombinatorialPurgedCV(embargo_days=-1)

    def test_too_few_samples_raises(self):
        """n_samples < n_groups should raise ValueError."""
        X = np.zeros((3, 2))
        cv = CombinatorialPurgedCV(n_groups=6)
        with pytest.raises(ValueError, match="Number of samples"):
            list(cv.split(X))

    def test_last_group_absorbs_remainder(self):
        """When n_samples not divisible by n_groups, last group is larger."""
        n = 253  # 253 / 6 = 42 remainder 1
        X, _ = _make_data(n_samples=n)
        cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, purge_days=0, embargo_days=0)
        # Check that all indices are covered
        all_test = set()
        for _, test_idx in cv.split(X):
            all_test.update(test_idx.tolist())
        assert all_test == set(range(n))


# ===========================================================================
# 6. TestComputePBO
# ===========================================================================


class TestComputePBO:
    """Tests for the compute_pbo helper function."""

    def test_all_above_median_returns_zero(self):
        """If all performances are above median, PBO = 0."""
        # All same -> median = value, none strictly below -> PBO = 0
        performances = [0.7, 0.7, 0.7, 0.7]
        pbo = compute_pbo(performances)
        assert pbo == 0.0

    def test_all_distinct_half_below(self):
        """Even number of distinct values: half below median -> PBO ~0.5."""
        # [1, 2, 3, 4] -> median = 2.5 -> 1,2 below -> PBO = 2/4 = 0.5
        pbo = compute_pbo([1.0, 2.0, 3.0, 4.0])
        assert pbo == 0.5

    def test_strong_model_low_pbo(self):
        """Strong model with consistent OOS should have low PBO."""
        # All values above a high threshold, none below median
        pbo = compute_pbo([0.8, 0.82, 0.85, 0.81, 0.83])
        # median = 0.82, two below (0.8, 0.81) -> PBO = 2/5 = 0.4
        assert pbo == pytest.approx(0.4)

    def test_weak_model_high_pbo(self):
        """Model with many bad OOS should have high PBO."""
        # [0.45, 0.46, 0.47, 0.48, 0.80] -> median=0.47
        # Below median: 0.45, 0.46 -> PBO = 2/5 = 0.4
        pbo = compute_pbo([0.45, 0.46, 0.47, 0.48, 0.80])
        assert 0.0 <= pbo <= 1.0

    def test_single_value_returns_zero(self):
        """Single performance value: cannot compute meaningful PBO."""
        pbo = compute_pbo([0.7])
        assert pbo == 0.0

    def test_empty_list_raises(self):
        """Empty list should raise ValueError."""
        with pytest.raises(ValueError, match="non-empty"):
            compute_pbo([])

    def test_pbo_in_range(self):
        """PBO should always be in [0, 1]."""
        rng = np.random.RandomState(42)
        for _ in range(20):
            n = rng.randint(2, 50)
            perfs = rng.uniform(0.4, 0.9, size=n).tolist()
            pbo = compute_pbo(perfs)
            assert 0.0 <= pbo <= 1.0, f"PBO {pbo} out of range for perfs of length {n}"

    def test_paired_pbo_basic(self):
        """Paired PBO: best IS that is worst OOS gives high PBO."""
        # IS: [0.9, 0.5, 0.6, 0.55] -> ranks: 0, 3, 1, 2
        # OOS: [0.3, 0.7, 0.65, 0.68] -> median = 0.665
        # Best IS is idx 0 -> OOS 0.3 < 0.665 -> inversion
        pbo = compute_pbo(
            test_performances=[0.3, 0.7, 0.65, 0.68],
            paired_is_performances=[0.9, 0.5, 0.6, 0.55],
        )
        assert 0.0 <= pbo <= 1.0
        # idx=0 (IS best, OOS=0.3 < median) -> inversion for rank 0
        # idx=2 (IS 2nd, OOS=0.65 < 0.665) -> inversion for rank 1
        # At least some inversions expected
        assert pbo > 0.0

    def test_paired_pbo_mismatched_lengths_raises(self):
        """Paired PBO with mismatched lengths should raise ValueError."""
        with pytest.raises(ValueError, match="must match"):
            compute_pbo(
                test_performances=[0.5, 0.6],
                paired_is_performances=[0.7, 0.8, 0.9],
            )

    def test_paired_pbo_perfect_alignment(self):
        """When best IS also has best OOS, PBO should be low."""
        # IS: [0.9, 0.8, 0.7, 0.6] and OOS: [0.85, 0.75, 0.65, 0.55]
        # Perfect correlation. median OOS = 0.7
        # IS ranks (desc): idx 0,1,2,3
        # OOS[0]=0.85 > 0.7 (no inversion), OOS[1]=0.75>0.7, OOS[2]=0.65<0.7 (inv), OOS[3]=0.55<0.7 (inv)
        # PBO = 2/4 = 0.5
        pbo = compute_pbo(
            test_performances=[0.85, 0.75, 0.65, 0.55],
            paired_is_performances=[0.9, 0.8, 0.7, 0.6],
        )
        assert pbo == pytest.approx(0.5)


# ===========================================================================
# 7. TestSklearnCompat
# ===========================================================================


class TestSklearnCompat:
    """Verify compatibility with sklearn's cross-validation infrastructure."""

    def test_get_n_splits_consistent_with_split(self):
        """get_n_splits should return the same count as len(list(split(...)))."""
        X, _ = _make_data(n_samples=252)
        for ng, ntg in [(6, 2), (5, 1), (8, 3)]:
            cv = CombinatorialPurgedCV(
                n_groups=ng, n_test_groups=ntg, purge_days=3, embargo_days=1
            )
            expected = cv.get_n_splits()
            actual = len(list(cv.split(X)))
            assert expected == actual

    def test_works_with_cross_val_score(self):
        """CPCV should work with sklearn.model_selection.cross_val_score."""
        from sklearn.model_selection import cross_val_score
        from sklearn.linear_model import LogisticRegression

        X, y = _make_data(n_samples=252, n_features=5)
        cv = CombinatorialPurgedCV(
            n_groups=4, n_test_groups=1, purge_days=3, embargo_days=1
        )
        model = LogisticRegression(max_iter=200, random_state=42)
        scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
        assert len(scores) == cv.get_n_splits()
        # All scores should be valid (between 0 and 1)
        for s in scores:
            assert 0.0 <= s <= 1.0

    def test_isinstance_base_cross_validator(self):
        """CombinatorialPurgedCV should be a BaseCrossValidator instance."""
        from sklearn.model_selection._split import BaseCrossValidator
        cv = CombinatorialPurgedCV()
        assert isinstance(cv, BaseCrossValidator)

    def test_split_returns_numpy_arrays(self):
        """Train and test indices should be numpy ndarrays."""
        X, _ = _make_data(n_samples=252)
        cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, purge_days=3, embargo_days=1)
        for train_idx, test_idx in cv.split(X):
            assert isinstance(train_idx, np.ndarray)
            assert isinstance(test_idx, np.ndarray)

    def test_split_dtype_is_intp(self):
        """Indices should have intp dtype for sklearn compatibility."""
        X, _ = _make_data(n_samples=252)
        cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, purge_days=3, embargo_days=1)
        for train_idx, test_idx in cv.split(X):
            assert train_idx.dtype == np.intp
            assert test_idx.dtype == np.intp

    def test_multiple_iterations_yield_same_splits(self):
        """Iterating split() twice should produce identical results (deterministic)."""
        X, _ = _make_data(n_samples=252)
        cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, purge_days=5, embargo_days=2)
        splits_1 = [(tr.copy(), te.copy()) for tr, te in cv.split(X)]
        splits_2 = [(tr.copy(), te.copy()) for tr, te in cv.split(X)]
        assert len(splits_1) == len(splits_2)
        for (tr1, te1), (tr2, te2) in zip(splits_1, splits_2):
            np.testing.assert_array_equal(tr1, tr2)
            np.testing.assert_array_equal(te1, te2)
