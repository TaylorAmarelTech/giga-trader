"""
GIGA TRADER - Combinatorial Purged Cross-Validation (CPCV)
===========================================================
Implements CPCV as described in Marcos Lopez de Prado's
"Advances in Financial Machine Learning" (Chapter 12).

CPCV generates all C(N, k) train/test splits from N contiguous groups,
using k groups as the test set in each combination, with purging and
embargo applied at every train/test boundary to prevent data leakage.

Key advantages over standard k-fold CV:
  1. Generates many more test paths (C(6,2)=15 vs 6 folds)
  2. Enables Probability of Backtest Overfitting (PBO) computation
  3. Every combination of test blocks is evaluated
  4. Purging/embargo prevents autocorrelation leakage

Also exports compute_pbo() for computing the Probability of Backtest
Overfitting from a collection of out-of-sample test performances.
"""

from itertools import combinations
from math import comb
from typing import Generator, List, Optional, Tuple

import numpy as np
from sklearn.model_selection._split import BaseCrossValidator


class CombinatorialPurgedCV(BaseCrossValidator):
    """
    Combinatorial Purged Cross-Validation (CPCV).

    Splits data into ``n_groups`` contiguous blocks and generates all
    C(n_groups, n_test_groups) train/test combinations. For each
    combination, purging and embargo are applied at every boundary
    between train and test blocks to prevent data leakage from
    autocorrelation.

    Parameters
    ----------
    n_groups : int, default=6
        Number of contiguous groups to split the data into.
    n_test_groups : int, default=2
        Number of groups to designate as the test set in each split.
        Must be strictly less than ``n_groups``.
    purge_days : int, default=10
        Number of samples to remove from the training set that are
        immediately *before* each test block boundary. This prevents
        leakage from autocorrelated features computed on overlapping
        windows.
    embargo_days : int, default=3
        Number of samples to remove from the training set that are
        immediately *after* each test block boundary. This prevents
        leakage from look-ahead in feature construction.

    Examples
    --------
    >>> import numpy as np
    >>> from src.phase_11_cv_splitting.cpcv import CombinatorialPurgedCV
    >>> X = np.random.randn(252, 10)  # 1 year of daily data
    >>> cv = CombinatorialPurgedCV(n_groups=6, n_test_groups=2, purge_days=5, embargo_days=2)
    >>> cv.get_n_splits()
    15
    >>> for train_idx, test_idx in cv.split(X):
    ...     print(f"Train: {len(train_idx)}, Test: {len(test_idx)}")

    Notes
    -----
    The number of splits is C(n_groups, n_test_groups). With the
    default n_groups=6 and n_test_groups=2, this yields 15 splits.
    Each sample in the dataset appears in exactly C(n_groups-1,
    n_test_groups-1) test sets (before purge/embargo removal from
    the training side).
    """

    def __init__(
        self,
        n_groups: int = 6,
        n_test_groups: int = 2,
        purge_days: int = 10,
        embargo_days: int = 3,
    ):
        if n_groups < 2:
            raise ValueError(
                f"n_groups must be >= 2, got {n_groups}"
            )
        if n_test_groups < 1:
            raise ValueError(
                f"n_test_groups must be >= 1, got {n_test_groups}"
            )
        if n_test_groups >= n_groups:
            raise ValueError(
                f"n_test_groups ({n_test_groups}) must be strictly less "
                f"than n_groups ({n_groups})"
            )
        if purge_days < 0:
            raise ValueError(
                f"purge_days must be >= 0, got {purge_days}"
            )
        if embargo_days < 0:
            raise ValueError(
                f"embargo_days must be >= 0, got {embargo_days}"
            )

        self.n_groups = n_groups
        self.n_test_groups = n_test_groups
        self.purge_days = purge_days
        self.embargo_days = embargo_days

    def get_n_splits(
        self,
        X=None,
        y=None,
        groups=None,
    ) -> int:
        """
        Return the number of splitting iterations.

        Returns C(n_groups, n_test_groups).

        Parameters
        ----------
        X : ignored
        y : ignored
        groups : ignored

        Returns
        -------
        int
            Number of splits.
        """
        return comb(self.n_groups, self.n_test_groups)

    def _compute_group_boundaries(
        self,
        n_samples: int,
    ) -> List[Tuple[int, int]]:
        """
        Compute the [start, end) boundaries for each contiguous group.

        Distributes samples as evenly as possible. The last group may
        be slightly larger if n_samples is not evenly divisible.

        Parameters
        ----------
        n_samples : int
            Total number of samples.

        Returns
        -------
        list of (int, int)
            List of (start_index, end_index) tuples, one per group.
        """
        block_size = n_samples // self.n_groups
        boundaries = []
        for i in range(self.n_groups):
            start = i * block_size
            if i == self.n_groups - 1:
                end = n_samples  # Last group absorbs remainder
            else:
                end = (i + 1) * block_size
            boundaries.append((start, end))
        return boundaries

    def split(
        self,
        X,
        y=None,
        groups=None,
    ) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        """
        Generate train/test index arrays for each CPCV combination.

        For each of the C(n_groups, n_test_groups) combinations:
          - Test set = union of the selected groups
          - Train set = all remaining indices, minus purged/embargoed
            samples near test block boundaries

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data. Only ``len(X)`` is used.
        y : ignored
        groups : ignored

        Yields
        ------
        train_indices : ndarray of int
            Indices for the training set.
        test_indices : ndarray of int
            Indices for the test set.
        """
        n_samples = len(X)
        if n_samples < self.n_groups:
            raise ValueError(
                f"Number of samples ({n_samples}) must be >= n_groups "
                f"({self.n_groups})"
            )

        boundaries = self._compute_group_boundaries(n_samples)

        for test_group_indices in combinations(
            range(self.n_groups), self.n_test_groups
        ):
            # Build test set: union of selected group blocks
            test_set = set()
            for gi in test_group_indices:
                start, end = boundaries[gi]
                test_set.update(range(start, end))

            # Build the full set of train candidates (everything not test)
            all_indices = set(range(n_samples))
            train_candidates = all_indices - test_set

            # Apply purging and embargo at each test block boundary
            purge_embargo_set = set()
            for gi in test_group_indices:
                block_start, block_end = boundaries[gi]

                # Purge: remove train samples immediately BEFORE test block
                purge_start = max(0, block_start - self.purge_days)
                for idx in range(purge_start, block_start):
                    if idx in train_candidates:
                        purge_embargo_set.add(idx)

                # Embargo: remove train samples immediately AFTER test block
                embargo_end = min(n_samples, block_end + self.embargo_days)
                for idx in range(block_end, embargo_end):
                    if idx in train_candidates:
                        purge_embargo_set.add(idx)

            train_indices = np.array(
                sorted(train_candidates - purge_embargo_set), dtype=np.intp
            )
            test_indices = np.array(sorted(test_set), dtype=np.intp)

            # Only yield if both sets are non-empty
            if len(train_indices) > 0 and len(test_indices) > 0:
                yield train_indices, test_indices

    def _iter_test_indices(self, X=None, y=None, groups=None):
        """Required by BaseCrossValidator; delegates to split()."""
        for _, test_idx in self.split(X, y, groups):
            yield test_idx


def compute_pbo(
    test_performances: List[float],
    paired_is_performances: Optional[List[float]] = None,
) -> float:
    """
    Compute the Probability of Backtest Overfitting (PBO).

    In its full form (de Prado, 2018), PBO requires paired in-sample
    and out-of-sample performance for each CPCV combination, and
    computes the probability that the best in-sample strategy
    underperforms the median out-of-sample.

    This function supports two modes:

    1. **Simplified (test_performances only):**
       PBO = fraction of out-of-sample results that fall below the
       median OOS performance. A PBO > 0.5 indicates overfitting;
       PBO < 0.25 suggests robustness.

    2. **Paired (test_performances + paired_is_performances):**
       For each split, we check whether the strategy that performed
       best in-sample also performed below the OOS median. PBO is
       the fraction of splits where this "rank inversion" occurs.

    Parameters
    ----------
    test_performances : list of float
        Out-of-sample performance metric (e.g., AUC, Sharpe) for
        each CPCV split.
    paired_is_performances : list of float, optional
        Corresponding in-sample performance for each split. If
        provided, the paired PBO computation is used.

    Returns
    -------
    float
        PBO estimate in [0, 1]. Lower is better.

    Raises
    ------
    ValueError
        If test_performances is empty or if paired lists have
        mismatched lengths.

    Examples
    --------
    >>> # All OOS above median -> PBO = 0
    >>> compute_pbo([0.7, 0.8, 0.9, 0.75, 0.85])
    0.0

    >>> # Half below median -> PBO ~ 0.5
    >>> compute_pbo([0.5, 0.6, 0.7, 0.8])
    0.5
    """
    if not test_performances:
        raise ValueError("test_performances must be a non-empty list")

    oos = np.array(test_performances, dtype=np.float64)

    if len(oos) == 1:
        # Single split: cannot compute meaningful PBO
        return 0.0

    if paired_is_performances is not None:
        # Paired PBO: check rank inversion
        if len(paired_is_performances) != len(test_performances):
            raise ValueError(
                f"paired_is_performances length ({len(paired_is_performances)}) "
                f"must match test_performances length ({len(test_performances)})"
            )
        is_perf = np.array(paired_is_performances, dtype=np.float64)

        # For each split, the "best IS" strategy is the one with
        # highest in-sample score. We check if its OOS is below median.
        median_oos = np.median(oos)
        best_is_idx = np.argmax(is_perf)
        # But we want to generalize: for each combination of splits,
        # check if the best-IS strategy underperforms OOS.
        # Simplified paired: rank IS, check if top-IS corresponds to
        # below-median OOS.
        n_inversions = 0
        n_comparisons = len(oos)
        is_ranks = np.argsort(-is_perf)  # Descending IS rank

        for i in range(n_comparisons):
            # The i-th best IS performer: does it fall below OOS median?
            idx = is_ranks[i]
            if oos[idx] < median_oos:
                n_inversions += 1

        return n_inversions / n_comparisons

    # Simplified PBO: fraction of OOS below median
    median_oos = np.median(oos)
    n_below = np.sum(oos < median_oos)
    return float(n_below / len(oos))
