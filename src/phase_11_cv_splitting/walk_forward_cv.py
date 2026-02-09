"""
GIGA TRADER - Walk-Forward Cross-Validation
=============================================
Walk-forward cross-validation with purging and embargo.

Prevents data leakage by:
  1. Always training on past data, testing on future data
  2. Purging: Remove N days between train and test (autocorrelation)
  3. Embargo: Remove N days after test before next train (information bleed)
"""

import numpy as np


class WalkForwardCV:
    """
    Walk-forward cross-validation with purging and embargo.

    Prevents data leakage by:
      1. Always training on past data, testing on future data
      2. Purging: Remove N days between train and test (autocorrelation)
      3. Embargo: Remove N days after test before next train (information bleed)
    """

    def __init__(
        self,
        n_splits: int = 5,
        purge_days: int = 5,
        embargo_days: int = 2,
        min_train_size: int = 100,
        test_size: int = 50,
    ):
        self.n_splits = n_splits
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.min_train_size = min_train_size
        self.test_size = test_size

    def split(self, X, y=None, dates=None):
        """
        Generate walk-forward train/test indices with purging and embargo.

        Args:
            X: Feature matrix
            y: Target vector (optional)
            dates: Array of dates for each sample (required for proper purging)

        Yields:
            train_idx, test_idx tuples
        """
        n_samples = len(X)

        if dates is None:
            # Fall back to simple time-based split (less accurate)
            dates = np.arange(n_samples)

        unique_dates = np.unique(dates)
        n_dates = len(unique_dates)

        # Calculate split points
        total_test = self.n_splits * self.test_size
        available_for_train = n_dates - total_test - self.n_splits * (self.purge_days + self.embargo_days)

        if available_for_train < self.min_train_size:
            # Reduce n_splits if not enough data
            self.n_splits = max(2, (n_dates - self.min_train_size) // (self.test_size + self.purge_days + self.embargo_days))

        for split_idx in range(self.n_splits):
            # Calculate date indices for this split
            test_end_date_idx = n_dates - split_idx * (self.test_size + self.embargo_days)
            test_start_date_idx = test_end_date_idx - self.test_size

            # Training ends before purge period
            train_end_date_idx = test_start_date_idx - self.purge_days
            train_start_date_idx = max(0, train_end_date_idx - (self.min_train_size + split_idx * 20))

            if train_end_date_idx <= train_start_date_idx:
                continue

            # Convert date indices to sample indices
            train_dates = unique_dates[train_start_date_idx:train_end_date_idx]
            test_dates = unique_dates[test_start_date_idx:test_end_date_idx]

            train_idx = np.where(np.isin(dates, train_dates))[0]
            test_idx = np.where(np.isin(dates, test_dates))[0]

            if len(train_idx) >= self.min_train_size and len(test_idx) > 0:
                yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits
