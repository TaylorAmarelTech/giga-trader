"""
GIGA TRADER - Leak-Proof Feature Selector
==========================================
Feature selector that MUST be fitted only on training data.

This wraps multiple selection methods and ensures they're only
fitted on the training fold, never seeing test data.

Fixes critical data leakage: feature selection INSIDE CV loop (not before).
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class LeakProofFeatureSelector(BaseEstimator, TransformerMixin):
    """
    Feature selector that MUST be fitted only on training data.

    This wraps multiple selection methods and ensures they're only
    fitted on the training fold, never seeing test data.
    """

    def __init__(
        self,
        method: str = "mutual_info",
        n_features: int = 30,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        random_state: int = 42,
    ):
        self.method = method
        self.n_features = n_features
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.random_state = random_state

        # Will be set during fit
        self.var_mask_ = None
        self.corr_mask_ = None
        self.selected_idx_ = None
        self.n_features_in_ = None
        self.n_features_out_ = None

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Fit feature selector on TRAINING data only.

        This method learns which features to keep based only on the
        training data, preventing any leakage from test data.
        """
        self.n_features_in_ = X.shape[1]

        # Stage 1: Variance threshold
        from sklearn.feature_selection import VarianceThreshold
        var_selector = VarianceThreshold(threshold=self.variance_threshold)
        var_selector.fit(X)
        self.var_mask_ = var_selector.get_support()
        X_var = X[:, self.var_mask_]

        # Stage 2: Correlation filter (remove redundant features)
        if X_var.shape[1] > 1:
            corr_matrix = np.corrcoef(X_var.T)
            # Handle NaN in correlation matrix
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)

            to_drop = set()
            for i in range(len(corr_matrix)):
                if i in to_drop:
                    continue
                for j in range(i + 1, len(corr_matrix)):
                    if j in to_drop:
                        continue
                    if abs(corr_matrix[i, j]) > self.correlation_threshold:
                        to_drop.add(j)

            self.corr_mask_ = np.array([i not in to_drop for i in range(X_var.shape[1])])
            X_corr = X_var[:, self.corr_mask_]
        else:
            self.corr_mask_ = np.ones(X_var.shape[1], dtype=bool)
            X_corr = X_var

        # Stage 3: Feature selection (Mutual Info or other)
        if y is not None and X_corr.shape[1] > self.n_features:
            if self.method == "mutual_info":
                from sklearn.feature_selection import mutual_info_classif

                # Compute MI scores on TRAINING data only
                mi_scores = mutual_info_classif(
                    X_corr, y,
                    n_neighbors=5,
                    random_state=self.random_state
                )

                # Select top features
                n_to_select = min(self.n_features, X_corr.shape[1])
                top_idx = np.argsort(mi_scores)[::-1][:n_to_select]
                self.selected_idx_ = np.sort(top_idx)  # Keep original order

            elif self.method == "f_classif":
                from sklearn.feature_selection import f_classif
                f_scores, _ = f_classif(X_corr, y)
                n_to_select = min(self.n_features, X_corr.shape[1])
                top_idx = np.argsort(f_scores)[::-1][:n_to_select]
                self.selected_idx_ = np.sort(top_idx)

            else:
                # No selection, keep all
                self.selected_idx_ = np.arange(X_corr.shape[1])
        else:
            self.selected_idx_ = np.arange(X_corr.shape[1])

        self.n_features_out_ = len(self.selected_idx_)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform using the fitted selection masks."""
        if self.var_mask_ is None:
            raise ValueError("FeatureSelector not fitted. Call fit() first.")

        # Apply all masks in order
        X_var = X[:, self.var_mask_]
        X_corr = X_var[:, self.corr_mask_]
        X_selected = X_corr[:, self.selected_idx_]

        return X_selected

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
