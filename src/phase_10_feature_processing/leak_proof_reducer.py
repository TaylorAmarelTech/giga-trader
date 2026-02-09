"""
GIGA TRADER - Leak-Proof Dimensionality Reducer
=================================================
Dimensionality reduction that MUST be fitted only on training data.

Supports multiple methods:
- kernel_pca: Non-linear PCA with RBF kernel
- ica: Independent Component Analysis
- pca: Standard PCA
- ensemble: Combination of methods

Fixes critical data leakage: dimensionality reduction INSIDE CV loop (not before).
"""

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler


class LeakProofDimReducer(BaseEstimator, TransformerMixin):
    """
    Dimensionality reduction that MUST be fitted only on training data.

    Supports multiple methods:
    - kernel_pca: Non-linear PCA with RBF kernel
    - ica: Independent Component Analysis
    - ensemble: Combination of methods
    """

    def __init__(
        self,
        method: str = "kernel_pca",
        n_components: int = 20,
        random_state: int = 42,
    ):
        self.method = method
        self.n_components = n_components
        self.random_state = random_state

        self.reducer_ = None
        self.scaler_ = None

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Fit reducer on TRAINING data only."""

        # Always scale first (fitted on training only)
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)

        n_components = min(self.n_components, X_scaled.shape[1] - 1, X_scaled.shape[0] - 1)

        if self.method == "kernel_pca":
            from sklearn.decomposition import KernelPCA
            self.reducer_ = KernelPCA(
                n_components=n_components,
                kernel="rbf",
                gamma=0.01,
                random_state=self.random_state,
                n_jobs=-1
            )
            self.reducer_.fit(X_scaled)

        elif self.method == "ica":
            from sklearn.decomposition import FastICA
            self.reducer_ = FastICA(
                n_components=n_components,
                max_iter=500,
                random_state=self.random_state,
                whiten="unit-variance"
            )
            try:
                self.reducer_.fit(X_scaled)
            except Exception:
                # ICA can fail, fall back to PCA
                from sklearn.decomposition import PCA
                self.reducer_ = PCA(n_components=n_components, random_state=self.random_state)
                self.reducer_.fit(X_scaled)

        elif self.method == "pca":
            from sklearn.decomposition import PCA
            self.reducer_ = PCA(n_components=n_components, random_state=self.random_state)
            self.reducer_.fit(X_scaled)

        else:
            # No reduction
            self.reducer_ = None

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform using fitted reducer."""
        if self.scaler_ is None:
            raise ValueError("DimReducer not fitted. Call fit() first.")

        X_scaled = self.scaler_.transform(X)

        if self.reducer_ is not None:
            return self.reducer_.transform(X_scaled)
        return X_scaled

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)
