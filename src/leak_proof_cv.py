"""
GIGA TRADER - Leak-Proof Cross-Validation Pipeline
===================================================
Fixes critical data leakage issues in the anti-overfitting module:

1. Scaler fitting INSIDE CV loop (not before)
2. Feature selection INSIDE CV loop (not before)
3. Dimensionality reduction INSIDE CV loop (not before)
4. Proper model ensembling for reduced overfitting

The key insight: ANY transformation that uses statistics from the data
must be fitted ONLY on the training fold, never on test fold data.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any
from dataclasses import dataclass
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
import warnings

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════════════════════
# 1. LEAK-PROOF FEATURE SELECTOR (fits inside CV fold)
# ═══════════════════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════════════════
# 2. LEAK-PROOF DIMENSIONALITY REDUCER
# ═══════════════════════════════════════════════════════════════════════════════
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


# ═══════════════════════════════════════════════════════════════════════════════
# 3. LEAK-PROOF CROSS-VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
@dataclass
class CVFoldResult:
    """Result from one CV fold."""
    fold_idx: int
    train_auc: float
    test_auc: float
    train_size: int
    test_size: int
    model: Any
    feature_selector: Any
    dim_reducer: Any
    scaler: Any


class LeakProofCV:
    """
    Cross-validation that prevents data leakage by fitting ALL
    transformations inside each fold.

    The key principle: test fold must never influence ANY parameter
    of ANY transformation or model.
    """

    def __init__(
        self,
        n_folds: int = 5,
        purge_days: int = 5,
        embargo_days: int = 2,
        feature_selection_method: str = "mutual_info",
        n_features: int = 30,
        dim_reduction_method: str = "kernel_pca",
        n_components: int = 20,
        random_state: int = 42,
    ):
        self.n_folds = n_folds
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.feature_selection_method = feature_selection_method
        self.n_features = n_features
        self.dim_reduction_method = dim_reduction_method
        self.n_components = n_components
        self.random_state = random_state

        self.fold_results_: List[CVFoldResult] = []

    def create_purged_folds(
        self,
        n_samples: int,
        n_folds: int,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        Create time-series aware folds with purging and embargo.

        Purging: Remove samples near the split boundary
        Embargo: Don't use samples immediately after test period for training
        """
        folds = []
        fold_size = n_samples // n_folds

        for fold_idx in range(n_folds):
            # Test set is the current fold
            test_start = fold_idx * fold_size
            test_end = (fold_idx + 1) * fold_size if fold_idx < n_folds - 1 else n_samples

            test_idx = np.arange(test_start, test_end)

            # Training set is everything else, with purging
            train_mask = np.ones(n_samples, dtype=bool)
            train_mask[test_idx] = False

            # Purge: remove samples near test boundaries
            purge_before_start = max(0, test_start - self.purge_days)
            purge_after_end = min(n_samples, test_end + self.embargo_days)

            train_mask[purge_before_start:test_start] = False
            train_mask[test_end:purge_after_end] = False

            train_idx = np.where(train_mask)[0]

            if len(train_idx) > 0 and len(test_idx) > 0:
                folds.append((train_idx, test_idx))

        return folds

    def cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray = None,
        model_class: type = None,
        model_params: Dict = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Run leak-proof cross-validation.

        In each fold:
        1. Split data into train/test (with purging)
        2. Fit feature selector on TRAIN only
        3. Fit dimensionality reducer on TRAIN only
        4. Fit scaler on TRAIN only
        5. Fit model on TRAIN only
        6. Evaluate on TEST
        """
        if model_class is None:
            model_class = LogisticRegression

        if model_params is None:
            model_params = {"C": 1.0, "max_iter": 2000, "random_state": self.random_state}

        if sample_weights is None:
            sample_weights = np.ones(len(y))

        # Create folds
        folds = self.create_purged_folds(len(X), self.n_folds)

        if verbose:
            print(f"\n[LEAK-PROOF CV] {len(folds)} folds with purge={self.purge_days}, embargo={self.embargo_days}")

        self.fold_results_ = []
        train_aucs = []
        test_aucs = []

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            # Split data
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            w_train = sample_weights[train_idx]

            # Step 1: Feature selection (fitted on TRAIN only)
            feature_selector = LeakProofFeatureSelector(
                method=self.feature_selection_method,
                n_features=self.n_features,
                random_state=self.random_state,
            )
            X_train_selected = feature_selector.fit_transform(X_train, y_train)
            X_test_selected = feature_selector.transform(X_test)

            # Step 2: Dimensionality reduction (fitted on TRAIN only)
            dim_reducer = LeakProofDimReducer(
                method=self.dim_reduction_method,
                n_components=self.n_components,
                random_state=self.random_state,
            )
            X_train_reduced = dim_reducer.fit_transform(X_train_selected)
            X_test_reduced = dim_reducer.transform(X_test_selected)

            # Step 3: Final scaling (fitted on TRAIN only)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_reduced)
            X_test_scaled = scaler.transform(X_test_reduced)

            # Step 4: Train model (on TRAIN only)
            model = model_class(**model_params)
            model.fit(X_train_scaled, y_train, sample_weight=w_train)

            # Step 5: Evaluate
            try:
                train_proba = model.predict_proba(X_train_scaled)[:, 1]
                test_proba = model.predict_proba(X_test_scaled)[:, 1]

                train_auc = roc_auc_score(y_train, train_proba) if len(np.unique(y_train)) > 1 else 0.5
                test_auc = roc_auc_score(y_test, test_proba) if len(np.unique(y_test)) > 1 else 0.5
            except Exception as e:
                train_auc = 0.5
                test_auc = 0.5

            train_aucs.append(train_auc)
            test_aucs.append(test_auc)

            # Store result
            result = CVFoldResult(
                fold_idx=fold_idx,
                train_auc=train_auc,
                test_auc=test_auc,
                train_size=len(train_idx),
                test_size=len(test_idx),
                model=model,
                feature_selector=feature_selector,
                dim_reducer=dim_reducer,
                scaler=scaler,
            )
            self.fold_results_.append(result)

            if verbose:
                gap = train_auc - test_auc
                gap_warning = " [OVERFIT!]" if gap > 0.10 else ""
                print(f"  Fold {fold_idx + 1}: Train AUC={train_auc:.3f}, Test AUC={test_auc:.3f}, Gap={gap:.3f}{gap_warning}")

        # Summary statistics
        mean_train = np.mean(train_aucs)
        mean_test = np.mean(test_aucs)
        std_test = np.std(test_aucs)
        gap = mean_train - mean_test

        if verbose:
            print(f"\n  CV Summary:")
            print(f"    Mean Train AUC: {mean_train:.3f}")
            print(f"    Mean Test AUC:  {mean_test:.3f} +/- {std_test:.3f}")
            print(f"    Train-Test Gap: {gap:.3f}")

            if gap > 0.10:
                print(f"    [WARNING] High train-test gap suggests overfitting!")
            elif gap < 0.03:
                print(f"    [GOOD] Low gap suggests robust model")

        return {
            "train_aucs": train_aucs,
            "test_aucs": test_aucs,
            "mean_train_auc": mean_train,
            "mean_test_auc": mean_test,
            "std_test_auc": std_test,
            "train_test_gap": gap,
            "fold_results": self.fold_results_,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 4. MODEL ENSEMBLING (reduces overfitting per user suggestion)
# ═══════════════════════════════════════════════════════════════════════════════
class EnsembleReducer:
    """
    Reduces overfitting through model ensembling.

    Strategies:
    1. Method ensembling: Combine L1, L2, ElasticNet, GB
    2. Parameter ensembling: Train with perturbed hyperparameters
    3. Feature ensembling: Train on different feature subsets
    4. Bootstrap ensembling: Train on bootstrap samples
    """

    def __init__(
        self,
        n_estimators: int = 5,
        ensemble_method: str = "voting",  # "voting" or "stacking"
        random_state: int = 42,
    ):
        self.n_estimators = n_estimators
        self.ensemble_method = ensemble_method
        self.random_state = random_state

        self.base_models_ = []
        self.ensemble_ = None

    def create_diverse_models(
        self,
        base_params: Dict = None,
    ) -> List[Tuple[str, ClassifierMixin]]:
        """
        Create diverse base models for ensembling.

        Diversity reduces overfitting because different models
        overfit to different aspects of the data.
        """
        if base_params is None:
            base_params = {}

        models = []

        # Model 1: L2 Logistic Regression (high regularization)
        models.append((
            "l2_strong",
            LogisticRegression(
                penalty="l2",
                C=0.1,
                max_iter=2000,
                random_state=self.random_state,
            )
        ))

        # Model 2: L2 Logistic Regression (medium regularization)
        models.append((
            "l2_medium",
            LogisticRegression(
                penalty="l2",
                C=1.0,
                max_iter=2000,
                random_state=self.random_state,
            )
        ))

        # Model 3: L1 Logistic Regression (sparse)
        models.append((
            "l1_sparse",
            LogisticRegression(
                penalty="l1",
                C=0.1,
                solver="saga",
                max_iter=2000,
                random_state=self.random_state,
            )
        ))

        # Model 4: ElasticNet
        models.append((
            "elastic",
            LogisticRegression(
                penalty="elasticnet",
                C=0.5,
                l1_ratio=0.5,
                solver="saga",
                max_iter=2000,
                random_state=self.random_state,
            )
        ))

        # Model 5: Shallow Gradient Boosting
        gb_params = {
            "n_estimators": base_params.get("gb_n_estimators", 50),
            "max_depth": min(base_params.get("gb_max_depth", 3), 5),  # Never > 5
            "learning_rate": base_params.get("gb_learning_rate", 0.1),
            "min_samples_leaf": base_params.get("gb_min_samples_leaf", 50),
            "subsample": base_params.get("gb_subsample", 0.8),
            "random_state": self.random_state,
        }
        models.append((
            "gb_shallow",
            GradientBoostingClassifier(**gb_params)
        ))

        return models

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray = None,
        base_params: Dict = None,
    ):
        """
        Fit ensemble of diverse models.

        Each model is trained on the same data but with different
        inductive biases, reducing overall overfitting.
        """
        self.base_models_ = self.create_diverse_models(base_params)

        if self.ensemble_method == "voting":
            # Soft voting ensemble
            self.ensemble_ = VotingClassifier(
                estimators=self.base_models_,
                voting="soft",
                n_jobs=-1,
            )

            if sample_weights is not None:
                self.ensemble_.fit(X, y, sample_weight=sample_weights)
            else:
                self.ensemble_.fit(X, y)
        else:
            # Simple averaging (manual implementation)
            for name, model in self.base_models_:
                if sample_weights is not None:
                    model.fit(X, y, sample_weight=sample_weights)
                else:
                    model.fit(X, y)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using ensemble."""
        if self.ensemble_ is not None:
            return self.ensemble_.predict_proba(X)

        # Manual averaging
        probas = []
        for name, model in self.base_models_:
            probas.append(model.predict_proba(X)[:, 1])

        avg_proba = np.mean(probas, axis=0)
        return np.column_stack([1 - avg_proba, avg_proba])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)[:, 1]
        return (proba > 0.5).astype(int)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. COMPLETE LEAK-PROOF PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════
class LeakProofPipeline:
    """
    Complete pipeline that prevents all forms of data leakage.

    This combines:
    1. Leak-proof feature selection
    2. Leak-proof dimensionality reduction
    3. Leak-proof cross-validation
    4. Model ensembling for reduced overfitting
    """

    def __init__(
        self,
        n_cv_folds: int = 5,
        purge_days: int = 5,
        embargo_days: int = 2,
        feature_selection_method: str = "mutual_info",
        n_features: int = 30,
        dim_reduction_method: str = "kernel_pca",
        n_components: int = 20,
        use_ensemble: bool = True,
        random_state: int = 42,
    ):
        self.n_cv_folds = n_cv_folds
        self.purge_days = purge_days
        self.embargo_days = embargo_days
        self.feature_selection_method = feature_selection_method
        self.n_features = n_features
        self.dim_reduction_method = dim_reduction_method
        self.n_components = n_components
        self.use_ensemble = use_ensemble
        self.random_state = random_state

        # Fitted components (for final model)
        self.feature_selector_ = None
        self.dim_reducer_ = None
        self.scaler_ = None
        self.model_ = None

        # CV results
        self.cv_results_ = None

    def fit_with_cv(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray = None,
        model_params: Dict = None,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Fit pipeline with cross-validation.

        Returns CV metrics and trains final model on all data.
        """
        if verbose:
            print("\n" + "=" * 70)
            print("LEAK-PROOF PIPELINE: CROSS-VALIDATION")
            print("=" * 70)

        # Step 1: Run leak-proof CV
        cv = LeakProofCV(
            n_folds=self.n_cv_folds,
            purge_days=self.purge_days,
            embargo_days=self.embargo_days,
            feature_selection_method=self.feature_selection_method,
            n_features=self.n_features,
            dim_reduction_method=self.dim_reduction_method,
            n_components=self.n_components,
            random_state=self.random_state,
        )

        if self.use_ensemble:
            # Use ensemble for CV
            self.cv_results_ = self._cv_with_ensemble(cv, X, y, sample_weights, model_params, verbose)
        else:
            # Standard single model CV
            self.cv_results_ = cv.cross_validate(
                X, y,
                sample_weights=sample_weights,
                model_class=LogisticRegression,
                model_params=model_params or {"C": 1.0, "max_iter": 2000, "random_state": self.random_state},
                verbose=verbose,
            )

        # Step 2: Train final model on ALL data
        if verbose:
            print("\n[FINAL MODEL] Training on all data...")

        self._fit_final_model(X, y, sample_weights, model_params)

        return self.cv_results_

    def _cv_with_ensemble(
        self,
        cv: LeakProofCV,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray,
        model_params: Dict,
        verbose: bool,
    ) -> Dict[str, Any]:
        """Run CV with ensemble model."""
        folds = cv.create_purged_folds(len(X), cv.n_folds)

        if verbose:
            print(f"\n[ENSEMBLE CV] {len(folds)} folds with model ensembling")

        if sample_weights is None:
            sample_weights = np.ones(len(y))

        train_aucs = []
        test_aucs = []

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            w_train = sample_weights[train_idx]

            # Feature selection (on train only)
            feature_selector = LeakProofFeatureSelector(
                method=self.feature_selection_method,
                n_features=self.n_features,
                random_state=self.random_state,
            )
            X_train_sel = feature_selector.fit_transform(X_train, y_train)
            X_test_sel = feature_selector.transform(X_test)

            # Dim reduction (on train only)
            dim_reducer = LeakProofDimReducer(
                method=self.dim_reduction_method,
                n_components=self.n_components,
                random_state=self.random_state,
            )
            X_train_red = dim_reducer.fit_transform(X_train_sel)
            X_test_red = dim_reducer.transform(X_test_sel)

            # Scaling (on train only)
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_red)
            X_test_scaled = scaler.transform(X_test_red)

            # Ensemble model (on train only)
            ensemble = EnsembleReducer(random_state=self.random_state)
            ensemble.fit(X_train_scaled, y_train, w_train, model_params)

            # Evaluate
            try:
                train_proba = ensemble.predict_proba(X_train_scaled)[:, 1]
                test_proba = ensemble.predict_proba(X_test_scaled)[:, 1]

                train_auc = roc_auc_score(y_train, train_proba) if len(np.unique(y_train)) > 1 else 0.5
                test_auc = roc_auc_score(y_test, test_proba) if len(np.unique(y_test)) > 1 else 0.5
            except Exception:
                train_auc = 0.5
                test_auc = 0.5

            train_aucs.append(train_auc)
            test_aucs.append(test_auc)

            if verbose:
                gap = train_auc - test_auc
                gap_warning = " [OVERFIT!]" if gap > 0.10 else ""
                print(f"  Fold {fold_idx + 1}: Train={train_auc:.3f}, Test={test_auc:.3f}, Gap={gap:.3f}{gap_warning}")

        mean_train = np.mean(train_aucs)
        mean_test = np.mean(test_aucs)
        std_test = np.std(test_aucs)
        gap = mean_train - mean_test

        if verbose:
            print(f"\n  Ensemble CV Summary:")
            print(f"    Mean Train AUC: {mean_train:.3f}")
            print(f"    Mean Test AUC:  {mean_test:.3f} +/- {std_test:.3f}")
            print(f"    Train-Test Gap: {gap:.3f}")

        return {
            "train_aucs": train_aucs,
            "test_aucs": test_aucs,
            "mean_train_auc": mean_train,
            "mean_test_auc": mean_test,
            "std_test_auc": std_test,
            "train_test_gap": gap,
        }

    def _fit_final_model(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray,
        model_params: Dict,
    ):
        """Fit final model on all data."""
        if sample_weights is None:
            sample_weights = np.ones(len(y))

        # Feature selection
        self.feature_selector_ = LeakProofFeatureSelector(
            method=self.feature_selection_method,
            n_features=self.n_features,
            random_state=self.random_state,
        )
        X_sel = self.feature_selector_.fit_transform(X, y)

        # Dim reduction
        self.dim_reducer_ = LeakProofDimReducer(
            method=self.dim_reduction_method,
            n_components=self.n_components,
            random_state=self.random_state,
        )
        X_red = self.dim_reducer_.fit_transform(X_sel)

        # Scaling
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_red)

        # Final model
        if self.use_ensemble:
            self.model_ = EnsembleReducer(random_state=self.random_state)
            self.model_.fit(X_scaled, y, sample_weights, model_params)
        else:
            params = model_params or {"C": 1.0, "max_iter": 2000, "random_state": self.random_state}
            self.model_ = LogisticRegression(**params)
            self.model_.fit(X_scaled, y, sample_weight=sample_weights)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities for new data."""
        X_sel = self.feature_selector_.transform(X)
        X_red = self.dim_reducer_.transform(X_sel)
        X_scaled = self.scaler_.transform(X_red)
        return self.model_.predict_proba(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for new data."""
        proba = self.predict_proba(X)[:, 1]
        return (proba > 0.5).astype(int)


# ═══════════════════════════════════════════════════════════════════════════════
# 6. INTEGRATION FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════
def train_with_leak_proof_cv(
    X: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray = None,
    config: Dict = None,
    verbose: bool = True,
) -> Tuple[LeakProofPipeline, Dict]:
    """
    Train models using leak-proof cross-validation.

    This is the main entry point for training with proper data leakage prevention.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target labels (n_samples,)
        sample_weights: Optional sample weights
        config: Configuration dict with CV/model parameters
        verbose: Print progress

    Returns:
        Tuple of (fitted_pipeline, cv_results)
    """
    if config is None:
        config = {}

    pipeline = LeakProofPipeline(
        n_cv_folds=config.get("n_cv_folds", 5),
        purge_days=config.get("purge_days", 5),
        embargo_days=config.get("embargo_days", 2),
        feature_selection_method=config.get("feature_selection_method", "mutual_info"),
        n_features=config.get("n_features", 30),
        dim_reduction_method=config.get("dim_reduction_method", "kernel_pca"),
        n_components=config.get("n_components", 20),
        use_ensemble=config.get("use_ensemble", True),
        random_state=config.get("random_state", 42),
    )

    cv_results = pipeline.fit_with_cv(
        X, y,
        sample_weights=sample_weights,
        model_params=config.get("model_params"),
        verbose=verbose,
    )

    return pipeline, cv_results


# ═══════════════════════════════════════════════════════════════════════════════
# 7. TESTING / VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 70)
    print("LEAK-PROOF CV - VERIFICATION TEST")
    print("=" * 70)

    # Create synthetic data with known pattern
    np.random.seed(42)
    n_samples = 1000
    n_features = 100

    # Features
    X = np.random.randn(n_samples, n_features)

    # Target with some signal in first few features
    signal = 0.3 * X[:, 0] + 0.2 * X[:, 1] + 0.1 * X[:, 2]
    noise = np.random.randn(n_samples) * 0.5
    y = (signal + noise > 0).astype(int)

    print(f"\nTest data: {n_samples} samples, {n_features} features")
    print(f"Target balance: {y.mean():.2%} positive")

    # Test leak-proof pipeline
    config = {
        "n_cv_folds": 5,
        "purge_days": 5,
        "embargo_days": 2,
        "feature_selection_method": "mutual_info",
        "n_features": 20,
        "dim_reduction_method": "pca",
        "n_components": 10,
        "use_ensemble": True,
    }

    pipeline, results = train_with_leak_proof_cv(X, y, config=config, verbose=True)

    print("\n" + "=" * 70)
    print("VERIFICATION COMPLETE")
    print("=" * 70)

    gap = results["train_test_gap"]
    if gap < 0.10:
        print(f"[PASS] Train-test gap ({gap:.3f}) is acceptable")
    else:
        print(f"[WARN] Train-test gap ({gap:.3f}) suggests potential overfitting")

    # Compare to leaky version (for demonstration)
    print("\n[INFO] Expected behavior:")
    print("  - Leak-proof CV should show LOWER test AUC than leaky version")
    print("  - But the test AUC is MORE RELIABLE for out-of-sample performance")
    print("  - Train-test gap should be smaller with leak-proof approach")
