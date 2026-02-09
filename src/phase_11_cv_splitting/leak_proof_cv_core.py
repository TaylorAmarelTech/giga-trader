"""
GIGA TRADER - Leak-Proof Cross-Validation Core
================================================
Cross-validation that prevents data leakage by fitting ALL
transformations inside each fold.

The key principle: test fold must never influence ANY parameter
of ANY transformation or model.
"""

from typing import Dict, List, Tuple, Any
from dataclasses import dataclass

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

from src.phase_10_feature_processing.leak_proof_selector import LeakProofFeatureSelector
from src.phase_10_feature_processing.leak_proof_reducer import LeakProofDimReducer


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
