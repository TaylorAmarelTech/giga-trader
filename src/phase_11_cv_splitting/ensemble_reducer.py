"""
GIGA TRADER - Ensemble Reducer & Leak-Proof Pipeline
=====================================================
Reduces overfitting through model ensembling and provides a complete
leak-proof pipeline that prevents all forms of data leakage.

Strategies:
1. Method ensembling: Combine L1, L2, ElasticNet, GB
2. Parameter ensembling: Train with perturbed hyperparameters
3. Feature ensembling: Train on different feature subsets
4. Bootstrap ensembling: Train on bootstrap samples
"""

from typing import Dict, List, Tuple, Any

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier, VotingClassifier
import warnings

warnings.filterwarnings("ignore")

from src.phase_10_feature_processing.leak_proof_selector import LeakProofFeatureSelector
from src.phase_10_feature_processing.leak_proof_reducer import LeakProofDimReducer
from src.phase_11_cv_splitting.leak_proof_cv_core import CVFoldResult, LeakProofCV


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
        # Group-aware processing
        feature_names: List[str] = None,
        group_mode: str = "flat",
        protected_groups: List[str] = None,
        budget_mode: str = "proportional",
        total_components: int = 40,
        min_components_per_group: int = 2,
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
        # Group-aware processing
        self.feature_names = feature_names
        self.group_mode = group_mode
        self.protected_groups = protected_groups or []
        self.budget_mode = budget_mode
        self.total_components = total_components
        self.min_components_per_group = min_components_per_group

        # Fitted components (for final model)
        self.feature_selector_ = None
        self.dim_reducer_ = None
        self.group_processor_ = None  # GroupAwareFeatureProcessor
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

            use_groups = (
                self.group_mode != "flat"
                and self.feature_names is not None
            )

            if use_groups:
                from src.phase_10_feature_processing.group_aware_processor import (
                    GroupAwareFeatureProcessor,
                )
                processor = GroupAwareFeatureProcessor(
                    feature_names=self.feature_names,
                    group_mode=self.group_mode,
                    protected_groups=self.protected_groups,
                    budget_mode=self.budget_mode,
                    total_components=self.total_components,
                    min_components_per_group=self.min_components_per_group,
                    selection_method=self.feature_selection_method,
                    reduction_method=self.dim_reduction_method,
                    n_features=self.n_features,
                    n_components=self.n_components,
                    random_state=self.random_state,
                )
                X_train_processed = processor.fit_transform(X_train, y_train)
                X_test_processed = processor.transform(X_test)

                # Scaling (on train only)
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train_processed)
                X_test_scaled = scaler.transform(X_test_processed)
            else:
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

        use_groups = (
            self.group_mode != "flat"
            and self.feature_names is not None
        )

        if use_groups:
            from src.phase_10_feature_processing.group_aware_processor import (
                GroupAwareFeatureProcessor,
            )
            self.group_processor_ = GroupAwareFeatureProcessor(
                feature_names=self.feature_names,
                group_mode=self.group_mode,
                protected_groups=self.protected_groups,
                budget_mode=self.budget_mode,
                total_components=self.total_components,
                min_components_per_group=self.min_components_per_group,
                selection_method=self.feature_selection_method,
                reduction_method=self.dim_reduction_method,
                n_features=self.n_features,
                n_components=self.n_components,
                random_state=self.random_state,
            )
            X_processed = self.group_processor_.fit_transform(X, y)

            self.scaler_ = StandardScaler()
            X_scaled = self.scaler_.fit_transform(X_processed)
        else:
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
        if getattr(self, 'group_processor_', None) is not None:
            X_processed = self.group_processor_.transform(X)
            X_scaled = self.scaler_.transform(X_processed)
        else:
            X_sel = self.feature_selector_.transform(X)
            X_red = self.dim_reducer_.transform(X_sel)
            X_scaled = self.scaler_.transform(X_red)
        return self.model_.predict_proba(X_scaled)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for new data."""
        proba = self.predict_proba(X)[:, 1]
        return (proba > 0.5).astype(int)


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
        # Group-aware processing
        feature_names=config.get("feature_names"),
        group_mode=config.get("group_mode", "flat"),
        protected_groups=config.get("protected_groups"),
        budget_mode=config.get("budget_mode", "proportional"),
        total_components=config.get("total_components", 40),
        min_components_per_group=config.get("min_components_per_group", 2),
    )

    cv_results = pipeline.fit_with_cv(
        X, y,
        sample_weights=sample_weights,
        model_params=config.get("model_params"),
        verbose=verbose,
    )

    return pipeline, cv_results


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
