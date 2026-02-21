"""
GIGA TRADER - Multi-Faceted Stability Suite (Wave 29)
=====================================================
Augments the existing HP perturbation stability (StabilityAnalyzer) with
four complementary stability measures:

1. Bootstrap Stability   — sensitivity to training data composition
2. Feature Dropout       — sensitivity to individual feature removal
3. Seed Stability        — sensitivity to random initialization
4. Prediction Agreement  — consistency of predictions across perturbations

Each measure produces a 0-1 score (1 = maximally stable). These scores are
tracked as metadata alongside the existing stability_score and do NOT replace
the current tier gating logic.
"""

import warnings

warnings.filterwarnings("ignore")

import numpy as np
from typing import Callable, Dict, List, Optional
from sklearn.model_selection import cross_val_score, StratifiedKFold


class StabilitySuite:
    """
    Multi-faceted stability measurement system.

    Runs 4 complementary stability tests and produces individual scores
    plus a composite score. Each test measures a different axis of
    model robustness.

    Parameters
    ----------
    n_bootstrap : int
        Number of bootstrap data subsamples to test (default 5).
    n_feature_dropout : int
        Number of random feature dropout trials (default 5).
    n_seeds : int
        Number of random seed variations to test (default 5).
    n_prediction_models : int
        Number of perturbed models for prediction agreement (default 5).
    dropout_fraction : float
        Fraction of features to drop per trial (default 0.15 = 15%).
    subsample_fraction : float
        Fraction of training data per bootstrap sample (default 0.80).
    """

    def __init__(
        self,
        n_bootstrap: int = 5,
        n_feature_dropout: int = 5,
        n_seeds: int = 5,
        n_prediction_models: int = 5,
        dropout_fraction: float = 0.15,
        subsample_fraction: float = 0.80,
    ):
        self.n_bootstrap = n_bootstrap
        self.n_feature_dropout = n_feature_dropout
        self.n_seeds = n_seeds
        self.n_prediction_models = n_prediction_models
        self.dropout_fraction = dropout_fraction
        self.subsample_fraction = subsample_fraction

    def run_all(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_factory_fn: Callable[[], object],
        base_auc: float,
        seed_model_factory_fn: Optional[Callable[[int], object]] = None,
    ) -> Dict:
        """
        Run all stability measures.

        Parameters
        ----------
        X : np.ndarray
            Training features (n_samples, n_features).
        y : np.ndarray
            Training labels.
        model_factory_fn : callable
            Returns a fresh model instance (same config, random_state=42).
        base_auc : float
            The base model's CV AUC for reference.
        seed_model_factory_fn : callable, optional
            Takes an int seed, returns a model with that random_state.
            If None, seed stability is skipped.

        Returns
        -------
        Dict with keys: bootstrap, feature_dropout, seed, prediction,
        composite, and per-measure diagnostics.
        """
        results = {}

        # 1. Bootstrap stability
        results["bootstrap"] = self._bootstrap_stability(
            X, y, model_factory_fn, base_auc
        )

        # 2. Feature dropout stability
        results["feature_dropout"] = self._feature_dropout_stability(
            X, y, model_factory_fn, base_auc
        )

        # 3. Seed stability
        if seed_model_factory_fn is not None:
            results["seed"] = self._seed_stability(
                X, y, seed_model_factory_fn
            )
        else:
            results["seed"] = {"score": -1.0, "skipped": True}

        # 4. Prediction agreement
        results["prediction"] = self._prediction_agreement(
            X, y, model_factory_fn
        )

        # Composite score (average of non-skipped measures)
        scores = [
            results[k]["score"]
            for k in ("bootstrap", "feature_dropout", "seed", "prediction")
            if results[k].get("score", -1) >= 0
        ]
        results["composite"] = float(np.mean(scores)) if scores else 0.0

        return results

    def _bootstrap_stability(
        self, X: np.ndarray, y: np.ndarray,
        model_factory_fn: Callable, base_auc: float,
    ) -> Dict:
        """
        Test sensitivity to training data composition.

        Train on random 80% subsets, measure AUC variance.
        Low variance = model doesn't depend on specific training examples.
        """
        n_samples = len(y)
        subsample_size = int(n_samples * self.subsample_fraction)
        aucs = []

        for i in range(self.n_bootstrap):
            rng = np.random.RandomState(42 + i)
            indices = rng.choice(n_samples, size=subsample_size, replace=False)
            X_sub, y_sub = X[indices], y[indices]

            # Need at least 2 classes for CV
            if len(np.unique(y_sub)) < 2:
                continue

            try:
                model = model_factory_fn()
                scores = cross_val_score(
                    model, X_sub, y_sub, cv=3, scoring="roc_auc"
                )
                aucs.append(float(scores.mean()))
            except Exception:
                pass

        if len(aucs) < 2:
            return {"score": 0.5, "n_trials": len(aucs), "aucs": aucs}

        # Stability: inverse of relative AUC variance
        avg_change = float(np.mean(np.abs(np.array(aucs) - base_auc)))
        sensitivity = avg_change / (base_auc + 1e-6)
        score = float(max(0.0, 1.0 - sensitivity * 15))

        return {
            "score": score,
            "n_trials": len(aucs),
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "avg_change": avg_change,
        }

    def _feature_dropout_stability(
        self, X: np.ndarray, y: np.ndarray,
        model_factory_fn: Callable, base_auc: float,
    ) -> Dict:
        """
        Test sensitivity to individual feature removal.

        Randomly drop 15% of features, retrain, measure AUC drop.
        Low drop = model doesn't over-rely on any single feature.
        """
        n_features = X.shape[1]
        n_drop = max(1, int(n_features * self.dropout_fraction))
        aucs = []

        for i in range(self.n_feature_dropout):
            rng = np.random.RandomState(100 + i)
            keep_mask = np.ones(n_features, dtype=bool)
            drop_indices = rng.choice(n_features, size=n_drop, replace=False)
            keep_mask[drop_indices] = False

            X_dropped = X[:, keep_mask]
            if X_dropped.shape[1] < 3:
                continue

            try:
                model = model_factory_fn()
                scores = cross_val_score(
                    model, X_dropped, y, cv=3, scoring="roc_auc"
                )
                aucs.append(float(scores.mean()))
            except Exception:
                pass

        if len(aucs) < 2:
            return {"score": 0.5, "n_trials": len(aucs), "aucs": aucs}

        avg_change = float(np.mean(np.abs(np.array(aucs) - base_auc)))
        sensitivity = avg_change / (base_auc + 1e-6)
        score = float(max(0.0, 1.0 - sensitivity * 15))

        return {
            "score": score,
            "n_trials": len(aucs),
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "avg_drop": float(base_auc - np.mean(aucs)),
            "max_drop": float(base_auc - min(aucs)),
        }

    def _seed_stability(
        self, X: np.ndarray, y: np.ndarray,
        seed_model_factory_fn: Callable[[int], object],
    ) -> Dict:
        """
        Test sensitivity to random initialization.

        Same config, different random seeds. High variance = initialization-dependent.
        Meaningful for MLPs, SGD, bagged ensembles; less so for deterministic models.
        """
        aucs = []

        for seed in range(self.n_seeds):
            try:
                model = seed_model_factory_fn(seed * 17 + 7)  # Spread seeds
                scores = cross_val_score(
                    model, X, y, cv=3, scoring="roc_auc"
                )
                aucs.append(float(scores.mean()))
            except Exception:
                pass

        if len(aucs) < 2:
            return {"score": 0.5, "n_trials": len(aucs), "aucs": aucs}

        auc_std = float(np.std(aucs))
        # Low std = stable across seeds
        # Scale: std=0 → score=1.0, std=0.02 → score=0.7, std=0.05 → score=0.25
        score = float(max(0.0, 1.0 - auc_std * 15))

        return {
            "score": score,
            "n_trials": len(aucs),
            "auc_mean": float(np.mean(aucs)),
            "auc_std": auc_std,
            "auc_range": float(max(aucs) - min(aucs)),
        }

    def _prediction_agreement(
        self, X: np.ndarray, y: np.ndarray,
        model_factory_fn: Callable,
    ) -> Dict:
        """
        Test prediction consistency across bootstrap-trained models.

        Train multiple models on different data subsets, compare their
        predicted probabilities on a held-out validation set.
        High correlation = models agree on individual predictions.
        """
        n_samples = len(y)
        val_size = max(50, int(n_samples * 0.2))
        train_size = n_samples - val_size

        # Fixed validation set
        rng = np.random.RandomState(999)
        all_indices = rng.permutation(n_samples)
        val_indices = all_indices[:val_size]
        train_pool = all_indices[val_size:]

        X_val, y_val = X[val_indices], y[val_indices]

        # Need both classes in validation
        if len(np.unique(y_val)) < 2:
            return {"score": 0.5, "n_models": 0, "reason": "single_class_val"}

        predictions = []

        for i in range(self.n_prediction_models):
            rng_i = np.random.RandomState(200 + i)
            subsample_size = int(len(train_pool) * self.subsample_fraction)
            train_idx = rng_i.choice(
                train_pool, size=subsample_size, replace=False
            )
            X_train_i, y_train_i = X[train_idx], y[train_idx]

            if len(np.unique(y_train_i)) < 2:
                continue

            try:
                model = model_factory_fn()
                model.fit(X_train_i, y_train_i)
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(X_val)[:, 1]
                elif hasattr(model, "decision_function"):
                    proba = model.decision_function(X_val)
                else:
                    proba = model.predict(X_val).astype(float)
                predictions.append(proba)
            except Exception:
                pass

        if len(predictions) < 2:
            return {"score": 0.5, "n_models": len(predictions)}

        # Pairwise correlation between prediction vectors
        pred_matrix = np.array(predictions)  # (n_models, n_val)
        correlations = []
        for i in range(len(predictions)):
            for j in range(i + 1, len(predictions)):
                # Handle constant predictions
                std_i = np.std(pred_matrix[i])
                std_j = np.std(pred_matrix[j])
                if std_i < 1e-10 or std_j < 1e-10:
                    correlations.append(1.0 if std_i < 1e-10 and std_j < 1e-10 else 0.5)
                else:
                    corr = float(np.corrcoef(pred_matrix[i], pred_matrix[j])[0, 1])
                    if np.isnan(corr):
                        corr = 0.5
                    correlations.append(corr)

        mean_corr = float(np.mean(correlations))
        # Correlation 0.95+ = very stable, 0.70 = moderate, <0.50 = unstable
        # Map: corr=1.0 → score=1.0, corr=0.8 → score=0.70, corr=0.5 → score=0.25
        score = float(max(0.0, min(1.0, (mean_corr - 0.2) / 0.8)))

        # Also compute prediction std per sample (how much models disagree)
        pred_std_per_sample = float(np.mean(np.std(pred_matrix, axis=0)))

        return {
            "score": score,
            "n_models": len(predictions),
            "mean_correlation": mean_corr,
            "min_correlation": float(min(correlations)),
            "pred_std_mean": pred_std_per_sample,
        }
