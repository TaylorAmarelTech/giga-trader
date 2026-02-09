"""
GIGA TRADER - Robustness Ensemble (Dimension + Parameter Perturbation)
=======================================================================
Creates an ensemble of models trained with perturbed dimensions and parameters.

Strategy:
  1. Train base model with optimal n_dimensions and parameters
  2. Train "adjacent" models with n-1 and n+1 dimensions
  3. Train models with perturbed parameters (+/- noise)
  4. Ensemble all models with optional weighting
  5. If ensemble performance drops drastically, solution is fragile

This reduces overfitting to specific dimension counts or parameter values.
"""

import os
import sys
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


class RobustnessEnsemble:
    """
    Creates an ensemble of models trained with perturbed dimensions and parameters.

    Strategy:
      1. Train base model with optimal n_dimensions and parameters
      2. Train "adjacent" models with n-1 and n+1 dimensions
      3. Train models with perturbed parameters (+/- noise)
      4. Ensemble all models with optional weighting
      5. If ensemble performance drops drastically, solution is fragile

    This reduces overfitting to specific dimension counts or parameter values.
    """

    def __init__(
        self,
        n_dimension_variants: int = 2,  # +/- this many dimensions
        n_param_variants: int = 2,  # Number of parameter perturbations
        param_noise_pct: float = 0.05,  # +/- 5% parameter noise
        center_weight: float = 0.5,  # Weight for center/optimal model
        adjacent_weight: float = 0.25,  # Weight for adjacent models (split)
    ):
        self.n_dimension_variants = n_dimension_variants
        self.n_param_variants = n_param_variants
        self.param_noise_pct = param_noise_pct
        self.center_weight = center_weight
        self.adjacent_weight = adjacent_weight

        self.models = {}  # Store all trained models
        self.weights = {}  # Store model weights
        self.fragility_score = None

    def create_dimension_variants(
        self,
        optimal_dims: int,
        min_dims: int = 5,
        max_dims: int = 100,
    ) -> List[int]:
        """
        Create list of dimension counts to try.

        Returns: [n-2, n-1, n, n+1, n+2] (within bounds)
        """
        variants = []

        for delta in range(-self.n_dimension_variants, self.n_dimension_variants + 1):
            n = optimal_dims + delta
            if min_dims <= n <= max_dims:
                variants.append(n)

        # Ensure we have at least 3 variants
        if len(variants) < 3:
            variants = [max(min_dims, optimal_dims - 1), optimal_dims, min(max_dims, optimal_dims + 1)]

        return sorted(set(variants))

    def create_parameter_variants(
        self,
        base_params: Dict,
        param_ranges: Dict[str, Tuple[float, float]] = None,
    ) -> List[Dict]:
        """
        Create parameter variants with noise.

        Returns list of parameter dicts including base + perturbed versions.
        """
        variants = [base_params.copy()]  # Always include base

        if param_ranges is None:
            # Default ranges for common parameters
            param_ranges = {
                "C": (0.01, 100.0),
                "l2_C": (0.01, 100.0),
                "n_estimators": (10, 500),
                "gb_n_estimators": (10, 500),
                "max_depth": (1, 10),
                "gb_max_depth": (1, 10),
                "learning_rate": (0.001, 1.0),
                "gb_learning_rate": (0.001, 1.0),
                "min_samples_leaf": (1, 200),
                "gb_min_samples_leaf": (1, 200),
            }

        for _ in range(self.n_param_variants):
            perturbed = base_params.copy()

            for param, value in base_params.items():
                if param not in param_ranges:
                    continue

                low, high = param_ranges[param]

                # Add random noise within +/- noise_pct
                noise = np.random.uniform(-self.param_noise_pct, self.param_noise_pct)
                new_value = value * (1 + noise)

                # Clip to valid range
                new_value = max(low, min(high, new_value))

                # Keep integers as integers
                if isinstance(value, int):
                    new_value = int(round(new_value))

                perturbed[param] = new_value

            variants.append(perturbed)

        return variants

    def train_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray = None,
        base_params: Dict = None,
        dim_reduction_fn=None,
        optimal_dims: int = 30,
        model_class=None,
        cv_folds: int = 3,
    ) -> Dict:
        """
        Train ensemble of models with dimension and parameter perturbations.

        Args:
            X: Feature matrix
            y: Target labels
            sample_weights: Sample weights
            base_params: Optimal parameters found via Optuna
            dim_reduction_fn: Function that reduces X to n dimensions
            optimal_dims: Optimal number of dimensions
            model_class: Sklearn model class to use
            cv_folds: Cross-validation folds for scoring

        Returns:
            Dict with models, weights, and fragility analysis
        """
        from sklearn.model_selection import cross_val_score
        from sklearn.linear_model import LogisticRegression

        if model_class is None:
            model_class = LogisticRegression

        if base_params is None:
            base_params = {"C": 1.0, "max_iter": 500, "random_state": 42}

        print("\n" + "=" * 70)
        print("ROBUSTNESS ENSEMBLE TRAINING")
        print("=" * 70)

        results = {
            "models": {},
            "scores": {},
            "weights": {},
            "dim_variants": [],
            "param_variants": [],
        }

        # ─────────────────────────────────────────────────────────────────────
        # 1. DIMENSION VARIANTS
        # ─────────────────────────────────────────────────────────────────────
        dim_variants = self.create_dimension_variants(optimal_dims)
        results["dim_variants"] = dim_variants
        print(f"\n[DIM VARIANTS] Testing dimensions: {dim_variants}")

        dim_scores = {}

        for n_dims in dim_variants:
            try:
                # Reduce dimensions if function provided
                if dim_reduction_fn is not None:
                    X_reduced = dim_reduction_fn(X, n_dims)
                else:
                    # Simple PCA fallback
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=min(n_dims, X.shape[1]))
                    X_reduced = pca.fit_transform(X)

                # Train model with base params
                model_params = {k: v for k, v in base_params.items()
                               if k in ["C", "max_iter", "random_state", "penalty", "solver"]}
                model = model_class(**model_params)

                # Cross-validate
                # Note: fit_params removed in sklearn 1.4+, use params instead
                cv_kwargs = {"estimator": model, "X": X_reduced, "y": y, "cv": cv_folds, "scoring": "roc_auc"}
                if sample_weights is not None:
                    cv_kwargs["params"] = {"sample_weight": sample_weights}
                scores = cross_val_score(**cv_kwargs)
                mean_score = scores.mean()
                dim_scores[n_dims] = mean_score

                # Train final model on full data
                model.fit(X_reduced, y, sample_weight=sample_weights)

                model_key = f"dim_{n_dims}"
                results["models"][model_key] = {
                    "model": model,
                    "n_dims": n_dims,
                    "cv_score": mean_score,
                    "type": "dimension_variant",
                }
                results["scores"][model_key] = mean_score

                is_optimal = " (OPTIMAL)" if n_dims == optimal_dims else ""
                print(f"  dim={n_dims}: AUC={mean_score:.4f}{is_optimal}")

            except Exception as e:
                print(f"  dim={n_dims}: FAILED - {e}")
                continue

        # ─────────────────────────────────────────────────────────────────────
        # 2. PARAMETER VARIANTS (using optimal dimensions)
        # ─────────────────────────────────────────────────────────────────────
        param_variants = self.create_parameter_variants(base_params)
        results["param_variants"] = param_variants
        print(f"\n[PARAM VARIANTS] Testing {len(param_variants)} parameter sets")

        # Use optimal dimension reduction
        if dim_reduction_fn is not None:
            X_optimal = dim_reduction_fn(X, optimal_dims)
        else:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(optimal_dims, X.shape[1]))
            X_optimal = pca.fit_transform(X)

        for i, params in enumerate(param_variants):
            try:
                model_params = {k: v for k, v in params.items()
                               if k in ["C", "max_iter", "random_state", "penalty", "solver"]}
                model = model_class(**model_params)

                # Note: fit_params removed in sklearn 1.4+, use params instead
                cv_kwargs = {"estimator": model, "X": X_optimal, "y": y, "cv": cv_folds, "scoring": "roc_auc"}
                if sample_weights is not None:
                    cv_kwargs["params"] = {"sample_weight": sample_weights}
                scores = cross_val_score(**cv_kwargs)
                mean_score = scores.mean()

                model.fit(X_optimal, y, sample_weight=sample_weights)

                model_key = f"param_{i}"
                results["models"][model_key] = {
                    "model": model,
                    "params": params,
                    "cv_score": mean_score,
                    "type": "parameter_variant",
                }
                results["scores"][model_key] = mean_score

                is_base = " (BASE)" if i == 0 else ""
                print(f"  variant_{i}: AUC={mean_score:.4f}{is_base}")

            except Exception as e:
                print(f"  variant_{i}: FAILED - {e}")
                continue

        # ─────────────────────────────────────────────────────────────────────
        # 3. COMPUTE WEIGHTS
        # ─────────────────────────────────────────────────────────────────────
        self._compute_weights(results, optimal_dims)

        # ─────────────────────────────────────────────────────────────────────
        # 4. FRAGILITY ANALYSIS
        # ─────────────────────────────────────────────────────────────────────
        fragility = self._analyze_fragility(results, optimal_dims)
        results["fragility"] = fragility

        # Store for later use
        self.models = results["models"]
        self.weights = results["weights"]
        self.fragility_score = fragility["fragility_score"]

        print(f"\n[ENSEMBLE] Trained {len(results['models'])} models")
        print(f"  Fragility Score: {fragility['fragility_score']:.3f} (0=robust, 1=fragile)")

        if fragility["fragility_score"] > 0.3:
            print("  [WARN] High fragility detected - solution may be overfit!")
        else:
            print("  [GOOD] Low fragility - solution appears robust")

        return results

    def _compute_weights(self, results: Dict, optimal_dims: int):
        """Compute ensemble weights based on configuration and scores."""
        weights = {}

        # Dimension variant weights
        dim_models = {k: v for k, v in results["models"].items() if v["type"] == "dimension_variant"}

        if len(dim_models) > 0:
            # Optimal dim gets center_weight, others split adjacent_weight
            total_dim_weight = self.center_weight + self.adjacent_weight
            n_adjacent = len(dim_models) - 1

            for key, model_info in dim_models.items():
                if model_info["n_dims"] == optimal_dims:
                    weights[key] = self.center_weight / total_dim_weight
                else:
                    weights[key] = (self.adjacent_weight / n_adjacent) / total_dim_weight if n_adjacent > 0 else 0

        # Parameter variant weights (base gets higher weight)
        param_models = {k: v for k, v in results["models"].items() if v["type"] == "parameter_variant"}

        if len(param_models) > 0:
            total_param_weight = self.center_weight + self.adjacent_weight
            n_perturbed = len(param_models) - 1

            for i, (key, model_info) in enumerate(param_models.items()):
                if i == 0:  # Base params
                    weights[key] = self.center_weight / total_param_weight
                else:
                    weights[key] = (self.adjacent_weight / n_perturbed) / total_param_weight if n_perturbed > 0 else 0

        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        results["weights"] = weights

    def _analyze_fragility(self, results: Dict, optimal_dims: int) -> Dict:
        """
        Analyze if the solution is fragile.

        Fragility indicators:
          - Large score drop with adjacent dimensions
          - Large score drop with parameter perturbation
          - High variance across variants
        """
        scores = list(results["scores"].values())

        if len(scores) < 2:
            return {
                "fragility_score": 0.5,
                "reason": "Not enough variants",
                "interpretation": "MODERATE - Insufficient variants for robust analysis"
            }

        # Get optimal model score
        optimal_key = f"dim_{optimal_dims}"
        base_key = "param_0"

        optimal_score = results["scores"].get(optimal_key, max(scores))
        base_score = results["scores"].get(base_key, optimal_score)
        best_score = max(optimal_score, base_score)

        # 1. Score variance across all variants
        score_variance = np.var(scores)
        score_std = np.std(scores)

        # 2. Max drop from optimal
        min_score = min(scores)
        max_drop = best_score - min_score

        # 3. Dimension sensitivity
        dim_scores = [v["cv_score"] for k, v in results["models"].items()
                     if v["type"] == "dimension_variant"]
        dim_sensitivity = np.std(dim_scores) if len(dim_scores) > 1 else 0

        # 4. Parameter sensitivity
        param_scores = [v["cv_score"] for k, v in results["models"].items()
                       if v["type"] == "parameter_variant"]
        param_sensitivity = np.std(param_scores) if len(param_scores) > 1 else 0

        # Compute fragility score (0 = robust, 1 = very fragile)
        # Normalize factors to 0-1 range
        variance_factor = min(score_variance * 100, 1.0)  # High variance = fragile
        drop_factor = min(max_drop / (best_score + 1e-6), 1.0)  # Large drop = fragile
        dim_factor = min(dim_sensitivity * 10, 1.0)  # High dim sensitivity = fragile
        param_factor = min(param_sensitivity * 10, 1.0)  # High param sensitivity = fragile

        fragility_score = (
            0.3 * variance_factor +
            0.3 * drop_factor +
            0.2 * dim_factor +
            0.2 * param_factor
        )

        fragility = {
            "fragility_score": fragility_score,
            "score_variance": score_variance,
            "score_std": score_std,
            "max_drop": max_drop,
            "best_score": best_score,
            "min_score": min_score,
            "dim_sensitivity": dim_sensitivity,
            "param_sensitivity": param_sensitivity,
            "interpretation": self._interpret_fragility(fragility_score),
        }

        return fragility

    def _interpret_fragility(self, score: float) -> str:
        """Interpret fragility score."""
        if score < 0.15:
            return "VERY_ROBUST - Solution is stable across perturbations"
        elif score < 0.25:
            return "ROBUST - Minor sensitivity, acceptable"
        elif score < 0.35:
            return "MODERATE - Some sensitivity, proceed with caution"
        elif score < 0.50:
            return "FRAGILE - Significant sensitivity, likely overfit"
        else:
            return "VERY_FRAGILE - Unstable solution, high overfit risk"

    def predict_ensemble(
        self,
        X: np.ndarray,
        dim_reduction_fn=None,
        use_proba: bool = True,
    ) -> np.ndarray:
        """
        Make predictions using the ensemble.

        Args:
            X: Feature matrix
            dim_reduction_fn: Function to reduce dimensions
            use_proba: Return probabilities (True) or hard predictions (False)

        Returns:
            Weighted ensemble predictions
        """
        if len(self.models) == 0:
            raise ValueError("No models trained. Call train_ensemble first.")

        predictions = []
        weights = []

        for key, model_info in self.models.items():
            model = model_info["model"]
            weight = self.weights.get(key, 1.0 / len(self.models))

            # Apply appropriate dimension reduction
            if model_info["type"] == "dimension_variant":
                n_dims = model_info["n_dims"]
                if dim_reduction_fn is not None:
                    X_reduced = dim_reduction_fn(X, n_dims)
                else:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=min(n_dims, X.shape[1]))
                    X_reduced = pca.fit_transform(X)
            else:
                # Parameter variants use optimal dims
                n_dims = self.models.get("dim_30", {}).get("n_dims", 30)
                if dim_reduction_fn is not None:
                    X_reduced = dim_reduction_fn(X, n_dims)
                else:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=min(n_dims, X.shape[1]))
                    X_reduced = pca.fit_transform(X)

            try:
                if use_proba and hasattr(model, "predict_proba"):
                    pred = model.predict_proba(X_reduced)[:, 1]
                else:
                    pred = model.predict(X_reduced)

                predictions.append(pred)
                weights.append(weight)
            except Exception as e:
                print(f"  [WARN] Prediction failed for {key}: {e}")
                continue

        if len(predictions) == 0:
            raise ValueError("All model predictions failed")

        # Weighted average
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

        ensemble_pred = np.average(predictions, axis=0, weights=weights)

        return ensemble_pred

    def evaluate_ensemble(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        dim_reduction_fn=None,
    ) -> Dict:
        """
        Evaluate ensemble vs individual models.

        Returns comparison metrics.
        """
        from sklearn.metrics import roc_auc_score, accuracy_score

        results = {"individual": {}, "ensemble": {}}

        # Ensemble prediction
        ensemble_proba = self.predict_ensemble(X_test, dim_reduction_fn, use_proba=True)
        ensemble_pred = (ensemble_proba > 0.5).astype(int)

        try:
            ensemble_auc = roc_auc_score(y_test, ensemble_proba)
            ensemble_acc = accuracy_score(y_test, ensemble_pred)
        except (ValueError, TypeError):
            ensemble_auc = 0.5
            ensemble_acc = 0.5

        results["ensemble"] = {
            "auc": ensemble_auc,
            "accuracy": ensemble_acc,
        }

        # Compare to best individual
        if self.models:
            best_individual_auc = max(self.models[k]["cv_score"] for k in self.models)
        else:
            best_individual_auc = 0.5  # Default when no models

        results["comparison"] = {
            "ensemble_auc": ensemble_auc,
            "best_individual_auc": best_individual_auc,
            "improvement": ensemble_auc - best_individual_auc,
            "ensemble_is_better": ensemble_auc >= best_individual_auc,
        }

        print(f"\n[ENSEMBLE EVALUATION]")
        print(f"  Ensemble AUC: {ensemble_auc:.4f}")
        print(f"  Best Individual AUC: {best_individual_auc:.4f}")
        print(f"  Improvement: {(ensemble_auc - best_individual_auc):.4f}")

        return results


def create_robustness_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray = None,
    base_params: Dict = None,
    optimal_dims: int = 30,
    n_dim_variants: int = 2,
    n_param_variants: int = 2,
    param_noise_pct: float = 0.05,
) -> Tuple[RobustnessEnsemble, Dict]:
    """
    Convenience function to create and train a robustness ensemble.

    Returns:
        Tuple of (RobustnessEnsemble instance, training results)
    """
    ensemble = RobustnessEnsemble(
        n_dimension_variants=n_dim_variants,
        n_param_variants=n_param_variants,
        param_noise_pct=param_noise_pct,
    )

    results = ensemble.train_ensemble(
        X=X,
        y=y,
        sample_weights=sample_weights,
        base_params=base_params,
        optimal_dims=optimal_dims,
    )

    return ensemble, results
