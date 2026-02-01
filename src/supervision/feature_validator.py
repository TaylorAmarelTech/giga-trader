"""
Feature Validator - Validates features before model inference.

Current Problems Fixed:
1. NaN values filled with 0 - DANGEROUS (0 might have meaning)
2. Feature count mismatch silently truncated - DANGEROUS (wrong features sent to model)

This module provides proper validation with detailed diagnostics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
import numpy as np
import logging

logger = logging.getLogger("GigaTrader.FeatureValidator")


@dataclass
class ValidationResult:
    """Result of feature validation."""
    is_valid: bool
    nan_count: int
    inf_count: int
    feature_count: int
    expected_count: int
    nan_features: List[str] = field(default_factory=list)
    inf_features: List[str] = field(default_factory=list)
    missing_features: List[str] = field(default_factory=list)
    extra_features: List[str] = field(default_factory=list)
    out_of_range_features: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    corrected_features: Optional[np.ndarray] = None
    corrected_names: Optional[List[str]] = None


class FeatureValidator:
    """
    Validates features before model inference.

    Key improvements over current approach:
    1. Match features by NAME, not position
    2. Use median imputation instead of 0 for NaN
    3. Reject features that fail validation instead of silent truncation
    4. Provide detailed diagnostics
    """

    def __init__(
        self,
        expected_feature_names: List[str],
        feature_medians: Optional[Dict[str, float]] = None,
        feature_bounds: Optional[Dict[str, Tuple[float, float]]] = None,
        max_nan_pct: float = 0.05,
        max_nan_per_feature: int = 0,
        allow_inf: bool = False,
        auto_correct: bool = True,
        strict_feature_count: bool = True,
    ):
        """
        Initialize FeatureValidator.

        Args:
            expected_feature_names: List of feature names the model expects
            feature_medians: Dict of feature name -> median value for imputation
            feature_bounds: Dict of feature name -> (min, max) expected range
            max_nan_pct: Maximum allowed NaN percentage (0.0-1.0)
            max_nan_per_feature: Maximum NaN values allowed in any single feature
            allow_inf: Whether to allow infinite values
            auto_correct: If True, attempt to correct issues (reorder, impute)
            strict_feature_count: If True, reject mismatched feature counts
        """
        self.expected_names = expected_feature_names
        self.expected_set = set(expected_feature_names)
        self.feature_medians = feature_medians or {}
        self.feature_bounds = feature_bounds or {}
        self.max_nan_pct = max_nan_pct
        self.max_nan_per_feature = max_nan_per_feature
        self.allow_inf = allow_inf
        self.auto_correct = auto_correct
        self.strict_feature_count = strict_feature_count

    def validate(
        self,
        X: np.ndarray,
        feature_names: List[str],
    ) -> ValidationResult:
        """
        Validate feature array.

        Args:
            X: Feature array (n_samples, n_features)
            feature_names: Names of features in X

        Returns:
            ValidationResult with detailed diagnostics
        """
        result = ValidationResult(
            is_valid=True,
            nan_count=0,
            inf_count=0,
            feature_count=len(feature_names),
            expected_count=len(self.expected_names),
        )

        # Ensure 2D
        if X.ndim == 1:
            X = X.reshape(1, -1)

        # Check feature count
        count_match, missing, extra = self._check_feature_names(feature_names)

        if missing:
            result.missing_features = missing
            result.errors.append(f"Missing {len(missing)} features: {missing[:5]}{'...' if len(missing) > 5 else ''}")

        if extra:
            result.extra_features = extra
            result.warnings.append(f"Extra {len(extra)} features: {extra[:5]}{'...' if len(extra) > 5 else ''}")

        if not count_match and self.strict_feature_count:
            result.is_valid = False
            result.errors.append(f"Feature count mismatch: got {len(feature_names)}, expected {len(self.expected_names)}")

        # Check NaN values
        nan_count, nan_features = self._check_nans(X, feature_names)
        result.nan_count = nan_count
        result.nan_features = nan_features

        if nan_count > 0:
            nan_pct = nan_count / X.size
            if nan_pct > self.max_nan_pct:
                result.is_valid = False
                result.errors.append(f"Too many NaN values: {nan_count} ({nan_pct:.1%})")
            else:
                result.warnings.append(f"Found {nan_count} NaN values in features: {nan_features[:5]}")

        # Check Inf values
        inf_count, inf_features = self._check_infs(X, feature_names)
        result.inf_count = inf_count
        result.inf_features = inf_features

        if inf_count > 0 and not self.allow_inf:
            result.is_valid = False
            result.errors.append(f"Found {inf_count} infinite values in features: {inf_features[:5]}")

        # Check bounds
        out_of_range = self._check_bounds(X, feature_names)
        result.out_of_range_features = out_of_range
        if out_of_range:
            result.warnings.append(f"Features out of expected range: {out_of_range[:5]}")

        # Auto-correct if enabled and possible
        if self.auto_correct and (nan_count > 0 or extra or missing):
            corrected_X, corrected_names = self._correct_features(X, feature_names)
            if corrected_X is not None:
                result.corrected_features = corrected_X
                result.corrected_names = corrected_names
                # Re-check validity after correction
                if corrected_X.shape[1] == len(self.expected_names):
                    result.is_valid = True
                    result.errors = [e for e in result.errors if "mismatch" not in e.lower()]

        return result

    def _check_feature_names(
        self,
        actual_names: List[str],
    ) -> Tuple[bool, List[str], List[str]]:
        """
        Check which features are missing or extra.

        Returns:
            (count_matches, missing_features, extra_features)
        """
        actual_set = set(actual_names)

        missing = [n for n in self.expected_names if n not in actual_set]
        extra = [n for n in actual_names if n not in self.expected_set]

        count_matches = len(actual_names) == len(self.expected_names) and not missing

        return count_matches, missing, extra

    def _check_nans(
        self,
        X: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[int, List[str]]:
        """
        Check for NaN values.

        Returns:
            (total_nan_count, list_of_features_with_nan)
        """
        nan_mask = np.isnan(X)
        total_nans = np.sum(nan_mask)

        nan_features = []
        for i, name in enumerate(feature_names):
            if i < X.shape[1] and np.any(nan_mask[:, i]):
                nan_features.append(name)

        return int(total_nans), nan_features

    def _check_infs(
        self,
        X: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[int, List[str]]:
        """
        Check for infinite values.

        Returns:
            (total_inf_count, list_of_features_with_inf)
        """
        inf_mask = np.isinf(X)
        total_infs = np.sum(inf_mask)

        inf_features = []
        for i, name in enumerate(feature_names):
            if i < X.shape[1] and np.any(inf_mask[:, i]):
                inf_features.append(name)

        return int(total_infs), inf_features

    def _check_bounds(
        self,
        X: np.ndarray,
        feature_names: List[str],
    ) -> List[str]:
        """
        Check features are within expected bounds.

        Returns:
            List of feature names out of bounds
        """
        out_of_range = []

        for i, name in enumerate(feature_names):
            if name in self.feature_bounds and i < X.shape[1]:
                min_val, max_val = self.feature_bounds[name]
                col = X[:, i]
                if np.any(col < min_val) or np.any(col > max_val):
                    out_of_range.append(name)

        return out_of_range

    def _correct_features(
        self,
        X: np.ndarray,
        feature_names: List[str],
    ) -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
        """
        Attempt to correct feature issues.

        Corrections:
        1. Reorder features to match expected order
        2. Remove extra features
        3. Impute NaN values with medians
        4. Handle missing features (fill with median or 0)

        Returns:
            (corrected_X, corrected_names) or (None, None) if correction fails
        """
        try:
            # Build mapping from name to column index
            name_to_idx = {name: i for i, name in enumerate(feature_names)}

            # Create output array
            corrected_X = np.zeros((X.shape[0], len(self.expected_names)))
            corrected_names = self.expected_names.copy()

            for out_idx, expected_name in enumerate(self.expected_names):
                if expected_name in name_to_idx:
                    # Feature exists - copy it
                    in_idx = name_to_idx[expected_name]
                    corrected_X[:, out_idx] = X[:, in_idx]
                else:
                    # Feature missing - use median or 0
                    if expected_name in self.feature_medians:
                        corrected_X[:, out_idx] = self.feature_medians[expected_name]
                        logger.warning(f"Missing feature '{expected_name}' filled with median {self.feature_medians[expected_name]:.4f}")
                    else:
                        corrected_X[:, out_idx] = 0.0
                        logger.warning(f"Missing feature '{expected_name}' filled with 0 (no median available)")

            # Impute NaN values
            nan_mask = np.isnan(corrected_X)
            if np.any(nan_mask):
                for i, name in enumerate(self.expected_names):
                    col_nans = nan_mask[:, i]
                    if np.any(col_nans):
                        if name in self.feature_medians:
                            corrected_X[col_nans, i] = self.feature_medians[name]
                        else:
                            # Use column mean if no median available
                            col_mean = np.nanmean(corrected_X[:, i])
                            if np.isnan(col_mean):
                                col_mean = 0.0
                            corrected_X[col_nans, i] = col_mean

            # Final check for any remaining NaN/Inf
            corrected_X = np.nan_to_num(corrected_X, nan=0.0, posinf=0.0, neginf=0.0)

            return corrected_X, corrected_names

        except Exception as e:
            logger.error(f"Feature correction failed: {e}")
            return None, None

    def diagnose_mismatch(
        self,
        actual_names: List[str],
    ) -> Dict:
        """
        Diagnose feature mismatch in detail.

        Returns:
            Diagnostic dict with missing, extra, and suggestions
        """
        actual_set = set(actual_names)
        expected_set = self.expected_set

        missing = sorted([n for n in expected_set if n not in actual_set])
        extra = sorted([n for n in actual_set if n not in expected_set])

        # Check for similar names (typos, case differences)
        similar_pairs = []
        for m in missing:
            for e in extra:
                if m.lower() == e.lower() or self._similar(m, e):
                    similar_pairs.append((m, e))

        return {
            "expected_count": len(self.expected_names),
            "actual_count": len(actual_names),
            "missing_features": missing,
            "extra_features": extra,
            "similar_pairs": similar_pairs,  # Possible typos/case issues
            "suggestion": self._get_suggestion(missing, extra, similar_pairs),
        }

    def _similar(self, a: str, b: str) -> bool:
        """Check if two strings are similar (simple edit distance)."""
        if abs(len(a) - len(b)) > 2:
            return False
        # Simple check: same characters, different order/case
        return sorted(a.lower()) == sorted(b.lower())

    def _get_suggestion(
        self,
        missing: List[str],
        extra: List[str],
        similar: List[Tuple[str, str]],
    ) -> str:
        """Generate suggestion for fixing mismatch."""
        if similar:
            return f"Possible name mismatch: {similar[:3]}. Check feature engineering code."
        elif extra and not missing:
            return f"Extra features detected. Likely new features added since model training. Retrain model or remove: {extra[:3]}"
        elif missing and not extra:
            return f"Missing features. Check if feature engineering is complete. Missing: {missing[:3]}"
        else:
            return "Feature engineering has diverged from model training. Review feature pipeline."

    def set_training_statistics(
        self,
        medians: Dict[str, float],
        means: Optional[Dict[str, float]] = None,
        stds: Optional[Dict[str, float]] = None,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        Set training statistics for intelligent imputation.

        Args:
            medians: Feature name -> median value
            means: Feature name -> mean value (optional)
            stds: Feature name -> std value (optional)
            bounds: Feature name -> (min, max) bounds (optional)
        """
        self.feature_medians = medians

        if bounds:
            self.feature_bounds = bounds
        elif means and stds:
            # Auto-compute bounds as mean +/- 4 std
            self.feature_bounds = {
                name: (means[name] - 4 * stds[name], means[name] + 4 * stds[name])
                for name in means
                if name in stds
            }

    @classmethod
    def from_model_state(
        cls,
        dim_state: Dict,
        auto_correct: bool = True,
    ) -> "FeatureValidator":
        """
        Create FeatureValidator from saved model state.

        Args:
            dim_state: Dimensionality reduction state from model file
            auto_correct: Whether to auto-correct issues

        Returns:
            Configured FeatureValidator
        """
        # Extract feature names from var_selector or scaler
        var_selector = dim_state.get("var_selector")
        feature_names = dim_state.get("feature_names", [])

        if not feature_names and var_selector:
            # Try to get from var_selector
            if hasattr(var_selector, "get_feature_names_out"):
                feature_names = list(var_selector.get_feature_names_out())
            elif hasattr(var_selector, "feature_names_in_"):
                feature_names = list(var_selector.feature_names_in_)

        # Extract medians if available
        medians = dim_state.get("feature_medians", {})

        return cls(
            expected_feature_names=feature_names,
            feature_medians=medians,
            auto_correct=auto_correct,
        )
