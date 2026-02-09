"""
Tests for ResolutionCascade - multi-resolution model ensemble.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.phase_26_temporal.resolution_cascade import (
    ResolutionCascade,
    ResolutionCascadePrediction,
    ResolutionModelResult,
)


# =============================================================================
# TESTS: ResolutionModelResult
# =============================================================================

class TestResolutionModelResult:
    """Tests for the ResolutionModelResult dataclass."""

    def test_creation(self):
        result = ResolutionModelResult(
            resolution_minutes=5,
            cv_auc=0.72,
            n_features_raw=150,
            n_features_final=30,
            n_samples=1000,
        )
        assert result.resolution_minutes == 5
        assert result.cv_auc == 0.72
        assert result.n_features_raw == 150

    def test_to_dict(self):
        result = ResolutionModelResult(resolution_minutes=1, cv_auc=0.65)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert d["resolution_minutes"] == 1
        assert d["cv_auc"] == 0.65


# =============================================================================
# TESTS: ResolutionCascadePrediction
# =============================================================================

class TestResolutionCascadePrediction:
    """Tests for the ResolutionCascadePrediction dataclass."""

    def test_creation(self):
        pred = ResolutionCascadePrediction(
            ensemble_probability=0.72,
            ensemble_confidence=0.44,
            agreement_score=0.92,
            agreement_direction="BULLISH",
            per_resolution={1: 0.70, 5: 0.73, 15: 0.74},
            per_resolution_weight={1: 0.33, 5: 0.34, 15: 0.33},
            n_resolutions_used=3,
        )
        assert pred.ensemble_probability == 0.72
        assert pred.agreement_direction == "BULLISH"
        assert pred.n_resolutions_used == 3

    def test_to_dict(self):
        pred = ResolutionCascadePrediction(
            ensemble_probability=0.5,
            per_resolution={1: 0.5, 5: 0.5},
        )
        d = pred.to_dict()
        assert isinstance(d, dict)
        assert "ensemble_probability" in d
        assert "per_resolution" in d

    def test_default_values(self):
        pred = ResolutionCascadePrediction()
        assert pred.ensemble_probability == 0.5
        assert pred.ensemble_confidence == 0.0
        assert pred.agreement_score == 0.0
        assert pred.agreement_direction == "NEUTRAL"
        assert pred.per_resolution == {}
        assert pred.n_resolutions_used == 0


# =============================================================================
# TESTS: ResolutionCascade
# =============================================================================

class TestResolutionCascade:
    """Tests for the ResolutionCascade class."""

    def test_init_default_resolutions(self):
        cascade = ResolutionCascade()
        assert cascade.resolutions == [1, 5, 15, 30]
        assert cascade.is_fitted is False

    def test_init_custom_resolutions(self):
        cascade = ResolutionCascade(resolutions=[3, 10, 30])
        assert cascade.resolutions == [3, 10, 30]

    def test_resolutions_sorted(self):
        cascade = ResolutionCascade(resolutions=[15, 1, 5])
        assert cascade.resolutions == [1, 5, 15]

    def test_predict_not_fitted_raises(self):
        cascade = ResolutionCascade()
        with pytest.raises(RuntimeError, match="not fitted"):
            cascade.predict({})

    def test_compute_weights_by_auc(self):
        """Weights should be proportional to AUC - 0.5."""
        cascade = ResolutionCascade(resolutions=[1, 5, 15])

        # Simulate results
        cascade.results = {
            1: ResolutionModelResult(resolution_minutes=1, cv_auc=0.60),
            5: ResolutionModelResult(resolution_minutes=5, cv_auc=0.70),
            15: ResolutionModelResult(resolution_minutes=15, cv_auc=0.80),
        }
        cascade.models = {1: {}, 5: {}, 15: {}}
        cascade._compute_weights()

        # AUC - 0.5: 0.10, 0.20, 0.30 → sum = 0.60
        assert abs(cascade.weights[1] - 0.10 / 0.60) < 1e-6
        assert abs(cascade.weights[5] - 0.20 / 0.60) < 1e-6
        assert abs(cascade.weights[15] - 0.30 / 0.60) < 1e-6

        # Weights should sum to 1
        assert abs(sum(cascade.weights.values()) - 1.0) < 1e-6

    def test_compute_weights_below_threshold(self):
        """Resolutions below min_cv_auc should be excluded."""
        cascade = ResolutionCascade(resolutions=[1, 5], min_cv_auc=0.55)

        cascade.results = {
            1: ResolutionModelResult(resolution_minutes=1, cv_auc=0.50),  # Below threshold
            5: ResolutionModelResult(resolution_minutes=5, cv_auc=0.70),
        }
        cascade.models = {1: {}, 5: {}}
        cascade._compute_weights()

        # Only resolution 5 should have weight
        assert 5 in cascade.weights
        assert cascade.weights[5] == 1.0  # Only valid model

    def test_compute_weights_equal(self):
        """Equal weights when weight_by_cv_auc is False."""
        cascade = ResolutionCascade(
            resolutions=[1, 5, 15],
            weight_by_cv_auc=False,
        )

        cascade.results = {
            1: ResolutionModelResult(resolution_minutes=1, cv_auc=0.60),
            5: ResolutionModelResult(resolution_minutes=5, cv_auc=0.70),
            15: ResolutionModelResult(resolution_minutes=15, cv_auc=0.80),
        }
        cascade.models = {1: {}, 5: {}, 15: {}}
        cascade._compute_weights()

        for w in cascade.weights.values():
            assert abs(w - 1.0 / 3) < 1e-6

    def test_predict_with_mock_models(self):
        """Test prediction with mock model artifacts."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        cascade = ResolutionCascade(resolutions=[1, 5])

        # Create simple mock models
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = (X[:, 0] > 0).astype(float)

        for res in [1, 5]:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = LogisticRegression(random_state=42)
            model.fit(X_scaled, y)

            cascade.models[res] = {
                "model": model,
                "scaler": scaler,
                "feature_indices": np.arange(10),
                "feature_cols": [f"f{i}" for i in range(10)],
                "n_features_raw": 10,
            }

        cascade.results = {
            1: ResolutionModelResult(resolution_minutes=1, cv_auc=0.70),
            5: ResolutionModelResult(resolution_minutes=5, cv_auc=0.75),
        }
        cascade._compute_weights()
        cascade.is_fitted = True

        # Predict
        test_X = np.random.randn(1, 10)
        pred = cascade.predict({1: test_X, 5: test_X})

        assert isinstance(pred, ResolutionCascadePrediction)
        assert 0.0 <= pred.ensemble_probability <= 1.0
        assert pred.n_resolutions_used == 2
        assert 1 in pred.per_resolution
        assert 5 in pred.per_resolution
        assert 0.0 <= pred.agreement_score <= 1.0

    def test_predict_agreement_all_bullish(self):
        """When all models predict similar probability, agreement should be high."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        cascade = ResolutionCascade(resolutions=[1, 5, 15])

        # Create models trained on heavily biased data (95% class 1)
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = np.ones(200)
        y[:10] = 0  # Need at least 2 classes for LogisticRegression

        for res in [1, 5, 15]:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = LogisticRegression(random_state=42, C=0.01)
            model.fit(X_scaled, y)

            cascade.models[res] = {
                "model": model,
                "scaler": scaler,
                "feature_indices": np.arange(5),
                "feature_cols": [f"f{i}" for i in range(5)],
                "n_features_raw": 5,
            }

        cascade.results = {
            1: ResolutionModelResult(resolution_minutes=1, cv_auc=0.70),
            5: ResolutionModelResult(resolution_minutes=5, cv_auc=0.70),
            15: ResolutionModelResult(resolution_minutes=15, cv_auc=0.70),
        }
        cascade._compute_weights()
        cascade.is_fitted = True

        test_X = np.random.randn(1, 5)
        pred = cascade.predict({1: test_X, 5: test_X, 15: test_X})

        # All models trained on same biased data should agree
        assert pred.agreement_score > 0.8
        assert pred.agreement_direction in ("BULLISH", "LEAN_BULLISH")

    def test_build_summary(self):
        """Test summary building."""
        cascade = ResolutionCascade(resolutions=[1, 5])
        cascade.results = {
            1: ResolutionModelResult(resolution_minutes=1, cv_auc=0.65),
            5: ResolutionModelResult(resolution_minutes=5, cv_auc=0.72),
        }
        cascade.weights = {1: 0.4, 5: 0.6}

        summary = cascade._build_summary(elapsed=10.0)

        assert summary["n_resolutions_total"] == 2
        assert summary["best_resolution"] == 5
        assert summary["best_cv_auc"] == 0.72
        assert summary["total_training_time"] == 10.0


# =============================================================================
# TESTS: Save/Load
# =============================================================================

class TestResolutionCascadePersistence:
    """Tests for save/load of ResolutionCascade."""

    def test_save_load_roundtrip(self, tmp_path):
        """Save and load should preserve all state."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        cascade = ResolutionCascade(resolutions=[1, 5])

        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = (X[:, 0] > 0).astype(float)

        for res in [1, 5]:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            model = LogisticRegression(random_state=42)
            model.fit(X_scaled, y)

            cascade.models[res] = {
                "model": model,
                "scaler": scaler,
                "feature_indices": np.arange(10),
                "feature_cols": [f"f{i}" for i in range(10)],
                "n_features_raw": 10,
            }

        cascade.results = {
            1: ResolutionModelResult(resolution_minutes=1, cv_auc=0.68),
            5: ResolutionModelResult(resolution_minutes=5, cv_auc=0.73),
        }
        cascade.weights = {1: 0.4, 5: 0.6}
        cascade.is_fitted = True

        # Save
        save_path = tmp_path / "cascade_test"
        cascade.save(save_path)

        # Load
        loaded = ResolutionCascade.load(save_path)

        assert loaded.resolutions == cascade.resolutions
        assert loaded.is_fitted is True
        assert set(loaded.models.keys()) == {1, 5}
        assert loaded.weights[1] == 0.4
        assert loaded.weights[5] == 0.6
        assert loaded.results[1].cv_auc == 0.68
        assert loaded.results[5].cv_auc == 0.73

        # Predictions should match
        test_X = np.random.randn(1, 10)
        pred_original = cascade.predict({1: test_X, 5: test_X})
        pred_loaded = loaded.predict({1: test_X, 5: test_X})

        assert abs(pred_original.ensemble_probability - pred_loaded.ensemble_probability) < 1e-6

    def test_load_missing_file_raises(self, tmp_path):
        """Loading from non-existent path should raise."""
        with pytest.raises(FileNotFoundError):
            ResolutionCascade.load(tmp_path / "nonexistent")


# =============================================================================
# TESTS: Integration with Enums
# =============================================================================

class TestResolutionCascadeEnums:
    """Tests for integration with registry enums."""

    def test_cascade_type_enum_exists(self):
        from src.phase_18_persistence.registry_enums import CascadeType
        assert hasattr(CascadeType, "MULTI_RESOLUTION")
        assert CascadeType.MULTI_RESOLUTION.value == "multi_resolution"

    def test_time_resolution_includes_new_values(self):
        from src.phase_18_persistence.registry_enums import TimeResolution
        assert hasattr(TimeResolution, "MINUTE_2")
        assert hasattr(TimeResolution, "MINUTE_10")
        assert TimeResolution.MINUTE_2.value == "2min"
        assert TimeResolution.MINUTE_10.value == "10min"

    def test_cascade_config_default_resolutions(self):
        from src.phase_18_persistence.registry_configs import CascadeConfig
        config = CascadeConfig()
        assert config.resolutions_minutes == [1, 2, 3, 5, 10, 15, 30]
