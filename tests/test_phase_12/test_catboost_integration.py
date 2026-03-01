"""
Tests for CatBoost integration in the experiment runner.

Validates:
  - CatBoost model creation (or fallback to HistGradientBoosting)
  - Model has sklearn-compatible API (fit, predict, predict_proba)
  - Integration with _create_model in experiment runner
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification

from src.experiment_config import ExperimentConfig, ModelConfig


# ─── Fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture
def catboost_config():
    """ExperimentConfig with catboost model_type."""
    config = ExperimentConfig()
    config.model = ModelConfig(
        model_type="catboost",
        gb_n_estimators=50,
        gb_max_depth=3,
        gb_learning_rate=0.1,
        gb_min_samples_leaf=50,
    )
    return config


@pytest.fixture
def binary_data():
    """Small binary classification dataset."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5,
        random_state=42,
    )
    return X, y


# ─── Tests ───────────────────────────────────────────────────────────────────


class TestCatBoostCreation:

    def test_creates_model_or_fallback(self, catboost_config):
        """CatBoost should be created if installed, else fallback to HistGradient."""
        from src.phase_21_continuous.experiment_runner import UnifiedExperimentRunner
        runner = UnifiedExperimentRunner()
        model = runner._create_model(catboost_config)

        # Should be either CatBoostClassifier or HistGradientBoostingClassifier
        name = type(model).__name__
        assert name in (
            "CatBoostClassifier",
            "HistGradientBoostingClassifier",
        ), f"Unexpected model type: {name}"

    def test_model_has_sklearn_api(self, catboost_config, binary_data):
        """Created model should have fit/predict/predict_proba."""
        from src.phase_21_continuous.experiment_runner import UnifiedExperimentRunner
        runner = UnifiedExperimentRunner()
        model = runner._create_model(catboost_config)

        assert hasattr(model, "fit")
        assert hasattr(model, "predict")
        assert hasattr(model, "predict_proba")

    def test_model_trains_and_predicts(self, catboost_config, binary_data):
        """Model should train and produce probability predictions."""
        from src.phase_21_continuous.experiment_runner import UnifiedExperimentRunner
        runner = UnifiedExperimentRunner()
        model = runner._create_model(catboost_config)

        X, y = binary_data
        X_train, X_test = X[:150], X[150:]
        y_train = y[:150]

        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)

        assert proba.shape == (50, 2)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_max_depth_capped(self, catboost_config):
        """Max depth should be capped at 4 (aggressive anti-overfit)."""
        catboost_config.model.gb_max_depth = 10  # Request 10
        from src.phase_21_continuous.experiment_runner import UnifiedExperimentRunner
        runner = UnifiedExperimentRunner()
        model = runner._create_model(catboost_config)

        # Check that depth is capped
        name = type(model).__name__
        if name == "CatBoostClassifier":
            depth = model.get_param("depth")
            assert depth <= 4
        else:
            assert model.max_depth <= 4


class TestCatBoostWithRegimeRouter:

    def test_regime_router_wraps_catboost(self, catboost_config, binary_data):
        """When use_regime_router=True, CatBoost should be wrapped."""
        catboost_config.model.use_regime_router = True
        catboost_config.model.regime_split_method = "vix_quartile"

        from src.phase_21_continuous.experiment_runner import UnifiedExperimentRunner
        runner = UnifiedExperimentRunner()
        model = runner._create_model(catboost_config)

        # Should be RegimeRouter wrapping CatBoost or HistGradient
        name = type(model).__name__
        assert name == "RegimeRouter"

    def test_regime_router_trains(self, catboost_config, binary_data):
        """RegimeRouter-wrapped model should train (fallback to global)."""
        catboost_config.model.use_regime_router = True

        from src.phase_21_continuous.experiment_runner import UnifiedExperimentRunner
        runner = UnifiedExperimentRunner()
        model = runner._create_model(catboost_config)

        X, y = binary_data
        # Train without vol_series -> falls back to global model
        model.fit(X[:150], y[:150])
        proba = model.predict_proba(X[150:])

        assert proba.shape == (50, 2)
        assert np.all(proba >= 0) and np.all(proba <= 1)
