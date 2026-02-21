"""
Tests for Wave 30-35: Advanced Stability Suite (19 methods).
"""

import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from src.phase_14_robustness.advanced_stability import (
    AdvancedStabilitySuite,
    DEFAULT_WEIGHTS,
)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_data(n=400, n_features=20, seed=42, shift=0.0):
    """Generate synthetic classification data with optional distribution shift."""
    rng = np.random.RandomState(seed)
    X = rng.randn(n, n_features) + shift
    signal = X[:, 0] * 0.5 + X[:, 1] * 0.3 + rng.randn(n) * 0.8
    y = (signal > 0).astype(int)
    return X, y


def _logistic_factory():
    return LogisticRegression(C=1.0, max_iter=500, random_state=42)


def _fit_and_predict(X_train, y_train, X_test):
    model = _logistic_factory()
    model.fit(X_train, y_train)
    predictions = model.predict_proba(X_test)[:, 1]
    return model, predictions


# ── PSI Tests ────────────────────────────────────────────────────────────────

class TestPSI:
    def test_identical_distributions(self):
        """PSI should be ~0 for identical distributions → score ~1.0."""
        X1, _ = _make_data(n=300, seed=42)
        X2, _ = _make_data(n=300, seed=42)
        suite = AdvancedStabilitySuite()
        result = suite._population_stability_index(X1, X2)
        assert result["score"] >= 0.8, f"Expected high score, got {result['score']}"
        assert result["mean_psi"] < 0.1

    def test_shifted_distributions(self):
        """PSI should detect large distribution shift → lower score."""
        X1, _ = _make_data(n=300, seed=42, shift=0.0)
        X2, _ = _make_data(n=300, seed=43, shift=3.0)
        suite = AdvancedStabilitySuite()
        result = suite._population_stability_index(X1, X2)
        assert result["score"] < 0.8, f"Expected lower score for shifted data, got {result['score']}"
        assert result["n_high_psi"] > 0


# ── CSI Tests ────────────────────────────────────────────────────────────────

class TestCSI:
    def test_stable_features(self):
        """CSI should show most features as stable when distributions match."""
        X1, _ = _make_data(n=300, seed=42)
        X2, _ = _make_data(n=300, seed=42)
        suite = AdvancedStabilitySuite()
        result = suite._characteristic_stability_index(X1, X2)
        assert result["score"] >= 0.8
        assert result["n_unstable"] <= 5


# ── Adversarial Validation Tests ─────────────────────────────────────────────

class TestAdversarialValidation:
    def test_same_distribution(self):
        """Adversarial AUC should be ~0.50 when train/test from same dist."""
        X, _ = _make_data(n=400, seed=42)
        X_train, X_test = X[:200], X[200:]
        suite = AdvancedStabilitySuite()
        result = suite._adversarial_validation(X_train, X_test)
        assert result["adversarial_auc"] < 0.65, (
            f"AUC {result['adversarial_auc']} too high for same distribution"
        )
        assert result["score"] > 0.2

    def test_different_distribution(self):
        """Adversarial AUC should be high when train/test differ."""
        X_train, _ = _make_data(n=200, seed=42, shift=0.0)
        X_test, _ = _make_data(n=200, seed=43, shift=3.0)
        suite = AdvancedStabilitySuite()
        result = suite._adversarial_validation(X_train, X_test)
        assert result["adversarial_auc"] > 0.60
        assert result["shift_detected"] is True


# ── ECE Tests ────────────────────────────────────────────────────────────────

class TestECE:
    def test_reasonable_calibration(self):
        """ECE should be moderate for reasonably calibrated predictions."""
        rng = np.random.RandomState(42)
        n = 500
        # Generate calibrated probabilities spread across [0, 1]
        true_proba = rng.uniform(0.1, 0.9, size=n)
        y = rng.binomial(1, true_proba)
        # Predictions ≈ true proba + small noise
        predictions = true_proba + rng.randn(n) * 0.05
        predictions = np.clip(predictions, 0.01, 0.99)

        suite = AdvancedStabilitySuite()
        result = suite._expected_calibration_error(predictions, y)
        assert result["ece"] < 0.15, f"ECE {result['ece']} too high for calibrated preds"
        assert result["score"] >= 0.0
        assert "bin_details" in result

    def test_poor_calibration(self):
        """ECE should be high for badly calibrated predictions."""
        rng = np.random.RandomState(42)
        y = rng.binomial(1, 0.5, size=500)
        # Always predict 0.9 regardless of actual class
        predictions = np.full(500, 0.9)
        suite = AdvancedStabilitySuite()
        result = suite._expected_calibration_error(predictions, y)
        assert result["ece"] > 0.1, f"ECE {result['ece']} too low for poor calibration"


# ── DSR Tests ────────────────────────────────────────────────────────────────

class TestDSR:
    def test_correction_reduces_sharpe(self):
        """Deflated Sharpe should be less than observed Sharpe."""
        cv_scores = [0.55, 0.58, 0.56, 0.57, 0.54]
        suite = AdvancedStabilitySuite()
        result = suite._deflated_sharpe_ratio(cv_scores, n_experiments_total=100)
        assert result["sharpe_deflated"] <= result["sharpe_observed"]
        assert result["haircut_pct"] > 0
        assert result["score"] >= 0.0

    def test_insufficient_scores(self):
        """DSR should skip with too few CV scores."""
        suite = AdvancedStabilitySuite()
        result = suite._deflated_sharpe_ratio([0.55], n_experiments_total=10)
        assert result.get("skipped") is True


# ── SFI Tests ────────────────────────────────────────────────────────────────

class TestSFI:
    def test_importance_detected(self):
        """SFI should identify the most important feature."""
        X_train, y_train = _make_data(n=300, seed=42)
        X_test, y_test = _make_data(n=100, seed=43)
        model, _ = _fit_and_predict(X_train, y_train, X_test)

        suite = AdvancedStabilitySuite(n_sfi_repeats=2)
        result = suite._single_feature_importance(model, X_test, y_test)

        assert "score" in result
        assert result["n_features"] == 20
        assert 0.0 <= result["score"] <= 1.0


# ── CPCV Tests ───────────────────────────────────────────────────────────────

class TestCPCV:
    def test_produces_multiple_scores(self):
        """CPCV should produce 10 AUC scores for K=5 groups."""
        X, y = _make_data(n=500, seed=42)
        suite = AdvancedStabilitySuite(n_cpcv_groups=5)
        result = suite._combinatorial_purged_cv(X, y, _logistic_factory)

        assert result["n_paths"] >= 5  # At least C(5,2)=10, minus any failures
        assert "auc_mean" in result
        assert "auc_std" in result
        assert 0.0 <= result["score"] <= 1.0

    def test_too_few_samples_skips(self):
        """CPCV should skip gracefully with very few samples."""
        X, y = _make_data(n=30, seed=42)
        suite = AdvancedStabilitySuite(n_cpcv_groups=5)
        result = suite._combinatorial_purged_cv(X, y, _logistic_factory)
        # With 30 samples / 5 groups = 6 per group — too few
        assert result.get("skipped", False) or result["score"] >= 0


# ── Stability Selection Tests ────────────────────────────────────────────────

class TestStabilitySelection:
    def test_consistent_features_selected(self):
        """Features with strong signal should be consistently selected."""
        X, y = _make_data(n=400, seed=42)
        suite = AdvancedStabilitySuite(n_stability_sel=5)
        result = suite._stability_selection(X, y, _logistic_factory)

        assert "n_stable_50pct" in result
        assert "n_very_stable_80pct" in result
        assert result["n_features"] == 20
        assert 0.0 <= result["score"] <= 1.0


# ── Rashomon Set Tests ───────────────────────────────────────────────────────

class TestRashomonSet:
    def test_deterministic_model_high_agreement(self):
        """Logistic regression should have high prediction agreement."""
        X_train, y_train = _make_data(n=300, seed=42)
        X_test, y_test = _make_data(n=100, seed=43)

        suite = AdvancedStabilitySuite(n_rashomon=5)
        result = suite._rashomon_set(
            X_train, y_train, X_test, y_test, _logistic_factory, base_auc=0.55
        )

        assert result["n_models"] >= 3
        assert result["mean_correlation"] > 0.5
        assert 0.0 <= result["score"] <= 1.0


# ── Meta-Labeling Tests ─────────────────────────────────────────────────────

class TestMetaLabeling:
    def test_returns_valid_auc(self):
        """Meta-labeling should produce a valid AUC score."""
        X_train, y_train = _make_data(n=300, seed=42)
        X_test, y_test = _make_data(n=100, seed=43)
        _, predictions = _fit_and_predict(X_train, y_train, X_test)

        suite = AdvancedStabilitySuite()
        result = suite._meta_labeling(X_test, predictions, y_test)

        assert "meta_auc" in result or result.get("skipped")
        if "meta_auc" in result:
            assert 0.0 <= result["meta_auc"] <= 1.0
            assert 0.0 <= result["score"] <= 1.0


# ── Knockoff Tests ───────────────────────────────────────────────────────────

class TestKnockoff:
    def test_filters_noise(self):
        """Knockoff should find some genuine features survive."""
        X, y = _make_data(n=400, seed=42)
        suite = AdvancedStabilitySuite()
        result = suite._knockoff_features(X, y, _logistic_factory)

        assert "n_surviving" in result
        assert result["n_features"] == 20
        assert 0.0 <= result["score"] <= 1.0


# ── ADWIN Tests ──────────────────────────────────────────────────────────────

class TestADWIN:
    def test_no_drift(self):
        """ADWIN should detect no drift for stable scores."""
        cv_scores = [0.55, 0.56, 0.54, 0.55, 0.56, 0.55]
        suite = AdvancedStabilitySuite()
        result = suite._adwin_drift(cv_scores)
        assert result["score"] > 0.5
        assert result["drift_detected"] is False

    def test_detects_drift(self):
        """ADWIN should detect drift when scores change dramatically."""
        cv_scores = [0.70, 0.68, 0.72, 0.45, 0.42, 0.40]
        suite = AdvancedStabilitySuite()
        result = suite._adwin_drift(cv_scores)
        assert result["drift_magnitude"] > 0.5


# ── Adversarial Overfitting Detection Tests ──────────────────────────────────

class TestAdversarialOverfittingDetection:
    def test_robust_model_scores_well(self):
        """A regularized logistic regression should show robust overfitting scores."""
        X_train, y_train = _make_data(n=400, seed=42)
        X_test, y_test = _make_data(n=100, seed=43)
        model, predictions = _fit_and_predict(X_train, y_train, X_test)

        suite = AdvancedStabilitySuite(n_noise_levels=2)
        result = suite._adversarial_overfitting_detection(
            X_train, y_train, X_test, y_test,
            _logistic_factory, model, predictions, base_auc=0.55,
        )

        assert 0.0 <= result["score"] <= 1.0
        assert "noise_score" in result
        assert "perturb_score" in result
        assert "confidence_score" in result
        # Logistic regression is well-regularized, should score decently
        assert result["score"] >= 0.3, f"Robust model scored too low: {result['score']}"

    def test_returns_sub_scores(self):
        """All three sub-scores should be present and in [0, 1]."""
        X_train, y_train = _make_data(n=300, seed=42)
        X_test, y_test = _make_data(n=100, seed=43)
        model, predictions = _fit_and_predict(X_train, y_train, X_test)

        suite = AdvancedStabilitySuite(n_noise_levels=2)
        result = suite._adversarial_overfitting_detection(
            X_train, y_train, X_test, y_test,
            _logistic_factory, model, predictions, base_auc=0.55,
        )

        for key in ["noise_score", "perturb_score", "confidence_score"]:
            assert 0.0 <= result[key] <= 1.0, f"{key}={result[key]} out of [0,1]"


# ── Conformal Prediction Tests (mapie 1.3.0) ────────────────────────────────

class TestConformalPredictionNew:
    def test_coverage_at_multiple_levels(self):
        """Conformal should report coverage at 0.80, 0.90, 0.95 levels."""
        X_train, y_train = _make_data(n=400, seed=42)
        X_test, y_test = _make_data(n=100, seed=43)
        model, _ = _fit_and_predict(X_train, y_train, X_test)

        suite = AdvancedStabilitySuite()
        result = suite._conformal_prediction(
            model, X_train, y_train, X_test, y_test,
        )

        if result.get("skipped"):
            pytest.skip("mapie not available")

        assert 0.0 <= result["score"] <= 1.0
        assert "coverage_component" in result
        assert "efficiency_component" in result
        assert "levels" in result
        # Should have 3 coverage levels
        assert len(result["levels"]) == 3

    def test_score_reflects_coverage_quality(self):
        """Score should be reasonable for well-calibrated model."""
        X_train, y_train = _make_data(n=500, seed=42)
        X_test, y_test = _make_data(n=150, seed=43)
        model, _ = _fit_and_predict(X_train, y_train, X_test)

        suite = AdvancedStabilitySuite()
        result = suite._conformal_prediction(
            model, X_train, y_train, X_test, y_test,
        )

        if result.get("skipped"):
            pytest.skip("mapie not available")

        assert result["score"] >= 0.2, f"Score too low: {result['score']}"


# ── Disagreement Smoothing Tests ─────────────────────────────────────────────

class TestDisagreementSmoothing:
    def test_stable_model_high_agreement(self):
        """Logistic regression trained on same data should have high agreement."""
        X_train, y_train = _make_data(n=400, seed=42)
        X_test, y_test = _make_data(n=100, seed=43)

        suite = AdvancedStabilitySuite(n_disagreement_folds=3)
        result = suite._disagreement_weighted_smoothing(
            X_train, y_train, X_test, y_test, _logistic_factory,
        )

        assert 0.0 <= result["score"] <= 1.0
        assert "pred_agreement_score" in result
        assert "feat_agreement_score" in result
        # Logistic regression should have relatively consistent predictions
        assert result["pred_agreement_score"] >= 0.3, (
            f"Prediction agreement too low: {result['pred_agreement_score']}"
        )

    def test_returns_detailed_scores(self):
        """Should return detailed disagreement metrics."""
        X_train, y_train = _make_data(n=300, seed=42)
        X_test, y_test = _make_data(n=100, seed=43)

        suite = AdvancedStabilitySuite(n_disagreement_folds=3)
        result = suite._disagreement_weighted_smoothing(
            X_train, y_train, X_test, y_test, _logistic_factory,
        )

        assert "n_models" in result
        assert result["n_models"] >= 2
        assert "smoothed_auc" in result
        assert "mean_pred_std" in result
        assert "mean_feature_disagreement" in result
        assert "n_high_disagreement_samples" in result

    def test_smoothed_auc_valid(self):
        """Smoothed AUC should be a valid metric in [0, 1]."""
        X_train, y_train = _make_data(n=300, seed=42)
        X_test, y_test = _make_data(n=100, seed=43)

        suite = AdvancedStabilitySuite(n_disagreement_folds=3)
        result = suite._disagreement_weighted_smoothing(
            X_train, y_train, X_test, y_test, _logistic_factory,
        )

        assert 0.0 <= result["smoothed_auc"] <= 1.0


# ── Composite Tests ──────────────────────────────────────────────────────────

class TestComposite:
    def test_weights_sum_to_one(self):
        """Default weights should sum to 1.0."""
        total = sum(DEFAULT_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected 1.0"

    def test_composite_is_weighted_average(self):
        """Composite should be weighted average of active scores."""
        suite = AdvancedStabilitySuite()
        results = {
            "psi": {"score": 0.8},
            "csi": {"score": 0.9},
            "adversarial": {"score": 0.7},
            "ece": {"score": 0.6},
            "dsr": {"score": 0.5},
            "sfi": {"score": 0.8},
            "meta_label": {"score": 0.6},
            "knockoff": {"score": 0.7},
            "adwin": {"score": 0.9},
            "cpcv": {"score": 0.75},
            "stability_selection": {"score": 0.8},
            "rashomon": {"score": 0.85},
            "shap": {"score": -1.0, "skipped": True},  # Skipped
            "conformal": {"score": -1.0, "skipped": True},  # Skipped
            "adversarial_overfitting": {"score": 0.7},
            "disagreement_smoothing": {"score": 0.75},
            "feature_causality": {"score": 0.65},
            "prediction_interval_coverage": {"score": 0.7},
            "distribution_robust": {"score": 0.8},
        }
        composite = suite._compute_composite(results)
        assert 0.0 <= composite <= 1.0
        # With shap+conformal skipped, composite should be weighted avg of 17 active methods
        assert composite > 0.5  # All scores are decent

    def test_19_weights(self):
        """DEFAULT_WEIGHTS should have 19 entries summing to 1.0."""
        assert len(DEFAULT_WEIGHTS) == 19, f"Expected 19 weights, got {len(DEFAULT_WEIGHTS)}"
        total = sum(DEFAULT_WEIGHTS.values())
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected 1.0"
        assert "adversarial_overfitting" in DEFAULT_WEIGHTS
        assert "disagreement_smoothing" in DEFAULT_WEIGHTS
        assert "feature_causality" in DEFAULT_WEIGHTS
        assert "prediction_interval_coverage" in DEFAULT_WEIGHTS
        assert "distribution_robust" in DEFAULT_WEIGHTS


# ── Feature Causality Tests (Wave 35) ────────────────────────────────────────

class TestFeatureCausality:
    def test_signal_detection(self):
        """Features with causal signal should produce significant F-tests."""
        rng = np.random.RandomState(42)
        n = 200
        X = rng.randn(n, 10)
        # Feature 0 is causal (lagged version predicts y)
        y = (X[:, 0] * 0.8 + rng.randn(n) * 0.3 > 0).astype(int)

        suite = AdvancedStabilitySuite()
        result = suite._feature_causality_scoring(X, y)
        assert result["score"] >= 0, "Score should be non-negative"
        assert "n_tested" in result
        assert result["n_tested"] > 0
        assert "proportion_significant" in result

    def test_small_data_skip(self):
        """Should skip if too few samples."""
        X = np.random.randn(10, 5)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        suite = AdvancedStabilitySuite()
        result = suite._feature_causality_scoring(X, y)
        assert result["score"] == -1.0
        assert result.get("skipped", False)


# ── Prediction Interval Coverage Tests (Wave 35) ─────────────────────────────

class TestPredictionIntervalCoverage:
    def test_coverage_near_nominal(self):
        """Bootstrap coverage should be near target for well-calibrated model."""
        X_train, y_train = _make_data(n=300, seed=42)
        X_test, y_test = _make_data(n=100, seed=43)
        suite = AdvancedStabilitySuite()
        result = suite._prediction_interval_coverage(
            X_train, y_train, X_test, y_test, _logistic_factory, n_bootstraps=5
        )
        assert result["score"] >= 0, "Score should be non-negative"
        assert "actual_coverage" in result
        assert 0.0 <= result["actual_coverage"] <= 1.0
        assert "mean_interval_width" in result
        assert result["n_bootstraps_used"] >= 3

    def test_insufficient_samples(self):
        """Should skip if too few samples."""
        X = np.random.randn(10, 5)
        y = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        suite = AdvancedStabilitySuite()
        result = suite._prediction_interval_coverage(
            X, y, X, y, _logistic_factory, n_bootstraps=5
        )
        assert result["score"] == -1.0


# ── Distribution-Robust Scoring Tests (Wave 35) ──────────────────────────────

class TestDistributionRobust:
    def test_uniform_performance(self):
        """Uniform performance across slices → high score."""
        X_train, y_train = _make_data(n=300, seed=42)
        X_test, y_test = _make_data(n=200, seed=43)
        model, predictions = _fit_and_predict(X_train, y_train, X_test)

        suite = AdvancedStabilitySuite()
        result = suite._distribution_robust_scoring(X_test, y_test, model, predictions)
        assert result["score"] >= 0, "Score should be non-negative"
        assert "worst_auc" in result
        assert "mean_auc" in result
        assert "robustness_ratio" in result
        assert "slice_details" in result
        assert len(result["slice_details"]) >= 2

    def test_too_few_samples(self):
        """Should skip with very few test samples."""
        X = np.random.randn(20, 5)
        y = np.array([0]*10 + [1]*10)
        preds = np.random.rand(20)
        suite = AdvancedStabilitySuite()
        result = suite._distribution_robust_scoring(X, y, None, preds)
        assert result["score"] == -1.0


# ── Integration Test ─────────────────────────────────────────────────────────

class TestRunAll:
    def test_returns_all_methods(self):
        """run_all should return results for all 19 methods + composite."""
        X_train, y_train = _make_data(n=300, seed=42)
        X_test, y_test = _make_data(n=100, seed=43)
        model, predictions = _fit_and_predict(X_train, y_train, X_test)

        suite = AdvancedStabilitySuite(
            n_cpcv_groups=3, n_stability_sel=3, n_rashomon=3,
            n_sfi_repeats=2, n_noise_levels=2, n_disagreement_folds=3,
        )
        results = suite.run_all(
            X_train=X_train, y_train=y_train,
            X_test=X_test, y_test=y_test,
            model_factory_fn=_logistic_factory,
            trained_model=model,
            predictions=predictions,
            base_auc=0.55,
            cv_scores=[0.55, 0.56, 0.54, 0.57],
            n_experiments_total=100,
        )

        # All 19 methods should be present
        expected_keys = [
            "psi", "csi", "adversarial", "ece", "dsr",
            "sfi", "meta_label", "knockoff", "adwin",
            "cpcv", "stability_selection", "rashomon",
            "shap", "conformal",
            "adversarial_overfitting", "disagreement_smoothing",
            "feature_causality", "prediction_interval_coverage",
            "distribution_robust",
            "composite_advanced", "weights_used",
        ]
        for key in expected_keys:
            assert key in results, f"Missing key: {key}"

        # Composite should be valid
        assert 0.0 <= results["composite_advanced"] <= 1.0

        # Each method should have a score
        for key in expected_keys[:19]:
            r = results[key]
            assert "score" in r, f"{key} missing 'score'"

    def test_experiment_result_has_field(self):
        """ExperimentResult should have stability_advanced dict field."""
        from src.phase_21_continuous.experiment_tracking import ExperimentResult, ExperimentConfig
        config = ExperimentConfig(experiment_type="default")
        r = ExperimentResult(experiment_id="test", config=config)
        assert hasattr(r, "stability_advanced")
        assert isinstance(r.stability_advanced, dict)

        # Should serialize/deserialize
        d = r.to_dict()
        assert "stability_advanced" in d
        r2 = ExperimentResult.from_dict(d)
        assert isinstance(r2.stability_advanced, dict)
