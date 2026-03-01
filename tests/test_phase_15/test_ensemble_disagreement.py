"""Tests for ensemble disagreement features."""

import math
import pytest

from src.phase_15_strategy.ensemble_disagreement import compute_ensemble_disagreement


class TestComputeEnsembleDisagreement:
    def test_empty_list(self):
        result = compute_ensemble_disagreement([])
        assert result["ens_std"] == 0.0
        assert result["ens_agreement"] == 1.0

    def test_single_model(self):
        result = compute_ensemble_disagreement([0.7])
        assert result["ens_std"] == 0.0
        assert result["ens_agreement"] == 1.0
        assert result["ens_range"] == 0.0

    def test_perfect_agreement_bullish(self):
        result = compute_ensemble_disagreement([0.8, 0.7, 0.9])
        assert result["ens_agreement"] == 1.0  # All bullish
        assert result["ens_entropy"] == 0.0    # No entropy when unanimous

    def test_perfect_agreement_bearish(self):
        result = compute_ensemble_disagreement([0.2, 0.3, 0.1])
        assert result["ens_agreement"] == 1.0
        assert result["ens_entropy"] == 0.0

    def test_maximum_disagreement(self):
        # 50/50 split
        result = compute_ensemble_disagreement([0.8, 0.2, 0.7, 0.3])
        assert result["ens_agreement"] == 0.5
        assert abs(result["ens_entropy"] - 1.0) < 0.01  # Max entropy = 1.0

    def test_std_increases_with_spread(self):
        narrow = compute_ensemble_disagreement([0.60, 0.62, 0.61])
        wide = compute_ensemble_disagreement([0.20, 0.50, 0.80])
        assert wide["ens_std"] > narrow["ens_std"]

    def test_range_calculation(self):
        result = compute_ensemble_disagreement([0.2, 0.5, 0.9])
        assert abs(result["ens_range"] - 0.7) < 0.001

    def test_agreement_ratio(self):
        # 3 bullish, 1 bearish
        result = compute_ensemble_disagreement([0.6, 0.7, 0.8, 0.3])
        assert result["ens_agreement"] == 0.75

    def test_all_keys_present(self):
        result = compute_ensemble_disagreement([0.5, 0.6])
        assert set(result.keys()) == {"ens_std", "ens_range", "ens_agreement", "ens_entropy"}

    def test_entropy_bounded(self):
        for probas in [[0.1, 0.9], [0.5, 0.5], [0.7, 0.3, 0.8, 0.2]]:
            result = compute_ensemble_disagreement(probas)
            assert 0 <= result["ens_entropy"] <= 1.0

    def test_std_non_negative(self):
        for probas in [[0.5, 0.5], [0.1, 0.9], [0.3, 0.7, 0.5]]:
            result = compute_ensemble_disagreement(probas)
            assert result["ens_std"] >= 0

    def test_realistic_5_model_ensemble(self):
        # Typical production scenario: 5 models, mostly agreeing
        result = compute_ensemble_disagreement([0.62, 0.58, 0.65, 0.55, 0.61])
        assert result["ens_agreement"] == 1.0  # All > 0.5
        assert result["ens_std"] < 0.1  # Low spread
        assert result["ens_range"] == pytest.approx(0.1, abs=0.001)
