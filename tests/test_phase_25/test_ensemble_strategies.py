"""Tests for ensemble strategies and model selector (Phase 25 Risk Management)."""

import numpy as np
import pytest

from src.phase_25_risk_management.ensemble_strategies import (
    ModelCandidate,
    EnsemblePrediction,
    EnsembleStrategy,
    WeightedAverageEnsemble,
    MedianEnsemble,
    VotingEnsemble,
    StackingEnsemble,
)


# ---------------------------------------------------------------------------
# ModelCandidate
# ---------------------------------------------------------------------------

class TestModelCandidate:
    def test_default_construction(self):
        mc = ModelCandidate(model_id="m1", model_path="/tmp/m1", config={})
        assert mc.model_id == "m1"
        assert mc.cv_auc == 0.0
        assert mc.tier == 1

    def test_score_calculation(self):
        mc = ModelCandidate(
            model_id="m1", model_path="/tmp", config={},
            cv_auc=0.70, test_auc=0.65,
            backtest_sharpe=1.2, wmes_score=0.60,
            stability_score=0.75, fragility_score=0.2,
        )
        score = mc.score()
        assert score > 0

    def test_score_penalizes_fragility(self):
        mc_robust = ModelCandidate(
            model_id="robust", model_path="/tmp", config={},
            cv_auc=0.70, test_auc=0.65, fragility_score=0.1,
        )
        mc_fragile = ModelCandidate(
            model_id="fragile", model_path="/tmp", config={},
            cv_auc=0.70, test_auc=0.65, fragility_score=0.9,
        )
        assert mc_robust.score() > mc_fragile.score()

    def test_matches_window_default(self):
        mc = ModelCandidate(model_id="m1", model_path="/tmp", config={})
        assert mc.matches_window()

    def test_matches_window_with_tolerance(self):
        mc = ModelCandidate(
            model_id="m1", model_path="/tmp", config={},
            entry_window_start=0, entry_window_end=120,
        )
        assert mc.matches_window(entry_window=(10, 110), tolerance=30)
        assert not mc.matches_window(entry_window=(60, 200), tolerance=10)


# ---------------------------------------------------------------------------
# EnsemblePrediction
# ---------------------------------------------------------------------------

class TestEnsemblePrediction:
    def test_construction(self):
        ep = EnsemblePrediction(
            swing_probability=0.72,
            timing_probability=0.61,
            confidence=0.85,
            direction="LONG",
        )
        assert ep.swing_probability == 0.72
        assert ep.direction == "LONG"
        assert ep.n_models == 0


# ---------------------------------------------------------------------------
# EnsembleStrategy (abstract)
# ---------------------------------------------------------------------------

class TestEnsembleStrategy:
    def test_base_raises(self):
        with pytest.raises(NotImplementedError):
            EnsembleStrategy().combine([], [])


# ---------------------------------------------------------------------------
# WeightedAverageEnsemble
# ---------------------------------------------------------------------------

class TestWeightedAverage:
    def test_empty_predictions(self):
        wae = WeightedAverageEnsemble()
        swing, timing = wae.combine([], [])
        assert swing == 0.5
        assert timing == 0.5

    def test_single_prediction(self):
        wae = WeightedAverageEnsemble()
        preds = [{"swing_proba": 0.8, "timing_proba": 0.6}]
        swing, timing = wae.combine(preds, [1.0])
        assert abs(swing - 0.8) < 1e-6
        assert abs(timing - 0.6) < 1e-6

    def test_equal_weights(self):
        wae = WeightedAverageEnsemble()
        preds = [
            {"swing_proba": 0.7, "timing_proba": 0.5},
            {"swing_proba": 0.9, "timing_proba": 0.7},
        ]
        swing, timing = wae.combine(preds, [1.0, 1.0])
        assert abs(swing - 0.8) < 1e-6
        assert abs(timing - 0.6) < 1e-6

    def test_unequal_weights(self):
        wae = WeightedAverageEnsemble()
        preds = [
            {"swing_proba": 0.6, "timing_proba": 0.5},
            {"swing_proba": 0.8, "timing_proba": 0.7},
        ]
        swing, timing = wae.combine(preds, [3.0, 1.0])
        # swing = (0.6*0.75 + 0.8*0.25) = 0.65
        assert abs(swing - 0.65) < 1e-6


# ---------------------------------------------------------------------------
# MedianEnsemble
# ---------------------------------------------------------------------------

class TestMedianEnsemble:
    def test_empty(self):
        me = MedianEnsemble()
        assert me.combine([], []) == (0.5, 0.5)

    def test_odd_count(self):
        me = MedianEnsemble()
        preds = [
            {"swing_proba": 0.6, "timing_proba": 0.4},
            {"swing_proba": 0.8, "timing_proba": 0.6},
            {"swing_proba": 0.9, "timing_proba": 0.8},
        ]
        swing, timing = me.combine(preds, [1, 1, 1])
        assert abs(swing - 0.8) < 1e-6
        assert abs(timing - 0.6) < 1e-6

    def test_robust_to_outlier(self):
        me = MedianEnsemble()
        preds = [
            {"swing_proba": 0.7, "timing_proba": 0.6},
            {"swing_proba": 0.72, "timing_proba": 0.62},
            {"swing_proba": 0.01, "timing_proba": 0.01},  # outlier
        ]
        swing, _ = me.combine(preds, [1, 1, 1])
        assert swing > 0.5  # Median ignores outlier


# ---------------------------------------------------------------------------
# VotingEnsemble
# ---------------------------------------------------------------------------

class TestVotingEnsemble:
    def test_empty(self):
        ve = VotingEnsemble()
        assert ve.combine([], []) == (0.5, 0.5)

    def test_unanimous_up(self):
        ve = VotingEnsemble(threshold=0.55)
        preds = [
            {"swing_proba": 0.7, "timing_proba": 0.6},
            {"swing_proba": 0.8, "timing_proba": 0.7},
        ]
        swing, timing = ve.combine(preds, [1, 1])
        assert swing > 0.5  # All voted up
        assert timing > 0.5

    def test_split_vote(self):
        ve = VotingEnsemble(threshold=0.55)
        preds = [
            {"swing_proba": 0.7, "timing_proba": 0.6},
            {"swing_proba": 0.4, "timing_proba": 0.4},
        ]
        swing, _ = ve.combine(preds, [1, 1])
        # 50/50 split = 0.5 vote pct → combined near 0.5
        assert abs(swing - 0.5) < 0.15


# ---------------------------------------------------------------------------
# StackingEnsemble
# ---------------------------------------------------------------------------

class TestStackingEnsemble:
    def test_no_meta_model_falls_back(self):
        se = StackingEnsemble()
        preds = [
            {"swing_proba": 0.7, "timing_proba": 0.6},
            {"swing_proba": 0.8, "timing_proba": 0.7},
        ]
        swing, timing = se.combine(preds, [1, 1])
        # Falls back to WeightedAverage
        assert abs(swing - 0.75) < 1e-6

    def test_empty(self):
        se = StackingEnsemble()
        assert se.combine([], []) == (0.5, 0.5)
