"""Tests for dynamic ensemble weighting."""

import pytest

from src.phase_15_strategy.dynamic_weights import DynamicEnsembleWeighter


class TestDynamicEnsembleWeighter:
    """Tests for DynamicEnsembleWeighter."""

    def test_equal_weights_when_no_history(self):
        """Models with no recorded scores get equal weight."""
        w = DynamicEnsembleWeighter()
        weights = w.get_weights(["m1", "m2", "m3"])
        assert len(weights) == 3
        for mid in ["m1", "m2", "m3"]:
            assert weights[mid] == pytest.approx(1.0 / 3, abs=1e-9)

    def test_better_model_gets_higher_weight(self):
        """A model with consistently higher scores should be weighted more."""
        w = DynamicEnsembleWeighter(lookback=10, temperature=1.0)
        # Model A is consistently better than model B
        for _ in range(5):
            w.update("A", 0.80)
            w.update("B", 0.55)
        weights = w.get_weights(["A", "B"])
        assert weights["A"] > weights["B"]

    def test_weights_sum_to_one(self):
        """Weights must sum to 1.0 regardless of input."""
        w = DynamicEnsembleWeighter(lookback=10, temperature=0.5)
        w.update("m1", 0.70)
        w.update("m2", 0.65)
        w.update("m3", 0.80)
        w.update("m4", 0.50)
        weights = w.get_weights()
        assert sum(weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_minimum_weight_floor_enforced(self):
        """No model should fall below (min_weight_fraction / N)."""
        w = DynamicEnsembleWeighter(
            lookback=10, min_weight_fraction=0.5, temperature=0.1
        )
        # One model is much better; low temperature concentrates weight
        for _ in range(10):
            w.update("good", 0.95)
            w.update("bad", 0.50)
        weights = w.get_weights(["good", "bad"])
        equal_w = 0.5  # 1/N where N=2
        floor = 0.5 * equal_w  # min_weight_fraction * equal_weight = 0.25
        assert weights["bad"] >= floor - 1e-9

    def test_lookback_window_limits_history(self):
        """Only the last `lookback` scores should be used."""
        w = DynamicEnsembleWeighter(lookback=3)
        # First add bad scores, then add good scores
        for _ in range(10):
            w.update("m1", 0.40)
        # Now overwrite with good scores (only last 3 should count)
        for _ in range(3):
            w.update("m1", 0.90)
        # Internal history should have length == lookback
        assert len(w._history["m1"]) == 3
        # All remaining scores should be the good ones
        assert all(s == 0.90 for s in w._history["m1"])

    def test_get_weighted_prediction(self):
        """Weighted prediction should reflect model weights."""
        w = DynamicEnsembleWeighter(lookback=10, temperature=1.0)
        # Make model A clearly better
        for _ in range(5):
            w.update("A", 0.90)
            w.update("B", 0.50)
        predictions = {"A": 0.80, "B": 0.20}
        result = w.get_weighted_prediction(predictions)
        # A has higher weight, so result should be closer to 0.80 than 0.20
        assert result > 0.50
        # Also verify it's between the two predictions
        assert 0.20 <= result <= 0.80

    def test_reset_clears_all_history(self):
        """reset() should remove all tracked models and scores."""
        w = DynamicEnsembleWeighter()
        w.update("m1", 0.70)
        w.update("m2", 0.60)
        assert w.n_models == 2
        w.reset()
        assert w.n_models == 0
        assert w.get_weights() == {}

    def test_single_model_gets_weight_one(self):
        """A single model should always receive weight 1.0."""
        w = DynamicEnsembleWeighter()
        w.update("solo", 0.65)
        weights = w.get_weights(["solo"])
        assert weights["solo"] == pytest.approx(1.0, abs=1e-9)

    def test_empty_model_list_returns_empty(self):
        """An empty model list should return an empty dict."""
        w = DynamicEnsembleWeighter()
        assert w.get_weights([]) == {}

    def test_temperature_affects_concentration(self):
        """Lower temperature should make weights more concentrated."""
        for _ in range(5):
            w_hot = DynamicEnsembleWeighter(temperature=5.0)
            w_cold = DynamicEnsembleWeighter(temperature=0.1)
            for weighter in [w_hot, w_cold]:
                weighter.update("A", 0.80)
                weighter.update("B", 0.55)

        weights_hot = w_hot.get_weights(["A", "B"])
        weights_cold = w_cold.get_weights(["A", "B"])
        # Cold temperature should give model A a larger share than hot
        assert weights_cold["A"] > weights_hot["A"]
