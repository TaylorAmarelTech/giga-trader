"""Tests for ThompsonSamplingSelector -- bandit-based model selection."""

import json
import pytest
import numpy as np
from pathlib import Path

from src.phase_15_strategy.thompson_selector import ThompsonSamplingSelector


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def model_ids():
    return ["swing", "timing", "ensemble"]


@pytest.fixture
def selector(model_ids):
    return ThompsonSamplingSelector(model_ids=model_ids)


# ------------------------------------------------------------------
# Initialization
# ------------------------------------------------------------------

class TestInitialization:
    def test_uniform_priors(self, selector, model_ids):
        """All models start with identical Beta(1, 1) uniform priors."""
        stats = selector.get_stats()
        for mid in model_ids:
            assert stats[mid]["alpha"] == 1.0
            assert stats[mid]["beta"] == 1.0
            assert stats[mid]["mean"] == pytest.approx(0.5)

    def test_custom_priors(self):
        sel = ThompsonSamplingSelector(
            model_ids=["a", "b"],
            prior_alpha=2.0,
            prior_beta=5.0,
        )
        stats = sel.get_stats()
        assert stats["a"]["alpha"] == 2.0
        assert stats["a"]["beta"] == 5.0
        assert stats["a"]["mean"] == pytest.approx(2.0 / 7.0)

    def test_empty_model_ids_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            ThompsonSamplingSelector(model_ids=[])

    def test_invalid_prior_raises(self):
        with pytest.raises(ValueError, match="positive"):
            ThompsonSamplingSelector(model_ids=["a"], prior_alpha=0.0)

    def test_invalid_decay_raises(self):
        with pytest.raises(ValueError, match="decay"):
            ThompsonSamplingSelector(model_ids=["a"], decay=0.0)


# ------------------------------------------------------------------
# Update
# ------------------------------------------------------------------

class TestUpdate:
    def test_success_increases_alpha(self, selector):
        """A reward of 1.0 should increase alpha for that model."""
        before = selector.get_stats()["swing"]["alpha"]
        selector.update("swing", reward=1.0)
        after = selector.get_stats()["swing"]["alpha"]
        assert after > before

    def test_failure_increases_beta(self, selector):
        """A reward of 0.0 should increase beta for that model."""
        before = selector.get_stats()["swing"]["beta"]
        selector.update("swing", reward=0.0)
        after = selector.get_stats()["swing"]["beta"]
        assert after > before

    def test_success_does_not_increase_beta(self, selector):
        """A reward of 1.0 should NOT increase beta (only alpha changes)."""
        # With decay=0.995, beta will actually decrease slightly due to decay.
        # So we use decay=1.0 to isolate the update effect.
        sel = ThompsonSamplingSelector(
            model_ids=["a"], prior_alpha=1.0, prior_beta=1.0, decay=1.0
        )
        before = sel.get_stats()["a"]["beta"]
        sel.update("a", reward=1.0)
        after = sel.get_stats()["a"]["beta"]
        assert after == pytest.approx(before)

    def test_failure_does_not_increase_alpha(self):
        """A reward of 0.0 should NOT increase alpha (only beta changes)."""
        sel = ThompsonSamplingSelector(
            model_ids=["a"], prior_alpha=1.0, prior_beta=1.0, decay=1.0
        )
        before = sel.get_stats()["a"]["alpha"]
        sel.update("a", reward=0.0)
        after = sel.get_stats()["a"]["alpha"]
        assert after == pytest.approx(before)

    def test_continuous_reward(self, selector):
        """A reward of 0.5 should fractionally update both alpha and beta."""
        sel = ThompsonSamplingSelector(
            model_ids=["a"], prior_alpha=1.0, prior_beta=1.0, decay=1.0
        )
        sel.update("a", reward=0.5)
        stats = sel.get_stats()["a"]
        # alpha should have increased by 0.5
        assert stats["alpha"] == pytest.approx(1.5)
        # beta should have increased by 0.5
        assert stats["beta"] == pytest.approx(1.5)
        # Mean should still be 0.5 (symmetric update)
        assert stats["mean"] == pytest.approx(0.5)

    def test_continuous_reward_asymmetric(self):
        """A reward of 0.7 updates alpha more than beta."""
        sel = ThompsonSamplingSelector(
            model_ids=["a"], prior_alpha=1.0, prior_beta=1.0, decay=1.0
        )
        sel.update("a", reward=0.7)
        stats = sel.get_stats()["a"]
        assert stats["alpha"] == pytest.approx(1.7)
        assert stats["beta"] == pytest.approx(1.3)

    def test_unknown_model_raises(self, selector):
        with pytest.raises(ValueError, match="Unknown model_id"):
            selector.update("nonexistent", reward=1.0)

    def test_reward_out_of_range_raises(self, selector):
        with pytest.raises(ValueError, match="reward must be in"):
            selector.update("swing", reward=1.5)
        with pytest.raises(ValueError, match="reward must be in"):
            selector.update("swing", reward=-0.1)


# ------------------------------------------------------------------
# Select
# ------------------------------------------------------------------

class TestSelect:
    def test_returns_valid_model_id(self, selector, model_ids):
        """select() must return one of the known model_ids."""
        rng = np.random.default_rng(42)
        for _ in range(50):
            chosen = selector.select(rng=rng)
            assert chosen in model_ids

    def test_deterministic_with_seed(self, selector):
        """Same seed produces same selection."""
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)
        assert selector.select(rng=rng1) == selector.select(rng=rng2)

    def test_biased_selection_after_updates(self):
        """After many successes for one model, it should be selected more often."""
        sel = ThompsonSamplingSelector(
            model_ids=["good", "bad"], decay=1.0
        )
        # Give 'good' many successes, 'bad' many failures
        for _ in range(100):
            sel.update("good", reward=1.0)
            sel.update("bad", reward=0.0)

        rng = np.random.default_rng(42)
        counts = {"good": 0, "bad": 0}
        for _ in range(1000):
            counts[sel.select(rng=rng)] += 1

        # 'good' should be selected overwhelmingly
        assert counts["good"] > 900


# ------------------------------------------------------------------
# Weights
# ------------------------------------------------------------------

class TestGetWeights:
    def test_weights_sum_to_one(self, selector):
        """Weights must sum to 1.0."""
        weights = selector.get_weights()
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_uniform_weights_at_init(self, selector, model_ids):
        """Before any updates, all models should have equal weight."""
        weights = selector.get_weights()
        expected = 1.0 / len(model_ids)
        for mid in model_ids:
            assert weights[mid] == pytest.approx(expected, abs=0.01)

    def test_min_weight_floor_respected(self):
        """Even after many failures, a model's weight should not go below min_weight (renormalized)."""
        sel = ThompsonSamplingSelector(
            model_ids=["good", "bad"],
            decay=1.0,
            min_weight=0.1,
        )
        # Heavily skew toward 'good'
        for _ in range(200):
            sel.update("good", reward=1.0)
            sel.update("bad", reward=0.0)

        weights = sel.get_weights()
        # 'bad' posterior mean is tiny, but floor should keep it at
        # at least min_weight / (min_weight + good_weight) after renorm.
        # With floor=0.1, the minimum renormalized weight is 0.1/(0.1+anything) > 0.
        # More precisely: 'bad' raw is clamped to 0.1, 'good' raw is ~1.0,
        # after renorm: bad >= 0.1/1.1 ~= 0.09
        assert weights["bad"] >= 0.05
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_convergence_to_best_model(self):
        """After many successes for one model, it should get the highest weight."""
        sel = ThompsonSamplingSelector(
            model_ids=["A", "B", "C"],
            decay=1.0,
            min_weight=0.05,
        )
        for _ in range(100):
            sel.update("A", reward=1.0)
            sel.update("B", reward=0.5)
            sel.update("C", reward=0.0)

        weights = sel.get_weights()
        assert weights["A"] > weights["B"] > weights["C"]
        assert sum(weights.values()) == pytest.approx(1.0)

    def test_two_models_equal_updates(self):
        """Two models with identical updates should have equal weight."""
        sel = ThompsonSamplingSelector(
            model_ids=["X", "Y"], decay=1.0
        )
        for _ in range(50):
            sel.update("X", reward=0.7)
            sel.update("Y", reward=0.7)

        weights = sel.get_weights()
        assert weights["X"] == pytest.approx(weights["Y"], abs=0.01)


# ------------------------------------------------------------------
# Decay
# ------------------------------------------------------------------

class TestDecay:
    def test_decay_reduces_old_observations(self):
        """With decay < 1, old successes should be gradually forgotten."""
        sel_decay = ThompsonSamplingSelector(
            model_ids=["a"], prior_alpha=1.0, prior_beta=1.0, decay=0.9
        )
        sel_no_decay = ThompsonSamplingSelector(
            model_ids=["a"], prior_alpha=1.0, prior_beta=1.0, decay=1.0
        )

        # Feed identical rewards
        for _ in range(20):
            sel_decay.update("a", reward=1.0)
            sel_no_decay.update("a", reward=1.0)

        # With decay, effective alpha should be much smaller
        alpha_decay = sel_decay.get_stats()["a"]["alpha"]
        alpha_no_decay = sel_no_decay.get_stats()["a"]["alpha"]
        assert alpha_decay < alpha_no_decay

    def test_no_decay_preserves_counts(self):
        """With decay=1.0, counts accumulate without loss."""
        sel = ThompsonSamplingSelector(
            model_ids=["a"], prior_alpha=1.0, prior_beta=1.0, decay=1.0
        )
        for _ in range(10):
            sel.update("a", reward=1.0)

        stats = sel.get_stats()["a"]
        assert stats["alpha"] == pytest.approx(11.0)  # 1.0 prior + 10 successes
        assert stats["beta"] == pytest.approx(1.0)    # 1.0 prior + 0 failures

    def test_decay_enables_adaptation(self):
        """After regime change, decay allows the selector to adapt."""
        sel = ThompsonSamplingSelector(
            model_ids=["a", "b"], decay=0.95
        )

        # Phase 1: 'a' is good
        for _ in range(50):
            sel.update("a", reward=1.0)
            sel.update("b", reward=0.0)

        w1 = sel.get_weights()
        assert w1["a"] > w1["b"]

        # Phase 2: regime change, 'b' is now good
        for _ in range(100):
            sel.update("a", reward=0.0)
            sel.update("b", reward=1.0)

        w2 = sel.get_weights()
        # 'b' should now dominate (decay forgets old 'a' successes)
        assert w2["b"] > w2["a"]


# ------------------------------------------------------------------
# Persistence
# ------------------------------------------------------------------

class TestPersistence:
    def test_save_load_roundtrip(self, selector, tmp_path):
        """State should survive a save/load cycle exactly."""
        # Update some state
        selector.update("swing", reward=1.0)
        selector.update("timing", reward=0.0)
        selector.update("ensemble", reward=0.7)

        state_file = tmp_path / "thompson_state.json"
        selector.save_state(state_file)

        # Load into a fresh selector
        loaded = ThompsonSamplingSelector(model_ids=["placeholder"])
        loaded.load_state(state_file)

        # Compare
        assert loaded.model_ids == selector.model_ids
        assert loaded.decay == selector.decay
        assert loaded.min_weight == selector.min_weight
        assert loaded._n_updates == selector._n_updates

        for mid in selector.model_ids:
            assert loaded._alphas[mid] == pytest.approx(selector._alphas[mid])
            assert loaded._betas[mid] == pytest.approx(selector._betas[mid])

    def test_save_creates_parent_dirs(self, tmp_path):
        sel = ThompsonSamplingSelector(model_ids=["a"])
        deep_path = tmp_path / "sub" / "dir" / "state.json"
        sel.save_state(deep_path)
        assert deep_path.is_file()

    def test_load_missing_file_raises(self, selector, tmp_path):
        with pytest.raises(FileNotFoundError):
            selector.load_state(tmp_path / "does_not_exist.json")

    def test_saved_file_is_valid_json(self, selector, tmp_path):
        state_file = tmp_path / "state.json"
        selector.save_state(state_file)
        with open(state_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "model_ids" in data
        assert "alphas" in data
        assert "betas" in data
        assert "n_updates" in data


# ------------------------------------------------------------------
# Stats
# ------------------------------------------------------------------

class TestGetStats:
    def test_stats_keys(self, selector, model_ids):
        stats = selector.get_stats()
        assert set(stats.keys()) == set(model_ids)
        for mid in model_ids:
            assert set(stats[mid].keys()) == {"alpha", "beta", "mean", "variance"}

    def test_variance_formula(self):
        """Verify Beta variance = ab / ((a+b)^2 * (a+b+1))."""
        sel = ThompsonSamplingSelector(
            model_ids=["x"], prior_alpha=3.0, prior_beta=7.0, decay=1.0
        )
        stats = sel.get_stats()["x"]
        a, b = 3.0, 7.0
        expected_var = (a * b) / ((a + b) ** 2 * (a + b + 1))
        assert stats["variance"] == pytest.approx(expected_var)


# ------------------------------------------------------------------
# Repr
# ------------------------------------------------------------------

class TestRepr:
    def test_repr_contains_info(self, selector):
        r = repr(selector)
        assert "ThompsonSamplingSelector" in r
        assert "n_models=3" in r
        assert "n_updates=0" in r
