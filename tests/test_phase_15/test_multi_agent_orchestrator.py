"""Tests for MultiAgentOrchestrator — multi-perspective signal aggregator."""

import math
import pytest

from src.phase_15_strategy.multi_agent_orchestrator import (
    MultiAgentOrchestrator,
    TrendAgent,
    MeanReversionAgent,
    VolatilityAgent,
    FlowAgent,
    SentimentAgent,
    SignalType,
    _BaseAgent,
)


# ------------------------------------------------------------------
# Fixtures
# ------------------------------------------------------------------

@pytest.fixture
def orchestrator():
    """Default orchestrator with standard settings."""
    return MultiAgentOrchestrator()


@pytest.fixture
def bullish_features():
    """Feature set that should trigger a BUY consensus."""
    return {
        # Trend: strong uptrend
        "rsi_14": 68.0,
        "macd_signal": 0.08,
        "sma_20_50_ratio": 1.04,
        "sma_50_200_ratio": 1.03,
        "adx_14": 40.0,
        # Mean reversion: NOT overextended (keep near centre)
        "bb_position": 0.55,
        "zscore_20": 0.2,
        # Volatility: calm market (bullish)
        "atr_ratio": 0.7,
        "vix": 12.0,
        "vol_regime": -0.8,
        "realised_vol_20": 0.08,
        # Flow: strong accumulation
        "volume_ratio": 1.6,
        "obv_trend": 0.15,
        "mfi_14": 72.0,
        "vwap_deviation": 0.04,
        # Sentiment: clearly positive
        "fear_greed": 72.0,
        "put_call_ratio": 0.6,
        "net_sentiment": 0.6,
        "breadth": 0.72,
    }


@pytest.fixture
def bearish_features():
    """Feature set that should trigger a SELL consensus."""
    return {
        "rsi_14": 35.0,
        "macd_signal": -0.06,
        "sma_20_50_ratio": 0.97,
        "sma_50_200_ratio": 0.98,
        "adx_14": 30.0,
        "bb_position": 0.45,
        "zscore_20": -0.5,
        "atr_ratio": 1.6,
        "vix": 32.0,
        "vol_regime": 1.0,
        "realised_vol_20": 0.25,
        "volume_ratio": 1.5,
        "obv_trend": -0.12,
        "mfi_14": 35.0,
        "vwap_deviation": -0.03,
        "fear_greed": 25.0,
        "put_call_ratio": 1.4,
        "net_sentiment": -0.5,
        "breadth": 0.30,
    }


@pytest.fixture
def neutral_features():
    """Feature set with mixed signals — should produce HOLD."""
    return {
        "rsi_14": 50.0,
        "macd_signal": 0.0,
        "sma_20_50_ratio": 1.0,
        "sma_50_200_ratio": 1.0,
        "adx_14": 15.0,
        "bb_position": 0.5,
        "zscore_20": 0.0,
        "atr_ratio": 1.0,
        "vix": 20.0,
        "vol_regime": 0.0,
        "realised_vol_20": 0.15,
        "volume_ratio": 1.0,
        "obv_trend": 0.0,
        "mfi_14": 50.0,
        "vwap_deviation": 0.0,
        "fear_greed": 50.0,
        "put_call_ratio": 1.0,
        "net_sentiment": 0.0,
        "breadth": 0.5,
    }


# ==================================================================
# Construction & Defaults
# ==================================================================

class TestConstruction:
    def test_default_construction(self, orchestrator):
        """Default orchestrator has 5 agents and weights summing to 1."""
        assert len(orchestrator.agent_names) == 5
        assert sum(orchestrator.weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_custom_weights(self):
        """Custom weights are normalised to sum to 1."""
        orch = MultiAgentOrchestrator(
            agent_weights={"trend": 5.0, "flow": 3.0}
        )
        assert orch.weights["trend"] > orch.weights["flow"]
        assert sum(orch.weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_custom_thresholds(self):
        """Agreement threshold and min_confidence are stored."""
        orch = MultiAgentOrchestrator(
            agreement_threshold=0.8, min_confidence=0.5
        )
        assert orch.agreement_threshold == 0.8
        assert orch.min_confidence == 0.5

    def test_unknown_agent_weight_ignored(self):
        """Weights for non-existent agents are silently dropped."""
        orch = MultiAgentOrchestrator(
            agent_weights={"trend": 1.0, "nonexistent_agent": 99.0}
        )
        assert "nonexistent_agent" not in orch.weights
        assert "trend" in orch.weights


# ==================================================================
# Individual Agent Votes
# ==================================================================

class TestAgentVotes:
    def _assert_valid_vote(self, vote: float, conf: float):
        assert -1.0 <= vote <= 1.0, f"vote {vote} out of [-1,1]"
        assert 0.0 <= conf <= 1.0, f"confidence {conf} out of [0,1]"

    def test_trend_agent_valid_bounds(self, bullish_features):
        """TrendAgent vote is in [-1,1] and confidence in [0,1]."""
        agent = TrendAgent()
        vote, conf = agent.evaluate(bullish_features)
        self._assert_valid_vote(vote, conf)

    def test_mean_reversion_agent_valid_bounds(self, bullish_features):
        """MeanReversionAgent vote is in [-1,1] and confidence in [0,1]."""
        agent = MeanReversionAgent()
        vote, conf = agent.evaluate(bullish_features)
        self._assert_valid_vote(vote, conf)

    def test_volatility_agent_valid_bounds(self, bullish_features):
        """VolatilityAgent vote is in [-1,1] and confidence in [0,1]."""
        agent = VolatilityAgent()
        vote, conf = agent.evaluate(bullish_features)
        self._assert_valid_vote(vote, conf)

    def test_flow_agent_valid_bounds(self, bullish_features):
        """FlowAgent vote is in [-1,1] and confidence in [0,1]."""
        agent = FlowAgent()
        vote, conf = agent.evaluate(bullish_features)
        self._assert_valid_vote(vote, conf)

    def test_sentiment_agent_valid_bounds(self, bullish_features):
        """SentimentAgent vote is in [-1,1] and confidence in [0,1]."""
        agent = SentimentAgent()
        vote, conf = agent.evaluate(bullish_features)
        self._assert_valid_vote(vote, conf)

    def test_trend_agent_bullish_on_uptrend(self):
        """TrendAgent should vote positive when RSI/MACD/SMA all bullish."""
        agent = TrendAgent()
        features = {
            "rsi_14": 70.0,
            "macd_signal": 0.08,
            "sma_20_50_ratio": 1.05,
            "sma_50_200_ratio": 1.03,
            "adx_14": 40.0,
        }
        vote, conf = agent.evaluate(features)
        assert vote > 0.0, "Trend agent should be bullish"
        assert conf > 0.3, "High ADX should raise confidence"

    def test_mean_reversion_agent_bullish_on_oversold(self):
        """MeanReversionAgent should vote positive (buy) when RSI oversold."""
        agent = MeanReversionAgent()
        features = {"rsi_14": 20.0, "bb_position": 0.05, "zscore_20": -2.5}
        vote, conf = agent.evaluate(features)
        assert vote > 0.0, "Contrarian agent should buy oversold"

    def test_volatility_agent_bearish_on_high_vix(self):
        """VolatilityAgent should vote negative (risk-off) when VIX is high."""
        agent = VolatilityAgent()
        features = {"vix": 40.0, "atr_ratio": 2.0, "vol_regime": 1.0}
        vote, conf = agent.evaluate(features)
        assert vote < 0.0, "High VIX should trigger bearish/risk-off vote"


# ==================================================================
# Orchestrator Evaluation
# ==================================================================

class TestEvaluation:
    def test_evaluate_returns_required_keys(self, orchestrator, bullish_features):
        """Evaluate returns all documented keys."""
        result = orchestrator.evaluate(bullish_features)
        required_keys = {
            "signal", "signal_type", "strength", "confidence",
            "agent_votes", "agreement_ratio", "dissenting_agents",
        }
        assert required_keys.issubset(result.keys())

    def test_buy_signal_on_bullish_consensus(self, orchestrator, bullish_features):
        """Majority bullish agents should produce a BUY signal."""
        result = orchestrator.evaluate(bullish_features)
        assert result["signal"] == "BUY"
        assert result["signal_type"] == SignalType.BUY
        assert result["strength"] > 0.0

    def test_sell_signal_on_bearish_consensus(self, orchestrator, bearish_features):
        """Majority bearish agents should produce a SELL signal."""
        result = orchestrator.evaluate(bearish_features)
        assert result["signal"] == "SELL"
        assert result["signal_type"] == SignalType.SELL
        assert result["strength"] < 0.0

    def test_hold_signal_on_neutral(self, orchestrator, neutral_features):
        """Mixed/neutral features should produce a HOLD signal."""
        result = orchestrator.evaluate(neutral_features)
        assert result["signal"] == "HOLD"
        assert result["signal_type"] == SignalType.HOLD

    def test_agreement_ratio_in_range(self, orchestrator, bullish_features):
        """Agreement ratio must be in [0, 1]."""
        result = orchestrator.evaluate(bullish_features)
        assert 0.0 <= result["agreement_ratio"] <= 1.0

    def test_strength_in_range(self, orchestrator, bullish_features):
        """Aggregated strength must be in [-1, 1]."""
        result = orchestrator.evaluate(bullish_features)
        assert -1.0 <= result["strength"] <= 1.0

    def test_confidence_in_range(self, orchestrator, bullish_features):
        """Overall confidence must be in [0, 1]."""
        result = orchestrator.evaluate(bullish_features)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_all_agent_votes_present(self, orchestrator, bullish_features):
        """agent_votes dict must contain all 5 agents."""
        result = orchestrator.evaluate(bullish_features)
        assert set(result["agent_votes"].keys()) == set(orchestrator.agent_names)

    def test_dissenting_agents_are_strings(self, orchestrator, bullish_features):
        """Dissenting agents list must contain strings (agent names)."""
        result = orchestrator.evaluate(bullish_features)
        assert isinstance(result["dissenting_agents"], list)
        for name in result["dissenting_agents"]:
            assert isinstance(name, str)
            assert name in orchestrator.agent_names

    def test_high_agreement_threshold_produces_hold(self, bullish_features):
        """With an impossibly high agreement threshold, everything is HOLD."""
        orch = MultiAgentOrchestrator(agreement_threshold=1.1)
        result = orch.evaluate(bullish_features)
        assert result["signal"] == "HOLD"

    def test_high_min_confidence_produces_hold(self, bullish_features):
        """With an impossibly high min_confidence, everything is HOLD."""
        orch = MultiAgentOrchestrator(min_confidence=2.0)
        result = orch.evaluate(bullish_features)
        assert result["signal"] == "HOLD"


# ==================================================================
# Weight Updates (Bandit-style)
# ==================================================================

class TestWeightUpdates:
    def test_positive_reward_increases_weight(self, orchestrator):
        """Rewarding an agent should increase its relative weight."""
        old_weight = orchestrator.weights["trend"]
        orchestrator.update_weights("trend", reward=1.0)
        new_weight = orchestrator.weights["trend"]
        assert new_weight > old_weight

    def test_negative_reward_decreases_weight(self, orchestrator):
        """Penalising an agent should decrease its relative weight."""
        old_weight = orchestrator.weights["trend"]
        orchestrator.update_weights("trend", reward=-1.0)
        new_weight = orchestrator.weights["trend"]
        assert new_weight < old_weight

    def test_weights_sum_to_one_after_update(self, orchestrator):
        """Weights must always sum to 1 after any update."""
        orchestrator.update_weights("flow", reward=0.8)
        orchestrator.update_weights("sentiment", reward=-0.5)
        assert sum(orchestrator.weights.values()) == pytest.approx(1.0, abs=1e-9)

    def test_unknown_agent_update_is_noop(self, orchestrator):
        """Updating a non-existent agent should not crash or change weights."""
        before = dict(orchestrator.weights)
        orchestrator.update_weights("nonexistent_agent", reward=1.0)
        assert orchestrator.weights == before

    def test_repeated_rewards_shift_weights_significantly(self, orchestrator):
        """Many positive rewards should make an agent clearly dominant."""
        for _ in range(20):
            orchestrator.update_weights("trend", reward=1.0)
            orchestrator.update_weights("flow", reward=-0.5)
        assert orchestrator.weights["trend"] > orchestrator.weights["flow"]


# ==================================================================
# Performance Tracking
# ==================================================================

class TestPerformanceTracking:
    def test_initial_performance_zeros(self, orchestrator):
        """Before any evaluations, all stats should be at zero."""
        perf = orchestrator.get_agent_performance()
        for name in orchestrator.agent_names:
            assert perf[name]["total_evaluations"] == 0
            assert perf[name]["cumulative_reward"] == 0.0

    def test_evaluation_increments_count(self, orchestrator, bullish_features):
        """Each evaluate() call should increment total_evaluations for all agents."""
        orchestrator.evaluate(bullish_features)
        orchestrator.evaluate(bullish_features)
        perf = orchestrator.get_agent_performance()
        for name in orchestrator.agent_names:
            assert perf[name]["total_evaluations"] == 2

    def test_accuracy_after_updates(self, orchestrator):
        """Accuracy should reflect correct_calls / total_updates."""
        orchestrator.update_weights("trend", reward=1.0)
        orchestrator.update_weights("trend", reward=1.0)
        orchestrator.update_weights("trend", reward=-1.0)
        perf = orchestrator.get_agent_performance()
        # 2 positive out of 3 total
        assert perf["trend"]["accuracy"] == pytest.approx(2.0 / 3.0, abs=1e-9)

    def test_performance_contains_all_agents(self, orchestrator):
        """Performance dict must have an entry for every agent."""
        perf = orchestrator.get_agent_performance()
        assert set(perf.keys()) == set(orchestrator.agent_names)

    def test_weight_in_performance(self, orchestrator, bullish_features):
        """Performance report includes current weight."""
        orchestrator.evaluate(bullish_features)
        perf = orchestrator.get_agent_performance()
        weight_sum = sum(p["weight"] for p in perf.values())
        assert weight_sum == pytest.approx(1.0, abs=1e-9)


# ==================================================================
# Edge Cases
# ==================================================================

class TestEdgeCases:
    def test_empty_features(self, orchestrator):
        """Passing an empty dict should not crash; returns HOLD."""
        result = orchestrator.evaluate({})
        assert result["signal"] in ("BUY", "SELL", "HOLD")
        assert -1.0 <= result["strength"] <= 1.0

    def test_none_features(self, orchestrator):
        """Passing None features should not crash."""
        result = orchestrator.evaluate(None)
        assert result["signal"] in ("BUY", "SELL", "HOLD")

    def test_missing_features_default_gracefully(self, orchestrator):
        """Partial features should still produce a valid signal."""
        partial = {"rsi_14": 55.0, "vix": 18.0}
        result = orchestrator.evaluate(partial)
        assert "signal" in result
        assert -1.0 <= result["strength"] <= 1.0

    def test_nan_features_handled(self, orchestrator):
        """NaN feature values should be treated as defaults."""
        features = {"rsi_14": float("nan"), "vix": float("nan")}
        result = orchestrator.evaluate(features)
        assert not math.isnan(result["strength"])
        assert not math.isnan(result["confidence"])

    def test_inf_features_handled(self, orchestrator):
        """Inf feature values should be treated as defaults."""
        features = {"rsi_14": float("inf"), "vix": float("-inf")}
        result = orchestrator.evaluate(features)
        assert not math.isinf(result["strength"])
        assert not math.isinf(result["confidence"])

    def test_string_feature_values_handled(self, orchestrator):
        """Non-numeric feature values should fall back to defaults."""
        features = {"rsi_14": "not_a_number", "vix": None}
        result = orchestrator.evaluate(features)
        assert "signal" in result


# ==================================================================
# _BaseAgent utility
# ==================================================================

class TestBaseAgentUtils:
    def test_clamp_within_range(self):
        assert _BaseAgent._clamp(0.5) == 0.5
        assert _BaseAgent._clamp(-2.0) == -1.0
        assert _BaseAgent._clamp(3.0) == 1.0

    def test_safe_get_returns_default_on_missing(self):
        assert _BaseAgent._safe_get({}, "missing", 42.0) == 42.0

    def test_safe_get_returns_default_on_none(self):
        assert _BaseAgent._safe_get({"k": None}, "k", 7.0) == 7.0

    def test_safe_get_handles_nan(self):
        assert _BaseAgent._safe_get({"k": float("nan")}, "k", 0.0) == 0.0
