"""
GIGA TRADER - Multi-Agent Signal Orchestrator
===============================================
A multi-perspective signal aggregator that combines several rule-based
"agent" viewpoints into a single consensus signal.

Each agent represents a different market analysis perspective:
  1. TrendAgent        -- Evaluates trend strength from momentum indicators
  2. MeanReversionAgent -- Detects overextended moves likely to reverse
  3. VolatilityAgent    -- Assesses risk from volatility regime
  4. FlowAgent          -- Reads order flow and volume signals
  5. SentimentAgent     -- Interprets fear/greed and sentiment indicators

Each agent produces:
  - vote: float in [-1, +1]  (bearish to bullish)
  - confidence: float in [0, 1]

The orchestrator aggregates votes using confidence-weighted voting with
exponential-moving-average weight adaptation (bandit-style).

IMPORTANT: This module is pure Python/numpy.  No LLM API calls, no deep
learning dependencies.  It is a deterministic, rule-based system.
"""

import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Integration with existing SignalType enum --------------------------------
try:
    from src.phase_19_paper_trading.alpaca_client import SignalType
except Exception:  # pragma: no cover – optional dependency
    from enum import Enum

    class SignalType(Enum):  # type: ignore[no-redef]
        BUY = "BUY"
        SELL = "SELL"
        HOLD = "HOLD"
        CLOSE = "CLOSE"


logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT BASE CLASS
# ═══════════════════════════════════════════════════════════════════════════════

class _BaseAgent:
    """Abstract base for all perspective agents.

    Subclasses MUST override ``evaluate(features) -> (vote, confidence)``.
    """

    name: str = "base"

    def evaluate(self, features: Dict[str, Any]) -> Tuple[float, float]:
        """Return (vote, confidence) for the given feature snapshot.

        Parameters
        ----------
        features : dict
            Arbitrary feature dict – each agent picks the keys it needs.

        Returns
        -------
        vote : float
            Value in [-1.0, +1.0].  Positive = bullish, negative = bearish.
        confidence : float
            Value in [0.0, 1.0].  How sure the agent is about its vote.
        """
        raise NotImplementedError

    # Utility -----------------------------------------------------------------

    @staticmethod
    def _clamp(value: float, lo: float = -1.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, value))

    @staticmethod
    def _safe_get(features: Dict[str, Any], key: str, default: float = 0.0) -> float:
        """Safely retrieve a numeric feature, returning *default* on miss."""
        val = features.get(key, default)
        if val is None:
            return default
        try:
            fval = float(val)
            if math.isnan(fval) or math.isinf(fval):
                return default
            return fval
        except (TypeError, ValueError):
            return default


# ═══════════════════════════════════════════════════════════════════════════════
# AGENT IMPLEMENTATIONS
# ═══════════════════════════════════════════════════════════════════════════════

class TrendAgent(_BaseAgent):
    """Evaluate trend strength from momentum indicators.

    Reads: rsi_14, macd_signal, sma_20_50_ratio, sma_50_200_ratio, adx_14
    """

    name = "trend"

    def evaluate(self, features: Dict[str, Any]) -> Tuple[float, float]:
        rsi = self._safe_get(features, "rsi_14", 50.0)
        macd_sig = self._safe_get(features, "macd_signal", 0.0)
        sma_20_50 = self._safe_get(features, "sma_20_50_ratio", 1.0)
        sma_50_200 = self._safe_get(features, "sma_50_200_ratio", 1.0)
        adx = self._safe_get(features, "adx_14", 20.0)

        # RSI component: map [0,100] -> [-1,+1], centre at 50
        rsi_vote = (rsi - 50.0) / 50.0

        # MACD signal: positive = bullish, clip to [-1,1]
        macd_vote = self._clamp(macd_sig * 10.0)

        # SMA ratio: >1 = bullish, <1 = bearish
        sma_short_vote = self._clamp((sma_20_50 - 1.0) * 20.0)
        sma_long_vote = self._clamp((sma_50_200 - 1.0) * 20.0)

        # Weighted combination
        vote = (
            0.25 * rsi_vote
            + 0.25 * macd_vote
            + 0.25 * sma_short_vote
            + 0.25 * sma_long_vote
        )
        vote = self._clamp(vote)

        # Confidence rises with ADX (trend strength). ADX > 25 = trending.
        adx_norm = self._clamp(adx / 50.0, 0.0, 1.0)
        confidence = 0.3 + 0.7 * adx_norm  # floor at 0.3

        return vote, confidence


class MeanReversionAgent(_BaseAgent):
    """Detect overextended moves likely to reverse.

    Reads: bb_position (0-1, Bollinger band percentile), rsi_14, zscore_20
    """

    name = "mean_reversion"

    def evaluate(self, features: Dict[str, Any]) -> Tuple[float, float]:
        bb_pos = self._safe_get(features, "bb_position", 0.5)
        rsi = self._safe_get(features, "rsi_14", 50.0)
        zscore = self._safe_get(features, "zscore_20", 0.0)

        # Contrarian signals: overbought -> sell, oversold -> buy
        # bb_position: 0 = at lower band, 1 = at upper band
        bb_vote = -(bb_pos - 0.5) * 2.0  # maps [0,1] -> [+1,-1]

        # RSI contrarian
        if rsi > 70:
            rsi_vote = -(rsi - 70) / 30.0  # negative (sell), up to -1
        elif rsi < 30:
            rsi_vote = (30 - rsi) / 30.0   # positive (buy), up to +1
        else:
            rsi_vote = 0.0

        # Z-score: high positive -> overbought (sell), vice versa
        zscore_vote = self._clamp(-zscore / 3.0)

        vote = 0.35 * bb_vote + 0.35 * rsi_vote + 0.30 * zscore_vote
        vote = self._clamp(vote)

        # Confidence is higher at extremes
        extremity = max(abs(bb_pos - 0.5) * 2.0, abs(zscore) / 3.0, 0.0)
        confidence = self._clamp(0.2 + 0.8 * extremity, 0.0, 1.0)

        return vote, confidence


class VolatilityAgent(_BaseAgent):
    """Assess risk from volatility regime.

    Reads: atr_ratio, vix, vol_regime, realised_vol_20
    """

    name = "volatility"

    def evaluate(self, features: Dict[str, Any]) -> Tuple[float, float]:
        atr_ratio = self._safe_get(features, "atr_ratio", 1.0)
        vix = self._safe_get(features, "vix", 20.0)
        vol_regime = self._safe_get(features, "vol_regime", 0.0)
        realised_vol = self._safe_get(features, "realised_vol_20", 0.15)

        # High vol -> mildly bearish (risk-off); low vol -> mildly bullish
        vix_vote = self._clamp(-(vix - 20.0) / 30.0)

        # ATR ratio > 1.5 -> elevated vol -> bearish bias
        atr_vote = self._clamp(-(atr_ratio - 1.0) * 1.5)

        # Vol regime: positive = high vol (bearish), negative = low vol
        regime_vote = self._clamp(-vol_regime * 0.5)

        vote = 0.40 * vix_vote + 0.30 * atr_vote + 0.30 * regime_vote
        vote = self._clamp(vote)

        # Confidence is higher when vol is extreme (either direction)
        vol_extremity = max(abs(vix - 20.0) / 30.0, abs(atr_ratio - 1.0), 0.0)
        confidence = self._clamp(0.3 + 0.5 * vol_extremity, 0.0, 1.0)

        return vote, confidence


class FlowAgent(_BaseAgent):
    """Read order flow and volume signals.

    Reads: volume_ratio, obv_trend, mfi_14, vwap_deviation
    """

    name = "flow"

    def evaluate(self, features: Dict[str, Any]) -> Tuple[float, float]:
        vol_ratio = self._safe_get(features, "volume_ratio", 1.0)
        obv_trend = self._safe_get(features, "obv_trend", 0.0)
        mfi = self._safe_get(features, "mfi_14", 50.0)
        vwap_dev = self._safe_get(features, "vwap_deviation", 0.0)

        # OBV trend: positive = accumulation (bullish)
        obv_vote = self._clamp(obv_trend * 5.0)

        # MFI (money flow index): similar to RSI but volume-weighted
        mfi_vote = (mfi - 50.0) / 50.0

        # VWAP deviation: positive = above VWAP (bullish short term)
        vwap_vote = self._clamp(vwap_dev * 10.0)

        vote = 0.40 * obv_vote + 0.35 * mfi_vote + 0.25 * vwap_vote
        vote = self._clamp(vote)

        # Confidence rises with volume (more volume = more meaningful signal)
        vol_conf = self._clamp(vol_ratio / 2.0, 0.0, 1.0)
        confidence = 0.2 + 0.6 * vol_conf

        return vote, confidence


class SentimentAgent(_BaseAgent):
    """Interpret fear/greed and sentiment indicators.

    Reads: fear_greed, put_call_ratio, net_sentiment, breadth
    """

    name = "sentiment"

    def evaluate(self, features: Dict[str, Any]) -> Tuple[float, float]:
        fear_greed = self._safe_get(features, "fear_greed", 50.0)
        pcr = self._safe_get(features, "put_call_ratio", 1.0)
        net_sent = self._safe_get(features, "net_sentiment", 0.0)
        breadth = self._safe_get(features, "breadth", 0.5)

        # Fear/greed: 0 = extreme fear (contrarian buy), 100 = extreme greed
        fg_vote = (fear_greed - 50.0) / 50.0

        # Put/call ratio: >1.2 = fearful (contrarian bullish), <0.7 = complacent
        pcr_vote = self._clamp(-(pcr - 1.0) * 2.0)

        # Net sentiment: direct signal
        sent_vote = self._clamp(net_sent)

        # Breadth: >0.5 = broad participation (bullish)
        breadth_vote = (breadth - 0.5) * 2.0

        vote = (
            0.25 * fg_vote
            + 0.25 * pcr_vote
            + 0.30 * sent_vote
            + 0.20 * breadth_vote
        )
        vote = self._clamp(vote)

        # Confidence from how extreme the readings are
        extremity = max(
            abs(fear_greed - 50.0) / 50.0,
            abs(pcr - 1.0),
            abs(net_sent),
        )
        confidence = self._clamp(0.2 + 0.6 * extremity, 0.0, 1.0)

        return vote, confidence


# ═══════════════════════════════════════════════════════════════════════════════
# DEFAULT AGENT REGISTRY
# ═══════════════════════════════════════════════════════════════════════════════

_DEFAULT_AGENTS: Dict[str, _BaseAgent] = {
    "trend": TrendAgent(),
    "mean_reversion": MeanReversionAgent(),
    "volatility": VolatilityAgent(),
    "flow": FlowAgent(),
    "sentiment": SentimentAgent(),
}

_DEFAULT_WEIGHTS: Dict[str, float] = {
    "trend": 0.25,
    "mean_reversion": 0.20,
    "volatility": 0.20,
    "flow": 0.15,
    "sentiment": 0.20,
}


# ═══════════════════════════════════════════════════════════════════════════════
# MULTI-AGENT ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

class MultiAgentOrchestrator:
    """Aggregate multiple rule-based agent viewpoints into a consensus signal.

    Parameters
    ----------
    agent_weights : dict, optional
        Mapping of agent name -> initial weight.  Missing agents get equal
        share of remaining weight.  All weights are normalised internally.
    agreement_threshold : float
        Minimum fraction of agents whose vote direction must agree for a
        non-HOLD signal.  Default 0.6.
    min_confidence : float
        Minimum confidence-weighted strength to produce a BUY/SELL signal.
        Below this threshold the result is HOLD.  Default 0.3.
    ema_alpha : float
        Exponential moving average decay for bandit-style weight updates.
        Higher = faster adaptation.  Default 0.1.
    """

    def __init__(
        self,
        agent_weights: Optional[Dict[str, float]] = None,
        agreement_threshold: float = 0.6,
        min_confidence: float = 0.3,
        ema_alpha: float = 0.1,
    ):
        # Agents
        self._agents: Dict[str, _BaseAgent] = dict(_DEFAULT_AGENTS)

        # Weights – normalise so they sum to 1
        if agent_weights is not None:
            raw = {k: max(v, 0.0) for k, v in agent_weights.items() if k in self._agents}
            # Fill in missing agents with a small floor weight
            for name in self._agents:
                if name not in raw:
                    raw[name] = 0.01
        else:
            raw = dict(_DEFAULT_WEIGHTS)
        self._weights = self._normalise(raw)

        # Thresholds
        self.agreement_threshold = agreement_threshold
        self.min_confidence = min_confidence
        self.ema_alpha = ema_alpha

        # Performance tracking
        self._performance: Dict[str, Dict[str, Any]] = {
            name: {
                "total_evaluations": 0,
                "cumulative_reward": 0.0,
                "ema_reward": 0.0,
                "correct_calls": 0,
                "total_updates": 0,
            }
            for name in self._agents
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Run all agents and return an aggregated signal.

        Parameters
        ----------
        features : dict
            Flat dict of feature names to numeric values.

        Returns
        -------
        dict with keys:
            signal      : str  ("BUY", "SELL", or "HOLD")
            signal_type : SignalType
            strength    : float  (-1 to +1)
            confidence  : float  (0 to 1)
            agent_votes : dict  {agent_name: {"vote": float, "confidence": float}}
            agreement_ratio : float
            dissenting_agents : list[str]
        """
        if features is None:
            features = {}

        # Collect votes
        agent_votes: Dict[str, Dict[str, float]] = {}
        for name, agent in self._agents.items():
            try:
                vote, conf = agent.evaluate(features)
            except Exception:
                logger.warning("Agent %s raised an exception; defaulting to neutral", name)
                vote, conf = 0.0, 0.0
            # Enforce bounds
            vote = _BaseAgent._clamp(vote, -1.0, 1.0)
            conf = _BaseAgent._clamp(conf, 0.0, 1.0)
            agent_votes[name] = {"vote": vote, "confidence": conf}

            # Track evaluations
            self._performance[name]["total_evaluations"] += 1

        # Weighted aggregation ------------------------------------------
        weighted_sum = 0.0
        weight_total = 0.0
        for name, vdict in agent_votes.items():
            w = self._weights.get(name, 0.0)
            weighted_sum += w * vdict["vote"] * vdict["confidence"]
            weight_total += w * vdict["confidence"]

        if weight_total > 0:
            strength = weighted_sum / weight_total
        else:
            strength = 0.0
        strength = _BaseAgent._clamp(strength, -1.0, 1.0)

        # Overall confidence: average agent confidence weighted by agent weight
        conf_sum = sum(
            self._weights.get(n, 0.0) * v["confidence"]
            for n, v in agent_votes.items()
        )
        w_sum = sum(self._weights.get(n, 0.0) for n in agent_votes)
        overall_confidence = conf_sum / w_sum if w_sum > 0 else 0.0
        overall_confidence = _BaseAgent._clamp(overall_confidence, 0.0, 1.0)

        # Agreement / dissent -------------------------------------------
        direction = 1.0 if strength >= 0 else -1.0
        n_agree = 0
        dissenting: List[str] = []
        for name, vdict in agent_votes.items():
            if vdict["vote"] == 0.0:
                # Neutral agents don't count as dissenting
                continue
            if (vdict["vote"] > 0 and direction > 0) or (vdict["vote"] < 0 and direction < 0):
                n_agree += 1
            else:
                dissenting.append(name)

        n_opinionated = n_agree + len(dissenting)
        agreement_ratio = n_agree / n_opinionated if n_opinionated > 0 else 0.0

        # Signal decision -----------------------------------------------
        abs_strength = abs(strength)
        if (
            agreement_ratio >= self.agreement_threshold
            and abs_strength >= self.min_confidence
        ):
            if strength > 0:
                signal_str = "BUY"
                signal_type = SignalType.BUY
            else:
                signal_str = "SELL"
                signal_type = SignalType.SELL
        else:
            signal_str = "HOLD"
            signal_type = SignalType.HOLD

        return {
            "signal": signal_str,
            "signal_type": signal_type,
            "strength": strength,
            "confidence": overall_confidence,
            "agent_votes": agent_votes,
            "agreement_ratio": agreement_ratio,
            "dissenting_agents": dissenting,
        }

    def update_weights(self, agent_name: str, reward: float) -> None:
        """Update an agent's weight based on outcome (bandit-style EMA).

        Parameters
        ----------
        agent_name : str
            Name of the agent to reward/penalise.
        reward : float
            Reward value; positive = correct, negative = incorrect.
            Typically in [-1, +1].
        """
        if agent_name not in self._agents:
            logger.warning("update_weights: unknown agent '%s'", agent_name)
            return

        perf = self._performance[agent_name]
        perf["total_updates"] += 1
        perf["cumulative_reward"] += reward
        if reward > 0:
            perf["correct_calls"] += 1

        # EMA update of tracked reward
        perf["ema_reward"] = (
            self.ema_alpha * reward + (1 - self.ema_alpha) * perf["ema_reward"]
        )

        # Adjust weight proportional to EMA reward.
        # Shift EMA reward from [-1,1] -> [0.5, 1.5] as a multiplier.
        multiplier = 1.0 + 0.5 * _BaseAgent._clamp(perf["ema_reward"], -1.0, 1.0)
        self._weights[agent_name] *= multiplier

        # Re-normalise
        self._weights = self._normalise(self._weights)

    def get_agent_performance(self) -> Dict[str, Dict[str, Any]]:
        """Return performance stats for each agent.

        Returns
        -------
        dict mapping agent name -> stats dict with keys:
            weight, total_evaluations, total_updates, cumulative_reward,
            ema_reward, correct_calls, accuracy
        """
        result: Dict[str, Dict[str, Any]] = {}
        for name in self._agents:
            perf = self._performance[name]
            updates = perf["total_updates"]
            result[name] = {
                "weight": self._weights.get(name, 0.0),
                "total_evaluations": perf["total_evaluations"],
                "total_updates": updates,
                "cumulative_reward": perf["cumulative_reward"],
                "ema_reward": perf["ema_reward"],
                "correct_calls": perf["correct_calls"],
                "accuracy": (
                    perf["correct_calls"] / updates if updates > 0 else 0.0
                ),
            }
        return result

    @property
    def weights(self) -> Dict[str, float]:
        """Current agent weights (read-only copy)."""
        return dict(self._weights)

    @property
    def agent_names(self) -> List[str]:
        """Ordered list of agent names."""
        return list(self._agents.keys())

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _normalise(raw: Dict[str, float]) -> Dict[str, float]:
        """Normalise weights so they sum to 1.0."""
        total = sum(raw.values())
        if total <= 0:
            n = len(raw) or 1
            return {k: 1.0 / n for k in raw}
        return {k: v / total for k, v in raw.items()}
