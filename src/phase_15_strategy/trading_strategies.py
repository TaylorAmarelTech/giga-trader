"""
GIGA TRADER - Trading Strategies
==================================
Base strategy class and all 5 strategy implementations.

Contains:
- BaseStrategy (ABC)
- MomentumStrategy
- ContrarianStrategy
- RegimeFollowerStrategy
- MeanReversionStrategy
- LeadLagStrategy
"""

import numpy as np
import pandas as pd
from typing import Dict, Any
from collections import deque
from abc import ABC, abstractmethod

from src.phase_15_strategy.signal_data import (
    SignalDirection,
    StrategySignal,
    StrategyPerformance,
)


# =============================================================================
# BASE STRATEGY CLASS
# =============================================================================

class BaseStrategy(ABC):
    """Base class for all strategies."""

    def __init__(self, name: str):
        self.name = name
        self.performance = StrategyPerformance(strategy_name=name)

    @abstractmethod
    def generate_signal(
        self,
        sentiment_data: Dict[str, Any],
        market_data: Dict[str, Any],
        regime: str,
    ) -> StrategySignal:
        """Generate a trading signal."""
        pass

    def validate_on_universe(
        self,
        universe_data: pd.DataFrame,
        universe_name: str,
    ) -> float:
        """
        Validate strategy on a synthetic universe.

        Returns accuracy on that universe.
        """
        # Subclasses can override for specific validation
        return 0.5  # Default: random


# =============================================================================
# STRATEGY IMPLEMENTATIONS
# =============================================================================

class MomentumStrategy(BaseStrategy):
    """
    Follow strong trends with sentiment confirmation.

    BUY when: sentiment rising AND price momentum positive
    SELL when: sentiment falling AND price momentum negative
    """

    def __init__(self, lookback: int = 5, threshold: float = 0.3):
        super().__init__("momentum")
        self.lookback = lookback
        self.threshold = threshold
        self.sentiment_history = deque(maxlen=20)

    def generate_signal(
        self,
        sentiment_data: Dict[str, Any],
        market_data: Dict[str, Any],
        regime: str,
    ) -> StrategySignal:
        net_sentiment = sentiment_data.get("net_sentiment", 0)
        self.sentiment_history.append(net_sentiment)

        if len(self.sentiment_history) < self.lookback:
            return StrategySignal(
                direction=SignalDirection.NEUTRAL,
                confidence=0.0,
                raw_score=0.0,
                strategy_name=self.name,
            )

        # Calculate sentiment momentum
        recent = list(self.sentiment_history)[-self.lookback:]
        old = list(self.sentiment_history)[:-self.lookback] if len(self.sentiment_history) > self.lookback else [0]

        sentiment_momentum = np.mean(recent) - np.mean(old)

        # Get price momentum if available
        price_momentum = market_data.get("momentum_5d", 0)

        # Combined score
        raw_score = 0.5 * sentiment_momentum + 0.5 * price_momentum

        # Determine direction
        if raw_score > self.threshold:
            direction = SignalDirection.STRONG_BUY if raw_score > self.threshold * 2 else SignalDirection.BUY
        elif raw_score < -self.threshold:
            direction = SignalDirection.STRONG_SELL if raw_score < -self.threshold * 2 else SignalDirection.SELL
        else:
            direction = SignalDirection.NEUTRAL

        confidence = min(1.0, abs(raw_score) / self.threshold)

        return StrategySignal(
            direction=direction,
            confidence=confidence,
            raw_score=raw_score,
            strategy_name=self.name,
            metadata={"sentiment_momentum": sentiment_momentum, "price_momentum": price_momentum},
        )


class ContrarianStrategy(BaseStrategy):
    """
    Fade extreme sentiment (mean reversion).

    SELL when: sentiment extremely bullish (crowd is wrong at extremes)
    BUY when: sentiment extremely bearish
    """

    def __init__(self, extreme_threshold: float = 0.7, lookback: int = 20):
        super().__init__("contrarian")
        self.extreme_threshold = extreme_threshold
        self.lookback = lookback
        self.sentiment_history = deque(maxlen=50)

    def generate_signal(
        self,
        sentiment_data: Dict[str, Any],
        market_data: Dict[str, Any],
        regime: str,
    ) -> StrategySignal:
        net_sentiment = sentiment_data.get("net_sentiment", 0)
        self.sentiment_history.append(net_sentiment)

        if len(self.sentiment_history) < self.lookback:
            return StrategySignal(
                direction=SignalDirection.NEUTRAL,
                confidence=0.0,
                raw_score=0.0,
                strategy_name=self.name,
            )

        # Calculate z-score of current sentiment
        recent = list(self.sentiment_history)[-self.lookback:]
        mean_sent = np.mean(recent)
        std_sent = np.std(recent) + 1e-6
        z_score = (net_sentiment - mean_sent) / std_sent

        # Contrarian: fade extremes
        raw_score = -z_score / 3  # Normalize to roughly -1 to 1

        # Only act on extremes
        if abs(z_score) > 2:
            if z_score > 2:  # Extremely bullish -> fade
                direction = SignalDirection.SELL
            else:  # Extremely bearish -> fade
                direction = SignalDirection.BUY
            confidence = min(1.0, (abs(z_score) - 2) / 2)
        else:
            direction = SignalDirection.NEUTRAL
            confidence = 0.0

        return StrategySignal(
            direction=direction,
            confidence=confidence,
            raw_score=raw_score,
            strategy_name=self.name,
            metadata={"z_score": z_score, "mean_sentiment": mean_sent},
        )


class RegimeFollowerStrategy(BaseStrategy):
    """
    Adapt to detected market regimes.

    TRENDING: Follow momentum
    RANGING: Mean reversion
    VOLATILE: Reduce exposure
    """

    def __init__(self):
        super().__init__("regime_follower")

    def generate_signal(
        self,
        sentiment_data: Dict[str, Any],
        market_data: Dict[str, Any],
        regime: str,
    ) -> StrategySignal:
        net_sentiment = sentiment_data.get("net_sentiment", 0)
        volatility = market_data.get("volatility_20d", 0.01)
        trend_strength = market_data.get("trend_strength", 0)

        # Adjust signal based on regime
        if regime == "TRENDING_UP":
            # Amplify bullish signals
            raw_score = net_sentiment * 1.5 if net_sentiment > 0 else net_sentiment * 0.5
        elif regime == "TRENDING_DOWN":
            # Amplify bearish signals
            raw_score = net_sentiment * 1.5 if net_sentiment < 0 else net_sentiment * 0.5
        elif regime == "HIGH_VOLATILITY":
            # Dampen all signals
            raw_score = net_sentiment * 0.3
        else:
            raw_score = net_sentiment

        # Scale confidence by regime clarity
        regime_confidence = {
            "TRENDING_UP": 0.8,
            "TRENDING_DOWN": 0.8,
            "RANGING": 0.6,
            "HIGH_VOLATILITY": 0.3,
            "NORMAL": 0.5,
        }.get(regime, 0.5)

        # Determine direction
        if raw_score > 0.3:
            direction = SignalDirection.BUY
        elif raw_score < -0.3:
            direction = SignalDirection.SELL
        else:
            direction = SignalDirection.NEUTRAL

        confidence = min(1.0, abs(raw_score)) * regime_confidence

        return StrategySignal(
            direction=direction,
            confidence=confidence,
            raw_score=raw_score,
            strategy_name=self.name,
            metadata={"regime": regime, "regime_confidence": regime_confidence},
        )


class MeanReversionStrategy(BaseStrategy):
    """
    Trade when sentiment deviates from its moving average.

    BUY when: sentiment below MA (expect reversion up)
    SELL when: sentiment above MA (expect reversion down)
    """

    def __init__(self, ma_period: int = 10, deviation_threshold: float = 0.2):
        super().__init__("mean_reversion")
        self.ma_period = ma_period
        self.deviation_threshold = deviation_threshold
        self.sentiment_history = deque(maxlen=50)

    def generate_signal(
        self,
        sentiment_data: Dict[str, Any],
        market_data: Dict[str, Any],
        regime: str,
    ) -> StrategySignal:
        net_sentiment = sentiment_data.get("net_sentiment", 0)
        self.sentiment_history.append(net_sentiment)

        if len(self.sentiment_history) < self.ma_period:
            return StrategySignal(
                direction=SignalDirection.NEUTRAL,
                confidence=0.0,
                raw_score=0.0,
                strategy_name=self.name,
            )

        # Calculate MA and deviation
        ma = np.mean(list(self.sentiment_history)[-self.ma_period:])
        deviation = net_sentiment - ma

        # Mean reversion: bet on return to mean
        raw_score = -deviation  # Negative because we expect reversion

        if abs(deviation) > self.deviation_threshold:
            if deviation > self.deviation_threshold:
                direction = SignalDirection.SELL  # Too high, expect down
            else:
                direction = SignalDirection.BUY   # Too low, expect up
            confidence = min(1.0, abs(deviation) / self.deviation_threshold / 2)
        else:
            direction = SignalDirection.NEUTRAL
            confidence = 0.0

        return StrategySignal(
            direction=direction,
            confidence=confidence,
            raw_score=raw_score,
            strategy_name=self.name,
            metadata={"ma": ma, "deviation": deviation},
        )


class LeadLagStrategy(BaseStrategy):
    """
    Use sector leaders (MAG7) to predict SPY direction.

    If tech leaders showing strength/weakness, SPY may follow.
    """

    def __init__(self, lead_time: int = 1):
        super().__init__("lead_lag")
        self.lead_time = lead_time
        self.mag7_history = deque(maxlen=10)

    def generate_signal(
        self,
        sentiment_data: Dict[str, Any],
        market_data: Dict[str, Any],
        regime: str,
    ) -> StrategySignal:
        # Get MAG7 sentiment if available
        mag7_sentiment = sentiment_data.get("mag7_sentiment", sentiment_data.get("net_sentiment", 0))
        spy_sentiment = sentiment_data.get("net_sentiment", 0)

        self.mag7_history.append(mag7_sentiment)

        if len(self.mag7_history) < 3:
            return StrategySignal(
                direction=SignalDirection.NEUTRAL,
                confidence=0.0,
                raw_score=0.0,
                strategy_name=self.name,
            )

        # Lead-lag: MAG7 sentiment change predicts SPY
        mag7_change = mag7_sentiment - list(self.mag7_history)[-2]

        # If MAG7 moving strongly, SPY may follow
        raw_score = mag7_change * 2  # Amplify the lead signal

        # Weight by divergence: bigger signal if MAG7 and SPY diverging
        divergence = mag7_sentiment - spy_sentiment
        confidence_boost = min(0.3, abs(divergence) * 0.5)

        if raw_score > 0.2:
            direction = SignalDirection.BUY
        elif raw_score < -0.2:
            direction = SignalDirection.SELL
        else:
            direction = SignalDirection.NEUTRAL

        confidence = min(1.0, abs(raw_score) / 0.5 + confidence_boost)

        return StrategySignal(
            direction=direction,
            confidence=confidence,
            raw_score=raw_score,
            strategy_name=self.name,
            metadata={"mag7_sentiment": mag7_sentiment, "spy_sentiment": spy_sentiment, "divergence": divergence},
        )
