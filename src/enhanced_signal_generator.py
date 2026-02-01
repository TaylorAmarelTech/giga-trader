"""
GIGA TRADER - Enhanced Signal Generator with Robust Anti-Overfitting
=====================================================================
Multi-strategy ensemble with validation against synthetic SPY universes.

KEY DESIGN PRINCIPLES (Anti-Overfitting):
1. Strategies must pass "what SPY could have been" robustness test
2. Weights determined by cross-validated performance, NOT recent returns
3. Strategy diversity is valued over individual performance
4. Drift detection triggers conservative mode, not aggressive adaptation
5. Confidence calibration based on long-term accuracy, not short-term

STRATEGIES:
- Momentum: Follow strong trends with sentiment confirmation
- Contrarian: Fade extreme sentiment (mean reversion)
- Regime Follower: Adapt to detected market regimes
- Mean Reversion: Trade when sentiment deviates from moving average
- Lead/Lag: Use sector leaders to predict SPY direction
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
from enum import Enum
from collections import deque
import logging
from abc import ABC, abstractmethod

logger = logging.getLogger("GigaTrader.EnhancedSignal")


# =============================================================================
# DATA STRUCTURES
# =============================================================================

class SignalDirection(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    NEUTRAL = "NEUTRAL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class StrategySignal:
    """Signal from a single strategy."""
    direction: SignalDirection
    confidence: float  # 0.0 to 1.0
    raw_score: float   # -1.0 to 1.0
    strategy_name: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)


@dataclass
class EnsembleSignal:
    """Combined signal from all strategies."""
    direction: SignalDirection
    confidence: float
    calibrated_confidence: float  # Adjusted for historical accuracy
    raw_score: float
    timestamp: datetime
    contributing_strategies: List[str]
    strategy_weights: Dict[str, float]
    drift_detected: bool = False
    drift_severity: float = 0.0
    regime: str = "NORMAL"
    metadata: Dict = field(default_factory=dict)


@dataclass
class StrategyPerformance:
    """Track strategy performance for robust weight calculation."""
    strategy_name: str
    total_signals: int = 0
    correct_signals: int = 0

    # Performance on real SPY
    real_spy_accuracy: float = 0.5
    real_spy_sharpe: float = 0.0
    real_spy_returns: List[float] = field(default_factory=list)

    # Performance on synthetic universes (critical for robustness)
    synthetic_accuracy: float = 0.5
    synthetic_sharpe: float = 0.0
    synthetic_variance: float = 0.0  # Lower = more robust
    universe_accuracies: Dict[str, float] = field(default_factory=dict)

    # Anti-overfitting metrics
    robustness_score: float = 0.5  # Combined real + synthetic
    overfitting_penalty: float = 0.0  # High if real >> synthetic

    # Weight in ensemble (determined by CV, not recent performance)
    cv_weight: float = 0.2  # Default equal weight for 5 strategies

    def update_robustness(self):
        """Calculate robustness score penalizing overfitting."""
        if self.synthetic_accuracy > 0:
            # Overfitting penalty: how much better on real vs synthetic
            overfit_gap = max(0, self.real_spy_accuracy - self.synthetic_accuracy)
            self.overfitting_penalty = overfit_gap * 2  # 2x penalty

            # Robustness = weighted average minus penalty minus variance
            self.robustness_score = (
                0.4 * self.real_spy_accuracy +
                0.6 * self.synthetic_accuracy -  # Weight synthetic more
                0.3 * self.overfitting_penalty -
                0.2 * min(1.0, self.synthetic_variance * 5)  # Variance penalty
            )
            self.robustness_score = max(0.0, min(1.0, self.robustness_score))


# =============================================================================
# DRIFT DETECTION (ADWIN Algorithm)
# =============================================================================

class ADWINDriftDetector:
    """
    ADWIN (ADaptive WINdowing) drift detection.

    Detects when the distribution of prediction errors changes significantly.
    When drift is detected, we DON'T aggressively adapt - we go conservative.
    """

    def __init__(self, delta: float = 0.002, min_window: int = 30):
        self.delta = delta
        self.min_window = min_window
        self.window = deque(maxlen=1000)
        self.n_detections = 0
        self.last_detection_time: Optional[datetime] = None

    def add_element(self, value: float) -> bool:
        """
        Add element and check for drift.

        Args:
            value: Prediction error or similar metric

        Returns:
            True if drift detected
        """
        self.window.append(value)

        if len(self.window) < self.min_window * 2:
            return False

        # Check for drift using statistical test
        drift_detected = self._check_drift()

        if drift_detected:
            self.n_detections += 1
            self.last_detection_time = datetime.now()
            # Shrink window to forget old data
            while len(self.window) > self.min_window:
                self.window.popleft()

        return drift_detected

    def _check_drift(self) -> bool:
        """Check for statistically significant drift."""
        window_list = list(self.window)
        n = len(window_list)

        # Test multiple split points
        for split in range(self.min_window, n - self.min_window):
            left = window_list[:split]
            right = window_list[split:]

            # Calculate means
            mean_left = np.mean(left)
            mean_right = np.mean(right)

            # Calculate bound for difference
            n_left, n_right = len(left), len(right)
            m = 1.0 / (1.0/n_left + 1.0/n_right)
            eps_cut = np.sqrt(
                (1.0 / (2.0 * m)) * np.log(4.0 / self.delta)
            )

            if abs(mean_left - mean_right) > eps_cut:
                return True

        return False

    def get_severity(self) -> float:
        """Get drift severity (0 = no drift, 1 = severe)."""
        if not self.last_detection_time:
            return 0.0

        # Severity based on recency and frequency
        time_since = (datetime.now() - self.last_detection_time).total_seconds()
        recency_factor = max(0, 1.0 - time_since / 3600)  # Decay over 1 hour
        frequency_factor = min(1.0, self.n_detections / 10)

        return 0.5 * recency_factor + 0.5 * frequency_factor


# =============================================================================
# CONFIDENCE CALIBRATION
# =============================================================================

class ConfidenceCalibrator:
    """
    Calibrate raw confidence to match historical accuracy.

    If a strategy says 70% confidence, it should be right ~70% of the time.
    Uses isotonic regression on historical data.
    """

    def __init__(self, n_bins: int = 10, min_samples: int = 20):
        self.n_bins = n_bins
        self.min_samples = min_samples
        self.calibration_map: Dict[int, float] = {}
        self.history: List[Tuple[float, bool]] = []  # (confidence, was_correct)

    def add_outcome(self, confidence: float, was_correct: bool):
        """Record an outcome for calibration."""
        self.history.append((confidence, was_correct))

        # Recalibrate periodically
        if len(self.history) % 50 == 0:
            self._recalibrate()

    def _recalibrate(self):
        """Recalibrate based on historical data."""
        if len(self.history) < self.min_samples:
            return

        # Bin by confidence
        bins = {}
        for conf, correct in self.history[-500:]:  # Last 500 samples
            bin_idx = min(self.n_bins - 1, int(conf * self.n_bins))
            if bin_idx not in bins:
                bins[bin_idx] = []
            bins[bin_idx].append(1.0 if correct else 0.0)

        # Calculate actual accuracy per bin
        for bin_idx, outcomes in bins.items():
            if len(outcomes) >= 5:  # Min samples per bin
                self.calibration_map[bin_idx] = np.mean(outcomes)

    def calibrate(self, raw_confidence: float) -> float:
        """
        Convert raw confidence to calibrated confidence.

        Args:
            raw_confidence: Strategy's raw confidence (0-1)

        Returns:
            Calibrated confidence based on historical accuracy
        """
        if not self.calibration_map:
            return raw_confidence

        bin_idx = min(self.n_bins - 1, int(raw_confidence * self.n_bins))

        if bin_idx in self.calibration_map:
            return self.calibration_map[bin_idx]

        # Interpolate from nearby bins
        lower = max(k for k in self.calibration_map.keys() if k <= bin_idx) if any(k <= bin_idx for k in self.calibration_map.keys()) else None
        upper = min(k for k in self.calibration_map.keys() if k >= bin_idx) if any(k >= bin_idx for k in self.calibration_map.keys()) else None

        if lower is not None and upper is not None and lower != upper:
            # Linear interpolation
            ratio = (bin_idx - lower) / (upper - lower)
            return (1 - ratio) * self.calibration_map[lower] + ratio * self.calibration_map[upper]
        elif lower is not None:
            return self.calibration_map[lower]
        elif upper is not None:
            return self.calibration_map[upper]

        return raw_confidence


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


# =============================================================================
# ROBUST ENSEMBLE
# =============================================================================

class RobustEnsembleSignalGenerator:
    """
    Robust ensemble signal generator with anti-overfitting measures.

    Key Features:
    1. Strategy weights from cross-validation, not recent performance
    2. Validates all strategies against synthetic SPY universes
    3. Penalizes strategies that overfit to real SPY
    4. Drift detection triggers conservative mode
    5. Confidence calibration for realistic probabilities
    """

    def __init__(self):
        # Initialize strategies
        self.strategies: Dict[str, BaseStrategy] = {
            "momentum": MomentumStrategy(),
            "contrarian": ContrarianStrategy(),
            "regime_follower": RegimeFollowerStrategy(),
            "mean_reversion": MeanReversionStrategy(),
            "lead_lag": LeadLagStrategy(),
        }

        # Default equal weights (will be updated by CV)
        n_strategies = len(self.strategies)
        self.strategy_weights = {name: 1.0 / n_strategies for name in self.strategies}

        # Anti-overfitting components
        self.drift_detector = ADWINDriftDetector()
        self.calibrator = ConfidenceCalibrator()

        # Performance tracking
        self.signal_history: List[EnsembleSignal] = []
        self.outcome_history: List[Tuple[datetime, bool]] = []

        # Dynamic thresholds (scale with volatility)
        self.base_buy_threshold = 0.3
        self.base_sell_threshold = -0.3

        # State
        self.current_regime = "NORMAL"
        self.is_conservative_mode = False

        logger.info("RobustEnsembleSignalGenerator initialized with 5 strategies")

    def detect_regime(self, market_data: Dict[str, Any]) -> str:
        """Detect current market regime."""
        volatility = market_data.get("volatility_20d", 0.01)
        trend_strength = market_data.get("trend_strength", 0)
        momentum = market_data.get("momentum_5d", 0)

        # High volatility regime
        if volatility > 0.025:  # ~25% annualized
            return "HIGH_VOLATILITY"

        # Trending regimes
        if abs(trend_strength) > 0.5:
            if trend_strength > 0 and momentum > 0.01:
                return "TRENDING_UP"
            elif trend_strength < 0 and momentum < -0.01:
                return "TRENDING_DOWN"

        # Ranging
        if abs(momentum) < 0.005:
            return "RANGING"

        return "NORMAL"

    def generate_signal(
        self,
        sentiment_data: Dict[str, Any],
        market_data: Dict[str, Any],
    ) -> EnsembleSignal:
        """
        Generate ensemble signal from all strategies.

        Args:
            sentiment_data: Dict with net_sentiment, mag7_sentiment, etc.
            market_data: Dict with volatility, momentum, etc.

        Returns:
            EnsembleSignal with combined prediction
        """
        # Detect regime
        self.current_regime = self.detect_regime(market_data)

        # Collect signals from all strategies
        strategy_signals: Dict[str, StrategySignal] = {}
        for name, strategy in self.strategies.items():
            signal = strategy.generate_signal(sentiment_data, market_data, self.current_regime)
            strategy_signals[name] = signal

        # Weighted combination
        weighted_score = 0.0
        weighted_confidence = 0.0
        contributing = []

        for name, signal in strategy_signals.items():
            weight = self.strategy_weights[name]

            # Only count non-neutral signals
            if signal.direction != SignalDirection.NEUTRAL:
                weighted_score += weight * signal.raw_score
                weighted_confidence += weight * signal.confidence
                contributing.append(name)

        # Dynamic thresholds based on volatility
        volatility = market_data.get("volatility_20d", 0.01)
        threshold_scale = 1.0 + max(0, (volatility - 0.015) * 10)  # Widen thresholds in high vol

        buy_threshold = self.base_buy_threshold * threshold_scale
        sell_threshold = self.base_sell_threshold * threshold_scale

        # Determine direction
        if weighted_score > buy_threshold * 2:
            direction = SignalDirection.STRONG_BUY
        elif weighted_score > buy_threshold:
            direction = SignalDirection.BUY
        elif weighted_score < sell_threshold * 2:
            direction = SignalDirection.STRONG_SELL
        elif weighted_score < sell_threshold:
            direction = SignalDirection.SELL
        else:
            direction = SignalDirection.NEUTRAL

        # Check for drift
        drift_detected = self.drift_detector.add_element(weighted_score)
        drift_severity = self.drift_detector.get_severity()

        # Conservative mode during drift
        if drift_detected or drift_severity > 0.5:
            self.is_conservative_mode = True
            weighted_confidence *= 0.5  # Halve confidence during drift
            logger.warning(f"Drift detected! Entering conservative mode (severity={drift_severity:.2f})")
        elif drift_severity < 0.2:
            self.is_conservative_mode = False

        # Calibrate confidence
        calibrated_confidence = self.calibrator.calibrate(weighted_confidence)

        # Build ensemble signal
        signal = EnsembleSignal(
            direction=direction,
            confidence=weighted_confidence,
            calibrated_confidence=calibrated_confidence,
            raw_score=weighted_score,
            timestamp=datetime.now(),
            contributing_strategies=contributing,
            strategy_weights=self.strategy_weights.copy(),
            drift_detected=drift_detected,
            drift_severity=drift_severity,
            regime=self.current_regime,
            metadata={
                "threshold_scale": threshold_scale,
                "conservative_mode": self.is_conservative_mode,
                "strategy_signals": {n: s.raw_score for n, s in strategy_signals.items()},
            },
        )

        self.signal_history.append(signal)

        return signal

    def record_outcome(self, was_correct: bool):
        """
        Record outcome of last signal for calibration and weight updates.

        IMPORTANT: We do NOT chase recent performance for weights.
        Weights are updated only via periodic cross-validation.
        """
        self.outcome_history.append((datetime.now(), was_correct))

        # Update calibrator
        if self.signal_history:
            last_signal = self.signal_history[-1]
            self.calibrator.add_outcome(last_signal.confidence, was_correct)

    def validate_on_synthetic_universes(
        self,
        synthetic_data: Dict[str, pd.DataFrame],
        real_data: pd.DataFrame,
    ) -> Dict[str, StrategyPerformance]:
        """
        Validate all strategies on synthetic SPY universes.

        This is the KEY anti-overfitting measure. A strategy must perform
        well not just on real SPY but on "what SPY could have been".

        Args:
            synthetic_data: Dict of universe_name -> DataFrame
            real_data: Real SPY DataFrame

        Returns:
            Dict of strategy_name -> StrategyPerformance
        """
        logger.info(f"Validating strategies on {len(synthetic_data)} synthetic universes...")

        performances = {}

        for name, strategy in self.strategies.items():
            perf = StrategyPerformance(strategy_name=name)

            # Validate on real data
            real_accuracy = self._backtest_strategy(strategy, real_data)
            perf.real_spy_accuracy = real_accuracy

            # Validate on each synthetic universe
            universe_accuracies = {}
            for universe_name, universe_data in synthetic_data.items():
                acc = self._backtest_strategy(strategy, universe_data)
                universe_accuracies[universe_name] = acc

            perf.universe_accuracies = universe_accuracies
            perf.synthetic_accuracy = np.mean(list(universe_accuracies.values()))
            perf.synthetic_variance = np.var(list(universe_accuracies.values()))

            # Calculate robustness score
            perf.update_robustness()

            performances[name] = perf

            logger.info(f"  {name}: real={perf.real_spy_accuracy:.3f}, "
                       f"synthetic={perf.synthetic_accuracy:.3f}, "
                       f"robustness={perf.robustness_score:.3f}")

        return performances

    def update_weights_from_cv(
        self,
        performances: Dict[str, StrategyPerformance],
    ):
        """
        Update strategy weights based on cross-validated performance.

        CRITICAL: We use robustness_score (which penalizes overfitting),
        NOT recent returns or real-SPY-only accuracy.
        """
        # Calculate weights from robustness scores
        total_robustness = sum(p.robustness_score for p in performances.values())

        if total_robustness > 0:
            for name, perf in performances.items():
                # Weight by robustness, not raw accuracy
                raw_weight = perf.robustness_score / total_robustness

                # Apply smoothing: blend with equal weight to prevent dominance
                equal_weight = 1.0 / len(performances)
                perf.cv_weight = 0.7 * raw_weight + 0.3 * equal_weight

                self.strategy_weights[name] = perf.cv_weight

        # Normalize
        total = sum(self.strategy_weights.values())
        self.strategy_weights = {k: v / total for k, v in self.strategy_weights.items()}

        logger.info(f"Updated weights from CV: {self.strategy_weights}")

    def _backtest_strategy(
        self,
        strategy: BaseStrategy,
        data: pd.DataFrame,
    ) -> float:
        """Backtest a strategy on given data."""
        # Simplified backtest - count directional accuracy
        correct = 0
        total = 0

        for i in range(len(data) - 1):
            # Prepare data
            sentiment_data = {"net_sentiment": data.iloc[i].get("sentiment", 0)}
            market_data = {
                "volatility_20d": data.iloc[i].get("volatility", 0.01),
                "momentum_5d": data.iloc[i].get("momentum", 0),
            }

            # Generate signal
            signal = strategy.generate_signal(sentiment_data, market_data, "NORMAL")

            # Check next day return
            if "day_return" in data.columns:
                next_return = data.iloc[i + 1]["day_return"]
            else:
                next_return = 0

            # Score
            if signal.direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY]:
                if next_return > 0:
                    correct += 1
                total += 1
            elif signal.direction in [SignalDirection.SELL, SignalDirection.STRONG_SELL]:
                if next_return < 0:
                    correct += 1
                total += 1

        return correct / total if total > 0 else 0.5

    def get_comprehensive_report(self) -> str:
        """Generate comprehensive status report."""
        lines = [
            "=" * 60,
            "ROBUST ENSEMBLE SIGNAL GENERATOR - STATUS REPORT",
            "=" * 60,
            "",
            f"Current Regime: {self.current_regime}",
            f"Conservative Mode: {self.is_conservative_mode}",
            f"Drift Severity: {self.drift_detector.get_severity():.3f}",
            "",
            "STRATEGY WEIGHTS (CV-determined):",
            "-" * 40,
        ]

        for name, weight in sorted(self.strategy_weights.items(), key=lambda x: -x[1]):
            perf = self.strategies[name].performance
            lines.append(f"  {name}: weight={weight:.1%}, robustness={perf.robustness_score:.3f}")

        lines.extend([
            "",
            "RECENT SIGNALS:",
            "-" * 40,
        ])

        for signal in self.signal_history[-5:]:
            lines.append(
                f"  {signal.timestamp.strftime('%H:%M:%S')}: {signal.direction.value} "
                f"(conf={signal.calibrated_confidence:.2f}, score={signal.raw_score:.3f})"
            )

        lines.extend([
            "",
            "CALIBRATION STATUS:",
            f"  Samples: {len(self.calibrator.history)}",
            f"  Bins calibrated: {len(self.calibrator.calibration_map)}",
            "",
            "DRIFT DETECTION:",
            f"  Total detections: {self.drift_detector.n_detections}",
            f"  Last detection: {self.drift_detector.last_detection_time or 'Never'}",
            "=" * 60,
        ])

        return "\n".join(lines)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

_enhanced_generator_instance: Optional[RobustEnsembleSignalGenerator] = None


def get_enhanced_signal_generator() -> RobustEnsembleSignalGenerator:
    """Get singleton instance of enhanced signal generator."""
    global _enhanced_generator_instance
    if _enhanced_generator_instance is None:
        _enhanced_generator_instance = RobustEnsembleSignalGenerator()
    return _enhanced_generator_instance


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("ROBUST ENSEMBLE SIGNAL GENERATOR - DEMO")
    print("=" * 60)

    gen = get_enhanced_signal_generator()

    # Generate some test signals
    for i in range(10):
        sentiment = {
            "net_sentiment": np.random.randn() * 0.3,
            "mag7_sentiment": np.random.randn() * 0.4,
        }
        market = {
            "volatility_20d": 0.01 + np.random.rand() * 0.02,
            "momentum_5d": np.random.randn() * 0.02,
            "trend_strength": np.random.randn() * 0.5,
        }

        signal = gen.generate_signal(sentiment, market)
        print(f"Signal {i+1}: {signal.direction.value} "
              f"(conf={signal.calibrated_confidence:.2f}, regime={signal.regime})")

        # Record random outcome
        gen.record_outcome(np.random.rand() > 0.45)

    print("\n" + gen.get_comprehensive_report())
