"""
GIGA TRADER - Robust Ensemble Signal Generator
=================================================
Multi-strategy ensemble with validation against synthetic SPY universes.

KEY DESIGN PRINCIPLES (Anti-Overfitting):
1. Strategies must pass "what SPY could have been" robustness test
2. Weights determined by cross-validated performance, NOT recent returns
3. Strategy diversity is valued over individual performance
4. Drift detection triggers conservative mode, not aggressive adaptation
5. Confidence calibration based on long-term accuracy, not short-term

Contains:
- RobustEnsembleSignalGenerator class
- get_enhanced_signal_generator() factory function
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
import logging

from src.phase_15_strategy.signal_data import (
    SignalDirection,
    StrategySignal,
    EnsembleSignal,
    StrategyPerformance,
)
from src.phase_15_strategy.signal_detectors import (
    ADWINDriftDetector,
    ConfidenceCalibrator,
)
from src.phase_15_strategy.trading_strategies import (
    BaseStrategy,
    MomentumStrategy,
    ContrarianStrategy,
    RegimeFollowerStrategy,
    MeanReversionStrategy,
    LeadLagStrategy,
)

# Temporal cascade signal generator
try:
    from src.phase_12_model_training.temporal_cascade_trainer import TemporalCascadeSignalGenerator
    TEMPORAL_CASCADE_AVAILABLE = True
except ImportError:
    TEMPORAL_CASCADE_AVAILABLE = False

logger = logging.getLogger("GigaTrader.EnhancedSignal")


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

        # Temporal cascade integration (optional)
        self.temporal_cascade = None

        logger.info("RobustEnsembleSignalGenerator initialized with 5 strategies")

    def set_temporal_cascade(self, cascade: Any):
        """Set temporal cascade signal source for enhanced predictions."""
        self.temporal_cascade = cascade
        logger.info("Temporal cascade integrated into ensemble signal generator")

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

        # Incorporate temporal cascade if available
        if self.temporal_cascade is not None:
            try:
                from datetime import datetime as _dt
                now = _dt.now()
                minutes_since_open = (now.hour - 9) * 60 + (now.minute - 30)
                if 0 <= minutes_since_open <= 390:
                    cascade_result = self.temporal_cascade.generate_realtime_signal(
                        historical_features=None,
                        df_1min_today=market_data,
                        minutes_since_open=minutes_since_open,
                    )
                    if cascade_result is not None:
                        cascade_score = (cascade_result.swing_direction - 0.5) * 2  # Scale to [-1, 1]
                        cascade_weight = 0.15  # 15% weight for temporal cascade
                        weighted_score = weighted_score * (1 - cascade_weight) + cascade_score * cascade_weight
                        contributing.append("temporal_cascade")
            except Exception as e:
                logger.debug(f"Temporal cascade unavailable: {e}")

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
