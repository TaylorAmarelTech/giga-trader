"""
GIGA TRADER - Rule-Based Trading Gates
========================================
Trading gates evaluate conditions from data sources that are too sparse
for ML training (weekly/quarterly) but can still influence trading
decisions through if-then logic at extremes.

Gates operate BETWEEN signal generation and order execution:
  Signal Generator → Trading Gates → Risk Manager → Order Manager

Each gate independently produces a decision:
  - PASS: No restriction (normal conditions)
  - BOOST: Increase confidence (contrarian favorable)
  - REDUCE: Decrease confidence or position size
  - BLOCK: Do not enter new positions (extreme readings)

Gate multipliers compose multiplicatively:
  final_confidence = signal.confidence * gate1.conf_mult * gate2.conf_mult * ...

Gates have NO effect during normal conditions — they only activate at extremes.
"""

import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("TRADING_GATES")


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class GateDecision:
    """Result of evaluating a single gate."""
    gate_name: str
    action: str  # "PASS", "BOOST", "REDUCE", "BLOCK"
    confidence_multiplier: float  # 1.0 = no change
    position_size_multiplier: float  # 1.0 = no change
    reason: str
    data_value: Optional[float] = None
    data_timestamp: Optional[datetime] = None
    is_stale: bool = False


@dataclass
class GateResult:
    """Aggregate result of all gates."""
    timestamp: datetime
    is_blocked: bool
    blocking_gates: List[str]
    confidence_multiplier: float
    position_size_multiplier: float
    individual_decisions: List[GateDecision]
    original_signal_type: str
    gated_signal_type: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging and dashboard."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "is_blocked": self.is_blocked,
            "blocking_gates": self.blocking_gates,
            "confidence_multiplier": round(self.confidence_multiplier, 4),
            "position_size_multiplier": round(self.position_size_multiplier, 4),
            "original_signal_type": self.original_signal_type,
            "gated_signal_type": self.gated_signal_type,
            "n_gates_evaluated": len(self.individual_decisions),
            "decisions": [
                {
                    "gate": d.gate_name,
                    "action": d.action,
                    "conf_mult": round(d.confidence_multiplier, 4),
                    "pos_mult": round(d.position_size_multiplier, 4),
                    "reason": d.reason,
                    "value": d.data_value,
                    "stale": d.is_stale,
                }
                for d in self.individual_decisions
            ],
        }


@dataclass
class TradingGatesConfig:
    """Configuration for all trading gates."""
    # Master switch
    gates_enabled: bool = True

    # CNN Fear & Greed gate
    fear_greed_enabled: bool = True
    fear_greed_extreme_fear: int = 20
    fear_greed_extreme_greed: int = 80
    fear_greed_confidence_boost: float = 1.15
    fear_greed_confidence_reduce: float = 0.80
    fear_greed_max_staleness_hours: int = 48

    # CFTC COT gate (Wave 37d) — contrarian positioning
    cot_enabled: bool = True
    cot_extreme_zscore: float = 2.0
    cot_position_boost: float = 1.20
    cot_position_reduce: float = 0.70
    cot_max_staleness_hours: int = 168  # 7 days

    # AAII Sentiment gate (Wave 37d) — contrarian retail sentiment
    aaii_enabled: bool = True
    aaii_extreme_spread: float = 30.0
    aaii_confidence_boost: float = 1.10
    aaii_confidence_reduce: float = 0.85
    aaii_max_staleness_hours: int = 168  # 7 days

    # GEX gate (Wave 37c) — gamma exposure regime
    gex_enabled: bool = True
    gex_trend_confidence_boost: float = 1.10
    gex_trend_confidence_reduce: float = 0.85
    gex_max_staleness_hours: int = 24

    # Insider Sentiment gate (Finnhub, auto-enabled if FINNHUB_API_KEY set)
    insider_enabled: bool = True
    insider_caution_percentile: float = 0.90  # Top/bottom 10%
    insider_position_reduce: float = 0.50  # 50% position on caution
    insider_max_staleness_hours: int = 168  # 7 days

    # SEC EDGAR gate (Wave 37d)
    edgar_enabled: bool = False  # Off by default (quarterly data, too slow)

    # Wave J: Macro calendar gate (FOMC/NFP/CPI event risk)
    macro_calendar_enabled: bool = True
    macro_calendar_max_staleness_hours: int = 24

    # Wave J: Volatility regime gate (VIX-based)
    vol_regime_gate_enabled: bool = True
    vol_regime_block_vix: float = 35.0
    vol_regime_reduce_vix: float = 28.0

    # Behavior when data is stale
    stale_data_action: str = "PASS"  # "PASS" = ignore gate, "BLOCK" = block trade

    # Multiplier clamps (safety bounds)
    min_confidence_multiplier: float = 0.3
    max_confidence_multiplier: float = 1.5
    min_position_multiplier: float = 0.2
    max_position_multiplier: float = 1.5


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PROVIDERS
# ═══════════════════════════════════════════════════════════════════════════════


class GateDataProvider(ABC):
    """Abstract base class for gate data providers."""

    @abstractmethod
    def fetch(self) -> Optional[Dict[str, Any]]:
        """Fetch latest data. Returns None on failure."""
        ...

    @abstractmethod
    def get_latest(self) -> Optional[Dict[str, Any]]:
        """Get most recently cached data without fetching."""
        ...

    @abstractmethod
    def get_data_age_hours(self) -> float:
        """Get age of cached data in hours."""
        ...


class MockGateDataProvider(GateDataProvider):
    """Mock provider for testing. Accepts manually set values."""

    def __init__(self, data: Optional[Dict[str, Any]] = None):
        self._data = data
        self._last_fetch = datetime.now()

    def set_data(self, data: Dict[str, Any]):
        self._data = data
        self._last_fetch = datetime.now()

    def fetch(self) -> Optional[Dict[str, Any]]:
        return self._data

    def get_latest(self) -> Optional[Dict[str, Any]]:
        return self._data

    def get_data_age_hours(self) -> float:
        return (datetime.now() - self._last_fetch).total_seconds() / 3600


class FearGreedProvider(GateDataProvider):
    """Fetches CNN Fear & Greed Index for gate evaluation."""

    def __init__(self):
        self._data: Optional[Dict[str, Any]] = None
        self._last_fetch: Optional[datetime] = None

    def fetch(self) -> Optional[Dict[str, Any]]:
        try:
            import requests
            url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
            headers = {"User-Agent": "GigaTrader/1.0"}
            resp = requests.get(url, headers=headers, timeout=15)

            if resp.status_code != 200:
                return self._data

            data = resp.json()
            fg = data.get("fear_and_greed", {})
            score = fg.get("score", None)

            if score is not None:
                self._data = {
                    "score": float(score),
                    "rating": fg.get("rating", ""),
                    "timestamp": datetime.now(),
                }
                self._last_fetch = datetime.now()
                return self._data

        except Exception as e:
            logger.debug(f"FearGreedProvider fetch failed: {e}")

        return self._data

    def get_latest(self) -> Optional[Dict[str, Any]]:
        return self._data

    def get_data_age_hours(self) -> float:
        if self._last_fetch is None:
            return 999.0
        return (datetime.now() - self._last_fetch).total_seconds() / 3600


# ═══════════════════════════════════════════════════════════════════════════════
# TRADING GATES
# ═══════════════════════════════════════════════════════════════════════════════


class TradingGates:
    """
    Rule-based trading gate system for sparse data sources.

    Usage:
        gates = TradingGates(config=TradingGatesConfig())
        gates.update_data()
        result = gates.evaluate(signal_type="BUY")
        if result.is_blocked:
            logger.info(f"Trade blocked: {result.blocking_gates}")
        else:
            confidence *= result.confidence_multiplier
            position_size *= result.position_size_multiplier
    """

    def __init__(
        self,
        config: Optional[TradingGatesConfig] = None,
        data_providers: Optional[Dict[str, GateDataProvider]] = None,
    ):
        self.config = config or TradingGatesConfig()
        self._providers: Dict[str, GateDataProvider] = data_providers or {}
        self._lock = threading.Lock()

        # Auto-register default providers for enabled gates (if no custom providers given)
        if not self._providers:
            self._auto_register_providers()

    def _auto_register_providers(self):
        """Auto-register data providers for all enabled gates.

        Each provider is imported lazily and wrapped in try/except so a
        missing dependency or API key only disables that one gate.
        """
        import os

        if self.config.fear_greed_enabled:
            self._providers["fear_greed"] = FearGreedProvider()

        if self.config.cot_enabled:
            try:
                from src.phase_19_paper_trading.cot_gate import COTDataProvider
                self._providers["cot"] = COTDataProvider()
            except Exception as e:
                logger.warning(f"COT gate provider unavailable: {e}")
                self.config.cot_enabled = False

        if self.config.aaii_enabled:
            try:
                from src.phase_19_paper_trading.aaii_gate import AAIIDataProvider
                self._providers["aaii"] = AAIIDataProvider()
            except Exception as e:
                logger.warning(f"AAII gate provider unavailable: {e}")
                self.config.aaii_enabled = False

        if self.config.gex_enabled:
            try:
                from src.phase_19_paper_trading.cot_gate import COTDataProvider as _
                # GEX doesn't have its own provider — it uses the GEX features data
                # evaluated directly in _evaluate_gex. No provider needed.
            except Exception:
                pass

        if self.config.insider_enabled:
            finnhub_key = os.environ.get("FINNHUB_API_KEY", "")
            if finnhub_key:
                try:
                    from src.phase_19_paper_trading.insider_gate import InsiderSentimentProvider
                    self._providers["insider"] = InsiderSentimentProvider()
                except Exception as e:
                    logger.warning(f"Insider gate provider unavailable: {e}")
                    self.config.insider_enabled = False
            else:
                logger.info("Insider gate disabled: FINNHUB_API_KEY not set")
                self.config.insider_enabled = False

        if self.config.macro_calendar_enabled:
            try:
                from src.phase_19_paper_trading.macro_calendar_gate import MacroCalendarDataProvider
                self._providers["macro_calendar"] = MacroCalendarDataProvider()
            except Exception as e:
                logger.warning(f"Macro calendar gate provider unavailable: {e}")
                self.config.macro_calendar_enabled = False

        if self.config.vol_regime_gate_enabled:
            try:
                from src.phase_19_paper_trading.vol_regime_gate import VolRegimeDataProvider
                self._providers["vol_regime"] = VolRegimeDataProvider()
            except Exception as e:
                logger.warning(f"Vol regime gate provider unavailable: {e}")
                self.config.vol_regime_gate_enabled = False

        enabled_gates = [
            name for name, enabled in [
                ("fear_greed", self.config.fear_greed_enabled),
                ("cot", self.config.cot_enabled),
                ("aaii", self.config.aaii_enabled),
                ("gex", self.config.gex_enabled),
                ("insider", self.config.insider_enabled),
                ("edgar", self.config.edgar_enabled),
                ("macro_calendar", self.config.macro_calendar_enabled),
                ("vol_regime", self.config.vol_regime_gate_enabled),
            ] if enabled
        ]
        logger.info(f"Trading gates enabled: {', '.join(enabled_gates)} ({len(enabled_gates)} active)")

    def update_data(self) -> Dict[str, bool]:
        """Fetch latest data from all enabled providers."""
        results = {}
        for name, provider in self._providers.items():
            try:
                data = provider.fetch()
                results[name] = data is not None
            except Exception as e:
                logger.warning(f"Gate data update failed for {name}: {e}")
                results[name] = False
        return results

    def refresh_data(self) -> Dict[str, bool]:
        """Refresh all gate data providers (alias for update_data).

        Called periodically by TradingBot to prevent stale data during
        long trading sessions.
        """
        results = self.update_data()
        refreshed = [k for k, v in results.items() if v]
        if refreshed:
            logger.info(f"Gate data refreshed: {', '.join(refreshed)}")
        return results

    def evaluate(self, signal_type: str = "HOLD") -> GateResult:
        """
        Evaluate all enabled gates for a given signal type.

        Args:
            signal_type: "BUY", "SELL", or "HOLD"

        Returns:
            GateResult with aggregate multipliers and block status.
        """
        if not self.config.gates_enabled or signal_type == "HOLD":
            return GateResult(
                timestamp=datetime.now(),
                is_blocked=False,
                blocking_gates=[],
                confidence_multiplier=1.0,
                position_size_multiplier=1.0,
                individual_decisions=[],
                original_signal_type=signal_type,
                gated_signal_type=signal_type,
            )

        decisions = []

        # Evaluate each enabled gate
        gate_evaluators = [
            (self.config.fear_greed_enabled, self._evaluate_fear_greed),
            (self.config.cot_enabled, self._evaluate_cot),
            (self.config.aaii_enabled, self._evaluate_aaii),
            (self.config.gex_enabled, self._evaluate_gex),
            (self.config.insider_enabled, self._evaluate_insider),
            (self.config.macro_calendar_enabled, self._evaluate_macro_calendar),
            (self.config.vol_regime_gate_enabled, self._evaluate_vol_regime),
        ]

        for enabled, evaluator in gate_evaluators:
            if enabled:
                try:
                    decision = evaluator(signal_type)
                    decisions.append(decision)
                except Exception as e:
                    logger.warning(f"Gate evaluation failed: {e}")

        # Aggregate
        conf_mult = 1.0
        pos_mult = 1.0
        blocking_gates = []

        for d in decisions:
            # Skip stale gates if configured to PASS
            if d.is_stale and self.config.stale_data_action == "PASS":
                continue
            if d.action == "BLOCK":
                blocking_gates.append(d.gate_name)
            conf_mult *= d.confidence_multiplier
            pos_mult *= d.position_size_multiplier

        # Clamp multipliers
        conf_mult = max(self.config.min_confidence_multiplier,
                       min(self.config.max_confidence_multiplier, conf_mult))
        pos_mult = max(self.config.min_position_multiplier,
                      min(self.config.max_position_multiplier, pos_mult))

        is_blocked = len(blocking_gates) > 0
        gated_type = "HOLD" if is_blocked else signal_type

        result = GateResult(
            timestamp=datetime.now(),
            is_blocked=is_blocked,
            blocking_gates=blocking_gates,
            confidence_multiplier=conf_mult,
            position_size_multiplier=pos_mult,
            individual_decisions=decisions,
            original_signal_type=signal_type,
            gated_signal_type=gated_type,
        )

        # Log
        for d in decisions:
            if d.action != "PASS":
                logger.info(
                    f"[GATES] {d.gate_name}: {d.action} "
                    f"(value={d.data_value}, conf_mult={d.confidence_multiplier:.2f}, "
                    f"reason={d.reason})"
                )

        if decisions:
            logger.info(
                f"[GATES] RESULT: conf_mult={conf_mult:.2f}, "
                f"pos_mult={pos_mult:.2f}, blocked={is_blocked}"
            )

        return result

    def _evaluate_fear_greed(self, signal_type: str) -> GateDecision:
        """Evaluate CNN Fear & Greed gate (contrarian)."""
        provider = self._providers.get("fear_greed")
        if not provider:
            return self._pass_decision("fear_greed", "Provider not configured")

        data = provider.get_latest()
        if data is None:
            return self._pass_decision("fear_greed", "No data available", is_stale=True)

        score = data.get("score", 50)
        data_age = provider.get_data_age_hours()
        is_stale = data_age > self.config.fear_greed_max_staleness_hours
        data_ts = data.get("timestamp")

        cfg = self.config

        # Normal conditions: no effect
        if cfg.fear_greed_extreme_fear <= score <= cfg.fear_greed_extreme_greed:
            return GateDecision(
                gate_name="fear_greed",
                action="PASS",
                confidence_multiplier=1.0,
                position_size_multiplier=1.0,
                reason=f"Normal range (score={score:.0f})",
                data_value=score,
                data_timestamp=data_ts,
                is_stale=is_stale,
            )

        # Extreme Fear: contrarian bullish
        if score < cfg.fear_greed_extreme_fear:
            if signal_type == "BUY":
                return GateDecision(
                    gate_name="fear_greed",
                    action="BOOST",
                    confidence_multiplier=cfg.fear_greed_confidence_boost,
                    position_size_multiplier=1.0,
                    reason=f"Extreme Fear ({score:.0f}) + BUY: contrarian bullish",
                    data_value=score,
                    data_timestamp=data_ts,
                    is_stale=is_stale,
                )
            elif signal_type == "SELL":
                return GateDecision(
                    gate_name="fear_greed",
                    action="REDUCE",
                    confidence_multiplier=cfg.fear_greed_confidence_reduce,
                    position_size_multiplier=1.0,
                    reason=f"Extreme Fear ({score:.0f}) + SELL: contrarian reduction",
                    data_value=score,
                    data_timestamp=data_ts,
                    is_stale=is_stale,
                )

        # Extreme Greed: contrarian bearish
        if score > cfg.fear_greed_extreme_greed:
            if signal_type == "BUY":
                return GateDecision(
                    gate_name="fear_greed",
                    action="REDUCE",
                    confidence_multiplier=cfg.fear_greed_confidence_reduce,
                    position_size_multiplier=1.0,
                    reason=f"Extreme Greed ({score:.0f}) + BUY: contrarian reduction",
                    data_value=score,
                    data_timestamp=data_ts,
                    is_stale=is_stale,
                )
            elif signal_type == "SELL":
                return GateDecision(
                    gate_name="fear_greed",
                    action="BOOST",
                    confidence_multiplier=cfg.fear_greed_confidence_boost,
                    position_size_multiplier=1.0,
                    reason=f"Extreme Greed ({score:.0f}) + SELL: contrarian bearish",
                    data_value=score,
                    data_timestamp=data_ts,
                    is_stale=is_stale,
                )

        return self._pass_decision("fear_greed", f"No action for {signal_type}",
                                    data_value=score, data_ts=data_ts)

    def _evaluate_cot(self, signal_type: str) -> GateDecision:
        """Evaluate CFTC COT positioning gate (contrarian)."""
        provider = self._providers.get("cot")
        if not provider:
            return self._pass_decision("cot", "Provider not configured")

        data = provider.get_latest()
        if data is None:
            return self._pass_decision("cot", "No data available", is_stale=True)

        data_age = provider.get_data_age_hours()
        is_stale = data_age > self.config.cot_max_staleness_hours
        data_ts = data.get("timestamp")

        try:
            from src.phase_19_paper_trading.cot_gate import evaluate_cot_gate
            result = evaluate_cot_gate(
                data, signal_type,
                caution_zscore=self.config.cot_extreme_zscore,
            )
            return GateDecision(
                gate_name="cot",
                action=result["action"],
                confidence_multiplier=result["confidence_multiplier"],
                position_size_multiplier=result["position_size_multiplier"],
                reason=result["reason"],
                data_value=data.get("net_speculator_zscore"),
                data_timestamp=data_ts,
                is_stale=is_stale,
            )
        except Exception as e:
            logger.warning(f"COT gate evaluation failed: {e}")
            return self._pass_decision("cot", f"Evaluation error: {e}")

    def _evaluate_aaii(self, signal_type: str) -> GateDecision:
        """Evaluate AAII sentiment gate (contrarian)."""
        provider = self._providers.get("aaii")
        if not provider:
            return self._pass_decision("aaii", "Provider not configured")

        data = provider.get_latest()
        if data is None:
            return self._pass_decision("aaii", "No data available", is_stale=True)

        data_age = provider.get_data_age_hours()
        is_stale = data_age > self.config.aaii_max_staleness_hours
        data_ts = data.get("timestamp")

        try:
            from src.phase_19_paper_trading.aaii_gate import evaluate_aaii_gate
            result = evaluate_aaii_gate(
                data, signal_type,
                caution_spread=self.config.aaii_extreme_spread,
            )
            return GateDecision(
                gate_name="aaii",
                action=result["action"],
                confidence_multiplier=result["confidence_multiplier"],
                position_size_multiplier=result["position_size_multiplier"],
                reason=result["reason"],
                data_value=data.get("bull_bear_spread"),
                data_timestamp=data_ts,
                is_stale=is_stale,
            )
        except Exception as e:
            logger.warning(f"AAII gate evaluation failed: {e}")
            return self._pass_decision("aaii", f"Evaluation error: {e}")

    def _evaluate_gex(self, signal_type: str) -> GateDecision:
        """
        Evaluate GEX (gamma exposure) gate.

        Positive GEX = dealers long gamma = mean-reverting market.
        Negative GEX = dealers short gamma = trending/gap-prone market.
        """
        provider = self._providers.get("gex")
        if not provider:
            return self._pass_decision("gex", "Provider not configured")

        data = provider.get_latest()
        if data is None:
            return self._pass_decision("gex", "No data available", is_stale=True)

        gex_regime = data.get("regime", "neutral")
        gex_proxy = data.get("proxy", 0)
        data_age = provider.get_data_age_hours()
        is_stale = data_age > self.config.gex_max_staleness_hours
        data_ts = data.get("timestamp")
        cfg = self.config

        if gex_regime == "neutral":
            return self._pass_decision("gex", f"Neutral GEX (proxy={gex_proxy:.2f})",
                                        data_value=gex_proxy, data_ts=data_ts, is_stale=is_stale)

        # Positive GEX: mean-reverting → contrarian signals get a boost
        if gex_regime == "positive_gex":
            # In mean-reverting market, contrarian trades are favorable
            return GateDecision(
                gate_name="gex",
                action="BOOST",
                confidence_multiplier=cfg.gex_trend_confidence_boost,
                position_size_multiplier=1.0,
                reason=f"Positive GEX ({gex_proxy:.2f}): mean-reverting market",
                data_value=gex_proxy,
                data_timestamp=data_ts,
                is_stale=is_stale,
            )

        # Negative GEX: trending → reduce confidence (higher gap risk)
        if gex_regime == "negative_gex":
            return GateDecision(
                gate_name="gex",
                action="REDUCE",
                confidence_multiplier=cfg.gex_trend_confidence_reduce,
                position_size_multiplier=1.0,
                reason=f"Negative GEX ({gex_proxy:.2f}): trending/gap risk",
                data_value=gex_proxy,
                data_timestamp=data_ts,
                is_stale=is_stale,
            )

        return self._pass_decision("gex", f"Unknown regime: {gex_regime}",
                                    data_value=gex_proxy, data_ts=data_ts)

    def _evaluate_insider(self, signal_type: str) -> GateDecision:
        """
        Evaluate insider sentiment gate.

        Uses Finnhub insider transaction data. Extreme net buying/selling
        provides contrarian signals.
        """
        provider = self._providers.get("insider")
        if not provider:
            return self._pass_decision("insider", "Provider not configured")

        data = provider.get_latest()
        if data is None:
            return self._pass_decision("insider", "No data available", is_stale=True)

        data_age = provider.get_data_age_hours()
        is_stale = data_age > self.config.insider_max_staleness_hours
        data_ts = data.get("timestamp")

        # Use the evaluate function from insider_gate module
        try:
            from src.phase_19_paper_trading.insider_gate import evaluate_insider_gate
            result = evaluate_insider_gate(
                data, signal_type,
                caution_percentile=self.config.insider_caution_percentile,
                caution_position_reduce=self.config.insider_position_reduce,
            )
            return GateDecision(
                gate_name="insider",
                action=result["action"],
                confidence_multiplier=result["confidence_multiplier"],
                position_size_multiplier=result["position_size_multiplier"],
                reason=result["reason"],
                data_value=data.get("net_buy_ratio"),
                data_timestamp=data_ts,
                is_stale=is_stale,
            )
        except Exception as e:
            logger.warning(f"Insider gate evaluation failed: {e}")
            return self._pass_decision("insider", f"Evaluation error: {e}")

    def _evaluate_macro_calendar(self, signal_type: str) -> GateDecision:
        """Evaluate macro calendar gate (FOMC/NFP/CPI event risk)."""
        provider = self._providers.get("macro_calendar")
        if not provider:
            return self._pass_decision("macro_calendar", "Provider not configured")

        data = provider.get_latest()
        if data is None:
            return self._pass_decision("macro_calendar", "No calendar data", is_stale=True)

        try:
            from src.phase_19_paper_trading.macro_calendar_gate import evaluate_macro_calendar_gate
            result = evaluate_macro_calendar_gate(data, signal_type)
            return GateDecision(
                gate_name="macro_calendar",
                action=result["action"],
                confidence_multiplier=result["confidence_multiplier"],
                position_size_multiplier=result["position_size_multiplier"],
                reason=result["reason"],
            )
        except Exception as e:
            logger.warning(f"Macro calendar gate evaluation failed: {e}")
            return self._pass_decision("macro_calendar", f"Evaluation error: {e}")

    def _evaluate_vol_regime(self, signal_type: str) -> GateDecision:
        """Evaluate volatility regime gate (VIX-based)."""
        provider = self._providers.get("vol_regime")
        if not provider:
            return self._pass_decision("vol_regime", "Provider not configured")

        data = provider.get_latest()
        if data is None:
            return self._pass_decision("vol_regime", "No VIX data", is_stale=True)

        try:
            from src.phase_19_paper_trading.vol_regime_gate import evaluate_vol_regime_gate
            result = evaluate_vol_regime_gate(
                data, signal_type,
                block_vix_threshold=self.config.vol_regime_block_vix,
                reduce_vix_threshold=self.config.vol_regime_reduce_vix,
            )
            vix_level = data.get("vix_level")
            return GateDecision(
                gate_name="vol_regime",
                action=result["action"],
                confidence_multiplier=result["confidence_multiplier"],
                position_size_multiplier=result["position_size_multiplier"],
                reason=result["reason"],
                data_value=vix_level,
            )
        except Exception as e:
            logger.warning(f"Vol regime gate evaluation failed: {e}")
            return self._pass_decision("vol_regime", f"Evaluation error: {e}")

    def _pass_decision(
        self,
        gate_name: str,
        reason: str,
        is_stale: bool = False,
        data_value: Optional[float] = None,
        data_ts: Optional[datetime] = None,
    ) -> GateDecision:
        """Create a PASS (no-op) gate decision."""
        return GateDecision(
            gate_name=gate_name,
            action="PASS",
            confidence_multiplier=1.0,
            position_size_multiplier=1.0,
            reason=reason,
            data_value=data_value,
            data_timestamp=data_ts,
            is_stale=is_stale,
        )

    def get_status(self) -> Dict[str, Any]:
        """Get current gate status for dashboard/monitoring."""
        status = {
            "enabled": self.config.gates_enabled,
            "gates": {},
        }

        for name, provider in self._providers.items():
            data = provider.get_latest()
            status["gates"][name] = {
                "has_data": data is not None,
                "data_age_hours": round(provider.get_data_age_hours(), 1),
                "last_value": data.get("score") if data else None,
            }

        return status
