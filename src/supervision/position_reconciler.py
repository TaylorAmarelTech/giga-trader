"""
Position Reconciler - Compares Alpaca positions with internal tracking.

Ensures consistency between:
1. What Alpaca says our positions are (source of truth)
2. What our internal tracking thinks we have

Alerts on any mismatch and optionally auto-corrects.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
import logging

logger = logging.getLogger("GigaTrader.Reconciler")


class ReconciliationStatus(Enum):
    """Status of position reconciliation."""
    MATCHED = "matched"
    MISMATCH_QUANTITY = "mismatch_quantity"
    MISMATCH_SIDE = "mismatch_side"
    MISSING_INTERNAL = "missing_internal"  # Alpaca has position, we don't track it
    MISSING_ALPACA = "missing_alpaca"      # We think we have position, Alpaca doesn't
    ORPHAN_ORDER = "orphan_order"          # Open order with no tracked position


@dataclass
class ReconciliationResult:
    """Result of reconciling a single position."""
    timestamp: datetime
    status: ReconciliationStatus
    symbol: str
    alpaca_quantity: float
    internal_quantity: float
    alpaca_side: Optional[str]
    internal_side: Optional[str]
    alpaca_avg_entry: float
    internal_avg_entry: float
    severity: str  # "info", "warning", "critical"
    recommended_action: str
    auto_corrected: bool = False


class PositionReconciler:
    """
    Reconciles Alpaca positions with internal position tracking.

    Key responsibilities:
    - Fetch current positions from Alpaca API (source of truth)
    - Compare with internal tracking state
    - Detect and classify mismatches
    - Generate alerts for discrepancies
    - Optionally auto-correct internal state to match Alpaca
    """

    def __init__(
        self,
        alpaca_client: Any,
        auto_correct_threshold: float = 1.0,
        reconcile_interval_seconds: int = 30,
        alert_callback: Optional[Callable] = None,
    ):
        """
        Initialize PositionReconciler.

        Args:
            alpaca_client: Alpaca trading client
            auto_correct_threshold: Auto-correct if quantity diff <= this
            reconcile_interval_seconds: How often to reconcile
            alert_callback: Function to call for alerts
        """
        self.client = alpaca_client
        self.auto_correct_threshold = auto_correct_threshold
        self.reconcile_interval_seconds = reconcile_interval_seconds
        self.alert_callback = alert_callback

        self._last_reconciliation: Optional[List[ReconciliationResult]] = None
        self._last_reconciliation_time: Optional[datetime] = None
        self._internal_positions: Dict[str, Dict] = {}

    def set_internal_positions(self, positions: Dict[str, Dict]):
        """
        Set the internal position tracking state.

        Args:
            positions: Dict of symbol -> position info
                       e.g., {"SPY": {"quantity": 10, "side": "long", "avg_entry": 450.0}}
        """
        self._internal_positions = positions.copy()

    def reconcile(self) -> List[ReconciliationResult]:
        """
        Perform full reconciliation between Alpaca and internal state.

        Returns:
            List of ReconciliationResult for all positions checked
        """
        results = []
        timestamp = datetime.now()

        try:
            # Get Alpaca positions (source of truth)
            alpaca_positions = self.get_alpaca_positions()
        except Exception as e:
            logger.error(f"Failed to get Alpaca positions: {e}")
            return [ReconciliationResult(
                timestamp=timestamp,
                status=ReconciliationStatus.MISSING_ALPACA,
                symbol="ERROR",
                alpaca_quantity=0,
                internal_quantity=0,
                alpaca_side=None,
                internal_side=None,
                alpaca_avg_entry=0,
                internal_avg_entry=0,
                severity="critical",
                recommended_action="Check Alpaca API connection",
            )]

        # Check each Alpaca position against internal
        alpaca_symbols = set(alpaca_positions.keys())
        internal_symbols = set(self._internal_positions.keys())

        # Positions in both
        for symbol in alpaca_symbols & internal_symbols:
            result = self._compare_position(
                symbol,
                alpaca_positions[symbol],
                self._internal_positions[symbol],
                timestamp,
            )
            results.append(result)

        # Positions only in Alpaca (we're missing internally)
        for symbol in alpaca_symbols - internal_symbols:
            alpaca_pos = alpaca_positions[symbol]
            result = ReconciliationResult(
                timestamp=timestamp,
                status=ReconciliationStatus.MISSING_INTERNAL,
                symbol=symbol,
                alpaca_quantity=alpaca_pos.get("quantity", 0),
                internal_quantity=0,
                alpaca_side=alpaca_pos.get("side"),
                internal_side=None,
                alpaca_avg_entry=alpaca_pos.get("avg_entry", 0),
                internal_avg_entry=0,
                severity="warning",
                recommended_action="Add to internal tracking or investigate",
            )
            results.append(result)
            logger.warning(f"Position in Alpaca but not tracked internally: {symbol}")

        # Positions only internal (Alpaca doesn't have)
        for symbol in internal_symbols - alpaca_symbols:
            internal_pos = self._internal_positions[symbol]
            # Only alert if internal thinks we have a non-zero position
            if internal_pos.get("quantity", 0) != 0:
                result = ReconciliationResult(
                    timestamp=timestamp,
                    status=ReconciliationStatus.MISSING_ALPACA,
                    symbol=symbol,
                    alpaca_quantity=0,
                    internal_quantity=internal_pos.get("quantity", 0),
                    alpaca_side=None,
                    internal_side=internal_pos.get("side"),
                    alpaca_avg_entry=0,
                    internal_avg_entry=internal_pos.get("avg_entry", 0),
                    severity="critical",
                    recommended_action="Position may have been closed externally - update internal state",
                )
                results.append(result)
                logger.error(f"Position tracked internally but not in Alpaca: {symbol}")

        self._last_reconciliation = results
        self._last_reconciliation_time = timestamp

        # Alert on critical mismatches
        self._alert_on_mismatches(results)

        return results

    def _compare_position(
        self,
        symbol: str,
        alpaca_pos: Dict,
        internal_pos: Dict,
        timestamp: datetime,
    ) -> ReconciliationResult:
        """Compare a single position between Alpaca and internal."""
        alpaca_qty = float(alpaca_pos.get("quantity", 0))
        internal_qty = float(internal_pos.get("quantity", 0))
        alpaca_side = alpaca_pos.get("side", "long")
        internal_side = internal_pos.get("side", "long")
        alpaca_entry = float(alpaca_pos.get("avg_entry", 0))
        internal_entry = float(internal_pos.get("avg_entry", 0))

        # Check for side mismatch (most severe)
        if alpaca_side != internal_side and alpaca_qty != 0:
            return ReconciliationResult(
                timestamp=timestamp,
                status=ReconciliationStatus.MISMATCH_SIDE,
                symbol=symbol,
                alpaca_quantity=alpaca_qty,
                internal_quantity=internal_qty,
                alpaca_side=alpaca_side,
                internal_side=internal_side,
                alpaca_avg_entry=alpaca_entry,
                internal_avg_entry=internal_entry,
                severity="critical",
                recommended_action="CRITICAL: Side mismatch - update internal state immediately",
            )

        # Check for quantity mismatch
        qty_diff = abs(alpaca_qty - internal_qty)
        if qty_diff > 0.01:  # Allow small floating point differences
            severity = "warning" if qty_diff <= self.auto_correct_threshold else "critical"
            return ReconciliationResult(
                timestamp=timestamp,
                status=ReconciliationStatus.MISMATCH_QUANTITY,
                symbol=symbol,
                alpaca_quantity=alpaca_qty,
                internal_quantity=internal_qty,
                alpaca_side=alpaca_side,
                internal_side=internal_side,
                alpaca_avg_entry=alpaca_entry,
                internal_avg_entry=internal_entry,
                severity=severity,
                recommended_action=f"Quantity mismatch: Alpaca={alpaca_qty}, Internal={internal_qty}",
            )

        # Matched
        return ReconciliationResult(
            timestamp=timestamp,
            status=ReconciliationStatus.MATCHED,
            symbol=symbol,
            alpaca_quantity=alpaca_qty,
            internal_quantity=internal_qty,
            alpaca_side=alpaca_side,
            internal_side=internal_side,
            alpaca_avg_entry=alpaca_entry,
            internal_avg_entry=internal_entry,
            severity="info",
            recommended_action="None - positions match",
        )

    def get_alpaca_positions(self) -> Dict[str, Dict]:
        """
        Fetch current positions from Alpaca.

        Returns:
            Dict of symbol -> position info
        """
        positions = {}

        try:
            alpaca_positions = self.client.get_all_positions()

            for pos in alpaca_positions:
                symbol = getattr(pos, 'symbol', str(pos))
                qty = float(getattr(pos, 'qty', 0))
                side = getattr(pos, 'side', 'long')
                avg_entry = float(getattr(pos, 'avg_entry_price', 0))
                market_value = float(getattr(pos, 'market_value', 0))
                unrealized_pl = float(getattr(pos, 'unrealized_pl', 0))

                positions[symbol] = {
                    "quantity": qty,
                    "side": side,
                    "avg_entry": avg_entry,
                    "market_value": market_value,
                    "unrealized_pl": unrealized_pl,
                }

        except Exception as e:
            logger.error(f"Failed to fetch Alpaca positions: {e}")
            raise

        return positions

    def check_open_orders(self) -> List[Dict]:
        """
        Check for orphan orders that don't match internal expectations.

        Returns:
            List of potentially orphaned orders
        """
        orphans = []

        try:
            # Get open orders from Alpaca
            orders = self.client.trading_client.get_orders()

            for order in orders:
                symbol = getattr(order, 'symbol', '')
                order_id = getattr(order, 'id', '')
                status = getattr(order, 'status', '')

                # Check if order is for a symbol we're not tracking
                if symbol not in self._internal_positions:
                    orphans.append({
                        "order_id": str(order_id),
                        "symbol": symbol,
                        "status": str(status),
                        "reason": "Order for untracked symbol",
                    })

        except Exception as e:
            logger.error(f"Failed to check open orders: {e}")

        return orphans

    def correct_internal_state(
        self,
        symbol: str,
        alpaca_position: Dict,
    ) -> bool:
        """
        Correct internal state to match Alpaca (source of truth).

        Only call for minor discrepancies or after manual approval.

        Args:
            symbol: Symbol to correct
            alpaca_position: Alpaca's position data

        Returns:
            True if correction was made
        """
        logger.info(f"Correcting internal state for {symbol} to match Alpaca")

        self._internal_positions[symbol] = {
            "quantity": alpaca_position.get("quantity", 0),
            "side": alpaca_position.get("side", "long"),
            "avg_entry": alpaca_position.get("avg_entry", 0),
        }

        return True

    def has_critical_mismatch(self) -> bool:
        """Check if any critical mismatches exist."""
        if not self._last_reconciliation:
            return False

        return any(
            r.severity == "critical"
            for r in self._last_reconciliation
        )

    def get_last_reconciliation(self) -> Optional[List[ReconciliationResult]]:
        """Get results of last reconciliation."""
        return self._last_reconciliation

    def _alert_on_mismatches(self, results: List[ReconciliationResult]):
        """Send alerts for any mismatches."""
        if not self.alert_callback:
            return

        for result in results:
            if result.status != ReconciliationStatus.MATCHED:
                self.alert_callback(
                    result.severity,
                    f"Position mismatch for {result.symbol}: {result.status.value} - {result.recommended_action}",
                )

    def get_reconciliation_summary(self) -> Dict:
        """Get summary of last reconciliation."""
        if not self._last_reconciliation:
            return {"status": "never_run"}

        return {
            "last_run": self._last_reconciliation_time.isoformat() if self._last_reconciliation_time else None,
            "total_positions": len(self._last_reconciliation),
            "matched": sum(1 for r in self._last_reconciliation if r.status == ReconciliationStatus.MATCHED),
            "mismatches": sum(1 for r in self._last_reconciliation if r.status != ReconciliationStatus.MATCHED),
            "critical": sum(1 for r in self._last_reconciliation if r.severity == "critical"),
            "warnings": sum(1 for r in self._last_reconciliation if r.severity == "warning"),
        }

    def sync_from_alpaca(self) -> int:
        """
        Sync all positions from Alpaca to internal tracking on startup.

        This ensures we always start with accurate position state from the
        source of truth (Alpaca).

        Returns:
            Number of positions synced
        """
        try:
            alpaca_positions = self.get_alpaca_positions()

            synced_count = 0
            for symbol, pos_data in alpaca_positions.items():
                if symbol not in self._internal_positions:
                    logger.info(f"Syncing position from Alpaca: {symbol} qty={pos_data.get('quantity')}")
                    self._internal_positions[symbol] = {
                        "quantity": pos_data.get("quantity", 0),
                        "side": pos_data.get("side", "long"),
                        "avg_entry": pos_data.get("avg_entry", 0),
                        "market_value": pos_data.get("market_value", 0),
                        "unrealized_pl": pos_data.get("unrealized_pl", 0),
                    }
                    synced_count += 1
                else:
                    # Update existing position to match Alpaca (source of truth)
                    self._internal_positions[symbol].update({
                        "quantity": pos_data.get("quantity", 0),
                        "side": pos_data.get("side", "long"),
                        "avg_entry": pos_data.get("avg_entry", 0),
                        "market_value": pos_data.get("market_value", 0),
                        "unrealized_pl": pos_data.get("unrealized_pl", 0),
                    })

            if synced_count > 0:
                logger.info(f"Synced {synced_count} positions from Alpaca")

            return synced_count

        except Exception as e:
            logger.error(f"Failed to sync positions from Alpaca: {e}")
            return 0

    def auto_correct_missing_internal(self) -> int:
        """
        Auto-correct any MISSING_INTERNAL positions (Alpaca has, we don't track).

        Call this after reconcile() to automatically add untracked positions.

        Returns:
            Number of positions auto-corrected
        """
        if not self._last_reconciliation:
            return 0

        corrected = 0
        alpaca_positions = self.get_alpaca_positions()

        for result in self._last_reconciliation:
            if result.status == ReconciliationStatus.MISSING_INTERNAL:
                symbol = result.symbol
                if symbol in alpaca_positions:
                    self.correct_internal_state(symbol, alpaca_positions[symbol])
                    result.auto_corrected = True
                    corrected += 1
                    logger.info(f"Auto-corrected internal tracking for {symbol}")

        return corrected
