"""
GIGA TRADER - Backtest Engine
==============================
Tests trading configurations on historical data with realistic simulation.

Features:
  - Walk-forward backtesting
  - Realistic slippage and commission modeling
  - Position sizing with batch entries
  - Stop loss and take profit execution
  - Performance metrics (Sharpe, Sortino, max drawdown, etc.)
  - Regime analysis (bull/bear/sideways markets)
  - Monte Carlo stress testing

Usage:
    from src.backtest_engine import BacktestEngine

    engine = BacktestEngine(initial_capital=100000)
    results = engine.run_backtest(
        daily_data=df_daily,
        intraday_data=df_1min,
        swing_model=swing_model,
        timing_model=timing_model,
        entry_exit_model=entry_exit_model,
        config=grid_config,
    )
"""

import os
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# 1. TRADE RECORD
# =============================================================================

class Trade:
    """Represents a single trade with all details."""

    def __init__(
        self,
        trade_id: int,
        entry_date: datetime,
        direction: str,  # "LONG" or "SHORT"
        entry_price: float,
        position_size: float,  # Number of shares/contracts
        entry_cost: float,  # Total $ invested
    ):
        self.trade_id = trade_id
        self.entry_date = entry_date
        self.direction = direction
        self.entry_price = entry_price
        self.position_size = position_size
        self.entry_cost = entry_cost

        # Exit info (filled on close)
        self.exit_date: Optional[datetime] = None
        self.exit_price: Optional[float] = None
        self.exit_reason: Optional[str] = None  # "signal", "stop_loss", "take_profit", "eod"

        # Risk parameters
        self.stop_loss: Optional[float] = None
        self.take_profit: Optional[float] = None
        self.trailing_stop: Optional[float] = None

        # Batch info
        self.batches: List[Dict] = []  # [{time, price, size}, ...]
        self.avg_entry_price: float = entry_price

        # P&L
        self.realized_pnl: float = 0.0
        self.commission: float = 0.0
        self.slippage: float = 0.0

    def close(
        self,
        exit_date: datetime,
        exit_price: float,
        reason: str,
        commission: float = 0.0,
        slippage: float = 0.0,
    ):
        """Close the trade."""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_reason = reason
        self.commission = commission
        self.slippage = slippage

        # Calculate P&L
        if self.direction == "LONG":
            gross_pnl = (self.exit_price - self.avg_entry_price) * self.position_size
        else:  # SHORT
            gross_pnl = (self.avg_entry_price - self.exit_price) * self.position_size

        self.realized_pnl = gross_pnl - self.commission - self.slippage

    @property
    def is_open(self) -> bool:
        return self.exit_date is None

    @property
    def return_pct(self) -> float:
        if self.entry_cost == 0:
            return 0.0
        return self.realized_pnl / self.entry_cost

    @property
    def holding_period(self) -> Optional[timedelta]:
        if self.exit_date is None:
            return None
        return self.exit_date - self.entry_date

    def to_dict(self) -> Dict:
        return {
            "trade_id": self.trade_id,
            "entry_date": self.entry_date,
            "exit_date": self.exit_date,
            "direction": self.direction,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "position_size": self.position_size,
            "entry_cost": self.entry_cost,
            "realized_pnl": self.realized_pnl,
            "return_pct": self.return_pct,
            "exit_reason": self.exit_reason,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
        }


# =============================================================================
# 2. PORTFOLIO STATE
# =============================================================================

class Portfolio:
    """Tracks portfolio state during backtest."""

    def __init__(
        self,
        initial_capital: float = 100000,
        commission_per_share: float = 0.005,
        slippage_pct: float = 0.0001,
    ):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission_per_share = commission_per_share
        self.slippage_pct = slippage_pct

        # Positions
        self.open_trades: List[Trade] = []
        self.closed_trades: List[Trade] = []
        self.trade_count = 0

        # Daily tracking
        self.equity_curve: List[Dict] = []
        self.daily_returns: List[float] = []

        # Statistics
        self.peak_equity = initial_capital
        self.max_drawdown = 0.0
        self.total_commission = 0.0
        self.total_slippage = 0.0

    @property
    def equity(self) -> float:
        """Current portfolio value."""
        open_value = sum(
            t.position_size * t.entry_price  # Simplified - should use current price
            for t in self.open_trades
        )
        return self.cash + open_value

    @property
    def open_position_value(self) -> float:
        return sum(t.entry_cost for t in self.open_trades)

    @property
    def position_count(self) -> int:
        return len(self.open_trades)

    def can_open_position(self, cost: float) -> bool:
        """Check if we have enough cash."""
        return self.cash >= cost

    def open_trade(
        self,
        date: datetime,
        direction: str,
        price: float,
        position_value: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
    ) -> Optional[Trade]:
        """Open a new trade."""
        # Apply slippage
        if direction == "LONG":
            entry_price = price * (1 + self.slippage_pct)
        else:
            entry_price = price * (1 - self.slippage_pct)

        # Calculate position size
        position_size = position_value / entry_price
        commission = position_size * self.commission_per_share

        # Check if we have enough cash
        total_cost = position_value + commission
        if not self.can_open_position(total_cost):
            return None

        # Create trade
        self.trade_count += 1
        trade = Trade(
            trade_id=self.trade_count,
            entry_date=date,
            direction=direction,
            entry_price=entry_price,
            position_size=position_size,
            entry_cost=position_value,
        )
        trade.stop_loss = stop_loss
        trade.take_profit = take_profit
        trade.commission = commission

        # Update portfolio
        self.cash -= total_cost
        self.total_commission += commission
        self.total_slippage += abs(entry_price - price) * position_size
        self.open_trades.append(trade)

        return trade

    def close_trade(
        self,
        trade: Trade,
        date: datetime,
        price: float,
        reason: str,
    ):
        """Close an existing trade."""
        if trade not in self.open_trades:
            return

        # Apply slippage
        if trade.direction == "LONG":
            exit_price = price * (1 - self.slippage_pct)
        else:
            exit_price = price * (1 + self.slippage_pct)

        # Commission
        commission = trade.position_size * self.commission_per_share

        # Close trade
        trade.close(
            exit_date=date,
            exit_price=exit_price,
            reason=reason,
            commission=commission,
            slippage=abs(exit_price - price) * trade.position_size,
        )

        # Update portfolio
        if trade.direction == "LONG":
            proceeds = exit_price * trade.position_size
        else:
            # For short: initial_cost + (entry - exit) * size
            proceeds = trade.entry_cost + (trade.entry_price - exit_price) * trade.position_size

        self.cash += proceeds - commission
        self.total_commission += commission
        self.total_slippage += abs(exit_price - price) * trade.position_size

        # Move to closed
        self.open_trades.remove(trade)
        self.closed_trades.append(trade)

    def record_daily(self, date: datetime, close_price: float):
        """Record daily equity and update drawdown."""
        # Calculate current equity with mark-to-market
        open_value = 0
        for trade in self.open_trades:
            if trade.direction == "LONG":
                open_value += trade.position_size * close_price
            else:
                # Short position value
                open_value += trade.entry_cost + (trade.entry_price - close_price) * trade.position_size

        current_equity = self.cash + open_value

        # Record
        self.equity_curve.append({
            "date": date,
            "equity": current_equity,
            "cash": self.cash,
            "positions_value": open_value,
            "n_positions": len(self.open_trades),
        })

        # Daily return
        if len(self.equity_curve) > 1:
            prev_equity = self.equity_curve[-2]["equity"]
            daily_return = (current_equity - prev_equity) / prev_equity if prev_equity > 0 else 0
            self.daily_returns.append(daily_return)
        else:
            self.daily_returns.append(0)

        # Drawdown
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity
        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        self.max_drawdown = max(self.max_drawdown, drawdown)


# =============================================================================
# 3. BACKTEST ENGINE
# =============================================================================

class BacktestEngine:
    """
    Main backtest engine for testing trading strategies.
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        commission_per_share: float = 0.005,
        slippage_pct: float = 0.0001,
        max_position_pct: float = 0.25,
        max_positions: int = 1,
        use_batching: bool = True,
    ):
        self.initial_capital = initial_capital
        self.commission_per_share = commission_per_share
        self.slippage_pct = slippage_pct
        self.max_position_pct = max_position_pct
        self.max_positions = max_positions
        self.use_batching = use_batching

        self.portfolio: Optional[Portfolio] = None
        self.results: Dict = {}

    def run_backtest(
        self,
        daily_data: pd.DataFrame,
        swing_predictions: pd.Series,
        timing_predictions: pd.Series,
        entry_exit_predictions: Optional[pd.DataFrame] = None,
        config: Optional[Dict] = None,
    ) -> Dict:
        """
        Run backtest on historical data.

        Args:
            daily_data: Daily OHLCV data indexed by date
            swing_predictions: Series of swing probabilities (0-1)
            timing_predictions: Series of timing probabilities (0-1)
            entry_exit_predictions: Optional DataFrame with entry/exit model outputs
            config: Optional configuration dict

        Returns:
            Dict with backtest results
        """
        print("[BACKTEST] Starting backtest...")

        # Initialize portfolio
        self.portfolio = Portfolio(
            initial_capital=self.initial_capital,
            commission_per_share=self.commission_per_share,
            slippage_pct=self.slippage_pct,
        )

        # Default config
        config = config or {}
        min_swing_conf = config.get("min_swing_confidence", 0.60)
        min_timing_conf = config.get("min_timing_confidence", 0.55)
        base_position_pct = config.get("base_position_pct", 0.10)
        stop_loss_pct = config.get("stop_loss_pct", 0.01)
        take_profit_pct = config.get("take_profit_pct", 0.02)
        long_only = config.get("long_only", True)

        # Align data
        common_dates = daily_data.index.intersection(swing_predictions.index)
        common_dates = common_dates.intersection(timing_predictions.index)

        print(f"  Trading days: {len(common_dates)}")

        # Main loop
        for date in common_dates:
            row = daily_data.loc[date]
            swing_prob = swing_predictions.loc[date]
            timing_prob = timing_predictions.loc[date]

            # Get today's prices
            open_price = row['open']
            high_price = row['high']
            low_price = row['low']
            close_price = row['close']

            # Check for stop loss / take profit hits on open positions
            for trade in list(self.portfolio.open_trades):
                should_close, close_reason, close_price_used = self._check_exit_conditions(
                    trade, high_price, low_price, close_price
                )
                if should_close:
                    self.portfolio.close_trade(trade, date, close_price_used, close_reason)

            # Generate signal
            signal = self._generate_signal(
                swing_prob, timing_prob, min_swing_conf, min_timing_conf, long_only
            )

            # Open new position if signal and no open position
            if signal != "HOLD" and self.portfolio.position_count < self.max_positions:
                # Calculate position size
                position_value = self.portfolio.equity * base_position_pct
                position_value = min(position_value, self.portfolio.equity * self.max_position_pct)

                # Get entry/exit predictions if available
                if entry_exit_predictions is not None and date in entry_exit_predictions.index:
                    ee_pred = entry_exit_predictions.loc[date]
                    position_pct = ee_pred.get("position_size_pct", base_position_pct)
                    position_value = self.portfolio.equity * position_pct
                    sl_pct = ee_pred.get("stop_loss_pct", stop_loss_pct)
                    tp_pct = ee_pred.get("take_profit_pct", take_profit_pct)
                else:
                    sl_pct = stop_loss_pct
                    tp_pct = take_profit_pct

                # Calculate stop/take profit prices
                if signal == "LONG":
                    stop_loss = open_price * (1 - sl_pct)
                    take_profit = open_price * (1 + tp_pct)
                else:
                    stop_loss = open_price * (1 + sl_pct)
                    take_profit = open_price * (1 - tp_pct)

                # Open trade
                trade = self.portfolio.open_trade(
                    date=date,
                    direction=signal,
                    price=open_price,
                    position_value=position_value,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                )

                if trade:
                    pass  # Trade opened successfully

            # Close all positions at end of day (no overnight)
            for trade in list(self.portfolio.open_trades):
                self.portfolio.close_trade(trade, date, close_price, "eod")

            # Record daily
            self.portfolio.record_daily(date, close_price)

        # Calculate final metrics
        self.results = self._calculate_metrics()

        print(f"  Trades: {len(self.portfolio.closed_trades)}")
        print(f"  Final equity: ${self.portfolio.equity:,.2f}")
        print(f"  Total return: {self.results['total_return_pct']:.2f}%")
        print(f"  Sharpe ratio: {self.results['sharpe_ratio']:.3f}")
        print(f"  Max drawdown: {self.results['max_drawdown_pct']:.2f}%")

        return self.results

    def _generate_signal(
        self,
        swing_prob: float,
        timing_prob: float,
        min_swing: float,
        min_timing: float,
        long_only: bool,
    ) -> str:
        """Generate trading signal from model probabilities."""
        # Bullish: high swing prob + timing prob (low before high)
        if swing_prob > min_swing and timing_prob > min_timing:
            return "LONG"

        # Bearish: low swing prob + low timing prob
        if not long_only and swing_prob < (1 - min_swing) and timing_prob < (1 - min_timing):
            return "SHORT"

        return "HOLD"

    def _check_exit_conditions(
        self,
        trade: Trade,
        high: float,
        low: float,
        close: float,
    ) -> Tuple[bool, str, float]:
        """Check if trade should be closed due to stop/TP."""
        if trade.direction == "LONG":
            # Check stop loss (hit if low touches stop)
            if trade.stop_loss and low <= trade.stop_loss:
                return True, "stop_loss", trade.stop_loss

            # Check take profit (hit if high touches TP)
            if trade.take_profit and high >= trade.take_profit:
                return True, "take_profit", trade.take_profit

        else:  # SHORT
            # Check stop loss (hit if high touches stop)
            if trade.stop_loss and high >= trade.stop_loss:
                return True, "stop_loss", trade.stop_loss

            # Check take profit (hit if low touches TP)
            if trade.take_profit and low <= trade.take_profit:
                return True, "take_profit", trade.take_profit

        return False, "", close

    def _calculate_metrics(self) -> Dict:
        """Calculate performance metrics."""
        equity_df = pd.DataFrame(self.portfolio.equity_curve)
        daily_returns = np.array(self.portfolio.daily_returns)

        # Basic metrics
        total_return = (self.portfolio.equity - self.initial_capital) / self.initial_capital
        total_return_pct = total_return * 100

        # Risk metrics
        if len(daily_returns) > 1:
            sharpe_ratio = np.sqrt(252) * np.mean(daily_returns) / (np.std(daily_returns) + 1e-8)

            # Sortino (downside deviation)
            downside_returns = daily_returns[daily_returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
            sortino_ratio = np.sqrt(252) * np.mean(daily_returns) / downside_std
        else:
            sharpe_ratio = 0
            sortino_ratio = 0

        # Trade metrics
        trades = self.portfolio.closed_trades
        if trades:
            trade_returns = [t.return_pct for t in trades]
            win_trades = [t for t in trades if t.realized_pnl > 0]
            loss_trades = [t for t in trades if t.realized_pnl <= 0]

            win_rate = len(win_trades) / len(trades) if trades else 0
            avg_win = np.mean([t.realized_pnl for t in win_trades]) if win_trades else 0
            avg_loss = np.mean([abs(t.realized_pnl) for t in loss_trades]) if loss_trades else 0
            profit_factor = (sum(t.realized_pnl for t in win_trades) /
                           max(abs(sum(t.realized_pnl for t in loss_trades)), 1))

            # Exit analysis
            exit_reasons = defaultdict(int)
            for t in trades:
                exit_reasons[t.exit_reason] += 1
        else:
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
            exit_reasons = {}

        return {
            # Returns
            "total_return_pct": total_return_pct,
            "total_return": total_return,
            "final_equity": self.portfolio.equity,

            # Risk
            "sharpe_ratio": sharpe_ratio,
            "sortino_ratio": sortino_ratio,
            "max_drawdown_pct": self.portfolio.max_drawdown * 100,
            "max_drawdown": self.portfolio.max_drawdown,

            # Trade metrics
            "n_trades": len(trades),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "exit_reasons": dict(exit_reasons),

            # Costs
            "total_commission": self.portfolio.total_commission,
            "total_slippage": self.portfolio.total_slippage,

            # Time series
            "equity_curve": equity_df.to_dict('records') if not equity_df.empty else [],
            "daily_returns": daily_returns.tolist(),
        }

    def get_trade_log(self) -> pd.DataFrame:
        """Get all trades as DataFrame."""
        if not self.portfolio:
            return pd.DataFrame()

        return pd.DataFrame([t.to_dict() for t in self.portfolio.closed_trades])


# =============================================================================
# 4. WALK-FORWARD BACKTEST
# =============================================================================

class WalkForwardBacktest:
    """
    Walk-forward analysis that re-trains models periodically.

    More realistic than standard backtest because:
    - Models are retrained on expanding/rolling window
    - No look-ahead bias
    - Tests model stability over time
    """

    def __init__(
        self,
        train_months: int = 24,
        test_months: int = 3,
        step_months: int = 3,
        anchored: bool = True,  # Expanding window vs rolling
    ):
        self.train_months = train_months
        self.test_months = test_months
        self.step_months = step_months
        self.anchored = anchored

        self.walk_results: List[Dict] = []

    def run_walk_forward(
        self,
        daily_data: pd.DataFrame,
        intraday_data: pd.DataFrame,
        feature_cols: List[str],
        model_trainer_fn,  # Function that trains and returns models
        config: Optional[Dict] = None,
    ) -> Dict:
        """
        Run walk-forward backtest.

        Args:
            daily_data: Daily data with features and targets
            intraday_data: Intraday data for timing model
            feature_cols: Feature column names
            model_trainer_fn: Function(train_data) -> (swing_model, timing_model)
            config: Backtest configuration

        Returns:
            Aggregated results
        """
        print("[WALK-FORWARD] Starting walk-forward analysis...")

        # Generate time periods
        dates = sorted(daily_data.index.unique())
        min_date = dates[0]
        max_date = dates[-1]

        # Calculate walk-forward windows
        windows = []
        train_start = min_date
        while True:
            if self.anchored:
                train_end = train_start + timedelta(days=self.train_months * 30)
            else:
                train_end = train_start + timedelta(days=self.train_months * 30)

            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=self.test_months * 30)

            if test_end > max_date:
                break

            windows.append({
                "train_start": train_start,
                "train_end": train_end,
                "test_start": test_start,
                "test_end": test_end,
            })

            if self.anchored:
                train_start = train_start  # Keep same start
            else:
                train_start = train_start + timedelta(days=self.step_months * 30)

            train_end = train_end + timedelta(days=self.step_months * 30)

        print(f"  Generated {len(windows)} walk-forward windows")

        # Run each window
        for i, window in enumerate(windows):
            print(f"\n  Window {i+1}/{len(windows)}: "
                  f"{window['train_start'].strftime('%Y-%m')} to {window['test_end'].strftime('%Y-%m')}")

            # Split data
            train_mask = (daily_data.index >= window['train_start']) & (daily_data.index <= window['train_end'])
            test_mask = (daily_data.index >= window['test_start']) & (daily_data.index <= window['test_end'])

            train_data = daily_data[train_mask]
            test_data = daily_data[test_mask]

            if len(train_data) < 100 or len(test_data) < 20:
                print("    Skipping - insufficient data")
                continue

            # Train models
            print(f"    Training on {len(train_data)} days...")
            try:
                swing_model, timing_model = model_trainer_fn(train_data, feature_cols)
            except Exception as e:
                print(f"    Training failed: {e}")
                continue

            # Generate predictions
            X_test = test_data[feature_cols].values
            swing_proba = swing_model.predict_proba(X_test)[:, 1]
            timing_proba = timing_model.predict_proba(X_test)[:, 1]

            swing_predictions = pd.Series(swing_proba, index=test_data.index)
            timing_predictions = pd.Series(timing_proba, index=test_data.index)

            # Run backtest
            engine = BacktestEngine(initial_capital=100000)
            results = engine.run_backtest(
                daily_data=test_data,
                swing_predictions=swing_predictions,
                timing_predictions=timing_predictions,
                config=config,
            )

            results["window"] = window
            self.walk_results.append(results)

        # Aggregate results
        return self._aggregate_results()

    def _aggregate_results(self) -> Dict:
        """Aggregate walk-forward results."""
        if not self.walk_results:
            return {}

        # Combine equity curves
        all_returns = []
        for r in self.walk_results:
            all_returns.extend(r.get("daily_returns", []))

        all_returns = np.array(all_returns)

        # Aggregate metrics
        return {
            "n_windows": len(self.walk_results),
            "total_return_pct": sum(r["total_return_pct"] for r in self.walk_results),
            "avg_window_return_pct": np.mean([r["total_return_pct"] for r in self.walk_results]),
            "avg_sharpe": np.mean([r["sharpe_ratio"] for r in self.walk_results]),
            "std_sharpe": np.std([r["sharpe_ratio"] for r in self.walk_results]),
            "worst_window_return": min(r["total_return_pct"] for r in self.walk_results),
            "best_window_return": max(r["total_return_pct"] for r in self.walk_results),
            "avg_win_rate": np.mean([r["win_rate"] for r in self.walk_results if r["n_trades"] > 0]),
            "total_trades": sum(r["n_trades"] for r in self.walk_results),
            "overall_sharpe": np.sqrt(252) * np.mean(all_returns) / (np.std(all_returns) + 1e-8) if len(all_returns) > 0 else 0,
            "window_results": self.walk_results,
        }


# =============================================================================
# 5. MONTE CARLO SIMULATION
# =============================================================================

class MonteCarloSimulator:
    """
    Monte Carlo simulation for stress testing.

    Methods:
    - Bootstrap: Resample historical trades with replacement
    - Shuffle: Random permutations of trade order
    - Synthetic: Generate synthetic trades based on distribution
    """

    def __init__(self, n_simulations: int = 1000, confidence_level: float = 0.95):
        self.n_simulations = n_simulations
        self.confidence_level = confidence_level

    def run_bootstrap(self, trades: List[Trade], initial_capital: float = 100000) -> Dict:
        """
        Bootstrap simulation - resample trades with replacement.
        """
        if not trades:
            return {}

        trade_returns = np.array([t.return_pct for t in trades])
        n_trades = len(trades)

        final_equities = []
        max_drawdowns = []

        for _ in range(self.n_simulations):
            # Resample trades
            sampled_returns = np.random.choice(trade_returns, size=n_trades, replace=True)

            # Calculate equity curve
            equity = initial_capital
            peak = initial_capital
            max_dd = 0

            for ret in sampled_returns:
                equity *= (1 + ret)
                peak = max(peak, equity)
                dd = (peak - equity) / peak
                max_dd = max(max_dd, dd)

            final_equities.append(equity)
            max_drawdowns.append(max_dd)

        final_equities = np.array(final_equities)
        max_drawdowns = np.array(max_drawdowns)

        # Calculate confidence intervals
        alpha = 1 - self.confidence_level
        return {
            "median_equity": np.median(final_equities),
            "mean_equity": np.mean(final_equities),
            "equity_ci_lower": np.percentile(final_equities, alpha/2 * 100),
            "equity_ci_upper": np.percentile(final_equities, (1 - alpha/2) * 100),
            "worst_case_equity": np.percentile(final_equities, 5),
            "best_case_equity": np.percentile(final_equities, 95),
            "median_max_dd": np.median(max_drawdowns),
            "dd_ci_upper": np.percentile(max_drawdowns, 95),
            "prob_profit": np.mean(final_equities > initial_capital),
            "prob_double": np.mean(final_equities > initial_capital * 2),
            "prob_lose_50pct": np.mean(final_equities < initial_capital * 0.5),
        }

    def run_regime_stress(
        self,
        trades: List[Trade],
        regime_labels: Optional[np.ndarray] = None,
        initial_capital: float = 100000,
    ) -> Dict:
        """
        Analyze performance in different market regimes.
        """
        if not trades:
            return {}

        if regime_labels is None:
            # Estimate regimes from trade returns
            trade_returns = np.array([t.return_pct for t in trades])

            # Simple regime classification based on rolling performance
            rolling_mean = pd.Series(trade_returns).rolling(20, min_periods=5).mean()
            rolling_std = pd.Series(trade_returns).rolling(20, min_periods=5).std()

            regime_labels = np.where(
                rolling_mean > rolling_std,
                "BULL",
                np.where(rolling_mean < -rolling_std, "BEAR", "SIDEWAYS")
            )

        # Analyze by regime
        regime_results = {}
        for regime in ["BULL", "BEAR", "SIDEWAYS"]:
            mask = regime_labels == regime
            if mask.sum() > 0:
                regime_trades = [t for t, m in zip(trades, mask) if m]
                if regime_trades:
                    returns = [t.return_pct for t in regime_trades]
                    pnl = [t.realized_pnl for t in regime_trades]
                    regime_results[regime] = {
                        "n_trades": len(regime_trades),
                        "win_rate": np.mean([r > 0 for r in returns]),
                        "avg_return": np.mean(returns),
                        "total_pnl": sum(pnl),
                        "sharpe": np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252),
                    }

        return regime_results


# =============================================================================
# 6. INTEGRATION
# =============================================================================

def run_full_backtest(
    daily_data: pd.DataFrame,
    swing_predictions: pd.Series,
    timing_predictions: pd.Series,
    entry_exit_predictions: Optional[pd.DataFrame] = None,
    config: Optional[Dict] = None,
    run_monte_carlo: bool = True,
) -> Dict:
    """
    Convenience function to run full backtest with Monte Carlo.
    """
    # Run main backtest
    engine = BacktestEngine(
        initial_capital=config.get("initial_capital", 100000) if config else 100000,
        max_position_pct=config.get("max_position_pct", 0.25) if config else 0.25,
    )

    results = engine.run_backtest(
        daily_data=daily_data,
        swing_predictions=swing_predictions,
        timing_predictions=timing_predictions,
        entry_exit_predictions=entry_exit_predictions,
        config=config,
    )

    # Run Monte Carlo if requested
    if run_monte_carlo and engine.portfolio.closed_trades:
        mc = MonteCarloSimulator(n_simulations=1000)
        mc_results = mc.run_bootstrap(engine.portfolio.closed_trades)
        results["monte_carlo"] = mc_results

        regime_results = mc.run_regime_stress(engine.portfolio.closed_trades)
        results["regime_analysis"] = regime_results

    return results


# =============================================================================
# MAIN - Testing
# =============================================================================

if __name__ == "__main__":
    print("Backtest Engine")
    print("=" * 60)

    # Create synthetic test data
    np.random.seed(42)
    n_days = 500

    dates = pd.date_range("2022-01-01", periods=n_days, freq="B")
    daily_data = pd.DataFrame({
        "open": 450 + np.cumsum(np.random.randn(n_days) * 2),
        "high": 0,
        "low": 0,
        "close": 0,
        "volume": np.random.randint(50000000, 100000000, n_days),
    }, index=dates)

    daily_data["high"] = daily_data["open"] + abs(np.random.randn(n_days)) * 3
    daily_data["low"] = daily_data["open"] - abs(np.random.randn(n_days)) * 3
    daily_data["close"] = daily_data["open"] + np.random.randn(n_days) * 1.5

    # Synthetic predictions (with some skill)
    true_direction = (daily_data["close"] > daily_data["open"]).astype(int)
    noise = np.random.randn(n_days) * 0.2

    swing_predictions = pd.Series(
        np.clip(true_direction * 0.6 + 0.2 + noise, 0, 1),
        index=dates
    )
    timing_predictions = pd.Series(
        np.clip(np.random.rand(n_days) * 0.4 + 0.4, 0, 1),
        index=dates
    )

    print(f"\nTest data: {n_days} days")
    print(f"Date range: {dates[0]} to {dates[-1]}")

    # Run backtest
    print("\n" + "=" * 60)
    print("Running Backtest...")
    print("=" * 60)

    config = {
        "min_swing_confidence": 0.55,
        "min_timing_confidence": 0.50,
        "base_position_pct": 0.10,
        "stop_loss_pct": 0.01,
        "take_profit_pct": 0.015,
        "long_only": True,
    }

    results = run_full_backtest(
        daily_data=daily_data,
        swing_predictions=swing_predictions,
        timing_predictions=timing_predictions,
        config=config,
        run_monte_carlo=True,
    )

    # Print results
    print("\n" + "=" * 60)
    print("BACKTEST RESULTS")
    print("=" * 60)

    print(f"\nPerformance:")
    print(f"  Total Return: {results['total_return_pct']:.2f}%")
    print(f"  Sharpe Ratio: {results['sharpe_ratio']:.3f}")
    print(f"  Sortino Ratio: {results['sortino_ratio']:.3f}")
    print(f"  Max Drawdown: {results['max_drawdown_pct']:.2f}%")

    print(f"\nTrade Statistics:")
    print(f"  Total Trades: {results['n_trades']}")
    print(f"  Win Rate: {results['win_rate']*100:.1f}%")
    print(f"  Profit Factor: {results['profit_factor']:.2f}")
    print(f"  Exit Reasons: {results['exit_reasons']}")

    if "monte_carlo" in results:
        mc = results["monte_carlo"]
        print(f"\nMonte Carlo Analysis (95% CI):")
        print(f"  Median Final Equity: ${mc['median_equity']:,.2f}")
        print(f"  Equity Range: ${mc['equity_ci_lower']:,.2f} - ${mc['equity_ci_upper']:,.2f}")
        print(f"  Probability of Profit: {mc['prob_profit']*100:.1f}%")
        print(f"  Worst Case (5th pct): ${mc['worst_case_equity']:,.2f}")

    if "regime_analysis" in results:
        print(f"\nRegime Analysis:")
        for regime, stats in results["regime_analysis"].items():
            print(f"  {regime}: {stats['n_trades']} trades, "
                  f"Win rate: {stats['win_rate']*100:.1f}%, "
                  f"Avg return: {stats['avg_return']*100:.2f}%")

    print("\n" + "=" * 60)
    print("Backtest Engine loaded successfully!")
    print("=" * 60)
