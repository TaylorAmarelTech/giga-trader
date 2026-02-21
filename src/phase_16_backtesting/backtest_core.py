"""
GIGA TRADER - Backtest Engine Core
====================================
Main backtest engine for testing trading strategies on historical data.

Contains:
- BacktestEngine class
- run_full_backtest() convenience function
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

from src.phase_16_backtesting.portfolio import Trade, Portfolio


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

            # Estimate recent volatility for dynamic slippage
            idx = common_dates.get_loc(date)
            if idx >= 20:
                recent_returns = daily_data['close'].iloc[idx-20:idx].pct_change().dropna()
                volatility = float(recent_returns.std() * np.sqrt(252)) if len(recent_returns) > 1 else 0.15
            else:
                volatility = 0.15  # default ~15% annualized

            # Check for stop loss / take profit hits on open positions
            for trade in list(self.portfolio.open_trades):
                should_close, close_reason, close_price_used = self._check_exit_conditions(
                    trade, high_price, low_price, close_price
                )
                if should_close:
                    self.portfolio.close_trade(trade, date, close_price_used, close_reason, volatility=volatility)

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
                    volatility=volatility,
                )

                if trade:
                    pass  # Trade opened successfully

            # Close all positions at end of day (no overnight)
            for trade in list(self.portfolio.open_trades):
                self.portfolio.close_trade(trade, date, close_price, "eod", volatility=volatility)

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
            risk_free_daily = 0.04 / 252  # ~4% annual risk-free rate
            excess_returns = np.mean(daily_returns) - risk_free_daily
            sharpe_ratio = np.sqrt(252) * excess_returns / (np.std(daily_returns) + 1e-8)

            # Sortino (downside deviation)
            downside_returns = daily_returns[daily_returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
            sortino_ratio = np.sqrt(252) * excess_returns / downside_std
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
    from src.phase_16_backtesting.backtest_variants import MonteCarloSimulator

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
