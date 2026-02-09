"""
GIGA TRADER - Backtest Variants
=================================
Walk-forward backtesting and Monte Carlo stress testing.

Contains:
- WalkForwardBacktest class
- MonteCarloSimulator class
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.phase_16_backtesting.portfolio import Trade, Portfolio
from src.phase_16_backtesting.backtest_core import BacktestEngine


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

        # Ensure datetime index
        if not isinstance(daily_data.index, pd.DatetimeIndex):
            if 'date' in daily_data.columns:
                daily_data = daily_data.set_index('date')
            elif 'Date' in daily_data.columns:
                daily_data = daily_data.set_index('Date')
            # Convert index to datetime if not already
            if not isinstance(daily_data.index, pd.DatetimeIndex):
                try:
                    daily_data.index = pd.to_datetime(daily_data.index)
                except (ValueError, TypeError):
                    return {"error": "Could not convert index to datetime"}

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
