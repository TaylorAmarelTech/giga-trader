"""
GIGA TRADER - Comprehensive Backtesting Harness
=================================================
Runs detailed backtests with robustness measurements, regime analysis,
and walk-forward validation during non-market hours.

This harness ensures the system is NEVER idle and constantly learning.

Features:
  - Walk-forward backtesting with retraining
  - Regime-specific analysis (bull/bear/sideways/high-vol)
  - Robustness testing (parameter perturbation)
  - Monte Carlo stress testing
  - Cross-validation fold analysis
  - Performance degradation detection
  - Automatic report generation

Usage:
    from src.backtesting_harness import BacktestingHarness

    harness = BacktestingHarness()
    results = harness.run_comprehensive_backtest()
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field, asdict
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from collections import defaultdict

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger("BACKTEST_HARNESS")


# =============================================================================
# REGIME DEFINITIONS
# =============================================================================
@dataclass
class MarketRegime:
    """Market regime classification."""
    name: str
    description: str
    start_date: datetime
    end_date: datetime
    spy_return: float  # Total SPY return during regime
    volatility: float  # Average volatility
    trend: str  # "bullish", "bearish", "sideways"


# Historical regimes for testing
HISTORICAL_REGIMES = [
    MarketRegime("COVID Crash", "March 2020 crash", datetime(2020, 2, 19), datetime(2020, 3, 23), -0.34, 0.80, "bearish"),
    MarketRegime("COVID Recovery", "V-shaped recovery", datetime(2020, 3, 24), datetime(2020, 8, 31), 0.55, 0.30, "bullish"),
    MarketRegime("2021 Bull", "Low vol bull run", datetime(2021, 1, 1), datetime(2021, 12, 31), 0.27, 0.15, "bullish"),
    MarketRegime("2022 Bear", "Rate hike bear market", datetime(2022, 1, 1), datetime(2022, 10, 12), -0.25, 0.25, "bearish"),
    MarketRegime("2022-23 Recovery", "Gradual recovery", datetime(2022, 10, 13), datetime(2023, 7, 31), 0.28, 0.18, "bullish"),
    MarketRegime("2023 Sideways", "Consolidation", datetime(2023, 8, 1), datetime(2023, 10, 31), -0.08, 0.15, "sideways"),
    MarketRegime("2023-24 Rally", "AI-driven rally", datetime(2023, 11, 1), datetime(2024, 3, 31), 0.25, 0.12, "bullish"),
]


# =============================================================================
# BACKTEST RESULT
# =============================================================================
@dataclass
class BacktestResult:
    """Comprehensive backtest result."""
    run_id: str
    run_date: datetime
    config_name: str

    # Period info
    start_date: datetime
    end_date: datetime
    trading_days: int

    # Performance metrics
    total_return: float
    annualized_return: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    calmar_ratio: float

    # Trading metrics
    total_trades: int
    win_rate: float
    profit_factor: float
    avg_trade_return: float
    avg_win: float
    avg_loss: float
    max_consecutive_wins: int
    max_consecutive_losses: int

    # Risk metrics
    volatility: float
    var_95: float  # 95% Value at Risk
    cvar_95: float  # Conditional VaR

    # Regime performance
    regime_results: Dict[str, Dict] = field(default_factory=dict)

    # Robustness metrics
    robustness_score: float = 0.0
    parameter_stability: float = 0.0

    # Metadata
    model_version: str = ""
    notes: str = ""


# =============================================================================
# BACKTESTING HARNESS
# =============================================================================
class BacktestingHarness:
    """
    Comprehensive backtesting harness for non-market hours.

    Runs a variety of backtests to ensure model robustness:
    1. Full historical backtest
    2. Walk-forward validation
    3. Regime-specific backtests
    4. Robustness testing (parameter perturbation)
    5. Monte Carlo stress testing
    """

    def __init__(
        self,
        initial_capital: float = 100000,
        output_dir: Optional[Path] = None,
    ):
        self.initial_capital = initial_capital
        self.output_dir = output_dir or (project_root / "reports" / "backtests")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.results: List[BacktestResult] = []
        self._models_loaded = False
        self._data_loaded = False

        # Load components
        self.swing_model = None
        self.timing_model = None
        self.scaler = None
        self.dim_state = None
        self.feature_cols = None

        self.daily_data = None
        self.intraday_data = None

    def load_models(self) -> bool:
        """Load trained models from disk."""
        try:
            import joblib

            model_dir = project_root / "models" / "production"
            model_files = list(model_dir.glob("spy_robust_models*.joblib"))

            if not model_files:
                logger.warning("No model files found")
                return False

            # Load most recent
            model_file = max(model_files, key=lambda f: f.stat().st_mtime)
            logger.info(f"Loading models from {model_file.name}")

            state = joblib.load(model_file)

            # Handle both flat and nested model structures
            if "models" in state:
                # Nested structure: state['models']['swing']['l2']
                models = state.get("models", {})
                self.swing_model = models.get("swing", {}).get("l2") or models.get("swing", {}).get("gb")
                self.timing_model = models.get("timing", {}).get("l2") or models.get("timing", {}).get("gb")
            else:
                # Flat structure: state['swing_model']
                self.swing_model = state.get("swing_model") or state.get("swing_l2")
                self.timing_model = state.get("timing_model") or state.get("timing_l2")

            self.scaler = state.get("scaler")
            self.dim_state = state.get("dim_reduction_state") or state.get("dim_state")
            self.feature_cols = state.get("feature_cols") or state.get("feature_columns", [])

            self._models_loaded = True
            logger.info(f"Models loaded: swing={self.swing_model is not None}, timing={self.timing_model is not None}")

            return True

        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            return False

    def load_data(self, lookback_years: int = 5) -> bool:
        """Load historical data for backtesting."""
        try:
            from src.data_manager import DataManager

            dm = DataManager()

            logger.info(f"Loading {lookback_years} years of data...")

            # Load data (DataManager uses years parameter)
            self.daily_data = dm.get_data(
                symbol="SPY",
                years=lookback_years,
            )

            # Intraday data - reuse daily data for now
            # (DataManager caches 1-min data internally)
            self.intraday_data = self.daily_data.copy()

            self._data_loaded = True
            logger.info(f"Data loaded: {len(self.daily_data)} bars")

            return True

        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return False

    def run_comprehensive_backtest(
        self,
        include_walk_forward: bool = True,
        include_regime_analysis: bool = True,
        include_robustness: bool = True,
        include_monte_carlo: bool = True,
        n_monte_carlo_runs: int = 100,
    ) -> Dict:
        """
        Run comprehensive backtesting suite.

        Returns summary of all backtests run.
        """
        logger.info("=" * 70)
        logger.info("COMPREHENSIVE BACKTESTING HARNESS")
        logger.info("=" * 70)

        start_time = datetime.now()

        # Ensure data and models are loaded
        if not self._models_loaded:
            if not self.load_models():
                return {"error": "Failed to load models"}

        if not self._data_loaded:
            if not self.load_data():
                return {"error": "Failed to load data"}

        summary = {
            "run_date": start_time.isoformat(),
            "tests_run": [],
            "results": {},
        }

        # 1. Full Historical Backtest
        logger.info("\n[1/5] Running full historical backtest...")
        try:
            full_result = self._run_full_backtest()
            summary["results"]["full_historical"] = asdict(full_result)
            summary["tests_run"].append("full_historical")
            logger.info(f"  Sharpe: {full_result.sharpe_ratio:.2f}, Return: {full_result.total_return:.1%}")
        except Exception as e:
            logger.error(f"  Full backtest failed: {e}")
            summary["results"]["full_historical"] = {"error": str(e)}

        # 2. Walk-Forward Analysis
        if include_walk_forward:
            logger.info("\n[2/5] Running walk-forward analysis...")
            try:
                wf_results = self._run_walk_forward()
                summary["results"]["walk_forward"] = wf_results
                summary["tests_run"].append("walk_forward")
                logger.info(f"  {len(wf_results.get('windows', []))} windows analyzed")
            except Exception as e:
                logger.error(f"  Walk-forward failed: {e}")
                summary["results"]["walk_forward"] = {"error": str(e)}

        # 3. Regime-Specific Analysis
        if include_regime_analysis:
            logger.info("\n[3/5] Running regime-specific analysis...")
            try:
                regime_results = self._run_regime_analysis()
                summary["results"]["regime_analysis"] = regime_results
                summary["tests_run"].append("regime_analysis")
                logger.info(f"  {len(regime_results)} regimes analyzed")
            except Exception as e:
                logger.error(f"  Regime analysis failed: {e}")
                summary["results"]["regime_analysis"] = {"error": str(e)}

        # 4. Robustness Testing
        if include_robustness:
            logger.info("\n[4/5] Running robustness testing...")
            try:
                robustness_results = self._run_robustness_tests()
                summary["results"]["robustness"] = robustness_results
                summary["tests_run"].append("robustness")
                logger.info(f"  Robustness score: {robustness_results.get('overall_score', 0):.2f}")
            except Exception as e:
                logger.error(f"  Robustness testing failed: {e}")
                summary["results"]["robustness"] = {"error": str(e)}

        # 5. Monte Carlo Stress Testing
        if include_monte_carlo:
            logger.info("\n[5/5] Running Monte Carlo stress testing...")
            try:
                mc_results = self._run_monte_carlo(n_runs=n_monte_carlo_runs)
                summary["results"]["monte_carlo"] = mc_results
                summary["tests_run"].append("monte_carlo")
                logger.info(f"  Median Sharpe: {mc_results.get('median_sharpe', 0):.2f}")
            except Exception as e:
                logger.error(f"  Monte Carlo failed: {e}")
                summary["results"]["monte_carlo"] = {"error": str(e)}

        # Calculate overall score
        summary["overall_robustness_score"] = self._calculate_overall_score(summary)

        # Save results
        elapsed = (datetime.now() - start_time).total_seconds()
        summary["elapsed_seconds"] = elapsed

        self._save_results(summary)

        logger.info("\n" + "=" * 70)
        logger.info(f"BACKTESTING COMPLETE in {elapsed:.1f}s")
        logger.info(f"Overall Robustness Score: {summary['overall_robustness_score']:.2f}")
        logger.info("=" * 70)

        return summary

    def _run_full_backtest(self) -> BacktestResult:
        """Run backtest on full historical data."""
        from src.backtest_engine import BacktestEngine
        from sklearn.linear_model import LogisticRegression
        import numpy as np

        # Prepare features
        df = self._prepare_features(self.daily_data)

        if df is None or len(df) < 100:
            raise ValueError("Insufficient data for backtest")

        # Check if expected features are available
        available = [c for c in self.feature_cols if c in df.columns]

        if len(available) >= len(self.feature_cols) * 0.8:
            # Most features available - use saved models
            X = df[available].values
            # Fill missing columns with zeros
            if len(available) < len(self.feature_cols):
                missing_count = len(self.feature_cols) - len(available)
                X = np.hstack([X, np.zeros((len(X), missing_count))])
            swing_proba = self.swing_model.predict_proba(X)[:, 1]
            timing_proba = self.timing_model.predict_proba(X)[:, 1]
            logger.info(f"Using saved models with {len(available)}/{len(self.feature_cols)} features")
        else:
            # Not enough features - train simple fallback models on available data
            logger.warning(f"Only {len(available)}/{len(self.feature_cols)} features available, using fallback models")

            # Get numeric features for fallback
            exclude_cols = ['target', 'date', 'Date', 'open', 'high', 'low', 'close', 'volume',
                           'Open', 'High', 'Low', 'Close', 'Volume', 'day_return', 'future_return']
            fallback_cols = [c for c in df.columns if c not in exclude_cols
                            and df[c].dtype in ['float64', 'int64', 'float32', 'int32']
                            and not df[c].isna().all()][:50]  # Limit to 50 features

            if len(fallback_cols) < 10:
                raise ValueError(f"Insufficient features for fallback: {len(fallback_cols)}")

            X_raw = df[fallback_cols].values
            X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

            # Create simple targets: next day up/down
            if 'day_return' in df.columns:
                y_swing = (df['day_return'].shift(-1) > 0).astype(int).fillna(0).values
            else:
                # Calculate from close prices
                df['_next_return'] = df['close'].pct_change().shift(-1)
                y_swing = (df['_next_return'] > 0).astype(int).fillna(0).values

            # Simple timing target (random baseline)
            y_timing = np.random.randint(0, 2, len(df))

            # Train simple fallback models
            fallback_swing = LogisticRegression(max_iter=500, C=0.1, random_state=42)
            fallback_timing = LogisticRegression(max_iter=500, C=0.1, random_state=42)

            # Use first 80% for training
            split_idx = int(len(X_raw) * 0.8)
            fallback_swing.fit(X_raw[:split_idx], y_swing[:split_idx])
            fallback_timing.fit(X_raw[:split_idx], y_timing[:split_idx])

            swing_proba = fallback_swing.predict_proba(X_raw)[:, 1]
            timing_proba = fallback_timing.predict_proba(X_raw)[:, 1]
            logger.info(f"Trained fallback models on {len(fallback_cols)} features")

        swing_predictions = pd.Series(swing_proba, index=df.index)
        timing_predictions = pd.Series(timing_proba, index=df.index)

        # Run backtest
        engine = BacktestEngine(initial_capital=self.initial_capital)
        raw_results = engine.run_backtest(
            daily_data=df,
            swing_predictions=swing_predictions,
            timing_predictions=timing_predictions,
        )

        # Convert to BacktestResult
        return BacktestResult(
            run_id=f"full_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            run_date=datetime.now(),
            config_name="full_historical",
            start_date=df.index.min(),
            end_date=df.index.max(),
            trading_days=len(df),
            total_return=raw_results.get("total_return", 0),
            annualized_return=raw_results.get("annualized_return", 0),
            sharpe_ratio=raw_results.get("sharpe_ratio", 0),
            sortino_ratio=raw_results.get("sortino_ratio", 0),
            max_drawdown=raw_results.get("max_drawdown", 0),
            calmar_ratio=raw_results.get("calmar_ratio", 0),
            total_trades=raw_results.get("total_trades", 0),
            win_rate=raw_results.get("win_rate", 0),
            profit_factor=raw_results.get("profit_factor", 0),
            avg_trade_return=raw_results.get("avg_trade_return", 0),
            avg_win=raw_results.get("avg_win", 0),
            avg_loss=raw_results.get("avg_loss", 0),
            max_consecutive_wins=raw_results.get("max_consecutive_wins", 0),
            max_consecutive_losses=raw_results.get("max_consecutive_losses", 0),
            volatility=raw_results.get("volatility", 0),
            var_95=raw_results.get("var_95", 0),
            cvar_95=raw_results.get("cvar_95", 0),
        )

    def _run_walk_forward(self) -> Dict:
        """Run walk-forward validation."""
        from src.backtest_engine import WalkForwardBacktest

        wf = WalkForwardBacktest(
            train_months=24,
            test_months=3,
            step_months=3,
            anchored=True,
        )

        # Prepare features
        df = self._prepare_features(self.daily_data)

        if df is None:
            return {"error": "Failed to prepare features"}

        # Define model trainer function
        def train_models(train_data, feature_cols):
            from sklearn.linear_model import LogisticRegression

            X = train_data[feature_cols].values
            y_swing = (train_data["day_return"] > 0.003).astype(int)
            y_timing = train_data.get("low_before_high", pd.Series(1, index=train_data.index)).astype(int)

            swing_model = LogisticRegression(C=1.0, max_iter=1000)
            swing_model.fit(X, y_swing)

            timing_model = LogisticRegression(C=1.0, max_iter=1000)
            timing_model.fit(X, y_timing)

            return swing_model, timing_model

        # Run walk-forward
        results = wf.run_walk_forward(
            daily_data=df,
            intraday_data=self.intraday_data,
            feature_cols=self.feature_cols,
            model_trainer_fn=train_models,
        )

        return results

    def _run_regime_analysis(self) -> Dict:
        """Run backtest on different market regimes."""
        regime_results = {}

        df = self._prepare_features(self.daily_data)
        if df is None:
            return {"error": "Failed to prepare features"}

        # Ensure datetime index for regime filtering
        if not isinstance(df.index, pd.DatetimeIndex):
            if 'date' in df.columns:
                df = df.set_index('date')
            elif 'Date' in df.columns:
                df = df.set_index('Date')
            # Convert index to datetime if it's not already
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except (ValueError, TypeError):
                    return {"error": "Could not convert index to datetime"}

        for regime in HISTORICAL_REGIMES:
            try:
                # Filter data to regime period
                mask = (df.index >= regime.start_date) & (df.index <= regime.end_date)
                regime_data = df[mask]

                if len(regime_data) < 20:
                    regime_results[regime.name] = {"skipped": "insufficient_data"}
                    continue

                # Generate predictions
                X = regime_data[self.feature_cols].values
                swing_proba = self.swing_model.predict_proba(X)[:, 1]
                timing_proba = self.timing_model.predict_proba(X)[:, 1]

                # Calculate metrics
                predictions = swing_proba > 0.5
                actual = (regime_data["day_return"] > 0.003).values

                accuracy = (predictions == actual).mean()

                # Simulated returns
                returns = []
                for i, (pred, ret) in enumerate(zip(predictions, regime_data["day_return"].values)):
                    if pred:  # Predicted up
                        returns.append(ret)
                    else:  # Predicted down or neutral
                        returns.append(-ret * 0.5)  # Partial benefit from correct sell

                total_return = np.sum(returns)
                sharpe = np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252)

                regime_results[regime.name] = {
                    "trend": regime.trend,
                    "spy_return": regime.spy_return,
                    "accuracy": accuracy,
                    "strategy_return": total_return,
                    "sharpe": sharpe,
                    "trading_days": len(regime_data),
                    "outperformed_spy": total_return > regime.spy_return,
                }

            except Exception as e:
                regime_results[regime.name] = {"error": str(e)}

        return regime_results

    def _run_robustness_tests(self) -> Dict:
        """Test model robustness with parameter perturbation."""
        results = {
            "parameter_perturbations": [],
            "noise_tolerance": [],
            "feature_subset_stability": [],
        }

        df = self._prepare_features(self.daily_data)
        if df is None:
            return {"error": "Failed to prepare features"}

        # Get baseline performance
        X = df[self.feature_cols].values
        y = (df["day_return"] > 0.003).astype(int).values

        baseline_proba = self.swing_model.predict_proba(X)[:, 1]
        baseline_pred = baseline_proba > 0.5
        baseline_acc = (baseline_pred == y).mean()

        # 1. Add noise to features
        for noise_level in [0.01, 0.05, 0.10]:
            X_noisy = X + np.random.normal(0, noise_level, X.shape)
            noisy_proba = self.swing_model.predict_proba(X_noisy)[:, 1]
            noisy_pred = noisy_proba > 0.5
            noisy_acc = (noisy_pred == y).mean()

            results["noise_tolerance"].append({
                "noise_level": noise_level,
                "accuracy": noisy_acc,
                "accuracy_drop": baseline_acc - noisy_acc,
            })

        # 2. Feature subset stability
        n_features = len(self.feature_cols)
        for drop_pct in [0.1, 0.2, 0.3]:
            n_drop = int(n_features * drop_pct)

            # Random feature subsets
            subset_accs = []
            for _ in range(5):
                keep_idx = np.random.choice(n_features, n_features - n_drop, replace=False)
                X_subset = X[:, keep_idx]

                # Retrain simple model
                from sklearn.linear_model import LogisticRegression
                subset_model = LogisticRegression(C=1.0, max_iter=1000)
                subset_model.fit(X_subset, y)
                subset_pred = subset_model.predict(X_subset)
                subset_accs.append((subset_pred == y).mean())

            results["feature_subset_stability"].append({
                "features_dropped_pct": drop_pct,
                "mean_accuracy": np.mean(subset_accs),
                "std_accuracy": np.std(subset_accs),
            })

        # Calculate overall robustness score
        noise_score = 1 - np.mean([r["accuracy_drop"] for r in results["noise_tolerance"]])
        subset_score = np.mean([r["mean_accuracy"] for r in results["feature_subset_stability"]]) / baseline_acc

        results["overall_score"] = 0.5 * noise_score + 0.5 * subset_score
        results["baseline_accuracy"] = baseline_acc

        return results

    def _run_monte_carlo(self, n_runs: int = 100) -> Dict:
        """Run Monte Carlo stress testing."""
        df = self._prepare_features(self.daily_data)
        if df is None:
            return {"error": "Failed to prepare features"}

        X = df[self.feature_cols].values
        returns = df["day_return"].values

        baseline_proba = self.swing_model.predict_proba(X)[:, 1]

        # Run simulations with random order shuffling
        sharpes = []
        total_returns = []
        max_drawdowns = []

        for _ in range(n_runs):
            # Bootstrap sample with replacement
            indices = np.random.choice(len(returns), len(returns), replace=True)

            sim_returns = []
            equity = self.initial_capital
            peak = equity
            max_dd = 0

            for i in indices:
                pred = baseline_proba[i] > 0.5
                ret = returns[i]

                if pred:
                    pnl = ret * 0.25  # 25% position
                else:
                    pnl = 0

                equity *= (1 + pnl)
                sim_returns.append(pnl)

                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd

            sim_returns = np.array(sim_returns)
            sharpe = np.mean(sim_returns) / (np.std(sim_returns) + 1e-6) * np.sqrt(252)
            total_ret = equity / self.initial_capital - 1

            sharpes.append(sharpe)
            total_returns.append(total_ret)
            max_drawdowns.append(max_dd)

        return {
            "n_simulations": n_runs,
            "median_sharpe": np.median(sharpes),
            "mean_sharpe": np.mean(sharpes),
            "sharpe_5th_percentile": np.percentile(sharpes, 5),
            "sharpe_95th_percentile": np.percentile(sharpes, 95),
            "median_return": np.median(total_returns),
            "mean_return": np.mean(total_returns),
            "median_max_drawdown": np.median(max_drawdowns),
            "worst_case_drawdown": np.percentile(max_drawdowns, 95),
            "probability_profitable": (np.array(total_returns) > 0).mean(),
        }

    def _prepare_features(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepare features for backtesting, including dimensionality reduction."""
        try:
            from src.train_robust_model import engineer_all_features, reduce_dimensions
            import numpy as np

            df_with_features = engineer_all_features(df.copy())

            # Check if we have dim_state for transformation
            if self.dim_state is not None:
                # Get raw feature columns (exclude target, date, etc.)
                exclude_cols = ['target', 'date', 'Date', 'open', 'high', 'low', 'close', 'volume',
                               'Open', 'High', 'Low', 'Close', 'Volume', 'day_return', 'future_return',
                               'target_up', 'target_timing', 'is_synthetic', 'sample_weight']
                raw_feature_cols = [c for c in df_with_features.columns if c not in exclude_cols
                                   and not c.startswith(('kpca_', 'mi_', 'ica_', 'medoid_', 'smoothed_'))]

                if len(raw_feature_cols) > 0:
                    X_raw = df_with_features[raw_feature_cols].values

                    # Handle NaN/Inf
                    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

                    # Check if feature count matches what dim_state expects
                    expected_features = None
                    if "var_selector" in self.dim_state:
                        expected_features = self.dim_state["var_selector"].n_features_in_

                    if expected_features is not None and X_raw.shape[1] != expected_features:
                        # Feature mismatch - skip dim reduction and use raw features
                        logger.warning(f"Feature mismatch: have {X_raw.shape[1]}, expected {expected_features}")
                        logger.warning("  Skipping dim reduction - will use fallback models")
                        # Set feature_cols to raw features for fallback
                        self.feature_cols = raw_feature_cols
                    else:
                        # Apply dimensionality reduction using saved state
                        try:
                            X_reduced, reduced_names, _ = reduce_dimensions(
                                X_raw, raw_feature_cols, y=None, fit=False, state=self.dim_state
                            )

                            # Add reduced features to dataframe
                            for i, name in enumerate(reduced_names):
                                df_with_features[name] = X_reduced[:, i]

                            # Update feature_cols to reduced features
                            self.feature_cols = reduced_names
                            logger.info(f"Applied dim reduction: {len(raw_feature_cols)} -> {len(reduced_names)} features")
                        except Exception as dim_error:
                            logger.warning(f"Dim reduction failed, using raw features: {dim_error}")
                            self.feature_cols = raw_feature_cols
            else:
                # No dim_state - use raw features
                exclude_cols = ['target', 'date', 'Date', 'open', 'high', 'low', 'close', 'volume',
                               'Open', 'High', 'Low', 'Close', 'Volume', 'day_return', 'future_return']
                self.feature_cols = [c for c in df_with_features.columns if c not in exclude_cols
                                    and df_with_features[c].dtype in ['float64', 'int64', 'float32', 'int32']][:50]

            # Filter to available features (check both raw and reduced)
            available = [c for c in self.feature_cols if c in df_with_features.columns]

            if len(available) < len(self.feature_cols) * 0.5:
                logger.warning(f"Only {len(available)}/{len(self.feature_cols)} features available")
                # Use whatever features are available
                self.feature_cols = available

            return df_with_features

        except Exception as e:
            logger.error(f"Feature preparation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

    def _calculate_overall_score(self, summary: Dict) -> float:
        """Calculate overall robustness score from all tests."""
        scores = []

        # Full backtest contribution
        if "full_historical" in summary["results"]:
            full = summary["results"]["full_historical"]
            if not isinstance(full, dict) or "error" not in full:
                sharpe = full.get("sharpe_ratio", 0)
                scores.append(min(sharpe / 2, 1.0))  # Normalize to 0-1

        # Robustness contribution
        if "robustness" in summary["results"]:
            rob = summary["results"]["robustness"]
            if isinstance(rob, dict) and "overall_score" in rob:
                scores.append(rob["overall_score"])

        # Monte Carlo contribution
        if "monte_carlo" in summary["results"]:
            mc = summary["results"]["monte_carlo"]
            if isinstance(mc, dict) and "probability_profitable" in mc:
                scores.append(mc["probability_profitable"])

        # Regime consistency
        if "regime_analysis" in summary["results"]:
            regimes = summary["results"]["regime_analysis"]
            if isinstance(regimes, dict):
                outperformed = [r for r in regimes.values()
                               if isinstance(r, dict) and r.get("outperformed_spy", False)]
                if regimes:
                    scores.append(len(outperformed) / len(regimes))

        return np.mean(scores) if scores else 0.0

    def _save_results(self, summary: Dict):
        """Save backtest results to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"backtest_results_{timestamp}.json"

        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)

        logger.info(f"Results saved to {output_file}")

    def run_quick_validation(self) -> Dict:
        """
        Run a quick validation backtest for the orchestrator.

        This is faster than the comprehensive backtest but still meaningful.
        """
        logger.info("Running quick validation backtest...")

        if not self._models_loaded:
            if not self.load_models():
                return {"error": "Failed to load models"}

        if not self._data_loaded:
            if not self.load_data(lookback_years=1):  # Just 1 year
                return {"error": "Failed to load data"}

        try:
            # Quick full backtest on recent data
            result = self._run_full_backtest()

            return {
                "sharpe_ratio": result.sharpe_ratio,
                "total_return": result.total_return,
                "win_rate": result.win_rate,
                "max_drawdown": result.max_drawdown,
                "trading_days": result.trading_days,
                "status": "success",
            }

        except Exception as e:
            logger.error(f"Quick validation failed: {e}")
            return {"error": str(e)}


# =============================================================================
# MAIN (for testing)
# =============================================================================
def main():
    """Run comprehensive backtesting."""
    harness = BacktestingHarness()
    results = harness.run_comprehensive_backtest()

    print("\n" + "=" * 70)
    print("FINAL RESULTS")
    print("=" * 70)
    print(json.dumps(results, indent=2, default=str))

    return 0


if __name__ == "__main__":
    sys.exit(main())
