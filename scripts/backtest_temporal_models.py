"""
GIGA TRADER - Multi-Timeframe Temporal Model Backtest
======================================================
Runs a backtest using temporal cascade models with proper intraday data.

This script:
  1. Loads temporal models from models/temporal_v2/
  2. Loads minute-level intraday data
  3. For each test day:
     - Generates temporal slice features (T0, T30, T60, T90, T120, T180)
     - Gets predictions from all cascade models
     - Simulates trades with intraday price data
  4. Calculates multi-timeframe performance metrics

TEMPORAL SLICES:
  T0:   Pre-market features only (9:30 AM)
  T30:  + First 30 minutes (10:00 AM)
  T60:  + First 60 minutes (10:30 AM)
  T90:  + First 90 minutes (11:00 AM)
  T120: + First 2 hours (11:30 AM)
  T180: + First 3 hours (12:30 PM)
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime, time, timedelta
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import joblib

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            project_root / "logs" / f"backtest_temporal_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )
    ]
)
logger = logging.getLogger("BACKTEST_TEMPORAL")


# Temporal slices (minutes from market open)
TEMPORAL_SLICES = [0, 30, 60, 90, 120, 180]


@dataclass
class BacktestConfig:
    """Backtest configuration."""
    initial_capital: float = 100000.0
    position_size_pct: float = 0.10  # 10% per trade
    max_position_pct: float = 0.25   # Max 25% in one position
    stop_loss_pct: float = 0.02      # 2% stop loss
    take_profit_pct: float = 0.03    # 3% take profit
    min_swing_conf: float = 0.55     # Min swing probability
    min_timing_conf: float = 0.50    # Min timing probability
    commission: float = 0.0          # Commission per trade
    slippage_pct: float = 0.0005     # 0.05% slippage
    entry_time_minutes: int = 60     # Enter at T60 (10:30 AM)
    exit_time_minutes: int = 360     # Exit at T360 (3:30 PM) or EOD


@dataclass
class Trade:
    """Individual trade record."""
    entry_date: str
    entry_price: float
    direction: str  # "LONG" or "SHORT"
    position_size: float
    entry_time_slice: int = 0
    exit_date: str = None
    exit_price: float = None
    pnl: float = 0.0
    pnl_pct: float = 0.0
    exit_reason: str = None
    swing_prob: float = 0.0
    timing_prob: float = 0.0
    cascade_agreement: float = 0.0


@dataclass
class BacktestResult:
    """Backtest results."""
    total_return_pct: float = 0.0
    annualized_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    total_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_pnl: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    trades: List[Trade] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    daily_returns: List[float] = field(default_factory=list)


class TemporalFeatureGenerator:
    """Generates features for temporal slices from intraday data."""

    TEMPORAL_SLICES = [0, 30, 60, 90, 120, 180]

    def __init__(self):
        self.historical_scaler = None

    def generate_slice_features(
        self,
        df_1min_day: pd.DataFrame,
        historical_features: np.ndarray,
        minutes_cutoff: int
    ) -> np.ndarray:
        """
        Generate features for a specific temporal slice.

        Args:
            df_1min_day: Minute-level data for the day (regular hours)
            historical_features: Pre-computed historical features
            minutes_cutoff: Minutes from market open (0, 30, 60, etc.)

        Returns:
            Feature vector for this slice
        """
        intraday_features = self._engineer_intraday_features(df_1min_day, minutes_cutoff)

        # Combine historical + intraday features
        combined = np.concatenate([historical_features, intraday_features])
        return combined

    def _engineer_intraday_features(
        self,
        df_1min: pd.DataFrame,
        minutes_cutoff: int
    ) -> np.ndarray:
        """Engineer intraday features up to the cutoff time."""
        features = []

        if minutes_cutoff == 0 or len(df_1min) == 0:
            # Pre-market: return zeros for intraday features
            return np.zeros(20)

        # Filter to only use data up to cutoff
        df = df_1min.head(minutes_cutoff).copy()

        if len(df) < 5:
            return np.zeros(20)

        # Basic price features
        open_price = df['open'].iloc[0]
        current_price = df['close'].iloc[-1]
        high_so_far = df['high'].max()
        low_so_far = df['low'].min()

        features.append((current_price - open_price) / open_price)  # intraday_return
        features.append((high_so_far - low_so_far) / open_price)    # intraday_range
        features.append(current_price / open_price - 1)              # price_vs_open
        features.append(high_so_far / open_price - 1)                # high_vs_open
        features.append(low_so_far / open_price - 1)                 # low_vs_open

        # Price position in range
        intraday_range = high_so_far - low_so_far
        if intraday_range > 0:
            features.append((current_price - low_so_far) / intraday_range)
        else:
            features.append(0.5)

        # Timing features
        high_idx = df['high'].idxmax()
        low_idx = df['low'].idxmin()
        high_minute = df.index.get_loc(high_idx) if high_idx in df.index else 0
        low_minute = df.index.get_loc(low_idx) if low_idx in df.index else 0

        features.append(high_minute / max(minutes_cutoff, 1))  # high_timing
        features.append(low_minute / max(minutes_cutoff, 1))   # low_timing
        features.append(1.0 if low_minute < high_minute else 0.0)  # low_before_high

        # Volume features
        total_volume = df['volume'].sum()
        avg_volume = total_volume / len(df) if len(df) > 0 else 0
        features.append(avg_volume / 1e6)  # normalized volume intensity

        # Momentum (returns at different points)
        if len(df) >= 10:
            features.append(df['close'].iloc[-1] / df['close'].iloc[-10] - 1)  # 10-bar momentum
        else:
            features.append(0)

        if len(df) >= 5:
            features.append(df['close'].iloc[-1] / df['close'].iloc[-5] - 1)   # 5-bar momentum
        else:
            features.append(0)

        # Volatility
        returns = df['close'].pct_change().dropna()
        features.append(returns.std() if len(returns) > 1 else 0)  # intraday volatility

        # VWAP deviation
        vwap = (df['close'] * df['volume']).sum() / df['volume'].sum() if df['volume'].sum() > 0 else df['close'].mean()
        features.append((current_price - vwap) / vwap if vwap > 0 else 0)

        # Bar direction ratio
        up_bars = (df['close'] > df['open']).sum()
        features.append(up_bars / len(df) if len(df) > 0 else 0.5)

        # First vs last bar comparison
        features.append(df['close'].iloc[-1] / df['open'].iloc[0] - 1)

        # Recent momentum (last 5 bars vs first 5)
        if len(df) >= 10:
            first_5_avg = df['close'].iloc[:5].mean()
            last_5_avg = df['close'].iloc[-5:].mean()
            features.append((last_5_avg - first_5_avg) / first_5_avg if first_5_avg > 0 else 0)
        else:
            features.append(0)

        # Pad to 20 features
        while len(features) < 20:
            features.append(0)

        return np.array(features[:20])


class TemporalModelLoader:
    """Loads and manages temporal models."""

    def __init__(self, models_dir: Path = None):
        self.models_dir = models_dir or (project_root / "models" / "temporal_v2")
        self.swing_models: Dict[str, Any] = {}
        self.timing_models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.feature_cols: List[str] = []

    def load_models(self) -> bool:
        """Load all temporal models."""
        logger.info("Loading temporal models...")

        swing_files = sorted(self.models_dir.glob("swing_*.joblib"))
        timing_files = sorted(self.models_dir.glob("timing_*.joblib"))

        if not swing_files or not timing_files:
            logger.error("No temporal models found!")
            return False

        # Load swing models
        for model_file in swing_files:
            model_name = model_file.stem
            try:
                data = joblib.load(model_file)
                if isinstance(data, dict):
                    if 'feature_cols' in data and not self.feature_cols:
                        self.feature_cols = data['feature_cols']
                        logger.info(f"  Loaded feature_cols: {len(self.feature_cols)} features")

                    # Base model (l2/gb)
                    if 'l2' in data and 'gb' in data:
                        self.swing_models[model_name] = {
                            'type': 'base',
                            'data': data
                        }
                        logger.info(f"  Loaded base: {model_name}")

                    # Masked model (TemporalMaskingWrapper)
                    elif 'model' in data:
                        model = data['model']
                        if hasattr(model, 'predict_proba'):
                            self.swing_models[model_name] = {
                                'type': 'masked',
                                'model': model
                            }
                            logger.info(f"  Loaded masked: {model_name}")

                    # Cascade models
                    elif 'cascade' in data:
                        cascade = data['cascade']
                        cascade_type = type(cascade).__name__
                        self.swing_models[model_name] = {
                            'type': 'cascade',
                            'cascade_type': cascade_type,
                            'cascade': cascade
                        }
                        logger.info(f"  Loaded cascade ({cascade_type}): {model_name}")

            except Exception as e:
                logger.warning(f"  Failed to load {model_name}: {e}")

        # Load timing models
        for model_file in timing_files:
            model_name = model_file.stem
            try:
                data = joblib.load(model_file)
                if isinstance(data, dict):
                    if 'feature_cols' in data and not self.feature_cols:
                        self.feature_cols = data['feature_cols']
                        logger.info(f"  Loaded feature_cols: {len(self.feature_cols)} features")

                    if 'l2' in data and 'gb' in data:
                        self.timing_models[model_name] = {
                            'type': 'base',
                            'data': data
                        }
                        logger.info(f"  Loaded base: {model_name}")

                    elif 'model' in data:
                        model = data['model']
                        if hasattr(model, 'predict_proba'):
                            self.timing_models[model_name] = {
                                'type': 'masked',
                                'model': model
                            }
                            logger.info(f"  Loaded masked: {model_name}")

                    elif 'cascade' in data:
                        cascade = data['cascade']
                        cascade_type = type(cascade).__name__
                        self.timing_models[model_name] = {
                            'type': 'cascade',
                            'cascade_type': cascade_type,
                            'cascade': cascade
                        }
                        logger.info(f"  Loaded cascade ({cascade_type}): {model_name}")

            except Exception as e:
                logger.warning(f"  Failed to load {model_name}: {e}")

        logger.info(f"Loaded {len(self.swing_models)} swing models, {len(self.timing_models)} timing models")
        return len(self.swing_models) > 0 and len(self.timing_models) > 0

    def predict_ensemble(
        self,
        X: np.ndarray,
        X_by_slice: Dict[int, np.ndarray],
        model_type: str = "swing"
    ) -> Tuple[float, float, float]:
        """
        Get ensemble prediction from all models.

        Args:
            X: Feature vector for base/masked models
            X_by_slice: Features organized by temporal slice for cascade models
            model_type: "swing" or "timing"

        Returns:
            (mean_probability, uncertainty, cascade_agreement)
        """
        models = self.swing_models if model_type == "swing" else self.timing_models
        predictions = []
        cascade_predictions = []

        for name, model_info in models.items():
            try:
                mtype = model_info['type']

                if mtype == 'base':
                    data = model_info['data']
                    scaler = data.get('scaler')
                    X_input = scaler.transform(X.reshape(1, -1)) if scaler else X.reshape(1, -1)
                    proba_l2 = data['l2'].predict_proba(X_input)[:, 1]
                    proba_gb = data['gb'].predict_proba(X_input)[:, 1]
                    proba = (proba_l2[0] + proba_gb[0]) / 2
                    predictions.append(proba)

                elif mtype == 'masked':
                    model = model_info['model']
                    proba = model.predict_proba(X.reshape(1, -1))[0, 1]
                    predictions.append(proba)

                elif mtype == 'cascade':
                    cascade = model_info['cascade']
                    cascade_type = model_info['cascade_type']

                    if cascade_type == 'IntermittentMaskedCascade':
                        # Takes X directly
                        pred = cascade.predict(X.reshape(1, -1))
                        proba = pred.swing_probability
                        predictions.append(proba)
                        cascade_predictions.append(proba)

                    elif cascade_type in ['CrossTemporalAttentionCascade', 'StochasticDepthCascade']:
                        # Takes X_by_slice
                        if X_by_slice:
                            pred = cascade.predict(X_by_slice)
                            proba = pred.swing_probability
                            predictions.append(proba)
                            cascade_predictions.append(proba)

            except Exception as e:
                # Silently skip failed predictions
                continue

        if not predictions:
            return 0.5, 1.0, 0.0

        mean_proba = np.mean(predictions)
        uncertainty = np.std(predictions) if len(predictions) > 1 else 0.0

        # Cascade agreement (how much cascade models agree)
        if len(cascade_predictions) > 1:
            cascade_agreement = 1 - np.std(cascade_predictions)
        elif len(cascade_predictions) == 1:
            cascade_agreement = 1.0
        else:
            cascade_agreement = 0.5

        return float(mean_proba), float(uncertainty), float(cascade_agreement)


class TemporalBacktester:
    """Multi-timeframe backtester using temporal cascade models."""

    def __init__(
        self,
        config: BacktestConfig = None,
        model_loader: TemporalModelLoader = None
    ):
        self.config = config or BacktestConfig()
        self.model_loader = model_loader or TemporalModelLoader()
        self.feature_generator = TemporalFeatureGenerator()
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.daily_returns: List[float] = []

    def run_backtest(
        self,
        df_features: pd.DataFrame,
        df_1min_dict: Dict[str, pd.DataFrame],
        feature_cols: List[str],
        test_start_idx: int = None,
    ) -> BacktestResult:
        """
        Run multi-timeframe backtest.

        Args:
            df_features: DataFrame with historical features
            df_1min_dict: Dictionary of date -> minute-level DataFrame
            feature_cols: List of feature column names
            test_start_idx: Index to start testing
        """
        logger.info("=" * 70)
        logger.info("MULTI-TIMEFRAME TEMPORAL BACKTEST")
        logger.info("=" * 70)

        if not self.model_loader.swing_models:
            if not self.model_loader.load_models():
                raise RuntimeError("Failed to load models")

        if test_start_idx is None:
            test_start_idx = int(len(df_features) * 0.8)

        test_data = df_features.iloc[test_start_idx:].copy()
        logger.info(f"Test period: {len(test_data)} days")
        logger.info(f"Initial capital: ${self.config.initial_capital:,.2f}")
        logger.info(f"Entry time: T{self.config.entry_time_minutes} ({self.config.entry_time_minutes} min from open)")

        capital = self.config.initial_capital
        self.equity_curve = [capital]
        self.daily_returns = []
        self.trades = []
        current_position: Optional[Trade] = None

        for i in range(len(test_data)):
            row = test_data.iloc[i]
            date_val = row.get('date', test_data.index[i])
            date_str = str(date_val)[:10] if hasattr(date_val, 'strftime') else str(date_val)[:10]

            # Get minute data for this day
            df_1min_day = df_1min_dict.get(date_str)
            if df_1min_day is None or len(df_1min_day) < 30:
                self.equity_curve.append(capital)
                self.daily_returns.append(0.0)
                continue

            # Filter to regular market hours
            if 'session' in df_1min_day.columns:
                df_regular = df_1min_day[df_1min_day['session'] == 'regular'].copy()
            else:
                df_regular = df_1min_day.copy()

            if len(df_regular) < 30:
                self.equity_curve.append(capital)
                self.daily_returns.append(0.0)
                continue

            df_regular = df_regular.reset_index(drop=True)

            # Get historical features
            try:
                X_hist = row[feature_cols].values.astype(float)
            except (ValueError, KeyError):
                self.equity_curve.append(capital)
                self.daily_returns.append(0.0)
                continue

            if np.isnan(X_hist).any():
                self.equity_curve.append(capital)
                self.daily_returns.append(0.0)
                continue

            # Generate features for each temporal slice
            X_by_slice = {}
            for ts in TEMPORAL_SLICES:
                slice_features = self.feature_generator.generate_slice_features(
                    df_regular, X_hist, ts
                )
                X_by_slice[ts] = slice_features

            # Use the entry time slice for base/masked models
            entry_slice = self.config.entry_time_minutes
            X_entry = X_by_slice.get(entry_slice, X_by_slice.get(60, X_hist))

            # Get ensemble predictions
            swing_prob, swing_unc, swing_agree = self.model_loader.predict_ensemble(
                X_entry, X_by_slice, "swing"
            )
            timing_prob, timing_unc, timing_agree = self.model_loader.predict_ensemble(
                X_entry, X_by_slice, "timing"
            )

            # Get prices
            open_price = df_regular['open'].iloc[0]
            close_price = df_regular['close'].iloc[-1]
            high_price = df_regular['high'].max()
            low_price = df_regular['low'].min()

            # Get entry price (at entry time)
            entry_bar_idx = min(self.config.entry_time_minutes, len(df_regular) - 1)
            entry_price_raw = df_regular['close'].iloc[entry_bar_idx]

            # Log periodically
            if i == 0 or i == len(test_data) - 1 or i % 50 == 0:
                logger.info(f"Day {i} ({date_str}): swing={swing_prob:.3f}±{swing_unc:.3f}, "
                           f"timing={timing_prob:.3f}, agree={swing_agree:.2f}")

            # Close existing position at EOD
            if current_position:
                exit_price = close_price
                exit_reason = "eod"

                if current_position.direction == "LONG":
                    if low_price <= current_position.entry_price * (1 - self.config.stop_loss_pct):
                        exit_price = current_position.entry_price * (1 - self.config.stop_loss_pct)
                        exit_reason = "stop_loss"
                    elif high_price >= current_position.entry_price * (1 + self.config.take_profit_pct):
                        exit_price = current_position.entry_price * (1 + self.config.take_profit_pct)
                        exit_reason = "take_profit"

                    pnl = (exit_price - current_position.entry_price) * current_position.position_size / current_position.entry_price
                else:  # SHORT
                    if high_price >= current_position.entry_price * (1 + self.config.stop_loss_pct):
                        exit_price = current_position.entry_price * (1 + self.config.stop_loss_pct)
                        exit_reason = "stop_loss"
                    elif low_price <= current_position.entry_price * (1 - self.config.take_profit_pct):
                        exit_price = current_position.entry_price * (1 - self.config.take_profit_pct)
                        exit_reason = "take_profit"

                    pnl = (current_position.entry_price - exit_price) * current_position.position_size / current_position.entry_price

                pnl -= current_position.position_size * self.config.slippage_pct

                current_position.exit_date = date_str
                current_position.exit_price = exit_price
                current_position.pnl = pnl
                current_position.pnl_pct = pnl / current_position.position_size
                current_position.exit_reason = exit_reason

                capital += pnl
                self.trades.append(current_position)
                current_position = None

            # Generate signal
            signal = None
            cascade_agreement = (swing_agree + timing_agree) / 2

            # Require higher confidence when models disagree
            conf_threshold = self.config.min_swing_conf
            if cascade_agreement < 0.7:
                conf_threshold += 0.05  # Raise threshold when models disagree

            if swing_prob > conf_threshold and timing_prob > self.config.min_timing_conf:
                signal = "LONG"
            elif swing_prob < (1 - conf_threshold) and timing_prob < (1 - self.config.min_timing_conf):
                signal = "SHORT"

            # Open new position
            if signal:
                position_value = min(
                    capital * self.config.position_size_pct,
                    capital * self.config.max_position_pct
                )

                if position_value > 100:
                    entry_price = entry_price_raw * (
                        1 + self.config.slippage_pct if signal == "LONG" else 1 - self.config.slippage_pct
                    )

                    current_position = Trade(
                        entry_date=date_str,
                        entry_price=entry_price,
                        direction=signal,
                        position_size=position_value,
                        entry_time_slice=entry_slice,
                        swing_prob=swing_prob,
                        timing_prob=timing_prob,
                        cascade_agreement=cascade_agreement,
                    )

            # Track equity
            prev_capital = self.equity_curve[-1] if self.equity_curve else capital
            daily_return = (capital - prev_capital) / prev_capital if prev_capital > 0 else 0
            self.equity_curve.append(capital)
            self.daily_returns.append(daily_return)

        result = self._calculate_results()
        self._print_summary(result)

        return result

    def _calculate_results(self) -> BacktestResult:
        """Calculate backtest metrics."""
        result = BacktestResult()
        result.trades = self.trades
        result.equity_curve = self.equity_curve
        result.daily_returns = self.daily_returns

        if not self.trades:
            return result

        result.total_trades = len(self.trades)
        wins = [t for t in self.trades if t.pnl > 0]
        losses = [t for t in self.trades if t.pnl <= 0]

        result.win_rate = len(wins) / result.total_trades if result.total_trades > 0 else 0

        total_pnl = sum(t.pnl for t in self.trades)
        result.avg_trade_pnl = total_pnl / result.total_trades if result.total_trades > 0 else 0
        result.avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        result.avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        result.profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        if self.equity_curve:
            result.total_return_pct = (self.equity_curve[-1] / self.equity_curve[0] - 1) * 100
            n_days = len(self.equity_curve)
            result.annualized_return = ((1 + result.total_return_pct / 100) ** (252 / n_days) - 1) * 100 if n_days > 0 else 0

        if self.daily_returns:
            returns = np.array(self.daily_returns)
            returns = returns[~np.isnan(returns)]

            if len(returns) > 1:
                result.sharpe_ratio = np.sqrt(252) * np.mean(returns) / (np.std(returns) + 1e-10)
                downside = returns[returns < 0]
                downside_std = np.std(downside) if len(downside) > 0 else 1e-10
                result.sortino_ratio = np.sqrt(252) * np.mean(returns) / (downside_std + 1e-10)

        if self.equity_curve:
            peak = self.equity_curve[0]
            max_dd = 0
            for equity in self.equity_curve:
                if equity > peak:
                    peak = equity
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd
            result.max_drawdown_pct = max_dd * 100

        return result

    def _print_summary(self, result: BacktestResult):
        """Print backtest summary."""
        logger.info("\n" + "=" * 70)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 70)

        logger.info(f"\nPERFORMANCE:")
        logger.info(f"  Total Return:     {result.total_return_pct:+.2f}%")
        logger.info(f"  Annualized:       {result.annualized_return:+.2f}%")
        logger.info(f"  Sharpe Ratio:     {result.sharpe_ratio:.3f}")
        logger.info(f"  Sortino Ratio:    {result.sortino_ratio:.3f}")
        logger.info(f"  Max Drawdown:     {result.max_drawdown_pct:.2f}%")

        logger.info(f"\nTRADING:")
        logger.info(f"  Total Trades:     {result.total_trades}")
        logger.info(f"  Win Rate:         {result.win_rate*100:.1f}%")
        logger.info(f"  Profit Factor:    {result.profit_factor:.2f}")
        logger.info(f"  Avg Trade P&L:    ${result.avg_trade_pnl:.2f}")
        logger.info(f"  Avg Win:          ${result.avg_win:.2f}")
        logger.info(f"  Avg Loss:         ${result.avg_loss:.2f}")

        if self.equity_curve:
            logger.info(f"\nCAPITAL:")
            logger.info(f"  Starting:         ${self.equity_curve[0]:,.2f}")
            logger.info(f"  Ending:           ${self.equity_curve[-1]:,.2f}")
            logger.info(f"  Net P&L:          ${self.equity_curve[-1] - self.equity_curve[0]:+,.2f}")


def main():
    """Run multi-timeframe temporal model backtest."""
    print("=" * 70)
    print("MULTI-TIMEFRAME TEMPORAL MODEL BACKTEST")
    print("=" * 70)
    print(f"Start time: {datetime.now()}")
    print()

    from src.data_manager import get_spy_data
    from src.train_robust_model import (
        engineer_all_features,
        add_rolling_features,
        create_soft_targets,
    )

    # Load data
    logger.info("Loading data...")
    df_1min = get_spy_data(years=3)
    logger.info(f"Loaded {len(df_1min):,} bars")

    # Create minute data dictionary by date
    logger.info("Creating intraday data dictionary...")
    df_1min['date'] = pd.to_datetime(df_1min['timestamp']).dt.date
    df_1min_dict = {}
    for date, group in df_1min.groupby('date'):
        df_1min_dict[str(date)] = group.copy().reset_index(drop=True)
    logger.info(f"Intraday data for {len(df_1min_dict)} days")

    # Engineer features
    logger.info("\nEngineering features...")
    df_features = engineer_all_features(df_1min.copy(), swing_threshold=0.0025)
    df_features = add_rolling_features(df_features)
    df_features = create_soft_targets(df_features, threshold=0.0025)
    logger.info(f"Features: {len(df_features.columns)} columns, {len(df_features)} days")

    # Load models to get feature_cols
    logger.info("\nLoading models...")
    model_loader = TemporalModelLoader()
    if not model_loader.load_models():
        logger.error("Failed to load models")
        return 1

    # Get feature columns
    if model_loader.feature_cols:
        feature_cols = model_loader.feature_cols
        logger.info(f"Using {len(feature_cols)} features from model")
    else:
        exclude_cols = ['date', 'open', 'high', 'low', 'close', 'volume',
                       'target_up', 'target_timing', 'day_return', 'timestamp',
                       'hour', 'minute', 'time', 'session', 'year']
        feature_cols = [c for c in df_features.columns if c not in exclude_cols]
        # Remove non-numeric
        feature_cols = [c for c in feature_cols
                       if df_features[c].dtype != 'object'
                       and df_features[c].dtype.name != 'datetime64[ns]']
        logger.info(f"Using {len(feature_cols)} computed features")

    # Clean data
    df_clean = df_features.dropna(subset=feature_cols).copy()
    logger.info(f"Clean samples: {len(df_clean)}")

    # Run backtest
    config = BacktestConfig(
        initial_capital=100000,
        position_size_pct=0.10,
        min_swing_conf=0.55,
        min_timing_conf=0.50,
        entry_time_minutes=60,  # Enter at T60 (10:30 AM)
    )

    backtester = TemporalBacktester(config=config, model_loader=model_loader)

    # Use 80% train, 20% test split
    test_start_idx = int(len(df_clean) * 0.8)

    result = backtester.run_backtest(
        df_features=df_clean,
        df_1min_dict=df_1min_dict,
        feature_cols=feature_cols,
        test_start_idx=test_start_idx,
    )

    # Save results
    results_dir = project_root / "reports" / "backtests"
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / f"temporal_backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    import json

    results_data = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'initial_capital': config.initial_capital,
            'position_size_pct': config.position_size_pct,
            'min_swing_conf': config.min_swing_conf,
            'min_timing_conf': config.min_timing_conf,
            'entry_time_minutes': config.entry_time_minutes,
        },
        'results': {
            'total_return_pct': result.total_return_pct,
            'annualized_return': result.annualized_return,
            'sharpe_ratio': result.sharpe_ratio,
            'sortino_ratio': result.sortino_ratio,
            'max_drawdown_pct': result.max_drawdown_pct,
            'total_trades': result.total_trades,
            'win_rate': result.win_rate,
            'profit_factor': result.profit_factor,
        },
        'trades': [
            {
                'entry_date': t.entry_date,
                'exit_date': t.exit_date,
                'direction': t.direction,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'pnl': t.pnl,
                'pnl_pct': t.pnl_pct,
                'exit_reason': t.exit_reason,
                'swing_prob': t.swing_prob,
                'timing_prob': t.timing_prob,
                'cascade_agreement': t.cascade_agreement,
            }
            for t in result.trades
        ]
    }

    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {results_file}")

    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)
    print(f"End time: {datetime.now()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
