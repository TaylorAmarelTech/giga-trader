"""
GIGA TRADER - Anti-Overfitting Module
======================================
Implements robust model evaluation and synthetic data generation to combat overfitting.

Components:
  1. Weighted Model Evaluation Score (WMES)
  2. Synthetic SPY Universes ("What SPY Could Have Been")
  3. Cross-Asset Feature Integration
  4. Hyperparameter Stability Analysis (Plateau Detection)

The #1 concern: Reduce overfitting with more realistic data while maintaining validity.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dotenv import load_dotenv

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
load_dotenv(project_root / ".env")


# ═══════════════════════════════════════════════════════════════════════════════
# ALPACA DATA HELPER (replaces yfinance)
# ═══════════════════════════════════════════════════════════════════════════════
class AlpacaDataHelper:
    """
    Helper class for fetching data from Alpaca API.
    Replaces yfinance for more reliable and consistent data access.
    """

    _instance = None
    _client = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if AlpacaDataHelper._client is None:
            self._init_client()

    def _init_client(self):
        """Initialize Alpaca data client."""
        try:
            from alpaca.data.historical import StockHistoricalDataClient

            api_key = os.getenv("ALPACA_API_KEY")
            secret_key = os.getenv("ALPACA_SECRET_KEY")

            if api_key and secret_key:
                AlpacaDataHelper._client = StockHistoricalDataClient(api_key, secret_key)
                print("[AlpacaHelper] Alpaca client initialized")
            else:
                print("[AlpacaHelper] WARNING: No Alpaca keys found in .env")
        except ImportError:
            print("[AlpacaHelper] WARNING: Alpaca SDK not installed (pip install alpaca-py)")

    @property
    def client(self):
        return AlpacaDataHelper._client

    def download_daily_bars(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Download daily OHLCV bars for multiple symbols.

        Args:
            symbols: List of ticker symbols
            start_date: Start date for data
            end_date: End date for data

        Returns:
            DataFrame with MultiIndex (date, symbol) or columns per symbol
        """
        if not self.client:
            print("[AlpacaHelper] Client not initialized, returning empty DataFrame")
            return pd.DataFrame()

        try:
            from alpaca.data.requests import StockBarsRequest
            from alpaca.data.timeframe import TimeFrame

            # Download in chunks to avoid rate limits
            all_bars = []
            chunk_size = 20  # Alpaca handles multiple symbols well

            for i in range(0, len(symbols), chunk_size):
                chunk_symbols = symbols[i:i + chunk_size]

                try:
                    request = StockBarsRequest(
                        symbol_or_symbols=chunk_symbols,
                        timeframe=TimeFrame.Day,
                        start=start_date,
                        end=end_date,
                    )

                    bars = self.client.get_stock_bars(request)

                    if hasattr(bars, 'df') and len(bars.df) > 0:
                        all_bars.append(bars.df)

                except Exception as e:
                    print(f"  [AlpacaHelper] Failed to download chunk {chunk_symbols[:3]}...: {e}")
                    continue

            if len(all_bars) == 0:
                return pd.DataFrame()

            # Combine all chunks
            combined = pd.concat(all_bars)

            # Handle MultiIndex (symbol, timestamp)
            if isinstance(combined.index, pd.MultiIndex):
                # Pivot to have symbols as columns with close prices
                close_prices = combined['close'].unstack(level='symbol')
                high_prices = combined['high'].unstack(level='symbol')
                low_prices = combined['low'].unstack(level='symbol')

                # Normalize timestamp index to just dates (no timezone)
                # This ensures compatibility with spy_daily date matching
                close_prices.index = pd.to_datetime(close_prices.index).normalize().tz_localize(None)
                high_prices.index = pd.to_datetime(high_prices.index).normalize().tz_localize(None)
                low_prices.index = pd.to_datetime(low_prices.index).normalize().tz_localize(None)

                # Return close prices with normalized date index
                return {
                    'close': close_prices,
                    'high': high_prices,
                    'low': low_prices,
                }

            return combined

        except Exception as e:
            print(f"[AlpacaHelper] Download error: {e}")
            return pd.DataFrame()

    def download_close_prices(
        self,
        symbols: List[str],
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Download just close prices for multiple symbols.
        Returns DataFrame with date index and symbol columns.
        """
        result = self.download_daily_bars(symbols, start_date, end_date)

        if isinstance(result, dict):
            return result.get('close', pd.DataFrame())
        elif isinstance(result, pd.DataFrame) and not result.empty:
            if 'close' in result.columns:
                return result[['close']]
            return result
        return pd.DataFrame()


# Create singleton instance
_alpaca_helper = None

def get_alpaca_helper() -> AlpacaDataHelper:
    """Get the singleton AlpacaDataHelper instance."""
    global _alpaca_helper
    if _alpaca_helper is None:
        _alpaca_helper = AlpacaDataHelper()
    return _alpaca_helper


# ═══════════════════════════════════════════════════════════════════════════════
# 1. WEIGHTED MODEL EVALUATION SCORE (WMES)
# ═══════════════════════════════════════════════════════════════════════════════
class WeightedModelEvaluator:
    """
    Evaluates models on multiple dimensions beyond simple win rate.

    Components:
      - Win Rate (traditional)
      - Robustness (stability across variations)
      - Profit Potential (risk-adjusted returns)
      - Noise Tolerance (performance on noisy data)
      - Plateau Stability (sensitivity to parameter changes)
      - Complexity Penalty (fewer features preferred)
    """

    def __init__(self, weights: Dict[str, float] = None):
        """Initialize with custom weights or defaults."""
        self.weights = weights or {
            "win_rate": 0.15,           # Traditional metric (reduced weight)
            "robustness": 0.25,         # Cross-validation stability
            "profit_potential": 0.20,   # Risk-adjusted returns
            "noise_tolerance": 0.15,    # Performance on noisy data
            "plateau_stability": 0.15,  # Parameter sensitivity
            "complexity_penalty": 0.10, # Fewer features preferred
        }

    def compute_wmes(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_proba: np.ndarray,
        returns: np.ndarray,
        cv_scores: List[float],
        n_features: int,
        hp_sensitivity: float,
        noise_scores: List[float] = None,
    ) -> Dict[str, float]:
        """
        Compute Weighted Model Evaluation Score.

        Args:
            y_true: Ground truth labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities
            returns: Actual returns for each prediction
            cv_scores: Cross-validation AUC scores
            n_features: Number of features used
            hp_sensitivity: How much AUC changes with small HP changes (lower=better)
            noise_scores: Performance on noisy test data

        Returns:
            Dictionary with component scores and final WMES
        """
        scores = {}

        # 1. Win Rate (but capped to prevent over-weighting)
        buy_mask = y_pred == 1
        if buy_mask.sum() > 0:
            buy_returns = returns[buy_mask]
            raw_win_rate = (buy_returns > 0).mean()
            # Cap win rate contribution (suspicious if > 75%)
            scores["win_rate"] = min(raw_win_rate, 0.75) / 0.75
        else:
            scores["win_rate"] = 0.5

        # 2. Robustness (CV score stability)
        cv_mean = np.mean(cv_scores)
        cv_std = np.std(cv_scores)
        # Penalize high variance in CV scores
        scores["robustness"] = cv_mean * (1 - min(cv_std / cv_mean, 0.5))

        # 3. Profit Potential (risk-adjusted)
        if buy_mask.sum() > 0:
            buy_returns = returns[buy_mask]
            avg_win = buy_returns[buy_returns > 0].mean() if (buy_returns > 0).any() else 0
            avg_loss = abs(buy_returns[buy_returns < 0].mean()) if (buy_returns < 0).any() else 0.001
            profit_factor = avg_win / avg_loss if avg_loss > 0 else 1
            # Sharpe-like ratio
            sharpe = buy_returns.mean() / (buy_returns.std() + 1e-6)
            scores["profit_potential"] = min((profit_factor * 0.3 + sharpe * 0.7), 1.0)
        else:
            scores["profit_potential"] = 0

        # 4. Noise Tolerance
        if noise_scores is not None and len(noise_scores) > 0:
            # How well does the model perform on noisy data?
            noise_degradation = (cv_mean - np.mean(noise_scores)) / cv_mean
            scores["noise_tolerance"] = max(1 - noise_degradation, 0)
        else:
            scores["noise_tolerance"] = 0.5  # Neutral if not tested

        # 5. Plateau Stability (lower sensitivity = better)
        # hp_sensitivity should be in [0, 1] where 0 = very stable
        scores["plateau_stability"] = 1 - min(hp_sensitivity, 1.0)

        # 6. Complexity Penalty (fewer features = better, with diminishing returns)
        optimal_features = 30  # Sweet spot
        if n_features <= optimal_features:
            scores["complexity_penalty"] = 1.0
        else:
            # Penalize excess features
            excess = n_features - optimal_features
            scores["complexity_penalty"] = max(1 - (excess / 100), 0.5)

        # Compute weighted final score
        wmes = sum(scores[k] * self.weights[k] for k in self.weights)
        scores["wmes"] = wmes

        return scores

    def evaluate(
        self,
        y_test: np.ndarray = None,
        y_pred_proba: np.ndarray = None,
        y_train: np.ndarray = None,
        y_train_proba: np.ndarray = None,
        cv_scores: List[float] = None,
        n_features: int = 50,
        **kwargs,
    ) -> Dict[str, float]:
        """
        Simplified evaluation interface for experiment engine.

        This is a wrapper around compute_wmes() with sensible defaults.

        Args:
            y_test: Test set labels
            y_pred_proba: Test set prediction probabilities
            y_train: Training set labels (optional)
            y_train_proba: Training set probabilities (optional)
            cv_scores: Cross-validation scores
            n_features: Number of features used

        Returns:
            Dictionary with wmes_score and component scores
        """
        from sklearn.metrics import roc_auc_score

        # Default CV scores if not provided
        if cv_scores is None:
            if y_test is not None and y_pred_proba is not None:
                try:
                    test_auc = roc_auc_score(y_test, y_pred_proba)
                    cv_scores = [test_auc] * 5  # Use test AUC as proxy
                except (ValueError, TypeError):
                    cv_scores = [0.5] * 5
            else:
                cv_scores = [0.5] * 5

        # Compute predictions from probabilities
        if y_pred_proba is not None:
            y_pred = (y_pred_proba > 0.5).astype(int)
        else:
            y_pred = np.zeros(len(y_test) if y_test is not None else 100)

        # Create synthetic returns based on predictions vs actuals
        if y_test is not None and y_pred_proba is not None:
            # Simulate returns: correct prediction = +0.01, wrong = -0.01
            correct = (y_pred == y_test).astype(float)
            returns = (correct * 2 - 1) * 0.01  # +/- 1%
        else:
            returns = np.random.normal(0, 0.01, len(y_pred))

        # Estimate hyperparameter sensitivity (default to moderate)
        hp_sensitivity = 0.3

        # Call compute_wmes with derived values
        if y_test is None:
            y_test = np.zeros(len(y_pred))
        if y_pred_proba is None:
            y_pred_proba = np.zeros(len(y_pred))

        result = self.compute_wmes(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_pred_proba,
            returns=returns,
            cv_scores=cv_scores,
            n_features=n_features,
            hp_sensitivity=hp_sensitivity,
            noise_scores=None,
        )

        # Add convenience alias
        result["wmes_score"] = result["wmes"]

        return result

    def analyze_hp_sensitivity(
        self,
        base_params: Dict,
        param_ranges: Dict[str, Tuple[float, float]],
        evaluate_fn,
        n_perturbations: int = 10,
    ) -> float:
        """
        Measure how sensitive the model is to small parameter changes.

        Returns:
            Sensitivity score (0 = very stable, 1 = very fragile)
        """
        base_score = evaluate_fn(base_params)
        score_changes = []

        for param, (low, high) in param_ranges.items():
            # Perturb each parameter by +/- 5%
            base_val = base_params.get(param)
            if base_val is None:
                continue

            for direction in [-0.05, 0.05]:
                perturbed = base_params.copy()
                new_val = base_val * (1 + direction)
                new_val = max(low, min(high, new_val))  # Clip to range
                perturbed[param] = new_val

                try:
                    perturbed_score = evaluate_fn(perturbed)
                    change = abs(perturbed_score - base_score) / (base_score + 1e-6)
                    score_changes.append(change)
                except (ValueError, TypeError, RuntimeError):
                    pass

        if len(score_changes) == 0:
            return 0.5  # Neutral

        # Average change indicates sensitivity
        avg_change = np.mean(score_changes)
        # Convert to 0-1 where high change = high sensitivity
        sensitivity = min(avg_change * 10, 1.0)  # Scale factor

        return sensitivity


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SYNTHETIC SPY UNIVERSES ("What SPY Could Have Been")
# ═══════════════════════════════════════════════════════════════════════════════
class SyntheticSPYGenerator:
    """
    Generate alternative SPY versions using component stocks.

    Methods:
      1. Filter extremes: Remove top/bottom N% performers each day
      2. Filter middle: Remove middle N% performers
      3. Sector rotation: Over/under-weight sectors
      4. Volatility filter: Remove high/low volatility stocks
      5. Momentum filter: Remove momentum outliers

    These create realistic "what could have been" scenarios that:
      - Increase effective training data
      - Reduce overfitting to specific SPY path
      - Create more robust models
    """

    def __init__(
        self,
        n_universes: int = 10,
        real_weight: float = 0.6,  # Weight on real SPY vs synthetic
    ):
        self.n_universes = n_universes
        self.real_weight = real_weight
        self.synthetic_weight = (1 - real_weight) / n_universes

    def download_spy_components(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Download historical data for SPY components using Alpaca."""
        print("\n[SYNTHETIC SPY] Downloading component data via Alpaca...")

        # Top SPY components by weight (as of 2024)
        # Note: BRK.B uses period format for Alpaca API
        components = [
            "AAPL", "MSFT", "AMZN", "NVDA", "GOOGL", "META", "TSLA", "BRK.B",
            "UNH", "XOM", "JNJ", "JPM", "V", "PG", "MA", "HD", "CVX", "MRK",
            "ABBV", "LLY", "PEP", "KO", "COST", "AVGO", "WMT", "MCD", "CSCO",
            "ACN", "TMO", "ABT", "DHR", "VZ", "ADBE", "CRM", "NKE", "CMCSA",
            "PFE", "INTC", "TXN", "AMD", "NEE", "PM", "RTX", "HON", "UNP",
            "IBM", "LOW", "SPGI", "BA", "CAT"
        ]

        try:
            helper = get_alpaca_helper()
            close_prices = helper.download_close_prices(components, start_date, end_date)

            if close_prices.empty:
                print("  [WARN] No data returned from Alpaca")
                return pd.DataFrame()

            print(f"  Downloaded {len(close_prices.columns)} components, {len(close_prices)} days")
            return close_prices

        except Exception as e:
            print(f"  [ERROR] Failed to download via Alpaca: {e}")
            return pd.DataFrame()

    def generate_universes(
        self,
        component_prices: pd.DataFrame,
        spy_data: pd.DataFrame,
    ) -> List[pd.DataFrame]:
        """
        Generate synthetic SPY universes.

        Each universe is a filtered/modified version of the equal-weighted
        component basket, then normalized to match SPY's characteristics.
        """
        if component_prices.empty:
            print("[WARN] No component data, skipping synthetic generation")
            return []

        print(f"\n[SYNTHETIC SPY] Generating {self.n_universes} alternative universes...")

        # Calculate daily returns for all components
        returns = component_prices.pct_change().dropna()

        universes = []

        # ─────────────────────────────────────────────────────────────────────
        # Universe 1-2: Filter Top/Bottom Performers
        # ─────────────────────────────────────────────────────────────────────
        for filter_pct in [0.1, 0.2]:  # Filter 10% and 20%
            filtered_returns = self._filter_extreme_performers(returns, filter_pct)
            universe = self._returns_to_spy_like(filtered_returns, spy_data)
            universe["universe_type"] = f"filter_extreme_{int(filter_pct*100)}pct"
            universes.append(universe)
            print(f"  Created: Filter extreme {int(filter_pct*100)}% performers")

        # ─────────────────────────────────────────────────────────────────────
        # Universe 3-4: Filter Middle Performers
        # ─────────────────────────────────────────────────────────────────────
        for filter_pct in [0.1, 0.2]:
            filtered_returns = self._filter_middle_performers(returns, filter_pct)
            universe = self._returns_to_spy_like(filtered_returns, spy_data)
            universe["universe_type"] = f"filter_middle_{int(filter_pct*100)}pct"
            universes.append(universe)
            print(f"  Created: Filter middle {int(filter_pct*100)}% performers")

        # ─────────────────────────────────────────────────────────────────────
        # Universe 5-6: Volatility Filter
        # ─────────────────────────────────────────────────────────────────────
        for vol_filter in ["low", "high"]:
            filtered_returns = self._filter_by_volatility(returns, vol_filter)
            universe = self._returns_to_spy_like(filtered_returns, spy_data)
            universe["universe_type"] = f"filter_{vol_filter}_vol"
            universes.append(universe)
            print(f"  Created: Filter {vol_filter} volatility stocks")

        # ─────────────────────────────────────────────────────────────────────
        # Universe 7-8: Momentum Filter
        # ─────────────────────────────────────────────────────────────────────
        for mom_filter in ["winners", "losers"]:
            filtered_returns = self._filter_by_momentum(returns, mom_filter)
            universe = self._returns_to_spy_like(filtered_returns, spy_data)
            universe["universe_type"] = f"filter_{mom_filter}"
            universes.append(universe)
            print(f"  Created: Filter momentum {mom_filter}")

        # ─────────────────────────────────────────────────────────────────────
        # Universe 9-10: Random Subset (Bootstrap)
        # ─────────────────────────────────────────────────────────────────────
        for seed in [42, 123]:
            np.random.seed(seed)
            n_keep = int(len(returns.columns) * 0.7)
            keep_cols = np.random.choice(returns.columns, n_keep, replace=False)
            filtered_returns = returns[keep_cols]
            universe = self._returns_to_spy_like(filtered_returns, spy_data)
            universe["universe_type"] = f"bootstrap_{seed}"
            universes.append(universe)
            print(f"  Created: Bootstrap sample (seed={seed})")

        print(f"[PASS] Generated {len(universes)} synthetic universes")
        return universes

    def _filter_extreme_performers(self, returns: pd.DataFrame, filter_pct: float) -> pd.DataFrame:
        """Remove top and bottom N% performers each day."""
        filtered = returns.copy()

        for idx in returns.index:
            day_returns = returns.loc[idx].dropna()
            if len(day_returns) < 10:
                continue

            n_filter = max(1, int(len(day_returns) * filter_pct))
            sorted_returns = day_returns.sort_values()

            # Remove top and bottom
            to_remove = list(sorted_returns.head(n_filter).index) + \
                       list(sorted_returns.tail(n_filter).index)
            filtered.loc[idx, to_remove] = np.nan

        return filtered

    def _filter_middle_performers(self, returns: pd.DataFrame, filter_pct: float) -> pd.DataFrame:
        """Remove middle N% performers each day (keep extremes)."""
        filtered = returns.copy()

        for idx in returns.index:
            day_returns = returns.loc[idx].dropna()
            if len(day_returns) < 10:
                continue

            n_filter = max(1, int(len(day_returns) * filter_pct))
            sorted_returns = day_returns.sort_values()

            # Remove middle
            mid_start = len(sorted_returns) // 2 - n_filter // 2
            mid_end = mid_start + n_filter
            to_remove = list(sorted_returns.iloc[mid_start:mid_end].index)
            filtered.loc[idx, to_remove] = np.nan

        return filtered

    def _filter_by_volatility(self, returns: pd.DataFrame, filter_type: str) -> pd.DataFrame:
        """Remove high or low volatility stocks."""
        # Calculate rolling volatility
        vol = returns.rolling(20).std().mean()

        if filter_type == "high":
            threshold = vol.quantile(0.8)
            keep = vol[vol < threshold].index
        else:  # low
            threshold = vol.quantile(0.2)
            keep = vol[vol > threshold].index

        return returns[keep]

    def _filter_by_momentum(self, returns: pd.DataFrame, filter_type: str) -> pd.DataFrame:
        """Remove momentum winners or losers (trailing 20-day)."""
        # Calculate trailing momentum
        momentum = returns.rolling(20).sum().iloc[-1]

        if filter_type == "winners":
            threshold = momentum.quantile(0.8)
            keep = momentum[momentum < threshold].index
        else:  # losers
            threshold = momentum.quantile(0.2)
            keep = momentum[momentum > threshold].index

        return returns[keep]

    def generate_spy_minus_component_universes(
        self,
        component_prices: pd.DataFrame,
        spy_data: pd.DataFrame,
    ) -> List[pd.DataFrame]:
        """
        Generate "SPY minus component" universes.

        For each major component, calculate its contribution to SPY and subtract it
        to create a "what if this stock didn't exist in SPY" scenario.

        This creates realistic alternative universes that help the model learn
        patterns that are robust to individual stock movements.
        """
        if component_prices.empty:
            return []

        print("\n[SPY-MINUS-COMPONENT] Generating component-subtracted universes...")

        # Calculate daily returns for components
        component_returns = component_prices.pct_change().dropna()

        # Normalize component returns index to date-only (no time, no timezone)
        component_returns.index = pd.to_datetime(component_returns.index).normalize()
        if component_returns.index.tz is not None:
            component_returns.index = component_returns.index.tz_localize(None)

        # Get SPY daily returns
        spy_returns = spy_data.groupby("date")["day_return"].first()
        # Normalize spy_returns index to same format (date-only, no timezone)
        spy_returns.index = pd.to_datetime(spy_returns.index).normalize()
        if spy_returns.index.tz is not None:
            spy_returns.index = spy_returns.index.tz_localize(None)

        # Debug: Print date ranges to help diagnose issues
        if len(component_returns) > 0 and len(spy_returns) > 0:
            print(f"  Component dates: {component_returns.index.min()} to {component_returns.index.max()} ({len(component_returns)} days)")
            print(f"  SPY dates: {spy_returns.index.min()} to {spy_returns.index.max()} ({len(spy_returns)} days)")

        common_dates = component_returns.index.intersection(spy_returns.index)

        if len(common_dates) == 0:
            print("  [WARN] No common dates between components and SPY")
            print(f"    Component index type: {type(component_returns.index)}, dtype: {component_returns.index.dtype}")
            print(f"    SPY index type: {type(spy_returns.index)}, dtype: {spy_returns.index.dtype}")
            if len(component_returns) > 0:
                print(f"    Sample component date: {repr(component_returns.index[0])}")
            if len(spy_returns) > 0:
                print(f"    Sample SPY date: {repr(spy_returns.index[0])}")
            return []

        print(f"  Common dates found: {len(common_dates)}")

        component_returns = component_returns.loc[common_dates]
        spy_ret = spy_returns.loc[common_dates]

        # Approximate SPY component weights (based on typical market cap weights)
        # These are rough estimates - actual weights change daily
        component_weights = {
            "AAPL": 0.07, "MSFT": 0.07, "NVDA": 0.06, "AMZN": 0.04, "GOOGL": 0.04,
            "META": 0.03, "TSLA": 0.02, "BRK.B": 0.02, "UNH": 0.01, "XOM": 0.01,
            "JNJ": 0.01, "JPM": 0.01, "V": 0.01, "PG": 0.01, "MA": 0.01,
            "HD": 0.01, "CVX": 0.01, "MRK": 0.01, "ABBV": 0.01, "LLY": 0.01,
        }

        universes = []

        # Create "SPY minus [component]" for top components
        top_components = ["NVDA", "AAPL", "MSFT", "AMZN", "GOOGL", "META", "TSLA"]

        for component in top_components:
            if component not in component_returns.columns:
                continue

            weight = component_weights.get(component, 0.02)
            comp_ret = component_returns[component]

            # Calculate component's contribution to SPY
            contribution = comp_ret * weight

            # Create "SPY minus component" return
            spy_minus_component = spy_ret - contribution.values

            universe = pd.DataFrame({
                "date": common_dates,
                "day_return": spy_minus_component.values,
                "synthetic_return": spy_minus_component.values,
                "real_return": spy_ret.values,
                "universe_type": f"spy_minus_{component}",
                "component_removed": component,
                "component_contribution": contribution.values,
            })
            universes.append(universe)
            print(f"  Created: SPY minus {component} (weight={weight:.1%})")

        # Create "SPY minus MAG7" (all top tech combined)
        mag7 = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]
        mag7_contribution = pd.Series(0.0, index=common_dates)
        for comp in mag7:
            if comp in component_returns.columns:
                weight = component_weights.get(comp, 0.02)
                mag7_contribution += component_returns[comp].values * weight

        spy_minus_mag7 = spy_ret - mag7_contribution.values
        universe = pd.DataFrame({
            "date": common_dates,
            "day_return": spy_minus_mag7.values,
            "synthetic_return": spy_minus_mag7.values,
            "real_return": spy_ret.values,
            "universe_type": "spy_minus_mag7",
        })
        universes.append(universe)
        print(f"  Created: SPY minus MAG7 (combined weight ~{sum(component_weights.get(c, 0) for c in mag7):.1%})")

        # Create "SPY with 2x component" (over-weighted scenarios)
        for component in ["NVDA", "AAPL", "MSFT"]:
            if component not in component_returns.columns:
                continue

            weight = component_weights.get(component, 0.02)
            comp_ret = component_returns[component]

            # Add extra contribution (simulate if component had 2x weight)
            extra_contribution = comp_ret * weight
            spy_plus_component = spy_ret + extra_contribution.values

            universe = pd.DataFrame({
                "date": common_dates,
                "day_return": spy_plus_component.values,
                "synthetic_return": spy_plus_component.values,
                "real_return": spy_ret.values,
                "universe_type": f"spy_plus_{component}_2x",
            })
            universes.append(universe)
            print(f"  Created: SPY plus 2x {component}")

        print(f"[PASS] Generated {len(universes)} component-modified universes")
        return universes

    def _returns_to_spy_like(
        self,
        filtered_returns: pd.DataFrame,
        spy_data: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert filtered component returns to SPY-like OHLCV data.

        1. Equal-weight average the filtered returns
        2. Scale to match SPY's volatility
        3. Generate OHLCV from returns
        """
        # Equal-weight average
        avg_return = filtered_returns.mean(axis=1)

        # Normalize avg_return index to date-only (no time, no timezone)
        avg_return.index = pd.to_datetime(avg_return.index).normalize()
        if avg_return.index.tz is not None:
            avg_return.index = avg_return.index.tz_localize(None)

        # Match SPY dates
        spy_returns = spy_data.groupby("date")["day_return"].first()
        # Normalize spy_returns index to same format (date-only, no timezone)
        spy_returns.index = pd.to_datetime(spy_returns.index).normalize()
        if spy_returns.index.tz is not None:
            spy_returns.index = spy_returns.index.tz_localize(None)

        common_dates = avg_return.index.intersection(spy_returns.index)

        if len(common_dates) == 0:
            return pd.DataFrame()

        avg_return = avg_return.loc[common_dates]
        spy_ret = spy_returns.loc[common_dates]

        # Scale to match SPY volatility (but keep direction differences)
        scale_factor = spy_ret.std() / (avg_return.std() + 1e-10)
        scaled_return = avg_return * scale_factor * 0.8  # 80% of SPY vol to be conservative

        # Blend with real SPY (don't deviate too much)
        blended_return = (
            self.real_weight * spy_ret +
            self.synthetic_weight * self.n_universes * scaled_return
        )

        # Create SPY-like dataframe
        result = pd.DataFrame({
            "date": common_dates,
            "day_return": blended_return.values,
            "synthetic_return": scaled_return.values,
            "real_return": spy_ret.values,
        })

        return result

    def create_augmented_dataset(
        self,
        real_spy_features: pd.DataFrame,
        universes: List[pd.DataFrame],
    ) -> pd.DataFrame:
        """
        Create augmented training dataset by combining real SPY with synthetic.

        The synthetic data gets lower weight to avoid over-influencing the model.
        """
        if len(universes) == 0:
            return real_spy_features

        print("\n[AUGMENTED DATA] Creating combined dataset...")

        # Add weight column to real data
        real_spy_features = real_spy_features.copy()
        real_spy_features["sample_weight_augment"] = self.real_weight

        augmented_samples = [real_spy_features]

        for i, universe in enumerate(universes):
            if "day_return" not in universe.columns:
                continue

            # Match dates with real features
            merged = real_spy_features.merge(
                universe[["date", "day_return"]],
                on="date",
                how="inner",
                suffixes=("", "_synth")
            )

            if len(merged) == 0:
                continue

            # Replace day_return with synthetic
            synth_features = merged.copy()
            synth_features["day_return"] = merged["day_return_synth"]
            synth_features["sample_weight_augment"] = self.synthetic_weight
            synth_features["universe_id"] = i

            # Recalculate targets based on synthetic returns
            threshold = 0.0025
            synth_features["target_up"] = synth_features["day_return"] > threshold
            synth_features["is_up_day"] = synth_features["day_return"] > threshold
            synth_features["is_down_day"] = synth_features["day_return"] < -threshold

            # Drop the _synth column
            synth_features = synth_features.drop(columns=["day_return_synth"], errors="ignore")

            augmented_samples.append(synth_features)

        combined = pd.concat(augmented_samples, ignore_index=True)

        print(f"  Real samples: {len(real_spy_features)}")
        print(f"  Synthetic samples: {len(combined) - len(real_spy_features)}")
        print(f"  Total augmented: {len(combined)}")
        print(f"  Effective weight ratio: {self.real_weight:.0%} real, {1-self.real_weight:.0%} synthetic")

        return combined


# ═══════════════════════════════════════════════════════════════════════════════
# 2B. COMPONENT STREAK BREADTH FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
class ComponentStreakFeatures:
    """
    Track % of SPY components and market-cap weighted % that have been green
    for consecutive days (2, 3, 4, 5+ days in a row).

    These breadth features indicate:
      - Momentum strength/exhaustion
      - Breadth divergences (SPY up but fewer stocks participating)
      - Potential reversal signals
    """

    # Approximate market cap weights for top SPY components (as of 2024)
    COMPONENT_WEIGHTS = {
        "AAPL": 0.072, "MSFT": 0.071, "AMZN": 0.036, "NVDA": 0.031, "GOOGL": 0.021,
        "META": 0.020, "TSLA": 0.018, "BRK.B": 0.017, "UNH": 0.013, "XOM": 0.013,
        "JNJ": 0.012, "JPM": 0.012, "V": 0.011, "PG": 0.011, "MA": 0.010,
        "HD": 0.010, "CVX": 0.010, "MRK": 0.009, "ABBV": 0.009, "LLY": 0.009,
        "PEP": 0.008, "KO": 0.008, "COST": 0.008, "AVGO": 0.008, "WMT": 0.007,
        "MCD": 0.007, "CSCO": 0.007, "ACN": 0.007, "TMO": 0.007, "ABT": 0.006,
        "DHR": 0.006, "VZ": 0.006, "ADBE": 0.006, "CRM": 0.006, "NKE": 0.005,
        "CMCSA": 0.005, "PFE": 0.005, "INTC": 0.005, "TXN": 0.005, "AMD": 0.005,
        "NEE": 0.005, "PM": 0.005, "RTX": 0.005, "HON": 0.005, "UNP": 0.004,
        "IBM": 0.004, "LOW": 0.004, "SPGI": 0.004, "BA": 0.004, "CAT": 0.004,
    }

    def __init__(self, max_streak: int = 10):
        self.max_streak = max_streak
        self.components = list(self.COMPONENT_WEIGHTS.keys())

    def download_component_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Download daily close prices for SPY components via Alpaca."""
        print("\n[BREADTH] Downloading component data for streak analysis via Alpaca...")

        try:
            helper = get_alpaca_helper()
            close_prices = helper.download_close_prices(self.components, start_date, end_date)

            if close_prices.empty:
                print("  [WARN] No data returned from Alpaca")
                return pd.DataFrame()

            print(f"  Downloaded {len(close_prices.columns)} components, {len(close_prices)} days")
            return close_prices

        except Exception as e:
            print(f"  [ERROR] Failed to download via Alpaca: {e}")
            return pd.DataFrame()

    def compute_streak_features(
        self,
        component_prices: pd.DataFrame,
        spy_daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Compute breadth features based on component streaks.

        Features:
          - pct_green_2d: % of components green 2+ days in a row
          - pct_green_3d: % of components green 3+ days in a row
          - pct_green_4d: % of components green 4+ days in a row
          - pct_green_5d: % of components green 5+ days in a row
          - wtd_pct_green_Nd: Same but weighted by market cap
          - pct_red_Nd: % of components red N+ days in a row
          - wtd_pct_red_Nd: Weighted version
          - breadth_divergence: SPY return vs breadth change
        """
        if component_prices.empty:
            return spy_daily

        print("\n[BREADTH] Computing component streak features...")

        # Calculate daily returns for each component
        returns = component_prices.pct_change()

        # Compute streaks for each component
        streaks = self._compute_all_streaks(returns)

        # Create features dataframe
        features_list = []

        for date in returns.index:
            if date not in streaks.index:
                continue

            day_streaks = streaks.loc[date]
            # Keep as timestamp for consistent merge with spy_daily
            record = {"date": pd.Timestamp(date.date())}

            # For each streak length
            for n in range(2, self.max_streak + 1):
                # Unweighted percentages
                green_count = (day_streaks >= n).sum()
                red_count = (day_streaks <= -n).sum()
                total = len(day_streaks.dropna())

                record[f"pct_green_{n}d"] = green_count / total if total > 0 else 0
                record[f"pct_red_{n}d"] = red_count / total if total > 0 else 0

                # Weighted percentages
                wtd_green = 0
                wtd_red = 0
                total_weight = 0

                for ticker in day_streaks.index:
                    streak = day_streaks[ticker]
                    weight = self.COMPONENT_WEIGHTS.get(ticker, 0.01)

                    if pd.notna(streak):
                        total_weight += weight
                        if streak >= n:
                            wtd_green += weight
                        elif streak <= -n:
                            wtd_red += weight

                record[f"wtd_pct_green_{n}d"] = wtd_green / total_weight if total_weight > 0 else 0
                record[f"wtd_pct_red_{n}d"] = wtd_red / total_weight if total_weight > 0 else 0

            # Summary features
            record["avg_streak_length"] = day_streaks.mean()
            record["max_green_streak"] = day_streaks.max()
            record["max_red_streak"] = day_streaks.min()
            record["streak_dispersion"] = day_streaks.std()

            # Net green (green - red) for various lengths
            for n in [2, 3, 5]:
                record[f"net_green_{n}d"] = record[f"pct_green_{n}d"] - record[f"pct_red_{n}d"]
                record[f"wtd_net_green_{n}d"] = record[f"wtd_pct_green_{n}d"] - record[f"wtd_pct_red_{n}d"]

            features_list.append(record)

        streak_features = pd.DataFrame(features_list)

        # Merge with spy_daily
        spy_daily = spy_daily.copy()
        result = spy_daily.merge(streak_features, on="date", how="left")

        # Fill NaN with 0 for streak features
        streak_cols = [c for c in result.columns if "green" in c or "red" in c or "streak" in c]
        result[streak_cols] = result[streak_cols].fillna(0)

        # Add lagged streak features (previous day's breadth)
        for col in ["pct_green_3d", "pct_red_3d", "wtd_net_green_3d"]:
            if col in result.columns:
                result[f"{col}_lag1"] = result[col].shift(1)
                result[f"{col}_change"] = result[col] - result[col].shift(1)

        # Breadth divergence: Is SPY going up while breadth is declining?
        if "day_return" in result.columns:
            result["breadth_divergence"] = (
                result["day_return"].rolling(3).sum() *
                result["net_green_3d"].diff().rolling(3).sum()
            )
            # Negative = divergence (SPY up, breadth down or vice versa)

        print(f"  Added {len(streak_cols)} streak-based breadth features")
        return result

    def _compute_all_streaks(self, returns: pd.DataFrame) -> pd.DataFrame:
        """
        Compute consecutive day streaks for all components.

        Returns DataFrame where positive = green streak, negative = red streak.
        """
        streaks = pd.DataFrame(index=returns.index, columns=returns.columns)

        for ticker in returns.columns:
            ticker_returns = returns[ticker]
            streak = 0

            for i, (date, ret) in enumerate(ticker_returns.items()):
                if pd.isna(ret):
                    streak = 0
                elif ret > 0:
                    if streak >= 0:
                        streak += 1
                    else:
                        streak = 1
                elif ret < 0:
                    if streak <= 0:
                        streak -= 1
                    else:
                        streak = -1
                else:
                    streak = 0

                streaks.loc[date, ticker] = streak

        return streaks

    def analyze_breadth_signal(self, features: pd.DataFrame) -> Dict:
        """
        Analyze current breadth conditions.

        Returns signal interpretation.
        """
        if len(features) == 0:
            return {}

        latest = features.iloc[-1]

        signal = {
            "date": latest.get("date"),
            "pct_green_3d": latest.get("pct_green_3d", 0),
            "pct_red_3d": latest.get("pct_red_3d", 0),
            "wtd_net_green_3d": latest.get("wtd_net_green_3d", 0),
            "breadth_divergence": latest.get("breadth_divergence", 0),
        }

        # Interpretation
        if signal["wtd_net_green_3d"] > 0.3:
            signal["interpretation"] = "STRONG_BULLISH_BREADTH"
        elif signal["wtd_net_green_3d"] > 0.1:
            signal["interpretation"] = "BULLISH_BREADTH"
        elif signal["wtd_net_green_3d"] < -0.3:
            signal["interpretation"] = "STRONG_BEARISH_BREADTH"
        elif signal["wtd_net_green_3d"] < -0.1:
            signal["interpretation"] = "BEARISH_BREADTH"
        else:
            signal["interpretation"] = "NEUTRAL_BREADTH"

        if signal["breadth_divergence"] < -0.01:
            signal["warning"] = "POTENTIAL_DIVERGENCE"

        return signal


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CROSS-ASSET FEATURES
# ═══════════════════════════════════════════════════════════════════════════════
class CrossAssetFeatures:
    """
    Add features from correlated assets: TLT, QQQ, GLD, VIX, etc.

    These provide:
      - Risk-on/risk-off signals
      - Interest rate sensitivity
      - Sector rotation hints
      - Volatility regime context
    """

    ASSETS = {
        "TLT": "Treasury bonds (20+ year)",
        "QQQ": "NASDAQ 100 (tech-heavy)",
        "GLD": "Gold ETF",
        "IWM": "Russell 2000 (small caps)",
        "EEM": "Emerging markets",
        "VXX": "VIX short-term futures",
        "UUP": "US Dollar index ETF",  # Changed from DXY (not a stock)
        "HYG": "High yield bonds",
    }

    def __init__(self, assets: List[str] = None):
        self.assets = assets or list(self.ASSETS.keys())

    def download_cross_assets(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Download cross-asset data via Alpaca."""
        print("\n[CROSS-ASSETS] Downloading correlated asset data via Alpaca...")

        try:
            helper = get_alpaca_helper()
            close_prices = helper.download_close_prices(self.assets, start_date, end_date)

            if close_prices.empty:
                print("  [WARN] No data returned from Alpaca")
                return pd.DataFrame()

            print(f"  Downloaded {len(close_prices.columns)} assets, {len(close_prices)} days")
            return close_prices

        except Exception as e:
            print(f"  [ERROR] Failed to download via Alpaca: {e}")
            return pd.DataFrame()

    def create_cross_asset_features(
        self,
        cross_data: pd.DataFrame,
        spy_daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create features from cross-asset data.

        Features:
          - Daily returns
          - Relative strength vs SPY
          - Correlation rolling
          - Regime indicators
        """
        if cross_data.empty:
            return spy_daily

        print("\n[CROSS-ASSETS] Engineering features...")

        features = spy_daily.copy()
        returns = cross_data.pct_change()

        for asset in returns.columns:
            asset_ret = returns[asset]

            # Match dates
            asset_features = pd.DataFrame(index=asset_ret.index)

            # 1. Daily return
            asset_features[f"{asset}_return"] = asset_ret

            # 2. 5-day return
            asset_features[f"{asset}_return_5d"] = asset_ret.rolling(5).sum()

            # 3. 20-day volatility
            asset_features[f"{asset}_vol_20d"] = asset_ret.rolling(20).std()

            # 4. 20-day momentum
            asset_features[f"{asset}_mom_20d"] = cross_data[asset].pct_change(20)

            # 5. RSI-like (simplified)
            delta = asset_ret.copy()
            gain = delta.where(delta > 0, 0).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            asset_features[f"{asset}_rsi"] = 100 - (100 / (1 + gain / (loss + 1e-10)))

            # Convert index to date (as Timestamp for consistent merge)
            asset_features["date"] = pd.to_datetime(asset_features.index.date)

            # Merge with spy_daily
            features = features.merge(
                asset_features.reset_index(drop=True),
                on="date",
                how="left"
            )

        # Fill NaN with 0 for cross-asset features
        cross_cols = [c for c in features.columns if any(a in c for a in self.assets)]
        features[cross_cols] = features[cross_cols].fillna(0)

        print(f"  Added {len(cross_cols)} cross-asset features")
        return features


# ═══════════════════════════════════════════════════════════════════════════════
# 3B. MAG MARKET BREADTH FEATURES (MAG3, MAG5, MAG6, MAG7, MAG10, MAG15)
# ═══════════════════════════════════════════════════════════════════════════════
class Mag7BreadthFeatures:
    """
    Market breadth features for various MAG (Magnificent) groupings.

    MAG3:  AAPL, MSFT, NVDA (Big 3 tech leaders, ~20% of S&P)
    MAG5:  AAPL, MSFT, NVDA, GOOGL, AMZN (Top 5 by market cap)
    MAG6:  MAG5 + META (Core tech mega-caps)
    MAG7:  MAG6 + TSLA (Magnificent 7)
    MAG10: MAG7 + BRK.B, UNH, XOM (Top 10 S&P weights)
    MAG15: MAG10 + JNJ, V, JPM, PG, MA (Top 15 S&P weights)

    These mega-caps drive ~35-40% of S&P 500 movement, so tracking them
    specifically provides valuable market leadership signals.

    Features for each group:
      - % advancing (breadth)
      - % at 52-week high/low
      - Average and weighted momentum
      - Breadth divergence from SPY
      - Relative strength vs SPY
      - Sector rotation signals
    """

    # MAG groupings (ordered by market cap, largest first)
    MAG3 = ["AAPL", "MSFT", "NVDA"]
    MAG5 = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
    MAG6 = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META"]
    MAG7 = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA"]
    MAG10 = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK.B", "UNH", "XOM"]
    MAG15 = ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "BRK.B", "UNH", "XOM",
             "JNJ", "V", "JPM", "PG", "MA"]

    # All unique tickers to download
    ALL_MAG_TICKERS = list(set(MAG15))

    # Approximate market cap weights (billions, for weighting) - Updated 2026
    MAG_WEIGHTS = {
        "AAPL": 3200, "MSFT": 3000, "NVDA": 2000, "GOOGL": 1900, "AMZN": 1800,
        "META": 1200, "TSLA": 900, "BRK.B": 800, "UNH": 550, "XOM": 500,
        "JNJ": 400, "V": 480, "JPM": 550, "PG": 380, "MA": 420,
    }

    # Tech vs Non-Tech classification for rotation analysis
    TECH_MAGS = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA"]
    NON_TECH_MAGS = ["BRK.B", "UNH", "XOM", "JNJ", "V", "JPM", "PG", "MA"]

    def __init__(self):
        self.data_cache = {}

    def download_mag_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Download MAG15 price data via Alpaca (covers all MAG groupings)."""
        print("\n[MAG] Downloading MAG3/5/6/7/10/15 data via Alpaca...")

        try:
            helper = get_alpaca_helper()

            # Download all MAG15 tickers (covers all smaller groups)
            tickers = self.ALL_MAG_TICKERS.copy()

            data = helper.download_daily_bars(tickers, start_date, end_date)

            if isinstance(data, dict):
                close = data.get("close", pd.DataFrame())
                high = data.get("high", pd.DataFrame())
                low = data.get("low", pd.DataFrame())
            else:
                print("  [WARN] Unexpected data format from Alpaca")
                return pd.DataFrame()

            if close.empty:
                print("  [WARN] No data returned from Alpaca")
                return pd.DataFrame()

            print(f"  Downloaded {len(close.columns)} MAG stocks, {len(close)} days")

            self.data_cache = {
                "close": close,
                "high": high,
                "low": low,
            }
            return close

        except Exception as e:
            print(f"  [ERROR] Failed to download MAG data via Alpaca: {e}")
            return pd.DataFrame()

    def create_mag_features(
        self,
        spy_daily: pd.DataFrame,
        lookback_52w: int = 252,
    ) -> pd.DataFrame:
        """
        Create MAG7/MAG10 breadth features.

        Features created:
          - mag7_pct_advancing: % of MAG7 green today
          - mag10_pct_advancing: % of MAG10 green today
          - mag7_avg_return: Average return across MAG7
          - mag10_avg_return: Average return across MAG10
          - mag7_wtd_return: Market-cap weighted return
          - mag10_wtd_return: Market-cap weighted return
          - mag7_at_52w_high: % near 52-week high
          - mag7_at_52w_low: % near 52-week low
          - mag7_momentum_5d: 5-day momentum
          - mag7_momentum_20d: 20-day momentum
          - mag7_breadth_divergence: Divergence from SPY
          - mag_tech_vs_nontch: Tech MAG vs non-tech MAG performance
        """
        if not self.data_cache:
            print("  [WARN] No MAG data cached. Call download_mag_data first.")
            return spy_daily

        close = self.data_cache.get("close", pd.DataFrame())
        high = self.data_cache.get("high", pd.DataFrame())
        low = self.data_cache.get("low", pd.DataFrame())

        if close.empty:
            return spy_daily

        print("\n[MAG] Engineering MAG3/5/6/7/10/15 breadth features...")

        features = spy_daily.copy()
        returns = close.pct_change()

        # Define all MAG groups with their tickers
        mag_groups = {
            "mag3": self.MAG3,
            "mag5": self.MAG5,
            "mag6": self.MAG6,
            "mag7": self.MAG7,
            "mag10": self.MAG10,
            "mag15": self.MAG15,
        }

        # Get available columns for each group
        mag_cols_map = {name: [c for c in tickers if c in returns.columns]
                        for name, tickers in mag_groups.items()}

        # Compute normalized weights for each group
        mag_norm_weights = {}
        for name, cols in mag_cols_map.items():
            weights = {k: v for k, v in self.MAG_WEIGHTS.items() if k in cols}
            total = sum(weights.values()) if weights else 1
            mag_norm_weights[name] = {k: v / total for k, v in weights.items()}

        mag_features = pd.DataFrame(index=returns.index)

        # 52-week high/low calculations (shared)
        rolling_high = high.rolling(lookback_52w).max()
        rolling_low = low.rolling(lookback_52w).min()
        pct_from_high = (close - rolling_high) / rolling_high
        pct_from_low = (close - rolling_low) / rolling_low
        near_high = pct_from_high > -0.05  # Within 5% of 52w high
        near_low = pct_from_low < 0.05     # Within 5% of 52w low

        # Generate features for each MAG group
        for name, cols in mag_cols_map.items():
            if len(cols) == 0:
                continue

            prefix = name  # e.g., "mag3", "mag5", etc.

            # 1. % Advancing (breadth)
            mag_features[f"{prefix}_pct_advancing"] = (returns[cols] > 0).mean(axis=1)

            # 2. Average return (equal weight)
            mag_features[f"{prefix}_avg_return"] = returns[cols].mean(axis=1)

            # 3. Market-cap weighted return
            norm_w = mag_norm_weights[name]
            wtd_ret = sum(returns.get(t, 0) * norm_w.get(t, 0) for t in cols)
            mag_features[f"{prefix}_wtd_return"] = wtd_ret

            # 4. 52-week high/low proximity
            mag_features[f"{prefix}_pct_near_52w_high"] = near_high[cols].mean(axis=1)
            mag_features[f"{prefix}_pct_near_52w_low"] = near_low[cols].mean(axis=1)

            # 5. Momentum (5-day and 20-day)
            mag_features[f"{prefix}_momentum_5d"] = returns[cols].rolling(5).sum().mean(axis=1)
            mag_features[f"{prefix}_momentum_20d"] = returns[cols].rolling(20).sum().mean(axis=1)

            # 6. Volatility (20-day)
            mag_features[f"{prefix}_volatility_20d"] = returns[cols].rolling(20).std().mean(axis=1)

            # 7. Relative strength (cumulative 20d return)
            mag_features[f"{prefix}_rel_strength_20d"] = returns[cols].rolling(20).sum().mean(axis=1)

            # 8. Streak features (consecutive advancing/declining days)
            advancing = (returns[cols] > 0).all(axis=1)
            declining = (returns[cols] < 0).all(axis=1)
            mag_features[f"{prefix}_all_advancing"] = advancing.astype(int)
            mag_features[f"{prefix}_all_declining"] = declining.astype(int)

        # Tech vs Non-Tech MAG rotation analysis
        tech_cols = [c for c in self.TECH_MAGS if c in returns.columns]
        non_tech_cols = [c for c in self.NON_TECH_MAGS if c in returns.columns]

        if len(tech_cols) > 0 and len(non_tech_cols) > 0:
            tech_return = returns[tech_cols].mean(axis=1)
            non_tech_return = returns[non_tech_cols].mean(axis=1)
            mag_features["mag_tech_vs_nontech"] = tech_return - non_tech_return
            mag_features["mag_tech_vs_nontech_5d"] = mag_features["mag_tech_vs_nontech"].rolling(5).sum()
            mag_features["mag_tech_vs_nontech_20d"] = mag_features["mag_tech_vs_nontech"].rolling(20).sum()

            # Tech leadership indicator
            mag_features["mag_tech_leading"] = (mag_features["mag_tech_vs_nontech_5d"] > 0).astype(int)
        else:
            mag_features["mag_tech_vs_nontech"] = 0
            mag_features["mag_tech_vs_nontech_5d"] = 0
            mag_features["mag_tech_vs_nontech_20d"] = 0
            mag_features["mag_tech_leading"] = 0

        # Cross-MAG breadth comparison (larger group vs smaller group)
        if "mag7_pct_advancing" in mag_features.columns and "mag3_pct_advancing" in mag_features.columns:
            # MAG3 vs MAG7 divergence (big 3 leading/lagging the MAG7)
            mag_features["mag3_vs_mag7_breadth"] = (
                mag_features["mag3_pct_advancing"] - mag_features["mag7_pct_advancing"]
            )

        if "mag15_pct_advancing" in mag_features.columns and "mag7_pct_advancing" in mag_features.columns:
            # MAG7 vs MAG15 divergence (core tech vs broader mega-caps)
            mag_features["mag7_vs_mag15_breadth"] = (
                mag_features["mag7_pct_advancing"] - mag_features["mag15_pct_advancing"]
            )

        # Concentration risk indicator (top 3 driving all returns)
        if "mag3_wtd_return" in mag_features.columns and "mag7_wtd_return" in mag_features.columns:
            mag3_wt = mag_features["mag3_wtd_return"]
            mag7_wt = mag_features["mag7_wtd_return"]
            # If MAG3 return > 80% of MAG7 return, concentration is high
            mag_features["mag_concentration_risk"] = (
                (mag3_wt.abs() > 0.8 * mag7_wt.abs()) & (mag7_wt.abs() > 0.001)
            ).astype(int)

        # SPY correlation (rolling 20-day) for key groups
        if "day_return" in features.columns:
            spy_ret = features.set_index("date")["day_return"] if "date" in features.columns else features["day_return"]
            try:
                for name in ["mag3", "mag7", "mag10", "mag15"]:
                    if f"{name}_avg_return" in mag_features.columns:
                        mag_features[f"{name}_spy_corr_20d"] = (
                            mag_features[f"{name}_avg_return"].rolling(20).corr(spy_ret)
                        )
            except (KeyError, ValueError):
                pass  # Correlation calc may fail on misaligned indices

        # Breadth divergence (MAG advancing but SPY flat/down)
        mag_features["date"] = pd.to_datetime(mag_features.index.date)

        # Merge with SPY daily
        features = features.merge(
            mag_features.reset_index(drop=True),
            on="date",
            how="left"
        )

        # Compute divergence after merge
        if "day_return" in features.columns:
            spy_direction = (features["day_return"] > 0).astype(int)
            for name in ["mag3", "mag7", "mag10", "mag15"]:
                col = f"{name}_pct_advancing"
                if col in features.columns:
                    mag_direction = (features[col] > 0.5).astype(int)
                    features[f"{name}_breadth_divergence"] = mag_direction - spy_direction

        # Fill NaN
        all_mag_cols = [c for c in features.columns if "mag" in c.lower()]
        features[all_mag_cols] = features[all_mag_cols].fillna(0)

        print(f"  Added {len(all_mag_cols)} MAG3/5/6/7/10/15 breadth features")
        return features

    def analyze_mag_leadership(self, features: pd.DataFrame) -> Dict:
        """
        Analyze current MAG7/MAG10 leadership conditions.

        Returns signal interpretation.
        """
        if len(features) == 0:
            return {}

        latest = features.iloc[-1]

        signal = {
            "date": latest.get("date"),
            # All MAG group breadths
            "mag3_advancing": latest.get("mag3_pct_advancing", 0),
            "mag5_advancing": latest.get("mag5_pct_advancing", 0),
            "mag6_advancing": latest.get("mag6_pct_advancing", 0),
            "mag7_advancing": latest.get("mag7_pct_advancing", 0),
            "mag10_advancing": latest.get("mag10_pct_advancing", 0),
            "mag15_advancing": latest.get("mag15_pct_advancing", 0),
            # Momentum
            "mag3_momentum_5d": latest.get("mag3_momentum_5d", 0),
            "mag7_momentum_5d": latest.get("mag7_momentum_5d", 0),
            "mag15_momentum_5d": latest.get("mag15_momentum_5d", 0),
            # 52-week position
            "mag7_near_52w_high": latest.get("mag7_pct_near_52w_high", 0),
            # Rotation
            "tech_vs_nontech": latest.get("mag_tech_vs_nontech_20d", 0),
            "tech_leading": latest.get("mag_tech_leading", 0),
            # Concentration
            "concentration_risk": latest.get("mag_concentration_risk", 0),
            # Cross-group divergence
            "mag3_vs_mag7_breadth": latest.get("mag3_vs_mag7_breadth", 0),
            "mag7_vs_mag15_breadth": latest.get("mag7_vs_mag15_breadth", 0),
        }

        # Interpretation based on MAG7 (core indicator)
        if signal["mag7_advancing"] >= 0.7 and signal["mag7_momentum_5d"] > 0.02:
            signal["interpretation"] = "STRONG_MAG_LEADERSHIP"
            signal["bias"] = "BULLISH"
        elif signal["mag7_advancing"] >= 0.5 and signal["mag7_momentum_5d"] > 0:
            signal["interpretation"] = "MODERATE_MAG_LEADERSHIP"
            signal["bias"] = "SLIGHTLY_BULLISH"
        elif signal["mag7_advancing"] <= 0.3 and signal["mag7_momentum_5d"] < -0.02:
            signal["interpretation"] = "MAG_WEAKNESS"
            signal["bias"] = "BEARISH"
        elif signal["mag7_advancing"] <= 0.5 and signal["mag7_momentum_5d"] < 0:
            signal["interpretation"] = "MODERATE_MAG_WEAKNESS"
            signal["bias"] = "SLIGHTLY_BEARISH"
        else:
            signal["interpretation"] = "NEUTRAL_MAG"
            signal["bias"] = "NEUTRAL"

        # Tech rotation signal
        if signal["tech_vs_nontech"] > 0.05:
            signal["rotation"] = "TECH_OUTPERFORMING"
        elif signal["tech_vs_nontech"] < -0.05:
            signal["rotation"] = "TECH_UNDERPERFORMING"
        else:
            signal["rotation"] = "NO_CLEAR_ROTATION"

        # Breadth divergence warning (big 3 leading but rest lagging)
        if signal["mag3_vs_mag7_breadth"] > 0.3:
            signal["breadth_warning"] = "BIG3_LEADING_NARROWLY"
        elif signal["mag7_vs_mag15_breadth"] > 0.3:
            signal["breadth_warning"] = "CORE_TECH_LEADING_NARROWLY"
        else:
            signal["breadth_warning"] = "NONE"

        # Concentration warning
        if signal["concentration_risk"] == 1:
            signal["concentration_warning"] = "HIGH_CONCENTRATION_IN_TOP3"
        else:
            signal["concentration_warning"] = "NONE"

        return signal


# ═══════════════════════════════════════════════════════════════════════════════
# 3C. SECTOR BREADTH FEATURES (S&P 500 Sectors)
# ═══════════════════════════════════════════════════════════════════════════════
class SectorBreadthFeatures:
    """
    Market breadth features by S&P 500 sector for validation.

    Tracks sector rotation, leadership, and divergence signals using sector ETFs.
    This provides additional validation dimensions beyond individual stocks.

    Sector ETFs:
      XLK - Technology
      XLF - Financials
      XLV - Healthcare
      XLE - Energy
      XLI - Industrials
      XLY - Consumer Discretionary
      XLP - Consumer Staples
      XLU - Utilities
      XLB - Materials
      XLRE - Real Estate
      XLC - Communication Services
    """

    SECTOR_ETFS = {
        "XLK": "Technology",
        "XLF": "Financials",
        "XLV": "Healthcare",
        "XLE": "Energy",
        "XLI": "Industrials",
        "XLY": "Consumer Discretionary",
        "XLP": "Consumer Staples",
        "XLU": "Utilities",
        "XLB": "Materials",
        "XLRE": "Real Estate",
        "XLC": "Communication Services",
    }

    # Defensive vs Cyclical classification
    DEFENSIVE_SECTORS = ["XLV", "XLP", "XLU", "XLRE"]
    CYCLICAL_SECTORS = ["XLK", "XLF", "XLE", "XLI", "XLY", "XLB", "XLC"]

    # Risk-on vs Risk-off classification
    RISK_ON_SECTORS = ["XLK", "XLY", "XLF", "XLE", "XLI"]
    RISK_OFF_SECTORS = ["XLV", "XLP", "XLU", "XLRE"]

    def __init__(self):
        self.data_cache = {}

    def download_sector_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """Download sector ETF data via Alpaca."""
        print("\n[SECTORS] Downloading sector ETF data via Alpaca...")

        try:
            helper = get_alpaca_helper()
            tickers = list(self.SECTOR_ETFS.keys())

            data = helper.download_daily_bars(tickers, start_date, end_date)

            if isinstance(data, dict):
                close = data.get("close", pd.DataFrame())
                volume = data.get("volume", pd.DataFrame())
            else:
                print("  [WARN] Unexpected data format from Alpaca")
                return pd.DataFrame()

            if close.empty:
                print("  [WARN] No sector data returned from Alpaca")
                return pd.DataFrame()

            print(f"  Downloaded {len(close.columns)} sector ETFs, {len(close)} days")

            self.data_cache = {
                "close": close,
                "volume": volume,
            }
            return close

        except Exception as e:
            print(f"  [ERROR] Failed to download sector data: {e}")
            return pd.DataFrame()

    def create_sector_features(
        self,
        spy_daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """Create sector breadth features for validation."""
        if not self.data_cache:
            print("  [WARN] No sector data cached. Call download_sector_data first.")
            return spy_daily

        close = self.data_cache.get("close", pd.DataFrame())
        volume = self.data_cache.get("volume", pd.DataFrame())

        if close.empty:
            return spy_daily

        print("\n[SECTORS] Engineering sector breadth features...")

        features = spy_daily.copy()
        returns = close.pct_change()

        # Get available sectors
        available_sectors = [c for c in self.SECTOR_ETFS.keys() if c in returns.columns]
        defensive_cols = [c for c in self.DEFENSIVE_SECTORS if c in returns.columns]
        cyclical_cols = [c for c in self.CYCLICAL_SECTORS if c in returns.columns]
        risk_on_cols = [c for c in self.RISK_ON_SECTORS if c in returns.columns]
        risk_off_cols = [c for c in self.RISK_OFF_SECTORS if c in returns.columns]

        sector_features = pd.DataFrame(index=returns.index)

        # 1. Overall sector breadth
        sector_features["sector_pct_advancing"] = (returns[available_sectors] > 0).mean(axis=1)
        sector_features["sector_avg_return"] = returns[available_sectors].mean(axis=1)

        # 2. Defensive vs Cyclical rotation
        if len(defensive_cols) > 0 and len(cyclical_cols) > 0:
            defensive_ret = returns[defensive_cols].mean(axis=1)
            cyclical_ret = returns[cyclical_cols].mean(axis=1)
            sector_features["sector_cyclical_vs_defensive"] = cyclical_ret - defensive_ret
            sector_features["sector_cyclical_vs_defensive_5d"] = sector_features["sector_cyclical_vs_defensive"].rolling(5).sum()
            sector_features["sector_cyclical_vs_defensive_20d"] = sector_features["sector_cyclical_vs_defensive"].rolling(20).sum()

        # 3. Risk-on vs Risk-off rotation
        if len(risk_on_cols) > 0 and len(risk_off_cols) > 0:
            risk_on_ret = returns[risk_on_cols].mean(axis=1)
            risk_off_ret = returns[risk_off_cols].mean(axis=1)
            sector_features["sector_risk_on_vs_off"] = risk_on_ret - risk_off_ret
            sector_features["sector_risk_appetite_5d"] = sector_features["sector_risk_on_vs_off"].rolling(5).sum()
            sector_features["sector_risk_appetite_20d"] = sector_features["sector_risk_on_vs_off"].rolling(20).sum()

        # 4. Leading/Lagging sectors
        sector_features["sector_best_return"] = returns[available_sectors].max(axis=1)
        sector_features["sector_worst_return"] = returns[available_sectors].min(axis=1)
        sector_features["sector_dispersion"] = sector_features["sector_best_return"] - sector_features["sector_worst_return"]

        # 5. Momentum for key sectors
        for sector in ["XLK", "XLF", "XLE", "XLV"]:
            if sector in returns.columns:
                sector_features[f"{sector.lower()}_momentum_5d"] = returns[sector].rolling(5).sum()
                sector_features[f"{sector.lower()}_momentum_20d"] = returns[sector].rolling(20).sum()
                sector_features[f"{sector.lower()}_rel_strength"] = (
                    returns[sector].rolling(20).sum() - sector_features["sector_avg_return"].rolling(20).sum()
                )

        # 6. Sector breadth divergence
        sector_features["sector_breadth_strong"] = (sector_features["sector_pct_advancing"] >= 0.7).astype(int)
        sector_features["sector_breadth_weak"] = (sector_features["sector_pct_advancing"] <= 0.3).astype(int)

        # 7. Volume-weighted sector strength (if volume available)
        if not volume.empty:
            vol_cols = [c for c in available_sectors if c in volume.columns]
            if len(vol_cols) > 0:
                vol_weights = volume[vol_cols].div(volume[vol_cols].sum(axis=1), axis=0)
                sector_features["sector_vol_wtd_return"] = (returns[vol_cols] * vol_weights).sum(axis=1)

        # Add date for merge
        sector_features["date"] = pd.to_datetime(sector_features.index.date)

        # Merge with SPY daily
        features = features.merge(
            sector_features.reset_index(drop=True),
            on="date",
            how="left"
        )

        # Compute divergence after merge
        if "day_return" in features.columns:
            spy_direction = (features["day_return"] > 0).astype(int)
            sector_direction = (features["sector_pct_advancing"] > 0.5).astype(int)
            features["sector_breadth_divergence"] = sector_direction - spy_direction

        # Fill NaN
        sector_cols = [c for c in features.columns if "sector_" in c.lower() or c.startswith("xl")]
        features[sector_cols] = features[sector_cols].fillna(0)

        print(f"  Added {len(sector_cols)} sector breadth features")
        return features

    def analyze_sector_rotation(self, features: pd.DataFrame) -> Dict:
        """Analyze current sector rotation signals."""
        if len(features) == 0:
            return {}

        latest = features.iloc[-1].to_dict()

        signal = {
            "date": latest.get("date"),
            "sector_advancing": latest.get("sector_pct_advancing", 0),
            "cyclical_vs_defensive": latest.get("sector_cyclical_vs_defensive_5d", 0),
            "risk_appetite": latest.get("sector_risk_appetite_5d", 0),
            "dispersion": latest.get("sector_dispersion", 0),
        }

        # Rotation interpretation
        if signal["cyclical_vs_defensive"] > 0.02:
            signal["rotation"] = "CYCLICAL_LEADING"
            signal["market_phase"] = "EXPANSION"
        elif signal["cyclical_vs_defensive"] < -0.02:
            signal["rotation"] = "DEFENSIVE_LEADING"
            signal["market_phase"] = "CONTRACTION"
        else:
            signal["rotation"] = "NEUTRAL"
            signal["market_phase"] = "TRANSITION"

        # Risk appetite
        if signal["risk_appetite"] > 0.02:
            signal["risk_sentiment"] = "RISK_ON"
        elif signal["risk_appetite"] < -0.02:
            signal["risk_sentiment"] = "RISK_OFF"
        else:
            signal["risk_sentiment"] = "NEUTRAL"

        return signal


# ═══════════════════════════════════════════════════════════════════════════════
# 4. HYPERPARAMETER STABILITY ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
class StabilityAnalyzer:
    """
    Analyze hyperparameter stability to detect fragile solutions.

    A good solution should be on a "plateau" - small parameter changes
    should not drastically affect performance.
    """

    def __init__(self, perturbation_pct: float = 0.05):
        self.perturbation_pct = perturbation_pct

    def compute_stability_score(
        self,
        base_params: Dict,
        base_score: float,
        param_ranges: Dict[str, Tuple[float, float]],
        score_fn,
        n_samples: int = 20,
    ) -> Dict:
        """
        Compute stability score by perturbing parameters.

        Returns:
            Dict with stability_score (0-1) and per-parameter sensitivity
        """
        print("\n[STABILITY] Analyzing hyperparameter sensitivity...")

        sensitivities = {}
        all_scores = [base_score]

        for param, (low, high) in param_ranges.items():
            base_val = base_params.get(param)
            if base_val is None:
                continue

            param_scores = []

            # Sample perturbations
            for _ in range(n_samples // len(param_ranges)):
                # Random perturbation within +/- perturbation_pct
                perturbation = np.random.uniform(
                    -self.perturbation_pct,
                    self.perturbation_pct
                )
                new_val = base_val * (1 + perturbation)
                new_val = max(low, min(high, new_val))

                perturbed_params = base_params.copy()
                perturbed_params[param] = new_val

                try:
                    score = score_fn(perturbed_params)
                    param_scores.append(score)
                    all_scores.append(score)
                except (ValueError, TypeError, RuntimeError):
                    pass

            if len(param_scores) > 0:
                # Compute sensitivity for this parameter
                score_std = np.std(param_scores)
                score_change = np.mean(np.abs(np.array(param_scores) - base_score))
                sensitivities[param] = {
                    "std": score_std,
                    "avg_change": score_change,
                    "sensitivity": score_change / (base_score + 1e-6),
                }

        # Overall stability score (lower sensitivity = higher stability)
        if len(sensitivities) > 0:
            avg_sensitivity = np.mean([s["sensitivity"] for s in sensitivities.values()])
            stability_score = max(0, 1 - avg_sensitivity * 10)  # Scale factor
        else:
            stability_score = 0.5

        # Score variance across all perturbations
        score_variance = np.var(all_scores)

        results = {
            "stability_score": stability_score,
            "score_variance": score_variance,
            "base_score": base_score,
            "n_samples": len(all_scores),
            "per_param_sensitivity": sensitivities,
        }

        print(f"  Stability Score: {stability_score:.3f} (1.0 = very stable)")
        print(f"  Score Variance: {score_variance:.6f}")

        if stability_score < 0.5:
            print("  [WARN] Solution may be fragile (low stability)")

        return results

    def detect_plateau(
        self,
        optuna_study,
        top_n: int = 10,
    ) -> bool:
        """
        Check if best solutions form a plateau (similar scores, different params).

        A plateau indicates a robust solution.
        """
        trials = sorted(
            [t for t in optuna_study.trials if t.value is not None],
            key=lambda t: t.value,
            reverse=True
        )[:top_n]

        if len(trials) < 3:
            return False

        scores = [t.value for t in trials]
        score_range = max(scores) - min(scores)

        # If top solutions have similar scores, it's a plateau
        is_plateau = score_range < 0.01  # Within 1% AUC

        if is_plateau:
            print(f"  [GOOD] Detected plateau: top {top_n} solutions within {score_range:.4f} AUC")
        else:
            print(f"  [WARN] No plateau: top {top_n} solutions span {score_range:.4f} AUC")

        return is_plateau


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════
def integrate_anti_overfit(
    df_daily: pd.DataFrame,
    spy_1min: pd.DataFrame = None,
    use_synthetic: bool = True,
    use_cross_assets: bool = True,
    use_breadth_streaks: bool = True,
    use_mag_breadth: bool = True,  # MAG3/5/6/7/10/15 features
    use_sector_breadth: bool = True,  # Sector rotation features
    synthetic_weight: float = 0.4,  # Weight for synthetic data (real = 1 - synthetic)
) -> Tuple[pd.DataFrame, Dict]:
    """
    Integrate all anti-overfitting measures.

    Args:
        df_daily: Daily feature dataframe
        spy_1min: 1-minute SPY data (optional)
        use_synthetic: Generate synthetic SPY universes
        use_cross_assets: Add TLT, QQQ, GLD features
        use_breadth_streaks: Add component streak breadth features
        use_mag_breadth: Add MAG3/5/6/7/10/15 market breadth features
        use_sector_breadth: Add sector rotation and breadth features
        synthetic_weight: Weight for synthetic data (0.4 = 40% synthetic)

    Returns:
        Augmented dataframe and metadata
    """
    print("\n" + "=" * 70)
    print("ANTI-OVERFITTING INTEGRATION")
    print("=" * 70)

    # Convert date column to pd.Timestamp for consistent merging
    # (spy_daily may have datetime.date objects which don't merge with Timestamps)
    df_daily = df_daily.copy()
    df_daily["date"] = pd.to_datetime(df_daily["date"])

    metadata = {}
    start_date = pd.to_datetime(df_daily["date"].min())
    end_date = pd.to_datetime(df_daily["date"].max())

    # 1. Component Streak Breadth Features
    if use_breadth_streaks:
        streak_features = ComponentStreakFeatures(max_streak=7)
        component_prices = streak_features.download_component_data(start_date, end_date)

        if not component_prices.empty:
            df_daily = streak_features.compute_streak_features(component_prices, df_daily)
            metadata["streak_features"] = True
            metadata["components_tracked"] = len(component_prices.columns)

            # Analyze current breadth
            signal = streak_features.analyze_breadth_signal(df_daily)
            if signal:
                print(f"  Current breadth: {signal.get('interpretation', 'N/A')}")

    # 2. Cross-Asset Features
    if use_cross_assets:
        cross_assets = CrossAssetFeatures()
        cross_data = cross_assets.download_cross_assets(start_date, end_date)

        if not cross_data.empty:
            df_daily = cross_assets.create_cross_asset_features(cross_data, df_daily)
            metadata["cross_assets"] = list(cross_data.columns)

    # 3. MAG Market Breadth Features (MAG3/5/6/7/10/15)
    if use_mag_breadth:
        mag_breadth = Mag7BreadthFeatures()
        mag_data = mag_breadth.download_mag_data(start_date, end_date)

        if not mag_data.empty:
            df_daily = mag_breadth.create_mag_features(df_daily)
            metadata["mag_features"] = True
            metadata["mag_tickers"] = mag_breadth.ALL_MAG_TICKERS
            metadata["mag_groups"] = ["MAG3", "MAG5", "MAG6", "MAG7", "MAG10", "MAG15"]

            # Analyze current MAG leadership
            mag_signal = mag_breadth.analyze_mag_leadership(df_daily)
            if mag_signal:
                print(f"  MAG7 Leadership: {mag_signal.get('interpretation', 'N/A')}")
                print(f"    Bias: {mag_signal.get('bias', 'N/A')}")
                print(f"    Tech Rotation: {mag_signal.get('rotation', 'N/A')}")
                if mag_signal.get('breadth_warning') != 'NONE':
                    print(f"    Breadth Warning: {mag_signal.get('breadth_warning')}")

    # 4. Sector Breadth Features (S&P 500 Sectors)
    if use_sector_breadth:
        sector_breadth = SectorBreadthFeatures()
        sector_data = sector_breadth.download_sector_data(start_date, end_date)

        if not sector_data.empty:
            df_daily = sector_breadth.create_sector_features(df_daily)
            metadata["sector_features"] = True
            metadata["sector_etfs"] = list(sector_breadth.SECTOR_ETFS.keys())

            # Analyze current sector rotation
            sector_signal = sector_breadth.analyze_sector_rotation(df_daily)
            if sector_signal:
                print(f"  Sector Rotation: {sector_signal.get('rotation', 'N/A')}")
                print(f"    Market Phase: {sector_signal.get('market_phase', 'N/A')}")
                print(f"    Risk Sentiment: {sector_signal.get('risk_sentiment', 'N/A')}")

    # 5. Synthetic SPY Universes (do last since it multiplies data)
    if use_synthetic:
        real_weight = 1 - synthetic_weight
        synth_gen = SyntheticSPYGenerator(n_universes=10, real_weight=real_weight)

        # Reuse component prices if available, otherwise download
        if "component_prices" not in dir() or component_prices.empty:
            component_prices = synth_gen.download_spy_components(start_date, end_date)

        if not component_prices.empty:
            # Generate filter-based universes
            universes = synth_gen.generate_universes(component_prices, df_daily)

            # Generate "SPY minus component" universes (new feature)
            component_universes = synth_gen.generate_spy_minus_component_universes(
                component_prices, df_daily
            )
            universes.extend(component_universes)

            df_daily = synth_gen.create_augmented_dataset(df_daily, universes)
            metadata["n_universes"] = len(universes)
            metadata["real_weight"] = real_weight
            metadata["synthetic_weight"] = synthetic_weight
            metadata["component_modified_universes"] = len(component_universes)

    print(f"\n[ANTI-OVERFIT] Final dataset: {len(df_daily)} samples")
    print(f"  Metadata: {metadata}")

    return df_daily, metadata


def compute_weighted_evaluation(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    returns: np.ndarray,
    cv_scores: List[float],
    n_features: int,
    hp_sensitivity: float = 0.0,
    noise_scores: List[float] = None,
) -> Dict[str, float]:
    """
    Convenience function to compute WMES.

    Returns comprehensive evaluation metrics.
    """
    evaluator = WeightedModelEvaluator()
    return evaluator.compute_wmes(
        y_true, y_pred, y_proba, returns,
        cv_scores, n_features, hp_sensitivity, noise_scores
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. ROBUSTNESS ENSEMBLE (DIMENSION + PARAMETER PERTURBATION)
# ═══════════════════════════════════════════════════════════════════════════════
class RobustnessEnsemble:
    """
    Creates an ensemble of models trained with perturbed dimensions and parameters.

    Strategy:
      1. Train base model with optimal n_dimensions and parameters
      2. Train "adjacent" models with n-1 and n+1 dimensions
      3. Train models with perturbed parameters (+/- noise)
      4. Ensemble all models with optional weighting
      5. If ensemble performance drops drastically, solution is fragile

    This reduces overfitting to specific dimension counts or parameter values.
    """

    def __init__(
        self,
        n_dimension_variants: int = 2,  # +/- this many dimensions
        n_param_variants: int = 2,  # Number of parameter perturbations
        param_noise_pct: float = 0.05,  # +/- 5% parameter noise
        center_weight: float = 0.5,  # Weight for center/optimal model
        adjacent_weight: float = 0.25,  # Weight for adjacent models (split)
    ):
        self.n_dimension_variants = n_dimension_variants
        self.n_param_variants = n_param_variants
        self.param_noise_pct = param_noise_pct
        self.center_weight = center_weight
        self.adjacent_weight = adjacent_weight

        self.models = {}  # Store all trained models
        self.weights = {}  # Store model weights
        self.fragility_score = None

    def create_dimension_variants(
        self,
        optimal_dims: int,
        min_dims: int = 5,
        max_dims: int = 100,
    ) -> List[int]:
        """
        Create list of dimension counts to try.

        Returns: [n-2, n-1, n, n+1, n+2] (within bounds)
        """
        variants = []

        for delta in range(-self.n_dimension_variants, self.n_dimension_variants + 1):
            n = optimal_dims + delta
            if min_dims <= n <= max_dims:
                variants.append(n)

        # Ensure we have at least 3 variants
        if len(variants) < 3:
            variants = [max(min_dims, optimal_dims - 1), optimal_dims, min(max_dims, optimal_dims + 1)]

        return sorted(set(variants))

    def create_parameter_variants(
        self,
        base_params: Dict,
        param_ranges: Dict[str, Tuple[float, float]] = None,
    ) -> List[Dict]:
        """
        Create parameter variants with noise.

        Returns list of parameter dicts including base + perturbed versions.
        """
        variants = [base_params.copy()]  # Always include base

        if param_ranges is None:
            # Default ranges for common parameters
            param_ranges = {
                "C": (0.01, 100.0),
                "l2_C": (0.01, 100.0),
                "n_estimators": (10, 500),
                "gb_n_estimators": (10, 500),
                "max_depth": (1, 10),
                "gb_max_depth": (1, 10),
                "learning_rate": (0.001, 1.0),
                "gb_learning_rate": (0.001, 1.0),
                "min_samples_leaf": (1, 200),
                "gb_min_samples_leaf": (1, 200),
            }

        for _ in range(self.n_param_variants):
            perturbed = base_params.copy()

            for param, value in base_params.items():
                if param not in param_ranges:
                    continue

                low, high = param_ranges[param]

                # Add random noise within +/- noise_pct
                noise = np.random.uniform(-self.param_noise_pct, self.param_noise_pct)
                new_value = value * (1 + noise)

                # Clip to valid range
                new_value = max(low, min(high, new_value))

                # Keep integers as integers
                if isinstance(value, int):
                    new_value = int(round(new_value))

                perturbed[param] = new_value

            variants.append(perturbed)

        return variants

    def train_ensemble(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weights: np.ndarray = None,
        base_params: Dict = None,
        dim_reduction_fn=None,
        optimal_dims: int = 30,
        model_class=None,
        cv_folds: int = 3,
    ) -> Dict:
        """
        Train ensemble of models with dimension and parameter perturbations.

        Args:
            X: Feature matrix
            y: Target labels
            sample_weights: Sample weights
            base_params: Optimal parameters found via Optuna
            dim_reduction_fn: Function that reduces X to n dimensions
            optimal_dims: Optimal number of dimensions
            model_class: Sklearn model class to use
            cv_folds: Cross-validation folds for scoring

        Returns:
            Dict with models, weights, and fragility analysis
        """
        from sklearn.model_selection import cross_val_score
        from sklearn.linear_model import LogisticRegression

        if model_class is None:
            model_class = LogisticRegression

        if base_params is None:
            base_params = {"C": 1.0, "max_iter": 500, "random_state": 42}

        print("\n" + "=" * 70)
        print("ROBUSTNESS ENSEMBLE TRAINING")
        print("=" * 70)

        results = {
            "models": {},
            "scores": {},
            "weights": {},
            "dim_variants": [],
            "param_variants": [],
        }

        # ─────────────────────────────────────────────────────────────────────
        # 1. DIMENSION VARIANTS
        # ─────────────────────────────────────────────────────────────────────
        dim_variants = self.create_dimension_variants(optimal_dims)
        results["dim_variants"] = dim_variants
        print(f"\n[DIM VARIANTS] Testing dimensions: {dim_variants}")

        dim_scores = {}

        for n_dims in dim_variants:
            try:
                # Reduce dimensions if function provided
                if dim_reduction_fn is not None:
                    X_reduced = dim_reduction_fn(X, n_dims)
                else:
                    # Simple PCA fallback
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=min(n_dims, X.shape[1]))
                    X_reduced = pca.fit_transform(X)

                # Train model with base params
                model_params = {k: v for k, v in base_params.items()
                               if k in ["C", "max_iter", "random_state", "penalty", "solver"]}
                model = model_class(**model_params)

                # Cross-validate
                # Note: fit_params removed in sklearn 1.4+, use params instead
                cv_kwargs = {"estimator": model, "X": X_reduced, "y": y, "cv": cv_folds, "scoring": "roc_auc"}
                if sample_weights is not None:
                    cv_kwargs["params"] = {"sample_weight": sample_weights}
                scores = cross_val_score(**cv_kwargs)
                mean_score = scores.mean()
                dim_scores[n_dims] = mean_score

                # Train final model on full data
                model.fit(X_reduced, y, sample_weight=sample_weights)

                model_key = f"dim_{n_dims}"
                results["models"][model_key] = {
                    "model": model,
                    "n_dims": n_dims,
                    "cv_score": mean_score,
                    "type": "dimension_variant",
                }
                results["scores"][model_key] = mean_score

                is_optimal = " (OPTIMAL)" if n_dims == optimal_dims else ""
                print(f"  dim={n_dims}: AUC={mean_score:.4f}{is_optimal}")

            except Exception as e:
                print(f"  dim={n_dims}: FAILED - {e}")
                continue

        # ─────────────────────────────────────────────────────────────────────
        # 2. PARAMETER VARIANTS (using optimal dimensions)
        # ─────────────────────────────────────────────────────────────────────
        param_variants = self.create_parameter_variants(base_params)
        results["param_variants"] = param_variants
        print(f"\n[PARAM VARIANTS] Testing {len(param_variants)} parameter sets")

        # Use optimal dimension reduction
        if dim_reduction_fn is not None:
            X_optimal = dim_reduction_fn(X, optimal_dims)
        else:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=min(optimal_dims, X.shape[1]))
            X_optimal = pca.fit_transform(X)

        for i, params in enumerate(param_variants):
            try:
                model_params = {k: v for k, v in params.items()
                               if k in ["C", "max_iter", "random_state", "penalty", "solver"]}
                model = model_class(**model_params)

                # Note: fit_params removed in sklearn 1.4+, use params instead
                cv_kwargs = {"estimator": model, "X": X_optimal, "y": y, "cv": cv_folds, "scoring": "roc_auc"}
                if sample_weights is not None:
                    cv_kwargs["params"] = {"sample_weight": sample_weights}
                scores = cross_val_score(**cv_kwargs)
                mean_score = scores.mean()

                model.fit(X_optimal, y, sample_weight=sample_weights)

                model_key = f"param_{i}"
                results["models"][model_key] = {
                    "model": model,
                    "params": params,
                    "cv_score": mean_score,
                    "type": "parameter_variant",
                }
                results["scores"][model_key] = mean_score

                is_base = " (BASE)" if i == 0 else ""
                print(f"  variant_{i}: AUC={mean_score:.4f}{is_base}")

            except Exception as e:
                print(f"  variant_{i}: FAILED - {e}")
                continue

        # ─────────────────────────────────────────────────────────────────────
        # 3. COMPUTE WEIGHTS
        # ─────────────────────────────────────────────────────────────────────
        self._compute_weights(results, optimal_dims)

        # ─────────────────────────────────────────────────────────────────────
        # 4. FRAGILITY ANALYSIS
        # ─────────────────────────────────────────────────────────────────────
        fragility = self._analyze_fragility(results, optimal_dims)
        results["fragility"] = fragility

        # Store for later use
        self.models = results["models"]
        self.weights = results["weights"]
        self.fragility_score = fragility["fragility_score"]

        print(f"\n[ENSEMBLE] Trained {len(results['models'])} models")
        print(f"  Fragility Score: {fragility['fragility_score']:.3f} (0=robust, 1=fragile)")

        if fragility["fragility_score"] > 0.3:
            print("  [WARN] High fragility detected - solution may be overfit!")
        else:
            print("  [GOOD] Low fragility - solution appears robust")

        return results

    def _compute_weights(self, results: Dict, optimal_dims: int):
        """Compute ensemble weights based on configuration and scores."""
        weights = {}

        # Dimension variant weights
        dim_models = {k: v for k, v in results["models"].items() if v["type"] == "dimension_variant"}

        if len(dim_models) > 0:
            # Optimal dim gets center_weight, others split adjacent_weight
            total_dim_weight = self.center_weight + self.adjacent_weight
            n_adjacent = len(dim_models) - 1

            for key, model_info in dim_models.items():
                if model_info["n_dims"] == optimal_dims:
                    weights[key] = self.center_weight / total_dim_weight
                else:
                    weights[key] = (self.adjacent_weight / n_adjacent) / total_dim_weight if n_adjacent > 0 else 0

        # Parameter variant weights (base gets higher weight)
        param_models = {k: v for k, v in results["models"].items() if v["type"] == "parameter_variant"}

        if len(param_models) > 0:
            total_param_weight = self.center_weight + self.adjacent_weight
            n_perturbed = len(param_models) - 1

            for i, (key, model_info) in enumerate(param_models.items()):
                if i == 0:  # Base params
                    weights[key] = self.center_weight / total_param_weight
                else:
                    weights[key] = (self.adjacent_weight / n_perturbed) / total_param_weight if n_perturbed > 0 else 0

        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {k: v / total_weight for k, v in weights.items()}

        results["weights"] = weights

    def _analyze_fragility(self, results: Dict, optimal_dims: int) -> Dict:
        """
        Analyze if the solution is fragile.

        Fragility indicators:
          - Large score drop with adjacent dimensions
          - Large score drop with parameter perturbation
          - High variance across variants
        """
        scores = list(results["scores"].values())

        if len(scores) < 2:
            return {
                "fragility_score": 0.5,
                "reason": "Not enough variants",
                "interpretation": "MODERATE - Insufficient variants for robust analysis"
            }

        # Get optimal model score
        optimal_key = f"dim_{optimal_dims}"
        base_key = "param_0"

        optimal_score = results["scores"].get(optimal_key, max(scores))
        base_score = results["scores"].get(base_key, optimal_score)
        best_score = max(optimal_score, base_score)

        # 1. Score variance across all variants
        score_variance = np.var(scores)
        score_std = np.std(scores)

        # 2. Max drop from optimal
        min_score = min(scores)
        max_drop = best_score - min_score

        # 3. Dimension sensitivity
        dim_scores = [v["cv_score"] for k, v in results["models"].items()
                     if v["type"] == "dimension_variant"]
        dim_sensitivity = np.std(dim_scores) if len(dim_scores) > 1 else 0

        # 4. Parameter sensitivity
        param_scores = [v["cv_score"] for k, v in results["models"].items()
                       if v["type"] == "parameter_variant"]
        param_sensitivity = np.std(param_scores) if len(param_scores) > 1 else 0

        # Compute fragility score (0 = robust, 1 = very fragile)
        # Normalize factors to 0-1 range
        variance_factor = min(score_variance * 100, 1.0)  # High variance = fragile
        drop_factor = min(max_drop / (best_score + 1e-6), 1.0)  # Large drop = fragile
        dim_factor = min(dim_sensitivity * 10, 1.0)  # High dim sensitivity = fragile
        param_factor = min(param_sensitivity * 10, 1.0)  # High param sensitivity = fragile

        fragility_score = (
            0.3 * variance_factor +
            0.3 * drop_factor +
            0.2 * dim_factor +
            0.2 * param_factor
        )

        fragility = {
            "fragility_score": fragility_score,
            "score_variance": score_variance,
            "score_std": score_std,
            "max_drop": max_drop,
            "best_score": best_score,
            "min_score": min_score,
            "dim_sensitivity": dim_sensitivity,
            "param_sensitivity": param_sensitivity,
            "interpretation": self._interpret_fragility(fragility_score),
        }

        return fragility

    def _interpret_fragility(self, score: float) -> str:
        """Interpret fragility score."""
        if score < 0.15:
            return "VERY_ROBUST - Solution is stable across perturbations"
        elif score < 0.25:
            return "ROBUST - Minor sensitivity, acceptable"
        elif score < 0.35:
            return "MODERATE - Some sensitivity, proceed with caution"
        elif score < 0.50:
            return "FRAGILE - Significant sensitivity, likely overfit"
        else:
            return "VERY_FRAGILE - Unstable solution, high overfit risk"

    def predict_ensemble(
        self,
        X: np.ndarray,
        dim_reduction_fn=None,
        use_proba: bool = True,
    ) -> np.ndarray:
        """
        Make predictions using the ensemble.

        Args:
            X: Feature matrix
            dim_reduction_fn: Function to reduce dimensions
            use_proba: Return probabilities (True) or hard predictions (False)

        Returns:
            Weighted ensemble predictions
        """
        if len(self.models) == 0:
            raise ValueError("No models trained. Call train_ensemble first.")

        predictions = []
        weights = []

        for key, model_info in self.models.items():
            model = model_info["model"]
            weight = self.weights.get(key, 1.0 / len(self.models))

            # Apply appropriate dimension reduction
            if model_info["type"] == "dimension_variant":
                n_dims = model_info["n_dims"]
                if dim_reduction_fn is not None:
                    X_reduced = dim_reduction_fn(X, n_dims)
                else:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=min(n_dims, X.shape[1]))
                    X_reduced = pca.fit_transform(X)
            else:
                # Parameter variants use optimal dims
                n_dims = self.models.get("dim_30", {}).get("n_dims", 30)
                if dim_reduction_fn is not None:
                    X_reduced = dim_reduction_fn(X, n_dims)
                else:
                    from sklearn.decomposition import PCA
                    pca = PCA(n_components=min(n_dims, X.shape[1]))
                    X_reduced = pca.fit_transform(X)

            try:
                if use_proba and hasattr(model, "predict_proba"):
                    pred = model.predict_proba(X_reduced)[:, 1]
                else:
                    pred = model.predict(X_reduced)

                predictions.append(pred)
                weights.append(weight)
            except Exception as e:
                print(f"  [WARN] Prediction failed for {key}: {e}")
                continue

        if len(predictions) == 0:
            raise ValueError("All model predictions failed")

        # Weighted average
        predictions = np.array(predictions)
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize

        ensemble_pred = np.average(predictions, axis=0, weights=weights)

        return ensemble_pred

    def evaluate_ensemble(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
        dim_reduction_fn=None,
    ) -> Dict:
        """
        Evaluate ensemble vs individual models.

        Returns comparison metrics.
        """
        from sklearn.metrics import roc_auc_score, accuracy_score

        results = {"individual": {}, "ensemble": {}}

        # Ensemble prediction
        ensemble_proba = self.predict_ensemble(X_test, dim_reduction_fn, use_proba=True)
        ensemble_pred = (ensemble_proba > 0.5).astype(int)

        try:
            ensemble_auc = roc_auc_score(y_test, ensemble_proba)
            ensemble_acc = accuracy_score(y_test, ensemble_pred)
        except (ValueError, TypeError):
            ensemble_auc = 0.5
            ensemble_acc = 0.5

        results["ensemble"] = {
            "auc": ensemble_auc,
            "accuracy": ensemble_acc,
        }

        # Compare to best individual
        if self.models:
            best_individual_auc = max(self.models[k]["cv_score"] for k in self.models)
        else:
            best_individual_auc = 0.5  # Default when no models

        results["comparison"] = {
            "ensemble_auc": ensemble_auc,
            "best_individual_auc": best_individual_auc,
            "improvement": ensemble_auc - best_individual_auc,
            "ensemble_is_better": ensemble_auc >= best_individual_auc,
        }

        print(f"\n[ENSEMBLE EVALUATION]")
        print(f"  Ensemble AUC: {ensemble_auc:.4f}")
        print(f"  Best Individual AUC: {best_individual_auc:.4f}")
        print(f"  Improvement: {(ensemble_auc - best_individual_auc):.4f}")

        return results


def create_robustness_ensemble(
    X: np.ndarray,
    y: np.ndarray,
    sample_weights: np.ndarray = None,
    base_params: Dict = None,
    optimal_dims: int = 30,
    n_dim_variants: int = 2,
    n_param_variants: int = 2,
    param_noise_pct: float = 0.05,
) -> Tuple[RobustnessEnsemble, Dict]:
    """
    Convenience function to create and train a robustness ensemble.

    Returns:
        Tuple of (RobustnessEnsemble instance, training results)
    """
    ensemble = RobustnessEnsemble(
        n_dimension_variants=n_dim_variants,
        n_param_variants=n_param_variants,
        param_noise_pct=param_noise_pct,
    )

    results = ensemble.train_ensemble(
        X=X,
        y=y,
        sample_weights=sample_weights,
        base_params=base_params,
        optimal_dims=optimal_dims,
    )

    return ensemble, results


if __name__ == "__main__":
    # Test the module
    print("Anti-Overfit Module - Test Run")
    print("=" * 50)

    # Test WeightedModelEvaluator
    evaluator = WeightedModelEvaluator()
    print(f"Default weights: {evaluator.weights}")

    # Test SyntheticSPYGenerator
    synth = SyntheticSPYGenerator()
    print(f"Synthetic universes: {synth.n_universes}")
    print(f"Real weight: {synth.real_weight}")

    # Test CrossAssetFeatures
    cross = CrossAssetFeatures()
    print(f"Cross assets: {cross.assets}")

    print("\nModule loaded successfully!")
