"""
GIGA TRADER - Synthetic SPY Universe Generator
================================================
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

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional

from src.phase_01_data_acquisition.alpaca_data_helper import get_alpaca_helper


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
