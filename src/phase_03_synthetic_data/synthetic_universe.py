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
        n_universes: int = 20,
        real_weight: float = 0.6,  # Weight on real SPY vs synthetic
        use_bear_universes: bool = True,
        bear_mean_shift_bps: Optional[List[int]] = None,
        bear_vol_amplify_factor: float = 1.5,
        bear_vol_dampen_factor: float = 0.7,
        use_multiscale_bootstrap: bool = True,
    ):
        self.n_universes = n_universes
        self.real_weight = real_weight
        self.synthetic_weight = (1 - real_weight) / n_universes
        self.use_bear_universes = use_bear_universes
        self.bear_mean_shift_bps = bear_mean_shift_bps or [5, 10]
        self.bear_vol_amplify_factor = bear_vol_amplify_factor
        self.bear_vol_dampen_factor = bear_vol_dampen_factor
        self.use_multiscale_bootstrap = use_multiscale_bootstrap

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

        # ─────────────────────────────────────────────────────────────────────
        # Universe 11-12: Sector-Balanced (tech vs non-tech) (Wave 16)
        # ─────────────────────────────────────────────────────────────────────
        tech_stocks = {"AAPL", "MSFT", "NVDA", "GOOGL", "META", "TSLA", "AMZN",
                       "AVGO", "CSCO", "ADBE", "CRM", "INTC", "TXN", "AMD"}
        tech_cols = [c for c in returns.columns if c in tech_stocks]
        non_tech_cols = [c for c in returns.columns if c not in tech_stocks]
        if tech_cols and non_tech_cols:
            # Tech-heavy: 2x weight on tech
            tech_heavy = returns.copy()
            tech_heavy[tech_cols] = tech_heavy[tech_cols] * 2.0
            universe = self._returns_to_spy_like(tech_heavy, spy_data)
            universe["universe_type"] = "sector_tech_heavy"
            universes.append(universe)
            print(f"  Created: Sector tech-heavy (2x tech weight)")

            # Defensive-heavy: 2x weight on non-tech
            def_heavy = returns.copy()
            def_heavy[non_tech_cols] = def_heavy[non_tech_cols] * 2.0
            universe = self._returns_to_spy_like(def_heavy, spy_data)
            universe["universe_type"] = "sector_defensive_heavy"
            universes.append(universe)
            print(f"  Created: Sector defensive-heavy (2x non-tech weight)")

        # ─────────────────────────────────────────────────────────────────────
        # Universe 13-14: Correlation Filter (Wave 16)
        # ─────────────────────────────────────────────────────────────────────
        basket_return = returns.mean(axis=1)
        if len(returns.columns) > 10:
            corr_with_basket = returns.corrwith(basket_return)
            corr_with_basket = corr_with_basket.dropna().sort_values()

            # Remove stocks most correlated (keep diversifiers)
            n_remove = max(1, len(corr_with_basket) // 5)
            diversified_cols = corr_with_basket.head(-n_remove).index.tolist()
            if len(diversified_cols) >= 10:
                universe = self._returns_to_spy_like(returns[diversified_cols], spy_data)
                universe["universe_type"] = "filter_high_corr"
                universes.append(universe)
                print(f"  Created: Remove high-correlation stocks ({n_remove} removed)")

            # Remove stocks least correlated (keep core basket)
            concentrated_cols = corr_with_basket.tail(-n_remove).index.tolist()
            if len(concentrated_cols) >= 10:
                universe = self._returns_to_spy_like(returns[concentrated_cols], spy_data)
                universe["universe_type"] = "filter_low_corr"
                universes.append(universe)
                print(f"  Created: Remove low-correlation stocks ({n_remove} removed)")

        # ─────────────────────────────────────────────────────────────────────
        # Universe 15-16: Rolling Bootstrap (different time periods) (Wave 16)
        # ─────────────────────────────────────────────────────────────────────
        n_days = len(returns)
        if n_days > 504:  # Need at least 2 years
            # First half bootstrap
            np.random.seed(777)
            half = n_days // 2
            first_half = returns.iloc[:half]
            n_keep = int(len(returns.columns) * 0.7)
            keep_cols = np.random.choice(returns.columns, n_keep, replace=False)
            filtered = pd.concat([first_half[keep_cols], returns.iloc[half:][keep_cols]])
            universe = self._returns_to_spy_like(filtered, spy_data)
            universe["universe_type"] = "rolling_bootstrap_first_half"
            universes.append(universe)
            print(f"  Created: Rolling bootstrap (first-half resampled)")

            # Second half bootstrap
            np.random.seed(888)
            keep_cols2 = np.random.choice(returns.columns, n_keep, replace=False)
            filtered2 = pd.concat([returns.iloc[:half][keep_cols2], returns.iloc[half:][keep_cols2]])
            universe = self._returns_to_spy_like(filtered2, spy_data)
            universe["universe_type"] = "rolling_bootstrap_second_half"
            universes.append(universe)
            print(f"  Created: Rolling bootstrap (second-half resampled)")

        # ─────────────────────────────────────────────────────────────────────
        # Universe 17-18: Inverse Momentum (contrarian weighting) (Wave 16)
        # ─────────────────────────────────────────────────────────────────────
        if len(returns) > 20:
            trailing_mom = returns.rolling(20).sum()
            for period_label, period_len in [("short", 5), ("long", 60)]:
                if len(returns) <= period_len:
                    continue
                mom = returns.rolling(period_len).sum()
                # Weight inversely: worst performers get highest weight
                for idx in returns.index:
                    day_mom = mom.loc[idx].dropna()
                    if len(day_mom) < 5:
                        continue
                # Instead of per-day reweighting (expensive), remove top momentum
                final_mom = returns.iloc[-min(period_len, len(returns)):].mean()
                final_mom = final_mom.dropna().sort_values()
                # Keep bottom 70% (remove top 30% momentum winners)
                contrarian_keep = final_mom.head(int(len(final_mom) * 0.7)).index.tolist()
                if len(contrarian_keep) >= 10:
                    universe = self._returns_to_spy_like(returns[contrarian_keep], spy_data)
                    universe["universe_type"] = f"contrarian_{period_label}"
                    universes.append(universe)
                    print(f"  Created: Contrarian {period_label}-term (remove top momentum)")

        # ─────────────────────────────────────────────────────────────────────
        # Universe 19-20: Additional Bootstrap Seeds (Wave 16)
        # ─────────────────────────────────────────────────────────────────────
        for seed in [256, 512]:
            np.random.seed(seed)
            n_keep = int(len(returns.columns) * 0.65)  # 65% subset (tighter than 70%)
            keep_cols = np.random.choice(returns.columns, n_keep, replace=False)
            filtered_returns = returns[keep_cols]
            universe = self._returns_to_spy_like(filtered_returns, spy_data)
            universe["universe_type"] = f"bootstrap_tight_{seed}"
            universes.append(universe)
            print(f"  Created: Tight bootstrap sample (65%, seed={seed})")

        # ─────────────────────────────────────────────────────────────────────
        # Universe 21-28: Bear Market Synthetic Series (Bull Bias Correction)
        # ─────────────────────────────────────────────────────────────────────
        if self.use_bear_universes:
            bear_universes = self._generate_bear_universes(returns, spy_data)
            universes.extend(bear_universes)

        # ─────────────────────────────────────────────────────────────────────
        # Universe 29-35: Multi-Timescale Regime Bootstrap
        # ─────────────────────────────────────────────────────────────────────
        if self.use_multiscale_bootstrap:
            multiscale_universes = self._generate_multiscale_universes(spy_data)
            universes.extend(multiscale_universes)

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

    # ═══════════════════════════════════════════════════════════════════════
    # BEAR MARKET SYNTHETIC UNIVERSES (Bull Bias Correction)
    # ═══════════════════════════════════════════════════════════════════════

    def _generate_bear_universes(
        self,
        returns: pd.DataFrame,
        spy_data: pd.DataFrame,
    ) -> List[pd.DataFrame]:
        """
        Generate synthetic bear market universes to counteract bull market bias.

        SPY 2016-2026 has ~55% up days vs ~38% down days. These methods create
        alternative time series with more balanced or bearish return distributions.

        All methods bypass _returns_to_spy_like() to avoid blending back real
        (bullish) SPY returns, constructing output DataFrames directly.

        Returns list of DataFrames with columns:
            date, day_return, synthetic_return, real_return, universe_type
        """
        print("\n[BEAR UNIVERSES] Generating bear market synthetic series (bull bias correction)...")

        # Extract SPY daily returns and normalize dates
        spy_returns = spy_data.groupby("date")["day_return"].first()
        spy_returns.index = pd.to_datetime(spy_returns.index).normalize()
        if spy_returns.index.tz is not None:
            spy_returns.index = spy_returns.index.tz_localize(None)

        common_dates = spy_returns.index
        spy_ret = spy_returns.loc[common_dates]

        universes = []

        # Method 1: Bear Block Bootstrap (2 universes)
        for seed in [1001, 1002]:
            universe = self._bear_block_bootstrap(spy_ret, common_dates, seed)
            if universe is not None:
                universes.append(universe)

        # Method 2: Regime-Balanced Resampling (2 universes)
        universe = self._bear_regime_balanced(
            spy_ret, common_dates,
            target_up_pct=0.45, target_down_pct=0.45,
            seed=2001, label="bear_regime_balanced",
        )
        if universe is not None:
            universes.append(universe)

        universe = self._bear_regime_balanced(
            spy_ret, common_dates,
            target_up_pct=0.35, target_down_pct=0.55,
            seed=2002, label="bear_regime_heavy",
        )
        if universe is not None:
            universes.append(universe)

        # Method 3: Mean Shift (2 universes)
        for shift_bps in self.bear_mean_shift_bps:
            universe = self._bear_mean_shift(spy_ret, common_dates, shift_bps)
            if universe is not None:
                universes.append(universe)

        # Method 4: Volatility-Amplified Bear (1 universe)
        universe = self._bear_vol_amplified(spy_ret, common_dates)
        if universe is not None:
            universes.append(universe)

        # Method 5: Defensive Rotation (1 universe)
        universe = self._bear_defensive_rotation(returns, spy_ret, common_dates)
        if universe is not None:
            universes.append(universe)

        print(f"[BEAR UNIVERSES] Generated {len(universes)} bear market universes")
        return universes

    def _bear_block_bootstrap(
        self,
        spy_ret: pd.Series,
        common_dates: pd.DatetimeIndex,
        seed: int,
    ) -> Optional[pd.DataFrame]:
        """
        Block bootstrap from drawdown periods.

        Identifies contiguous runs where rolling 10-day return < -2%,
        then block-bootstraps from these bear periods to create a full-length
        series where all returns come from actual drawdown days.
        """
        # Identify drawdown periods
        rolling_10d = spy_ret.rolling(10).sum()
        bear_mask = rolling_10d < -0.02

        # Expand to include the 5 days following each drawdown
        bear_expanded = bear_mask.copy()
        bear_indices = bear_mask[bear_mask].index
        for idx in bear_indices:
            pos = spy_ret.index.get_loc(idx)
            for offset in range(1, 6):
                if pos + offset < len(spy_ret):
                    bear_expanded.iloc[pos + offset] = True

        bear_dates = spy_ret.index[bear_expanded.fillna(False)]

        if len(bear_dates) < 30:
            print(f"  [SKIP] bear_block_bootstrap_{seed}: only {len(bear_dates)} bear dates found (need 30+)")
            return None

        # Block bootstrap: sample contiguous blocks of 5-20 days
        rng = np.random.RandomState(seed)
        n_target = len(common_dates)
        sampled_returns = []

        # Find contiguous blocks within bear_dates
        bear_positions = [spy_ret.index.get_loc(d) for d in bear_dates]
        blocks = []
        current_block = [bear_positions[0]]

        for i in range(1, len(bear_positions)):
            if bear_positions[i] == bear_positions[i - 1] + 1:
                current_block.append(bear_positions[i])
            else:
                if len(current_block) >= 3:
                    blocks.append(current_block)
                current_block = [bear_positions[i]]
        if len(current_block) >= 3:
            blocks.append(current_block)

        if len(blocks) == 0:
            print(f"  [SKIP] bear_block_bootstrap_{seed}: no contiguous blocks of 3+ days found")
            return None

        # Sample blocks with replacement until we have enough returns
        while len(sampled_returns) < n_target:
            block = blocks[rng.randint(0, len(blocks))]
            # Take a random sub-block of 5-20 days (or full block if shorter)
            block_len = min(rng.randint(5, 21), len(block))
            start = rng.randint(0, max(1, len(block) - block_len + 1))
            sub_block = block[start:start + block_len]
            for pos in sub_block:
                sampled_returns.append(spy_ret.iloc[pos])
                if len(sampled_returns) >= n_target:
                    break

        bear_returns = np.array(sampled_returns[:n_target])

        label = f"bear_block_bootstrap_{seed}"
        up_pct = (bear_returns > 0.0025).mean()
        print(f"  Created: {label} (up_pct={up_pct:.1%}, mean={bear_returns.mean():.4%})")

        return pd.DataFrame({
            "date": common_dates,
            "day_return": bear_returns,
            "synthetic_return": bear_returns,
            "real_return": spy_ret.values,
            "universe_type": label,
        })

    def _bear_regime_balanced(
        self,
        spy_ret: pd.Series,
        common_dates: pd.DatetimeIndex,
        target_up_pct: float,
        target_down_pct: float,
        seed: int,
        label: str,
    ) -> Optional[pd.DataFrame]:
        """
        Resample returns to achieve target up/down day proportions.

        Every return is a real SPY return — only the proportions change.
        """
        threshold = 0.0025

        up_returns = spy_ret[spy_ret > threshold].values
        down_returns = spy_ret[spy_ret < -threshold].values
        flat_returns = spy_ret[(spy_ret >= -threshold) & (spy_ret <= threshold)].values

        if len(up_returns) < 10 or len(down_returns) < 10:
            print(f"  [SKIP] {label}: insufficient up ({len(up_returns)}) or down ({len(down_returns)}) days")
            return None

        rng = np.random.RandomState(seed)
        n_target = len(common_dates)
        n_up = int(n_target * target_up_pct)
        n_down = int(n_target * target_down_pct)
        n_flat = n_target - n_up - n_down

        # Sample with replacement from each bucket
        sampled_up = rng.choice(up_returns, size=n_up, replace=True)
        sampled_down = rng.choice(down_returns, size=n_down, replace=True)
        if len(flat_returns) > 0 and n_flat > 0:
            sampled_flat = rng.choice(flat_returns, size=n_flat, replace=True)
        else:
            # Split flat allocation between up and down
            extra_down = n_flat // 2
            extra_up = n_flat - extra_down
            sampled_up = np.concatenate([sampled_up, rng.choice(up_returns, size=extra_up, replace=True)])
            sampled_down = np.concatenate([sampled_down, rng.choice(down_returns, size=extra_down, replace=True)])
            sampled_flat = np.array([])

        # Combine and shuffle
        combined = np.concatenate([sampled_up, sampled_down, sampled_flat])
        rng.shuffle(combined)

        up_pct = (combined > threshold).mean()
        print(f"  Created: {label} (up_pct={up_pct:.1%}, mean={combined.mean():.4%})")

        return pd.DataFrame({
            "date": common_dates,
            "day_return": combined[:n_target],
            "synthetic_return": combined[:n_target],
            "real_return": spy_ret.values,
            "universe_type": label,
        })

    def _bear_mean_shift(
        self,
        spy_ret: pd.Series,
        common_dates: pd.DatetimeIndex,
        shift_bps: int,
    ) -> Optional[pd.DataFrame]:
        """
        Shift the return distribution downward by a fixed amount.

        Preserves exact temporal structure, volatility clustering, and
        autocorrelation — only the drift (first moment) changes.

        5bp shift → ~flat market (50/50 up/down)
        10bp shift → mild bear (~45% up, ~48% down)
        """
        shift = shift_bps / 10000.0  # Convert basis points to decimal

        shifted = spy_ret.values - shift
        shifted = np.clip(shifted, -0.10, 0.10)

        label = f"bear_mean_shift_{shift_bps}bp"
        up_pct = (shifted > 0.0025).mean()
        print(f"  Created: {label} (up_pct={up_pct:.1%}, mean={shifted.mean():.4%})")

        return pd.DataFrame({
            "date": common_dates,
            "day_return": shifted,
            "synthetic_return": shifted,
            "real_return": spy_ret.values,
            "universe_type": label,
        })

    def _bear_vol_amplified(
        self,
        spy_ret: pd.Series,
        common_dates: pd.DatetimeIndex,
    ) -> Optional[pd.DataFrame]:
        """
        Amplify negative returns and dampen positive returns.

        Preserves which days are up/down but changes magnitudes:
        - Negative returns × 1.5 (sharper selloffs)
        - Positive returns × 0.7 (weaker rallies)

        Many marginal up days fall below the 0.25% target threshold.
        """
        modified = spy_ret.values.copy()

        bear_mask = modified < 0
        bull_mask = modified >= 0

        modified[bear_mask] = modified[bear_mask] * self.bear_vol_amplify_factor
        modified[bull_mask] = modified[bull_mask] * self.bear_vol_dampen_factor

        modified = np.clip(modified, -0.10, 0.10)

        label = "bear_vol_amplified"
        up_pct = (modified > 0.0025).mean()
        print(f"  Created: {label} (up_pct={up_pct:.1%}, mean={modified.mean():.4%})")

        return pd.DataFrame({
            "date": common_dates,
            "day_return": modified,
            "synthetic_return": modified,
            "real_return": spy_ret.values,
            "universe_type": label,
        })

    def _bear_defensive_rotation(
        self,
        returns: pd.DataFrame,
        spy_ret: pd.Series,
        common_dates: pd.DatetimeIndex,
    ) -> Optional[pd.DataFrame]:
        """
        Create a bear market via defensive sector rotation.

        Simulates a portfolio where cyclicals (tech) get crushed (1.4x amplified
        losses) while defensives (healthcare, staples) hold up better (0.9x).
        The net result is a declining portfolio mimicking real bear market
        sector dynamics.
        """
        defensive_tickers = {
            "JNJ", "PG", "KO", "PEP", "MRK", "ABBV", "VZ", "PFE",
            "MCD", "WMT", "COST", "PM", "UNH",
        }
        cyclical_tickers = {
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
            "AMD", "AVGO", "ADBE", "CRM",
        }

        def_cols = [c for c in returns.columns if c in defensive_tickers]
        cyc_cols = [c for c in returns.columns if c in cyclical_tickers]

        if len(def_cols) < 3 or len(cyc_cols) < 3:
            print(f"  [SKIP] bear_defensive_rotation: insufficient defensive ({len(def_cols)}) or cyclical ({len(cyc_cols)}) stocks")
            return None

        # Normalize returns index
        ret_idx = pd.to_datetime(returns.index).normalize()
        if ret_idx.tz is not None:
            ret_idx = ret_idx.tz_localize(None)

        dates_overlap = ret_idx.intersection(common_dates)
        if len(dates_overlap) < 100:
            print(f"  [SKIP] bear_defensive_rotation: only {len(dates_overlap)} overlapping dates")
            return None

        returns_aligned = returns.copy()
        returns_aligned.index = ret_idx

        # Equal-weight sub-baskets
        def_return = returns_aligned.loc[dates_overlap, def_cols].mean(axis=1)
        cyc_return = returns_aligned.loc[dates_overlap, cyc_cols].mean(axis=1)

        # Bear rotation: defensives hold up (0.9x), cyclicals crushed (1.4x loss amplification)
        bear_portfolio = 0.6 * def_return * 0.9 + 0.4 * cyc_return * 1.4

        # Scale to match SPY volatility
        spy_aligned = spy_ret.loc[dates_overlap]
        scale = spy_aligned.std() / (bear_portfolio.std() + 1e-10)
        bear_scaled = bear_portfolio * scale

        # Ensure net bearish: shift mean negative if portfolio is net positive
        if bear_scaled.mean() > 0:
            bear_scaled = bear_scaled - bear_scaled.mean() - 0.0003

        bear_scaled = np.clip(bear_scaled.values, -0.10, 0.10)

        # Build output aligned to common_dates
        # For dates not in overlap, use mean-shifted SPY return as fallback
        full_returns = spy_ret.values.copy() - 0.0005  # fallback: mild shift
        for i, d in enumerate(common_dates):
            if d in dates_overlap:
                pos = dates_overlap.get_loc(d)
                full_returns[i] = bear_scaled[pos]

        label = "bear_defensive_rotation"
        up_pct = (full_returns > 0.0025).mean()
        print(f"  Created: {label} (up_pct={up_pct:.1%}, mean={full_returns.mean():.4%})")

        return pd.DataFrame({
            "date": common_dates,
            "day_return": full_returns,
            "synthetic_return": full_returns,
            "real_return": spy_ret.values,
            "universe_type": label,
        })

    # ═══════════════════════════════════════════════════════════════════════
    # MULTI-TIMESCALE REGIME BOOTSTRAP
    # ═══════════════════════════════════════════════════════════════════════

    def _generate_multiscale_universes(
        self,
        spy_data: pd.DataFrame,
    ) -> List[pd.DataFrame]:
        """
        Generate synthetic universes via multi-timescale regime bootstrapping.

        Addresses overfitting to timescale-specific biases:
        - Weekly: too many bull weeks in 2016-2026
        - Monthly: drawdown months are rare
        - Magnitude: large swings underrepresented
        - Volatility: low-vol regimes dominate

        All methods use real SPY returns (resampled at block or day level).
        """
        print("\n[MULTISCALE] Generating multi-timescale regime bootstrap universes...")

        # Extract SPY daily returns and normalize dates
        spy_returns = spy_data.groupby("date")["day_return"].first()
        spy_returns.index = pd.to_datetime(spy_returns.index).normalize()
        if spy_returns.index.tz is not None:
            spy_returns.index = spy_returns.index.tz_localize(None)

        common_dates = spy_returns.index
        spy_ret = spy_returns.loc[common_dates]

        universes = []

        # Method 1: Weekly Regime Balanced Bootstrap (2 universes)
        for target, label in [
            ((0.40, 0.40, 0.20), "multiscale_weekly_balanced"),
            ((0.30, 0.50, 0.20), "multiscale_weekly_bear"),
        ]:
            u = self._multiscale_weekly_regime(spy_ret, common_dates, target, label, seed=3001 + len(universes))
            if u is not None:
                universes.append(u)

        # Method 2: Monthly Drawdown Oversampling (1 universe)
        u = self._multiscale_monthly_drawdown(spy_ret, common_dates, seed=3010)
        if u is not None:
            universes.append(u)

        # Method 3: Swing Magnitude Stratified Bootstrap (2 universes)
        u = self._multiscale_swing_magnitude(
            spy_ret, common_dates,
            target_large_pct=0.40, label="multiscale_swing_large", seed=3020,
        )
        if u is not None:
            universes.append(u)

        u = self._multiscale_swing_magnitude(
            spy_ret, common_dates,
            target_flat_pct=0.40, label="multiscale_swing_flat", seed=3021,
        )
        if u is not None:
            universes.append(u)

        # Method 4: Volatility Regime Block Bootstrap (1 universe)
        u = self._multiscale_vol_regime_block(spy_ret, common_dates, seed=3030)
        if u is not None:
            universes.append(u)

        # Method 5: Cross-Scale Cascading Bootstrap (1 universe)
        u = self._multiscale_cascade(spy_ret, common_dates, seed=3040)
        if u is not None:
            universes.append(u)

        print(f"[MULTISCALE] Generated {len(universes)} multi-timescale universes")
        return universes

    def _multiscale_weekly_regime(
        self,
        spy_ret: pd.Series,
        common_dates: pd.DatetimeIndex,
        target_pcts: Tuple[float, float, float],
        label: str,
        seed: int,
    ) -> Optional[pd.DataFrame]:
        """
        Block-bootstrap entire trading weeks to balance bull/bear/choppy weeks.

        Preserves intra-week autocorrelation by resampling 5-day blocks.
        """
        threshold = 0.0025
        returns_arr = spy_ret.values
        n_days = len(returns_arr)

        # Chunk into 5-day (trading week) blocks
        block_size = 5
        n_blocks = n_days // block_size
        if n_blocks < 10:
            print(f"  [SKIP] {label}: only {n_blocks} weekly blocks (need 10+)")
            return None

        blocks = []
        block_labels = []
        for i in range(n_blocks):
            start = i * block_size
            end = start + block_size
            block = returns_arr[start:end]
            blocks.append(block)

            # Classify: count up/down days in this week
            n_up = np.sum(block > threshold)
            n_down = np.sum(block < -threshold)

            if n_up >= 3:
                block_labels.append("bull")
            elif n_down >= 3:
                block_labels.append("bear")
            else:
                block_labels.append("choppy")

        block_labels = np.array(block_labels)

        # Group blocks by label
        bull_blocks = [b for b, l in zip(blocks, block_labels) if l == "bull"]
        bear_blocks = [b for b, l in zip(blocks, block_labels) if l == "bear"]
        choppy_blocks = [b for b, l in zip(blocks, block_labels) if l == "choppy"]

        # Need at least some of each type
        if len(bull_blocks) < 3 or len(bear_blocks) < 3 or len(choppy_blocks) < 2:
            print(f"  [SKIP] {label}: insufficient regime diversity "
                  f"(bull={len(bull_blocks)}, bear={len(bear_blocks)}, choppy={len(choppy_blocks)})")
            return None

        target_bull_pct, target_bear_pct, target_choppy_pct = target_pcts
        rng = np.random.RandomState(seed)

        n_target_blocks = n_blocks
        n_bull = int(n_target_blocks * target_bull_pct)
        n_bear = int(n_target_blocks * target_bear_pct)
        n_choppy = n_target_blocks - n_bull - n_bear

        # Sample blocks with replacement
        sampled = []
        for pool, count in [(bull_blocks, n_bull), (bear_blocks, n_bear), (choppy_blocks, n_choppy)]:
            indices = rng.randint(0, len(pool), size=count)
            for idx in indices:
                sampled.append(pool[idx])

        # Shuffle block order
        rng.shuffle(sampled)

        # Flatten to daily returns
        resampled_returns = np.concatenate(sampled)[:n_days]

        # Handle remainder days (if n_days not divisible by 5)
        remainder = n_days - len(resampled_returns)
        if remainder > 0:
            extra = rng.choice(returns_arr, size=remainder, replace=True)
            resampled_returns = np.concatenate([resampled_returns, extra])

        # Report stats
        up_pct = (resampled_returns > threshold).mean()
        print(f"  Created: {label} (up_pct={up_pct:.1%}, mean={resampled_returns.mean():.4%}, "
              f"target={target_bull_pct:.0%}/{target_bear_pct:.0%}/{target_choppy_pct:.0%} bull/bear/choppy)")

        return pd.DataFrame({
            "date": common_dates,
            "day_return": resampled_returns,
            "synthetic_return": resampled_returns,
            "real_return": spy_ret.values,
            "universe_type": label,
        })

    def _multiscale_monthly_drawdown(
        self,
        spy_ret: pd.Series,
        common_dates: pd.DatetimeIndex,
        seed: int,
    ) -> Optional[pd.DataFrame]:
        """
        Block-bootstrap entire month-blocks to oversample drawdown months.

        Month = 21 trading days. Resamples to 40% drawdown / 30% rally / 30% flat.
        Preserves volatility clustering within each month.
        """
        returns_arr = spy_ret.values
        n_days = len(returns_arr)
        block_size = 21  # ~1 trading month

        n_blocks = n_days // block_size
        if n_blocks < 6:
            print(f"  [SKIP] multiscale_monthly_drawdown: only {n_blocks} month-blocks (need 6+)")
            return None

        blocks = []
        block_labels = []
        for i in range(n_blocks):
            start = i * block_size
            end = start + block_size
            block = returns_arr[start:end]
            blocks.append(block)

            cum_return = np.sum(block)  # Approximate cumulative return
            if cum_return < -0.03:
                block_labels.append("drawdown")
            elif cum_return > 0.03:
                block_labels.append("rally")
            else:
                block_labels.append("flat")

        block_labels = np.array(block_labels)

        drawdown_blocks = [b for b, l in zip(blocks, block_labels) if l == "drawdown"]
        rally_blocks = [b for b, l in zip(blocks, block_labels) if l == "rally"]
        flat_blocks = [b for b, l in zip(blocks, block_labels) if l == "flat"]

        # Need at least some drawdown months
        if len(drawdown_blocks) < 2:
            print(f"  [SKIP] multiscale_monthly_drawdown: only {len(drawdown_blocks)} drawdown months (need 2+)")
            return None

        # If we're short on any category, merge flat into the deficit
        if len(rally_blocks) < 2:
            rally_blocks = flat_blocks[:len(flat_blocks) // 2] if flat_blocks else drawdown_blocks[:1]
        if len(flat_blocks) < 2:
            flat_blocks = rally_blocks[:len(rally_blocks) // 2] if rally_blocks else drawdown_blocks[:1]

        rng = np.random.RandomState(seed)

        # Target: 40% drawdown, 30% rally, 30% flat
        n_draw = max(1, int(n_blocks * 0.40))
        n_rally = max(1, int(n_blocks * 0.30))
        n_flat = n_blocks - n_draw - n_rally

        sampled = []
        for pool, count in [(drawdown_blocks, n_draw), (rally_blocks, n_rally), (flat_blocks, n_flat)]:
            indices = rng.randint(0, len(pool), size=count)
            for idx in indices:
                sampled.append(pool[idx])

        rng.shuffle(sampled)
        resampled_returns = np.concatenate(sampled)[:n_days]

        # Handle remainder
        remainder = n_days - len(resampled_returns)
        if remainder > 0:
            extra = rng.choice(returns_arr, size=remainder, replace=True)
            resampled_returns = np.concatenate([resampled_returns, extra])

        up_pct = (resampled_returns > 0.0025).mean()
        print(f"  Created: multiscale_monthly_drawdown (up_pct={up_pct:.1%}, mean={resampled_returns.mean():.4%})")

        return pd.DataFrame({
            "date": common_dates,
            "day_return": resampled_returns,
            "synthetic_return": resampled_returns,
            "real_return": spy_ret.values,
            "universe_type": "multiscale_monthly_drawdown",
        })

    def _multiscale_swing_magnitude(
        self,
        spy_ret: pd.Series,
        common_dates: pd.DatetimeIndex,
        target_large_pct: float = 0.0,
        target_flat_pct: float = 0.0,
        label: str = "multiscale_swing",
        seed: int = 3020,
    ) -> Optional[pd.DataFrame]:
        """
        Stratified bootstrap by daily return magnitude.

        Classifies days as large-swing (|ret|>1%), normal (0.25%-1%), flat (<0.25%).
        Oversamples either large swings or flat days to the target percentage.
        """
        returns_arr = spy_ret.values
        abs_returns = np.abs(returns_arr)

        large_mask = abs_returns > 0.01
        flat_mask = abs_returns <= 0.0025
        normal_mask = ~large_mask & ~flat_mask

        large_days = returns_arr[large_mask]
        normal_days = returns_arr[normal_mask]
        flat_days = returns_arr[flat_mask]

        if len(large_days) < 5 or len(normal_days) < 10 or len(flat_days) < 5:
            print(f"  [SKIP] {label}: insufficient magnitude diversity "
                  f"(large={len(large_days)}, normal={len(normal_days)}, flat={len(flat_days)})")
            return None

        rng = np.random.RandomState(seed)
        n_target = len(common_dates)

        if target_large_pct > 0:
            # Oversample large swings
            n_large = int(n_target * target_large_pct)
            n_remaining = n_target - n_large
            # Split remaining proportionally between normal and flat
            normal_ratio = len(normal_days) / (len(normal_days) + len(flat_days))
            n_normal = int(n_remaining * normal_ratio)
            n_flat = n_remaining - n_normal
        elif target_flat_pct > 0:
            # Oversample flat days
            n_flat = int(n_target * target_flat_pct)
            n_remaining = n_target - n_flat
            # Split remaining proportionally between large and normal
            large_ratio = len(large_days) / (len(large_days) + len(normal_days))
            n_large = int(n_remaining * large_ratio)
            n_normal = n_remaining - n_large
        else:
            print(f"  [SKIP] {label}: no target specified")
            return None

        sampled_large = rng.choice(large_days, size=n_large, replace=True)
        sampled_normal = rng.choice(normal_days, size=n_normal, replace=True)
        sampled_flat = rng.choice(flat_days, size=n_flat, replace=True)

        combined = np.concatenate([sampled_large, sampled_normal, sampled_flat])
        rng.shuffle(combined)

        up_pct = (combined > 0.0025).mean()
        large_actual = (np.abs(combined) > 0.01).mean()
        flat_actual = (np.abs(combined) <= 0.0025).mean()
        print(f"  Created: {label} (up_pct={up_pct:.1%}, large={large_actual:.1%}, flat={flat_actual:.1%})")

        return pd.DataFrame({
            "date": common_dates,
            "day_return": combined[:n_target],
            "synthetic_return": combined[:n_target],
            "real_return": spy_ret.values,
            "universe_type": label,
        })

    def _multiscale_vol_regime_block(
        self,
        spy_ret: pd.Series,
        common_dates: pd.DatetimeIndex,
        seed: int,
    ) -> Optional[pd.DataFrame]:
        """
        Block-bootstrap by volatility regime to equalize low/med/high vol exposure.

        Computes 10-day rolling vol, classifies each date, finds contiguous
        same-regime blocks (min 3 days), then resamples blocks to ~33% each.
        """
        returns_arr = spy_ret.values
        n_days = len(returns_arr)

        if n_days < 30:
            print(f"  [SKIP] multiscale_vol_regime: only {n_days} days (need 30+)")
            return None

        # Compute 10-day rolling volatility (annualized)
        rolling_vol = pd.Series(returns_arr).rolling(10).std() * np.sqrt(252)
        rolling_vol = rolling_vol.values

        # Classify each day
        regime = np.full(n_days, "med", dtype=object)
        for i in range(n_days):
            if np.isnan(rolling_vol[i]):
                continue
            if rolling_vol[i] < 0.12:
                regime[i] = "low"
            elif rolling_vol[i] > 0.22:
                regime[i] = "high"

        # Find contiguous blocks of same regime (min 3 days)
        blocks_by_regime = {"low": [], "med": [], "high": []}
        current_regime = regime[0]
        current_block_start = 0

        for i in range(1, n_days):
            if regime[i] != current_regime or i == n_days - 1:
                block_end = i if regime[i] != current_regime else i + 1
                block_len = block_end - current_block_start
                if block_len >= 3 and current_regime in blocks_by_regime:
                    blocks_by_regime[current_regime].append(
                        returns_arr[current_block_start:block_end]
                    )
                current_regime = regime[i]
                current_block_start = i

        n_low = sum(len(b) for b in blocks_by_regime["low"])
        n_med = sum(len(b) for b in blocks_by_regime["med"])
        n_high = sum(len(b) for b in blocks_by_regime["high"])

        if not blocks_by_regime["low"] or not blocks_by_regime["med"] or not blocks_by_regime["high"]:
            print(f"  [SKIP] multiscale_vol_regime: missing regime blocks "
                  f"(low={len(blocks_by_regime['low'])}, med={len(blocks_by_regime['med'])}, "
                  f"high={len(blocks_by_regime['high'])})")
            return None

        rng = np.random.RandomState(seed)

        # Sample blocks from each regime to fill ~1/3 each
        target_days_per_regime = n_days // 3
        sampled = []

        for reg in ["low", "med", "high"]:
            pool = blocks_by_regime[reg]
            collected_days = 0
            while collected_days < target_days_per_regime:
                block = pool[rng.randint(0, len(pool))]
                sampled.append(block)
                collected_days += len(block)

        # Shuffle block order and flatten
        rng.shuffle(sampled)
        resampled_returns = np.concatenate(sampled)[:n_days]

        # Handle remainder
        remainder = n_days - len(resampled_returns)
        if remainder > 0:
            extra = rng.choice(returns_arr, size=remainder, replace=True)
            resampled_returns = np.concatenate([resampled_returns, extra])

        up_pct = (resampled_returns > 0.0025).mean()
        print(f"  Created: multiscale_vol_regime (up_pct={up_pct:.1%}, mean={resampled_returns.mean():.4%}, "
              f"regime_days: low={n_low}, med={n_med}, high={n_high})")

        return pd.DataFrame({
            "date": common_dates,
            "day_return": resampled_returns,
            "synthetic_return": resampled_returns,
            "real_return": spy_ret.values,
            "universe_type": "multiscale_vol_regime",
        })

    def _multiscale_cascade(
        self,
        spy_ret: pd.Series,
        common_dates: pd.DatetimeIndex,
        seed: int,
    ) -> Optional[pd.DataFrame]:
        """
        Cross-scale cascading bootstrap — stratified resampling across
        multiple timescale dimensions simultaneously.

        Assigns each day a composite key (month_regime, week_regime, magnitude)
        and resamples to equalize representation across all combinations.
        This is the hardest synthetic series for a model to overfit because
        no single timescale's bias remains.
        """
        returns_arr = spy_ret.values
        n_days = len(returns_arr)

        if n_days < 63:  # Need at least 3 months
            print(f"  [SKIP] multiscale_cascade: only {n_days} days (need 63+)")
            return None

        threshold = 0.0025

        # === Assign composite keys ===

        # 1. Month regime (21-day blocks)
        month_regime = np.full(n_days, "flat_m", dtype=object)
        block_size = 21
        for i in range(n_days // block_size):
            start = i * block_size
            end = start + block_size
            cum = np.sum(returns_arr[start:end])
            label = "draw_m" if cum < -0.03 else ("rally_m" if cum > 0.03 else "flat_m")
            month_regime[start:end] = label
        # Assign remainder to last regime
        remainder_start = (n_days // block_size) * block_size
        if remainder_start < n_days:
            month_regime[remainder_start:] = month_regime[remainder_start - 1]

        # 2. Week regime (5-day blocks)
        week_regime = np.full(n_days, "choppy_w", dtype=object)
        for i in range(n_days // 5):
            start = i * 5
            end = start + 5
            block = returns_arr[start:end]
            n_up = np.sum(block > threshold)
            n_down = np.sum(block < -threshold)
            label = "bull_w" if n_up >= 3 else ("bear_w" if n_down >= 3 else "choppy_w")
            week_regime[start:end] = label
        remainder_start = (n_days // 5) * 5
        if remainder_start < n_days:
            week_regime[remainder_start:] = week_regime[remainder_start - 1]

        # 3. Magnitude bucket
        abs_ret = np.abs(returns_arr)
        magnitude = np.where(
            abs_ret > 0.01, "large",
            np.where(abs_ret <= 0.0025, "flat", "normal")
        )

        # === Build composite key and group ===
        composite_keys = [
            f"{month_regime[i]}|{week_regime[i]}|{magnitude[i]}"
            for i in range(n_days)
        ]

        # Group days by composite key
        key_to_days: Dict[str, List[int]] = {}
        for i, key in enumerate(composite_keys):
            if key not in key_to_days:
                key_to_days[key] = []
            key_to_days[key].append(i)

        # === Resampling: weight underrepresented combos higher ===
        rng = np.random.RandomState(seed)

        n_keys = len(key_to_days)
        if n_keys < 5:
            print(f"  [SKIP] multiscale_cascade: only {n_keys} unique composite keys (need 5+)")
            return None

        # Target: equal representation across all composite keys
        target_per_key = n_days // n_keys

        sampled_indices = []
        for key, day_indices in key_to_days.items():
            # Sample up to target_per_key from this bucket
            n_sample = min(target_per_key, n_days)  # Cap at total
            idx = rng.choice(day_indices, size=n_sample, replace=True)
            sampled_indices.extend(idx.tolist())

        # If we have too many, downsample; if too few, add random days
        if len(sampled_indices) > n_days:
            sampled_indices = rng.choice(sampled_indices, size=n_days, replace=False).tolist()
        elif len(sampled_indices) < n_days:
            extra = rng.choice(range(n_days), size=n_days - len(sampled_indices), replace=True)
            sampled_indices.extend(extra.tolist())

        rng.shuffle(sampled_indices)
        resampled_returns = returns_arr[sampled_indices[:n_days]]

        up_pct = (resampled_returns > threshold).mean()
        print(f"  Created: multiscale_cascade (up_pct={up_pct:.1%}, mean={resampled_returns.mean():.4%}, "
              f"composite_keys={n_keys})")

        return pd.DataFrame({
            "date": common_dates,
            "day_return": resampled_returns,
            "synthetic_return": resampled_returns,
            "real_return": spy_ret.values,
            "universe_type": "multiscale_cascade",
        })

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
