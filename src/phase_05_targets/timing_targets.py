"""
GIGA TRADER - Target Labeling for Entry/Exit Timing
====================================================
Creates target labels for entry/exit timing from intraday historical data.

For each trading day, computes:
  - Optimal entry time (minute that gave best entry price)
  - Optimal exit time (minute that gave best exit price)
  - Optimal position size (based on volatility and opportunity)
  - Optimal stop/take profit (based on actual price movement)
  - Whether batching would have helped
"""

from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd


class TargetLabeler:
    """
    Creates target labels for entry/exit timing from intraday historical data.

    For each trading day, computes:
      - Optimal entry time (minute that gave best entry price)
      - Optimal exit time (minute that gave best exit price)
      - Optimal position size (based on volatility and opportunity)
      - Optimal stop/take profit (based on actual price movement)
      - Whether batching would have helped
    """

    def __init__(
        self,
        market_open_minutes: int = 0,      # Minutes from 9:30 AM
        market_close_minutes: int = 390,   # 4:00 PM = 390 minutes from open
        entry_window: Tuple[int, int] = (0, 120),   # First 2 hours
        exit_window: Tuple[int, int] = (180, 385),  # Last ~3.5 hours
        time_resolution: int = 5,  # Group into 5-minute bins
    ):
        self.market_open_minutes = market_open_minutes
        self.market_close_minutes = market_close_minutes
        self.entry_window = entry_window
        self.exit_window = exit_window
        self.time_resolution = time_resolution

    def compute_optimal_targets(
        self,
        daily_bars: pd.DataFrame,
        direction: str = "LONG",
    ) -> Dict[str, Any]:
        """
        Given intraday bars for one day, compute optimal entry/exit targets.

        Args:
            daily_bars: DataFrame with ['datetime', 'open', 'high', 'low', 'close', 'volume']
            direction: "LONG" or "SHORT"

        Returns:
            Dict with optimal targets
        """
        if daily_bars.empty or len(daily_bars) < 10:
            return None

        # Ensure datetime index
        if 'datetime' in daily_bars.columns:
            daily_bars = daily_bars.set_index('datetime')

        # Calculate minutes from market open (9:30 AM)
        market_open = daily_bars.index[0].replace(hour=9, minute=30, second=0)
        daily_bars['minutes_from_open'] = [
            (ts - market_open).total_seconds() / 60
            for ts in daily_bars.index
        ]

        # Filter to entry and exit windows
        entry_bars = daily_bars[
            (daily_bars['minutes_from_open'] >= self.entry_window[0]) &
            (daily_bars['minutes_from_open'] <= self.entry_window[1])
        ]

        exit_bars = daily_bars[
            (daily_bars['minutes_from_open'] >= self.exit_window[0]) &
            (daily_bars['minutes_from_open'] <= self.exit_window[1])
        ]

        if entry_bars.empty or exit_bars.empty:
            return None

        # Find optimal entry and exit
        if direction == "LONG":
            # Best entry: lowest price in entry window
            optimal_entry_idx = entry_bars['low'].idxmin()
            optimal_entry_price = entry_bars.loc[optimal_entry_idx, 'low']
            optimal_entry_minute = entry_bars.loc[optimal_entry_idx, 'minutes_from_open']

            # Best exit: highest price in exit window (after entry)
            valid_exits = exit_bars[exit_bars.index > optimal_entry_idx]
            if valid_exits.empty:
                valid_exits = exit_bars

            optimal_exit_idx = valid_exits['high'].idxmax()
            optimal_exit_price = valid_exits.loc[optimal_exit_idx, 'high']
            optimal_exit_minute = valid_exits.loc[optimal_exit_idx, 'minutes_from_open']

        else:  # SHORT
            # Best entry: highest price in entry window
            optimal_entry_idx = entry_bars['high'].idxmax()
            optimal_entry_price = entry_bars.loc[optimal_entry_idx, 'high']
            optimal_entry_minute = entry_bars.loc[optimal_entry_idx, 'minutes_from_open']

            # Best exit: lowest price in exit window (after entry)
            valid_exits = exit_bars[exit_bars.index > optimal_entry_idx]
            if valid_exits.empty:
                valid_exits = exit_bars

            optimal_exit_idx = valid_exits['low'].idxmin()
            optimal_exit_price = valid_exits.loc[optimal_exit_idx, 'low']
            optimal_exit_minute = valid_exits.loc[optimal_exit_idx, 'minutes_from_open']

        # Calculate metrics
        day_open = daily_bars['open'].iloc[0]
        day_close = daily_bars['close'].iloc[-1]
        day_high = daily_bars['high'].max()
        day_low = daily_bars['low'].min()
        day_range = day_high - day_low
        day_volatility = daily_bars['close'].pct_change().std() * np.sqrt(len(daily_bars))

        # Calculate achieved return vs potential
        if direction == "LONG":
            optimal_return = (optimal_exit_price - optimal_entry_price) / optimal_entry_price
            max_possible_return = (day_high - day_low) / day_low
        else:
            optimal_return = (optimal_entry_price - optimal_exit_price) / optimal_entry_price
            max_possible_return = (day_high - day_low) / day_high

        # Calculate where stop loss should have been
        if direction == "LONG":
            # Find lowest point before the best exit
            pre_exit_bars = daily_bars[daily_bars.index <= optimal_exit_idx]
            post_entry_bars = pre_exit_bars[pre_exit_bars.index >= optimal_entry_idx]
            if not post_entry_bars.empty:
                max_adverse = (optimal_entry_price - post_entry_bars['low'].min()) / optimal_entry_price
            else:
                max_adverse = 0
            optimal_stop_pct = max_adverse + 0.002  # Add small buffer
        else:
            pre_exit_bars = daily_bars[daily_bars.index <= optimal_exit_idx]
            post_entry_bars = pre_exit_bars[pre_exit_bars.index >= optimal_entry_idx]
            if not post_entry_bars.empty:
                max_adverse = (post_entry_bars['high'].max() - optimal_entry_price) / optimal_entry_price
            else:
                max_adverse = 0
            optimal_stop_pct = max_adverse + 0.002

        # Calculate optimal take profit
        optimal_take_profit_pct = abs(optimal_return) - 0.001  # Slightly less than achieved

        # Determine if batching would help
        # Compare price at entry vs average price over entry window
        entry_prices = entry_bars['close'].values
        avg_entry_price = np.mean(entry_prices)
        batching_benefit = abs(avg_entry_price - optimal_entry_price) / optimal_entry_price
        should_batch = batching_benefit > 0.001  # Benefit > 0.1%

        # Optimal batch count based on volatility
        if day_volatility > 0.02:
            optimal_batches = 4
        elif day_volatility > 0.015:
            optimal_batches = 3
        elif day_volatility > 0.01:
            optimal_batches = 2
        else:
            optimal_batches = 1

        # Calculate optimal position size based on confidence and volatility
        # Higher volatility = lower position, higher expected return = higher position
        base_position = 0.10
        volatility_adjustment = max(0.5, min(1.5, 0.015 / max(day_volatility, 0.005)))
        return_adjustment = max(0.5, min(2.0, abs(optimal_return) / 0.01))
        optimal_position_pct = base_position * volatility_adjustment * return_adjustment
        optimal_position_pct = min(0.25, max(0.05, optimal_position_pct))  # Cap between 5-25%

        return {
            # Core targets
            "optimal_entry_minute": int(optimal_entry_minute),
            "optimal_exit_minute": int(optimal_exit_minute),
            "optimal_entry_price": optimal_entry_price,
            "optimal_exit_price": optimal_exit_price,

            # Position sizing
            "optimal_position_pct": optimal_position_pct,
            "day_volatility": day_volatility,

            # Stop/Take profit
            "optimal_stop_pct": min(0.05, optimal_stop_pct),  # Cap at 5%
            "optimal_take_profit_pct": min(0.05, max(0.005, optimal_take_profit_pct)),
            "max_adverse_excursion": max_adverse,

            # Batching
            "should_batch": should_batch,
            "optimal_batches": optimal_batches,
            "batching_benefit": batching_benefit,

            # Metrics
            "optimal_return": optimal_return,
            "max_possible_return": max_possible_return,
            "capture_efficiency": optimal_return / max(max_possible_return, 0.001),

            # Day stats
            "day_range": day_range,
            "day_range_pct": day_range / day_open,
        }

    def create_training_targets(
        self,
        intraday_data: pd.DataFrame,
        directions: pd.Series,
    ) -> pd.DataFrame:
        """
        Create target labels for all days in the dataset.

        Args:
            intraday_data: Multi-day intraday data with date column
            directions: Series of directions indexed by date ("LONG" or "SHORT")

        Returns:
            DataFrame with target columns for each day
        """
        targets = []

        # Group by date
        if 'date' not in intraday_data.columns:
            intraday_data['date'] = intraday_data.index.date

        for date, daily_bars in intraday_data.groupby('date'):
            direction = directions.get(date, "LONG")

            result = self.compute_optimal_targets(daily_bars.copy(), direction)
            if result is not None:
                result['date'] = date
                targets.append(result)

        return pd.DataFrame(targets).set_index('date')
