"""
GIGA TRADER - Event Recency Features
======================================
"Days since" features that capture market memory — how long since the
market experienced specific types of stress, euphoria, or structural shifts.

All features are computed from data already available in df_daily by the
time this step runs (step 17 in integrate_anti_overfit). No extra API
calls or downloads needed.

88 features generated (prefix: dts_), 19 sections.

Sections 1-11: Original 36 features (price, pattern, vol, streaks,
  volume, cross-asset, technicals, breadth, vol-regime, rotation, momentum).
Sections 12-19: 52 additional features (candlestick, distribution,
  volume-price divergence, gaps, weekly cycle, statistical anomaly,
  cross-asset stress, event clustering meta).
"""

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger("EVENT_RECENCY")

# Cap to prevent extremely large values dominating the feature space
MAX_DAYS_CAP = 252  # ~1 trading year


def _days_since_condition(mask: pd.Series, max_cap: int = MAX_DAYS_CAP) -> pd.Series:
    """
    Given a boolean mask (True on event days), compute a Series of
    'trading days since last True' for each row.

    First occurrence (no prior event) gets max_cap.
    """
    result = pd.Series(np.nan, index=mask.index, dtype=float)
    last_event_idx = -max_cap  # sentinel

    for i, (idx, is_event) in enumerate(mask.items()):
        if is_event:
            last_event_idx = i
            result.iloc[i] = 0.0
        else:
            result.iloc[i] = min(float(i - last_event_idx), float(max_cap))

    return result


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder smoothing."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _find_col(df: pd.DataFrame, candidates: list) -> Optional[str]:
    """Find first matching column name from a list of candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


class EventRecencyFeatures:
    """
    Compute 'days since last event' features from OHLCV data and
    cross-asset / breadth / volatility columns already in df_daily.

    Pattern: no download needed — works entirely off df_daily columns
    that have been merged by prior integration steps (1-16).
    """

    REQUIRED_COLS = {"date", "close"}

    def __init__(self, max_cap: int = MAX_DAYS_CAP):
        self._max_cap = max_cap

    def create_event_recency_features(
        self,
        spy_daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create all days-since features and merge into spy_daily.

        Requires at minimum: date, close columns.
        Optional columns used if available:
          - open, high, low, volume (OHLCV)
          - TLT_return_1d, GLD_return_1d, QQQ_return_1d (cross-asset)
          - econ_vix_close / ^VIX (VIX level)
          - sector_pct_advancing, sector_cyclical_vs_defensive (breadth)
          - rsi_14 (technical)

        Returns spy_daily with new dts_* columns added.
        """
        df = spy_daily.copy()
        n_before = len(df.columns)

        print("\n[EVENT_RECENCY] Engineering days-since features...")

        if "close" not in df.columns:
            print("  [WARN] No 'close' column — skipping event recency")
            return df

        close = df["close"].astype(float)
        daily_return = close.pct_change()
        cap = self._max_cap

        # ==================================================================
        # SECTION 1: Price-Based Event Recency (4 features)
        # ==================================================================

        df["dts_last_drop_1pct"] = _days_since_condition(
            daily_return <= -0.01, cap
        )
        df["dts_last_drop_2pct"] = _days_since_condition(
            daily_return <= -0.02, cap
        )
        df["dts_last_rally_1pct"] = _days_since_condition(
            daily_return >= 0.01, cap
        )
        df["dts_last_rally_2pct"] = _days_since_condition(
            daily_return >= 0.02, cap
        )

        # ==================================================================
        # SECTION 2: Pattern-Based (4 features)
        # ==================================================================

        prev_green = daily_return.shift(1) > 0
        today_red = daily_return <= -0.005
        df["dts_last_reversal_down"] = _days_since_condition(
            prev_green & today_red, cap
        )

        prev_red = daily_return.shift(1) < 0
        today_green = daily_return >= 0.005
        df["dts_last_reversal_up"] = _days_since_condition(
            prev_red & today_green, cap
        )

        if "open" in df.columns:
            open_price = df["open"].astype(float)
            prev_close = close.shift(1)
            gap_pct = (open_price - prev_close) / (prev_close + 1e-10)
            df["dts_last_gap_up"] = _days_since_condition(gap_pct >= 0.005, cap)
            df["dts_last_gap_down"] = _days_since_condition(gap_pct <= -0.005, cap)
        else:
            df["dts_last_gap_up"] = float(cap)
            df["dts_last_gap_down"] = float(cap)

        # ==================================================================
        # SECTION 3: Volatility / Extreme Events (4 features)
        # ==================================================================

        if "high" in df.columns and "low" in df.columns:
            high = df["high"].astype(float)
            low = df["low"].astype(float)
            intraday_range = (high - low) / (close + 1e-10)
            df["dts_last_high_vol_day"] = _days_since_condition(
                intraday_range >= 0.02, cap
            )
        else:
            df["dts_last_high_vol_day"] = _days_since_condition(
                daily_return.abs() >= 0.02, cap
            )

        # VIX spike (15%+ daily jump)
        vix_col = _find_col(df, ["^VIX", "VIX", "vix", "econ_vix_close"])
        if vix_col is not None:
            vix = df[vix_col].astype(float)
            vix_return = vix.pct_change()
            df["dts_last_vix_spike"] = _days_since_condition(
                vix_return >= 0.15, cap
            )
        else:
            df["dts_last_vix_spike"] = _days_since_condition(
                daily_return <= -0.015, cap
            )

        # 52-week high / low
        rolling_high = close.rolling(252, min_periods=20).max()
        df["dts_last_52w_high"] = _days_since_condition(close >= rolling_high, cap)

        rolling_low = close.rolling(252, min_periods=20).min()
        df["dts_last_52w_low"] = _days_since_condition(close <= rolling_low, cap)

        # ==================================================================
        # SECTION 4: Streak-Based (2 features)
        # ==================================================================

        is_green = daily_return > 0
        is_red = daily_return < 0

        green_streak = pd.Series(0, index=df.index, dtype=int)
        red_streak = pd.Series(0, index=df.index, dtype=int)
        for i in range(1, len(df)):
            if is_green.iloc[i]:
                green_streak.iloc[i] = green_streak.iloc[i - 1] + 1
            if is_red.iloc[i]:
                red_streak.iloc[i] = red_streak.iloc[i - 1] + 1

        was_winning_3 = green_streak.shift(1) >= 3
        not_green = daily_return <= 0
        df["dts_last_3day_winning"] = _days_since_condition(
            was_winning_3 & not_green, cap
        )
        was_losing_3 = red_streak.shift(1) >= 3
        not_red = daily_return >= 0
        df["dts_last_3day_losing"] = _days_since_condition(
            was_losing_3 & not_red, cap
        )

        # ==================================================================
        # SECTION 5: Volume Events (3 features)
        # ==================================================================

        if "volume" in df.columns:
            vol = df["volume"].astype(float)
            vol_ma20 = vol.rolling(20, min_periods=5).mean()

            # Volume spike: > 2x 20-day average
            df["dts_last_volume_spike"] = _days_since_condition(
                vol > 2.0 * vol_ma20, cap
            )

            # Volume dry-up: < 0.5x 20-day average
            df["dts_last_volume_dryup"] = _days_since_condition(
                vol < 0.5 * vol_ma20, cap
            )

            # Volume climax: highest volume in 60 days
            vol_60_max = vol.rolling(60, min_periods=10).max()
            df["dts_last_volume_climax"] = _days_since_condition(
                vol >= vol_60_max, cap
            )
        else:
            df["dts_last_volume_spike"] = float(cap)
            df["dts_last_volume_dryup"] = float(cap)
            df["dts_last_volume_climax"] = float(cap)

        # ==================================================================
        # SECTION 6: Cross-Asset Divergence Events (4 features)
        # ==================================================================

        # SPY-TLT correlation divergence (normally negative; >+0.5 = abnormal)
        tlt_col = _find_col(df, ["TLT_return", "TLT_return_1d", "TLT_daily_return"])
        if tlt_col is not None:
            spy_tlt_corr = daily_return.rolling(5, min_periods=3).corr(
                df[tlt_col].astype(float)
            )
            df["dts_last_spy_tlt_diverge"] = _days_since_condition(
                spy_tlt_corr > 0.5, cap
            )
        else:
            df["dts_last_spy_tlt_diverge"] = float(cap)

        # Gold outperformance (GLD beats SPY by 1%+ in a day)
        gld_col = _find_col(df, ["GLD_return", "GLD_return_1d", "GLD_daily_return"])
        if gld_col is not None:
            gold_vs_spy = df[gld_col].astype(float) - daily_return
            df["dts_last_gold_outperform"] = _days_since_condition(
                gold_vs_spy >= 0.01, cap
            )
        else:
            df["dts_last_gold_outperform"] = float(cap)

        # QQQ-SPY divergence (>1% gap in either direction)
        qqq_col = _find_col(df, ["QQQ_return", "QQQ_return_1d", "QQQ_daily_return"])
        if qqq_col is not None:
            qqq_spy_gap = (df[qqq_col].astype(float) - daily_return).abs()
            df["dts_last_qqq_spy_diverge"] = _days_since_condition(
                qqq_spy_gap >= 0.01, cap
            )
        else:
            df["dts_last_qqq_spy_diverge"] = float(cap)

        # VIX above 30 (crisis level)
        if vix_col is not None:
            vix = df[vix_col].astype(float)
            df["dts_last_vix_above_30"] = _days_since_condition(vix >= 30, cap)
        else:
            df["dts_last_vix_above_30"] = float(cap)

        # ==================================================================
        # SECTION 7: Technical Level Events (4 features)
        # ==================================================================

        # 50-day MA crossovers
        ma50 = close.rolling(50, min_periods=20).mean()
        above_50 = close > ma50
        above_50_prev = above_50.shift(1).fillna(0).astype(bool)
        crossed_up_50 = above_50 & (~above_50_prev)
        crossed_down_50 = (~above_50) & above_50_prev
        df["dts_last_ma_cross_up"] = _days_since_condition(crossed_up_50, cap)
        df["dts_last_ma_cross_down"] = _days_since_condition(crossed_down_50, cap)

        # 20-day MA reclaim (was below, now above)
        ma20 = close.rolling(20, min_periods=10).mean()
        above_20 = close > ma20
        above_20_prev = above_20.shift(1).fillna(0).astype(bool)
        reclaimed_20 = above_20 & (~above_20_prev)
        df["dts_last_close_above_20ma"] = _days_since_condition(reclaimed_20, cap)

        # Support bounce: intraday undercut of 20d low then close recovery
        if "low" in df.columns:
            low_series = df["low"].astype(float)
            rolling_low_20 = close.rolling(20, min_periods=5).min().shift(1)
            undercut = low_series <= rolling_low_20
            recovered = close > rolling_low_20
            df["dts_last_support_bounce"] = _days_since_condition(
                undercut & recovered, cap
            )
        else:
            df["dts_last_support_bounce"] = float(cap)

        # ==================================================================
        # SECTION 8: Breadth Events (3 features)
        # ==================================================================

        breadth_col = _find_col(df, [
            "sector_pct_advancing", "sector_breadth", "pct_green_1d",
        ])

        if breadth_col is not None:
            breadth = df[breadth_col].astype(float)

            # Breadth thrust: 90%+ sectors advancing
            df["dts_last_breadth_thrust"] = _days_since_condition(
                breadth >= 0.9, cap
            )
            # Breadth collapse: <10% sectors advancing
            df["dts_last_breadth_collapse"] = _days_since_condition(
                breadth <= 0.1, cap
            )
            # Breadth divergence: SPY at 20d high but <50% sectors green
            at_20d_high = close >= close.rolling(20, min_periods=5).max()
            df["dts_last_breadth_divergence"] = _days_since_condition(
                at_20d_high & (breadth < 0.5), cap
            )
        else:
            df["dts_last_breadth_thrust"] = float(cap)
            df["dts_last_breadth_collapse"] = float(cap)
            df["dts_last_breadth_divergence"] = float(cap)

        # ==================================================================
        # SECTION 9: Volatility Regime Events (3 features)
        # ==================================================================

        # VIX crosses above 20 from below
        if vix_col is not None:
            vix = df[vix_col].astype(float)
            vix_above_20 = vix >= 20
            vix_above_20_prev = vix_above_20.shift(1).fillna(0).astype(bool)
            crossed_20 = vix_above_20 & (~vix_above_20_prev)
            df["dts_last_vix_cross_20"] = _days_since_condition(crossed_20, cap)
        else:
            df["dts_last_vix_cross_20"] = float(cap)

        # Vol compression: 10d realized vol hits 60d low
        realized_vol_10 = daily_return.rolling(10, min_periods=5).std()
        rv_60_min = realized_vol_10.rolling(60, min_periods=20).min()
        df["dts_last_vol_compression"] = _days_since_condition(
            realized_vol_10 <= rv_60_min, cap
        )

        # Vol expansion: 10d realized vol hits 60d 90th percentile
        rv_60_q90 = realized_vol_10.rolling(60, min_periods=20).quantile(0.9)
        df["dts_last_vol_expansion"] = _days_since_condition(
            realized_vol_10 >= rv_60_q90, cap
        )

        # ==================================================================
        # SECTION 10: Sector Rotation Events (2 features)
        # ==================================================================

        rotation_col = _find_col(df, [
            "sector_cyclical_vs_defensive", "sector_rotation",
        ])

        if rotation_col is not None:
            rotation = df[rotation_col].astype(float)
            # Defensive takeover: defensives outperform cyclicals by 0.5%+
            df["dts_last_defensive_takeover"] = _days_since_condition(
                rotation <= -0.005, cap
            )
            # Cyclical breakout: cyclicals outperform defensives by 0.5%+
            df["dts_last_cyclical_breakout"] = _days_since_condition(
                rotation >= 0.005, cap
            )
        else:
            df["dts_last_defensive_takeover"] = float(cap)
            df["dts_last_cyclical_breakout"] = float(cap)

        # ==================================================================
        # SECTION 11: Momentum Exhaustion Events (3 features)
        # ==================================================================

        # Use existing RSI column if available, otherwise compute
        rsi_col = _find_col(df, ["rsi_14", "RSI_14", "rsi"])
        if rsi_col is not None:
            rsi = df[rsi_col].astype(float)
        else:
            rsi = _compute_rsi(close, period=14)

        df["dts_last_overbought"] = _days_since_condition(rsi > 70, cap)
        df["dts_last_oversold"] = _days_since_condition(rsi < 30, cap)

        # Momentum divergence: price at 20d high but RSI below its 20d high
        at_price_high = close >= close.rolling(20, min_periods=5).max()
        rsi_20_max = rsi.rolling(20, min_periods=5).max()
        rsi_below_peak = rsi < rsi_20_max
        df["dts_last_momentum_divergence"] = _days_since_condition(
            at_price_high & rsi_below_peak, cap
        )

        # ==================================================================
        # SECTION 12: Candlestick Patterns (8 features)
        # ==================================================================

        if "open" in df.columns and "high" in df.columns and "low" in df.columns:
            o = df["open"].astype(float)
            h = df["high"].astype(float)
            lo = df["low"].astype(float)
            c = close
            body = (c - o).abs()
            full_range = h - lo + 1e-10

            # Inside day: today's range entirely inside yesterday's range
            prev_high = h.shift(1)
            prev_low = lo.shift(1)
            inside_day = (h <= prev_high) & (lo >= prev_low)
            df["dts_last_inside_day"] = _days_since_condition(inside_day, cap)

            # Outside day: today's range engulfs yesterday's range
            outside_day = (h > prev_high) & (lo < prev_low)
            df["dts_last_outside_day"] = _days_since_condition(outside_day, cap)

            # Doji: body < 10% of full range
            doji = body < 0.10 * full_range
            df["dts_last_doji"] = _days_since_condition(doji, cap)

            # Hammer: lower wick > 2x body, small upper wick, bullish context
            lower_wick = pd.concat([o, c], axis=1).min(axis=1) - lo
            upper_wick = h - pd.concat([o, c], axis=1).max(axis=1)
            hammer = (lower_wick > 2.0 * body) & (upper_wick < body) & (body > 0)
            df["dts_last_hammer"] = _days_since_condition(hammer, cap)

            # Shooting star: upper wick > 2x body, small lower wick
            shooting_star = (upper_wick > 2.0 * body) & (lower_wick < body) & (body > 0)
            df["dts_last_shooting_star"] = _days_since_condition(shooting_star, cap)

            # Bullish engulfing: yesterday bearish, today bullish covers yesterday's body
            prev_bearish = (o.shift(1) > c.shift(1))
            today_bullish = c > o
            engulf_bull = (prev_bearish & today_bullish
                          & (o <= c.shift(1)) & (c >= o.shift(1)))
            df["dts_last_engulfing_bull"] = _days_since_condition(engulf_bull, cap)

            # Bearish engulfing: yesterday bullish, today bearish covers yesterday's body
            prev_bullish = (c.shift(1) > o.shift(1))
            today_bearish = o > c
            engulf_bear = (prev_bullish & today_bearish
                          & (o >= c.shift(1)) & (c <= o.shift(1)))
            df["dts_last_engulfing_bear"] = _days_since_condition(engulf_bear, cap)

            # Narrow range 7: today's range is the smallest in 7 days
            range_abs = h - lo
            range_7_min = range_abs.rolling(7, min_periods=3).min()
            nr7 = range_abs <= range_7_min
            df["dts_last_narrow_range_7"] = _days_since_condition(nr7, cap)
        else:
            for feat in ["inside_day", "outside_day", "doji", "hammer",
                         "shooting_star", "engulfing_bull", "engulfing_bear",
                         "narrow_range_7"]:
                df[f"dts_last_{feat}"] = float(cap)

        # ==================================================================
        # SECTION 13: Distribution / Accumulation (6 features)
        # ==================================================================

        if "volume" in df.columns:
            vol = df["volume"].astype(float)
            vol_ma20 = vol.rolling(20, min_periods=5).mean()

            # Distribution day: down > 0.2% on above-average volume
            dist_day = (daily_return < -0.002) & (vol > vol_ma20)
            df["dts_last_distribution_day"] = _days_since_condition(dist_day, cap)

            # Accumulation day: up > 0.2% on above-average volume
            accum_day = (daily_return > 0.002) & (vol > vol_ma20)
            df["dts_last_accumulation_day"] = _days_since_condition(accum_day, cap)

            # Stalling day: up <0.2% on above-average volume after 3+ green days
            stalling = ((daily_return >= 0) & (daily_return < 0.002)
                        & (vol > vol_ma20) & (green_streak >= 3))
            df["dts_last_stalling_day"] = _days_since_condition(stalling, cap)

            # Follow-through day: up >1.5% on above-avg volume after recent selloff
            recent_selloff = daily_return.rolling(10, min_periods=3).min() < -0.02
            fthru = (daily_return > 0.015) & (vol > vol_ma20) & recent_selloff
            df["dts_last_follow_through_day"] = _days_since_condition(fthru, cap)

            # Count features: number of dist/accum days in last 25 days
            df["dts_n_distribution_25d"] = dist_day.astype(float).rolling(
                25, min_periods=1
            ).sum()
            df["dts_n_accumulation_25d"] = accum_day.astype(float).rolling(
                25, min_periods=1
            ).sum()
        else:
            df["dts_last_distribution_day"] = float(cap)
            df["dts_last_accumulation_day"] = float(cap)
            df["dts_last_stalling_day"] = float(cap)
            df["dts_last_follow_through_day"] = float(cap)
            df["dts_n_distribution_25d"] = 0.0
            df["dts_n_accumulation_25d"] = 0.0

        # ==================================================================
        # SECTION 14: Volume-Price Divergence (7 features)
        # ==================================================================

        if "volume" in df.columns:
            vol = df["volume"].astype(float)
            vol_ma20 = vol.rolling(20, min_periods=5).mean()
            high_vol = vol > 1.5 * vol_ma20
            low_vol = vol < 0.6 * vol_ma20
            is_up = daily_return > 0
            is_down = daily_return < 0

            # Up on high volume (confirmation)
            df["dts_last_up_high_vol"] = _days_since_condition(is_up & high_vol, cap)

            # Down on high volume (selling pressure)
            df["dts_last_down_high_vol"] = _days_since_condition(is_down & high_vol, cap)

            # Up on low volume (weak rally)
            df["dts_last_up_low_vol"] = _days_since_condition(is_up & low_vol, cap)

            # Down on low volume (weak selling)
            df["dts_last_down_low_vol"] = _days_since_condition(is_down & low_vol, cap)

            # Volume breakout: vol > 3x 20-day average
            vol_breakout = vol > 3.0 * vol_ma20
            df["dts_last_vol_breakout"] = _days_since_condition(vol_breakout, cap)

            # Volume MA crossovers (20d vs 50d volume)
            vol_ma50 = vol.rolling(50, min_periods=20).mean()
            vol_above = vol_ma20 > vol_ma50
            vol_above_prev = vol_above.shift(1).fillna(False).astype(bool)
            df["dts_last_vol_ma_cross_up"] = _days_since_condition(
                vol_above & (~vol_above_prev), cap
            )
            df["dts_last_vol_ma_cross_down"] = _days_since_condition(
                (~vol_above) & vol_above_prev, cap
            )
        else:
            for feat in ["up_high_vol", "down_high_vol", "up_low_vol",
                         "down_low_vol", "vol_breakout",
                         "vol_ma_cross_up", "vol_ma_cross_down"]:
                df[f"dts_last_{feat}"] = float(cap)

        # ==================================================================
        # SECTION 15: Gap Analysis (5 features)
        # ==================================================================

        if "open" in df.columns:
            open_price = df["open"].astype(float)
            prev_close = close.shift(1)
            gap_pct = (open_price - prev_close) / (prev_close + 1e-10)

            # Gap fill up: gapped up but close fell back to prev close
            gap_up = gap_pct > 0.003
            filled_up = close <= prev_close
            df["dts_last_gap_fill_up"] = _days_since_condition(
                gap_up & filled_up, cap
            )

            # Gap fill down: gapped down but close rose back to prev close
            gap_down = gap_pct < -0.003
            filled_down = close >= prev_close
            df["dts_last_gap_fill_down"] = _days_since_condition(
                gap_down & filled_down, cap
            )

            # Unfilled gap up: gapped up and low stayed above prev close
            if "low" in df.columns:
                lo_series = df["low"].astype(float)
                unfilled_up = gap_up & (lo_series > prev_close)
                df["dts_last_unfilled_gap_up"] = _days_since_condition(
                    unfilled_up, cap
                )
            else:
                unfilled_up = gap_up & (close > prev_close)
                df["dts_last_unfilled_gap_up"] = _days_since_condition(
                    unfilled_up, cap
                )

            # Unfilled gap down: gapped down and high stayed below prev close
            if "high" in df.columns:
                hi_series = df["high"].astype(float)
                unfilled_down = gap_down & (hi_series < prev_close)
                df["dts_last_unfilled_gap_down"] = _days_since_condition(
                    unfilled_down, cap
                )
            else:
                unfilled_down = gap_down & (close < prev_close)
                df["dts_last_unfilled_gap_down"] = _days_since_condition(
                    unfilled_down, cap
                )

            # Island reversal: gap in one direction followed by gap back within 5 days
            gap_up_big = gap_pct > 0.005
            gap_down_big = gap_pct < -0.005
            recent_gap_up = gap_up_big.astype(float).rolling(5, min_periods=1).sum() > 0
            recent_gap_down = gap_down_big.astype(float).rolling(5, min_periods=1).sum() > 0
            island_rev = (gap_up_big & recent_gap_down.shift(1)) | (
                gap_down_big & recent_gap_up.shift(1)
            )
            df["dts_last_island_reversal"] = _days_since_condition(
                island_rev.fillna(False), cap
            )
        else:
            for feat in ["gap_fill_up", "gap_fill_down", "unfilled_gap_up",
                         "unfilled_gap_down", "island_reversal"]:
                df[f"dts_last_{feat}"] = float(cap)

        # ==================================================================
        # SECTION 16: Weekly Cycle (8 features)
        # ==================================================================

        # Extract date for day-of-week/month calculations
        if "date" in df.columns:
            date_series = pd.to_datetime(df["date"])
        elif isinstance(df.index, pd.DatetimeIndex):
            date_series = df.index.to_series()
        else:
            date_series = None

        if date_series is not None:
            # Week return: rolling 5-day return
            df["dts_week_return"] = close.pct_change(5)

            # Week range position: where close sits within 5-day range
            week_high = close.rolling(5, min_periods=2).max()
            week_low = close.rolling(5, min_periods=2).min()
            week_range = week_high - week_low + 1e-10
            df["dts_week_range_pos"] = (close - week_low) / week_range

            # Previous week return
            df["dts_prev_week_return"] = close.pct_change(5).shift(5)

            # Last weekly reversal: week return sign changed from prior week
            week_ret = close.pct_change(5)
            prev_week_ret = week_ret.shift(5)
            weekly_reversal = (
                ((week_ret > 0) & (prev_week_ret < 0))
                | ((week_ret < 0) & (prev_week_ret > 0))
            )
            df["dts_last_weekly_reversal"] = _days_since_condition(
                weekly_reversal.fillna(False), cap
            )

            # Month day: trading day of month (1-22ish)
            month_arr = date_series.dt.month.values
            day_count = np.ones(len(df), dtype=float)
            for i in range(1, len(df)):
                if month_arr[i] == month_arr[i - 1]:
                    day_count[i] = day_count[i - 1] + 1
                else:
                    day_count[i] = 1
            df["dts_month_day"] = day_count

            # Turn of month: first or last 3 trading days of month
            is_turn = np.zeros(len(df), dtype=float)
            for i in range(len(df)):
                if day_count[i] <= 3:
                    is_turn[i] = 1.0
                # Check if within 3 days of month end
                if i + 1 < len(df) and month_arr[i] != month_arr[i + 1]:
                    is_turn[i] = 1.0
                    if i >= 1 and month_arr[i] == month_arr[i - 1]:
                        is_turn[i - 1] = 1.0
                    if i >= 2 and month_arr[i] == month_arr[i - 2]:
                        is_turn[i - 2] = 1.0
            df["dts_is_turn_of_month"] = is_turn

            # Strong week: 5-day return > 1%
            strong_week = week_ret > 0.01
            df["dts_last_strong_week"] = _days_since_condition(
                strong_week.fillna(False), cap
            )

            # Weak week: 5-day return < -1%
            weak_week = week_ret < -0.01
            df["dts_last_weak_week"] = _days_since_condition(
                weak_week.fillna(False), cap
            )
        else:
            df["dts_week_return"] = 0.0
            df["dts_week_range_pos"] = 0.5
            df["dts_prev_week_return"] = 0.0
            df["dts_last_weekly_reversal"] = float(cap)
            df["dts_month_day"] = 1.0
            df["dts_is_turn_of_month"] = 0.0
            df["dts_last_strong_week"] = float(cap)
            df["dts_last_weak_week"] = float(cap)

        # ==================================================================
        # SECTION 17: Statistical Anomaly (5 features)
        # ==================================================================

        # 3-sigma move: |daily_return| > mean + 3*std over 60d
        ret_mean = daily_return.rolling(60, min_periods=20).mean()
        ret_std = daily_return.rolling(60, min_periods=20).std()
        sigma_3 = daily_return.abs() > (ret_mean.abs() + 3 * ret_std)
        df["dts_last_3sigma_move"] = _days_since_condition(sigma_3.fillna(False), cap)

        # Close at high: close within top 5% of day's range
        if "high" in df.columns and "low" in df.columns:
            h_stat = df["high"].astype(float)
            lo_stat = df["low"].astype(float)
            range_stat = h_stat - lo_stat + 1e-10
            close_at_high = (close - lo_stat) / range_stat >= 0.95
            close_at_low = (close - lo_stat) / range_stat <= 0.05
            df["dts_last_close_at_high"] = _days_since_condition(close_at_high, cap)
            df["dts_last_close_at_low"] = _days_since_condition(close_at_low, cap)

            # Wide range bar: today's range > 90th percentile of 60d range
            range_q90 = range_stat.rolling(60, min_periods=20).quantile(0.9)
            wide_bar = range_stat >= range_q90
            df["dts_last_wide_range_bar"] = _days_since_condition(wide_bar, cap)
        else:
            df["dts_last_close_at_high"] = float(cap)
            df["dts_last_close_at_low"] = float(cap)
            df["dts_last_wide_range_bar"] = float(cap)

        # Return streak break: green_streak or red_streak was >= 3 and just ended
        streak_break = (
            ((green_streak.shift(1) >= 3) & (is_red))
            | ((red_streak.shift(1) >= 3) & (is_green))
        )
        df["dts_last_return_streak_break"] = _days_since_condition(streak_break, cap)

        # ==================================================================
        # SECTION 18: Cross-Asset Stress Events (7 features)
        # ==================================================================

        # Yield spike / plunge (TNX)
        tnx_col = _find_col(df, [
            "econ_tnx_chg_1d", "TNX_return", "TNX_return_1d",
        ])
        if tnx_col is not None:
            tnx_chg = df[tnx_col].astype(float)
            df["dts_last_yield_spike"] = _days_since_condition(tnx_chg > 0.03, cap)
            df["dts_last_yield_plunge"] = _days_since_condition(tnx_chg < -0.03, cap)
        else:
            df["dts_last_yield_spike"] = float(cap)
            df["dts_last_yield_plunge"] = float(cap)

        # Credit stress: HYG underperforms SPY by 0.5%+ in a day
        hyg_col = _find_col(df, ["HYG_return", "HYG_return_1d", "HYG_daily_return"])
        if hyg_col is not None:
            hyg_vs_spy = df[hyg_col].astype(float) - daily_return
            df["dts_last_credit_stress"] = _days_since_condition(
                hyg_vs_spy < -0.005, cap
            )
        else:
            df["dts_last_credit_stress"] = float(cap)

        # Oil crash: USO/oil down > 3% in a day
        oil_col = _find_col(df, [
            "econ_uso_chg_1d", "USO_return", "USO_return_1d",
        ])
        if oil_col is not None:
            df["dts_last_oil_crash"] = _days_since_condition(
                df[oil_col].astype(float) < -0.03, cap
            )
        else:
            df["dts_last_oil_crash"] = float(cap)

        # EEM divergence: EEM-SPY gap > 1%
        eem_col = _find_col(df, ["EEM_return", "EEM_return_1d", "EEM_daily_return"])
        if eem_col is not None:
            eem_gap = (df[eem_col].astype(float) - daily_return).abs()
            df["dts_last_eem_diverge"] = _days_since_condition(eem_gap > 0.01, cap)
        else:
            df["dts_last_eem_diverge"] = float(cap)

        # IWM underperformance: IWM lags SPY by 0.5%+
        iwm_col = _find_col(df, ["IWM_return", "IWM_return_1d", "IWM_daily_return"])
        if iwm_col is not None:
            iwm_lag = daily_return - df[iwm_col].astype(float)
            df["dts_last_iwm_underperform"] = _days_since_condition(
                iwm_lag > 0.005, cap
            )
        else:
            df["dts_last_iwm_underperform"] = float(cap)

        # All risk-off: TLT up AND GLD up AND SPY down on same day
        tlt_up = False
        gld_up = False
        if tlt_col is not None:
            tlt_up = df[tlt_col].astype(float) > 0
        if gld_col is not None:
            gld_up = df[gld_col].astype(float) > 0
        spy_down = daily_return < 0
        if tlt_col is not None and gld_col is not None:
            all_risk_off = tlt_up & gld_up & spy_down
            df["dts_last_all_risk_off"] = _days_since_condition(all_risk_off, cap)
        else:
            df["dts_last_all_risk_off"] = float(cap)

        # ==================================================================
        # SECTION 19: Event Clustering Meta-Features (6 features)
        # Must run LAST — computed from all prior dts_ columns.
        # ==================================================================

        dts_cols_so_far = [c for c in df.columns if c.startswith("dts_")]
        # Only use true days-since columns (exclude value features)
        value_features = {
            "dts_week_return", "dts_week_range_pos", "dts_prev_week_return",
            "dts_month_day", "dts_is_turn_of_month",
            "dts_n_distribution_25d", "dts_n_accumulation_25d",
        }
        days_since_cols = [c for c in dts_cols_so_far if c not in value_features]

        if len(days_since_cols) >= 5:
            dts_matrix = df[days_since_cols].astype(float)

            # Count events within 5 days and 20 days
            events_5d = (dts_matrix <= 5).sum(axis=1).astype(float)
            events_20d = (dts_matrix <= 20).sum(axis=1).astype(float)
            df["dts_event_count_5d"] = events_5d
            df["dts_event_count_20d"] = events_20d

            # Event intensity: weighted count (closer events have more weight)
            # weight = 1 / (1 + days_since) for each event
            weights = 1.0 / (1.0 + dts_matrix)
            df["dts_event_intensity"] = weights.sum(axis=1)

            # Average recency across all features
            df["dts_avg_recency"] = dts_matrix.mean(axis=1)

            # Min recency: most recent event of any type
            df["dts_min_recency"] = dts_matrix.min(axis=1)

            # Recency dispersion: std of days-since values (low = clustered)
            df["dts_recency_dispersion"] = dts_matrix.std(axis=1)
        else:
            df["dts_event_count_5d"] = 0.0
            df["dts_event_count_20d"] = 0.0
            df["dts_event_intensity"] = 0.0
            df["dts_avg_recency"] = float(cap)
            df["dts_min_recency"] = float(cap)
            df["dts_recency_dispersion"] = 0.0

        # ==================================================================
        # CLEANUP
        # ==================================================================

        # Pre-fill value features with sensible defaults (NOT cap)
        value_defaults = {
            "dts_week_return": 0.0,
            "dts_week_range_pos": 0.5,
            "dts_prev_week_return": 0.0,
            "dts_month_day": 1.0,
            "dts_is_turn_of_month": 0.0,
            "dts_n_distribution_25d": 0.0,
            "dts_n_accumulation_25d": 0.0,
            "dts_event_count_5d": 0.0,
            "dts_event_count_20d": 0.0,
            "dts_event_intensity": 0.0,
            "dts_recency_dispersion": 0.0,
        }
        for col_name, default_val in value_defaults.items():
            if col_name in df.columns:
                df[col_name] = df[col_name].fillna(default_val)

        # Fill all remaining dts_ columns with cap
        dts_cols = [c for c in df.columns if c.startswith("dts_")]
        df[dts_cols] = df[dts_cols].fillna(float(cap))

        n_added = len(df.columns) - n_before
        print(f"  [EVENT_RECENCY] Added {n_added} days-since features")

        return df

    def analyze_current_recency(self, spy_daily: pd.DataFrame) -> Optional[dict]:
        """
        Analyze current event recency state for dashboard display.

        Returns dict with recent event distances and interpretation.
        """
        dts_cols = [c for c in spy_daily.columns if c.startswith("dts_")]
        if not dts_cols:
            return None

        last_row = spy_daily[dts_cols].iloc[-1]
        conditions = {}

        # ── Stress regime ────────────────────────────────────────────
        drop_1 = last_row.get("dts_last_drop_1pct", self._max_cap)
        drop_2 = last_row.get("dts_last_drop_2pct", self._max_cap)
        conditions["days_since_1pct_drop"] = int(drop_1)
        conditions["days_since_2pct_drop"] = int(drop_2)

        if drop_1 > 20 and drop_2 > 40:
            conditions["stress_regime"] = "COMPLACENT"
        elif drop_1 <= 3 or drop_2 <= 5:
            conditions["stress_regime"] = "STRESSED"
        else:
            conditions["stress_regime"] = "NORMAL"

        # ── Trend regime ─────────────────────────────────────────────
        high_dist = last_row.get("dts_last_52w_high", self._max_cap)
        low_dist = last_row.get("dts_last_52w_low", self._max_cap)
        conditions["days_since_52w_high"] = int(high_dist)
        conditions["days_since_52w_low"] = int(low_dist)

        if high_dist <= 5:
            conditions["trend_regime"] = "AT_HIGHS"
        elif low_dist <= 10:
            conditions["trend_regime"] = "AT_LOWS"
        elif high_dist < low_dist:
            conditions["trend_regime"] = "BULLISH"
        else:
            conditions["trend_regime"] = "BEARISH"

        # ── Volatility regime ────────────────────────────────────────
        vol_comp = last_row.get("dts_last_vol_compression", self._max_cap)
        vol_exp = last_row.get("dts_last_vol_expansion", self._max_cap)
        if vol_comp < vol_exp and vol_comp <= 10:
            conditions["vol_regime"] = "COMPRESSED"
        elif vol_exp < vol_comp and vol_exp <= 10:
            conditions["vol_regime"] = "EXPANDING"
        else:
            conditions["vol_regime"] = "NORMAL"

        # ── Momentum regime ──────────────────────────────────────────
        ob = last_row.get("dts_last_overbought", self._max_cap)
        os_val = last_row.get("dts_last_oversold", self._max_cap)
        if ob <= 5:
            conditions["momentum_regime"] = "OVERBOUGHT"
        elif os_val <= 5:
            conditions["momentum_regime"] = "OVERSOLD"
        else:
            conditions["momentum_regime"] = "NEUTRAL"

        # ── Candlestick regime ──────────────────────────────────────
        engulf_bull = last_row.get("dts_last_engulfing_bull", self._max_cap)
        engulf_bear = last_row.get("dts_last_engulfing_bear", self._max_cap)
        hammer = last_row.get("dts_last_hammer", self._max_cap)
        star = last_row.get("dts_last_shooting_star", self._max_cap)
        if engulf_bull <= 3 or hammer <= 3:
            conditions["candlestick_regime"] = "BULLISH_PATTERN"
        elif engulf_bear <= 3 or star <= 3:
            conditions["candlestick_regime"] = "BEARISH_PATTERN"
        else:
            conditions["candlestick_regime"] = "NEUTRAL"

        # ── Supply/Demand regime ────────────────────────────────────
        n_dist = last_row.get("dts_n_distribution_25d", 0)
        n_accum = last_row.get("dts_n_accumulation_25d", 0)
        if n_dist >= 5 and n_dist > n_accum * 1.5:
            conditions["supply_demand_regime"] = "DISTRIBUTION"
        elif n_accum >= 5 and n_accum > n_dist * 1.5:
            conditions["supply_demand_regime"] = "ACCUMULATION"
        else:
            conditions["supply_demand_regime"] = "BALANCED"

        # ── Event clustering ────────────────────────────────────────
        ev_5d = last_row.get("dts_event_count_5d", 0)
        ev_20d = last_row.get("dts_event_count_20d", 0)
        if ev_5d >= 15:
            conditions["event_clustering"] = "HIGH_ACTIVITY"
        elif ev_20d <= 5:
            conditions["event_clustering"] = "QUIET"
        else:
            conditions["event_clustering"] = "NORMAL"

        return conditions
