"""
GIGA TRADER - Block Structure Features
========================================
Multi-day block features (3-day and 5-day aggregates) that compress
daily noise into structural signals. These are fundamentally harder to
overfit than single-day lags because the model can't memorize specific
day-level patterns — it must learn regime-level structure instead.

54 features generated (prefix: blk_), 6 sections.

Section 1: Block Return Patterns (12) — direction, consistency, acceleration
Section 2: Multi-Scale Momentum Cascades (10) — cross-timescale alignment
Section 3: Intra-Block Structure (10) — shape of move within blocks
Section 4: Block Volume Profile (8) — volume confirmation at block level
Section 5: Block Boundary / Transition (8) — seam effects between blocks
Section 6: Block Autocorrelation & Texture (6) — trending vs choppy regime
"""

import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from src.core.feature_base import FeatureModuleBase

warnings.filterwarnings("ignore")

logger = logging.getLogger("BLOCK_STRUCTURE")


class BlockStructureFeatures(FeatureModuleBase):
    """
    Compute multi-day block structure features from OHLCV data.

    All features use the blk_ prefix and are computed from close/volume
    columns already in df_daily. No extra downloads needed.
    """
    FEATURE_NAMES = ["blk_3d_return", "blk_5d_return", "blk_3d_acceleration", "blk_5d_acceleration", "blk_3d_reversal_boundary", "blk_5d_reversal_boundary", "blk_3d_vs_5d_agreement", "blk_sequential_strength", "blk_cascade_3_5_10", "blk_cascade_5_10_20", "blk_cascade_full", "blk_cascade_break_short", "blk_cascade_break_long", "blk_cascade_ratio_3_10", "blk_cascade_ratio_5_20", "blk_cascade_vol_expand", "blk_cascade_vol_compress", "blk_cascade_return_rank", "blk_3d_front_vs_back", "blk_5d_front_vs_back", "blk_boundary_3d_gap", "blk_boundary_5d_gap", "blk_boundary_3d_vol_shift", "blk_boundary_5d_vol_shift", "blk_boundary_3d_revert", "blk_boundary_5d_revert", "blk_return_roughness", "blk_3d_kurtosis", "blk_5d_dispersion", "blk_hurst_proxy"]


    REQUIRED_COLS = {"close"}

    def create_block_structure_features(
        self,
        spy_daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create all 54 block structure features and merge into spy_daily.

        Requires at minimum: close column.
        Optional: volume, open, high, low.

        Returns spy_daily with new blk_* columns added.
        """
        df = spy_daily.copy()
        n_before = len(df.columns)

        print("\n[BLOCK_STRUCTURE] Engineering block structure features...")

        if "close" not in df.columns:
            print("  [WARN] No 'close' column — skipping block structure")
            return df

        close = df["close"].astype(float)
        daily_return = close.pct_change()

        # ==================================================================
        # SECTION 1: Block Return Patterns (12 features)
        # ==================================================================

        # 3-day and 5-day block returns
        df["blk_3d_return"] = close.pct_change(3)
        df["blk_5d_return"] = close.pct_change(5)

        # Direction consistency: fraction of days with majority sign
        ret_sign = np.sign(daily_return)
        for window, label in [(3, "3d"), (5, "5d")]:
            pos_count = (ret_sign > 0).astype(float).rolling(window, min_periods=1).sum()
            neg_count = (ret_sign < 0).astype(float).rolling(window, min_periods=1).sum()
            majority = pd.concat([pos_count, neg_count], axis=1).max(axis=1)
            df[f"blk_{label}_consistency"] = majority / window

        # Weighted pattern score: recent days weighted more
        # 3d: weights [1, 2, 3] / 6, 5d: weights [1, 2, 3, 4, 5] / 15
        for window, label, norm in [(3, "3d", 6.0), (5, "5d", 15.0)]:
            score = pd.Series(0.0, index=df.index)
            for lag in range(window):
                weight = float(window - lag)  # Most recent gets highest weight
                score += ret_sign.shift(lag) * weight
            df[f"blk_{label}_pattern_score"] = score / norm

        # Acceleration: is the block speeding up or fading?
        r1 = daily_return.shift(2)  # day 1 of 3-day block
        r3 = daily_return           # day 3 of 3-day block
        df["blk_3d_acceleration"] = r3 - r1

        r_d1 = daily_return.shift(4)  # day 1 of 5-day block
        r_d2 = daily_return.shift(3)  # day 2
        r_d4 = daily_return.shift(1)  # day 4
        r_d5 = daily_return           # day 5
        df["blk_5d_acceleration"] = (r_d4 + r_d5) / 2.0 - (r_d1 + r_d2) / 2.0

        # Block boundary reversal: did current block reverse from prior?
        ret_3d = close.pct_change(3)
        prev_3d = ret_3d.shift(3)
        df["blk_3d_reversal_boundary"] = (
            (np.sign(ret_3d) != np.sign(prev_3d)) & prev_3d.notna()
        ).astype(float)

        ret_5d = close.pct_change(5)
        prev_5d = ret_5d.shift(5)
        df["blk_5d_reversal_boundary"] = (
            (np.sign(ret_5d) != np.sign(prev_5d)) & prev_5d.notna()
        ).astype(float)

        # 3d vs 5d agreement
        df["blk_3d_vs_5d_agreement"] = np.where(
            np.sign(ret_3d) == np.sign(ret_5d), 1.0, -1.0
        )

        # Sequential strength: momentum continuation ratio
        seq_raw = ret_3d / (prev_3d.replace(0, np.nan) + 1e-10)
        df["blk_sequential_strength"] = seq_raw.clip(-3.0, 3.0)

        # ==================================================================
        # SECTION 2: Multi-Scale Momentum Cascades (10 features)
        # ==================================================================

        ret_3 = close.pct_change(3)
        ret_5 = close.pct_change(5)
        ret_10 = close.pct_change(10)
        ret_20 = close.pct_change(20)

        s3 = np.sign(ret_3)
        s5 = np.sign(ret_5)
        s10 = np.sign(ret_10)
        s20 = np.sign(ret_20)

        # Sign agreement across timescales: mean of pairwise agreements
        # 3 pairs for 3 timescales, 3 pairs for 3 timescales
        agree_3_5 = (s3 == s5).astype(float)
        agree_5_10 = (s5 == s10).astype(float)
        agree_3_10 = (s3 == s10).astype(float)
        df["blk_cascade_3_5_10"] = (agree_3_5 + agree_5_10 + agree_3_10) / 3.0 * 2 - 1

        agree_5_10b = (s5 == s10).astype(float)
        agree_10_20 = (s10 == s20).astype(float)
        agree_5_20 = (s5 == s20).astype(float)
        df["blk_cascade_5_10_20"] = (agree_5_10b + agree_10_20 + agree_5_20) / 3.0 * 2 - 1

        # Full cascade: all 4 agree?
        all_pos = (s3 > 0) & (s5 > 0) & (s10 > 0) & (s20 > 0)
        all_neg = (s3 < 0) & (s5 < 0) & (s10 < 0) & (s20 < 0)
        df["blk_cascade_full"] = np.where(all_pos, 1.0, np.where(all_neg, -1.0, 0.0))

        # Cascade breaks
        df["blk_cascade_break_short"] = (s3 != s10).astype(float)
        df["blk_cascade_break_long"] = (s10 != s20).astype(float)

        # Momentum ratios (clipped to prevent outliers)
        ratio_3_10 = ret_3 / (ret_10.replace(0, np.nan).fillna(1e-10))
        df["blk_cascade_ratio_3_10"] = ratio_3_10.clip(-5.0, 5.0)

        ratio_5_20 = ret_5 / (ret_20.replace(0, np.nan).fillna(1e-10))
        df["blk_cascade_ratio_5_20"] = ratio_5_20.clip(-5.0, 5.0)

        # Volume cascade (expansion/compression)
        if "volume" in df.columns:
            vol = df["volume"].astype(float)
            vol_3d = vol.rolling(3, min_periods=1).mean()
            vol_5d = vol.rolling(5, min_periods=1).mean()
            vol_10d = vol.rolling(10, min_periods=1).mean()
            df["blk_cascade_vol_expand"] = (
                (vol_3d > vol_5d) & (vol_5d > vol_10d)
            ).astype(float)
            df["blk_cascade_vol_compress"] = (
                (vol_3d < vol_5d) & (vol_5d < vol_10d)
            ).astype(float)
        else:
            df["blk_cascade_vol_expand"] = 0.0
            df["blk_cascade_vol_compress"] = 0.0

        # Return rank within 5-day block
        rank_vals = pd.Series(np.nan, index=df.index)
        abs_rets = daily_return.abs()
        for i in range(4, len(df)):
            block = abs_rets.iloc[i - 4:i + 1].values
            today_val = abs_rets.iloc[i]
            if np.isnan(today_val) or np.all(np.isnan(block)):
                rank_vals.iloc[i] = 0.5
            else:
                valid = block[~np.isnan(block)]
                rank_vals.iloc[i] = float(np.sum(valid <= today_val)) / max(len(valid), 1)
        df["blk_cascade_return_rank"] = rank_vals

        # ==================================================================
        # SECTION 3: Intra-Block Structure (10 features)
        # ==================================================================

        # Where extreme: which day in the block had the largest |return|?
        for window, label in [(3, "3d"), (5, "5d")]:
            where_vals = pd.Series(np.nan, index=df.index)
            for i in range(window - 1, len(df)):
                block = daily_return.iloc[i - window + 1:i + 1].abs().values
                if np.all(np.isnan(block)):
                    where_vals.iloc[i] = 0.5
                else:
                    max_idx = np.nanargmax(block)
                    where_vals.iloc[i] = float(max_idx) / max(window - 1, 1)
            df[f"blk_{label}_where_extreme"] = where_vals

        # Front vs back loading
        df["blk_3d_front_vs_back"] = (
            daily_return.shift(2) -
            (daily_return.shift(1) + daily_return) / 2.0
        )
        df["blk_5d_front_vs_back"] = (
            (daily_return.shift(4) + daily_return.shift(3)) / 2.0 -
            (daily_return.shift(2) + daily_return.shift(1) + daily_return) / 3.0
        )

        # Block range position
        if "high" in df.columns and "low" in df.columns:
            for window, label in [(3, "3d"), (5, "5d")]:
                blk_high = df["high"].astype(float).rolling(window, min_periods=1).max()
                blk_low = df["low"].astype(float).rolling(window, min_periods=1).min()
                blk_range = blk_high - blk_low + 1e-10
                df[f"blk_{label}_range_position"] = (close - blk_low) / blk_range
        else:
            for window, label in [(3, "3d"), (5, "5d")]:
                c_high = close.rolling(window, min_periods=1).max()
                c_low = close.rolling(window, min_periods=1).min()
                c_range = c_high - c_low + 1e-10
                df[f"blk_{label}_range_position"] = (close - c_low) / c_range

        # Max drawdown within block
        for window, label in [(3, "3d"), (5, "5d")]:
            dd_vals = pd.Series(np.nan, index=df.index)
            rec_vals = pd.Series(np.nan, index=df.index)
            for i in range(window - 1, len(df)):
                block_close = close.iloc[i - window + 1:i + 1].values
                if np.any(np.isnan(block_close)) or block_close[0] == 0:
                    dd_vals.iloc[i] = 0.0
                    rec_vals.iloc[i] = 0.0
                    continue
                # Normalize to block start
                normed = block_close / block_close[0]
                peak = np.maximum.accumulate(normed)
                drawdowns = (normed - peak) / (peak + 1e-10)
                max_dd = abs(drawdowns.min())
                dd_vals.iloc[i] = max_dd

                # Recovery: how much of the dd was recovered?
                if max_dd > 1e-6:
                    dd_idx = int(np.argmin(drawdowns))
                    if dd_idx < window - 1:
                        recovery = (normed[-1] - normed[dd_idx]) / (
                            peak[dd_idx] - normed[dd_idx] + 1e-10
                        )
                        rec_vals.iloc[i] = min(max(recovery, 0.0), 1.0)
                    else:
                        rec_vals.iloc[i] = 0.0
                else:
                    rec_vals.iloc[i] = 1.0  # No drawdown = fully recovered

            df[f"blk_{label}_max_drawdown"] = dd_vals
            df[f"blk_{label}_recovery_ratio"] = rec_vals

        # ==================================================================
        # SECTION 4: Block Volume Profile (8 features)
        # ==================================================================

        if "volume" in df.columns:
            vol = df["volume"].astype(float)
            vol_chg = vol.pct_change()

            # Volume trend (normalized slope proxy via first-last diff)
            vol_mean = vol.rolling(20, min_periods=5).mean() + 1e-10
            for window, label in [(3, "3d"), (5, "5d")]:
                vol_start = vol.shift(window - 1)
                vol_end = vol
                df[f"blk_{label}_vol_trend"] = (
                    (vol_end - vol_start) / vol_mean
                ).clip(-3.0, 3.0)

            # Volume-return agreement: do they move together?
            ret_sign_f = np.sign(daily_return)
            vol_chg_sign = np.sign(vol_chg)
            for window, label in [(3, "3d"), (5, "5d")]:
                agree = (ret_sign_f == vol_chg_sign).astype(float)
                df[f"blk_{label}_vol_return_agree"] = agree.rolling(
                    window, min_periods=1
                ).mean()

            # Volume concentration: what fraction of block volume on biggest day?
            for window, label in [(3, "3d"), (5, "5d")]:
                vol_max = vol.rolling(window, min_periods=1).max()
                vol_sum = vol.rolling(window, min_periods=1).sum() + 1e-10
                df[f"blk_{label}_vol_concentration"] = vol_max / vol_sum

            # VWAP vs close approximation (volume-weighted avg close)
            for window, label in [(3, "3d"), (5, "5d")]:
                vol_close = (close * vol).rolling(window, min_periods=1).sum()
                vol_total = vol.rolling(window, min_periods=1).sum() + 1e-10
                vwap = vol_close / vol_total
                df[f"blk_{label}_vwap_vs_close"] = (vwap - close) / (close + 1e-10)
        else:
            for feat in [
                "blk_3d_vol_trend", "blk_5d_vol_trend",
                "blk_3d_vol_return_agree", "blk_5d_vol_return_agree",
                "blk_3d_vol_concentration", "blk_5d_vol_concentration",
                "blk_3d_vwap_vs_close", "blk_5d_vwap_vs_close",
            ]:
                df[feat] = 0.0

        # ==================================================================
        # SECTION 5: Block Boundary / Transition (8 features)
        # ==================================================================

        # Boundary gap: return at the seam between blocks
        if "open" in df.columns:
            open_price = df["open"].astype(float)
            prev_close_3 = close.shift(3)
            open_at_boundary_3 = open_price.shift(2)
            df["blk_boundary_3d_gap"] = (
                (open_at_boundary_3 - prev_close_3) / (prev_close_3 + 1e-10)
            )

            prev_close_5 = close.shift(5)
            open_at_boundary_5 = open_price.shift(4)
            df["blk_boundary_5d_gap"] = (
                (open_at_boundary_5 - prev_close_5) / (prev_close_5 + 1e-10)
            )
        else:
            # Approximate with close-to-close
            df["blk_boundary_3d_gap"] = daily_return.shift(2)
            df["blk_boundary_5d_gap"] = daily_return.shift(4)

        # Boundary volume shift
        if "volume" in df.columns:
            vol = df["volume"].astype(float)
            df["blk_boundary_3d_vol_shift"] = (
                vol.shift(2) / (vol.shift(3) + 1e-10)
            ).clip(0.1, 10.0)
            df["blk_boundary_5d_vol_shift"] = (
                vol.shift(4) / (vol.shift(5) + 1e-10)
            ).clip(0.1, 10.0)
        else:
            df["blk_boundary_3d_vol_shift"] = 1.0
            df["blk_boundary_5d_vol_shift"] = 1.0

        # Block streak: consecutive same-sign blocks
        for period, label in [(3, "3d"), (5, "5d")]:
            blk_ret = close.pct_change(period)
            blk_sign = np.sign(blk_ret)
            streak = pd.Series(1.0, index=df.index)
            prev_sign = blk_sign.shift(period)
            for i in range(period, len(df)):
                if (blk_sign.iloc[i] == prev_sign.iloc[i] and
                        not np.isnan(blk_sign.iloc[i]) and
                        blk_sign.iloc[i] != 0):
                    streak.iloc[i] = streak.iloc[i - period] + 1
                else:
                    streak.iloc[i] = 1.0
            df[f"blk_boundary_{label}_streak"] = streak

        # Reversion pressure: z-score from rolling mean
        ma15 = close.rolling(15, min_periods=5).mean()
        std15 = close.rolling(15, min_periods=5).std() + 1e-10
        df["blk_boundary_3d_revert"] = ((close - ma15) / std15).clip(-4.0, 4.0)

        ma25 = close.rolling(25, min_periods=10).mean()
        std25 = close.rolling(25, min_periods=10).std() + 1e-10
        df["blk_boundary_5d_revert"] = ((close - ma25) / std25).clip(-4.0, 4.0)

        # ==================================================================
        # SECTION 6: Block Autocorrelation & Texture (6 features)
        # ==================================================================

        # Autocorrelation of block returns (rolling window)
        for period, label, acf_window in [(3, "3d", 60), (5, "5d", 50)]:
            blk_ret = close.pct_change(period)
            blk_ret_lag = blk_ret.shift(period)
            df[f"blk_{label}_autocorr"] = blk_ret.rolling(
                acf_window, min_periods=max(10, period * 3)
            ).corr(blk_ret_lag)

        # Return roughness: path length vs displacement
        abs_daily_sum = daily_return.abs().rolling(5, min_periods=2).sum()
        displacement = ret_5.abs() + 1e-10
        df["blk_return_roughness"] = (abs_daily_sum / displacement).clip(0.0, 20.0)

        # Kurtosis of returns over 15 days (captures fat tails)
        df["blk_3d_kurtosis"] = daily_return.rolling(15, min_periods=8).kurt()
        # Clip extreme kurtosis
        df["blk_3d_kurtosis"] = df["blk_3d_kurtosis"].clip(-10.0, 50.0)

        # Dispersion of 5d returns over last 50 days
        df["blk_5d_dispersion"] = ret_5.rolling(50, min_periods=10).std()

        # Hurst proxy: sign persistence ratio
        sign_same = (ret_sign == ret_sign.shift(1)).astype(float)
        df["blk_hurst_proxy"] = sign_same.rolling(20, min_periods=5).mean()

        # ==================================================================
        # CLEANUP
        # ==================================================================

        blk_cols = [c for c in df.columns if c.startswith("blk_")]
        df[blk_cols] = df[blk_cols].fillna(0.0)

        # Replace any inf/-inf with 0
        for col in blk_cols:
            df[col] = df[col].replace([np.inf, -np.inf], 0.0)

        n_added = len(df.columns) - n_before
        print(f"  [BLOCK_STRUCTURE] Added {n_added} block structure features")

        return df

    def analyze_current_structure(
        self, spy_daily: pd.DataFrame
    ) -> Optional[dict]:
        """
        Analyze current block structure state for dashboard display.

        Returns dict with regime signals and key metrics.
        """
        blk_cols = [c for c in spy_daily.columns if c.startswith("blk_")]
        if not blk_cols:
            return None

        last = spy_daily[blk_cols].iloc[-1]
        conditions = {}

        # ── Momentum cascade regime ─────────────────────────────────
        cascade = last.get("blk_cascade_full", 0.0)
        if cascade == 1.0:
            conditions["cascade_regime"] = "FULL_BULL"
        elif cascade == -1.0:
            conditions["cascade_regime"] = "FULL_BEAR"
        else:
            conditions["cascade_regime"] = "MIXED"

        # ── Block trend regime ──────────────────────────────────────
        consistency_5d = last.get("blk_5d_consistency", 0.5)
        ret_5d = last.get("blk_5d_return", 0.0)
        if consistency_5d >= 0.8 and ret_5d > 0:
            conditions["block_trend"] = "STRONG_UP"
        elif consistency_5d >= 0.8 and ret_5d < 0:
            conditions["block_trend"] = "STRONG_DOWN"
        elif consistency_5d <= 0.4:
            conditions["block_trend"] = "CHOPPY"
        else:
            conditions["block_trend"] = "NORMAL"

        # ── Texture regime ──────────────────────────────────────────
        roughness = last.get("blk_return_roughness", 1.0)
        hurst = last.get("blk_hurst_proxy", 0.5)
        if roughness > 5.0 or hurst < 0.35:
            conditions["texture_regime"] = "MEAN_REVERTING"
        elif roughness < 2.0 and hurst > 0.6:
            conditions["texture_regime"] = "TRENDING"
        else:
            conditions["texture_regime"] = "NEUTRAL"

        # ── Key metrics ─────────────────────────────────────────────
        conditions["blk_5d_return"] = round(float(ret_5d), 4)
        conditions["blk_3d_return"] = round(float(last.get("blk_3d_return", 0)), 4)
        conditions["cascade_alignment"] = round(float(cascade), 1)

        return conditions
