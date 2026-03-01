"""
GIGA TRADER - Market Structure Features
=========================================
Compression-expansion dynamics, price magnetism (attractor levels), inflection
zones, and directional bias features.  These capture the empirical observation
that volatility compression precedes expansion — periods of tight, indecisive
price action build "energy" that releases as explosive directional moves.

18 features generated (prefix: mstr_), 4 sections.

Section 1: Compression Detection (7) — squeeze, range contraction, vol compression
Section 2: Attractor / Gravity (3)   — VWAP, Volume POC, MA ribbon
Section 3: Inflection / Decision (4) — Hurst boundary, CUSUM buildup, IV-RV, confluence
Section 4: Directional Bias (4)      — position in range, volume skew, OBV, energy composite
"""

import logging
import warnings
from typing import Optional, Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger("MARKET_STRUCTURE")


class MarketStructureFeatures:
    """
    Compute market structure features from OHLCV data.

    All features use the mstr_ prefix and are computed from standard
    OHLCV columns already in df_daily.  No external downloads needed.
    """

    REQUIRED_COLS = {"close"}

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def create_market_structure_features(
        self,
        spy_daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create all 18 market structure features and merge into spy_daily.

        Requires at minimum: close column.
        Optional: open, high, low, volume (used when present for richer features).

        Returns spy_daily with new mstr_* columns added.
        """
        df = spy_daily.copy()

        print("\n[MSTR] Engineering market structure features...")

        if "close" not in df.columns:
            print("  [WARN] No 'close' column — skipping market structure")
            return df

        close = df["close"].values.astype(np.float64)
        has_ohlv = all(c in df.columns for c in ("open", "high", "low", "volume"))
        has_hl = all(c in df.columns for c in ("high", "low"))

        high = df["high"].values.astype(np.float64) if "high" in df.columns else close
        low = df["low"].values.astype(np.float64) if "low" in df.columns else close
        opn = df["open"].values.astype(np.float64) if "open" in df.columns else close
        volume = df["volume"].values.astype(np.float64) if "volume" in df.columns else np.ones(len(close))

        n = len(close)

        # Shared intermediates
        daily_range = high - low
        daily_return = np.empty(n)
        daily_return[0] = 0.0
        daily_return[1:] = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-10)

        atr_14 = self._rolling_atr(high, low, close, 14)
        atr_5 = self._rolling_atr(high, low, close, 5)
        atr_40 = self._rolling_atr(high, low, close, 40)

        # ----- Section 1: Compression Detection (7) ----- #
        print("  Section 1: Compression detection (7 features)...")

        # 1. ATR ratio 5/40
        with np.errstate(divide="ignore", invalid="ignore"):
            atr_ratio = np.where(atr_40 > 1e-10, atr_5 / atr_40, 1.0)
        df["mstr_atr_ratio_5_40"] = atr_ratio

        # 2. Bollinger Band Width percentile (120d)
        bb_mid = pd.Series(close).rolling(20, min_periods=20).mean()
        bb_std = pd.Series(close).rolling(20, min_periods=20).std()
        bb_upper = bb_mid + 2.0 * bb_std
        bb_lower = bb_mid - 2.0 * bb_std
        with np.errstate(divide="ignore", invalid="ignore"):
            bbw = np.where(bb_mid.values > 1e-10, (bb_upper.values - bb_lower.values) / bb_mid.values, 0.0)
        bbw_pctile = pd.Series(bbw).rolling(120, min_periods=60).rank(pct=True)
        df["mstr_bbw_percentile_120"] = bbw_pctile.values

        # 3. Realized vol percentile (252d)
        rv_20d = pd.Series(daily_return).rolling(20, min_periods=15).std()
        rv_pctile = rv_20d.rolling(252, min_periods=120).rank(pct=True)
        df["mstr_rv_percentile_252"] = rv_pctile.values

        # 4. Range percentile (60d)
        with np.errstate(divide="ignore", invalid="ignore"):
            range_pct = np.where(close > 1e-10, daily_range / close, 0.0)
        range_pctile = pd.Series(range_pct).rolling(60, min_periods=30).rank(pct=True)
        df["mstr_range_percentile_60"] = range_pctile.values

        # 5. NR7 count in last 21 days
        range_s = pd.Series(daily_range)
        is_nr7 = range_s == range_s.rolling(7, min_periods=7).min()
        nr_count = is_nr7.astype(float).rolling(21, min_periods=10).sum()
        df["mstr_nr_count_21"] = nr_count.values

        # 6. Squeeze on (BB inside Keltner Channel)
        kc_mid = pd.Series(close).rolling(20, min_periods=20).mean()
        kc_atr = pd.Series(atr_14).rolling(20, min_periods=20).mean()
        kc_upper = kc_mid + 1.5 * kc_atr
        kc_lower = kc_mid - 1.5 * kc_atr
        squeeze_on = ((bb_upper < kc_upper) & (bb_lower > kc_lower)).astype(float)
        df["mstr_squeeze_on"] = squeeze_on.values

        # 7. Squeeze duration (consecutive days in squeeze)
        sq = squeeze_on.values
        duration = np.zeros(n)
        for i in range(1, n):
            if sq[i] == 1.0:
                duration[i] = duration[i - 1] + 1.0
            else:
                duration[i] = 0.0
        df["mstr_squeeze_duration"] = duration

        # ----- Section 2: Attractor / Gravity (3) ----- #
        print("  Section 2: Attractor / gravity (3 features)...")

        # 8. VWAP deviation (volume-weighted average price over 20d)
        vol_s = pd.Series(volume)
        vwap_num = (pd.Series(close) * vol_s).rolling(20, min_periods=10).sum()
        vwap_den = vol_s.rolling(20, min_periods=10).sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            vwap = np.where(vwap_den.values > 1e-10, vwap_num.values / vwap_den.values, close)
        with np.errstate(divide="ignore", invalid="ignore"):
            vwap_dev = np.where(atr_14 > 1e-10, (close - vwap) / atr_14, 0.0)
        df["mstr_vwap_deviation"] = vwap_dev

        # 9. POC deviation (Point of Control from volume profile, 20d)
        poc_dev = self._compute_poc_deviation(close, volume, high, low, atr_14, window=20)
        df["mstr_poc_deviation"] = poc_dev

        # 10. MA ribbon width
        ma10 = pd.Series(close).rolling(10, min_periods=10).mean().values
        ma20 = pd.Series(close).rolling(20, min_periods=20).mean().values
        ma50 = pd.Series(close).rolling(50, min_periods=50).mean().values
        ma_stack = np.column_stack([ma10, ma20, ma50])
        ma_max = np.nanmax(ma_stack, axis=1)
        ma_min = np.nanmin(ma_stack, axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            ribbon_width = np.where(atr_14 > 1e-10, (ma_max - ma_min) / atr_14, 0.0)
        df["mstr_ma_ribbon_width"] = ribbon_width

        # ----- Section 3: Inflection / Decision Zone (4) ----- #
        print("  Section 3: Inflection / decision zone (4 features)...")

        # 11. Hurst distance from 0.5
        hurst_50d = self._rolling_hurst(close, window=50)
        df["mstr_hurst_distance_50"] = np.abs(hurst_50d - 0.5)

        # 12. CUSUM buildup
        cusum = self._compute_cusum_buildup(daily_return)
        df["mstr_cusum_buildup"] = cusum

        # 13. IV-RV spread z-score (graceful if no VIX data)
        if "opt_vix_rv_spread" in df.columns:
            iv_rv = df["opt_vix_rv_spread"].values.astype(np.float64)
        else:
            # Approximate: use 20d std of returns * sqrt(252) as RV proxy
            rv_ann = rv_20d.values * np.sqrt(252)
            # Use a simple constant VIX proxy (mean RV * 1.2) — neutral signal
            iv_rv = np.full(n, 0.0)
        iv_rv_mean = pd.Series(iv_rv).rolling(60, min_periods=30).mean()
        iv_rv_std = pd.Series(iv_rv).rolling(60, min_periods=30).std()
        with np.errstate(divide="ignore", invalid="ignore"):
            iv_rv_z = np.where(iv_rv_std.values > 1e-10,
                               (iv_rv - iv_rv_mean.values) / iv_rv_std.values, 0.0)
        df["mstr_iv_rv_spread_z"] = np.clip(iv_rv_z, -4.0, 4.0)

        # 14. Confluence score (count of S/R methods at current price)
        confluence = self._compute_confluence_score(close, high, low, vwap, ma20, ma50)
        df["mstr_confluence_score"] = confluence

        # ----- Section 4: Directional Bias (4) ----- #
        print("  Section 4: Directional bias (4 features)...")

        # 15. Close position in 20d range
        roll_high = pd.Series(high).rolling(20, min_periods=10).max().values
        roll_low = pd.Series(low).rolling(20, min_periods=10).min().values
        range_span = roll_high - roll_low
        with np.errstate(divide="ignore", invalid="ignore"):
            close_in_range = np.where(range_span > 1e-10,
                                      (close - roll_low) / range_span, 0.5)
        df["mstr_close_in_range"] = np.clip(close_in_range, 0.0, 1.0)

        # 16. Volume skew (up-volume ratio over 10d)
        up_day = (close > opn).astype(float)
        up_vol = pd.Series(volume * up_day).rolling(10, min_periods=5).sum()
        total_vol = pd.Series(volume).rolling(10, min_periods=5).sum()
        with np.errstate(divide="ignore", invalid="ignore"):
            vol_skew = np.where(total_vol.values > 1e-10,
                                up_vol.values / total_vol.values, 0.5)
        df["mstr_volume_skew"] = np.clip(vol_skew, 0.0, 1.0)

        # 17. OBV slope (20d normalized)
        obv_sign = np.sign(daily_return)
        obv = np.cumsum(volume * obv_sign)
        obv_slope = self._rolling_slope(obv, window=20)
        vol_mean = pd.Series(volume).rolling(20, min_periods=10).mean().values
        with np.errstate(divide="ignore", invalid="ignore"):
            obv_slope_norm = np.where(vol_mean > 1e-10, obv_slope / vol_mean, 0.0)
        df["mstr_obv_slope_20d"] = np.clip(obv_slope_norm, -5.0, 5.0)

        # 18. Compression energy composite
        # Weighted average of compression indicators, higher = more compressed
        comp_atr = np.clip(1.0 - atr_ratio, 0.0, 1.0)  # lower ratio → more compressed
        comp_bbw = np.clip(1.0 - bbw_pctile.fillna(0.5).values, 0.0, 1.0)
        comp_sq = np.clip(duration / 20.0, 0.0, 1.0)
        comp_nr = np.clip(nr_count.fillna(0).values / 5.0, 0.0, 1.0)
        energy = 0.30 * comp_atr + 0.30 * comp_bbw + 0.20 * comp_sq + 0.20 * comp_nr
        df["mstr_compression_energy"] = np.clip(energy, 0.0, 1.0)

        # ----- Cleanup ----- #
        mstr_cols = [c for c in df.columns if c.startswith("mstr_")]
        for col in mstr_cols:
            df[col] = df[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)

        n_added = len(mstr_cols)
        print(f"  [MSTR] Total: {n_added} market structure features added")

        return df

    def analyze_current_structure(
        self,
        spy_daily: pd.DataFrame,
    ) -> Optional[Dict]:
        """Return a snapshot of the current market structure state."""
        mstr_cols = [c for c in spy_daily.columns if c.startswith("mstr_")]
        if not mstr_cols or len(spy_daily) < 2:
            return None

        last = spy_daily.iloc[-1]

        atr_ratio = float(last.get("mstr_atr_ratio_5_40", 1.0))
        squeeze = bool(last.get("mstr_squeeze_on", 0.0))
        squeeze_dur = int(last.get("mstr_squeeze_duration", 0))
        energy = float(last.get("mstr_compression_energy", 0.0))
        close_pos = float(last.get("mstr_close_in_range", 0.5))

        if atr_ratio < 0.5 or squeeze:
            compression_regime = "COMPRESSED"
        elif atr_ratio > 1.3:
            compression_regime = "EXPANDING"
        else:
            compression_regime = "NORMAL"

        if close_pos > 0.7:
            bias = "BULLISH"
        elif close_pos < 0.3:
            bias = "BEARISH"
        else:
            bias = "NEUTRAL"

        return {
            "compression_regime": compression_regime,
            "atr_ratio": round(atr_ratio, 3),
            "squeeze_on": squeeze,
            "squeeze_duration": squeeze_dur,
            "compression_energy": round(energy, 3),
            "directional_bias": bias,
            "close_in_range": round(close_pos, 3),
        }

    # ------------------------------------------------------------------ #
    #  Private helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _rolling_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                     window: int) -> np.ndarray:
        """Average True Range via vectorized numpy."""
        n = len(close)
        tr = np.empty(n)
        tr[0] = high[0] - low[0] if high[0] != low[0] else 1e-10
        for i in range(1, n):
            tr[i] = max(
                high[i] - low[i],
                abs(high[i] - close[i - 1]),
                abs(low[i] - close[i - 1]),
            )
        atr = pd.Series(tr).rolling(window, min_periods=max(window // 2, 1)).mean().to_numpy(copy=True)
        atr[atr < 1e-10] = 1e-10
        return atr

    @staticmethod
    def _compute_poc_deviation(close: np.ndarray, volume: np.ndarray,
                               high: np.ndarray, low: np.ndarray,
                               atr: np.ndarray, window: int = 20) -> np.ndarray:
        """Approximate volume-profile Point of Control over rolling window."""
        n = len(close)
        poc_dev = np.zeros(n)
        n_bins = 20

        for i in range(window, n):
            start = i - window
            seg_close = close[start:i]
            seg_vol = volume[start:i]
            seg_low = np.min(low[start:i])
            seg_high = np.max(high[start:i])

            if seg_high - seg_low < 1e-10:
                poc_dev[i] = 0.0
                continue

            bin_edges = np.linspace(seg_low, seg_high, n_bins + 1)
            bin_idx = np.clip(
                np.digitize(seg_close, bin_edges) - 1, 0, n_bins - 1
            )
            bin_vol = np.zeros(n_bins)
            for j in range(len(seg_close)):
                bin_vol[bin_idx[j]] += seg_vol[j]

            poc_bin = np.argmax(bin_vol)
            poc_price = (bin_edges[poc_bin] + bin_edges[poc_bin + 1]) / 2.0

            if atr[i] > 1e-10:
                poc_dev[i] = (close[i] - poc_price) / atr[i]
            else:
                poc_dev[i] = 0.0

        return np.clip(poc_dev, -5.0, 5.0)

    @staticmethod
    def _rolling_hurst(close: np.ndarray, window: int = 50) -> np.ndarray:
        """Rolling R/S Hurst exponent (simplified)."""
        n = len(close)
        hurst = np.full(n, 0.5)
        returns = np.diff(np.log(np.maximum(close, 1e-10)))

        for i in range(window, n):
            seg = returns[i - window:i]
            seg_mean = np.mean(seg)
            cumdev = np.cumsum(seg - seg_mean)
            r = np.max(cumdev) - np.min(cumdev)
            s = np.std(seg, ddof=1)
            if s > 1e-10 and r > 1e-10:
                hurst[i] = np.log(r / s) / np.log(window)

        hurst = np.clip(hurst, 0.0, 1.0)
        return hurst

    @staticmethod
    def _compute_cusum_buildup(returns: np.ndarray) -> np.ndarray:
        """Symmetric CUSUM statistic (pressure before a changepoint trigger)."""
        n = len(returns)
        mean_r = pd.Series(returns).rolling(60, min_periods=30).mean().fillna(0).values
        std_r = pd.Series(returns).rolling(60, min_periods=30).std().fillna(1e-10).values

        cusum_pos = np.zeros(n)
        cusum_neg = np.zeros(n)
        threshold = 1.0  # Number of std devs before reset

        for i in range(1, n):
            s = std_r[i] if std_r[i] > 1e-10 else 1e-10
            z = (returns[i] - mean_r[i]) / s
            cusum_pos[i] = max(0.0, cusum_pos[i - 1] + z - 0.5 * threshold)
            cusum_neg[i] = max(0.0, cusum_neg[i - 1] - z - 0.5 * threshold)

        buildup = np.maximum(cusum_pos, cusum_neg)
        # Normalize by threshold so values are interpretable
        buildup = buildup / max(threshold, 1e-10)
        return np.clip(buildup, 0.0, 10.0)

    @staticmethod
    def _compute_confluence_score(close: np.ndarray, high: np.ndarray,
                                  low: np.ndarray, vwap: np.ndarray,
                                  ma20: np.ndarray, ma50: np.ndarray) -> np.ndarray:
        """Count how many S/R methods converge at current price level."""
        n = len(close)
        score = np.zeros(n)
        tolerance = 0.005  # 0.5% proximity threshold

        roll_high_20 = pd.Series(high).rolling(20, min_periods=10).max().values
        roll_low_20 = pd.Series(low).rolling(20, min_periods=10).min().values

        for i in range(50, n):
            c = close[i]
            if c < 1e-10:
                continue
            count = 0
            # 1. Near 20d MA
            if abs(c - ma20[i]) / c < tolerance:
                count += 1
            # 2. Near 50d MA
            if abs(c - ma50[i]) / c < tolerance:
                count += 1
            # 3. Near VWAP
            if abs(c - vwap[i]) / c < tolerance:
                count += 1
            # 4. Near 20d high (resistance)
            if abs(c - roll_high_20[i]) / c < tolerance:
                count += 1
            # 5. Near 20d low (support)
            if abs(c - roll_low_20[i]) / c < tolerance:
                count += 1
            score[i] = count

        return score

    @staticmethod
    def _rolling_slope(arr: np.ndarray, window: int = 20) -> np.ndarray:
        """Linear regression slope over rolling window."""
        n = len(arr)
        slope = np.zeros(n)
        x = np.arange(window, dtype=np.float64)
        x_mean = x.mean()
        x_var = np.sum((x - x_mean) ** 2)

        for i in range(window, n):
            y = arr[i - window:i]
            y_mean = np.mean(y)
            slope[i] = np.sum((x - x_mean) * (y - y_mean)) / max(x_var, 1e-10)

        return slope
