"""
GIGA TRADER - Dynamic Copula Tail Dependence Features
======================================================
Measure extreme co-movement between SPY returns and a benchmark series
using empirical copula tail dependence coefficients.

Tail dependence captures whether two assets tend to crash (or rally)
TOGETHER — a property that standard correlation completely misses.

Formula (upper tail):
    lambda_U = P(U > 1-q, V > 1-q) / q
    where U, V are empirical CDFs and q is the quantile threshold.

Formula (lower tail):
    lambda_L = P(U < q, V < q) / q

Both are clipped to [0, 1]. Values near 1.0 indicate that extreme moves
are almost always co-located; values near 0.0 indicate independence in
the tails.

Benchmark series selection (in priority order):
    1. QQQ_return — if the column exists in the input dataframe
    2. TLT_return — if the column exists in the input dataframe
    3. Lagged SPY return (close.pct_change().shift(1)) — always available

Features generated (prefix: copula_):
    copula_upper_tail   — Upper tail dependence lambda_U (rolling 60-day window)
    copula_lower_tail   — Lower tail dependence lambda_L (crash co-movement)
    copula_tail_asymmetry — lambda_U - lambda_L (rally vs crash co-movement)
    copula_tail_z       — 60-day z-score of copula_lower_tail (risk signal)

All features are pure numpy/pandas. No external packages required.
"""

import logging
import warnings
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger("COPULA_FEATURES")


class CopulaFeatures:
    """
    Compute empirical copula tail dependence features from SPY daily data.

    The class follows the same download-then-compute pattern as other phase-08
    feature modules.  Since all computation is purely from existing price
    columns (close + optional cross-asset columns already present in the
    daily dataframe), the download step is a no-op that always returns an
    empty DataFrame.

    Parameters
    ----------
    window : int
        Rolling window length in trading days (default 60).
    quantile : float
        Tail quantile threshold — the lower (upper) tail uses the bottom
        (top) ``quantile`` fraction of the empirical CDF.  Must be in
        (0, 0.5).  Default 0.10.
    """

    # Cross-asset columns preferred as the benchmark, in priority order.
    # The first column found in the input dataframe is used.
    PREFERRED_BENCHMARKS: List[str] = ["QQQ_return", "TLT_return"]

    def __init__(self, window: int = 60, quantile: float = 0.10) -> None:
        if not (0 < quantile < 0.5):
            raise ValueError(f"quantile must be in (0, 0.5), got {quantile}")
        self.window = window
        self.quantile = quantile

    # ─── Download (no-op — all data already lives in spy_daily) ─────────────

    def download_copula_data(
        self,
        start_date,  # unused — kept for API consistency
        end_date,    # unused — kept for API consistency
    ) -> pd.DataFrame:
        """
        No external data needed — copula features are computed from existing
        close (and optional cross-asset) columns.  Returns an empty DataFrame
        for API consistency with other phase-08 modules.
        """
        return pd.DataFrame()

    # ─── Feature Creation ────────────────────────────────────────────────────

    def create_copula_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create copula tail dependence features and add them to *df*.

        The input dataframe must contain a ``close`` column.  If the
        dataframe also contains ``QQQ_return`` or ``TLT_return`` those are
        used as the benchmark; otherwise the lagged SPY daily return is used
        as a self-dependence measure.

        Parameters
        ----------
        df : pd.DataFrame
            Daily SPY feature dataframe.  Must contain at least ``close``.

        Returns
        -------
        pd.DataFrame
            Original dataframe with four new ``copula_*`` columns appended.
            Returns *df* unchanged if ``close`` is not present.
        """
        if "close" not in df.columns:
            logger.warning("[COPULA] 'close' column not found — skipping copula features")
            return df

        print("\n[COPULA] Engineering tail dependence features...")

        result = df.copy()

        # ── Build the two return series ───────────────────────────────────

        spy_ret = result["close"].pct_change()

        # Select benchmark in priority order
        benchmark_col = None
        for col in self.PREFERRED_BENCHMARKS:
            if col in result.columns:
                benchmark_col = col
                break

        if benchmark_col is not None:
            benchmark_ret = result[benchmark_col].copy()
            print(f"  [COPULA] Using '{benchmark_col}' as benchmark series")
        else:
            benchmark_ret = spy_ret.shift(1)
            print("  [COPULA] Using lagged SPY return as self-dependence benchmark")

        # ── Rolling tail dependence computation ───────────────────────────

        n = len(result)
        upper_tail = np.full(n, np.nan)
        lower_tail = np.full(n, np.nan)

        spy_arr = spy_ret.to_numpy(dtype=float)
        bench_arr = benchmark_ret.to_numpy(dtype=float)

        for i in range(self.window - 1, n):
            s_window = spy_arr[i - self.window + 1 : i + 1]
            b_window = bench_arr[i - self.window + 1 : i + 1]

            # Drop rows where either series is NaN
            valid = ~(np.isnan(s_window) | np.isnan(b_window))
            s_valid = s_window[valid]
            b_valid = b_window[valid]

            n_valid = len(s_valid)
            if n_valid < max(10, self.window // 4):
                # Too few valid observations — leave as NaN
                continue

            # Empirical CDF ranks (scaled to (0, 1) to avoid ties at 0/1)
            s_rank = (np.argsort(np.argsort(s_valid)) + 1) / (n_valid + 1)
            b_rank = (np.argsort(np.argsort(b_valid)) + 1) / (n_valid + 1)

            q = self.quantile

            # Upper tail: both above (1 - q)
            upper_mask = (s_rank > 1.0 - q) & (b_rank > 1.0 - q)
            # Lower tail: both below q
            lower_mask = (s_rank < q) & (b_rank < q)

            upper_tail[i] = upper_mask.sum() / (n_valid * q)
            lower_tail[i] = lower_mask.sum() / (n_valid * q)

        # Clip to [0, 1]
        upper_tail = np.clip(upper_tail, 0.0, 1.0)
        lower_tail = np.clip(lower_tail, 0.0, 1.0)

        # ── Derived features ──────────────────────────────────────────────

        # Tail asymmetry: positive = more co-movement in rallies, negative = crashes
        tail_asymmetry = upper_tail - lower_tail

        # Z-score of lower tail (crash risk signal)
        lower_series = pd.Series(lower_tail)
        roll_mean = lower_series.rolling(self.window, min_periods=self.window // 2).mean()
        roll_std = lower_series.rolling(self.window, min_periods=self.window // 2).std()
        tail_z = (lower_series - roll_mean) / (roll_std + 1e-10)
        tail_z_arr = tail_z.to_numpy(dtype=float)

        # ── Attach to dataframe ───────────────────────────────────────────

        result["copula_upper_tail"] = upper_tail
        result["copula_lower_tail"] = lower_tail
        result["copula_tail_asymmetry"] = tail_asymmetry
        result["copula_tail_z"] = tail_z_arr

        # Fill NaN with 0 (consistent with other phase-08 modules)
        copula_cols = [
            "copula_upper_tail",
            "copula_lower_tail",
            "copula_tail_asymmetry",
            "copula_tail_z",
        ]
        result[copula_cols] = result[copula_cols].fillna(0)

        print(f"  [COPULA] Added {len(copula_cols)} tail dependence features "
              f"(window={self.window}, q={self.quantile})")
        return result

    # ─── Dashboard / Signal Analysis ─────────────────────────────────────────

    def analyze_current_copula(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Analyze current tail dependence conditions for dashboard display.

        Returns a dict with:
            copula_upper_tail    : float — latest upper tail lambda
            copula_lower_tail    : float — latest lower tail lambda
            copula_tail_asymmetry: float — latest asymmetry
            copula_tail_z        : float — latest z-score of lower tail
            tail_regime          : str   — 'CRASH_RISK' | 'NORMAL' | 'RALLY_DEPENDENT'

        Returns None if the required columns are absent or all-NaN.
        """
        required = ["copula_upper_tail", "copula_lower_tail",
                    "copula_tail_asymmetry", "copula_tail_z"]

        # Run feature creation first if features not yet present
        if not all(c in df.columns for c in required):
            df = self.create_copula_features(df)

        if not all(c in df.columns for c in required):
            return None

        # Use last non-zero (or last) row
        subset = df[required].dropna()
        if subset.empty:
            return None

        last = subset.iloc[-1]
        upper = float(last["copula_upper_tail"])
        lower = float(last["copula_lower_tail"])
        asym = float(last["copula_tail_asymmetry"])
        z = float(last["copula_tail_z"])

        # Classify tail regime
        if lower > 0.5 or z > 1.5:
            tail_regime = "CRASH_RISK"
        elif upper > 0.5 or (asym > 0.2 and upper > 0.3):
            tail_regime = "RALLY_DEPENDENT"
        else:
            tail_regime = "NORMAL"

        return {
            "copula_upper_tail": upper,
            "copula_lower_tail": lower,
            "copula_tail_asymmetry": asym,
            "copula_tail_z": z,
            "tail_regime": tail_regime,
        }
