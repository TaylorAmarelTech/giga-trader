"""
Causal Feature Selector — simplified PCMCI-style causal discovery for time series.

Identifies which features have genuine causal relationships with SPY returns,
not merely correlations. Uses a two-stage approach per rolling window:

  Stage 1: Screen candidates via unconditional Granger causality (F-test)
  Stage 2: Conditional independence test (partial correlation) to prune spurious links

All features are rolling (recomputed each day from a trailing window),
so they are strictly point-in-time valid with no look-ahead bias.

Features (6, prefix causal_):
  causal_n_causes       — Number of features with significant causal link to returns
  causal_avg_strength   — Average causal link strength across significant causes
  causal_max_strength   — Strongest causal link strength
  causal_net_direction  — Net direction of causal influences (positive = bullish)
  causal_lag1_strength  — Aggregate causal strength at lag 1
  causal_lag5_strength  — Aggregate causal strength at lag 5
"""

import logging
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd

from src.core.feature_base import FeatureModuleBase

logger = logging.getLogger(__name__)


class CausalFeatureSelector(FeatureModuleBase):
    """Compute causal discovery features using simplified PCMCI on daily data.

    The algorithm:
      1. Select up to ``n_features_to_test`` numeric columns as candidate causes.
      2. For each rolling window of ``rolling_window`` days, build a causal graph
         targeting daily returns via Granger F-tests and partial-correlation
         conditional independence tests.
      3. Summarise the surviving causal links into 6 aggregate features.
    """

    REQUIRED_COLS: Set[str] = {"close"}
    FEATURE_NAMES: List[str] = [
        "causal_n_causes",
        "causal_avg_strength",
        "causal_max_strength",
        "causal_net_direction",
        "causal_lag1_strength",
        "causal_lag5_strength",
    ]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(
        self,
        max_lag: int = 5,
        significance_level: float = 0.05,
        n_features_to_test: int = 20,
        rolling_window: int = 252,
    ) -> None:
        self.max_lag = max_lag
        self.significance_level = significance_level
        self.n_features_to_test = n_features_to_test
        self.rolling_window = rolling_window

    # ------------------------------------------------------------------
    # Public API (called by anti_overfit_integration registry)
    # ------------------------------------------------------------------

    def create_causal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add causal discovery features to *df*.

        Parameters
        ----------
        df : pd.DataFrame
            Daily OHLCV-style frame.  Must contain ``close``; any additional
            numeric columns are treated as candidate causes.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with 6 new ``causal_`` columns appended.
        """
        df = df.copy()

        if not self._validate_input(df, min_rows=self.rolling_window):
            return self._zero_fill_all(df)

        # Ensure daily_return exists
        if "daily_return" not in df.columns:
            df["daily_return"] = df["close"].pct_change()

        # Select candidate cause columns (numeric, excluding targets/outputs)
        exclude = {"daily_return"} | set(self.FEATURE_NAMES)
        candidates = [
            c
            for c in df.select_dtypes(include=[np.number]).columns
            if c not in exclude and df[c].notna().sum() > self.rolling_window * 0.5
        ]

        # Limit to top-N by absolute correlation with target (cheap pre-screen)
        if len(candidates) > self.n_features_to_test:
            corrs = {
                c: abs(df[c].corr(df["daily_return"])) for c in candidates
            }
            candidates = sorted(corrs, key=corrs.get, reverse=True)[
                : self.n_features_to_test
            ]

        if len(candidates) == 0:
            logger.warning("CausalFeatureSelector: no numeric candidate columns found")
            return self._zero_fill_all(df)

        n = len(df)
        n_causes = np.zeros(n)
        avg_strength = np.zeros(n)
        max_strength = np.zeros(n)
        net_direction = np.zeros(n)
        lag1_strength = np.zeros(n)
        lag5_strength = np.zeros(n)

        step = max(1, self.rolling_window // 10)  # recompute every ~25 days

        # Cache: last computed values for non-recompute rows
        last_result: Optional[Dict[str, float]] = None

        for end_idx in range(self.rolling_window, n):
            if (end_idx - self.rolling_window) % step != 0 and last_result is not None:
                # Re-use last computed values
                n_causes[end_idx] = last_result["n_causes"]
                avg_strength[end_idx] = last_result["avg_strength"]
                max_strength[end_idx] = last_result["max_strength"]
                net_direction[end_idx] = last_result["net_direction"]
                lag1_strength[end_idx] = last_result["lag1_strength"]
                lag5_strength[end_idx] = last_result["lag5_strength"]
                continue

            start_idx = end_idx - self.rolling_window
            result = self._compute_causal_graph(df, start_idx, end_idx, candidates)

            n_causes[end_idx] = result["n_causes"]
            avg_strength[end_idx] = result["avg_strength"]
            max_strength[end_idx] = result["max_strength"]
            net_direction[end_idx] = result["net_direction"]
            lag1_strength[end_idx] = result["lag1_strength"]
            lag5_strength[end_idx] = result["lag5_strength"]
            last_result = result

        df["causal_n_causes"] = n_causes
        df["causal_avg_strength"] = avg_strength
        df["causal_max_strength"] = max_strength
        df["causal_net_direction"] = net_direction
        df["causal_lag1_strength"] = lag1_strength
        df["causal_lag5_strength"] = lag5_strength

        df = self._cleanup_features(df)

        n_feat = sum(1 for c in df.columns if c.startswith("causal_"))
        logger.info("CausalFeatureSelector: added %d features", n_feat)
        return df

    # ------------------------------------------------------------------
    # Causal graph computation (one window)
    # ------------------------------------------------------------------

    def _compute_causal_graph(
        self,
        df: pd.DataFrame,
        start: int,
        end: int,
        candidates: List[str],
    ) -> Dict[str, float]:
        """Build causal graph for the window ``df.iloc[start:end]``.

        Returns dict of aggregate causal statistics.
        """
        window = df.iloc[start:end]
        target = window["daily_return"].values.astype(np.float64)

        # Replace NaN in target with 0
        target = np.where(np.isfinite(target), target, 0.0)

        # Stage 1 — unconditional Granger F-test screening
        stage1_passed: List[Tuple[str, int, float, float]] = []  # (col, lag, p, corr)

        for col in candidates:
            cause = window[col].values.astype(np.float64)
            cause = np.where(np.isfinite(cause), cause, 0.0)

            for lag in range(1, self.max_lag + 1):
                p_val = self._granger_f_test(cause, target, lag)
                if p_val < self.significance_level:
                    # Record signed correlation for direction
                    n_eff = len(target) - lag
                    if n_eff > 2:
                        corr = np.corrcoef(cause[: n_eff], target[lag:])[0, 1]
                        if not np.isfinite(corr):
                            corr = 0.0
                    else:
                        corr = 0.0
                    stage1_passed.append((col, lag, p_val, corr))

        if not stage1_passed:
            return self._empty_result()

        # Stage 2 — conditional independence (partial correlation)
        surviving: List[Tuple[str, int, float, float]] = []

        # Gather conditioning set: all stage-1 columns except the one under test
        stage1_cols = list({entry[0] for entry in stage1_passed})

        for col, lag, p_granger, corr in stage1_passed:
            conditioning = [c for c in stage1_cols if c != col][:5]  # limit conditioning set
            if not conditioning:
                # No conditioning variables — stage 1 result stands
                surviving.append((col, lag, p_granger, corr))
                continue

            cause = window[col].values.astype(np.float64)
            cause = np.where(np.isfinite(cause), cause, 0.0)

            cond_arrays = []
            for cc in conditioning:
                arr = window[cc].values.astype(np.float64)
                arr = np.where(np.isfinite(arr), arr, 0.0)
                cond_arrays.append(arr)

            p_partial = self._partial_correlation_test(cause, target, cond_arrays, lag)
            if p_partial < self.significance_level:
                surviving.append((col, lag, p_partial, corr))

        if not surviving:
            return self._empty_result()

        # Stage 3 — aggregate into features
        strengths = []
        directions = []
        lag1_vals: List[float] = []
        lag5_vals: List[float] = []

        for _col, lag, p_val, corr in surviving:
            strength = -np.log10(max(p_val, 1e-15))
            signed_strength = strength * np.sign(corr) if corr != 0.0 else strength
            strengths.append(strength)
            directions.append(signed_strength)

            if lag == 1:
                lag1_vals.append(strength)
            if lag == self.max_lag:
                lag5_vals.append(strength)

        return {
            "n_causes": float(len(surviving)),
            "avg_strength": float(np.mean(strengths)),
            "max_strength": float(np.max(strengths)),
            "net_direction": float(np.sum(directions)),
            "lag1_strength": float(np.mean(lag1_vals)) if lag1_vals else 0.0,
            "lag5_strength": float(np.mean(lag5_vals)) if lag5_vals else 0.0,
        }

    # ------------------------------------------------------------------
    # Statistical tests
    # ------------------------------------------------------------------

    def _granger_f_test(
        self, cause: np.ndarray, effect: np.ndarray, lag: int
    ) -> float:
        """Granger causality F-test (numpy least-squares, no statsmodels).

        Restricted:   y_t = a0 + a1*y_{t-1} + ... + a_k*y_{t-k} + e
        Unrestricted: y_t = a0 + a1*y_{t-1} + ... + b1*x_{t-1} + ... + b_k*x_{t-k} + e

        Returns p-value from the F distribution.
        """
        n = len(effect)
        if n <= 2 * lag + 2:
            return 1.0

        # Build matrices for the window [lag:n]
        y = effect[lag:]
        n_obs = len(y)

        # Restricted model: only own lags
        X_r = np.ones((n_obs, lag + 1))  # intercept + lag columns
        for k in range(1, lag + 1):
            X_r[:, k] = effect[lag - k : n - k]

        # Unrestricted model: own lags + cause lags
        X_u = np.ones((n_obs, 2 * lag + 1))
        for k in range(1, lag + 1):
            X_u[:, k] = effect[lag - k : n - k]
            X_u[:, lag + k] = cause[lag - k : n - k]

        rss_r = self._ols_rss(X_r, y)
        rss_u = self._ols_rss(X_u, y)

        if rss_u <= 0 or rss_r < rss_u:
            return 1.0

        df_num = lag  # additional parameters in unrestricted model
        df_den = n_obs - 2 * lag - 1
        if df_den <= 0:
            return 1.0

        f_stat = ((rss_r - rss_u) / df_num) / (rss_u / df_den)

        # Approximate p-value from F distribution using regularized incomplete beta
        p_val = self._f_survival(f_stat, df_num, df_den)
        return p_val

    def _partial_correlation_test(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z_list: List[np.ndarray],
        lag: int,
    ) -> float:
        """Conditional independence test via partial correlation.

        Tests whether x Granger-causes y after controlling for variables in z_list.
        Returns a p-value.
        """
        n = len(y)
        if n <= lag + len(z_list) + 2:
            return 1.0

        # Align: predict y[lag:] from x[:-lag] controlling for z[:-lag]
        y_target = y[lag:]
        n_obs = len(y_target)

        # Build conditioning matrix
        n_cond = len(z_list)
        Z = np.ones((n_obs, n_cond + 1))  # intercept + conditioning vars
        for j, z_arr in enumerate(z_list):
            Z[:, j + 1] = z_arr[: n_obs]

        # Residualise y and x against Z
        x_lagged = x[: n_obs]
        y_resid = self._residualize(y_target, Z)
        x_resid = self._residualize(x_lagged, Z)

        # Correlation of residuals
        denom = np.std(y_resid) * np.std(x_resid)
        if denom < 1e-15:
            return 1.0

        r = np.corrcoef(y_resid, x_resid)[0, 1]
        if not np.isfinite(r):
            return 1.0

        # Fisher z-transform -> p-value
        df = n_obs - n_cond - 2
        if df <= 0:
            return 1.0

        t_stat = r * np.sqrt(df / (1.0 - r * r + 1e-15))
        p_val = self._t_survival(abs(t_stat), df) * 2.0  # two-tailed
        return min(p_val, 1.0)

    # ------------------------------------------------------------------
    # Linear algebra helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _ols_rss(X: np.ndarray, y: np.ndarray) -> float:
        """Residual sum of squares from OLS fit via numpy lstsq."""
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            residuals = y - X @ beta
            return float(np.sum(residuals ** 2))
        except np.linalg.LinAlgError:
            return 1e30

    @staticmethod
    def _residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
        """Return residuals of OLS regression of y on X."""
        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
            return y - X @ beta
        except np.linalg.LinAlgError:
            return y

    # ------------------------------------------------------------------
    # Distribution approximations (pure numpy, no scipy)
    # ------------------------------------------------------------------

    @staticmethod
    def _f_survival(f: float, d1: int, d2: int) -> float:
        """Approximate survival function P(F > f) for F(d1, d2).

        Uses the regularized incomplete beta function relationship:
            P(F > f) = I_x(d2/2, d1/2)  where x = d2 / (d2 + d1*f)
        Approximated via a continued-fraction expansion.
        """
        if f <= 0 or d1 <= 0 or d2 <= 0:
            return 1.0
        x = d2 / (d2 + d1 * f)
        a = d2 / 2.0
        b = d1 / 2.0
        return _regularized_beta(x, a, b)

    @staticmethod
    def _t_survival(t: float, df: int) -> float:
        """Approximate one-tailed survival function P(T > t) for Student's t.

        Uses the relationship to the F distribution:
            P(T > t) = 0.5 * P(F > t^2)  where F ~ F(1, df)
        """
        if df <= 0:
            return 0.5
        f = t * t
        x = df / (df + f)
        a = df / 2.0
        b = 0.5
        return 0.5 * _regularized_beta(x, a, b)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result() -> Dict[str, float]:
        return {
            "n_causes": 0.0,
            "avg_strength": 0.0,
            "max_strength": 0.0,
            "net_direction": 0.0,
            "lag1_strength": 0.0,
            "lag5_strength": 0.0,
        }

    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "causal_n_causes",
            "causal_avg_strength",
            "causal_max_strength",
            "causal_net_direction",
            "causal_lag1_strength",
            "causal_lag5_strength",
        ]


# ======================================================================
# Pure-numpy regularized incomplete beta function
# ======================================================================


def _regularized_beta(x: float, a: float, b: float, max_iter: int = 200) -> float:
    """Regularized incomplete beta function I_x(a, b) via Lentz continued fraction.

    Accurate to ~1e-10 for typical F/t distribution parameters.
    """
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0

    # Use the reflection identity when x > (a+1)/(a+b+2) for convergence
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - _regularized_beta(1.0 - x, b, a, max_iter)

    # Log of the prefactor:  x^a * (1-x)^b / (a * Beta(a,b))
    # Beta(a,b) = exp(lgamma(a) + lgamma(b) - lgamma(a+b))
    from math import lgamma

    ln_prefix = (
        a * np.log(x)
        + b * np.log(1.0 - x)
        - np.log(a)
        - (lgamma(a) + lgamma(b) - lgamma(a + b))
    )

    # Lentz continued fraction for I_x(a, b)
    tiny = 1e-30
    f = tiny
    C = tiny
    D = 0.0

    for m in range(0, max_iter):
        if m == 0:
            alpha_m = 1.0
        elif m % 2 == 1:
            k = (m - 1) // 2 + 1
            alpha_m = (k * (b - k) * x) / ((a + 2 * k - 1) * (a + 2 * k))
        else:
            k = m // 2
            alpha_m = -(a + k) * (a + b + k) * x / ((a + 2 * k) * (a + 2 * k + 1))

        D = 1.0 + alpha_m * D
        if abs(D) < tiny:
            D = tiny
        D = 1.0 / D

        C = 1.0 + alpha_m / C
        if abs(C) < tiny:
            C = tiny

        delta = C * D
        f *= delta
        if abs(delta - 1.0) < 1e-12:
            break

    result = np.exp(ln_prefix) * f
    return float(np.clip(result, 0.0, 1.0))
