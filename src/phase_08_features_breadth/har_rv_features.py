"""
GIGA TRADER - HAR-RV Features
===============================
Heterogeneous Autoregressive Realized Volatility model features.

The HAR-RV model (Corsi, 2009) decomposes realized volatility into
daily, weekly (5d), and monthly (22d) components.  The model captures
the empirical observation that volatility cascades across timescales
— different investor types (day-traders, swing traders, institutions)
operate at different frequencies.

4 features generated (prefix: harv_).

Key insight: the RESIDUAL (actual RV minus HAR prediction) captures
unexpected volatility — vol surprises that the multi-horizon structure
cannot explain.  The component ratio (RV_1d / RV_22d) indicates
volatility compression (ratio < 1) vs expansion (ratio > 1).
"""

import logging
import warnings
from typing import Optional, Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger("HAR_RV")


class HARRVFeatures:
    """
    Generate HAR-RV model features from daily close prices.

    All features use the harv_ prefix.  Pure numpy implementation,
    no external dependencies beyond numpy/pandas.
    """

    REQUIRED_COLS = {"close"}

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def create_har_rv_features(
        self,
        spy_daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create HAR-RV features and merge into spy_daily.

        Returns spy_daily with new harv_* columns added.
        """
        df = spy_daily.copy()

        print("\n[HARV] Engineering HAR-RV features...")

        if "close" not in df.columns:
            print("  [WARN] No 'close' column — skipping HAR-RV")
            return df

        close = df["close"].values.astype(np.float64)
        n = len(close)

        if n < 30:
            print("  [WARN] Insufficient data (<30 rows) — skipping")
            for name in self._all_feature_names():
                df[name] = 0.0
            return df

        # Compute daily squared returns as realized variance proxy
        returns = np.empty(n)
        returns[0] = 0.0
        returns[1:] = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-10)

        rv_1d = returns ** 2  # Daily realized variance

        # Rolling RV at different horizons
        rv_5d = pd.Series(rv_1d).rolling(5, min_periods=3).mean().to_numpy(copy=True)
        rv_22d = pd.Series(rv_1d).rolling(22, min_periods=10).mean().to_numpy(copy=True)

        # HAR model: RV_t+1 = α + β_d * RV_1d_t + β_w * RV_5d_t + β_m * RV_22d_t
        predicted = np.zeros(n)
        residual = np.zeros(n)

        min_window = 60
        refit_interval = 20

        # Rolling OLS coefficients
        beta = np.array([0.0, 0.3, 0.3, 0.3])  # [intercept, β_d, β_w, β_m]

        for i in range(min_window, n):
            # Predict using current coefficients
            if not (np.isnan(rv_5d[i - 1]) or np.isnan(rv_22d[i - 1])):
                pred = beta[0] + beta[1] * rv_1d[i - 1] + beta[2] * rv_5d[i - 1] + beta[3] * rv_22d[i - 1]
                predicted[i] = max(pred, 0.0)
                residual[i] = rv_1d[i] - predicted[i]

            # Refit periodically via OLS
            if (i - min_window) % refit_interval == 0:
                beta = self._fit_har_ols(rv_1d, rv_5d, rv_22d, i, min_window)

        # Feature 1: HAR predicted RV
        df["harv_predicted"] = predicted

        # Feature 2: HAR residual (volatility surprise)
        df["harv_residual"] = residual

        # Feature 3: Z-scored residual
        resid_series = pd.Series(residual)
        resid_mean = resid_series.rolling(60, min_periods=30).mean()
        resid_std = resid_series.rolling(60, min_periods=30).std()
        resid_std = resid_std.replace(0.0, 1e-10)
        df["harv_residual_z"] = np.clip(((resid_series - resid_mean) / resid_std).to_numpy(copy=True), -5.0, 5.0)

        # Feature 4: Component ratio (daily / monthly — compression indicator)
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = np.where(rv_22d > 1e-15, rv_1d / rv_22d, 1.0)
        df["harv_component_ratio"] = np.clip(ratio, 0.0, 10.0)

        # Cleanup
        for col in self._all_feature_names():
            df[col] = df[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)

        print(f"  [HARV] Total: {len(self._all_feature_names())} HAR-RV features added")
        return df

    def analyze_current_harv(
        self,
        spy_daily: pd.DataFrame,
    ) -> Optional[Dict]:
        """Return snapshot of current HAR-RV state."""
        if "harv_predicted" not in spy_daily.columns or len(spy_daily) < 2:
            return None

        last = spy_daily.iloc[-1]
        resid_z = float(last.get("harv_residual_z", 0.0))
        ratio = float(last.get("harv_component_ratio", 1.0))

        if resid_z > 2.0:
            vol_regime = "VOL_SPIKE"
        elif resid_z < -1.5:
            vol_regime = "VOL_COMPRESSION"
        else:
            vol_regime = "NORMAL"

        return {
            "vol_regime": vol_regime,
            "predicted_rv": round(float(last.get("harv_predicted", 0.0)), 6),
            "residual_z": round(resid_z, 3),
            "component_ratio": round(ratio, 3),
        }

    # ------------------------------------------------------------------ #
    #  Private helpers                                                    #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _all_feature_names():
        return ["harv_predicted", "harv_residual", "harv_residual_z", "harv_component_ratio"]

    @staticmethod
    def _fit_har_ols(rv_1d, rv_5d, rv_22d, end_idx, min_window):
        """Fit HAR model via OLS on available data up to end_idx."""
        start = max(0, end_idx - 252)  # Use up to 1 year

        # Build X matrix: [1, rv_1d_{t-1}, rv_5d_{t-1}, rv_22d_{t-1}]
        # Target: rv_1d_t
        valid_idx = []
        for t in range(max(start, 22), end_idx):
            if not (np.isnan(rv_5d[t - 1]) or np.isnan(rv_22d[t - 1])):
                valid_idx.append(t)

        if len(valid_idx) < 30:
            return np.array([0.0, 0.3, 0.3, 0.3])

        idx = np.array(valid_idx)
        y = rv_1d[idx]
        X = np.column_stack([
            np.ones(len(idx)),
            rv_1d[idx - 1],
            rv_5d[idx - 1],
            rv_22d[idx - 1],
        ])

        try:
            # OLS: beta = (X'X)^-1 X'y
            XtX = X.T @ X
            XtX += np.eye(4) * 1e-8  # Ridge regularization
            Xty = X.T @ y
            beta = np.linalg.solve(XtX, Xty)
            # Clip coefficients to reasonable range
            beta = np.clip(beta, -1.0, 2.0)
            return beta
        except Exception:
            return np.array([0.0, 0.3, 0.3, 0.3])
