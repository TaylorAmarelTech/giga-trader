"""
Absorption Ratio Features — systemic risk via PCA eigenvalue concentration.

Kritzman et al. (2011) Absorption Ratio: fraction of total variance explained
by top eigenvectors of a cross-asset return matrix. Higher AR = more tightly
coupled markets = more systemic risk.

Features (3, prefix ar_):
  ar_ratio       — Absorption ratio (top-3 eigenvalues / total variance)
  ar_change_20d  — 20-day delta of ar_ratio
  ar_z           — 100-day z-score of ar_ratio, clipped [-4, 4]
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from src.core.feature_base import FeatureModuleBase

logger = logging.getLogger(__name__)

# Cross-asset return columns expected from anti-overfit integration
_CROSS_ASSET_RETURN_COLS = [
    "TLT_return", "QQQ_return", "GLD_return",
    "IWM_return", "EEM_return", "HYG_return", "VXX_return",
]

_N_TOP_EIGENVALUES = 3
_ROLLING_WINDOW = 60
_ROLLING_MIN_PERIODS = 30
_Z_WINDOW = 100
_Z_MIN_PERIODS = 40


class AbsorptionRatioFeatures(FeatureModuleBase):
    """Compute absorption ratio features from cross-asset or single-asset data."""
    FEATURE_NAMES = ["ar_ratio", "ar_change_20d", "ar_z"]


    REQUIRED_COLS = {"close"}

    def create_absorption_ratio_features(self, df_daily: pd.DataFrame) -> pd.DataFrame:
        """
        Add absorption ratio features to df_daily.

        Parameters
        ----------
        df_daily : pd.DataFrame
            Must have 'close' column. Cross-asset return columns optional;
            if missing, synthetic multi-scale features are constructed from close.

        Returns
        -------
        pd.DataFrame
            Original df_daily with 3 new ar_ columns added.
        """
        df = df_daily.copy()

        if "close" not in df.columns:
            logger.warning("AbsorptionRatioFeatures: 'close' column missing, skipping")
            return df

        # Build the multi-column return matrix for PCA
        return_matrix = self._build_return_matrix(df)

        if return_matrix is None or return_matrix.shape[1] < 3:
            logger.info("AbsorptionRatioFeatures: insufficient data, defaulting to 0.0")
            for col in self._all_feature_names():
                df[col] = 0.0
            return df

        # Compute rolling absorption ratio
        ar_series = self._rolling_absorption_ratio(return_matrix)
        df["ar_ratio"] = ar_series

        # 20-day change
        df["ar_change_20d"] = df["ar_ratio"].diff(20)

        # 100-day z-score
        rolling_mean = df["ar_ratio"].rolling(_Z_WINDOW, min_periods=_Z_MIN_PERIODS).mean()
        rolling_std = df["ar_ratio"].rolling(_Z_WINDOW, min_periods=_Z_MIN_PERIODS).std()
        df["ar_z"] = ((df["ar_ratio"] - rolling_mean) / (rolling_std + 1e-10)).clip(-4, 4)

        # Cleanup: NaN -> 0.0, no infinities
        for col in self._all_feature_names():
            df[col] = df[col].fillna(0.0).replace([np.inf, -np.inf], 0.0)

        n_features = sum(1 for c in df.columns if c.startswith("ar_"))
        logger.info(f"AbsorptionRatioFeatures: added {n_features} features")
        return df

    def analyze_current_absorption_ratio(self, df_daily: pd.DataFrame) -> Optional[Dict]:
        """Analyze current systemic risk regime for dashboard display."""
        if "ar_z" not in df_daily.columns or len(df_daily) < 2:
            return None

        last = df_daily.iloc[-1]
        z = last.get("ar_z", 0.0)
        ratio = last.get("ar_ratio", 0.0)

        if z > 1.5:
            regime = "HIGH_RISK"
        elif z < -1.0:
            regime = "LOW_RISK"
        else:
            regime = "MODERATE"

        return {
            "absorption_regime": regime,
            "ar_ratio": round(float(ratio), 4),
            "ar_z": round(float(z), 3),
            "ar_change_20d": round(float(last.get("ar_change_20d", 0.0)), 4),
        }

    @staticmethod
    def _all_feature_names() -> List[str]:
        """Return list of all feature column names produced by this class."""
        return ["ar_ratio", "ar_change_20d", "ar_z"]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_return_matrix(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """
        Build multi-column return matrix for PCA.

        If cross-asset return columns are present, use them directly.
        Otherwise, construct 7 synthetic features from close returns
        at different scales (5d, 10d, 20d, 50d returns, vol_5d, vol_20d, momentum).
        """
        # Check for cross-asset columns
        available = [c for c in _CROSS_ASSET_RETURN_COLS if c in df.columns]

        if len(available) >= 3:
            logger.info(
                f"AbsorptionRatioFeatures: using {len(available)} cross-asset return columns"
            )
            return df[available].copy()

        # Fallback: construct synthetic multi-scale features from close
        if "close" not in df.columns:
            return None

        logger.info("AbsorptionRatioFeatures: cross-asset columns not found, "
                     "using synthetic multi-scale features from close")

        close = df["close"]
        ret_1d = close.pct_change()

        matrix = pd.DataFrame(index=df.index)
        matrix["ret_5d"] = close.pct_change(5)
        matrix["ret_10d"] = close.pct_change(10)
        matrix["ret_20d"] = close.pct_change(20)
        matrix["ret_50d"] = close.pct_change(50)
        matrix["vol_5d"] = ret_1d.rolling(5, min_periods=3).std()
        matrix["vol_20d"] = ret_1d.rolling(20, min_periods=10).std()
        matrix["momentum"] = close / close.rolling(20, min_periods=10).mean() - 1.0

        return matrix

    def _rolling_absorption_ratio(self, return_matrix: pd.DataFrame) -> pd.Series:
        """
        Compute rolling absorption ratio over a 60-day window.

        AR = sum(top-3 eigenvalues) / sum(all eigenvalues)
        where eigenvalues come from PCA of the rolling return matrix.
        """
        n_rows = len(return_matrix)
        n_cols = return_matrix.shape[1]
        ar_values = np.full(n_rows, np.nan)

        n_components = min(n_cols, _N_TOP_EIGENVALUES)

        for i in range(n_rows):
            if i < _ROLLING_MIN_PERIODS - 1:
                continue

            start = max(0, i - _ROLLING_WINDOW + 1)
            window = return_matrix.iloc[start:i + 1]

            # Drop rows with any NaN
            window_clean = window.dropna()

            if len(window_clean) < _ROLLING_MIN_PERIODS:
                continue

            # Standardize to zero mean, unit variance within window
            std = window_clean.std()
            valid_cols = std[std > 1e-10].index
            if len(valid_cols) < 3:
                continue

            window_clean = window_clean[valid_cols]
            window_norm = (window_clean - window_clean.mean()) / (window_clean.std() + 1e-10)

            try:
                n_comp = min(n_components, len(valid_cols), len(window_norm))
                pca = PCA(n_components=n_comp)
                pca.fit(window_norm.values)

                # AR = sum of top eigenvalues / total variance
                top_variance = pca.explained_variance_ratio_[:n_comp].sum()
                ar_values[i] = top_variance
            except Exception:
                # PCA can fail on degenerate matrices
                continue

        return pd.Series(ar_values, index=return_matrix.index)
