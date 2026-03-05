"""
GIGA TRADER - Wavelet Decomposition Features
=============================================
Multi-resolution price decomposition using moving-average based
wavelet-like analysis (Haar wavelet equivalent). No PyWavelets dependency.

The decomposition separates price action into:
  - Trend components (low-frequency, captured by rolling MAs)
  - Detail components (high-frequency, residuals and short-term returns)
  - Energy measures (ratio of high-freq to low-freq variance)
  - Regime classification (trending vs noisy vs mixed)

Features generated (prefix: wav_):
  wav_trend_5d   : Residual from 5-day MA trend (normalised)
  wav_trend_3d   : Residual from 3-day MA trend (normalised)
  wav_detail_1d  : 1-day return (highest-frequency detail)
  wav_detail_2d  : 2-day return (medium-frequency detail)
  wav_energy_ratio    : rolling_10d_std(detail_1d) / rolling_10d_std(trend_5d)
  wav_trend_momentum  : 20-day slope of the trend_5d component
  wav_noise_level     : rolling_20d_std(detail_1d) — recent noise intensity
  wav_denoised_return : EMA(10) of daily returns — high-freq noise removed
  wav_regime          : +1 (noisy), -1 (trending), 0 (mixed) — categorical
  wav_cross_scale     : rolling_20d correlation(detail_1d, trend_5d)
"""

import logging
import warnings
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.core.feature_base import FeatureModuleBase

warnings.filterwarnings("ignore")

logger = logging.getLogger("WAVELET_FEATURES")

# Thresholds that define the wav_regime classification.
#
# Derivation:
#   For a perfectly linear (trending) price series the energy ratio converges
#   to ~0.5 (daily return ≈ MA-residual at 1× step size).  We therefore set
#   the TRENDING threshold at 0.7 so that smooth/trending series are labelled
#   correctly.  Values above 1.5 indicate the high-frequency noise component
#   dominates — "noisy" regime.
_NOISY_THRESHOLD = 1.5    # energy_ratio above this → noisy
_TRENDING_THRESHOLD = 0.7  # energy_ratio below this → trending

# Rolling window sizes (configurable via __init__)
_DEFAULT_SHORT_WINDOW = 3
_DEFAULT_LONG_WINDOW = 5


class WaveletFeatures(FeatureModuleBase):
    """
    Compute wavelet-like multi-resolution features from daily close prices.

    Uses successive moving-average decomposition (equivalent to Haar wavelets)
    instead of PyWavelets to avoid the external dependency.

    Parameters
    ----------
    short_window : int
        Window for the short-term trend MA (default 3).
    long_window : int
        Window for the long-term trend MA (default 5).
    """

    FEATURE_PREFIX = "wav_"

    FEATURE_NAMES = [
        "wav_trend_5d",
        "wav_trend_3d",
        "wav_detail_1d",
        "wav_detail_2d",
        "wav_energy_ratio",
        "wav_trend_momentum",
        "wav_noise_level",
        "wav_denoised_return",
        "wav_regime",
        "wav_cross_scale",
    ]

    def __init__(
        self,
        short_window: int = _DEFAULT_SHORT_WINDOW,
        long_window: int = _DEFAULT_LONG_WINDOW,
    ) -> None:
        self.short_window = short_window
        self.long_window = long_window

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def download_wavelet_data(
        self,
        start_date: datetime,
        end_date: datetime,
    ) -> pd.DataFrame:
        """
        Wavelet features are computed purely from existing price columns.

        No external data is needed; this method exists only to satisfy the
        standard module interface expected by anti_overfit_integration.

        Returns an empty DataFrame.
        """
        return pd.DataFrame()

    def create_wavelet_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add 10 wavelet-decomposition features to *df* and return the result.

        The input DataFrame must contain a ``close`` column (or one named
        ``Close``, ``CLOSE``, or ``adj_close`` — detected case-insensitively).
        If the required column is absent the DataFrame is returned unchanged.

        Parameters
        ----------
        df : pd.DataFrame
            Daily price data.  Must contain a close-price column.

        Returns
        -------
        pd.DataFrame
            Copy of *df* with ``wav_*`` columns appended.
        """
        df = df.copy()

        # Locate the close column
        close = self._get_close(df)
        if close is None:
            logger.warning(
                "[WAV] No 'close' column found — wavelet features skipped."
            )
            return df

        if len(close) < max(self.long_window, 20) + 1:
            logger.warning(
                "[WAV] Insufficient rows (%d) for wavelet decomposition "
                "(need at least %d).",
                len(close),
                max(self.long_window, 20) + 1,
            )
            # Still return df with zero-filled columns so the pipeline stays consistent
            for col in self.FEATURE_NAMES:
                df[col] = 0.0
            return df

        # ---- Core decomposition ----------------------------------------

        # Trend components — residual relative to rolling MA
        long_ma = close.rolling(self.long_window, min_periods=1).mean()
        short_ma = close.rolling(self.short_window, min_periods=1).mean()

        # wav_trend_5d / wav_trend_3d: how far price deviates from MA (normalised)
        wav_trend_5d = (close - long_ma) / (close.abs() + 1e-10)
        wav_trend_3d = (close - short_ma) / (close.abs() + 1e-10)

        # Detail components — short-horizon returns
        # wav_detail_1d: single-period return (highest frequency)
        wav_detail_1d = close.pct_change(1)          # (close - close.shift(1)) / close.shift(1)
        # wav_detail_2d: 2-period return
        wav_detail_2d = close.pct_change(2)

        # ---- Energy and regime -----------------------------------------

        eps = 1e-10
        detail_std_10d = wav_detail_1d.rolling(10, min_periods=3).std()
        trend_std_10d = wav_trend_5d.rolling(10, min_periods=3).std()

        # wav_energy_ratio: high-freq energy / low-freq energy
        wav_energy_ratio = detail_std_10d / (trend_std_10d + eps)

        # wav_trend_momentum: 20-day slope of the trend component
        wav_trend_momentum = (
            wav_trend_5d - wav_trend_5d.shift(20)
        ) / 20.0

        # wav_noise_level: rolling 20-day std of 1-day detail
        wav_noise_level = wav_detail_1d.rolling(20, min_periods=5).std()

        # wav_denoised_return: EMA(10) of daily returns — removes high-freq noise
        wav_denoised_return = wav_detail_1d.ewm(span=10, adjust=False).mean()

        # wav_regime: categorical signal from energy ratio
        wav_regime = pd.Series(0.0, index=close.index)
        wav_regime[wav_energy_ratio > _NOISY_THRESHOLD] = 1.0
        wav_regime[wav_energy_ratio < _TRENDING_THRESHOLD] = -1.0

        # wav_cross_scale: rolling 20-day correlation between detail_1d and trend_5d
        wav_cross_scale = (
            wav_detail_1d.rolling(20, min_periods=5)
            .corr(wav_trend_5d)
        )

        # ---- Assemble and clip ------------------------------------------

        feature_map = {
            "wav_trend_5d": wav_trend_5d,
            "wav_trend_3d": wav_trend_3d,
            "wav_detail_1d": wav_detail_1d,
            "wav_detail_2d": wav_detail_2d,
            "wav_energy_ratio": wav_energy_ratio,
            "wav_trend_momentum": wav_trend_momentum,
            "wav_noise_level": wav_noise_level,
            "wav_denoised_return": wav_denoised_return,
            "wav_regime": wav_regime,
            "wav_cross_scale": wav_cross_scale,
        }

        for col, series in feature_map.items():
            # Align to df's index regardless of whether df uses a DatetimeIndex
            # or integer index.  close already shares df's index, so series do too.
            clipped = self._clip_and_fill(series, col)
            df[col] = clipped

        n_features = len([c for c in df.columns if c.startswith(self.FEATURE_PREFIX)])
        logger.info("[WAV] Added %d wavelet features.", n_features)
        return df

    def analyze_current_wavelet(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Analyse the most recent wavelet state and return a summary dict.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame that already contains ``wav_*`` columns (i.e., output of
            ``create_wavelet_features``).  If the columns are absent the method
            attempts to compute them.

        Returns
        -------
        dict or None
            Keys include ``wavelet_regime`` (TRENDING / NOISY / MIXED),
            ``energy_ratio``, ``noise_level``, ``trend_momentum``, and
            ``cross_scale``.  Returns *None* if features cannot be computed.
        """
        # If wavelet columns aren't present yet, compute them first
        if "wav_regime" not in df.columns:
            close = self._get_close(df)
            if close is None:
                return None
            df = self.create_wavelet_features(df)

        if df.empty or "wav_regime" not in df.columns:
            return None

        last = df.iloc[-1]

        energy_ratio = last.get("wav_energy_ratio", np.nan)
        if energy_ratio > _NOISY_THRESHOLD:
            wavelet_regime = "NOISY"
        elif energy_ratio < _TRENDING_THRESHOLD:
            wavelet_regime = "TRENDING"
        else:
            wavelet_regime = "MIXED"

        return {
            "wavelet_regime": wavelet_regime,
            "energy_ratio": float(energy_ratio) if not pd.isna(energy_ratio) else None,
            "noise_level": float(last.get("wav_noise_level", np.nan))
            if not pd.isna(last.get("wav_noise_level", np.nan)) else None,
            "trend_momentum": float(last.get("wav_trend_momentum", np.nan))
            if not pd.isna(last.get("wav_trend_momentum", np.nan)) else None,
            "cross_scale": float(last.get("wav_cross_scale", np.nan))
            if not pd.isna(last.get("wav_cross_scale", np.nan)) else None,
            "trend_5d": float(last.get("wav_trend_5d", np.nan))
            if not pd.isna(last.get("wav_trend_5d", np.nan)) else None,
            "denoised_return": float(last.get("wav_denoised_return", np.nan))
            if not pd.isna(last.get("wav_denoised_return", np.nan)) else None,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_close(self, df: pd.DataFrame) -> Optional[pd.Series]:
        """Return the close-price series, trying several column-name variants."""
        for candidate in ("close", "Close", "CLOSE", "adj_close", "Adj Close"):
            if candidate in df.columns:
                return df[candidate].astype(float)
        return None

    @staticmethod
    def _clip_and_fill(series: pd.Series, name: str = "") -> pd.Series:
        """
        Replace NaN / Inf with 0 and clip extreme values to [-10, 10].

        wav_regime is integer-valued; all others are continuous.
        wav_energy_ratio is clipped to [0, 20] instead (always non-negative).
        """
        series = series.replace([np.inf, -np.inf], np.nan)
        series = series.fillna(0.0)

        if name == "wav_energy_ratio":
            series = series.clip(0.0, 20.0)
        elif name == "wav_regime":
            series = series.clip(-1.0, 1.0)
        else:
            series = series.clip(-10.0, 10.0)

        return series
