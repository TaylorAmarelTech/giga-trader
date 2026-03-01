"""
Wavelet Scattering Features -- translation-invariant multi-scale features.

Scattering transforms are stable to deformations and capture multi-scale
frequency interactions. Unlike raw wavelet coefficients, scattering
coefficients are invariant to time-shifts and robust to noise.

The scattering transform cascades:
  Layer 0: Low-pass averaging (captures mean)
  Layer 1: |W1 * x| -> average (captures first-order frequency content)
  Layer 2: |W2 * |W1 * x|| -> average (captures intermittency/burstiness)

Features (12, prefix wscat_):
  wscat_s0_20d          -- Layer-0: mean |return| over 20 days
  wscat_s1_scale2_20d   -- Layer-1 scattering at scale 2 (high-freq energy)
  wscat_s1_scale4_20d   -- Layer-1 scattering at scale 4 (mid-freq energy)
  wscat_s1_scale8_20d   -- Layer-1 scattering at scale 8 (low-freq energy)
  wscat_s2_2x4_20d      -- Layer-2 scattering (scale 2->4 interaction, intermittency)
  wscat_s2_2x8_20d      -- Layer-2 scattering (scale 2->8 interaction)
  wscat_s2_4x8_20d      -- Layer-2 scattering (scale 4->8 interaction)
  wscat_energy_ratio    -- s1_scale2 / (s1_scale8 + 1e-10) -- high/low freq ratio
  wscat_intermittency   -- s2_2x8 / (s1_scale2 * s1_scale8 + 1e-10)
  wscat_s0_z            -- 60-day z-score of s0
  wscat_energy_trend    -- 10-day slope of s1_scale4 (frequency energy trend)
  wscat_regime          -- 1 if energy_ratio > 2x its 60-day median
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.signal import fftconvolve

logger = logging.getLogger(__name__)


def _ricker_wavelet(points: int, a: float) -> np.ndarray:
    """Generate a Ricker (Mexican hat) wavelet.

    This is a self-contained implementation equivalent to the former
    ``scipy.signal.ricker`` (removed in scipy >= 1.15).

    Parameters
    ----------
    points : int
        Number of points in the wavelet.
    a : float
        Width parameter of the wavelet.

    Returns
    -------
    np.ndarray
        Array of length *points* containing the wavelet values.
    """
    A = 2.0 / (np.sqrt(3.0 * a) * (np.pi ** 0.25))
    wsq = a ** 2
    vec = np.arange(0, points) - (points - 1.0) / 2.0
    tsq = vec ** 2
    mod = 1.0 - tsq / wsq
    gauss = np.exp(-tsq / (2.0 * wsq))
    return A * mod * gauss

# Scattering transform scales (measured in trading days).
_SCALES = [2, 4, 8]

# Rolling window for scattering coefficient averaging.
_AVERAGING_WINDOW = 20

# Z-score lookback for s0_z.
_ZSCORE_WINDOW = 60

# Energy trend slope lookback.
_ENERGY_TREND_WINDOW = 10


class WaveletScatteringFeatures:
    """Compute wavelet scattering transform features from daily OHLCV data.

    Uses the Ricker (Mexican hat) wavelet as the mother wavelet.  The
    scattering transform cascades wavelet modulus
    operators to produce translation-invariant, multi-scale descriptors
    that are stable to small deformations of the input signal.

    Parameters
    ----------
    scales : list[int], optional
        Wavelet scales in trading days.  Default ``[2, 4, 8]``.
    averaging_window : int, optional
        Rolling window for low-pass averaging of scattering coefficients.
        Default ``20``.
    """

    REQUIRED_COLS = {"close"}

    FEATURE_NAMES = [
        "wscat_s0_20d",
        "wscat_s1_scale2_20d",
        "wscat_s1_scale4_20d",
        "wscat_s1_scale8_20d",
        "wscat_s2_2x4_20d",
        "wscat_s2_2x8_20d",
        "wscat_s2_4x8_20d",
        "wscat_energy_ratio",
        "wscat_intermittency",
        "wscat_s0_z",
        "wscat_energy_trend",
        "wscat_regime",
    ]

    def __init__(
        self,
        scales: Optional[List[int]] = None,
        averaging_window: int = _AVERAGING_WINDOW,
    ) -> None:
        self.scales = scales or list(_SCALES)
        self.averaging_window = averaging_window

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def create_wavelet_scattering_features(
        self, df_daily: pd.DataFrame
    ) -> pd.DataFrame:
        """Add 12 wavelet scattering features to *df_daily*.

        Parameters
        ----------
        df_daily : pd.DataFrame
            Must have a ``close`` column.

        Returns
        -------
        pd.DataFrame
            Original df_daily with 12 new ``wscat_`` columns added.
        """
        df = df_daily.copy()

        if "close" not in df.columns:
            logger.warning(
                "WaveletScatteringFeatures: 'close' column missing, skipping"
            )
            return df

        close = df["close"].values.astype(float)
        returns = np.empty_like(close)
        returns[0] = 0.0
        returns[1:] = close[1:] / close[:-1] - 1.0

        # Track which indices were originally NaN so we can restore them.
        nan_mask = np.isnan(returns)
        # Replace NaN with 0 for convolution, then restore later.
        returns_clean = np.where(nan_mask, 0.0, returns)

        n = len(returns_clean)

        # ------------------------------------------------------------------
        # Layer 0: low-pass averaging of |returns|
        # ------------------------------------------------------------------
        abs_returns = np.abs(returns_clean)
        s0 = self._rolling_mean(abs_returns, self.averaging_window)

        # ------------------------------------------------------------------
        # Layer 1: |W_j1 * x| averaged
        #   For each scale j1, convolve returns with Ricker wavelet,
        #   take modulus, then rolling mean.
        # ------------------------------------------------------------------
        layer1: Dict[int, np.ndarray] = {}          # scale -> averaged modulus
        layer1_raw: Dict[int, np.ndarray] = {}      # scale -> raw modulus (for layer 2)

        for scale in self.scales:
            wavelet = _ricker_wavelet(points=max(scale * 10, 10), a=float(scale))
            conv = fftconvolve(returns_clean, wavelet, mode="same")
            modulus = np.abs(conv)
            layer1_raw[scale] = modulus
            layer1[scale] = self._rolling_mean(modulus, self.averaging_window)

        # ------------------------------------------------------------------
        # Layer 2: |W_j2 * |W_j1 * x|| averaged, for j2 > j1
        # ------------------------------------------------------------------
        layer2: Dict[str, np.ndarray] = {}

        for i, j1 in enumerate(self.scales):
            for j2 in self.scales[i + 1:]:
                wavelet = _ricker_wavelet(points=max(j2 * 10, 10), a=float(j2))
                conv2 = fftconvolve(layer1_raw[j1], wavelet, mode="same")
                modulus2 = np.abs(conv2)
                key = f"{j1}x{j2}"
                layer2[key] = self._rolling_mean(
                    modulus2, self.averaging_window
                )

        # ------------------------------------------------------------------
        # Assemble features
        # ------------------------------------------------------------------
        s1_scale2 = layer1.get(2, np.zeros(n))
        s1_scale4 = layer1.get(4, np.zeros(n))
        s1_scale8 = layer1.get(8, np.zeros(n))

        s2_2x4 = layer2.get("2x4", np.zeros(n))
        s2_2x8 = layer2.get("2x8", np.zeros(n))
        s2_4x8 = layer2.get("4x8", np.zeros(n))

        eps = 1e-10

        # wscat_energy_ratio: high-freq / low-freq energy
        energy_ratio = s1_scale2 / (s1_scale8 + eps)

        # wscat_intermittency: measures burst structure across scales
        intermittency = s2_2x8 / (s1_scale2 * s1_scale8 + eps)

        # wscat_s0_z: 60-day z-score of s0
        s0_series = pd.Series(s0)
        s0_mean = s0_series.rolling(_ZSCORE_WINDOW, min_periods=20).mean()
        s0_std = s0_series.rolling(_ZSCORE_WINDOW, min_periods=20).std()
        s0_z = ((s0_series - s0_mean) / (s0_std + eps)).clip(-4, 4).values

        # wscat_energy_trend: 10-day slope of s1_scale4
        s4_series = pd.Series(s1_scale4)
        energy_trend = (
            (s4_series - s4_series.shift(_ENERGY_TREND_WINDOW))
            / _ENERGY_TREND_WINDOW
        ).values

        # wscat_regime: 1 if energy_ratio > 2x its 60-day rolling median, else 0
        er_series = pd.Series(energy_ratio)
        er_median = er_series.rolling(
            _ZSCORE_WINDOW, min_periods=20
        ).median()
        regime = np.where(er_series.values > 2.0 * er_median.values, 1.0, 0.0)

        # ------------------------------------------------------------------
        # Assign to DataFrame
        # ------------------------------------------------------------------
        feature_map = {
            "wscat_s0_20d": s0,
            "wscat_s1_scale2_20d": s1_scale2,
            "wscat_s1_scale4_20d": s1_scale4,
            "wscat_s1_scale8_20d": s1_scale8,
            "wscat_s2_2x4_20d": s2_2x4,
            "wscat_s2_2x8_20d": s2_2x8,
            "wscat_s2_4x8_20d": s2_4x8,
            "wscat_energy_ratio": energy_ratio,
            "wscat_intermittency": intermittency,
            "wscat_s0_z": s0_z,
            "wscat_energy_trend": energy_trend,
            "wscat_regime": regime,
        }

        for col, values in feature_map.items():
            arr = np.asarray(values, dtype=float)
            arr = np.where(np.isinf(arr), 0.0, arr)
            arr = np.where(np.isnan(arr), 0.0, arr)
            df[col] = arr

        n_features = sum(1 for c in df.columns if c.startswith("wscat_"))
        logger.info(
            "WaveletScatteringFeatures: added %d features", n_features
        )
        return df

    def analyze_current_scattering(
        self, df_daily: pd.DataFrame
    ) -> Optional[Dict]:
        """Analyze current scattering regime for dashboard display.

        Parameters
        ----------
        df_daily : pd.DataFrame
            DataFrame with ``wscat_`` columns (output of
            ``create_wavelet_scattering_features``).

        Returns
        -------
        dict or None
            Summary with regime, energy ratio, intermittency, etc.
        """
        if "wscat_regime" not in df_daily.columns or len(df_daily) < 2:
            return None

        last = df_daily.iloc[-1]

        regime_val = last.get("wscat_regime", 0.0)
        regime_str = "HIGH_FREQUENCY" if regime_val == 1.0 else "NORMAL"

        return {
            "scattering_regime": regime_str,
            "energy_ratio": round(
                float(last.get("wscat_energy_ratio", 0.0)), 4
            ),
            "intermittency": round(
                float(last.get("wscat_intermittency", 0.0)), 4
            ),
            "s0_z": round(float(last.get("wscat_s0_z", 0.0)), 3),
            "energy_trend": round(
                float(last.get("wscat_energy_trend", 0.0)), 6
            ),
            "s1_high_freq": round(
                float(last.get("wscat_s1_scale2_20d", 0.0)), 6
            ),
            "s1_low_freq": round(
                float(last.get("wscat_s1_scale8_20d", 0.0)), 6
            ),
        }

    @staticmethod
    def _all_feature_names() -> List[str]:
        return [
            "wscat_s0_20d",
            "wscat_s1_scale2_20d",
            "wscat_s1_scale4_20d",
            "wscat_s1_scale8_20d",
            "wscat_s2_2x4_20d",
            "wscat_s2_2x8_20d",
            "wscat_s2_4x8_20d",
            "wscat_energy_ratio",
            "wscat_intermittency",
            "wscat_s0_z",
            "wscat_energy_trend",
            "wscat_regime",
        ]

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling mean using cumsum for speed.

        Returns an array of the same length as *arr*.  Positions with
        fewer than ``min(5, window)`` valid points are set to NaN.
        """
        n = len(arr)
        result = np.full(n, np.nan)
        min_periods = min(5, window)

        if n < min_periods:
            return result

        cumsum = np.cumsum(arr)
        # For positions >= window, use full window
        for i in range(n):
            start = max(0, i - window + 1)
            count = i - start + 1
            if count >= min_periods:
                if start == 0:
                    result[i] = cumsum[i] / count
                else:
                    result[i] = (cumsum[i] - cumsum[start - 1]) / count

        return result
