"""
GIGA TRADER - Time Series Model Features
==========================================
Uses classical statistical models (ARIMA) and optionally zero-shot foundation
models (Chronos-Bolt) as *feature generators*, NOT as direct predictors.

Key insight: the RESIDUAL (what the model can't explain) is the signal.
Large positive residual = unexpectedly strong market.  Large negative = stress.
Prediction interval width = model uncertainty = compression/inflection indicator.

10-15 features generated (prefix: tsm_), 4 sections.

Section 1: Classical ARIMA Residuals (3)    — always available via statsmodels
Section 2: Chronos Foundation Model (5)     — optional, requires chronos-forecasting
Section 3: Cross-Model Disagreement (2)     — ARIMA vs Chronos consensus
Section 4: catch22 Canonical Features (5)   — optional, requires pycatch22
"""

import logging
import warnings
from typing import Optional, Dict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

logger = logging.getLogger("TS_MODEL")


class TimeSeriesModelFeatures:
    """
    Generate features from time series model residuals and uncertainty.

    All features use the tsm_ prefix.  Chronos and catch22 features
    gracefully degrade to 0.0 when their optional dependencies are absent.
    """

    REQUIRED_COLS = {"close"}

    def __init__(self, use_chronos: bool = True, use_catch22: bool = True):
        self._use_chronos = use_chronos
        self._use_catch22 = use_catch22

        # Lazy availability checks
        self._chronos_available = False
        self._catch22_available = False
        self._chronos_pipeline = None

        if use_chronos:
            try:
                # noinspection PyUnresolvedReferences
                import torch  # noqa: F401
                from chronos import ChronosPipeline  # noqa: F401
                self._chronos_available = True
            except ImportError:
                self._chronos_available = False

        if use_catch22:
            try:
                import pycatch22  # noqa: F401
                self._catch22_available = True
            except ImportError:
                self._catch22_available = False

    # ------------------------------------------------------------------ #
    #  Public API                                                         #
    # ------------------------------------------------------------------ #

    def create_time_series_model_features(
        self,
        spy_daily: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Create time series model features and merge into spy_daily.

        Returns spy_daily with new tsm_* columns added.
        """
        df = spy_daily.copy()

        print("\n[TSM] Engineering time series model features...")

        if "close" not in df.columns:
            print("  [WARN] No 'close' column — skipping time series models")
            return df

        close = df["close"].values.astype(np.float64)
        n = len(close)

        if n < 30:
            print("  [WARN] Insufficient data (<30 rows) — skipping")
            return df

        daily_return = np.empty(n)
        daily_return[0] = 0.0
        daily_return[1:] = (close[1:] - close[:-1]) / np.maximum(close[:-1], 1e-10)

        # ATR for normalization
        if "high" in df.columns and "low" in df.columns:
            high = df["high"].values.astype(np.float64)
            low = df["low"].values.astype(np.float64)
            raw_range = high - low
            atr_20 = pd.Series(raw_range).rolling(20, min_periods=10).mean().to_numpy(copy=True)
            atr_20[atr_20 < 1e-10] = 1e-10
        else:
            atr_20 = pd.Series(np.abs(daily_return)).rolling(20, min_periods=10).mean().to_numpy(copy=True) * close
            atr_20[atr_20 < 1e-10] = 1e-10

        # Section 1: ARIMA residuals
        self._add_arima_features(df, close, daily_return, atr_20, n)

        # Section 2: Chronos features
        self._add_chronos_features(df, close, daily_return, n)

        # Section 3: Cross-model disagreement
        self._add_cross_model_features(df, n)

        # Section 4: catch22 features
        self._add_catch22_features(df, daily_return, n)

        # Cleanup
        tsm_cols = [c for c in df.columns if c.startswith("tsm_")]
        for col in tsm_cols:
            df[col] = df[col].replace([np.inf, -np.inf], 0.0).fillna(0.0)

        n_added = len(tsm_cols)
        print(f"  [TSM] Total: {n_added} time series model features added")

        return df

    def analyze_current_ts(
        self,
        spy_daily: pd.DataFrame,
    ) -> Optional[Dict]:
        """Return snapshot of current time series model state."""
        tsm_cols = [c for c in spy_daily.columns if c.startswith("tsm_")]
        if not tsm_cols or len(spy_daily) < 2:
            return None

        last = spy_daily.iloc[-1]
        return {
            "arima_available": True,
            "chronos_available": self._chronos_available,
            "catch22_available": self._catch22_available,
            "arima_residual": round(float(last.get("tsm_arima_residual", 0.0)), 5),
            "arima_residual_vol": round(float(last.get("tsm_arima_residual_vol", 0.0)), 5),
            "model_disagreement": round(float(last.get("tsm_model_disagreement", 0.0)), 4),
        }

    # ------------------------------------------------------------------ #
    #  Section 1: ARIMA Residuals                                         #
    # ------------------------------------------------------------------ #

    def _add_arima_features(self, df: pd.DataFrame, close: np.ndarray,
                            returns: np.ndarray, atr: np.ndarray, n: int):
        """ARIMA(1,0,1) 1-step-ahead residuals on expanding window."""
        print("  Section 1: ARIMA residuals (3 features)...")

        residuals = np.zeros(n)
        min_window = 120

        # Try statsmodels ARIMA; fallback to AR(1)
        use_statsmodels = True
        try:
            from statsmodels.tsa.arima.model import ARIMA  # noqa: F401
        except ImportError:
            use_statsmodels = False

        # Compute rolling 1-step-ahead forecast residuals
        # For efficiency, re-fit only every 20 rows
        last_pred = 0.0
        refit_interval = 20

        for i in range(min_window, n):
            # Record residual from previous prediction
            residuals[i] = returns[i] - last_pred

            # Re-fit model periodically
            if (i - min_window) % refit_interval == 0 or i == min_window:
                seg = returns[max(0, i - 252):i]  # Use up to 252 days
                last_pred = self._fit_and_predict(seg, use_statsmodels)

        # 1. ARIMA residual
        df["tsm_arima_residual"] = residuals

        # 2. Residual volatility (rolling 20d std)
        resid_vol = pd.Series(residuals).rolling(20, min_periods=10).std().values
        df["tsm_arima_residual_vol"] = resid_vol

        # 3. Trend component (MA20 - MA60, normalized)
        ma20 = pd.Series(close).rolling(20, min_periods=15).mean().values
        ma60 = pd.Series(close).rolling(60, min_periods=30).mean().values
        with np.errstate(divide="ignore", invalid="ignore"):
            trend = np.where(atr > 1e-10, (ma20 - ma60) / atr, 0.0)
        df["tsm_arima_trend"] = np.clip(trend, -5.0, 5.0)

    @staticmethod
    def _fit_and_predict(returns_seg: np.ndarray, use_statsmodels: bool) -> float:
        """Fit ARIMA(1,0,1) or AR(1) and return 1-step forecast."""
        if len(returns_seg) < 30:
            return 0.0

        if use_statsmodels:
            try:
                from statsmodels.tsa.arima.model import ARIMA
                model = ARIMA(returns_seg, order=(1, 0, 1))
                fit = model.fit(method_kwargs={"maxiter": 50, "disp": False})
                forecast = fit.forecast(steps=1)
                return float(forecast[0])
            except Exception:
                pass

        # Fallback: AR(1) via numpy
        try:
            if len(returns_seg) < 3:
                return 0.0
            x = returns_seg[:-1]
            y = returns_seg[1:]
            if np.std(x) < 1e-10:
                return float(np.mean(y))
            slope = np.sum((x - np.mean(x)) * (y - np.mean(y))) / np.sum((x - np.mean(x)) ** 2)
            intercept = np.mean(y) - slope * np.mean(x)
            return float(slope * returns_seg[-1] + intercept)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------ #
    #  Section 2: Chronos Foundation Model                                #
    # ------------------------------------------------------------------ #

    def _add_chronos_features(self, df: pd.DataFrame, close: np.ndarray,
                              returns: np.ndarray, n: int):
        """Chronos-Bolt zero-shot features (optional)."""
        print(f"  Section 2: Chronos foundation model (5 features, "
              f"available={self._chronos_available})...")

        # Initialize columns
        df["tsm_chronos_residual_1d"] = 0.0
        df["tsm_chronos_residual_5d"] = 0.0
        df["tsm_chronos_interval_width"] = 0.0
        df["tsm_chronos_interval_pctile"] = 0.5
        df["tsm_chronos_surprise"] = 0.0

        if not self._chronos_available or not self._use_chronos:
            return

        try:
            import torch
            from chronos import ChronosPipeline

            # Load model once (smallest model, CPU)
            if self._chronos_pipeline is None:
                self._chronos_pipeline = ChronosPipeline.from_pretrained(
                    "amazon/chronos-bolt-tiny",
                    device_map="cpu",
                    torch_dtype=torch.float32,
                )

            pipeline = self._chronos_pipeline
            context_len = 63  # ~3 months of trading days

            # Generate predictions for a subset of rows (every 5th for efficiency)
            residual_1d = np.zeros(n)
            residual_5d = np.zeros(n)
            interval_width = np.zeros(n)
            interval_pctile = np.full(n, 0.5)
            surprise = np.zeros(n)

            step = 5  # Re-predict every 5 rows during training
            for i in range(context_len, n, step):
                try:
                    context = torch.tensor(close[max(0, i - context_len):i]).unsqueeze(0).float()
                    forecast = pipeline.predict(
                        context,
                        prediction_length=5,
                        num_samples=20,
                    )
                    # forecast shape: (1, num_samples, prediction_length)
                    samples = forecast[0].numpy()  # (num_samples, 5)

                    median_pred = np.median(samples, axis=0)
                    q05 = np.percentile(samples, 5, axis=0)
                    q95 = np.percentile(samples, 95, axis=0)

                    # 1-day residual (predicted price vs actual)
                    if i < n:
                        pred_return = (median_pred[0] - close[i - 1]) / max(close[i - 1], 1e-10)
                        residual_1d[i] = returns[i] - pred_return

                    # 5-day residual
                    if i + 4 < n:
                        actual_5d = (close[i + 4] - close[i - 1]) / max(close[i - 1], 1e-10)
                        pred_5d = (median_pred[4] - close[i - 1]) / max(close[i - 1], 1e-10)
                        residual_5d[i] = actual_5d - pred_5d

                    # Interval width
                    width = (q95[0] - q05[0]) / max(close[i - 1], 1e-10)
                    interval_width[i] = width

                    # Percentile position
                    if i < n and width > 1e-10:
                        actual_price = close[i]
                        interval_pctile[i] = np.clip(
                            (actual_price - q05[0]) / (q95[0] - q05[0]), 0.0, 1.0
                        )

                    # Surprise
                    if width > 1e-10:
                        surprise[i] = abs(residual_1d[i]) / width

                    # Forward-fill for skipped rows
                    fill_end = min(i + step, n)
                    for j in range(i + 1, fill_end):
                        residual_1d[j] = residual_1d[i]
                        interval_width[j] = interval_width[i]
                        interval_pctile[j] = interval_pctile[i]
                        surprise[j] = surprise[i]

                except Exception:
                    continue

            df["tsm_chronos_residual_1d"] = residual_1d
            df["tsm_chronos_residual_5d"] = residual_5d
            df["tsm_chronos_interval_width"] = interval_width
            df["tsm_chronos_interval_pctile"] = interval_pctile
            df["tsm_chronos_surprise"] = np.clip(surprise, 0.0, 10.0)

        except Exception as e:
            print(f"    [WARN] Chronos feature generation failed: {e}")

    # ------------------------------------------------------------------ #
    #  Section 3: Cross-Model Disagreement                                #
    # ------------------------------------------------------------------ #

    def _add_cross_model_features(self, df: pd.DataFrame, n: int):
        """Disagreement between ARIMA and Chronos predictions."""
        print("  Section 3: Cross-model disagreement (2 features)...")

        arima_dir = np.sign(df["tsm_arima_residual"].values)

        if self._chronos_available and "tsm_chronos_residual_1d" in df.columns:
            chronos_dir = np.sign(df["tsm_chronos_residual_1d"].values)
            # Disagreement: std of direction signals (0 = agree, ~1 = disagree)
            dirs = np.column_stack([arima_dir, chronos_dir])
            disagreement = np.std(dirs, axis=1)
            agreement = (arima_dir == chronos_dir).astype(float)
        else:
            disagreement = np.zeros(n)
            agreement = np.full(n, 0.5)  # Neutral when only one model

        df["tsm_model_disagreement"] = disagreement
        df["tsm_directional_agreement"] = agreement

    # ------------------------------------------------------------------ #
    #  Section 4: catch22 Canonical Features                              #
    # ------------------------------------------------------------------ #

    def _add_catch22_features(self, df: pd.DataFrame, returns: np.ndarray, n: int):
        """Selected catch22 canonical time series features."""
        print(f"  Section 4: catch22 canonical features (5 features, "
              f"available={self._catch22_available})...")

        # Initialize columns
        feature_names = [
            "tsm_c22_first_min_acf",
            "tsm_c22_trev_1_num",
            "tsm_c22_sp_trev",
            "tsm_c22_mean_cross",
            "tsm_c22_outlier_count",
        ]
        for name in feature_names:
            df[name] = 0.0

        if not self._catch22_available or not self._use_catch22:
            return

        try:
            import pycatch22

            window = 60
            step = 5  # Compute every 5th row for efficiency

            # Indices of the 5 features we want from catch22's 22
            # 0: DN_HistogramMode_5
            # 3: CO_FirstMin_ac (first minimum of ACF)
            # 5: CO_trev_1_num (time reversibility)
            # 7: SP_Summaries_welch_rect (spectral)
            # 17: SB_BinaryStats_mean_longstretch1 (mean crossing)
            target_indices = [3, 5, 7, 17, 21]  # Mapped to our 5 features

            results = {name: np.zeros(n) for name in feature_names}

            for i in range(window, n, step):
                seg = returns[i - window:i].tolist()
                try:
                    all_feats = pycatch22.catch22_all(seg)
                    values = all_feats["values"]

                    for feat_idx, (catch_idx, name) in enumerate(
                        zip(target_indices, feature_names)
                    ):
                        if catch_idx < len(values):
                            val = float(values[catch_idx])
                            results[name][i] = val
                            # Forward-fill
                            fill_end = min(i + step, n)
                            for j in range(i + 1, fill_end):
                                results[name][j] = val
                except Exception:
                    continue

            for name in feature_names:
                df[name] = results[name]

        except Exception as e:
            print(f"    [WARN] catch22 feature generation failed: {e}")
