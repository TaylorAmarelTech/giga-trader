"""
Transfer Entropy Features — directional information flow into SPY.

Transfer Entropy (TE) measures asymmetric, directional information transfer
between two time series.  A positive TE(X -> Y) means knowing X's past
reduces uncertainty about Y's future, beyond Y's own history.

Here we use a binned Mutual Information (MI) approximation:

    TE(X -> Y) ≈ MI(X_{t-1}; Y_t | Y_{t-1})
               = H(Y_t | Y_{t-1}) - H(Y_t | Y_{t-1}, X_{t-1})

Computed via 3-way contingency table over discretised bins.

Features (6, prefix te_):
  te_vix_to_spy   — Rolling 50-day TE from VIX returns to SPY returns
  te_tlt_to_spy   — Rolling 50-day TE from TLT returns to SPY returns
  te_qqq_to_spy   — Rolling 50-day TE from QQQ returns to SPY returns
  te_gld_to_spy   — Rolling 50-day TE from GLD returns to SPY returns
  te_max_inflow   — max(te_vix, te_tlt, te_qqq, te_gld): dominant driver
  te_net_flow     — mean across available non-zero assets: total info inflow

Cross-asset return columns (VXX_return, TLT_return, QQQ_return, GLD_return)
are optional — the corresponding feature is filled with 0.0 when absent.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _digitize_series(x: np.ndarray, n_bins: int = 3) -> np.ndarray:
    """
    Discretize x into n_bins using global percentile boundaries.

    Bins:
        0 — below (mean - 1*std)
        1 — between (mean - std) and (mean + std)
        2 — above (mean + std)

    Handles constant series by returning all-1 (middle bin).
    """
    std = np.nanstd(x)
    if std < 1e-15:
        return np.ones(len(x), dtype=np.int32)

    mean = np.nanmean(x)
    lo = mean - std
    hi = mean + std

    bins = np.ones(len(x), dtype=np.int32)
    bins[x < lo] = 0
    bins[x > hi] = 2
    return bins


def _rolling_te(
    source: np.ndarray,
    target: np.ndarray,
    window: int = 50,
    n_bins: int = 3,
) -> np.ndarray:
    """
    Compute rolling Transfer Entropy TE(source -> target) using binned MI.

    TE(X -> Y) = MI(X_{t-1}; Y_t | Y_{t-1})
               = sum_{y_t, y_{t-1}, x_{t-1}} p(y_t, y_{t-1}, x_{t-1})
                 * log( p(y_t | y_{t-1}, x_{t-1}) / p(y_t | y_{t-1}) )

    Estimated from counts in a rolling window of length *window*.

    Parameters
    ----------
    source : np.ndarray
        Source series (X). Length N.
    target : np.ndarray
        Target series (Y). Length N.
    window : int
        Rolling window size (number of observations).
    n_bins : int
        Number of discretisation bins (3 for low/mid/high).

    Returns
    -------
    np.ndarray
        TE values, shape (N,). Filled with 0.0 for the warm-up period.
    """
    n = len(source)
    result = np.zeros(n, dtype=float)

    # Pre-discretise globally to save time inside the loop
    src_bins = _digitize_series(source, n_bins)
    tgt_bins = _digitize_series(target, n_bins)

    for i in range(window, n):
        # Slices: indices [i-window, i)
        # y_t: target[i-window+1 .. i]    (shifted +1 relative to x/y_lag)
        # y_lag: target[i-window   .. i-1] (Y_{t-1})
        # x_lag: source[i-window   .. i-1] (X_{t-1})
        end = i
        start = i - window

        y_t = tgt_bins[start + 1 : end + 1]   # length = window
        y_lag = tgt_bins[start : end]          # length = window
        x_lag = src_bins[start : end]          # length = window

        # Remove positions where any is NaN (NaN detection on int array: skip —
        # NaN only happens in float; after _digitize_series we have ints already)
        n_valid = len(y_t)
        if n_valid < 10:
            continue

        # Build 3D contingency table: shape (n_bins, n_bins, n_bins)
        # axes: (x_lag, y_lag, y_t)
        counts_xyz = np.zeros((n_bins, n_bins, n_bins), dtype=float)
        for j in range(n_valid):
            counts_xyz[x_lag[j], y_lag[j], y_t[j]] += 1

        total = counts_xyz.sum()
        if total < 1:
            continue

        # Marginalise to get joint and conditional distributions
        # p(y_lag, y_t)  — shape (n_bins, n_bins)
        p_yl_yt = counts_xyz.sum(axis=0) / total          # sum over x_lag

        # p(x_lag, y_lag, y_t) — normalised counts
        p_xyz = counts_xyz / total

        # TE = sum over (x, y_lag, y_t): p(x,y_lag,y_t) * log(p(y_t|y_lag,x) / p(y_t|y_lag))
        te_val = 0.0
        for xi in range(n_bins):
            for yi in range(n_bins):
                p_yl_val = p_yl_yt[yi, :].sum()  # p(y_lag = yi)
                for yt_i in range(n_bins):
                    p_joint = p_xyz[xi, yi, yt_i]
                    if p_joint < 1e-12:
                        continue

                    # p(y_t | y_lag) = p(y_lag, y_t) / p(y_lag)
                    p_yt_given_yl = (
                        p_yl_yt[yi, yt_i] / p_yl_val
                        if p_yl_val > 1e-12
                        else 0.0
                    )

                    # p(y_t | y_lag, x_lag) = p(x,y_lag,y_t) / p(x,y_lag)
                    p_x_yl = p_xyz[xi, yi, :].sum()
                    p_yt_given_xl_yl = (
                        p_joint / p_x_yl if p_x_yl > 1e-12 else 0.0
                    )

                    if p_yt_given_yl > 1e-12 and p_yt_given_xl_yl > 1e-12:
                        te_val += p_joint * np.log(
                            p_yt_given_xl_yl / p_yt_given_yl
                        )

        result[i] = max(0.0, te_val)  # TE is non-negative; clamp numeric noise

    return result


class TransferEntropyFeatures:
    """
    Compute directional information flow (Transfer Entropy) from cross-asset
    returns into SPY returns.

    Usage
    -----
    te = TransferEntropyFeatures(window=50, n_bins=3)
    df_out = te.create_transfer_entropy_features(df_daily)
    info = te.analyze_current_te(df_out)
    """

    FEATURE_NAMES: List[str] = [
        "te_vix_to_spy",
        "te_tlt_to_spy",
        "te_qqq_to_spy",
        "te_gld_to_spy",
        "te_max_inflow",
        "te_net_flow",
    ]

    # Map feature -> source column in input DataFrame
    _SOURCE_COLS: Dict[str, str] = {
        "te_vix_to_spy": "VXX_return",
        "te_tlt_to_spy": "TLT_return",
        "te_qqq_to_spy": "QQQ_return",
        "te_gld_to_spy": "GLD_return",
    }

    def __init__(self, window: int = 50, n_bins: int = 3) -> None:
        """
        Parameters
        ----------
        window : int
            Rolling window length for TE estimation (default 50 trading days).
        n_bins : int
            Number of discretisation bins (default 3: low / mid / high).
        """
        self.window = window
        self.n_bins = n_bins

    # ─── Public API ───────────────────────────────────────────────────────────

    def download_te_data(
        self,
        start_date,
        end_date,
    ) -> pd.DataFrame:
        """
        Placeholder download method (follows project architecture pattern).

        Transfer Entropy features are computed directly from cross-asset
        return columns already present in df_daily (populated by
        CrossAssetFeatures / EconomicFeatures upstream).  No separate
        download is needed.

        Returns
        -------
        pd.DataFrame
            Empty DataFrame (no-op).
        """
        return pd.DataFrame()

    def create_transfer_entropy_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add 6 transfer entropy features to df.

        Required columns
        ----------------
        close : float
            SPY close prices (used to derive SPY returns when day_return absent).

        Optional columns (filled with 0.0 if missing)
        ---------------
        VXX_return, TLT_return, QQQ_return, GLD_return : float
            Pre-computed daily returns from cross-asset features pipeline.

        Parameters
        ----------
        df : pd.DataFrame
            Input daily DataFrame.

        Returns
        -------
        pd.DataFrame
            Copy of df with 6 new te_ columns appended.
        """
        out = df.copy()

        if "close" not in out.columns:
            logger.warning(
                "TransferEntropyFeatures: 'close' column missing — skipping"
            )
            for col in self.FEATURE_NAMES:
                out[col] = 0.0
            return out

        # Derive SPY returns
        if "day_return" in out.columns:
            spy_ret = out["day_return"].values.astype(float)
        else:
            close = out["close"].values.astype(float)
            spy_ret = np.concatenate([[0.0], np.diff(close) / (close[:-1] + 1e-15)])

        spy_ret = np.where(np.isfinite(spy_ret), spy_ret, 0.0)

        # Compute individual TEs
        te_values: Dict[str, np.ndarray] = {}
        for feat_name, src_col in self._SOURCE_COLS.items():
            if src_col in out.columns:
                src_ret = out[src_col].values.astype(float)
                src_ret = np.where(np.isfinite(src_ret), src_ret, 0.0)
                te_arr = _rolling_te(
                    source=src_ret,
                    target=spy_ret,
                    window=self.window,
                    n_bins=self.n_bins,
                )
            else:
                te_arr = np.zeros(len(out), dtype=float)

            te_values[feat_name] = te_arr
            out[feat_name] = te_arr

        # te_max_inflow: element-wise max across 4 assets
        te_stack = np.stack(list(te_values.values()), axis=1)  # (N, 4)
        out["te_max_inflow"] = te_stack.max(axis=1)

        # te_net_flow: mean across assets that have non-zero data
        # (only count assets for which the source column was present)
        available_cols = [
            col for col in self._SOURCE_COLS.keys() if self._SOURCE_COLS[col] in df.columns
        ]
        if available_cols:
            net_stack = np.stack([te_values[c] for c in available_cols], axis=1)
            out["te_net_flow"] = net_stack.mean(axis=1)
        else:
            out["te_net_flow"] = np.zeros(len(out), dtype=float)

        # Final cleanup: fill any residual NaN/inf with 0.0
        for col in self.FEATURE_NAMES:
            out[col] = (
                out[col]
                .replace([np.inf, -np.inf], 0.0)
                .fillna(0.0)
            )

        n_feat = sum(1 for c in out.columns if c.startswith("te_"))
        logger.info(f"TransferEntropyFeatures: added {n_feat} features")
        return out

    def analyze_current_te(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Analyse the current (most recent row) transfer entropy regime.

        Parameters
        ----------
        df : pd.DataFrame
            Output of create_transfer_entropy_features().

        Returns
        -------
        dict or None
            Keys:
              dominant_source   — str: 'VIX' | 'TLT' | 'QQQ' | 'GLD' | 'NONE'
              information_flow  — str: 'HIGH' | 'LOW' | 'NORMAL'
              te_values         — dict of {asset: float} for last row
            None if TE features are not present.
        """
        te_feat_cols = [c for c in self.FEATURE_NAMES if c in df.columns]
        if not te_feat_cols or len(df) < 2:
            return None

        last = df.iloc[-1]

        asset_map = {
            "te_vix_to_spy": "VIX",
            "te_tlt_to_spy": "TLT",
            "te_qqq_to_spy": "QQQ",
            "te_gld_to_spy": "GLD",
        }

        te_vals = {
            asset_map[col]: float(last.get(col, 0.0))
            for col in asset_map
            if col in df.columns
        }

        # Dominant source: asset with highest TE
        if te_vals:
            best_asset = max(te_vals, key=lambda k: te_vals[k])
            dominant_source = best_asset if te_vals[best_asset] > 1e-6 else "NONE"
        else:
            dominant_source = "NONE"

        net_flow = float(last.get("te_net_flow", 0.0))

        # Compare current net_flow against rolling history
        if "te_net_flow" in df.columns and len(df) >= self.window:
            recent = df["te_net_flow"].tail(self.window)
            mean_flow = recent.mean()
            std_flow = recent.std()
            if std_flow > 1e-10:
                z = (net_flow - mean_flow) / std_flow
                if z > 1.0:
                    information_flow = "HIGH"
                elif z < -1.0:
                    information_flow = "LOW"
                else:
                    information_flow = "NORMAL"
            else:
                information_flow = "NORMAL"
        else:
            information_flow = "NORMAL"

        return {
            "dominant_source": dominant_source,
            "information_flow": information_flow,
            "te_values": {k: round(v, 6) for k, v in te_vals.items()},
            "te_net_flow": round(net_flow, 6),
            "te_max_inflow": round(float(last.get("te_max_inflow", 0.0)), 6),
        }
