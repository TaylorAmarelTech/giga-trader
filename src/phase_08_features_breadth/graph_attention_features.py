"""
GIGA TRADER - Graph Attention Network Features (Pure NumPy)
============================================================
Cross-asset Graph Attention Network features that model inter-asset
relationships through an attention-based message-passing framework.

Instead of treating each asset independently, this module:
  1. Builds a correlation graph where nodes = assets (SPY, TLT, GLD, QQQ,
     IWM, VXX, HYG).
  2. Computes multi-head attention weights between assets based on rolling
     correlation strength and learned (random-projection) queries/keys.
  3. Aggregates neighbor information using attention-weighted message passing.
  4. Extracts SPY-centric features from the attention-weighted aggregation.

Features (8, prefix gat_):
  gat_spy_attention_entropy  - How dispersed SPY's attention is across assets.
                               High entropy = broadly connected;
                               low entropy = dominated by one asset.
  gat_spy_weighted_return    - Attention-weighted average return of neighbors.
  gat_spy_safe_haven_flow    - Sum of attention toward TLT + GLD (flight to safety).
  gat_spy_risk_on_flow       - Sum of attention toward QQQ + IWM (risk appetite).
  gat_spy_fear_attention     - Attention toward VXX (fear / volatility).
  gat_spy_credit_attention   - Attention toward HYG (credit risk sentiment).
  gat_graph_density          - Fraction of pairwise correlations above threshold.
  gat_regime_cluster         - K-means cluster (k=3) on the attention pattern
                               (0, 1, or 2 mapped to a float).

IMPORTANT: This is a PURE NUMPY implementation.  No PyTorch, TensorFlow,
or any deep learning framework is used.
"""

import logging
from typing import ClassVar, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from src.core.feature_base import FeatureModuleBase

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

_ASSET_NAMES: List[str] = ["SPY", "TLT", "GLD", "QQQ", "IWM", "VXX", "HYG"]

_CROSS_ASSET_RETURN_COLS: Dict[str, str] = {
    "TLT": "TLT_return",
    "GLD": "GLD_return",
    "QQQ": "QQQ_return",
    "IWM": "IWM_return",
    "VXX": "VXX_return",
    "HYG": "HYG_return",
}

# Safe-haven and risk-on groupings (by index in _ASSET_NAMES)
_SAFE_HAVEN_ASSETS = {"TLT", "GLD"}
_RISK_ON_ASSETS = {"QQQ", "IWM"}
_FEAR_ASSETS = {"VXX"}
_CREDIT_ASSETS = {"HYG"}

_DENSITY_THRESHOLD = 0.3  # correlation threshold for graph density
_KMEANS_K = 3
_KMEANS_MAX_ITER = 30
_RANDOM_SEED = 42


# ── Helpers ───────────────────────────────────────────────────────────────────

def _softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax, numerically stable."""
    # x shape: (n,) or (n_heads, n)
    if x.ndim == 1:
        e = np.exp(x - np.max(x))
        s = e.sum()
        return e / s if s > 0 else np.ones_like(e) / len(e)
    # 2-D: softmax along axis=1
    shifted = x - x.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    s = e.sum(axis=1, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    return e / s


def _kmeans_1d(data: np.ndarray, k: int = 3, max_iter: int = 30,
               seed: int = 42) -> np.ndarray:
    """
    Simple k-means on rows of `data` (each row is a point in R^d).
    Returns cluster labels (int array of shape (n,)).
    """
    n = data.shape[0]
    if n == 0:
        return np.array([], dtype=int)
    if n <= k:
        return np.arange(n, dtype=int) % k

    rng = np.random.RandomState(seed)
    # Initialize centers using random selection
    idx = rng.choice(n, size=k, replace=False)
    centers = data[idx].copy()

    labels = np.zeros(n, dtype=int)
    for _ in range(max_iter):
        # Assign to nearest center
        dists = np.array([
            np.sum((data - centers[j]) ** 2, axis=1)
            for j in range(k)
        ])  # shape (k, n)
        new_labels = dists.argmin(axis=0)

        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # Update centers
        for j in range(k):
            mask = labels == j
            if mask.sum() > 0:
                centers[j] = data[mask].mean(axis=0)

    return labels


# ── Main class ────────────────────────────────────────────────────────────────

class GraphAttentionFeatures(FeatureModuleBase):
    """
    Compute cross-asset Graph Attention Network features (pure NumPy).

    Parameters
    ----------
    window : int
        Rolling window (trading days) for computing correlations and
        attention weights.  Default 20.
    n_heads : int
        Number of attention heads.  Each head uses a different random
        projection.  Final attention is the average across heads.  Default 4.
    density_threshold : float
        Absolute correlation above which an edge is counted for graph
        density.  Default 0.3.
    """

    REQUIRED_COLS: ClassVar[Set[str]] = {"close"}
    FEATURE_NAMES: ClassVar[List[str]] = [
        "gat_spy_attention_entropy",
        "gat_spy_weighted_return",
        "gat_spy_safe_haven_flow",
        "gat_spy_risk_on_flow",
        "gat_spy_fear_attention",
        "gat_spy_credit_attention",
        "gat_graph_density",
        "gat_regime_cluster",
    ]

    def __init__(
        self,
        window: int = 20,
        n_heads: int = 4,
        density_threshold: float = _DENSITY_THRESHOLD,
    ) -> None:
        self.window = max(2, window)
        self.n_heads = max(1, n_heads)
        self.density_threshold = density_threshold

        # Pre-generate random projection matrices (fixed seed for reproducibility)
        # Each head: project window-length return vectors into a small dimension d_k
        self._d_k = 4  # dimension of query/key space per head
        rng = np.random.RandomState(_RANDOM_SEED)
        # W_Q and W_K: shape (n_heads, window, d_k)
        self._W_Q = rng.randn(self.n_heads, self.window, self._d_k) * 0.1
        self._W_K = rng.randn(self.n_heads, self.window, self._d_k) * 0.1

    # ── Public API ────────────────────────────────────────────────────────

    def download_cross_asset_data(
        self,
        start_date,
        end_date,
    ) -> pd.DataFrame:
        """
        Download cross-asset data (TLT, GLD, QQQ, IWM, VXX, HYG) via yfinance.

        Returns an empty DataFrame if download fails.  The main feature
        creation method works with whatever cross-asset return columns
        are already present in df_daily, so this download is optional.
        """
        logger.info("GraphAttentionFeatures: downloading cross-asset data via yfinance")
        try:
            import yfinance as yf
            tickers = ["TLT", "GLD", "QQQ", "IWM", "VXX", "HYG"]
            data = yf.download(
                tickers, start=str(start_date), end=str(end_date),
                progress=False, auto_adjust=True,
            )
            if data.empty:
                logger.warning("GraphAttentionFeatures: yfinance returned empty data")
                return pd.DataFrame()
            # Extract close prices
            if isinstance(data.columns, pd.MultiIndex):
                close = data["Close"]
            else:
                close = data[["Close"]].rename(columns={"Close": tickers[0]})
            logger.info(
                f"GraphAttentionFeatures: downloaded {len(close)} days "
                f"for {len(close.columns)} assets"
            )
            return close
        except Exception as e:
            logger.warning(f"GraphAttentionFeatures: download failed: {e}")
            return pd.DataFrame()

    def create_graph_attention_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the 8 gat_ features and append them to df.

        Parameters
        ----------
        df : pd.DataFrame
            Daily bar DataFrame.  Must contain 'close'.  Cross-asset return
            columns (TLT_return, QQQ_return, ...) are used when present;
            when absent, only SPY self-returns are available and neighbor
            features default to zero.

        Returns
        -------
        pd.DataFrame
            Original df with 8 gat_ columns appended.  All features are
            zero-filled on failure.
        """
        out = df.copy()

        if not self._validate_input(out, min_rows=2):
            return self._zero_fill_all(out)

        # ── Build the return matrix (SPY + available cross-assets) ────
        return_matrix = self._build_return_matrix(out)
        n_rows = len(return_matrix)
        n_assets = return_matrix.shape[1]
        asset_cols = list(return_matrix.columns)

        if n_assets < 2:
            logger.info(
                "GraphAttentionFeatures: fewer than 2 return series, "
                "defaulting to 0.0"
            )
            return self._zero_fill_all(out)

        # Map asset column names back to canonical names for grouping
        col_to_asset = self._map_columns_to_assets(asset_cols)

        # SPY column index in return_matrix
        spy_idx = self._find_spy_index(asset_cols)

        # ── Pre-allocate output arrays ────────────────────────────────
        entropy_arr = np.zeros(n_rows)
        weighted_ret_arr = np.zeros(n_rows)
        safe_haven_arr = np.zeros(n_rows)
        risk_on_arr = np.zeros(n_rows)
        fear_arr = np.zeros(n_rows)
        credit_arr = np.zeros(n_rows)
        density_arr = np.zeros(n_rows)

        # Collect attention vectors for regime clustering
        attention_history = np.zeros((n_rows, n_assets))

        returns_np = return_matrix.values  # (n_rows, n_assets)

        min_periods = max(2, self.window // 3)

        for i in range(n_rows):
            start = max(0, i - self.window + 1)
            window_len = i - start + 1

            if window_len < min_periods:
                continue

            window_slice = returns_np[start: i + 1]  # (window_len, n_assets)

            # ── Compute pairwise correlation ──────────────────────────
            corr_matrix = self._rolling_corr(window_slice)
            if corr_matrix is None:
                continue

            # ── Multi-head attention for SPY node ─────────────────────
            attn_weights = self._compute_multi_head_attention(
                window_slice, spy_idx, corr_matrix
            )
            # attn_weights: shape (n_assets,) — SPY's attention over all nodes

            attention_history[i] = attn_weights

            # ── Extract features from attention weights ───────────────
            # 1. Attention entropy
            eps = 1e-10
            a_clip = np.clip(attn_weights, eps, 1.0)
            entropy_arr[i] = -np.sum(a_clip * np.log(a_clip))

            # 2. Weighted return of neighbors (current-day returns)
            current_returns = returns_np[i]
            weighted_ret_arr[i] = np.dot(attn_weights, current_returns)

            # 3-6. Group-level attention
            for j, col_name in enumerate(asset_cols):
                asset_name = col_to_asset.get(col_name, "")
                if asset_name in _SAFE_HAVEN_ASSETS:
                    safe_haven_arr[i] += attn_weights[j]
                if asset_name in _RISK_ON_ASSETS:
                    risk_on_arr[i] += attn_weights[j]
                if asset_name in _FEAR_ASSETS:
                    fear_arr[i] += attn_weights[j]
                if asset_name in _CREDIT_ASSETS:
                    credit_arr[i] += attn_weights[j]

            # 7. Graph density
            density_arr[i] = self._graph_density(corr_matrix)

        # ── Regime cluster (k=3 on attention patterns) ────────────────
        cluster_arr = self._compute_regime_clusters(attention_history)

        # ── Assign to output ──────────────────────────────────────────
        out["gat_spy_attention_entropy"] = entropy_arr
        out["gat_spy_weighted_return"] = weighted_ret_arr
        out["gat_spy_safe_haven_flow"] = safe_haven_arr
        out["gat_spy_risk_on_flow"] = risk_on_arr
        out["gat_spy_fear_attention"] = fear_arr
        out["gat_spy_credit_attention"] = credit_arr
        out["gat_graph_density"] = density_arr
        out["gat_regime_cluster"] = cluster_arr.astype(float)

        out = self._cleanup_features(out)

        n_feats = sum(1 for c in out.columns if c.startswith("gat_"))
        logger.info(f"GraphAttentionFeatures: added {n_feats} features")
        return out

    def analyze_current_graph(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Analyze the most recent graph attention state.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame that already contains the gat_ feature columns.

        Returns
        -------
        dict | None
            Summary of the current attention pattern and regime.
        """
        missing = [c for c in self.FEATURE_NAMES if c not in df.columns]
        if missing or len(df) < 1:
            return None

        last = df.iloc[-1]

        entropy = float(last.get("gat_spy_attention_entropy", 0.0))
        safe = float(last.get("gat_spy_safe_haven_flow", 0.0))
        risk = float(last.get("gat_spy_risk_on_flow", 0.0))
        fear = float(last.get("gat_spy_fear_attention", 0.0))
        density = float(last.get("gat_graph_density", 0.0))
        cluster = int(last.get("gat_regime_cluster", 0))

        # Regime classification
        if fear > 0.3 and safe > 0.3:
            regime = "RISK_OFF"
        elif risk > 0.4 and fear < 0.15:
            regime = "RISK_ON"
        elif density > 0.7:
            regime = "SYSTEMIC"
        else:
            regime = "NEUTRAL"

        return {
            "graph_regime": regime,
            "gat_spy_attention_entropy": round(entropy, 4),
            "gat_spy_safe_haven_flow": round(safe, 4),
            "gat_spy_risk_on_flow": round(risk, 4),
            "gat_spy_fear_attention": round(fear, 4),
            "gat_graph_density": round(density, 4),
            "gat_regime_cluster": cluster,
        }

    # ── Private helpers ───────────────────────────────────────────────────

    def _build_return_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assemble the return matrix: SPY + available cross-asset returns.
        """
        matrix = pd.DataFrame(index=df.index)

        # SPY return from close
        matrix["SPY_return"] = df["close"].pct_change().fillna(0.0)

        # Append cross-asset return columns that are present
        for asset_name, col_name in _CROSS_ASSET_RETURN_COLS.items():
            if col_name in df.columns:
                matrix[col_name] = df[col_name].fillna(0.0).values

        # If daily_return column exists and is different from SPY_return, skip
        # (we already have SPY from close)

        return matrix

    @staticmethod
    def _map_columns_to_assets(asset_cols: List[str]) -> Dict[str, str]:
        """Map return-matrix column names to canonical asset names."""
        mapping = {}
        for col in asset_cols:
            if col == "SPY_return":
                mapping[col] = "SPY"
            else:
                # e.g., "TLT_return" -> "TLT"
                for asset_name, col_name in _CROSS_ASSET_RETURN_COLS.items():
                    if col == col_name:
                        mapping[col] = asset_name
                        break
        return mapping

    @staticmethod
    def _find_spy_index(asset_cols: List[str]) -> int:
        """Return the index of the SPY column in the return matrix."""
        for i, col in enumerate(asset_cols):
            if col == "SPY_return":
                return i
        return 0  # Fallback: first column

    @staticmethod
    def _rolling_corr(window_slice: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute pairwise correlation matrix from a window of returns.
        Returns None if computation fails (e.g., constant columns).
        """
        n_assets = window_slice.shape[1]
        if window_slice.shape[0] < 2 or n_assets < 2:
            return None

        # Standardize each column
        means = window_slice.mean(axis=0)
        stds = window_slice.std(axis=0)
        stds = np.where(stds < 1e-10, 1.0, stds)  # avoid division by zero
        normed = (window_slice - means) / stds

        # Correlation = dot product of normalized columns / (n-1)
        n = window_slice.shape[0]
        corr = (normed.T @ normed) / max(n - 1, 1)

        # Clip to [-1, 1] and handle NaN
        corr = np.clip(corr, -1.0, 1.0)
        corr = np.where(np.isnan(corr), 0.0, corr)
        np.fill_diagonal(corr, 1.0)

        return corr

    def _compute_multi_head_attention(
        self,
        window_slice: np.ndarray,
        spy_idx: int,
        corr_matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Compute multi-head attention weights for the SPY node.

        For each head:
          - Project the return windows through random Q and K matrices.
          - Compute attention scores as softmax(Q_spy . K_j^T / sqrt(d_k))
            modulated by the correlation strength.
        Final attention is the mean across heads.

        Parameters
        ----------
        window_slice : np.ndarray
            Shape (window_len, n_assets).
        spy_idx : int
            Index of SPY in the asset dimension.
        corr_matrix : np.ndarray
            Shape (n_assets, n_assets) pairwise correlations.

        Returns
        -------
        np.ndarray
            Shape (n_assets,) attention weights summing to 1.
        """
        n_assets = window_slice.shape[1]
        window_len = window_slice.shape[0]
        all_attn = np.zeros((self.n_heads, n_assets))

        for h in range(self.n_heads):
            # Adapt projection matrices to actual window length
            # If window_len != self.window, truncate or pad the projection
            w_q = self._get_projection(self._W_Q[h], window_len)  # (window_len, d_k)
            w_k = self._get_projection(self._W_K[h], window_len)  # (window_len, d_k)

            # Project each asset's return window into query/key space
            # asset_returns: (window_len,) -> Q or K: (d_k,)
            q_spy = window_slice[:, spy_idx] @ w_q  # (d_k,)
            scale = np.sqrt(self._d_k)

            scores = np.zeros(n_assets)
            for j in range(n_assets):
                k_j = window_slice[:, j] @ w_k  # (d_k,)
                # Scaled dot-product attention
                raw_score = np.dot(q_spy, k_j) / (scale + 1e-10)
                # Modulate by correlation strength (amplify strongly correlated)
                corr_weight = abs(corr_matrix[spy_idx, j])
                scores[j] = raw_score * (1.0 + corr_weight)

            # Softmax to get attention weights
            all_attn[h] = _softmax(scores)

        # Average across heads
        mean_attn = all_attn.mean(axis=0)

        # Re-normalize to ensure sum = 1
        total = mean_attn.sum()
        if total > 0:
            mean_attn = mean_attn / total

        return mean_attn

    def _get_projection(self, w: np.ndarray, actual_len: int) -> np.ndarray:
        """
        Adapt a projection matrix of shape (self.window, d_k) to
        (actual_len, d_k) by truncating or zero-padding.
        """
        if actual_len == self.window:
            return w
        elif actual_len < self.window:
            return w[:actual_len]
        else:
            # Pad with zeros
            pad = np.zeros((actual_len - self.window, self._d_k))
            return np.vstack([w, pad])

    def _graph_density(self, corr_matrix: np.ndarray) -> float:
        """
        Fraction of off-diagonal pairwise correlations above the threshold.
        """
        n = corr_matrix.shape[0]
        if n < 2:
            return 0.0

        # Upper triangle (excluding diagonal)
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        upper_corrs = np.abs(corr_matrix[mask])
        n_edges = (upper_corrs > self.density_threshold).sum()
        max_edges = n * (n - 1) / 2.0
        return float(n_edges / max_edges) if max_edges > 0 else 0.0

    @staticmethod
    def _compute_regime_clusters(attention_history: np.ndarray) -> np.ndarray:
        """
        Assign each day's attention pattern to one of 3 regime clusters
        using k-means.

        Parameters
        ----------
        attention_history : np.ndarray
            Shape (n_rows, n_assets).

        Returns
        -------
        np.ndarray
            Shape (n_rows,) of cluster labels (0, 1, or 2).
        """
        n_rows = attention_history.shape[0]
        if n_rows < _KMEANS_K:
            return np.zeros(n_rows, dtype=int)

        # Only cluster rows where attention is non-trivial (not all zeros)
        row_sums = attention_history.sum(axis=1)
        active_mask = row_sums > 1e-8

        if active_mask.sum() < _KMEANS_K:
            return np.zeros(n_rows, dtype=int)

        labels = np.zeros(n_rows, dtype=int)
        active_data = attention_history[active_mask]
        active_labels = _kmeans_1d(
            active_data, k=_KMEANS_K,
            max_iter=_KMEANS_MAX_ITER, seed=_RANDOM_SEED,
        )
        labels[active_mask] = active_labels

        return labels
