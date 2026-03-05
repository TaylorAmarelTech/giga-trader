"""
GIGA TRADER - Correlation Network Centrality Features
======================================================
Model the cross-asset universe as a graph where each asset is a node
and an edge exists between two assets when their rolling return correlation
exceeds a threshold.  Five graph-theoretic features capture systemic risk
regimes that are invisible to individual-asset metrics.

Features (5, prefix: netw_):
  netw_density       : Fraction of possible edges that are "active"
                       (|corr| > threshold).  High density = tightly
                       coupled markets = systemic risk warning.
  netw_avg_centrality: Average degree centrality across nodes.
                       degree_centrality(i) = degree(i) / (n-1).
                       Mean across all nodes gives a single scalar.
  netw_centrality_z  : 60-day rolling z-score of netw_density.
  netw_modularity    : Proxy for graph modularity via connected
                       components.  Uses BFS (no networkx).
                       Value = 1 - 1/n_components.  0 = fully
                       connected (no modules); near 1 = fragmented.
  netw_hub_disconnect: Rate at which the most-connected node loses
                       edges.  (max_degree_today - max_degree_5d_ago)
                       / (n_nodes - 1).  Negative value = hub
                       disconnecting = regime-change signal.

Cross-asset return columns recognised (from CrossAssetFeatures):
  TLT_return, QQQ_return, GLD_return, IWM_return,
  EEM_return, HYG_return, VXX_return

Minimum requirement: a 'close' column (SPY daily return is derived
from it).  Cross-asset columns are optional but improve the network.
"""

import logging
from collections import deque
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.core.feature_base import FeatureModuleBase

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────────────────────────────────────
# Module-level constants
# ──────────────────────────────────────────────────────────────────────────────

_CROSS_ASSET_RETURN_COLS: List[str] = [
    "TLT_return",
    "QQQ_return",
    "GLD_return",
    "IWM_return",
    "EEM_return",
    "HYG_return",
    "VXX_return",
]

_FEATURE_NAMES: List[str] = [
    "netw_density",
    "netw_avg_centrality",
    "netw_centrality_z",
    "netw_modularity",
    "netw_hub_disconnect",
]


# ──────────────────────────────────────────────────────────────────────────────
# BFS helper (no networkx)
# ──────────────────────────────────────────────────────────────────────────────

def _count_connected_components(adj: np.ndarray) -> int:
    """
    Count connected components in an undirected adjacency matrix using BFS.

    Parameters
    ----------
    adj : np.ndarray
        Square boolean adjacency matrix (n x n).

    Returns
    -------
    int
        Number of connected components.  Returns 1 when n == 0.
    """
    n = adj.shape[0]
    if n == 0:
        return 1

    visited = np.zeros(n, dtype=bool)
    n_components = 0

    for start in range(n):
        if visited[start]:
            continue
        # BFS from 'start'
        queue: deque[int] = deque([start])
        visited[start] = True
        while queue:
            node = queue.popleft()
            for neighbour in range(n):
                if not visited[neighbour] and adj[node, neighbour]:
                    visited[neighbour] = True
                    queue.append(neighbour)
        n_components += 1

    return n_components


# ──────────────────────────────────────────────────────────────────────────────
# NetworkFeatures
# ──────────────────────────────────────────────────────────────────────────────

class NetworkFeatures(FeatureModuleBase):
    """
    Compute correlation-network centrality features from daily return data.

    Parameters
    ----------
    window : int
        Rolling window (trading days) for pairwise correlation.  Default 60.
    correlation_threshold : float
        Absolute correlation above which an edge exists in the graph.
        Default 0.5.
    """

    REQUIRED_COLS = {"close"}
    FEATURE_NAMES = [
        "netw_density",
        "netw_avg_centrality",
        "netw_centrality_z",
        "netw_modularity",
        "netw_hub_disconnect",
    ]

    def __init__(
        self,
        window: int = 60,
        correlation_threshold: float = 0.5,
    ) -> None:
        self.window = window
        self.correlation_threshold = correlation_threshold

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def download_network_data(
        self,
        start_date,  # datetime | str
        end_date,    # datetime | str
    ) -> pd.DataFrame:
        """
        Placeholder download method — network features are derived entirely
        from data already present in df_daily (close + cross-asset returns).

        Returns an empty DataFrame; no external data is fetched.
        """
        logger.info("NetworkFeatures.download_network_data: no external data needed")
        return pd.DataFrame()

    def create_network_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build the 5 netw_ features and append them to df.

        Parameters
        ----------
        df : pd.DataFrame
            Daily bar DataFrame.  Must contain 'close'.  Cross-asset return
            columns (TLT_return, QQQ_return, …) are used when present.

        Returns
        -------
        pd.DataFrame
            Original df with 5 netw_ columns appended.  All features are
            filled with 0.0 on failure so downstream code is never broken.
        """
        print("\n[NETWORK] Computing correlation-network centrality features...")

        out = df.copy()

        if "close" not in out.columns:
            logger.warning("NetworkFeatures: 'close' column missing — skipping")
            for col in _FEATURE_NAMES:
                out[col] = 0.0
            return out

        # ── Build the return matrix ─────────────────────────────────────
        return_matrix = self._build_return_matrix(out)
        n_assets = return_matrix.shape[1]

        if n_assets < 2:
            logger.info(
                "NetworkFeatures: fewer than 2 return series — "
                "network is trivial, defaulting to 0.0"
            )
            for col in _FEATURE_NAMES:
                out[col] = 0.0
            return out

        n_rows = len(return_matrix)

        # Pre-allocate result arrays
        density_arr = np.full(n_rows, np.nan)
        avg_centrality_arr = np.full(n_rows, np.nan)
        modularity_arr = np.full(n_rows, np.nan)
        max_degree_arr = np.full(n_rows, np.nan)

        min_periods = max(2, self.window // 3)

        for i in range(n_rows):
            start = max(0, i - self.window + 1)
            window_slice = return_matrix.iloc[start : i + 1]
            window_clean = window_slice.dropna(how="all")

            # Need at least min_periods rows and 2 columns with data
            if len(window_clean) < min_periods:
                continue

            # Drop columns that are all-NaN in this window
            window_clean = window_clean.dropna(axis=1, how="all")
            n = window_clean.shape[1]
            if n < 2:
                continue

            # Pairwise correlation matrix (n x n)
            try:
                corr = window_clean.corr().values
            except Exception:
                continue

            # Replace NaN in correlation with 0 (treat as uncorrelated).
            # Use np.where instead of nan_to_num(copy=False) because the
            # array returned by pandas .values can be read-only.
            corr = np.where(np.isnan(corr), 0.0, corr)

            # Adjacency matrix: edge if |corr| > threshold, no self-loops
            adj = (np.abs(corr) > self.correlation_threshold).astype(np.int8)
            np.fill_diagonal(adj, 0)

            # ── Graph metrics ──────────────────────────────────────────
            max_possible_edges = n * (n - 1) / 2.0

            # Density (upper triangle edges only)
            n_edges = int(np.triu(adj, k=1).sum())
            density = n_edges / max_possible_edges if max_possible_edges > 0 else 0.0

            # Degree of each node
            degrees = adj.sum(axis=1).astype(float)

            # Average degree centrality = mean(degree / (n-1))
            avg_centrality = (degrees / (n - 1)).mean() if n > 1 else 0.0

            # Max degree (for hub-disconnect tracking)
            max_degree = float(degrees.max())

            # Modularity proxy via connected components (BFS)
            adj_bool = adj.astype(bool)
            n_components = _count_connected_components(adj_bool)
            modularity = 1.0 - 1.0 / n_components if n_components > 0 else 0.0

            density_arr[i] = density
            avg_centrality_arr[i] = avg_centrality
            modularity_arr[i] = modularity
            max_degree_arr[i] = max_degree

        # ── Convert to Series ──────────────────────────────────────────
        idx = return_matrix.index

        density_s = pd.Series(density_arr, index=idx)
        avg_centrality_s = pd.Series(avg_centrality_arr, index=idx)
        modularity_s = pd.Series(modularity_arr, index=idx)
        max_degree_s = pd.Series(max_degree_arr, index=idx)

        # ── netw_centrality_z: rolling z-score of density ─────────────
        z_min = max(2, self.window // 3)
        roll_mean = density_s.rolling(self.window, min_periods=z_min).mean()
        roll_std = density_s.rolling(self.window, min_periods=z_min).std()
        centrality_z_s = (density_s - roll_mean) / (roll_std + 1e-10)

        # ── netw_hub_disconnect: change in max-degree over 5 days ──────
        hub_disconnect_s = max_degree_s.diff(5)
        # Normalise by (n_assets - 1) so it is scale-invariant
        if n_assets > 1:
            hub_disconnect_s = hub_disconnect_s / (n_assets - 1)

        # ── Assign to output DataFrame ─────────────────────────────────
        out["netw_density"] = density_s.values
        out["netw_avg_centrality"] = avg_centrality_s.values
        out["netw_centrality_z"] = centrality_z_s.values
        out["netw_modularity"] = modularity_s.values
        out["netw_hub_disconnect"] = hub_disconnect_s.values

        # ── Cleanup: fill NaN / inf → 0.0 ─────────────────────────────
        for col in _FEATURE_NAMES:
            out[col] = (
                out[col]
                .fillna(0.0)
                .replace([np.inf, -np.inf], 0.0)
            )

        print(f"  [NETWORK] Added {len(_FEATURE_NAMES)} network centrality features")
        return out

    def analyze_current_network(self, df: pd.DataFrame) -> Optional[Dict]:
        """
        Analyse the most recent network state and return a regime dict.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame that already contains the netw_ feature columns
            (i.e. output of create_network_features).

        Returns
        -------
        dict | None
            Keys:
              network_regime  : "SYSTEMIC_RISK" | "NORMAL" | "FRAGMENTED"
              netw_density    : float
              netw_avg_centrality : float
              netw_centrality_z   : float
              netw_modularity     : float
              netw_hub_disconnect : float
            Returns None if the feature columns are absent or df is too short.
        """
        missing = [c for c in _FEATURE_NAMES if c not in df.columns]
        if missing or len(df) < 1:
            return None

        last = df.iloc[-1]

        density = float(last.get("netw_density", 0.0))
        z = float(last.get("netw_centrality_z", 0.0))
        modularity = float(last.get("netw_modularity", 0.0))
        avg_centrality = float(last.get("netw_avg_centrality", 0.0))
        hub_disconnect = float(last.get("netw_hub_disconnect", 0.0))

        # Regime classification
        # SYSTEMIC_RISK : highly dense + positively elevated z-score
        # FRAGMENTED    : high modularity (many disconnected sub-graphs)
        # NORMAL        : everything else
        if density > 0.6 and z > 1.0:
            regime = "SYSTEMIC_RISK"
        elif modularity > 0.5:
            regime = "FRAGMENTED"
        else:
            regime = "NORMAL"

        return {
            "network_regime": regime,
            "netw_density": round(density, 4),
            "netw_avg_centrality": round(avg_centrality, 4),
            "netw_centrality_z": round(z, 3),
            "netw_modularity": round(modularity, 4),
            "netw_hub_disconnect": round(hub_disconnect, 4),
        }

    # ──────────────────────────────────────────────────────────────────────
    # Private helpers
    # ──────────────────────────────────────────────────────────────────────

    def _build_return_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Assemble the return matrix used to build the correlation network.

        Uses SPY daily return (derived from 'close') and any cross-asset
        return columns that are present in df.

        Returns
        -------
        pd.DataFrame
            Columns are individual return series.  Index matches df.
        """
        matrix = pd.DataFrame(index=df.index)

        # SPY return from close
        matrix["SPY_return"] = df["close"].pct_change()

        # Append any cross-asset return columns that are present
        for col in _CROSS_ASSET_RETURN_COLS:
            if col in df.columns:
                matrix[col] = df[col].values

        return matrix
