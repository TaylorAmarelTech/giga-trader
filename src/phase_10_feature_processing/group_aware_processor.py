"""
GIGA TRADER - Group-Aware Feature Processor
=============================================
Replaces flat feature selection + dim reduction with group-aware processing.

Modes:
- flat: Current behavior (LeakProofFeatureSelector + LeakProofDimReducer)
- protected: Protected groups pass through untouched; rest get flat reduction
- grouped: Per-group dim reduction with proportional budgets
- grouped_protected: Protected groups pass through; non-protected get per-group reduction
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold, mutual_info_classif, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA, FastICA


# =============================================================================
# FEATURE GROUP DEFINITIONS
# =============================================================================

FEATURE_GROUPS: Dict[str, List[str]] = {
    "premarket": ["pm_", "ah_", "overnight_"],
    "breadth_streak": [
        "pct_green_", "pct_red_", "wtd_pct_green_", "wtd_pct_red_",
        "net_green_", "wtd_net_green_", "avg_streak_", "max_green_",
        "max_red_", "streak_dispersion", "breadth_divergence",
        "pct_green_3d_lag", "pct_green_3d_change",
        "pct_red_3d_lag", "pct_red_3d_change",
        "wtd_net_green_3d_lag", "wtd_net_green_3d_change",
    ],
    "mag_breadth": [
        "mag3_", "mag5_", "mag6_", "mag7_", "mag10_", "mag15_", "mag_",
    ],
    "sector": ["sector_", "xlk_", "xlf_", "xle_", "xlv_"],
    "cross_asset": [
        "TLT_", "QQQ_", "GLD_", "IWM_", "EEM_", "VXX_", "UUP_", "HYG_",
    ],
    "volatility": ["vxx_", "vol_", "realized_vol_"],
    "calendar": ["cal_", "fomc_", "opex_", "econ_"],
    "sentiment": ["sent_"],
    "fear_greed": ["fg_"],
    "social_sentiment": ["reddit_"],
    "crypto_sentiment": ["crypto_"],
    "options_flow": ["gex_"],
    "finnhub_sentiment": ["finnhub_social_"],
    "dark_pool": ["dp_"],
    "options_iv": ["opt_"],
    "event_recency": ["dts_"],
    "block_structure": ["blk_"],
    "liquidity": ["liq_"],
    "range_vol": ["rvol_"],
    "information_theory": ["ent_", "hurst_", "nmi_"],
    "regime": ["ar_", "drift_", "cpd_", "hmm_"],
    "microstructure": ["vpin_", "imom_"],
    "congressional": ["congress_"],
    "insider_aggregate": ["insider_agg_"],
    "etf_flow": ["etf_flow_"],
    "futures": ["basis_"],
    "signal_processing": ["wav_", "sax_", "te_"],
    "fractal": ["mfdfa_", "rqa_"],
    "tail_dependence": ["copula_"],
    "network": ["netw_"],
    "path_signature": ["psig_"],
    "wavelet_scattering": ["wscat_"],
    "wasserstein_regime": ["wreg_"],
    "market_structure": ["mstr_"],
    "time_series_model": ["tsm_"],
    "har_rv": ["harv_"],
    "l_moments": ["lmom_"],
    "multiscale_entropy": ["mse_"],
    "rv_signature": ["rvsp_"],
    "tda_homology": ["tda_"],
    "credit_spread": ["cred_"],
    "yield_curve": ["yc_"],
    "vol_term_structure": ["vts_"],
    "macro_surprise": ["msurp_"],
    "cross_asset_momentum": ["xmom_"],
    "skew_kurtosis": ["skku_"],
    "seasonality": ["seas_"],
    "order_flow": ["ofi_"],
    "correlation_regime": ["corr_"],
    "fama_french": ["ff_"],
    "put_call_ratio": ["pcr_"],
    "multi_horizon": ["mh_"],
    "earnings_revision": ["ern_"],
    "short_interest": ["si_"],
    "dollar_index": ["dxy_"],
    "institutional_flow": ["inst_"],
    "google_trends": ["gtrend_"],
    "commodity_signal": ["cmdty_"],
    "treasury_auction": ["tauct_"],
    "fed_liquidity": ["fedliq_"],
    "earnings_calendar": ["ecal_"],
    "analyst_rating": ["anlst_"],
    "expanded_macro": ["xmacro_"],
    "vvix": ["vvix_"],
    "sector_rotation": ["secrot_"],
    "fx_carry": ["fxc_"],
    "money_market": ["mmkt_"],
    "financial_stress": ["fstress_"],
    "global_equity": ["gleq_"],
    "retail_sentiment": ["rflow_"],
    "cboe_pcr": ["cboe_"],
    "stocktwits": ["stwit_"],
    "alpaca_news": ["anews_"],
    "gnews_headlines": ["gnews_"],
    "finbert_nlp": ["nlp_"],
    "wsb_sentiment": ["wsb_"],
    "kronos": ["kron_"],
    "graph_attention": ["gat_"],
    "patch_temporal": ["ptst_"],
    "intraday": [
        "return_at_", "high_to_", "low_to_", "range_to_",
        "rsi_at_", "macd_at_", "bb_at_", "return_from_low_",
    ],
    "daily": [
        "day_return", "day_range", "gap_", "return_ma", "return_std",
        "range_ma", "up_streak", "down_streak", "vol_ratio_", "return_zscore_",
    ],
    "technical": ["rsi_14", "macd", "bb_", "stoch_", "atr_"],
}


def assign_feature_groups(
    feature_names: List[str],
    group_defs: Dict[str, List[str]] = None,
) -> Dict[str, List[int]]:
    """
    Assign each feature to a group by prefix matching.

    Returns dict mapping group_name -> list of column indices.
    Features not matching any group go into "other".
    """
    if group_defs is None:
        group_defs = FEATURE_GROUPS

    groups: Dict[str, List[int]] = {name: [] for name in group_defs}
    groups["other"] = []

    for idx, fname in enumerate(feature_names):
        assigned = False
        for group_name, prefixes in group_defs.items():
            for prefix in prefixes:
                if fname.startswith(prefix):
                    groups[group_name].append(idx)
                    assigned = True
                    break
            if assigned:
                break
        if not assigned:
            groups["other"].append(idx)

    # Remove empty groups
    return {k: v for k, v in groups.items() if len(v) > 0}


# =============================================================================
# GROUP-AWARE FEATURE PROCESSOR
# =============================================================================

class GroupAwareFeatureProcessor(BaseEstimator, TransformerMixin):
    """
    Group-aware feature selection and dimensionality reduction.

    Supports four modes:
    - flat: Falls back to standard LeakProofFeatureSelector + LeakProofDimReducer
    - protected: Protected groups pass through (scaled only); rest get flat reduction
    - grouped: Per-group dim reduction with proportional component budgets
    - grouped_protected: Protected groups pass through; non-protected get per-group reduction
    """

    def __init__(
        self,
        feature_names: List[str],
        group_mode: str = "flat",
        protected_groups: Optional[List[str]] = None,
        budget_mode: str = "proportional",
        total_components: int = 45,
        min_components_per_group: int = 2,
        selection_method: str = "mutual_info",
        reduction_method: str = "pca",
        n_features: int = 30,
        n_components: int = 20,
        variance_threshold: float = 0.01,
        correlation_threshold: float = 0.95,
        neutralize_features: bool = False,
        random_state: int = 42,
        n_jobs: int = -1,
        nystroem_threshold: int = 5000,
    ):
        self.feature_names = list(feature_names)
        self.group_mode = group_mode
        self.protected_groups = protected_groups or []
        self.budget_mode = budget_mode
        self.total_components = total_components
        self.min_components_per_group = min_components_per_group
        self.selection_method = selection_method
        self.reduction_method = reduction_method
        self.n_features = n_features
        self.n_components = n_components
        self.variance_threshold = variance_threshold
        self.correlation_threshold = correlation_threshold
        self.neutralize_features = neutralize_features
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.nystroem_threshold = nystroem_threshold

        # Fitted state
        self._groups = None
        self._group_pipelines = {}  # group_name -> fitted pipeline dict
        self._flat_selector = None
        self._flat_reducer = None
        self._output_order = []  # ordered list of group names for transform
        self._n_features_out = None

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """Fit the processor on training data only."""
        if self.group_mode == "flat":
            return self._fit_flat(X, y)

        # Assign features to groups
        self._groups = assign_feature_groups(self.feature_names)

        if self.group_mode == "protected":
            return self._fit_protected(X, y)
        elif self.group_mode == "grouped":
            return self._fit_grouped(X, y)
        elif self.group_mode == "grouped_protected":
            return self._fit_grouped_protected(X, y)
        else:
            return self._fit_flat(X, y)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted pipelines."""
        if self.group_mode == "flat":
            return self._transform_flat(X)

        parts = []
        for group_name in self._output_order:
            pipeline = self._group_pipelines[group_name]
            indices = pipeline["indices"]
            X_group = X[:, indices]

            # Apply variance mask
            if pipeline.get("var_mask") is not None:
                X_group = X_group[:, pipeline["var_mask"]]

            # Apply correlation mask
            if pipeline.get("corr_mask") is not None:
                X_group = X_group[:, pipeline["corr_mask"]]

            # Apply selection indices
            if pipeline.get("selected_idx") is not None:
                X_group = X_group[:, pipeline["selected_idx"]]

            # Apply scaler
            if pipeline.get("scaler") is not None:
                X_group = pipeline["scaler"].transform(X_group)

            # Apply neutralizer (Wave E1)
            if pipeline.get("neutralizer") is not None:
                X_group = pipeline["neutralizer"].transform(X_group)

            # Apply reducer
            if pipeline.get("reducer") is not None:
                X_group = pipeline["reducer"].transform(X_group)

            parts.append(X_group)

        return np.hstack(parts) if parts else np.empty((X.shape[0], 0))

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X, y).transform(X)

    # =========================================================================
    # FLAT MODE (fallback to current behavior)
    # =========================================================================

    def _fit_flat(self, X: np.ndarray, y: np.ndarray = None):
        """Flat mode: standard feature selection + dim reduction."""
        from src.phase_10_feature_processing.leak_proof_selector import (
            LeakProofFeatureSelector,
        )
        from src.phase_10_feature_processing.leak_proof_reducer import (
            LeakProofDimReducer,
        )

        self._flat_selector = LeakProofFeatureSelector(
            method=self.selection_method,
            n_features=self.n_features,
            variance_threshold=self.variance_threshold,
            correlation_threshold=self.correlation_threshold,
            random_state=self.random_state,
        )
        X_sel = self._flat_selector.fit_transform(X, y)

        self._flat_reducer = LeakProofDimReducer(
            method=self.reduction_method,
            n_components=self.n_components,
            random_state=self.random_state,
        )
        self._flat_reducer.fit(X_sel, y)

        return self

    def _transform_flat(self, X: np.ndarray) -> np.ndarray:
        """Flat mode transform."""
        X_sel = self._flat_selector.transform(X)
        return self._flat_reducer.transform(X_sel)

    # =========================================================================
    # PROTECTED MODE
    # =========================================================================

    def _fit_protected(self, X: np.ndarray, y: np.ndarray = None):
        """Protected mode: protect some groups, flat-reduce the rest."""
        protected_indices = []
        non_protected_indices = []

        for group_name, indices in self._groups.items():
            if group_name in self.protected_groups:
                protected_indices.extend(indices)
            else:
                non_protected_indices.extend(indices)

        self._output_order = []
        self._group_pipelines = {}

        # Fit protected groups (variance + scale only)
        if protected_indices:
            self._fit_group_protected(
                X, "protected", sorted(protected_indices),
            )
            self._output_order.append("protected")

        # Fit non-protected with flat selection + reduction
        if non_protected_indices:
            sorted_np = sorted(non_protected_indices)
            self._fit_group_reduced(
                X, y, "non_protected", sorted_np,
                n_select=self.n_features,
                n_reduce=self.total_components,
            )
            self._output_order.append("non_protected")

        return self

    # =========================================================================
    # GROUPED MODE
    # =========================================================================

    def _fit_grouped(self, X: np.ndarray, y: np.ndarray = None):
        """Grouped mode: per-group dim reduction."""
        # Compute budgets
        reducible_groups = {k: v for k, v in self._groups.items()}
        budgets = self._compute_budgets(reducible_groups)

        self._output_order = []
        self._group_pipelines = {}

        for group_name in sorted(reducible_groups.keys()):
            indices = reducible_groups[group_name]
            budget = budgets[group_name]
            self._fit_group_reduced(
                X, y, group_name, indices,
                n_select=max(budget * 2, len(indices)),  # select more than budget
                n_reduce=budget,
            )
            self._output_order.append(group_name)

        return self

    # =========================================================================
    # GROUPED_PROTECTED MODE
    # =========================================================================

    def _fit_grouped_protected(self, X: np.ndarray, y: np.ndarray = None):
        """Grouped + protected mode."""
        protected_groups = {}
        reducible_groups = {}

        for group_name, indices in self._groups.items():
            if group_name in self.protected_groups:
                protected_groups[group_name] = indices
            else:
                reducible_groups[group_name] = indices

        # Compute budgets for reducible groups only
        budgets = self._compute_budgets(reducible_groups)

        self._output_order = []
        self._group_pipelines = {}

        # Protected groups first
        for group_name in sorted(protected_groups.keys()):
            self._fit_group_protected(
                X, group_name, protected_groups[group_name],
            )
            self._output_order.append(group_name)

        # Reducible groups
        for group_name in sorted(reducible_groups.keys()):
            indices = reducible_groups[group_name]
            budget = budgets[group_name]
            self._fit_group_reduced(
                X, y, group_name, indices,
                n_select=max(budget * 2, len(indices)),
                n_reduce=budget,
            )
            self._output_order.append(group_name)

        return self

    # =========================================================================
    # PER-GROUP FITTING HELPERS
    # =========================================================================

    def _fit_group_protected(
        self,
        X: np.ndarray,
        group_name: str,
        indices: List[int],
    ):
        """Fit a protected group: variance threshold + scale only."""
        X_group = X[:, indices]
        pipeline = {"indices": indices}

        # Variance threshold
        var_mask = self._apply_variance_threshold(X_group)
        pipeline["var_mask"] = var_mask
        X_group = X_group[:, var_mask]

        pipeline["corr_mask"] = None
        pipeline["selected_idx"] = None

        # Scale
        if X_group.shape[1] > 0:
            scaler = StandardScaler()
            scaler.fit(X_group)
            pipeline["scaler"] = scaler
        else:
            pipeline["scaler"] = None

        pipeline["reducer"] = None
        self._group_pipelines[group_name] = pipeline

    def _fit_group_reduced(
        self,
        X: np.ndarray,
        y: np.ndarray,
        group_name: str,
        indices: List[int],
        n_select: int,
        n_reduce: int,
    ):
        """Fit a group with selection + reduction."""
        X_group = X[:, indices]
        pipeline = {"indices": indices}

        # Stage 1: Variance threshold
        var_mask = self._apply_variance_threshold(X_group)
        pipeline["var_mask"] = var_mask
        X_group = X_group[:, var_mask]

        if X_group.shape[1] == 0:
            pipeline["corr_mask"] = None
            pipeline["selected_idx"] = None
            pipeline["scaler"] = None
            pipeline["reducer"] = None
            self._group_pipelines[group_name] = pipeline
            return

        # Stage 2: Correlation filter
        corr_mask = self._apply_correlation_filter(X_group)
        pipeline["corr_mask"] = corr_mask
        X_group = X_group[:, corr_mask]

        if X_group.shape[1] == 0:
            pipeline["selected_idx"] = None
            pipeline["scaler"] = None
            pipeline["reducer"] = None
            self._group_pipelines[group_name] = pipeline
            return

        # Stage 3: Feature selection (MI or f_classif)
        actual_select = min(n_select, X_group.shape[1])
        if y is not None and X_group.shape[1] > actual_select:
            selected_idx = self._apply_selection(X_group, y, actual_select)
            pipeline["selected_idx"] = selected_idx
            X_group = X_group[:, selected_idx]
        else:
            pipeline["selected_idx"] = None

        # Stage 4: Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit(X_group).transform(X_group)
        pipeline["scaler"] = scaler

        # Stage 4b: Feature neutralization (Wave E1) — remove market beta
        if self.neutralize_features and X_scaled.shape[1] > 1:
            try:
                from src.phase_10_feature_processing.feature_neutralizer import (
                    FeatureNeutralizer,
                )
                neutralizer = FeatureNeutralizer(method="demeaning")
                X_scaled = neutralizer.fit_transform(X_scaled)
                pipeline["neutralizer"] = neutralizer
            except Exception:
                pipeline["neutralizer"] = None
        else:
            pipeline["neutralizer"] = None

        # Stage 5: Dim reduction
        actual_reduce = min(n_reduce, X_scaled.shape[1] - 1, X_scaled.shape[0] - 1)
        if actual_reduce > 0 and X_scaled.shape[1] > actual_reduce:
            reducer = self._create_reducer(actual_reduce, n_samples=X_scaled.shape[0])
            try:
                reducer.fit(X_scaled)
            except Exception:
                # Fallback to PCA
                reducer = PCA(
                    n_components=actual_reduce,
                    random_state=self.random_state,
                )
                reducer.fit(X_scaled)
            pipeline["reducer"] = reducer
        else:
            pipeline["reducer"] = None

        self._group_pipelines[group_name] = pipeline

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    def _apply_variance_threshold(self, X: np.ndarray) -> np.ndarray:
        """Return boolean mask of features passing variance threshold."""
        if X.shape[1] == 0:
            return np.array([], dtype=bool)
        # Cast and replace inf/nan for safe variance computation
        X_safe = np.asarray(X, dtype=np.float64)
        X_safe = np.nan_to_num(X_safe, nan=0.0, posinf=0.0, neginf=0.0)
        vt = VarianceThreshold(threshold=self.variance_threshold)
        vt.fit(X_safe)
        return vt.get_support()

    def _apply_correlation_filter(self, X: np.ndarray) -> np.ndarray:
        """Return boolean mask removing highly correlated features."""
        if X.shape[1] <= 1:
            return np.ones(X.shape[1], dtype=bool)

        # Cast to float64 and replace inf/nan for safe correlation
        X_safe = np.asarray(X, dtype=np.float64)
        X_safe = np.where(np.isfinite(X_safe), X_safe, 0.0)

        try:
            corr = np.corrcoef(X_safe.T)
            corr = np.nan_to_num(corr, nan=0.0)
        except (ValueError, AttributeError, FloatingPointError):
            # Fallback: keep all features if corrcoef fails
            return np.ones(X.shape[1], dtype=bool)

        # Ensure corr is 2D (can be scalar/0d if only 1 feature after filtering)
        if corr.ndim < 2:
            return np.ones(X.shape[1], dtype=bool)

        to_drop = set()
        for i in range(len(corr)):
            if i in to_drop:
                continue
            for j in range(i + 1, len(corr)):
                if j in to_drop:
                    continue
                if abs(corr[i, j]) > self.correlation_threshold:
                    to_drop.add(j)

        return np.array([i not in to_drop for i in range(X.shape[1])])

    def _apply_selection(
        self, X: np.ndarray, y: np.ndarray, n_select: int,
    ) -> np.ndarray:
        """Select top features by MI or f_classif. Returns index array."""
        if self.selection_method == "f_classif":
            scores, _ = f_classif(X, y)
            scores = np.nan_to_num(scores, nan=0.0)
        else:
            scores = mutual_info_classif(
                X, y, n_neighbors=5, random_state=self.random_state,
            )

        top_idx = np.argsort(scores)[::-1][:n_select]
        return np.sort(top_idx)

    def _create_reducer(self, n_components: int, n_samples: int = 0):
        """Create a dimensionality reducer based on configured method."""
        if self.reduction_method == "kernel_pca":
            if n_samples > self.nystroem_threshold:
                # Use Nystroem approximation to avoid O(n^2) memory
                from sklearn.kernel_approximation import Nystroem
                from sklearn.pipeline import Pipeline
                n_nystroem = min(1000, n_samples // 5)
                return Pipeline([
                    ("nystroem", Nystroem(
                        kernel="rbf", gamma=0.01,
                        n_components=n_nystroem,
                        random_state=self.random_state, n_jobs=self.n_jobs
                    )),
                    ("pca", PCA(n_components=n_components, random_state=self.random_state)),
                ])
            return KernelPCA(
                n_components=n_components,
                kernel="rbf",
                gamma=0.01,
                random_state=self.random_state,
                n_jobs=self.n_jobs,
            )
        elif self.reduction_method == "ica":
            return FastICA(
                n_components=n_components,
                max_iter=500,
                random_state=self.random_state,
                whiten="unit-variance",
            )
        else:
            return PCA(
                n_components=n_components,
                random_state=self.random_state,
            )

    def _compute_budgets(
        self, groups: Dict[str, List[int]],
    ) -> Dict[str, int]:
        """Compute per-group component budgets."""
        if not groups:
            return {}

        total_features = sum(len(v) for v in groups.values())
        budgets = {}

        if self.budget_mode == "equal":
            n_groups = len(groups)
            per_group = max(
                self.min_components_per_group,
                self.total_components // n_groups,
            )
            for group_name in groups:
                budgets[group_name] = per_group
        else:
            # Proportional
            for group_name, indices in groups.items():
                if total_features > 0:
                    raw = self.total_components * len(indices) / total_features
                    budgets[group_name] = max(
                        self.min_components_per_group,
                        round(raw),
                    )
                else:
                    budgets[group_name] = self.min_components_per_group

        return budgets
