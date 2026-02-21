"""
GIGA TRADER - Feature Research Agent (Wave 32)
================================================
Iterative feature discovery system that analyzes experiment results,
generates candidate features from templates, injects them into the
experiment pipeline, and graduates proven features.

Components:
  - FeatureCandidate: Dataclass describing a candidate feature
  - FeatureResearchAgent: Main class with analyze/generate/inject/graduate
"""

import json
import logging
import random
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("GigaTrader")

project_root = Path(__file__).parent.parent.parent


# =============================================================================
# FEATURE CANDIDATE
# =============================================================================

@dataclass
class FeatureCandidate:
    """A candidate feature to be tested in research experiments."""
    name: str                          # Column name (e.g., "rc_pm_return_div_atr_pct")
    template_type: str                 # "ratio", "interaction", "lag_diff", "zscore", "kernel_rbf", "rank_ratio"
    source_features: List[str]         # Features used to compute this candidate
    params: Dict[str, Any] = field(default_factory=dict)  # Template-specific parameters

    # Tracking (populated after experiments run)
    n_experiments: int = 0             # How many experiments tested this candidate
    n_tier1_pass: int = 0              # How many passed Tier 1 with this feature
    avg_wmes: float = 0.0              # Average WMES across experiments
    avg_wf_pass_rate: float = 0.0      # Walk-forward pass rate
    graduated: bool = False            # Whether this feature has been promoted
    graduated_at: str = ""
    created_at: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

    def to_dict(self) -> Dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict) -> "FeatureCandidate":
        valid_keys = {
            "name", "template_type", "source_features", "params",
            "n_experiments", "n_tier1_pass", "avg_wmes", "avg_wf_pass_rate",
            "graduated", "graduated_at", "created_at",
        }
        d = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**d)


# =============================================================================
# FEATURE TEMPLATES
# =============================================================================

def _compute_ratio(df: pd.DataFrame, f1: str, f2: str) -> pd.Series:
    """Ratio: f1 / (f2 + epsilon)."""
    return df[f1] / (df[f2].abs() + 1e-8)


def _compute_interaction(df: pd.DataFrame, f1: str, f2: str) -> pd.Series:
    """Interaction: f1 * f2."""
    return df[f1] * df[f2]


def _compute_lag_diff(df: pd.DataFrame, f1: str, lag: int = 5) -> pd.Series:
    """Lag difference: f1 - f1.shift(lag)."""
    return df[f1] - df[f1].shift(lag)


def _compute_zscore(df: pd.DataFrame, f1: str, window: int = 20) -> pd.Series:
    """Rolling Z-score: (f1 - rolling_mean) / rolling_std."""
    roll_mean = df[f1].rolling(window, min_periods=max(1, window // 2)).mean()
    roll_std = df[f1].rolling(window, min_periods=max(1, window // 2)).std()
    return (df[f1] - roll_mean) / (roll_std + 1e-8)


def _compute_kernel_rbf(df: pd.DataFrame, f1: str, f2: str, gamma: float = 1.0) -> pd.Series:
    """Kernel RBF: exp(-gamma * (f1 - f2)^2)."""
    return np.exp(-gamma * (df[f1] - df[f2]) ** 2)


def _compute_rank_ratio(df: pd.DataFrame, f1: str, f2: str) -> pd.Series:
    """Rank ratio: rank(f1) / rank(f2)."""
    r1 = df[f1].rank(pct=True)
    r2 = df[f2].rank(pct=True)
    return r1 / (r2 + 1e-8)


TEMPLATE_REGISTRY = {
    "ratio": {
        "n_sources": 2,
        "compute": _compute_ratio,
        "prefix": "rc_ratio",
    },
    "interaction": {
        "n_sources": 2,
        "compute": _compute_interaction,
        "prefix": "rc_interact",
    },
    "lag_diff": {
        "n_sources": 1,
        "compute": _compute_lag_diff,
        "prefix": "rc_lagdiff",
        "param_choices": {"lag": [3, 5, 10, 20]},
    },
    "zscore": {
        "n_sources": 1,
        "compute": _compute_zscore,
        "prefix": "rc_zscore",
        "param_choices": {"window": [10, 20, 50]},
    },
    "kernel_rbf": {
        "n_sources": 2,
        "compute": _compute_kernel_rbf,
        "prefix": "rc_krbf",
        "param_choices": {"gamma": [0.1, 0.5, 1.0, 5.0]},
    },
    "rank_ratio": {
        "n_sources": 2,
        "compute": _compute_rank_ratio,
        "prefix": "rc_rankr",
    },
}


# =============================================================================
# FEATURE RESEARCH AGENT
# =============================================================================

class FeatureResearchAgent:
    """
    Iterative feature discovery agent.

    Analyzes experiment history to identify high-value features,
    generates candidate features from templates, injects them into
    research experiments, and graduates proven candidates.
    """

    # Graduation thresholds
    MIN_EXPERIMENTS_TO_GRADUATE = 3      # Minimum experiments before graduation
    TIER1_IMPROVEMENT_THRESHOLD = 0.05   # 5 percentage points above baseline
    MIN_WMES_IMPROVEMENT = 0.02          # 2% WMES improvement over baseline

    # Generation config
    MAX_ACTIVE_CANDIDATES = 20           # Don't create more if this many pending

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or (project_root / "data")
        self.candidates_path = self.data_dir / "feature_candidates.json"
        self.graduated_path = self.data_dir / "graduated_features.json"
        self._candidates: Dict[str, FeatureCandidate] = {}
        self._graduated: Dict[str, FeatureCandidate] = {}
        self._load_state()

    def _load_state(self):
        """Load candidates and graduated features from disk."""
        if self.candidates_path.is_file():
            try:
                with open(self.candidates_path, "r") as f:
                    data = json.load(f)
                self._candidates = {
                    k: FeatureCandidate.from_dict(v) for k, v in data.items()
                }
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                logger.warning(f"[FEATURE_RESEARCH] Failed to load candidates: {e}")
                self._candidates = {}

        if self.graduated_path.is_file():
            try:
                with open(self.graduated_path, "r") as f:
                    data = json.load(f)
                self._graduated = {
                    k: FeatureCandidate.from_dict(v) for k, v in data.items()
                }
            except (json.JSONDecodeError, TypeError, KeyError) as e:
                logger.warning(f"[FEATURE_RESEARCH] Failed to load graduated: {e}")
                self._graduated = {}

    def _save_state(self):
        """Persist candidates and graduated features to disk."""
        # Atomic write pattern
        import tempfile

        for path, data in [
            (self.candidates_path, {k: v.to_dict() for k, v in self._candidates.items()}),
            (self.graduated_path, {k: v.to_dict() for k, v in self._graduated.items()}),
        ]:
            try:
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path = path.with_suffix(".tmp")
                with open(tmp_path, "w") as f:
                    json.dump(data, f, indent=2, default=str)
                tmp_path.replace(path)
            except OSError as e:
                logger.error(f"[FEATURE_RESEARCH] Failed to save {path}: {e}")

    # ─── ANALYSIS ───────────────────────────────────────────────────────

    def analyze_feature_landscape(self) -> Dict[str, Any]:
        """
        Analyze experiment history to identify high-value features and gaps.

        Returns dict with:
          - consensus_features: List of (name, score, n_models) from UniversalFeatureMap
          - available_features: List of feature names available in recent experiments
          - n_graduated: Number of graduated features
          - n_active_candidates: Number of candidates being tested
        """
        result = {
            "consensus_features": [],
            "available_features": [],
            "n_graduated": len(self._graduated),
            "n_active_candidates": len(self._candidates),
        }

        # Try to load UniversalFeatureMap
        try:
            from src.phase_23_analytics.symbolic_cross_learner import UniversalFeatureMap
            ufm_path = self.data_dir / "universal_feature_map.json"
            if ufm_path.is_file():
                ufm = UniversalFeatureMap()
                ufm.load(ufm_path)
                result["consensus_features"] = ufm.get_consensus_features(top_n=30)
                result["available_features"] = list(ufm.profiles.keys())
        except (ImportError, Exception) as e:
            logger.debug(f"[FEATURE_RESEARCH] Could not load UniversalFeatureMap: {e}")

        return result

    def _get_source_features(self, landscape: Dict[str, Any], n: int = 2) -> List[str]:
        """
        Pick source features for candidate generation.
        Biased toward consensus features (proven valuable across models).
        Falls back to a curated list of known-good features.
        """
        # Known-good features that are always available in the pipeline
        FALLBACK_FEATURES = [
            "pm_return", "pm_range", "pm_direction", "pm_vwap_dev",
            "gap_open_pct", "day_return_lag1", "day_range_lag1",
            "rsi_14_at_0930", "macd_at_0930", "bb_position_at_0930",
            "atr_pct_at_0930", "volume_ratio_at_0930",
            "return_at_1000", "return_at_1030", "return_at_1100",
            "mom_5_at_0930", "mom_15_at_0930",
            "up_streak", "down_streak", "vol_ratio_5d",
        ]

        # Build weighted pool: consensus features get higher weight
        pool = []
        weights = []

        consensus = landscape.get("consensus_features", [])
        for name, score, n_models in consensus[:20]:
            pool.append(name)
            weights.append(max(0.1, score * 10))  # Scale up for meaningful sampling

        # Add fallback features with lower weight
        for f in FALLBACK_FEATURES:
            if f not in pool:
                pool.append(f)
                weights.append(0.5)

        if not pool:
            pool = FALLBACK_FEATURES
            weights = [1.0] * len(pool)

        # Sample n features (without replacement if possible)
        if len(pool) >= n:
            selected = []
            remaining_pool = list(zip(pool, weights))
            for _ in range(n):
                names, ws = zip(*remaining_pool)
                total = sum(ws)
                probs = [w / total for w in ws]
                idx = np.random.choice(len(names), p=probs)
                selected.append(names[idx])
                remaining_pool.pop(idx)
            return selected
        return random.sample(pool, min(n, len(pool)))

    # ─── GENERATION ─────────────────────────────────────────────────────

    def generate_candidates(self, n_candidates: int = 3) -> List[FeatureCandidate]:
        """
        Generate a batch of candidate features based on analysis.

        Each candidate is a new column computed from existing features
        using one of the registered templates (ratio, interaction, etc.).

        Returns list of FeatureCandidate objects.
        """
        # Don't generate more if we have too many active
        active_count = sum(1 for c in self._candidates.values() if not c.graduated)
        if active_count >= self.MAX_ACTIVE_CANDIDATES:
            logger.info(f"[FEATURE_RESEARCH] {active_count} active candidates, skipping generation")
            # Return existing un-graduated candidates for re-testing
            untested = [c for c in self._candidates.values()
                        if not c.graduated and c.n_experiments < self.MIN_EXPERIMENTS_TO_GRADUATE]
            return untested[:n_candidates]

        landscape = self.analyze_feature_landscape()
        candidates = []
        template_types = list(TEMPLATE_REGISTRY.keys())

        for _ in range(n_candidates):
            # Pick a random template
            ttype = random.choice(template_types)
            tmpl = TEMPLATE_REGISTRY[ttype]
            n_sources = tmpl["n_sources"]

            # Pick source features
            sources = self._get_source_features(landscape, n=n_sources)
            if len(sources) < n_sources:
                continue

            # Pick template-specific params
            params = {}
            for pname, choices in tmpl.get("param_choices", {}).items():
                params[pname] = random.choice(choices)

            # Build unique name
            src_str = "_".join(s.replace("_at_0930", "").replace("_lag1", "L1")[:12]
                               for s in sources)
            param_str = "_".join(f"{v}" for v in params.values()) if params else ""
            name = f"{tmpl['prefix']}_{src_str}"
            if param_str:
                name += f"_{param_str}"
            # Ensure uniqueness
            name_hash = hashlib.md5(name.encode()).hexdigest()[:4]
            name = f"{name}_{name_hash}"

            # Skip if already exists
            if name in self._candidates or name in self._graduated:
                continue

            candidate = FeatureCandidate(
                name=name,
                template_type=ttype,
                source_features=sources,
                params=params,
            )
            candidates.append(candidate)
            self._candidates[name] = candidate

        if candidates:
            self._save_state()
            logger.info(f"[FEATURE_RESEARCH] Generated {len(candidates)} new candidates: "
                        f"{[c.name for c in candidates]}")

        return candidates

    # ─── INJECTION ──────────────────────────────────────────────────────

    def inject_candidates(
        self, df: pd.DataFrame, config: "ExperimentConfig"
    ) -> List[str]:
        """
        Inject candidate feature columns into the DataFrame.

        Reads candidate specs from config.metadata["candidates"] and
        computes each column using the registered template functions.

        Also injects any graduated features.

        Returns list of newly added column names.
        """
        added_cols = []

        # 1. Inject graduated features (always)
        for name, grad in self._graduated.items():
            col = self._compute_candidate_column(df, grad)
            if col is not None:
                df[name] = col
                added_cols.append(name)

        # 2. Inject research candidates from config metadata
        candidate_specs = config.metadata.get("candidates", [])
        for spec in candidate_specs:
            if isinstance(spec, dict):
                candidate = FeatureCandidate.from_dict(spec)
            elif isinstance(spec, FeatureCandidate):
                candidate = spec
            else:
                continue

            col = self._compute_candidate_column(df, candidate)
            if col is not None:
                df[candidate.name] = col
                added_cols.append(candidate.name)

        if added_cols:
            logger.info(f"[FEATURE_RESEARCH] Injected {len(added_cols)} candidate columns: "
                        f"{added_cols[:5]}{'...' if len(added_cols) > 5 else ''}")

        return added_cols

    def _compute_candidate_column(
        self, df: pd.DataFrame, candidate: FeatureCandidate
    ) -> Optional[pd.Series]:
        """Compute a single candidate feature column from the DataFrame."""
        tmpl = TEMPLATE_REGISTRY.get(candidate.template_type)
        if tmpl is None:
            logger.warning(f"[FEATURE_RESEARCH] Unknown template: {candidate.template_type}")
            return None

        # Check that source features exist in df
        for src in candidate.source_features:
            if src not in df.columns:
                logger.debug(f"[FEATURE_RESEARCH] Source feature '{src}' not in DataFrame, "
                             f"skipping candidate '{candidate.name}'")
                return None

        try:
            compute_fn = tmpl["compute"]
            sources = candidate.source_features
            params = candidate.params

            if candidate.template_type in ("ratio", "interaction", "kernel_rbf", "rank_ratio"):
                col = compute_fn(df, sources[0], sources[1], **params)
            elif candidate.template_type in ("lag_diff", "zscore"):
                col = compute_fn(df, sources[0], **params)
            else:
                return None

            # Clean: replace inf with NaN, fill NaN with 0
            col = col.replace([np.inf, -np.inf], np.nan).fillna(0.0)
            return col

        except Exception as e:
            logger.warning(f"[FEATURE_RESEARCH] Failed to compute '{candidate.name}': {e}")
            return None

    # ─── EVALUATION & GRADUATION ────────────────────────────────────────

    def update_candidate_stats(
        self,
        candidate_names: List[str],
        tier1_passed: bool,
        wmes_score: float,
        walk_forward_passed: bool,
    ):
        """
        Update tracking stats for candidates after an experiment completes.

        Called by the experiment runner after a feature_research experiment finishes.
        """
        for name in candidate_names:
            if name in self._candidates:
                c = self._candidates[name]
                c.n_experiments += 1
                if tier1_passed:
                    c.n_tier1_pass += 1
                # Running average WMES
                if c.n_experiments == 1:
                    c.avg_wmes = wmes_score
                else:
                    c.avg_wmes = (c.avg_wmes * (c.n_experiments - 1) + wmes_score) / c.n_experiments
                # Walk-forward pass rate
                wf_val = 1.0 if walk_forward_passed else 0.0
                if c.n_experiments == 1:
                    c.avg_wf_pass_rate = wf_val
                else:
                    c.avg_wf_pass_rate = (c.avg_wf_pass_rate * (c.n_experiments - 1) + wf_val) / c.n_experiments

        self._save_state()

    def check_graduations(self, baseline_tier1_rate: float = 0.30) -> List[str]:
        """
        Check all candidates for graduation eligibility.

        A candidate graduates if:
          1. Tested in >= MIN_EXPERIMENTS_TO_GRADUATE experiments
          2. Tier 1 pass rate > baseline + TIER1_IMPROVEMENT_THRESHOLD

        Returns list of newly graduated feature names.
        """
        newly_graduated = []

        for name, candidate in list(self._candidates.items()):
            if candidate.graduated:
                continue
            if candidate.n_experiments < self.MIN_EXPERIMENTS_TO_GRADUATE:
                continue

            candidate_tier1_rate = candidate.n_tier1_pass / candidate.n_experiments
            improvement = candidate_tier1_rate - baseline_tier1_rate

            if improvement >= self.TIER1_IMPROVEMENT_THRESHOLD:
                candidate.graduated = True
                candidate.graduated_at = datetime.now().isoformat()
                self._graduated[name] = candidate
                newly_graduated.append(name)
                logger.info(
                    f"[FEATURE_RESEARCH] GRADUATED: {name} "
                    f"(tier1_rate={candidate_tier1_rate:.1%} vs baseline={baseline_tier1_rate:.1%}, "
                    f"improvement={improvement:+.1%}, WMES={candidate.avg_wmes:.3f})"
                )

        if newly_graduated:
            self._save_state()

        return newly_graduated

    def get_baseline_stats(self) -> Dict[str, float]:
        """
        Compute baseline statistics from recent non-research experiments.

        Uses RegistryDB to query recent experiment results.
        """
        baseline = {
            "tier1_pass_rate": 0.30,  # Conservative default
            "avg_wmes": 0.45,
            "wf_pass_rate": 0.50,
        }

        try:
            from src.core.registry_db import RegistryDB
            db = RegistryDB()
            conn = db._get_conn()

            # Count recent non-research experiments (last 100)
            rows = conn.execute("""
                SELECT test_auc, wmes_score, walk_forward_passed
                FROM experiments
                WHERE status = 'completed'
                ORDER BY rowid DESC
                LIMIT 100
            """).fetchall()

            if len(rows) >= 10:
                n_total = len(rows)
                # Tier 1 pass = AUC > 0.55 AND WMES >= 0.40 AND WF passed
                n_tier1 = sum(
                    1 for r in rows
                    if r[0] and r[0] > 0.55
                    and r[1] and r[1] >= 0.40
                    and r[2]
                )
                baseline["tier1_pass_rate"] = n_tier1 / n_total

                wmes_scores = [r[1] for r in rows if r[1] is not None and r[1] > 0]
                if wmes_scores:
                    baseline["avg_wmes"] = sum(wmes_scores) / len(wmes_scores)

                wf_results = [r[2] for r in rows if r[2] is not None]
                if wf_results:
                    baseline["wf_pass_rate"] = sum(1 for w in wf_results if w) / len(wf_results)

        except Exception as e:
            logger.debug(f"[FEATURE_RESEARCH] Could not compute baseline: {e}")

        return baseline

    def get_candidates_for_config(self, n: int = 3) -> List[Dict]:
        """
        Get candidate specs to embed in experiment config metadata.

        Returns list of candidate dicts suitable for config.metadata["candidates"].
        """
        candidates = self.generate_candidates(n_candidates=n)
        return [c.to_dict() for c in candidates]

    def get_graduated_feature_names(self) -> List[str]:
        """Get list of graduated feature names for inclusion in all experiments."""
        return list(self._graduated.keys())

    def summary(self) -> str:
        """Human-readable summary of the feature research state."""
        active = sum(1 for c in self._candidates.values() if not c.graduated)
        tested = sum(1 for c in self._candidates.values() if c.n_experiments > 0)
        lines = [
            f"[FEATURE RESEARCH] Active candidates: {active}, Tested: {tested}, "
            f"Graduated: {len(self._graduated)}",
        ]
        if self._graduated:
            lines.append("  Graduated features:")
            for name, c in self._graduated.items():
                lines.append(f"    - {name} (tier1={c.n_tier1_pass}/{c.n_experiments}, "
                             f"WMES={c.avg_wmes:.3f})")
        return "\n".join(lines)
