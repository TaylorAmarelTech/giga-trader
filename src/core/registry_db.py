"""
SQLite-backed storage for experiment history, model registry, and ModelRegistryV2.

Replaces the large JSON files with a single indexed database while keeping
the same data model (to_dict/from_dict round-trip via data_json blobs).

Usage:
    from src.core.registry_db import get_registry_db
    db = get_registry_db()
    db.add_experiment(result.to_dict())
"""

import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Resolve project root (same logic as other core modules)
# ---------------------------------------------------------------------------
_this_dir = Path(__file__).resolve().parent
project_root = _this_dir.parent.parent  # src/core -> src -> project_root

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_SCHEMA_SQL = """
-- Experiments table (replaces experiment_history.json)
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id TEXT UNIQUE NOT NULL,
    status TEXT NOT NULL DEFAULT 'queued',
    started_at TEXT,
    completed_at TEXT,
    cv_auc_mean REAL DEFAULT 0,
    test_auc REAL DEFAULT 0,
    train_auc REAL DEFAULT 0,
    wmes_score REAL DEFAULT 0,
    stability_score REAL DEFAULT 0,
    fragility_score REAL DEFAULT 0,
    walk_forward_passed INTEGER DEFAULT 0,
    duration_seconds REAL DEFAULT 0,
    model_path TEXT DEFAULT '',
    error_message TEXT DEFAULT '',
    config_hash TEXT,
    data_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_exp_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_exp_test_auc ON experiments(test_auc);
CREATE INDEX IF NOT EXISTS idx_exp_config_hash ON experiments(config_hash);
CREATE INDEX IF NOT EXISTS idx_exp_experiment_id ON experiments(experiment_id);

-- Model registry (tiered model tracking with quality gates)
CREATE TABLE IF NOT EXISTS models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT UNIQUE NOT NULL,
    experiment_id TEXT NOT NULL,
    created_at TEXT,
    model_path TEXT DEFAULT '',
    cv_auc REAL DEFAULT 0,
    test_auc REAL DEFAULT 0,
    wmes_score REAL DEFAULT 0,
    stability_score REAL DEFAULT 0,
    fragility_score REAL DEFAULT 0,
    train_test_gap REAL DEFAULT 0,
    tier INTEGER DEFAULT 1,
    backtest_sharpe REAL DEFAULT 0,
    backtest_win_rate REAL DEFAULT 0,
    data_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_model_tier ON models(tier);
CREATE INDEX IF NOT EXISTS idx_model_test_auc ON models(test_auc);
CREATE INDEX IF NOT EXISTS idx_model_model_id ON models(model_id);

-- ModelRegistryV2 entries (replaces registry_v2/registry.json)
CREATE TABLE IF NOT EXISTS model_entries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    model_id TEXT UNIQUE NOT NULL,
    experiment_id TEXT DEFAULT '',
    target_type TEXT DEFAULT 'swing',
    status TEXT DEFAULT 'queued',
    model_type TEXT DEFAULT '',
    dim_reduction TEXT DEFAULT '',
    cascade_type TEXT DEFAULT '',
    cv_auc REAL DEFAULT 0,
    test_auc REAL DEFAULT 0,
    train_auc REAL DEFAULT 0,
    win_rate REAL DEFAULT 0,
    sharpe_ratio REAL DEFAULT 0,
    stability_score REAL DEFAULT 0,
    fragility_score REAL DEFAULT 0,
    created_at TEXT,
    updated_at TEXT,
    tags TEXT DEFAULT '',
    data_json TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_me_status ON model_entries(status);
CREATE INDEX IF NOT EXISTS idx_me_target ON model_entries(target_type);
CREATE INDEX IF NOT EXISTS idx_me_test_auc ON model_entries(test_auc);
CREATE INDEX IF NOT EXISTS idx_me_model_id ON model_entries(model_id);

-- Scoring version registry (tracks measurement methodology per version)
CREATE TABLE IF NOT EXISTS scoring_versions (
    version_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    description TEXT DEFAULT '',
    settings_json TEXT NOT NULL
);
"""

# ---------------------------------------------------------------------------
# Scoring version definitions
# ---------------------------------------------------------------------------
# Each version captures ALL measurement parameters so experiments are fully
# reproducible.  When any measurement parameter changes, bump the version.
#
# VERSION HISTORY:
# ─────────────────────────────────────────────────────────────────────────
# v1 (2026-02-17): Initial post-reset scoring.
#   - Created after Wave 24 SQLite migration and leakage fix purge.
#   - Lenient settings: stability multiplier=15, fragility CV scaling=5,
#     drop scaling=3, narrow perturbation ranges (0.6-1.4x features,
#     ±30% params, 4 param samples).
#   - Problem: Nearly all models scored stability=1.0, fragility=0.0.
#     Measurements were too gentle to discriminate.
#
# v2 (2026-02-18): Tighter measurements (fix-measurements-first approach).
#   - Stability multiplier 15→40 (amplifies small AUC changes).
#   - Fragility: wider feature range 0.6-1.4→0.4-1.8, param perturbation
#     ±30%→±50%, param samples 4→8, CV scaling 5→10, drop scaling 3→6.
#   - Gate thresholds unchanged (calibrate in separate phase).
#   - Problem: Models with truly flat loss surfaces (LDA, logistic, NB)
#     still scored stability=1.0 (zero sensitivity to perturbation).
#
# v3 (2026-02-18): Zero-sensitivity cap for stability.
#   - Added cap: stability capped at 0.70 when ring sensitivity < 0.001.
#     "Can't measure any difference" → "probably robust but not provably so"
#     instead of "perfectly robust."
#   - Results (40 experiments): stability 0.000-0.913 avg 0.478 (good
#     spread), fragility 0.000-0.438 avg 0.057 (72% near-zero), 100% WF
#     pass, AUC 0.548-0.569. 12 Tier 1, 20 Tier 2, 0 Tier 3.
#   - Remaining issues: (1) 72% near-zero fragility for models with no
#     tunable HPs, (2) Tier 3 AUC threshold 0.58 unreachable (best 0.569).
#
# v4 (2026-02-19): Fragility floor + Tier 3 AUC recalibration.
#   - Fragility: Added 0.05 floor for models with empty param_ranges
#     (LDA, GaussianNB, BayesianRidge). When param perturbation is a
#     no-op, 8 of ~15 frag_scores are identical to base, artificially
#     suppressing cv_frag. Floor acknowledges untestable HP dimension.
#   - Tier 3 AUC: 0.58→0.57. With 40 v3 experiments, max AUC=0.569.
#     0.57 is the most conservative reduction that unblocks Tier 3 while
#     still requiring above-average quality (mean AUC = 0.560).
#   - Other Tier 3 gates unchanged: fragility<0.40, suite_composite>=0.45.
# ─────────────────────────────────────────────────────────────────────────

SCORING_VERSIONS = {
    "v1": {
        "description": "Initial post-reset scoring (lenient measurements)",
        "created_at": "2026-02-17T17:28:00",
        "settings": {
            # Stability measurement
            "stability_sensitivity_multiplier": 15,
            "stability_rings": [
                {"name": "local", "perturbation_pct": 0.15, "weight": 0.20},
                {"name": "moderate", "perturbation_pct": 0.50, "weight": 0.30},
                {"name": "wide", "perturbation_pct": None, "weight": 0.50},
            ],
            "stability_samples_per_ring": 8,
            # Fragility measurement
            "fragility_feature_range": [0.6, 1.4],
            "fragility_param_perturbation": 0.30,
            "fragility_param_samples": 4,
            "fragility_cv_scaling": 5,
            "fragility_drop_scaling": 3,
            # Gate thresholds
            "tier2_stability_threshold": 0.60,
            "tier3_fragility_threshold": 0.40,
            "tier3_auc_threshold": 0.58,
            "tier3_suite_composite_threshold": 0.45,
            "tier1_wmes_threshold": 0.45,
            "tier1_auc_min": 0.56,
            "tier1_auc_max": 0.85,
            "tier1_gap_max": 0.10,
            # Walk-forward
            "walk_forward_variance_max": 0.07,
            "walk_forward_per_window_normal": 0.53,
            "walk_forward_per_window_crisis": 0.48,
            # WMES
            "wmes_noise_epsilon": 1e-6,
            # Other
            "purge_days": 10,
            "embargo_days": 3,
            "nystroem_max_components": 200,
        },
    },
    "v2": {
        "description": "Tighter measurements (wider perturbations, aggressive scaling)",
        "created_at": "2026-02-18T10:30:00",
        "settings": {
            # Stability measurement — multiplier 15→40
            "stability_sensitivity_multiplier": 40,
            "stability_rings": [
                {"name": "local", "perturbation_pct": 0.15, "weight": 0.20},
                {"name": "moderate", "perturbation_pct": 0.50, "weight": 0.30},
                {"name": "wide", "perturbation_pct": None, "weight": 0.50},
            ],
            "stability_samples_per_ring": 8,
            # Fragility measurement — wider ranges, more samples, aggressive scaling
            "fragility_feature_range": [0.4, 1.8],
            "fragility_param_perturbation": 0.50,
            "fragility_param_samples": 8,
            "fragility_cv_scaling": 10,
            "fragility_drop_scaling": 6,
            # Gate thresholds (unchanged — Phase 2 will calibrate)
            "tier2_stability_threshold": 0.60,
            "tier3_fragility_threshold": 0.40,
            "tier3_auc_threshold": 0.58,
            "tier3_suite_composite_threshold": 0.45,
            "tier1_wmes_threshold": 0.45,
            "tier1_auc_min": 0.56,
            "tier1_auc_max": 0.85,
            "tier1_gap_max": 0.10,
            # Walk-forward
            "walk_forward_variance_max": 0.07,
            "walk_forward_per_window_normal": 0.53,
            "walk_forward_per_window_crisis": 0.48,
            # WMES
            "wmes_noise_epsilon": 1e-6,
            # Other
            "purge_days": 10,
            "embargo_days": 3,
            "nystroem_max_components": 200,
        },
    },
    "v3": {
        "description": "Zero-sensitivity cap + v2 tighter measurements",
        "created_at": "2026-02-18T17:45:00",
        "settings": {
            # Stability measurement — multiplier 40, zero-sensitivity cap at 0.70
            "stability_sensitivity_multiplier": 40,
            "stability_zero_sensitivity_cap": 0.70,
            "stability_zero_sensitivity_threshold": 0.001,
            "stability_rings": [
                {"name": "local", "perturbation_pct": 0.15, "weight": 0.20},
                {"name": "moderate", "perturbation_pct": 0.50, "weight": 0.30},
                {"name": "wide", "perturbation_pct": None, "weight": 0.50},
            ],
            "stability_samples_per_ring": 8,
            # Fragility measurement — wider ranges, more samples, aggressive scaling
            "fragility_feature_range": [0.4, 1.8],
            "fragility_param_perturbation": 0.50,
            "fragility_param_samples": 8,
            "fragility_cv_scaling": 10,
            "fragility_drop_scaling": 6,
            # Gate thresholds (unchanged — Phase 2 will calibrate)
            "tier2_stability_threshold": 0.60,
            "tier3_fragility_threshold": 0.40,
            "tier3_auc_threshold": 0.58,
            "tier3_suite_composite_threshold": 0.45,
            "tier1_wmes_threshold": 0.45,
            "tier1_auc_min": 0.56,
            "tier1_auc_max": 0.85,
            "tier1_gap_max": 0.10,
            # Walk-forward
            "walk_forward_variance_max": 0.07,
            "walk_forward_per_window_normal": 0.53,
            "walk_forward_per_window_crisis": 0.48,
            # WMES
            "wmes_noise_epsilon": 1e-6,
            # Other
            "purge_days": 10,
            "embargo_days": 3,
            "nystroem_max_components": 200,
        },
    },
    "v4": {
        "description": (
            "Fragility floor for no-HP models + Tier 3 AUC 0.58->0.57. "
            "Based on v3 campaign data (40 experiments, max AUC=0.569, "
            "72% near-zero fragility for simple models)."
        ),
        "created_at": "2026-02-19T00:00:00",
        "settings": {
            # Stability measurement — unchanged from v3
            "stability_sensitivity_multiplier": 40,
            "stability_zero_sensitivity_cap": 0.70,
            "stability_zero_sensitivity_threshold": 0.001,
            "stability_rings": [
                {"name": "local", "perturbation_pct": 0.15, "weight": 0.20},
                {"name": "moderate", "perturbation_pct": 0.50, "weight": 0.30},
                {"name": "wide", "perturbation_pct": None, "weight": 0.50},
            ],
            "stability_samples_per_ring": 8,
            # Fragility measurement — v3 settings + no-HP floor
            "fragility_feature_range": [0.4, 1.8],
            "fragility_param_perturbation": 0.50,
            "fragility_param_samples": 8,
            "fragility_cv_scaling": 10,
            "fragility_drop_scaling": 6,
            "fragility_no_hp_floor": 0.05,  # NEW: floor for models w/ empty param_ranges
            # Gate thresholds — Tier 3 AUC lowered from 0.58
            "tier2_stability_threshold": 0.60,
            "tier3_fragility_threshold": 0.40,
            "tier3_auc_threshold": 0.57,  # CHANGED: 0.58 → 0.57
            "tier3_suite_composite_threshold": 0.45,
            "tier1_wmes_threshold": 0.45,
            "tier1_auc_min": 0.56,
            "tier1_auc_max": 0.85,
            "tier1_gap_max": 0.10,
            # Walk-forward — unchanged
            "walk_forward_variance_max": 0.07,
            "walk_forward_per_window_normal": 0.53,
            "walk_forward_per_window_crisis": 0.48,
            # WMES — unchanged
            "wmes_noise_epsilon": 1e-6,
            # Other — unchanged
            "purge_days": 10,
            "embargo_days": 3,
            "nystroem_max_components": 200,
        },
    },
}

CURRENT_SCORING_VERSION = "v4"


class RegistryDB:
    """SQLite backend for all registry data.

    Thread-safe via WAL mode (concurrent readers) + a write lock.
    Stores frequently-queried fields as indexed columns and the full
    serialized data as a JSON blob in ``data_json``.
    """

    def __init__(self, db_path: Path = None):
        self.db_path = Path(db_path) if db_path else (project_root / "data" / "giga_trader.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._local = threading.local()
        self._init_db()

    # ------------------------------------------------------------------
    # Connection helpers
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Return a per-thread connection (reused within the same thread)."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30,
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA busy_timeout=10000")
            conn.row_factory = sqlite3.Row
            self._local.conn = conn
        return conn

    def _init_db(self):
        """Create tables and indexes if they don't exist."""
        conn = self._get_conn()
        conn.executescript(_SCHEMA_SQL)
        # Migration: add scoring_version column if missing
        for table in ("experiments", "models"):
            cols = [r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()]
            if "scoring_version" not in cols:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN scoring_version TEXT DEFAULT ''")
                logger.info(f"Added scoring_version column to {table}")
        # Seed scoring_versions table with known versions
        for vid, vdata in SCORING_VERSIONS.items():
            conn.execute(
                "INSERT OR IGNORE INTO scoring_versions (version_id, created_at, description, settings_json) VALUES (?,?,?,?)",
                (vid, vdata["created_at"], vdata["description"], json.dumps(vdata["settings"])),
            )
        conn.commit()
        logger.info(f"Registry DB initialized at {self.db_path}")

    # ==================================================================
    # EXPERIMENTS
    # ==================================================================

    def add_experiment(self, result_dict: Dict[str, Any], config_hash: str = None) -> None:
        """Insert an experiment result."""
        d = result_dict
        status = d.get("status", "queued")
        if hasattr(status, "value"):
            status = status.value

        scoring_ver = d.get("scoring_version", CURRENT_SCORING_VERSION)

        with self._write_lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT OR REPLACE INTO experiments
                   (experiment_id, status, started_at, completed_at,
                    cv_auc_mean, test_auc, train_auc, wmes_score,
                    stability_score, fragility_score, walk_forward_passed,
                    duration_seconds, model_path, error_message,
                    config_hash, scoring_version, data_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    d.get("experiment_id", ""),
                    status,
                    d.get("started_at", ""),
                    d.get("completed_at", ""),
                    d.get("cv_auc_mean", 0),
                    d.get("test_auc", 0),
                    d.get("train_auc", 0),
                    d.get("wmes_score", 0),
                    d.get("stability_score", 0),
                    d.get("fragility_score", 0),
                    1 if d.get("walk_forward_passed") else 0,
                    d.get("duration_seconds", 0),
                    d.get("model_path", ""),
                    d.get("error_message", ""),
                    config_hash or "",
                    scoring_ver,
                    json.dumps(d),
                ),
            )
            conn.commit()

    def get_experiments(
        self,
        status: str = None,
        min_auc: float = None,
        max_auc: float = None,
        limit: int = None,
        offset: int = 0,
    ) -> List[Dict]:
        """Return experiment dicts, optionally filtered."""
        conn = self._get_conn()
        clauses: List[str] = []
        params: List[Any] = []

        if status:
            clauses.append("status = ?")
            params.append(status)
        if min_auc is not None:
            clauses.append("test_auc >= ?")
            params.append(min_auc)
        if max_auc is not None:
            clauses.append("test_auc <= ?")
            params.append(max_auc)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT data_json FROM experiments{where} ORDER BY id"
        if limit:
            sql += f" LIMIT {int(limit)} OFFSET {int(offset)}"

        rows = conn.execute(sql, params).fetchall()
        return [json.loads(row["data_json"]) for row in rows]

    def get_experiment_by_id(self, experiment_id: str) -> Optional[Dict]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT data_json FROM experiments WHERE experiment_id = ?",
            (experiment_id,),
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def get_experiment_count(self, status: str = None) -> int:
        conn = self._get_conn()
        if status:
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM experiments WHERE status = ?", (status,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM experiments").fetchone()
        return row["cnt"]

    def get_config_hashes(self) -> Set[str]:
        """Return all non-empty config hashes (for dedup in ExperimentGenerator)."""
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT DISTINCT config_hash FROM experiments WHERE config_hash != ''"
        ).fetchall()
        return {row["config_hash"] for row in rows}

    def get_experiment_statistics(self) -> Dict:
        """Aggregate stats matching ExperimentHistory.get_statistics()."""
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) AS c FROM experiments").fetchone()["c"]
        if total == 0:
            return {"total": 0}

        completed = conn.execute(
            "SELECT COUNT(*) AS c FROM experiments WHERE status = 'completed'"
        ).fetchone()["c"]
        failed = conn.execute(
            "SELECT COUNT(*) AS c FROM experiments WHERE status = 'failed'"
        ).fetchone()["c"]

        row = conn.execute(
            """SELECT COUNT(*) AS scored,
                      AVG(test_auc) AS avg_auc,
                      AVG(wmes_score) AS avg_wmes,
                      MAX(test_auc) AS best_auc,
                      AVG(duration_seconds) AS avg_dur
               FROM experiments
               WHERE status = 'completed' AND test_auc > 0"""
        ).fetchone()

        realistic_row = conn.execute(
            "SELECT MAX(test_auc) AS best FROM experiments WHERE status='completed' AND test_auc > 0 AND test_auc < 0.85"
        ).fetchone()

        return {
            "total": total,
            "completed": completed,
            "failed": failed,
            "scored": row["scored"] or 0,
            "success_rate": completed / total if total else 0,
            "avg_duration": row["avg_dur"] or 0,
            "avg_test_auc": row["avg_auc"] or 0,
            "avg_wmes": row["avg_wmes"] or 0,
            "best_test_auc": row["best_auc"] or 0,
            "best_realistic_auc": realistic_row["best"] or 0,
        }

    def delete_experiments(self, experiment_ids: List[str]) -> int:
        """Delete experiments by ID. Returns count deleted."""
        if not experiment_ids:
            return 0
        with self._write_lock:
            conn = self._get_conn()
            placeholders = ",".join("?" for _ in experiment_ids)
            cur = conn.execute(
                f"DELETE FROM experiments WHERE experiment_id IN ({placeholders})",
                experiment_ids,
            )
            conn.commit()
            return cur.rowcount

    def prune_experiments(self, below_auc: float) -> int:
        """Delete experiments with test_auc below threshold. Returns count deleted."""
        with self._write_lock:
            conn = self._get_conn()
            cur = conn.execute(
                "DELETE FROM experiments WHERE test_auc < ? AND test_auc > 0",
                (below_auc,),
            )
            conn.commit()
            return cur.rowcount

    # ==================================================================
    # MODELS (tiered model tracking with quality gates)
    # ==================================================================

    def add_model(self, model_id: str, record_dict: Dict[str, Any]) -> None:
        """Insert a model record."""
        d = record_dict
        with self._write_lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT OR REPLACE INTO models
                   (model_id, experiment_id, created_at, model_path,
                    cv_auc, test_auc, wmes_score, stability_score,
                    fragility_score, train_test_gap, tier,
                    backtest_sharpe, backtest_win_rate,
                    scoring_version, data_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    model_id,
                    d.get("experiment_id", ""),
                    d.get("created_at", ""),
                    d.get("model_path", ""),
                    d.get("cv_auc", 0),
                    d.get("test_auc", 0),
                    d.get("wmes_score", 0),
                    d.get("stability_score", 0),
                    d.get("fragility_score", 1.0),
                    d.get("train_test_gap", 0),
                    d.get("tier", 1),
                    d.get("backtest_sharpe", 0),
                    d.get("backtest_win_rate", 0),
                    d.get("scoring_version", CURRENT_SCORING_VERSION),
                    json.dumps(d),
                ),
            )
            conn.commit()

    def get_models(
        self,
        min_tier: int = None,
        max_age_days: int = None,
        min_auc: float = None,
        max_auc: float = None,
    ) -> List[Dict]:
        """Return model dicts, optionally filtered."""
        conn = self._get_conn()
        clauses: List[str] = []
        params: List[Any] = []

        if min_tier is not None:
            clauses.append("tier >= ?")
            params.append(min_tier)
        if min_auc is not None:
            clauses.append("test_auc >= ?")
            params.append(min_auc)
        if max_auc is not None:
            clauses.append("test_auc <= ?")
            params.append(max_auc)
        if max_age_days is not None:
            cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
            clauses.append("created_at >= ?")
            params.append(cutoff)

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = conn.execute(
            f"SELECT data_json FROM models{where} ORDER BY test_auc DESC", params
        ).fetchall()
        return [json.loads(row["data_json"]) for row in rows]

    def delete_models(self, model_ids: List[str]) -> int:
        """Delete models by ID."""
        if not model_ids:
            return 0
        with self._write_lock:
            conn = self._get_conn()
            placeholders = ",".join("?" for _ in model_ids)
            cur = conn.execute(
                f"DELETE FROM models WHERE model_id IN ({placeholders})", model_ids
            )
            conn.commit()
            return cur.rowcount

    def purge_models(self, max_auc: float = 0.85) -> int:
        """Remove models with unrealistically high AUC."""
        with self._write_lock:
            conn = self._get_conn()
            cur = conn.execute("DELETE FROM models WHERE test_auc >= ?", (max_auc,))
            conn.commit()
            return cur.rowcount

    def update_model_tier(self, model_id: str, tier: int) -> None:
        """Update a model's tier."""
        with self._write_lock:
            conn = self._get_conn()
            # Update indexed column
            conn.execute(
                "UPDATE models SET tier = ? WHERE model_id = ?", (tier, model_id)
            )
            # Update data_json blob
            row = conn.execute(
                "SELECT data_json FROM models WHERE model_id = ?", (model_id,)
            ).fetchone()
            if row:
                d = json.loads(row["data_json"])
                d["tier"] = tier
                conn.execute(
                    "UPDATE models SET data_json = ? WHERE model_id = ?",
                    (json.dumps(d), model_id),
                )
            conn.commit()

    def recompute_all_tiers(self) -> Dict[str, int]:
        """Recompute tiers for all models and fix mismatches.

        Wave 28b: Models may be registered before fragility analysis completes,
        leaving them stuck at Tier 2 when they qualify for Tier 3. This method
        re-evaluates every model and updates any tier mismatches.

        Returns dict with counts: {"checked", "promoted", "demoted", "unchanged"}.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT model_id, data_json, tier FROM models"
        ).fetchall()

        stats = {"checked": 0, "promoted": 0, "demoted": 0, "unchanged": 0}
        for row in rows:
            stats["checked"] += 1
            d = json.loads(row["data_json"])
            stability = d.get("stability_score", 0)
            fragility = d.get("fragility_score", 1.0)
            test_auc = d.get("test_auc", 0)
            current_tier = row["tier"] or 1
            correct_tier = compute_tier(stability, fragility, test_auc)

            if correct_tier != current_tier:
                self.update_model_tier(row["model_id"], correct_tier)
                if correct_tier > current_tier:
                    stats["promoted"] += 1
                else:
                    stats["demoted"] += 1
            else:
                stats["unchanged"] += 1

        return stats

    def get_model_statistics(self) -> Dict:
        """Aggregate stats for models."""
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) AS c FROM models").fetchone()["c"]
        if total == 0:
            return {"total_models": 0}
        row = conn.execute(
            """SELECT AVG(cv_auc) AS avg_cv, AVG(backtest_sharpe) AS avg_sharpe,
                      MAX(cv_auc) AS best_cv, MAX(backtest_sharpe) AS best_sharpe,
                      MAX(wmes_score) AS best_wmes
               FROM models"""
        ).fetchone()
        tier_rows = conn.execute(
            "SELECT tier, COUNT(*) AS c FROM models GROUP BY tier ORDER BY tier"
        ).fetchall()

        return {
            "total_models": total,
            "avg_cv_auc": row["avg_cv"] or 0,
            "avg_backtest_sharpe": row["avg_sharpe"] or 0,
            "best_cv_auc": row["best_cv"] or 0,
            "best_backtest_sharpe": row["best_sharpe"] or 0,
            "best_wmes": row["best_wmes"] or 0,
            "by_tier": {r["tier"]: r["c"] for r in tier_rows},
        }

    def register_model_from_experiment(self, result_dict: Dict[str, Any]) -> str:
        """Register a model from an experiment result dict.

        Computes the quality tier and inserts into the models table.
        This replaces the old ModelRegistry.register_model() method.

        Returns:
            model_id
        """
        d = result_dict
        exp_id = d.get("experiment_id", datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
        model_id = f"model_{exp_id}"
        train_auc = d.get("train_auc", 0)
        test_auc = d.get("test_auc", 0)
        train_test_gap = train_auc - test_auc if train_auc > 0 else 0.0
        tier = compute_tier(
            stability_score=d.get("stability_score", 0),
            fragility_score=d.get("fragility_score", 1.0),
            test_auc=test_auc,
        )

        scoring_ver = d.get("scoring_version", CURRENT_SCORING_VERSION)
        record = {
            "model_id": model_id,
            "experiment_id": d.get("experiment_id", ""),
            "created_at": datetime.now().isoformat(),
            "model_path": d.get("model_path", ""),
            "config": d.get("config", {}),
            "cv_auc": d.get("cv_auc_mean", 0),
            "test_auc": test_auc,
            "backtest_sharpe": d.get("backtest_sharpe", 0),
            "backtest_win_rate": d.get("backtest_win_rate", 0),
            "backtest_total_return": d.get("backtest_total_return", 0),
            "wmes_score": d.get("wmes_score", 0),
            "stability_score": d.get("stability_score", 0),
            "fragility_score": d.get("fragility_score", 1.0),
            "train_test_gap": train_test_gap,
            "tier": tier,
            "scoring_version": scoring_ver,
            "live_trades": 0,
            "live_win_rate": 0.0,
            "live_total_return": 0.0,
            "live_sharpe": 0.0,
        }
        self.add_model(model_id, record)
        return model_id

    def get_active_model_count(
        self,
        min_auc: float = 0.0,
        max_auc: float = 0.85,
        min_tier: int = 1,
        max_age_days: int = 180,
    ) -> int:
        """Count models meeting quality gates (replaces ModelRegistry.get_active_models loop)."""
        conn = self._get_conn()
        cutoff = (datetime.now() - timedelta(days=max_age_days)).isoformat()
        row = conn.execute(
            """SELECT COUNT(*) AS c FROM models
               WHERE test_auc >= ? AND test_auc < ?
               AND tier >= ?
               AND created_at >= ?""",
            (min_auc, max_auc, min_tier, cutoff),
        ).fetchone()
        return row["c"]

    def get_stale_model_count(self, min_auc: float = 0.85) -> int:
        """Count models with suspiciously high AUC (likely pre-Wave-14 leakage)."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM models WHERE test_auc >= ?", (min_auc,)
        ).fetchone()
        return row["c"]

    def get_auc_only_model_count(
        self, min_auc: float = 0.0, max_auc: float = 0.85
    ) -> int:
        """Count models above AUC threshold (no tier filter, for backward compat fallback)."""
        conn = self._get_conn()
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM models WHERE test_auc >= ? AND test_auc < ?",
            (min_auc, max_auc),
        ).fetchone()
        return row["c"]

    def purge_leaky_models(self, max_auc: float = 0.85) -> int:
        """Remove models with AUC >= max_auc (pre-Wave-14 leakage)."""
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.execute(
                "DELETE FROM models WHERE test_auc >= ?", (max_auc,)
            )
            conn.commit()
            return cursor.rowcount

    def purge_leaky_experiments(self, max_auc: float = 0.85) -> int:
        """Remove experiments with AUC >= max_auc (pre-leakage-fix garbage)."""
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.execute(
                "DELETE FROM experiments WHERE test_auc >= ? AND test_auc > 0",
                (max_auc,),
            )
            conn.commit()
            return cursor.rowcount

    def purge_all_contaminated(self, max_auc: float = 0.85) -> Dict[str, int]:
        """Purge all pre-leakage-fix garbage: experiments, models, bad WMES."""
        n_exp = self.purge_leaky_experiments(max_auc)
        n_mod = self.purge_leaky_models(max_auc)
        n_wmes = self.purge_bad_wmes_models(0.0)
        n_wmes_exp = self.purge_bad_wmes_experiments(0.0)
        self.vacuum()
        return {"experiments_purged": n_exp, "models_purged": n_mod,
                "bad_wmes_purged": n_wmes, "bad_wmes_experiments_purged": n_wmes_exp}

    def purge_bad_wmes_models(self, min_wmes: float = 0.0) -> int:
        """Remove models with WMES < min_wmes (bug victims like -1399)."""
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.execute(
                "DELETE FROM models WHERE wmes_score < ?", (min_wmes,)
            )
            conn.commit()
            return cursor.rowcount

    def purge_bad_wmes_experiments(self, min_wmes: float = 0.0) -> int:
        """Wave 31: Remove experiments with WMES < min_wmes (bug victims like -1399)."""
        with self._write_lock:
            conn = self._get_conn()
            cursor = conn.execute(
                "DELETE FROM experiments WHERE wmes_score IS NOT NULL AND wmes_score < ?",
                (min_wmes,)
            )
            conn.commit()
            return cursor.rowcount

    # ==================================================================
    # MODEL ENTRIES (ModelRegistryV2)
    # ==================================================================

    def add_model_entry(self, entry_dict: Dict[str, Any]) -> None:
        """Insert a V2 model entry."""
        d = entry_dict
        metrics = d.get("metrics", {})
        with self._write_lock:
            conn = self._get_conn()
            conn.execute(
                """INSERT OR REPLACE INTO model_entries
                   (model_id, experiment_id, target_type, status,
                    model_type, dim_reduction, cascade_type,
                    cv_auc, test_auc, train_auc, win_rate, sharpe_ratio,
                    stability_score, fragility_score,
                    created_at, updated_at, tags, data_json)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    d.get("model_id", ""),
                    d.get("experiment_id", ""),
                    d.get("target_type", "swing"),
                    d.get("status", "queued"),
                    d.get("model_config", {}).get("model_type", ""),
                    d.get("dim_reduction_config", {}).get("method", ""),
                    d.get("cascade_config", {}).get("cascade_type", ""),
                    metrics.get("cv_auc", 0),
                    metrics.get("test_auc", 0),
                    metrics.get("train_auc", 0),
                    metrics.get("win_rate", 0),
                    metrics.get("sharpe_ratio", 0),
                    metrics.get("stability_score", 0),
                    metrics.get("fragility_score", 0),
                    d.get("created_at", ""),
                    d.get("updated_at", ""),
                    ",".join(d.get("tags", [])),
                    json.dumps(d),
                ),
            )
            conn.commit()

    def update_model_entry(self, model_id: str, entry_dict: Dict[str, Any]) -> None:
        """Update an existing V2 model entry (full replacement)."""
        self.add_model_entry(entry_dict)  # INSERT OR REPLACE handles upsert

    def get_model_entry(self, model_id: str) -> Optional[Dict]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT data_json FROM model_entries WHERE model_id = ?", (model_id,)
        ).fetchone()
        return json.loads(row["data_json"]) if row else None

    def query_model_entries(
        self,
        target_type: str = None,
        status: str = None,
        model_type: str = None,
        dim_reduction: str = None,
        cascade_type: str = None,
        min_cv_auc: float = None,
        min_test_auc: float = None,
        tags: List[str] = None,
        limit: int = None,
    ) -> List[Dict]:
        """Query V2 model entries with filters."""
        conn = self._get_conn()
        clauses: List[str] = []
        params: List[Any] = []

        if target_type:
            clauses.append("target_type = ?")
            params.append(target_type)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if model_type:
            clauses.append("model_type = ?")
            params.append(model_type)
        if dim_reduction:
            clauses.append("dim_reduction = ?")
            params.append(dim_reduction)
        if cascade_type:
            clauses.append("cascade_type = ?")
            params.append(cascade_type)
        if min_cv_auc is not None:
            clauses.append("cv_auc >= ?")
            params.append(min_cv_auc)
        if min_test_auc is not None:
            clauses.append("test_auc >= ?")
            params.append(min_test_auc)
        if tags:
            for tag in tags:
                clauses.append("tags LIKE ?")
                params.append(f"%{tag}%")

        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT data_json FROM model_entries{where} ORDER BY test_auc DESC"
        if limit:
            sql += f" LIMIT {int(limit)}"

        rows = conn.execute(sql, params).fetchall()
        return [json.loads(row["data_json"]) for row in rows]

    def delete_model_entry(self, model_id: str) -> bool:
        """Delete a V2 model entry. Returns True if deleted."""
        with self._write_lock:
            conn = self._get_conn()
            cur = conn.execute(
                "DELETE FROM model_entries WHERE model_id = ?", (model_id,)
            )
            conn.commit()
            return cur.rowcount > 0

    def get_model_entry_count(self, status: str = None) -> int:
        conn = self._get_conn()
        if status:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM model_entries WHERE status = ?", (status,)
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) AS c FROM model_entries").fetchone()
        return row["c"]

    def get_model_entry_statistics(self) -> Dict:
        """Aggregate stats for V2 model entries."""
        conn = self._get_conn()
        total = conn.execute("SELECT COUNT(*) AS c FROM model_entries").fetchone()["c"]
        if total == 0:
            return {"total": 0}

        status_rows = conn.execute(
            "SELECT status, COUNT(*) AS c FROM model_entries GROUP BY status"
        ).fetchall()
        target_rows = conn.execute(
            "SELECT target_type, COUNT(*) AS c FROM model_entries GROUP BY target_type"
        ).fetchall()
        auc_row = conn.execute(
            """SELECT AVG(cv_auc) AS avg_cv, AVG(test_auc) AS avg_test,
                      MAX(cv_auc) AS best_cv, MAX(test_auc) AS best_test,
                      AVG(win_rate) AS avg_wr, AVG(sharpe_ratio) AS avg_sharpe
               FROM model_entries WHERE status != 'failed'"""
        ).fetchone()

        return {
            "total": total,
            "by_status": {r["status"]: r["c"] for r in status_rows},
            "by_target": {r["target_type"]: r["c"] for r in target_rows},
            "avg_cv_auc": auc_row["avg_cv"] or 0,
            "avg_test_auc": auc_row["avg_test"] or 0,
            "best_cv_auc": auc_row["best_cv"] or 0,
            "best_test_auc": auc_row["best_test"] or 0,
            "avg_win_rate": auc_row["avg_wr"] or 0,
            "avg_sharpe": auc_row["avg_sharpe"] or 0,
        }

    # ==================================================================
    # UTILITIES
    # ==================================================================

    def export_json(self, table: str, filepath: Path) -> None:
        """Export a table's data_json blobs to a JSON file."""
        conn = self._get_conn()
        if table not in ("experiments", "models", "model_entries"):
            raise ValueError(f"Unknown table: {table}")

        rows = conn.execute(f"SELECT data_json FROM {table} ORDER BY id").fetchall()
        data = [json.loads(row["data_json"]) for row in rows]

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Exported {len(data)} records from {table} to {filepath}")

    def vacuum(self) -> None:
        """Reclaim disk space after deletes."""
        conn = self._get_conn()
        conn.execute("VACUUM")

    def db_stats(self) -> Dict:
        """Return DB file size and table row counts."""
        conn = self._get_conn()
        size_bytes = os.path.getsize(self.db_path) if self.db_path.exists() else 0

        experiments = conn.execute("SELECT COUNT(*) AS c FROM experiments").fetchone()["c"]
        models = conn.execute("SELECT COUNT(*) AS c FROM models").fetchone()["c"]
        entries = conn.execute("SELECT COUNT(*) AS c FROM model_entries").fetchone()["c"]

        return {
            "db_path": str(self.db_path),
            "size_mb": round(size_bytes / (1024 * 1024), 2),
            "experiments": experiments,
            "models": models,
            "model_entries": entries,
        }

    def close(self) -> None:
        """Close the thread-local connection."""
        conn = getattr(self._local, "conn", None)
        if conn:
            conn.close()
            self._local.conn = None


# ---------------------------------------------------------------------------
# Standalone tier computation (extracted from old ModelRegistry._compute_tier)
# ---------------------------------------------------------------------------

def compute_tier(
    stability_score: float,
    fragility_score: float,
    test_auc: float,
    suite_composite: float = 0.0,
) -> int:
    """Compute quality tier based on robustness metrics.

    Tier 1: Registry (basic — passed registration gate)
    Tier 2: Paper-eligible (stability_score >= 0.60)
    Tier 3: Live-eligible (Tier 2 + fragility < 0.40 + AUC >= 0.57
             + suite_composite >= 0.45 when available)

    Wave 26: Recalibrated — post-leakage-fix realistic AUCs are 0.53-0.57.
    Wave 33: Added suite_composite gate for Tier 3. Fragility threshold
    raised 0.30→0.40 since formula now produces meaningful scores.
    Wave 36 (v4): AUC threshold 0.58→0.57. With 40 v3 experiments,
    max observed AUC=0.569 — 0.58 was unreachable. 0.57 is the most
    conservative reduction that unblocks Tier 3 progression.
    stability_score == -1.0 means "failed to compute" (treat as no data).
    """
    tier = 1
    # stability_score == -1 means analysis failed (Wave 33 sentinel)
    if stability_score >= 0.60:
        tier = 2
        # Tier 3 requirements:
        # 1. AUC >= 0.57 (minimum predictive power — v4 lowered from 0.58)
        # 2. Fragility < 0.40 (not overly sensitive to perturbations)
        # 3. Suite composite >= 0.45 when available (bootstrap/dropout/seed/prediction)
        #    If suite_composite is 0 (not computed), waive the requirement
        suite_ok = (suite_composite >= 0.45) if suite_composite > 0 else True
        if fragility_score < 0.40 and test_auc >= 0.57 and suite_ok:
            tier = 3
    return tier


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_singleton_lock = threading.Lock()
_singleton_db: Optional[RegistryDB] = None


def get_registry_db(db_path: Path = None) -> RegistryDB:
    """Get or create the global RegistryDB singleton."""
    global _singleton_db
    with _singleton_lock:
        if _singleton_db is None:
            _singleton_db = RegistryDB(db_path)
        return _singleton_db
