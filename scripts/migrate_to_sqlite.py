"""
Migrate existing JSON registry files to SQLite.

Reads:
  - experiments/experiment_history.json  → experiments table
  - experiments/model_registry.json      → models table
  - models/registry_v2/registry.json     → model_entries table

Usage:
    python scripts/migrate_to_sqlite.py [--db-path data/giga_trader.db]
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.core.registry_db import RegistryDB


def _config_hash_from_dict(config: dict) -> str:
    """Compute config hash from a serialized ExperimentConfig dict."""
    try:
        dim = config.get("dim_reduction", {})
        model = config.get("model", {})
        feat = config.get("feature_engineering", {})
        cv = config.get("cross_validation", {})
        key_parts = [
            str(config.get("experiment_type", "")),
            str(dim.get("method", "")),
            str(dim.get("target_dimensions", 0)),
            str(dim.get("feature_selection_method", "")),
            str(dim.get("mi_n_features", 0)),
            str(dim.get("kpca_n_components", 0)),
            str(dim.get("kpca_kernel", "")),
            str(dim.get("ica_n_components", 0)),
            str(dim.get("umap_n_components", 0)),
            str(model.get("model_type", "")),
            str(model.get("regularization", "")),
            str(model.get("l2_C", 0)),
            str(model.get("gb_n_estimators", 0)),
            str(model.get("gb_max_depth", 0)),
            str(model.get("gb_learning_rate", 0)),
            str(model.get("gb_min_samples_leaf", 0)),
            str(model.get("gb_subsample", 0)),
            str(feat.get("use_premarket_features", True)),
            str(feat.get("use_afterhours_features", True)),
            str(feat.get("use_pattern_recognition", True)),
            str(feat.get("swing_threshold", 0)),
            str(cv.get("n_cv_folds", 5)),
            str(cv.get("use_soft_targets", True)),
        ]
        return hashlib.md5("|".join(key_parts).encode()).hexdigest()[:12]
    except Exception:
        return ""


def migrate_experiments(db: RegistryDB, filepath: Path) -> int:
    """Migrate experiment_history.json → experiments table."""
    if not filepath.exists():
        print(f"  [SKIP] {filepath} not found")
        return 0

    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        print(f"  [SKIP] {filepath} is not a list")
        return 0

    count = 0
    for entry in data:
        config = entry.get("config", {})
        config_hash = _config_hash_from_dict(config) if isinstance(config, dict) else ""
        db.add_experiment(entry, config_hash)
        count += 1

    return count


def migrate_models(db: RegistryDB, filepath: Path) -> int:
    """Migrate model_registry.json → models table."""
    if not filepath.exists():
        print(f"  [SKIP] {filepath} not found")
        return 0

    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        print(f"  [SKIP] {filepath} is not a dict")
        return 0

    count = 0
    for model_id, record in data.items():
        record["model_id"] = model_id
        db.add_model(model_id, record)
        count += 1

    return count


def migrate_model_entries(db: RegistryDB, filepath: Path) -> int:
    """Migrate registry_v2/registry.json → model_entries table."""
    if not filepath.exists():
        print(f"  [SKIP] {filepath} not found")
        return 0

    with open(filepath, encoding="utf-8") as f:
        data = json.load(f)

    models = data.get("models", {})
    count = 0
    for model_id, entry in models.items():
        entry["model_id"] = model_id
        db.add_model_entry(entry)
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Migrate JSON registries to SQLite")
    parser.add_argument("--db-path", type=str, default=None, help="SQLite DB path")
    args = parser.parse_args()

    db_path = Path(args.db_path) if args.db_path else None
    db = RegistryDB(db_path)

    print("=" * 60)
    print("MIGRATING JSON REGISTRIES TO SQLITE")
    print("=" * 60)
    print(f"Database: {db.db_path}")
    print()

    # 1. Experiment history
    exp_file = project_root / "experiments" / "experiment_history.json"
    print(f"[1/3] Migrating experiments from {exp_file.name}...")
    n_exp = migrate_experiments(db, exp_file)
    print(f"  Imported {n_exp} experiments")

    # 2. Legacy model registry
    model_file = project_root / "experiments" / "model_registry.json"
    print(f"[2/3] Migrating legacy models from {model_file.name}...")
    n_models = migrate_models(db, model_file)
    print(f"  Imported {n_models} models")

    # 3. ModelRegistryV2
    v2_file = project_root / "models" / "registry_v2" / "registry.json"
    print(f"[3/3] Migrating V2 entries from {v2_file.name}...")
    n_entries = migrate_model_entries(db, v2_file)
    print(f"  Imported {n_entries} model entries")

    # Validation
    print()
    print("=" * 60)
    print("VALIDATION")
    print("=" * 60)
    stats = db.db_stats()
    print(f"  DB size:        {stats['size_mb']:.2f} MB")
    print(f"  Experiments:    {stats['experiments']} (expected {n_exp})")
    print(f"  Models:         {stats['models']} (expected {n_models})")
    print(f"  Model entries:  {stats['model_entries']} (expected {n_entries})")

    ok = True
    if stats["experiments"] != n_exp:
        print(f"  [WARN] Experiment count mismatch!")
        ok = False
    if stats["models"] != n_models:
        print(f"  [WARN] Model count mismatch!")
        ok = False
    if stats["model_entries"] != n_entries:
        print(f"  [WARN] Model entry count mismatch!")
        ok = False

    if ok:
        print()
        print("  [OK] Migration successful!")
    else:
        print()
        print("  [WARN] Some counts don't match. Check for duplicates.")

    print("=" * 60)


if __name__ == "__main__":
    main()
