#!/usr/bin/env python3
"""
Reset the model registry, experiments, and artifacts for a fresh training campaign.

Use this after implementing scoring/gating changes that invalidate old models.
Archives old data to a timestamped backup directory instead of deleting.

Usage:
    python scripts/reset_registry.py              # Dry run (preview only)
    python scripts/reset_registry.py --execute     # Actually perform the reset
    python scripts/reset_registry.py --execute --purge-logs   # Also clear old logs
"""

import argparse
import json
import os
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def get_size_mb(path: Path) -> float:
    """Get total size of a file or directory in MB."""
    if path.is_file():
        return path.stat().st_size / (1024 * 1024)
    if path.is_dir():
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
        return total / (1024 * 1024)
    return 0.0


def count_files(path: Path, pattern: str = "*") -> int:
    """Count files matching pattern in directory."""
    if not path.is_dir():
        return 0
    return sum(1 for _ in path.glob(pattern))


def snapshot_db_stats(db_path: Path) -> dict:
    """Capture current DB state before reset."""
    if not db_path.is_file():
        return {"exists": False}
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    stats = {"exists": True}
    for table in ("experiments", "models", "model_entries"):
        try:
            row = conn.execute(f"SELECT COUNT(*) AS c FROM {table}").fetchone()
            stats[table] = row["c"]
        except sqlite3.OperationalError:
            stats[table] = 0
    conn.close()
    return stats


def reset_sqlite(db_path: Path, backup_dir: Path, dry_run: bool) -> dict:
    """Clear all SQLite tables and vacuum. Backs up the DB file first."""
    stats = {"backed_up": False, "tables_cleared": []}

    if not db_path.is_file():
        print("  [SKIP] No SQLite database found")
        return stats

    # Backup the entire DB file
    backup_db = backup_dir / "giga_trader.db"
    if dry_run:
        print(f"  [DRY] Would backup {db_path} -> {backup_db}")
        print(f"  [DRY] Would clear tables: experiments, models, model_entries")
        print(f"  [DRY] Would reset sqlite_sequence")
        print(f"  [DRY] Would VACUUM database")
        return stats

    backup_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(db_path), str(backup_db))
    stats["backed_up"] = True
    print(f"  [OK] Backed up database -> {backup_db.name}")

    # Also backup WAL and SHM if present
    for suffix in ("-wal", "-shm"):
        wal = db_path.parent / (db_path.name + suffix)
        if wal.is_file():
            shutil.copy2(str(wal), str(backup_dir / (db_path.name + suffix)))

    # Clear tables
    conn = sqlite3.connect(str(db_path))
    for table in ("experiments", "models", "model_entries"):
        try:
            conn.execute(f"DELETE FROM {table}")
            stats["tables_cleared"].append(table)
            print(f"  [OK] Cleared table: {table}")
        except sqlite3.OperationalError as e:
            print(f"  [WARN] Could not clear {table}: {e}")

    # Reset autoincrement counters
    try:
        conn.execute("DELETE FROM sqlite_sequence")
        print("  [OK] Reset autoincrement counters")
    except sqlite3.OperationalError:
        pass

    conn.commit()
    conn.close()

    # VACUUM must run outside any transaction — use a fresh autocommit connection
    vac_conn = sqlite3.connect(str(db_path), isolation_level=None)
    vac_conn.execute("VACUUM")
    vac_conn.close()
    print("  [OK] Database vacuumed")
    return stats


def archive_directory(src: Path, dest: Path, pattern: str, dry_run: bool) -> dict:
    """Move files matching pattern from src to dest."""
    stats = {"files": 0, "size_mb": 0.0}
    if not src.is_dir():
        print(f"  [SKIP] Directory not found: {src}")
        return stats

    files = list(src.glob(pattern))
    if not files:
        print(f"  [SKIP] No files matching {pattern} in {src.name}/")
        return stats

    total_size = sum(f.stat().st_size for f in files) / (1024 * 1024)
    stats["files"] = len(files)
    stats["size_mb"] = total_size

    if dry_run:
        print(f"  [DRY] Would archive {len(files)} files ({total_size:.1f} MB) from {src.name}/")
        return stats

    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.move(str(f), str(dest / f.name))
    print(f"  [OK] Archived {len(files)} files ({total_size:.1f} MB) -> {dest.relative_to(PROJECT_ROOT)}")
    return stats


def clear_json_registries(backup_dir: Path, dry_run: bool) -> list:
    """Archive stale JSON registry files."""
    cleared = []
    json_targets = [
        PROJECT_ROOT / "models" / "registry_v2" / "registry.json",
        PROJECT_ROOT / "experiments" / "experiment_history.json",
        PROJECT_ROOT / "experiments" / "model_registry.json",
        PROJECT_ROOT / "experiments" / "model_registry.json.backup",
        PROJECT_ROOT / "experiments" / "temporal_model_registry.json",
        PROJECT_ROOT / "experiments" / "experiment_registry.db",
    ]

    for path in json_targets:
        if not path.is_file():
            continue
        size = path.stat().st_size
        if size <= 2:  # Empty or "{}"
            if dry_run:
                print(f"  [DRY] Would delete empty file: {path.name}")
            else:
                path.unlink()
                print(f"  [OK] Deleted empty file: {path.name}")
            cleared.append(str(path.name))
            continue

        dest = backup_dir / "json" / path.name
        if dry_run:
            print(f"  [DRY] Would archive {path.name} ({size / 1024:.1f} KB)")
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(dest))
            print(f"  [OK] Archived {path.name} ({size / 1024:.1f} KB)")
        cleared.append(str(path.name))

    if not cleared:
        print("  [SKIP] No JSON registry files found")
    return cleared


def archive_production_models(backup_dir: Path, dry_run: bool) -> int:
    """Archive production models (keep as reference, not delete)."""
    prod_dir = PROJECT_ROOT / "models" / "production"
    if not prod_dir.is_dir():
        print("  [SKIP] No production models directory")
        return 0

    files = [f for f in prod_dir.iterdir() if f.is_file() and f.suffix == ".joblib"]
    if not files:
        print("  [SKIP] No production model files")
        return 0

    total_size = sum(f.stat().st_size for f in files) / (1024 * 1024)
    if dry_run:
        print(f"  [DRY] Would archive {len(files)} production models ({total_size:.1f} MB)")
        return len(files)

    dest = backup_dir / "production"
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.move(str(f), str(dest / f.name))
    print(f"  [OK] Archived {len(files)} production models ({total_size:.1f} MB)")
    return len(files)


def clear_thick_weave_checkpoints(backup_dir: Path, dry_run: bool) -> int:
    """Archive thick weave search checkpoints."""
    cp_dir = PROJECT_ROOT / "models" / "thick_weave_checkpoints"
    if not cp_dir.is_dir():
        return 0
    files = list(cp_dir.glob("*.json"))
    if not files:
        return 0

    if dry_run:
        print(f"  [DRY] Would archive {len(files)} thick weave checkpoints")
        return len(files)

    dest = backup_dir / "thick_weave_checkpoints"
    dest.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.move(str(f), str(dest / f.name))
    print(f"  [OK] Archived {len(files)} thick weave checkpoints")
    return len(files)


def clear_reports(backup_dir: Path, dry_run: bool) -> int:
    """Archive backtest and thick weave reports."""
    reports_dir = PROJECT_ROOT / "reports"
    if not reports_dir.is_dir():
        return 0

    files = list(reports_dir.rglob("*.json"))
    if not files:
        return 0

    if dry_run:
        print(f"  [DRY] Would archive {len(files)} report files")
        return len(files)

    dest = backup_dir / "reports"
    for f in files:
        rel = f.relative_to(reports_dir)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(f), str(target))
    print(f"  [OK] Archived {len(files)} report files")
    return len(files)


def purge_old_logs(backup_dir: Path, dry_run: bool) -> dict:
    """Archive log files (optional)."""
    logs_dir = PROJECT_ROOT / "logs"
    if not logs_dir.is_dir():
        return {"files": 0, "size_mb": 0.0}

    log_files = [f for f in logs_dir.iterdir() if f.is_file()]
    if not log_files:
        return {"files": 0, "size_mb": 0.0}

    total_size = sum(f.stat().st_size for f in log_files) / (1024 * 1024)

    # Keep current-day logs, archive the rest
    today = datetime.now().strftime("%Y%m%d")
    to_archive = [f for f in log_files if today not in f.name]
    archive_size = sum(f.stat().st_size for f in to_archive) / (1024 * 1024)

    if dry_run:
        print(f"  [DRY] Would archive {len(to_archive)}/{len(log_files)} log files ({archive_size:.1f} MB)")
        print(f"  [DRY] Keeping {len(log_files) - len(to_archive)} files from today")
        return {"files": len(to_archive), "size_mb": archive_size}

    if to_archive:
        dest = backup_dir / "logs"
        dest.mkdir(parents=True, exist_ok=True)
        for f in to_archive:
            shutil.move(str(f), str(dest / f.name))
        print(f"  [OK] Archived {len(to_archive)} log files ({archive_size:.1f} MB)")

    return {"files": len(to_archive), "size_mb": archive_size}


def reset_feature_research(backup_dir: Path, dry_run: bool) -> int:
    """Archive feature research candidates (they need re-graduation under new criteria)."""
    targets = [
        PROJECT_ROOT / "data" / "feature_candidates.json",
        PROJECT_ROOT / "data" / "graduated_features.json",
    ]
    archived = 0
    for path in targets:
        if not path.is_file():
            continue
        if dry_run:
            print(f"  [DRY] Would archive {path.name}")
        else:
            dest = backup_dir / "feature_research" / path.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(dest))
            print(f"  [OK] Archived {path.name}")
        archived += 1

    if not archived:
        print("  [SKIP] No feature research files found")
    return archived


def reset_campaign_data(backup_dir: Path, dry_run: bool) -> int:
    """Archive campaign directories."""
    campaigns_dir = PROJECT_ROOT / "campaigns"
    if not campaigns_dir.is_dir():
        print("  [SKIP] No campaigns directory")
        return 0

    subdirs = [d for d in campaigns_dir.iterdir() if d.is_dir()]
    if not subdirs:
        print("  [SKIP] No campaign data")
        return 0

    if dry_run:
        print(f"  [DRY] Would archive {len(subdirs)} campaign(s)")
        return len(subdirs)

    dest = backup_dir / "campaigns"
    dest.mkdir(parents=True, exist_ok=True)
    for d in subdirs:
        shutil.move(str(d), str(dest / d.name))
    print(f"  [OK] Archived {len(subdirs)} campaign(s)")
    return len(subdirs)


def print_summary(before_stats: dict, backup_dir: Path, dry_run: bool):
    """Print a summary of what was (or would be) done."""
    mode = "DRY RUN" if dry_run else "COMPLETED"
    print(f"\n{'=' * 60}")
    print(f"  RESET {mode}")
    print(f"{'=' * 60}")
    print(f"  Before: {before_stats.get('experiments', 0)} experiments, "
          f"{before_stats.get('models', 0)} models, "
          f"{before_stats.get('model_entries', 0)} model entries")
    print(f"  Backup: {backup_dir.relative_to(PROJECT_ROOT)}")
    if not dry_run:
        print(f"\n  The system is ready for a fresh training campaign.")
        print(f"  Start with: python scripts/launch_campaign.py")
    else:
        print(f"\n  Run with --execute to perform the reset.")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Reset model registry for fresh training campaign")
    parser.add_argument("--execute", action="store_true", help="Actually perform the reset (default is dry run)")
    parser.add_argument("--purge-logs", action="store_true", help="Also archive old log files")
    args = parser.parse_args()

    dry_run = not args.execute
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = PROJECT_ROOT / "backups" / f"pre_reset_{timestamp}"

    print(f"\n{'=' * 60}")
    print(f"  GIGA TRADER - REGISTRY RESET")
    print(f"  Mode: {'DRY RUN (preview only)' if dry_run else 'EXECUTE'}")
    print(f"  Backup: backups/pre_reset_{timestamp}/")
    print(f"{'=' * 60}")

    # Snapshot current state
    db_path = PROJECT_ROOT / "data" / "giga_trader.db"
    before_stats = snapshot_db_stats(db_path)

    if before_stats.get("exists"):
        print(f"\n  Current DB: {before_stats.get('experiments', 0)} experiments, "
              f"{before_stats.get('models', 0)} models, "
              f"{before_stats.get('model_entries', 0)} model entries")
    else:
        print("\n  No existing database found.")

    exp_count = count_files(PROJECT_ROOT / "models" / "experiments", "*.joblib")
    exp_size = get_size_mb(PROJECT_ROOT / "models" / "experiments")
    print(f"  Experiment artifacts: {exp_count} files ({exp_size:.1f} MB)")

    prod_count = count_files(PROJECT_ROOT / "models" / "production", "*.joblib")
    prod_size = get_size_mb(PROJECT_ROOT / "models" / "production")
    print(f"  Production models: {prod_count} files ({prod_size:.1f} MB)")

    # --- Step 1: SQLite ---
    print(f"\n[1/8] SQLite Database")
    reset_sqlite(db_path, backup_dir / "sqlite", dry_run)

    # --- Step 2: Experiment artifacts ---
    print(f"\n[2/8] Experiment Artifacts (joblib)")
    archive_directory(
        PROJECT_ROOT / "models" / "experiments",
        backup_dir / "experiments",
        "*.joblib",
        dry_run,
    )

    # --- Step 3: Production models ---
    print(f"\n[3/8] Production Models")
    archive_production_models(backup_dir, dry_run)

    # --- Step 4: JSON registries ---
    print(f"\n[4/8] JSON Registry Files")
    clear_json_registries(backup_dir, dry_run)

    # --- Step 5: Thick weave checkpoints ---
    print(f"\n[5/8] Thick Weave Checkpoints")
    clear_thick_weave_checkpoints(backup_dir, dry_run)

    # --- Step 6: Reports ---
    print(f"\n[6/8] Backtest & Analysis Reports")
    clear_reports(backup_dir, dry_run)

    # --- Step 7: Feature research ---
    print(f"\n[7/8] Feature Research Candidates")
    reset_feature_research(backup_dir, dry_run)

    # --- Step 8: Campaigns ---
    print(f"\n[8/8] Campaign Data")
    reset_campaign_data(backup_dir, dry_run)

    # Optional: Logs
    if args.purge_logs:
        print(f"\n[EXTRA] Log Files")
        purge_old_logs(backup_dir, dry_run)

    print_summary(before_stats, backup_dir, dry_run)


if __name__ == "__main__":
    main()
