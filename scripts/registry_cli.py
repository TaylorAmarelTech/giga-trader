"""
Registry management CLI for Giga Trader.

Usage:
    python scripts/registry_cli.py experiments list [--status STATUS] [--min-auc N] [--limit N]
    python scripts/registry_cli.py experiments stats
    python scripts/registry_cli.py experiments prune --below-auc N [--dry-run]

    python scripts/registry_cli.py models list [--min-tier N] [--min-auc N]
    python scripts/registry_cli.py models stats
    python scripts/registry_cli.py models promote MODEL_ID --tier N
    python scripts/registry_cli.py models prune --below-auc N [--dry-run]

    python scripts/registry_cli.py v2 list [--status STATUS] [--target TYPE] [--min-auc N] [--limit N]
    python scripts/registry_cli.py v2 stats
    python scripts/registry_cli.py v2 delete MODEL_ID
    python scripts/registry_cli.py v2 compare ID1 ID2
    python scripts/registry_cli.py v2 export --output FILE

    python scripts/registry_cli.py db stats
    python scripts/registry_cli.py db vacuum
"""

import argparse
import json
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.core.registry_db import RegistryDB


def _fmt_float(v, decimals=4):
    if v is None:
        return "N/A"
    return f"{v:.{decimals}f}"


def _print_table(headers, rows, col_widths=None):
    """Print a simple aligned table."""
    if not rows:
        print("  (no results)")
        return

    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = len(str(h))
            for row in rows:
                val = row[i] if i < len(row) else ""
                max_w = max(max_w, len(str(val)))
            col_widths.append(min(max_w + 1, 50))

    fmt = "  ".join(f"{{:<{w}}}" for w in col_widths)
    print(fmt.format(*[str(h) for h in headers]))
    print(fmt.format(*["-" * w for w in col_widths]))
    for row in rows:
        vals = [str(v) if v is not None else "" for v in row]
        print(fmt.format(*vals[:len(headers)]))


# ========================================================================
# EXPERIMENTS
# ========================================================================

def cmd_experiments_list(db: RegistryDB, args):
    results = db.get_experiments(
        status=args.status,
        min_auc=args.min_auc,
        limit=args.limit,
    )
    headers = ["experiment_id", "status", "test_auc", "wmes", "duration_s", "model_path"]
    rows = []
    for d in results:
        rows.append([
            d.get("experiment_id", "")[:40],
            d.get("status", ""),
            _fmt_float(d.get("test_auc")),
            _fmt_float(d.get("wmes_score")),
            f"{d.get('duration_seconds', 0):.0f}",
            d.get("model_path", "")[:30] if d.get("model_path") else "",
        ])
    print(f"\nExperiments ({len(rows)} shown):")
    _print_table(headers, rows)


def cmd_experiments_stats(db: RegistryDB, args):
    stats = db.get_experiment_statistics()
    print("\nExperiment Statistics:")
    print(f"  Total:             {stats.get('total', 0)}")
    print(f"  Completed:         {stats.get('completed', 0)}")
    print(f"  Failed:            {stats.get('failed', 0)}")
    print(f"  Scored:            {stats.get('scored', 0)}")
    print(f"  Success rate:      {stats.get('success_rate', 0):.1%}")
    print(f"  Avg duration:      {stats.get('avg_duration', 0):.0f}s")
    print(f"  Avg test AUC:      {_fmt_float(stats.get('avg_test_auc'))}")
    print(f"  Avg WMES:          {_fmt_float(stats.get('avg_wmes'))}")
    print(f"  Best test AUC:     {_fmt_float(stats.get('best_test_auc'))}")
    print(f"  Best realistic:    {_fmt_float(stats.get('best_realistic_auc'))}")


def cmd_experiments_prune(db: RegistryDB, args):
    if args.below_auc is None:
        print("Error: --below-auc is required")
        return

    # Preview
    targets = db.get_experiments(max_auc=args.below_auc)
    # Filter to only those with test_auc > 0 (don't prune zero-auc)
    targets = [d for d in targets if d.get("test_auc", 0) > 0]
    print(f"\nFound {len(targets)} experiments with test_auc < {args.below_auc}")

    if args.dry_run:
        print("  (dry run — no changes made)")
        for d in targets[:10]:
            print(f"  - {d['experiment_id']}: AUC={_fmt_float(d.get('test_auc'))}")
        if len(targets) > 10:
            print(f"  ... and {len(targets) - 10} more")
    else:
        ids = [d["experiment_id"] for d in targets]
        deleted = db.delete_experiments(ids)
        print(f"  Deleted {deleted} experiments")


# ========================================================================
# LEGACY MODELS
# ========================================================================

def cmd_models_list(db: RegistryDB, args):
    results = db.get_models(
        min_tier=args.min_tier,
        min_auc=args.min_auc,
    )
    headers = ["model_id", "tier", "test_auc", "cv_auc", "wmes", "sharpe", "created_at"]
    rows = []
    for d in results:
        rows.append([
            d.get("model_id", "")[:30],
            d.get("tier", 1),
            _fmt_float(d.get("test_auc")),
            _fmt_float(d.get("cv_auc")),
            _fmt_float(d.get("wmes_score")),
            _fmt_float(d.get("backtest_sharpe")),
            d.get("created_at", "")[:19],
        ])
    print(f"\nModels ({len(rows)} shown):")
    _print_table(headers, rows)


def cmd_models_stats(db: RegistryDB, args):
    stats = db.get_model_statistics()
    print("\nModel Statistics:")
    print(f"  Total models:      {stats.get('total_models', 0)}")
    print(f"  Avg CV AUC:        {_fmt_float(stats.get('avg_cv_auc'))}")
    print(f"  Avg Sharpe:        {_fmt_float(stats.get('avg_backtest_sharpe'))}")
    print(f"  Best CV AUC:       {_fmt_float(stats.get('best_cv_auc'))}")
    print(f"  Best Sharpe:       {_fmt_float(stats.get('best_backtest_sharpe'))}")
    print(f"  Best WMES:         {_fmt_float(stats.get('best_wmes'))}")
    by_tier = stats.get("by_tier", {})
    if by_tier:
        print("  By tier:")
        for tier, count in sorted(by_tier.items()):
            print(f"    Tier {tier}: {count}")


def cmd_models_promote(db: RegistryDB, args):
    db.update_model_tier(args.model_id, args.tier)
    print(f"  Updated {args.model_id} to tier {args.tier}")


def cmd_models_prune(db: RegistryDB, args):
    if args.below_auc is None:
        print("Error: --below-auc is required")
        return

    targets = db.get_models(max_auc=args.below_auc)
    print(f"\nFound {len(targets)} models with test_auc < {args.below_auc}")

    if args.dry_run:
        print("  (dry run — no changes made)")
        for d in targets[:10]:
            print(f"  - {d['model_id']}: AUC={_fmt_float(d.get('test_auc'))}")
    else:
        ids = [d["model_id"] for d in targets]
        deleted = db.delete_models(ids)
        print(f"  Deleted {deleted} models")


# ========================================================================
# V2 MODEL ENTRIES
# ========================================================================

def cmd_v2_list(db: RegistryDB, args):
    results = db.query_model_entries(
        status=args.status,
        target_type=args.target,
        min_test_auc=args.min_auc,
        limit=args.limit,
    )
    headers = ["model_id", "status", "target", "test_auc", "cv_auc", "win_rate", "sharpe"]
    rows = []
    for d in results:
        m = d.get("metrics", {})
        rows.append([
            d.get("model_id", "")[:40],
            d.get("status", ""),
            d.get("target_type", ""),
            _fmt_float(m.get("test_auc")),
            _fmt_float(m.get("cv_auc")),
            _fmt_float(m.get("win_rate")),
            _fmt_float(m.get("sharpe_ratio")),
        ])
    print(f"\nV2 Model Entries ({len(rows)} shown):")
    _print_table(headers, rows)


def cmd_v2_stats(db: RegistryDB, args):
    stats = db.get_model_entry_statistics()
    print("\nV2 Model Entry Statistics:")
    print(f"  Total:             {stats.get('total', 0)}")
    by_status = stats.get("by_status", {})
    if by_status:
        print("  By status:")
        for status, count in sorted(by_status.items()):
            print(f"    {status}: {count}")
    by_target = stats.get("by_target", {})
    if by_target:
        print("  By target:")
        for target, count in sorted(by_target.items()):
            print(f"    {target}: {count}")
    print(f"  Avg CV AUC:        {_fmt_float(stats.get('avg_cv_auc'))}")
    print(f"  Avg test AUC:      {_fmt_float(stats.get('avg_test_auc'))}")
    print(f"  Best CV AUC:       {_fmt_float(stats.get('best_cv_auc'))}")
    print(f"  Best test AUC:     {_fmt_float(stats.get('best_test_auc'))}")
    print(f"  Avg win rate:      {_fmt_float(stats.get('avg_win_rate'))}")
    print(f"  Avg Sharpe:        {_fmt_float(stats.get('avg_sharpe'))}")


def cmd_v2_delete(db: RegistryDB, args):
    deleted = db.delete_model_entry(args.model_id)
    if deleted:
        print(f"  Deleted {args.model_id}")
    else:
        print(f"  Model {args.model_id} not found")


def cmd_v2_compare(db: RegistryDB, args):
    entries = []
    for mid in [args.id1, args.id2]:
        d = db.get_model_entry(mid)
        if d is None:
            print(f"  Model {mid} not found")
            return
        entries.append(d)

    metrics_keys = ["cv_auc", "test_auc", "train_auc", "win_rate", "sharpe_ratio",
                    "stability_score", "fragility_score", "train_test_gap"]

    headers = ["Metric", entries[0]["model_id"][:25], entries[1]["model_id"][:25]]
    rows = []
    rows.append(["status", entries[0].get("status"), entries[1].get("status")])
    rows.append(["target", entries[0].get("target_type"), entries[1].get("target_type")])
    for key in metrics_keys:
        v1 = entries[0].get("metrics", {}).get(key, 0)
        v2 = entries[1].get("metrics", {}).get(key, 0)
        rows.append([key, _fmt_float(v1), _fmt_float(v2)])

    print(f"\nModel Comparison:")
    _print_table(headers, rows)


def cmd_v2_export(db: RegistryDB, args):
    output = Path(args.output) if args.output else project_root / "data" / "registry_export.json"
    db.export_json("model_entries", output)
    print(f"  Exported to {output}")


# ========================================================================
# DB
# ========================================================================

def cmd_db_stats(db: RegistryDB, args):
    stats = db.db_stats()
    print("\nDatabase Statistics:")
    print(f"  Path:              {stats['db_path']}")
    print(f"  Size:              {stats['size_mb']:.2f} MB")
    print(f"  Experiments:       {stats['experiments']}")
    print(f"  Models (legacy):   {stats['models']}")
    print(f"  Model entries (V2):{stats['model_entries']}")


def cmd_db_vacuum(db: RegistryDB, args):
    db.vacuum()
    stats = db.db_stats()
    print(f"  Vacuumed. New size: {stats['size_mb']:.2f} MB")


def cmd_db_purge_leaky(db: RegistryDB, args):
    max_auc = getattr(args, "max_auc", 0.85)
    dry_run = getattr(args, "dry_run", False)

    # Preview counts
    conn = db._get_conn()
    exp_count = conn.execute(
        "SELECT COUNT(*) AS c FROM experiments WHERE test_auc >= ? AND test_auc > 0",
        (max_auc,),
    ).fetchone()["c"]
    mod_count = conn.execute(
        "SELECT COUNT(*) AS c FROM models WHERE test_auc >= ?", (max_auc,)
    ).fetchone()["c"]
    wmes_count = conn.execute(
        "SELECT COUNT(*) AS c FROM models WHERE wmes_score < 0"
    ).fetchone()["c"]

    print(f"\nPurge preview (AUC >= {max_auc}):")
    print(f"  Experiments to delete:   {exp_count}")
    print(f"  Models to delete:        {mod_count}")
    print(f"  Bad WMES models:         {wmes_count}")

    if dry_run:
        print("  [DRY RUN] No changes made.")
        return

    result = db.purge_all_contaminated(max_auc)
    print(f"\n  Purged: {result['experiments_purged']} experiments, "
          f"{result['models_purged']} models, {result['bad_wmes_purged']} bad WMES")
    stats = db.db_stats()
    print(f"  DB size after vacuum: {stats['size_mb']:.2f} MB")


# ========================================================================
# MAIN
# ========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Giga Trader Registry CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--db-path", type=str, default=None, help="SQLite DB path")
    subparsers = parser.add_subparsers(dest="group", help="Command group")

    # --- experiments ---
    exp_parser = subparsers.add_parser("experiments", aliases=["exp"], help="Experiment history")
    exp_sub = exp_parser.add_subparsers(dest="command")

    p = exp_sub.add_parser("list", help="List experiments")
    p.add_argument("--status", type=str, default=None)
    p.add_argument("--min-auc", type=float, default=None)
    p.add_argument("--limit", type=int, default=20)

    exp_sub.add_parser("stats", help="Show statistics")

    p = exp_sub.add_parser("prune", help="Prune low-AUC experiments")
    p.add_argument("--below-auc", type=float, required=True)
    p.add_argument("--dry-run", action="store_true")

    # --- models ---
    mod_parser = subparsers.add_parser("models", aliases=["mod"], help="Legacy model registry")
    mod_sub = mod_parser.add_subparsers(dest="command")

    p = mod_sub.add_parser("list", help="List models")
    p.add_argument("--min-tier", type=int, default=None)
    p.add_argument("--min-auc", type=float, default=None)

    mod_sub.add_parser("stats", help="Show statistics")

    p = mod_sub.add_parser("promote", help="Promote model tier")
    p.add_argument("model_id", type=str)
    p.add_argument("--tier", type=int, required=True)

    p = mod_sub.add_parser("prune", help="Prune low-AUC models")
    p.add_argument("--below-auc", type=float, required=True)
    p.add_argument("--dry-run", action="store_true")

    # --- v2 ---
    v2_parser = subparsers.add_parser("v2", help="ModelRegistryV2 entries")
    v2_sub = v2_parser.add_subparsers(dest="command")

    p = v2_sub.add_parser("list", help="List V2 model entries")
    p.add_argument("--status", type=str, default=None)
    p.add_argument("--target", type=str, default=None)
    p.add_argument("--min-auc", type=float, default=None)
    p.add_argument("--limit", type=int, default=20)

    v2_sub.add_parser("stats", help="Show statistics")

    p = v2_sub.add_parser("delete", help="Delete a model entry")
    p.add_argument("model_id", type=str)

    p = v2_sub.add_parser("compare", help="Compare two models")
    p.add_argument("id1", type=str)
    p.add_argument("id2", type=str)

    p = v2_sub.add_parser("export", help="Export to JSON")
    p.add_argument("--output", type=str, default=None)

    # --- db ---
    db_parser = subparsers.add_parser("db", help="Database operations")
    db_sub = db_parser.add_subparsers(dest="command")
    db_sub.add_parser("stats", help="Show DB stats")
    db_sub.add_parser("vacuum", help="Reclaim space")

    p = db_sub.add_parser("purge-leaky", help="Purge pre-leakage-fix garbage (AUC >= 0.85)")
    p.add_argument("--max-auc", type=float, default=0.85, help="AUC threshold (default 0.85)")
    p.add_argument("--dry-run", action="store_true", help="Preview without deleting")

    args = parser.parse_args()

    if not args.group:
        parser.print_help()
        return

    db_path = Path(args.db_path) if args.db_path else None
    db = RegistryDB(db_path)

    dispatch = {
        ("experiments", "list"): cmd_experiments_list,
        ("exp", "list"): cmd_experiments_list,
        ("experiments", "stats"): cmd_experiments_stats,
        ("exp", "stats"): cmd_experiments_stats,
        ("experiments", "prune"): cmd_experiments_prune,
        ("exp", "prune"): cmd_experiments_prune,
        ("models", "list"): cmd_models_list,
        ("mod", "list"): cmd_models_list,
        ("models", "stats"): cmd_models_stats,
        ("mod", "stats"): cmd_models_stats,
        ("models", "promote"): cmd_models_promote,
        ("mod", "promote"): cmd_models_promote,
        ("models", "prune"): cmd_models_prune,
        ("mod", "prune"): cmd_models_prune,
        ("v2", "list"): cmd_v2_list,
        ("v2", "stats"): cmd_v2_stats,
        ("v2", "delete"): cmd_v2_delete,
        ("v2", "compare"): cmd_v2_compare,
        ("v2", "export"): cmd_v2_export,
        ("db", "stats"): cmd_db_stats,
        ("db", "vacuum"): cmd_db_vacuum,
        ("db", "purge-leaky"): cmd_db_purge_leaky,
    }

    key = (args.group, args.command)
    handler = dispatch.get(key)
    if handler:
        handler(db, args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
