"""
Daily Campaign Report Generator.
=================================
Queries SQLite for experiments and generates summary reports.

Usage:
    python scripts/campaign_report.py                    # Today's report
    python scripts/campaign_report.py --date 2026-02-14  # Specific date
    python scripts/campaign_report.py --all              # Full campaign summary
"""

import argparse
import json
import sys
from datetime import date, datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

import numpy as np


def generate_daily_report(db, target_date: date = None) -> dict:
    """Generate report for a single day's experiments."""
    target_date = target_date or date.today()
    date_str = target_date.isoformat()

    # Get all completed experiments for this day
    all_completed = db.get_experiments(status="completed")
    today_completed = [
        e for e in all_completed
        if e.get("completed_at", "").startswith(date_str)
    ]

    all_failed = db.get_experiments(status="failed")
    today_failed = [
        e for e in all_failed
        if e.get("completed_at", "").startswith(date_str)
        or e.get("started_at", "").startswith(date_str)
    ]

    n_completed = len(today_completed)
    n_failed = len(today_failed)
    n_total = n_completed + n_failed
    success_rate = n_completed / max(n_total, 1)

    # AUC stats
    aucs = [e["test_auc"] for e in today_completed if e.get("test_auc", 0) > 0]
    wmes_scores = [e["wmes_score"] for e in today_completed if e.get("wmes_score", 0) > 0]
    wf_passed = sum(1 for e in today_completed if e.get("walk_forward_passed"))

    # Duration stats
    durations = [e.get("duration_seconds", 0) for e in today_completed if e.get("duration_seconds", 0) > 0]

    # Model tier counts
    model_stats = db.get_model_statistics()
    tier_dist = model_stats.get("by_tier", {})

    # Feature consensus from cross-learner
    consensus_features = _get_feature_consensus()

    report = {
        "date": date_str,
        "generated_at": datetime.now().isoformat(),
        "experiments": {
            "completed": n_completed,
            "failed": n_failed,
            "total": n_total,
            "success_rate": round(success_rate, 3),
        },
        "auc": {
            "mean": round(float(np.mean(aucs)), 4) if aucs else 0,
            "best": round(float(max(aucs)), 4) if aucs else 0,
            "worst": round(float(min(aucs)), 4) if aucs else 0,
            "std": round(float(np.std(aucs)), 4) if aucs else 0,
        },
        "wmes": {
            "mean": round(float(np.mean(wmes_scores)), 4) if wmes_scores else 0,
            "best": round(float(max(wmes_scores)), 4) if wmes_scores else 0,
        },
        "walk_forward": {
            "passed": wf_passed,
            "total": n_completed,
            "pass_rate": round(wf_passed / max(n_completed, 1), 3),
        },
        "duration": {
            "mean_seconds": round(float(np.mean(durations)), 1) if durations else 0,
            "total_hours": round(sum(durations) / 3600, 2) if durations else 0,
        },
        "models": {
            "total": model_stats.get("total_models", 0),
            "tier_1": tier_dist.get(1, 0),
            "tier_2": tier_dist.get(2, 0),
            "tier_3": tier_dist.get(3, 0),
        },
        "feature_consensus": consensus_features[:10],
    }

    return report


def generate_campaign_summary(db) -> dict:
    """Generate full campaign summary across all experiments."""
    exp_stats = db.get_experiment_statistics()
    model_stats = db.get_model_statistics()
    tier_dist = model_stats.get("by_tier", {})

    return {
        "generated_at": datetime.now().isoformat(),
        "experiments": exp_stats,
        "models": {
            "total": model_stats.get("total_models", 0),
            "tier_1": tier_dist.get(1, 0),
            "tier_2": tier_dist.get(2, 0),
            "tier_3": tier_dist.get(3, 0),
            "best_wmes": model_stats.get("best_wmes", 0),
            "best_cv_auc": model_stats.get("best_cv_auc", 0),
            "best_backtest_sharpe": model_stats.get("best_backtest_sharpe", 0),
        },
        "feature_consensus": _get_feature_consensus(),
    }


def _get_feature_consensus() -> list:
    """Get top feature consensus from cross-learner if available."""
    fmap_path = project_root / "models" / "feature_importance_map.json"
    if not fmap_path.is_file():
        return []
    try:
        from src.phase_23_analytics.symbolic_cross_learner import UniversalFeatureMap
        fmap = UniversalFeatureMap(persist_path=fmap_path)
        return fmap.get_consensus_features(top_n=20, min_models=3)
    except Exception:
        return []


def save_report(report: dict, output_dir: Path = None):
    """Save report to JSON file."""
    output_dir = output_dir or (project_root / "reports" / "campaign")
    output_dir.mkdir(parents=True, exist_ok=True)

    date_str = report.get("date", date.today().isoformat())
    filepath = output_dir / f"{date_str}.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    return filepath


def print_report(report: dict):
    """Pretty-print a campaign report."""
    print()
    print("=" * 60)
    print(f"  Campaign Report: {report.get('date', 'Summary')}")
    print("=" * 60)

    exp = report.get("experiments", {})
    print(f"\n  Experiments:  {exp.get('completed', 0)} completed, "
          f"{exp.get('failed', 0)} failed "
          f"({exp.get('success_rate', 0):.0%} success)")

    auc = report.get("auc", {})
    if auc.get("mean"):
        print(f"  AUC:          mean={auc['mean']:.4f}, "
              f"best={auc['best']:.4f}, std={auc['std']:.4f}")

    wmes = report.get("wmes", {})
    if wmes.get("mean"):
        print(f"  WMES:         mean={wmes['mean']:.4f}, best={wmes['best']:.4f}")

    wf = report.get("walk_forward", {})
    print(f"  Walk-Forward: {wf.get('passed', 0)}/{wf.get('total', 0)} "
          f"({wf.get('pass_rate', 0):.0%} pass rate)")

    dur = report.get("duration", {})
    print(f"  Compute:      {dur.get('total_hours', 0):.1f} hours, "
          f"avg {dur.get('mean_seconds', 0):.0f}s per experiment")

    models = report.get("models", {})
    print(f"\n  Models:       {models.get('total', 0)} total")
    print(f"    Tier 1:     {models.get('tier_1', 0)}")
    print(f"    Tier 2:     {models.get('tier_2', 0)} (paper-eligible)")
    print(f"    Tier 3:     {models.get('tier_3', 0)} (live-eligible)")

    consensus = report.get("feature_consensus", [])
    if consensus:
        print(f"\n  Top Features (consensus):")
        for feat, score, n in consensus[:5]:
            print(f"    {feat:40s}  score={score:.4f}  ({n} models)")

    print()


def main():
    parser = argparse.ArgumentParser(description="Campaign Report Generator")
    parser.add_argument("--date", type=str, default=None, help="Report date (YYYY-MM-DD)")
    parser.add_argument("--all", action="store_true", help="Full campaign summary")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    args = parser.parse_args()

    from src.core.registry_db import get_registry_db
    db = get_registry_db()

    if args.all:
        report = generate_campaign_summary(db)
        print_report(report)
        filepath = save_report(report, Path(args.output) if args.output else None)
        print(f"  Saved to: {filepath}")
    else:
        target_date = date.fromisoformat(args.date) if args.date else date.today()
        report = generate_daily_report(db, target_date)
        print_report(report)
        filepath = save_report(report, Path(args.output) if args.output else None)
        print(f"  Saved to: {filepath}")


if __name__ == "__main__":
    main()
