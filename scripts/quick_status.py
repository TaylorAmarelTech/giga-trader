"""
Quick Status Check for Giga Trader
====================================
Run anytime to check experiment progress, model tiers, and system health.

Usage:
    python scripts/quick_status.py
    python scripts/quick_status.py --full     # Include recent experiment details
"""
import sys
import json
import time
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.core.registry_db import get_registry_db

project_root = Path(__file__).parent.parent


def main():
    full = "--full" in sys.argv

    db = get_registry_db()
    conn = db._get_conn()

    # ── Experiment Stats ──────────────────────────────────────────────
    total_exp = conn.execute("SELECT COUNT(*) FROM experiments").fetchone()[0]
    completed = conn.execute("SELECT COUNT(*) FROM experiments WHERE status='completed'").fetchone()[0]
    failed = conn.execute("SELECT COUNT(*) FROM experiments WHERE status='failed'").fetchone()[0]

    print("=" * 60)
    print("  GIGA TRADER - Quick Status")
    print("=" * 60)
    print()
    print(f"  Experiments:   {total_exp} total ({completed} completed, {failed} failed)")
    print(f"  Success rate:  {completed/max(1,total_exp)*100:.1f}%")

    # ── Model Stats ───────────────────────────────────────────────────
    tier_counts = {}
    for tier in (1, 2, 3):
        count = conn.execute("SELECT COUNT(*) FROM models WHERE tier=?", (tier,)).fetchone()[0]
        tier_counts[tier] = count

    total_models = sum(tier_counts.values())
    best_row = conn.execute(
        "SELECT MAX(test_auc), MAX(wmes_score) FROM models WHERE test_auc IS NOT NULL"
    ).fetchone()
    best_auc = best_row[0] or 0
    best_wmes = best_row[1] or 0

    print()
    print(f"  Models:        {total_models} total")
    print(f"    Tier 1:      {tier_counts[1]}")
    print(f"    Tier 2:      {tier_counts[2]}")
    print(f"    Tier 3:      {tier_counts[3]}")
    print()
    print(f"  Best AUC:      {best_auc:.4f}")
    print(f"  Best WMES:     {best_wmes:.4f}")

    # ── Walk-Forward Stats ────────────────────────────────────────────
    wf_passed = conn.execute(
        "SELECT COUNT(*) FROM experiments WHERE status='completed' AND walk_forward_passed=1"
    ).fetchone()[0]
    print(f"  Walk-forward:  {wf_passed}/{completed} passed ({wf_passed/max(1,completed)*100:.1f}%)")

    # ── AUC Distribution ──────────────────────────────────────────────
    aucs_rows = conn.execute(
        "SELECT test_auc FROM experiments WHERE status='completed' AND test_auc > 0"
    ).fetchall()
    aucs = [r[0] for r in aucs_rows]
    if aucs:
        arr = np.array(aucs)
        print()
        print(f"  AUC stats:     mean={arr.mean():.4f}, median={np.median(arr):.4f}, std={arr.std():.4f}")
        print()
        print("  AUC Distribution:")
        bins = [(0.50, 0.52), (0.52, 0.54), (0.54, 0.56), (0.56, 0.58),
                (0.58, 0.60), (0.60, 0.65), (0.65, 1.0)]
        for lo, hi in bins:
            count = sum(1 for a in aucs if lo <= a < hi)
            bar = "#" * min(count, 40)
            print(f"    {lo:.2f}-{hi:.2f}: {count:4d} {bar}")

    # ── Top Tier 2+ Models ────────────────────────────────────────────
    t2_rows = conn.execute(
        "SELECT data_json FROM models WHERE tier >= 2 ORDER BY test_auc DESC LIMIT 10"
    ).fetchall()
    if t2_rows:
        print()
        print(f"  Top Tier 2+ Models:")
        for row in t2_rows:
            m = json.loads(row[0])
            mid = m.get("model_id", "?")[:35]
            auc = m.get("test_auc", 0)
            wmes = m.get("wmes_score", 0)
            stab = m.get("stability_score", 0)
            frag = m.get("fragility_score", 1.0)
            tier = m.get("tier", 1)
            print(f"    T{tier} AUC={auc:.3f} WMES={wmes:.3f} stab={stab:.2f} frag={frag:.3f} {mid}")

    # ── Config Diversity (last 50) ────────────────────────────────────
    cfgs = conn.execute(
        "SELECT data_json FROM experiments WHERE status='completed' ORDER BY completed_at DESC LIMIT 50"
    ).fetchall()
    methods = []
    model_types = []
    for row in cfgs:
        try:
            data = json.loads(row[0])
            cfg = data.get("config", data)
            dr = cfg.get("dim_reduction", {})
            m = cfg.get("model", {})
            methods.append(dr.get("method", "?"))
            model_types.append(m.get("model_type", "?"))
        except Exception:
            pass

    if methods:
        print()
        print(f"  Config Diversity (last 50 experiments):")
        print(f"    Dim reduction: {dict(Counter(methods))}")
        print(f"    Model types:   {dict(Counter(model_types))}")

    # ── System Health ─────────────────────────────────────────────────
    health_path = project_root / "logs" / "health_status.json"
    if health_path.exists():
        try:
            with open(health_path) as f:
                health = json.load(f)
            print()
            print(f"  System Health: {health.get('overall_status', 'UNKNOWN')}")
            for name, check in health.get("checks", {}).items():
                status = check.get("status", "?")
                msg = check.get("message", "")
                icon = {"HEALTHY": "OK", "DEGRADED": "WARN", "UNHEALTHY": "FAIL"}.get(status, "??")
                print(f"    [{icon:4s}] {name}: {msg}")
        except Exception:
            pass

    # ── Process Check ─────────────────────────────────────────────────
    try:
        import psutil
        python_procs = [p for p in psutil.process_iter(["pid", "name", "cmdline"])
                        if p.info["name"] and "python" in p.info["name"].lower()]
        system_procs = [p for p in python_procs
                        if p.info.get("cmdline") and any("start_system" in c for c in p.info["cmdline"])]
        print()
        print(f"  Processes:     {len(python_procs)} Python, {len(system_procs)} start_system.py")
    except ImportError:
        pass

    # ── Recent Experiments (--full) ───────────────────────────────────
    if full:
        recent = conn.execute("""
            SELECT experiment_id, test_auc, wmes_score, walk_forward_passed, duration_seconds
            FROM experiments ORDER BY completed_at DESC LIMIT 20
        """).fetchall()
        print()
        print("  Last 20 Experiments:")
        for r in recent:
            eid = (r[0] or "?")[:40]
            auc = r[1] or 0
            wmes = r[2] or 0
            wf = "PASS" if r[3] else "FAIL"
            dur = (r[4] or 0) / 60
            print(f"    {eid:40s} AUC={auc:.3f} WMES={wmes:.3f} WF={wf} {dur:.1f}m")

    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
