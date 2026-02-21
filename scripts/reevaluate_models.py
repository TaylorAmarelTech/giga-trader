"""
Wave 33: Re-evaluate stuck models.

Many high-AUC models are stuck at Tier 1 with stability=0.000 and
fragility=1.000 because they were created before the stability fixes.
This script re-computes their tiers using the updated compute_tier()
that accounts for stability suite measures.

Also demotes Tier 3 models that lack suite validation (pre-Wave 29).

Usage:
    python scripts/reevaluate_models.py [--dry-run]
"""

import json
import sys
import sqlite3
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.registry_db import compute_tier


def main():
    dry_run = "--dry-run" in sys.argv
    db_path = project_root / "data" / "giga_trader.db"
    conn = sqlite3.connect(str(db_path))
    c = conn.cursor()

    print("=" * 70)
    print("WAVE 33: Model Re-evaluation")
    print("=" * 70)
    if dry_run:
        print("  (DRY RUN - no changes will be made)\n")

    # --- 0. Demote Tier 3 models with broken fragility (pre-Wave 33) ---
    # These got Tier 3 because the old RobustnessEnsemble always produced
    # fragility ~0 (used generic LogisticRegression+PCA, not actual model).
    # Demote to Tier 2 so they can re-earn Tier 3 with proper validation.
    c.execute("""SELECT id, model_id, test_auc, fragility_score, data_json
                 FROM models WHERE tier = 3""")
    t3_rows = c.fetchall()
    demoted = []
    for row_id, model_id, test_auc, frag, data_json in t3_rows:
        d = json.loads(data_json) if data_json else {}
        suite_composite = d.get("stability_composite", 0.0) or 0.0
        has_bootstrap = d.get("stability_bootstrap", 0) > 0

        # Demote if: no suite data AND fragility < 0.05 (clearly from broken test)
        if not has_bootstrap and (frag is None or frag < 0.05):
            demoted.append((model_id, test_auc, frag, suite_composite))
            if not dry_run:
                # Set tier=2 and reset fragility to 0.5 (unknown) so compute_tier
                # won't re-promote based on the broken fragility score
                c.execute("UPDATE models SET tier = 2, fragility_score = 0.5 WHERE id = ?", (row_id,))
                # Also update fragility in data_json
                d["fragility_score"] = 0.5
                c.execute("UPDATE models SET data_json = ? WHERE id = ?", (json.dumps(d), row_id))

    if demoted:
        print(f"\n--- DEMOTED T3->T2 ({len(demoted)} models, no suite data + broken fragility) ---")
        for mid, auc, frag, sc in demoted:
            print(f"  {mid[-12:]} | AUC={auc:.3f} | Frag={frag} | Suite={sc}")
    else:
        print("\n  No Tier 3 demotions needed.")

    # --- 1. Re-tier ALL models using updated compute_tier ---
    c.execute("SELECT id, model_id, tier, test_auc, stability_score, fragility_score, data_json FROM models")
    rows = c.fetchall()

    upgrades = []
    downgrades = []
    unchanged = 0

    for row_id, model_id, old_tier, test_auc, stab, frag, data_json in rows:
        d = json.loads(data_json) if data_json else {}

        # Extract suite composite from data_json (not in SQL columns)
        suite_composite = d.get("stability_composite", 0.0) or 0.0

        # Use stored values (may be 0/1 defaults for old models)
        stab = stab or 0.0
        frag = frag if frag is not None else 1.0

        new_tier = compute_tier(stab, frag, test_auc, suite_composite)

        if new_tier != old_tier:
            if new_tier > old_tier:
                upgrades.append((model_id, old_tier, new_tier, test_auc, stab, frag, suite_composite))
            else:
                downgrades.append((model_id, old_tier, new_tier, test_auc, stab, frag, suite_composite))

            if not dry_run:
                c.execute("UPDATE models SET tier = ? WHERE id = ?", (new_tier, row_id))
        else:
            unchanged += 1

    # --- 2. Report ---
    print(f"\nTotal models: {len(rows)}")
    print(f"Unchanged: {unchanged}")

    if upgrades:
        print(f"\n--- UPGRADES ({len(upgrades)}) ---")
        for mid, ot, nt, auc, stab, frag, sc in upgrades:
            print(f"  {mid[-12:]} | T{ot}->T{nt} | AUC={auc:.3f} | Stab={stab:.3f} | Frag={frag:.3f} | Suite={sc:.3f}")

    if downgrades:
        print(f"\n--- DOWNGRADES ({len(downgrades)}) ---")
        for mid, ot, nt, auc, stab, frag, sc in downgrades:
            print(f"  {mid[-12:]} | T{ot}->T{nt} | AUC={auc:.3f} | Stab={stab:.3f} | Frag={frag:.3f} | Suite={sc:.3f}")

    # --- 3. Identify stuck models (high AUC, T1 with stab=0 or frag=1) ---
    c.execute("""SELECT model_id, test_auc, stability_score, fragility_score, data_json
                 FROM models WHERE tier = 1 AND test_auc >= 0.58
                 ORDER BY test_auc DESC""")
    stuck = c.fetchall()

    if stuck:
        print(f"\n--- STUCK AT TIER 1 (AUC >= 0.58, need re-testing) ---")
        print(f"  Found {len(stuck)} models with high AUC stuck at Tier 1")
        for mid, auc, stab, frag, dj in stuck[:10]:
            d = json.loads(dj) if dj else {}
            sc = d.get("stability_composite", 0)
            reason = []
            if stab == 0 or stab is None:
                reason.append("stab=0(never ran)")
            elif stab < 0.60:
                reason.append(f"stab={stab:.3f}<0.60")
            if frag >= 0.40 or frag is None:
                reason.append(f"frag={frag}>=0.40")
            if sc > 0 and sc < 0.45:
                reason.append(f"suite={sc:.3f}<0.45")
            print(f"  {mid[-12:]} | AUC={auc:.3f} | Stab={stab} | Frag={frag} | Suite={sc} | Reason: {', '.join(reason) or 'unknown'}")

        print(f"\n  These models need their experiments re-run through the updated pipeline")
        print(f"  to get proper stability/fragility scores.")

    if not dry_run:
        conn.commit()
        print(f"\n[COMMITTED] {len(upgrades)} upgrades, {len(downgrades)} downgrades")
    else:
        print(f"\n[DRY RUN] Would apply {len(upgrades)} upgrades, {len(downgrades)} downgrades")

    conn.close()


if __name__ == "__main__":
    main()
