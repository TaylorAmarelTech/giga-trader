"""
Run the Thick Multi-Weave Grid Search.

This script kicks off an intelligent multi-prong grid search that finds
thick, wide optimization paths across both model hyperparameters AND
feature spaces. Each thread explores a different region of the joint
(model × dim_reduction × feature_selection × scaling) space.

Usage:
    python scripts/run_thick_weave_search.py                    # Default (200 evals, ~17h)
    python scripts/run_thick_weave_search.py --budget 50        # Quick (50 evals, ~4h)
    python scripts/run_thick_weave_search.py --budget 20 --threads 4  # Minimal test
"""

import sys
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.phase_23_analytics.thick_weave_search import ThickWeaveSearch, ThickWeaveConfig


def setup_logging():
    """Configure logging to both console and file."""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"thick_weave_search_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, encoding="utf-8"),
        ],
    )
    return log_file


def main():
    parser = argparse.ArgumentParser(description="Thick Multi-Weave Grid Search")
    parser.add_argument("--budget", type=int, default=200,
                        help="Total evaluations budget (default: 200)")
    parser.add_argument("--threads", type=int, default=6,
                        help="Number of initial seed threads (default: 6, max: 12)")
    parser.add_argument("--rounds", type=int, default=40,
                        help="Max exploration rounds (default: 40)")
    parser.add_argument("--target", type=str, default="swing",
                        choices=["swing", "timing"],
                        help="Target type (default: swing)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--checkpoint-dir", type=str,
                        default="models/thick_weave_checkpoints",
                        help="Checkpoint directory")
    args = parser.parse_args()

    log_file = setup_logging()
    logger = logging.getLogger("THICK_WEAVE_RUNNER")

    logger.info("=" * 70)
    logger.info("THICK MULTI-WEAVE GRID SEARCH RUNNER")
    logger.info("=" * 70)
    logger.info(f"  Budget: {args.budget} evaluations")
    logger.info(f"  Seed threads: {args.threads}")
    logger.info(f"  Max rounds: {args.rounds}")
    logger.info(f"  Target: {args.target}")
    logger.info(f"  Log file: {log_file}")
    logger.info("")

    config = ThickWeaveConfig(
        n_initial_threads=min(args.threads, 12),
        max_total_evaluations=args.budget,
        max_rounds=args.rounds,
        random_seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
    )

    search = ThickWeaveSearch(config)
    report = search.run(target_type=args.target)

    # Save report
    report_dir = project_root / "reports"
    report_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = report_dir / f"thick_weave_report_{timestamp}.json"

    with open(report_file, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"\nReport saved to: {report_file}")

    # Print summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"  Total evaluations: {report['search_stats']['total_evaluated']}")
    print(f"  Time: {report['search_stats']['elapsed_minutes']:.1f} minutes")
    print(f"  Best WMES: {report['best_overall_wmes']:.4f}")
    print(f"  Thick paths found: {len(report['thick_paths'])}")

    if report["production_candidates"]:
        print(f"\n  Production Candidates ({len(report['production_candidates'])}):")
        for cand in report["production_candidates"]:
            print(f"    [{cand['thread_id']}] WMES={cand['wmes']:.4f}, "
                  f"PTS={cand['pts']:.3f}, model={cand['model_type']}, "
                  f"dim_red={cand['dim_reduction']}")
    else:
        print("\n  No thick paths found yet (increase budget or lower threshold)")

    print(f"\n  Full report: {report_file}")
    print(f"  Log: {log_file}")


if __name__ == "__main__":
    main()
