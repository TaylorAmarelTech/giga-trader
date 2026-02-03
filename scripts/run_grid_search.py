"""
GIGA TRADER - Run Full Entry/Exit Window Grid Search
=====================================================
Runs systematic grid search across all entry/exit window combinations
and stores results in the model registry.

Usage:
    python scripts/run_grid_search.py [--max-configs N] [--quick]

Options:
    --max-configs N    Limit to N configurations (default: all ~96 combinations)
    --quick           Run quick test with 3 configurations
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Setup path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")


def setup_logging():
    """Setup logging."""
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f"grid_search_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ]
    )
    return logging.getLogger("GridSearch")


def main():
    parser = argparse.ArgumentParser(description="Run Entry/Exit Window Grid Search")
    parser.add_argument("--max-configs", type=int, default=None,
                       help="Maximum number of configurations to test")
    parser.add_argument("--quick", action="store_true",
                       help="Quick test mode (3 configurations)")
    args = parser.parse_args()

    logger = setup_logging()

    print("=" * 70)
    print("GIGA TRADER - Entry/Exit Window Grid Search")
    print("=" * 70)

    # Import grid search components
    try:
        from src.dynamic_model_selector import EntryExitGridSearchRunner
        from src.experiment_engine import ExperimentEngine
    except ImportError as e:
        print(f"[ERROR] Failed to import grid search components: {e}")
        return 1

    # Determine max configs
    if args.quick:
        max_configs = 3
        print(f"\n[QUICK MODE] Testing with {max_configs} configurations")
    elif args.max_configs:
        max_configs = args.max_configs
        print(f"\n[LIMITED MODE] Testing with {max_configs} configurations")
    else:
        max_configs = None
        print("\n[FULL MODE] Testing all entry/exit window combinations")

    # Initialize grid search runner
    print("\nInitializing experiment engine...")
    engine = ExperimentEngine()
    runner = EntryExitGridSearchRunner(experiment_engine=engine)

    # Show all configurations that will be tested
    configs = runner.generate_all_configs()
    print(f"\nTotal configurations to test: {len(configs)}")

    if max_configs:
        configs_to_run = min(max_configs, len(configs))
    else:
        configs_to_run = len(configs)

    print(f"Configurations to run: {configs_to_run}")

    # Show sample configurations
    print("\nSample configurations:")
    for i, config in enumerate(configs[:5]):
        entry = config.entry_exit
        print(f"  {i+1}. Entry: {entry['entry_window_start']}-{entry['entry_window_end']}min, "
              f"Exit: {entry['exit_window_start']}-{entry['exit_window_end']}min")

    print("\n" + "=" * 70)
    print("Starting Grid Search...")
    print("=" * 70)

    # Run grid search
    try:
        results = runner.run_grid_search(max_configs=max_configs)
    except KeyboardInterrupt:
        print("\n[INTERRUPTED] Grid search stopped by user")
        results = runner.results

    # Summary
    print("\n" + "=" * 70)
    print("GRID SEARCH COMPLETE")
    print("=" * 70)

    completed = [r for r in results if r.get("status") == "completed"]
    failed = [r for r in results if r.get("status") == "failed"]

    print(f"\nResults:")
    print(f"  Total tested: {len(results)}")
    print(f"  Completed: {len(completed)}")
    print(f"  Failed: {len(failed)}")

    if completed:
        # Get best configs
        best = runner.get_best_configs(n=5)
        print(f"\nTop 5 Configurations:")
        for i, r in enumerate(best):
            print(f"  {i+1}. Entry: {r['entry_window']}, Exit: {r['exit_window']}")
            print(f"     Test AUC: {r.get('test_auc', 0):.4f}, Sharpe: {r.get('backtest_sharpe', 0):.3f}")

        # Show registry status
        print("\nModel Registry Status:")
        stats = engine.registry.get_statistics()
        print(f"  Total models: {stats.get('total_models', 0)}")
        print(f"  Best CV AUC: {stats.get('best_cv_auc', 0):.4f}")
        print(f"  Best Sharpe: {stats.get('best_backtest_sharpe', 0):.3f}")

    # Save results summary
    results_path = project_root / "experiments" / "grid_search_summary.json"
    import json
    with open(results_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "n_tested": len(results),
            "n_completed": len(completed),
            "n_failed": len(failed),
            "best_configs": runner.get_best_configs(n=10),
        }, f, indent=2, default=str)

    print(f"\nResults saved to: {results_path}")
    print("\nTo use the best models in paper trading, the system will automatically")
    print("select from the registry using the DynamicModelSelector.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
