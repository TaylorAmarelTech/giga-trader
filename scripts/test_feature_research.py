"""
Wave 32 Smoke Test: Run a feature_research experiment end-to-end.

Forces the experiment type to feature_research (bypassing the 9% random chance)
and runs it through the FULL pipeline to verify:
  1. Candidate feature generation works
  2. Candidates are injected into the DataFrame
  3. The experiment completes through walk-forward, WMES, stability
  4. Candidate stats are updated after completion
  5. Graduation logic runs without errors

Usage:
    python scripts/test_feature_research.py
"""

import sys
import logging
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from dotenv import load_dotenv
load_dotenv(project_root / ".env")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

from src.experiment_config import create_default_config
from src.core.registry_db import RegistryDB
from src.phase_21_continuous.experiment_runner import ExperimentEngine
from src.phase_21_continuous.experiment_tracking import ExperimentGenerator
from src.phase_09_features_calendar.feature_researcher import FeatureResearchAgent


def main():
    print("=" * 70)
    print("WAVE 32 SMOKE TEST: Feature Research Experiment")
    print("=" * 70)

    # Step 1: Generate candidates
    print("\n[1/5] Generating feature candidates...")
    agent = FeatureResearchAgent()
    candidates = agent.generate_candidates(n_candidates=3)
    print(f"  Generated {len(candidates)} candidates:")
    for c in candidates:
        print(f"    - {c.name} ({c.template_type}): {c.source_features}")

    # Step 2: Create a feature_research config
    print("\n[2/5] Creating feature_research experiment config...")
    generator = ExperimentGenerator()
    config = create_default_config("feature_research_smoke_test")
    config.experiment_type = "feature_research"
    config.description = "Wave 32 smoke test: feature research"
    config.metadata["candidates"] = [c.to_dict() for c in candidates]

    # Speed up for testing: fewer Optuna trials, faster settings
    config.hp_optimization.use_optuna = False
    config.hp_optimization.optuna_n_trials = 5
    config.anti_overfit.use_anti_overfit = False  # Skip augmentation for speed

    print(f"  Config ID: {config.experiment_id}")
    print(f"  Type: {config.experiment_type}")
    print(f"  Candidates in metadata: {len(config.metadata.get('candidates', []))}")
    print(f"  Dim reduction: {config.dim_reduction.method}")
    print(f"  Model type: {config.model.model_type}")

    # Step 3: Run the experiment
    print("\n[3/5] Running experiment through full pipeline...")
    print("  (This will download data, engineer features, inject candidates,")
    print("   run walk-forward, train model, compute WMES, etc.)")
    print("-" * 70)

    db = RegistryDB()
    engine = ExperimentEngine(db=db)
    result = engine.run_experiment(config)

    print("-" * 70)
    print(f"\n[4/5] RESULT:")
    print(f"  Status:     {result.status.value}")
    print(f"  Test AUC:   {result.test_auc:.4f}")
    print(f"  WMES:       {result.wmes_score:.3f}")
    print(f"  WF Passed:  {result.walk_forward_passed}")
    print(f"  Stability:  {result.stability_score:.3f}")
    print(f"  Features:   {result.n_features_initial} initial -> {result.n_features_final} final")
    print(f"  Duration:   {result.duration_seconds:.1f}s")
    if result.error_message:
        print(f"  Error:      {result.error_message}")

    # Step 5: Check candidate stats
    print(f"\n[5/5] Feature research agent state:")
    agent2 = FeatureResearchAgent()  # Reload from disk
    print(agent2.summary())
    baseline = agent2.get_baseline_stats()
    print(f"  Baseline Tier 1 rate: {baseline['tier1_pass_rate']:.1%}")
    print(f"  Baseline avg WMES:    {baseline['avg_wmes']:.3f}")

    print("\n" + "=" * 70)
    if result.status.value == "completed":
        print("SMOKE TEST PASSED - Feature research experiment completed successfully")
    else:
        print(f"SMOKE TEST RESULT: {result.status.value}")
        if result.error_message:
            print(f"  Error: {result.error_message}")
    print("=" * 70)


if __name__ == "__main__":
    main()
