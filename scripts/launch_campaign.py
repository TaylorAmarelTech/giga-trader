"""
10-Day Training Campaign Launcher.
====================================
Orchestrates a structured training campaign with three phases:
  - Exploration (day 1-3): Wide parameter search, max diversity
  - Exploitation (day 4-7): Narrow to top regions, thick weave focus
  - Evaluation (day 8-10): Backtest Tier 2+ models, generate reports

Usage:
    python scripts/launch_campaign.py --phase exploration
    python scripts/launch_campaign.py --phase exploitation
    python scripts/launch_campaign.py --phase evaluation
    python scripts/launch_campaign.py --auto              # Auto-detect phase
    python scripts/launch_campaign.py --report             # Generate campaign report
    python scripts/launch_campaign.py --purge              # Purge pre-leakage garbage
    python scripts/launch_campaign.py --readiness          # Check paper trading readiness
"""

import argparse
import json
import sys
import time
from datetime import date, datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


class CampaignManager:
    """Manages the 10-day training campaign lifecycle."""

    def __init__(self, campaign_id: str = None, start_date: date = None):
        from src.core.registry_db import get_registry_db
        self.db = get_registry_db()
        self.campaign_id = campaign_id or f"campaign_{date.today().isoformat()}"
        self.start_date = start_date or date.today()
        self.config_dir = project_root / "campaigns" / self.campaign_id
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.logger = self._setup_logger()

    def _setup_logger(self):
        import logging
        logger = logging.getLogger("CAMPAIGN")
        if not logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S"
            ))
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def get_current_phase(self) -> str:
        """Determine phase based on days elapsed since campaign start."""
        days_elapsed = (date.today() - self.start_date).days
        if days_elapsed < 3:
            return "exploration"
        elif days_elapsed < 7:
            return "exploitation"
        else:
            return "evaluation"

    def save_phase_config(self, phase: str, config: dict):
        """Save phase-specific configuration."""
        filepath = self.config_dir / "phase_config.json"
        data = {
            "campaign_id": self.campaign_id,
            "start_date": self.start_date.isoformat(),
            "current_phase": phase,
            "updated_at": datetime.now().isoformat(),
            "config": config,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        self.logger.info(f"Phase config saved: {filepath}")

    def configure_exploration(self) -> dict:
        """Phase 1 (day 1-3): Wide parameter search, max diversity."""
        config = {
            "phase": "exploration",
            "optuna_n_trials": 25,
            "thick_weave_budget": 20,
            "inter_experiment_sleep": 10,
            "fast_screen_enabled": True,
            "description": "Wide search: all model types, equal weights, fast screening",
        }
        self.save_phase_config("exploration", config)
        self.logger.info("Configured EXPLORATION phase: diverse configs, fast screening")
        return config

    def configure_exploitation(self) -> dict:
        """Phase 2 (day 4-7): Narrow to top-performing regions."""
        # Load top configs from registry
        top_models = self.db.get_models(min_tier=1, max_auc=0.85)
        n_top = len(top_models)

        config = {
            "phase": "exploitation",
            "optuna_n_trials": 50,
            "thick_weave_budget": 100,
            "inter_experiment_sleep": 10,
            "fast_screen_enabled": True,
            "n_top_models_for_seed": min(n_top, 20),
            "description": f"Exploit top {min(n_top, 20)} configs with deeper search",
        }
        self.save_phase_config("exploitation", config)
        self.logger.info(f"Configured EXPLOITATION phase: seeding from {n_top} models")
        return config

    def configure_evaluation(self) -> dict:
        """Phase 3 (day 8-10): Backtest and rank models."""
        config = {
            "phase": "evaluation",
            "stop_new_experiments": True,
            "run_backtests": True,
            "generate_summary": True,
            "description": "Evaluate: backtest Tier 2+ models, generate summary",
        }
        self.save_phase_config("evaluation", config)
        self.logger.info("Configured EVALUATION phase: backtesting + reporting")
        return config

    def purge_contaminated(self):
        """Purge pre-leakage-fix garbage from registry."""
        self.logger.info("Purging contaminated registry entries...")
        result = self.db.purge_all_contaminated(0.85)
        self.logger.info(
            f"Purged: {result['experiments_purged']} experiments, "
            f"{result['models_purged']} models, "
            f"{result['bad_wmes_purged']} bad WMES"
        )
        stats = self.db.db_stats()
        self.logger.info(f"DB size after purge: {stats['size_mb']:.2f} MB")
        return result

    def backtest_tier2_models(self) -> list:
        """Run backtests on all Tier 2+ models."""
        models = self.db.get_models(min_tier=2, max_auc=0.85)
        if not models:
            self.logger.warning("No Tier 2+ models found for backtesting")
            return []

        self.logger.info(f"Backtesting {len(models)} Tier 2+ models...")
        results = []

        for i, model_rec in enumerate(models, 1):
            model_path = model_rec.get("model_path", "")
            model_id = model_rec.get("model_id", f"unknown_{i}")

            if not model_path or not Path(model_path).is_file():
                self.logger.warning(f"  [{i}/{len(models)}] Skipping {model_id} — no model file")
                continue

            try:
                import joblib
                saved = joblib.load(model_path)

                # Extract backtest-relevant info
                test_auc = model_rec.get("test_auc", 0)
                wmes = model_rec.get("wmes_score", 0)
                tier = model_rec.get("tier", 1)
                stability = model_rec.get("stability_score", 0)

                backtest_result = {
                    "model_id": model_id,
                    "test_auc": test_auc,
                    "wmes_score": wmes,
                    "tier": tier,
                    "stability_score": stability,
                    "model_path": model_path,
                    "has_model_file": True,
                }
                results.append(backtest_result)
                self.logger.info(
                    f"  [{i}/{len(models)}] {model_id}: "
                    f"AUC={test_auc:.4f}, WMES={wmes:.3f}, Tier={tier}"
                )

            except Exception as e:
                self.logger.warning(f"  [{i}/{len(models)}] {model_id} failed: {e}")

        # Rank by composite score
        results.sort(key=lambda r: r.get("test_auc", 0) * r.get("wmes_score", 0), reverse=True)

        # Save results
        output_path = self.config_dir / "backtest_results.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"Backtest results saved: {output_path}")

        return results

    def check_paper_readiness(self) -> dict:
        """Check if system is ready for paper trading."""
        from src.giga_orchestrator import ExperimentGateChecker
        checker = ExperimentGateChecker()
        gates_passed, gate_status = checker.check_gates(trading_mode="paper")

        model_stats = self.db.get_model_statistics()
        exp_stats = self.db.get_experiment_statistics()
        tier_dist = model_stats.get("by_tier", {})

        tier2_models = self.db.get_models(min_tier=2, max_auc=0.85)
        tier3_models = self.db.get_models(min_tier=3, max_auc=0.85)

        tier3_count = len(tier3_models)

        if tier3_count >= 3 and gates_passed:
            recommendation = "READY: Start paper trading with top Tier 3 ensemble"
        elif tier3_count >= 1 and gates_passed:
            recommendation = "CAUTIOUS: Paper trading possible with limited Tier 3 models"
        elif gates_passed:
            recommendation = "CONTINUE: Gates pass but no Tier 3 models — continue training"
        else:
            recommendation = "NOT READY: Trading gates not met — extend campaign"

        summary = {
            "generated_at": datetime.now().isoformat(),
            "campaign_id": self.campaign_id,
            "paper_gates_passed": gates_passed,
            "gate_details": gate_status,
            "total_experiments": exp_stats.get("total", 0),
            "completed_experiments": exp_stats.get("completed", 0),
            "success_rate": exp_stats.get("success_rate", 0),
            "best_realistic_auc": exp_stats.get("best_realistic_auc", 0),
            "tier_distribution": {str(k): v for k, v in tier_dist.items()},
            "tier2_count": len(tier2_models),
            "tier3_count": tier3_count,
            "top_5_models": [
                {
                    "model_id": m.get("model_id"),
                    "test_auc": m.get("test_auc", 0),
                    "wmes": m.get("wmes_score", 0),
                    "tier": m.get("tier", 1),
                    "stability": m.get("stability_score", 0),
                }
                for m in tier2_models[:5]
            ],
            "recommendation": recommendation,
        }

        # Save summary
        report_dir = project_root / "reports" / "campaign"
        report_dir.mkdir(parents=True, exist_ok=True)
        filepath = report_dir / "campaign_summary.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        return summary

    def run_phase(self, phase: str):
        """Execute a campaign phase."""
        if phase == "exploration":
            self.configure_exploration()
        elif phase == "exploitation":
            self.configure_exploitation()
        elif phase == "evaluation":
            self.configure_evaluation()
            results = self.backtest_tier2_models()
            summary = self.check_paper_readiness()
            self._print_readiness(summary)
            return summary
        else:
            self.logger.error(f"Unknown phase: {phase}")

    def _print_readiness(self, summary: dict):
        """Print paper trading readiness."""
        print()
        print("=" * 60)
        print("  CAMPAIGN READINESS CHECK")
        print("=" * 60)
        print(f"\n  Total experiments:   {summary['total_experiments']}")
        print(f"  Completed:           {summary['completed_experiments']}")
        print(f"  Success rate:        {summary['success_rate']:.1%}")
        print(f"  Best realistic AUC:  {summary['best_realistic_auc']:.4f}")
        print(f"\n  Tier 2 models:       {summary['tier2_count']}")
        print(f"  Tier 3 models:       {summary['tier3_count']}")
        print(f"  Paper gates passed:  {summary['paper_gates_passed']}")
        print(f"\n  >>> {summary['recommendation']}")
        print()


def main():
    parser = argparse.ArgumentParser(description="10-Day Training Campaign Launcher")
    parser.add_argument("--phase", choices=["exploration", "exploitation", "evaluation"],
                        help="Campaign phase to configure")
    parser.add_argument("--auto", action="store_true", help="Auto-detect phase from start date")
    parser.add_argument("--start-date", type=str, default=None, help="Campaign start date (YYYY-MM-DD)")
    parser.add_argument("--purge", action="store_true", help="Purge pre-leakage garbage first")
    parser.add_argument("--report", action="store_true", help="Generate daily report")
    parser.add_argument("--readiness", action="store_true", help="Check paper trading readiness")
    parser.add_argument("--backtest", action="store_true", help="Backtest Tier 2+ models")
    parser.add_argument("--start", action="store_true", help="Start the system after configuring")
    args = parser.parse_args()

    start_date = date.fromisoformat(args.start_date) if args.start_date else date.today()
    manager = CampaignManager(start_date=start_date)

    if args.purge:
        manager.purge_contaminated()

    if args.report:
        from scripts.campaign_report import generate_daily_report, print_report, save_report
        report = generate_daily_report(manager.db)
        print_report(report)
        save_report(report)
        return

    if args.readiness:
        summary = manager.check_paper_readiness()
        manager._print_readiness(summary)
        return

    if args.backtest:
        manager.backtest_tier2_models()
        return

    if args.auto:
        phase = manager.get_current_phase()
        print(f"  Auto-detected phase: {phase} (day {(date.today() - start_date).days})")
    elif args.phase:
        phase = args.phase
    else:
        parser.print_help()
        return

    manager.run_phase(phase)

    if args.start:
        print("\n  Starting system with campaign configuration...")
        import subprocess
        subprocess.run([
            sys.executable, str(project_root / "scripts" / "start_system.py"),
            "--no-trading",
        ])


if __name__ == "__main__":
    main()
