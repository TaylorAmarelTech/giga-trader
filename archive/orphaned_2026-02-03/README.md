# Archived Orphaned Files - 2026-02-03

These files were moved from `src/` because they are not connected to the main pipeline.
The main pipeline (`giga_orchestrator.py`) works without them.

## Why Archived (Not Deleted)

These files contain working code that may be useful for reference or future features.

## Files Archived

### `long_runner/` (6 files)
- **Purpose**: Alternative orchestration via Claude Code CLI subprocesses
- **Status**: Superseded by `giga_orchestrator.py` which does the same thing directly
- **Files**: orchestrator.py, grid_search.py, monitoring.py, process_manager.py, state_manager.py, service.py

### `core/` (3 files)
- **Purpose**: Abstract base classes for pipeline phases
- **Status**: Never integrated - main code doesn't use these abstractions
- **Files**: base.py, exceptions.py, logging.py

### `multi_agent_system.py`
- **Purpose**: 10-agent parallel system with message bus
- **Status**: Over-engineered - simpler approaches in main pipeline work better
- **Unique**: Has SignalValidatorAgent and TestGenerationAgent not elsewhere

### `train_dual_model.py`
- **Purpose**: Trains swing + timing models
- **Status**: Superseded by `train_robust_model.py` which has more features

### `train_intraday_model.py`
- **Purpose**: Morning dip + afternoon swing patterns
- **Status**: Merged into `train_robust_model.py` as EDGE 3

### `train_optimized_model.py`
- **Purpose**: Grid-searched model training
- **Status**: Replaced by Optuna Bayesian optimization in `train_robust_model.py`

### `train_spy_model.py`
- **Purpose**: Basic SPY model from 1-year data
- **Status**: Superseded by 5-year training in `train_robust_model.py`

## To Restore

If you need any of this code, simply move it back to `src/`:
```bash
mv archive/orphaned_2026-02-03/long_runner src/
```

## Active Pipeline (What's Still Used)

```
giga_orchestrator.py (ENTRY POINT)
├── train_robust_model.py + anti_overfit.py + leak_proof_cv.py
├── paper_trading.py + supervision/*
├── experiment_engine.py + backtesting_harness.py
└── dashboard_server.py
```
