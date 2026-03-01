# Giga Trader — Getting Started

> Complete guide to installing, configuring, and running the ML trading pipeline.

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [First Run](#first-run)
5. [Running Tests](#running-tests)
6. [System Resource Detection](#system-resource-detection)
7. [Training Your First Model](#training-your-first-model)
8. [Paper Trading Setup](#paper-trading-setup)
9. [Common Operations](#common-operations)
10. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required

| Requirement | Version | Notes |
|-------------|---------|-------|
| **Python** | 3.11 or 3.12 | 3.13+ is NOT supported (package compatibility) |
| **RAM** | 16 GB minimum | Pipeline adapts to available memory |
| **Disk** | 10 GB free | For market data, models, and cache |
| **OS** | Windows 10+, Linux, macOS | Tested primarily on Windows 11 |

### Recommended

| Requirement | Version | Notes |
|-------------|---------|-------|
| **RAM** | 32+ GB | Enables full pipeline without skipping methods |
| **GPU** | CUDA-capable | XGBoost GPU acceleration (auto-detected) |
| **CPU** | 8+ cores | Parallel feature computation and CV |

### API Keys

| Service | Required | Purpose | Signup |
|---------|----------|---------|--------|
| **Alpaca** | Yes | Market data + paper trading | [alpaca.markets](https://alpaca.markets) |
| Alpha Vantage | Optional | News sentiment data | [alphavantage.co](https://www.alphavantage.co) |
| Finnhub | Optional | Company news | [finnhub.io](https://finnhub.io) |
| NewsAPI | Optional | General news | [newsapi.org](https://newsapi.org) |
| FRED | Optional | Economic indicators | [fred.stlouisfed.org](https://fred.stlouisfed.org) |
| Mistral | Optional | LLM sentiment analysis | [mistral.ai](https://mistral.ai) |

Only Alpaca is required to run the core pipeline. Other APIs add optional features.

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/TaylorAmarelTech/giga-trader.git
cd giga-trader
```

### 2. Create Virtual Environment

```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3.12 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**Optional packages** (for advanced features):

```bash
# CatBoost (gradient boosting with categorical support)
pip install catboost>=1.2.0

# giotto-tda (Topological Data Analysis features)
pip install giotto-tda>=0.6.0

# psutil (system resource detection - highly recommended)
pip install psutil>=5.9.0

# CUDA support for XGBoost (if you have an NVIDIA GPU)
# XGBoost auto-detects GPU - no extra install needed if CUDA is set up
```

### 4. Verify Installation

```bash
# Check Python version
python --version  # Should show 3.11.x or 3.12.x

# Check imports
python -c "from src.core.system_resources import get_system_resources; print(get_system_resources().summary())"

# Expected output:
# SYSTEM RESOURCES (detected ...)
#   Tier: MEDIUM
#   RAM:  31.8 GB total, 12.4 GB available
#   CPU:  10 physical / 16 logical cores
#   GPU:  XGBoost GPU detected
```

---

## Configuration

### 1. Environment Variables

```bash
# Copy the example .env file
cp .env.example .env
```

Edit `.env` with your API keys:

```bash
# Required
ALPACA_API_KEY=your_key_here
ALPACA_SECRET_KEY=your_secret_here

# Optional (adds more features)
ALPHA_VANTAGE_API_KEY=your_key
FINNHUB_API_KEY=your_key
NEWSAPI_API_KEY=your_key
FRED_API_KEY=your_key

# Optional LLM providers (for sentiment analysis)
MISTRAL_API_KEY=your_key
OPENAI_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
```

### 2. Resource Configuration

The system automatically detects your hardware and configures itself. You can verify the detection:

```bash
python -c "
from src.core.system_resources import get_system_resources, create_resource_config
sr = get_system_resources()
rc = create_resource_config()
print(sr.summary())
print(f'Synthetic universes: {rc.n_synthetic_universes}')
print(f'Optuna trials: {rc.optuna_n_trials}')
print(f'n_jobs: {rc.n_jobs}')
"
```

### 3. Override Defaults (Optional)

To override auto-detected settings, modify the CONFIG dict in `src/train_robust_model.py`:

```python
CONFIG = {
    # ... existing settings ...

    # Override resource defaults
    "resource_overrides": {
        "n_synthetic_universes": 5,   # Reduce if OOM
        "n_jobs": 2,                   # Limit parallelism
        "optuna_n_trials": 20,         # Fewer HPO trials
    }
}
```

---

## First Run

### Quick Syntax Check

```bash
# Verify no syntax errors in the main training script
python -c "import ast; ast.parse(open('src/train_robust_model.py', encoding='utf-8').read()); print('OK')"
```

### Import Check

```bash
# Verify all modules load correctly
python -c "from src.experiment_config import create_default_config; print('Config OK')"
python -c "from src.core import SystemResources, ResourceConfig; print('Core OK')"
```

### Run Training

```bash
python src/train_robust_model.py
```

This will:
1. Print system resource summary (RAM, CPU, GPU, tier)
2. Download 5 years of SPY 1-minute data from Alpaca
3. Engineer 200+ features across 50 integration steps
4. Run GC between feature steps (on MEDIUM tier)
5. Reduce dimensions via Ensemble+ method
6. Train swing and timing models with Optuna HPO
7. Run robustness validation (22 stability methods)
8. Save models to `models/production/`

**Expected runtime**: 30-90 minutes depending on hardware and tier.

---

## Running Tests

### Full Test Suite

```bash
python -m pytest tests/ -q
# Expected: 2,719 passed, 0 failures
```

### Run Specific Phase Tests

```bash
# System resource tests
python -m pytest tests/test_core/test_system_resources.py -v

# Feature module tests
python -m pytest tests/test_phase_08/ -v

# Robustness tests
python -m pytest tests/test_phase_14/ -v

# Strategy tests
python -m pytest tests/test_phase_15/ -v

# Paper trading tests
python -m pytest tests/test_phase_19/ -v
```

### Run with Coverage

```bash
python -m pytest tests/ --cov=src --cov-report=html
# Open htmlcov/index.html in browser
```

---

## System Resource Detection

The pipeline adapts to your hardware automatically:

### Tier Reference

| Tier | RAM | Synthetic Universes | Optuna Trials | Stability | n_jobs | GC Between Steps |
|------|-----|--------------------:|:--------------|-----------|--------|-----------------|
| **LOW** | < 16 GB | 3 | 10 | Skip expensive | 1 | Yes |
| **MEDIUM** | 16-32 GB | 10 | 30 | All 22 methods | cores/2 | Yes |
| **HIGH** | 32-128 GB | 20 | 50 | All 22 methods | cores-1 | No |
| **ULTRA** | 128+ GB | 30 | 100 | All 22 (large N) | all cores | No |

### What Gets Skipped on LOW Tier

On systems with less than 16 GB RAM, these expensive methods are skipped:
- Combinatorial Purged CV (CPCV)
- Stability selection
- Rashomon set analysis
- Adversarial overfitting detection

These are the most memory/compute-intensive validation methods. The remaining 18 methods still provide strong robustness validation.

### GPU Detection

The system checks for GPU support in this order:
1. PyTorch CUDA (`torch.cuda.is_available()`)
2. XGBoost GPU (`tree_method='gpu_hist'`)

If a GPU is detected, XGBoost models automatically use `tree_method='gpu_hist'` on HIGH/ULTRA tiers.

---

## Training Your First Model

### Step 1: Verify Data Access

```bash
python scripts/test_alpaca_data.py
```

This tests your Alpaca API connection and downloads a small sample.

### Step 2: Run Training

```bash
python src/train_robust_model.py
```

### Step 3: Check Results

After training completes, verify:

```bash
# Check for saved models
ls models/production/

# Expected:
# spy_robust_models.joblib
# conformal_sizer.joblib  (if conformal sizing enabled)
# thompson_state.json     (if Thompson sampling enabled)
```

### Step 4: Review Performance

The training script prints a summary including:

```
═══════════════════════════════════════════════════
SWING DIRECTION MODEL
  AUC: 0.XXX
  Accuracy: XX.X%
  Precision (Up): XX%

TIMING MODEL
  AUC: 0.XXX
  Accuracy: XX.X%

COMBINED SIGNALS
  Buy signals: XX.X% of days
  Sell signals: XX.X% of days
  Buy win rate: XX.X%
═══════════════════════════════════════════════════
```

---

## Paper Trading Setup

### 1. Ensure Models are Trained

Verify `models/production/spy_robust_models.joblib` exists.

### 2. Configure Alpaca Paper Trading

Your Alpaca API keys should be for paper trading (not live). Verify at [app.alpaca.markets](https://app.alpaca.markets).

### 3. Start Paper Trading

```bash
python scripts/test_paper_trade.py
```

The trading bot:
- Runs health checks every 60 seconds
- Generates signals every hour (configurable)
- Checks all 5 trading gates before execution
- Enforces circuit breaker limits
- Records all decisions to logs

---

## Common Operations

### Run Grid Search Campaign

```bash
python scripts/run_thick_weave_search.py
```

### Check System Health

```bash
python scripts/health_check.py
```

### View Registry

```bash
python scripts/registry_cli.py list
python scripts/registry_cli.py best --metric auc
```

### Start Web Dashboard

```bash
python src/phase_20_monitoring/dashboard_server.py
# Open http://localhost:5000 in browser
```

---

## Troubleshooting

### Out of Memory (OOM)

**Symptoms**: Process killed, MemoryError, system freeze.

**Solutions**:
1. The system should auto-detect and scale down. Check your tier:
   ```bash
   python -c "from src.core.system_resources import get_system_resources; print(get_system_resources().tier)"
   ```
2. Reduce synthetic universes in CONFIG:
   ```python
   CONFIG["resource_overrides"] = {"n_synthetic_universes": 3}
   ```
3. Disable expensive features:
   ```python
   CONFIG["use_tda_homology"] = False
   CONFIG["use_network_features"] = False
   ```

### Import Errors

**Symptoms**: `ModuleNotFoundError`

**Solutions**:
1. Ensure virtual environment is activated
2. Check Python version: `python --version` (must be 3.11 or 3.12)
3. Reinstall: `pip install -r requirements.txt`

### Alpaca API Errors

**Symptoms**: `APIError`, `Forbidden`, rate limit errors

**Solutions**:
1. Verify API keys in `.env`
2. Check account status at [app.alpaca.markets](https://app.alpaca.markets)
3. Ensure using paper trading keys (not live)
4. Rate limits: The pipeline retries with exponential backoff (3 attempts)

### Slow Training

**Solutions**:
1. Check resource tier — LOW tier is significantly slower
2. Reduce Optuna trials: `CONFIG["optuna_n_trials"] = 10`
3. Use fewer CV folds: `CONFIG["cv_folds"] = 3`
4. Disable optional features (anything with `use_*` flag)

### GPU Not Detected

**Symptoms**: Training runs on CPU despite having GPU

**Solutions**:
1. Check detection:
   ```bash
   python -c "from src.core.system_resources import get_system_resources; sr = get_system_resources(); print(f'CUDA: {sr.has_cuda}, XGB GPU: {sr._xgb_gpu}')"
   ```
2. Install CUDA toolkit if using PyTorch
3. XGBoost GPU should work with CUDA installed — try:
   ```bash
   python -c "import xgboost as xgb; m = xgb.XGBClassifier(tree_method='gpu_hist'); print('GPU OK')"
   ```

---

## Next Steps

1. **Read the architecture**: [ARCHITECTURE.md](ARCHITECTURE.md) for system design
2. **Explore components**: [COMPONENTS.md](COMPONENTS.md) for module reference
3. **Review config space**: [GRID_SEARCH_DIAGRAM.md](GRID_SEARCH_DIAGRAM.md) for parameter visualization
4. **Run experiments**: Use `scripts/run_thick_weave_search.py` for systematic exploration
5. **Monitor performance**: Start the web dashboard for real-time monitoring

---

*Last updated: 2026-02-28*
