# Giga Trader

**Production-grade ML trading system for SPY swing trading with 200+ engineered features, adaptive resource scaling, and multi-layer robustness validation.**

```
     ╔═══════════════════════════════════════════════════════╗
     ║            G I G A   T R A D E R   v 1 . 1           ║
     ║                                                       ║
     ║   27 Phases  ·  210 Modules  ·  84K Lines of Code    ║
     ║   200+ Features  ·  43 Feature Groups  ·  22 Tests   ║
     ╚═══════════════════════════════════════════════════════╝
```

---

## What Is This?

Giga Trader is an end-to-end machine learning pipeline that predicts SPY (S&P 500 ETF) intraday price direction and optimal entry/exit timing. It combines:

- **Swing direction prediction** — Will SPY close up or down today?
- **Timing prediction** — Does the daily low occur before the daily high? (determines entry strategy)
- **Entry/exit timing model** — Optimal minute-level entry, exit, stop-loss, and position sizing
- **200+ engineered features** spanning technical indicators, market microstructure, alternative data, and cross-asset signals
- **22 robustness validation methods** ensuring models generalize beyond historical data
- **Adaptive resource scaling** — automatically detects system hardware (RAM, CPU, GPU) and scales the pipeline up or down

### Current Performance

| Metric | Value |
|--------|-------|
| Swing Model AUC | **0.818** |
| Timing Model AUC | **0.778** |
| Buy Signal Win Rate | **81.8%** |
| Sell Signal Win Rate | **77.3%** |
| Train-Test Gap | **~0.05** |
| Training Data | 1,260 days (5 years, 1-min bars with extended hours) |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA SOURCES                                 │
│  Alpaca API · yfinance · FRED · Alpha Vantage · Finnhub · NewsAPI  │
└─────────────┬───────────────────────────────────────┬───────────────┘
              │                                       │
              ▼                                       ▼
┌─────────────────────────┐         ┌─────────────────────────────────┐
│   DATA PREPROCESSING    │         │     SENTIMENT ANALYSIS          │
│  OHLC validation        │         │  Multi-source news collection   │
│  Information bars        │         │  LLM-based scoring (Mistral)   │
│  Extended hours merge   │         │  Market-cap weighted aggregation│
└────────────┬────────────┘         └────────────────┬────────────────┘
             │                                       │
             ▼                                       │
┌─────────────────────────────────────────────────────────────────────┐
│                    FEATURE ENGINEERING (200+)                        │
│                                                                     │
│  Technical ─── Microstructure ─── Sentiment ─── Cross-Asset        │
│  Entropy  ─── Wavelets ────────── Calendar ──── Extended Hours     │
│  Options  ─── Dark Pool ──────── Economic ──── Path Signatures     │
│  HMM      ─── Changepoint ────── Network ───── Market Structure   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              FEATURE PROCESSING & SELECTION                         │
│                                                                     │
│  Group-Aware Processor ─── Feature Neutralizer ─── Interaction     │
│  Ensemble+ Dim Reduction (MI + KernelPCA + ICA + K-Medoids)       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     MODEL TRAINING                                  │
│                                                                     │
│  Logistic (L1/L2) ─── Gradient Boosting ─── XGBoost ─── CatBoost  │
│  Stacking Ensemble ─── Quantile Forest ─── Regime Router           │
│  Optuna Bayesian HPO ─── Walk-Forward CV ─── CPCV                  │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 ROBUSTNESS VALIDATION                                │
│                                                                     │
│  22 Stability Methods ─── Synthetic SPY Universes ─── WMES Score   │
│  Knockoff Gate ─── Label Noise Test ─── FI Stability ──────────    │
│  Adversarial Detection ─── Rashomon Set ─── HP Perturbation ──     │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                STRATEGY & EXECUTION                                 │
│                                                                     │
│  Meta-Labeling ─── Conformal Sizing ─── CVaR Position Sizer       │
│  Thompson Sampling ─── Bayesian Model Averaging ─── Trading Gates  │
│  Circuit Breaker ─── Batch Entry/Exit ─── Staleness Checker       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│              PAPER TRADING & MONITORING                              │
│                                                                     │
│  Alpaca Paper Trading ─── Feature Drift Monitor ─── Health Checks  │
│  Online Model Updates ─── Web Dashboard ─── Alert System           │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Key Features

### 5 Edge-Generating Strategies

| # | Strategy | Description |
|---|----------|-------------|
| 1 | **Regularization-First** | Heavy L1/L2 regularization, max tree depth 5, simple models find real signals |
| 2 | **Extended Hours Signals** | Premarket (4AM-9:30AM) and afterhours features most algorithms miss |
| 3 | **Intraday Patterns** | Morning dip, afternoon swing, gap fade, opening range breakout detection |
| 4 | **Confidence Soft Targets** | Sigmoid/smoothed labels (0.0-1.0) instead of binary, reduces overfitting |
| 5 | **Batch Position Scaling** | Scale in/out in 3 batches to reduce timing risk and improve fill prices |

### 43 Feature Groups

Features span 10 categories:

- **Technical** — RSI, MACD, Bollinger Bands, momentum, volatility
- **Microstructure** — VPIN, dark pools, block structure, Amihud illiquidity
- **Sentiment** — News, Reddit, crypto, fear/greed, congressional trading
- **Cross-Asset** — TLT, QQQ, GLD, IWM, EEM, VXX, HYG correlations
- **Extended Hours** — Premarket/afterhours returns, volume, momentum
- **Economic** — VIX term structure, yield curve, commodities, FRED data
- **Statistical** — Entropy, Hurst exponent, wavelets, path signatures, TDA
- **Regime** — HMM, changepoint detection, Wasserstein distance, drift
- **Calendar** — Day of week, month, FOMC meetings, options expiry
- **Options** — Put/call ratio, implied volatility, gamma exposure

### Adaptive Resource Scaling

The pipeline automatically detects system hardware and scales:

| Resource | LOW (<16GB) | MEDIUM (16-32GB) | HIGH (32-128GB) | ULTRA (128GB+) |
|----------|-------------|-------------------|-----------------|----------------|
| Synthetic universes | 3 | 10 | 20 | 30 |
| Optuna trials | 10 | 30 | 50 | 100 |
| Stability methods | Skip expensive | All 22 | All 22 | All 22 (larger N) |
| n_jobs | 1 | cores/2 | cores-1 | all cores |
| GC pressure | Aggressive | Between steps | None | None |

### 22 Robustness Validation Methods

Every model must pass multi-layer validation:

1. **Tier 1** — AUC > 0.56, gap < 0.10, WMES >= 0.45, walk-forward pass
2. **Tier 2** — Multi-radius HP perturbation, stability score >= 0.60
3. **Tier 3** — Model-aware fragility < 0.40, suite composite >= 0.45

Methods include: CPCV, stability selection, Rashomon set analysis, adversarial overfitting detection, SHAP consistency, permutation importance, label noise tolerance, knockoff gates, and more.

---

## Project Structure

```
giga_trader/
├── src/                          # 210 Python modules (84K lines)
│   ├── train_robust_model.py     # Main training entry point
│   ├── giga_orchestrator.py      # Pipeline orchestration
│   ├── experiment_config.py      # 13-field configuration schema
│   ├── core/                     # Base classes, registry, resource detection
│   ├── phase_01_data_acquisition/  through  phase_27_live_trading/
│   ├── mega_ensemble/            # 5-layer ensemble pipeline
│   └── supervision/              # Trading supervision service
├── tests/                        # 122 test files, 2,700+ tests
├── config/                       # 9 YAML configuration files
├── scripts/                      # 23 utility scripts
├── data/                         # Raw, processed, synthetic data
├── models/                       # Checkpoints and production models
├── docs/                         # Architecture and component docs
├── reports/                      # Generated analysis reports
└── campaigns/                    # Grid search campaign results
```

### 27 Pipeline Phases

| Phase | Name | Files | Description |
|-------|------|-------|-------------|
| 01 | Data Acquisition | 4 | Alpaca API client, historical data, constituents |
| 02 | Preprocessing | 4 | OHLC validation, information bars, cleaning |
| 03 | Synthetic Data | 2 | Bear universes, multiscale bootstrap |
| 04 | Probabilistic | 1 | Probabilistic model support |
| 05 | Targets | 4 | Target variables, CUSUM filter, triple barrier |
| 06 | Intraday Features | 2 | Minute-level pattern features |
| 07 | Daily Features | 1 | Daily technical indicators |
| 08 | Feature Breadth | **48** | 40+ feature modules (largest phase) |
| 09 | Calendar Features | 3 | Calendar effects, feature research |
| 10 | Feature Processing | 6 | Dimensionality reduction, neutralization |
| 11 | CV Splitting | 5 | Cross-validation with purging/embargo |
| 12 | Model Training | 8 | Model types, ensembles, HPO |
| 13 | Validation | 3 | Anti-overfitting integration (50 steps) |
| 14 | Robustness | 9 | 22 stability methods, knockoff gates |
| 15 | Strategy | 14 | Signal generation, position sizing |
| 16 | Backtesting | 5 | Historical simulation |
| 17 | Monte Carlo | 1 | Stochastic simulation |
| 18 | Persistence | 5 | Model registry, grid search configs |
| 19 | Paper Trading | 12 | Trading gates, circuit breaker, risk |
| 20 | Monitoring | 6 | Health checks, drift detection, dashboard |
| 21 | Continuous | 4 | Experiment runner, online updates |
| 22 | Automation | 1 | Scheduled automation |
| 23 | Analytics | 6 | Thick weave search, advanced analytics |
| 24 | Advanced Backtest | 1 | Extended backtesting |
| 25 | Risk Management | 3 | Dynamic risk, model selection |
| 26 | Temporal/External | 4 | Temporal cascade models |
| 27 | Live Trading | 1 | Live execution interface |

---

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.11-3.12 |
| **ML** | scikit-learn, XGBoost, LightGBM, CatBoost, Optuna |
| **Data** | pandas, numpy, scipy, statsmodels |
| **Market Data** | Alpaca API, yfinance, FRED |
| **Sentiment** | Alpha Vantage, Finnhub, NewsAPI, Mistral LLM |
| **Visualization** | matplotlib, seaborn, plotly |
| **Storage** | SQLite (registry), joblib (models), YAML (config) |
| **Testing** | pytest (2,700+ tests) |
| **GPU** | XGBoost GPU acceleration (auto-detected) |

---

## Quick Start

### Prerequisites

- Python 3.11 or 3.12 (3.13+ not supported)
- Alpaca API account (paper trading)
- 16GB+ RAM recommended (adapts to available resources)

### Installation

```bash
git clone https://github.com/TaylorAmarelTech/giga-trader.git
cd giga-trader
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your API keys:
#   ALPACA_API_KEY=...
#   ALPACA_SECRET_KEY=...
#   ALPHA_VANTAGE_API_KEY=...  (optional)
#   FINNHUB_API_KEY=...        (optional)
```

### Run Training

```bash
# System auto-detects hardware and prints resource tier
python src/train_robust_model.py

# Output:
# SYSTEM RESOURCES
#   Tier: MEDIUM
#   RAM:  31.8 GB total
#   CPU:  10 physical / 16 logical cores
#   GPU:  XGBoost GPU detected
```

### Run Tests

```bash
python -m pytest tests/ -q
# 2,719 passed, 0 failures
```

See [docs/GETTING_STARTED.md](docs/GETTING_STARTED.md) for detailed setup instructions.

---

## Documentation

| Document | Description |
|----------|-------------|
| [ARCHITECTURE.md](docs/ARCHITECTURE.md) | System architecture, data flow, and design decisions |
| [COMPONENTS.md](docs/COMPONENTS.md) | Detailed reference for all 27 phases and modules |
| [GETTING_STARTED.md](docs/GETTING_STARTED.md) | Installation, configuration, and first run guide |
| [GRID_SEARCH_DIAGRAM.md](docs/GRID_SEARCH_DIAGRAM.md) | Visual parameter space (16,000+ configurations) |

---

## Configuration Space

The grid search explores **5.48 x 10^47 theoretical configurations** across 24 dimensions:

```
┌────────────────────────────────────────────────────────────┐
│  CONFIGURATION DIMENSIONS                                  │
├────────────────────────────────────────────────────────────┤
│  Entry/Exit Windows       45 combinations                  │
│  Swing Thresholds          5 values                        │
│  Model Types              44 options                       │
│  Dim Reduction Methods     7 methods × params              │
│  Feature Groups           43 toggleable groups             │
│  Anti-Overfit Params      50+ tunable settings             │
│  Risk Management          12 combinations                  │
│  CV Parameters            27 combinations                  │
│                                                            │
│  Practical: Optuna Bayesian HPO (50-100 trials)           │
│  Campaign: Thick weave search across bundles               │
└────────────────────────────────────────────────────────────┘
```

---

## Anti-Overfitting Philosophy

> "Win rates > 75% are suspicious. Every model must prove it generalizes."

The system implements a **3-tier validation gate**:

```
MODEL TRAINED
    │
    ▼
┌──────────────────────────────────┐
│  TIER 1: Basic Sanity            │
│  AUC > 0.56, AUC < 0.85         │  ← Too-good = data leakage
│  Train-test gap < 0.10           │
│  WMES >= 0.45                    │
│  Walk-forward passed             │
└──────────────┬───────────────────┘
               │ PASS
               ▼
┌──────────────────────────────────┐
│  TIER 2: Stability Analysis      │
│  Multi-radius HP perturbation    │
│  Stability score >= 0.60         │
│  22 independent stability tests  │
└──────────────┬───────────────────┘
               │ PASS
               ▼
┌──────────────────────────────────┐
│  TIER 3: Fragility Detection     │
│  Model-aware fragility < 0.40    │
│  Suite composite >= 0.45         │
│  AUC >= 0.58                     │
└──────────────┬───────────────────┘
               │ PASS
               ▼
         PRODUCTION READY
```

---

## Data Flow

```
Alpaca 1-min bars (5 years, extended hours)
    │
    ├── OHLC Validation (High >= max(O,C), etc.)
    ├── Information Bars (dollar/volume/tick bars)
    ├── Extended Hours Merge (4AM-8PM ET)
    │
    ▼
200+ Raw Features
    │
    ├── Group-Aware Selection (43 feature groups)
    ├── Feature Neutralization (sector/market removal)
    ├── Ensemble+ Reduction (MI + KernelPCA + ICA + K-Medoids)
    │
    ▼
~30-50 Robust Features
    │
    ├── Walk-Forward CV (purge 5 days, embargo 2 days)
    ├── Synthetic SPY Universes (10-30 alternative histories)
    ├── Optuna Bayesian HPO (30-100 trials)
    │
    ▼
Validated Models → Paper Trading → Live Trading
```

---

## License

This project is proprietary. All rights reserved.

---

## Author

**Taylor Amarel** — [TaylorAmarelTech](https://github.com/TaylorAmarelTech)
