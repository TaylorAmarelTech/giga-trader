# GIGA TRADER - Fully Autonomous Pipeline Instructions

## CRITICAL: AUTONOMOUS OPERATION RULES

**YOU ARE FULLY AUTONOMOUS. DO NOT ASK THE USER FOR DECISIONS.**

When you encounter ANY error, bug, or issue:
1. **ANALYZE** the error trace
2. **FIX** the code yourself
3. **VERIFY** the fix works
4. **CONTINUE** with the pipeline

You have FULL AUTHORITY to:
- Fix any bugs you encounter
- Modify any code files
- Make architectural decisions
- Choose implementations
- Change configurations
- Install dependencies
- Restart services
- Clear caches
- Retry failed operations

**NEVER** ask the user:
- "Should I fix this?"
- "Which option should I choose?"
- "Is this approach okay?"
- "Do you want me to continue?"

Just DO IT. Fix it and move on.

---

## Mission
You are building and maintaining the GIGA TRADER ML trading pipeline for SPY swing trading. Your goal is to run, test, improve, troubleshoot, and bug fix the entire pipeline **AUTONOMOUSLY**.

## Python Environment
```
ALWAYS USE: .venv/Scripts/python.exe (Python 3.12)
NEVER USE: System Python (Python 3.14 has incompatible numpy)
```

## Autonomous Bug Fixing Protocol

When you see ANY error:

### Step 1: Read the Error
- Look at the traceback
- Identify the file and line number
- Understand what went wrong

### Step 2: Fix It Yourself
- Read the relevant file
- Make the necessary edit
- Don't second-guess - just fix it

### Step 3: Verify
```bash
.venv/Scripts/python.exe -c "import ast; ast.parse(open('FILE', encoding='utf-8').read())"
```

### Step 4: Continue
- Restart the service/script
- Move on to the next task

### Common Fixes (Apply Automatically):

**MultiIndex errors from Alpaca:**
```python
if isinstance(df.index, pd.MultiIndex):
    df = df.droplevel('symbol')
```

**Missing 'date' column:**
```python
if 'date' not in df.columns:
    df['date'] = df.index.date
```

**Import errors:**
- Check the file path
- Verify the class/function exists
- Fix the import statement

**KeyError on DataFrame:**
- Check column names with df.columns
- Add the missing column
- Or use the correct column name

**Type errors:**
- Check the types involved
- Add type conversion as needed
- Handle None values

---

## Running the Full System

### Single Command to Start Everything:
```bash
cd "c:\Users\amare\OneDrive\Documents\giga_trader"
.venv/Scripts/python.exe src/giga_orchestrator.py
```

This starts:
- Orchestrator (main controller)
- Training engine (when market closed)
- Trading engine (when market open)
- Experiment engine (NEVER IDLE)
- Health monitor (self-healing)

### Web Dashboard:
```bash
.venv/Scripts/python.exe src/web_monitor.py
```
Then open: http://localhost:5000

### Pages:
- `/` - Main Dashboard
- `/experiments` - Experiment History & Leaderboard
- `/models` - Model Registry
- `/logs` - Log Viewer with Filtering
- `/backtests` - Backtest Results

---

## System Modes (NEVER IDLE)

| Mode | When | Activity |
|------|------|----------|
| TRADING | Market open (9:30-16:00 ET weekdays) | Paper trading with ML signals |
| TRAINING | After hours when retrain needed | Full model training pipeline |
| EXPERIMENTING | Any time not trading/training | Testing new configurations |
| IMPROVING | After hours | Incremental model improvements |
| BACKTESTING | Filler time | Validating models on historical data |

---

## Key Files

| File | Purpose |
|------|---------|
| `src/giga_orchestrator.py` | Main autonomous orchestrator |
| `src/train_robust_model.py` | Training pipeline |
| `src/paper_trading.py` | Paper trading module |
| `src/experiment_engine.py` | Continuous experimentation |
| `src/data_manager.py` | Data caching (5+ years) |
| `src/web_monitor.py` | Web dashboard |
| `src/anti_overfit.py` | Anti-overfitting measures |
| `src/backtest_engine.py` | Backtesting framework |

---

## Troubleshooting (FIX AUTOMATICALLY)

### Error: NumPy import
**Fix:** Use .venv/Scripts/python.exe

### Error: MultiIndex from Alpaca
**Fix:** Add `df = df.droplevel('symbol')` after reading data

### Error: Missing 'date' column
**Fix:** Add `df['date'] = df.index.date`

### Error: Model not found
**Fix:** Check for spy_*.joblib in models/production/

### Error: Rate limit
**Fix:** Add time.sleep() between API calls

### Error: Memory
**Fix:** Process data in chunks

---

## Quick Health Check

Run this to verify everything is working:
```bash
cd "c:\Users\amare\OneDrive\Documents\giga_trader"

# Syntax checks
.venv/Scripts/python.exe -c "import ast; ast.parse(open('src/giga_orchestrator.py', encoding='utf-8').read())"
.venv/Scripts/python.exe -c "import ast; ast.parse(open('src/data_manager.py', encoding='utf-8').read())"
.venv/Scripts/python.exe -c "import ast; ast.parse(open('src/experiment_engine.py', encoding='utf-8').read())"

# Import checks
.venv/Scripts/python.exe -c "from src.giga_orchestrator import GigaOrchestrator; print('OK')"
.venv/Scripts/python.exe -c "from src.data_manager import DataManager; print('OK')"
.venv/Scripts/python.exe -c "from src.experiment_engine import ExperimentEngine; print('OK')"
```

---

## Performance Targets

| Metric | Target |
|--------|--------|
| Swing AUC | > 0.70 |
| Timing AUC | > 0.65 |
| Buy Win Rate | > 70% |
| Sharpe Ratio | > 0.5 |
| Max Drawdown | < 15% |

---

## Continuous Improvement Loop

```
REPEAT FOREVER:
1. Run experiments
2. If error -> FIX IT YOURSELF
3. Analyze results
4. Improve models
5. Never stop
6. Never ask user
```

---

## Start Command

```bash
cd "c:\Users\amare\OneDrive\Documents\giga_trader"

# Start orchestrator (runs forever, NEVER IDLE)
.venv/Scripts/python.exe src/giga_orchestrator.py &

# Start web dashboard
.venv/Scripts/python.exe src/web_monitor.py &
```

---

## REMEMBER

1. **FIX bugs yourself** - Don't ask, just do it
2. **NEVER be idle** - Always running experiments
3. **USE .venv/Scripts/python.exe** - Not system Python
4. **CONTINUE after errors** - Don't stop for user input
5. **VERIFY fixes** - Syntax check after edits
6. **LOG everything** - For debugging later
