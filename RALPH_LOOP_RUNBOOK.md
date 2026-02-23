# Ralph Loop Runbook

Last updated: 2026-02-23

This runbook defines the working iteration loop for this project.

## Core loop

1. Activate env and verify GPU.
2. Train target symbols (GPU XGBoost).
3. Backtest immediately across multiple windows (long + short).
4. Compare against buy-and-hold and inspect weak symbols/windows.
5. Change one subsystem at a time.
6. Re-run the same matrix to verify improvement.
7. Keep only changes that survive cross-symbol checks.

## Non-negotiables

- Always:

```bash
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate ai-stocks
cd /home/thunderboltdy/ai-stocks/python-ai-service
```

- GPU must be active when training (`xgboost_cuda_ok` + `nvidia-smi`).
- Use out-of-sample windows; do not trust one window.
- Keep inventory constraints in execution simulation (no impossible sells).

## Current production path

- Model: GBM-first (`training/train_gbm.py`)
- Inference blend:
  - strict ML quality gate (`model_quality_gate_strict`)
  - regime fallback (`regime_exposure_from_prices`)
  - optional ML overlay when gate passes
- Backtester:
  - `LongOnlyExecutionBacktester`
  - commission/slippage/margin costs
  - long-only inventory barrier
- Short-window support:
  - `run_backtest.py` now uses warmup history and minimum evaluation rows

## Working baseline command set

- Train 5 symbols:

```bash
for s in AAPL XOM JPM KO TSLA; do
  python -m training.train_gbm "$s" --overwrite --n-trials 10 --no-lgb --target-horizon 1 --max-features 50
done
```

- Multi-window matrix:

```bash
python - <<'PY'
from run_backtest import BacktestConfig, UnifiedBacktester
symbols=['AAPL','XOM','JPM','KO','TSLA']
windows=[('2020-01-01','2024-12-31'),('2023-01-01','2024-12-31'),('2024-10-01','2024-12-31'),('2024-11-15','2024-12-31')]
for s in symbols:
  for start,end in windows:
    print(UnifiedBacktester(BacktestConfig(symbol=s,start=start,end=end,warmup_days=252,min_eval_days=20)).run())
PY
```

## Latest accepted iteration notes

- Added warmup-aware short-window backtesting (`warmup_days`, `min_eval_days`).
- Added inventory-aware execution constraints with trade-level notes.
- Preserved GPU-first training path (`--no-lgb` + XGBoost CUDA).
- Rebuilt `/ai` frontend page to expose:
  - prediction chart with trade markers,
  - backtest equity vs buy-and-hold,
  - forward simulation chart,
  - trade log details and execution notes.

## Rejection criteria for new ideas

Reject a change if it improves one symbol/window but materially degrades:
- AAPL long window,
- cross-symbol median alpha,
- or short-window stability.

All accepted changes must survive the same matrix comparison.
