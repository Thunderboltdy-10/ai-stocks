# Ralph Loop Runbook

Last updated: 2026-02-23

This is the practical workflow to iterate autonomously and manually reproduce results.

## Core idea

1. Train one symbol with GPU.
2. Backtest immediately.
3. Diagnose failures (bias, variance collapse, weak direction, overfit).
4. Change one thing at a time.
5. Re-train and re-backtest.
6. Keep the best configuration and log it.

## Hard rules

- Always activate env first:

```bash
eval "$(/home/thunderboltdy/miniconda3/bin/conda shell.bash hook)"
conda activate ai-stocks
cd /home/thunderboltdy/ai-stocks/python-ai-service
```

- Use GPU (`xgboost_cuda_ok` + `nvidia-smi`).
- Never trust one run. Validate on multiple windows and cost settings.
- Preserve useful legacy blueprints (sentiment, other modules) even if inactive.

## Iteration template

### Step A: Train

```bash
python -m training.train_gbm AAPL --overwrite --n-trials 10 --no-lgb --target-horizon 5 --max-features 50
```

### Step B: Inspect training metadata

```bash
python - <<'PY'
import json
p='saved_models/AAPL/gbm/training_metadata.json'
with open(p) as f:
    m=json.load(f)
print('dir_acc',m['holdout']['ensemble']['dir_acc'])
print('pred_std',m['holdout']['ensemble']['pred_std'])
print('positive_pct',m['holdout']['ensemble']['positive_pct'])
print('wfe',m['wfe'])
print('gpu',m['runtime'])
PY
```

### Step C: Backtest

```bash
python run_backtest.py --symbol AAPL --start 2020-01-01 --end 2024-12-31
```

### Step D: Robustness check

Use the sweep command in `TRAINING_COMMAND_REFERENCE.md`.

## What to change when metrics are bad

- `pred_std` too low: raise model capacity or reduce regularization.
- `positive_pct` skewed: rebalance objective / sample weighting.
- Holdout < walk-forward: reduce overfit, lower complexity, shorten feature set.
- Backtest alpha weak but Sharpe okay: tune position sizing/risk caps.
- Good single-window result but poor robustness: reject and continue iterating.

## Current baseline (2026-02-23, latest)

- AAPL, 2020-01-01 to 2024-12-31:
  - `strategy_return`: 2.8408
  - `buy_hold_return`: 2.4400
  - `alpha`: +0.4007
  - `sharpe`: 0.9149

- AAPL robustness sweep also recorded strong outperformance in older windows; see
  recent `python-ai-service/backtest_results/AAPL_20260223_190734_791858/summary.json`,
  `python-ai-service/backtest_results/AAPL_20260223_190735_011651/summary.json`,
  `python-ai-service/backtest_results/AAPL_20260223_190735_627113/summary.json`,
  and `python-ai-service/backtest_results/AAPL_20260223_190735_878992/summary.json`.
