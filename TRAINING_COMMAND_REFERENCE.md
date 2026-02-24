# Training Command Reference (GPU + Intraday)

Last updated: 2026-02-24

## 1) Start Session

```bash
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate ai-stocks
cd /home/thunderboltdy/ai-stocks/python-ai-service
```

## 2) Verify GPU Is Active

```bash
python - <<'PY'
import xgboost as xgb, numpy as np
X=np.random.randn(256,8).astype('float32')
y=np.random.randn(256).astype('float32')
m=xgb.XGBRegressor(device='cuda',tree_method='hist',n_estimators=8,objective='reg:squarederror')
m.fit(X,y,verbose=False)
print('xgboost_cuda_ok')
PY

nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader
```

## 3) Train Daily GBM (baseline)

```bash
python -m training.train_gbm AAPL \
  --overwrite --n-trials 10 --no-lgb \
  --target-horizon 1 --max-features 55 \
  --data-period max --data-interval 1d
```

## 4) Train Intraday Hourly GBM (day-trading profile)

```bash
python -m training.train_gbm AAPL \
  --overwrite --n-trials 10 --no-lgb \
  --target-horizon 1 --max-features 55 \
  --data-period 730d --data-interval 1h \
  --model-suffix intraday_1h_v5
```

Artifacts:
- daily: `saved_models/{SYMBOL}/gbm/`
- intraday: `saved_models/{SYMBOL}/gbm_intraday_1h_v5/`

## 5) Backtest Daily

```bash
python run_backtest.py \
  --symbol AAPL \
  --start 2020-01-01 --end 2024-12-31 \
  --max-long 1.6 --max-short 0.25 \
  --warmup-days 252 --min-eval-days 20
```

## 6) Backtest Intraday (flat-at-day-end auto-enabled)

```bash
python run_backtest.py \
  --symbol AAPL \
  --model-variant auto_intraday \
  --data-period 730d --data-interval 1h \
  --start "2025-01-01 00:00:00" --end "2026-02-20 16:00:00" \
  --max-long 1.4 --max-short 0.2 \
  --commission 0.0001 --slippage 0.0001 \
  --warmup-days 40 --min-eval-days 10
```

Notes:
- `commission`/`slippage` here are decimal percentages (1 bps = `0.0001`).
- API `/ai` page fields are in bps-style; backend converts to decimal safely.

## 7) Run Intraday Matrix (10-Symbol Universe, train + backtest)

```bash
python scripts/run_intraday_hourly_matrix.py \
  --symbol-set intraday10 \
  --n-trials 3 --max-features 55 \
  --period 730d --interval 1h \
  --model-suffix intraday_1h_v5 \
  --target-horizon 1 \
  --max-short 0.2 \
  --commission 0.0001 --slippage 0.0001 \
  --overwrite
```

No retrain (reuse existing artifacts):

```bash
python scripts/run_intraday_hourly_matrix.py \
  --symbol-set intraday10 \
  --period 730d --interval 1h \
  --model-suffix intraday_1h_v5 \
  --target-horizon 1 \
  --max-short 0.2 \
  --commission 0.0001 --slippage 0.0001
```

## 8) Run Daily Matrix (15-Symbol Universe, train + backtest)

```bash
python scripts/run_daily_matrix.py \
  --symbol-set daily15 \
  --n-trials 2 --max-features 50 \
  --period max --interval 1d \
  --model-suffix daily_v4 \
  --max-long 1.6 --max-short 0.6 \
  --commission 0.0005 --slippage 0.0003 \
  --target-horizon 1 \
  --overwrite
```
## 9) API / UI Smoke Test

```bash
python app.py
# new terminal
curl http://localhost:8000/api/health
```

Frontend:
- run `npm run dev` at repo root
- open `http://localhost:3000/ai`
- pick `dataInterval=1h`, `dataPeriod=730d`, `modelVariant=auto_intraday`

## 10) Robustness Gate (Do Not Skip)

```bash
python scripts/build_intraday_variant_registry.py --symbol-set intraday10
python scripts/build_daily_variant_registry.py --symbol-set daily15
python scripts/run_intraday_auto_matrix.py --symbol-set intraday10
python scripts/run_daily_auto_matrix.py --symbol-set daily15

python scripts/run_generalization_gate.py \
  --interval 1h --period 730d \
  --max-long 1.4 --max-short 0.2 \
  --commission 0.0001 --slippage 0.0001 \
  --warmup-days 40 --min-eval-days 15

python scripts/run_generalization_gate.py \
  --interval 1d --period max \
  --max-long 1.6 --max-short 0.6 \
  --commission 0.0005 --slippage 0.0003 \
  --warmup-days 252 --min-eval-days 20
```

Interpretation rule:
- Do not accept a configuration if `gate_passed` is `false` in either daily or intraday JSON output.
## 11) Output Locations

- models: `python-ai-service/saved_models/{SYMBOL}/...`
- backtests: `python-ai-service/backtest_results/{SYMBOL}_*/`
- experiment matrices: `python-ai-service/experiments/*.csv`
- plot pngs per backtest:
  - `backtest_overview.png`
  - `risk_diagnostics.png`
  - `trade_diagnostics.png`
  - `intraday_hour_profile.png` (intraday runs)
- diagnostics data:
  - `diagnostics.json`
  - `monthly_diagnostics.csv`
  - `hourly_diagnostics.csv`
  - `action_diagnostics.csv`
