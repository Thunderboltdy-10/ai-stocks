# Manual Train/Test Playbook

This is the fastest manual workflow for daily + intraday validation.

## 1) Start Session

```bash
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate ai-stocks
cd /home/thunderboltdy/ai-stocks/python-ai-service
```

## 2) Confirm GPU

```bash
nvidia-smi
python - <<'PY'
import xgboost as xgb, numpy as np
X=np.random.randn(256,8).astype('float32')
y=np.random.randn(256).astype('float32')
m=xgb.XGBRegressor(device='cuda',tree_method='hist',n_estimators=8,objective='reg:squarederror')
m.fit(X,y,verbose=False)
print("xgboost_cuda_ok")
PY
```

## 3) Train Daily Model

```bash
python -m training.train_gbm AAPL \
  --overwrite --n-trials 10 --no-lgb \
  --target-horizon 1 --max-features 55 \
  --data-period max --data-interval 1d
```

## 4) Train Intraday Hourly Model

```bash
python -m training.train_gbm AAPL \
  --overwrite --n-trials 10 --no-lgb \
  --target-horizon 1 --max-features 55 \
  --data-period 730d --data-interval 1h \
  --model-suffix intraday_1h_v5
```

## 5) Validate Training Metadata

```bash
python - <<'PY'
import json
for p in [
  "saved_models/AAPL/gbm/training_metadata.json",
  "saved_models/AAPL/gbm_intraday_1h_v5/training_metadata.json",
]:
  with open(p) as f:
    m=json.load(f)
  h=m["holdout"]["ensemble"]
  print("\\n", p)
  print("dir_acc:", round(h["dir_acc"],4), "pred_std:", round(h["pred_std"],6), "wfe:", round(m["wfe"],2))
  print("runtime:", m.get("runtime",{}))
PY
```

## 6) Backtest Daily

```bash
python run_backtest.py \
  --symbol AAPL \
  --start 2020-01-01 --end 2024-12-31 \
  --max-long 1.6 --max-short 0.25 \
  --warmup-days 252 --min-eval-days 20
```

## 7) Backtest Intraday

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

## 8) Train + Test Intraday Universe (10 Symbols)

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

## 9) Build Variant Registries + Auto Matrices

```bash
python scripts/build_intraday_variant_registry.py --symbol-set intraday10
python scripts/build_daily_variant_registry.py --symbol-set daily15
python scripts/run_intraday_auto_matrix.py --symbol-set intraday10
python scripts/run_daily_auto_matrix.py --symbol-set daily15
```

## 10) Train + Test Daily Universe (15 Symbols)

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

## 11) Generalization Gate (Required)

```bash
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

Gate outputs:
- `experiments/generalization_gate_intraday_*.json`
- `experiments/generalization_gate_daily_*.json`
- `experiments/generalization_matrix_intraday_*.csv`
- `experiments/generalization_matrix_daily_*.csv`

## 12) Launch API + Frontend

Terminal 1:

```bash
cd /home/thunderboltdy/ai-stocks/python-ai-service
python app.py
```

Terminal 2:

```bash
cd /home/thunderboltdy/ai-stocks
npm run dev
```

Open `http://localhost:3000/ai` and set:
- `Interval`: `1h`
- `Period`: `730d`
- `Model Variant`: `auto_intraday`

## 13) Where Files Go

- models: `python-ai-service/saved_models/`
- backtest folders + `backtest_overview.png`: `python-ai-service/backtest_results/`
- matrix csv outputs: `python-ai-service/experiments/`
- extra diagnostics in each backtest folder:
  - `diagnostics.json`
  - `risk_diagnostics.png`
  - `trade_diagnostics.png`
  - `intraday_hour_profile.png` (intraday only)
