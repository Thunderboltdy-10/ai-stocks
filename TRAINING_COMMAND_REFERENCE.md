# Training Command Reference (GPU-First)

Last updated: 2026-02-23

This is the current, working command set for the GBM-first pipeline.

## 1) Environment (always first)

```bash
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate ai-stocks
cd /home/thunderboltdy/ai-stocks/python-ai-service
```

## 2) Verify GPU is actually used

```bash
python - <<'PY'
import xgboost as xgb, numpy as np
X=np.random.randn(256,8).astype('float32')
y=np.random.randn(256).astype('float32')
m=xgb.XGBRegressor(device='cuda',tree_method='hist',n_estimators=8,objective='reg:squarederror')
m.fit(X,y,verbose=False)
print('xgboost_cuda_ok')
PY

nvidia-smi --query-gpu=name,utilization.gpu,memory.used --format=csv,noheader
```

## 3) Train one symbol (example: AAPL)

```bash
python -m training.train_gbm AAPL --overwrite --n-trials 10 --no-lgb --target-horizon 1 --max-features 50
```

Notes:
- `--no-lgb` is intentional here because LightGBM GPU backend is unavailable in this environment.
- XGBoost uses CUDA (`device=cuda`) and remains the primary model.

## 4) Backtest one symbol (long + short windows)

Long window:

```bash
python run_backtest.py --symbol AAPL --start 2020-01-01 --end 2024-12-31
```

Short window (weeks) with warmup enabled by default:

```bash
python run_backtest.py --symbol AAPL --start 2024-11-15 --end 2024-12-31 --warmup-days 252 --min-eval-days 20
```

## 5) Full 5-symbol train loop

```bash
for s in AAPL XOM JPM KO TSLA; do
  python -m training.train_gbm "$s" --overwrite --n-trials 10 --no-lgb --target-horizon 1 --max-features 50
  python - <<'PY' "$s"
import json,sys
sym=sys.argv[1]
p=f'saved_models/{sym}/gbm/training_metadata.json'
with open(p) as f:m=json.load(f)
h=m['holdout']['ensemble']
print(sym,'dir_acc',round(h['dir_acc'],4),'pred_std',round(h['pred_std'],6),'pos',round(h['positive_pct'],3),'wfe',round(m['wfe'],1),'gpu',m['runtime']['xgb_gpu_enabled'])
PY
done
```

## 6) Full 5-symbol multi-window backtest matrix

```bash
python - <<'PY'
from pathlib import Path
import pandas as pd
from run_backtest import BacktestConfig, UnifiedBacktester

symbols=['AAPL','XOM','JPM','KO','TSLA']
windows=[
  ('2020-01-01','2024-12-31','long_5y'),
  ('2023-01-01','2024-12-31','mid_2y'),
  ('2024-10-01','2024-12-31','short_q4_2024'),
  ('2024-11-15','2024-12-31','short_7w_2024'),
]
rows=[]
for sym in symbols:
  for start,end,label in windows:
    cfg=BacktestConfig(symbol=sym,start=start,end=end,warmup_days=252,min_eval_days=20)
    r=UnifiedBacktester(cfg).run()
    r['window']=label
    rows.append(r)

df=pd.DataFrame(rows)
out=Path('experiments')/f'multiwindow_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
df.to_csv(out,index=False)
print('saved:',out)
print(df[['symbol','window','alpha','strategy_return','buy_hold_return','sharpe_ratio','n_days']])
PY
```

## 7) API smoke test

```bash
python app.py
# New terminal:
curl http://localhost:8000/api/health
curl -X POST http://localhost:8000/api/predict -H "Content-Type: application/json" -d '{"symbol":"AAPL","horizon":10,"daysOnChart":180,"fusion":{"mode":"weighted","regressorScale":15,"buyThreshold":0.3,"sellThreshold":0.45,"regimeFilters":{"bull":true,"bear":true}}}'
```

## 8) Current architecture summary

- Training: `training/train_gbm.py` (Optuna + SHAP feature selection + purged CV)
- Inference: GBM predictions + strict quality gate + regime fallback
- Backtest: `evaluation/execution_backtester.py` (inventory-aware, no impossible sells)
- CLI: `run_backtest.py` supports warmup-aware short-window evaluation
- Service: `service/prediction_service.py` returns prediction + backtest + forward simulation
