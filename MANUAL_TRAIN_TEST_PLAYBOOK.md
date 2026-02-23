# Manual Train/Test Playbook

If you want to run the full workflow manually with minimal friction, use this file.

## 0) One-time

Make sure NVIDIA driver and CUDA are visible:

```bash
nvidia-smi
```

## 1) Start every session like this

```bash
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate ai-stocks
cd /home/thunderboltdy/ai-stocks/python-ai-service
```

## 2) Quick GPU check

```bash
python - <<'PY'
import xgboost as xgb, numpy as np
X=np.random.randn(256,8).astype('float32')
y=np.random.randn(256).astype('float32')
m=xgb.XGBRegressor(device='cuda',tree_method='hist',n_estimators=8,objective='reg:squarederror')
m.fit(X,y,verbose=False)
print('xgboost_cuda_ok')
PY
```

## 3) Train one symbol

```bash
python -m training.train_gbm AAPL --overwrite --n-trials 10 --no-lgb --target-horizon 1 --max-features 50
```

Inspect quality quickly:

```bash
python - <<'PY'
import json
p='saved_models/AAPL/gbm/training_metadata.json'
with open(p) as f:m=json.load(f)
h=m['holdout']['ensemble']
print('dir_acc',h['dir_acc'])
print('pred_std',h['pred_std'])
print('positive_pct',h['positive_pct'])
print('wfe',m['wfe'])
print('gpu',m['runtime'])
PY
```

## 4) Backtest one symbol (long window)

```bash
python run_backtest.py --symbol AAPL --start 2020-01-01 --end 2024-12-31 --warmup-days 252 --min-eval-days 20
```

## 5) Backtest one symbol (short/weeks window)

```bash
python run_backtest.py --symbol AAPL --start 2024-11-15 --end 2024-12-31 --warmup-days 252 --min-eval-days 20
```

## 6) Train 5 diverse symbols

```bash
for s in AAPL XOM JPM KO TSLA; do
  python -m training.train_gbm "$s" --overwrite --n-trials 10 --no-lgb --target-horizon 1 --max-features 50
done
```

## 7) Run 5-symbol multi-window matrix

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
out=Path('experiments')/f'manual_matrix_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.csv'
df.to_csv(out,index=False)
print('saved',out)
print(df[['symbol','window','alpha','strategy_return','buy_hold_return','sharpe_ratio','n_days']])
PY
```

## 8) Open API + frontend to inspect visually

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

Then open `http://localhost:3000/ai`.

## 9) Where outputs are

- Model artifacts: `python-ai-service/saved_models/{SYMBOL}/gbm/`
- Backtest artifacts: `python-ai-service/backtest_results/{SYMBOL}_*/`
- Experiment CSV summaries: `python-ai-service/experiments/`

## 10) Debug checklist

- Training slow or CPU-like:
  - verify `conda activate ai-stocks`
  - run `nvidia-smi`
  - run GPU smoke command above
- Backtest fails on short ranges:
  - increase `warmup-days`
  - confirm `min-eval-days` <= available trading days
- Weird execution behavior:
  - inspect `trade_log.csv` in latest backtest folder
  - check `blocked_reason` and `notes` fields
