# Training Command Reference (GBM-First, GPU)

Last updated: 2026-02-23

This file replaces the old multi-architecture command set.

## 1) Environment

Always run Python commands from `python-ai-service/` and activate the env exactly:

```bash
eval "$(/home/thunderboltdy/miniconda3/bin/conda shell.bash hook)"
conda activate ai-stocks
cd /home/thunderboltdy/ai-stocks/python-ai-service
```

## 2) Verify GPU

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

## 3) Train AAPL (recommended)

GPU-only XGBoost training (LightGBM disabled if GPU backend is unavailable):

```bash
python -m training.train_gbm AAPL --overwrite --n-trials 10 --no-lgb --target-horizon 5 --max-features 50
```

## 4) Backtest AAPL

```bash
python run_backtest.py --symbol AAPL --start 2020-01-01 --end 2024-12-31
```

Volatility-targeted mode is on by default (`--vol-target-annual 0.25`).
You can tune it explicitly:

```bash
python run_backtest.py --symbol AAPL --start 2020-01-01 --end 2024-12-31 --vol-target-annual 0.25
python run_backtest.py --symbol AAPL --start 2020-01-01 --end 2024-12-31 --vol-target-annual 0.0   # disable
```

## 5) Train/Backtest second symbol (robustness)

Example with XOM:

```bash
python -m training.train_gbm XOM --overwrite --n-trials 12 --no-lgb --target-horizon 1 --max-features 50
python run_backtest.py --symbol XOM --start 2020-01-01 --end 2024-12-31
```

## 6) Robustness sweep (AAPL)

Runs multiple windows + cost settings and writes a CSV summary:

```bash
python - <<'PY'
import pandas as pd
from pathlib import Path
from run_backtest import BacktestConfig, UnifiedBacktester

rows=[]
windows=[('2010-01-01','2014-12-31'),('2015-01-01','2019-12-31'),('2020-01-01','2024-12-31'),('2010-01-01','2024-12-31')]
costs=[(0.0005,0.0003),(0.0010,0.0005)]
for start,end in windows:
    for com,slip in costs:
        cfg=BacktestConfig(symbol='AAPL',start=start,end=end,commission_pct=com,slippage_pct=slip)
        r=UnifiedBacktester(cfg).run()
        r['commission']=com
        r['slippage']=slip
        rows.append(r)
df=pd.DataFrame(rows)
out=Path('backtest_results')/f"AAPL_robustness_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
df.to_csv(out,index=False)
print(out)
print(df[['start','end','commission','slippage','strategy_return','buy_hold_return','alpha','sharpe_ratio','max_drawdown']])
PY
```

## 7) API smoke test

```bash
python app.py
# in another terminal:
curl http://localhost:8000/api/health
curl -X POST http://localhost:8000/api/predict -H "Content-Type: application/json" -d '{"symbol":"AAPL","horizon":5,"daysOnChart":120}'
```

## 8) Key artifacts

- Models: `python-ai-service/saved_models/{SYMBOL}/gbm/`
- Backtests: `python-ai-service/backtest_results/`
- Metadata: `python-ai-service/saved_models/{SYMBOL}/gbm/training_metadata.json`

## 9) Safety behavior

- Backtest/prediction pipeline applies a model-quality gate using holdout diagnostics.
- If diagnostics are below minimum thresholds, positions fall back to passive long exposure instead of forcing weak signals.
