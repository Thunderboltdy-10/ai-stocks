# AI Stocks Command Reference

Quick reference for training, backtesting, and running the AI trading system.

---

## 1. Start Session

```bash
# Activate conda environment
eval "$($HOME/miniconda3/bin/conda shell.bash hook)"
conda activate ai-stocks
cd /home/thunderboltdy/ai-stocks/python-ai-service
```

---

## 2. Verify GPU

```bash
# Check GPU is available
nvidia-smi

# Quick XGBoost GPU test
python - <<'PY'
import xgboost as xgb, numpy as np
X=np.random.randn(256,8).astype('float32')
y=np.random.randn(256).astype('float32')
m=xgb.XGBRegressor(device='cuda',tree_method='hist',n_estimators=8,objective='reg:squarederror')
m.fit(X,y,verbose=False)
print('xgboost_cuda_ok')
PY
```

---

## 3. Train Daily Model

Trains a model using daily data (1 candle = 1 day). Best for swing trading.

```bash
python -m training.train_gbm AAPL \
  --overwrite --n-trials 10 --no-lgb \
  --target-horizon 1 --max-features 55 \
  --data-period max --data-interval 1d
```

**What each flag means:**
- `--overwrite` - Replace existing model
- `--n-trials 10` - Try 10 hyperparameter combinations (more = better but slower)
- `--no-lgb` - Use only XGBoost (faster than XGB+LGB)
- `--target-horizon 1` - Predict 1 day ahead
- `--max-features 55` - Use top 55 features
- `--data-period max` - Use all available history
- `--data-interval 1d` - Daily candles

**Output:** `saved_models/{SYMBOL}/gbm/`

**Training time:** ~10-30 minutes with GPU

---

## 4. Train Intraday Hourly Model

Trains a model using hourly data. Best for day trading.

```bash
python -m training.train_gbm AAPL \
  --overwrite --n-trials 10 --no-lgb \
  --target-horizon 1 --max-features 55 \
  --data-period 730d --data-interval 1h \
  --model-suffix intraday_1h_v5
```

**What each flag means:**
- `--data-period 730d` - Use 730 days of history (2 years)
- `--data-interval 1h` - Hourly candles
- `--model-suffix intraday_1h_v5` - Name suffix to distinguish from daily model

**Output:** `saved_models/{SYMBOL}/gbm_intraday_1h_v5/`

**Training time:** ~5-15 minutes with GPU

---

## 5. Backtest Daily

Run backtest using daily model on historical data.

```bash
python run_backtest.py \
  --symbol AAPL \
  --start 2020-01-01 --end 2024-12-31 \
  --max-long 1.6 --max-short 0.25 \
  --warmup-days 252 --min-eval-days 20
```

**Parameters:**
- `--start` / `--end` - Backtest date range
- `--max-long 1.6` - Max 160% long position (1.6x capital)
- `--max-short 0.25` - Max 25% short position
- `--warmup-days 252` - Days needed for model warmup (1 year)
- `--min-eval-days 20` - Minimum days to evaluate

**Output location:** `backtest_results/AAPL_*/`

---

## 6. Backtest Intraday

Run backtest using hourly model.

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

**Parameters:**
- `--commission 0.0001` - 1 basis point = 0.01% (very low cost)
- `--slippage 0.0001` - 1 basis point slippage
- `--warmup-days 40` - 40 hours for warmup
- `--flat-at-day-end` - Auto-enable for intraday

---

## 7. Run Full Universe (Multiple Symbols)

### Daily Universe (15 symbols)

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

### Intraday Universe (10 symbols)

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

---

## 8. Validate Model Quality

```bash
python - <<'PY'
import json
for p in [
  "saved_models/AAPL/gbm/training_metadata.json",
  "saved_models/AAPL/gbm_intraday_1h_v5/training_metadata.json",
]:
  try:
    with open(p) as f:
      m=json.load(f)
    h=m["holdout"]["ensemble"]
    print("\n", p)
    print("dir_acc:", round(h["dir_acc"],4), "pred_std:", round(h["pred_std"],6), "wfe:", round(m["wfe"],2))
  except Exception as e:
    print(f"Error reading {p}: {e}")
PY
```

**Quality metrics to check:**
- `dir_acc` - Direction accuracy (above 0.52 is good)
- `pred_std` - Prediction standard deviation (higher = more confident)
- `wfe` - Walk-forward efficiency (above 1.0 is profitable)

---

## 9. Launch API Server

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

Open http://localhost:3000/ai

---

## 10. Quick Test Commands

```bash
# Health check
curl http://localhost:8000/api/health

# Run prediction via API
curl -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "symbol": "AAPL",
    "modelVariant": "auto",
    "dataInterval": "1d",
    "dataPeriod": "10y",
    "horizon": 10,
    "daysOnChart": 180,
    "maxLong": 1.6,
    "maxShort": 0.25
  }'
```

---

## File Locations

| Type | Location |
|------|----------|
| Trained models | `python-ai-service/saved_models/{SYMBOL}/` |
| Backtest results | `python-ai-service/backtest_results/{SYMBOL}_*/` |
| Experiment matrices | `python-ai-service/experiments/*.csv` |
| Backtest charts (PNG) | Each backtest folder |
| Diagnostics JSON | Each backtest folder |

---

## Model Variants

| Variant | Description |
|---------|-------------|
| `auto` | Automatically pick best model |
| `gbm` | Gradient boosting (XGBoost) |
| `gbm_intraday_1h_v5` | Hourly model for day trading |
| `daily_v4` | Daily model version 4 |
| `auto_intraday` | Auto-select intraday model |

---

## Troubleshooting

### Model not found
- Train first: `python -m training.train_gbm {SYMBOL} ...`
- Check saved_models folder exists

### GPU not working
- Run: `nvidia-smi`
- If error, restart conda: `conda deactivate && conda activate ai-stocks`

### Backtest fails
- Check warmup days: need at least 252 for daily, 40 for intraday
- Check data period: `--data-period 730d` for intraday, `--data-period max` for daily

### Poor results
- Increase `--n-trials` to 20-30 for better hyperparameters
- Try different `--max-features` (40-60)
- Check model quality gate in diagnostics
