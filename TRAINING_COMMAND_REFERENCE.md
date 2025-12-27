# Training and Validation Command Reference

Quick reference for training, inference, and backtesting all model types.

**Last Updated**: 2025-12-27 (Nuclear Redesign)

---

## Nuclear Redesign Changes (December 2025)

### Key Fixes
1. **GBM Data Leakage Fixed**: Now uses 60/20/20 Train/Val/Test split
2. **Backtest Alignment Fixed**: 1-day position lag applied by default
3. **Walk-Forward Validation**: Proper WFE metric for overfitting detection
4. **Stacking Ensemble**: XGBoost meta-learner replaces naive fusion

### New Model: xLSTM-TS
- Extended LSTM with exponential gating
- Wavelet denoising for noise reduction
- State-of-the-art for financial forecasting

---

## Training Commands

### 1-Day Regressor (LSTM+Transformer)

**Standard Training** (uses defaults: batch_size=512, epochs=50):
```bash
cd python-ai-service
python training/train_1d_regressor_final.py AAPL
```

**Custom Arguments**:
```bash
python training/train_1d_regressor_final.py AAPL \
    --epochs 100 \
    --batch-size 1024 \
    --sequence-length 120 \
    --variance-regularization 1.5
```

**Quick Test** (5 epochs for debugging):
```bash
python training/train_1d_regressor_final.py AAPL --epochs 5
```

**Disable Anti-Collapse** (if causing NaN issues):
```bash
python training/train_1d_regressor_final.py AAPL --no-use-anti-collapse-loss
```

---

### Binary Classifiers (BUY/SELL Signals)

**Standard Training**:
```bash
cd python-ai-service
python training/train_binary_classifiers_final.py AAPL
```

**Custom Thresholds**:
```bash
python training/train_binary_classifiers_final.py AAPL \
    --epochs 30 \
    --batch-size 512 \
    --buy-percentile 70 \
    --sell-percentile 30
```

**Adjust Focal Loss Parameters** (for class imbalance):
```bash
python training/train_binary_classifiers_final.py AAPL \
    --focal-gamma 2.0 \
    --focal-alpha 0.75 \
    --sell-focal-gamma 2.5 \
    --sell-focal-alpha 0.8
```

---

### GBM Models (XGBoost/LightGBM)

**Train Both Models** (recommended):
```bash
cd python-ai-service
python training/train_gbm_baseline.py AAPL
```

**Custom Boosting Rounds**:
```bash
python training/train_gbm_baseline.py AAPL --epochs 500
```

**Force Overwrite Existing Models**:
```bash
python training/train_gbm_baseline.py AAPL --overwrite
```

**Force Data Refresh**:
```bash
python training/train_gbm_baseline.py AAPL --force-refresh
```

---

### Quantile Regressor (Uncertainty Estimation)

**Standard Training**:
```bash
cd python-ai-service
python training/train_quantile_regressor.py AAPL --epochs 50
```

---

### xLSTM-TS (Nuclear Redesign - NEW)

**Standard Training** (with Walk-Forward Validation):
```bash
cd python-ai-service
python training/train_xlstm_ts.py --symbol AAPL --epochs 50
```

**Production Mode** (train on all data after WFE validation):
```bash
python training/train_xlstm_ts.py --symbol AAPL --epochs 50 --production
```

**Custom Configuration**:
```bash
python training/train_xlstm_ts.py --symbol AAPL \
    --epochs 100 \
    --batch-size 512 \
    --hidden-dim 128 \
    --num-layers 3 \
    --dropout 0.3 \
    --use-wavelet
```

---

### Stacking Ensemble (Nuclear Redesign - NEW)

**Train Complete Stacking Ensemble**:
```bash
cd python-ai-service
python training/train_stacking_ensemble.py --symbol AAPL
```

**Production Mode** (after validation):
```bash
python training/train_stacking_ensemble.py --symbol AAPL --production
```

---

### Production Pipeline (Nuclear Redesign - NEW)

**Full Validation + Production Training**:
```bash
cd python-ai-service
python -m pipeline.production_pipeline --symbol AAPL --validate --train
```

**Validation Only** (check WFE before production):
```bash
python -m pipeline.production_pipeline --symbol AAPL --validate
```

**Production Training Only** (after WFE confirmed):
```bash
python -m pipeline.production_pipeline --symbol AAPL --train --skip-validation
```

---

## Inference Commands

### Single Prediction (Latest Data)

**Regressor Only**:
```bash
cd python-ai-service
python inference/predict_ensemble.py --symbol AAPL --fusion_mode weighted
```

**With Classifiers**:
```bash
python inference/predict_ensemble.py --symbol AAPL --fusion_mode classifier
```

**GBM Only**:
```bash
python inference/predict_ensemble.py --symbol AAPL --fusion_mode gbm_only
```

### Stacking Ensemble Prediction (Nuclear Redesign - NEW)

**Stacking Predictor** (replaces fusion modes with trained meta-learner):
```bash
cd python-ai-service
python -c "
from inference.stacking_predictor import StackingPredictor
predictor = StackingPredictor('AAPL')
# Load latest features and predict
result = predictor.predict(features)
print(result.summary())
"
```

---

## Backtesting Commands

### Forward Simulation (Nuclear Redesign - NEW)

**True Forward-Looking Simulation** (NO look-ahead bias):
```bash
cd python-ai-service
python -c "
from evaluation.forward_simulator import ForwardSimulator, run_forward_simulation
from inference.stacking_predictor import StackingPredictor

predictor = StackingPredictor('AAPL')
results = run_forward_simulation(
    symbol='AAPL',
    predictor=predictor,
    start_date='2024-01-01',
    end_date='2024-12-31'
)
print(results.summary())
"
```

---

### Standard Backtest (2020-2024)

**Weighted Ensemble** (default):
```bash
cd python-ai-service
python inference_and_backtest.py \
    --symbol AAPL \
    --start_date 2020-01-01 \
    --end_date 2024-12-31 \
    --fusion_mode weighted
```

**Quick Backtest** (1 year):
```bash
python inference_and_backtest.py \
    --symbol AAPL \
    --start_date 2024-01-01 \
    --end_date 2024-12-31 \
    --fusion_mode weighted
```

**Compare All Fusion Modes**:
```bash
for mode in weighted balanced gbm_heavy lstm_heavy classifier gbm_only; do
    echo "=== Testing $mode ==="
    python inference_and_backtest.py --symbol AAPL --fusion_mode $mode \
        --start_date 2020-01-01 --end_date 2024-12-31 \
        2>&1 | grep -E "Total Return|Sharpe|Drawdown|Win Rate"
done
```

**Custom Backtest Window**:
```bash
python inference_and_backtest.py \
    --symbol AAPL \
    --start_date 2022-01-01 \
    --end_date 2023-12-31 \
    --fusion_mode weighted
```

---

## Validation Commands

### Walk-Forward Validation (Nuclear Redesign - NEW)

**Run Walk-Forward Validation**:
```bash
cd python-ai-service
python -c "
from validation.walk_forward import WalkForwardValidator, WalkForwardConfig, WalkForwardMode
import numpy as np

# Configure walk-forward validation
config = WalkForwardConfig(
    mode=WalkForwardMode.ANCHORED,  # Expanding window
    n_iterations=5,
    train_pct=0.60,
    validation_pct=0.15,
    test_pct=0.25,
)

# Run validation
validator = WalkForwardValidator(config)
results = validator.validate(model_factory, X, y)
print(results.summary())
print(f'Aggregate WFE: {results.aggregate_wfe:.1f}%')
"
```

**WFE Thresholds**:
- **WFE > 60%**: Strategy is robust, safe for production
- **WFE 50-60%**: Acceptable, minor overfitting
- **WFE 40-50%**: Marginal, monitor closely
- **WFE < 40%**: Significant overfitting, DO NOT deploy

---

### No-Leakage Verification Tests (Nuclear Redesign - NEW)

**Run All Leakage Tests**:
```bash
cd python-ai-service
pytest tests/test_no_leakage.py -v
```

**Test Specific Components**:
```bash
# Test GBM no leakage
pytest tests/test_no_leakage.py::test_gbm_no_leakage -v

# Test walk-forward no overlap
pytest tests/test_no_leakage.py::test_walk_forward_no_overlap -v

# Test forward sim no look-ahead
pytest tests/test_no_leakage.py::test_forward_sim_no_lookahead -v
```

---

### F4 Standalone Tests

**Test Regressor**:
```bash
cd python-ai-service/tests
python test_regressor_standalone.py
TEST_SYMBOL=MSFT python test_regressor_standalone.py  # Custom symbol
```

**Test GBM**:
```bash
python test_gbm_standalone.py
```

**Test Classifiers**:
```bash
python test_classifier_standalone.py
```

**Test Quantile Regressor**:
```bash
python test_quantile_standalone.py
```

**Test All Models**:
```bash
for symbol in AAPL GOOGL NVDA META; do
    echo "=== Testing $symbol ==="
    TEST_SYMBOL=$symbol python test_regressor_standalone.py
    TEST_SYMBOL=$symbol python test_gbm_standalone.py
    TEST_SYMBOL=$symbol python test_classifier_standalone.py
done
```

---

## Batch Training (Multiple Symbols)

### Train Regressors for All Symbols

**Production Universe** (8 symbols):
```bash
cd python-ai-service
for symbol in AAPL GOOGL NVDA META SPY KO ASML IWM; do
    echo "=== Training Regressor: $symbol ==="
    python training/train_1d_regressor_final.py $symbol
done
```

**Extended Universe** (16 symbols):
```bash
for symbol in AAPL GOOGL NVDA META SPY KO ASML IWM MSFT TSLA AMZN JPM V MA DIS NFLX; do
    echo "=== Training Regressor: $symbol ==="
    python training/train_1d_regressor_final.py $symbol
done
```

### Train GBM for All Symbols

```bash
cd python-ai-service
for symbol in AAPL GOOGL NVDA META SPY KO ASML IWM; do
    echo "=== Training GBM: $symbol ==="
    python training/train_gbm_baseline.py $symbol
done
```

### Train Binary Classifiers for All Symbols

```bash
cd python-ai-service
for symbol in AAPL GOOGL NVDA META SPY KO ASML IWM; do
    echo "=== Training Classifiers: $symbol ==="
    python training/train_binary_classifiers_final.py $symbol
done
```

### Train Everything for a Symbol

**Full Training Pipeline** (Regressor + GBM + Classifiers):
```bash
cd python-ai-service
symbol=AAPL

# Step 1: Train regressor
python training/train_1d_regressor_final.py $symbol --epochs 50

# Step 2: Train GBM
python training/train_gbm_baseline.py $symbol

# Step 3: Train classifiers
python training/train_binary_classifiers_final.py $symbol --epochs 30

# Step 4: Run backtest to validate
python inference_and_backtest.py --symbol $symbol --fusion_mode weighted
```

---

## Fusion Modes Reference

| Mode | Description | Use Case | Expected Performance |
|------|-------------|----------|---------------------|
| `weighted` | Dynamic weights based on recent performance | **Default** - adaptive ensemble | Sharpe 1.0-1.5 |
| `balanced` | Equal contribution from all models | Conservative, stable | Sharpe 0.8-1.2 |
| `gbm_heavy` | 60% GBM, 30% LSTM, 10% classifiers | Favor gradient boosting | Sharpe 0.9-1.3 |
| `lstm_heavy` | 60% LSTM, 30% GBM, 10% classifiers | Favor deep learning | Sharpe 1.0-1.4 |
| `classifier` | Binary signals gate regressor predictions | High conviction trades only | Sharpe 0.6-1.0 (lower trades) |
| `gbm_only` | Pure GBM strategy | Simplest, most reliable | Sharpe 0.8-1.2 |

---

## Default Arguments (After GPU Optimization)

### Regressor Training
- **Batch size**: 512 (powerful GPU optimized)
- **Epochs**: 50 (reduced from 100 for faster iteration)
- **Sequence length**: 90 days (quarterly patterns)
- **Anti-collapse loss**: Enabled by default
- **Target balancing**: Enabled by default
- **Variance regularization**: 1.0 (doubled from 0.5)
- **Multi-task learning**: Enabled (sign + volatility heads)
- **Direction loss**: Enabled

### Binary Classifiers
- **Batch size**: 512 (increased from 32)
- **Epochs**: 30 (reduced from 80)
- **Focal loss**: Enabled (gamma=2.0, alpha=0.75)
- **Oversampling**: Enabled (BorderlineSMOTE)
- **Buy percentile**: 60 (top 40%)
- **Sell percentile**: 40 (bottom 40%)

### GBM
- **Boosting rounds**: 1000
- **Model type**: Both XGBoost and LightGBM
- **CV splits**: 5 (time-series aware)
- **No batch size** (tree-based, not neural network)

---

## Performance Monitoring

### Check Training Logs

**Regressor logs**:
```bash
tail -f logs/training_regressor_AAPL_*.log
```

**GBM logs**:
```bash
tail -f logs/training_gbm_AAPL_*.log
```

**Classifier logs**:
```bash
tail -f logs/training_classifiers_AAPL_*.log
```

### Extract Key Metrics from Backtest

```bash
python inference_and_backtest.py --symbol AAPL --fusion_mode weighted 2>&1 | \
    grep -E "Total Return|Annual Return|Sharpe|Max Drawdown|Win Rate|Trades"
```

### Compare Symbol Performance

```bash
for symbol in AAPL GOOGL NVDA META; do
    echo "=== $symbol Performance ==="
    python inference_and_backtest.py --symbol $symbol --fusion_mode weighted 2>&1 | \
        grep -E "Total Return|Sharpe"
done
```

---

## Troubleshooting

### NaN Loss During Training

**Problem**: Loss becomes NaN after a few epochs.

**Solutions**:
```bash
# 1. Disable anti-collapse loss
python training/train_1d_regressor_final.py AAPL --no-use-anti-collapse-loss

# 2. Reduce variance regularization
python training/train_1d_regressor_final.py AAPL --variance-regularization 0.1

# 3. Lower batch size
python training/train_1d_regressor_final.py AAPL --batch-size 128
```

### Variance Collapse (Flat Predictions)

**Problem**: Model predicts near-zero for all samples.

**Solutions**:
```bash
# 1. Enable anti-collapse loss (already default)
python training/train_1d_regressor_final.py AAPL --use-anti-collapse-loss

# 2. Increase variance regularization
python training/train_1d_regressor_final.py AAPL --variance-regularization 2.0

# 3. Enable target balancing
python training/train_1d_regressor_final.py AAPL --balance-targets
```

### GPU Out of Memory

**Problem**: CUDA OOM error during training.

**Solutions**:
```bash
# 1. Reduce batch size
python training/train_1d_regressor_final.py AAPL --batch-size 256

# 2. Reduce sequence length
python training/train_1d_regressor_final.py AAPL --sequence-length 60

# 3. Monitor GPU memory
watch -n 1 nvidia-smi
```

### Feature Count Mismatch

**Problem**: Shape mismatch during inference.

**Solutions**:
```bash
# 1. Force feature refresh
python training/train_1d_regressor_final.py AAPL --force-refresh

# 2. Delete cached features
rm -rf cache/AAPL_features_*.pkl

# 3. Verify feature count
python -c "from data.feature_engineer import get_feature_columns, EXPECTED_FEATURE_COUNT; print(len(get_feature_columns(True)))"
```

---

## Tips and Best Practices

### 1. Quick Debugging

Use low epochs for fast iteration:
```bash
python training/train_1d_regressor_final.py AAPL --epochs 5
```

### 2. Monitor Training Progress

Save output to log file:
```bash
python training/train_1d_regressor_final.py AAPL 2>&1 | tee /tmp/training_AAPL.log
```

### 3. GPU Memory Management

Reduce batch size if OOM errors occur:
- RTX 5060 Ti (16GB): batch_size=512 (default)
- RTX 3060 (12GB): batch_size=256-384
- GTX 1080 (8GB): batch_size=128-256

### 4. Compare to Buy-and-Hold

Always include baseline comparison:
```bash
python inference_and_backtest.py --symbol AAPL --fusion_mode weighted 2>&1 | \
    grep -E "Buy-and-Hold|Strategy"
```

### 5. Production Deployment

Use these fusion modes for production:
- **weighted**: Best overall performance (adaptive)
- **gbm_only**: Most reliable (simpler, less overfitting)
- **balanced**: Conservative (stable across symbols)

### 6. Parallel Training

Train multiple symbols in parallel (adjust based on GPU memory):
```bash
# Terminal 1
python training/train_1d_regressor_final.py AAPL &

# Terminal 2
python training/train_1d_regressor_final.py GOOGL &

# Terminal 3
python training/train_1d_regressor_final.py NVDA &
```

### 7. Verify Model Quality

Run F4 validation after training:
```bash
cd python-ai-service/tests
TEST_SYMBOL=AAPL python test_regressor_standalone.py
TEST_SYMBOL=AAPL python test_gbm_standalone.py
```

---

## Quick Start Guide

**New to the system? Start here:**

1. **Train your first model** (5-minute test):
   ```bash
   cd python-ai-service
   python training/train_1d_regressor_final.py AAPL --epochs 5
   ```

2. **Run a quick backtest**:
   ```bash
   python inference_and_backtest.py --symbol AAPL --fusion_mode weighted
   ```

3. **Train production models**:
   ```bash
   # Full training (30-60 minutes with GPU)
   python training/train_1d_regressor_final.py AAPL
   python training/train_gbm_baseline.py AAPL
   python training/train_binary_classifiers_final.py AAPL
   ```

4. **Validate everything works**:
   ```bash
   cd tests
   TEST_SYMBOL=AAPL python test_regressor_standalone.py
   TEST_SYMBOL=AAPL python test_gbm_standalone.py
   ```

5. **Deploy to production**:
   - Use `weighted` fusion mode for adaptive ensemble
   - Monitor performance daily
   - Retrain models quarterly or when performance degrades

---

## Additional Resources

- **System Architecture**: `/home/thunderboltdy/ai-stocks/SYSTEM_ARCHITECTURE.md`
- **Current Implementation**: `python-ai-service/current_implementation.md`
- **Model Test Results**: `python-ai-service/model_test_results.md`
- **CLAUDE.md**: Project overview and development guide
- **API Integration**: `API_INTEGRATION_GUIDE.md`

---

**Last Updated**: 2025-12-22
**GPU Optimization**: RTX 5060 Ti (16GB VRAM)
**Default Batch Size**: 512
**Default Epochs**: 50 (regressor), 30 (classifiers), 1000 (GBM)
