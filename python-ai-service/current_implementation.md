# Current Implementation: AI-Stocks System (Phase 7 - Nuclear Redesign)

**Last Updated**: December 27, 2025
**Version**: v4.0 (Nuclear Redesign)
**Purpose**: Business-standard, scientifically validated AI prediction system

---

## Executive Summary: What Changed

The **Nuclear Redesign** (Phase 7) was a complete overhaul addressing critical issues:

| Issue | Before | After |
|-------|--------|-------|
| **Data Leakage** | GBM trained on 80/20, metrics on val set | 60/20/20 split, metrics ONLY on held-out test |
| **Validation** | No walk-forward, overfitting not detected | Walk-Forward CV with WFE metric (>50% threshold) |
| **Ensemble** | 6 naive fusion modes (weighted, balanced, etc.) | XGBoost meta-learner stacking ensemble |
| **Backtest Bias** | Position[T] applied to Return[T] (look-ahead) | Position[T] applies to Return[T+1] (correct) |
| **New Model** | None | xLSTM-TS (Extended LSTM for Time Series) |
| **Production Pipeline** | Ad-hoc training | Validate → Gate → Train on ALL data |

---

## System Architecture (Phase 7)

```
┌──────────────────────────────────────────────────────────────────────┐
│                     PHASE 1: VALIDATION                               │
│                                                                        │
│   Walk-Forward Cross-Validation (Anchored/Expanding Window)          │
│   ┌───────────────────────────────────────────────────────────┐       │
│   │ Fold 1: [TRAIN 60%    |VAL 15% |TEST 25%]                 │       │
│   │ Fold 2: [TRAIN 65%    |   VAL 15%|TEST 20%]               │       │
│   │ Fold 3: [TRAIN 70%    |      VAL 15%|TEST 15%]            │       │
│   │ Fold 4: [TRAIN 75%    |         VAL 15%|TEST 10%]         │       │
│   │ Fold 5: [TRAIN 80%    |            VAL 15%|TEST 5%]       │       │
│   └───────────────────────────────────────────────────────────┘       │
│                            ↓                                           │
│   WFE = (Test Sharpe / Val Sharpe) × 100                              │
│   • WFE > 60%: Good - proceed to production                           │
│   • WFE 40-60%: Acceptable - some overfitting                         │
│   • WFE < 40%: FAIL - do NOT deploy                                   │
└──────────────────────────────────────────────────────────────────────┘
                            ↓
                    ┌───────────────┐
                    │  GATE CHECK   │
                    │  WFE >= 50%?  │
                    └───────┬───────┘
                            │
              ┌─────────────┴─────────────┐
              │                           │
        [YES: PASS]                 [NO: FAIL]
              ↓                           ↓
┌──────────────────────────┐    ┌──────────────────────┐
│ PHASE 2: PRODUCTION      │    │  STOP!               │
│ Train on 100% data       │    │  Investigate         │
│ Deploy ensemble          │    │  overfitting         │
└──────────────────────────┘    └──────────────────────┘
```

---

## Model Architecture

### The Three Base Models

```
┌─────────────────────────────────────────────────────────────────────┐
│                         RAW DATA (yfinance)                         │
│                    OHLCV + Volume + Sentiment                       │
└────────────────────────────┬────────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────────┐
│              FEATURE ENGINEERING (157 Features)                     │
│   • 118 Technical (RSI, MACD, Bollinger, etc.)                     │
│   • 29 Sentiment (news scores, sentiment volatility)               │
│   • 10 Regime (volatility regime, support/resistance)              │
└────────────────────────────┬────────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         │                   │                   │
         ▼                   ▼                   ▼
┌─────────────────┐ ┌─────────────────┐ ┌─────────────────┐
│   LSTM+Trans    │ │    xLSTM-TS     │ │       GBM       │
│   (Existing)    │ │    (NEW)        │ │  (XGB+LightGBM) │
│                 │ │                 │ │                 │
│ • 48 LSTM units │ │ • Exp. gating   │ │ • Early stop    │
│ • 4 Transformer │ │ • Wavelet noise │ │ • 60/20/20 split│
│   blocks        │ │   reduction     │ │ • WFE validated │
│ • Anti-collapse │ │ • Residual      │ │                 │
│   loss          │ │   connections   │ │                 │
└────────┬────────┘ └────────┬────────┘ └────────┬────────┘
         │                   │                   │
         │     pred_lstm     │   pred_xlstm      │     pred_gbm
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │
                             ▼
         ┌─────────────────────────────────────────┐
         │          STACKING ENSEMBLE               │
         │                                          │
         │  Inputs to XGBoost Meta-Learner:        │
         │  ┌──────────────────────────────────┐   │
         │  │ • Base predictions (3)            │   │
         │  │ • Regime features (6)             │   │
         │  │ • Model agreement (3)             │   │
         │  │   - prediction_std                │   │
         │  │   - sign_agreement                │   │
         │  │   - max_min_spread                │   │
         │  └──────────────────────────────────┘   │
         │                                          │
         │         XGBoost Meta-Learner             │
         │  (Trained on OOF predictions from CV)   │
         └─────────────────┬───────────────────────┘
                           │
                           ▼
         ┌─────────────────────────────────────────┐
         │         FINAL PREDICTION                 │
         │                                          │
         │  • prediction: next-day return          │
         │  • confidence: model agreement based    │
         │  • position_size: [-0.5, +1.0]          │
         │  • regime: current market regime        │
         └─────────────────────────────────────────┘
```

---

## Model Details

### 1. LSTM+Transformer (Existing, Unchanged)

**File**: `models/lstm_transformer_paper.py`
**Training**: `training/train_1d_regressor_final.py`

- LSTM layer (48 units) → captures temporal patterns
- 4 Transformer blocks → self-attention over sequence
- DirectionalHuberLoss → penalizes wrong direction predictions
- **Output**: 1-day return prediction

### 2. xLSTM-TS (NEW)

**File**: `models/xlstm_ts.py`
**Training**: `training/train_xlstm_ts.py`

Based on Beck et al. (2024) "xLSTM: Extended Long Short-Term Memory":

```python
xLSTMConfig(
    input_dim=157,           # 157 features
    hidden_dim=64,           # Hidden layer size
    num_layers=2,            # Number of xLSTM blocks
    dropout=0.2,
    use_exponential_gating=True,  # Key xLSTM feature
    use_wavelet_denoise=True,     # Noise reduction
    residual_connection=True      # Skip connections
)
```

**Key Features**:
- **Exponential Gating**: Better gradient flow than sigmoid
- **Wavelet Denoising**: Reduces noise in input sequences
- **Residual Connections**: Prevents degradation in deep networks

### 3. GBM (XGBoost/LightGBM) - FIXED

**File**: `training/train_gbm_baseline.py`

**What Was Fixed**:
```python
# BEFORE (LEAKED):
split = 80/20
metrics computed on validation data (WRONG!)

# AFTER (NO LEAKAGE):
train_end = int(len(X) * 0.60)  # 60% train
val_end = int(len(X) * 0.80)    # 20% validation
test_start = val_end             # 20% test (HELD OUT)

# Validation ONLY used for early stopping
# Final metrics ONLY on held-out test set
```

---

## The Stacking Ensemble

### What Replaced the Old Fusion Modes

**Before (6 Naive Modes)**:
```python
# weighted, balanced, gbm_heavy, lstm_heavy, classifier, gbm_only
# These were hand-coded weight combinations
position = 0.6 * gbm + 0.3 * lstm + 0.1 * classifier  # gbm_heavy
```

**After (Learned Meta-Learner)**:
```python
# XGBoost learns optimal weights from data
meta_features = [
    pred_lstm, pred_xlstm, pred_gbm,     # Base predictions
    vol_regime_high, vol_regime_normal,  # Regime features
    prediction_std, sign_agreement,      # Model agreement
]
final_prediction = meta_learner.predict(meta_features)
```

### Why Stacking is Better

1. **Learns from data** - not hardcoded weights
2. **Regime-aware** - adjusts based on market conditions
3. **Uses model agreement** - more confident when models agree
4. **Trained on OOF predictions** - no data leakage

---

## Directory Structure (Updated)

```
saved_models/
└── {SYMBOL}/
    ├── regressor/              # LSTM+Transformer (existing)
    │   ├── model.keras
    │   ├── regressor.weights.h5
    │   ├── feature_scaler.pkl
    │   └── metadata.pkl
    │
    ├── xlstm/                  # xLSTM-TS (NEW)
    │   ├── model.keras
    │   ├── xlstm.weights.h5
    │   ├── feature_scaler.pkl
    │   ├── target_scaler.pkl
    │   └── metadata.pkl
    │
    ├── gbm/                    # XGBoost/LightGBM (FIXED)
    │   ├── xgboost_model.pkl
    │   └── lightgbm_model.pkl
    │
    ├── stacking/               # Stacking Ensemble (NEW)
    │   ├── meta_learner.pkl    # XGBoost meta-learner
    │   ├── regime_detector.pkl
    │   ├── meta_scaler.pkl
    │   └── stacking_metadata.pkl
    │
    ├── classifiers/            # (Still available, optional)
    │   └── ...
    │
    ├── quantile/               # (Still available, optional)
    │   └── ...
    │
    └── feature_columns.pkl     # Canonical feature list
```

---

## Training Commands

### Step 1: Train Individual Models

```bash
cd python-ai-service
conda activate ai-stocks

# 1. Train LSTM+Transformer (existing model)
python training/train_1d_regressor_final.py \
    --symbol AAPL \
    --epochs 50 \
    --batch_size 512

# 2. Train xLSTM-TS (NEW model)
python training/train_xlstm_ts.py \
    --symbol AAPL \
    --epochs 100 \
    --batch_size 512

# 3. Train GBM (FIXED - no data leakage)
python training/train_gbm_baseline.py AAPL --overwrite
```

### Step 2: Run Production Pipeline (Validates + Trains Ensemble)

```bash
# Full pipeline: validate all models → check WFE → train on 100% data
python -m pipeline.production_pipeline --symbol AAPL

# Validate only (don't train production models)
python -m pipeline.production_pipeline --symbol AAPL --validate-only
```

### Step 3: Backtest

```bash
# Backtest with stacking ensemble
python inference_and_backtest.py \
    --symbol AAPL \
    --start_date 2020-01-01 \
    --end_date 2024-12-31 \
    --fusion_mode stacking  # NEW mode
```

---

## Why Your Pipeline Failed

The error shows:
```
lstm: WFE=0.0% [FAIL]
gbm: WFE=0.0% [FAIL] [VARIANCE COLLAPSE]
xlstm_ts: WFE=0.0% [FAIL]
```

**Root Cause**: The models either:
1. **Don't exist** - you need to train them first
2. **Aren't compatible** - old models weren't trained with walk-forward validation
3. **Have variance collapse** - GBM is predicting constant values

### How to Fix

**Option A: Train All Models Fresh**
```bash
# Train LSTM+Transformer
python training/train_1d_regressor_final.py AAPL --epochs 50 --batch-size 512

# Train xLSTM-TS
python training/train_xlstm_ts.py --symbol AAPL --epochs 100 --batch_size 512

# Train GBM (fixed version)
python training/train_gbm_baseline.py AAPL --overwrite

# Then run pipeline
python -m pipeline.production_pipeline --symbol AAPL
```

**Option B: Use Standalone Validation Scripts** (for debugging)
```bash
# Test individual model types
python tests/test_regressor_standalone.py
python tests/test_gbm_standalone.py
```

---

## Walk-Forward Validation (NEW)

### How It Works

```
Data: [2010-01-01 ───────────────────────────── 2024-12-31]

Fold 1:
Train: [2010 ──── 2018]    Val: [2018-2020]    Test: [2020-2024]
                  60%              15%               25%

Fold 2:
Train: [2010 ───── 2019]   Val: [2019-2021]   Test: [2021-2024]
                  65%              15%              20%

...and so on with expanding window
```

### WFE (Walk Forward Efficiency)

```
WFE = (Test Sharpe / Validation Sharpe) × 100

Example:
  Val Sharpe = 1.5
  Test Sharpe = 0.9
  WFE = 0.9 / 1.5 × 100 = 60% ← PASS

Thresholds:
  • WFE > 60%: Good - proceed to production
  • WFE 40-60%: Acceptable with caution
  • WFE < 50%: FAIL - do NOT deploy (our threshold)
  • WFE < 40%: Serious overfitting problem
```

---

## Files Created/Modified

### New Files (Nuclear Redesign)

| File | Purpose |
|------|---------|
| `validation/walk_forward.py` | Walk-forward CV framework |
| `validation/wfe_metrics.py` | WFE calculation, overfitting detection |
| `training/train_stacking_ensemble.py` | Stacking ensemble trainer |
| `training/train_xlstm_ts.py` | xLSTM-TS training script |
| `inference/stacking_predictor.py` | Production inference with stacking |
| `models/xlstm_ts.py` | xLSTM-TS model architecture |
| `evaluation/forward_simulator.py` | True forward simulation (no look-ahead) |
| `pipeline/production_pipeline.py` | End-to-end pipeline |
| `tests/test_no_leakage.py` | 24 data leakage verification tests |

### Modified Files

| File | Change |
|------|--------|
| `training/train_gbm_baseline.py` | Fixed 80/20 → 60/20/20 split |
| `evaluation/advanced_backtester.py` | Fixed position lag (T→T+1) |
| `utils/model_paths.py` | Added xLSTMPaths class |

---

## Testing

### Run All No-Leakage Tests
```bash
cd python-ai-service
pytest tests/test_no_leakage.py -v

# Expected: 24/24 tests pass
```

### What These Tests Verify

1. **TestGBMTrainingNoLeakage** - 60/20/20 split correct
2. **TestWalkForwardNoOverlap** - No overlap between train/val/test
3. **TestForwardSimulationNoLookAhead** - Features from T-1 only
4. **TestBacktestAlignment** - Position lag applied correctly
5. **TestNoLeakageIntegration** - End-to-end verification

---

## Key Concepts Summary

| Term | Definition |
|------|------------|
| **Walk-Forward CV** | Expanding window validation, no future data leakage |
| **WFE** | Walk Forward Efficiency - measures overfitting (higher = better) |
| **Stacking** | Training a meta-learner on base model predictions |
| **OOF Predictions** | Out-of-fold predictions used to train meta-learner |
| **Position Lag** | Position[T] applies to Return[T+1] (realistic execution) |
| **Variance Collapse** | Model predicts constant/near-constant values (bad) |
| **xLSTM-TS** | Extended LSTM with exponential gating for time series |

---

## Workflow Summary

```
1. TRAIN BASE MODELS
   └── python training/train_1d_regressor_final.py --symbol AAPL
   └── python training/train_xlstm_ts.py --symbol AAPL
   └── python training/train_gbm_baseline.py AAPL

2. RUN PRODUCTION PIPELINE
   └── python -m pipeline.production_pipeline --symbol AAPL
   └── This runs walk-forward CV and checks WFE

3. IF WFE >= 50%: DEPLOY
   └── Models trained on 100% data
   └── Stacking ensemble created
   └── Ready for live inference

4. IF WFE < 50%: INVESTIGATE
   └── Check for variance collapse
   └── Check for overfitting
   └── May need more data or different architecture

5. BACKTEST
   └── python inference_and_backtest.py --symbol AAPL
```

---

## Comparison: Phase 6 vs Phase 7 (Nuclear Redesign)

| Aspect | Phase 6 | Phase 7 (Now) |
|--------|---------|---------------|
| **GBM Split** | 80/20 (leaky) | 60/20/20 (clean) |
| **Validation** | None | Walk-Forward with WFE |
| **Ensemble** | 6 hardcoded fusion modes | XGBoost meta-learner |
| **Backtest** | Position[T]→Return[T] | Position[T]→Return[T+1] |
| **Models** | LSTM, GBM, Classifiers, Quantile | LSTM, xLSTM-TS, GBM, Stacking |
| **Production** | Ad-hoc training | Validate → Gate → Full Train |
| **Testing** | Basic | 24 no-leakage tests |

---

*This document reflects the Nuclear Redesign (Phase 7) of the AI-Stocks system. The old Phase 6 documentation is preserved in git history.*
