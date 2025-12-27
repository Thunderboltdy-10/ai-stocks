# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

---

## Project Overview

**AI-Stocks** is a hybrid Next.js + Python ML trading system that predicts stock returns using ensemble deep learning models. The system consists of:

1. **Frontend** (Next.js 15 + React 19): Web interface for viewing predictions, backtests, and managing watchlists
2. **Backend API** (Next.js API routes): Authentication (Better Auth), MongoDB persistence, Inngest job scheduling
3. **Python AI Service**: ML training, inference, backtesting with GPU-accelerated TensorFlow/Keras models

**Current System State**: Phase 6 (v3.1+)
- 157 engineered features (118 technical + 29 sentiment + advanced regime/support features)
- 4 active model types: LSTM+Transformer Regressor, Binary Classifiers (BUY/SELL), GBM (XGBoost/LightGBM), Quantile Regressor
- 6 fusion modes for ensemble prediction
- Advanced backtesting with margin costs and risk management

---

## Development Commands

### Frontend (Next.js)

```bash
# Development server (with Turbopack)
npm run dev

# Production build
npm run build

# Start production server
npm start

# Linting
npm run lint

# Testing
npm test              # Run Vitest tests
npm run test:watch    # Watch mode
npm run test:e2e      # Playwright E2E tests
```

### Python AI Service

```bash
cd python-ai-service

# Install dependencies
pip install -r requirements.txt

# Train models for a symbol
python training/train_1d_regressor_final.py --symbol AAPL --epochs 50
python training/train_binary_classifiers_final.py --symbol AAPL --epochs 30
python training/train_gbm_baseline.py --symbol AAPL

# Run inference and backtest
python inference_and_backtest.py --symbol AAPL --start_date 2020-01-01 --end_date 2024-12-31 --fusion_mode weighted

# Validate individual models (F4 tests)
python tests/test_regressor_standalone.py
python tests/test_gbm_standalone.py
python tests/test_classifier_standalone.py
python tests/test_quantile_standalone.py

# Run with custom symbol
TEST_SYMBOL=MSFT python tests/test_regressor_standalone.py

# Activate GPU acceleration (Linux)
bash activate_gpu.sh
```

---

## Documentation Guidelines

### File Creation Policy

**IMPORTANT**: Minimize creation of new documentation files. Follow these rules:

1. **Use Existing Files First**:
   - `implementation_progress.md`: Track all work, fixes, and progress from current session
   - `implementation_plan.md`: Document planned changes and next steps
   - `current_implementation.md`: System state and architecture (Phase 6)
   - `timeline.md`: Historical evolution of the project
   - `last_known_good.md`: Baseline reference (Phase 2)
   - `system_architecture.md`: Complete system reference

2. **DO NOT Create New .md Files** Unless:
   - Absolutely necessary for long-term reference
   - User explicitly requests it
   - Adding to existing project documentation structure (e.g., API docs, user guides)

3. **For Session Work**:
   - Log all progress in `implementation_progress.md`
   - Keep it chronological with dates and summaries
   - Include: what was tried, what worked, what failed, next steps
   - Delete temporary analysis/investigation files after merging into progress

4. **For Root Cause Analysis**:
   - Add findings to `implementation_progress.md` under dated sections
   - Only create separate files if analysis is >2000 lines or needs permanent reference

5. **Cleanup Policy**:
   - After completing work, merge temporary docs into `implementation_progress.md`
   - Delete redundant analysis files
   - Keep file count minimal

### Permanent Documentation Files

These files should be maintained:
- `CLAUDE.md` - This file, guidance for Claude Code
- `README.md` - Project overview for users
- `system_architecture.md` - Technical architecture reference
- `current_implementation.md` - Current system state
- `timeline.md` - Historical evolution
- `last_known_good.md` - Baseline reference
- `implementation_plan.md` - Planned future work
- `implementation_progress.md` - Session work log

All other .md files are temporary and should be merged/deleted after use.

---

## Architecture Overview

### Monorepo Structure

```
ai-stocks/
├── app/                          # Next.js App Router
│   ├── (auth)/                  # Auth pages (sign-in, sign-up)
│   ├── (root)/                  # Main app pages
│   │   ├── page.tsx            # Dashboard
│   │   ├── stocks/[symbol]/    # Stock detail page
│   │   ├── watchlist/          # Watchlist page
│   │   └── ai/                 # AI predictions page
│   ├── api/                     # API routes
│   │   ├── inngest/            # Inngest webhook
│   │   ├── predict/            # ML prediction endpoint
│   │   └── predict-ensemble/   # Ensemble prediction endpoint
│   └── layout.tsx              # Root layout
├── components/                  # React components
├── lib/                         # Shared utilities
│   ├── actions/                # Server actions
│   ├── better-auth/            # Auth configuration
│   ├── inngest/                # Background job definitions
│   └── nodemailer/             # Email templates
├── database/                    # MongoDB models (Mongoose)
├── python-ai-service/          # ML system (see below)
└── models/                      # Root-level model definitions
```

### Python AI Service Architecture

**Critical Concept**: The Python service operates on a **canonical feature contract** - all models trained on the same symbol must use the same feature list in the same order, persisted in `feature_columns.pkl`.

#### Data Flow

```
1. Data Fetching (data/data_fetcher.py)
   └─> OHLCV from yfinance

2. Feature Engineering (data/feature_engineer.py)
   └─> 157 features (deterministic, no look-ahead)
   └─> Saves feature_columns.pkl

3. Training (training/*.py)
   ├─> Loads features, creates sequences
   ├─> Trains models with GPU acceleration (mixed precision float16)
   └─> Saves: model weights, scalers, metadata, feature_columns.pkl

4. Inference (inference/predict_ensemble.py)
   ├─> Loads feature_columns.pkl (enforces same order)
   ├─> Runs engineer_features on latest data
   ├─> Loads models and scalers
   └─> Generates predictions

5. Fusion (inference/hybrid_predictor.py)
   └─> Combines predictions using fusion modes (weighted, balanced, etc.)

6. Backtesting (evaluation/advanced_backtester.py)
   └─> Simulates trades, calculates metrics (Sharpe, drawdown, etc.)
```

#### Key Files

**Training Scripts** (`python-ai-service/training/`):
- `train_1d_regressor_final.py`: LSTM+Transformer regressor (primary model)
- `train_binary_classifiers_final.py`: BUY/SELL signal classifiers
- `train_gbm_baseline.py`: XGBoost/LightGBM gradient boosting
- `train_quantile_regressor.py`: Uncertainty estimation (Q10/Q50/Q90)

**Model Definitions** (`python-ai-service/models/`):
- `lstm_transformer_paper.py`: LSTM+Transformer hybrid architecture (Ruiru et al. 2024)
- `quantile_regressor.py`: Quantile regression with custom loss

**Inference** (`python-ai-service/inference/`):
- `predict_ensemble.py`: Main prediction interface (loads models, returns predictions)
- `hybrid_predictor.py`: Fusion logic (6 modes: weighted, balanced, gbm_heavy, lstm_heavy, classifier, gbm_only)
- `load_gbm_models.py`: GBM-specific loading utilities

**Core Orchestrator**:
- `inference_and_backtest.py`: CLI tool that runs end-to-end: data → features → inference → fusion → backtest

**Utilities** (`python-ai-service/utils/`):
- `model_paths.py`: Centralized path management (supports new organized structure + legacy flat structure)
- `losses.py`: Custom loss functions (DirectionalHuberLoss, AsymmetricDirectionalLoss)
- `training_logger.py`: Training diagnostics

---

## Model Directory Structure

### New Organized Structure (v3.1+)

```
saved_models/
└── {SYMBOL}/
    ├── regressor/
    │   ├── model.keras              # Full SavedModel
    │   ├── regressor.weights.h5     # Weights only
    │   ├── feature_scaler.pkl       # RobustScaler for features
    │   ├── target_scaler_robust.pkl # RobustScaler for targets
    │   ├── features.pkl             # Feature list
    │   └── metadata.pkl             # Training metadata
    ├── classifiers/
    │   ├── buy_model.keras
    │   ├── sell_model.keras
    │   ├── buy_calibrated.pkl       # Calibrated probabilities
    │   ├── sell_calibrated.pkl
    │   └── metadata.pkl
    ├── gbm/
    │   ├── xgboost_model.pkl
    │   └── lightgbm_model.pkl
    ├── quantile/
    │   ├── quantile.weights.h5
    │   └── metadata.pkl
    └── feature_columns.pkl          # Canonical feature list (shared)
```

**Backward Compatibility**: System also supports legacy flat structure (`saved_models/{SYMBOL}_*`). `ModelPaths` class handles both.

---

## Critical Workflows

### Training a New Model

```bash
cd python-ai-service

# 1. Train regressor (primary model)
python training/train_1d_regressor_final.py \
    --symbol AAPL \
    --epochs 50 \
    --batch_size 32 \
    --sequence_length 90

# 2. Train classifiers (optional, for 'classifier' fusion mode)
python training/train_binary_classifiers_final.py \
    --symbol AAPL \
    --epochs 30 \
    --signal_threshold 0.15

# 3. Train GBM (recommended for ensemble)
python training/train_gbm_baseline.py \
    --symbol AAPL \
    --model_type xgboost
```

**Important**: All training scripts for the same symbol must use the same feature engineering settings. Check `feature_columns.pkl` matches across models.

### Running Backtests

```bash
# Basic backtest
python inference_and_backtest.py \
    --symbol AAPL \
    --start_date 2020-01-01 \
    --end_date 2024-12-31 \
    --fusion_mode weighted

# Compare fusion modes
for mode in weighted balanced gbm_heavy lstm_heavy classifier; do
    python inference_and_backtest.py \
        --symbol AAPL \
        --fusion_mode $mode \
        --output_dir results/fusion_comparison
done
```

### Validating Models (F4 Framework)

Before deploying models to production, run standalone validation tests:

```bash
cd python-ai-service/tests

# Test each model type independently
python test_regressor_standalone.py
python test_gbm_standalone.py
python test_classifier_standalone.py
python test_quantile_standalone.py

# Batch test multiple symbols
for symbol in AAPL MSFT TSLA GOOGL; do
    TEST_SYMBOL=$symbol python test_regressor_standalone.py
done
```

**What These Tests Check**:
1. Model loads from saved artifacts (new + legacy paths)
2. Predictions have correct shape
3. Variance collapse detection (std > 0.005, balanced positive/negative predictions)
4. Basic backtest runs and produces reasonable Sharpe ratio

---

## Feature Engineering Contract

**CRITICAL**: Feature engineering must be deterministic and reproducible across training and inference.

### Rules

1. **Always use `get_feature_columns(include_sentiment=True)`** to get the canonical feature list
2. **Expected feature count**: `EXPECTED_FEATURE_COUNT = 157` (in `data/feature_engineer.py`)
3. **Sentiment features are optional**: Use `include_sentiment=False` for 128 features (backward compatible)
4. **No look-ahead bias**: All features must use only past data

### Feature Categories

- **Technical (118)**: Price-based, momentum, trend, volume indicators
- **Sentiment (29)**: News sentiment scores, volatility, trend (optional)
- **Regime (16)**: Volatility regime detection, regime-conditional features
- **Support/Resistance (4)**: Distance to support/resistance, strength metrics

### Validation

```python
from data.feature_engineer import get_feature_columns, EXPECTED_FEATURE_COUNT

# Get canonical features
features = get_feature_columns(include_sentiment=True)

# Verify count
assert len(features) == EXPECTED_FEATURE_COUNT  # Should be 157
```

---

## GPU Acceleration

The system uses TensorFlow with mixed precision (float16) and XLA compilation for GPU acceleration.

### Setup

```bash
# Linux: Install NVIDIA libraries (already in requirements.txt)
pip install nvidia-cudnn-cu11 nvidia-cublas-cu11

# Activate GPU environment
bash activate_gpu.sh

# Verify GPU detected
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Training Configuration

All training scripts automatically configure GPU:

```python
# Mixed precision (float16)
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

# XLA compilation
os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2'
```

**Benefits**: ~2x faster training, ~50% lower memory usage

---

## Testing Strategy

### Test Levels

1. **Unit Tests**: Data processing, feature engineering (`tests/test_data_integrity.py`, `tests/test_feature_count.py`)
2. **Model Validation (F4)**: Standalone model tests (`tests/test_*_standalone.py`)
3. **Integration Tests**: End-to-end workflows (`tests/test_diverse_universe.py`)
4. **Frontend Tests**: Vitest unit tests, Playwright E2E

### Running Tests

```bash
# Frontend tests
npm test                    # Vitest
npm run test:e2e           # Playwright

# Python model validation
cd python-ai-service/tests
python test_regressor_standalone.py
python test_gbm_standalone.py

# Python integration tests
python test_diverse_universe.py
python test_data_integrity.py
```

---

## Fusion Modes Explained

The system supports 6 fusion modes for combining model predictions:

| Mode | Strategy | Use Case |
|------|----------|----------|
| `weighted` | Dynamic weights based on recent performance | **Default** - adaptive |
| `balanced` | Equal contribution from all models | Stable, conservative |
| `gbm_heavy` | 60% GBM, 30% LSTM, 10% classifiers | Favor gradient boosting |
| `lstm_heavy` | 60% LSTM, 30% GBM, 10% classifiers | Favor deep learning |
| `classifier` | Binary signals gate regressor predictions | High conviction trades only |
| `gbm_only` | Pure GBM strategy, ignore LSTM/classifiers | Simplest ensemble |

**Position Sizing**: All modes output positions in range `[-0.5, 1.0]` (max 50% short, 100% long).

---

## Common Pitfalls

### 1. Feature Count Mismatch

**Problem**: Model trained on 157 features but inference uses 128 features (or vice versa).

**Symptom**: Shape mismatch error during prediction.

**Fix**:
```python
# Always use saved feature list
feature_cols = pickle.load(open('saved_models/AAPL/feature_columns.pkl', 'rb'))
X = df[feature_cols]  # Subset and order match training
```

### 2. Variance Collapse

**Problem**: Model always predicts near-zero or same value.

**Symptoms**:
- Prediction std < 0.005
- >95% of predictions are positive or negative
- Sharpe ratio near 0

**Causes**:
- Training stopped too early
- Learning rate too high
- Loss function not directional-aware

**Fix**: Use directional losses (`DirectionalHuberLoss`) and monitor variance during training.

### 3. Path Confusion (New vs Legacy)

**Problem**: Can't find model files due to path structure changes.

**Solution**: Use `ModelPaths` class which handles both:

```python
from utils.model_paths import ModelPaths

paths = ModelPaths('AAPL')
paths.regressor.model       # -> saved_models/AAPL/regressor/model.keras (new)
                           # or saved_models/AAPL_1d_regressor_final_model (legacy)
```

### 4. Sentiment Feature Dependency

**Problem**: Model trained with sentiment features but inference runs without news data.

**Fix**:
```python
# Training
df_features = engineer_features(df, include_sentiment=True)

# Inference - must match
df_features = engineer_features(df, include_sentiment=True)
```

Or train separate models with/without sentiment.

---

## Key Documentation Files

Read these for deep understanding:

- **`python-ai-service/current_implementation.md`**: How current saved models work (Phase 6 implementation)
- **`python-ai-service/last_known_good.md`**: Historical baseline (Regressor + GBM only, Phase 2)
- **`python-ai-service/timeline.md`**: Evolution from Phase 1 to Phase 6
- **`python-ai-service/model_test_results.md`**: F4 validation framework guide
- **`SYSTEM_ARCHITECTURE.md`**: Complete system reference (data flow, contracts, artifacts)
- **`python-ai-service/CLEANUP_PROGRESS.md`**: F1/F2/F3 audit completion summary

---

## Environment Variables

```bash
# MongoDB connection
MONGODB_URI=mongodb://localhost:27017/ai-stocks

# Better Auth
BETTER_AUTH_SECRET=<secret>
BETTER_AUTH_URL=http://localhost:3000

# Inngest
INNGEST_EVENT_KEY=<key>
INNGEST_SIGNING_KEY=<key>

# Email (Nodemailer)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=<email>
SMTP_PASS=<password>

# Python testing
TEST_SYMBOL=AAPL  # Override default test symbol
```

---

## Database Schema (MongoDB/Mongoose)

### User Model
```typescript
{
  name: string
  email: string
  emailVerified: boolean
  image?: string
  createdAt: Date
  updatedAt: Date
}
```

### Watchlist Model
```typescript
{
  userId: ObjectId
  symbol: string
  addedAt: Date
}
```

**Indexes**: Compound index on `(userId, symbol)` for fast lookups.

---

## API Integration Points

### Next.js ↔ Python Service

**Prediction Endpoint**: `app/api/predict/route.ts`
```typescript
POST /api/predict
Body: { symbol: string, period?: string }
Response: { prediction: number, confidence: number, ... }
```

Internally calls Python via subprocess:
```bash
python python-ai-service/inference/predict_ensemble.py --symbol AAPL
```

**Ensemble Endpoint**: `app/api/predict-ensemble/route.ts`
```typescript
POST /api/predict-ensemble
Body: { symbol: string, fusion_mode?: string }
Response: {
  position: number,
  predictions: { regressor: number, gbm?: number },
  classifiers?: { buy: number, sell: number }
}
```

---

## Inngest Background Jobs

**Job**: Daily stock predictions (`lib/inngest/functions.ts`)

```typescript
inngest.createFunction(
  { id: "daily-predictions" },
  { cron: "0 9 * * 1-5" },  // 9 AM weekdays
  async ({ step }) => {
    // Fetch watchlist symbols
    // Run predictions for each
    // Send email digest
  }
)
```

---

## Migration Notes

### If Models Fail to Load

1. Check if using new or legacy directory structure
2. Verify `feature_columns.pkl` exists and matches model input shape
3. Ensure scalers are compatible (RobustScaler vs MinMaxScaler)
4. Try loading with `ModelPaths` class fallback logic

### If Adding New Features

1. Update `EXPECTED_FEATURE_COUNT` in `data/feature_engineer.py`
2. Retrain ALL models for each symbol with new feature count
3. Update documentation in `current_implementation.md`
4. Run F4 validation tests to verify no collapse

### If Changing Model Architecture

1. Increment version in metadata
2. Train new models in parallel directory (`saved_models/{SYMBOL}/regressor_v2/`)
3. Compare performance before switching
4. Update `ModelPaths` to support new structure

---

## Performance Benchmarks

**Expected Metrics** (symbol-dependent):
- **Sharpe Ratio**: 0.8 - 1.5
- **Directional Accuracy**: 52-58%
- **Max Drawdown**: 15-25%
- **Variance (predictions)**: > 0.005 (0.5%)

**Inference Speed**:
- Feature engineering: ~0.5-2 seconds (cached)
- Model loading: ~1-3 seconds (first call)
- Prediction: <100ms per symbol
- Training: 10-30 minutes with GPU

---

## Quick Reference

### Train Everything for a Symbol

```bash
cd python-ai-service

# One-liner: train all model types
symbol=AAPL
python training/train_1d_regressor_final.py --symbol $symbol --epochs 50
python training/train_binary_classifiers_final.py --symbol $symbol --epochs 30
python training/train_gbm_baseline.py --symbol $symbol --model_type xgboost
python training/train_quantile_regressor.py --symbol $symbol --epochs 50
```

### Validate Everything

```bash
cd python-ai-service/tests

# Run all F4 tests
for test in test_regressor_standalone.py test_gbm_standalone.py test_classifier_standalone.py test_quantile_standalone.py; do
    python $test
done
```

### Full Development Loop

```bash
# Terminal 1: Frontend
npm run dev

# Terminal 2: MongoDB
mongod --dbpath ./data/db

# Terminal 3: Python testing
cd python-ai-service
python inference_and_backtest.py --symbol AAPL
```

---

This guide should enable rapid onboarding to the AI-Stocks codebase architecture and development workflows.
- Activate the python conda environment using conda activate ai-stocks to run any python script
- Avoid generating extra markdown documents unless specifically stated. Display summaries instead directly to the user, or add into implementation_plan.md, and implementation_progress.md. Also, try and keep SYSTEM_ARCHITECTURE.md updated
- Use GPU Acceleration for training, and a high batch-size (512 or greater)