# AI-Stocks Python Service: Complete Rewrite Plan

## Context

The AI-Stocks project was last worked on ~2 months ago (Dec 2025). Despite months of development across 7 "phases", the models never produced profitable results. The codebase has ballooned to **154 Python files** with massive dead code (deprecated TFT, xLSTM-TS, binary classifiers, PatchTST), recurring variance collapse in deep learning models, and multiple unfixed bugs. The user has given **full freedom** to delete, rewrite, and redesign everything.

**Root causes of failure:**
1. Over-engineering: 7 model architectures when only GBM consistently worked
2. Variance collapse: LSTM+Transformer attention collapses every time (epochs 25-35)
3. Bugs: LR scheduler was 50x too low (2e-05 vs 1e-03), GBM sample weights missing from final training
4. 157 features with no selection -- noise overwhelms signal
5. No Optuna tuning -- fixed hyperparameters for all models
6. Code sprawl making debugging impossible (154 files, 73K+ lines)

**Strategy: GBM-first, simplicity over complexity.** Research consensus (2025-2026) confirms gradient boosting (XGBoost/LightGBM) consistently outperforms deep learning on tabular financial data. An optional simplified LSTM (no transformer attention) serves as ensemble secondary.

**This plan is designed for autonomous execution via a Ralph Loop** -- an AI agent iterating continuously, training models, checking results, fixing issues, and re-iterating without human intervention.

---

## Phase 0: Cleanup -- Delete Dead Code (Mandatory)

**Goal:** Reduce from 154 to ~40 Python files. Remove all deprecated/dead code.

### Files to DELETE

**Models (delete 5 of 7):**
- `models/lstm_transformer_paper_deprecated_v6.py`
- `models/patchtst.py`
- `models/tft_loader.py`
- `models/xlstm_ts.py`
- `models/quantile_regressor.py`

**Training (delete 8 of 14):**
- `training/train_tft.py`, `training/train_with_tft.py`, `training/train_xlstm_ts.py`
- `training/train_binary_classifiers_final.py`, `training/train_smart_regressor.py`
- `training/train_quantile_regressor.py`, `training/train_meta_learner.py`
- `training/train_stacking_ensemble.py`

**Inference (delete 5+):**
- `inference/predict_tft.py`, `inference/rl_position_sizer.py`
- `inference/quantile_ensemble_inference.py`, `inference/stacking_inference.py`
- `inference/predict_multiday.py`

**Entire directories to DELETE:**
- `audit_tools/`, `f4_validation_results/`, `lightning_logs/`
- `optimization_results/`, `orchestration_logs/`, `monitoring_logs/`
- `plots/`, `results/`, `validation_results/`

**Root-level junk:** All `test_*.py` files in root (not in `tests/`), `extract_embeddings.py`, `parse_f4_validation_results.py`, `train_all.py`, `FIX_QUICK_REFERENCE.txt`, `RUN_F4_VALIDATION.txt`, all `training_*.log` and `training_output_*.log` in root, shell scripts (`run_f4_validation_tests.sh`, `monitor_training.sh`)

**Scripts (delete ~20, keep 3):** Delete everything in `scripts/` except `health_check.py`, `validate_pipeline.py`

**Tests (delete deprecated):** `test_classifier_standalone.py`, `test_classifier_with_calibration.py`, `test_quantile_standalone.py`, `test_augmentation.py`, `test_bear_market_msft.py`, `test_target_balancing.py`, `analyze_backtest_pickle.py`, `verify_*.py`

**Data:** `data/tft_dataset_builder.py`, `data/augmentation.py`

**The 355KB monolith `inference_and_backtest.py`:** DELETE (replaced by new `run_backtest.py` in Phase 5)

### Files to KEEP (will be modified in later phases)
- `app.py`, `data/data_fetcher.py`, `data/feature_engineer.py`, `data/target_engineering.py`
- `data/cache_manager.py`, `data/news_fetcher.py`, `data/sentiment_*.py`, `data/support_resistance_features.py`, `data/volatility_features.py`
- `models/simple_lstm_regressor.py`, `models/lstm_transformer_paper.py` (reference only)
- `training/train_gbm_baseline.py`, `training/train_1d_regressor_final.py` (reference), `training/feature_selection.py`, `training/rolling_cv.py`
- `inference/predict_ensemble.py`, `inference/load_gbm_models.py`, `inference/hybrid_predictor.py`, `inference/confidence_scorer.py`, `inference/model_validator.py`
- `evaluation/advanced_backtester.py` (excellent, production-ready), `evaluation/model_diagnostics.py`, `evaluation/model_validation_suite.py`
- `validation/walk_forward.py`, `validation/wfe_metrics.py`
- `service/prediction_service.py`, `service/model_registry.py`, `service/training_service.py`
- `utils/model_paths.py`, `utils/losses.py`, `utils/standardized_logger.py`, `utils/training_logger.py`, `utils/model_io.py`
- `tests/test_data_integrity.py`, `tests/test_feature_count.py`, `tests/test_gbm_standalone.py`, `tests/test_regressor_standalone.py`, `tests/test_diverse_universe.py`, `tests/test_feature_selection.py`

### Also in Phase 0: Simplify `utils/model_paths.py`
Remove ALL legacy flat-file path support. Only support the organized structure: `saved_models/{SYMBOL}/gbm/` and `saved_models/{SYMBOL}/lstm/`.

### Quality Gate
```bash
find python-ai-service -name "*.py" | wc -l  # Must be <= 45
python -c "from data.data_fetcher import fetch_stock_data; print('OK')"
python -c "from data.feature_engineer import engineer_features; print('OK')"
python -c "from evaluation.advanced_backtester import AdvancedBacktester; print('OK')"
```

---

## Phase 1: Data Pipeline Rebuild (Mandatory)

**Goal:** Lean feature engineering (~70 features), proper splitting with purging/embargo, data validation.

### 1.1 Refactor `data/feature_engineer.py`

Reduce from 157 features to ~70 (without sentiment) or ~85 (with sentiment). Remove:
- `returns`, `log_returns`, `momentum_1d` -- these correlate with target (leakage risk)
- Redundant VPI features, RSI divergence features (too noisy)
- Regime-conditional interaction features (regime_low_vol * macd, etc.)

Add proven indicators: ADX, Williams %R, CCI, TSI, MFI. Simplify sentiment to max 15 features. Update `EXPECTED_FEATURE_COUNT` and `get_feature_columns()`.

Feature categories (target ~70 without sentiment):
1. **Returns & Volatility** (8): volatility_5d/20d/ratio, garman_klass, parkinson, atr variants
2. **Momentum** (12): rsi_14/7, macd/signal/histogram, stoch_k/d, williams_r, cci_20, roc_10/20, tsi
3. **Trend** (10): sma/ema percent distances, price_to_sma, crosses, adx_14
4. **Volatility Bands** (6): bollinger upper/lower/width/position, keltner upper/lower
5. **Volume** (6): volume_sma_ratio, obv_slope, vwap_distance, mfi_14, vpt, a/d
6. **Multi-timeframe** (8): velocity_5d/10d/20d, acceleration_5d, rsi/volume velocity, momentum
7. **Pattern** (6): higher_high, lower_low, gaps, body_size, inside_bar
8. **Regime** (6): volatility_regime (3 one-hot), trend_strength/direction, mean_reversion
9. **Microstructure** (4): high_low_range, close_location, intraday_vol, overnight_return
10. **Cross-asset** (4, optional): vix_level/change, rv_iv_spread/zscore

### 1.2 Create `data/data_splitter.py` (NEW)

Implement `PurgedTimeSeriesSplit` based on Marcos Lopez de Prado's methodology:
- Embargo period (gap) between train and test to prevent label leakage
- Purging of overlapping samples near boundaries
- `create_train_val_test_split()` with configurable embargo_days (default 5)

### 1.3 Create `data/data_validator.py` (NEW)

Validation checks:
- `validate_no_leakage()`: No features correlate >0.95 with target
- `validate_no_nans()`: All NaN imputed
- `validate_feature_stationarity()`: ADF test on subset of features
- `validate_target_distribution()`: Balanced positive/negative split
- `run_all_validations()`: Combined report

### 1.4 Update `data/cache_manager.py`
Add version tracking to cache metadata. Stale caches (wrong feature count) auto-invalidate.

### Quality Gate
```bash
python -c "from data.feature_engineer import get_feature_columns; cols=get_feature_columns(False); print(len(cols)); assert 60<=len(cols)<=80"
python -c "from data.data_splitter import PurgedTimeSeriesSplit; print('OK')"
python -c "from data.data_validator import DataValidator; print('OK')"
python -m pytest tests/test_data_integrity.py tests/test_feature_count.py -v
```

---

## Phase 2: GBM-First Model System (Mandatory, Highest Priority)

**Goal:** Production-quality XGBoost + LightGBM with Optuna tuning, SHAP feature selection, walk-forward validation.

### 2.1 Refactor `training/train_gbm.py` (rename from train_gbm_baseline.py)

Create a `GBMTrainer` class with this pipeline:
1. **Load & validate data** using new data pipeline
2. **Optuna hyperparameter search** (30-50 trials) using `PurgedTimeSeriesSplit`
   - XGB params: n_estimators (200-2000), learning_rate (0.01-0.3 log), max_depth (3-8), subsample (0.6-1.0), colsample_bytree (0.5-1.0), min_child_weight (1-10), reg_alpha/lambda (1e-4 to 1.0 log)
   - LGB params: similar, plus num_leaves (20-150), min_data_in_leaf (5-50)
3. **SHAP feature selection**: Train quick model, compute SHAP values, select top 30-50 features
4. **Walk-forward validation** with purged CV
5. **Train final models** on full training data with best params
6. **Save**: model, scaler, selected feature list, SHAP importances, hyperparameters, metadata

Key points:
- Use `compute_regression_sample_weights()` from existing code (it works)
- Target is 1-day forward log-returns (from target_engineering.py)
- RobustScaler for features (keep existing approach)
- Both XGBoost and LightGBM trained and saved

### 2.2 Update `validation/walk_forward.py`
- Integrate PurgedTimeSeriesSplit
- Add per-fold Sharpe ratio computation
- Add feature stability check across folds

### 2.3 New saved model structure
```
saved_models/{SYMBOL}/
  gbm/
    xgb_model.joblib
    lgb_model.joblib
    xgb_scaler.joblib
    lgb_scaler.joblib
    feature_columns.pkl       # SHAP-selected features
    feature_importance.json
    hyperparameters.json
    training_metadata.json
```

### Quality Gate
```bash
python training/train_gbm.py AAPL --overwrite --tune --n-trials 30
# Verify outputs:
# pred_std > 0.005
# 30% < positive_pct < 70%
# WFE > 40%
# Direction accuracy > 51% on test set
# Feature importance saved, top features selected
# Models load for inference
```

---

## Phase 3: Optional Simplified LSTM (Can be deferred)

**Goal:** Simple 2-layer LSTM as ensemble secondary. NO transformer attention.

### 3.1 Create `models/simple_lstm.py`
Architecture: Input -> LSTM(64, return_sequences=True) -> Dropout(0.3) -> LSTM(32) -> Dropout(0.3) -> Dense(16, relu) -> Dense(1, linear). ~25K parameters. No attention, no positional encoding.

### 3.2 Create `training/train_lstm.py`
LSTMTrainer class with:
- Sequence length 30 (shorter = less overfitting)
- Batch size 512+ for GPU
- Huber loss (not MSE)
- LR warmup: 1e-4 -> 1e-3 over 5 epochs, then cosine annealing
- Max 50 epochs
- Variance collapse monitor every 5 epochs (stop if pred_std < 0.003)
- Save model, scaler, feature list, metadata

### Quality Gate
```bash
python training/train_lstm.py AAPL --epochs 50 --batch-size 512
# No variance collapse (pred_std > 0.005)
# Direction accuracy >= 50%
```

---

## Phase 4: Ensemble & Position Sizing (Can be deferred if only GBM)

### 4.1 Create `inference/ensemble.py`
`SimpleEnsemble` class: weighted average of GBM (70%) + LSTM (30%). If LSTM unavailable, falls back to GBM-only. Dynamic weight adjustment based on recent walk-forward performance.

### 4.2 Create `inference/position_sizing.py`
Half-Kelly position sizing:
- `f* = (p*b - q) / b` where p=win_rate, b=avg_win/avg_loss
- Half-Kelly: `f = f*/2`
- Scale by signal strength (predicted_return / prediction_std)
- Dead zone: if |predicted_return| < 0.1%, go flat
- Limits: max 100% long, max 30% short
- Drawdown circuit breaker: if drawdown > 15%, reduce max position to 25%

### Quality Gate
```bash
python -c "from inference.ensemble import SimpleEnsemble; e=SimpleEnsemble('AAPL'); print('OK')"
# Positions in [-0.3, 1.0]
# Balanced position distribution
```

---

## Phase 5: Backtesting & Validation (High Priority)

### 5.1 Create `run_backtest.py` (replaces 355KB monolith)
Clean CLI (~300 lines) with `UnifiedBacktester` class:
```bash
python run_backtest.py --symbol AAPL --start 2020-01-01 --end 2024-12-31
python run_backtest.py --symbol AAPL --mode gbm_only
python run_backtest.py --symbols AAPL,MSFT,GOOGL --compare
```

Uses existing `AdvancedBacktester` (which is excellent). Pipeline: load data -> features -> load models -> predict -> position sizing -> backtest with transaction costs -> compare vs buy-and-hold -> print report.

Report metrics: Sharpe ratio, annual return, max drawdown, win rate, direction accuracy, alpha vs buy-and-hold, turnover, total costs, cost-adjusted Sharpe.

### 5.2 Multi-symbol validation (DEFERRED -- future follow-up)
Start with AAPL only. Once AAPL works perfectly, expand to MSFT, GOOGL, etc. as a separate task.

### 5.3 Create `tests/test_backtest.py`
Tests: no look-ahead in positions, transaction costs applied, drawdown circuit breaker, buy-hold benchmark correct, multi-symbol generalization.

### Quality Gate
```bash
python run_backtest.py --symbol AAPL --start 2020-01-01 --end 2024-12-31
# Sharpe > 0.3 (conservative)
# Max drawdown < 35%
# Direction accuracy > 51%
# Cost-adjusted Sharpe still positive
python -m pytest tests/test_backtest.py -v
```

---

## Phase 6: API Integration (High Priority)

### 6.1 Rewrite `service/prediction_service.py`
Current file is 1348 lines of deprecated classifier logic. Rewrite to ~200 lines using new ensemble. Must output JSON matching `types/ai.ts` PredictionResult interface exactly.

### 6.2 Update `service/model_registry.py`
Scan new directory structure (`saved_models/{SYMBOL}/gbm/` and `saved_models/{SYMBOL}/lstm/`).

### 6.3 Update `app.py`
New fusion modes: `"gbm_only"`, `"lstm_only"`, `"ensemble"`. Remove classifier-specific fields.

### 6.4 Update `types/ai.ts` and frontend
Update `FusionMode` type, `ModelType` type. `classifierProbabilities` becomes optional empty array for backward compat.

### Quality Gate
```bash
cd python-ai-service && python app.py &
curl http://localhost:8000/api/health
curl -X POST http://localhost:8000/api/predict -H "Content-Type: application/json" -d '{"symbol":"AAPL","horizon":5,"daysOnChart":120}'
# Valid JSON response matching PredictionResult interface
```

---

## Priority Order (if running low on time/iterations)

1. **Phase 0** -- MANDATORY (cleanup)
2. **Phase 1** -- MANDATORY (data pipeline)
3. **Phase 2** -- MANDATORY (GBM training -- this IS the core)
4. **Phase 5** -- HIGH (backtest validation)
5. **Phase 6** -- HIGH (API so frontend works)
6. **Phase 3** -- DEFERRABLE (LSTM secondary)
7. **Phase 4** -- DEFERRABLE (ensemble, GBM-only is fine)

---

## Ralph Loop Instructions for the Executing Agent

### How to Work Autonomously

You are executing in a **Ralph Loop** -- a continuous iteration paradigm where you work, verify, and iterate without human intervention. Your state persists across iterations through files and git, not context windows.

### State Files
- **`implementation_progress.md`**: Log ALL work chronologically. Each iteration: date, what you did, what worked, what failed, metrics achieved.
- **Git commits**: Commit after each phase. Branch: `rewrite/gbm-first`.
- **`AGENTS.md`**: Update with discovered patterns, gotchas, and conventions.

### Each Iteration Should:
1. Read `implementation_progress.md` to understand current state
2. Check git log for latest commits
3. Determine which phase/step to work on next
4. Execute the work (write code, run training, run tests)
5. Run quality gate checks for current phase
6. Log results to `implementation_progress.md`
7. Commit working changes
8. If phase complete, move to next phase. If quality gate fails, diagnose and fix.

### Environment Setup (CRITICAL)
**ALWAYS activate the conda environment before ANY Python command:**
```bash
conda activate ai-stocks
```
This must be done at the START of every iteration. Without it, imports will fail.

**Install required new packages** (first iteration only):
```bash
conda activate ai-stocks
pip install optuna shap
```

### Symbol Strategy
Start with **AAPL only**. Get it working perfectly before expanding to other symbols. Multi-symbol validation is a follow-up task, not part of the core rewrite.

### Autonomous Training & Evaluation
You MUST actually run the training and evaluation commands, not just write code:
```bash
conda activate ai-stocks
cd /home/thunderboltdy/ai-stocks/python-ai-service

# Train GBM (Phase 2)
python training/train_gbm.py AAPL --overwrite

# Run backtest (Phase 5)
python run_backtest.py --symbol AAPL --start 2020-01-01 --end 2024-12-31

# Run all tests
python -m pytest tests/ -v

# Start API (Phase 6)
python app.py
```

### Failure Handling
- If a quality gate fails, attempt to fix (max 3 attempts per gate)
- Log EVERY failure with full error traceback in `implementation_progress.md`
- If 3 attempts fail on same gate, mark as "BLOCKED: [reason]" and move to next independent phase
- If a training run produces variance collapse (pred_std < 0.003), try: lower learning rate, fewer epochs, more regularization
- If predictions are biased (>70% one direction), check sample weights, check target distribution

### Freedom & Creativity
- You have **total freedom** over the codebase. Delete, rewrite, restructure anything.
- If you discover a better approach mid-execution, pivot to it. Log the reasoning.
- If you find more dead code not listed above, delete it.
- If a feature doesn't improve metrics, remove it.
- The plan is a guide, not a straitjacket. Adapt based on what you discover.

### Completion Signal
The rewrite is COMPLETE when ALL are true:
1. File count <= 45 Python files
2. GBM trains successfully on AAPL
3. Walk-forward WFE > 40%
4. Backtest Sharpe > 0 (positive, any amount)
5. Direction accuracy > 51%
6. No variance collapse (pred_std > 0.005)
7. No prediction bias (30% < positive_pct < 70%)
8. FastAPI starts and `/api/predict` returns valid JSON
9. `python -m pytest tests/ -v` -- all pass

### Key Files Reference
| File | Role | Action |
|------|------|--------|
| `data/feature_engineer.py` | Core features | Refactor to ~70 features |
| `training/train_gbm_baseline.py` | Primary model | Refactor with Optuna + SHAP |
| `service/prediction_service.py` | API layer | Complete rewrite (1348 -> ~200 lines) |
| `evaluation/advanced_backtester.py` | Backtester | KEEP unchanged (excellent) |
| `utils/model_paths.py` | Path management | Simplify, remove legacy |
| `utils/losses.py` | Loss functions | Simplify, keep SimpleDirectionalMSE |
| `types/ai.ts` | Frontend contract | Update FusionMode, ModelType |

---

## Verification Plan

After all phases complete:
1. **Unit tests**: `python -m pytest tests/ -v` -- all green
2. **Training smoke test**: `python training/train_gbm.py AAPL --overwrite` -- completes without error
3. **Backtest validation**: `python run_backtest.py --symbol AAPL` -- positive Sharpe, > 51% direction accuracy
4. **API test**: Start `python app.py`, hit prediction endpoint, get valid JSON
5. **Frontend test**: Run `npm run dev`, navigate to /ai page, select AAPL, verify chart renders
6. **Multi-symbol**: Train and backtest on 3+ symbols to confirm generalization
