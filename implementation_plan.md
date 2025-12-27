Nuclear Redesign: Python AI Service for Stock Prediction

 Executive Summary

 Complete redesign of the python-ai-service to create a
 business-standard, scientifically validated AI prediction system.
 This addresses critical data leakage, implements proper
 Walk-Forward validation, builds a true stacking ensemble with
 XGBoost meta-learner, and adds xLSTM-TS as a new model
 architecture.

 ---
 Part 1: Critical Issues to Fix

 1.1 DATA LEAKAGE IN GBM (P0 - CRITICAL)

 File: python-ai-service/training/train_gbm_baseline.py

 The Bug (Lines 691-775):
 # Line 691-694: 80/20 split
 split_idx = int(len(X) * 0.8)
 X_train, X_val = X_scaled[:split_idx], X_scaled[split_idx:]

 # Line 704-708: XGBoost uses validation for early stopping
 model.fit(X_train, y_train,
     eval_set=[(X_train, y_train), (X_val, y_val)],  # ← LEAKAGE
     verbose=False)

 # Line 732-737: LightGBM Phase 1 uses validation for iteration
 selection
 model_cv.fit(X_train, y_train,
     eval_set=[(X_train, y_train), (X_val, y_val)],  # ← LEAKAGE
     callbacks=cv_callbacks)

 # Line 771-775: Metrics on FULL dataset (including validation!)
 y_pred_full = model.predict(X_scaled)  # ← LEAKAGE: includes val
 data
 ic = np.corrcoef(y.flatten(), y_pred_full.flatten())[0, 1]  # ←
 INFLATED

 Impact: Validation data influences model selection, then metrics
 computed on same data = inflated results.

 1.2 FORWARD_SIM IS DEAD CODE (P1)

 File: python-ai-service/inference_and_backtest.py

 - --forward-sim and --forward-days CLI flags exist (lines
 7135-7138)
 - Parameters passed to main() (lines 4372-4373)
 - Never used in function body - no implementation exists

 1.3 LOOK-AHEAD BIAS IN BACKTEST (P0)

 Current Flow:
 - Features from day T predict day T+1
 - But backtest applies position to day T's return (should be T+1)
 - Creates systematic positive bias

 ---
 Part 2: Architecture Decisions

 2.1 Models to KEEP (Elite Tier)

 | Model              | File                                       |
  Rationale                                               |
 |--------------------|--------------------------------------------|
 ---------------------------------------------------------|
 | LSTM+Transformer   | models/lstm_transformer_paper.py           |
  Primary model, well-implemented with directional losses |
 | XGBoost            | training/train_gbm_baseline.py             |
  Robust, fast, needs leakage fix                         |
 | LightGBM           | training/train_gbm_baseline.py             |
  Best IC (0.652), needs leakage fix                      |
 | Binary Classifiers | training/train_binary_classifiers_final.py |
  Direction signals, properly implemented                 |

 2.2 Models to ADD

 | Model                | Priority | Rationale

        |
 |----------------------|----------|--------------------------------
 -------------------------------------------------------------------
 -------|
 | xLSTM-TS             | P1       | Specifically tested on S&P 500,
  outperforms TCN/N-BEATS for stock prediction, includes wavelet
 denoising |
 | XGBoost Meta-Learner | P0       | Replace naive fusion with
 trained stacking
             |

 2.3 Models to DEPRECATE

 | Model                       | Rationale
                                           |
 |-----------------------------|------------------------------------
 ------------------------------------------|
 | PatchTST                    | Implemented but not integrated -
 absorb architecture into xLSTM-TS or remove |
 | TFT                         | PyTorch-only creates complexity -
 defer                                      |
 | Argument-based fusion modes | Replace with trained stacking
 (weighted, balanced, gbm_heavy, etc.)          |

 2.4 Models to CONSIDER LATER (Phase 2)

 | Model | Rationale
       |
 |-------|----------------------------------------------------------
 ------|
 | TiDE  | 10x faster, MLP-based - good for speed-critical inference
       |
 | TCN   | Parallelizable, stable - good backup if xLSTM-TS
 underperforms |

 ---
 Part 3: Walk-Forward Validation Framework

 3.1 Configuration (User Selection: Anchored/Expanding)

 WalkForwardConfig = {
     'mode': 'anchored',           # Anchored (expanding) window
     'n_iterations': 5,            # 5 walk-forward folds
     'train_pct': 0.60,            # Minimum 60% training
     'validation_pct': 0.15,       # 15% validation
     'test_pct': 0.25,             # 25% out-of-sample
     'gap_days': 1,                # Gap between train/val
     'purge_days': 60,             # Purge overlapping sequences
 }

 3.2 Walk-Forward Efficiency (WFE) Metric

 WFE = (Test Sharpe / Validation Sharpe) * 100

 Interpretation:
 - WFE > 60%: Good - strategy is likely robust
 - WFE 40-60%: Acceptable - some overfitting
 - WFE < 40%: Poor - significant overfitting

 3.3 Data Split Visualization

 Iteration 1: [TRAIN TRAIN TRAIN TRAIN|GAP|VAL VAL|TEST TEST TEST]
 Iteration 2: [TRAIN TRAIN TRAIN TRAIN TRAIN|GAP|VAL VAL|TEST TEST]
 Iteration 3: [TRAIN TRAIN TRAIN TRAIN TRAIN TRAIN|GAP|VAL VAL|TEST]
              ^--- anchored at start, expands ---^

 3.4 Production Model (User Selection: Full Retrain)

 After WFE validation proves robustness (WFE > 50%):
 1. Retrain ALL models on 100% of available data
 2. No held-out set for production model
 3. Maximizes information for live predictions

 ---
 Part 4: Stacking Ensemble Architecture

 4.1 Architecture Overview

                     ┌─────────────────┐
                     │   Raw OHLCV     │
                     └────────┬────────┘
                              │
                     ┌────────▼────────┐
                     │Feature Engineer │
                     │   (157 features)│
                     └────────┬────────┘
                              │
     ┌────────────────────────┼────────────────────────┐
     │                        │                        │
     ▼                        ▼                        ▼
 ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
 │LSTM+Trans   │      │  xLSTM-TS   │      │  GBM        │
 │(existing)   │      │  (NEW)      │      │(XGB+LGB)    │
 └──────┬──────┘      └──────┬──────┘      └──────┬──────┘
        │                    │                    │
        │ pred_lstm          │ pred_xlstm         │ pred_gbm
        │                    │                    │
        └────────────────────┼────────────────────┘
                             │
                    ┌────────▼────────┐
                    │  REGIME DETECTOR │
                    │(vol, trend, sent)│
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │  XGBoost        │
                    │  META-LEARNER   │
                    │                 │
                    │ Inputs:         │
                    │ - Base preds    │
                    │ - Regime feats  │
                    │ - Disagreement  │
                    │ - Recent errors │
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ FINAL PREDICTION│
                    └─────────────────┘

 4.2 Meta-Learner Features

 META_FEATURES = [
     # Base model predictions
     'pred_lstm', 'pred_xlstm', 'pred_xgb', 'pred_lgb',

     # Regime indicators
     'vol_regime_high', 'vol_regime_normal', 'vol_regime_low',
     'trend_bullish', 'trend_bearish',
     'sentiment_positive', 'sentiment_negative',

     # Model agreement/disagreement
     'prediction_std',      # std of all predictions
     'sign_agreement',      # 1 if all same sign, 0 otherwise
     'max_min_spread',      # max(preds) - min(preds)

     # Recent model quality (trailing 20-day)
     'lstm_recent_mae',
     'xlstm_recent_mae',
     'gbm_recent_mae',
 ]

 4.3 Two-Stage Training Process

 Stage 1: Generate OOF Predictions
 For each base model:
     Run Walk-Forward CV → Collect out-of-fold predictions

 Stage 2: Train Meta-Learner
 Stack OOF predictions + regime features
 Train XGBoost meta-learner on stacked features

 Stage 3: Production Models
 Retrain all base models on 100% data
 Retrain meta-learner on 100% data
 Deploy ensemble

 ---
 Part 5: Implementation Plan

 Phase 1: Critical Bug Fixes (P0)

 Step 1.1: Fix GBM Data Leakage

 File: python-ai-service/training/train_gbm_baseline.py

 Changes:
 1. Add proper 3-way split: Train (60%) / Val (20%) / Test (20%)
 2. Use validation ONLY for early stopping, never for final training
 3. Compute final metrics ONLY on held-out test set
 4. Add WFE calculation

 def split_data_walkforward(X, y, train_pct=0.60, val_pct=0.20):
     """Split data with clear boundaries and NO LEAKAGE."""
     n = len(X)
     train_end = int(n * train_pct)
     val_end = train_end + int(n * val_pct)

     return {
         'train': (X[:train_end], y[:train_end]),
         'val': (X[train_end:val_end], y[train_end:val_end]),
         'test': (X[val_end:], y[val_end:]),  # NEVER TOUCHED
     }

 Step 1.2: Fix Backtest Alignment

 File: python-ai-service/evaluation/advanced_backtester.py

 Changes:
 1. Shift predictions by 1 day: position from day T applied to day
 T+1
 2. Add logging of prediction date vs execution date
 3. Add sanity check

 # Position from T applied to T+1's return
 positions_lagged = np.zeros(len(positions))
 positions_lagged[1:] = positions[:-1]  # Lag by 1
 positions_lagged[0] = 0  # No position on first day

 Phase 2: Walk-Forward Framework (P0)

 Step 2.1: Create Walk-Forward Module

 New File: python-ai-service/validation/walk_forward.py

 class WalkForwardValidator:
     """
     Anchored (expanding) walk-forward validation.

     Key features:
     - Proper purging for sequence models
     - WFE metric calculation
     - No data leakage guarantee
     """

     def generate_splits(self, n_samples) -> List[WalkForwardSplit]:
         """Generate train/val/test splits with no overlap."""
         ...

     def validate(self, model_factory, X, y) -> WalkForwardResults:
         """Run full walk-forward validation."""
         ...

 Step 2.2: Create WFE Metrics Module

 New File: python-ai-service/validation/wfe_metrics.py

 def calculate_wfe(val_sharpe: float, test_sharpe: float) -> float:
     """Walk Forward Efficiency."""
     if val_sharpe <= 0:
         return 0.0
     return (test_sharpe / val_sharpe) * 100

 Phase 3: Stacking Ensemble (P0)

 Step 3.1: Stacking Trainer

 New File: python-ai-service/training/train_stacking_ensemble.py

 class StackingEnsembleTrainer:
     """
     Two-stage stacking with XGBoost meta-learner.

     Stage 1: Generate OOF predictions via walk-forward CV
     Stage 2: Train XGBoost meta-learner on OOF + regime features
     Stage 3: Train production models on all data
     """

 Step 3.2: Stacking Predictor

 New File: python-ai-service/inference/stacking_predictor.py

 class StackingPredictor:
     """
     Production inference with stacking ensemble.
     Replaces argument-based fusion modes.
     """

     def predict(self, features) -> PredictionResult:
         # Get base model predictions
         # Compute regime features
         # Run through meta-learner
         # Return final prediction with confidence

 Phase 4: xLSTM-TS Model (P1)

 Step 4.1: Implement xLSTM-TS Architecture

 New File: python-ai-service/models/xlstm_ts.py

 Key features:
 - Exponential gating for better normalization
 - Revised memory structure (scalar + matrix variants)
 - Wavelet denoising integration
 - Residual block backbone

 class xLSTMBlock(keras.layers.Layer):
     """
     Extended LSTM block with:
     - Exponential gating
     - Matrix memory variant (mLSTM)
     - Scalar memory variant (sLSTM)
     """

 class xLSTM_TS(keras.Model):
     """
     xLSTM optimized for time series.
     Based on Beck et al. (2024) with wavelet denoising.
     """

 Step 4.2: Training Script

 New File: python-ai-service/training/train_xlstm_ts.py

 def train_xlstm_ts(symbol, config):
     """
     Train xLSTM-TS model with:
     - Wavelet denoising preprocessing
     - AntiCollapseDirectionalLoss
     - Walk-forward validation
     """

 Phase 5: Forward Simulation (P1)

 Step 5.1: Implement True Forward Simulation

 New File: python-ai-service/evaluation/forward_simulator.py

 class ForwardSimulator:
     """
     True forward-looking simulation with NO look-ahead.

     At each step T:
     1. Access ONLY data up to T-1
     2. Engineer features from historical data
     3. Make prediction for T
     4. Execute at T's open
     5. Record result at T's close
     """

     def simulate(self, model, data, start_date, end_date):
         """Step-by-step simulation without future access."""
         for T in date_range(start_date, end_date):
             historical_data = data.loc[:T - 1]  # Only past
             features = engineer_features(historical_data)
             prediction = model.predict(features[-seq_len:])
             # Execute and record...

 Phase 6: Production Pipeline (P1)

 Step 6.1: End-to-End Pipeline

 New File: python-ai-service/pipeline/production_pipeline.py

 class ProductionPipeline:
     """
     Complete pipeline separating:
     1. VALIDATION: Walk-forward CV to prove robustness
     2. PRODUCTION: Train on ALL data, deploy for inference
     """

     def run_validation(self, symbol) -> ValidationReport:
         """Run walk-forward validation, return WFE metrics."""

     def train_production(self, symbol) -> ProductionModel:
         """After validation passes, train on 100% data."""

     def predict(self, symbol, latest_data) -> Prediction:
         """Generate live prediction using production model."""

 Phase 7: Cleanup & Deprecation (P2)

 Step 7.1: Remove Dead Code

 - Remove forward_sim and forward_days parameters from
 inference_and_backtest.py
 - Remove unused fusion mode functions from hybrid_predictor.py

 Step 7.2: Archive Deprecated Files

 - Move inference/horizon_weighting.py to deprecated/
 - Move training/rolling_cv.py to deprecated/ (replaced by
 walk_forward.py)

 ---
 Part 6: Files Summary

 Files to CREATE

 | File                                | Purpose
        | Priority |
 |-------------------------------------|----------------------------
 -------|----------|
 | validation/walk_forward.py          | Walk-forward validation
 framework | P0       |
 | validation/wfe_metrics.py           | WFE and validation metrics
        | P0       |
 | training/train_stacking_ensemble.py | Stacking ensemble trainer
        | P0       |
 | inference/stacking_predictor.py     | Production stacking
 inference     | P0       |
 | models/xlstm_ts.py                  | xLSTM-TS model architecture
        | P1       |
 | training/train_xlstm_ts.py          | xLSTM-TS training script
        | P1       |
 | evaluation/forward_simulator.py     | True forward simulation
        | P1       |
 | pipeline/production_pipeline.py     | End-to-end production
 pipeline    | P1       |
 | tests/test_walk_forward.py          | Walk-forward unit tests
        | P0       |
 | tests/test_stacking_integration.py  | Integration tests
        | P1       |
 | tests/test_no_leakage.py            | Data leakage verification
 tests   | P0       |

 Files to MODIFY

 | File                                 | Changes
                   | Priority |
 |--------------------------------------|---------------------------
 ------------------|----------|
 | training/train_gbm_baseline.py       | Fix data leakage, add WFE
                   | P0       |
 | evaluation/advanced_backtester.py    | Fix 1-day lag alignment
                   | P0       |
 | training/train_1d_regressor_final.py | Integrate walk-forward
                   | P1       |
 | inference_and_backtest.py            | Remove dead forward_sim,
 integrate stacking | P1       |
 | inference/hybrid_predictor.py        | Replace fusion modes with
 stacking          | P1       |

 Files to DEPRECATE/ARCHIVE

 | File                                         | Reason
             |
 |----------------------------------------------|-------------------
 ------------|
 | training/rolling_cv.py                       | Superseded by
 walk_forward.py |
 | inference/horizon_weighting.py               | Replaced by
 meta-learner      |
 | Fusion mode functions in hybrid_predictor.py | Replaced by
 stacking          |

 ---
 Part 7: Testing & Validation Strategy

 7.1 No-Leakage Tests (P0)

 def test_gbm_no_leakage():
     """Verify GBM training has no data leakage."""
     # Train model
     model, metrics = trainer.train(X, y)

     # Verify metrics computed ONLY on held-out test
     assert metrics['test_samples'] < len(X) * 0.25
     assert 'validation_used_for_training' not in metrics

 def test_walk_forward_no_overlap():
     """Ensure train/val/test splits have no overlap."""
     splits = validator.generate_splits(n_samples=1000)

     for split in splits:
         assert split.train[1] < split.val[0]  # No train/val
 overlap
         assert split.val[1] < split.test[0]   # No val/test overlap

 7.2 WFE Threshold Tests (P0)

 def test_walk_forward_efficiency():
     """Ensure models pass WFE threshold."""
     report = pipeline.run_validation('AAPL')

     assert report.aggregate_wfe > 40, f"WFE too low:
 {report.aggregate_wfe}%"

 7.3 Variance Collapse Tests (Existing - Enhance)

 def test_variance_collapse_all_models():
     """Ensure no model exhibits variance collapse."""
     for model_type in ['lstm', 'xlstm', 'xgboost', 'lightgbm',
 'stacking']:
         predictions = model.predict(X_test)

         assert np.std(predictions) > 0.005  # Variance check
         assert (predictions > 0).mean() > 0.30  # Not all positive
         assert (predictions < 0).mean() > 0.30  # Not all negative

 ---
 Part 8: Success Criteria

 1. No Data Leakage: All metrics computed on truly held-out data
 2. WFE > 50%: Walk-forward efficiency proves robustness
 3. Stacking Outperforms Naive: Meta-learner beats simple averaging
 4. xLSTM-TS Integration: New model adds value to ensemble
 5. Forward Sim Works: True step-by-step simulation without
 look-ahead
 6. All Tests Pass: Including leakage, WFE, variance collapse tests

 ---
 Research Sources

 - https://research.google/pubs/long-horizon-forecasting-with-tide-t
 ime-series-dense-encoder/
 - https://github.com/gonzalopezgil/xlstm-ts
 - https://arxiv.org/pdf/2407.10240
 -
 https://blog.quantinsti.com/walk-forward-optimization-introduction/
 - https://www.mdpi.com/2227-7072/13/4/201
 - https://www.sciencedirect.com/science/article/abs/pii/S0378437119
 313093
