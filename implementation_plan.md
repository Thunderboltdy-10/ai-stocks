 Nuclear Model Redesign Plan

 Executive Summary

 After extensive analysis and deep research, the true root causes have been identified:

 Critical Discovery: Variance Collapse is Architecture-Induced

 Research finding: Regularization (weight decay) directly causes Neural Regression Collapse.
 Even minimal regularization triggers collapse; without it, no collapse occurs. The solution
 is architecture-based, not loss function penalties.

 Root Causes Identified

 1. LSTM Variance Collapse:
   - Loss function penalties (AntiCollapseDirectionalLoss) create competing objectives
   - Architecture lacks residual connections for difference prediction
   - Regularization induces collapse (research-proven)
 2. GBM Directional Bias:
   - LightGBM doesn't capture trend (predictions poor when data exceeds historical range)
   - Sample weighting insufficient for natural market bias
   - Should use log-returns as target
 3. Ensemble Issues:
   - Meta-learner trained on contaminated data (not strictly OOF)
   - Missing non-negative weight constraints

 Recommended Approach

 Phase 1: Infrastructure (logging system, agent updates)
 Phase 2: Architecture fixes (residual connections, difference targets)
 Phase 3: Loss function simplification (remove anti-collapse penalties)
 Phase 4: GBM improvements (CatBoost, log-returns)
 Phase 5: Ensemble fixes (walk-forward OOF, constrained meta-learner)

 ---
 Research Findings (Evidence-Based)

 Variance Collapse Prevention (Key Finding)

 Root Cause Proven: Regularization (weight decay) causes Neural Regression Collapse.
 - Without regularization: No collapse
 - With even minimal regularization: Collapse emerges

 Proven Solutions (architecture-based, not loss-based):
 1. Residual/Difference Prediction: Predict change from previous value, not absolute
 2. Initialize output layer with zeros: Ensures initial predicted changes are small
 3. Stateful LSTMs: stateful=True provides long memory
 4. Stacked LSTMs with return_sequences=True: Passes temporal information

 Loss Functions (Research-Backed)

 | Loss Function                         | Performance                               |
 |---------------------------------------|-------------------------------------------|
 | Custom Asymmetric                     | Outperforms MSE AND beats buy-and-hold    |
 | MSE alone                             | Never beats buy-and-hold                  |
 | MADL (Mean Absolute Directional Loss) | Directly evaluates profit/loss            |
 | Huber + Directional                   | Less sensitive to outliers, good accuracy |

 Recommended: L = 0.6 * DirectionalLoss + 0.4 * MSE

 GBM Findings

 - LightGBM doesn't capture trend - predictions poor when data exceeds historical range
 - CatBoost's Ordered Boosting mitigates target leakage and prediction bias
 - Use log-returns as target for more stationary distribution

 Ensemble Methods

 - Stacking gives 10-15% accuracy improvement but risky for look-ahead bias
 - Critical: Constrain meta-learner weights to non-negative (use ElasticNet)
 - Strictly walk-forward OOF predictions for meta-learner training

 Realistic Benchmarks

 | Metric               | Realistic | Good    | Excellent |
 |----------------------|-----------|---------|-----------|
 | Sharpe Ratio         | 0.6-0.8   | 0.8-1.2 | 1.2+      |
 | Directional Accuracy | 52-55%    | 55-60%  | 60-65%    |
 | Beat Buy & Hold      | 5-10%     | 10-20%  | 20%+      |

 Note: Claims of >90% accuracy are often cherry-picked. Real out-of-sample: 55-65%.

 ---
 User Preferences

 - Training: User runs commands, Claude analyzes logs automatically
 - Architecture: Research-first, open to fixing current or trying new
 - Risk Tolerance: Maximize returns (higher risk acceptable)
 - Timeline: Take time to do it right

 ---
 Implementation Plan

 Phase 1: Logging Infrastructure (Priority 0)

 Create automatic logging system so user can run training and Claude can analyze.

 1.1 Standardized Log Format

 All training scripts will output logs to:
 python-ai-service/training_logs/{SYMBOL}_{MODEL}_{TIMESTAMP}.log

 Example filenames:
 - AAPL_lstm_20251229_143022.log
 - TSLA_gbm_xgb_20251229_150133.log
 - SPY_xlstm_20251229_161544.log
 - AAPL_backtest_20251229_170012.log

 1.2 Log Content Structure

 Each log will contain standardized sections:
 === TRAINING CONFIG ===
 Symbol: AAPL
 Model: lstm_regressor
 Epochs: 50
 Batch Size: 512
 Learning Rate: 1e-3
 Features: 154

 === EPOCH METRICS ===
 Epoch 1: loss=0.0234, val_loss=0.0289, pred_std=0.0512, pos_pct=48.2%
 Epoch 2: loss=0.0198, val_loss=0.0267, pred_std=0.0498, pos_pct=51.3%
 ...

 === FINAL METRICS ===
 Direction Accuracy: 54.2%
 Prediction Std: 0.0423
 Positive %: 52.1%
 WFE: 62.3%
 Test Sharpe: 0.87

 === WARNINGS ===
 [None or list of issues]

 === STATUS ===
 SUCCESS / FAILED / COLLAPSED

 1.3 Files to Create/Modify

 | File                                       | Change                           |
 |--------------------------------------------|----------------------------------|
 | python-ai-service/utils/training_logger.py | New standardized logging utility |
 | training/train_1d_regressor_final.py       | Integrate new logger             |
 | training/train_gbm_baseline.py             | Integrate new logger             |
 | training/train_xlstm_ts.py                 | Integrate new logger             |
 | inference_and_backtest.py                  | Integrate new logger             |

 ---
 Phase 2: Agent Redesign

 2.1 Command-Executor Agent (Complete Redesign)

 New purpose: Log Analyzer Agent

 The agent will:
 1. Scan training_logs/ directory for new logs
 2. Parse standardized log format
 3. Extract key metrics and warnings
 4. Report status to user
 5. Detect patterns across multiple training runs

 2.2 Deep-Researcher Agent Updates

 Add codebase context:
 - Feature contract (154 features after leakage fix)
 - Current loss functions and their issues
 - Known failure modes with solutions
 - Research-backed recommendations

 2.3 Code-Analyzer Agent Updates

 - Update root causes to match current system
 - Add iteration strategy for cascading fixes
 - Include verification checklist

 ---
 Phase 3: LSTM Architecture Fix

 3.1 Remove AntiCollapseDirectionalLoss

 The complex loss with competing objectives is the primary cause of instability.

 Replace with:
 # Simple directional loss without anti-collapse penalties
 class DirectionalMSELoss(keras.losses.Loss):
     def __init__(self, direction_weight=0.5, **kwargs):
         super().__init__(**kwargs)
         self.direction_weight = direction_weight

     def call(self, y_true, y_pred):
         mse = ops.square(y_true - y_pred)
         # Penalize wrong direction
         wrong_dir = ops.cast(
             ops.sign(y_true) != ops.sign(y_pred),
             y_pred.dtype
         )
         directional = ops.abs(y_true - y_pred) * wrong_dir
         return ops.mean(mse + self.direction_weight * directional)

 3.2 Add Residual Connection to Model

 Predict difference from previous instead of absolute value:

 # In LSTMTransformerPaper model
 def call(self, inputs, training=False):
     # Get last value from sequence for residual
     last_value = inputs[:, -1, 0:1]  # Assuming first feature is price/return

     # Normal forward pass
     x = self.lstm(inputs)
     # ... transformer blocks ...
     prediction_delta = self.output_dense(x)  # Predict CHANGE

     # Residual connection: final = last + delta
     return last_value + prediction_delta

 3.3 Initialize Output Layer with Zeros

 self.output_dense = layers.Dense(
     1,
     kernel_initializer='zeros',  # Start predicting zero change
     bias_initializer='zeros'
 )

 3.4 Learning Rate Fix

 # Replace aggressive cosine decay with ReduceLROnPlateau
 optimizer = keras.optimizers.Adam(learning_rate=1e-3, clipnorm=1.0)

 callbacks = [
     keras.callbacks.ReduceLROnPlateau(
         monitor='val_loss',
         factor=0.5,
         patience=10,
         min_lr=1e-6
     ),
     keras.callbacks.EarlyStopping(
         patience=30,  # Increased from 15
         restore_best_weights=True
     )
 ]

 ---
 Phase 4: GBM Fix

 4.1 Add CatBoost as Alternative

 CatBoost's Ordered Boosting prevents target leakage and prediction bias.

 from catboost import CatBoostRegressor

 model = CatBoostRegressor(
     iterations=1000,
     learning_rate=0.03,
     depth=6,
     l2_leaf_reg=3.0,
     loss_function='RMSE',
     eval_metric='RMSE',
     early_stopping_rounds=100,
     verbose=100
 )

 4.2 Use Log-Returns as Target

 # Transform target to log-returns (more stationary)
 y_train = np.log1p(y_train)  # log(1 + return)
 y_val = np.log1p(y_val)

 # After prediction, transform back
 y_pred = np.expm1(model.predict(X_test))  # exp(pred) - 1

 4.3 Restore Regularization

 # XGBoost
 params = {
     'reg_alpha': 0.01,   # Restore from 0.0001
     'reg_lambda': 0.01,  # Restore from 0.0001
     'max_depth': 6,      # Limit depth
     'learning_rate': 0.05,
     'subsample': 0.8,
     'colsample_bytree': 0.8
 }

 ---
 Phase 5: Ensemble Fix

 5.1 Strictly Walk-Forward OOF

 # Generate OOF predictions for meta-learner
 oof_preds = np.zeros((len(X), n_models))

 for fold_idx, (train_idx, val_idx) in enumerate(walk_forward_splits):
     # Train each base model on train_idx
     # Predict on val_idx only
     oof_preds[val_idx, :] = base_predictions

 5.2 Constrained Meta-Learner

 from sklearn.linear_model import ElasticNet

 meta_learner = ElasticNet(
     alpha=0.01,
     l1_ratio=0.5,
     positive=True,  # Force non-negative weights!
     max_iter=1000
 )

 ---
 Testing Protocol

 Validation Checkpoints

 | Checkpoint              | Criteria            | Action if Fail            |
 |-------------------------|---------------------|---------------------------|
 | 1. Training completes   | No NaN, no crash    | Check loss function, LR   |
 | 2. Variance maintained  | pred_std > 0.01     | Check residual connection |
 | 3. Balanced predictions | 35-65% positive     | Check sample weights      |
 | 4. WFE > 50%            | Walk-forward passes | Check for overfitting     |
 | 5. Backtest positive    | Beats buy & hold    | Check transaction costs   |

 Success Metrics

 | Metric             | Minimum | Target | Stretch |
 |--------------------|---------|--------|---------|
 | Sharpe Ratio       | > 0.6   | > 1.0  | > 1.5   |
 | Direction Accuracy | > 52%   | > 55%  | > 60%   |
 | Beat Buy & Hold    | > 5%    | > 15%  | > 25%   |
 | Max Drawdown       | < 35%   | < 25%  | < 15%   |
 | WFE                | > 50%   | > 60%  | > 70%   |

 ---
 Files to Modify (Priority Order)

 | Priority | File                                 | Change
                 |
 |----------|--------------------------------------|------------------------------------------
 ----------------|
 | P0       | utils/training_logger.py             | CREATE: Standardized logging utility
                 |
 | P0       | .claude/agents/command-executor.md   | REWRITE: Log analyzer agent
                 |
 | P1       | models/lstm_transformer_paper.py     | ADD residual connection, zero init
                 |
 | P1       | utils/losses.py                      | SIMPLIFY: Remove anti-collapse, keep
 directional         |
 | P1       | training/train_1d_regressor_final.py | FIX: LR, early stopping, integrate logger
                 |
 | P1       | training/train_gbm_baseline.py       | ADD CatBoost, log-returns, restore reg,
 integrate logger |
 | P2       | training/train_stacking_ensemble.py  | FIX: Walk-forward OOF, constrained
 meta-learner          |
 | P2       | .claude/agents/deep-researcher.md    | UPDATE: Add codebase context
                 |
 | P2       | .claude/agents/code-analyzer.md      | UPDATE: Current failure modes
                 |

 ---
 Workflow (User + Claude)

 Training Workflow

 1. Claude: Makes code changes, provides training command
 2. User: Runs training command (faster execution)
 3. Training Script: Automatically writes to training_logs/
 4. Claude: Analyzes log using command-executor agent
 5. Claude: Diagnoses issues, proposes next iteration
 6. Repeat until success metrics met

 Example Session

 Claude: "I've updated the LSTM architecture. Please run:"
         python training/train_1d_regressor_final.py AAPL --epochs 50

 User: [runs command]

 Claude: "Analyzing training_logs/AAPL_lstm_20251229_143022.log..."
         "Result: Direction accuracy 54.2%, pred_std 0.042 - GOOD"
         "WFE: 58.3% - PASS"
         "Ready for backtest. Please run:"
         python inference_and_backtest.py --symbol AAPL

 ---
 Risk Mitigation

 | Risk                       | Mitigation                             |
 |----------------------------|----------------------------------------|
 | New architecture fails     | Keep current models as fallback        |
 | CatBoost not available     | XGBoost with better params as backup   |
 | Residual connection breaks | Test incrementally with unit tests     |
 | Log system breaks training | Wrap in try/catch, don't fail training |

 ---
 Timeline Estimate

 | Phase   | Work                                    |
 |---------|-----------------------------------------|
 | Phase 1 | Logging infrastructure + agent redesign |
 | Phase 2 | LSTM architecture fixes                 |
 | Phase 3 | GBM fixes (CatBoost, log-returns)       |
 | Phase 4 | Ensemble fixes                          |
 | Phase 5 | Multi-symbol validation                 |
 | Phase 6 | Final tuning and documentation          |

 Each phase includes: implementation, user training run, log analysis, iteration.