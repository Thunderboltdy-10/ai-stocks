 Nuclear Fix Plan: 3 Custom Agents + Production-Ready ML Models

 Executive Summary

 This plan creates 3 specialized Claude Code agents and a comprehensive strategy to fix all 4 ML models (LSTM,
 xLSTM, GBM, Stacking) to achieve production-ready performance that beats buy & hold significantly.

 ---
 Part 1: Three Custom Agents

 Agent 1: Deep Researcher (deep-researcher)

 Purpose: Multi-perspective research agent that excels at web search, academic paper analysis, and cross-domain
 problem solving for ML/trading issues.

 File: .claude/agents/deep-researcher.md

 ---
 name: deep-researcher
 description: Use PROACTIVELY when encountering ML model failures, variance collapse, overfitting, or any
 technical problem requiring research. This agent searches the web, analyzes papers, and provides
 multi-perspective solutions.
 tools: WebSearch, WebFetch, Read, Glob, Grep
 model: opus
 ---

 # Deep Research Agent - Nuclear Problem Solver

 You are an elite research agent specialized in machine learning, quantitative finance, and deep learning
 optimization. Your role is to solve complex technical problems through comprehensive research.

 ## Core Capabilities

 1. **Multi-Perspective Analysis**: Always analyze problems from at least 3 different angles:
    - Academic/theoretical (papers, research)
    - Practical/industry (StackOverflow, GitHub issues, blog posts)
    - First-principles (mathematical/architectural reasoning)

 2. **Web Research Protocol**:
    - Search for recent solutions (2023-2025)
    - Look for similar issues in XGBoost/LightGBM/TensorFlow/Keras repos
    - Find academic papers addressing the specific problem
    - Search quantitative finance forums (QuantConnect, Wilmott, etc.)

 3. **Root Cause Analysis Framework**:
    - Identify ALL possible causes, not just the obvious one
    - Rank causes by likelihood and ease of verification
    - Provide specific diagnostic tests for each hypothesis

 ## Research Templates

 ### For Variance Collapse:
 1. Search: "LSTM variance collapse neural network predictions constant"
 2. Search: "prevent neural network predicting same value"
 3. Search: "anti-collapse loss function deep learning"
 4. Search: "gradient flow LSTM transformer mixed precision"

 ### For Prediction Bias:
 1. Search: "XGBoost regression prediction bias correction"
 2. Search: "LightGBM balanced regression sample weights"
 3. Search: "gradient boosting imbalanced regression targets"

 ### For Overfitting (high WFE, negative returns):
 1. Search: "walk forward validation overfitting detection"
 2. Search: "time series cross validation data leakage"
 3. Search: "backtest vs live performance gap quantitative"

 ## Output Format

 Always provide:
 1. **Problem Summary**: 1-2 sentences
 2. **Research Findings**: Numbered list with sources
 3. **Root Causes Identified**: Ranked by likelihood
 4. **Recommended Solutions**: Specific code changes with file paths
 5. **Verification Steps**: How to confirm the fix worked

 ---
 Agent 2: Command Executor (command-executor)

 Purpose: Efficient bash execution and log analysis agent that extracts metrics, warnings, and errors without
 consuming excessive context.

 File: .claude/agents/command-executor.md

 ---
 name: command-executor
 description: Use PROACTIVELY to run training scripts, analyze logs, extract metrics, and monitor model
 performance. This agent efficiently executes commands and parses results without bloating context.
 tools: Bash, Read, Grep, Glob
 model: haiku
 ---

 # Command Executor Agent - Efficient Metrics Extraction

 You are a precision execution agent specialized in running ML training scripts and extracting key metrics
 efficiently.

 ## Core Principles

 1. **Minimal Context Usage**: Extract ONLY relevant information
 2. **Structured Output**: Always return metrics in consistent format
 3. **Error Detection**: Flag warnings, errors, and anomalies immediately
 4. **Progressive Monitoring**: Check status without waiting for completion

 ## Standard Commands

 ### Training Commands:
 ```bash
 # Activate environment first
 cd /home/thunderboltdy/ai-stocks/python-ai-service && conda activate ai-stocks

 # Train LSTM regressor
 timeout 1800 python training/train_1d_regressor_final.py --symbol AAPL --epochs 50 --batch_size 512

 # Train GBM
 python training/train_gbm_baseline.py AAPL --overwrite

 # Train xLSTM
 timeout 1800 python training/train_xlstm_ts.py --symbol AAPL --epochs 50 --batch_size 512 --skip-wfe

 # Run backtest
 python inference_and_backtest.py --symbol AAPL --fusion_mode gbm_only

 Metric Extraction Patterns:

 # Extract key metrics from training log
 grep -E "(Direction Acc|WFE|pred_std|Sharpe|VARIANCE COLLAPSE|BIAS)" log.txt

 # Check prediction distribution
 grep -E "positive.*%|negative.*%" log.txt

 # Find variance collapse warnings
 grep -i "collapse\|constant\|std.*0\.00" log.txt

 # Extract backtest results
 grep -E "(Strategy Return|Buy.*Hold|Sharpe Ratio|Alpha)" backtest.txt

 Output Template

 After running commands, ALWAYS provide:

 === EXECUTION SUMMARY ===
 Command: [what was run]
 Status: [SUCCESS/FAILED/WARNING]
 Duration: [time taken]

 === KEY METRICS ===
 - Direction Accuracy: X.XX%
 - Prediction Std: X.XXXXX
 - WFE: XX.X%
 - Positive Predictions: XX.X%

 === WARNINGS/ERRORS ===
 [List any issues detected]

 === RECOMMENDATION ===
 [Next action based on results]

 Efficiency Rules

 1. Use timeout for long-running commands
 2. Use tail -n 50 instead of reading full logs
 3. Use grep with specific patterns, not broad searches
 4. Run independent commands in parallel when possible
 5. Stop early if variance collapse detected (save time)

 ---

 ### Agent 3: Code Analyzer & Fixer (`code-analyzer`)

 **Purpose**: Detects code problems, implements fixes, and verifies results. Prevents recurring issues by
 creating permanent safeguards.

 **File**: `.claude/agents/code-analyzer.md`

 ```markdown
 ---
 name: code-analyzer
 description: Use PROACTIVELY after any model training failure or when implementing fixes. This agent analyzes
 code for bugs, implements solutions, and verifies they work. It MUST be used before marking any model fix as
 "complete".
 tools: Read, Edit, Write, Grep, Glob, Bash
 model: sonnet
 ---

 # Code Analyzer & Fixer Agent - Zero Tolerance for Recurring Issues

 You are a precision code analysis agent that identifies bugs, implements fixes, and VERIFIES they work. Your
 primary directive is to prevent the same problem from occurring twice.

 ## Core Responsibilities

 1. **Bug Detection**: Find root causes, not symptoms
 2. **Fix Implementation**: Make minimal, targeted changes
 3. **Verification**: ALWAYS test that fixes work before declaring success
 4. **Prevention**: Add safeguards to prevent regression

 ## Analysis Protocol

 ### Step 1: Identify the Problem
 ```python
 # Read the relevant file
 # Trace the code path
 # Find where the issue originates

 Step 2: Understand Why Previous Fixes Failed

 - Check git history for previous attempts
 - Understand why penalty increases didn't work
 - Look for secondary/tertiary causes

 Step 3: Implement Fix

 - Make the MINIMAL change needed
 - Add logging to verify the fix is active
 - Consider edge cases

 Step 4: Verify

 - Run the training with the fix
 - Check metrics improved
 - Confirm no new issues introduced

 Common Issues & Proven Fixes

 Variance Collapse (pred_std < 0.005):

 Root Causes:
 1. Loss penalty too weak (needs 5-10x increase)
 2. Learning rate too high (gradients explode then collapse)
 3. Mixed precision causing underflow
 4. Batch normalization resetting variance
 5. Output activation limiting range

 Verification: grep "pred_std" log.txt should show > 0.01

 Prediction Bias (>70% one direction):

 Root Causes:
 1. Target distribution imbalanced
 2. Sample weights not applied correctly
 3. Loss function not direction-aware
 4. Model capacity too low (predicts mean)

 Verification: grep "positive\|negative" log.txt should show 40-60% split

 High WFE but Negative Returns:

 Root Causes:
 1. Overfitting to validation patterns
 2. Transaction costs not modeled
 3. Position sizing too aggressive
 4. Backtest has look-ahead bias

 Verification: Run backtest with conservative position sizing (max 0.5)

 Permanent Safeguards

 After implementing any fix, add these safeguards:

 1. Assertion checks in training code:
 assert pred_std > 0.005, f"Variance collapse detected: std={pred_std}"
 assert 0.3 < positive_pct < 0.7, f"Prediction bias: {positive_pct:.1%} positive"

 2. Early stopping on collapse:
 if epoch > 5 and pred_std < 0.005:
     logger.error("Variance collapse - stopping training")
     raise ValueError("Variance collapse detected")

 3. Metric logging for monitoring:
 logger.info(f"Epoch {epoch}: pred_std={pred_std:.6f}, pos_pct={pos_pct:.1%}")

 Output Format

 === ANALYSIS COMPLETE ===

 Files Analyzed:
 - [list of files read]

 Issues Found:
 1. [Issue] in [file:line] - [severity]
    Root cause: [explanation]

 Fixes Applied:
 1. [File]: [change description]
    Before: [old code]
    After: [new code]

 Verification:
 - Command: [what was run]
 - Result: [SUCCESS/FAILED]
 - Metrics: [key values]

 Safeguards Added:
 - [list of permanent protections]

 ---

 ## Part 2: CLAUDE.md Update

 Add to CLAUDE.md under "## Custom Agents":

 ```markdown
 ---

 ## Custom Agents (Use PROACTIVELY)

 Three specialized agents are available for ML model development:

 ### 1. Deep Researcher (`deep-researcher`)
 **When**: Encountering model failures, variance collapse, overfitting, or any technical problem
 **How**: Searches web, analyzes papers, provides multi-perspective solutions
 **Invoke**: "Use deep-researcher to find solutions for [problem]"

 ### 2. Command Executor (`command-executor`)
 **When**: Running training scripts, analyzing logs, extracting metrics
 **How**: Efficiently executes commands and parses results without context bloat
 **Invoke**: "Use command-executor to train [model] and report metrics"

 ### 3. Code Analyzer (`code-analyzer`)
 **When**: After any training failure OR before marking any fix as "complete"
 **How**: Analyzes code, implements fixes, VERIFIES they work
 **Invoke**: "Use code-analyzer to fix [issue] in [file]"

 **CRITICAL RULE**: Never mark a model fix as "complete" without:
 1. Running the fix through code-analyzer
 2. Verifying metrics improved (pred_std > 0.005, balanced predictions)
 3. Running a backtest that shows improvement over previous results

 ---
 Part 3: Model Fix Strategy

 Phase 1: Diagnose Current State (Day 1)

 Use command-executor to run diagnostic commands:

 # Check LSTM predictions
 python -c "
 import pickle
 import numpy as np
 with open('saved_models/AAPL/regressor/metadata.pkl', 'rb') as f:
     meta = pickle.load(f)
 print('LSTM metadata:', meta)
 "

 # Check GBM predictions
 python -c "
 import joblib
 import numpy as np
 model = joblib.load('saved_models/AAPL/gbm/lgb_reg.joblib')
 print('GBM n_estimators:', model.n_estimators_)
 "

 Phase 2: Fix Variance Collapse (LSTM/xLSTM)

 Use deep-researcher to find latest solutions:
 - Search: "LSTM predicting constant value tensorflow 2024"
 - Search: "prevent variance collapse deep learning regression"
 - Search: "anti-collapse loss function financial prediction"

 Use code-analyzer to implement fixes:

 File: models/lstm_transformer_paper.py
 # Line 218-220: ALREADY INCREASED - but still collapsing
 # New approach: Add explicit variance monitoring in forward pass

 def call(self, inputs, training=False):
     # ... existing code ...
     output = self.output_dense(x)

     # Anti-collapse: Add noise during training if variance too low
     if training:
         output_std = ops.std(output)
         # If variance collapses, add small noise to prevent gradient death
         noise_scale = ops.maximum(0.01 - output_std, 0.0) * 0.1
         output = output + ops.random.normal(ops.shape(output)) * noise_scale

     return output

 File: training/train_1d_regressor_final.py
 # Add variance monitoring callback
 class VarianceMonitorCallback(keras.callbacks.Callback):
     def on_epoch_end(self, epoch, logs=None):
         preds = self.model.predict(self.validation_data[0], verbose=0)
         pred_std = np.std(preds)
         if pred_std < 0.005:
             print(f"WARNING: Variance collapse at epoch {epoch}: std={pred_std:.6f}")
             # Reduce learning rate to try to recover
             current_lr = self.model.optimizer.learning_rate
             self.model.optimizer.learning_rate = current_lr * 0.5

 Phase 3: Fix GBM Prediction Bias

 Use deep-researcher:
 - Search: "XGBoost regression prediction always positive bias"
 - Search: "LightGBM calibration for balanced predictions"

 Use code-analyzer to implement:

 File: training/train_gbm_baseline.py
 # After line 312, add output calibration:
 def calibrate_predictions(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
     """Post-hoc calibration to center predictions."""
     pred_mean = np.mean(y_pred)
     true_mean = np.mean(y_true)

     # Shift predictions to match true distribution center
     calibrated = y_pred - pred_mean + true_mean

     # Scale to match true variance
     pred_std = np.std(y_pred)
     true_std = np.std(y_true)
     if pred_std > 0:
         calibrated = (calibrated - true_mean) * (true_std / pred_std) + true_mean

     return calibrated

 Phase 4: Fix Stacking Meta-Learner

 The stacking doesn't train because base models fail validation. Fix sequence:

 1. Fix LSTM variance collapse (Phase 2)
 2. Fix GBM bias (Phase 3)
 3. Re-train base models
 4. Run stacking training

 File: pipeline/production_pipeline.py
 # Line 1066-1070: Already allows 1+ models
 # Add: Skip variance-collapsed models from stacking

 def _filter_valid_models(self, base_models):
     """Remove models with variance collapse or severe bias."""
     valid = {}
     for name, model in base_models.items():
         preds = model.predict(self.X_val)
         pred_std = np.std(preds)
         pos_pct = (preds > 0).mean()

         if pred_std < 0.005:
             logger.warning(f"Excluding {name}: variance collapse (std={pred_std:.6f})")
             continue
         if pos_pct > 0.85 or pos_pct < 0.15:
             logger.warning(f"Excluding {name}: severe bias ({pos_pct:.1%} positive)")
             continue

         valid[name] = model
     return valid

 Phase 5: Backtest Validation

 Use command-executor to run comprehensive backtest:

 # Test each fusion mode
 for mode in gbm_only weighted stacking; do
     python inference_and_backtest.py \
         --symbol AAPL \
         --start_date 2020-01-01 \
         --fusion_mode $mode \
         2>&1 | grep -E "Strategy Return|Buy.*Hold|Sharpe"
 done

 Success Criteria (Per Symbol: AAPL, SPY, TSLA):
 - Strategy Return > Buy & Hold Return by 50%+
 - Sharpe Ratio > 1.0
 - Prediction std > 0.01
 - Prediction balance: 40-60% positive
 - All 3 symbols must pass for production deployment

 ---
 Part 4: Implementation Order

 1. Create agent files (5 min)
   - .claude/agents/deep-researcher.md
   - .claude/agents/command-executor.md
   - .claude/agents/code-analyzer.md
 2. Update CLAUDE.md (2 min)
   - Add custom agents section
 3. Fix LSTM variance collapse (30 min)
   - Use deep-researcher for solutions
   - Use code-analyzer to implement
   - Use command-executor to verify
 4. Fix GBM prediction bias (20 min)
   - Add output calibration
   - Verify balanced predictions
 5. Re-train all models (60 min)
   - LSTM: 30 epochs
   - GBM: default settings
   - xLSTM: 30 epochs
 6. Train stacking meta-learner (15 min)
   - Only after base models pass validation
 7. Run comprehensive backtest (10 min)
   - Compare all fusion modes
   - Verify beats buy & hold

 ---
 Files to Modify

 | File                                 | Change                  | Priority |
 |--------------------------------------|-------------------------|----------|
 | .claude/agents/deep-researcher.md    | CREATE                  | P0       |
 | .claude/agents/command-executor.md   | CREATE                  | P0       |
 | .claude/agents/code-analyzer.md      | CREATE                  | P0       |
 | CLAUDE.md                            | ADD agents section      | P0       |
 | models/lstm_transformer_paper.py     | Add anti-collapse noise | P1       |
 | training/train_1d_regressor_final.py | Add variance callback   | P1       |
 | training/train_gbm_baseline.py       | Add output calibration  | P1       |
 | pipeline/production_pipeline.py      | Filter invalid models   | P2       |

 ---
 Success Metrics (AGGRESSIVE TARGETS)

 | Metric          | Current | Target |
 |-----------------|---------|--------|
 | LSTM pred_std   | 0.0015  | > 0.01 |
 | GBM positive %  | 89%     | 45-55% |
 | xLSTM WFE       | 43.2%   | > 60%  |
 | Backtest vs B&H | -45%    | > +50% |
 | Sharpe Ratio    | -0.56   | > 1.0  |

 Test Symbols (Diverse Coverage)

 | Symbol | Type            | Why                                |
 |--------|-----------------|------------------------------------|
 | AAPL   | Large-cap Tech  | Primary development target         |
 | SPY    | Index ETF       | Market benchmark, lower volatility |
 | TSLA   | High Volatility | Stress test model robustness       |

 Auto-Stop Safeguards (MANDATORY)

 The following safeguards will be added to ALL training scripts:

 # Add to every training loop
 class AutoStopOnCollapse(keras.callbacks.Callback):
     def on_epoch_end(self, epoch, logs=None):
         if epoch >= 3:  # Give model 3 epochs to stabilize
             preds = self.model.predict(self.validation_data[0][:100], verbose=0)
             pred_std = np.std(preds)
             pos_pct = (preds > 0).mean()

             if pred_std < 0.005:
                 raise ValueError(f"VARIANCE COLLAPSE at epoch {epoch}: std={pred_std:.6f}")
             if pos_pct > 0.85 or pos_pct < 0.15:
                 raise ValueError(f"PREDICTION BIAS at epoch {epoch}: {pos_pct:.1%} positive")

 This prevents wasting time on training runs that have already failed.