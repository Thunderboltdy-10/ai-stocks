# System Evolution Timeline

## Overview
This document tracks the major architectural changes, model additions, and deprecations in the AI-Stocks trading system.

---

## Phase 1: Foundation (Early Development)
### Regressor-Only Architecture
**Status**: Working, Well-Understood

**Components**:
- Single LSTM-based regressor for return prediction
- Simple data pipeline with technical features
- Basic backtesting framework

**Characteristics**:
- ~89 technical features
- 1-day prediction horizon
- Direct regression approach

---

## Phase 2: GBM Integration
### Regressor + GBM Ensemble
**Status**: Last Known Good State

**Key Addition**: Gradient Boosting Machine (GBM) for ensemble predictions

**Why This Worked**:
- GBM provided complementary non-linear patterns
- Simple ensemble averaging
- Both models independently validated
- Clear separation of concerns

**Performance**: Strong Sharpe ratios, consistent directional accuracy

---

## Phase 3: Binary Classifiers Addition
### Tri-Model System
**Status**: Increased Complexity

**Components Added**:
- Buy classifier (predicting buy signals)
- Sell classifier (predicting sell signals)
- More sophisticated fusion logic

**Rationale**:
- Direct signal prediction vs indirect (regression)
- Separate buy/sell decision paths

**Challenges Introduced**:
- Coordination between 3 model types
- Calibration requirements
- More complex fusion modes (conservative, balanced, risk_aware)

---

## Phase 4: Quantile Regressor
### Uncertainty Quantification
**Status**: Experimental

**Addition**: Quantile regression for confidence intervals

**Purpose**:
- Estimate prediction uncertainty
- Risk-aware position sizing
- Better understand model confidence

**Integration Status**: Available but usage unclear

---

## Phase 5: Advanced Models Experimentation
### TFT and PatchTST
**Status**: Problematic/Deprecated

**Models Attempted**:
1. **Temporal Fusion Transformer (TFT)**
   - Purpose: Capture complex temporal patterns
   - Issues: Training instability, collapse, complexity
   - Current Status: Deprecated (tft_loader.py exists but flagged)

2. **PatchTST**
   - Purpose: Patch-based time series prediction
   - Status: Implementation exists, usage unclear

**Why These Failed**:
- High complexity without clear performance gain
- Difficult to debug when things go wrong
- Variance collapse issues
- Overengineering for problem complexity

---

## Phase 6: Feature Expansion
### Sentiment Integration
**Status**: In Progress

**Changes**:
- Added news sentiment features (29 features)
- Expanded to 147 total features (118 technical + 29 sentiment)
- Backward compatibility maintained (89 feature fallback)

**Files Affected**:
- `data/feature_engineer.py`
- All training scripts updated for 147 features
- Validation scripts created (verify_validation_updates.py)

---

## Current State Analysis

### Active Components (Based on F1 Audit)
- **192 training files** identified
- **4 inference files**
- **206 model artifacts** (136 orphaned - HIGH PRIORITY)
- **497 test files**

### Key Architecture Files
1. **God Files** (Most Critical Dependencies):
   - `data/feature_engineer.py` (27.4% centrality)
   - `data/data_fetcher.py` (18.8% centrality)
   - `data/cache_manager.py` (17.9% centrality)

2. **Main Entry Points**:
   - `app.py` - FastAPI service
   - `inference_and_backtest.py` - Main inference pipeline
   - `orchestrate_training.py` - Training orchestrator (moved to scripts/)

### Organizational Improvements (Dec 2025)
- **F1 Audit Completed**: Comprehensive codebase analysis
- **13 root files relocated**: Better organization
- **Dependency analysis**: 1 circular dependency, 75 orphaned files identified

---

## Deprecation History

### Confirmed Deprecated
1. **TFT Models**
   - Reason: Training instability, variance collapse
   - Orphaned: No active training/inference
   - Recommendation: Archive

2. **Old Model Versions**
   - Pattern: `*_old_v1.pkl`, `*_v2.pkl`
   - Age: >6 months
   - Total: 136 orphaned models consuming 163.6 MB

### Under Review
1. **Binary Classifiers**
   - Status: Implementation exists
   - Usage: Training scripts present
   - Question: Performance gain vs complexity?

2. **Quantile Regressor**
   - Status: Trained but inference usage unclear
   - Keep if: Adds value to risk-aware trading

---

## Complexity Inflection Points

### When Things "Went Sideways"
Based on file analysis and audit findings:

1. **Multi-Model Coordination** (Phase 3-4)
   - 3+ models requiring fusion logic
   - Calibration complexity
   - Debugging difficulty increased

2. **Advanced Architecture Attempts** (Phase 5)
   - TFT introduction
   - PatchTST experimentation
   - Training instability

3. **Feature Explosion** (Phase 6)
   - 89 → 147 features
   - Backward compatibility burden
   - Validation script proliferation

---

## Lessons Learned

### What Worked
✅ Simple regressor approach
✅ GBM ensemble (interpretable, stable)
✅ Modular data pipeline
✅ Comprehensive caching system

### What Didn't Work
❌ Complex transformer models without clear benefit
❌ Too many model types simultaneously
❌ Experimental code left in production paths
❌ Inadequate deprecation discipline

### Current Priorities
1. Clean up 136 orphaned models (F3 identified)
2. Validate which models truly add value (F4 needed)
3. Simplify fusion logic or justify complexity
4. Better separation: experimental vs production code

---

## Related Documents
- `last_known_good.md` - Details of Regressor + GBM state
- `change_log_analysis.md` - Detailed commit analysis
- `audit_tools/audit_results/` - F1 comprehensive audit
- `unused_files.txt`, `orphaned_models.txt` - F3 cleanup targets
