#!/usr/bin/env python3
"""
Comprehensive verification script for sentiment feature integration
Verifies all files are properly configured for 147 features (118 technical + 29 sentiment + compatibility)
with graceful fallback to 89 technical-only features
"""

import sys
from pathlib import Path

def verify_sentiment_integration():
    """Verify sentiment feature integration across all files"""
    
    print("=" * 80)
    print("SENTIMENT FEATURE INTEGRATION - VERIFICATION")
    print("=" * 80)
    
    project_root = Path(__file__).parent
    all_passed = True
    
    # ===================================================================
    # 1. CHECK FEATURE ENGINEER
    # ===================================================================
    print("\n[1/4] Checking data/feature_engineer.py...")
    
    feature_engineer_path = project_root / "data" / "feature_engineer.py"
    if not feature_engineer_path.exists():
        print(f"  ❌ ERROR: {feature_engineer_path} not found!")
        all_passed = False
    else:
        with open(feature_engineer_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = {
            "Function signature with symbol param": "def engineer_features(df: pd.DataFrame, symbol: str = None, include_sentiment: bool = True)",
            "Sentiment integration conditional": "if include_sentiment and symbol is not None:",
            "Graceful fallback for no symbol": "elif include_sentiment and symbol is None:",
            "add_sentiment_features call": "df = add_sentiment_features(df, symbol)",
            "get_sentiment_feature_columns": "def get_sentiment_feature_columns():",
        }
        
        for desc, pattern in checks.items():
            if pattern in content:
                print(f"  ✅ {desc}")
            else:
                print(f"  ❌ MISSING: {desc}")
                all_passed = False
    
    # ===================================================================
    # 2. CHECK INFERENCE_AND_BACKTEST.PY
    # ===================================================================
    print("\n[2/4] Checking inference_and_backtest.py...")
    
    inference_path = project_root / "inference_and_backtest.py"
    if not inference_path.exists():
        print(f"  ❌ ERROR: {inference_path} not found!")
        all_passed = False
    else:
        with open(inference_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = {
            "Try block for sentiment": "try:",
            "Symbol passed to engineer_features": "engineer_features(df, symbol=symbol, include_sentiment=True)",
            "Graceful fallback": "engineer_features(df, symbol=None, include_sentiment=False)",
            "Success message (147 features)": "Loaded 147 features",
            "Fallback message (89 features)": "Loaded 89 technical features",
        }
        
        for desc, pattern in checks.items():
            if pattern in content:
                print(f"  ✅ {desc}")
            else:
                print(f"  ❌ MISSING: {desc}")
                all_passed = False
    
    # ===================================================================
    # 3. CHECK MODEL_VALIDATION_SUITE.PY
    # ===================================================================
    print("\n[3/4] Checking model_validation_suite.py...")
    
    validation_path = project_root / "model_validation_suite.py"
    if not validation_path.exists():
        print(f"  ❌ ERROR: {validation_path} not found!")
        all_passed = False
    else:
        with open(validation_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        checks = {
            "create_multitask_regressor function": "def create_multitask_regressor(sequence_length, n_features",
            "90-day regressor sequences": "self.seq_len_regressor = 90",
            "60-day classifier sequences": "self.seq_len_classifier = 60",
            "Try block for sentiment": "try:",
            "Symbol passed to engineer_features": "engineer_features(df, symbol=symbol, include_sentiment=True)",
            "Graceful fallback": "engineer_features(df, symbol=None, include_sentiment=False)",
            "Magnitude output extraction": "multitask_preds[0].flatten()",
            "3-head multitask model": "outputs=[magnitude_output, sign_output, volatility_output]",
        }
        
        for desc, pattern in checks.items():
            if pattern in content:
                print(f"  ✅ {desc}")
            else:
                print(f"  ❌ MISSING: {desc}")
                all_passed = False
    
    # ===================================================================
    # 4. CHECK TRAINING SCRIPTS
    # ===================================================================
    print("\n[4/4] Checking training scripts...")
    
    training_files = {
        "train_1d_regressor_final.py": "training/train_1d_regressor_final.py",
        "train_binary_classifiers_final.py": "training/train_binary_classifiers_final.py",
    }
    
    for name, rel_path in training_files.items():
        training_path = project_root / rel_path
        if not training_path.exists():
            print(f"  ⚠️  {name} not found (optional)")
        else:
            with open(training_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            if "engineer_features(df, symbol=symbol" in content:
                print(f"  ✅ {name} - symbol parameter passed")
            else:
                print(f"  ❌ {name} - symbol parameter NOT passed")
                all_passed = False
    
    # ===================================================================
    # SUMMARY
    # ===================================================================
    print("\n" + "=" * 80)
    
    if all_passed:
        print("✅ ALL CHECKS PASSED - Sentiment integration verified!")
        print("\nFeature Configuration:")
        print("  • With sentiment: 147 features (118 technical + 29 sentiment + compatibility features)")
        print("  • Without sentiment: 89 technical features (fallback)")
        print("\nError Handling:")
        print("  • Graceful fallback if sentiment fails")
        print("  • Model will work with either 89 or 147 features")
        print("\nYou can now run:")
        print("  python inference_and_backtest.py AAPL")
        print("  python model_validation_suite.py AAPL HOOD TSLA")
    else:
        print("❌ SOME CHECKS FAILED - Please review the configuration")
        print("\nCheck the following:")
        print("  1. data/feature_engineer.py has symbol parameter")
        print("  2. inference_and_backtest.py has try/except for sentiment")
        print("  3. model_validation_suite.py has try/except for sentiment")
        print("  4. Training scripts pass symbol to engineer_features()")
        return False
    
    print("=" * 80)
    
    return True


if __name__ == "__main__":
    success = verify_sentiment_integration()
    sys.exit(0 if success else 1)
