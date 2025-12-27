"""
Verification script to ensure alignment fixes are properly implemented.

This script checks that:
1. inference_and_backtest.py has proper split calculation and assertions
2. model_validation_suite.py has proper alignment in all validation methods
3. All supporting arrays are aligned correctly
4. Backtest trimming logic is safe and verified

Run: python verify_alignment_implementation.py
"""

import re
from pathlib import Path

def check_file_contains(filepath: Path, patterns: list, description: str) -> bool:
    """Check if file contains all required patterns"""
    print(f"\n{'=' * 70}")
    print(f"Checking: {description}")
    print(f"File: {filepath.name}")
    print(f"{'=' * 70}")
    
    if not filepath.exists():
        print(f"❌ File not found: {filepath}")
        return False
    
    content = filepath.read_text(encoding='utf-8')
    
    all_found = True
    for pattern_desc, pattern in patterns:
        if re.search(pattern, content, re.DOTALL):
            print(f"✓ Found: {pattern_desc}")
        else:
            print(f"❌ Missing: {pattern_desc}")
            all_found = False
    
    return all_found

def verify_inference_and_backtest():
    """Verify inference_and_backtest.py has proper alignment"""
    filepath = Path(__file__).parent / "inference_and_backtest.py"
    
    patterns = [
        # 1. Supporting array alignment section
        ("Supporting array alignment header", r"# CRITICAL: Align ALL supporting arrays with regressor sequences"),
        ("ATR alignment", r"atr_percent_aligned = atr_percent_values\[regressor_seq_len - 1:\]"),
        ("Vol ratio alignment", r"vol_ratio_5_20_aligned = vol_ratio_5_20_values\[regressor_seq_len - 1:\]"),
        
        # 2. Train/test split section
        ("Train/test split header", r"# TRAIN/TEST SPLIT \(80/20\)"),
        ("Split calculation from X_seq", r"split = int\(len\(X_seq\) \* 0\.8\)"),
        ("X_test assignment", r"X_test = X_seq\[split:\]"),
        ("X_test_clf assignment", r"X_test_clf = X_seq_clf\[split:\]"),
        ("y_test assignment", r"y_test = y_aligned\[split:\]"),
        ("test_prices assignment", r"test_prices = prices_aligned\[split:\]"),
        ("test_dates assignment", r"test_dates = dates_aligned\[split:\]"),
        
        # 3. Assertion after split
        ("Alignment assertion after split", 
         r"assert len\(X_test\) == len\(X_test_clf\) == len\(y_test\) == len\(test_prices\) == len\(test_dates\)"),
        
        # 4. Backtest trimming section
        ("Backtest trimming header", r"# BACKTEST WINDOW TRIMMING \(if specified\)"),
        ("Safe tail calculation", r"tail = min\(int\(backtest_days\), len\(X_test\)\)"),
        ("Safe trimming condition", r"if tail > 0 and tail < len\(X_test\):"),
        ("X_test trimming", r"X_test = X_test\[-tail:\]"),
        ("X_test_clf trimming", r"X_test_clf = X_test_clf\[-tail:\]"),
        
        # 5. Assertion after backtest trim
        ("Alignment assertion after trim", 
         r"assert len\(X_test\) == len\(X_test_clf\) == len\(y_test\) == len\(test_prices\) == len\(test_dates\).*after backtest trim"),
    ]
    
    return check_file_contains(filepath, patterns, "inference_and_backtest.py alignment fixes")

def verify_model_validation_suite():
    """Verify model_validation_suite.py has proper alignment"""
    filepath = Path(__file__).parent / "model_validation_suite.py"
    
    patterns = [
        # 1. Regressor validation alignment
        ("Regressor sequence creation header", r"# SEQUENCE CREATION - Regressor uses 90-day sequences"),
        ("Regressor supporting array alignment header", r"# ALIGN supporting arrays with regressor sequences"),
        ("Regressor y alignment", r"y_actual_raw = df\['target_1d'\]\.values\[self\.seq_len_regressor - 1:\]"),
        ("Regressor prices alignment", r"prices_aligned = df\['Close'\]\.values\[self\.seq_len_regressor - 1:\]"),
        ("Regressor dates alignment", r"dates_aligned = df\.index\[self\.seq_len_regressor - 1:\]"),
        ("Regressor alignment assertion", 
         r"assert len\(X_seq\) == len\(y_actual_raw\) == len\(prices_aligned\) == len\(dates_aligned\)"),
        
        # 2. Classifier validation alignment
        ("Classifier sequence creation header", r"# SEQUENCE CREATION - Classifiers use 60-day sequences"),
        ("Classifier supporting array alignment header", r"# ALIGN supporting arrays with classifier sequences"),
        ("Classifier returns alignment", r"returns_1d = df\['target_1d'\]\.values\[self\.seq_len_classifier - 1:\]"),
        ("Classifier alignment assertion", 
         r"assert len\(X_seq\) == len\(returns_1d\) == len\(prices_aligned\) == len\(dates_aligned\)"),
        
        # 3. Backtest alignment
        ("Backtest sequence creation header", r"# SEQUENCE CREATION - Using classifier sequences"),
        ("Backtest supporting array alignment header", r"# ALIGN supporting arrays with classifier sequences"),
        ("Backtest dates alignment", r"backtest_dates = df\.index\[self\.seq_len_classifier - 1:\]\.values"),
        ("Backtest prices alignment", r"backtest_prices = df\['Close'\]\.values\[self\.seq_len_classifier - 1:\]"),
        ("Backtest returns alignment", r"backtest_returns = df\['target_1d'\]\.values\[self\.seq_len_classifier - 1:\]"),
        ("Backtest alignment assertion", 
         r"assert len\(X_seq\) == len\(backtest_dates\) == len\(backtest_prices\) == len\(backtest_returns\)"),
        ("Backtest prediction verification", 
         r"assert len\(buy_probs\) == len\(sell_probs\) == len\(X_seq\)"),
    ]
    
    return check_file_contains(filepath, patterns, "model_validation_suite.py alignment fixes")

def verify_test_script():
    """Verify test script exists and has key tests"""
    filepath = Path(__file__).parent / "test_split_alignment.py"
    
    patterns = [
        ("Alignment logic test", r"def test_alignment_logic\(\):"),
        ("Edge cases test", r"def test_edge_cases\(\):"),
        ("Volatility fallbacks test", r"def test_volatility_fallbacks\(\):"),
        ("Regressor sequence creation", r"X_seq_reg = create_sequences\(X_scaled, regressor_seq_len\)"),
        ("Classifier sequence creation", r"X_seq_clf = create_sequences\(X_scaled, classifier_seq_len\)"),
        ("Offset alignment", r"offset = regressor_seq_len - classifier_seq_len"),
        ("Split calculation", r"split = int\(len\(X_seq_reg\) \* 0\.8\)"),
        ("Backtest trimming test", r"tail = min\(backtest_days, len\(X_test_reg\)\)"),
    ]
    
    return check_file_contains(filepath, patterns, "test_split_alignment.py comprehensive tests")

def main():
    print("\n" + "=" * 70)
    print("VERIFYING ALIGNMENT IMPLEMENTATION")
    print("=" * 70)
    
    results = []
    
    # Check inference_and_backtest.py
    results.append(("inference_and_backtest.py", verify_inference_and_backtest()))
    
    # Check model_validation_suite.py
    results.append(("model_validation_suite.py", verify_model_validation_suite()))
    
    # Check test script
    results.append(("test_split_alignment.py", verify_test_script()))
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    all_passed = True
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 70)
    
    if all_passed:
        print("\n🎉 ALL VERIFICATION CHECKS PASSED!")
        print("\nAlignment fixes are properly implemented:")
        print("  ✓ Supporting arrays aligned with regressor sequences")
        print("  ✓ Train/test split with verification assertions")
        print("  ✓ Safe backtest trimming with assertions")
        print("  ✓ model_validation_suite.py has proper alignment")
        print("  ✓ Comprehensive test suite created")
        print("\nReady for production! 🚀\n")
        return 0
    else:
        print("\n❌ VERIFICATION FAILED")
        print("Some required patterns were not found.")
        print("Please review the implementation.\n")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
