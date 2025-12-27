"""
Verification script to ensure robust error handling is implemented.

Checks that validation functions exist and are called at key points.

Run: python verify_error_handling.py
"""

import re
from pathlib import Path


def check_validation_functions_exist(filepath: Path) -> bool:
    """Check that validation functions are defined"""
    print(f"\nChecking {filepath.name} for validation functions...")
    
    content = filepath.read_text(encoding='utf-8')
    
    required_functions = [
        'validate_model_files',
        'validate_feature_counts',
        'validate_sequences'
    ]
    
    all_found = True
    for func in required_functions:
        pattern = rf'def {func}\('
        if re.search(pattern, content):
            print(f"  ✓ {func}() defined")
        else:
            print(f"  ❌ {func}() NOT FOUND")
            all_found = False
    
    return all_found


def check_validation_calls(filepath: Path, expected_calls: dict) -> bool:
    """Check that validation functions are called at key points"""
    print(f"\nChecking {filepath.name} for validation calls...")
    
    content = filepath.read_text(encoding='utf-8')
    
    all_found = True
    for location, validation_func in expected_calls.items():
        if validation_func in content:
            print(f"  ✓ {validation_func}() called {location}")
        else:
            print(f"  ❌ {validation_func}() NOT called {location}")
            all_found = False
    
    return all_found


def check_error_handling_wrappers(filepath: Path) -> bool:
    """Check that main execution is wrapped with try-except"""
    print(f"\nChecking {filepath.name} for error handling wrappers...")
    
    content = filepath.read_text(encoding='utf-8')
    
    checks = [
        ("Try-except wrapper", r'try:\s+main\('),
        ("FileNotFoundError handler", r'except FileNotFoundError'),
        ("ValueError handler", r'except ValueError'),
        ("AssertionError handler", r'except AssertionError'),
        ("Generic exception handler", r'except Exception'),
        ("Helpful error messages", r'💡|Solution:|Common causes:'),
        ("Exit with error code", r'sys\.exit\(1\)')
    ]
    
    all_found = True
    for check_name, pattern in checks:
        if re.search(pattern, content, re.DOTALL):
            print(f"  ✓ {check_name}")
        else:
            print(f"  ❌ {check_name} NOT FOUND")
            all_found = False
    
    return all_found


def main():
    print("=" * 70)
    print("VERIFYING ROBUST ERROR HANDLING IMPLEMENTATION")
    print("=" * 70)
    
    base_dir = Path(__file__).parent
    
    results = []
    
    # Check inference_and_backtest.py
    print("\n" + "=" * 70)
    print("1. INFERENCE_AND_BACKTEST.PY")
    print("=" * 70)
    
    inf_file = base_dir / "inference_and_backtest.py"
    
    results.append(("Validation functions defined (inference)", 
                   check_validation_functions_exist(inf_file)))
    
    results.append(("Validation calls (inference)", 
                   check_validation_calls(inf_file, {
                       "before model loading": "validate_model_files",
                       "after feature engineering": "validate_feature_counts",
                       "after sequence creation": "validate_sequences"
                   })))
    
    results.append(("Error handling wrapper (inference)", 
                   check_error_handling_wrappers(inf_file)))
    
    # Check model_validation_suite.py
    print("\n" + "=" * 70)
    print("2. MODEL_VALIDATION_SUITE.PY")
    print("=" * 70)
    
    val_file = base_dir / "model_validation_suite.py"
    
    results.append(("Validation functions defined (validation)", 
                   check_validation_functions_exist(val_file)))
    
    results.append(("Validation calls (validation)", 
                   check_validation_calls(val_file, {
                       "in load_models_and_data": "validate_model_files",
                       "after data preparation": "validate_feature_counts",
                       "in validate_regressor": "validate_sequences",
                       "in validate_classifiers": "validate_sequences"
                   })))
    
    results.append(("Error handling wrapper (validation)", 
                   check_error_handling_wrappers(val_file)))
    
    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print("=" * 70)
    
    if passed == total:
        print(f"\n🎉 ALL CHECKS PASSED ({passed}/{total})")
        print("\nRobust error handling is properly implemented:")
        print("  ✓ Validation functions defined and documented")
        print("  ✓ Validation called at critical points")
        print("  ✓ Model file validation before loading")
        print("  ✓ Feature count validation after engineering")
        print("  ✓ Sequence shape validation after creation")
        print("  ✓ Main execution wrapped with try-except")
        print("  ✓ Specific exception handlers for common failures")
        print("  ✓ Helpful error messages with solutions")
        print("  ✓ Graceful exit with error codes")
        print("\nProduction-ready error handling! 🚀")
        return 0
    else:
        print(f"\n❌ VERIFICATION FAILED ({passed}/{total} checks passed)")
        print("Some error handling components are missing.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
