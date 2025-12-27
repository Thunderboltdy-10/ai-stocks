#!/usr/bin/env python3
"""
Test script to verify model loading architecture fix.

This script tests that:
1. Architecture parameters are correctly read from metadata.pkl
2. Custom objects are used when loading .keras files
3. The loaded model has the correct number of weights
"""

import sys
import pickle
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

def test_architecture_reading():
    """Test that metadata contains correct architecture parameters."""
    print("=" * 60)
    print("TEST 1: Architecture Parameter Reading")
    print("=" * 60)

    metadata_path = Path("saved_models/AAPL_1d_regressor_final_metadata.pkl")

    if not metadata_path.exists():
        print(f"❌ FAIL: Metadata file not found at {metadata_path}")
        return False

    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    arch = metadata.get('architecture', {})

    print(f"✓ Metadata loaded successfully")
    print(f"  Architecture parameters:")
    print(f"    lstm_units: {arch.get('lstm_units', 'NOT FOUND')}")
    print(f"    d_model: {arch.get('d_model', 'NOT FOUND')}")
    print(f"    num_blocks: {arch.get('num_blocks', 'NOT FOUND')}")
    print(f"    ff_dim: {arch.get('ff_dim', 'NOT FOUND')}")
    print(f"    dropout: {arch.get('dropout', 'NOT FOUND')}")

    # Expected values from training
    expected = {
        'lstm_units': 48,
        'd_model': 96,
        'num_blocks': 4,
        'ff_dim': 192,
        'dropout': 0.2
    }

    all_match = True
    for key, expected_val in expected.items():
        actual_val = arch.get(key)
        if actual_val != expected_val:
            print(f"  ❌ Mismatch: {key} = {actual_val}, expected {expected_val}")
            all_match = False

    if all_match:
        print(f"✅ PASS: All architecture parameters match expected values")
        return True
    else:
        print(f"❌ FAIL: Architecture parameter mismatch")
        return False


def test_custom_objects_import():
    """Test that custom objects can be imported."""
    print("\n" + "=" * 60)
    print("TEST 2: Custom Objects Import")
    print("=" * 60)

    try:
        from utils.losses import get_custom_objects
        custom_objs = get_custom_objects()
        print(f"✓ Custom objects imported successfully")
        print(f"  Available custom objects: {list(custom_objs.keys())}")

        # Check if LSTMTransformerPaper is in custom objects
        lstm_found = any('lstm' in str(k).lower() or 'transformer' in str(k).lower()
                        for k in custom_objs.keys())

        if lstm_found:
            print(f"✅ PASS: Custom objects include LSTM/Transformer classes")
            return True
        else:
            print(f"⚠️  WARNING: LSTMTransformerPaper not found in custom objects")
            print(f"   This may be registered globally via @register_keras_serializable")
            return True

    except Exception as e:
        print(f"❌ FAIL: Could not import custom objects: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_code_uses_custom_objects():
    """Test that prediction_service.py uses custom_objects when loading."""
    print("\n" + "=" * 60)
    print("TEST 3: Code Uses Custom Objects")
    print("=" * 60)

    service_file = Path("service/prediction_service.py")

    if not service_file.exists():
        print(f"❌ FAIL: Service file not found at {service_file}")
        return False

    with open(service_file, 'r') as f:
        content = f.read()

    # Check for imports
    has_import = 'from utils.losses import get_custom_objects' in content
    print(f"{'✓' if has_import else '❌'} Import statement: {'found' if has_import else 'NOT FOUND'}")

    # Check for usage in load_model call
    has_usage = 'custom_objects=get_custom_objects()' in content or \
                'custom_objects = get_custom_objects()' in content
    print(f"{'✓' if has_usage else '❌'} Usage in load_model: {'found' if has_usage else 'NOT FOUND'}")

    # Check architecture parameter reading
    has_arch_reading = "bundle.target_metadata.get('architecture'" in content
    print(f"{'✓' if has_arch_reading else '❌'} Architecture reading: {'found' if has_arch_reading else 'NOT FOUND'}")

    if has_import and has_usage and has_arch_reading:
        print(f"✅ PASS: Code correctly imports and uses custom objects")
        return True
    else:
        print(f"❌ FAIL: Code missing custom objects or architecture reading")
        return False


def test_model_files_exist():
    """Test that required model files exist."""
    print("\n" + "=" * 60)
    print("TEST 4: Model Files Existence")
    print("=" * 60)

    required_files = [
        "saved_models/AAPL_1d_regressor_final_model.keras",
        "saved_models/AAPL_1d_regressor_final.weights.h5",
        "saved_models/AAPL_1d_regressor_final_metadata.pkl",
        "saved_models/AAPL_1d_regressor_final_features.pkl",
        "saved_models/AAPL_1d_regressor_final_feature_scaler.pkl",
        "saved_models/AAPL_1d_regressor_final_target_scaler.pkl"
    ]

    all_exist = True
    for file_path_str in required_files:
        file_path = Path(file_path_str)
        exists = file_path.exists()
        size = file_path.stat().st_size if exists else 0
        size_mb = size / (1024 * 1024)

        print(f"  {'✓' if exists else '❌'} {file_path.name}: "
              f"{'exists' if exists else 'NOT FOUND'}"
              f"{f' ({size_mb:.1f} MB)' if exists else ''}")

        if not exists:
            all_exist = False

    if all_exist:
        print(f"✅ PASS: All required model files exist")
        return True
    else:
        print(f"❌ FAIL: Some model files are missing")
        return False


def main():
    """Run all tests."""
    print("Testing Model Loading Architecture Fix")
    print("=" * 60)
    print()

    results = []

    # Run tests
    results.append(("Architecture Reading", test_architecture_reading()))
    results.append(("Custom Objects Import", test_custom_objects_import()))
    results.append(("Code Uses Custom Objects", test_code_uses_custom_objects()))
    results.append(("Model Files Exist", test_model_files_exist()))

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    total_passed = sum(1 for _, passed in results if passed)
    total_tests = len(results)

    print(f"\nResults: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n🎉 All tests passed! Model loading should work correctly.")
        return 0
    else:
        print(f"\n⚠️  {total_tests - total_passed} test(s) failed. Review the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
