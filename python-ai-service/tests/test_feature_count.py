"""
Quick test to verify feature_engineer.py syntax and feature count
"""
import sys
import os
# Use dynamic path based on script location
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

print("Testing feature_engineer.py...")

try:
    from data.feature_engineer import get_feature_columns, EXPECTED_FEATURE_COUNT
    print(f"✓ Successfully imported from feature_engineer")
    
    # Test feature count
    features = get_feature_columns(include_sentiment=True)
    print(f"✓ Feature count with sentiment: {len(features)}")
    print(f"✓ Expected feature count: {EXPECTED_FEATURE_COUNT}")
    
    if len(features) == EXPECTED_FEATURE_COUNT:
        print(f"✓ SUCCESS: Feature count matches expected ({EXPECTED_FEATURE_COUNT})")
    else:
        print(f"✗ MISMATCH: Got {len(features)}, expected {EXPECTED_FEATURE_COUNT}")
        
except SyntaxError as e:
    print(f"✗ SYNTAX ERROR: {e}")
    import traceback
    traceback.print_exc()
except Exception as e:
    print(f"✗ ERROR: {e}")
    import traceback
    traceback.print_exc()
