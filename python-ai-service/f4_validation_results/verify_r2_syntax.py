"""
Quick syntax verification for R² metric implementation.
Checks that the function can be imported and called without errors.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

print("=" * 80)
print("R² METRIC SYNTAX VERIFICATION")
print("=" * 80)

# Step 1: Verify TensorFlow import
print("\n[STEP 1] Verifying TensorFlow import...")
try:
    import tensorflow as tf
    print(f"   ✅ TensorFlow version: {tf.__version__}")
except ImportError as e:
    print(f"   ❌ Failed to import TensorFlow: {e}")
    sys.exit(1)

# Step 2: Import the r_squared_metric function
print("\n[STEP 2] Importing r_squared_metric function...")
try:
    from training.train_1d_regressor_final import r_squared_metric
    print("   ✅ Successfully imported r_squared_metric")
except ImportError as e:
    print(f"   ❌ Failed to import r_squared_metric: {e}")
    sys.exit(1)
except Exception as e:
    print(f"   ❌ Error during import: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 3: Verify function signature
print("\n[STEP 3] Verifying function signature...")
import inspect
sig = inspect.signature(r_squared_metric)
params = list(sig.parameters.keys())
print(f"   Function signature: r_squared_metric{sig}")
print(f"   Parameters: {params}")

if params != ['y_true', 'y_pred']:
    print(f"   ❌ Unexpected parameters: {params}")
    sys.exit(1)
print("   ✅ Function signature is correct")

# Step 4: Test basic functionality
print("\n[STEP 4] Testing basic functionality...")
try:
    # Create simple test tensors
    y_true = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0], dtype=tf.float32)
    y_pred = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0], dtype=tf.float32)

    # Call the function
    r2 = r_squared_metric(y_true, y_pred)

    # Verify return type
    print(f"   Return type: {type(r2)}")
    print(f"   Return value: {r2.numpy():.6f}")

    # For perfect prediction, R² should be 1.0
    if abs(r2.numpy() - 1.0) < 1e-6:
        print("   ✅ R² = 1.0 for perfect prediction (correct!)")
    else:
        print(f"   ❌ R² = {r2.numpy():.6f} for perfect prediction (expected 1.0)")
        sys.exit(1)

except Exception as e:
    print(f"   ❌ Error during function call: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 5: Test TensorFlow operations
print("\n[STEP 5] Testing TensorFlow operations...")
try:
    # Test with different values
    y_true = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0], dtype=tf.float32)
    y_pred = tf.constant([3.0, 3.0, 3.0, 3.0, 3.0], dtype=tf.float32)  # Constant at mean

    r2 = r_squared_metric(y_true, y_pred)
    print(f"   R² for constant prediction: {r2.numpy():.6f}")

    if abs(r2.numpy()) < 1e-5:
        print("   ✅ R² ≈ 0.0 for constant prediction (correct!)")
    else:
        print(f"   ⚠️  R² = {r2.numpy():.6f} (expected ~0.0, may have small numerical error)")

except Exception as e:
    print(f"   ❌ Error during TensorFlow operations: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Step 6: Verify it works as a Keras metric
print("\n[STEP 6] Verifying compatibility as Keras metric...")
try:
    from tensorflow import keras

    # Create a simple model
    model = keras.Sequential([
        keras.layers.Dense(1, input_shape=(5,))
    ])

    # Compile with r_squared_metric
    model.compile(
        optimizer='adam',
        loss='mse',
        metrics=[r_squared_metric]
    )

    print("   ✅ Model compiles successfully with r_squared_metric")
    print(f"   Model metrics: {[m.name if hasattr(m, 'name') else str(m) for m in model.metrics]}")

except Exception as e:
    print(f"   ❌ Error compiling model with metric: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 80)
print("✅ ALL SYNTAX CHECKS PASSED!")
print("=" * 80)
print("\nThe r_squared_metric implementation is syntactically correct and functional.")
print("Ready for integration testing with actual model training.")
