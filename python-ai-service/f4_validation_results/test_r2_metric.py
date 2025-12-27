"""
Test R² Metric Implementation
Validates TensorFlow R² calculation against sklearn's r2_score
"""

import numpy as np
import tensorflow as tf
from sklearn.metrics import r2_score

# Import the custom R² metric from training script
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from training.train_1d_regressor_final import r_squared_metric


def test_r2_perfect_prediction():
    """Test R² = 1.0 for perfect predictions."""
    print("\n[TEST 1] Perfect Prediction (R² should = 1.0)")

    # Perfect prediction: y_pred = y_true
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)

    # Sklearn R²
    sklearn_r2 = r2_score(y_true, y_pred)

    # TensorFlow R²
    tf_r2 = r_squared_metric(
        tf.constant(y_true, dtype=tf.float32),
        tf.constant(y_pred, dtype=tf.float32)
    ).numpy()

    print(f"   Sklearn R²:     {sklearn_r2:.6f}")
    print(f"   TensorFlow R²:  {tf_r2:.6f}")
    print(f"   Difference:     {abs(sklearn_r2 - tf_r2):.10f}")

    assert abs(sklearn_r2 - 1.0) < 1e-6, "Sklearn R² should be 1.0"
    assert abs(tf_r2 - 1.0) < 1e-6, "TensorFlow R² should be 1.0"
    assert abs(sklearn_r2 - tf_r2) < 1e-6, "R² values should match"

    print("   ✅ PASS")


def test_r2_constant_prediction():
    """Test R² = 0.0 for constant (mean) predictions."""
    print("\n[TEST 2] Constant Prediction (R² should = 0.0)")

    # Constant prediction at mean
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    y_pred = np.array([3.0, 3.0, 3.0, 3.0, 3.0], dtype=np.float32)  # Mean of y_true

    # Sklearn R²
    sklearn_r2 = r2_score(y_true, y_pred)

    # TensorFlow R²
    tf_r2 = r_squared_metric(
        tf.constant(y_true, dtype=tf.float32),
        tf.constant(y_pred, dtype=tf.float32)
    ).numpy()

    print(f"   Sklearn R²:     {sklearn_r2:.6f}")
    print(f"   TensorFlow R²:  {tf_r2:.6f}")
    print(f"   Difference:     {abs(sklearn_r2 - tf_r2):.10f}")

    assert abs(sklearn_r2 - 0.0) < 1e-6, "Sklearn R² should be 0.0"
    assert abs(tf_r2 - 0.0) < 1e-6, "TensorFlow R² should be 0.0"
    assert abs(sklearn_r2 - tf_r2) < 1e-6, "R² values should match"

    print("   ✅ PASS")


def test_r2_negative():
    """Test R² < 0 for predictions worse than baseline."""
    print("\n[TEST 3] Worse Than Baseline (R² should < 0)")

    # Bad predictions (worse than mean)
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    y_pred = np.array([5.0, 1.0, 5.0, 1.0, 5.0], dtype=np.float32)  # Terrible predictions

    # Sklearn R²
    sklearn_r2 = r2_score(y_true, y_pred)

    # TensorFlow R²
    tf_r2 = r_squared_metric(
        tf.constant(y_true, dtype=tf.float32),
        tf.constant(y_pred, dtype=tf.float32)
    ).numpy()

    print(f"   Sklearn R²:     {sklearn_r2:.6f}")
    print(f"   TensorFlow R²:  {tf_r2:.6f}")
    print(f"   Difference:     {abs(sklearn_r2 - tf_r2):.10f}")

    assert sklearn_r2 < 0, "Sklearn R² should be negative"
    assert tf_r2 < 0, "TensorFlow R² should be negative"
    assert abs(sklearn_r2 - tf_r2) < 1e-5, "R² values should match"

    print("   ✅ PASS")


def test_r2_partial_prediction():
    """Test R² for realistic partial predictions (0 < R² < 1)."""
    print("\n[TEST 4] Partial Prediction (0 < R² < 1)")

    # Realistic predictions with some error
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=np.float32)
    y_pred = np.array([1.2, 2.1, 2.8, 4.2, 4.9], dtype=np.float32)  # Close but not perfect

    # Sklearn R²
    sklearn_r2 = r2_score(y_true, y_pred)

    # TensorFlow R²
    tf_r2 = r_squared_metric(
        tf.constant(y_true, dtype=tf.float32),
        tf.constant(y_pred, dtype=tf.float32)
    ).numpy()

    print(f"   Sklearn R²:     {sklearn_r2:.6f}")
    print(f"   TensorFlow R²:  {tf_r2:.6f}")
    print(f"   Difference:     {abs(sklearn_r2 - tf_r2):.10f}")

    assert 0 < sklearn_r2 < 1, "Sklearn R² should be between 0 and 1"
    assert 0 < tf_r2 < 1, "TensorFlow R² should be between 0 and 1"
    assert abs(sklearn_r2 - tf_r2) < 1e-5, "R² values should match"

    print("   ✅ PASS")


def test_r2_financial_returns():
    """Test R² on realistic financial returns data."""
    print("\n[TEST 5] Financial Returns (Realistic Stock Data)")

    # Simulate realistic stock returns and predictions
    np.random.seed(42)
    n_samples = 1000

    # True returns: mean=0.0005 (0.05% daily), std=0.015 (1.5% volatility)
    y_true = np.random.normal(loc=0.0005, scale=0.015, size=n_samples).astype(np.float32)

    # Predictions: correlated with true but with noise (R² ~ 0.1-0.3 is realistic)
    noise = np.random.normal(loc=0, scale=0.012, size=n_samples).astype(np.float32)
    y_pred = (0.3 * y_true + 0.7 * noise).astype(np.float32)  # Weak correlation

    # Sklearn R²
    sklearn_r2 = r2_score(y_true, y_pred)

    # TensorFlow R²
    tf_r2 = r_squared_metric(
        tf.constant(y_true, dtype=tf.float32),
        tf.constant(y_pred, dtype=tf.float32)
    ).numpy()

    print(f"   Sklearn R²:     {sklearn_r2:.6f}")
    print(f"   TensorFlow R²:  {tf_r2:.6f}")
    print(f"   Difference:     {abs(sklearn_r2 - tf_r2):.10f}")
    print(f"   Data stats:")
    print(f"      - True returns mean: {np.mean(y_true):.6f}")
    print(f"      - True returns std:  {np.std(y_true):.6f}")
    print(f"      - Pred returns mean: {np.mean(y_pred):.6f}")
    print(f"      - Pred returns std:  {np.std(y_pred):.6f}")

    # For financial returns, R² is typically low (-0.5 to 0.3)
    assert -1.0 < sklearn_r2 < 0.5, f"Sklearn R² seems unrealistic: {sklearn_r2}"
    assert -1.0 < tf_r2 < 0.5, f"TensorFlow R² seems unrealistic: {tf_r2}"
    assert abs(sklearn_r2 - tf_r2) < 1e-5, "R² values should match"

    print("   ✅ PASS")


def test_r2_batch_processing():
    """Test R² metric with batched data (TensorFlow typical use case)."""
    print("\n[TEST 6] Batch Processing (TensorFlow Batches)")

    # Create batched data
    batch_size = 32
    n_batches = 10

    all_true = []
    all_pred = []
    batch_r2_values = []

    for i in range(n_batches):
        y_true = np.random.normal(loc=0.001, scale=0.02, size=batch_size).astype(np.float32)
        y_pred = y_true + np.random.normal(loc=0, scale=0.01, size=batch_size).astype(np.float32)

        all_true.append(y_true)
        all_pred.append(y_pred)

        # Compute R² for this batch
        tf_r2 = r_squared_metric(
            tf.constant(y_true, dtype=tf.float32),
            tf.constant(y_pred, dtype=tf.float32)
        ).numpy()
        batch_r2_values.append(tf_r2)

    # Compute overall R² on concatenated data
    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)

    sklearn_r2_overall = r2_score(y_true_all, y_pred_all)
    tf_r2_overall = r_squared_metric(
        tf.constant(y_true_all, dtype=tf.float32),
        tf.constant(y_pred_all, dtype=tf.float32)
    ).numpy()

    print(f"   Overall Sklearn R²:     {sklearn_r2_overall:.6f}")
    print(f"   Overall TensorFlow R²:  {tf_r2_overall:.6f}")
    print(f"   Mean batch R²:          {np.mean(batch_r2_values):.6f}")
    print(f"   Std batch R²:           {np.std(batch_r2_values):.6f}")
    print(f"   Difference:             {abs(sklearn_r2_overall - tf_r2_overall):.10f}")

    assert abs(sklearn_r2_overall - tf_r2_overall) < 1e-5, "Overall R² values should match"

    print("   ✅ PASS")


def main():
    """Run all R² metric validation tests."""
    print("=" * 80)
    print("R² METRIC VALIDATION TEST SUITE")
    print("=" * 80)
    print("\nValidating TensorFlow R² implementation against sklearn.metrics.r2_score")

    try:
        test_r2_perfect_prediction()
        test_r2_constant_prediction()
        test_r2_negative()
        test_r2_partial_prediction()
        test_r2_financial_returns()
        test_r2_batch_processing()

        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED!")
        print("=" * 80)
        print("\nR² metric implementation is correct and matches sklearn's r2_score.")
        print("The metric is ready for use in regressor training.")

        return True

    except AssertionError as e:
        print("\n" + "=" * 80)
        print("❌ TEST FAILED!")
        print("=" * 80)
        print(f"\nError: {e}")
        return False
    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ UNEXPECTED ERROR!")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
