#!/usr/bin/env python3
"""
Quick test to verify AntiCollapseDirectionalLoss produces valid outputs.
"""

import numpy as np
import tensorflow as tf
from models.lstm_transformer_paper import AntiCollapseDirectionalLoss, create_paper_model

print("="*70)
print("TESTING AntiCollapseDirectionalLoss - Single-Task Mode")
print("="*70)

# Create loss function with same params as training script
loss_fn = AntiCollapseDirectionalLoss(
    delta=1.0,
    direction_weight=1.0,
    variance_penalty_weight=2.0,
    min_variance_target=0.003,
    sign_diversity_weight=1.0
)

print("\n[TEST 1] Normal predictions")
y_true = tf.constant(np.random.randn(512, 1).astype(np.float32))
y_pred = tf.constant(np.random.randn(512, 1).astype(np.float32) * 0.1)
loss = loss_fn(y_true, y_pred)
print(f"  Loss: {float(loss):.6f}")
print(f"  Is NaN: {bool(tf.math.is_nan(loss).numpy())}")
print(f"  Is Inf: {bool(tf.math.is_inf(loss).numpy())}")
assert not tf.math.is_nan(loss), "Loss should not be NaN!"
assert not tf.math.is_inf(loss), "Loss should not be Inf!"

print("\n[TEST 2] All zero predictions (worst case)")
y_true_zeros = tf.constant(np.random.randn(512, 1).astype(np.float32))
y_pred_zeros = tf.zeros((512, 1), dtype=tf.float32)
loss_zeros = loss_fn(y_true_zeros, y_pred_zeros)
print(f"  Loss: {float(loss_zeros):.6f}")
print(f"  Is NaN: {bool(tf.math.is_nan(loss_zeros).numpy())}")
print(f"  Is Inf: {bool(tf.math.is_inf(loss_zeros).numpy())}")
assert not tf.math.is_nan(loss_zeros), "Loss with zero preds should not be NaN!"
assert not tf.math.is_inf(loss_zeros), "Loss with zero preds should not be Inf!"

print("\n[TEST 3] Very small predictions")
y_true_small = tf.constant(np.random.randn(512, 1).astype(np.float32))
y_pred_small = tf.constant(np.random.randn(512, 1).astype(np.float32) * 1e-8)
loss_small = loss_fn(y_true_small, y_pred_small)
print(f"  Loss: {float(loss_small):.6f}")
print(f"  Is NaN: {bool(tf.math.is_nan(loss_small).numpy())}")
print(f"  Is Inf: {bool(tf.math.is_inf(loss_small).numpy())}")
assert not tf.math.is_nan(loss_small), "Loss with small preds should not be NaN!"

print("\n[TEST 4] Training simulation (1 epoch, 3 batches)")
# Create small model
model = create_paper_model(sequence_length=90, n_features=157)

# Compile with AntiCollapseDirectionalLoss
from training.train_1d_regressor_final import directional_accuracy_metric, r_squared_metric
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss=loss_fn,
    metrics=['mae', directional_accuracy_metric, r_squared_metric]
)

# Create dummy data
X_train = np.random.randn(512, 90, 157).astype(np.float32)
y_train = np.random.randn(512, 1).astype(np.float32)

# Train for 1 step
print("  Training for 3 batches...")
history = model.fit(X_train, y_train, epochs=1, batch_size=512, verbose=1)

# Check metrics
final_loss = history.history['loss'][0]
final_mae = history.history['mae'][0]
final_dir_acc = history.history['directional_accuracy_metric'][0]

print(f"\n  Final loss: {final_loss:.6f}")
print(f"  Final MAE: {final_mae:.6f}")
print(f"  Final dir_acc: {final_dir_acc:.4f}")

# Assertions
assert not np.isnan(final_loss), "Training loss should not be NaN!"
assert not np.isinf(final_loss), "Training loss should not be Inf!"
assert not np.isnan(final_mae), "MAE should not be NaN!"
assert final_loss < 999, "Loss should not be stuck at 1000!"

print("\n" + "="*70)
print("ALL TESTS PASSED!")
print("="*70)
print("\nThe AntiCollapseDirectionalLoss is numerically stable and ready for training.")
