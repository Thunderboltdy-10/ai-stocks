"""
LSTM+Transformer Hybrid Model - Based on Ruiru et al. 2024

Paper reference: "LSTM versus Transformers: A Practical Comparison of 
Deep Learning Models for Trading Financial Instruments"

Key findings:
- LSTM32 + 4 Transformer blocks = 95.9% accuracy on financial data
- Pure transformers are unstable on financial time series
- ~301K parameters prevents overfitting on limited financial data

Architecture:
    Input [batch, 60, features]
      ↓
    LSTM(32, return_sequences=True)  ← Stabilizes sequence
      ↓
    Dense(64)  ← Project to d_model
      ↓
    + Positional Encoding
      ↓
    Transformer Block x4  ← Captures patterns
      ↓
    GlobalAveragePooling1D
      ↓
    Dense(1)  ← Output: log return
"""

from __future__ import annotations

import os
import numpy as np

# ============================================================================
# KERAS 3 BACKEND-AGNOSTIC IMPORTS
# ============================================================================
# Support both TensorFlow and PyTorch backends via Keras 3
try:
    import keras
    from keras import Model, layers, ops
    # Use keras.ops for backend-agnostic operations
    K = ops
    KERAS_3 = True
except ImportError:
    # Fallback to TensorFlow
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import Model, layers
    K = tf
    KERAS_3 = False

# Backend detection
def get_backend():
    if KERAS_3:
        return keras.backend.backend()
    return 'tensorflow'


# =============================================================================
# CUSTOM LOSS FUNCTIONS
# =============================================================================

@keras.saving.register_keras_serializable(package="models.lstm_transformer_paper")
class DirectionalHuberLoss(keras.losses.Loss):
    """
    Custom Huber loss that heavily penalizes wrong-direction predictions.

    This loss function combines the robustness of Huber loss with a penalty
    for predictions that have the wrong sign (direction) compared to the
    true value. This is particularly important for trading applications
    where predicting the correct direction is more important than the
    exact magnitude.

    The loss is computed as:
        loss = huber_loss * (1.0 + direction_penalty)

    Where direction_penalty = direction_weight when signs don't match, else 0.

    Args:
        delta: Huber delta parameter. Controls transition between linear
            and quadratic regimes. Default 1.0.
        direction_weight: Multiplier for wrong-direction errors. Higher values
            penalize directional mistakes more heavily. Default 0.1 (reduced from 2.0
            to prevent multi-task gradient conflicts).
        name: Name for the loss function.

    Example:
        >>> loss_fn = DirectionalHuberLoss(delta=1.0, direction_weight=0.1)
        >>> model.compile(optimizer='adam', loss=loss_fn)

        # If y_true=+0.02 and y_pred=-0.01 (wrong direction):
        #   base_loss = huber(0.02, -0.01)
        #   final_loss = base_loss * (1.0 + 0.1) = base_loss * 1.1

        # If y_true=+0.02 and y_pred=+0.05 (correct direction):
        #   base_loss = huber(0.02, 0.05)
        #   final_loss = base_loss * (1.0 + 0.0) = base_loss * 1.0
    """

    def __init__(
        self,
        delta: float = 1.0,
        direction_weight: float = 0.1,  # REDUCED from 2.0 to prevent gradient conflicts
        reduction: str = 'sum_over_batch_size',
        name: str = 'directional_huber',
    ):
        super().__init__(name=name, reduction=reduction)
        self.delta = delta
        self.direction_weight = direction_weight
        self.huber = keras.losses.Huber(delta=delta, reduction='none')
    
    def call(self, y_true, y_pred):
        """
        Compute directional Huber loss (mixed precision safe).

        Args:
            y_true: True target values.
            y_pred: Predicted values.

        Returns:
            Scalar loss tensor.
        """
        # Flatten inputs for consistent processing
        y_true = ops.reshape(y_true, [-1])
        y_pred = ops.reshape(y_pred, [-1])

        # Standard Huber loss (per-sample)
        base_loss = self.huber(y_true, y_pred)

        # Direction penalty - use float32 for stability
        y_true_fp32 = ops.cast(y_true, 'float32')
        y_pred_fp32 = ops.cast(y_pred, 'float32')
        y_true_sign = ops.sign(y_true_fp32)
        y_pred_sign = ops.sign(y_pred_fp32)

        # Identify wrong-direction predictions (signs don't match)
        wrong_direction = ops.cast(
            ops.not_equal(y_true_sign, y_pred_sign),
            'float32'
        )

        # Apply penalty multiplier to wrong-direction predictions
        direction_penalty = wrong_direction * self.direction_weight

        # Combined loss: base_loss * (1.0 + penalty)
        # Correct direction: base_loss * 1.0
        # Wrong direction: base_loss * (1.0 + direction_weight)
        direction_penalty_casted = ops.cast(direction_penalty, base_loss.dtype)
        weighted_loss = base_loss * (1.0 + direction_penalty_casted)

        # Final safety check: replace NaN/inf with large finite value
        mean_loss = ops.mean(weighted_loss)
        mean_loss = ops.where(
            ops.logical_or(ops.isnan(mean_loss), ops.isinf(mean_loss)),
            ops.cast(1000.0, mean_loss.dtype),
            mean_loss
        )

        return mean_loss
    
    def get_config(self) -> dict:
        """Return config for serialization."""
        config = super().get_config()
        config.update({
            'delta': self.delta,
            'direction_weight': self.direction_weight,
        })
        return config


@keras.saving.register_keras_serializable(package="models.lstm_transformer_paper")
@keras.saving.register_keras_serializable(package="models.lstm_transformer_paper")
class AntiCollapseDirectionalLoss(keras.losses.Loss):
    """
    SIMPLIFIED Anti-collapse loss (v6) - No log operations, just Huber + penalties
    
    This is a super simple, numerically stable version that avoids -inf issues.
    """
    
    def __init__(
        self,
        delta: float = 1.0,
        direction_weight: float = 0.1,
        variance_penalty_weight: float = 2.0,  # INCREASED from 1.0 for stronger variance enforcement
        min_variance_target: float = 0.008,   # INCREASED from 0.005 for more realistic targets
        sign_diversity_weight: float = 5.0,   # INCREASED from 0.3 to STRONGLY prevent prediction bias
        name: str = 'anti_collapse_directional_v6',
        reduction: str = 'sum_over_batch_size',
    ):
        super().__init__(name=name, reduction=reduction)
        self.delta = delta
        self.direction_weight = direction_weight
        self.variance_penalty_weight = variance_penalty_weight
        self.min_variance_target = min_variance_target
        self.sign_diversity_weight = sign_diversity_weight
        self.huber = keras.losses.Huber(delta=delta, reduction='none')
    
    def call(self, y_true, y_pred):
        """Compute loss with ZERO log operations to avoid NaN."""
        # Flatten to 1D
        y_true = ops.reshape(y_true, [-1])
        y_pred = ops.reshape(y_pred, [-1])
        
        # Cast to float32 for numerical stability
        y_true = ops.cast(y_true, 'float32')
        y_pred = ops.cast(y_pred, 'float32')
        
        # 1. BASE HUBER LOSS
        base_loss = self.huber(y_true, y_pred)
        
        # 2. DIRECTION PENALTY - penalize wrong-sign predictions
        wrong_direction = ops.cast(
            ops.not_equal(ops.sign(y_true), ops.sign(y_pred)),
            'float32'
        )
        base_loss = base_loss * (1.0 + wrong_direction * self.direction_weight)
        
        # 3. VARIANCE PENALTY - only when std is too small (no log!)
        pred_std = ops.sqrt(ops.maximum(ops.var(y_pred), 1e-7))
        # Penalize only when pred_std < min_variance_target
        # Use simple quadratic: if std < target, penalty = (target - std)^2
        variance_gap = ops.maximum(self.min_variance_target - pred_std, 0.0)
        variance_penalty = self.variance_penalty_weight * ops.square(variance_gap)
        
        # 4. SIGN DIVERSITY PENALTY - encourage 50/50 positive/negative
        positive_frac = ops.mean(ops.cast(ops.greater(y_pred, 0.0), 'float32'))
        sign_diversity_penalty = self.sign_diversity_weight * ops.square(positive_frac - 0.5) * 4.0
        
        # TOTAL LOSS
        total_loss = ops.mean(base_loss) + variance_penalty + sign_diversity_penalty
        
        # Safety: if NaN, return fallback (MSE only)
        is_nan = ops.isnan(total_loss)
        fallback = ops.mean(ops.square(y_true - y_pred))
        total_loss = ops.where(is_nan, fallback, total_loss)
        
        return total_loss
    
    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'delta': self.delta,
            'direction_weight': self.direction_weight,
            'variance_penalty_weight': self.variance_penalty_weight,
            'min_variance_target': self.min_variance_target,
            'sign_diversity_weight': self.sign_diversity_weight,
        })
        return config


class AsymmetricDirectionalLoss(keras.losses.Loss):
    """
    Asymmetric loss that can penalize false positives vs false negatives differently.
    
    Useful when missing a downside move is more costly than missing an upside move
    (or vice versa), common in risk-averse trading strategies.
    
    Args:
        delta: Huber delta parameter.
        direction_weight: Base multiplier for wrong-direction errors.
        false_positive_weight: Extra penalty for predicting positive when actually negative.
        false_negative_weight: Extra penalty for predicting negative when actually positive.
        name: Name for the loss function.
    """
    
    def __init__(
        self,
        delta: float = 1.0,
        direction_weight: float = 2.0,
        false_positive_weight: float = 1.0,
        false_negative_weight: float = 1.0,
        reduction: str = 'sum_over_batch_size',
        name: str = 'asymmetric_directional',
    ):
        super().__init__(name=name, reduction=reduction)
        self.delta = delta
        self.direction_weight = direction_weight
        self.false_positive_weight = false_positive_weight
        self.false_negative_weight = false_negative_weight
        self.huber = keras.losses.Huber(delta=delta, reduction='none')
    
    def call(self, y_true, y_pred):
        """Compute asymmetric directional loss (mixed precision safe)."""
        y_true = ops.reshape(y_true, [-1])
        y_pred = ops.reshape(y_pred, [-1])

        base_loss = self.huber(y_true, y_pred)

        # Use float32 for comparisons to avoid precision issues
        y_true_fp32 = ops.cast(y_true, 'float32')
        y_pred_fp32 = ops.cast(y_pred, 'float32')

        # Identify error types
        y_true_positive = ops.cast(y_true_fp32 > 0, 'float32')
        y_true_negative = ops.cast(y_true_fp32 < 0, 'float32')
        y_pred_positive = ops.cast(y_pred_fp32 > 0, 'float32')
        y_pred_negative = ops.cast(y_pred_fp32 < 0, 'float32')

        # False positive: predicted positive but actually negative
        false_positive = y_pred_positive * y_true_negative

        # False negative: predicted negative but actually positive
        false_negative = y_pred_negative * y_true_positive

        # Compute directional penalty
        direction_penalty = (
            self.direction_weight * (false_positive + false_negative)
            + (self.false_positive_weight - 1.0) * false_positive
            + (self.false_negative_weight - 1.0) * false_negative
        )

        # Cast penalty back to base loss dtype
        direction_penalty_casted = ops.cast(direction_penalty, base_loss.dtype)
        weighted_loss = base_loss * (1.0 + direction_penalty_casted)

        # Safety check
        mean_loss = ops.mean(weighted_loss)
        mean_loss = ops.where(
            ops.logical_or(ops.isnan(mean_loss), ops.isinf(mean_loss)),
            ops.cast(1000.0, mean_loss.dtype),
            mean_loss
        )

        return mean_loss
    
    def get_config(self) -> dict:
        """Return config for serialization."""
        config = super().get_config()
        config.update({
            'delta': self.delta,
            'direction_weight': self.direction_weight,
            'false_positive_weight': self.false_positive_weight,
            'false_negative_weight': self.false_negative_weight,
        })
        return config


# =============================================================================
# CUSTOM METRICS
# =============================================================================

@keras.saving.register_keras_serializable(package="models.lstm_transformer_paper")
def directional_accuracy_metric(y_true, y_pred):
    """
    Compute directional accuracy: fraction of predictions with correct sign.
    
    This metric measures how often the model correctly predicts whether
    the return will be positive or negative, regardless of magnitude.
    This is often more important than MAE for trading applications.
    
    Args:
        y_true: True target values.
        y_pred: Predicted values.
    
    Returns:
        Scalar tensor with directional accuracy (0.0 to 1.0).
    
    Example:
        >>> y_true = tf.constant([0.02, -0.01, 0.05, -0.03])
        >>> y_pred = tf.constant([0.01, -0.02, -0.01, -0.01])
        >>> acc = directional_accuracy_metric(y_true, y_pred)
        >>> # Correct: +/+, -/-, -/-; Wrong: +/-
        >>> # Accuracy = 3/4 = 0.75
    """
    y_true = ops.reshape(y_true, [-1])
    y_pred = ops.reshape(y_pred, [-1])
    
    y_true_sign = ops.sign(y_true)
    y_pred_sign = ops.sign(y_pred)
    
    correct = ops.cast(ops.equal(y_true_sign, y_pred_sign), 'float32')
    
    return ops.mean(correct)


def create_directional_accuracy_metric():
    """
    Factory function to create directional accuracy metric.
    
    Use this when you need a metric instance rather than a function.
    
    Returns:
        Keras metric that computes directional accuracy.
    """
    return keras.metrics.MeanMetricWrapper(
        fn=directional_accuracy_metric,
        name='directional_accuracy',
    )


# =============================================================================
# MODEL CLASSES
# =============================================================================

@keras.saving.register_keras_serializable(package="models.lstm_transformer_paper")
class LSTMTransformerPaper(Model):
    """Paper-proven LSTM+Transformer hybrid."""

    def __init__(
        self,
        sequence_length: int = 60,
        n_features: int = 30,
        lstm_units: int = 32,
        d_model: int = 64,
        num_heads: int = 4,
        num_blocks: int = 4,
        ff_dim: int = 128,
        dropout: float = 0.2,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self.sequence_length = sequence_length
        self.n_features = n_features

        # STEP 1: LSTM preprocessing
        self.lstm_layer = layers.LSTM(
            lstm_units,
            return_sequences=True,
            dropout=dropout,
            recurrent_dropout=0.0,
            name="lstm_preprocessor",
        )

        # STEP 2: Project to d_model
        self.projection = layers.Dense(d_model, name="lstm_projection")

        # STEP 3: Positional encoding (stored as numpy, converted during forward pass)
        self._d_model = d_model
        self._max_len = sequence_length
        self._pe_numpy = self._create_positional_encoding_np(sequence_length, d_model)

        # STEP 4: Transformer blocks
        self.transformer_blocks = []
        for i in range(num_blocks):
            block = self._create_transformer_block(
                d_model, num_heads, ff_dim, dropout, name=f"transformer_{i}"
            )
            self.transformer_blocks.append(block)

        # STEP 5: Output layers
        self.global_pool = layers.GlobalAveragePooling1D(name="global_pool")
        self.dropout_out = layers.Dropout(dropout, name="dropout_output")
        self.output_dense = layers.Dense(1, name="prediction_output")

    @property
    def pos_encoding(self):
        """Backward-compatible property for accessing positional encoding.

        Some functions (create_binary_classifier, create_multitask_regressor in
        model_validation_suite.py) access base.pos_encoding instead of base._pe_numpy.
        This property provides backward compatibility.
        """
        return self._pe_numpy

    @staticmethod
    def _create_positional_encoding_np(max_len: int, d_model: int) -> np.ndarray:
        """Sinusoidal positional encoding as numpy array."""
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pe = np.zeros((max_len, d_model), dtype=np.float32)
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)

        return pe[np.newaxis, :, :]

    @staticmethod
    def _create_transformer_block(
        d_model: int, num_heads: int, ff_dim: int, dropout: float, name: str
    ) -> dict[str, layers.Layer]:
        """Single transformer encoder block."""
        return {
            "attention": layers.MultiHeadAttention(
                num_heads=num_heads,
                key_dim=d_model // num_heads,
                dropout=dropout,
                name=f"{name}_mha",
            ),
            "norm1": layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm1"),
            "ffn1": layers.Dense(ff_dim, activation="relu", name=f"{name}_ffn1"),
            "ffn2": layers.Dense(d_model, name=f"{name}_ffn2"),
            "norm2": layers.LayerNormalization(epsilon=1e-6, name=f"{name}_norm2"),
            "dropout1": layers.Dropout(dropout, name=f"{name}_drop1"),
            "dropout2": layers.Dropout(dropout, name=f"{name}_drop2"),
        }

    def call(self, inputs, training: bool = False):
        """Forward pass: LSTM → Project → Add PE → Transform → Pool → Predict."""
        x = self.lstm_layer(inputs, training=training)
        x = self.projection(x)

        # Get sequence length dynamically and slice positional encoding
        seq_len = ops.shape(x)[1]
        pe = ops.convert_to_tensor(self._pe_numpy[:, :seq_len, :])
        # Phase 6 FIX: Cast PE to input dtype (half for mixed precision)
        pe = ops.cast(pe, x.dtype)
        x = x + pe

        for block in self.transformer_blocks:
            attn_out = block["attention"](x, x, training=training)
            attn_out = block["dropout1"](attn_out, training=training)
            x = block["norm1"](x + attn_out)

            ffn_out = block["ffn2"](block["ffn1"](x))
            ffn_out = block["dropout2"](ffn_out, training=training)
            x = block["norm2"](x + ffn_out)

        x = self.global_pool(x)
        x = self.dropout_out(x, training=training)
        output = self.output_dense(x)

        # NUCLEAR FIX: Anti-collapse noise injection during training
        # If variance is too low, add small random noise to prevent gradient death
        if training:
            import tensorflow as tf  # Use TensorFlow for random ops
            output_fp32 = ops.cast(output, 'float32')
            output_std = ops.std(output_fp32)
            # Target minimum std is 0.01 (1%)
            # If std is below this, inject noise proportional to the gap
            target_std = 0.01
            noise_scale = ops.maximum(target_std - output_std, 0.0) * 0.5
            # Only add noise if variance is collapsing
            # Use TensorFlow's random.normal since Keras ops doesn't have random
            noise = tf.random.normal(tf.shape(output), dtype=tf.float32) * noise_scale
            output = output + ops.cast(noise, output.dtype)

        return output

    def get_config(self) -> dict[str, int | float]:
        """Serialize config for saving."""
        return {
            "sequence_length": self.sequence_length,
            "n_features": self.n_features,
            "lstm_units": 32,   
            "d_model": 64,      
            "num_heads": 4,
            "num_blocks": 3,    
            "ff_dim": 128,      
            "dropout": 0.35,    
        }


def create_paper_model(sequence_length: int = 60, n_features: int = 30) -> LSTMTransformerPaper:
    """Factory function to create reduced-capacity architecture for overfitting prevention.
    
    Reduced from original paper's 301K parameters to ~75-100K parameters:
    - lstm_units: 64 -> 32 (halved)
    - d_model: 128 -> 64 (halved)
    - num_blocks: 6 -> 3 (halved)
    - ff_dim: 256 -> 128 (halved)
    - dropout: 0.2 -> 0.35 (increased for regularization)
    """
    model = LSTMTransformerPaper(
        sequence_length=sequence_length,
        n_features=n_features,
        lstm_units=32,    # Reduced from 64 to prevent overfitting
        d_model=64,       # Reduced from 128 to prevent overfitting
        num_heads=4,
        num_blocks=3,     # Reduced from 6 to prevent overfitting
        ff_dim=128,       # Reduced from 256 to prevent overfitting
        dropout=0.35,     # Increased from 0.2 for regularization
    )

    # Use backend-agnostic random input
    dummy_input = np.random.randn(1, sequence_length, n_features).astype(np.float32)
    _ = model(dummy_input, training=False)

    param_count = model.count_params()
    print(f"✅ Reduced-capacity model created: {param_count:,} parameters")
    print(f"   Expected: ~75K-100K parameters (reduced from 301K to prevent overfitting)")

    if 50_000 < param_count < 150_000:
        print("   ✓ Parameter count in target range for small datasets")
    else:
        print(f"   ⚠️ WARNING: Parameter count {param_count:,} outside expected range (50K-150K)!")

    return model


if __name__ == "__main__":
    print("Testing LSTM+Transformer Paper Model\n")
    print(f"Backend: {get_backend()}")

    model = create_paper_model(sequence_length=60, n_features=30)

    test_input = np.random.randn(4, 60, 30).astype(np.float32)
    output = model(test_input, training=False)

    print("\nTest forward pass:")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Output shape: {output.shape}")
    output_np = np.array(output)
    print(
        f"  Output range: [{output_np.min():.4f}, {output_np.max():.4f}]"
    )

    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    dummy_y = np.random.randn(4, 1).astype(np.float32)
    history = model.fit(test_input, dummy_y, epochs=2, verbose=0)

    print("\nTraining test:")
    print(f"  Initial loss: {history.history['loss'][0]:.6f}")
    print(f"  Final loss: {history.history['loss'][-1]:.6f}")
    print(
        "  Model can train (loss decreased: "
        f"{history.history['loss'][0] > history.history['loss'][-1]})"
    )

    # Test DirectionalHuberLoss
    print("\n" + "="*50)
    print("Testing DirectionalHuberLoss")
    print("="*50)
    
    loss_fn = DirectionalHuberLoss(delta=1.0, direction_weight=2.0)
    
    # Test case 1: Correct direction predictions
    y_true_correct = np.array([0.02, -0.01, 0.05, -0.03], dtype=np.float32)
    y_pred_correct = np.array([0.01, -0.02, 0.03, -0.01], dtype=np.float32)  # All correct direction
    loss_correct = loss_fn(y_true_correct, y_pred_correct)
    print(f"\n  Correct directions:")
    print(f"    y_true: {y_true_correct}")
    print(f"    y_pred: {y_pred_correct}")
    print(f"    Loss: {float(loss_correct):.6f}")
    
    # Test case 2: Wrong direction predictions
    y_true_wrong = np.array([0.02, -0.01, 0.05, -0.03], dtype=np.float32)
    y_pred_wrong = np.array([-0.01, 0.02, -0.03, 0.01], dtype=np.float32)  # All wrong direction
    loss_wrong = loss_fn(y_true_wrong, y_pred_wrong)
    print(f"\n  Wrong directions:")
    print(f"    y_true: {y_true_wrong}")
    print(f"    y_pred: {y_pred_wrong}")
    print(f"    Loss: {float(loss_wrong):.6f}")
    
    # Wrong direction should have higher loss
    assert float(loss_wrong) > float(loss_correct), "Wrong direction should have higher loss!"
    print(f"\n  Loss ratio (wrong/correct): {float(loss_wrong) / float(loss_correct):.2f}x")
    print("  DirectionalHuberLoss test passed!")
    
    # Test directional_accuracy_metric
    print("\n" + "="*50)
    print("Testing directional_accuracy_metric")
    print("="*50)
    
    y_true_mixed = np.array([0.02, -0.01, 0.05, -0.03], dtype=np.float32)
    y_pred_mixed = np.array([0.01, -0.02, -0.01, -0.01], dtype=np.float32)  # 3/4 correct
    acc = directional_accuracy_metric(y_true_mixed, y_pred_mixed)
    print(f"\n  y_true: {y_true_mixed}")
    print(f"  y_pred: {y_pred_mixed}")
    print(f"  Expected accuracy: 0.75 (3/4 correct)")
    print(f"  Computed accuracy: {float(acc):.2f}")
    assert abs(float(acc) - 0.75) < 0.01, "Directional accuracy should be 0.75!"
    print("  directional_accuracy_metric test passed!")
    
    # Test model with DirectionalHuberLoss
    print("\n" + "="*50)
    print("Testing model with DirectionalHuberLoss")
    print("="*50)
    
    model2 = create_paper_model(sequence_length=60, n_features=30)
    model2.compile(
        optimizer="adam",
        loss=DirectionalHuberLoss(delta=1.0, direction_weight=2.0),
        metrics=["mae", directional_accuracy_metric]
    )
    
    # Create training data with clear directional patterns
    np.random.seed(42)
    X_train = np.random.randn(100, 60, 30).astype(np.float32)
    y_train = np.random.randn(100, 1).astype(np.float32) * 0.02
    
    history2 = model2.fit(X_train, y_train, epochs=3, verbose=1, validation_split=0.2)
    
    print(f"\n  Final metrics:")
    print(f"    loss: {history2.history['loss'][-1]:.4f}")
    print(f"    directional_accuracy: {history2.history['directional_accuracy_metric'][-1]:.3f}")
    print(f"    val_directional_accuracy: {history2.history['val_directional_accuracy_metric'][-1]:.3f}")
    
    print("\n" + "="*50)
    print("All tests passed!")
    print("="*50)
