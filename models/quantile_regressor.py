"""
Quantile regressor model and loss.

Provides:
- QuantileLoss: pinball loss for [0.1, 0.5, 0.9]
- QuantileRegressor: LSTM + Transformer shared encoder with three quantile heads
- Helper functions: generate_signal_from_quantiles, compute_position_size

This module is designed to be compatible with the existing training pipeline
by replacing a single-output regressor with three quantile outputs.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class QuantileLoss(keras.losses.Loss):
    """Pinball (quantile) loss that accepts a dict-like prediction.

    Usage: compile(model, loss=QuantileLoss([0.1,0.5,0.9])) when the model
    returns a dict of predictions {'q10':..., 'q50':..., 'q90':...}.
    """
    def __init__(self, quantiles=[0.1, 0.5, 0.9], weights=None, name='quantile_loss'):
        super().__init__(name=name)
        self.quantiles = list(quantiles)
        self.weights = weights if weights is not None else [1.0] * len(self.quantiles)

    def call(self, y_true, y_pred):
        # y_pred may be a dict (preferred) or a tensor with shape (batch, n_quantiles)
        # Normalize y_true to shape (batch, 1)
        y_true = tf.cast(y_true, dtype=tf.float32)
        y_true = tf.reshape(y_true, (-1, 1))

        total_loss = 0.0
        if isinstance(y_pred, dict):
            # Expect keys in order of quantiles
            for i, tau in enumerate(self.quantiles):
                key = f'q{int(tau*100):02d}' if not (tau == 0.5 and 'q50' in y_pred) else f'q{int(tau*100)}'
                # fallback mapping: 0.1->'q10', 0.5->'q50', 0.9->'q90'
                if f'q{int(tau*100)}' in y_pred:
                    pred = tf.cast(y_pred[f'q{int(tau*100)}'], tf.float32)
                else:
                    # try common names
                    name_map = {0.1: 'q10', 0.5: 'q50', 0.9: 'q90'}
                    pred = tf.cast(y_pred.get(name_map.get(tau)), tf.float32)

                if pred is None:
                    raise ValueError(f"Prediction missing for quantile {tau}")

                # ensure shape (batch,1)
                pred = tf.reshape(pred, (-1, 1))

                error = y_true - pred
                loss = tf.maximum(tau * error, (tau - 1.0) * error)
                loss = tf.reduce_mean(loss)
                total_loss += self.weights[i] * loss
        else:
            # Assume tensor with last dim == n_quantiles in same order as self.quantiles
            y_pred = tf.cast(y_pred, tf.float32)
            n_q = tf.shape(y_pred)[-1]
            # Note: We can't check n_q value in python if it's symbolic, but we can trust the model output structure
            # or use tf.debugging.assert_equal if strictly needed. For now, we assume correctness.
            
            for i, tau in enumerate(self.quantiles):
                pred = y_pred[..., i]
                pred = tf.reshape(pred, (-1, 1))
                
                error = y_true - pred
                loss = tf.maximum(tau * error, (tau - 1.0) * error)
                loss = tf.reduce_mean(loss)
                total_loss += self.weights[i] * loss

        return total_loss


class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, mlp_dim=None, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)

        self.mlp_dim = mlp_dim or (d_model * 4)
        self.dense_hidden = layers.Dense(self.mlp_dim, activation='relu')
        self.dense_out = layers.Dense(d_model)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout2 = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_out = self.att(x, x, attention_mask=None)
        attn_out = self.dropout1(attn_out, training=training)
        out1 = self.norm1(x + attn_out)

        mlp_out = self.dense_hidden(out1)
        mlp_out = self.dense_out(mlp_out)
        mlp_out = self.dropout2(mlp_out, training=training)
        return self.norm2(out1 + mlp_out)


class QuantileRegressor(keras.Model):
    """LSTM + Transformer encoder with three quantile heads."""
    def __init__(self, n_features=123, sequence_length=90, d_model=128,
                 n_transformer_blocks=6, n_heads=4, dropout=0.2, lstm_units=64):
        super().__init__()
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.n_transformer_blocks = n_transformer_blocks

        # LSTM encoder
        self.lstm = layers.LSTM(lstm_units, return_sequences=True)

        # Project to d_model for transformer
        self.project = layers.Dense(d_model)

        # Transformer blocks
        self.transformer_blocks = [TransformerBlock(d_model, num_heads=n_heads, dropout=dropout)
                                   for _ in range(n_transformer_blocks)]

        # Pooling to get fixed-length vector
        self.pool = layers.GlobalAveragePooling1D()

        # Shared dense bottleneck
        self.shared_dense = layers.Dense(d_model, activation='relu')
        self.shared_dropout = layers.Dropout(dropout)

        # Quantile heads
        def make_head():
            return keras.Sequential([
                layers.Dense(32, activation='relu'),
                layers.Dropout(dropout),
                layers.Dense(1)
            ])

        self.q10_head = make_head()
        self.q50_head = make_head()
        self.q90_head = make_head()

    def call(self, inputs, training=False):
        """Forward pass.

        Args:
            inputs: Tensor with shape (batch, sequence_length, n_features)
        Returns:
            dict: {'q10': tensor, 'q50': tensor, 'q90': tensor}
        """
        x = tf.cast(inputs, tf.float32)
        # LSTM produces (batch, seq_len, lstm_units)
        x = self.lstm(x, training=training)
        # Project
        x = self.project(x)

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x, training=training)

        # Pool to single vector
        x = self.pool(x)
        x = self.shared_dense(x)
        x = self.shared_dropout(x, training=training)

        q10 = self.q10_head(x, training=training)
        q50 = self.q50_head(x, training=training)
        q90 = self.q90_head(x, training=training)

        # Ensure shape (batch,) for convenience sometimes
        q10 = tf.reshape(q10, (-1, 1))
        q50 = tf.reshape(q50, (-1, 1))
        q90 = tf.reshape(q90, (-1, 1))

        return {'q10': q10, 'q50': q50, 'q90': q90}

    def get_config(self):
        return {
            'n_features': self.n_features,
            'sequence_length': self.sequence_length,
            'd_model': self.d_model,
            'n_transformer_blocks': self.n_transformer_blocks
        }


def generate_signal_from_quantiles(q10, q50, q90, risk_tolerance='moderate'):
    """Generate trading signal and position size from quantile predictions."""
    # Ensure numeric scalars
    q10 = float(q10)
    q50 = float(q50)
    q90 = float(q90)

    if risk_tolerance == 'conservative':
        if q10 > 0.002 and q50 > 0.005 and q90 > 0.015:
            return 'BUY', compute_position_size(q10, q50, q90)
    elif risk_tolerance == 'moderate':
        if q10 > -0.005 and q50 > 0.005 and q90 > 0.020:
            return 'BUY', compute_position_size(q10, q50, q90)
    elif risk_tolerance == 'aggressive':
        if q50 > 0.003 and q90 > 0.030:
            return 'BUY', compute_position_size(q10, q50, q90)

    return 'HOLD', 0.0


def compute_position_size(q10, q50, q90):
    """Kelly-inspired sizing based on quantile spread and asymmetry."""
    expected_return = float(q50)
    downside_risk = max(abs(float(q10)), 1e-6)
    upside_potential = float(q90)

    distribution_width = max(upside_potential - q10, 1e-6)
    width_penalty = 1.0 / (1.0 + distribution_width * 10.0)

    asymmetry_ratio = upside_potential / max(downside_risk, 1e-6)
    asymmetry_bonus = min(asymmetry_ratio / 3.0, 1.5)

    base_size = 0.25
    adjusted_size = base_size * width_penalty * asymmetry_bonus
    return float(np.clip(adjusted_size, 0.05, 0.50))
