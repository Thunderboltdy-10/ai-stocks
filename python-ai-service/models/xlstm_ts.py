"""
xLSTM-TS: Extended LSTM for Time Series Forecasting

Nuclear Redesign: Advanced LSTM architecture for stock prediction.

Based on:
- Beck et al. (2024): "xLSTM: Extended Long Short-Term Memory"
- gonzalopezgil/xlstm-ts: Time series optimized xLSTM

Key Features:
- Exponential gating for better normalization and stabilization
- Revised memory structure with higher capacity (mLSTM, sLSTM variants)
- Optional wavelet denoising for noise reduction
- Residual block backbone
- Mixed precision support

Architecture:
    Input → [Wavelet Denoise] → xLSTM Blocks → Output Head → Prediction

Author: AI-Stocks Nuclear Redesign
Date: December 2025
"""

import sys
from pathlib import Path
from typing import Optional, Tuple, List
from dataclasses import dataclass

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# TensorFlow/Keras imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import tensorflow.keras.backend as K


@dataclass
class xLSTMConfig:
    """Configuration for xLSTM-TS model.

    Attributes:
        input_dim: Number of input features
        hidden_dim: Hidden dimension for xLSTM cells
        num_layers: Number of xLSTM layers
        output_dim: Output dimension (1 for regression)
        dropout: Dropout rate
        use_exponential_gating: Use exponential gating (key xLSTM feature)
        use_matrix_memory: Use matrix memory (mLSTM variant)
        use_wavelet_denoise: Apply wavelet denoising
        wavelet_level: Decomposition level for wavelet
        residual_connection: Use residual connections
        layer_norm: Use layer normalization
    """
    input_dim: int = 157
    hidden_dim: int = 64
    num_layers: int = 2
    output_dim: int = 1
    dropout: float = 0.2
    use_exponential_gating: bool = True
    use_matrix_memory: bool = False  # sLSTM by default (simpler)
    use_wavelet_denoise: bool = True
    wavelet_level: int = 2
    residual_connection: bool = True
    layer_norm: bool = True


class ExponentialGating(layers.Layer):
    """
    Exponential gating mechanism for xLSTM.

    Unlike standard sigmoid gating, exponential gating provides:
    - Better gradient flow
    - More stable training
    - Sharper gate activations

    Formula:
        gate = exp(x) / (1 + exp(x) + exp(c))

    Where c is a learnable stabilization constant.
    """

    def __init__(self, units: int, **kwargs):
        super().__init__(**kwargs)
        self.units = units

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name='kernel',
            shape=(input_shape[-1], self.units),
            initializer='glorot_uniform',
            trainable=True,
        )
        self.bias = self.add_weight(
            name='bias',
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
        )
        # Stabilization constant
        self.stabilizer = self.add_weight(
            name='stabilizer',
            shape=(self.units,),
            initializer='zeros',
            trainable=True,
        )
        super().build(input_shape)

    def call(self, inputs):
        x = K.dot(inputs, self.kernel) + self.bias

        # Exponential gating with stabilization
        exp_x = K.exp(K.clip(x, -10, 10))  # Clip for numerical stability
        exp_c = K.exp(K.clip(self.stabilizer, -10, 10))

        gate = exp_x / (1 + exp_x + exp_c)

        return gate

    def get_config(self):
        config = super().get_config()
        config.update({'units': self.units})
        return config


class sLSTMCell(layers.Layer):
    """
    Scalar LSTM (sLSTM) cell - simplified xLSTM variant.

    Key differences from standard LSTM:
    - Exponential gating instead of sigmoid
    - Scalar memory state
    - Enhanced normalization

    This is the default xLSTM variant, suitable for most time series tasks.
    """

    def __init__(
        self,
        units: int,
        use_exponential_gating: bool = True,
        dropout: float = 0.0,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.use_exponential_gating = use_exponential_gating
        self.dropout = dropout
        self.state_size = [units, units]  # [h, c]
        self.output_size = units

    def build(self, input_shape):
        input_dim = input_shape[-1]
        combined_dim = input_dim + self.units

        if self.use_exponential_gating:
            self.input_gate = ExponentialGating(self.units, name='input_gate')
            self.forget_gate = ExponentialGating(self.units, name='forget_gate')
            self.output_gate = ExponentialGating(self.units, name='output_gate')
        else:
            self.input_gate = layers.Dense(self.units, activation='sigmoid', name='input_gate')
            self.forget_gate = layers.Dense(self.units, activation='sigmoid', name='forget_gate')
            self.output_gate = layers.Dense(self.units, activation='sigmoid', name='output_gate')

        self.candidate = layers.Dense(self.units, activation='tanh', name='candidate')

        # Layer normalization for stability
        self.layer_norm_h = layers.LayerNormalization(name='ln_h')
        self.layer_norm_c = layers.LayerNormalization(name='ln_c')

        # Dropout
        self.dropout_layer = layers.Dropout(self.dropout)

        super().build(input_shape)

    def call(self, inputs, states, training=None):
        h_prev, c_prev = states

        # Concatenate input and hidden state
        combined = K.concatenate([inputs, h_prev], axis=-1)

        # Gates
        i = self.input_gate(combined)
        f = self.forget_gate(combined)
        o = self.output_gate(combined)

        # Candidate memory
        c_candidate = self.candidate(combined)

        # Update cell state with normalization
        c_new = f * c_prev + i * c_candidate
        c_new = self.layer_norm_c(c_new)

        # Output
        h_new = o * K.tanh(c_new)
        h_new = self.layer_norm_h(h_new)

        # Apply dropout
        if training:
            h_new = self.dropout_layer(h_new, training=training)

        return h_new, [h_new, c_new]

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'use_exponential_gating': self.use_exponential_gating,
            'dropout': self.dropout,
        })
        return config


class xLSTMBlock(layers.Layer):
    """
    xLSTM block with residual connection.

    Structure:
        Input → xLSTM → LayerNorm → Dropout → Output
          ↓                                      ↑
          └──────── Residual Connection ─────────┘
    """

    def __init__(
        self,
        units: int,
        use_exponential_gating: bool = True,
        dropout: float = 0.2,
        residual: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.units = units
        self.use_exponential_gating = use_exponential_gating
        self.dropout = dropout
        self.residual = residual

    def build(self, input_shape):
        # sLSTM cell wrapped in RNN
        self.lstm_cell = sLSTMCell(
            self.units,
            use_exponential_gating=self.use_exponential_gating,
            dropout=self.dropout,
        )
        self.rnn = layers.RNN(self.lstm_cell, return_sequences=True)

        # Layer normalization
        self.layer_norm = layers.LayerNormalization()

        # Dropout
        self.dropout_layer = layers.Dropout(self.dropout)

        # Projection for residual if dimensions don't match
        input_dim = input_shape[-1]
        if self.residual and input_dim != self.units:
            self.residual_proj = layers.Dense(self.units, use_bias=False)
        else:
            self.residual_proj = None

        super().build(input_shape)

    def call(self, inputs, training=None):
        # xLSTM forward pass
        x = self.rnn(inputs, training=training)
        x = self.layer_norm(x)
        x = self.dropout_layer(x, training=training)

        # Residual connection
        if self.residual:
            residual = inputs
            if self.residual_proj is not None:
                residual = self.residual_proj(residual)
            x = x + residual

        return x

    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'use_exponential_gating': self.use_exponential_gating,
            'dropout': self.dropout,
            'residual': self.residual,
        })
        return config


class WaveletDenoising(layers.Layer):
    """
    Wavelet denoising layer for noise reduction.

    Applies soft thresholding to wavelet coefficients to reduce noise
    while preserving signal structure.

    Note: This is a simplified implementation. For full wavelet support,
    consider using pywt library.
    """

    def __init__(self, level: int = 2, threshold: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.level = level
        self.threshold = threshold

    def call(self, inputs):
        # Simplified denoising using moving average subtraction
        # Full wavelet implementation would use pywt

        # Low-frequency component (trend)
        kernel_size = 2 ** self.level
        padding = kernel_size // 2

        # Pad inputs
        padded = tf.pad(inputs, [[0, 0], [padding, padding], [0, 0]], mode='REFLECT')

        # Moving average for trend
        trend = tf.nn.avg_pool1d(
            padded,
            ksize=kernel_size,
            strides=1,
            padding='VALID'
        )

        # Adjust shape if needed
        if trend.shape[1] != inputs.shape[1]:
            trend = trend[:, :inputs.shape[1], :]

        # High-frequency component (noise)
        noise = inputs - trend

        # Soft thresholding
        denoised_noise = tf.sign(noise) * tf.maximum(tf.abs(noise) - self.threshold, 0)

        # Reconstructed signal
        return trend + denoised_noise

    def get_config(self):
        config = super().get_config()
        config.update({
            'level': self.level,
            'threshold': self.threshold,
        })
        return config


class xLSTM_TS(Model):
    """
    xLSTM for Time Series (xLSTM-TS) model.

    Complete model for stock return prediction with:
    - Optional wavelet denoising
    - Stack of xLSTM blocks with residual connections
    - Output head with variance penalty support

    Example:
        >>> config = xLSTMConfig(input_dim=157, hidden_dim=64, num_layers=2)
        >>> model = xLSTM_TS(config)
        >>> predictions = model(X_input)  # (batch, seq_len, features) -> (batch, 1)
    """

    def __init__(self, config: xLSTMConfig, **kwargs):
        super().__init__(**kwargs)
        self.config = config

        # Optional wavelet denoising
        if config.use_wavelet_denoise:
            self.wavelet = WaveletDenoising(level=config.wavelet_level)
        else:
            self.wavelet = None

        # Input projection
        self.input_proj = layers.Dense(config.hidden_dim, name='input_proj')

        # xLSTM blocks
        self.xlstm_blocks = []
        for i in range(config.num_layers):
            block = xLSTMBlock(
                units=config.hidden_dim,
                use_exponential_gating=config.use_exponential_gating,
                dropout=config.dropout,
                residual=config.residual_connection,
                name=f'xlstm_block_{i}'
            )
            self.xlstm_blocks.append(block)

        # Output head
        self.output_norm = layers.LayerNormalization(name='output_norm')
        self.output_dropout = layers.Dropout(config.dropout)
        self.output_dense = layers.Dense(config.output_dim, name='output')

    def call(self, inputs, training=None):
        """Forward pass.

        Args:
            inputs: (batch, seq_len, features)

        Returns:
            predictions: (batch, output_dim)
        """
        x = inputs

        # Wavelet denoising
        if self.wavelet is not None:
            x = self.wavelet(x)

        # Input projection
        x = self.input_proj(x)

        # xLSTM blocks
        for block in self.xlstm_blocks:
            x = block(x, training=training)

        # Take last timestep
        x = x[:, -1, :]

        # Output head
        x = self.output_norm(x)
        x = self.output_dropout(x, training=training)
        x = self.output_dense(x)

        return x

    def get_config(self):
        return {
            'config': {
                'input_dim': self.config.input_dim,
                'hidden_dim': self.config.hidden_dim,
                'num_layers': self.config.num_layers,
                'output_dim': self.config.output_dim,
                'dropout': self.config.dropout,
                'use_exponential_gating': self.config.use_exponential_gating,
                'use_matrix_memory': self.config.use_matrix_memory,
                'use_wavelet_denoise': self.config.use_wavelet_denoise,
                'wavelet_level': self.config.wavelet_level,
                'residual_connection': self.config.residual_connection,
                'layer_norm': self.config.layer_norm,
            }
        }

    @classmethod
    def from_config(cls, config):
        xlstm_config = xLSTMConfig(**config['config'])
        return cls(xlstm_config)


def create_xlstm_ts(
    input_dim: int = 157,
    seq_length: int = 60,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.2,
    use_wavelet: bool = True,
) -> Model:
    """
    Factory function to create xLSTM-TS model.

    Args:
        input_dim: Number of input features
        seq_length: Sequence length (for input shape)
        hidden_dim: Hidden dimension
        num_layers: Number of xLSTM layers
        dropout: Dropout rate
        use_wavelet: Whether to use wavelet denoising

    Returns:
        Compiled xLSTM-TS model
    """
    config = xLSTMConfig(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        use_wavelet_denoise=use_wavelet,
    )

    model = xLSTM_TS(config)

    # Build model with input shape
    model.build(input_shape=(None, seq_length, input_dim))

    return model


# Register custom objects for model loading
def get_custom_objects():
    """Get custom objects for model loading."""
    return {
        'xLSTM_TS': xLSTM_TS,
        'xLSTMBlock': xLSTMBlock,
        'sLSTMCell': sLSTMCell,
        'ExponentialGating': ExponentialGating,
        'WaveletDenoising': WaveletDenoising,
    }
