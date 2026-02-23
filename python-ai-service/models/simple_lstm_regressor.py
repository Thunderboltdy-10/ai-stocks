"""
Simple LSTM Regressor - No Transformer, No Attention

NUCLEAR FIX v8.0 (January 2026): Pure LSTM without transformer attention.

The LSTM+Transformer hybrid consistently experiences variance collapse around
epoch 25-35. Research suggests the attention mechanism may be the source of
instability for this financial prediction task.

This simpler architecture removes:
- Transformer blocks (no attention)
- Multi-head ensemble (train-test mismatch source)
- Complex regularization

And keeps:
- Bidirectional LSTM for sequence learning
- Simple dense output
- Dropout for regularization

Architecture:
    Input (seq_len, features)
    -> Bidirectional LSTM (64 units)
    -> Dropout (0.3)
    -> Dense (32, relu)
    -> Dropout (0.2)
    -> Dense (1, linear)
"""

import keras
from keras import layers, ops
import numpy as np


class SimpleLSTMRegressor(keras.Model):
    """
    Pure LSTM regressor without transformer attention.

    Designed for stability over complexity. If the LSTM+Transformer
    keeps collapsing, this simpler model may be more reliable.
    """

    def __init__(
        self,
        seq_len: int = 90,
        num_features: int = 115,
        lstm_units: int = 64,
        dense_units: int = 32,
        dropout_rate: float = 0.3,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.seq_len = seq_len
        self.num_features = num_features
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout_rate = dropout_rate

        # Input normalization
        self.input_norm = layers.LayerNormalization(epsilon=1e-6, name="input_norm")

        # Bidirectional LSTM - captures both forward and backward patterns
        self.lstm = layers.Bidirectional(
            layers.LSTM(
                lstm_units,
                return_sequences=False,  # Only need final state
                dropout=0.1,
                recurrent_dropout=0.0,  # Avoid recurrent dropout (slow on GPU)
                kernel_regularizer=keras.regularizers.L2(1e-5),
            ),
            name="bilstm"
        )

        # Dense layers
        self.dropout1 = layers.Dropout(dropout_rate, name="dropout1")
        self.dense1 = layers.Dense(
            dense_units,
            activation="relu",
            kernel_regularizer=keras.regularizers.L2(1e-5),
            name="dense1"
        )
        self.dropout2 = layers.Dropout(dropout_rate * 0.5, name="dropout2")

        # Output layer - simple linear projection
        self.output_dense = layers.Dense(
            1,
            kernel_initializer=keras.initializers.HeNormal(seed=42),
            bias_initializer="zeros",
            name="output"
        )

    def call(self, inputs, training=False):
        # Normalize input
        x = self.input_norm(inputs)

        # LSTM encoding
        x = self.lstm(x, training=training)

        # Dense layers with dropout
        x = self.dropout1(x, training=training)
        x = self.dense1(x)
        x = self.dropout2(x, training=training)

        # Output prediction
        prediction = self.output_dense(x)

        return prediction

    def get_config(self):
        config = super().get_config()
        config.update({
            "seq_len": self.seq_len,
            "num_features": self.num_features,
            "lstm_units": self.lstm_units,
            "dense_units": self.dense_units,
            "dropout_rate": self.dropout_rate,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)


def build_simple_lstm_regressor(
    seq_len: int = 90,
    num_features: int = 115,
    lstm_units: int = 64,
    dense_units: int = 32,
    dropout_rate: float = 0.3,
) -> keras.Model:
    """
    Build a simple LSTM regressor using Functional API.

    This is an alternative to the class-based model for easier serialization.
    """
    inputs = layers.Input(shape=(seq_len, num_features), name="input")

    # Normalize
    x = layers.LayerNormalization(epsilon=1e-6, name="input_norm")(inputs)

    # Bidirectional LSTM
    x = layers.Bidirectional(
        layers.LSTM(
            lstm_units,
            return_sequences=False,
            dropout=0.1,
            kernel_regularizer=keras.regularizers.L2(1e-5),
        ),
        name="bilstm"
    )(x)

    # Dense layers
    x = layers.Dropout(dropout_rate, name="dropout1")(x)
    x = layers.Dense(
        dense_units,
        activation="relu",
        kernel_regularizer=keras.regularizers.L2(1e-5),
        name="dense1"
    )(x)
    x = layers.Dropout(dropout_rate * 0.5, name="dropout2")(x)

    # Output
    outputs = layers.Dense(
        1,
        kernel_initializer=keras.initializers.HeNormal(seed=42),
        bias_initializer="zeros",
        name="output"
    )(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="simple_lstm_regressor")

    return model


if __name__ == "__main__":
    # Quick test
    model = build_simple_lstm_regressor(seq_len=90, num_features=115)
    model.summary()

    # Test forward pass
    test_input = np.random.randn(32, 90, 115).astype(np.float32)
    output = model(test_input, training=False)
    print(f"\nTest output shape: {output.shape}")
    print(f"Test output range: [{output.numpy().min():.4f}, {output.numpy().max():.4f}]")
