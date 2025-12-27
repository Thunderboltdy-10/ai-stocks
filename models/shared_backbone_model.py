"""
Shared backbone model with symbol-specific heads.

Provides:
- SharedBackboneModel: Keras Model with LSTM+Transformer encoder and per-symbol heads
- quantile_loss: pinball loss for quantile regression
- train_phase1_pretrain, train_phase2_finetune_heads, train_phase3_endtoend: helper training routines

Notes:
- Input expectations: x -> shape (N, seq_len, n_features), symbol_ids -> (N,) ints,
  y -> shape (N,) scalar target (e.g., next-day return). During training the model
  predicts a vector of quantiles per sample (n_quantiles).
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from typing import List, Dict, Optional


class TransformerBlock(layers.Layer):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model // num_heads)
        self.norm1 = layers.LayerNormalization()
        self.ffn = keras.Sequential([
            layers.Dense(d_model * 4, activation='relu'),
            layers.Dense(d_model),
        ])
        self.norm2 = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout)

    def call(self, x, training=False):
        attn_out = self.att(x, x, training=training)
        out1 = self.norm1(x + self.dropout(attn_out, training=training))
        ffn_out = self.ffn(out1)
        out2 = self.norm2(out1 + self.dropout(ffn_out, training=training))
        return out2


class SharedBackboneModel(keras.Model):
    """Multi-symbol model with shared encoder and symbol-specific decoders.

    Call signature: model(x, symbol_ids, training=False) -> tensor (batch, n_quantiles)
    """
    def __init__(self, n_features: int = 123, sequence_length: int = 90, d_model: int = 128,
                 n_transformer_blocks: int = 6, n_heads: int = 4, dropout: float = 0.3,
                 symbols: Optional[List[str]] = None, n_quantiles: int = 3):
        super().__init__()
        self.n_features = int(n_features)
        self.sequence_length = int(sequence_length)
        self.d_model = int(d_model)
        self.n_transformer_blocks = int(n_transformer_blocks)
        self.n_heads = int(n_heads)
        self.dropout = float(dropout)
        self.n_quantiles = int(n_quantiles)

        if symbols is None:
            symbols = ['AAPL', 'TSLA', 'HOOD']
        self.symbols = list(symbols)
        self.symbol_to_index = {s: i for i, s in enumerate(self.symbols)}

        # Build encoder and heads
        self.encoder = self.build_shared_encoder(self.n_features, self.sequence_length,
                                                 self.d_model, self.n_transformer_blocks,
                                                 self.n_heads, self.dropout)

        self.symbol_heads: Dict[str, keras.Model] = {}
        for s in self.symbols:
            self.symbol_heads[s] = self.build_symbol_head(self.d_model, self.n_quantiles, self.dropout)

    def build_shared_encoder(self, n_features, sequence_length, d_model, n_transformer_blocks, n_heads, dropout):
        """Build shared LSTM + Transformer encoder returning a Layer that maps
        (batch, seq_len, n_features) -> (batch, d_model) latent vector.
        """
        inputs = keras.Input(shape=(sequence_length, n_features), name='encoder_input')
        # LSTM to capture sequence dynamics
        x = layers.Masking()(inputs)
        x = layers.LSTM(64, return_sequences=True)(x)
        # Project to d_model
        x = layers.TimeDistributed(layers.Dense(d_model))(x)

        # Transformer blocks
        for _ in range(n_transformer_blocks):
            x = TransformerBlock(d_model, num_heads=n_heads, dropout=dropout)(x)

        # Pool to fixed-size representation
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(d_model, activation='relu')(x)
        model = keras.Model(inputs=inputs, outputs=x, name='shared_encoder')
        return model

    def build_symbol_head(self, d_model, n_quantiles, dropout):
        """Build symbol-specific quantile regression head.

        Returns a Keras Model mapping (batch, d_model) -> (batch, n_quantiles)
        """
        inp = keras.Input(shape=(d_model,), name='head_input')
        x = layers.Dense(64, activation='relu')(inp)
        x = layers.Dropout(dropout)(x)
        out = layers.Dense(n_quantiles, activation='linear')(x)
        return keras.Model(inputs=inp, outputs=out)

    def call(self, inputs, symbol_ids, training=False):
        """Forward pass.

        Args:
            inputs:  Tensor shape (batch, seq_len, n_features)
            symbol_ids: Tensor shape (batch,) ints

        Returns:
            Tensor shape (batch, n_quantiles) with predictions for each sample.
        """
        # Ensure tensors
        x = tf.convert_to_tensor(inputs)
        symbol_ids = tf.convert_to_tensor(symbol_ids)

        # Encode
        latent = self.encoder(x, training=training)  # (batch, d_model)

        batch_size = tf.shape(latent)[0]
        # Prepare output tensor
        outputs = tf.zeros((batch_size, self.n_quantiles), dtype=latent.dtype)

        # For each symbol index, compute head outputs for masked rows and scatter back
        for i, sym in enumerate(self.symbols):
            mask = tf.equal(symbol_ids, i)
            indices = tf.where(mask)
            if tf.shape(indices)[0] == 0:
                continue
            selected = tf.boolean_mask(latent, mask)
            preds = self.symbol_heads[sym](selected, training=training)  # (k, n_quantiles)
            # scatter into outputs
            outputs = tf.tensor_scatter_nd_update(outputs, indices, preds)

        return outputs

    def freeze_encoder(self):
        """Freeze shared encoder weights for Phase 2 training."""
        self.encoder.trainable = False

    def unfreeze_encoder(self):
        """Unfreeze encoder for Phase 3 end-to-end training."""
        self.encoder.trainable = True

    def get_symbol_head(self, symbol: str) -> keras.Model:
        """Get specific symbol's decoder head for isolated training."""
        return self.symbol_heads[symbol]


def quantile_loss(quantiles=[0.1, 0.5, 0.9]):
    q = tf.constant(quantiles, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        # y_true: (batch,) or (batch,1), y_pred: (batch, n_q)
        y_true_exp = tf.expand_dims(tf.cast(y_true, tf.float32), axis=-1)
        errors = y_true_exp - y_pred
        losses = tf.maximum(q * errors, (q - 1) * errors)
        return tf.reduce_mean(losses)

    return loss_fn


def train_phase1_pretrain(model: SharedBackboneModel, x, symbol_ids, y,
                          batch_size=64, epochs=50, lr=1e-3, callbacks=None):
    """Phase 1: pre-train encoder + heads on mixed-symbol batches.

    x: np.ndarray shape (N, seq_len, n_features)
    symbol_ids: np.ndarray shape (N,)
    y: np.ndarray shape (N,) scalar targets
    """
    model.compile(optimizer=keras.optimizers.Adam(lr), loss=quantile_loss())
    # Keras expects inputs as a list matching call signature
    model.fit(x=[x, symbol_ids], y=y, batch_size=batch_size, epochs=epochs, callbacks=callbacks)


def train_phase2_finetune_heads(model: SharedBackboneModel, x, symbol_ids, y,
                                batch_size=64, epochs=30, lr=1e-3, save_dir: Optional[str] = None):
    """Phase 2: freeze encoder, train each head separately on its symbol data.

    This function will iterate over symbols, select symbol-specific samples,
    set all heads non-trainable except the target head, and fit for `epochs`.
    If `save_dir` is provided, head weights will be saved to disk under that folder.
    """
    model.freeze_encoder()
    for sym in model.symbols:
        # select samples
        mask = (symbol_ids == model.symbol_to_index[sym])
        if np.sum(mask) == 0:
            print(f'No samples for {sym}, skipping')
            continue
        x_sym = x[mask]
        y_sym = y[mask]

        # set head trainable flags
        for s2 in model.symbols:
            model.symbol_heads[s2].trainable = (s2 == sym)

        # compile and train
        model.compile(optimizer=keras.optimizers.Adam(lr), loss=quantile_loss())
        model.fit(x=[x_sym, np.full(shape=(len(x_sym),), fill_value=model.symbol_to_index[sym])],
                  y=y_sym, batch_size=batch_size, epochs=epochs)

        # save head weights
        if save_dir is not None:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
            head_path = Path(save_dir) / f'head_{sym}_weights.npz'
            # save weights as numpy arrays
            weights = model.symbol_heads[sym].get_weights()
            np.savez(head_path, *weights)


def train_phase3_endtoend(model: SharedBackboneModel, x, symbol_ids, y,
                          batch_size=64, epochs=20, encoder_lr=1e-5, heads_lr=1e-4):
    """Phase 3: unfreeze encoder and fine-tune with discriminative learning rates.

    Uses two optimizers and applies gradients separately to encoder vs head variables.
    """
    model.unfreeze_encoder()

    # prepare datasets
    ds = tf.data.Dataset.from_tensor_slices(((x, symbol_ids), y)).shuffle(1024).batch(batch_size)

    opt_encoder = keras.optimizers.Adam(learning_rate=encoder_lr)
    opt_heads = keras.optimizers.Adam(learning_rate=heads_lr)
    loss_fn = quantile_loss()

    # identify variables
    encoder_vars = model.encoder.trainable_variables
    # all trainable variables minus encoder vars are heads' vars
    heads_vars = [v for v in model.trainable_variables if v not in encoder_vars]

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        for step, ((xb, symb), yb) in enumerate(ds):
            with tf.GradientTape() as tape:
                preds = model(xb, symb, training=True)
                loss_value = loss_fn(yb, preds)

            grads = tape.gradient(loss_value, model.trainable_variables)
            # split grads
            grads_encoder = grads[:len(encoder_vars)] if len(encoder_vars) > 0 else []
            grads_heads = grads[len(encoder_vars):]

            if len(encoder_vars) > 0:
                opt_encoder.apply_gradients(zip(grads_encoder, encoder_vars))
            if len(heads_vars) > 0:
                opt_heads.apply_gradients(zip(grads_heads, heads_vars))

        print(f'End epoch {epoch+1}, loss: {float(loss_value):.6f}')


__all__ = [
    'SharedBackboneModel', 'quantile_loss',
    'train_phase1_pretrain', 'train_phase2_finetune_heads', 'train_phase3_endtoend'
]
