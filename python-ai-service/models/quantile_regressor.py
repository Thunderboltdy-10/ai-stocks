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
from pathlib import Path

# Compatibility wrapper for register_keras_serializable across TF/Keras versions
try:
    # Newer TF may expose keras.saving.register_keras_serializable
    register_keras_serializable = keras.saving.register_keras_serializable  # type: ignore[attr-defined]
except Exception:
    try:
        # Fallback to tf.keras.utils.register_keras_serializable
        from tensorflow.keras.utils import register_keras_serializable  # type: ignore
    except Exception:
        try:
            # Fallback to standalone keras.utils (if installed separately)
            from keras.utils import register_keras_serializable  # type: ignore
        except Exception:
            # Last-resort no-op decorator to avoid import errors (will not register)
            def register_keras_serializable(package=None):
                def _decorator(x):
                    return x
                return _decorator


@register_keras_serializable(package="ai_stocks")
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

    def get_config(self):
        return {
            'quantiles': self.quantiles,
            'weights': self.weights,
            'name': self.name
        }


class TransformerBlock(layers.Layer):
    def __init__(self, d_model, num_heads, mlp_dim=None, dropout=0.1, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout_val = dropout
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

    def get_config(self):
        return {
            'd_model': self.d_model,
            'num_heads': self.num_heads,
            'mlp_dim': self.mlp_dim,
            'dropout': self.dropout_val,
        }


@register_keras_serializable(package="ai_stocks")
class QuantileRegressor(keras.Model):
    """LSTM + Transformer encoder with three quantile heads."""
    def __init__(self, n_features=123, sequence_length=90, d_model=128,
                 n_transformer_blocks=6, n_heads=4, dropout=0.2, lstm_units=64):
        super().__init__()
        self.n_features = n_features
        self.sequence_length = sequence_length
        self.d_model = d_model
        self.n_transformer_blocks = n_transformer_blocks
        # store additional hyperparams for saving/loading
        self.n_heads = n_heads
        self.dropout = dropout
        self.lstm_units = lstm_units

        # LSTM encoder (explicit name to help weight-by-name loading)
        self.lstm = layers.LSTM(lstm_units, return_sequences=True, name='lstm')

        # Project to d_model for transformer
        self.project = layers.Dense(d_model, name='project')

        # Transformer blocks
        self.transformer_blocks = [TransformerBlock(d_model, num_heads=n_heads, dropout=dropout)
                                   for _ in range(n_transformer_blocks)]

        # Pooling to get fixed-length vector
        self.pool = layers.GlobalAveragePooling1D()

        # Shared dense bottleneck
        self.shared_dense = layers.Dense(d_model, activation='relu', name='shared_dense')
        self.shared_dropout = layers.Dropout(dropout, name='shared_dropout')

        # Quantile heads
        def make_head(name_prefix: str):
            return keras.Sequential([
                layers.Dense(32, activation='relu', name=f'{name_prefix}_dense1'),
                layers.Dropout(dropout, name=f'{name_prefix}_dropout'),
                layers.Dense(1, name=f'{name_prefix}_out')
            ], name=f'{name_prefix}_head')

        self.q10_head = make_head('q10')
        self.q50_head = make_head('q50')
        self.q90_head = make_head('q90')

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
            'n_transformer_blocks': self.n_transformer_blocks,
            'n_heads': self.n_heads,
            'dropout': self.dropout,
            'lstm_units': self.lstm_units
        }

    def save(self, symbol: str, save_dir: str | Path = 'saved_models', feature_columns: list | None = None, metadata_extra: dict | None = None, use_new_naming: bool = False):
        """Save model weights, config JSON and metadata pickle for later loading.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            save_dir: Directory to save files to
            feature_columns: Optional list of feature column names
            metadata_extra: Optional dict of extra metadata to include
            use_new_naming: If True, use new naming convention (quantile.weights.h5, config.json, metadata.pkl)
                           If False, use legacy naming ({symbol}_quantile_regressor.weights.h5, etc.)
        
        Files written to `save_dir`:
        - Legacy: {symbol}_quantile_regressor.weights.h5, {symbol}_quantile_regressor_config.json, {symbol}_quantile_metadata.pkl
        - New: quantile.weights.h5, config.json, metadata.pkl
        """
        from pathlib import Path
        import json, pickle, time

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Determine file naming based on use_new_naming flag
        if use_new_naming:
            cfg_filename = 'config.json'
            weights_filename = 'quantile.weights.h5'
            weights_pkl_filename = 'quantile.weights.pkl'
            meta_filename = 'metadata.pkl'
        else:
            cfg_filename = f'{symbol}_quantile_regressor_config.json'
            weights_filename = f'{symbol}_quantile_regressor.weights.h5'
            weights_pkl_filename = f'{symbol}_quantile_regressor.weights.pkl'
            meta_filename = f'{symbol}_quantile_metadata.pkl'

        # Export model config
        cfg = self.get_config()
        cfg_path = save_dir / cfg_filename
        with open(cfg_path, 'w') as f:
            json.dump(cfg, f, indent=2)

        # Build an export wrapper (concatenated outputs) to save weights in the same format training used
        try:
            inp = keras.Input(shape=(self.sequence_length, self.n_features))
            outs = self(inp)
            concat = keras.layers.Concatenate(axis=-1)([outs['q10'], outs['q50'], outs['q90']])
            export_model = keras.Model(inputs=inp, outputs=concat)
            # Ensure weights are present by calling on dummy
            _ = export_model(tf.zeros((1, self.sequence_length, self.n_features)))
            h5_path = save_dir / weights_filename
            try:
                export_model.save_weights(str(h5_path), save_format='h5')
            except TypeError:
                export_model.save_weights(str(h5_path))
        except Exception:
            # Fallback: save base model weights via pickle (portable)
            h5_path = save_dir / weights_filename
            try:
                values = self.get_weights()
                with open(save_dir / weights_pkl_filename, 'wb') as bf:
                    pickle.dump({'values': values}, bf)
            except Exception:
                pass

        # Save metadata
        metadata = {
            'symbol': symbol,
            'saved_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'tf_version': tf.__version__,
            'keras_version': keras.__version__,
            'config_file': str(cfg_path),
            'weights_file': str(h5_path) if 'h5_path' in locals() else None,
        }
        if feature_columns is not None:
            metadata['feature_columns'] = feature_columns
        if metadata_extra is not None:
            metadata.update(metadata_extra)

        meta_path = save_dir / meta_filename
        with open(meta_path, 'wb') as f:
            pickle.dump(metadata, f)

        return {'config': str(cfg_path), 'weights': str(h5_path) if 'h5_path' in locals() else None, 'metadata': str(meta_path)}

    @classmethod
    def load(cls, symbol: str, save_dir: str | Path = 'saved_models', strict: bool = False):
        """Load a QuantileRegressor by reading config, rebuilding, and loading weights.

        Supports both new naming (config.json, quantile.weights.h5, metadata.pkl)
        and legacy naming ({symbol}_quantile_regressor_config.json, etc.).

        Returns an instance of `QuantileRegressor`. If weights are missing, returns
        a model with random initialization (and logs a warning).
        """
        from pathlib import Path
        import json, pickle

        save_dir = Path(save_dir)
        
        # Try new naming convention first, then legacy
        cfg_path_new = save_dir / 'config.json'
        cfg_path_legacy = save_dir / f'{symbol}_quantile_regressor_config.json'
        meta_path_new = save_dir / 'metadata.pkl'
        meta_path_legacy = save_dir / f'{symbol}_quantile_metadata.pkl'
        h5_path_new = save_dir / 'quantile.weights.h5'
        h5_path_legacy = save_dir / f'{symbol}_quantile_regressor.weights.h5'
        backup_pickle_new = save_dir / 'quantile.weights.pkl'
        backup_pickle_legacy = save_dir / f'{symbol}_quantile_regressor.weights.pkl'
        
        # Determine which paths exist
        cfg_path = cfg_path_new if cfg_path_new.exists() else cfg_path_legacy
        meta_path = meta_path_new if meta_path_new.exists() else meta_path_legacy
        h5_path = h5_path_new if h5_path_new.exists() else h5_path_legacy
        backup_pickle = backup_pickle_new if backup_pickle_new.exists() else backup_pickle_legacy

        cfg = None
        if cfg_path.exists():
            with open(cfg_path, 'r') as f:
                cfg = json.load(f)
        elif meta_path.exists():
            try:
                with open(meta_path, 'rb') as f:
                    meta = pickle.load(f)
                # meta may contain sequence_length/n_features
                cfg = {
                    'n_features': meta.get('n_features') or meta.get('n_features'),
                    'sequence_length': meta.get('sequence_length') or meta.get('sequence_length'),
                    'd_model': meta.get('d_model', 128),
                    'n_transformer_blocks': meta.get('n_transformer_blocks', 4),
                    'n_heads': meta.get('n_heads', 4),
                    'dropout': meta.get('dropout', 0.2),
                    'lstm_units': meta.get('lstm_units', 64)
                }
            except Exception:
                cfg = None

        if cfg is None:
            raise FileNotFoundError(f"QuantileRegressor config not found for {symbol} (tried {cfg_path_new}, {cfg_path_legacy}, {meta_path_new}, and {meta_path_legacy})")

        # Ensure integers
        cfg_clean = {
            'n_features': int(cfg.get('n_features', 123)),
            'sequence_length': int(cfg.get('sequence_length', 90)),
            'd_model': int(cfg.get('d_model', 128)),
            'n_transformer_blocks': int(cfg.get('n_transformer_blocks', 6)),
            'n_heads': int(cfg.get('n_heads', 4)),
            'dropout': float(cfg.get('dropout', 0.2)),
            'lstm_units': int(cfg.get('lstm_units', 64))
        }

        # Reconstruct model
        model = cls(
            n_features=cfg_clean['n_features'],
            sequence_length=cfg_clean['sequence_length'],
            d_model=cfg_clean['d_model'],
            n_transformer_blocks=cfg_clean['n_transformer_blocks'],
            n_heads=cfg_clean['n_heads'],
            dropout=cfg_clean['dropout'],
            lstm_units=cfg_clean['lstm_units']
        )

        # Build weights by calling on dummy input
        try:
            _ = model(tf.zeros((1, cfg_clean['sequence_length'], cfg_clean['n_features'])))
        except Exception:
            pass

        # Try loading weights: prefer pickle, then HDF5
        loaded = False
        if backup_pickle.exists():
            try:
                with open(backup_pickle, 'rb') as bf:
                    data = pickle.load(bf)
                vals = data.get('values') if isinstance(data, dict) else data
                model.set_weights(vals)
                loaded = True
            except Exception:
                loaded = False

        if not loaded and h5_path.exists():
            # Attempt by-name loading first (more tolerant)
            try:
                model.load_weights(str(h5_path), by_name=True, skip_mismatch=not strict)
                loaded = True
            except Exception:
                try:
                    # Try load into a wrapper model and transfer weights
                    inp = keras.Input(shape=(cfg_clean['sequence_length'], cfg_clean['n_features']))
                    outs = model(inp)
                    concat = keras.layers.Concatenate(axis=-1)([outs['q10'], outs['q50'], outs['q90']])
                    export_model = keras.Model(inputs=inp, outputs=concat)
                    export_model.load_weights(str(h5_path))
                    # Transfer weights where possible
                    try:
                        model.set_weights(export_model.get_weights())
                    except Exception:
                        pass
                    loaded = True
                except Exception:
                    loaded = False

        if not loaded:
            msg = f"Warning: Could not load weights for QuantileRegressor {symbol}. Model initialized with random weights."
            if strict:
                raise RuntimeError(msg)
            else:
                print(msg)

        return model


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
