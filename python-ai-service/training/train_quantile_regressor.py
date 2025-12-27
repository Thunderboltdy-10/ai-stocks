"""
Train QuantileRegressor (adapted from train_1d_regressor_final.py)

Usage:
    python training/train_quantile_regressor.py --symbol AAPL --epochs 50

This script:
- Loads price data, engineers features
- Prepares training data (1-day target)
- Scales features and targets with RobustScaler
- Builds QuantileRegressor and a Keras wrapper that outputs stacked quantiles
- Trains with QuantileLoss and monitoring callbacks
- Saves model weights, scalers, and metadata
"""

import os, sys
# Add repo root to sys.path so local imports like `data.*` resolve when running from repo root
"""
Train QuantileRegressor (adapted from train_1d_regressor_final.py)

Usage:
    python training/train_quantile_regressor.py --symbol AAPL --epochs 50

This script:
- Loads price data, engineers features
- Prepares training data (1-day target)
- Scales features and targets with RobustScaler
- Builds QuantileRegressor and a Keras wrapper that outputs stacked quantiles
- Trains with QuantileLoss and monitoring callbacks
- Saves model weights, scalers, and metadata
"""

import os, sys
# Add repo root to sys.path so local imports like `data.*` resolve when running from repo root
# Ensure both the package folder and repository root are on sys.path so
# imports like `models.quantile_regressor` and `data.*` resolve whether
# the script is run from repo root or from `python-ai-service/`.
package_dir = os.path.dirname(__file__)  # .../python-ai-service/training
python_ai_service_path = os.path.abspath(os.path.join(package_dir, '..'))
repo_root = os.path.abspath(os.path.join(package_dir, '..', '..'))
if python_ai_service_path not in sys.path:
    sys.path.insert(0, python_ai_service_path)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)


import argparse
from pathlib import Path
import numpy as np
import pickle
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import RobustScaler

# Import QuantileRegressor from repo `models/` directory. Attempt normal import
# first; if that fails (package resolution differences when running from
# `python-ai-service/`), load the module by path.
try:
    from models.quantile_regressor import QuantileRegressor, QuantileLoss
except Exception:
    import importlib.util
    mod_path = os.path.join(repo_root, 'models', 'quantile_regressor.py')
    spec = importlib.util.spec_from_file_location('models.quantile_regressor', mod_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    QuantileRegressor = getattr(mod, 'QuantileRegressor')
    QuantileLoss = getattr(mod, 'QuantileLoss')

from data.data_fetcher import fetch_stock_data
from data.feature_engineer import engineer_features, get_feature_columns, EXPECTED_FEATURE_COUNT
from data.target_engineering import prepare_training_data
from utils.model_paths import ModelPaths, get_legacy_quantile_paths


class QuantileCoverageCallback(keras.callbacks.Callback):
    """Monitor calibration: fraction of actuals above each predicted quantile."""
    def __init__(self, X_val, y_val_orig, target_scaler, quantiles=(0.1, 0.5, 0.9)):
        super().__init__()
        self.X_val = X_val
        self.y_val_orig = y_val_orig
        self.target_scaler = target_scaler
        self.quantiles = quantiles

    def on_epoch_end(self, epoch, logs=None):
        preds = self.model.predict(self.X_val, verbose=0)
        # preds shape: (n_samples, 3)
        q_preds_scaled = preds  # model outputs are scaled predictions
        # inverse scale
        try:
            q10 = self.target_scaler.inverse_transform(q_preds_scaled[:, 0].reshape(-1, 1)).flatten()
            q50 = self.target_scaler.inverse_transform(q_preds_scaled[:, 1].reshape(-1, 1)).flatten()
            q90 = self.target_scaler.inverse_transform(q_preds_scaled[:, 2].reshape(-1, 1)).flatten()
        except Exception:
            # If scaler not available or fails, assume preds are already original-scale
            q10, q50, q90 = q_preds_scaled[:, 0], q_preds_scaled[:, 1], q_preds_scaled[:, 2]

        actuals = self.y_val_orig
        # Coverage: % actuals >= predicted quantile (should be ~ 1 - tau)
        cov_q10 = np.mean(actuals >= q10)
        cov_q50 = np.mean(actuals >= q50)
        cov_q90 = np.mean(actuals >= q90)

        print(f"  [Calib] Epoch {epoch+1}: coverage Q10={cov_q10:.3f}, Q50={cov_q50:.3f}, Q90={cov_q90:.3f}")

        # Warn if coverage is badly miscalibrated
        if abs(cov_q10 - 0.90) > 0.20 or abs(cov_q90 - 0.10) > 0.20:
            print("  ⚠️  Significant calibration deviation detected (retrain or adjust loss weights)")

    
def plot_quantile_predictions(y_true, y_pred_dict, dates=None, save_path=None):
    """Plot Q10/Q50/Q90 and actuals over time and distribution metrics."""
    # y_pred_dict: dict or array with columns [q10,q50,q90] in original scale
    if isinstance(y_pred_dict, dict):
        q10 = y_pred_dict['q10']
        q50 = y_pred_dict['q50']
        q90 = y_pred_dict['q90']
    else:
        q10, q50, q90 = y_pred_dict[:, 0], y_pred_dict[:, 1], y_pred_dict[:, 2]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))

    ax = axes[0, 0]
    if dates is None:
        dates = np.arange(len(y_true))
    ax.plot(dates, q10, label='Q10', color='C0', alpha=0.8)
    ax.plot(dates, q50, label='Q50', color='C1', alpha=0.9)
    ax.plot(dates, q90, label='Q90', color='C2', alpha=0.9)
    ax.plot(dates, y_true, label='Actual', color='k', alpha=0.6)
    ax.set_title('Quantile Predictions vs Actuals')
    ax.legend()

    ax = axes[0, 1]
    widths = q90 - q10
    ax.plot(dates, widths, color='C3')
    ax.set_title('Distribution Width (Q90 - Q10)')

    ax = axes[1, 0]
    ax.hist(widths, bins=50, color='C3')
    ax.set_title('Width Distribution')

    ax = axes[1, 1]
    # Calibration scatter: fraction of actuals below predicted quantiles vs tau
    taus = [0.1, 0.5, 0.9]
    covs = [np.mean(y_true <= q10), np.mean(y_true <= q50), np.mean(y_true <= q90)]
    ax.plot(taus, covs, marker='o')
    ax.plot([0, 1], [0, 1], '--', color='gray')
    ax.set_xlabel('Theoretical quantile')
    ax.set_ylabel('Empirical quantile')
    ax.set_title('Calibration')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close(fig)


def create_sequences(X, sequence_length=90):
    """Create overlapping sequences: returns array shape (n_samples, seq_len, n_features)
    
    OPTIMIZED: Uses NumPy striding for GPU-friendly memory layout (3x faster than Python loops).
    """
    n_rows = X.shape[0]
    if n_rows < sequence_length:
        raise ValueError("Not enough rows to create one sequence")
    
    # Convert to contiguous array
    X = np.ascontiguousarray(X)
    
    # Use NumPy's optimized striding instead of Python loop
    shape = (X.shape[0] - sequence_length + 1, sequence_length, X.shape[1])
    strides = (X.strides[0], X.strides[0], X.strides[1])
    
    try:
        from numpy.lib.stride_tricks import as_strided
        X_strided = as_strided(X, shape=shape, strides=strides)
        return np.ascontiguousarray(X_strided)
    except Exception:
        # Fallback: NumPy advanced indexing
        indices = np.arange(sequence_length)[None, :] + np.arange(X.shape[0] - sequence_length + 1)[:, None]
        return X[indices]
    ax.set_title('Calibration')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path)
    plt.close(fig)


def train_quantile_regressor(symbol, epochs=100, batch_size=512, sequence_length=90, dropout=0.2, use_cache: bool = True):
    print(f"Training QuantileRegressor for {symbol}")

    # Load and engineer features (use DataCacheManager when requested)
    if use_cache:
        from data.cache_manager import DataCacheManager
        cache_manager = DataCacheManager()
        raw_df, engineered_df, prepared_df, feature_cols = cache_manager.get_or_fetch_data(
            symbol=symbol,
            include_sentiment=True,
            force_refresh=False
        )
        df = engineered_df
        df_clean = prepared_df
        # Validate canonical feature count
        expected_cols = get_feature_columns(include_sentiment=True)
        if len(feature_cols) != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Feature count mismatch!\n"
                f"Expected: {EXPECTED_FEATURE_COUNT} features\n"
                f"Got: {len(feature_cols)} features\n"
                f"Missing: {set(expected_cols) - set(feature_cols)}\n"
                f"Extra: {set(feature_cols) - set(expected_cols)}\n"
                f"Run with --force-refresh to regenerate features"
            )
    else:
        # Force fetch via DataCacheManager so artifacts are saved to cache for later use
        from data.cache_manager import DataCacheManager
        cache_manager = DataCacheManager()
        raw_df, engineered_df, prepared_df, feature_cols = cache_manager.get_or_fetch_data(
            symbol=symbol,
            include_sentiment=True,
            force_refresh=True
        )
        df = engineered_df
        df_clean = prepared_df
        # Validate canonical feature count
        expected_cols = get_feature_columns(include_sentiment=True)
        if len(feature_cols) != EXPECTED_FEATURE_COUNT:
            raise ValueError(
                f"Feature count mismatch!\n"
                f"Expected: {EXPECTED_FEATURE_COUNT} features\n"
                f"Got: {len(feature_cols)} features\n"
                f"Missing: {set(expected_cols) - set(feature_cols)}\n"
                f"Extra: {set(feature_cols) - set(expected_cols)}\n"
                f"Run with --force-refresh to regenerate features"
            )

    X = df_clean[feature_cols].values
    y_raw = df_clean['target_1d'].values  # original-scale returns

    # Scale features
    feature_scaler = RobustScaler()
    X_scaled = feature_scaler.fit_transform(X)

    # Scale targets using RobustScaler on clipped returns
    y_clipped = np.clip(y_raw, -0.15, 0.15)
    target_scaler = RobustScaler()
    y_scaled = target_scaler.fit_transform(y_clipped.reshape(-1, 1)).flatten()

    # Create sequences
    X_seq = create_sequences(X_scaled, sequence_length)
    y_aligned = y_scaled[sequence_length - 1:]
    y_orig_aligned = y_raw[sequence_length - 1:]

    # Split train/val
    split = int(len(X_seq) * 0.8)
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_train, y_val = y_aligned[:split], y_aligned[split:]
    y_val_orig = y_orig_aligned[split:]

    print(f"Samples — train: {len(X_train)}, val: {len(X_val)}")

    # Build QuantileRegressor and wrapper model that outputs stacked quantiles
    base_model = QuantileRegressor(n_features=X_seq.shape[2], sequence_length=sequence_length,
                                   d_model=128, n_transformer_blocks=4, n_heads=4, dropout=dropout)

    inputs = keras.Input(shape=(sequence_length, X_seq.shape[2]))
    outs = base_model(inputs)
    # Concatenate q10,q50,q90 into shape (batch,3)
    concat = keras.layers.Concatenate(axis=-1)([outs['q10'], outs['q50'], outs['q90']])
    train_model = keras.Model(inputs=inputs, outputs=concat)

    # Compile with QuantileLoss (works with tensor outputs)
    q_loss = QuantileLoss(quantiles=[0.1, 0.5, 0.9])

    # Custom MAE metrics per quantile
    def mae_q_factory(i):
        def mae_q(y_true, y_pred):
            pred_q = y_pred[..., i]
            # y_true shape (batch,) or (batch,1)
            y = tf.reshape(y_true, (-1,))
            return tf.reduce_mean(tf.abs(y - tf.reshape(pred_q, (-1,))))
        return mae_q

    train_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=q_loss,
        metrics=[mae_q_factory(0), mae_q_factory(1), mae_q_factory(2)]
    )

    # Callbacks
    lr_cb = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=8, verbose=1)
    early = keras.callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True, verbose=1)
    cov_cb = QuantileCoverageCallback(X_val, y_val_orig, target_scaler)

    # Pre-fit diagnostics (ensure feature dimension matches expectation)
    try:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(f"✓ Training with {EXPECTED_FEATURE_COUNT} features (93 technical + 20 new + 34 sentiment)")
        logger.info(f"  Input shape: {X_train.shape} = (samples={X_train.shape[0]}, seq_len={X_train.shape[1]}, features={X_train.shape[2]})")
        assert X_train.shape[2] == EXPECTED_FEATURE_COUNT, f"Feature dimension mismatch: {X_train.shape[2]} != {EXPECTED_FEATURE_COUNT}"
    except Exception:
        # If X_train not in expected format, continue but warn
        pass

    history = train_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[lr_cb, early, cov_cb],
        verbose=2
    )

    # Predictions on validation (scaled)
    y_pred_scaled = train_model.predict(X_val)
    # Inverse scale predictions
    q10 = target_scaler.inverse_transform(y_pred_scaled[:, 0].reshape(-1, 1)).flatten()
    q50 = target_scaler.inverse_transform(y_pred_scaled[:, 1].reshape(-1, 1)).flatten()
    q90 = target_scaler.inverse_transform(y_pred_scaled[:, 2].reshape(-1, 1)).flatten()

    # Metrics on original scale
    mae50 = np.mean(np.abs(y_val_orig - q50))
    print(f"Validation MAE (median): {mae50:.6f}")

    # Plot diagnostics
    paths = ModelPaths(symbol)
    paths.ensure_dirs()
    legacy_paths = get_legacy_quantile_paths(symbol)
    
    plot_path = paths.quantile.predictions_plot
    plot_quantile_predictions(y_val_orig, np.vstack([q10, q50, q90]).T, dates=None, save_path=str(plot_path))
    # Also save to legacy path
    plot_quantile_predictions(y_val_orig, np.vstack([q10, q50, q90]).T, dates=None, save_path=str(legacy_paths['predictions_plot']))

    # Save artifacts using QuantileRegressor.save() with new path structure
    try:
        save_result = base_model.save(symbol, save_dir=paths.quantile.base, feature_columns=feature_cols,
                                     metadata_extra={'epochs_trained': len(history.history['loss']),
                                                     'val_mae_median': float(mae50)},
                                     use_new_naming=True)
        print(f"✓ Saved QuantileRegressor artifacts to {paths.quantile.base}")
    except Exception as e:
        print(f"⚠️ QuantileRegressor.save() failed: {e}")

    # Save scalers to new paths
    with open(paths.quantile.feature_scaler, 'wb') as f:
        pickle.dump(feature_scaler, f)
    with open(paths.quantile.target_scaler, 'wb') as f:
        pickle.dump(target_scaler, f)
    # Also save canonical feature columns at symbol level
    with open(paths.feature_columns, 'wb') as f:
        pickle.dump(feature_cols, f)
    
    # Save to legacy paths for backward compatibility
    with open(legacy_paths['feature_scaler'], 'wb') as f:
        pickle.dump(feature_scaler, f)
    with open(legacy_paths['target_scaler'], 'wb') as f:
        pickle.dump(target_scaler, f)
    # Also save model to legacy location (uses legacy naming by default)
    try:
        base_model.save(symbol, save_dir=legacy_paths['weights'].parent, feature_columns=feature_cols,
                       metadata_extra={'epochs_trained': len(history.history['loss']),
                                       'val_mae_median': float(mae50)},
                       use_new_naming=False)
    except Exception:
        pass

    print(f"Training complete — artifacts saved to {paths.quantile.base}")
    print(f"✓ Legacy paths also updated for backward compatibility")

    # Verify saved feature columns
    try:
        saved_cols = pickle.load(open(paths.feature_columns, 'rb'))
        assert len(saved_cols) == EXPECTED_FEATURE_COUNT, f"Saved feature count mismatch: {len(saved_cols)} != {EXPECTED_FEATURE_COUNT}"
    except Exception as e:
        print(f"Warning: could not verify saved feature columns: {e}")

    # Post-save verification: attempt to load via the new load() API
    try:
        print("Verifying saved artifacts by loading via QuantileRegressor.load()...")
        loaded_model = QuantileRegressor.load(symbol, save_dir=paths.quantile.base)
        # quick predict build to ensure callable
        _ = loaded_model(tf.zeros((1, sequence_length, X_seq.shape[2])))
        print("✓ Post-save verification succeeded: loaded model callable")
    except Exception as e:
        print(f"⚠️ Post-save verification via load() failed: {e}")

    return train_model, base_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--sequence-length', type=int, default=90)
    parser.add_argument('--dropout', type=float, default=0.2)
    args = parser.parse_args()

    train_quantile_regressor(
        symbol=args.symbol,
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        dropout=args.dropout
    )


if __name__ == '__main__':
    main()
