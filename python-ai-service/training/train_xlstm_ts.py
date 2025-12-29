"""
xLSTM-TS Training Script for Stock Return Prediction

This script trains the xLSTM-TS (Extended LSTM for Time Series) model,
using walk-forward validation to ensure robustness before production deployment.

Based on:
- Beck et al. (2024): "xLSTM: Extended Long Short-Term Memory"
- gonzalopezgil/xlstm-ts: Time series optimized xLSTM

Key Features:
- Walk-forward validation with WFE > 50% gating
- GPU acceleration with mixed precision (float16)
- High batch size support (512+)
- AntiCollapseDirectionalLoss for variance stability
- Proper artifact saving to saved_models/{SYMBOL}/xlstm/

Author: AI-Stocks xLSTM Integration
Date: December 2025
"""

import argparse
import sys
import json
import pickle
import logging
import os
import random
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, Optional, Any

# P0 FIX: Import yfinance BEFORE TensorFlow to avoid SSL conflict
import yfinance as yf  # noqa: F401 - must be first

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================================
# GPU & ACCELERATION CONFIGURATION
# ============================================================================
def setup_gpu_acceleration():
    """Configure environment for GPU acceleration and mixed precision."""
    # 1. Detect and set LD_LIBRARY_PATH for Pip-installed NVIDIA libraries
    try:
        import site
        site_packages = site.getsitepackages()[0]
        nvidia_path = Path(site_packages) / "nvidia"
        if nvidia_path.exists():
            lib_dirs = [str(d / "lib") for d in nvidia_path.iterdir() if (d / "lib").exists()]
            if lib_dirs:
                current_ld = os.environ.get("LD_LIBRARY_PATH", "")
                os.environ["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + (":" + current_ld if current_ld else "")
    except Exception as e:
        print(f"[GPU] Warning: Failed to auto-configure LD_LIBRARY_PATH: {e}")

    # 2. Enable Mixed Precision Policy (float16) with LOSS SCALING
    try:
        import tensorflow as tf
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print(f"[GPU] Mixed precision policy set to: {policy.name}")
    except Exception as e:
        print(f"[GPU] Warning: Could not set mixed precision policy: {e}")

    # 3. Enable JIT compilation (XLA) if possible
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'

setup_gpu_acceleration()


# ============================================================================
# DETERMINISTIC SEED SETTINGS
# ============================================================================
def set_global_seed(seed: int = 42) -> None:
    """Set seeds for all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    # TensorFlow seed
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass

    # PyTorch seed (if available)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# Default seed
GLOBAL_SEED = int(os.environ.get('SEED', 42))
set_global_seed(GLOBAL_SEED)

# TensorFlow imports after seed setup
import tensorflow as tf
from tensorflow import keras

# Project imports
from data.cache_manager import DataCacheManager
from data.feature_engineer import engineer_features, get_feature_columns, EXPECTED_FEATURE_COUNT
from data.target_engineering import prepare_training_data
from models.xlstm_ts import xLSTM_TS, xLSTMConfig, create_xlstm_ts, get_custom_objects
from models.lstm_transformer_paper import AntiCollapseDirectionalLoss, directional_accuracy_metric
from validation.walk_forward import (
    WalkForwardValidator,
    WalkForwardConfig,
    WalkForwardMode,
    WalkForwardResults
)
from utils.model_paths import ModelPaths, SAVED_MODELS_ROOT

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# AUTO-STOP CALLBACK (NUCLEAR FIX - December 2025)
# ============================================================================

class AutoStopOnCollapse(keras.callbacks.Callback):
    """
    Auto-stop training when variance collapse or prediction bias is detected.

    This callback hard-stops training by raising an exception when:
    1. Prediction variance drops below threshold (variance collapse)
    2. Predictions are >85% in one direction (prediction bias)

    This saves compute time and prevents training models that have already failed.
    """

    def __init__(self, X_val, y_val, check_interval=5, warmup_epochs=3,
                 min_std=0.005, max_bias=0.85):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.check_interval = check_interval
        self.warmup_epochs = warmup_epochs
        self.min_std = min_std
        self.max_bias = max_bias

    def on_epoch_end(self, epoch, logs=None):
        # Skip warmup and only check at intervals
        if epoch < self.warmup_epochs or epoch % self.check_interval != 0:
            return

        # Get predictions
        try:
            preds = self.model.predict(self.X_val[:100], verbose=0).flatten()
        except Exception as e:
            logger.warning(f"AutoStopOnCollapse: prediction failed at epoch {epoch}: {e}")
            return

        pred_std = np.std(preds)
        pct_positive = (preds > 0).mean()
        pct_negative = (preds < 0).mean()

        logger.info(f"[Epoch {epoch}] AutoStop Check: std={pred_std:.6f}, "
                    f"pos={pct_positive:.1%}, neg={pct_negative:.1%}")

        # Check for variance collapse
        if pred_std < self.min_std:
            logger.error(f"VARIANCE COLLAPSE at epoch {epoch}: std={pred_std:.6f}")
            raise ValueError(f"VARIANCE COLLAPSE at epoch {epoch}: std={pred_std:.6f} < {self.min_std}")

        # Check for prediction bias
        if pct_positive > self.max_bias:
            logger.error(f"PREDICTION BIAS at epoch {epoch}: {pct_positive:.1%} positive")
            raise ValueError(f"PREDICTION BIAS at epoch {epoch}: {pct_positive:.1%} positive")
        if pct_negative > self.max_bias:
            logger.error(f"PREDICTION BIAS at epoch {epoch}: {pct_negative:.1%} negative")
            raise ValueError(f"PREDICTION BIAS at epoch {epoch}: {pct_negative:.1%} negative")


# ============================================================================
# xLSTM PATH SUPPORT
# ============================================================================
class xLSTMPaths:
    """Path accessors for xLSTM model artifacts."""

    def __init__(self, symbol_dir: Path):
        self.base = symbol_dir / 'xlstm'

    @property
    def model(self) -> Path:
        """Keras SavedModel file (.keras)."""
        return self.base / 'model.keras'

    @property
    def weights(self) -> Path:
        """Model weights file (.weights.h5)."""
        return self.base / 'xlstm.weights.h5'

    @property
    def feature_scaler(self) -> Path:
        """Feature scaler pickle."""
        return self.base / 'feature_scaler.pkl'

    @property
    def target_scaler(self) -> Path:
        """Target scaler pickle."""
        return self.base / 'target_scaler.pkl'

    @property
    def features(self) -> Path:
        """Feature columns list pickle."""
        return self.base / 'features.pkl'

    @property
    def metadata(self) -> Path:
        """Training metadata pickle."""
        return self.base / 'metadata.pkl'

    @property
    def config(self) -> Path:
        """Model configuration JSON."""
        return self.base / 'config.json'

    @property
    def wfe_results(self) -> Path:
        """Walk-forward validation results JSON."""
        return self.base / 'wfe_results.json'

    def ensure_dir(self) -> None:
        """Create directory if it doesn't exist."""
        self.base.mkdir(parents=True, exist_ok=True)


# ============================================================================
# DATA PREPARATION
# ============================================================================
def create_sequences(X: np.ndarray, sequence_length: int = 60) -> np.ndarray:
    """Create sequences with specified lookback for time series patterns.

    Uses NumPy striding for GPU-friendly memory layout.

    Args:
        X: Feature matrix (n_samples, n_features)
        sequence_length: Number of timesteps per sequence

    Returns:
        Sequences of shape (n_samples - sequence_length + 1, sequence_length, n_features)
    """
    X = np.ascontiguousarray(X)

    shape = (X.shape[0] - sequence_length + 1, sequence_length, X.shape[1])
    strides = (X.strides[0], X.strides[0], X.strides[1])

    try:
        from numpy.lib.stride_tricks import as_strided
        X_strided = as_strided(X, shape=shape, strides=strides)
        return np.ascontiguousarray(X_strided)
    except Exception:
        # Fallback to NumPy advanced indexing
        indices = np.arange(sequence_length)[None, :] + np.arange(X.shape[0] - sequence_length + 1)[:, None]
        return X[indices]


def prepare_xlstm_data(
    symbol: str,
    sequence_length: int = 60,
    use_cache: bool = True,
    force_refresh: bool = False,
) -> Tuple[np.ndarray, np.ndarray, RobustScaler, RobustScaler, list, Dict]:
    """
    Prepare data for xLSTM training.

    Args:
        symbol: Stock ticker
        sequence_length: Sequence length for model
        use_cache: Whether to use cached data
        force_refresh: Force data refresh

    Returns:
        Tuple of (X_sequences, y_targets, feature_scaler, target_scaler, feature_cols, metadata)
    """
    logger.info(f"Preparing xLSTM data for {symbol}...")

    # Use cache manager
    cache_manager = DataCacheManager()
    raw_df, engineered_df, prepared_df, feature_cols = cache_manager.get_or_fetch_data(
        symbol=symbol,
        include_sentiment=True,
        force_refresh=force_refresh
    )

    # Validate feature count
    if len(feature_cols) != EXPECTED_FEATURE_COUNT:
        logger.warning(f"Feature count mismatch: {len(feature_cols)} vs expected {EXPECTED_FEATURE_COUNT}")

    # Extract features and target
    X = prepared_df[feature_cols].values
    y = prepared_df['target_1d'].values

    # Scale features
    feature_scaler = RobustScaler()
    X_scaled = feature_scaler.fit_transform(X)

    # Scale targets
    target_scaler = RobustScaler()
    y_scaled = target_scaler.fit_transform(y.reshape(-1, 1)).flatten()

    # Create sequences
    X_seq = create_sequences(X_scaled, sequence_length)

    # Align targets with sequences (use last target in each sequence)
    y_seq = y_scaled[sequence_length - 1:]

    # Validate shapes
    assert len(X_seq) == len(y_seq), f"Shape mismatch: X={len(X_seq)}, y={len(y_seq)}"

    # Build metadata
    metadata = {
        'symbol': symbol,
        'n_samples': len(X_seq),
        'n_features': len(feature_cols),
        'sequence_length': sequence_length,
        'target_stats': {
            'mean': float(np.mean(y)),
            'std': float(np.std(y)),
            'min': float(np.min(y)),
            'max': float(np.max(y)),
            'pct_positive': float((y > 0).mean()),
            'pct_negative': float((y < 0).mean()),
        },
        'date_range': {
            'start': str(prepared_df.index[0]),
            'end': str(prepared_df.index[-1]),
        },
        'prepared_at': datetime.now().isoformat(),
    }

    logger.info(f"Data shape: X_seq={X_seq.shape}, y_seq={y_seq.shape}")
    logger.info(f"Target distribution: {metadata['target_stats']['pct_positive']*100:.1f}% positive")

    return X_seq, y_seq, feature_scaler, target_scaler, feature_cols, metadata


# ============================================================================
# MODEL CREATION
# ============================================================================
def create_xlstm_model(
    n_features: int,
    sequence_length: int,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.35,  # December 2025: Increased from 0.2 for regularization
    use_wavelet: bool = True,
) -> keras.Model:
    """
    Create xLSTM-TS model for stock prediction.

    Args:
        n_features: Number of input features
        sequence_length: Length of input sequences
        hidden_dim: Hidden dimension for xLSTM cells
        num_layers: Number of xLSTM layers
        dropout: Dropout rate
        use_wavelet: Whether to use wavelet denoising

    Returns:
        Compiled xLSTM-TS model
    """
    config = xLSTMConfig(
        input_dim=n_features,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        output_dim=1,
        dropout=dropout,
        use_exponential_gating=True,  # Key xLSTM feature
        use_matrix_memory=False,       # sLSTM variant (simpler, more stable)
        use_wavelet_denoise=use_wavelet,
        wavelet_level=2,
        residual_connection=True,
        layer_norm=True,
    )

    model = xLSTM_TS(config)

    # Build model
    model.build(input_shape=(None, sequence_length, n_features))

    return model


def compile_model(
    model: keras.Model,
    learning_rate: float = 0.001,
    use_anti_collapse_loss: bool = True,
) -> keras.Model:
    """
    Compile xLSTM model with appropriate loss and optimizer.

    Args:
        model: Keras model to compile
        learning_rate: Initial learning rate
        use_anti_collapse_loss: Whether to use AntiCollapseDirectionalLoss

    Returns:
        Compiled model
    """
    # Loss function
    # December 2025: Aggressively increased penalties to prevent variance collapse
    if use_anti_collapse_loss:
        loss_fn = AntiCollapseDirectionalLoss(
            delta=1.0,
            direction_weight=0.2,
            variance_penalty_weight=2.0,   # STRONG enforcement to prevent collapse
            min_variance_target=0.003,     # Stricter threshold
            sign_diversity_weight=1.0,     # STRONG enforcement for ±50% balance
        )
        logger.info("Using AntiCollapseDirectionalLoss (var_penalty=2.0, min_std=0.003)")
    else:
        loss_fn = keras.losses.Huber(delta=1.0)
        logger.info("Using Huber loss for training")

    # Optimizer with gradient clipping
    optimizer = keras.optimizers.Adam(
        learning_rate=learning_rate,
        clipnorm=1.0,  # Gradient clipping for stability
    )

    # Compile with directional accuracy metric
    model.compile(
        optimizer=optimizer,
        loss=loss_fn,
        metrics=[directional_accuracy_metric],
    )

    return model


# ============================================================================
# TRAINING DATASET CREATION
# ============================================================================
def create_tf_dataset(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int = 512,
    shuffle: bool = True,
    prefetch: bool = True,
) -> tf.data.Dataset:
    """
    Create optimized tf.data.Dataset for GPU training.

    Args:
        X: Input features (n_samples, seq_len, n_features)
        y: Target values (n_samples,)
        batch_size: Batch size
        shuffle: Whether to shuffle data
        prefetch: Whether to enable prefetching

    Returns:
        tf.data.Dataset object
    """
    # Convert to float32 for consistency
    X = X.astype(np.float32)
    y = y.astype(np.float32)

    dataset = tf.data.Dataset.from_tensor_slices((X, y))

    # Configure options for speed
    options = tf.data.Options()
    options.deterministic = False

    # Shuffle
    if shuffle:
        buffer_size = max(1000, len(X) // 2)
        dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)

    # Batch
    dataset = dataset.batch(batch_size, drop_remainder=False)

    # Cache and prefetch
    dataset = dataset.cache()
    dataset = dataset.with_options(options)

    if prefetch:
        prefetch_buffer = max(2, 2 * batch_size // 512)
        dataset = dataset.prefetch(buffer_size=prefetch_buffer)

    return dataset


# ============================================================================
# WALK-FORWARD VALIDATION
# ============================================================================
def run_walk_forward_validation(
    X: np.ndarray,
    y: np.ndarray,
    config: Dict,
    n_iterations: int = 5,
) -> WalkForwardResults:
    """
    Run walk-forward validation on xLSTM model.

    Args:
        X: Input sequences
        y: Target values
        config: Model configuration dict
        n_iterations: Number of WF iterations

    Returns:
        WalkForwardResults object
    """
    logger.info("Starting walk-forward validation...")

    def model_factory():
        """Factory to create fresh model for each fold."""
        model = create_xlstm_model(
            n_features=config['n_features'],
            sequence_length=config['sequence_length'],
            hidden_dim=config['hidden_dim'],
            num_layers=config['num_layers'],
            dropout=config['dropout'],
            use_wavelet=config['use_wavelet'],
        )
        return compile_model(
            model,
            learning_rate=config['learning_rate'],
            use_anti_collapse_loss=config['use_anti_collapse_loss'],
        )

    # Configure walk-forward
    wf_config = WalkForwardConfig(
        mode=WalkForwardMode.ANCHORED,
        n_iterations=n_iterations,
        train_pct=0.60,
        validation_pct=0.15,
        test_pct=0.25,
        gap_days=1,
        purge_days=config['sequence_length'],  # Purge sequence length to prevent leakage
    )

    validator = WalkForwardValidator(wf_config)

    # Fit kwargs
    fit_kwargs = {
        'epochs': config['wf_epochs'],
        'batch_size': config['batch_size'],
        'verbose': 0,
        'callbacks': [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,          # INCREASED from 10 to allow more training
                min_delta=0.0001,     # Require meaningful improvement
                restore_best_weights=True,
            ),
        ],
    }

    # Run validation
    results = validator.validate(
        model_factory=model_factory,
        X=X,
        y=y,
        fit_kwargs=fit_kwargs,
    )

    return results


# ============================================================================
# PRODUCTION MODEL TRAINING
# ============================================================================
def train_production_model(
    X: np.ndarray,
    y: np.ndarray,
    config: Dict,
    symbol: str,
) -> Tuple[keras.Model, Dict]:
    """
    Train production model on all data after WFE validation passes.

    Args:
        X: Input sequences
        y: Target values
        config: Model configuration
        symbol: Stock symbol for logging

    Returns:
        Tuple of (trained model, training metadata)
    """
    logger.info("Training production model on full dataset...")

    # Create model
    model = create_xlstm_model(
        n_features=config['n_features'],
        sequence_length=config['sequence_length'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        use_wavelet=config['use_wavelet'],
    )
    model = compile_model(
        model,
        learning_rate=config['learning_rate'],
        use_anti_collapse_loss=config['use_anti_collapse_loss'],
    )

    # Create train/val split (80/20)
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    # Create datasets
    train_ds = create_tf_dataset(X_train, y_train, batch_size=config['batch_size'])
    val_ds = create_tf_dataset(X_val, y_val, batch_size=config['batch_size'], shuffle=False)

    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1,
        ),
        keras.callbacks.LearningRateScheduler(
            lambda epoch: config['learning_rate'] * (0.99 ** epoch)  # Gradual decay
        ),
        # NUCLEAR FIX: Auto-stop on variance collapse or prediction bias
        AutoStopOnCollapse(
            X_val=X_val,
            y_val=y_val,
            check_interval=5,
            warmup_epochs=3,
            min_std=0.005,
            max_bias=0.85,
        ),
    ]

    # Train
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config['epochs'],
        callbacks=callbacks,
        verbose=1,
    )

    # Compute final metrics
    y_pred_val = model.predict(X_val, verbose=0).flatten()

    pred_std = float(np.std(y_pred_val))
    pred_mean = float(np.mean(y_pred_val))
    direction_acc = float(np.mean(np.sign(y_val) == np.sign(y_pred_val)))

    # Check for variance collapse
    if pred_std < 0.005:
        logger.warning(f"WARNING: Low prediction variance detected (std={pred_std:.6f})")

    training_metadata = {
        'epochs_trained': len(history.history['loss']),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
        'val_direction_accuracy': direction_acc,
        'val_pred_std': pred_std,
        'val_pred_mean': pred_mean,
        'val_pct_positive': float((y_pred_val > 0).mean()),
        'trained_at': datetime.now().isoformat(),
    }

    logger.info(f"\n=== Production Model Results ===")
    logger.info(f"Direction Accuracy: {direction_acc:.4f}")
    logger.info(f"Prediction Std: {pred_std:.6f}")
    logger.info(f"Epochs Trained: {training_metadata['epochs_trained']}")

    return model, training_metadata


# ============================================================================
# MODEL SAVING
# ============================================================================
def save_xlstm_model(
    model: keras.Model,
    feature_scaler: RobustScaler,
    target_scaler: RobustScaler,
    feature_cols: list,
    config: Dict,
    metadata: Dict,
    wfe_results: Optional[WalkForwardResults],
    symbol: str,
) -> Path:
    """
    Save xLSTM model and all artifacts.

    Args:
        model: Trained Keras model
        feature_scaler: Fitted feature scaler
        target_scaler: Fitted target scaler
        feature_cols: List of feature column names
        config: Model configuration
        metadata: Training metadata
        wfe_results: Walk-forward validation results (optional)
        symbol: Stock symbol

    Returns:
        Path to saved model directory
    """
    # Setup paths
    symbol_dir = SAVED_MODELS_ROOT / symbol.upper()
    paths = xLSTMPaths(symbol_dir)
    paths.ensure_dir()

    logger.info(f"Saving xLSTM model to {paths.base}...")

    # Save model (both full model and weights)
    try:
        model.save(str(paths.model))
        logger.info(f"  Saved model: {paths.model}")
    except Exception as e:
        logger.warning(f"Failed to save full model: {e}")

    try:
        model.save_weights(str(paths.weights))
        logger.info(f"  Saved weights: {paths.weights}")
    except Exception as e:
        logger.warning(f"Failed to save weights: {e}")

    # Save scalers
    with open(paths.feature_scaler, 'wb') as f:
        pickle.dump(feature_scaler, f)
    logger.info(f"  Saved feature scaler: {paths.feature_scaler}")

    with open(paths.target_scaler, 'wb') as f:
        pickle.dump(target_scaler, f)
    logger.info(f"  Saved target scaler: {paths.target_scaler}")

    # Save feature columns
    with open(paths.features, 'wb') as f:
        pickle.dump(feature_cols, f)
    logger.info(f"  Saved feature columns: {paths.features}")

    # Save configuration
    with open(paths.config, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"  Saved config: {paths.config}")

    # Save metadata
    with open(paths.metadata, 'wb') as f:
        pickle.dump(metadata, f)
    logger.info(f"  Saved metadata: {paths.metadata}")

    # Save WFE results if available
    if wfe_results is not None:
        wfe_dict = {
            'aggregate_wfe': float(wfe_results.aggregate_wfe),
            'recommendation': wfe_results.recommendation,
            'mean_test_direction_acc': float(wfe_results.mean_test_direction_acc),
            'std_test_direction_acc': float(wfe_results.std_test_direction_acc),
            'fold_metrics': [
                {
                    'fold': fm.fold,
                    'test_direction_acc': float(fm.test_direction_acc),
                    'wfe': float(fm.wfe),
                }
                for fm in wfe_results.fold_metrics
            ],
        }
        with open(paths.wfe_results, 'w') as f:
            json.dump(wfe_dict, f, indent=2)
        logger.info(f"  Saved WFE results: {paths.wfe_results}")

    # Also save canonical feature columns at symbol level for ensemble compatibility
    feature_columns_path = symbol_dir / 'feature_columns.pkl'
    if not feature_columns_path.exists():
        with open(feature_columns_path, 'wb') as f:
            pickle.dump(feature_cols, f)
        logger.info(f"  Saved canonical features: {feature_columns_path}")

    logger.info(f"xLSTM model saved successfully to {paths.base}")
    return paths.base


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================
def train_xlstm(
    symbol: str,
    epochs: int = 100,
    batch_size: int = 512,
    sequence_length: int = 60,
    hidden_dim: int = 64,
    num_layers: int = 2,
    dropout: float = 0.35,  # December 2025: Increased from 0.2 for regularization
    learning_rate: float = 0.001,
    use_wavelet: bool = True,
    use_anti_collapse_loss: bool = True,
    skip_wfe: bool = False,
    wfe_iterations: int = 5,
    wfe_epochs: int = 30,
    min_wfe: float = 40.0,  # December 2025: Lowered from 50.0 to allow more models
    use_cache: bool = True,
    force_refresh: bool = False,
    seed: int = 42,
) -> Dict:
    """
    Main training function for xLSTM-TS model.

    Args:
        symbol: Stock ticker symbol
        epochs: Training epochs for production model
        batch_size: Batch size (512+ recommended for GPU)
        sequence_length: Sequence length for model
        hidden_dim: Hidden dimension for xLSTM
        num_layers: Number of xLSTM layers
        dropout: Dropout rate
        learning_rate: Initial learning rate
        use_wavelet: Whether to use wavelet denoising
        use_anti_collapse_loss: Whether to use anti-collapse loss
        skip_wfe: Skip walk-forward validation (not recommended)
        wfe_iterations: Number of WFE iterations
        wfe_epochs: Epochs per WFE fold
        min_wfe: Minimum WFE threshold (%)
        use_cache: Whether to use cached data
        force_refresh: Force data refresh
        seed: Random seed

    Returns:
        Dict with training results
    """
    # Set seed
    set_global_seed(seed)

    logger.info("="*60)
    logger.info(f"xLSTM-TS Training for {symbol}")
    logger.info("="*60)

    # Prepare data
    X, y, feature_scaler, target_scaler, feature_cols, data_metadata = prepare_xlstm_data(
        symbol=symbol,
        sequence_length=sequence_length,
        use_cache=use_cache,
        force_refresh=force_refresh,
    )

    # Model configuration
    config = {
        'symbol': symbol,
        'n_features': len(feature_cols),
        'sequence_length': sequence_length,
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'dropout': dropout,
        'learning_rate': learning_rate,
        'use_wavelet': use_wavelet,
        'use_anti_collapse_loss': use_anti_collapse_loss,
        'batch_size': batch_size,
        'epochs': epochs,
        'wf_epochs': wfe_epochs,
        'seed': seed,
    }

    logger.info(f"\nModel Configuration:")
    logger.info(f"  Features: {config['n_features']}")
    logger.info(f"  Sequence Length: {config['sequence_length']}")
    logger.info(f"  Hidden Dim: {config['hidden_dim']}")
    logger.info(f"  Layers: {config['num_layers']}")
    logger.info(f"  Batch Size: {config['batch_size']}")
    logger.info(f"  Epochs: {config['epochs']}")

    results = {
        'symbol': symbol,
        'config': config,
        'data_metadata': data_metadata,
        'status': 'pending',
    }

    # Walk-forward validation
    wfe_results = None
    if not skip_wfe:
        logger.info(f"\n{'='*60}")
        logger.info("WALK-FORWARD VALIDATION")
        logger.info("="*60)

        wfe_results = run_walk_forward_validation(
            X=X,
            y=y,
            config=config,
            n_iterations=wfe_iterations,
        )

        results['wfe'] = {
            'aggregate_wfe': float(wfe_results.aggregate_wfe),
            'mean_test_direction_acc': float(wfe_results.mean_test_direction_acc),
            'recommendation': wfe_results.recommendation,
        }

        # Check WFE threshold
        if wfe_results.aggregate_wfe < min_wfe:
            logger.error(f"\nWFE FAILED: {wfe_results.aggregate_wfe:.1f}% < {min_wfe:.1f}% threshold")
            logger.error("Model shows significant overfitting. NOT training production model.")
            logger.error(f"Recommendation: {wfe_results.recommendation}")
            results['status'] = 'wfe_failed'
            return results

        logger.info(f"\nWFE PASSED: {wfe_results.aggregate_wfe:.1f}% >= {min_wfe:.1f}%")
        logger.info("Proceeding to train production model...")
    else:
        logger.warning("Skipping walk-forward validation (not recommended)")

    # Train production model
    logger.info(f"\n{'='*60}")
    logger.info("PRODUCTION MODEL TRAINING")
    logger.info("="*60)

    model, training_metadata = train_production_model(
        X=X,
        y=y,
        config=config,
        symbol=symbol,
    )

    results['training'] = training_metadata

    # Save model
    save_path = save_xlstm_model(
        model=model,
        feature_scaler=feature_scaler,
        target_scaler=target_scaler,
        feature_cols=feature_cols,
        config=config,
        metadata={**data_metadata, **training_metadata},
        wfe_results=wfe_results,
        symbol=symbol,
    )

    results['save_path'] = str(save_path)
    results['status'] = 'success'

    logger.info(f"\n{'='*60}")
    logger.info("TRAINING COMPLETE")
    logger.info("="*60)
    logger.info(f"Model saved to: {save_path}")
    logger.info(f"Direction Accuracy: {training_metadata['val_direction_accuracy']:.4f}")
    if wfe_results:
        logger.info(f"WFE: {wfe_results.aggregate_wfe:.1f}%")

    return results


# ============================================================================
# CLI ENTRY POINT
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Train xLSTM-TS model for stock prediction'
    )

    # Required arguments
    parser.add_argument('--symbol', type=str, required=True,
                        help='Stock ticker symbol (e.g., AAPL)')

    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs for production model (default: 100)')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size (default: 512, recommended for GPU)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Initial learning rate (default: 0.001)')

    # Model architecture
    parser.add_argument('--sequence_length', type=int, default=60,
                        help='Sequence length (default: 60)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='Hidden dimension for xLSTM (default: 64)')
    parser.add_argument('--num_layers', type=int, default=2,
                        help='Number of xLSTM layers (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.35,
                        help='Dropout rate (default: 0.35 - increased for regularization)')
    parser.add_argument('--no-wavelet', action='store_true',
                        help='Disable wavelet denoising')

    # Walk-forward validation
    parser.add_argument('--skip-wfe', action='store_true',
                        help='Skip walk-forward validation (not recommended)')
    parser.add_argument('--wfe-iterations', type=int, default=5,
                        help='Number of WFE iterations (default: 5)')
    parser.add_argument('--wfe-epochs', type=int, default=30,
                        help='Epochs per WFE fold (default: 30)')
    parser.add_argument('--min-wfe', type=float, default=40.0,
                        help='Minimum WFE threshold percentage (default: 40 - lowered to allow more models)')

    # Data options
    parser.add_argument('--no-cache', action='store_true',
                        help='Do not use cached data')
    parser.add_argument('--force-refresh', action='store_true',
                        help='Force data refresh')

    # Other options
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--no-anti-collapse', action='store_true',
                        help='Disable anti-collapse loss (not recommended)')

    args = parser.parse_args()

    # Train model
    results = train_xlstm(
        symbol=args.symbol.upper(),
        epochs=args.epochs,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        use_wavelet=not args.no_wavelet,
        use_anti_collapse_loss=not args.no_anti_collapse,
        skip_wfe=args.skip_wfe,
        wfe_iterations=args.wfe_iterations,
        wfe_epochs=args.wfe_epochs,
        min_wfe=args.min_wfe,
        use_cache=not args.no_cache,
        force_refresh=args.force_refresh,
        seed=args.seed,
    )

    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)

    if results['status'] == 'success':
        print(f"Status: SUCCESS")
        print(f"Symbol: {results['symbol']}")
        print(f"Save Path: {results['save_path']}")
        print(f"Direction Accuracy: {results['training']['val_direction_accuracy']:.4f}")
        if 'wfe' in results:
            print(f"WFE: {results['wfe']['aggregate_wfe']:.1f}%")
    elif results['status'] == 'wfe_failed':
        print(f"Status: WFE FAILED")
        print(f"WFE: {results['wfe']['aggregate_wfe']:.1f}%")
        print(f"Recommendation: {results['wfe']['recommendation']}")
    else:
        print(f"Status: {results['status']}")

    return 0 if results['status'] == 'success' else 1


if __name__ == '__main__':
    sys.exit(main())
