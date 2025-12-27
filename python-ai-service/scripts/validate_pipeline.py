"""
Full Pipeline Validation for SOTA Financial Time Series Prediction

This script validates the complete redesigned system:
1. Data pipeline with look-ahead bias fixes
2. PatchTST model architecture
3. Training with batch_size >= 512 on GPU
4. Backtesting with position sizing
5. Performance metrics comparison

Target: Recover 15-20% annualized returns
"""

from __future__ import annotations

import os
import sys
import warnings
import logging
import gc
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd

# Set Keras backend before imports - use TensorFlow for stable memory management
# If TensorFlow is not available, fall back to PyTorch
try:
    import tensorflow as tf
    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    BACKEND = 'tensorflow'
except ImportError:
    os.environ['KERAS_BACKEND'] = 'torch'
    BACKEND = 'torch'

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')

# Imports
import keras
from keras import ops

from data.data_fetcher import fetch_stock_data
from data.feature_engineer import engineer_features
from models.patchtst import create_patchtst_model, PatchTST
from models.lstm_transformer_paper import (
    AntiCollapseDirectionalLoss,
    DirectionalHuberLoss,
    LSTMTransformerPaper,
    create_paper_model,
)

# Backend-specific GPU configuration
if BACKEND == 'tensorflow':
    # Configure TensorFlow memory growth to prevent OOM
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        for device in physical_devices:
            try:
                tf.config.experimental.set_memory_growth(device, True)
            except RuntimeError:
                pass
        logger.info(f"TensorFlow GPU devices: {[d.name for d in physical_devices]}")
    else:
        logger.warning("No GPU detected - training will be slow")
else:
    import torch
    if torch.cuda.is_available():
        logger.info(f"PyTorch GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        torch.cuda.empty_cache()
    else:
        logger.warning("No GPU detected - training will be slow")


def prepare_data(
    symbol: str = 'AAPL',
    seq_len: int = 60,
    include_sentiment: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Prepare train/val/test data."""
    import pickle
    from pathlib import Path
    
    logger.info(f"Preparing data for {symbol}...")
    
    # Cache is at root level, not inside python-ai-service
    cache_dir = Path(__file__).parent.parent.parent / 'cache' / symbol
    engineered_cache = cache_dir / 'engineered_features.pkl'
    raw_cache = cache_dir / 'raw_data.pkl'
    
    # Try to load engineered features from cache first
    df = None
    if engineered_cache.exists():
        logger.info(f"Loading cached engineered features from {engineered_cache}")
        try:
            with open(engineered_cache, 'rb') as f:
                df = pickle.load(f)
            logger.info(f"Loaded engineered features: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            logger.warning(f"Engineered cache load failed: {e}")
            df = None
    
    # If no cache, fetch fresh or from raw cache
    if df is None:
        try:
            raw_df = fetch_stock_data(symbol)
        except ValueError as e:
            if raw_cache.exists():
                logger.info(f"API failed, loading from raw data cache: {raw_cache}")
                with open(raw_cache, 'rb') as f:
                    raw_df = pickle.load(f)
            else:
                raise e
        df = engineer_features(raw_df, symbol=symbol, include_sentiment=include_sentiment)
    
    # Create target
    df['target'] = df['Close'].pct_change().shift(-1)
    df = df.dropna(subset=['target'])
    
    # Get feature columns
    exclude_cols = ['target', 'Date', 'date', 'Symbol', 'symbol', 
                    'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close',
                    'Dividends', 'Stock Splits']
    feature_cols = [c for c in df.columns 
                    if c not in exclude_cols 
                    and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    logger.info(f"Features: {len(feature_cols)}")
    
    # Convert to arrays
    X = df[feature_cols].values.astype(np.float32)
    y = df['target'].values.astype(np.float32)
    
    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Create sequences
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X)):
        X_seq.append(X[i-seq_len:i])
        y_seq.append(y[i])
    
    X_seq = np.array(X_seq, dtype=np.float32)
    y_seq = np.array(y_seq, dtype=np.float32).reshape(-1, 1)
    
    # Temporal split: 70% train, 15% val, 15% test
    n = len(X_seq)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)
    
    X_train = X_seq[:train_end]
    y_train = y_seq[:train_end]
    X_val = X_seq[train_end:val_end]
    y_val = y_seq[train_end:val_end]
    X_test = X_seq[val_end:]
    y_test = y_seq[val_end:]
    
    # Normalize
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std
    
    logger.info(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # Return test dates for backtesting
    test_dates = df.index[val_end + seq_len:] if hasattr(df.index, '__iter__') else None
    
    return X_train, y_train, X_val, y_val, X_test, y_test, df


def train_patchtst(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_features: int,
    seq_len: int = 60,
    epochs: int = 50,
    batch_size: int = 512,
    save_path: Optional[str] = None,
) -> Tuple[keras.Model, dict]:
    """Train PatchTST model."""
    
    # Clear session before training to prevent memory accumulation
    keras.backend.clear_session()
    gc.collect()
    
    # Backend-specific cleanup
    if BACKEND == 'torch':
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    logger.info("Training PatchTST model...")
    logger.info(f"  Epochs: {epochs}, Batch size: {batch_size}")
    
    # Create model
    model = PatchTST(
        seq_len=seq_len,
        pred_len=1,
        n_features=n_features,
        patch_len=16,
        stride=8,
        d_model=96,
        n_heads=4,
        n_layers=3,
        d_ff=384,
        dropout=0.15,
        use_revin=True,
    )
    
    # Build
    model.build(input_shape=(None, seq_len, n_features))
    
    # Compile with anti-collapse loss
    loss_fn = AntiCollapseDirectionalLoss(
        delta=1.0,
        direction_weight=0.2,
        variance_penalty_weight=0.3,
        min_variance_target=0.008,
        sign_diversity_weight=0.15,
    )
    
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=5e-4,
            weight_decay=0.01,
        ),
        loss=loss_fn,
        metrics=['mae'],
    )
    
    logger.info(f"Model parameters: {model.count_params():,}")
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
        ),
    ]
    
    # Add model checkpoint if save_path provided
    if save_path is not None:
        save_dir = Path(save_path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=save_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False,
                verbose=1,
            )
        )
    
    # Custom callback to clear session periodically and prevent memory leaks
    class MemoryCleanupCallback(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            # Force garbage collection every 5 epochs
            if (epoch + 1) % 5 == 0:
                gc.collect()
    
    callbacks.append(MemoryCleanupCallback())
    
    # Train
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    
    # Final garbage collection
    gc.collect()
    
    return model, history.history


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Dict[str, float]:
    """Compute evaluation metrics."""
    
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Directional accuracy
    correct = np.sum(np.sign(y_pred) == np.sign(y_true))
    directional_acc = correct / len(y_true)
    
    # Correlation
    correlation = np.corrcoef(y_pred, y_true)[0, 1]
    if np.isnan(correlation):
        correlation = 0.0
    
    # MAE
    mae = np.mean(np.abs(y_pred - y_true))
    
    # Prediction variance
    pred_std = np.std(y_pred)
    
    # Simple backtest
    positions = np.sign(y_pred)
    daily_returns = positions * y_true
    
    cumulative_return = np.sum(daily_returns)
    sharpe = np.mean(daily_returns) / (np.std(daily_returns) + 1e-8) * np.sqrt(252)
    
    # Max drawdown
    equity = np.cumsum(daily_returns)
    peak = np.maximum.accumulate(equity)
    drawdown = peak - equity
    max_drawdown = np.max(drawdown)
    
    # Win rate
    wins = np.sum(daily_returns > 0)
    win_rate = wins / len(daily_returns)
    
    # Annualized return (assuming ~252 trading days)
    n_years = len(daily_returns) / 252
    annualized_return = cumulative_return / n_years if n_years > 0 else cumulative_return
    
    return {
        'directional_accuracy': directional_acc,
        'correlation': correlation,
        'mae': mae,
        'pred_std': pred_std,
        'cumulative_return': cumulative_return,
        'annualized_return': annualized_return,
        'sharpe_ratio': sharpe,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
    }


def run_validation(
    symbol: str = 'AAPL',
    epochs: int = 50,
    batch_size: int = 512,
    load_model_path: Optional[str] = None,
    save_model_path: Optional[str] = None,
    output_dir: str = 'validation_results',
) -> Dict[str, Any]:
    """Run full pipeline validation.
    
    Args:
        symbol: Stock symbol to validate
        epochs: Training epochs
        batch_size: Batch size for training
        load_model_path: Path to load pre-trained model (skip training)
        save_model_path: Path to save trained model
        output_dir: Directory for saving validation results
    """
    
    # Clear session at start
    keras.backend.clear_session()
    gc.collect()
    
    # Backend-specific cleanup
    if BACKEND == 'torch':
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print("\n" + "=" * 70)
    print("FULL PIPELINE VALIDATION - SOTA Financial Time Series Prediction")
    print("=" * 70)
    print(f"Symbol: {symbol}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    if load_model_path:
        print(f"Loading model from: {load_model_path}")
    print("=" * 70 + "\n")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test, df = prepare_data(
        symbol=symbol,
        seq_len=60,
        include_sentiment=False,
    )
    
    n_features = X_train.shape[2]
    
    # Either load or train model
    if load_model_path and Path(load_model_path).exists():
        print(f"Loading pre-trained model from {load_model_path}...")
        model = load_model(load_model_path)
        history = {'loss': [], 'val_loss': []}  # No training history when loading
    else:
        # Determine save path
        if save_model_path is None:
            save_model_path = str(output_path / f'{symbol}_model.keras')
        
        # Train model
        model, history = train_patchtst(
            X_train, y_train,
            X_val, y_val,
            n_features=n_features,
            epochs=epochs,
            batch_size=batch_size,
            save_path=save_model_path,
        )
        print(f"\nModel saved to: {save_model_path}")
    
    # Evaluate on test set
    print("\n" + "-" * 50)
    print("TEST SET EVALUATION")
    print("-" * 50)
    
    y_pred = model.predict(X_test, batch_size=batch_size, verbose=0)
    metrics = compute_metrics(y_test, y_pred)
    
    for name, value in metrics.items():
        if isinstance(value, float):
            if 'return' in name or 'drawdown' in name:
                print(f"  {name}: {value:.2%}")
            elif 'accuracy' in name or 'rate' in name:
                print(f"  {name}: {value:.2%}")
            else:
                print(f"  {name}: {value:.4f}")
    
    # Validation metrics
    print("\n" + "-" * 50)
    print("VALIDATION SET METRICS")
    print("-" * 50)
    
    y_pred_val = model.predict(X_val, batch_size=batch_size, verbose=0)
    val_metrics = compute_metrics(y_val, y_pred_val)
    
    for name, value in val_metrics.items():
        if isinstance(value, float):
            if 'return' in name or 'drawdown' in name:
                print(f"  {name}: {value:.2%}")
            elif 'accuracy' in name or 'rate' in name:
                print(f"  {name}: {value:.2%}")
            else:
                print(f"  {name}: {value:.4f}")
    
    # Compare predictions variance
    print("\n" + "-" * 50)
    print("VARIANCE COLLAPSE CHECK")
    print("-" * 50)
    print(f"  Prediction std (test): {metrics['pred_std']:.6f}")
    print(f"  Target std (test): {np.std(y_test):.6f}")
    print(f"  Ratio: {metrics['pred_std'] / np.std(y_test):.4f}")
    
    if metrics['pred_std'] < 0.005:
        print("  ⚠️ WARNING: Possible variance collapse!")
    else:
        print("  ✓ Predictions have healthy variance")
    
    # Training history (only show if we actually trained)
    if history and len(history.get('loss', [])) > 0:
        print("\n" + "-" * 50)
        print("TRAINING HISTORY")
        print("-" * 50)
        print(f"  Final train loss: {history['loss'][-1]:.6f}")
        print(f"  Final val loss: {history['val_loss'][-1]:.6f}")
        print(f"  Best val loss: {min(history['val_loss']):.6f}")
        print(f"  Epochs trained: {len(history['loss'])}")
    else:
        print("\n" + "-" * 50)
        print("TRAINING HISTORY")
        print("-" * 50)
        print("  (Loaded pre-trained model, no training history)")
    
    # Summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    
    target_met = metrics['annualized_return'] >= 0.15  # 15%
    
    print(f"  Target: 15-20% annualized return")
    print(f"  Achieved: {metrics['annualized_return']:.2%}")
    print(f"  Status: {'✓ TARGET MET' if target_met else '✗ TARGET NOT MET'}")
    print("=" * 70)
    
    # Save validation results
    results = {
        'model': model,
        'test_metrics': metrics,
        'val_metrics': val_metrics,
        'history': history,
    }
    
    # Save metrics to JSON
    metrics_output = {
        'symbol': symbol,
        'test_metrics': metrics,
        'val_metrics': val_metrics,
        'training_history': {
            'epochs_trained': len(history['loss']) if history['loss'] else 0,
            'final_train_loss': history['loss'][-1] if history['loss'] else None,
            'final_val_loss': history['val_loss'][-1] if history['val_loss'] else None,
        },
        'target_met': target_met,
        'timestamp': datetime.now().isoformat(),
    }
    
    metrics_file = output_path / f'{symbol}_validation_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics_output, f, indent=2, default=str)
    
    print(f"\nValidation metrics saved to: {metrics_file}")
    
    # Cleanup
    gc.collect()
    
    return results


def load_model(model_path: str, metadata_path: str = None) -> keras.Model:
    """
    Load a saved model with custom objects registered.
    
    For subclassed models (LSTMTransformerPaper), we need to recreate the
    architecture and load weights, rather than using keras.models.load_model.
    """
    custom_objects = {
        'AntiCollapseDirectionalLoss': AntiCollapseDirectionalLoss,
        'DirectionalHuberLoss': DirectionalHuberLoss,
        'PatchTST': PatchTST,
        'LSTMTransformerPaper': LSTMTransformerPaper,
    }
    
    # First try direct load (works for functional models)
    try:
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
        return model
    except (ValueError, Exception) as e:
        logger.warning(f"Direct model load failed: {e}")
        logger.info("Attempting to rebuild model and load weights...")
    
    # For subclassed models, we need metadata to recreate the architecture
    if metadata_path is None:
        # Try to find metadata next to model
        model_dir = Path(model_path).parent
        metadata_path = model_dir / 'best_model_metadata.json'
    
    if not Path(metadata_path).exists():
        raise FileNotFoundError(
            f"Model metadata not found at {metadata_path}. "
            "Required for loading subclassed models."
        )
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    params = metadata['best_params']
    data_meta = metadata.get('data_metadata', {})
    
    seq_len = data_meta.get('seq_len', 60)
    n_features = data_meta.get('n_features', 147)
    
    model_type = params.get('model_type', 'lstm_transformer')
    dropout = params.get('dropout', 0.2)
    
    # Recreate model architecture
    if model_type == 'patchtst':
        model = PatchTST(
            seq_len=seq_len,
            pred_len=1,
            n_features=n_features,
            patch_len=params.get('patchtst_patch_len', 16),
            stride=params.get('patchtst_patch_len', 16) // 2,
            d_model=params.get('patchtst_d_model', 96),
            n_heads=params.get('patchtst_n_heads', 4),
            n_layers=params.get('patchtst_n_layers', 3),
            d_ff=params.get('patchtst_d_model', 96) * 4,
            dropout=dropout,
            use_revin=True,
        )
    else:
        model = LSTMTransformerPaper(
            sequence_length=seq_len,
            n_features=n_features,
            lstm_units=params.get('lstm_units', 32),
            d_model=params.get('lstm_d_model', 64),
            num_heads=params.get('lstm_num_heads', 4),
            num_blocks=params.get('lstm_num_blocks', 4),
            ff_dim=params.get('lstm_ff_dim', 128),
            dropout=dropout,
        )
    
    # Build model with dummy input
    import numpy as np
    dummy = np.zeros((1, seq_len, n_features), dtype=np.float32)
    _ = model(dummy, training=False)
    
    # Load weights from .keras file
    model.load_weights(model_path)
    logger.info(f"Successfully loaded model weights from {model_path}")
    
    return model


def validate_from_optimization(
    optimization_dir: str,
    symbol: str = 'AAPL',
    output_dir: str = 'validation_results',
) -> Dict[str, Any]:
    """
    Load the best model from optimization results and validate it.
    
    Args:
        optimization_dir: Directory containing optimization results
        symbol: Stock symbol
        output_dir: Output directory for validation results
    
    Returns:
        Validation results dict
    """
    opt_path = Path(optimization_dir)
    
    # Find best model
    best_model_path = opt_path / 'best_model.keras'
    if not best_model_path.exists():
        raise FileNotFoundError(f"Best model not found at {best_model_path}")
    
    # Load metadata
    metadata_path = opt_path / 'best_model_metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        print(f"Loaded model metadata: {metadata['best_params']}")
    
    return run_validation(
        symbol=symbol,
        epochs=0,  # No training
        load_model_path=str(best_model_path),
        output_dir=output_dir,
    )


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbol', '-s', type=str, default='AAPL')
    parser.add_argument('--epochs', '-e', type=int, default=50)
    parser.add_argument('--batch_size', '-b', type=int, default=512)
    parser.add_argument('--load_model', '-l', type=str, default=None,
                        help='Path to load pre-trained model')
    parser.add_argument('--save_model', type=str, default=None,
                        help='Path to save trained model')
    parser.add_argument('--output_dir', '-o', type=str, default='validation_results',
                        help='Output directory for results')
    parser.add_argument('--from_optimization', type=str, default=None,
                        help='Load best model from optimization results directory')
    
    args = parser.parse_args()
    
    if args.from_optimization:
        results = validate_from_optimization(
            optimization_dir=args.from_optimization,
            symbol=args.symbol,
            output_dir=args.output_dir,
        )
    else:
        results = run_validation(
            symbol=args.symbol,
            epochs=args.epochs,
            batch_size=args.batch_size,
            load_model_path=args.load_model,
            save_model_path=args.save_model,
            output_dir=args.output_dir,
        )
