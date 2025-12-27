"""
Optuna Hyperparameter Optimization for Financial Time Series Models

This script implements systematic hyperparameter search using Optuna with:
    - Tree-structured Parzen Estimator (TPE) sampler
    - Median pruner for early stopping of bad trials
    - Multi-objective optimization (return vs drawdown)
    - GPU-accelerated training with batch_size >= 512

Optimization targets:
    1. Model architecture (PatchTST vs LSTM-Transformer)
    2. Training hyperparameters (lr, batch_size, epochs)
    3. Loss function parameters
    4. Feature engineering options

Usage:
    python scripts/optimize_hyperparams.py --n_trials 100 --symbol AAPL
    python scripts/optimize_hyperparams.py --n_trials 50 --symbols AAPL,GOOGL,MSFT --parallel 4
"""

from __future__ import annotations

import os
import sys
import argparse
import logging
import json
import warnings
import gc
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import optuna
from optuna.trial import Trial
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler

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

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import project modules
import keras
from keras import ops

# Backend-specific imports and GPU configuration
if BACKEND == 'tensorflow':
    import tensorflow as tf
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
        # Enable memory efficient settings
        torch.cuda.empty_cache()
    else:
        logger.warning("No GPU detected - training will be slow")

# Project imports
from data.data_fetcher import fetch_stock_data
from data.feature_engineer import engineer_features
from models.patchtst import create_patchtst_model, PatchTST
from models.lstm_transformer_paper import (
    create_paper_model,
    LSTMTransformerPaper,
    AntiCollapseDirectionalLoss,
    DirectionalHuberLoss
)


# =============================================================================
# DATA PREPARATION
# =============================================================================

def prepare_data(
    symbol: str,
    seq_len: int = 60,
    val_split: float = 0.15,
    test_split: float = 0.15,
    include_sentiment: bool = False,
    use_cache: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Prepare data for hyperparameter optimization.
    
    Args:
        use_cache: If True, try to load cached prepared data first.
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, metadata
    """
    import pickle
    from pathlib import Path
    
    # Cache is at root level, not inside python-ai-service
    cache_dir = Path(__file__).parent.parent.parent / 'cache' / symbol
    engineered_cache = cache_dir / 'engineered_features.pkl'
    raw_cache = cache_dir / 'raw_data.pkl'
    
    # Try to load engineered features from cache (for when API fails)
    df = None
    if use_cache and engineered_cache.exists():
        logger.info(f"Loading cached engineered features from {engineered_cache}")
        try:
            with open(engineered_cache, 'rb') as f:
                df = pickle.load(f)
            logger.info(f"Loaded engineered features: {len(df)} rows, {len(df.columns)} columns")
        except Exception as e:
            logger.warning(f"Engineered cache load failed: {e}")
            df = None
    
    # If no engineered features, try fetching fresh or from raw cache
    if df is None:
        logger.info(f"Preparing data for {symbol}...")
        
        try:
            raw_df = fetch_stock_data(symbol)
        except ValueError as e:
            # Try loading from raw_data cache
            if use_cache and raw_cache.exists():
                logger.info(f"API failed, loading from raw data cache: {raw_cache}")
                with open(raw_cache, 'rb') as f:
                    raw_df = pickle.load(f)
            else:
                raise e
        
        df = engineer_features(raw_df, symbol=symbol, include_sentiment=include_sentiment)
    
    # Create target: next day return
    df['target'] = df['Close'].pct_change().shift(-1)
    df = df.dropna(subset=['target'])
    
    # Get feature columns (exclude target and non-feature columns)
    exclude_cols = ['target', 'Date', 'date', 'Symbol', 'symbol', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close']
    feature_cols = [c for c in df.columns if c not in exclude_cols and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    
    # Convert to numpy
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
    
    # Split data (temporal)
    n = len(X_seq)
    train_end = int(n * (1 - val_split - test_split))
    val_end = int(n * (1 - test_split))
    
    X_train, y_train = X_seq[:train_end], y_seq[:train_end]
    X_val, y_val = X_seq[train_end:val_end], y_seq[train_end:val_end]
    X_test, y_test = X_seq[val_end:], y_seq[val_end:]
    
    # Normalize features
    mean = X_train.mean(axis=(0, 1), keepdims=True)
    std = X_train.std(axis=(0, 1), keepdims=True) + 1e-8
    
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std
    
    metadata = {
        'symbol': symbol,
        'n_features': X_train.shape[2],
        'seq_len': seq_len,
        'train_size': len(X_train),
        'val_size': len(X_val),
        'test_size': len(X_test),
        'feature_cols': feature_cols,
        'mean': mean,
        'std': std,
    }
    
    logger.info(f"Data prepared: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}, features={X_train.shape[2]}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, metadata


# =============================================================================
# PRUNING CALLBACK (GLOBAL SCOPE TO PREVENT MEMORY LEAKS)
# =============================================================================

class OptunaPruningCallback(keras.callbacks.Callback):
    """
    Optuna pruning callback - defined at module level to prevent
    class recreation and memory leaks during optimization.
    """
    
    def __init__(self, trial: Trial, monitor: str = 'val_loss'):
        super().__init__()
        self.trial = trial
        self.monitor = monitor
        self.best_val = float('inf')
        self.epoch = 0
    
    def on_epoch_end(self, epoch, logs=None):
        self.epoch = epoch
        val_loss = logs.get(self.monitor, float('inf'))
        
        # Track best
        if val_loss < self.best_val:
            self.best_val = val_loss
        
        # Report to Optuna
        self.trial.report(val_loss, epoch)
        
        # Prune if needed
        if self.trial.should_prune():
            raise optuna.TrialPruned()


# =============================================================================
# OBJECTIVE FUNCTION
# =============================================================================

def create_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_features: int,
    seq_len: int = 60,
    max_epochs: int = 50,
    min_batch_size: int = 512,
    save_dir: Optional[Path] = None,
):
    """Create Optuna objective function."""
    
    def objective(trial: Trial) -> float:
        """
        Optuna objective function.
        
        Optimizes for validation directional accuracy + risk-adjusted returns.
        """
        # CRITICAL: Clear session and collect garbage to prevent memory leaks
        keras.backend.clear_session()
        gc.collect()
        
        # Backend-specific cleanup
        if BACKEND == 'tensorflow':
            tf.compat.v1.reset_default_graph()
        else:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        # Suggest model type
        model_type = trial.suggest_categorical('model_type', ['patchtst', 'lstm_transformer'])
        
        # Common hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
        batch_size = trial.suggest_categorical('batch_size', [512, 1024, 2048])
        dropout = trial.suggest_float('dropout', 0.05, 0.3)
        
        # Loss function
        loss_type = trial.suggest_categorical('loss_type', ['huber', 'anti_collapse', 'directional_huber'])
        
        if model_type == 'patchtst':
            # PatchTST hyperparameters
            d_model = trial.suggest_categorical('patchtst_d_model', [64, 96, 128])
            n_layers = trial.suggest_int('patchtst_n_layers', 2, 4)
            n_heads = trial.suggest_categorical('patchtst_n_heads', [4, 8])
            patch_len = trial.suggest_categorical('patchtst_patch_len', [8, 12, 16, 20])
            stride = patch_len // 2
            
            model = PatchTST(
                seq_len=seq_len,
                pred_len=1,
                n_features=n_features,
                patch_len=patch_len,
                stride=stride,
                d_model=d_model,
                n_heads=n_heads,
                n_layers=n_layers,
                d_ff=d_model * 4,
                dropout=dropout,
                use_revin=True,
            )
            
        else:  # lstm_transformer
            lstm_units = trial.suggest_categorical('lstm_units', [32, 64, 128])
            d_model = trial.suggest_categorical('lstm_d_model', [64, 128])
            num_heads = trial.suggest_categorical('lstm_num_heads', [4, 8])
            num_blocks = trial.suggest_int('lstm_num_blocks', 2, 4)
            ff_dim = trial.suggest_categorical('lstm_ff_dim', [128, 256])
            
            model = LSTMTransformerPaper(
                sequence_length=seq_len,
                n_features=n_features,
                lstm_units=lstm_units,
                d_model=d_model,
                num_heads=num_heads,
                num_blocks=num_blocks,
                ff_dim=ff_dim,
                dropout=dropout,
            )
        
        # Build model
        model.build(input_shape=(None, seq_len, n_features))
        
        # Select loss function
        if loss_type == 'huber':
            loss_fn = keras.losses.Huber(delta=1.0)
        elif loss_type == 'anti_collapse':
            variance_penalty = trial.suggest_float('variance_penalty_weight', 0.1, 0.5)
            min_variance = trial.suggest_float('min_variance_target', 0.005, 0.015)
            loss_fn = AntiCollapseDirectionalLoss(
                variance_penalty_weight=variance_penalty,
                min_variance_target=min_variance,
                sign_diversity_weight=0.15,
            )
        else:  # directional_huber
            direction_weight = trial.suggest_float('direction_weight', 0.5, 3.0)
            loss_fn = DirectionalHuberLoss(
                delta=1.0,
                direction_weight=direction_weight,
            )
        
        # Compile model
        model.compile(
            optimizer=keras.optimizers.AdamW(
                learning_rate=learning_rate,
                weight_decay=0.01,
            ),
            loss=loss_fn,
            metrics=['mae'],
        )
        
        # Setup callbacks - use global OptunaPruningCallback class
        callbacks = [
            OptunaPruningCallback(trial, monitor='val_loss'),
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
            ),
        ]
        
        # Add model checkpoint if save_dir provided
        if save_dir is not None:
            trial_model_path = save_dir / f"trial_{trial.number}_model.keras"
            callbacks.append(
                keras.callbacks.ModelCheckpoint(
                    filepath=str(trial_model_path),
                    monitor='val_loss',
                    save_best_only=True,
                    save_weights_only=False,
                    verbose=0,
                )
            )
        
        # Train
        try:
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=max_epochs,
                batch_size=batch_size,
                callbacks=callbacks,
                verbose=0,
            )
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.warning(f"Training failed: {e}")
            raise optuna.TrialPruned()
        
        # Evaluate on validation set
        y_pred = model.predict(X_val, verbose=0)
        
        # Calculate metrics
        # 1. Directional accuracy
        correct_direction = np.sum(np.sign(y_pred.flatten()) == np.sign(y_val.flatten()))
        directional_accuracy = correct_direction / len(y_val)
        
        # 2. Prediction variance (penalize collapse)
        pred_std = np.std(y_pred)
        variance_penalty = max(0, 0.005 - pred_std) * 100  # Penalty if std < 0.5%
        
        # 3. Correlation with actual returns
        correlation = np.corrcoef(y_pred.flatten(), y_val.flatten())[0, 1]
        if np.isnan(correlation):
            correlation = 0.0
        
        # 4. Simulated returns (simple backtest)
        positions = np.sign(y_pred.flatten())
        returns = positions * y_val.flatten()
        cumulative_return = np.sum(returns)
        sharpe = np.mean(returns) / (np.std(returns) + 1e-8) * np.sqrt(252)
        
        # Combined score (higher is better, so negate for minimization)
        # Weight: 40% directional accuracy, 30% Sharpe, 20% correlation, 10% variance
        score = (
            0.4 * directional_accuracy +
            0.3 * (sharpe / 3 + 0.5) +  # Normalize Sharpe to ~[0, 1]
            0.2 * (correlation + 1) / 2 +  # Normalize correlation to [0, 1]
            0.1 * (1 - variance_penalty)  # Penalty for collapse
        )
        
        # Report metrics
        trial.set_user_attr('directional_accuracy', directional_accuracy)
        trial.set_user_attr('sharpe_ratio', sharpe)
        trial.set_user_attr('correlation', correlation)
        trial.set_user_attr('pred_std', pred_std)
        trial.set_user_attr('cumulative_return', cumulative_return)
        trial.set_user_attr('best_val_loss', min(history.history['val_loss']))
        
        logger.info(f"Trial {trial.number}: score={score:.4f}, dir_acc={directional_accuracy:.4f}, sharpe={sharpe:.2f}")
        
        # Cleanup: delete model and collect garbage to prevent memory accumulation
        del model
        keras.backend.clear_session()
        gc.collect()
        
        # Backend-specific cleanup
        if BACKEND == 'torch':
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        
        # Return negative score (Optuna minimizes by default)
        return -score
    
    return objective


# =============================================================================
# MAIN OPTIMIZATION
# =============================================================================

def run_optimization(
    symbol: str,
    n_trials: int = 100,
    n_startup_trials: int = 20,
    n_jobs: int = 1,
    study_name: Optional[str] = None,
    storage: Optional[str] = None,
    seq_len: int = 60,
    max_epochs: int = 50,
    output_dir: str = 'optimization_results',
) -> optuna.Study:
    """
    Run hyperparameter optimization.
    
    Args:
        symbol: Stock symbol to optimize for
        n_trials: Number of optimization trials
        n_startup_trials: Random trials before TPE kicks in
        n_jobs: Number of parallel jobs
        study_name: Name for the study
        storage: Optuna storage URL (e.g., sqlite:///study.db)
        seq_len: Sequence length
        max_epochs: Maximum training epochs
        output_dir: Directory for saving results
        
    Returns:
        Optuna study object
    """
    logger.info(f"Starting optimization for {symbol}")
    logger.info(f"Trials: {n_trials}, Startup: {n_startup_trials}, Jobs: {n_jobs}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Prepare data
    X_train, y_train, X_val, y_val, X_test, y_test, metadata = prepare_data(
        symbol=symbol,
        seq_len=seq_len,
    )
    
    # Create sampler and pruner
    sampler = TPESampler(
        n_startup_trials=n_startup_trials,
        multivariate=True,
        seed=42,
    )
    
    pruner = MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=10,
        interval_steps=1,
    )
    
    # Create or load study
    if study_name is None:
        study_name = f"hyperparam_opt_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    study = optuna.create_study(
        study_name=study_name,
        direction='minimize',  # We return negative score
        sampler=sampler,
        pruner=pruner,
        storage=storage,
        load_if_exists=True,
    )
    
    # Create models directory
    models_dir = output_path / 'models'
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Create objective
    objective = create_objective(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        n_features=metadata['n_features'],
        seq_len=seq_len,
        max_epochs=max_epochs,
        save_dir=models_dir,
    )
    
    # Run optimization
    study.optimize(
        objective,
        n_trials=n_trials,
        n_jobs=n_jobs,
        show_progress_bar=True,
        gc_after_trial=True,  # Enable Optuna's built-in garbage collection
    )
    
    # Save results and best model
    save_results(study, metadata, output_path)
    
    # Retrain and save the best model
    save_best_model(study, X_train, y_train, X_val, y_val, metadata, output_path)
    
    # Print best trial
    print_best_trial(study)
    
    return study


def save_results(
    study: optuna.Study,
    metadata: Dict[str, Any],
    output_path: Path,
) -> None:
    """Save optimization results."""
    
    # Best trial info
    best_trial = study.best_trial
    
    results = {
        'study_name': study.study_name,
        'best_value': -best_trial.value,  # Negate back to positive score
        'best_params': best_trial.params,
        'best_user_attrs': best_trial.user_attrs,
        'n_trials': len(study.trials),
        'metadata': {
            'symbol': metadata['symbol'],
            'n_features': metadata['n_features'],
            'train_size': metadata['train_size'],
            'val_size': metadata['val_size'],
            'test_size': metadata['test_size'],
        },
        'timestamp': datetime.now().isoformat(),
    }
    
    # Save JSON
    results_file = output_path / f"{study.study_name}_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to {results_file}")
    
    # Save trials dataframe
    df = study.trials_dataframe()
    csv_file = output_path / f"{study.study_name}_trials.csv"
    df.to_csv(csv_file, index=False)
    
    logger.info(f"Trials CSV saved to {csv_file}")


def print_best_trial(study: optuna.Study) -> None:
    """Print best trial information."""
    
    best_trial = study.best_trial
    
    print("\n" + "=" * 60)
    print("BEST TRIAL")
    print("=" * 60)
    print(f"Score: {-best_trial.value:.4f}")  # Negate back
    print(f"Trial number: {best_trial.number}")
    print("\nBest hyperparameters:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    print("\nMetrics:")
    for key, value in best_trial.user_attrs.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    print("=" * 60)


def save_best_model(
    study: optuna.Study,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metadata: Dict[str, Any],
    output_path: Path,
) -> None:
    """
    Retrain the best model with best hyperparameters and save to disk.
    """
    logger.info("Retraining best model for final save...")
    
    # Clear session before retraining
    keras.backend.clear_session()
    gc.collect()
    
    if BACKEND == 'torch':
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    best_trial = study.best_trial
    params = best_trial.params
    
    seq_len = metadata['seq_len']
    n_features = metadata['n_features']
    
    # Recreate model with best params
    model_type = params['model_type']
    dropout = params['dropout']
    learning_rate = params['learning_rate']
    loss_type = params['loss_type']
    
    if model_type == 'patchtst':
        d_model = params['patchtst_d_model']
        n_layers = params['patchtst_n_layers']
        n_heads = params['patchtst_n_heads']
        patch_len = params['patchtst_patch_len']
        
        model = PatchTST(
            seq_len=seq_len,
            pred_len=1,
            n_features=n_features,
            patch_len=patch_len,
            stride=patch_len // 2,
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            d_ff=d_model * 4,
            dropout=dropout,
            use_revin=True,
        )
    else:
        model = LSTMTransformerPaper(
            sequence_length=seq_len,
            n_features=n_features,
            lstm_units=params['lstm_units'],
            d_model=params['lstm_d_model'],
            num_heads=params['lstm_num_heads'],
            num_blocks=params['lstm_num_blocks'],
            ff_dim=params['lstm_ff_dim'],
            dropout=dropout,
        )
    
    model.build(input_shape=(None, seq_len, n_features))
    
    # Select loss
    if loss_type == 'huber':
        loss_fn = keras.losses.Huber(delta=1.0)
    elif loss_type == 'anti_collapse':
        loss_fn = AntiCollapseDirectionalLoss(
            variance_penalty_weight=params.get('variance_penalty_weight', 0.3),
            min_variance_target=params.get('min_variance_target', 0.008),
            sign_diversity_weight=0.15,
        )
    else:
        loss_fn = DirectionalHuberLoss(
            delta=1.0,
            direction_weight=params.get('direction_weight', 1.5),
        )
    
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=0.01,
        ),
        loss=loss_fn,
        metrics=['mae'],
    )
    
    # Train with best params
    batch_size = params['batch_size']
    
    best_model_path = output_path / 'best_model.keras'
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=15,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=str(best_model_path),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            verbose=1,
        ),
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,  # More epochs for final training
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    
    # Save metadata alongside model
    model_metadata = {
        'study_name': study.study_name,
        'best_trial_number': best_trial.number,
        'best_params': params,
        'best_metrics': best_trial.user_attrs,
        'training_history': {
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'best_val_loss': min(history.history['val_loss']),
            'epochs_trained': len(history.history['loss']),
        },
        'data_metadata': {
            'symbol': metadata['symbol'],
            'n_features': n_features,
            'seq_len': seq_len,
            'train_size': metadata['train_size'],
        },
        'timestamp': datetime.now().isoformat(),
    }
    
    metadata_path = output_path / 'best_model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(model_metadata, f, indent=2, default=str)
    
    logger.info(f"Best model saved to: {best_model_path}")
    logger.info(f"Model metadata saved to: {metadata_path}")


# =============================================================================
# QUICK OPTIMIZATION FOR TESTING
# =============================================================================

def quick_optimize(
    symbol: str = 'AAPL',
    n_trials: int = 10,
    max_epochs: int = 20,
) -> Dict[str, Any]:
    """
    Quick optimization for testing/development.
    
    Runs a small optimization study to verify the pipeline works.
    """
    logger.info(f"Quick optimization: {n_trials} trials, {max_epochs} epochs")
    
    study = run_optimization(
        symbol=symbol,
        n_trials=n_trials,
        n_startup_trials=5,
        max_epochs=max_epochs,
        output_dir='optimization_results/quick',
    )
    
    return {
        'best_score': -study.best_value,
        'best_params': study.best_trial.params,
        'best_metrics': study.best_trial.user_attrs,
    }


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Optuna Hyperparameter Optimization for Financial Time Series'
    )
    
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        default='AAPL',
        help='Stock symbol to optimize for'
    )
    
    parser.add_argument(
        '--n_trials', '-n',
        type=int,
        default=100,
        help='Number of optimization trials'
    )
    
    parser.add_argument(
        '--n_startup', 
        type=int,
        default=20,
        help='Number of random startup trials'
    )
    
    parser.add_argument(
        '--n_jobs', '-j',
        type=int,
        default=1,
        help='Number of parallel jobs'
    )
    
    parser.add_argument(
        '--max_epochs', '-e',
        type=int,
        default=50,
        help='Maximum training epochs per trial'
    )
    
    parser.add_argument(
        '--seq_len',
        type=int,
        default=60,
        help='Sequence length (lookback window)'
    )
    
    parser.add_argument(
        '--output_dir', '-o',
        type=str,
        default='optimization_results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--study_name',
        type=str,
        default=None,
        help='Name for the Optuna study'
    )
    
    parser.add_argument(
        '--storage',
        type=str,
        default=None,
        help='Optuna storage URL (e.g., sqlite:///study.db)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick optimization for testing'
    )
    
    args = parser.parse_args()
    
    if args.quick:
        results = quick_optimize(
            symbol=args.symbol,
            n_trials=10,
            max_epochs=20,
        )
        print("\nQuick optimization results:")
        print(f"  Best score: {results['best_score']:.4f}")
        print(f"  Best params: {results['best_params']}")
    else:
        run_optimization(
            symbol=args.symbol,
            n_trials=args.n_trials,
            n_startup_trials=args.n_startup,
            n_jobs=args.n_jobs,
            max_epochs=args.max_epochs,
            seq_len=args.seq_len,
            output_dir=args.output_dir,
            study_name=args.study_name,
            storage=args.storage,
        )


if __name__ == '__main__':
    main()
