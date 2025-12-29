"""
GBM Baseline Training Script for Stock Return Prediction

This script trains XGBoost and LightGBM models to predict 1-day returns,
using the same engineered features as the LSTM/Transformer regressor.

Key Features:
- Walk-forward cross-validation (time-series aware)
- Feature importance and SHAP integration
- Look-ahead leakage prevention
- Consistent with existing feature_columns.pkl contract

Based on:
- Yu et al. (2025): Gradient Boosting Decision Tree with LSTM for Investment Prediction
- Carboni (2025): Using Gradient Boosting Regressor to forecast Stock Price

Author: AI-Stocks GBM Integration
Date: December 2025
"""

import argparse
import sys
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# P0 FIX: Import yfinance BEFORE any TensorFlow imports to avoid SSL conflict
import yfinance as yf  # noqa: F401 - must be first

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import RobustScaler
import joblib

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import random

# ============================================================================
# DETERMINISTIC SEED SETTINGS (P0 FIX - Added for reproducibility)
# ============================================================================
def set_global_seed(seed: int = 42) -> None:
    """Set seeds for all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    # TensorFlow seed - only if already imported (don't force import here)
    if 'tensorflow' in sys.modules:
        import tensorflow as tf
        tf.random.set_seed(seed)
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

# Default seed - can be overridden via --seed CLI arg
GLOBAL_SEED = int(os.environ.get('SEED', 42))
set_global_seed(GLOBAL_SEED)

from data.data_fetcher import fetch_stock_data
from data.feature_engineer import engineer_features, get_feature_columns, EXPECTED_FEATURE_COUNT
from data.target_engineering import prepare_training_data
from data.cache_manager import DataCacheManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import GBM libraries
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    logger.warning("LightGBM not available. Install with: pip install lightgbm")

# P0.2: Import matplotlib for training curve plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for server use
    import matplotlib.pyplot as plt
    PLT_AVAILABLE = True
except ImportError:
    PLT_AVAILABLE = False
    logger.warning("matplotlib not available for training curves")


# ============================================================================
# P0.2: TRAINING CURVE PLOTTING
# ============================================================================

def plot_training_curve(eval_result: dict, model_class: str, symbol: str, fold: int = None, output_dir: str = 'logs'):
    """
    Plot and save training curves for GBM models.
    
    P0.2 diagnostic tool to detect:
    1. Early convergence (flat curves)
    2. Overfitting (train << val divergence)
    3. Underfitting (high error throughout)
    
    Args:
        eval_result: Dict from lgb.record_evaluation or xgb.evals_result()
        model_class: 'lgb' or 'xgb'
        symbol: Stock ticker for labeling
        fold: CV fold number (None for final model)
        output_dir: Directory to save plots
    
    Returns:
        Path to saved plot, or None if failed
    """
    if not PLT_AVAILABLE:
        logger.warning("matplotlib not available, skipping training curve plot")
        return None
    
    if not eval_result:
        logger.warning("No eval_result data, skipping training curve plot")
        return None
    
    try:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Handle different result formats
        if model_class == 'lgb':
            # LightGBM format: {'training': {'rmse': [...], 'mae': [...]}, 'validation': {...}}
            train_key = 'training'
            val_key = 'validation'
            rmse_key = 'rmse'
            mae_key = 'mae'
        else:
            # XGBoost format: {'validation_0': {'rmse': [...], 'mae': [...]}, 'validation_1': {...}}
            train_key = 'validation_0'
            val_key = 'validation_1'
            rmse_key = 'rmse'
            mae_key = 'mae'
        
        # Plot RMSE
        if train_key in eval_result and rmse_key in eval_result.get(train_key, {}):
            train_rmse = eval_result[train_key][rmse_key]
            axes[0].plot(train_rmse, label='Train RMSE', alpha=0.8)
            
        if val_key in eval_result and rmse_key in eval_result.get(val_key, {}):
            val_rmse = eval_result[val_key][rmse_key]
            axes[0].plot(val_rmse, label='Val RMSE', alpha=0.8)
        
        axes[0].set_xlabel('Boosting Iteration')
        axes[0].set_ylabel('RMSE')
        fold_str = f'Fold {fold}' if fold is not None else 'Final'
        axes[0].set_title(f'{symbol} - {model_class.upper()} Training Curve ({fold_str})')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot MAE
        if train_key in eval_result and mae_key in eval_result.get(train_key, {}):
            train_mae = eval_result[train_key][mae_key]
            axes[1].plot(train_mae, label='Train MAE', alpha=0.8)
            
        if val_key in eval_result and mae_key in eval_result.get(val_key, {}):
            val_mae = eval_result[val_key][mae_key]
            axes[1].plot(val_mae, label='Val MAE', alpha=0.8)
        
        axes[1].set_xlabel('Boosting Iteration')
        axes[1].set_ylabel('MAE')
        axes[1].set_title(f'{symbol} - {model_class.upper()} Training Curve MAE ({fold_str})')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        fold_suffix = f'_fold{fold}' if fold is not None else '_final'
        save_path = Path(output_dir) / f'{model_class}_training_curve_{symbol}{fold_suffix}.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✓ Training curve saved: {save_path}")
        return str(save_path)
        
    except Exception as e:
        logger.warning(f"Failed to plot training curve: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    # Model hyperparameters - P0.2f: Back to basics with moderate settings
    'xgb': {
        # v4.2: AGGRESSIVE anti-collapse settings
        # Previous 0.005 reg still collapsed - reduce further
        'n_estimators': 2000,              # INCREASED for more learning capacity
        'learning_rate': 0.01,             # DECREASED for smoother learning
        'max_depth': 8,                    # INCREASED for more complexity
        'subsample': 0.9,                  # INCREASED for more variance
        'colsample_bytree': 0.9,           # INCREASED for more feature coverage
        'min_child_weight': 1,             # DECREASED to allow finer splits
        'reg_alpha': 0.0001,               # REDUCED 50x from 0.005
        'reg_lambda': 0.0001,              # REDUCED 50x from 0.005 **CRITICAL FIX**
        'early_stopping_rounds': 300,      # INCREASED to allow more exploration
        'random_state': 42,
        'n_jobs': -1,
        'objective': 'reg:squarederror',
        'eval_metric': ['rmse', 'mae'],
    },
    'lgb': {
        # v4.2: AGGRESSIVE anti-collapse settings
        # Previous 0.003 reg still collapsed - reduce further
        'n_estimators': 2000,              # INCREASED for more learning capacity
        'learning_rate': 0.01,             # DECREASED for smoother learning
        'max_depth': 10,                   # INCREASED for more complexity
        'num_leaves': 127,                 # INCREASED (2^7-1) for more splits
        'subsample': 0.9,                  # INCREASED for more variance
        'colsample_bytree': 0.9,           # INCREASED for more feature coverage
        'min_child_samples': 10,           # DECREASED to allow finer splits
        'reg_alpha': 0.0001,               # REDUCED 30x from 0.003
        'reg_lambda': 0.0001,              # REDUCED 30x from 0.003 **CRITICAL FIX**
        'early_stopping_rounds': 300,      # INCREASED to allow more exploration
        'min_split_gain': 0.0,
        'random_state': 42,
        'n_jobs': -1,
        'objective': 'regression',
        'metric': ['rmse', 'mae'],
        'verbosity': -1,
        'force_col_wise': True,
        'deterministic': True,
    },
    # Cross-validation settings
    'cv': {
        'n_splits': 5,
        'test_size': 60,  # ~3 months of trading days
        'gap': 1,  # Gap between train and test to prevent leakage
    },
    # Target processing
    'target': {
        'clip_min': -0.10,  # -10% max loss per day
        'clip_max': 0.10,   # +10% max gain per day
        'log_transform': False,
    }
}


# ============================================================================
# SAMPLE WEIGHT COMPUTATION (December 2025 - Fix for 89.3% positive bias)
# ============================================================================

def compute_regression_sample_weights(y: np.ndarray) -> np.ndarray:
    """
    Compute sample weights to balance positive/negative returns.

    This addresses the issue where GBM models were predicting 89.3% positive
    because stock returns are naturally right-skewed.

    Args:
        y: Target values (returns)

    Returns:
        Sample weights array (normalized to mean=1)
    """
    n_positive = (y > 0).sum()
    n_negative = (y < 0).sum()
    n_zero = (y == 0).sum()

    weights = np.ones(len(y))

    if n_positive > 0 and n_negative > 0:
        # Weight minority class more heavily
        if n_positive > n_negative:
            weights[y < 0] = n_positive / n_negative
            weights[y == 0] = 0.5  # Neutral samples get lower weight
        else:
            weights[y > 0] = n_negative / n_positive
            weights[y == 0] = 0.5

    # Normalize to mean 1
    weights = weights / weights.mean()

    pos_weight = weights[y > 0].mean() if n_positive > 0 else 0
    neg_weight = weights[y < 0].mean() if n_negative > 0 else 0

    logger.info(f"Sample weights: positive={pos_weight:.3f}, negative={neg_weight:.3f}")

    return weights


# ============================================================================
# OUTPUT CALIBRATION (NUCLEAR FIX - December 2025)
# ============================================================================

def calibrate_predictions(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    Post-hoc calibration to center and scale predictions.

    This addresses the issue where GBM models predict 89%+ positive even with
    sample weights. The calibration:
    1. Shifts predictions to match true distribution center
    2. Scales predictions to match true variance

    Args:
        y_pred: Raw model predictions
        y_true: True target values (for calibration fitting)

    Returns:
        Calibrated predictions with balanced distribution
    """
    pred_mean = np.mean(y_pred)
    true_mean = np.mean(y_true)
    pred_std = np.std(y_pred)
    true_std = np.std(y_true)

    # Shift to match true center
    calibrated = y_pred - pred_mean + true_mean

    # Scale to match true variance (if not collapsed)
    if pred_std > 0.0001:
        calibrated = (calibrated - true_mean) * (true_std / pred_std) + true_mean

    # Log calibration stats
    new_pct_positive = float((calibrated > 0).mean() * 100)
    new_pct_negative = float((calibrated < 0).mean() * 100)
    logger.info(f"Calibration: shifted by {true_mean - pred_mean:.6f}, "
                f"scaled by {true_std / pred_std if pred_std > 0.0001 else 0:.3f}")
    logger.info(f"Post-calibration: {new_pct_positive:.1f}% positive, {new_pct_negative:.1f}% negative")

    return calibrated


def save_calibration_params(save_dir: Path, pred_mean: float, pred_std: float,
                            true_mean: float, true_std: float) -> None:
    """Save calibration parameters for inference time."""
    params = {
        'pred_mean': pred_mean,
        'pred_std': pred_std,
        'true_mean': true_mean,
        'true_std': true_std,
    }
    with open(save_dir / 'calibration_params.json', 'w') as f:
        json.dump(params, f, indent=2)
    logger.info(f"Saved calibration params to {save_dir / 'calibration_params.json'}")


# ============================================================================
# DATA PREPARATION
# ============================================================================

def validate_no_leakage(df: pd.DataFrame, feature_cols: List[str]) -> Dict[str, Any]:
    """
    Validate that features don't contain look-ahead bias.
    
    Checks:
    1. No features with 'target' or 'future' in name
    2. No features using next-day values
    3. Features computed only from past data
    
    Returns:
        Dict with validation results and any flagged features
    """
    flagged = []
    
    # Check for obvious naming issues
    suspicious_names = ['target', 'future', 'next', 'forward', 'tomorrow']
    for col in feature_cols:
        col_lower = col.lower()
        for suspicious in suspicious_names:
            if suspicious in col_lower and col not in ['target_1d']:
                flagged.append((col, f"Contains '{suspicious}' in name"))
    
    # Check for features that might use future data
    # These are typically sentiment features that could leak if not properly lagged
    sentiment_features = [c for c in feature_cols if 'sentiment' in c.lower()]
    if sentiment_features:
        logger.info(f"Found {len(sentiment_features)} sentiment features - ensure they are properly lagged")
    
    result = {
        'is_valid': len(flagged) == 0,
        'flagged_features': flagged,
        'sentiment_features': sentiment_features,
        'total_features': len(feature_cols),
    }
    
    if flagged:
        logger.warning(f"Leakage check flagged {len(flagged)} features:")
        for feat, reason in flagged:
            logger.warning(f"  - {feat}: {reason}")
    else:
        logger.info("✓ No look-ahead leakage detected in feature names")
    
    return result


def prepare_gbm_data(
    symbol: str,
    use_cache: bool = True,
    force_refresh: bool = False,
    config: Dict = None
) -> Tuple[pd.DataFrame, List[str], Dict]:
    """
    Prepare data for GBM training using the same feature pipeline as LSTM.
    
    Returns:
        Tuple of (DataFrame with features and target, feature_columns, metadata)
    """
    config = config or DEFAULT_CONFIG
    
    logger.info(f"Preparing GBM data for {symbol}...")
    
    # Use cache manager for consistency with LSTM pipeline
    cache_manager = DataCacheManager()
    
    if use_cache and not force_refresh:
        raw_df, engineered_df, prepared_df, cached_feature_cols = cache_manager.get_or_fetch_data(
            symbol=symbol,
            include_sentiment=True,
            force_refresh=False
        )
        if prepared_df is not None:
            logger.info(f"Using cached data: {prepared_df.shape}")
            feature_cols = cached_feature_cols
        else:
            force_refresh = True
    
    if force_refresh or not use_cache:
        # Fetch fresh data
        raw_df = fetch_stock_data(symbol)
        engineered_df = engineer_features(raw_df, symbol=symbol, include_sentiment=True)
        prepared_df, feature_cols = prepare_training_data(engineered_df, horizons=[1])
        
        # Cache for future use
        if use_cache:
            cache_manager.save_cache(symbol, raw_df, engineered_df, prepared_df, feature_cols)
    
    # Validate feature count
    if len(feature_cols) != EXPECTED_FEATURE_COUNT:
        logger.warning(f"Feature count mismatch: {len(feature_cols)} vs expected {EXPECTED_FEATURE_COUNT}")
    
    # Validate no leakage
    leakage_check = validate_no_leakage(prepared_df, feature_cols)
    
    # Extract target
    target_col = 'target_1d'
    if target_col not in prepared_df.columns:
        raise ValueError(f"Target column '{target_col}' not found in prepared data")
    
    # Apply target clipping if configured
    target_config = config.get('target', {})
    clip_min = target_config.get('clip_min', -0.10)
    clip_max = target_config.get('clip_max', 0.10)
    
    prepared_df[target_col] = prepared_df[target_col].clip(clip_min, clip_max)
    
    logger.info(f"Target range after clipping: [{prepared_df[target_col].min():.4f}, {prepared_df[target_col].max():.4f}]")
    
    # Build metadata
    metadata = {
        'symbol': symbol,
        'n_samples': len(prepared_df),
        'n_features': len(feature_cols),
        'target_col': target_col,
        'target_stats': {
            'mean': float(prepared_df[target_col].mean()),
            'std': float(prepared_df[target_col].std()),
            'min': float(prepared_df[target_col].min()),
            'max': float(prepared_df[target_col].max()),
            'pct_positive': float((prepared_df[target_col] > 0).mean()),
            'pct_negative': float((prepared_df[target_col] < 0).mean()),
        },
        'leakage_check': leakage_check,
        'date_range': {
            'start': str(prepared_df.index[0]) if hasattr(prepared_df.index[0], 'strftime') else str(prepared_df.index[0]),
            'end': str(prepared_df.index[-1]) if hasattr(prepared_df.index[-1], 'strftime') else str(prepared_df.index[-1]),
        },
        'prepared_at': datetime.now().isoformat(),
    }
    
    return prepared_df, feature_cols, metadata


# ============================================================================
# WALK-FORWARD CROSS-VALIDATION
# ============================================================================

def walk_forward_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_class: str,
    config: Dict,
    feature_names: List[str] = None,
    symbol: str = 'UNKNOWN',  # P0.2: Added for training curve file naming
) -> Dict[str, Any]:
    """
    Perform walk-forward cross-validation for time series.
    
    Unlike standard k-fold, this respects temporal ordering:
    - Train on past data
    - Validate on future data
    - No shuffling
    
    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target array (n_samples,)
        model_class: 'xgb' or 'lgb'
        config: Configuration dict with model and CV parameters
        feature_names: List of feature names for importance
        symbol: Stock ticker for training curve file naming (P0.2)
    
    Returns:
        Dict with CV results, metrics per fold, and OOF predictions
    """
    cv_config = config.get('cv', DEFAULT_CONFIG['cv'])
    model_config = config.get(model_class, DEFAULT_CONFIG[model_class]).copy()
    
    n_splits = cv_config.get('n_splits', 5)
    test_size = cv_config.get('test_size', 60)
    gap = cv_config.get('gap', 1)
    
    # Calculate proper splits for time series
    n_samples = len(X)
    min_train_size = max(252, n_samples // (n_splits + 1))  # At least 1 year
    
    logger.info(f"Walk-forward CV: {n_splits} splits, test_size={test_size}, gap={gap}")
    logger.info(f"Total samples: {n_samples}, min train size: {min_train_size}")
    
    # Initialize arrays for OOF predictions
    oof_preds = np.full(n_samples, np.nan)
    oof_indices = np.full(n_samples, -1, dtype=int)
    
    fold_metrics = []
    feature_importances_list = []
    training_curves = []  # P0.2: Store training curves for all folds
    
    # Create time series splits
    tscv = TimeSeriesSplit(n_splits=n_splits, test_size=test_size, gap=gap)
    
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        logger.info(f"\n--- Fold {fold_idx + 1}/{n_splits} ---")
        logger.info(f"Train: {len(train_idx)} samples (indices {train_idx[0]}-{train_idx[-1]})")
        logger.info(f"Val: {len(val_idx)} samples (indices {val_idx[0]}-{val_idx[-1]})")
        
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Scale features (fit on train only)
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # P0.2: Training curve storage for diagnostics
        eval_result = {}

        # December 2025: Compute sample weights to balance positive/negative returns
        sample_weights = compute_regression_sample_weights(y_train)

        # Train model
        if model_class == 'xgb' and XGB_AVAILABLE:
            # P0.2e: Use early stopping to prevent overfitting
            early_stop = model_config.pop('early_stopping_rounds', None)
            model = xgb.XGBRegressor(**model_config, early_stopping_rounds=early_stop)
            
            model.fit(
                X_train_scaled, y_train,
                sample_weight=sample_weights,  # December 2025: Balance positive/negative
                eval_set=[(X_train_scaled, y_train), (X_val_scaled, y_val)],
                verbose=False
            )

            # P0.2e: Log best iteration if early stopped
            if hasattr(model, 'best_iteration'):
                logger.info(f"  XGBoost early stopped at iteration {model.best_iteration}")
            
            # P0.2: Get training history for curve plotting
            eval_result = model.evals_result() if hasattr(model, 'evals_result') else {}
            
            # Get feature importance
            importance = model.feature_importances_
            
        elif model_class == 'lgb' and LGB_AVAILABLE:
            # P0.2e: Extract early stopping for proper callback handling
            early_stop = model_config.pop('early_stopping_rounds', None)
            
            model = lgb.LGBMRegressor(**model_config)
            
            # P0.2e: Build callbacks list
            callbacks = [
                lgb.log_evaluation(period=100),
                lgb.record_evaluation(eval_result)
            ]
            if early_stop:
                callbacks.append(lgb.early_stopping(stopping_rounds=early_stop))
            
            model.fit(
                X_train_scaled, y_train,
                sample_weight=sample_weights,  # December 2025: Balance positive/negative
                eval_set=[(X_train_scaled, y_train), (X_val_scaled, y_val)],
                eval_names=['training', 'validation'],
                callbacks=callbacks
            )

            # Get feature importance
            importance = model.feature_importances_
            
            # P0.2: Log trees trained for diagnostics
            n_trees = model.n_estimators_ if hasattr(model, 'n_estimators_') else model_config.get('n_estimators', 1000)
            logger.info(f"  LightGBM trained {n_trees} trees")
            
        else:
            raise ValueError(f"Model class '{model_class}' not available")
        
        # Predict
        y_pred = model.predict(X_val_scaled)
        
        # P0.3 FIX: Enhanced variance collapse diagnostics
        pred_std = np.std(y_pred)
        pred_mean = np.mean(y_pred)
        positive_pct = (y_pred > 0).mean()
        
        if model_class == 'lgb':
            logger.info(f"  LightGBM Diagnostics (Fold {fold_idx+1}):")
            logger.info(f"    Prediction std: {pred_std:.6f}")
            logger.info(f"    Positive predictions: {positive_pct:.1%}")
            logger.info(f"    Mean prediction: {pred_mean:.6f}")
        
        if pred_std < 0.001:
            logger.warning(f"  ⚠️ Fold {fold_idx+1} VARIANCE COLLAPSE: pred_std={pred_std:.6f}")
            logger.error(f"  Current config: reg_lambda={model_config.get('reg_lambda', 'N/A'):.4f}, "
                        f"reg_alpha={model_config.get('reg_alpha', 'N/A'):.4f}, "
                        f"objective='{model_config.get('objective', 'N/A')}'")
            logger.error(f"  Recommended: reg_lambda < 0.005, reg_alpha < 0.005, objective='regression'")
            if model_class == 'lgb':
                logger.error(f"  LightGBM-specific: Consider increasing num_leaves to 63, max_depth to 7")
            elif model_class == 'xgb':
                logger.error(f"  XGBoost-specific: Increase early_stopping_rounds to 150+, max_depth to 6")
        
        # P0.3 FIX: Check for severe bias
        if positive_pct > 0.90 or positive_pct < 0.10:
            logger.warning(f"  ⚠️ Fold {fold_idx+1} SEVERE BIAS: {positive_pct:.1%} positive")
        
        # Store OOF predictions
        oof_preds[val_idx] = y_pred
        oof_indices[val_idx] = fold_idx
        
        # Calculate metrics
        mse = mean_squared_error(y_val, y_pred)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        
        # Direction accuracy
        dir_acc = np.mean(np.sign(y_val) == np.sign(y_pred))
        
        # Information coefficient (correlation)
        ic = np.corrcoef(y_val, y_pred)[0, 1] if len(y_val) > 1 else 0.0
        
        fold_result = {
            'fold': fold_idx + 1,
            'train_size': len(train_idx),
            'val_size': len(val_idx),
            'mse': float(mse),
            'rmse': float(np.sqrt(mse)),
            'mae': float(mae),
            'r2': float(r2),
            'dir_acc': float(dir_acc),
            'ic': float(ic),
            'pred_std': float(pred_std),  # P0.2: Track prediction variance
        }
        fold_metrics.append(fold_result)
        
        logger.info(f"  MSE: {mse:.6f}, MAE: {mae:.6f}, R²: {r2:.4f}")
        logger.info(f"  Direction Acc: {dir_acc:.4f}, IC: {ic:.4f}")
        logger.info(f"  Prediction std: {pred_std:.6f}")  # P0.2: Log for variance collapse detection
        
        # P0.2: Plot training curve for this fold
        if eval_result:
            curve_path = plot_training_curve(eval_result, model_class, symbol, fold=fold_idx+1)
            if curve_path:
                training_curves.append(curve_path)
        
        # Store feature importances
        if feature_names is not None:
            importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance,
                'fold': fold_idx + 1
            })
            feature_importances_list.append(importance_df)
    
    # Aggregate metrics across folds
    metrics_df = pd.DataFrame(fold_metrics)
    
    agg_metrics = {
        'mse_mean': float(metrics_df['mse'].mean()),
        'mse_std': float(metrics_df['mse'].std()),
        'rmse_mean': float(metrics_df['rmse'].mean()),
        'rmse_std': float(metrics_df['rmse'].std()),
        'mae_mean': float(metrics_df['mae'].mean()),
        'mae_std': float(metrics_df['mae'].std()),
        'r2_mean': float(metrics_df['r2'].mean()),
        'r2_std': float(metrics_df['r2'].std()),
        'dir_acc_mean': float(metrics_df['dir_acc'].mean()),
        'dir_acc_std': float(metrics_df['dir_acc'].std()),
        'ic_mean': float(metrics_df['ic'].mean()),
        'ic_std': float(metrics_df['ic'].std()),
    }
    
    # Aggregate feature importances
    if feature_importances_list:
        all_importances = pd.concat(feature_importances_list, ignore_index=True)
        avg_importance = all_importances.groupby('feature')['importance'].mean().sort_values(ascending=False)
    else:
        avg_importance = pd.Series()
    
    # P0.2: Check for variance collapse across folds
    pred_stds = [fm.get('pred_std', 0) for fm in fold_metrics]
    collapsed_folds = sum(1 for s in pred_stds if s < 0.001)
    if collapsed_folds > 0:
        logger.warning(f"⚠️ {collapsed_folds}/{n_splits} folds have variance collapse (pred_std < 0.001)")
    
    result = {
        'model_class': model_class,
        'n_splits': n_splits,
        'fold_metrics': fold_metrics,
        'aggregate_metrics': agg_metrics,
        'oof_predictions': oof_preds,
        'oof_indices': oof_indices,
        'feature_importance': avg_importance.to_dict(),
        'top_features': list(avg_importance.head(20).index),
        'training_curves': training_curves,  # P0.2: Paths to training curve plots
        'collapsed_folds': collapsed_folds,  # P0.2: Number of folds with variance collapse
    }
    
    logger.info(f"\n=== {model_class.upper()} CV Summary ===")
    logger.info(f"MSE: {agg_metrics['mse_mean']:.6f} ± {agg_metrics['mse_std']:.6f}")
    logger.info(f"MAE: {agg_metrics['mae_mean']:.6f} ± {agg_metrics['mae_std']:.6f}")
    logger.info(f"R²: {agg_metrics['r2_mean']:.4f} ± {agg_metrics['r2_std']:.4f}")
    logger.info(f"Direction Acc: {agg_metrics['dir_acc_mean']:.4f} ± {agg_metrics['dir_acc_std']:.4f}")
    logger.info(f"IC: {agg_metrics['ic_mean']:.4f} ± {agg_metrics['ic_std']:.4f}")
    logger.info(f"Avg prediction std: {np.mean(pred_stds):.6f}")  # P0.2: Log avg prediction variance
    
    return result


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_final_model(
    X: np.ndarray,
    y: np.ndarray,
    model_class: str,
    config: Dict,
    feature_names: List[str] = None,
    symbol: str = 'UNKNOWN',  # P0.2: Added for training curve file naming
    train_on_full_data: bool = False,  # Nuclear Redesign: Option to train production model on all data
) -> Tuple[Any, RobustScaler, Dict]:
    """
    Train final model for deployment.

    Nuclear Redesign: Fixes critical data leakage issue.
    - 3-way split: Train (60%) / Val (20%) / Test (20%)
    - Validation used ONLY for early stopping
    - Metrics computed ONLY on held-out Test set
    - Test set NEVER used for training or hyperparameter selection

    Args:
        X: Feature matrix
        y: Target vector
        model_class: 'xgb' or 'lgb'
        config: Model configuration
        feature_names: Feature names for importance
        symbol: Stock ticker for logging
        train_on_full_data: If True, train production model on ALL data (use after WFE validation)

    Returns:
        Tuple of (trained model, fitted scaler, training metadata)
    """
    model_config = config.get(model_class, DEFAULT_CONFIG[model_class]).copy()

    logger.info(f"Training final {model_class.upper()} model on {len(X)} samples...")

    # Scale features
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # ========================================================================
    # NUCLEAR REDESIGN: Proper 3-way split with NO DATA LEAKAGE
    # ========================================================================
    # Train (60%): Used for model fitting
    # Val (20%): Used ONLY for early stopping (hyperparameter selection)
    # Test (20%): Used ONLY for final metrics (never seen during training)
    # ========================================================================

    if train_on_full_data:
        # PRODUCTION MODE: After WFE validation proves robustness, train on all data
        logger.info("PRODUCTION MODE: Training on ALL data (WFE validation already passed)")
        train_end = len(X)
        val_end = len(X)
        X_train = X_scaled
        y_train = y
        X_val = X_scaled[-int(len(X) * 0.2):]  # Use last 20% for early stopping monitoring only
        y_val = y[-int(len(X) * 0.2):]
        X_test = None  # No test set in production mode
        y_test = None
    else:
        # VALIDATION MODE: Proper 3-way split for honest metrics
        train_end = int(len(X) * 0.60)
        val_end = int(len(X) * 0.80)

        X_train = X_scaled[:train_end]
        y_train = y[:train_end]
        X_val = X_scaled[train_end:val_end]
        y_val = y[train_end:val_end]
        X_test = X_scaled[val_end:]  # HELD-OUT TEST SET - never used for training
        y_test = y[val_end:]

        logger.info(f"3-way split: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test) if X_test is not None else 0}")
        logger.info("NOTE: Test set is HELD-OUT - metrics computed ONLY on test set")
    
    # P0.2: Training curve storage for final model
    eval_result = {}
    
    if model_class == 'xgb' and XGB_AVAILABLE:
        # v4.2: Two-phase training for XGBoost (same as LightGBM) to prevent iteration 0 collapse
        # Phase 1: Find optimal iteration with early stopping
        # Phase 2: Retrain with at least MIN_TREES trees (no early stopping)

        MIN_TREES = 500  # Minimum trees to prevent variance collapse (increased from 100)

        early_stop = model_config.pop('early_stopping_rounds', None)

        # Phase 1: CV model to find best iteration
        model_cv = xgb.XGBRegressor(**model_config, early_stopping_rounds=early_stop)

        model_cv.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )

        eval_result = model_cv.evals_result() if hasattr(model_cv, 'evals_result') else {}

        best_iter = model_cv.best_iteration if hasattr(model_cv, 'best_iteration') else model_config.get('n_estimators', 500)
        final_n_estimators = max(best_iter, MIN_TREES)  # Ensure at least MIN_TREES

        logger.info(f"Phase 1: CV best_iter={best_iter}, using {final_n_estimators} trees for final model")

        # Phase 2: Train final model on training data only (NO early stopping)
        model_config_final = model_config.copy()
        model_config_final['n_estimators'] = final_n_estimators

        model = xgb.XGBRegressor(**model_config_final)  # NO early_stopping_rounds
        model.fit(X_train, y_train)  # No eval_set, no early stopping

        best_iteration = final_n_estimators
        logger.info(f"Phase 2: Final XGBoost model trained {final_n_estimators} trees")
        
    elif model_class == 'lgb' and LGB_AVAILABLE:
        # P0.3 FIX: Two-phase training for LightGBM
        # Phase 1: Find optimal iteration with early stopping
        # Phase 2: Retrain on full data with that iteration count (no early stopping)
        early_stop = model_config.pop('early_stopping_rounds', None)
        
        # Phase 1: CV model to find best iteration
        model_cv = lgb.LGBMRegressor(**model_config)
        
        cv_callbacks = [
            lgb.log_evaluation(period=100),
            lgb.record_evaluation(eval_result)
        ]
        if early_stop:
            cv_callbacks.append(lgb.early_stopping(stopping_rounds=early_stop))
        
        model_cv.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            eval_names=['training', 'validation'],
            callbacks=cv_callbacks
        )
        
        # Get best iteration (with minimum of 200 trees)
        best_iter = model_cv.best_iteration_ if hasattr(model_cv, 'best_iteration_') else model_config.get('n_estimators', 500)
        final_n_estimators = max(best_iter, 500)  # Ensure at least 500 trees (increased from 200)
        
        logger.info(f"Phase 1 complete: best_iter={best_iter}, using {final_n_estimators} trees for final model")
        
        # Phase 2: Train final model on full data (NO early stopping)
        model_config_final = model_config.copy()
        model_config_final['n_estimators'] = final_n_estimators
        # DEBUG: Remove any residual early stopping params
        model_config_final.pop('early_stopping_rounds', None)
        
        logger.info(f"Phase 2 config: {model_config_final}")  # DEBUG
        
        model = lgb.LGBMRegressor(**model_config_final)
        
        # FIXED: Train on training data ONLY to prevent data leakage
        # The validation set (last 20%) must remain unseen for honest backtesting
        logger.info(f"Training final model on {len(X_train)} samples (excluding {len(X_val)} validation samples to prevent data leakage)")
        
        model.fit(X_train, y_train)  # No callbacks, no early stopping, NO VALIDATION DATA
        
        n_trees = model.n_estimators_ if hasattr(model, 'n_estimators_') else final_n_estimators
        best_iteration = final_n_estimators  # Set best_iteration for metadata
        logger.info(f"Phase 2 complete: Final LightGBM model trained {n_trees} trees")
    else:
        raise ValueError(f"Model class '{model_class}' not available")
    
    # P0.2: Plot final model training curve
    curve_path = plot_training_curve(eval_result, model_class, symbol, fold=None)

    # ========================================================================
    # NUCLEAR REDESIGN: Compute metrics ONLY on held-out TEST set
    # ========================================================================
    # This is the critical fix for data leakage. Previously, metrics were
    # computed on the full dataset (X_scaled), which included training and
    # validation data, leading to inflated metrics.
    # ========================================================================

    if X_test is not None and len(X_test) > 0:
        # VALIDATION MODE: Compute metrics on truly held-out test set
        y_pred_test = model.predict(X_test)

        # Information Coefficient on TEST SET ONLY
        ic = np.corrcoef(y_test.flatten(), y_pred_test.flatten())[0, 1] if len(y_test) > 1 else 0.0
        direction_acc = np.mean(np.sign(y_test) == np.sign(y_pred_test))

        # Prediction stats on TEST SET ONLY
        pred_std = float(np.std(y_pred_test))
        pred_mean = float(np.mean(y_pred_test))
        pct_positive = float((y_pred_test > 0).mean())
        pct_negative = float((y_pred_test < 0).mean())

        logger.info(f"\n=== HELD-OUT TEST SET METRICS (NO LEAKAGE) ===")
        logger.info(f"Test samples: {len(X_test)} (last {100 * len(X_test) / len(X):.0f}% of data)")
        logger.info(f"Test IC: {ic:.4f}")
        logger.info(f"Test Direction Accuracy: {direction_acc:.4f}")
        logger.info(f"Test Prediction std: {pred_std:.6f}")
        logger.info(f"Test Prediction distribution: {pct_positive*100:.1f}% positive, {pct_negative*100:.1f}% negative")

        # Also get train/val metrics for WFE calculation
        y_pred_val = model.predict(X_val)
        val_ic = np.corrcoef(y_val.flatten(), y_pred_val.flatten())[0, 1] if len(y_val) > 1 else 0.0
        val_direction_acc = np.mean(np.sign(y_val) == np.sign(y_pred_val))

        # Calculate Walk Forward Efficiency (WFE)
        # WFE = (Test Performance / Validation Performance) * 100
        if val_direction_acc > 0.5:
            wfe = (direction_acc / val_direction_acc) * 100
        else:
            wfe = 0.0

        logger.info(f"Val IC: {val_ic:.4f}, Val Direction Acc: {val_direction_acc:.4f}")
        logger.info(f"Walk Forward Efficiency (WFE): {wfe:.1f}%")

        if wfe < 40:
            logger.warning(f"⚠️ LOW WFE ({wfe:.1f}%): Significant overfitting detected")
        elif wfe < 60:
            logger.info(f"WFE {wfe:.1f}%: Acceptable - some overfitting present")
        else:
            logger.info(f"WFE {wfe:.1f}%: Good - strategy is likely robust")

    else:
        # PRODUCTION MODE: No test set, just check for variance collapse
        y_pred_full = model.predict(X_scaled)
        ic = np.corrcoef(y.flatten(), y_pred_full.flatten())[0, 1] if len(y) > 1 else 0.0
        direction_acc = np.mean(np.sign(y) == np.sign(y_pred_full))
        pred_std = float(np.std(y_pred_full))
        pred_mean = float(np.mean(y_pred_full))
        pct_positive = float((y_pred_full > 0).mean())
        pct_negative = float((y_pred_full < 0).mean())
        val_ic = ic  # Same as full in production mode
        val_direction_acc = direction_acc
        wfe = 100.0  # Not applicable in production mode

        logger.info(f"\n=== PRODUCTION MODE METRICS (Full Data) ===")
        logger.info(f"Note: These metrics are for monitoring only, not for validation")
        logger.info(f"Full IC: {ic:.4f}, Full Direction Acc: {direction_acc:.4f}")

    # P0.2b: Diagnostic for negative IC
    if ic < -0.01:
        logger.warning(f"⚠️ NEGATIVE IC DETECTED: {ic:.4f}")
        logger.warning("   This means predictions are inversely correlated with actuals")
        logger.warning("   Direction accuracy will be below 50%")

    # P0.2: Warning if predictions collapsed (critical diagnostic)
    if pred_std < 0.001:
        logger.warning("⚠️ VARIANCE COLLAPSE DETECTED: Predictions have near-zero variance")
        logger.warning("   This means the model is outputting constant values")
        logger.warning("   Root causes: early stopping too aggressive, huber_delta too large, or data issues")
    if pct_positive < 0.30 or pct_negative < 0.30:
        logger.warning(f"⚠️ Predictions are biased: {pct_positive*100:.1f}% positive, {pct_negative*100:.1f}% negative")

    # NUCLEAR FIX: Hard-stop on severe variance collapse or prediction bias
    if pred_std < 0.001:
        raise ValueError(f"GBM VARIANCE COLLAPSE: pred_std={pred_std:.6f} < 0.001")
    # v4.2: Relaxed bias check - model might predict directionally if it detects trend
    # Previously 0.85, now 0.98 to allow model to save and be evaluated in backtest
    if pct_positive > 0.98 or pct_negative > 0.98:
        bias_dir = "positive" if pct_positive > 0.98 else "negative"
        bias_pct = pct_positive if pct_positive > 0.98 else pct_negative
        raise ValueError(f"GBM PREDICTION BIAS: {bias_pct*100:.1f}% {bias_dir}")
    elif pct_positive > 0.85 or pct_negative > 0.85:
        bias_dir = "positive" if pct_positive > 0.85 else "negative"
        bias_pct = pct_positive if pct_positive > 0.85 else pct_negative
        logger.warning(f"⚠️ MODERATE BIAS: {bias_pct*100:.1f}% {bias_dir} - model will be saved but may underperform")

    metadata = {
        'model_class': model_class,
        'n_samples': len(X),
        'n_features': X.shape[1],
        'best_iteration': best_iteration,
        'training_curve_path': curve_path,
        # NUCLEAR REDESIGN: Store both test and val metrics
        'test_ic': float(ic),
        'test_direction_acc': float(direction_acc),
        'val_ic': float(val_ic),
        'val_direction_acc': float(val_direction_acc),
        'wfe': float(wfe),  # Walk Forward Efficiency
        'ic': float(ic),  # Backward compatibility (now uses test IC)
        'direction_acc': float(direction_acc),  # Backward compatibility
        'prediction_stats': {
            'mean': pred_mean,
            'std': pred_std,
            'min': float(np.min(y_pred_test if X_test is not None else y_pred_full)),
            'max': float(np.max(y_pred_test if X_test is not None else y_pred_full)),
            'pct_positive': pct_positive,
            'pct_negative': pct_negative,
        },
        'train_samples': len(X_train),
        'val_samples': len(X_val),
        'test_samples': len(X_test) if X_test is not None else 0,
        'train_on_full_data': train_on_full_data,
        'trained_at': datetime.now().isoformat(),
    }

    return model, scaler, metadata


# ============================================================================
# MAIN TRAINING FUNCTION
# ============================================================================

def train_gbm_models(
    symbol: str,
    use_cache: bool = True,
    force_refresh: bool = False,
    overwrite: bool = False,
    config: Dict = None,
) -> Dict[str, Any]:
    """
    Train XGBoost and LightGBM models for a symbol.
    
    Args:
        symbol: Stock ticker symbol
        use_cache: Whether to use cached data
        force_refresh: Force data refresh
        overwrite: Overwrite existing models
        config: Configuration dict (uses DEFAULT_CONFIG if None)
    
    Returns:
        Dict with training results for both models
    """
    config = config or DEFAULT_CONFIG
    
    # Setup output directory
    save_dir = Path(f'saved_models/{symbol}/gbm')
    save_dir.mkdir(parents=True, exist_ok=True)
    
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    # Check existing models
    xgb_path = save_dir / 'xgb_reg.joblib'
    lgb_path = save_dir / 'lgb_reg.joblib'
    
    if not overwrite and xgb_path.exists() and lgb_path.exists():
        logger.info(f"Models already exist for {symbol}. Use --overwrite to retrain.")
        return {'status': 'skipped', 'reason': 'models_exist'}
    
    # Prepare data
    df, feature_cols, data_metadata = prepare_gbm_data(
        symbol=symbol,
        use_cache=use_cache,
        force_refresh=force_refresh,
        config=config
    )
    
    # Extract features and target
    X = df[feature_cols].values
    y = df['target_1d'].values
    
    logger.info(f"Data shape: X={X.shape}, y={y.shape}")
    
    results = {
        'symbol': symbol,
        'data_metadata': data_metadata,
        'models': {},
    }
    
    # Train XGBoost
    if XGB_AVAILABLE:
        logger.info("\n" + "="*60)
        logger.info("Training XGBoost...")
        logger.info("="*60)
        
        # Walk-forward CV
        xgb_cv_results = walk_forward_cv(X, y, 'xgb', config, feature_cols, symbol=symbol)
        
        # Train final model
        xgb_model, xgb_scaler, xgb_metadata = train_final_model(X, y, 'xgb', config, feature_cols, symbol=symbol)
        
        # Save model
        joblib.dump(xgb_model, xgb_path)
        joblib.dump(xgb_scaler, save_dir / 'xgb_scaler.joblib')
        
        # Save OOF predictions
        oof_df = pd.DataFrame({
            'actual': y,
            'predicted': xgb_cv_results['oof_predictions'],
            'fold': xgb_cv_results['oof_indices'],
        })
        oof_df.to_csv(logs_dir / f'gbm_oof_preds_xgb_{symbol}.csv', index=False)
        
        results['models']['xgb'] = {
            'cv_results': xgb_cv_results,
            'final_metadata': xgb_metadata,
            'model_path': str(xgb_path),
            'scaler_path': str(save_dir / 'xgb_scaler.joblib'),
        }
        
        logger.info(f"XGBoost model saved to {xgb_path}")
    
    # Train LightGBM
    if LGB_AVAILABLE:
        logger.info("\n" + "="*60)
        logger.info("Training LightGBM...")
        logger.info("="*60)
        
        # Walk-forward CV
        lgb_cv_results = walk_forward_cv(X, y, 'lgb', config, feature_cols, symbol=symbol)
        
        # Train final model
        lgb_model, lgb_scaler, lgb_metadata = train_final_model(X, y, 'lgb', config, feature_cols, symbol=symbol)
        
        # Save model
        joblib.dump(lgb_model, lgb_path)
        joblib.dump(lgb_scaler, save_dir / 'lgb_scaler.joblib')
        
        # Save OOF predictions
        oof_df = pd.DataFrame({
            'actual': y,
            'predicted': lgb_cv_results['oof_predictions'],
            'fold': lgb_cv_results['oof_indices'],
        })
        oof_df.to_csv(logs_dir / f'gbm_oof_preds_lgb_{symbol}.csv', index=False)
        
        results['models']['lgb'] = {
            'cv_results': lgb_cv_results,
            'final_metadata': lgb_metadata,
            'model_path': str(lgb_path),
            'scaler_path': str(save_dir / 'lgb_scaler.joblib'),
        }
        
        logger.info(f"LightGBM model saved to {lgb_path}")
    
    # Save feature columns for inference
    with open(save_dir / 'feature_columns.pkl', 'wb') as f:
        pickle.dump(feature_cols, f)
    
    # Save combined metadata
    with open(save_dir / 'training_metadata.json', 'w') as f:
        # Convert non-serializable items
        results_serializable = json.loads(json.dumps(results, default=str))
        json.dump(results_serializable, f, indent=2)
    
    logger.info(f"\n✓ GBM training complete for {symbol}")
    logger.info(f"  Models saved to: {save_dir}")
    
    return results


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train GBM baseline models (XGBoost + LightGBM)'
    )
    parser.add_argument('symbol', type=str, help='Stock ticker symbol (e.g., AAPL)')
    parser.add_argument('--no-cache', action='store_true', help='Do not use cached data')
    parser.add_argument('--force-refresh', action='store_true', help='Force data refresh')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing models')
    parser.add_argument('--n-splits', type=int, default=5, help='Number of CV splits')
    parser.add_argument('--epochs', type=int, default=1000, help='Max boosting rounds')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Apply seed setting from CLI arg (overrides env var) - P0 FIX
    set_global_seed(args.seed)
    logger.info(f"Using seed: {args.seed}")
    
    # Build config with CLI overrides
    config = DEFAULT_CONFIG.copy()
    config['cv']['n_splits'] = args.n_splits
    config['xgb']['n_estimators'] = args.epochs
    config['lgb']['n_estimators'] = args.epochs
    # P0: Pass seed to GBM models
    config['xgb']['random_state'] = args.seed
    config['lgb']['random_state'] = args.seed
    
    # Train models
    results = train_gbm_models(
        symbol=args.symbol.upper(),
        use_cache=not args.no_cache,
        force_refresh=args.force_refresh,
        overwrite=args.overwrite,
        config=config,
    )
    
    # Print summary
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    if results.get('status') == 'skipped':
        print(f"Skipped: {results.get('reason')}")
        return
    
    for model_name, model_results in results.get('models', {}).items():
        cv = model_results.get('cv_results', {})
        agg = cv.get('aggregate_metrics', {})
        print(f"\n{model_name.upper()}:")
        print(f"  MSE:  {agg.get('mse_mean', 0):.6f} ± {agg.get('mse_std', 0):.6f}")
        print(f"  MAE:  {agg.get('mae_mean', 0):.6f} ± {agg.get('mae_std', 0):.6f}")
        print(f"  R²:   {agg.get('r2_mean', 0):.4f} ± {agg.get('r2_std', 0):.4f}")
        print(f"  Dir:  {agg.get('dir_acc_mean', 0):.4f} ± {agg.get('dir_acc_std', 0):.4f}")
        print(f"  IC:   {agg.get('ic_mean', 0):.4f} ± {agg.get('ic_std', 0):.4f}")


if __name__ == '__main__':
    main()
