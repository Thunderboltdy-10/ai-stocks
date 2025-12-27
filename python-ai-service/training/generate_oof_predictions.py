"""
Out-of-Fold (OOF) Prediction Generator for Stacking Ensemble

This script generates out-of-fold predictions from LSTM regressor and GBM models
using time-series cross-validation. The OOF predictions are used to train a
meta-learner (stacking ensemble) without data leakage.

Key Features:
- TimeSeriesSplit with gap to prevent look-ahead bias
- Generates predictions for both LSTM and GBM models
- Saves OOF predictions for meta-learner training
- Tracks prediction variance and model quality metrics

Based on:
- Wolpert (1992): "Stacked Generalization"
- Breiman (1996): "Stacked Regressions"

Usage:
    python training/generate_oof_predictions.py --symbol AAPL --n-splits 5 --gap 1
"""

import argparse
import sys
import json
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import RobustScaler

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.data_fetcher import fetch_stock_data
from data.feature_engineer import engineer_features, get_feature_columns, EXPECTED_FEATURE_COUNT
from data.target_engineering import prepare_training_data
from utils.model_paths import ModelPaths

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_CONFIG = {
    'n_splits': 5,        # Number of CV folds
    'gap': 1,             # Days gap between train/test to prevent leakage
    'min_train_size': 252,  # Minimum training days (~1 year)
    'sequence_length': 10,  # LSTM sequence length
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def configure_gpu():
    """Configure TensorFlow to use GPU efficiently."""
    import tensorflow as tf
    
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logger.info(f"✓ GPU configured: {len(gpus)} device(s) available")
        except RuntimeError as e:
            logger.warning(f"GPU configuration warning: {e}")
    else:
        logger.warning("⚠️ No GPU detected, running on CPU")


def load_lstm_model(symbol: str, model_paths: ModelPaths):
    """Load pre-trained LSTM regressor model and scalers."""
    import tensorflow as tf
    
    regressor_paths = model_paths.regressor_paths
    
    model = None
    feature_scaler = None
    target_scaler = None
    
    # Try to load model
    model_path = regressor_paths.model
    if model_path.exists():
        try:
            model = tf.keras.models.load_model(str(model_path), compile=False)
            logger.info(f"✓ Loaded LSTM model from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load model from {model_path}: {e}")
            # Try SavedModel format
            try:
                model = tf.saved_model.load(str(model_path))
                logger.info(f"✓ Loaded LSTM as SavedModel from {model_path}")
            except Exception as e2:
                logger.error(f"Failed to load model in any format: {e2}")
    
    # Load scalers
    scaler_path = regressor_paths.feature_scaler
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            feature_scaler = pickle.load(f)
        logger.info(f"✓ Loaded feature scaler from {scaler_path}")
    
    target_scaler_path = regressor_paths.target_scaler
    if target_scaler_path.exists():
        with open(target_scaler_path, 'rb') as f:
            target_scaler = pickle.load(f)
        logger.info(f"✓ Loaded target scaler from {target_scaler_path}")
    
    return model, feature_scaler, target_scaler


def load_gbm_models(symbol: str, model_paths: ModelPaths):
    """Load pre-trained GBM models."""
    gbm_paths = model_paths.gbm_paths
    
    models = {}
    
    # Load XGBoost
    xgb_path = gbm_paths.xgb_model
    if xgb_path.exists():
        try:
            import xgboost as xgb
            model = xgb.XGBRegressor()
            model.load_model(str(xgb_path))
            models['xgb'] = model
            logger.info(f"✓ Loaded XGBoost model from {xgb_path}")
        except Exception as e:
            logger.warning(f"Failed to load XGBoost: {e}")
    
    # Load LightGBM
    lgb_path = gbm_paths.lgb_model
    if lgb_path.exists():
        try:
            import lightgbm as lgb
            model = lgb.Booster(model_file=str(lgb_path))
            models['lgb'] = model
            logger.info(f"✓ Loaded LightGBM model from {lgb_path}")
        except Exception as e:
            logger.warning(f"Failed to load LightGBM: {e}")
    
    # Load feature scaler
    scaler_path = gbm_paths.feature_scaler
    scaler = None
    if scaler_path.exists():
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logger.info(f"✓ Loaded GBM feature scaler from {scaler_path}")
    
    return models, scaler


def create_sequences_for_lstm(X: np.ndarray, sequence_length: int = 10) -> np.ndarray:
    """Create sequences for LSTM input."""
    if len(X) < sequence_length:
        return np.array([])
    
    sequences = []
    for i in range(len(X) - sequence_length + 1):
        sequences.append(X[i:i + sequence_length])
    
    return np.array(sequences)


def predict_lstm(model, X_seq: np.ndarray, target_scaler=None) -> np.ndarray:
    """Generate predictions from LSTM model."""
    import tensorflow as tf
    
    if X_seq.shape[0] == 0:
        return np.array([])
    
    try:
        # Handle different model types (Keras vs SavedModel)
        if hasattr(model, 'predict'):
            preds = model.predict(X_seq, verbose=0)
        elif hasattr(model, 'signatures'):
            # SavedModel format
            infer = model.signatures['serving_default']
            preds = infer(tf.constant(X_seq, dtype=tf.float32))
            preds = list(preds.values())[0].numpy()
        else:
            raise ValueError("Unknown model type")
        
        # Inverse transform if scaler provided
        if target_scaler is not None and hasattr(target_scaler, 'inverse_transform'):
            preds = target_scaler.inverse_transform(preds.reshape(-1, 1)).flatten()
        else:
            preds = preds.flatten()
        
        return preds
    except Exception as e:
        logger.error(f"LSTM prediction failed: {e}")
        return np.array([])


def predict_gbm(models: Dict, X: np.ndarray) -> Dict[str, np.ndarray]:
    """Generate predictions from GBM models."""
    predictions = {}
    
    for name, model in models.items():
        try:
            if name == 'lgb' and hasattr(model, 'predict'):
                # LightGBM Booster
                preds = model.predict(X)
            elif hasattr(model, 'predict'):
                # XGBoost or sklearn-style API
                preds = model.predict(X)
            else:
                continue
            predictions[name] = preds
        except Exception as e:
            logger.warning(f"GBM {name} prediction failed: {e}")
    
    return predictions


# ============================================================================
# OOF GENERATION
# ============================================================================

def generate_oof_predictions(
    symbol: str,
    n_splits: int = 5,
    gap: int = 1,
    min_train_size: int = 252,
    sequence_length: int = 10,
    save_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Generate out-of-fold predictions for LSTM and GBM models.
    
    Uses TimeSeriesSplit to ensure no look-ahead bias.
    
    Args:
        symbol: Stock ticker symbol
        n_splits: Number of cross-validation folds
        gap: Gap between train and test sets
        min_train_size: Minimum samples in training set
        sequence_length: LSTM sequence length
        save_dir: Directory to save OOF predictions
    
    Returns:
        Dictionary with OOF predictions and metadata
    """
    configure_gpu()
    
    logger.info(f"="*60)
    logger.info(f"Generating OOF Predictions for {symbol}")
    logger.info(f"n_splits={n_splits}, gap={gap}, min_train={min_train_size}")
    logger.info(f"="*60)
    
    # =========================================================================
    # 1. Load data and engineer features
    # =========================================================================
    logger.info("\n📊 Loading and preparing data...")
    
    df = fetch_stock_data(symbol, period="10y", use_cache=True)
    if df is None or df.empty:
        raise ValueError(f"Failed to fetch data for {symbol}")
    
    logger.info(f"   Raw data: {len(df)} rows, date range: {df.index[0]} to {df.index[-1]}")
    
    # Engineer features
    df = engineer_features(df, include_sentiment=True)
    feature_cols = get_feature_columns()
    
    # Prepare target
    df, metadata = prepare_training_data(df, horizon=1)
    target_col = 'target_1d'
    
    # Drop rows with NaN in features or target
    df = df.dropna(subset=feature_cols + [target_col])
    
    logger.info(f"   After feature engineering: {len(df)} rows")
    logger.info(f"   Feature columns: {len(feature_cols)}")
    
    # Extract arrays
    X = df[feature_cols].values
    y = df[target_col].values
    dates = df.index.values
    
    # =========================================================================
    # 2. Load pre-trained models
    # =========================================================================
    logger.info("\n📦 Loading pre-trained models...")
    
    model_paths = ModelPaths(symbol)
    
    # Load LSTM
    lstm_model, lstm_feature_scaler, lstm_target_scaler = load_lstm_model(symbol, model_paths)
    lstm_available = lstm_model is not None
    
    # Load GBM
    gbm_models, gbm_scaler = load_gbm_models(symbol, model_paths)
    gbm_available = len(gbm_models) > 0
    
    if not lstm_available and not gbm_available:
        raise ValueError("No models available. Train LSTM and/or GBM models first.")
    
    logger.info(f"   LSTM available: {lstm_available}")
    logger.info(f"   GBM available: {gbm_available} (models: {list(gbm_models.keys())})")
    
    # =========================================================================
    # 3. Initialize OOF arrays
    # =========================================================================
    n_samples = len(y)
    
    oof_lstm = np.full(n_samples, np.nan)
    oof_xgb = np.full(n_samples, np.nan)
    oof_lgb = np.full(n_samples, np.nan)
    
    fold_metrics = []
    
    # =========================================================================
    # 4. Time-series cross-validation
    # =========================================================================
    logger.info(f"\n🔄 Running {n_splits}-fold time-series cross-validation...")
    
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=gap)
    
    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X)):
        logger.info(f"\n--- Fold {fold_idx + 1}/{n_splits} ---")
        
        # Skip if training set too small
        if len(train_idx) < min_train_size:
            logger.warning(f"   Skipping: train size {len(train_idx)} < {min_train_size}")
            continue
        
        # Get train/test data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        logger.info(f"   Train: {len(train_idx)} samples ({dates[train_idx[0]]} to {dates[train_idx[-1]]})")
        logger.info(f"   Test:  {len(test_idx)} samples ({dates[test_idx[0]]} to {dates[test_idx[-1]]})")
        
        fold_result = {
            'fold': fold_idx + 1,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
        }
        
        # ---------------------------------------------------------------------
        # LSTM Predictions
        # ---------------------------------------------------------------------
        if lstm_available:
            try:
                # Scale features
                if lstm_feature_scaler is not None:
                    X_test_scaled = lstm_feature_scaler.transform(X_test)
                else:
                    X_test_scaled = X_test
                
                # Create sequences
                X_test_seq = create_sequences_for_lstm(X_test_scaled, sequence_length)
                
                # Adjust test indices for sequence offset
                seq_offset = sequence_length - 1
                valid_test_idx = test_idx[seq_offset:]
                
                if len(X_test_seq) > 0:
                    preds = predict_lstm(lstm_model, X_test_seq, lstm_target_scaler)
                    
                    if len(preds) == len(valid_test_idx):
                        oof_lstm[valid_test_idx] = preds
                        fold_result['lstm_mean'] = float(np.mean(preds))
                        fold_result['lstm_std'] = float(np.std(preds))
                        logger.info(f"   LSTM: mean={fold_result['lstm_mean']:.6f}, std={fold_result['lstm_std']:.6f}")
                    else:
                        logger.warning(f"   LSTM: prediction length mismatch ({len(preds)} vs {len(valid_test_idx)})")
            except Exception as e:
                logger.error(f"   LSTM prediction failed: {e}")
        
        # ---------------------------------------------------------------------
        # GBM Predictions (XGBoost, LightGBM)
        # ---------------------------------------------------------------------
        if gbm_available:
            try:
                # Scale features if scaler available
                if gbm_scaler is not None:
                    X_test_gbm = gbm_scaler.transform(X_test)
                else:
                    X_test_gbm = X_test
                
                gbm_preds = predict_gbm(gbm_models, X_test_gbm)
                
                if 'xgb' in gbm_preds:
                    oof_xgb[test_idx] = gbm_preds['xgb']
                    fold_result['xgb_mean'] = float(np.mean(gbm_preds['xgb']))
                    fold_result['xgb_std'] = float(np.std(gbm_preds['xgb']))
                    logger.info(f"   XGB: mean={fold_result['xgb_mean']:.6f}, std={fold_result['xgb_std']:.6f}")
                
                if 'lgb' in gbm_preds:
                    oof_lgb[test_idx] = gbm_preds['lgb']
                    fold_result['lgb_mean'] = float(np.mean(gbm_preds['lgb']))
                    fold_result['lgb_std'] = float(np.std(gbm_preds['lgb']))
                    logger.info(f"   LGB: mean={fold_result['lgb_mean']:.6f}, std={fold_result['lgb_std']:.6f}")
            except Exception as e:
                logger.error(f"   GBM prediction failed: {e}")
        
        fold_metrics.append(fold_result)
    
    # =========================================================================
    # 5. Aggregate OOF statistics
    # =========================================================================
    logger.info("\n📈 OOF Prediction Statistics:")
    
    # Mask for valid (non-NaN) predictions
    valid_mask = ~np.isnan(oof_lstm) | ~np.isnan(oof_xgb) | ~np.isnan(oof_lgb)
    n_valid = np.sum(valid_mask)
    
    logger.info(f"   Valid OOF samples: {n_valid}/{n_samples} ({100*n_valid/n_samples:.1f}%)")
    
    # LSTM stats
    lstm_valid = ~np.isnan(oof_lstm)
    if np.any(lstm_valid):
        lstm_stats = {
            'count': int(np.sum(lstm_valid)),
            'mean': float(np.nanmean(oof_lstm)),
            'std': float(np.nanstd(oof_lstm)),
            'min': float(np.nanmin(oof_lstm)),
            'max': float(np.nanmax(oof_lstm)),
        }
        logger.info(f"   LSTM OOF: n={lstm_stats['count']}, mean={lstm_stats['mean']:.6f}, std={lstm_stats['std']:.6f}")
    else:
        lstm_stats = None
    
    # XGBoost stats
    xgb_valid = ~np.isnan(oof_xgb)
    if np.any(xgb_valid):
        xgb_stats = {
            'count': int(np.sum(xgb_valid)),
            'mean': float(np.nanmean(oof_xgb)),
            'std': float(np.nanstd(oof_xgb)),
            'min': float(np.nanmin(oof_xgb)),
            'max': float(np.nanmax(oof_xgb)),
        }
        logger.info(f"   XGB OOF:  n={xgb_stats['count']}, mean={xgb_stats['mean']:.6f}, std={xgb_stats['std']:.6f}")
    else:
        xgb_stats = None
    
    # LightGBM stats
    lgb_valid = ~np.isnan(oof_lgb)
    if np.any(lgb_valid):
        lgb_stats = {
            'count': int(np.sum(lgb_valid)),
            'mean': float(np.nanmean(oof_lgb)),
            'std': float(np.nanstd(oof_lgb)),
            'min': float(np.nanmin(oof_lgb)),
            'max': float(np.nanmax(oof_lgb)),
        }
        logger.info(f"   LGB OOF:  n={lgb_stats['count']}, mean={lgb_stats['mean']:.6f}, std={lgb_stats['std']:.6f}")
    else:
        lgb_stats = None
    
    # =========================================================================
    # 6. Save OOF predictions
    # =========================================================================
    if save_dir is None:
        save_dir = PROJECT_ROOT / 'saved_models' / symbol / 'oof'
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Create OOF dataframe
    oof_df = pd.DataFrame({
        'date': dates,
        'y_true': y,
        'oof_lstm': oof_lstm,
        'oof_xgb': oof_xgb,
        'oof_lgb': oof_lgb,
    })
    
    # Save as CSV and pickle
    oof_csv_path = save_dir / 'oof_predictions.csv'
    oof_df.to_csv(oof_csv_path, index=False)
    logger.info(f"\n💾 Saved OOF predictions to {oof_csv_path}")
    
    oof_pkl_path = save_dir / 'oof_predictions.pkl'
    with open(oof_pkl_path, 'wb') as f:
        pickle.dump(oof_df, f)
    logger.info(f"   Saved OOF pickle to {oof_pkl_path}")
    
    # Save metadata
    metadata = {
        'symbol': symbol,
        'generated_at': datetime.now().isoformat(),
        'n_splits': n_splits,
        'gap': gap,
        'min_train_size': min_train_size,
        'sequence_length': sequence_length,
        'n_samples': n_samples,
        'n_valid_oof': int(n_valid),
        'lstm_stats': lstm_stats,
        'xgb_stats': xgb_stats,
        'lgb_stats': lgb_stats,
        'fold_metrics': fold_metrics,
    }
    
    metadata_path = save_dir / 'oof_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"   Saved metadata to {metadata_path}")
    
    logger.info(f"\n✅ OOF generation complete!")
    
    return {
        'oof_df': oof_df,
        'metadata': metadata,
        'save_dir': str(save_dir),
    }


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Generate Out-of-Fold predictions for stacking ensemble'
    )
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        default='AAPL',
        help='Stock symbol (default: AAPL)'
    )
    parser.add_argument(
        '--n-splits', '-n',
        type=int,
        default=5,
        help='Number of CV folds (default: 5)'
    )
    parser.add_argument(
        '--gap', '-g',
        type=int,
        default=1,
        help='Gap between train/test (default: 1)'
    )
    parser.add_argument(
        '--min-train', '-m',
        type=int,
        default=252,
        help='Minimum training samples (default: 252)'
    )
    parser.add_argument(
        '--sequence-length', '-l',
        type=int,
        default=10,
        help='LSTM sequence length (default: 10)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory for OOF predictions'
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    try:
        result = generate_oof_predictions(
            symbol=args.symbol,
            n_splits=args.n_splits,
            gap=args.gap,
            min_train_size=args.min_train,
            sequence_length=args.sequence_length,
            save_dir=output_dir,
        )
        
        print(f"\n{'='*60}")
        print(f"SUCCESS: OOF predictions saved to {result['save_dir']}")
        print(f"{'='*60}")
        
        return 0
    except Exception as e:
        logger.error(f"OOF generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
