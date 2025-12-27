"""
Meta-Learner Training for Stacking Ensemble

This script trains a meta-learner (Ridge regression) on out-of-fold predictions
from LSTM and GBM models to create a stacking ensemble.

Key Features:
- Ridge regression with cross-validated alpha selection
- Handles missing OOF predictions gracefully
- Saves model weights and diagnostics
- Computes directional accuracy improvement

Based on:
- Wolpert (1992): "Stacked Generalization"
- Hastie et al. (2009): "Elements of Statistical Learning" - Stacking chapter

Usage:
    python training/train_meta_learner.py --symbol AAPL --alpha 1.0
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
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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
    'alpha': 1.0,              # Ridge regularization strength
    'alpha_range': [0.01, 0.1, 1.0, 10.0, 100.0],  # For CV search
    'fit_intercept': True,
    'min_valid_samples': 100,  # Minimum samples needed for training
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_oof_predictions(symbol: str, oof_dir: Optional[Path] = None) -> pd.DataFrame:
    """Load OOF predictions from saved files."""
    if oof_dir is None:
        oof_dir = PROJECT_ROOT / 'saved_models' / symbol / 'oof'
    
    oof_pkl_path = oof_dir / 'oof_predictions.pkl'
    if oof_pkl_path.exists():
        with open(oof_pkl_path, 'rb') as f:
            oof_df = pickle.load(f)
        logger.info(f"✓ Loaded OOF predictions from {oof_pkl_path}")
        return oof_df
    
    oof_csv_path = oof_dir / 'oof_predictions.csv'
    if oof_csv_path.exists():
        oof_df = pd.read_csv(oof_csv_path)
        logger.info(f"✓ Loaded OOF predictions from {oof_csv_path}")
        return oof_df
    
    raise FileNotFoundError(f"No OOF predictions found in {oof_dir}")


def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate directional accuracy (sign agreement)."""
    # Only evaluate where both are non-zero
    nonzero_mask = (y_true != 0) & (y_pred != 0)
    if np.sum(nonzero_mask) == 0:
        return 0.5
    
    sign_match = np.sign(y_true[nonzero_mask]) == np.sign(y_pred[nonzero_mask])
    return float(np.mean(sign_match))


def compute_model_weights(oof_df: pd.DataFrame, target_col: str = 'y_true') -> Dict[str, float]:
    """
    Compute optimal model weights based on OOF performance.
    
    Uses inverse MSE weighting: better models get higher weights.
    """
    model_cols = ['oof_lstm', 'oof_xgb', 'oof_lgb']
    y_true = oof_df[target_col].values
    
    mse_scores = {}
    for col in model_cols:
        if col in oof_df.columns:
            valid_mask = ~np.isnan(oof_df[col].values)
            if np.sum(valid_mask) > 10:
                y_pred = oof_df[col].values[valid_mask]
                y_t = y_true[valid_mask]
                mse = mean_squared_error(y_t, y_pred)
                mse_scores[col] = mse
    
    if not mse_scores:
        return {}
    
    # Inverse MSE weighting
    inv_mse = {k: 1.0 / (v + 1e-10) for k, v in mse_scores.items()}
    total_inv = sum(inv_mse.values())
    weights = {k: v / total_inv for k, v in inv_mse.items()}
    
    return weights


# ============================================================================
# META-LEARNER TRAINING
# ============================================================================

def train_meta_learner(
    symbol: str,
    alpha: Optional[float] = None,
    cv_alpha: bool = True,
    oof_dir: Optional[Path] = None,
    save_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Train a Ridge regression meta-learner on OOF predictions.
    
    Args:
        symbol: Stock ticker symbol
        alpha: Ridge regularization parameter (if None, uses CV)
        cv_alpha: Whether to cross-validate alpha selection
        oof_dir: Directory containing OOF predictions
        save_dir: Directory to save trained meta-learner
    
    Returns:
        Dictionary with trained model and metadata
    """
    logger.info(f"="*60)
    logger.info(f"Training Meta-Learner for {symbol}")
    logger.info(f"="*60)
    
    # =========================================================================
    # 1. Load OOF predictions
    # =========================================================================
    logger.info("\n📊 Loading OOF predictions...")
    
    oof_df = load_oof_predictions(symbol, oof_dir)
    logger.info(f"   Total samples: {len(oof_df)}")
    
    # Identify available OOF columns
    model_cols = []
    for col in ['oof_lstm', 'oof_xgb', 'oof_lgb']:
        if col in oof_df.columns:
            valid_count = (~np.isnan(oof_df[col])).sum()
            if valid_count > 0:
                model_cols.append(col)
                logger.info(f"   {col}: {valid_count} valid predictions")
    
    if len(model_cols) < 2:
        logger.warning("⚠️ Need at least 2 models for stacking. Using simple weighted average.")
    
    # =========================================================================
    # 2. Prepare training data
    # =========================================================================
    logger.info("\n🔧 Preparing training data...")
    
    # Get target
    y_true = oof_df['y_true'].values
    
    # Build feature matrix (only rows where all models have predictions)
    valid_mask = np.ones(len(oof_df), dtype=bool)
    for col in model_cols:
        valid_mask &= ~np.isnan(oof_df[col].values)
    
    n_valid = np.sum(valid_mask)
    logger.info(f"   Valid samples (all models have predictions): {n_valid}")
    
    if n_valid < DEFAULT_CONFIG['min_valid_samples']:
        # Fall back to using available predictions with mean imputation
        logger.warning(f"   Only {n_valid} complete samples. Using mean imputation.")
        
        X_train_list = []
        for col in model_cols:
            values = oof_df[col].values.copy()
            nan_mask = np.isnan(values)
            if np.any(nan_mask):
                values[nan_mask] = np.nanmean(values)
            X_train_list.append(values)
        
        X_train = np.column_stack(X_train_list)
        y_train = y_true
        valid_mask = ~np.isnan(y_train)
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]
    else:
        # Use only complete samples
        X_train = np.column_stack([oof_df[col].values[valid_mask] for col in model_cols])
        y_train = y_true[valid_mask]
    
    logger.info(f"   Training samples: {len(y_train)}")
    logger.info(f"   Features: {X_train.shape[1]} ({', '.join(model_cols)})")
    
    # =========================================================================
    # 3. Train Ridge meta-learner
    # =========================================================================
    logger.info("\n🏋️ Training Ridge meta-learner...")
    
    if cv_alpha and alpha is None:
        # Cross-validated alpha selection
        alphas = DEFAULT_CONFIG['alpha_range']
        tscv = TimeSeriesSplit(n_splits=3, gap=1)
        
        meta_learner = RidgeCV(
            alphas=alphas,
            fit_intercept=DEFAULT_CONFIG['fit_intercept'],
            cv=tscv,
        )
        meta_learner.fit(X_train, y_train)
        
        best_alpha = meta_learner.alpha_
        logger.info(f"   CV selected alpha: {best_alpha}")
    else:
        # Use provided alpha
        if alpha is None:
            alpha = DEFAULT_CONFIG['alpha']
        
        meta_learner = Ridge(
            alpha=alpha,
            fit_intercept=DEFAULT_CONFIG['fit_intercept'],
        )
        meta_learner.fit(X_train, y_train)
        best_alpha = alpha
    
    # Log model coefficients
    logger.info(f"\n   Meta-learner weights:")
    for i, col in enumerate(model_cols):
        logger.info(f"     {col}: {meta_learner.coef_[i]:.4f}")
    logger.info(f"     intercept: {meta_learner.intercept_:.6f}")
    
    # =========================================================================
    # 4. Evaluate meta-learner
    # =========================================================================
    logger.info("\n📊 Evaluating meta-learner...")
    
    # Predictions on training data
    y_pred_meta = meta_learner.predict(X_train)
    
    # Metrics for meta-learner
    meta_mse = mean_squared_error(y_train, y_pred_meta)
    meta_mae = mean_absolute_error(y_train, y_pred_meta)
    meta_r2 = r2_score(y_train, y_pred_meta)
    meta_dir_acc = directional_accuracy(y_train, y_pred_meta)
    
    logger.info(f"   Meta-learner:")
    logger.info(f"     MSE: {meta_mse:.8f}")
    logger.info(f"     MAE: {meta_mae:.6f}")
    logger.info(f"     R²:  {meta_r2:.4f}")
    logger.info(f"     Directional Accuracy: {meta_dir_acc:.2%}")
    
    # Compare with individual models
    logger.info(f"\n   Individual model comparison:")
    individual_metrics = {}
    for i, col in enumerate(model_cols):
        y_pred_ind = X_train[:, i]
        ind_mse = mean_squared_error(y_train, y_pred_ind)
        ind_mae = mean_absolute_error(y_train, y_pred_ind)
        ind_dir_acc = directional_accuracy(y_train, y_pred_ind)
        individual_metrics[col] = {
            'mse': float(ind_mse),
            'mae': float(ind_mae),
            'dir_acc': float(ind_dir_acc),
        }
        logger.info(f"     {col}: MSE={ind_mse:.8f}, MAE={ind_mae:.6f}, Dir={ind_dir_acc:.2%}")
    
    # Compute improvement
    best_individual_mse = min(m['mse'] for m in individual_metrics.values())
    improvement = (best_individual_mse - meta_mse) / best_individual_mse * 100
    logger.info(f"\n   Meta-learner MSE improvement over best individual: {improvement:.1f}%")
    
    # =========================================================================
    # 5. Save meta-learner
    # =========================================================================
    if save_dir is None:
        save_dir = PROJECT_ROOT / 'saved_models' / symbol / 'meta_learner'
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = save_dir / 'meta_learner.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump(meta_learner, f)
    logger.info(f"\n💾 Saved meta-learner to {model_path}")
    
    # Save metadata
    metadata = {
        'symbol': symbol,
        'trained_at': datetime.now().isoformat(),
        'alpha': float(best_alpha),
        'cv_alpha': cv_alpha,
        'n_training_samples': int(len(y_train)),
        'feature_columns': model_cols,
        'coefficients': {col: float(meta_learner.coef_[i]) for i, col in enumerate(model_cols)},
        'intercept': float(meta_learner.intercept_),
        'metrics': {
            'mse': float(meta_mse),
            'mae': float(meta_mae),
            'r2': float(meta_r2),
            'directional_accuracy': float(meta_dir_acc),
        },
        'individual_metrics': individual_metrics,
        'improvement_over_best': float(improvement),
    }
    
    metadata_path = save_dir / 'meta_learner_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"   Saved metadata to {metadata_path}")
    
    # Save feature column names (needed for inference)
    cols_path = save_dir / 'feature_columns.json'
    with open(cols_path, 'w') as f:
        json.dump(model_cols, f)
    logger.info(f"   Saved feature columns to {cols_path}")
    
    logger.info(f"\n✅ Meta-learner training complete!")
    
    return {
        'meta_learner': meta_learner,
        'metadata': metadata,
        'save_dir': str(save_dir),
    }


# ============================================================================
# MAIN
# ============================================================================

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train meta-learner for stacking ensemble'
    )
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        default='AAPL',
        help='Stock symbol (default: AAPL)'
    )
    parser.add_argument(
        '--alpha', '-a',
        type=float,
        default=None,
        help='Ridge regularization strength (default: CV selected)'
    )
    parser.add_argument(
        '--no-cv',
        action='store_true',
        help='Disable cross-validation for alpha selection'
    )
    parser.add_argument(
        '--oof-dir',
        type=str,
        default=None,
        help='Directory containing OOF predictions'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default=None,
        help='Output directory for meta-learner'
    )
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    oof_dir = Path(args.oof_dir) if args.oof_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    try:
        result = train_meta_learner(
            symbol=args.symbol,
            alpha=args.alpha,
            cv_alpha=not args.no_cv,
            oof_dir=oof_dir,
            save_dir=output_dir,
        )
        
        print(f"\n{'='*60}")
        print(f"SUCCESS: Meta-learner saved to {result['save_dir']}")
        print(f"Improvement: {result['metadata']['improvement_over_best']:.1f}% over best individual model")
        print(f"{'='*60}")
        
        return 0
    except Exception as e:
        logger.error(f"Meta-learner training failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
