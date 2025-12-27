#!/usr/bin/env python
"""
Rolling-Window Cross-Validation for Time Series Models

P1.1: Implements proper time-series aware cross-validation to prevent
look-ahead bias and detect overfitting.

Key features:
- Expanding window (more training data each fold)
- Rolling window (fixed training window)
- Gap between train and test to prevent leakage
- Purging to remove overlapping sequences

References:
- Bergmeir et al. (2018): "A note on the validity of cross-validation 
  for evaluating autoregressive time series prediction"
- de Prado (2018): "Advances in Financial Machine Learning", Chapter 7

Usage:
    from training.rolling_cv import RollingWindowCV, ExpandingWindowCV
    
    cv = RollingWindowCV(n_splits=5, test_size=60, gap=1)
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        # ... train and evaluate
"""

import numpy as np
from typing import Iterator, Tuple, Optional, List
import logging

logger = logging.getLogger(__name__)


class RollingWindowCV:
    """
    Rolling-window cross-validation for time series.
    
    Uses a fixed-size training window that rolls forward in time.
    Each fold uses the same amount of training data, but from
    progressively later time periods.
    
    Example with n_splits=3, train_size=100, test_size=20, gap=1:
        Fold 1: train [0:100], gap [100:101], test [101:121]
        Fold 2: train [20:120], gap [120:121], test [121:141]
        Fold 3: train [40:140], gap [140:141], test [141:161]
    
    Args:
        n_splits: Number of cross-validation folds
        test_size: Number of samples in each test set
        train_size: Size of training window (None = expanding window)
        gap: Number of samples to skip between train and test (prevents leakage)
        purge: Additional samples to remove at train/test boundary for sequences
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = 60,
        train_size: Optional[int] = None,
        gap: int = 1,
        purge: int = 0
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.train_size = train_size
        self.gap = gap
        self.purge = purge
    
    def split(self, X: np.ndarray, y: np.ndarray = None, groups: np.ndarray = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """
        Generate train/test split indices.
        
        Args:
            X: Feature array (n_samples, n_features) or just n_samples
            y: Target array (ignored, for sklearn compatibility)
            groups: Group labels (ignored, for sklearn compatibility)
        
        Yields:
            (train_indices, test_indices) for each fold
        """
        n_samples = len(X) if hasattr(X, '__len__') else X
        
        # Calculate step size between folds
        # Total samples needed: train_size + gap + purge + test_size * n_splits
        # But we want folds to overlap test sets minimally
        if self.train_size is not None:
            # Rolling window: fixed train size
            min_required = self.train_size + self.gap + self.purge + self.test_size * self.n_splits
            if n_samples < min_required:
                logger.warning(f"Not enough samples ({n_samples}) for {self.n_splits} folds "
                             f"with train_size={self.train_size}, test_size={self.test_size}")
            
            step = self.test_size  # Each fold shifts by test_size
            
            for fold in range(self.n_splits):
                # Test set position (from end)
                test_end = n_samples - fold * step
                test_start = test_end - self.test_size
                
                # Train set position (fixed size before test)
                train_end = test_start - self.gap - self.purge
                train_start = train_end - self.train_size
                
                if train_start < 0:
                    break  # Not enough data
                
                train_idx = np.arange(train_start, train_end)
                test_idx = np.arange(test_start, test_end)
                
                yield train_idx, test_idx
        else:
            # Expanding window: train size grows with each fold
            # Work backwards from end of data
            step = self.test_size
            min_train = 200  # Minimum training samples
            
            for fold in range(self.n_splits):
                # Test set position
                test_end = n_samples - fold * step
                test_start = test_end - self.test_size
                
                # Train set: from 0 to just before test (with gap)
                train_end = test_start - self.gap - self.purge
                train_start = 0
                
                if train_end - train_start < min_train:
                    break  # Not enough training data
                
                train_idx = np.arange(train_start, train_end)
                test_idx = np.arange(test_start, test_end)
                
                yield train_idx, test_idx
    
    def get_n_splits(self) -> int:
        """Return the number of splits."""
        return self.n_splits


class ExpandingWindowCV:
    """
    Expanding-window cross-validation for time series.
    
    Training set grows with each fold (anchored at start).
    This is the most conservative approach for financial data.
    
    Example with n_splits=3, test_size=20, gap=1:
        Fold 1: train [0:100], gap [100:101], test [101:121]
        Fold 2: train [0:121], gap [121:122], test [122:142]
        Fold 3: train [0:142], gap [142:143], test [143:163]
    
    Args:
        n_splits: Number of folds
        test_size: Samples per test set
        min_train_size: Minimum initial training size
        gap: Samples between train end and test start
        purge: Additional purge samples for sequence models
    """
    
    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = 60,
        min_train_size: int = 252,  # ~1 year of trading days
        gap: int = 1,
        purge: int = 0
    ):
        self.n_splits = n_splits
        self.test_size = test_size
        self.min_train_size = min_train_size
        self.gap = gap
        self.purge = purge
    
    def split(self, X: np.ndarray, y: np.ndarray = None, groups: np.ndarray = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate expanding window splits."""
        n_samples = len(X) if hasattr(X, '__len__') else X
        
        # Calculate spacing between folds
        # Reserve: min_train + gap + purge + n_splits * test_size
        reserved = self.min_train_size + self.gap + self.purge + self.n_splits * self.test_size
        
        if n_samples < reserved:
            logger.warning(f"Not enough samples ({n_samples}) for {self.n_splits} folds")
        
        # Work forward from min_train_size
        for fold in range(self.n_splits):
            # Train: from 0 to expanding end
            extra_train = fold * self.test_size
            train_end = self.min_train_size + extra_train
            train_start = 0
            
            # Test: after gap
            test_start = train_end + self.gap + self.purge
            test_end = test_start + self.test_size
            
            if test_end > n_samples:
                break  # Not enough data
            
            train_idx = np.arange(train_start, train_end)
            test_idx = np.arange(test_start, test_end)
            
            yield train_idx, test_idx
    
    def get_n_splits(self) -> int:
        return self.n_splits


class PurgedKFoldCV:
    """
    K-Fold CV with purging for overlapping sequences.
    
    When using sequence models (LSTM), adjacent samples share overlapping
    data due to the lookback window. This can cause data leakage if a
    sample's lookback window overlaps with test data.
    
    Purging removes samples from training that have any overlap with test.
    
    Args:
        n_splits: Number of folds
        purge_length: Number of samples to purge (typically sequence_length)
    """
    
    def __init__(self, n_splits: int = 5, purge_length: int = 60):
        self.n_splits = n_splits
        self.purge_length = purge_length
    
    def split(self, X: np.ndarray, y: np.ndarray = None, groups: np.ndarray = None) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Generate purged K-Fold splits."""
        n_samples = len(X) if hasattr(X, '__len__') else X
        
        indices = np.arange(n_samples)
        fold_size = n_samples // self.n_splits
        
        for fold in range(self.n_splits):
            # Test set for this fold
            test_start = fold * fold_size
            test_end = (fold + 1) * fold_size if fold < self.n_splits - 1 else n_samples
            test_idx = indices[test_start:test_end]
            
            # Train set: all indices except test and purge zones
            # Purge zone: purge_length before test_start and after test_end
            purge_before_start = max(0, test_start - self.purge_length)
            purge_after_end = min(n_samples, test_end + self.purge_length)
            
            # Train indices: before purge_before_start AND after purge_after_end
            train_before = indices[:purge_before_start]
            train_after = indices[purge_after_end:]
            train_idx = np.concatenate([train_before, train_after])
            
            if len(train_idx) < 100:  # Minimum training samples
                continue
            
            yield train_idx, test_idx
    
    def get_n_splits(self) -> int:
        return self.n_splits


# ============================================================================
# TRAINING UTILITIES
# ============================================================================

def train_with_rolling_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_factory: callable,
    cv_splitter,
    fit_kwargs: dict = None,
    return_models: bool = False
) -> dict:
    """
    Train model with cross-validation and return out-of-fold predictions.
    
    Args:
        X: Feature array (n_samples, n_features) or (n_samples, seq_len, n_features)
        y: Target array (n_samples,)
        model_factory: Callable that returns a fresh model instance
        cv_splitter: CV splitter with split() method
        fit_kwargs: Additional kwargs for model.fit()
        return_models: Whether to return trained models
    
    Returns:
        Dict with OOF predictions, fold metrics, and optionally models
    """
    fit_kwargs = fit_kwargs or {}
    
    n_samples = len(X)
    oof_predictions = np.full(n_samples, np.nan)
    fold_metrics = []
    models = [] if return_models else None
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv_splitter.split(X)):
        logger.info(f"Fold {fold_idx + 1}/{cv_splitter.get_n_splits()}")
        logger.info(f"  Train: {len(train_idx)} samples [{train_idx[0]}-{train_idx[-1]}]")
        logger.info(f"  Test:  {len(test_idx)} samples [{test_idx[0]}-{test_idx[-1]}]")
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Create and train model
        model = model_factory()
        model.fit(X_train, y_train, **fit_kwargs)
        
        # Predict
        y_pred = model.predict(X_test)
        if y_pred.ndim > 1:
            y_pred = y_pred.flatten()
        
        # Store OOF predictions
        oof_predictions[test_idx] = y_pred
        
        # Calculate fold metrics
        mse = np.mean((y_test - y_pred) ** 2)
        mae = np.mean(np.abs(y_test - y_pred))
        
        # Direction accuracy
        if y_test.std() > 0 and y_pred.std() > 0:
            dir_acc = np.mean(np.sign(y_test) == np.sign(y_pred))
            corr = np.corrcoef(y_test, y_pred)[0, 1]
        else:
            dir_acc = 0.5
            corr = 0.0
        
        fold_result = {
            'fold': fold_idx + 1,
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'mse': float(mse),
            'mae': float(mae),
            'dir_acc': float(dir_acc),
            'correlation': float(corr),
            'pred_std': float(np.std(y_pred)),
        }
        fold_metrics.append(fold_result)
        
        logger.info(f"  MSE: {mse:.6f}, MAE: {mae:.6f}, Dir Acc: {dir_acc:.4f}, Corr: {corr:.4f}")
        
        if return_models:
            models.append(model)
    
    # Aggregate metrics
    valid_folds = [f for f in fold_metrics if f['mse'] is not None]
    agg_metrics = {
        'mse_mean': np.mean([f['mse'] for f in valid_folds]),
        'mse_std': np.std([f['mse'] for f in valid_folds]),
        'mae_mean': np.mean([f['mae'] for f in valid_folds]),
        'mae_std': np.std([f['mae'] for f in valid_folds]),
        'dir_acc_mean': np.mean([f['dir_acc'] for f in valid_folds]),
        'dir_acc_std': np.std([f['dir_acc'] for f in valid_folds]),
        'correlation_mean': np.mean([f['correlation'] for f in valid_folds]),
        'correlation_std': np.std([f['correlation'] for f in valid_folds]),
    }
    
    result = {
        'oof_predictions': oof_predictions,
        'fold_metrics': fold_metrics,
        'aggregate_metrics': agg_metrics,
        'n_folds': len(valid_folds),
    }
    
    if return_models:
        result['models'] = models
    
    # Log summary
    logger.info(f"\n=== CV Summary ({len(valid_folds)} folds) ===")
    logger.info(f"MSE: {agg_metrics['mse_mean']:.6f} ± {agg_metrics['mse_std']:.6f}")
    logger.info(f"MAE: {agg_metrics['mae_mean']:.6f} ± {agg_metrics['mae_std']:.6f}")
    logger.info(f"Dir Acc: {agg_metrics['dir_acc_mean']:.4f} ± {agg_metrics['dir_acc_std']:.4f}")
    logger.info(f"Corr: {agg_metrics['correlation_mean']:.4f} ± {agg_metrics['correlation_std']:.4f}")
    
    return result


def compute_cv_metrics_summary(fold_metrics: List[dict]) -> dict:
    """
    Compute summary statistics from fold metrics.
    
    Args:
        fold_metrics: List of dicts with per-fold metrics
    
    Returns:
        Dict with mean, std, min, max for each metric
    """
    if not fold_metrics:
        return {}
    
    # Get all numeric keys
    numeric_keys = [k for k, v in fold_metrics[0].items() 
                    if isinstance(v, (int, float)) and k != 'fold']
    
    summary = {}
    for key in numeric_keys:
        values = [f[key] for f in fold_metrics if key in f and f[key] is not None]
        if values:
            summary[f'{key}_mean'] = float(np.mean(values))
            summary[f'{key}_std'] = float(np.std(values))
            summary[f'{key}_min'] = float(np.min(values))
            summary[f'{key}_max'] = float(np.max(values))
    
    return summary


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def create_time_series_cv(
    cv_type: str = 'expanding',
    n_splits: int = 5,
    test_size: int = 60,
    gap: int = 1,
    **kwargs
):
    """
    Factory function to create CV splitter.
    
    Args:
        cv_type: 'expanding', 'rolling', or 'purged'
        n_splits: Number of folds
        test_size: Test set size per fold
        gap: Gap between train and test
        **kwargs: Additional arguments for specific CV types
    
    Returns:
        CV splitter instance
    """
    cv_type = cv_type.lower()
    
    if cv_type == 'expanding':
        return ExpandingWindowCV(
            n_splits=n_splits,
            test_size=test_size,
            gap=gap,
            min_train_size=kwargs.get('min_train_size', 252),
            purge=kwargs.get('purge', 0)
        )
    elif cv_type == 'rolling':
        return RollingWindowCV(
            n_splits=n_splits,
            test_size=test_size,
            train_size=kwargs.get('train_size', None),
            gap=gap,
            purge=kwargs.get('purge', 0)
        )
    elif cv_type == 'purged':
        return PurgedKFoldCV(
            n_splits=n_splits,
            purge_length=kwargs.get('purge_length', 60)
        )
    else:
        raise ValueError(f"Unknown cv_type: {cv_type}. Use 'expanding', 'rolling', or 'purged'")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    # Demo with synthetic data
    np.random.seed(42)
    
    n_samples = 1000
    X = np.random.randn(n_samples, 10)
    y = np.random.randn(n_samples)
    
    print("Testing RollingWindowCV (fixed train window):")
    print("-" * 50)
    cv = RollingWindowCV(n_splits=5, test_size=60, train_size=200, gap=1)
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        print(f"Fold {fold_idx+1}: train [{train_idx[0]}-{train_idx[-1]}] ({len(train_idx)}), "
              f"test [{test_idx[0]}-{test_idx[-1]}] ({len(test_idx)})")
    
    print("\nTesting ExpandingWindowCV (growing train window):")
    print("-" * 50)
    cv = ExpandingWindowCV(n_splits=5, test_size=60, min_train_size=200, gap=1)
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        print(f"Fold {fold_idx+1}: train [{train_idx[0]}-{train_idx[-1]}] ({len(train_idx)}), "
              f"test [{test_idx[0]}-{test_idx[-1]}] ({len(test_idx)})")
    
    print("\nTesting PurgedKFoldCV (with sequence purging):")
    print("-" * 50)
    cv = PurgedKFoldCV(n_splits=5, purge_length=10)
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X)):
        print(f"Fold {fold_idx+1}: train ({len(train_idx)} samples), test ({len(test_idx)} samples)")
        print(f"  Purge creates gap: train ends at {train_idx[-1] if len(train_idx) else 'N/A'}, "
              f"test starts at {test_idx[0] if len(test_idx) else 'N/A'}")
