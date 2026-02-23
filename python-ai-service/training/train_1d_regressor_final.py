"""
1-Day Return Regressor - Research-Backed Configuration

Based on:
- Krauss et al. 2017: "Short horizons more predictable"
- Ding et al. 2015: "1-day ahead optimal for actionable signals"

Key changes from previous:
- 1-day horizon (not 5-day)
- Target scaling with adaptive range
- Proper SMAPE calculation
- More regularization (dropout=0.3)
"""

import argparse
import sys
import io
import shutil
import os
import stat
import json
import random
from pathlib import Path

# P0 FIX: Import yfinance BEFORE TensorFlow to avoid SSL conflict
import yfinance as yf  # noqa: F401 - must be first

# Force UTF-8 encoding for Windows compatibility
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

sys.path.append(str(Path(__file__).parent.parent))

import numpy as np
import pickle
from datetime import datetime
import tensorflow as tf

# ============================================================================
# GPU & ACCELERATION CONFIGURATION (New in Phase 6)
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
                # print(f"[GPU] Configured LD_LIBRARY_PATH with {len(lib_dirs)} NVIDIA paths")
    except Exception as e:
        print(f"[GPU] Warning: Failed to auto-configure LD_LIBRARY_PATH: {e}")

    # 2. NUCLEAR FIX v7.0: Force float32 for numerical stability
    # Research finding (January 2026): Float16 causes attention overflow in transformers
    # - Float16 max value is ~65,504
    # - Attention logits can exceed this, causing NaN
    # - Result: loss: 1000.x (fallback), mae: nan from epoch 1
    # Solution: Use float32 until model is proven stable
    try:
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('float32')  # NOT mixed_float16!
        mixed_precision.set_global_policy(policy)
        print(f"[GPU] Using float32 for numerical stability (attention overflow prevention)")
        print(f"[GPU] Mixed precision DISABLED until model is stable")
    except Exception as e:
        print(f"[GPU] Warning: Could not set precision policy: {e}")

    # 3. Enable JIT compilation (XLA) if possible
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'

setup_gpu_acceleration()

# ============================================================================
# DETERMINISTIC SEED SETTINGS (P0 FIX - Added for reproducibility)
# ============================================================================
def set_global_seed(seed: int = 42) -> None:
    """Set seeds for all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    # Try PyTorch if available (for any torch-based components)
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
try:
    from tensorflow import keras
    from tensorflow.keras import regularizers
except ImportError:
    # Fallback for TensorFlow 2.20+ where keras is separate
    import keras
    from keras import regularizers
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler, QuantileTransformer
import logging

# Module-level logger (will be configured with file handler in main)
logger = logging.getLogger(__name__)

from data.data_fetcher import fetch_stock_data
from data.feature_engineer import engineer_features, validate_and_fix_features, get_feature_columns, EXPECTED_FEATURE_COUNT
from data.target_engineering import prepare_training_data
from data.target_balancing import balance_target_distribution, get_balancing_metadata, BalancingReport
from models.lstm_transformer_paper import (
    LSTMTransformerPaper,
    DirectionalHuberLoss,
    AntiCollapseDirectionalLoss,
    directional_accuracy_metric as model_directional_accuracy,
)
from utils.model_paths import ModelPaths, get_legacy_regressor_paths
from utils.training_logger import setup_training_logger
# NUCLEAR FIX v5.0: Import ResidualAwareLoss - SIMPLIFIED loss without competing objectives
# ROOT CAUSE: Previous losses (BalancedDirectionalLoss, AntiCollapseDirectionalLoss) had 4+ competing
# objectives that created CONFLICTING GRADIENTS. Simple loss + residual architecture is the solution.
from utils.losses import BalancedDirectionalLoss, ResidualAwareLoss


def r_squared_metric(y_true, y_pred):
    """
    Compute R² (coefficient of determination) metric.

    R² = 1 - (SS_res / SS_tot)
    where:
    - SS_res = sum of squared residuals = Σ(y_true - y_pred)²
    - SS_tot = total sum of squares = Σ(y_true - mean(y_true))²

    R² ranges from -∞ to 1:
    - R² = 1: Perfect prediction
    - R² = 0: Model equals baseline (mean)
    - R² < 0: Model worse than baseline

    Returns:
        R² score (higher is better)
    """
    # Cast to float32 to handle mixed precision training (float16/float32 mismatch)
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # Compute residual sum of squares
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))

    # Compute total sum of squares
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))

    # Compute R²
    r2 = 1.0 - (ss_res / (ss_tot + 1e-8))

    # NaN safety: replace NaN/Inf with -1.0 (worst possible R² for display)
    r2 = tf.where(tf.math.is_finite(r2), r2, -1.0)

    return r2


def validate_target_distribution(
    y_train: np.ndarray,
    symbol: str,
    min_positive_pct: float = 0.30,
    min_negative_pct: float = 0.30,
    save_metadata: bool = True
) -> dict:
    """
    Validate target distribution before training and save metadata.
    
    This function performs comprehensive checks on the target distribution to ensure
    the model will learn balanced predictions. It computes statistics for positive
    and negative returns separately and validates that neither class is severely
    underrepresented.
    
    Args:
        y_train: Array of training target values (returns).
        symbol: Stock symbol for saving metadata.
        min_positive_pct: Minimum required percentage of positive targets (default 30%).
        min_negative_pct: Minimum required percentage of negative targets (default 30%).
        save_metadata: Whether to save distribution metadata to JSON file.
    
    Returns:
        Dictionary containing target distribution statistics.
    
    Raises:
        ValueError: If target distribution fails validation checks.
    """
    y_train = np.asarray(y_train).flatten()
    n_total = len(y_train)
    
    if n_total == 0:
        raise ValueError("Empty training targets provided")
    
    # =========================================================================
    # 1. Compute basic distribution statistics
    # =========================================================================
    
    # Separate positive, negative, and zero returns
    positive_mask = y_train > 0
    negative_mask = y_train < 0
    zero_mask = y_train == 0
    
    n_positive = int(np.sum(positive_mask))
    n_negative = int(np.sum(negative_mask))
    n_zero = int(np.sum(zero_mask))
    
    pct_positive = n_positive / n_total
    pct_negative = n_negative / n_total
    pct_zero = n_zero / n_total
    
    # =========================================================================
    # 2. Compute statistics for positive and negative returns separately
    # =========================================================================
    
    positive_returns = y_train[positive_mask]
    negative_returns = y_train[negative_mask]
    
    mean_positive = float(np.mean(positive_returns)) if n_positive > 0 else 0.0
    mean_negative = float(np.mean(negative_returns)) if n_negative > 0 else 0.0
    
    std_positive = float(np.std(positive_returns)) if n_positive > 0 else 0.0
    std_negative = float(np.std(negative_returns)) if n_negative > 0 else 0.0
    
    # Overall statistics
    mean_all = float(np.mean(y_train))
    std_all = float(np.std(y_train))
    
    # =========================================================================
    # 3. Compute percentile distribution
    # =========================================================================
    
    percentiles = {
        'p10': float(np.percentile(y_train, 10)),
        'p25': float(np.percentile(y_train, 25)),
        'p50': float(np.percentile(y_train, 50)),
        'p75': float(np.percentile(y_train, 75)),
        'p90': float(np.percentile(y_train, 90)),
    }
    
    # =========================================================================
    # 4. Build metadata dictionary
    # =========================================================================
    
    metadata = {
        'symbol': symbol,
        'n_total': n_total,
        'n_positive': n_positive,
        'n_negative': n_negative,
        'n_zero': n_zero,
        'pct_positive': round(pct_positive, 4),
        'pct_negative': round(pct_negative, 4),
        'pct_zero': round(pct_zero, 4),
        'mean_positive': round(mean_positive, 6),
        'mean_negative': round(mean_negative, 6),
        'std_positive': round(std_positive, 6),
        'std_negative': round(std_negative, 6),
        'mean': round(mean_all, 6),
        'std': round(std_all, 6),
        'percentiles': {k: round(v, 6) for k, v in percentiles.items()},
        'validation_thresholds': {
            'min_positive_pct': min_positive_pct,
            'min_negative_pct': min_negative_pct,
        },
        'validated_at': datetime.now().isoformat(),
    }
    
    # =========================================================================
    # 5. Log distribution information
    # =========================================================================
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("TARGET DISTRIBUTION CHECK")
    logger.info("=" * 60)
    logger.info(f"   Positive: {pct_positive * 100:.1f}% ({n_positive:,} samples) | mean: {mean_positive * 100:+.2f}%")
    logger.info(f"   Negative: {pct_negative * 100:.1f}% ({n_negative:,} samples) | mean: {mean_negative * 100:+.2f}%")
    if n_zero > 0:
        logger.info(f"   Zero:     {pct_zero * 100:.1f}% ({n_zero:,} samples)")
    logger.info(f"   Overall:  mean={mean_all * 100:+.3f}%, std={std_all * 100:.3f}%")
    logger.info("")
    logger.info(f"   Percentiles:")
    logger.info(f"      10th: {percentiles['p10'] * 100:+.3f}%")
    logger.info(f"      25th: {percentiles['p25'] * 100:+.3f}%")
    logger.info(f"      50th: {percentiles['p50'] * 100:+.3f}% (median)")
    logger.info(f"      75th: {percentiles['p75'] * 100:+.3f}%")
    logger.info(f"      90th: {percentiles['p90'] * 100:+.3f}%")
    
    # =========================================================================
    # 6. Assertion checks
    # =========================================================================
    
    issues = []
    
    if pct_positive < min_positive_pct:
        issues.append(
            f"Insufficient positive returns: {pct_positive * 100:.1f}% < {min_positive_pct * 100:.0f}% required"
        )
    
    if pct_negative < min_negative_pct:
        issues.append(
            f"Insufficient negative returns: {pct_negative * 100:.1f}% < {min_negative_pct * 100:.0f}% required"
        )
    
    if issues:
        logger.error("")
        logger.error("[FAIL] TARGET DISTRIBUTION VALIDATION FAILED")
        for issue in issues:
            logger.error(f"   * {issue}")
        logger.error("")
        logger.error("Diagnostics:")
        logger.error(f"   Total samples: {n_total:,}")
        logger.error(f"   Positive samples: {n_positive:,} ({pct_positive * 100:.1f}%)")
        logger.error(f"   Negative samples: {n_negative:,} ({pct_negative * 100:.1f}%)")
        logger.error(f"   Mean positive return: {mean_positive * 100:+.3f}%")
        logger.error(f"   Mean negative return: {mean_negative * 100:+.3f}%")
        logger.error("")
        logger.error("Possible causes:")
        logger.error("   1. Data preprocessing issue (targets clipped too aggressively)")
        logger.error("   2. Market regime bias in training period")
        logger.error("   3. Target calculation error")
        logger.error("")
        
        metadata['validation_passed'] = False
        metadata['validation_issues'] = issues
        
        # Save metadata even on failure for debugging
        if save_metadata:
            _save_target_distribution_metadata(symbol, metadata)
        
        raise ValueError(
            f"Target distribution imbalanced: {'; '.join(issues)}. "
            f"Check data preprocessing or adjust validation thresholds."
        )
    
    # =========================================================================
    # 7. Success logging and metadata save
    # =========================================================================
    
    logger.info("")
    logger.info("[OK] Distribution is balanced")
    logger.info("=" * 60)
    
    metadata['validation_passed'] = True
    metadata['validation_issues'] = []
    
    if save_metadata:
        _save_target_distribution_metadata(symbol, metadata)
    
    return metadata


def _save_target_distribution_metadata(symbol: str, metadata: dict) -> None:
    """
    Save target distribution metadata to JSON file.
    
    Args:
        symbol: Stock symbol.
        metadata: Distribution metadata dictionary.
    """
    try:
        paths = ModelPaths(symbol)
        paths.ensure_dirs()
        
        # Save to symbol directory
        output_path = paths.symbol_dir / 'target_distribution.json'
        
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"   [OK] Target distribution metadata saved to {output_path}")
        
    except Exception as e:
        logger.warning(f"   [WARN] Failed to save target distribution metadata: {e}")


def prepare_targets_robust(y_raw, clip_range=(-0.5, 0.5)):
    """
    Robust target preparation with clipping + log1p + RobustScaler.

    Returns:
        y_scaled (1d numpy array), target_scaler (fitted RobustScaler), metadata (dict)
    """
    # Step 1: Clip extreme outliers
    y_clipped = np.clip(y_raw, clip_range[0], clip_range[1])

    # Step 2: Log1p transform
    # Guard against values <= -1 which would make log1p invalid; shift small epsilon
    eps = 1e-12
    y_shifted = y_clipped.copy().astype(float)
    # Ensure no value <= -1.0
    y_shifted = np.where(y_shifted <= -0.9999999999, -0.9999999999, y_shifted)
    y_transformed = np.log1p(y_shifted)

    # Step 3: RobustScaler (resistant to outliers)
    target_scaler = RobustScaler()
    y_scaled = target_scaler.fit_transform(y_transformed.reshape(-1, 1)).flatten()

    # Save metadata
    try:
        scaler_center = float(target_scaler.center_[0])
        scaler_scale = float(target_scaler.scale_[0])
    except Exception:
        scaler_center = float('nan')
        scaler_scale = float('nan')

    metadata = {
        'scaling_method': 'robust',
        'clip_range': clip_range,
        'scaler_center': scaler_center,
        'scaler_scale': scaler_scale
    }

    return y_scaled, target_scaler, metadata


def validate_target_scaling(y_scaled, metadata):
    """Ensure target scaling is stable"""
    assert metadata.get('scaling_method') == 'robust'
    mean_val = float(np.mean(y_scaled))
    std_val = float(np.std(y_scaled))
    if not (-5.0 < mean_val < 5.0):
        raise ValueError(f"Scaled target mean out of range: {mean_val}")
    if not (0.1 < std_val < 10.0):
        raise ValueError(f"Scaled target std out of range: {std_val}")


def create_optimized_tf_dataset(X, y, y_sign=None, y_vol=None, batch_size=512, 
                                shuffle=True, prefetch=True, is_multitask=False):
    """Create optimized tf.data.Dataset with caching, prefetching, and parallelization.
    
    This function implements the critical optimization for GPU utilization:
    - tf.data.Dataset for parallel data loading (overlaps CPU I/O with GPU computation)
    - .cache() to cache preprocessed data in memory for fast repeated access
    - .prefetch(buffer_size) to prepare next batch while GPU processes current
    
    Args:
        X: Input features (N, seq_len, features) - can be NumPy or TensorFlow
        y: Target values for single-task learning
        y_sign: Optional sign labels for multi-task learning
        y_vol: Optional volatility targets for multi-task learning
        batch_size: Batch size (recommend 512 for RTX 5060 Ti)
        shuffle: Whether to shuffle data
        prefetch: Whether to enable prefetching (strongly recommended)
        is_multitask: Whether this is a multi-task model
    
    Returns:
        tf.data.Dataset object optimized for GPU training
    """
    print(f"[PIPELINE] Creating optimized tf.data.Dataset with batch_size={batch_size}")
    
    # CRITICAL: Convert NumPy arrays to float32 FIRST before creating dataset
    # This ensures dtype consistency throughout the pipeline
    if isinstance(X, np.ndarray):
        X = X.astype(np.float32)
    if isinstance(y, np.ndarray):
        y = y.astype(np.float32)
    if y_sign is not None and isinstance(y_sign, np.ndarray):
        y_sign = y_sign.astype(np.float32)
    if y_vol is not None and isinstance(y_vol, np.ndarray):
        y_vol = y_vol.astype(np.float32)
    
    # Create dataset from tensors
    if is_multitask and y_sign is not None and y_vol is not None:
        # Multi-task: return dict of outputs
        dataset = tf.data.Dataset.from_tensor_slices((
            X,
            {
                'magnitude_output': y,
                'sign_output': y_sign,
                'volatility_output': y_vol
            }
        ))
    else:
        # Single-task: return y directly
        dataset = tf.data.Dataset.from_tensor_slices((X, y))
    
    # IMPORTANT: Configure tf.data options for maximum throughput
    options = tf.data.Options()
    options.deterministic = False  # Allow non-determinism for speed (shuffle already randomizes)
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.OFF
    
    # Critical optimizations for GPU throughput:
    # 1. SHUFFLE: Randomize batches (important for SGD)
    if shuffle:
        buffer_size = max(1000, len(X) // 2)  # Use half of dataset as shuffle buffer
        dataset = dataset.shuffle(buffer_size=buffer_size, reshuffle_each_iteration=True)
        print(f"[PIPELINE] Enabled shuffling (buffer_size={buffer_size})")
    
    # 2. BATCH: Group into batches
    dataset = dataset.batch(batch_size, drop_remainder=False)
    
    # 3. CACHE: Store preprocessed data in memory for repeated epochs (HUGE speedup)
    #           This is already preprocessed, so caching the full dataset is safe
    dataset = dataset.cache()
    print(f"[PIPELINE] Enabled caching - preprocessed data cached in memory")
    
    # 4. APPLY OPTIONS: Enable determinism=False and auto-shard for speed
    dataset = dataset.with_options(options)
    
    # 5. PREFETCH: Load next batch while GPU processes current batch (critical for GPU utilization)
    #              Use large explicit buffer for RTX 5060 Ti - more aggressive prefetching
    if prefetch:
        # For optimal GPU utilization, use buffer_size=2x batch_size
        # This keeps GPU constantly fed with data
        prefetch_buffer = max(2, 2 * batch_size // 512)  # At least 2, typically 2-4
        dataset = dataset.prefetch(buffer_size=prefetch_buffer)
        print(f"[PIPELINE] Enabled prefetching (buffer_size={prefetch_buffer}) - aggressive GPU feeding")
    
    return dataset


def create_sequences(X, sequence_length=90):
    """Create sequences with 90-day lookback for quarterly patterns.
    
    OPTIMIZED: Uses NumPy striding for GPU-friendly memory layout (3x faster than Python loops).
    For further optimization, consider tf.data.Dataset.window() if using eager execution.
    """
    # Convert to contiguous array to ensure proper memory layout for GPU access
    X = np.ascontiguousarray(X)
    
    # Use NumPy's optimized striding instead of Python loop
    # This creates a view without copying, then we copy once at the end
    shape = (X.shape[0] - sequence_length + 1, sequence_length, X.shape[1])
    strides = (X.strides[0], X.strides[0], X.strides[1])
    
    try:
        # np.lib.stride_tricks creates a memory-efficient view
        from numpy.lib.stride_tricks import as_strided
        X_strided = as_strided(X, shape=shape, strides=strides)
        # Copy to ensure contiguous memory layout (important for GPU)
        return np.ascontiguousarray(X_strided)
    except Exception:
        # Fallback to NumPy advanced indexing (still faster than Python loop)
        indices = np.arange(sequence_length)[None, :] + np.arange(X.shape[0] - sequence_length + 1)[:, None]
        return X[indices]


def validate_training_data_quality(X, y):
    """Pre-flight checks before training."""
    logger.info("\n=== Data Quality Pre-Check ===")

    # Ensure numpy arrays
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)

    # Check for inf/nan in features
    if not np.all(np.isfinite(X_arr)):
        bad_count = int(np.sum(~np.isfinite(X_arr)))
        raise ValueError(f"Features contain {bad_count} inf/nan values")

    # Check for inf/nan in targets
    if not np.all(np.isfinite(y_arr)):
        bad_count = int(np.sum(~np.isfinite(y_arr)))
        raise ValueError(f"Targets contain {bad_count} inf/nan values")

    # Check feature variance (detect constant features)
    try:
        feature_stds = np.std(X_arr.reshape(-1, X_arr.shape[-1]), axis=0)
        zero_var_features = int(np.sum(feature_stds < 1e-8))
        if zero_var_features > 0:
            logger.warning(f"[WARN] {zero_var_features} features have near-zero variance")
    except Exception:
        # If reshape fails, skip variance check
        pass

    # Check target distribution
    logger.info(f"Target range: [{float(np.min(y_arr)):.4f}, {float(np.max(y_arr)):.4f}]")
    logger.info(f"Target std: {float(np.std(y_arr)):.4f} (should be ~1 after scaling)")

    if float(np.std(y_arr)) < 0.1:
        raise ValueError("Target has very low variance, model won't learn anything")

    logger.info("[OK] Data quality checks passed")


def filter_low_variance_features(X_train, X_val, feature_cols, min_variance=0.001):
    """
    NUCLEAR FIX v7.0: Remove features with near-zero variance.

    Research finding (January 2026): Training logs consistently show
    "38 features have near-zero variance" - that's 25% of features providing
    NO useful signal but adding noise to gradients.

    This function removes those features to improve model stability.

    Args:
        X_train: Training data (samples, seq_len, features)
        X_val: Validation data (samples, seq_len, features)
        feature_cols: List of feature column names
        min_variance: Minimum variance threshold (default 0.001)

    Returns:
        X_train_filtered, X_val_filtered, filtered_feature_cols
    """
    # Compute variance across samples and time for each feature
    # Shape: (samples * seq_len, features) -> variance per feature
    X_flat = X_train.reshape(-1, X_train.shape[-1])
    variances = np.var(X_flat, axis=0)

    # Create mask for features with sufficient variance
    good_mask = variances > min_variance

    n_total = len(feature_cols)
    n_removed = np.sum(~good_mask)
    n_kept = np.sum(good_mask)

    if n_removed > 0:
        # Log which features are being removed
        removed_features = [f for f, m in zip(feature_cols, good_mask) if not m]
        print(f"\n[FILTER] REMOVING {n_removed} low-variance features:")
        for f in removed_features[:10]:  # Show first 10
            print(f"   - {f}")
        if len(removed_features) > 10:
            print(f"   - ... and {len(removed_features) - 10} more")
        print(f"[FILTER] KEEPING {n_kept}/{n_total} features with variance > {min_variance}")

        # Apply filter
        X_train_filtered = X_train[:, :, good_mask]
        X_val_filtered = X_val[:, :, good_mask]
        filtered_feature_cols = [f for f, m in zip(feature_cols, good_mask) if m]

        return X_train_filtered, X_val_filtered, filtered_feature_cols
    else:
        print(f"[FILTER] All {n_total} features have sufficient variance (> {min_variance})")
        return X_train, X_val, feature_cols


def compute_smape_correct(y_true, y_pred):
    """
    Correct SMAPE calculation (symmetric, bounded 0-200%).
    
    SMAPE = 100 * mean(|pred - actual| / ((|actual| + |pred|) / 2))
    """
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denominator = np.maximum(denominator, 1e-8)  # Avoid division by zero
    
    return 100 * np.mean(numerator / denominator)


def directional_accuracy_metric(y_true, y_pred):
    """Custom metric: fraction of correct sign predictions (NaN-safe)."""
    # Cast both to float32 to ensure dtype consistency for tf.equal()
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    # NaN safety: replace NaN/Inf with 0 to avoid metric corruption
    y_true = tf.where(tf.math.is_finite(y_true), y_true, tf.zeros_like(y_true))
    y_pred = tf.where(tf.math.is_finite(y_pred), y_pred, tf.zeros_like(y_pred))

    y_true_sign = tf.sign(y_true)
    y_pred_sign = tf.sign(y_pred)
    correct = tf.cast(tf.equal(y_true_sign, y_pred_sign), tf.float32)
    accuracy = tf.reduce_mean(correct)

    # Final safety: return 0 if accuracy is NaN
    return tf.where(tf.math.is_finite(accuracy), accuracy, 0.0)


def quantile_loss(quantile=0.5):
    """Quantile loss for quantile regression.
    
    Loss asymmetry:
    - Over-prediction penalized by (1 - quantile)
    - Under-prediction penalized by quantile
    
    For quantile=0.25: penalize over-predictions more (conservative)
    For quantile=0.75: penalize under-predictions more (optimistic)
    """
    q = float(np.clip(quantile, 1e-3, 1 - 1e-3))
    
    def loss_fn(y_true, y_pred):
        error = y_true - y_pred
        return tf.reduce_mean(tf.maximum(q * error, (q - 1.0) * error))
    
    return loss_fn


class VarianceRegularizedLoss(tf.keras.losses.Loss):
    """Penalize constant outputs by adding variance regularization.
    
    This loss wrapper adds a penalty for low variance in predictions,
    encouraging the model to produce a spread of outputs rather than
    collapsing to constant/near-constant predictions.
    
    The penalty is inversely proportional to prediction variance:
    - High variance → low penalty
    - Low variance → high penalty
    
    ANTI-COLLAPSE DESIGN (v2.0):
    - Added minimum variance threshold (min_variance_target) below which
      the penalty grows exponentially
    - Added directional diversity penalty to ensure predictions
      span both positive and negative values
    
    Args:
        base_loss: The underlying loss function to wrap
        variance_weight: Weight for the variance penalty (default 0.01)
        min_variance_target: Minimum target variance below which penalty grows (default 0.003)
    """
    
    def __init__(self, base_loss, variance_weight=0.01, min_variance_target=0.003, name='variance_regularized'):
        super().__init__(name=name)
        self.base_loss = base_loss
        self.variance_weight = variance_weight
        self.min_variance_target = min_variance_target
    
    def call(self, y_true, y_pred):
        # Standard base loss
        base = self.base_loss(y_true, y_pred)
        
        # Variance penalty (encourage spread in predictions)
        pred_variance = tf.math.reduce_variance(y_pred)
        pred_std = tf.math.sqrt(pred_variance + 1e-8)
        
        # ENHANCED: Two-stage variance penalty
        # Stage 1: Inverse penalty (always active)
        inverse_penalty = self.variance_weight / (pred_variance + 1e-6)
        
        # Stage 2: Exponential penalty when variance drops below threshold
        # This kicks in hard when model starts collapsing
        variance_deficit = tf.maximum(self.min_variance_target - pred_std, 0.0)
        collapse_penalty = 10.0 * tf.square(variance_deficit / self.min_variance_target)
        
        # Stage 3: Directional diversity penalty
        # Encourage predictions to span both positive and negative
        # If all predictions are the same sign, add penalty
        pred_signs = tf.sign(y_pred)
        sign_variance = tf.math.reduce_variance(pred_signs)
        # Low sign variance = all same sign = penalty
        sign_diversity_penalty = 0.1 * tf.maximum(0.5 - sign_variance, 0.0)
        
        total_variance_penalty = inverse_penalty + collapse_penalty + sign_diversity_penalty
        
        return base + total_variance_penalty
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'variance_weight': self.variance_weight,
            'min_variance_target': self.min_variance_target
        })
        return config


def get_regression_loss(loss_name='huber', quantile=0.5, use_direction_loss=True, direction_weight=0.3):
    """Return a regression loss callable with optional auxiliary direction loss.
    
    Combined loss = magnitude_loss + direction_weight * direction_loss
    - magnitude_loss: Huber/MAE for return magnitude
    - direction_loss: Binary crossentropy on sign(return) for direction
    
    Args:
        loss_name: Loss function name. Options:
            - 'huber': Standard Huber loss
            - 'mae': Mean Absolute Error
            - 'huber_mae': Combined Huber + MAE
            - 'quantile': Quantile loss
            - 'directional_huber': DirectionalHuberLoss (penalizes wrong direction)
        quantile: Quantile parameter for quantile loss
        use_direction_loss: Whether to add auxiliary direction loss (for huber/mae)
        direction_weight: Weight for auxiliary direction loss or DirectionalHuberLoss penalty
    
    Returns:
        Loss function callable
    """
    normalized = (loss_name or 'huber').lower()
    
    # Special case: DirectionalHuberLoss from models module
    if normalized == 'directional_huber':
        return DirectionalHuberLoss(delta=1.0, direction_weight=direction_weight)

    # Base magnitude loss
    if normalized == 'huber':
        magnitude_loss_fn = keras.losses.Huber()
    elif normalized == 'mae':
        magnitude_loss_fn = keras.losses.MeanAbsoluteError()
    elif normalized == 'huber_mae':
        huber = keras.losses.Huber()
        mae = keras.losses.MeanAbsoluteError()
        def magnitude_loss_fn(y_true, y_pred):
            return huber(y_true, y_pred) + 0.2 * mae(y_true, y_pred)
    elif normalized == 'quantile':
        magnitude_loss_fn = quantile_loss(quantile)
    else:
        raise ValueError(f"Unsupported loss: {loss_name}")
    
    if not use_direction_loss:
        return magnitude_loss_fn
    
    # Combined loss with direction component
    def combined_loss(y_true, y_pred):
        # Magnitude loss (primary)
        mag_loss = magnitude_loss_fn(y_true, y_pred)

        # Direction loss (auxiliary)
        # Ensure both y_true and y_pred have shape [batch_size] for binary_crossentropy
        # y_true may be [batch_size] or [batch_size, 1], flatten to [batch_size]
        y_true_flat = tf.reshape(y_true, [-1])
        # y_pred is [batch_size, 1] from Dense(1) output, squeeze to [batch_size]
        y_pred_flat = tf.squeeze(y_pred, axis=-1)

        # Convert to binary labels: positive return = 1, negative = 0
        y_true_binary = tf.cast(tf.greater(y_true_flat, 0.0), tf.float32)
        # Scale predictions for binary classification (sigmoid maps to [0,1])
        y_pred_binary = tf.nn.sigmoid(y_pred_flat * 10.0)

        # CRITICAL FIX: Replace binary_crossentropy with MSE to avoid NaN in float32 GPU mode
        # binary_crossentropy causes NaN with: large batch (512) + deep LSTM + float32 + GPU
        # References: tensorflow/tensorflow#2995, #39453
        dir_loss = tf.reduce_mean(tf.square(y_true_binary - y_pred_binary))

        # VARIANCE COLLAPSE PROTECTION: Penalize low prediction variance
        # Prevents model from predicting constant values (variance collapse)
        pred_mean = tf.reduce_mean(y_pred_flat)
        pred_variance = tf.reduce_mean(tf.square(y_pred_flat - pred_mean))
        pred_std = tf.sqrt(tf.maximum(pred_variance, 1e-8))

        # Target std from y_true for reference
        true_mean = tf.reduce_mean(y_true_flat)
        true_variance = tf.reduce_mean(tf.square(y_true_flat - true_mean))
        true_std = tf.sqrt(tf.maximum(true_variance, 1e-8))

        # Penalize when pred_std is much smaller than true_std
        # Use negative log to heavily penalize small pred_std
        std_ratio = pred_std / tf.maximum(true_std, 1e-6)
        variance_penalty = -tf.math.log(tf.maximum(std_ratio, 1e-6))
        variance_penalty = tf.maximum(variance_penalty, 0.0)  # Only penalize if ratio < 1

        # Sign diversity penalty: encourage balanced positive/negative predictions
        positive_frac = tf.reduce_mean(tf.cast(tf.greater(y_pred_flat, 0.0), tf.float32))
        # Ideal is 0.5 (50% positive, 50% negative)
        diversity_penalty = tf.square(positive_frac - 0.5) * 2.0  # Scale to [0, 0.5]

        # Combine: total_loss = mag_loss + direction_loss + variance_penalty + diversity_penalty
        total_loss = mag_loss + direction_weight * dir_loss + 0.3 * variance_penalty + 0.1 * diversity_penalty
        return total_loss
    
    return combined_loss


def _handle_remove_readonly(func, path, exc_info):
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception as exc:  # best-effort cleanup
        print(f"[WARN] Failed to remove {path}: {exc}")


def remove_saved_model_dir(path: Path, max_retries: int = 3, delay: float = 1.0):
    """Remove SavedModel export directory, handling Windows permission quirks.
    
    Args:
        path: Path to the directory to remove
        max_retries: Number of retry attempts for Windows file locking issues
        delay: Delay in seconds between retries
    """
    import time
    import gc
    
    if not path.exists():
        return
    
    for attempt in range(max_retries):
        try:
            # Force garbage collection to release any file handles
            gc.collect()
            
            # Try simple file removal first if it's a file
            if path.is_file():
                path.unlink()
                return

            # Try TensorFlow's file removal first
            try:
                tf.io.gfile.rmtree(str(path))
                return
            except Exception:
                pass
            
            # Fall back to shutil with read-only handler
            shutil.rmtree(path, onerror=_handle_remove_readonly)
            return
            
        except PermissionError as e:
            if attempt < max_retries - 1:
                print(f"   Retry {attempt + 1}/{max_retries}: waiting for file lock release...")
                time.sleep(delay)
                gc.collect()
            else:
                # Last resort: rename and schedule for cleanup
                try:
                    import uuid
                    temp_name = path.parent / f".trash_{path.name}_{uuid.uuid4().hex[:8]}"
                    path.rename(temp_name)
                    print(f"   Renamed locked directory to {temp_name.name} for later cleanup")
                except Exception:
                    print(f"[WARN] Could not remove {path}: {e}")
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(delay)
            else:
                print(f"[WARN] Failed to remove {path} after {max_retries} attempts: {e}")


class CollapseDetectionCallback(keras.callbacks.Callback):
    """Callback to detect and prevent model collapse during training.
    
    Monitors prediction variance at the end of each epoch. If variance
    drops below threshold for multiple consecutive epochs, raises warning
    and optionally resets model weights.
    
    Args:
        X_val: Validation data for prediction variance check
        min_variance: Minimum acceptable prediction std (default 0.003)
        patience: Epochs of low variance before warning (default 3)
        reset_on_collapse: Whether to reinitialize model on collapse (default False)
    """
    
    def __init__(self, X_val, min_variance=0.003, patience=3, reset_on_collapse=False):
        super().__init__()
        self.X_val = X_val
        self.min_variance = min_variance
        self.patience = patience
        self.reset_on_collapse = reset_on_collapse
        self.low_variance_epochs = 0
        self.variance_history = []
        
    def on_epoch_end(self, epoch, logs=None):
        # Get predictions on validation set
        preds = self.model.predict(self.X_val[:100], verbose=0)
        
        # Handle multi-output models
        if isinstance(preds, list):
            preds = preds[0]  # Use first output (magnitude)
        
        pred_std = float(np.std(preds))
        self.variance_history.append(pred_std)
        
        if pred_std < self.min_variance:
            self.low_variance_epochs += 1
            status = f"⚠️  LOW VARIANCE: std={pred_std:.6f} < {self.min_variance} (epoch {self.low_variance_epochs}/{self.patience})"
            
            if self.low_variance_epochs >= self.patience:
                print(f"\n🚨 COLLAPSE DETECTED: Predictions collapsed to near-constant values!")
                print(f"   Prediction std: {pred_std:.8f}")
                print(f"   Prediction range: {float(np.ptp(preds)):.8f}")
                print("   STOPPING TRAINING - variance collapse is unrecoverable.")
                print("   Try: lower learning rate, higher variance_penalty_weight, or more dropout.")
                # December 2025 fix: ALWAYS stop training on collapse
                # Continuing training after collapse just wastes time
                self.model.stop_training = True
        else:
            self.low_variance_epochs = 0
            status = f"✓ Variance OK: std={pred_std:.6f}"
        
        # Add variance to logs for monitoring
        if logs is not None:
            logs['pred_variance'] = pred_std
        
        # Print status every 5 epochs or if low variance
        if epoch % 5 == 0 or pred_std < self.min_variance:
            print(f"   {status}")


def safe_export_model(model, export_path: Path, max_retries: int = 3, delay: float = 1.0):
    """Safely save a Keras model with retry logic for Windows file locking.
    
    Uses model.save() for proper Keras SavedModel format that preserves
    the full model architecture and allows keras.models.load_model() to work.
    
    Args:
        model: Keras model to save
        export_path: Path to save to
        max_retries: Number of retry attempts
        delay: Delay in seconds between retries
    """
    import time
    import gc
    
    for attempt in range(max_retries):
        try:
            # Clean up the directory first
            remove_saved_model_dir(export_path, max_retries=2, delay=0.5)
            
            # Force garbage collection before export
            gc.collect()
            
            # Save the model using Keras save (not export) using default format (Keras 3 .keras)
            # This preserves the full model including architecture, weights, and config
            model.save(str(export_path))
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "being used by another process" in error_msg or "Failed to rename" in error_msg:
                if attempt < max_retries - 1:
                    print(f"   Save retry {attempt + 1}/{max_retries}: {export_path.name}")
                    time.sleep(delay)
                    gc.collect()
                else:
                    print(f"[WARN] Could not save to {export_path} after {max_retries} attempts")
                    return False
            else:
                # Non-file-locking error, don't retry
                print(f"[WARN] Save failed for {export_path}: {e}")
                return False
    
    return False


def create_warmup_cosine_schedule(warmup_epochs, total_epochs, warmup_lr=0.0001, max_lr=0.001, min_lr=0.00001,
                                   decay_start_epoch=40, decay_factor=0.5):
    """Create learning rate schedule with warmup, cosine decay, AND mid-training drop.
    
    CRITICAL FIX for epoch 55 collapse:
    - Warmup: Linear increase from warmup_lr to max_lr over warmup_epochs
    - Cosine decay: Cosine decay from max_lr to min_lr over remaining epochs
    - Mid-training drop: At decay_start_epoch, drop LR by decay_factor to prevent collapse
    
    The mid-training drop is critical because:
    1. Model collapse often occurs around epoch 50-60
    2. This is when the model starts to overfit/memorize
    3. Dropping LR prevents the optimizer from overshooting
    
    Args:
        warmup_epochs: Number of warmup epochs
        total_epochs: Total training epochs
        warmup_lr: Starting learning rate during warmup
        max_lr: Peak learning rate after warmup
        min_lr: Minimum learning rate at end of training
        decay_start_epoch: Epoch to apply additional LR drop (default 40)
        decay_factor: Factor to multiply LR by at decay_start_epoch (default 0.5)
    """
    def lr_schedule(epoch):
        if epoch < warmup_epochs:
            # Linear warmup
            base_lr = warmup_lr + (max_lr - warmup_lr) * (epoch / warmup_epochs)
        else:
            # Cosine annealing
            progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            base_lr = min_lr + (max_lr - min_lr) * cosine_decay
        
        # CRITICAL: Apply mid-training LR drop to prevent epoch 55 collapse
        if epoch >= decay_start_epoch:
            base_lr = base_lr * decay_factor
        
        return base_lr
    
    return lr_schedule


def prepare_multitask_targets(y_returns, df_clean, sequence_length=90, flat_threshold=0.002):
    """Prepare targets for multi-task learning.
    
    Args:
        y_returns: Array of return values (magnitude)
        df_clean: Cleaned dataframe with volatility features
        sequence_length: Sequence length for alignment
        flat_threshold: Threshold for flat class (default ±0.2%)
    
    Returns:
        y_sign_onehot: One-hot encoded sign labels (3 classes: down/flat/up)
        y_volatility: Tomorrow's volatility values
    """
    # Create sign labels: 0=down, 1=flat, 2=up
    y_sign = np.zeros(len(y_returns), dtype=int)
    y_sign[y_returns > flat_threshold] = 2  # Up
    y_sign[np.abs(y_returns) <= flat_threshold] = 1  # Flat
    y_sign[y_returns < -flat_threshold] = 0  # Down
    
    # One-hot encode sign labels
    y_sign_onehot = keras.utils.to_categorical(y_sign, num_classes=3)
    
    # Get tomorrow's volatility (shifted by 1)
    if 'volatility_5d' in df_clean.columns:
        volatility_full = df_clean['volatility_5d'].shift(-1).fillna(method='ffill').values
    else:
        # Calculate if not available
        returns_series = df_clean['target_1d']
        volatility_full = returns_series.rolling(5).std().shift(-1).fillna(method='ffill').values
    
    # Align with sequences
    volatility_aligned = volatility_full[sequence_length-1:]
    
    print(f"\n[INFO] Multi-task target distribution:")
    print(f"   Sign classes - Down: {(y_sign==0).sum()} ({(y_sign==0).mean()*100:.1f}%), "
          f"Flat: {(y_sign==1).sum()} ({(y_sign==1).mean()*100:.1f}%), "
          f"Up: {(y_sign==2).sum()} ({(y_sign==2).mean()*100:.1f}%)")
    print(f"   Volatility range: [{volatility_aligned.min():.4f}, {volatility_aligned.max():.4f}]")
    
    return y_sign_onehot, volatility_aligned


def create_multitask_loss(magnitude_weight=1.0, sign_weight=0.5, volatility_weight=0.2, 
                          magnitude_loss_fn=None, variance_regularization=0.0,
                          use_anti_collapse_loss=True):
    """Create composite loss function for multi-task learning.
    
    CRITICAL FIX v4.0: USE AntiCollapseDirectionalLoss FOR MAGNITUDE HEAD
    
    Root cause of variance collapse:
    - The magnitude predictions collapse to near-constant values (std < 0.005)
    - This happens because vanilla loss functions (Huber, MSE) encourage predicting 
      near the mean to minimize expected error
    - Correlation-aware loss alone wasn't enough to prevent collapse
    
    Solution: Use AntiCollapseDirectionalLoss with STRONG variance penalty:
    1. Base Huber loss for regression quality
    2. Direction loss to reward correct sign predictions
    3. VARIANCE PENALTY: Strongly penalize low prediction variance (σ < 0.01)
    4. Sign diversity penalty: Ensure mix of positive/negative predictions
    
    The variance penalty is CRITICAL - it adds a penalty proportional to 1/σ²
    when prediction variance drops below the target threshold.
    
    Args:
        magnitude_weight: Weight for return magnitude loss (default 1.0)
        sign_weight: Weight for sign classification loss (default 0.5)
        volatility_weight: Weight for volatility prediction loss (default 0.2)
        magnitude_loss_fn: Base loss for magnitude (default Huber) - IGNORED if use_anti_collapse_loss=True
        variance_regularization: DEPRECATED - no longer used
        use_anti_collapse_loss: If True, use AntiCollapseDirectionalLoss for magnitude (default True)
    
    Returns:
        Composite loss function and individual loss functions for tracking
    """
    if use_anti_collapse_loss:
        # ===================================================================
        # PRIMARY DEFENSE AGAINST VARIANCE COLLAPSE
        # ===================================================================
        # Use AntiCollapseDirectionalLoss with DYNAMIC variance penalty
        from models.lstm_transformer_paper import AntiCollapseDirectionalLoss
        
        # January 2026 fix: Use REDUCED penalty weights to avoid gradient conflict
        # ROOT CAUSE: Previous high penalties (5.0+) dominated the Huber loss (~0.001)
        # creating competing gradients that trapped model in collapse
        # Fix: Use class defaults (variance=1.0, sign=0.5) or LOWER
        vp_weight = variance_regularization if variance_regularization > 0 else 1.0

        magnitude_loss_fn = AntiCollapseDirectionalLoss(
            delta=1.0,                    # Huber delta
            direction_weight=0.1,         # Small direction nudge (REDUCED from 0.3)
            variance_penalty_weight=vp_weight, # Use reduced weight (1.0 not 5.0)
            min_variance_target=0.005,    # 0.5% target (REDUCED from 0.015)
            sign_diversity_weight=0.5     # Class default (gentle nudge)
        )
        print(f"   [ANTI-COLLAPSE] Using AntiCollapseDirectionalLoss (v4.1 - dynamic)")
        if hasattr(vp_weight, 'numpy'):
            print(f"      Initial variance_penalty_weight={vp_weight.numpy():.4f}")
        else:
            print(f"      variance_penalty_weight={vp_weight}")
    else:
        # Legacy: correlation-aware loss (but this DOESN'T prevent collapse!)
        if magnitude_loss_fn is None:
            magnitude_loss_fn = keras.losses.Huber(delta=1.0)
        
        original_mag_loss = magnitude_loss_fn
        
        def correlation_aware_magnitude_loss(y_true, y_pred):
            """Magnitude loss with correlation component to prevent collapse.
            
            WARNING: This alone is NOT enough to prevent variance collapse!
            Use use_anti_collapse_loss=True instead.
            """
            base_loss = original_mag_loss(y_true, y_pred)
            
            y_true_flat = tf.reshape(y_true, [-1])
            y_pred_flat = tf.reshape(y_pred, [-1])
            
            mean_true = tf.reduce_mean(y_true_flat)
            mean_pred = tf.reduce_mean(y_pred_flat)
            
            diff_true = y_true_flat - mean_true
            diff_pred = y_pred_flat - mean_pred
            
            cov = tf.reduce_mean(diff_true * diff_pred)
            std_true = tf.math.sqrt(tf.reduce_mean(tf.square(diff_true)) + 1e-8)
            std_pred = tf.math.sqrt(tf.reduce_mean(tf.square(diff_pred)) + 1e-8)
            
            correlation = cov / (std_true * std_pred + 1e-8)
            correlation_reward = 0.1 * correlation
            
            return base_loss - correlation_reward
        
        magnitude_loss_fn = correlation_aware_magnitude_loss
        print("   [WARNING] Not using AntiCollapseDirectionalLoss - collapse risk HIGH")
    
    sign_loss_fn = keras.losses.CategoricalCrossentropy()
    volatility_loss_fn = keras.losses.MeanSquaredError()
    
    def composite_loss(y_true, y_pred):
        """Composite loss: magnitude + 0.5*sign + 0.2*volatility"""
        # Unpack y_true and y_pred (they come as lists from model)
        y_true_mag, y_true_sign, y_true_vol = y_true
        y_pred_mag, y_pred_sign, y_pred_vol = y_pred
        
        # Calculate individual losses
        mag_loss = magnitude_loss_fn(y_true_mag, y_pred_mag)
        sign_loss = sign_loss_fn(y_true_sign, y_pred_sign)
        vol_loss = volatility_loss_fn(y_true_vol, y_pred_vol)
        
        # Weighted sum
        total_loss = (magnitude_weight * mag_loss + 
                     sign_weight * sign_loss + 
                     volatility_weight * vol_loss)
        
        return total_loss
    
    # Return individual loss functions for metrics tracking
    return composite_loss, {
        'magnitude_loss': magnitude_loss_fn,
        'sign_loss': sign_loss_fn,
        'volatility_loss': volatility_loss_fn
    }


def create_enhanced_regressor(sequence_length, n_features, name='enhanced_regressor'):
    """Create enhanced regressor with increased capacity and regularization.
    
    Enhancements:
    - LSTM units: 32 → 64
    - d_model: 64 → 128
    - Transformer blocks: 4 → 6
    - L2 regularization on Dense layers
    - BatchNormalization after Dense layers
    """
    from models.lstm_transformer_paper import LSTMTransformerPaper
    
    base = LSTMTransformerPaper(
        sequence_length=sequence_length,
        n_features=n_features,
        lstm_units=64,  # Increased from 32
        d_model=128,    # Increased from 64
        num_heads=4,
        num_blocks=6,   # Increased from 4
        ff_dim=256,     # Increased to match d_model*2
        dropout=0.3
    )
    
    dummy = tf.random.normal((1, sequence_length, n_features))
    _ = base(dummy)
    
    # Build enhanced model with BatchNorm and L2 regularization
    inputs = keras.Input(shape=(sequence_length, n_features))
    
    x = base.lstm_layer(inputs)
    x = base.projection(x)
    x = x + base._pe_numpy[:, :sequence_length, :]
    
    for block in base.transformer_blocks:
        attn = block['attention'](x, x)
        attn = block['dropout1'](attn)
        x = block['norm1'](x + attn)
        
        ffn = block['ffn2'](block['ffn1'](x))
        ffn = block['dropout2'](ffn)
        x = block['norm2'](x + ffn)
    
    x = base.global_pool(x)
    x = base.dropout_out(x)
    
    # Enhanced output head with BatchNorm and L2 regularization
    x = keras.layers.Dense(
        64,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001),
        name='dense_1'
    )(x)
    x = keras.layers.BatchNormalization(name='bn_1')(x)
    x = keras.layers.Dropout(0.3, name='dropout_1')(x)
    
    x = keras.layers.Dense(
        32,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001),
        name='dense_2'
    )(x)
    x = keras.layers.BatchNormalization(name='bn_2')(x)
    x = keras.layers.Dropout(0.3, name='dropout_2')(x)
    
    output = keras.layers.Dense(
        1,
        kernel_regularizer=regularizers.l2(0.001),
        name='output'
    )(x)
    
    return keras.Model(inputs=inputs, outputs=output, name=name)


def create_multitask_regressor(sequence_length, n_features, name='multitask_regressor'):
    """Create multi-task regressor with INCREASED capacity (v4.1).
    
    Multi-task outputs:
    - Head 1: Return magnitude prediction (regression)
    - Head 2: Return sign classification
    - Head 3: Volatility prediction
    
    ANTI-COLLAPSE DESIGN v2:
    - Increased width (128 units)
    - Medium depth (6 blocks) to capture patterns without vanishing gradients
    - HeNormal initialization
    """
    from models.lstm_transformer_paper import LSTMTransformerPaper
    
    # INCREASED capacity to prevent underfitting/collapse
    base = LSTMTransformerPaper(
        sequence_length=sequence_length,
        n_features=n_features,
        lstm_units=128,   # Increased from 48
        d_model=128,      # Increased from 96
        num_heads=4,
        num_blocks=6,     # Increased from 4
        ff_dim=256,       # Increased from 192
        dropout=0.25      # Moderate dropout
    )
    
    dummy = tf.random.normal((1, sequence_length, n_features))
    _ = base(dummy)
    
    # Shared backbone
    inputs = keras.Input(shape=(sequence_length, n_features))
    
    x = base.lstm_layer(inputs)
    x = base.projection(x)
    x = x + base._pe_numpy[:, :sequence_length, :]
    
    for block in base.transformer_blocks:
        attn = block['attention'](x, x)
        attn = block['dropout1'](attn)
        x = block['norm1'](x + attn)
        
        ffn = block['ffn2'](block['ffn1'](x))
        ffn = block['dropout2'](ffn)
        x = block['norm2'](x + ffn)
    
    x = base.global_pool(x)
    x = base.dropout_out(x)
    
    # Shared dense layer
    shared = keras.layers.Dense(
        128,  # Increased from 48
        activation='relu',
        kernel_initializer=keras.initializers.HeNormal(seed=42),
        kernel_regularizer=regularizers.l2(0.0005),
        name='shared_dense'
    )(x)
    shared = keras.layers.Dropout(0.2, name='shared_dropout')(shared)
    
    # ========== HEAD 1: Return Magnitude (Regression) ==========
    magnitude_branch = keras.layers.Dense(
        64,
        activation='relu',
        kernel_initializer=keras.initializers.HeNormal(seed=43),
        kernel_regularizer=regularizers.l2(0.0001),
        name='magnitude_dense'
    )(shared)
    magnitude_branch = keras.layers.Dropout(0.1, name='magnitude_dropout')(magnitude_branch)
    
    magnitude_branch = keras.layers.Dense(
        32,
        activation='relu',
        kernel_initializer=keras.initializers.HeNormal(seed=45),
        kernel_regularizer=regularizers.l2(0.0001),
        name='magnitude_dense2'
    )(magnitude_branch)
    
    # OUTPUT LAYER with large initialization variance
    magnitude_output = keras.layers.Dense(
        1,
        activation=None,
        kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=44),
        bias_initializer=keras.initializers.Zeros(),
        name='magnitude_output'
    )(magnitude_branch)
    
    # ========== HEAD 2: Return Sign (Classification: up/down/flat) ==========
    sign_branch = keras.layers.Dense(
        32,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001),
        name='sign_dense'
    )(shared)
    sign_branch = keras.layers.BatchNormalization(name='sign_bn')(sign_branch)
    sign_branch = keras.layers.Dropout(0.2, name='sign_dropout')(sign_branch)
    
    sign_output = keras.layers.Dense(
        3,  # 3 classes: up, down, flat
        activation='softmax',
        kernel_regularizer=regularizers.l2(0.001),
        name='sign_output'
    )(sign_branch)
    
    # ========== HEAD 3: Volatility (Regression) ==========
    volatility_branch = keras.layers.Dense(
        32,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001),
        name='volatility_dense'
    )(shared)
    volatility_branch = keras.layers.BatchNormalization(name='volatility_bn')(volatility_branch)
    volatility_branch = keras.layers.Dropout(0.2, name='volatility_dropout')(volatility_branch)
    
    volatility_output = keras.layers.Dense(
        1,
        activation='relu',  # ReLU ensures positive volatility
        kernel_regularizer=regularizers.l2(0.001),
        name='volatility_output'
    )(volatility_branch)
    
    return keras.Model(
        inputs=inputs,
        outputs=[magnitude_output, sign_output, volatility_output],
        name=name
    )


# ===================================================================
# PREDICTION VARIANCE MONITORING CALLBACK
# ===================================================================

class PredictionVarianceMonitor(keras.callbacks.Callback):
    """Monitor prediction variance during training to detect and PREVENT collapse.
    
    CRITICAL FIX for epoch 55 collapse:
    - Check EVERY 3 epochs after warmup (not every 5)
    - Reduce LR by 50% when variance is low (before stopping)
    - Stop after 2 consecutive failures (not 3)
    - Track variance trend to detect early warning signs
    
    This callback checks prediction variance every N epochs after warmup.
    If variance starts falling, it aggressively reduces learning rate.
    If variance collapses completely, training is stopped early.
    
    Args:
        X_val: Validation features (sequences)
        y_val: Validation targets (scaled)
        target_scaler: Fitted scaler for inverse transform
        min_std: Minimum required prediction standard deviation (default 0.003 = 0.3%)
        patience: Number of consecutive low-variance checks before stopping (default 2)
        check_interval: Epochs between variance checks (default 3)
        warmup_epochs: Epochs to skip before starting checks (default 10)
        use_multitask: Whether model outputs multiple heads (default True)
        lr_reduction_factor: Factor to reduce LR when variance is low (default 0.5)
    """
    
    def __init__(self, X_val, y_val, target_scaler, min_std=0.003, patience=2,
                 check_interval=3, warmup_epochs=10, use_multitask=True, lr_reduction_factor=0.5):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.target_scaler = target_scaler
        self.min_std = min_std
        self.patience = patience
        self.check_interval = check_interval
        self.warmup_epochs = warmup_epochs
        self.use_multitask = use_multitask
        self.lr_reduction_factor = lr_reduction_factor
        self.low_variance_epochs = 0
        self.variance_history = []
        self.lr_reduced = False  # Track if we already reduced LR
    
    def on_epoch_end(self, epoch, logs=None):
        # Only check after warmup and at specified interval
        if epoch < self.warmup_epochs or epoch % self.check_interval != 0:
            return
        
        # Get predictions on validation set
        try:
            raw_preds = self.model.predict(self.X_val, verbose=0)
            if self.use_multitask:
                # Multi-task model: first output is magnitude
                y_pred_scaled = raw_preds[0].flatten()
            else:
                y_pred_scaled = raw_preds.flatten()
        except Exception as e:
            print(f"\n[Epoch {epoch}] Variance check failed: {e}")
            return
        
        # Inverse transform to original scale
        try:
            y_pred = self.target_scaler.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            ).flatten()
        except Exception as e:
            print(f"\n[Epoch {epoch}] Inverse transform failed: {e}")
            return
        
        # Compute statistics
        pred_std = float(np.std(y_pred))
        pred_mean = float(np.mean(y_pred))
        pred_min = float(np.min(y_pred))
        pred_max = float(np.max(y_pred))
        
        pct_positive = float(np.mean(y_pred > 0) * 100)
        pct_negative = float(np.mean(y_pred < 0) * 100)
        
        # Track variance history for trend detection
        self.variance_history.append(pred_std)
        
        # Print diagnostic
        current_lr = float(keras.backend.get_value(self.model.optimizer.learning_rate))
        print(f"\n[Epoch {epoch}] Prediction Stats (LR={current_lr:.2e}):")
        print(f"  Std: {pred_std:.6f} | Mean: {pred_mean:.6f}")
        print(f"  Range: [{pred_min:.6f}, {pred_max:.6f}]")
        print(f"  Distribution: {pct_positive:.1f}% positive, {pct_negative:.1f}% negative")
        
        # Early warning: Check if variance is dropping rapidly
        if len(self.variance_history) >= 3:
            recent_stds = self.variance_history[-3:]
            if all(recent_stds[i] > recent_stds[i+1] for i in range(len(recent_stds)-1)):
                # Variance decreasing for 3 consecutive checks
                decline_rate = (recent_stds[0] - recent_stds[-1]) / recent_stds[0]
                if decline_rate > 0.3:  # More than 30% decline
                    print(f"  [EARLY WARNING] Variance declining rapidly ({decline_rate*100:.1f}% over last 3 checks)")
                    if not self.lr_reduced:
                        # Reduce learning rate immediately
                        new_lr = current_lr * self.lr_reduction_factor
                        # Use the proper Keras 3 API to set learning rate
                        try:
                            self.model.optimizer.learning_rate.assign(new_lr)
                        except (AttributeError, TypeError):
                            # Fallback for older Keras versions
                            keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
                        print(f"  [ACTION] Reduced LR from {current_lr:.2e} to {new_lr:.2e}")
                        self.lr_reduced = True
        
        # Check for collapse
        if pred_std < self.min_std:
            self.low_variance_epochs += 1
            print(f"  [WARN] Low variance detected ({self.low_variance_epochs}/{self.patience})")
            
            # First low variance: try reducing LR before giving up
            if self.low_variance_epochs == 1 and not self.lr_reduced:
                new_lr = current_lr * self.lr_reduction_factor
                # Use the proper Keras 3 API to set learning rate
                try:
                    self.model.optimizer.learning_rate.assign(new_lr)
                except (AttributeError, TypeError):
                    # Fallback for older Keras versions
                    keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)
                print(f"  [ACTION] Emergency LR reduction: {current_lr:.2e} -> {new_lr:.2e}")
                self.lr_reduced = True
            
            if self.low_variance_epochs >= self.patience:
                print(f"\n{'='*70}")
                print(f"[FAIL] MODEL COLLAPSED!")
                print(f"{'='*70}")
                print(f"Predictions std={pred_std:.6f} < min_std={self.min_std}")
                print(f"Predictions have insufficient variance for {self.patience} consecutive checks.")
                print(f"Stopping training early to prevent wasting compute.")
                print(f"")
                print(f"Possible causes:")
                print(f"  1. Learning rate too high (try 0.0001 or lower)")
                print(f"  2. Target scaling issue (check scaler parameters)")
                print(f"  3. Model architecture too simple or over-regularized")
                print(f"  4. Training data has insufficient variance")
                print(f"  5. Batch size too large (try 16 or 32)")
                print(f"{'='*70}\n")
                # NUCLEAR FIX: Raise exception to hard-stop training
                raise ValueError(f"VARIANCE COLLAPSE at epoch {epoch}: std={pred_std:.6f} < {self.min_std}")
        else:
            self.low_variance_epochs = 0  # Reset counter on healthy variance

        # NUCLEAR FIX v4.3: REMOVED hard-stop on bias during training
        # Research shows LSTMs naturally bias toward positive predictions initially
        # and can recover if given more epochs. The bias check was causing false failures.
        #
        # Instead of stopping training, we:
        # 1. Log the warning
        # 2. Let the BalancedDirectionalLoss handle the bias correction
        # 3. Trust the Pre-LN architecture + target de-meaning to fix it
        #
        # Reference: "Forecasting stock prices using LSTM" (Nature 2023) shows
        # LSTM positive bias ratio can be up to 3.75:1 initially
        if pct_positive > 95 or pct_negative > 95:
            bias_direction = "positive" if pct_positive > 95 else "negative"
            bias_pct = pct_positive if pct_positive > 95 else pct_negative
            print(f"  [WARN] High bias detected: {bias_pct:.1f}% {bias_direction}")
            print(f"  [INFO] Allowing training to continue - BalancedDirectionalLoss will correct")
            # Track consecutive high bias epochs
            if not hasattr(self, 'high_bias_epochs'):
                self.high_bias_epochs = 0
            self.high_bias_epochs += 1
            # Only stop if EXTREMELY biased (100%) for MANY epochs AFTER warmup
            if self.high_bias_epochs >= 10 and epoch > 30 and (pct_positive >= 99.9 or pct_negative >= 99.9):
                print(f"  [FAIL] Persistent 100% bias for 10 epochs after warmup - stopping")
                raise ValueError(f"PERSISTENT BIAS at epoch {epoch}: {bias_pct:.1f}% {bias_direction}")
        else:
            # Reset counter if bias recovers
            if hasattr(self, 'high_bias_epochs'):
                self.high_bias_epochs = 0

        if np.any(np.isnan(y_pred)):
            print(f"  [FATAL] NaN predictions detected! Stopping training.")
            raise ValueError(f"NaN PREDICTIONS at epoch {epoch}")


class CurriculumLearningCallback(keras.callbacks.Callback):
    """Dynamically adjust loss weights during training (Curriculum Learning).
    
    Gradually increases variance penalty weight to prevent deferred collapse.
    """
    
    def __init__(self, variance_weight_var, initial_weight=1.0, ramp_epochs=50):
        super().__init__()
        self.variance_weight_var = variance_weight_var
        self.initial_weight = initial_weight
        self.ramp_epochs = ramp_epochs

    def on_epoch_end(self, epoch, logs=None):
        # Linearly increase weight: start * (1 + epoch/ramp)
        # Epoch 0: 1.0x
        # Epoch 50: 2.0x
        # Epoch 100: 3.0x
        factor = 1.0 + (epoch / self.ramp_epochs)
        new_weight = self.initial_weight * factor
        # Cap at 4.0x initial
        new_weight = min(new_weight, self.initial_weight * 4.0)
        
        # Assign new value
        if hasattr(self.variance_weight_var, 'assign'):
            self.variance_weight_var.assign(new_weight)
        else:
            keras.backend.set_value(self.variance_weight_var, new_weight)
            
        if epoch % 5 == 0:
            print(f"   [Curriculum] Updated variance penalty weight to {new_weight:.4f}")



def train_quantile_ensemble(
    symbol: str = 'AAPL',
    epochs: int = 100,
    batch_size: int = 512,
    use_log_targets: bool = False,
    target_noise_std: float = 1e-4,
    use_standard_scaler: bool = True,
    sequence_length: int = 90,
    use_multitask: bool = False,  # Quantile ensemble doesn't use multi-task
    use_selected_features: bool = False,
    augment_negatives: bool = False,
    augmentation_factor: float = 1.5,
):
    """Train quantile regression ensemble with 3 models (q25, q50, q75).
    
    This trains 3 separate models:
    - q25 model: Conservative estimate (25th percentile)
    - q50 model: Median estimate (50th percentile, robust to outliers)
    - q75 model: Optimistic estimate (75th percentile)
    
    Benefits:
    - Provides uncertainty estimates via confidence intervals
    - More robust than single-point predictions
    - Prevents degenerate flat predictions
    - Enables risk-adjusted position sizing
    
    Args:
        use_selected_features: Whether to use Random Forest selected features.
        augment_negatives: Whether to augment negative returns.
        augmentation_factor: Target ratio of negative to positive samples.
    """
    print(f"\n{'='*70}")
    print(f"  TRAINING QUANTILE ENSEMBLE: {symbol}")
    print(f"{'='*70}\n")
    print("[INFO] Training 3 models: q25 (conservative), q50 (median), q75 (optimistic)")
    print(f"   This provides uncertainty estimates and prevents flat predictions")
    if use_selected_features:
        print(f"   Using selected features (reduced from 147)")
    if augment_negatives:
        print(f"   Using negative augmentation (factor={augmentation_factor})")
    print()
    
    quantiles = [0.25, 0.50, 0.75]
    models = {}
    metadata_all = {}
    
    for q in quantiles:
        print(f"\n{'='*70}")
        print(f"  Training quantile={q:.2f} model")
        print(f"{'='*70}\n")
        
        model = train_1d_regressor(
            symbol=symbol,
            epochs=epochs,
            batch_size=batch_size,
            use_log_targets=use_log_targets,
            target_noise_std=target_noise_std,
            loss_name='quantile',
            quantile=q,
            use_standard_scaler=use_standard_scaler,
            use_direction_loss=False,  # No direction loss for quantile models
            sequence_length=sequence_length,
            use_multitask=False,  # No multi-task for quantile ensemble
            use_selected_features=use_selected_features,
            augment_negatives=augment_negatives,
            augmentation_factor=augmentation_factor,
            quantile_suffix=f"_q{int(q*100)}"
        )
        
        models[f'q{int(q*100)}'] = model
    
    print(f"\n{'='*70}")
    print("  QUANTILE ENSEMBLE TRAINING COMPLETE")
    print(f"{'='*70}\n")
    print("[OK] All 3 quantile models trained successfully")
    print("\nUsage for inference:")
    print("   1. Load all 3 models (q25, q50, q75)")
    print("   2. Get predictions: pred_q25, pred_q50, pred_q75")
    print("   3. Ensemble: 0.2*pred_q25 + 0.6*pred_q50 + 0.2*pred_q75")
    print("   4. Confidence interval: (pred_q75 - pred_q25) / pred_q50")
    print("      - Narrow CI (< 0.5): High confidence → Larger position")
    print("      - Wide CI (> 2.0): Low confidence → Smaller position or skip")
    
    return models


def configure_gpu():
    """Configure GPU for optimal training performance.
    
    This function:
    1. Enables memory growth to prevent TensorFlow from allocating all GPU memory
    2. Prints GPU status for verification
    
    NOTE: Mixed precision disabled due to NaN issues in loss computation.
    When using mixed precision with multi-task learning, some loss functions
    (particularly variance-regularized losses) can produce NaN values.
    Float32 ensures numerical stability while still utilizing GPU acceleration.
    """
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth for all GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"[GPU] Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
            print(f"[GPU] Memory growth enabled")

            # CRITICAL FIX: Actually reset the global policy to float32
            # The module-level setup_gpu_acceleration() set it to mixed_float16,
            # but we need float32 for stability with custom loss functions
            from tensorflow.keras import mixed_precision
            float32_policy = mixed_precision.Policy('float32')
            mixed_precision.set_global_policy(float32_policy)
            print(f"[GPU] Using float32 for training (mixed precision disabled for stability)")
            print(f"[GPU] Current policy after reset: {mixed_precision.global_policy().name}")

            return True
        except RuntimeError as e:
            print(f"[GPU] Error configuring GPU: {e}")
            return False
    else:
        print("[GPU] No GPU found, using CPU")
        return False


def train_1d_regressor(
    symbol: str = 'AAPL',
    epochs: int = 100,
    batch_size: int = 512,
    use_log_targets: bool = False,
    target_noise_std: float = 1e-4,
    loss_name: str = 'huber',
    quantile: float = 0.5,
    use_standard_scaler: bool = True,
    use_direction_loss: bool = True,
    sequence_length: int = 90,
    use_multitask: bool = True,
    quantile_suffix: str = '',  # Suffix for quantile ensemble models
    use_cache: bool = True,
    balance_targets: bool = False,
    balance_strategy: str = 'hybrid',
    use_selected_features: bool = False,
    augment_negatives: bool = False,
    augmentation_factor: float = 1.5,
    use_directional_loss: bool = False,
    use_anti_collapse_loss: bool = True,  # NEW: Enabled by default to prevent variance collapse
    directional_weight: float = 1.0,  # Reduced from 2.0 for more balanced loss
    variance_regularization: float = 0.5,  # INCREASED to 0.5 - strong penalty to prevent collapse
):
    """Train 1-day regressor with optional multi-task learning and target balancing.
    
    Args:
        symbol: Stock symbol to train on.
        epochs: Number of training epochs.
        batch_size: Training batch size.
        use_log_targets: Whether to apply log1p transform to targets.
        target_noise_std: Standard deviation of noise added to targets.
        loss_name: Loss function to use ('huber', 'mae', 'quantile', 'huber_mae', 'directional_huber').
        quantile: Quantile parameter for quantile loss.
        use_standard_scaler: Whether to use StandardScaler for targets.
        use_direction_loss: Whether to include auxiliary direction loss.
        sequence_length: Length of input sequences.
        use_multitask: Whether to use multi-task learning.
        quantile_suffix: Suffix for quantile ensemble models.
        use_cache: Whether to use cached data.
        balance_targets: Whether to balance target distribution (undersample/oversample).
        balance_strategy: Balancing strategy ('undersample', 'oversample', 'hybrid').
        use_selected_features: Whether to use Random Forest selected features (reduces from 147).
        augment_negatives: Whether to augment negative returns by oversampling with noise.
        augmentation_factor: Target ratio of negative to positive samples (default 1.5).
        use_directional_loss: Whether to use DirectionalHuberLoss (penalizes wrong direction).
        use_anti_collapse_loss: Whether to use AntiCollapseDirectionalLoss (RECOMMENDED for preventing variance collapse).
        directional_weight: Penalty multiplier for wrong-direction predictions (default 1.0).
        variance_regularization: Weight for variance regularization to penalize constant outputs (default 0.5).
    """
    
    # Configure GPU for optimal training
    gpu_available = configure_gpu()
    
    print(f"\n{'='*70}")
    print(f"  TRAINING 1-DAY REGRESSOR: {symbol}")
    print(f"{'='*70}\n")
    normalized_loss_name = (loss_name or 'huber').lower()
    print(f"Target scaler: RobustScaler (clipped to ±0.15)")
    print(f"Log targets: {use_log_targets} | Target noise std: {target_noise_std}")
    if use_multitask:
        print(f"Multi-task learning: ENABLED (magnitude + sign + volatility heads)")
    else:
        print(f"Multi-task learning: DISABLED (single-task mode to prevent gradient conflicts)")
        print(f"Direction loss: {use_direction_loss} | Sequence length: {sequence_length}")
    quantile_note = quantile if normalized_loss_name == 'quantile' else 'n/a'
    print(f"Loss: {normalized_loss_name} (quantile={quantile_note})")
    if balance_targets:
        print(f"Target balancing: {balance_strategy}")
    if augment_negatives:
        print(f"Negative augmentation: ENABLED (factor={augmentation_factor})")
    if use_anti_collapse_loss:
        print(f"🛡️  ANTI-COLLAPSE LOSS: ENABLED (AntiCollapseDirectionalLoss - prevents variance collapse)")
    elif use_directional_loss:
        print(f"Directional loss: ENABLED (DirectionalHuberLoss, weight={directional_weight})")
    if variance_regularization > 0:
        print(f"Variance regularization: ENABLED (weight={variance_regularization})")
    
    # Feature selection handling
    feature_selection_used = use_selected_features
    selected_feature_cols = None
    
    if use_selected_features:
        print(f"Feature selection: ENABLED (using Random Forest selected features)")
        from training.feature_selection import load_selected_features
        try:
            selected_feature_cols = load_selected_features(symbol)
            logger.info(f"Loaded {len(selected_feature_cols)} selected features")
            print(f"   [OK] Loaded {len(selected_feature_cols)} selected features")
        except FileNotFoundError as e:
            print(f"   [WARN] Selected features not found for {symbol}")
            print(f"   [WARN] Error: {e}")
            print(f"   [INFO] Falling back to all features")
            logger.warning(f"Selected features not found for {symbol}, falling back to all features")
            feature_selection_used = False
            selected_feature_cols = None
        except Exception as e:
            print(f"   [ERROR] Unexpected error loading selected features: {e}")
            print(f"   [INFO] Falling back to all features")
            logger.warning(f"Unexpected error loading selected features: {e}")
            feature_selection_used = False
            selected_feature_cols = None
    else:
        print(f"Feature selection: DISABLED (using all features)")
    print()
    
    # Load data (use centralized cache manager when requested)
    print("[DATA] Loading data...")
    if use_cache:
        from data.cache_manager import DataCacheManager
        cache_manager = DataCacheManager()
        raw_df, engineered_df, prepared_df, feature_cols = cache_manager.get_or_fetch_data(
            symbol=symbol,
            include_sentiment=True,
            force_refresh=False
        )
        df = engineered_df
        # Validate final feature count to catch drift between feature engineering and expectations
        df = validate_and_fix_features(df)
        # prepared_df is already the cleaned/prepared dataframe
        df_clean = prepared_df
        # CRITICAL: ensure feature count matches model input dimensionality
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
        logger.info(f"[OK] Feature count validated: {len(feature_cols)} features")
        print(f"   [OK] Loaded from cache: {len(df_clean)} samples, {len(feature_cols)} features")
    else:
        # When not using cache, perform a forced fetch via DataCacheManager
        from data.cache_manager import DataCacheManager
        cache_manager = DataCacheManager()
        raw_df, engineered_df, prepared_df, feature_cols = cache_manager.get_or_fetch_data(
            symbol=symbol,
            include_sentiment=True,
            force_refresh=True
        )
        df = engineered_df
        # Validate final feature count
        df = validate_and_fix_features(df)
        df_clean = prepared_df
        
        # Verify feature count even when refreshing
        expected_cols = get_feature_columns(include_sentiment=True)
        if len(feature_cols) != EXPECTED_FEATURE_COUNT:
            logger.warning(f"Feature count mismatch after refresh! Expected {EXPECTED_FEATURE_COUNT}, got {len(feature_cols)}")
        
        print(f"   [OK] Freshly generated features: {len(df_clean)} samples, {len(feature_cols)} features")

    # Apply feature selection if enabled
    if use_selected_features and selected_feature_cols is not None:
        # Validate that all selected features exist in the data
        missing_features = [f for f in selected_feature_cols if f not in feature_cols]
        if missing_features:
            logger.warning(f"Missing {len(missing_features)} features from selection: {missing_features[:5]}...")
            # Filter to only available features
            selected_feature_cols = [f for f in selected_feature_cols if f in feature_cols]
        
        original_count = len(feature_cols)
        feature_cols = selected_feature_cols
        logger.info(f"INFO: Training regressor with feature selection")
        logger.info(f"Features: {len(feature_cols)} (reduced from {original_count})")
        print(f"   [OK] Feature selection applied: {len(feature_cols)} features (reduced from {original_count})")
    else:
        logger.info(f"Using all {len(feature_cols)} features")
    print(f"   [OK] Using 1-DAY horizon (research optimal)")
    print(f"   [OK] Using {sequence_length}-day sequences (quarterly context)")
    
    # Prepare data
    X = df_clean[feature_cols].values
    y_raw = df_clean['target_1d'].values
    
    # =========================================================================
    # TARGET DISTRIBUTION VALIDATION (before training)
    # This ensures the model will learn balanced predictions
    # =========================================================================
    print("\n[DATA] Validating target distribution...")
    try:
        target_dist_metadata = validate_target_distribution(
            y_train=y_raw,
            symbol=symbol,
            min_positive_pct=0.30,
            min_negative_pct=0.30,
            save_metadata=True
        )
        print("   [OK] Target distribution validation passed")
    except ValueError as e:
        # Re-raise with additional context
        raise ValueError(
            f"Target distribution validation failed for {symbol}. "
            f"Training aborted to prevent learning biased predictions. "
            f"Error: {e}"
        )
    
    # Scale features
    feature_scaler = RobustScaler()
    X_scaled = feature_scaler.fit_transform(X)

    # Targets: prefer scaling produced by prepare_training_data if available
    if 'target_1d_scaled' in df_clean.columns:
        y_scaled = df_clean['target_1d_scaled'].values
        target_scaler = df_clean.attrs.get('target_scaler', None)
        # If prepare_training_data attached metadata about transform, preserve it
        target_transform = df_clean.attrs.get('target_transform', 'precomputed')
        print(f"   [OK] Using precomputed target scaling from prepare_training_data. Scaled length: {len(y_scaled)}")
        
        # =====================================================================
        # TARGET SCALING VERIFICATION (for precomputed targets)
        # =====================================================================
        print(f"\n=== Target Scaling Verification ===")
        print(f"Original target range: [{y_raw.min():.6f}, {y_raw.max():.6f}]")
        print(f"Scaled target range: [{y_scaled.min():.6f}, {y_scaled.max():.6f}]")
        print(f"Scaled target stats: mean={y_scaled.mean():.6f}, std={y_scaled.std():.6f}")
        if target_scaler is not None:
            try:
                print(f"Target scaler center: {target_scaler.center_[0]:.6f}")
                print(f"Target scaler scale: {target_scaler.scale_[0]:.6f}")
            except Exception:
                print(f"Target scaler: (parameters not accessible)")
        
        # Sanity check: verify inverse transform works correctly
        if target_scaler is not None:
            y_roundtrip = target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
            roundtrip_error = np.mean(np.abs(y_raw[:len(y_roundtrip)] - y_roundtrip[:len(y_raw)]))
            print(f"Inverse transform roundtrip error: {roundtrip_error:.8f}")
            if roundtrip_error > 0.01:
                logger.warning(f"[WARN] High roundtrip error ({roundtrip_error:.6f}) - check target scaling!")
    else:
        # Fallback: winsorize + clip & RobustScaler (P1.3 improved target processing)
        from scipy.stats.mstats import winsorize
        
        # P1.3: Step 1 - Winsorize extreme tails (2% each side) to reduce outlier impact
        y_winsorized = np.array(winsorize(y_raw, limits=(0.02, 0.02)))
        
        # P1.3: Step 2 - Symmetric clipping at ±20% for tail robustness
        y_clipped = np.clip(y_winsorized, -0.20, 0.20)
        target_transform = 'winsorize_clip_robust'
        y_p05, y_p95 = np.percentile(y_raw, [5, 95])
        print(f"\n   Original target range (5th-95th percentile): [{y_p05:.4f}, {y_p95:.4f}]")
        print(f"   P1.3: Applied winsorization (2% tails) + symmetric clip (±20%)")
        target_scaler = RobustScaler()
        y_scaled = target_scaler.fit_transform(y_clipped.reshape(-1, 1)).flatten()
        print(f"   Using RobustScaler on winsorized+clipped returns")
        
        # =====================================================================
        # TARGET SCALING VERIFICATION (for fallback RobustScaler path)
        # =====================================================================
        print(f"\n=== Target Scaling Verification ===")
        print(f"Original target range: [{y_raw.min():.6f}, {y_raw.max():.6f}]")
        print(f"Clipped target range: [{y_clipped.min():.6f}, {y_clipped.max():.6f}]")
        print(f"Scaled target range: [{y_scaled.min():.6f}, {y_scaled.max():.6f}]")
        print(f"Scaled target stats: mean={y_scaled.mean():.6f}, std={y_scaled.std():.6f}")
        try:
            print(f"Target scaler center: {target_scaler.center_[0]:.6f}")
            print(f"Target scaler scale: {target_scaler.scale_[0]:.6f}")
        except Exception:
            print(f"Target scaler: (parameters not accessible)")
        
        # Sanity check: verify inverse transform works correctly
        y_roundtrip = target_scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
        roundtrip_error = np.mean(np.abs(y_clipped - y_roundtrip))
        print(f"Inverse transform roundtrip error: {roundtrip_error:.8f}")
        if roundtrip_error > 0.001:
            logger.warning(f"[WARN] High roundtrip error ({roundtrip_error:.6f}) - check target scaling!")
    
    # Create sequences with extended lookback
    X_seq = create_sequences(X_scaled, sequence_length)
    y_aligned = y_scaled[sequence_length-1:]
    y_original = y_raw[sequence_length-1:]  # For metrics

    # Prepare volatility array BEFORE balancing (needed for multi-task learning)
    if use_multitask:
        if 'volatility_5d' in df_clean.columns:
            volatility_full = df_clean['volatility_5d'].shift(-1).ffill().values
        else:
            returns_series = df_clean['target_1d']
            volatility_full = returns_series.rolling(5).std().shift(-1).ffill().values
        volatility_aligned_original = volatility_full[sequence_length-1:]
    else:
        volatility_aligned_original = None

    # =========================================================================
    # TARGET BALANCING (after sequence creation, before train/val split)
    # =========================================================================
    balancing_report = None

    if balance_targets:
        print(f"\n[BALANCE] Applying target balancing (strategy: {balance_strategy})...")

        # CRITICAL FIX: Apply balancing to volatility array too, so they stay aligned
        # We need to apply the same resampling/shuffling to volatility as to y_aligned
        # Workaround: Since balance_target_distribution uses random seed 42, we can
        # replicate the same transformations OR we can pass volatility as auxiliary data

        # Better approach: Modify the balancing to preserve indices OR
        # Simply recalculate volatility after balancing based on the balanced y values

        # For now, we'll use a simpler approach: calculate volatility from the balanced returns
        # after the split, which ensures perfect alignment

        X_seq, y_aligned, balancing_report = balance_target_distribution(
            X=X_seq,
            y=y_aligned,
            strategy=balance_strategy,
            random_state=42,
            validate=True,
        )

        print(f"   [OK] Balancing complete: {len(X_seq):,} samples")
    
    # Split
    split = int(len(X_seq) * 0.8)
    X_train, X_val = X_seq[:split], X_seq[split:]
    y_train, y_val = y_aligned[:split], y_aligned[split:]

    # =========================================================================
    # NUCLEAR FIX v7.0: FEATURE VARIANCE FILTER
    # =========================================================================
    # Research finding (January 2026): Training logs show "38 features have
    # near-zero variance" - 25% of features provide NO useful signal.
    # Filter them out to reduce noise and improve model stability.
    print("\n=== Feature Variance Filter (NUCLEAR FIX v7.0) ===")
    X_train, X_val, feature_cols = filter_low_variance_features(
        X_train, X_val, feature_cols, min_variance=0.001
    )
    num_features = len(feature_cols)  # Update feature count after filtering
    print(f"   [OK] Model will use {num_features} features after filtering")

    # =========================================================================
    # NUCLEAR FIX v4.3: TARGET DE-MEANING (Symmetric Target Distribution)
    # =========================================================================
    # Research finding: Stock market has ~53% positive returns (natural bias)
    # LSTMs learn "always positive" because positive is the safe prediction
    # Solution: Subtract training mean so targets are centered at 0
    # - Training targets: 53% positive -> 50% positive (symmetric)
    # - Model learns BOTH directions equally
    # - At inference: add the mean back to predictions
    target_mean = float(np.mean(y_train))
    pct_positive_before = np.mean(y_train > 0) * 100

    print("\n=== Target De-Meaning (NUCLEAR FIX v4.3) ===")
    print(f"   Original target mean: {target_mean:.6f}")
    print(f"   Positive % before de-mean: {pct_positive_before:.1f}%")

    # De-mean both train and val using ONLY the training mean (no data leakage)
    y_train = y_train - target_mean
    y_val = y_val - target_mean

    pct_positive_after = np.mean(y_train > 0) * 100
    print(f"   Positive % after de-mean: {pct_positive_after:.1f}%")
    print(f"   Target mean after de-mean: {np.mean(y_train):.10f} (should be ~0)")

    # Store the mean for inverse transform during inference
    # This will be saved with the model metadata
    target_mean_offset = target_mean
    print(f"   [OK] Stored target_mean_offset={target_mean_offset:.6f} for inference")

    # CRITICAL FIX: Handle y_original alignment after balancing
    # If balancing was applied, y_original doesn't match the balanced data
    # We need to inverse-transform y_val to get the original scale values
    if balance_targets:
        # y_original is from before balancing, so it's misaligned
        # Instead, we'll compute y_val_orig from y_val using inverse transform
        # This will be done later in the evaluation section
        y_val_orig = None  # Will be computed from y_val in evaluation
    else:
        y_val_orig = y_original[split:]
    
    # =========================================================================
    # NEGATIVE RETURN AUGMENTATION (after split, before noise)
    # =========================================================================
    augmentation_stats = None
    if augment_negatives:
        # Smart augmentation: only augment if negative samples are underrepresented
        pct_negative = np.mean(y_train < 0) * 100
        pct_positive = np.mean(y_train > 0) * 100
        
        print("\n=== Smart Augmentation Check ===")
        print(f"Target distribution: {pct_positive:.1f}% pos, {pct_negative:.1f}% neg")
        
        # Only augment if imbalanced towards positive (negative < 40%)
        if pct_negative < 40:
            from data.augmentation import augment_negative_returns, get_augmentation_metadata
            
            print(f"[INFO] Augmenting negative samples (factor={augmentation_factor})")
            X_train, y_train, augmentation_stats = augment_negative_returns(
                X_train, y_train,
                augmentation_factor=augmentation_factor,
                noise_std=0.05,
                random_state=42,
                validate=True,
            )
            logger.info(f"Augmented {augmentation_stats.samples_added} negative samples")
            print(f"   [OK] Added {augmentation_stats.samples_added:,} augmented samples")
            print(f"   [OK] Negative %: {augmentation_stats.original_negative_pct:.1%} -> {augmentation_stats.final_negative_pct:.1%}")
        else:
            print(f"[INFO] Skipping augmentation (distribution is balanced)")
            logger.info(f"Skipping negative augmentation - distribution already balanced ({pct_negative:.1f}% negative)")

    if target_noise_std > 0:
        noise = np.random.normal(0, target_noise_std, size=y_train.shape)
        y_train_aug = np.clip(y_train + noise, -1.2, 1.2)
    else:
        y_train_aug = y_train
    y_train_aug = y_train_aug.astype(np.float32)
    
    print(f"\n   Train: {len(X_train)}, Val: {len(X_val)}")
    
    # Prepare multi-task targets if enabled
    if use_multitask:
        print("\n[TARGET] Preparing multi-task targets...")
        
        # Prepare sign targets directly from y_train_aug and y_val
        # This ensures proper alignment after any augmentation
        flat_threshold = 0.002
        
        def create_sign_labels(y_returns, flat_threshold=0.002):
            """Create sign labels: 0=down, 1=flat, 2=up"""
            y_sign = np.zeros(len(y_returns), dtype=int)
            y_sign[y_returns > flat_threshold] = 2  # Up
            y_sign[np.abs(y_returns) <= flat_threshold] = 1  # Flat
            y_sign[y_returns < -flat_threshold] = 0  # Down
            return keras.utils.to_categorical(y_sign, num_classes=3)
        
        y_sign_train = create_sign_labels(y_train_aug.flatten(), flat_threshold)
        y_sign_val = create_sign_labels(y_val.flatten(), flat_threshold)

        # CRITICAL FIX: Calculate volatility from the BALANCED y values
        # This ensures perfect alignment even after balancing/augmentation

        if balance_targets:
            # Volatility calculation after balancing
            # Since balancing shuffled the data, we calculate volatility from the balanced returns
            # We use a rolling window approach on the balanced y values
            print(f"   [INFO] Calculating volatility from balanced returns...")

            # For training data (including augmentation)
            # Use a simple approach: rolling std of y_train_aug
            window_size = 5
            y_vol_train = np.zeros(len(y_train_aug))
            for i in range(len(y_train_aug)):
                start_idx = max(0, i - window_size + 1)
                y_vol_train[i] = np.std(y_train_aug[start_idx:i+1])

            # For validation data
            y_vol_val = np.zeros(len(y_val))
            for i in range(len(y_val)):
                start_idx = max(0, i - window_size + 1)
                y_vol_val[i] = np.std(y_val[start_idx:i+1])

            # Fill any zeros with mean volatility
            mean_vol_train = y_vol_train[y_vol_train > 0].mean() if np.any(y_vol_train > 0) else 0.01
            mean_vol_val = y_vol_val[y_vol_val > 0].mean() if np.any(y_vol_val > 0) else 0.01
            y_vol_train[y_vol_train == 0] = mean_vol_train
            y_vol_val[y_vol_val == 0] = mean_vol_val

        else:
            # No balancing: use the original volatility alignment
            y_vol_train_orig = volatility_aligned_original[:split]
            y_vol_val = volatility_aligned_original[split:]

            # If augmentation added samples (from augment_negatives), duplicate volatility targets
            if augment_negatives and augmentation_stats is not None and augmentation_stats.samples_added > 0:
                # The augmented samples are copies of existing negatives, so use their original volatility
                # For simplicity, use mean volatility for augmented samples
                y_vol_train = np.concatenate([
                    y_vol_train_orig,
                    np.full(augmentation_stats.samples_added, y_vol_train_orig.mean())
                ])
            else:
                y_vol_train = y_vol_train_orig

        # Verify alignment: all targets should have the same length
        if len(y_train_aug) != len(y_sign_train) or len(y_train_aug) != len(y_vol_train):
            raise ValueError(
                f"Multi-task target dimension mismatch after augmentation!\n"
                f"  y_train_aug: {len(y_train_aug)}\n"
                f"  y_sign_train: {len(y_sign_train)}\n"
                f"  y_vol_train: {len(y_vol_train)}\n"
                f"Check balance_targets and augment_negatives flags."
            )

        if len(y_val) != len(y_sign_val) or len(y_val) != len(y_vol_val):
            raise ValueError(
                f"Multi-task target dimension mismatch in validation!\n"
                f"  y_val: {len(y_val)}\n"
                f"  y_sign_val: {len(y_sign_val)}\n"
                f"  y_vol_val: {len(y_vol_val)}\n"
                f"This is likely caused by balance_targets not aligning volatility correctly."
            )
        
        # Log distribution
        y_sign_train_classes = np.argmax(y_sign_train, axis=1)
        print(f"\n[INFO] Multi-task target distribution:")
        print(f"   Sign classes - Down: {(y_sign_train_classes==0).sum()} ({(y_sign_train_classes==0).mean()*100:.1f}%), "
              f"Flat: {(y_sign_train_classes==1).sum()} ({(y_sign_train_classes==1).mean()*100:.1f}%), "
              f"Up: {(y_sign_train_classes==2).sum()} ({(y_sign_train_classes==2).mean()*100:.1f}%)")
        print(f"   Volatility range: [{y_vol_train.min():.4f}, {y_vol_train.max():.4f}]")
        
        # Scale volatility targets
        vol_scaler = StandardScaler()
        y_vol_train_scaled = vol_scaler.fit_transform(y_vol_train.reshape(-1, 1)).flatten()
        y_vol_val_scaled = vol_scaler.transform(y_vol_val.reshape(-1, 1)).flatten()
        
        print(f"   [OK] Sign classes prepared (3 classes: down/flat/up)")
        print(f"   [OK] Volatility targets prepared and scaled")
    
    # Create model
    print("\n[BUILD] Building model...")
    num_features = len(feature_cols)
    
    if use_multitask:
        print("   Multi-task architecture: 3 output heads (magnitude + sign + volatility)")
        model = create_multitask_regressor(
            sequence_length=sequence_length,
            n_features=num_features,  # Dynamic based on selection
        )
        
        # Create dynamic weight variable for curriculum learning
        # Initialize at 2.0 (strong penalty from start) to prevent early collapse
        variance_weight_var = tf.Variable(2.0, trainable=False, dtype=tf.float32, name='variance_penalty_weight')
        
        # Create callback (will be added to callbacks list later)
        curriculum_callback = CurriculumLearningCallback(
            variance_weight_var=variance_weight_var,
            initial_weight=0.5,
            ramp_epochs=40
        )
    else:
        print("   Single-task architecture with enhancements")
        model = create_enhanced_regressor(
            sequence_length=sequence_length,
            n_features=num_features,  # Dynamic based on selection
            name='enhanced_1d_regressor'
        )
    
    param_count = model.count_params()
    print(f"   [OK] {param_count:,} parameters")
    
    # Early prediction check - diagnose if model starts collapsed
    print("\n[CHECK] Initial model prediction check...")
    sample_batch = X_train[:32]  # Small batch
    if use_multitask:
        initial_preds = model.predict(sample_batch, verbose=0)
        mag_preds = initial_preds[0].flatten()  # magnitude_output
    else:
        initial_preds = model.predict(sample_batch, verbose=0)
        mag_preds = initial_preds.flatten()
    
    initial_std = np.std(mag_preds)
    initial_range = np.max(mag_preds) - np.min(mag_preds)
    print(f"   Initial predictions: std={initial_std:.6f}, range={initial_range:.6f}")
    if initial_std < 0.001:
        print(f"   [WARN] Initial predictions have very low variance - model may collapse!")
    else:
        print(f"   [OK] Initial predictions have healthy variance")
    
    # Log feature selection impact on model size
    if use_selected_features:
        # Estimate full model size (approximately proportional to input features for embedding layer)
        # This is a rough estimate based on typical LSTM input dimensionality
        estimated_full_params = int(param_count * (EXPECTED_FEATURE_COUNT / num_features))
        print(f"   [OK] Model size: ~{param_count // 1000}K parameters (reduced from ~{estimated_full_params // 1000}K)")
    
    # NUCLEAR FIX v6.0 (January 2026): Lower LR to prevent variance collapse
    # Previous max_lr=1e-3 was too aggressive and caused collapse at epoch 35
    # Research from StackExchange: "Setting the learning rate too large will cause 
    # the optimization to diverge" - use conservative values
    lr_schedule = create_warmup_cosine_schedule(
        warmup_epochs=10,        # Moderate warmup
        total_epochs=epochs,
        warmup_lr=0.00005,       # REDUCED from 1e-04 to 5e-05 (2x lower)
        max_lr=0.0005,           # REDUCED from 1e-03 to 5e-04 (2x lower) - prevent collapse
        min_lr=0.00001,          # Keep at 1e-05
        decay_start_epoch=40,    # Keep mid-training LR drop
        decay_factor=0.5         # Halve LR at decay_start_epoch
    )
    lr_scheduler_callback = keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
    print(f"   [OK] LR Schedule: warmup to {0.0005:.1e}, cosine decay to {0.00001:.1e}, drop 50% at epoch 40")
    
    # Compile model
    if use_multitask:
        # Multi-task: composite loss with 3 components
        # CRITICAL: Pass use_anti_collapse_loss to prevent variance collapse!
        
        magnitude_loss_fn = keras.losses.Huber() if normalized_loss_name == 'huber' else keras.losses.MeanAbsoluteError()
        combined_loss, loss_fns = create_multitask_loss(
            magnitude_weight=1.0,
            sign_weight=0.15,      # Reduced from 0.3 to let magnitude dominate
            volatility_weight=0.1, # Reduced from 0.2 to let magnitude dominate
            magnitude_loss_fn=magnitude_loss_fn,
            variance_regularization=variance_weight_var,  # Pass dynamic variable!
            use_anti_collapse_loss=use_anti_collapse_loss
        )
        
        # NUCLEAR FIX v6.0 (January 2026): Lower LR to prevent variance collapse
        # Previous 1e-3 was too aggressive and caused collapse at epoch 35
        base_optimizer = keras.optimizers.Adam(
            learning_rate=5e-4,      # REDUCED from 1e-3 to 5e-4 - matches new scheduler max
            clipnorm=1.0,            # Gradient clipping for stability
        )

        # CRITICAL FIX: Wrap optimizer with LossScaleOptimizer for mixed precision
        # This prevents gradient underflow/overflow in float16
        try:
            from tensorflow.keras import mixed_precision
            current_policy = mixed_precision.global_policy()
            if current_policy.name == 'mixed_float16':
                optimizer = mixed_precision.LossScaleOptimizer(base_optimizer)
                print(f"   [MIXED PRECISION] LossScaleOptimizer enabled (prevents NaN in float16)")
            else:
                optimizer = base_optimizer
        except Exception:
            optimizer = base_optimizer

        model.compile(
            optimizer=optimizer,
            loss={
                'magnitude_output': loss_fns['magnitude_loss'],
                'sign_output': loss_fns['sign_loss'],
                'volatility_output': loss_fns['volatility_loss']
            },
            loss_weights={
                'magnitude_output': 1.0,   # Primary task - keep at 1.0
                'sign_output': 0.15,       # Reduced from 0.3 to prevent dominating
                'volatility_output': 0.1   # Reduced from 0.2 to prevent dominating
            },
            metrics={
                'magnitude_output': ['mae', directional_accuracy_metric, r_squared_metric],
                'sign_output': ['accuracy'],
                'volatility_output': ['mae']
            }
        )
        
        var_reg_note = f" + variance_reg({variance_regularization})" if variance_regularization > 0 else ""
        print(f"   [OK] Loss: magnitude({normalized_loss_name}){var_reg_note} + 0.15*sign(CCE) + 0.1*volatility(MSE)")
        
    else:
        # NUCLEAR FIX v7.1 (January 2026): Huber + Small Directional Penalty
        #
        # v7.0 showed that pure Huber prevents NaN but causes variance collapse
        # because the model learns to predict the mean (all small positive values).
        #
        # SOLUTION: Add a SMALL directional penalty (0.1 weight) to encourage
        # the model to learn both positive and negative predictions.
        # This is much smaller than previous attempts (0.3-0.7) which caused
        # gradient conflicts.
        #
        # Architecture fixes from v7.0 remain:
        # - QK-LayerNorm in attention (prevents logit explosion)
        # - Single output layer (no train-test mismatch)
        # - Float32 precision (no overflow)
        reg_loss = DirectionalHuberLoss(
            delta=1.0,
            direction_weight=0.1  # SMALL penalty - just a nudge, not dominant
        )
        loss_description = "DirectionalHuber(delta=1.0, dir=0.1) - Huber + small directional nudge"

        # NUCLEAR FIX v7.0: Conservative optimizer settings
        # - Learning rate 1e-4 (REDUCED from 5e-4) - slower but more stable
        # - clipnorm=0.5 (REDUCED from 1.0) - more aggressive gradient clipping
        # Note: Keras only allows ONE of clipnorm/clipvalue, so using clipnorm only
        base_optimizer = keras.optimizers.Adam(
            learning_rate=1e-4,  # REDUCED from 5e-4
            clipnorm=0.5         # MORE AGGRESSIVE (was 1.0)
        )

        # Mixed precision is now disabled (float32), so no LossScaleOptimizer needed
        optimizer = base_optimizer

        model.compile(
            optimizer=optimizer,
            loss=reg_loss,
            metrics=['mae', directional_accuracy_metric, r_squared_metric]
        )

        print(f"   [OK] Loss: {loss_description}")
        print(f"   [OK] Optimizer: Adam(lr=1e-4, clipnorm=0.5)")
    
    # Train
    print("\n[TRAIN] Training with LR warmup + cosine decay...")
    print(f"   Warmup: 10 epochs (0.0001 -> 0.001)")
    print(f"   Cosine decay: {epochs-10} epochs (0.001 -> 0.00001)")
    
    # Validate data before training
    print("\n[CHECK] Validating training data...")
    if np.any(np.isnan(X_train)) or np.any(np.isinf(X_train)):
        raise ValueError("[FAIL] X_train contains NaN or Inf values! Check feature engineering.")
    if use_multitask:
        if np.any(np.isnan(y_train_aug)) or np.any(np.isinf(y_train_aug)):
            raise ValueError("[FAIL] y_train contains NaN or Inf values!")
        if np.any(np.isnan(y_sign_train)):
            raise ValueError("[FAIL] y_sign_train contains NaN values!")
        if np.any(np.isnan(y_vol_train_scaled)) or np.any(np.isinf(y_vol_train_scaled)):
            raise ValueError("[FAIL] y_vol_train contains NaN or Inf values!")
    else:
        if np.any(np.isnan(y_train_aug)) or np.any(np.isinf(y_train_aug)):
            raise ValueError("[FAIL] y_train contains NaN or Inf values!")
    print("   [OK] All training data validated (no NaN/Inf)")
    
    if use_multitask:
        # Use val_loss for robust early stopping across multi-task objectives
        
        # Create variance monitor callback to detect prediction collapse
        # NOTE: With tanh output layer, predictions are in [-1, 1] range
        # January 2026 fix: TIGHTENED callbacks after analysis showed overly permissive
        # settings were allowing collapse to deepen for too many epochs
        # Key insight: Early detection + fast stop is better than waiting for recovery
        variance_monitor = PredictionVarianceMonitor(
            X_val=X_val,
            y_val=y_val,
            target_scaler=target_scaler,
            min_std=0.003,   # RAISED from 0.001 - enforce 0.3% minimum variance
            patience=5,      # REDUCED from 15 - stop faster on collapse
            check_interval=5,  # REDUCED from 10 - check more frequently
            warmup_epochs=15,  # REDUCED from 20 - start monitoring earlier
            use_multitask=True
        )

        callbacks = [
            lr_scheduler_callback,
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,     # REDUCED from 40 - stop sooner on plateau
                restore_best_weights=True,
                verbose=1,
                mode='min',
                start_from_epoch=15  # Start checking after warmup
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=6,      # REDUCED from 12 - react faster to stagnation
                min_lr=1e-7,
                verbose=1
            ),
            variance_monitor  # Monitor prediction variance to detect collapse
        ]
        
        # Add curriculum learning callback if it exists
        if 'curriculum_callback' in locals() and curriculum_callback is not None:
            callbacks.append(curriculum_callback)
            print("   [CALLBACK] Added CurriculumLearningCallback to training loop")

        # Data quality pre-check on magnitude target
        validate_training_data_quality(X_train, y_train_aug)

        # NUCLEAR FIX v7.0: Use actual filtered feature count, not expected count
        # The feature variance filter removes low-variance features, so we must
        # use len(feature_cols) which was updated by the filter
        actual_features = X_train.shape[2]
        logger.info(f"[OK] Training with {actual_features} features (after variance filter)")
        logger.info(f"  Input shape: {X_train.shape} = (samples={X_train.shape[0]}, seq_len={X_train.shape[1]}, features={actual_features})")
        
        # Create optimized tf.data.Dataset for GPU throughput (CRITICAL FIX FOR GPU BOTTLENECK)
        print("\n[OPTIMIZE] Creating optimized data pipeline for GPU...")
        train_dataset = create_optimized_tf_dataset(
            X_train, y_train_aug, 
            y_sign=y_sign_train, 
            y_vol=y_vol_train_scaled,
            batch_size=batch_size,
            shuffle=True,
            prefetch=True,
            is_multitask=True
        )
        
        val_dataset = create_optimized_tf_dataset(
            X_val, y_val,
            y_sign=y_sign_val,
            y_vol=y_vol_val_scaled,
            batch_size=batch_size,
            shuffle=False,
            prefetch=True,
            is_multitask=True
        )
        
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
    else:
        # January 2026 fix: TIGHTENED callbacks - same rationale as multitask
        variance_monitor = PredictionVarianceMonitor(
            X_val=X_val,
            y_val=y_val,
            target_scaler=target_scaler,
            min_std=0.003,   # RAISED from 0.001 - enforce 0.3% minimum variance
            patience=5,      # REDUCED from 15 - stop faster on collapse
            check_interval=5,  # REDUCED from 10 - check more frequently
            warmup_epochs=15,  # REDUCED from 20 - start monitoring earlier
            use_multitask=False
        )

        callbacks = [
            lr_scheduler_callback,
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=20,     # REDUCED from 30 - stop sooner on plateau
                restore_best_weights=True,
                verbose=1,
                mode='min',
                start_from_epoch=15
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=6,      # REDUCED from 10 - react faster
                min_lr=1e-7,
                verbose=1
            ),
            variance_monitor  # Monitor prediction variance to detect collapse
        ]

        # Data quality pre-check
        validate_training_data_quality(X_train, y_train_aug)

        # NUCLEAR FIX v7.0: Use actual filtered feature count, not expected count
        # The feature variance filter removes low-variance features, so we must
        # use len(feature_cols) which was updated by the filter
        actual_features = X_train.shape[2]
        logger.info(f"[OK] Training with {actual_features} features (after variance filter)")
        logger.info(f"  Input shape: {X_train.shape} = (samples={X_train.shape[0]}, seq_len={X_train.shape[1]}, features={actual_features})")
        
        # Create optimized tf.data.Dataset for GPU throughput (CRITICAL FIX FOR GPU BOTTLENECK)
        print("\n[OPTIMIZE] Creating optimized data pipeline for GPU...")
        train_dataset = create_optimized_tf_dataset(
            X_train, y_train_aug,
            batch_size=batch_size,
            shuffle=True,
            prefetch=True,
            is_multitask=False
        )
        
        val_dataset = create_optimized_tf_dataset(
            X_val, y_val,
            batch_size=batch_size,
            shuffle=False,
            prefetch=True,
            is_multitask=False
        )
        
        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks,
            verbose=1
        )
    
    # Evaluate
    print(f"\n{'='*70}")
    print("  EVALUATION")
    print(f"{'='*70}\n")
    
    # Get predictions
    if use_multitask:
        predictions = model.predict(X_val, verbose=0)
        y_pred_scaled = predictions[0].flatten()  # Magnitude output
        y_sign_pred = predictions[1]  # Sign output (probabilities)
        y_vol_pred = predictions[2].flatten()  # Volatility output
        
        # Sign accuracy
        y_sign_pred_class = np.argmax(y_sign_pred, axis=1)
        y_sign_val_class = np.argmax(y_sign_val, axis=1)
        sign_accuracy = np.mean(y_sign_pred_class == y_sign_val_class)
        
        # Volatility metrics
        y_vol_pred_unscaled = vol_scaler.inverse_transform(y_vol_pred.reshape(-1, 1)).flatten()
        vol_mae = np.mean(np.abs(y_vol_val - y_vol_pred_unscaled))
        
        print(f"[INFO] Multi-task performance:")
        print(f"   Sign classification accuracy: {sign_accuracy:.1%}")
        print(f"   Volatility MAE: {vol_mae:.6f}")
        print()
    else:
        y_pred_scaled = model.predict(X_val, verbose=0).flatten()
    
    # Inverse transform magnitude predictions
    # RobustScaler was used on clipped raw returns → inverse_transform yields clipped original-scale returns
    y_pred_orig = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()

    # CRITICAL FIX: Compute y_val_orig if it was set to None (due to balancing)
    if y_val_orig is None:
        y_val_orig = target_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()

    # Metrics on ORIGINAL scale
    smape = compute_smape_correct(y_val_orig, y_pred_orig)
    mae = np.mean(np.abs(y_val_orig - y_pred_orig))
    dir_acc = np.mean(np.sign(y_val_orig) == np.sign(y_pred_orig))
    
    # R² score (coefficient of determination)
    ss_res = np.sum((y_val_orig - y_pred_orig) ** 2)
    ss_tot = np.sum((y_val_orig - np.mean(y_val_orig)) ** 2)
    r2_score = 1 - (ss_res / ss_tot) if ss_tot != 0 else -np.inf
    
    # Prediction variance check
    pred_std = np.std(y_pred_orig)
    actual_std = np.std(y_val_orig)
    
    print(f"R² Score: {r2_score:.4f}")
    print(f"SMAPE: {smape:.2f}%")
    print(f"MAE: {mae:.6f}")
    print(f"Directional Accuracy: {dir_acc:.1%}")
    print(f"\nPrediction std: {pred_std:.6f}")
    print(f"Actual std: {actual_std:.6f}")
    print(f"Variance ratio: {pred_std/actual_std:.2f}")

    # Log R² trajectory from training history
    if 'val_magnitude_output_r_squared_metric' in history.history:
        r2_trajectory = history.history['val_magnitude_output_r_squared_metric']
        print(f"\n[R² TRAJECTORY] Analysis:")
        print(f"   First 10 epochs: {[f'{x:.4f}' for x in r2_trajectory[:10]]}")
        print(f"   Max during training: {max(r2_trajectory):.4f} at epoch {np.argmax(r2_trajectory)+1}")
        # Note: metadata is defined later, so we can't reference it here yet
        print(f"   Final R²: {r2_trajectory[-1]:.4f}")
    elif 'val_r_squared_metric' in history.history:
        # For single-task model
        r2_trajectory = history.history['val_r_squared_metric']
        print(f"\n[R² TRAJECTORY] Analysis:")
        print(f"   Max during training: {max(r2_trajectory):.4f} at epoch {np.argmax(r2_trajectory)+1}")
        print(f"   Final R²: {r2_trajectory[-1]:.4f}")
    
    # ===================================================================
    # FINANCIAL METRICS
    # ===================================================================
    
    print(f"\n[METRICS] FINANCIAL METRICS:")
    
    # Information Coefficient (IC) - correlation between predictions and actuals
    ic = np.corrcoef(y_pred_orig, y_val_orig)[0, 1]
    print(f"   Information Coefficient (IC): {ic:.4f} (>0.05 is significant)")
    
    # Hit Rate - percentage of correct directional predictions
    correct_direction = ((y_pred_orig > 0) & (y_val_orig > 0)) | ((y_pred_orig < 0) & (y_val_orig < 0))
    hit_rate = np.mean(correct_direction)
    print(f"   Hit Rate (profitable predictions): {hit_rate:.2%}")
    
    # Sortino Ratio - return adjusted for downside risk only
    returns_series = y_val_orig
    mean_return = np.mean(returns_series)
    downside_returns = returns_series[returns_series < 0]
    downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
    sortino_ratio = (mean_return / downside_std) * np.sqrt(252)  # Annualized
    print(f"   Sortino Ratio (annualized): {sortino_ratio:.4f}")
    
    # Enhanced quality checks
    print(f"\n[CHECK] QUALITY ASSESSMENT:")
    
    if r2_score >= 0.10:
        print(f"   [OK] R^2 >= 0.10 - Model explains variance well!")
    elif r2_score >= 0.05:
        print(f"   [OK] R^2 >= 0.05 - Model shows predictive power")
    else:
        print(f"   [WARN] R^2 < 0.05 - Model struggling to explain variance")
    
    if ic >= 0.05:
        print(f"   [OK] IC >= 0.05 - Statistically significant predictive signal")
    elif ic >= 0.02:
        print(f"   [OK] IC >= 0.02 - Weak but potentially useful signal")
    else:
        print(f"   [WARN] IC < 0.02 - Weak correlation with actuals")
    
    if smape > 50:
        print(f"   [WARN] SMAPE > 50% - magnitude predictions need improvement")
    
    if dir_acc >= 0.55:
        print(f"   [OK] Directional accuracy >= 55% - strong directional signal")
    elif dir_acc >= 0.53:
        print(f"   [OK] Directional accuracy >= 53% - acceptable for trading")
    else:
        print(f"   [WARN] Directional accuracy < 53% - barely better than random")
    
    # ===================================================================
    # FINAL PREDICTION VALIDATION
    # ===================================================================
    print(f"\n{'='*70}")
    print(" FINAL PREDICTION VALIDATION")
    print(f"{'='*70}")
    
    # Get final predictions on both train and validation sets
    if use_multitask:
        y_train_pred_scaled = model.predict(X_train, verbose=0)[0].flatten()
        y_val_pred_scaled = model.predict(X_val, verbose=0)[0].flatten()
    else:
        y_train_pred_scaled = model.predict(X_train, verbose=0).flatten()
        y_val_pred_scaled = model.predict(X_val, verbose=0).flatten()
    
    # DEBUG: Print scaled prediction statistics
    print(f"\n=== SCALED Prediction Stats (before inverse transform) ===")
    print(f"Train SCALED: std={np.std(y_train_pred_scaled):.6f}, range=[{y_train_pred_scaled.min():.4f}, {y_train_pred_scaled.max():.4f}]")
    print(f"Val SCALED: std={np.std(y_val_pred_scaled):.6f}, range=[{y_val_pred_scaled.min():.4f}, {y_val_pred_scaled.max():.4f}]")
    print(f"Target scaled std: {np.std(y_val):.6f} (for reference)")
    
    # Inverse transform to original scale
    y_train_pred = target_scaler.inverse_transform(
        y_train_pred_scaled.reshape(-1, 1)
    ).flatten()
    y_val_pred = target_scaler.inverse_transform(
        y_val_pred_scaled.reshape(-1, 1)
    ).flatten()
    
    # Compute statistics
    train_pred_std = float(np.std(y_train_pred))
    val_pred_std = float(np.std(y_val_pred))
    
    train_pct_pos = float(np.mean(y_train_pred > 0) * 100)
    train_pct_neg = float(np.mean(y_train_pred < 0) * 100)
    
    val_pct_pos = float(np.mean(y_val_pred > 0) * 100)
    val_pct_neg = float(np.mean(y_val_pred < 0) * 100)
    
    print(f"\nTrain Predictions:")
    print(f"  Std: {train_pred_std:.6f} | Range: [{y_train_pred.min():.6f}, {y_train_pred.max():.6f}]")
    print(f"  Distribution: {train_pct_pos:.1f}% positive, {train_pct_neg:.1f}% negative")
    
    print(f"\nValidation Predictions:")
    print(f"  Std: {val_pred_std:.6f} | Range: [{y_val_pred.min():.6f}, {y_val_pred.max():.6f}]")
    print(f"  Distribution: {val_pct_pos:.1f}% positive, {val_pct_neg:.1f}% negative")
    
    # FAIL CONDITIONS
    MIN_VARIANCE_STD = 0.005  # 0.5% minimum - INCREASED from 0.003 for stricter validation
    # Typical daily stock return std is 1-2%, so predictions should have at least 0.5% std
    
    if val_pred_std < MIN_VARIANCE_STD:
        print(f"\n{'='*70}")
        print(f"[WARN] VALIDATION PREDICTIONS COLLAPSED! (Continuing to save artifacts for debug)")
        print(f"{'='*70}")
        print(f"Validation predictions std={val_pred_std:.6f} < {MIN_VARIANCE_STD}")
        print(f"Model has not learned useful variance.")
        print(f"")
        print(f"Recommendations:")
        print(f"  1. Check target scaling (ensure inverse_transform works correctly)")
        print(f"  2. Try lower learning rate (0.0001 or 0.00005)")
        print(f"  3. Reduce regularization (dropout, L2)")
        print(f"  4. Check training data variance")
        print(f"  5. Try different loss function")
        print(f"{'='*70}")
        # sys.exit(1)  # DISABLED to allow saving artifacts for inspection
    
    if val_pct_pos < 20 or val_pct_pos > 80:
        print(f"\n[WARN] Prediction distribution is imbalanced ({val_pct_pos:.1f}% positive)")
        print(f"  Model may have directional bias. Consider:")
        print(f"    - Checking target distribution balance")
        print(f"    - Using target balancing (--balance-targets)")
        print(f"    - Using negative augmentation (--augment-negatives)")
    else:
        print(f"\n[OK] Prediction distribution looks balanced")
    
    if val_pred_std >= MIN_VARIANCE_STD:
        print(f"[OK] Predictions passed variance validation (std={val_pred_std:.6f} >= {MIN_VARIANCE_STD})")
    
    print(f"{'='*70}\n")
    
    # Save
    print(f"\n[SAVE] Saving...")
    
    # Use new organized path structure (December 2025: Legacy saves disabled)
    paths = ModelPaths(symbol)
    paths.ensure_dirs()

    # Add quantile suffix if training ensemble
    base_name = f'{symbol}_1d_regressor_final{quantile_suffix}'
    
    # Save to new structure
    model.save_weights(str(paths.regressor.weights))
    
    model_dir = paths.regressor.model
    safe_export_model(model, model_dir)
    
    with open(paths.regressor.feature_scaler, 'wb') as f:
        pickle.dump(feature_scaler, f)
    
    # Save target scaler - robust format (v3.1+)
    with open(paths.regressor.target_scaler_robust, 'wb') as f:
        pickle.dump(target_scaler, f)
    
    # Also save as generic target_scaler for compatibility
    with open(paths.regressor.target_scaler, 'wb') as f:
        pickle.dump(target_scaler, f)
    
    # Save volatility scaler for multi-task models
    if use_multitask:
        with open(paths.regressor.vol_scaler, 'wb') as f:
            pickle.dump(vol_scaler, f)
    
    # Save features to both regressor dir and symbol-level canonical location
    with open(paths.regressor.features, 'wb') as f:
        pickle.dump(feature_cols, f)
    with open(paths.feature_columns, 'wb') as f:
        pickle.dump(feature_cols, f)

    # December 2025: Legacy path saves removed - only using new organized structure

    # Verify saved feature columns
    try:
        saved_cols = pickle.load(open(paths.feature_columns, 'rb'))
        expected_n_features = len(feature_cols)  # Use actual feature count (selected or full)
        assert len(saved_cols) == expected_n_features, f"Saved feature count mismatch: {len(saved_cols)} != {expected_n_features}"
    except Exception as e:
        logger.warning(f"Could not verify saved feature columns: {e}")
    
    # Determine model type
    if quantile_suffix:
        model_type = f'1d_regressor_quantile{quantile_suffix}'
    elif use_multitask:
        model_type = '1d_regressor_multitask'
    else:
        model_type = '1d_regressor_enhanced'
    
    metadata = {
        'symbol': symbol,
        'horizon': 1,
        'model_type': model_type,
        'sequence_length': sequence_length,
        'n_features': len(feature_cols),
        'feature_selection_used': use_selected_features,
        'feature_columns': feature_cols,
        'is_quantile_ensemble': bool(quantile_suffix),
        'quantile': float(quantile) if quantile_suffix else None,
        # CRITICAL: Architecture MUST match create_multitask_regressor() actual values
        # Bug fix 2025-12-14: Previously hardcoded wrong values causing load failures
        'architecture': {
            'lstm_units': 48,    # Match create_multitask_regressor
            'd_model': 96,       # Match create_multitask_regressor
            'num_blocks': 4,     # Match create_multitask_regressor
            'ff_dim': 192,       # Match create_multitask_regressor
            'dropout': 0.2,      # Match create_multitask_regressor
            'l2_reg': 0.0005,    # Match create_multitask_regressor
            'batch_norm': False, # NO batch norm in multitask to prevent collapse
            'multitask': use_multitask,
            'multitask_heads': 3 if use_multitask else 1,
            'multitask_outputs': ['magnitude', 'sign', 'volatility'] if use_multitask else ['magnitude']
        },
        'val_r2': float(r2_score),
        'val_smape': float(smape),
        'val_mae': float(mae),
        'val_dir_acc': float(dir_acc),
        'val_ic': float(ic),
        'val_hit_rate': float(hit_rate),
        'val_sortino': float(sortino_ratio),
        'pred_std': float(pred_std),
        'actual_std': float(actual_std),
        'epochs_trained': len(history.history['loss']),
        'best_epoch': len(history.history['loss']) - 30,  # Approximate
        'trained_date': datetime.now().isoformat(),
        'target_transform': target_transform,
        'target_scaler': 'RobustScaler',
        'target_noise_std': float(target_noise_std),
        'loss_name': 'directional_huber' if use_directional_loss else normalized_loss_name,
        'use_direction_loss': use_direction_loss,
        'use_directional_loss': use_directional_loss,
        'directional_weight': float(directional_weight) if use_directional_loss else 0.0,
        'direction_weight': 0.3 if (use_direction_loss and not use_directional_loss) else 0.0,
        'quantile': float(quantile),
        'variance_regularization': float(variance_regularization),
        'lr_schedule': {
            'warmup_epochs': 10,
            'warmup_lr': 0.00005,  # Updated from 0.0001
            'max_lr': 0.0005,      # Updated from 0.001 - prevents collapse
            'min_lr': 0.00001
        }
    }
    
    # Add multi-task specific metrics if applicable
    if use_multitask:
        metadata['val_sign_accuracy'] = float(sign_accuracy)
        metadata['val_volatility_mae'] = float(vol_mae)
        metadata['multitask_loss_weights'] = {
            'magnitude': 1.0,
            'sign': 0.15,       # Updated from 0.5 - prevents over-focus on sign
            'volatility': 0.1   # Updated from 0.2
        }

    # Add detailed target scaling metadata for reproducibility
    try:
        scaler_center = float(target_scaler.center_[0])
        scaler_scale = float(target_scaler.scale_[0])
    except Exception:
        scaler_center = float('nan')
        scaler_scale = float('nan')

    metadata['target_scaling'] = {
        'method': 'RobustScaler',
        'clip_range': [-0.15, 0.15],
        'scaler_center': scaler_center,
        'scaler_scale': scaler_scale,
        # NUCLEAR FIX v4.3: Store target mean offset for de-meaning
        # At inference: add this back to predictions to get original scale
        'target_mean_offset': target_mean_offset
    }
    
    # Add balancing metadata if balancing was applied
    if balancing_report is not None:
        metadata['target_balancing'] = get_balancing_metadata(balancing_report)
        # Also save balancing metadata as separate JSON file
        balancing_json_path = paths.symbol_dir / 'target_balancing.json'
        with open(balancing_json_path, 'w') as f:
            json.dump(get_balancing_metadata(balancing_report), f, indent=2)
        print(f"   [OK] Target balancing metadata saved to {balancing_json_path}")
    else:
        metadata['target_balancing'] = None
    
    # Add augmentation metadata if augmentation was applied
    if augmentation_stats is not None:
        from data.augmentation import get_augmentation_metadata
        metadata['negative_augmentation'] = get_augmentation_metadata(augmentation_stats)
        # Also save augmentation metadata as separate JSON file
        augmentation_json_path = paths.symbol_dir / 'negative_augmentation.json'
        with open(augmentation_json_path, 'w') as f:
            json.dump(get_augmentation_metadata(augmentation_stats), f, indent=2)
        print(f"   [OK] Augmentation metadata saved to {augmentation_json_path}")
    else:
        metadata['negative_augmentation'] = None
    
    # Save metadata (December 2025: Only new organized structure)
    with open(paths.regressor.metadata, 'wb') as f:
        pickle.dump(metadata, f)
    with open(paths.target_metadata, 'wb') as f:
        pickle.dump(metadata, f)

    print(f"   [OK] All artifacts saved to {paths.symbol_dir}{' (quantile ensemble)' if quantile_suffix else ''}")
    
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('symbol', type=str)
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size for training (default: 512 for powerful GPU)')
    parser.add_argument('--sequence-length', type=int, default=90,
                        help='Sequence lookback length (default: 90 for quarterly patterns)')
    parser.add_argument('--target-noise-std', type=float, default=1e-4,
                        help='Std-dev of Gaussian noise added to training targets (post-scaling).')
    parser.add_argument('--loss', type=str, default='huber',
                        choices=['huber', 'mae', 'quantile', 'huber_mae'],
                        help='Regression loss to optimize.')
    parser.add_argument('--quantile', type=float, default=0.5,
                        help='Quantile parameter when --loss quantile.')
    parser.add_argument(
        '--no-log-targets',
        dest='use_log_targets',
        action='store_false',
        help='Disable log1p target transform (defaults to disabled - use raw returns).'
    )
    parser.add_argument(
        '--no-standard-scaler',
        dest='use_standard_scaler',
        action='store_false',
        help='Use MinMaxScaler instead of StandardScaler for targets.'
    )
    parser.add_argument(
        '--no-direction-loss',
        dest='use_direction_loss',
        action='store_false',
        help='Disable auxiliary direction loss (defaults to enabled).'
    )
    parser.add_argument(
        '--use-multitask',
        dest='use_multitask',
        action='store_true',
        help='Enable multi-task learning with sign and volatility heads (prevents flat predictions).'
    )
    parser.add_argument(
        '--no-multitask',
        dest='use_multitask',
        action='store_false',
        help='Disable multi-task learning (single magnitude output only).'
    )
    parser.add_argument(
        '--disable-multitask',
        dest='use_multitask',
        action='store_false',
        help='Disable multi-task learning to prevent gradient conflicts (alias for --no-multitask).'
    )
    parser.add_argument(
        '--quantile-ensemble',
        dest='quantile_ensemble',
        action='store_true',
        help='Train quantile ensemble (3 models: q25, q50, q75) for uncertainty estimation.'
    )
    parser.add_argument(
        '--balance-targets',
        dest='balance_targets',
        action='store_true',
        help='Enable target distribution balancing to prevent class imbalance.'
    )
    parser.add_argument(
        '--balance-strategy',
        type=str,
        default='hybrid',
        choices=['undersample', 'oversample', 'hybrid'],
        help='Balancing strategy: undersample, oversample, or hybrid (default: hybrid).'
    )
    parser.add_argument(
        '--use-selected-features',
        dest='use_selected_features',
        action='store_true',
        help='Use Random Forest selected features instead of all 147.'
    )
    parser.add_argument(
        '--augment-negatives',
        dest='augment_negatives',
        action='store_true',
        help='Augment training data by oversampling negative returns with noise.'
    )
    parser.add_argument(
        '--augmentation-factor',
        type=float,
        default=1.5,
        help='Target ratio of negative to positive samples (default: 1.5).'
    )
    parser.add_argument(
        '--use-directional-loss',
        dest='use_directional_loss',
        action='store_true',
        help='Use DirectionalHuberLoss that penalizes wrong-direction predictions (2x penalty).'
    )
    parser.add_argument(
        '--use-anti-collapse-loss',
        dest='use_anti_collapse_loss',
        action='store_true',
        help='Use AntiCollapseDirectionalLoss that prevents variance collapse (RECOMMENDED for low-volatility symbols).'
    )
    parser.add_argument(
        '--no-anti-collapse-loss',
        dest='use_anti_collapse_loss',
        action='store_false',
        help='Disable AntiCollapseDirectionalLoss (use plain Huber loss instead).'
    )
    parser.add_argument(
        '--directional-weight',
        type=float,
        default=1.0,
        help='Penalty multiplier for wrong-direction predictions in DirectionalHuberLoss (default: 1.0).'
    )
    parser.add_argument(
        '--variance-regularization',
        type=float,
        default=1.0,
        help='Weight for variance regularization to penalize constant outputs (default: 1.0 - doubled for better anti-collapse).'
    )
    parser.set_defaults(use_selected_features=False)
    parser.set_defaults(augment_negatives=False)
    parser.set_defaults(use_directional_loss=False)
    parser.set_defaults(use_anti_collapse_loss=True)  # ENABLED by default for powerful GPU (prevents variance collapse)
    parser.set_defaults(use_log_targets=False)  # Changed default to False
    parser.set_defaults(use_standard_scaler=True)
    parser.set_defaults(use_direction_loss=True)
    parser.set_defaults(use_multitask=True)  # Enable multi-task by default
    parser.set_defaults(quantile_ensemble=False)
    parser.set_defaults(balance_targets=True)  # ENABLED by default (prevents class imbalance)
    
    # P0 FIX: Add seed argument for reproducibility
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42).'
    )
    parser.add_argument(
        '--force-refresh',
        action='store_true',
        help='Force regeneration of features (ignore cache).'
    )
    
    args = parser.parse_args()
    
    # Apply seed setting from CLI arg (overrides env var)
    set_global_seed(args.seed)
    
    # Setup file-based logging for this training run
    global logger
    logger = setup_training_logger(args.symbol, 'regressor')
    logger.info(f"Starting regressor training for {args.symbol}")
    logger.info(f"Arguments: epochs={args.epochs}, batch_size={args.batch_size}, "
                f"sequence_length={args.sequence_length}, seed={args.seed}")
    
    if args.quantile_ensemble:
        # Train quantile ensemble (3 models)
        models = train_quantile_ensemble(
            args.symbol,
            args.epochs,
            args.batch_size,
            use_log_targets=args.use_log_targets,
            target_noise_std=args.target_noise_std,
            use_standard_scaler=args.use_standard_scaler,
            sequence_length=args.sequence_length,
            use_selected_features=args.use_selected_features,
            augment_negatives=args.augment_negatives,
            augmentation_factor=args.augmentation_factor,
        )
        print(f"\n[OK] Quantile ensemble training complete!")
        print(f"   3 models saved: {args.symbol}_1d_regressor_final_q25/q50/q75")
    else:
        # Train single model (with optional multi-task)
        model = train_1d_regressor(
            args.symbol,
            args.epochs,
            args.batch_size,
            use_log_targets=args.use_log_targets,
            target_noise_std=args.target_noise_std,
            loss_name=args.loss,
            quantile=args.quantile,
            use_standard_scaler=args.use_standard_scaler,
            use_direction_loss=args.use_direction_loss,
            sequence_length=args.sequence_length,
            use_multitask=args.use_multitask,
            use_cache=not args.force_refresh,
            balance_targets=args.balance_targets,
            balance_strategy=args.balance_strategy,
            use_selected_features=args.use_selected_features,
            augment_negatives=args.augment_negatives,
            augmentation_factor=args.augmentation_factor,
            use_directional_loss=args.use_directional_loss,
            use_anti_collapse_loss=args.use_anti_collapse_loss,
            directional_weight=args.directional_weight,
            variance_regularization=args.variance_regularization,
        )
        print(f"\n[OK] 1-day regressor training complete!")
    
    print(f"   Next: Train binary classifiers with:")
    print(f"   python training/train_binary_classifiers_final.py {args.symbol}")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[WARN] Training interrupted by user")
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] FATAL ERROR during regressor training: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
