"""
Data Augmentation Module for Stock Return Prediction.

This module provides functions to augment training data, particularly to address
class imbalance in return prediction. The primary focus is on ensuring the model
sees sufficient negative return samples during training.

Key Functions:
- augment_negative_returns: Duplicate and noise-augment negative return samples
- validate_augmentation: Verify augmentation quality and sequence integrity

Usage:
    from data.augmentation import augment_negative_returns
    
    X_aug, y_aug, stats = augment_negative_returns(
        X_train, y_train, augmentation_factor=1.5
    )
"""

import logging
from dataclasses import dataclass, field
from typing import Tuple, Optional, List

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class AugmentationStats:
    """
    Statistics from negative return augmentation.
    
    Attributes:
        original_samples: Total samples before augmentation.
        samples_added: Number of augmented samples added.
        final_samples: Total samples after augmentation.
        original_negative_count: Negative samples before augmentation.
        final_negative_count: Negative samples after augmentation.
        original_negative_pct: Percentage of negatives before.
        final_negative_pct: Percentage of negatives after.
        original_positive_count: Positive samples before augmentation.
        noise_std: Standard deviation of noise applied.
        augmentation_factor: Target factor for negative/positive ratio.
        sequence_shape: Shape of input sequences (samples, seq_len, features).
    """
    
    original_samples: int = 0
    samples_added: int = 0
    final_samples: int = 0
    original_negative_count: int = 0
    final_negative_count: int = 0
    original_negative_pct: float = 0.0
    final_negative_pct: float = 0.0
    original_positive_count: int = 0
    noise_std: float = 0.05
    augmentation_factor: float = 1.5
    sequence_shape: Tuple[int, ...] = field(default_factory=tuple)


@dataclass
class AugmentationValidationResult:
    """
    Result of augmentation validation checks.
    
    Attributes:
        is_valid: Whether all validation checks passed.
        warnings: List of warning messages.
        errors: List of error messages.
        outlier_count: Number of outliers detected in augmented data.
        sequence_integrity_ok: Whether sequence structure is preserved.
        feature_range_ok: Whether feature values are within expected range.
    """
    
    is_valid: bool = True
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    outlier_count: int = 0
    sequence_integrity_ok: bool = True
    feature_range_ok: bool = True


def augment_negative_returns(
    X_seq: np.ndarray,
    y: np.ndarray,
    augmentation_factor: float = 1.5,
    noise_std: float = 0.05,
    random_state: int = 42,
    validate: bool = True,
    max_augmentation_ratio: float = 2.0,
) -> Tuple[np.ndarray, np.ndarray, AugmentationStats]:
    """
    Augment training data by duplicating negative return samples with noise.
    
    This function addresses class imbalance by oversampling negative return
    samples. Each duplicated sample has small Gaussian noise added to its
    features to create variation and prevent overfitting to specific patterns.
    
    The augmentation process:
    1. Identify all negative return samples (y < 0)
    2. Calculate target number of negatives based on augmentation_factor
    3. Randomly sample (with replacement) from existing negatives
    4. Add small Gaussian noise to feature values
    5. Concatenate with original data and shuffle
    
    Args:
        X_seq: Feature sequences of shape (n_samples, seq_length, n_features).
            Each sample is a sequence of historical feature vectors.
        y: Target values of shape (n_samples,).
            Typically 1-day forward returns.
        augmentation_factor: Target ratio of negative to positive samples.
            Default 1.5 means 1.5x as many negatives as positives.
        noise_std: Standard deviation of Gaussian noise added to features.
            Default 0.05 (5% of feature std). Higher values create more
            diverse augmented samples but may introduce unrealistic patterns.
        random_state: Random seed for reproducibility.
        validate: Whether to run validation checks on augmented data.
        max_augmentation_ratio: Maximum allowed ratio of augmented to original
            samples. Prevents excessive duplication. Default 2.0.
    
    Returns:
        Tuple of:
        - X_augmented: Augmented feature sequences, shape (n_new_samples, seq_len, n_features)
        - y_augmented: Augmented target values, shape (n_new_samples,)
        - stats: AugmentationStats with before/after metrics
    
    Raises:
        ValueError: If input shapes are invalid or augmentation fails validation.
    
    Example:
        >>> X_train, y_train, stats = augment_negative_returns(
        ...     X_train, y_train, augmentation_factor=1.5
        ... )
        >>> print(f"Added {stats.samples_added} samples")
        >>> print(f"Negative %: {stats.original_negative_pct:.1%} -> {stats.final_negative_pct:.1%}")
    """
    # Set random seed for reproducibility
    np.random.seed(random_state)
    
    # =========================================================================
    # Input validation
    # =========================================================================
    X_seq = np.asarray(X_seq)
    y = np.asarray(y).flatten()
    
    if len(X_seq) != len(y):
        raise ValueError(
            f"X_seq and y must have same length: {len(X_seq)} != {len(y)}"
        )
    
    if X_seq.ndim != 3:
        raise ValueError(
            f"X_seq must be 3D (samples, seq_length, features), got shape {X_seq.shape}"
        )
    
    if len(X_seq) == 0:
        raise ValueError("Empty input arrays provided")
    
    n_samples, seq_length, n_features = X_seq.shape
    
    # =========================================================================
    # Step (a): Identify negative return samples
    # =========================================================================
    negative_mask = y < 0
    positive_mask = y > 0
    
    negative_X = X_seq[negative_mask]
    negative_y = y[negative_mask]
    
    num_positive = int(np.sum(positive_mask))
    num_negative = int(np.sum(negative_mask))
    num_zero = int(np.sum(y == 0))
    
    original_negative_pct = num_negative / n_samples if n_samples > 0 else 0.0
    
    logger.info("")
    logger.info("=" * 60)
    logger.info("NEGATIVE RETURN AUGMENTATION")
    logger.info("=" * 60)
    logger.info(f"   Original samples: {n_samples:,}")
    logger.info(f"   Positive: {num_positive:,} ({num_positive/n_samples*100:.1f}%)")
    logger.info(f"   Negative: {num_negative:,} ({num_negative/n_samples*100:.1f}%)")
    if num_zero > 0:
        logger.info(f"   Zero: {num_zero:,} ({num_zero/n_samples*100:.1f}%)")
    
    # =========================================================================
    # Step (b): Calculate how many samples to add
    # =========================================================================
    target_negative = int(num_positive * augmentation_factor)
    num_to_add = max(0, target_negative - num_negative)
    
    # Apply maximum augmentation ratio limit
    max_to_add = int(n_samples * max_augmentation_ratio)
    if num_to_add > max_to_add:
        logger.warning(
            f"   Capping augmentation: {num_to_add} -> {max_to_add} "
            f"(max ratio: {max_augmentation_ratio})"
        )
        num_to_add = max_to_add
    
    logger.info(f"   Target negative count: {target_negative:,}")
    logger.info(f"   Samples to add: {num_to_add:,}")
    
    # Initialize stats
    stats = AugmentationStats(
        original_samples=n_samples,
        samples_added=0,
        final_samples=n_samples,
        original_negative_count=num_negative,
        final_negative_count=num_negative,
        original_negative_pct=original_negative_pct,
        final_negative_pct=original_negative_pct,
        original_positive_count=num_positive,
        noise_std=noise_std,
        augmentation_factor=augmentation_factor,
        sequence_shape=X_seq.shape,
    )
    
    # =========================================================================
    # Early exit if no augmentation needed
    # =========================================================================
    if num_to_add == 0:
        logger.info("   No augmentation needed - negative samples sufficient")
        logger.info("=" * 60)
        return X_seq.copy(), y.copy(), stats
    
    if num_negative == 0:
        logger.warning("   No negative samples to augment from!")
        logger.info("=" * 60)
        return X_seq.copy(), y.copy(), stats
    
    # =========================================================================
    # Step (c): Duplicate random negative samples
    # =========================================================================
    indices = np.random.choice(
        len(negative_X), 
        size=num_to_add, 
        replace=True  # Allow sampling same sample multiple times
    )
    
    X_augmented = negative_X[indices].copy()
    y_augmented = negative_y[indices].copy()
    
    # =========================================================================
    # Step (d): Add small noise to augmented sequences
    # =========================================================================
    # Noise is proportional to feature standard deviation
    # This creates realistic variations without introducing outliers
    
    # Calculate per-feature std for scaling noise appropriately
    feature_stds = np.std(X_seq.reshape(-1, n_features), axis=0)
    feature_stds = np.maximum(feature_stds, 1e-8)  # Avoid division by zero
    
    # Generate noise scaled by feature std
    noise = np.random.normal(0, 1, X_augmented.shape)
    
    # Scale noise by feature std and noise_std parameter
    for f in range(n_features):
        noise[:, :, f] *= feature_stds[f] * noise_std
    
    X_augmented = X_augmented + noise
    
    logger.info(f"   Added Gaussian noise: std={noise_std:.2%} of feature std")
    
    # =========================================================================
    # Step (e): Concatenate and shuffle
    # =========================================================================
    X_combined = np.concatenate([X_seq, X_augmented], axis=0)
    y_combined = np.concatenate([y, y_augmented], axis=0)
    
    # Shuffle while maintaining X-y correspondence
    shuffle_indices = np.random.permutation(len(X_combined))
    X_combined = X_combined[shuffle_indices]
    y_combined = y_combined[shuffle_indices]
    
    # =========================================================================
    # Update stats
    # =========================================================================
    final_negative_count = int(np.sum(y_combined < 0))
    final_samples = len(y_combined)
    final_negative_pct = final_negative_count / final_samples if final_samples > 0 else 0.0
    
    stats.samples_added = num_to_add
    stats.final_samples = final_samples
    stats.final_negative_count = final_negative_count
    stats.final_negative_pct = final_negative_pct
    
    logger.info("")
    logger.info("   AFTER AUGMENTATION:")
    logger.info(f"   Total samples: {final_samples:,} (+{num_to_add:,})")
    logger.info(f"   Negative: {final_negative_count:,} ({final_negative_pct*100:.1f}%)")
    logger.info(f"   Positive: {num_positive:,} ({num_positive/final_samples*100:.1f}%)")
    
    # =========================================================================
    # Validation
    # =========================================================================
    if validate:
        validation_result = validate_augmentation(
            X_original=X_seq,
            X_augmented=X_combined,
            y_original=y,
            y_augmented=y_combined,
            stats=stats,
        )
        
        if not validation_result.is_valid:
            for error in validation_result.errors:
                logger.error(f"   ❌ {error}")
            raise ValueError(
                f"Augmentation validation failed: {validation_result.errors}"
            )
        
        if validation_result.warnings:
            for warning in validation_result.warnings:
                logger.warning(f"   ⚠️ {warning}")
        
        logger.info("   ✓ Validation passed")
    
    logger.info("=" * 60)
    
    return X_combined, y_combined, stats


def validate_augmentation(
    X_original: np.ndarray,
    X_augmented: np.ndarray,
    y_original: np.ndarray,
    y_augmented: np.ndarray,
    stats: AugmentationStats,
    outlier_threshold: float = 5.0,
) -> AugmentationValidationResult:
    """
    Validate augmented data quality and sequence integrity.
    
    Performs the following checks:
    1. Sequence structure preserved (shape, dimensions)
    2. No extreme outliers introduced (values within threshold * std)
    3. Feature ranges reasonable (within original range + noise margin)
    4. No NaN or Inf values introduced
    5. Target distribution improved (more balanced)
    
    Args:
        X_original: Original feature sequences.
        X_augmented: Augmented feature sequences.
        y_original: Original target values.
        y_augmented: Augmented target values.
        stats: AugmentationStats from augmentation.
        outlier_threshold: Number of standard deviations beyond which
            a value is considered an outlier. Default 5.0.
    
    Returns:
        AugmentationValidationResult with validation status and details.
    """
    result = AugmentationValidationResult()
    
    # =========================================================================
    # Check 1: Sequence structure preserved
    # =========================================================================
    if X_augmented.ndim != 3:
        result.is_valid = False
        result.sequence_integrity_ok = False
        result.errors.append(
            f"Augmented X has wrong dimensions: {X_augmented.ndim} != 3"
        )
        return result  # Early return to prevent shape errors later
    
    if X_original.shape[1:] != X_augmented.shape[1:]:
        result.is_valid = False
        result.sequence_integrity_ok = False
        result.errors.append(
            f"Sequence shape mismatch: {X_original.shape[1:]} != {X_augmented.shape[1:]}"
        )
        return result  # Early return to prevent shape errors later
    
    if len(X_augmented) != len(y_augmented):
        result.is_valid = False
        result.errors.append(
            f"X/y length mismatch: {len(X_augmented)} != {len(y_augmented)}"
        )
    
    # =========================================================================
    # Check 2: No NaN or Inf values
    # =========================================================================
    if np.any(np.isnan(X_augmented)):
        result.is_valid = False
        result.errors.append("NaN values detected in augmented features")
    
    if np.any(np.isinf(X_augmented)):
        result.is_valid = False
        result.errors.append("Inf values detected in augmented features")
    
    if np.any(np.isnan(y_augmented)):
        result.is_valid = False
        result.errors.append("NaN values detected in augmented targets")
    
    # =========================================================================
    # Check 3: No extreme outliers introduced
    # =========================================================================
    # Calculate original feature statistics
    X_original_flat = X_original.reshape(-1, X_original.shape[-1])
    X_augmented_flat = X_augmented.reshape(-1, X_augmented.shape[-1])
    
    original_means = np.mean(X_original_flat, axis=0)
    original_stds = np.std(X_original_flat, axis=0)
    original_stds = np.maximum(original_stds, 1e-8)
    
    # Check how many values are beyond threshold
    z_scores = np.abs((X_augmented_flat - original_means) / original_stds)
    outlier_mask = z_scores > outlier_threshold
    outlier_count = int(np.sum(outlier_mask))
    
    result.outlier_count = outlier_count
    
    total_values = X_augmented_flat.size
    outlier_pct = outlier_count / total_values if total_values > 0 else 0
    
    # Allow up to 1% outliers (noise can push some values slightly beyond)
    if outlier_pct > 0.01:
        result.warnings.append(
            f"High outlier rate: {outlier_count:,} values ({outlier_pct:.2%}) "
            f"beyond {outlier_threshold}σ"
        )
    
    # Fail if more than 5% are outliers
    if outlier_pct > 0.05:
        result.is_valid = False
        result.feature_range_ok = False
        result.errors.append(
            f"Excessive outliers: {outlier_pct:.1%} of values beyond {outlier_threshold}σ"
        )
    
    # =========================================================================
    # Check 4: Feature ranges reasonable
    # =========================================================================
    original_min = np.min(X_original_flat, axis=0)
    original_max = np.max(X_original_flat, axis=0)
    original_range = original_max - original_min
    
    # Allow 20% extension beyond original range due to noise
    margin = 0.2 * original_range
    
    augmented_min = np.min(X_augmented_flat, axis=0)
    augmented_max = np.max(X_augmented_flat, axis=0)
    
    below_range = augmented_min < (original_min - margin)
    above_range = augmented_max > (original_max + margin)
    
    n_features_out_of_range = int(np.sum(below_range | above_range))
    
    if n_features_out_of_range > X_original.shape[-1] * 0.1:  # >10% features
        result.warnings.append(
            f"{n_features_out_of_range} features have values outside expected range"
        )
    
    # =========================================================================
    # Check 5: Target distribution improved
    # =========================================================================
    original_neg_pct = np.mean(y_original < 0)
    augmented_neg_pct = np.mean(y_augmented < 0)
    
    # Warn if augmentation made distribution worse
    if augmented_neg_pct < original_neg_pct - 0.01:
        result.warnings.append(
            f"Negative percentage decreased: {original_neg_pct:.1%} -> {augmented_neg_pct:.1%}"
        )
    
    return result


def get_augmentation_metadata(stats: AugmentationStats) -> dict:
    """
    Convert AugmentationStats to a JSON-serializable dictionary.
    
    Args:
        stats: AugmentationStats instance.
    
    Returns:
        Dictionary with augmentation metadata.
    """
    return {
        'original_samples': stats.original_samples,
        'samples_added': stats.samples_added,
        'final_samples': stats.final_samples,
        'original_negative_count': stats.original_negative_count,
        'final_negative_count': stats.final_negative_count,
        'original_negative_pct': round(stats.original_negative_pct, 4),
        'final_negative_pct': round(stats.final_negative_pct, 4),
        'original_positive_count': stats.original_positive_count,
        'noise_std': stats.noise_std,
        'augmentation_factor': stats.augmentation_factor,
        'sequence_shape': list(stats.sequence_shape),
        'augmentation_type': 'negative_return_duplication',
    }


def augment_with_time_shift(
    X_seq: np.ndarray,
    y: np.ndarray,
    shift_range: Tuple[int, int] = (-2, 2),
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Augment by applying small time shifts to sequences.
    
    This creates synthetic samples by shifting the feature sequences
    slightly in time, simulating minor timing variations in patterns.
    
    Note: This is an optional augmentation method for future use.
    The primary augmentation is augment_negative_returns().
    
    Args:
        X_seq: Feature sequences (n_samples, seq_length, n_features).
        y: Target values (n_samples,).
        shift_range: Range of shifts to apply (min, max). Negative shifts
            move patterns earlier, positive shifts move them later.
        random_state: Random seed.
    
    Returns:
        Tuple of (X_augmented, y_augmented, stats_dict).
    """
    np.random.seed(random_state)
    
    X_seq = np.asarray(X_seq)
    y = np.asarray(y).flatten()
    
    n_samples, seq_length, n_features = X_seq.shape
    
    # Generate random shifts for each sample
    shifts = np.random.randint(shift_range[0], shift_range[1] + 1, size=n_samples)
    
    X_shifted = np.zeros_like(X_seq)
    
    for i, shift in enumerate(shifts):
        if shift == 0:
            X_shifted[i] = X_seq[i]
        elif shift > 0:
            # Shift right (pad left with first value)
            X_shifted[i, shift:, :] = X_seq[i, :-shift, :]
            X_shifted[i, :shift, :] = X_seq[i, 0:1, :]  # Replicate first timestep
        else:
            # Shift left (pad right with last value)
            shift = abs(shift)
            X_shifted[i, :-shift, :] = X_seq[i, shift:, :]
            X_shifted[i, -shift:, :] = X_seq[i, -1:, :]  # Replicate last timestep
    
    stats = {
        'augmentation_type': 'time_shift',
        'shift_range': shift_range,
        'samples_processed': n_samples,
    }
    
    return X_shifted, y.copy(), stats
