"""
Target Balancing Module for Regression Training.

This module provides functions to balance target distributions in regression
datasets, preventing model bias toward majority classes (e.g., too many
positive or negative returns).

Strategies:
- undersample: Remove samples from majority class
- oversample: Augment minority class with noise-based interpolation
- hybrid: Undersample majority to 60%, oversample minority to 40%
"""

import numpy as np
import logging
from typing import Tuple, Literal, Optional, Dict, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class BalancingReport:
    """
    Report containing target balancing results and metadata.

    Attributes:
        strategy: The balancing strategy used.
        original_counts: Dictionary with original class counts.
        balanced_counts: Dictionary with balanced class counts.
        samples_removed: Number of samples removed (undersampling).
        samples_added: Number of samples added (oversampling).
        original_ratio: Original positive/negative ratio.
        balanced_ratio: Balanced positive/negative ratio.
        random_state: Random state used for reproducibility.
    """

    strategy: str
    original_counts: Dict[str, int] = field(default_factory=dict)
    balanced_counts: Dict[str, int] = field(default_factory=dict)
    samples_removed: int = 0
    samples_added: int = 0
    original_ratio: float = 0.0
    balanced_ratio: float = 0.0
    random_state: int = 42


def balance_target_distribution(
    X: np.ndarray,
    y: np.ndarray,
    strategy: Literal['undersample', 'oversample', 'hybrid'] = 'undersample',
    random_state: int = 42,
    target_positive_pct: float = 0.50,
    noise_scale: float = 0.01,
    validate: bool = True,
) -> Tuple[np.ndarray, np.ndarray, BalancingReport]:
    """
    Balance target distribution by resampling positive and negative returns.

    This function addresses class imbalance in regression targets by adjusting
    the number of samples in each class (positive vs negative returns).

    Args:
        X: Feature array of shape (n_samples, seq_length, n_features) or
           (n_samples, n_features). Sequences are preserved.
        y: Target array of shape (n_samples,) containing return values.
        strategy: Balancing strategy to use:
            - 'undersample': Remove samples from majority class to match minority.
            - 'oversample': Add augmented samples to minority class to match majority.
            - 'hybrid': Undersample majority to 60%, oversample minority to 40%.
        random_state: Random seed for reproducibility.
        target_positive_pct: Target percentage of positive samples (for hybrid strategy).
            Default 0.50 for 50/50 split.
        noise_scale: Scale of noise to add when oversampling. Default 0.01.
        validate: Whether to validate the balanced dataset. Default True.

    Returns:
        Tuple containing:
            - X_balanced: Balanced feature array with same shape structure.
            - y_balanced: Balanced target array.
            - report: BalancingReport with balancing metadata.

    Raises:
        ValueError: If strategy is invalid or balancing fails validation.

    Example:
        >>> X_seq, y_aligned = create_sequences(X_scaled, y_scaled, seq_len=90)
        >>> X_balanced, y_balanced, report = balance_target_distribution(
        ...     X_seq, y_aligned, strategy='hybrid'
        ... )
        >>> print(f"Balanced: {report.balanced_counts}")
    """
    np.random.seed(random_state)

    # Validate inputs
    if len(X) != len(y):
        raise ValueError(f"X and y must have same length: {len(X)} != {len(y)}")

    if len(y) == 0:
        raise ValueError("Empty arrays provided")

    y = np.asarray(y).flatten()
    X = np.asarray(X)

    # Separate positive and negative indices
    positive_mask = y > 0
    negative_mask = y < 0
    zero_mask = y == 0

    positive_indices = np.where(positive_mask)[0]
    negative_indices = np.where(negative_mask)[0]
    zero_indices = np.where(zero_mask)[0]

    n_positive = len(positive_indices)
    n_negative = len(negative_indices)
    n_zero = len(zero_indices)
    n_total = len(y)

    # Calculate original ratio
    original_ratio = n_positive / n_negative if n_negative > 0 else float('inf')

    logger.info("")
    logger.info("=" * 60)
    logger.info("TARGET BALANCING")
    logger.info("=" * 60)
    logger.info(f"   Strategy: {strategy}")
    logger.info(f"   Original distribution:")
    logger.info(f"      Positive: {n_positive:,} ({n_positive/n_total*100:.1f}%)")
    logger.info(f"      Negative: {n_negative:,} ({n_negative/n_total*100:.1f}%)")
    if n_zero > 0:
        logger.info(f"      Zero:     {n_zero:,} ({n_zero/n_total*100:.1f}%)")
    logger.info(f"      Ratio (pos/neg): {original_ratio:.2f}")

    # Apply balancing strategy
    if strategy == 'undersample':
        X_balanced, y_balanced, samples_removed, samples_added = _undersample(
            X, y, positive_indices, negative_indices, zero_indices
        )
    elif strategy == 'oversample':
        X_balanced, y_balanced, samples_removed, samples_added = _oversample(
            X, y, positive_indices, negative_indices, zero_indices, noise_scale
        )
    elif strategy == 'hybrid':
        X_balanced, y_balanced, samples_removed, samples_added = _hybrid_resample(
            X, y, positive_indices, negative_indices, zero_indices,
            target_positive_pct, noise_scale
        )
    else:
        raise ValueError(
            f"Invalid strategy: {strategy}. "
            f"Must be one of: 'undersample', 'oversample', 'hybrid'"
        )

    # Calculate balanced distribution
    y_balanced = np.asarray(y_balanced).flatten()
    balanced_positive = np.sum(y_balanced > 0)
    balanced_negative = np.sum(y_balanced < 0)
    balanced_zero = np.sum(y_balanced == 0)
    balanced_total = len(y_balanced)
    balanced_ratio = balanced_positive / balanced_negative if balanced_negative > 0 else float('inf')

    logger.info(f"   Balanced distribution:")
    if balanced_total > 0:
        logger.info(f"      Positive: {balanced_positive:,} ({balanced_positive/balanced_total*100:.1f}%)")
        logger.info(f"      Negative: {balanced_negative:,} ({balanced_negative/balanced_total*100:.1f}%)")
        if balanced_zero > 0:
            logger.info(f"      Zero:     {balanced_zero:,} ({balanced_zero/balanced_total*100:.1f}%)")
    else:
        logger.info(f"      Positive: {balanced_positive:,} (N/A%)")
        logger.info(f"      Negative: {balanced_negative:,} (N/A%)")
    logger.info(f"      Ratio (pos/neg): {balanced_ratio:.2f}" if np.isfinite(balanced_ratio) else f"      Ratio (pos/neg): inf")
    logger.info(f"   Samples removed: {samples_removed:,}")
    logger.info(f"   Samples added: {samples_added:,}")

    # Validate balanced dataset
    if validate:
        _validate_balanced_dataset(
            y_balanced, balanced_positive, balanced_negative, balanced_total
        )
        logger.info("   ✓ Validation passed (40-60% split)")

    logger.info("=" * 60)

    # Build report
    report = BalancingReport(
        strategy=strategy,
        original_counts={
            'positive': n_positive,
            'negative': n_negative,
            'zero': n_zero,
            'total': n_total,
        },
        balanced_counts={
            'positive': int(balanced_positive),
            'negative': int(balanced_negative),
            'zero': int(balanced_zero),
            'total': balanced_total,
        },
        samples_removed=samples_removed,
        samples_added=samples_added,
        original_ratio=original_ratio,
        balanced_ratio=balanced_ratio,
        random_state=random_state,
    )

    return X_balanced, y_balanced, report


def _undersample(
    X: np.ndarray,
    y: np.ndarray,
    positive_indices: np.ndarray,
    negative_indices: np.ndarray,
    zero_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Undersample majority class to match minority class.

    Args:
        X: Feature array.
        y: Target array.
        positive_indices: Indices of positive samples.
        negative_indices: Indices of negative samples.
        zero_indices: Indices of zero samples.

    Returns:
        Tuple of (X_balanced, y_balanced, samples_removed, samples_added).
    """
    n_positive = len(positive_indices)
    n_negative = len(negative_indices)

    if n_positive > n_negative:
        # Undersample positive class
        target_size = n_negative
        sampled_positive = np.random.choice(
            positive_indices, size=target_size, replace=False
        )
        selected_indices = np.concatenate([sampled_positive, negative_indices, zero_indices])
        samples_removed = n_positive - target_size
    else:
        # Undersample negative class
        target_size = n_positive
        sampled_negative = np.random.choice(
            negative_indices, size=target_size, replace=False
        )
        selected_indices = np.concatenate([positive_indices, sampled_negative, zero_indices])
        samples_removed = n_negative - target_size

    # Sort indices to preserve temporal order (important for sequences)
    selected_indices = np.sort(selected_indices)

    X_balanced = X[selected_indices]
    y_balanced = y[selected_indices]

    return X_balanced, y_balanced, samples_removed, 0


def _oversample(
    X: np.ndarray,
    y: np.ndarray,
    positive_indices: np.ndarray,
    negative_indices: np.ndarray,
    zero_indices: np.ndarray,
    noise_scale: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Oversample minority class using noise-based augmentation.

    This is a SMOTE-like approach that creates new samples by adding
    small noise to existing minority class samples.

    Args:
        X: Feature array.
        y: Target array.
        positive_indices: Indices of positive samples.
        negative_indices: Indices of negative samples.
        zero_indices: Indices of zero samples.
        noise_scale: Scale of noise to add to augmented samples.

    Returns:
        Tuple of (X_balanced, y_balanced, samples_removed, samples_added).
    """
    n_positive = len(positive_indices)
    n_negative = len(negative_indices)

    if n_positive < n_negative:
        # Oversample positive class
        minority_indices = positive_indices
        majority_indices = negative_indices
        n_minority = n_positive
        n_majority = n_negative
    else:
        # Oversample negative class
        minority_indices = negative_indices
        majority_indices = positive_indices
        n_minority = n_negative
        n_majority = n_positive

    # Calculate how many samples to add
    samples_to_add = n_majority - n_minority

    # Sample with replacement from minority class
    augment_indices = np.random.choice(
        minority_indices, size=samples_to_add, replace=True
    )

    # Create augmented samples with noise
    X_augment = X[augment_indices].copy()
    y_augment = y[augment_indices].copy()

    # Add noise to features (preserving sequence structure)
    feature_std = np.std(X, axis=0)
    noise = np.random.normal(0, noise_scale, X_augment.shape) * feature_std
    X_augment = X_augment + noise

    # Add small noise to targets (to prevent exact duplicates)
    target_noise = np.random.normal(0, noise_scale * 0.1, y_augment.shape)
    y_augment = y_augment + target_noise

    # Combine original and augmented data
    X_balanced = np.concatenate([X, X_augment], axis=0)
    y_balanced = np.concatenate([y, y_augment], axis=0)

    # Shuffle to mix augmented samples
    shuffle_indices = np.random.permutation(len(y_balanced))
    X_balanced = X_balanced[shuffle_indices]
    y_balanced = y_balanced[shuffle_indices]

    return X_balanced, y_balanced, 0, samples_to_add


def _hybrid_resample(
    X: np.ndarray,
    y: np.ndarray,
    positive_indices: np.ndarray,
    negative_indices: np.ndarray,
    zero_indices: np.ndarray,
    target_positive_pct: float = 0.50,
    noise_scale: float = 0.01,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Hybrid resampling: undersample majority to 60%, oversample minority to 40%.

    This approach provides a middle ground between under and oversampling,
    reducing majority class while augmenting minority class.

    Args:
        X: Feature array.
        y: Target array.
        positive_indices: Indices of positive samples.
        negative_indices: Indices of negative samples.
        zero_indices: Indices of zero samples.
        target_positive_pct: Target percentage for positive samples (default 0.50).
        noise_scale: Scale of noise for augmentation.

    Returns:
        Tuple of (X_balanced, y_balanced, samples_removed, samples_added).
    """
    n_positive = len(positive_indices)
    n_negative = len(negative_indices)
    n_zero = len(zero_indices)
    n_total = n_positive + n_negative + n_zero

    # Determine majority and minority
    if n_positive > n_negative:
        majority_indices = positive_indices
        minority_indices = negative_indices
        n_majority = n_positive
        n_minority = n_negative
        majority_is_positive = True
    else:
        majority_indices = negative_indices
        minority_indices = positive_indices
        n_majority = n_negative
        n_minority = n_positive
        majority_is_positive = False

    # Target: 50/50 split (excluding zeros)
    # Calculate target counts
    target_total = int(n_total * 0.9)  # Allow some reduction
    target_per_class = target_total // 2

    # Ensure we don't exceed what we have
    target_majority = min(target_per_class, n_majority)
    target_minority = target_per_class

    # Calculate samples to remove/add
    samples_removed = max(0, n_majority - target_majority)
    samples_to_add = max(0, target_minority - n_minority)

    # Undersample majority class
    if n_majority > target_majority:
        sampled_majority = np.random.choice(
            majority_indices, size=target_majority, replace=False
        )
    else:
        sampled_majority = majority_indices

    # Keep all minority samples
    kept_minority = minority_indices

    # Combine undersampled majority + minority + zeros
    selected_indices = np.concatenate([sampled_majority, kept_minority, zero_indices])
    selected_indices = np.sort(selected_indices)

    X_intermediate = X[selected_indices]
    y_intermediate = y[selected_indices]

    # Now oversample minority if needed
    if samples_to_add > 0:
        # Find minority indices in the intermediate dataset
        y_intermediate_flat = y_intermediate.flatten()
        if majority_is_positive:
            minority_mask = y_intermediate_flat < 0
        else:
            minority_mask = y_intermediate_flat > 0

        minority_indices_intermediate = np.where(minority_mask)[0]

        if len(minority_indices_intermediate) > 0:
            # Sample with replacement from minority
            augment_indices = np.random.choice(
                minority_indices_intermediate, size=samples_to_add, replace=True
            )

            # Create augmented samples
            X_augment = X_intermediate[augment_indices].copy()
            y_augment = y_intermediate[augment_indices].copy()

            # Add noise
            feature_std = np.std(X_intermediate, axis=0)
            noise = np.random.normal(0, noise_scale, X_augment.shape) * feature_std
            X_augment = X_augment + noise

            target_noise = np.random.normal(0, noise_scale * 0.1, y_augment.shape)
            y_augment = y_augment + target_noise

            # Combine
            X_balanced = np.concatenate([X_intermediate, X_augment], axis=0)
            y_balanced = np.concatenate([y_intermediate, y_augment], axis=0)

            # Shuffle
            shuffle_indices = np.random.permutation(len(y_balanced))
            X_balanced = X_balanced[shuffle_indices]
            y_balanced = y_balanced[shuffle_indices]
        else:
            X_balanced = X_intermediate
            y_balanced = y_intermediate
    else:
        X_balanced = X_intermediate
        y_balanced = y_intermediate

    return X_balanced, y_balanced, samples_removed, samples_to_add


def _validate_balanced_dataset(
    y_balanced: np.ndarray,
    balanced_positive: int,
    balanced_negative: int,
    balanced_total: int,
    min_pct: float = 0.40,
    max_pct: float = 0.60,
) -> None:
    """
    Validate that balanced dataset has 40-60% split.

    Args:
        y_balanced: Balanced target array.
        balanced_positive: Count of positive samples.
        balanced_negative: Count of negative samples.
        balanced_total: Total sample count.
        min_pct: Minimum acceptable percentage for each class.
        max_pct: Maximum acceptable percentage for each class.

    Raises:
        ValueError: If validation fails.
    """
    if balanced_total == 0:
        raise ValueError("Balanced dataset is empty")

    # Calculate percentages (excluding zeros)
    non_zero_total = balanced_positive + balanced_negative
    if non_zero_total == 0:
        raise ValueError("No non-zero samples in balanced dataset")

    positive_pct = balanced_positive / non_zero_total
    negative_pct = balanced_negative / non_zero_total

    issues = []

    if positive_pct < min_pct:
        issues.append(
            f"Positive percentage {positive_pct:.1%} below minimum {min_pct:.0%}"
        )
    if positive_pct > max_pct:
        issues.append(
            f"Positive percentage {positive_pct:.1%} above maximum {max_pct:.0%}"
        )
    if negative_pct < min_pct:
        issues.append(
            f"Negative percentage {negative_pct:.1%} below minimum {min_pct:.0%}"
        )
    if negative_pct > max_pct:
        issues.append(
            f"Negative percentage {negative_pct:.1%} above maximum {max_pct:.0%}"
        )

    if issues:
        raise ValueError(
            f"Balanced dataset validation failed: {'; '.join(issues)}"
        )


def check_sequence_leakage(
    X_train: np.ndarray,
    X_val: np.ndarray,
    sequence_length: int,
) -> bool:
    """
    Check for data leakage between train and validation sequences.

    This verifies that no sequence in the validation set overlaps with
    sequences in the training set (which would cause leakage).

    Args:
        X_train: Training feature array.
        X_val: Validation feature array.
        sequence_length: Length of each sequence.

    Returns:
        True if no leakage detected, False otherwise.

    Note:
        This is an expensive O(n*m) check and should only be used for debugging.
    """
    logger.info("Checking for sequence leakage...")

    # Sample a subset for efficiency
    n_check = min(100, len(X_val))
    sample_val_indices = np.random.choice(len(X_val), size=n_check, replace=False)

    for val_idx in sample_val_indices:
        val_seq = X_val[val_idx]

        # Check if this sequence exists in training data
        for train_idx in range(len(X_train)):
            if np.allclose(val_seq, X_train[train_idx], rtol=1e-5):
                logger.warning(
                    f"Leakage detected: val[{val_idx}] matches train[{train_idx}]"
                )
                return False

    logger.info(f"   ✓ No leakage detected (checked {n_check} validation sequences)")
    return True


def get_balancing_metadata(report: BalancingReport) -> Dict[str, Any]:
    """
    Convert BalancingReport to a dictionary for saving with model artifacts.

    Args:
        report: BalancingReport from balance_target_distribution.

    Returns:
        Dictionary suitable for JSON serialization.
    """
    return {
        'strategy': report.strategy,
        'original_counts': report.original_counts,
        'balanced_counts': report.balanced_counts,
        'samples_removed': report.samples_removed,
        'samples_added': report.samples_added,
        'original_ratio': float(report.original_ratio) if np.isfinite(report.original_ratio) else None,
        'balanced_ratio': float(report.balanced_ratio) if np.isfinite(report.balanced_ratio) else None,
        'random_state': report.random_state,
    }
