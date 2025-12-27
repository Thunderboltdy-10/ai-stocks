"""
Time-Series Safe Data Augmentation

This module provides augmentation techniques that preserve temporal properties
and avoid creating unrealistic synthetic samples by mixing unrelated time windows.

Key principles:
- Never mix samples from different chronological windows
- Apply realistic transformations only (jitter, scaling, shifts)
- Target moderate oversampling (40-60% of majority, not 100%)
"""

import numpy as np
from typing import Tuple


def jitter(X: np.ndarray, std: float = 0.01) -> np.ndarray:
    """
    Add small Gaussian noise to time series
    
    Args:
        X: Input sequences of shape (n_samples, timesteps, features)
        std: Standard deviation of noise (default 0.01 = 1% of normalized range)
    
    Returns:
        Augmented sequences with added noise
    """
    noise = np.random.normal(0, std, X.shape)
    return X + noise


def scaling(X: np.ndarray, scale_range: Tuple[float, float] = (0.98, 1.02)) -> np.ndarray:
    """
    Scale time series by random factor
    
    Args:
        X: Input sequences of shape (n_samples, timesteps, features)
        scale_range: Min and max scaling factors (default ±2%)
    
    Returns:
        Scaled sequences
    """
    scale_factors = np.random.uniform(scale_range[0], scale_range[1], size=(X.shape[0], 1, 1))
    return X * scale_factors


def window_shift(X: np.ndarray, shift_max: int = 1) -> np.ndarray:
    """
    Shift sequence by small number of timesteps (within same window)
    
    Args:
        X: Input sequences of shape (n_samples, timesteps, features)
        shift_max: Maximum shift in timesteps (default 1)
    
    Returns:
        Shifted sequences (padded with edge values)
    """
    X_shifted = np.copy(X)
    
    for i in range(X.shape[0]):
        shift = np.random.randint(-shift_max, shift_max + 1)
        
        if shift > 0:
            # Shift forward, pad at start
            X_shifted[i, shift:, :] = X[i, :-shift, :]
            X_shifted[i, :shift, :] = X[i, 0, :]  # Repeat first value
        elif shift < 0:
            # Shift backward, pad at end
            X_shifted[i, :shift, :] = X[i, -shift:, :]
            X_shifted[i, shift:, :] = X[i, -1, :]  # Repeat last value
    
    return X_shifted


def time_drop(X: np.ndarray, drop_max: int = 2) -> np.ndarray:
    """
    Drop random timesteps and forward-fill
    
    Args:
        X: Input sequences of shape (n_samples, timesteps, features)
        drop_max: Maximum timesteps to drop (default 2)
    
    Returns:
        Sequences with dropped timesteps
    """
    X_dropped = np.copy(X)
    
    for i in range(X.shape[0]):
        n_drop = np.random.randint(1, drop_max + 1)
        # Drop random timesteps (not at boundaries)
        drop_indices = np.random.choice(
            range(1, X.shape[1] - 1), 
            size=min(n_drop, X.shape[1] - 2), 
            replace=False
        )
        
        for idx in sorted(drop_indices):
            # Forward fill
            X_dropped[i, idx, :] = X_dropped[i, idx - 1, :]
    
    return X_dropped


def augment_minority(
    X: np.ndarray, 
    y: np.ndarray, 
    target_ratio: float = 0.5,
    techniques: list = ['jitter', 'scaling', 'window_shift'],
    verbose: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment minority classes to target ratio of majority class
    
    Uses only time-series safe transformations, no SMOTE-style interpolation.
    
    Args:
        X: Input sequences of shape (n_samples, timesteps, features)
        y: Labels of shape (n_samples,)
        target_ratio: Target size as ratio of majority class (default 0.5 = 50%)
        techniques: List of augmentation techniques to apply
        verbose: Print augmentation statistics
    
    Returns:
        Tuple of (X_augmented, y_augmented) with minority classes augmented
    """
    # Get class distribution
    unique_classes, counts = np.unique(y, return_counts=True)
    class_counts = dict(zip(unique_classes, counts))
    majority_count = max(counts)
    target_size = int(majority_count * target_ratio)
    
    if verbose:
        print(f"\n📊 Original class distribution:")
        for cls, count in class_counts.items():
            print(f"   Class {cls}: {count} samples ({count/len(y)*100:.1f}%)")
        print(f"\n🎯 Target minority size: {target_size} ({target_ratio*100:.0f}% of majority)")
    
    X_augmented_list = [X]
    y_augmented_list = [y]
    
    for cls in unique_classes:
        count = class_counts[cls]
        
        # Only augment if below target
        if count >= target_size:
            if verbose and count == majority_count:
                print(f"   Class {cls}: Majority class, no augmentation")
            else:
                print(f"   Class {cls}: Already above target, no augmentation")
            continue
        
        # Calculate needed samples
        n_needed = target_size - count
        
        # Get indices for this class
        class_indices = np.where(y == cls)[0]
        
        if len(class_indices) < 2:
            if verbose:
                print(f"   Class {cls}: Too few samples ({len(class_indices)}), skipping")
            continue
        
        # Generate augmented samples
        n_copies = (n_needed // len(techniques)) + 1
        X_cls = X[class_indices]
        
        augmented_samples = []
        
        for technique in techniques:
            # Sample with replacement if needed
            n_samples_this_technique = min(n_copies, n_needed - len(augmented_samples))
            if n_samples_this_technique <= 0:
                break
                
            indices_to_augment = np.random.choice(
                len(X_cls), 
                size=n_samples_this_technique, 
                replace=True
            )
            X_to_augment = X_cls[indices_to_augment]
            
            # Apply technique
            if technique == 'jitter':
                X_aug = jitter(X_to_augment, std=0.01)
            elif technique == 'scaling':
                X_aug = scaling(X_to_augment, scale_range=(0.98, 1.02))
            elif technique == 'window_shift':
                X_aug = window_shift(X_to_augment, shift_max=1)
            elif technique == 'time_drop':
                X_aug = time_drop(X_to_augment, drop_max=2)
            else:
                continue
            
            augmented_samples.append(X_aug)
        
        # Concatenate augmented samples
        if augmented_samples:
            X_cls_augmented = np.vstack(augmented_samples)[:n_needed]
            y_cls_augmented = np.full(len(X_cls_augmented), cls)
            
            X_augmented_list.append(X_cls_augmented)
            y_augmented_list.append(y_cls_augmented)
            
            if verbose:
                print(f"   Class {cls}: Generated {len(X_cls_augmented)} augmented samples")
    
    # Combine original and augmented
    X_final = np.vstack(X_augmented_list)
    y_final = np.concatenate(y_augmented_list)
    
    # Shuffle
    shuffle_idx = np.random.permutation(len(X_final))
    X_final = X_final[shuffle_idx]
    y_final = y_final[shuffle_idx]
    
    if verbose:
        print(f"\n✅ Final distribution:")
        unique_final, counts_final = np.unique(y_final, return_counts=True)
        for cls, count in zip(unique_final, counts_final):
            print(f"   Class {cls}: {count} samples ({count/len(y_final)*100:.1f}%)")
    
    return X_final, y_final


def validate_augmentation_quality(
    X_real: np.ndarray,
    X_augmented: np.ndarray,
    feature_names: list = None,
    max_std_diff: float = 0.1,
    verbose: bool = True
) -> bool:
    """
    Validate that augmented samples are similar to real samples
    
    Args:
        X_real: Real samples of shape (n_samples, timesteps, features)
        X_augmented: Augmented samples of same shape
        feature_names: Optional list of feature names
        max_std_diff: Maximum allowed std deviation difference
        verbose: Print validation results
    
    Returns:
        True if augmentation is valid, False otherwise
    """
    # Flatten to (n_samples, timesteps * features)
    X_real_flat = X_real.reshape(X_real.shape[0], -1)
    X_aug_flat = X_augmented.reshape(X_augmented.shape[0], -1)
    
    # Compute statistics
    real_mean = np.mean(X_real_flat, axis=0)
    real_std = np.std(X_real_flat, axis=0)
    aug_mean = np.mean(X_aug_flat, axis=0)
    aug_std = np.std(X_aug_flat, axis=0)
    
    # Check differences
    mean_diff = np.abs(real_mean - aug_mean)
    std_diff = np.abs(real_std - aug_std)
    
    mean_diff_pct = np.mean(mean_diff / (np.abs(real_mean) + 1e-7)) * 100
    std_diff_pct = np.mean(std_diff / (real_std + 1e-7)) * 100
    
    is_valid = std_diff_pct < (max_std_diff * 100)
    
    if verbose:
        print(f"\n🔍 Augmentation quality check:")
        print(f"   Mean difference: {mean_diff_pct:.2f}%")
        print(f"   Std difference: {std_diff_pct:.2f}%")
        print(f"   Status: {'✅ Valid' if is_valid else '⚠️  Invalid (std diff too large)'}")
    
    return is_valid
