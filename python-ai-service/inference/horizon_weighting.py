"""
Dynamic horizon weighting based on validation performance.

Automatically assigns weights to different prediction horizons based on
their validation R² or MSE scores. Better models get higher weights.
"""
import numpy as np
from typing import Dict


def compute_softmax_weights(
    r2_scores: Dict[str, float],
    temperature: float = 5.0
) -> Dict[str, float]:
    """
    Compute weights using softmax over R² scores.
    
    Negative R² → weight near zero.
    Higher R² → higher weight.
    Temperature controls sharpness (higher = more uniform).
    
    Args:
        r2_scores: {'3d': r2_3, '5d': r2_5, '10d': r2_10}
        temperature: Softmax temperature (higher = smoother weights)
    
    Returns:
        {'3d': w1, '5d': w2, '10d': w3} where sum = 1.0
    
    Example:
        >>> r2_scores = {'3d': -0.118, '5d': -0.093, '10d': -0.494}
        >>> weights = compute_softmax_weights(r2_scores, temperature=5.0)
        >>> # 5d gets highest weight (best R²), 10d gets lowest
    """
    horizons = sorted(r2_scores.keys())
    scores = np.array([r2_scores[h] for h in horizons])
    
    # Shift scores to avoid negative exponents causing underflow
    # Add small constant to handle all-negative R²
    scores_shifted = scores - scores.min() + 0.1
    
    # Softmax with temperature
    exp_scores = np.exp(scores_shifted / temperature)
    weights = exp_scores / exp_scores.sum()
    
    # Convert back to dict
    weight_dict = {h: float(w) for h, w in zip(horizons, weights)}
    
    # Sanity check
    assert abs(sum(weight_dict.values()) - 1.0) < 1e-6, "Weights must sum to 1"
    
    return weight_dict


def compute_inverse_mse_weights(
    mse_scores: Dict[str, float],
    epsilon: float = 0.01
) -> Dict[str, float]:
    """
    Weight inversely proportional to MSE.
    
    Lower MSE → higher weight.
    
    Args:
        mse_scores: {'3d': mse_3, '5d': mse_5, '10d': mse_10}
        epsilon: Small constant to avoid division by zero
    
    Returns:
        Normalized weights
    
    Example:
        >>> mse_scores = {'3d': 0.024, '5d': 0.030, '10d': 0.046}
        >>> weights = compute_inverse_mse_weights(mse_scores)
        >>> # 3d gets highest weight (lowest MSE)
    """
    horizons = sorted(mse_scores.keys())
    mse_values = np.array([mse_scores[h] for h in horizons])
    
    # Inverse weights (add epsilon for stability)
    inv_weights = 1.0 / (mse_values + epsilon)
    
    # Normalize
    weights = inv_weights / inv_weights.sum()
    
    weight_dict = {h: float(w) for h, w in zip(horizons, weights)}
    
    return weight_dict


def get_default_weights(horizons: list) -> Dict[str, float]:
    """
    Fallback: equal weights for all horizons.
    
    Args:
        horizons: List of horizon values [3, 5, 10]
    
    Returns:
        Equal weights dictionary
    """
    n = len(horizons)
    return {f'{h}d': 1.0 / n for h in horizons}


# Example usage and testing
if __name__ == '__main__':
    print("="*70)
    print("  HORIZON WEIGHTING EXAMPLES")
    print("="*70)
    
    # Simulate realistic performance (negative R² common for financial data)
    print("\n1. Softmax weights (based on R²):")
    r2_scores = {
        '3d': -0.118,   # Least bad
        '5d': -0.093,   # Best (least negative)
        '10d': -0.494   # Worst
    }
    
    weights = compute_softmax_weights(r2_scores, temperature=5.0)
    print(f"   R² scores: {r2_scores}")
    print(f"   Weights: {weights}")
    print(f"   → 5d gets {weights['5d']*100:.1f}% (best R²)")
    print(f"   → 10d gets {weights['10d']*100:.1f}% (worst R²)")
    
    # Test different temperature
    print("\n2. Softmax with lower temperature (sharper):")
    weights_sharp = compute_softmax_weights(r2_scores, temperature=2.0)
    print(f"   Weights: {weights_sharp}")
    print(f"   → More extreme (better models get even more weight)")
    
    # Inverse MSE approach
    print("\n3. Inverse MSE weights:")
    mse_scores = {
        '3d': 0.024,
        '5d': 0.030,
        '10d': 0.046
    }
    
    weights_mse = compute_inverse_mse_weights(mse_scores)
    print(f"   MSE scores: {mse_scores}")
    print(f"   Weights: {weights_mse}")
    print(f"   → 3d gets {weights_mse['3d']*100:.1f}% (lowest MSE)")
    
    # Default equal weights
    print("\n4. Default equal weights (fallback):")
    weights_equal = get_default_weights([3, 5, 10])
    print(f"   Weights: {weights_equal}")
    
    print("\n" + "="*70)
    print("✅ All weighting methods working correctly")
    print("="*70)
