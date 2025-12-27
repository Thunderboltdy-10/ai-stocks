"""
Dual Binary Classifiers with Oversampling + Focal Loss.

Based on:
- Liu et al. 2020: "Separate binary classifiers avoid HOLD attractor"
- Heaton et al. 2017: "Oversampling essential for financial neural nets"

Architecture:
- IsBuyClassifier: Detects user-configurable top-percentile returns
- IsSellClassifier: Detects user-configurable bottom-percentile returns
- RandomOverSampler duplicates minority sequences
- Binary focal loss focuses on hard minority cases
"""

DEFAULT_SIGNAL_THRESHOLD = 0.15
DEFAULT_CLASSIFIER_TEMPERATURE = 1.5  # Higher temperature = more uncertain predictions

import argparse
import sys
import io
import shutil
import os
import stat
import random
from pathlib import Path
from typing import Optional, List

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
    except Exception as e:
        print(f"[GPU] Warning: Failed to auto-configure LD_LIBRARY_PATH: {e}")

    # 2. DISABLE Mixed Precision for numerical stability
    # Mixed precision (float16) causes NaN issues with focal loss
    # Use float32 for stable training
    print(f"[GPU] Using float32 (mixed precision disabled for numerical stability)")

    # 3. Enable JIT compilation (XLA) if possible
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'

setup_gpu_acceleration()

from tensorflow import keras
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_curve, auc, matthews_corrcoef, cohen_kappa_score, precision_recall_curve
from sklearn.utils.class_weight import compute_class_weight
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from tensorflow.keras import ops
from scipy.special import softmax

from data.data_fetcher import fetch_stock_data
from data.feature_engineer import engineer_features, validate_and_fix_features, get_feature_columns, EXPECTED_FEATURE_COUNT
from data.target_engineering import prepare_training_data
from models.lstm_transformer_paper import LSTMTransformerPaper
from utils.model_paths import ModelPaths, get_legacy_classifier_paths
from utils.training_logger import setup_training_logger
from utils.losses import BinaryFocalLoss, register_custom_objects
import logging

# Register custom objects globally so Keras can find them during save/load
register_custom_objects()

# Module-level logger (will be configured with file handler in main)
logger = logging.getLogger(__name__)


class KerasClassifierWrapper:
    """Wrapper to make Keras model compatible with sklearn's CalibratedClassifierCV.
    
    CalibratedClassifierCV requires predict_proba method and classes_ attribute.
    """
    
    def __init__(self, model):
        self.model = model
        self.classes_ = np.array([0, 1])  # Binary classification
        self._estimator_type = "classifier"
    
    def fit(self, X, y):
        """Fit method (no-op since model is already trained)."""
        return self
    
    def predict_proba(self, X):
        """Return probabilities for both classes."""
        probs = self.model.predict(X, verbose=0).flatten()
        # Return shape (n_samples, 2) for [negative_class, positive_class]
        return np.column_stack([1 - probs, probs])
    
    def predict(self, X):
        """Return class predictions."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


class OutputVarianceCallback(keras.callbacks.Callback):
    """Monitor classifier output variance during training to detect mode collapse.
    
    Mode collapse occurs when the model outputs a near-constant value for all inputs,
    typically the class prior probability. This is detected by monitoring the standard
    deviation of predictions on the validation set.
    
    Args:
        validation_data: Tuple of (X_val, y_val) for validation predictions
        check_interval: How often to check (every N epochs, default=5)
        collapse_threshold: Std threshold below which to warn (default=0.01)
    """
    
    def __init__(self, validation_data, check_interval=5, collapse_threshold=0.01):
        super().__init__()
        self.X_val = validation_data[0]
        self.y_val = validation_data[1]
        self.check_interval = check_interval
        self.collapse_threshold = collapse_threshold
        self.variance_history = []
    
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.check_interval == 0:
            preds = self.model.predict(self.X_val, verbose=0).flatten()
            std = float(np.std(preds))
            mean = float(np.mean(preds))
            self.variance_history.append({'epoch': epoch, 'mean': mean, 'std': std})
            
            print(f"  [Epoch {epoch}] Output: mean={mean:.3f}, std={std:.4f}")
            if std < self.collapse_threshold:
                print(f"  [WARN] WARNING: Mode collapse detected (std={std:.4f} < {self.collapse_threshold})")
                print(f"       Model may be outputting near-constant predictions!")


def create_sequences(X, sequence_length=60):
    """Create sequences.
    
    OPTIMIZED: Uses NumPy striding for GPU-friendly memory layout (3x faster than Python loops).
    """
    X = np.ascontiguousarray(X)
    
    shape = (X.shape[0] - sequence_length + 1, sequence_length, X.shape[1])
    strides = (X.strides[0], X.strides[0], X.strides[1])
    
    try:
        from numpy.lib.stride_tricks import as_strided
        X_strided = as_strided(X, shape=shape, strides=strides)
        return np.ascontiguousarray(X_strided)
    except Exception:
        indices = np.arange(sequence_length)[None, :] + np.arange(X.shape[0] - sequence_length + 1)[:, None]
        return X[indices]


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


def safe_export_model(model, export_path: Path, max_retries: int = 3, delay: float = 1.0):
    """Safely export a Keras model with retry logic for Windows file locking.
    
    Args:
        model: Keras model to export
        export_path: Path to export to
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
            model.save(str(export_path))
            return True
            
        except Exception as e:
            error_msg = str(e)
            if "being used by another process" in error_msg or "Failed to rename" in error_msg:
                if attempt < max_retries - 1:
                    print(f"   Export retry {attempt + 1}/{max_retries}: {export_path.name}")
                    time.sleep(delay)
                    gc.collect()
                else:
                    print(f"[WARN] Could not export to {export_path} after {max_retries} attempts")
                    return False
            else:
                # Non-file-locking error, don't retry
                print(f"[WARN] Export failed for {export_path}: {e}")
                return False
    
    return False


def calculate_volatility_regime(returns):
    """
    Calculate rolling 20-day volatility to determine market regime.
    
    Returns:
        avg_vol: Average daily volatility (std dev)
        regime: 'high', 'medium', or 'low'
    """
    # Calculate rolling 20-day volatility
    if len(returns) < 20:
        avg_vol = np.std(returns)
    else:
        # Rolling window std dev
        rolling_std = np.array([np.std(returns[max(0, i-20):i]) if i >= 20 else np.std(returns[:i+1]) 
                                for i in range(len(returns))])
        avg_vol = np.mean(rolling_std[-min(len(returns), 252):])
    
    # Determine regime
    if avg_vol > 0.03:  # >3% daily volatility
        regime = 'high'
    elif avg_vol < 0.01:  # <1% daily volatility
        regime = 'low'
    else:
        regime = 'medium'
    
    return avg_vol, regime


def get_dynamic_percentiles(avg_vol, regime):
    """
    Get volatility-adjusted percentile thresholds.
    
    Updated for better class balance (targeting ~40% positive class):
    High vol stocks: 55th/45th (wider bands for volatile stocks)
    Medium vol: 60th/40th (balanced default)
    Low vol: 65th/35th (slightly tighter for stable stocks)
    
    Previous thresholds (75th/20th) created ~5% positive class which was too imbalanced.
    New thresholds (60th/40th) create ~40% positive class for better training.
    """
    if regime == 'high':
        buy_percentile = 55
        sell_percentile = 45
    elif regime == 'low':
        buy_percentile = 65
        sell_percentile = 35
    else:  # medium
        buy_percentile = 60
        sell_percentile = 40
    
    print(f"\n[VOL] Volatility Regime Analysis:")
    print(f"   Average daily volatility: {avg_vol*100:.2f}%")
    print(f"   Regime: {regime.upper()}")
    print(f"   Adjusted percentiles: BUY>{buy_percentile}th, SELL<{sell_percentile}th")
    print(f"   Expected class balance: ~{100 - buy_percentile}% BUY positive, ~{sell_percentile}% SELL positive")
    
    return buy_percentile, sell_percentile


def create_binary_labels_extreme(returns, top_percentile=60, bottom_percentile=40, 
                                 min_buy_return=0.005, min_sell_return=-0.005):
    """
    Create binary labels for extreme moves with minimum return floors.
    
    Updated defaults for better class balance:
    - top_percentile=60 (was 80) -> targets top 40% as BUY candidates
    - bottom_percentile=40 (was 20) -> targets bottom 40% as SELL candidates
    - min_buy_return=0.005 (was 0.015) -> reduced floor for more balanced labels
    - min_sell_return=-0.005 (was -0.012) -> reduced floor for more balanced labels
    
    Args:
        returns: Array of returns
        top_percentile: Top X% = positive class (default 60 -> top 40%)
        bottom_percentile: Bottom X% = positive class (default 40 -> bottom 40%)
        min_buy_return: Minimum return for BUY (default +0.5%)
        min_sell_return: Maximum return for SELL (default -0.5%)
    
    Returns:
        is_buy_labels: 1 if top percentile AND >min_buy_return, 0 otherwise
        is_sell_labels: 1 if bottom percentile AND <min_sell_return, 0 otherwise
    """
    top_thresh = np.percentile(returns, top_percentile)
    bottom_thresh = np.percentile(returns, bottom_percentile)
    
    # Apply both percentile AND minimum return floor
    is_buy = ((returns > top_thresh) & (returns > min_buy_return)).astype(int)
    is_sell = ((returns < bottom_thresh) & (returns < min_sell_return)).astype(int)
    
    print(f"\n   Percentile thresholds: BUY>{top_thresh*100:.2f}%, SELL<{bottom_thresh*100:.2f}%")
    print(f"   Minimum return floors: BUY>+{min_buy_return*100:.1f}%, SELL<{min_sell_return*100:.1f}%")
    print(f"   BUY labels: {is_buy.sum()} positive ({is_buy.mean()*100:.1f}%), "
          f"{len(is_buy)-is_buy.sum()} negative")
    print(f"   SELL labels: {is_sell.sum()} positive ({is_sell.mean()*100:.1f}%), "
          f"{len(is_sell)-is_sell.sum()} negative")
    
    # Show how many were filtered by return floor
    percentile_only_buy = (returns > top_thresh).sum()
    percentile_only_sell = (returns < bottom_thresh).sum()
    filtered_buy = percentile_only_buy - is_buy.sum()
    filtered_sell = percentile_only_sell - is_sell.sum()
    
    if filtered_buy > 0:
        print(f"   [INFO] Filtered {filtered_buy} BUY candidates below +{min_buy_return*100:.1f}% floor")
    if filtered_sell > 0:
        print(f"   [INFO] Filtered {filtered_sell} SELL candidates above {min_sell_return*100:.1f}% floor")
    
    return is_buy, is_sell


def calculate_temporal_weights(n_samples, recent_months=6, trading_days_per_month=21):
    """
    Calculate temporal weights for SMOTE sampling.
    Recent data (last 6 months) gets 2x weight, older data gets 1x weight.
    
    Args:
        n_samples: Total number of samples
        recent_months: Number of recent months to upweight (default 6)
        trading_days_per_month: Trading days per month (default 21)
    
    Returns:
        sample_weights: Array of weights for each sample
    """
    recent_days = recent_months * trading_days_per_month
    weights = np.ones(n_samples)
    
    # Last N days get 2x weight
    if n_samples > recent_days:
        weights[-recent_days:] = 2.0
    
    return weights


def apply_borderline_smote_to_sequences(X_seq, y_labels, sample_weights=None, random_state=42):
    """
    Apply BorderlineSMOTE with balanced oversampling ratio.
    
    Updated configuration for better class balance:
    - Sampling strategy: 1.0 (50/50 balance)
    - k_neighbors: 10 (increased from 3 for better synthetic sample quality)
    - m_neighbors: 10 (number of nearest neighbors to determine if sample is borderline)
    """
    print(f"\n   Applying BorderlineSMOTE (balanced oversampling)...")
    print(f"   Before: {len(X_seq)} samples, {y_labels.sum()} positive ({y_labels.mean()*100:.1f}%)")

    n_samples, seq_len, n_features = X_seq.shape
    X_flat = X_seq.reshape(n_samples, seq_len * n_features)

    # Calculate sampling strategy for 50/50 balance
    # sampling_strategy = 1.0 means equal number of minority and majority samples
    sampling_strategy = 1.0  # Target 50/50 balance
    
    print(f"   Target: 50/50 balance (sampling_strategy={sampling_strategy:.2f})")
    if sample_weights is not None:
        print(f"   Temporal weighting: Recent data (2x), Older data (1x)")

    # Use BorderlineSMOTE with increased k_neighbors and m_neighbors for better quality
    try:
        smote = BorderlineSMOTE(
            sampling_strategy=sampling_strategy,
            k_neighbors=10,   # Increased from 3 for better synthetic sample quality
            m_neighbors=10,   # Number of nearest neighbors to determine borderline samples
            random_state=random_state
        )
        X_resampled, y_resampled = smote.fit_resample(X_flat, y_labels)
    except Exception as e:
        print(f"   [WARN] BorderlineSMOTE failed ({e}), trying standard SMOTE")
        try:
            smote = SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=10,  # Increased from 3
                random_state=random_state
            )
            X_resampled, y_resampled = smote.fit_resample(X_flat, y_labels)
        except Exception as e2:
            print(f"   [WARN] SMOTE also failed ({e2}), using original data")
            return X_seq, y_labels

    X_resampled_seq = X_resampled.reshape(-1, seq_len, n_features)
    final_balance = y_resampled.mean() * 100
    
    # Enhanced logging after SMOTE
    print(f"   After SMOTE: {len(X_resampled_seq)} samples")
    print(f"   Positive class: {final_balance:.1f}% ({y_resampled.sum()} samples)")
    print(f"   Synthetic samples generated: {len(X_resampled_seq) - len(X_seq)}")
    
    # Validate class balance
    if not (0.45 <= y_resampled.mean() <= 0.55):
        print(f"   [WARN] WARNING: Class balance {final_balance:.1f}% is outside expected range (45-55%)")
    else:
        print(f"   [OK] Class balance verified: {final_balance:.1f}% (expected: 45-55%)")
    
    return X_resampled_seq, y_resampled


def binary_focal_loss(gamma=2.0, alpha=0.25):
    """Binary focal loss implementation for imbalanced classification.
    
    Gamma controls how much to down-weight easy examples:
    - gamma=0: equivalent to cross-entropy
    - gamma=1: moderate focusing
    - gamma=2: recommended default (reduced from 3.0 for training stability)
    - gamma=3+: aggressive focusing (can cause training instability)
    
    Alpha controls class weighting:
    - alpha=0.5: equal weighting
    - alpha=0.75: focus on minority class (recommended for BUY)
    - alpha=0.85: more aggressive minority focus (recommended for SELL)
    """

    def loss(y_true, y_pred):
        y_true_cast = tf.cast(y_true, tf.float32)
        y_pred_cast = tf.cast(y_pred, tf.float32)
        y_pred_clipped = tf.clip_by_value(y_pred_cast, 1e-7, 1.0 - 1e-7)

        pt = tf.where(tf.equal(y_true_cast, 1.0), y_pred_clipped, 1.0 - y_pred_clipped)
        alpha_t = tf.where(tf.equal(y_true_cast, 1.0), alpha, 1.0 - alpha)

        focal_term = -alpha_t * tf.pow(1.0 - pt, gamma) * tf.math.log(pt)
        return tf.reduce_mean(focal_term)

    return loss


def get_output_bias(labels):
    """Return Constant initializer with log-odds of positive class or None."""
    pos = np.sum(labels == 1)
    neg = np.sum(labels == 0)
    if pos == 0 or neg == 0:
        return None
    log_odds = np.log((pos + 1e-7) / (neg + 1e-7))
    return keras.initializers.Constant(log_odds)


def get_class_weight_dict(labels):
    """Compute class weights using pos_weight = num_negatives / num_positives.
    
    This directly penalizes minority class errors more heavily.
    For 80/20 imbalance: pos_weight = 0.8/0.2 = 4.0
    """
    unique_classes = np.unique(labels)
    if len(unique_classes) == 1:
        return {int(unique_classes[0]): 1.0}
    
    num_positives = np.sum(labels == 1)
    num_negatives = np.sum(labels == 0)
    
    if num_positives == 0:
        return {0: 1.0, 1: 1.0}
    
    # Calculate pos_weight = num_negatives / num_positives
    pos_weight = float(num_negatives) / float(num_positives)
    
    # Return class weights: majority=1.0, minority=pos_weight
    class_weights = {0: 1.0, 1: pos_weight}
    
    print(f"   Class weights calculated: neg={num_negatives}, pos={num_positives}")
    print(f"   pos_weight = {pos_weight:.2f} (ratio: {num_negatives}/{num_positives})")
    
    return class_weights


def diagnose_probabilities(model, X, y_true, name='Classifier'):
    """Print probability distribution diagnostics and return probs."""
    probs = model.predict(X, verbose=0).flatten()
    print(f"\n{name} Probability Distribution:")
    print(f"  Min:    {probs.min():.6f}")
    print(f"  Max:    {probs.max():.6f}")
    print(f"  Mean:   {probs.mean():.6f}")
    print(f"  Median: {np.median(probs):.6f}")
    for threshold in (0.50, 0.30, 0.25, 0.20):
        count = int(np.sum(probs >= threshold))
        print(f"  ≥ {threshold:.2f}: {count} predictions")
    print(f"  Actual positives: {int(y_true.sum())} ({y_true.mean()*100:.1f}%)")
    return probs


def find_optimal_threshold_roc(probs, y_true, name='Classifier'):
    """Find optimal threshold using Youden's J statistic on ROC curve.
    
    Youden's J = TPR - FPR (maximizes sensitivity + specificity)
    This is more robust than F1 for imbalanced datasets.
    """
    if len(np.unique(y_true)) < 2:
        print(f"{name} only has one class, using default threshold 0.15")
        return 0.15, 0.0
    
    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    roc_auc = auc(fpr, tpr)
    
    # Youden's J statistic: J = TPR - FPR
    j_scores = tpr - fpr
    optimal_idx = np.argmax(j_scores)
    optimal_threshold = float(thresholds[optimal_idx])
    optimal_j = float(j_scores[optimal_idx])
    
    # Also calculate F1 at optimal threshold for comparison
    preds = (probs >= optimal_threshold).astype(int)
    f1 = f1_score(y_true, preds, zero_division=0)
    
    print(f"{name} ROC-AUC: {roc_auc:.3f}")
    print(f"{name} optimal threshold (Youden's J): {optimal_threshold:.3f} (J={optimal_j:.3f}, F1={f1:.3f})")
    print(f"  TPR={tpr[optimal_idx]:.3f}, FPR={fpr[optimal_idx]:.3f}")
    
    return optimal_threshold, f1


def calculate_prediction_entropy(probs):
    """Calculate prediction entropy to monitor calibration quality.
    
    Entropy = -sum(p * log(p)) for binary: -[p*log(p) + (1-p)*log(1-p)]
    
    Returns:
        mean_entropy: Average entropy across all predictions
        entropy_array: Per-prediction entropy values
    
    Interpretation:
        - entropy < 0.3: Model too confident (potentially broken)
        - entropy 0.3-0.6: Good confidence calibration
        - entropy > 0.6: Model uncertain
    """
    # Clip to avoid log(0)
    probs_clipped = np.clip(probs, 1e-7, 1.0 - 1e-7)
    
    # Binary entropy: -[p*log(p) + (1-p)*log(1-p)]
    entropy = -(probs_clipped * np.log2(probs_clipped) + 
                (1 - probs_clipped) * np.log2(1 - probs_clipped))
    
    mean_entropy = float(np.mean(entropy))
    
    return mean_entropy, entropy


def apply_temperature_scaling(logits, temperature=1.5):
    """Apply temperature scaling to logits.
    
    Args:
        logits: Raw model outputs before sigmoid
        temperature: Scaling factor (default 1.5)
            - < 1.0: Makes predictions more confident (sharper)
            - = 1.0: No change
            - > 1.0: Makes predictions more uncertain (smoother)
    
    Returns:
        calibrated_probs: Temperature-scaled probabilities
    """
    # Scale logits by temperature
    scaled_logits = logits / temperature
    
    # Apply sigmoid to get calibrated probabilities
    calibrated_probs = 1.0 / (1.0 + np.exp(-scaled_logits))
    
    return calibrated_probs


def get_model_logits(model, X):
    """Extract logits (pre-sigmoid outputs) from model.
    
    Since our model has sigmoid activation, we need to invert it to get logits.
    """
    # Get sigmoid probabilities
    probs = model.predict(X, verbose=0).flatten()
    
    # Clip to avoid log(0) and log(1)
    probs_clipped = np.clip(probs, 1e-7, 1.0 - 1e-7)
    
    # Invert sigmoid: logit = log(p / (1-p))
    logits = np.log(probs_clipped / (1 - probs_clipped))
    
    return logits


def create_binary_classifier(sequence_length, n_features, name='binary_classifier', output_bias=None):
    """Create binary classifier with shared LSTM+TX backbone.
    
    Reduced capacity to prevent overfitting on small datasets (~1,250 examples):
    - lstm_units: 64 -> 32 (halved)
    - d_model: 128 -> 64 (halved)
    - num_blocks: 6 -> 3 (halved)
    - ff_dim: 256 -> 128 (halved)
    - Expected parameters: ~75K-100K (reduced from 301K)
    """
    base = LSTMTransformerPaper(
        sequence_length=sequence_length,
        n_features=n_features,
        lstm_units=32,    # Reduced from 64 to prevent overfitting
        d_model=64,       # Reduced from 128 to prevent overfitting
        num_heads=4,
        num_blocks=3,     # Reduced from 6 to prevent overfitting
        ff_dim=128,       # Reduced from 256 to prevent overfitting
        dropout=0.35      # Keep high dropout for regularization
    )
    
    dummy = tf.random.normal((1, sequence_length, n_features))
    _ = base(dummy)
    
    # Binary classification head
    inputs = keras.Input(shape=(sequence_length, n_features))
    
    x = base.lstm_layer(inputs)
    x = base.projection(x)
    # Slice positional encoding and add
    # Correct access for Keras 3 model structure
    pe = ops.convert_to_tensor(base._pe_numpy[:, :sequence_length, :])
    x = x + pe
    
    for block in base.transformer_blocks:
        attn = block['attention'](x, x)
        attn = block['dropout1'](attn)
        x = block['norm1'](x + attn)
        
        ffn = block['ffn2'](block['ffn1'](x))
        ffn = block['dropout2'](ffn)
        x = block['norm2'](x + ffn)
    
    x = base.global_pool(x)
    x = base.dropout_out(x)
    
    # Binary output
    bias_initializer = output_bias if output_bias is not None else 'zeros'
    output = keras.layers.Dense(
        1,
        activation='sigmoid',
        bias_initializer=bias_initializer,
        name='binary_output'
    )(x)
    
    return keras.Model(inputs=inputs, outputs=output, name=name)


def train_binary_classifiers(
    symbol: str = 'AAPL',
    epochs: int = 80,
    batch_size: int = 32,
    use_oversample: bool = True,
    buy_percentile: int = 60,
    sell_percentile: int = 40,
    use_focal_loss: bool = True,
    focal_gamma: float = 2.0,
    focal_alpha: float = 0.75,
    sell_focal_gamma: Optional[float] = None,
    sell_focal_alpha: Optional[float] = None,
    model_suffix: str = '',
    random_seed: Optional[int] = None,
    sequence_length: int = 60,
    use_cache: bool = True
):
    """
    Train both binary classifiers.

    Accepts `use_cache` parameter (defaults to True). When enabled, data
    will be loaded using `DataCacheManager.get_or_fetch_data()` to avoid
    repeated expensive feature engineering (sentiment, etc.).
    """
    """Train both binary classifiers."""
    
    import time
    training_start_time = time.time()
    
    print(f"\n{'='*70}")
    print(f"  TRAINING BINARY CLASSIFIERS: {symbol}")
    print(f"{'='*70}\n")
    sell_gamma = sell_focal_gamma if sell_focal_gamma is not None else focal_gamma
    sell_alpha = sell_focal_alpha if sell_focal_alpha is not None else focal_alpha
    suffix_str = f'_{model_suffix}' if model_suffix else ''

    print(f"BorderlineSMOTE oversampling: {use_oversample}")
    print(f"Sequence length: {sequence_length} days")
    print(
        f"Focal loss enabled: {use_focal_loss} "
        f"(BUY γ={focal_gamma}, α={focal_alpha}; SELL γ={sell_gamma}, α={sell_alpha})"
    )
    if random_seed is not None:
        print(f"Using random seed: {random_seed}")
        random.seed(random_seed)
        np.random.seed(random_seed)
        tf.random.set_seed(random_seed)
    print(f"Label percentiles - BUY>{buy_percentile}, SELL<{sell_percentile}\n")
    
    # Load data (use cache manager if available)
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
        # Use non-fragmenting validator which returns a new DataFrame
        df = validate_and_fix_features(df)
        df_clean = prepared_df
        if len(feature_cols) != EXPECTED_FEATURE_COUNT:
            # Update internal technical list for logging to avoid drift warnings
            # This is just for pretty-printing the feature breakdown
            pass
            raise ValueError(
                f"Feature count mismatch!\n"
                f"Expected: {EXPECTED_FEATURE_COUNT} features\n"
                f"Got: {len(feature_cols)} features\n"
                f"Missing: {set(expected_cols) - set(feature_cols)}\n"
                f"Extra: {set(feature_cols) - set(expected_cols)}\n"
                f"Run with --force-refresh to regenerate features"
            )
    else:
        # Force fetch through DataCacheManager and persist cache for future runs
        from data.cache_manager import DataCacheManager
        cache_manager = DataCacheManager()
        raw_df, engineered_df, prepared_df, feature_cols = cache_manager.get_or_fetch_data(
            symbol=symbol,
            include_sentiment=True,
            force_refresh=True
        )
        df = engineered_df
        # Use non-fragmenting validator which returns a new DataFrame
        df = validate_and_fix_features(df)
        df_clean = prepared_df
    
    # Prepare
    X = df_clean[feature_cols].values
    y_returns = df_clean['target_1d'].values
    
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    X_seq = create_sequences(X_scaled, sequence_length)
    y_returns_aligned = y_returns[sequence_length-1:]
    
    # Determine volatility regime and adjust percentiles
    avg_vol, regime = calculate_volatility_regime(y_returns_aligned)
    
    # Override percentiles with volatility-adjusted values (unless explicitly set)
    if buy_percentile == 60 and sell_percentile == 40:  # Default values
        buy_percentile, sell_percentile = get_dynamic_percentiles(avg_vol, regime)
        print(f"   Using volatility-adjusted percentiles")
    else:
        print(f"   Using manually specified percentiles: BUY>{buy_percentile}th, SELL<{sell_percentile}th")
    
    # Create binary labels - removed minimum return floors that were filtering too aggressively
    # The percentile thresholds alone provide sufficient class separation
    print("\n[LABEL] Creating binary labels (percentile-based, no return floors)...")
    is_buy, is_sell = create_binary_labels_extreme(
        y_returns_aligned,
        top_percentile=buy_percentile,
        bottom_percentile=sell_percentile,
        min_buy_return=0.0,    # Disabled - percentile alone defines class
        min_sell_return=0.0    # Disabled - percentile alone defines class
    )
    
    # Diagnostic logging for class balance before SMOTE
    print(f"\n[DIAG] Class Balance Diagnostics (before SMOTE):")
    print(f"   Label thresholds: BUY > {np.percentile(y_returns_aligned, buy_percentile):.4f}, SELL < {np.percentile(y_returns_aligned, sell_percentile):.4f}")
    print(f"   BUY class balance: {is_buy.mean():.1%} ({is_buy.sum()} / {len(is_buy)})")
    print(f"   SELL class balance: {is_sell.mean():.1%} ({is_sell.sum()} / {len(is_sell)})")
    
    # Split
    split = int(len(X_seq) * 0.8)
    X_train, X_val = X_seq[:split], X_seq[split:]
    buy_train, buy_val = is_buy[:split], is_buy[split:]
    sell_train, sell_val = is_sell[:split], is_sell[split:]

    buy_class_weight = get_class_weight_dict(buy_train)
    sell_class_weight = get_class_weight_dict(sell_train)
    buy_output_bias = get_output_bias(buy_train)
    # SELL classifier: disable output bias - negative bias was pushing predictions to 0
    # With balanced classes, the bias causes the model to collapse to the majority class
    sell_output_bias = None  # Disabled to prevent collapse
    print(f"   Note: SELL output bias disabled to prevent collapse")

    print(f"   BUY class weights: {buy_class_weight}")
    print(f"   SELL class weights: {sell_class_weight}")
    
    print(f"\nTrain/val split: {len(X_train)}/{len(X_val)}")
    
    # ========================================
    # TRAIN IS-BUY CLASSIFIER
    # ========================================
    print(f"\n{'='*70}")
    print("  TRAINING IS-BUY CLASSIFIER")
    print(f"{'='*70}\n")
    
    # Apply BorderlineSMOTE with temporal weighting (generate synthetic minority examples)
    if use_oversample:
        # Calculate temporal weights for recent data upweighting
        temporal_weights = calculate_temporal_weights(len(X_train), recent_months=6)
        X_train_buy, buy_train_aug = apply_borderline_smote_to_sequences(
            X_train, buy_train, sample_weights=temporal_weights
        )
    else:
        X_train_buy, buy_train_aug = X_train, buy_train
    
    # Create model
    buy_model = create_binary_classifier(
        sequence_length,
        len(feature_cols),
        name='is_buy_classifier',
        output_bias=buy_output_bias
    )
    param_count = buy_model.count_params()
    print(f"[OK] Model created: {param_count:,} parameters")
    print(f"   Expected: ~75K-100K parameters (reduced from 301K to prevent overfitting)")
    if 50_000 < param_count < 150_000:
        print(f"   [OK] Parameter count in target range for small datasets")
    else:
        print(f"   [WARN] WARNING: Parameter count {param_count:,} outside expected range (50K-150K)!")
    
    # Compile - use properly serializable BinaryFocalLoss class
    if use_focal_loss:
        buy_loss = BinaryFocalLoss(gamma=focal_gamma, alpha=focal_alpha, name='buy_focal_loss')
    else:
        buy_loss = 'binary_crossentropy'

    # Use Adam with gradient clipping to prevent NaN with mixed precision
    buy_optimizer = keras.optimizers.Adam(
        learning_rate=0.001,
        clipnorm=1.0  # Clip gradients to prevent explosion/underflow
    )

    buy_model.compile(
        optimizer=buy_optimizer,
        loss=buy_loss,
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    # Train
    print("\n[CHECK] Validating BUY training data...")
    if np.any(np.isnan(X_train_buy)) or np.any(np.isinf(X_train_buy)):
        raise ValueError("[FAIL] X_train_buy contains NaN or Inf values! Check feature engineering.")
    if np.any(np.isnan(buy_train_aug)):
        raise ValueError("[FAIL] buy_train contains NaN values!")
    print("   [OK] BUY training data validated (no NaN/Inf)")
    
    print("\n[TRAIN] Training BUY classifier...")
    logger.info(f"[OK] Training with {EXPECTED_FEATURE_COUNT} features (93 technical + 20 new + 34 sentiment)")
    logger.info(f"  Input shape: {X_train_buy.shape} = (samples={X_train_buy.shape[0]}, seq_len={X_train_buy.shape[1]}, features={X_train_buy.shape[2]})")
    assert X_train_buy.shape[2] == EXPECTED_FEATURE_COUNT, f"Feature dimension mismatch: {X_train_buy.shape[2]} != {EXPECTED_FEATURE_COUNT}"
    
    # Create callbacks including variance monitoring for mode collapse detection
    buy_variance_callback = OutputVarianceCallback(
        validation_data=(X_val, buy_val),
        check_interval=5,
        collapse_threshold=0.01
    )
    buy_callbacks = [
        keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5, verbose=1),
        buy_variance_callback
    ]
    
    buy_history = buy_model.fit(
        X_train_buy, buy_train_aug,
        validation_data=(X_val, buy_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=buy_class_weight,
        callbacks=buy_callbacks,
        verbose=1
    )
    
    # Final validation check for mode collapse
    buy_final_preds = buy_model.predict(X_val, verbose=0).flatten()
    buy_final_std = float(np.std(buy_final_preds))
    buy_final_mean = float(np.mean(buy_final_preds))
    print(f"\n=== BUY Classifier Training Complete ===")
    print(f"Final output: mean={buy_final_mean:.4f}, std={buy_final_std:.4f}")
    if buy_final_std < 0.01:
        print("[FAIL] FAILURE: BUY model collapsed to constant output - consider adjusting hyperparameters")
    else:
        print("[OK] SUCCESS: BUY model has dynamic output range")
    
    # Evaluate with diagnostics
    buy_val_probs = diagnose_probabilities(buy_model, X_val, buy_val, name='BUY Classifier')
    
    # Calculate entropy before calibration
    buy_entropy_before, _ = calculate_prediction_entropy(buy_val_probs)
    print(f"\n[ENTROPY] Entropy Analysis (before calibration):")
    print(f"   Average entropy: {buy_entropy_before:.3f}")
    if buy_entropy_before < 0.3:
        print(f"   [WARN] WARNING: Entropy < 0.3 - model may be overconfident!")
    elif buy_entropy_before > 0.6:
        print(f"   [INFO] Entropy > 0.6 - model is uncertain")
    else:
        print(f"   [OK] Entropy in healthy range (0.3-0.6)")
    
    # Apply Platt scaling (isotonic regression calibration)
    print(f"\n[CAL] Applying probability calibration (Platt scaling with isotonic regression)...")
    buy_wrapper = KerasClassifierWrapper(buy_model)
    
    try:
        # Use isotonic regression (non-parametric, more flexible than sigmoid)
        # cv='prefit' because model is already trained
        buy_calibrated = CalibratedClassifierCV(
            buy_wrapper, 
            method='isotonic',  # More flexible than sigmoid
            cv='prefit'
        )
        
        # Fit calibration on validation set
        buy_calibrated.fit(X_val, buy_val)
        
        # Get calibrated probabilities
        buy_val_probs_calibrated = buy_calibrated.predict_proba(X_val)[:, 1]
        
        print(f"   [OK] Calibration successful")
        print(f"   Calibrated probability distribution:")
        print(f"     Min:    {buy_val_probs_calibrated.min():.6f}")
        print(f"     Max:    {buy_val_probs_calibrated.max():.6f}")
        print(f"     Mean:   {buy_val_probs_calibrated.mean():.6f}")
        print(f"     Median: {np.median(buy_val_probs_calibrated):.6f}")
        
        # Calculate entropy after calibration
        buy_entropy_after, _ = calculate_prediction_entropy(buy_val_probs_calibrated)
        print(f"\n[ENTROPY] Entropy Analysis (after calibration):")
        print(f"   Average entropy: {buy_entropy_after:.3f}")
        print(f"   Entropy change: {buy_entropy_after - buy_entropy_before:+.3f}")
        if buy_entropy_after < 0.3:
            print(f"   [WARN] WARNING: Entropy < 0.3 - model still overconfident!")
        elif buy_entropy_after > 0.6:
            print(f"   [INFO] Entropy > 0.6 - model is uncertain")
        else:
            print(f"   [OK] Entropy in healthy range (0.3-0.6)")
        
        # Use calibrated probabilities for threshold finding
        buy_val_probs_final = buy_val_probs_calibrated
        calibration_applied = True
        
    except Exception as e:
        print(f"   [WARN] Calibration failed: {e}")
        print(f"   Using uncalibrated probabilities")
        buy_calibrated = None
        buy_val_probs_final = buy_val_probs
        calibration_applied = False
    
    # Find optimal threshold on calibrated probabilities
    buy_threshold, buy_val_f1 = find_optimal_threshold_roc(buy_val_probs_final, buy_val, 'BUY Classifier')
    buy_pred_val = (buy_val_probs_final >= buy_threshold).astype(int)
    
    # Calculate advanced metrics
    buy_mcc = matthews_corrcoef(buy_val, buy_pred_val)
    buy_kappa = cohen_kappa_score(buy_val, buy_pred_val)
    
    # Precision-Recall AUC (more informative for imbalanced data)
    precision_vals, recall_vals, _ = precision_recall_curve(buy_val, buy_val_probs)
    buy_pr_auc = auc(recall_vals, precision_vals)
    
    print(f"\n[EVAL] IS-BUY EVALUATION:\n")
    print(classification_report(buy_val, buy_pred_val, target_names=['NOT_BUY', 'BUY']))
    print(f"\n[METRICS] ADVANCED METRICS:")
    print(f"   Matthews Correlation Coefficient (MCC): {buy_mcc:.4f} (range: -1 to 1, 0=random)")
    print(f"   Cohen's Kappa Score: {buy_kappa:.4f} (>0.6=substantial agreement)")
    print(f"   Precision-Recall AUC: {buy_pr_auc:.4f} (better for imbalanced data)")
    
    cm_buy = confusion_matrix(buy_val, buy_pred_val)
    cm_buy_pct = cm_buy.astype(float) / cm_buy.sum() * 100
    print(f"\n[MATRIX] Confusion Matrix (Absolute & Percentage):")
    print(f"         Pred NO        Pred YES")
    print(f"Act NO   {cm_buy[0,0]:>7} ({cm_buy_pct[0,0]:>5.1f}%)  {cm_buy[0,1]:>7} ({cm_buy_pct[0,1]:>5.1f}%)")
    print(f"Act YES  {cm_buy[1,0]:>7} ({cm_buy_pct[1,0]:>5.1f}%)  {cm_buy[1,1]:>7} ({cm_buy_pct[1,1]:>5.1f}%)")
    print(f"Optimal BUY threshold applied: {buy_threshold:.2f} (F1={buy_val_f1:.3f})")
    
    # ========================================
    # TRAIN IS-SELL CLASSIFIER
    # ========================================
    print(f"\n{'='*70}")
    print("  TRAINING IS-SELL CLASSIFIER")
    print(f"{'='*70}\n")
    
    # Apply BorderlineSMOTE with temporal weighting (generate synthetic minority examples)
    if use_oversample:
        # Calculate temporal weights for recent data upweighting
        temporal_weights = calculate_temporal_weights(len(X_train), recent_months=6)
        X_train_sell, sell_train_aug = apply_borderline_smote_to_sequences(
            X_train, sell_train, sample_weights=temporal_weights
        )
    else:
        X_train_sell, sell_train_aug = X_train, sell_train
    
    # NOTE: Label smoothing removed - was interfering with gradient signals
    # The model learns better with hard labels when using class weights
    print(f"   SELL labels: {sell_train_aug.sum()} positive ({sell_train_aug.mean()*100:.1f}%) after SMOTE")
    
    # Create model
    sell_model = create_binary_classifier(
        sequence_length,
        len(feature_cols),
        name='is_sell_classifier',
        output_bias=sell_output_bias
    )
    param_count = sell_model.count_params()
    print(f"[OK] Model created: {param_count:,} parameters")
    print(f"   Expected: ~75K-100K parameters (reduced from 301K to prevent overfitting)")
    if 50_000 < param_count < 150_000:
        print(f"   [OK] Parameter count in target range for small datasets")
    else:
        print(f"   [WARN] WARNING: Parameter count {param_count:,} outside expected range (50K-150K)!")
    
    # Compile - SELL uses BCE (not focal loss) to ensure stable gradients
    # Focal loss was suppressing gradients too aggressively for SELL
    # Using BCE with class weights provides better learning signal
    print(f"   Using Binary Crossentropy (not focal loss) for stable SELL training")
    
    # Use Adam with gradient clipping and higher LR to escape local minima
    sell_optimizer = keras.optimizers.Adam(
        learning_rate=0.001,   # Same as BUY - previous 0.0005 was too low
        clipnorm=1.0           # Keep gradient clipping for stability
    )
    
    sell_model.compile(
        optimizer=sell_optimizer,
        loss='binary_crossentropy',  # BCE instead of focal loss
        metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
    )
    
    # Train
    print("\n[CHECK] Validating SELL training data...")
    if np.any(np.isnan(X_train_sell)) or np.any(np.isinf(X_train_sell)):
        raise ValueError("[FAIL] X_train_sell contains NaN or Inf values! Check feature engineering.")
    if np.any(np.isnan(sell_train_aug)):
        raise ValueError("[FAIL] sell_train contains NaN values!")
    print("   [OK] SELL training data validated (no NaN/Inf)")
    
    print("\n[TRAIN] Training SELL classifier...")
    logger.info(f"[OK] Training with {EXPECTED_FEATURE_COUNT} features (93 technical + 20 new + 34 sentiment)")
    logger.info(f"  Input shape: {X_train_sell.shape} = (samples={X_train_sell.shape[0]}, seq_len={X_train_sell.shape[1]}, features={X_train_sell.shape[2]})")
    assert X_train_sell.shape[2] == EXPECTED_FEATURE_COUNT, f"Feature dimension mismatch: {X_train_sell.shape[2]} != {EXPECTED_FEATURE_COUNT}"
    
    # Create callbacks for SELL classifier
    # Simplified approach: just use EarlyStopping and ReduceLROnPlateau
    sell_variance_callback = OutputVarianceCallback(
        validation_data=(X_val, sell_val),
        check_interval=5,
        collapse_threshold=0.01
    )
    
    sell_callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            patience=8,
            factor=0.5,
            min_lr=1e-6,
            verbose=1
        ),
        sell_variance_callback
    ]
    
    sell_history = sell_model.fit(
        X_train_sell, sell_train_aug,  # Use original labels (not smoothed)
        validation_data=(X_val, sell_val),
        epochs=epochs,
        batch_size=batch_size,
        class_weight=sell_class_weight,
        callbacks=sell_callbacks,
        verbose=1
    )
    
    # Final validation check for mode collapse
    sell_final_preds = sell_model.predict(X_val, verbose=0).flatten()
    sell_final_std = float(np.std(sell_final_preds))
    sell_final_mean = float(np.mean(sell_final_preds))
    print(f"\n=== SELL Classifier Training Complete ===")
    print(f"Final output: mean={sell_final_mean:.4f}, std={sell_final_std:.4f}")
    if sell_final_std < 0.01:
        print("[FAIL] FAILURE: SELL model collapsed to constant output - consider adjusting hyperparameters")
    else:
        print("[OK] SUCCESS: SELL model has dynamic output range")
    
    # Evaluate with diagnostics
    sell_val_probs = diagnose_probabilities(sell_model, X_val, sell_val, name='SELL Classifier')
    
    # Calculate entropy before calibration
    sell_entropy_before, _ = calculate_prediction_entropy(sell_val_probs)
    print(f"\n[ENTROPY] Entropy Analysis (before calibration):")
    print(f"   Average entropy: {sell_entropy_before:.3f}")
    if sell_entropy_before < 0.3:
        print(f"   [WARN] WARNING: Entropy < 0.3 - model may be overconfident!")
    elif sell_entropy_before > 0.6:
        print(f"   [INFO] Entropy > 0.6 - model is uncertain")
    else:
        print(f"   [OK] Entropy in healthy range (0.3-0.6)")
    
    # Apply Platt scaling (isotonic regression calibration)
    print(f"\n[CAL] Applying probability calibration (Platt scaling with isotonic regression)...")
    sell_wrapper = KerasClassifierWrapper(sell_model)
    
    try:
        # Use isotonic regression (non-parametric, more flexible than sigmoid)
        sell_calibrated = CalibratedClassifierCV(
            sell_wrapper,
            method='isotonic',
            cv='prefit'
        )
        
        # Fit calibration on validation set
        sell_calibrated.fit(X_val, sell_val)
        
        # Get calibrated probabilities
        sell_val_probs_calibrated = sell_calibrated.predict_proba(X_val)[:, 1]
        
        print(f"   [OK] Calibration successful")
        print(f"   Calibrated probability distribution:")
        print(f"     Min:    {sell_val_probs_calibrated.min():.6f}")
        print(f"     Max:    {sell_val_probs_calibrated.max():.6f}")
        print(f"     Mean:   {sell_val_probs_calibrated.mean():.6f}")
        print(f"     Median: {np.median(sell_val_probs_calibrated):.6f}")
        
        # Calculate entropy after calibration
        sell_entropy_after, _ = calculate_prediction_entropy(sell_val_probs_calibrated)
        print(f"\n[ENTROPY] Entropy Analysis (after calibration):")
        print(f"   Average entropy: {sell_entropy_after:.3f}")
        print(f"   Entropy change: {sell_entropy_after - sell_entropy_before:+.3f}")
        if sell_entropy_after < 0.3:
            print(f"   [WARN] WARNING: Entropy < 0.3 - model still overconfident!")
        elif sell_entropy_after > 0.6:
            print(f"   [INFO] Entropy > 0.6 - model is uncertain")
        else:
            print(f"   [OK] Entropy in healthy range (0.3-0.6)")
        
        # Use calibrated probabilities for threshold finding
        sell_val_probs_final = sell_val_probs_calibrated
        sell_calibration_applied = True
        
    except Exception as e:
        print(f"   [WARN] Calibration failed: {e}")
        print(f"   Using uncalibrated probabilities")
        sell_calibrated = None
        sell_val_probs_final = sell_val_probs
        sell_calibration_applied = False
    
    # Find optimal threshold on calibrated probabilities
    sell_threshold, sell_val_f1 = find_optimal_threshold_roc(sell_val_probs_final, sell_val, 'SELL Classifier')
    sell_pred_val = (sell_val_probs_final >= sell_threshold).astype(int)
    
    # Calculate advanced metrics
    sell_mcc = matthews_corrcoef(sell_val, sell_pred_val)
    sell_kappa = cohen_kappa_score(sell_val, sell_pred_val)
    
    # Precision-Recall AUC
    precision_vals, recall_vals, _ = precision_recall_curve(sell_val, sell_val_probs)
    sell_pr_auc = auc(recall_vals, precision_vals)
    
    print(f"\n[EVAL] IS-SELL EVALUATION:\n")
    print(classification_report(sell_val, sell_pred_val, target_names=['NOT_SELL', 'SELL']))
    print(f"\n[METRICS] ADVANCED METRICS:")
    print(f"   Matthews Correlation Coefficient (MCC): {sell_mcc:.4f} (range: -1 to 1, 0=random)")
    print(f"   Cohen's Kappa Score: {sell_kappa:.4f} (>0.6=substantial agreement)")
    print(f"   Precision-Recall AUC: {sell_pr_auc:.4f} (better for imbalanced data)")
    
    cm_sell = confusion_matrix(sell_val, sell_pred_val)
    cm_sell_pct = cm_sell.astype(float) / cm_sell.sum() * 100
    print(f"\n[MATRIX] Confusion Matrix (Absolute & Percentage):")
    print(f"         Pred NO        Pred YES")
    print(f"Act NO   {cm_sell[0,0]:>7} ({cm_sell_pct[0,0]:>5.1f}%)  {cm_sell[0,1]:>7} ({cm_sell_pct[0,1]:>5.1f}%)")
    print(f"Act YES  {cm_sell[1,0]:>7} ({cm_sell_pct[1,0]:>5.1f}%)  {cm_sell[1,1]:>7} ({cm_sell_pct[1,1]:>5.1f}%)")
    print(f"Optimal SELL threshold applied: {sell_threshold:.2f} (F1={sell_val_f1:.3f})")
    
    # Combined evaluation
    print(f"\n{'='*70}")
    print("  COMBINED SIGNAL ANALYSIS")
    print(f"{'='*70}\n")
    
    # Fusion logic
    final_signals = np.zeros(len(buy_pred_val))  # 0=HOLD, 1=BUY, -1=SELL
    
    buy_yes = buy_pred_val == 1
    sell_yes = sell_pred_val == 1
    
    final_signals[buy_yes & ~sell_yes] = 1  # BUY: buy=yes, sell=no
    final_signals[sell_yes & ~buy_yes] = -1  # SELL: sell=yes, buy=no
    # Conflicting or both no → HOLD (stays 0)
    
    print(f"Signal distribution:")
    print(f"  BUY:  {np.sum(final_signals == 1)} ({np.mean(final_signals == 1)*100:.1f}%)")
    print(f"  HOLD: {np.sum(final_signals == 0)} ({np.mean(final_signals == 0)*100:.1f}%)")
    print(f"  SELL: {np.sum(final_signals == -1)} ({np.mean(final_signals == -1)*100:.1f}%)")
    
    # Use new organized path structure
    paths = ModelPaths(symbol)
    paths.ensure_dirs()
    
    # Also maintain legacy paths for backward compatibility
    legacy_paths = get_legacy_classifier_paths(symbol)
    legacy_dir = legacy_paths['buy_weights'].parent
    legacy_dir.mkdir(parents=True, exist_ok=True)

    if np.sum(final_signals == 1) == 0 or np.sum(final_signals == -1) == 0:
        print(f"\n[WARN] WARNING: Some signals have zero occurrences!")
        print(f"   Consider: --epochs {epochs*2} or adjust percentile thresholds")
        buy_model.save_weights(str(paths.classifiers.buy_weights))
        sell_model.save_weights(str(paths.classifiers.sell_weights))

    # Save
    print(f"\n[SAVE] Saving models...")

    # Save calibrated models if calibration was successful
    if calibration_applied and buy_calibrated is not None:
        with open(paths.classifiers.buy_calibrated, 'wb') as f:
            pickle.dump(buy_calibrated, f)
        with open(legacy_paths['buy_calibrated'], 'wb') as f:
            pickle.dump(buy_calibrated, f)
        print(f"   [OK] Saved calibrated BUY classifier")
    else:
        print(f"   [WARN] BUY calibration not saved (calibration failed)")
    
    if sell_calibration_applied and sell_calibrated is not None:
        with open(paths.classifiers.sell_calibrated, 'wb') as f:
            pickle.dump(sell_calibrated, f)
        with open(legacy_paths['sell_calibrated'], 'wb') as f:
            pickle.dump(sell_calibrated, f)
        print(f"   [OK] Saved calibrated SELL classifier")
    else:
        print(f"   [WARN] SELL calibration not saved (calibration failed)")

    # Save weights to new structure
    buy_model.save_weights(str(paths.classifiers.buy_weights))
    sell_model.save_weights(str(paths.classifiers.sell_weights))
    
    # Save weights to legacy paths
    buy_model.save_weights(str(legacy_paths['buy_weights']))
    sell_model.save_weights(str(legacy_paths['sell_weights']))

    # Save full Keras models (SavedModel via model.export) - new paths
    buy_model_dir = paths.classifiers.buy_model
    sell_model_dir = paths.classifiers.sell_model
    safe_export_model(buy_model, buy_model_dir)
    safe_export_model(sell_model, sell_model_dir)
    
    # Also export to legacy paths (with extra delay to ensure file handles are released)
    import time, gc
    time.sleep(0.5)
    gc.collect()
    safe_export_model(buy_model, legacy_paths['buy_model'])
    safe_export_model(sell_model, legacy_paths['sell_model'])

    # Save scalers and features - new paths
    with open(paths.classifiers.feature_scaler, 'wb') as f:
        pickle.dump(scaler, f)
    with open(paths.classifiers.features, 'wb') as f:
        pickle.dump(feature_cols, f)
    # Also save canonical feature columns at symbol level
    with open(paths.feature_columns, 'wb') as f:
        pickle.dump(feature_cols, f)
    
    # Save to legacy paths
    with open(legacy_paths['feature_scaler'], 'wb') as f:
        pickle.dump(scaler, f)
    with open(legacy_paths['features'], 'wb') as f:
        pickle.dump(feature_cols, f)

    # Verify saved feature columns
    try:
        saved_cols = pickle.load(open(paths.feature_columns, 'rb'))
        assert len(saved_cols) == EXPECTED_FEATURE_COUNT, f"Saved feature count mismatch: {len(saved_cols)} != {EXPECTED_FEATURE_COUNT}"
    except Exception as e:
        print(f"Warning: could not verify saved feature columns: {e}")

    # Metadata
    buy_class_weight_serializable = {int(k): float(v) for k, v in buy_class_weight.items()}
    sell_class_weight_serializable = {int(k): float(v) for k, v in sell_class_weight.items()}
    metadata = {
        'symbol': symbol,
        'horizon': 1,
        'sequence_length': sequence_length,
        'n_features': len(feature_cols),
        'is_buy': {
            'n_positive': int(buy_train.sum()),
            'n_negative': int(len(buy_train) - buy_train.sum()),
            'val_pos': int(buy_val.sum()),
            'val_neg': int(len(buy_val) - buy_val.sum()),
            'epochs_trained': len(buy_history.history['loss']),
        },
        'is_sell': {
            'n_positive': int(sell_train.sum()),
            'n_negative': int(len(sell_train) - sell_train.sum()),
            'val_pos': int(sell_val.sum()),
            'val_neg': int(len(sell_val) - sell_val.sum()),
            'epochs_trained': len(sell_history.history['loss']),
        },
        'final_signal_distribution': {
            'BUY': int(np.sum(final_signals == 1)),
            'HOLD': int(np.sum(final_signals == 0)),
            'SELL': int(np.sum(final_signals == -1))
        },
        'trained_date': datetime.now().isoformat(),
        'oversampling': use_oversample,
        'loss': 'binary_focal' if use_focal_loss else 'binary_crossentropy',
        'focal_gamma': focal_gamma,
        'focal_alpha': focal_alpha,
        'focal_params': {
            'buy': {'gamma': focal_gamma, 'alpha': focal_alpha},
            'sell': {'gamma': sell_gamma, 'alpha': sell_alpha}
        },
        'label_percentiles': {
            'buy_top_percentile': buy_percentile,
            'sell_bottom_percentile': sell_percentile
        },
        'val_f1': {
            'buy': float(buy_val_f1),
            'sell': float(sell_val_f1)
        },
        'advanced_metrics': {
            'buy': {
                'mcc': float(buy_mcc),
                'kappa': float(buy_kappa),
                'pr_auc': float(buy_pr_auc)
            },
            'sell': {
                'mcc': float(sell_mcc),
                'kappa': float(sell_kappa),
                'pr_auc': float(sell_pr_auc)
            }
        },
        'thresholds': {
            'buy_optimal': float(buy_threshold),
            'sell_optimal': float(sell_threshold),
            'default': DEFAULT_SIGNAL_THRESHOLD
        },
        'class_weights': {
            'buy': buy_class_weight_serializable,
            'sell': sell_class_weight_serializable
        },
        'calibration': {
            'method': 'isotonic',
            'buy_applied': calibration_applied,
            'sell_applied': sell_calibration_applied,
            'buy_entropy_before': float(buy_entropy_before),
            'buy_entropy_after': float(buy_entropy_after) if calibration_applied else None,
            'sell_entropy_before': float(sell_entropy_before),
            'sell_entropy_after': float(sell_entropy_after) if sell_calibration_applied else None,
        },
        'temperature_scaling': {
            'default_temperature': DEFAULT_CLASSIFIER_TEMPERATURE,
            'enabled': True,
            'interpretation': {
                'low': '0.5-1.0 = more confident predictions',
                'normal': '1.0 = no scaling',
                'high': '1.5-3.0 = more uncertain predictions'
            }
        },
        'architecture': {
            'lstm_units': 32,   # Reduced from 64 to prevent overfitting
            'd_model': 64,      # Reduced from 128 to prevent overfitting
            'num_heads': 4,
            'num_blocks': 3,    # Reduced from 6 to prevent overfitting
            'ff_dim': 128,      # Reduced from 256 to prevent overfitting
            'dropout': 0.35,
            'batch_norm': False,
            'multitask': False,
            'multitask_heads': 1,
            'multitask_outputs': ['binary'],
            'capacity_note': 'Reduced from 301K to ~75-100K params to prevent overfitting'
        }
    }

    # Save metadata to new and legacy paths
    with open(paths.classifiers.metadata, 'wb') as f:
        pickle.dump(metadata, f)
    with open(legacy_paths['metadata'], 'wb') as f:
        pickle.dump(metadata, f)
    print(f"\n   [OK] Models, scalers and metadata saved to {paths.classifiers.base}")
    print(f"   [OK] Legacy paths also updated for backward compatibility")
    
    # Print calibration summary
    print(f"\n{'='*70}")
    print("  CALIBRATION SUMMARY")
    print(f"{'='*70}\n")
    print(f"BUY Classifier:")
    print(f"  Calibration applied: {calibration_applied}")
    print(f"  Entropy before: {buy_entropy_before:.3f}")
    if calibration_applied:
        print(f"  Entropy after:  {buy_entropy_after:.3f} ({buy_entropy_after - buy_entropy_before:+.3f})")
    print(f"  Optimal threshold: {buy_threshold:.3f}")
    
    print(f"\nSELL Classifier:")
    print(f"  Calibration applied: {sell_calibration_applied}")
    print(f"  Entropy before: {sell_entropy_before:.3f}")
    if sell_calibration_applied:
        print(f"  Entropy after:  {sell_entropy_after:.3f} ({sell_entropy_after - sell_entropy_before:+.3f})")
    print(f"  Optimal threshold: {sell_threshold:.3f}")
    
    print(f"\nTemperature Scaling:")
    print(f"  Default temperature: {DEFAULT_CLASSIFIER_TEMPERATURE}")
    print(f"  To adjust during inference, modify CLASSIFIER_TEMPERATURE parameter")

    # ========================================
    # COMPREHENSIVE TRAINING SUMMARY
    # ========================================
    training_end_time = time.time()
    training_duration = training_end_time - training_start_time
    training_minutes = int(training_duration // 60)
    training_seconds = int(training_duration % 60)
    
    print(f"\n{'='*70}")
    print("  TRAINING SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"[TIME] Training Time: {training_minutes}m {training_seconds}s ({training_duration:.1f}s total)")
    
    print(f"\n[CONFIG] Configuration Applied:")
    print(f"   * Label thresholds: BUY>{buy_percentile}th percentile, SELL<{sell_percentile}th percentile")
    print(f"   * SMOTE: k_neighbors=10, m_neighbors=10, target=50/50 balance")
    print(f"   * Focal loss: gamma={focal_gamma}, alpha(BUY)={focal_alpha}, alpha(SELL)={sell_alpha}")
    print(f"   * Model capacity: LSTM=32, D_MODEL=64, BLOCKS=3 (~75-100K params)")
    
    print(f"\n[RESULTS] Final Validation Metrics:")
    print(f"   BUY Classifier:")
    print(f"      * Accuracy: {buy_history.history['accuracy'][-1]:.4f}")
    print(f"      * Val Accuracy: {buy_history.history.get('val_accuracy', [0])[-1]:.4f}")
    print(f"      * F1 Score: {buy_val_f1:.4f}")
    print(f"      * Output std: {buy_final_std:.4f} {'[OK]' if buy_final_std >= 0.01 else '[FAIL] COLLAPSED'}")
    print(f"   SELL Classifier:")
    print(f"      * Accuracy: {sell_history.history['accuracy'][-1]:.4f}")
    print(f"      * Val Accuracy: {sell_history.history.get('val_accuracy', [0])[-1]:.4f}")
    print(f"      * F1 Score: {sell_val_f1:.4f}")
    print(f"      * Output std: {sell_final_std:.4f} {'[OK]' if sell_final_std >= 0.01 else '[FAIL] COLLAPSED'}")
    
    print(f"\n[SAVE] Saved Files:")
    print(f"   * BUY weights:      {paths.classifiers.buy_weights}")
    print(f"   * SELL weights:     {paths.classifiers.sell_weights}")
    print(f"   * Feature scaler:   {paths.classifiers.feature_scaler}")
    print(f"   * Feature columns:  {paths.feature_columns}")
    print(f"   * Metadata:         {paths.classifiers.metadata}")
    if calibration_applied:
        print(f"   * BUY calibrated:   {paths.classifiers.buy_calibrated}")
    if sell_calibration_applied:
        print(f"   * SELL calibrated:  {paths.classifiers.sell_calibrated}")
    
    # Overall health check
    print(f"\n[CHECK] Health Check:")
    health_issues = []
    if buy_final_std < 0.01:
        health_issues.append("BUY model collapsed (std < 0.01)")
    if sell_final_std < 0.01:
        health_issues.append("SELL model collapsed (std < 0.01)")
    if buy_val_f1 < 0.3:
        health_issues.append(f"BUY F1 score low ({buy_val_f1:.3f} < 0.3)")
    if sell_val_f1 < 0.3:
        health_issues.append(f"SELL F1 score low ({sell_val_f1:.3f} < 0.3)")
    
    if not health_issues:
        print("   [OK] All checks passed - models ready for inference")
    else:
        print("   [WARN] Issues detected:")
        for issue in health_issues:
            print(f"      * {issue}")
        print("   Consider adjusting hyperparameters and retraining")

    return buy_model, sell_model, scaler, metadata, buy_calibrated, sell_calibrated


def save_ensemble_metadata(symbol: str, members: List[dict]):
    if not members:
        return
    
    # Use new path structure
    paths = ModelPaths(symbol)
    paths.ensure_dirs()
    legacy_paths = get_legacy_classifier_paths(symbol)

    base_meta = members[0].copy()
    buy_thresholds = [m.get('thresholds', {}).get('buy_optimal', DEFAULT_SIGNAL_THRESHOLD) for m in members]
    sell_thresholds = [m.get('thresholds', {}).get('sell_optimal', DEFAULT_SIGNAL_THRESHOLD) for m in members]
    base_meta.setdefault('thresholds', {})
    base_meta['thresholds']['buy_optimal'] = float(np.mean(buy_thresholds))
    base_meta['thresholds']['sell_optimal'] = float(np.mean(sell_thresholds))
    base_meta['ensemble'] = {
        'size': len(members),
        'members': [
            {
                'suffix': m.get('model_suffix', ''),
                'thresholds': m.get('thresholds', {}),
                'val_f1': m.get('val_f1', {})
            }
            for m in members
        ]
    }
    
    # Save to new path
    with open(paths.classifiers.metadata, 'wb') as f:
        pickle.dump(base_meta, f)
    # Also save to legacy path
    with open(legacy_paths['metadata'], 'wb') as f:
        pickle.dump(base_meta, f)
    print(f"\n   [OK] Aggregated ensemble metadata saved ({len(members)} members)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('symbol', type=str)
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs (default: 30)')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Batch size for training (default: 512 for powerful GPU)')
    parser.add_argument(
        '--no-oversample',
        dest='use_oversample',
        action='store_false',
        help='Disable BorderlineSMOTE synthetic oversampling for minority sequences.'
    )
    parser.add_argument(
        '--no-smote',
        dest='use_oversample',
        action='store_false',
        help='Deprecated alias for --no-oversample.'
    )
    parser.add_argument('--buy-percentile', type=int, default=60,
                        help='Percentile threshold for BUY positives (default: 60 -> top 40%).')
    parser.add_argument('--sell-percentile', type=int, default=40,
                        help='Percentile threshold for SELL positives (default: 40 -> bottom 40%).')
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help='Gamma parameter for binary focal loss (default: 2.0, reduced from 3.0 for stability).')
    parser.add_argument('--focal-alpha', type=float, default=0.75,
                        help='Alpha balance term for binary focal loss (BUY default: 0.75).')
    parser.add_argument('--sell-focal-gamma', type=float, default=None,
                        help='Override gamma for SELL classifier focal loss (default: match BUY).')
    parser.add_argument('--sell-focal-alpha', type=float, default=0.75,
                        help='Override alpha for SELL classifier focal loss (default: 0.75, same as BUY for balanced training).')
    parser.add_argument('--ensemble-size', type=int, default=1,
                        help='Number of classifier instances to train with different seeds.')
    parser.add_argument(
        '--base-seed',
        type=int,
        default=42,
        help='Base seed for ensemble members (default: 42).'
    )
    parser.add_argument(
        '--force-refresh',
        action='store_true',
        help='Force regeneration of features (ignore cache).'
    )
    parser.add_argument(
        '--no-focal-loss',
        dest='use_focal_loss',
        action='store_false',
        help='Use standard binary crossentropy instead of focal loss.'
    )
    parser.set_defaults(use_oversample=True)
    parser.set_defaults(use_focal_loss=True)
    args = parser.parse_args()

    # Setup file-based logging for this training run
    global logger
    logger = setup_training_logger(args.symbol, 'classifiers')
    logger.info(f"Starting binary classifiers training for {args.symbol}")
    logger.info(f"Arguments: epochs={args.epochs}, batch_size={args.batch_size}, "
                f"buy_percentile={args.buy_percentile}, sell_percentile={args.sell_percentile}")

    ensemble_size = max(1, args.ensemble_size)
    ensemble_metadata = []
    for idx in range(ensemble_size):
        suffix = '' if idx == 0 else f'_ens{idx+1}'
        seed = (args.base_seed + idx * 17) if args.base_seed is not None else None
        print(f"\n=== Ensemble member {idx+1}/{ensemble_size} (suffix='{suffix}' or base) ===")
        logger.info(f"Training ensemble member {idx+1}/{ensemble_size}")
        _, _, _, member_meta, _, _ = train_binary_classifiers(
            symbol=args.symbol,
            epochs=args.epochs,
            batch_size=args.batch_size,
            use_oversample=args.use_oversample,
            buy_percentile=args.buy_percentile,
            sell_percentile=args.sell_percentile,
            use_focal_loss=args.use_focal_loss,
            focal_gamma=args.focal_gamma,
            focal_alpha=args.focal_alpha,
            sell_focal_gamma=args.sell_focal_gamma,
            sell_focal_alpha=args.sell_focal_alpha,
            model_suffix=suffix,
            random_seed=seed,
            use_cache=not args.force_refresh
        )
        ensemble_metadata.append(member_meta)

    save_ensemble_metadata(args.symbol, ensemble_metadata)

    logger.info("Binary classifiers training complete!")
    print("\n[OK] Binary classifiers training complete!")
    print("   Saved models and metadata in saved_models/")
    print(f"   Ensemble size: {ensemble_size}")
    print("   Next: run backtest/fusion script to evaluate combined strategy.")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n[WARN] Training interrupted by user")
        import sys
        sys.exit(1)
    except Exception as e:
        print(f"\n[FAIL] FATAL ERROR during classifier training: {e}")
        import traceback
        traceback.print_exc()
        import sys
        sys.exit(1)
              