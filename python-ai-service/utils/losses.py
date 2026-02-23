"""
Custom Loss Functions for Stock Prediction Models

This module contains custom loss functions designed to improve:
1. Directional accuracy (directional_mse)
2. Class imbalance handling (focal_loss)

These losses are registered for model loading/saving.
"""

import tensorflow as tf
from tensorflow import keras


# Use tf.keras.utils.register_keras_serializable for TensorFlow 2.x compatibility
@tf.keras.utils.register_keras_serializable(package="CustomLosses", name="directional_mse")
def directional_mse(y_true, y_pred):
    """
    Custom loss that heavily penalizes wrong direction predictions

    Combines standard MSE with a directional penalty to encourage
    the model to predict the correct direction of price movement,
    which is more important than exact magnitude for trading decisions.

    Args:
        y_true: Actual log returns (shape: [batch_size, 1] or [batch_size])
        y_pred: Predicted log returns (shape: [batch_size, 1] or [batch_size])

    Returns:
        Combined MSE + directional penalty loss (scalar)

    Formula:
        loss = MSE + 3.0 * mean(|y_true - y_pred| * wrong_direction_indicator)

    Where wrong_direction_indicator = 1 if sign(y_true) != sign(y_pred), else 0

    Example:
        If actual = +0.02 (2% gain) and predicted = -0.01 (1% loss):
        - MSE = 0.0009
        - Direction penalty = 3.0 * 0.03 = 0.09
        - Total loss = 0.0009 + 0.09 = 0.0909 (heavily penalized!)

    Edge Cases:
        - Zero returns: sign(0) = 0, treated as neutral
        - Near-zero returns: May cause instability if used as primary loss

    Notes:
        - 3x multiplier chosen empirically (can be tuned)
        - Works best when combined with proper data normalization
        - Effective for time series with clear directional patterns
    """
    # Flatten if needed (handle both [batch, 1] and [batch] shapes)
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    # Standard MSE component
    mse = tf.reduce_mean(tf.square(y_true - y_pred))

    # Direction penalty (2x weight when wrong direction - reduced from 3x)
    true_sign = tf.sign(y_true)
    pred_sign = tf.sign(y_pred)
    wrong_direction = tf.cast(tf.not_equal(true_sign, pred_sign), tf.float32)
    direction_penalty = 2.0 * wrong_direction * tf.abs(y_true - y_pred)

    # Total loss
    total_loss = mse + tf.reduce_mean(direction_penalty)

    return total_loss


@tf.keras.utils.register_keras_serializable(package='CustomLosses', name='SimpleDirectionalMSE')
class SimpleDirectionalMSE(tf.keras.losses.Loss):
    """
    NUCLEAR FIX: Simple directional MSE loss without anti-collapse penalties.

    Research finding (December 2025): Variance collapse is caused by REGULARIZATION,
    not weak loss penalties. The solution is architecture-based (zero-initialized
    output layer, residual connections), NOT loss function penalties.

    This loss follows the research-backed formula:
        L = mse_weight * MSE + direction_weight * DirectionalPenalty

    Recommended: mse_weight=0.4, direction_weight=0.6 (from research)

    NO anti-collapse penalties (they create competing objectives and cause instability)
    NO variance penalties (architecture handles this now)
    NO sign diversity penalties (these fight with directional penalty)

    Args:
        mse_weight: Weight for MSE component (default 0.4)
        direction_weight: Weight for directional penalty (default 0.6)
        name: Name for the loss function

    Example:
        >>> loss_fn = SimpleDirectionalMSE(mse_weight=0.4, direction_weight=0.6)
        >>> model.compile(optimizer='adam', loss=loss_fn)
    """

    def __init__(self, mse_weight=0.4, direction_weight=0.6, name='simple_directional_mse', **kwargs):
        super().__init__(name=name, **kwargs)
        self.mse_weight = mse_weight
        self.direction_weight = direction_weight

    def call(self, y_true, y_pred):
        """Compute simple directional MSE loss."""
        # Flatten if needed
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        # Cast to float32 for numerical stability
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # 1. MSE component (magnitude accuracy)
        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        # 2. Directional penalty (direction accuracy)
        # Penalize predictions that have wrong sign
        true_sign = tf.sign(y_true)
        pred_sign = tf.sign(y_pred)
        wrong_direction = tf.cast(tf.not_equal(true_sign, pred_sign), tf.float32)
        # Penalty proportional to error magnitude when direction is wrong
        directional_penalty = tf.reduce_mean(wrong_direction * tf.abs(y_true - y_pred))

        # Combined loss (research-backed formula)
        total_loss = self.mse_weight * mse + self.direction_weight * directional_penalty

        # Safety: if NaN, return MSE only
        total_loss = tf.where(
            tf.math.is_finite(total_loss),
            total_loss,
            mse
        )

        return total_loss

    def get_config(self):
        """Config for serialization."""
        config = super().get_config()
        config.update({
            'mse_weight': self.mse_weight,
            'direction_weight': self.direction_weight,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Reconstruct from config."""
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package='CustomLosses', name='BalancedDirectionalLoss')
class BalancedDirectionalLoss(tf.keras.losses.Loss):
    """
    NUCLEAR FIX v4.3: Balanced directional loss with INVERSE FREQUENCY weighting.

    Problem: SimpleDirectionalMSE penalizes wrong direction equally for both positive
    and negative targets. But if targets are biased positive (~53% in stocks), the model
    learns to always predict positive (safer choice).

    Solution: Weight the directional penalty by the INVERSE of class frequency.
    This makes missing a rare negative as costly as missing a common positive.

    Formula:
        For positive targets (more common): penalty = base_penalty / positive_freq
        For negative targets (less common): penalty = base_penalty / negative_freq

    This is similar to class weights in classification but applied to regression.

    Args:
        mse_weight: Weight for MSE component (default 0.3)
        direction_weight: Weight for directional penalty (default 0.7)
        positive_freq: Estimated frequency of positive targets (default 0.53)
        name: Name for the loss function

    Research backing:
    - "Forecasting stock prices using LSTM" (Nature Scientific Reports 2023) shows
      LSTMs bias toward positive predictions (ratio up to 3.75:1)
    - Inverse frequency weighting is standard for class imbalance
    """

    def __init__(
        self,
        mse_weight=0.3,
        direction_weight=0.7,
        positive_freq=0.53,  # Stock market positive bias ~53%
        name='balanced_directional_loss',
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.mse_weight = mse_weight
        self.direction_weight = direction_weight
        self.positive_freq = positive_freq

        # Compute inverse frequency weights
        # Higher weight for minority class (negatives)
        self.positive_penalty = 1.0 / positive_freq
        self.negative_penalty = 1.0 / (1.0 - positive_freq)
        # Normalize so average penalty is 1.0
        avg_penalty = 0.5 * (self.positive_penalty + self.negative_penalty)
        self.positive_penalty /= avg_penalty
        self.negative_penalty /= avg_penalty

    def call(self, y_true, y_pred):
        """Compute balanced directional loss."""
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # MSE component
        mse = tf.reduce_mean(tf.square(y_true - y_pred))

        # Identify wrong direction predictions
        true_positive = tf.cast(y_true > 0, tf.float32)
        true_negative = tf.cast(y_true < 0, tf.float32)
        pred_positive = tf.cast(y_pred > 0, tf.float32)
        pred_negative = tf.cast(y_pred < 0, tf.float32)

        # Wrong direction cases
        false_positive = pred_positive * true_negative  # Predicted positive when negative
        false_negative = pred_negative * true_positive  # Predicted negative when positive

        # Apply BALANCED penalties (inverse frequency weighting)
        # Missing a negative (rare) is penalized MORE than missing a positive (common)
        direction_error = tf.abs(y_true - y_pred) * (
            false_positive * self.negative_penalty +
            false_negative * self.positive_penalty
        )
        directional_penalty = tf.reduce_mean(direction_error)

        # Combined loss
        total_loss = self.mse_weight * mse + self.direction_weight * directional_penalty

        # Safety: if NaN, return MSE only
        total_loss = tf.where(
            tf.math.is_finite(total_loss),
            total_loss,
            mse
        )

        return total_loss

    def get_config(self):
        """Config for serialization."""
        config = super().get_config()
        config.update({
            'mse_weight': self.mse_weight,
            'direction_weight': self.direction_weight,
            'positive_freq': self.positive_freq,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Reconstruct from config."""
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package="CustomLosses", name="huber_with_variance_penalty")
def huber_with_variance_penalty(y_true, y_pred, delta=1.0, variance_weight=10.0):
    """
    Huber loss with STRONG variance penalty to prevent predicting near-zero
    
    Problem: Standard Huber loss allows model to predict ~0 for all inputs
    because log returns are small (~0.001), so constant-zero predictions
    minimize loss even though R² is negative.
    
    Solution: Add STRONG penalty for having low prediction variance. This forces
    model to make diverse predictions rather than collapsing to constant.
    
    Args:
        y_true: Actual log returns
        y_pred: Predicted log returns
        delta: Huber loss threshold (default 1.0)
        variance_weight: Weight for variance penalty (default 10.0, INCREASED from 2.0)
    
    Returns:
        Combined Huber loss + variance penalty
    
    Formula:
        loss = Huber(y_true, y_pred) + variance_weight * (1 / (var(y_pred) + epsilon))
        
    Effect:
        - If model predicts constant (var ~ 0): penalty → VERY large
        - If model predicts diverse values: penalty → ~0
        - Forces model to explore prediction space, not collapse to zero
    
    Note: Increased variance_weight to 10.0 for stronger enforcement.
          Capped at 100.0 to prevent extreme gradients.
    """
    # Standard Huber loss
    huber = tf.keras.losses.Huber(delta=delta)
    huber_loss = huber(y_true, y_pred)
    
    # STRONG variance penalty (inverse of variance)
    pred_variance = tf.math.reduce_variance(y_pred)
    # Prevent division by zero or NaN
    pred_variance = tf.maximum(pred_variance, 1e-8)
    # Cap penalty to prevent extreme values
    variance_penalty = tf.minimum(variance_weight / pred_variance, 100.0)
    
    return huber_loss + variance_penalty


@tf.keras.utils.register_keras_serializable(package="CustomLosses", name="r2_metric")
def r2_metric(y_true, y_pred):
    """
    R² (coefficient of determination) metric for regression models

    Measures how well predictions explain variance in the true values.
    R² = 1 means perfect predictions, R² = 0 means model is as good as
    predicting the mean, R² < 0 means model is worse than mean predictor.

    Args:
        y_true: True values (shape: [batch_size])
        y_pred: Predicted values (shape: [batch_size])

    Returns:
        R² score (scalar)

    Formula:
        R² = 1 - (SS_res / SS_tot)
        where:
        - SS_res = Σ(y_true - y_pred)² (residual sum of squares)
        - SS_tot = Σ(y_true - mean(y_true))² (total sum of squares)

    Interpretation:
        - R² = 1.0: Perfect predictions
        - R² > 0.7: Strong predictive power
        - R² > 0.3: Moderate predictive power
        - R² > 0.0: Better than predicting mean
        - R² = 0.0: As good as predicting mean (useless)
        - R² < 0.0: Worse than predicting mean (broken model)

    Note:
        This is used as a monitoring metric during training.
        The model still optimizes the loss function (e.g., Huber loss),
        but R² gives us interpretable feedback on prediction quality.
    """
    ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
    ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))
    return 1.0 - ss_res / (ss_tot + tf.keras.backend.epsilon())


@tf.keras.utils.register_keras_serializable(package="CustomLosses", name="directional_accuracy_metric")
def directional_accuracy_metric(y_true, y_pred):
    """
    Custom metric: fraction of correct sign predictions (NaN-safe).

    Compute directional accuracy: fraction of predictions with correct sign.

    This metric measures how often the model correctly predicts whether
    the return will be positive or negative, regardless of magnitude.
    This is often more important than MAE for trading applications.

    Args:
        y_true: True target values
        y_pred: Predicted values

    Returns:
        Scalar tensor with directional accuracy (0.0 to 1.0)

    Example:
        >>> y_true = tf.constant([0.02, -0.01, 0.05, -0.03])
        >>> y_pred = tf.constant([0.01, -0.02, -0.01, -0.01])
        >>> acc = directional_accuracy_metric(y_true, y_pred)
        >>> # Correct: +/+, -/-, -/-; Wrong: +/-
        >>> # Accuracy = 3/4 = 0.75
    """
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


@tf.keras.utils.register_keras_serializable(package="CustomLosses", name="focal_loss")
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    """
    Focal loss for handling class imbalance in multi-class classification
    
    Focuses training on hard-to-classify examples by down-weighting
    well-classified examples. Particularly effective for imbalanced datasets
    where SELL and BUY classes are minority.
    
    Args:
        y_true: True labels as integers (shape: [batch_size], values: 0, 1, 2)
                0 = SELL, 1 = HOLD, 2 = BUY
        y_pred: Predicted class probabilities (shape: [batch_size, 3])
                Output from softmax layer
        gamma: Focusing parameter (default 2.0)
               Higher gamma = more focus on hard examples
               gamma=0 reduces to standard cross-entropy
        alpha: Balance parameter for class weighting (default 0.25)
               Can be scalar or array for per-class weights
    
    Returns:
        Focal loss value (scalar)
    
    Formula:
        FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
        
    Where p_t is the model's predicted probability for the true class.
    
    Behavior:
        - Well-classified (p_t ~ 1): (1 - p_t)^gamma ~ 0 → loss ~ 0
        - Misclassified (p_t ~ 0): (1 - p_t)^gamma ~ 1 → loss = alpha * CE
        - Hard examples get more weight during training
    
    Example:
        Easy example: y_true=2 (BUY), y_pred=[0.05, 0.05, 0.90]
        - p_t = 0.90
        - focal_weight = (1 - 0.90)^2 = 0.01
        - Loss contribution is small (down-weighted)
        
        Hard example: y_true=0 (SELL), y_pred=[0.35, 0.40, 0.25]
        - p_t = 0.35
        - focal_weight = (1 - 0.35)^2 = 0.42
        - Loss contribution is significant (normal weight)
    
    Edge Cases:
        - Predictions near 0 or 1: Clipped to [1e-7, 1-1e-7] to prevent log(0)
        - All same class: Still works but may need lower gamma
    
    Notes:
        - Introduced in "Focal Loss for Dense Object Detection" (Lin et al., 2017)
        - gamma=2.0, alpha=0.25 are paper defaults (work well in practice)
        - Can tune gamma higher (e.g., 3.0) for extreme imbalance
        - Compatible with sparse integer labels (converts to one-hot internally)
    """
    # Convert sparse labels to one-hot (handle both int32 and int64)
    y_true_int = tf.cast(y_true, tf.int32)
    y_true_one_hot = tf.one_hot(y_true_int, depth=3)
    
    # Clip predictions to prevent log(0) and division by zero
    # Range: [1e-7, 1 - 1e-7]
    epsilon = 1e-7
    y_pred = tf.clip_by_value(y_pred, epsilon, 1.0 - epsilon)
    
    # Cross entropy component: -sum(y_true * log(y_pred))
    ce = -tf.reduce_sum(y_true_one_hot * tf.math.log(y_pred), axis=-1)
    
    # Get predicted probability for true class
    # p_t = sum(y_true * y_pred) extracts the probability of the correct class
    p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
    
    # Focal weight: (1 - p_t)^gamma
    # This is the key innovation - down-weights easy examples
    focal_weight = tf.pow(1.0 - p_t, gamma)
    
    # Apply focal weight and alpha balance
    focal_loss_value = alpha * focal_weight * ce
    
    return tf.reduce_mean(focal_loss_value)


@tf.keras.utils.register_keras_serializable(package="CustomLosses", name="focal_loss_multiclass")
def focal_loss_multiclass(y_true, y_pred, gamma=2.0, alpha_vec=None, from_logits=False):
    """
    Focal loss with auto-computed per-class alpha weights
    
    This improved version computes alpha weights from class frequencies
    automatically, providing better handling of class imbalance.
    
    Args:
        y_true: True labels (sparse integers 0,1,2) of shape (batch,) or (batch,1)
        y_pred: Predicted probabilities of shape (batch, num_classes)
        gamma: Focusing parameter (default 2.0)
        alpha_vec: Optional list/array of per-class weights. If None, uses equal weighting.
                   Should be computed as: total / (n_classes * class_count)
        from_logits: If True, applies softmax to y_pred first
    
    Returns:
        Focal loss (scalar)
    
    Example alpha computation:
        counts = {0: 100, 1: 1000, 2: 150}  # SELL, HOLD, BUY
        total = 1250
        alpha = [total/(3*100), total/(3*1000), total/(3*150)]
              = [4.17, 0.42, 2.78]  # SELL and BUY get higher weight
    """
    # Apply softmax if logits
    if from_logits:
        y_pred = tf.nn.softmax(y_pred, axis=-1)
    
    # Clip predictions
    y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
    
    # Get number of classes
    num_classes = tf.shape(y_pred)[-1]
    
    # Convert sparse labels to one-hot
    if y_true.shape.rank == 1 or (y_true.shape.rank == 2 and y_true.shape[-1] == 1):
        y_true_one_hot = tf.one_hot(tf.cast(tf.reshape(y_true, [-1]), tf.int32), depth=num_classes)
    else:
        y_true_one_hot = tf.cast(y_true, tf.float32)
    
    # Get probability of true class
    p_t = tf.reduce_sum(y_true_one_hot * y_pred, axis=-1)
    
    # Cross entropy
    ce = -tf.reduce_sum(y_true_one_hot * tf.math.log(y_pred), axis=-1)
    
    # Focal weight
    focal_weight = tf.pow(1.0 - p_t, gamma)
    
    # Apply per-class alpha
    if alpha_vec is None:
        alpha = 1.0
    else:
        # Convert python list/dict to tensor
        if isinstance(alpha_vec, dict):
            alpha_list = [alpha_vec[i] for i in range(len(alpha_vec))]
        else:
            alpha_list = list(alpha_vec)
        
        alpha_tensor = tf.constant(alpha_list, dtype=tf.float32)
        alpha = tf.reduce_sum(y_true_one_hot * alpha_tensor, axis=-1)
    
    loss = alpha * focal_weight * ce
    return tf.reduce_mean(loss)


def compute_class_alpha(y_train, method='balanced', scale=1.0):
    """
    Compute per-class alpha weights from training labels
    
    Args:
        y_train: Training labels array
        method: 'balanced' or 'sqrt' (sqrt is less aggressive)
        scale: Multiplier to strengthen/weaken alpha weights (default 1.0)
               Use scale > 1.0 to focus more on minority classes
    
    Returns:
        Dict mapping class_id -> alpha_weight
    """
    import numpy as np
    unique, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    n_classes = len(unique)
    
    if method == 'balanced':
        # Standard balanced: total / (n_classes * count)
        alpha_dict = {int(c): float(scale * total / (n_classes * count)) 
                      for c, count in zip(unique, counts)}
    elif method == 'sqrt':
        # Less aggressive: sqrt of balanced ratio
        alpha_dict = {int(c): float(scale * np.sqrt(total / (n_classes * count)))
                      for c, count in zip(unique, counts)}
    else:
        # Equal weighting
        alpha_dict = {int(c): 1.0 for c in unique}
    
    scale_info = f" (scale={scale})" if scale != 1.0 else ""
    print(f"\n⚖️  Computed focal loss alpha weights ({method}{scale_info}):")
    for c, alpha in sorted(alpha_dict.items()):
        print(f"   Class {c}: {alpha:.3f}")
    
    return alpha_dict


@tf.keras.utils.register_keras_serializable(package='Custom')
class FocalLossWithAlpha:
    """
    Serializable focal loss wrapper with custom alpha weights

    This class is needed because local functions can't be saved with Keras models.
    Use this instead of defining focal_loss_with_alpha as a local function.
    """
    def __init__(self, alpha_dict, gamma=2.0, name='focal_loss_with_alpha'):
        """
        Args:
            alpha_dict: Dict mapping class_id -> alpha_weight
            gamma: Focusing parameter (default 2.0)
            name: Name for the loss function
        """
        self.alpha_dict = alpha_dict
        self.gamma = gamma
        self.name = name

    def __call__(self, y_true, y_pred):
        """Call the focal loss function"""
        return focal_loss_multiclass(y_true, y_pred, gamma=self.gamma, alpha_vec=self.alpha_dict)

    def get_config(self):
        """Config for serialization"""
        return {
            'alpha_dict': self.alpha_dict,
            'gamma': self.gamma,
            'name': self.name
        }

    @classmethod
    def from_config(cls, config):
        """Reconstruct from config"""
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package='Custom', name='BinaryFocalLoss')
class BinaryFocalLoss(tf.keras.losses.Loss):
    """
    Serializable binary focal loss for imbalanced classification.

    This is a proper Keras Loss subclass that can be saved and loaded with models.

    Args:
        gamma: Focusing parameter (default 2.0)
            - gamma=0: equivalent to cross-entropy
            - gamma=2: recommended default
            - gamma=3+: aggressive focusing (may cause instability)
        alpha: Balance parameter for class weighting (default 0.25)
            - alpha=0.5: equal weighting
            - alpha=0.75: focus on minority class
        name: Name for the loss function

    Formula:
        FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma=2.0, alpha=0.25, name='binary_focal_loss', **kwargs):
        super().__init__(name=name, **kwargs)  # Pass reduction and other kwargs to parent
        self.gamma = gamma
        self.alpha = alpha

    def call(self, y_true, y_pred):
        """Compute binary focal loss."""
        # Cast to float32 for numerical stability (mixed precision can cause NaN)
        y_true_cast = tf.cast(y_true, tf.float32)
        y_pred_cast = tf.cast(y_pred, tf.float32)

        # Clip predictions to prevent log(0) and numerical instability
        epsilon = 1e-6  # Increased from 1e-7 for better stability
        y_pred_clipped = tf.clip_by_value(y_pred_cast, epsilon, 1.0 - epsilon)

        # Get predicted probability for true class
        pt = tf.where(tf.equal(y_true_cast, 1.0), y_pred_clipped, 1.0 - y_pred_clipped)

        # Get alpha weight for true class
        alpha_t = tf.where(tf.equal(y_true_cast, 1.0), self.alpha, 1.0 - self.alpha)

        # Compute focal loss with numerical stability checks
        focal_weight = tf.pow(1.0 - pt, self.gamma)
        focal_weight = tf.clip_by_value(focal_weight, 0.0, 1e6)  # Prevent overflow

        log_pt = tf.math.log(pt)
        log_pt = tf.clip_by_value(log_pt, -100.0, 0.0)  # Prevent underflow

        focal_term = -alpha_t * focal_weight * log_pt

        # Check for NaN and replace with large but finite value
        focal_term = tf.where(tf.math.is_finite(focal_term), focal_term, tf.ones_like(focal_term) * 10.0)

        return tf.reduce_mean(focal_term)

    def get_config(self):
        """Config for serialization."""
        config = super().get_config()
        config.update({
            'gamma': self.gamma,
            'alpha': self.alpha
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Reconstruct from config."""
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package='CustomLosses', name='ResidualAwareLoss')
class ResidualAwareLoss(tf.keras.losses.Loss):
    """
    TRULY SIMPLE loss - Huber + small direction penalty ONLY.

    NUCLEAR FIX v6.0 (January 2026) - COMPLETE SIMPLIFICATION:
    Previous version CLAIMED to be simple but actually had 4 competing objectives!
    This caused CONFLICTING GRADIENTS that destabilized training.

    NOW it is ACTUALLY simple:
    - Base: Huber loss (robust to outliers, smooth gradients)
    - Small direction penalty (0.1 weight) - just a nudge for wrong signs
    - NOTHING ELSE - no variance penalty, no balance penalty

    WHY this works:
    1. Simpler loss = stable gradients = no collapse
    2. Variance/balance penalties fight with MSE, creating instability
    3. Architecture handles diversity (multiple output heads, zero-init)
    4. Research shows complex losses hurt more than help

    Args:
        delta: Huber delta parameter (default 1.0)
        direction_weight: Small penalty for wrong direction (default 0.1, NOT 0.5!)
        name: Loss function name

    Research backing:
    - "On Layer Normalization in the Transformer Architecture" (2020) - simpler is better
    - "Deep Learning for Time Series Forecasting" (Zhu et al. 2021) - residual + simple loss
    """

    def __init__(self, delta=1.0, direction_weight=0.1, name='residual_aware_loss', **kwargs):
        # Note: removed bias_weight - we don't use variance/balance penalties anymore
        super().__init__(name=name, **kwargs)
        self.delta = delta
        self.direction_weight = direction_weight
        # Keep Huber loss instance for robust gradient behavior
        self.huber = tf.keras.losses.Huber(delta=delta, reduction='none')

    def call(self, y_true, y_pred):
        """
        TRULY SIMPLE: Huber loss + small direction penalty.
        
        NO variance penalty. NO balance penalty. NO competing objectives.
        Let the architecture handle diversity, not the loss function.
        """
        # Flatten
        y_true = tf.reshape(y_true, [-1])
        y_pred = tf.reshape(y_pred, [-1])

        # Cast to float32 for stability
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        # 1. BASE HUBER LOSS (robust to outliers, smooth gradients)
        # Huber is more stable than MSE because it has bounded gradients
        huber_loss = self.huber(y_true, y_pred)

        # 2. SMALL DIRECTION PENALTY (just a nudge, not aggressive)
        # When target is negative but prediction is positive (or vice versa),
        # add a SMALL penalty. The penalty is proportional to |error| so
        # it naturally scales with the prediction quality.
        true_sign = tf.sign(y_true)
        pred_sign = tf.sign(y_pred)
        wrong_direction = tf.cast(tf.not_equal(true_sign, pred_sign), tf.float32)
        # Small penalty: 0.1 * wrong_direction * |error|
        direction_penalty = self.direction_weight * wrong_direction * tf.abs(y_true - y_pred)

        # TOTAL LOSS = Huber + direction penalty (that's it!)
        total_loss = tf.reduce_mean(huber_loss + direction_penalty)

        # Safety: if NaN, fall back to pure Huber
        fallback = tf.reduce_mean(huber_loss)
        total_loss = tf.where(
            tf.math.is_finite(total_loss),
            total_loss,
            fallback
        )

        return total_loss

    def get_config(self):
        """Config for serialization."""
        config = super().get_config()
        config.update({
            'delta': self.delta,
            'direction_weight': self.direction_weight,
        })
        return config

    @classmethod
    def from_config(cls, config):
        """Reconstruct from config."""
        return cls(**config)


@tf.keras.utils.register_keras_serializable(package='CustomLosses', name='VarianceRegularizedLoss')
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
        base_loss: The underlying loss function to wrap (not serialized, reconstructed in from_config)
        variance_weight: Weight for the variance penalty (default 0.5)
        min_variance_target: Minimum target variance below which penalty grows (default 0.003)
    """

    def __init__(self, base_loss=None, variance_weight=0.5, min_variance_target=0.003, name='variance_regularized'):
        super().__init__(name=name)
        # Use Huber as default base loss if none provided
        self.base_loss = base_loss if base_loss is not None else tf.keras.losses.Huber(delta=1.0)
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

    @classmethod
    def from_config(cls, config):
        """Reconstruct from config - base_loss will be set to default Huber"""
        return cls(
            base_loss=tf.keras.losses.Huber(delta=1.0),
            variance_weight=config.get('variance_weight', 0.5),
            min_variance_target=config.get('min_variance_target', 0.003),
            name=config.get('name', 'variance_regularized')
        )


# Register custom objects for model loading
def get_custom_objects():
    """
    Returns dictionary of custom objects for model loading

    Usage:
        from utils.losses import get_custom_objects
        model = tf.keras.models.load_model(path, custom_objects=get_custom_objects())

    Or register globally:
        from utils.losses import register_custom_objects
        register_custom_objects()
        model = tf.keras.models.load_model(path)  # Auto-detects custom objects
    """
    # Import custom losses from lstm_transformer_paper module
    try:
        from models.lstm_transformer_paper import (
            AntiCollapseDirectionalLoss,
            DirectionalHuberLoss,
        )
        paper_losses = {
            'AntiCollapseDirectionalLoss': AntiCollapseDirectionalLoss,
            'DirectionalHuberLoss': DirectionalHuberLoss,
        }
    except ImportError:
        paper_losses = {}

    return {
        'directional_mse': directional_mse,
        'SimpleDirectionalMSE': SimpleDirectionalMSE,  # NUCLEAR FIX: Simple loss without anti-collapse
        'simple_directional_mse': SimpleDirectionalMSE,  # Alias
        'ResidualAwareLoss': ResidualAwareLoss,  # NUCLEAR FIX v5.0: Simplified loss for residual prediction
        'residual_aware_loss': ResidualAwareLoss,  # Alias
        'focal_loss': focal_loss,
        'focal_loss_multiclass': focal_loss_multiclass,
        'FocalLossWithAlpha': FocalLossWithAlpha,
        'focal_loss_with_alpha': FocalLossWithAlpha,  # Alias for backward compatibility
        'BinaryFocalLoss': BinaryFocalLoss,  # NEW: Serializable binary focal loss
        'binary_focal_loss': BinaryFocalLoss,  # Alias for backward compatibility
        'huber_with_variance_penalty': huber_with_variance_penalty,  # NEW: Prevents zero predictions
        'r2_metric': r2_metric,  # NEW: R² metric for regression monitoring
        'r_squared_metric': r2_metric,  # Alias for r2_metric
        'directional_accuracy_metric': directional_accuracy_metric,  # NEW: Directional accuracy metric
        'VarianceRegularizedLoss': VarianceRegularizedLoss,  # NEW: Variance regularization wrapper
        **paper_losses,  # Add losses from lstm_transformer_paper
    }


def register_custom_objects():
    """
    Register custom objects globally with TensorFlow/Keras
    
    Call this once at the start of your script to enable automatic
    detection of custom losses when loading models.
    
    Usage:
        from utils.losses import register_custom_objects
        register_custom_objects()
        
        # Now you can load models without specifying custom_objects
        model = tf.keras.models.load_model('path/to/model.keras')
    """
    custom_objects = get_custom_objects()
    for name, obj in custom_objects.items():
        tf.keras.utils.get_custom_objects()[name] = obj
    
    print(f"[OK] Registered custom objects: {list(custom_objects.keys())}")
