"""
PatchTST: A Time Series is Worth 64 Words - Channel-Independent Patch Transformer

Implementation based on:
    Nie, Y., Nguyen, N. H., Sinthong, P., & Kalagnanam, J. (2023).
    "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
    ICLR 2023
    
Key innovations from the paper:
1. **Patching** - Segments time series into subseries-level patches
   - Reduces sequence length by patch_len factor
   - Provides local semantic information per patch
   - Enables longer look-back with manageable sequence length
   
2. **Channel Independence** - Each feature/channel processed independently
   - Reduces number of parameters drastically
   - Prevents cross-channel noise interference
   - Better generalizes to unseen time series
   
3. **Reversible Instance Normalization** - RevIN
   - Removes and restores distribution shift
   - Handles non-stationary financial data
   - Critical for handling different market regimes

Architecture:
    Input [batch, seq_len, n_features]
      ↓
    RevIN (normalize per channel)
      ↓
    Patching [batch * n_features, n_patches, patch_len]
      ↓
    Linear Projection to d_model
      ↓
    + Learnable Position Encoding
      ↓
    Transformer Encoder × n_layers
      ↓
    Flatten patches
      ↓
    Linear Head → [batch, pred_len]
      ↓
    RevIN (denormalize)
      ↓
    Output

This implementation is optimized for:
    - Financial time series forecasting
    - Multi-day horizon prediction
    - GPU batch processing (batch_size >= 512)
    - Keras 3 with PyTorch backend
"""

from __future__ import annotations

import numpy as np

# Keras 3 backend-agnostic imports
try:
    import keras
    from keras import Model, layers, ops
    K = ops
    KERAS_3 = True
except ImportError:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import Model, layers
    K = tf
    KERAS_3 = False


# =============================================================================
# REVERSIBLE INSTANCE NORMALIZATION (RevIN)
# =============================================================================

class RevIN(layers.Layer):
    """
    Reversible Instance Normalization for handling distribution shift.
    
    From: "Reversible Instance Normalization for Accurate Time-Series Forecasting"
    Kim et al., ICLR 2022
    
    RevIN normalizes input by removing instance-specific mean and std,
    then restores them on the output. This handles non-stationarity in
    financial time series where different market regimes have different
    statistical properties.
    
    Args:
        num_features: Number of input features/channels
        eps: Epsilon for numerical stability
        affine: Whether to learn scale/shift parameters
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        affine: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        
    def build(self, input_shape):
        if self.affine:
            # Learnable affine parameters per feature
            self.affine_weight = self.add_weight(
                name='affine_weight',
                shape=(self.num_features,),
                initializer='ones',
                trainable=True
            )
            self.affine_bias = self.add_weight(
                name='affine_bias',
                shape=(self.num_features,),
                initializer='zeros',
                trainable=True
            )
        super().build(input_shape)
        
    def call(self, x, mode: str = 'norm'):
        """
        Apply or reverse normalization.
        
        Args:
            x: Input tensor [batch, seq_len, n_features]
            mode: 'norm' to normalize, 'denorm' to denormalize
            
        Returns:
            Normalized or denormalized tensor
        """
        if mode == 'norm':
            # Compute instance statistics across time dimension
            self._mean = ops.mean(x, axis=1, keepdims=True)  # [B, 1, C]
            self._std = ops.sqrt(ops.var(x, axis=1, keepdims=True) + self.eps)  # [B, 1, C]
            
            # Normalize
            x = (x - self._mean) / self._std
            
            # Apply learnable affine transform
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
                
            return x
            
        elif mode == 'denorm':
            # Reverse affine transform
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps)
            
            # Restore original statistics
            x = x * self._std + self._mean
            
            return x
        else:
            raise ValueError(f"mode must be 'norm' or 'denorm', got {mode}")
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'num_features': self.num_features,
            'eps': self.eps,
            'affine': self.affine,
        })
        return config


# =============================================================================
# PATCH EMBEDDING
# =============================================================================

class PatchEmbedding(layers.Layer):
    """
    Patch embedding layer that segments time series into patches.
    
    This is the key innovation from PatchTST - treating time series
    segments as "tokens" similar to patches in Vision Transformer.
    
    For input shape [batch, seq_len, n_features]:
    1. Reshape to [batch * n_features, seq_len, 1] (channel independence)
    2. Create patches [batch * n_features, n_patches, patch_len]
    3. Project to d_model: [batch * n_features, n_patches, d_model]
    
    Args:
        patch_len: Length of each patch
        stride: Stride between patches (default: patch_len // 2 for overlap)
        d_model: Transformer model dimension
        n_features: Number of input features
    """
    
    def __init__(
        self,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_features: int = 147,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_features = n_features
        
    def build(self, input_shape):
        # Linear projection from patch_len to d_model
        self.projection = layers.Dense(
            self.d_model,
            use_bias=True,
            kernel_initializer='glorot_uniform',
            name='patch_projection'
        )
        super().build(input_shape)
        
    def call(self, x):
        """
        Create patch embeddings from input sequence.
        
        Args:
            x: Input [batch, seq_len, n_features]
            
        Returns:
            Patch embeddings [batch * n_features, n_patches, d_model]
        """
        batch_size = ops.shape(x)[0]
        seq_len = ops.shape(x)[1]
        
        # Compute number of patches
        n_patches = (seq_len - self.patch_len) // self.stride + 1
        
        # Transpose for channel independence: [batch, n_features, seq_len]
        x = ops.transpose(x, [0, 2, 1])
        
        # Reshape to treat each channel separately: [batch * n_features, seq_len]
        x = ops.reshape(x, [batch_size * self.n_features, seq_len])
        
        # Extract patches using unfold-like operation
        # Create indices for patch extraction
        patch_indices = ops.arange(n_patches) * self.stride
        
        # Extract patches manually (Keras 3 compatible)
        patches = []
        for i in range(n_patches):
            start_idx = i * self.stride
            patch = x[:, start_idx:start_idx + self.patch_len]
            patches.append(patch)
        
        # Stack patches: [batch * n_features, n_patches, patch_len]
        x = ops.stack(patches, axis=1)
        
        # Project to d_model
        x = self.projection(x)  # [batch * n_features, n_patches, d_model]
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'patch_len': self.patch_len,
            'stride': self.stride,
            'd_model': self.d_model,
            'n_features': self.n_features,
        })
        return config


# =============================================================================
# POSITIONAL ENCODING
# =============================================================================

class LearnablePositionalEncoding(layers.Layer):
    """
    Learnable positional encoding for patches.
    
    Unlike fixed sinusoidal encoding, learnable positions allow the model
    to discover optimal position representations for financial data.
    
    Args:
        max_patches: Maximum number of patches
        d_model: Model dimension
    """
    
    def __init__(
        self,
        max_patches: int = 64,
        d_model: int = 64,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.max_patches = max_patches
        self.d_model = d_model
        
    def build(self, input_shape):
        self.pos_embedding = self.add_weight(
            name='pos_embedding',
            shape=(1, self.max_patches, self.d_model),
            initializer='truncated_normal',
            trainable=True
        )
        super().build(input_shape)
        
    def call(self, x):
        """
        Add positional encoding to patch embeddings.
        
        Args:
            x: [batch * n_features, n_patches, d_model]
            
        Returns:
            Position-encoded tensor
        """
        n_patches = ops.shape(x)[1]
        return x + self.pos_embedding[:, :n_patches, :]
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'max_patches': self.max_patches,
            'd_model': self.d_model,
        })
        return config


# =============================================================================
# TRANSFORMER ENCODER
# =============================================================================

class TransformerEncoderLayer(layers.Layer):
    """
    Standard Transformer encoder layer with pre-norm architecture.
    
    Pre-norm (LayerNorm before attention/FFN) is more stable for
    training deep transformers without learning rate warmup.
    
    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        d_ff: Feed-forward dimension (default: 4 * d_model)
        dropout: Dropout rate
    """
    
    def __init__(
        self,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.dropout_rate = dropout
        
    def build(self, input_shape):
        # Pre-norm layers
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        
        # Multi-head attention
        self.attention = layers.MultiHeadAttention(
            num_heads=self.n_heads,
            key_dim=self.d_model // self.n_heads,
            dropout=self.dropout_rate,
        )
        
        # Feed-forward network
        self.ffn = keras.Sequential([
            layers.Dense(self.d_ff, activation='gelu'),
            layers.Dropout(self.dropout_rate),
            layers.Dense(self.d_model),
            layers.Dropout(self.dropout_rate),
        ])
        
        self.dropout = layers.Dropout(self.dropout_rate)
        
        super().build(input_shape)
        
    def call(self, x, training=None):
        """
        Forward pass through transformer encoder layer.
        
        Args:
            x: [batch, seq_len, d_model]
            training: Boolean for dropout
            
        Returns:
            Encoded tensor
        """
        # Pre-norm attention
        attn_input = self.norm1(x)
        attn_output = self.attention(
            attn_input, attn_input,
            training=training
        )
        x = x + self.dropout(attn_output, training=training)
        
        # Pre-norm feed-forward
        ffn_input = self.norm2(x)
        ffn_output = self.ffn(ffn_input, training=training)
        x = x + ffn_output
        
        return x
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'd_ff': self.d_ff,
            'dropout': self.dropout_rate,
        })
        return config


# =============================================================================
# PATCHTST MODEL
# =============================================================================

class PatchTST(Model):
    """
    PatchTST: Patch Time Series Transformer for financial forecasting.
    
    This model implements the full PatchTST architecture optimized for
    financial time series prediction with multi-day horizon support.
    
    Key hyperparameters from paper ablation studies:
    - patch_len: 16 works well for daily data
    - stride: patch_len // 2 for 50% overlap
    - d_model: 64-128 sufficient for most tasks
    - n_layers: 3-4 layers optimal
    - n_heads: 4-8 heads
    
    Args:
        seq_len: Input sequence length (lookback window)
        pred_len: Prediction horizon length
        n_features: Number of input features/channels
        patch_len: Patch length (default: 16)
        stride: Patch stride (default: 8)
        d_model: Transformer dimension (default: 64)
        n_heads: Number of attention heads (default: 4)
        n_layers: Number of transformer layers (default: 3)
        d_ff: Feed-forward dimension (default: 256)
        dropout: Dropout rate (default: 0.1)
        use_revin: Whether to use RevIN (default: True)
    
    Example:
        >>> model = PatchTST(
        ...     seq_len=60,
        ...     pred_len=1,
        ...     n_features=147,
        ...     patch_len=16,
        ...     d_model=64,
        ...     n_layers=3,
        ... )
        >>> # Input: [batch, 60, 147]
        >>> # Output: [batch, 1]
    """
    
    def __init__(
        self,
        seq_len: int = 60,
        pred_len: int = 1,
        n_features: int = 147,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 256,
        dropout: float = 0.1,
        use_revin: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.dropout_rate = dropout
        self.use_revin = use_revin
        
        # Calculate number of patches
        self.n_patches = (seq_len - patch_len) // stride + 1
        
        # RevIN for distribution normalization
        if use_revin:
            self.revin = RevIN(n_features)
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            patch_len=patch_len,
            stride=stride,
            d_model=d_model,
            n_features=n_features,
        )
        
        # Positional encoding
        self.pos_encoding = LearnablePositionalEncoding(
            max_patches=self.n_patches + 10,  # Buffer for variable lengths
            d_model=d_model,
        )
        
        # Transformer encoder layers
        self.encoder_layers = [
            TransformerEncoderLayer(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                dropout=dropout,
                name=f'encoder_layer_{i}'
            )
            for i in range(n_layers)
        ]
        
        # Final layer norm
        self.final_norm = layers.LayerNormalization(epsilon=1e-6)
        
        # Dropout
        self.dropout = layers.Dropout(dropout)
        
        # Flatten and head
        # Output: from [batch * n_features, n_patches, d_model] -> [batch, pred_len]
        self.flatten = layers.Flatten()
        
        # Head: project flattened representation to prediction
        # Individual head per channel, then aggregate
        head_input_dim = self.n_patches * d_model
        self.prediction_head = keras.Sequential([
            layers.InputLayer(shape=(head_input_dim,)),
            layers.Dense(128, activation='gelu'),
            layers.Dropout(dropout),
            layers.Dense(pred_len),
        ], name='prediction_head')
        
    def call(self, x, training=None):
        """
        Forward pass through PatchTST.
        
        Args:
            x: Input tensor [batch, seq_len, n_features]
            training: Boolean for dropout/training mode
            
        Returns:
            Predictions [batch, pred_len]
        """
        batch_size = ops.shape(x)[0]
        
        # 1. RevIN normalize
        if self.use_revin:
            x = self.revin(x, mode='norm')
        
        # 2. Patch embedding
        # [batch, seq_len, n_features] -> [batch * n_features, n_patches, d_model]
        x = self.patch_embedding(x)
        
        # 3. Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout(x, training=training)
        
        # 4. Transformer encoder
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x, training=training)
        
        x = self.final_norm(x)
        
        # 5. Reshape back: [batch * n_features, n_patches, d_model] -> [batch, n_features, n_patches * d_model]
        x = self.flatten(x)  # [batch * n_features, n_patches * d_model]
        x = ops.reshape(x, [batch_size, self.n_features, -1])
        
        # 6. Aggregate across channels (mean pooling for channel independence)
        x = ops.mean(x, axis=1)  # [batch, n_patches * d_model]
        
        # 7. Prediction head
        output = self.prediction_head(x, training=training)  # [batch, pred_len]
        
        # 8. RevIN denormalize (for the target channel if applicable)
        # Note: For return prediction, we skip denorm as target is already standardized
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'n_features': self.n_features,
            'patch_len': self.patch_len,
            'stride': self.stride,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'd_ff': self.d_ff,
            'dropout': self.dropout_rate,
            'use_revin': self.use_revin,
        })
        return config


# =============================================================================
# MODEL FACTORY FUNCTIONS
# =============================================================================

def create_patchtst_model(
    seq_len: int = 60,
    pred_len: int = 1,
    n_features: int = 147,
    patch_len: int = 16,
    stride: int = 8,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 3,
    d_ff: int = 256,
    dropout: float = 0.1,
    use_revin: bool = True,
    learning_rate: float = 1e-4,
    loss: str = 'huber',
) -> Model:
    """
    Create and compile a PatchTST model.
    
    This factory function creates a PatchTST model with optimal defaults
    for financial time series forecasting.
    
    Recommended configurations:
    
    1. SMALL (fast training, good baseline):
        - d_model=64, n_layers=2, n_heads=4
        - ~400K parameters
        
    2. MEDIUM (balanced):
        - d_model=128, n_layers=3, n_heads=4
        - ~1.5M parameters
        
    3. LARGE (maximum capacity):
        - d_model=256, n_layers=4, n_heads=8
        - ~5M parameters
    
    Args:
        seq_len: Input sequence length
        pred_len: Prediction horizon
        n_features: Number of input features
        patch_len: Patch length
        stride: Patch stride
        d_model: Transformer dimension
        n_heads: Number of attention heads
        n_layers: Number of transformer layers
        d_ff: Feed-forward dimension
        dropout: Dropout rate
        use_revin: Whether to use RevIN
        learning_rate: Learning rate for optimizer
        loss: Loss function ('huber', 'mse', 'mae')
        
    Returns:
        Compiled Keras model
    """
    model = PatchTST(
        seq_len=seq_len,
        pred_len=pred_len,
        n_features=n_features,
        patch_len=patch_len,
        stride=stride,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        use_revin=use_revin,
    )
    
    # Build model
    model.build(input_shape=(None, seq_len, n_features))
    
    # Select loss function
    if loss == 'huber':
        loss_fn = keras.losses.Huber(delta=1.0)
    elif loss == 'mse':
        loss_fn = keras.losses.MeanSquaredError()
    elif loss == 'mae':
        loss_fn = keras.losses.MeanAbsoluteError()
    else:
        loss_fn = loss  # Custom loss passed directly
    
    # Compile
    model.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=learning_rate,
            weight_decay=0.01,
        ),
        loss=loss_fn,
        metrics=['mae'],
    )
    
    return model


def create_patchtst_small(n_features: int = 147, seq_len: int = 60, **kwargs) -> Model:
    """Create small PatchTST (~400K params) for fast iteration."""
    return create_patchtst_model(
        seq_len=seq_len,
        n_features=n_features,
        d_model=64,
        n_layers=2,
        n_heads=4,
        d_ff=256,
        **kwargs
    )


def create_patchtst_medium(n_features: int = 147, seq_len: int = 60, **kwargs) -> Model:
    """Create medium PatchTST (~1.5M params) for balanced performance."""
    return create_patchtst_model(
        seq_len=seq_len,
        n_features=n_features,
        d_model=128,
        n_layers=3,
        n_heads=4,
        d_ff=512,
        **kwargs
    )


def create_patchtst_large(n_features: int = 147, seq_len: int = 60, **kwargs) -> Model:
    """Create large PatchTST (~5M params) for maximum capacity."""
    return create_patchtst_model(
        seq_len=seq_len,
        n_features=n_features,
        d_model=256,
        n_layers=4,
        n_heads=8,
        d_ff=1024,
        **kwargs
    )


# =============================================================================
# ENSEMBLE SUPPORT
# =============================================================================

class PatchTSTEnsemble(Model):
    """
    Ensemble of PatchTST models with different hyperparameters.
    
    Combines predictions from multiple PatchTST variants for
    improved robustness and uncertainty estimation.
    
    Args:
        models: List of PatchTST models or configs
        ensemble_method: 'mean', 'median', or 'learned'
    """
    
    def __init__(
        self,
        seq_len: int = 60,
        pred_len: int = 1,
        n_features: int = 147,
        ensemble_method: str = 'mean',
        n_models: int = 3,
        **kwargs
    ):
        super().__init__(**kwargs)
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_features = n_features
        self.ensemble_method = ensemble_method
        self.n_models = n_models
        
        # Create diverse ensemble members
        self.models = []
        
        # Model 1: Small with short patches
        self.models.append(PatchTST(
            seq_len=seq_len, pred_len=pred_len, n_features=n_features,
            patch_len=8, stride=4, d_model=64, n_layers=2, n_heads=4,
            name='patchtst_small'
        ))
        
        # Model 2: Medium with standard patches
        self.models.append(PatchTST(
            seq_len=seq_len, pred_len=pred_len, n_features=n_features,
            patch_len=16, stride=8, d_model=96, n_layers=3, n_heads=4,
            name='patchtst_medium'
        ))
        
        # Model 3: Large with long patches
        if n_models >= 3:
            self.models.append(PatchTST(
                seq_len=seq_len, pred_len=pred_len, n_features=n_features,
                patch_len=20, stride=10, d_model=128, n_layers=3, n_heads=4,
                name='patchtst_large'
            ))
        
        if ensemble_method == 'learned':
            self.ensemble_weights = self.add_weight(
                name='ensemble_weights',
                shape=(len(self.models),),
                initializer='ones',
                trainable=True
            )
            
    def call(self, x, training=None):
        """
        Forward pass through ensemble.
        
        Args:
            x: Input [batch, seq_len, n_features]
            training: Boolean
            
        Returns:
            Ensemble prediction [batch, pred_len]
        """
        predictions = [model(x, training=training) for model in self.models]
        predictions = ops.stack(predictions, axis=-1)  # [batch, pred_len, n_models]
        
        if self.ensemble_method == 'mean':
            return ops.mean(predictions, axis=-1)
        elif self.ensemble_method == 'median':
            # Note: median not directly available in all backends
            return ops.mean(predictions, axis=-1)  # Fallback to mean
        elif self.ensemble_method == 'learned':
            weights = ops.softmax(self.ensemble_weights)
            return ops.sum(predictions * weights, axis=-1)
        else:
            return ops.mean(predictions, axis=-1)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'seq_len': self.seq_len,
            'pred_len': self.pred_len,
            'n_features': self.n_features,
            'ensemble_method': self.ensemble_method,
            'n_models': self.n_models,
        })
        return config


# =============================================================================
# TESTING / VALIDATION
# =============================================================================

if __name__ == '__main__':
    import os
    os.environ['KERAS_BACKEND'] = 'torch'
    
    print("=" * 60)
    print("PatchTST Model Test")
    print("=" * 60)
    
    # Test parameters
    batch_size = 512
    seq_len = 60
    n_features = 147
    pred_len = 1
    
    print(f"\nTest configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Number of features: {n_features}")
    print(f"  Prediction length: {pred_len}")
    
    # Create model
    print("\nCreating PatchTST model...")
    model = create_patchtst_model(
        seq_len=seq_len,
        pred_len=pred_len,
        n_features=n_features,
        patch_len=16,
        stride=8,
        d_model=64,
        n_layers=3,
        n_heads=4,
    )
    
    # Print summary
    print("\nModel summary:")
    model.summary()
    
    # Count parameters
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    
    # Test forward pass
    print("\nTesting forward pass...")
    x = np.random.randn(batch_size, seq_len, n_features).astype(np.float32)
    y = np.random.randn(batch_size, pred_len).astype(np.float32)
    
    # Predict
    pred = model.predict(x, verbose=0)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {pred.shape}")
    print(f"Output mean: {pred.mean():.6f}")
    print(f"Output std: {pred.std():.6f}")
    
    # Test training step
    print("\nTesting training step...")
    history = model.fit(x, y, epochs=1, batch_size=batch_size, verbose=1)
    print(f"Training loss: {history.history['loss'][0]:.6f}")
    
    print("\n" + "=" * 60)
    print("PatchTST test PASSED!")
    print("=" * 60)
