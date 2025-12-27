import yfinance as yf
import argparse
import sys
import os
import json
import random
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime

# ============================================================================
# GPU & ACCELERATION CONFIGURATION
# ============================================================================
def setup_gpu():
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

setup_gpu()

import tensorflow as tf
from tensorflow import keras

# Import project modules
sys.path.append(str(Path(__file__).parent.parent))
from data.cache_manager import DataCacheManager
from data.target_engineering import prepare_training_data
from models.lstm_transformer_paper import LSTMTransformerPaper
from models.patchtst import create_patchtst_model

# Enable Mixed Precision and XLA AFTER TF import
def configure_tf():
    try:
        from tensorflow.keras import mixed_precision
        policy = mixed_precision.Policy('mixed_float16')
        mixed_precision.set_global_policy(policy)
        print(f"[GPU] Mixed precision enabled: {policy.name}")
    except Exception as e:
        print(f"[GPU] Warning: Mixed precision failed: {e}")
    
    os.environ['TF_XLA_FLAGS'] = '--tf_xla_auto_jit=2 --tf_xla_cpu_global_jit'

configure_tf()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(model_type, seq_len, n_features, learning_rate=1e-4):
    if model_type == 'patchtst':
        print(f"[Model] Creating PatchTST (seq={seq_len}, feat={n_features})")
        return create_patchtst_model(
            seq_len=seq_len,
            n_features=n_features,
            d_model=128,
            n_layers=3,
            n_heads=4,
            learning_rate=learning_rate
        )
    else:
        print(f"[Model] Creating LSTM-Transformer (seq={seq_len}, feat={n_features})")
        # Use a simplified multi-task wrap or single task for smart target
        base = LSTMTransformerPaper(
            sequence_length=seq_len,
            n_features=n_features,
            lstm_units=128,
            d_model=128,
            num_blocks=4,
            ff_dim=256
        )
        inputs = keras.Input(shape=(seq_len, n_features))
        outputs = base(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(
            optimizer=keras.optimizers.AdamW(learning_rate=learning_rate),
            loss='huber',
            metrics=['mae']
        )
        return model

# ============================================================================
# TRAINING LOGIC
# ============================================================================

def train_smart_model(symbol, model_type='lstm_transformer', epochs=50, batch_size=256, seq_len=90):
    set_seed(42)
    cache = DataCacheManager()
    
    print(f"--- Training {model_type.upper()} for {symbol} (Smart Target) ---")
    
    # 1. Load Data
    # get_or_fetch_data returns (raw, engineered, prepared, features)
    _, df, _, _ = cache.get_or_fetch_data(symbol)
    if df is None:
        raise ValueError(f"No data for {symbol}")
        
    # 2. Prepare Smart Target
    df_clean, features = prepare_training_data(df, use_smart_target=True)
    n_features = len(features)
    
    # 3. Create Sequences
    X, y = [], []
    data_values = df_clean[features].values
    target_values = df_clean['target_smart_scaled'].values
    
    for i in range(seq_len, len(df_clean)):
        X.append(data_values[i-seq_len:i])
        y.append(target_values[i])
        
    X, y = np.array(X), np.array(y)
    
    # Split
    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]
    
    # 4. Create Model
    model = create_model(model_type, seq_len, n_features)
    
    # 5. Train
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
    ]
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    # 6. Save Artifacts
    save_dir = Path(f"models/smart_{symbol}_{model_type}")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model.save(save_dir / "model.keras")
    
    # Save target scaler (Pickle)
    if 'target_scaler' in df_clean.attrs:
        import pickle
        with open(save_dir / "target_scaler.pkl", 'wb') as f:
            pickle.dump(df_clean.attrs['target_scaler'], f)
        print(f"   [OK] Saved target scaler to {save_dir}/target_scaler.pkl")
    
    metadata = {
        'symbol': symbol,
        'model_type': model_type,
        'features': features,
        'target_col': 'target_smart',
        'seq_len': seq_len,
        'timestamp': datetime.now().isoformat()
    }
    
    with open(save_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=4)
        
    print(f"--- Training Complete: {save_dir} ---")
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('symbol', type=str)
    parser.add_argument('--model-type', type=str, default='lstm_transformer', choices=['lstm_transformer', 'patchtst'])
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    
    train_smart_model(args.symbol, args.model_type, args.epochs)
