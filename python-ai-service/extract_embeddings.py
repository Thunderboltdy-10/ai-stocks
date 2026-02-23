#!/usr/bin/env python3
"""
Extract Embeddings from Trained LSTM-Transformer Model

This script extracts the learned embeddings (from the global_pool layer) 
from a trained stock prediction model. These embeddings can be used for:
- Visualizing learned representations (t-SNE, UMAP)
- Detecting prediction collapse patterns
- Analyzing model behavior over time

Usage:
    python extract_embeddings.py              # Extract for AAPL (default)
    python extract_embeddings.py TSLA         # Extract for specific symbol
    python extract_embeddings.py AAPL TSLA    # Extract for multiple symbols
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import pickle
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Allow expired cache by default - we want to use cached data for embedding extraction
# rather than fetching new data (which may fail due to API issues)
os.environ.setdefault('ALLOW_EXPIRED_CACHE', 'true')

# Import data utilities
from data.cache_manager import DataCacheManager
from data.feature_engineer import get_feature_columns

# Import model architecture
from models.lstm_transformer_paper import (
    LSTMTransformerPaper,
    DirectionalHuberLoss,
    AntiCollapseDirectionalLoss,
    directional_accuracy_metric
)

# Import and register all custom loss functions for model loading
from utils.losses import register_custom_objects, get_custom_objects
register_custom_objects()


def extract(symbol: str = 'AAPL', sequence_length: int = 90) -> bool:
    """
    Extract embeddings for a given stock symbol.
    
    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'TSLA')
        sequence_length: Length of input sequences (must match training)
    
    Returns:
        True if successful, False otherwise
    """
    print(f"\n{'='*50}")
    print(f"--- Extracting Embeddings for {symbol} ---")
    print(f"{'='*50}")
    
    # 1. Load the Data (exactly as training did)
    print("\n[1/5] Loading Data...")
    cache = DataCacheManager()
    
    try:
        _, _, df_clean, feature_cols = cache.get_or_fetch_data(symbol, include_sentiment=True)
    except Exception as e:
        print(f"   ERROR: Failed to load data for {symbol}: {e}")
        return False
    
    print(f"   Loaded {len(df_clean)} rows with {len(feature_cols)} features")
    
    # 2. Scale Features (Must use same scaler as training)
    print("\n[2/5] Scaling Features...")
    scaler_path = f"saved_models/{symbol}/regressor/feature_scaler.pkl"
    
    if not os.path.exists(scaler_path):
        print(f"   ERROR: No scaler found at {scaler_path}")
        print(f"   Make sure you have trained a model for {symbol} first.")
        return False

    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    # Extract and scale features
    X_raw = df_clean[feature_cols].values
    X_scaled = scaler.transform(X_raw)
    
    # Create sequences
    X_seq = []
    for i in range(len(X_scaled) - sequence_length + 1):
        X_seq.append(X_scaled[i : i + sequence_length])
    
    X_seq = np.array(X_seq, dtype=np.float32)
    timestamps = df_clean.index[sequence_length - 1 : sequence_length - 1 + len(X_seq)]
    
    print(f"   Created {len(X_seq)} sequences of length {sequence_length}")

    # 3. Load the Trained Model
    print("\n[3/5] Loading Model...")
    model_path = f"saved_models/{symbol}/regressor/model.keras"
    
    if not os.path.exists(model_path):
        print(f"   ERROR: No model found at {model_path}")
        print(f"   Make sure you have trained a model for {symbol} first.")
        return False
    
    # Build custom objects dict with all registered losses + model classes
    custom_objects = get_custom_objects()
    custom_objects.update({
        'LSTMTransformerPaper': LSTMTransformerPaper,
        'DirectionalHuberLoss': DirectionalHuberLoss,
        'AntiCollapseDirectionalLoss': AntiCollapseDirectionalLoss,
        'directional_accuracy_metric': directional_accuracy_metric
    })
    
    try:
        model = keras.models.load_model(model_path, custom_objects=custom_objects)
        print("   Model loaded successfully.")
    except Exception as e:
        print(f"   ERROR: Failed to load model: {e}")
        return False

    # 4. Create Embedding Extractor
    # Extract output from 'global_pool' layer (the embedding) AND final predictions
    print("\n[4/5] Setting up embedding extractor...")
    
    try:
        embedding_layer = model.get_layer('global_pool')
    except ValueError:
        print("   ERROR: Could not find 'global_pool' layer.")
        print("   Available layers:")
        for layer in model.layers:
            print(f"     - {layer.name}")
        return False

    extractor = keras.Model(
        inputs=model.inputs,
        outputs=[embedding_layer.output, model.output]
    )
    
    # 5. Extract Embeddings
    print("\n[5/5] Computing Embeddings...")
    embeddings, predictions = extractor.predict(X_seq, batch_size=512, verbose=1)
    
    # Handle multi-head output if present
    if isinstance(predictions, list):
        predictions = predictions[0]  # Take magnitude output
    
    predictions = predictions.flatten()

    # 6. Save Results
    print("\n[6/6] Saving Results...")
    
    output_dir = Path(f"saved_models/{symbol}/embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create metadata DataFrame
    metadata = pd.DataFrame({
        'index': range(len(embeddings)),
        'date': timestamps,
        'prediction': predictions,
        'is_collapsed': np.abs(predictions) < 0.001  # Flag near-zero predictions
    })
    
    # Save embeddings as compressed numpy archive
    np.savez_compressed(
        output_dir / 'embeddings.npz',
        embeddings=embeddings,
        predictions=predictions,
        dates=timestamps.values.astype(str)
    )
    
    # Save metadata as CSV for easy inspection
    metadata.to_csv(output_dir / 'metadata.csv', index=False)
    
    # Print summary
    print(f"\n{'='*50}")
    print("SUCCESS!")
    print(f"{'='*50}")
    print(f"  Symbol:           {symbol}")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Date range:       {timestamps.min().date()} to {timestamps.max().date()}")
    print(f"  Collapsed preds:  {(np.abs(predictions) < 0.001).sum()} / {len(predictions)}")
    print(f"\nOutput files:")
    print(f"  1. {output_dir}/embeddings.npz")
    print(f"  2. {output_dir}/metadata.csv")
    print(f"\nNow you can run visualization scripts on these embeddings.")
    
    return True


def main():
    """Main entry point."""
    # Parse command line arguments
    if len(sys.argv) > 1:
        symbols = [s.upper() for s in sys.argv[1:]]
    else:
        symbols = ['AAPL']
    
    print(f"Extracting embeddings for: {', '.join(symbols)}")
    
    # Process each symbol
    results = {}
    for symbol in symbols:
        success = extract(symbol)
        results[symbol] = 'SUCCESS' if success else 'FAILED'
    
    # Print summary if multiple symbols
    if len(symbols) > 1:
        print(f"\n{'='*50}")
        print("SUMMARY")
        print(f"{'='*50}")
        for symbol, status in results.items():
            print(f"  {symbol}: {status}")


if __name__ == "__main__":
    main()
