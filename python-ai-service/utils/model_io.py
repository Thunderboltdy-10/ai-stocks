"""
Model Loading/Saving Utilities with Custom Objects Support

Centralizes model I/O operations to ensure custom losses are properly
handled across training, inference, and server code.
"""

import os
from pathlib import Path
from tensorflow import keras
from utils.losses import get_custom_objects, register_custom_objects


def save_model_with_metadata(model, filepath, metadata=None):
    """
    Save model with optional metadata
    
    Args:
        model: Keras model to save
        filepath: Path to save model (e.g., 'saved_models/AAPL_conservative_classifier.keras')
        metadata: Optional dict of metadata to save alongside model
    
    Returns:
        Path where model was saved
    """
    # Ensure directory exists
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    
    # Save model (Keras will detect custom objects from compilation)
    model.save(filepath)
    print(f"   ✅ Saved model: {filepath}")
    
    # Save metadata if provided
    if metadata:
        import joblib
        metadata_path = filepath.replace('.keras', '_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        print(f"   ✅ Saved metadata: {metadata_path}")
    
    return filepath


def load_model_safe(filepath, custom_objects=None, verbose=True):
    """
    Safely load Keras model with custom objects support
    
    Args:
        filepath: Path to model file
        custom_objects: Optional dict of custom objects (auto-detected if None)
        verbose: Print loading information
    
    Returns:
        Loaded Keras model
    
    Raises:
        FileNotFoundError: If model file doesn't exist
        Exception: If loading fails
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model not found: {filepath}")
    
    # Use provided custom objects or get defaults
    if custom_objects is None:
        custom_objects = get_custom_objects()
    
    if verbose:
        print(f"📂 Loading model: {filepath}")
        print(f"   Custom objects: {list(custom_objects.keys())}")
    
    try:
        model = keras.models.load_model(filepath, custom_objects=custom_objects)
        
        if verbose:
            print(f"   ✅ Loaded successfully")
            print(f"   Model: {model.name}")
            print(f"   Parameters: {model.count_params():,}")
            print(f"   Layers: {len(model.layers)}")
            
            # Check if model uses custom losses
            if hasattr(model, 'loss') and model.loss:
                loss_name = model.loss.__name__ if hasattr(model.loss, '__name__') else str(model.loss)
                print(f"   Loss: {loss_name}")
        
        return model
    
    except Exception as e:
        print(f"   ❌ Failed to load model: {e}")
        raise


def load_model_artifacts(symbol, risk_profile, base_path=None, verbose=True):
    """
    Load all model artifacts for a symbol and risk profile
    
    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        risk_profile: 'conservative' or 'aggressive'
        base_path: Base directory containing saved models
        verbose: Print loading information
    
    Returns:
        dict with keys: 'classifier', 'regressor', 'scaler', 'features', 'metadata'
    
    Raises:
        FileNotFoundError: If any required file is missing
    """
    import joblib
    
    if base_path is None:
        base_path = Path(__file__).resolve().parents[1] / "saved_models"
    else:
        base_path = Path(base_path)
    prefix = f"{symbol}_{risk_profile}"
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"Loading Model Artifacts: {symbol} ({risk_profile.upper()})")
        print(f"{'='*70}\n")
    
    artifacts = {}
    
    # Load classifier (risk-profile specific)
    classifier_path = base_path / f"{prefix}_classifier.keras"
    artifacts['classifier'] = load_model_safe(classifier_path, verbose=verbose)
    
    # Load regressor (SHARED across risk profiles)
    # Try shared path first, fall back to old per-profile path for backwards compatibility
    regressor_shared_path = base_path / f"{symbol}_regressor.keras"
    regressor_profile_path = base_path / f"{prefix}_regressor.keras"
    
    if regressor_shared_path.exists():
        if verbose:
            print(f"\n📂 Loading shared regressor: {regressor_shared_path}")
        artifacts['regressor'] = load_model_safe(regressor_shared_path, verbose=verbose)
    elif regressor_profile_path.exists():
        if verbose:
            print(f"\n📂 Loading regressor (old format): {regressor_profile_path}")
            print(f"   ⚠️  NOTE: Using old per-profile regressor. Retrain to use shared model.")
        artifacts['regressor'] = load_model_safe(regressor_profile_path, verbose=verbose)
    else:
        raise FileNotFoundError(
            f"Regressor not found. Tried:\n"
            f"  - {regressor_shared_path} (shared)\n"
            f"  - {regressor_profile_path} (old format)"
        )
    
    # Load scaler
    scaler_path = base_path / f"{prefix}_scaler.pkl"
    if verbose:
        print(f"\n📂 Loading scaler: {scaler_path}")
    artifacts['scaler'] = joblib.load(scaler_path)
    if verbose:
        print(f"   ✅ Loaded successfully")
    
    # Load features
    features_path = base_path / f"{prefix}_features.pkl"
    if verbose:
        print(f"\n📂 Loading features: {features_path}")
    artifacts['features'] = joblib.load(features_path)
    if verbose:
        print(f"   ✅ Loaded {len(artifacts['features'])} features")
    
    # Load metadata
    metadata_path = base_path / f"{prefix}_metadata.pkl"
    if verbose:
        print(f"\n📂 Loading metadata: {metadata_path}")
    artifacts['metadata'] = joblib.load(metadata_path)
    if verbose:
        print(f"   ✅ Loaded metadata")
        print(f"   Sequence length: {artifacts['metadata'].get('sequence_length', 'N/A')}")
        print(f"   Features: {artifacts['metadata'].get('n_features', 'N/A')}")
        print(f"   Threshold multiplier: {artifacts['metadata'].get('threshold_multiplier', 'N/A')}")
        if artifacts['metadata'].get('shared_regression'):
            print(f"   Regression model: SHARED (same for conservative & aggressive)")
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"✅ All artifacts loaded successfully")
        print(f"{'='*70}\n")
    
    return artifacts


def get_model_info(model):
    """
    Extract model information for debugging/monitoring
    
    Args:
        model: Keras model
    
    Returns:
        dict with model information
    """
    info = {
        'name': model.name,
        'total_params': model.count_params(),
        'trainable_params': sum([keras.backend.count_params(w) for w in model.trainable_weights]),
        'non_trainable_params': sum([keras.backend.count_params(w) for w in model.non_trainable_weights]),
        'layers': len(model.layers),
        'input_shape': model.input_shape,
        'output_shape': model.output_shape
    }
    
    # Get loss function name
    if hasattr(model, 'loss') and model.loss:
        info['loss'] = model.loss.__name__ if hasattr(model.loss, '__name__') else str(model.loss)
    else:
        info['loss'] = 'Unknown'
    
    # Get optimizer name
    if hasattr(model, 'optimizer') and model.optimizer:
        info['optimizer'] = model.optimizer.__class__.__name__
    else:
        info['optimizer'] = 'Unknown'
    
    return info
