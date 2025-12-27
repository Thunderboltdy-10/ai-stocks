"""
GBM Model Loader for Inference Pipeline

Loads trained GBM models (XGBoost, LightGBM) for use in the inference
and backtesting pipeline. Mirrors the pattern used by load_model_with_metadata().

Author: AI-Stocks GBM Integration
Date: December 2025
"""

import sys
import pickle
import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List

import numpy as np
import joblib

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# GBM MODEL LOADING
# ============================================================================

class GBMModelBundle:
    """
    Container for loaded GBM models and their artifacts.
    
    Attributes:
        xgb_model: Trained XGBoost regressor (or None)
        lgb_model: Trained LightGBM regressor (or None)
        xgb_scaler: Scaler for XGBoost features (or None)
        lgb_scaler: Scaler for LightGBM features (or None)
        feature_columns: List of feature column names
        training_metadata: Dict with training configuration
    """
    
    def __init__(self):
        self.xgb_model = None
        self.lgb_model = None
        self.xgb_scaler = None
        self.lgb_scaler = None
        self.feature_columns: List[str] = []
        self.training_metadata: Dict[str, Any] = {}
        self.model_loaded: Dict[str, bool] = {'xgb': False, 'lgb': False}
    
    def has_xgb(self) -> bool:
        return self.xgb_model is not None
    
    def has_lgb(self) -> bool:
        return self.lgb_model is not None
    
    def any_loaded(self) -> bool:
        return self.has_xgb() or self.has_lgb()
    
    def get_available_models(self) -> List[str]:
        """Return list of available model names."""
        models = []
        if self.has_xgb():
            models.append('xgb')
        if self.has_lgb():
            models.append('lgb')
        return models


def load_gbm_models(
    symbol: str,
    model_dir: Path = None,
    prefer: str = 'lgb',  # Default to LightGBM (better performance)
) -> Tuple[Optional[GBMModelBundle], Dict[str, Any]]:
    """
    Load trained GBM models for a symbol.

    Args:
        symbol: Stock ticker symbol
        model_dir: Directory containing GBM models
        prefer: Preferred model ('lgb' or 'xgb') - defaults to LightGBM
    
    Returns:
        Tuple of (GBMModelBundle or None, metadata dict)
    """
    model_dir = model_dir or Path(f'saved_models/{symbol}/gbm')
    
    if not model_dir.exists():
        logger.warning(f"GBM model directory not found: {model_dir}")
        return None, {'error': 'directory_not_found'}
    
    bundle = GBMModelBundle()
    metadata = {
        'symbol': symbol,
        'model_dir': str(model_dir),
        'models_loaded': [],
        'errors': []
    }
    
    # Load feature columns (required)
    feature_cols_path = model_dir / 'feature_columns.pkl'
    if not feature_cols_path.exists():
        # Try parent directory (shared with LSTM)
        feature_cols_path = model_dir.parent / 'feature_columns.pkl'
    
    if feature_cols_path.exists():
        try:
            with open(feature_cols_path, 'rb') as f:
                bundle.feature_columns = pickle.load(f)
            logger.info(f"Loaded {len(bundle.feature_columns)} feature columns")
        except Exception as e:
            logger.error(f"Failed to load feature columns: {e}")
            metadata['errors'].append(f'feature_columns: {e}')
            return None, metadata
    else:
        logger.error(f"Feature columns file not found: {feature_cols_path}")
        metadata['errors'].append('feature_columns_not_found')
        return None, metadata
    
    # Load training metadata
    training_meta_path = model_dir / 'training_metadata.json'
    if training_meta_path.exists():
        try:
            import json
            with open(training_meta_path, 'r') as f:
                bundle.training_metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load training metadata: {e}")
    
    # Load XGBoost model
    xgb_path = model_dir / 'xgb_reg.joblib'
    xgb_scaler_path = model_dir / 'xgb_scaler.joblib'
    
    if xgb_path.exists() and xgb_scaler_path.exists():
        try:
            bundle.xgb_model = joblib.load(xgb_path)
            bundle.xgb_scaler = joblib.load(xgb_scaler_path)
            bundle.model_loaded['xgb'] = True
            metadata['models_loaded'].append('xgb')
            logger.info(f"Loaded XGBoost model from {xgb_path}")
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
            metadata['errors'].append(f'xgb: {e}')
    
    # Load LightGBM model
    lgb_path = model_dir / 'lgb_reg.joblib'
    lgb_scaler_path = model_dir / 'lgb_scaler.joblib'
    
    if lgb_path.exists() and lgb_scaler_path.exists():
        try:
            bundle.lgb_model = joblib.load(lgb_path)
            bundle.lgb_scaler = joblib.load(lgb_scaler_path)
            bundle.model_loaded['lgb'] = True
            metadata['models_loaded'].append('lgb')
            logger.info(f"Loaded LightGBM model from {lgb_path}")
        except Exception as e:
            logger.error(f"Failed to load LightGBM model: {e}")
            metadata['errors'].append(f'lgb: {e}')
    
    if not bundle.any_loaded():
        logger.error("No GBM models loaded successfully")
        return None, metadata
    
    logger.info(f"GBM bundle ready: {bundle.get_available_models()}")
    
    return bundle, metadata


# ============================================================================
# GBM PREDICTION
# ============================================================================

def predict_with_gbm(
    bundle: GBMModelBundle,
    features: np.ndarray,
    model: str = 'lgb',  # Default to LightGBM (better performance)
    return_both: bool = False,
    # UPDATED: LightGBM significantly outperforms XGBoost on AAPL
    # LightGBM: IC=0.652, Dir Acc=65.8%, Variance=0.00569
    # XGBoost:  IC=0.401, Dir Acc=55.1%, Variance=0.00209 (collapsed), 87% positive bias
    # Recommendation: Use LightGBM primarily until XGBoost issues fixed
    lgb_weight: float = 0.7,  # Weight LightGBM more heavily (superior performance)
    xgb_weight: float = 0.3,  # Reduced weight for XGBoost (needs calibration)
) -> np.ndarray | Dict[str, np.ndarray]:
    """
    Generate predictions using loaded GBM models.

    Args:
        bundle: GBMModelBundle with loaded models
        features: 2D feature matrix (n_samples, n_features)
        model: Which model to use ('lgb', 'xgb', 'avg', 'weighted') - defaults to LightGBM
        return_both: If True, return predictions from all available models
        lgb_weight: Weight for LightGBM in weighted average (default 0.7, superior performance)
        xgb_weight: Weight for XGBoost in weighted average (default 0.3, needs calibration)

    Returns:
        Predictions array or dict of predictions per model
    """
    if not bundle.any_loaded():
        raise ValueError("No GBM models loaded in bundle")
    
    features = np.asarray(features)
    if features.ndim == 1:
        features = features.reshape(1, -1)
    
    predictions = {}
    
    # XGBoost prediction
    if bundle.has_xgb() and (model in ['xgb', 'avg', 'weighted'] or return_both):
        X_scaled = bundle.xgb_scaler.transform(features)
        predictions['xgb'] = bundle.xgb_model.predict(X_scaled)
    
    # LightGBM prediction
    if bundle.has_lgb() and (model in ['lgb', 'avg', 'weighted'] or return_both):
        X_scaled = bundle.lgb_scaler.transform(features)
        predictions['lgb'] = bundle.lgb_model.predict(X_scaled)
    
    if return_both:
        return predictions
    
    if model == 'avg':
        # Simple average of available predictions
        preds_list = list(predictions.values())
        return np.mean(preds_list, axis=0)
    
    if model == 'weighted':
        # Weighted average favoring LightGBM (better IC and direction accuracy)
        if 'xgb' in predictions and 'lgb' in predictions:
            return lgb_weight * predictions['lgb'] + xgb_weight * predictions['xgb']
        return list(predictions.values())[0]
    
    return predictions.get(model, list(predictions.values())[0])


def predict_with_ensemble(
    gbm_bundle: GBMModelBundle,
    regressor_pred: Optional[np.ndarray],
    features: np.ndarray,
    fusion_mode: str = 'weighted',
    weights: Dict[str, float] = None,
) -> Dict[str, Any]:
    """
    Ensemble predictions from GBM and LSTM/Transformer regressor.
    
    Args:
        gbm_bundle: GBMModelBundle with loaded models
        regressor_pred: Prediction from LSTM/Transformer (or None)
        features: Feature matrix for GBM prediction
        fusion_mode: How to fuse predictions ('weighted', 'avg', 'gbm_only', 'regressor_only')
        weights: Dict of model weights (e.g., {'regressor': 0.6, 'xgb': 0.2, 'lgb': 0.2})
    
    Returns:
        Dict with fused prediction and component predictions
    """
    result = {
        'fusion_mode': fusion_mode,
        'predictions': {},
        'weights_used': {},
        'fused_prediction': None,
    }
    
    # Get GBM predictions
    if gbm_bundle is not None and gbm_bundle.any_loaded():
        gbm_preds = predict_with_gbm(gbm_bundle, features, return_both=True)
        result['predictions'].update(gbm_preds)
    
    # Add regressor prediction
    if regressor_pred is not None:
        result['predictions']['regressor'] = np.asarray(regressor_pred).flatten()
    
    # Default weights if not provided
    if weights is None:
        weights = {
            'regressor': 0.5,
            'xgb': 0.25,
            'lgb': 0.25,
        }
    
    # Handle fusion modes
    if fusion_mode == 'regressor_only' and 'regressor' in result['predictions']:
        result['fused_prediction'] = result['predictions']['regressor']
        result['weights_used'] = {'regressor': 1.0}
    
    elif fusion_mode == 'gbm_only':
        # Average GBM predictions only
        gbm_keys = [k for k in result['predictions'] if k != 'regressor']
        if gbm_keys:
            gbm_preds = [result['predictions'][k] for k in gbm_keys]
            result['fused_prediction'] = np.mean(gbm_preds, axis=0)
            result['weights_used'] = {k: 1.0/len(gbm_keys) for k in gbm_keys}
    
    elif fusion_mode == 'avg':
        # Simple average of all available predictions
        all_preds = list(result['predictions'].values())
        result['fused_prediction'] = np.mean(all_preds, axis=0)
        n = len(all_preds)
        result['weights_used'] = {k: 1.0/n for k in result['predictions']}
    
    elif fusion_mode == 'weighted':
        # Weighted average using provided weights
        weighted_sum = np.zeros_like(list(result['predictions'].values())[0])
        total_weight = 0.0
        
        for model, pred in result['predictions'].items():
            w = weights.get(model, 0.0)
            if w > 0:
                weighted_sum += w * pred
                total_weight += w
                result['weights_used'][model] = w
        
        if total_weight > 0:
            result['fused_prediction'] = weighted_sum / total_weight
            # Normalize reported weights
            result['weights_used'] = {k: v/total_weight for k, v in result['weights_used'].items()}
    
    return result


# ============================================================================
# SIGNAL GENERATION
# ============================================================================

def generate_gbm_signals(
    predictions: np.ndarray,
    buy_threshold: float = 0.002,
    sell_threshold: float = -0.002,
    confidence_scale: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Generate trading signals from GBM predictions.
    
    Args:
        predictions: Array of predicted returns
        buy_threshold: Minimum return for buy signal
        sell_threshold: Maximum return for sell signal (negative)
        confidence_scale: Scale confidence by prediction magnitude
    
    Returns:
        Dict with 'signal' (1=buy, 0=hold, -1=sell) and 'confidence' arrays
    """
    predictions = np.asarray(predictions).flatten()
    
    signals = np.zeros(len(predictions), dtype=np.int32)
    signals[predictions > buy_threshold] = 1
    signals[predictions < sell_threshold] = -1
    
    if confidence_scale:
        # Scale confidence by prediction magnitude
        max_pred = np.percentile(np.abs(predictions), 99)
        if max_pred > 0:
            confidence = np.clip(np.abs(predictions) / max_pred, 0, 1)
        else:
            confidence = np.zeros(len(predictions))
    else:
        confidence = np.ones(len(predictions))
    
    return {
        'signal': signals,
        'confidence': confidence,
        'raw_prediction': predictions,
    }


# ============================================================================
# CLI FOR TESTING
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Test GBM model loading')
    parser.add_argument('symbol', type=str, help='Stock ticker symbol')
    
    args = parser.parse_args()
    symbol = args.symbol.upper()
    
    print(f"\nLoading GBM models for {symbol}...")
    bundle, metadata = load_gbm_models(symbol)
    
    if bundle is None:
        print("Failed to load GBM models")
        print(f"Errors: {metadata.get('errors', [])}")
        sys.exit(1)
    
    print(f"Models loaded: {bundle.get_available_models()}")
    print(f"Feature columns: {len(bundle.feature_columns)}")
    
    # Test prediction with dummy data
    n_features = len(bundle.feature_columns)
    dummy_features = np.random.randn(10, n_features)
    
    for model in bundle.get_available_models():
        preds = predict_with_gbm(bundle, dummy_features, model=model)
        print(f"\n{model.upper()} predictions (dummy data):")
        print(f"  Shape: {preds.shape}")
        print(f"  Range: [{preds.min():.4f}, {preds.max():.4f}]")
        print(f"  Mean: {preds.mean():.4f}")
