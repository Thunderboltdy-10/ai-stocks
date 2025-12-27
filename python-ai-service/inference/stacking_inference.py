"""
Stacking Inference Module

This module provides inference functions for the stacking ensemble, combining
LSTM regressor and GBM model predictions through a meta-learner.

Key Features:
- Loads trained meta-learner and base models
- Generates stacked predictions
- Handles missing model predictions gracefully
- Provides prediction confidence and diagnostics

Usage:
    from inference.stacking_inference import predict_with_stacking
    
    predictions = predict_with_stacking(
        symbol='AAPL',
        features=feature_array,
        lstm_preds=lstm_predictions,  # Optional
        xgb_preds=xgb_predictions,    # Optional
        lgb_preds=lgb_predictions,    # Optional
    )
"""

import sys
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class StackingResult:
    """Container for stacking inference results."""
    predictions: np.ndarray
    confidence: np.ndarray
    base_predictions: Dict[str, np.ndarray]
    meta_learner_loaded: bool
    models_used: List[str]
    metadata: Dict[str, Any]


@dataclass
class ModelBundle:
    """Container for loaded models and metadata."""
    meta_learner: Any
    feature_columns: List[str]
    coefficients: Dict[str, float]
    intercept: float
    metadata: Dict[str, Any]


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_meta_learner(symbol: str, model_dir: Optional[Path] = None) -> Optional[ModelBundle]:
    """
    Load the trained meta-learner and its metadata.
    
    Args:
        symbol: Stock ticker symbol
        model_dir: Directory containing meta-learner (default: saved_models/{symbol}/meta_learner)
    
    Returns:
        ModelBundle with meta-learner and metadata, or None if not found
    """
    if model_dir is None:
        model_dir = PROJECT_ROOT / 'saved_models' / symbol / 'meta_learner'
    model_dir = Path(model_dir)
    
    model_path = model_dir / 'meta_learner.pkl'
    metadata_path = model_dir / 'meta_learner_metadata.json'
    cols_path = model_dir / 'feature_columns.json'
    
    if not model_path.exists():
        logger.warning(f"Meta-learner not found at {model_path}")
        return None
    
    try:
        # Load model
        with open(model_path, 'rb') as f:
            meta_learner = pickle.load(f)
        
        # Load metadata
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        else:
            metadata = {}
        
        # Load feature columns
        if cols_path.exists():
            with open(cols_path, 'r') as f:
                feature_columns = json.load(f)
        else:
            # Try to get from metadata
            feature_columns = metadata.get('feature_columns', ['oof_lstm', 'oof_xgb', 'oof_lgb'])
        
        # Get coefficients
        coefficients = metadata.get('coefficients', {})
        intercept = metadata.get('intercept', 0.0)
        
        logger.info(f"✓ Loaded meta-learner from {model_path}")
        logger.info(f"  Feature columns: {feature_columns}")
        logger.info(f"  Coefficients: {coefficients}")
        
        return ModelBundle(
            meta_learner=meta_learner,
            feature_columns=feature_columns,
            coefficients=coefficients,
            intercept=intercept,
            metadata=metadata,
        )
    except Exception as e:
        logger.error(f"Failed to load meta-learner: {e}")
        return None


# ============================================================================
# PREDICTION FUNCTIONS
# ============================================================================

def predict_with_stacking(
    symbol: str,
    lstm_preds: Optional[np.ndarray] = None,
    xgb_preds: Optional[np.ndarray] = None,
    lgb_preds: Optional[np.ndarray] = None,
    model_dir: Optional[Path] = None,
    fallback_weights: Optional[Dict[str, float]] = None,
) -> StackingResult:
    """
    Generate stacked predictions using the meta-learner.
    
    The meta-learner combines base model predictions (LSTM, XGBoost, LightGBM)
    into a final prediction using trained Ridge regression weights.
    
    If meta-learner is not available, falls back to weighted average.
    
    Args:
        symbol: Stock ticker symbol
        lstm_preds: LSTM regressor predictions (shape: n_samples,)
        xgb_preds: XGBoost predictions (shape: n_samples,)
        lgb_preds: LightGBM predictions (shape: n_samples,)
        model_dir: Directory containing meta-learner
        fallback_weights: Weights for fallback averaging (default: equal weights)
    
    Returns:
        StackingResult with predictions and metadata
    """
    # Determine sample count from available predictions
    n_samples = 0
    for preds in [lstm_preds, xgb_preds, lgb_preds]:
        if preds is not None:
            n_samples = len(preds)
            break
    
    if n_samples == 0:
        raise ValueError("At least one prediction array must be provided")
    
    # Load meta-learner
    model_bundle = load_meta_learner(symbol, model_dir)
    meta_learner_loaded = model_bundle is not None
    
    # Prepare base predictions dictionary
    base_predictions = {}
    if lstm_preds is not None:
        base_predictions['oof_lstm'] = lstm_preds
    if xgb_preds is not None:
        base_predictions['oof_xgb'] = xgb_preds
    if lgb_preds is not None:
        base_predictions['oof_lgb'] = lgb_preds
    
    models_used = list(base_predictions.keys())
    logger.info(f"Available base predictions: {models_used}")
    
    if meta_learner_loaded:
        # Use meta-learner
        predictions, confidence = _predict_with_meta_learner(
            model_bundle,
            lstm_preds,
            xgb_preds,
            lgb_preds,
            n_samples,
        )
        
        return StackingResult(
            predictions=predictions,
            confidence=confidence,
            base_predictions=base_predictions,
            meta_learner_loaded=True,
            models_used=models_used,
            metadata={
                'method': 'meta_learner',
                'coefficients': model_bundle.coefficients,
                'intercept': model_bundle.intercept,
            }
        )
    else:
        # Fallback to weighted average
        logger.warning("Meta-learner not available, using weighted average fallback")
        predictions, confidence = _predict_with_weighted_average(
            lstm_preds,
            xgb_preds,
            lgb_preds,
            n_samples,
            fallback_weights,
        )
        
        return StackingResult(
            predictions=predictions,
            confidence=confidence,
            base_predictions=base_predictions,
            meta_learner_loaded=False,
            models_used=models_used,
            metadata={
                'method': 'weighted_average',
                'weights': fallback_weights or 'equal',
            }
        )


def _predict_with_meta_learner(
    model_bundle: ModelBundle,
    lstm_preds: Optional[np.ndarray],
    xgb_preds: Optional[np.ndarray],
    lgb_preds: Optional[np.ndarray],
    n_samples: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate predictions using the meta-learner."""
    
    feature_columns = model_bundle.feature_columns
    meta_learner = model_bundle.meta_learner
    
    # Build feature matrix matching meta-learner's expected columns
    pred_map = {
        'oof_lstm': lstm_preds,
        'oof_xgb': xgb_preds,
        'oof_lgb': lgb_preds,
    }
    
    # Create feature matrix
    X = np.zeros((n_samples, len(feature_columns)))
    available_mask = np.ones(n_samples, dtype=bool)
    
    for i, col in enumerate(feature_columns):
        preds = pred_map.get(col)
        if preds is not None:
            X[:, i] = preds
        else:
            # Missing predictions - use mean imputation
            # This is a fallback; ideally all models should provide predictions
            X[:, i] = 0.0
            logger.warning(f"Missing predictions for {col}, using zero imputation")
    
    # Generate predictions
    predictions = meta_learner.predict(X)
    
    # Compute confidence based on model agreement
    confidence = _compute_agreement_confidence(
        lstm_preds, xgb_preds, lgb_preds
    )
    
    return predictions, confidence


def _predict_with_weighted_average(
    lstm_preds: Optional[np.ndarray],
    xgb_preds: Optional[np.ndarray],
    lgb_preds: Optional[np.ndarray],
    n_samples: int,
    weights: Optional[Dict[str, float]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fallback prediction using weighted average."""
    
    # Default equal weights
    if weights is None:
        weights = {
            'lstm': 0.4,
            'xgb': 0.3,
            'lgb': 0.3,
        }
    
    # Collect available predictions and their weights
    available_preds = []
    available_weights = []
    
    if lstm_preds is not None:
        available_preds.append(lstm_preds)
        available_weights.append(weights.get('lstm', 0.33))
    if xgb_preds is not None:
        available_preds.append(xgb_preds)
        available_weights.append(weights.get('xgb', 0.33))
    if lgb_preds is not None:
        available_preds.append(lgb_preds)
        available_weights.append(weights.get('lgb', 0.33))
    
    # Normalize weights
    total_weight = sum(available_weights)
    if total_weight > 0:
        available_weights = [w / total_weight for w in available_weights]
    
    # Weighted average
    predictions = np.zeros(n_samples)
    for preds, weight in zip(available_preds, available_weights):
        predictions += weight * preds
    
    # Compute confidence
    confidence = _compute_agreement_confidence(lstm_preds, xgb_preds, lgb_preds)
    
    return predictions, confidence


def _compute_agreement_confidence(
    lstm_preds: Optional[np.ndarray],
    xgb_preds: Optional[np.ndarray],
    lgb_preds: Optional[np.ndarray],
) -> np.ndarray:
    """
    Compute confidence based on agreement between models.
    
    Higher confidence when models agree on direction and magnitude.
    """
    available = []
    if lstm_preds is not None:
        available.append(lstm_preds)
    if xgb_preds is not None:
        available.append(xgb_preds)
    if lgb_preds is not None:
        available.append(lgb_preds)
    
    if len(available) < 2:
        # Single model - use prediction magnitude as confidence proxy
        preds = available[0] if available else np.array([0.0])
        confidence = np.clip(np.abs(preds) * 10, 0.0, 1.0)  # Scale to [0, 1]
        return confidence
    
    n_samples = len(available[0])
    confidence = np.zeros(n_samples)
    
    for i in range(n_samples):
        vals = [p[i] for p in available]
        
        # Directional agreement: all same sign?
        signs = [np.sign(v) for v in vals]
        dir_agree = len(set(signs)) == 1 and signs[0] != 0
        
        # Magnitude agreement: low std / mean
        mean_val = np.mean(vals)
        std_val = np.std(vals)
        mag_agree = 1.0 - min(std_val / (np.abs(mean_val) + 1e-6), 1.0)
        
        # Combined confidence
        if dir_agree:
            confidence[i] = 0.5 + 0.5 * mag_agree
        else:
            confidence[i] = 0.3 * mag_agree
    
    return np.clip(confidence, 0.0, 1.0)


# ============================================================================
# INTEGRATION WITH INFERENCE PIPELINE
# ============================================================================

def get_stacking_predictions_for_inference(
    symbol: str,
    lstm_regressor_preds: np.ndarray,
    gbm_preds: Optional[np.ndarray] = None,
    model_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Integration function for main inference pipeline.
    
    Takes LSTM regressor predictions and optional GBM predictions,
    returns stacked predictions compatible with compute_hybrid_positions.
    
    Args:
        symbol: Stock ticker symbol
        lstm_regressor_preds: Predictions from LSTM regressor
        gbm_preds: Optional averaged GBM predictions (XGB + LGB)
        model_dir: Meta-learner directory
    
    Returns:
        Dictionary with stacked predictions and metadata
    """
    try:
        # If we have separate XGB and LGB predictions, split them
        # Otherwise, treat gbm_preds as a single GBM ensemble
        xgb_preds = gbm_preds
        lgb_preds = None  # If available separately, pass here
        
        result = predict_with_stacking(
            symbol=symbol,
            lstm_preds=lstm_regressor_preds,
            xgb_preds=xgb_preds,
            lgb_preds=lgb_preds,
            model_dir=model_dir,
        )
        
        return {
            'predictions': result.predictions,
            'confidence': result.confidence,
            'method': 'stacking' if result.meta_learner_loaded else 'weighted_average',
            'models_used': result.models_used,
            'metadata': result.metadata,
        }
    except Exception as e:
        logger.error(f"Stacking inference failed: {e}")
        # Fallback to LSTM only
        return {
            'predictions': lstm_regressor_preds,
            'confidence': np.ones(len(lstm_regressor_preds)) * 0.5,
            'method': 'lstm_only_fallback',
            'models_used': ['lstm'],
            'metadata': {'error': str(e)},
        }


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    """CLI for testing stacking inference."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test stacking inference')
    parser.add_argument('--symbol', '-s', type=str, default='AAPL', help='Stock symbol')
    parser.add_argument('--test', action='store_true', help='Run test with dummy data')
    args = parser.parse_args()
    
    if args.test:
        # Generate dummy predictions for testing
        n = 100
        np.random.seed(42)
        
        lstm_preds = np.random.randn(n) * 0.01
        xgb_preds = lstm_preds + np.random.randn(n) * 0.005
        lgb_preds = lstm_preds + np.random.randn(n) * 0.005
        
        print(f"\nTesting stacking inference for {args.symbol}...")
        print(f"  LSTM preds: mean={lstm_preds.mean():.6f}, std={lstm_preds.std():.6f}")
        print(f"  XGB preds:  mean={xgb_preds.mean():.6f}, std={xgb_preds.std():.6f}")
        print(f"  LGB preds:  mean={lgb_preds.mean():.6f}, std={lgb_preds.std():.6f}")
        
        result = predict_with_stacking(
            symbol=args.symbol,
            lstm_preds=lstm_preds,
            xgb_preds=xgb_preds,
            lgb_preds=lgb_preds,
        )
        
        print(f"\nResults:")
        print(f"  Meta-learner loaded: {result.meta_learner_loaded}")
        print(f"  Method: {result.metadata.get('method', 'unknown')}")
        print(f"  Models used: {result.models_used}")
        print(f"  Stacked preds: mean={result.predictions.mean():.6f}, std={result.predictions.std():.6f}")
        print(f"  Confidence: mean={result.confidence.mean():.3f}, std={result.confidence.std():.3f}")
    else:
        # Just load and verify meta-learner
        model_bundle = load_meta_learner(args.symbol)
        if model_bundle:
            print(f"\n✅ Meta-learner loaded for {args.symbol}")
            print(f"  Feature columns: {model_bundle.feature_columns}")
            print(f"  Coefficients: {model_bundle.coefficients}")
            print(f"  Intercept: {model_bundle.intercept:.6f}")
        else:
            print(f"\n❌ No meta-learner found for {args.symbol}")
            print(f"  Run: python training/train_meta_learner.py --symbol {args.symbol}")


if __name__ == '__main__':
    main()
