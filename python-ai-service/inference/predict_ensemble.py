"""Hybrid ensemble prediction CLI helper."""

from __future__ import annotations

import json
import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import pickle
import logging
from tensorflow.keras.models import load_model

from inference.hybrid_predictor import EnsemblePredictor  # noqa: E402
from utils.model_paths import ModelPaths, find_model_path  # noqa: E402


def load_model_with_metadata(symbol: str, model_type: str = 'regressor'):
    """Load model + scaler + metadata for backward compatibility.

    Behavior:
    - First check new path structure: saved_models/{symbol}/regressor/
    - Fallback to legacy paths: saved_models/{symbol}_*
    - Load `target_scaler_robust` when `scaling_method == 'robust'`, else legacy scaler
    - Load the saved model directory using `keras.models.load_model`
    Returns: (model, target_scaler, metadata)
    """
    logger = logging.getLogger(__name__)
    paths = ModelPaths(symbol)
    
    # Try to load metadata - check new paths first, then legacy
    metadata = None
    scaling_method = None
    
    # New path: saved_models/{symbol}/target_metadata.pkl or regressor/metadata.pkl
    metadata_candidates = [
        paths.target_metadata,
        paths.regressor.metadata,
        Path('saved_models') / f'{symbol}_target_metadata.pkl',
        Path('saved_models') / f'{symbol}_1d_regressor_final_metadata.pkl',
    ]
    
    for meta_path in metadata_candidates:
        if meta_path.exists():
            try:
                metadata = pickle.load(open(meta_path, 'rb'))
                scaling_method = metadata.get('scaling_method')
                break
            except Exception:
                continue
    
    if metadata is None:
        logger.warning(f"No metadata found for {symbol}, assuming v3.0 MinMaxScaler")
        metadata = {'scaling_method': 'minmax'}
        scaling_method = 'minmax'

    # Load appropriate scaler - check new paths first
    target_scaler = None
    scaler_candidates = []
    
    if scaling_method == 'robust':
        scaler_candidates = [
            paths.regressor.target_scaler_robust,
            paths.regressor.target_scaler,
            Path('saved_models') / f'{symbol}_1d_target_scaler_robust.pkl',
            Path('saved_models') / f'{symbol}_target_scaler_robust.pkl',
        ]
    else:
        scaler_candidates = [
            paths.regressor.target_scaler,
            Path('saved_models') / f'{symbol}_target_scaler.pkl',
            Path('saved_models') / f'{symbol}_1d_regressor_final_target_scaler.pkl',
        ]
    
    for scaler_path in scaler_candidates:
        if scaler_path.exists():
            try:
                target_scaler = pickle.load(open(scaler_path, 'rb'))
                break
            except Exception:
                continue
    
    if target_scaler is None:
        raise FileNotFoundError(f"Target scaler not found for {symbol}. Tried: {scaler_candidates}")

    # Load model - check new paths first
    model = None
    model_candidates = [
        paths.regressor.model,
        Path('saved_models') / f'{symbol}_1d_regressor_final_model',
        Path('saved_models') / f'{symbol}_{model_type}_final_model',
    ]
    
    for model_path in model_candidates:
        if model_path.exists():
            try:
                model = load_model(str(model_path))
                break
            except Exception:
                continue
    
    if model is None:
        raise FileNotFoundError(f"Model not found for {symbol}. Tried: {model_candidates}")

    return model, target_scaler, metadata


def load_ensemble_models(symbol: str, risk_profile: str = "conservative"):
    """Load hybrid ensemble artifacts while matching legacy return format."""

    print(f"\n📦 Loading hybrid ensemble models ({symbol}, {risk_profile})…")
    try:
        predictor = EnsemblePredictor(symbol=symbol, risk_profile=risk_profile)
    except Exception as exc:
        print(f"   ❌ Failed to initialise hybrid predictor: {exc}")
        import traceback
        traceback.print_exc()
        return None, None, None, None

    models = {}
    for horizon in predictor.horizons:
        horizon_key = f"{horizon}d"
        models[f"{horizon_key}_reg"] = predictor.regressors[horizon_key]
        models[f"{horizon_key}_class"] = predictor.classifiers[horizon_key]

    print(f"   ✅ Loaded {len(predictor.horizons)} horizons: {predictor.horizons}")
    return models, predictor.metadata, predictor.scalers, predictor.selected_features


def predict_ensemble(
    symbol: str,
    risk_profile: str = "conservative",
    data_period: str = "6mo",
    include_features: bool = False,
    feature_subset=None,
) -> dict:
    """Return ensemble prediction by delegating to the hybrid predictor."""

    predictor = EnsemblePredictor(symbol=symbol, risk_profile=risk_profile)
    return predictor.predict_latest(
        data_period=data_period,
        include_features=include_features,
        feature_subset=feature_subset,
    )


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run ensemble prediction for a stock symbol")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Stock symbol to predict")
    parser.add_argument("--risk_profile", "--risk-profile", type=str, default="conservative", 
                        help="Risk profile: conservative, moderate, aggressive")
    parser.add_argument("--fusion_mode", "--fusion-mode", type=str, default=None,
                        help="Fusion mode (optional, for compatibility)")
    
    args = parser.parse_args()
    
    result = predict_ensemble(args.symbol, args.risk_profile)
    print(json.dumps(result, indent=2))
