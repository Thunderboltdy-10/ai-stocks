"""
Quantile Ensemble Inference Helper

This module provides utilities for loading and using quantile ensemble models
to make predictions with uncertainty estimates.

Usage:
    from inference.quantile_ensemble_inference import QuantileEnsemblePredictor
    
    predictor = QuantileEnsemblePredictor('AAPL')
    prediction, ci, position_size = predictor.predict(recent_data)
"""

import numpy as np
import pickle
from pathlib import Path
import tensorflow as tf

try:
    from utils.model_paths import ModelPaths, get_legacy_quantile_paths
except ImportError:
    ModelPaths = None
    get_legacy_quantile_paths = None


class QuantileEnsemblePredictor:
    """Load and use quantile ensemble models for prediction with uncertainty."""
    
    def __init__(self, symbol: str, models_dir: str = 'saved_models'):
        """Load all 3 quantile models (q25, q50, q75).
        
        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            models_dir: Directory containing saved models
        """
        self.symbol = symbol
        self.models_dir = Path(models_dir)
        
        # Try new path structure first if available
        if ModelPaths is not None:
            paths = ModelPaths(symbol)
            self._load_with_new_paths(paths)
        else:
            self._load_with_legacy_paths()
    
    def _load_with_new_paths(self, paths):
        """Load models using new organized path structure."""
        self.models = {}
        self.feature_scaler = None
        self.target_scaler = None
        self.feature_cols = None
        
        # Check if new structure exists
        quantile_base = paths.quantile.base
        if quantile_base.exists():
            # Load from new structure
            try:
                with open(paths.quantile.feature_scaler, 'rb') as f:
                    self.feature_scaler = pickle.load(f)
                with open(paths.quantile.target_scaler, 'rb') as f:
                    self.target_scaler = pickle.load(f)
                # Feature columns may be in quantile dir or symbol level
                if paths.quantile.base / 'features.pkl':
                    with open(paths.quantile.base / 'features.pkl', 'rb') as f:
                        self.feature_cols = pickle.load(f)
                elif paths.feature_columns.exists():
                    with open(paths.feature_columns, 'rb') as f:
                        self.feature_cols = pickle.load(f)
                
                # Load models for each quantile
                for q in [25, 50, 75]:
                    model_path = paths.quantile.base / f'q{q}_model'
                    if model_path.exists():
                        self.models[f'q{q}'] = tf.saved_model.load(str(model_path))
                
                if self.models:
                    print(f"✅ Loaded quantile ensemble for {self.symbol} (new structure)")
                    print(f"   Models: {', '.join(self.models.keys())}")
                    return
            except Exception:
                pass
        
        # Fallback to legacy paths
        self._load_with_legacy_paths()
    
    def _load_with_legacy_paths(self):
        """Load models using legacy flat path structure."""
        self.models = {}
        self.feature_scaler = None
        self.target_scaler = None
        self.feature_cols = None
        
        for q in [25, 50, 75]:
            model_name = f'{self.symbol}_1d_regressor_final_q{q}'
            
            # Load Keras model
            model_path = self.models_dir / f'{model_name}_model'
            if model_path.exists():
                self.models[f'q{q}'] = tf.saved_model.load(str(model_path))
            
            # Load scalers and features (same for all models)
            if self.feature_scaler is None:
                scaler_path = self.models_dir / f'{model_name}_feature_scaler.pkl'
                if scaler_path.exists():
                    with open(scaler_path, 'rb') as f:
                        self.feature_scaler = pickle.load(f)
                
                target_scaler_path = self.models_dir / f'{model_name}_target_scaler.pkl'
                if target_scaler_path.exists():
                    with open(target_scaler_path, 'rb') as f:
                        self.target_scaler = pickle.load(f)
                
                features_path = self.models_dir / f'{model_name}_features.pkl'
                if features_path.exists():
                    with open(features_path, 'rb') as f:
                        self.feature_cols = pickle.load(f)
        
        if self.models:
            print(f"✅ Loaded quantile ensemble for {self.symbol} (legacy structure)")
            print(f"   Models: {', '.join(self.models.keys())}")
    
    def predict(self, data, sequence_length=90):
        """Make prediction with uncertainty estimates.
        
        Args:
            data: DataFrame with features (must have at least sequence_length rows)
            sequence_length: Lookback window (default: 90)
        
        Returns:
            tuple: (ensemble_prediction, confidence_interval, position_size_multiplier)
                - ensemble_prediction: Weighted average of quantile predictions
                - confidence_interval: (q75 - q25) / q50 (lower is better)
                - position_size_multiplier: 1.0 for CI < 0.5, decreasing to 0 for CI > 2.0
        """
        # Extract features
        X = data[self.feature_cols].values
        
        # Create sequences
        if len(X) < sequence_length:
            raise ValueError(f"Need at least {sequence_length} rows, got {len(X)}")
        
        # Get most recent sequence
        X_seq = X[-sequence_length:]
        
        # Scale features
        X_scaled = self.feature_scaler.transform(X_seq)
        X_input = np.expand_dims(X_scaled, axis=0)  # Add batch dimension
        
        # Get predictions from all 3 models
        predictions_scaled = {}
        for q in [25, 50, 75]:
            pred_scaled = self.models[f'q{q}'](X_input).numpy().flatten()[0]
            predictions_scaled[f'q{q}'] = pred_scaled
        
        # Inverse transform to original scale
        predictions = {}
        for q in [25, 50, 75]:
            pred_orig = self.target_scaler.inverse_transform(
                [[predictions_scaled[f'q{q}']]]
            )[0][0]
            predictions[f'q{q}'] = pred_orig
        
        # Ensemble prediction: weighted average
        # Give more weight to median (q50) for robustness
        ensemble_pred = (
            0.2 * predictions['q25'] +
            0.6 * predictions['q50'] +
            0.2 * predictions['q75']
        )
        
        # Confidence interval (normalized by median)
        ci = abs((predictions['q75'] - predictions['q25']) / (predictions['q50'] + 1e-8))
        
        # Position size multiplier based on confidence
        # Narrow CI (< 0.5): High confidence → 1.0 (full position)
        # Medium CI (0.5-1.0): Moderate confidence → 0.5-1.0
        # Wide CI (1.0-2.0): Low confidence → 0.25-0.5
        # Very wide CI (> 2.0): Very low confidence → 0 (skip trade)
        if ci < 0.5:
            position_multiplier = 1.0
        elif ci < 1.0:
            position_multiplier = 1.0 - 0.5 * (ci - 0.5) / 0.5
        elif ci < 2.0:
            position_multiplier = 0.5 - 0.25 * (ci - 1.0) / 1.0
        else:
            position_multiplier = 0.0
        
        return ensemble_pred, ci, position_multiplier
    
    def predict_batch(self, data_list, sequence_length=90):
        """Make predictions for multiple sequences.
        
        Args:
            data_list: List of DataFrames, each with features
            sequence_length: Lookback window
        
        Returns:
            list: List of (prediction, ci, position_multiplier) tuples
        """
        results = []
        for data in data_list:
            pred, ci, pos_mult = self.predict(data, sequence_length)
            results.append((pred, ci, pos_mult))
        return results
    
    def get_prediction_details(self, data, sequence_length=90):
        """Get detailed prediction breakdown.
        
        Args:
            data: DataFrame with features
            sequence_length: Lookback window
        
        Returns:
            dict: Detailed prediction information
        """
        # Extract features and create sequence
        X = data[self.feature_cols].values[-sequence_length:]
        X_scaled = self.feature_scaler.transform(X)
        X_input = np.expand_dims(X_scaled, axis=0)
        
        # Get predictions
        predictions_scaled = {}
        for q in [25, 50, 75]:
            pred_scaled = self.models[f'q{q}'](X_input).numpy().flatten()[0]
            predictions_scaled[f'q{q}'] = pred_scaled
        
        # Inverse transform
        predictions = {}
        for q in [25, 50, 75]:
            pred_orig = self.target_scaler.inverse_transform(
                [[predictions_scaled[f'q{q}']]]
            )[0][0]
            predictions[f'q{q}'] = pred_orig
        
        # Ensemble
        ensemble_pred = (
            0.2 * predictions['q25'] +
            0.6 * predictions['q50'] +
            0.2 * predictions['q75']
        )
        
        ci = abs((predictions['q75'] - predictions['q25']) / (predictions['q50'] + 1e-8))
        
        if ci < 0.5:
            confidence_level = "HIGH"
            position_multiplier = 1.0
        elif ci < 1.0:
            confidence_level = "MEDIUM-HIGH"
            position_multiplier = 1.0 - 0.5 * (ci - 0.5) / 0.5
        elif ci < 2.0:
            confidence_level = "MEDIUM-LOW"
            position_multiplier = 0.5 - 0.25 * (ci - 1.0) / 1.0
        else:
            confidence_level = "LOW (SKIP)"
            position_multiplier = 0.0
        
        return {
            'symbol': self.symbol,
            'ensemble_prediction': ensemble_pred,
            'q25_prediction': predictions['q25'],
            'q50_prediction': predictions['q50'],
            'q75_prediction': predictions['q75'],
            'confidence_interval': ci,
            'confidence_level': confidence_level,
            'position_multiplier': position_multiplier,
            'prediction_range': (predictions['q25'], predictions['q75']),
            'spread': predictions['q75'] - predictions['q25']
        }


def example_usage():
    """Example of how to use the quantile ensemble predictor."""
    
    # Initialize predictor
    predictor = QuantileEnsemblePredictor('AAPL')
    
    # Load some data (you'd use your actual data pipeline)
    # from data.data_fetcher import fetch_stock_data
    # from data.feature_engineer import engineer_features
    # df = fetch_stock_data('AAPL', period='1y')
    # df = engineer_features(df)
    
    # Make prediction
    # prediction, ci, position_mult = predictor.predict(df)
    
    # Get detailed breakdown
    # details = predictor.get_prediction_details(df)
    
    # Print results
    # print(f"Ensemble prediction: {prediction:.4f}")
    # print(f"Confidence interval: {ci:.2f}")
    # print(f"Position multiplier: {position_mult:.2f}")
    # print(f"\nDetailed breakdown:")
    # for key, value in details.items():
    #     print(f"  {key}: {value}")
    
    pass


if __name__ == '__main__':
    example_usage()
