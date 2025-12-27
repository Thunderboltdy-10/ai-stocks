# =============================================================================
# Multi-day prediction with error propagation handling
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow import keras
import joblib
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from data.cache_manager import DataCacheManager
from data.feature_engineer import engineer_features, get_feature_columns, validate_and_fix_features


class MultiDayPredictor:
    """
    Predict multiple days ahead with confidence degradation tracking
    Uses recursive prediction with synthetic feature generation
    """
    
    def __init__(self, symbol, risk_profile):
        self.symbol = symbol
        self.risk_profile = risk_profile
        
        # Load models
        self.model_reg = keras.models.load_model(
            f"../saved_models/{symbol}_regressor.keras"  # SHARED regression model
        )
        self.model_class = keras.models.load_model(
            f"../saved_models/{symbol}_{risk_profile}_classifier.keras"
        )
        self.scaler = joblib.load(f"../saved_models/{symbol}_{risk_profile}_scaler.pkl")
        self.metadata = joblib.load(f"../saved_models/{symbol}_{risk_profile}_metadata.pkl")
        self.feature_columns = get_feature_columns()
        
        self.sequence_length = self.metadata['sequence_length']
    
    def predict_n_days(self, n_days=5):
        """
        Predict multiple 5-day periods ahead
        
        IMPORTANT: Model predicts 5 trading days ahead, NOT 1 day.
        
        Strategy:
        - Day 1: Model's 5-day prediction from real data (most reliable)
        - Day 2: Another 5-day prediction from day 1's synthetic data (10 days total, less reliable)
        - Day 3: Another 5-day prediction from day 2's synthetic data (15 days total, even less reliable)
        
        Args:
            n_days: Number of 5-day periods to predict (1-3 recommended, each = 5 trading days)
        
        Returns:
            List of predictions with confidence scores
        """
        print(f"\n{'='*70}")
        print(f"MULTI-DAY PREDICTION: {self.symbol} ({self.risk_profile.upper()})")
        print(f"{'='*70}\n")
        print(f"⚠️  Note: Model predicts 5 trading days ahead per step")
        print(f"Generating {n_days} predictions ({n_days * 5} trading days total)...\n")
        
        # Fetch and prepare data via cache manager (engineered features returned)
        cm = DataCacheManager()
        raw_df, engineered_df, prepared_df, feature_cols = cm.get_or_fetch_data(self.symbol, include_sentiment=True, force_refresh=False)
        df = engineered_df
        df_original = raw_df.copy()
        print("   ✓ Using engineered features from cache manager (including sentiment)")
        # Validate feature count (non-fragmenting)
        df = validate_and_fix_features(df)
        
        feature_data = df[self.feature_columns].values
        feature_data_scaled = self.scaler.transform(feature_data)
        
        current_price = df['Close'].iloc[-1]
        predictions = []
        
        # Keep track of sequence for recursive prediction
        sequence = feature_data_scaled[-self.sequence_length:].copy()
        
        for step in range(1, n_days + 1):
            trading_days_ahead = step * 5
            print(f"Prediction {step} (trading days +{trading_days_ahead})...")
            
            # Prepare input
            X_input = sequence[-self.sequence_length:].reshape(
                1, self.sequence_length, len(self.feature_columns)
            )
            
            # Get predictions
            pred_log_return = self.model_reg.predict(X_input, verbose=0)[0, 0]
            pred_class_probs = self.model_class.predict(X_input, verbose=0)[0]
            pred_class = pred_class_probs.argmax()
            class_confidence = pred_class_probs[pred_class]
            
            # Calculate predicted price (5 trading days from current reference point)
            predicted_price = current_price * np.exp(pred_log_return)
            
            # Confidence degradation (exponential decay per 5-day period)
            confidence_factor = 0.7 ** (step - 1)  # 70% retention per 5-day period
            adjusted_confidence = class_confidence * confidence_factor
            
            # Classification
            class_names = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
            recommendation = class_names[pred_class]
            
            # Store prediction
            predictions.append({
                'step': step,
                'trading_days_ahead': trading_days_ahead,
                'predicted_price': float(predicted_price),
                'log_return': float(pred_log_return),
                'return_pct': float((np.exp(pred_log_return) - 1) * 100),
                'recommendation': recommendation,
                'class_confidence': float(class_confidence),
                'adjusted_confidence': float(adjusted_confidence),
                'confidence_factor': float(confidence_factor)
            })
            
            print(f"  Price: ${predicted_price:.2f} ({(np.exp(pred_log_return)-1)*100:+.2f}%)")
            print(f"  Recommendation: {recommendation} ({adjusted_confidence*100:.1f}% confidence)")
            
            # Generate synthetic features for next iteration
            if step < n_days:
                new_features = self._generate_synthetic_features(
                    df, predicted_price, sequence[-1]
                )
                
                # Scale new features
                new_features_scaled = self.scaler.transform(new_features.reshape(1, -1))[0]
                
                # Update sequence (sliding window)
                sequence = np.vstack([sequence[1:], new_features_scaled])
                
                # Update current price for next iteration
                current_price = predicted_price
        
        print(f"\n{'='*70}\n")
        
        return predictions
    
    def _generate_synthetic_features(self, df_historical, predicted_price, last_features):
        """
        Generate synthetic features for predicted price
        
        This is the tricky part - we need to estimate technical indicators
        for a price that doesn't exist yet
        
        Strategy:
        1. Simple indicators (returns, momentum): calculate from predicted price
        2. Complex indicators (RSI, MACD): use trend extrapolation
        3. Volume: use average recent volume
        """
        # Get recent actual data for context
        recent_close = df_historical['Close'].iloc[-1]
        recent_volume = df_historical['Volume'].iloc[-20:].mean()
        
        # Calculate what we can directly
        returns = (predicted_price - recent_close) / recent_close
        log_returns = np.log(predicted_price / recent_close)
        
        # For other features, use last known values with small random walk
        # This is an approximation - real features would need full price history
        
        # Simple approach: take last features and apply small perturbation
        synthetic_features = last_features.copy()
        
        # Update features we can calculate
        feature_dict = dict(zip(self.feature_columns, synthetic_features))
        
        # Update calculable features
        if 'returns' in feature_dict:
            feature_dict['returns'] = returns
        if 'log_returns' in feature_dict:
            feature_dict['log_returns'] = log_returns
        if 'momentum_1d' in feature_dict:
            feature_dict['momentum_1d'] = predicted_price - recent_close
        
        # Trend-based updates for other features (simplified)
        # In production, you'd want more sophisticated synthetic feature generation
        for key in feature_dict:
            if 'velocity' in key or 'acceleration' in key:
                # Apply momentum from prediction
                feature_dict[key] = feature_dict[key] * (1 + returns * 0.5)
        
        # Convert back to array in correct order
        synthetic_array = np.array([feature_dict[col] for col in self.feature_columns])
        
        return synthetic_array


def predict_multiday_cli():
    """Command-line interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-horizon stock prediction')
    parser.add_argument('symbol', type=str, help='Stock symbol')
    parser.add_argument('risk_profile', type=str, choices=['conservative', 'aggressive'])
    parser.add_argument('--steps', type=int, default=3, help='Number of 5-day periods (default=3 → 15 days)')
    
    args = parser.parse_args()
    
    predictor = MultiDayPredictor(args.symbol, args.risk_profile)
    predictions = predictor.predict_n_days(n_days=args.steps)
    
    print("\n" + "="*70)
    print("MULTI-HORIZON FORECAST SUMMARY")
    print("="*70 + "\n")
    print("⚠️  Each step represents a 5-day prediction horizon:\n")
    
    for pred in predictions:
        days_ahead = pred['trading_days_ahead']
        weeks = days_ahead / 5
        print(f"  Step {pred['step']} ({weeks:.0f}x 5-day, ~{days_ahead} trading days): ${pred['predicted_price']:.2f} "
              f"({pred['return_pct']:+.2f}%) - {pred['recommendation']} "
              f"({pred['adjusted_confidence']*100:.1f}% confidence)")
    
    return predictions


if __name__ == "__main__":
    predict_multiday_cli()
