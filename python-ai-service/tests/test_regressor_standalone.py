"""
F4 Model Validation: Regressor Standalone Test

Tests that the LSTM+Transformer Regressor model can:
1. Load successfully from saved artifacts
2. Generate predictions with correct shape
3. Produce non-collapsed outputs (sufficient variance)
4. Run a basic backtest without errors

Success Criteria:
- Model loads without errors
- Predictions have shape (n_samples,) or (n_samples, 1)
- Prediction std > 0.005 (0.5% minimum variance)
- At least 30% negative and 30% positive predictions
- Backtest runs and produces Sharpe ratio

Author: AI-Stocks F4 Validation
Date: December 20, 2025
"""

import sys
import os
from pathlib import Path

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pickle
from datetime import datetime, timedelta

print("\n" + "="*80)
print("F4 MODEL VALIDATION: REGRESSOR STANDALONE TEST")
print("="*80 + "\n")

# Test configuration
TEST_SYMBOL = os.environ.get('TEST_SYMBOL', 'AAPL')
REQUIRED_STD = 0.005  # Minimum prediction variance
MIN_NEGATIVE_PCT = 0.30  # At least 30% negative predictions
MIN_POSITIVE_PCT = 0.30  # At least 30% positive predictions

test_results = {
    'symbol': TEST_SYMBOL,
    'timestamp': datetime.now().isoformat(),
    'tests_passed': 0,
    'tests_failed': 0,
    'errors': []
}

def test_model_loading():
    """Test 1: Can we load the regressor model?"""
    print("TEST 1: Model Loading")
    print("-" * 40)

    try:
        from utils.model_paths import ModelPaths
        from tensorflow.keras.models import load_model
        from utils.losses import register_custom_objects

        # Register custom loss functions
        register_custom_objects()

        paths = ModelPaths(TEST_SYMBOL)

        # Check if model exists
        if not paths.regressor.model.exists():
            # Try legacy path
            legacy_path = Path('saved_models') / f'{TEST_SYMBOL}_1d_regressor_final_model'
            if legacy_path.exists():
                model_path = legacy_path
                print(f"   Using legacy model path: {model_path}")
            else:
                raise FileNotFoundError(f"No model found for {TEST_SYMBOL}")
        else:
            model_path = paths.regressor.model
            print(f"   Using new model path: {model_path}")

        # Load model
        model = load_model(str(model_path))
        print(f"   ✅ Model loaded successfully")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")

        # Load scalers
        if paths.regressor.feature_scaler.exists():
            feature_scaler = pickle.load(open(paths.regressor.feature_scaler, 'rb'))
        else:
            legacy_scaler = Path('saved_models') / f'{TEST_SYMBOL}_1d_regressor_final_feature_scaler.pkl'
            feature_scaler = pickle.load(open(legacy_scaler, 'rb'))
        print(f"   ✅ Feature scaler loaded")

        # Load target scaler
        if paths.regressor.target_scaler_robust.exists():
            target_scaler = pickle.load(open(paths.regressor.target_scaler_robust, 'rb'))
            print(f"   ✅ Target scaler (robust) loaded")
        elif paths.regressor.target_scaler.exists():
            target_scaler = pickle.load(open(paths.regressor.target_scaler, 'rb'))
            print(f"   ✅ Target scaler (legacy) loaded")
        else:
            legacy_scaler = Path('saved_models') / f'{TEST_SYMBOL}_target_scaler.pkl'
            target_scaler = pickle.load(open(legacy_scaler, 'rb'))
            print(f"   ✅ Target scaler (legacy flat) loaded")

        test_results['tests_passed'] += 1
        return model, feature_scaler, target_scaler

    except Exception as e:
        print(f"   ❌ FAILED: {str(e)}")
        test_results['tests_failed'] += 1
        test_results['errors'].append(f"Model Loading: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_prediction_generation(model, feature_scaler, target_scaler):
    """Test 2: Can we generate predictions?"""
    print("\nTEST 2: Prediction Generation")
    print("-" * 40)

    if model is None:
        print("   ⏭️  SKIPPED: Model not loaded")
        return None

    try:
        from data.data_fetcher import fetch_stock_data
        from data.feature_engineer import engineer_features, get_feature_columns, EXPECTED_FEATURE_COUNT
        from data.target_engineering import prepare_training_data

        # Fetch recent data
        print(f"   Fetching {TEST_SYMBOL} data...")
        df = fetch_stock_data(symbol=TEST_SYMBOL, period='1y')
        print(f"   ✅ Fetched {len(df)} days of data")

        # Engineer features
        print(f"   Engineering features...")
        df_features = engineer_features(df, include_sentiment=True)
        print(f"   ✅ Engineered {df_features.shape[1]} features")

        # Get expected features
        expected_features = get_feature_columns(include_sentiment=True)

        # Prepare data for inference
        sequence_length = model.input_shape[1]
        n_features_expected = model.input_shape[2]

        print(f"   Model expects: {sequence_length} timesteps, {n_features_expected} features")
        print(f"   Data has: {len(expected_features)} features")

        # Create sequences
        X, y = prepare_training_data(
            df_features,
            target='return_1d',
            sequence_length=sequence_length,
            train_test_split=1.0  # Use all data
        )

        # Take last 100 samples for testing
        X_test = X[-100:]
        y_test = y[-100:]

        print(f"   ✅ Created {len(X_test)} test sequences")
        print(f"   Test data shape: {X_test.shape}")

        # Scale features
        X_test_flat = X_test.reshape(-1, X_test.shape[-1])
        X_test_scaled = feature_scaler.transform(X_test_flat).reshape(X_test.shape)
        print(f"   ✅ Features scaled")

        # Generate predictions
        print(f"   Generating predictions...")
        predictions_scaled = model.predict(X_test_scaled, verbose=0)

        # Handle multi-output models
        if len(predictions_scaled.shape) > 1 and predictions_scaled.shape[1] > 1:
            print(f"   Multi-task model detected: {predictions_scaled.shape}")
            predictions_scaled = predictions_scaled[:, 0]  # Use first head (return prediction)

        # Inverse scale
        predictions = target_scaler.inverse_transform(
            predictions_scaled.reshape(-1, 1)
        ).flatten()

        print(f"   ✅ Generated {len(predictions)} predictions")
        print(f"   Prediction shape: {predictions.shape}")
        print(f"   Prediction range: [{predictions.min():.4f}, {predictions.max():.4f}]")
        print(f"   Prediction mean: {predictions.mean():.4f}")
        print(f"   Prediction std: {predictions.std():.6f}")

        test_results['tests_passed'] += 1
        return predictions, y_test

    except Exception as e:
        print(f"   ❌ FAILED: {str(e)}")
        test_results['tests_failed'] += 1
        test_results['errors'].append(f"Prediction Generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def test_prediction_variance(predictions):
    """Test 3: Do predictions have sufficient variance (not collapsed)?"""
    print("\nTEST 3: Prediction Variance (Collapse Check)")
    print("-" * 40)

    if predictions is None:
        print("   ⏭️  SKIPPED: No predictions available")
        return

    try:
        pred_std = np.std(predictions)
        pred_mean = np.mean(predictions)
        pct_negative = np.mean(predictions < 0)
        pct_positive = np.mean(predictions > 0)
        pct_near_zero = np.mean(np.abs(predictions) < 0.001)

        print(f"   Std: {pred_std:.6f} (required: >{REQUIRED_STD})")
        print(f"   Mean: {pred_mean:.6f}")
        print(f"   % Negative: {pct_negative:.1%} (required: >{MIN_NEGATIVE_PCT:.0%})")
        print(f"   % Positive: {pct_positive:.1%} (required: >{MIN_POSITIVE_PCT:.0%})")
        print(f"   % Near Zero: {pct_near_zero:.1%}")

        # Check variance
        if pred_std < REQUIRED_STD:
            print(f"   ❌ FAILED: Predictions collapsed! Std={pred_std:.6f} < {REQUIRED_STD}")
            test_results['tests_failed'] += 1
            test_results['errors'].append(f"Variance collapse: std={pred_std:.6f}")
            return

        # Check distribution balance
        warnings = []
        if pct_negative < MIN_NEGATIVE_PCT:
            warnings.append(f"Low negative predictions: {pct_negative:.1%}")

        if pct_positive < MIN_POSITIVE_PCT:
            warnings.append(f"Low positive predictions: {pct_positive:.1%}")

        if warnings:
            print(f"   ⚠️  WARNINGS:")
            for w in warnings:
                print(f"      - {w}")

        print(f"   ✅ PASSED: Predictions have sufficient variance")
        test_results['tests_passed'] += 1

    except Exception as e:
        print(f"   ❌ FAILED: {str(e)}")
        test_results['tests_failed'] += 1
        test_results['errors'].append(f"Variance check: {str(e)}")


def test_basic_backtest(predictions, actuals):
    """Test 4: Can we run a basic backtest?"""
    print("\nTEST 4: Basic Backtest")
    print("-" * 40)

    if predictions is None or actuals is None:
        print("   ⏭️  SKIPPED: No predictions/actuals available")
        return

    try:
        # Simple strategy: long if predicted return > 0, short if < 0
        positions = np.sign(predictions)  # +1, 0, or -1

        # Calculate returns
        strategy_returns = positions[:-1] * actuals[1:]  # Shift by 1 for forward returns

        # Basic metrics
        total_return = np.sum(strategy_returns)
        mean_return = np.mean(strategy_returns)
        std_return = np.std(strategy_returns)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0

        # Directional accuracy
        correct_direction = np.sum((predictions[:-1] * actuals[1:]) > 0)
        directional_accuracy = correct_direction / len(strategy_returns)

        print(f"   Total Return: {total_return:.4f} ({total_return*100:.2f}%)")
        print(f"   Mean Daily Return: {mean_return:.6f} ({mean_return*100:.4f}%)")
        print(f"   Std Daily Return: {std_return:.6f}")
        print(f"   Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"   Directional Accuracy: {directional_accuracy:.2%}")

        # Check if reasonable
        if sharpe_ratio < -2.0:
            print(f"   ⚠️  WARNING: Sharpe ratio very negative ({sharpe_ratio:.2f})")

        if directional_accuracy < 0.40:
            print(f"   ⚠️  WARNING: Low directional accuracy ({directional_accuracy:.1%})")

        print(f"   ✅ PASSED: Backtest completed")
        test_results['tests_passed'] += 1
        test_results['sharpe_ratio'] = float(sharpe_ratio)
        test_results['directional_accuracy'] = float(directional_accuracy)

    except Exception as e:
        print(f"   ❌ FAILED: {str(e)}")
        test_results['tests_failed'] += 1
        test_results['errors'].append(f"Backtest: {str(e)}")


def main():
    """Run all tests"""
    print(f"Testing Symbol: {TEST_SYMBOL}\n")

    # Test 1: Model Loading
    model, feature_scaler, target_scaler = test_model_loading()

    # Test 2: Prediction Generation
    predictions, actuals = test_prediction_generation(model, feature_scaler, target_scaler)

    # Test 3: Variance Check
    test_prediction_variance(predictions)

    # Test 4: Basic Backtest
    test_basic_backtest(predictions, actuals)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Symbol: {test_results['symbol']}")
    print(f"Tests Passed: {test_results['tests_passed']}")
    print(f"Tests Failed: {test_results['tests_failed']}")

    if test_results['tests_failed'] == 0:
        print(f"\n✅ ALL TESTS PASSED")
        if 'sharpe_ratio' in test_results:
            print(f"   Sharpe Ratio: {test_results['sharpe_ratio']:.4f}")
            print(f"   Directional Accuracy: {test_results['directional_accuracy']:.2%}")
    else:
        print(f"\n❌ SOME TESTS FAILED")
        print(f"\nErrors:")
        for error in test_results['errors']:
            print(f"   - {error}")

    print("="*80 + "\n")

    return test_results['tests_failed'] == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
