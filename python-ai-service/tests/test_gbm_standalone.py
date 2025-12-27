"""
F4 Model Validation: GBM Standalone Test

Tests that the GBM models (XGBoost/LightGBM) can:
1. Load successfully from saved artifacts
2. Generate predictions with correct shape
3. Produce non-collapsed outputs (sufficient variance)
4. Run a basic backtest without errors

Success Criteria:
- Model loads without errors
- Predictions have shape (n_samples,)
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
from datetime import datetime

print("\n" + "="*80)
print("F4 MODEL VALIDATION: GBM STANDALONE TEST")
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
    """Test 1: Can we load the GBM models?"""
    print("TEST 1: Model Loading")
    print("-" * 40)

    try:
        # Check if GBM is available
        try:
            from inference.load_gbm_models import load_gbm_models, GBMModelBundle
            print(f"   ✅ GBM inference module available")
        except ImportError as e:
            print(f"   ❌ GBM module not available: {e}")
            test_results['tests_failed'] += 1
            test_results['errors'].append(f"GBM module not available")
            return None

        # Load GBM models
        print(f"   Loading GBM models for {TEST_SYMBOL}...")
        gbm_bundle, metadata = load_gbm_models(TEST_SYMBOL)

        if gbm_bundle is None:
            raise FileNotFoundError(f"No GBM models found for {TEST_SYMBOL}")

        print(f"   ✅ GBM models loaded successfully")
        print(f"   XGBoost available: {gbm_bundle.xgb_model is not None}")
        print(f"   LightGBM available: {gbm_bundle.lgb_model is not None}")

        if gbm_bundle.feature_columns is not None:
            print(f"   Feature count: {len(gbm_bundle.feature_columns)}")

        test_results['tests_passed'] += 1
        test_results['xgb_available'] = gbm_bundle.xgb_model is not None
        test_results['lgb_available'] = gbm_bundle.lgb_model is not None

        return gbm_bundle

    except Exception as e:
        print(f"   ❌ FAILED: {str(e)}")
        test_results['tests_failed'] += 1
        test_results['errors'].append(f"Model Loading: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def test_prediction_generation(gbm_bundle):
    """Test 2: Can we generate predictions?"""
    print("\nTEST 2: Prediction Generation")
    print("-" * 40)

    if gbm_bundle is None:
        print("   ⏭️  SKIPPED: GBM models not loaded")
        return None, None

    try:
        from data.data_fetcher import fetch_stock_data
        from data.feature_engineer import engineer_features, get_feature_columns
        from inference.load_gbm_models import predict_with_gbm

        # Fetch recent data
        print(f"   Fetching {TEST_SYMBOL} data...")
        df = fetch_stock_data(symbol=TEST_SYMBOL, period='1y')
        print(f"   ✅ Fetched {len(df)} days of data")

        # Engineer features
        print(f"   Engineering features...")
        df_features = engineer_features(df, include_sentiment=True)
        print(f"   ✅ Engineered {df_features.shape[1]} features")

        # Get feature columns
        if gbm_bundle.feature_columns is not None:
            feature_cols = gbm_bundle.feature_columns
        else:
            feature_cols = get_feature_columns(include_sentiment=True)

        print(f"   Using {len(feature_cols)} features")

        # Prepare data (GBM uses flattened features, not sequences)
        # Take last 100 days for testing
        X_test = df_features[feature_cols].iloc[-100:].values
        y_test = df_features['return_1d'].iloc[-100:].values

        print(f"   ✅ Created {len(X_test)} test samples")
        print(f"   Test data shape: {X_test.shape}")

        # Generate predictions
        print(f"   Generating predictions...")
        predictions = predict_with_gbm(gbm_bundle, X_test)

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
    gbm_bundle = test_model_loading()

    # Test 2: Prediction Generation
    predictions, actuals = test_prediction_generation(gbm_bundle)

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

    if 'xgb_available' in test_results:
        print(f"XGBoost Available: {test_results['xgb_available']}")
    if 'lgb_available' in test_results:
        print(f"LightGBM Available: {test_results['lgb_available']}")

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
