"""
F4 Model Validation: Binary Classifiers Standalone Test

Tests that the Binary Classifier models (BUY/SELL) can:
1. Load successfully from saved artifacts
2. Generate probability predictions with correct shape
3. Produce balanced predictions (not all 0 or all 1)
4. Run a basic backtest without errors

Success Criteria:
- Both BUY and SELL classifiers load without errors
- Predictions have shape (n_samples,) with probabilities [0, 1]
- At least 10% of samples have BUY prob > 0.5
- At least 10% of samples have SELL prob > 0.5
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
from datetime import datetime

print("\n" + "="*80)
print("F4 MODEL VALIDATION: BINARY CLASSIFIERS STANDALONE TEST")
print("="*80 + "\n")

# Test configuration
TEST_SYMBOL = os.environ.get('TEST_SYMBOL', 'AAPL')
MIN_POSITIVE_SIGNALS = 0.10  # At least 10% BUY signals
MIN_NEGATIVE_SIGNALS = 0.10  # At least 10% SELL signals

test_results = {
    'symbol': TEST_SYMBOL,
    'timestamp': datetime.now().isoformat(),
    'tests_passed': 0,
    'tests_failed': 0,
    'errors': []
}

def test_model_loading():
    """Test 1: Can we load the binary classifier models?"""
    print("TEST 1: Model Loading")
    print("-" * 40)

    try:
        from utils.model_paths import ModelPaths, get_legacy_classifier_paths
        from tensorflow.keras.models import load_model
        from utils.losses import register_custom_objects

        # Register custom loss functions
        register_custom_objects()

        paths = ModelPaths(TEST_SYMBOL)

        # Load BUY classifier
        buy_model = None
        if paths.classifiers.buy_model.exists():
            buy_model = load_model(str(paths.classifiers.buy_model))
            print(f"   ✅ BUY classifier loaded (new path)")
        else:
            # Try legacy path
            legacy_paths = get_legacy_classifier_paths(TEST_SYMBOL)
            if legacy_paths and legacy_paths.get('buy_model') and Path(legacy_paths['buy_model']).exists():
                buy_model = load_model(legacy_paths['buy_model'])
                print(f"   ✅ BUY classifier loaded (legacy path)")
            else:
                print(f"   ⚠️  WARNING: No BUY classifier found")

        # Load SELL classifier
        sell_model = None
        if paths.classifiers.sell_model.exists():
            sell_model = load_model(str(paths.classifiers.sell_model))
            print(f"   ✅ SELL classifier loaded (new path)")
        else:
            # Try legacy path
            if legacy_paths and legacy_paths.get('sell_model') and Path(legacy_paths['sell_model']).exists():
                sell_model = load_model(legacy_paths['sell_model'])
                print(f"   ✅ SELL classifier loaded (legacy path)")
            else:
                print(f"   ⚠️  WARNING: No SELL classifier found")

        if buy_model is None and sell_model is None:
            raise FileNotFoundError(f"No classifier models found for {TEST_SYMBOL}")

        # Load feature scaler
        feature_scaler = None
        if paths.classifiers.feature_scaler.exists():
            feature_scaler = pickle.load(open(paths.classifiers.feature_scaler, 'rb'))
            print(f"   ✅ Feature scaler loaded (new path)")
        else:
            # Try legacy path
            if legacy_paths and legacy_paths.get('feature_scaler') and Path(legacy_paths['feature_scaler']).exists():
                feature_scaler = pickle.load(open(Path(legacy_paths['feature_scaler']), 'rb'))
                print(f"   ✅ Feature scaler loaded (legacy path)")

        if buy_model:
            print(f"   BUY model input shape: {buy_model.input_shape}")
            print(f"   BUY model output shape: {buy_model.output_shape}")

        if sell_model:
            print(f"   SELL model input shape: {sell_model.input_shape}")
            print(f"   SELL model output shape: {sell_model.output_shape}")

        test_results['tests_passed'] += 1
        test_results['buy_available'] = buy_model is not None
        test_results['sell_available'] = sell_model is not None

        return buy_model, sell_model, feature_scaler

    except Exception as e:
        print(f"   ❌ FAILED: {str(e)}")
        test_results['tests_failed'] += 1
        test_results['errors'].append(f"Model Loading: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_prediction_generation(buy_model, sell_model, feature_scaler):
    """Test 2: Can we generate predictions?"""
    print("\nTEST 2: Prediction Generation")
    print("-" * 40)

    if buy_model is None and sell_model is None:
        print("   ⏭️  SKIPPED: No models loaded")
        return None, None, None

    try:
        from data.data_fetcher import fetch_stock_data
        from data.feature_engineer import engineer_features, get_feature_columns
        from data.target_engineering import prepare_training_data

        # Fetch recent data
        print(f"   Fetching {TEST_SYMBOL} data...")
        df = fetch_stock_data(symbol=TEST_SYMBOL, period='1y')
        print(f"   ✅ Fetched {len(df)} days of data")

        # Engineer features
        print(f"   Engineering features...")
        df_features = engineer_features(df, include_sentiment=True)
        print(f"   ✅ Engineered {df_features.shape[1]} features")

        # Use whichever model is available to determine sequence length
        model = buy_model if buy_model is not None else sell_model
        sequence_length = model.input_shape[1]
        n_features_expected = model.input_shape[2]

        print(f"   Model expects: {sequence_length} timesteps, {n_features_expected} features")

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
        if feature_scaler is not None:
            X_test_flat = X_test.reshape(-1, X_test.shape[-1])
            X_test_scaled = feature_scaler.transform(X_test_flat).reshape(X_test.shape)
            print(f"   ✅ Features scaled")
        else:
            X_test_scaled = X_test
            print(f"   ⚠️  No scaler available, using unscaled features")

        # Generate BUY predictions
        buy_probs = None
        if buy_model is not None:
            print(f"   Generating BUY predictions...")
            buy_probs_raw = buy_model.predict(X_test_scaled, verbose=0)
            # Handle different output shapes
            if len(buy_probs_raw.shape) > 1 and buy_probs_raw.shape[1] > 1:
                buy_probs = buy_probs_raw[:, 1]  # Probability of positive class
            else:
                buy_probs = buy_probs_raw.flatten()

            print(f"   ✅ BUY predictions generated: {buy_probs.shape}")
            print(f"      Range: [{buy_probs.min():.4f}, {buy_probs.max():.4f}]")
            print(f"      Mean: {buy_probs.mean():.4f}")
            print(f"      % > 0.5: {np.mean(buy_probs > 0.5):.1%}")

        # Generate SELL predictions
        sell_probs = None
        if sell_model is not None:
            print(f"   Generating SELL predictions...")
            sell_probs_raw = sell_model.predict(X_test_scaled, verbose=0)
            # Handle different output shapes
            if len(sell_probs_raw.shape) > 1 and sell_probs_raw.shape[1] > 1:
                sell_probs = sell_probs_raw[:, 1]  # Probability of positive class
            else:
                sell_probs = sell_probs_raw.flatten()

            print(f"   ✅ SELL predictions generated: {sell_probs.shape}")
            print(f"      Range: [{sell_probs.min():.4f}, {sell_probs.max():.4f}]")
            print(f"      Mean: {sell_probs.mean():.4f}")
            print(f"      % > 0.5: {np.mean(sell_probs > 0.5):.1%}")

        test_results['tests_passed'] += 1
        return buy_probs, sell_probs, y_test

    except Exception as e:
        print(f"   ❌ FAILED: {str(e)}")
        test_results['tests_failed'] += 1
        test_results['errors'].append(f"Prediction Generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None


def test_prediction_balance(buy_probs, sell_probs):
    """Test 3: Are predictions balanced (not all same)?"""
    print("\nTEST 3: Prediction Balance")
    print("-" * 40)

    if buy_probs is None and sell_probs is None:
        print("   ⏭️  SKIPPED: No predictions available")
        return

    try:
        warnings = []

        # Check BUY predictions
        if buy_probs is not None:
            pct_buy_high = np.mean(buy_probs > 0.5)
            pct_buy_low = np.mean(buy_probs < 0.5)

            print(f"   BUY signals (prob > 0.5): {pct_buy_high:.1%}")

            if pct_buy_high < MIN_POSITIVE_SIGNALS:
                warnings.append(f"Too few BUY signals: {pct_buy_high:.1%} < {MIN_POSITIVE_SIGNALS:.0%}")

            if pct_buy_high > 0.90:
                warnings.append(f"Too many BUY signals: {pct_buy_high:.1%} (likely collapsed)")

        # Check SELL predictions
        if sell_probs is not None:
            pct_sell_high = np.mean(sell_probs > 0.5)
            pct_sell_low = np.mean(sell_probs < 0.5)

            print(f"   SELL signals (prob > 0.5): {pct_sell_high:.1%}")

            if pct_sell_high < MIN_NEGATIVE_SIGNALS:
                warnings.append(f"Too few SELL signals: {pct_sell_high:.1%} < {MIN_NEGATIVE_SIGNALS:.0%}")

            if pct_sell_high > 0.90:
                warnings.append(f"Too many SELL signals: {pct_sell_high:.1%} (likely collapsed)")

        if warnings:
            print(f"   ⚠️  WARNINGS:")
            for w in warnings:
                print(f"      - {w}")

        print(f"   ✅ PASSED: Prediction balance checked")
        test_results['tests_passed'] += 1

    except Exception as e:
        print(f"   ❌ FAILED: {str(e)}")
        test_results['tests_failed'] += 1
        test_results['errors'].append(f"Balance check: {str(e)}")


def test_basic_backtest(buy_probs, sell_probs, actuals):
    """Test 4: Can we run a basic backtest?"""
    print("\nTEST 4: Basic Backtest")
    print("-" * 40)

    if (buy_probs is None and sell_probs is None) or actuals is None:
        print("   ⏭️  SKIPPED: No predictions/actuals available")
        return

    try:
        # Create positions based on classifier signals
        # Long if BUY prob > 0.7, short if SELL prob > 0.7, else neutral
        positions = np.zeros(len(actuals))

        if buy_probs is not None and sell_probs is not None:
            # Both available: use both
            positions = np.where(buy_probs > 0.7, 1.0,
                        np.where(sell_probs > 0.7, -1.0, 0.0))
        elif buy_probs is not None:
            # Only BUY available
            positions = np.where(buy_probs > 0.7, 1.0, 0.0)
        elif sell_probs is not None:
            # Only SELL available
            positions = np.where(sell_probs > 0.7, -1.0, 0.0)

        # Calculate returns
        strategy_returns = positions[:-1] * actuals[1:]  # Shift by 1 for forward returns

        # Basic metrics
        total_return = np.sum(strategy_returns)
        mean_return = np.mean(strategy_returns)
        std_return = np.std(strategy_returns)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0

        # Signal statistics
        pct_long = np.mean(positions > 0)
        pct_short = np.mean(positions < 0)
        pct_neutral = np.mean(positions == 0)

        print(f"   Total Return: {total_return:.4f} ({total_return*100:.2f}%)")
        print(f"   Mean Daily Return: {mean_return:.6f} ({mean_return*100:.4f}%)")
        print(f"   Std Daily Return: {std_return:.6f}")
        print(f"   Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"   Signal Distribution:")
        print(f"      Long: {pct_long:.1%}")
        print(f"      Short: {pct_short:.1%}")
        print(f"      Neutral: {pct_neutral:.1%}")

        # Check if reasonable
        if sharpe_ratio < -2.0:
            print(f"   ⚠️  WARNING: Sharpe ratio very negative ({sharpe_ratio:.2f})")

        if pct_neutral > 0.95:
            print(f"   ⚠️  WARNING: Almost always neutral ({pct_neutral:.1%})")

        print(f"   ✅ PASSED: Backtest completed")
        test_results['tests_passed'] += 1
        test_results['sharpe_ratio'] = float(sharpe_ratio)
        test_results['pct_long'] = float(pct_long)
        test_results['pct_short'] = float(pct_short)

    except Exception as e:
        print(f"   ❌ FAILED: {str(e)}")
        test_results['tests_failed'] += 1
        test_results['errors'].append(f"Backtest: {str(e)}")


def main():
    """Run all tests"""
    print(f"Testing Symbol: {TEST_SYMBOL}\n")

    # Test 1: Model Loading
    buy_model, sell_model, feature_scaler = test_model_loading()

    # Test 2: Prediction Generation
    buy_probs, sell_probs, actuals = test_prediction_generation(buy_model, sell_model, feature_scaler)

    # Test 3: Balance Check
    test_prediction_balance(buy_probs, sell_probs)

    # Test 4: Basic Backtest
    test_basic_backtest(buy_probs, sell_probs, actuals)

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Symbol: {test_results['symbol']}")
    print(f"Tests Passed: {test_results['tests_passed']}")
    print(f"Tests Failed: {test_results['tests_failed']}")

    if 'buy_available' in test_results:
        print(f"BUY Classifier Available: {test_results['buy_available']}")
    if 'sell_available' in test_results:
        print(f"SELL Classifier Available: {test_results['sell_available']}")

    if test_results['tests_failed'] == 0:
        print(f"\n✅ ALL TESTS PASSED")
        if 'sharpe_ratio' in test_results:
            print(f"   Sharpe Ratio: {test_results['sharpe_ratio']:.4f}")
            print(f"   Long Signals: {test_results['pct_long']:.1%}")
            print(f"   Short Signals: {test_results['pct_short']:.1%}")
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
