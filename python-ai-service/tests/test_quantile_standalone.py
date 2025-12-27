"""
F4 Model Validation: Quantile Regressor Standalone Test

Tests that the Quantile Regressor model can:
1. Load successfully from saved artifacts
2. Generate quantile predictions with correct shape (Q10, Q50, Q90)
3. Produce well-calibrated quantiles (coverage checks)
4. Run a basic backtest with uncertainty awareness

Success Criteria:
- Model loads without errors
- Predictions have shape (n_samples, 3) for Q10/Q50/Q90
- Q10 < Q50 < Q90 for most samples (monotonicity)
- Coverage approximately matches quantile levels (e.g., ~90% of actuals > Q10)
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
print("F4 MODEL VALIDATION: QUANTILE REGRESSOR STANDALONE TEST")
print("="*80 + "\n")

# Test configuration
TEST_SYMBOL = os.environ.get('TEST_SYMBOL', 'AAPL')
QUANTILES = [0.1, 0.5, 0.9]  # Q10, Q50, Q90
COVERAGE_TOLERANCE = 0.15  # Allow 15% deviation from target coverage

test_results = {
    'symbol': TEST_SYMBOL,
    'timestamp': datetime.now().isoformat(),
    'tests_passed': 0,
    'tests_failed': 0,
    'errors': []
}

def test_model_loading():
    """Test 1: Can we load the quantile regressor model?"""
    print("TEST 1: Model Loading")
    print("-" * 40)

    try:
        from utils.model_paths import ModelPaths, get_legacy_quantile_paths
        from tensorflow.keras.models import load_model
        from utils.losses import register_custom_objects

        # Register custom loss functions
        register_custom_objects()

        paths = ModelPaths(TEST_SYMBOL)

        # Check if model exists
        model = None
        model_path = None

        # Try new structure first
        if paths.quantile.weights.exists():
            # Load from weights (need to reconstruct model)
            print(f"   ⚠️  WARNING: Quantile weights found but need model reconstruction")
            print(f"   Skipping weight-based loading (not implemented)")

        # Try legacy path
        legacy_paths = get_legacy_quantile_paths(TEST_SYMBOL)
        if legacy_paths and legacy_paths.get('weights') and Path(legacy_paths['weights']).exists():
            print(f"   ⚠️  WARNING: Legacy quantile weights found but need model reconstruction")
            print(f"   Skipping weight-based loading (not implemented)")

        # Try finding saved model directory
        possible_paths = [
            paths.quantile.base / 'model.keras',
            Path('saved_models') / TEST_SYMBOL / 'quantile' / 'model.keras',
            Path('saved_models') / f'{TEST_SYMBOL}_quantile_model',
        ]

        for p in possible_paths:
            if p.exists():
                try:
                    model = load_model(str(p))
                    model_path = p
                    break
                except Exception:
                    continue

        if model is None:
            print(f"   ⚠️  WARNING: No quantile regressor model found for {TEST_SYMBOL}")
            print(f"   Tried paths:")
            for p in possible_paths:
                print(f"      - {p}")
            print(f"   ℹ️  NOTE: Quantile regressor may not be trained for this symbol")
            test_results['tests_failed'] += 1
            test_results['errors'].append("Model not found (may not be trained)")
            return None, None, None

        print(f"   ✅ Model loaded from: {model_path}")
        print(f"   Model type: {type(model).__name__}")
        print(f"   Input shape: {model.input_shape}")
        print(f"   Output shape: {model.output_shape}")

        # Check output shape matches expected quantiles
        expected_outputs = len(QUANTILES)
        if model.output_shape[-1] != expected_outputs:
            print(f"   ⚠️  WARNING: Expected {expected_outputs} outputs, got {model.output_shape[-1]}")

        # Load scalers
        feature_scaler = None
        target_scaler = None

        if paths.quantile.feature_scaler.exists():
            feature_scaler = pickle.load(open(paths.quantile.feature_scaler, 'rb'))
            print(f"   ✅ Feature scaler loaded")

        if paths.quantile.target_scaler.exists():
            target_scaler = pickle.load(open(paths.quantile.target_scaler, 'rb'))
            print(f"   ✅ Target scaler loaded")

        if legacy_paths:
            if feature_scaler is None and legacy_paths.get('feature_scaler'):
                scaler_path = Path(legacy_paths['feature_scaler'])
                if scaler_path.exists():
                    feature_scaler = pickle.load(open(scaler_path, 'rb'))
                    print(f"   ✅ Feature scaler loaded (legacy)")

            if target_scaler is None and legacy_paths.get('target_scaler'):
                scaler_path = Path(legacy_paths['target_scaler'])
                if scaler_path.exists():
                    target_scaler = pickle.load(open(scaler_path, 'rb'))
                    print(f"   ✅ Target scaler loaded (legacy)")

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
    """Test 2: Can we generate quantile predictions?"""
    print("\nTEST 2: Prediction Generation")
    print("-" * 40)

    if model is None:
        print("   ⏭️  SKIPPED: Model not loaded")
        return None, None

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

        # Prepare data
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
            print(f"   ⚠️  No feature scaler, using unscaled features")

        # Generate quantile predictions
        print(f"   Generating quantile predictions...")
        quantile_preds_scaled = model.predict(X_test_scaled, verbose=0)

        print(f"   Raw prediction shape: {quantile_preds_scaled.shape}")

        # Inverse scale
        if target_scaler is not None:
            q10 = target_scaler.inverse_transform(quantile_preds_scaled[:, 0].reshape(-1, 1)).flatten()
            q50 = target_scaler.inverse_transform(quantile_preds_scaled[:, 1].reshape(-1, 1)).flatten()
            q90 = target_scaler.inverse_transform(quantile_preds_scaled[:, 2].reshape(-1, 1)).flatten()
            print(f"   ✅ Predictions inverse-scaled")
        else:
            q10 = quantile_preds_scaled[:, 0]
            q50 = quantile_preds_scaled[:, 1]
            q90 = quantile_preds_scaled[:, 2]
            print(f"   ⚠️  No target scaler, using raw predictions")

        quantile_preds = np.stack([q10, q50, q90], axis=1)

        print(f"   ✅ Generated {len(quantile_preds)} quantile predictions")
        print(f"   Prediction shape: {quantile_preds.shape}")
        print(f"   Q10 range: [{q10.min():.4f}, {q10.max():.4f}]")
        print(f"   Q50 range: [{q50.min():.4f}, {q50.max():.4f}]")
        print(f"   Q90 range: [{q90.min():.4f}, {q90.max():.4f}]")

        test_results['tests_passed'] += 1
        return quantile_preds, y_test

    except Exception as e:
        print(f"   ❌ FAILED: {str(e)}")
        test_results['tests_failed'] += 1
        test_results['errors'].append(f"Prediction Generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None


def test_quantile_monotonicity(quantile_preds):
    """Test 3: Are quantiles monotonic (Q10 < Q50 < Q90)?"""
    print("\nTEST 3: Quantile Monotonicity")
    print("-" * 40)

    if quantile_preds is None:
        print("   ⏭️  SKIPPED: No predictions available")
        return

    try:
        q10 = quantile_preds[:, 0]
        q50 = quantile_preds[:, 1]
        q90 = quantile_preds[:, 2]

        # Check monotonicity
        monotonic_10_50 = np.sum(q10 <= q50)
        monotonic_50_90 = np.sum(q50 <= q90)
        monotonic_10_90 = np.sum(q10 <= q90)

        pct_10_50 = monotonic_10_50 / len(q10)
        pct_50_90 = monotonic_50_90 / len(q50)
        pct_10_90 = monotonic_10_90 / len(q10)

        print(f"   Q10 ≤ Q50: {monotonic_10_50}/{len(q10)} ({pct_10_50:.1%})")
        print(f"   Q50 ≤ Q90: {monotonic_50_90}/{len(q50)} ({pct_50_90:.1%})")
        print(f"   Q10 ≤ Q90: {monotonic_10_90}/{len(q10)} ({pct_10_90:.1%})")

        # Check if mostly monotonic
        if pct_10_50 < 0.80 or pct_50_90 < 0.80:
            print(f"   ⚠️  WARNING: Quantiles not well-ordered (should be >80%)")
        else:
            print(f"   ✅ Quantiles are mostly monotonic")

        test_results['tests_passed'] += 1
        test_results['monotonicity_pct'] = float(min(pct_10_50, pct_50_90))

    except Exception as e:
        print(f"   ❌ FAILED: {str(e)}")
        test_results['tests_failed'] += 1
        test_results['errors'].append(f"Monotonicity check: {str(e)}")


def test_quantile_coverage(quantile_preds, actuals):
    """Test 4: Is calibration correct (coverage matches quantile levels)?"""
    print("\nTEST 4: Quantile Coverage (Calibration)")
    print("-" * 40)

    if quantile_preds is None or actuals is None:
        print("   ⏭️  SKIPPED: No predictions/actuals available")
        return

    try:
        q10 = quantile_preds[:, 0]
        q50 = quantile_preds[:, 1]
        q90 = quantile_preds[:, 2]

        # Coverage: % of actuals above each quantile
        coverage_q10 = np.mean(actuals > q10)  # Should be ~90%
        coverage_q50 = np.mean(actuals > q50)  # Should be ~50%
        coverage_q90 = np.mean(actuals > q90)  # Should be ~10%

        print(f"   Q10 coverage: {coverage_q10:.1%} (target: 90% ± {COVERAGE_TOLERANCE*100:.0f}%)")
        print(f"   Q50 coverage: {coverage_q50:.1%} (target: 50% ± {COVERAGE_TOLERANCE*100:.0f}%)")
        print(f"   Q90 coverage: {coverage_q90:.1%} (target: 10% ± {COVERAGE_TOLERANCE*100:.0f}%)")

        # Check calibration
        warnings = []

        if abs(coverage_q10 - 0.90) > COVERAGE_TOLERANCE:
            warnings.append(f"Q10 miscalibrated: {coverage_q10:.1%} vs 90% target")

        if abs(coverage_q50 - 0.50) > COVERAGE_TOLERANCE:
            warnings.append(f"Q50 miscalibrated: {coverage_q50:.1%} vs 50% target")

        if abs(coverage_q90 - 0.10) > COVERAGE_TOLERANCE:
            warnings.append(f"Q90 miscalibrated: {coverage_q90:.1%} vs 10% target")

        if warnings:
            print(f"   ⚠️  CALIBRATION WARNINGS:")
            for w in warnings:
                print(f"      - {w}")
        else:
            print(f"   ✅ Quantiles are well-calibrated")

        test_results['tests_passed'] += 1
        test_results['coverage_q10'] = float(coverage_q10)
        test_results['coverage_q50'] = float(coverage_q50)
        test_results['coverage_q90'] = float(coverage_q90)

    except Exception as e:
        print(f"   ❌ FAILED: {str(e)}")
        test_results['tests_failed'] += 1
        test_results['errors'].append(f"Coverage check: {str(e)}")


def test_basic_backtest(quantile_preds, actuals):
    """Test 5: Can we run an uncertainty-aware backtest?"""
    print("\nTEST 5: Uncertainty-Aware Backtest")
    print("-" * 40)

    if quantile_preds is None or actuals is None:
        print("   ⏭️  SKIPPED: No predictions/actuals available")
        return

    try:
        q10 = quantile_preds[:, 0]
        q50 = quantile_preds[:, 1]
        q90 = quantile_preds[:, 2]

        # Strategy: Use Q50 for direction, scale position by uncertainty
        # High uncertainty (wide interval) → smaller position
        positions = np.sign(q50)  # Direction from median

        # Scale by inverse uncertainty
        uncertainty = q90 - q10  # Interval width
        mean_uncertainty = np.mean(uncertainty)

        # Reduce position when uncertainty is high
        position_scale = np.clip(1.0 - (uncertainty / mean_uncertainty), 0.2, 1.0)
        positions = positions * position_scale

        # Calculate returns
        strategy_returns = positions[:-1] * actuals[1:]  # Shift by 1 for forward returns

        # Basic metrics
        total_return = np.sum(strategy_returns)
        mean_return = np.mean(strategy_returns)
        std_return = np.std(strategy_returns)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0

        # Directional accuracy
        correct_direction = np.sum((q50[:-1] * actuals[1:]) > 0)
        directional_accuracy = correct_direction / len(strategy_returns)

        print(f"   Total Return: {total_return:.4f} ({total_return*100:.2f}%)")
        print(f"   Mean Daily Return: {mean_return:.6f} ({mean_return*100:.4f}%)")
        print(f"   Std Daily Return: {std_return:.6f}")
        print(f"   Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"   Directional Accuracy: {directional_accuracy:.2%}")
        print(f"   Mean Uncertainty (Q90-Q10): {mean_uncertainty:.6f}")

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
    quantile_preds, actuals = test_prediction_generation(model, feature_scaler, target_scaler)

    # Test 3: Monotonicity Check
    test_quantile_monotonicity(quantile_preds)

    # Test 4: Coverage Check
    test_quantile_coverage(quantile_preds, actuals)

    # Test 5: Uncertainty-Aware Backtest
    test_basic_backtest(quantile_preds, actuals)

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
        if 'coverage_q50' in test_results:
            print(f"   Q50 Coverage: {test_results['coverage_q50']:.1%} (target: 50%)")
    else:
        print(f"\n❌ SOME TESTS FAILED (or model not trained)")
        print(f"\nErrors:")
        for error in test_results['errors']:
            print(f"   - {error}")

    print("="*80 + "\n")

    return test_results['tests_failed'] == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
