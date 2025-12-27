"""
F4 Model Validation: Binary Classifiers with Calibration Metrics

Tests classifier models and evaluates calibration quality using Brier score.

Key Metrics:
- Brier Score: Measures calibration quality (lower is better, <0.25 is good)
- Prediction distribution: Check for collapsed confidence
- Sharpe ratio: Strategy performance
- Signal balance: BUY/SELL/neutral distribution

Author: AI-Stocks F4 Validation - Agent 2
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
import json
from datetime import datetime
from sklearn.metrics import brier_score_loss

print("\n" + "="*80)
print("F4 CLASSIFIER VALIDATION: CALIBRATION QUALITY ASSESSMENT")
print("="*80 + "\n")

# Test configuration
TEST_SYMBOL = os.environ.get('TEST_SYMBOL', 'AAPL')
MIN_POSITIVE_SIGNALS = 0.10
BRIER_GOOD_THRESHOLD = 0.25
BRIER_CRITICAL_THRESHOLD = 0.40

test_results = {
    'symbol': TEST_SYMBOL,
    'timestamp': datetime.now().isoformat(),
    'status': 'UNKNOWN',
    'tests_passed': 0,
    'tests_failed': 0,
    'errors': [],
    'issues': []
}


def calculate_brier_score(y_true, y_prob):
    """Calculate Brier score for calibration quality.

    Brier score = mean((y_true - y_prob)^2)

    Interpretation:
        < 0.25: Good calibration
        0.25-0.40: Moderate calibration issues
        > 0.40: Poor calibration (critical)
    """
    return brier_score_loss(y_true, y_prob)


def analyze_prediction_distribution(probs, name=""):
    """Analyze prediction distribution for collapsed confidence."""
    print(f"\n   {name} Prediction Distribution:")
    print(f"      Min:     {probs.min():.6f}")
    print(f"      Max:     {probs.max():.6f}")
    print(f"      Mean:    {probs.mean():.6f}")
    print(f"      Median:  {np.median(probs):.6f}")
    print(f"      Std:     {probs.std():.6f}")

    # Check for collapsed confidence (all predictions in narrow range)
    range_width = probs.max() - probs.min()
    if range_width < 0.2:
        print(f"      ⚠️  WARNING: Collapsed confidence (range={range_width:.3f})")
        return True

    # Check if predictions are all near 0.3-0.4 (typical collapse pattern)
    in_collapse_range = np.sum((probs >= 0.3) & (probs <= 0.4)) / len(probs)
    if in_collapse_range > 0.8:
        print(f"      ⚠️  WARNING: {in_collapse_range:.1%} predictions in 0.3-0.4 range")
        return True

    return False


def test_model_loading():
    """Test 1: Load classifier models and calibration"""
    print("TEST 1: Model Loading & Calibration")
    print("-" * 40)

    try:
        from utils.model_paths import ModelPaths, get_legacy_classifier_paths
        from tensorflow.keras.models import load_model
        from utils.losses import register_custom_objects

        register_custom_objects()
        paths = ModelPaths(TEST_SYMBOL)

        # Load BUY classifier
        buy_model = None
        buy_calibrated = None
        if paths.classifiers.buy_model.exists():
            buy_model = load_model(str(paths.classifiers.buy_model))
            print(f"   ✅ BUY classifier loaded")

            # Try to load calibrated version
            if paths.classifiers.buy_calibrated.exists():
                with open(paths.classifiers.buy_calibrated, 'rb') as f:
                    buy_calibrated = pickle.load(f)
                print(f"   ✅ BUY calibrated model loaded")
        else:
            # Try legacy path
            legacy_paths = get_legacy_classifier_paths(TEST_SYMBOL)
            if legacy_paths and legacy_paths.get('buy_model') and Path(legacy_paths['buy_model']).exists():
                buy_model = load_model(legacy_paths['buy_model'])
                print(f"   ✅ BUY classifier loaded (legacy)")

                if legacy_paths.get('buy_calibrated') and Path(legacy_paths['buy_calibrated']).exists():
                    with open(legacy_paths['buy_calibrated'], 'rb') as f:
                        buy_calibrated = pickle.load(f)
                    print(f"   ✅ BUY calibrated model loaded (legacy)")
            else:
                print(f"   ⚠️  WARNING: No BUY classifier found")

        # Load SELL classifier
        sell_model = None
        sell_calibrated = None
        if paths.classifiers.sell_model.exists():
            sell_model = load_model(str(paths.classifiers.sell_model))
            print(f"   ✅ SELL classifier loaded")

            if paths.classifiers.sell_calibrated.exists():
                with open(paths.classifiers.sell_calibrated, 'rb') as f:
                    sell_calibrated = pickle.load(f)
                print(f"   ✅ SELL calibrated model loaded")
        else:
            if legacy_paths and legacy_paths.get('sell_model') and Path(legacy_paths['sell_model']).exists():
                sell_model = load_model(legacy_paths['sell_model'])
                print(f"   ✅ SELL classifier loaded (legacy)")

                if legacy_paths.get('sell_calibrated') and Path(legacy_paths['sell_calibrated']).exists():
                    with open(legacy_paths['sell_calibrated'], 'rb') as f:
                        sell_calibrated = pickle.load(f)
                    print(f"   ✅ SELL calibrated model loaded (legacy)")
            else:
                print(f"   ⚠️  WARNING: No SELL classifier found")

        if buy_model is None and sell_model is None:
            raise FileNotFoundError(f"No classifier models found for {TEST_SYMBOL}")

        # Load feature scaler
        feature_scaler = None
        if paths.classifiers.feature_scaler.exists():
            feature_scaler = pickle.load(open(paths.classifiers.feature_scaler, 'rb'))
            print(f"   ✅ Feature scaler loaded")
        elif legacy_paths and legacy_paths.get('feature_scaler'):
            feature_scaler = pickle.load(open(Path(legacy_paths['feature_scaler']), 'rb'))
            print(f"   ✅ Feature scaler loaded (legacy)")

        test_results['tests_passed'] += 1
        test_results['buy_available'] = buy_model is not None
        test_results['sell_available'] = sell_model is not None
        test_results['buy_calibrated_available'] = buy_calibrated is not None
        test_results['sell_calibrated_available'] = sell_calibrated is not None

        return buy_model, sell_model, buy_calibrated, sell_calibrated, feature_scaler

    except Exception as e:
        print(f"   ❌ FAILED: {str(e)}")
        test_results['tests_failed'] += 1
        test_results['errors'].append(f"Model Loading: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None


def test_prediction_and_calibration(buy_model, sell_model, buy_calibrated, sell_calibrated, feature_scaler):
    """Test 2: Generate predictions and evaluate calibration quality"""
    print("\nTEST 2: Prediction Generation & Calibration Quality")
    print("-" * 40)

    if buy_model is None and sell_model is None:
        print("   ⏭️  SKIPPED: No models loaded")
        return None, None, None, None, None

    try:
        from data.data_fetcher import fetch_stock_data
        from data.feature_engineer import engineer_features
        from data.target_engineering import prepare_training_data

        # Fetch data
        print(f"   Fetching {TEST_SYMBOL} data...")
        df = fetch_stock_data(symbol=TEST_SYMBOL, period='1y')
        print(f"   ✅ Fetched {len(df)} days of data")

        # Engineer features
        df_features = engineer_features(df, include_sentiment=True)
        print(f"   ✅ Engineered {df_features.shape[1]} features")

        # Get model parameters
        model = buy_model if buy_model is not None else sell_model
        sequence_length = model.input_shape[1]
        n_features_expected = model.input_shape[2]

        # Create sequences
        X, y = prepare_training_data(
            df_features,
            target='return_1d',
            sequence_length=sequence_length,
            train_test_split=1.0
        )

        # Use last 100 samples for testing
        X_test = X[-100:]
        y_test = y[-100:]

        print(f"   ✅ Created {len(X_test)} test sequences")

        # Scale features
        if feature_scaler is not None:
            X_test_flat = X_test.reshape(-1, X_test.shape[-1])
            X_test_scaled = feature_scaler.transform(X_test_flat).reshape(X_test.shape)
        else:
            X_test_scaled = X_test
            print(f"   ⚠️  No scaler available")

        # BUY classifier predictions
        buy_probs_raw = None
        buy_probs_calibrated = None
        buy_brier_raw = None
        buy_brier_calibrated = None

        if buy_model is not None:
            print(f"\n   === BUY CLASSIFIER ===")

            # Raw predictions
            buy_probs_raw_output = buy_model.predict(X_test_scaled, verbose=0)
            buy_probs_raw = buy_probs_raw_output[:, 1] if len(buy_probs_raw_output.shape) > 1 and buy_probs_raw_output.shape[1] > 1 else buy_probs_raw_output.flatten()

            print(f"   Raw predictions:")
            collapsed_raw = analyze_prediction_distribution(buy_probs_raw, "BUY")
            if collapsed_raw:
                test_results['issues'].append("BUY predictions show collapsed confidence")

            # Create binary labels (top 40% = BUY signal)
            buy_labels = (y_test >= np.percentile(y_test, 60)).astype(int)

            # Calculate Brier score for raw predictions
            buy_brier_raw = calculate_brier_score(buy_labels, buy_probs_raw)
            print(f"   Brier Score (raw): {buy_brier_raw:.4f}", end="")
            if buy_brier_raw < BRIER_GOOD_THRESHOLD:
                print(" ✅ (Good)")
            elif buy_brier_raw < BRIER_CRITICAL_THRESHOLD:
                print(" ⚠️  (Moderate)")
            else:
                print(" ❌ (Poor)")
                test_results['issues'].append(f"BUY Brier score {buy_brier_raw:.3f} > {BRIER_CRITICAL_THRESHOLD}")

            # Calibrated predictions
            if buy_calibrated is not None:
                buy_probs_calibrated = buy_calibrated.predict_proba(X_test_scaled)[:, 1]

                print(f"\n   Calibrated predictions:")
                collapsed_cal = analyze_prediction_distribution(buy_probs_calibrated, "BUY")
                if collapsed_cal:
                    test_results['issues'].append("BUY calibrated predictions show collapsed confidence")

                buy_brier_calibrated = calculate_brier_score(buy_labels, buy_probs_calibrated)
                print(f"   Brier Score (calibrated): {buy_brier_calibrated:.4f}", end="")
                if buy_brier_calibrated < BRIER_GOOD_THRESHOLD:
                    print(" ✅ (Good)")
                elif buy_brier_calibrated < BRIER_CRITICAL_THRESHOLD:
                    print(" ⚠️  (Moderate)")
                else:
                    print(" ❌ (Poor)")

                improvement = buy_brier_raw - buy_brier_calibrated
                print(f"   Calibration improvement: {improvement:+.4f}")
                if improvement < 0:
                    print(f"   ⚠️  WARNING: Calibration made it worse!")
                    test_results['issues'].append("BUY calibration degraded performance")

        # SELL classifier predictions
        sell_probs_raw = None
        sell_probs_calibrated = None
        sell_brier_raw = None
        sell_brier_calibrated = None

        if sell_model is not None:
            print(f"\n   === SELL CLASSIFIER ===")

            # Raw predictions
            sell_probs_raw_output = sell_model.predict(X_test_scaled, verbose=0)
            sell_probs_raw = sell_probs_raw_output[:, 1] if len(sell_probs_raw_output.shape) > 1 and sell_probs_raw_output.shape[1] > 1 else sell_probs_raw_output.flatten()

            print(f"   Raw predictions:")
            collapsed_raw = analyze_prediction_distribution(sell_probs_raw, "SELL")
            if collapsed_raw:
                test_results['issues'].append("SELL predictions show collapsed confidence")

            # Create binary labels (bottom 40% = SELL signal)
            sell_labels = (y_test <= np.percentile(y_test, 40)).astype(int)

            # Calculate Brier score
            sell_brier_raw = calculate_brier_score(sell_labels, sell_probs_raw)
            print(f"   Brier Score (raw): {sell_brier_raw:.4f}", end="")
            if sell_brier_raw < BRIER_GOOD_THRESHOLD:
                print(" ✅ (Good)")
            elif sell_brier_raw < BRIER_CRITICAL_THRESHOLD:
                print(" ⚠️  (Moderate)")
            else:
                print(" ❌ (Poor)")
                test_results['issues'].append(f"SELL Brier score {sell_brier_raw:.3f} > {BRIER_CRITICAL_THRESHOLD}")

            # Calibrated predictions
            if sell_calibrated is not None:
                sell_probs_calibrated = sell_calibrated.predict_proba(X_test_scaled)[:, 1]

                print(f"\n   Calibrated predictions:")
                collapsed_cal = analyze_prediction_distribution(sell_probs_calibrated, "SELL")
                if collapsed_cal:
                    test_results['issues'].append("SELL calibrated predictions show collapsed confidence")

                sell_brier_calibrated = calculate_brier_score(sell_labels, sell_probs_calibrated)
                print(f"   Brier Score (calibrated): {sell_brier_calibrated:.4f}", end="")
                if sell_brier_calibrated < BRIER_GOOD_THRESHOLD:
                    print(" ✅ (Good)")
                elif sell_brier_calibrated < BRIER_CRITICAL_THRESHOLD:
                    print(" ⚠️  (Moderate)")
                else:
                    print(" ❌ (Poor)")

                improvement = sell_brier_raw - sell_brier_calibrated
                print(f"   Calibration improvement: {improvement:+.4f}")
                if improvement < 0:
                    print(f"   ⚠️  WARNING: Calibration made it worse!")
                    test_results['issues'].append("SELL calibration degraded performance")

        test_results['tests_passed'] += 1
        test_results['buy_brier_raw'] = float(buy_brier_raw) if buy_brier_raw is not None else None
        test_results['buy_brier_calibrated'] = float(buy_brier_calibrated) if buy_brier_calibrated is not None else None
        test_results['sell_brier_raw'] = float(sell_brier_raw) if sell_brier_raw is not None else None
        test_results['sell_brier_calibrated'] = float(sell_brier_calibrated) if sell_brier_calibrated is not None else None

        # Use calibrated if available, otherwise raw
        buy_probs_final = buy_probs_calibrated if buy_probs_calibrated is not None else buy_probs_raw
        sell_probs_final = sell_probs_calibrated if sell_probs_calibrated is not None else sell_probs_raw

        return buy_probs_final, sell_probs_final, y_test, buy_probs_raw, sell_probs_raw

    except Exception as e:
        print(f"   ❌ FAILED: {str(e)}")
        test_results['tests_failed'] += 1
        test_results['errors'].append(f"Prediction/Calibration: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None, None, None


def test_signal_balance(buy_probs, sell_probs):
    """Test 3: Signal balance and distribution"""
    print("\nTEST 3: Signal Balance")
    print("-" * 40)

    if buy_probs is None and sell_probs is None:
        print("   ⏭️  SKIPPED: No predictions available")
        return

    try:
        if buy_probs is not None:
            pct_buy_high = np.mean(buy_probs > 0.5)
            print(f"   BUY signals (prob > 0.5): {pct_buy_high:.1%}")

            if pct_buy_high < MIN_POSITIVE_SIGNALS:
                test_results['issues'].append(f"Too few BUY signals: {pct_buy_high:.1%}")
            elif pct_buy_high > 0.90:
                test_results['issues'].append(f"Too many BUY signals: {pct_buy_high:.1%}")

            test_results['pct_buy_signals'] = float(pct_buy_high)

        if sell_probs is not None:
            pct_sell_high = np.mean(sell_probs > 0.5)
            print(f"   SELL signals (prob > 0.5): {pct_sell_high:.1%}")

            if pct_sell_high < MIN_POSITIVE_SIGNALS:
                test_results['issues'].append(f"Too few SELL signals: {pct_sell_high:.1%}")
            elif pct_sell_high > 0.90:
                test_results['issues'].append(f"Too many SELL signals: {pct_sell_high:.1%}")

            test_results['pct_sell_signals'] = float(pct_sell_high)

        print(f"   ✅ PASSED")
        test_results['tests_passed'] += 1

    except Exception as e:
        print(f"   ❌ FAILED: {str(e)}")
        test_results['tests_failed'] += 1
        test_results['errors'].append(f"Signal balance: {str(e)}")


def test_backtest(buy_probs, sell_probs, actuals):
    """Test 4: Basic backtest with Sharpe ratio"""
    print("\nTEST 4: Backtest Performance")
    print("-" * 40)

    if (buy_probs is None and sell_probs is None) or actuals is None:
        print("   ⏭️  SKIPPED: No predictions/actuals available")
        return

    try:
        # Create positions: Long if BUY > 0.7, short if SELL > 0.7
        positions = np.zeros(len(actuals))

        if buy_probs is not None and sell_probs is not None:
            positions = np.where(buy_probs > 0.7, 1.0,
                        np.where(sell_probs > 0.7, -1.0, 0.0))
        elif buy_probs is not None:
            positions = np.where(buy_probs > 0.7, 1.0, 0.0)
        elif sell_probs is not None:
            positions = np.where(sell_probs > 0.7, -1.0, 0.0)

        # Calculate returns
        strategy_returns = positions[:-1] * actuals[1:]

        # Metrics
        total_return = np.sum(strategy_returns)
        mean_return = np.mean(strategy_returns)
        std_return = np.std(strategy_returns)
        sharpe_ratio = (mean_return / std_return) * np.sqrt(252) if std_return > 0 else 0

        # Signal distribution
        pct_long = np.mean(positions > 0)
        pct_short = np.mean(positions < 0)
        pct_neutral = np.mean(positions == 0)

        print(f"   Sharpe Ratio: {sharpe_ratio:.4f}")
        print(f"   Signal Distribution:")
        print(f"      Long:    {pct_long:.1%}")
        print(f"      Short:   {pct_short:.1%}")
        print(f"      Neutral: {pct_neutral:.1%}")

        if sharpe_ratio < -2.0:
            test_results['issues'].append(f"Very negative Sharpe: {sharpe_ratio:.2f}")

        if pct_neutral > 0.95:
            test_results['issues'].append(f"Almost always neutral: {pct_neutral:.1%}")

        print(f"   ✅ PASSED")
        test_results['tests_passed'] += 1
        test_results['sharpe_ratio'] = float(sharpe_ratio)
        test_results['pct_long'] = float(pct_long)
        test_results['pct_short'] = float(pct_short)
        test_results['pct_neutral'] = float(pct_neutral)

    except Exception as e:
        print(f"   ❌ FAILED: {str(e)}")
        test_results['tests_failed'] += 1
        test_results['errors'].append(f"Backtest: {str(e)}")


def main():
    """Run all tests"""
    print(f"Testing Symbol: {TEST_SYMBOL}\n")

    # Test 1: Model Loading
    buy_model, sell_model, buy_calibrated, sell_calibrated, feature_scaler = test_model_loading()

    # Test 2: Prediction & Calibration
    buy_probs, sell_probs, actuals, buy_raw, sell_raw = test_prediction_and_calibration(
        buy_model, sell_model, buy_calibrated, sell_calibrated, feature_scaler
    )

    # Test 3: Signal Balance
    test_signal_balance(buy_probs, sell_probs)

    # Test 4: Backtest
    test_backtest(buy_probs, sell_probs, actuals)

    # Determine overall status
    if test_results['tests_failed'] > 0:
        test_results['status'] = 'FAILED'
    elif len(test_results['issues']) > 0:
        test_results['status'] = 'PASSED_WITH_ISSUES'
    else:
        test_results['status'] = 'PASSED'

    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Symbol: {test_results['symbol']}")
    print(f"Status: {test_results['status']}")
    print(f"Tests Passed: {test_results['tests_passed']}")
    print(f"Tests Failed: {test_results['tests_failed']}")

    if test_results.get('buy_brier_calibrated'):
        print(f"\nBUY Classifier:")
        print(f"  Brier Score: {test_results['buy_brier_calibrated']:.4f}")
        print(f"  Signal %: {test_results.get('pct_buy_signals', 0):.1%}")

    if test_results.get('sell_brier_calibrated'):
        print(f"\nSELL Classifier:")
        print(f"  Brier Score: {test_results['sell_brier_calibrated']:.4f}")
        print(f"  Signal %: {test_results.get('pct_sell_signals', 0):.1%}")

    if test_results.get('sharpe_ratio') is not None:
        print(f"\nBacktest:")
        print(f"  Sharpe Ratio: {test_results['sharpe_ratio']:.4f}")

    if len(test_results['issues']) > 0:
        print(f"\n⚠️  ISSUES DETECTED ({len(test_results['issues'])}):")
        for issue in test_results['issues']:
            print(f"   - {issue}")

    if len(test_results['errors']) > 0:
        print(f"\n❌ ERRORS ({len(test_results['errors'])}):")
        for error in test_results['errors']:
            print(f"   - {error}")

    print("="*80 + "\n")

    # Save results
    output_file = PROJECT_ROOT / f"f4_validation_results/{TEST_SYMBOL}_classifier_calibration.json"
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(test_results, f, indent=2)
    print(f"Results saved to: {output_file}\n")

    return test_results['status'] in ['PASSED', 'PASSED_WITH_ISSUES']


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
