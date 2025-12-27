#!/usr/bin/env python3
"""
Quick validation test for binary classifiers.
Tests model loading, prediction generation, and basic performance metrics.
"""

import sys
from pathlib import Path
import numpy as np
import pickle
import tensorflow as tf

# Setup path
CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.append(str(CURRENT_DIR))

from utils.losses import register_custom_objects

# Register custom losses
register_custom_objects()

def test_classifier_models(symbol='AAPL'):
    """Test binary classifier models."""

    print(f"\n{'='*80}")
    print(f"BINARY CLASSIFIER VALIDATION TEST - {symbol}")
    print(f"{'='*80}\n")

    # Test 1: Load models
    print("TEST 1: Model Loading")
    print("-" * 40)
    try:
        buy_model = tf.keras.models.load_model(f'saved_models/{symbol}/classifiers/buy_model.keras')
        sell_model = tf.keras.models.load_model(f'saved_models/{symbol}/classifiers/sell_model.keras')

        with open(f'saved_models/{symbol}/classifiers/feature_scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)

        with open(f'saved_models/{symbol}/classifiers/metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)

        print(f"✅ BUY model loaded: {buy_model.input_shape} -> {buy_model.output_shape}")
        print(f"✅ SELL model loaded: {sell_model.input_shape} -> {sell_model.output_shape}")
        print(f"✅ Feature scaler loaded")
        print(f"✅ Metadata loaded")
        print(f"   Sequence length: {metadata['sequence_length']}")
        print(f"   Features: {metadata['n_features']}")
        print(f"   BUY threshold: {metadata['thresholds']['buy_optimal']:.3f}")
        print(f"   SELL threshold: {metadata['thresholds']['sell_optimal']:.3f}")

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

    # Test 2: Load cached data
    print(f"\nTEST 2: Data Loading")
    print("-" * 40)
    try:
        # Load from cache pickle directly
        cache_path = Path(f'cache/{symbol}/prepared_training.pkl')
        if not cache_path.exists():
            print("❌ No cached data found")
            return False

        with open(cache_path, 'rb') as f:
            df = pickle.load(f)

        print(f"✅ Loaded {len(df)} rows from cache")
        print(f"   Columns: {len(df.columns)}")

        # Load feature columns
        with open(f'saved_models/{symbol}/classifiers/features.pkl', 'rb') as f:
            feature_cols = pickle.load(f)

        print(f"   Feature columns: {len(feature_cols)}")

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

    # Test 3: Generate predictions
    print(f"\nTEST 3: Prediction Generation")
    print("-" * 40)
    try:
        # Prepare sequences
        sequence_length = metadata['sequence_length']
        n_features = metadata['n_features']

        # Get feature data
        X_features = df[feature_cols].values
        X_scaled = scaler.transform(X_features)

        # Create sequences
        sequences = []
        for i in range(sequence_length, len(X_scaled)):
            seq = X_scaled[i-sequence_length:i]
            sequences.append(seq)

        X_seq = np.array(sequences, dtype=np.float32)

        print(f"   Created {len(X_seq)} sequences of shape {X_seq.shape}")

        # Generate predictions
        buy_probs = buy_model.predict(X_seq, verbose=0)
        sell_probs = sell_model.predict(X_seq, verbose=0)

        buy_probs = buy_probs.flatten()
        sell_probs = sell_probs.flatten()

        print(f"✅ Generated predictions")
        print(f"   BUY probabilities: mean={buy_probs.mean():.3f}, std={buy_probs.std():.3f}")
        print(f"   SELL probabilities: mean={sell_probs.mean():.3f}, std={sell_probs.std():.3f}")

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Test 4: Analyze signal distribution
    print(f"\nTEST 4: Signal Distribution")
    print("-" * 40)
    try:
        buy_threshold = metadata['thresholds']['buy_optimal']
        sell_threshold = metadata['thresholds']['sell_optimal']

        buy_signals = buy_probs > buy_threshold
        sell_signals = sell_probs > sell_threshold

        n_buy = buy_signals.sum()
        n_sell = sell_signals.sum()
        n_hold = len(buy_signals) - n_buy - n_sell

        print(f"   BUY signals: {n_buy} ({100*n_buy/len(buy_signals):.1f}%)")
        print(f"   SELL signals: {n_sell} ({100*n_sell/len(sell_signals):.1f}%)")
        print(f"   HOLD signals: {n_hold} ({100*n_hold/len(buy_signals):.1f}%)")

        # Check for variance collapse
        if buy_probs.std() < 0.01:
            print(f"   ⚠️  BUY model: Low variance (std={buy_probs.std():.4f})")
        else:
            print(f"   ✓ BUY model: Healthy variance (std={buy_probs.std():.4f})")

        if sell_probs.std() < 0.01:
            print(f"   ⚠️  SELL model: Low variance (std={sell_probs.std():.4f})")
        else:
            print(f"   ✓ SELL model: Healthy variance (std={sell_probs.std():.4f})")

        # Check probability ranges
        print(f"   BUY prob range: [{buy_probs.min():.3f}, {buy_probs.max():.3f}]")
        print(f"   SELL prob range: [{sell_probs.min():.3f}, {sell_probs.max():.3f}]")

    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

    # Test 5: Simulate simple backtest
    print(f"\nTEST 5: Simple Backtest Simulation")
    print("-" * 40)
    try:
        # Get returns
        returns = df['target_1d'].iloc[sequence_length:].values[:len(buy_signals)]

        # Create positions based on signals
        positions = np.zeros(len(buy_signals))
        positions[buy_signals] = 1.0  # Long on BUY
        positions[sell_signals] = -0.5  # Short on SELL

        # Calculate strategy returns
        strategy_returns = positions * returns

        # Metrics
        total_return = strategy_returns.sum()
        sharpe = strategy_returns.mean() / (strategy_returns.std() + 1e-8) * np.sqrt(252)
        win_rate = (strategy_returns[strategy_returns != 0] > 0).mean()
        n_trades = (positions != 0).sum()

        print(f"   Total Return: {total_return:.2%}")
        print(f"   Sharpe Ratio: {sharpe:.2f}")
        print(f"   Win Rate: {win_rate:.1%}")
        print(f"   Number of Trades: {n_trades}")

        # Check if performance is reasonable
        if sharpe > 0.5:
            print(f"   ✅ Performance is reasonable (Sharpe > 0.5)")
        else:
            print(f"   ⚠️  Performance is below expectations (Sharpe = {sharpe:.2f})")

    except Exception as e:
        print(f"❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

    print(f"\n{'='*80}")
    print("✅ ALL TESTS PASSED")
    print(f"{'='*80}\n")

    return True


if __name__ == '__main__':
    symbol = sys.argv[1] if len(sys.argv) > 1 else 'AAPL'
    success = test_classifier_models(symbol)
    sys.exit(0 if success else 1)
