#!/usr/bin/env python3
"""Test GBM inference on latest data for top symbols."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import yfinance as yf
from inference.load_gbm_models import load_gbm_models, predict_with_gbm
from data.feature_engineer import engineer_features

def test_inference(symbol):
    print(f"\n{'='*60}")
    print(f"Testing {symbol} - GBM Inference")
    print('='*60)

    # Load models
    print(f"Loading GBM models for {symbol}...")
    bundle, metadata = load_gbm_models(symbol)

    if bundle is None:
        print(f"❌ FAILED: {metadata.get('errors', [])}")
        return None

    print(f"✅ Models loaded: {bundle.get_available_models()}")
    print(f"   Features: {len(bundle.feature_columns)}")

    # Fetch latest data
    print(f"Fetching latest market data for {symbol}...")
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period='2y')  # Get 2 years of data

        if df.empty:
            print(f"❌ No data available for {symbol}")
            return None

        print(f"   Data points: {len(df)}")
        print(f"   Date range: {df.index[0].date()} to {df.index[-1].date()}")
    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None

    # Engineer features
    print(f"Engineering features...")
    try:
        # Always use include_sentiment=False to avoid API issues
        df_features = engineer_features(df, include_sentiment=False)

        if df_features.empty:
            print(f"❌ Feature engineering failed")
            return None

        print(f"   Features engineered: {df_features.shape[1]} columns, {len(df_features)} rows")

        # Check if we have the required features
        missing_features = set(bundle.feature_columns) - set(df_features.columns)
        if missing_features:
            print(f"   ⚠️  Missing {len(missing_features)} features (likely sentiment)")
            print(f"   Model expects {len(bundle.feature_columns)} features but we have {df_features.shape[1]}")
            print(f"   Skipping {symbol} - retrain model without sentiment")
            return None

        # Get the latest features using the saved feature columns
        latest_features = df_features[bundle.feature_columns].iloc[-1:].values
        print(f"   Latest feature vector shape: {latest_features.shape}")

    except Exception as e:
        print(f"❌ Error engineering features: {e}")
        import traceback
        traceback.print_exc()
        return None

    # Generate predictions
    print(f"Generating predictions...")
    try:
        # Test both models
        results = {}

        if bundle.has_lgb():
            lgb_pred = predict_with_gbm(bundle, latest_features, model='lgb')
            results['lgb'] = float(lgb_pred[0])
            print(f"   LightGBM prediction: {results['lgb']:.4f} ({results['lgb']*100:.2f}%)")

        if bundle.has_xgb():
            xgb_pred = predict_with_gbm(bundle, latest_features, model='xgb')
            results['xgb'] = float(xgb_pred[0])
            print(f"   XGBoost prediction: {results['xgb']:.4f} ({results['xgb']*100:.2f}%)")

        # Weighted average (favor LightGBM)
        if 'lgb' in results and 'xgb' in results:
            weighted = 0.7 * results['lgb'] + 0.3 * results['xgb']
            results['weighted'] = weighted
            print(f"   Weighted avg (70% LGB): {weighted:.4f} ({weighted*100:.2f}%)")
        elif 'lgb' in results:
            results['weighted'] = results['lgb']
        elif 'xgb' in results:
            results['weighted'] = results['xgb']

        # Generate position signal
        pred = results.get('weighted', 0.0)
        if pred > 0.002:
            signal = 'BUY'
            position = min(pred * 50, 1.0)  # Scale to position size
        elif pred < -0.002:
            signal = 'SELL'
            position = max(pred * 50, -0.5)  # Max 50% short
        else:
            signal = 'HOLD'
            position = 0.0

        results['signal'] = signal
        results['position'] = position

        print(f"\n   📊 SIGNAL: {signal}")
        print(f"   📈 Position: {position:.2%}")
        print(f"   🎯 Confidence: {'HIGH' if abs(pred) > 0.01 else 'MEDIUM' if abs(pred) > 0.005 else 'LOW'}")

        return results

    except Exception as e:
        print(f"❌ Error generating predictions: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == '__main__':
    # Test symbols that don't require sentiment features (147 features)
    symbols = ['GOOGL', 'NVDA', 'META', 'SPY']

    all_results = {}
    for symbol in symbols:
        result = test_inference(symbol)
        if result:
            all_results[symbol] = result

    # Summary table
    print(f"\n{'='*60}")
    print("INFERENCE SUMMARY")
    print('='*60)
    print(f"{'Symbol':<8} {'LGB Pred':<10} {'XGB Pred':<10} {'Signal':<8} {'Position':<10}")
    print('-'*60)

    for symbol, results in all_results.items():
        lgb = f"{results.get('lgb', 0)*100:+.2f}%" if 'lgb' in results else 'N/A'
        xgb = f"{results.get('xgb', 0)*100:+.2f}%" if 'xgb' in results else 'N/A'
        signal = results.get('signal', 'N/A')
        position = f"{results.get('position', 0):.2%}"

        print(f"{symbol:<8} {lgb:<10} {xgb:<10} {signal:<8} {position:<10}")

    print('='*60)
    print(f"Successfully tested: {len(all_results)}/{len(symbols)} symbols")
