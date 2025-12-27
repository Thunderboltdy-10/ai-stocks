#!/usr/bin/env python3
"""Quick test to verify GBM models load for all symbols."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from inference.load_gbm_models import load_gbm_models

def test_symbol(symbol):
    print(f"=== Testing {symbol} ===")
    bundle, metadata = load_gbm_models(symbol)

    if bundle is None:
        print(f"  ❌ FAILED: {metadata.get('errors', [])}")
        return False

    has_lgb = bundle.has_lgb()
    has_xgb = bundle.has_xgb()
    print(f"  LightGBM loaded: {has_lgb}")
    print(f"  XGBoost loaded: {has_xgb}")
    print(f"  Feature count: {len(bundle.feature_columns)}")
    print(f"  ✅ Models OK for {symbol}")
    return True

if __name__ == '__main__':
    symbols = ['AAPL', 'GOOGL', 'NVDA', 'META', 'SPY', 'KO', 'ASML', 'IWM']

    results = {}
    for symbol in symbols:
        results[symbol] = test_symbol(symbol)
        print()

    print("="*50)
    print("SUMMARY:")
    loaded = sum(results.values())
    print(f"Models loaded: {loaded}/{len(symbols)} symbols ✅")

    if loaded < len(symbols):
        print("\nFailed symbols:")
        for sym, ok in results.items():
            if not ok:
                print(f"  - {sym}")
