"""
Force-refresh cache for a set of symbols and validate feature counts (147 canonical features).

Usage:
  .\python-ai-service\venv\Scripts\python.exe python-ai-service\scripts\refresh_cache_for_147.py AAPL TSLA
  .\python-ai-service\venv\Scripts\python.exe python-ai-service\scripts\refresh_cache_for_147.py --dry-run AAPL

Default symbols: AAPL, TSLA, HOOD
"""

import sys
import argparse
from pathlib import Path

# Ensure package imports work when running from repo root
project_pkg = Path(__file__).resolve().parents[1]
if str(project_pkg) not in sys.path:
    sys.path.insert(0, str(project_pkg))

from data.cache_manager import DataCacheManager
from data.feature_engineer import EXPECTED_FEATURE_COUNT


def refresh_symbols(symbols, dry_run: bool = False):
    cache = DataCacheManager()

    for symbol in symbols:
        symbol = symbol.upper()
        print('\n' + '=' * 60)
        print(f"Refreshing cache for {symbol}")
        print('=' * 60)

        info = cache.get_cache_info(symbol)
        if info:
            print(f"Current cache: {info.get('feature_count')} features (age: {info.get('age_hours'):.1f}h)")
        else:
            print("No cache present (will fetch if not dry-run)")

        if dry_run:
            print("--dry-run: skipping clear and fetch")
            continue

        # Clear cache for symbol
        try:
            cache.clear_cache(symbol)
            print(f"✓ Cleared cache for {symbol}")
        except Exception as e:
            print(f"Failed to clear cache for {symbol}: {e}")

        # Force refresh (this will fetch OHLCV + news and re-engineer features)
        try:
            raw_df, eng_df, prep_df, feature_cols = cache.get_or_fetch_data(symbol, include_sentiment=True, force_refresh=True)
            print(f"✓ Refreshed cache: {len(feature_cols)} features")
            print(f"  Shapes: raw={raw_df.shape}, engineered={eng_df.shape}, prepared={prep_df.shape}")

            if len(feature_cols) != EXPECTED_FEATURE_COUNT:
                print(f"  ❌ WARNING: Expected {EXPECTED_FEATURE_COUNT}, got {len(feature_cols)}")
            else:
                print(f"  ✅ Feature count validated ({EXPECTED_FEATURE_COUNT})")
        except Exception as e:
            print(f"Failed to refresh cache for {symbol}: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Refresh caches and validate feature counts (147 canonical features)')
    parser.add_argument('symbols', nargs='*', help='Symbols to refresh (default: AAPL TSLA HOOD)')
    parser.add_argument('--dry-run', action='store_true', help='Preview actions without clearing or fetching')
    args = parser.parse_args()

    symbols = [s.upper() for s in args.symbols] if args.symbols else ['AAPL', 'TSLA', 'HOOD']

    refresh_symbols(symbols, dry_run=args.dry_run)
