"""
Validate feature consistency across the repository components.

Checks performed:
- Canonical feature list length matches EXPECTED_FEATURE_COUNT
- Cache metadata per-symbol in `saved_models/` via `DataCacheManager.get_cache_info()`
- Saved model directories under `saved_models/` for `feature_columns.pkl` (or fallback feature files)

Usage:
    python scripts/validate_feature_consistency.py --symbols AAPL TSLA HOOD

This script is best run from the repository root.
"""

import sys
from pathlib import Path
import argparse
import pickle
import json
import csv

# Ensure the `python-ai-service` package directory is on sys.path.
# When this script is located at `python-ai-service/scripts/...`,
# `Path(__file__).parents[1]` resolves to the `python-ai-service` directory.
# Add that directory directly to `sys.path` so imports like
# `from data.feature_engineer import ...` resolve regardless of CWD.
python_ai_service_path = Path(__file__).resolve().parents[1]
if str(python_ai_service_path) not in sys.path:
    sys.path.insert(0, str(python_ai_service_path))

try:
    from data.feature_engineer import get_feature_columns, EXPECTED_FEATURE_COUNT
    from data.cache_manager import DataCacheManager
except Exception as e:
    print(f"Failed to import project modules: {e}")
    raise


def check_canonical():
    canonical_features = get_feature_columns(include_sentiment=True)
    print(f"Canonical Feature Count: {len(canonical_features)}")
    print(f"Expected: {EXPECTED_FEATURE_COUNT}")
    if len(canonical_features) != EXPECTED_FEATURE_COUNT:
        print("\n⚠️  Canonical feature count mismatch:")
        print(f"   - get_feature_columns(include_sentiment=True) returned {len(canonical_features)} features")
        print(f"   - EXPECTED_FEATURE_COUNT = {EXPECTED_FEATURE_COUNT}")
        # Provide helpful diagnostic: list extras/shortage
        if len(canonical_features) > EXPECTED_FEATURE_COUNT:
            print(f"   - The canonical list contains {len(canonical_features) - EXPECTED_FEATURE_COUNT} extra features. Consider reconciling get_sentiment_feature_columns() or updating EXPECTED_FEATURE_COUNT.")
        else:
            print(f"   - The canonical list is missing {EXPECTED_FEATURE_COUNT - len(canonical_features)} features. Check get_sentiment_feature_columns() and NEW_FEATURES.")
        # Continue execution to collect other diagnostics
    return canonical_features


def check_cache(symbols):
    # Ensure DataCacheManager points to the project's cache directory
    project_cache_dir = Path(__file__).resolve().parents[1] / 'cache'
    cache = DataCacheManager(cache_dir=str(project_cache_dir))
    cache_results = {}
    for symbol in symbols:
        info = cache.get_cache_info(symbol)
        if info:
            feature_count = info.get('feature_count')
            print(f"{symbol}: Cached {feature_count} features (expected {EXPECTED_FEATURE_COUNT})")
            if feature_count != EXPECTED_FEATURE_COUNT:
                print(f"  ❌ MISMATCH - Cache needs refresh for {symbol}")
            cache_results[symbol] = feature_count
        else:
            print(f"{symbol}: No cache found")
            cache_results[symbol] = None
    return cache_results


def check_saved_models():
    # Look under the python-ai-service package for saved models
    saved_dir = Path(__file__).resolve().parents[1] / 'saved_models'
    results = {}
    if not saved_dir.exists():
        print("No saved_models/ directory found.")
        return results

    # First, scan top-level files for any feature pickles we can use
    top_level_feature_map = {}
    for p in saved_dir.iterdir():
        if p.is_file() and 'feature' in p.name.lower() and p.suffix.lower() in ('.pkl', '.json', '.csv'):
            try:
                if p.suffix.lower() == '.pkl':
                    candidate = pickle.load(open(p, 'rb'))
                    if isinstance(candidate, (list, tuple)) and all(isinstance(x, str) for x in candidate):
                        top_level_feature_map[p.stem] = (list(candidate), p)
                elif p.suffix.lower() == '.json':
                    candidate = json.load(open(p, 'r'))
                    if isinstance(candidate, list) and all(isinstance(x, str) for x in candidate):
                        top_level_feature_map[p.stem] = (list(candidate), p)
                elif p.suffix.lower() == '.csv':
                    # Heuristic: CSV with single-column list of features
                    cols = [r[0] for r in csv.reader(open(p, 'r', encoding='utf-8'))]
                    top_level_feature_map[p.stem] = (cols, p)
            except Exception:
                continue

    # Now process directories and try to match them to internal or top-level feature files
    for symbol_dir in sorted(saved_dir.iterdir()):
        # Handle directories (model folders)
        # Special-case: if this is the `tft` directory, inspect per-symbol subfolders
        if symbol_dir.is_dir() and symbol_dir.name.lower() == 'tft':
            # tft/ may contain per-symbol directories with dataset parameters or prepared pickles
            for sub in sorted(symbol_dir.iterdir()):
                if not sub.is_dir():
                    continue
                loaded_cols = None
                feature_file = None
                # Try dataset_parameters.pkl first
                ds_params = sub / 'dataset_parameters.pkl'
                if ds_params.exists():
                    try:
                        dp = pickle.load(open(ds_params, 'rb'))
                        # Look for a value that is a list of strings
                        if isinstance(dp, dict):
                            for k, v in dp.items():
                                if isinstance(v, (list, tuple)) and all(isinstance(x, str) for x in v):
                                    loaded_cols = list(v)
                                    feature_file = ds_params
                                    break
                    except Exception:
                        pass

                # Fallback: look for a prepared pickle (e.g., AAPL_prepared.pkl) inside sub
                if loaded_cols is None:
                    for p in sub.glob('*prepared*.pkl'):
                        try:
                            df = pickle.load(open(p, 'rb'))
                            # If this is a DataFrame-like object with columns attribute
                            cols = None
                            if hasattr(df, 'columns'):
                                cols = list(df.columns)
                            elif isinstance(df, dict) and 'columns' in df:
                                cols = list(df['columns'])
                            if cols:
                                # Exclude OHLCV/Date columns
                                excluded = {'Open', 'High', 'Low', 'Close', 'Volume', 'Date'}
                                loaded_cols = [c for c in cols if c not in excluded]
                                feature_file = p
                                break
                        except Exception:
                            continue

                # If still not found, try any feature pickles inside sub
                if loaded_cols is None:
                    for p in sub.rglob('*feature*.pkl'):
                        try:
                            candidate = pickle.load(open(p, 'rb'))
                            if isinstance(candidate, (list, tuple)) and all(isinstance(x, str) for x in candidate):
                                loaded_cols = list(candidate)
                                feature_file = p
                                break
                        except Exception:
                            continue

                if loaded_cols is not None:
                    status = "✅" if len(loaded_cols) == EXPECTED_FEATURE_COUNT else "❌"
                    print(f"{status} tft/{sub.name}: {len(loaded_cols)} features (from {feature_file.name})")
                    if len(loaded_cols) != EXPECTED_FEATURE_COUNT:
                        print(f"  --> MISMATCH: Missing: {set(get_feature_columns(True)) - set(loaded_cols)}")
                    results[f"tft/{sub.name}"] = {'count': len(loaded_cols), 'file': str(feature_file)}
                else:
                    print(f"⚠️  tft/{sub.name}: No feature_columns or prepared feature file found")
                    results[f"tft/{sub.name}"] = {'count': None, 'file': None}

            # done handling tft
            continue

        if symbol_dir.is_dir():
            feature_file = symbol_dir / 'feature_columns.pkl'
            found = False
            loaded_cols = None
            if feature_file.exists():
                try:
                    loaded_cols = pickle.load(open(feature_file, 'rb'))
                    found = True
                except Exception as e:
                    print(f"Failed to load {feature_file}: {e}")
            else:
                # Look for feature pickles inside the folder
                for p in symbol_dir.glob('*feature*.pkl'):
                    try:
                        candidate = pickle.load(open(p, 'rb'))
                        if isinstance(candidate, (list, tuple)) and all(isinstance(x, str) for x in candidate):
                            loaded_cols = list(candidate)
                            found = True
                            feature_file = p
                            break
                    except Exception:
                        continue

            # If not found inside the folder, try matching a top-level feature file by name prefix
            if not found:
                # Prefer matching by removing trailing `_model` suffix
                base_name = symbol_dir.name
                if base_name.endswith('_model'):
                    base_key = base_name[:-6]
                else:
                    base_key = base_name

                # Heuristic 1: exact prefix match
                for k, (cols, p) in top_level_feature_map.items():
                    if k.startswith(base_key) or k.startswith(base_key.replace('_model', '')):
                        loaded_cols = cols
                        feature_file = p
                        found = True
                        break

                # Heuristic 2: map binary features to buy/sell classifier folders
                if not found and ('is_buy' in base_name or 'is_sell' in base_name or 'binary' in base_name):
                    # Attempt to extract ticker prefix (e.g., AAPL from AAPL_is_buy...)
                    ticker = base_name.split('_')[0]
                    # Look for a top-level file containing '<TICKER>_binary' or '<TICKER>_binary_features' or 'binary_features'
                    for k, (cols, p) in top_level_feature_map.items():
                        if k.lower().startswith(f"{ticker.lower()}_binary") or ('binary' in k.lower() and k.lower().startswith(ticker.lower())) or (ticker.lower() in k.lower() and 'binary' in k.lower()):
                            loaded_cols = cols
                            feature_file = p
                            found = True
                            break

                # Heuristic 3: for buy/sell classifiers, fallback to any top-level features named like '<TICKER>_features'
                if not found and ('is_buy' in base_name or 'is_sell' in base_name):
                    ticker = base_name.split('_')[0]
                    for k, (cols, p) in top_level_feature_map.items():
                        if k.lower().startswith(f"{ticker.lower()}_") and 'feature' in k.lower():
                            loaded_cols = cols
                            feature_file = p
                            found = True
                            break

            if found and loaded_cols is not None:
                status = "✅" if len(loaded_cols) == EXPECTED_FEATURE_COUNT else "❌"
                print(f"{status} {symbol_dir.name}: {len(loaded_cols)} features (from {feature_file.name})")
                if len(loaded_cols) != EXPECTED_FEATURE_COUNT:
                    print(f"  --> MISMATCH: Missing: {set(get_feature_columns(True)) - set(loaded_cols)}")
                results[symbol_dir.name] = {'count': len(loaded_cols), 'file': str(feature_file)}
            else:
                print(f"⚠️  {symbol_dir.name}: No feature_columns.pkl or compatible feature file found")
                results[symbol_dir.name] = {'count': None, 'file': None}

    # Also include standalone top-level feature files that don't map to a folder
    for k, (cols, p) in top_level_feature_map.items():
        # If this stem already matched a directory, skip
        if k in results:
            continue
        status = "✅" if len(cols) == EXPECTED_FEATURE_COUNT else "❌"
        print(f"{status} {k}: {len(cols)} features (from {p.name})")
        if len(cols) != EXPECTED_FEATURE_COUNT:
            print(f"  --> MISMATCH: Missing: {set(get_feature_columns(True)) - set(cols)}")
        results[k] = {'count': len(cols), 'file': str(p)}

    return results


def generate_summary(canonical, cache_results, saved_results):
    print('\n' + '='*60)
    print('SUMMARY REPORT')
    print('='*60)
    print(f"Canonical features: {len(canonical)} (expected {EXPECTED_FEATURE_COUNT})")

    print('\nCache status:')
    for s, cnt in cache_results.items():
        status = 'OK' if cnt == EXPECTED_FEATURE_COUNT else ('MISSING' if cnt is None else 'MISMATCH')
        print(f" - {s}: {cnt} -> {status}")

    print('\nSaved models:')
    for name, info in saved_results.items():
        cnt = info.get('count')
        status = 'OK' if cnt == EXPECTED_FEATURE_COUNT else ('MISSING' if cnt is None else 'MISMATCH')
        print(f" - {name}: {cnt} -> {status} (file: {info.get('file')})")

    # Components needing update
    needs_refresh = [s for s, c in cache_results.items() if c != EXPECTED_FEATURE_COUNT]
    needs_saved_fix = [n for n, i in saved_results.items() if i.get('count') != EXPECTED_FEATURE_COUNT]

    print('\nActions suggested:')
    if needs_refresh:
        print(f" - Refresh cache for symbols: {', '.join(needs_refresh)} (use --force-refresh or clear cache)")
    else:
        print(" - Cache: all checked symbols match expected feature count")

    if needs_saved_fix:
        print(f" - Update saved model feature files for: {', '.join(needs_saved_fix)}")
    else:
        print(" - Saved model feature files: all OK or none found")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--symbols', nargs='*', default=['AAPL', 'TSLA', 'HOOD'],
                        help='Symbols to check cache for (default: AAPL TSLA HOOD)')
    args = parser.parse_args()

    canonical = check_canonical()
    cache_results = check_cache(args.symbols)
    saved_results = check_saved_models()
    generate_summary(canonical, cache_results, saved_results)


if __name__ == '__main__':
    main()
