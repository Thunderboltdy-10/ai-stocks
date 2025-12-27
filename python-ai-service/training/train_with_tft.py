"""
Training orchestrator with data caching and NaN validation.

This script centralizes data fetching via `DataCacheManager`, calls each
training step with the prepared datasets, and exposes CLI options for
cache management.

Usage examples:
  # Linux/WSL2:
  cd /path/to/ai-stocks/python-ai-service
  python training/train_with_tft.py --symbol AAPL --use-tft --tft-epochs 1 --epochs 5
  
  # Or use the shell script:
  ./run_training.sh AAPL

"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Callable, Any

# Path setup so imports work when running from `python-ai-service/`
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
PY_AI = REPO_ROOT / 'python-ai-service'

if str(PY_AI) not in sys.path:
    sys.path.insert(0, str(PY_AI))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(1, str(REPO_ROOT))


from data.cache_manager import DataCacheManager

# Try importing training entrypoints; allow graceful fallback if missing
from training.train_1d_regressor_final import train_1d_regressor
from training.train_binary_classifiers_final import train_binary_classifiers

try:
    from training.train_quantile_regressor import train_quantile_regressor
except Exception:
    train_quantile_regressor = None

try:
    from training.train_tft import train_tft
except Exception:
    train_tft = None


logger = logging.getLogger(__name__)


def _call_with_fallback(func: Callable, kwargs: Dict[str, Any]):
    """Call `func` trying several common signatures.

    Priority:
      1) func(symbol=..., raw_df=..., engineered_df=..., prepared_df=..., feature_cols=..., use_cache=..., epochs=...)
      2) func(symbol=..., prepared_df=..., feature_cols=..., use_cache=..., epochs=...)
      3) func(symbol=..., use_cache=..., epochs=...)
      4) func(symbol=..., epochs=...)
      5) func(symbol)

    Returns the function return value or raises the last exception.
    """
    # candidate signatures
    attempts = [
        ['symbol', 'raw_df', 'engineered_df', 'prepared_df', 'feature_cols', 'use_cache', 'epochs'],
        ['symbol', 'prepared_df', 'feature_cols', 'use_cache', 'epochs'],
        ['symbol', 'use_cache', 'epochs'],
        ['symbol', 'epochs'],
        ['symbol']
    ]

    last_exc = None
    for keys in attempts:
        call_kwargs = {k: kwargs[k] for k in keys if k in kwargs}
        try:
            return func(**call_kwargs)
        except TypeError as e:
            last_exc = e
            continue
    # if none worked, raise the last TypeError
    raise last_exc if last_exc is not None else RuntimeError('Failed to call function')


def train_with_tft(symbol: str, use_tft: bool = False, tft_epochs: int = 50,
                   tft_batch_size: int = 64, force_refresh: bool = False,
                   skip_cache: bool = False, clear_cache_before: bool = False,
                   epochs: int = 100,
                   skip_regressor: bool = False,
                   skip_classifiers: bool = False,
                   skip_quantile: bool = False) -> Dict:
    """Train pipeline orchestrator that fetches data once and reuses it.

    Returns a results dict describing each step outcome.
    """
    logger.info(f"=== Training Pipeline for {symbol} ===")
    logger.info(f"Cache: {'disabled' if skip_cache else 'enabled'}")
    logger.info(f"Force refresh: {force_refresh}")
    logger.info(f"Global epochs: {epochs}")

    results = {}

    cache_manager: Optional[DataCacheManager] = None
    raw_df = engineered_df = prepared_df = feature_cols = None

    if not skip_cache:
        cache_manager = DataCacheManager(cache_dir='cache', cache_lifetime_hours=24)

        if clear_cache_before:
            logger.info(f"Clearing cache for {symbol} before starting")
            cache_manager.clear_cache(symbol)

        logger.info("📊 Fetching/Loading data (once)...")
        raw_df, engineered_df, prepared_df, feature_cols = cache_manager.get_or_fetch_data(
            symbol=symbol,
            include_sentiment=True,
            force_refresh=force_refresh
        )

        results['cache_info'] = cache_manager.get_cache_info(symbol)
        results['data_shape'] = {
            'raw': getattr(raw_df, 'shape', None),
            'engineered': getattr(engineered_df, 'shape', None),
            'prepared': getattr(prepared_df, 'shape', None)
        }
        results['feature_count'] = len(feature_cols) if feature_cols is not None else None

        logger.info(f"✓ Data loaded: {getattr(prepared_df, 'shape', None)} ({results['feature_count']} features)")
    else:
        logger.info("⚠️  Cache disabled, each step will fetch independently")
        results['cache_info'] = None

    # Common kwargs to try passing into training functions
    common_kwargs = dict(
        symbol=symbol,
        raw_df=raw_df,
        engineered_df=engineered_df,
        prepared_df=prepared_df,
        feature_cols=feature_cols,
        use_cache=not skip_cache,
        epochs=epochs
    )

    # Step 1: Train Regressor
    if not skip_regressor:
        logger.info("\n=== Step 1/4: Training Regressor ===")
        try:
            res = _call_with_fallback(train_1d_regressor, common_kwargs)
            results['regressor'] = res
            logger.info("✓ Regressor training complete")
        except Exception as e:
            logger.error(f"❌ Regressor training failed: {e}")
            raise
    else:
        logger.info("Skipping regressor as requested")

    # Step 2: Train Classifiers
    if not skip_classifiers:
        logger.info("\n=== Step 2/4: Training Classifiers ===")
        try:
            res = _call_with_fallback(train_binary_classifiers, common_kwargs)
            results['classifiers'] = res
            logger.info("✓ Classifier training complete")
        except Exception as e:
            logger.error(f"❌ Classifier training failed: {e}")
            raise
    else:
        logger.info("Skipping classifiers as requested")

    # Step 3: Train Quantile Regressor (optional)
    if not skip_quantile:
        if train_quantile_regressor is not None:
            logger.info("\n=== Step 3/4: Training Quantile Regressor ===")
            try:
                res = _call_with_fallback(train_quantile_regressor, common_kwargs)
                results['quantile'] = res
                logger.info("✓ Quantile regressor training complete")
            except Exception as e:
                logger.warning(f"⚠️  Quantile regressor training failed: {e}")
                results['quantile'] = None
        else:
            logger.info("⚠️  Quantile regressor module not available, skipping")
    else:
        logger.info("Skipping quantile regressor as requested")

    # Step 4: Train TFT (optional)
    if use_tft:
        logger.info("\n=== Step 4/4: Training TFT ===")
        if train_tft is None:
            logger.warning("⚠️  TFT training requested but `train_tft` import failed")
            logger.warning("   Install with: pip install pytorch-forecasting torch pytorch-lightning")
            results['tft'] = None
        else:
            try:
                tft_kwargs = dict(
                    symbol=symbol,
                    epochs=tft_epochs,
                    batch_size=tft_batch_size,
                    use_cache=not skip_cache,
                    raw_df=raw_df,
                    engineered_df=engineered_df,
                    prepared_df=prepared_df,
                    feature_cols=feature_cols
                )
                res = _call_with_fallback(train_tft, tft_kwargs)
                results['tft'] = res
                logger.info("✓ TFT training complete")
            except Exception as e:
                logger.error(f"❌ TFT training failed: {e}")
                import traceback
                traceback.print_exc()
                results['tft'] = None
    else:
        logger.info("⚠️  TFT training not requested, skipping")

    logger.info(f"\n✓ Training pipeline complete for {symbol}")
    return results


def _print_cache_info(cache_manager: DataCacheManager, symbol: str):
    info = cache_manager.get_cache_info(symbol)
    if info:
        print(f"\n=== Cache Info for {symbol} ===")
        print(f"Timestamp: {info['timestamp']}")
        print(f"Age hours: {info['age_hours']:.2f}")
        print(f"Valid: {info['is_valid']}")
        print(f"Prepared shape: {info.get('prepared_shape')}")
        print(f"Features: {info['feature_count']}")
        print(f"Path: {info['cache_path']}")
    else:
        print(f"No cache found for {symbol}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train models with data caching')
    parser.add_argument('--symbol', required=True, help='Stock ticker symbol')
    parser.add_argument('--use-tft', action='store_true', help='Train TFT model')
    parser.add_argument('--tft-epochs', type=int, default=50, help='TFT training epochs')
    parser.add_argument('--tft-batch-size', type=int, default=64, help='TFT batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Epochs for standard models (regressor, classifiers, quantile)')
    parser.add_argument('--force-refresh', action='store_true', help='Force data refetch')
    parser.add_argument('--skip-cache', action='store_true', help='Disable caching entirely')
    parser.add_argument('--clear-cache', action='store_true', help='Clear cache before training')
    parser.add_argument('--cache-info', action='store_true', help='Show cache info and exit')
    
    # Skip flags
    parser.add_argument('--skip-regressor', action='store_true', help='Skip regressor training')
    parser.add_argument('--skip-classifiers', action='store_true', help='Skip classifier training')
    parser.add_argument('--skip-quantile', action='store_true', help='Skip quantile training')

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    cache_manager = DataCacheManager(cache_dir='cache', cache_lifetime_hours=24)

    if args.cache_info:
        _print_cache_info(cache_manager, args.symbol)
        sys.exit(0)

    if args.clear_cache:
        cache_manager.clear_cache(args.symbol)

    results = train_with_tft(
        symbol=args.symbol,
        use_tft=args.use_tft,
        tft_epochs=args.tft_epochs,
        tft_batch_size=args.tft_batch_size,
        force_refresh=args.force_refresh,
        skip_cache=args.skip_cache,
        clear_cache_before=args.clear_cache,
        epochs=args.epochs,
        skip_regressor=args.skip_regressor,
        skip_classifiers=args.skip_classifiers,
        skip_quantile=args.skip_quantile
    )

    print("\n=== Training Results ===")
    for step, result in results.items():
        if step in ('cache_info', 'data_shape', 'feature_count'):
            continue
        status = "✓" if result else "❌"
        print(f"{status} {step}: {result}")
