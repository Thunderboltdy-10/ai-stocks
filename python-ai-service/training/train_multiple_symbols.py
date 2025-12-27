"""
Parallel multi-symbol training harness.

Usage (from repo root):
  python python-ai-service/training/train_multiple_symbols.py AAPL TSLA HOOD --parallel

The script will:
 - Train regressor and binary classifiers per symbol (using existing training functions)
 - Use multiprocessing to run symbols in parallel (Windows-safe spawn context)
 - Collect success/failure status and available validation metrics
 - Save a combined CSV/JSON report under `saved_models/multi_train_report_<ts>.csv` and `.json`
"""

import argparse
import json
import multiprocessing
from multiprocessing import get_context
import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Tuple
import time
import traceback
import csv

# Ensure repo and python-ai-service on path for worker imports
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))
python_ai_service_path = repo_root / 'python-ai-service'
if str(python_ai_service_path) not in sys.path:
    sys.path.insert(0, str(python_ai_service_path))


def train_single_symbol(symbol: str, epochs: int = 50, batch_size: int = 512) -> Dict[str, Any]:
    """Train regressor + classifiers for a single symbol.

    Returns a dict with keys: symbol, success (bool), error (str|None), metrics (dict)
    """
    out = {
        'symbol': symbol,
        'success': False,
        'error': None,
        'metrics': {},
        'elapsed_sec': None
    }
    start = time.time()
    try:
        # Local imports inside worker to avoid pickling issues
        from training.train_1d_regressor_final import train_1d_regressor
        from training.train_binary_classifiers_final import train_binary_classifiers
        # Train regressor (use_cache=True)
        # Note: train_1d_regressor prints and saves artifacts; return value is Keras model
        model = train_1d_regressor(symbol=symbol, epochs=epochs, batch_size=batch_size, use_cache=True)

        # After regressor training, attempt to load regressor metadata saved to disk
        reg_meta = None
        try:
            import pickle
            meta_path = Path('saved_models') / f'{symbol}_1d_regressor_final_metadata.pkl'
            if meta_path.exists():
                with open(meta_path, 'rb') as f:
                    reg_meta = pickle.load(f)
                    out['metrics']['regressor_metadata'] = reg_meta
        except Exception as e:
            out['metrics']['regressor_metadata_error'] = str(e)

        # Train classifiers
        buy_model, sell_model, scaler, cls_meta, buy_cal, sell_cal = train_binary_classifiers(
            symbol=symbol, epochs=epochs, batch_size=batch_size, use_cache=True
        )
        out['metrics']['classifier_metadata'] = cls_meta

        # Collect a few summary metrics if present
        # Prefer explicit 'val_sharpe' if present, else fall back to val_sortino
        sharpe = None
        if isinstance(reg_meta, dict):
            if 'val_sharpe' in reg_meta:
                sharpe = reg_meta.get('val_sharpe')
            elif 'val_sortino' in reg_meta:
                sharpe = reg_meta.get('val_sortino')
        if isinstance(cls_meta, dict):
            # classifiers have advanced_metrics and val_f1
            out['metrics']['buy_val_f1'] = cls_meta.get('val_f1', {}).get('buy')
            out['metrics']['sell_val_f1'] = cls_meta.get('val_f1', {}).get('sell')
        out['metrics']['approx_sharpe'] = sharpe

        out['success'] = True
    except Exception as e:
        out['error'] = str(e)
        out['traceback'] = traceback.format_exc()
        out['success'] = False
    finally:
        out['elapsed_sec'] = time.time() - start
    return out


def train_single_symbol_wrapper(args_tuple: Tuple[str, int, int]) -> Dict[str, Any]:
    symbol, epochs, batch_size = args_tuple
    try:
        return train_single_symbol(symbol, epochs=epochs, batch_size=batch_size)
    except Exception as e:
        return {
            'symbol': symbol,
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc(),
            'metrics': {}
        }


def run_parallel(symbols: List[str], n_workers: int, epochs: int, batch_size: int = 512) -> List[Dict[str, Any]]:
    ctx = get_context('spawn') if os.name == 'nt' else get_context('fork')
    pool = ctx.Pool(processes=n_workers)
    try:
        args = [(s.upper(), epochs, batch_size) for s in symbols]
        results = pool.map(train_single_symbol_wrapper, args)
    finally:
        pool.close()
        pool.join()
    return results


def run_sequential(symbols: List[str], epochs: int, batch_size: int = 512) -> List[Dict[str, Any]]:
    results = []
    for s in symbols:
        results.append(train_single_symbol(s.upper(), epochs=epochs, batch_size=batch_size))
    return results


def summarize_results(results: List[Dict[str, Any]], out_dir: Path) -> Dict[str, Any]:
    success = sum(1 for r in results if r.get('success'))
    failure = len(results) - success
    # average sharpe over available metrics
    sharpe_vals = [r.get('metrics', {}).get('approx_sharpe') for r in results if r.get('metrics', {}).get('approx_sharpe') is not None]
    sharpe_vals = [float(v) for v in sharpe_vals if v is not None]
    avg_sharpe = float(sum(sharpe_vals) / len(sharpe_vals)) if len(sharpe_vals) > 0 else None

    failed_symbols = [{ 'symbol': r.get('symbol'), 'error': r.get('error')} for r in results if not r.get('success')]

    summary = {
        'total': len(results),
        'success': success,
        'failure': failure,
        'avg_sharpe': avg_sharpe,
        'failed': failed_symbols,
        'results': results
    }

    ts = time.strftime('%Y%m%d_%H%M%S')
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f'multi_train_report_{ts}.csv'
    json_path = out_dir / f'multi_train_report_{ts}.json'

    # Write CSV flat summary per symbol
    with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.writer(cf)
        header = ['symbol', 'success', 'error', 'elapsed_sec', 'approx_sharpe']
        writer.writerow(header)
        for r in results:
            writer.writerow([
                r.get('symbol'),
                r.get('success'),
                r.get('error') or '',
                f"{r.get('elapsed_sec'):.1f}" if r.get('elapsed_sec') is not None else '',
                r.get('metrics', {}).get('approx_sharpe') if r.get('metrics') else ''
            ])

    with open(json_path, 'w', encoding='utf-8') as jf:
        json.dump(summary, jf, indent=2)

    return summary


def main():
    parser = argparse.ArgumentParser(description='Parallel training for multiple symbols')
    parser.add_argument('symbols', nargs='+', help='List of symbols to train')
    parser.add_argument('--parallel', action='store_true', help='Run training in parallel')
    parser.add_argument('--workers', type=int, default=None, help='Number of parallel workers')
    parser.add_argument('--epochs', type=int, default=50, help='Epochs to train per model')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size per training')
    parser.add_argument('--out-dir', type=str, default='saved_models', help='Directory to save combined report')

    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols]
    n_workers = args.workers if args.workers is not None else min(multiprocessing.cpu_count(), len(symbols))
    n_workers = max(1, n_workers)

    print(f"Running multi-symbol training for: {symbols}")
    print(f"Parallel: {args.parallel} | workers: {n_workers} | epochs: {args.epochs} | batch_size: {args.batch_size}")

    start_all = time.time()
    if args.parallel:
        results = run_parallel(symbols, n_workers, args.epochs, args.batch_size)
    else:
        results = run_sequential(symbols, args.epochs, args.batch_size)
    total_time = time.time() - start_all

    out_dir = Path(args.out_dir)
    summary = summarize_results(results, out_dir)
    summary['total_elapsed_sec'] = total_time

    print('\n=== Multi-Training Summary ===')
    print(f"Total symbols: {summary['total']}")
    print(f"Success: {summary['success']} | Failure: {summary['failure']}")
    if summary['avg_sharpe'] is not None:
        print(f"Avg approximate Sharpe (from metadata): {summary['avg_sharpe']:.3f}")
    else:
        print('Avg approximate Sharpe: N/A (no sharpe metadata present)')
    if summary['failed']:
        print('Failed symbols:')
        for f in summary['failed']:
            print(f"  {f['symbol']}: {f['error']}")

    print(f"Detailed report saved to: {out_dir}")
    print(f"Total elapsed time: {total_time/60:.1f} minutes")


if __name__ == '__main__':
    main()
