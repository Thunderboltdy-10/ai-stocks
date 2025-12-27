#!/usr/bin/env python
"""
Batch Model Training Script

Trains both Regressor and GBM models for multiple symbols in sequence.
Designed for the follow-up audit to train missing symbol models.

Usage:
    python scripts/batch_train_models.py --symbols MSFT AMZN NVDA TSLA XOM JPM
    python scripts/batch_train_models.py --symbols HOOD --regressor-epochs 500  # Special case for HOOD
"""

import argparse
import subprocess
import sys
import json
import time
from pathlib import Path
from datetime import datetime

# Add project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def run_training(symbol: str, model_type: str, extra_args: list = None, log_dir: Path = None) -> dict:
    """Run training for a single symbol and model type."""
    
    if model_type == 'regressor':
        script = 'training/train_1d_regressor_final.py'
        default_args = ['--epochs', '300', '--batch-size', '64', '--seed', '42']
    elif model_type == 'gbm':
        script = 'training/train_gbm_baseline.py'
        default_args = ['--epochs', '5000', '--seed', '42']
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    cmd = ['python', script, symbol] + default_args
    if extra_args:
        cmd.extend(extra_args)
    
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} for {symbol}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Run with output capture
    result = subprocess.run(
        cmd,
        cwd=str(PROJECT_ROOT),
        capture_output=True,
        text=True,
    )
    
    elapsed = time.time() - start_time
    
    # Save log if log_dir provided
    if log_dir:
        log_file = log_dir / f'{symbol}_{model_type}_{datetime.now().strftime("%H%M%S")}.log'
        with open(log_file, 'w') as f:
            f.write(f"Command: {' '.join(cmd)}\n")
            f.write(f"Exit code: {result.returncode}\n")
            f.write(f"Duration: {elapsed:.1f}s\n")
            f.write("\n=== STDOUT ===\n")
            f.write(result.stdout)
            f.write("\n=== STDERR ===\n")
            f.write(result.stderr)
        print(f"Log saved: {log_file}")
    
    # Print summary
    if result.returncode == 0:
        print(f"✓ {symbol} {model_type} completed in {elapsed:.1f}s")
    else:
        print(f"✗ {symbol} {model_type} FAILED (exit code {result.returncode})")
        print(f"  Last 20 lines of stderr:")
        for line in result.stderr.strip().split('\n')[-20:]:
            print(f"  | {line}")
    
    return {
        'symbol': symbol,
        'model_type': model_type,
        'success': result.returncode == 0,
        'duration_seconds': elapsed,
        'exit_code': result.returncode,
    }


def main():
    parser = argparse.ArgumentParser(description='Batch train models for multiple symbols')
    parser.add_argument('--symbols', nargs='+', required=True, help='Symbols to train')
    parser.add_argument('--model-types', nargs='+', default=['gbm', 'regressor'],
                        help='Model types to train (default: gbm regressor)')
    parser.add_argument('--regressor-epochs', type=int, default=300,
                        help='Epochs for regressor training (default: 300)')
    parser.add_argument('--gbm-epochs', type=int, default=5000,
                        help='Max boosting rounds for GBM (default: 5000)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', type=str, default=None,
                        help='Directory for logs (default: results/batch_train_{timestamp})')
    args = parser.parse_args()
    
    # Setup output directory
    if args.output_dir:
        log_dir = Path(args.output_dir)
    else:
        log_dir = Path('results') / f'batch_train_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nBatch Training Started")
    print(f"Symbols: {', '.join(args.symbols)}")
    print(f"Model types: {', '.join(args.model_types)}")
    print(f"Log directory: {log_dir}")
    print(f"Seed: {args.seed}")
    
    results = []
    
    for symbol in args.symbols:
        for model_type in args.model_types:
            extra_args = ['--seed', str(args.seed)]
            
            if model_type == 'regressor':
                extra_args.extend(['--epochs', str(args.regressor_epochs)])
            elif model_type == 'gbm':
                extra_args.extend(['--epochs', str(args.gbm_epochs)])
            
            result = run_training(symbol, model_type, extra_args, log_dir)
            results.append(result)
    
    # Save summary
    summary_file = log_dir / 'training_summary.json'
    with open(summary_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'symbols': args.symbols,
            'model_types': args.model_types,
            'results': results,
        }, f, indent=2)
    
    # Print summary table
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    print(f"{'Symbol':<10} {'Model':<12} {'Status':<10} {'Duration':<12}")
    print(f"{'-'*10} {'-'*12} {'-'*10} {'-'*12}")
    
    success_count = 0
    for r in results:
        status = '✓ OK' if r['success'] else '✗ FAIL'
        duration = f"{r['duration_seconds']:.1f}s"
        print(f"{r['symbol']:<10} {r['model_type']:<12} {status:<10} {duration:<12}")
        if r['success']:
            success_count += 1
    
    print(f"\nTotal: {success_count}/{len(results)} succeeded")
    print(f"Summary saved: {summary_file}")


if __name__ == '__main__':
    main()
