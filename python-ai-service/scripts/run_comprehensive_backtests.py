#!/usr/bin/env python
"""
Comprehensive Backtesting Script

Runs backtests for all symbols and fusion modes, collecting metrics
and producing summary tables for the follow-up audit.
"""

import subprocess
import sys
import json
import re
import time
from pathlib import Path
from datetime import datetime

# Symbols to test
SYMBOLS = ['AAPL', 'HOOD', 'MSFT', 'AMZN', 'NVDA', 'TSLA', 'XOM', 'JPM']

# Fusion modes
FUSION_MODES = ['gbm_only', 'regressor_only', 'gbm_heavy', 'balanced', 'lstm_heavy']

# Output directory
RUN_TS = datetime.now().strftime('%Y%m%d_%H%M%S')
OUTPUT_DIR = Path(f'results/backtest_run_{RUN_TS}')


def parse_backtest_output(output: str) -> dict:
    """Parse backtest output for key metrics."""
    metrics = {
        'sharpe': None,
        'cum_return': None,
        'max_drawdown': None,
        'total_trades': None,
        'win_rate': None,
    }
    
    # Extract Sharpe
    sharpe_match = re.search(r'Sharpe[:\s]+([+-]?\d+\.?\d*)', output)
    if sharpe_match:
        metrics['sharpe'] = float(sharpe_match.group(1))
    
    # Extract cumulative return
    cum_match = re.search(r'Cumulative Return[:\s]+([+-]?\d+\.?\d*)%?', output)
    if cum_match:
        metrics['cum_return'] = float(cum_match.group(1))
    
    # Extract max drawdown
    dd_match = re.search(r'Max.*Drawdown[:\s]+([+-]?\d+\.?\d*)%?', output, re.IGNORECASE)
    if dd_match:
        metrics['max_drawdown'] = float(dd_match.group(1))
    
    # Extract trade count
    trades_match = re.search(r'Total Trades[:\s]+(\d+)', output)
    if trades_match:
        metrics['total_trades'] = int(trades_match.group(1))
    
    # Extract win rate
    win_match = re.search(r'Win Rate[:\s]+([+-]?\d+\.?\d*)%?', output)
    if win_match:
        metrics['win_rate'] = float(win_match.group(1))
    
    return metrics


def run_backtest(symbol: str, fusion_mode: str, backtest_days: int = 60) -> dict:
    """Run a single backtest and return results."""
    
    cmd = [
        'python', 'inference_and_backtest.py',
        '--symbol', symbol,
        '--fusion-mode', fusion_mode,
        '--backtest-days', str(backtest_days),
        '--skip-analysis'
    ]
    
    start_time = time.time()
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=300  # 5 min timeout
    )
    elapsed = time.time() - start_time
    
    # Parse output
    output = result.stdout + result.stderr
    metrics = parse_backtest_output(output)
    
    return {
        'symbol': symbol,
        'fusion_mode': fusion_mode,
        'success': result.returncode == 0,
        'metrics': metrics,
        'duration': elapsed,
        'error': result.stderr[-500:] if result.returncode != 0 else None
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"=== Comprehensive Backtest Run ===")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Symbols: {SYMBOLS}")
    print(f"Fusion modes: {FUSION_MODES}")
    print()
    
    results = []
    
    for symbol in SYMBOLS:
        for mode in FUSION_MODES:
            print(f"Running {symbol} / {mode}...", end=' ', flush=True)
            try:
                result = run_backtest(symbol, mode)
                results.append(result)
                
                if result['success'] and result['metrics']['sharpe'] is not None:
                    print(f"Sharpe={result['metrics']['sharpe']:.3f}")
                elif result['success']:
                    print("OK (no metrics)")
                else:
                    print("FAILED")
            except subprocess.TimeoutExpired:
                print("TIMEOUT")
                results.append({
                    'symbol': symbol,
                    'fusion_mode': mode,
                    'success': False,
                    'error': 'TIMEOUT'
                })
    
    # Save raw results
    with open(OUTPUT_DIR / 'raw_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Create summary CSV
    with open(OUTPUT_DIR / 'summary.csv', 'w') as f:
        f.write('Symbol,FusionMode,Sharpe,CumReturn,MaxDD,Trades,WinRate,Status\n')
        for r in results:
            m = r.get('metrics', {})
            status = 'OK' if r['success'] else 'FAIL'
            f.write(f"{r['symbol']},{r['fusion_mode']},{m.get('sharpe', '')},{m.get('cum_return', '')},{m.get('max_drawdown', '')},{m.get('total_trades', '')},{m.get('win_rate', '')},{status}\n")
    
    # Print summary table
    print(f"\n{'='*80}")
    print("SUMMARY TABLE")
    print(f"{'='*80}")
    print(f"{'Symbol':<8} {'Mode':<15} {'Sharpe':>8} {'CumRet':>8} {'MaxDD':>8}")
    print(f"{'-'*8} {'-'*15} {'-'*8} {'-'*8} {'-'*8}")
    
    for r in results:
        if r['success']:
            m = r.get('metrics', {})
            sharpe = f"{m.get('sharpe', 0):.3f}" if m.get('sharpe') else 'N/A'
            cum_ret = f"{m.get('cum_return', 0):.2f}%" if m.get('cum_return') else 'N/A'
            max_dd = f"{m.get('max_drawdown', 0):.2f}%" if m.get('max_drawdown') else 'N/A'
            print(f"{r['symbol']:<8} {r['fusion_mode']:<15} {sharpe:>8} {cum_ret:>8} {max_dd:>8}")
        else:
            print(f"{r['symbol']:<8} {r['fusion_mode']:<15} {'FAILED':>8}")
    
    print(f"\nResults saved to {OUTPUT_DIR}")
    
    # Count successes
    success_count = sum(1 for r in results if r['success'])
    print(f"\nTotal: {success_count}/{len(results)} successful")


if __name__ == '__main__':
    main()
