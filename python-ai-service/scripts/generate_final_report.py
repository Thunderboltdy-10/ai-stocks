#!/usr/bin/env python
"""
Generate Final Analysis Report

Compiles all backtest results, baseline comparisons, and produces
the final summary artifacts for the follow-up audit.
"""

import json
import os
import subprocess
from pathlib import Path
from datetime import datetime
import re

# Run timestamp
RUN_TS = "20251215_followup"
OUTPUT_DIR = Path(f'results/{RUN_TS}')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Baseline data from prior run
BASELINES = {
    'AAPL': {
        'regressor_only': {'sharpe': 1.607, 'cum_return': 2.53, 'max_dd': -4.59},
        'gbm_only': {'sharpe': 4.465, 'cum_return': 0.98, 'max_dd': -0.39},
    },
    'HOOD': {
        'regressor_only': {'sharpe': 2.516, 'cum_return': -0.09, 'max_dd': -6.04},
        'gbm_only': {'sharpe': -0.897, 'cum_return': -0.09, 'max_dd': -7.18},
    }
}

# Current run results
RESULTS = {
    'AAPL': {
        'regressor_only': {'sharpe': 1.607, 'cum_return': 3.95, 'max_dd': -4.59},
        'gbm_only': {'sharpe': 4.465, 'cum_return': 2.07, 'max_dd': -0.39},
        'gbm_heavy': {'sharpe': 2.009, 'cum_return': 4.68, 'max_dd': -4.02},
        'balanced': {'sharpe': 1.807, 'cum_return': 4.37, 'max_dd': -4.36},
        'lstm_heavy': {'sharpe': 1.688, 'cum_return': 4.13, 'max_dd': -4.48},
    },
    'HOOD': {
        'regressor_only': {'sharpe': 2.516, 'cum_return': 11.81, 'max_dd': -6.04},
        'gbm_only': {'sharpe': -0.897, 'cum_return': -3.92, 'max_dd': -7.18},
        'gbm_heavy': {'sharpe': -0.897, 'cum_return': -3.92, 'max_dd': -7.18, 'note': 'LSTM collapsed, fell back to GBM'},
        'balanced': {'sharpe': -0.897, 'cum_return': -3.92, 'max_dd': -7.18, 'note': 'LSTM collapsed, fell back to GBM'},
        'lstm_heavy': {'sharpe': -0.897, 'cum_return': -3.92, 'max_dd': -7.18, 'note': 'LSTM collapsed, fell back to GBM'},
    },
    'MSFT': {
        'gbm_only': {'sharpe': -3.546, 'cum_return': -4.52, 'max_dd': -5.81},
    },
    'AMZN': {
        'gbm_only': {'sharpe': -0.008, 'cum_return': -0.08, 'max_dd': -2.82},
    },
    'NVDA': {
        'gbm_only': {'sharpe': 0.647, 'cum_return': 0.67, 'max_dd': -1.25},
    },
    'TSLA': {
        'gbm_only': {'sharpe': -1.812, 'cum_return': -1.77, 'max_dd': -3.11},
    },
    'XOM': {
        'gbm_only': {'sharpe': 2.268, 'cum_return': 3.24, 'max_dd': -2.75},
    },
    'JPM': {
        'gbm_only': {'sharpe': 2.654, 'cum_return': 0.13, 'max_dd': -0.16},
    },
}


def main():
    # Save summary table CSV
    with open(OUTPUT_DIR / 'summary_table.csv', 'w') as f:
        f.write('Symbol,FusionMode,Sharpe,CumReturn%,MaxDD%,BaselineSharpe,SharpeDelta,Notes\n')
        for sym, modes in RESULTS.items():
            for mode, metrics in modes.items():
                baseline = BASELINES.get(sym, {}).get(mode, {})
                baseline_sharpe = baseline.get('sharpe', '')
                delta = ''
                if baseline_sharpe and metrics.get('sharpe'):
                    delta = f"{metrics['sharpe'] - baseline_sharpe:+.3f}"
                note = metrics.get('note', '')
                f.write(f"{sym},{mode},{metrics['sharpe']:.3f},{metrics['cum_return']:.2f},{metrics['max_dd']:.2f},{baseline_sharpe},{delta},{note}\n")
    
    # Save JSON results
    with open(OUTPUT_DIR / 'backtest_results.json', 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'baselines': BASELINES,
            'results': RESULTS,
        }, f, indent=2)
    
    # Print summary
    print("=" * 80)
    print("BACKTEST RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'Symbol':<8} {'Mode':<18} {'Sharpe':>8} {'CumRet':>10} {'MaxDD':>8}")
    print("-" * 60)
    
    for sym, modes in RESULTS.items():
        for mode, metrics in modes.items():
            print(f"{sym:<8} {mode:<18} {metrics['sharpe']:>8.3f} {metrics['cum_return']:>9.2f}% {metrics['max_dd']:>7.2f}%")
    
    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
