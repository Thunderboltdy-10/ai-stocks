#!/usr/bin/env python3
"""
F4 Fusion Mode Analysis Script

Tests all 6 fusion modes across available symbols to identify best combinations.

Fusion Modes:
1. regressor_only: Pure LSTM+Transformer
2. classifier_only: BUY/SELL signals only
3. regressor_classifier: Combined (hybrid)
4. gbm_only: Pure GBM
5. regressor_gbm: LSTM + GBM (no classifiers)
6. all_models: All three model types

Output: JSON with performance comparison and recommendations
"""

import sys
import json
import subprocess
from pathlib import Path
from datetime import datetime
import argparse

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

FUSION_MODES = [
    'regressor_only',
    'classifier_only',
    'regressor_classifier',
    'gbm_only',
    'regressor_gbm',
    'all_models'
]

def run_backtest(symbol: str, fusion_mode: str, timeout: int = 300):
    """
    Run backtest for a specific symbol and fusion mode

    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        fusion_mode: Fusion mode to test
        timeout: Timeout in seconds

    Returns:
        dict: Backtest results or error info
    """
    print(f"\n[{symbol}] Testing fusion mode: {fusion_mode}")

    try:
        cmd = [
            'python',
            'inference_and_backtest.py',
            symbol,
            '--mode', fusion_mode
        ]

        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=timeout
        )

        if result.returncode != 0:
            print(f"  ❌ FAILED: {result.stderr[:200]}")
            return {
                'status': 'FAILED',
                'error': result.stderr[:500],
                'stdout': result.stdout[:500]
            }

        # Try to find and parse backtest results
        # Results are typically saved to backtest_results/{SYMBOL}_{timestamp}/
        backtest_dir = PROJECT_ROOT / 'backtest_results'

        # Find most recent result for this symbol
        symbol_dirs = sorted(
            [d for d in backtest_dir.glob(f"{symbol}_*") if d.is_dir()],
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )

        if not symbol_dirs:
            print(f"  ⚠️  WARNING: No backtest results found")
            return {
                'status': 'NO_RESULTS',
                'error': 'Backtest completed but no results directory found'
            }

        latest_dir = symbol_dirs[0]
        metrics_file = latest_dir / 'metrics.json'

        if not metrics_file.exists():
            print(f"  ⚠️  WARNING: No metrics.json found in {latest_dir}")
            return {
                'status': 'NO_METRICS',
                'result_dir': str(latest_dir)
            }

        with open(metrics_file, 'r') as f:
            metrics = json.load(f)

        # Extract key metrics
        sharpe = metrics.get('sharpe_ratio', 0.0)
        total_return = metrics.get('total_return', 0.0)
        max_dd = metrics.get('max_drawdown', 0.0)
        win_rate = metrics.get('win_rate')

        print(f"  ✅ SUCCESS: Sharpe={sharpe:.2f}, Return={total_return:.2%}, MaxDD={max_dd:.2%}")

        return {
            'status': 'SUCCESS',
            'sharpe_ratio': sharpe,
            'total_return': total_return,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'result_dir': str(latest_dir),
            'all_metrics': metrics
        }

    except subprocess.TimeoutExpired:
        print(f"  ❌ TIMEOUT: Exceeded {timeout}s")
        return {
            'status': 'TIMEOUT',
            'error': f'Backtest exceeded {timeout}s timeout'
        }
    except Exception as e:
        print(f"  ❌ ERROR: {str(e)}")
        return {
            'status': 'ERROR',
            'error': str(e)
        }


def analyze_fusion_modes(symbols: list, output_file: Path):
    """
    Test all fusion modes across all symbols and analyze results

    Args:
        symbols: List of symbols to test
        output_file: Path to save JSON results
    """
    print("\n" + "="*80)
    print("F4 FUSION MODE ANALYSIS")
    print("="*80)
    print(f"\nSymbols: {', '.join(symbols)}")
    print(f"Fusion Modes: {', '.join(FUSION_MODES)}")
    print(f"Total Tests: {len(symbols)} symbols × {len(FUSION_MODES)} modes = {len(symbols) * len(FUSION_MODES)} backtests")
    print("\n" + "="*80)

    # Initialize results structure
    results = {
        'metadata': {
            'analysis_date': datetime.now().isoformat(),
            'symbols_tested': symbols,
            'fusion_modes': FUSION_MODES,
            'total_tests': len(symbols) * len(FUSION_MODES)
        },
        'symbol_results': {},
        'mode_statistics': {},
        'best_combinations': {},
        'recommendations': []
    }

    # Test each symbol × mode combination
    for symbol in symbols:
        print(f"\n{'='*80}")
        print(f"Testing Symbol: {symbol}")
        print(f"{'='*80}")

        symbol_results = {}

        for mode in FUSION_MODES:
            result = run_backtest(symbol, mode, timeout=300)
            symbol_results[mode] = result

        results['symbol_results'][symbol] = symbol_results

        # Find best mode for this symbol
        successful_modes = {
            mode: res for mode, res in symbol_results.items()
            if res.get('status') == 'SUCCESS' and res.get('sharpe_ratio') is not None
        }

        if successful_modes:
            best_mode = max(
                successful_modes.items(),
                key=lambda x: x[1]['sharpe_ratio']
            )
            print(f"\n🏆 Best mode for {symbol}: {best_mode[0]} (Sharpe: {best_mode[1]['sharpe_ratio']:.2f})")
            results['best_combinations'][symbol] = {
                'mode': best_mode[0],
                'sharpe': best_mode[1]['sharpe_ratio'],
                'total_return': best_mode[1]['total_return']
            }
        else:
            print(f"\n⚠️  No successful modes for {symbol}")

    # Calculate mode statistics (average across symbols)
    print("\n" + "="*80)
    print("CROSS-SYMBOL MODE STATISTICS")
    print("="*80)

    for mode in FUSION_MODES:
        mode_sharpes = []
        mode_returns = []
        success_count = 0

        for symbol in symbols:
            result = results['symbol_results'][symbol].get(mode, {})
            if result.get('status') == 'SUCCESS' and result.get('sharpe_ratio') is not None:
                mode_sharpes.append(result['sharpe_ratio'])
                mode_returns.append(result['total_return'])
                success_count += 1

        if mode_sharpes:
            import numpy as np
            avg_sharpe = np.mean(mode_sharpes)
            std_sharpe = np.std(mode_sharpes)
            avg_return = np.mean(mode_returns)

            print(f"\n{mode}:")
            print(f"  Success Rate: {success_count}/{len(symbols)} ({success_count/len(symbols):.0%})")
            print(f"  Avg Sharpe: {avg_sharpe:.3f} ± {std_sharpe:.3f}")
            print(f"  Avg Return: {avg_return:.2%}")

            results['mode_statistics'][mode] = {
                'success_rate': success_count / len(symbols),
                'success_count': success_count,
                'avg_sharpe': float(avg_sharpe),
                'std_sharpe': float(std_sharpe),
                'avg_return': float(avg_return),
                'consistency': float(1.0 / (1.0 + std_sharpe)) if std_sharpe > 0 else 1.0
            }
        else:
            print(f"\n{mode}:")
            print(f"  ❌ No successful backtests")
            results['mode_statistics'][mode] = {
                'success_rate': 0.0,
                'success_count': 0,
                'avg_sharpe': None,
                'std_sharpe': None,
                'avg_return': None,
                'consistency': 0.0
            }

    # Generate recommendations
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)

    # Find best universal mode (highest avg Sharpe with good consistency)
    successful_modes = {
        mode: stats for mode, stats in results['mode_statistics'].items()
        if stats['avg_sharpe'] is not None
    }

    if successful_modes:
        # Rank by avg Sharpe
        best_by_sharpe = max(
            successful_modes.items(),
            key=lambda x: x[1]['avg_sharpe']
        )

        # Rank by consistency (low std)
        best_by_consistency = max(
            successful_modes.items(),
            key=lambda x: x[1]['consistency']
        )

        print(f"\n🏆 Best Universal Mode (by Sharpe): {best_by_sharpe[0]}")
        print(f"   Avg Sharpe: {best_by_sharpe[1]['avg_sharpe']:.3f}")
        print(f"   Success Rate: {best_by_sharpe[1]['success_rate']:.0%}")

        print(f"\n🎯 Most Consistent Mode: {best_by_consistency[0]}")
        print(f"   Consistency: {best_by_consistency[1]['consistency']:.3f}")
        print(f"   Sharpe Std: ±{best_by_consistency[1]['std_sharpe']:.3f}")

        # Central model viability decision
        win_count = {}
        for symbol, best in results['best_combinations'].items():
            mode = best['mode']
            win_count[mode] = win_count.get(mode, 0) + 1

        most_wins = max(win_count.items(), key=lambda x: x[1]) if win_count else None

        if most_wins:
            win_rate = most_wins[1] / len(symbols)
            print(f"\n📊 Mode with Most Wins: {most_wins[0]} ({most_wins[1]}/{len(symbols)} symbols)")

            if win_rate >= 0.5 and best_by_sharpe[1]['std_sharpe'] < 0.3:
                decision = "YES - Use central architecture with recommended mode"
                recommended_mode = most_wins[0]
            else:
                decision = "NO - Use per-symbol customized modes"
                recommended_mode = None

            print(f"\n💡 Central Model Viability: {decision}")
            if recommended_mode:
                print(f"   Recommended Mode: {recommended_mode}")

        results['recommendations'] = {
            'best_universal_mode': best_by_sharpe[0],
            'best_universal_sharpe': best_by_sharpe[1]['avg_sharpe'],
            'most_consistent_mode': best_by_consistency[0],
            'mode_with_most_wins': most_wins[0] if most_wins else None,
            'central_model_viable': win_rate >= 0.5 if most_wins else False,
            'decision': decision if most_wins else "INSUFFICIENT_DATA"
        }
    else:
        print("\n⚠️  No successful fusion modes found")
        results['recommendations'] = {
            'decision': 'ALL_MODES_FAILED',
            'action': 'Fix critical issues before fusion analysis'
        }

    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"✅ Fusion analysis results saved to: {output_file}")
    print(f"{'='*80}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Analyze fusion mode performance across symbols'
    )
    parser.add_argument(
        '--symbols',
        nargs='+',
        default=['AAPL', 'ASML', 'IWM', 'KO', 'MSFT'],
        help='Symbols to test (default: symbols with regressor models)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=PROJECT_ROOT / 'f4_validation_results' / 'fusion_comparison.json',
        help='Output JSON file path'
    )
    parser.add_argument(
        '--modes',
        nargs='+',
        choices=FUSION_MODES,
        help='Specific modes to test (default: all modes)'
    )

    args = parser.parse_args()

    # Override FUSION_MODES if specific modes requested
    global FUSION_MODES
    if args.modes:
        FUSION_MODES = args.modes

    # Run analysis
    results = analyze_fusion_modes(args.symbols, args.output)

    return 0


if __name__ == '__main__':
    sys.exit(main())
