#!/usr/bin/env python3
"""
F4 Results Analysis Utility

Quick analysis tool for F4 validation results:
- Parse JSON results
- Generate statistics
- Identify problematic models
- Export filtered results

Usage:
    python3 analyze_f4_results.py [--json FILE] [--symbol SYMBOL] [--test TEST_TYPE]
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
import argparse
from datetime import datetime

RESULTS_DIR = Path(__file__).resolve().parent.parent / "f4_validation_results"


def load_results(results_file=None):
    """Load validation results from JSON."""
    if results_file is None:
        results_file = RESULTS_DIR / "all_models_validation.json"

    results_file = Path(results_file)

    if not results_file.exists():
        print(f"ERROR: Results file not found: {results_file}")
        print(f"Run: python3 scripts/run_all_f4_validations.py")
        sys.exit(1)

    with open(results_file, 'r') as f:
        return json.load(f)


def print_overview(results):
    """Print overview statistics."""
    meta = results['metadata']

    print("\n" + "="*60)
    print("VALIDATION OVERVIEW")
    print("="*60)
    print(f"Date: {meta['orchestration_date']}")
    print(f"Framework: {meta['validation_framework']}")
    print()
    print(f"Total Symbols: {meta['total_symbols']}")
    print(f"Tests Executed: {meta['tests_executed']}")
    print(f"Tests Passed: {meta['tests_passed']}")
    print(f"Tests Failed: {meta['tests_failed']}")

    success_rate = 100 * meta['tests_passed'] / max(1, meta['tests_executed'])
    print(f"Success Rate: {success_rate:.1f}%")
    print("="*60 + "\n")


def print_by_symbol(results):
    """Print summary by symbol."""
    print("\n" + "="*80)
    print("RESULTS BY SYMBOL")
    print("="*80)

    # Print table header
    print(f"\n{'Symbol':<10} {'Regressor':<12} {'GBM':<12} {'Classifier':<12} {'Quantile':<12} {'Status':<10}")
    print("-" * 80)

    passed_count = defaultdict(int)
    failed_count = defaultdict(int)

    for symbol in sorted(results['symbols'].keys()):
        tests = results['symbols'][symbol]

        status_map = {}
        for test_name in ['regressor', 'gbm', 'classifier', 'quantile']:
            test_result = tests.get(test_name, {})
            if test_result.get('passed'):
                status_map[test_name] = '✅'
                passed_count[test_name] += 1
            else:
                status_map[test_name] = '❌'
                failed_count[test_name] += 1

        all_passed = all(test_result.get('passed', False) for test_result in tests.values() if test_result)
        overall = "✅ PASS" if all_passed else "⚠️ MIXED"

        print(f"{symbol:<10} {status_map.get('regressor', '❓'):<12} "
              f"{status_map.get('gbm', '❓'):<12} {status_map.get('classifier', '❓'):<12} "
              f"{status_map.get('quantile', '❓'):<12} {overall:<10}")

    # Summary by test type
    print("\n" + "-" * 80)
    print("SUMMARY BY TEST TYPE:")
    print("-" * 80)
    for test_name in ['regressor', 'gbm', 'classifier', 'quantile']:
        total = passed_count[test_name] + failed_count[test_name]
        if total > 0:
            rate = 100 * passed_count[test_name] / total
            print(f"{test_name.title():<15} {passed_count[test_name]}/{total} ({rate:.0f}%)")

    print("="*80 + "\n")


def print_symbol_details(results, symbol):
    """Print detailed results for a symbol."""
    if symbol not in results['symbols']:
        print(f"ERROR: Symbol {symbol} not found in results")
        return

    print("\n" + "="*60)
    print(f"DETAILS FOR {symbol}")
    print("="*60)

    tests = results['symbols'][symbol]

    for test_name in ['regressor', 'gbm', 'classifier', 'quantile']:
        test_result = tests.get(test_name, {})

        if not test_result:
            continue

        status = "✅ PASSED" if test_result.get('passed') else f"❌ FAILED ({test_result.get('status', 'unknown')})"
        print(f"\n{test_name.upper()}: {status}")

        # Metrics
        metrics = test_result.get('metrics', {})
        if metrics:
            print("  Metrics:")
            for key, value in sorted(metrics.items()):
                if isinstance(value, float):
                    if key.endswith('ratio'):
                        print(f"    - {key}: {value:.4f}")
                    elif key.startswith('pct_') or key.endswith('_accuracy'):
                        print(f"    - {key}: {value*100:.2f}%")
                    else:
                        print(f"    - {key}: {value:.6f}")
                else:
                    print(f"    - {key}: {value}")

        # Errors
        errors = test_result.get('errors', [])
        if errors:
            print("  Errors:")
            for error in errors[:3]:
                print(f"    - {error}")

    print("="*60 + "\n")


def print_test_analysis(results, test_type):
    """Print analysis for a specific test type."""
    if test_type not in ['regressor', 'gbm', 'classifier', 'quantile']:
        print(f"ERROR: Unknown test type: {test_type}")
        return

    print("\n" + "="*60)
    print(f"ANALYSIS: {test_type.upper()}")
    print("="*60)

    passed = []
    failed = []
    metrics_all = defaultdict(list)

    for symbol, tests in results['symbols'].items():
        test_result = tests.get(test_type, {})

        if not test_result:
            continue

        if test_result.get('passed'):
            passed.append(symbol)
        else:
            failed.append((symbol, test_result.get('status'), test_result.get('errors')))

        # Collect metrics
        metrics = test_result.get('metrics', {})
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics_all[key].append((symbol, value))

    # Print summary
    print(f"\nPassed: {len(passed)}/{len(passed)+len(failed)}")
    if passed:
        print(f"  {', '.join(passed)}")

    if failed:
        print(f"\nFailed: {len(failed)}/{len(passed)+len(failed)}")
        for symbol, status, errors in failed:
            print(f"  {symbol}: {status}")
            if errors:
                print(f"    Error: {errors[0][:60]}")

    # Statistics for numeric metrics
    if metrics_all:
        print("\nMetrics Statistics:")
        print("-" * 60)
        for metric_name in sorted(metrics_all.keys()):
            values = [v for _, v in metrics_all[metric_name]]
            if values:
                min_val = min(values)
                mean_val = sum(values) / len(values)
                max_val = max(values)

                min_symbol = [s for s, v in metrics_all[metric_name] if v == min_val][0]
                max_symbol = [s for s, v in metrics_all[metric_name] if v == max_val][0]

                if 'ratio' in metric_name or 'accuracy' in metric_name:
                    print(f"{metric_name}:")
                    print(f"  Min: {min_val:.4f} ({min_symbol})")
                    print(f"  Mean: {mean_val:.4f}")
                    print(f"  Max: {max_val:.4f} ({max_symbol})")
                else:
                    print(f"{metric_name}:")
                    print(f"  Min: {min_val:.6f} ({min_symbol})")
                    print(f"  Mean: {mean_val:.6f}")
                    print(f"  Max: {max_val:.6f} ({max_symbol})")

    print("="*60 + "\n")


def export_csv(results):
    """Export results as CSV."""
    output_file = RESULTS_DIR / "f4_results.csv"

    with open(output_file, 'w') as f:
        # Header
        f.write("Symbol,Test Type,Status,Passed,Sharpe Ratio,Directional Accuracy,Prediction Std\n")

        # Data
        for symbol in sorted(results['symbols'].keys()):
            tests = results['symbols'][symbol]

            for test_name in ['regressor', 'gbm', 'classifier', 'quantile']:
                test_result = tests.get(test_name, {})

                if not test_result:
                    continue

                metrics = test_result.get('metrics', {})
                status = test_result.get('status', 'unknown')
                passed = 'Yes' if test_result.get('passed') else 'No'

                sharpe = metrics.get('sharpe_ratio', '')
                dir_acc = metrics.get('directional_accuracy', '')
                pred_std = metrics.get('prediction_std', '')

                f.write(f"{symbol},{test_name},{status},{passed},{sharpe},{dir_acc},{pred_std}\n")

    print(f"CSV export saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description='F4 Results Analysis Utility')
    parser.add_argument('--json', help='Path to results JSON file', default=None)
    parser.add_argument('--symbol', help='Analyze specific symbol', default=None)
    parser.add_argument('--test', help='Analyze specific test type', default=None)
    parser.add_argument('--csv', action='store_true', help='Export as CSV')

    args = parser.parse_args()

    # Load results
    results = load_results(args.json)

    # Print overview
    print_overview(results)

    # Print by symbol
    print_by_symbol(results)

    # Symbol-specific analysis
    if args.symbol:
        print_symbol_details(results, args.symbol)

    # Test type analysis
    if args.test:
        print_test_analysis(results, args.test)

    # CSV export
    if args.csv:
        export_csv(results)


if __name__ == '__main__':
    main()
