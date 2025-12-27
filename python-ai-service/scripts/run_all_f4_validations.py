#!/usr/bin/env python3
"""
Comprehensive F4 Validation Orchestration Script

Runs all 4 F4 test types (regressor, GBM, classifier, quantile) across all available symbols.
Collects results in structured JSON format and generates a markdown summary report.

Output:
  - /f4_validation_results/all_models_validation.json: Detailed results
  - /f4_validation_results/VALIDATION_SUMMARY.md: Human-readable report

Author: AI-Stocks F4 Validation
Date: December 21, 2025
"""

import sys
import os
import json
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict
import re

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration
TESTS_DIR = PROJECT_ROOT / "tests"
RESULTS_DIR = PROJECT_ROOT / "f4_validation_results"
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"

# Test types to run
TEST_TYPES = [
    'test_regressor_standalone.py',
    'test_gbm_standalone.py',
    'test_classifier_standalone.py',
    'test_quantile_standalone.py'
]

# Result structure
RESULTS = {
    'metadata': {
        'orchestration_date': datetime.now().isoformat(),
        'validation_framework': 'F4 Comprehensive Orchestration',
        'tests_executed': 0,
        'tests_passed': 0,
        'tests_failed': 0,
        'total_symbols': 0,
    },
    'symbols': {}
}


def discover_symbols():
    """Discover all available stock symbols in saved_models directory."""
    symbols = set()

    if not SAVED_MODELS_DIR.exists():
        print(f"WARNING: saved_models directory not found: {SAVED_MODELS_DIR}")
        return list(symbols)

    # Look for symbol directories (uppercase folders)
    for item in SAVED_MODELS_DIR.iterdir():
        if item.is_dir() and item.name.isupper() and len(item.name) <= 5:
            symbols.add(item.name)

    # Also extract from legacy filename patterns
    for item in SAVED_MODELS_DIR.iterdir():
        if item.is_file() and '_1d_regressor_final' in item.name:
            # Extract symbol from filename like AAPL_1d_regressor_final_model.keras
            match = re.match(r'^([A-Z]+?)_', item.name)
            if match:
                symbols.add(match.group(1))

    return sorted(list(symbols))


def parse_test_output(output):
    """Parse F4 test output to extract key metrics."""
    metrics = {
        'passed': False,
        'errors': [],
        'metrics': {}
    }

    # Check for test summary
    if 'TEST SUMMARY' in output:
        # Extract passed/failed counts
        if '✅ ALL TESTS PASSED' in output:
            metrics['passed'] = True
        elif '❌ SOME TESTS FAILED' in output:
            metrics['passed'] = False

    # Extract specific metrics
    # Sharpe Ratio
    sharpe_match = re.search(r'Sharpe Ratio: ([-\d.]+)', output)
    if sharpe_match:
        metrics['metrics']['sharpe_ratio'] = float(sharpe_match.group(1))

    # Directional Accuracy (Regressor)
    dir_acc_match = re.search(r'Directional Accuracy: ([\d.]+)%', output)
    if dir_acc_match:
        metrics['metrics']['directional_accuracy'] = float(dir_acc_match.group(1)) / 100

    # Prediction Std (Regressor/GBM)
    pred_std_match = re.search(r'Prediction std: ([\d.e+-]+)', output)
    if pred_std_match:
        metrics['metrics']['prediction_std'] = float(pred_std_match.group(1))

    # Percentage metrics
    pct_negative_match = re.search(r'% Negative: ([\d.]+)%', output)
    if pct_negative_match:
        metrics['metrics']['pct_negative'] = float(pct_negative_match.group(1)) / 100

    pct_positive_match = re.search(r'% Positive: ([\d.]+)%', output)
    if pct_positive_match:
        metrics['metrics']['pct_positive'] = float(pct_positive_match.group(1)) / 100

    # BUY/SELL signals
    buy_signals_match = re.search(r'BUY signals \(prob > 0\.5\): ([\d.]+)%', output)
    if buy_signals_match:
        metrics['metrics']['buy_signals_pct'] = float(buy_signals_match.group(1)) / 100

    sell_signals_match = re.search(r'SELL signals \(prob > 0\.5\): ([\d.]+)%', output)
    if sell_signals_match:
        metrics['metrics']['sell_signals_pct'] = float(sell_signals_match.group(1)) / 100

    # Long/Short/Neutral distribution (Classifier)
    long_match = re.search(r'Long: ([\d.]+)%', output)
    if long_match:
        metrics['metrics']['pct_long'] = float(long_match.group(1)) / 100

    short_match = re.search(r'Short: ([\d.]+)%', output)
    if short_match:
        metrics['metrics']['pct_short'] = float(short_match.group(1)) / 100

    neutral_match = re.search(r'Neutral: ([\d.]+)%', output)
    if neutral_match:
        metrics['metrics']['pct_neutral'] = float(neutral_match.group(1)) / 100

    # Model availability flags
    xgb_match = re.search(r'XGBoost available: (True|False)', output)
    if xgb_match:
        metrics['metrics']['xgb_available'] = xgb_match.group(1) == 'True'

    lgb_match = re.search(r'LightGBM available: (True|False)', output)
    if lgb_match:
        metrics['metrics']['lgb_available'] = lgb_match.group(1) == 'True'

    buy_available_match = re.search(r'BUY Classifier Available: (True|False)', output)
    if buy_available_match:
        metrics['metrics']['buy_classifier_available'] = buy_available_match.group(1) == 'True'

    sell_available_match = re.search(r'SELL Classifier Available: (True|False)', output)
    if sell_available_match:
        metrics['metrics']['sell_classifier_available'] = sell_available_match.group(1) == 'True'

    # Extract error messages
    if '❌ FAILED' in output or '❌' in output:
        error_lines = [line.strip() for line in output.split('\n') if '❌' in line]
        metrics['errors'] = error_lines[:5]  # First 5 errors

    return metrics


def run_test(symbol, test_type):
    """Run a single F4 test for a symbol."""
    test_file = TESTS_DIR / test_type

    if not test_file.exists():
        return {
            'status': 'not_found',
            'test_file': str(test_file),
            'passed': False,
            'errors': [f'Test file not found: {test_file}']
        }

    # Set environment variable for symbol
    env = os.environ.copy()
    env['TEST_SYMBOL'] = symbol

    try:
        # Run test with timeout
        result = subprocess.run(
            ['python3', str(test_file)],
            cwd=str(PROJECT_ROOT),
            env=env,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout per test
        )

        # Parse output
        metrics = parse_test_output(result.stdout + result.stderr)
        metrics['status'] = 'completed'
        metrics['return_code'] = result.returncode
        metrics['output_length'] = len(result.stdout) + len(result.stderr)

        return metrics

    except subprocess.TimeoutExpired:
        return {
            'status': 'timeout',
            'passed': False,
            'errors': [f'Test timed out (>300s)']
        }
    except Exception as e:
        return {
            'status': 'error',
            'passed': False,
            'errors': [str(e)]
        }


def run_all_validations():
    """Run all F4 tests for all symbols."""

    print("\n" + "="*80)
    print("F4 COMPREHENSIVE VALIDATION ORCHESTRATION")
    print("="*80 + "\n")

    # Discover symbols
    symbols = discover_symbols()
    print(f"Discovered symbols: {', '.join(symbols)}")
    print(f"Total symbols: {len(symbols)}\n")

    RESULTS['metadata']['total_symbols'] = len(symbols)

    # Run tests for each symbol
    for symbol in symbols:
        print(f"\nTesting {symbol}")
        print("-" * 40)

        RESULTS['symbols'][symbol] = {}

        for test_type in TEST_TYPES:
            test_name = test_type.replace('test_', '').replace('_standalone.py', '')
            print(f"  Running {test_name}...", end=' ', flush=True)

            RESULTS['metadata']['tests_executed'] += 1

            result = run_test(symbol, test_type)
            RESULTS['symbols'][symbol][test_name] = result

            if result.get('passed', False):
                RESULTS['metadata']['tests_passed'] += 1
                print("✅ PASSED")
            else:
                RESULTS['metadata']['tests_failed'] += 1
                status = result.get('status', 'unknown')
                if result.get('errors'):
                    print(f"❌ FAILED ({status})")
                    print(f"     Errors: {result['errors'][0][:60]}...")
                else:
                    print(f"❌ FAILED ({status})")

    print("\n" + "="*80)
    print("ORCHESTRATION COMPLETE")
    print("="*80)
    print(f"Tests executed: {RESULTS['metadata']['tests_executed']}")
    print(f"Tests passed: {RESULTS['metadata']['tests_passed']}")
    print(f"Tests failed: {RESULTS['metadata']['tests_failed']}")
    print(f"Success rate: {100*RESULTS['metadata']['tests_passed']/max(1, RESULTS['metadata']['tests_executed']):.1f}%")

    return RESULTS


def save_results(results):
    """Save results to JSON file."""

    # Ensure output directory exists
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Save full results
    results_file = RESULTS_DIR / "all_models_validation.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to: {results_file}")

    return results_file


def generate_markdown_summary(results):
    """Generate markdown summary report."""

    summary_lines = [
        "# F4 Comprehensive Validation Results",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Overview",
        "",
        f"- **Total Symbols Tested:** {results['metadata']['total_symbols']}",
        f"- **Tests Executed:** {results['metadata']['tests_executed']}",
        f"- **Tests Passed:** {results['metadata']['tests_passed']}",
        f"- **Tests Failed:** {results['metadata']['tests_failed']}",
        f"- **Success Rate:** {100*results['metadata']['tests_passed']/max(1, results['metadata']['tests_executed']):.1f}%",
        "",
        "## Test Results by Symbol",
        ""
    ]

    # Summary table
    summary_lines.extend([
        "| Symbol | Regressor | GBM | Classifier | Quantile | Overall |",
        "|--------|-----------|-----|-----------|----------|---------|"
    ])

    for symbol in sorted(results['symbols'].keys()):
        tests = results['symbols'][symbol]

        # Check each test
        regressor_status = "✅" if tests.get('regressor', {}).get('passed') else "❌"
        gbm_status = "✅" if tests.get('gbm', {}).get('passed') else "❌"
        classifier_status = "✅" if tests.get('classifier', {}).get('passed') else "❌"
        quantile_status = "✅" if tests.get('quantile', {}).get('passed') else "❌"

        all_passed = all([
            tests.get(t, {}).get('passed', False) or tests.get(t, {}).get('status') == 'not_found'
            for t in ['regressor', 'gbm', 'classifier', 'quantile']
        ])
        overall_status = "✅" if all_passed else "⚠️"

        summary_lines.append(
            f"| {symbol} | {regressor_status} | {gbm_status} | {classifier_status} | {quantile_status} | {overall_status} |"
        )

    summary_lines.extend([
        "",
        "## Detailed Results",
        ""
    ])

    # Detailed results per symbol
    for symbol in sorted(results['symbols'].keys()):
        summary_lines.append(f"### {symbol}")
        summary_lines.append("")

        tests = results['symbols'][symbol]

        for test_name in ['regressor', 'gbm', 'classifier', 'quantile']:
            test_result = tests.get(test_name, {})

            if not test_result:
                continue

            # Test header
            status_icon = "✅" if test_result.get('passed') else "❌"
            status_text = "PASSED" if test_result.get('passed') else f"FAILED ({test_result.get('status', 'unknown')})"

            summary_lines.append(f"#### {test_name.title()} {status_icon} {status_text}")
            summary_lines.append("")

            # Metrics
            metrics = test_result.get('metrics', {})
            if metrics:
                summary_lines.append("**Metrics:**")
                summary_lines.append("")
                for key, value in sorted(metrics.items()):
                    if isinstance(value, float):
                        if key.endswith('ratio'):
                            summary_lines.append(f"- {key}: {value:.4f}")
                        elif key.startswith('pct_') or key.endswith('_accuracy'):
                            summary_lines.append(f"- {key}: {value*100:.2f}%")
                        else:
                            summary_lines.append(f"- {key}: {value:.6f}")
                    else:
                        summary_lines.append(f"- {key}: {value}")
                summary_lines.append("")

            # Errors
            errors = test_result.get('errors', [])
            if errors:
                summary_lines.append("**Errors:**")
                summary_lines.append("")
                for error in errors[:3]:  # First 3 errors
                    summary_lines.append(f"- {error}")
                summary_lines.append("")

        summary_lines.append("")

    # Summary statistics
    summary_lines.extend([
        "## Model Availability Summary",
        ""
    ])

    # Count available models by type
    model_counts = {
        'regressor': 0,
        'gbm': 0,
        'classifier': 0,
        'quantile': 0
    }

    for symbol in results['symbols'].values():
        for test_name in model_counts.keys():
            if symbol.get(test_name, {}).get('passed', False):
                model_counts[test_name] += 1

    summary_lines.append("| Model Type | Available |")
    summary_lines.append("|------------|-----------|")
    for model_type, count in sorted(model_counts.items()):
        summary_lines.append(f"| {model_type.title()} | {count}/{results['metadata']['total_symbols']} |")

    summary_lines.extend([
        "",
        "## Key Metrics Analysis",
        ""
    ])

    # Aggregate metrics
    aggregate_metrics = defaultdict(list)

    for symbol_tests in results['symbols'].values():
        for test_name, test_result in symbol_tests.items():
            metrics = test_result.get('metrics', {})
            for key, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    aggregate_metrics[f"{test_name}_{key}"].append(value)

    if aggregate_metrics:
        summary_lines.append("| Metric | Min | Mean | Max |")
        summary_lines.append("|--------|-----|------|-----|")

        for metric_name in sorted(aggregate_metrics.keys()):
            values = aggregate_metrics[metric_name]
            if values:
                min_val = min(values)
                mean_val = sum(values) / len(values)
                max_val = max(values)

                summary_lines.append(f"| {metric_name} | {min_val:.4f} | {mean_val:.4f} | {max_val:.4f} |")

    summary_lines.extend([
        "",
        "---",
        "",
        f"*Generated by F4 Comprehensive Validation Orchestration*",
        f"*Framework: {results['metadata']['validation_framework']}*"
    ])

    # Write summary
    summary_file = RESULTS_DIR / "VALIDATION_SUMMARY.md"
    with open(summary_file, 'w') as f:
        f.write('\n'.join(summary_lines))

    print(f"Summary saved to: {summary_file}")

    return summary_file


def main():
    """Main entry point."""
    try:
        # Run all validations
        results = run_all_validations()

        # Save results
        results_file = save_results(results)

        # Generate summary
        summary_file = generate_markdown_summary(results)

        print("\n" + "="*80)
        print("SUCCESS: F4 Comprehensive Validation Complete")
        print("="*80)
        print(f"Results: {results_file}")
        print(f"Summary: {summary_file}")
        print("="*80 + "\n")

        return 0

    except Exception as e:
        print(f"\nERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
