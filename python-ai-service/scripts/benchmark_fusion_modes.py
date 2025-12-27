#!/usr/bin/env python3
"""
Performance Benchmarking and Comparison Suite for Fusion Modes

Compares all available fusion modes (weighted, balanced, gbm_heavy, lstm_heavy, classifier, gbm_only)
across key performance metrics:
- Sharpe Ratio
- Maximum Drawdown
- Directional Accuracy
- Total Return
- Information Coefficient
- Win Rate

Outputs comparison table in markdown and JSON formats.
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# Add parent directory to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from data.data_fetcher import fetch_stock_data
from data.feature_engineer import engineer_features
from evaluation.advanced_backtester import AdvancedBacktester
import pickle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Fusion modes to benchmark
FUSION_MODES = ['weighted', 'balanced', 'gbm_heavy', 'lstm_heavy', 'classifier', 'gbm_only']

def run_fusion_backtest(
    symbol: str,
    fusion_mode: str,
    backtest_days: int = 360,
    initial_capital: float = 10000.0
) -> Optional[Dict]:
    """
    Run a single backtest for a given fusion mode.

    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        fusion_mode: Fusion mode to test
        backtest_days: Number of most recent days to backtest (0 = all)
        initial_capital: Initial capital for backtest

    Returns:
        Dictionary of backtest metrics, or None if failed
    """
    try:
        logger.info(f"Running backtest for {symbol} with mode '{fusion_mode}'...")

        # Import inference script functionality
        sys.path.insert(0, str(PROJECT_ROOT))
        from inference_and_backtest import main

        # Run backtest via main function
        result = main(
            symbol=symbol,
            backtest_days=backtest_days,
            fusion_mode=fusion_mode,
            use_cache=True,
            force_refresh=False,
            skip_regressor_backtest=False,
            get_data=False  # Don't print analysis
        )

        if result is None:
            logger.warning(f"Backtest failed for {symbol} with mode '{fusion_mode}'")
            return None

        return result

    except Exception as e:
        logger.error(f"Error running backtest for {symbol} with mode '{fusion_mode}': {e}")
        import traceback
        traceback.print_exc()
        return None

def calculate_metrics(result: Dict) -> Dict[str, float]:
    """
    Extract and calculate comparison metrics from backtest result.

    Args:
        result: Backtest result dictionary

    Returns:
        Dictionary of metrics
    """
    metrics = {}

    # Direct metrics from backtest
    if 'metrics' in result:
        bt_metrics = result['metrics']

        # Sharpe Ratio
        metrics['sharpe_ratio'] = bt_metrics.get('sharpe_ratio', 0.0)

        # Maximum Drawdown
        metrics['max_drawdown'] = bt_metrics.get('max_drawdown', 0.0)

        # Return
        metrics['total_return'] = bt_metrics.get('total_return', 0.0)

        # Win Rate
        metrics['win_rate'] = bt_metrics.get('win_rate', 0.0)

        # Directional Accuracy (if available)
        metrics['directional_accuracy'] = bt_metrics.get('directional_accuracy', 0.0)

        # Information Coefficient (if available)
        metrics['information_coefficient'] = bt_metrics.get('information_coefficient', 0.0)

        # Profit Factor
        metrics['profit_factor'] = bt_metrics.get('profit_factor', 0.0)

        # Number of trades
        metrics['num_trades'] = bt_metrics.get('num_trades', 0)

        # Average Trade Return
        metrics['avg_trade_return'] = bt_metrics.get('avg_trade_return', 0.0)

    # Calculate additional metrics if needed
    if 'daily_returns' in result:
        returns = np.array(result['daily_returns'])
        metrics['volatility'] = np.std(returns) * np.sqrt(252)  # Annualized
        metrics['skewness'] = pd.Series(returns).skew()
        metrics['kurtosis'] = pd.Series(returns).kurtosis()

    return metrics

def format_metrics_table(results: Dict[str, Dict[str, float]]) -> str:
    """
    Format benchmark results as markdown table.

    Args:
        results: Dictionary mapping fusion mode to metrics

    Returns:
        Markdown formatted table string
    """
    # Define key metrics to display
    key_metrics = [
        'total_return',
        'sharpe_ratio',
        'max_drawdown',
        'directional_accuracy',
        'win_rate',
        'profit_factor',
        'num_trades',
        'volatility'
    ]

    # Create DataFrame
    df = pd.DataFrame(results).T

    # Filter to available metrics
    available_metrics = [m for m in key_metrics if m in df.columns]
    df = df[available_metrics]

    # Sort by Sharpe ratio (descending)
    if 'sharpe_ratio' in df.columns:
        df = df.sort_values('sharpe_ratio', ascending=False)

    # Format numeric columns
    format_dict = {
        'total_return': '{:.2%}',
        'sharpe_ratio': '{:.3f}',
        'max_drawdown': '{:.2%}',
        'directional_accuracy': '{:.2%}',
        'win_rate': '{:.2%}',
        'profit_factor': '{:.2f}',
        'volatility': '{:.2%}',
        'num_trades': '{:.0f}'
    }

    # Format the dataframe
    formatted_df = df.copy()
    for col, fmt in format_dict.items():
        if col in formatted_df.columns:
            if formatted_df[col].dtype in [float, np.floating]:
                formatted_df[col] = formatted_df[col].apply(
                    lambda x: fmt.format(x) if pd.notna(x) else 'N/A'
                )

    # Convert to markdown table
    markdown_table = formatted_df.to_markdown()

    return markdown_table

def benchmark_fusion_modes(
    symbol: str,
    backtest_days: int = 360,
    output_dir: str = 'benchmarks',
    modes: Optional[List[str]] = None
) -> Dict:
    """
    Benchmark all fusion modes for a given symbol.

    Args:
        symbol: Stock symbol
        backtest_days: Number of most recent days to backtest (0 = all)
        output_dir: Directory to save results
        modes: List of modes to test (defaults to all)

    Returns:
        Dictionary with benchmark results and comparison
    """
    if modes is None:
        modes = FUSION_MODES

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Starting benchmark for {symbol}")
    logger.info(f"Testing {len(modes)} fusion modes: {', '.join(modes)}")
    logger.info(f"Backtest period: {backtest_days} days")

    results = {}
    successful_modes = []
    failed_modes = []

    # Run backtest for each mode
    for mode in modes:
        logger.info(f"\n{'='*60}")
        logger.info(f"Testing fusion mode: {mode}")
        logger.info(f"{'='*60}")

        result = run_fusion_backtest(
            symbol=symbol,
            fusion_mode=mode,
            backtest_days=backtest_days
        )

        if result is not None:
            metrics = calculate_metrics(result)
            results[mode] = metrics
            successful_modes.append(mode)
            logger.info(f"✓ {mode}: Sharpe={metrics.get('sharpe_ratio', 0):.3f}, "
                       f"Return={metrics.get('total_return', 0):.2%}, "
                       f"MaxDD={metrics.get('max_drawdown', 0):.2%}")
        else:
            failed_modes.append(mode)
            logger.warning(f"✗ {mode}: Backtest failed")

    # Generate comparison report
    logger.info(f"\n{'='*60}")
    logger.info("BENCHMARK SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Symbol: {symbol}")
    logger.info(f"Backtest period: {backtest_days} days")
    logger.info(f"Successful modes: {len(successful_modes)}/{len(modes)}")
    if failed_modes:
        logger.info(f"Failed modes: {', '.join(failed_modes)}")

    # Create comparison table
    if results:
        markdown_table = format_metrics_table(results)
        logger.info(f"\n{markdown_table}")

    # Determine best mode
    best_mode = None
    best_sharpe = -np.inf
    if results:
        for mode, metrics in results.items():
            sharpe = metrics.get('sharpe_ratio', -np.inf)
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_mode = mode

        logger.info(f"\nBest Mode: {best_mode} (Sharpe: {best_sharpe:.3f})")

    # Prepare output dictionary
    output = {
        'symbol': symbol,
        'backtest_days': backtest_days,
        'timestamp': datetime.now().isoformat(),
        'modes_tested': modes,
        'successful_modes': successful_modes,
        'failed_modes': failed_modes,
        'results': results,
        'best_mode': best_mode,
        'best_sharpe': float(best_sharpe) if best_sharpe != -np.inf else None,
        'markdown_table': format_metrics_table(results) if results else None
    }

    # Save results
    results_file = os.path.join(output_dir, f'{symbol}_benchmark_results.json')
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {results_file}")

    # Save markdown table
    if output['markdown_table']:
        md_file = os.path.join(output_dir, f'{symbol}_benchmark_comparison.md')
        with open(md_file, 'w') as f:
            f.write(f"# {symbol} Fusion Mode Benchmark\n\n")
            f.write(f"**Period**: Last {backtest_days} trading days\n")
            f.write(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"## Results Summary\n\n")
            f.write(f"**Best Mode**: {best_mode} (Sharpe: {best_sharpe:.3f})\n\n")
            f.write(f"## Performance Comparison\n\n")
            f.write(output['markdown_table'])
            f.write(f"\n\n## Modes Tested\n\n")
            f.write(f"- Successful: {', '.join(successful_modes)}\n")
            if failed_modes:
                f.write(f"- Failed: {', '.join(failed_modes)}\n")
        logger.info(f"Markdown report saved to: {md_file}")

    return output

def create_comparison_charts(
    results: Dict[str, Dict[str, float]],
    symbol: str,
    output_dir: str = 'benchmarks'
) -> None:
    """
    Create visualization charts comparing fusion modes.

    Args:
        results: Dictionary of mode results
        symbol: Stock symbol
        output_dir: Directory to save charts
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available, skipping chart generation")
        return

    os.makedirs(output_dir, exist_ok=True)

    # Prepare data
    modes = list(results.keys())
    sharpe_ratios = [results[m].get('sharpe_ratio', 0) for m in modes]
    max_drawdowns = [results[m].get('max_drawdown', 0) for m in modes]
    total_returns = [results[m].get('total_return', 0) for m in modes]
    win_rates = [results[m].get('win_rate', 0) for m in modes]

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{symbol} - Fusion Mode Performance Comparison', fontsize=16, fontweight='bold')

    # Sharpe Ratio
    ax = axes[0, 0]
    colors = ['green' if x > 0 else 'red' for x in sharpe_ratios]
    ax.bar(modes, sharpe_ratios, color=colors, alpha=0.7)
    ax.set_title('Sharpe Ratio (Higher is Better)')
    ax.set_ylabel('Sharpe Ratio')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)

    # Max Drawdown
    ax = axes[0, 1]
    colors = ['green' if x < 0 else 'red' for x in max_drawdowns]
    ax.bar(modes, max_drawdowns, color=colors, alpha=0.7)
    ax.set_title('Maximum Drawdown (Lower is Better)')
    ax.set_ylabel('Max Drawdown')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)

    # Total Return
    ax = axes[1, 0]
    colors = ['green' if x > 0 else 'red' for x in total_returns]
    ax.bar(modes, total_returns, color=colors, alpha=0.7)
    ax.set_title('Total Return (Higher is Better)')
    ax.set_ylabel('Return')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(axis='y', alpha=0.3)

    # Win Rate
    ax = axes[1, 1]
    ax.bar(modes, win_rates, color='steelblue', alpha=0.7)
    ax.set_title('Win Rate (Higher is Better)')
    ax.set_ylabel('Win Rate')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.1%}'.format(y)))
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    # Save figure
    chart_file = os.path.join(output_dir, f'{symbol}_benchmark_comparison.png')
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    logger.info(f"Chart saved to: {chart_file}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(
        description='Benchmark fusion modes for a stock symbol'
    )
    parser.add_argument('--symbol', required=True, help='Stock symbol (e.g., AAPL)')
    parser.add_argument(
        '--backtest-days',
        type=int,
        default=360,
        help='Number of most recent trading days to backtest (default: 360, 0=all)'
    )
    parser.add_argument(
        '--output-dir',
        default='benchmarks',
        help='Output directory for results (default: benchmarks)'
    )
    parser.add_argument(
        '--modes',
        nargs='+',
        default=FUSION_MODES,
        help=f'Fusion modes to test (default: {" ".join(FUSION_MODES)})'
    )
    parser.add_argument(
        '--skip-charts',
        action='store_true',
        help='Skip chart generation'
    )

    args = parser.parse_args()

    # Run benchmark
    output = benchmark_fusion_modes(
        symbol=args.symbol,
        backtest_days=args.backtest_days,
        output_dir=args.output_dir,
        modes=args.modes
    )

    # Create charts if requested
    if not args.skip_charts and output['results']:
        create_comparison_charts(
            results=output['results'],
            symbol=args.symbol,
            output_dir=args.output_dir
        )

    # Final summary
    logger.info(f"\n{'='*60}")
    logger.info("BENCHMARK COMPLETE")
    logger.info(f"{'='*60}")
    logger.info(f"Results saved to: {args.output_dir}")
    if output['best_mode']:
        logger.info(f"Recommended mode: {output['best_mode']} (Sharpe: {output['best_sharpe']:.3f})")

    return output

if __name__ == '__main__':
    main()
