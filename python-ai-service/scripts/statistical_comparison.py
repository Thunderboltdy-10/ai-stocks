#!/usr/bin/env python
"""
Statistical comparison of fusion modes with paired bootstrap tests.

P0.3: Implements statistical significance testing for comparing backtest results
across different fusion modes. Uses paired bootstrap to compute confidence intervals
and p-values for Sharpe ratio differences.

Usage:
    python scripts/statistical_comparison.py --modes regressor_only balanced gbm_only --symbol AAPL
    python scripts/statistical_comparison.py --modes regressor_only balanced --baseline regressor_only
    python scripts/statistical_comparison.py --from-pickle path/to/backtest.pkl --compare path/to/other.pkl

References:
- Ledoit & Wolf (2008): "Robust Performance Hypothesis Testing with the Sharpe Ratio"
- DiBartolomeo (2007): "On the Estimation of Transaction Costs"
"""

import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from typing import Dict, List, Tuple, Optional, Any
import pickle
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import matplotlib
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    PLT_AVAILABLE = True
except ImportError:
    PLT_AVAILABLE = False
    logger.warning("matplotlib not available for plotting")


# ============================================================================
# BOOTSTRAP UTILITIES
# ============================================================================

def compute_sharpe_ratio(returns: np.ndarray, annualization: int = 252) -> float:
    """
    Compute annualized Sharpe ratio.
    
    Args:
        returns: Array of period returns (daily assumed)
        annualization: Number of periods per year (252 for daily)
    
    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(annualization)


def paired_bootstrap_sharpe(
    returns_a: np.ndarray,
    returns_b: np.ndarray,
    n_bootstrap: int = 10000,
    random_seed: int = 42,
    annualization: int = 252
) -> Dict[str, Any]:
    """
    Paired bootstrap test for Sharpe ratio difference.
    
    H0: Sharpe(A) = Sharpe(B)
    
    Uses paired bootstrap to preserve the temporal correlation between
    the two return series (same market conditions).
    
    Args:
        returns_a: Returns for strategy A
        returns_b: Returns for strategy B
        n_bootstrap: Number of bootstrap samples
        random_seed: Random seed for reproducibility
        annualization: Number of periods per year
    
    Returns:
        Dict with test results including p-value and confidence intervals
    """
    np.random.seed(random_seed)
    
    # Ensure same length
    n = min(len(returns_a), len(returns_b))
    returns_a = np.asarray(returns_a)[:n]
    returns_b = np.asarray(returns_b)[:n]
    
    # Observed Sharpe ratios
    sharpe_a = compute_sharpe_ratio(returns_a, annualization)
    sharpe_b = compute_sharpe_ratio(returns_b, annualization)
    observed_diff = sharpe_a - sharpe_b
    
    # Bootstrap resampling (paired: same indices for both)
    boot_diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        boot_a = returns_a[idx]
        boot_b = returns_b[idx]
        
        s_a = compute_sharpe_ratio(boot_a, annualization)
        s_b = compute_sharpe_ratio(boot_b, annualization)
        boot_diffs.append(s_a - s_b)
    
    boot_diffs = np.array(boot_diffs)
    
    # Two-tailed p-value: probability of seeing difference as extreme as observed
    # under the null hypothesis that true difference is 0
    # We use the centered bootstrap distribution
    centered_diffs = boot_diffs - np.mean(boot_diffs)
    p_value = np.mean(np.abs(centered_diffs) >= np.abs(observed_diff))
    
    # Alternative: p-value for difference from zero
    # p_value = 2 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0))
    
    # Confidence intervals
    ci_lower, ci_upper = np.percentile(boot_diffs, [2.5, 97.5])
    ci_90_lower, ci_90_upper = np.percentile(boot_diffs, [5, 95])
    
    # Standard error
    std_error = np.std(boot_diffs)
    
    return {
        'sharpe_a': float(sharpe_a),
        'sharpe_b': float(sharpe_b),
        'observed_diff': float(observed_diff),
        'p_value': float(p_value),
        'ci_95': (float(ci_lower), float(ci_upper)),
        'ci_90': (float(ci_90_lower), float(ci_90_upper)),
        'std_error': float(std_error),
        'n_samples': n,
        'n_bootstrap': n_bootstrap,
        'boot_diffs': boot_diffs  # For plotting
    }


def bootstrap_sharpe_ci(
    returns: np.ndarray,
    n_bootstrap: int = 10000,
    random_seed: int = 42,
    annualization: int = 252
) -> Dict[str, Any]:
    """
    Bootstrap confidence interval for a single Sharpe ratio.
    
    Args:
        returns: Array of period returns
        n_bootstrap: Number of bootstrap samples
        random_seed: Random seed
        annualization: Periods per year
    
    Returns:
        Dict with Sharpe estimate and confidence intervals
    """
    np.random.seed(random_seed)
    n = len(returns)
    returns = np.asarray(returns)
    
    # Observed Sharpe
    sharpe = compute_sharpe_ratio(returns, annualization)
    
    # Bootstrap
    boot_sharpes = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, size=n, replace=True)
        boot_ret = returns[idx]
        boot_sharpes.append(compute_sharpe_ratio(boot_ret, annualization))
    
    boot_sharpes = np.array(boot_sharpes)
    ci_lower, ci_upper = np.percentile(boot_sharpes, [2.5, 97.5])
    
    return {
        'sharpe': float(sharpe),
        'ci_95': (float(ci_lower), float(ci_upper)),
        'std_error': float(np.std(boot_sharpes)),
        'n_samples': n,
    }


# ============================================================================
# DATA LOADING
# ============================================================================

def load_backtest_pickle(filepath: str) -> Dict[str, Any]:
    """Load a backtest results pickle file."""
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return data


def extract_daily_returns(backtest_data: Dict) -> np.ndarray:
    """
    Extract daily strategy returns from backtest data.
    
    Tries multiple keys that might contain returns:
    - 'returns': direct returns array
    - 'equity': equity curve to compute returns
    - 'daily_returns': explicit daily returns
    """
    if 'returns' in backtest_data:
        return np.asarray(backtest_data['returns'])
    
    if 'daily_returns' in backtest_data:
        return np.asarray(backtest_data['daily_returns'])
    
    if 'equity' in backtest_data:
        equity = np.asarray(backtest_data['equity'])
        # Compute returns from equity curve
        returns = np.diff(equity) / equity[:-1]
        return returns
    
    if 'equity_curve' in backtest_data:
        equity = np.asarray(backtest_data['equity_curve'])
        returns = np.diff(equity) / equity[:-1]
        return returns
    
    raise ValueError("Could not find returns in backtest data")


def load_backtest_results_from_dir(
    symbol: str,
    mode: str,
    base_dir: str = 'backtest_results'
) -> Tuple[Optional[Dict], Optional[np.ndarray]]:
    """
    Load backtest results for a specific symbol and fusion mode.
    
    Looks for files matching the pattern:
    - {base_dir}/{symbol}_{mode}_*.pkl
    - {base_dir}/{mode}_backtest_*.pkl
    
    Returns:
        Tuple of (backtest_data dict, daily_returns array)
    """
    base_path = Path(base_dir)
    
    # Try different naming patterns
    patterns = [
        f'{symbol}_{mode}_*.pkl',
        f'{mode}_backtest_*.pkl',
        f'{symbol}_backtest_*{mode}*.pkl',
    ]
    
    for pattern in patterns:
        matches = list(base_path.glob(pattern))
        if matches:
            # Use most recent file
            filepath = max(matches, key=lambda p: p.stat().st_mtime)
            logger.info(f"Loading {mode}: {filepath}")
            data = load_backtest_pickle(str(filepath))
            try:
                returns = extract_daily_returns(data)
                return data, returns
            except ValueError as e:
                logger.warning(f"Could not extract returns from {filepath}: {e}")
    
    logger.warning(f"No backtest file found for {symbol}/{mode} in {base_dir}")
    return None, None


# ============================================================================
# PLOTTING
# ============================================================================

def plot_bootstrap_distribution(
    result: Dict[str, Any],
    mode_a: str,
    mode_b: str,
    output_path: str
) -> str:
    """
    Plot bootstrap sampling distribution for Sharpe ratio difference.
    
    Shows:
    - Histogram of bootstrap differences
    - Observed difference (vertical line)
    - 95% confidence interval (shaded)
    - p-value annotation
    """
    if not PLT_AVAILABLE:
        logger.warning("matplotlib not available, skipping plot")
        return None
    
    boot_diffs = result['boot_diffs']
    observed = result['observed_diff']
    ci = result['ci_95']
    p_value = result['p_value']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Histogram
    ax.hist(boot_diffs, bins=50, alpha=0.7, color='steelblue', edgecolor='white',
            label='Bootstrap Distribution')
    
    # Observed difference
    ax.axvline(observed, color='red', linestyle='--', linewidth=2,
               label=f'Observed Δ: {observed:.3f}')
    
    # Zero line
    ax.axvline(0, color='black', linestyle='-', linewidth=1, alpha=0.5,
               label='No Difference')
    
    # Confidence interval
    ax.axvline(ci[0], color='orange', linestyle=':', linewidth=2)
    ax.axvline(ci[1], color='orange', linestyle=':', linewidth=2,
               label=f'95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]')
    
    # Fill CI region
    ylim = ax.get_ylim()
    ax.axvspan(ci[0], ci[1], alpha=0.2, color='orange')
    ax.set_ylim(ylim)
    
    # Labels
    ax.set_xlabel('Sharpe Ratio Difference (A - B)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'Bootstrap Test: {mode_a} vs {mode_b}\n'
                 f'p-value: {p_value:.4f}')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Significance annotation
    if p_value < 0.001:
        sig_text = '*** p < 0.001'
    elif p_value < 0.01:
        sig_text = '** p < 0.01'
    elif p_value < 0.05:
        sig_text = '* p < 0.05'
    elif p_value < 0.10:
        sig_text = '† p < 0.10'
    else:
        sig_text = 'n.s.'
    
    ax.annotate(sig_text, xy=(0.02, 0.98), xycoords='axes fraction',
                fontsize=14, fontweight='bold', va='top')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved bootstrap plot: {output_path}")
    return output_path


def plot_sharpe_comparison(
    results: Dict[str, Dict],
    output_path: str
) -> str:
    """
    Plot comparison of Sharpe ratios across modes with error bars.
    """
    if not PLT_AVAILABLE:
        logger.warning("matplotlib not available, skipping plot")
        return None
    
    modes = list(results.keys())
    sharpes = [results[m]['sharpe'] for m in modes]
    ci_lowers = [results[m]['ci_95'][0] for m in modes]
    ci_uppers = [results[m]['ci_95'][1] for m in modes]
    
    errors = [[s - l for s, l in zip(sharpes, ci_lowers)],
              [u - s for s, u in zip(sharpes, ci_uppers)]]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(modes))
    bars = ax.bar(x, sharpes, yerr=errors, capsize=5, color='steelblue',
                  edgecolor='white', alpha=0.8)
    
    # Color bars by significance vs first mode
    # (positive = green, negative = red, not significant = blue)
    
    ax.set_xlabel('Fusion Mode')
    ax.set_ylabel('Annualized Sharpe Ratio')
    ax.set_title('Sharpe Ratio Comparison with 95% Confidence Intervals')
    ax.set_xticks(x)
    ax.set_xticklabels(modes, rotation=45, ha='right')
    ax.axhline(0, color='black', linestyle='-', linewidth=0.5)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved comparison plot: {output_path}")
    return output_path


# ============================================================================
# MAIN COMPARISON FUNCTIONS
# ============================================================================

def compare_modes(
    modes: List[str],
    symbol: str,
    baseline: Optional[str] = None,
    base_dir: str = 'backtest_results',
    output_dir: str = 'results/statistical_tests',
    n_bootstrap: int = 10000
) -> Dict[str, Any]:
    """
    Compare multiple fusion modes using paired bootstrap tests.
    
    Args:
        modes: List of fusion modes to compare
        symbol: Stock ticker symbol
        baseline: Mode to use as baseline (default: first mode)
        base_dir: Directory containing backtest results
        output_dir: Directory for output files
        n_bootstrap: Number of bootstrap iterations
    
    Returns:
        Dict with all comparison results
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    baseline = baseline or modes[0]
    if baseline not in modes:
        modes = [baseline] + modes
    
    # Load data for all modes
    logger.info(f"Loading backtest results for {symbol}...")
    data = {}
    for mode in modes:
        backtest, returns = load_backtest_results_from_dir(symbol, mode, base_dir)
        if returns is not None:
            data[mode] = {
                'backtest': backtest,
                'returns': returns
            }
            logger.info(f"  {mode}: {len(returns)} days, "
                       f"mean={returns.mean():.6f}, std={returns.std():.6f}")
        else:
            logger.warning(f"  {mode}: No data found")
    
    if len(data) < 2:
        raise ValueError(f"Need at least 2 modes with data, found {len(data)}")
    
    if baseline not in data:
        raise ValueError(f"Baseline mode '{baseline}' not found in data")
    
    results = {
        'symbol': symbol,
        'baseline': baseline,
        'n_bootstrap': n_bootstrap,
        'timestamp': datetime.now().isoformat(),
        'individual_sharpes': {},
        'pairwise_comparisons': {},
    }
    
    # Compute individual Sharpe CIs
    logger.info("\nComputing individual Sharpe ratios...")
    for mode in data:
        sharpe_result = bootstrap_sharpe_ci(data[mode]['returns'], n_bootstrap)
        results['individual_sharpes'][mode] = sharpe_result
        logger.info(f"  {mode}: Sharpe = {sharpe_result['sharpe']:.3f} "
                   f"[{sharpe_result['ci_95'][0]:.3f}, {sharpe_result['ci_95'][1]:.3f}]")
    
    # Pairwise comparisons
    logger.info(f"\nPairwise comparisons (baseline: {baseline})")
    logger.info("=" * 70)
    
    baseline_returns = data[baseline]['returns']
    
    for mode in data:
        if mode == baseline:
            continue
        
        mode_returns = data[mode]['returns']
        
        # Paired bootstrap test
        test_result = paired_bootstrap_sharpe(
            baseline_returns,
            mode_returns,
            n_bootstrap=n_bootstrap
        )
        
        # Remove bootstrap samples from result (for JSON serialization)
        test_result_clean = {k: v for k, v in test_result.items() if k != 'boot_diffs'}
        results['pairwise_comparisons'][f'{baseline}_vs_{mode}'] = test_result_clean
        
        # Print results
        print(f"\n{baseline} vs {mode}:")
        print(f"  Sharpe ({baseline}): {test_result['sharpe_a']:.3f}")
        print(f"  Sharpe ({mode}):     {test_result['sharpe_b']:.3f}")
        print(f"  Difference:         {test_result['observed_diff']:+.3f}")
        print(f"  95% CI:             [{test_result['ci_95'][0]:+.3f}, {test_result['ci_95'][1]:+.3f}]")
        print(f"  p-value:            {test_result['p_value']:.4f}")
        
        # Significance interpretation
        if test_result['p_value'] < 0.001:
            sig = "*** HIGHLY SIGNIFICANT *** (p < 0.001)"
        elif test_result['p_value'] < 0.01:
            sig = "** SIGNIFICANT ** (p < 0.01)"
        elif test_result['p_value'] < 0.05:
            sig = "* SIGNIFICANT * (p < 0.05)"
        elif test_result['p_value'] < 0.10:
            sig = "† MARGINALLY SIGNIFICANT † (p < 0.10)"
        else:
            sig = "NOT SIGNIFICANT (p ≥ 0.10)"
        print(f"  Conclusion: {sig}")
        
        # Plot bootstrap distribution
        plot_path = output_path / f'{symbol}_{baseline}_vs_{mode}_bootstrap.png'
        plot_bootstrap_distribution(test_result, baseline, mode, str(plot_path))
    
    # Plot overall comparison
    comparison_plot_path = output_path / f'{symbol}_sharpe_comparison.png'
    plot_sharpe_comparison(results['individual_sharpes'], str(comparison_plot_path))
    
    # Save results JSON
    results_json_path = output_path / f'{symbol}_statistical_tests.json'
    with open(results_json_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved: {results_json_path}")
    
    return results


def compare_pickles(
    pickle_a: str,
    pickle_b: str,
    label_a: str = 'Strategy A',
    label_b: str = 'Strategy B',
    output_dir: str = 'results/statistical_tests',
    n_bootstrap: int = 10000
) -> Dict[str, Any]:
    """
    Compare two backtest pickle files directly.
    
    Useful for comparing specific backtest runs without mode naming conventions.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load data
    data_a = load_backtest_pickle(pickle_a)
    data_b = load_backtest_pickle(pickle_b)
    
    returns_a = extract_daily_returns(data_a)
    returns_b = extract_daily_returns(data_b)
    
    logger.info(f"Loaded {label_a}: {len(returns_a)} days")
    logger.info(f"Loaded {label_b}: {len(returns_b)} days")
    
    # Run comparison
    test_result = paired_bootstrap_sharpe(returns_a, returns_b, n_bootstrap)
    
    # Print and plot
    print(f"\n{label_a} vs {label_b}:")
    print(f"  Sharpe ({label_a}): {test_result['sharpe_a']:.3f}")
    print(f"  Sharpe ({label_b}): {test_result['sharpe_b']:.3f}")
    print(f"  Difference:        {test_result['observed_diff']:+.3f}")
    print(f"  95% CI:            [{test_result['ci_95'][0]:+.3f}, {test_result['ci_95'][1]:+.3f}]")
    print(f"  p-value:           {test_result['p_value']:.4f}")
    
    # Plot
    plot_path = output_path / f'comparison_{label_a}_vs_{label_b}_bootstrap.png'
    plot_bootstrap_distribution(test_result, label_a, label_b, str(plot_path))
    
    return test_result


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Statistical comparison of backtest results using paired bootstrap tests',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare multiple fusion modes
  python statistical_comparison.py --modes regressor_only balanced gbm_only --symbol AAPL
  
  # Compare with specific baseline
  python statistical_comparison.py --modes regressor_only lstm_heavy balanced --baseline regressor_only
  
  # Compare two specific pickle files
  python statistical_comparison.py --from-pickle path/to/a.pkl --compare path/to/b.pkl
        """
    )
    
    parser.add_argument('--modes', nargs='+', help='Fusion modes to compare')
    parser.add_argument('--symbol', default='AAPL', help='Stock ticker symbol')
    parser.add_argument('--baseline', help='Baseline mode for comparison (default: first mode)')
    parser.add_argument('--output', default='results/statistical_tests', help='Output directory')
    parser.add_argument('--n-bootstrap', type=int, default=10000, help='Bootstrap iterations')
    parser.add_argument('--base-dir', default='backtest_results', help='Directory with backtest files')
    parser.add_argument('--from-pickle', help='Path to first backtest pickle file')
    parser.add_argument('--compare', help='Path to second backtest pickle file')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.from_pickle and args.compare:
        # Direct pickle comparison
        compare_pickles(
            args.from_pickle,
            args.compare,
            label_a=Path(args.from_pickle).stem,
            label_b=Path(args.compare).stem,
            output_dir=args.output,
            n_bootstrap=args.n_bootstrap
        )
    elif args.modes:
        # Mode comparison
        compare_modes(
            modes=args.modes,
            symbol=args.symbol,
            baseline=args.baseline,
            base_dir=args.base_dir,
            output_dir=args.output,
            n_bootstrap=args.n_bootstrap
        )
    else:
        parser.print_help()
        print("\nError: Specify either --modes or --from-pickle + --compare")
        return 1
    
    print("\n✓ Statistical comparison complete")
    return 0


if __name__ == '__main__':
    exit(main())
