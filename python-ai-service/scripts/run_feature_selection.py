#!/usr/bin/env python
"""
CLI wrapper for running feature selection across multiple symbols.

This script provides a user-friendly interface for running Random Forest
feature selection on one or more stock symbols with progress tracking,
parallel processing, and summary reporting.

Usage:
    python scripts/run_feature_selection.py AAPL TSLA MSFT --n-features 40
    python scripts/run_feature_selection.py AAPL --n-features 30 --parallel
    python scripts/run_feature_selection.py AAPL TSLA --output-dir custom_dir

Features:
    - Single or multiple symbol processing
    - Progress bar for long-running operations
    - Parallel processing with --parallel flag
    - Summary table at completion
    - Error handling with continued processing
"""

import argparse
import sys
import io
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Force UTF-8 encoding for Windows compatibility
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Ensure the python-ai-service package is on sys.path
python_ai_service_path = Path(__file__).resolve().parents[1]
if str(python_ai_service_path) not in sys.path:
    sys.path.insert(0, str(python_ai_service_path))

import numpy as np

# Try to import tqdm for progress bars
try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    tqdm = None

# Try to import tabulate for summary table
try:
    from tabulate import tabulate
    HAS_TABULATE = True
except ImportError:
    HAS_TABULATE = False
    tabulate = None


@dataclass
class SymbolResult:
    """Result of feature selection for a single symbol."""
    symbol: str
    success: bool
    r2_score: float = 0.0
    mae: float = 0.0
    n_features: int = 0
    coverage: float = 0.0
    top_feature: str = ""
    validation_passed: bool = False
    error_message: str = ""
    elapsed_time: float = 0.0


def print_colored(text: str, color: str = "default") -> None:
    """Print text with optional color (if terminal supports it)."""
    colors = {
        "green": "\033[92m",
        "red": "\033[91m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "cyan": "\033[96m",
        "default": "\033[0m",
        "bold": "\033[1m",
    }
    reset = "\033[0m"
    
    # Check if terminal supports colors
    try:
        import os
        if os.name == 'nt':
            # Enable ANSI colors on Windows
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
    except Exception:
        pass
    
    color_code = colors.get(color, colors["default"])
    print(f"{color_code}{text}{reset}")


def process_symbol(
    symbol: str,
    n_features: int,
    output_dir: str,
    verbose: bool = False,
) -> SymbolResult:
    """
    Run feature selection for a single symbol.
    
    Args:
        symbol: Stock symbol to process.
        n_features: Number of features to select.
        output_dir: Output directory for artifacts.
        verbose: Whether to print detailed logs.
    
    Returns:
        SymbolResult with processing outcome.
    """
    start_time = time.time()
    
    try:
        # Import here to avoid slow imports at script start
        from data.data_fetcher import fetch_stock_data
        from data.feature_engineer import engineer_features
        from data.target_engineering import prepare_training_data
        from training.feature_selection import (
            train_feature_selector,
            validate_selected_features,
            load_feature_importance,
        )
        
        # Step 1: Fetch data
        if verbose:
            print(f"   [DATA] Fetching data for {symbol}...")
        df = fetch_stock_data(symbol)
        
        if df is None or len(df) < 100:
            raise ValueError(f"Insufficient data for {symbol}: {len(df) if df is not None else 0} rows")
        
        # Step 2: Engineer features
        if verbose:
            print(f"   [BUILD] Engineering features...")
        df = engineer_features(df, include_sentiment=True)
        
        # Step 3: Prepare training data
        if verbose:
            print(f"   [TARGET] Preparing training data...")
        df_clean, feature_cols = prepare_training_data(df, horizons=[1])
        
        X = df_clean[feature_cols].values
        y = df_clean['target_1d'].values
        
        if len(X) < 100:
            raise ValueError(f"Insufficient clean samples for {symbol}: {len(X)}")
        
        # Step 4: Run feature selection
        if verbose:
            print(f"   [RF] Training Random Forest...")
        
        # Suppress logging during parallel runs
        import logging
        if not verbose:
            logging.getLogger('training.feature_selection').setLevel(logging.WARNING)
        
        report = train_feature_selector(
            X=X,
            y=y,
            symbol=symbol,
            feature_names=feature_cols,
            n_top_features=n_features,
            output_dir=output_dir,
        )
        
        # Get top feature from importance
        try:
            importance_df = load_feature_importance(symbol, output_dir)
            # Filter to selected features
            selected_set = set(report.selected_features)
            top_features = importance_df[importance_df['feature'].isin(selected_set)]
            top_feature = top_features.iloc[0]['feature'] if len(top_features) > 0 else "N/A"
        except Exception:
            top_feature = report.selected_features[0] if report.selected_features else "N/A"
        
        # Validate
        validation_result = validate_selected_features(
            report.selected_features,
            feature_cols,
        )
        
        elapsed = time.time() - start_time
        
        return SymbolResult(
            symbol=symbol,
            success=True,
            r2_score=report.validation_r2,
            mae=report.validation_mae,
            n_features=report.n_features_selected,
            coverage=report.cumulative_importance,
            top_feature=top_feature,
            validation_passed=validation_result.validation_passed,
            elapsed_time=elapsed,
        )
        
    except Exception as e:
        elapsed = time.time() - start_time
        return SymbolResult(
            symbol=symbol,
            success=False,
            error_message=str(e),
            elapsed_time=elapsed,
        )


def print_symbol_result(result: SymbolResult, index: int, total: int) -> None:
    """Print the result for a single symbol."""
    prefix = f"[{index}/{total}]"
    
    if result.success:
        print(f"{prefix} Processing {result.symbol}...")
        print_colored(
            f"   [OK] Trained Random Forest (R^2: {result.r2_score:.3f}, MAE: {result.mae:.4f})",
            "green"
        )
        print_colored(
            f"   [OK] Selected {result.n_features} features (coverage: {result.coverage:.1%})",
            "green"
        )
        
        if result.validation_passed:
            print_colored("   [OK] Validation passed", "green")
        else:
            print_colored("   [WARN] Validation warnings (see report)", "yellow")
    else:
        print(f"{prefix} Processing {result.symbol}...")
        print_colored(f"   [FAIL] Failed: {result.error_message}", "red")


def print_summary_table(results: List[SymbolResult], output_dir: str) -> None:
    """Print a summary table of all results with ASCII-safe formatting."""
    print("\n" + "=" * 50)
    print("                    SUMMARY")
    print("=" * 50 + "\n")
    
    # Prepare table data - ASCII-safe headers
    headers = ["Symbol", "R^2 Score", "Top Feature", "Coverage", "Status"]
    rows = []
    
    for r in results:
        if r.success:
            status = "[OK]" if r.validation_passed else "[WARN]"
            rows.append([
                r.symbol,
                f"{r.r2_score:.3f}",
                r.top_feature[:25] + "..." if len(r.top_feature) > 25 else r.top_feature,
                f"{r.coverage:.1%}",
                status,
            ])
        else:
            rows.append([
                r.symbol,
                "N/A",
                "N/A",
                "N/A",
                "[FAIL]",
            ])
    
    if HAS_TABULATE:
        table_str = tabulate(rows, headers=headers, tablefmt="simple")
        print(table_str)
    else:
        # Fallback: manual table formatting
        col_widths = [8, 10, 28, 10, 10]
        header_line = "  ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        print(header_line)
        print("-" * len(header_line))
        table_lines = [header_line, "-" * len(header_line)]
        for row in rows:
            row_line = "  ".join(str(c).ljust(w) for c, w in zip(row, col_widths))
            print(row_line)
            table_lines.append(row_line)
        table_str = "\n".join(table_lines)
    
    # Save summary to file as backup (in case console has issues)
    import os
    summary_path = os.path.join(output_dir, 'feature_selection_summary.txt')
    try:
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(table_str)
        print(f"\n[INFO] Summary saved to: {summary_path}")
    except Exception as e:
        print(f"\n[WARN] Could not save summary file: {e}")
    
    # Print statistics
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    print(f"\nProcessed: {len(results)} symbols")
    print(f"Successful: {len(successful)}")
    if failed:
        print_colored(f"Failed: {len(failed)}", "red")
        for r in failed:
            print_colored(f"  - {r.symbol}: {r.error_message[:50]}", "red")
    
    print(f"\nAll artifacts saved to: {output_dir}/")


def run_sequential(
    symbols: List[str],
    n_features: int,
    output_dir: str,
    verbose: bool,
) -> List[SymbolResult]:
    """Run feature selection sequentially for all symbols."""
    results = []
    total = len(symbols)
    
    if HAS_TQDM and not verbose:
        iterator = tqdm(enumerate(symbols, 1), total=total, desc="Feature Selection")
    else:
        iterator = enumerate(symbols, 1)
    
    for idx, symbol in iterator:
        if not HAS_TQDM or verbose:
            print(f"\n[{idx}/{total}] Processing {symbol}...")
        
        result = process_symbol(symbol, n_features, output_dir, verbose)
        results.append(result)
        
        if not HAS_TQDM or verbose:
            print_symbol_result(result, idx, total)
    
    return results


def run_parallel(
    symbols: List[str],
    n_features: int,
    output_dir: str,
    max_workers: int = 4,
    verbose: bool = False,
) -> List[SymbolResult]:
    """Run feature selection in parallel for all symbols."""
    results = []
    total = len(symbols)
    
    print(f"Running in parallel with {max_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_symbol = {
            executor.submit(process_symbol, symbol, n_features, output_dir, False): symbol
            for symbol in symbols
        }
        
        # Collect results as they complete
        if HAS_TQDM:
            futures_iter = tqdm(
                as_completed(future_to_symbol),
                total=total,
                desc="Feature Selection"
            )
        else:
            futures_iter = as_completed(future_to_symbol)
        
        for future in futures_iter:
            symbol = future_to_symbol[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append(SymbolResult(
                    symbol=symbol,
                    success=False,
                    error_message=str(e),
                ))
    
    # Sort results to match input order
    symbol_order = {s: i for i, s in enumerate(symbols)}
    results.sort(key=lambda r: symbol_order.get(r.symbol, 999))
    
    # Print results after parallel processing completes
    print("\n")
    for idx, result in enumerate(results, 1):
        print_symbol_result(result, idx, total)
    
    return results


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(
        description='Run feature selection across one or more stock symbols',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_feature_selection.py AAPL
  python scripts/run_feature_selection.py AAPL TSLA MSFT --n-features 40
  python scripts/run_feature_selection.py AAPL TSLA --parallel --workers 4
  python scripts/run_feature_selection.py AAPL --output-dir custom_models
        """
    )
    
    parser.add_argument(
        'symbols',
        nargs='+',
        type=str,
        help='Stock symbol(s) to process (e.g., AAPL TSLA MSFT)'
    )
    
    parser.add_argument(
        '--n-features', '-n',
        type=int,
        default=40,
        help='Number of top features to select (default: 40)'
    )
    
    parser.add_argument(
        '--output-dir', '-o',
        type=str,
        default='saved_models',
        help='Output directory for artifacts (default: saved_models)'
    )
    
    parser.add_argument(
        '--parallel', '-p',
        action='store_true',
        help='Enable parallel processing of multiple symbols'
    )
    
    parser.add_argument(
        '--workers', '-w',
        type=int,
        default=4,
        help='Number of parallel workers (default: 4, only used with --parallel)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output with detailed logging'
    )
    
    args = parser.parse_args()
    
    # Normalize symbols to uppercase
    symbols = [s.upper() for s in args.symbols]
    
    # Print header
    print("\n" + "=" * 50)
    print("         FEATURE SELECTION PIPELINE")
    print("=" * 50)
    print(f"\nSymbols: {', '.join(symbols)}")
    print(f"Features to select: {args.n_features}")
    print(f"Output directory: {args.output_dir}")
    if args.parallel:
        print(f"Mode: Parallel ({args.workers} workers)")
    else:
        print("Mode: Sequential")
    print("")
    
    # Check for required packages
    if not HAS_TQDM:
        print("Note: Install 'tqdm' for progress bars: pip install tqdm")
    if not HAS_TABULATE:
        print("Note: Install 'tabulate' for better summary tables: pip install tabulate")
    
    start_time = time.time()
    
    # Run processing
    if args.parallel and len(symbols) > 1:
        results = run_parallel(
            symbols=symbols,
            n_features=args.n_features,
            output_dir=args.output_dir,
            max_workers=args.workers,
            verbose=args.verbose,
        )
    else:
        results = run_sequential(
            symbols=symbols,
            n_features=args.n_features,
            output_dir=args.output_dir,
            verbose=args.verbose,
        )
    
    total_time = time.time() - start_time
    
    # Print summary
    print_summary_table(results, args.output_dir)
    print(f"\nTotal time: {total_time:.1f}s")
    
    # Exit with error code if any failures
    failed_count = sum(1 for r in results if not r.success)
    if failed_count > 0:
        sys.exit(1)


if __name__ == '__main__':
    main()
