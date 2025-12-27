#!/usr/bin/env python
"""
Position Sizing Diagnostic Script

Investigates position sizing calculations to identify any scaling issues.
Compares expected position sizes (based on SYSTEM_ARCHITECTURE.md formula)
to actual position_shares values from backtest trade logs.

Usage:
    python scripts/diagnose_position_sizing.py --backtest-dir backtest_results/AAPL_latest/
    python scripts/diagnose_position_sizing.py  # Uses most recent backtest
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Position sizing constants from inference_and_backtest.py
HYBRID_BASE_FRACTION = 0.15
HYBRID_SCALE_FACTOR = 4.0
HYBRID_MAX_POSITION = 0.40
HYBRID_MIN_SHARES_THRESHOLD = 1.0
HYBRID_REFERENCE_EQUITY = 10_000.0
KELLY_MIN_FRACTION = 0.05
KELLY_MAX_FRACTION = 0.25


def find_latest_backtest_dir(base_dir: Path) -> Path:
    """Find the most recent backtest directory."""
    if not base_dir.exists():
        raise FileNotFoundError(f"Backtest results directory not found: {base_dir}")
    
    # Get all subdirectories
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No backtest directories found in {base_dir}")
    
    # Sort by modification time (most recent first)
    subdirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return subdirs[0]


def load_trade_log(backtest_dir: Path) -> pd.DataFrame:
    """Load the trade log CSV from a backtest directory."""
    # Try different possible filenames
    candidates = [
        backtest_dir / "confidence_trade_log.csv",
        backtest_dir / "backtest_trades.csv",
        backtest_dir / "trade_log.csv",
    ]
    
    for candidate in candidates:
        if candidate.exists():
            print(f"✓ Loading trade log from: {candidate}")
            return pd.read_csv(candidate)
    
    raise FileNotFoundError(f"No trade log found in {backtest_dir}. Tried: {[c.name for c in candidates]}")


def load_equity_data(backtest_dir: Path) -> pd.DataFrame:
    """Load equity comparison CSV for equity values."""
    equity_file = backtest_dir / "equity_comparison.csv"
    if equity_file.exists():
        print(f"✓ Loading equity data from: {equity_file}")
        return pd.read_csv(equity_file)
    return None


def compute_expected_shares(
    confidence: float,
    equity: float,
    price: float,
    method: str = "hybrid"
) -> dict:
    """
    Compute expected shares using the position sizing formula from SYSTEM_ARCHITECTURE.md.
    
    Methods:
    - 'hybrid': Uses HYBRID_BASE_FRACTION + confidence * HYBRID_SCALE_FACTOR
    - 'kelly': Uses Kelly criterion (bounded by KELLY_MIN/MAX_FRACTION)
    - 'simple': Uses confidence directly as position fraction
    
    Returns dict with intermediate calculations for debugging.
    """
    result = {
        'confidence': confidence,
        'equity': equity,
        'price': price,
        'method': method,
    }
    
    if method == "hybrid":
        # Formula from SYSTEM_ARCHITECTURE.md:
        # target_fraction = HYBRID_BASE_FRACTION + confidence_strength * HYBRID_SCALE_FACTOR
        confidence_strength = max(0, confidence - 0.5)  # excess confidence above 0.5
        target_fraction = HYBRID_BASE_FRACTION + confidence_strength * HYBRID_SCALE_FACTOR
        target_fraction = min(target_fraction, HYBRID_MAX_POSITION)  # cap at max
        
        result['confidence_strength'] = confidence_strength
        result['base_fraction'] = HYBRID_BASE_FRACTION
        result['scale_contribution'] = confidence_strength * HYBRID_SCALE_FACTOR
        
    elif method == "kelly":
        # Kelly criterion position sizing
        # Simplified: fraction = win_prob - (1-win_prob)/win_loss_ratio
        # Assuming win_loss_ratio = 1.5 (default)
        win_loss_ratio = 1.5
        kelly_fraction = confidence - (1 - confidence) / win_loss_ratio
        target_fraction = np.clip(kelly_fraction, KELLY_MIN_FRACTION, KELLY_MAX_FRACTION)
        
        result['kelly_raw'] = kelly_fraction
        
    elif method == "simple":
        # Direct confidence as position fraction (what compute_hybrid_positions uses now)
        target_fraction = confidence
        
    else:
        target_fraction = confidence
    
    result['target_fraction'] = target_fraction
    result['target_value'] = target_fraction * equity
    result['expected_shares'] = (target_fraction * equity) / price if price > 0 else 0
    
    return result


def analyze_buy_signals(trade_log: pd.DataFrame, equity_data: pd.DataFrame = None):
    """Analyze BUY signals and compare expected vs actual position sizes."""
    
    print("\n" + "=" * 70)
    print("                    POSITION SIZING DIAGNOSTIC")
    print("=" * 70)
    
    # Filter to BUY signals only
    buy_mask = trade_log['action'] == 'BUY'
    buy_rows = trade_log[buy_mask].copy()
    
    print(f"\nTotal rows in trade log: {len(trade_log)}")
    print(f"BUY signals: {buy_mask.sum()}")
    print(f"SELL signals: {(trade_log['action'] == 'SELL').sum()}")
    print(f"HOLD signals: {(trade_log['action'] == 'HOLD').sum()}")
    
    if len(buy_rows) == 0:
        print("\n⚠️ No BUY signals found in trade log!")
        print("Cannot diagnose position sizing without BUY signals.")
        return
    
    # Default equity if not available
    default_equity = HYBRID_REFERENCE_EQUITY
    
    # Merge equity data if available
    if equity_data is not None and 'date' in equity_data.columns and 'date' in buy_rows.columns:
        # Try to merge by date
        try:
            # Normalize date formats - equity file has ISO timestamps, trade log has YYYY-MM-DD
            equity_data = equity_data.copy()
            equity_data['date_norm'] = pd.to_datetime(equity_data['date']).dt.strftime('%Y-%m-%d')
            buy_rows = buy_rows.copy()
            buy_rows['date_norm'] = pd.to_datetime(buy_rows['date']).dt.strftime('%Y-%m-%d')
            
            # Create lookup dict from equity data
            equity_lookup = dict(zip(
                equity_data['date_norm'], 
                equity_data['strategy_equity_usd']
            ))
            
            # Apply lookup
            buy_rows['equity'] = buy_rows['date_norm'].map(equity_lookup).fillna(default_equity)
            matched = buy_rows['date_norm'].isin(equity_lookup.keys()).sum()
            print(f"✓ Merged equity data for {matched}/{len(buy_rows)} rows")
        except Exception as e:
            print(f"Warning: Could not merge equity data: {e}")
            buy_rows['equity'] = default_equity
    else:
        buy_rows['equity'] = default_equity
        print(f"ℹ️ Using default equity: ${default_equity:.2f}")
    
    print("\n" + "-" * 70)
    print("DETAILED BUY SIGNAL ANALYSIS")
    print("-" * 70)
    
    ratios = []
    
    for idx, row in buy_rows.iterrows():
        date = row.get('date', 'N/A')
        price = row.get('price', 0)
        unified_confidence = row.get('unified_confidence', row.get('confidence', 0))
        position_shares = row.get('position_shares', 0)
        equity = row.get('equity', default_equity)
        
        # Compute expected shares using different methods
        expected_hybrid = compute_expected_shares(unified_confidence, equity, price, 'hybrid')
        expected_kelly = compute_expected_shares(unified_confidence, equity, price, 'kelly')
        expected_simple = compute_expected_shares(unified_confidence, equity, price, 'simple')
        
        print(f"\n📅 Date: {date}")
        print(f"   Price: ${price:.2f}")
        print(f"   Equity: ${equity:.2f}")
        print(f"   Unified Confidence: {unified_confidence:.6f}")
        print(f"   Actual position_shares: {position_shares:.6f}")
        
        print(f"\n   Expected (Hybrid formula):")
        print(f"     confidence_strength = max(0, {unified_confidence:.4f} - 0.5) = {expected_hybrid['confidence_strength']:.4f}")
        print(f"     target_fraction = {HYBRID_BASE_FRACTION} + {expected_hybrid['confidence_strength']:.4f} * {HYBRID_SCALE_FACTOR} = {expected_hybrid['target_fraction']:.4f}")
        print(f"     expected_shares = {expected_hybrid['target_fraction']:.4f} * ${equity:.2f} / ${price:.2f} = {expected_hybrid['expected_shares']:.4f}")
        
        print(f"\n   Expected (Kelly formula):")
        print(f"     target_fraction = {expected_kelly['target_fraction']:.4f}")
        print(f"     expected_shares = {expected_kelly['expected_shares']:.4f}")
        
        print(f"\n   Expected (Simple confidence):")
        print(f"     target_fraction = {expected_simple['target_fraction']:.4f}")
        print(f"     expected_shares = {expected_simple['expected_shares']:.4f}")
        
        # Calculate ratios
        if expected_hybrid['expected_shares'] > 0:
            ratio_hybrid = position_shares / expected_hybrid['expected_shares']
            ratios.append({
                'method': 'hybrid',
                'ratio': ratio_hybrid,
                'date': date,
            })
            print(f"\n   📊 Ratio (actual/expected_hybrid): {ratio_hybrid:.6f}")
            if ratio_hybrid < 0.01:
                print(f"      ⚠️ LIKELY 100x SCALING ISSUE (ratio ≈ {1/ratio_hybrid:.0f}x smaller)")
            elif ratio_hybrid < 0.1:
                print(f"      ⚠️ LIKELY 10x SCALING ISSUE")
        
        if expected_simple['expected_shares'] > 0:
            ratio_simple = position_shares / expected_simple['expected_shares']
            ratios.append({
                'method': 'simple',
                'ratio': ratio_simple,
                'date': date,
            })
            print(f"   📊 Ratio (actual/expected_simple): {ratio_simple:.6f}")
            if abs(ratio_simple - 1.0) < 0.01:
                print(f"      ✓ Matches simple confidence formula!")
    
    # Summary statistics
    print("\n" + "=" * 70)
    print("                    RATIO SUMMARY")
    print("=" * 70)
    
    if ratios:
        hybrid_ratios = [r['ratio'] for r in ratios if r['method'] == 'hybrid']
        simple_ratios = [r['ratio'] for r in ratios if r['method'] == 'simple']
        
        if hybrid_ratios:
            print(f"\nHybrid Formula Comparison:")
            print(f"  Mean ratio (actual/expected): {np.mean(hybrid_ratios):.6f}")
            print(f"  Std ratio: {np.std(hybrid_ratios):.6f}")
            print(f"  Min ratio: {np.min(hybrid_ratios):.6f}")
            print(f"  Max ratio: {np.max(hybrid_ratios):.6f}")
            
            # Identify scaling factor
            mean_ratio = np.mean(hybrid_ratios)
            if 0.0009 < mean_ratio < 0.0011:
                print(f"\n  🔴 DETECTED: ~1000x scaling issue (mean ratio ≈ 0.001)")
                print(f"     Actual positions are 1000x smaller than expected!")
            elif 0.009 < mean_ratio < 0.011:
                print(f"\n  🟡 DETECTED: ~100x scaling issue (mean ratio ≈ 0.01)")
            elif 0.09 < mean_ratio < 0.11:
                print(f"\n  🟡 DETECTED: ~10x scaling issue (mean ratio ≈ 0.1)")
            elif 0.9 < mean_ratio < 1.1:
                print(f"\n  ✅ Position sizing matches expected (ratio ≈ 1.0)")
        
        if simple_ratios:
            print(f"\nSimple Confidence Comparison:")
            print(f"  Mean ratio (actual/expected): {np.mean(simple_ratios):.6f}")
            print(f"  Std ratio: {np.std(simple_ratios):.6f}")
            
            mean_ratio = np.mean(simple_ratios)
            if 0.9 < mean_ratio < 1.1:
                print(f"\n  ✅ Matches simple formula: position_shares = confidence")
                print(f"     This is the CURRENT behavior in compute_hybrid_positions")
    
    # Check if position_shares are fractional exposures vs dollar amounts
    print("\n" + "-" * 70)
    print("POSITION INTERPRETATION CHECK")
    print("-" * 70)
    
    if len(buy_rows) > 0:
        pos_shares = buy_rows['position_shares'].values
        print(f"\nposition_shares statistics:")
        print(f"  Mean: {np.mean(pos_shares):.6f}")
        print(f"  Std: {np.std(pos_shares):.6f}")
        print(f"  Min: {np.min(pos_shares):.6f}")
        print(f"  Max: {np.max(pos_shares):.6f}")
        
        # Interpretation
        if np.max(pos_shares) <= 1.0 and np.min(pos_shares) >= -1.0:
            print(f"\n  📊 Interpretation: position_shares appears to be FRACTIONAL EXPOSURE")
            print(f"     (values in [-1, 1] represent fraction of portfolio)")
            print(f"     To get dollar value: position_value = position_shares * equity")
            print(f"     To get shares: actual_shares = position_value / price")
        elif np.mean(pos_shares) > 1:
            print(f"\n  📊 Interpretation: position_shares appears to be SHARE COUNT")
        else:
            print(f"\n  ❓ Interpretation unclear - review the compute_hybrid_positions function")


def main():
    parser = argparse.ArgumentParser(description='Diagnose position sizing calculations')
    parser.add_argument(
        '--backtest-dir',
        type=str,
        default=None,
        help='Path to backtest results directory (default: most recent in backtest_results/)'
    )
    args = parser.parse_args()
    
    # Resolve backtest directory
    base_dir = Path(__file__).resolve().parent.parent / 'backtest_results'
    
    if args.backtest_dir:
        backtest_dir = Path(args.backtest_dir)
        if not backtest_dir.is_absolute():
            backtest_dir = base_dir / args.backtest_dir
    else:
        backtest_dir = find_latest_backtest_dir(base_dir)
    
    print(f"🔍 Analyzing backtest: {backtest_dir}")
    
    # Load data
    trade_log = load_trade_log(backtest_dir)
    equity_data = load_equity_data(backtest_dir)
    
    # Run analysis
    analyze_buy_signals(trade_log, equity_data)
    
    print("\n" + "=" * 70)
    print("DIAGNOSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
