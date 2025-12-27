"""
Comprehensive Backtesting Dashboard

Creates a 2x3 subplot dashboard combining all key metrics:
- Equity curves with drawdown and trade markers
- Rolling performance metrics (Sharpe, win rate, alpha)
- Position sizing analysis
- Confidence distribution
- Regime analysis
- Kelly criterion statistics

Integrates with the existing backtesting pipeline in inference_and_backtest.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, Optional, List, Tuple
from datetime import datetime


def create_backtest_dashboard(
    dates: np.ndarray,
    prices: np.ndarray,
    returns: np.ndarray,
    strategy_positions: np.ndarray,
    equity_curve: np.ndarray,
    buy_hold_equity: np.ndarray,
    final_signals: np.ndarray,
    buy_probs: Optional[np.ndarray] = None,
    sell_probs: Optional[np.ndarray] = None,
    regimes: Optional[np.ndarray] = None,
    kelly_fractions: Optional[np.ndarray] = None,
    max_position_limits: Optional[np.ndarray] = None,
    symbol: str = 'UNKNOWN',
    timestamp: Optional[str] = None,
    out_dir: Optional[str] = None
) -> Dict:
    """
    Create comprehensive 2x3 dashboard with all backtest metrics.
    
    Args:
        dates: Trading dates
        prices: Closing prices
        returns: Daily returns
        strategy_positions: Position fractions taken by strategy
        equity_curve: Strategy equity curve
        buy_hold_equity: Buy-and-hold equity curve
        final_signals: Discrete signals (1=BUY, -1=SELL, 0=HOLD)
        buy_probs: BUY probabilities from classifier
        sell_probs: SELL probabilities from classifier
        regimes: Market regimes (1=bullish, -1=bearish, 0=sideways)
        kelly_fractions: Kelly criterion fractions
        max_position_limits: Dynamic position limits (varies with drawdown)
        symbol: Stock symbol
        timestamp: Timestamp for filename
        out_dir: Output directory
    
    Returns:
        Dict with plot_path and summary metrics
    """
    
    # Input validation
    n = len(dates)
    assert len(prices) == n
    assert len(returns) == n
    assert len(strategy_positions) == n
    assert len(equity_curve) == n
    assert len(buy_hold_equity) == n
    assert len(final_signals) == n
    
    # Convert dates to datetime
    if not isinstance(dates[0], (pd.Timestamp, datetime, np.datetime64)):
        try:
            dates = pd.to_datetime(dates)
        except:
            dates = np.arange(n)
    
    # Handle optional parameters
    if buy_probs is None:
        buy_probs = np.full(n, 0.5)
    if sell_probs is None:
        sell_probs = np.full(n, 0.5)
    if regimes is None:
        regimes = np.zeros(n, dtype=int)
    if kelly_fractions is None:
        kelly_fractions = np.full(n, 0.1)
    if max_position_limits is None:
        max_position_limits = np.full(n, 1.0)
    
    # Ensure arrays
    buy_probs = np.asarray(buy_probs)
    sell_probs = np.asarray(sell_probs)
    regimes = np.asarray(regimes, dtype=int)
    kelly_fractions = np.asarray(kelly_fractions)
    max_position_limits = np.asarray(max_position_limits)
    
    # Generate timestamp if not provided
    if timestamp is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create figure with GridSpec for better layout control
    fig = plt.figure(figsize=(20, 12), dpi=100)
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3, 
                  left=0.08, right=0.95, top=0.94, bottom=0.06)
    
    # ============================================================================
    # SUBPLOT 1 (Top-Left): Equity Curves with Drawdown and Trade Markers
    # ============================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Plot equity curves
    ax1.plot(dates, equity_curve, label='Strategy', color='#2E86AB', linewidth=2)
    ax1.plot(dates, buy_hold_equity, label='Buy & Hold', color='#A23B72', linewidth=1.5, linestyle='--')
    
    # Trade markers
    buy_mask = final_signals == 1
    sell_mask = final_signals == -1
    if np.any(buy_mask):
        ax1.scatter(dates[buy_mask], equity_curve[buy_mask], marker='^', s=80, 
                   c='#00FF00', edgecolors='darkgreen', linewidths=1, zorder=5, label='BUY', alpha=0.7)
    if np.any(sell_mask):
        ax1.scatter(dates[sell_mask], equity_curve[sell_mask], marker='v', s=80,
                   c='#FF0000', edgecolors='darkred', linewidths=1, zorder=5, label='SELL', alpha=0.7)
    
    ax1.set_ylabel('Equity (Multiple of Initial)', fontsize=11, fontweight='bold')
    ax1.set_title('Equity Curves & Trade Markers', fontsize=13, fontweight='bold', pad=10)
    ax1.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(labelsize=9)
    
    # Secondary y-axis for drawdown
    ax1_dd = ax1.twinx()
    running_max = np.maximum.accumulate(equity_curve)
    drawdown_pct = ((equity_curve - running_max) / running_max) * 100
    ax1_dd.fill_between(dates, 0, drawdown_pct, color='orange', alpha=0.3, label='Drawdown %')
    ax1_dd.set_ylabel('Drawdown (%)', fontsize=11, color='#D55E00', fontweight='bold')
    ax1_dd.tick_params(axis='y', labelcolor='#D55E00', labelsize=9)
    ax1_dd.set_ylim([min(drawdown_pct.min() * 1.1, -1), 1])
    ax1_dd.spines['right'].set_color('#D55E00')
    
    # ============================================================================
    # SUBPLOT 2 (Top-Right): Rolling Metrics
    # ============================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    window = 20
    strategy_returns = strategy_positions * returns
    
    # Rolling Sharpe ratio
    rolling_sharpe = pd.Series(strategy_returns).rolling(window).apply(
        lambda x: (x.mean() / x.std() * np.sqrt(252)) if x.std() > 0 else 0, raw=True
    )
    
    # Rolling win rate
    rolling_winrate = pd.Series((strategy_returns > 0).astype(float)).rolling(window).mean() * 100
    
    # Plot on dual axis
    ax2.plot(dates, rolling_sharpe, label='Sharpe', color='#2E86AB', linewidth=2.5)
    ax2.set_ylabel('Sharpe Ratio', fontsize=11, color='#2E86AB', fontweight='bold')
    ax2.tick_params(axis='y', labelcolor='#2E86AB', labelsize=9)
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax2.spines['left'].set_color('#2E86AB')
    
    ax2_wr = ax2.twinx()
    ax2_wr.plot(dates, rolling_winrate, label='Win Rate', color='#F18F01', linewidth=2.5)
    ax2_wr.set_ylabel('Win Rate (%)', fontsize=11, color='#F18F01', fontweight='bold')
    ax2_wr.tick_params(axis='y', labelcolor='#F18F01', labelsize=9)
    ax2_wr.set_ylim([0, 100])
    ax2_wr.spines['right'].set_color('#F18F01')
    
    ax2.set_title('Rolling Metrics (20-Day)', fontsize=13, fontweight='bold', pad=10)
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(labelsize=9)
    
    # Combined legend
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_wr.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9, framealpha=0.9)
    
    # ============================================================================
    # SUBPLOT 3 (Middle-Left): Position Sizing Over Time
    # ============================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Actual position (filled area)
    ax3.fill_between(dates, 0, strategy_positions * 100, alpha=0.4, color='#2E86AB', 
                     label='Actual Position')
    
    # Target position (line) - use kelly_fractions as target
    target_positions = np.clip(kelly_fractions, 0, 1) * np.sign(strategy_positions)
    ax3.plot(dates, target_positions * 100, color='#A23B72', linewidth=2, 
            label='Kelly Target', alpha=0.8)
    ax3.axhline(y=0, color='black', linewidth=1.5, alpha=0.6)
    ax3.set_ylabel('Position Size (%)', fontsize=11, fontweight='bold')
    ax3.set_title('Position Sizing Over Time', fontsize=13, fontweight='bold', pad=10)
    ax3.legend(loc='upper left', fontsize=9, framealpha=0.9)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([-110, 110])
    ax3.tick_params(labelsize=9)
    
    # ============================================================================
    # SUBPLOT 4 (Middle-Right): Confidence Distribution
    # ============================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Extract confidence for BUY and SELL signals
    buy_confidences = buy_probs[final_signals == 1]
    sell_confidences = sell_probs[final_signals == -1]
    
    # Histograms
    bins = np.linspace(0, 1, 21)
    if len(buy_confidences) > 0:
        ax4.hist(buy_confidences, bins=bins, alpha=0.6, color='#00AA00', 
                label=f'BUY Signals (n={len(buy_confidences)})', edgecolor='black')
    if len(sell_confidences) > 0:
        ax4.hist(sell_confidences, bins=bins, alpha=0.6, color='#AA0000',
                label=f'SELL Signals (n={len(sell_confidences)})', edgecolor='black')
    
    ax4.set_xlabel('Confidence', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax4.set_title('Confidence Distribution', fontsize=13, fontweight='bold', pad=10)
    ax4.legend(loc='upper right', fontsize=9, framealpha=0.9)
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_xlim([0, 1])
    ax4.tick_params(labelsize=9)
    
    # ============================================================================
    # SUBPLOT 5 (Bottom-Left): Regime Analysis
    # ============================================================================
    ax5 = fig.add_subplot(gs[2, 0])
    
    # Analyze by regime
    regime_labels = {1: 'Bullish', -1: 'Bearish', 0: 'Sideways'}
    regime_data = {
        'trades': [],
        'win_rate': [],
        'avg_return': []
    }
    
    regime_order = [1, 0, -1]
    regime_names = []
    
    for regime_val in regime_order:
        regime_mask = (regimes == regime_val) & (final_signals != 0)
        regime_names.append(regime_labels.get(regime_val, 'Unknown'))
        
        # Count trades
        n_trades = int(regime_mask.sum())
        regime_data['trades'].append(n_trades)
        
        # Win rate
        if n_trades > 0:
            regime_returns = strategy_returns[regime_mask]
            wins = (regime_returns > 0).sum()
            win_rate = (wins / n_trades) * 100
            avg_ret = regime_returns.mean() * 100
        else:
            win_rate = 0
            avg_ret = 0
        
        regime_data['win_rate'].append(win_rate)
        regime_data['avg_return'].append(avg_ret)
    
    # Create grouped bar chart
    x = np.arange(len(regime_names))
    width = 0.25
    
    ax5.bar(x - width, regime_data['trades'], width, label='# Trades', 
            color='#2E86AB', alpha=0.8)
    
    ax5_pct = ax5.twinx()
    ax5_pct.bar(x, regime_data['win_rate'], width, label='Win Rate %',
                color='#F18F01', alpha=0.8)
    ax5_pct.bar(x + width, regime_data['avg_return'], width, label='Avg Return %',
                color='#6A994E', alpha=0.8)
    
    ax5.set_xlabel('Market Regime', fontsize=11, fontweight='bold')
    ax5.set_ylabel('# Trades', fontsize=11, fontweight='bold', color='#2E86AB')
    ax5.tick_params(axis='y', labelcolor='#2E86AB', labelsize=9)
    ax5.spines['left'].set_color('#2E86AB')
    ax5_pct.set_ylabel('Percentage (%)', fontsize=11, fontweight='bold')
    ax5_pct.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax5_pct.tick_params(labelsize=9)
    
    ax5.set_title('Regime Analysis', fontsize=13, fontweight='bold', pad=10)
    ax5.set_xticks(x)
    ax5.set_xticklabels(regime_names, fontsize=10)
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Legend
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_pct.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9, framealpha=0.9)
    
    # ============================================================================
    # SUBPLOT 6 (Bottom-Right): Kelly Criterion Statistics
    # ============================================================================
    ax6 = fig.add_subplot(gs[2, 1])
    
    # Box plot of Kelly fractions by regime
    kelly_by_regime = []
    regime_labels_box = []
    
    for regime_val in regime_order:
        regime_mask = regimes == regime_val
        kelly_subset = kelly_fractions[regime_mask]
        kelly_subset = kelly_subset[~np.isnan(kelly_subset)]
        if len(kelly_subset) > 0:
            kelly_by_regime.append(kelly_subset)
            regime_labels_box.append(regime_labels.get(regime_val, 'Unknown'))
    
    if kelly_by_regime:
        bp = ax6.boxplot(kelly_by_regime, labels=regime_labels_box, patch_artist=True,
                        showmeans=True, meanline=True)
        
        # Color boxes by regime
        colors = ['#90EE90', '#D3D3D3', '#FFB6C6']  # Light green, gray, light red
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        
        # Style
        for element in ['whiskers', 'fliers', 'means', 'medians', 'caps']:
            plt.setp(bp[element], color='black', linewidth=1.5)
    
    ax6.set_ylabel('Kelly Fraction', fontsize=11, fontweight='bold')
    ax6.set_xlabel('Market Regime', fontsize=11, fontweight='bold')
    ax6.set_title('Kelly Criterion Stats', fontsize=13, fontweight='bold', pad=10)
    ax6.grid(True, alpha=0.3, axis='y')
    ax6.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax6.tick_params(labelsize=9)
    
    # ============================================================================
    plt.suptitle(f'{symbol} - Comprehensive Backtest Dashboard', 
                fontsize=17, fontweight='bold', y=0.98)
    
    # Determine output path
    if out_dir is None:
        out_dir = Path('backtest_results')
    else:
        out_dir = Path(out_dir)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Simple filename (timestamp and symbol in folder name)
    plot_filename = 'dashboard.png'
    plot_path = out_dir / plot_filename
    
    fig.savefig(plot_path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f'✓ Saved backtest dashboard: {plot_path}')
    
    plt.close(fig)
    
    # ============================================================================
    # COMPUTE SUMMARY METRICS
    # ============================================================================
    
    total_return_strategy = (equity_curve[-1] / equity_curve[0] - 1) * 100
    total_return_bh = (buy_hold_equity[-1] / buy_hold_equity[0] - 1) * 100
    max_drawdown = drawdown_pct.min()
    
    n_trades = int((final_signals != 0).sum())
    n_buy = int((final_signals == 1).sum())
    n_sell = int((final_signals == -1).sum())
    
    strategy_rets = strategy_positions * returns
    sharpe = (strategy_rets.mean() / strategy_rets.std() * np.sqrt(252)) if strategy_rets.std() > 0 else 0
    win_rate = ((strategy_rets > 0).sum() / n_trades * 100) if n_trades > 0 else 0
    
    avg_kelly = float(np.nanmean(kelly_fractions))
    avg_position = float(np.mean(np.abs(strategy_positions)))
    
    summary = {
        'total_return_strategy_pct': float(total_return_strategy),
        'total_return_buyhold_pct': float(total_return_bh),
        'max_drawdown_pct': float(max_drawdown),
        'sharpe_ratio': float(sharpe),
        'win_rate_pct': float(win_rate),
        'total_trades': n_trades,
        'buy_trades': n_buy,
        'sell_trades': n_sell,
        'avg_kelly_fraction': avg_kelly,
        'avg_position_size': avg_position,
        'regime_performance': {
            regime_labels[rv]: {
                'trades': int(regime_data['trades'][i]),
                'win_rate_pct': float(regime_data['win_rate'][i]),
                'avg_return_pct': float(regime_data['avg_return'][i])
            } for i, rv in enumerate(regime_order)
        }
    }
    
    return {
        'plot_path': str(plot_path),
        'summary': summary
    }


def integrate_with_backtest(
    dates: np.ndarray,
    prices: np.ndarray,
    returns: np.ndarray,
    strategy_positions: np.ndarray,
    equity_curve: np.ndarray,
    buy_hold_equity: np.ndarray,
    final_signals: np.ndarray,
    buy_probs: Optional[np.ndarray] = None,
    sell_probs: Optional[np.ndarray] = None,
    regimes: Optional[np.ndarray] = None,
    kelly_fractions: Optional[np.ndarray] = None,
    max_position_limits: Optional[np.ndarray] = None,
    symbol: str = 'UNKNOWN',
    timestamp: Optional[str] = None,
    out_dir: Optional[str] = None
) -> Dict:
    """
    Wrapper to integrate dashboard creation into backtesting pipeline.
    
    All parameters passed directly to create_backtest_dashboard().
    """
    return create_backtest_dashboard(
        dates=dates,
        prices=prices,
        returns=returns,
        strategy_positions=strategy_positions,
        equity_curve=equity_curve,
        buy_hold_equity=buy_hold_equity,
        final_signals=final_signals,
        buy_probs=buy_probs,
        sell_probs=sell_probs,
        regimes=regimes,
        kelly_fractions=kelly_fractions,
        max_position_limits=max_position_limits,
        symbol=symbol,
        timestamp=timestamp,
        out_dir=out_dir
    )
