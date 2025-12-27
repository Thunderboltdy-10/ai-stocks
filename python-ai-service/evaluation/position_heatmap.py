"""
Position Sizing Heatmap Visualization

Creates a comprehensive 2D heatmap showing position sizing decisions over time
with overlays for market regimes, drawdowns, volatility, and trading signals.

Integrates with the existing backtesting pipeline in inference_and_backtest.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.cm import ScalarMappable
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple


def create_position_heatmap(
    dates: np.ndarray,
    prices: np.ndarray,
    positions: np.ndarray,
    final_signals: np.ndarray,
    confidence: Optional[np.ndarray] = None,
    buy_probs: Optional[np.ndarray] = None,
    sell_probs: Optional[np.ndarray] = None,
    regimes: Optional[np.ndarray] = None,
    atr_percent: Optional[np.ndarray] = None,
    equity_curve: Optional[np.ndarray] = None,
    symbol: str = 'UNKNOWN',
    timestamp: Optional[str] = None,
    out_dir: Optional[str] = None,
    show_plot: bool = False
) -> Dict:
    """
    Create a 2D position sizing heatmap with comprehensive overlays.
    
    Visualization includes:
    - X-axis: Trading dates
    - Y-axis: Position size as % of portfolio
    - Color intensity: Confidence level (from classifiers or unified scorer)
    - Markers: BUY (green triangle up), SELL (red triangle down), HOLD (gray circle)
    - Background bands: Market regime (green=bullish, red=bearish, gray=sideways)
    - Reference lines: 25%, 50%, 75%, 100% position levels
    - Shaded regions: Drawdown periods
    - Opacity modulation: Volatility (ATR) - higher vol = more transparent
    
    Args:
        dates: Array of trading dates
        prices: Array of closing prices
        positions: Array of position fractions (0-1.0, or negative for short)
        final_signals: Array of discrete signals (1=BUY, -1=SELL, 0=HOLD)
        confidence: Array of unified confidence scores [0, 1]
        buy_probs: Array of BUY probabilities (fallback if confidence not provided)
        sell_probs: Array of SELL probabilities (fallback if confidence not provided)
        regimes: Array of market regimes (1=bullish, -1=bearish, 0=sideways)
        atr_percent: Array of ATR as percentage (for volatility bands)
        equity_curve: Array of portfolio equity (for drawdown detection)
        symbol: Stock symbol for title
        timestamp: Timestamp string for filename
        out_dir: Output directory for saving PNG
        show_plot: Whether to display plot interactively
    
    Returns:
        Dict with keys: 'plot_path', 'data_summary', 'metrics'
    """
    # ============================================================================
    # INPUT VALIDATION & PREPROCESSING
    # ============================================================================
    
    n = len(dates)
    assert len(prices) == n, f"prices length {len(prices)} != dates length {n}"
    assert len(positions) == n, f"positions length {len(positions)} != dates length {n}"
    assert len(final_signals) == n, f"signals length {len(final_signals)} != dates length {n}"
    
    # Convert dates to datetime if needed
    if not isinstance(dates[0], (pd.Timestamp, datetime, np.datetime64)):
        try:
            dates = pd.to_datetime(dates)
        except Exception:
            # Fallback: use numeric index as x-axis
            dates = np.arange(n)
            use_date_labels = False
    else:
        use_date_labels = True
    
    # Build unified confidence array
    if confidence is None:
        # Fallback: use buy/sell probs based on signal direction
        confidence = np.zeros(n, dtype=float)
        if buy_probs is not None and sell_probs is not None:
            buy_mask = final_signals == 1
            sell_mask = final_signals == -1
            confidence[buy_mask] = buy_probs[buy_mask]
            confidence[sell_mask] = sell_probs[sell_mask]
        else:
            # No confidence data - use constant mid-level
            confidence = np.full(n, 0.5, dtype=float)
    else:
        confidence = np.asarray(confidence, dtype=float)
        assert len(confidence) == n, f"confidence length {len(confidence)} != dates length {n}"
    
    # Default regimes if not provided
    if regimes is None:
        regimes = np.zeros(n, dtype=int)  # All sideways
    else:
        regimes = np.asarray(regimes, dtype=int)
        assert len(regimes) == n, f"regimes length {len(regimes)} != dates length {n}"
    
    # Default ATR if not provided
    if atr_percent is None:
        atr_percent = np.full(n, 0.02, dtype=float)  # 2% baseline
    else:
        atr_percent = np.asarray(atr_percent, dtype=float)
        assert len(atr_percent) == n, f"atr_percent length {len(atr_percent)} != dates length {n}"
    
    # Compute drawdown periods if equity provided
    drawdown_periods = []
    if equity_curve is not None:
        equity_curve = np.asarray(equity_curve, dtype=float)
        assert len(equity_curve) == n, f"equity_curve length {len(equity_curve)} != dates length {n}"
        drawdown_periods = _detect_drawdown_periods(equity_curve, dates)
    
    # ============================================================================
    # FIGURE SETUP
    # ============================================================================
    
    fig, ax = plt.subplots(figsize=(16, 8), dpi=100)
    
    # Convert positions to percentage (0-100%)
    positions_pct = positions * 100.0
    
    # ============================================================================
    # BACKGROUND: MARKET REGIME BANDS
    # ============================================================================
    
    # Define regime colors with transparency
    regime_colors = {
        1: (0.0, 0.8, 0.0, 0.15),   # Bullish: transparent green
        -1: (0.8, 0.0, 0.0, 0.15),  # Bearish: transparent red
        0: (0.5, 0.5, 0.5, 0.08)    # Sideways: transparent gray
    }
    
    # Plot regime bands as vertical spans
    regime_changes = np.where(np.diff(regimes, prepend=regimes[0]) != 0)[0]
    regime_changes = np.append(regime_changes, n)  # Add final boundary
    
    for i in range(len(regime_changes) - 1):
        start_idx = regime_changes[i]
        end_idx = regime_changes[i + 1]
        regime_val = int(regimes[start_idx])
        
        if use_date_labels:
            x_start = dates[start_idx]
            x_end = dates[min(end_idx, n - 1)]
        else:
            x_start = start_idx
            x_end = end_idx
        
        color = regime_colors.get(regime_val, regime_colors[0])
        ax.axvspan(x_start, x_end, facecolor=color, edgecolor='none', zorder=0)
    
    # ============================================================================
    # DRAWDOWN SHADED REGIONS
    # ============================================================================
    
    for dd_start, dd_end, dd_depth in drawdown_periods:
        if use_date_labels:
            x_start = dates[dd_start]
            x_end = dates[dd_end]
        else:
            x_start = dd_start
            x_end = dd_end
        
        # Shade with intensity proportional to drawdown depth
        alpha = min(0.3, abs(dd_depth) * 2.0)  # Max 30% opacity
        ax.axvspan(x_start, x_end, facecolor='orange', alpha=alpha, edgecolor='none', zorder=1)
    
    # ============================================================================
    # REFERENCE LINES: 25%, 50%, 75%, 100%
    # ============================================================================
    
    reference_levels = [25, 50, 75, 100]
    for level in reference_levels:
        ax.axhline(y=level, color='gray', linestyle='--', linewidth=0.8, alpha=0.4, zorder=2)
        # Label on right y-axis
        ax.text(dates[-1] if use_date_labels else n-1, level, f' {level}%', 
                verticalalignment='center', fontsize=9, color='gray', alpha=0.6)
    
    # Add 0% baseline
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1.0, alpha=0.5, zorder=2)
    
    # ============================================================================
    # MAIN HEATMAP: POSITION SIZE WITH CONFIDENCE COLOR
    # ============================================================================
    
    # Create custom colormap: low confidence (pale) -> high confidence (vivid)
    # For BUY: pale green -> dark green
    # For SELL: pale red -> dark red
    # For HOLD: gray
    
    # Normalize confidence to [0, 1]
    conf_norm = np.clip(confidence, 0.0, 1.0)
    
    # Compute bar colors based on signal type and confidence
    bar_colors = []
    for i in range(n):
        sig = int(final_signals[i])
        conf = float(conf_norm[i])
        
        if sig == 1:  # BUY
            # Green intensity: conf=0 -> pale (0.6,1,0.6), conf=1 -> dark (0,0.6,0)
            green_val = 0.6 + (0.4 * (1 - conf))  # Lighter when less confident
            bar_colors.append((0.0, green_val, 0.0, 0.8))
        elif sig == -1:  # SELL
            # Red intensity: conf=0 -> pale (1,0.6,0.6), conf=1 -> dark (0.8,0,0)
            red_base = 0.8
            red_val = 0.6 + (0.2 * conf)  # Darker when more confident
            bar_colors.append((red_val, 0.0, 0.0, 0.8))
        else:  # HOLD
            # Gray with slight transparency
            bar_colors.append((0.5, 0.5, 0.5, 0.3))
    
    # Modulate alpha by volatility (higher ATR = more transparent)
    # Normalize ATR to [0, 1] range
    atr_norm = np.clip(atr_percent / 0.05, 0.0, 1.0)  # 5% ATR = max transparency
    
    # Adjust bar colors with volatility-based alpha
    bar_colors_vol = []
    for i in range(n):
        r, g, b, a_base = bar_colors[i]
        # Reduce alpha for high volatility (make more transparent)
        a_vol = a_base * (1.0 - 0.4 * atr_norm[i])  # Up to 40% reduction
        bar_colors_vol.append((r, g, b, a_vol))
    
    # Plot position bars
    if use_date_labels:
        x_vals = dates
    else:
        x_vals = np.arange(n)
    
    bar_width = 0.8 if not use_date_labels else pd.Timedelta(days=0.8)
    
    ax.bar(x_vals, positions_pct, width=bar_width, color=bar_colors_vol, 
           edgecolor='none', zorder=3, label='Position Size')
    
    # ============================================================================
    # TRADE MARKERS: BUY/SELL/HOLD - Now matches actual backtest signals
    # ============================================================================
    
    # Use final_signals directly for accurate matching with backtest.png
    # BUY = signal == 1, SELL = signal == -1, HOLD = signal == 0
    buy_indices = np.where(final_signals == 1)[0]
    sell_indices = np.where(final_signals == -1)[0]
    hold_indices = np.where(final_signals == 0)[0]
    
    # Plot markers at position bar tops
    marker_size = 80
    
    if len(buy_indices) > 0:
        x_buy = dates[buy_indices] if use_date_labels else buy_indices
        y_buy = positions_pct[buy_indices]
        ax.scatter(x_buy, y_buy, marker='^', s=marker_size, c='#00ff00', 
                   edgecolors='darkgreen', linewidths=1.5, zorder=5, label='BUY')
    
    if len(sell_indices) > 0:
        x_sell = dates[sell_indices] if use_date_labels else sell_indices
        y_sell = positions_pct[sell_indices]
        ax.scatter(x_sell, y_sell, marker='v', s=marker_size, c='#ff0000', 
                   edgecolors='darkred', linewidths=1.5, zorder=5, label='SELL')
    
    # HOLD markers (optional - can clutter chart)
    # Uncomment to show HOLD signals
    # if len(hold_indices) > 0:
    #     x_hold = dates[hold_indices] if use_date_labels else hold_indices
    #     y_hold = positions_pct[hold_indices]
    #     ax.scatter(x_hold, y_hold, marker='o', s=marker_size*0.5, c='gray', 
    #                edgecolors='darkgray', linewidths=0.8, alpha=0.3, zorder=4, label='HOLD')
    
    # ============================================================================
    # AXES & LABELS
    # ============================================================================
    
    ax.set_xlabel('Date' if use_date_labels else 'Trading Day', fontsize=12, fontweight='bold')
    ax.set_ylabel('Position Size (% of Portfolio)', fontsize=12, fontweight='bold')
    
    # Title with timestamp
    ts = timestamp if timestamp else datetime.now().strftime('%Y%m%d_%H%M%S')
    title = f'{symbol} - Position Sizing Heatmap\nDate: {ts}'
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Format x-axis for dates
    if use_date_labels:
        fig.autofmt_xdate(rotation=45, ha='right')
        # Reduce number of x-ticks if too many
        if n > 100:
            ax.xaxis.set_major_locator(plt.MaxNLocator(20))
    
    # Grid
    ax.grid(True, alpha=0.2, linestyle=':', linewidth=0.5, zorder=1)
    
    # Y-axis limits (extend slightly beyond 100% for visibility)
    y_min = min(positions_pct.min(), -10)  # Allow for short positions
    y_max = max(positions_pct.max(), 105)
    ax.set_ylim(y_min, y_max)
    
    # ============================================================================
    # LEGEND - MULTI-SECTION
    # ============================================================================
    
    # Build legend elements
    legend_elements = []
    
    # Trading signals
    if len(buy_indices) > 0:
        legend_elements.append(plt.Line2D([0], [0], marker='^', color='w', 
                                          markerfacecolor='#00ff00', markeredgecolor='darkgreen',
                                          markersize=10, label='BUY Signal'))
    if len(sell_indices) > 0:
        legend_elements.append(plt.Line2D([0], [0], marker='v', color='w', 
                                          markerfacecolor='#ff0000', markeredgecolor='darkred',
                                          markersize=10, label='SELL Signal'))
    
    # Regime bands
    legend_elements.append(mpatches.Patch(facecolor=regime_colors[1], label='Bullish Regime'))
    legend_elements.append(mpatches.Patch(facecolor=regime_colors[-1], label='Bearish Regime'))
    legend_elements.append(mpatches.Patch(facecolor=regime_colors[0], label='Sideways Regime'))
    
    # Drawdown shading
    if len(drawdown_periods) > 0:
        legend_elements.append(mpatches.Patch(facecolor='orange', alpha=0.3, label='Drawdown Period'))
    
    # Confidence indicator
    legend_elements.append(plt.Line2D([0], [0], color='darkgreen', linewidth=8, 
                                      label='High Confidence (dark)'))
    legend_elements.append(plt.Line2D([0], [0], color=(0.6, 1, 0.6), linewidth=8, 
                                      label='Low Confidence (pale)'))
    
    # Volatility indicator
    legend_elements.append(plt.Line2D([0], [0], color='gray', linewidth=8, alpha=0.8,
                                      label='Low Volatility (opaque)'))
    legend_elements.append(plt.Line2D([0], [0], color='gray', linewidth=8, alpha=0.4,
                                      label='High Volatility (transparent)'))
    
    # Place legend outside plot area
    ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1.0),
              fontsize=9, framealpha=0.95, edgecolor='gray')
    
    # ============================================================================
    # SAVE FIGURE
    # ============================================================================
    
    plt.tight_layout()
    
    # Determine output path
    if out_dir is None:
        out_dir = Path('python-ai-service') / 'backtest_results'
    else:
        out_dir = Path(out_dir)
    
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Simple filename (timestamp and symbol in folder name)
    plot_filename = 'position_heatmap.png'
    plot_path = out_dir / plot_filename
    
    fig.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'✓ Saved position heatmap: {plot_path}')
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    # ============================================================================
    # COMPUTE DATA SUMMARY & METRICS
    # ============================================================================
    
    buy_count = len(buy_indices)
    sell_count = len(sell_indices)
    hold_count = len(hold_indices)
    
    avg_buy_position = float(positions_pct[buy_indices].mean()) if buy_count > 0 else 0.0
    avg_sell_position = float(positions_pct[sell_indices].mean()) if sell_count > 0 else 0.0
    max_position = float(positions_pct.max())
    min_position = float(positions_pct.min())
    
    avg_buy_confidence = float(confidence[buy_indices].mean()) if buy_count > 0 else 0.0
    avg_sell_confidence = float(confidence[sell_indices].mean()) if sell_count > 0 else 0.0
    
    regime_counts = {
        'bullish': int(np.sum(regimes == 1)),
        'bearish': int(np.sum(regimes == -1)),
        'sideways': int(np.sum(regimes == 0))
    }
    
    avg_atr = float(atr_percent.mean())
    max_atr = float(atr_percent.max())
    
    data_summary = {
        'total_days': n,
        'buy_signals': buy_count,
        'sell_signals': sell_count,
        'hold_signals': hold_count,
        'avg_buy_position_pct': avg_buy_position,
        'avg_sell_position_pct': avg_sell_position,
        'max_position_pct': max_position,
        'min_position_pct': min_position,
        'avg_buy_confidence': avg_buy_confidence,
        'avg_sell_confidence': avg_sell_confidence,
        'regime_distribution': regime_counts,
        'avg_atr_pct': avg_atr * 100,
        'max_atr_pct': max_atr * 100,
        'drawdown_periods': len(drawdown_periods)
    }
    
    # Print summary
    print(f'\n=== Position Heatmap Summary ===')
    print(f'Total Days: {n}')
    print(f'Trading Signals: BUY={buy_count}, SELL={sell_count}, HOLD={hold_count}')
    print(f'Average BUY Position: {avg_buy_position:.1f}% (confidence: {avg_buy_confidence:.2f})')
    print(f'Average SELL Position: {avg_sell_position:.1f}% (confidence: {avg_sell_confidence:.2f})')
    print(f'Position Range: {min_position:.1f}% to {max_position:.1f}%')
    print(f'Market Regimes: Bullish={regime_counts["bullish"]}, Bearish={regime_counts["bearish"]}, Sideways={regime_counts["sideways"]}')
    print(f'Average ATR: {avg_atr*100:.2f}% (max: {max_atr*100:.2f}%)')
    print(f'Drawdown Periods: {len(drawdown_periods)}')
    
    return {
        'plot_path': str(plot_path),
        'data_summary': data_summary,
        'metrics': {
            'position_utilization': avg_buy_position / 100.0 if buy_count > 0 else 0.0,
            'signal_diversity': (buy_count + sell_count) / n if n > 0 else 0.0,
            'confidence_quality': (avg_buy_confidence + avg_sell_confidence) / 2.0
        }
    }


def _detect_drawdown_periods(equity_curve: np.ndarray, dates: np.ndarray) -> List[Tuple[int, int, float]]:
    """
    Detect drawdown periods from equity curve.
    
    Returns:
        List of tuples: (start_idx, end_idx, depth_pct)
    """
    n = len(equity_curve)
    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    
    # Threshold for significant drawdown
    threshold = -0.02  # 2% drawdown minimum
    
    periods = []
    in_drawdown = False
    start_idx = 0
    
    for i in range(n):
        if drawdown[i] <= threshold and not in_drawdown:
            # Start of drawdown
            in_drawdown = True
            start_idx = i
        elif drawdown[i] > threshold and in_drawdown:
            # End of drawdown (recovered)
            in_drawdown = False
            depth = float(drawdown[start_idx:i].min())
            periods.append((start_idx, i - 1, depth))
    
    # Handle open drawdown at end
    if in_drawdown:
        depth = float(drawdown[start_idx:].min())
        periods.append((start_idx, n - 1, depth))
    
    return periods


# ============================================================================
# INTEGRATION HELPER FOR INFERENCE_AND_BACKTEST.PY
# ============================================================================

def integrate_with_backtest(
    trade_log_df: pd.DataFrame,
    equity_curve: np.ndarray,
    symbol: str,
    timestamp: str,
    out_dir: Optional[str] = None
) -> Dict:
    """
    Convenience wrapper to create heatmap from trade log DataFrame.
    
    Args:
        trade_log_df: DataFrame with columns: date, price, action, shares, 
                      portfolio_pct, kelly_fraction, regime, atr, vol_ratio, 
                      drawdown_pct, unified_confidence, etc.
        equity_curve: Array of portfolio equity values
        symbol: Stock symbol
        timestamp: Timestamp string for filename
        out_dir: Output directory (defaults to backtest_results)
    
    Returns:
        Dict with plot_path, data_summary, metrics
    """
    # Extract required arrays from trade log
    dates = trade_log_df['date'].values
    prices = trade_log_df['price'].values
    
    # Map actions to signals
    action_to_signal = {'BUY': 1, 'SELL': -1, 'HOLD': 0}
    final_signals = np.array([action_to_signal.get(str(a).upper(), 0) 
                              for a in trade_log_df['action'].values])
    
    # Positions as fraction (portfolio_pct is already 0-1)
    positions = trade_log_df['portfolio_pct'].values
    
    # Optional columns with fallbacks
    confidence = None
    if 'unified_confidence' in trade_log_df.columns:
        confidence = trade_log_df['unified_confidence'].values
    elif 'buy_prob' in trade_log_df.columns:
        confidence = trade_log_df['buy_prob'].values
    elif 'sell_prob' in trade_log_df.columns:
        confidence = trade_log_df['sell_prob'].values
    
    buy_probs = trade_log_df['buy_prob'].values if 'buy_prob' in trade_log_df.columns else None
    sell_probs = trade_log_df['sell_prob'].values if 'sell_prob' in trade_log_df.columns else None
    
    # Regime mapping - if already numeric (1, -1, 0), use directly
    regime_map = {'BULLISH': 1, 'BEARISH': -1, 'SIDEWAYS': 0, 'RANGE': 0}
    regimes = None
    if 'regime' in trade_log_df.columns:
        regime_vals = trade_log_df['regime'].values
        # Check if already numeric or strings
        if isinstance(regime_vals[0], (int, np.integer, float, np.floating)):
            regimes = np.asarray(regime_vals, dtype=int)
        else:
            regimes = np.array([regime_map.get(str(r).upper(), 0) for r in regime_vals])
    
    atr_percent = trade_log_df['atr'].values if 'atr' in trade_log_df.columns else None
    
    return create_position_heatmap(
        dates=dates,
        prices=prices,
        positions=positions,
        final_signals=final_signals,
        confidence=confidence,
        buy_probs=buy_probs,
        sell_probs=sell_probs,
        regimes=regimes,
        atr_percent=atr_percent,
        equity_curve=equity_curve,
        symbol=symbol,
        timestamp=timestamp,
        out_dir=out_dir,
        show_plot=False
    )
