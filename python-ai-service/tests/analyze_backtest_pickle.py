"""analyze_backtest_pickle.py

Utility to inspect and extract CSVs/plots from backtest .pkl files.

Usage:
    python analyze_backtest_pickle.py path/to/backtest_file.pkl

Outputs (saved next to input file):
    - trades.csv
    - daily_positions.csv
    - signals.csv
    - equity_curve.png
    - positions_hist.png
    - confidence_scatter.png

This script is defensive and will attempt to locate common keys used
by the project's backtester/inference outputs.
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None


COMMON_DATE_KEYS = ['dates', 'date', 'index', 'timestamps']
COMMON_PRICE_KEYS = ['prices', 'price', 'close', 'Close']
COMMON_POSITION_KEYS = ['positions', 'position', 'positions_series']
COMMON_EQUITY_KEYS = ['equity', 'equity_curve', 'portfolio_value', 'portfolio']
COMMON_BUY_HOLD_KEYS = ['buy_hold_equity', 'buy_hold', 'buy_hold_curve']
COMMON_TRADES_KEYS = ['trade_log', 'trades', 'trade_log_list', 'tradeMarkers', 'trade_markers']
COMMON_BUY_PROB = ['buy_probs', 'buy_prob', 'buy_probabilities', 'buy_probability']
COMMON_SELL_PROB = ['sell_probs', 'sell_prob', 'sell_probabilities', 'sell_probability']
COMMON_SIGNAL_KEYS = ['signals', 'final_signals', 'final_signal']


def load_pickle(path: Path) -> Any:
    with open(path, 'rb') as f:
        return pickle.load(f)


def flatten_paths(obj: Any, prefix: str = '') -> dict:
    """Return a mapping from dotted path -> value for dict-like structures."""
    out = {}

    def _rec(o, p):
        if isinstance(o, dict):
            for k, v in o.items():
                _rec(v, f"{p}.{k}" if p else k)
        elif hasattr(o, '__dict__') and not isinstance(o, (list, tuple, np.ndarray, pd.DataFrame)):
            for k, v in vars(o).items():
                _rec(v, f"{p}.{k}" if p else k)
        else:
            out[p] = o

    _rec(obj, prefix)
    return out


def find_by_names(mapping: dict, candidates: list[str]):
    for name in candidates:
        # exact match
        if name in mapping:
            return mapping[name], name
    # try case-insensitive contains
    for k in mapping.keys():
        for name in candidates:
            if name.lower() in k.lower():
                return mapping[k], k
    return None, None


def to_series_like(x):
    if x is None:
        return None
    if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
        return np.asarray(x)
    return None


def try_make_dataframe(dates, prices, positions, equity):
    # coerce dates
    idx = None
    try:
        idx = pd.to_datetime(dates)
    except Exception:
        idx = pd.RangeIndex(start=0, stop=len(dates) if dates is not None else len(positions) if positions is not None else 0)

    df = pd.DataFrame(index=idx)
    if prices is not None:
        df['price'] = prices
    if positions is not None:
        df['position'] = positions
    if equity is not None:
        df['equity'] = equity
    return df


def analyze(path: Path, max_pos_limit: float = 1.0, output_dir: Path | None = None):
    data = load_pickle(path)

    print(f"Loaded: {path} -> type={type(data)}")

    # Print top-level structure
    if isinstance(data, dict):
        print("Top-level keys:")
        for k in sorted(data.keys()):
            v = data[k]
            t = type(v)
            summary = ''
            try:
                if hasattr(v, '__len__'):
                    summary = f'len={len(v)}'
            except Exception:
                summary = ''
            print(f" - {k}: {t.__name__} {summary}")
    else:
        # Try to introspect attributes
        print("Top-level object members (dir):")
        for name in sorted(dir(data)):
            if name.startswith('_'):
                continue
            try:
                val = getattr(data, name)
                tname = type(val).__name__
            except Exception:
                tname = '<?> '
            print(f" - {name}: {tname}")

    # Flatten
    flat = flatten_paths(data)

    # Find arrays
    dates_v, dates_key = find_by_names(flat, COMMON_DATE_KEYS)
    prices_v, price_key = find_by_names(flat, COMMON_PRICE_KEYS)
    positions_v, pos_key = find_by_names(flat, COMMON_POSITION_KEYS)
    equity_v, equity_key = find_by_names(flat, COMMON_EQUITY_KEYS)
    buy_hold_v, buy_hold_key = find_by_names(flat, COMMON_BUY_HOLD_KEYS)
    trades_v, trades_key = find_by_names(flat, COMMON_TRADES_KEYS)
    buy_v, buy_key = find_by_names(flat, COMMON_BUY_PROB)
    sell_v, sell_key = find_by_names(flat, COMMON_SELL_PROB)
    signals_v, signals_key = find_by_names(flat, COMMON_SIGNAL_KEYS)

    # Fallback: often backtest saved as {'backtest': {...}}
    if dates_v is None and isinstance(data, dict) and 'backtest' in data and isinstance(data['backtest'], dict):
        back = data['backtest']
        flat_back = flatten_paths(back, prefix='backtest')
        d2, _ = find_by_names(flat_back, COMMON_DATE_KEYS)
        if d2 is not None:
            dates_v = d2
        p2, _ = find_by_names(flat_back, COMMON_PRICE_KEYS)
        if p2 is not None:
            prices_v = p2
        e2, _ = find_by_names(flat_back, COMMON_EQUITY_KEYS)
        if e2 is not None:
            equity_v = e2

    # Normalize series
    dates = to_series_like(dates_v)
    prices = to_series_like(prices_v)
    positions = to_series_like(positions_v)
    equity = to_series_like(equity_v)
    buy_hold = to_series_like(buy_hold_v)
    buy_probs = to_series_like(buy_v)
    sell_probs = to_series_like(sell_v)
    signals = to_series_like(signals_v)

    # Summaries
    if dates is not None:
        try:
            dt0 = pd.to_datetime(dates[0])
            dt1 = pd.to_datetime(dates[-1])
            print(f"Date range: {dt0} -> {dt1} ({len(dates)} rows)")
        except Exception:
            print(f"Date-like array length: {len(dates)}")
    else:
        # try infer from equity or positions length
        n = None
        for arr in (prices, positions, equity, buy_probs, sell_probs):
            if arr is not None:
                n = len(arr)
                break
        print(f"No dates found; inferred length={n}")

    # Trades
    trades_df = None
    trades_count = 0
    if trades_v is not None:
        try:
            trades_df = pd.DataFrame(trades_v)
            trades_count = len(trades_df)
            print(f"Found trade log ({trades_key}) with {trades_count} entries")
        except Exception:
            print(f"Found trade log at {trades_key} but couldn't convert to DataFrame")
    else:
        # Try nested keys
        for k in flat.keys():
            if k.lower().endswith('trade_log') or k.lower().endswith('trades'):
                try:
                    trades_df = pd.DataFrame(flat[k])
                    trades_count = len(trades_df)
                    print(f"Found trade log at nested key {k} with {trades_count} entries")
                    break
                except Exception:
                    continue

    # Position counts and signal distribution
    non_zero_positions = 0
    signal_dist = {'buy': 0, 'sell': 0, 'hold': 0}
    if positions is not None:
        non_zero_positions = int(np.sum(np.abs(positions) > 0))
        signal_dist['buy'] = int(np.sum(positions > 0))
        signal_dist['sell'] = int(np.sum(positions < 0))
        signal_dist['hold'] = int(np.sum(positions == 0))
        print(f"Non-zero position days: {non_zero_positions}")
        print(f"Signal distribution (by positions): BUY={signal_dist['buy']} SELL={signal_dist['sell']} HOLD={signal_dist['hold']}")
    elif signals is not None:
        # signals array might be 1/-1/0
        signal_dist['buy'] = int(np.sum(signals == 1))
        signal_dist['sell'] = int(np.sum(signals == -1))
        signal_dist['hold'] = int(np.sum(signals == 0))
        print(f"Signal distribution (from signals): BUY={signal_dist['buy']} SELL={signal_dist['sell']} HOLD={signal_dist['hold']}")
    else:
        print("No positions or signals found to compute signal distribution")

    # Create output dir
    if output_dir is None:
        output_dir = path.parent
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save trades.csv
    if trades_df is not None:
        out_trades = output_dir / (path.stem + '_trades.csv')
        trades_df.to_csv(out_trades, index=False)
        print(f"Saved trades CSV -> {out_trades}")

    # Save daily_positions.csv
    if any(x is not None for x in (dates, prices, positions, equity)):
        df = try_make_dataframe(dates if dates is not None else list(range(len(prices) if prices is not None else (len(positions) if positions is not None else 0))), prices, positions, equity)
        out_pos = output_dir / (path.stem + '_daily_positions.csv')
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'date'}, inplace=True)
        df.to_csv(out_pos, index=False)
        print(f"Saved daily positions CSV -> {out_pos}")

    # Save signals.csv
    sig_df = None
    if buy_probs is not None or sell_probs is not None or signals is not None:
        n = None
        for arr in (buy_probs, sell_probs, signals):
            if arr is not None:
                n = len(arr)
                break
        idx = None
        if dates is not None:
            try:
                idx = pd.to_datetime(dates[:n])
            except Exception:
                idx = pd.RangeIndex(n)
        else:
            idx = pd.RangeIndex(n)
        sig_df = pd.DataFrame(index=idx)
        if buy_probs is not None:
            sig_df['buy_prob'] = buy_probs[:n]
        if sell_probs is not None:
            sig_df['sell_prob'] = sell_probs[:n]
        if signals is not None:
            sig_df['final_signal'] = signals[:n]
        else:
            # derive final signal from positions if available
            if positions is not None:
                sig_df['final_signal'] = np.sign(positions[:n]).astype(int)
        out_sig = output_dir / (path.stem + '_signals.csv')
        sig_df.reset_index(inplace=True)
        sig_df.rename(columns={'index': 'date'}, inplace=True)
        sig_df.to_csv(out_sig, index=False)
        print(f"Saved signals CSV -> {out_sig}")

    # Quick plots
    if plt is None:
        print("matplotlib not available; skipping plots")
    else:
        try:
            # Equity curve
            if equity is not None:
                fig, ax = plt.subplots(figsize=(10, 4))
                idx = pd.to_datetime(dates) if dates is not None else range(len(equity))
                ax.plot(idx, equity, label='Strategy Equity', color='tab:blue')
                # If buy-hold series present, plot it for comparison
                if buy_hold is not None:
                    try:
                        bh = np.asarray(buy_hold, dtype=float)
                        # Align lengths if necessary
                        if len(bh) != len(equity):
                            bh = bh[-len(equity):]
                        ax.plot(idx, bh, label='Buy & Hold', color='tab:green', linestyle='--', alpha=0.9)
                        ax.set_ylabel('Equity / Benchmark')
                    except Exception:
                        pass
                else:
                    ax.set_ylabel('Equity')

                ax.set_title('Equity Curve')
                ax.set_xlabel('Date')
                ax.grid(True)
                ax.legend(loc='upper left')
                out_plot = output_dir / (path.stem + '_equity_curve.png')
                # Check if equity_curve.png already exists (created by inference_and_backtest)
                simple_equity_path = output_dir / 'equity_curve.png'
                if simple_equity_path.exists():
                    print(f"Skipping equity plot (already exists at {simple_equity_path})")
                    plt.close(fig)
                else:
                    fig.tight_layout()
                    fig.savefig(out_plot)
                    plt.close(fig)
                    print(f"Saved equity plot -> {out_plot}")

            # Position distribution histogram
            if positions is not None:
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.hist(positions, bins=40)
                ax.set_title('Position Distribution')
                ax.set_xlabel('Position size')
                ax.set_ylabel('Count')
                out_plot = output_dir / (path.stem + '_positions_hist.png')
                fig.tight_layout()
                fig.savefig(out_plot)
                plt.close(fig)
                print(f"Saved positions histogram -> {out_plot}")

            # Confidence scatter: buy_prob vs returns (if returns present in flat)
            returns_v, returns_key = find_by_names(flat, ['returns', 'rets', 'predicted_returns', 'returns_array'])
            returns = to_series_like(returns_v)
            if buy_probs is not None and returns is not None:
                n = min(len(buy_probs), len(returns))
                fig, ax = plt.subplots(figsize=(6, 4))
                ax.scatter(buy_probs[:n], returns[:n], alpha=0.6)
                ax.set_xlabel('buy_prob')
                ax.set_ylabel('return')
                ax.set_title('Buy prob vs returns')
                out_plot = output_dir / (path.stem + '_confidence_scatter.png')
                fig.tight_layout()
                fig.savefig(out_plot)
                plt.close(fig)
                print(f"Saved confidence scatter -> {out_plot}")
        except Exception as e:
            print(f"Plotting error: {e}")

    # Validation checks
    print('\n=== Validation checks ===')
    # NaNs in positions
    if positions is not None:
        nan_count = int(np.sum(np.isnan(positions)))
        print(f"NaNs in positions: {nan_count}")
    else:
        print("Positions not available for NaN check")

    # Position limits
    if positions is not None:
        max_pos = float(np.nanmax(np.abs(positions)))
        print(f"Max absolute position observed: {max_pos:.4f}")
        if max_pos > max_pos_limit:
            print(f"  ⚠️ Position limit exceeded: observed {max_pos:.3f} > allowed {max_pos_limit}")
        else:
            print("  Position limits respected")
    else:
        print("Positions not available to check limits")

    # Dates alignment
    lens = {}
    for name, arr in [('dates', dates), ('prices', prices), ('positions', positions), ('equity', equity)]:
        if arr is not None:
            lens[name] = len(arr)
    if lens:
        print(f"Lengths: {lens}")
        unique_lengths = set(lens.values())
        if len(unique_lengths) > 1:
            print("  ⚠️ Length mismatch detected across date/price/position/equity arrays")
        else:
            print("  Date/price/position/equity arrays aligned")

    print('\nAnalysis complete')


def main_cli():
    p = argparse.ArgumentParser(description='Analyze backtest .pkl files and extract CSVs/plots')
    p.add_argument('pkl', type=str, help='Path to backtest .pkl file')
    p.add_argument('--max-pos', type=float, default=1.0, help='Maximum allowed absolute position (validation)')
    p.add_argument('--outdir', type=str, default=None, help='Output directory for CSVs/plots (defaults to pkl folder)')
    args = p.parse_args()

    path = Path(args.pkl)
    if not path.exists():
        print(f"File not found: {path}")
        raise SystemExit(1)

    analyze(path, max_pos_limit=args.max_pos, output_dir=Path(args.outdir) if args.outdir else None)


if __name__ == '__main__':
    main_cli()
