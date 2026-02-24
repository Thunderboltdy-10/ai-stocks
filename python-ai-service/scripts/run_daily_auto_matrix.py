"""Run daily matrix using automatic variant routing (no retraining)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_backtest import BacktestConfig, UnifiedBacktester
from utils.symbol_universe import DEFAULT_DAILY_SET, available_symbol_sets, resolve_symbols


def _windows(now: pd.Timestamp) -> list[tuple[str, str, str]]:
    end = now.normalize()
    return [
        ((end - pd.Timedelta(days=365 * 5)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "daily_5y"),
        ((end - pd.Timedelta(days=365 * 2)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "daily_2y"),
        ((end - pd.Timedelta(days=365)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "daily_1y"),
        ((end - pd.Timedelta(days=180)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "daily_6m"),
        ((end - pd.Timedelta(days=90)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "daily_3m"),
        ((end - pd.Timedelta(days=30)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "daily_1m"),
    ]


class _FastBacktester(UnifiedBacktester):
    def _save_outputs(self, *args, **kwargs):
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Daily matrix with auto variant routing")
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument(
        "--symbol-set",
        type=str,
        default=DEFAULT_DAILY_SET,
        help=f"Preset symbol set ({', '.join(available_symbol_sets())})",
    )
    parser.add_argument("--period", type=str, default="max")
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument("--max-long", type=float, default=1.6)
    parser.add_argument("--max-short", type=float, default=0.6)
    parser.add_argument("--commission", type=float, default=0.0005)
    parser.add_argument("--slippage", type=float, default=0.0003)
    parser.add_argument("--warmup-days", type=int, default=252)
    parser.add_argument("--min-eval-days", type=int, default=20)
    args = parser.parse_args()

    symbols = resolve_symbols(args.symbols, symbol_set=args.symbol_set, fallback_set=DEFAULT_DAILY_SET)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    windows = _windows(now)
    print(f"[INFO] symbols ({len(symbols)}): {', '.join(symbols)}")

    rows: list[dict] = []
    for symbol in symbols:
        for start, end, label in windows:
            cfg = BacktestConfig(
                symbol=symbol,
                start=start,
                end=end,
                model_variant="auto",
                data_period=args.period,
                data_interval=args.interval,
                max_long=args.max_long,
                max_short=args.max_short,
                commission_pct=args.commission,
                slippage_pct=args.slippage,
                warmup_days=args.warmup_days,
                min_eval_days=args.min_eval_days,
            )
            try:
                summary = _FastBacktester(cfg).run()
                summary["window"] = label
                rows.append(summary)
                print(
                    f"[BT] {symbol} {label} variant={summary.get('model_variant')} "
                    f"ret={summary['strategy_return']:.4f} alpha={summary['alpha']:.4f}"
                )
            except Exception as exc:
                print(f"[BT-FAIL] {symbol} {label}: {exc}")

    out = pd.DataFrame(rows)
    out_dir = Path("experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"daily_auto_matrix_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    if not out.empty:
        print(
            out[["symbol", "window", "model_variant", "strategy_return", "alpha", "buy_hold_return", "sharpe_ratio"]]
            .sort_values(["symbol", "window"])
            .to_string(index=False)
        )


if __name__ == "__main__":
    main()
