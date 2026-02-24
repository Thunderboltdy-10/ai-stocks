"""Train and evaluate hourly (intraday) GBM models on a symbol matrix.

Usage:
  python scripts/run_intraday_hourly_matrix.py --symbols AAPL,TSLA,XOM,JPM,KO --overwrite
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_backtest import BacktestConfig, UnifiedBacktester
from training.train_gbm import GBMTrainer
from utils.symbol_universe import DEFAULT_INTRADAY_SET, available_symbol_sets, resolve_symbols


def _parse_windows(now: pd.Timestamp) -> list[tuple[str, str, str]]:
    end = now.floor("h")
    return [
        ((end - pd.Timedelta(days=365)).strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S"), "intraday_1y"),
        ((end - pd.Timedelta(days=180)).strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S"), "intraday_6m"),
        ((end - pd.Timedelta(days=90)).strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S"), "intraday_3m"),
        ((end - pd.Timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S"), "intraday_1m"),
        ((end - pd.Timedelta(days=14)).strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S"), "intraday_2w"),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Hourly intraday train/backtest matrix")
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument(
        "--symbol-set",
        type=str,
        default=DEFAULT_INTRADAY_SET,
        help=f"Preset symbol set ({', '.join(available_symbol_sets())})",
    )
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--max-features", type=int, default=50)
    parser.add_argument("--period", type=str, default="730d")
    parser.add_argument("--interval", type=str, default="1h")
    parser.add_argument("--model-suffix", type=str, default="intraday_1h_v3")
    parser.add_argument("--max-long", type=float, default=1.4)
    parser.add_argument("--max-short", type=float, default=0.2)
    parser.add_argument("--commission", type=float, default=0.0001, help="Decimal pct per trade (0.0001 = 1 bps)")
    parser.add_argument("--slippage", type=float, default=0.0001, help="Decimal pct per trade")
    parser.add_argument("--target-horizon", type=int, default=1, help="Forward horizon in bars for target generation")
    parser.add_argument("--warmup-days", type=int, default=40)
    parser.add_argument("--min-eval-days", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    args = parser.parse_args()

    symbols = resolve_symbols(args.symbols, symbol_set=args.symbol_set, fallback_set=DEFAULT_INTRADAY_SET)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    windows = _parse_windows(now)
    print(f"[INFO] symbols ({len(symbols)}): {', '.join(symbols)}")

    rows: list[dict] = []
    model_variant = f"gbm_{args.model_suffix}" if args.model_suffix else f"gbm_{args.interval}_{args.period}"

    for symbol in symbols:
        trainer = GBMTrainer(
            symbol=symbol,
            n_trials=args.n_trials,
            overwrite=args.overwrite,
            max_features=args.max_features,
            use_lgb=False,
            allow_cpu_fallback=args.allow_cpu_fallback,
            data_period=args.period,
            data_interval=args.interval,
            model_suffix=args.model_suffix,
            target_horizon=args.target_horizon,
        )
        metadata = trainer.train()
        holdout = metadata.get("holdout", {}).get("ensemble", {})
        print(
            f"[TRAIN] {symbol} variant={model_variant} dir_acc={holdout.get('dir_acc', 0):.4f} "
            f"pred_std={holdout.get('pred_std', 0):.6f} wfe={metadata.get('wfe', 0):.1f}"
        )

        for start, end, label in windows:
            cfg = BacktestConfig(
                symbol=symbol,
                start=start,
                end=end,
                model_variant=model_variant,
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
                summary = UnifiedBacktester(cfg).run()
                summary["window"] = label
                rows.append(summary)
                print(
                    f"[BT] {symbol} {label} alpha={summary['alpha']:.4f} "
                    f"sharpe={summary['sharpe_ratio']:.3f} n={summary['n_days']}"
                )
            except Exception as exc:
                print(f"[BT-FAIL] {symbol} {label}: {exc}")

    df = pd.DataFrame(rows)
    out_dir = Path("experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"intraday_hourly_matrix_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")
    if not df.empty:
        cols = ["symbol", "window", "alpha", "strategy_return", "buy_hold_return", "sharpe_ratio", "n_days"]
        print(df[cols].sort_values(["symbol", "window"]).to_string(index=False))


if __name__ == "__main__":
    main()
