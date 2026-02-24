"""Train and evaluate daily GBM models on a symbol matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_backtest import BacktestConfig, UnifiedBacktester
from training.train_gbm import GBMTrainer
from utils.symbol_universe import DEFAULT_DAILY_SET, available_symbol_sets, resolve_symbols


def _parse_windows(now: pd.Timestamp) -> list[tuple[str, str, str]]:
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
    parser = argparse.ArgumentParser(description="Daily train/backtest matrix")
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument(
        "--symbol-set",
        type=str,
        default=DEFAULT_DAILY_SET,
        help=f"Preset symbol set ({', '.join(available_symbol_sets())})",
    )
    parser.add_argument("--n-trials", type=int, default=6)
    parser.add_argument("--max-features", type=int, default=50)
    parser.add_argument("--period", type=str, default="max")
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument("--model-suffix", type=str, default="daily_v3")
    parser.add_argument("--max-long", type=float, default=1.6)
    parser.add_argument("--max-short", type=float, default=0.6)
    parser.add_argument("--commission", type=float, default=0.0005)
    parser.add_argument("--slippage", type=float, default=0.0003)
    parser.add_argument("--target-horizon", type=int, default=1)
    parser.add_argument("--warmup-days", type=int, default=252)
    parser.add_argument("--min-eval-days", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    args = parser.parse_args()

    symbols = resolve_symbols(args.symbols, symbol_set=args.symbol_set, fallback_set=DEFAULT_DAILY_SET)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    windows = _parse_windows(now)
    model_variant = f"gbm_{args.model_suffix}" if args.model_suffix else "gbm"
    print(f"[INFO] symbols ({len(symbols)}): {', '.join(symbols)}")

    rows: list[dict] = []
    train_summary: list[dict] = []

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
        hold = metadata.get("holdout", {}).get("ensemble", {})
        hold_cal = metadata.get("holdout", {}).get("ensemble_calibrated", {})
        train_summary.append(
            {
                "symbol": symbol,
                "dir_raw": hold.get("dir_acc", 0.0),
                "dir_cal": hold_cal.get("dir_acc", 0.0),
                "net_sharpe_raw": hold.get("net_sharpe", 0.0),
                "net_sharpe_cal": hold_cal.get("net_sharpe", 0.0),
                "std_raw": hold.get("pred_std", 0.0),
                "std_cal": hold_cal.get("pred_std", 0.0),
                "wfe": metadata.get("wfe", 0.0),
                "cal_enabled": bool(metadata.get("direction_calibrator", {}).get("enabled", False)),
                "policy_enabled": bool(metadata.get("execution_calibration", {}).get("enabled", False)),
            }
        )
        print(
            f"[TRAIN] {symbol} variant={model_variant} "
            f"dir_acc={max(float(hold.get('dir_acc', 0)), float(hold_cal.get('dir_acc', 0))):.4f} "
            f"wfe={float(metadata.get('wfe', 0.0)):.1f}"
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
                summary = _FastBacktester(cfg).run()
                summary["window"] = label
                rows.append(summary)
                print(
                    f"[BT] {symbol} {label} alpha={summary['alpha']:.4f} "
                    f"ret={summary['strategy_return']:.4f} bh={summary['buy_hold_return']:.4f}"
                )
            except Exception as exc:
                print(f"[BT-FAIL] {symbol} {label}: {exc}")

    df = pd.DataFrame(rows)
    out_dir = Path("experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    csv_path = out_dir / f"daily_matrix_{ts}.csv"
    json_path = out_dir / f"daily_train_summary_{ts}.json"
    df.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(train_summary, fh, indent=2)

    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    if not df.empty:
        cols = ["symbol", "window", "alpha", "strategy_return", "buy_hold_return", "sharpe_ratio", "n_days"]
        print(df[cols].sort_values(["symbol", "window"]).to_string(index=False))


if __name__ == "__main__":
    main()
