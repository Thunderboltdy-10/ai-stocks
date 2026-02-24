"""Evaluate generalization across core vs holdout symbol groups.

This script enforces a simple anti-overfit gate: improvements are accepted only
if holdout behavior remains healthy, not just the core symbols.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_backtest import BacktestConfig, UnifiedBacktester
from utils.symbol_universe import (
    available_symbol_sets,
    resolve_symbols,
)
from utils.timeframe import is_intraday_interval


class _FastBacktester(UnifiedBacktester):
    def _save_outputs(self, *args, **kwargs):
        return


def _windows(now: pd.Timestamp, *, intraday: bool) -> List[Tuple[str, str, str]]:
    if intraday:
        end = now.floor("h")
        return [
            ((end - pd.Timedelta(days=365)).strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S"), "intraday_1y"),
            ((end - pd.Timedelta(days=180)).strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S"), "intraday_6m"),
            ((end - pd.Timedelta(days=90)).strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S"), "intraday_3m"),
            ((end - pd.Timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S"), "intraday_1m"),
            ((end - pd.Timedelta(days=14)).strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S"), "intraday_2w"),
        ]

    end = now.normalize()
    return [
        ((end - pd.Timedelta(days=365 * 5)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "daily_5y"),
        ((end - pd.Timedelta(days=365 * 2)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "daily_2y"),
        ((end - pd.Timedelta(days=365)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "daily_1y"),
        ((end - pd.Timedelta(days=180)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "daily_6m"),
        ((end - pd.Timedelta(days=90)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "daily_3m"),
        ((end - pd.Timedelta(days=30)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "daily_1m"),
    ]


def _dedupe_keep_order(symbols: List[str]) -> List[str]:
    out: List[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        s = str(symbol).strip().upper()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _group_metrics(rows: pd.DataFrame) -> Dict[str, float]:
    if rows.empty:
        return {
            "n_runs": 0,
            "n_symbols": 0,
            "mean_alpha": 0.0,
            "median_alpha": 0.0,
            "alpha_hit_rate": 0.0,
            "mean_strategy_return": 0.0,
            "mean_buy_hold_return": 0.0,
            "mean_sharpe": 0.0,
            "worst_alpha": 0.0,
            "best_alpha": 0.0,
        }

    alpha = rows["alpha"].astype(float)
    return {
        "n_runs": int(len(rows)),
        "n_symbols": int(rows["symbol"].nunique()),
        "mean_alpha": float(alpha.mean()),
        "median_alpha": float(alpha.median()),
        "alpha_hit_rate": float((alpha > 0.0).mean()),
        "mean_strategy_return": float(rows["strategy_return"].astype(float).mean()),
        "mean_buy_hold_return": float(rows["buy_hold_return"].astype(float).mean()),
        "mean_sharpe": float(rows["sharpe_ratio"].astype(float).mean()),
        "worst_alpha": float(alpha.min()),
        "best_alpha": float(alpha.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generalization gate across core vs holdout symbols")
    parser.add_argument("--core-symbols", type=str, default="")
    parser.add_argument("--holdout-symbols", type=str, default="")
    parser.add_argument(
        "--core-symbol-set",
        type=str,
        default="",
        help=f"Preset symbol set ({', '.join(available_symbol_sets())})",
    )
    parser.add_argument(
        "--holdout-symbol-set",
        type=str,
        default="",
        help=f"Preset symbol set ({', '.join(available_symbol_sets())})",
    )
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument("--period", type=str, default="")
    parser.add_argument("--model-variant", type=str, default="")
    parser.add_argument("--max-long", type=float, default=1.6)
    parser.add_argument("--max-short", type=float, default=0.6)
    parser.add_argument("--commission", type=float, default=0.0005)
    parser.add_argument("--slippage", type=float, default=0.0003)
    parser.add_argument("--warmup-days", type=int, default=252)
    parser.add_argument("--min-eval-days", type=int, default=20)
    parser.add_argument("--min-holdout-mean-alpha", type=float, default=float("nan"))
    parser.add_argument("--min-holdout-hit-rate", type=float, default=float("nan"))
    parser.add_argument("--max-generalization-gap", type=float, default=float("nan"))
    parser.add_argument("--min-holdout-worst-alpha", type=float, default=float("nan"))
    args = parser.parse_args()

    intraday = is_intraday_interval(args.interval)

    default_core_set = "core5" if intraday else "core10"
    default_holdout_set = "holdout5" if intraday else "holdout5_ext"

    core_symbols = resolve_symbols(
        args.core_symbols,
        symbol_set=(args.core_symbol_set or default_core_set),
        fallback_set=default_core_set,
    )
    holdout_symbols = resolve_symbols(
        args.holdout_symbols,
        symbol_set=(args.holdout_symbol_set or default_holdout_set),
        fallback_set=default_holdout_set,
    )

    core_symbols = _dedupe_keep_order(core_symbols)
    holdout_symbols = [s for s in _dedupe_keep_order(holdout_symbols) if s not in set(core_symbols)]

    if not core_symbols or not holdout_symbols:
        raise ValueError("Both core and holdout groups must be non-empty")

    data_period = args.period or ("730d" if intraday else "max")
    model_variant = args.model_variant or ("auto_intraday" if intraday else "auto")
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    windows = _windows(now, intraday=intraday)

    min_holdout_mean_alpha = (
        args.min_holdout_mean_alpha
        if pd.notna(args.min_holdout_mean_alpha)
        else (-0.01 if intraday else 0.00)
    )
    min_holdout_hit_rate = (
        args.min_holdout_hit_rate
        if pd.notna(args.min_holdout_hit_rate)
        else (0.40 if intraday else 0.45)
    )
    max_generalization_gap = (
        args.max_generalization_gap
        if pd.notna(args.max_generalization_gap)
        else (0.05 if intraday else 0.08)
    )
    min_holdout_worst_alpha = (
        args.min_holdout_worst_alpha
        if pd.notna(args.min_holdout_worst_alpha)
        else (-0.20 if intraday else -0.30)
    )

    print(f"[INFO] interval={args.interval} intraday={intraday} model_variant={model_variant}")
    print(f"[INFO] core ({len(core_symbols)}): {', '.join(core_symbols)}")
    print(f"[INFO] holdout ({len(holdout_symbols)}): {', '.join(holdout_symbols)}")

    rows: List[Dict] = []
    for group_name, symbols in [("core", core_symbols), ("holdout", holdout_symbols)]:
        for symbol in symbols:
            for start, end, label in windows:
                cfg = BacktestConfig(
                    symbol=symbol,
                    start=start,
                    end=end,
                    model_variant=model_variant,
                    data_period=data_period,
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
                    summary["group"] = group_name
                    rows.append(summary)
                    print(
                        f"[BT] {group_name.upper()} {symbol} {label} "
                        f"alpha={summary['alpha']:.4f} ret={summary['strategy_return']:.4f}"
                    )
                except Exception as exc:
                    print(f"[BT-FAIL] {group_name.upper()} {symbol} {label}: {exc}")

    out = pd.DataFrame(rows)
    core_metrics = _group_metrics(out[out["group"] == "core"] if not out.empty else pd.DataFrame())
    holdout_metrics = _group_metrics(out[out["group"] == "holdout"] if not out.empty else pd.DataFrame())

    generalization_gap = float(core_metrics["mean_alpha"] - holdout_metrics["mean_alpha"])
    checks = {
        "holdout_mean_alpha_ok": holdout_metrics["mean_alpha"] >= min_holdout_mean_alpha,
        "holdout_hit_rate_ok": holdout_metrics["alpha_hit_rate"] >= min_holdout_hit_rate,
        "generalization_gap_ok": generalization_gap <= max_generalization_gap,
        "holdout_worst_alpha_ok": holdout_metrics["worst_alpha"] >= min_holdout_worst_alpha,
    }
    gate_passed = bool(all(checks.values()))

    summary = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "interval": args.interval,
        "intraday": intraday,
        "model_variant": model_variant,
        "data_period": data_period,
        "core_symbols": core_symbols,
        "holdout_symbols": holdout_symbols,
        "core_metrics": core_metrics,
        "holdout_metrics": holdout_metrics,
        "generalization_gap": generalization_gap,
        "thresholds": {
            "min_holdout_mean_alpha": float(min_holdout_mean_alpha),
            "min_holdout_hit_rate": float(min_holdout_hit_rate),
            "max_generalization_gap": float(max_generalization_gap),
            "min_holdout_worst_alpha": float(min_holdout_worst_alpha),
        },
        "checks": checks,
        "gate_passed": gate_passed,
    }

    out_dir = Path("experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    tag = "intraday" if intraday else "daily"
    csv_path = out_dir / f"generalization_matrix_{tag}_{ts}.csv"
    json_path = out_dir / f"generalization_gate_{tag}_{ts}.json"
    out.to_csv(csv_path, index=False)
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")
    print(
        f"[SUMMARY] core_mean_alpha={core_metrics['mean_alpha']:.4f} "
        f"holdout_mean_alpha={holdout_metrics['mean_alpha']:.4f} "
        f"gap={generalization_gap:.4f} gate_passed={gate_passed}"
    )
    print(f"[CHECKS] {checks}")


if __name__ == "__main__":
    main()
