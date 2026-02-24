"""Build per-symbol intraday variant registry from backtest benchmarks.

This script evaluates available `gbm_intraday_1h*` variants and stores the
best long-window and short-window variants in:

  saved_models/{SYMBOL}/intraday_variant_registry.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Dict, List

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_backtest import BacktestConfig, UnifiedBacktester
from inference.variant_quality import load_training_metadata, score_variant_quality
from utils.symbol_universe import DEFAULT_INTRADAY_SET, available_symbol_sets, resolve_symbols


WINDOWS = [
    ("intraday_1y", 365),
    ("intraday_6m", 180),
    ("intraday_3m", 90),
    ("intraday_1m", 30),
    ("intraday_2w", 14),
]
LONG_WINDOWS = {"intraday_1y", "intraday_6m", "intraday_3m"}
SHORT_WINDOWS = {"intraday_1m", "intraday_2w"}


class _FastBacktester(UnifiedBacktester):
    def _save_outputs(self, *args, **kwargs):
        return


def _variant_dirs(symbol: str) -> List[str]:
    root = Path("saved_models") / symbol.upper()
    if not root.exists():
        return []
    out = []
    for d in sorted(root.glob("gbm_intraday_1h*")):
        if not d.is_dir():
            continue
        if not (d / "xgb_model.joblib").exists():
            continue
        out.append(d.name)
    return out


def _mean_or_default(values: List[float], default: float = -999.0) -> float:
    if not values:
        return default
    return float(sum(values) / len(values))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build intraday variant registry")
    parser.add_argument("--symbols", type=str, default="")
    parser.add_argument(
        "--symbol-set",
        type=str,
        default=DEFAULT_INTRADAY_SET,
        help=f"Preset symbol set ({', '.join(available_symbol_sets())})",
    )
    parser.add_argument("--period", type=str, default="730d")
    parser.add_argument("--interval", type=str, default="1h")
    parser.add_argument("--max-long", type=float, default=1.4)
    parser.add_argument("--max-short", type=float, default=0.2)
    parser.add_argument("--commission", type=float, default=0.0001)
    parser.add_argument("--slippage", type=float, default=0.0001)
    parser.add_argument("--warmup-days", type=int, default=40)
    parser.add_argument("--min-eval-days", type=int, default=15)
    args = parser.parse_args()

    symbols = resolve_symbols(args.symbols, symbol_set=args.symbol_set, fallback_set=DEFAULT_INTRADAY_SET)
    end = pd.Timestamp.now(tz="UTC").tz_localize(None).floor("h")
    print(f"[INFO] symbols ({len(symbols)}): {', '.join(symbols)}")

    rows: List[Dict] = []
    for symbol in symbols:
        variants = _variant_dirs(symbol)
        if not variants:
            print(f"[SKIP] {symbol}: no intraday variants found")
            continue

        variant_scores: Dict[str, Dict[str, object]] = {}
        for variant in variants:
            quality = score_variant_quality(
                load_training_metadata(symbol, variant),
                intraday=True,
            )
            bucket = {
                "overall_ret": [],
                "overall_alpha": [],
                "overall_sharpe": [],
                "overall_hit": [],
                "long_ret": [],
                "long_alpha": [],
                "long_hit": [],
                "short_ret": [],
                "short_alpha": [],
                "short_hit": [],
            }

            for label, days in WINDOWS:
                start = end - pd.Timedelta(days=days)
                cfg = BacktestConfig(
                    symbol=symbol,
                    start=start.strftime("%Y-%m-%d %H:%M:%S"),
                    end=end.strftime("%Y-%m-%d %H:%M:%S"),
                    model_variant=variant,
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
                except Exception as exc:
                    print(f"[FAIL] {symbol} {variant} {label}: {exc}")
                    continue

                ret = float(summary.get("strategy_return", 0.0))
                alpha = float(summary.get("alpha", 0.0))
                sharpe = float(summary.get("sharpe_ratio", 0.0))
                rows.append(
                    {
                        "symbol": symbol,
                        "variant": variant,
                        "window": label,
                        "strategy_return": ret,
                        "alpha": alpha,
                        "sharpe": sharpe,
                    }
                )

                bucket["overall_ret"].append(ret)
                bucket["overall_alpha"].append(alpha)
                bucket["overall_sharpe"].append(sharpe)
                bucket["overall_hit"].append(1.0 if alpha > 0.0 else 0.0)
                if label in LONG_WINDOWS:
                    bucket["long_ret"].append(ret)
                    bucket["long_alpha"].append(alpha)
                    bucket["long_hit"].append(1.0 if alpha > 0.0 else 0.0)
                elif label in SHORT_WINDOWS:
                    bucket["short_ret"].append(ret)
                    bucket["short_alpha"].append(alpha)
                    bucket["short_hit"].append(1.0 if alpha > 0.0 else 0.0)

            mean_ret = _mean_or_default(bucket["overall_ret"])
            mean_alpha = _mean_or_default(bucket["overall_alpha"])
            mean_sharpe = _mean_or_default(bucket["overall_sharpe"])
            long_ret = _mean_or_default(bucket["long_ret"])
            long_alpha = _mean_or_default(bucket["long_alpha"])
            short_ret = _mean_or_default(bucket["short_ret"])
            short_alpha = _mean_or_default(bucket["short_alpha"])
            overall_hit_rate = _mean_or_default(bucket["overall_hit"], default=0.0)
            long_hit_rate = _mean_or_default(bucket["long_hit"], default=0.0)
            short_hit_rate = _mean_or_default(bucket["short_hit"], default=0.0)
            variant_scores[variant] = {
                "mean_strategy_return": mean_ret,
                "mean_alpha": mean_alpha,
                "mean_sharpe": mean_sharpe,
                "alpha_hit_rate": overall_hit_rate,
                "long_window_return": long_ret,
                "long_window_alpha": long_alpha,
                "long_window_hit_rate": long_hit_rate,
                "short_window_return": short_ret,
                "short_window_alpha": short_alpha,
                "short_window_hit_rate": short_hit_rate,
                "quality_score": float(quality.score),
                "quality_pass": bool(quality.passed),
                "quality_reasons": quality.reasons,
                "quality_metrics": quality.metrics,
                "overall_score": float(
                    2.1 * mean_alpha
                    + 0.9 * mean_ret
                    + 0.18 * mean_sharpe
                    + 1.2 * (overall_hit_rate - 0.5)
                    + 0.30 * quality.score
                    + (0.70 if quality.passed else -1.40)
                ),
                "long_window_score": float(
                    2.1 * long_alpha
                    + 0.9 * long_ret
                    + 1.0 * (long_hit_rate - 0.5)
                    + 0.20 * quality.score
                    + (0.45 if quality.passed else -0.90)
                ),
                "short_window_score": float(
                    2.1 * short_alpha
                    + 0.9 * short_ret
                    + 1.0 * (short_hit_rate - 0.5)
                    + 0.20 * quality.score
                    + (0.45 if quality.passed else -0.90)
                ),
            }

        if not variant_scores:
            continue

        def _pick(metric: str) -> str:
            quality_ok = [v for v in variant_scores if bool(variant_scores[v].get("quality_pass", False))]
            candidate_pool = quality_ok if quality_ok else list(variant_scores.keys())
            return max(candidate_pool, key=lambda v: float(variant_scores[v].get(metric, -999.0)))

        registry = {
            "symbol": symbol,
            "generated_at": pd.Timestamp.utcnow().isoformat(),
            "interval": args.interval,
            "period": args.period,
            "symbol_set": args.symbol_set,
            "overall_variant": _pick("overall_score"),
            "long_window_variant": _pick("long_window_score"),
            "short_window_variant": _pick("short_window_score"),
            "variant_scores": variant_scores,
        }

        reg_path = Path("saved_models") / symbol / "intraday_variant_registry.json"
        reg_path.parent.mkdir(parents=True, exist_ok=True)
        with reg_path.open("w", encoding="utf-8") as fh:
            json.dump(registry, fh, indent=2)

        print(
            f"[REGISTRY] {symbol} overall={registry['overall_variant']} "
            f"long={registry['long_window_variant']} short={registry['short_window_variant']}"
        )

    out = pd.DataFrame(rows)
    out_dir = Path("experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"intraday_variant_registry_matrix_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv"
    out.to_csv(csv_path, index=False)
    print(f"Saved matrix: {csv_path}")


if __name__ == "__main__":
    main()
