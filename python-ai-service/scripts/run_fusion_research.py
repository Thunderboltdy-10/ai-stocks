"""Sweep fusion-layer parameters for an already-trained GBM variant.

This is intentionally backtest-only: it keeps the trained model fixed and
searches the regime/ML blending architecture to see whether the final strategy
is leaving standalone ML edge on the table.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_backtest import BacktestConfig, UnifiedBacktester


def _parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def _parse_float_csv(raw: str) -> list[float]:
    return [float(part) for part in _parse_csv(raw)]


def _parse_windows(end: pd.Timestamp, mode: str) -> list[tuple[str, str, str]]:
    end = end.normalize()
    if mode == "fast":
        return [
            ((end - pd.Timedelta(days=365 * 5)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "5y"),
            ((end - pd.Timedelta(days=365)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "1y"),
        ]
    if mode == "medium":
        return [
            ((end - pd.Timedelta(days=365 * 5)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "5y"),
            ((end - pd.Timedelta(days=365 * 2)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "2y"),
            ((end - pd.Timedelta(days=365)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "1y"),
        ]
    return [
        ((end - pd.Timedelta(days=365 * 5)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "5y"),
        ((end - pd.Timedelta(days=365 * 2)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "2y"),
        ((end - pd.Timedelta(days=365)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "1y"),
        ((end - pd.Timedelta(days=180)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "6m"),
    ]


def _combo_id(row: dict) -> str:
    return (
        f"ml{row['ml_weight_override']:.2f}"
        f"_fb{row['fallback_ml_weight_override']:.2f}"
        f"_core{row['core_long_override']:.2f}"
        f"_omin{row['overlay_min_override']:.2f}"
        f"_omax{row['overlay_max_override']:.2f}"
        f"_sm{row['smooth_alpha_override']:.2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fusion architecture sweeps for a trained variant")
    parser.add_argument("--symbol", type=str, required=True)
    parser.add_argument("--model-variant", type=str, required=True)
    parser.add_argument("--period", type=str, default="max")
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument("--window-set", type=str, default="fast", choices=["fast", "medium", "full"])
    parser.add_argument("--ml-weights", type=str, default="0.18,0.28,0.40,0.55")
    parser.add_argument("--fallback-ml-weights", type=str, default="0.04,0.08")
    parser.add_argument("--core-longs", type=str, default="0.25,0.55,0.85")
    parser.add_argument("--overlay-mins", type=str, default="0.01,0.03")
    parser.add_argument("--overlay-maxs", type=str, default="0.30,0.50,0.75")
    parser.add_argument("--smooth-alphas", type=str, default="0.12,0.20,0.32")
    parser.add_argument("--max-long", type=float, default=1.6)
    parser.add_argument("--max-short", type=float, default=0.6)
    parser.add_argument("--commission", type=float, default=0.0005)
    parser.add_argument("--slippage", type=float, default=0.0003)
    parser.add_argument("--warmup-days", type=int, default=252)
    parser.add_argument("--min-eval-days", type=int, default=20)
    args = parser.parse_args()

    symbol = args.symbol.upper()
    windows = _parse_windows(pd.Timestamp.now(tz="UTC").tz_localize(None), args.window_set)
    rows: list[dict] = []

    combos = list(
        product(
            _parse_float_csv(args.ml_weights),
            _parse_float_csv(args.fallback_ml_weights),
            _parse_float_csv(args.core_longs),
            _parse_float_csv(args.overlay_mins),
            _parse_float_csv(args.overlay_maxs),
            _parse_float_csv(args.smooth_alphas),
        )
    )
    print(f"[FUSION] {symbol} {args.model_variant} | combos={len(combos)} windows={len(windows)}")

    for ml_weight, fallback_weight, core_long, overlay_min, overlay_max, smooth_alpha in combos:
        if overlay_max <= overlay_min:
            continue
        combo = {
            "ml_weight_override": float(ml_weight),
            "fallback_ml_weight_override": float(fallback_weight),
            "core_long_override": float(core_long),
            "overlay_min_override": float(overlay_min),
            "overlay_max_override": float(overlay_max),
            "smooth_alpha_override": float(smooth_alpha),
        }
        combo_key = _combo_id(combo)
        for start, end_str, label in windows:
            cfg = BacktestConfig(
                symbol=symbol,
                start=start,
                end=end_str,
                model_variant=args.model_variant,
                data_period=args.period,
                data_interval=args.interval,
                max_long=args.max_long,
                max_short=args.max_short,
                commission_pct=args.commission,
                slippage_pct=args.slippage,
                warmup_days=args.warmup_days,
                min_eval_days=args.min_eval_days,
                **combo,
            )
            try:
                summary = UnifiedBacktester(cfg).run()
                rows.append(
                    {
                        "symbol": symbol,
                        "variant": args.model_variant,
                        "window": label,
                        "combo_id": combo_key,
                        **combo,
                        **summary,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "symbol": symbol,
                        "variant": args.model_variant,
                        "window": label,
                        "combo_id": combo_key,
                        "status": "failed",
                        "error": str(exc),
                        **combo,
                    }
                )

    df = pd.DataFrame(rows)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "experiments" / f"fusion_research_{symbol}_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / "fusion_detail.csv"
    df.to_csv(detail_path, index=False)

    ok = df.loc[df["status"] != "failed"].copy() if "status" in df.columns else df.copy()
    summary_path = out_dir / "fusion_summary.csv"
    top_json_path = out_dir / "top_fusion_configs.json"

    if ok.empty:
        pd.DataFrame().to_csv(summary_path, index=False)
        with top_json_path.open("w", encoding="utf-8") as fh:
            json.dump([], fh, indent=2)
        print(f"Saved empty fusion results to: {out_dir}")
        return

    grouped = (
        ok.groupby("combo_id")
        .agg(
            symbol=("symbol", "first"),
            variant=("variant", "first"),
            windows=("window", lambda x: "|".join(sorted(set(str(v) for v in x)))),
            ml_weight_override=("ml_weight_override", "first"),
            fallback_ml_weight_override=("fallback_ml_weight_override", "first"),
            core_long_override=("core_long_override", "first"),
            overlay_min_override=("overlay_min_override", "first"),
            overlay_max_override=("overlay_max_override", "first"),
            smooth_alpha_override=("smooth_alpha_override", "first"),
            alpha_mean=("alpha", "mean"),
            alpha_min=("alpha", "min"),
            strategy_return_mean=("strategy_return", "mean"),
            ml_only_alpha_mean=("ml_only_alpha", "mean"),
            ml_only_alpha_min=("ml_only_alpha", "min"),
            ml_incremental_alpha_mean=("ml_incremental_alpha_vs_regime", "mean"),
            ml_incremental_alpha_min=("ml_incremental_alpha_vs_regime", "min"),
            regime_only_alpha_mean=("regime_only_alpha", "mean"),
            max_drawdown_mean=("max_drawdown", "mean"),
            max_drawdown_worst=("max_drawdown", "min"),
            quality_score_mean=("model_quality_score", "mean"),
            gate_pass_rate=("model_quality_gate_passed", "mean"),
        )
        .reset_index()
    )
    grouped["robust_score"] = (
        3.0 * grouped["alpha_mean"]
        + 2.0 * grouped["alpha_min"]
        + 1.8 * grouped["ml_incremental_alpha_mean"]
        + 1.2 * grouped["ml_incremental_alpha_min"]
        + 0.8 * grouped["ml_only_alpha_mean"]
        + 0.2 * grouped["quality_score_mean"] / 10.0
        - 0.6 * grouped["max_drawdown_mean"].abs()
    )
    grouped = grouped.sort_values(
        ["robust_score", "alpha_min", "alpha_mean", "ml_incremental_alpha_mean"],
        ascending=[False, False, False, False],
    )
    grouped.to_csv(summary_path, index=False)

    top_rows = grouped.head(12).to_dict(orient="records")
    with top_json_path.open("w", encoding="utf-8") as fh:
        json.dump(top_rows, fh, indent=2)

    print(f"Saved: {out_dir}")
    cols = [
        "combo_id",
        "robust_score",
        "alpha_mean",
        "alpha_min",
        "ml_incremental_alpha_mean",
        "ml_incremental_alpha_min",
        "ml_only_alpha_mean",
        "core_long_override",
        "ml_weight_override",
        "overlay_max_override",
        "smooth_alpha_override",
    ]
    print(grouped.head(12)[cols].to_string(index=False))


if __name__ == "__main__":
    main()
