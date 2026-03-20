"""Run diversified daily/intraday benchmark sweeps and persist aggregate summaries."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Dict, Iterable, Tuple

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.benchmark_lab import summarize_mode, write_benchmark_run
from run_backtest import BacktestConfig, UnifiedBacktester
from scripts.run_daily_research_suite import run_suite as run_daily_suite
from training.train_gbm import GBMTrainer
from utils.symbol_universe import (
    DEFAULT_DAILY_SET,
    DEFAULT_INTRADAY_SET,
    available_symbol_sets,
    resolve_symbols,
)


def _parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def _parse_ints(raw: str) -> list[int]:
    return [int(part.strip()) for part in str(raw or "").split(",") if part.strip()]


def _intraday_windows(now: pd.Timestamp) -> list[tuple[str, str, str]]:
    end = now.floor("h")
    return [
        ((end - pd.Timedelta(days=365)).strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S"), "intraday_1y"),
        ((end - pd.Timedelta(days=180)).strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S"), "intraday_6m"),
        ((end - pd.Timedelta(days=90)).strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S"), "intraday_3m"),
        ((end - pd.Timedelta(days=30)).strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S"), "intraday_1m"),
        ((end - pd.Timedelta(days=14)).strftime("%Y-%m-%d %H:%M:%S"), end.strftime("%Y-%m-%d %H:%M:%S"), "intraday_2w"),
    ]


def _safe_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if pd.notna(out) else default
    except Exception:
        return default


def _collect_train_row(symbol: str, metadata: dict, variant_name: str) -> dict:
    quality = metadata.get("quality_gate", {})
    metrics = quality.get("effective_metrics", {})
    return {
        "symbol": symbol,
        "variant": variant_name,
        "feature_profile": metadata.get("feature_profile", ""),
        "feature_selection_mode": metadata.get("feature_selection_mode", ""),
        "target_horizon_days": metadata.get("target_horizon_days", 1),
        "quality_gate_passed": bool(quality.get("passed", False)),
        "quality_score": _safe_float(quality.get("score", 0.0)),
        "holdout_net_return": _safe_float(metrics.get("net_return", 0.0)),
        "holdout_net_sharpe": _safe_float(metrics.get("net_sharpe", 0.0)),
        "wfe": _safe_float(metadata.get("wfe", 0.0)),
    }


class _FastBacktester(UnifiedBacktester):
    def _save_outputs(self, *args, **kwargs):
        return


def run_intraday_suite(
    *,
    symbols: Iterable[str],
    n_trials: int,
    max_features: int,
    period: str,
    interval: str,
    model_suffix: str,
    max_long: float,
    max_short: float,
    commission: float,
    slippage: float,
    target_horizon: int,
    warmup_days: int,
    min_eval_days: int,
    overwrite: bool,
    allow_cpu_fallback: bool,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    windows = _intraday_windows(now)
    rows: list[dict] = []
    train_rows: list[dict] = []
    variant_name = f"gbm_{model_suffix}" if model_suffix else f"gbm_{interval}_{period}"

    for symbol in symbols:
        trainer = GBMTrainer(
            symbol=symbol,
            n_trials=n_trials,
            overwrite=overwrite,
            max_features=max_features,
            use_lgb=False,
            allow_cpu_fallback=allow_cpu_fallback,
            data_period=period,
            data_interval=interval,
            model_suffix=model_suffix,
            target_horizon=target_horizon,
        )
        metadata = trainer.train()
        train_rows.append(_collect_train_row(symbol, metadata, variant_name))

        for start, end, label in windows:
            cfg = BacktestConfig(
                symbol=symbol,
                start=start,
                end=end,
                model_variant=variant_name,
                data_period=period,
                data_interval=interval,
                max_long=max_long,
                max_short=max_short,
                commission_pct=commission,
                slippage_pct=slippage,
                warmup_days=warmup_days,
                min_eval_days=min_eval_days,
            )
            try:
                summary = _FastBacktester(cfg).run()
            except Exception as exc:
                print(f"[INTRADAY-FAIL] {symbol} {label}: {exc}")
                continue
            summary["window"] = label
            summary["variant"] = variant_name
            rows.append(summary)
            print(
                f"[INTRADAY] {symbol} {label} alpha={summary['alpha']:.4f} "
                f"sharpe={summary['sharpe_ratio']:.3f}"
            )

    return pd.DataFrame(rows), pd.DataFrame(train_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified benchmark lab")
    parser.add_argument("--modes", type=str, default="daily,intraday", help="Comma-separated: daily,intraday")
    parser.add_argument("--daily-symbol-set", type=str, default=DEFAULT_DAILY_SET, choices=available_symbol_sets())
    parser.add_argument("--intraday-symbol-set", type=str, default=DEFAULT_INTRADAY_SET, choices=available_symbol_sets())
    parser.add_argument("--daily-symbols", type=str, default="")
    parser.add_argument("--intraday-symbols", type=str, default="")
    parser.add_argument("--daily-n-trials", type=int, default=8)
    parser.add_argument("--intraday-n-trials", type=int, default=8)
    parser.add_argument("--daily-max-features", type=int, default=50)
    parser.add_argument("--intraday-max-features", type=int, default=50)
    parser.add_argument("--daily-feature-profiles", type=str, default="full,compact,trend")
    parser.add_argument("--daily-selection-modes", type=str, default="shap_diverse,shap_ranked")
    parser.add_argument("--daily-target-horizons", type=str, default="1,3,5")
    parser.add_argument("--daily-base-suffix", type=str, default="daily_v5")
    parser.add_argument("--daily-windows-mode", type=str, default="full", choices=["fast", "full"])
    parser.add_argument("--intraday-model-suffix", type=str, default="intraday_1h_v4")
    parser.add_argument("--intraday-period", type=str, default="730d")
    parser.add_argument("--intraday-interval", type=str, default="1h")
    parser.add_argument("--intraday-target-horizon", type=int, default=1)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--use-lgb", action="store_true")
    args = parser.parse_args()

    modes = {part.strip().lower() for part in str(args.modes or "").split(",") if part.strip()}
    mode_payloads: Dict[str, Dict] = {}

    if "daily" in modes:
        daily_symbols = resolve_symbols(
            args.daily_symbols,
            symbol_set=args.daily_symbol_set,
            fallback_set=args.daily_symbol_set,
        )
        daily_detail, daily_train = run_daily_suite(
            symbols=daily_symbols,
            feature_profiles=_parse_csv(args.daily_feature_profiles),
            feature_selection_modes=_parse_csv(args.daily_selection_modes),
            target_horizons=_parse_ints(args.daily_target_horizons),
            n_trials=args.daily_n_trials,
            max_features=args.daily_max_features,
            windows_mode=args.daily_windows_mode,
            base_suffix=args.daily_base_suffix,
            overwrite=args.overwrite,
            allow_cpu_fallback=args.allow_cpu_fallback,
            use_lgb=args.use_lgb,
        )
        mode_payloads["daily"] = summarize_mode(
            mode="daily",
            detail_df=daily_detail,
            train_df=daily_train,
            symbol_set=args.daily_symbol_set,
            settings={
                "nTrials": args.daily_n_trials,
                "maxFeatures": args.daily_max_features,
                "featureProfiles": _parse_csv(args.daily_feature_profiles),
                "selectionModes": _parse_csv(args.daily_selection_modes),
                "targetHorizons": _parse_ints(args.daily_target_horizons),
                "baseSuffix": args.daily_base_suffix,
                "windowsMode": args.daily_windows_mode,
                "symbols": daily_symbols,
            },
        )

    if "intraday" in modes:
        intraday_symbols = resolve_symbols(
            args.intraday_symbols,
            symbol_set=args.intraday_symbol_set,
            fallback_set=args.intraday_symbol_set,
        )
        intraday_detail, intraday_train = run_intraday_suite(
            symbols=intraday_symbols,
            n_trials=args.intraday_n_trials,
            max_features=args.intraday_max_features,
            period=args.intraday_period,
            interval=args.intraday_interval,
            model_suffix=args.intraday_model_suffix,
            max_long=1.4,
            max_short=0.2,
            commission=0.0001,
            slippage=0.0001,
            target_horizon=args.intraday_target_horizon,
            warmup_days=40,
            min_eval_days=15,
            overwrite=args.overwrite,
            allow_cpu_fallback=args.allow_cpu_fallback,
        )
        mode_payloads["intraday"] = summarize_mode(
            mode="intraday",
            detail_df=intraday_detail,
            train_df=intraday_train,
            symbol_set=args.intraday_symbol_set,
            settings={
                "nTrials": args.intraday_n_trials,
                "maxFeatures": args.intraday_max_features,
                "targetHorizon": args.intraday_target_horizon,
                "modelSuffix": args.intraday_model_suffix,
                "period": args.intraday_period,
                "interval": args.intraday_interval,
                "symbols": intraday_symbols,
            },
        )

    payload = write_benchmark_run(
        modes=mode_payloads,
        settings={
            "modes": sorted(mode_payloads.keys()),
            "overwrite": bool(args.overwrite),
            "allowCpuFallback": bool(args.allow_cpu_fallback),
            "useLgb": bool(args.use_lgb),
        },
    )
    print(f"Saved benchmark run: {payload['id']}")


if __name__ == "__main__":
    main()
