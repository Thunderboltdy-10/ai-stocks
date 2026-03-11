"""Run a daily GBM research matrix across horizons, feature profiles, and selectors.

The goal is not just to find the best full-strategy backtest, but to identify
variants whose ML component contributes incremental value beyond the regime/core
overlay. Outputs are saved under `experiments/`.
"""

from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.feature_engineer import available_feature_profiles
from run_backtest import BacktestConfig, UnifiedBacktester
from training.train_gbm import GBMTrainer


def _parse_csv(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def _parse_int_csv(raw: str) -> list[int]:
    return [int(part) for part in _parse_csv(raw)]


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


def _variant_rows(symbol: str, variant_configs: Iterable[dict], args) -> list[dict]:
    rows: list[dict] = []
    end = pd.Timestamp.now(tz="UTC").tz_localize(None)
    windows = _parse_windows(end, args.window_set)

    for config in variant_configs:
        suffix = config["model_suffix"]
        model_variant = f"gbm_{suffix}" if suffix else "gbm"
        trainer = GBMTrainer(
            symbol=symbol,
            n_trials=args.n_trials,
            overwrite=args.overwrite,
            max_features=args.max_features,
            allow_cpu_fallback=args.allow_cpu_fallback,
            target_horizon=config["target_horizon"],
            feature_profile=config["feature_profile"],
            feature_selection_mode=config["feature_selection_mode"],
            use_lgb=config["use_lgb"],
            data_period=args.period,
            data_interval=args.interval,
            model_suffix=suffix,
        )
        try:
            metadata = trainer.train()
        except Exception as exc:
            rows.append(
                {
                    "symbol": symbol,
                    "variant": model_variant,
                    "status": "train_failed",
                    "error": str(exc),
                    **config,
                }
            )
            continue

        holdout = metadata.get("quality_gate", {})
        selected_metrics = holdout.get("effective_metrics", {})
        for start, end_str, label in windows:
            cfg = BacktestConfig(
                symbol=symbol,
                start=start,
                end=end_str,
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
                rows.append(
                    {
                        "symbol": symbol,
                        "variant": model_variant,
                        "window": label,
                        "status": "ok",
                        "quality_gate_passed": bool(holdout.get("passed", False)),
                        "quality_score": float(holdout.get("score", 0.0)),
                        "quality_reasons": "|".join(holdout.get("reasons", [])),
                        "holdout_dir_acc": float(selected_metrics.get("dir_acc", 0.0)),
                        "holdout_pred_std": float(selected_metrics.get("pred_std", 0.0)),
                        "holdout_pred_target_std_ratio": float(selected_metrics.get("pred_target_std_ratio", 0.0)),
                        "holdout_net_return": float(selected_metrics.get("net_return", 0.0)),
                        "holdout_net_sharpe": float(selected_metrics.get("net_sharpe", 0.0)),
                        "holdout_wfe": float(selected_metrics.get("wfe", 0.0)),
                        "selected_feature_families": json.dumps(metadata.get("selected_feature_families", {}), sort_keys=True),
                        "n_features_selected": int(metadata.get("n_features_selected", 0)),
                        "masked_by_regime": bool(
                            summary.get("ml_incremental_alpha_vs_regime", 0.0) <= 0.01
                            and summary.get("alpha", 0.0) > 0.0
                        ),
                        **config,
                        **summary,
                    }
                )
            except Exception as exc:
                rows.append(
                    {
                        "symbol": symbol,
                        "variant": model_variant,
                        "window": label,
                        "status": "backtest_failed",
                        "error": str(exc),
                        **config,
                    }
                )

    return rows


def _save_outputs(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "research_matrix.csv"
    json_path = out_dir / "top_variants.json"
    df.to_csv(csv_path, index=False)

    good = df.loc[df["status"] == "ok"].copy()
    if good.empty:
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump([], fh, indent=2)
        return

    rank = good.sort_values(
        ["ml_incremental_alpha_vs_regime", "quality_score", "alpha"],
        ascending=[False, False, False],
    )
    top = rank.head(12).to_dict(orient="records")
    with json_path.open("w", encoding="utf-8") as fh:
        json.dump(top, fh, indent=2)

    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    chart = rank.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.barh(chart["variant"], chart["ml_incremental_alpha_vs_regime"], color="#f59e0b", alpha=0.85)
    ax.set_title("Top Variants by ML Incremental Alpha vs Regime")
    ax.set_xlabel("Incremental Alpha")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "leaderboard_incremental_alpha.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 7))
    masked = chart["masked_by_regime"].astype(bool)
    ax.scatter(
        chart.loc[~masked, "quality_score"],
        chart.loc[~masked, "ml_incremental_alpha_vs_regime"],
        color="#22c55e",
        label="Not Masked",
        alpha=0.85,
    )
    if masked.any():
        ax.scatter(
            chart.loc[masked, "quality_score"],
            chart.loc[masked, "ml_incremental_alpha_vs_regime"],
            color="#ef4444",
            label="Masked by Regime",
            alpha=0.85,
        )
    ax.axhline(0.0, color="#64748b", linewidth=1.0)
    ax.set_title("Quality Score vs ML Incremental Alpha")
    ax.set_xlabel("Quality Score")
    ax.set_ylabel("Incremental Alpha vs Regime")
    ax.legend(loc="best")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_dir / "quality_vs_incremental_alpha.png", dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a GBM research matrix")
    parser.add_argument("--symbols", type=str, default="AAPL")
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--max-features", type=int, default=50)
    parser.add_argument("--period", type=str, default="max")
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument("--window-set", type=str, default="fast", choices=["fast", "medium", "full"])
    parser.add_argument("--horizons", type=str, default="1,3,5")
    parser.add_argument("--feature-profiles", type=str, default="full,compact")
    parser.add_argument("--feature-selection-modes", type=str, default="shap_diverse,shap_ranked")
    parser.add_argument("--use-lgb-options", type=str, default="false")
    parser.add_argument("--base-suffix", type=str, default="research_v7")
    parser.add_argument("--max-long", type=float, default=1.6)
    parser.add_argument("--max-short", type=float, default=0.6)
    parser.add_argument("--commission", type=float, default=0.0005)
    parser.add_argument("--slippage", type=float, default=0.0003)
    parser.add_argument("--warmup-days", type=int, default=252)
    parser.add_argument("--min-eval-days", type=int, default=20)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    args = parser.parse_args()

    feature_profiles = _parse_csv(args.feature_profiles)
    valid_profiles = set(available_feature_profiles())
    invalid = [profile for profile in feature_profiles if profile not in valid_profiles]
    if invalid:
        raise ValueError(f"Unknown feature profiles: {invalid}. Valid: {sorted(valid_profiles)}")

    variants: list[dict] = []
    for target_horizon, feature_profile, selection_mode, use_lgb_raw in product(
        _parse_int_csv(args.horizons),
        feature_profiles,
        _parse_csv(args.feature_selection_modes),
        _parse_csv(args.use_lgb_options),
    ):
        use_lgb = str(use_lgb_raw).strip().lower() in {"1", "true", "yes", "y"}
        suffix = f"{args.base_suffix}_h{target_horizon}_{feature_profile}_{selection_mode}_{'ens' if use_lgb else 'xgb'}"
        variants.append(
            {
                "target_horizon": target_horizon,
                "feature_profile": feature_profile,
                "feature_selection_mode": selection_mode,
                "use_lgb": use_lgb,
                "model_suffix": suffix,
            }
        )

    all_rows: list[dict] = []
    for symbol in _parse_csv(args.symbols):
        print(f"[RESEARCH] {symbol} | variants={len(variants)}")
        rows = _variant_rows(symbol, variants, args)
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_dir = ROOT / "experiments" / f"gbm_research_{ts}"
    _save_outputs(df, out_dir)

    print(f"Saved: {out_dir}")
    if not df.empty:
        ok = df.loc[df["status"] == "ok"].copy()
        if not ok.empty:
            cols = [
                "symbol",
                "variant",
                "window",
                "quality_gate_passed",
                "quality_score",
                "alpha",
                "regime_only_alpha",
                "ml_only_alpha",
                "ml_incremental_alpha_vs_regime",
                "holdout_pred_target_std_ratio",
                "masked_by_regime",
            ]
            print(ok[cols].sort_values(["symbol", "ml_incremental_alpha_vs_regime"], ascending=[True, False]).to_string(index=False))


if __name__ == "__main__":
    main()
