"""Run a daily GBM research sweep across multiple variant definitions."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import sys
from typing import Iterable

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_backtest import BacktestConfig, UnifiedBacktester
from training.train_gbm import GBMTrainer


def _parse_csv_arg(raw: str) -> list[str]:
    return [part.strip() for part in str(raw or "").split(",") if part.strip()]


def _parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in str(raw or "").split(",") if part.strip()]


def _windows(now: pd.Timestamp, mode: str) -> list[tuple[str, str, str]]:
    end = now.normalize()
    if mode == "fast":
        return [
            ((end - pd.Timedelta(days=365 * 5)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "5y"),
            ((end - pd.Timedelta(days=365)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "1y"),
        ]
    return [
        ((end - pd.Timedelta(days=365 * 5)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "5y"),
        ((end - pd.Timedelta(days=365 * 2)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "2y"),
        ((end - pd.Timedelta(days=365)).strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), "1y"),
    ]


class _FastBacktester(UnifiedBacktester):
    def _save_outputs(self, *args, **kwargs):
        return


def _safe_float(value, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if pd.notna(out) else default
    except Exception:
        return default


def _variant_suffix(base: str, feature_profile: str, selection_mode: str, horizon: int) -> str:
    selection_short = "ranked" if selection_mode == "shap_ranked" else "diverse"
    parts = [base.strip(), feature_profile.strip(), f"h{int(horizon)}", selection_short]
    return "_".join([part for part in parts if part])


def _collect_train_row(symbol: str, metadata: dict, variant_name: str) -> dict:
    quality = metadata.get("quality_gate", {})
    metrics = quality.get("effective_metrics", {})
    holdout = metadata.get("holdout", {})
    return {
        "symbol": symbol,
        "variant": variant_name,
        "feature_profile": metadata.get("feature_profile", ""),
        "feature_selection_mode": metadata.get("feature_selection_mode", ""),
        "target_horizon_days": metadata.get("target_horizon_days", 1),
        "quality_gate_passed": bool(quality.get("passed", False)),
        "quality_score": _safe_float(quality.get("score", 0.0)),
        "quality_reasons": "; ".join(quality.get("reasons", [])),
        "holdout_source": metadata.get("effective_holdout_source", ""),
        "holdout_dir_acc": _safe_float(metrics.get("dir_acc", 0.0)),
        "holdout_ic": _safe_float(metrics.get("ic", 0.0)),
        "holdout_pred_std": _safe_float(metrics.get("pred_std", 0.0)),
        "holdout_pred_target_std_ratio": _safe_float(metrics.get("pred_target_std_ratio", 0.0)),
        "holdout_net_return": _safe_float(metrics.get("net_return", 0.0)),
        "holdout_net_sharpe": _safe_float(metrics.get("net_sharpe", 0.0)),
        "wfe": _safe_float(metadata.get("wfe", 0.0)),
        "n_features_selected": int(metadata.get("n_features_selected", 0)),
        "family_summary": json.dumps(metadata.get("selected_feature_families", {}), sort_keys=True),
        "xgb_wf_score": _safe_float(holdout.get("xgb", {}).get("pred_std", 0.0)),
    }


def _plot_suite(detail_df: pd.DataFrame, out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return

    if detail_df.empty:
        return

    plot_df = detail_df.copy()
    plot_df["label"] = (
        plot_df["symbol"].astype(str)
        + " "
        + plot_df["feature_profile"].astype(str)
        + " h"
        + plot_df["target_horizon_days"].astype(str)
        + " "
        + plot_df["feature_selection_mode"].astype(str).str.replace("shap_", "", regex=False)
    )
    score_col = "model_quality_score" if "model_quality_score" in plot_df.columns else "quality_score"
    plot_df["quality_score"] = pd.to_numeric(plot_df.get(score_col, 0.0), errors="coerce").fillna(0.0)
    plot_df["alpha"] = pd.to_numeric(plot_df["alpha"], errors="coerce").fillna(0.0)
    plot_df["ml_only_alpha"] = pd.to_numeric(plot_df["ml_only_alpha"], errors="coerce").fillna(0.0)
    plot_df["regime_only_alpha"] = pd.to_numeric(plot_df["regime_only_alpha"], errors="coerce").fillna(0.0)

    top = plot_df.sort_values(["ml_incremental_alpha_vs_regime", "alpha"], ascending=False).head(18)
    fig, ax = plt.subplots(figsize=(14, 8))
    scatter = ax.scatter(
        top["regime_only_alpha"],
        top["ml_only_alpha"],
        c=top["quality_score"],
        s=80,
        cmap="viridis",
        alpha=0.85,
    )
    for _, row in top.iterrows():
        ax.annotate(str(row["label"]), (row["regime_only_alpha"], row["ml_only_alpha"]), fontsize=8, alpha=0.85)
    ax.axhline(0.0, color="#94a3b8", linewidth=1.0, linestyle="--")
    ax.axvline(0.0, color="#94a3b8", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Regime-Only Alpha")
    ax.set_ylabel("ML-Only Alpha")
    ax.set_title("Signal Decomposition: ML vs Regime")
    fig.colorbar(scatter, label="Quality Score")
    fig.tight_layout()
    fig.savefig(out_dir / "signal_decomposition.png", dpi=160)
    plt.close(fig)

    heat = (
        plot_df.pivot_table(
            index="label",
            columns="window",
            values="ml_incremental_alpha_vs_regime",
            aggfunc="mean",
        )
        .fillna(0.0)
        .sort_index()
    )
    if not heat.empty:
        fig, ax = plt.subplots(figsize=(10, max(5, 0.35 * len(heat))))
        im = ax.imshow(heat.to_numpy(dtype=float), aspect="auto", cmap="RdYlGn")
        ax.set_xticks(range(len(heat.columns)))
        ax.set_xticklabels(list(heat.columns))
        ax.set_yticks(range(len(heat.index)))
        ax.set_yticklabels(list(heat.index), fontsize=8)
        ax.set_title("ML Incremental Alpha vs Regime")
        fig.colorbar(im, label="Alpha Lift")
        fig.tight_layout()
        fig.savefig(out_dir / "incremental_alpha_heatmap.png", dpi=160)
        plt.close(fig)


def run_suite(
    *,
    symbols: Iterable[str],
    feature_profiles: list[str],
    feature_selection_modes: list[str],
    target_horizons: list[int],
    n_trials: int,
    max_features: int,
    windows_mode: str,
    base_suffix: str,
    overwrite: bool,
    allow_cpu_fallback: bool,
    use_lgb: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    windows = _windows(now, windows_mode)

    detail_rows: list[dict] = []
    train_rows: list[dict] = []

    for symbol in symbols:
        for feature_profile, selection_mode, target_horizon in itertools.product(
            feature_profiles,
            feature_selection_modes,
            target_horizons,
        ):
            suffix = _variant_suffix(base_suffix, feature_profile, selection_mode, target_horizon)
            variant_name = f"gbm_{suffix}" if suffix else "gbm"
            trainer = GBMTrainer(
                symbol=symbol,
                n_trials=n_trials,
                overwrite=overwrite,
                max_features=max_features,
                allow_cpu_fallback=allow_cpu_fallback,
                target_horizon=target_horizon,
                feature_profile=feature_profile,
                feature_selection_mode=selection_mode,
                use_lgb=use_lgb,
                data_period="max",
                data_interval="1d",
                model_suffix=suffix,
            )
            metadata = trainer.train()
            train_rows.append(_collect_train_row(symbol, metadata, variant_name))

            for start, end, label in windows:
                cfg = BacktestConfig(
                    symbol=symbol,
                    start=start,
                    end=end,
                    model_variant=variant_name,
                    data_period="max",
                    data_interval="1d",
                    max_long=1.6,
                    max_short=0.6,
                    commission_pct=0.0005,
                    slippage_pct=0.0003,
                    warmup_days=252,
                    min_eval_days=15,
                )
                try:
                    summary = _FastBacktester(cfg).run()
                    summary["window"] = label
                    summary["feature_profile"] = feature_profile
                    summary["feature_selection_mode"] = selection_mode
                    summary["target_horizon_days"] = target_horizon
                    summary["variant"] = variant_name
                    detail_rows.append(summary)
                    print(
                        f"[OK] {symbol} {variant_name} {label} "
                        f"alpha={summary['alpha']:.4f} ml={summary['ml_only_alpha']:.4f} "
                        f"regime={summary['regime_only_alpha']:.4f} gate={summary['model_quality_gate_passed']}"
                    )
                except Exception as exc:
                    print(f"[FAIL] {symbol} {variant_name} {label}: {exc}")

    detail_df = pd.DataFrame(detail_rows)
    train_df = pd.DataFrame(train_rows)
    return detail_df, train_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a daily research suite across GBM variants")
    parser.add_argument("--symbols", type=str, default="AAPL,GOOGL")
    parser.add_argument("--feature-profiles", type=str, default="full,compact,trend")
    parser.add_argument("--feature-selection-modes", type=str, default="shap_diverse,shap_ranked")
    parser.add_argument("--target-horizons", type=str, default="1,3")
    parser.add_argument("--n-trials", type=int, default=1)
    parser.add_argument("--max-features", type=int, default=50)
    parser.add_argument("--windows-mode", type=str, default="fast", choices=["fast", "full"])
    parser.add_argument("--base-suffix", type=str, default="daily_suite")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--use-lgb", action="store_true")
    args = parser.parse_args()

    symbols = [s.upper() for s in _parse_csv_arg(args.symbols)]
    feature_profiles = _parse_csv_arg(args.feature_profiles)
    feature_selection_modes = _parse_csv_arg(args.feature_selection_modes)
    target_horizons = _parse_int_list(args.target_horizons)

    detail_df, train_df = run_suite(
        symbols=symbols,
        feature_profiles=feature_profiles,
        feature_selection_modes=feature_selection_modes,
        target_horizons=target_horizons,
        n_trials=args.n_trials,
        max_features=args.max_features,
        windows_mode=args.windows_mode,
        base_suffix=args.base_suffix,
        overwrite=args.overwrite,
        allow_cpu_fallback=args.allow_cpu_fallback,
        use_lgb=bool(args.use_lgb),
    )

    out_dir = ROOT / "experiments" / f"daily_research_suite_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True, exist_ok=True)

    detail_path = out_dir / "suite_detail.csv"
    train_path = out_dir / "suite_train.csv"
    summary_path = out_dir / "suite_summary.csv"
    detail_df.to_csv(detail_path, index=False)
    train_df.to_csv(train_path, index=False)

    summary_df = pd.DataFrame()
    if not detail_df.empty:
        grouped = detail_df.groupby(["symbol", "variant", "feature_profile", "feature_selection_mode", "target_horizon_days"], dropna=False)
        summary_df = grouped.agg(
            quality_score=("model_quality_score", "mean"),
            gate_pass_rate=("model_quality_gate_passed", "mean"),
            alpha_mean=("alpha", "mean"),
            alpha_min=("alpha", "min"),
            ml_only_alpha_mean=("ml_only_alpha", "mean"),
            regime_only_alpha_mean=("regime_only_alpha", "mean"),
            ml_incremental_alpha_vs_regime_mean=("ml_incremental_alpha_vs_regime", "mean"),
            sharpe_mean=("sharpe_ratio", "mean"),
            holdout_pred_target_std_ratio=("holdout_pred_target_std_ratio", "mean"),
        ).reset_index()
        summary_df = summary_df.sort_values(
            ["ml_incremental_alpha_vs_regime_mean", "alpha_mean", "quality_score"],
            ascending=False,
        )
        summary_df.to_csv(summary_path, index=False)
    else:
        summary_df.to_csv(summary_path, index=False)

    _plot_suite(detail_df, out_dir)

    print(f"Saved detail: {detail_path}")
    print(f"Saved train: {train_path}")
    print(f"Saved summary: {summary_path}")
    if not summary_df.empty:
        print(summary_df.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
