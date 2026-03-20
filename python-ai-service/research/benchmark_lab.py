"""Benchmark run aggregation and storage helpers."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


BENCHMARK_RUNS_DIR = Path(__file__).resolve().parents[1] / "benchmark_runs"


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        return out if pd.notna(out) else default
    except Exception:
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _to_records(df: pd.DataFrame, limit: int) -> List[Dict[str, Any]]:
    if df.empty:
        return []
    clean = df.head(limit).copy()
    clean = clean.where(pd.notna(clean), None)
    return json.loads(clean.to_json(orient="records"))


def summarize_mode(
    *,
    mode: str,
    detail_df: pd.DataFrame,
    train_df: pd.DataFrame | None,
    symbol_set: str,
    settings: Dict[str, Any],
) -> Dict[str, Any]:
    detail_df = detail_df.copy() if detail_df is not None else pd.DataFrame()
    train_df = train_df.copy() if train_df is not None else pd.DataFrame()

    if detail_df.empty:
        return {
            "mode": mode,
            "symbolSet": symbol_set,
            "settings": settings,
            "symbolsTested": 0,
            "variantsEvaluated": 0,
            "windowsEvaluated": 0,
            "aggregate": {
                "meanAlpha": 0.0,
                "medianAlpha": 0.0,
                "positiveAlphaRate": 0.0,
                "meanSharpe": 0.0,
                "medianSharpe": 0.0,
                "meanStrategyReturn": 0.0,
                "meanBuyHoldReturn": 0.0,
                "meanMlLiftVsRegime": 0.0,
                "qualityPassRate": 0.0,
            },
            "leaders": [],
            "laggards": [],
            "qualityLeaders": [],
        }

    for col in (
        "alpha",
        "sharpe_ratio",
        "strategy_return",
        "buy_hold_return",
        "ml_incremental_alpha_vs_regime",
        "model_quality_score",
    ):
        if col in detail_df.columns:
            detail_df[col] = pd.to_numeric(detail_df[col], errors="coerce").fillna(0.0)

    if "model_quality_gate_passed" in detail_df.columns:
        detail_df["model_quality_gate_passed"] = detail_df["model_quality_gate_passed"].astype(bool)
    else:
        detail_df["model_quality_gate_passed"] = False

    by_symbol = (
        detail_df.groupby("symbol", dropna=False)
        .agg(
            variant=("variant", lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0]),
            mean_alpha=("alpha", "mean"),
            median_alpha=("alpha", "median"),
            mean_sharpe=("sharpe_ratio", "mean"),
            mean_strategy_return=("strategy_return", "mean"),
            mean_buy_hold_return=("buy_hold_return", "mean"),
            mean_ml_lift=("ml_incremental_alpha_vs_regime", "mean"),
            positive_alpha_rate=("alpha", lambda s: float((pd.Series(s) > 0.0).mean())),
            windows=("window", "nunique"),
            quality_score=("model_quality_score", "mean"),
            quality_pass_rate=("model_quality_gate_passed", "mean"),
        )
        .reset_index()
        .sort_values(["mean_alpha", "mean_sharpe"], ascending=False)
    )

    quality_df = pd.DataFrame()
    if not train_df.empty:
        for col in ("quality_score", "holdout_net_return", "holdout_net_sharpe", "wfe"):
            if col in train_df.columns:
                train_df[col] = pd.to_numeric(train_df[col], errors="coerce").fillna(0.0)
        if "quality_gate_passed" in train_df.columns:
            train_df["quality_gate_passed"] = train_df["quality_gate_passed"].astype(bool)

        quality_df = (
            train_df.sort_values(
                ["quality_gate_passed", "quality_score", "holdout_net_return"],
                ascending=[False, False, False],
            )
            .loc[
                :,
                [
                    c
                    for c in (
                        "symbol",
                        "variant",
                        "feature_profile",
                        "feature_selection_mode",
                        "target_horizon_days",
                        "quality_gate_passed",
                        "quality_score",
                        "holdout_net_return",
                        "holdout_net_sharpe",
                        "wfe",
                    )
                    if c in train_df.columns
                ],
            ]
        )

    return {
        "mode": mode,
        "symbolSet": symbol_set,
        "settings": settings,
        "symbolsTested": int(detail_df["symbol"].nunique()),
        "variantsEvaluated": int(detail_df["variant"].nunique()) if "variant" in detail_df.columns else 0,
        "windowsEvaluated": int(detail_df["window"].nunique()) if "window" in detail_df.columns else 0,
        "aggregate": {
            "meanAlpha": _safe_float(detail_df["alpha"].mean()),
            "medianAlpha": _safe_float(detail_df["alpha"].median()),
            "positiveAlphaRate": _safe_float((detail_df["alpha"] > 0.0).mean()),
            "meanSharpe": _safe_float(detail_df["sharpe_ratio"].mean()) if "sharpe_ratio" in detail_df.columns else 0.0,
            "medianSharpe": _safe_float(detail_df["sharpe_ratio"].median()) if "sharpe_ratio" in detail_df.columns else 0.0,
            "meanStrategyReturn": _safe_float(detail_df["strategy_return"].mean()) if "strategy_return" in detail_df.columns else 0.0,
            "meanBuyHoldReturn": _safe_float(detail_df["buy_hold_return"].mean()) if "buy_hold_return" in detail_df.columns else 0.0,
            "meanMlLiftVsRegime": _safe_float(detail_df["ml_incremental_alpha_vs_regime"].mean()) if "ml_incremental_alpha_vs_regime" in detail_df.columns else 0.0,
            "qualityPassRate": _safe_float(detail_df["model_quality_gate_passed"].mean()),
        },
        "leaders": _to_records(by_symbol, limit=6),
        "laggards": _to_records(by_symbol.sort_values(["mean_alpha", "mean_sharpe"], ascending=True), limit=6),
        "qualityLeaders": _to_records(quality_df, limit=6),
    }


def write_benchmark_run(
    *,
    modes: Dict[str, Dict[str, Any]],
    settings: Dict[str, Any],
    output_dir: Path | None = None,
) -> Dict[str, Any]:
    base_dir = output_dir or BENCHMARK_RUNS_DIR
    base_dir.mkdir(parents=True, exist_ok=True)

    ts = datetime.now(timezone.utc)
    run_id = ts.strftime("%Y%m%d_%H%M%S")
    run_dir = base_dir / run_id
    suffix = 1
    while run_dir.exists():
        suffix += 1
        run_id = f"{ts.strftime('%Y%m%d_%H%M%S')}_{suffix}"
        run_dir = base_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "id": run_id,
        "createdAt": ts.isoformat(),
        "settings": settings,
        "modes": modes,
    }

    with (run_dir / "summary.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    with (base_dir / "latest.json").open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    return payload


def load_benchmark_run(run_id: str, *, output_dir: Path | None = None) -> Dict[str, Any] | None:
    base_dir = output_dir or BENCHMARK_RUNS_DIR
    summary_path = base_dir / run_id / "summary.json"
    if not summary_path.exists():
        return None
    try:
        with summary_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return None


def list_benchmark_runs(*, limit: int = 5, output_dir: Path | None = None) -> List[Dict[str, Any]]:
    base_dir = output_dir or BENCHMARK_RUNS_DIR
    if not base_dir.exists():
        return []

    runs: List[Dict[str, Any]] = []
    for summary_path in sorted(base_dir.glob("*/summary.json"), reverse=True):
        try:
            with summary_path.open("r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except Exception:
            continue

        runs.append(
            {
                "id": str(payload.get("id", summary_path.parent.name)),
                "createdAt": str(payload.get("createdAt", "")),
                "settings": payload.get("settings", {}),
                "modes": payload.get("modes", {}),
            }
        )
        if len(runs) >= max(1, int(limit)):
            break

    return runs
