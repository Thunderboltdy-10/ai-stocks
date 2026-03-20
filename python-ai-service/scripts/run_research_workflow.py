"""Top-level research workflow orchestrator for daily/intraday GBM loops.

This script coordinates the safer, reproducible path for the project:
1. Train/evaluate across diversified symbol sets.
2. Build per-symbol auto-routing registries.
3. Run matrix backtests on auto-routed variants.
4. Enforce a core-vs-holdout generalization gate.

It emits structured progress events on stdout with the `@@EVENT@@` prefix so the
API training service can stream meaningful updates to the frontend.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from utils.symbol_universe import resolve_symbols


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"
EVENT_PREFIX = "@@EVENT@@"


def emit(event_type: str, **payload) -> None:
    data = {"type": event_type, **payload}
    print(f"{EVENT_PREFIX}{json.dumps(data, sort_keys=True)}", flush=True)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _csv(items: Iterable[object]) -> str:
    return ",".join(str(item) for item in items if str(item).strip())


def _latest_paths(pattern: str, *, since_ts: float) -> list[Path]:
    matches = [path for path in EXPERIMENTS_DIR.glob(pattern) if path.stat().st_mtime >= since_ts]
    return sorted(matches, key=lambda path: path.stat().st_mtime, reverse=True)


def _read_json(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _append_flag(cmd: list[str], enabled: bool, flag: str) -> None:
    if enabled:
        cmd.append(flag)


def _run_command(*, step: str, progress: float, cmd: list[str]) -> list[str]:
    emit("stage", step=step, progress=progress, message="starting", command=" ".join(cmd))
    proc = subprocess.Popen(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output_lines: list[str] = []
    assert proc.stdout is not None
    for raw_line in proc.stdout:
        line = raw_line.rstrip()
        if not line:
            continue
        output_lines.append(line)
        print(line, flush=True)
    return_code = proc.wait()
    if return_code != 0:
        raise RuntimeError(f"{step} failed with exit code {return_code}")
    emit("stage", step=step, progress=progress, message="completed")
    return output_lines[-50:]


@dataclass
class StageResult:
    step: str
    status: str
    artifacts: list[str]
    tail: list[str]
    summary: dict


def _record_stage(
    *,
    results: list[StageResult],
    step: str,
    artifact_patterns: list[str],
    since_ts: float,
    tail: list[str],
    summary: dict | None = None,
) -> None:
    artifacts: list[str] = []
    for pattern in artifact_patterns:
        artifacts.extend([str(path) for path in _latest_paths(pattern, since_ts=since_ts)])
    results.append(
        StageResult(
            step=step,
            status="completed",
            artifacts=artifacts[:8],
            tail=tail[-10:],
            summary=summary or {},
        )
    )


def run_daily_workflow(args, *, stage_offset: int, stage_count: int, results: list[StageResult]) -> dict:
    resolved_daily_symbols = resolve_symbols(
        args.daily_symbols,
        symbol_set=args.daily_symbol_set or "daily15",
        fallback_set="daily15",
    )
    symbols_arg = _csv(resolved_daily_symbols)
    symbol_set = args.daily_symbol_set or "daily15"
    feature_profiles = _csv(args.feature_profiles or ["full", "compact", "trend"])
    selection_modes = _csv(args.feature_selection_modes or ["shap_diverse", "shap_ranked"])
    target_horizons = _csv(args.target_horizons or [1, 3])
    before = datetime.now().timestamp()
    cmd = [
        sys.executable,
        "-u",
        "scripts/run_daily_research_suite.py",
        "--symbols",
        symbols_arg,
        "--feature-profiles",
        feature_profiles,
        "--feature-selection-modes",
        selection_modes,
        "--target-horizons",
        target_horizons,
        "--n-trials",
        str(args.n_trials),
        "--max-features",
        str(args.max_features),
        "--windows-mode",
        args.windows_mode,
        "--base-suffix",
        args.base_suffix_daily,
    ]
    _append_flag(cmd, args.overwrite, "--overwrite")
    _append_flag(cmd, args.allow_cpu_fallback, "--allow-cpu-fallback")
    _append_flag(cmd, args.use_lgb, "--use-lgb")
    tail = _run_command(
        step="daily_research_suite",
        progress=(stage_offset + 1) / stage_count,
        cmd=cmd,
    )
    _record_stage(
        results=results,
        step="daily_research_suite",
        artifact_patterns=["daily_research_suite_*"],
        since_ts=before,
        tail=tail,
    )

    before = datetime.now().timestamp()
    tail = _run_command(
        step="daily_variant_registry",
        progress=(stage_offset + 2) / stage_count,
        cmd=[
            sys.executable,
            "-u",
            "scripts/build_daily_variant_registry.py",
            "--symbols",
            symbols_arg,
            "--symbol-set",
            symbol_set,
            "--period",
            args.daily_period,
            "--interval",
            args.daily_interval,
        ],
    )
    _record_stage(
        results=results,
        step="daily_variant_registry",
        artifact_patterns=["daily_variant_registry_matrix_*.csv"],
        since_ts=before,
        tail=tail,
    )

    before = datetime.now().timestamp()
    tail = _run_command(
        step="daily_auto_matrix",
        progress=(stage_offset + 3) / stage_count,
        cmd=[
            sys.executable,
            "-u",
            "scripts/run_daily_auto_matrix.py",
            "--symbols",
            symbols_arg,
            "--symbol-set",
            symbol_set,
            "--period",
            args.daily_period,
            "--interval",
            args.daily_interval,
        ],
    )
    _record_stage(
        results=results,
        step="daily_auto_matrix",
        artifact_patterns=["daily_auto_matrix_*.csv"],
        since_ts=before,
        tail=tail,
    )

    before = datetime.now().timestamp()
    tail = _run_command(
        step="daily_generalization_gate",
        progress=(stage_offset + 4) / stage_count,
        cmd=[
            sys.executable,
            "-u",
            "scripts/run_generalization_gate.py",
            "--core-symbol-set",
            args.daily_core_symbol_set,
            "--holdout-symbol-set",
            args.daily_holdout_symbol_set,
            "--interval",
            args.daily_interval,
            "--period",
            args.daily_period,
        ],
    )
    gate_files = _latest_paths("generalization_gate_daily_*.json", since_ts=before)
    gate_summary = _read_json(gate_files[0]) if gate_files else {}
    _record_stage(
        results=results,
        step="daily_generalization_gate",
        artifact_patterns=["generalization_gate_daily_*.json", "generalization_matrix_daily_*.csv"],
        since_ts=before,
        tail=tail,
        summary={
            "gate_passed": bool(gate_summary.get("gate_passed", False)),
            "core_mean_alpha": float(gate_summary.get("core_metrics", {}).get("mean_alpha", 0.0)),
            "holdout_mean_alpha": float(gate_summary.get("holdout_metrics", {}).get("mean_alpha", 0.0)),
            "generalization_gap": float(gate_summary.get("generalization_gap", 0.0)),
        },
    )
    return gate_summary


def run_intraday_workflow(args, *, stage_offset: int, stage_count: int, results: list[StageResult]) -> dict:
    resolved_intraday_symbols = resolve_symbols(
        args.intraday_symbols,
        symbol_set=args.intraday_symbol_set or "intraday10",
        fallback_set="intraday10",
    )
    symbols_arg = _csv(resolved_intraday_symbols)
    symbol_set = args.intraday_symbol_set or "intraday10"
    gate_summary: dict = {}

    horizons = list(args.target_horizons or [1, 3])
    for idx, horizon in enumerate(horizons, start=1):
        before = datetime.now().timestamp()
        cmd = [
            sys.executable,
            "-u",
            "scripts/run_intraday_hourly_matrix.py",
            "--symbols",
            symbols_arg,
            "--symbol-set",
            symbol_set,
            "--n-trials",
            str(args.n_trials),
            "--max-features",
            str(args.max_features),
            "--period",
            args.intraday_period,
            "--interval",
            args.intraday_interval,
            "--model-suffix",
            f"{args.base_suffix_intraday}_h{int(horizon)}",
            "--target-horizon",
            str(int(horizon)),
        ]
        _append_flag(cmd, args.overwrite, "--overwrite")
        _append_flag(cmd, args.allow_cpu_fallback, "--allow-cpu-fallback")
        tail = _run_command(
            step=f"intraday_hourly_matrix_h{horizon}",
            progress=(stage_offset + idx) / stage_count,
            cmd=cmd,
        )
        _record_stage(
            results=results,
            step=f"intraday_hourly_matrix_h{horizon}",
            artifact_patterns=["intraday_hourly_matrix_*.csv"],
            since_ts=before,
            tail=tail,
        )

    before = datetime.now().timestamp()
    tail = _run_command(
        step="intraday_variant_registry",
        progress=(stage_offset + len(horizons) + 1) / stage_count,
        cmd=[
            sys.executable,
            "-u",
            "scripts/build_intraday_variant_registry.py",
            "--symbols",
            symbols_arg,
            "--symbol-set",
            symbol_set,
            "--period",
            args.intraday_period,
            "--interval",
            args.intraday_interval,
        ],
    )
    _record_stage(
        results=results,
        step="intraday_variant_registry",
        artifact_patterns=["intraday_variant_registry_matrix_*.csv"],
        since_ts=before,
        tail=tail,
    )

    before = datetime.now().timestamp()
    tail = _run_command(
        step="intraday_auto_matrix",
        progress=(stage_offset + len(horizons) + 2) / stage_count,
        cmd=[
            sys.executable,
            "-u",
            "scripts/run_intraday_auto_matrix.py",
            "--symbols",
            symbols_arg,
            "--symbol-set",
            symbol_set,
            "--period",
            args.intraday_period,
            "--interval",
            args.intraday_interval,
        ],
    )
    _record_stage(
        results=results,
        step="intraday_auto_matrix",
        artifact_patterns=["intraday_auto_matrix_*.csv"],
        since_ts=before,
        tail=tail,
    )

    before = datetime.now().timestamp()
    tail = _run_command(
        step="intraday_generalization_gate",
        progress=(stage_offset + len(horizons) + 3) / stage_count,
        cmd=[
            sys.executable,
            "-u",
            "scripts/run_generalization_gate.py",
            "--core-symbol-set",
            args.intraday_core_symbol_set,
            "--holdout-symbol-set",
            args.intraday_holdout_symbol_set,
            "--interval",
            args.intraday_interval,
            "--period",
            args.intraday_period,
        ],
    )
    gate_files = _latest_paths("generalization_gate_intraday_*.json", since_ts=before)
    gate_summary = _read_json(gate_files[0]) if gate_files else {}
    _record_stage(
        results=results,
        step="intraday_generalization_gate",
        artifact_patterns=["generalization_gate_intraday_*.json", "generalization_matrix_intraday_*.csv"],
        since_ts=before,
        tail=tail,
        summary={
            "gate_passed": bool(gate_summary.get("gate_passed", False)),
            "core_mean_alpha": float(gate_summary.get("core_metrics", {}).get("mean_alpha", 0.0)),
            "holdout_mean_alpha": float(gate_summary.get("holdout_metrics", {}).get("mean_alpha", 0.0)),
            "generalization_gap": float(gate_summary.get("generalization_gap", 0.0)),
        },
    )
    return gate_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the top-level GBM research workflow")
    parser.add_argument("--workflow", choices=["daily", "intraday", "full"], default="full")
    parser.add_argument("--n-trials", type=int, default=8)
    parser.add_argument("--max-features", type=int, default=50)
    parser.add_argument("--target-horizons", type=int, nargs="+", default=[1, 3])
    parser.add_argument("--feature-profiles", nargs="+", default=["full", "compact", "trend"])
    parser.add_argument("--feature-selection-modes", nargs="+", default=["shap_diverse", "shap_ranked"])
    parser.add_argument("--windows-mode", choices=["fast", "full"], default="full")
    parser.add_argument("--use-lgb", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--allow-cpu-fallback", action="store_true")
    parser.add_argument("--daily-symbols", default="")
    parser.add_argument("--daily-symbol-set", default="daily15")
    parser.add_argument("--daily-core-symbol-set", default="core10")
    parser.add_argument("--daily-holdout-symbol-set", default="holdout5_ext")
    parser.add_argument("--daily-period", default="max")
    parser.add_argument("--daily-interval", default="1d")
    parser.add_argument("--base-suffix-daily", default="workflow_daily")
    parser.add_argument("--intraday-symbols", default="")
    parser.add_argument("--intraday-symbol-set", default="intraday10")
    parser.add_argument("--intraday-core-symbol-set", default="core5")
    parser.add_argument("--intraday-holdout-symbol-set", default="holdout5")
    parser.add_argument("--intraday-period", default="730d")
    parser.add_argument("--intraday-interval", default="1h")
    parser.add_argument("--base-suffix-intraday", default="intraday_1h_workflow")
    args = parser.parse_args()

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    run_dir = EXPERIMENTS_DIR / f"research_workflow_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)

    requested = {
        "workflow": args.workflow,
        "n_trials": args.n_trials,
        "max_features": args.max_features,
        "target_horizons": list(args.target_horizons),
        "feature_profiles": list(args.feature_profiles),
        "feature_selection_modes": list(args.feature_selection_modes),
        "windows_mode": args.windows_mode,
        "use_lgb": bool(args.use_lgb),
        "overwrite": bool(args.overwrite),
    }
    emit("started", workflow=args.workflow, message="research workflow started")

    stage_results: list[StageResult] = []
    summary = {
        "id": run_dir.name,
        "workflow": args.workflow,
        "requested": requested,
        "started_at": _utc_now(),
        "run_dir": str(run_dir),
        "daily": {},
        "intraday": {},
        "stages": [],
    }

    try:
        total_stage_count = 0
        if args.workflow in {"daily", "full"}:
            total_stage_count += 4
        if args.workflow in {"intraday", "full"}:
            total_stage_count += len(args.target_horizons) + 3

        stage_offset = 0
        if args.workflow in {"daily", "full"}:
            summary["daily"] = run_daily_workflow(
                args,
                stage_offset=stage_offset,
                stage_count=total_stage_count,
                results=stage_results,
            )
            stage_offset += 4
        if args.workflow in {"intraday", "full"}:
            summary["intraday"] = run_intraday_workflow(
                args,
                stage_offset=stage_offset,
                stage_count=total_stage_count,
                results=stage_results,
            )

        summary["stages"] = [asdict(stage) for stage in stage_results]
        summary["completed_at"] = _utc_now()
        summary["status"] = "completed"
        summary["daily_gate_passed"] = bool(summary.get("daily", {}).get("gate_passed", False))
        summary["intraday_gate_passed"] = bool(summary.get("intraday", {}).get("gate_passed", False))
        with (run_dir / "summary.json").open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        emit(
            "completed",
            workflow=args.workflow,
            progress=1.0,
            summary_path=str(run_dir / "summary.json"),
            daily_gate_passed=summary["daily_gate_passed"],
            intraday_gate_passed=summary["intraday_gate_passed"],
        )
    except Exception as exc:
        summary["stages"] = [asdict(stage) for stage in stage_results]
        summary["completed_at"] = _utc_now()
        summary["status"] = "failed"
        summary["error"] = str(exc)
        with (run_dir / "summary.json").open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        emit("failed", workflow=args.workflow, error=str(exc), summary_path=str(run_dir / "summary.json"))
        raise


if __name__ == "__main__":
    main()
