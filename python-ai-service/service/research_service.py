"""Read research artifacts for API consumption."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

from research.benchmark_lab import list_benchmark_runs, load_benchmark_run
from service.research_runs import list_research_runs as list_workflow_runs


def _workflow_run_map(limit: int = 12) -> Dict[str, Dict[str, Any]]:
    return {str(run.get("id")): run for run in list_workflow_runs(limit=limit)}


def list_research_runs(limit: int = 5) -> List[Dict[str, Any]]:
    workflow_rows = list_workflow_runs(limit=limit)
    if workflow_rows:
        return workflow_rows
    return list_benchmark_runs(limit=limit)


def get_research_run(run_id: str) -> Dict[str, Any] | None:
    workflow_runs = _workflow_run_map(limit=24)
    if run_id in workflow_runs:
        run = workflow_runs[run_id]
        summary_path = Path(str(run.get("summaryPath", "")))
        if summary_path.exists():
            try:
                import json

                with summary_path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, dict):
                    return data
            except Exception:
                return run
    return load_benchmark_run(run_id)
