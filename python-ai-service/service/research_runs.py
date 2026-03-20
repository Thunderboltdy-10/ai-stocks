"""Helpers for listing completed research workflow runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS_DIR = ROOT / "experiments"


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def list_research_runs(limit: int = 12) -> List[Dict[str, Any]]:
    if not EXPERIMENTS_DIR.exists():
        return []

    summaries = sorted(
        EXPERIMENTS_DIR.glob("research_workflow_*/summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    rows: List[Dict[str, Any]] = []
    for summary_path in summaries[: max(1, int(limit))]:
        data = _read_json(summary_path)
        if not data:
            continue
        row = {
            "id": data.get("id") or summary_path.parent.name,
            "workflow": data.get("workflow", ""),
            "status": data.get("status", "unknown"),
            "startedAt": data.get("started_at", ""),
            "completedAt": data.get("completed_at", ""),
            "dailyGatePassed": bool(data.get("daily_gate_passed", False)),
            "intradayGatePassed": bool(data.get("intraday_gate_passed", False)),
            "daily": data.get("daily", {}),
            "intraday": data.get("intraday", {}),
            "summaryPath": str(summary_path),
            "runDir": str(summary_path.parent),
        }
        rows.append(row)
    return rows
