"""Quality scoring for model variants based on training metadata.

The goal is to prevent weak/stale variants from being auto-selected purely
because they performed well on a narrow backtest slice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class VariantQuality:
    passed: bool
    score: float
    reasons: List[str]
    metrics: Dict[str, float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if out != out:  # NaN
            return default
        return out
    except Exception:
        return default


def load_training_metadata(symbol: str, variant: str) -> Dict[str, Any]:
    path = Path("saved_models") / symbol.upper() / variant / "training_metadata.json"
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _combined_holdout_metrics(metadata: Dict[str, Any]) -> Dict[str, float]:
    holdout = metadata.get("holdout", {}) if isinstance(metadata, dict) else {}
    ens = holdout.get("ensemble", {}) if isinstance(holdout, dict) else {}
    cal = holdout.get("ensemble_calibrated", {}) if isinstance(holdout, dict) else {}

    def _max_metric(key: str, default: float) -> float:
        return max(_safe_float(ens.get(key), default), _safe_float(cal.get(key), default))

    def _from_cal_first(key: str, default: float) -> float:
        cal_val = _safe_float(cal.get(key), default)
        ens_val = _safe_float(ens.get(key), default)
        return cal_val if abs(cal_val - default) > 1e-12 else ens_val

    return {
        "dir_acc": _max_metric("dir_acc", 0.5),
        "pred_std": _max_metric("pred_std", 0.0),
        "net_return": _max_metric("net_return", -1.0),
        "net_sharpe": _max_metric("net_sharpe", -5.0),
        "positive_pct": _from_cal_first("positive_pct", _safe_float(ens.get("positive_pct"), 0.5)),
        "ic": _max_metric("ic", -1.0),
        "wfe": _safe_float(metadata.get("wfe"), 0.0),
    }


def score_variant_quality(metadata: Dict[str, Any], *, intraday: bool) -> VariantQuality:
    metrics = _combined_holdout_metrics(metadata)
    dir_acc = metrics["dir_acc"]
    pred_std = metrics["pred_std"]
    net_return = metrics["net_return"]
    net_sharpe = metrics["net_sharpe"]
    pos_pct = metrics["positive_pct"]
    wfe = metrics["wfe"]

    if intraday:
        min_dir = 0.495
        min_std = 0.0006
        min_wfe = 20.0
        min_net_return = -0.35
        min_pos = 0.15
        max_pos = 0.85
        score = (
            7.5 * net_return
            + 1.6 * net_sharpe
            + 5.0 * (dir_acc - 0.5)
            + 20.0 * max(pred_std - 0.0008, 0.0)
            + 0.7 * (wfe / 100.0)
            - 1.7 * abs(pos_pct - 0.5)
        )
    else:
        min_dir = 0.50
        min_std = 0.0010
        min_wfe = 25.0
        min_net_return = -0.30
        min_pos = 0.20
        max_pos = 0.80
        score = (
            6.0 * net_return
            + 1.8 * net_sharpe
            + 5.5 * (dir_acc - 0.5)
            + 24.0 * max(pred_std - 0.0015, 0.0)
            + 0.8 * (wfe / 100.0)
            - 1.4 * abs(pos_pct - 0.5)
        )

    reasons: List[str] = []
    if dir_acc < min_dir:
        reasons.append(f"dir_acc<{min_dir:.3f}")
    if pred_std < min_std:
        reasons.append(f"pred_std<{min_std:.4f}")
    if net_return < min_net_return:
        reasons.append(f"net_return<{min_net_return:.2f}")
    if wfe < min_wfe:
        reasons.append(f"wfe<{min_wfe:.0f}")
    if pos_pct < min_pos or pos_pct > max_pos:
        reasons.append(f"positive_pct outside [{min_pos:.2f}, {max_pos:.2f}]")

    passed = len(reasons) == 0
    return VariantQuality(
        passed=passed,
        score=float(score),
        reasons=reasons,
        metrics={k: float(v) for k, v in metrics.items()},
    )
