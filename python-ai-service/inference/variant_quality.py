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
    selected_source: str

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


def _wfe_from_metric(val_metric: float, test_metric: float, baseline: float = 0.0) -> float:
    val_above = float(val_metric) - float(baseline)
    test_above = float(test_metric) - float(baseline)
    if val_above <= 0.0 or test_above <= 0.0:
        return 0.0
    return float((test_above / val_above) * 100.0)


def _extract_holdout_metrics(metadata: Dict[str, Any], source: str) -> Dict[str, float]:
    holdout = metadata.get("holdout", {}) if isinstance(metadata, dict) else {}
    source_metrics = holdout.get(source, {}) if isinstance(holdout, dict) else {}
    walk_forward = metadata.get("walk_forward", {}) if isinstance(metadata, dict) else {}
    wf_xgb_raw = walk_forward.get("xgb", {}) if isinstance(walk_forward, dict) else {}
    wf_lgb_raw = walk_forward.get("lgb", {}) if isinstance(walk_forward, dict) else {}
    wf_xgb = wf_xgb_raw.get("aggregate", {}) if isinstance(wf_xgb_raw, dict) else {}
    wf_lgb = wf_lgb_raw.get("aggregate", {}) if isinstance(wf_lgb_raw, dict) else {}

    val_net_sharpe = max(_safe_float(wf_xgb.get("net_sharpe_mean"), 0.0), _safe_float(wf_lgb.get("net_sharpe_mean"), 0.0))
    val_net_return = max(_safe_float(wf_xgb.get("net_return_mean"), 0.0), _safe_float(wf_lgb.get("net_return_mean"), 0.0))
    val_dir_acc = max(_safe_float(wf_xgb.get("dir_acc_mean"), 0.5), _safe_float(wf_lgb.get("dir_acc_mean"), 0.5))

    net_sharpe = _safe_float(source_metrics.get("net_sharpe"), -5.0)
    net_return = _safe_float(source_metrics.get("net_return"), -1.0)
    dir_acc = _safe_float(source_metrics.get("dir_acc"), 0.5)

    wfe_cost = _wfe_from_metric(val_net_sharpe, net_sharpe, baseline=0.0)
    if wfe_cost <= 0.0:
        wfe_cost = _wfe_from_metric(val_net_return, net_return, baseline=0.0)
    wfe_directional = _wfe_from_metric(val_dir_acc, dir_acc, baseline=0.5)
    stored_wfe = _safe_float(metadata.get("wfe"), 0.0)
    has_cost_history = any(
        abs(v) > 1e-12
        for v in (val_net_sharpe, val_net_return, net_sharpe, net_return)
    )
    if has_cost_history:
        effective_wfe = wfe_cost
    else:
        effective_wfe = wfe_directional
    if effective_wfe <= 0.0 and stored_wfe > 0.0 and not metadata.get("wfe_details") and not has_cost_history:
        effective_wfe = stored_wfe

    return {
        "dir_acc": dir_acc,
        "pred_std": _safe_float(source_metrics.get("pred_std"), 0.0),
        "pred_target_std_ratio": _safe_float(source_metrics.get("pred_target_std_ratio"), 0.0),
        "net_return": net_return,
        "net_sharpe": net_sharpe,
        "positive_pct": _safe_float(source_metrics.get("positive_pct"), 0.5),
        "ic": _safe_float(source_metrics.get("ic"), -1.0),
        "wfe": effective_wfe,
        "wfe_cost": wfe_cost,
        "wfe_directional": wfe_directional,
        "sharpe": _safe_float(source_metrics.get("sharpe"), 0.0),
        "gross_return": _safe_float(source_metrics.get("gross_return"), 0.0),
        "turnover": _safe_float(source_metrics.get("turnover"), 0.0),
        "trade_win_rate": _safe_float(source_metrics.get("trade_win_rate"), 0.0),
        "source": source,
    }


def _quality_thresholds(*, intraday: bool) -> Dict[str, float]:
    if intraday:
        return {
            "min_dir": 0.495,
            "min_std": 0.0006,
            "min_pred_target_std_ratio": 0.12,
            "min_wfe": 10.0,
            "min_net_return": -0.25,
            "min_net_sharpe": -0.05,
            "min_ic": -0.01,
            "min_pos": 0.15,
            "max_pos": 0.85,
        }
    return {
        "min_dir": 0.50,
        "min_std": 0.0010,
        "min_pred_target_std_ratio": 0.15,
        "min_wfe": 10.0,
        "min_net_return": -0.15,
        "min_net_sharpe": -0.05,
        "min_ic": -0.02,
        "min_pos": 0.20,
        "max_pos": 0.80,
    }


def _quality_score(metrics: Dict[str, float], *, intraday: bool) -> float:
    dir_acc = metrics["dir_acc"]
    pred_std = metrics["pred_std"]
    pred_target_std_ratio = metrics.get("pred_target_std_ratio", 0.0)
    net_return = metrics["net_return"]
    net_sharpe = metrics["net_sharpe"]
    pos_pct = metrics["positive_pct"]
    wfe = metrics["wfe"]
    ic = metrics["ic"]

    if intraday:
        return float(
            7.0 * net_return
            + 1.6 * net_sharpe
            + 5.0 * (dir_acc - 0.5)
            + 18.0 * max(pred_std - 0.0008, 0.0)
            + 0.8 * max(pred_target_std_ratio - 0.14, 0.0)
            + 0.7 * (wfe / 100.0)
            + 0.6 * ic
            - 1.7 * abs(pos_pct - 0.5)
        )

    return float(
        6.2 * net_return
        + 1.8 * net_sharpe
        + 5.4 * (dir_acc - 0.5)
        + 22.0 * max(pred_std - 0.0015, 0.0)
        + 1.0 * max(pred_target_std_ratio - 0.18, 0.0)
        + 0.7 * (wfe / 100.0)
        + 0.8 * ic
        - 1.4 * abs(pos_pct - 0.5)
    )


def _quality_reasons(metrics: Dict[str, float], *, intraday: bool) -> List[str]:
    thresholds = _quality_thresholds(intraday=intraday)
    reasons: List[str] = []
    if metrics["dir_acc"] < thresholds["min_dir"]:
        reasons.append(f"dir_acc<{thresholds['min_dir']:.3f}")
    if metrics["pred_std"] < thresholds["min_std"]:
        reasons.append(f"pred_std<{thresholds['min_std']:.4f}")
    if metrics.get("pred_target_std_ratio", 0.0) < thresholds["min_pred_target_std_ratio"]:
        reasons.append(f"pred_target_std_ratio<{thresholds['min_pred_target_std_ratio']:.2f}")
    if metrics["net_return"] < thresholds["min_net_return"]:
        reasons.append(f"net_return<{thresholds['min_net_return']:.2f}")
    if metrics["net_sharpe"] < thresholds["min_net_sharpe"]:
        reasons.append(f"net_sharpe<{thresholds['min_net_sharpe']:.2f}")
    if metrics["ic"] < thresholds["min_ic"]:
        reasons.append(f"ic<{thresholds['min_ic']:.2f}")
    if metrics["positive_pct"] < thresholds["min_pos"] or metrics["positive_pct"] > thresholds["max_pos"]:
        reasons.append(f"positive_pct outside [{thresholds['min_pos']:.2f}, {thresholds['max_pos']:.2f}]")

    robust_edge = (
        metrics["wfe"] >= thresholds["min_wfe"]
        or (metrics["net_sharpe"] >= 0.15 and metrics["net_return"] >= 0.05)
        or (metrics["net_return"] >= 0.12 and metrics["dir_acc"] >= thresholds["min_dir"])
    )
    if not robust_edge:
        reasons.append("insufficient_oos_edge")
    return reasons


def select_holdout_candidate(metadata: Dict[str, Any], *, intraday: bool) -> tuple[str, Dict[str, float]]:
    candidates: List[tuple[str, Dict[str, float]]] = []
    holdout = metadata.get("holdout", {}) if isinstance(metadata, dict) else {}
    for source in ("ensemble", "ensemble_directional", "ensemble_calibrated"):
        source_metrics = holdout.get(source, {}) if isinstance(holdout, dict) else {}
        if not isinstance(source_metrics, dict) or not source_metrics:
            continue
        metrics = _extract_holdout_metrics(metadata, source)
        candidates.append((source, metrics))

    if not candidates:
        return "ensemble", {
            "dir_acc": 0.5,
            "pred_std": 0.0,
            "pred_target_std_ratio": 0.0,
            "net_return": -1.0,
            "net_sharpe": -5.0,
            "positive_pct": 0.5,
            "ic": -1.0,
            "wfe": 0.0,
            "wfe_cost": 0.0,
            "wfe_directional": 0.0,
            "sharpe": 0.0,
            "gross_return": 0.0,
            "turnover": 0.0,
            "trade_win_rate": 0.0,
            "source": "ensemble",
        }

    ranked = sorted(candidates, key=lambda item: (_quality_score(item[1], intraday=intraday), item[1]["net_return"]), reverse=True)
    return ranked[0]


def score_variant_quality(metadata: Dict[str, Any], *, intraday: bool) -> VariantQuality:
    selected_source, metrics = select_holdout_candidate(metadata, intraday=intraday)
    score = _quality_score(metrics, intraday=intraday)
    reasons = _quality_reasons(metrics, intraday=intraday)
    passed = len(reasons) == 0
    return VariantQuality(
        passed=passed,
        score=float(score),
        reasons=reasons,
        metrics={k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
        selected_source=selected_source,
    )
