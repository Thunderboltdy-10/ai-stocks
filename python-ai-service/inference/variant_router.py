"""Model-variant routing helpers.

Routes requests to the best available variant for a symbol/timeframe using a
saved registry, then falls back to metadata-based scoring if registry is absent.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from inference.variant_quality import score_variant_quality
from utils.timeframe import is_intraday_interval


@dataclass
class VariantCandidate:
    name: str
    path: Path
    metadata: Dict[str, Any]


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
        if out != out:  # NaN
            return default
        return out
    except Exception:
        return default


def _variant_dir(symbol: str, variant: str) -> Path:
    return Path("saved_models") / symbol.upper() / variant


def _load_metadata(path: Path) -> Dict[str, Any]:
    meta_path = path / "training_metadata.json"
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _collect_candidates(symbol: str, patterns: List[str], include_names: List[str] | None = None) -> List[VariantCandidate]:
    root = Path("saved_models") / symbol.upper()
    if not root.exists():
        return []

    seen: set[str] = set()
    out: List[VariantCandidate] = []

    for pattern in patterns:
        for d in sorted(root.glob(pattern)):
            if not d.is_dir() or d.name in seen:
                continue
            if not (d / "xgb_model.joblib").exists():
                continue
            seen.add(d.name)
            out.append(VariantCandidate(name=d.name, path=d, metadata=_load_metadata(d)))

    for name in include_names or []:
        d = root / name
        if not d.is_dir() or d.name in seen:
            continue
        if not (d / "xgb_model.joblib").exists():
            continue
        seen.add(d.name)
        out.append(VariantCandidate(name=d.name, path=d, metadata=_load_metadata(d)))

    return out


def _intraday_candidates(symbol: str) -> List[VariantCandidate]:
    return _collect_candidates(symbol, patterns=["gbm_intraday_1h*"], include_names=[])


def _daily_candidates(symbol: str) -> List[VariantCandidate]:
    return _collect_candidates(symbol, patterns=["gbm_daily*"], include_names=["gbm"])


def _score_candidate(meta: Dict[str, Any], *, segment: str, intraday: bool) -> float:
    quality = score_variant_quality(meta, intraday=intraday)
    hold = meta.get("holdout", {}) if isinstance(meta, dict) else {}
    ens = hold.get("ensemble", {}) if isinstance(hold, dict) else {}
    cal = hold.get("ensemble_calibrated", {}) if isinstance(hold, dict) else {}

    dir_acc = max(_safe_float(ens.get("dir_acc"), 0.5), _safe_float(cal.get("dir_acc"), 0.5))
    pred_std = max(_safe_float(ens.get("pred_std"), 0.0), _safe_float(cal.get("pred_std"), 0.0))
    net_return = max(_safe_float(ens.get("net_return"), -0.5), _safe_float(cal.get("net_return"), -0.5))
    net_sharpe = max(_safe_float(ens.get("net_sharpe"), -3.0), _safe_float(cal.get("net_sharpe"), -3.0))
    pos_pct = _safe_float(cal.get("positive_pct"), _safe_float(ens.get("positive_pct"), 0.5))
    bias_penalty = abs(pos_pct - 0.5)
    wfe = _safe_float(meta.get("wfe"), 0.0)

    horizon = max(1.0, _safe_float(meta.get("target_horizon_days"), 1.0))
    horizon_scale = 6.0 if intraday else 5.0
    if segment == "long":
        horizon_pref = min(1.0, horizon / horizon_scale)
    else:
        horizon_pref = 1.0 - min(1.0, abs(horizon - 1.0) / horizon_scale)

    if intraday:
        score = (
            4.0 * net_return
            + 1.2 * net_sharpe
            + 6.0 * (dir_acc - 0.5)
            + 24.0 * max(pred_std - 0.0015, 0.0)
            + 0.6 * (wfe / 100.0)
            + 0.9 * horizon_pref
            - 2.2 * bias_penalty
        )
    else:
        score = (
            5.0 * net_return
            + 1.4 * net_sharpe
            + 6.0 * (dir_acc - 0.5)
            + 30.0 * max(pred_std - 0.003, 0.0)
            + 0.6 * (wfe / 100.0)
            + 0.7 * horizon_pref
            - 2.0 * bias_penalty
        )
    # Strongly prefer variants that pass holdout quality checks.
    score += 0.35 * quality.score
    score += 0.9 if quality.passed else -1.6
    return float(score)


def _segment_from_window(start: str | None, end: str | None, *, intraday: bool) -> str:
    if not start or not end:
        return "long"
    try:
        s = pd.Timestamp(start)
        e = pd.Timestamp(end)
        span_days = float((e - s).total_seconds() / 86400.0)
    except Exception:
        return "long"

    if intraday:
        return "long" if span_days >= 70.0 else "short"
    return "long" if span_days >= 365.0 else "short"


def _select_from_registry(symbol: str, segment: str, registry_name: str, *, intraday: bool) -> Optional[str]:
    reg_path = Path("saved_models") / symbol.upper() / registry_name
    if not reg_path.exists():
        return None

    try:
        with reg_path.open("r", encoding="utf-8") as fh:
            registry = json.load(fh)
    except Exception:
        return None

    if not isinstance(registry, dict):
        return None

    key = "long_window_variant" if segment == "long" else "short_window_variant"
    candidate = str(registry.get(key, "")).strip()
    if candidate and _variant_dir(symbol, candidate).exists():
        metadata = _load_metadata(_variant_dir(symbol, candidate))
        if score_variant_quality(metadata, intraday=intraday).passed:
            return candidate

    fallback = str(registry.get("overall_variant", "")).strip()
    if fallback and _variant_dir(symbol, fallback).exists():
        metadata = _load_metadata(_variant_dir(symbol, fallback))
        if score_variant_quality(metadata, intraday=intraday).passed:
            return fallback
    return None


def _existing_or_default(symbol: str, preferred: str, default_variant: str) -> str:
    if _variant_dir(symbol, preferred).exists():
        return preferred
    return default_variant


def resolve_model_variant(
    *,
    symbol: str,
    requested_variant: str,
    data_interval: str,
    start: str | None = None,
    end: str | None = None,
    default_variant: str = "gbm",
) -> str:
    """Resolve `auto*` requests to a concrete saved-model variant."""
    requested = str(requested_variant or "").strip() or default_variant
    if requested not in {"auto", "auto_intraday", "auto_daily"}:
        return requested

    intraday = is_intraday_interval(data_interval)

    if requested == "auto_intraday" and not intraday:
        return default_variant
    if requested == "auto_daily" and intraday:
        return default_variant

    if intraday:
        registry_name = "intraday_variant_registry.json"
        segment = _segment_from_window(start, end, intraday=True)
        default_fallback = _existing_or_default(symbol, "gbm_intraday_1h_v3", default_variant)
        from_registry = _select_from_registry(
            symbol,
            segment=segment,
            registry_name=registry_name,
            intraday=True,
        )
        if from_registry:
            return from_registry
        candidates = _intraday_candidates(symbol)
        if not candidates:
            return default_fallback
        scored = [
            (c, _score_candidate(c.metadata, segment=segment, intraday=True), score_variant_quality(c.metadata, intraday=True))
            for c in candidates
        ]
        passed = [item for item in scored if item[2].passed]
        pool = passed if passed else scored
        ranked = sorted(pool, key=lambda item: item[1], reverse=True)
        return ranked[0][0].name if ranked else default_fallback

    registry_name = "daily_variant_registry.json"
    segment = _segment_from_window(start, end, intraday=False)
    default_fallback = _existing_or_default(symbol, "gbm_daily_v4", default_variant)
    from_registry = _select_from_registry(
        symbol,
        segment=segment,
        registry_name=registry_name,
        intraday=False,
    )
    if from_registry:
        return from_registry

    candidates = _daily_candidates(symbol)
    if not candidates:
        return default_fallback
    scored = [
        (c, _score_candidate(c.metadata, segment=segment, intraday=False), score_variant_quality(c.metadata, intraday=False))
        for c in candidates
    ]
    passed = [item for item in scored if item[2].passed]
    pool = passed if passed else scored
    ranked = sorted(pool, key=lambda item: item[1], reverse=True)
    return ranked[0][0].name if ranked else default_fallback
