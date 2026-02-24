"""Signal calibration and policy-based position construction."""

from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from utils.timeframe import annualization_factor_for_interval


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    x = np.clip(x, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-x))


def apply_direction_calibration(
    predicted_returns: np.ndarray,
    calibrator: Dict | None,
) -> Tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(predicted_returns, dtype=float).reshape(-1)
    std_pred = float(max(1e-9, np.nanstd(pred)))
    neutral_conf = np.tanh(pred / (2.0 * std_pred))
    if not calibrator or not bool(calibrator.get("enabled", False)):
        return pred, neutral_conf.astype(float)

    mean = float(calibrator.get("mean_pred", 0.0))
    std = float(max(1e-9, calibrator.get("std_pred", 1.0)))
    coef = float(calibrator.get("coef", 0.0))
    intercept = float(calibrator.get("intercept", 0.0))
    min_scale = float(np.clip(calibrator.get("min_scale", 0.65), 0.35, 1.0))
    confidence_floor = float(np.clip(calibrator.get("confidence_floor", 0.05), 0.0, 0.45))

    z = (pred - mean) / std
    p_up = _sigmoid(coef * z + intercept)
    signed_conf = 2.0 * p_up - 1.0

    # Use calibrated direction probability to flip/reduce weak edges.
    conf_mag = np.maximum(np.abs(signed_conf), confidence_floor)
    scale = min_scale + (1.0 - min_scale) * conf_mag
    scaled = np.sign(signed_conf) * np.abs(pred) * scale
    return scaled.astype(float), signed_conf.astype(float)


def positions_from_policy(
    predicted_returns: np.ndarray,
    prediction_stds: np.ndarray,
    policy: Dict | None,
    *,
    max_long: float,
    max_short: float,
) -> np.ndarray:
    pred = np.asarray(predicted_returns, dtype=float).reshape(-1)
    std = np.asarray(prediction_stds, dtype=float).reshape(-1)
    if len(std) != len(pred):
        std = np.full_like(pred, float(np.nanstd(pred) + 1e-6))
    std_ref = float(max(1e-6, np.nanstd(pred)))

    if not policy or not bool(policy.get("enabled", False)):
        # Fallback neutral continuous policy.
        strength = np.tanh(pred / max(1e-9, std_ref))
        pos = np.where(strength >= 0.0, strength * max_long, strength * max_short)
        return np.clip(pos, -max_short, max_long)

    long_threshold = float(max(0.0, policy.get("long_threshold", 0.0)))
    short_threshold = float(max(0.0, policy.get("short_threshold", 0.0)))
    hold_threshold = float(max(0.0, policy.get("hold_threshold", 0.0)))
    long_scale = float(np.clip(policy.get("long_scale", 1.0), 0.0, 3.0))
    short_scale = float(np.clip(policy.get("short_scale", 1.0), 0.0, 3.0))
    long_conf = float(max(0.0, policy.get("long_conf", 0.0)))
    short_conf = float(max(0.0, policy.get("short_conf", 0.0)))
    smooth_alpha = float(np.clip(policy.get("smooth_alpha", 0.2), 0.01, 1.0))

    conf = np.abs(pred) / (std + 1e-9)
    strength = np.tanh(np.abs(pred) / max(1e-9, std_ref))

    target = np.zeros_like(pred)
    long_mask = (pred > long_threshold) & (conf >= long_conf)
    short_mask = (pred < -short_threshold) & (conf >= short_conf)

    target[long_mask] = np.minimum(max_long, long_scale * strength[long_mask])
    target[short_mask] = -np.minimum(max_short, short_scale * strength[short_mask])
    target[np.abs(pred) <= hold_threshold] = 0.0

    smoothed = pd.Series(target).ewm(alpha=smooth_alpha, adjust=False).mean().to_numpy(dtype=float)
    return np.clip(smoothed, -max_short, max_long)


def calibrate_execution_policy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    interval: str = "1d",
    max_long: float = 1.6,
    max_short: float = 0.25,
) -> Dict:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(pred)
    y = y[mask]
    pred = pred[mask]

    if len(y) < 260:
        return {"enabled": False, "reason": "insufficient_samples"}

    abs_pred = np.abs(pred)
    std_ref = float(max(1e-6, np.nanstd(pred)))
    ann = float(max(1.0, annualization_factor_for_interval(interval)))
    cost_per_turn = 0.0002 if interval in {"1h", "30m", "15m", "5m", "1m"} else 0.0008

    split_idx = int(len(y) * 0.70)
    split_idx = min(max(180, split_idx), len(y) - 80)

    # Candidate thresholds from empirical prediction distribution.
    abs_train = abs_pred[:split_idx]
    q = [0.60, 0.72, 0.82, 0.90] if interval in {"1h", "30m", "15m", "5m", "1m"} else [0.50, 0.65, 0.80]
    thr = sorted({float(np.nanquantile(abs_train, qq)) for qq in q if np.isfinite(np.nanquantile(abs_train, qq))})
    if not thr:
        thr = [float(np.nanmedian(abs_pred)) if np.isfinite(np.nanmedian(abs_pred)) else 0.0]
    intraday = interval in {"1h", "30m", "15m", "5m", "1m"}
    if intraday:
        long_scales = [0.90, 1.15, 1.40]
        short_scales = [0.60, 0.85, 1.10]
        conf_levels = [0.0, 0.6, 1.0]
        smooth_levels = [0.10, 0.18, 0.28]
        min_change_levels = [0.02, 0.05, 0.08, 0.12, 0.18]
    else:
        long_scales = [0.90, 1.30]
        short_scales = [0.90, 1.30]
        conf_levels = [0.0, 0.6]
        smooth_levels = [0.15, 0.30]
        min_change_levels = [0.0, 0.05]

    def _simulate(cfg: Dict, start: int, end: int) -> Dict:
        p = pred[start:end]
        y_seg = y[start:end]
        if len(p) < 40:
            return {
                "score": -999.0,
                "cum_return": 0.0,
                "sharpe": 0.0,
                "max_drawdown": 0.0,
                "win_rate": 0.0,
                "turnover": 0.0,
            }
        pos = positions_from_policy(
            p,
            np.full_like(p, std_ref),
            cfg | {"enabled": True},
            max_long=max_long,
            max_short=max_short,
        )
        min_change = float(max(0.0, cfg.get("min_position_change", 0.0)))
        if min_change > 0.0 and len(pos) > 1:
            pos_adj = pos.copy()
            for i in range(1, len(pos_adj)):
                if abs(pos_adj[i] - pos_adj[i - 1]) < min_change:
                    pos_adj[i] = pos_adj[i - 1]
            pos = pos_adj
        lag = np.zeros_like(pos)
        lag[1:] = pos[:-1]
        turnover = np.abs(np.diff(pos, prepend=0.0))
        gross = lag * y_seg
        net = gross - cost_per_turn * turnover
        eq = np.cumprod(1.0 + net)
        cum = float(eq[-1] - 1.0)
        std = float(np.std(net))
        sharpe = float(np.sqrt(ann) * np.mean(net) / std) if std > 1e-12 else 0.0
        run_max = np.maximum.accumulate(eq)
        mdd = float(np.min(eq / np.maximum(run_max, 1e-9) - 1.0))
        win = float(np.mean(net > 0.0))
        trade_activity = float(np.mean(turnover))
        # Intraday demands stronger turnover control and cost-adjusted expectancy.
        if intraday:
            score = (
                2.8 * cum
                + 1.2 * sharpe
                + 0.35 * win
                - 1.1 * abs(mdd)
                - 1.6 * trade_activity
            )
        else:
            score = 2.4 * cum + 1.1 * sharpe + 0.4 * win - 1.0 * abs(mdd) - 0.9 * trade_activity
        return {
            "score": float(score),
            "cum_return": cum,
            "sharpe": sharpe,
            "max_drawdown": mdd,
            "win_rate": win,
            "turnover": trade_activity,
        }

    baseline_cfg = {
        "long_threshold": float(np.max(abs_pred) + 1.0),
        "short_threshold": float(np.max(abs_pred) + 1.0),
        "hold_threshold": float(np.max(abs_pred) + 1.0),
        "long_scale": 1.0,
        "short_scale": 1.0,
        "long_conf": 0.0,
        "short_conf": 0.0,
        "smooth_alpha": 0.20,
        "min_position_change": 0.0,
    }
    baseline_train = _simulate(baseline_cfg, 0, split_idx)
    baseline_val = _simulate(baseline_cfg, split_idx, len(y))
    baseline_combined = 0.35 * baseline_train["score"] + 0.65 * baseline_val["score"]

    best_cfg: Dict | None = None
    best_perf: Dict | None = None

    for long_thr in thr:
        short_thr_candidates = sorted(
            {
                float(long_thr),
                float(long_thr * (1.20 if intraday else 1.0)),
                float(thr[-1]),
            }
        )
        for short_thr in short_thr_candidates:
            for hold in [0.0, 0.5 * min(long_thr, short_thr)]:
                for ls in long_scales:
                    for ss in short_scales:
                        for lc in conf_levels:
                            for sc in conf_levels:
                                for alpha in smooth_levels:
                                    for min_change in min_change_levels:
                                        cfg = {
                                            "long_threshold": long_thr,
                                            "short_threshold": short_thr,
                                            "hold_threshold": hold,
                                            "long_scale": ls,
                                            "short_scale": ss,
                                            "long_conf": lc,
                                            "short_conf": sc,
                                            "smooth_alpha": alpha,
                                            "min_position_change": min_change,
                                        }
                                        train_perf = _simulate(cfg, 0, split_idx)
                                        val_perf = _simulate(cfg, split_idx, len(y))
                                        combined = 0.35 * train_perf["score"] + 0.65 * val_perf["score"]
                                        perf = {
                                            "score": combined,
                                            "train": train_perf,
                                            "val": val_perf,
                                        }
                                        if best_perf is None or perf["score"] > best_perf["score"]:
                                            best_cfg, best_perf = cfg, perf

    if best_cfg is None or best_perf is None:
        return {"enabled": False, "reason": "search_failed"}

    val_perf = best_perf["val"]
    if intraday:
        near_zero_activity = val_perf["turnover"] < 0.004 and val_perf["cum_return"] < baseline_val["cum_return"] + 0.002
        no_edge = (
            best_perf["score"] <= baseline_combined + 0.05
            or val_perf["cum_return"] <= 0.001
            or val_perf["turnover"] > 0.11
            or near_zero_activity
            or (val_perf["cum_return"] <= 0.0 and val_perf["sharpe"] <= 0.0)
        )
    else:
        no_edge = (
            best_perf["score"] <= baseline_combined + 0.02
            or (val_perf["cum_return"] <= 0.0 and val_perf["sharpe"] <= 0.0)
        )

    if no_edge:
        return {
            "enabled": False,
            "reason": "no_policy_edge",
            "baseline": {
                "train": baseline_train,
                "val": baseline_val,
                "combined_score": baseline_combined,
            },
            "candidate": {
                "train": best_perf["train"],
                "val": best_perf["val"],
                "combined_score": best_perf["score"],
            },
        }

    if intraday and "min_position_change" in best_cfg:
        best_cfg["min_position_change"] = float(min(0.10, max(0.0, best_cfg["min_position_change"])))

    return {
        "enabled": True,
        **best_cfg,
        "calibration_performance": {
            "train": best_perf["train"],
            "val": best_perf["val"],
            "combined_score": best_perf["score"],
        },
        "baseline_performance": {
            "train": baseline_train,
            "val": baseline_val,
            "combined_score": baseline_combined,
        },
        "interval": interval,
    }
