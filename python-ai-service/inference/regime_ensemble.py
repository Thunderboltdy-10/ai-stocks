"""Regime-aware ensemble helpers used by inference and backtesting."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd

from utils.timeframe import is_intraday_interval


def model_quality_gate_strict(metadata: Dict) -> bool:
    """Require robust holdout quality before trusting ML sizing.

    The previous gate allowed weak models to influence sizing. This stricter
    gate only enables ML overlay when direction, dispersion, and walk-forward
    efficiency all clear thresholds.
    """
    holdout_all = metadata.get("holdout", {})
    holdout = holdout_all.get("ensemble", {})
    holdout_cal = holdout_all.get("ensemble_calibrated", {})
    interval = str(metadata.get("data_interval", "1d"))
    intraday = is_intraday_interval(interval)
    dir_acc = max(float(holdout.get("dir_acc", 0.0)), float(holdout_cal.get("dir_acc", 0.0)))
    ic = max(float(holdout.get("ic", 0.0)), float(holdout_cal.get("ic", 0.0)))
    pred_std = max(float(holdout.get("pred_std", 0.0)), float(holdout_cal.get("pred_std", 0.0)))
    sharpe = max(float(holdout.get("sharpe", 0.0)), float(holdout_cal.get("sharpe", 0.0)))
    net_sharpe = max(float(holdout.get("net_sharpe", 0.0)), float(holdout_cal.get("net_sharpe", 0.0)))
    wfe = float(metadata.get("wfe", 0.0))

    if intraday:
        return (
            dir_acc >= 0.510
            and ic >= -0.005
            and pred_std >= 0.0018
            and (wfe >= 10.0 or sharpe >= 0.10 or net_sharpe >= 0.10)
        )

    return (
        dir_acc >= 0.51
        and ic >= 0.0
        and pred_std >= 0.004
        and wfe >= 20.0
    )


def regime_exposure_from_prices(
    close: pd.Series,
    max_long: float = 1.6,
    max_short: float = 0.0,
    base_exposure: float = 0.95,
    bull_boost: float = 0.35,
    trend_boost: float = 0.20,
    bear_penalty: float = 0.30,
    bear_short_boost: float = 0.20,
    momentum_weight: float = 0.15,
    vol_target_annual: float = 0.40,
    min_scale: float = 0.90,
    max_scale: float = 1.60,
    periods_per_year: float = 252.0,
) -> np.ndarray:
    """Build a regime exposure curve from price action (long-only or signed)."""
    c = pd.to_numeric(close, errors="coerce")
    returns_1d = c.pct_change()
    intraday = float(periods_per_year) > 300.0
    if intraday:
        fast = c.ewm(span=24, adjust=False).mean()
        slow = c.ewm(span=96, adjust=False).mean()
        macro = c.ewm(span=240, adjust=False).mean()
        bull_regime = ((c > slow) & (fast > slow)).astype(float)
        trend_regime = ((fast > slow) & (slow > macro)).astype(float)
        bear_regime = ((c < slow) & (fast < slow)).astype(float)
    else:
        sma_50 = c.rolling(50, min_periods=20).mean()
        sma_200 = c.rolling(200, min_periods=80).mean()
        bull_regime = (c > sma_200).astype(float)
        trend_regime = (sma_50 > sma_200).astype(float)
        bear_regime = ((c < sma_200) & (sma_50 < sma_200)).astype(float)

    base_long = (
        base_exposure
        + bull_boost * bull_regime
        + trend_boost * trend_regime
        - bear_penalty * (1.0 - bull_regime)
    )
    base_short = -(bear_short_boost * bear_regime * (1.0 - trend_regime))
    base = base_long + np.where(max_short > 0.0, base_short, 0.0)

    ann_vol_20 = returns_1d.rolling(20, min_periods=5).std() * np.sqrt(max(1.0, float(periods_per_year)))
    vol_scale = (vol_target_annual / (ann_vol_20 + 1e-6)).clip(lower=min_scale, upper=max_scale).fillna(1.0)

    # Short-horizon momentum term helps weekly windows without introducing future data.
    momentum_20 = c / c.shift(20) - 1.0
    momentum_z = (momentum_20 / (returns_1d.rolling(20, min_periods=5).std() + 1e-6)).clip(-2.0, 2.0)

    exposure = (base * vol_scale + momentum_weight * momentum_z).bfill().fillna(base_exposure)

    if intraday:
        # Keep stronger long carry in persistent uptrends and reduce false short drift.
        momentum_60 = c / c.shift(60) - 1.0
        macro_bull = ((c > slow) & (momentum_60 > 0.04)).fillna(False)
        macro_bear = ((c < slow) & (momentum_60 < -0.04)).fillna(False)
        exposure = pd.Series(exposure, index=c.index)
        exposure.loc[macro_bull] = np.maximum(exposure.loc[macro_bull], 0.95)
        exposure.loc[macro_bear] = np.minimum(exposure.loc[macro_bear], 0.10)
        exposure = exposure.to_numpy(dtype=float)

    exposure_arr = np.asarray(exposure, dtype=float)
    exposure_arr = np.clip(exposure_arr, -max_short, max_long)
    return exposure_arr


def combine_ml_and_regime(
    regime_positions: np.ndarray,
    ml_positions: np.ndarray | None,
    ml_gate_passed: bool,
    ml_weight: float = 0.20,
    fallback_ml_weight: float = 0.08,
    max_long: float = 1.6,
    max_short: float = 0.0,
) -> np.ndarray:
    """Blend regime and ML positions when ML passes strict quality checks."""
    regime = np.asarray(regime_positions, dtype=float)
    if ml_positions is None:
        return np.clip(regime, -max_short, max_long)

    ml = np.asarray(ml_positions, dtype=float)
    if len(ml) != len(regime):
        return np.clip(regime, -max_short, max_long)

    weight_ml = float(np.clip(ml_weight if ml_gate_passed else fallback_ml_weight, 0.0, 1.0))
    blended = (1.0 - weight_ml) * regime + weight_ml * ml
    return np.clip(blended, -max_short, max_long)


def apply_short_safety_filter(
    close: pd.Series,
    positions: np.ndarray,
    predicted_returns: np.ndarray | None = None,
    max_short: float = 0.0,
    interval: str = "1d",
) -> np.ndarray:
    """Allow shorts only under bearish structure to avoid trend-fighting whipsaws."""
    pos = np.asarray(positions, dtype=float).reshape(-1)
    if max_short <= 0.0:
        return np.clip(pos, 0.0, np.inf)
    if len(pos) == 0:
        return pos

    c = pd.to_numeric(close, errors="coerce")
    sma50 = c.rolling(50, min_periods=20).mean()
    sma200 = c.rolling(200, min_periods=80).mean()
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema80 = c.ewm(span=80, adjust=False).mean()
    momentum20 = c / c.shift(20) - 1.0

    bear_trend = (c < sma200) & (sma50 < sma200)
    bear_momentum = momentum20 < -0.01
    intraday = is_intraday_interval(interval)
    fast_bear = (ema20 < ema80) & (momentum20 < -0.002)
    momentum60 = c / c.shift(60) - 1.0
    macro_bull = ((c > sma200) & (momentum60 > 0.04)).fillna(False).to_numpy(dtype=bool)

    allow_short = (bear_trend & bear_momentum).fillna(False).to_numpy(dtype=bool)
    if predicted_returns is not None:
        pred = np.asarray(predicted_returns, dtype=float).reshape(-1)
        if len(pred) == len(pos):
            if intraday:
                # Require stronger downside edge for intraday shorts to avoid trend-fighting.
                allow_short = allow_short | (
                    (pred < -0.0025)
                    & (fast_bear.fillna(False).to_numpy(dtype=bool) | bear_trend.fillna(False).to_numpy(dtype=bool))
                )
            else:
                allow_short = allow_short | (pred < -0.0018)

    if intraday and len(macro_bull) == len(allow_short):
        allow_short = allow_short & (~macro_bull)

    out = pos.copy()
    out[(out < 0.0) & (~allow_short)] = 0.0

    if intraday:
        # Scale short capacity with bearish trend strength.
        trend_gap = (
            ((sma200 - c) / (sma200.abs() + 1e-9)).clip(lower=0.0, upper=0.08)
            + 0.6 * ((ema80 - c) / (ema80.abs() + 1e-9)).clip(lower=0.0, upper=0.05)
        ).clip(lower=0.0, upper=0.08).fillna(0.0)
        cap = max_short * (0.20 + 0.80 * (trend_gap / 0.08))
        cap_np = cap.to_numpy(dtype=float)
        if len(cap_np) == len(out):
            out = np.maximum(out, -cap_np)

    return np.clip(out, -max_short, np.inf)


def adaptive_overlay_weight(
    close: pd.Series,
    predicted_returns: np.ndarray,
    lookback: int = 60,
    min_weight: float = 0.05,
    max_weight: float = 0.90,
) -> np.ndarray:
    """Causal confidence score for how much active overlay to trust."""
    c = pd.to_numeric(close, errors="coerce")
    ret = c.pct_change()
    pred = pd.Series(np.asarray(predicted_returns, dtype=float), index=c.index).shift(1)

    dir_hit = (np.sign(pred) == np.sign(ret)).astype(float)
    dir_quality = dir_hit.rolling(lookback, min_periods=max(20, lookback // 3)).mean()
    ic = pred.rolling(lookback, min_periods=max(20, lookback // 3)).corr(ret).fillna(0.0)

    dir_score = ((dir_quality - 0.5) / 0.08).clip(0.0, 1.0)
    ic_score = ((ic + 0.02) / 0.08).clip(0.0, 1.0)
    pred_score = (dir_score * ic_score).fillna(0.0)

    sma50 = c.rolling(50, min_periods=20).mean()
    sma200 = c.rolling(200, min_periods=80).mean()
    trend_gap = (sma50 - sma200) / (sma200.abs() + 1e-9)
    momentum60 = c / c.shift(60) - 1.0
    trend_strength = ((trend_gap.abs() - 0.003) / 0.05).clip(0.0, 1.0).fillna(0.0)
    trend_up = ((trend_gap > 0.0) & (momentum60 > 0.0)).astype(float)
    trend_down = ((trend_gap < 0.0) & (momentum60 < 0.0)).astype(float)
    regime_score = (0.8 * trend_strength * trend_up + 0.6 * trend_strength * trend_down).fillna(0.0)

    # Prefer predictive confidence first, then promote active overlay in strong trends.
    score = np.maximum(pred_score.to_numpy(dtype=float), (0.65 * regime_score).to_numpy(dtype=float))
    score = pd.Series(score, index=c.index).clip(0.0, 1.0)
    weight = min_weight + (max_weight - min_weight) * score
    return weight.to_numpy(dtype=float)


def blend_positions_with_core(
    active_positions: np.ndarray,
    overlay_weights: np.ndarray,
    core_long: float = 1.0,
    max_long: float = 1.6,
    max_short: float = 0.0,
    smooth_alpha: float = 0.20,
) -> np.ndarray:
    """Blend active strategy with passive long core and smooth churn."""
    active = np.asarray(active_positions, dtype=float).reshape(-1)
    w = np.asarray(overlay_weights, dtype=float).reshape(-1)
    if len(active) == 0:
        return active
    if len(w) != len(active):
        w = np.full_like(active, float(np.nanmean(w)) if len(w) else 0.2)

    core = np.full_like(active, float(core_long))
    blended = core + np.clip(w, 0.0, 1.0) * (active - core)

    # Causal smoothing to reduce turnover and transaction-cost drag.
    out = np.zeros_like(blended)
    out[0] = blended[0]
    alpha = float(np.clip(smooth_alpha, 0.01, 1.0))
    for i in range(1, len(blended)):
        out[i] = alpha * blended[i] + (1.0 - alpha) * out[i - 1]

    return np.clip(out, -max_short, max_long)
