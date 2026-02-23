"""Regime-aware ensemble helpers used by inference and backtesting."""

from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


def model_quality_gate_strict(metadata: Dict) -> bool:
    """Require robust holdout quality before trusting ML sizing.

    The previous gate allowed weak models to influence sizing. This stricter
    gate only enables ML overlay when direction, dispersion, and walk-forward
    efficiency all clear thresholds.
    """
    holdout = metadata.get("holdout", {}).get("ensemble", {})
    dir_acc = float(holdout.get("dir_acc", 0.0))
    ic = float(holdout.get("ic", 0.0))
    pred_std = float(holdout.get("pred_std", 0.0))
    wfe = float(metadata.get("wfe", 0.0))

    return (
        dir_acc >= 0.52
        and ic >= 0.0
        and pred_std >= 0.005
        and wfe >= 40.0
    )


def regime_exposure_from_prices(
    close: pd.Series,
    max_long: float = 1.6,
    base_exposure: float = 0.95,
    bull_boost: float = 0.35,
    trend_boost: float = 0.20,
    bear_penalty: float = 0.15,
    momentum_weight: float = 0.15,
    vol_target_annual: float = 0.40,
    min_scale: float = 0.90,
    max_scale: float = 1.60,
) -> np.ndarray:
    """Build a long-only regime exposure curve from price action.

    This path is intentionally model-agnostic so the strategy remains tradable
    even when ML models fail strict quality checks.
    """
    c = pd.to_numeric(close, errors="coerce")
    returns_1d = c.pct_change()

    sma_50 = c.rolling(50, min_periods=20).mean()
    sma_200 = c.rolling(200, min_periods=80).mean()

    bull_regime = (c > sma_200).astype(float)
    trend_regime = (sma_50 > sma_200).astype(float)

    base = (
        base_exposure
        + bull_boost * bull_regime
        + trend_boost * trend_regime
        - bear_penalty * (1.0 - bull_regime)
    )

    ann_vol_20 = returns_1d.rolling(20, min_periods=5).std() * np.sqrt(252.0)
    vol_scale = (vol_target_annual / (ann_vol_20 + 1e-6)).clip(lower=min_scale, upper=max_scale).fillna(1.0)

    # Short-horizon momentum term helps weekly windows without introducing future data.
    momentum_20 = c / c.shift(20) - 1.0
    momentum_z = (momentum_20 / (returns_1d.rolling(20, min_periods=5).std() + 1e-6)).clip(-2.0, 2.0)

    exposure = (base * vol_scale + momentum_weight * momentum_z).bfill().fillna(base_exposure)

    exposure = exposure.clip(lower=0.0, upper=max_long)
    return exposure.to_numpy(dtype=float)


def combine_ml_and_regime(
    regime_positions: np.ndarray,
    ml_positions: np.ndarray | None,
    ml_gate_passed: bool,
    ml_weight: float = 0.20,
    max_long: float = 1.6,
) -> np.ndarray:
    """Blend regime and ML positions when ML passes strict quality checks."""
    regime = np.asarray(regime_positions, dtype=float)
    if not ml_gate_passed or ml_positions is None:
        return np.clip(regime, 0.0, max_long)

    ml = np.asarray(ml_positions, dtype=float)
    if len(ml) != len(regime):
        return np.clip(regime, 0.0, max_long)

    weight_ml = float(np.clip(ml_weight, 0.0, 1.0))
    blended = (1.0 - weight_ml) * regime + weight_ml * ml
    return np.clip(blended, 0.0, max_long)
