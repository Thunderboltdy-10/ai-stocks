"""Intraday signal construction helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd


def intraday_hybrid_positions(
    close: pd.Series,
    predicted_returns: np.ndarray,
    prediction_stds: np.ndarray,
    *,
    max_long: float,
    max_short: float,
    cost_buffer: float = 0.0002,
    smooth_alpha: float = 0.22,
) -> np.ndarray:
    """Blend ML momentum, price-action trend, and mild mean reversion."""
    c = pd.to_numeric(close, errors="coerce").ffill().bfill()
    pred = np.asarray(predicted_returns, dtype=float).reshape(-1)
    std = np.asarray(prediction_stds, dtype=float).reshape(-1)
    if len(pred) != len(c):
        n = min(len(pred), len(c))
        c = c.iloc[-n:]
        pred = pred[-n:]
        std = std[-n:]
    if len(std) != len(pred):
        std = np.full_like(pred, float(np.nanstd(pred) + 1e-6))

    # Sparse momentum signal from model confidence after explicit cost hurdle.
    edge = np.sign(pred) * np.maximum(0.0, np.abs(pred) - (1.2 * max(0.0, float(cost_buffer))))
    strength = edge / (std + 1e-6)
    abs_strength = np.abs(strength)
    q = float(np.nanquantile(abs_strength, 0.70)) if len(abs_strength) else 1.2
    threshold = max(0.9, q)
    mom = np.tanh(strength / 2.0)
    mom = np.where(mom >= 0.0, mom * (0.84 * max_long), mom * (0.75 * max_short))
    mom[(abs_strength < threshold) | (np.abs(edge) <= 0.0)] = 0.0

    # Price-action trend breakout component.
    ema12 = c.ewm(span=12, adjust=False).mean()
    ema48 = c.ewm(span=48, adjust=False).mean()
    trend_spread_fast = (ema12 - ema48) / (ema48.abs() + 1e-9)
    trend_dir = np.tanh((trend_spread_fast / 0.003).to_numpy(dtype=float))
    trend = np.where(trend_dir >= 0.0, trend_dir * (0.55 * max_long), trend_dir * (0.55 * max_short))
    upper = c.rolling(20, min_periods=8).max().shift(1)
    lower = c.rolling(20, min_periods=8).min().shift(1)
    breakout_up = (c > upper).fillna(False).to_numpy(dtype=bool)
    breakout_dn = (c < lower).fillna(False).to_numpy(dtype=bool)
    trend[breakout_up] = np.maximum(trend[breakout_up], 0.75 * max_long)
    trend[breakout_dn] = np.minimum(trend[breakout_dn], -0.75 * max_short)

    # Mean-reversion component on local z-score.
    ema20 = c.ewm(span=20, adjust=False).mean()
    dev = (c - ema20) / (ema20.abs() + 1e-9)
    dev_z = (dev / (dev.rolling(30, min_periods=10).std() + 1e-9)).clip(-3.0, 3.0).fillna(0.0)
    rev = -np.tanh(dev_z.to_numpy(dtype=float) / 1.6)
    rev = np.where(rev >= 0.0, rev * (0.28 * max_long), rev * (0.28 * max_short))

    # Trend regime picks blend weights.
    ema50 = c.ewm(span=50, adjust=False).mean()
    trend_spread = (ema20 - ema50) / (ema50.abs() + 1e-9)
    trend_gap = trend_spread.abs().fillna(0.0)
    trend_w = (0.45 + 0.42 * ((trend_gap - 0.001) / 0.008).clip(0.0, 1.0)).to_numpy(dtype=float)
    bearish = (trend_spread < -0.0010).fillna(False).to_numpy(dtype=bool)
    bullish = (trend_spread > 0.0015).fillna(False).to_numpy(dtype=bool)

    raw = 0.55 * trend + trend_w * mom + (1.0 - trend_w) * rev
    raw[bearish] = np.minimum(raw[bearish], 0.0) + 0.24 * np.clip(raw[bearish], -max_short, 0.0)
    raw[bullish] = np.maximum(raw[bullish], 0.0)
    smoothed = pd.Series(raw).ewm(alpha=float(np.clip(smooth_alpha, 0.05, 0.8)), adjust=False).mean().to_numpy(dtype=float)
    smoothed = pd.Series(smoothed).rolling(3, min_periods=1).median().to_numpy(dtype=float)

    # Turnover limiter: ignore tiny shifts that are unlikely to beat costs.
    out = np.zeros_like(smoothed)
    if len(smoothed):
        out[0] = smoothed[0]
    for i in range(1, len(smoothed)):
        # Keep small adaptive deadband so short windows do not collapse to no-trade.
        adaptive_min_change = 0.04 + 0.02 * min(1.0, abs(smoothed[i]))
        if abs(smoothed[i] - out[i - 1]) < adaptive_min_change:
            out[i] = out[i - 1]
        else:
            out[i] = smoothed[i]
    return np.clip(out, -max_short, max_long)
