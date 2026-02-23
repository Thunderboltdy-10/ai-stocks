"""Position sizing with half-Kelly and drawdown controls."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PositionSizingConfig:
    max_long: float = 1.8
    max_short: float = 0.1
    dead_zone: float = 0.0007
    drawdown_circuit_breaker: float = 0.25
    drawdown_max_position: float = 0.60
    base_long_bias: float = 1.30
    bearish_risk_off: float = 0.10


class PositionSizer:
    def __init__(self, config: PositionSizingConfig | None = None) -> None:
        self.config = config or PositionSizingConfig()

    @staticmethod
    def half_kelly_fraction(win_rate: float, avg_win: float, avg_loss: float) -> float:
        avg_loss = max(avg_loss, 1e-6)
        b = avg_win / avg_loss
        p = np.clip(win_rate, 1e-6, 1 - 1e-6)
        q = 1.0 - p
        kelly = (p * b - q) / max(b, 1e-6)
        return max(0.0, 0.5 * kelly)

    def size_position(
        self,
        predicted_return: float,
        prediction_std: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        current_equity: float,
        high_water_mark: float,
        base_long_bias: float | None = None,
    ) -> float:
        cfg = self.config

        kelly = self.half_kelly_fraction(win_rate, avg_win, avg_loss)
        signal = float(np.tanh(predicted_return / max(prediction_std, 1e-6)))
        alpha = float(1.4 * kelly * signal)
        base_bias = float(cfg.base_long_bias if base_long_bias is None else base_long_bias)

        # Long-biased baseline for drift-dominated equities; model signal tilts around baseline.
        position = base_bias + alpha

        if predicted_return < -cfg.dead_zone:
            position -= cfg.bearish_risk_off
        elif abs(predicted_return) < cfg.dead_zone:
            position = max(0.7 * base_bias, 0.0)

        position = float(np.clip(position, -cfg.max_short, cfg.max_long))

        drawdown = 0.0
        if high_water_mark > 0:
            drawdown = max(0.0, (high_water_mark - current_equity) / high_water_mark)

        if drawdown > cfg.drawdown_circuit_breaker:
            cap = cfg.drawdown_max_position
            position = float(np.clip(position, -min(cfg.max_short, cap), cap))

        return position

    def size_batch(
        self,
        predicted_returns: np.ndarray,
        prediction_stds: np.ndarray,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        trend_signal: np.ndarray | None = None,
    ) -> np.ndarray:
        preds = np.asarray(predicted_returns, dtype=float).reshape(-1)
        stds = np.asarray(prediction_stds, dtype=float).reshape(-1)
        if len(stds) != len(preds):
            stds = np.full_like(preds, float(np.nanstd(preds) + 1e-6))
        trend = None
        if trend_signal is not None:
            trend = np.asarray(trend_signal, dtype=float).reshape(-1)
            if len(trend) != len(preds):
                trend = None

        positions = np.zeros_like(preds)
        equity = 1.0
        hwm = 1.0

        for i in range(len(preds)):
            base_bias = None
            if trend is not None:
                # Trend-aware baseline: risk-off in weak trend, aggressive in strong trend.
                t = float(np.clip(trend[i], 0.0, 1.0))
                base_bias = 0.35 + t * (self.config.base_long_bias - 0.35)
            pos = self.size_position(
                predicted_return=float(preds[i]),
                prediction_std=float(stds[i]),
                win_rate=win_rate,
                avg_win=avg_win,
                avg_loss=avg_loss,
                current_equity=equity,
                high_water_mark=hwm,
                base_long_bias=base_bias,
            )
            positions[i] = pos

            # synthetic path update for drawdown-aware scaling
            equity *= 1.0 + pos * preds[i]
            hwm = max(hwm, equity)

        return positions
