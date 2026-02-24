"""Timeframe helpers for interval-aware execution settings."""

from __future__ import annotations

def interval_to_minutes(interval: str) -> int | None:
    val = str(interval or "1d").strip().lower()
    if val.endswith("m"):
        try:
            return max(1, int(float(val[:-1])))
        except Exception:
            return None
    if val.endswith("h"):
        try:
            return max(1, int(float(val[:-1]) * 60.0))
        except Exception:
            return None
    return None


def is_intraday_interval(interval: str) -> bool:
    mins = interval_to_minutes(interval)
    return mins is not None and mins < 24 * 60


def bars_per_trading_day(interval: str) -> float:
    mins = interval_to_minutes(interval)
    if mins is None:
        return 1.0
    # US regular cash session is 6.5h ~= 390 minutes.
    return max(1.0, 390.0 / float(mins))


def annualization_factor_for_interval(interval: str) -> float:
    val = str(interval or "1d").strip().lower()
    if is_intraday_interval(val):
        return 252.0 * bars_per_trading_day(val)
    if val.endswith("wk"):
        return 52.0
    if val.endswith("mo"):
        if val == "3mo":
            return 4.0
        return 12.0
    if val == "5d":
        return 252.0 / 5.0
    return 252.0


def execution_profile(interval: str) -> dict[str, float | bool]:
    intraday = is_intraday_interval(interval)
    annual = annualization_factor_for_interval(interval)
    bars_day = bars_per_trading_day(interval)
    return {
        "is_intraday": intraday,
        "annualization_factor": float(max(1.0, annual)),
        "bars_per_day": float(max(1.0, bars_day)),
        "flat_at_day_end": False if intraday else False,
        "day_end_flatten_fraction": 0.0 if intraday else 0.0,
        "core_long": 0.92 if intraday else 0.95,
        "overlay_min": 0.14 if intraday else 0.10,
        "overlay_max": 0.95 if intraday else 0.88,
        "base_long_bias": 0.70 if intraday else 1.08,
        "bearish_risk_off": 0.06 if intraday else 0.15,
        # Regime template tuned for persistent intraday trends with short support in drawdowns.
        "regime_base_exposure": 0.95 if intraday else 0.95,
        "regime_bull_boost": 0.45 if intraday else 0.35,
        "regime_bear_penalty": 0.60 if intraday else 0.30,
        "regime_momentum_weight": 0.22 if intraday else 0.15,
        "regime_bear_short_boost": 0.30 if intraday else 0.20,
    }


def strategy_mix_for_window(num_points: int, *, is_intraday: bool) -> dict[str, float]:
    n = int(max(1, num_points))
    if is_intraday:
        # Intraday: short windows prefer faster adaptation, long windows more stability.
        if n <= 220:
            return {"overlay_multiplier": 1.18, "core_adjust": -0.08, "smooth_alpha": 0.36}
        if n >= 1200:
            return {"overlay_multiplier": 0.88, "core_adjust": 0.05, "smooth_alpha": 0.24}
        return {"overlay_multiplier": 1.0, "core_adjust": 0.0, "smooth_alpha": 0.30}
    # Daily: short windows need more active overlay; long windows keep stronger core.
    if n <= 120:
        return {"overlay_multiplier": 1.15, "core_adjust": -0.05, "smooth_alpha": 0.24}
    if n >= 650:
        return {"overlay_multiplier": 0.90, "core_adjust": 0.08, "smooth_alpha": 0.16}
    return {"overlay_multiplier": 1.0, "core_adjust": 0.0, "smooth_alpha": 0.20}
