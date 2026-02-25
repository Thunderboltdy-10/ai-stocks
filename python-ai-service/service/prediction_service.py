"""Prediction and backtest service layer for the GBM-first API."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from data.data_fetcher import fetch_stock_data
from data.feature_engineer import engineer_features
from data.target_engineering import prevent_lookahead_bias
from evaluation.execution_backtester import LongOnlyExecutionBacktester
from inference.intraday_signal import intraday_hybrid_positions
from inference.load_gbm_models import GBMModelBundle, load_gbm_models, predict_with_gbm
from inference.position_sizing import PositionSizer, PositionSizingConfig
from inference.regime_ensemble import (
    adaptive_overlay_weight,
    apply_short_safety_filter,
    blend_positions_with_core,
    combine_ml_and_regime,
    model_quality_gate_strict,
    regime_exposure_from_prices,
)
from inference.signal_policy import apply_direction_calibration, positions_from_policy
from inference.variant_router import resolve_model_variant
from utils.timeframe import execution_profile, interval_to_minutes, is_intraday_interval


def _to_daily_arithmetic_return(pred_log_return: np.ndarray, target_horizon_days: int) -> np.ndarray:
    horizon = max(1, int(target_horizon_days))
    return np.expm1(np.asarray(pred_log_return, dtype=float) / horizon)


def _business_days_after(date_str: str, n: int) -> List[str]:
    start = pd.Timestamp(date_str)
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start=start + BDay(1), periods=n)]


def _future_dates_after(last_date: str, n: int, interval: str) -> List[str]:
    if n <= 0:
        return []
    start = pd.Timestamp(last_date)
    if is_intraday_interval(interval):
        step_mins = interval_to_minutes(interval) or 60
        out: List[str] = []
        curr = start
        for _ in range(n):
            curr = curr + pd.Timedelta(minutes=step_mins)
            out.append(curr.strftime("%Y-%m-%d %H:%M:%S"))
        return out
    return _business_days_after(last_date, n)


def _fmt_timestamp(value, intraday: bool) -> str:
    ts = pd.to_datetime(value)
    return ts.strftime("%Y-%m-%d %H:%M:%S") if intraday else ts.strftime("%Y-%m-%d")


def _cost_to_decimal(raw: float) -> float:
    val = float(raw)
    # UI sends bps-style values (e.g., 0.5), CLI sends decimal pct (e.g., 0.0005).
    if val >= 0.05:
        return val / 10_000.0
    return val


def _as_float(value, default: float) -> float:
    if value is None:
        return float(default)
    try:
        if isinstance(value, str) and not value.strip():
            return float(default)
        out = float(value)
        return out if np.isfinite(out) else float(default)
    except Exception:
        return float(default)


def _as_bool(value, default: bool) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "y", "on"}:
            return True
        if v in {"0", "false", "no", "n", "off"}:
            return False
    return bool(value)


def _sanitize_for_json(value):
    if isinstance(value, dict):
        return {str(k): _sanitize_for_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, tuple):
        return [_sanitize_for_json(v) for v in value]
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if np.isfinite(x) else 0.0
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, np.ndarray):
        return [_sanitize_for_json(v) for v in value.tolist()]
    return value


def _load_bundle(
    symbol: str,
    model_variant: str = "gbm",
    *,
    data_interval: str = "1d",
    start: str | None = None,
    end: str | None = None,
) -> tuple[GBMModelBundle, str]:
    resolved_variant = resolve_model_variant(
        symbol=symbol,
        requested_variant=model_variant,
        data_interval=data_interval,
        start=start,
        end=end,
        default_variant="gbm",
    )
    model_dir = Path("saved_models") / symbol.upper() / resolved_variant
    bundle, metadata = load_gbm_models(symbol, model_dir=model_dir)
    if bundle is None:
        raise FileNotFoundError(f"Unable to load GBM models for {symbol} ({resolved_variant}): {metadata.get('errors')}")
    return bundle, resolved_variant


def _predict_series(bundle: GBMModelBundle, frame: pd.DataFrame) -> Dict[str, np.ndarray]:
    engineered = engineer_features(frame, symbol=bundle.symbol, include_sentiment=False)
    engineered = prevent_lookahead_bias(engineered, feature_cols=bundle.feature_columns)
    engineered[bundle.feature_columns] = engineered[bundle.feature_columns].fillna(0.0)
    missing = [c for c in bundle.feature_columns if c not in engineered.columns]
    if missing:
        raise ValueError(f"Missing model features: {missing[:10]}")

    X = engineered[bundle.feature_columns].to_numpy(dtype=np.float32)
    pred_parts = predict_with_gbm(bundle, X, return_components=True)
    target_horizon = int(bundle.metadata.get("target_horizon_days", 1))
    pred_parts = {k: _to_daily_arithmetic_return(v, target_horizon) for k, v in pred_parts.items()}

    if "xgb" in pred_parts and "lgb" in pred_parts:
        weights = bundle.metadata.get("ensemble_weights", {"xgb": 0.5, "lgb": 0.5})
        wx = float(weights.get("xgb", 0.5))
        wl = float(weights.get("lgb", 0.5))
        total = max(wx + wl, 1e-9)
        pred_ens = (wx * pred_parts["xgb"] + wl * pred_parts["lgb"]) / total
        pred_std = np.std(np.vstack([pred_parts["xgb"], pred_parts["lgb"]]), axis=0) + 1e-4
    else:
        pred_ens = list(pred_parts.values())[0]
        pred_std = pd.Series(pred_ens).rolling(20, min_periods=5).std().fillna(np.nanstd(pred_ens) + 1e-4).to_numpy()

    return {
        "pred_xgb": pred_parts.get("xgb", np.full(len(frame), np.nan)),
        "pred_lgb": pred_parts.get("lgb", np.full(len(frame), np.nan)),
        "pred_ens": pred_ens,
        "pred_std": pred_std,
    }


def _sizing_win_rate(bundle: GBMModelBundle) -> float:
    hold = bundle.metadata.get("holdout", {})
    holdout_dir = max(
        float(hold.get("ensemble", {}).get("dir_acc", 0.53)),
        float(hold.get("ensemble_calibrated", {}).get("dir_acc", 0.53)),
    )
    train_pos = float(bundle.metadata.get("target_distribution", {}).get("train_positive_pct", holdout_dir))
    return float(np.clip(max(holdout_dir, train_pos), 0.50, 0.60))


def _model_quality_gate(bundle: GBMModelBundle) -> bool:
    return model_quality_gate_strict(bundle.metadata)


def _regime_positions(
    close_series: pd.Series,
    profile: Dict[str, float | bool],
    max_long: float = 1.6,
    max_short: float = 0.25,
) -> np.ndarray:
    return regime_exposure_from_prices(
        close=close_series,
        max_long=max_long,
        max_short=max_short,
        vol_target_annual=0.40,
        min_scale=0.90,
        max_scale=1.60,
        periods_per_year=float(profile["annualization_factor"]),
        base_exposure=float(profile["regime_base_exposure"]),
        bull_boost=float(profile["regime_bull_boost"]),
        bear_penalty=float(profile["regime_bear_penalty"]),
        momentum_weight=float(profile["regime_momentum_weight"]),
        bear_short_boost=float(profile["regime_bear_short_boost"]),
    )


def _adaptive_ml_weights(bundle: GBMModelBundle) -> tuple[float, float]:
    hold = bundle.metadata.get("holdout", {})
    ens = hold.get("ensemble", {})
    cal = hold.get("ensemble_calibrated", {})
    dir_acc = max(float(ens.get("dir_acc", 0.0)), float(cal.get("dir_acc", 0.0)))
    pred_std = max(float(ens.get("pred_std", 0.0)), float(cal.get("pred_std", 0.0)))
    sharpe = max(float(ens.get("sharpe", 0.0)), float(cal.get("sharpe", 0.0)))
    wfe = float(bundle.metadata.get("wfe", 0.0))
    intraday = is_intraday_interval(str(bundle.metadata.get("data_interval", "1d")))

    if model_quality_gate_strict(bundle.metadata):
        return (0.24, 0.08) if intraday else (0.24, 0.08)
    if intraday and dir_acc >= 0.50 and pred_std >= 0.0015 and (wfe >= 5.0 or sharpe >= 0.0):
        return 0.16, 0.08
    if (not intraday) and dir_acc >= 0.50 and pred_std >= 0.003 and wfe >= 15.0:
        return 0.16, 0.08
    return (0.06, 0.04) if intraday else (0.08, 0.04)


def _action_from_delta(prev_pos: float, new_pos: float) -> str:
    delta = new_pos - prev_pos
    if abs(delta) < 1e-8:
        return "HOLD"
    if delta > 0:
        return "COVER" if prev_pos < 0 else "BUY"
    return "SHORT" if new_pos < 0 else "SELL"


def _build_trade_markers(
    dates: List[str],
    prices: np.ndarray,
    positions: np.ndarray,
    *,
    scope: str,
    segment: str,
) -> List[Dict]:
    markers: List[Dict] = []
    prev = 0.0
    for i, pos in enumerate(positions):
        delta = float(pos - prev)
        if abs(delta) >= 0.04:
            action = _action_from_delta(prev, float(pos))
            marker_type = "buy" if action in {"BUY", "COVER"} else "sell"
            markers.append(
                {
                    "date": dates[i],
                    "price": float(prices[i]),
                    "type": marker_type,
                    "shares": float(abs(delta) * 120),
                    "confidence": float(min(1.0, abs(delta))),
                    "scope": scope,
                    "segment": segment,
                    "explanation": action,
                }
            )
        prev = pos
    return markers


def _downsample_records(records: List[Dict], max_points: int = 700) -> List[Dict]:
    if len(records) <= max_points:
        return records
    step = max(1, len(records) // max_points)
    sampled = records[::step]
    if sampled[-1] != records[-1]:
        sampled.append(records[-1])
    return sampled


def _build_backtest_diagnostics(
    *,
    dates: List[str],
    prices: np.ndarray,
    bt,
    trade_log: List[Dict],
    intraday: bool,
    annualization: float,
) -> Dict:
    dt = pd.to_datetime(pd.Series(dates), errors="coerce")
    dt = dt.ffill().bfill()

    equity = np.asarray(bt.equity_curve, dtype=float)
    buy_hold = np.asarray(bt.buy_hold_equity, dtype=float)
    positions = np.asarray(bt.effective_positions, dtype=float)
    strat_ret = pd.Series(np.asarray(bt.daily_returns, dtype=float), index=dt).fillna(0.0)
    bh_ret = pd.Series(pd.Series(prices, index=dt).pct_change().fillna(0.0).to_numpy(dtype=float), index=dt)
    drawdown = (equity / np.maximum.accumulate(equity)) - 1.0 if len(equity) else np.array([], dtype=float)

    roll = 48 if intraday else 20
    ann = max(1.0, float(annualization))
    rolling_mean = strat_ret.rolling(roll, min_periods=max(8, roll // 3)).mean()
    rolling_std = strat_ret.rolling(roll, min_periods=max(8, roll // 3)).std().replace(0.0, np.nan)
    rolling_sharpe = (rolling_mean / rolling_std * np.sqrt(ann)).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    strat_cum = pd.Series(equity, index=dt) / max(float(equity[0]) if len(equity) else 1.0, 1e-9) - 1.0
    bh_cum = pd.Series(buy_hold, index=dt) / max(float(buy_hold[0]) if len(buy_hold) else 1.0, 1e-9) - 1.0
    rolling_alpha = (strat_cum - bh_cum).fillna(0.0)

    rolling_records = [
        {
            "date": dt.iloc[i].strftime("%Y-%m-%d %H:%M:%S") if intraday else dt.iloc[i].strftime("%Y-%m-%d"),
            "equity": float(equity[i]),
            "buyHoldEquity": float(buy_hold[i]),
            "drawdown": float(drawdown[i]) if i < len(drawdown) else 0.0,
            "rollingSharpe": float(rolling_sharpe.iloc[i]),
            "rollingAlpha": float(rolling_alpha.iloc[i]),
            "position": float(positions[i]) if i < len(positions) else 0.0,
            "strategyReturn": float(strat_ret.iloc[i]),
            "buyHoldReturn": float(bh_ret.iloc[i]),
        }
        for i in range(len(dt))
    ]
    rolling_records = _downsample_records(rolling_records, max_points=700)

    perf_df = pd.DataFrame({"strategy": strat_ret, "buy_hold": bh_ret})
    monthly = []
    try:
        monthly_ret = (1.0 + perf_df).groupby(perf_df.index.to_period("M")).prod() - 1.0
        for idx, row in monthly_ret.tail(36).iterrows():
            s = float(row.get("strategy", 0.0))
            b = float(row.get("buy_hold", 0.0))
            monthly.append(
                {
                    "period": str(idx),
                    "strategyReturn": s,
                    "buyHoldReturn": b,
                    "alpha": s - b,
                }
            )
    except Exception:
        monthly = []

    action_df = pd.DataFrame(trade_log) if trade_log else pd.DataFrame()
    action_breakdown: List[Dict] = []
    if not action_df.empty and "action" in action_df.columns:
        grouped = action_df.groupby(action_df["action"].astype(str))
        for action, rows in grouped:
            pnl = pd.to_numeric(rows.get("pnl", 0.0), errors="coerce").fillna(0.0)
            action_breakdown.append(
                {
                    "action": str(action),
                    "count": int(len(rows)),
                    "winRate": float((pnl > 0.0).mean()) if len(pnl) else 0.0,
                    "avgPnl": float(pnl.mean()) if len(pnl) else 0.0,
                    "totalPnl": float(pnl.sum()) if len(pnl) else 0.0,
                }
            )
        action_breakdown.sort(key=lambda x: x["count"], reverse=True)

    hourly: List[Dict] = []
    if intraday and len(dt) > 0:
        hours = pd.Series(dt.dt.hour.to_numpy(dtype=int), index=dt)
        ret_by_hour = strat_ret.groupby(hours).mean()
        pos_by_hour = pd.Series(positions, index=dt).groupby(hours).mean() if len(positions) else pd.Series(dtype=float)
        trade_counts: Dict[int, int] = {}
        if not action_df.empty and "date" in action_df.columns:
            th = pd.to_datetime(action_df["date"], errors="coerce").dt.hour.dropna().astype(int)
            trade_counts = th.value_counts().to_dict()
        for h in range(24):
            hourly.append(
                {
                    "hour": h,
                    "avgStrategyReturn": float(ret_by_hour.get(h, 0.0)),
                    "avgPosition": float(pos_by_hour.get(h, 0.0)) if not pos_by_hour.empty else 0.0,
                    "tradeCount": int(trade_counts.get(h, 0)),
                }
            )

    total_ret = strat_ret.to_numpy(dtype=float)
    losses = total_ret[total_ret < 0.0]
    loss_cut = float(np.nanquantile(losses, 0.05)) if len(losses) else 0.0
    risk = {
        "exposureMean": float(np.nanmean(np.abs(positions))) if len(positions) else 0.0,
        "exposureStd": float(np.nanstd(positions)) if len(positions) else 0.0,
        "tailLossP95": loss_cut,
        "cvar95": float(np.nanmean(losses[losses <= loss_cut])) if len(losses) else 0.0,
        "bestBar": float(np.nanmax(total_ret)) if len(total_ret) else 0.0,
        "worstBar": float(np.nanmin(total_ret)) if len(total_ret) else 0.0,
    }

    return {
        "rolling": rolling_records,
        "monthly": monthly,
        "actionBreakdown": action_breakdown,
        "hourly": hourly,
        "risk": risk,
    }


def generate_prediction(payload: Dict) -> Dict:
    symbol = str(payload.get("symbol", "AAPL")).upper()
    horizon = int(payload.get("horizon", 5))
    days_on_chart = int(payload.get("daysOnChart", 120))
    model_variant = str(payload.get("modelVariant", "gbm")).strip() or "gbm"
    requested_model_variant = model_variant
    data_interval = str(payload.get("dataInterval", "1d"))
    data_period = str(payload.get("dataPeriod", "10y"))

    fusion = payload.get("fusion", {}) if isinstance(payload.get("fusion"), dict) else {}
    fusion_mode = str(fusion.get("mode", "gbm_only"))

    floors = payload.get("confidenceFloors", {}) if isinstance(payload.get("confidenceFloors"), dict) else {}
    buy_thr = float(floors.get("buy", 0.30))
    sell_thr = float(floors.get("sell", 0.45))

    try:
        bundle, model_variant = _load_bundle(
            symbol,
            model_variant=model_variant,
            data_interval=data_interval,
        )
    except FileNotFoundError:
        if model_variant != "gbm":
            bundle, model_variant = _load_bundle(
                symbol,
                model_variant="gbm",
                data_interval=data_interval,
            )
            model_variant = "gbm"
        else:
            raise

    data_interval = str(payload.get("dataInterval", bundle.metadata.get("data_interval", data_interval)))
    data_period = str(payload.get("dataPeriod", bundle.metadata.get("data_period", data_period)))
    profile = execution_profile(data_interval)
    intraday = bool(profile["is_intraday"])
    max_long = float(payload.get("maxLong", 1.4 if intraday else 1.6))
    max_short = float(payload.get("maxShort", 0.2 if intraday else 0.25))
    time_fmt = "%Y-%m-%d %H:%M:%S" if intraday else "%Y-%m-%d"

    raw = fetch_stock_data(symbol=symbol, period=data_period, interval=data_interval)
    preds = _predict_series(bundle, raw)
    calibrated_pred, direction_conf = apply_direction_calibration(
        preds["pred_ens"],
        bundle.metadata.get("direction_calibrator"),
    )
    execution_policy = bundle.metadata.get("execution_calibration", {})
    recommended_min_change = (
        float(max(0.0, execution_policy.get("min_position_change", 0.0)))
        if isinstance(execution_policy, dict) and bool(execution_policy.get("enabled", False))
        else 0.0
    )
    if intraday:
        recommended_min_change = max(recommended_min_change, 0.03)

    ml_gate = _model_quality_gate(bundle)
    ml_weight, fallback_ml_weight = _adaptive_ml_weights(bundle)

    sizer = PositionSizer(
        PositionSizingConfig(
            max_long=max_long,
            max_short=max_short,
            base_long_bias=float(profile["base_long_bias"]),
            bearish_risk_off=float(profile["bearish_risk_off"]),
        )
    )
    if isinstance(execution_policy, dict) and bool(execution_policy.get("enabled", False)):
        ml_positions = positions_from_policy(
            calibrated_pred,
            preds["pred_std"],
            execution_policy,
            max_long=max_long,
            max_short=max_short,
        )
    elif intraday:
        ml_positions = intraday_hybrid_positions(
            close=raw["Close"],
            predicted_returns=calibrated_pred,
            prediction_stds=preds["pred_std"],
            max_long=max_long,
            max_short=max_short,
            cost_buffer=0.0002,
        )
    else:
        ml_positions = sizer.size_batch(
            predicted_returns=calibrated_pred,
            prediction_stds=preds["pred_std"],
            win_rate=_sizing_win_rate(bundle),
            avg_win=0.012,
            avg_loss=0.010,
        )
    ml_positions = np.clip(ml_positions, -max_short, max_long)

    regime_positions = _regime_positions(
        raw["Close"],
        profile=profile,
        max_long=max_long,
        max_short=max_short,
    )
    positions = combine_ml_and_regime(
        regime_positions=regime_positions,
        ml_positions=ml_positions,
        ml_gate_passed=ml_gate,
        ml_weight=ml_weight,
        fallback_ml_weight=fallback_ml_weight,
        max_long=max_long,
        max_short=max_short,
    )
    overlay_weights = adaptive_overlay_weight(
        close=raw["Close"],
        predicted_returns=calibrated_pred,
        lookback=60,
        min_weight=(float(profile["overlay_min"]) if ml_gate else 0.02),
        max_weight=(float(profile["overlay_max"]) if ml_gate else 0.35),
    )
    positions = blend_positions_with_core(
        active_positions=positions,
        overlay_weights=overlay_weights,
        core_long=float(profile["core_long"]),
        max_long=max_long,
        max_short=max_short,
        smooth_alpha=0.30 if intraday else 0.20,
    )
    positions = apply_short_safety_filter(
        close=raw["Close"],
        positions=positions,
        predicted_returns=calibrated_pred,
        max_short=max_short,
        interval=data_interval,
    )

    close = raw["Close"].to_numpy(dtype=float)
    dates = [pd.to_datetime(d).strftime(time_fmt) for d in pd.to_datetime(raw.index)]
    actual_returns = raw["Close"].pct_change().fillna(0.0).to_numpy(dtype=float)

    # Trim to chart window.
    keep = min(days_on_chart, len(raw))
    close = close[-keep:]
    dates = dates[-keep:]
    actual_returns = actual_returns[-keep:]
    pred_ret = np.asarray(calibrated_pred[-keep:], dtype=float)
    pos = np.asarray(positions[-keep:], dtype=float)

    predicted_prices = close * (1.0 + pred_ret)
    trade_markers = _build_trade_markers(dates, close, pos, scope="prediction", segment="history")

    # Forward projection: mean-reverting return path + dynamic position recompute.
    rolling_mean = pd.Series(pred_ret).ewm(span=20, adjust=False).mean().to_numpy(dtype=float)
    drift = float(rolling_mean[-1]) if len(rolling_mean) else 0.0
    vol = float(np.nanstd(pred_ret[-60:])) if len(pred_ret) else 0.005
    vol = max(vol, 0.0025)

    future_returns = np.zeros(horizon, dtype=float)
    prev_ret = float(pred_ret[-1]) if len(pred_ret) else 0.0
    for step in range(horizon):
        mean_reversion = -0.20 * (prev_ret - drift)
        drift_push = 0.35 * drift
        momentum = 0.45 * prev_ret
        simulated_ret = momentum + drift_push + mean_reversion
        simulated_ret = float(np.clip(simulated_ret, -0.04, 0.04))
        future_returns[step] = simulated_ret
        prev_ret = simulated_ret

    latest_price = float(close[-1])
    future_prices = latest_price * np.cumprod(1.0 + future_returns)
    future_dates = _future_dates_after(dates[-1], horizon, data_interval)

    # Recompute dynamic future positions from synthetic path.
    synthetic_close = np.concatenate([close, future_prices])
    regime_future = _regime_positions(
        pd.Series(synthetic_close),
        profile=profile,
        max_long=max_long,
        max_short=max_short,
    )[-horizon:]

    future_std = np.full(horizon, max(vol, float(np.nanmean(preds["pred_std"][-20:])) if len(preds["pred_std"]) else vol))
    if isinstance(execution_policy, dict) and bool(execution_policy.get("enabled", False)):
        ml_future = positions_from_policy(
            future_returns,
            future_std,
            execution_policy,
            max_long=max_long,
            max_short=max_short,
        )
    elif intraday:
        ml_future = intraday_hybrid_positions(
            close=pd.Series(synthetic_close[-horizon:], index=pd.RangeIndex(horizon)),
            predicted_returns=future_returns,
            prediction_stds=future_std,
            max_long=max_long,
            max_short=max_short,
            cost_buffer=0.0002,
        )
    else:
        ml_future = sizer.size_batch(
            predicted_returns=future_returns,
            prediction_stds=future_std,
            win_rate=_sizing_win_rate(bundle),
            avg_win=0.012,
            avg_loss=0.010,
        )
    ml_future = np.clip(ml_future, -max_short, max_long)

    future_positions = combine_ml_and_regime(
        regime_positions=regime_future,
        ml_positions=ml_future,
        ml_gate_passed=ml_gate,
        ml_weight=ml_weight,
        fallback_ml_weight=fallback_ml_weight,
        max_long=max_long,
        max_short=max_short,
    )
    last_overlay = float(overlay_weights[-1]) if len(overlay_weights) else (float(profile["overlay_min"]) if ml_gate else 0.05)
    future_overlay = np.linspace(last_overlay, max(0.02, last_overlay * 0.70), horizon)
    future_positions = blend_positions_with_core(
        active_positions=future_positions,
        overlay_weights=future_overlay,
        core_long=float(profile["core_long"]),
        max_long=max_long,
        max_short=max_short,
        smooth_alpha=0.35 if intraday else 0.25,
    )
    future_positions = apply_short_safety_filter(
        close=pd.Series(synthetic_close),
        positions=np.concatenate([pos, future_positions]),
        predicted_returns=np.concatenate([pred_ret, future_returns]),
        max_short=max_short,
        interval=data_interval,
    )[-horizon:]
    future_actions = []
    prev_future = float(pos[-1]) if len(pos) else 0.0
    for i in range(horizon):
        action = _action_from_delta(prev_future, float(future_positions[i]))
        future_actions.append(
            {
                "date": future_dates[i],
                "action": action,
                "price": float(future_prices[i]),
                "targetPosition": float(future_positions[i]),
            }
        )
        prev_future = float(future_positions[i])

    forecast_markers = _build_trade_markers(
        future_dates,
        future_prices,
        future_positions,
        scope="prediction",
        segment="forecast",
    )
    trade_markers.extend(forecast_markers)

    result = {
        "symbol": symbol,
        "dates": dates,
        "prices": [float(x) for x in close],
        "actualReturns": [float(x) for x in actual_returns],
        "predictedPrices": [float(x) for x in predicted_prices],
        "predictedReturns": [float(x) for x in pred_ret],
        "fusedPositions": [float(x) for x in pos],
        "classifierProbabilities": [],
        "tradeMarkers": trade_markers,
        "overlays": [
            {
                "type": "predicted-path",
                "points": [{"date": d, "value": float(v)} for d, v in zip(dates, predicted_prices)],
            }
        ],
        "candles": [
            {
                "date": pd.to_datetime(d).strftime(time_fmt),
                "open": float(o),
                "high": float(h),
                "low": float(l),
                "close": float(c),
                "volume": float(v),
            }
            for d, o, h, l, c, v in zip(
                pd.to_datetime(raw.index[-keep:]),
                raw["Open"].to_numpy()[-keep:],
                raw["High"].to_numpy()[-keep:],
                raw["Low"].to_numpy()[-keep:],
                raw["Close"].to_numpy()[-keep:],
                raw["Volume"].to_numpy()[-keep:],
            )
        ],
        "forecast": {
            "dates": future_dates,
            "prices": [float(x) for x in future_prices],
            "returns": [float(x) for x in future_returns],
            "positions": [float(x) for x in future_positions],
            "actions": future_actions,
        },
        "metadata": {
            "modelId": f"{symbol.lower()}-{model_variant}",
            "fusionMode": fusion_mode,
            "buyThreshold": buy_thr,
            "sellThreshold": sell_thr,
            "horizon": horizon,
            "modelQualityGatePassed": ml_gate,
            "executionMode": "signed_inventory",
            "maxLong": max_long,
            "maxShort": max_short,
            "mlOverlayWeight": ml_weight if ml_gate else fallback_ml_weight,
            "directionConfidenceMean": float(np.mean(np.abs(direction_conf))),
            "modelVariantRequested": requested_model_variant,
            "modelVariant": model_variant,
            "dataInterval": data_interval,
            "dataPeriod": data_period,
            "isIntraday": intraday,
            "annualizationFactor": float(profile["annualization_factor"]),
            "flatAtDayEnd": bool(profile["flat_at_day_end"]),
            "dayEndFlattenFraction": float(
                profile.get("day_end_flatten_fraction", 1.0) if bool(profile["flat_at_day_end"]) else 0.0
            ),
            "recommendedMinPositionChange": recommended_min_change,
        },
    }

    return _sanitize_for_json(result)


def _forward_simulate(prediction: Dict, params: Dict) -> Dict | None:
    forecast = prediction.get("forecast", {})
    if not isinstance(forecast, dict):
        return None

    dates = forecast.get("dates", [])
    prices = np.asarray(forecast.get("prices", []), dtype=float)
    positions = np.asarray(forecast.get("positions", []), dtype=float)

    if len(dates) == 0 or len(prices) == 0 or len(prices) != len(positions):
        return None

    max_long = _as_float(params.get("maxLong"), _as_float(prediction.get("metadata", {}).get("maxLong"), 1.6))
    max_short = _as_float(params.get("maxShort"), _as_float(prediction.get("metadata", {}).get("maxShort"), 0.25))
    interval = str(params.get("dataInterval", prediction.get("metadata", {}).get("dataInterval", "1d")))
    profile = execution_profile(interval)
    intraday = bool(profile["is_intraday"])
    annualization = _as_float(params.get("annualizationFactor"), _as_float(profile["annualization_factor"], 252.0))
    flat_at_day_end = _as_bool(params.get("flatAtDayEnd"), bool(profile["flat_at_day_end"]))
    day_end_flatten_fraction = _as_float(
        params.get("dayEndFlattenFraction"),
        _as_float(
            prediction.get("metadata", {}).get("dayEndFlattenFraction"),
            _as_float(profile.get("day_end_flatten_fraction", 1.0), 1.0),
        ),
    )
    backtester = LongOnlyExecutionBacktester(
        initial_capital=_as_float(params.get("initialCapital"), 10_000.0),
        commission_pct=_cost_to_decimal(_as_float(params.get("commission"), 0.0)),
        slippage_pct=_cost_to_decimal(_as_float(params.get("slippage"), 0.0)),
        min_position_change=max(
            _as_float(params.get("minPositionChange"), 0.03 if intraday else 0.0),
            _as_float(prediction.get("metadata", {}).get("recommendedMinPositionChange"), 0.0),
        ),
    )
    bt = backtester.backtest(
        dates=np.asarray(dates),
        prices=prices,
        target_positions=positions,
        max_long=max_long,
        max_short=max_short,
        apply_position_lag=False,
        annualization_factor=annualization,
        flat_at_day_end=flat_at_day_end,
        day_end_flatten_fraction=day_end_flatten_fraction,
    )

    action_rows = []
    markers = []
    for i, trade in enumerate(bt.trade_log):
        raw_action = str(trade.get("action", "BUY")).upper()
        marker_type = "buy" if raw_action in {"BUY", "COVER", "COVER_BUY"} else "sell"
        action_rows.append(
            {
                "id": f"fwd-{i+1}",
                "date": _fmt_timestamp(trade.get("date"), intraday),
                "action": raw_action,
                "price": float(trade.get("price", 0.0)),
                "shares": float(trade.get("shares", 0.0)),
                "targetWeight": float(trade.get("target_weight", 0.0)),
                "effectiveWeightAfter": float(trade.get("effective_weight_after", 0.0)),
                "notes": str(trade.get("blocked_reason") or "executed"),
            }
        )
        markers.append(
            {
                "date": _fmt_timestamp(trade.get("date"), intraday),
                "price": float(trade.get("price", 0.0)),
                "type": marker_type,
                "shares": float(trade.get("shares", 0.0)),
                "confidence": float(min(1.0, abs(float(trade.get("signed_shares", 0.0))))),
                "scope": "backtest",
                "segment": "forecast",
                "explanation": raw_action,
            }
        )

    return {
        "dates": [str(d) for d in dates],
        "prices": [float(x) for x in prices],
        "equityCurve": [float(x) for x in bt.equity_curve],
        "sharpe": float(bt.metrics.get("sharpe", 0.0)),
        "maxDrawdown": float(bt.metrics.get("max_drawdown", 0.0)),
        "trades": int(len(bt.trade_log)),
        "actions": action_rows,
        "markers": markers,
        "totalCosts": float(bt.metrics.get("total_transaction_costs", 0.0)),
        "borrowFee": float(bt.metrics.get("total_borrow_fee", 0.0)),
    }


def run_backtest(prediction: Dict, params: Dict) -> Dict:
    dates = prediction.get("dates", [])
    prices = np.asarray(prediction.get("prices", []), dtype=float)
    target_positions = np.asarray(prediction.get("fusedPositions", []), dtype=float)

    if len(prices) == 0 or len(target_positions) == 0 or len(prices) != len(target_positions):
        raise ValueError("Invalid prediction payload for backtest")

    max_long = _as_float(params.get("maxLong"), _as_float(prediction.get("metadata", {}).get("maxLong"), 1.6))
    max_short = _as_float(params.get("maxShort"), _as_float(prediction.get("metadata", {}).get("maxShort"), 0.25))
    interval = str(params.get("dataInterval", prediction.get("metadata", {}).get("dataInterval", "1d")))
    profile = execution_profile(interval)
    intraday = bool(profile["is_intraday"])
    annualization = _as_float(params.get("annualizationFactor"), _as_float(profile["annualization_factor"], 252.0))
    flat_at_day_end = _as_bool(params.get("flatAtDayEnd"), bool(profile["flat_at_day_end"]))
    day_end_flatten_fraction = _as_float(
        params.get("dayEndFlattenFraction"),
        _as_float(
            prediction.get("metadata", {}).get("dayEndFlattenFraction"),
            _as_float(profile.get("day_end_flatten_fraction", 1.0), 1.0),
        ),
    )
    backtester = LongOnlyExecutionBacktester(
        initial_capital=_as_float(params.get("initialCapital"), 10_000.0),
        commission_pct=_cost_to_decimal(_as_float(params.get("commission"), 0.0)),
        slippage_pct=_cost_to_decimal(_as_float(params.get("slippage"), 0.0)),
        min_position_change=max(
            _as_float(params.get("minPositionChange"), 0.03 if intraday else 0.0),
            _as_float(prediction.get("metadata", {}).get("recommendedMinPositionChange"), 0.0),
        ),
    )
    target_positions_clipped = np.clip(target_positions, -max_short, max_long)
    bt = backtester.backtest(
        dates=np.asarray(dates),
        prices=prices,
        target_positions=target_positions_clipped,
        max_long=max_long,
        max_short=max_short,
        apply_position_lag=True,
        annualization_factor=annualization,
        flat_at_day_end=flat_at_day_end,
        day_end_flatten_fraction=day_end_flatten_fraction,
    )

    equity = bt.equity_curve
    running_max = np.maximum.accumulate(equity)
    drawdown = np.where(running_max > 0, equity / running_max - 1.0, 0.0)

    equity_curve = [
        {"date": dates[i], "equity": float(equity[i]), "drawdown": float(drawdown[i])}
        for i in range(len(equity))
    ]
    price_series = [{"date": dates[i], "price": float(prices[i])} for i in range(len(prices))]

    returns = np.zeros_like(prices)
    if len(prices) > 1:
        returns[1:] = prices[1:] / prices[:-1] - 1.0

    trade_log = []
    annotations = []
    for i, trade in enumerate(bt.trade_log):
        idx = int(trade.get("index", 0))
        idx = max(0, min(idx, len(dates) - 1))
        action = str(trade.get("action", "BUY")).upper()
        chart_action = "buy" if action in {"BUY", "COVER", "COVER_BUY"} else "sell"
        position_val = float(bt.effective_positions[idx]) if len(bt.effective_positions) > idx else 0.0
        cumulative_pnl = float(equity[idx] - _as_float(params.get("initialCapital"), 10_000.0))
        pnl = 0.0 if idx == 0 else float(equity[idx] - equity[idx - 1])
        notes = str(trade.get("blocked_reason") or "inventory-aware execution")

        record = {
            "id": f"trade-{i+1}",
            "date": _fmt_timestamp(trade.get("date"), intraday),
            "action": action,
            "price": float(trade.get("price", 0.0)),
            "shares": float(trade.get("shares", 0.0)),
            "position": position_val,
            "pnl": pnl,
            "cumulativePnl": cumulative_pnl,
            "commission": float(trade.get("commission", 0.0)),
            "slippage": float(trade.get("slippage", 0.0)),
            "explanation": {
                "classifierProb": 0.0,
                "regressorReturn": 0.0,
                "fusionMode": str(prediction.get("metadata", {}).get("fusionMode", "gbm_only")),
                "notes": notes or "inventory-aware execution",
            },
        }
        trade_log.append(record)
        annotations.append(
            {
                "date": record["date"],
                "price": record["price"],
                "type": chart_action,
                "shares": record["shares"],
                "confidence": float(min(1.0, abs(position_val))),
                "scope": "backtest",
                "explanation": action,
            }
        )

    pred_returns = np.asarray(prediction.get("predictedReturns", []), dtype=float)
    if len(pred_returns) != len(returns):
        pred_returns = np.zeros_like(returns)

    corr_val = float(np.corrcoef(pred_returns, returns)[0, 1]) if len(returns) > 3 else 0.0
    if not np.isfinite(corr_val):
        corr_val = 0.0

    metrics = {
        "cumulativeReturn": float(bt.metrics.get("cum_return", 0.0)),
        "cumulativeReturnPct": float(bt.metrics.get("cum_return", 0.0) * 100.0),
        "buyHoldReturnPct": float(bt.metrics.get("buy_hold_cum_return", 0.0) * 100.0),
        "excessReturnPct": float(bt.metrics.get("alpha", 0.0) * 100.0),
        "sharpeRatio": float(bt.metrics.get("sharpe", 0.0)),
        "sortinoRatio": float(bt.metrics.get("sortino", 0.0)),
        "calmarRatio": float(bt.metrics.get("calmar", 0.0)),
        "maxDrawdown": float(bt.metrics.get("max_drawdown", 0.0)),
        "winRate": float(bt.metrics.get("win_rate", 0.0)),
        "averageTradeProfit": float(np.mean([t["pnl"] for t in trade_log])) if trade_log else 0.0,
        "totalTrades": len(trade_log),
        "directionalAccuracy": float(np.mean(np.sign(pred_returns) == np.sign(returns))),
        "correlation": corr_val,
        "rmse": float(np.sqrt(np.mean((pred_returns - returns) ** 2))) if len(returns) else 0.0,
        "smape": float(
            np.mean(
                2.0 * np.abs(pred_returns - returns)
                / (np.abs(pred_returns) + np.abs(returns) + 1e-9)
            )
        ) if len(returns) else 0.0,
        "transactionCosts": float(bt.metrics.get("total_transaction_costs", 0.0)),
        "borrowFee": float(bt.metrics.get("total_borrow_fee", 0.0)),
    }

    buy_hold_equity = [
        {"date": dates[i], "equity": float(bt.buy_hold_equity[i])}
        for i in range(len(dates))
    ]

    csv_rows = ["date,price,target_position,effective_position,equity"]
    for i in range(len(dates)):
        csv_rows.append(
            f"{dates[i]},{prices[i]:.6f},{target_positions_clipped[i]:.6f},{bt.effective_positions[i]:.6f},{equity[i]:.6f}"
        )

    forward_sim = None
    if bool(params.get("enableForwardSim", False)):
        forward_sim = _forward_simulate(prediction, params)

    diagnostics = _build_backtest_diagnostics(
        dates=[str(d) for d in dates],
        prices=prices,
        bt=bt,
        trade_log=trade_log,
        intraday=intraday,
        annualization=annualization,
    )

    return _sanitize_for_json({
        "equityCurve": equity_curve,
        "priceSeries": price_series,
        "tradeLog": trade_log,
        "metrics": metrics,
        "diagnostics": diagnostics,
        "annotations": annotations,
        "buyHoldEquity": buy_hold_equity,
        "csv": "\n".join(csv_rows),
        "forwardSimulation": forward_sim,
    })
