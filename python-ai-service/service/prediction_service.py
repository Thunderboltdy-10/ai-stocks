"""Prediction and backtest service layer for the GBM-first API."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from data.data_fetcher import fetch_stock_data
from data.feature_engineer import engineer_features
from data.target_engineering import prevent_lookahead_bias
from evaluation.execution_backtester import LongOnlyExecutionBacktester
from inference.load_gbm_models import GBMModelBundle, load_gbm_models, predict_with_gbm
from inference.position_sizing import PositionSizer
from inference.regime_ensemble import (
    combine_ml_and_regime,
    model_quality_gate_strict,
    regime_exposure_from_prices,
)


def _to_daily_arithmetic_return(pred_log_return: np.ndarray, target_horizon_days: int) -> np.ndarray:
    horizon = max(1, int(target_horizon_days))
    return np.expm1(np.asarray(pred_log_return, dtype=float) / horizon)


def _business_days_after(date_str: str, n: int) -> List[str]:
    start = pd.Timestamp(date_str)
    return [d.strftime("%Y-%m-%d") for d in pd.bdate_range(start=start + BDay(1), periods=n)]


def _load_bundle(symbol: str) -> GBMModelBundle:
    bundle, metadata = load_gbm_models(symbol)
    if bundle is None:
        raise FileNotFoundError(f"Unable to load GBM models for {symbol}: {metadata.get('errors')}")
    return bundle


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
    holdout_dir = float(bundle.metadata.get("holdout", {}).get("ensemble", {}).get("dir_acc", 0.53))
    train_pos = float(bundle.metadata.get("target_distribution", {}).get("train_positive_pct", holdout_dir))
    return float(np.clip(max(holdout_dir, train_pos), 0.50, 0.60))


def _model_quality_gate(bundle: GBMModelBundle) -> bool:
    return model_quality_gate_strict(bundle.metadata)


def _regime_positions(close_series: pd.Series, max_long: float = 1.6) -> np.ndarray:
    return regime_exposure_from_prices(
        close=close_series,
        max_long=max_long,
        vol_target_annual=0.40,
        min_scale=0.90,
        max_scale=1.60,
    )


def _build_trade_markers(dates: List[str], prices: np.ndarray, positions: np.ndarray) -> List[Dict]:
    markers: List[Dict] = []
    prev = 0.0
    for i, pos in enumerate(positions):
        delta = float(pos - prev)
        if abs(delta) >= 0.05:
            marker_type = "buy" if delta > 0 else "sell"
            markers.append(
                {
                    "date": dates[i],
                    "price": float(prices[i]),
                    "type": marker_type,
                    "shares": float(abs(delta) * 100),
                    "confidence": float(min(1.0, abs(delta))),
                    "scope": "prediction",
                    "segment": "history",
                }
            )
        prev = pos
    return markers


def generate_prediction(payload: Dict) -> Dict:
    symbol = str(payload.get("symbol", "AAPL")).upper()
    horizon = int(payload.get("horizon", 5))
    days_on_chart = int(payload.get("daysOnChart", 120))

    fusion = payload.get("fusion", {}) if isinstance(payload.get("fusion"), dict) else {}
    fusion_mode = str(fusion.get("mode", "gbm_only"))

    floors = payload.get("confidenceFloors", {}) if isinstance(payload.get("confidenceFloors"), dict) else {}
    buy_thr = float(floors.get("buy", 0.30))
    sell_thr = float(floors.get("sell", 0.45))

    bundle = _load_bundle(symbol)
    raw = fetch_stock_data(symbol=symbol, period="10y")
    preds = _predict_series(bundle, raw)

    ml_gate = _model_quality_gate(bundle)
    ml_positions = None
    if ml_gate:
        sizer = PositionSizer()
        ml_positions = sizer.size_batch(
            predicted_returns=preds["pred_ens"],
            prediction_stds=preds["pred_std"],
            win_rate=_sizing_win_rate(bundle),
            avg_win=0.012,
            avg_loss=0.010,
        )
        ml_positions = np.clip(ml_positions, 0.0, 1.6)

    regime_positions = _regime_positions(raw["Close"], max_long=1.6)
    positions = combine_ml_and_regime(
        regime_positions=regime_positions,
        ml_positions=ml_positions,
        ml_gate_passed=ml_gate,
        ml_weight=0.20,
        max_long=1.6,
    )

    close = raw["Close"].to_numpy(dtype=float)
    dates = [d.strftime("%Y-%m-%d") for d in pd.to_datetime(raw.index)]
    actual_returns = raw["Close"].pct_change().fillna(0.0).to_numpy(dtype=float)

    # Trim to chart window.
    keep = min(days_on_chart, len(raw))
    close = close[-keep:]
    dates = dates[-keep:]
    actual_returns = actual_returns[-keep:]
    pred_ret = np.asarray(preds["pred_ens"][-keep:], dtype=float)
    pos = np.asarray(positions[-keep:], dtype=float)

    predicted_prices = close * (1.0 + pred_ret)
    trade_markers = _build_trade_markers(dates, close, pos)

    # Simple forward projection from latest prediction.
    latest_price = float(close[-1])
    latest_pred = float(pred_ret[-1])
    future_returns = np.full(horizon, latest_pred, dtype=float)
    future_prices = latest_price * np.cumprod(1.0 + future_returns)
    future_positions = np.full(horizon, float(pos[-1]), dtype=float)
    future_dates = _business_days_after(dates[-1], horizon)

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
                "date": d.strftime("%Y-%m-%d"),
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
        },
        "metadata": {
            "modelId": f"{symbol.lower()}-gbm",
            "fusionMode": fusion_mode,
            "buyThreshold": buy_thr,
            "sellThreshold": sell_thr,
            "horizon": horizon,
            "modelQualityGatePassed": ml_gate,
            "executionMode": "long_only_inventory",
            "mlOverlayWeight": 0.20 if ml_gate else 0.0,
        },
    }

    return result


def _forward_simulate(prediction: Dict, params: Dict) -> Dict | None:
    forecast = prediction.get("forecast", {})
    if not isinstance(forecast, dict):
        return None

    dates = forecast.get("dates", [])
    prices = np.asarray(forecast.get("prices", []), dtype=float)
    positions = np.asarray(forecast.get("positions", []), dtype=float)

    if len(dates) == 0 or len(prices) == 0 or len(prices) != len(positions):
        return None

    max_long = float(params.get("maxLong", 1.6))
    backtester = LongOnlyExecutionBacktester(
        initial_capital=float(params.get("initialCapital", 10_000.0)),
        commission_pct=float(params.get("commission", 0.0)),
        slippage_pct=float(params.get("slippage", 0.0)),
        min_position_change=float(params.get("minPositionChange", 0.0)),
    )
    bt = backtester.backtest(
        dates=np.asarray(dates),
        prices=prices,
        target_positions=positions,
        max_long=max_long,
        apply_position_lag=False,
    )

    return {
        "dates": [str(d) for d in dates],
        "prices": [float(x) for x in prices],
        "equityCurve": [float(x) for x in bt.equity_curve],
        "sharpe": float(bt.metrics.get("sharpe", 0.0)),
        "maxDrawdown": float(bt.metrics.get("max_drawdown", 0.0)),
        "trades": int(len(bt.trade_log)),
    }


def run_backtest(prediction: Dict, params: Dict) -> Dict:
    dates = prediction.get("dates", [])
    prices = np.asarray(prediction.get("prices", []), dtype=float)
    target_positions = np.asarray(prediction.get("fusedPositions", []), dtype=float)

    if len(prices) == 0 or len(target_positions) == 0 or len(prices) != len(target_positions):
        raise ValueError("Invalid prediction payload for backtest")

    max_long = float(params.get("maxLong", 1.6))
    backtester = LongOnlyExecutionBacktester(
        initial_capital=float(params.get("initialCapital", 10_000.0)),
        commission_pct=float(params.get("commission", 0.0)),
        slippage_pct=float(params.get("slippage", 0.0)),
        min_position_change=float(params.get("minPositionChange", 0.0)),
    )
    bt = backtester.backtest(
        dates=np.asarray(dates),
        prices=prices,
        target_positions=np.clip(target_positions, 0.0, max_long),
        max_long=max_long,
        apply_position_lag=True,
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
        position_val = float(bt.effective_positions[idx]) if len(bt.effective_positions) > idx else 0.0
        cumulative_pnl = float(equity[idx] - float(params.get("initialCapital", 10_000.0)))
        pnl = 0.0 if idx == 0 else float(equity[idx] - equity[idx - 1])
        notes = str(trade.get("blocked_reason") or "")

        record = {
            "id": f"trade-{i+1}",
            "date": pd.to_datetime(trade.get("date")).strftime("%Y-%m-%d"),
            "action": "BUY" if action == "BUY" else "SELL",
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
                "type": "buy" if record["action"] == "BUY" else "sell",
                "shares": record["shares"],
                "confidence": float(min(1.0, abs(position_val))),
                "scope": "backtest",
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
        "sharpeRatio": float(bt.metrics.get("sharpe", 0.0)),
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
    }

    buy_hold_equity = [
        {"date": dates[i], "equity": float(bt.buy_hold_equity[i])}
        for i in range(len(dates))
    ]

    csv_rows = ["date,price,target_position,effective_position,equity"]
    for i in range(len(dates)):
        csv_rows.append(
            f"{dates[i]},{prices[i]:.6f},{target_positions[i]:.6f},{bt.effective_positions[i]:.6f},{equity[i]:.6f}"
        )

    forward_sim = None
    if bool(params.get("enableForwardSim", False)):
        forward_sim = _forward_simulate(prediction, params)

    return {
        "equityCurve": equity_curve,
        "priceSeries": price_series,
        "tradeLog": trade_log,
        "metrics": metrics,
        "annotations": annotations,
        "buyHoldEquity": buy_hold_equity,
        "csv": "\n".join(csv_rows),
        "forwardSimulation": forward_sim,
    }
