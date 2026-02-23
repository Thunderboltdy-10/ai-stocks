"""Prediction and backtest service layer for the GBM-first API."""

from __future__ import annotations

from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from data.data_fetcher import fetch_stock_data
from data.feature_engineer import engineer_features
from data.target_engineering import prevent_lookahead_bias
from evaluation.advanced_backtester import AdvancedBacktester
from inference.load_gbm_models import GBMModelBundle, load_gbm_models, predict_with_gbm
from inference.position_sizing import PositionSizer


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


def _apply_volatility_targeting(
    positions: np.ndarray,
    close_series: pd.Series,
    vol_target_annual: float = 0.25,
    min_scale: float = 0.5,
    max_scale: float = 2.2,
    max_long: float = 1.8,
    max_short: float = 0.2,
) -> np.ndarray:
    if vol_target_annual <= 0:
        return np.asarray(positions, dtype=float)

    rets = close_series.pct_change().fillna(0.0)
    ann_vol = rets.rolling(20, min_periods=5).std() * np.sqrt(252.0)
    scale = (vol_target_annual / (ann_vol + 1e-6)).clip(lower=min_scale, upper=max_scale).fillna(1.0)

    scaled = np.asarray(positions, dtype=float) * scale.to_numpy(dtype=float)
    return np.clip(scaled, -max_short, max_long)


def _model_quality_gate(bundle: GBMModelBundle) -> bool:
    holdout = bundle.metadata.get("holdout", {}).get("ensemble", {})
    dir_acc = float(holdout.get("dir_acc", 0.0))
    ic = float(holdout.get("ic", 0.0))
    pred_std = float(holdout.get("pred_std", 0.0))
    return (dir_acc >= 0.50) and (ic >= 0.0) and (pred_std >= 0.005)


def _build_trade_markers(dates: List[str], prices: np.ndarray, positions: np.ndarray) -> List[Dict]:
    markers: List[Dict] = []
    prev = 0.0
    for i, pos in enumerate(positions):
        if np.sign(pos) != np.sign(prev):
            if pos > 0:
                marker_type = "buy"
            elif pos < 0:
                marker_type = "sell"
            else:
                marker_type = "hold"
            markers.append(
                {
                    "date": dates[i],
                    "price": float(prices[i]),
                    "type": marker_type,
                    "shares": float(abs(pos) * 100),
                    "confidence": float(min(1.0, abs(pos))),
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

    if _model_quality_gate(bundle):
        sizer = PositionSizer()
        positions = sizer.size_batch(
            predicted_returns=preds["pred_ens"],
            prediction_stds=preds["pred_std"],
            win_rate=_sizing_win_rate(bundle),
            avg_win=0.012,
            avg_loss=0.010,
        )
        positions = _apply_volatility_targeting(
            positions=positions,
            close_series=raw["Close"],
            vol_target_annual=0.25,
            min_scale=0.5,
            max_scale=2.2,
            max_long=1.8,
            max_short=0.2,
        )
    else:
        positions = np.ones(len(raw), dtype=float)

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
            "modelQualityGatePassed": _model_quality_gate(bundle),
        },
    }

    return result


def run_backtest(prediction: Dict, params: Dict) -> Dict:
    dates = prediction.get("dates", [])
    prices = np.asarray(prediction.get("prices", []), dtype=float)
    positions = np.asarray(prediction.get("fusedPositions", []), dtype=float)

    if len(prices) == 0 or len(positions) == 0 or len(prices) != len(positions):
        raise ValueError("Invalid prediction payload for backtest")

    returns = np.zeros_like(prices)
    if len(prices) > 1:
        returns[1:] = prices[1:] / prices[:-1] - 1.0

    backtester = AdvancedBacktester(
        initial_capital=float(params.get("initialCapital", 10_000.0)),
        commission_pct=float(params.get("commission", 0.0)),
        slippage_pct=float(params.get("slippage", 0.0)),
        min_commission=0.0,
    )

    max_long = float(params.get("maxLong", 1.8))
    max_short = float(params.get("maxShort", 0.2))

    bt = backtester.backtest_with_positions(
        dates=np.asarray(dates),
        prices=prices,
        returns=returns,
        positions=positions,
        max_long=max_long,
        max_short=max_short,
        apply_position_lag=True,
    )

    equity = bt.equity_curve[1:]
    running_max = np.maximum.accumulate(equity)
    drawdown = np.where(running_max > 0, equity / running_max - 1.0, 0.0)

    equity_curve = [
        {"date": dates[i], "equity": float(equity[i]), "drawdown": float(drawdown[i])}
        for i in range(len(equity))
    ]
    price_series = [{"date": dates[i], "price": float(prices[i])} for i in range(len(prices))]

    trade_log = []
    annotations = []
    for i, trade in enumerate(bt.trade_log):
        action = str(trade.get("action", "BUY"))
        position_val = float(trade.get("position", 0.0))
        record = {
            "id": f"trade-{i+1}",
            "date": pd.to_datetime(trade.get("date")).strftime("%Y-%m-%d"),
            "action": "BUY" if action == "BUY" else "SELL",
            "price": float(trade.get("price", 0.0)),
            "shares": float(trade.get("shares", 0.0)),
            "position": position_val,
            "pnl": float(trade.get("equity_before", 0.0)),
            "cumulativePnl": float(trade.get("equity_before", 0.0)),
            "commission": float(trade.get("commission", 0.0)),
            "slippage": float(trade.get("slippage", 0.0)),
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

    metrics = {
        "cumulativeReturn": float(bt.metrics.get("cum_return", 0.0)),
        "sharpeRatio": float(bt.metrics.get("sharpe", 0.0)),
        "maxDrawdown": float(bt.metrics.get("max_drawdown", 0.0)),
        "winRate": float(bt.metrics.get("win_rate", 0.0)),
        "averageTradeProfit": float(np.mean([t["pnl"] for t in trade_log])) if trade_log else 0.0,
        "totalTrades": len(trade_log),
        "directionalAccuracy": float(np.mean(np.sign(np.asarray(prediction.get("predictedReturns", []), dtype=float)) == np.sign(returns)))
        if len(prediction.get("predictedReturns", [])) == len(returns)
        else 0.0,
        "correlation": 0.0,
        "rmse": 0.0,
        "smape": 0.0,
    }

    buy_hold_equity = [
        {"date": dates[i], "equity": float(bt.buy_hold_equity[1:][i])}
        for i in range(len(dates))
    ]

    csv_rows = ["date,price,position,equity"]
    for i in range(len(dates)):
        csv_rows.append(f"{dates[i]},{prices[i]:.6f},{positions[i]:.6f},{equity[i]:.6f}")

    return {
        "equityCurve": equity_curve,
        "priceSeries": price_series,
        "tradeLog": trade_log,
        "metrics": metrics,
        "annotations": annotations,
        "buyHoldEquity": buy_hold_equity,
        "csv": "\n".join(csv_rows),
    }
