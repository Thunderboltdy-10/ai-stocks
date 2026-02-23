"""GBM-first prediction helper for latest snapshot inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from data.data_fetcher import fetch_stock_data
from data.feature_engineer import engineer_features
from data.target_engineering import prevent_lookahead_bias
from inference.load_gbm_models import load_gbm_models, predict_with_gbm


def _to_daily_arithmetic_return(pred_log_return: np.ndarray, target_horizon_days: int) -> np.ndarray:
    horizon = max(1, int(target_horizon_days))
    return np.expm1(np.asarray(pred_log_return, dtype=float) / horizon)


def predict_ensemble(
    symbol: str,
    risk_profile: str = "conservative",
    data_period: str = "6mo",
    include_features: bool = False,
    feature_subset: Optional[list[str]] = None,
) -> Dict:
    symbol = symbol.upper()
    bundle, metadata = load_gbm_models(symbol)
    if bundle is None:
        raise FileNotFoundError(f"Failed loading GBM models for {symbol}: {metadata.get('errors')}")

    df = fetch_stock_data(symbol=symbol, period=data_period)
    engineered = engineer_features(df, symbol=symbol, include_sentiment=False)
    engineered = prevent_lookahead_bias(engineered, feature_cols=bundle.feature_columns)
    engineered[bundle.feature_columns] = engineered[bundle.feature_columns].fillna(0.0)

    use_cols = bundle.feature_columns
    if feature_subset:
        use_cols = [c for c in feature_subset if c in engineered.columns]
    if not use_cols:
        use_cols = bundle.feature_columns

    X = engineered[use_cols].to_numpy(dtype=np.float32)
    pred_components = predict_with_gbm(bundle, X, return_components=True)
    target_horizon = int(bundle.metadata.get("target_horizon_days", 1))
    pred_components = {k: _to_daily_arithmetic_return(v, target_horizon) for k, v in pred_components.items()}

    if "xgb" in pred_components and "lgb" in pred_components:
        weights = bundle.metadata.get("ensemble_weights", {"xgb": 0.5, "lgb": 0.5})
        wx = float(weights.get("xgb", 0.5))
        wl = float(weights.get("lgb", 0.5))
        total = max(wx + wl, 1e-9)
        pred = (wx * pred_components["xgb"] + wl * pred_components["lgb"]) / total
    else:
        pred = list(pred_components.values())[0]

    latest_pred = float(pred[-1])
    latest_price = float(df["Close"].iloc[-1])
    next_price = latest_price * (1.0 + latest_pred)

    out = {
        "symbol": symbol,
        "risk_profile": risk_profile,
        "latest_date": str(df.index[-1].date()),
        "latest_price": latest_price,
        "predicted_return_1d": latest_pred,
        "predicted_price_1d": next_price,
        "prediction_std": float(np.std(pred)),
    }

    if include_features:
        out["features"] = {c: float(engineered[c].iloc[-1]) for c in use_cols}

    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict latest next-day return using GBM ensemble")
    parser.add_argument("--symbol", type=str, default="AAPL")
    parser.add_argument("--risk-profile", type=str, default="conservative")
    parser.add_argument("--data-period", type=str, default="6mo")
    args = parser.parse_args()

    result = predict_ensemble(symbol=args.symbol, risk_profile=args.risk_profile, data_period=args.data_period)
    print(json.dumps(result, indent=2))
