"""Unified GBM-first backtest runner."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from data.data_fetcher import fetch_stock_data
from data.feature_engineer import engineer_features
from data.target_engineering import prevent_lookahead_bias
from evaluation.advanced_backtester import AdvancedBacktester
from inference.load_gbm_models import GBMModelBundle, load_gbm_models, predict_with_gbm
from inference.position_sizing import PositionSizer


def _to_daily_arithmetic_return(pred_log_return: np.ndarray, target_horizon_days: int) -> np.ndarray:
    horizon = max(1, int(target_horizon_days))
    return np.expm1(np.asarray(pred_log_return, dtype=float) / horizon)


@dataclass
class BacktestConfig:
    symbol: str = "AAPL"
    start: str = "2020-01-01"
    end: str = "2024-12-31"
    mode: str = "gbm_only"
    include_sentiment: bool = False
    max_long: float = 1.8
    max_short: float = 0.2
    commission_pct: float = 0.0005
    slippage_pct: float = 0.0003


class UnifiedBacktester:
    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self.symbol = config.symbol.upper()

    def _load_bundle(self) -> GBMModelBundle:
        bundle, metadata = load_gbm_models(self.symbol)
        if bundle is None:
            raise FileNotFoundError(f"Failed to load GBM models for {self.symbol}: {metadata.get('errors')}")
        return bundle

    def _load_price_frame(self) -> pd.DataFrame:
        raw = fetch_stock_data(self.symbol, period="max")
        raw.index = pd.to_datetime(raw.index)
        if getattr(raw.index, "tz", None) is not None:
            raw.index = raw.index.tz_localize(None)

        start_ts = pd.Timestamp(self.config.start)
        end_ts = pd.Timestamp(self.config.end)

        window = raw.loc[(raw.index >= start_ts) & (raw.index <= end_ts)].copy()
        if len(window) < 150:
            raise ValueError(
                f"Backtest window too short ({len(window)} rows). "
                f"Use a larger date range."
            )
        return window

    def _predict(self, bundle: GBMModelBundle, frame: pd.DataFrame) -> Dict[str, np.ndarray]:
        engineered = engineer_features(
            frame,
            symbol=self.symbol,
            include_sentiment=self.config.include_sentiment,
        )
        engineered = prevent_lookahead_bias(engineered, feature_cols=bundle.feature_columns)
        engineered[bundle.feature_columns] = engineered[bundle.feature_columns].fillna(0.0)

        missing = [c for c in bundle.feature_columns if c not in engineered.columns]
        if missing:
            raise ValueError(f"Missing required model features: {missing[:10]}")

        X = engineered[bundle.feature_columns].to_numpy(dtype=np.float32)
        components = predict_with_gbm(bundle, X, return_components=True)
        target_horizon = int(bundle.metadata.get("target_horizon_days", 1))
        components = {k: _to_daily_arithmetic_return(v, target_horizon) for k, v in components.items()}

        if "xgb" in components and "lgb" in components:
            weights = bundle.metadata.get("ensemble_weights", {"xgb": 0.5, "lgb": 0.5})
            wx = float(weights.get("xgb", 0.5))
            wl = float(weights.get("lgb", 0.5))
            total = max(wx + wl, 1e-9)
            pred_ens = (wx * components["xgb"] + wl * components["lgb"]) / total
            pred_std = np.std(np.vstack([components["xgb"], components["lgb"]]), axis=0) + 1e-4
        else:
            only = list(components.values())[0]
            pred_ens = only
            pred_std = pd.Series(pred_ens).rolling(20, min_periods=5).std().fillna(np.nanstd(pred_ens) + 1e-4).to_numpy()

        return {
            "pred_xgb": components.get("xgb", np.full(len(X), np.nan)),
            "pred_lgb": components.get("lgb", np.full(len(X), np.nan)),
            "pred_ens": pred_ens,
            "pred_std": pred_std,
            "engineered": engineered,
        }

    @staticmethod
    def _derive_sizing_stats(bundle: GBMModelBundle) -> Dict[str, float]:
        holdout = bundle.metadata.get("holdout", {}).get("ensemble", {})
        dir_acc = float(holdout.get("dir_acc", 0.53))
        target_dist = bundle.metadata.get("target_distribution", {})
        train_pos = float(target_dist.get("train_positive_pct", dir_acc))
        # Use training target balance as a stabilizing prior for Kelly win-rate.
        win_rate = float(np.clip(max(dir_acc, train_pos), 0.50, 0.60))
        return {
            "win_rate": win_rate,
            "avg_win": 0.012,
            "avg_loss": 0.010,
        }

    def run(self) -> Dict:
        bundle = self._load_bundle()
        frame = self._load_price_frame()
        preds = self._predict(bundle, frame)

        close = frame["Close"].to_numpy(dtype=float)
        returns = frame["Close"].pct_change().fillna(0.0).to_numpy(dtype=float)
        forward_returns = frame["Close"].pct_change().shift(-1).to_numpy(dtype=float)

        stats = self._derive_sizing_stats(bundle)
        sizer = PositionSizer()
        positions = sizer.size_batch(
            predicted_returns=preds["pred_ens"],
            prediction_stds=preds["pred_std"],
            win_rate=stats["win_rate"],
            avg_win=stats["avg_win"],
            avg_loss=stats["avg_loss"],
        )

        backtester = AdvancedBacktester(
            initial_capital=100_000.0,
            commission_pct=self.config.commission_pct,
            slippage_pct=self.config.slippage_pct,
            min_commission=0.0,
        )

        results = backtester.backtest_with_positions(
            dates=frame.index.to_numpy(),
            prices=close,
            returns=returns,
            positions=positions,
            max_long=self.config.max_long,
            max_short=self.config.max_short,
            apply_position_lag=True,
        )

        valid_dir = ~np.isnan(forward_returns)
        direction_accuracy = float(np.mean(np.sign(preds["pred_ens"][valid_dir]) == np.sign(forward_returns[valid_dir])))

        summary = {
            "symbol": self.symbol,
            "start": self.config.start,
            "end": self.config.end,
            "mode": self.config.mode,
            "n_days": int(len(frame)),
            "strategy_return": float(results.metrics.get("cum_return", 0.0)),
            "buy_hold_return": float(results.metrics.get("buy_hold_cum_return", 0.0)),
            "alpha": float(results.metrics.get("alpha", 0.0)),
            "sharpe_ratio": float(results.metrics.get("sharpe", 0.0)),
            "max_drawdown": float(results.metrics.get("max_drawdown", 0.0)),
            "win_rate": float(results.metrics.get("win_rate", 0.0)),
            "direction_accuracy": direction_accuracy,
            "prediction_std": float(np.std(preds["pred_ens"])),
            "pred_positive_pct": float((preds["pred_ens"] > 0).mean()),
            "turnover": float(results.metrics.get("turnover", 0.0)),
            "total_transaction_costs": float(results.metrics.get("total_transaction_costs", 0.0)),
            "cost_adjusted_sharpe": float(results.metrics.get("sharpe", 0.0)),
        }

        self._save_outputs(frame, preds, positions, returns, forward_returns, results, summary)
        return summary

    def _save_outputs(
        self,
        frame: pd.DataFrame,
        preds: Dict[str, np.ndarray],
        positions: np.ndarray,
        returns: np.ndarray,
        forward_returns: np.ndarray,
        results,
        summary: Dict,
    ) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_dir = Path("backtest_results") / f"{self.symbol}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)

        daily = pd.DataFrame(
            {
                "date": frame.index,
                "close": frame["Close"].to_numpy(),
                "pred_xgb": preds["pred_xgb"],
                "pred_lgb": preds["pred_lgb"],
                "pred_ens": preds["pred_ens"],
                "pred_std": preds["pred_std"],
                "position": positions,
                "asset_return": returns,
                "forward_return": forward_returns,
                "strategy_daily_return": results.daily_returns,
            }
        )
        daily.to_csv(out_dir / "daily_results.csv", index=False)

        trade_df = pd.DataFrame(results.trade_log)
        trade_df.to_csv(out_dir / "trade_log.csv", index=False)

        equity = pd.DataFrame(
            {
                "date": frame.index,
                "strategy_equity": results.equity_curve[1:],
                "buy_hold_equity": results.buy_hold_equity[1:],
            }
        )
        equity.to_csv(out_dir / "equity_comparison.csv", index=False)

        with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        if hasattr(results, "metrics"):
            with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
                json.dump(results.metrics, fh, indent=2)

        print(f"Saved backtest outputs to: {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GBM-first backtest")
    parser.add_argument("--symbol", type=str, default="AAPL")
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--mode", type=str, default="gbm_only", choices=["gbm_only", "ensemble"])
    parser.add_argument("--include-sentiment", action="store_true")
    parser.add_argument("--max-long", type=float, default=1.8)
    parser.add_argument("--max-short", type=float, default=0.2)
    parser.add_argument("--commission", type=float, default=0.0005)
    parser.add_argument("--slippage", type=float, default=0.0003)
    args = parser.parse_args()

    cfg = BacktestConfig(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        mode=args.mode,
        include_sentiment=args.include_sentiment,
        max_long=args.max_long,
        max_short=args.max_short,
        commission_pct=args.commission,
        slippage_pct=args.slippage,
    )

    summary = UnifiedBacktester(cfg).run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
