"""Unified GBM-first backtest runner."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from pandas.tseries.offsets import BDay

from data.data_fetcher import fetch_stock_data
from data.feature_engineer import engineer_features
from data.target_engineering import prevent_lookahead_bias
from evaluation.execution_backtester import LongOnlyExecutionBacktester
from inference.load_gbm_models import GBMModelBundle, blend_with_direction_head, load_gbm_models, predict_with_gbm
from inference.intraday_signal import intraday_hybrid_positions
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
from inference.variant_quality import score_variant_quality
from utils.timeframe import execution_profile, strategy_mix_for_window


def _to_daily_arithmetic_return(pred_log_return: np.ndarray, target_horizon_days: int) -> np.ndarray:
    horizon = max(1, int(target_horizon_days))
    return np.expm1(np.asarray(pred_log_return, dtype=float) / horizon)


def _to_model_target_return(daily_return: np.ndarray, target_horizon_days: int) -> np.ndarray:
    horizon = max(1, int(target_horizon_days))
    arr = np.asarray(daily_return, dtype=float)
    if horizon == 1:
        return arr
    return np.log1p(np.clip(arr, -0.95, None)) * horizon


@dataclass
class BacktestConfig:
    symbol: str = "AAPL"
    start: str = "2020-01-01"
    end: str = "2024-12-31"
    mode: str = "gbm_only"
    include_sentiment: bool = False
    model_variant: str = "gbm"
    data_period: str = "max"
    data_interval: str = "1d"
    max_long: float = 1.6
    max_short: float = 0.25
    commission_pct: float = 0.0005
    slippage_pct: float = 0.0003
    vol_target_annual: float = 0.40
    min_vol_scale: float = 0.90
    max_vol_scale: float = 1.60
    warmup_days: int = 252
    min_eval_days: int = 20
    min_position_change: float = 0.0
    ml_weight_override: Optional[float] = None
    fallback_ml_weight_override: Optional[float] = None
    core_long_override: Optional[float] = None
    overlay_min_override: Optional[float] = None
    overlay_max_override: Optional[float] = None
    smooth_alpha_override: Optional[float] = None


class UnifiedBacktester:
    def __init__(self, config: BacktestConfig) -> None:
        self.config = config
        self.symbol = config.symbol.upper()
        self.resolved_model_variant = config.model_variant

    def _load_bundle(self) -> GBMModelBundle:
        resolved_variant = resolve_model_variant(
            symbol=self.symbol,
            requested_variant=self.config.model_variant,
            data_interval=self.config.data_interval,
            start=self.config.start,
            end=self.config.end,
            default_variant="gbm",
        )
        self.resolved_model_variant = resolved_variant
        model_dir = Path("saved_models") / self.symbol / resolved_variant
        bundle, metadata = load_gbm_models(self.symbol, model_dir=model_dir)
        if bundle is None:
            raise FileNotFoundError(f"Failed to load GBM models for {self.symbol}: {metadata.get('errors')}")
        return bundle

    @staticmethod
    def _model_quality_gate(bundle: GBMModelBundle) -> bool:
        return model_quality_gate_strict(bundle.metadata)

    def _load_price_frame(self) -> Tuple[pd.DataFrame, pd.Series]:
        raw = fetch_stock_data(
            self.symbol,
            period=self.config.data_period,
            interval=self.config.data_interval,
        )
        raw.index = pd.to_datetime(raw.index)
        if getattr(raw.index, "tz", None) is not None:
            raw.index = raw.index.tz_localize(None)

        start_ts = pd.Timestamp(self.config.start)
        end_ts = pd.Timestamp(self.config.end)
        warmup_start = start_ts - BDay(max(0, int(self.config.warmup_days)))

        window = raw.loc[(raw.index >= warmup_start) & (raw.index <= end_ts)].copy()
        eval_mask = (window.index >= start_ts) & (window.index <= end_ts)
        eval_days = int(eval_mask.sum())
        if eval_days < int(self.config.min_eval_days):
            raise ValueError(
                f"Backtest window too short ({eval_days} rows). "
                f"Need at least {self.config.min_eval_days} rows."
            )
        return window, eval_mask

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
        raw_components = {
            k: np.asarray(v, dtype=float)
            for k, v in components.items()
            if k in {"xgb", "lgb"}
        }
        direction_prob = np.asarray(components.get("direction_prob", np.full(len(X), np.nan)), dtype=float)
        daily_components = {
            k: _to_daily_arithmetic_return(v, target_horizon)
            for k, v in raw_components.items()
        }

        if "xgb" in raw_components and "lgb" in raw_components:
            weights = bundle.metadata.get("ensemble_weights", {"xgb": 0.5, "lgb": 0.5})
            wx = float(weights.get("xgb", 0.5))
            wl = float(weights.get("lgb", 0.5))
            total = max(wx + wl, 1e-9)
            pred_raw = (wx * raw_components["xgb"] + wl * raw_components["lgb"]) / total
            pred_std_raw = np.std(np.vstack([raw_components["xgb"], raw_components["lgb"]]), axis=0) + 1e-4
            pred_std = np.std(np.vstack([daily_components["xgb"], daily_components["lgb"]]), axis=0) + 1e-4
        else:
            pred_raw = next(iter(raw_components.values()))
            pred_std_raw = (
                pd.Series(pred_raw)
                .rolling(20, min_periods=5)
                .std()
                .fillna(np.nanstd(pred_raw) + 1e-4)
                .to_numpy()
            )
            pred_std = (
                pd.Series(_to_daily_arithmetic_return(pred_raw, target_horizon))
                .rolling(20, min_periods=5)
                .std()
                .fillna(np.nanstd(_to_daily_arithmetic_return(pred_raw, target_horizon)) + 1e-4)
                .to_numpy()
            )

        pred_ens_raw = blend_with_direction_head(pred_raw, components.get("direction_prob"), bundle.metadata)
        pred_ens = _to_daily_arithmetic_return(pred_ens_raw, target_horizon)

        return {
            "pred_xgb": daily_components.get("xgb", np.full(len(X), np.nan)),
            "pred_lgb": daily_components.get("lgb", np.full(len(X), np.nan)),
            "pred_ens": pred_ens,
            "pred_ens_raw": pred_ens_raw,
            "pred_std": pred_std,
            "pred_std_raw": pred_std_raw,
            "direction_prob": direction_prob,
            "engineered": engineered,
        }

    def _regime_positions(self, frame: pd.DataFrame, profile: Dict[str, float | bool]) -> np.ndarray:
        return regime_exposure_from_prices(
            close=frame["Close"],
            max_long=self.config.max_long,
            max_short=self.config.max_short,
            vol_target_annual=self.config.vol_target_annual,
            min_scale=self.config.min_vol_scale,
            max_scale=self.config.max_vol_scale,
            periods_per_year=float(profile["annualization_factor"]),
            base_exposure=float(profile["regime_base_exposure"]),
            bull_boost=float(profile["regime_bull_boost"]),
            bear_penalty=float(profile["regime_bear_penalty"]),
            momentum_weight=float(profile["regime_momentum_weight"]),
            bear_short_boost=float(profile["regime_bear_short_boost"]),
        )

    @staticmethod
    def _derive_sizing_stats(bundle: GBMModelBundle) -> Dict[str, float]:
        hold = bundle.metadata.get("holdout", {})
        dir_acc = max(
            float(hold.get("ensemble", {}).get("dir_acc", 0.53)),
            float(hold.get("ensemble_directional", {}).get("dir_acc", 0.53)),
            float(hold.get("ensemble_calibrated", {}).get("dir_acc", 0.53)),
        )
        target_dist = bundle.metadata.get("target_distribution", {})
        train_pos = float(target_dist.get("train_positive_pct", dir_acc))
        # Use training target balance as a stabilizing prior for Kelly win-rate.
        win_rate = float(np.clip(max(dir_acc, train_pos), 0.50, 0.60))
        return {
            "win_rate": win_rate,
            "avg_win": 0.012,
            "avg_loss": 0.010,
        }

    @staticmethod
    def _adaptive_ml_weights(bundle: GBMModelBundle) -> tuple[float, float]:
        intraday = bool(execution_profile(str(bundle.metadata.get("data_interval", "1d"))).get("is_intraday", False))
        quality = score_variant_quality(bundle.metadata, intraday=intraday)
        metrics = quality.metrics
        dir_acc = float(metrics.get("dir_acc", 0.0))
        pred_std = float(metrics.get("pred_std", 0.0))
        net_sharpe = float(metrics.get("net_sharpe", 0.0))
        net_return = float(metrics.get("net_return", 0.0))
        wfe = float(metrics.get("wfe", 0.0))

        if quality.passed:
            return (0.24, 0.08) if intraday else (0.24, 0.08)
        if intraday and dir_acc >= 0.50 and pred_std >= 0.0012 and (wfe >= 5.0 or net_sharpe >= 0.05 or net_return >= 0.04):
            return 0.16, 0.08
        if (not intraday) and dir_acc >= 0.50 and pred_std >= 0.0015 and (net_sharpe >= 0.10 or net_return >= 0.06):
            return 0.14, 0.06
        return (0.06, 0.04) if intraday else (0.08, 0.04)

    @staticmethod
    def _run_execution(
        backtester: LongOnlyExecutionBacktester,
        *,
        dates: np.ndarray,
        prices: np.ndarray,
        positions: np.ndarray,
        profile: Dict[str, float | bool],
        max_long: float,
        max_short: float,
    ):
        return backtester.backtest(
            dates=dates,
            prices=prices,
            target_positions=positions,
            max_long=max_long,
            max_short=max_short,
            apply_position_lag=True,
            annualization_factor=float(profile["annualization_factor"]),
            flat_at_day_end=bool(profile["flat_at_day_end"]),
            day_end_flatten_fraction=float(profile.get("day_end_flatten_fraction", 1.0)),
        )

    @staticmethod
    def _execution_summary(results) -> Dict[str, float]:
        return {
            "return": float(results.metrics.get("cum_return", 0.0)),
            "buy_hold_return": float(results.metrics.get("buy_hold_cum_return", 0.0)),
            "alpha": float(results.metrics.get("alpha", 0.0)),
            "sharpe": float(results.metrics.get("sharpe", 0.0)),
            "max_drawdown": float(results.metrics.get("max_drawdown", 0.0)),
            "turnover": float(results.metrics.get("turnover", 0.0)),
        }

    def _finalize_positions(
        self,
        frame: pd.DataFrame,
        active_positions: np.ndarray,
        calibrated_pred: np.ndarray,
        profile: Dict[str, float | bool],
        window_mix: Dict[str, float],
    ) -> np.ndarray:
        overlay_min = (
            float(self.config.overlay_min_override)
            if self.config.overlay_min_override is not None
            else float(np.clip(float(profile["overlay_min"]) * window_mix["overlay_multiplier"], 0.01, 0.95))
        )
        overlay_max = (
            float(self.config.overlay_max_override)
            if self.config.overlay_max_override is not None
            else float(np.clip(float(profile["overlay_max"]) * window_mix["overlay_multiplier"], 0.05, 0.98))
        )
        overlay_weights = adaptive_overlay_weight(
            close=frame["Close"],
            predicted_returns=calibrated_pred,
            lookback=60,
            min_weight=overlay_min,
            max_weight=overlay_max,
        )
        core_long = float(self.config.core_long_override) if self.config.core_long_override is not None else float(
            np.clip(float(profile["core_long"]) + window_mix["core_adjust"], -self.config.max_short, self.config.max_long)
        )
        smooth_alpha = (
            float(self.config.smooth_alpha_override)
            if self.config.smooth_alpha_override is not None
            else float(window_mix["smooth_alpha"])
        )
        positions = blend_positions_with_core(
            active_positions=active_positions,
            overlay_weights=overlay_weights,
            core_long=core_long,
            max_long=self.config.max_long,
            max_short=self.config.max_short,
            smooth_alpha=smooth_alpha,
        )
        return apply_short_safety_filter(
            close=frame["Close"],
            positions=positions,
            predicted_returns=calibrated_pred,
            max_short=self.config.max_short,
            interval=self.config.data_interval,
        )

    def run(self) -> Dict:
        profile = execution_profile(self.config.data_interval)
        bundle = self._load_bundle()
        frame_raw, _ = self._load_price_frame()
        preds = self._predict(bundle, frame_raw)

        # Align raw prices to engineered-feature index (feature generation drops warmup rows).
        if not isinstance(preds.get("engineered"), pd.DataFrame):
            raise ValueError("Prediction pipeline did not return engineered frame for index alignment")
        engineered_index = preds["engineered"].index
        frame = frame_raw.loc[engineered_index].copy()

        start_ts = pd.Timestamp(self.config.start)
        end_ts = pd.Timestamp(self.config.end)
        eval_mask = (frame.index >= start_ts) & (frame.index <= end_ts)
        eval_days = int(eval_mask.sum())
        if eval_days < int(self.config.min_eval_days):
            raise ValueError(
                f"Backtest window too short after feature warmup ({eval_days} rows). "
                f"Need at least {self.config.min_eval_days} rows."
            )
        window_mix = strategy_mix_for_window(eval_days, is_intraday=bool(profile["is_intraday"]))

        close_full = frame["Close"].to_numpy(dtype=float)
        forward_returns_full = frame["Close"].pct_change().shift(-1).to_numpy(dtype=float)

        stats = self._derive_sizing_stats(bundle)
        use_model_positions = self._model_quality_gate(bundle)
        ml_weight, fallback_ml_weight = self._adaptive_ml_weights(bundle)
        if self.config.ml_weight_override is not None:
            ml_weight = float(self.config.ml_weight_override)
        if self.config.fallback_ml_weight_override is not None:
            fallback_ml_weight = float(self.config.fallback_ml_weight_override)
        quality = score_variant_quality(bundle.metadata, intraday=bool(profile["is_intraday"]))
        target_horizon = int(bundle.metadata.get("target_horizon_days", 1))
        calibrated_pred_raw, direction_conf = apply_direction_calibration(
            preds["pred_ens_raw"],
            bundle.metadata.get("direction_calibrator"),
        )
        calibrated_pred = _to_daily_arithmetic_return(calibrated_pred_raw, target_horizon)
        policy = bundle.metadata.get("execution_calibration", {})
        sizer_cfg = PositionSizingConfig(
            max_long=self.config.max_long,
            max_short=self.config.max_short,
            base_long_bias=float(profile["base_long_bias"]),
            bearish_risk_off=float(profile["bearish_risk_off"]),
        )
        sizer = PositionSizer(sizer_cfg)
        if isinstance(policy, dict) and bool(policy.get("enabled", False)):
            ml_positions = positions_from_policy(
                calibrated_pred,
                preds["pred_std_raw"],
                policy,
                max_long=self.config.max_long,
                max_short=self.config.max_short,
            )
        elif bool(profile["is_intraday"]):
            ml_positions = intraday_hybrid_positions(
                close=frame["Close"],
                predicted_returns=calibrated_pred_raw,
                prediction_stds=preds["pred_std_raw"],
                max_long=self.config.max_long,
                max_short=self.config.max_short,
                cost_buffer=2.0 * float(self.config.commission_pct + self.config.slippage_pct),
            )
        else:
            ml_positions = sizer.size_batch(
                predicted_returns=calibrated_pred,
                prediction_stds=preds["pred_std_raw"],
                win_rate=stats["win_rate"],
                avg_win=stats["avg_win"],
                avg_loss=stats["avg_loss"],
            )
        regime_positions = self._regime_positions(frame, profile=profile)
        active_positions_full = combine_ml_and_regime(
            regime_positions=regime_positions,
            ml_positions=ml_positions,
            ml_gate_passed=use_model_positions,
            ml_weight=ml_weight,
            fallback_ml_weight=fallback_ml_weight,
            max_long=self.config.max_long,
            max_short=self.config.max_short,
        )
        active_positions_regime_only = np.clip(np.asarray(regime_positions, dtype=float), -self.config.max_short, self.config.max_long)
        active_positions_ml_only = np.clip(np.asarray(ml_positions, dtype=float), -self.config.max_short, self.config.max_long)
        positions_full = self._finalize_positions(
            frame=frame,
            active_positions=active_positions_full,
            calibrated_pred=calibrated_pred,
            profile=profile,
            window_mix=window_mix,
        )
        positions_regime_only = self._finalize_positions(
            frame=frame,
            active_positions=active_positions_regime_only,
            calibrated_pred=calibrated_pred,
            profile=profile,
            window_mix=window_mix,
        )
        positions_ml_only = self._finalize_positions(
            frame=frame,
            active_positions=active_positions_ml_only,
            calibrated_pred=calibrated_pred,
            profile=profile,
            window_mix=window_mix,
        )

        # Evaluate only in requested date range while using full history for feature warmup.
        close = close_full[eval_mask]
        forward_returns = forward_returns_full[eval_mask]
        positions = positions_full[eval_mask]
        positions_regime_eval = positions_regime_only[eval_mask]
        positions_ml_eval = positions_ml_only[eval_mask]
        frame_eval = frame.loc[eval_mask].copy()

        effective_min_change = float(max(0.0, self.config.min_position_change))
        if bool(profile["is_intraday"]):
            effective_min_change = max(effective_min_change, 0.03)
        if isinstance(policy, dict) and bool(policy.get("enabled", False)):
            effective_min_change = max(effective_min_change, float(max(0.0, policy.get("min_position_change", 0.0))))

        backtester = LongOnlyExecutionBacktester(
            initial_capital=100_000.0,
            commission_pct=self.config.commission_pct,
            slippage_pct=self.config.slippage_pct,
            min_position_change=effective_min_change,
        )

        results = backtester.backtest(
            dates=frame_eval.index.to_numpy(),
            prices=close,
            target_positions=positions,
            max_long=self.config.max_long,
            max_short=self.config.max_short,
            apply_position_lag=True,
            annualization_factor=float(profile["annualization_factor"]),
            flat_at_day_end=bool(profile["flat_at_day_end"]),
            day_end_flatten_fraction=float(profile.get("day_end_flatten_fraction", 1.0)),
        )
        regime_results = backtester.backtest(
            dates=frame_eval.index.to_numpy(),
            prices=close,
            target_positions=positions_regime_eval,
            max_long=self.config.max_long,
            max_short=self.config.max_short,
            apply_position_lag=True,
            annualization_factor=float(profile["annualization_factor"]),
            flat_at_day_end=bool(profile["flat_at_day_end"]),
            day_end_flatten_fraction=float(profile.get("day_end_flatten_fraction", 1.0)),
        )
        ml_results = backtester.backtest(
            dates=frame_eval.index.to_numpy(),
            prices=close,
            target_positions=positions_ml_eval,
            max_long=self.config.max_long,
            max_short=self.config.max_short,
            apply_position_lag=True,
            annualization_factor=float(profile["annualization_factor"]),
            flat_at_day_end=bool(profile["flat_at_day_end"]),
            day_end_flatten_fraction=float(profile.get("day_end_flatten_fraction", 1.0)),
        )

        valid_dir = ~np.isnan(forward_returns)
        pred_eval = calibrated_pred[eval_mask]
        direction_accuracy = float(np.mean(np.sign(pred_eval[valid_dir]) == np.sign(forward_returns[valid_dir]))) if np.any(valid_dir) else 0.0

        summary = {
            "symbol": self.symbol,
            "model_variant_requested": self.config.model_variant,
            "model_variant": self.resolved_model_variant,
            "data_period": self.config.data_period,
            "data_interval": self.config.data_interval,
            "start": self.config.start,
            "end": self.config.end,
            "mode": self.config.mode,
            "n_days": int(len(frame_eval)),
            "warmup_days": int(self.config.warmup_days),
            "strategy_return": float(results.metrics.get("cum_return", 0.0)),
            "buy_hold_return": float(results.metrics.get("buy_hold_cum_return", 0.0)),
            "alpha": float(results.metrics.get("alpha", 0.0)),
            "regime_only_return": float(regime_results.metrics.get("cum_return", 0.0)),
            "regime_only_alpha": float(regime_results.metrics.get("alpha", 0.0)),
            "ml_only_return": float(ml_results.metrics.get("cum_return", 0.0)),
            "ml_only_alpha": float(ml_results.metrics.get("alpha", 0.0)),
            "ml_incremental_return_vs_regime": float(results.metrics.get("cum_return", 0.0) - regime_results.metrics.get("cum_return", 0.0)),
            "ml_incremental_alpha_vs_regime": float(results.metrics.get("alpha", 0.0) - regime_results.metrics.get("alpha", 0.0)),
            "strategy_return_pct": float(results.metrics.get("cum_return", 0.0) * 100.0),
            "buy_hold_return_pct": float(results.metrics.get("buy_hold_cum_return", 0.0) * 100.0),
            "excess_return_pct": float(results.metrics.get("alpha", 0.0) * 100.0),
            "sharpe_ratio": float(results.metrics.get("sharpe", 0.0)),
            "sortino_ratio": float(results.metrics.get("sortino", 0.0)),
            "calmar_ratio": float(results.metrics.get("calmar", 0.0)),
            "max_drawdown": float(results.metrics.get("max_drawdown", 0.0)),
            "win_rate": float(results.metrics.get("win_rate", 0.0)),
            "direction_accuracy": direction_accuracy,
            "prediction_std": float(np.std(pred_eval)),
            "prediction_std_raw": float(np.std(calibrated_pred_raw[eval_mask])),
            "pred_positive_pct": float((pred_eval > 0).mean()),
            "turnover": float(results.metrics.get("turnover", 0.0)),
            "total_transaction_costs": float(results.metrics.get("total_transaction_costs", 0.0)),
            "total_borrow_fee": float(results.metrics.get("total_borrow_fee", 0.0)),
            "cost_adjusted_sharpe": float(results.metrics.get("sharpe", 0.0)),
            "vol_target_annual": float(self.config.vol_target_annual),
            "model_quality_gate_passed": bool(use_model_positions),
            "model_quality_score": float(quality.score),
            "model_quality_reasons": list(quality.reasons),
            "holdout_metric_source": quality.selected_source,
            "holdout_pred_target_std_ratio": float(quality.metrics.get("pred_target_std_ratio", 0.0)),
            "execution_mode": "signed_inventory" if self.config.max_short > 0 else "long_only_inventory",
            "ml_overlay_weight": ml_weight if use_model_positions else fallback_ml_weight,
            "ml_weight_override": self.config.ml_weight_override,
            "fallback_ml_weight_override": self.config.fallback_ml_weight_override,
            "core_long_override": self.config.core_long_override,
            "overlay_min_override": self.config.overlay_min_override,
            "overlay_max_override": self.config.overlay_max_override,
            "smooth_alpha_override": self.config.smooth_alpha_override,
            "direction_confidence_mean": float(np.mean(np.abs(direction_conf))),
            "direction_prob_mean": float(np.nanmean(preds["direction_prob"])),
            "direction_head_enabled": bool(bundle.metadata.get("direction_head", {}).get("enabled", False)),
            "is_intraday": bool(profile["is_intraday"]),
            "annualization_factor": float(profile["annualization_factor"]),
            "flat_at_day_end": bool(profile["flat_at_day_end"]),
            "day_end_flatten_fraction": float(
                profile.get("day_end_flatten_fraction", 1.0) if bool(profile["flat_at_day_end"]) else 0.0
            ),
        }

        preds_eval = {
            "pred_xgb": preds["pred_xgb"][eval_mask],
            "pred_lgb": preds["pred_lgb"][eval_mask],
            "pred_ens": pred_eval,
            "pred_std": preds["pred_std"][eval_mask],
            "engineered": None,
        }

        self._save_outputs(
            frame_eval,
            preds_eval,
            {
                "full": positions,
                "regime_only": positions_regime_eval,
                "ml_only": positions_ml_eval,
            },
            forward_returns,
            {
                "full": results,
                "regime_only": regime_results,
                "ml_only": ml_results,
            },
            summary,
        )
        return summary

    def _save_outputs(
        self,
        frame: pd.DataFrame,
        preds: Dict[str, np.ndarray],
        positions: Dict[str, np.ndarray],
        forward_returns: np.ndarray,
        results: Dict[str, object],
        summary: Dict,
    ) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        out_dir = Path("backtest_results") / f"{self.symbol}_{ts}"
        out_dir.mkdir(parents=True, exist_ok=True)
        results_full = results["full"]
        positions_full = np.asarray(positions["full"], dtype=float)

        daily = pd.DataFrame(
            {
                "date": frame.index,
                "close": frame["Close"].to_numpy(),
                "pred_xgb": preds["pred_xgb"],
                "pred_lgb": preds["pred_lgb"],
                "pred_ens": preds["pred_ens"],
                "pred_std": preds["pred_std"],
                "position": positions_full,
                "position_regime_only": np.asarray(positions["regime_only"], dtype=float),
                "position_ml_only": np.asarray(positions["ml_only"], dtype=float),
                "effective_position": results_full.effective_positions,
                "asset_return": frame["Close"].pct_change().fillna(0.0).to_numpy(dtype=float),
                "forward_return": forward_returns,
                "strategy_daily_return": results_full.daily_returns,
                "strategy_equity": results_full.equity_curve,
                "buy_hold_equity": results_full.buy_hold_equity,
                "regime_only_equity": results["regime_only"].equity_curve,
                "ml_only_equity": results["ml_only"].equity_curve,
            }
        )
        daily.to_csv(out_dir / "daily_results.csv", index=False)

        trade_df = pd.DataFrame(results_full.trade_log)
        trade_df.to_csv(out_dir / "trade_log.csv", index=False)

        equity = pd.DataFrame(
            {
                "date": frame.index,
                "strategy_equity": results_full.equity_curve,
                "regime_only_equity": results["regime_only"].equity_curve,
                "ml_only_equity": results["ml_only"].equity_curve,
                "buy_hold_equity": results_full.buy_hold_equity,
            }
        )
        equity.to_csv(out_dir / "equity_comparison.csv", index=False)

        with (out_dir / "summary.json").open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)

        if hasattr(results_full, "metrics"):
            with (out_dir / "metrics.json").open("w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "full": results_full.metrics,
                        "regime_only": results["regime_only"].metrics,
                        "ml_only": results["ml_only"].metrics,
                    },
                    fh,
                    indent=2,
                )

        diagnostics = self._build_diagnostics(frame, results_full, summary)
        with (out_dir / "diagnostics.json").open("w", encoding="utf-8") as fh:
            json.dump(diagnostics, fh, indent=2)

        monthly_df = pd.DataFrame(diagnostics.get("monthly", []))
        if not monthly_df.empty:
            monthly_df.to_csv(out_dir / "monthly_diagnostics.csv", index=False)

        hourly_df = pd.DataFrame(diagnostics.get("hourly", []))
        if not hourly_df.empty:
            hourly_df.to_csv(out_dir / "hourly_diagnostics.csv", index=False)

        action_df = pd.DataFrame(diagnostics.get("action_breakdown", []))
        if not action_df.empty:
            action_df.to_csv(out_dir / "action_diagnostics.csv", index=False)

        self._save_plots(out_dir, frame, results, summary, diagnostics)
        print(f"Saved backtest outputs to: {out_dir}")

    @staticmethod
    def _build_diagnostics(frame: pd.DataFrame, results, summary: Dict) -> Dict:
        dates = pd.to_datetime(frame.index)
        intraday = bool(summary.get("is_intraday", False))
        ann = float(max(1.0, summary.get("annualization_factor", 252.0)))
        equity = np.asarray(results.equity_curve, dtype=float)
        buy_hold = np.asarray(results.buy_hold_equity, dtype=float)
        pos = np.asarray(results.effective_positions, dtype=float)
        rets = pd.Series(np.asarray(results.daily_returns, dtype=float), index=dates).fillna(0.0)
        bh_rets = pd.Series(frame["Close"].pct_change().fillna(0.0).to_numpy(dtype=float), index=dates)

        roll = 48 if intraday else 20
        r_mean = rets.rolling(roll, min_periods=max(8, roll // 3)).mean()
        r_std = rets.rolling(roll, min_periods=max(8, roll // 3)).std().replace(0.0, np.nan)
        rolling_sharpe = (r_mean / r_std * np.sqrt(ann)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        strat_cum = pd.Series(equity, index=dates) / max(float(equity[0]), 1e-9) - 1.0
        bh_cum = pd.Series(buy_hold, index=dates) / max(float(buy_hold[0]), 1e-9) - 1.0
        rolling_alpha = (strat_cum - bh_cum).fillna(0.0)

        monthly = []
        try:
            monthly_ret = (1.0 + pd.DataFrame({"strategy": rets, "buy_hold": bh_rets})).groupby(
                dates.to_period("M")
            ).prod() - 1.0
            for idx, row in monthly_ret.tail(48).iterrows():
                s = float(row.get("strategy", 0.0))
                b = float(row.get("buy_hold", 0.0))
                monthly.append(
                    {
                        "period": str(idx),
                        "strategy_return": s,
                        "buy_hold_return": b,
                        "alpha": s - b,
                    }
                )
        except Exception:
            monthly = []

        trades = pd.DataFrame(results.trade_log)
        action_breakdown = []
        if not trades.empty and "action" in trades.columns:
            by_action = trades.groupby(trades["action"].astype(str))
            for action, rows in by_action:
                action_breakdown.append(
                    {
                        "action": str(action),
                        "count": int(len(rows)),
                        "avg_commission": float(pd.to_numeric(rows.get("commission", 0.0), errors="coerce").fillna(0.0).mean()),
                        "avg_slippage": float(pd.to_numeric(rows.get("slippage", 0.0), errors="coerce").fillna(0.0).mean()),
                    }
                )
            action_breakdown.sort(key=lambda x: x["count"], reverse=True)

        hourly = []
        if intraday:
            hour_idx = pd.Series(dates.hour.to_numpy(dtype=int), index=dates)
            ret_by_hour = rets.groupby(hour_idx).mean()
            pos_by_hour = pd.Series(pos, index=dates).groupby(hour_idx).mean()
            trade_counts: Dict[int, int] = {}
            if not trades.empty and "date" in trades.columns:
                h = pd.to_datetime(trades["date"], errors="coerce").dt.hour.dropna().astype(int)
                trade_counts = h.value_counts().to_dict()
            for h in range(24):
                hourly.append(
                    {
                        "hour": int(h),
                        "avg_strategy_return": float(ret_by_hour.get(h, 0.0)),
                        "avg_position": float(pos_by_hour.get(h, 0.0)),
                        "trade_count": int(trade_counts.get(h, 0)),
                    }
                )

        rolling_records = []
        for i in range(len(dates)):
            rolling_records.append(
                {
                    "date": dates[i].isoformat(),
                    "rolling_sharpe": float(rolling_sharpe.iloc[i]),
                    "rolling_alpha": float(rolling_alpha.iloc[i]),
                    "drawdown": float((equity[i] / max(float(np.max(equity[: i + 1])), 1e-9)) - 1.0),
                    "equity": float(equity[i]),
                    "buy_hold_equity": float(buy_hold[i]),
                    "position": float(pos[i]) if i < len(pos) else 0.0,
                }
            )

        return {
            "rolling": rolling_records,
            "monthly": monthly,
            "hourly": hourly,
            "action_breakdown": action_breakdown,
            "risk": {
                "exposure_mean": float(np.nanmean(np.abs(pos))) if len(pos) else 0.0,
                "exposure_std": float(np.nanstd(pos)) if len(pos) else 0.0,
                "best_bar": float(np.nanmax(rets.to_numpy(dtype=float))) if len(rets) else 0.0,
                "worst_bar": float(np.nanmin(rets.to_numpy(dtype=float))) if len(rets) else 0.0,
            },
        }

    @staticmethod
    def _save_plots(out_dir: Path, frame: pd.DataFrame, results: Dict[str, object], summary: Dict, diagnostics: Dict) -> None:
        try:
            import matplotlib.pyplot as plt
        except Exception:
            return

        results_full = results["full"]
        dates = pd.to_datetime(frame.index)
        equity = np.asarray(results_full.equity_curve, dtype=float)
        regime_equity = np.asarray(results["regime_only"].equity_curve, dtype=float)
        ml_equity = np.asarray(results["ml_only"].equity_curve, dtype=float)
        buy_hold = np.asarray(results_full.buy_hold_equity, dtype=float)
        close = frame["Close"].to_numpy(dtype=float)
        positions = np.asarray(results_full.effective_positions, dtype=float)

        fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
        axes[0].plot(dates, equity, label="Strategy", color="#facc15", linewidth=2.0)
        axes[0].plot(dates, regime_equity, label="Regime Only", color="#34d399", linewidth=1.4, linestyle="-.")
        axes[0].plot(dates, ml_equity, label="ML Only", color="#f472b6", linewidth=1.4, linestyle=":")
        axes[0].plot(dates, buy_hold, label="Buy & Hold", color="#60a5fa", linewidth=1.8, linestyle="--")
        axes[0].set_ylabel("Equity")
        axes[0].set_title(
            f"{summary.get('symbol','?')} {summary.get('start','')} to {summary.get('end','')} "
            f"| alpha={summary.get('alpha', 0.0):.3f} | sharpe={summary.get('sharpe_ratio', 0.0):.2f}"
        )
        axes[0].legend(loc="upper left")
        axes[0].grid(alpha=0.2)

        axes[1].plot(dates, close, label="Close", color="#e5e7eb", linewidth=1.7)
        axes[1].set_ylabel("Price")
        axes[1].grid(alpha=0.2)

        pos_ax = axes[1].twinx()
        pos_ax.plot(dates, positions, label="Effective Position", color="#34d399", linewidth=1.2, alpha=0.8)
        pos_ax.set_ylabel("Position")
        pos_ax.axhline(0.0, color="#64748b", linewidth=0.8, alpha=0.6)
        pos_ax.grid(False)

        trade_df = pd.DataFrame(results_full.trade_log)
        if not trade_df.empty and "date" in trade_df and "action" in trade_df:
            trade_df["date"] = pd.to_datetime(trade_df["date"])
            buy_like = trade_df["action"].astype(str).str.upper().isin(["BUY", "COVER", "COVER_BUY"])
            sell_like = ~buy_like
            if buy_like.any():
                rows = trade_df.loc[buy_like]
                axes[1].scatter(rows["date"], rows["price"], marker="^", color="#22c55e", s=30, label="Buy/Cover")
            if sell_like.any():
                rows = trade_df.loc[sell_like]
                axes[1].scatter(rows["date"], rows["price"], marker="v", color="#ef4444", s=30, label="Sell/Short")

        lines, labels = axes[1].get_legend_handles_labels()
        lines2, labels2 = pos_ax.get_legend_handles_labels()
        axes[1].legend(lines + lines2, labels + labels2, loc="upper left")

        fig.tight_layout()
        fig.savefig(out_dir / "backtest_overview.png", dpi=160)
        plt.close(fig)

        # Risk diagnostics (rolling Sharpe/alpha + drawdown + return histogram)
        rolling = pd.DataFrame(diagnostics.get("rolling", []))
        if not rolling.empty:
            rolling["date"] = pd.to_datetime(rolling["date"], errors="coerce")
            fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
            axes[0].plot(rolling["date"], rolling["rolling_sharpe"], color="#22d3ee", linewidth=1.5, label="Rolling Sharpe")
            axes[0].plot(rolling["date"], rolling["rolling_alpha"], color="#f59e0b", linewidth=1.4, label="Rolling Alpha")
            axes[0].axhline(0.0, color="#64748b", linewidth=0.8, alpha=0.7)
            axes[0].legend(loc="upper left")
            axes[0].grid(alpha=0.25)
            axes[0].set_ylabel("Signal")

            axes[1].fill_between(
                rolling["date"],
                rolling["drawdown"],
                0.0,
                color="#ef4444",
                alpha=0.25,
                label="Drawdown",
            )
            axes[1].set_ylabel("Drawdown")
            axes[1].grid(alpha=0.25)
            axes[1].legend(loc="lower left")

            daily_ret = np.asarray(results_full.daily_returns, dtype=float)
            axes[2].hist(daily_ret[np.isfinite(daily_ret)], bins=50, color="#60a5fa", alpha=0.8)
            axes[2].set_ylabel("Count")
            axes[2].set_xlabel("Return")
            axes[2].grid(alpha=0.25)

            fig.tight_layout()
            fig.savefig(out_dir / "risk_diagnostics.png", dpi=160)
            plt.close(fig)

        # Action diagnostics bar chart.
        action_df = pd.DataFrame(diagnostics.get("action_breakdown", []))
        if not action_df.empty:
            fig, ax = plt.subplots(figsize=(12, 4.5))
            ax.bar(action_df["action"], action_df["count"], color="#a78bfa", alpha=0.85)
            ax.set_ylabel("Trade Count")
            ax.set_title("Action Breakdown")
            ax.grid(axis="y", alpha=0.25)
            fig.tight_layout()
            fig.savefig(out_dir / "trade_diagnostics.png", dpi=160)
            plt.close(fig)

        # Intraday hour profile.
        if bool(summary.get("is_intraday", False)):
            hourly_df = pd.DataFrame(diagnostics.get("hourly", []))
            if not hourly_df.empty:
                fig, ax1 = plt.subplots(figsize=(12, 4.5))
                ax1.bar(hourly_df["hour"], hourly_df["trade_count"], color="#f59e0b", alpha=0.7, label="Trades")
                ax1.set_xlabel("Hour of Day")
                ax1.set_ylabel("Trade Count")
                ax1.grid(axis="y", alpha=0.2)
                ax2 = ax1.twinx()
                ax2.plot(hourly_df["hour"], hourly_df["avg_strategy_return"], color="#22d3ee", linewidth=2.0, label="Avg Return")
                ax2.set_ylabel("Avg Strategy Return")
                fig.tight_layout()
                fig.savefig(out_dir / "intraday_hour_profile.png", dpi=160)
                plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run GBM-first backtest")
    parser.add_argument("--symbol", type=str, default="AAPL")
    parser.add_argument("--start", type=str, default="2020-01-01")
    parser.add_argument("--end", type=str, default="2024-12-31")
    parser.add_argument("--mode", type=str, default="gbm_only", choices=["gbm_only", "ensemble"])
    parser.add_argument("--include-sentiment", action="store_true")
    parser.add_argument("--model-variant", type=str, default="gbm")
    parser.add_argument("--data-period", type=str, default="max")
    parser.add_argument("--data-interval", type=str, default="1d")
    parser.add_argument("--max-long", type=float, default=1.6)
    parser.add_argument("--max-short", type=float, default=0.25)
    parser.add_argument("--commission", type=float, default=0.0005)
    parser.add_argument("--slippage", type=float, default=0.0003)
    parser.add_argument("--vol-target-annual", type=float, default=0.40)
    parser.add_argument("--min-vol-scale", type=float, default=0.90)
    parser.add_argument("--max-vol-scale", type=float, default=1.60)
    parser.add_argument("--warmup-days", type=int, default=252)
    parser.add_argument("--min-eval-days", type=int, default=20)
    parser.add_argument("--min-position-change", type=float, default=0.0)
    args = parser.parse_args()

    cfg = BacktestConfig(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        mode=args.mode,
        include_sentiment=args.include_sentiment,
        model_variant=args.model_variant,
        data_period=args.data_period,
        data_interval=args.data_interval,
        max_long=args.max_long,
        max_short=args.max_short,
        commission_pct=args.commission,
        slippage_pct=args.slippage,
        vol_target_annual=args.vol_target_annual,
        min_vol_scale=args.min_vol_scale,
        max_vol_scale=args.max_vol_scale,
        warmup_days=args.warmup_days,
        min_eval_days=args.min_eval_days,
        min_position_change=args.min_position_change,
    )

    summary = UnifiedBacktester(cfg).run()
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
