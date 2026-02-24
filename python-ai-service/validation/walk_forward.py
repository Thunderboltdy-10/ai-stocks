"""Walk-forward validation helpers for GBM models."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

from data.data_splitter import PurgedTimeSeriesSplit


@dataclass
class FoldMetrics:
    fold: int
    rmse: float
    mae: float
    dir_acc: float
    sharpe: float
    pred_std: float
    positive_pct: float


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or len(b) < 2:
        return 0.0
    if float(np.std(a)) <= 1e-12 or float(np.std(b)) <= 1e-12:
        return 0.0
    corr = np.corrcoef(a, b)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(corr)


def _sharpe(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return 0.0
    std = float(np.std(returns))
    if std <= 1e-12:
        return 0.0
    return float(np.sqrt(252.0) * np.mean(returns) / std)


def _cost_adjusted_trade_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    annualization_factor: float = 252.0,
    cost_per_turn: float = 0.0008,
) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=float).reshape(-1)
    pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y) == 0 or len(pred) == 0 or len(y) != len(pred):
        return {
            "net_sharpe": 0.0,
            "net_return": 0.0,
            "gross_return": 0.0,
            "turnover": 0.0,
            "trade_win_rate": 0.0,
        }

    std_pred = float(max(1e-9, np.nanstd(pred)))
    strength = np.tanh(pred / (1.5 * std_pred))
    deadband = np.abs(pred) < (0.35 * std_pred)
    strength[deadband] = 0.0
    position = pd.Series(strength).ewm(alpha=0.30, adjust=False).mean().to_numpy(dtype=float)
    position = np.clip(position, -1.0, 1.0)

    lag = np.zeros_like(position)
    lag[1:] = position[:-1]
    turnover = np.abs(np.diff(position, prepend=0.0))

    gross = lag * y
    net = gross - float(max(0.0, cost_per_turn)) * turnover

    gross_eq = np.cumprod(1.0 + gross)
    net_eq = np.cumprod(1.0 + net)
    gross_return = float(gross_eq[-1] - 1.0)
    net_return = float(net_eq[-1] - 1.0)

    net_std = float(np.std(net))
    ann = float(max(1.0, annualization_factor))
    net_sharpe = float(np.sqrt(ann) * np.mean(net) / net_std) if net_std > 1e-12 else 0.0
    win_rate = float(np.mean(net > 0.0))
    return {
        "net_sharpe": net_sharpe,
        "net_return": net_return,
        "gross_return": gross_return,
        "turnover": float(np.mean(turnover)),
        "trade_win_rate": win_rate,
    }


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    annualization_factor: float = 252.0,
    cost_per_turn: float = 0.0008,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    dir_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    pred_std = float(np.std(y_pred))
    positive_pct = float((y_pred > 0).mean())
    pnl = np.sign(y_pred) * y_true
    sharpe = _sharpe(pnl)
    cost_metrics = _cost_adjusted_trade_metrics(
        y_true,
        y_pred,
        annualization_factor=annualization_factor,
        cost_per_turn=cost_per_turn,
    )

    return {
        "rmse": rmse,
        "mae": mae,
        "dir_acc": dir_acc,
        "pred_std": pred_std,
        "positive_pct": positive_pct,
        "sharpe": sharpe,
        "ic": _safe_corr(y_true, y_pred),
        **cost_metrics,
    }


def _feature_stability(importances: List[pd.Series]) -> Dict[str, float]:
    if not importances:
        return {"stability": 0.0, "top_overlap": 0.0}

    # Pairwise rank-correlation of aligned importance vectors.
    aligned = pd.concat(importances, axis=1).fillna(0.0)
    corr_vals: List[float] = []
    for i in range(aligned.shape[1]):
        for j in range(i + 1, aligned.shape[1]):
            corr = aligned.iloc[:, i].corr(aligned.iloc[:, j], method="spearman")
            if np.isfinite(corr):
                corr_vals.append(float(corr))

    top_sets = [set(s.sort_values(ascending=False).head(20).index) for s in importances]
    overlaps: List[float] = []
    for i in range(len(top_sets)):
        for j in range(i + 1, len(top_sets)):
            union = len(top_sets[i] | top_sets[j])
            if union == 0:
                continue
            overlaps.append(len(top_sets[i] & top_sets[j]) / union)

    return {
        "stability": float(np.mean(corr_vals)) if corr_vals else 0.0,
        "top_overlap": float(np.mean(overlaps)) if overlaps else 0.0,
    }


def run_walk_forward_validation(
    X: np.ndarray,
    y: np.ndarray,
    fit_predict_fn: Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], Tuple[np.ndarray, Optional[pd.Series]]],
    splitter: PurgedTimeSeriesSplit,
    eval_kwargs: Optional[Dict[str, float]] = None,
) -> Dict:
    """Run purged walk-forward validation using a model-specific fit/predict callback.

    `fit_predict_fn` receives `(X_train, y_train, X_test, y_test)` and returns
    `(y_pred, feature_importance_series_or_none)`.
    """
    fold_rows: List[FoldMetrics] = []
    oof_pred = np.full(len(X), np.nan, dtype=float)
    feature_importances: List[pd.Series] = []

    for fold_idx, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        y_pred, importance = fit_predict_fn(X_train, y_train, X_test, y_test)
        metrics = evaluate_predictions(y_test, y_pred, **(eval_kwargs or {}))
        oof_pred[test_idx] = y_pred

        fold_rows.append(
            FoldMetrics(
                fold=fold_idx,
                rmse=metrics["rmse"],
                mae=metrics["mae"],
                dir_acc=metrics["dir_acc"],
                sharpe=metrics["sharpe"],
                pred_std=metrics["pred_std"],
                positive_pct=metrics["positive_pct"],
            )
        )

        if importance is not None:
            feature_importances.append(importance)

    if not fold_rows:
        raise ValueError("No folds were generated by the splitter")

    fold_df = pd.DataFrame([f.__dict__ for f in fold_rows])
    agg = {
        "rmse_mean": float(fold_df["rmse"].mean()),
        "mae_mean": float(fold_df["mae"].mean()),
        "dir_acc_mean": float(fold_df["dir_acc"].mean()),
        "sharpe_mean": float(fold_df["sharpe"].mean()),
        "pred_std_mean": float(fold_df["pred_std"].mean()),
        "positive_pct_mean": float(fold_df["positive_pct"].mean()),
    }

    valid_oof = np.isfinite(oof_pred)
    if np.any(valid_oof):
        oof_metrics = evaluate_predictions(y[valid_oof], oof_pred[valid_oof], **(eval_kwargs or {}))
        agg["net_sharpe_mean"] = float(oof_metrics.get("net_sharpe", 0.0))
        agg["net_return_mean"] = float(oof_metrics.get("net_return", 0.0))
        agg["turnover_mean"] = float(oof_metrics.get("turnover", 0.0))
    else:
        agg["net_sharpe_mean"] = 0.0
        agg["net_return_mean"] = 0.0
        agg["turnover_mean"] = 0.0

    return {
        "folds": fold_df.to_dict(orient="records"),
        "aggregate": agg,
        "oof_predictions": oof_pred,
        "feature_stability": _feature_stability(feature_importances),
    }
