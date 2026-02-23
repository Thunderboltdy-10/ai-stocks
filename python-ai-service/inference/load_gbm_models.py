"""Load and run GBM artifacts from `saved_models/{SYMBOL}/gbm/`."""

from __future__ import annotations

import json
import logging
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GBMModelBundle:
    symbol: str
    feature_columns: List[str]
    xgb_model: Any
    lgb_model: Any
    xgb_scaler: Any
    lgb_scaler: Any
    metadata: Dict[str, Any]

    def has_xgb(self) -> bool:
        return self.xgb_model is not None and self.xgb_scaler is not None

    def has_lgb(self) -> bool:
        return self.lgb_model is not None and self.lgb_scaler is not None


def _legacy_path(model_dir: Path, new_name: str, legacy_name: str) -> Path:
    new_path = model_dir / new_name
    if new_path.exists():
        return new_path
    return model_dir / legacy_name


def load_gbm_models(symbol: str, model_dir: Optional[Path] = None) -> Tuple[Optional[GBMModelBundle], Dict[str, Any]]:
    symbol = symbol.upper()
    model_dir = model_dir or Path("saved_models") / symbol / "gbm"

    metadata: Dict[str, Any] = {"symbol": symbol, "model_dir": str(model_dir), "errors": []}
    if not model_dir.exists():
        metadata["errors"].append(f"Model directory does not exist: {model_dir}")
        return None, metadata

    feature_path = _legacy_path(model_dir, "feature_columns.pkl", "feature_columns.pkl")
    xgb_model_path = _legacy_path(model_dir, "xgb_model.joblib", "xgb_reg.joblib")
    lgb_model_path = _legacy_path(model_dir, "lgb_model.joblib", "lgb_reg.joblib")
    xgb_scaler_path = _legacy_path(model_dir, "xgb_scaler.joblib", "xgb_scaler.joblib")
    lgb_scaler_path = _legacy_path(model_dir, "lgb_scaler.joblib", "lgb_scaler.joblib")
    train_meta_path = _legacy_path(model_dir, "training_metadata.json", "training_metadata.json")

    if not feature_path.exists():
        metadata["errors"].append("Missing feature_columns.pkl")
        return None, metadata

    with feature_path.open("rb") as fh:
        feature_columns = pickle.load(fh)

    xgb_model = xgb_scaler = lgb_model = lgb_scaler = None

    try:
        if xgb_model_path.exists() and xgb_scaler_path.exists():
            xgb_model = joblib.load(xgb_model_path)
            xgb_scaler = joblib.load(xgb_scaler_path)
    except Exception as exc:
        metadata["errors"].append(f"XGB load failed: {exc}")

    try:
        if lgb_model_path.exists() and lgb_scaler_path.exists():
            lgb_model = joblib.load(lgb_model_path)
            lgb_scaler = joblib.load(lgb_scaler_path)
    except Exception as exc:
        metadata["errors"].append(f"LGB load failed: {exc}")

    train_meta: Dict[str, Any] = {}
    if train_meta_path.exists():
        try:
            with train_meta_path.open("r", encoding="utf-8") as fh:
                train_meta = json.load(fh)
        except Exception as exc:
            metadata["errors"].append(f"training_metadata read failed: {exc}")

    bundle = GBMModelBundle(
        symbol=symbol,
        feature_columns=feature_columns,
        xgb_model=xgb_model,
        lgb_model=lgb_model,
        xgb_scaler=xgb_scaler,
        lgb_scaler=lgb_scaler,
        metadata=train_meta,
    )

    if not bundle.has_xgb() and not bundle.has_lgb():
        metadata["errors"].append("No GBM models available")
        return None, metadata

    return bundle, metadata


def _ensure_2d(features: np.ndarray) -> np.ndarray:
    arr = np.asarray(features, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def predict_with_gbm(
    bundle: GBMModelBundle,
    features: np.ndarray,
    model: str = "ensemble",
    weights: Optional[Dict[str, float]] = None,
    return_components: bool = False,
) -> np.ndarray | Dict[str, np.ndarray]:
    X = _ensure_2d(features)
    outputs: Dict[str, np.ndarray] = {}

    if bundle.has_xgb():
        X_xgb = bundle.xgb_scaler.transform(X)
        outputs["xgb"] = np.asarray(bundle.xgb_model.predict(X_xgb), dtype=float)

    if bundle.has_lgb():
        X_lgb = bundle.lgb_scaler.transform(X)
        outputs["lgb"] = np.asarray(bundle.lgb_model.predict(X_lgb), dtype=float)

    if return_components:
        return outputs

    if not outputs:
        raise ValueError("No GBM model outputs available")

    model = model.lower()
    if model == "xgb" and "xgb" in outputs:
        return outputs["xgb"]
    if model == "lgb" and "lgb" in outputs:
        return outputs["lgb"]

    if model in {"ensemble", "avg", "weighted"}:
        if "xgb" in outputs and "lgb" in outputs:
            w = weights or bundle.metadata.get("ensemble_weights", {"xgb": 0.5, "lgb": 0.5})
            total = max(w.get("xgb", 0.0) + w.get("lgb", 0.0), 1e-9)
            return (w.get("xgb", 0.0) * outputs["xgb"] + w.get("lgb", 0.0) * outputs["lgb"]) / total
        return list(outputs.values())[0]

    return list(outputs.values())[0]


def predict_with_ensemble(
    gbm_bundle: GBMModelBundle,
    regressor_pred: Optional[np.ndarray],
    features: np.ndarray,
    fusion_mode: str = "gbm_only",
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """Compatibility helper used by a few legacy scripts."""
    gbm_pred = predict_with_gbm(gbm_bundle, features, model="ensemble", weights=weights)
    fused = np.asarray(gbm_pred, dtype=float)

    if fusion_mode.lower() in {"ensemble", "weighted"} and regressor_pred is not None:
        reg = np.asarray(regressor_pred, dtype=float).reshape(-1)
        if len(reg) == len(fused):
            fused = 0.7 * fused + 0.3 * reg

    return {
        "fusion_mode": fusion_mode,
        "predictions": {"gbm": fused},
        "weights_used": weights or {"xgb": 0.5, "lgb": 0.5},
        "fused_prediction": fused,
    }
