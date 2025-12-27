"""Model registry utilities for the AI Stocks FastAPI service."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional
import io
import pickle
import zipfile

@dataclass
class ModelMetadata:
    id: str
    symbol: str
    createdAt: str
    fusionModeDefault: str
    metrics: Dict[str, float]
    scalers: Optional[Dict[str, str]]
    sequenceLength: int
    ensembleSize: int
    notes: Optional[str]

class ModelRegistry:
    def __init__(self, saved_models_dir: Path):
        self.saved_models_dir = saved_models_dir

    def list_models(self) -> List[ModelMetadata]:
        entries: List[ModelMetadata] = []
        if not self.saved_models_dir.exists():
            return entries
        for meta_path in self.saved_models_dir.glob("*_regressor_final_metadata.pkl"):
            symbol = meta_path.name.split("_")[0]
            entry = self._build_model_entry(symbol, meta_path)
            if entry:
                entries.append(entry)
        return sorted(entries, key=lambda item: (item.symbol, item.createdAt or ""))

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        for model in self.list_models():
            if model.id == model_id:
                return model
        return None

    def package_artifacts(self, symbol: str) -> io.BytesIO:
        artifacts = [p for p in self.saved_models_dir.iterdir() if p.name.startswith(f"{symbol}_")]
        if not artifacts:
            raise FileNotFoundError(f"No artifacts found for {symbol}")
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in artifacts:
                if path.is_file():
                    archive.write(path, arcname=path.name)
                else:
                    for child in path.rglob("*"):
                        if child.is_file():
                            arcname = Path(path.name) / child.relative_to(path)
                            archive.write(child, arcname=str(arcname))
        buffer.seek(0)
        return buffer

    def _build_model_entry(self, symbol: str, meta_path: Path) -> Optional[ModelMetadata]:
        reg_meta = self._load_pickle(meta_path)
        if not reg_meta:
            return None
        classifier_meta_path = self.saved_models_dir / f"{symbol}_binary_classifiers_final_metadata.pkl"
        classifier_meta = self._load_pickle(classifier_meta_path) if classifier_meta_path.exists() else None
        metrics = {
            "cumulativeReturn": 0.0,
            "sharpeRatio": 0.0,
            "maxDrawdown": 0.0,
            "winRate": 0.0,
            "averageTradeProfit": 0.0,
            "totalTrades": 0.0,
            "directionalAccuracy": float(reg_meta.get("val_dir_acc", 0.0)),
            "correlation": 0.0,
            "smape": float(reg_meta.get("val_smape", 0.0)),
            "rmse": float(reg_meta.get("val_rmse", 0.0)),
        }
        feature_scaler_path = self.saved_models_dir / f"{symbol}_1d_regressor_final_feature_scaler.pkl"
        target_scaler_path = self.saved_models_dir / f"{symbol}_1d_regressor_final_target_scaler.pkl"
        scalers = None
        if feature_scaler_path.exists() or target_scaler_path.exists():
            scalers = {
                "feature": feature_scaler_path.name if feature_scaler_path.exists() else None,
                "target": target_scaler_path.name if target_scaler_path.exists() else None,
            }
        ensemble_size = self._infer_ensemble_size(symbol, classifier_meta)
        fusion_mode = "hybrid" if classifier_meta else "regressor"
        trained_date = str(reg_meta.get("trained_date") or "")
        notes = self._build_notes(reg_meta, classifier_meta)
        model_id = f"{symbol.lower()}-{reg_meta.get('model_type', 'regressor')}"
        return ModelMetadata(
            id=model_id,
            symbol=symbol,
            createdAt=trained_date,
            fusionModeDefault=fusion_mode,
            metrics=metrics,
            scalers=scalers,
            sequenceLength=int(reg_meta.get("sequence_length", 60)),
            ensembleSize=ensemble_size,
            notes=notes
        )

    def _infer_ensemble_size(self, symbol: str, classifier_meta: Optional[dict]) -> int:
        if classifier_meta:
            ensemble_info = classifier_meta.get("ensemble")
            if isinstance(ensemble_info, dict):
                members = ensemble_info.get("members")
                if isinstance(members, Iterable):
                    members_list = list(members)
                    if members_list:
                        return len(members_list)
            extra_sell = list(self.saved_models_dir.glob(f"{symbol}_is_sell_classifier_final*weights.h5"))
            extra_buy = list(self.saved_models_dir.glob(f"{symbol}_is_buy_classifier_final*weights.h5"))
            base_count = 1
            extra_members = max(len(extra_sell), len(extra_buy)) - 1
            return max(base_count + max(extra_members, 0), 2)
        return 1

    @staticmethod
    def _build_notes(reg_meta: dict, classifier_meta: Optional[dict]) -> Optional[str]:
        notes: List[str] = []
        loss = reg_meta.get("loss_name")
        if loss:
            notes.append(f"loss={loss}")
        if classifier_meta and classifier_meta.get("thresholds"):
            thresholds = classifier_meta["thresholds"]
            buy_thr = thresholds.get("buy_optimal", thresholds.get("default"))
            sell_thr = thresholds.get("sell_optimal", thresholds.get("default"))
            notes.append(f"thresholds(buy={float(buy_thr):.2f}, sell={float(sell_thr):.2f})")
        return "; ".join(notes) if notes else None

    @staticmethod
    def _load_pickle(path: Path) -> Optional[dict]:
        try:
            with path.open("rb") as fh:
                return pickle.load(fh)
        except Exception:
            return None
