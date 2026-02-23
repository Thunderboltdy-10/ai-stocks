"""Model registry for GBM/LSTM directory layout."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import io
import json
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

        for symbol_dir in sorted(self.saved_models_dir.glob("*")):
            if not symbol_dir.is_dir():
                continue
            symbol = symbol_dir.name.upper()

            gbm_meta = symbol_dir / "gbm" / "training_metadata.json"
            lstm_meta = symbol_dir / "lstm" / "training_metadata.json"

            if not gbm_meta.exists() and not lstm_meta.exists():
                continue

            model = self._build_entry(symbol, gbm_meta, lstm_meta)
            if model is not None:
                entries.append(model)

        return entries

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        for model in self.list_models():
            if model.id == model_id:
                return model
        return None

    def package_artifacts(self, symbol: str) -> io.BytesIO:
        symbol = symbol.upper()
        symbol_dir = self.saved_models_dir / symbol
        if not symbol_dir.exists():
            raise FileNotFoundError(f"No artifacts found for {symbol}")

        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for file in symbol_dir.rglob("*"):
                if file.is_file():
                    archive.write(file, arcname=str(file.relative_to(symbol_dir.parent)))
        buffer.seek(0)
        return buffer

    def _build_entry(self, symbol: str, gbm_meta_path: Path, lstm_meta_path: Path) -> Optional[ModelMetadata]:
        gbm_meta = self._load_json(gbm_meta_path) if gbm_meta_path.exists() else {}
        lstm_meta = self._load_json(lstm_meta_path) if lstm_meta_path.exists() else {}

        if not gbm_meta and not lstm_meta:
            return None

        has_gbm = bool(gbm_meta)
        has_lstm = bool(lstm_meta)

        if has_gbm and has_lstm:
            fusion_default = "ensemble"
        elif has_gbm:
            fusion_default = "gbm_only"
        else:
            fusion_default = "lstm_only"

        holdout = gbm_meta.get("holdout", {}).get("ensemble", {}) if has_gbm else {}

        metrics = {
            "cumulativeReturn": float(holdout.get("sharpe", 0.0)),
            "sharpeRatio": float(holdout.get("sharpe", 0.0)),
            "maxDrawdown": float(gbm_meta.get("max_drawdown", 0.0)),
            "winRate": float(holdout.get("dir_acc", 0.0)),
            "averageTradeProfit": 0.0,
            "totalTrades": 0.0,
            "directionalAccuracy": float(holdout.get("dir_acc", 0.0)),
            "correlation": float(holdout.get("ic", 0.0)),
            "smape": 0.0,
            "rmse": float(holdout.get("rmse", 0.0)),
        }

        scalers = None
        gbm_dir = self.saved_models_dir / symbol / "gbm"
        if gbm_dir.exists():
            scalers = {
                "feature": "xgb_scaler.joblib",
                "target": "N/A",
            }

        created_at = gbm_meta.get("trained_at") or gbm_meta.get("created_at") or ""
        if not created_at:
            created_at = str(Path(gbm_meta_path).stat().st_mtime)

        notes_parts = []
        if has_gbm:
            notes_parts.append(f"gbm_features={gbm_meta.get('n_features_selected', 'n/a')}")
            notes_parts.append(f"wfe={gbm_meta.get('wfe', 'n/a')}")
        if has_lstm:
            notes_parts.append("lstm_available=true")

        notes = "; ".join(notes_parts) if notes_parts else None

        return ModelMetadata(
            id=f"{symbol.lower()}-{fusion_default}",
            symbol=symbol,
            createdAt=str(created_at),
            fusionModeDefault=fusion_default,
            metrics=metrics,
            scalers=scalers,
            sequenceLength=60,
            ensembleSize=2 if has_gbm and has_lstm else 1,
            notes=notes,
        )

    @staticmethod
    def _load_json(path: Path) -> Dict:
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}
