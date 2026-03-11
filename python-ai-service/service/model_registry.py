"""Model registry for the GBM-first production path."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import io
import json
import zipfile

from inference.variant_quality import load_training_metadata, score_variant_quality


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
    modelVariant: Optional[str] = None
    dataInterval: str = "1d"
    dataPeriod: str = "max"
    qualityGatePassed: bool = False
    qualityScore: float = 0.0
    qualityReasons: Optional[List[str]] = None
    holdoutMetricSource: Optional[str] = None


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
            model = self._build_entry(symbol)
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

    def _build_entry(self, symbol: str) -> Optional[ModelMetadata]:
        variant = self._resolve_primary_variant(symbol)
        if not variant:
            return None

        metadata = load_training_metadata(symbol, variant)
        if not metadata:
            return None

        interval = str(metadata.get("data_interval", "1d"))
        intraday = interval != "1d"
        quality = score_variant_quality(metadata, intraday=intraday)

        metrics = {
            "cumulativeReturn": float(quality.metrics.get("net_return", 0.0)),
            "sharpeRatio": float(quality.metrics.get("net_sharpe", 0.0)),
            "maxDrawdown": float(metadata.get("max_drawdown", 0.0)),
            "winRate": float(quality.metrics.get("trade_win_rate", quality.metrics.get("dir_acc", 0.0))),
            "averageTradeProfit": 0.0,
            "totalTrades": 0.0,
            "directionalAccuracy": float(quality.metrics.get("dir_acc", 0.0)),
            "correlation": float(quality.metrics.get("ic", 0.0)),
            "smape": 0.0,
            "rmse": float(metadata.get("holdout", {}).get(quality.selected_source, {}).get("rmse", 0.0)),
        }

        notes_parts = [
            f"variant={variant}",
            f"wfe={quality.metrics.get('wfe', 0.0):.1f}",
            f"interval={interval}",
        ]
        notes_parts.append("status=ready" if not quality.reasons else "status=needs_review")

        created_at = (
            str(metadata.get("trained_at") or metadata.get("created_at") or "")
            or str((self.saved_models_dir / symbol / variant / "training_metadata.json").stat().st_mtime)
        )

        return ModelMetadata(
            id=f"{symbol.lower()}-{variant}",
            symbol=symbol,
            createdAt=created_at,
            fusionModeDefault="gbm_only",
            metrics=metrics,
            scalers={"feature": "xgb_scaler.joblib", "target": "N/A"},
            sequenceLength=int(metadata.get("sequence_length", 60) or 60),
            ensembleSize=1,
            notes="; ".join(notes_parts),
            modelVariant=variant,
            dataInterval=interval,
            dataPeriod=str(metadata.get("data_period", "max")),
            qualityGatePassed=bool(quality.passed),
            qualityScore=float(quality.score),
            qualityReasons=list(quality.reasons),
            holdoutMetricSource=quality.selected_source,
        )

    def _resolve_primary_variant(self, symbol: str) -> Optional[str]:
        symbol_dir = self.saved_models_dir / symbol
        if not symbol_dir.exists():
            return None

        for registry_name in ("daily_variant_registry.json", "intraday_variant_registry.json"):
            registry_path = symbol_dir / registry_name
            registry = self._load_json(registry_path)
            candidate = str(registry.get("overall_variant", "")).strip()
            if candidate and (symbol_dir / candidate / "training_metadata.json").exists():
                return candidate

        fallback_candidates = []
        for variant_dir in sorted(symbol_dir.glob("gbm*")):
            if not variant_dir.is_dir():
                continue
            if not (variant_dir / "training_metadata.json").exists():
                continue
            fallback_candidates.append(variant_dir.name)

        if fallback_candidates:
            ranked = []
            for variant in fallback_candidates:
                metadata = load_training_metadata(symbol, variant)
                interval = str(metadata.get("data_interval", "1d"))
                intraday = interval != "1d"
                quality = score_variant_quality(metadata, intraday=intraday)
                ranked.append((quality.passed, quality.score, variant))
            ranked.sort(reverse=True)
            return ranked[0][2]
        return None

    @staticmethod
    def _load_json(path: Path) -> Dict:
        if not path.exists():
            return {}
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return {}
