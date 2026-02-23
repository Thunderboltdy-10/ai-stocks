"""Model path helpers (rewritten, no legacy flat-path support)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def get_saved_models_root() -> Path:
    return Path(__file__).resolve().parent.parent / "saved_models"


@dataclass
class GBMPaths:
    base: Path

    @property
    def xgb_model(self) -> Path:
        return self.base / "xgb_model.joblib"

    @property
    def lgb_model(self) -> Path:
        return self.base / "lgb_model.joblib"

    @property
    def xgb_scaler(self) -> Path:
        return self.base / "xgb_scaler.joblib"

    @property
    def lgb_scaler(self) -> Path:
        return self.base / "lgb_scaler.joblib"

    @property
    def feature_columns(self) -> Path:
        return self.base / "feature_columns.pkl"

    @property
    def training_metadata(self) -> Path:
        return self.base / "training_metadata.json"

    @property
    def hyperparameters(self) -> Path:
        return self.base / "hyperparameters.json"

    @property
    def feature_importance(self) -> Path:
        return self.base / "feature_importance.json"


@dataclass
class LSTMPaths:
    base: Path

    @property
    def model(self) -> Path:
        return self.base / "model.keras"

    @property
    def scaler(self) -> Path:
        return self.base / "scaler.joblib"

    @property
    def feature_columns(self) -> Path:
        return self.base / "feature_columns.pkl"

    @property
    def metadata(self) -> Path:
        return self.base / "training_metadata.json"


class ModelPaths:
    def __init__(self, symbol: str, root: Optional[Path] = None) -> None:
        self.symbol = symbol.upper()
        self.root = root or get_saved_models_root()
        self.symbol_dir = self.root / self.symbol
        self.gbm = GBMPaths(self.symbol_dir / "gbm")
        self.lstm = LSTMPaths(self.symbol_dir / "lstm")

    def ensure_dirs(self) -> None:
        self.gbm.base.mkdir(parents=True, exist_ok=True)
        self.lstm.base.mkdir(parents=True, exist_ok=True)

    def exists(self) -> bool:
        return self.symbol_dir.exists()

    def has_gbm(self) -> bool:
        return self.gbm.xgb_model.exists() or self.gbm.lgb_model.exists()

    def has_lstm(self) -> bool:
        return self.lstm.model.exists()


def find_model_path(symbol: str, model_type: str, artifact: str, root: Optional[Path] = None) -> Optional[Path]:
    paths = ModelPaths(symbol, root=root)
    accessor = getattr(paths, model_type, None)
    if accessor is None:
        return None
    path = getattr(accessor, artifact, None)
    if path is None:
        return None
    return path if path.exists() else None
