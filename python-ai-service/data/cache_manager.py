"""Lightweight data cache manager for deterministic training and inference."""

from __future__ import annotations

import json
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from data.data_fetcher import fetch_stock_data
from data.feature_engineer import EXPECTED_FEATURE_COUNT, engineer_features, get_feature_columns
from data.target_engineering import prepare_training_data

logger = logging.getLogger(__name__)


class DataCacheManager:
    """Caches raw, engineered, and training-ready data per symbol."""

    def __init__(
        self,
        cache_dir: str = "cache",
        cache_lifetime_hours: int = 72,
        cache_version: str = "gbm_v2",
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_lifetime = timedelta(hours=cache_lifetime_hours)
        self.cache_version = cache_version

    def _symbol_dir(self, symbol: str) -> Path:
        path = self.cache_dir / symbol.upper()
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _paths(self, symbol: str) -> Dict[str, Path]:
        base = self._symbol_dir(symbol)
        return {
            "raw": base / "raw_data.pkl",
            "engineered": base / "engineered_features.pkl",
            "prepared": base / "prepared_training.pkl",
            "features": base / "feature_columns.pkl",
            "metadata": base / "metadata.json",
        }

    def _cache_valid(self, metadata: Dict, include_sentiment: bool, horizons: List[int]) -> bool:
        if metadata.get("cache_version") != self.cache_version:
            return False
        if metadata.get("include_sentiment") != include_sentiment:
            return False
        if metadata.get("horizons") != horizons:
            return False

        expected = len(get_feature_columns(include_sentiment=include_sentiment))
        if metadata.get("feature_count") != expected:
            return False

        created_at = metadata.get("created_at")
        if not created_at:
            return False

        age = datetime.now() - datetime.fromisoformat(created_at)
        return age <= self.cache_lifetime

    def _load_cache(
        self,
        symbol: str,
        include_sentiment: bool,
        horizons: List[int],
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]]:
        paths = self._paths(symbol)
        if not all(p.exists() for p in paths.values()):
            return None

        try:
            with paths["metadata"].open("r", encoding="utf-8") as fh:
                metadata = json.load(fh)
        except Exception as exc:
            logger.warning("Failed reading cache metadata for %s: %s", symbol, exc)
            return None

        if not self._cache_valid(metadata, include_sentiment=include_sentiment, horizons=horizons):
            logger.info("Cache invalid/stale for %s; refreshing", symbol)
            return None

        try:
            raw_df = pd.read_pickle(paths["raw"])
            engineered_df = pd.read_pickle(paths["engineered"])
            prepared_df = pd.read_pickle(paths["prepared"])
            with paths["features"].open("rb") as fh:
                feature_cols = pickle.load(fh)
        except Exception as exc:
            logger.warning("Failed loading cache files for %s: %s", symbol, exc)
            return None

        return raw_df, engineered_df, prepared_df, feature_cols

    def save_cache(
        self,
        symbol: str,
        raw_df: pd.DataFrame,
        engineered_df: pd.DataFrame,
        prepared_df: pd.DataFrame,
        feature_cols: list,
        include_sentiment: bool = False,
        horizons: List[int] | None = None,
    ) -> None:
        paths = self._paths(symbol)
        horizons = horizons or [1]

        raw_df.to_pickle(paths["raw"])
        engineered_df.to_pickle(paths["engineered"])
        prepared_df.to_pickle(paths["prepared"])
        with paths["features"].open("wb") as fh:
            pickle.dump(feature_cols, fh)

        metadata = {
            "symbol": symbol.upper(),
            "created_at": datetime.now().isoformat(),
            "cache_version": self.cache_version,
            "include_sentiment": include_sentiment,
            "horizons": horizons,
            "feature_count": len(feature_cols),
            "expected_feature_count": len(get_feature_columns(include_sentiment=include_sentiment)),
            "raw_rows": len(raw_df),
            "engineered_rows": len(engineered_df),
            "prepared_rows": len(prepared_df),
        }
        with paths["metadata"].open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)

    def get_or_fetch_data(
        self,
        symbol: str,
        include_sentiment: bool = False,
        force_refresh: bool = False,
        period: str = "max",
        horizons: List[int] | None = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list]:
        symbol = symbol.upper()
        horizons = horizons or [1]

        if not force_refresh:
            cached = self._load_cache(
                symbol,
                include_sentiment=include_sentiment,
                horizons=horizons,
            )
            if cached is not None:
                return cached

        raw_df = fetch_stock_data(symbol=symbol, period=period)
        engineered_df = engineer_features(raw_df, symbol=symbol, include_sentiment=include_sentiment)
        prepared_df, feature_cols = prepare_training_data(
            engineered_df,
            horizons=horizons,
            shift_features=True,
            include_sentiment=include_sentiment,
        )

        if not feature_cols:
            raise ValueError(f"No feature columns created for {symbol}")

        if len(feature_cols) != EXPECTED_FEATURE_COUNT and not include_sentiment:
            raise ValueError(
                f"Feature contract mismatch for {symbol}. expected={EXPECTED_FEATURE_COUNT}, got={len(feature_cols)}"
            )

        self.save_cache(
            symbol=symbol,
            raw_df=raw_df,
            engineered_df=engineered_df,
            prepared_df=prepared_df,
            feature_cols=feature_cols,
            include_sentiment=include_sentiment,
            horizons=horizons,
        )
        return raw_df, engineered_df, prepared_df, feature_cols

    def clear_cache(self, symbol: str) -> None:
        base = self._symbol_dir(symbol)
        for p in base.glob("*"):
            if p.is_file():
                p.unlink(missing_ok=True)

    def get_cache_info(self, symbol: str) -> Dict:
        paths = self._paths(symbol)
        if not paths["metadata"].exists():
            return {"symbol": symbol.upper(), "exists": False}

        with paths["metadata"].open("r", encoding="utf-8") as fh:
            metadata = json.load(fh)

        created_at = datetime.fromisoformat(metadata["created_at"])
        age_hours = (datetime.now() - created_at).total_seconds() / 3600.0
        metadata["age_hours"] = round(age_hours, 2)
        metadata["exists"] = True
        return metadata
