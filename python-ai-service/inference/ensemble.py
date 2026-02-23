"""Simple ensemble logic for GBM-first inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class EnsembleWeights:
    gbm: float = 0.7
    lstm: float = 0.3

    def normalized(self) -> "EnsembleWeights":
        total = max(self.gbm + self.lstm, 1e-9)
        return EnsembleWeights(gbm=self.gbm / total, lstm=self.lstm / total)


class SimpleEnsemble:
    def __init__(self, symbol: str, gbm_weight: float = 0.7, lstm_weight: float = 0.3) -> None:
        self.symbol = symbol.upper()
        self.weights = EnsembleWeights(gbm=gbm_weight, lstm=lstm_weight).normalized()

    def update_weights_from_performance(self, recent_metrics: Dict[str, Dict[str, float]]) -> None:
        """Adjust weights based on recent walk-forward Sharpe estimates."""
        gbm_sharpe = float(recent_metrics.get("gbm", {}).get("sharpe", 0.0))
        lstm_sharpe = float(recent_metrics.get("lstm", {}).get("sharpe", 0.0))

        gbm_score = max(gbm_sharpe, 0.0) + 0.05
        lstm_score = max(lstm_sharpe, 0.0) + 0.05
        total = gbm_score + lstm_score

        if total > 0:
            self.weights = EnsembleWeights(gbm=gbm_score / total, lstm=lstm_score / total)

    def fuse(
        self,
        gbm_pred: np.ndarray,
        lstm_pred: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        gbm_arr = np.asarray(gbm_pred, dtype=float).reshape(-1)
        if lstm_pred is None:
            return gbm_arr

        lstm_arr = np.asarray(lstm_pred, dtype=float).reshape(-1)
        if len(lstm_arr) != len(gbm_arr):
            return gbm_arr

        w = self.weights.normalized()
        return w.gbm * gbm_arr + w.lstm * lstm_arr
