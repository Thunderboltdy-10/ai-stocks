"""Purged time-series splitting utilities (Lopez de Prado style)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generator, Iterable, Tuple

import numpy as np


@dataclass
class SplitConfig:
    n_splits: int = 5
    test_size: int = 252
    embargo_days: int = 5
    purge_days: int = 5
    min_train_size: int = 756


class PurgedTimeSeriesSplit:
    """Time-series splitter with purge and embargo gaps.

    - `purge_days` removes potentially overlapping samples near the test boundary.
    - `embargo_days` inserts a hard gap between train and test segments.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = 252,
        embargo_days: int = 5,
        purge_days: int = 5,
        min_train_size: int = 756,
    ) -> None:
        self.config = SplitConfig(
            n_splits=n_splits,
            test_size=test_size,
            embargo_days=embargo_days,
            purge_days=purge_days,
            min_train_size=min_train_size,
        )

    def get_n_splits(self, X: Iterable | None = None, y: Iterable | None = None) -> int:
        return self.config.n_splits

    def split(self, X: np.ndarray, y: np.ndarray | None = None) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        n_samples = len(X)
        cfg = self.config

        if n_samples <= cfg.min_train_size + cfg.test_size:
            raise ValueError(
                f"Not enough samples ({n_samples}) for split config "
                f"min_train={cfg.min_train_size}, test={cfg.test_size}"
            )

        # Create folds from oldest->newest where each fold tests a contiguous chunk.
        latest_test_start = n_samples - cfg.test_size
        earliest_test_start = cfg.min_train_size + cfg.embargo_days + cfg.purge_days

        if latest_test_start <= earliest_test_start:
            raise ValueError("Split configuration leaves no valid test windows")

        test_starts = np.linspace(
            earliest_test_start,
            latest_test_start,
            num=cfg.n_splits,
            dtype=int,
        )

        for test_start in test_starts:
            test_end = min(test_start + cfg.test_size, n_samples)

            train_end = test_start - cfg.embargo_days
            purge_cutoff = max(0, train_end - cfg.purge_days)

            if purge_cutoff < cfg.min_train_size:
                continue

            train_idx = np.arange(0, purge_cutoff)
            test_idx = np.arange(test_start, test_end)

            if len(test_idx) == 0:
                continue
            yield train_idx, test_idx


def create_train_val_test_split(
    n_samples: int,
    train_pct: float = 0.70,
    val_pct: float = 0.15,
    embargo_days: int = 5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a single chronological split with train/val/test and embargo gaps."""
    if not 0 < train_pct < 1:
        raise ValueError("train_pct must be in (0, 1)")
    if not 0 < val_pct < 1:
        raise ValueError("val_pct must be in (0, 1)")
    if train_pct + val_pct >= 1:
        raise ValueError("train_pct + val_pct must be < 1")

    train_end = int(n_samples * train_pct)
    val_end = int(n_samples * (train_pct + val_pct))

    val_start = min(train_end + embargo_days, val_end)
    test_start = min(val_end + embargo_days, n_samples)

    train_idx = np.arange(0, train_end)
    val_idx = np.arange(val_start, val_end)
    test_idx = np.arange(test_start, n_samples)

    return train_idx, val_idx, test_idx
