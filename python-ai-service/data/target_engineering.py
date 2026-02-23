"""Target engineering helpers for the rewritten GBM-first pipeline."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd

from data.feature_engineer import BASE_COLUMNS, get_feature_columns


TARGET_PREFIX = "target_"


def create_forward_returns(
    df: pd.DataFrame,
    horizons: List[int] | None = None,
    price_col: str = "Close",
) -> pd.DataFrame:
    """Create forward log-return targets for the requested horizons."""
    horizons = horizons or [1]
    out = df.copy()
    price = pd.to_numeric(out[price_col], errors="coerce")

    for horizon in horizons:
        out[f"target_{horizon}d"] = np.log(price.shift(-horizon) / price)

    return out


def calculate_smart_target(
    df: pd.DataFrame,
    horizon: int = 1,
    vol_window: int = 20,
    price_col: str = "Close",
) -> pd.DataFrame:
    """Compatibility helper: volatility-normalized one-step forward return."""
    out = create_forward_returns(df.copy(), horizons=[horizon], price_col=price_col)
    target_col = f"target_{horizon}d"
    vol = np.log(out[price_col] / out[price_col].shift(1)).rolling(vol_window, min_periods=vol_window).std()
    out["target_smart"] = (out[target_col] / (vol + 1e-9)).clip(-5.0, 5.0)
    return out


def prevent_lookahead_bias(df: pd.DataFrame, feature_cols: List[str] | None = None) -> pd.DataFrame:
    """Shift features by one step so row t uses information available at t-1."""
    out = df.copy()
    if feature_cols is None:
        feature_cols = [c for c in out.columns if c not in BASE_COLUMNS and not c.startswith(TARGET_PREFIX)]

    out[feature_cols] = out[feature_cols].shift(1)
    return out


def prepare_training_data(
    df: pd.DataFrame,
    horizons: List[int] | None = None,
    shift_features: bool = True,
    include_sentiment: bool = False,
) -> Tuple[pd.DataFrame, List[str]]:
    """Create targets, optionally shift features, and return a clean training frame."""
    horizons = horizons or [1]

    out = create_forward_returns(df.copy(), horizons=horizons)
    feature_cols = get_feature_columns(include_sentiment=include_sentiment)

    if shift_features:
        out = prevent_lookahead_bias(out, feature_cols=feature_cols)

    target_cols = [f"target_{h}d" for h in horizons]
    required_cols = BASE_COLUMNS + feature_cols + target_cols
    present_cols = [c for c in required_cols if c in out.columns]
    out = out[present_cols].replace([np.inf, -np.inf], np.nan)
    out = out.dropna(subset=feature_cols + target_cols)

    return out, feature_cols
