import numpy as np
import pandas as pd

from data.feature_engineer import (
    EXPECTED_FEATURE_COUNT,
    engineer_features,
    get_feature_columns,
    resolve_feature_columns,
)
from inference.load_gbm_models import blend_with_direction_head
from data.target_engineering import create_forward_returns, prevent_lookahead_bias
from training.train_gbm import _coerce_shap_values
from validation.walk_forward import evaluate_predictions


def _sample_ohlcv(rows: int = 320) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    idx = pd.date_range("2020-01-01", periods=rows, freq="B")
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0004, 0.018, size=rows)))
    open_ = close * (1.0 + rng.normal(0.0, 0.003, size=rows))
    high = np.maximum(open_, close) * (1.0 + rng.uniform(0.0005, 0.01, size=rows))
    low = np.minimum(open_, close) * (1.0 - rng.uniform(0.0005, 0.01, size=rows))
    volume = rng.integers(1_000_000, 6_000_000, size=rows)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


def test_engineer_features_matches_canonical_feature_list():
    df = _sample_ohlcv()
    engineered = engineer_features(df, symbol="TEST", include_sentiment=False)
    feature_cols = [c for c in engineered.columns if c not in {"Open", "High", "Low", "Close", "Volume"}]
    assert len(feature_cols) == EXPECTED_FEATURE_COUNT
    assert feature_cols == get_feature_columns(include_sentiment=False)


def test_prevent_lookahead_bias_shifts_features_without_touching_targets():
    df = _sample_ohlcv()
    engineered = engineer_features(df, symbol="TEST", include_sentiment=False)
    with_targets = create_forward_returns(engineered, horizons=[1])
    feature_cols = get_feature_columns(include_sentiment=False)

    shifted = prevent_lookahead_bias(with_targets, feature_cols=feature_cols)

    probe_col = "ret_1d"
    assert np.isnan(shifted.iloc[0][probe_col])
    assert shifted.iloc[10][probe_col] == with_targets.iloc[9][probe_col]
    assert shifted.iloc[10]["target_1d"] == with_targets.iloc[10]["target_1d"]


def test_feature_profiles_are_ordered_subsets_of_canonical_features():
    canonical = get_feature_columns(include_sentiment=False)
    compact = resolve_feature_columns("compact")
    trend = resolve_feature_columns("trend")

    assert compact
    assert trend
    assert all(col in canonical for col in compact)
    assert all(col in canonical for col in trend)
    assert compact == [col for col in canonical if col in set(compact)]
    assert trend == [col for col in canonical if col in set(trend)]
    assert len(compact) < len(canonical)
    assert len(trend) < len(canonical)


def test_evaluate_predictions_exposes_dispersion_ratio():
    y_true = np.array([0.02, -0.01, 0.015, -0.03, 0.01], dtype=float)
    y_pred = np.array([0.01, -0.005, 0.007, -0.012, 0.006], dtype=float)

    metrics = evaluate_predictions(y_true, y_pred)

    assert "pred_target_std_ratio" in metrics
    assert metrics["pred_target_std_ratio"] > 0.0
    assert metrics["pred_target_std_ratio"] < 1.0


def test_direction_head_blend_penalizes_confident_disagreement():
    base = np.array([0.02, -0.015, 0.01], dtype=float)
    direction_prob = np.array([0.95, 0.10, 0.05], dtype=float)
    metadata = {
        "direction_head": {
            "enabled": True,
            "alignment_floor": 0.18,
            "confidence_boost": 0.85,
            "disagreement_penalty": 1.55,
        }
    }

    blended = blend_with_direction_head(base, direction_prob, metadata)

    assert blended[0] > 0.0
    assert abs(blended[0]) >= abs(base[0]) * 0.8
    assert abs(blended[1]) >= abs(base[1]) * 0.8
    assert blended[2] >= 0.0
    assert abs(blended[2]) < abs(base[2]) * 0.3


def test_coerce_shap_values_handles_bracketed_string_scalars():
    raw = np.array([["[7.2101154E-4]", "-0.0025"], ["0.0012", "[3.4E-4]"]], dtype=object)

    coerced = _coerce_shap_values(raw)

    assert coerced.shape == (2, 2)
    assert np.isfinite(coerced).all()
    assert abs(coerced[0, 0] - 7.2101154e-4) < 1e-12
