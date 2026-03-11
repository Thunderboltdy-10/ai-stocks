import numpy as np

from data.feature_engineer import feature_family, summarize_feature_families
from validation.walk_forward import evaluate_predictions


def test_feature_family_summary_groups_expected_features():
    selected = ["rsi_14", "macd", "vol_20d", "obv", "trend_dir", "hour_sin"]
    summary = summarize_feature_families(selected)

    assert feature_family("rsi_14") == "momentum"
    assert feature_family("macd") == "trend"
    assert summary["momentum"] == 1
    assert summary["trend"] == 1
    assert summary["volatility"] == 1
    assert summary["volume"] == 1
    assert summary["regime"] == 1
    assert summary["calendar"] == 1


def test_evaluate_predictions_dailyizes_multi_day_targets():
    y_true = np.array([0.05, -0.04, 0.03, -0.02], dtype=float)
    y_pred = np.array([0.04, -0.03, 0.02, -0.01], dtype=float)

    raw = evaluate_predictions(y_true, y_pred, target_horizon_days=1)
    dailyized = evaluate_predictions(y_true, y_pred, target_horizon_days=5)

    assert dailyized["dir_acc"] == raw["dir_acc"]
    assert dailyized["pred_std"] < raw["pred_std"]
    assert dailyized["target_std"] < raw["target_std"]
    assert dailyized["pred_target_std_ratio"] > 0.0
