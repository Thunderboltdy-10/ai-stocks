from inference.variant_quality import score_variant_quality, select_holdout_candidate


def test_select_holdout_candidate_keeps_one_coherent_source():
    metadata = {
        "holdout": {
            "ensemble": {
                "dir_acc": 0.51,
                "pred_std": 0.002,
                "net_return": -0.20,
                "net_sharpe": -0.10,
                "positive_pct": 0.52,
                "ic": 0.01,
            },
            "ensemble_calibrated": {
                "dir_acc": 0.505,
                "pred_std": 0.012,
                "net_return": 0.24,
                "net_sharpe": 0.42,
                "positive_pct": 0.51,
                "ic": 0.06,
            },
        },
        "walk_forward": {
            "xgb": {
                "aggregate": {
                    "net_sharpe_mean": 0.30,
                    "net_return_mean": 0.12,
                    "dir_acc_mean": 0.52,
                }
            },
            "lgb": None,
        },
    }

    source, metrics = select_holdout_candidate(metadata, intraday=False)

    assert source == "ensemble_calibrated"
    assert metrics["net_return"] == 0.24
    assert metrics["net_sharpe"] == 0.42
    assert metrics["dir_acc"] == 0.505


def test_score_variant_quality_uses_cost_history_before_directional_wfe():
    metadata = {
        "wfe": 135.0,
        "holdout": {
            "ensemble": {
                "dir_acc": 0.56,
                "pred_std": 0.004,
                "net_return": -0.08,
                "net_sharpe": -0.12,
                "positive_pct": 0.54,
                "ic": 0.01,
            }
        },
        "walk_forward": {
            "xgb": {
                "aggregate": {
                    "net_sharpe_mean": 0.22,
                    "net_return_mean": 0.10,
                    "dir_acc_mean": 0.58,
                }
            },
            "lgb": None,
        },
    }

    quality = score_variant_quality(metadata, intraday=False)

    assert quality.metrics["wfe"] == 0.0
    assert not quality.passed


def test_select_holdout_candidate_can_promote_direction_head_source():
    metadata = {
        "holdout": {
            "ensemble": {
                "dir_acc": 0.505,
                "pred_std": 0.002,
                "pred_target_std_ratio": 0.12,
                "net_return": 0.01,
                "net_sharpe": 0.02,
                "positive_pct": 0.50,
                "ic": 0.01,
            },
            "ensemble_directional": {
                "dir_acc": 0.529,
                "pred_std": 0.009,
                "pred_target_std_ratio": 0.36,
                "net_return": 0.14,
                "net_sharpe": 0.38,
                "positive_pct": 0.52,
                "ic": 0.07,
            },
            "ensemble_calibrated": {
                "dir_acc": 0.507,
                "pred_std": 0.004,
                "pred_target_std_ratio": 0.16,
                "net_return": 0.03,
                "net_sharpe": 0.08,
                "positive_pct": 0.51,
                "ic": 0.02,
            },
        },
        "walk_forward": {
            "xgb": {
                "aggregate": {
                    "net_sharpe_mean": 0.19,
                    "net_return_mean": 0.08,
                    "dir_acc_mean": 0.52,
                }
            },
            "lgb": None,
        },
    }

    source, metrics = select_holdout_candidate(metadata, intraday=False)

    assert source == "ensemble_directional"
    assert metrics["dir_acc"] == 0.529
    assert metrics["net_return"] == 0.14
