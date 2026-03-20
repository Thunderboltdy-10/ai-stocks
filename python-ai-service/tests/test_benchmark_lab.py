import pandas as pd

from research.benchmark_lab import list_benchmark_runs, summarize_mode, write_benchmark_run


def test_summarize_mode_aggregates_symbol_leaders():
    detail_df = pd.DataFrame(
        [
            {"symbol": "AAPL", "variant": "gbm_a", "window": "5y", "alpha": 0.10, "sharpe_ratio": 1.4, "strategy_return": 0.35, "buy_hold_return": 0.20, "ml_incremental_alpha_vs_regime": 0.03, "model_quality_gate_passed": True, "model_quality_score": 1.2},
            {"symbol": "AAPL", "variant": "gbm_a", "window": "1y", "alpha": 0.05, "sharpe_ratio": 1.1, "strategy_return": 0.18, "buy_hold_return": 0.12, "ml_incremental_alpha_vs_regime": 0.02, "model_quality_gate_passed": True, "model_quality_score": 1.1},
            {"symbol": "XOM", "variant": "gbm_b", "window": "5y", "alpha": -0.03, "sharpe_ratio": 0.4, "strategy_return": 0.08, "buy_hold_return": 0.11, "ml_incremental_alpha_vs_regime": -0.01, "model_quality_gate_passed": False, "model_quality_score": -0.4},
        ]
    )
    train_df = pd.DataFrame(
        [
            {"symbol": "AAPL", "variant": "gbm_a", "target_horizon_days": 1, "quality_gate_passed": True, "quality_score": 1.2, "holdout_net_return": 0.14, "holdout_net_sharpe": 0.55, "wfe": 90.0},
            {"symbol": "XOM", "variant": "gbm_b", "target_horizon_days": 1, "quality_gate_passed": False, "quality_score": -0.4, "holdout_net_return": -0.05, "holdout_net_sharpe": -0.10, "wfe": 0.0},
        ]
    )

    summary = summarize_mode(
        mode="daily",
        detail_df=detail_df,
        train_df=train_df,
        symbol_set="daily15",
        settings={"nTrials": 8},
    )

    assert summary["symbolsTested"] == 2
    assert summary["aggregate"]["positiveAlphaRate"] > 0.5
    assert summary["leaders"][0]["symbol"] == "AAPL"
    assert summary["qualityLeaders"][0]["variant"] == "gbm_a"


def test_write_benchmark_run_lists_latest_first(tmp_path):
    run_a = write_benchmark_run(
        modes={"daily": {"mode": "daily", "leaders": [], "laggards": [], "qualityLeaders": [], "aggregate": {}, "settings": {}, "symbolSet": "daily15", "symbolsTested": 1, "variantsEvaluated": 1, "windowsEvaluated": 1}},
        settings={"modes": ["daily"]},
        output_dir=tmp_path,
    )
    run_b = write_benchmark_run(
        modes={"intraday": {"mode": "intraday", "leaders": [], "laggards": [], "qualityLeaders": [], "aggregate": {}, "settings": {}, "symbolSet": "intraday10", "symbolsTested": 1, "variantsEvaluated": 1, "windowsEvaluated": 1}},
        settings={"modes": ["intraday"]},
        output_dir=tmp_path,
    )

    runs = list_benchmark_runs(limit=2, output_dir=tmp_path)

    assert runs[0]["id"] == run_b["id"]
    assert runs[1]["id"] == run_a["id"]
