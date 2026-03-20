import numpy as np

from service.prediction_service import _split_history_windows


def _prediction_payload(n: int = 180) -> dict:
    dates = [f"2025-01-{(i % 28) + 1:02d}" for i in range(n)]
    prices = np.linspace(100.0, 140.0, n).tolist()
    positions = np.linspace(-0.2, 1.0, n).tolist()
    returns = np.linspace(-0.01, 0.02, n).tolist()
    return {
        "dates": dates,
        "prices": prices,
        "fusedPositions": positions,
        "predictedReturns": returns,
    }


def test_split_history_windows_respects_backtest_and_forward_windows():
    payload = _prediction_payload(180)

    windows = _split_history_windows(
        payload,
        {
            "backtestWindow": 120,
            "forwardWindow": 20,
            "enableForwardSim": True,
        },
    )

    assert windows["totalPoints"] == 180
    assert len(windows["backtest"]["dates"]) == 120
    assert len(windows["forward"]["dates"]) == 20
    assert windows["backtest"]["dates"][0] == payload["dates"][40]
    assert windows["forward"]["dates"][0] == payload["dates"][160]


def test_split_history_windows_uses_requested_backtest_window_without_forward_holdout():
    payload = _prediction_payload(90)

    windows = _split_history_windows(
        payload,
        {
            "backtestWindow": 45,
            "enableForwardSim": False,
        },
    )

    assert len(windows["backtest"]["dates"]) == 45
    assert "forward" not in windows
    assert windows["backtest"]["dates"][0] == payload["dates"][45]
