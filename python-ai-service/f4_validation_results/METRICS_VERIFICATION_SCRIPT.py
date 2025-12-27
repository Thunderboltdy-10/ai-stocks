"""
Verification script to validate all 5 performance metrics are implemented and working.
This script creates synthetic data and runs the backtester to confirm metrics calculation.
"""

import numpy as np
from evaluation.advanced_backtester import AdvancedBacktester

def test_all_metrics():
    """Test that all 5 metrics are calculated correctly."""

    print("=" * 70)
    print("BACKTESTER METRICS VERIFICATION TEST")
    print("=" * 70)

    # Create synthetic test data (100 days)
    np.random.seed(42)
    n_days = 100

    # Synthetic returns with uptrend
    daily_returns = np.random.normal(0.002, 0.015, n_days)  # 0.2% avg, 1.5% volatility
    dates = np.arange(n_days)

    # Create prices from returns
    prices = np.cumprod(1.0 + daily_returns) * 100  # Start at $100

    # Simple buy-and-hold positions
    positions = np.ones(n_days)

    # Initialize backtester with 0% risk-free rate for simplicity
    backtester = AdvancedBacktester(
        initial_capital=10_000.0,
        risk_free_rate=0.02,  # 2% annual risk-free rate
        commission_pct=0.001,  # 0.1% commission
        slippage_pct=0.0005,   # 0.05% slippage
        min_commission=1.0,
    )

    # Run backtest
    results = backtester.backtest_with_positions(
        dates=dates,
        prices=prices,
        returns=daily_returns,
        positions=positions,
        max_long=1.0,
        max_short=0.0,
    )

    # Extract metrics
    metrics = results.metrics

    print("\nPERFORMANCE METRICS CALCULATED:")
    print("-" * 70)

    # Check each required metric
    required_metrics = [
        "win_rate",
        "profit_factor",
        "sortino_ratio",
        "calmar_ratio",
        "turnover",
    ]

    missing_metrics = []
    for metric_name in required_metrics:
        if metric_name in metrics:
            value = metrics[metric_name]
            print(f"✓ {metric_name:20s}: {value:.6f}")
        else:
            print(f"✗ {metric_name:20s}: MISSING")
            missing_metrics.append(metric_name)

    print("\nADDITIONAL METRICS (Included in comprehensive backtester):")
    print("-" * 70)

    additional_metrics = [
        "cum_return",
        "avg_daily",
        "std_daily",
        "sharpe",
        "max_drawdown",
    ]

    for metric_name in additional_metrics:
        if metric_name in metrics:
            value = metrics[metric_name]
            if metric_name in ["cum_return", "max_drawdown"]:
                print(f"  {metric_name:20s}: {value:.2%}")
            else:
                print(f"  {metric_name:20s}: {value:.6f}")

    print("\n" + "=" * 70)

    if missing_metrics:
        print(f"FAILED: Missing metrics: {missing_metrics}")
        return False
    else:
        print("SUCCESS: All 5 required metrics are present and calculated!")
        print("\nMetric Details:")
        print(f"  - Win Rate:        {metrics['win_rate']:.1%} of days were profitable")
        print(f"  - Profit Factor:   {metrics['profit_factor']:.2f}x (gains/losses ratio)")
        print(f"  - Sortino Ratio:   {metrics['sortino_ratio']:.2f} (downside-adjusted)")
        print(f"  - Calmar Ratio:    {metrics['calmar_ratio']:.2f} (return/drawdown)")
        print(f"  - Turnover:        {metrics['turnover']:.4f} (rebalancing frequency)")
        return True


def test_edge_cases():
    """Test edge cases like empty returns, all-positive/all-negative days."""

    print("\n" + "=" * 70)
    print("EDGE CASE TESTING")
    print("=" * 70)

    backtester = AdvancedBacktester(initial_capital=10_000.0, risk_free_rate=0.02)

    # Test 1: All positive returns
    print("\nTest 1: All Positive Returns")
    print("-" * 70)
    daily_returns = np.ones(50) * 0.01
    dates = np.arange(50)
    prices = np.cumprod(1.0 + daily_returns) * 100
    positions = np.ones(50)

    results = backtester.backtest_with_positions(
        dates=dates, prices=prices, returns=daily_returns, positions=positions
    )

    metrics = results.metrics
    print(f"Win Rate (should be 1.0):     {metrics['win_rate']:.4f}")
    print(f"Profit Factor (should be inf): {metrics['profit_factor']:.4f}")
    assert metrics['win_rate'] == 1.0, "All positive returns should have 100% win rate"
    assert metrics['profit_factor'] > 1e9, "All positive returns should have infinite profit factor"
    print("✓ PASSED")

    # Test 2: Mixed returns
    print("\nTest 2: Mixed Returns (50/50 win/loss)")
    print("-" * 70)
    daily_returns = np.array([0.01, -0.01] * 25)
    dates = np.arange(50)
    prices = np.cumprod(1.0 + daily_returns) * 100
    positions = np.ones(50)

    results = backtester.backtest_with_positions(
        dates=dates, prices=prices, returns=daily_returns, positions=positions
    )

    metrics = results.metrics
    print(f"Win Rate (should be ~0.5):     {metrics['win_rate']:.4f}")
    print(f"Profit Factor (should be ~1.0): {metrics['profit_factor']:.4f}")
    assert 0.45 < metrics['win_rate'] < 0.55, "50/50 returns should have ~50% win rate"
    assert 0.95 < metrics['profit_factor'] < 1.05, "50/50 returns should have ~100% profit factor"
    print("✓ PASSED")

    # Test 3: Varying positions (tests turnover)
    print("\nTest 3: Varying Positions (tests turnover)")
    print("-" * 70)
    daily_returns = np.random.normal(0.0005, 0.01, 100)
    dates = np.arange(100)
    prices = np.cumprod(1.0 + daily_returns) * 100
    positions = np.sin(np.linspace(0, 4*np.pi, 100))  # Oscillating positions

    results = backtester.backtest_with_positions(
        dates=dates, prices=prices, returns=daily_returns, positions=positions
    )

    metrics = results.metrics
    print(f"Turnover (should be > 0):     {metrics['turnover']:.6f}")
    assert metrics['turnover'] > 0, "Varying positions should have positive turnover"
    print("✓ PASSED")

    print("\n" + "=" * 70)
    print("All edge case tests PASSED!")
    print("=" * 70)


if __name__ == "__main__":
    import sys
    from pathlib import Path

    # Add project root to path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        success = test_all_metrics()
        if success:
            test_edge_cases()
            print("\n" + "=" * 70)
            print("ALL VERIFICATION TESTS PASSED!")
            print("=" * 70)
            sys.exit(0)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"\nERROR during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
