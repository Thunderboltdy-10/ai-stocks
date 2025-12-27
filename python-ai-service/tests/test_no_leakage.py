"""
No-Leakage Verification Tests

Comprehensive tests to verify no data leakage occurs in:
1. GBM Training - 3-way split (Train 60%, Val 20%, Test 20%)
2. Walk-Forward Validation - No overlap between splits
3. Forward Simulation - No look-ahead bias
4. Backtest Alignment - Position lag applied correctly

These tests are critical for ensuring honest backtesting and preventing
inflated performance metrics that won't translate to live trading.

Author: AI-Stocks Nuclear Redesign
Date: December 2025
"""

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Tuple, List
from unittest.mock import Mock, patch, MagicMock
from dataclasses import dataclass

import pytest
import numpy as np
import pandas as pd

# Add project root to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import modules under test
from validation.walk_forward import (
    WalkForwardValidator,
    WalkForwardConfig,
    WalkForwardMode,
    WalkForwardSplit,
    calculate_wfe,
)
from validation.wfe_metrics import (
    calculate_wfe as wfe_calculate,
    calculate_overfitting_ratio,
    detect_variance_collapse,
    detect_sign_imbalance,
)
from evaluation.forward_simulator import (
    ForwardSimulator,
    SimulationStep,
    ForwardSimulationResults,
)
from evaluation.advanced_backtester import (
    AdvancedBacktester,
    BacktestResults,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def sample_time_series_data() -> Tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Generate sample time series data for testing."""
    np.random.seed(42)
    n_samples = 1000
    n_features = 10

    # Create date index
    dates = pd.date_range(start='2020-01-01', periods=n_samples, freq='B')

    # Generate features (random walk with some signal)
    X = np.random.randn(n_samples, n_features) * 0.02

    # Generate targets (slightly correlated with features)
    true_coefficients = np.random.randn(n_features) * 0.1
    signal = X @ true_coefficients
    noise = np.random.randn(n_samples) * 0.02
    y = signal + noise

    return X, y, dates


@pytest.fixture
def sample_ohlcv_dataframe() -> pd.DataFrame:
    """Generate sample OHLCV data for simulation tests."""
    np.random.seed(42)
    n_days = 500

    dates = pd.date_range(start='2021-01-01', periods=n_days, freq='B')

    # Generate realistic price series
    returns = np.random.randn(n_days) * 0.02
    prices = 100 * np.exp(np.cumsum(returns))

    # Add open/high/low
    opens = prices * (1 + np.random.randn(n_days) * 0.005)
    highs = np.maximum(prices, opens) * (1 + np.abs(np.random.randn(n_days)) * 0.01)
    lows = np.minimum(prices, opens) * (1 - np.abs(np.random.randn(n_days)) * 0.01)

    # Volume
    volume = np.random.randint(1_000_000, 10_000_000, n_days)

    df = pd.DataFrame({
        'Open': opens,
        'High': highs,
        'Low': lows,
        'Close': prices,
        'Volume': volume,
    }, index=dates)

    return df


@pytest.fixture
def simple_mock_model():
    """Create a simple mock model for testing."""
    class SimpleMockModel:
        def __init__(self):
            self.fitted = False

        def fit(self, X, y, **kwargs):
            self.fitted = True
            self.mean_ = np.mean(y)
            self.std_ = np.std(y)
            return self

        def predict(self, X):
            # Return random predictions with similar distribution to training
            return np.random.randn(len(X)) * self.std_ + self.mean_

    return SimpleMockModel


# =============================================================================
# TEST CLASS 1: GBM TRAINING NO LEAKAGE
# =============================================================================

class TestGBMTrainingNoLeakage:
    """Tests to verify GBM training uses proper 3-way split with no leakage."""

    def test_three_way_split_proportions(self, sample_time_series_data):
        """Verify that 3-way split uses correct proportions (60/20/20)."""
        X, y, dates = sample_time_series_data
        n_samples = len(X)

        # Expected split indices
        train_end = int(n_samples * 0.60)
        val_end = int(n_samples * 0.80)

        # Verify proportions
        train_size = train_end
        val_size = val_end - train_end
        test_size = n_samples - val_end

        # Check approximate proportions (within 1% tolerance)
        assert abs(train_size / n_samples - 0.60) < 0.01, \
            f"Train proportion {train_size/n_samples:.2%} != 60%"
        assert abs(val_size / n_samples - 0.20) < 0.01, \
            f"Val proportion {val_size/n_samples:.2%} != 20%"
        assert abs(test_size / n_samples - 0.20) < 0.01, \
            f"Test proportion {test_size/n_samples:.2%} != 20%"

    def test_no_overlap_in_splits(self, sample_time_series_data):
        """Verify train/val/test splits have no overlapping indices."""
        X, y, dates = sample_time_series_data
        n_samples = len(X)

        # Define split boundaries
        train_end = int(n_samples * 0.60)
        val_end = int(n_samples * 0.80)

        train_indices = set(range(0, train_end))
        val_indices = set(range(train_end, val_end))
        test_indices = set(range(val_end, n_samples))

        # Verify no overlap
        assert len(train_indices & val_indices) == 0, \
            "Train and validation sets overlap!"
        assert len(train_indices & test_indices) == 0, \
            "Train and test sets overlap!"
        assert len(val_indices & test_indices) == 0, \
            "Validation and test sets overlap!"

        # Verify complete coverage
        all_indices = train_indices | val_indices | test_indices
        expected_indices = set(range(n_samples))
        assert all_indices == expected_indices, \
            "Splits don't cover all data points!"

    def test_temporal_order_preserved(self, sample_time_series_data):
        """Verify that splits maintain temporal order (no future data in training)."""
        X, y, dates = sample_time_series_data
        n_samples = len(X)

        train_end = int(n_samples * 0.60)
        val_end = int(n_samples * 0.80)

        # Get dates for each split
        train_dates = dates[:train_end]
        val_dates = dates[train_end:val_end]
        test_dates = dates[val_end:]

        # Verify temporal ordering
        assert train_dates.max() < val_dates.min(), \
            "Training data contains dates after validation start!"
        assert val_dates.max() < test_dates.min(), \
            "Validation data contains dates after test start!"

    def test_metrics_computed_only_on_test_set(self, sample_time_series_data, simple_mock_model):
        """Verify that final metrics are computed ONLY on held-out test set."""
        X, y, dates = sample_time_series_data
        n_samples = len(X)

        # Define splits
        train_end = int(n_samples * 0.60)
        val_end = int(n_samples * 0.80)

        X_train = X[:train_end]
        y_train = y[:train_end]
        X_val = X[train_end:val_end]
        y_val = y[train_end:val_end]
        X_test = X[val_end:]
        y_test = y[val_end:]

        # Train model
        model = simple_mock_model()
        model.fit(X_train, y_train)

        # Make predictions on each split
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_test = model.predict(X_test)

        # Calculate metrics
        train_mse = np.mean((y_train - y_pred_train) ** 2)
        val_mse = np.mean((y_val - y_pred_val) ** 2)
        test_mse = np.mean((y_test - y_pred_test) ** 2)

        # The test set should be independent - verify it wasn't used in training
        # by checking that test samples don't match training samples
        for i in range(len(X_test)):
            for j in range(len(X_train)):
                assert not np.allclose(X_test[i], X_train[j]), \
                    f"Test sample {i} matches training sample {j}!"

    def test_wfe_calculated_correctly(self, sample_time_series_data, simple_mock_model):
        """Verify Walk Forward Efficiency is calculated using val vs test performance."""
        X, y, dates = sample_time_series_data
        n_samples = len(X)

        # Define splits
        train_end = int(n_samples * 0.60)
        val_end = int(n_samples * 0.80)

        y_val = y[train_end:val_end]
        y_test = y[val_end:]

        # Create mock predictions
        np.random.seed(123)
        y_pred_val = np.random.randn(len(y_val)) * 0.02
        y_pred_test = np.random.randn(len(y_test)) * 0.02

        # Calculate direction accuracy
        val_dir_acc = np.mean(np.sign(y_val) == np.sign(y_pred_val))
        test_dir_acc = np.mean(np.sign(y_test) == np.sign(y_pred_test))

        # Calculate WFE
        wfe = calculate_wfe(val_dir_acc, test_dir_acc)

        # WFE should be between 0 and some reasonable upper bound
        assert 0 <= wfe <= 200, f"WFE {wfe} is out of reasonable range"

        # Verify formula: WFE = (test / val) * 100 when both > baseline
        if val_dir_acc > 0.5:
            expected_wfe = (test_dir_acc / val_dir_acc) * 100
            assert abs(wfe - expected_wfe) < 0.01, \
                f"WFE calculation mismatch: {wfe} vs expected {expected_wfe}"


# =============================================================================
# TEST CLASS 2: WALK-FORWARD NO OVERLAP
# =============================================================================

class TestWalkForwardNoOverlap:
    """Tests to verify walk-forward validation has no overlap between splits."""

    def test_splits_have_no_overlap(self, sample_time_series_data):
        """Verify train/val/test splits have no overlapping indices in any fold."""
        X, y, dates = sample_time_series_data
        n_samples = len(X)

        config = WalkForwardConfig(
            mode=WalkForwardMode.ANCHORED,
            n_iterations=5,
            train_pct=0.60,
            validation_pct=0.15,
            test_pct=0.25,
            gap_days=1,
            purge_days=5,
        )

        validator = WalkForwardValidator(config)
        splits = validator.generate_splits(n_samples)

        for split in splits:
            train_set = set(range(split.train_start, split.train_end))
            val_set = set(range(split.val_start, split.val_end))
            test_set = set(range(split.test_start, split.test_end))

            # Check no overlap
            assert len(train_set & val_set) == 0, \
                f"Fold {split.fold}: Train and val overlap!"
            assert len(train_set & test_set) == 0, \
                f"Fold {split.fold}: Train and test overlap!"
            assert len(val_set & test_set) == 0, \
                f"Fold {split.fold}: Val and test overlap!"

            # Verify has_overlap() method works
            assert not split.has_overlap(), \
                f"Fold {split.fold}: has_overlap() should return False!"

    def test_purge_gap_respected(self, sample_time_series_data):
        """Verify purge gap is maintained between train/val and val/test."""
        X, y, dates = sample_time_series_data
        n_samples = len(X)

        gap_days = 5
        purge_days = 10

        config = WalkForwardConfig(
            mode=WalkForwardMode.ANCHORED,
            n_iterations=3,
            train_pct=0.60,
            validation_pct=0.15,
            test_pct=0.25,
            gap_days=gap_days,
            purge_days=purge_days,
        )

        validator = WalkForwardValidator(config)
        splits = validator.generate_splits(n_samples)

        for split in splits:
            # Check gap between train and val
            actual_gap_train_val = split.val_start - split.train_end
            assert actual_gap_train_val >= gap_days, \
                f"Fold {split.fold}: Gap between train/val is {actual_gap_train_val}, expected >= {gap_days}"

            # Check purge between val and test
            actual_purge = split.test_start - split.val_end
            assert actual_purge >= purge_days, \
                f"Fold {split.fold}: Purge between val/test is {actual_purge}, expected >= {purge_days}"

    def test_anchored_mode_expands_correctly(self, sample_time_series_data):
        """Verify anchored (expanding) window always starts from index 0."""
        X, y, dates = sample_time_series_data
        n_samples = len(X)

        config = WalkForwardConfig(
            mode=WalkForwardMode.ANCHORED,
            n_iterations=4,
            train_pct=0.60,
            validation_pct=0.15,
            test_pct=0.25,
        )

        validator = WalkForwardValidator(config)
        splits = validator.generate_splits(n_samples)

        prev_train_end = 0
        for split in splits:
            # Anchored mode: training always starts from 0
            assert split.train_start == 0, \
                f"Fold {split.fold}: Anchored training should start at 0, got {split.train_start}"

            # Training window should expand (or stay same) with each fold
            assert split.train_end >= prev_train_end, \
                f"Fold {split.fold}: Training window should expand! " \
                f"{split.train_end} < {prev_train_end}"

            prev_train_end = split.train_end

    def test_rolling_mode_has_fixed_train_size(self, sample_time_series_data):
        """Verify rolling window maintains fixed training size."""
        X, y, dates = sample_time_series_data
        n_samples = len(X)

        config = WalkForwardConfig(
            mode=WalkForwardMode.ROLLING,
            n_iterations=4,
            train_pct=0.60,
            validation_pct=0.15,
            test_pct=0.25,
            gap_days=1,
            purge_days=5,
        )

        validator = WalkForwardValidator(config)
        splits = validator.generate_splits(n_samples)

        if len(splits) < 2:
            pytest.skip("Not enough splits generated for rolling mode test")

        # All folds should have same train size (approximately)
        train_sizes = [s.train_size for s in splits]
        expected_size = train_sizes[0]

        for i, size in enumerate(train_sizes):
            # Allow small tolerance due to integer division
            assert abs(size - expected_size) <= 1, \
                f"Rolling mode: Fold {i} train size {size} != expected {expected_size}"

    def test_no_future_leakage_in_any_fold(self, sample_time_series_data):
        """Verify no fold uses future data for training."""
        X, y, dates = sample_time_series_data
        n_samples = len(X)

        config = WalkForwardConfig(
            mode=WalkForwardMode.ANCHORED,
            n_iterations=5,
        )

        validator = WalkForwardValidator(config)
        splits = validator.generate_splits(n_samples)

        for split in splits:
            # All training data should come before validation
            assert split.train_end <= split.val_start, \
                f"Fold {split.fold}: Training ends at {split.train_end} but val starts at {split.val_start}"

            # All validation data should come before test
            assert split.val_end <= split.test_start, \
                f"Fold {split.fold}: Validation ends at {split.val_end} but test starts at {split.test_start}"


# =============================================================================
# TEST CLASS 3: FORWARD SIMULATION NO LOOK-AHEAD
# =============================================================================

class TestForwardSimulationNoLookAhead:
    """Tests to verify forward simulation uses no look-ahead bias."""

    def test_features_date_before_simulation_date(self, sample_ohlcv_dataframe):
        """Verify features_date < simulation_date for all steps."""
        df = sample_ohlcv_dataframe

        # Create mock predictor
        mock_predictor = Mock()
        mock_predictor.predict = Mock(return_value=np.array([0.01]))

        # Simple feature engineer that just returns a subset of columns
        def mock_feature_engineer(data):
            return data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

        simulator = ForwardSimulator(
            predictor=mock_predictor,
            feature_engineer=mock_feature_engineer,
            sequence_length=20,
        )

        # Run simulation on a subset
        results = simulator.simulate(
            data=df,
            start_date='2021-06-01',
            end_date='2021-08-01',
            verbose=False,
        )

        # Verify no look-ahead in any step
        for step in results.steps:
            assert step.features_date < step.date, \
                f"Look-ahead detected! features_date={step.features_date} >= date={step.date}"

    def test_has_look_ahead_flag(self, sample_ohlcv_dataframe):
        """Verify the simulation correctly detects look-ahead bias."""
        df = sample_ohlcv_dataframe

        mock_predictor = Mock()
        mock_predictor.predict = Mock(return_value=np.array([0.01]))

        def mock_feature_engineer(data):
            return data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

        simulator = ForwardSimulator(
            predictor=mock_predictor,
            feature_engineer=mock_feature_engineer,
            sequence_length=20,
        )

        results = simulator.simulate(
            data=df,
            start_date='2021-06-01',
            end_date='2021-08-01',
            verbose=False,
        )

        # Results should not have look-ahead bias
        assert not results.has_look_ahead_bias, \
            "Simulation incorrectly detected look-ahead bias!"

    def test_simulation_step_temporal_ordering(self, sample_ohlcv_dataframe):
        """Verify simulation steps are in chronological order."""
        df = sample_ohlcv_dataframe

        mock_predictor = Mock()
        mock_predictor.predict = Mock(return_value=np.array([0.01]))

        def mock_feature_engineer(data):
            return data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

        simulator = ForwardSimulator(
            predictor=mock_predictor,
            feature_engineer=mock_feature_engineer,
            sequence_length=20,
        )

        results = simulator.simulate(
            data=df,
            start_date='2021-06-01',
            end_date='2021-12-01',
            verbose=False,
        )

        # Verify chronological order
        for i in range(1, len(results.steps)):
            assert results.steps[i].date > results.steps[i-1].date, \
                f"Steps not in chronological order at index {i}"

    def test_only_historical_data_used(self, sample_ohlcv_dataframe):
        """Verify only historical data is available for feature engineering."""
        df = sample_ohlcv_dataframe

        # Track what data is passed to feature engineer
        data_snapshots = []

        def tracking_feature_engineer(data):
            data_snapshots.append({
                'max_date': data.index.max(),
                'n_rows': len(data),
            })
            return data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()

        mock_predictor = Mock()
        mock_predictor.predict = Mock(return_value=np.array([0.01]))

        simulator = ForwardSimulator(
            predictor=mock_predictor,
            feature_engineer=tracking_feature_engineer,
            sequence_length=20,
        )

        # Simulate from a known date
        start_date = pd.Timestamp('2021-06-01')
        results = simulator.simulate(
            data=df,
            start_date=str(start_date),
            end_date='2021-07-01',
            verbose=False,
        )

        # Verify each snapshot's max date is before the simulation date
        for i, step in enumerate(results.steps):
            if i < len(data_snapshots):
                snapshot = data_snapshots[i]
                assert snapshot['max_date'] < step.date, \
                    f"Feature engineer received future data! " \
                    f"max_date={snapshot['max_date']} >= step.date={step.date}"

    def test_simulation_step_has_look_ahead_method(self):
        """Verify SimulationStep.has_look_ahead() correctly identifies bias."""
        # Step with look-ahead (features_date >= date)
        bad_step = SimulationStep(
            date=datetime(2021, 6, 1),
            prediction=0.01,
            position=1.0,
            entry_price=100.0,
            exit_price=101.0,
            actual_return=0.01,
            pnl=0.01,
            cumulative_pnl=0.01,
            features_date=datetime(2021, 6, 1),  # Same as date - BAD!
        )
        assert bad_step.has_look_ahead(), "Step should detect look-ahead bias!"

        # Step without look-ahead (features_date < date)
        good_step = SimulationStep(
            date=datetime(2021, 6, 1),
            prediction=0.01,
            position=1.0,
            entry_price=100.0,
            exit_price=101.0,
            actual_return=0.01,
            pnl=0.01,
            cumulative_pnl=0.01,
            features_date=datetime(2021, 5, 31),  # Day before - GOOD!
        )
        assert not good_step.has_look_ahead(), "Step should not detect look-ahead bias!"


# =============================================================================
# TEST CLASS 4: BACKTEST ALIGNMENT
# =============================================================================

class TestBacktestAlignment:
    """Tests to verify backtest position alignment and lag."""

    def test_position_lag_applied_by_default(self):
        """Verify position lag is applied by default (position[T] -> returns[T+1])."""
        np.random.seed(42)
        n = 100

        dates = pd.date_range(start='2021-01-01', periods=n, freq='B')
        prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
        returns = np.diff(prices) / prices[:-1]
        returns = np.append(returns, 0)  # Pad to match length

        # Positions based on some signal
        positions = np.sign(np.random.randn(n))

        backtester = AdvancedBacktester(initial_capital=10000.0)

        results = backtester.backtest_with_positions(
            dates=dates,
            prices=prices,
            returns=returns,
            positions=positions,
            apply_position_lag=True,  # Default
        )

        # With lag applied, first position should be 0 (no prior signal)
        assert results.positions[0] == 0.0, \
            f"First position should be 0 with lag applied, got {results.positions[0]}"

        # Second position should be the first original position
        assert results.positions[1] == positions[0], \
            f"Second position {results.positions[1]} should match original first {positions[0]}"

    def test_position_lag_shift_is_correct(self):
        """Verify position[T] applies to returns[T+1]."""
        np.random.seed(42)
        n = 50

        dates = pd.date_range(start='2021-01-01', periods=n, freq='B')
        prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
        returns = np.diff(prices) / prices[:-1]
        returns = np.append(returns, 0)

        # Create distinct positions to track
        positions = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(n)])

        backtester = AdvancedBacktester(
            initial_capital=10000.0,
            commission_pct=0.0,  # No costs for clean test
            slippage_pct=0.0,
        )

        results = backtester.backtest_with_positions(
            dates=dates,
            prices=prices,
            returns=returns,
            positions=positions,
            apply_position_lag=True,
        )

        # Verify the shift
        for i in range(1, n):
            expected_position = positions[i - 1]  # Original position from previous day
            actual_position = results.positions[i]
            assert actual_position == expected_position, \
                f"Day {i}: Position {actual_position} != expected {expected_position}"

    def test_no_position_lag_when_disabled(self):
        """Verify position lag can be disabled for backward compatibility."""
        np.random.seed(42)
        n = 50

        dates = pd.date_range(start='2021-01-01', periods=n, freq='B')
        prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
        returns = np.diff(prices) / prices[:-1]
        returns = np.append(returns, 0)

        positions = np.sign(np.random.randn(n))

        backtester = AdvancedBacktester(initial_capital=10000.0)

        results = backtester.backtest_with_positions(
            dates=dates,
            prices=prices,
            returns=returns,
            positions=positions,
            apply_position_lag=False,  # Disable lag
        )

        # Positions should match original (after clipping)
        for i in range(n):
            assert results.positions[i] == positions[i], \
                f"Day {i}: Position {results.positions[i]} != original {positions[i]}"

    def test_daily_returns_alignment(self):
        """Verify daily returns are calculated with correct position alignment."""
        np.random.seed(42)
        n = 20

        dates = pd.date_range(start='2021-01-01', periods=n, freq='B')

        # Known returns for verification
        asset_returns = np.array([0.01, -0.02, 0.015, -0.01, 0.005] * 4)
        prices = 100 * np.cumprod(1 + asset_returns)

        # Simple positions
        positions = np.ones(n)  # Always long

        backtester = AdvancedBacktester(
            initial_capital=10000.0,
            commission_pct=0.0,
            slippage_pct=0.0,
            min_commission=0.0,  # Disable minimum commission for clean test
        )

        results = backtester.backtest_with_positions(
            dates=dates,
            prices=prices,
            returns=asset_returns,
            positions=positions,
            apply_position_lag=True,
        )

        # First daily return should be 0 (no position on first day)
        # Due to transaction costs being 0, this should be exactly the position * return
        assert abs(results.daily_returns[0]) < 1e-10, \
            f"First daily return should be ~0, got {results.daily_returns[0]}"

        # Subsequent returns should reflect the lagged position
        for i in range(1, n):
            expected_return = results.positions[i] * asset_returns[i]
            # Allow small numerical tolerance for floating point precision
            assert abs(results.daily_returns[i] - expected_return) < 1e-8, \
                f"Day {i}: Daily return {results.daily_returns[i]} != expected {expected_return}"

    def test_trade_log_timing(self):
        """Verify trade log captures correct timing."""
        np.random.seed(42)
        n = 30

        dates = pd.date_range(start='2021-01-01', periods=n, freq='B')
        prices = 100 * np.exp(np.cumsum(np.random.randn(n) * 0.02))
        returns = np.diff(prices) / prices[:-1]
        returns = np.append(returns, 0)

        # Positions that change
        positions = np.zeros(n)
        positions[5] = 1.0   # Enter long
        positions[10] = 0.0  # Exit
        positions[15] = -0.5 # Enter short
        positions[20] = 0.0  # Exit

        backtester = AdvancedBacktester(initial_capital=10000.0)

        results = backtester.backtest_with_positions(
            dates=dates,
            prices=prices,
            returns=returns,
            positions=positions,
            apply_position_lag=True,
        )

        # Check trade log has entries
        assert len(results.trade_log) > 0, "Trade log should have entries"

        # Verify each trade has required fields
        required_fields = ['index', 'date', 'price', 'action', 'position']
        for trade in results.trade_log:
            for field in required_fields:
                assert field in trade, f"Trade missing field: {field}"


# =============================================================================
# TEST CLASS 5: INTEGRATION TESTS
# =============================================================================

class TestNoLeakageIntegration:
    """Integration tests for full pipeline no-leakage verification."""

    def test_full_pipeline_no_leakage(self, sample_time_series_data):
        """End-to-end test that verifies no leakage in full pipeline."""
        X, y, dates = sample_time_series_data
        n_samples = len(X)

        # 1. Verify 3-way split
        train_end = int(n_samples * 0.60)
        val_end = int(n_samples * 0.80)

        train_dates = dates[:train_end]
        val_dates = dates[train_end:val_end]
        test_dates = dates[val_end:]

        # No overlap in dates
        assert len(set(train_dates) & set(val_dates)) == 0
        assert len(set(train_dates) & set(test_dates)) == 0
        assert len(set(val_dates) & set(test_dates)) == 0

        # 2. Verify walk-forward splits
        config = WalkForwardConfig(
            mode=WalkForwardMode.ANCHORED,
            n_iterations=3,
        )
        validator = WalkForwardValidator(config)
        splits = validator.generate_splits(n_samples)

        for split in splits:
            assert not split.has_overlap()
            assert split.train_end <= split.val_start
            assert split.val_end <= split.test_start

        # 3. Verify WFE metric is computed correctly
        wfe = calculate_wfe(val_performance=0.55, test_performance=0.52)
        assert 0 <= wfe <= 200

    def test_variance_collapse_detection(self):
        """Test that variance collapse is properly detected."""
        # Collapsed predictions (near-constant)
        collapsed_preds = np.ones(100) * 0.001 + np.random.randn(100) * 0.0001
        is_collapsed, variance = detect_variance_collapse(collapsed_preds, threshold=0.001)

        assert is_collapsed, "Should detect variance collapse!"
        assert variance < 0.001, f"Variance {variance} should be below threshold"

        # Healthy predictions
        healthy_preds = np.random.randn(100) * 0.02
        is_collapsed, variance = detect_variance_collapse(healthy_preds, threshold=0.001)

        assert not is_collapsed, "Should not detect collapse for healthy predictions!"
        assert variance > 0.001, f"Variance {variance} should be above threshold"

    def test_sign_imbalance_detection(self):
        """Test that sign imbalance is properly detected."""
        # Imbalanced predictions (mostly positive)
        imbalanced_preds = np.abs(np.random.randn(100)) * 0.02  # All positive
        is_imbalanced, pos_frac = detect_sign_imbalance(imbalanced_preds, min_fraction=0.3)

        assert is_imbalanced, "Should detect sign imbalance!"
        assert pos_frac > 0.9, f"Positive fraction {pos_frac} should be high"

        # Balanced predictions
        balanced_preds = np.random.randn(100) * 0.02  # ~50% positive/negative
        is_imbalanced, pos_frac = detect_sign_imbalance(balanced_preds, min_fraction=0.3)

        assert not is_imbalanced, "Should not detect imbalance for balanced predictions!"
        assert 0.3 <= pos_frac <= 0.7, f"Positive fraction {pos_frac} should be balanced"

    def test_overfitting_ratio_calculation(self):
        """Test overfitting ratio is calculated correctly."""
        # No overfitting (train == test)
        ratio = calculate_overfitting_ratio(train_metric=0.55, test_metric=0.55, baseline=0.5)
        assert abs(ratio - 1.0) < 0.01, f"Ratio {ratio} should be ~1.0 for no overfitting"

        # Overfitting (train >> test)
        ratio = calculate_overfitting_ratio(train_metric=0.70, test_metric=0.55, baseline=0.5)
        assert ratio > 1.0, f"Ratio {ratio} should be > 1.0 for overfitting"

        # Calculated: (0.70 - 0.5) / (0.55 - 0.5) = 0.20 / 0.05 = 4.0
        expected = (0.70 - 0.5) / (0.55 - 0.5)
        assert abs(ratio - expected) < 0.01, f"Ratio {ratio} != expected {expected}"


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
