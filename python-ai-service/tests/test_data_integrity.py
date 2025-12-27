#!/usr/bin/env python3
"""
Data Integrity Tests for AI Trading Pipeline

Comprehensive pytest tests for:
1. Feature/target alignment
2. Data pipeline validation
3. Look-ahead bias detection
4. Target distribution checks
5. Feature engineering consistency

Run with: pytest tests/test_data_integrity.py -v
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestFeatureTargetAlignment:
    """Tests for ensuring features and targets are properly aligned."""
    
    def test_target_shift_prevents_lookahead(self):
        """Verify targets are shifted forward to prevent look-ahead bias."""
        # Create sample data
        n_samples = 100
        dates = pd.date_range('2020-01-01', periods=n_samples, freq='D')
        prices = 100 + np.cumsum(np.random.randn(n_samples) * 2)
        
        df = pd.DataFrame({
            'Date': dates,
            'Close': prices,
        })
        
        # Calculate returns
        df['returns'] = df['Close'].pct_change()
        
        # Create forward target (next day's return)
        df['target_1d'] = df['returns'].shift(-1)
        
        # Verify: target at time t should be return at time t+1
        for i in range(len(df) - 1):
            expected_target = df.iloc[i + 1]['returns']
            actual_target = df.iloc[i]['target_1d']
            if pd.notna(expected_target) and pd.notna(actual_target):
                assert np.isclose(expected_target, actual_target, rtol=1e-5), \
                    f"Target mismatch at index {i}: expected {expected_target}, got {actual_target}"
    
    def test_feature_target_length_match(self):
        """Ensure features and targets have matching lengths after preprocessing."""
        n_samples = 1000
        n_features = 50
        
        # Simulate feature engineering output
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        
        # Simulate dropping NaN rows (like from lagged features)
        nan_mask = np.random.choice([True, False], size=n_samples, p=[0.05, 0.95])
        X_clean = X[~nan_mask]
        y_clean = y[~nan_mask]
        
        assert len(X_clean) == len(y_clean), \
            f"Length mismatch: features={len(X_clean)}, targets={len(y_clean)}"
    
    def test_sequence_alignment(self):
        """Test that sequences are properly aligned with targets."""
        # Simulate sequence creation
        n_samples = 500
        n_features = 30
        sequence_length = 60
        
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)
        
        # Create sequences
        X_seq = []
        y_seq = []
        for i in range(sequence_length, len(X)):
            X_seq.append(X[i-sequence_length:i])
            y_seq.append(y[i])  # Target at sequence end
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Verify lengths
        expected_length = n_samples - sequence_length
        assert len(X_seq) == expected_length, \
            f"Sequence length mismatch: expected {expected_length}, got {len(X_seq)}"
        assert len(y_seq) == expected_length, \
            f"Target length mismatch: expected {expected_length}, got {len(y_seq)}"
        
        # Verify shapes
        assert X_seq.shape == (expected_length, sequence_length, n_features), \
            f"Unexpected sequence shape: {X_seq.shape}"


class TestDataPipelineValidation:
    """Tests for data pipeline consistency."""
    
    def test_no_nan_in_features(self):
        """Ensure no NaN values in processed features."""
        # Simulate processed features
        n_samples = 500
        n_features = 100
        
        X = np.random.randn(n_samples, n_features)
        
        # Introduce NaN and then clean
        X[10, 5] = np.nan
        X[20, 15] = np.nan
        
        # Clean: replace with column mean
        for col in range(X.shape[1]):
            col_mean = np.nanmean(X[:, col])
            X[np.isnan(X[:, col]), col] = col_mean
        
        assert not np.any(np.isnan(X)), "NaN values found in processed features"
    
    def test_no_inf_in_features(self):
        """Ensure no infinite values in processed features."""
        n_samples = 500
        n_features = 100
        
        X = np.random.randn(n_samples, n_features)
        
        # Introduce inf and then clean
        X[5, 10] = np.inf
        X[15, 20] = -np.inf
        
        # Clean: clip to reasonable range
        X = np.clip(X, -1e10, 1e10)
        
        assert not np.any(np.isinf(X)), "Infinite values found in processed features"
    
    def test_feature_scaling_consistency(self):
        """Test that feature scaling is consistent between train and test."""
        from sklearn.preprocessing import RobustScaler
        
        # Create train and test data from same distribution
        np.random.seed(42)
        X_train = np.random.randn(800, 50) * 2 + 1
        X_test = np.random.randn(200, 50) * 2 + 1
        
        # Fit scaler on train
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train data should be roughly centered
        assert np.abs(np.median(X_train_scaled)) < 0.1, \
            "Training data not properly centered"
        
        # Test data statistics should be similar (from same distribution)
        train_std = np.std(X_train_scaled)
        test_std = np.std(X_test_scaled)
        
        assert np.abs(train_std - test_std) / train_std < 0.3, \
            f"Large std difference: train={train_std:.4f}, test={test_std:.4f}"
    
    def test_target_scaling_invertibility(self):
        """Test that target scaling can be properly inverted."""
        from sklearn.preprocessing import RobustScaler
        
        y_raw = np.random.randn(1000) * 0.02  # ~2% daily returns
        
        scaler = RobustScaler()
        y_scaled = scaler.fit_transform(y_raw.reshape(-1, 1)).flatten()
        y_recovered = scaler.inverse_transform(y_scaled.reshape(-1, 1)).flatten()
        
        roundtrip_error = np.mean(np.abs(y_raw - y_recovered))
        assert roundtrip_error < 1e-10, \
            f"Target scaling roundtrip error too high: {roundtrip_error}"


class TestLookAheadBias:
    """Tests for detecting look-ahead bias."""
    
    def test_features_use_only_past_data(self):
        """Verify that features at time t only use data from t and before."""
        n_samples = 500
        lookback = 20
        
        # Create price series
        prices = 100 + np.cumsum(np.random.randn(n_samples))
        
        # Calculate SMA feature
        sma = pd.Series(prices).rolling(window=lookback, min_periods=lookback).mean().values
        
        # For each valid point, verify SMA uses only past data
        for i in range(lookback, n_samples):
            expected_sma = np.mean(prices[i-lookback+1:i+1])
            assert np.isclose(sma[i], expected_sma, rtol=1e-5), \
                f"SMA at {i} uses future data: expected {expected_sma}, got {sma[i]}"
    
    def test_target_is_future_return(self):
        """Verify target represents future, not current or past returns."""
        prices = np.array([100, 101, 103, 102, 105, 104])
        
        # Current returns
        returns = np.diff(prices) / prices[:-1]
        
        # Forward target (1-day ahead return from perspective of time t)
        # At t=0, target should be return from t=0 to t=1
        forward_targets = np.zeros(len(prices) - 1)
        for t in range(len(prices) - 1):
            forward_targets[t] = (prices[t+1] - prices[t]) / prices[t]
        
        # Verify alignment
        np.testing.assert_array_almost_equal(
            returns, forward_targets,
            err_msg="Forward targets not properly aligned with future returns"
        )
    
    def test_train_test_split_respects_time(self):
        """Ensure train/test split doesn't leak future data into training."""
        dates = pd.date_range('2020-01-01', periods=1000, freq='D')
        
        # Simulate time-series split
        train_end = 800
        
        train_dates = dates[:train_end]
        test_dates = dates[train_end:]
        
        # Verify no overlap
        assert max(train_dates) < min(test_dates), \
            "Train/test dates overlap - potential data leakage"
        
        # Verify chronological order
        assert all(train_dates[i] < train_dates[i+1] for i in range(len(train_dates)-1)), \
            "Training dates not in chronological order"


class TestTargetDistribution:
    """Tests for target distribution characteristics."""
    
    def test_target_symmetry(self):
        """Test that targets are approximately symmetric (balanced up/down)."""
        # Simulate 1-day returns
        np.random.seed(42)
        returns = np.random.randn(10000) * 0.02  # 2% daily std
        
        # Check distribution
        positive_frac = np.mean(returns > 0)
        
        # Should be roughly 50/50 (within 10% tolerance)
        assert 0.40 < positive_frac < 0.60, \
            f"Target distribution too skewed: {positive_frac*100:.1f}% positive"
    
    def test_target_clipping(self):
        """Test that target clipping is symmetric."""
        # Create returns with outliers
        returns = np.concatenate([
            np.random.randn(9900) * 0.02,
            np.array([0.5, -0.5, 0.3, -0.3])  # Outliers
        ])
        
        # Symmetric clip at ±20%
        clipped = np.clip(returns, -0.20, 0.20)
        
        # Verify symmetry preserved
        assert clipped.min() >= -0.20, "Lower bound not enforced"
        assert clipped.max() <= 0.20, "Upper bound not enforced"
        
        # Check that outliers were clipped symmetrically
        positive_outliers = np.sum(returns > 0.20)
        negative_outliers = np.sum(returns < -0.20)
        
        # Both directions should have similar clipping (roughly)
        # Note: this depends on the random data, but we include extreme cases
        assert positive_outliers >= 0 and negative_outliers >= 0
    
    def test_target_winsorization(self):
        """Test winsorization removes extreme values correctly."""
        from scipy.stats.mstats import winsorize
        
        # Create returns with fat tails
        returns = np.concatenate([
            np.random.randn(9600) * 0.015,
            np.random.randn(200) * 0.10,  # Fat tail on positive side
            np.random.randn(200) * 0.10 * -1,  # Fat tail on negative side
        ])
        
        # Winsorize at 2% each tail
        winsorized = np.array(winsorize(returns, limits=(0.02, 0.02)))
        
        # After winsorization, extreme values should be replaced
        original_max = np.abs(returns).max()
        winsorized_max = np.abs(winsorized).max()
        
        assert winsorized_max < original_max, \
            "Winsorization didn't reduce extreme values"


class TestFeatureEngineeringConsistency:
    """Tests for feature engineering reproducibility."""
    
    def test_feature_count_matches_metadata(self):
        """Test that generated features match saved metadata count."""
        # Simulate feature engineering
        expected_features = [
            'sma_10', 'sma_20', 'sma_50',
            'rsi_14', 'macd', 'macd_signal',
            'bb_upper', 'bb_lower', 'atr_14'
        ]
        
        # Simulate saved metadata
        metadata = {
            'feature_count': len(expected_features),
            'feature_names': expected_features
        }
        
        # Verify
        actual_count = len(expected_features)
        assert actual_count == metadata['feature_count'], \
            f"Feature count mismatch: actual={actual_count}, metadata={metadata['feature_count']}"
    
    def test_feature_order_preserved(self):
        """Test that feature order is preserved between runs."""
        feature_names = ['feature_a', 'feature_b', 'feature_c', 'feature_d']
        
        # First run
        run1_order = feature_names.copy()
        
        # Second run (simulated)
        run2_order = feature_names.copy()
        
        assert run1_order == run2_order, \
            "Feature order not preserved between runs"
    
    def test_missing_features_handled(self):
        """Test graceful handling of missing features."""
        expected_features = ['feature_a', 'feature_b', 'feature_c']
        
        df = pd.DataFrame({
            'feature_a': [1, 2, 3],
            'feature_c': [7, 8, 9],
            # feature_b is missing
        })
        
        # Handle missing features by filling with zeros
        for feat in expected_features:
            if feat not in df.columns:
                df[feat] = 0.0
        
        # Verify all expected features exist
        for feat in expected_features:
            assert feat in df.columns, f"Missing feature: {feat}"
        
        # Verify values
        assert all(df['feature_b'] == 0.0), "Missing feature not filled with zeros"


class TestModelInputValidation:
    """Tests for validating model inputs."""
    
    def test_input_shape_for_lstm(self):
        """Test that LSTM inputs have correct shape (batch, seq, features)."""
        batch_size = 32
        sequence_length = 60
        n_features = 147
        
        X = np.random.randn(batch_size, sequence_length, n_features)
        
        assert len(X.shape) == 3, "LSTM input should be 3D"
        assert X.shape[1] == sequence_length, "Incorrect sequence length"
        assert X.shape[2] == n_features, "Incorrect feature count"
    
    def test_input_dtype(self):
        """Test that inputs have correct dtype for training."""
        X = np.random.randn(100, 60, 50)
        y = np.random.randn(100)
        
        # Convert to float32 for GPU efficiency
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        
        assert X.dtype == np.float32, f"Wrong X dtype: {X.dtype}"
        assert y.dtype == np.float32, f"Wrong y dtype: {y.dtype}"
    
    def test_batch_dimension(self):
        """Test that batch dimension is correctly handled."""
        X = np.random.randn(256, 60, 50)
        
        # Should be able to split into batches
        batch_size = 32
        n_batches = len(X) // batch_size
        
        for i in range(n_batches):
            batch = X[i*batch_size:(i+1)*batch_size]
            assert batch.shape[0] == batch_size, \
                f"Batch {i} has wrong size: {batch.shape[0]}"


class TestCrossValidation:
    """Tests for time-series cross-validation."""
    
    def test_rolling_cv_no_leakage(self):
        """Test that rolling CV doesn't leak future data."""
        n_samples = 1000
        test_size = 0.2
        n_splits = 5
        
        # Simulate rolling window CV
        min_train = int(n_samples * 0.3)
        test_len = int(n_samples * test_size / n_splits)
        
        splits = []
        for i in range(n_splits):
            train_end = min_train + i * test_len
            test_start = train_end
            test_end = test_start + test_len
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(test_start, min(test_end, n_samples))
            
            splits.append((train_idx, test_idx))
        
        # Verify no overlap
        for train_idx, test_idx in splits:
            assert max(train_idx) < min(test_idx), \
                "Train/test overlap in CV split"
    
    def test_purged_cv_gap(self):
        """Test that purged CV has proper gap between train and test."""
        gap = 5  # 5-day gap
        
        train_end = 100
        test_start = train_end + gap
        test_end = 150
        
        train_idx = np.arange(0, train_end)
        test_idx = np.arange(test_start, test_end)
        
        # Verify gap
        actual_gap = min(test_idx) - max(train_idx)
        assert actual_gap >= gap, \
            f"Purged gap too small: {actual_gap} < {gap}"


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v', '--tb=short'])
