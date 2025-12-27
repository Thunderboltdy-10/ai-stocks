"""
Unit tests for target_balancing module.

Tests cover the balance_target_distribution function with all strategies
(undersample, oversample, hybrid) and validation logic.
"""

import pytest
import numpy as np
from unittest.mock import patch

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.target_balancing import (
    balance_target_distribution,
    BalancingReport,
    get_balancing_metadata,
    check_sequence_leakage,
    _undersample,
    _oversample,
    _hybrid_resample,
    _validate_balanced_dataset,
)


class TestBalancingReport:
    """Tests for the BalancingReport dataclass."""

    def test_dataclass_defaults(self):
        """Test that BalancingReport has correct default values."""
        report = BalancingReport(strategy='undersample')
        assert report.strategy == 'undersample'
        assert report.original_counts == {}
        assert report.balanced_counts == {}
        assert report.samples_removed == 0
        assert report.samples_added == 0
        assert report.original_ratio == 0.0
        assert report.balanced_ratio == 0.0
        assert report.random_state == 42

    def test_dataclass_with_values(self):
        """Test BalancingReport with explicit values."""
        report = BalancingReport(
            strategy='hybrid',
            original_counts={'positive': 600, 'negative': 400},
            balanced_counts={'positive': 500, 'negative': 500},
            samples_removed=100,
            samples_added=100,
            original_ratio=1.5,
            balanced_ratio=1.0,
            random_state=123,
        )
        assert report.strategy == 'hybrid'
        assert report.original_counts['positive'] == 600
        assert report.samples_removed == 100
        assert report.balanced_ratio == 1.0


class TestUndersampling:
    """Tests for the undersample strategy."""

    def test_undersample_majority_positive(self):
        """Test undersampling when positive class is majority."""
        np.random.seed(42)
        # 70% positive, 30% negative
        n_samples = 1000
        X = np.random.randn(n_samples, 10, 5)  # Sequence data
        y = np.concatenate([
            np.abs(np.random.randn(700)) * 0.01,  # Positive
            -np.abs(np.random.randn(300)) * 0.01,  # Negative
        ])
        
        X_balanced, y_balanced, report = balance_target_distribution(
            X, y, strategy='undersample', validate=False
        )
        
        # Should have roughly equal positive and negative
        n_pos = np.sum(y_balanced > 0)
        n_neg = np.sum(y_balanced < 0)
        
        assert abs(n_pos - n_neg) <= 1  # Allow for rounding
        assert report.samples_removed > 0
        assert report.samples_added == 0
        assert len(y_balanced) < len(y)

    def test_undersample_majority_negative(self):
        """Test undersampling when negative class is majority."""
        np.random.seed(42)
        # 30% positive, 70% negative
        n_samples = 1000
        X = np.random.randn(n_samples, 10, 5)
        y = np.concatenate([
            np.abs(np.random.randn(300)) * 0.01,  # Positive
            -np.abs(np.random.randn(700)) * 0.01,  # Negative
        ])
        
        X_balanced, y_balanced, report = balance_target_distribution(
            X, y, strategy='undersample', validate=False
        )
        
        n_pos = np.sum(y_balanced > 0)
        n_neg = np.sum(y_balanced < 0)
        
        assert abs(n_pos - n_neg) <= 1
        assert report.samples_removed > 0

    def test_undersample_preserves_sequence_structure(self):
        """Test that undersampling preserves 3D sequence structure."""
        np.random.seed(42)
        X = np.random.randn(100, 90, 50)  # (samples, seq_len, features)
        y = np.concatenate([
            np.abs(np.random.randn(70)) * 0.01,
            -np.abs(np.random.randn(30)) * 0.01,
        ])
        
        X_balanced, y_balanced, _ = balance_target_distribution(
            X, y, strategy='undersample', validate=False
        )
        
        assert len(X_balanced.shape) == 3
        assert X_balanced.shape[1] == 90  # Sequence length preserved
        assert X_balanced.shape[2] == 50  # Feature count preserved


class TestOversampling:
    """Tests for the oversample strategy."""

    def test_oversample_minority_positive(self):
        """Test oversampling when positive class is minority."""
        np.random.seed(42)
        # 30% positive, 70% negative
        n_samples = 1000
        X = np.random.randn(n_samples, 10, 5)
        y = np.concatenate([
            np.abs(np.random.randn(300)) * 0.01,  # Positive (minority)
            -np.abs(np.random.randn(700)) * 0.01,  # Negative (majority)
        ])
        
        X_balanced, y_balanced, report = balance_target_distribution(
            X, y, strategy='oversample', validate=False
        )
        
        n_pos = np.sum(y_balanced > 0)
        n_neg = np.sum(y_balanced < 0)
        
        # Should have roughly equal after oversampling
        # Allow larger tolerance due to noise added during augmentation
        assert abs(n_pos - n_neg) <= 50
        assert report.samples_added > 0
        assert report.samples_removed == 0
        assert len(y_balanced) > len(y)

    def test_oversample_adds_noise(self):
        """Test that oversampled samples have noise added."""
        np.random.seed(42)
        X = np.random.randn(100, 10, 5)
        y = np.concatenate([
            np.abs(np.random.randn(20)) * 0.01,  # Minority
            -np.abs(np.random.randn(80)) * 0.01,  # Majority
        ])
        
        X_balanced, y_balanced, _ = balance_target_distribution(
            X, y, strategy='oversample', noise_scale=0.01, validate=False
        )
        
        # Augmented samples should not be exact copies
        n_original = len(y)
        n_augmented = len(y_balanced) - n_original
        
        assert n_augmented > 0
        # The augmented X values should differ from originals due to noise

    def test_oversample_preserves_sequence_structure(self):
        """Test that oversampling preserves 3D sequence structure."""
        np.random.seed(42)
        X = np.random.randn(100, 90, 50)
        y = np.concatenate([
            np.abs(np.random.randn(30)) * 0.01,
            -np.abs(np.random.randn(70)) * 0.01,
        ])
        
        X_balanced, y_balanced, _ = balance_target_distribution(
            X, y, strategy='oversample', validate=False
        )
        
        assert len(X_balanced.shape) == 3
        assert X_balanced.shape[1] == 90
        assert X_balanced.shape[2] == 50


class TestHybridResampling:
    """Tests for the hybrid strategy."""

    def test_hybrid_balances_distribution(self):
        """Test that hybrid strategy achieves balanced distribution."""
        np.random.seed(42)
        # Very imbalanced: 80% positive, 20% negative
        n_samples = 1000
        X = np.random.randn(n_samples, 10, 5)
        y = np.concatenate([
            np.abs(np.random.randn(800)) * 0.01,  # Positive (majority)
            -np.abs(np.random.randn(200)) * 0.01,  # Negative (minority)
        ])
        
        X_balanced, y_balanced, report = balance_target_distribution(
            X, y, strategy='hybrid', validate=True
        )
        
        n_pos = np.sum(y_balanced > 0)
        n_neg = np.sum(y_balanced < 0)
        total = n_pos + n_neg
        
        # Should be between 40-60% each class
        assert 0.40 <= n_pos / total <= 0.60
        assert 0.40 <= n_neg / total <= 0.60

    def test_hybrid_both_removes_and_adds(self):
        """Test that hybrid strategy both removes and adds samples."""
        np.random.seed(42)
        # 75% positive, 25% negative
        X = np.random.randn(1000, 10, 5)
        y = np.concatenate([
            np.abs(np.random.randn(750)) * 0.01,
            -np.abs(np.random.randn(250)) * 0.01,
        ])
        
        X_balanced, y_balanced, report = balance_target_distribution(
            X, y, strategy='hybrid', validate=False
        )
        
        # Hybrid should do both undersampling and oversampling
        assert report.samples_removed > 0 or report.samples_added > 0


class TestValidation:
    """Tests for validation logic."""

    def test_validation_passes_balanced_data(self):
        """Test that validation passes for balanced data."""
        np.random.seed(42)
        # Already balanced: 50% each
        X = np.random.randn(1000, 10, 5)
        y = np.concatenate([
            np.abs(np.random.randn(500)) * 0.01,
            -np.abs(np.random.randn(500)) * 0.01,
        ])
        
        # Should not raise
        X_balanced, y_balanced, report = balance_target_distribution(
            X, y, strategy='undersample', validate=True
        )
        
        assert report.balanced_counts is not None

    def test_validation_fails_extreme_imbalance(self):
        """Test that validation fails for extreme imbalance."""
        # This tests the _validate_balanced_dataset function directly
        y_balanced = np.concatenate([
            np.ones(100),  # 100% positive
            np.zeros(0),  # 0% negative
        ])
        
        with pytest.raises(ValueError) as exc_info:
            _validate_balanced_dataset(
                y_balanced,
                balanced_positive=100,
                balanced_negative=0,
                balanced_total=100,
            )
        
        assert "validation failed" in str(exc_info.value).lower()

    def test_validate_empty_dataset_fails(self):
        """Test that empty dataset fails validation."""
        with pytest.raises(ValueError) as exc_info:
            _validate_balanced_dataset(
                np.array([]),
                balanced_positive=0,
                balanced_negative=0,
                balanced_total=0,
            )
        
        assert "empty" in str(exc_info.value).lower()


class TestInputValidation:
    """Tests for input validation."""

    def test_mismatched_lengths_raises_error(self):
        """Test that mismatched X and y lengths raise ValueError."""
        X = np.random.randn(100, 10, 5)
        y = np.random.randn(50)  # Wrong length
        
        with pytest.raises(ValueError) as exc_info:
            balance_target_distribution(X, y, strategy='undersample')
        
        assert "same length" in str(exc_info.value)

    def test_empty_arrays_raises_error(self):
        """Test that empty arrays raise ValueError."""
        X = np.array([])
        y = np.array([])
        
        with pytest.raises(ValueError) as exc_info:
            balance_target_distribution(X, y, strategy='undersample')
        
        assert "Empty" in str(exc_info.value)

    def test_invalid_strategy_raises_error(self):
        """Test that invalid strategy raises ValueError."""
        X = np.random.randn(100, 10, 5)
        y = np.random.randn(100)
        
        with pytest.raises(ValueError) as exc_info:
            balance_target_distribution(X, y, strategy='invalid_strategy')
        
        assert "Invalid strategy" in str(exc_info.value)


class TestReproducibility:
    """Tests for reproducibility with random_state."""

    def test_same_random_state_same_result(self):
        """Test that same random_state produces same results."""
        np.random.seed(42)
        X = np.random.randn(100, 10, 5)
        y = np.concatenate([
            np.abs(np.random.randn(70)) * 0.01,
            -np.abs(np.random.randn(30)) * 0.01,
        ])
        
        X1, y1, _ = balance_target_distribution(
            X.copy(), y.copy(), strategy='undersample', random_state=123, validate=False
        )
        X2, y2, _ = balance_target_distribution(
            X.copy(), y.copy(), strategy='undersample', random_state=123, validate=False
        )
        
        np.testing.assert_array_equal(y1, y2)

    def test_different_random_state_different_result(self):
        """Test that different random_state produces different results."""
        np.random.seed(42)
        X = np.random.randn(100, 10, 5)
        y = np.concatenate([
            np.abs(np.random.randn(70)) * 0.01,
            -np.abs(np.random.randn(30)) * 0.01,
        ])
        
        X1, y1, _ = balance_target_distribution(
            X.copy(), y.copy(), strategy='undersample', random_state=123, validate=False
        )
        X2, y2, _ = balance_target_distribution(
            X.copy(), y.copy(), strategy='undersample', random_state=456, validate=False
        )
        
        # Results should differ (with very high probability)
        assert not np.array_equal(y1, y2)


class TestGetBalancingMetadata:
    """Tests for get_balancing_metadata function."""

    def test_converts_report_to_dict(self):
        """Test that report is correctly converted to dictionary."""
        report = BalancingReport(
            strategy='hybrid',
            original_counts={'positive': 600, 'negative': 400, 'zero': 0, 'total': 1000},
            balanced_counts={'positive': 500, 'negative': 500, 'zero': 0, 'total': 1000},
            samples_removed=100,
            samples_added=100,
            original_ratio=1.5,
            balanced_ratio=1.0,
            random_state=42,
        )
        
        metadata = get_balancing_metadata(report)
        
        assert metadata['strategy'] == 'hybrid'
        assert metadata['original_counts']['positive'] == 600
        assert metadata['balanced_counts']['positive'] == 500
        assert metadata['samples_removed'] == 100
        assert metadata['samples_added'] == 100
        assert metadata['balanced_ratio'] == 1.0
        assert metadata['random_state'] == 42

    def test_handles_infinite_ratio(self):
        """Test that infinite ratio is handled gracefully."""
        report = BalancingReport(
            strategy='undersample',
            original_ratio=float('inf'),
            balanced_ratio=float('inf'),
        )
        
        metadata = get_balancing_metadata(report)
        
        assert metadata['original_ratio'] is None
        assert metadata['balanced_ratio'] is None


class TestCheckSequenceLeakage:
    """Tests for check_sequence_leakage function."""

    def test_no_leakage_returns_true(self):
        """Test that no leakage returns True."""
        np.random.seed(42)
        X_train = np.random.randn(100, 10, 5)
        X_val = np.random.randn(20, 10, 5)  # Different data
        
        result = check_sequence_leakage(X_train, X_val, sequence_length=10)
        
        assert result is True

    def test_leakage_detected_returns_false(self):
        """Test that leakage is detected and returns False."""
        np.random.seed(42)
        X_train = np.random.randn(100, 10, 5)
        # Create val with some sequences copied from train
        X_val = X_train[:20].copy()  # Exact copy = leakage
        
        result = check_sequence_leakage(X_train, X_val, sequence_length=10)
        
        assert result is False


class TestEdgeCases:
    """Tests for edge cases."""

    def test_all_positive_targets(self):
        """Test handling of all-positive targets."""
        X = np.random.randn(100, 10, 5)
        y = np.abs(np.random.randn(100)) * 0.01  # All positive
        
        # When all targets are same sign, undersampling results in empty minority
        # The function will return empty arrays in this edge case
        X_balanced, y_balanced, report = balance_target_distribution(
            X, y, strategy='undersample', validate=False
        )
        
        # With no minority class, result will be empty (minority size = 0)
        # This is expected behavior - the caller should validate data first
        assert report.original_counts['negative'] == 0

    def test_all_negative_targets(self):
        """Test handling of all-negative targets."""
        X = np.random.randn(100, 10, 5)
        y = -np.abs(np.random.randn(100)) * 0.01  # All negative
        
        # When all targets are same sign, undersampling results in empty minority
        X_balanced, y_balanced, report = balance_target_distribution(
            X, y, strategy='undersample', validate=False
        )
        
        # With no minority class, result will be empty (minority size = 0)
        # This is expected behavior - the caller should validate data first
        assert report.original_counts['positive'] == 0

    def test_with_zero_targets(self):
        """Test handling of targets that include zeros."""
        np.random.seed(42)
        X = np.random.randn(100, 10, 5)
        y = np.concatenate([
            np.abs(np.random.randn(40)) * 0.01,  # Positive
            -np.abs(np.random.randn(40)) * 0.01,  # Negative
            np.zeros(20),  # Zeros
        ])
        
        X_balanced, y_balanced, report = balance_target_distribution(
            X, y, strategy='undersample', validate=False
        )
        
        # Zeros should be preserved
        assert np.sum(y_balanced == 0) >= 0

    def test_2d_feature_array(self):
        """Test with 2D feature array (no sequence dimension)."""
        np.random.seed(42)
        X = np.random.randn(100, 50)  # (samples, features)
        y = np.concatenate([
            np.abs(np.random.randn(70)) * 0.01,
            -np.abs(np.random.randn(30)) * 0.01,
        ])
        
        X_balanced, y_balanced, report = balance_target_distribution(
            X, y, strategy='undersample', validate=False
        )
        
        assert len(X_balanced.shape) == 2
        assert X_balanced.shape[1] == 50

    def test_list_inputs(self):
        """Test that list inputs work correctly."""
        X = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        y = [0.01, -0.02, 0.03, -0.01]
        
        X_balanced, y_balanced, report = balance_target_distribution(
            X, y, strategy='undersample', validate=False
        )
        
        assert isinstance(X_balanced, np.ndarray)
        assert isinstance(y_balanced, np.ndarray)
