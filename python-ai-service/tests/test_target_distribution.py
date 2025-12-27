"""
Unit tests for target distribution validation in train_1d_regressor_final.

Tests cover the validate_target_distribution function which ensures
balanced target distributions before model training.
"""

import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

# Import the module under test
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.train_1d_regressor_final import validate_target_distribution


class TestValidateTargetDistribution:
    """Tests for the validate_target_distribution function."""

    def test_balanced_distribution_passes(self):
        """Test that a balanced distribution passes validation."""
        np.random.seed(42)
        # Create balanced returns: roughly 50% positive, 50% negative
        y_train = np.random.normal(0, 0.02, 1000)
        
        with patch('training.train_1d_regressor_final._save_target_distribution_metadata'):
            result = validate_target_distribution(
                y_train=y_train,
                symbol='TEST',
                min_positive_pct=0.30,
                min_negative_pct=0.30,
                save_metadata=False
            )
        
        assert result['validation_passed'] is True
        assert len(result['validation_issues']) == 0
        assert result['pct_positive'] > 0.30
        assert result['pct_negative'] > 0.30

    def test_too_many_positive_fails(self):
        """Test that too many positive returns fails validation."""
        np.random.seed(42)
        # Create mostly positive returns (80% positive)
        y_train = np.abs(np.random.normal(0, 0.02, 800))
        y_negative = -np.abs(np.random.normal(0, 0.02, 200))
        y_train = np.concatenate([y_train, y_negative])
        
        with patch('training.train_1d_regressor_final._save_target_distribution_metadata'):
            with pytest.raises(ValueError) as exc_info:
                validate_target_distribution(
                    y_train=y_train,
                    symbol='TEST',
                    min_positive_pct=0.30,
                    min_negative_pct=0.30,
                    save_metadata=False
                )
        
        assert "Insufficient negative returns" in str(exc_info.value)

    def test_too_many_negative_fails(self):
        """Test that too many negative returns fails validation."""
        np.random.seed(42)
        # Create mostly negative returns (80% negative)
        y_train = -np.abs(np.random.normal(0, 0.02, 800))
        y_positive = np.abs(np.random.normal(0, 0.02, 200))
        y_train = np.concatenate([y_train, y_positive])
        
        with patch('training.train_1d_regressor_final._save_target_distribution_metadata'):
            with pytest.raises(ValueError) as exc_info:
                validate_target_distribution(
                    y_train=y_train,
                    symbol='TEST',
                    min_positive_pct=0.30,
                    min_negative_pct=0.30,
                    save_metadata=False
                )
        
        assert "Insufficient positive returns" in str(exc_info.value)

    def test_custom_thresholds(self):
        """Test that custom thresholds are respected."""
        np.random.seed(42)
        # 60% positive, 40% negative
        y_positive = np.abs(np.random.normal(0.01, 0.02, 600))
        y_negative = -np.abs(np.random.normal(0.01, 0.02, 400))
        y_train = np.concatenate([y_positive, y_negative])
        
        # With default 30% thresholds, this should pass
        with patch('training.train_1d_regressor_final._save_target_distribution_metadata'):
            result = validate_target_distribution(
                y_train=y_train,
                symbol='TEST',
                min_positive_pct=0.30,
                min_negative_pct=0.30,
                save_metadata=False
            )
        
        assert result['validation_passed'] is True
        
        # With 45% threshold for negative, this should fail
        with patch('training.train_1d_regressor_final._save_target_distribution_metadata'):
            with pytest.raises(ValueError):
                validate_target_distribution(
                    y_train=y_train,
                    symbol='TEST',
                    min_positive_pct=0.30,
                    min_negative_pct=0.45,
                    save_metadata=False
                )

    def test_percentile_calculation(self):
        """Test that percentiles are calculated correctly."""
        # Create deterministic data
        y_train = np.linspace(-0.1, 0.1, 101)
        
        with patch('training.train_1d_regressor_final._save_target_distribution_metadata'):
            result = validate_target_distribution(
                y_train=y_train,
                symbol='TEST',
                min_positive_pct=0.30,
                min_negative_pct=0.30,
                save_metadata=False
            )
        
        # Check percentiles are in expected order
        percentiles = result['percentiles']
        assert percentiles['p10'] < percentiles['p25']
        assert percentiles['p25'] < percentiles['p50']
        assert percentiles['p50'] < percentiles['p75']
        assert percentiles['p75'] < percentiles['p90']
        
        # Check median is close to 0 for symmetric distribution
        assert abs(percentiles['p50']) < 0.01

    def test_mean_positive_negative_calculation(self):
        """Test that mean positive and negative returns are calculated correctly."""
        # Create specific data
        y_positive = np.array([0.01, 0.02, 0.03, 0.04, 0.05])  # mean = 0.03
        y_negative = np.array([-0.01, -0.02, -0.03, -0.04, -0.05])  # mean = -0.03
        y_train = np.concatenate([y_positive, y_negative])
        
        with patch('training.train_1d_regressor_final._save_target_distribution_metadata'):
            result = validate_target_distribution(
                y_train=y_train,
                symbol='TEST',
                min_positive_pct=0.30,
                min_negative_pct=0.30,
                save_metadata=False
            )
        
        assert abs(result['mean_positive'] - 0.03) < 0.0001
        assert abs(result['mean_negative'] - (-0.03)) < 0.0001

    def test_metadata_structure(self):
        """Test that returned metadata has correct structure."""
        np.random.seed(42)
        y_train = np.random.normal(0, 0.02, 1000)
        
        with patch('training.train_1d_regressor_final._save_target_distribution_metadata'):
            result = validate_target_distribution(
                y_train=y_train,
                symbol='TEST',
                save_metadata=False
            )
        
        # Check all expected keys are present
        expected_keys = [
            'symbol', 'n_total', 'n_positive', 'n_negative', 'n_zero',
            'pct_positive', 'pct_negative', 'pct_zero',
            'mean_positive', 'mean_negative', 'std_positive', 'std_negative',
            'mean', 'std', 'percentiles', 'validation_thresholds',
            'validated_at', 'validation_passed', 'validation_issues'
        ]
        
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        
        # Check percentiles structure
        percentile_keys = ['p10', 'p25', 'p50', 'p75', 'p90']
        for key in percentile_keys:
            assert key in result['percentiles'], f"Missing percentile: {key}"

    def test_empty_array_raises_error(self):
        """Test that empty array raises ValueError."""
        with patch('training.train_1d_regressor_final._save_target_distribution_metadata'):
            with pytest.raises(ValueError) as exc_info:
                validate_target_distribution(
                    y_train=np.array([]),
                    symbol='TEST',
                    save_metadata=False
                )
        
        assert "Empty training targets" in str(exc_info.value)

    def test_all_zeros_fails(self):
        """Test that all-zero targets fail validation."""
        y_train = np.zeros(1000)
        
        with patch('training.train_1d_regressor_final._save_target_distribution_metadata'):
            with pytest.raises(ValueError) as exc_info:
                validate_target_distribution(
                    y_train=y_train,
                    symbol='TEST',
                    min_positive_pct=0.30,
                    min_negative_pct=0.30,
                    save_metadata=False
                )
        
        # Should fail both positive and negative checks
        assert "Insufficient" in str(exc_info.value)

    def test_list_input(self):
        """Test that list input works correctly."""
        y_train = [0.01, -0.02, 0.03, -0.01, 0.02, -0.03, 0.01, -0.02, 0.01, -0.01]
        
        with patch('training.train_1d_regressor_final._save_target_distribution_metadata'):
            result = validate_target_distribution(
                y_train=y_train,
                symbol='TEST',
                min_positive_pct=0.30,
                min_negative_pct=0.30,
                save_metadata=False
            )
        
        assert result['n_total'] == 10
        assert result['validation_passed'] is True

    def test_2d_array_flattened(self):
        """Test that 2D arrays are flattened correctly."""
        np.random.seed(42)
        y_train = np.random.normal(0, 0.02, (100, 1))
        
        with patch('training.train_1d_regressor_final._save_target_distribution_metadata'):
            result = validate_target_distribution(
                y_train=y_train,
                symbol='TEST',
                save_metadata=False
            )
        
        assert result['n_total'] == 100


class TestTargetDistributionMetadataSave:
    """Tests for metadata saving functionality."""

    def test_save_metadata_called(self):
        """Test that metadata save is called when save_metadata=True."""
        np.random.seed(42)
        y_train = np.random.normal(0, 0.02, 100)
        
        with patch('training.train_1d_regressor_final._save_target_distribution_metadata') as mock_save:
            validate_target_distribution(
                y_train=y_train,
                symbol='TEST',
                save_metadata=True
            )
        
        mock_save.assert_called_once()
        call_args = mock_save.call_args[0]
        assert call_args[0] == 'TEST'  # symbol
        assert isinstance(call_args[1], dict)  # metadata

    def test_save_metadata_not_called_when_disabled(self):
        """Test that metadata save is not called when save_metadata=False."""
        np.random.seed(42)
        y_train = np.random.normal(0, 0.02, 100)
        
        with patch('training.train_1d_regressor_final._save_target_distribution_metadata') as mock_save:
            validate_target_distribution(
                y_train=y_train,
                symbol='TEST',
                save_metadata=False
            )
        
        mock_save.assert_not_called()

    def test_save_metadata_called_on_failure(self):
        """Test that metadata is saved even when validation fails."""
        # Create imbalanced data (90% positive)
        y_train = np.concatenate([
            np.abs(np.random.normal(0.01, 0.02, 900)),
            -np.abs(np.random.normal(0.01, 0.02, 100))
        ])
        
        with patch('training.train_1d_regressor_final._save_target_distribution_metadata') as mock_save:
            with pytest.raises(ValueError):
                validate_target_distribution(
                    y_train=y_train,
                    symbol='TEST',
                    min_positive_pct=0.30,
                    min_negative_pct=0.30,
                    save_metadata=True
                )
        
        # Should still save metadata for debugging
        mock_save.assert_called_once()
        call_args = mock_save.call_args[0]
        metadata = call_args[1]
        assert metadata['validation_passed'] is False
        assert len(metadata['validation_issues']) > 0
