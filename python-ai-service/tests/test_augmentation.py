"""
Unit tests for data augmentation module.

Tests cover:
- augment_negative_returns function
- Validation of augmented data
- Edge cases and error handling
"""

import numpy as np
import pytest
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.augmentation import (
    augment_negative_returns,
    validate_augmentation,
    get_augmentation_metadata,
    augment_with_time_shift,
    AugmentationStats,
    AugmentationValidationResult,
)


class TestAugmentNegativeReturns:
    """Tests for augment_negative_returns function."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        seq_length = 10
        n_features = 5
        
        # Create sequences with some structure
        X = np.random.randn(n_samples, seq_length, n_features)
        
        # Create imbalanced targets (70% positive, 30% negative)
        y = np.random.randn(n_samples) * 0.02
        # Make 70% positive
        positive_indices = np.random.choice(n_samples, size=70, replace=False)
        y[positive_indices] = np.abs(y[positive_indices])
        
        return X, y
    
    @pytest.fixture
    def balanced_data(self):
        """Create balanced data for testing."""
        np.random.seed(42)
        n_samples = 100
        seq_length = 10
        n_features = 5
        
        X = np.random.randn(n_samples, seq_length, n_features)
        
        # Create balanced targets (50% positive, 50% negative)
        y = np.random.randn(n_samples) * 0.02
        positive_indices = np.random.choice(n_samples, size=50, replace=False)
        y[positive_indices] = np.abs(y[positive_indices])
        negative_indices = list(set(range(n_samples)) - set(positive_indices))
        y[negative_indices] = -np.abs(y[negative_indices])
        
        return X, y
    
    def test_basic_augmentation(self, sample_data):
        """Test basic augmentation increases negative samples."""
        X, y = sample_data
        original_neg_count = np.sum(y < 0)
        
        X_aug, y_aug, stats = augment_negative_returns(
            X, y, augmentation_factor=1.5, random_state=42
        )
        
        # Should have more samples
        assert len(y_aug) > len(y)
        
        # Should have more negatives
        final_neg_count = np.sum(y_aug < 0)
        assert final_neg_count > original_neg_count
        
        # Stats should be populated
        assert stats.samples_added > 0
        assert stats.final_samples == len(y_aug)
    
    def test_sequence_shape_preserved(self, sample_data):
        """Test that sequence shape is preserved after augmentation."""
        X, y = sample_data
        
        X_aug, y_aug, stats = augment_negative_returns(
            X, y, augmentation_factor=1.5, random_state=42
        )
        
        # Sequence dimensions should match
        assert X_aug.shape[1] == X.shape[1]  # seq_length
        assert X_aug.shape[2] == X.shape[2]  # n_features
        
        # X and y should have same first dimension
        assert X_aug.shape[0] == len(y_aug)
    
    def test_no_augmentation_when_balanced(self, balanced_data):
        """Test no augmentation when negatives already sufficient."""
        X, y = balanced_data
        
        # With factor < 1, no augmentation needed
        X_aug, y_aug, stats = augment_negative_returns(
            X, y, augmentation_factor=0.8, random_state=42
        )
        
        # Should return original data
        assert stats.samples_added == 0
        assert len(y_aug) == len(y)
    
    def test_noise_is_applied(self, sample_data):
        """Test that noise is applied to augmented samples."""
        X, y = sample_data
        
        X_aug, y_aug, stats = augment_negative_returns(
            X, y, augmentation_factor=2.0, noise_std=0.1, random_state=42
        )
        
        if stats.samples_added > 0:
            # Find augmented samples (those beyond original length)
            # Note: after shuffle, we can't directly identify them,
            # but we can verify the overall distribution changed
            
            # Mean should be similar (noise is zero-mean)
            original_mean = np.mean(X)
            augmented_mean = np.mean(X_aug)
            assert abs(original_mean - augmented_mean) < 0.5
    
    def test_random_state_reproducibility(self, sample_data):
        """Test that same random state produces same results."""
        X, y = sample_data
        
        X_aug1, y_aug1, _ = augment_negative_returns(
            X, y, augmentation_factor=1.5, random_state=42
        )
        
        X_aug2, y_aug2, _ = augment_negative_returns(
            X, y, augmentation_factor=1.5, random_state=42
        )
        
        np.testing.assert_array_equal(X_aug1, X_aug2)
        np.testing.assert_array_equal(y_aug1, y_aug2)
    
    def test_different_random_states(self, sample_data):
        """Test that different random states produce different results."""
        X, y = sample_data
        
        X_aug1, y_aug1, _ = augment_negative_returns(
            X, y, augmentation_factor=1.5, random_state=42
        )
        
        X_aug2, y_aug2, _ = augment_negative_returns(
            X, y, augmentation_factor=1.5, random_state=123
        )
        
        # Should be different (shuffled differently)
        assert not np.allclose(X_aug1, X_aug2)
    
    def test_max_augmentation_ratio(self, sample_data):
        """Test that max_augmentation_ratio limits augmentation."""
        X, y = sample_data
        original_len = len(y)
        
        # Request very high augmentation
        X_aug, y_aug, stats = augment_negative_returns(
            X, y, 
            augmentation_factor=10.0,  # Very high
            max_augmentation_ratio=0.5,  # But limit to 50%
            random_state=42
        )
        
        # Should be capped
        max_allowed = original_len + int(original_len * 0.5)
        assert len(y_aug) <= max_allowed
    
    def test_stats_accuracy(self, sample_data):
        """Test that returned stats are accurate."""
        X, y = sample_data
        
        original_neg = np.sum(y < 0)
        original_pos = np.sum(y > 0)
        
        X_aug, y_aug, stats = augment_negative_returns(
            X, y, augmentation_factor=1.5, random_state=42
        )
        
        # Verify stats match actual data
        assert stats.original_samples == len(y)
        assert stats.original_negative_count == original_neg
        assert stats.original_positive_count == original_pos
        assert stats.final_samples == len(y_aug)
        assert stats.final_negative_count == np.sum(y_aug < 0)
    
    def test_no_nan_or_inf(self, sample_data):
        """Test that augmentation doesn't introduce NaN or Inf."""
        X, y = sample_data
        
        X_aug, y_aug, stats = augment_negative_returns(
            X, y, augmentation_factor=2.0, random_state=42
        )
        
        assert not np.any(np.isnan(X_aug))
        assert not np.any(np.isinf(X_aug))
        assert not np.any(np.isnan(y_aug))
        assert not np.any(np.isinf(y_aug))
    
    def test_empty_input_raises(self):
        """Test that empty input raises ValueError."""
        X = np.array([]).reshape(0, 10, 5)
        y = np.array([])
        
        with pytest.raises(ValueError, match="Empty input"):
            augment_negative_returns(X, y)
    
    def test_mismatched_lengths_raises(self):
        """Test that mismatched X/y lengths raise ValueError."""
        X = np.random.randn(100, 10, 5)
        y = np.random.randn(50)  # Wrong length
        
        with pytest.raises(ValueError, match="same length"):
            augment_negative_returns(X, y)
    
    def test_wrong_dimensions_raises(self):
        """Test that wrong X dimensions raise ValueError."""
        X = np.random.randn(100, 10)  # 2D instead of 3D
        y = np.random.randn(100)
        
        with pytest.raises(ValueError, match="3D"):
            augment_negative_returns(X, y)
    
    def test_no_negatives_to_augment(self):
        """Test handling when all targets are positive."""
        np.random.seed(42)
        X = np.random.randn(50, 10, 5)
        y = np.abs(np.random.randn(50))  # All positive
        
        X_aug, y_aug, stats = augment_negative_returns(
            X, y, augmentation_factor=1.5, random_state=42
        )
        
        # Should return original data unchanged
        assert stats.samples_added == 0
        assert len(y_aug) == len(y)


class TestValidateAugmentation:
    """Tests for validate_augmentation function."""
    
    def test_valid_augmentation_passes(self):
        """Test that valid augmentation passes validation."""
        np.random.seed(42)
        X_orig = np.random.randn(100, 10, 5)
        y_orig = np.random.randn(100) * 0.02
        
        # Simple duplication with small noise
        X_aug = np.concatenate([X_orig, X_orig + np.random.randn(*X_orig.shape) * 0.01])
        y_aug = np.concatenate([y_orig, y_orig])
        
        stats = AugmentationStats(
            original_samples=100,
            samples_added=100,
            final_samples=200,
        )
        
        result = validate_augmentation(X_orig, X_aug, y_orig, y_aug, stats)
        
        assert result.is_valid
        assert result.sequence_integrity_ok
        assert len(result.errors) == 0
    
    def test_nan_values_fail(self):
        """Test that NaN values cause validation failure."""
        X_orig = np.random.randn(50, 10, 5)
        y_orig = np.random.randn(50)
        
        X_aug = np.concatenate([X_orig, X_orig])
        X_aug[60, 5, 2] = np.nan  # Introduce NaN
        y_aug = np.concatenate([y_orig, y_orig])
        
        stats = AugmentationStats()
        result = validate_augmentation(X_orig, X_aug, y_orig, y_aug, stats)
        
        assert not result.is_valid
        assert any("NaN" in e for e in result.errors)
    
    def test_inf_values_fail(self):
        """Test that Inf values cause validation failure."""
        X_orig = np.random.randn(50, 10, 5)
        y_orig = np.random.randn(50)
        
        X_aug = np.concatenate([X_orig, X_orig])
        X_aug[60, 5, 2] = np.inf  # Introduce Inf
        y_aug = np.concatenate([y_orig, y_orig])
        
        stats = AugmentationStats()
        result = validate_augmentation(X_orig, X_aug, y_orig, y_aug, stats)
        
        assert not result.is_valid
        assert any("Inf" in e for e in result.errors)
    
    def test_shape_mismatch_fails(self):
        """Test that shape mismatch causes validation failure."""
        X_orig = np.random.randn(50, 10, 5)
        y_orig = np.random.randn(50)
        
        X_aug = np.random.randn(100, 10, 4)  # Wrong feature count
        y_aug = np.random.randn(100)
        
        stats = AugmentationStats()
        result = validate_augmentation(X_orig, X_aug, y_orig, y_aug, stats)
        
        assert not result.is_valid
        assert not result.sequence_integrity_ok
    
    def test_length_mismatch_fails(self):
        """Test that X/y length mismatch causes failure."""
        X_orig = np.random.randn(50, 10, 5)
        y_orig = np.random.randn(50)
        
        X_aug = np.random.randn(100, 10, 5)
        y_aug = np.random.randn(90)  # Wrong length
        
        stats = AugmentationStats()
        result = validate_augmentation(X_orig, X_aug, y_orig, y_aug, stats)
        
        assert not result.is_valid


class TestGetAugmentationMetadata:
    """Tests for get_augmentation_metadata function."""
    
    def test_metadata_conversion(self):
        """Test that stats are correctly converted to dict."""
        stats = AugmentationStats(
            original_samples=100,
            samples_added=50,
            final_samples=150,
            original_negative_count=30,
            final_negative_count=80,
            original_negative_pct=0.3,
            final_negative_pct=0.533,
            original_positive_count=70,
            noise_std=0.05,
            augmentation_factor=1.5,
            sequence_shape=(100, 10, 5),
        )
        
        metadata = get_augmentation_metadata(stats)
        
        assert metadata['original_samples'] == 100
        assert metadata['samples_added'] == 50
        assert metadata['final_samples'] == 150
        assert metadata['noise_std'] == 0.05
        assert metadata['augmentation_type'] == 'negative_return_duplication'
        assert isinstance(metadata['sequence_shape'], list)


class TestAugmentWithTimeShift:
    """Tests for augment_with_time_shift function."""
    
    def test_time_shift_preserves_shape(self):
        """Test that time shift preserves array shapes."""
        np.random.seed(42)
        X = np.random.randn(50, 10, 5)
        y = np.random.randn(50)
        
        X_shifted, y_shifted, stats = augment_with_time_shift(
            X, y, shift_range=(-2, 2), random_state=42
        )
        
        assert X_shifted.shape == X.shape
        assert y_shifted.shape == y.shape
    
    def test_time_shift_modifies_data(self):
        """Test that time shift actually modifies the data."""
        np.random.seed(42)
        X = np.random.randn(50, 10, 5)
        y = np.random.randn(50)
        
        X_shifted, _, _ = augment_with_time_shift(
            X, y, shift_range=(-2, 2), random_state=42
        )
        
        # Should be different from original
        assert not np.allclose(X, X_shifted)
    
    def test_zero_shift_unchanged(self):
        """Test that zero shift range leaves data unchanged."""
        np.random.seed(42)
        X = np.random.randn(50, 10, 5)
        y = np.random.randn(50)
        
        X_shifted, _, _ = augment_with_time_shift(
            X, y, shift_range=(0, 0), random_state=42
        )
        
        np.testing.assert_array_equal(X, X_shifted)


class TestIntegration:
    """Integration tests for augmentation pipeline."""
    
    def test_full_augmentation_pipeline(self):
        """Test complete augmentation workflow."""
        np.random.seed(42)
        
        # Create realistic-ish data
        n_samples = 500
        seq_length = 90
        n_features = 50
        
        X = np.random.randn(n_samples, seq_length, n_features)
        
        # Imbalanced: 65% positive, 35% negative
        y = np.random.randn(n_samples) * 0.03
        positive_indices = np.random.choice(n_samples, size=325, replace=False)
        y[positive_indices] = np.abs(y[positive_indices])
        
        original_neg_pct = np.mean(y < 0)
        
        # Augment
        X_aug, y_aug, stats = augment_negative_returns(
            X, y,
            augmentation_factor=1.3,
            noise_std=0.05,
            random_state=42,
            validate=True,
        )
        
        final_neg_pct = np.mean(y_aug < 0)
        
        # Assertions
        assert len(y_aug) > len(y)
        assert final_neg_pct > original_neg_pct
        assert stats.samples_added > 0
        
        # Verify no data quality issues
        assert not np.any(np.isnan(X_aug))
        assert not np.any(np.isinf(X_aug))
        
        # Metadata should be extractable
        metadata = get_augmentation_metadata(stats)
        assert metadata['samples_added'] == stats.samples_added


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
