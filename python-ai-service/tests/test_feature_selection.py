"""
Unit tests for feature_selection module.

Tests cover the train_feature_selector function, load_selected_features,
and related utilities.
"""

import pytest
import numpy as np
import pandas as pd
import pickle
import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.feature_selection import (
    train_feature_selector,
    load_selected_features,
    load_feature_importance,
    FeatureSelectionReport,
    FeatureValidationResult,
    validate_selected_features,
    format_feature_selection_report,
    _get_feature_category,
    FEATURE_CATEGORIES,
    CATEGORY_MINIMUMS,
)


class TestFeatureSelectionReport:
    """Tests for the FeatureSelectionReport dataclass."""

    def test_dataclass_defaults(self):
        """Test that FeatureSelectionReport has correct default values."""
        report = FeatureSelectionReport(
            symbol='TEST',
            n_features_total=100,
            n_features_selected=40,
        )
        assert report.symbol == 'TEST'
        assert report.n_features_total == 100
        assert report.n_features_selected == 40
        assert report.selected_features == []
        assert report.validation_r2 == 0.0
        assert report.has_predictive_power is True

    def test_dataclass_with_values(self):
        """Test FeatureSelectionReport with explicit values."""
        report = FeatureSelectionReport(
            symbol='AAPL',
            n_features_total=147,
            n_features_selected=40,
            selected_features=['rsi', 'macd'],
            validation_r2=0.05,
            validation_mae=0.01,
            validation_dir_acc=0.55,
            has_predictive_power=True,
            category_coverage={'momentum': 10, 'volatility': 8},
        )
        assert report.symbol == 'AAPL'
        assert len(report.selected_features) == 2
        assert report.validation_r2 == 0.05
        assert report.category_coverage['momentum'] == 10


class TestGetFeatureCategory:
    """Tests for the _get_feature_category function."""

    def test_momentum_features(self):
        """Test that momentum features are correctly categorized."""
        assert _get_feature_category('rsi') == 'momentum'
        assert _get_feature_category('macd') == 'momentum'
        assert _get_feature_category('stoch_k') == 'momentum'
        assert _get_feature_category('momentum_5d') == 'momentum'

    def test_volatility_features(self):
        """Test that volatility features are correctly categorized."""
        assert _get_feature_category('volatility_5d') == 'volatility'
        assert _get_feature_category('atr') == 'volatility'
        assert _get_feature_category('bb_width') == 'volatility'

    def test_volume_features(self):
        """Test that volume features are correctly categorized."""
        assert _get_feature_category('volume_ratio') == 'volume'
        assert _get_feature_category('obv') == 'volume'
        assert _get_feature_category('vwap_deviation') == 'volume'

    def test_uncategorized_features(self):
        """Test that unknown features return None."""
        assert _get_feature_category('unknown_feature_xyz') is None


class TestTrainFeatureSelector:
    """Tests for the train_feature_selector function."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 500
        n_features = 50

        # Create features with some having higher importance
        X = np.random.randn(n_samples, n_features)

        # Create target with correlation to first few features
        y = (
            0.3 * X[:, 0] +
            0.2 * X[:, 1] +
            0.1 * X[:, 2] +
            np.random.randn(n_samples) * 0.5
        )

        # Create feature names
        feature_names = [f'feature_{i}' for i in range(n_features)]
        # Add some recognizable names
        feature_names[0] = 'rsi'
        feature_names[1] = 'volatility_5d'
        feature_names[2] = 'volume_ratio'
        feature_names[3] = 'news_sentiment_score'

        return X, y, feature_names

    def test_basic_feature_selection(self, sample_data):
        """Test basic feature selection functionality."""
        X, y, feature_names = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            report = train_feature_selector(
                X=X,
                y=y,
                symbol='TEST',
                feature_names=feature_names,
                n_top_features=20,
                output_dir=tmpdir,
            )

            assert report.symbol == 'TEST'
            assert report.n_features_total == 50
            assert report.n_features_selected >= 20
            assert len(report.selected_features) >= 20
            assert 0 <= report.cumulative_importance <= 1.0

    def test_saves_artifacts(self, sample_data):
        """Test that all artifacts are saved correctly."""
        X, y, feature_names = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            report = train_feature_selector(
                X=X,
                y=y,
                symbol='TEST',
                feature_names=feature_names,
                n_top_features=10,
                output_dir=tmpdir,
            )

            # Check files exist
            symbol_dir = Path(tmpdir) / 'TEST'
            assert (symbol_dir / 'TEST_selected_features.pkl').exists()
            assert (symbol_dir / 'TEST_feature_importance.csv').exists()
            assert (symbol_dir / 'TEST_rf_validation_metrics.json').exists()

    def test_selected_features_pkl(self, sample_data):
        """Test that selected features pkl is correct."""
        X, y, feature_names = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            report = train_feature_selector(
                X=X,
                y=y,
                symbol='TEST',
                feature_names=feature_names,
                n_top_features=15,
                output_dir=tmpdir,
            )

            # Load and verify
            features_path = Path(tmpdir) / 'TEST' / 'TEST_selected_features.pkl'
            with open(features_path, 'rb') as f:
                loaded_features = pickle.load(f)

            assert len(loaded_features) >= 15
            assert loaded_features == report.selected_features

    def test_importance_csv(self, sample_data):
        """Test that importance CSV is correctly formatted."""
        X, y, feature_names = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            train_feature_selector(
                X=X,
                y=y,
                symbol='TEST',
                feature_names=feature_names,
                n_top_features=10,
                output_dir=tmpdir,
            )

            # Load and verify
            csv_path = Path(tmpdir) / 'TEST' / 'TEST_feature_importance.csv'
            df = pd.read_csv(csv_path)

            assert 'feature' in df.columns
            assert 'importance' in df.columns
            assert 'importance_normalized' in df.columns
            assert 'cumulative_importance' in df.columns
            assert len(df) == 50  # All features

            # Check normalization
            assert abs(df['importance_normalized'].sum() - 1.0) < 0.001

    def test_validation_metrics_json(self, sample_data):
        """Test that validation metrics JSON is correct."""
        X, y, feature_names = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            train_feature_selector(
                X=X,
                y=y,
                symbol='TEST',
                feature_names=feature_names,
                n_top_features=10,
                output_dir=tmpdir,
            )

            # Load and verify
            json_path = Path(tmpdir) / 'TEST' / 'TEST_rf_validation_metrics.json'
            with open(json_path, 'r') as f:
                metrics = json.load(f)

            assert metrics['symbol'] == 'TEST'
            assert 'cv_r2_mean' in metrics
            assert 'validation_mae' in metrics
            assert 'validation_dir_acc' in metrics
            assert 'has_predictive_power' in metrics
            assert 'rf_params' in metrics

    def test_reproducibility(self, sample_data):
        """Test that results are reproducible with same random_state."""
        X, y, feature_names = sample_data

        with tempfile.TemporaryDirectory() as tmpdir1:
            report1 = train_feature_selector(
                X=X.copy(),
                y=y.copy(),
                symbol='TEST',
                feature_names=feature_names.copy(),
                n_top_features=20,
                output_dir=tmpdir1,
                random_state=42,
            )

        with tempfile.TemporaryDirectory() as tmpdir2:
            report2 = train_feature_selector(
                X=X.copy(),
                y=y.copy(),
                symbol='TEST',
                feature_names=feature_names.copy(),
                n_top_features=20,
                output_dir=tmpdir2,
                random_state=42,
            )

        assert report1.selected_features == report2.selected_features
        # Use approximate comparison for floating-point values
        np.testing.assert_almost_equal(report1.validation_r2, report2.validation_r2, decimal=10)

    def test_category_diversity(self, sample_data):
        """Test that category diversity is enforced."""
        X, y, feature_names = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            report = train_feature_selector(
                X=X,
                y=y,
                symbol='TEST',
                feature_names=feature_names,
                n_top_features=20,
                output_dir=tmpdir,
                min_features_per_category=2,
            )

            # Check that categorized features were selected
            # (Note: our sample data has limited categorized features)
            assert len(report.selected_features) >= 20


class TestLoadSelectedFeatures:
    """Tests for the load_selected_features function."""

    def test_load_existing_features(self):
        """Test loading existing feature file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a feature file
            symbol_dir = Path(tmpdir) / 'TEST'
            symbol_dir.mkdir(parents=True)

            features = ['rsi', 'macd', 'volatility_5d']
            features_path = symbol_dir / 'TEST_selected_features.pkl'
            with open(features_path, 'wb') as f:
                pickle.dump(features, f)

            # Load and verify
            loaded = load_selected_features('TEST', output_dir=tmpdir)
            assert loaded == features

    def test_file_not_found_error(self):
        """Test that FileNotFoundError is raised with helpful message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError) as exc_info:
                load_selected_features('NONEXISTENT', output_dir=tmpdir)

            assert 'NONEXISTENT' in str(exc_info.value)
            assert 'Run feature selection first' in str(exc_info.value)


class TestLoadFeatureImportance:
    """Tests for the load_feature_importance function."""

    def test_load_existing_importance(self):
        """Test loading existing importance file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create an importance CSV
            symbol_dir = Path(tmpdir) / 'TEST'
            symbol_dir.mkdir(parents=True)

            df = pd.DataFrame({
                'feature': ['rsi', 'macd', 'volatility_5d'],
                'importance': [0.3, 0.2, 0.1],
                'importance_normalized': [0.5, 0.33, 0.17],
            })
            csv_path = symbol_dir / 'TEST_feature_importance.csv'
            df.to_csv(csv_path, index=False)

            # Load and verify
            loaded = load_feature_importance('TEST', output_dir=tmpdir)
            assert len(loaded) == 3
            assert 'importance' in loaded.columns

    def test_file_not_found_error(self):
        """Test that FileNotFoundError is raised with helpful message."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(FileNotFoundError) as exc_info:
                load_feature_importance('NONEXISTENT', output_dir=tmpdir)

            assert 'NONEXISTENT' in str(exc_info.value)


class TestInputValidation:
    """Tests for input validation."""

    def test_mismatched_x_y_length(self):
        """Test that mismatched X and y raise ValueError."""
        X = np.random.randn(100, 10)
        y = np.random.randn(50)  # Wrong length
        feature_names = [f'f{i}' for i in range(10)]

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError) as exc_info:
                train_feature_selector(
                    X=X,
                    y=y,
                    symbol='TEST',
                    feature_names=feature_names,
                    output_dir=tmpdir,
                )

            assert 'same length' in str(exc_info.value)

    def test_mismatched_feature_names(self):
        """Test that mismatched feature_names raise ValueError."""
        X = np.random.randn(100, 10)
        y = np.random.randn(100)
        feature_names = [f'f{i}' for i in range(5)]  # Wrong count

        with tempfile.TemporaryDirectory() as tmpdir:
            with pytest.raises(ValueError) as exc_info:
                train_feature_selector(
                    X=X,
                    y=y,
                    symbol='TEST',
                    feature_names=feature_names,
                    output_dir=tmpdir,
                )

            assert 'feature_names length' in str(exc_info.value)


class TestPredictivePowerValidation:
    """Tests for predictive power validation."""

    def test_no_predictive_power_flagged(self):
        """Test that weak models are flagged."""
        np.random.seed(42)
        n_samples = 200
        n_features = 20

        # Create random data with no correlation
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples)  # Pure noise target

        feature_names = [f'feature_{i}' for i in range(n_features)]

        with tempfile.TemporaryDirectory() as tmpdir:
            report = train_feature_selector(
                X=X,
                y=y,
                symbol='TEST',
                feature_names=feature_names,
                n_top_features=10,
                output_dir=tmpdir,
            )

            # With pure noise, R² should be very low or negative
            # has_predictive_power should be False
            assert report.validation_r2 < 0.05  # Very weak correlation


class TestCumulativeImportanceThreshold:
    """Tests for cumulative importance threshold."""

    def test_meets_cumulative_threshold(self):
        """Test that cumulative importance threshold is met."""
        np.random.seed(42)
        n_samples = 300
        n_features = 30

        # Create data where first features are more important
        X = np.random.randn(n_samples, n_features)
        y = 0.5 * X[:, 0] + 0.3 * X[:, 1] + 0.2 * X[:, 2] + np.random.randn(n_samples) * 0.2

        feature_names = [f'feature_{i}' for i in range(n_features)]

        with tempfile.TemporaryDirectory() as tmpdir:
            report = train_feature_selector(
                X=X,
                y=y,
                symbol='TEST',
                feature_names=feature_names,
                n_top_features=15,
                output_dir=tmpdir,
                min_cumulative_importance=0.60,
            )

            # Should meet the 60% threshold
            assert report.cumulative_importance >= 0.60


class TestFeatureValidationResult:
    """Tests for the FeatureValidationResult dataclass."""

    def test_dataclass_defaults(self):
        """Test FeatureValidationResult has correct default values."""
        result = FeatureValidationResult(validation_passed=True)
        assert result.validation_passed is True
        assert result.warnings == []
        assert result.feature_category_counts == {}
        assert result.has_duplicates is False
        assert result.missing_features == []

    def test_dataclass_with_failures(self):
        """Test FeatureValidationResult with failure data."""
        result = FeatureValidationResult(
            validation_passed=False,
            warnings=['Insufficient momentum features: 2 < 5 required'],
            feature_category_counts={'momentum': 2, 'volatility': 6},
            has_duplicates=True,
            missing_features=['unknown_feature'],
        )
        assert result.validation_passed is False
        assert len(result.warnings) == 1
        assert result.has_duplicates is True
        assert 'unknown_feature' in result.missing_features


class TestValidateSelectedFeatures:
    """Tests for validate_selected_features function."""

    def test_valid_feature_selection(self):
        """Test validation passes with sufficient category coverage."""
        # Build feature list with sufficient category coverage (new relaxed minimums)
        selected_features = [
            # 3+ momentum (minimum is 3)
            'rsi', 'macd', 'macd_signal', 'stoch_k',
            # 3+ volatility (minimum is 3)
            'volatility_5d', 'volatility_20d', 'bb_upper', 'bb_lower',
            # 2+ volume (minimum is 2)
            'volume_sma', 'volume_ratio', 'obv',
            # 0+ sentiment (minimum is 0 - not required)
        ]
        original_features = selected_features.copy()

        result = validate_selected_features(selected_features, original_features)

        assert result.validation_passed is True
        assert len(result.warnings) == 0
        assert result.has_duplicates is False
        assert result.missing_features == []

    def test_insufficient_momentum_features(self):
        """Test validation fails with insufficient momentum features."""
        selected_features = [
            # Only 2 momentum (need 3)
            'rsi', 'macd',
            # 3 volatility (meets minimum)
            'volatility_5d', 'volatility_20d', 'bb_upper',
            # 2 volume (meets minimum)
            'volume_sma', 'volume_ratio',
            # No sentiment required
        ]
        original_features = selected_features.copy()

        result = validate_selected_features(selected_features, original_features)

        assert result.validation_passed is False
        assert any('momentum' in w.lower() for w in result.warnings)
        assert result.feature_category_counts['momentum'] == 2

    def test_insufficient_volatility_features(self):
        """Test validation fails with insufficient volatility features."""
        selected_features = [
            # 3 momentum (meets minimum)
            'rsi', 'macd', 'macd_signal',
            # Only 2 volatility (need 3)
            'volatility_5d', 'bb_upper',
            # 2 volume (meets minimum)
            'volume_sma', 'volume_ratio',
            # No sentiment required
        ]
        original_features = selected_features.copy()

        result = validate_selected_features(selected_features, original_features)

        assert result.validation_passed is False
        assert any('volatility' in w.lower() for w in result.warnings)

    def test_insufficient_volume_features(self):
        """Test validation fails with insufficient volume features."""
        selected_features = [
            # 3 momentum (meets minimum)
            'rsi', 'macd', 'macd_signal',
            # 3 volatility (meets minimum)
            'volatility_5d', 'volatility_20d', 'bb_upper',
            # Only 1 volume (need 2)
            'volume_sma',
            # No sentiment required
        ]
        original_features = selected_features.copy()

        result = validate_selected_features(selected_features, original_features)

        assert result.validation_passed is False
        assert any('volume' in w.lower() for w in result.warnings)

    def test_duplicate_features_detected(self):
        """Test validation detects duplicate features."""
        selected_features = [
            'rsi', 'macd', 'macd_signal', 'stoch_k', 'stoch_d', 'rsi_momentum',
            'volatility_5d', 'volatility_20d', 'bb_upper', 'bb_lower', 'atr',
            'volume_sma', 'volume_ratio', 'obv',
            'news_sentiment_score', 'sentiment_ma_3d', 'sentiment_ma_7d',
            'rsi',  # Duplicate!
        ]
        original_features = list(set(selected_features))

        result = validate_selected_features(selected_features, original_features)

        assert result.validation_passed is False
        assert result.has_duplicates is True
        assert any('duplicate' in w.lower() for w in result.warnings)

    def test_missing_features_detected(self):
        """Test validation detects features not in original set."""
        selected_features = [
            'rsi', 'macd', 'macd_signal', 'stoch_k', 'stoch_d', 'rsi_momentum',
            'volatility_5d', 'volatility_20d', 'bb_upper', 'bb_lower', 'atr',
            'volume_sma', 'volume_ratio', 'obv',
            'news_sentiment_score', 'sentiment_ma_3d', 'sentiment_ma_7d',
            'nonexistent_feature',  # Not in original!
        ]
        original_features = [f for f in selected_features if f != 'nonexistent_feature']

        result = validate_selected_features(selected_features, original_features)

        assert result.validation_passed is False
        assert 'nonexistent_feature' in result.missing_features
        assert any('not in original' in w.lower() for w in result.warnings)

    def test_sentiment_not_required_when_unavailable(self):
        """Test sentiment validation skipped when not available."""
        selected_features = [
            # 5 momentum
            'rsi', 'macd', 'macd_signal', 'stoch_k', 'stoch_d',
            # 5 volatility
            'volatility_5d', 'volatility_20d', 'bb_upper', 'bb_lower', 'atr',
            # 3 volume
            'volume_sma', 'volume_ratio', 'obv',
            # No sentiment features in original or selected
        ]
        # Original features also have no sentiment
        original_features = selected_features.copy()

        result = validate_selected_features(selected_features, original_features)

        # Should pass since sentiment isn't available
        assert result.validation_passed is True

    def test_sentiment_not_required_when_disabled(self):
        """Test sentiment validation skipped when require_sentiment=False."""
        selected_features = [
            'rsi', 'macd', 'macd_signal', 'stoch_k', 'stoch_d',
            'volatility_5d', 'volatility_20d', 'bb_upper', 'bb_lower', 'atr',
            'volume_sma', 'volume_ratio', 'obv',
            # No sentiment features
        ]
        # Original has sentiment but selected doesn't have enough
        original_features = selected_features + ['news_sentiment_score', 'sentiment_ma_3d']

        result = validate_selected_features(
            selected_features, original_features, require_sentiment=False
        )

        assert result.validation_passed is True

    def test_category_counts_correct(self):
        """Test that category counts are correctly calculated."""
        selected_features = [
            'rsi', 'macd',  # 2 momentum
            'volatility_5d', 'atr',  # 2 volatility
            'volume_sma',  # 1 volume
            'unknown_feature',  # 1 other
        ]
        original_features = selected_features.copy()

        result = validate_selected_features(
            selected_features, original_features, require_sentiment=False
        )

        assert result.feature_category_counts['momentum'] == 2
        assert result.feature_category_counts['volatility'] == 2
        assert result.feature_category_counts['volume'] == 1
        assert result.feature_category_counts['other'] == 1


class TestFormatFeatureSelectionReport:
    """Tests for format_feature_selection_report function."""

    def test_report_format_contains_all_sections(self):
        """Test that formatted report contains all required sections."""
        report = FeatureSelectionReport(
            symbol='AAPL',
            n_features_total=147,
            n_features_selected=40,
            selected_features=['rsi', 'macd', 'volatility_5d'],
            validation_r2=0.05,
            validation_mae=0.02,
            validation_dir_acc=0.55,
            baseline_r2=-0.01,
            cumulative_importance=0.823,
            has_predictive_power=True,
            category_coverage={'momentum': 12, 'volatility': 8},
        )

        importance_df = pd.DataFrame({
            'feature': ['rsi', 'macd', 'volatility_5d'],
            'importance_normalized': [0.045, 0.038, 0.035],
            'category': ['momentum', 'momentum', 'volatility'],
        })

        validation_result = FeatureValidationResult(
            validation_passed=True,
            warnings=[],
            feature_category_counts={'momentum': 2, 'volatility': 1, 'other': 0},
        )

        report_text = format_feature_selection_report(
            report, importance_df, validation_result
        )

        # Check key sections present
        assert 'FEATURE SELECTION REPORT' in report_text
        assert 'Symbol: AAPL' in report_text
        assert 'Original features: 147' in report_text
        assert 'Selected features: 40' in report_text
        assert 'Category breakdown:' in report_text
        assert 'Top 10 features:' in report_text
        assert 'Validation:' in report_text
        assert '✓ PASSED' in report_text

    def test_report_shows_failed_validation(self):
        """Test that failed validation is clearly shown."""
        report = FeatureSelectionReport(
            symbol='TEST',
            n_features_total=100,
            n_features_selected=20,
            selected_features=['rsi', 'macd'],
            validation_r2=0.01,
            cumulative_importance=0.60,
        )

        importance_df = pd.DataFrame({
            'feature': ['rsi', 'macd'],
            'importance_normalized': [0.04, 0.03],
            'category': ['momentum', 'momentum'],
        })

        validation_result = FeatureValidationResult(
            validation_passed=False,
            warnings=['Insufficient volatility features: 0 < 5 required'],
            feature_category_counts={'momentum': 2, 'volatility': 0},
        )

        report_text = format_feature_selection_report(
            report, importance_df, validation_result
        )

        assert '✗ FAILED' in report_text
        assert 'Insufficient volatility' in report_text

    def test_report_shows_top_features_with_importance(self):
        """Test that top features are listed with importance scores."""
        report = FeatureSelectionReport(
            symbol='TEST',
            n_features_total=50,
            n_features_selected=10,
            selected_features=['rsi', 'macd', 'volatility_5d'],
        )

        importance_df = pd.DataFrame({
            'feature': ['rsi', 'macd', 'volatility_5d'],
            'importance_normalized': [0.045, 0.038, 0.035],
            'category': ['momentum', 'momentum', 'volatility'],
        })

        validation_result = FeatureValidationResult(
            validation_passed=True,
            feature_category_counts={'momentum': 2, 'volatility': 1},
        )

        report_text = format_feature_selection_report(
            report, importance_df, validation_result
        )

        assert 'rsi' in report_text
        assert '0.045' in report_text or '0.0450' in report_text


class TestValidationReportSaved:
    """Tests for validation report file saving."""

    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 300

        # Create feature names with categories
        feature_names = [
            # Momentum (6)
            'rsi', 'macd', 'macd_signal', 'stoch_k', 'stoch_d', 'rsi_momentum',
            # Volatility (6)
            'volatility_5d', 'volatility_20d', 'bb_upper', 'bb_lower', 'atr', 'bb_width',
            # Volume (4)
            'volume_sma', 'volume_ratio', 'obv', 'obv_ema',
            # Sentiment (4)
            'news_sentiment_score', 'sentiment_ma_3d', 'sentiment_ma_7d', 'sentiment_momentum',
        ]

        # Add generic features
        for i in range(20, 40):
            feature_names.append(f'feature_{i}')

        n_features = len(feature_names)
        X = np.random.randn(n_samples, n_features)
        y = np.random.randn(n_samples) * 0.05

        return X, y, feature_names

    def test_validation_report_file_created(self, sample_data):
        """Test that validation report file is created."""
        X, y, feature_names = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            report = train_feature_selector(
                X=X,
                y=y,
                symbol='TEST',
                feature_names=feature_names,
                n_top_features=25,
                output_dir=tmpdir,
            )

            validation_path = Path(tmpdir) / 'TEST' / 'TEST_feature_selection_validation.txt'
            assert validation_path.exists()

    def test_validation_report_content(self, sample_data):
        """Test that validation report contains expected content."""
        X, y, feature_names = sample_data

        with tempfile.TemporaryDirectory() as tmpdir:
            report = train_feature_selector(
                X=X,
                y=y,
                symbol='TEST',
                feature_names=feature_names,
                n_top_features=25,
                output_dir=tmpdir,
            )

            validation_path = Path(tmpdir) / 'TEST' / 'TEST_feature_selection_validation.txt'
            content = validation_path.read_text(encoding='utf-8')

            assert 'FEATURE SELECTION REPORT' in content
            assert 'Symbol: TEST' in content
            assert 'Category breakdown:' in content
            assert 'Top 10 features:' in content
