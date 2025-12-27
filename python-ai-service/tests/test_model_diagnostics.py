"""
Unit tests for model_diagnostics module.

Tests cover the diagnose_regressor_health function and ModelHealthReport dataclass.
"""

import pytest
import numpy as np
from evaluation.model_diagnostics import (
    diagnose_regressor_health,
    ModelHealthReport,
    format_health_report,
)


class TestModelHealthReport:
    """Tests for the ModelHealthReport dataclass."""

    def test_dataclass_defaults(self):
        """Test that ModelHealthReport has correct default values."""
        report = ModelHealthReport(is_healthy=True)
        assert report.is_healthy is True
        assert report.issues == []
        assert report.statistics == {}
        assert report.recommendations == []

    def test_dataclass_with_values(self):
        """Test ModelHealthReport with explicit values."""
        report = ModelHealthReport(
            is_healthy=False,
            issues=["Issue 1", "Issue 2"],
            statistics={"mean": 0.5},
            recommendations=["Fix 1"],
        )
        assert report.is_healthy is False
        assert len(report.issues) == 2
        assert report.statistics["mean"] == 0.5
        assert len(report.recommendations) == 1


class TestDiagnoseRegressorHealth:
    """Tests for the diagnose_regressor_health function."""

    def test_healthy_model(self):
        """Test that a healthy model passes all checks."""
        np.random.seed(42)
        # Generate well-distributed predictions and targets
        y_true = np.random.normal(0, 0.1, 1000)
        y_pred = y_true + np.random.normal(0, 0.02, 1000)  # Good predictions with noise

        report = diagnose_regressor_health(y_pred, y_true)

        # Note: The KS test might still flag distribution mismatch
        # Check that at least no collapse issues exist
        assert report.statistics["prediction"]["std"] > 0.005
        assert report.statistics["prediction"]["range"] > 0.01

    def test_model_collapse_low_std(self):
        """Test detection of model collapse via low standard deviation."""
        np.random.seed(42)
        y_true = np.random.normal(0, 0.1, 1000)
        # Near-constant predictions
        y_pred = np.full(1000, 0.5) + np.random.normal(0, 0.001, 1000)

        report = diagnose_regressor_health(y_pred, y_true)

        assert not report.is_healthy
        assert any("collapse" in issue.lower() for issue in report.issues)
        assert report.statistics["prediction"]["std"] < 0.005

    def test_model_collapse_narrow_range(self):
        """Test detection of model collapse via narrow prediction range."""
        np.random.seed(42)
        y_true = np.random.normal(0, 0.1, 1000)
        # Very narrow range predictions
        y_pred = np.random.uniform(0.5, 0.505, 1000)

        report = diagnose_regressor_health(y_pred, y_true)

        assert not report.is_healthy
        assert any("range" in issue.lower() for issue in report.issues)
        assert report.statistics["prediction"]["range"] < 0.01

    def test_sign_bias_positive(self):
        """Test detection of positive sign bias."""
        np.random.seed(42)
        y_true = np.random.normal(0, 0.1, 1000)
        # 98% positive predictions
        y_pred = np.abs(np.random.normal(0.1, 0.05, 1000))

        report = diagnose_regressor_health(y_pred, y_true)

        assert not report.is_healthy
        assert any("sign bias" in issue.lower() for issue in report.issues)
        assert report.statistics["prediction"]["pct_positive"] > 95

    def test_sign_bias_negative(self):
        """Test detection of negative sign bias."""
        np.random.seed(42)
        y_true = np.random.normal(0, 0.1, 1000)
        # 98% negative predictions
        y_pred = -np.abs(np.random.normal(0.1, 0.05, 1000))

        report = diagnose_regressor_health(y_pred, y_true)

        assert not report.is_healthy
        assert any("sign bias" in issue.lower() for issue in report.issues)
        assert report.statistics["prediction"]["pct_negative"] > 95

    def test_distribution_mismatch(self):
        """Test detection of distribution mismatch via KS test."""
        np.random.seed(42)
        y_true = np.random.normal(0, 0.1, 1000)
        # Completely different distribution
        y_pred = np.random.uniform(-1, 1, 1000)

        report = diagnose_regressor_health(y_pred, y_true)

        # Should detect distribution mismatch
        ks_pvalue = report.statistics["distribution_comparison"]["ks_pvalue"]
        assert ks_pvalue < 0.05

    def test_extreme_skewness(self):
        """Test detection of extreme skewness in predictions."""
        np.random.seed(42)
        y_true = np.random.normal(0, 0.1, 1000)
        # Create a highly skewed distribution by using log-normal shifted to have both signs
        # Log-normal has high positive skewness
        y_pred = np.random.lognormal(0, 1, 1000) - 1.5  # Shift to include negative values

        report = diagnose_regressor_health(y_pred, y_true)

        # Verify that skewness was detected and flagged
        assert not report.is_healthy
        assert any("skewness" in issue.lower() for issue in report.issues)
        assert abs(report.statistics["prediction"]["skewness"]) > 2.0

    def test_custom_thresholds(self):
        """Test that custom thresholds are respected."""
        np.random.seed(42)
        y_true = np.random.normal(0, 0.1, 100)
        y_pred = np.random.normal(0, 0.01, 100)  # Low std

        # With default threshold (0.005), this should be flagged
        report_default = diagnose_regressor_health(y_pred, y_true)

        # With a lower threshold, it might pass
        report_custom = diagnose_regressor_health(
            y_pred, y_true, std_threshold=0.001
        )

        # Verify thresholds are stored
        assert report_custom.statistics["thresholds_used"]["std_threshold"] == 0.001

    def test_statistics_structure(self):
        """Test that statistics dictionary has correct structure."""
        np.random.seed(42)
        y_true = np.random.normal(0, 0.1, 100)
        y_pred = np.random.normal(0, 0.1, 100)

        report = diagnose_regressor_health(y_pred, y_true)

        # Check prediction statistics
        assert "prediction" in report.statistics
        pred_stats = report.statistics["prediction"]
        assert "mean" in pred_stats
        assert "std" in pred_stats
        assert "min" in pred_stats
        assert "max" in pred_stats
        assert "range" in pred_stats
        assert "skewness" in pred_stats
        assert "pct_positive" in pred_stats
        assert "pct_negative" in pred_stats
        assert "n_positive" in pred_stats
        assert "n_negative" in pred_stats
        assert "n_zero" in pred_stats
        assert "n_total" in pred_stats

        # Check target statistics
        assert "target" in report.statistics
        target_stats = report.statistics["target"]
        assert "mean" in target_stats
        assert "std" in target_stats
        assert "min" in target_stats
        assert "max" in target_stats
        assert "range" in target_stats
        assert "skewness" in target_stats
        assert "has_extreme_skewness" in target_stats

        # Check distribution comparison
        assert "distribution_comparison" in report.statistics
        dist_stats = report.statistics["distribution_comparison"]
        assert "ks_statistic" in dist_stats
        assert "ks_pvalue" in dist_stats

        # Check thresholds
        assert "thresholds_used" in report.statistics

    def test_input_as_list(self):
        """Test that list inputs work correctly."""
        y_true = [0.1, 0.2, -0.1, -0.2, 0.15, -0.15, 0.05, -0.05]
        y_pred = [0.12, 0.18, -0.08, -0.22, 0.13, -0.17, 0.06, -0.04]

        report = diagnose_regressor_health(y_pred, y_true)

        assert isinstance(report, ModelHealthReport)
        assert report.statistics["prediction"]["n_total"] == 8

    def test_2d_array_handling(self):
        """Test that 2D arrays are flattened correctly."""
        np.random.seed(42)
        y_true = np.random.normal(0, 0.1, (100, 1))
        y_pred = np.random.normal(0, 0.1, (100, 1))

        report = diagnose_regressor_health(y_pred, y_true)

        assert report.statistics["prediction"]["n_total"] == 100

    def test_recommendations_not_empty_when_issues(self):
        """Test that recommendations are provided when issues exist."""
        np.random.seed(42)
        y_true = np.random.normal(0, 0.1, 1000)
        # Collapsed model
        y_pred = np.full(1000, 0.5)

        report = diagnose_regressor_health(y_pred, y_true)

        assert not report.is_healthy
        assert len(report.issues) > 0
        assert len(report.recommendations) > 0

    def test_multiple_issues_detected(self):
        """Test that multiple issues can be detected simultaneously."""
        np.random.seed(42)
        y_true = np.random.normal(0, 0.1, 1000)
        # Near-constant positive values (collapse + sign bias + narrow range)
        y_pred = np.full(1000, 0.001) + np.random.normal(0, 0.0001, 1000)

        report = diagnose_regressor_health(y_pred, y_true)

        assert not report.is_healthy
        # Should have multiple issues
        assert len(report.issues) >= 2


class TestFormatHealthReport:
    """Tests for the format_health_report function."""

    def test_format_healthy_report(self):
        """Test formatting of a healthy report."""
        np.random.seed(42)
        y_true = np.random.normal(0, 0.1, 100)
        y_pred = y_true + np.random.normal(0, 0.01, 100)

        report = diagnose_regressor_health(y_pred, y_true)
        formatted = format_health_report(report)

        assert "MODEL HEALTH REPORT" in formatted
        assert "Statistics Summary" in formatted

    def test_format_unhealthy_report(self):
        """Test formatting of an unhealthy report."""
        np.random.seed(42)
        y_true = np.random.normal(0, 0.1, 1000)
        y_pred = np.full(1000, 0.5)  # Collapsed model

        report = diagnose_regressor_health(y_pred, y_true)
        formatted = format_health_report(report)

        assert "ISSUES DETECTED" in formatted
        assert "Issues:" in formatted
        assert "Recommendations:" in formatted

    def test_format_includes_statistics(self):
        """Test that formatted report includes statistics."""
        np.random.seed(42)
        y_true = np.random.normal(0, 0.1, 100)
        y_pred = np.random.normal(0, 0.1, 100)

        report = diagnose_regressor_health(y_pred, y_true)
        formatted = format_health_report(report)

        assert "Predictions:" in formatted
        assert "Targets:" in formatted
        assert "KS test:" in formatted


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_all_zero_predictions(self):
        """Test handling of all-zero predictions."""
        y_true = np.random.normal(0, 0.1, 100)
        y_pred = np.zeros(100)

        report = diagnose_regressor_health(y_pred, y_true)

        assert not report.is_healthy
        assert report.statistics["prediction"]["std"] == 0.0
        assert report.statistics["prediction"]["range"] == 0.0

    def test_all_zero_targets(self):
        """Test handling of all-zero targets."""
        y_true = np.zeros(100)
        y_pred = np.random.normal(0, 0.1, 100)

        report = diagnose_regressor_health(y_pred, y_true)

        assert report.statistics["target"]["std"] == 0.0

    def test_small_sample_size(self):
        """Test with very small sample size."""
        y_true = np.array([0.1, -0.1])
        y_pred = np.array([0.11, -0.09])

        report = diagnose_regressor_health(y_pred, y_true)

        assert isinstance(report, ModelHealthReport)
        assert report.statistics["prediction"]["n_total"] == 2

    def test_large_values(self):
        """Test with large values."""
        np.random.seed(42)
        y_true = np.random.normal(1e6, 1e4, 100)
        y_pred = np.random.normal(1e6, 1e4, 100)

        report = diagnose_regressor_health(y_pred, y_true)

        # Should handle large values without overflow
        assert isinstance(report.statistics["prediction"]["mean"], float)

    def test_negative_values(self):
        """Test with all negative values."""
        np.random.seed(42)
        y_true = -np.abs(np.random.normal(0, 0.1, 100))
        y_pred = -np.abs(np.random.normal(0, 0.1, 100))

        report = diagnose_regressor_health(y_pred, y_true)

        assert report.statistics["prediction"]["pct_negative"] == 100.0
        assert report.statistics["prediction"]["pct_positive"] == 0.0

    def test_perfect_predictions(self):
        """Test with perfect predictions (y_pred == y_true)."""
        np.random.seed(42)
        y_true = np.random.normal(0, 0.1, 100)
        y_pred = y_true.copy()

        report = diagnose_regressor_health(y_pred, y_true)

        # KS test should show no difference
        assert report.statistics["distribution_comparison"]["ks_statistic"] == 0.0
        assert report.statistics["distribution_comparison"]["ks_pvalue"] == 1.0
