"""
Model Diagnostics Module for Regressor Health Checks.

This module provides diagnostic functions to detect model collapse,
distribution issues, and other health problems in regression models.
"""

from dataclasses import dataclass, field
from typing import Union, List, Dict, Any
import numpy as np
from scipy import stats


@dataclass
class ModelHealthReport:
    """
    Report containing model health diagnostics.

    Attributes:
        is_healthy: Whether the model passed all health checks.
        issues: List of identified issues with the model.
        statistics: Dictionary containing computed statistics.
        recommendations: List of recommended actions to address issues.
    """

    is_healthy: bool
    issues: List[str] = field(default_factory=list)
    statistics: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


def diagnose_regressor_health(
    y_pred: Union[np.ndarray, List[float]],
    y_true: Union[np.ndarray, List[float]],
    std_threshold: float = 0.005,
    same_sign_threshold: float = 0.95,
    range_threshold: float = 0.01,
    skewness_threshold: float = 2.0,
) -> ModelHealthReport:
    """
    Diagnose the health of a regression model by analyzing predictions and targets.

    This function performs comprehensive checks to detect:
    - Model collapse (predictions too similar)
    - Sign bias (too many predictions of the same sign)
    - Distribution mismatches between predictions and targets
    - Extreme skewness in predictions or targets

    Args:
        y_pred: Array of model predictions.
        y_true: Array of true target values.
        std_threshold: Minimum acceptable standard deviation for predictions.
            Values below this indicate model collapse. Default: 0.005.
        same_sign_threshold: Maximum acceptable proportion of predictions
            with the same sign. Default: 0.95 (95%).
        range_threshold: Minimum acceptable range for predictions.
            Default: 0.01.
        skewness_threshold: Maximum acceptable absolute skewness value.
            Default: 2.0.

    Returns:
        ModelHealthReport: A dataclass containing:
            - is_healthy: True if no critical issues detected
            - issues: List of identified problems
            - statistics: Dict with computed metrics
            - recommendations: List of suggested fixes

    Example:
        >>> import numpy as np
        >>> y_pred = np.random.normal(0, 0.1, 1000)
        >>> y_true = np.random.normal(0, 0.1, 1000)
        >>> report = diagnose_regressor_health(y_pred, y_true)
        >>> if not report.is_healthy:
        ...     print(f"Model issues: {report.issues}")
        ...     for rec in report.recommendations:
        ...         print(f"→ {rec}")
    """
    # Convert to numpy arrays
    y_pred = np.asarray(y_pred).flatten()
    y_true = np.asarray(y_true).flatten()

    issues: List[str] = []
    recommendations: List[str] = []

    # =========================================================================
    # 1. Compute prediction statistics
    # =========================================================================
    pred_mean = float(np.mean(y_pred))
    pred_std = float(np.std(y_pred))
    pred_min = float(np.min(y_pred))
    pred_max = float(np.max(y_pred))
    pred_range = pred_max - pred_min

    # Percentage of positive vs negative predictions
    n_positive = int(np.sum(y_pred > 0))
    n_negative = int(np.sum(y_pred < 0))
    n_zero = int(np.sum(y_pred == 0))
    n_total = len(y_pred)

    pct_positive = n_positive / n_total if n_total > 0 else 0.0
    pct_negative = n_negative / n_total if n_total > 0 else 0.0

    # Target distribution statistics
    target_mean = float(np.mean(y_true))
    target_std = float(np.std(y_true))
    target_min = float(np.min(y_true))
    target_max = float(np.max(y_true))
    target_range = target_max - target_min

    # Skewness calculations
    pred_skewness = float(stats.skew(y_pred))
    target_skewness = float(stats.skew(y_true))

    # =========================================================================
    # 2. Check for model collapse
    # =========================================================================

    # Flag if std < threshold (predictions too similar)
    if pred_std < std_threshold:
        issues.append(
            f"Model collapse detected: prediction std ({pred_std:.6f}) "
            f"< threshold ({std_threshold})"
        )
        recommendations.append(
            "Consider increasing model capacity or adjusting learning rate"
        )
        recommendations.append(
            "Check for gradient vanishing issues in deep networks"
        )

    # Flag if >95% predictions same sign
    max_same_sign_pct = max(pct_positive, pct_negative)
    if max_same_sign_pct > same_sign_threshold:
        dominant_sign = "positive" if pct_positive > pct_negative else "negative"
        issues.append(
            f"Sign bias detected: {max_same_sign_pct * 100:.1f}% of predictions "
            f"are {dominant_sign} (threshold: {same_sign_threshold * 100:.0f}%)"
        )
        recommendations.append(
            "Review target distribution for class imbalance"
        )
        recommendations.append(
            "Consider using balanced sampling or loss weighting"
        )

    # Flag if prediction range < threshold
    if pred_range < range_threshold:
        issues.append(
            f"Narrow prediction range: {pred_range:.6f} < threshold ({range_threshold})"
        )
        recommendations.append(
            "Model may be outputting near-constant values"
        )
        recommendations.append(
            "Check output layer activation and bias initialization"
        )

    # =========================================================================
    # 3. Validate target distribution
    # =========================================================================

    # Compare prediction distribution to target distribution using KS test
    ks_statistic, ks_pvalue = stats.ks_2samp(y_pred, y_true)

    if ks_pvalue < 0.05:
        issues.append(
            f"Distribution mismatch: KS test p-value={ks_pvalue:.4f} "
            f"(statistic={ks_statistic:.4f})"
        )
        recommendations.append(
            "Prediction distribution significantly differs from target distribution"
        )
        recommendations.append(
            "Consider distribution-aware loss functions or output calibration"
        )

    # Check for extreme skewness in predictions
    if abs(pred_skewness) > skewness_threshold:
        issues.append(
            f"Extreme prediction skewness: {pred_skewness:.4f} "
            f"(threshold: ±{skewness_threshold})"
        )
        recommendations.append(
            "Consider log-transforming targets or using quantile regression"
        )

    # Check for extreme skewness in targets (informational)
    target_skewness_issue = abs(target_skewness) > skewness_threshold

    # =========================================================================
    # 4. Build statistics dictionary
    # =========================================================================
    statistics = {
        "prediction": {
            "mean": pred_mean,
            "std": pred_std,
            "min": pred_min,
            "max": pred_max,
            "range": pred_range,
            "skewness": pred_skewness,
            "pct_positive": pct_positive * 100,
            "pct_negative": pct_negative * 100,
            "n_positive": n_positive,
            "n_negative": n_negative,
            "n_zero": n_zero,
            "n_total": n_total,
        },
        "target": {
            "mean": target_mean,
            "std": target_std,
            "min": target_min,
            "max": target_max,
            "range": target_range,
            "skewness": target_skewness,
            "has_extreme_skewness": target_skewness_issue,
        },
        "distribution_comparison": {
            "ks_statistic": float(ks_statistic),
            "ks_pvalue": float(ks_pvalue),
        },
        "thresholds_used": {
            "std_threshold": std_threshold,
            "same_sign_threshold": same_sign_threshold,
            "range_threshold": range_threshold,
            "skewness_threshold": skewness_threshold,
        },
    }

    # =========================================================================
    # 5. Determine overall health status
    # =========================================================================
    is_healthy = len(issues) == 0

    return ModelHealthReport(
        is_healthy=is_healthy,
        issues=issues,
        statistics=statistics,
        recommendations=recommendations,
    )


def format_health_report(report: ModelHealthReport) -> str:
    """
    Format a ModelHealthReport as a human-readable string.

    Args:
        report: The health report to format.

    Returns:
        A formatted string representation of the report.
    """
    lines = []
    lines.append("=" * 60)
    lines.append("MODEL HEALTH REPORT")
    lines.append("=" * 60)

    status = "✓ HEALTHY" if report.is_healthy else "✗ ISSUES DETECTED"
    lines.append(f"Status: {status}")
    lines.append("")

    if report.issues:
        lines.append("Issues:")
        for i, issue in enumerate(report.issues, 1):
            lines.append(f"  {i}. {issue}")
        lines.append("")

    if report.recommendations:
        lines.append("Recommendations:")
        for rec in report.recommendations:
            lines.append(f"  → {rec}")
        lines.append("")

    # Statistics summary
    lines.append("Statistics Summary:")
    pred_stats = report.statistics.get("prediction", {})
    target_stats = report.statistics.get("target", {})
    dist_stats = report.statistics.get("distribution_comparison", {})

    lines.append(f"  Predictions: mean={pred_stats.get('mean', 'N/A'):.4f}, "
                 f"std={pred_stats.get('std', 'N/A'):.4f}, "
                 f"range=[{pred_stats.get('min', 'N/A'):.4f}, {pred_stats.get('max', 'N/A'):.4f}]")
    lines.append(f"  Targets:     mean={target_stats.get('mean', 'N/A'):.4f}, "
                 f"std={target_stats.get('std', 'N/A'):.4f}, "
                 f"range=[{target_stats.get('min', 'N/A'):.4f}, {target_stats.get('max', 'N/A'):.4f}]")
    lines.append(f"  Sign distribution: {pred_stats.get('pct_positive', 0):.1f}% positive, "
                 f"{pred_stats.get('pct_negative', 0):.1f}% negative")
    lines.append(f"  KS test: statistic={dist_stats.get('ks_statistic', 'N/A'):.4f}, "
                 f"p-value={dist_stats.get('ks_pvalue', 'N/A'):.4f}")

    lines.append("=" * 60)

    return "\n".join(lines)
