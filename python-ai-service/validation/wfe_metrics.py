"""
Walk Forward Efficiency (WFE) and Related Validation Metrics

Nuclear Redesign: Comprehensive metrics for validating model robustness.

Metrics Included:
- Walk Forward Efficiency (WFE)
- Overfitting Ratio
- Consistency Score
- Regime Stability
- Sharpe Ratio calculation

Author: AI-Stocks Nuclear Redesign
Date: December 2025
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class ValidationMetrics:
    """Comprehensive validation metrics for a model.

    Attributes:
        wfe: Walk Forward Efficiency (%)
        overfitting_ratio: Train/Test performance ratio (>1 = overfitting)
        consistency_score: Consistency across folds (0-1, higher = better)
        direction_accuracy: Directional accuracy on test set
        sharpe_ratio: Annualized Sharpe ratio on test set
        information_coefficient: Correlation between predictions and actuals
        prediction_variance: Variance of predictions (collapse detection)
        sign_balance: Fraction of predictions that are positive (should be ~0.5)
    """
    wfe: float
    overfitting_ratio: float
    consistency_score: float
    direction_accuracy: float
    sharpe_ratio: float = 0.0
    information_coefficient: float = 0.0
    prediction_variance: float = 0.0
    sign_balance: float = 0.5

    @property
    def is_robust(self) -> bool:
        """Check if model passes robustness thresholds."""
        return (
            self.wfe >= 50.0 and
            self.overfitting_ratio < 1.5 and
            self.consistency_score >= 0.6 and
            self.prediction_variance > 0.001 and
            0.3 <= self.sign_balance <= 0.7
        )

    @property
    def recommendation(self) -> str:
        """Generate human-readable recommendation."""
        issues = []

        if self.wfe < 40:
            issues.append(f"Low WFE ({self.wfe:.1f}%): significant overfitting")
        elif self.wfe < 50:
            issues.append(f"Marginal WFE ({self.wfe:.1f}%): some overfitting")

        if self.overfitting_ratio > 2.0:
            issues.append(f"High overfitting ratio ({self.overfitting_ratio:.2f})")
        elif self.overfitting_ratio > 1.5:
            issues.append(f"Moderate overfitting ratio ({self.overfitting_ratio:.2f})")

        if self.consistency_score < 0.5:
            issues.append(f"Low consistency ({self.consistency_score:.2f})")

        if self.prediction_variance < 0.001:
            issues.append("VARIANCE COLLAPSE detected")

        if self.sign_balance < 0.3 or self.sign_balance > 0.7:
            issues.append(f"Prediction bias ({self.sign_balance*100:.0f}% positive)")

        if not issues:
            return "ROBUST: Model passes all validation checks. Safe for production."
        elif len(issues) == 1 and "Marginal" in issues[0]:
            return f"ACCEPTABLE: {issues[0]}. Monitor in production."
        else:
            return f"CAUTION: {'; '.join(issues)}"


def calculate_wfe(
    val_metric: float,
    test_metric: float,
    baseline: float = 0.5,
) -> float:
    """
    Calculate Walk Forward Efficiency.

    WFE measures how well in-sample (validation) performance translates
    to out-of-sample (test) performance. It's a key metric for detecting
    overfitting in trading strategies.

    Formula:
        WFE = (Test Metric / Validation Metric) * 100

    Interpretation:
    - WFE > 60%: Good - strategy is likely robust
    - WFE 50-60%: Acceptable - minor overfitting
    - WFE 40-50%: Marginal - some overfitting
    - WFE < 40%: Poor - significant overfitting

    Args:
        val_metric: Validation performance (e.g., direction accuracy, Sharpe)
        test_metric: Test performance (same metric as validation)
        baseline: Baseline for the metric (0.5 for direction accuracy, 0 for Sharpe)

    Returns:
        WFE as a percentage (0-100+)
    """
    # Adjust for baseline
    val_above_baseline = val_metric - baseline
    test_above_baseline = test_metric - baseline

    if val_above_baseline <= 0:
        # If validation is at or below baseline, WFE is undefined/zero
        return 0.0

    if test_above_baseline <= 0:
        # If test is at or below baseline, strategy doesn't work
        return 0.0

    return (test_above_baseline / val_above_baseline) * 100


def calculate_overfitting_ratio(
    train_metric: float,
    test_metric: float,
    baseline: float = 0.5,
) -> float:
    """
    Calculate overfitting ratio.

    Measures how much better the model performs on training data vs test data.
    A ratio > 1.0 indicates overfitting.

    Formula:
        Ratio = (Train - Baseline) / (Test - Baseline)

    Args:
        train_metric: Training performance
        test_metric: Test performance
        baseline: Baseline for the metric

    Returns:
        Overfitting ratio (1.0 = no overfitting, >1.0 = overfitting)
    """
    train_above = train_metric - baseline
    test_above = test_metric - baseline

    if test_above <= 0:
        return float('inf')  # Complete failure on test

    return train_above / test_above


def calculate_consistency_score(fold_metrics: List[float]) -> float:
    """
    Calculate consistency score across folds.

    Measures how consistent the model is across different time periods.
    A high consistency score (close to 1.0) indicates stable performance.

    Formula:
        Consistency = 1 - (std / mean)  [coefficient of variation inverse]

    Args:
        fold_metrics: List of metric values from each fold

    Returns:
        Consistency score (0-1, higher = more consistent)
    """
    if len(fold_metrics) < 2:
        return 1.0

    mean = np.mean(fold_metrics)
    std = np.std(fold_metrics)

    if mean == 0:
        return 0.0

    cv = std / abs(mean)  # Coefficient of variation

    # Convert to score (inverse, capped at 1.0)
    consistency = 1.0 - min(cv, 1.0)

    return max(0.0, consistency)


def calculate_sharpe_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized Sharpe ratio.

    Formula:
        Sharpe = (mean(returns) - rf) / std(returns) * sqrt(periods)

    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate (default 0)
        periods_per_year: Trading periods per year (252 for daily)

    Returns:
        Annualized Sharpe ratio
    """
    if len(returns) == 0 or np.std(returns) == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    sharpe = np.mean(excess_returns) / np.std(excess_returns)

    # Annualize
    annualized_sharpe = sharpe * np.sqrt(periods_per_year)

    return float(annualized_sharpe)


def calculate_sortino_ratio(
    returns: np.ndarray,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Calculate annualized Sortino ratio (downside deviation only).

    Args:
        returns: Array of period returns
        risk_free_rate: Annual risk-free rate
        periods_per_year: Trading periods per year

    Returns:
        Annualized Sortino ratio
    """
    if len(returns) == 0:
        return 0.0

    excess_returns = returns - (risk_free_rate / periods_per_year)
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or np.std(downside_returns) == 0:
        return float('inf') if np.mean(excess_returns) > 0 else 0.0

    sortino = np.mean(excess_returns) / np.std(downside_returns)

    # Annualize
    return float(sortino * np.sqrt(periods_per_year))


def calculate_information_coefficient(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calculate Information Coefficient (IC).

    IC is the Pearson correlation between predictions and actual returns.
    A high IC indicates predictive power.

    Interpretation:
    - IC > 0.05: Good predictive power
    - IC > 0.10: Strong predictive power
    - IC < 0: Inversely correlated (bad)

    Args:
        y_true: Actual returns
        y_pred: Predicted returns

    Returns:
        Information Coefficient (-1 to 1)
    """
    if len(y_true) < 2 or len(y_pred) < 2:
        return 0.0

    # Flatten arrays
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    # Check for variance
    if np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0

    ic = np.corrcoef(y_true, y_pred)[0, 1]

    return float(ic) if not np.isnan(ic) else 0.0


def calculate_direction_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> float:
    """
    Calculate directional accuracy.

    Measures the percentage of times the model correctly predicts
    the direction of the move (up or down).

    Args:
        y_true: Actual returns
        y_pred: Predicted returns

    Returns:
        Direction accuracy (0-1)
    """
    if len(y_true) == 0:
        return 0.5

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    return float(np.mean(np.sign(y_true) == np.sign(y_pred)))


def detect_variance_collapse(
    predictions: np.ndarray,
    threshold: float = 0.001,
) -> Tuple[bool, float]:
    """
    Detect variance collapse in predictions.

    Variance collapse occurs when a model outputs near-constant values,
    often clustering around zero or the mean target.

    Args:
        predictions: Model predictions
        threshold: Minimum acceptable variance

    Returns:
        Tuple of (is_collapsed, variance)
    """
    if len(predictions) == 0:
        return True, 0.0

    variance = float(np.std(predictions))
    is_collapsed = variance < threshold

    return is_collapsed, variance


def detect_sign_imbalance(
    predictions: np.ndarray,
    min_fraction: float = 0.3,
) -> Tuple[bool, float]:
    """
    Detect sign imbalance in predictions.

    A healthy model should predict both positive and negative returns.
    If >70% of predictions are one sign, the model may be biased.

    Args:
        predictions: Model predictions
        min_fraction: Minimum fraction of each sign required

    Returns:
        Tuple of (is_imbalanced, positive_fraction)
    """
    if len(predictions) == 0:
        return True, 0.5

    positive_frac = float((predictions > 0).mean())
    negative_frac = float((predictions < 0).mean())

    is_imbalanced = positive_frac < min_fraction or negative_frac < min_fraction

    return is_imbalanced, positive_frac


def compute_validation_metrics(
    train_predictions: np.ndarray,
    train_actuals: np.ndarray,
    val_predictions: np.ndarray,
    val_actuals: np.ndarray,
    test_predictions: np.ndarray,
    test_actuals: np.ndarray,
    fold_test_metrics: Optional[List[float]] = None,
) -> ValidationMetrics:
    """
    Compute comprehensive validation metrics.

    Args:
        train_predictions: Training set predictions
        train_actuals: Training set actual values
        val_predictions: Validation set predictions
        val_actuals: Validation set actual values
        test_predictions: Test set predictions
        test_actuals: Test set actual values
        fold_test_metrics: Optional list of test metrics from each fold

    Returns:
        ValidationMetrics with all computed metrics
    """
    # Direction accuracy
    train_dir_acc = calculate_direction_accuracy(train_actuals, train_predictions)
    val_dir_acc = calculate_direction_accuracy(val_actuals, val_predictions)
    test_dir_acc = calculate_direction_accuracy(test_actuals, test_predictions)

    # WFE and overfitting ratio
    wfe = calculate_wfe(val_dir_acc, test_dir_acc, baseline=0.5)
    overfitting_ratio = calculate_overfitting_ratio(train_dir_acc, test_dir_acc, baseline=0.5)

    # Consistency (if fold metrics provided)
    if fold_test_metrics is not None:
        consistency = calculate_consistency_score(fold_test_metrics)
    else:
        consistency = 1.0  # Assume consistent if no fold data

    # Sharpe ratio on test returns
    test_returns = test_predictions * test_actuals  # Simple position * return
    sharpe = calculate_sharpe_ratio(test_returns)

    # Information coefficient
    ic = calculate_information_coefficient(test_actuals, test_predictions)

    # Variance and sign balance
    _, variance = detect_variance_collapse(test_predictions)
    _, sign_balance = detect_sign_imbalance(test_predictions)

    return ValidationMetrics(
        wfe=wfe,
        overfitting_ratio=overfitting_ratio,
        consistency_score=consistency,
        direction_accuracy=test_dir_acc,
        sharpe_ratio=sharpe,
        information_coefficient=ic,
        prediction_variance=variance,
        sign_balance=sign_balance,
    )
