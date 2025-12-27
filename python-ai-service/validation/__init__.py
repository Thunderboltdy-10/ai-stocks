"""
Validation package for AI-Stocks Python AI Service.

Nuclear Redesign: Proper validation framework with no data leakage.

Modules:
- walk_forward: Walk-forward validation framework
- wfe_metrics: Walk Forward Efficiency and related metrics
"""

from .walk_forward import (
    WalkForwardConfig,
    WalkForwardMode,
    WalkForwardSplit,
    WalkForwardValidator,
    WalkForwardResults,
    FoldMetrics,
    calculate_wfe,
)

from .wfe_metrics import (
    ValidationMetrics,
    calculate_overfitting_ratio,
    calculate_consistency_score,
    calculate_sharpe_ratio,
    calculate_sortino_ratio,
    calculate_information_coefficient,
    calculate_direction_accuracy,
    detect_variance_collapse,
    detect_sign_imbalance,
    compute_validation_metrics,
)

__all__ = [
    # Walk-forward
    "WalkForwardConfig",
    "WalkForwardMode",
    "WalkForwardSplit",
    "WalkForwardValidator",
    "WalkForwardResults",
    "FoldMetrics",
    "calculate_wfe",
    # Metrics
    "ValidationMetrics",
    "calculate_overfitting_ratio",
    "calculate_consistency_score",
    "calculate_sharpe_ratio",
    "calculate_sortino_ratio",
    "calculate_information_coefficient",
    "calculate_direction_accuracy",
    "detect_variance_collapse",
    "detect_sign_imbalance",
    "compute_validation_metrics",
]
