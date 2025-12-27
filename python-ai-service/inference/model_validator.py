"""
Model Validator for Production Inference

This module provides quality gates and validation checks for model predictions
before they are used in production trading decisions.

Key Features:
- Variance checks to detect collapsed predictions
- Directional accuracy validation
- Prediction range sanity checks
- Model health scoring
- Automatic fallback recommendations

Quality Gates:
1. Variance Gate: std(predictions) >= MIN_STD_THRESHOLD (0.003)
2. Direction Gate: directional_accuracy >= MIN_DIR_ACC (0.50)
3. Range Gate: predictions within reasonable bounds (-0.15, 0.15)
4. Distribution Gate: predictions not all same value

Usage:
    from inference.model_validator import ModelValidator
    
    validator = ModelValidator()
    result = validator.validate_predictions(predictions, y_true)
    
    if result.is_valid:
        # Use predictions
    else:
        # Fall back to alternative model
        print(f"Validation failed: {result.failures}")
"""

import sys
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum

import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================

class ValidationThresholds:
    """Configurable thresholds for model validation."""
    
    # Variance checks
    MIN_STD_THRESHOLD = 0.003      # Minimum std for healthy predictions
    WARN_STD_THRESHOLD = 0.005    # Warning threshold
    
    # Directional accuracy
    MIN_DIR_ACC = 0.50            # Minimum directional accuracy (50% = random)
    GOOD_DIR_ACC = 0.52           # Good directional accuracy
    EXCELLENT_DIR_ACC = 0.55      # Excellent directional accuracy
    
    # Prediction range
    MIN_REASONABLE_PRED = -0.15   # -15% (max expected daily loss)
    MAX_REASONABLE_PRED = 0.15    # +15% (max expected daily gain)
    
    # Distribution checks
    MAX_SAME_SIGN_RATIO = 0.95    # Max ratio of same-sign predictions
    MIN_UNIQUE_VALUES = 10        # Minimum unique prediction values
    
    # Overall health thresholds
    HEALTHY_SCORE = 0.7           # Minimum score for healthy model
    WARNING_SCORE = 0.5           # Warning threshold


class ValidationStatus(Enum):
    """Status of model validation."""
    HEALTHY = "healthy"
    WARNING = "warning"
    FAILED = "failed"
    COLLAPSED = "collapsed"
    UNKNOWN = "unknown"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class ValidationResult:
    """Container for validation results."""
    is_valid: bool
    status: ValidationStatus
    health_score: float
    checks: Dict[str, bool]
    metrics: Dict[str, float]
    failures: List[str]
    warnings: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'is_valid': self.is_valid,
            'status': self.status.value,
            'health_score': self.health_score,
            'checks': self.checks,
            'metrics': self.metrics,
            'failures': self.failures,
            'warnings': self.warnings,
            'recommendations': self.recommendations,
        }


@dataclass
class ModelQualityReport:
    """Detailed quality report for a model."""
    model_name: str
    validation: ValidationResult
    prediction_stats: Dict[str, float]
    distribution_analysis: Dict[str, Any]
    timestamp: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'model_name': self.model_name,
            'validation': self.validation.to_dict(),
            'prediction_stats': self.prediction_stats,
            'distribution_analysis': self.distribution_analysis,
            'timestamp': self.timestamp,
        }


# ============================================================================
# VALIDATOR CLASS
# ============================================================================

class ModelValidator:
    """
    Validates model predictions against quality gates.
    
    This validator checks predictions for common failure modes:
    - Collapsed variance (all similar values)
    - Poor directional accuracy
    - Out-of-range predictions
    - Degenerate distributions
    """
    
    def __init__(self, thresholds: Optional[ValidationThresholds] = None):
        """Initialize validator with optional custom thresholds."""
        self.thresholds = thresholds or ValidationThresholds()
    
    def validate_predictions(
        self,
        predictions: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        model_name: str = "unknown",
    ) -> ValidationResult:
        """
        Validate model predictions against quality gates.
        
        Args:
            predictions: Model predictions (1D array)
            y_true: Optional true values for directional accuracy
            model_name: Name of the model being validated
        
        Returns:
            ValidationResult with pass/fail status and diagnostics
        """
        predictions = np.asarray(predictions).flatten()
        
        checks = {}
        metrics = {}
        failures = []
        warnings = []
        recommendations = []
        
        # =====================================================================
        # 1. Variance Check (Collapse Detection)
        # =====================================================================
        pred_std = np.std(predictions)
        pred_mean = np.mean(predictions)
        metrics['std'] = float(pred_std)
        metrics['mean'] = float(pred_mean)
        
        checks['variance'] = pred_std >= self.thresholds.MIN_STD_THRESHOLD
        
        if pred_std < self.thresholds.MIN_STD_THRESHOLD:
            failures.append(f"COLLAPSED: std={pred_std:.6f} < {self.thresholds.MIN_STD_THRESHOLD}")
            recommendations.append("Retrain regressor with VarianceRegularizedLoss")
            recommendations.append("Consider using GBM-only mode (--fusion-mode gbm_only)")
        elif pred_std < self.thresholds.WARN_STD_THRESHOLD:
            warnings.append(f"Low variance: std={pred_std:.6f}")
            recommendations.append("Monitor model for potential collapse")
        
        # =====================================================================
        # 2. Directional Accuracy Check
        # =====================================================================
        if y_true is not None:
            y_true = np.asarray(y_true).flatten()
            if len(y_true) == len(predictions):
                dir_acc = self._directional_accuracy(y_true, predictions)
                metrics['directional_accuracy'] = float(dir_acc)
                
                checks['direction'] = dir_acc >= self.thresholds.MIN_DIR_ACC
                
                if dir_acc < self.thresholds.MIN_DIR_ACC:
                    failures.append(f"Poor direction: acc={dir_acc:.2%} < {self.thresholds.MIN_DIR_ACC:.0%}")
                    recommendations.append("Model performs worse than random")
                    recommendations.append("Consider retraining with more data or different features")
                elif dir_acc < self.thresholds.GOOD_DIR_ACC:
                    warnings.append(f"Marginal direction: acc={dir_acc:.2%}")
            else:
                checks['direction'] = True  # Skip if lengths don't match
        else:
            checks['direction'] = True  # Skip if no y_true provided
        
        # =====================================================================
        # 3. Range Check
        # =====================================================================
        pred_min = np.min(predictions)
        pred_max = np.max(predictions)
        metrics['min'] = float(pred_min)
        metrics['max'] = float(pred_max)
        
        in_range = (
            pred_min >= self.thresholds.MIN_REASONABLE_PRED and
            pred_max <= self.thresholds.MAX_REASONABLE_PRED
        )
        checks['range'] = in_range
        
        if not in_range:
            warnings.append(f"Predictions outside reasonable range: [{pred_min:.4f}, {pred_max:.4f}]")
            recommendations.append("Check target scaling in training")
        
        # =====================================================================
        # 4. Distribution Check
        # =====================================================================
        n_positive = np.sum(predictions > 0)
        n_negative = np.sum(predictions < 0)
        n_total = len(predictions)
        
        positive_ratio = n_positive / n_total if n_total > 0 else 0
        negative_ratio = n_negative / n_total if n_total > 0 else 0
        max_ratio = max(positive_ratio, negative_ratio)
        
        metrics['positive_ratio'] = float(positive_ratio)
        metrics['negative_ratio'] = float(negative_ratio)
        
        n_unique = len(np.unique(np.round(predictions, decimals=6)))
        metrics['n_unique'] = int(n_unique)
        
        distribution_ok = (
            max_ratio < self.thresholds.MAX_SAME_SIGN_RATIO and
            n_unique >= self.thresholds.MIN_UNIQUE_VALUES
        )
        checks['distribution'] = distribution_ok
        
        if max_ratio >= self.thresholds.MAX_SAME_SIGN_RATIO:
            warnings.append(f"Biased predictions: {max_ratio:.0%} same sign")
        if n_unique < self.thresholds.MIN_UNIQUE_VALUES:
            failures.append(f"Degenerate distribution: only {n_unique} unique values")
            recommendations.append("Model may be outputting constant predictions")
        
        # =====================================================================
        # 5. Compute Health Score
        # =====================================================================
        health_score = self._compute_health_score(checks, metrics)
        metrics['health_score'] = float(health_score)
        
        # =====================================================================
        # 6. Determine Status
        # =====================================================================
        if pred_std < self.thresholds.MIN_STD_THRESHOLD:
            status = ValidationStatus.COLLAPSED
        elif len(failures) > 0:
            status = ValidationStatus.FAILED
        elif len(warnings) > 0:
            status = ValidationStatus.WARNING
        elif health_score >= self.thresholds.HEALTHY_SCORE:
            status = ValidationStatus.HEALTHY
        else:
            status = ValidationStatus.WARNING
        
        is_valid = status in (ValidationStatus.HEALTHY, ValidationStatus.WARNING)
        
        return ValidationResult(
            is_valid=is_valid,
            status=status,
            health_score=health_score,
            checks=checks,
            metrics=metrics,
            failures=failures,
            warnings=warnings,
            recommendations=recommendations,
        )
    
    def _directional_accuracy(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Compute directional accuracy (sign agreement)."""
        # Only evaluate where both are non-zero
        nonzero_mask = (y_true != 0) & (y_pred != 0)
        if np.sum(nonzero_mask) == 0:
            return 0.5
        
        sign_match = np.sign(y_true[nonzero_mask]) == np.sign(y_pred[nonzero_mask])
        return float(np.mean(sign_match))
    
    def _compute_health_score(
        self,
        checks: Dict[str, bool],
        metrics: Dict[str, float],
    ) -> float:
        """Compute overall health score (0-1)."""
        scores = []
        
        # Variance score (0-1)
        std = metrics.get('std', 0)
        if std >= self.thresholds.WARN_STD_THRESHOLD:
            variance_score = 1.0
        elif std >= self.thresholds.MIN_STD_THRESHOLD:
            variance_score = 0.7
        else:
            variance_score = 0.0
        scores.append(variance_score * 0.4)  # 40% weight
        
        # Direction score (0-1)
        dir_acc = metrics.get('directional_accuracy', 0.5)
        if dir_acc >= self.thresholds.EXCELLENT_DIR_ACC:
            direction_score = 1.0
        elif dir_acc >= self.thresholds.GOOD_DIR_ACC:
            direction_score = 0.8
        elif dir_acc >= self.thresholds.MIN_DIR_ACC:
            direction_score = 0.5
        else:
            direction_score = 0.0
        scores.append(direction_score * 0.4)  # 40% weight
        
        # Distribution score (0-1)
        distribution_score = 1.0 if checks.get('distribution', False) else 0.3
        scores.append(distribution_score * 0.2)  # 20% weight
        
        return sum(scores)
    
    def get_best_available_model(
        self,
        model_validations: Dict[str, ValidationResult],
    ) -> Tuple[str, ValidationResult]:
        """
        Select the best available model based on validation results.
        
        Priority:
        1. Healthy models with highest health score
        2. Warning models (some issues but usable)
        3. Fallback to GBM if LSTM collapsed
        
        Args:
            model_validations: Dict mapping model names to ValidationResult
        
        Returns:
            Tuple of (best_model_name, its_validation_result)
        """
        # Separate by status
        healthy = {k: v for k, v in model_validations.items() 
                   if v.status == ValidationStatus.HEALTHY}
        warning = {k: v for k, v in model_validations.items() 
                   if v.status == ValidationStatus.WARNING}
        failed = {k: v for k, v in model_validations.items() 
                  if v.status in (ValidationStatus.FAILED, ValidationStatus.COLLAPSED)}
        
        # Priority: healthy > warning > failed
        if healthy:
            # Return model with highest health score
            best = max(healthy.items(), key=lambda x: x[1].health_score)
            return best
        elif warning:
            best = max(warning.items(), key=lambda x: x[1].health_score)
            logger.warning(f"Using model with warnings: {best[0]}")
            return best
        elif failed:
            # Even failed models might be better than nothing
            # Prefer GBM over collapsed LSTM
            if 'gbm' in failed:
                logger.warning(f"Using failed GBM as fallback")
                return 'gbm', failed['gbm']
            best = max(failed.items(), key=lambda x: x[1].health_score)
            logger.error(f"All models failed, using {best[0]} as last resort")
            return best
        
        raise ValueError("No models available for selection")
    
    def generate_quality_report(
        self,
        predictions: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        model_name: str = "unknown",
    ) -> ModelQualityReport:
        """Generate a detailed quality report for a model."""
        from datetime import datetime
        
        predictions = np.asarray(predictions).flatten()
        validation = self.validate_predictions(predictions, y_true, model_name)
        
        # Compute additional stats
        prediction_stats = {
            'count': int(len(predictions)),
            'mean': float(np.mean(predictions)),
            'std': float(np.std(predictions)),
            'min': float(np.min(predictions)),
            'max': float(np.max(predictions)),
            'median': float(np.median(predictions)),
            'q25': float(np.percentile(predictions, 25)),
            'q75': float(np.percentile(predictions, 75)),
        }
        
        # Distribution analysis
        bins = np.linspace(-0.05, 0.05, 21)
        hist, _ = np.histogram(predictions, bins=bins)
        distribution_analysis = {
            'histogram_bins': bins.tolist(),
            'histogram_counts': hist.tolist(),
            'n_positive': int(np.sum(predictions > 0)),
            'n_negative': int(np.sum(predictions < 0)),
            'n_zero': int(np.sum(predictions == 0)),
            'skewness': float(self._compute_skewness(predictions)),
            'kurtosis': float(self._compute_kurtosis(predictions)),
        }
        
        return ModelQualityReport(
            model_name=model_name,
            validation=validation,
            prediction_stats=prediction_stats,
            distribution_analysis=distribution_analysis,
            timestamp=datetime.now().isoformat(),
        )
    
    def _compute_skewness(self, x: np.ndarray) -> float:
        """Compute skewness of distribution."""
        n = len(x)
        if n < 3:
            return 0.0
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-10:
            return 0.0
        return float(np.mean(((x - mean) / std) ** 3))
    
    def _compute_kurtosis(self, x: np.ndarray) -> float:
        """Compute kurtosis of distribution."""
        n = len(x)
        if n < 4:
            return 0.0
        mean = np.mean(x)
        std = np.std(x)
        if std < 1e-10:
            return 0.0
        return float(np.mean(((x - mean) / std) ** 4) - 3)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def validate_regressor_predictions(
    predictions: np.ndarray,
    y_true: Optional[np.ndarray] = None,
) -> ValidationResult:
    """Quick validation of regressor predictions."""
    validator = ModelValidator()
    return validator.validate_predictions(predictions, y_true, "regressor")


def is_model_collapsed(predictions: np.ndarray) -> bool:
    """Quick check if model predictions are collapsed."""
    std = np.std(predictions)
    return std < ValidationThresholds.MIN_STD_THRESHOLD


def get_recommended_fallback(validation: ValidationResult) -> str:
    """Get recommended fallback mode based on validation result."""
    if validation.status == ValidationStatus.COLLAPSED:
        return 'gbm_only'  # GBM as primary
    elif validation.status == ValidationStatus.FAILED:
        return 'gbm_heavy'  # Heavy GBM weighting
    elif validation.status == ValidationStatus.WARNING:
        return 'balanced'  # 50/50 split
    else:
        return 'lstm_heavy'  # Trust LSTM


# ============================================================================
# CLI
# ============================================================================

def main():
    """CLI for testing model validator."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Test model validator')
    parser.add_argument('--test', action='store_true', help='Run test with synthetic data')
    parser.add_argument('--test-collapsed', action='store_true', help='Test collapsed model detection')
    args = parser.parse_args()
    
    validator = ModelValidator()
    
    if args.test_collapsed:
        print("\n=== Testing Collapsed Model Detection ===")
        # Simulate collapsed predictions
        collapsed_preds = np.full(100, 0.0001) + np.random.randn(100) * 0.00001
        result = validator.validate_predictions(collapsed_preds, model_name="collapsed_test")
        
        print(f"Status: {result.status.value}")
        print(f"Is Valid: {result.is_valid}")
        print(f"Health Score: {result.health_score:.2f}")
        print(f"Std: {result.metrics['std']:.8f}")
        print(f"Failures: {result.failures}")
        print(f"Recommendations: {result.recommendations}")
        
    elif args.test:
        print("\n=== Testing Model Validator ===")
        
        # Test 1: Healthy predictions
        print("\n--- Test 1: Healthy Predictions ---")
        np.random.seed(42)
        healthy_preds = np.random.randn(100) * 0.01
        y_true = np.random.randn(100) * 0.01
        
        result = validator.validate_predictions(healthy_preds, y_true, "healthy_test")
        print(f"Status: {result.status.value}")
        print(f"Health Score: {result.health_score:.2f}")
        print(f"Checks: {result.checks}")
        
        # Test 2: Low variance
        print("\n--- Test 2: Low Variance Predictions ---")
        low_var_preds = np.random.randn(100) * 0.002
        result = validator.validate_predictions(low_var_preds, y_true, "low_var_test")
        print(f"Status: {result.status.value}")
        print(f"Health Score: {result.health_score:.2f}")
        print(f"Warnings: {result.warnings}")
        
        # Test 3: Collapsed
        print("\n--- Test 3: Collapsed Predictions ---")
        collapsed_preds = np.full(100, 0.001)
        result = validator.validate_predictions(collapsed_preds, y_true, "collapsed_test")
        print(f"Status: {result.status.value}")
        print(f"Health Score: {result.health_score:.2f}")
        print(f"Failures: {result.failures}")
        print(f"Fallback recommendation: {get_recommended_fallback(result)}")
        
        # Test 4: Model selection
        print("\n--- Test 4: Model Selection ---")
        validations = {
            'lstm': validator.validate_predictions(healthy_preds, y_true, "lstm"),
            'xgb': validator.validate_predictions(healthy_preds * 0.8, y_true, "xgb"),
            'lgb': validator.validate_predictions(collapsed_preds, y_true, "lgb"),
        }
        
        best_model, best_result = validator.get_best_available_model(validations)
        print(f"Best model: {best_model}")
        print(f"Best model status: {best_result.status.value}")
        print(f"Best model score: {best_result.health_score:.2f}")
    
    else:
        print("Use --test or --test-collapsed to run validation tests")


if __name__ == '__main__':
    main()
