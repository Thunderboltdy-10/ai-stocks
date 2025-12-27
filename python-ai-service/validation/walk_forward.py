"""
Walk-Forward Validation Framework for Time Series Models

Nuclear Redesign: Proper walk-forward validation with no data leakage.

Key Features:
- Anchored (expanding) window: Training starts from beginning, grows over time
- Rolling window: Fixed-size training window slides forward
- Proper purging for sequence models (gap between train/val and val/test)
- Walk Forward Efficiency (WFE) metric
- OOF (out-of-fold) predictions for stacking

Based on:
- Walk-forward optimization best practices (QuantInsti, QuantConnect)
- Time series cross-validation (sklearn TimeSeriesSplit)

Author: AI-Stocks Nuclear Redesign
Date: December 2025
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Callable, Any, Optional, Dict
from enum import Enum
import numpy as np
import logging

logger = logging.getLogger(__name__)


class WalkForwardMode(Enum):
    """Walk-forward validation mode."""
    ANCHORED = "anchored"    # Training always starts from beginning (expanding window)
    ROLLING = "rolling"      # Fixed-size training window slides forward


@dataclass
class WalkForwardConfig:
    """Configuration for walk-forward validation.

    Attributes:
        mode: 'anchored' (expanding) or 'rolling' window
        n_iterations: Number of walk-forward folds
        train_pct: Minimum training percentage (0.60 = 60%)
        validation_pct: Validation window percentage (0.15 = 15%)
        test_pct: Out-of-sample test percentage (0.25 = 25%)
        gap_days: Gap between train and validation to prevent leakage
        purge_days: Purge overlapping sequences (for LSTM/sequence models)
    """
    mode: WalkForwardMode = WalkForwardMode.ANCHORED
    n_iterations: int = 5
    train_pct: float = 0.60
    validation_pct: float = 0.15
    test_pct: float = 0.25
    gap_days: int = 1
    purge_days: int = 60  # For sequence models with seq_length=60

    def __post_init__(self):
        """Validate configuration."""
        total = self.train_pct + self.validation_pct + self.test_pct
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Split percentages sum to {total}, not 1.0. Normalizing.")
            self.train_pct /= total
            self.validation_pct /= total
            self.test_pct /= total


@dataclass
class WalkForwardSplit:
    """A single walk-forward split with train/val/test indices.

    Attributes:
        fold: Fold number (0-indexed)
        train_start: Start index for training data
        train_end: End index for training data (exclusive)
        val_start: Start index for validation data
        val_end: End index for validation data (exclusive)
        test_start: Start index for test data
        test_end: End index for test data (exclusive)
    """
    fold: int
    train_start: int
    train_end: int
    val_start: int
    val_end: int
    test_start: int
    test_end: int

    @property
    def train_size(self) -> int:
        return self.train_end - self.train_start

    @property
    def val_size(self) -> int:
        return self.val_end - self.val_start

    @property
    def test_size(self) -> int:
        return self.test_end - self.test_start

    def has_overlap(self) -> bool:
        """Check if there's any overlap between splits."""
        return (
            self.train_end > self.val_start or
            self.val_end > self.test_start
        )

    def __repr__(self):
        return (
            f"WalkForwardSplit(fold={self.fold}, "
            f"train=[{self.train_start}:{self.train_end}] ({self.train_size}), "
            f"val=[{self.val_start}:{self.val_end}] ({self.val_size}), "
            f"test=[{self.test_start}:{self.test_end}] ({self.test_size}))"
        )


@dataclass
class FoldMetrics:
    """Metrics for a single walk-forward fold.

    Attributes:
        fold: Fold number
        train_mse: Training MSE
        val_mse: Validation MSE
        test_mse: Test MSE
        train_direction_acc: Training directional accuracy
        val_direction_acc: Validation directional accuracy
        test_direction_acc: Test directional accuracy
        val_sharpe: Validation Sharpe ratio (if computed)
        test_sharpe: Test Sharpe ratio (if computed)
        wfe: Walk Forward Efficiency for this fold
    """
    fold: int
    train_mse: float
    val_mse: float
    test_mse: float
    train_direction_acc: float
    val_direction_acc: float
    test_direction_acc: float
    val_sharpe: float = 0.0
    test_sharpe: float = 0.0
    wfe: float = 0.0

    @property
    def is_overfitting(self) -> bool:
        """Check if this fold shows signs of overfitting."""
        # If validation performance is significantly better than test
        return self.wfe < 40.0


@dataclass
class WalkForwardResults:
    """Results from walk-forward validation.

    Attributes:
        config: Configuration used
        fold_metrics: List of metrics per fold
        oof_predictions: Out-of-fold predictions (for stacking)
        oof_indices: Indices corresponding to OOF predictions
        aggregate_wfe: Aggregate Walk Forward Efficiency
        recommendation: Human-readable recommendation
    """
    config: WalkForwardConfig
    fold_metrics: List[FoldMetrics]
    oof_predictions: np.ndarray
    oof_indices: np.ndarray
    aggregate_wfe: float
    recommendation: str

    @property
    def mean_test_direction_acc(self) -> float:
        return np.mean([f.test_direction_acc for f in self.fold_metrics])

    @property
    def std_test_direction_acc(self) -> float:
        return np.std([f.test_direction_acc for f in self.fold_metrics])

    @property
    def mean_test_mse(self) -> float:
        return np.mean([f.test_mse for f in self.fold_metrics])

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            "=" * 60,
            "WALK-FORWARD VALIDATION RESULTS",
            "=" * 60,
            f"Mode: {self.config.mode.value}",
            f"Folds: {len(self.fold_metrics)}",
            "",
            "Per-Fold Metrics:",
        ]

        for fm in self.fold_metrics:
            lines.append(
                f"  Fold {fm.fold}: Test Dir Acc={fm.test_direction_acc:.4f}, "
                f"WFE={fm.wfe:.1f}%"
            )

        lines.extend([
            "",
            "Aggregate Metrics:",
            f"  Mean Test Direction Accuracy: {self.mean_test_direction_acc:.4f} +/- {self.std_test_direction_acc:.4f}",
            f"  Mean Test MSE: {self.mean_test_mse:.6f}",
            f"  Aggregate WFE: {self.aggregate_wfe:.1f}%",
            "",
            f"Recommendation: {self.recommendation}",
            "=" * 60,
        ])

        return "\n".join(lines)


class WalkForwardValidator:
    """
    Walk-forward validation for time series models.

    Implements anchored (expanding) and rolling window approaches with proper
    purging for sequence models and WFE metric calculation.

    Example:
        >>> config = WalkForwardConfig(mode=WalkForwardMode.ANCHORED, n_iterations=5)
        >>> validator = WalkForwardValidator(config)
        >>> results = validator.validate(
        ...     model_factory=lambda: LSTMModel(),
        ...     X=X_features,
        ...     y=y_targets,
        ... )
        >>> print(results.summary())
    """

    def __init__(self, config: WalkForwardConfig):
        """Initialize validator with configuration.

        Args:
            config: WalkForwardConfig with validation settings
        """
        self.config = config

    def generate_splits(self, n_samples: int) -> List[WalkForwardSplit]:
        """Generate train/val/test splits with no overlap.

        Args:
            n_samples: Total number of samples

        Returns:
            List of WalkForwardSplit objects
        """
        splits = []

        # Calculate base sizes
        min_train_size = int(n_samples * self.config.train_pct)
        val_size = int(n_samples * self.config.validation_pct)
        test_size_per_fold = int(n_samples * self.config.test_pct / self.config.n_iterations)

        if self.config.mode == WalkForwardMode.ANCHORED:
            # Anchored (expanding) window: training starts from 0, grows
            for i in range(self.config.n_iterations):
                # Training expands with each fold
                additional_train = (i * test_size_per_fold)
                train_end = min_train_size + additional_train

                # Gap before validation
                val_start = train_end + self.config.gap_days
                val_end = val_start + val_size

                # Purge before test
                test_start = val_end + self.config.purge_days
                test_end = min(test_start + test_size_per_fold, n_samples)

                # Validate we have room for this fold
                if test_start >= n_samples:
                    logger.warning(f"Not enough data for fold {i}, stopping at fold {i-1}")
                    break

                splits.append(WalkForwardSplit(
                    fold=i,
                    train_start=0,  # Always start from 0 for anchored
                    train_end=train_end,
                    val_start=val_start,
                    val_end=min(val_end, n_samples),
                    test_start=test_start,
                    test_end=test_end,
                ))

        elif self.config.mode == WalkForwardMode.ROLLING:
            # Rolling window: fixed training size that slides forward
            train_size = min_train_size

            for i in range(self.config.n_iterations):
                # Training window slides forward
                train_start = i * test_size_per_fold
                train_end = train_start + train_size

                # Gap before validation
                val_start = train_end + self.config.gap_days
                val_end = val_start + val_size

                # Purge before test
                test_start = val_end + self.config.purge_days
                test_end = min(test_start + test_size_per_fold, n_samples)

                # Validate we have room for this fold
                if test_start >= n_samples:
                    logger.warning(f"Not enough data for fold {i}, stopping at fold {i-1}")
                    break

                splits.append(WalkForwardSplit(
                    fold=i,
                    train_start=train_start,
                    train_end=train_end,
                    val_start=val_start,
                    val_end=min(val_end, n_samples),
                    test_start=test_start,
                    test_end=test_end,
                ))

        # Validate no overlaps
        for split in splits:
            if split.has_overlap():
                raise ValueError(f"Split {split.fold} has overlap! {split}")

        return splits

    def validate(
        self,
        model_factory: Callable[[], Any],
        X: np.ndarray,
        y: np.ndarray,
        fit_kwargs: Optional[Dict] = None,
        compute_sharpe: bool = False,
        sharpe_fn: Optional[Callable[[np.ndarray, np.ndarray], float]] = None,
    ) -> WalkForwardResults:
        """Run full walk-forward validation.

        Args:
            model_factory: Callable that returns a new model instance
            X: Feature matrix (n_samples, n_features) or (n_samples, seq_len, n_features)
            y: Target vector (n_samples,)
            fit_kwargs: Additional kwargs for model.fit()
            compute_sharpe: Whether to compute Sharpe ratios
            sharpe_fn: Optional function to compute Sharpe: (y_true, y_pred) -> float

        Returns:
            WalkForwardResults with all metrics and OOF predictions
        """
        splits = self.generate_splits(len(X))

        # Initialize OOF predictions array
        oof_predictions = np.full(len(X), np.nan)
        oof_indices = []
        fold_metrics = []

        fit_kwargs = fit_kwargs or {}

        logger.info(f"Starting walk-forward validation with {len(splits)} folds...")

        for split in splits:
            logger.info(f"Processing fold {split.fold}: {split}")

            # Extract data for this fold
            X_train = X[split.train_start:split.train_end]
            y_train = y[split.train_start:split.train_end]
            X_val = X[split.val_start:split.val_end]
            y_val = y[split.val_start:split.val_end]
            X_test = X[split.test_start:split.test_end]
            y_test = y[split.test_start:split.test_end]

            # Create and train model
            model = model_factory()

            # Handle different fit signatures
            try:
                # Try with validation_data (Keras-style)
                model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    **fit_kwargs
                )
            except TypeError:
                # Fall back to simple fit (sklearn-style)
                try:
                    model.fit(X_train, y_train, **fit_kwargs)
                except TypeError:
                    model.fit(X_train, y_train)

            # Generate predictions
            y_pred_train = model.predict(X_train).flatten()
            y_pred_val = model.predict(X_val).flatten()
            y_pred_test = model.predict(X_test).flatten()

            # Store OOF predictions for test set
            oof_predictions[split.test_start:split.test_end] = y_pred_test
            oof_indices.extend(range(split.test_start, split.test_end))

            # Calculate metrics
            train_mse = float(np.mean((y_train - y_pred_train) ** 2))
            val_mse = float(np.mean((y_val - y_pred_val) ** 2))
            test_mse = float(np.mean((y_test - y_pred_test) ** 2))

            train_dir_acc = float(np.mean(np.sign(y_train) == np.sign(y_pred_train)))
            val_dir_acc = float(np.mean(np.sign(y_val) == np.sign(y_pred_val)))
            test_dir_acc = float(np.mean(np.sign(y_test) == np.sign(y_pred_test)))

            # Compute Sharpe if requested
            val_sharpe = 0.0
            test_sharpe = 0.0
            if compute_sharpe and sharpe_fn is not None:
                val_sharpe = sharpe_fn(y_val, y_pred_val)
                test_sharpe = sharpe_fn(y_test, y_pred_test)

            # Calculate WFE (Walk Forward Efficiency)
            # WFE = (Test Performance / Validation Performance) * 100
            if val_dir_acc > 0.5:
                wfe = (test_dir_acc / val_dir_acc) * 100
            else:
                wfe = 0.0

            fold_metrics.append(FoldMetrics(
                fold=split.fold,
                train_mse=train_mse,
                val_mse=val_mse,
                test_mse=test_mse,
                train_direction_acc=train_dir_acc,
                val_direction_acc=val_dir_acc,
                test_direction_acc=test_dir_acc,
                val_sharpe=val_sharpe,
                test_sharpe=test_sharpe,
                wfe=wfe,
            ))

            logger.info(
                f"Fold {split.fold}: Train Dir Acc={train_dir_acc:.4f}, "
                f"Val Dir Acc={val_dir_acc:.4f}, Test Dir Acc={test_dir_acc:.4f}, "
                f"WFE={wfe:.1f}%"
            )

        # Aggregate WFE
        aggregate_wfe = np.mean([fm.wfe for fm in fold_metrics])

        # Generate recommendation
        if aggregate_wfe >= 60:
            recommendation = "STRONG: Strategy is likely robust. Safe to train production model on all data."
        elif aggregate_wfe >= 50:
            recommendation = "GOOD: Strategy shows reasonable robustness. Consider production deployment."
        elif aggregate_wfe >= 40:
            recommendation = "ACCEPTABLE: Some overfitting detected. Monitor closely in production."
        else:
            recommendation = "POOR: Significant overfitting detected. Do NOT deploy to production."

        results = WalkForwardResults(
            config=self.config,
            fold_metrics=fold_metrics,
            oof_predictions=oof_predictions,
            oof_indices=np.array(oof_indices),
            aggregate_wfe=aggregate_wfe,
            recommendation=recommendation,
        )

        logger.info(f"\n{results.summary()}")

        return results


def calculate_wfe(val_performance: float, test_performance: float) -> float:
    """Calculate Walk Forward Efficiency.

    WFE = (Test Performance / Validation Performance) * 100

    Interpretation:
    - WFE > 60%: Good - strategy is likely robust
    - WFE 40-60%: Acceptable - some overfitting present
    - WFE < 40%: Poor - significant overfitting

    Args:
        val_performance: Validation metric (e.g., direction accuracy, Sharpe)
        test_performance: Test metric (same as validation)

    Returns:
        WFE percentage (0-100+)
    """
    if val_performance <= 0 or val_performance <= 0.5:
        return 0.0
    return (test_performance / val_performance) * 100


# Create __init__.py for the validation package
def _create_init():
    """Helper to remind about package init."""
    pass
