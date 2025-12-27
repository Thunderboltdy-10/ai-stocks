"""
Production Pipeline for AI-Stocks

End-to-end pipeline that separates VALIDATION from PRODUCTION phases.

Workflow:
    1. VALIDATION PHASE:
       - Run walk-forward CV on all model types (LSTM, GBM, xLSTM-TS)
       - Compute per-model and aggregate WFE
       - Generate ValidationReport

    2. GATE CHECK:
       - If aggregate WFE < 50%, STOP (do not deploy)
       - If aggregate WFE >= 50%, proceed to production

    3. PRODUCTION PHASE:
       - Train all models on FULL dataset
       - Train stacking meta-learner
       - Save complete ensemble for deployment

Key Principles:
- Validation uses walk-forward CV (no data leakage)
- Production training uses ALL available data
- WFE threshold (50%) prevents overfitted models from deployment

Author: AI-Stocks Nuclear Redesign
Date: December 2025
"""

import sys
import pickle
import logging
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Callable
from datetime import datetime
from enum import Enum

import numpy as np
import pandas as pd

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

# Import validation components
from validation.walk_forward import (
    WalkForwardConfig,
    WalkForwardMode,
    WalkForwardValidator,
    WalkForwardResults,
)
from validation.wfe_metrics import (
    ValidationMetrics,
    compute_validation_metrics,
    calculate_wfe,
    calculate_consistency_score,
    calculate_sharpe_ratio,
    detect_variance_collapse,
)

# Import training components
from training.train_stacking_ensemble import (
    StackingConfig,
    StackingEnsembleTrainer,
    StackingEnsemble,
)

# Import inference components
from inference.stacking_predictor import (
    StackingPredictor,
    PredictionResult,
)

# Import data components
from data.data_fetcher import fetch_stock_data
from data.feature_engineer import engineer_features, get_feature_columns, EXPECTED_FEATURE_COUNT

# Import model paths
from utils.model_paths import ModelPaths, SAVED_MODELS_ROOT


class ModelType(Enum):
    """Supported model types in the ensemble."""
    LSTM = "lstm"           # LSTM+Transformer regressor
    GBM = "gbm"             # XGBoost/LightGBM
    XLSTM_TS = "xlstm_ts"   # xLSTM-TS (extended LSTM)


@dataclass
class PipelineConfig:
    """Configuration for production pipeline.

    Attributes:
        wfe_threshold: Minimum WFE required to proceed to production (default: 50%)
        walk_forward_folds: Number of walk-forward validation folds
        sequence_length: Sequence length for LSTM models
        batch_size: Training batch size (high for GPU acceleration)
        epochs_validation: Epochs during validation phase (fewer)
        epochs_production: Epochs during production training (more)
        include_lstm: Include LSTM+Transformer in ensemble
        include_gbm: Include GBM (XGBoost/LightGBM) in ensemble
        include_xlstm: Include xLSTM-TS in ensemble
        meta_learner_type: Type of meta-learner ('xgboost', 'lightgbm', 'ridge')
        gpu_acceleration: Enable GPU acceleration
        mixed_precision: Enable mixed precision training
        verbose: Verbosity level
    """
    wfe_threshold: float = 50.0
    walk_forward_folds: int = 5
    sequence_length: int = 60
    batch_size: int = 512
    epochs_validation: int = 30
    epochs_production: int = 50
    include_lstm: bool = True
    include_gbm: bool = True
    include_xlstm: bool = True
    meta_learner_type: str = 'xgboost'
    gpu_acceleration: bool = True
    mixed_precision: bool = True
    verbose: int = 1


@dataclass
class ModelValidationResult:
    """Validation results for a single model type.

    Attributes:
        model_type: Type of model validated
        wfe: Walk Forward Efficiency (0-100+)
        direction_accuracy: Mean directional accuracy on test folds
        direction_accuracy_std: Std of directional accuracy across folds
        sharpe_ratio: Mean Sharpe ratio on test folds
        consistency_score: Consistency across folds (0-1)
        fold_results: Per-fold detailed results
        oof_predictions: Out-of-fold predictions (for stacking)
        oof_indices: Indices for OOF predictions
        variance_collapsed: Whether variance collapse was detected
        passed: Whether model passed validation (WFE >= threshold)
    """
    model_type: ModelType
    wfe: float
    direction_accuracy: float
    direction_accuracy_std: float
    sharpe_ratio: float
    consistency_score: float
    fold_results: List[Dict[str, float]]
    oof_predictions: np.ndarray
    oof_indices: np.ndarray
    variance_collapsed: bool
    passed: bool

    def summary(self) -> str:
        """Generate summary string."""
        status = "PASS" if self.passed else "FAIL"
        collapse_warning = " [VARIANCE COLLAPSE]" if self.variance_collapsed else ""
        return (
            f"{self.model_type.value}: WFE={self.wfe:.1f}% "
            f"Dir Acc={self.direction_accuracy:.4f}+/-{self.direction_accuracy_std:.4f} "
            f"Sharpe={self.sharpe_ratio:.2f} "
            f"Consistency={self.consistency_score:.2f} "
            f"[{status}]{collapse_warning}"
        )


@dataclass
class ValidationReport:
    """Complete validation report for all models.

    Attributes:
        symbol: Stock symbol
        timestamp: Validation timestamp
        config: Pipeline configuration used
        model_results: Dict of model type -> ModelValidationResult
        aggregate_wfe: Aggregate WFE across all models
        passed: Whether aggregate WFE passes threshold
        recommendation: Human-readable recommendation
        data_summary: Summary of data used for validation
    """
    symbol: str
    timestamp: datetime
    config: PipelineConfig
    model_results: Dict[ModelType, ModelValidationResult]
    aggregate_wfe: float
    passed: bool
    recommendation: str
    data_summary: Dict[str, Any]

    def summary(self) -> str:
        """Generate comprehensive summary."""
        lines = [
            "=" * 70,
            "VALIDATION REPORT",
            "=" * 70,
            f"Symbol: {self.symbol}",
            f"Timestamp: {self.timestamp.isoformat()}",
            f"WFE Threshold: {self.config.wfe_threshold}%",
            "",
            "Data Summary:",
            f"  Total Samples: {self.data_summary.get('total_samples', 'N/A')}",
            f"  Date Range: {self.data_summary.get('start_date', 'N/A')} to {self.data_summary.get('end_date', 'N/A')}",
            f"  Features: {self.data_summary.get('n_features', 'N/A')}",
            "",
            "Per-Model Results:",
        ]

        for model_type, result in self.model_results.items():
            lines.append(f"  {result.summary()}")

        lines.extend([
            "",
            f"Aggregate WFE: {self.aggregate_wfe:.1f}%",
            f"Status: {'PASSED' if self.passed else 'FAILED'}",
            "",
            f"Recommendation: {self.recommendation}",
            "=" * 70,
        ])

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'symbol': self.symbol,
            'timestamp': self.timestamp.isoformat(),
            'aggregate_wfe': float(self.aggregate_wfe),
            'passed': bool(self.passed),
            'recommendation': self.recommendation,
            'data_summary': self.data_summary,
            'model_results': {
                mt.value: {
                    'wfe': float(mr.wfe),
                    'direction_accuracy': float(mr.direction_accuracy),
                    'direction_accuracy_std': float(mr.direction_accuracy_std),
                    'sharpe_ratio': float(mr.sharpe_ratio),
                    'consistency_score': float(mr.consistency_score),
                    'variance_collapsed': bool(mr.variance_collapsed),
                    'passed': bool(mr.passed),
                }
                for mt, mr in self.model_results.items()
            },
        }


@dataclass
class ProductionModel:
    """Container for a trained production model.

    Attributes:
        model_type: Type of model
        model: The trained model object
        scaler: Feature scaler used
        feature_columns: List of feature column names
        metadata: Training metadata
        is_trained: Whether model has been trained
    """
    model_type: ModelType
    model: Any
    scaler: Any
    feature_columns: List[str]
    metadata: Dict[str, Any]
    is_trained: bool = False

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions from the model.

        Args:
            X: Input features (n_samples, n_features) or (n_samples, seq_len, n_features)

        Returns:
            Predictions array
        """
        if not self.is_trained:
            raise RuntimeError(f"Model {self.model_type.value} has not been trained")

        return self.model.predict(X)


class ProductionPipeline:
    """
    Main orchestrator for the production pipeline.

    Separates validation from production training:
    1. run_validation() - Validates models with walk-forward CV
    2. train_production() - Trains on all data (only if validation passes)
    3. predict() - Generates live predictions

    Example:
        >>> pipeline = ProductionPipeline('AAPL')
        >>> report = pipeline.run_validation()
        >>> if report.passed:
        ...     pipeline.train_production()
        ...     prediction = pipeline.predict(latest_features)
    """

    def __init__(
        self,
        symbol: str,
        config: Optional[PipelineConfig] = None,
        output_dir: Optional[Path] = None,
    ):
        """Initialize production pipeline.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            config: Pipeline configuration (defaults to PipelineConfig())
            output_dir: Output directory for saved models (defaults to saved_models/)
        """
        self.symbol = symbol.upper()
        self.config = config or PipelineConfig()
        self.output_dir = output_dir or SAVED_MODELS_ROOT / self.symbol

        # Model paths
        self.paths = ModelPaths(self.symbol)

        # Data containers (populated during run)
        self.df_raw: Optional[pd.DataFrame] = None
        self.df_features: Optional[pd.DataFrame] = None
        self.X: Optional[np.ndarray] = None
        self.y: Optional[np.ndarray] = None
        self.feature_columns: List[str] = []

        # Validation results
        self.validation_report: Optional[ValidationReport] = None

        # Production models
        self.production_models: Dict[ModelType, ProductionModel] = {}
        self.stacking_ensemble: Optional[StackingEnsemble] = None
        self.stacking_predictor: Optional[StackingPredictor] = None

        # Setup GPU if enabled
        if self.config.gpu_acceleration:
            self._setup_gpu()

    def _setup_gpu(self):
        """Configure GPU acceleration and mixed precision."""
        try:
            import tensorflow as tf
            from tensorflow.keras import mixed_precision

            # Set mixed precision policy
            if self.config.mixed_precision:
                policy = mixed_precision.Policy('mixed_float16')
                mixed_precision.set_global_policy(policy)
                logger.info(f"Mixed precision enabled: {policy.name}")

            # Check GPU availability
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                logger.info(f"Found {len(gpus)} GPU(s): {gpus}")
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            else:
                logger.warning("No GPU found, using CPU")

        except Exception as e:
            logger.warning(f"GPU setup failed: {e}")

    def _load_data(self) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Load and prepare data for training/validation.

        Returns:
            Tuple of (df_features, X_sequences, y_targets)
        """
        logger.info(f"Loading data for {self.symbol}...")

        # Fetch raw data
        self.df_raw = fetch_stock_data(self.symbol, period='max')
        logger.info(f"Fetched {len(self.df_raw)} days of raw data")

        # Engineer features
        self.df_features = engineer_features(
            self.df_raw,
            symbol=self.symbol,
            include_sentiment=True,
        )
        logger.info(f"Engineered features: {self.df_features.shape}")

        # Get feature columns (exclude OHLCV and targets)
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends',
                       'Stock Splits', 'Date', 'forward_return', 'label']
        exclude_cols += [c for c in self.df_features.columns if c.startswith('target_')]
        self.feature_columns = [c for c in self.df_features.columns if c not in exclude_cols]

        # Create target (1-day forward returns)
        if 'returns' not in self.df_features.columns:
            self.df_features['returns'] = self.df_features['Close'].pct_change().shift(-1)

        # Drop NaN rows
        df_clean = self.df_features.dropna(subset=['returns'] + self.feature_columns)
        logger.info(f"Clean data shape: {df_clean.shape}")

        # Create sequences for LSTM models
        X, y = self._create_sequences(
            df_clean[self.feature_columns].values,
            df_clean['returns'].values,
            self.config.sequence_length,
        )

        self.X = X
        self.y = y

        logger.info(f"Prepared {len(X)} sequences with {X.shape[-1]} features")

        return self.df_features, X, y

    def _create_sequences(
        self,
        features: np.ndarray,
        targets: np.ndarray,
        sequence_length: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM models.

        Args:
            features: Feature matrix (n_samples, n_features)
            targets: Target vector (n_samples,)
            sequence_length: Length of input sequences

        Returns:
            Tuple of (X_sequences, y_targets)
        """
        X, y = [], []

        for i in range(len(features) - sequence_length):
            X.append(features[i:i + sequence_length])
            y.append(targets[i + sequence_length - 1])  # Target at end of sequence

        return np.array(X), np.array(y)

    def _create_model_factories(self) -> Dict[str, Callable[[], Any]]:
        """Create model factory functions for walk-forward validation.

        Returns:
            Dict mapping model name to factory function
        """
        factories = {}
        n_features = self.X.shape[-1] if len(self.X.shape) == 3 else self.X.shape[1]

        if self.config.include_lstm:
            def lstm_factory():
                from models.lstm_transformer_paper import (
                    create_paper_model,
                    DirectionalHuberLoss,
                )
                model = create_paper_model(
                    sequence_length=self.config.sequence_length,
                    n_features=n_features,
                )
                model.compile(
                    optimizer='adam',
                    loss=DirectionalHuberLoss(),
                )
                return model
            factories['lstm'] = lstm_factory

        if self.config.include_gbm:
            def gbm_factory():
                try:
                    import xgboost as xgb
                    return xgb.XGBRegressor(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=0.1,
                        early_stopping_rounds=30,
                        random_state=42,
                    )
                except ImportError:
                    import lightgbm as lgb
                    return lgb.LGBMRegressor(
                        n_estimators=200,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        reg_lambda=0.1,
                        random_state=42,
                    )
            factories['gbm'] = gbm_factory

        if self.config.include_xlstm:
            def xlstm_factory():
                from models.xlstm_ts import create_xlstm_ts
                model = create_xlstm_ts(
                    input_dim=n_features,
                    seq_length=self.config.sequence_length,
                    hidden_dim=64,
                    num_layers=2,
                    dropout=0.2,
                    use_wavelet=True,
                )
                model.compile(
                    optimizer='adam',
                    loss='mse',
                )
                return model
            factories['xlstm_ts'] = xlstm_factory

        return factories

    def _validate_model(
        self,
        model_type: ModelType,
        factory: Callable[[], Any],
        X: np.ndarray,
        y: np.ndarray,
    ) -> ModelValidationResult:
        """Run walk-forward validation for a single model type.

        Args:
            model_type: Type of model
            factory: Factory function to create model instances
            X: Feature matrix
            y: Target vector

        Returns:
            ModelValidationResult with validation metrics
        """
        logger.info(f"\n{'='*50}")
        logger.info(f"Validating {model_type.value}...")
        logger.info(f"{'='*50}")

        # Configure walk-forward validation
        wf_config = WalkForwardConfig(
            mode=WalkForwardMode.ANCHORED,
            n_iterations=self.config.walk_forward_folds,
            train_pct=0.60,
            validation_pct=0.15,
            test_pct=0.25,
            gap_days=1,
            purge_days=self.config.sequence_length,
        )

        validator = WalkForwardValidator(wf_config)

        # Prepare data based on model type
        if model_type == ModelType.GBM:
            # GBM uses flattened features
            X_flat = X.reshape(len(X), -1) if len(X.shape) == 3 else X
            fit_kwargs = {}
        else:
            # LSTM/xLSTM use sequences
            X_flat = X
            fit_kwargs = {
                'epochs': self.config.epochs_validation,
                'batch_size': self.config.batch_size,
                'verbose': 0,
            }

        # Run walk-forward validation
        try:
            results = validator.validate(
                model_factory=factory,
                X=X_flat,
                y=y,
                fit_kwargs=fit_kwargs,
            )

            # Extract metrics
            fold_dir_accs = [fm.test_direction_acc for fm in results.fold_metrics]
            fold_wfes = [fm.wfe for fm in results.fold_metrics]

            # Check for variance collapse
            collapsed, variance = detect_variance_collapse(results.oof_predictions)

            # Calculate consistency
            consistency = calculate_consistency_score(fold_dir_accs)

            # Calculate mean Sharpe (simplified)
            sharpe = 0.0
            if not np.isnan(results.oof_predictions).all():
                valid_mask = ~np.isnan(results.oof_predictions)
                if valid_mask.sum() > 10:
                    oof_returns = results.oof_predictions[valid_mask] * y[valid_mask]
                    sharpe = calculate_sharpe_ratio(oof_returns)

            passed = results.aggregate_wfe >= self.config.wfe_threshold and not collapsed

            return ModelValidationResult(
                model_type=model_type,
                wfe=results.aggregate_wfe,
                direction_accuracy=results.mean_test_direction_acc,
                direction_accuracy_std=results.std_test_direction_acc,
                sharpe_ratio=sharpe,
                consistency_score=consistency,
                fold_results=[
                    {
                        'fold': fm.fold,
                        'train_dir_acc': fm.train_direction_acc,
                        'val_dir_acc': fm.val_direction_acc,
                        'test_dir_acc': fm.test_direction_acc,
                        'wfe': fm.wfe,
                    }
                    for fm in results.fold_metrics
                ],
                oof_predictions=results.oof_predictions,
                oof_indices=results.oof_indices,
                variance_collapsed=collapsed,
                passed=passed,
            )

        except Exception as e:
            logger.error(f"Validation failed for {model_type.value}: {e}")
            import traceback
            traceback.print_exc()

            # Return failed result
            return ModelValidationResult(
                model_type=model_type,
                wfe=0.0,
                direction_accuracy=0.5,
                direction_accuracy_std=0.0,
                sharpe_ratio=0.0,
                consistency_score=0.0,
                fold_results=[],
                oof_predictions=np.zeros(len(y)),
                oof_indices=np.arange(len(y)),
                variance_collapsed=True,
                passed=False,
            )

    def run_validation(self, symbol: Optional[str] = None) -> ValidationReport:
        """Run walk-forward validation on all models.

        This is the VALIDATION PHASE - no production training happens here.

        Args:
            symbol: Optional symbol override (uses self.symbol if not provided)

        Returns:
            ValidationReport with all validation results
        """
        if symbol:
            self.symbol = symbol.upper()

        logger.info(f"\n{'='*70}")
        logger.info(f"VALIDATION PHASE: {self.symbol}")
        logger.info(f"{'='*70}")

        # Load data
        _, X, y = self._load_data()

        # Create model factories
        factories = self._create_model_factories()

        # Validate each model type
        model_results: Dict[ModelType, ModelValidationResult] = {}

        model_type_map = {
            'lstm': ModelType.LSTM,
            'gbm': ModelType.GBM,
            'xlstm_ts': ModelType.XLSTM_TS,
        }

        for name, factory in factories.items():
            model_type = model_type_map.get(name)
            if model_type is None:
                continue

            result = self._validate_model(model_type, factory, X, y)
            model_results[model_type] = result

            logger.info(f"\n{result.summary()}")

        # Calculate aggregate WFE (weighted by passed models)
        passed_results = [r for r in model_results.values() if r.passed]
        if passed_results:
            aggregate_wfe = np.mean([r.wfe for r in passed_results])
        else:
            aggregate_wfe = np.mean([r.wfe for r in model_results.values()])

        # Determine overall pass/fail
        passed = aggregate_wfe >= self.config.wfe_threshold

        # Generate recommendation
        if passed:
            if aggregate_wfe >= 60:
                recommendation = (
                    "STRONG: Strategy is robust. Safe to train production models on all data. "
                    f"All {len(passed_results)} models passed validation."
                )
            else:
                recommendation = (
                    f"ACCEPTABLE: Strategy shows reasonable robustness (WFE={aggregate_wfe:.1f}%). "
                    "Proceed with production training but monitor closely."
                )
        else:
            recommendation = (
                f"FAIL: Aggregate WFE ({aggregate_wfe:.1f}%) below threshold ({self.config.wfe_threshold}%). "
                "Do NOT proceed to production. Investigate overfitting issues."
            )

        # Data summary
        data_summary = {
            'total_samples': len(X),
            'start_date': str(self.df_raw.index[0].date()) if self.df_raw is not None else 'N/A',
            'end_date': str(self.df_raw.index[-1].date()) if self.df_raw is not None else 'N/A',
            'n_features': X.shape[-1] if len(X.shape) == 3 else X.shape[1],
            'sequence_length': self.config.sequence_length,
        }

        # Create report
        self.validation_report = ValidationReport(
            symbol=self.symbol,
            timestamp=datetime.now(),
            config=self.config,
            model_results=model_results,
            aggregate_wfe=aggregate_wfe,
            passed=passed,
            recommendation=recommendation,
            data_summary=data_summary,
        )

        logger.info(f"\n{self.validation_report.summary()}")

        # Save validation report
        self._save_validation_report()

        return self.validation_report

    def _save_validation_report(self):
        """Save validation report to disk."""
        self.paths.ensure_dirs()

        report_path = self.output_dir / 'validation_report.json'
        with open(report_path, 'w') as f:
            json.dump(self.validation_report.to_dict(), f, indent=2)

        logger.info(f"Validation report saved to {report_path}")

    def train_production(self, symbol: Optional[str] = None, force: bool = False) -> bool:
        """Train production models on ALL data.

        This is the PRODUCTION PHASE - only runs if validation passed.

        Args:
            symbol: Optional symbol override
            force: Force training even if validation failed

        Returns:
            True if training succeeded, False otherwise
        """
        if symbol:
            self.symbol = symbol.upper()

        # Check validation status
        if self.validation_report is None:
            logger.warning("No validation report found. Running validation first...")
            self.run_validation()

        if not self.validation_report.passed and not force:
            logger.error(
                f"Validation FAILED (WFE={self.validation_report.aggregate_wfe:.1f}%). "
                "Production training blocked. Use force=True to override."
            )
            return False

        logger.info(f"\n{'='*70}")
        logger.info(f"PRODUCTION PHASE: {self.symbol}")
        logger.info(f"{'='*70}")

        # Ensure data is loaded
        if self.X is None or self.y is None:
            self._load_data()

        # Train each model type on ALL data
        n_features = self.X.shape[-1] if len(self.X.shape) == 3 else self.X.shape[1]

        # Use 90/10 split for early stopping only (not validation)
        train_size = int(len(self.X) * 0.9)
        X_train, X_val = self.X[:train_size], self.X[train_size:]
        y_train, y_val = self.y[:train_size], self.y[train_size:]

        # Train LSTM+Transformer
        if self.config.include_lstm and ModelType.LSTM in self.validation_report.model_results:
            if self.validation_report.model_results[ModelType.LSTM].passed:
                logger.info("\nTraining production LSTM+Transformer...")
                self._train_production_lstm(X_train, y_train, X_val, y_val, n_features)

        # Train GBM
        if self.config.include_gbm and ModelType.GBM in self.validation_report.model_results:
            if self.validation_report.model_results[ModelType.GBM].passed:
                logger.info("\nTraining production GBM...")
                self._train_production_gbm(X_train, y_train, X_val, y_val)

        # Train xLSTM-TS
        if self.config.include_xlstm and ModelType.XLSTM_TS in self.validation_report.model_results:
            if self.validation_report.model_results[ModelType.XLSTM_TS].passed:
                logger.info("\nTraining production xLSTM-TS...")
                self._train_production_xlstm(X_train, y_train, X_val, y_val, n_features)

        # Train stacking meta-learner
        logger.info("\nTraining stacking meta-learner...")
        self._train_stacking_ensemble()

        # Save all models
        self._save_production_models()

        logger.info(f"\nProduction training complete for {self.symbol}")
        return True

    def _train_production_lstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_features: int,
    ):
        """Train production LSTM+Transformer model."""
        try:
            from models.lstm_transformer_paper import (
                create_paper_model,
                DirectionalHuberLoss,
            )
            from sklearn.preprocessing import RobustScaler
            import tensorflow as tf

            model = create_paper_model(
                sequence_length=self.config.sequence_length,
                n_features=n_features,
            )

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=DirectionalHuberLoss(),
            )

            # Early stopping callback
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
            )

            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.epochs_production,
                batch_size=self.config.batch_size,
                callbacks=[early_stop],
                verbose=self.config.verbose,
            )

            # Create scaler (fit on flattened training data)
            scaler = RobustScaler()
            X_flat = X_train.reshape(len(X_train), -1)
            scaler.fit(X_flat)

            self.production_models[ModelType.LSTM] = ProductionModel(
                model_type=ModelType.LSTM,
                model=model,
                scaler=scaler,
                feature_columns=self.feature_columns,
                metadata={
                    'epochs': self.config.epochs_production,
                    'batch_size': self.config.batch_size,
                    'n_features': n_features,
                    'trained_at': datetime.now().isoformat(),
                },
                is_trained=True,
            )

            logger.info("LSTM+Transformer production model trained successfully")

        except Exception as e:
            logger.error(f"Failed to train LSTM: {e}")
            import traceback
            traceback.print_exc()

    def _train_production_gbm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ):
        """Train production GBM model."""
        try:
            from sklearn.preprocessing import RobustScaler

            # Flatten for GBM
            X_train_flat = X_train.reshape(len(X_train), -1) if len(X_train.shape) == 3 else X_train
            X_val_flat = X_val.reshape(len(X_val), -1) if len(X_val.shape) == 3 else X_val

            try:
                import xgboost as xgb
                model = xgb.XGBRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=0.1,
                    early_stopping_rounds=30,
                    random_state=42,
                )
                model.fit(
                    X_train_flat, y_train,
                    eval_set=[(X_val_flat, y_val)],
                    verbose=False,
                )
                logger.info(f"XGBoost trained: {model.best_iteration} iterations")

            except ImportError:
                import lightgbm as lgb
                model = lgb.LGBMRegressor(
                    n_estimators=500,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=0.1,
                    random_state=42,
                )
                model.fit(
                    X_train_flat, y_train,
                    eval_set=[(X_val_flat, y_val)],
                    callbacks=[lgb.early_stopping(stopping_rounds=30)],
                )
                logger.info("LightGBM trained")

            # Create scaler
            scaler = RobustScaler()
            scaler.fit(X_train_flat)

            self.production_models[ModelType.GBM] = ProductionModel(
                model_type=ModelType.GBM,
                model=model,
                scaler=scaler,
                feature_columns=self.feature_columns,
                metadata={
                    'n_features': X_train_flat.shape[1],
                    'trained_at': datetime.now().isoformat(),
                },
                is_trained=True,
            )

            logger.info("GBM production model trained successfully")

        except Exception as e:
            logger.error(f"Failed to train GBM: {e}")
            import traceback
            traceback.print_exc()

    def _train_production_xlstm(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_features: int,
    ):
        """Train production xLSTM-TS model."""
        try:
            from models.xlstm_ts import create_xlstm_ts
            from sklearn.preprocessing import RobustScaler
            import tensorflow as tf

            model = create_xlstm_ts(
                input_dim=n_features,
                seq_length=self.config.sequence_length,
                hidden_dim=64,
                num_layers=2,
                dropout=0.2,
                use_wavelet=True,
            )

            model.compile(
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss='mse',
            )

            # Early stopping callback
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True,
            )

            model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=self.config.epochs_production,
                batch_size=self.config.batch_size,
                callbacks=[early_stop],
                verbose=self.config.verbose,
            )

            # Create scaler
            scaler = RobustScaler()
            X_flat = X_train.reshape(len(X_train), -1)
            scaler.fit(X_flat)

            self.production_models[ModelType.XLSTM_TS] = ProductionModel(
                model_type=ModelType.XLSTM_TS,
                model=model,
                scaler=scaler,
                feature_columns=self.feature_columns,
                metadata={
                    'epochs': self.config.epochs_production,
                    'batch_size': self.config.batch_size,
                    'n_features': n_features,
                    'trained_at': datetime.now().isoformat(),
                },
                is_trained=True,
            )

            logger.info("xLSTM-TS production model trained successfully")

        except Exception as e:
            logger.error(f"Failed to train xLSTM-TS: {e}")
            import traceback
            traceback.print_exc()

    def _train_stacking_ensemble(self):
        """Train stacking meta-learner on OOF predictions."""
        try:
            # Collect OOF predictions from validation
            oof_features = []
            feature_names = []

            for model_type in [ModelType.LSTM, ModelType.GBM, ModelType.XLSTM_TS]:
                if model_type in self.validation_report.model_results:
                    result = self.validation_report.model_results[model_type]
                    if result.passed and len(result.oof_predictions) == len(self.y):
                        oof_features.append(result.oof_predictions.reshape(-1, 1))
                        feature_names.append(f'pred_{model_type.value}')

            if len(oof_features) < 2:
                logger.warning("Need at least 2 models for stacking, skipping meta-learner")
                return

            # Stack OOF predictions
            meta_X = np.hstack(oof_features)

            # Add agreement features
            pred_std = np.std(meta_X, axis=1, keepdims=True)
            meta_X = np.hstack([meta_X, pred_std])
            feature_names.append('prediction_std')

            # Get valid samples (non-NaN)
            valid_mask = ~np.isnan(meta_X).any(axis=1)
            meta_X_valid = meta_X[valid_mask]
            y_valid = self.y[valid_mask]

            # Train meta-learner
            train_size = int(len(meta_X_valid) * 0.8)
            X_train = meta_X_valid[:train_size]
            y_train = y_valid[:train_size]
            X_val = meta_X_valid[train_size:]
            y_val = y_valid[train_size:]

            try:
                import xgboost as xgb
                meta_learner = xgb.XGBRegressor(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=4,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_lambda=0.1,
                    early_stopping_rounds=30,
                )
                meta_learner.fit(
                    X_train, y_train,
                    eval_set=[(X_val, y_val)],
                    verbose=False,
                )
                logger.info(f"XGBoost meta-learner trained: {meta_learner.best_iteration} iterations")

            except ImportError:
                from sklearn.linear_model import Ridge
                meta_learner = Ridge(alpha=0.1)
                meta_learner.fit(X_train, y_train)
                logger.info("Ridge meta-learner trained")

            # Store ensemble
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler()
            scaler.fit(meta_X_valid)

            self.stacking_ensemble = StackingEnsemble(
                base_models={
                    mt.value: self.production_models.get(mt)
                    for mt in [ModelType.LSTM, ModelType.GBM, ModelType.XLSTM_TS]
                    if mt in self.production_models
                },
                meta_learner=meta_learner,
                feature_columns=self.feature_columns,
                meta_feature_names=feature_names,
                regime_detector=None,
                scaler=scaler,
                metadata={
                    'n_base_models': len(oof_features),
                    'base_model_names': feature_names[:-1],  # Exclude pred_std
                    'trained_at': datetime.now().isoformat(),
                },
            )

            logger.info("Stacking ensemble trained successfully")

        except Exception as e:
            logger.error(f"Failed to train stacking ensemble: {e}")
            import traceback
            traceback.print_exc()

    def _save_production_models(self):
        """Save all production models to disk."""
        self.paths.ensure_dirs()

        # Save LSTM
        if ModelType.LSTM in self.production_models:
            pm = self.production_models[ModelType.LSTM]
            pm.model.save(str(self.paths.regressor.model))
            with open(self.paths.regressor.feature_scaler, 'wb') as f:
                pickle.dump(pm.scaler, f)
            with open(self.paths.regressor.metadata, 'wb') as f:
                pickle.dump(pm.metadata, f)
            logger.info(f"LSTM saved to {self.paths.regressor.base}")

        # Save GBM
        if ModelType.GBM in self.production_models:
            pm = self.production_models[ModelType.GBM]
            gbm_dir = self.output_dir / 'gbm'
            gbm_dir.mkdir(parents=True, exist_ok=True)

            model_type = 'xgboost' if 'xgboost' in str(type(pm.model)).lower() else 'lightgbm'
            with open(gbm_dir / f'{model_type}_model.pkl', 'wb') as f:
                pickle.dump(pm.model, f)
            with open(gbm_dir / 'feature_scaler.pkl', 'wb') as f:
                pickle.dump(pm.scaler, f)
            with open(gbm_dir / 'metadata.pkl', 'wb') as f:
                pickle.dump(pm.metadata, f)
            logger.info(f"GBM saved to {gbm_dir}")

        # Save xLSTM-TS
        if ModelType.XLSTM_TS in self.production_models:
            pm = self.production_models[ModelType.XLSTM_TS]
            xlstm_dir = self.output_dir / 'xlstm'
            xlstm_dir.mkdir(parents=True, exist_ok=True)

            pm.model.save(str(xlstm_dir / 'model.keras'))
            with open(xlstm_dir / 'feature_scaler.pkl', 'wb') as f:
                pickle.dump(pm.scaler, f)
            with open(xlstm_dir / 'metadata.pkl', 'wb') as f:
                pickle.dump(pm.metadata, f)
            logger.info(f"xLSTM-TS saved to {xlstm_dir}")

        # Save stacking ensemble
        if self.stacking_ensemble is not None:
            stacking_dir = self.output_dir / 'stacking'
            stacking_dir.mkdir(parents=True, exist_ok=True)

            with open(stacking_dir / 'meta_learner.pkl', 'wb') as f:
                pickle.dump(self.stacking_ensemble.meta_learner, f)
            with open(stacking_dir / 'meta_scaler.pkl', 'wb') as f:
                pickle.dump(self.stacking_ensemble.scaler, f)
            with open(stacking_dir / 'stacking_metadata.pkl', 'wb') as f:
                pickle.dump({
                    'feature_columns': self.stacking_ensemble.feature_columns,
                    'meta_feature_names': self.stacking_ensemble.meta_feature_names,
                    'metadata': self.stacking_ensemble.metadata,
                }, f)
            logger.info(f"Stacking ensemble saved to {stacking_dir}")

        # Save feature columns
        with open(self.paths.feature_columns, 'wb') as f:
            pickle.dump(self.feature_columns, f)
        logger.info(f"Feature columns saved to {self.paths.feature_columns}")

    def predict(
        self,
        latest_data: Optional[np.ndarray] = None,
        symbol: Optional[str] = None,
    ) -> PredictionResult:
        """Generate live prediction using the ensemble.

        Args:
            latest_data: Latest feature data (seq_len, n_features)
                        If None, fetches latest data automatically
            symbol: Optional symbol override

        Returns:
            PredictionResult with prediction, confidence, and position size
        """
        if symbol:
            self.symbol = symbol.upper()

        # Load predictor if not already loaded
        if self.stacking_predictor is None:
            self.stacking_predictor = StackingPredictor(
                self.symbol,
                model_dir=self.output_dir,
            )

        # Fetch latest data if not provided
        if latest_data is None:
            latest_data = self._get_latest_features()

        # Generate prediction
        return self.stacking_predictor.predict(latest_data)

    def _get_latest_features(self) -> np.ndarray:
        """Fetch and engineer latest features for prediction."""
        # Fetch recent data (need sequence_length + buffer for indicators)
        df = fetch_stock_data(self.symbol, period='3mo')

        # Engineer features
        df_features = engineer_features(
            df,
            symbol=self.symbol,
            include_sentiment=True,
        )

        # Get feature columns
        if self.feature_columns:
            feature_cols = self.feature_columns
        else:
            feature_cols = get_feature_columns(include_sentiment=True)

        # Extract latest sequence
        X = df_features[feature_cols].values

        # Take last sequence_length rows
        if len(X) >= self.config.sequence_length:
            X = X[-self.config.sequence_length:]

        return X.reshape(1, *X.shape)  # Add batch dimension


def main():
    """CLI entry point for production pipeline."""
    import argparse

    parser = argparse.ArgumentParser(description='Production Pipeline for AI-Stocks')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol')
    parser.add_argument('--validate-only', action='store_true', help='Run validation only')
    parser.add_argument('--force', action='store_true', help='Force production training even if validation fails')
    parser.add_argument('--wfe-threshold', type=float, default=50.0, help='WFE threshold (default: 50)')
    parser.add_argument('--epochs', type=int, default=50, help='Production training epochs')
    parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
    parser.add_argument('--output-dir', type=str, help='Output directory')

    args = parser.parse_args()

    # Create config
    config = PipelineConfig(
        wfe_threshold=args.wfe_threshold,
        epochs_production=args.epochs,
        batch_size=args.batch_size,
    )

    # Create pipeline
    output_dir = Path(args.output_dir) if args.output_dir else None
    pipeline = ProductionPipeline(args.symbol, config=config, output_dir=output_dir)

    # Run validation
    report = pipeline.run_validation()

    if args.validate_only:
        print("\nValidation only mode - skipping production training")
        return

    # Train production if validation passed (or forced)
    if report.passed or args.force:
        pipeline.train_production(force=args.force)
    else:
        print("\nValidation failed - production training blocked")
        print("Use --force to override")


if __name__ == '__main__':
    main()
