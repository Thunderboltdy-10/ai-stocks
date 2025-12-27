"""
Stacking Ensemble Training with XGBoost Meta-Learner

Nuclear Redesign: Proper stacking ensemble that replaces naive weighted fusion.

Architecture:
    Base Models → OOF Predictions → Meta-Learner → Final Prediction
                      ↓
              Regime Features
              Model Agreement

Two-Stage Process:
    Stage 1: Generate OOF predictions from base models via walk-forward CV
    Stage 2: Train XGBoost meta-learner on stacked features

Based on:
- Stacking ensemble best practices (Kaggle, ML literature)
- XGBoost meta-learner (industry standard)

Author: AI-Stocks Nuclear Redesign
Date: December 2025
"""

import argparse
import sys
import pickle
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from validation.walk_forward import (
    WalkForwardConfig,
    WalkForwardMode,
    WalkForwardValidator,
)
from validation.wfe_metrics import (
    ValidationMetrics,
    compute_validation_metrics,
    calculate_wfe,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Try to import GBM libraries
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    logger.warning("XGBoost not available for meta-learner")

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False


@dataclass
class StackingConfig:
    """Configuration for stacking ensemble training.

    Attributes:
        walk_forward: Walk-forward validation config
        meta_learner_type: 'xgboost' or 'lightgbm' or 'ridge'
        meta_learner_params: Parameters for meta-learner
        include_regime_features: Whether to include regime features
        include_agreement_features: Whether to include model agreement features
        include_recent_errors: Whether to include recent error features
        production_mode: If True, train on all data (after validation)
    """
    walk_forward: WalkForwardConfig = field(
        default_factory=lambda: WalkForwardConfig(
            mode=WalkForwardMode.ANCHORED,
            n_iterations=5,
        )
    )
    meta_learner_type: str = 'xgboost'
    meta_learner_params: Dict = field(default_factory=lambda: {
        'n_estimators': 200,
        'learning_rate': 0.05,
        'max_depth': 4,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_lambda': 0.1,
        'reg_alpha': 0.05,
        'early_stopping_rounds': 30,
    })
    include_regime_features: bool = True
    include_agreement_features: bool = True
    include_recent_errors: bool = False  # Experimental
    production_mode: bool = False


@dataclass
class BaseModelPredictions:
    """Container for base model OOF predictions.

    Attributes:
        model_name: Name of the base model
        oof_predictions: Out-of-fold predictions
        oof_indices: Indices corresponding to OOF predictions
        validation_wfe: Walk Forward Efficiency from validation
    """
    model_name: str
    oof_predictions: np.ndarray
    oof_indices: np.ndarray
    validation_wfe: float


@dataclass
class StackingEnsemble:
    """Trained stacking ensemble ready for inference.

    Attributes:
        base_models: Dict of trained base models
        meta_learner: Trained meta-learner
        feature_columns: Feature column names used for base models
        meta_feature_names: Feature names for meta-learner
        regime_detector: Optional regime detector
        scaler: Feature scaler
        metadata: Training metadata
    """
    base_models: Dict[str, Any]
    meta_learner: Any
    feature_columns: List[str]
    meta_feature_names: List[str]
    regime_detector: Optional[Any]
    scaler: RobustScaler
    metadata: Dict


class RegimeDetector:
    """
    Detect market regime for regime-aware ensemble weighting.

    Regimes:
    - high_volatility: Vol > 1.5 * median vol
    - normal_volatility: 0.7 * median < vol < 1.5 * median
    - low_volatility: Vol < 0.7 * median vol
    - trending_up: SMA20 > SMA50 and price > SMA20
    - trending_down: SMA20 < SMA50 and price < SMA20
    """

    def __init__(self, volatility_lookback: int = 20, trend_short: int = 20, trend_long: int = 50):
        self.volatility_lookback = volatility_lookback
        self.trend_short = trend_short
        self.trend_long = trend_long
        self.volatility_median = None

    def fit(self, returns: np.ndarray):
        """Fit detector on historical returns."""
        # Rolling volatility
        if len(returns) < self.volatility_lookback:
            self.volatility_median = np.std(returns) if len(returns) > 0 else 0.01
        else:
            rolling_vols = []
            for i in range(self.volatility_lookback, len(returns)):
                window = returns[i-self.volatility_lookback:i]
                rolling_vols.append(np.std(window))
            self.volatility_median = np.median(rolling_vols) if rolling_vols else 0.01

        return self

    def detect(self, features: np.ndarray, returns: Optional[np.ndarray] = None) -> Dict[str, float]:
        """Detect current regime.

        Returns:
            Dict with regime indicator values (0-1)
        """
        regime = {
            'vol_regime_high': 0.0,
            'vol_regime_normal': 0.0,
            'vol_regime_low': 0.0,
            'trend_bullish': 0.0,
            'trend_bearish': 0.0,
            'trend_neutral': 0.0,
        }

        if returns is not None and len(returns) >= self.volatility_lookback:
            current_vol = np.std(returns[-self.volatility_lookback:])

            if self.volatility_median > 0:
                if current_vol > 1.5 * self.volatility_median:
                    regime['vol_regime_high'] = 1.0
                elif current_vol < 0.7 * self.volatility_median:
                    regime['vol_regime_low'] = 1.0
                else:
                    regime['vol_regime_normal'] = 1.0

        # Default to normal if not enough data
        if sum(regime.values()) == 0:
            regime['vol_regime_normal'] = 1.0
            regime['trend_neutral'] = 1.0

        return regime

    def to_features(self, regime: Dict[str, float]) -> np.ndarray:
        """Convert regime dict to feature array."""
        return np.array([
            regime.get('vol_regime_high', 0),
            regime.get('vol_regime_normal', 0),
            regime.get('vol_regime_low', 0),
            regime.get('trend_bullish', 0),
            regime.get('trend_bearish', 0),
            regime.get('trend_neutral', 0),
        ])


class StackingEnsembleTrainer:
    """
    Train stacking ensemble with XGBoost meta-learner.

    Two-stage training:
    1. Generate OOF predictions from base models via walk-forward CV
    2. Train meta-learner on stacked OOF predictions + regime features

    Example:
        >>> config = StackingConfig()
        >>> trainer = StackingEnsembleTrainer('AAPL', config)
        >>> ensemble = trainer.train(X, y, base_model_factories)
        >>> predictions = ensemble.predict(X_new)
    """

    def __init__(self, symbol: str, config: StackingConfig):
        self.symbol = symbol
        self.config = config
        self.walk_forward = WalkForwardValidator(config.walk_forward)
        self.regime_detector = RegimeDetector()
        self.feature_columns: List[str] = []

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_model_factories: Dict[str, Callable[[], Any]],
        feature_columns: Optional[List[str]] = None,
        returns: Optional[np.ndarray] = None,
    ) -> StackingEnsemble:
        """
        Train complete stacking ensemble.

        Args:
            X: Feature matrix (n_samples, n_features) or (n_samples, seq_len, n_features)
            y: Target vector (n_samples,)
            base_model_factories: Dict mapping model_name -> factory function
            feature_columns: Optional list of feature names
            returns: Optional returns for regime detection

        Returns:
            Trained StackingEnsemble ready for inference
        """
        self.feature_columns = feature_columns or []

        logger.info(f"Training stacking ensemble for {self.symbol}")
        logger.info(f"Base models: {list(base_model_factories.keys())}")
        logger.info(f"Samples: {len(X)}, Features: {X.shape[-1]}")

        # Fit regime detector
        if returns is not None:
            self.regime_detector.fit(returns)
        else:
            # Use y as returns proxy
            self.regime_detector.fit(y)

        # Stage 1: Generate OOF predictions from base models
        logger.info("\n=== Stage 1: Generating OOF Predictions ===")
        base_predictions = self._generate_oof_predictions(X, y, base_model_factories)

        # Validate we got predictions
        if not base_predictions:
            raise ValueError("No base model predictions generated")

        # Stage 2: Build meta-features
        logger.info("\n=== Stage 2: Building Meta-Features ===")
        meta_X, meta_feature_names = self._build_meta_features(
            base_predictions, X, y, returns
        )

        # Get valid indices (where we have OOF predictions)
        valid_mask = ~np.isnan(meta_X).any(axis=1)
        meta_X_valid = meta_X[valid_mask]
        y_valid = y[valid_mask]

        logger.info(f"Meta-features shape: {meta_X_valid.shape}")
        logger.info(f"Meta-feature names: {meta_feature_names}")

        # Stage 3: Train meta-learner
        logger.info("\n=== Stage 3: Training Meta-Learner ===")
        meta_learner = self._train_meta_learner(meta_X_valid, y_valid)

        # Stage 4: Train production base models (on all data)
        if self.config.production_mode:
            logger.info("\n=== Stage 4: Training Production Models ===")
            production_models = self._train_production_models(X, y, base_model_factories)
        else:
            # Use last fold's models as production models
            production_models = {name: None for name in base_model_factories.keys()}
            logger.info("Skipping production model training (validation mode)")

        # Scale features for consistency
        scaler = RobustScaler()
        scaler.fit(X.reshape(len(X), -1) if len(X.shape) > 2 else X)

        # Compute validation metrics
        aggregate_wfe = np.mean([bp.validation_wfe for bp in base_predictions])

        metadata = {
            'symbol': self.symbol,
            'n_samples': len(X),
            'n_base_models': len(base_model_factories),
            'base_model_names': list(base_model_factories.keys()),
            'aggregate_wfe': aggregate_wfe,
            'meta_learner_type': self.config.meta_learner_type,
            'production_mode': self.config.production_mode,
            'trained_at': datetime.now().isoformat(),
        }

        return StackingEnsemble(
            base_models=production_models,
            meta_learner=meta_learner,
            feature_columns=self.feature_columns,
            meta_feature_names=meta_feature_names,
            regime_detector=self.regime_detector,
            scaler=scaler,
            metadata=metadata,
        )

    def _generate_oof_predictions(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_model_factories: Dict[str, Callable[[], Any]],
    ) -> List[BaseModelPredictions]:
        """Generate OOF predictions for each base model using walk-forward CV."""
        predictions = []

        for model_name, factory in base_model_factories.items():
            logger.info(f"\nGenerating OOF predictions for {model_name}...")

            try:
                results = self.walk_forward.validate(
                    model_factory=factory,
                    X=X,
                    y=y,
                )

                predictions.append(BaseModelPredictions(
                    model_name=model_name,
                    oof_predictions=results.oof_predictions,
                    oof_indices=results.oof_indices,
                    validation_wfe=results.aggregate_wfe,
                ))

                logger.info(f"{model_name}: WFE={results.aggregate_wfe:.1f}%")

            except Exception as e:
                logger.error(f"Failed to generate OOF for {model_name}: {e}")
                continue

        return predictions

    def _build_meta_features(
        self,
        base_predictions: List[BaseModelPredictions],
        X: np.ndarray,
        y: np.ndarray,
        returns: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Build meta-feature matrix from base predictions and auxiliary features."""
        n_samples = len(X)
        features = []
        feature_names = []

        # 1. Base model predictions
        for bp in base_predictions:
            features.append(bp.oof_predictions.reshape(-1, 1))
            feature_names.append(f'pred_{bp.model_name}')

        # Stack base predictions
        meta_X = np.hstack(features)

        # 2. Model agreement features
        if self.config.include_agreement_features and len(base_predictions) > 1:
            # Get indices where we have all predictions
            preds = np.column_stack([bp.oof_predictions for bp in base_predictions])

            # Prediction std (disagreement)
            pred_std = np.nanstd(preds, axis=1, keepdims=True)
            meta_X = np.hstack([meta_X, pred_std])
            feature_names.append('prediction_std')

            # Sign agreement (1 if all same sign)
            signs = np.sign(preds)
            sign_agreement = (np.nanstd(signs, axis=1) == 0).astype(float).reshape(-1, 1)
            meta_X = np.hstack([meta_X, sign_agreement])
            feature_names.append('sign_agreement')

            # Max-min spread
            pred_range = np.nanmax(preds, axis=1) - np.nanmin(preds, axis=1)
            meta_X = np.hstack([meta_X, pred_range.reshape(-1, 1)])
            feature_names.append('max_min_spread')

        # 3. Regime features
        if self.config.include_regime_features:
            regime_features = np.zeros((n_samples, 6))

            # For each sample, detect regime using historical data up to that point
            for i in range(n_samples):
                if returns is not None and i > 20:
                    regime = self.regime_detector.detect(X[i], returns[:i])
                else:
                    regime = self.regime_detector.detect(X[i], y[:i] if i > 20 else None)

                regime_features[i] = self.regime_detector.to_features(regime)

            meta_X = np.hstack([meta_X, regime_features])
            feature_names.extend([
                'vol_regime_high', 'vol_regime_normal', 'vol_regime_low',
                'trend_bullish', 'trend_bearish', 'trend_neutral',
            ])

        return meta_X, feature_names

    def _train_meta_learner(
        self,
        meta_X: np.ndarray,
        y: np.ndarray,
    ) -> Any:
        """Train XGBoost meta-learner."""
        if self.config.meta_learner_type == 'xgboost' and XGB_AVAILABLE:
            return self._train_xgb_meta_learner(meta_X, y)
        elif self.config.meta_learner_type == 'lightgbm' and LGB_AVAILABLE:
            return self._train_lgb_meta_learner(meta_X, y)
        else:
            return self._train_ridge_meta_learner(meta_X, y)

    def _train_xgb_meta_learner(self, meta_X: np.ndarray, y: np.ndarray) -> xgb.XGBRegressor:
        """Train XGBoost meta-learner."""
        params = self.config.meta_learner_params.copy()
        early_stopping = params.pop('early_stopping_rounds', 30)

        # 80/20 split for early stopping
        split_idx = int(len(meta_X) * 0.8)
        X_train, X_val = meta_X[:split_idx], meta_X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        model = xgb.XGBRegressor(**params, early_stopping_rounds=early_stopping)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        logger.info(f"XGBoost meta-learner trained: {model.best_iteration} iterations")

        return model

    def _train_lgb_meta_learner(self, meta_X: np.ndarray, y: np.ndarray) -> lgb.LGBMRegressor:
        """Train LightGBM meta-learner."""
        params = self.config.meta_learner_params.copy()
        early_stopping = params.pop('early_stopping_rounds', 30)

        # 80/20 split for early stopping
        split_idx = int(len(meta_X) * 0.8)
        X_train, X_val = meta_X[:split_idx], meta_X[split_idx:]
        y_train, y_val = y[:split_idx], y[split_idx:]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=early_stopping)],
        )

        logger.info(f"LightGBM meta-learner trained")

        return model

    def _train_ridge_meta_learner(self, meta_X: np.ndarray, y: np.ndarray):
        """Train Ridge regression meta-learner (fallback)."""
        from sklearn.linear_model import Ridge

        model = Ridge(alpha=0.1)
        model.fit(meta_X, y)

        logger.info("Ridge meta-learner trained")

        return model

    def _train_production_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        base_model_factories: Dict[str, Callable[[], Any]],
    ) -> Dict[str, Any]:
        """Train production models on all data."""
        production_models = {}

        for model_name, factory in base_model_factories.items():
            logger.info(f"Training production {model_name}...")

            try:
                model = factory()

                # Use last 20% as validation for early stopping only
                split_idx = int(len(X) * 0.8)
                X_train, X_val = X[:split_idx], X[split_idx:]
                y_train, y_val = y[:split_idx], y[split_idx:]

                try:
                    model.fit(X_train, y_train, validation_data=(X_val, y_val))
                except TypeError:
                    model.fit(X_train, y_train)

                production_models[model_name] = model

            except Exception as e:
                logger.error(f"Failed to train production {model_name}: {e}")
                production_models[model_name] = None

        return production_models

    def save(self, ensemble: StackingEnsemble, output_dir: Path):
        """Save trained ensemble to disk."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save meta-learner
        meta_path = output_dir / 'meta_learner.pkl'
        with open(meta_path, 'wb') as f:
            pickle.dump(ensemble.meta_learner, f)

        # Save regime detector
        regime_path = output_dir / 'regime_detector.pkl'
        with open(regime_path, 'wb') as f:
            pickle.dump(ensemble.regime_detector, f)

        # Save scaler
        scaler_path = output_dir / 'meta_scaler.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump(ensemble.scaler, f)

        # Save metadata
        meta_data_path = output_dir / 'stacking_metadata.pkl'
        with open(meta_data_path, 'wb') as f:
            pickle.dump({
                'feature_columns': ensemble.feature_columns,
                'meta_feature_names': ensemble.meta_feature_names,
                'metadata': ensemble.metadata,
            }, f)

        logger.info(f"Ensemble saved to {output_dir}")


def main():
    """CLI entry point for training stacking ensemble."""
    parser = argparse.ArgumentParser(description='Train stacking ensemble')
    parser.add_argument('--symbol', type=str, required=True, help='Stock symbol')
    parser.add_argument('--production', action='store_true', help='Production mode')
    parser.add_argument('--output-dir', type=str, default='saved_models',
                        help='Output directory')

    args = parser.parse_args()

    # This is a placeholder - actual training requires loading data and base models
    logger.info(f"Would train stacking ensemble for {args.symbol}")
    logger.info("Note: Full training requires base model factories and data loading")
    logger.info("Use the ProductionPipeline class for complete workflow")


if __name__ == '__main__':
    main()
