"""
Stacking Predictor for Production Inference

Nuclear Redesign: Production inference with trained stacking ensemble.
Replaces argument-based fusion modes with trained meta-learner.

Usage:
    >>> predictor = StackingPredictor('AAPL')
    >>> result = predictor.predict(features)
    >>> print(f"Prediction: {result.prediction}, Confidence: {result.confidence}")

Author: AI-Stocks Nuclear Redesign
Date: December 2025
"""

import sys
import pickle
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple

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


@dataclass
class PredictionResult:
    """Result from stacking predictor.

    Attributes:
        prediction: Final predicted return
        confidence: Prediction confidence (0-1)
        position_size: Recommended position size (-0.5 to 1.0)
        regime: Current detected regime
        base_predictions: Dict of individual base model predictions
        meta_features: Meta-features used for final prediction
    """
    prediction: float
    confidence: float
    position_size: float
    regime: Dict[str, float]
    base_predictions: Dict[str, float]
    meta_features: Optional[np.ndarray] = None

    def summary(self) -> str:
        """Generate summary string."""
        direction = "LONG" if self.prediction > 0 else "SHORT" if self.prediction < 0 else "FLAT"
        regime_str = max(self.regime.items(), key=lambda x: x[1])[0] if self.regime else "unknown"

        lines = [
            f"Prediction: {self.prediction:+.4f} ({direction})",
            f"Confidence: {self.confidence:.2%}",
            f"Position Size: {self.position_size:+.2f}",
            f"Regime: {regime_str}",
            "Base Predictions:",
        ]
        for name, pred in self.base_predictions.items():
            lines.append(f"  {name}: {pred:+.4f}")

        return "\n".join(lines)


class StackingPredictor:
    """
    Production inference with stacking ensemble.

    Loads trained stacking ensemble and generates predictions using:
    1. Base model predictions (LSTM, XGBoost, LightGBM)
    2. Regime detection
    3. Model agreement features
    4. XGBoost meta-learner

    Example:
        >>> predictor = StackingPredictor('AAPL')
        >>> result = predictor.predict(latest_features)
        >>> if result.confidence > 0.6:
        ...     execute_trade(result.position_size)
    """

    # Position sizing constants
    LONG_SCALE = 50.0   # +1% prediction -> 50% long
    SHORT_SCALE = 25.0  # -1% prediction -> 25% short
    MAX_LONG = 1.0      # Maximum 100% long
    MAX_SHORT = 0.5     # Maximum 50% short

    def __init__(
        self,
        symbol: str,
        model_dir: Optional[Path] = None,
        load_base_models: bool = True,
    ):
        """
        Initialize stacking predictor.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            model_dir: Directory containing saved models
            load_base_models: Whether to load base models (False for meta-only)
        """
        self.symbol = symbol
        self.model_dir = model_dir or Path(PROJECT_ROOT) / 'saved_models' / symbol

        self.meta_learner = None
        self.regime_detector = None
        self.scaler = None
        self.feature_columns: List[str] = []
        self.meta_feature_names: List[str] = []
        self.base_models: Dict[str, Any] = {}
        self.metadata: Dict = {}

        self._load_ensemble(load_base_models)

    def _load_ensemble(self, load_base_models: bool = True):
        """Load trained ensemble from disk."""
        stacking_dir = self.model_dir / 'stacking'

        if not stacking_dir.exists():
            # Try flat structure
            stacking_dir = self.model_dir
            logger.warning(f"Using flat model directory: {stacking_dir}")

        # Load meta-learner
        meta_path = stacking_dir / 'meta_learner.pkl'
        if meta_path.exists():
            with open(meta_path, 'rb') as f:
                self.meta_learner = pickle.load(f)
            logger.info(f"Loaded meta-learner from {meta_path}")
        else:
            logger.warning(f"Meta-learner not found at {meta_path}")

        # Load regime detector
        regime_path = stacking_dir / 'regime_detector.pkl'
        if regime_path.exists():
            with open(regime_path, 'rb') as f:
                self.regime_detector = pickle.load(f)
            logger.info(f"Loaded regime detector from {regime_path}")

        # Load scaler
        scaler_path = stacking_dir / 'meta_scaler.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)

        # Load metadata
        meta_data_path = stacking_dir / 'stacking_metadata.pkl'
        if meta_data_path.exists():
            with open(meta_data_path, 'rb') as f:
                data = pickle.load(f)
                self.feature_columns = data.get('feature_columns', [])
                self.meta_feature_names = data.get('meta_feature_names', [])
                self.metadata = data.get('metadata', {})

        # Load base models if requested
        if load_base_models:
            self._load_base_models()

    def _load_base_models(self):
        """Load base models (LSTM, xLSTM, XGBoost, LightGBM) and their reliability scores."""
        # Initialize model reliability scores (used for dynamic weighting)
        self.model_reliability = {}

        try:
            from utils.model_paths import ModelPaths
            import tensorflow as tf
            paths = ModelPaths(self.symbol)

            # LSTM/Transformer Regressor
            if paths.regressor.model.exists():
                self.base_models['lstm'] = tf.keras.models.load_model(
                    str(paths.regressor.model),
                    compile=False,
                )
                logger.info(f"Loaded LSTM model from {paths.regressor.model}")
                # Load reliability score from metadata
                if paths.regressor.metadata.exists():
                    with open(paths.regressor.metadata, 'rb') as f:
                        meta = pickle.load(f)
                        self.model_reliability['lstm'] = meta.get('wfe_score', 0.5)

            # xLSTM-TS Model (December 2025 addition)
            xlstm_path = self.model_dir / 'xlstm' / 'model.keras'
            if xlstm_path.exists():
                self.base_models['xlstm'] = tf.keras.models.load_model(
                    str(xlstm_path),
                    compile=False,
                )
                logger.info(f"Loaded xLSTM model from {xlstm_path}")
                # Load reliability score
                xlstm_meta_path = self.model_dir / 'xlstm' / 'metadata.pkl'
                if xlstm_meta_path.exists():
                    with open(xlstm_meta_path, 'rb') as f:
                        meta = pickle.load(f)
                        self.model_reliability['xlstm'] = meta.get('wfe_score', 0.4)

            # XGBoost
            xgb_path = self.model_dir / 'gbm' / 'xgboost_model.pkl'
            if not xgb_path.exists():
                xgb_path = self.model_dir / f'{self.symbol}_xgboost_model.pkl'
            if xgb_path.exists():
                with open(xgb_path, 'rb') as f:
                    self.base_models['xgboost'] = pickle.load(f)
                logger.info(f"Loaded XGBoost model from {xgb_path}")
                # Load reliability score from GBM metadata
                gbm_meta_path = self.model_dir / 'gbm' / 'training_metadata.json'
                if gbm_meta_path.exists():
                    import json
                    with open(gbm_meta_path, 'r') as f:
                        meta = json.load(f)
                        val_ic = meta.get('validation', {}).get('xgb', {}).get('ic', 0.0)
                        # Convert IC to reliability score (IC of 0.1 = 60% reliability)
                        self.model_reliability['xgboost'] = 0.5 + abs(val_ic) * 1.0

            # LightGBM
            lgb_path = self.model_dir / 'gbm' / 'lightgbm_model.pkl'
            if not lgb_path.exists():
                lgb_path = self.model_dir / f'{self.symbol}_lightgbm_model.pkl'
            if lgb_path.exists():
                with open(lgb_path, 'rb') as f:
                    self.base_models['lightgbm'] = pickle.load(f)
                logger.info(f"Loaded LightGBM model from {lgb_path}")
                # Load reliability score
                if gbm_meta_path.exists():
                    import json
                    with open(gbm_meta_path, 'r') as f:
                        meta = json.load(f)
                        val_ic = meta.get('validation', {}).get('lgb', {}).get('ic', 0.0)
                        self.model_reliability['lightgbm'] = 0.5 + abs(val_ic) * 1.0

            logger.info(f"Model reliability scores: {self.model_reliability}")

        except Exception as e:
            logger.warning(f"Error loading base models: {e}")

    def predict(
        self,
        features: np.ndarray,
        returns: Optional[np.ndarray] = None,
    ) -> PredictionResult:
        """
        Generate stacked prediction.

        Args:
            features: Feature matrix (seq_len, n_features) for LSTM
                      or (n_features,) for GBM
            returns: Optional historical returns for regime detection

        Returns:
            PredictionResult with prediction, confidence, and position size
        """
        # Ensure proper shape
        if len(features.shape) == 1:
            features = features.reshape(1, -1)
        elif len(features.shape) == 3:
            # (batch, seq_len, features) - take last sequence
            features = features[-1:]

        # Get base model predictions
        base_predictions = self._get_base_predictions(features)

        if not base_predictions:
            logger.warning("No base model predictions available")
            return PredictionResult(
                prediction=0.0,
                confidence=0.0,
                position_size=0.0,
                regime={},
                base_predictions={},
            )

        # Detect regime
        regime = self._detect_regime(features, returns)

        # Build meta-features
        meta_features = self._build_meta_features(base_predictions, regime)

        # Get final prediction from meta-learner
        if self.meta_learner is not None:
            prediction = float(self.meta_learner.predict(meta_features.reshape(1, -1))[0])
        else:
            # Fall back to simple average
            prediction = float(np.mean(list(base_predictions.values())))
            logger.warning("Using simple average (no meta-learner)")

        # Calculate confidence
        confidence = self._calculate_confidence(base_predictions, regime)

        # Calculate position size
        position_size = self._calculate_position_size(prediction, confidence)

        return PredictionResult(
            prediction=prediction,
            confidence=confidence,
            position_size=position_size,
            regime=regime,
            base_predictions=base_predictions,
            meta_features=meta_features,
        )

    def _get_base_predictions(self, features: np.ndarray) -> Dict[str, float]:
        """Get predictions from all base models (LSTM, xLSTM, XGBoost, LightGBM)."""
        predictions = {}

        # LSTM prediction
        if 'lstm' in self.base_models and self.base_models['lstm'] is not None:
            try:
                # LSTM expects (batch, seq_len, features)
                if len(features.shape) == 2:
                    lstm_input = features.reshape(1, *features.shape)
                else:
                    lstm_input = features

                pred = self.base_models['lstm'].predict(lstm_input, verbose=0)
                predictions['lstm'] = float(pred.flatten()[-1])
            except Exception as e:
                logger.warning(f"LSTM prediction failed: {e}")

        # xLSTM prediction (December 2025 addition)
        if 'xlstm' in self.base_models and self.base_models['xlstm'] is not None:
            try:
                # xLSTM also expects (batch, seq_len, features)
                if len(features.shape) == 2:
                    xlstm_input = features.reshape(1, *features.shape)
                else:
                    xlstm_input = features

                pred = self.base_models['xlstm'].predict(xlstm_input, verbose=0)
                predictions['xlstm'] = float(pred.flatten()[-1])
            except Exception as e:
                logger.warning(f"xLSTM prediction failed: {e}")

        # XGBoost prediction
        if 'xgboost' in self.base_models and self.base_models['xgboost'] is not None:
            try:
                # XGBoost expects (n_samples, n_features)
                gbm_input = features.reshape(1, -1) if len(features.shape) == 1 else features
                if len(gbm_input.shape) == 3:
                    gbm_input = gbm_input.reshape(gbm_input.shape[0], -1)
                    gbm_input = gbm_input[-1:]  # Take last sample

                pred = self.base_models['xgboost'].predict(gbm_input)
                predictions['xgboost'] = float(pred[-1])
            except Exception as e:
                logger.warning(f"XGBoost prediction failed: {e}")

        # LightGBM prediction
        if 'lightgbm' in self.base_models and self.base_models['lightgbm'] is not None:
            try:
                gbm_input = features.reshape(1, -1) if len(features.shape) == 1 else features
                if len(gbm_input.shape) == 3:
                    gbm_input = gbm_input.reshape(gbm_input.shape[0], -1)
                    gbm_input = gbm_input[-1:]

                pred = self.base_models['lightgbm'].predict(gbm_input)
                predictions['lightgbm'] = float(pred[-1])
            except Exception as e:
                logger.warning(f"LightGBM prediction failed: {e}")

        return predictions

    def _detect_regime(
        self,
        features: np.ndarray,
        returns: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Detect current market regime."""
        if self.regime_detector is not None:
            return self.regime_detector.detect(features, returns)

        # Default regime
        return {
            'vol_regime_high': 0.0,
            'vol_regime_normal': 1.0,
            'vol_regime_low': 0.0,
            'trend_bullish': 0.0,
            'trend_bearish': 0.0,
            'trend_neutral': 1.0,
        }

    def _build_meta_features(
        self,
        base_predictions: Dict[str, float],
        regime: Dict[str, float],
    ) -> np.ndarray:
        """Build meta-feature vector for meta-learner.

        December 2025 update: Added xLSTM predictions and model reliability scores
        for dynamic weighting based on each model's validation performance.
        """
        features = []

        # Base model predictions (4 models: lstm, xlstm, xgboost, lightgbm)
        for name in ['lstm', 'xlstm', 'xgboost', 'lightgbm']:
            features.append(base_predictions.get(name, 0.0))

        # Model agreement features
        preds = list(base_predictions.values())
        if len(preds) > 1:
            features.append(np.std(preds))  # prediction_std
            signs = [1 if p > 0 else -1 for p in preds]
            features.append(float(len(set(signs)) == 1))  # sign_agreement
            features.append(max(preds) - min(preds))  # max_min_spread
        else:
            features.extend([0.0, 1.0, 0.0])

        # Regime features
        features.append(regime.get('vol_regime_high', 0))
        features.append(regime.get('vol_regime_normal', 0))
        features.append(regime.get('vol_regime_low', 0))
        features.append(regime.get('trend_bullish', 0))
        features.append(regime.get('trend_bearish', 0))
        features.append(regime.get('trend_neutral', 0))

        # Model reliability scores (December 2025 - dynamic weighting)
        # These allow the meta-learner to weight models based on their validation performance
        reliability = getattr(self, 'model_reliability', {})
        features.append(reliability.get('lstm', 0.5))
        features.append(reliability.get('xlstm', 0.4))
        features.append(reliability.get('xgboost', 0.5))
        features.append(reliability.get('lightgbm', 0.5))

        return np.array(features)

    def _calculate_confidence(
        self,
        base_predictions: Dict[str, float],
        regime: Dict[str, float],
    ) -> float:
        """Calculate prediction confidence based on model agreement."""
        if len(base_predictions) < 2:
            return 0.5

        preds = list(base_predictions.values())

        # Model agreement factor
        pred_std = np.std(preds)
        agreement = np.exp(-pred_std / 0.01)  # High agreement = low std

        # Sign agreement
        signs = [np.sign(p) for p in preds]
        sign_agreement = 1.0 if len(set(signs)) == 1 else 0.5

        # Regime factor (lower confidence in high volatility)
        vol_factor = 1.0
        if regime.get('vol_regime_high', 0) > 0.5:
            vol_factor = 0.7
        elif regime.get('vol_regime_low', 0) > 0.5:
            vol_factor = 1.1  # Slightly boost in calm markets

        confidence = agreement * sign_agreement * vol_factor
        return float(np.clip(confidence, 0.0, 1.0))

    def _calculate_position_size(
        self,
        prediction: float,
        confidence: float,
    ) -> float:
        """Calculate recommended position size.

        Position sizing:
        - +1% prediction -> 50% long (LONG_SCALE=50)
        - +2% prediction -> 100% long (capped)
        - -1% prediction -> 25% short (SHORT_SCALE=25)
        - -2% prediction -> 50% short (capped)

        Adjusted by confidence.
        """
        if prediction > 0:
            raw_position = prediction * 100 * self.LONG_SCALE / 100
            position = min(raw_position, self.MAX_LONG)
        else:
            raw_position = prediction * 100 * self.SHORT_SCALE / 100
            position = max(raw_position, -self.MAX_SHORT)

        # Apply confidence adjustment
        position *= confidence

        return float(np.clip(position, -self.MAX_SHORT, self.MAX_LONG))

    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """Get feature importance from meta-learner."""
        if self.meta_learner is None:
            return None

        try:
            importances = self.meta_learner.feature_importances_
            return dict(zip(self.meta_feature_names, importances))
        except AttributeError:
            return None


def load_stacking_predictor(symbol: str) -> Optional[StackingPredictor]:
    """
    Load stacking predictor for a symbol.

    Args:
        symbol: Stock symbol

    Returns:
        StackingPredictor or None if not available
    """
    try:
        return StackingPredictor(symbol)
    except Exception as e:
        logger.error(f"Failed to load stacking predictor for {symbol}: {e}")
        return None
