"""Hybrid ensemble predictor utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAVED_MODELS_DIR = PROJECT_ROOT / "saved_models"

from data.data_fetcher import fetch_stock_data  # noqa: E402
from data.feature_engineer import engineer_features  # noqa: E402
from data.feature_engineer import get_feature_columns, EXPECTED_FEATURE_COUNT  # noqa: E402
import logging
import pickle
from inference.horizon_weighting import (  # noqa: E402
    compute_softmax_weights,
    compute_inverse_mse_weights,
    get_default_weights,
)
from utils.losses import register_custom_objects  # noqa: E402
from utils.model_paths import ModelPaths, find_model_path, get_legacy_regressor_paths, get_legacy_classifier_paths  # noqa: E402

# GBM Integration
try:
    from inference.load_gbm_models import load_gbm_models, predict_with_gbm, GBMModelBundle
    GBM_AVAILABLE = True
except ImportError:
    GBM_AVAILABLE = False
    GBMModelBundle = None  # type: ignore

register_custom_objects()


def fuse_predictions_regressor_only(regressor_preds: np.ndarray, atr_percent: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Pure regressor strategy - ignores classifiers completely.
    
    Uses regressor predictions directly to generate positions with the following logic:
    - Predict +1% → 50% long position
    - Predict +2% → 100% long position
    - Predict -1% → 25% short position
    - Predict -2% → 50% short position
    
    Args:
        regressor_preds: Array of predicted returns (e.g., 0.002 = 0.2% predicted gain)
        atr_percent: Optional volatility adjustment (ATR as percentage of price)
    
    Returns:
        positions: Array of position sizes in [-0.5, 1.0]
    """
    regressor_preds = np.asarray(regressor_preds, dtype=float)
    positions = np.zeros_like(regressor_preds)
    
    # Scale regressor predictions to position sizes
    # Predict +1% → 50% long position
    # Predict +2% → 100% long position
    # Predict -1% → 25% short position
    # Predict -2% → 50% short position
    LONG_SCALE = 50.0   # Multiply by 50: 0.02 (2%) → 1.0 (100% long)
    SHORT_SCALE = 25.0  # Multiply by 25: -0.02 (-2%) → -0.5 (50% short)
    
    for i in range(len(regressor_preds)):
        pred = regressor_preds[i]
        
        if pred > 0:
            # Long position
            position = pred * LONG_SCALE
            position = np.clip(position, 0, 1.0)
        else:
            # Short position
            position = pred * SHORT_SCALE
            position = np.clip(position, -0.5, 0)
        
        # Optional: reduce position size in high volatility
        if atr_percent is not None and i < len(atr_percent) and atr_percent[i] > 0.03:  # 3% ATR
            position *= 0.7  # Reduce by 30%
        
        positions[i] = position
    
    return positions


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, (np.floating, np.integer)):
            return float(value)
        if isinstance(value, (float, int)):
            return float(value)
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return None


class EnsemblePredictor:
    """Load hybrid ensemble artifacts and generate predictions."""

    def __init__(self, symbol: str, risk_profile: str = "conservative", weighting_method: str = "softmax") -> None:
        self.symbol = symbol
        self.risk_profile = risk_profile

        print(f"   📂 SAVED_MODELS_DIR: {SAVED_MODELS_DIR}")
        
        metadata_path = SAVED_MODELS_DIR / f"{symbol}_ensemble_metadata.pkl"
        print(f"   📂 Looking for metadata: {metadata_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        self.metadata = joblib.load(metadata_path)
        if not str(self.metadata.get("model_version", "")).startswith("hybrid_"):
            raise ValueError("Metadata is not from the hybrid trainer")

        self.horizons: List[int] = list(self.metadata["horizons"])
        # Prefer feature list saved on disk (feature_columns.pkl or top-level feature files)
        self.selected_features: List[str] = list(self.metadata.get("selected_features", []))
        logger = logging.getLogger(__name__)

        # Try several locations for saved feature lists
        saved_feature_cols = None
        paths = ModelPaths(symbol)
        
        # 1) New structure: saved_models/{symbol}/feature_columns.pkl
        if paths.feature_columns.exists():
            try:
                saved_feature_cols = pickle.load(open(paths.feature_columns, 'rb'))
            except Exception:
                saved_feature_cols = None
        
        # 2) Legacy folder: saved_models/{symbol}/feature_columns.pkl (already checked above)
        if saved_feature_cols is None:
            candidate = SAVED_MODELS_DIR / symbol / 'feature_columns.pkl'
            if candidate.exists():
                try:
                    saved_feature_cols = pickle.load(open(candidate, 'rb'))
                except Exception:
                    saved_feature_cols = None

        # 3) Legacy top-level files like saved_models/{symbol}_*_feature*.pkl
        if saved_feature_cols is None:
            for p in SAVED_MODELS_DIR.iterdir():
                if p.is_file() and p.stem.lower().startswith(symbol.lower()) and 'feature' in p.name.lower():
                    try:
                        saved_feature_cols = pickle.load(open(p, 'rb'))
                        break
                    except Exception:
                        continue

        if saved_feature_cols is not None and isinstance(saved_feature_cols, (list, tuple)):
            saved_feature_cols = list(saved_feature_cols)
            if len(saved_feature_cols) != EXPECTED_FEATURE_COUNT:
                raise ValueError(
                    f"Model feature mismatch for {symbol}!\n"
                    f"Model trained with: {len(saved_feature_cols)} features\n"
                    f"System expects: {EXPECTED_FEATURE_COUNT} features\n"
                    f"Action: Retrain model with latest feature set"
                )
            self.selected_features = saved_feature_cols
            logger.info(f"Using saved feature list for {symbol} ({len(self.selected_features)} features)")
        self.sequence_length: int = int(self.metadata.get("sequence_length", 60))

        # NEW: Compute dynamic weights based on validation metrics
        if "validation_metrics" in self.metadata:
            val_metrics = self.metadata["validation_metrics"]
            
            if weighting_method == "softmax":
                r2_scores = {h: val_metrics[h]["r2"] for h in val_metrics}
                self.weights = compute_softmax_weights(r2_scores, temperature=5.0)
            
            elif weighting_method == "inverse_mse":
                mse_scores = {h: val_metrics[h]["mse"] for h in val_metrics}
                self.weights = compute_inverse_mse_weights(mse_scores)
            
            else:  # "equal" or fallback
                self.weights = get_default_weights(self.horizons)
            
            print(f"   🎯 Using {weighting_method} weights:")
            for h, w in self.weights.items():
                r2 = val_metrics[h]["r2"]
                print(f"      {h}: {w*100:.1f}% (R²={r2:.4f})")
        else:
            # Fallback: use stored weights or equal weights
            self.weights = dict(self.metadata.get("weights", {}))
            if not self.weights:
                self.weights = get_default_weights(self.horizons)
            print(f"   ⚠️  No validation metrics found, using stored/equal weights")

        print(f"   🎯 Horizons: {self.horizons}")
        print(f"   🎯 Risk profile: {risk_profile}")
        
        self.regressors: Dict[str, keras.Model] = {}
        self.classifiers: Dict[str, keras.Model] = {}
        self.scalers: Dict[str, object] = {}

        for horizon in self.horizons:
            horizon_key = f"{horizon}d"
            reg_path = SAVED_MODELS_DIR / f"{symbol}_{horizon}d_regressor.keras"
            class_path = SAVED_MODELS_DIR / f"{symbol}_{horizon}d_{risk_profile}_classifier.keras"
            scaler_path = SAVED_MODELS_DIR / f"{symbol}_{horizon}d_scaler.pkl"

            print(f"   📂 Loading {horizon_key}:")
            print(f"      Reg: {reg_path}")
            print(f"      Class: {class_path}")
            print(f"      Scaler: {scaler_path}")
            
            if not (reg_path.exists() and class_path.exists() and scaler_path.exists()):
                raise FileNotFoundError(
                    f"Missing artifacts for {horizon_key}. Expected {reg_path}, {class_path}, {scaler_path}."
                )

            self.regressors[horizon_key] = keras.models.load_model(reg_path)
            self.classifiers[horizon_key] = keras.models.load_model(class_path)
            self.scalers[horizon] = joblib.load(scaler_path)
        
        # Load GBM models if available
        self.gbm_bundle: Optional[GBMModelBundle] = None
        self.gbm_weight: float = 0.0  # Weight for GBM in fusion (0 = disabled)
        
        if GBM_AVAILABLE:
            try:
                gbm_bundle, gbm_meta = load_gbm_models(symbol)
                if gbm_bundle is not None and gbm_bundle.any_loaded():
                    self.gbm_bundle = gbm_bundle
                    # Default: give GBM 20% weight when available
                    self.gbm_weight = 0.2
                    print(f"   🌲 Loaded GBM models: {gbm_bundle.get_available_models()}")
                    print(f"   🌲 GBM fusion weight: {self.gbm_weight*100:.0f}%")
            except Exception as e:
                logger.warning(f"Failed to load GBM models: {e}")
                self.gbm_bundle = None

    def _prepare_sequences(self, df: pd.DataFrame) -> Tuple[Dict[str, np.ndarray], int]:
        sequences: Dict[str, np.ndarray] = {}
        # Ensure the dataframe contains the expected feature columns (autofill missing with zeros)
        canonical_features = get_feature_columns(include_sentiment=True)
        # If selected_features missing some columns, warn and auto-fill
        missing = set(self.selected_features) - set(df.columns)
        if missing:
            logging.getLogger(__name__).warning(
                f"Auto-filling {len(missing)} missing model features with zeros: {missing}"
            )
            for col in missing:
                df[col] = 0.0

        # Also ensure canonical features exist in df when possible (best-effort)
        missing_canonical = set(canonical_features) - set(df.columns)
        if missing_canonical:
            logging.getLogger(__name__).warning(f"TFT/Inference: missing canonical features: {missing_canonical}")

        # Reorder to match model's expected feature order
        df_feat = df.copy()
        df_feat = df_feat.reindex(columns=self.selected_features, fill_value=0.0)

        feature_values = df_feat.dropna()
        if len(feature_values) < self.sequence_length:
            raise ValueError("Not enough rows after feature engineering to build sequence")

        last_valid_idx = feature_values.index[-1]
        feature_matrix = feature_values.values
        for horizon in self.horizons:
            scaler = self.scalers[horizon]
            scaled = scaler.transform(feature_matrix)
            X_input = scaled[-self.sequence_length :].reshape(1, self.sequence_length, -1)
            # Diagnostic logging and sanity check before returning sequences
            logger = logging.getLogger(__name__)
            logger.info(f"Inference sequences for {self.symbol}: {X_input.shape} (samples, seq_len, features)")
            assert X_input.shape[2] == len(self.selected_features), (
                f"Feature dimension mismatch: expected {len(self.selected_features)}, got {X_input.shape[2]}"
            )
            # Also assert matches global EXPECTED_FEATURE_COUNT where applicable
            if len(self.selected_features) == EXPECTED_FEATURE_COUNT:
                assert X_input.shape[2] == EXPECTED_FEATURE_COUNT, (
                    f"Model expects {EXPECTED_FEATURE_COUNT} features"
                )

            sequences[f"{horizon}d"] = X_input
        return sequences, last_valid_idx
    
    def _extract_sentiment_features(self, latest_row: pd.Series) -> Dict[str, float]:
        """
        Extract sentiment features from latest data row.
        
        Args:
            latest_row: Latest row from DataFrame with features
        
        Returns:
            Dictionary with sentiment feature values (0.0 if not available)
        """
        sentiment_features = {
            'sentiment_mean': 0.0,
            'sentiment_momentum': 0.0,
            'news_volume': 0.0,
            'price_up_sentiment_down': 0.0,
            'price_down_sentiment_up': 0.0,
            'bullish_regime': 0.0,
            'bearish_regime': 0.0,
            'sentiment_volatility': 0.0,
            'high_volume_positive': 0.0,
            'high_volume_negative': 0.0,
        }
        
        # Extract values if they exist
        for key in sentiment_features.keys():
            if key in latest_row:
                value = latest_row[key]
                if pd.notna(value):
                    sentiment_features[key] = float(value)
        
        return sentiment_features
    
    def _get_gbm_prediction(self, df: pd.DataFrame, last_valid_idx: int) -> Optional[Dict[str, float]]:
        """
        Get GBM prediction for the latest row.
        
        Args:
            df: DataFrame with engineered features
            last_valid_idx: Index of the last valid row
        
        Returns:
            Dict with GBM predictions or None if not available
        """
        if self.gbm_bundle is None or not self.gbm_bundle.any_loaded():
            return None
        
        try:
            # Get feature values for GBM (uses same features as LSTM)
            feature_cols = self.gbm_bundle.feature_columns
            if not all(col in df.columns for col in feature_cols):
                missing = [col for col in feature_cols if col not in df.columns]
                logging.getLogger(__name__).warning(f"GBM missing features: {missing[:5]}...")
                return None
            
            # Get the latest row features
            latest_features = df.loc[[last_valid_idx], feature_cols].values
            
            # Get predictions from all available models
            gbm_preds = predict_with_gbm(self.gbm_bundle, latest_features, return_both=True)
            
            result = {}
            for model_name, pred in gbm_preds.items():
                result[f'{model_name}_return'] = float(pred[0])
            
            # Compute average if multiple models
            if len(gbm_preds) > 1:
                result['gbm_avg_return'] = float(np.mean(list(pred[0] for pred in gbm_preds.values())))
            elif len(gbm_preds) == 1:
                result['gbm_avg_return'] = float(list(gbm_preds.values())[0][0])
            
            return result
        
        except Exception as e:
            logging.getLogger(__name__).warning(f"GBM prediction failed: {e}")
            return None
    
    def _apply_sentiment_adjustment(
        self, 
        recommendation: str, 
        confidence: float,
        sentiment: Dict[str, float]
    ) -> Tuple[str, float, Dict[str, str]]:
        """
        Apply sentiment-based adjustments to position sizing and confidence.
        
        This adjusts model predictions based on news sentiment to improve signal quality:
        - Boosts positions when sentiment aligns with prediction
        - Reduces positions when sentiment contradicts prediction
        - Filters low-reliability signals (low news volume)
        - Applies sentiment regime overlay
        
        Args:
            recommendation: Model recommendation (BUY/HOLD/SELL)
            confidence: Model confidence (0-1)
            sentiment: Dictionary with sentiment features
        
        Returns:
            Tuple of (adjusted_recommendation, adjusted_confidence, adjustments_made)
        """
        sentiment_mean = sentiment['sentiment_mean']
        sentiment_momentum = sentiment['sentiment_momentum']
        news_volume = sentiment['news_volume']
        price_up_sentiment_down = sentiment['price_up_sentiment_down']
        price_down_sentiment_up = sentiment['price_down_sentiment_up']
        bullish_regime = sentiment['bullish_regime']
        bearish_regime = sentiment['bearish_regime']
        
        adjustments = []
        confidence_multiplier = 1.0
        
        # ===================================================================
        # 1. NEWS VOLUME FILTERING
        # ===================================================================
        # Require minimum news volume for sentiment signals to be reliable
        
        if news_volume < 5:
            # Low news volume → sentiment unreliable, use original prediction
            adjustments.append("Low news volume: sentiment ignored")
            return recommendation, confidence, {"adjustments": ", ".join(adjustments)}
        
        # High news volume → sentiment very reliable, boost adjustments
        volume_boost = 1.0
        if news_volume > 20:
            volume_boost = 2.0
            adjustments.append(f"High news volume ({news_volume:.0f}): doubled sentiment impact")
        elif news_volume >= 5:
            adjustments.append(f"Adequate news volume ({news_volume:.0f}): normal sentiment impact")
        
        # ===================================================================
        # 2. SENTIMENT ALIGNMENT BOOST/REDUCTION
        # ===================================================================
        
        if recommendation == "BUY":
            # Boost BUY when sentiment is positive
            if sentiment_mean > 0.3 and sentiment_momentum > 0.1:
                # Strong positive sentiment + positive momentum
                boost = 1.0 + min(0.3, sentiment_mean * 0.5) * volume_boost
                confidence_multiplier *= boost
                adjustments.append(f"Positive sentiment aligned with BUY: +{(boost-1)*100:.1f}% confidence")
            
            elif sentiment_mean < -0.3:
                # Negative sentiment contradicts BUY
                reduction = 0.5
                confidence_multiplier *= reduction
                adjustments.append(f"Negative sentiment contradicts BUY: -{(1-reduction)*100:.1f}% confidence")
                
                # Strong contradiction may flip to HOLD
                if sentiment_mean < -0.5 and confidence < 0.7:
                    adjustments.append("Strong negative sentiment: downgraded BUY → HOLD")
                    recommendation = "HOLD"
        
        elif recommendation == "SELL":
            # Boost SELL when sentiment is negative
            if sentiment_mean < -0.3 and sentiment_momentum < -0.1:
                # Strong negative sentiment + negative momentum
                boost = 1.0 + min(0.3, abs(sentiment_mean) * 0.5) * volume_boost
                confidence_multiplier *= boost
                adjustments.append(f"Negative sentiment aligned with SELL: +{(boost-1)*100:.1f}% confidence")
            
            elif sentiment_mean > 0.3:
                # Positive sentiment contradicts SELL
                reduction = 0.5
                confidence_multiplier *= reduction
                adjustments.append(f"Positive sentiment contradicts SELL: -{(1-reduction)*100:.1f}% confidence")
                
                # Strong contradiction may flip to HOLD
                if sentiment_mean > 0.5 and confidence < 0.7:
                    adjustments.append("Strong positive sentiment: downgraded SELL → HOLD")
                    recommendation = "HOLD"
        
        # ===================================================================
        # 3. SENTIMENT DIVERGENCE SIGNALS
        # ===================================================================
        # Divergences often predict reversals
        
        if price_up_sentiment_down > 0:  # Bearish divergence
            if recommendation == "BUY":
                # Price rising but sentiment falling → reduce BUY confidence
                reduction = 0.6  # 40% reduction
                confidence_multiplier *= reduction
                adjustments.append(f"Bearish divergence detected: -{(1-reduction)*100:.1f}% BUY confidence")
            elif recommendation == "SELL":
                # Divergence supports SELL
                boost = 1.2  # 20% boost
                confidence_multiplier *= boost
                adjustments.append(f"Bearish divergence supports SELL: +{(boost-1)*100:.1f}% confidence")
        
        if price_down_sentiment_up > 0:  # Bullish divergence
            if recommendation == "SELL":
                # Price falling but sentiment rising → reduce SELL confidence
                reduction = 0.6  # 40% reduction
                confidence_multiplier *= reduction
                adjustments.append(f"Bullish divergence detected: -{(1-reduction)*100:.1f}% SELL confidence")
            elif recommendation == "BUY":
                # Divergence supports BUY
                boost = 1.2  # 20% boost
                confidence_multiplier *= boost
                adjustments.append(f"Bullish divergence supports BUY: +{(boost-1)*100:.1f}% confidence")
        
        # ===================================================================
        # 4. SENTIMENT REGIME OVERLAY
        # ===================================================================
        # Macro sentiment regime affects all positions
        
        if bullish_regime > 0:
            adjustments.append("Bullish sentiment regime detected")
            if recommendation == "BUY":
                boost = 1.15  # 15% boost
                confidence_multiplier *= boost
                adjustments.append(f"Bullish regime boosts BUY: +{(boost-1)*100:.1f}%")
            elif recommendation == "SELL":
                reduction = 0.7  # 30% reduction
                confidence_multiplier *= reduction
                adjustments.append(f"Bullish regime reduces SELL: -{(1-reduction)*100:.1f}%")
        
        if bearish_regime > 0:
            adjustments.append("Bearish sentiment regime detected")
            if recommendation == "SELL":
                boost = 1.15  # 15% boost
                confidence_multiplier *= boost
                adjustments.append(f"Bearish regime boosts SELL: +{(boost-1)*100:.1f}%")
            elif recommendation == "BUY":
                reduction = 0.7  # 30% reduction
                confidence_multiplier *= reduction
                adjustments.append(f"Bearish regime reduces BUY: -{(1-reduction)*100:.1f}%")
        
        # ===================================================================
        # 5. APPLY CONFIDENCE ADJUSTMENT
        # ===================================================================
        
        adjusted_confidence = confidence * confidence_multiplier
        adjusted_confidence = np.clip(adjusted_confidence, 0.0, 1.0)
        
        # If confidence drops too low, downgrade to HOLD
        if adjusted_confidence < 0.3 and recommendation != "HOLD":
            adjustments.append(f"Low adjusted confidence ({adjusted_confidence:.2f}): downgraded to HOLD")
            recommendation = "HOLD"
        
        adjustment_summary = ", ".join(adjustments) if adjustments else "No sentiment adjustments"
        
        return recommendation, float(adjusted_confidence), {"adjustments": adjustment_summary}

    def predict_latest(
        self,
        data_period: str = "6mo",
        include_features: bool = False,
        feature_subset: Optional[Sequence[str]] = None,
        apply_sentiment: bool = True,
    ) -> Dict:
        """
        Generate ensemble prediction with optional sentiment adjustment.
        
        Args:
            data_period: Historical data period to fetch
            include_features: Whether to include feature snapshot in output
            feature_subset: Subset of features to include (if include_features=True)
            apply_sentiment: Whether to apply sentiment-based adjustments (default: True)
        
        Returns:
            Dictionary with prediction results and sentiment adjustments
        """
        # Use DataCacheManager to fetch/engineer and persist cache. We ignore the
        # short `data_period` here because the centralized cache uses 'max' OHLCV
        # fetch and can be sliced later. This ensures fetched artifacts are saved.
        from data.cache_manager import DataCacheManager
        cache_manager = DataCacheManager()
        raw_df, engineered_df, prepared_df, feature_cols = cache_manager.get_or_fetch_data(
            symbol=self.symbol,
            include_sentiment=apply_sentiment,
            force_refresh=False
        )

        df = engineered_df
        sequences, last_valid_idx = self._prepare_sequences(df)
        current_price = float(df.loc[last_valid_idx, "Close"])
        latest_row = df.loc[last_valid_idx]
        
        # Extract sentiment features for adjustment
        sentiment_features = self._extract_sentiment_features(latest_row) if apply_sentiment else None

        horizons = [f"{h}d" for h in self.horizons]
        class_names = {0: "SELL", 1: "HOLD", 2: "BUY"}

        individual_predictions = []
        weighted_log_returns = []
        weighted_class_probs = np.zeros(3)

        for horizon_key in horizons:
            model_reg = self.regressors[horizon_key]
            model_class = self.classifiers[horizon_key]
            weight = self.weights.get(horizon_key, 1 / len(horizons))

            X_input = sequences[horizon_key]
            pred_log_return = model_reg.predict(X_input, verbose=0)[0, 0]
            pred_class_probs = model_class.predict(X_input, verbose=0)[0]
            pred_class = int(np.argmax(pred_class_probs))

            predicted_price = current_price * np.exp(pred_log_return)
            return_pct = (np.exp(pred_log_return) - 1) * 100

            individual_predictions.append(
                {
                    "horizon": horizon_key,
                    "recommendation": class_names[pred_class],
                    "confidence": float(pred_class_probs[pred_class]),
                    "price": float(predicted_price),
                    "return_pct": float(return_pct),
                    "class_probs": {
                        "sell": float(pred_class_probs[0]),
                        "hold": float(pred_class_probs[1]),
                        "buy": float(pred_class_probs[2]),
                    },
                    "weight": float(weight),
                }
            )

            weighted_log_returns.append(pred_log_return * weight)
            weighted_class_probs += pred_class_probs * weight

        ensemble_log_return = float(np.sum(weighted_log_returns))
        ensemble_price = current_price * np.exp(ensemble_log_return)
        ensemble_return_pct = (np.exp(ensemble_log_return) - 1) * 100

        ensemble_class = int(np.argmax(weighted_class_probs))
        ensemble_confidence = float(weighted_class_probs[ensemble_class])
        ensemble_recommendation = class_names[ensemble_class]
        
        # Get GBM predictions and fuse with LSTM ensemble
        gbm_predictions = self._get_gbm_prediction(df, last_valid_idx)
        gbm_fused = False
        
        if gbm_predictions is not None and self.gbm_weight > 0:
            gbm_return = gbm_predictions.get('gbm_avg_return', 0.0)
            
            # Fuse log return: weighted average of LSTM and GBM
            lstm_weight = 1.0 - self.gbm_weight
            fused_return = (ensemble_log_return * lstm_weight) + (gbm_return * self.gbm_weight)
            
            # Update ensemble values with fused prediction
            ensemble_log_return = float(fused_return)
            ensemble_price = current_price * np.exp(ensemble_log_return)
            ensemble_return_pct = (np.exp(ensemble_log_return) - 1) * 100
            gbm_fused = True
            
            logging.getLogger(__name__).info(
                f"GBM fusion: LSTM={ensemble_log_return:.4f}, GBM={gbm_return:.4f}, "
                f"Fused={fused_return:.4f} (weight={self.gbm_weight:.0%})"
            )
        
        # Apply sentiment-based adjustment
        sentiment_adjustments = None
        original_recommendation = ensemble_recommendation
        original_confidence = ensemble_confidence
        
        if apply_sentiment and sentiment_features is not None:
            ensemble_recommendation, ensemble_confidence, sentiment_adjustments = self._apply_sentiment_adjustment(
                ensemble_recommendation,
                ensemble_confidence,
                sentiment_features
            )

        prob_sum = float(np.sum(weighted_class_probs)) or 1.0
        max_prob = float(np.max(weighted_class_probs))
        agreement_score = float((max_prob - (prob_sum - max_prob)) / prob_sum)

        if agreement_score > 0.5:
            adjusted_confidence = min(1.0, ensemble_confidence * 1.2)
            agreement_level = "🟢 Strong Agreement"
        elif agreement_score > 0.2:
            adjusted_confidence = ensemble_confidence
            agreement_level = "🟡 Moderate Agreement"
        else:
            adjusted_confidence = max(0.0, ensemble_confidence * 0.8)
            agreement_level = "🔴 Weak Agreement"

        feature_snapshot: Optional[Dict[str, Optional[float]]] = None
        if include_features:
            columns = feature_subset or list(df.columns)
            feature_snapshot = {}
            for column in columns:
                if column in latest_row:
                    feature_snapshot[column] = _safe_float(latest_row[column])

        result: Dict[str, object] = {
            "symbol": self.symbol,
            "current_price": float(current_price),
            "risk_profile": self.risk_profile,
            "ensemble_recommendation": ensemble_recommendation,
            "ensemble_confidence": float(adjusted_confidence),
            "ensemble_price": float(ensemble_price),
            "ensemble_return_pct": float(ensemble_return_pct),
            "ensemble_class_probs": {
                "sell": float(weighted_class_probs[0]),
                "hold": float(weighted_class_probs[1]),
                "buy": float(weighted_class_probs[2]),
            },
            "individual_predictions": individual_predictions,
            "agreement_score": float(agreement_score),
            "agreement_level": agreement_level,
            "metadata": {
                "model_version": self.metadata.get("model_version"),
                "weights": self.weights,
                "sequence_length": self.sequence_length,
                "n_features": len(self.selected_features),
                "selected_features": self.selected_features,
                "horizons": self.horizons,
                "gbm_enabled": gbm_fused,
                "gbm_weight": self.gbm_weight if gbm_fused else 0.0,
            },
        }
        
        # Add GBM prediction details if available
        if gbm_predictions is not None:
            result["gbm_analysis"] = {
                "gbm_fused": gbm_fused,
                "gbm_weight": self.gbm_weight,
                "predictions": gbm_predictions,
                "models_available": self.gbm_bundle.get_available_models() if self.gbm_bundle else [],
            }
        
        # Add sentiment analysis information if applied
        if apply_sentiment and sentiment_features is not None and sentiment_adjustments is not None:
            result["sentiment_analysis"] = {
                "original_recommendation": original_recommendation,
                "original_confidence": float(original_confidence),
                "sentiment_adjusted": original_recommendation != ensemble_recommendation or abs(original_confidence - ensemble_confidence) > 0.01,
                "sentiment_features": {
                    "sentiment_mean": sentiment_features['sentiment_mean'],
                    "sentiment_momentum": sentiment_features['sentiment_momentum'],
                    "news_volume": sentiment_features['news_volume'],
                    "bullish_regime": bool(sentiment_features['bullish_regime']),
                    "bearish_regime": bool(sentiment_features['bearish_regime']),
                    "price_up_sentiment_down": bool(sentiment_features['price_up_sentiment_down']),
                    "price_down_sentiment_up": bool(sentiment_features['price_down_sentiment_up']),
                },
                "adjustments": sentiment_adjustments["adjustments"],
            }
        
        if feature_snapshot is not None:
            result["latest_features"] = feature_snapshot

        return result
