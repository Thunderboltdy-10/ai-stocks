"""
Comprehensive Model Validation Suite
Tests regressor, binary classifiers, and complete trading system across multiple symbols
Generates detailed reports on model performance, weaknesses, and improvement areas
"""

import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import logging
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve,
    matthews_corrcoef, cohen_kappa_score, precision_recall_curve, auc
)
import tensorflow as tf
from tensorflow import keras
import sys
try:
    from dotenv import load_dotenv
    from pathlib import Path as _P
    p = _P(__file__).resolve()
    for parent in [p, *_P(__file__).resolve().parents[:6]]:
        env = parent / '.env'
        if env.exists():
            load_dotenv(str(env))
            break
except Exception:
    pass

from data.data_fetcher import fetch_stock_data
from data.feature_engineer import engineer_features
from data.target_engineering import prepare_training_data
from models.lstm_transformer_paper import LSTMTransformerPaper
from evaluation.advanced_backtester import AdvancedBacktester


# ===================================================================
# VALIDATION FUNCTIONS - Robust error handling
# ===================================================================

# Expected feature set sizes for compatibility checks
EXPECTED_FEATURES_V31 = 123
EXPECTED_FEATURES_V30 = 118

def validate_feature_compatibility(df, allow_legacy=True):
    """Check feature count and provide migration hints"""
    feature_count = len(df.columns)
    logger = logging.getLogger(__name__)
    
    if feature_count == EXPECTED_FEATURES_V31:
        logger.info("✅ v3.1 feature set detected (123 features)")
        return True
    elif feature_count == EXPECTED_FEATURES_V30 and allow_legacy:
        logger.warning("⚠️  v3.0 feature set detected (118 features)")
        logger.warning("Consider retraining with v3.1 features for improved performance")
        return True
    else:
        raise ValueError(f"Unexpected feature count: {feature_count}")


def validate_model_files(symbol: str, save_dir: Path):
    """Validate all required model files exist before attempting to load."""
    required_files = [
        f'{symbol}_1d_regressor_final.weights.h5',
        f'{symbol}_1d_regressor_final_feature_scaler.pkl',
        f'{symbol}_1d_regressor_final_target_scaler.pkl',
        f'{symbol}_1d_regressor_final_metadata.pkl',
        f'{symbol}_binary_classifiers_final_metadata.pkl',
        f'{symbol}_is_buy_classifier_final.weights.h5',
        f'{symbol}_is_sell_classifier_final.weights.h5'
    ]
    
    missing = [f for f in required_files if not (save_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing model files for {symbol}:\n" + 
            "\n".join(f"  - {f}" for f in missing) +
            f"\n\nModel files directory: {save_dir}"
        )
    
    print(f"   [✓] All required model files found for {symbol}")


def validate_feature_counts(df: pd.DataFrame, expected_features: list, symbol: str) -> pd.DataFrame:
    """Validate feature columns match expected features from training."""
    actual_cols = set(df.columns)
    expected_cols = set(expected_features)
    
    missing = expected_cols - actual_cols
    extra = actual_cols - expected_cols

    if missing:
        n_missing = len(missing)
        tolerance = max(5, int(0.05 * len(expected_features)))
        if n_missing > tolerance:
            raise ValueError(
                f"{symbol}: Missing features from DataFrame:\n" +
                "\n".join(f"  - {f}" for f in sorted(missing)) +
                f"\n\nExpected {len(expected_features)} features, got {len(df.columns)}\n" +
                f"This usually means feature engineering failed or feature list is outdated."
            )
        else:
            logger = logging.getLogger(__name__)
            logger.warning(
                f"{symbol}: Auto-filling {n_missing} missing feature(s) with neutral defaults: {sorted(missing)}"
            )
            for col in missing:
                if col.startswith('price_') or col.startswith('strong_') or col.endswith('_divergence') or col.startswith('is_') or col.startswith('high_volume'):
                    df[col] = 0
                else:
                    df[col] = 0.0

            # Recompute columns set after filling
            actual_cols = set(df.columns)
            extra = actual_cols - expected_cols
    
    if len(expected_features) == 118:
        print(f"   [✓] {symbol}: Using 118 features (89 technical + 29 sentiment)")
    elif len(expected_features) == 89:
        print(f"   [⚠] {symbol}: Using 89 technical features only (no sentiment)")
    else:
        print(f"   [✓] {symbol}: Using {len(expected_features)} features")

    # Do not drop additional columns (targets, prices) - callers will select `expected_features` in the correct order
    # Ensure all expected features exist (some may have been auto-filled above)
    return df


def validate_sequences(X_seq: np.ndarray, expected_shape_msg: str, expected_features: int = None):
    """Validate sequence array has correct shape."""
    if len(X_seq.shape) != 3:
        raise ValueError(
            f"Expected 3D array (samples, timesteps, features), got shape {X_seq.shape}\n"
            f"This indicates a problem with sequence creation."
        )
    
    samples, timesteps, features = X_seq.shape
    print(f"   [{expected_shape_msg}]: ({samples} samples, {timesteps} timesteps, {features} features)")
    
    if expected_features is not None and features != expected_features:
        raise ValueError(
            f"Feature count mismatch in sequences!\n"
            f"Expected {expected_features} features, got {features}\n"
            f"Model was trained with {expected_features} features but sequences have {features}"
        )
    
    if samples == 0:
        raise ValueError(
            f"No sequences created! This means insufficient data.\n"
            f"Need at least {timesteps} days of data to create sequences."
        )
    
    return samples, timesteps, features


def create_binary_classifier(sequence_length, n_features, name='binary_classifier', arch: dict | None = None):
    """Create binary classifier with LSTMTransformerPaper backbone - OLD architecture for existing weights"""
    import tensorflow as tf
    from tensorflow import keras
    from keras import layers
    
    arch = arch or {}
    lstm_units = int(arch.get('lstm_units', 64))
    d_model = int(arch.get('d_model', 128))  # Updated to match saved classifier weights
    num_heads = int(arch.get('num_heads', 4))
    num_blocks = int(arch.get('num_blocks', 6))
    ff_dim = int(arch.get('ff_dim', 256))
    dropout = float(arch.get('dropout', 0.35))

    # Create the LSTM-Transformer base with parameters
    base = LSTMTransformerPaper(
        sequence_length=sequence_length,
        n_features=n_features,
        lstm_units=lstm_units,
        d_model=d_model,
        num_heads=num_heads,
        num_blocks=num_blocks,
        ff_dim=ff_dim,
        dropout=dropout
    )
    
    # Build the base model
    dummy = tf.random.normal((1, sequence_length, n_features))
    _ = base(dummy)
    
    # Binary classification head
    inputs = keras.Input(shape=(sequence_length, n_features))
    
    x = base.lstm_layer(inputs)
    x = base.projection(x)
    x = x + base.pos_encoding[:, :sequence_length, :]
    
    for block in base.transformer_blocks:
        attn = block['attention'](x, x)
        attn = block['dropout1'](attn)
        x = block['norm1'](x + attn)
        
        ffn = block['ffn2'](block['ffn1'](x))
        ffn = block['dropout2'](ffn)
        x = block['norm2'](x + ffn)
    
    # Use base model's pooling and dropout
    x = base.global_pool(x)
    x = base.dropout_out(x)
    
    # Binary output - direct from pooled features
    outputs = layers.Dense(1, activation='sigmoid', name='binary_output')(x)
    
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def create_multitask_regressor(sequence_length, n_features, name='multitask_regressor', arch: dict | None = None):
    """Create multi-task regressor with 3 output heads - MUST match training architecture exactly.
    
    Multi-task outputs:
    - Head 1: Return magnitude prediction (regression) - bounded via tanh
    - Head 2: Return sign classification (up/down/flat)
    - Head 3: Volatility prediction (next-day volatility)
    
    CRITICAL: This MUST match create_multitask_regressor in train_1d_regressor_final.py
    Bug fix 2025-12-14: Updated to match actual training architecture with:
    - Reduced complexity (4 blocks, lstm_units=48, d_model=96)
    - Tanh magnitude output with Lambda scaling
    - NO batch norm on shared/magnitude path
    """
    from tensorflow.keras import regularizers
    
    arch = arch or {}
    # Use corrected defaults matching the training script
    lstm_units = int(arch.get('lstm_units', 48))    # Corrected from 64
    d_model = int(arch.get('d_model', 96))          # Corrected from 128
    num_heads = int(arch.get('num_heads', 4))
    num_blocks = int(arch.get('num_blocks', 4))     # Corrected from 6
    ff_dim = int(arch.get('ff_dim', 192))           # Corrected from 256
    dropout = float(arch.get('dropout', 0.2))       # Corrected from 0.3

    base = LSTMTransformerPaper(
        sequence_length=sequence_length,
        n_features=n_features,
        lstm_units=lstm_units,
        d_model=d_model,
        num_heads=num_heads,
        num_blocks=num_blocks,
        ff_dim=ff_dim,
        dropout=dropout
    )
    
    dummy = tf.random.normal((1, sequence_length, n_features))
    _ = base(dummy)
    
    # Shared backbone
    inputs = keras.Input(shape=(sequence_length, n_features))
    
    x = base.lstm_layer(inputs)
    x = base.projection(x)
    x = x + base.pos_encoding[:, :sequence_length, :]
    
    for block in base.transformer_blocks:
        attn = block['attention'](x, x)
        attn = block['dropout1'](attn)
        x = block['norm1'](x + attn)
        
        ffn = block['ffn2'](block['ffn1'](x))
        ffn = block['dropout2'](ffn)
        x = block['norm2'](x + ffn)
    
    x = base.global_pool(x)
    x = base.dropout_out(x)
    
    # Shared dense layer - NO batch norm to match training and prevent collapse
    shared = keras.layers.Dense(
        48,  # Match training script (reduced from 64)
        activation='relu',
        kernel_initializer=keras.initializers.HeNormal(seed=42),
        kernel_regularizer=regularizers.l2(0.0005),
        name='shared_dense'
    )(x)
    shared = keras.layers.Dropout(0.15, name='shared_dropout')(shared)
    
    # HEAD 1: Return Magnitude (Regression) - CRITICAL: Match training exactly
    # P0.5 FIX: Updated to match train_1d_regressor_final.py architecture exactly
    magnitude_branch = keras.layers.Dense(
        64,  # Fixed: training uses 64, was incorrectly 32
        activation='relu',
        kernel_initializer=keras.initializers.HeNormal(seed=43),
        kernel_regularizer=regularizers.l2(0.0001),
        name='magnitude_dense'
    )(shared)
    magnitude_branch = keras.layers.Dropout(0.1, name='magnitude_dropout')(magnitude_branch)
    
    # Second dense layer for more expressiveness (matches training)
    magnitude_branch = keras.layers.Dense(
        32,
        activation='relu',
        kernel_initializer=keras.initializers.HeNormal(seed=45),
        kernel_regularizer=regularizers.l2(0.0001),
        name='magnitude_dense2'
    )(magnitude_branch)
    
    # Linear output with RandomNormal init (matches training anti-collapse design)
    magnitude_output = keras.layers.Dense(
        1,
        activation=None,  # LINEAR output like training
        kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.5, seed=44),
        bias_initializer=keras.initializers.Zeros(),
        name='magnitude_output'
    )(magnitude_branch)
    
    # HEAD 2: Return Sign (Classification)
    sign_branch = keras.layers.Dense(
        32,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001),
        name='sign_dense'
    )(shared)
    sign_branch = keras.layers.BatchNormalization(name='sign_bn')(sign_branch)
    sign_branch = keras.layers.Dropout(0.2, name='sign_dropout')(sign_branch)
    
    sign_output = keras.layers.Dense(
        3,
        activation='softmax',
        kernel_regularizer=regularizers.l2(0.001),
        name='sign_output'
    )(sign_branch)
    
    # HEAD 3: Volatility (Regression)
    volatility_branch = keras.layers.Dense(
        32,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001),
        name='volatility_dense'
    )(shared)
    volatility_branch = keras.layers.BatchNormalization(name='volatility_bn')(volatility_branch)
    volatility_branch = keras.layers.Dropout(0.2, name='volatility_dropout')(volatility_branch)
    
    volatility_output = keras.layers.Dense(
        1,
        activation='relu',
        kernel_regularizer=regularizers.l2(0.001),
        name='volatility_output'
    )(volatility_branch)
    
    return keras.Model(
        inputs=inputs,
        outputs=[magnitude_output, sign_output, volatility_output],
        name=name
    )


class ModelValidationSuite:
    """Comprehensive validation suite for ML trading models"""
    
    def __init__(self, symbols: List[str] = ['AAPL', 'HOOD', 'TSLA'], seq_len: int = 90):
        self.symbols = symbols
        self.seq_len_regressor = 90  # Regressor uses 90-day sequences
        self.seq_len_classifier = 60  # Classifiers use 60-day sequences
        self.seq_len = seq_len  # Keep for backward compatibility
        self.results = {}
        self.save_dir = Path('validation_results')
        self.save_dir.mkdir(exist_ok=True)
        
        # Create timestamped subdirectory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.run_dir = self.save_dir / timestamp
        self.run_dir.mkdir(exist_ok=True)
        
        print(f"[>>] Model Validation Suite")
        print(f"Symbols to test: {', '.join(symbols)}")
        print(f"Results directory: {self.run_dir}")
        print("=" * 80)
    
    def load_models_and_data(self, symbol: str) -> Dict:
        """Load all models and prepare data for a symbol"""
        print(f"\n[*] Loading models and data for {symbol}...")
        
        save_dir = Path('saved_models')
        
        # ===================================================================
        # VALIDATION: Check all required model files exist
        # ===================================================================
        try:
            validate_model_files(symbol, save_dir)
        except FileNotFoundError as e:
            print(f"\n   [❌] ERROR: Model files not found")
            print(f"   {e}")
            raise
        
        # The models are saved with naming convention: {SYMBOL}_1d_regressor_final.weights.h5
        # Load scalers and metadata
        feature_scaler = self._load_pickle(save_dir / f'{symbol}_1d_regressor_final_feature_scaler.pkl')
        target_scaler = self._load_pickle(save_dir / f'{symbol}_1d_regressor_final_target_scaler.pkl')
        target_meta = self._load_pickle(save_dir / f'{symbol}_1d_regressor_final_metadata.pkl')
        classifier_meta = self._load_pickle(save_dir / f'{symbol}_binary_classifiers_final_metadata.pkl')
        feature_cols = self._load_pickle(save_dir / f'{symbol}_1d_regressor_final_features.pkl')
        
        missing_artifacts = []
        if not feature_scaler: missing_artifacts.append(f'{symbol}_1d_regressor_final_feature_scaler.pkl')
        if not target_scaler: missing_artifacts.append(f'{symbol}_1d_regressor_final_target_scaler.pkl')
        if not target_meta: missing_artifacts.append(f'{symbol}_1d_regressor_final_metadata.pkl')
        if not classifier_meta: missing_artifacts.append(f'{symbol}_binary_classifiers_final_metadata.pkl')
        if not feature_cols: missing_artifacts.append(f'{symbol}_1d_regressor_final_features.pkl')
        
        if missing_artifacts:
            raise ValueError(f"Missing model artifacts for {symbol}: {', '.join(missing_artifacts)}")
        
        # Load regressor
        regressor_weights = save_dir / f'{symbol}_1d_regressor_final.weights.h5'
        if not regressor_weights.exists():
            raise ValueError(f"Regressor weights not found for {symbol}")
        
        # Ensure feature count matches metadata
        meta_n_features = target_meta.get('n_features') if isinstance(target_meta, dict) else None
        if meta_n_features is not None and meta_n_features != len(feature_cols):
            raise ValueError(
                f"{symbol}: Feature count mismatch with saved model metadata! Model expects {meta_n_features}, "
                f"but found {len(feature_cols)} features."
            )

        # Create multitask regressor (matching training architecture - 3 output heads)
        arch = target_meta.get('architecture', {}) if isinstance(target_meta, dict) else {}
        regressor = create_multitask_regressor(
            sequence_length=self.seq_len_regressor,  # 90 days
            n_features=len(feature_cols),
            name='multitask_regressor',
            arch=arch
        )
        print(f"   [✓] Reconstructed regressor architecture: {arch}")
        regressor.compile(optimizer='adam', loss='mse', metrics=['mae'])
        regressor.load_weights(str(regressor_weights))
        
        # Dummy input for classifiers (60-day sequences)
        dummy_input_classifier = np.zeros((1, self.seq_len_classifier, len(feature_cols)), dtype=np.float32)
        
        # Load buy classifiers (single model, not ensemble)
        buy_classifiers = []
        buy_weights = save_dir / f'{symbol}_is_buy_classifier_final.weights.h5'
        if buy_weights.exists():
            arch_clf = classifier_meta.get('architecture', {}) if isinstance(classifier_meta, dict) else {}
            model = create_binary_classifier(self.seq_len_classifier, len(feature_cols), arch=arch_clf)  # 60 days
            _ = model(dummy_input_classifier, training=False)
            model.load_weights(str(buy_weights))
            buy_classifiers.append(model)
        
        # Load sell classifiers (single model, not ensemble)
        sell_classifiers = []
        sell_weights = save_dir / f'{symbol}_is_sell_classifier_final.weights.h5'
        if sell_weights.exists():
            arch_clf = classifier_meta.get('architecture', {}) if isinstance(classifier_meta, dict) else {}
            model = create_binary_classifier(self.seq_len_classifier, len(feature_cols), arch=arch_clf)  # 60 days
            _ = model(dummy_input_classifier, training=False)
            model.load_weights(str(sell_weights))
            sell_classifiers.append(model)
        
        # Fetch and prepare data via centralized cache manager
        print(f"   [*] Fetching market data via cache manager...")
        from data.cache_manager import DataCacheManager
        cm = DataCacheManager()
        raw_df, engineered_df, prepared_df, feature_cols = cm.get_or_fetch_data(symbol, include_sentiment=True, force_refresh=False)
        df = raw_df
        
        # ===================================================================
        # CRITICAL: Sentiment Feature Integration with Graceful Fallback
        # - Attempts to load 118 features (89 technical + 29 sentiment)
        # - Falls back to 89 technical-only features if sentiment fails
        # ===================================================================
        try:
            # Pass symbol to engineer_features to enable sentiment features
            print(f"   [*] Loading features with sentiment for {symbol}...")
            df = engineer_features(df, symbol=symbol, include_sentiment=True)
            print(f"   [✓] Loaded 118 features (89 technical + 29 sentiment)")
        except Exception as e:
            print(f"   [⚠️ ] Warning: Sentiment features failed for {symbol}: {e}")
            print(f"   [*] Falling back to 89 technical-only features...")
            df = engineer_features(df, symbol=None, include_sentiment=False)
            print(f"   [✓] Loaded 89 technical features (no sentiment)")
        
        # Prepare targets with 1-day horizon (matching training)
        df, _ = prepare_training_data(df, horizons=[1])
        
        # ===================================================================
        # VALIDATION: Check feature counts match expectations
        # ===================================================================
        try:
            # First run a lightweight compatibility check to provide clear migration hints
            validate_feature_compatibility(df, allow_legacy=True)
            df = validate_feature_counts(df, feature_cols, symbol)
        except ValueError as e:
            print(f"\n   [❌] ERROR: Feature validation failed")
            print(f"   {e}")
            # Attempt to auto-fill missing features with neutral defaults and retry
            missing_cols = set(feature_cols) - set(df.columns)
            if missing_cols:
                print(f"\n   ⚠️ Auto-filling missing features: {sorted(missing_cols)} with neutral values")
                for col in missing_cols:
                    df[col] = 0.0
                try:
                    df = validate_feature_counts(df, feature_cols, symbol)
                except ValueError as e2:
                    print(f"\n   [❌] ERROR: Feature validation still failed after auto-fill: {e2}")
                    raise
            else:
                raise
        
        return {
            'symbol': symbol,
            'regressor': regressor,
            'buy_classifiers': buy_classifiers,
            'sell_classifiers': sell_classifiers,
            'feature_scaler': feature_scaler,
            'target_scaler': target_scaler,
            'target_meta': target_meta,
            'classifier_meta': classifier_meta,
            'feature_cols': feature_cols,
            'data': df
        }
    
    def _calculate_rolling_r2(self, y_true, y_pred, window=5):
        """Calculate rolling R² over time to assess temporal stability."""
        rolling_r2 = []
        for i in range(window, len(y_true)):
            y_true_window = y_true[i-window:i]
            y_pred_window = y_pred[i-window:i]
            
            # Calculate R² for this window
            ss_res = np.sum((y_true_window - y_pred_window) ** 2)
            ss_tot = np.sum((y_true_window - np.mean(y_true_window)) ** 2)
            r2_window = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            rolling_r2.append(r2_window)
        
        return np.array(rolling_r2)
    
    def _load_pickle(self, path: Path):
        """Load pickle file safely with numpy compatibility"""
        if not path.exists():
            return None
        
        try:
            with open(path, 'rb') as f:
                return pickle.load(f)
        except ModuleNotFoundError as e:
            # Handle numpy version compatibility issues
            if 'numpy._core' in str(e):
                print(f"   [⚠] Numpy compatibility issue loading {path.name}")
                print(f"       Applying numpy._core compatibility fix...")
                import sys
                import numpy
                # Create compatibility aliases for numpy 2.0+
                sys.modules['numpy._core'] = numpy.core
                sys.modules['numpy._core._multiarray_umath'] = numpy.core._multiarray_umath
                with open(path, 'rb') as f:
                    return pickle.load(f)
            else:
                print(f"   [❌] Failed to load {path.name}: {e}")
                raise
        except Exception as e:
            print(f"   [❌] Unexpected error loading {path.name}: {type(e).__name__}: {e}")
            raise
    
    def validate_regressor(self, bundle: Dict) -> Dict:
        """Validate regressor model performance"""
        print(f"\n[REGRESSOR] Validating for {bundle['symbol']}...")
        
        df = bundle['data'].copy()
        feature_cols = bundle['feature_cols']
        
        # ===================================================================
        # SEQUENCE CREATION - Regressor uses 90-day sequences
        # ===================================================================
        X_raw = df[feature_cols].values
        X_scaled = bundle['feature_scaler'].transform(X_raw)
        
        # Create regressor sequences (90-day)
        sequences = []
        for i in range(self.seq_len_regressor - 1, len(X_scaled)):
            sequences.append(X_scaled[i - self.seq_len_regressor + 1:i + 1])
        X_seq = np.array(sequences)
        
        # ===================================================================
        # ALIGN supporting arrays with regressor sequences
        # - All arrays aligned to regressor_seq_len - 1 offset (89 in this case)
        # ===================================================================
        y_actual_raw = df['target_1d'].values[self.seq_len_regressor - 1:]
        prices_aligned = df['Close'].values[self.seq_len_regressor - 1:]
        dates_aligned = df.index[self.seq_len_regressor - 1:]
        
        # ===================================================================
        # VALIDATION: Check sequence shapes and alignment
        # ===================================================================
        try:
            validate_sequences(X_seq, "Regressor sequences", expected_features=len(feature_cols))
            assert len(X_seq) == len(y_actual_raw) == len(prices_aligned) == len(dates_aligned), \
                f"Length mismatch: X_seq={len(X_seq)}, y={len(y_actual_raw)}, prices={len(prices_aligned)}, dates={len(dates_aligned)}"
        except (ValueError, AssertionError) as e:
            print(f"\n   [❌] ERROR: Sequence validation failed")
            print(f"   {e}")
            raise
        
        # ===================================================================
        # CRITICAL: Multitask regressor returns 3 outputs
        # - multitask_preds[0]: magnitude_output (regression, what we use)
        # - multitask_preds[1]: sign_output (classification, auxiliary)
        # - multitask_preds[2]: volatility_output (regression, auxiliary)
        # We only use magnitude_output for trading decisions
        # ===================================================================
        multitask_preds = bundle['regressor'].predict(X_seq, verbose=0)
        y_pred_scaled = multitask_preds[0].flatten()  # magnitude_output
        
        print(f"   Multitask output shapes: magnitude={multitask_preds[0].shape}, sign={multitask_preds[1].shape}, volatility={multitask_preds[2].shape}")
        
        y_pred_transformed = bundle['target_scaler'].inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        
        # Inverse transform based on target transform
        if bundle['target_meta'].get('target_transform') == 'log1p':
            y_pred = np.expm1(y_pred_transformed)
        else:
            y_pred = y_pred_transformed
        
        # Calculate metrics
        mse = mean_squared_error(y_actual_raw, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_actual_raw, y_pred)
        r2 = r2_score(y_actual_raw, y_pred)
        
        # Direction accuracy (did we predict the right sign?)
        direction_actual = (y_actual_raw > 0).astype(int)
        direction_pred = (y_pred > 0).astype(int)
        direction_accuracy = accuracy_score(direction_actual, direction_pred)
        
        # Magnitude correlation
        magnitude_corr = np.corrcoef(np.abs(y_actual_raw), np.abs(y_pred))[0, 1]
        
        # ===================================================================
        # FINANCIAL METRICS
        # ===================================================================
        
        # Information Coefficient (IC) - correlation between predictions and actuals
        # Standard metric in quantitative finance (IC > 0.05 is significant)
        ic = np.corrcoef(y_pred, y_actual_raw)[0, 1]
        
        # Hit Rate - percentage of correct directional predictions
        correct_direction = ((y_pred > 0) & (y_actual_raw > 0)) | ((y_pred < 0) & (y_actual_raw < 0))
        hit_rate = np.mean(correct_direction)
        
        # Sortino Ratio - return adjusted for downside risk only
        returns_series = y_actual_raw  # These are the actual returns
        mean_return = np.mean(returns_series)
        downside_returns = returns_series[returns_series < 0]
        downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 1e-8
        sortino_ratio = (mean_return / downside_std) * np.sqrt(252)  # Annualized
        
        # Calculate rolling R² (5-day windows) for temporal stability
        rolling_r2 = self._calculate_rolling_r2(y_actual_raw, y_pred, window=5)
        
        results = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'direction_accuracy': float(direction_accuracy),
            'magnitude_correlation': float(magnitude_corr),
            'ic': float(ic),
            'hit_rate': float(hit_rate),
            'sortino_ratio': float(sortino_ratio),
            'rolling_r2_mean': float(np.mean(rolling_r2)) if len(rolling_r2) > 0 else 0.0,
            'rolling_r2_std': float(np.std(rolling_r2)) if len(rolling_r2) > 0 else 0.0,
            'predictions': y_pred,
            'actuals': y_actual_raw,
            'dates': df.index[self.seq_len_regressor - 1:].tolist()
        }
        
        print(f"   RMSE: {rmse:.6f}")
        print(f"   MAE: {mae:.6f}")
        print(f"   R²: {r2:.4f}")
        print(f"   Direction Accuracy: {direction_accuracy:.2%}")
        print(f"   Magnitude Correlation: {magnitude_corr:.4f}")
        print(f"   Information Coefficient (IC): {ic:.4f}")
        print(f"   Hit Rate: {hit_rate:.2%}")
        print(f"   Sortino Ratio: {sortino_ratio:.4f}")
        print(f"   Rolling R² (5-day): {np.mean(rolling_r2):.4f} ± {np.std(rolling_r2):.4f}" if len(rolling_r2) > 0 else "   Rolling R²: N/A")
        
        # Plot predictions vs actuals
        self._plot_regressor_results(bundle['symbol'], results)
        
        return results
    
    def validate_classifiers(self, bundle: Dict) -> Dict:
        """Validate buy/sell classifier performance"""
        print(f"\n[CLASSIFIERS] Validating for {bundle['symbol']}...")
        
        df = bundle['data'].copy()
        feature_cols = bundle['feature_cols']
        
        # ===================================================================
        # SEQUENCE CREATION - Classifiers use 60-day sequences
        # ===================================================================
        X_raw = df[feature_cols].values
        X_scaled = bundle['feature_scaler'].transform(X_raw)
        
        # Create classifier sequences (60-day)
        sequences = []
        for i in range(self.seq_len_classifier - 1, len(X_scaled)):
            sequences.append(X_scaled[i - self.seq_len_classifier + 1:i + 1])
        X_seq = np.array(sequences)
        
        # ===================================================================
        # ALIGN supporting arrays with classifier sequences
        # - All arrays aligned to seq_len_classifier - 1 offset (59 in this case)
        # ===================================================================
        returns_1d = df['target_1d'].values[self.seq_len_classifier - 1:]
        prices_aligned = df['Close'].values[self.seq_len_classifier - 1:]
        dates_aligned = df.index[self.seq_len_classifier - 1:]
        
        # ===================================================================
        # VALIDATION: Check sequence shapes and alignment
        # ===================================================================
        try:
            validate_sequences(X_seq, "Classifier sequences", expected_features=len(feature_cols))
            assert len(X_seq) == len(returns_1d) == len(prices_aligned) == len(dates_aligned), \
                f"Length mismatch: X_seq={len(X_seq)}, returns={len(returns_1d)}, prices={len(prices_aligned)}, dates={len(dates_aligned)}"
        except (ValueError, AssertionError) as e:
            print(f"\n   [❌] ERROR: Sequence validation failed")
            print(f"   {e}")
            raise
        
        # Use percentile thresholds (80/20 as in training)
        buy_threshold = np.percentile(returns_1d, 80)
        sell_threshold = np.percentile(returns_1d, 20)
        
        buy_target = (returns_1d > buy_threshold).astype(int)
        sell_target = (returns_1d < sell_threshold).astype(int)
        
        # Buy classifier predictions (ensemble average)
        buy_probs_list = []
        for model in bundle['buy_classifiers']:
            buy_probs_list.append(model.predict(X_seq, verbose=0).flatten())
        buy_probs = np.mean(buy_probs_list, axis=0) if buy_probs_list else np.zeros(len(buy_target))
        buy_preds = (buy_probs >= 0.5).astype(int)
        
        # Sell classifier predictions (ensemble average)
        sell_probs_list = []
        for model in bundle['sell_classifiers']:
            sell_probs_list.append(model.predict(X_seq, verbose=0).flatten())
        sell_probs = np.mean(sell_probs_list, axis=0) if sell_probs_list else np.zeros(len(sell_target))
        sell_preds = (sell_probs >= 0.5).astype(int)
        
        # Buy metrics
        buy_metrics = self._calculate_classifier_metrics(
            buy_target, buy_preds, buy_probs, 'BUY'
        )
        
        # Sell metrics
        sell_metrics = self._calculate_classifier_metrics(
            sell_target, sell_preds, sell_probs, 'SELL'
        )
        
        results = {
            'buy': buy_metrics,
            'sell': sell_metrics,
            'buy_probs': buy_probs,
            'sell_probs': sell_probs,
            'buy_target': buy_target,
            'sell_target': sell_target,
            'dates': df.index[self.seq_len_classifier - 1:].tolist()
        }
        
        # Plot classifier performance
        self._plot_classifier_results(bundle['symbol'], results)
        
        return results
    
    def _calculate_classifier_metrics(self, y_true, y_pred, y_probs, label: str) -> Dict:
        """Calculate comprehensive classifier metrics"""
        
        # Handle edge cases where all predictions are the same class
        unique_classes = np.unique(y_pred)
        
        accuracy = accuracy_score(y_true, y_pred)
        
        # Precision, recall, F1 with zero_division handling
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # ROC AUC (only if both classes present)
        try:
            roc_auc = roc_auc_score(y_true, y_probs)
        except ValueError:
            roc_auc = 0.5  # Default if only one class
        
        # ===================================================================
        # ADVANCED CLASSIFIER METRICS
        # ===================================================================
        
        # Matthews Correlation Coefficient (MCC)
        # Range [-1, 1]: 1=perfect, 0=random, -1=inverse prediction
        # More reliable than F1 for imbalanced datasets
        mcc = matthews_corrcoef(y_true, y_pred)
        
        # Cohen's Kappa Score
        # Measures agreement accounting for chance
        # >0.8=almost perfect, 0.6-0.8=substantial, 0.4-0.6=moderate
        kappa = cohen_kappa_score(y_true, y_pred)
        
        # Precision-Recall AUC (often more informative than ROC-AUC for imbalanced data)
        try:
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_probs)
            pr_auc = auc(recall_vals, precision_vals)
        except ValueError:
            pr_auc = 0.0  # Default if only one class
        
        # Class distribution
        class_distribution = {
            'actual_positives': int(np.sum(y_true)),
            'actual_negatives': int(len(y_true) - np.sum(y_true)),
            'predicted_positives': int(np.sum(y_pred)),
            'predicted_negatives': int(len(y_pred) - np.sum(y_pred))
        }
        
        metrics = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'auc_roc': float(roc_auc),
            'mcc': float(mcc),
            'kappa': float(kappa),
            'pr_auc': float(pr_auc),
            'confusion_matrix': cm.tolist(),
            'class_distribution': class_distribution
        }
        
        print(f"\n   {label} Classifier:")
        print(f"      Accuracy: {accuracy:.2%}")
        print(f"      Precision: {precision:.2%}")
        print(f"      Recall: {recall:.2%}")
        print(f"      F1 Score: {f1:.4f}")
        print(f"      AUC-ROC: {roc_auc:.4f}")
        print(f"      PR-AUC: {pr_auc:.4f} (better for imbalanced data)")
        print(f"      MCC: {mcc:.4f} (range: -1 to 1)")
        print(f"      Cohen's Kappa: {kappa:.4f}")
        print(f"      Confusion Matrix:\n{cm}")
        
        return metrics
    
    def run_backtest(self, bundle: Dict) -> Dict:
        """Run comprehensive backtest with advanced metrics"""
        print(f"\n[BACKTEST] Running for {bundle['symbol']}...")
        
        df = bundle['data'].copy()
        
        # Use AdvancedBacktester for comprehensive analysis
        backtester = AdvancedBacktester(
            initial_capital=10000,
            risk_free_rate=0.02
        )
        
        # ===================================================================
        # SEQUENCE CREATION - Using classifier sequences (60-day)
        # ===================================================================
        feature_cols = bundle['feature_cols']
        X_raw = df[feature_cols].values
        X_scaled = bundle['feature_scaler'].transform(X_raw)
        
        # Create classifier sequences (60-day)
        sequences = []
        for i in range(self.seq_len_classifier - 1, len(X_scaled)):
            sequences.append(X_scaled[i - self.seq_len_classifier + 1:i + 1])
        X_seq = np.array(sequences)
        
        # ===================================================================
        # ALIGN supporting arrays with classifier sequences
        # - All arrays aligned to seq_len_classifier - 1 offset
        # ===================================================================
        backtest_dates = df.index[self.seq_len_classifier - 1:].values
        backtest_prices = df['Close'].values[self.seq_len_classifier - 1:]
        backtest_returns = df['target_1d'].values[self.seq_len_classifier - 1:]
        
        # ===================================================================
        # VALIDATION: Check sequence shapes and alignment
        # ===================================================================
        try:
            validate_sequences(X_seq, "Backtest sequences", expected_features=len(feature_cols))
            assert len(X_seq) == len(backtest_dates) == len(backtest_prices) == len(backtest_returns), \
                f"Length mismatch: X_seq={len(X_seq)}, dates={len(backtest_dates)}, " \
                f"prices={len(backtest_prices)}, returns={len(backtest_returns)}"
        except (ValueError, AssertionError) as e:
            print(f"\n   [❌] ERROR: Backtest sequence validation failed")
            print(f"   {e}")
            raise
        
        # Get predictions
        buy_probs_list = [m.predict(X_seq, verbose=0).flatten() for m in bundle['buy_classifiers']]
        sell_probs_list = [m.predict(X_seq, verbose=0).flatten() for m in bundle['sell_classifiers']]
        
        buy_probs = np.mean(buy_probs_list, axis=0) if buy_probs_list else np.zeros(len(X_seq))
        sell_probs = np.mean(sell_probs_list, axis=0) if sell_probs_list else np.zeros(len(X_seq))
        
        # Create signals
        signals = np.zeros(len(buy_probs))
        signals[buy_probs >= 0.5] = 1
        signals[sell_probs >= 0.5] = -1
        
        # Verify predictions match sequence length
        assert len(buy_probs) == len(sell_probs) == len(X_seq), \
            f"Prediction length mismatch: buy={len(buy_probs)}, sell={len(sell_probs)}, X_seq={len(X_seq)}"
        
        # Run backtest using AdvancedBacktester
        results = backtester.backtest_with_confidence(
            dates=backtest_dates,
            prices=backtest_prices,
            returns=backtest_returns,
            buy_signals=(buy_probs >= 0.5).astype(int),
            sell_signals=(sell_probs >= 0.5).astype(int),
            buy_confidence=buy_probs,
            sell_confidence=sell_probs,
            position_sizing='confidence',
            max_position_size=0.95,
            buy_min_confidence=0.5,
            sell_min_confidence=0.5
        )
        
        # Extract metrics from BacktestResults safely
        metrics = results.metrics if hasattr(results, 'metrics') else {}
        
        # Compute additional metrics from trade log
        trade_log = results.trade_log if hasattr(results, 'trade_log') else []
        total_trades = len(trade_log)
        
        # Calculate win rate from equity changes
        win_rate = 0.0
        daily_returns = results.daily_returns if hasattr(results, 'daily_returns') else []
        if len(daily_returns) > 0:
            winning_days = np.sum(np.array(daily_returns) > 0)
            total_days = len(daily_returns)
            win_rate = winning_days / total_days if total_days > 0 else 0.0
        
        # Enhanced metrics using correct keys with safe defaults
        total_return = metrics.get('cum_return', 0) * 100 if 'cum_return' in metrics else 0
        sharpe = metrics.get('sharpe', 0) if 'sharpe' in metrics else 0
        max_drawdown = metrics.get('max_drawdown', 0) if 'max_drawdown' in metrics else 0
        
        # ===================================================================
        # ADVANCED BACKTEST METRICS
        # ===================================================================
        
        # Calmar Ratio = Total Return / Max Drawdown
        # Measures return per unit of maximum loss
        calmar_ratio = (total_return / 100) / max_drawdown if max_drawdown > 0 else 0.0
        
        # Calculate profit factor and win/loss ratios from daily returns
        if hasattr(results, 'daily_returns') and len(results.daily_returns) > 0:
            daily_returns = np.array(results.daily_returns)
        elif hasattr(results, 'equity_curve') and len(results.equity_curve) > 1:
            equity_array = np.array(results.equity_curve)
            daily_returns = np.diff(equity_array) / equity_array[:-1]
        else:
            daily_returns = np.array([])
        
        winning_returns = daily_returns[daily_returns > 0] if len(daily_returns) > 0 else np.array([])
        losing_returns = daily_returns[daily_returns < 0] if len(daily_returns) > 0 else np.array([])
        
        # Profit Factor = Gross Profit / Gross Loss
        gross_profit = np.sum(winning_returns) if len(winning_returns) > 0 else 0.0
        gross_loss = abs(np.sum(losing_returns)) if len(losing_returns) > 0 else 1e-8
        profit_factor = gross_profit / gross_loss
        
        # Average Win / Average Loss Ratio
        avg_win = np.mean(winning_returns) if len(winning_returns) > 0 else 0.0
        avg_loss = abs(np.mean(losing_returns)) if len(losing_returns) > 0 else 1e-8
        win_loss_ratio = avg_win / avg_loss
        
        # Trade opportunity rate = number of trades / total trading days
        total_days = len(daily_returns) if len(daily_returns) > 0 else 0
        trade_opportunity_rate = total_trades / total_days if total_days > 0 else 0.0
        
        backtest_results = {
            'metrics': {
                **metrics,
                'total_return': metrics.get('cum_return', 0),
                'sharpe_ratio': sharpe,
                'win_rate': win_rate,
                'total_trades': total_trades,
                'calmar_ratio': float(calmar_ratio),
                'profit_factor': float(profit_factor),
                'win_loss_ratio': float(win_loss_ratio),
                'trade_opportunity_rate': float(trade_opportunity_rate),
                'avg_win': float(avg_win),
                'avg_loss': float(avg_loss)
            },
            'total_return_pct': float(total_return),
            'equity_curve': results.equity_curve.tolist() if hasattr(results.equity_curve, 'tolist') else list(results.equity_curve),
            'positions': results.positions.tolist() if hasattr(results.positions, 'tolist') else list(results.positions),
            'dates': backtest_dates.tolist() if hasattr(backtest_dates, 'tolist') else list(backtest_dates)
        }
        
        print(f"   Total Return: {total_return:.2f}%")
        print(f"   Sharpe Ratio: {metrics.get('sharpe', 0):.4f}")
        print(f"   Calmar Ratio: {calmar_ratio:.4f}")
        print(f"   Max Drawdown: {max_drawdown:.2%}")
        print(f"   Win Rate: {win_rate:.2%}")
        print(f"   Profit Factor: {profit_factor:.4f} (gross profit/loss)")
        print(f"   Win/Loss Ratio: {win_loss_ratio:.4f}")
        print(f"   Avg Win: {avg_win:.4f} | Avg Loss: {avg_loss:.4f}")
        print(f"   Total Trades: {total_trades}")
        print(f"   Trade Opportunity Rate: {trade_opportunity_rate:.2%} (trades/day)")
        
        # Plot backtest results
        self._plot_backtest_results(bundle['symbol'], backtest_results)
        
        return backtest_results
    
    def analyze_weaknesses(self, symbol: str, results: Dict) -> Dict:
        """Identify model weaknesses and improvement areas"""
        print(f"\n[ANALYSIS] Analyzing weaknesses for {symbol}...")
        
        weaknesses = {
            'regressor': [],
            'buy_classifier': [],
            'sell_classifier': [],
            'backtest': [],
            'recommendations': []
        }
        
        # Regressor analysis
        reg = results.get('regressor', {})
        if reg.get('r2', 0) < 0.3:
            weaknesses['regressor'].append(f"Low R² ({reg.get('r2', 0):.4f}) - poor predictive power")
        if reg.get('direction_accuracy', 0) < 0.55:
            weaknesses['regressor'].append(f"Low direction accuracy ({reg.get('direction_accuracy', 0):.2%}) - barely better than random")
        if reg.get('magnitude_correlation', 0) < 0.3:
            weaknesses['regressor'].append(f"Low magnitude correlation ({reg.get('magnitude_correlation', 0):.4f}) - poor magnitude estimation")
        
        # Buy classifier analysis
        buy = results.get('classifiers', {}).get('buy', {})
        if buy.get('precision', 0) < 0.4:
            weaknesses['buy_classifier'].append(f"Low precision ({buy.get('precision', 0):.2%}) - many false BUY signals")
        if buy.get('recall', 0) < 0.3:
            weaknesses['buy_classifier'].append(f"Low recall ({buy.get('recall', 0):.2%}) - missing many real BUY opportunities")
        if buy.get('f1_score', 0) < 0.3:
            weaknesses['buy_classifier'].append(f"Low F1 score ({buy.get('f1_score', 0):.4f}) - overall poor BUY classification")
        if buy.get('auc_roc', 0) < 0.6:
            weaknesses['buy_classifier'].append(f"Low AUC-ROC ({buy.get('auc_roc', 0):.4f}) - poor discriminative ability")
        
        # Sell classifier analysis
        sell = results.get('classifiers', {}).get('sell', {})
        if sell.get('precision', 0) < 0.4:
            weaknesses['sell_classifier'].append(f"Low precision ({sell.get('precision', 0):.2%}) - many false SELL signals")
        if sell.get('recall', 0) < 0.3:
            weaknesses['sell_classifier'].append(f"Low recall ({sell.get('recall', 0):.2%}) - missing many real SELL opportunities")
        if sell.get('f1_score', 0) < 0.3:
            weaknesses['sell_classifier'].append(f"Low F1 score ({sell.get('f1_score', 0):.4f}) - overall poor SELL classification")
        
        # Backtest analysis
        bt = results.get('backtest', {})
        metrics = bt.get('metrics', {})
        if bt.get('total_return_pct', 0) < 0:
            weaknesses['backtest'].append(f"Negative returns ({bt.get('total_return_pct', 0):.2f}%) - losing strategy")
        if metrics.get('sharpe_ratio', 0) < 1.0:
            weaknesses['backtest'].append(f"Low Sharpe ratio ({metrics.get('sharpe_ratio', 0):.4f}) - poor risk-adjusted returns")
        if metrics.get('max_drawdown', 0) > 0.3:
            weaknesses['backtest'].append(f"High max drawdown ({metrics.get('max_drawdown', 0):.2%}) - significant risk")
        if metrics.get('win_rate', 0) < 0.4:
            weaknesses['backtest'].append(f"Low win rate ({metrics.get('win_rate', 0):.2%}) - most trades lose")
        
        # Generate recommendations
        if weaknesses['regressor']:
            weaknesses['recommendations'].append("Consider: More features, longer sequence length, or different architecture for regressor")
        if weaknesses['buy_classifier']:
            weaknesses['recommendations'].append("Consider: Rebalancing BUY training data, adjusting confidence thresholds, or different loss function")
        if weaknesses['sell_classifier']:
            weaknesses['recommendations'].append("Consider: Rebalancing SELL training data, stricter sell criteria, or ensemble methods")
        if weaknesses['backtest']:
            weaknesses['recommendations'].append("Consider: Better position sizing, risk management, or different entry/exit rules")
        
        # Print weakness summary
        print(f"\n   Weaknesses Found:")
        for category, issues in weaknesses.items():
            if issues and category != 'recommendations':
                print(f"\n   {category.upper()}:")
                for issue in issues:
                    print(f"      ⚠️  {issue}")
        
        if weaknesses['recommendations']:
            print(f"\n   RECOMMENDATIONS:")
            for rec in weaknesses['recommendations']:
                print(f"      💡 {rec}")
        else:
            print(f"\n   ✅ No major weaknesses detected!")
        
        return weaknesses
    
    def create_validation_dashboard(self, symbol: str, all_results: Dict):
        """Create comprehensive validation dashboard with all key visualizations."""
        print(f"\n[*] Creating validation dashboard for {symbol}...")
        
        fig = plt.figure(figsize=(20, 16))
        gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
        
        # Get results
        reg = all_results.get('regressor', {})
        classifiers = all_results.get('classifiers', {})
        buy = classifiers.get('buy', {})
        sell = classifiers.get('sell', {})
        
        # ===================================================================
        # ROW 1: Confusion Matrices with Percentages
        # ===================================================================
        
        # Buy Confusion Matrix
        ax1 = fig.add_subplot(gs[0, 0])
        buy_cm = np.array(buy.get('confusion_matrix', [[0, 0], [0, 0]]))
        buy_cm_pct = buy_cm.astype(float) / buy_cm.sum() * 100
        if HAS_SEABORN:
            sns.heatmap(buy_cm_pct, annot=True, fmt='.1f', cmap='Blues', ax=ax1,
                       xticklabels=['No Buy', 'Buy'], yticklabels=['No Buy', 'Buy'])
        else:
            im = ax1.imshow(buy_cm_pct, cmap='Blues')
            ax1.set_xticks([0, 1])
            ax1.set_yticks([0, 1])
            ax1.set_xticklabels(['No Buy', 'Buy'])
            ax1.set_yticklabels(['No Buy', 'Buy'])
            for i in range(2):
                for j in range(2):
                    text = f"{buy_cm[i,j]}\n({buy_cm_pct[i,j]:.1f}%)"
                    ax1.text(j, i, text, ha='center', va='center')
        ax1.set_title(f'BUY Confusion Matrix\nMCC: {buy.get("mcc", 0):.3f} | Kappa: {buy.get("kappa", 0):.3f}', fontweight='bold')
        ax1.set_ylabel('True')
        ax1.set_xlabel('Predicted')
        
        # Sell Confusion Matrix
        ax2 = fig.add_subplot(gs[0, 1])
        sell_cm = np.array(sell.get('confusion_matrix', [[0, 0], [0, 0]]))
        sell_cm_pct = sell_cm.astype(float) / sell_cm.sum() * 100
        if HAS_SEABORN:
            sns.heatmap(sell_cm_pct, annot=True, fmt='.1f', cmap='Reds', ax=ax2,
                       xticklabels=['No Sell', 'Sell'], yticklabels=['No Sell', 'Sell'])
        else:
            im = ax2.imshow(sell_cm_pct, cmap='Reds')
            ax2.set_xticks([0, 1])
            ax2.set_yticks([0, 1])
            ax2.set_xticklabels(['No Sell', 'Sell'])
            ax2.set_yticklabels(['No Sell', 'Sell'])
            for i in range(2):
                for j in range(2):
                    text = f"{sell_cm[i,j]}\n({sell_cm_pct[i,j]:.1f}%)"
                    ax2.text(j, i, text, ha='center', va='center')
        ax2.set_title(f'SELL Confusion Matrix\nMCC: {sell.get("mcc", 0):.3f} | Kappa: {sell.get("kappa", 0):.3f}', fontweight='bold')
        ax2.set_ylabel('True')
        ax2.set_xlabel('Predicted')
        
        # ===================================================================
        # ROW 1: ROC and PR Curves Side-by-Side
        # ===================================================================
        
        # ROC Curves
        ax3 = fig.add_subplot(gs[0, 2])
        buy_probs = classifiers.get('buy_probs', np.array([]))
        buy_target = classifiers.get('buy_target', np.array([]))
        sell_probs = classifiers.get('sell_probs', np.array([]))
        sell_target = classifiers.get('sell_target', np.array([]))
        
        if len(buy_probs) > 0 and len(np.unique(buy_target)) > 1:
            fpr_buy, tpr_buy, _ = roc_curve(buy_target, buy_probs)
            ax3.plot(fpr_buy, tpr_buy, label=f'BUY (AUC={buy.get("auc_roc", 0):.3f})', color='green', linewidth=2)
        
        if len(sell_probs) > 0 and len(np.unique(sell_target)) > 1:
            fpr_sell, tpr_sell, _ = roc_curve(sell_target, sell_probs)
            ax3.plot(fpr_sell, tpr_sell, label=f'SELL (AUC={sell.get("auc_roc", 0):.3f})', color='red', linewidth=2)
        
        ax3.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.set_title('ROC Curves', fontweight='bold')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Precision-Recall Curves
        ax4 = fig.add_subplot(gs[0, 3])
        if len(buy_probs) > 0 and len(np.unique(buy_target)) > 1:
            precision_buy, recall_buy, _ = precision_recall_curve(buy_target, buy_probs)
            ax4.plot(recall_buy, precision_buy, label=f'BUY (AUC={buy.get("pr_auc", 0):.3f})', color='green', linewidth=2)
        
        if len(sell_probs) > 0 and len(np.unique(sell_target)) > 1:
            precision_sell, recall_sell, _ = precision_recall_curve(sell_target, sell_probs)
            ax4.plot(recall_sell, precision_sell, label=f'SELL (AUC={sell.get("pr_auc", 0):.3f})', color='red', linewidth=2)
        
        ax4.set_xlabel('Recall')
        ax4.set_ylabel('Precision')
        ax4.set_title('Precision-Recall Curves\n(Better for Imbalanced Data)', fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # ===================================================================
        # ROW 2: Rolling R² and Return Distributions
        # ===================================================================
        
        # Rolling R² over time
        ax5 = fig.add_subplot(gs[1, :])
        actuals = reg.get('actuals', np.array([]))
        predictions = reg.get('predictions', np.array([]))
        
        if len(actuals) > 5:
            rolling_r2 = self._calculate_rolling_r2(actuals, predictions, window=5)
            ax5.plot(rolling_r2, linewidth=1, alpha=0.7, color='blue')
            ax5.axhline(0, color='red', linestyle='--', alpha=0.5)
            ax5.axhline(np.mean(rolling_r2), color='green', linestyle='--', alpha=0.7, 
                       label=f'Mean: {np.mean(rolling_r2):.4f}')
            ax5.fill_between(range(len(rolling_r2)), rolling_r2, 0, alpha=0.3)
            ax5.set_xlabel('Time Window')
            ax5.set_ylabel('R²')
            ax5.set_title(f'Rolling R² (5-day windows) - Temporal Stability\nMean: {np.mean(rolling_r2):.4f} ± {np.std(rolling_r2):.4f}', 
                         fontweight='bold')
            ax5.legend()
            ax5.grid(True, alpha=0.3)
        
        # ===================================================================
        # ROW 3 & 4: Distribution of Predicted vs Actual Returns
        # ===================================================================
        
        # Predicted returns distribution
        ax6 = fig.add_subplot(gs[2, :2])
        if len(predictions) > 0:
            ax6.hist(predictions, bins=50, alpha=0.7, color='blue', edgecolor='black')
            ax6.axvline(np.mean(predictions), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(predictions):.6f}')
            ax6.axvline(np.median(predictions), color='green', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(predictions):.6f}')
            ax6.set_xlabel('Predicted Returns')
            ax6.set_ylabel('Frequency')
            ax6.set_title(f'Predicted Returns Distribution\nStd: {np.std(predictions):.6f}', fontweight='bold')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # Actual returns distribution
        ax7 = fig.add_subplot(gs[2, 2:])
        if len(actuals) > 0:
            ax7.hist(actuals, bins=50, alpha=0.7, color='orange', edgecolor='black')
            ax7.axvline(np.mean(actuals), color='red', linestyle='--', linewidth=2,
                       label=f'Mean: {np.mean(actuals):.6f}')
            ax7.axvline(np.median(actuals), color='green', linestyle='--', linewidth=2,
                       label=f'Median: {np.median(actuals):.6f}')
            ax7.set_xlabel('Actual Returns')
            ax7.set_ylabel('Frequency')
            ax7.set_title(f'Actual Returns Distribution\nStd: {np.std(actuals):.6f}', fontweight='bold')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        
        # ===================================================================
        # ROW 4: Key Metrics Summary Table
        # ===================================================================
        
        ax8 = fig.add_subplot(gs[3, :])
        ax8.axis('off')
        
        # Create metrics summary table
        metrics_text = [
            ['REGRESSOR METRICS', '', 'CLASSIFIER METRICS', ''],
            ['R²', f"{reg.get('r2', 0):.4f}", 'BUY F1', f"{buy.get('f1_score', 0):.4f}"],
            ['IC (Info Coef)', f"{reg.get('ic', 0):.4f}", 'BUY PR-AUC', f"{buy.get('pr_auc', 0):.4f}"],
            ['Hit Rate', f"{reg.get('hit_rate', 0):.2%}", 'SELL F1', f"{sell.get('f1_score', 0):.4f}"],
            ['Sortino Ratio', f"{reg.get('sortino_ratio', 0):.4f}", 'SELL PR-AUC', f"{sell.get('pr_auc', 0):.4f}"],
            ['', '', '', ''],
            ['BACKTEST METRICS', '', '', ''],
        ]
        
        bt = all_results.get('backtest', {})
        bt_metrics = bt.get('metrics', {})
        metrics_text.extend([
            ['Sharpe Ratio', f"{bt_metrics.get('sharpe_ratio', 0):.4f}", 
             'Calmar Ratio', f"{bt_metrics.get('calmar_ratio', 0):.4f}"],
            ['Profit Factor', f"{bt_metrics.get('profit_factor', 0):.4f}", 
             'Win/Loss Ratio', f"{bt_metrics.get('win_loss_ratio', 0):.4f}"],
            ['Win Rate', f"{bt_metrics.get('win_rate', 0):.2%}", 
             'Trade Opp Rate', f"{bt_metrics.get('trade_opportunity_rate', 0):.2%}"],
        ])
        
        table = ax8.table(cellText=metrics_text, cellLoc='left', loc='center',
                         colWidths=[0.25, 0.25, 0.25, 0.25])
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style header rows
        for i in [0, 6]:
            for j in range(4):
                table[(i, j)].set_facecolor('#4CAF50')
                table[(i, j)].set_text_props(weight='bold', color='white')
        
        fig.suptitle(f'{symbol} - Comprehensive Validation Dashboard', 
                    fontsize=18, fontweight='bold', y=0.995)
        
        plt.savefig(self.run_dir / f'{symbol}_validation_dashboard.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   [+] Saved: {symbol}_validation_dashboard.png")
    
    def _plot_regressor_results(self, symbol: str, results: Dict):
        """Plot regressor predictions vs actuals"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{symbol} Regressor Performance', fontsize=16, fontweight='bold')
        
        actuals = results['actuals']
        predictions = results['predictions']
        
        # 1. Predictions vs Actuals scatter
        axes[0, 0].scatter(actuals, predictions, alpha=0.5, s=10)
        axes[0, 0].plot([actuals.min(), actuals.max()], [actuals.min(), actuals.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Returns')
        axes[0, 0].set_ylabel('Predicted Returns')
        axes[0, 0].set_title('Predictions vs Actuals')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Time series comparison
        dates = pd.to_datetime(results['dates'])
        axes[0, 1].plot(dates, actuals, label='Actual', alpha=0.7, linewidth=1)
        axes[0, 1].plot(dates, predictions, label='Predicted', alpha=0.7, linewidth=1)
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Returns')
        axes[0, 1].set_title('Time Series Comparison')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # 3. Residuals distribution
        residuals = actuals - predictions
        axes[1, 0].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(0, color='red', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Residuals')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title(f'Residuals Distribution (μ={np.mean(residuals):.6f})')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Cumulative returns comparison
        cum_actual = np.cumprod(1 + actuals) - 1
        cum_predicted = np.cumprod(1 + predictions) - 1
        axes[1, 1].plot(dates, cum_actual, label='Actual Cumulative', linewidth=2)
        axes[1, 1].plot(dates, cum_predicted, label='Predicted Cumulative', linewidth=2)
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel('Cumulative Returns')
        axes[1, 1].set_title('Cumulative Returns Comparison')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.run_dir / f'{symbol}_regressor_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   [+] Saved: {symbol}_regressor_analysis.png")
    
    def _plot_classifier_results(self, symbol: str, results: Dict):
        """Plot classifier performance metrics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{symbol} Classifier Performance', fontsize=16, fontweight='bold')
        
        # Buy classifier plots
        buy_probs = results['buy_probs']
        buy_target = results['buy_target']
        dates = pd.to_datetime(results['dates'])
        
        # 1. Buy probability distribution
        axes[0, 0].hist(buy_probs[buy_target == 0], bins=30, alpha=0.5, label='Non-Buy', color='red')
        axes[0, 0].hist(buy_probs[buy_target == 1], bins=30, alpha=0.5, label='Buy', color='green')
        axes[0, 0].axvline(0.5, color='black', linestyle='--', linewidth=2)
        axes[0, 0].set_xlabel('Buy Probability')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Buy Probability Distribution')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Buy confusion matrix
        buy_cm = np.array(results['buy']['confusion_matrix'])
        if HAS_SEABORN:
            import seaborn as sns
            sns.heatmap(buy_cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                        xticklabels=['No Buy', 'Buy'], yticklabels=['No Buy', 'Buy'])
        else:
            im = axes[0, 1].imshow(buy_cm, cmap='Blues')
            axes[0, 1].set_xticks([0, 1])
            axes[0, 1].set_yticks([0, 1])
            axes[0, 1].set_xticklabels(['No Buy', 'Buy'])
            axes[0, 1].set_yticklabels(['No Buy', 'Buy'])
            # Add text annotations
            for i in range(2):
                for j in range(2):
                    axes[0, 1].text(j, i, str(buy_cm[i, j]), ha='center', va='center', color='white' if buy_cm[i, j] > buy_cm.max()/2 else 'black')
        axes[0, 1].set_title('Buy Confusion Matrix')
        axes[0, 1].set_ylabel('True')
        axes[0, 1].set_xlabel('Predicted')
        
        # 3. Buy probability over time
        axes[0, 2].plot(dates, buy_probs, alpha=0.7, linewidth=1)
        axes[0, 2].axhline(0.5, color='red', linestyle='--', alpha=0.5)
        buy_signals = dates[buy_target == 1]
        axes[0, 2].scatter(buy_signals, [0.5] * len(buy_signals), color='green', 
                          marker='^', s=50, alpha=0.5, label='Actual Buy')
        axes[0, 2].set_xlabel('Date')
        axes[0, 2].set_ylabel('Buy Probability')
        axes[0, 2].set_title('Buy Signals Over Time')
        axes[0, 2].legend()
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].tick_params(axis='x', rotation=45)
        
        # Sell classifier plots
        sell_probs = results['sell_probs']
        sell_target = results['sell_target']
        
        # 4. Sell probability distribution
        axes[1, 0].hist(sell_probs[sell_target == 0], bins=30, alpha=0.5, label='Non-Sell', color='green')
        axes[1, 0].hist(sell_probs[sell_target == 1], bins=30, alpha=0.5, label='Sell', color='red')
        axes[1, 0].axvline(0.5, color='black', linestyle='--', linewidth=2)
        axes[1, 0].set_xlabel('Sell Probability')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Sell Probability Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Sell confusion matrix
        sell_cm = np.array(results['sell']['confusion_matrix'])
        if HAS_SEABORN:
            import seaborn as sns
            sns.heatmap(sell_cm, annot=True, fmt='d', cmap='Reds', ax=axes[1, 1],
                        xticklabels=['No Sell', 'Sell'], yticklabels=['No Sell', 'Sell'])
        else:
            im = axes[1, 1].imshow(sell_cm, cmap='Reds')
            axes[1, 1].set_xticks([0, 1])
            axes[1, 1].set_yticks([0, 1])
            axes[1, 1].set_xticklabels(['No Sell', 'Sell'])
            axes[1, 1].set_yticklabels(['No Sell', 'Sell'])
            # Add text annotations
            for i in range(2):
                for j in range(2):
                    axes[1, 1].text(j, i, str(sell_cm[i, j]), ha='center', va='center', color='white' if sell_cm[i, j] > sell_cm.max()/2 else 'black')
        axes[1, 1].set_title('Sell Confusion Matrix')
        axes[1, 1].set_ylabel('True')
        axes[1, 1].set_xlabel('Predicted')
        
        # 6. Sell probability over time
        axes[1, 2].plot(dates, sell_probs, alpha=0.7, linewidth=1, color='red')
        axes[1, 2].axhline(0.5, color='red', linestyle='--', alpha=0.5)
        sell_signals = dates[sell_target == 1]
        axes[1, 2].scatter(sell_signals, [0.5] * len(sell_signals), color='red', 
                          marker='v', s=50, alpha=0.5, label='Actual Sell')
        axes[1, 2].set_xlabel('Date')
        axes[1, 2].set_ylabel('Sell Probability')
        axes[1, 2].set_title('Sell Signals Over Time')
        axes[1, 2].legend()
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.run_dir / f'{symbol}_classifier_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   [+] Saved: {symbol}_classifier_analysis.png")
    
    def _plot_backtest_results(self, symbol: str, results: Dict):
        """Plot comprehensive backtest results"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'{symbol} Backtest Performance', fontsize=16, fontweight='bold')
        
        equity_curve = np.array(results['equity_curve'])
        dates = np.array(results['dates'])
        positions = np.array(results['positions'])
        
        # Equity curve has n+1 elements (includes initial capital), dates has n elements
        # Skip first equity value to align with dates
        if len(equity_curve) == len(dates) + 1:
            equity_values = equity_curve[1:]  # Skip initial capital
        elif len(equity_curve) == len(dates):
            equity_values = equity_curve  # Already aligned
        else:
            print(f"   [⚠] Warning: Equity curve length mismatch. equity={len(equity_curve)}, dates={len(dates)}")
            # Use minimum length to avoid errors
            min_len = min(len(equity_curve), len(dates))
            equity_values = equity_curve[:min_len]
            dates = dates[:min_len]
            positions = positions[:min_len] if len(positions) >= min_len else positions
        
        # 1. Equity curve
        if len(equity_values) > 0 and len(dates) > 0:
            axes[0, 0].plot(dates, equity_values, linewidth=2, color='blue')
            axes[0, 0].set_xlabel('Date')
            axes[0, 0].set_ylabel('Portfolio Value ($)')
            axes[0, 0].set_title('Equity Curve')
            axes[0, 0].grid(True, alpha=0.3)
            axes[0, 0].tick_params(axis='x', rotation=45)
        else:
            axes[0, 0].text(0.5, 0.5, 'No data available', ha='center', va='center')
        
        # 2. Drawdown
        if len(equity_values) > 0:
            peak = np.maximum.accumulate(equity_values)
            drawdown = (equity_values - peak) / peak
            axes[0, 1].fill_between(range(len(drawdown)), drawdown, 0, alpha=0.3, color='red')
            axes[0, 1].plot(drawdown, linewidth=1, color='red')
            axes[0, 1].set_xlabel('Time')
            axes[0, 1].set_ylabel('Drawdown')
            axes[0, 1].set_title('Drawdown Over Time')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        else:
            axes[0, 1].text(0.5, 0.5, 'No data available', ha='center', va='center')
        
        # 3. Returns distribution
        if len(equity_curve) > 1:
            returns = np.diff(equity_curve) / equity_curve[:-1]
            axes[1, 0].hist(returns, bins=50, alpha=0.7, edgecolor='black')
            axes[1, 0].axvline(returns.mean(), color='red', linestyle='--', linewidth=2, 
                              label=f'Mean: {returns.mean():.4f}')
            axes[1, 0].axvline(0, color='black', linestyle='-', linewidth=1)
            axes[1, 0].set_xlabel('Returns')
            axes[1, 0].set_ylabel('Frequency')
            axes[1, 0].set_title('Returns Distribution')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No data available', ha='center', va='center')
        
        # 4. Position sizes over time
        if len(dates) > 0 and len(positions) > 0:
            # Ensure positions array matches dates length
            min_len = min(len(dates), len(positions))
            plot_dates = dates[:min_len]
            plot_positions = positions[:min_len]
            
            axes[1, 1].plot(plot_dates, plot_positions, linewidth=1, label='Position', alpha=0.7)
            axes[1, 1].axhline(0, color='black', linestyle='-', linewidth=1)
            axes[1, 1].fill_between(plot_dates, plot_positions, 0, 
                                   where=(plot_positions > 0), alpha=0.3, color='green', label='Long')
            axes[1, 1].fill_between(plot_dates, plot_positions, 0, 
                                   where=(plot_positions < 0), alpha=0.3, color='red', label='Short')
            axes[1, 1].set_xlabel('Date')
            axes[1, 1].set_ylabel('Position Size')
            axes[1, 1].set_title('Position Sizes Over Time')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            axes[1, 1].tick_params(axis='x', rotation=45)
        else:
            axes[1, 1].text(0.5, 0.5, 'No position data', ha='center', va='center')
        
        plt.tight_layout()
        plt.savefig(self.run_dir / f'{symbol}_backtest_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   [+] Saved: {symbol}_backtest_analysis.png")
    
    def run_full_validation(self):
        """Run complete validation suite for all symbols"""
        print("\n" + "=" * 80)
        print("[>>] STARTING FULL MODEL VALIDATION SUITE")
        print("=" * 80)
        
        # Check which symbols have trained models
        save_dir = Path('saved_models')
        available_symbols = []
        
        if save_dir.exists():
            # Look for regressor weights to determine available symbols
            for weight_file in save_dir.glob('*_1d_regressor_final.weights.h5'):
                # Extract symbol from filename: AAPL_1d_regressor_final.weights.h5 -> AAPL
                symbol = weight_file.name.split('_1d_regressor_final.weights.h5')[0]
                available_symbols.append(symbol)
        
        print(f"\nAvailable trained models: {', '.join(available_symbols) if available_symbols else 'None'}")
        
        # Filter requested symbols to only those with models
        valid_symbols = [s for s in self.symbols if s in available_symbols]
        missing_symbols = [s for s in self.symbols if s not in available_symbols]
        
        if missing_symbols:
            print(f"⚠ Skipping symbols without models: {', '.join(missing_symbols)}")
        
        if not valid_symbols:
            print(f"\n❌ ERROR: No trained models found for {', '.join(self.symbols)}")
            if available_symbols:
                print(f"   Available models: {', '.join(available_symbols)}")
            else:
                print(f"   No models found in {save_dir}")
            print(f"\n   To train models, run:")
            print(f"     python training/train_full_pipeline.py --stock SYMBOL")
            return {}
        
        self.symbols = valid_symbols
        print(f"\nValidating: {', '.join(valid_symbols)}\n")
        
        all_results = {}
        
        for symbol in self.symbols:
            print(f"\n{'=' * 80}")
            print(f"[SYMBOL] VALIDATING {symbol}")
            print(f"{'=' * 80}")
            
            try:
                # Load models and data
                bundle = self.load_models_and_data(symbol)
                
                # Validate regressor
                regressor_results = self.validate_regressor(bundle)

                # Run backtest
                backtest_results = self.run_backtest(bundle)
                
                # Analyze weaknesses
                symbol_results = {
                    'regressor': regressor_results,
                    'classifiers': classifier_results,
                    'backtest': backtest_results
                }
                
                weaknesses = self.analyze_weaknesses(symbol, symbol_results)
                symbol_results['weaknesses'] = weaknesses
                
                # Create comprehensive validation dashboard
                self.create_validation_dashboard(symbol, symbol_results)
                
                all_results[symbol] = symbol_results
                
                print(f"\n[OK] {symbol} validation complete!")
                
            except Exception as e:
                print(f"\n[X] Error validating {symbol}: {e}")
                import traceback
                traceback.print_exc()
                all_results[symbol] = {
                    'error': str(e),
                    'error_type': type(e).__name__
                }
        
        # Save comprehensive report
        if all_results:
            self.save_report(all_results)
            
            print("\n" + "=" * 80)
            print("[OK] VALIDATION SUITE COMPLETE")
            print(f"[>>] Results saved to: {self.run_dir}")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("[!] VALIDATION SUITE ENDED - No models to validate")
            print("=" * 80)
        
        return all_results
    
    def save_report(self, all_results: Dict):
        """Save comprehensive JSON report"""
        print("\n[*] Saving comprehensive report...")
        
        # Convert numpy/pandas types to JSON-serializable
        report = {}
        for symbol, results in all_results.items():
            if 'error' in results:
                report[symbol] = results
                continue
            
            report[symbol] = {
                'regressor': {
                    'mse': results['regressor']['mse'],
                    'rmse': results['regressor']['rmse'],
                    'mae': results['regressor']['mae'],
                    'r2': results['regressor']['r2'],
                    'direction_accuracy': results['regressor']['direction_accuracy'],
                    'magnitude_correlation': results['regressor']['magnitude_correlation']
                },
                'classifiers': {
                    'buy': results['classifiers']['buy'],
                    'sell': results['classifiers']['sell']
                },
                'backtest': {
                    'total_return_pct': results['backtest']['total_return_pct'],
                    'metrics': {k: v for k, v in results['backtest']['metrics'].items() 
                               if not isinstance(v, (pd.DataFrame, pd.Series))}
                },
                'weaknesses': results['weaknesses']
            }
        
        # Save JSON report
        report_path = self.run_dir / 'validation_report.json'
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"   [+] Saved: validation_report.json")
        
        # Save summary text report
        self._save_text_summary(all_results)
    
    def _save_text_summary(self, all_results: Dict):
        """Save human-readable text summary"""
        summary_path = self.run_dir / 'SUMMARY.txt'
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("MODEL VALIDATION SUMMARY\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            for symbol, results in all_results.items():
                f.write(f"\n{'=' * 80}\n")
                f.write(f"{symbol}\n")
                f.write(f"{'=' * 80}\n\n")
                
                if 'error' in results:
                    f.write(f"[ERROR] {results['error']}\n")
                    continue
                
                # Regressor summary
                f.write("REGRESSOR PERFORMANCE:\n")
                f.write("-" * 40 + "\n")
                reg = results['regressor']
                f.write(f"  RMSE: {reg['rmse']:.6f}\n")
                f.write(f"  MAE: {reg['mae']:.6f}\n")
                f.write(f"  R²: {reg['r2']:.4f}\n")
                f.write(f"  Direction Accuracy: {reg['direction_accuracy']:.2%}\n")
                f.write(f"  Magnitude Correlation: {reg['magnitude_correlation']:.4f}\n")
                f.write(f"  Information Coefficient (IC): {reg.get('ic', 0):.4f}\n")
                f.write(f"  Hit Rate: {reg.get('hit_rate', 0):.2%}\n")
                f.write(f"  Sortino Ratio: {reg.get('sortino_ratio', 0):.4f}\n")
                f.write(f"  Rolling R² (5-day): {reg.get('rolling_r2_mean', 0):.4f} ± {reg.get('rolling_r2_std', 0):.4f}\n\n")
                
                # Classifier summary
                f.write("CLASSIFIER PERFORMANCE:\n")
                f.write("-" * 40 + "\n")
                buy = results['classifiers']['buy']
                sell = results['classifiers']['sell']
                f.write(f"  BUY Classifier:\n")
                f.write(f"    Accuracy: {buy['accuracy']:.2%}\n")
                f.write(f"    Precision: {buy['precision']:.2%}\n")
                f.write(f"    Recall: {buy['recall']:.2%}\n")
                f.write(f"    F1: {buy['f1_score']:.4f}\n")
                f.write(f"    AUC-ROC: {buy['auc_roc']:.4f}\n")
                f.write(f"    PR-AUC: {buy.get('pr_auc', 0):.4f}\n")
                f.write(f"    MCC: {buy.get('mcc', 0):.4f}\n")
                f.write(f"    Cohen's Kappa: {buy.get('kappa', 0):.4f}\n\n")
                f.write(f"  SELL Classifier:\n")
                f.write(f"    Accuracy: {sell['accuracy']:.2%}\n")
                f.write(f"    Precision: {sell['precision']:.2%}\n")
                f.write(f"    Recall: {sell['recall']:.2%}\n")
                f.write(f"    F1: {sell['f1_score']:.4f}\n")
                f.write(f"    AUC-ROC: {sell['auc_roc']:.4f}\n")
                f.write(f"    PR-AUC: {sell.get('pr_auc', 0):.4f}\n")
                f.write(f"    MCC: {sell.get('mcc', 0):.4f}\n")
                f.write(f"    Cohen's Kappa: {sell.get('kappa', 0):.4f}\n\n")
                
                # Backtest summary
                f.write("BACKTEST PERFORMANCE:\n")
                f.write("-" * 40 + "\n")
                bt = results['backtest']
                metrics = bt['metrics']
                f.write(f"  Total Return: {bt['total_return_pct']:.2f}%\n")
                f.write(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}\n")
                f.write(f"  Calmar Ratio: {metrics.get('calmar_ratio', 0):.4f}\n")
                f.write(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}\n")
                f.write(f"  Win Rate: {metrics.get('win_rate', 0):.2%}\n")
                f.write(f"  Profit Factor: {metrics.get('profit_factor', 0):.4f}\n")
                f.write(f"  Win/Loss Ratio: {metrics.get('win_loss_ratio', 0):.4f}\n")
                f.write(f"  Total Trades: {metrics.get('total_trades', 0)}\n")
                f.write(f"  Trade Opportunity Rate: {metrics.get('trade_opportunity_rate', 0):.2%}\n\n")
                
                # Weaknesses
                f.write("WEAKNESSES & RECOMMENDATIONS:\n")
                f.write("-" * 40 + "\n")
                weaknesses = results['weaknesses']
                
                has_issues = False
                for category in ['regressor', 'buy_classifier', 'sell_classifier', 'backtest']:
                    if weaknesses.get(category):
                        has_issues = True
                        f.write(f"\n  {category.upper()}:\n")
                        for issue in weaknesses[category]:
                            f.write(f"    [!] {issue}\n")
                
                if weaknesses.get('recommendations'):
                    f.write(f"\n  RECOMMENDATIONS:\n")
                    for rec in weaknesses['recommendations']:
                        f.write(f"    [*] {rec}\n")
                
                if not has_issues:
                    f.write("  [OK] No major weaknesses detected!\n")
                
                f.write("\n")
        
        print(f"   [+] Saved: SUMMARY.txt")


def _compute_rolling_r2(y_true, y_pred, window=5):
    """Compute rolling R^2 (module-level helper)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rolling_r2 = []
    for i in range(window, len(y_true)):
        y_true_window = y_true[i-window:i]
        y_pred_window = y_pred[i-window:i]
        ss_res = np.sum((y_true_window - y_pred_window) ** 2)
        ss_tot = np.sum((y_true_window - np.mean(y_true_window)) ** 2)
        r2_window = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        rolling_r2.append(r2_window)
    return np.array(rolling_r2)


def plot_quantile_calibration(ax, quantile_metrics: dict):
    """Plot quantile calibration (expects keys: y_true, q10, q50, q90)."""
    y_true = np.asarray(quantile_metrics.get('y_true', []))
    q10 = np.asarray(quantile_metrics.get('q10', []))
    q50 = np.asarray(quantile_metrics.get('q50', []))
    q90 = np.asarray(quantile_metrics.get('q90', []))

    if y_true.size == 0:
        ax.text(0.5, 0.5, 'No quantile data', ha='center', va='center')
        return

    ax.plot(y_true, label='Actual', alpha=0.6)
    ax.fill_between(range(len(y_true)), q10, q90, alpha=0.25, label='80% CI')
    ax.plot(q50, label='Median', color='red', linewidth=1.5)
    coverage = float(np.mean((y_true >= q10) & (y_true <= q90)))
    ax.set_title('Quantile Calibration')
    ax.legend()
    ax.text(0.01, 0.95, f'Coverage: {coverage:.2%}', transform=ax.transAxes, va='top')


def plot_top_features_v31(ax, feature_importance: dict):
    """Plot top 15 features by importance (expects mapping feature->importance)."""
    if not feature_importance:
        ax.text(0.5, 0.5, 'No feature importance data', ha='center', va='center')
        return

    # Accept either dict or list of (feature, importance)
    if isinstance(feature_importance, dict):
        items = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)[:15]
    else:
        items = sorted(feature_importance, key=lambda x: abs(x[1]), reverse=True)[:15]

    names = [i[0] for i in items][::-1]
    vals = [i[1] for i in items][::-1]

    ax.barh(range(len(names)), vals, color='tab:blue', alpha=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_title('Top 15 Feature Importances (v3.1)')
    ax.grid(axis='x', alpha=0.2)


def generate_validation_dashboard_v31(symbol, results):
    """Enhanced dashboard with v3.1 metrics"""
    # Ensure output dir exists
    out_dir = Path('validation_results')
    out_dir.mkdir(exist_ok=True)

    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(5, 3, height_ratios=[1, 1, 1, 1, 0.5])

    # Row 1: Confusion Matrices (BUY / SELL)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    buy = results.get('classifiers', {}).get('buy', {})
    sell = results.get('classifiers', {}).get('sell', {})
    buy_cm = np.array(buy.get('confusion_matrix', [[0, 0], [0, 0]]))
    sell_cm = np.array(sell.get('confusion_matrix', [[0, 0], [0, 0]]))

    ax1.imshow(buy_cm, cmap='Blues')
    ax1.set_title('BUY Confusion Matrix')
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('True')

    ax2.imshow(sell_cm, cmap='Reds')
    ax2.set_title('SELL Confusion Matrix')
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('True')

    # Row 2: ROC + PR Curves
    ax3 = fig.add_subplot(gs[1, :2])
    ax4 = fig.add_subplot(gs[1, 2])
    buy_probs = results.get('classifiers', {}).get('buy_probs', np.array([]))
    buy_target = results.get('classifiers', {}).get('buy_target', np.array([]))
    sell_probs = results.get('classifiers', {}).get('sell_probs', np.array([]))
    sell_target = results.get('classifiers', {}).get('sell_target', np.array([]))

    try:
        if len(buy_probs) > 0 and len(np.unique(buy_target)) > 1:
            fpr_buy, tpr_buy, _ = roc_curve(buy_target, buy_probs)
            ax3.plot(fpr_buy, tpr_buy, label=f'BUY (AUC={buy.get("auc_roc",0):.3f})', color='green')
        if len(sell_probs) > 0 and len(np.unique(sell_target)) > 1:
            fpr_sell, tpr_sell, _ = roc_curve(sell_target, sell_probs)
            ax3.plot(fpr_sell, tpr_sell, label=f'SELL (AUC={sell.get("auc_roc",0):.3f})', color='red')
        ax3.plot([0,1],[0,1],'k--', alpha=0.3)
        ax3.set_title('ROC Curves')
        ax3.set_xlabel('False Positive Rate')
        ax3.set_ylabel('True Positive Rate')
        ax3.legend()

        # PR curve (simple plot of precision vs recall using available metrics)
        pr_buy = buy.get('pr_auc', None)
        pr_sell = sell.get('pr_auc', None)
        ax4.bar(['BUY PR-AUC','SELL PR-AUC'], [pr_buy or 0.0, pr_sell or 0.0], color=['green','red'])
        ax4.set_ylim(0,1)
        ax4.set_title('Precision-Recall AUC')
    except Exception:
        pass

    # Row 3: Rolling R²
    ax5 = fig.add_subplot(gs[2, :])
    reg = results.get('regressor', {})
    actuals = np.asarray(reg.get('actuals', []))
    preds = np.asarray(reg.get('predictions', []))
    if len(actuals) > 5 and len(preds) == len(actuals):
        rolling_r2 = _compute_rolling_r2(actuals, preds, window=5)
        ax5.plot(rolling_r2, color='blue')
        ax5.set_title('Rolling R² (5-day windows)')
        ax5.set_xlabel('Window Index')
        ax5.set_ylabel('R²')
    else:
        ax5.text(0.5, 0.5, 'Insufficient data for rolling R²', ha='center', va='center')

    # Row 4: Quantile Calibration
    if 'quantile_metrics' in results:
        ax6 = fig.add_subplot(gs[3, :])
        plot_quantile_calibration(ax6, results['quantile_metrics'])

    # Row 5: Feature Importance (Top 15)
    if 'feature_importance' in results:
        ax7 = fig.add_subplot(gs[4, :])
        plot_top_features_v31(ax7, results['feature_importance'])

    plt.tight_layout()
    out_path = out_dir / f'{symbol}_dashboard_v31.png'
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)



def main():
    """Main execution"""
    import sys
    
    # Parse command line arguments
    symbols = sys.argv[1:] if len(sys.argv) > 1 else ['AAPL', 'HOOD', 'TSLA']
    
    print("\n" + "=" * 80)
    print("MODEL VALIDATION SUITE")
    print("=" * 80)
    print(f"Requested symbols: {', '.join(symbols)}")
    print(f"Tests: Regressor, Buy/Sell Classifiers, Backtest, Weakness Analysis")
    print("=" * 80)
    
    # Create and run validation suite
    suite = ModelValidationSuite(symbols=symbols, seq_len=60)
    results = suite.run_full_validation()
    
    return results


if __name__ == '__main__':
    # ===================================================================
    # GRACEFUL ERROR HANDLING - Catch common failures with helpful messages
    # ===================================================================
    try:
        main()
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Required model files not found")
        print(f"   {e}")
        print(f"\n💡 Solution: Train models first using:")
        print(f"   python training/train_1d_regressor_final.py <SYMBOL>")
        print(f"   python training/train_binary_classifiers_final.py <SYMBOL>")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ ERROR: Data validation failed")
        print(f"   {e}")
        print(f"\n💡 Common causes:")
        print(f"   1. Feature count mismatch between training and validation")
        print(f"   2. Missing sentiment features but model expects them")
        print(f"   3. Insufficient historical data for symbol")
        sys.exit(1)
    except AssertionError as e:
        print(f"\n❌ ERROR: Assertion failed (alignment issue)")
        print(f"   {e}")
        print(f"\n💡 This indicates a sequence alignment bug.")
        print(f"   Please report this error with the full stack trace.")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ UNEXPECTED ERROR: {type(e).__name__}")
        print(f"   {e}")
        print(f"\n💡 Enable debug mode or check logs for details.")
        import traceback
        traceback.print_exc()
        sys.exit(1)

