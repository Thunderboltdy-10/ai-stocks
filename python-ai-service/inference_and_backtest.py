"""
Inference + Fusion + Simple Backtest
- Loads saved scalers/features and model weights produced by the training scripts
- Rebuilds models with same architecture assumptions
- Produces regressor predictions (1-day), buy/sell probs, fused signals
- Runs a naive next-day backtest and prints metrics
"""

# P0 FIX: Import yfinance BEFORE TensorFlow to avoid SSL conflict with yfinance 0.2.66+
import yfinance as yf  # noqa: F401 - must be first

import pickle
from pathlib import Path
import numpy as np
import logging
import pandas as pd
import json
from pandas.tseries.offsets import BDay
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import RobustScaler
# Optional plotting for backtest reports
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None
from colorama import Fore, Style, init
init(autoreset=True)
from data.data_fetcher import fetch_stock_data
from data.feature_engineer import engineer_features, get_feature_columns, EXPECTED_FEATURE_COUNT
from data.target_engineering import prepare_training_data
from data.cache_manager import DataCacheManager
from models.lstm_transformer_paper import (
    LSTMTransformerPaper, 
    AntiCollapseDirectionalLoss, 
    DirectionalHuberLoss,
    AsymmetricDirectionalLoss,
    directional_accuracy_metric,
    create_directional_accuracy_metric
)
from models.patchtst import (
    PatchTST,
    RevIN,
    PatchEmbedding,
    LearnablePositionalEncoding,
    TransformerEncoderLayer
)
from evaluation.advanced_backtester import AdvancedBacktester
from evaluation.model_validation_suite import validate_feature_counts, validate_sequences
# Note: create_binary_classifier removed - binary classifiers deprecated December 2025
from inference.confidence_scorer import ConfidenceScorer

# Import ModelPaths for organized model storage (with fallback)
try:
    from utils.model_paths import ModelPaths, find_model_path
except ImportError:
    ModelPaths = None
    find_model_path = None

# Import StackingPredictor for stacking ensemble mode
try:
    from inference.stacking_predictor import StackingPredictor, load_stacking_predictor
    STACKING_AVAILABLE = True
except ImportError:
    StackingPredictor = None
    load_stacking_predictor = None
    STACKING_AVAILABLE = False
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

# Setup module-level logger
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ===================================================================
# PREDICTION DISTRIBUTION DIAGNOSTICS
# ===================================================================

def plot_prediction_distribution(predictions, actuals, symbol, output_dir='backtest_results'):
    """Compare predicted vs actual return distributions.
    
    Creates a two-panel figure:
    1. Histogram comparison of predicted vs actual returns
    2. Q-Q plot showing if predictions follow normal distribution
    
    Args:
        predictions: Array of predicted returns (unscaled, real return space)
        actuals: Array of actual returns
        symbol: Stock symbol for labeling
        output_dir: Directory to save the plot
    
    Returns:
        Path to saved plot, or None if plotting failed
    """
    import os
    
    if plt is None:
        print("[WARN] matplotlib not available, skipping prediction distribution plot")
        return None
    
    try:
        from scipy import stats
    except ImportError:
        print("[WARN] scipy not available for Q-Q plot, using simplified version")
        stats = None
    
    try:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Panel 1: Histogram comparison
        axes[0].hist(predictions, bins=50, alpha=0.7, label='Predicted', color='blue', density=True)
        axes[0].hist(actuals, bins=50, alpha=0.7, label='Actual', color='orange', density=True)
        axes[0].set_xlabel('Return')
        axes[0].set_ylabel('Density')
        axes[0].set_title(f'{symbol} - Return Distribution Comparison')
        axes[0].legend()
        axes[0].axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
        
        # Add statistics annotations
        pred_stats = f"Pred: μ={np.mean(predictions):.4f}, σ={np.std(predictions):.4f}"
        act_stats = f"Actual: μ={np.mean(actuals):.4f}, σ={np.std(actuals):.4f}"
        axes[0].annotate(pred_stats, xy=(0.02, 0.95), xycoords='axes fraction', 
                        fontsize=8, color='blue', va='top')
        axes[0].annotate(act_stats, xy=(0.02, 0.88), xycoords='axes fraction', 
                        fontsize=8, color='orange', va='top')
        
        # Panel 2: Q-Q plot
        if stats is not None:
            stats.probplot(predictions, dist="norm", plot=axes[1])
            axes[1].set_title(f'{symbol} - Prediction Q-Q Plot (Normal Distribution)')
        else:
            # Simplified Q-Q plot without scipy
            sorted_preds = np.sort(predictions)
            theoretical = np.linspace(-3, 3, len(sorted_preds))
            axes[1].scatter(theoretical, sorted_preds, alpha=0.5, s=10)
            axes[1].plot([-3, 3], [np.mean(predictions) - 3*np.std(predictions), 
                                   np.mean(predictions) + 3*np.std(predictions)], 
                        'r--', label='Normal reference')
            axes[1].set_xlabel('Theoretical Quantiles')
            axes[1].set_ylabel('Sample Quantiles')
            axes[1].set_title(f'{symbol} - Prediction Q-Q Plot (Simplified)')
            axes[1].legend()
        
        plt.tight_layout()
        
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, f'{symbol}_prediction_distribution.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"[OK] Saved prediction distribution plot: {save_path}")
        plt.close()
        
        return save_path
    except Exception as e:
        print(f"[WARN] Failed to create prediction distribution plot: {e}")
        if 'fig' in locals():
            plt.close(fig)
        return None


def validate_prediction_variance(predictions, min_std=0.001, min_negative_pct=0.30, min_positive_pct=0.30):
    """Validate that predictions have healthy variance and distribution.
    
    Checks:
    1. Standard deviation is above threshold (predictions not collapsed)
    2. At least min_negative_pct of predictions are negative
    3. At least min_positive_pct of predictions are positive
    
    Args:
        predictions: Array of predicted returns
        min_std: Minimum required standard deviation
        min_negative_pct: Minimum required percentage of negative predictions
        min_positive_pct: Minimum required percentage of positive predictions
    
    Returns:
        dict with validation results and statistics
    
    Raises:
        ValueError: If predictions fail validation (collapsed variance)
    """
    pred_std = float(np.std(predictions))
    pred_mean = float(np.mean(predictions))
    pred_min = float(np.min(predictions))
    pred_max = float(np.max(predictions))
    
    n_total = len(predictions)
    n_negative = int(np.sum(predictions < 0))
    n_positive = int(np.sum(predictions > 0))
    pct_negative = n_negative / n_total
    pct_positive = n_positive / n_total
    
    validation_result = {
        'std': pred_std,
        'mean': pred_mean,
        'min': pred_min,
        'max': pred_max,
        'n_total': n_total,
        'n_negative': n_negative,
        'n_positive': n_positive,
        'pct_negative': pct_negative,
        'pct_positive': pct_positive,
        'passed': True,
        'issues': []
    }
    
    # Check variance
    if pred_std < min_std:
        validation_result['passed'] = False
        validation_result['issues'].append(
            f"Predictions collapsed! Std={pred_std:.6f} < {min_std}. Model failed to learn variance."
        )
    
    # Check distribution balance
    if pct_negative < min_negative_pct:
        validation_result['issues'].append(
            f"Low negative predictions: {pct_negative:.1%} < {min_negative_pct:.0%} threshold"
        )
    
    if pct_positive < min_positive_pct:
        validation_result['issues'].append(
            f"Low positive predictions: {pct_positive:.1%} < {min_positive_pct:.0%} threshold"
        )
    
    return validation_result


# ===================================================================
# VALIDATION FUNCTIONS - Robust error handling
# ===================================================================

def validate_model_files(symbol: str, save_dir: Path):
    """Validate all required model files exist before attempting to load.
    
    Checks new organized structure first, then falls back to legacy paths.
    """
    # Try new organized path structure first if ModelPaths available
    if ModelPaths is not None:
        paths = ModelPaths(symbol)
        # Check if new structure has required files
        new_structure_valid = (
            paths.regressor.model.exists() and
            paths.regressor.feature_scaler.exists() and
            paths.regressor.target_scaler.exists()
        )
        if new_structure_valid:
            return True
    
    # Fall back to legacy paths
    required_files = [
        save_dir / f'{symbol}_1d_regressor_final.weights.h5',
        save_dir / f'{symbol}_1d_regressor_final_feature_scaler.pkl',
        save_dir / f'{symbol}_1d_regressor_final_target_scaler.pkl',
    ]

    missing = [str(p) for p in required_files if not Path(p).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required model artifacts for {symbol}:\n  - " + "\n  - ".join(missing))
    return True


def create_multitask_regressor(sequence_length: int, n_features: int, name: str = 'multitask_regressor', arch: dict | None = None):
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
        dropout=dropout,
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
    # Training uses: Dense(64) -> Dense(32) -> Dense(1, linear) with RandomNormal init
    magnitude_branch = keras.layers.Dense(
        64,  # Fixed: training uses 64, was incorrectly 32
        activation='relu',
        kernel_initializer=keras.initializers.HeNormal(seed=43),
        kernel_regularizer=regularizers.l2(0.0001),  # Match training l2
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


DEFAULT_CLASSIFIER_THRESHOLD = 0.25
REGIME_WINDOW = 60
REGIME_BULL_THRESHOLD = 0.10
REGIME_BEAR_THRESHOLD = -0.12
BULL_SELL_MARGIN = 0.0
BEAR_BUY_MARGIN = 0.02
MIN_FORCED_SIGNALS = 5
TARGET_SELL_RATIO = 0.7
MIN_SELL_SIGNALS = 20
HYBRID_SCALE_FACTOR = 4.0
HYBRID_BASE_FRACTION = 0.15
HYBRID_MAX_POSITION = 0.40  # Base max position, adjusted down by volatility regime
HYBRID_MIN_SHARES_THRESHOLD = 1.0
HYBRID_REFERENCE_EQUITY = 10_000.0
HYBRID_MIN_CONFIDENCE = 0.03

# Volatility-based position sizing configuration
VOL_BASELINE_LOOKBACK = 252  # 1 year for baseline volatility calculation
VOL_HIGH_REGIME_THRESHOLD = 1.5  # vol_ratio_5_20 > 1.5 = very high volatility
VOL_LOW_REGIME_THRESHOLD = 0.7   # vol_ratio_5_20 < 0.7 = very low volatility
VOL_HIGH_MAX_POSITION = 0.25     # Reduce to 25% max in high volatility
VOL_LOW_MAX_POSITION = 0.50      # Allow up to 50% max in low volatility
VOL_ADJUST_MIN = 0.5             # Minimum volatility adjustment (high vol)
VOL_ADJUST_MAX = 2.0             # Maximum volatility adjustment (low vol)

# Volatility-based position sizing configuration
ATR_PERIOD = 14  # 14-day Average True Range for volatility measure
BASELINE_VOLATILITY = 0.02  # 2% baseline ATR (typical for moderate volatility)
VOLATILITY_SCALE_MIN = 0.5  # Minimum scale factor during high volatility
VOLATILITY_SCALE_MAX = 2.0  # Maximum scale factor during low volatility

# Adaptive ATR-based profit-taking configuration
ATR_TARGET_MULTIPLIERS = [2.0, 3.5, 5.0, 8.0]  # ATR multipliers for dynamic targets
ATR_SCALE_OUT_PERCENTAGES = [0.30, 0.30, 0.20, 0.20]  # Exit 30%, 30%, 20%, 20% at each target
INITIAL_STOP_ATR_MULTIPLIER = 2.0  # Initial stop: 2x ATR below entry
BREAKEVEN_STOP_LEVEL = 0  # Move to breakeven after target 1
TARGET1_STOP_OFFSET = 0  # Lock in profit at target 1 price after target 2
TRAILING_STOP_ATR_TIGHT = 1.5  # Trail with 1.5x ATR after target 3
TRAILING_STOP_ATR_FINAL = 1.0  # Trail with 1.0x ATR after target 4 (runner)

# Time-based exit degradation
MAX_POSITION_DAYS_STAGE1 = 10  # Days before reducing targets by 20%
MAX_POSITION_DAYS_STAGE2 = 20  # Days before forced exit on next signal
TIME_DEGRADATION_FACTOR = 0.80  # Reduce targets to 80% after stage 1

# Support/resistance aware exit configuration
RESISTANCE_PROXIMITY_THRESHOLD = 0.01  # Within 1% of resistance
EARLY_EXIT_DISCOUNT = 0.90  # Take 90% of target when near resistance
VOLUME_SURGE_THRESHOLD = 1.5  # 50% above average volume = breakout signal

# Legacy fixed targets (kept for backward compatibility, but not used)
PROFIT_TARGETS = [0.05, 0.10, 0.15, 0.20]  # +5%, +10%, +15%, +20% profit targets
SCALE_OUT_PERCENTAGES = [0.20, 0.25, 0.25, 0.10]  # Reduce by 20%, 25%, 25%, 10% at each target
FINAL_POSITION_RESERVE = 0.20  # Keep 20% for final exit
INITIAL_STOP_LOSS = 0.08  # 8% initial stop loss
TRAILING_STOP_LEVELS = [
    (0.05, 0.05),   # At +5% profit, tighten stop to 5%
    (0.10, 0.03),   # At +10% profit, tighten stop to 3%
    (0.15, 0.015),  # At +15% profit, tighten stop to 1.5%
    (0.20, 0.0025), # At +20% profit, tighten stop to 0.25%
]

# Profit-target order system configuration (legacy - replaced by ATR-based)
PROFIT_TARGET_LEVELS = [0.03, 0.07, 0.12, 0.18]  # +3%, +7%, +12%, +18% from entry
PROFIT_TARGET_SIZES = [0.25, 0.25, 0.25, 0.25]   # Sell 25% at each level

# Tolerance band rebalancing configuration
TARGET_POSITION_WEIGHT = 0.25  # 25% moderate baseline allocation
REBALANCE_TOLERANCE = 0.20     # ±20% deviation triggers rebalancing
UPPER_REBALANCE_BAND = TARGET_POSITION_WEIGHT * (1 + REBALANCE_TOLERANCE)  # 30%
LOWER_REBALANCE_BAND = TARGET_POSITION_WEIGHT * (1 - REBALANCE_TOLERANCE)  # 20%

# Drawdown-aware position sizing configuration
DRAWDOWN_THRESHOLD_1 = 0.05  # 5% drawdown - first level of risk reduction
DRAWDOWN_THRESHOLD_2 = 0.10  # 10% drawdown - second level of risk reduction
DRAWDOWN_THRESHOLD_3 = 0.15  # 15% drawdown - maximum risk reduction
DRAWDOWN_RECOVERY_THRESHOLD = 0.03  # Within 3% of high-water mark to restore limits
DRAWDOWN_MAX_POSITION_1 = 0.50  # 50% max position at 5% drawdown
DRAWDOWN_MAX_POSITION_2 = 0.30  # 30% max position at 10% drawdown
DRAWDOWN_MAX_POSITION_3 = 0.15  # 15% max position at 15% drawdown
DRAWDOWN_SCALE_FACTOR_2 = 2.0   # Reduced scale factor at 10% drawdown
DRAWDOWN_SCALE_FACTOR_3 = 1.0   # Minimum scale factor at 15% drawdown

# Kelly Criterion position sizing configuration
KELLY_LOOKBACK = 50  # Number of recent trades to analyze for win/loss ratio
KELLY_MAX_FRACTION = 0.25  # Quarter-Kelly (25% max) to avoid over-leveraging
KELLY_MIN_FRACTION = 0.05  # Minimum Kelly fraction (5%)
DEFAULT_WIN_LOSS_RATIO = 1.5  # Default win/loss ratio if insufficient history

# Regime-aware position sizing configuration
REGIME_SMA_PERIOD = 50  # 50-day SMA for regime detection
REGIME_SLOPE_WINDOW = 5  # Window for calculating SMA slope
BULLISH_REGIME_BUY_BOOST = 1.20  # +20% to BUY positions in bullish regime
BULLISH_REGIME_SELL_REDUCTION = 0.60  # -40% to SELL positions in bullish regime
BEARISH_REGIME_SELL_BOOST = 1.20  # +20% to SELL positions in bearish regime
BEARISH_REGIME_BUY_REDUCTION = 0.60  # -40% to BUY positions in bearish regime

# Position limits and leverage constraints
MAX_SINGLE_POSITION = 0.40  # 40% of equity max per position
MAX_TOTAL_LEVERAGE = 1.5  # 150% total exposure limit
CORRELATION_THRESHOLD_HIGH = 0.7  # High correlation threshold
CORRELATION_THRESHOLD_LOW = 0.3  # Low correlation threshold
CORRELATION_LOOKBACK = 20  # Days for correlation calculation
HIGH_CORRELATION_REDUCTION = 0.70  # Reduce combined position by 30% if correlated

# -------------------------
# Helpers
# -------------------------
# Model availability flags (default values). These will be updated at runtime
# based on which artifacts could be loaded successfully.
regressor_available = False
classifiers_available = False
quantile_available = False
tft_available = False


def select_best_available_fusion_mode(requested_mode: str, available_models: dict):
    """Choose the best fusion mode given requested mode and available models.

    Rules:
    - 'stacking' is the RECOMMENDED mode and requires trained stacking model
    - If stacking requested but not available, ERROR (don't silently fallback)
    - GBM fusion modes (gbm_only, gbm_heavy, balanced, lstm_heavy) only need regressor+GBM.
    - gbm_only mode ONLY requires GBM model, not the LSTM regressor.
    - If no models are available, return None to indicate no viable fusion options.

    Note: Binary classifiers have been deprecated and removed. 'classifier', 'hybrid',
    and 'weighted' modes are no longer supported.
    """
    required = {
        'stacking': ['stacking'],        # Stacking requires trained meta-learner
        'regressor': ['regressor'],
        'regressor_only': ['regressor'],
        # GBM fusion modes: need regressor (GBM is optional, uses LSTM-only if GBM missing)
        'gbm_only': ['gbm'],             # GBM-only ONLY requires GBM model
        'gbm_heavy': ['regressor'],      # 70% GBM, 30% LSTM (needs LSTM as base)
        'balanced': ['regressor'],       # 50% GBM, 50% LSTM (needs LSTM as base)
        'lstm_heavy': ['regressor'],     # 30% GBM, 70% LSTM (needs LSTM as base)
    }

    req = required.get(requested_mode, [])
    # If requested mode requires nothing or all required models are present, return it
    if all(available_models.get(r, False) for r in req):
        return requested_mode

    # Stacking mode: ERROR if not available (don't silently fallback)
    if requested_mode == 'stacking':
        raise RuntimeError(
            "Stacking ensemble not found. You must train the stacking model first:\n"
            "  python train_all.py --symbol <SYMBOL>\n"
            "Or use a fallback mode: --fusion-mode gbm_only"
        )

    # Fallback preferences for non-stacking modes
    if available_models.get('gbm', False):
        return 'gbm_only'  # Fallback to GBM-only if GBM is available
    if available_models.get('regressor', False):
        return 'regressor'

    # No usable models available
    return None
def calculate_kelly_fraction(win_prob, win_loss_ratio):
    """
    Calculate Kelly Criterion fraction for position sizing.
    
    Kelly formula: f* = p - (1-p)/b
    Where:
    - p = probability of winning
    - b = win/loss ratio (avg_win / avg_loss)
    - f* = optimal fraction of capital to bet
    
    We use quarter-Kelly (25% max) for safety to avoid over-leveraging.
    
    Args:
        win_prob: Probability of winning trade (from classifier)
        win_loss_ratio: Average win / average loss ratio from recent trades
    
    Returns:
        Kelly fraction clipped to [KELLY_MIN_FRACTION, KELLY_MAX_FRACTION]
    
    Example:
        >>> # 60% win prob, 2:1 win/loss ratio
        >>> kelly = calculate_kelly_fraction(0.60, 2.0)
        >>> # kelly = 0.60 - (0.40 / 2.0) = 0.60 - 0.20 = 0.40
        >>> # Clipped to 0.25 (quarter-Kelly max)
    """
    if win_loss_ratio <= 0:
        return KELLY_MIN_FRACTION
    
    # Kelly formula
    kelly_fraction = win_prob - ((1 - win_prob) / win_loss_ratio)
    
    # Clip to safe range (quarter-Kelly max)
    kelly_fraction = np.clip(kelly_fraction, KELLY_MIN_FRACTION, KELLY_MAX_FRACTION)
    
    return float(kelly_fraction)


def calculate_win_loss_ratio_from_history(trade_history, lookback=KELLY_LOOKBACK):
    """
    Calculate win/loss ratio from recent trade history.
    
    Args:
        trade_history: List of dicts with 'pnl_pct' key
        lookback: Number of recent trades to analyze
    
    Returns:
        Win/loss ratio (avg_win / avg_loss), or DEFAULT_WIN_LOSS_RATIO if insufficient data
    """
    if not trade_history or len(trade_history) < 10:
        return DEFAULT_WIN_LOSS_RATIO
    
    # Get recent trades
    recent_trades = trade_history[-lookback:] if len(trade_history) > lookback else trade_history
    
    # Separate wins and losses
    wins = [t['pnl_pct'] for t in recent_trades if t.get('pnl_pct', 0) > 0]
    losses = [abs(t['pnl_pct']) for t in recent_trades if t.get('pnl_pct', 0) < 0]
    
    if not wins or not losses:
        return DEFAULT_WIN_LOSS_RATIO
    
    avg_win = np.mean(wins)
    avg_loss = np.mean(losses)
    
    if avg_loss == 0:
        return DEFAULT_WIN_LOSS_RATIO
    
    win_loss_ratio = avg_win / avg_loss
    
    # Reasonable bounds
    return float(np.clip(win_loss_ratio, 0.5, 5.0))


def detect_market_regime(prices, sma_period=REGIME_SMA_PERIOD, slope_window=REGIME_SLOPE_WINDOW):
    """
    Detect market regime (bullish/bearish/sideways) using SMA and slope.
    
    Regime classification:
    - Bullish: Close > SMA_50 AND SMA_50 slope > 0
    - Bearish: Close < SMA_50 AND SMA_50 slope < 0
    - Sideways: Neither condition met
    
    Args:
        prices: Array of closing prices
        sma_period: Period for SMA calculation (default 50)
        slope_window: Window for calculating SMA slope (default 5)
    
    Returns:
        Array of regime labels: 1=bullish, -1=bearish, 0=sideways
    """
    if len(prices) < sma_period + slope_window:
        # Insufficient data, assume sideways
        return np.zeros(len(prices), dtype=int)
    
    # Calculate SMA
    sma = np.full(len(prices), np.nan)
    for i in range(sma_period - 1, len(prices)):
        sma[i] = np.mean(prices[max(0, i - sma_period + 1):i + 1])
    
    # Calculate SMA slope
    sma_slope = np.zeros(len(prices))
    for i in range(sma_period + slope_window - 1, len(prices)):
        if not np.isnan(sma[i]) and not np.isnan(sma[i - slope_window]):
            sma_slope[i] = (sma[i] - sma[i - slope_window]) / slope_window
    
    # Classify regime
    regime = np.zeros(len(prices), dtype=int)
    
    for i in range(sma_period, len(prices)):
        if np.isnan(sma[i]):
            regime[i] = 0
            continue
        
        # Bullish: Close > SMA AND SMA trending up
        if prices[i] > sma[i] and sma_slope[i] > 0:
            regime[i] = 1  # Bullish
        # Bearish: Close < SMA AND SMA trending down
        elif prices[i] < sma[i] and sma_slope[i] < 0:
            regime[i] = -1  # Bearish
        else:
            regime[i] = 0  # Sideways
    
    return regime


def apply_regime_adjustment(position_fraction, regime, signal):
    """
    Apply regime-aware position sizing adjustment.
    
    Args:
        position_fraction: Base position fraction
        regime: Market regime (1=bullish, -1=bearish, 0=sideways)
        signal: Trading signal (1=BUY, -1=SELL, 0=HOLD)
    
    Returns:
        Adjusted position fraction
    """
    if regime == 1:  # Bullish regime
        if signal > 0:  # BUY signal
            return position_fraction * BULLISH_REGIME_BUY_BOOST
        elif signal < 0:  # SELL signal
            return position_fraction * BULLISH_REGIME_SELL_REDUCTION
    
    elif regime == -1:  # Bearish regime
        if signal < 0:  # SELL signal
            return position_fraction * BEARISH_REGIME_SELL_BOOST
        elif signal > 0:  # BUY signal
            return position_fraction * BEARISH_REGIME_BUY_REDUCTION
    
    # Sideways or HOLD - no adjustment
    return position_fraction


def create_sequences(X, seq_len=60):
    return np.array([X[i:i+seq_len] for i in range(X.shape[0] - seq_len + 1)])

def safe_load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def load_model_with_metadata(symbol: str, model_type: str = 'regressor', save_dir: Path | None = None):
    """Load model + scaler + metadata for backward compatibility.

    - First tries new organized path structure: saved_models/{symbol}/regressor/
    - Falls back to legacy flat structure: saved_models/{symbol}_1d_regressor_final_*
    - Attempts to read metadata for scaler configuration
    - Loads model using `keras.models.load_model`

    Returns: (model, target_scaler, metadata)
    """
    logger = logging.getLogger(__name__)
    if save_dir is None:
        save_dir = Path('saved_models')

    # Try new organized path structure first if ModelPaths available
    if ModelPaths is not None:
        paths = ModelPaths(symbol)
        
        # Try to load from new organized structure
        if paths.regressor.base.exists():
            try:
                # Metadata
                metadata = None
                if paths.regressor.metadata.exists():
                    metadata = safe_load_pickle(paths.regressor.metadata)
                elif (paths.regressor.base / 'target_metadata.pkl').exists():
                    metadata = safe_load_pickle(paths.regressor.base / 'target_metadata.pkl')
                
                if metadata is None:
                    metadata = {'scaling_method': 'minmax'}
                
                # Target scaler
                target_scaler = None
                if paths.regressor.target_scaler.exists():
                    target_scaler = safe_load_pickle(paths.regressor.target_scaler)
                
                # Model - try multiple loading approaches
                model = None
                if paths.regressor.saved_model.exists():
                    try:
                        # First try standard Keras loading
                        custom_objects = {
                            'LSTMTransformerPaper': LSTMTransformerPaper,
                            'AntiCollapseDirectionalLoss': AntiCollapseDirectionalLoss,
                            'DirectionalHuberLoss': DirectionalHuberLoss,
                            'AsymmetricDirectionalLoss': AsymmetricDirectionalLoss,
                            'directional_accuracy_metric': directional_accuracy_metric,
                            'create_directional_accuracy_metric': create_directional_accuracy_metric,
                            'PatchTST': PatchTST,
                            'RevIN': RevIN,
                            'PatchEmbedding': PatchEmbedding,
                            'LearnablePositionalEncoding': LearnablePositionalEncoding,
                            'TransformerEncoderLayer': TransformerEncoderLayer
                        }
                        model = keras.models.load_model(str(paths.regressor.saved_model), custom_objects=custom_objects)
                    except Exception as e:
                        print(f"DEBUG: Keras load failed: {e}")
                        # Fall back to TF SavedModel loading (for model.export() format)
                        try:
                            import tensorflow as tf
                            model = tf.saved_model.load(str(paths.regressor.saved_model))
                            logger.info(f"Loaded {symbol} regressor as TF SavedModel (non-Keras format)")
                        except Exception as e2:
                            logger.warning(f"Failed to load model as TF SavedModel: {e2}")
                
                if target_scaler is not None and model is not None:
                    logger.info(f"Loaded {symbol} regressor from new organized path structure")
                    return model, target_scaler, metadata
            except Exception as e:
                logger.warning(f"Failed to load from new structure: {e}, trying legacy or smart paths")

    # Smart model paths (Phase 7)
    for model_type_name in ['patchtst', 'lstm_transformer']:
        smart_dir = Path(f"models/smart_{symbol}_{model_type_name}")
        if (smart_dir / "model.keras").exists():
            try:
                custom_objects = {
                    'LSTMTransformerPaper': LSTMTransformerPaper,
                    'AntiCollapseDirectionalLoss': AntiCollapseDirectionalLoss,
                    'DirectionalHuberLoss': DirectionalHuberLoss,
                    'AsymmetricDirectionalLoss': AsymmetricDirectionalLoss,
                    'directional_accuracy_metric': directional_accuracy_metric,
                    'create_directional_accuracy_metric': create_directional_accuracy_metric,
                    'PatchTST': PatchTST,
                    'RevIN': RevIN,
                    'PatchEmbedding': PatchEmbedding,
                    'LearnablePositionalEncoding': LearnablePositionalEncoding,
                    'TransformerEncoderLayer': TransformerEncoderLayer
                }
                model = keras.models.load_model(str(smart_dir / "model.keras"), custom_objects=custom_objects)
                
                # Load metadata
                metadata = {}
                if (smart_dir / "metadata.json").exists():
                    with open(smart_dir / "metadata.json", 'r') as f:
                        metadata = json.load(f)
                
                # Smart target models are always robust/scaled
                # The train_smart_regressor.py uses scaling internally
                # For inference, we need to know how to inverse scale
                # Currently metadata.json doesn't save the scaler pickle
                # I should update train_smart_regressor.py to save the scaler
                
                # Smart target models use a dedicated scaler
                target_scaler = None
                if (smart_dir / "target_scaler.pkl").exists():
                    target_scaler = safe_load_pickle(smart_dir / "target_scaler.pkl")
                
                logger.info(f"Loaded {symbol} {model_type_name} from smart model directory")
                return model, target_scaler, metadata
            except Exception as e:
                logger.warning(f"Failed to load from smart dir {smart_dir}: {e}")

    # Fall back to legacy flat path structure
    # Try multiple metadata filenames (backwards compatibility)
    metadata_candidates = [
        save_dir / f"{symbol}_1d_regressor_final_metadata.pkl",
        save_dir / f"{symbol}_target_metadata.pkl",
    ]
    metadata = None
    scaling_method = None
    for p in metadata_candidates:
        if p.exists():
            try:
                with open(p, 'rb') as f:
                    metadata = pickle.load(f)
                    scaling_method = metadata.get('scaling_method')
                    break
            except Exception:
                continue

    if metadata is None:
        logger.warning(f"No metadata found for {symbol}, assuming legacy MinMax/legacy naming")
        metadata = {'scaling_method': 'minmax'}
        scaling_method = 'minmax'

    # Prefer robust scaler if metadata indicates it, try several fallback names
    scaler_candidates = []
    if scaling_method == 'robust':
        scaler_candidates += [
            save_dir / f"{symbol}_1d_target_scaler_robust.pkl",
            save_dir / f"{symbol}_target_scaler_robust.pkl",
        ]
    scaler_candidates += [
        save_dir / f"{symbol}_1d_regressor_final_target_scaler.pkl",
        save_dir / f"{symbol}_1d_target_scaler_robust.pkl",
        save_dir / f"{symbol}_1d_target_scaler.pkl",
        save_dir / f"{symbol}_target_scaler.pkl",
        save_dir / f"{symbol}_1d_regressor_final_target_scaler.pkl",
    ]

    target_scaler = None
    for s in scaler_candidates:
        if s.exists():
            try:
                target_scaler = safe_load_pickle(s)
                break
            except Exception:
                continue

    if target_scaler is None:
        raise FileNotFoundError(f"Target scaler not found. Tried: {scaler_candidates}")

    # Model directory fallback names (training historically exported under several conventions)
    model_candidates = [
        save_dir / f"{symbol}_1d_regressor_final_model",
        save_dir / f"{symbol}_{model_type}_final_model",
    ]

    model = None
    for m in model_candidates:
        if m.exists():
            try:
                model = keras.models.load_model(str(m))
                break
            except Exception:
                # try next candidate
                continue

    if model is None:
        raise FileNotFoundError(f"Could not locate or load model directory. Tried: {model_candidates}")

    return model, target_scaler, metadata


def run_backtest(symbol: str, backtest_days: int = 60, fusion_mode: str = 'weighted', use_cache: bool = True):
    """Enhanced backtest entry that enforces v3.1 feature usage.

    This helper engineers features with sentiment, validates the feature count
    equals the v3.1 expected count (123) and loads the regressor model and
    scaler metadata for downstream backtest logic.
    """
    logger = logging.getLogger(__name__)

    # Load data (use cache manager when requested)
    if use_cache:
        cache_manager = DataCacheManager()
        raw_df, engineered_df, prepared_df, feature_cols = cache_manager.get_or_fetch_data(
            symbol=symbol,
            include_sentiment=True,
            force_refresh=False
        )
        df = engineered_df
        # prefer prepared_df if caller expects prepared rows
        # but return the engineered df to preserve original behavior
    else:
        # When not using cache, force a fetch-and-cache so the result is saved for future runs.
        from data.cache_manager import DataCacheManager
        cm = DataCacheManager()
        raw_df, engineered_df, prepared_df, feature_cols = cm.get_or_fetch_data(symbol=symbol, include_sentiment=True, force_refresh=True)
        df = engineered_df

    # Validate feature count using canonical feature list (exclude OHLCV columns)
    try:
        expected_cols = get_feature_columns(include_sentiment=True)
        expected_count = len(expected_cols)
    except Exception:
        expected_count = EXPECTED_FEATURE_COUNT

    exclude_cols = {'Open', 'High', 'Low', 'Close', 'Volume', 'Date'}
    actual_feature_cols = [c for c in df.columns if c not in exclude_cols]
    actual_count = len(actual_feature_cols)

    if actual_count != expected_count:
        logger.error(f"Feature mismatch for {symbol}: expected {expected_count}, got {actual_count}")
        logger.error("Please retrain models or re-run feature engineering with include_sentiment=True")
        raise ValueError("Feature count mismatch - retrain required")

    # Load regressor and its target scaler + metadata
    regressor, target_scaler, metadata = load_model_with_metadata(symbol, 'regressor')

    # Return prepared objects for the caller to run the rest of the backtest
    return {
        'symbol': symbol,
        'df': df,
        'regressor': regressor,
        'target_scaler': target_scaler,
        'metadata': metadata,
        'backtest_days': backtest_days,
        'fusion_mode': fusion_mode
    }


def compute_positions_from_tft(forecast: dict, reg_threshold: float = 0.01, reg_scale: float = 15.0, max_short: float = 0.5):
    """Convert a TFT forecast dict into regressor-style positions.

    This helper uses the median (q50) forecast for horizon 1 as a point
    return prediction and feeds it into `compute_regressor_positions` to
    produce a position fraction.
    """
    # Safe access to horizon 1 median prediction
    horizons = forecast.get('horizons', [])
    if not horizons:
        raise ValueError('Forecast contains no horizons')
    h1 = horizons[0]
    preds = forecast.get('predictions', {})
    if h1 not in preds:
        # fallback to first available
        try:
            h1 = list(preds.keys())[0]
        except Exception:
            raise ValueError('No predictions present in forecast')

    q50 = float(preds[h1].get('q50', 0.0))

    # Build a 1-element returns array and compute a single-day position
    returns = np.array([q50], dtype=float)
    positions = compute_regressor_positions(returns, strategy='confidence_scaling', threshold=reg_threshold, scale=reg_scale, max_short=max_short)
    return returns, positions


def run_backtest_with_tft(symbol: str, use_tft: bool = False, model_path: str | None = None, horizon: int = 5,
                          reg_threshold: float = 0.01, reg_scale: float = 15.0, max_short: float = 0.5):
    """Optional backtest runner that can use a TFT forecast as input.

    Behavior:
    - If `use_tft` is True, attempts to load a TFT model (from `model_path` or
      `saved_models/{symbol}_tft.ckpt`) and generate a multi-horizon forecast.
      The horizon-1 median (`q50`) is used as a point forecast to compute a
      short positions vector which is then backtested with `AdvancedBacktester`.
    - If `use_tft` is False, this function returns None to indicate caller
      should use standard ensemble/backtest flow.

    Note: This is a lightweight integration for exploratory use. For full
    integration you may prefer to wire TFT-derived positions into the
    main backtest flow where sequence alignment and historical windows are
    already established.
    """
    logger = logging.getLogger(__name__)

    if not use_tft:
        logger.info('TFT integration disabled; use standard ensemble backtest')
        return None

    # Import TFT helpers lazily (may not be installed in all environments)
    try:
        from inference.predict_tft import load_tft_model, prepare_inference_data, generate_forecast, generate_trading_signal
    except Exception as e:
        logger.error(f'Failed to import TFT inference utilities: {e}')
        raise

    # Resolve model path with fallback to multiple patterns
    if model_path is None:
        # Try new organized path structure first if ModelPaths available
        if ModelPaths is not None:
            paths = ModelPaths(symbol)
            if paths.tft.checkpoint.exists():
                model_path = str(paths.tft.checkpoint)
                logger.info(f'Found TFT checkpoint at (new structure): {model_path}')
        
        # Fall back to legacy paths if not found
        if model_path is None:
            candidates = [
                Path('saved_models') / 'tft' / symbol / 'best_model.ckpt',  # Legacy TFT format
                Path('saved_models') / 'tft' / symbol.upper() / 'best_model.ckpt',  # Uppercase variant
                Path('saved_models') / f'{symbol}_tft.ckpt',  # Very old format
                Path('saved_models') / f'{symbol.upper()}_tft.ckpt',  # Very old uppercase
            ]
            
            for candidate in candidates:
                if candidate.exists():
                    model_path = str(candidate)
                    logger.info(f'Found TFT checkpoint at (legacy): {model_path}')
                    break
            else:
                # None found - provide helpful error with all tried paths
                all_tried = []
                if ModelPaths is not None:
                    all_tried.append(str(ModelPaths(symbol).tft.checkpoint))
                all_tried.extend([str(c) for c in candidates])
                tried_paths = '\n  - '.join(all_tried)
                raise FileNotFoundError(
                    f'TFT model not found for {symbol}. Tried:\n  - {tried_paths}\n'
                    f'Please train the model first using: python training/train_tft.py --symbols {symbol}'
                )

    # Load model and dataset params
    model, ds_params = load_tft_model(model_path)

    # Prepare inference dataframe (cache-only for backtest to avoid refetch)
    lookback = int(ds_params.get('max_encoder_length', 60)) if isinstance(ds_params, dict) else 60
    try:
        df_prep = prepare_inference_data(
            symbol,
            lookback_days=lookback,
            metadata=None,
            use_cache=True,
            use_cache_only=True,
            include_sentiment=True,
        )
    except FileNotFoundError as e:
        print(f"✗ TFT backtest requires cached data: {e}")
        return None

    # Generate forecast and trading signal
    horizons = [1, 3, 5] if horizon is None else [1, horizon]
    # generate_forecast expects a singular `horizon` parameter (max horizon to produce)
    max_h = max(horizons)
    forecast = generate_forecast(model, df_prep, horizon=max_h)
    signal, position_size = generate_trading_signal(forecast)

    # Convert forecast into returns & positions (short lightweight scenario)
    returns, positions = compute_positions_from_tft(forecast, reg_threshold=reg_threshold, reg_scale=reg_scale, max_short=max_short)

    # Build minimal price/date arrays for backtester using most recent price
    # Note: prepare_inference_data preserves Close/close column when available
    price_col = 'Close' if 'Close' in df_prep.columns else ('close' if 'close' in df_prep.columns else None)
    if price_col is None:
        raise RuntimeError('No price column available in TFT prepared data for backtesting')
    last_price = float(df_prep[price_col].iloc[-1])
    # Create a synthetic next-day price path from predicted returns
    simulated_prices = simulate_price_path(last_price, returns)
    # Dates: use business days starting after last index if index is datetime-like
    try:
        last_date = pd.to_datetime(df_prep.index[-1])
        dates = pd.bdate_range(last_date + BDay(1), periods=len(simulated_prices)).to_pydatetime()
    except Exception:
        dates = np.arange(len(simulated_prices))

    # Run lightweight backtest
    # Standalone regressor backtest
    backtester = AdvancedBacktester(
        initial_capital=10_000,
        margin_requirement=margin_requirement,
        borrow_rate=borrow_rate
    )
    adv_results = backtester.backtest_with_positions(
        dates=dates,
        prices=simulated_prices,
        returns=returns,
        positions=positions,
        max_long=1.0,
        max_short=max_short
    )

    # Attach forecast metadata to results for inspection
    adv_results_forecast = {
        'backtest': adv_results,
        'forecast': forecast,
        'signal': signal,
        'position_size': position_size
    }
    return adv_results_forecast


def load_optimal_thresholds(threshold_file_path):
    """Load optimal thresholds JSON file and validate structure.

    Expects JSON with keys 'buy' and 'sell' mapping to float values.
    Returns tuple (buy_threshold, sell_threshold).
    Raises FileNotFoundError if file is missing; caller may fallback to defaults.
    """
    defaults = (0.30, 0.45)
    if not threshold_file_path:
        return defaults
    p = Path(threshold_file_path)
    if not p.exists():
        raise FileNotFoundError(f"Threshold file not found: {p}")
    try:
        with open(p, 'r') as f:
            data = json.load(f)
    except Exception as e:
        raise ValueError(f"Failed to parse threshold JSON {p}: {e}")
    if not isinstance(data, dict):
        raise ValueError(f"Threshold file {p} must contain a JSON object with 'buy' and 'sell' keys")
    if 'buy' not in data or 'sell' not in data:
        raise ValueError(f"Threshold file {p} missing required keys 'buy' and/or 'sell'")
    try:
        buy = float(data['buy'])
        sell = float(data['sell'])
    except Exception:
        raise ValueError(f"Threshold values must be numeric in {p}")
    return buy, sell

def calculate_atr(prices, period=ATR_PERIOD):
    """Calculate Average True Range for volatility measure.
    
    ATR measures market volatility by decomposing the entire range of price movement.
    Used to dynamically adjust position sizing based on current market conditions.
    
    Args:
        prices: Array of closing prices
        period: Lookback period for ATR calculation (default 14 days)
    
    Returns:
        Array of ATR values (same length as prices, padded with baseline for initial period)
    """
    if len(prices) < 2:
        return np.full_like(prices, BASELINE_VOLATILITY)
    
    # Calculate true range components
    # TR = max(high-low, abs(high-prev_close), abs(low-prev_close))
    # Since we only have closing prices, approximate with price changes
    price_changes = np.abs(np.diff(prices))
    pct_changes = price_changes / prices[:-1]
    
    # Calculate rolling ATR as percentage of price
    atr_values = np.zeros(len(prices))
    atr_values[:period] = BASELINE_VOLATILITY  # Use baseline for initial period
    
    for i in range(period, len(prices)):
        atr_values[i] = np.mean(pct_changes[max(0, i-period):i])
    
    return atr_values

def calculate_volatility_adjusted_scale(current_atr):
    """Calculate volatility-adjusted scale factor for position sizing.
    
    Formula: adjusted_scale = HYBRID_SCALE_FACTOR * (baseline_volatility / current_ATR)
    - High volatility (high ATR) → lower scale factor → smaller positions
    - Low volatility (low ATR) → higher scale factor → larger positions
    
    Args:
        current_atr: Current ATR value
    
    Returns:
        Adjusted scale factor clamped to [VOLATILITY_SCALE_MIN * base, VOLATILITY_SCALE_MAX * base]
    """
    if current_atr <= 0:
        return HYBRID_SCALE_FACTOR
    
    adjusted_scale = HYBRID_SCALE_FACTOR * (BASELINE_VOLATILITY / current_atr)
    
    # Clamp to reasonable range
    min_scale = HYBRID_SCALE_FACTOR * VOLATILITY_SCALE_MIN
    max_scale = HYBRID_SCALE_FACTOR * VOLATILITY_SCALE_MAX
    
    return float(np.clip(adjusted_scale, min_scale, max_scale))

def calculate_drawdown_adjustments(current_equity, high_water_mark):
    """Calculate drawdown-aware position sizing adjustments.
    
    Reduces maximum position size and scale factor during drawdown periods to protect capital.
    Gradually restores limits as equity recovers to within 3% of peak.
    
    Drawdown tiers:
    - > 5%: max_position = 50%
    - > 10%: max_position = 30%, scale_factor = 2.0
    - > 15%: max_position = 15%, scale_factor = 1.0
    - < 3% from peak: restore to baseline (40%, 4.0)
    
    Args:
        current_equity: Current portfolio equity value
        high_water_mark: Peak equity value (highest historical equity)
    
    Returns:
        Dict with 'max_position', 'scale_factor', 'drawdown_pct', 'multiplier'
    """
    if high_water_mark <= 0:
        return {
            'max_position': HYBRID_MAX_POSITION,
            'scale_factor': HYBRID_SCALE_FACTOR,
            'drawdown_pct': 0.0,
            'multiplier': 1.0
        }
    
    # Calculate drawdown percentage
    drawdown_pct = (high_water_mark - current_equity) / high_water_mark
    drawdown_pct = max(0.0, drawdown_pct)  # Clamp to non-negative
    
    # Determine adjustments based on drawdown severity
    if drawdown_pct >= DRAWDOWN_THRESHOLD_3:  # > 15% drawdown
        max_position = DRAWDOWN_MAX_POSITION_3
        scale_factor = DRAWDOWN_SCALE_FACTOR_3
        multiplier = 0.25  # 75% position size reduction
    elif drawdown_pct >= DRAWDOWN_THRESHOLD_2:  # > 10% drawdown
        max_position = DRAWDOWN_MAX_POSITION_2
        scale_factor = DRAWDOWN_SCALE_FACTOR_2
        multiplier = 0.50  # 50% position size reduction
    elif drawdown_pct >= DRAWDOWN_THRESHOLD_1:  # > 5% drawdown
        max_position = DRAWDOWN_MAX_POSITION_1
        scale_factor = HYBRID_SCALE_FACTOR  # Keep full scale factor
        multiplier = 0.75  # 25% position size reduction
    elif drawdown_pct <= DRAWDOWN_RECOVERY_THRESHOLD:  # Within 3% of peak
        # Restore to baseline
        max_position = HYBRID_MAX_POSITION
        scale_factor = HYBRID_SCALE_FACTOR
        multiplier = 1.0  # Full position sizing
    else:  # Between 3% and 5% - gradual recovery
        # Linear interpolation between recovery and first threshold
        recovery_progress = (DRAWDOWN_THRESHOLD_1 - drawdown_pct) / (DRAWDOWN_THRESHOLD_1 - DRAWDOWN_RECOVERY_THRESHOLD)
        max_position = DRAWDOWN_MAX_POSITION_1 + (HYBRID_MAX_POSITION - DRAWDOWN_MAX_POSITION_1) * recovery_progress
        scale_factor = HYBRID_SCALE_FACTOR
        multiplier = 0.75 + 0.25 * recovery_progress  # Gradually restore to 1.0
    
    return {
        'max_position': float(max_position),
        'scale_factor': float(scale_factor),
        'drawdown_pct': float(drawdown_pct),
        'multiplier': float(multiplier)
    }


def summarize_probabilities(name, probs):
    if probs.size == 0:
        return
    quantiles = np.percentile(probs, [0, 25, 50, 75, 100])
    print(
        f"{name} probs -> min {quantiles[0]:.2f}, "
        f"25% {quantiles[1]:.2f}, median {quantiles[2]:.2f}, "
        f"75% {quantiles[3]:.2f}, max {quantiles[4]:.2f}, "
        f"mean {np.mean(probs):.2f}"
    )


def restore_signals(pred_mask, probs, label, opposing_mask=None, min_signals=1):
    """Ensure at least `min_signals` survive by re-enabling top-confidence indices."""
    if min_signals <= 0 or probs.size == 0:
        return
    sorted_idx = np.argsort(probs)[::-1]
    restored = 0
    # First pass: prefer indices where opposing signal is already off
    for idx in sorted_idx:
        if pred_mask[idx] == 1:
            continue
        if opposing_mask is not None and opposing_mask[idx] == 1:
            continue
        pred_mask[idx] = 1
        restored += 1
        if restored >= min_signals:
            break
    # Second pass: if still short, allow overriding opposing signals
    if restored < min_signals:
        for idx in sorted_idx:
            if pred_mask[idx] == 1:
                continue
            pred_mask[idx] = 1
            if opposing_mask is not None:
                opposing_mask[idx] = 0
            restored += 1
            if restored >= min_signals:
                break
    if restored > 0:
        print(f"  ⚠️  Restored {restored} {label} signals to avoid empty set")


def detect_regime(prices, window=REGIME_WINDOW, bull_thresh=REGIME_BULL_THRESHOLD,
                  bear_thresh=REGIME_BEAR_THRESHOLD):
    """Simple regime detection based on trailing window returns."""
    prices = np.asarray(prices)
    regimes = []
    for i in range(len(prices)):
        if i < window:
            regimes.append('RANGE')
            continue
        trailing_ret = (prices[i] / prices[i - window]) - 1
        if trailing_ret > bull_thresh:
            regimes.append('BULL')
        elif trailing_ret < bear_thresh:
            regimes.append('BEAR')
        else:
            regimes.append('RANGE')
    return np.array(regimes)


# ============================================================================
# P0.1 FIX: Signal Generation from Predictions
# ============================================================================

def compute_signals_from_predictions(
    predictions: np.ndarray,
    buy_threshold: float = 0.003,
    sell_threshold: float = 0.003,
    confidence_scores: np.ndarray = None,
    min_confidence: float = 0.30
) -> np.ndarray:
    """
    Convert raw predictions to discrete signals {-1, 0, +1}.
    
    P0.1 FIX: This function ensures proper signal generation from regressor
    predictions, avoiding the bug where final_signal was hardcoded to 0.
    
    For regressor-only and GBM modes, this converts continuous predicted returns
    into discrete trading signals based on thresholds.
    
    Args:
        predictions: Raw model outputs (predicted returns, typically in [-0.1, 0.1])
        buy_threshold: Threshold above which to generate BUY signal (default: 0.3% = 0.003)
        sell_threshold: Threshold below which to generate SELL signal (default: -0.3%)
        confidence_scores: Optional confidence filtering
        min_confidence: Minimum confidence to allow signal
        
    Returns:
        signals: Array of {-1, 0, +1} where:
            +1 = BUY (predicted return > buy_threshold)
             0 = HOLD (between thresholds)
            -1 = SELL (predicted return < -sell_threshold)
    """
    predictions = np.asarray(predictions, dtype=float)
    signals = np.zeros_like(predictions, dtype=np.int8)
    
    # Demean predictions to handle biased models
    pred_mean = np.mean(predictions)
    demeaned = predictions - pred_mean
    
    # Apply confidence filter if provided
    if confidence_scores is not None:
        confidence_scores = np.asarray(confidence_scores, dtype=float)
        mask = confidence_scores >= min_confidence
    else:
        mask = np.ones_like(predictions, dtype=bool)
    
    # Generate signals based on thresholds (using demeaned predictions)
    buy_mask = (demeaned > buy_threshold) & mask
    sell_mask = (demeaned < -sell_threshold) & mask
    
    signals[buy_mask] = 1
    signals[sell_mask] = -1
    
    return signals


def signals_to_positions(
    signals: np.ndarray,
    predictions: np.ndarray = None,
    kelly_fractions: np.ndarray = None,
    max_long: float = 1.0,
    max_short: float = 0.5,
    baseline: float = 0.30,
    scale: float = 15.0
) -> np.ndarray:
    """
    Convert discrete signals to continuous position sizes.
    
    P0.1 FIX: This function ensures proper position sizing from signals,
    using Kelly criterion when available or confidence scaling otherwise.
    
    Args:
        signals: {-1, 0, +1} signal array
        predictions: Raw predictions for confidence scaling
        kelly_fractions: Optional Kelly-optimal position sizes
        max_long: Maximum long position (1.0 = 100%)
        max_short: Maximum short position (0.5 = 50%)
        baseline: Baseline position size when predictions not available
        scale: Scaling factor for confidence-based sizing
        
    Returns:
        positions: Continuous position sizes in [-max_short, +max_long]
    """
    signals = np.asarray(signals, dtype=int)
    positions = np.zeros_like(signals, dtype=float)
    
    if kelly_fractions is not None:
        # Use Kelly sizing when available
        kelly_fractions = np.asarray(kelly_fractions, dtype=float)
        positions = np.where(signals != 0, kelly_fractions * np.sign(signals), 0.0)
    elif predictions is not None:
        # Use confidence scaling from predictions
        predictions = np.asarray(predictions, dtype=float)
        pred_mean = np.mean(predictions)
        demeaned = predictions - pred_mean
        
        # Scale demeaned predictions into positions
        positions = np.clip(demeaned * scale, -max_short, max_long)
        # Zero out positions where signal is 0 (HOLD)
        positions = np.where(signals != 0, positions, 0.0)
    else:
        # Use baseline position sizing
        positions = signals.astype(float) * baseline
    
    # Apply position limits
    positions = np.clip(positions, -max_short, max_long)
    
    return positions


def compute_smape(y_true, y_pred):
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    denom = np.where(denom == 0, 1e-8, denom)
    return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def evaluate_regressor_performance(y_true, y_pred):
    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(np.square(errors))))
    smape = compute_smape(y_true, y_pred)
    dir_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    corr = 0.0
    if len(y_true) > 1:
        corr_val = np.corrcoef(y_true, y_pred)[0, 1]
        corr = float(corr_val) if np.isfinite(corr_val) else 0.0
    return {
        'mae': mae,
        'rmse': rmse,
        'smape': smape,
        'directional_accuracy': dir_acc,
        'correlation': corr
    }


def compute_regressor_positions(pred_returns, strategy, threshold, scale, max_short):
    """
    Convert predicted returns into position sizes.
    
    P0.1-FIX: Demean predictions to handle biased models (e.g., LSTM that predicts 
    mostly positive returns). This ensures position sizing is based on relative
    predictions, not absolute bias.
    
    Args:
        pred_returns: Array of predicted returns
        strategy: 'simple_threshold' or 'confidence_scaling'
        threshold: Threshold for simple_threshold strategy
        scale: Scaling factor for confidence_scaling strategy
        max_short: Maximum short exposure
    
    Returns:
        Array of position sizes in [-max_short, 1.0]
    """
    pred_returns = np.asarray(pred_returns, dtype=float)
    
    # P0.1-FIX: Demean predictions to remove bias
    # This ensures position sizing reflects relative strength of signal, not absolute bias
    pred_mean = np.mean(pred_returns)
    demeaned_returns = pred_returns - pred_mean
    
    if strategy == 'simple_threshold':
        positions = np.zeros_like(pred_returns)
        # Use demeaned returns for threshold comparison
        positions[demeaned_returns > threshold] = 1.0
        positions[demeaned_returns < -threshold] = -max_short
    elif strategy == 'confidence_scaling':
        # Use demeaned returns for scaling
        positions = np.clip(demeaned_returns * scale, -max_short, 1.0)
    else:
        raise ValueError(f"Unsupported regressor strategy: {strategy}")
    return positions


def compute_classifier_positions(final_signals, buy_probs, sell_probs, max_short):
    """
    Compute position sizes from classifier probabilities.
    
    Classifier-only mode: Uses classifier probabilities directly for position sizing,
    ignoring regressor entirely. Position sizes are clipped to meaningful ranges:
    - BUY positions: 0.3 to 1.0 (minimum 30% position when signal fires)
    - SELL positions: 0.1 to 0.5 (with 0.5x multiplier for risk management)
    
    Args:
        final_signals: Array of signals (1=BUY, -1=SELL, 0=HOLD)
        buy_probs: BUY classifier probabilities
        sell_probs: SELL classifier probabilities  
        max_short: Maximum short exposure (typically 0.5)
    
    Returns:
        Array of position sizes
    """
    positions = np.zeros_like(buy_probs, dtype=float)
    
    buy_mask = final_signals == 1
    sell_mask = final_signals == -1
    
    # BUY positions: use buy probability as position size, clipped to [0.3, 1.0]
    if buy_mask.any():
        positions[buy_mask] = np.clip(buy_probs[buy_mask], 0.3, 1.0)
    
    # SELL positions: use sell probability with 0.5x multiplier for risk management, clipped to [0.1, 0.5]
    if sell_mask.any():
        positions[sell_mask] = -np.clip(sell_probs[sell_mask] * 0.5, 0.1, max_short)
    
    return positions


def fuse_positions(mode, classifier_positions, regressor_positions, final_signals, max_short,
                   regressor_preds=None, atr_percent=None, return_confidence=False, regime=None):
    """
    Fuse model positions based on fusion mode.

    Fusion Modes (December 2025 - Classifiers DEPRECATED):
    - 'stacking': Uses trained XGBoost meta-learner (RECOMMENDED - handled separately)
    - 'regressor': Uses only regressor predictions for position sizing.
    - 'regressor_only': Pure regressor strategy with confidence filtering.
    - 'gbm_only', 'gbm_heavy', 'balanced', 'lstm_heavy': GBM fusion modes.

    Note: Binary classifiers ('classifier', 'hybrid', 'weighted') were deprecated December 2025.

    Args:
        mode: Fusion mode ('stacking', 'regressor', 'regressor_only', 'gbm_*')
        classifier_positions: DEPRECATED - not used, kept for API compatibility
        regressor_positions: Position sizes from regressor
        final_signals: Array of signals (1=BUY, -1=SELL, 0=HOLD)
        max_short: Maximum short exposure
        regressor_preds: Raw regressor predictions (used for regressor_only mode)
        atr_percent: ATR as percentage of price (used for volatility adjustment)
        return_confidence: If True, returns (positions, confidence)
        regime: Market regime for position adjustment

    Returns:
        Fused position sizes array, or tuple (positions, confidence) if return_confidence=True
    """
    # Stacking mode is handled separately in main() with StackingPredictor
    if mode == 'stacking':
        raise RuntimeError("Stacking mode should be handled by StackingPredictor, not fuse_positions()")

    if mode == 'regressor':
        return regressor_positions

    if mode == 'regressor_only':
        # Pure regressor strategy
        positions, confidence = fuse_predictions_regressor_only(regressor_preds, atr_percent, regime=regime)
        if return_confidence:
            return positions, confidence
        return positions

    if mode in ('gbm_only', 'gbm_heavy', 'balanced', 'lstm_heavy'):
        # GBM fusion modes: use regressor_positions which have been
        # replaced with GBM-fused predictions in the main function
        positions, confidence = fuse_predictions_regressor_only(regressor_preds, atr_percent, regime=regime)
        if return_confidence:
            return positions, confidence
        return positions

    # Unknown mode - error instead of silent fallback
    raise ValueError(f"Unknown fusion mode '{mode}'. Valid modes: stacking, regressor, regressor_only, gbm_only, gbm_heavy, balanced, lstm_heavy")


def compute_regressor_confidence(regressor_preds: np.ndarray, window: int = 3) -> np.ndarray:
    """
    Compute confidence based on prediction magnitude and consistency.
    
    Confidence is based on:
    1. Magnitude: higher absolute predicted returns = higher confidence
       - Scaled adaptively based on prediction distribution
       - GBM models have smaller predictions (~0.002) vs LSTM (~0.01)
    2. Consistency: % of recent predictions with same sign
       - If last N days all predict same direction, confidence increases
    
    Args:
        regressor_preds: Array of predicted returns
        window: Lookback window for consistency calculation (default 3)
    
    Returns:
        Array of confidence scores in [0, 1]
    """
    regressor_preds = np.asarray(regressor_preds, dtype=float)
    confidence = np.zeros(len(regressor_preds))
    
    # Adaptive scaling: use actual prediction std to calibrate confidence
    # This makes confidence independent of model type (GBM vs LSTM)
    pred_std = np.std(regressor_preds)
    # Reference: predictions at 2 std from mean are "high confidence"
    # Cap reference at 0.005 minimum to avoid division issues
    magnitude_reference = max(pred_std * 2, 0.005)
    
    for i in range(len(regressor_preds)):
        # Adaptive magnitude confidence: scale based on prediction distribution
        magnitude_conf = min(abs(regressor_preds[i]) / magnitude_reference, 1.0)
        
        # Consistency confidence: % of recent predictions with same sign
        if i >= window:
            recent = regressor_preds[i-window:i]
            current_sign = np.sign(regressor_preds[i])
            same_sign = np.sum(np.sign(recent) == current_sign)
            consistency_conf = same_sign / window
        else:
            consistency_conf = 0.5  # Neutral if not enough history
        
        # Combined confidence (weighted average)
        # Magnitude weighted 50%, consistency 50% for more balanced signals
        confidence[i] = 0.5 * magnitude_conf + 0.5 * consistency_conf
    
    return confidence


def fuse_predictions_regressor_only(regressor_preds: np.ndarray, atr_percent: np.ndarray = None, regime: np.ndarray = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Pure regressor strategy - ignores classifiers completely.
    
    Uses regressor predictions directly to generate positions with ADAPTIVE scaling:
    - P0.1-FIX: Predictions are DEMEANED to remove model bias
    - Positions are scaled based on prediction distribution (std-normalized)
    - Predictions at +2 std → 100% long position
    - Predictions at -2 std → 50% short position
    
    Confidence filtering:
    - Predictions with confidence < MIN_CONFIDENCE are set to HOLD (0 position)
    - Confidence is based on prediction magnitude and directional consistency
    
    Args:
        regressor_preds: Array of predicted returns (e.g., 0.002 = 0.2% predicted gain)
        atr_percent: Optional volatility adjustment (ATR as percentage of price)
    
    Returns:
        tuple: (positions array in [-0.5, 1.0], confidence array in [0, 1])
    """
    if regressor_preds is None:
        raise ValueError("regressor_only mode requires regressor_preds to be provided")
    
    regressor_preds = np.asarray(regressor_preds, dtype=float)
    positions = np.zeros_like(regressor_preds)
    
    # P0.1-FIX: Demean predictions to remove model bias
    # This is CRITICAL for biased models like LSTM that predict mostly positive returns
    pred_mean = np.mean(regressor_preds)
    demeaned_preds = regressor_preds - pred_mean
    
    # Compute confidence scores on ORIGINAL predictions (for magnitude)
    confidence = compute_regressor_confidence(regressor_preds, window=3)
    MIN_CONFIDENCE = 0.25  # Lowered from 0.3 to allow more signals
    
    # ADAPTIVE position scaling based on prediction distribution
    # This makes the strategy work for both LSTM (~2% predictions) and GBM (~0.2% predictions)
    pred_std = np.std(demeaned_preds)  # Use demeaned for std calculation
    pred_std = max(pred_std, 0.001)  # Minimum to avoid division by zero
    
    # Normalize predictions: 2 std → full position
    # Scale factor: 1.0 / (2 * std) so predictions at 2*std map to 1.0
    ADAPTIVE_SCALE = 1.0 / (2.0 * pred_std)
    
    for i in range(len(regressor_preds)):
        pred = demeaned_preds[i]  # Use DEMEANED prediction for position
        
        # Filter out low-confidence predictions
        if confidence[i] < MIN_CONFIDENCE:
            positions[i] = 0  # Set to HOLD if confidence too low
            continue
        
        # Adaptive position sizing based on normalized prediction
        normalized_pred = pred * ADAPTIVE_SCALE
        
        if pred > 0:
            # Long position: clip to [0, 1.0]
            position = np.clip(normalized_pred, 0, 1.0)
            
            # P0.3-FIX: Reduce long exposure in Bear Regime (-1)
            if regime is not None and i < len(regime) and regime[i] == -1:
                position *= 0.5  # Reduce longs by 50% in bear market
        else:
            # Short position handling
            # P0.3-FIX: Aggressive short selling in Bear Regime
            if regime is not None and i < len(regime) and regime[i] == -1:
                # Bear Regime: Allow full -1.0 short exposure (Aggressive)
                position = np.clip(normalized_pred, -1.0, 0)
            else:
                # Normal/Bull Regime: Cap short at -0.5 (Conservative)
                position = np.clip(normalized_pred, -0.5, 0)
        
        # Optional: reduce position size in high volatility (ATR > 3%)
        if atr_percent is not None and i < len(atr_percent) and atr_percent[i] > 0.03:
            position *= 0.7  # Reduce by 30%
        
        positions[i] = position
    
    return positions, confidence


def simulate_price_path(start_price, predicted_returns):
    prices = []
    current = float(start_price)
    for r in predicted_returns:
        current = current * (1.0 + float(r))
        prices.append(current)
    return np.array(prices)


def create_adaptive_profit_targets(entry_price, current_atr, initial_shares, date_idx, entry_date_idx, 
                                   resistance_levels=None, current_volume_ratio=1.0):
    """
    Create adaptive ATR-based profit targets that respond to volatility and market structure.
    
    Instead of fixed percentage targets, uses ATR multiples for dynamic targets that adapt to:
    - Current market volatility (ATR)
    - Time degradation (reduce targets if held too long)
    - Support/resistance levels (early exit near resistance)
    - Volume confirmation (hold for next target on breakout)
    
    Args:
        entry_price: Entry price for the position
        current_atr: Current ATR as percentage of price
        initial_shares: Total shares in position
        date_idx: Current date index
        entry_date_idx: Entry date index (for time degradation calc)
        resistance_levels: List of resistance prices (optional)
        current_volume_ratio: Current volume / average volume
    
    Returns:
        List of order dictionaries with adaptive targets
    """
    orders = []
    
    # Calculate days held for time degradation
    days_held = date_idx - entry_date_idx
    
    # Apply time degradation to targets
    if days_held >= MAX_POSITION_DAYS_STAGE1:
        time_factor = TIME_DEGRADATION_FACTOR  # Reduce targets by 20%
    else:
        time_factor = 1.0
    
    # Calculate ATR-based targets
    for atr_mult, size_pct in zip(ATR_TARGET_MULTIPLIERS, ATR_SCALE_OUT_PERCENTAGES):
        # Base target from ATR
        target_move = current_atr * atr_mult * time_factor
        trigger_price = entry_price * (1.0 + target_move)
        
        # Check if near resistance level
        near_resistance = False
        if resistance_levels:
            for resistance in resistance_levels:
                if abs(trigger_price - resistance) / resistance < RESISTANCE_PROXIMITY_THRESHOLD:
                    near_resistance = True
                    # Approaching resistance - take profits early at 90% of target
                    if current_volume_ratio < VOLUME_SURGE_THRESHOLD:
                        trigger_price = resistance * EARLY_EXIT_DISCOUNT
                    # Else: volume surge detected, hold for breakout
                    break
        
        shares_to_sell = initial_shares * size_pct
        
        orders.append({
            'date_idx': date_idx,
            'trigger_price': trigger_price,
            'shares_to_sell': shares_to_sell,
            'atr_multiplier': atr_mult,
            'near_resistance': near_resistance,
            'time_degraded': days_held >= MAX_POSITION_DAYS_STAGE1,
            'executed': False
        })
    
    return orders


def calculate_adaptive_stop_loss(entry_price, current_price, entry_atr, current_atr, 
                                 targets_hit, highest_price):
    """
    Calculate adaptive stop-loss based on ATR and profit progress.
    
    Stop evolution:
    1. Initial: entry - 2*ATR
    2. After target 1: Move to breakeven (entry price)
        # Attempt reconstruction if no detailed history provided
        if (not trades) and trade_log_df is not None and not trade_log_df.empty:
            # Lot-based FIFO reconstruction:
            # - Treat any increase in 'shares' as a new lot (BUY/adding to position)
            # - Treat any decrease in 'shares' as closing lots (SELL/scale-out)
            # This produces one or more trade records per buy/scale event and handles multiple
            # BUY signals that build a position (e.g., 6 buys) and subsequent partial exits.
            trades = []
            open_lots = []  # list of dicts: {'entry_index','entry_price','shares_remaining','entry_date'}
            prev_shares = 0.0
            # Choose share column robustly
            share_cols = None
            for k in ('shares', 'current_shares', 'position_shares'):
                if k in trade_log_df.columns:
                    share_cols = k
                    break

            for idx, row in trade_log_df.iterrows():
                try:
                    shares = float(row.get(share_cols)) if share_cols and not pd.isna(row.get(share_cols)) else 0.0
                except Exception:
                    shares = 0.0
                price = row.get('price') if 'price' in row.index else None
                date = row.get('date') if 'date' in row.index else None

                # Detect increase -> new lot (BUY or add)
                if shares > prev_shares + 1e-12:
                    added = shares - prev_shares
                    # Ignore tiny noise below min threshold
                    if added >= (HYBRID_MIN_SHARES_THRESHOLD if 'HYBRID_MIN_SHARES_THRESHOLD' in globals() else 0.0):
                        lot = {
                            'entry_index': int(idx),
                            'entry_price': float(price) if price is not None and not pd.isna(price) else None,
                            'shares_remaining': float(added),
                            'entry_date': date
                        }
                        open_lots.append(lot)

                # Detect decrease -> consume lots (SELL / scale-out)
                if shares < prev_shares - 1e-12:
                    reduction = prev_shares - shares
                    # Consume FIFO
                    while reduction > 1e-12 and open_lots:
                        lot = open_lots[0]
                        consume = min(lot['shares_remaining'], reduction)
                        # Record an exit for the consumed portion
                        exit_price = float(price) if price is not None and not pd.isna(price) else None
                        days_held = None
                        try:
                            days_held = int(int(idx) - int(lot.get('entry_index', int(idx))))
                        except Exception:
                            days_held = None
                        pnl = None
                        if lot.get('entry_price') is not None and exit_price is not None and lot.get('entry_price') != 0:
                            pnl = (exit_price / lot.get('entry_price') - 1.0)

                        exit_row = {
                            'entry_price': lot.get('entry_price'),
                            'exit_price': exit_price,
                            'entry_index': lot.get('entry_index'),
                            'exit_index': int(idx),
                            'days_held': days_held,
                            'targets_hit': [False, False, False, False],
                            'time_degraded': False,
                            'resistance_exit': False,
                            'pnl_pct': pnl,
                            'price_path': [lot.get('entry_price'), exit_price] if lot.get('entry_price') is not None and exit_price is not None else None
                        }
                        trades.append(exit_row)

                        # Update lot and reduction
                        lot['shares_remaining'] -= consume
                        reduction -= consume
                        if lot['shares_remaining'] <= 1e-12:
                            # fully consumed -> pop
                            open_lots.pop(0)
                        else:
                            # partially consumed -> keep remaining in front
                            open_lots[0] = lot

                prev_shares = shares

            # After iterating, any remaining open lots are considered open trades at end
            if open_lots:
                try:
                    last_idx = int(trade_log_df.index[-1])
                except Exception:
                    last_idx = None
                last_row = trade_log_df.iloc[-1]
                last_price = last_row.get('price') if 'price' in last_row.index else None
                exit_price = float(last_price) if last_price is not None and not pd.isna(last_price) else None
                for lot in open_lots:
                    days_held = None
                    try:
                        days_held = int(last_idx - int(lot.get('entry_index', last_idx))) if last_idx is not None else None
                    except Exception:
                        days_held = None
                    pnl = None
                    if lot.get('entry_price') is not None and exit_price is not None and lot.get('entry_price') != 0:
                        pnl = (exit_price / lot.get('entry_price') - 1.0)
                    exit_row = {
                        'entry_price': lot.get('entry_price'),
                        'exit_price': exit_price,
                        'entry_index': lot.get('entry_index'),
                        'exit_index': last_idx,
                        'days_held': days_held,
                        'targets_hit': [False, False, False, False],
                        'time_degraded': False,
                        'resistance_exit': False,
                        'pnl_pct': pnl,
                        'price_path': [lot.get('entry_price'), exit_price] if lot.get('entry_price') is not None and exit_price is not None else None,
                        'open_at_end': True
                    }
                    trades.append(exit_row)
    - Profit-target order system with trailing stops
    - Drawdown-aware risk management
    - Position limits and leverage constraints
    
    Args:
        final_signals: Array of signals (1=BUY, -1=SELL, 0=HOLD)
        buy_probs: Array of BUY probabilities (from classifier)
        sell_probs: Array of SELL probabilities (from classifier)
        prices: Array of prices
        atr_percent: Array of ATR as percentage of price (from feature_engineer)
        vol_ratio_5_20: Array of 5-day/20-day volatility ratio (from feature_engineer)
        verbose: If True, log every 50th iteration. If False, only log trades.
    
    Returns:
        Array of position sizes (as fraction of equity)
    """
    # ===================================================================
    # INPUT VALIDATION AND ARRAY CONVERSION
    # ===================================================================
    signals = np.asarray(final_signals, dtype=int)
    buy_probs = np.asarray(buy_probs, dtype=float)
    sell_probs = np.asarray(sell_probs, dtype=float)
    prices = np.asarray(prices, dtype=float)
    
    # Validate all core arrays have same length
    assert len(signals) == len(buy_probs) == len(sell_probs) == len(prices), \
        f"Length mismatch: signals={len(signals)}, buy_probs={len(buy_probs)}, " \
        f"sell_probs={len(sell_probs)}, prices={len(prices)}"
    
    positions = np.zeros_like(signals, dtype=float)
    
    # ===================================================================
    # VOLATILITY FEATURES WITH FALLBACK
    # ===================================================================
    
    # Process ATR (Average True Range) for volatility measurement
    if atr_percent is not None:
        atr_percent = np.asarray(atr_percent, dtype=float)
        # Validate alignment with signals
        assert len(atr_percent) == len(signals), \
            f"ATR length mismatch: atr_percent={len(atr_percent)} vs signals={len(signals)}"
    else:
        # Fallback: calculate simple volatility from price changes
        print("  ⚠️  No ATR data provided, calculating from price changes...")
        atr_percent = np.zeros_like(prices, dtype=float)
        if len(prices) > 1:
            returns = np.diff(prices) / prices[:-1]
            atr_percent[1:] = np.abs(returns)
        atr_percent[0] = np.median(atr_percent[1:]) if len(atr_percent) > 1 else 0.02
    
    # Process volatility ratio (5-day/20-day)
    if vol_ratio_5_20 is not None:
        vol_ratio_5_20 = np.asarray(vol_ratio_5_20, dtype=float)
        # Validate alignment with signals
        assert len(vol_ratio_5_20) == len(signals), \
            f"Vol ratio length mismatch: vol_ratio_5_20={len(vol_ratio_5_20)} vs signals={len(signals)}"
    else:
        # Fallback: assume normal volatility regime
        print("  ⚠️  No volatility ratio data provided, using normal regime (1.0)...")
        vol_ratio_5_20 = np.ones_like(prices, dtype=float)
    
    # Calculate baseline volatility (median of last 252 days, or available data)
    lookback = min(VOL_BASELINE_LOOKBACK, len(atr_percent))
    if lookback > 0:
        baseline_atr = np.median(atr_percent[-lookback:] if lookback < len(atr_percent) else atr_percent)
    else:
        baseline_atr = 0.02  # 2% default
    
    baseline_atr = max(baseline_atr, 0.001)  # Prevent division by zero
    
    # Detect market regime for all time periods
    market_regime = detect_market_regime(prices)
    
    print(f"\n=== Kelly Criterion + Regime-Aware Position Sizing ===")
    print(f"Baseline ATR (median last {lookback} days): {baseline_atr:.4f} ({baseline_atr*100:.2f}%)")
    
    # Count regimes
    bullish_count = np.sum(market_regime == 1)
    bearish_count = np.sum(market_regime == -1)
    sideways_count = np.sum(market_regime == 0)
    print(f"Market Regimes: Bullish={bullish_count}, Bearish={bearish_count}, Sideways={sideways_count}")
    
    current_shares = 0.0
    available_cash = HYBRID_REFERENCE_EQUITY
    
    # Trade history for Kelly Criterion
    trade_history = []  # List of completed trades with P&L
    
    # Volatility metrics tracking
    vol_adjustments_log = []
    regime_changes_log = []

    # Comprehensive per-step trade log (optional exportable DataFrame)
    trade_log_entries = []
    
    # Drawdown tracking for risk management
    high_water_mark = HYBRID_REFERENCE_EQUITY
    
    # Position tracking for scale-out strategy
    entry_price = 0.0
    entry_date_idx = 0  # Track entry time for time degradation
    entry_atr = 0.0  # Track ATR at entry for adaptive stops
    initial_shares = 0.0
    profit_targets_hit = [False] * len(PROFIT_TARGETS)
    current_stop_loss = INITIAL_STOP_LOSS
    highest_price_since_entry = 0.0
    
    # Profit-target order queue
    trade_orders_queue = []  # List of pending profit-target orders
    
    # Detect resistance levels for adaptive exits (pre-compute once before loop)
    resistance_levels = detect_resistance_levels(prices)
    
    # Pre-cache regime labels to avoid dict lookups in loop
    REGIME_LABELS = {1: 'BULLISH', -1: 'BEARISH', 0: 'SIDEWAYS'}
    
    # Tolerance band rebalancing state
    position_state = {
        'target_weight': TARGET_POSITION_WEIGHT,
        'current_weight': 0.0,
        'upper_band': UPPER_REBALANCE_BAND,
        'lower_band': LOWER_REBALANCE_BAND
    }
    
    # Track position metrics
    position_metrics = {
        'current_leverage': 0.0,
        'max_leverage_seen': 0.0,
        'positions_adjusted_by_kelly': 0,
        'positions_adjusted_by_regime': 0
    }

    for i in range(signals.shape[0]):
        price = float(prices[i]) if i < prices.size else 0.0
        if price <= 0.0:
            positions[i] = 0.0
            continue
            
        signal = int(signals[i])
        equity = max(available_cash + current_shares * price, 1e-6)
        
        # Get current volatility metrics
        current_atr = float(atr_percent[i]) if i < len(atr_percent) else baseline_atr
        current_vol_ratio = float(vol_ratio_5_20[i]) if i < len(vol_ratio_5_20) else 1.0
        
        # Calculate volatility adjustment factor
        # vol_adjust = clip(baseline_vol / current_atr, 0.5, 2.0)
        # When volatility high (ATR large): vol_adjust < 1.0 → reduce position
        # When volatility low (ATR small): vol_adjust > 1.0 → increase position
        vol_adjust = np.clip(baseline_atr / current_atr, VOL_ADJUST_MIN, VOL_ADJUST_MAX)
        
        # Detect volatility regime and adjust max position
        if current_vol_ratio > VOL_HIGH_REGIME_THRESHOLD:
            # Very high volatility - reduce max position to 25%
            vol_regime_max_position = VOL_HIGH_MAX_POSITION
            regime_label = "HIGH"
        elif current_vol_ratio < VOL_LOW_REGIME_THRESHOLD:
            # Very low volatility - allow up to 50%
            vol_regime_max_position = VOL_LOW_MAX_POSITION
            regime_label = "LOW"
        else:
            # Normal volatility - use base max position
            vol_regime_max_position = HYBRID_MAX_POSITION
            regime_label = "NORMAL"
        
        # Track volatility adjustments for logging (only if verbose)
        if verbose and i % 50 == 0:  # Log every 50 periods only in verbose mode
            vol_adjustments_log.append({
                'index': i,
                'current_atr': current_atr,
                'vol_adjust': vol_adjust,
                'vol_ratio': current_vol_ratio,
                'regime': regime_label,
                'max_position': vol_regime_max_position
            })
        
        # Update high-water mark and calculate drawdown adjustments
        high_water_mark = max(high_water_mark, equity)
        drawdown_adj = calculate_drawdown_adjustments(equity, high_water_mark)
        drawdown_max_position = drawdown_adj['max_position']
        drawdown_scale_factor = drawdown_adj['scale_factor']
        drawdown_multiplier = drawdown_adj['multiplier']
        
        # Combine volatility and drawdown limits (use most conservative)
        effective_max_position = min(vol_regime_max_position, drawdown_max_position)
        
        # Check and execute pending profit-target orders
        if trade_orders_queue and current_shares > 0:
            for order in trade_orders_queue:
                if not order.get('executed', False) and price >= order['trigger_price']:
                    # Execute profit-target order
                    shares_to_sell = min(order['shares_to_sell'], current_shares)
                    if shares_to_sell >= HYBRID_MIN_SHARES_THRESHOLD:
                        sale_value = shares_to_sell * price
                        available_cash += sale_value
                        current_shares -= shares_to_sell
                        order['executed'] = True
                        
                        # Calculate unrealized P&L at execution
                        unrealized_pnl_pct = (price - entry_price) / entry_price if entry_price > 0 else 0.0
                        position_pct_sold = (shares_to_sell / initial_shares) * 100 if initial_shares > 0 else 0.0
                        
                        # Get order details
                        atr_mult = order.get('atr_multiplier', 0)
                        time_degraded = order.get('time_degraded', False)
                        near_resistance = order.get('near_resistance', False)
                        
                        # Log trade marker
                        print(f"  🎯 ADAPTIVE PROFIT-TARGET ORDER EXECUTED at +{unrealized_pnl_pct*100:.1f}% "
                              f"({atr_mult:.1f}xATR, {position_pct_sold:.0f}% position, ${sale_value:.2f})")
                        if time_degraded:
                            print(f"     ⚠️ Time-degraded exit")
                        if near_resistance:
                            print(f"     📍 Near resistance (early exit)")
                        
                        # If position reduced below threshold, close entirely and cancel orders
                        if current_shares < HYBRID_MIN_SHARES_THRESHOLD:
                            final_sale_value = current_shares * price
                            available_cash += final_sale_value
                            total_pnl = ((price - entry_price) / entry_price) * 100 if entry_price > 0 else 0.0
                            total_pnl_decimal = total_pnl / 100.0
                            
                            # Add to trade history for Kelly Criterion
                            trade_history.append({'pnl_pct': total_pnl_decimal})
                            
                            print(f"  ✅ POSITION CLOSED (below threshold) - Total P&L: +{total_pnl:.2f}%")
                            current_shares = 0.0
                            entry_price = 0.0
                            entry_date_idx = 0
                            entry_atr = 0.0
                            initial_shares = 0.0
                            trade_orders_queue = []  # Cancel all pending orders
                            profit_targets_hit = [False] * len(PROFIT_TARGETS)
                            current_stop_loss = INITIAL_STOP_LOSS
                            highest_price_since_entry = 0.0
                            break
        
        # Update current position weight
        equity = max(available_cash + current_shares * price, 1e-6)
        current_position_value = current_shares * price
        position_state['current_weight'] = current_position_value / equity if equity > 0 else 0.0
        
        # Time-based forced exit check (positions held >20 days without hitting targets)
        if current_shares > 0 and entry_price > 0:
            days_held = i - entry_date_idx
            if days_held >= MAX_POSITION_DAYS_STAGE2:
                # Force exit at market after 20 days
                shares_to_sell = current_shares
                exit_value = shares_to_sell * price
                available_cash += exit_value
                
                total_pnl = ((price - entry_price) / entry_price) * 100 if entry_price > 0 else 0.0
                total_pnl_decimal = total_pnl / 100.0
                
                # Add to trade history
                trade_history.append({'pnl_pct': total_pnl_decimal})
                
                print(f"  ⏰ TIME-BASED EXIT: Position held {days_held} days (>{MAX_POSITION_DAYS_STAGE2} days)")
                print(f"     Exiting {shares_to_sell:.2f} shares @ ${price:.2f} (P&L: {total_pnl:+.2f}%, ${exit_value:.2f})")
                
                current_shares = 0.0
                entry_price = 0.0
                entry_date_idx = 0
                entry_atr = 0.0
                initial_shares = 0.0
                trade_orders_queue = []
                profit_targets_hit = [False] * len(PROFIT_TARGETS)
                current_stop_loss = INITIAL_STOP_LOSS
                highest_price_since_entry = 0.0
                positions[i] = 0.0
                continue
        
        # Tolerance band rebalancing check (before signal processing)
        if current_shares > 0:
            # Check if position exceeds upper band - trigger partial sell
            if position_state['current_weight'] > position_state['upper_band']:
                excess_weight = position_state['current_weight'] - position_state['target_weight']
                excess_value = excess_weight * equity
                shares_to_rebalance = excess_value / price if price > 0 else 0.0
                
                if shares_to_rebalance >= HYBRID_MIN_SHARES_THRESHOLD:
                    # Execute rebalancing sell
                    available_cash += shares_to_rebalance * price
                    current_shares -= shares_to_rebalance
                    # Update position weight after rebalance
                    current_position_value = current_shares * price
                    position_state['current_weight'] = current_position_value / equity if equity > 0 else 0.0
            
            # Check if position falls below lower band - trigger buy if BUY signal active
            elif position_state['current_weight'] < position_state['lower_band'] and signal > 0:
                deficit_weight = position_state['target_weight'] - position_state['current_weight']
                deficit_value = deficit_weight * equity
                shares_to_add = deficit_value / price if price > 0 else 0.0
                max_affordable = available_cash / price if price > 0 else 0.0
                shares_to_add = min(shares_to_add, max_affordable)
                
                if shares_to_add >= HYBRID_MIN_SHARES_THRESHOLD:
                    # Execute rebalancing buy
                    available_cash -= shares_to_add * price
                    current_shares += shares_to_add
                    available_cash = max(available_cash, 0.0)
                    # Update average entry price
                    if entry_price > 0:
                        total_cost = (initial_shares * entry_price) + (shares_to_add * price)
                        new_total_shares = initial_shares + shares_to_add
                        entry_price = total_cost / new_total_shares if new_total_shares > 0 else price
                        initial_shares = new_total_shares
                    # Update position weight after rebalance
                    current_position_value = current_shares * price
                    position_state['current_weight'] = current_position_value / equity if equity > 0 else 0.0
        
        # Update highest price tracking for trailing stop
        if current_shares > 0 and price > highest_price_since_entry:
            highest_price_since_entry = price
            
        # Check for adaptive stop-loss exit
        if current_shares > 0 and entry_price > 0 and entry_atr > 0:
            unrealized_pnl_pct = (price - entry_price) / entry_price
            
            # Calculate adaptive stop price using ATR
            adaptive_stop_price = calculate_adaptive_stop_loss(
                entry_price=entry_price,
                current_price=price,
                entry_atr=entry_atr,
                current_atr=current_atr,
                targets_hit=profit_targets_hit,
                highest_price=highest_price_since_entry
            )
            
            # Check if stop-loss triggered
            if price <= adaptive_stop_price:
                # Stop-loss hit - exit entire position and cancel orders
                shares_to_sell = current_shares
                exit_value = shares_to_sell * price
                available_cash += exit_value
                
                # Calculate final P&L
                total_pnl = unrealized_pnl_pct * 100
                total_pnl_decimal = unrealized_pnl_pct
                stop_distance = ((adaptive_stop_price - entry_price) / entry_price) * 100
                
                # Add to trade history for Kelly Criterion
                trade_history.append({'pnl_pct': total_pnl_decimal})
                
                # Log stop-loss trigger
                print(f"  🛑 ADAPTIVE STOP-LOSS TRIGGERED at {stop_distance:+.1f}% from entry "
                      f"(P&L: {total_pnl:+.2f}%, ${exit_value:.2f})")
                
                current_shares = 0.0
                entry_price = 0.0
                entry_date_idx = 0
                entry_atr = 0.0
                initial_shares = 0.0
                trade_orders_queue = []  # Cancel all pending orders
                profit_targets_hit = [False] * len(PROFIT_TARGETS)
                current_stop_loss = INITIAL_STOP_LOSS
                highest_price_since_entry = 0.0
                positions[i] = 0.0
                continue
            
            # Check adaptive profit targets for scale-out
            if trade_orders_queue:
                for order in trade_orders_queue:
                    target_idx = order.get('target_index', 0)
                    if not profit_targets_hit[target_idx] and price >= order['trigger_price']:
                        # Execute scale-out at this target
                        shares_to_sell = min(order['shares_to_sell'], current_shares)
                        
                        if shares_to_sell >= HYBRID_MIN_SHARES_THRESHOLD:
                            sale_value = shares_to_sell * price
                            available_cash += sale_value
                            current_shares -= shares_to_sell
                            profit_targets_hit[target_idx] = True
                            
                            # Log scale-out execution
                            atr_mult = order.get('atr_multiplier', 0)
                            time_degraded = order.get('time_degraded', False)
                            near_resistance = order.get('near_resistance', False)
                            position_pct = (shares_to_sell / initial_shares) * 100 if initial_shares > 0 else 0
                            
                            print(f"  📊 ADAPTIVE SCALE-OUT at {atr_mult:.1f}xATR target "
                                  f"({position_pct:.0f}% of initial position, ${sale_value:.2f})")
                            if time_degraded:
                                print(f"     ⚠️ Time-degraded target (position held >{MAX_POSITION_DAYS_STAGE1} days)")
                            if near_resistance:
                                print(f"     📍 Near resistance level (early exit at {EARLY_EXIT_DISCOUNT*100:.0f}%)")
                            
                            # If position reduced below threshold, close entirely
                            if current_shares < HYBRID_MIN_SHARES_THRESHOLD:
                                final_sale = current_shares * price
                                available_cash += final_sale
                                total_pnl = ((price - entry_price) / entry_price) * 100 if entry_price > 0 else 0.0
                                total_pnl_decimal = total_pnl / 100.0
                                
                                # Add to trade history for Kelly Criterion
                                trade_history.append({'pnl_pct': total_pnl_decimal})
                                
                                print(f"  ✅ POSITION CLOSED (below threshold) - Total P&L: +{total_pnl:.2f}%")
                                current_shares = 0.0
                                entry_price = 0.0
                                entry_date_idx = 0
                                entry_atr = 0.0
                                initial_shares = 0.0
                                profit_targets_hit = [False] * len(PROFIT_TARGETS)
                                current_stop_loss = INITIAL_STOP_LOSS
                                highest_price_since_entry = 0.0
                                break
        
        # Handle HOLD signal - maintain current position
        if signal == 0:
            positions[i] = (current_shares * price) / equity if equity > 0 and price > 0 else 0.0
            continue

        # ===================================================================
        # KELLY CRITERION + REGIME-AWARE POSITION SIZING
        # ===================================================================
        
        # Get win probability from classifier
        win_prob = float(buy_probs[i] if signal > 0 else sell_probs[i])
        win_prob = np.clip(win_prob, 0.01, 0.99)  # Avoid extremes
        
        # Calculate win/loss ratio from trade history
        win_loss_ratio = calculate_win_loss_ratio_from_history(trade_history)
        
        # Calculate Kelly fraction
        kelly_fraction = calculate_kelly_fraction(win_prob, win_loss_ratio)
        
        # Get market regime for this time period
        regime = int(market_regime[i]) if i < len(market_regime) else 0
        
        # Apply regime adjustment to Kelly fraction
        regime_adjusted_kelly = apply_regime_adjustment(kelly_fraction, regime, signal)
        
        # Apply volatility adjustment
        # vol_adjust = baseline_vol / current_vol (higher vol = smaller position)
        vol_scalar = np.clip(baseline_atr / current_atr, VOL_ADJUST_MIN, VOL_ADJUST_MAX)
        volatility_adjusted_fraction = regime_adjusted_kelly * vol_scalar
        
        # Apply drawdown adjustment
        volatility_adjusted_fraction *= drawdown_multiplier
        
        # Apply concentration-based position limits
        current_position_pct = (current_shares * price) / equity if equity > 0 and price > 0 else 0.0
        concentration_factor = max(0.5, 1.0 - (current_position_pct / effective_max_position))
        
        # Final position fraction with all adjustments
        target_fraction = volatility_adjusted_fraction * concentration_factor
        
        # Enforce hard limits
        target_fraction = float(np.clip(target_fraction, KELLY_MIN_FRACTION, min(effective_max_position, MAX_SINGLE_POSITION)))
        
        # Track position metrics
        position_metrics['current_leverage'] = abs(current_position_pct)
        position_metrics['max_leverage_seen'] = max(position_metrics['max_leverage_seen'], position_metrics['current_leverage'])
        
        if abs(kelly_fraction - regime_adjusted_kelly) > 0.01:
            position_metrics['positions_adjusted_by_regime'] += 1
        
        if win_prob > 0.5:  # Only count when we have edge
            position_metrics['positions_adjusted_by_kelly'] += 1
        
        # Log position sizing calculation every 50 periods (only if verbose)
        if verbose and i % 50 == 0:
            regime_label = REGIME_LABELS[regime]
            print(f"  [i={i}] Kelly: {kelly_fraction:.3f} | Regime: {regime_label} | "
                  f"Adj: {regime_adjusted_kelly:.3f} | Vol: {vol_scalar:.2f} | "
                  f"Final: {target_fraction:.3f} ({target_fraction*100:.1f}%)")
        
        if signal > 0:  # BUY signal
            target_position_value = target_fraction * equity
            target_shares = target_position_value / price if price > 0 else current_shares
            share_delta = target_shares - current_shares
            
            if share_delta > 0:
                max_buy = available_cash / price if price > 0 else 0.0
                share_delta = min(share_delta, max_buy)
                
                if abs(share_delta) >= HYBRID_MIN_SHARES_THRESHOLD:
                    # New entry or adding to position
                    if current_shares == 0:
                        # Fresh entry - reset tracking and create adaptive profit-target orders
                        entry_price = price
                        entry_date_idx = i  # Track entry time
                        entry_atr = current_atr  # Track ATR at entry
                        initial_shares = share_delta
                        profit_targets_hit = [False] * len(PROFIT_TARGETS)
                        current_stop_loss = INITIAL_STOP_LOSS
                        highest_price_since_entry = price
                        
                        # Get volume ratio for breakout confirmation
                        volume_ratio = 1.0  # Default if not available
                        # Note: Would need to pass volume data to use actual ratios
                        
                        # Create adaptive profit-target orders
                        trade_orders_queue = create_adaptive_profit_targets(
                            entry_price=entry_price,
                            current_atr=current_atr,
                            initial_shares=share_delta,
                            date_idx=i,
                            entry_date_idx=entry_date_idx,
                            resistance_levels=resistance_levels,
                            current_volume_ratio=volume_ratio
                        )
                        
                        # Log entry with Kelly/regime info
                        entry_value = share_delta * price
                        regime_label = REGIME_LABELS[regime]
                        initial_stop = entry_price * (1 - INITIAL_STOP_ATR_MULTIPLIER * current_atr)
                        
                        print(f"\n  🟢 NEW POSITION ENTRY: {share_delta:.2f} shares @ ${price:.2f} (${entry_value:.2f})")
                        print(f"     Kelly: {kelly_fraction:.3f} | W/L Ratio: {win_loss_ratio:.2f} | Win Prob: {win_prob:.2f}")
                        print(f"     Regime: {regime_label} | Position Size: {target_fraction*100:.1f}%")
                        print(f"     ATR: {current_atr*100:.2f}% | Initial Stop: ${initial_stop:.2f} ({INITIAL_STOP_ATR_MULTIPLIER}xATR)")
                        
                        # Show adaptive profit targets
                        if trade_orders_queue:
                            print(f"     Adaptive Targets (ATR-based):")
                            for idx, order in enumerate(trade_orders_queue):
                                target_price = order['trigger_price']
                                target_pct = ((target_price - entry_price) / entry_price) * 100
                                atr_mult = order.get('atr_multiplier', 0)
                                shares_pct = (order['shares_to_sell'] / initial_shares) * 100 if initial_shares > 0 else 0
                                print(f"       T{idx+1}: ${target_price:.2f} (+{target_pct:.1f}%, {atr_mult:.1f}xATR) - "
                                      f"scale {shares_pct:.0f}%")
                    else:
                        # Adding to existing position - update average entry and recreate adaptive orders
                        old_entry = entry_price
                        total_cost = (current_shares * entry_price) + (share_delta * price)
                        new_total_shares = current_shares + share_delta
                        entry_price = total_cost / new_total_shares
                        
                        # Update ATR (weighted average based on share amounts)
                        entry_atr = ((current_shares * entry_atr) + (share_delta * current_atr)) / new_total_shares
                        
                        initial_shares = new_total_shares
                        # Reset profit targets and recreate orders when adding to position
                        profit_targets_hit = [False] * len(PROFIT_TARGETS)
                        highest_price_since_entry = max(highest_price_since_entry, price)
                        
                        # Get volume ratio
                        volume_ratio = 1.0  # Default
                        
                        # Recreate adaptive profit-target orders with new average entry
                        trade_orders_queue = create_adaptive_profit_targets(
                            entry_price=entry_price,
                            current_atr=entry_atr,
                            initial_shares=new_total_shares,
                            date_idx=i,
                            entry_date_idx=entry_date_idx,  # Keep original entry time
                            resistance_levels=resistance_levels,
                            current_volume_ratio=volume_ratio
                        )
                        
                        # Log position addition
                        add_value = share_delta * price
                        print(f"  🔼 ADDING TO POSITION: +{share_delta:.2f} shares @ ${price:.2f} (${add_value:.2f})")
                        print(f"     New avg entry: ${old_entry:.2f} → ${entry_price:.2f} | Total: {new_total_shares:.2f} shares")
                        print(f"     Updated ATR: {entry_atr*100:.2f}% | Adaptive targets recalculated")
                    
                    current_shares += share_delta
                    available_cash -= share_delta * price
                    available_cash = max(available_cash, 0.0)
                    
        else:  # SELL signal
            # Exit remaining position and cancel all pending orders
            if current_shares > 0:
                # Sell all remaining shares on SELL signal
                shares_to_sell = current_shares
                
                if shares_to_sell >= HYBRID_MIN_SHARES_THRESHOLD:
                    exit_value = shares_to_sell * price
                    available_cash += exit_value
                    
                    # Calculate final P&L
                    total_pnl = ((price - entry_price) / entry_price) * 100 if entry_price > 0 else 0.0
                    
                    # Count cancelled orders
                    cancelled_orders = sum(1 for order in trade_orders_queue if not order.get('executed', False))
                    
                    # Log SELL signal exit
                    print(f"  🔴 SELL SIGNAL EXIT: {shares_to_sell:.2f} shares @ ${price:.2f} "
                          f"(P&L: {total_pnl:+.2f}%, ${exit_value:.2f})")
                    if cancelled_orders > 0:
                        print(f"     Cancelled {cancelled_orders} pending adaptive profit-target order(s)")
                    
                    current_shares = 0.0
                    entry_price = 0.0
                    entry_date_idx = 0
                    entry_atr = 0.0
                    initial_shares = 0.0
                    trade_orders_queue = []  # Cancel all pending orders
                    profit_targets_hit = [False] * len(PROFIT_TARGETS)
                    current_stop_loss = INITIAL_STOP_LOSS
                    highest_price_since_entry = 0.0
        
        equity = max(available_cash + current_shares * price, 1e-6)
        positions[i] = (current_shares * price) / equity if equity > 0 and price > 0 else 0.0

        # Build a compact trade-log entry for this step (best-effort fields)
        try:
            entry_date = dates[i] if (dates is not None and len(dates) > i) else i
        except Exception:
            entry_date = i

        try:
            current_regime_label = REGIME_LABELS.get(int(market_regime[i])) if i < len(market_regime) else None
        except Exception:
            current_regime_label = None

        # Defensive captures for values that may not exist in all branches
        try:
            _kelly = float(kelly_fraction)
        except Exception:
            _kelly = None
        try:
            _target_frac = float(target_fraction)
        except Exception:
            _target_frac = None
        try:
            _drawdown_pct = float(drawdown_adj.get('drawdown_pct', 0.0))
        except Exception:
            _drawdown_pct = None

        # --- Unified confidence computation (optional) ---
        unified_confidence = None
        confidence_tier = None
        confidence_result = None
        try:
            if confidence_scorer is not None:
                hist_vol = None
                try:
                    # estimate historical volatility as std of recent returns
                    if i >= 20:
                        hist_vol = float(np.std(prices[max(0, i-20):i+1]) / (prices[i] if prices[i] != 0 else 1.0))
                    else:
                        hist_vol = float(np.std(prices[:i+1]) / (prices[i] if prices[i] != 0 else 1.0)) if i > 0 else 0.02
                except Exception:
                    hist_vol = 0.02

                q_preds = None
                if quantile_preds is not None:
                    try:
                        q_preds = {
                            'q10': float(quantile_preds[i, 0]) if hasattr(quantile_preds, 'shape') else None,
                            'q50': float(quantile_preds[i, 1]) if hasattr(quantile_preds, 'shape') else None,
                            'q90': float(quantile_preds[i, 2]) if hasattr(quantile_preds, 'shape') else None,
                        }
                    except Exception:
                        q_preds = None

                tft_fc = None
                if tft_forecasts is not None and i < len(tft_forecasts):
                    tft_fc = tft_forecasts[i]

                confidence_inputs = {
                    'signal': signal,
                    'buy_prob': float(buy_probs[i]) if buy_probs is not None and i < len(buy_probs) else None,
                    'sell_prob': float(sell_probs[i]) if sell_probs is not None and i < len(sell_probs) else None,
                    'regressor_pred': float(regressor_positions[i]) if regressor_positions is not None and i < len(regressor_positions) else (float(regressor_preds[i]) if regressor_preds is not None and i < len(regressor_preds) else None),
                    'historical_vol': hist_vol,
                    'quantile_preds': q_preds,
                    'tft_forecasts': tft_fc,
                    'regime': (int(regimes[i]) if (regimes is not None and i < len(regimes)) else None),
                    'model_availability': model_availability or {}
                }

                confidence_result = confidence_scorer.compute_unified_confidence(**confidence_inputs)
                unified_confidence = confidence_result.get('unified_confidence')
                confidence_tier = confidence_scorer.get_confidence_tier(unified_confidence)
        except Exception:
            unified_confidence = None
            confidence_tier = None
            confidence_result = None

        reasoning_parts = []
        if _kelly is not None:
            reasoning_parts.append(f"kelly={_kelly:.3f}")
        try:
            reasoning_parts.append(f"vol_adj={vol_adjust:.2f}")
        except Exception:
            pass
        try:
            reasoning_parts.append(f"drawdown_mult={drawdown_multiplier:.2f}")
        except Exception:
            pass
        if current_regime_label:
            reasoning_parts.append(f"regime={current_regime_label}")

        reasoning_str = ", ".join(reasoning_parts) if reasoning_parts else None

        trade_log_entries.append({
            'index': i,
            'date': entry_date,
            'price': price,
            'action': 'BUY' if signal > 0 else ('SELL' if signal < 0 else 'HOLD'),
            'buy_prob': float(buy_probs[i]) if i < len(buy_probs) else None,
            'sell_prob': float(sell_probs[i]) if i < len(sell_probs) else None,
            'regressor_pred': float(regressor_preds[i]) if (regressor_preds is not None and len(regressor_preds) > i) else None,
            'target_fraction': _target_frac,
            'shares': float(current_shares),
            'portfolio_pct': float(positions[i]) if positions[i] is not None else None,
            'kelly_fraction': _kelly,
            'regime': current_regime_label,
            'atr': float(current_atr) if 'current_atr' in locals() else None,
            'vol_ratio': float(current_vol_ratio) if 'current_vol_ratio' in locals() else None,
            'drawdown_pct': _drawdown_pct,
            'max_position_limit': float(effective_max_position) if 'effective_max_position' in locals() else None,
            'scale_factor': float(drawdown_scale_factor) if 'drawdown_scale_factor' in locals() else None,
            # Unified confidence fields (preferred)
            'unified_confidence': float(unified_confidence) if unified_confidence is not None else (float(buy_probs[i]) if signal == 1 and i < len(buy_probs) else (float(sell_probs[i]) if signal == -1 and i < len(sell_probs) else None)),
            'confidence_tier': confidence_tier,
            'confidence_attribution': confidence_result.get('attribution') if confidence_result is not None else None,
            'classifier_confidence': confidence_result.get('component_confidences', {}).get('classifier') if confidence_result is not None else None,
            'regressor_confidence': confidence_result.get('component_confidences', {}).get('regressor') if confidence_result is not None else None,
            'quantile_confidence': confidence_result.get('component_confidences', {}).get('quantile') if confidence_result is not None else None,
            'tft_confidence': confidence_result.get('component_confidences', {}).get('tft') if confidence_result is not None else None,
            'regime_multiplier': confidence_result.get('regime_multiplier') if confidence_result is not None else None,
            'reasoning': reasoning_str
        })

    # Print volatility metrics summary
    if vol_adjustments_log:
        print(f"\n=== Volatility Adjustments Summary ===")
        print(f"Samples logged: {len(vol_adjustments_log)}")
        
        # Aggregate by regime
        regime_counts = {}
        for log_entry in vol_adjustments_log:
            regime = log_entry['regime']
            regime_counts[regime] = regime_counts.get(regime, 0) + 1
        
        print(f"Volatility regime distribution:")
        for regime, count in sorted(regime_counts.items()):
            pct = (count / len(vol_adjustments_log)) * 100
            print(f"  {regime}: {count} periods ({pct:.1f}%)")
        
        # Show sample adjustments
        print(f"\nSample volatility adjustments (first 5):")
        for log_entry in vol_adjustments_log[:5]:
            print(f"  Day {log_entry['index']}: ATR={log_entry['current_atr']:.4f}, "
                  f"vol_adjust={log_entry['vol_adjust']:.2f}, "
                  f"vol_ratio={log_entry['vol_ratio']:.2f}, "
                  f"regime={log_entry['regime']}, max_pos={log_entry['max_position']:.2f}")
        
        # Summary statistics
        avg_vol_adjust = np.mean([x['vol_adjust'] for x in vol_adjustments_log])
        avg_atr = np.mean([x['current_atr'] for x in vol_adjustments_log])
        avg_vol_ratio = np.mean([x['vol_ratio'] for x in vol_adjustments_log])
        
        print(f"\nAverage metrics:")
        print(f"  ATR: {avg_atr:.4f} ({avg_atr*100:.2f}%)")
        print(f"  Vol adjust factor: {avg_vol_adjust:.2f}")
        print(f"  Vol ratio (5d/20d): {avg_vol_ratio:.2f}")
        print(f"  Baseline ATR: {baseline_atr:.4f} ({baseline_atr*100:.2f}%)")
    
    # Summary header for profit-target system
    print(f"\n=== Profit-Target Order System ===")
    print(f"Configuration: {len(PROFIT_TARGET_LEVELS)} targets at "
          f"{', '.join([f'+{x*100:.0f}%' for x in PROFIT_TARGET_LEVELS])}")
    print(f"Scale-out sizes: {', '.join([f'{x*100:.0f}%' for x in PROFIT_TARGET_SIZES])}")
    print(f"Initial stop-loss: -{INITIAL_STOP_LOSS*100:.0f}%")
    print(f"Trailing stops: " + ", ".join([f"+{p*100:.0f}%→-{s*100:.1f}%" for p, s in TRAILING_STOP_LEVELS]))
    
    # Kelly Criterion and Position Metrics Summary
    print(f"\n=== Kelly Criterion & Position Sizing Summary ===")
    print(f"Total completed trades: {len(trade_history)}")
    
    if trade_history:
        wins = [t['pnl_pct'] for t in trade_history if t['pnl_pct'] > 0]
        losses = [t['pnl_pct'] for t in trade_history if t['pnl_pct'] < 0]
        
        win_rate = (len(wins) / len(trade_history)) * 100 if trade_history else 0
        avg_win = np.mean(wins) * 100 if wins else 0
        avg_loss = np.mean([abs(l) for l in losses]) * 100 if losses else 0
        win_loss_ratio = (avg_win / avg_loss) if avg_loss > 0 else 0
        
        print(f"Win rate: {win_rate:.1f}% ({len(wins)} wins, {len(losses)} losses)")
        print(f"Average win: +{avg_win:.2f}%")
        print(f"Average loss: -{avg_loss:.2f}%")
        print(f"Win/Loss ratio: {win_loss_ratio:.2f}")
    else:
        print("No completed trades (insufficient history)")
    
    print(f"\nPosition adjustments:")
    print(f"  Kelly-adjusted positions: {position_metrics['positions_adjusted_by_kelly']}")
    print(f"  Regime-adjusted positions: {position_metrics['positions_adjusted_by_regime']}")
    print(f"  Max leverage seen: {position_metrics['max_leverage_seen']*100:.1f}%")
    print(f"  Current leverage: {position_metrics['current_leverage']*100:.1f}%")
    
    print(f"\nPosition limits enforced:")
    print(f"  Max single position: {MAX_SINGLE_POSITION*100:.0f}%")
    print(f"  Max total leverage: {MAX_TOTAL_LEVERAGE*100:.0f}%")

    # Build trade-log DataFrame and return it alongside positions
    try:
        trade_log_df = pd.DataFrame(trade_log_entries)
    except Exception:
        trade_log_df = None

    # Run confidence analysis and persist results for inspection
    try:
        # Compute simple returns aligned to indices (next-bar returns approximation)
        try:
            returns_arr = np.concatenate([[np.nan], np.diff(prices) / prices[:-1]]) if len(prices) > 1 else np.array([np.nan] * len(prices))
        except Exception:
            returns_arr = np.array([np.nan] * (len(trade_log_entries) if trade_log_entries else 0))

        # Signals array used earlier
        signals_arr = np.asarray(final_signals, dtype=int) if 'final_signals' in locals() else np.asarray(signals, dtype=int)

        # Prefer unified confidence from trade_log_df when available for analysis.
        try:
            if trade_log_df is not None and 'unified_confidence' in trade_log_df.columns:
                # build arrays where analysis will pick unified confidence regardless of BUY/SELL
                unified_vals = trade_log_df['unified_confidence'].to_numpy(dtype=float)
                # fill NaNs with original buy/sell probs where available
                # Prepare buy_for_analysis and sell_for_analysis so analyze_signal_confidence
                # computes confidences = buy_for_analysis (for BUY) or sell_for_analysis (for SELL).
                buy_for_analysis = np.where(~np.isnan(unified_vals), unified_vals, np.asarray(buy_probs, dtype=float))
                sell_for_analysis = np.where(~np.isnan(unified_vals), unified_vals, np.asarray(sell_probs, dtype=float))
                analysis = analyze_signal_confidence(buy_for_analysis, sell_for_analysis, positions, returns_arr, signals_arr)
            else:
                analysis = analyze_signal_confidence(buy_probs, sell_probs, positions, returns_arr, signals_arr)
        except Exception:
            # Fallback to legacy inputs on any failure
            analysis = analyze_signal_confidence(buy_probs, sell_probs, positions, returns_arr, signals_arr)

        # Prepare output directory
        from datetime import datetime
        if out_dir is None:
            out_dir = Path(__file__).resolve().parent / 'backtest_results'
            out_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')

        # Save analysis JSON
        out_json = out_dir / 'confidence_analysis.json'
        try:
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
            print(f"Saved confidence analysis to: {out_json}")
        except Exception as e:
            print(f"Failed to write confidence analysis JSON: {e}")

        # If trade_log_df exists, append per-row confidence and bucket labels and save CSV
        if trade_log_df is not None and not trade_log_df.empty:
            try:
                # compute per-row confidence used (prefer unified_confidence when present)
                buy_series = trade_log_df.get('buy_prob')
                sell_series = trade_log_df.get('sell_prob')
                unified_series = trade_log_df.get('unified_confidence') if 'unified_confidence' in trade_log_df.columns else None

                def _row_confidence(row):
                    # Use unified if available
                    try:
                        if unified_series is not None and not pd.isna(row.get('unified_confidence')):
                            return float(row.get('unified_confidence'))
                    except Exception:
                        pass
                    if row.get('action') == 'BUY':
                        return float(row.get('buy_prob')) if row.get('buy_prob') is not None and not pd.isna(row.get('buy_prob')) else float('nan')
                    if row.get('action') == 'SELL':
                        return float(row.get('sell_prob')) if row.get('sell_prob') is not None and not pd.isna(row.get('sell_prob')) else float('nan')
                    return float('nan')

                trade_log_df['confidence'] = trade_log_df.apply(_row_confidence, axis=1)

                # map confidence to bucket labels
                cbuckets = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
                clabels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
                conf_vals = trade_log_df['confidence'].to_numpy(dtype=float)
                inds = np.digitize(conf_vals, bins=cbuckets, right=False) - 1
                inds = np.clip(inds, 0, len(clabels) - 1)
                trade_log_df['confidence_bucket'] = [clabels[i] if not np.isnan(v) else None for i, v in zip(inds, conf_vals)]

                out_csv = out_dir / 'confidence_trade_log.csv'
                trade_log_df.to_csv(out_csv, index=False)
                print(f"Saved trade log with confidence buckets to: {out_csv}")
            except Exception as e:
                print(f"Failed to append/save trade_log_df with confidence info: {e}")

        # Run calibration analysis (Brier score + reliability diagram)
        try:
            # Determine the arrays used for analysis (prefer unified where prepared)
            if 'buy_for_analysis' in locals() and 'sell_for_analysis' in locals():
                bfa = buy_for_analysis
                sfa = sell_for_analysis
            else:
                bfa = np.asarray(buy_probs, dtype=float)
                sfa = np.asarray(sell_probs, dtype=float)

            calib = analyze_confidence_calibration(bfa, sfa, signals_arr, returns_arr, symbol=(symbol if 'symbol' in locals() else None), out_dir=str(out_dir), timestamp=ts, plot=('plt' in globals() and plt is not None))
            out_calib = out_dir / 'calibration_analysis.json'
            try:
                with open(out_calib, 'w', encoding='utf-8') as f:
                    json.dump(calib, f, ensure_ascii=False, indent=2)
                print(f"Saved calibration analysis to: {out_calib}")
            except Exception as e:
                print(f"Failed to write calibration analysis JSON: {e}")

            # If the calibration function produced a PNG, print its location
            try:
                plot_path = calib.get('plot_path') if isinstance(calib, dict) else None
                if plot_path:
                    print(f"Saved calibration plot to: {plot_path}")
            except Exception:
                pass
        except Exception as e:
            print(f"Calibration analysis failed: {e}")

        # Optionally save a histogram PNG of confidence deltas if matplotlib available
        try:
            if 'plt' in globals() and plt is not None:
                # Use the absolute confidence delta from analysis if available, else compute
                hist_info = analysis.get('confidence_delta_hist') if isinstance(analysis, dict) else None
                if hist_info and hist_info.get('counts'):
                    bins = hist_info.get('bins')
                    counts = hist_info.get('counts')
                    fig, ax = plt.subplots()
                    # Plot histogram using bins and counts
                    ax.bar([(bins[i] + bins[i+1]) / 2.0 for i in range(len(counts))], counts, width=(bins[1]-bins[0]) * 0.9)
                    ax.set_xlabel('Confidence Delta (abs)')
                    ax.set_ylabel('Count')
                    ax.set_title('Histogram of Confidence Delta for Executed Trades')
                    png_path = out_dir / f'confidence_delta_hist_{ts}.png'
                    fig.tight_layout()
                    fig.savefig(png_path)
                    plt.close(fig)
                    print(f"Saved confidence delta histogram to: {png_path}")
        except Exception:
            pass

    except Exception as e:
        print(f"Confidence analysis failed: {e}")

    return positions, trade_log_df
# -------------------------
# Confidence analysis utilities

def analyze_signal_confidence(buy_probs, sell_probs, positions, returns, final_signals):
    """Analyze signal confidence vs outcomes.

    Args:
        buy_probs: array-like of buy probabilities (same length as signals)
        sell_probs: array-like of sell probabilities
        positions: array-like of position sizes (may be unused but kept for compatibility)
        returns: array-like of realized returns for each trade/date (aligned)
        final_signals: array-like with values in {1, -1, 0} representing executed signals

    Returns:
        dict with keys:
          - confidence_buckets: mapping bucket_name -> {'range': (low, high), 'count': int}
          - win_rates: mapping bucket_name -> win rate (float or None)
          - avg_returns: mapping bucket_name -> average return (float or None)
          - kelly_distribution: dict with overall stats and per-bucket lists
          - confidence_delta_hist: dict with 'bins' and 'counts'
          - confidence_return_correlation: Pearson r between confidence and returns (float or None)

    Notes:
        Uses `calculate_kelly_fraction` with a default win/loss ratio when computing per-trade Kelly estimates.
    """
    import numpy as _np

    buy_probs = _np.asarray(buy_probs, dtype=float)
    sell_probs = _np.asarray(sell_probs, dtype=float)
    positions = _np.asarray(positions, dtype=float) if positions is not None else None
    returns = _np.asarray(returns, dtype=float)
    final_signals = _np.asarray(final_signals, dtype=int)

    executed_mask = final_signals != 0
    if executed_mask.sum() == 0:
        return {
            'confidence_buckets': {},
            'win_rates': {},
            'avg_returns': {},
            'kelly_distribution': {'overall': {}, 'per_bucket': {}},
            'confidence_delta_hist': {'bins': [], 'counts': []},
            'confidence_return_correlation': None,
        }

    confidences = _np.where(final_signals == 1, buy_probs, _np.where(final_signals == -1, sell_probs, _np.nan))
    confidences = confidences[executed_mask]

    signed_delta = (buy_probs - sell_probs) * _np.sign(final_signals)
    signed_delta = signed_delta[executed_mask]
    abs_delta = _np.abs(buy_probs - sell_probs)[executed_mask]

    returns_exec = returns[executed_mask]

    buckets = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
    labels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
    inds = _np.digitize(confidences, bins=buckets, right=False) - 1
    inds = _np.clip(inds, 0, len(labels) - 1)

    confidence_buckets = {}
    win_rates = {}
    avg_returns = {}
    per_bucket_kellys = {}

    try:
        kellys_all = [_np.nan if _np.isnan(c) else calculate_kelly_fraction(float(c), DEFAULT_WIN_LOSS_RATIO) for c in confidences]
    except Exception:
        kellys_all = [_np.nan for _ in confidences]

    for i, label in enumerate(labels):
        low = buckets[i]
        high = buckets[i + 1]
        mask = inds == i
        count = int(mask.sum())
        confidence_buckets[label] = {'range': (float(low), float(high)), 'count': count}

        if count == 0:
            win_rates[label] = None
            avg_returns[label] = None
            per_bucket_kellys[label] = []
            continue

        wins = (returns_exec[mask] > 0).sum()
        win_rates[label] = float(wins) / float(count)
        avg_returns[label] = float(_np.nanmean(returns_exec[mask]))
        per_bucket_kellys[label] = [_float for _float in _np.asarray(kellys_all)[mask].tolist() if not _np.isnan(_float)]

    kellys_clean = [k for k in kellys_all if not _np.isnan(k)]
    kelly_summary = {}
    if kellys_clean:
        kelly_summary['mean'] = float(_np.mean(kellys_clean))
        kelly_summary['median'] = float(_np.median(kellys_clean))
        kelly_summary['std'] = float(_np.std(kellys_clean))
    else:
        kelly_summary['mean'] = kelly_summary['median'] = kelly_summary['std'] = None

    try:
        hist_counts, hist_bins = _np.histogram(abs_delta, bins=20, range=(0.0, 1.0))
        hist = {'bins': hist_bins.tolist(), 'counts': hist_counts.tolist()}
    except Exception:
        hist = {'bins': [], 'counts': []}

    corr = None
    try:
        if len(confidences) >= 2 and not _np.all(_np.isnan(confidences)):
            valid_mask = ~_np.isnan(confidences) & ~_np.isnan(returns_exec)
            if valid_mask.sum() >= 2:
                corr = float(_np.corrcoef(confidences[valid_mask], returns_exec[valid_mask])[0, 1])
    except Exception:
        corr = None

    return {
        'confidence_buckets': confidence_buckets,
        'win_rates': win_rates,
        'avg_returns': avg_returns,
        'kelly_distribution': {'overall': kelly_summary, 'per_bucket': per_bucket_kellys},
        'confidence_delta_hist': hist,
        'confidence_return_correlation': corr,
    }


def analyze_confidence_calibration(buy_probs, sell_probs, final_signals, returns, symbol: str | None = None, out_dir: str | None = None, timestamp: str | None = None, plot: bool = True):
    """Evaluate calibration of BUY/SELL classifier confidence scores.

    - Bins BUY/SELL probabilities into [0.3-0.4,0.4-0.5,...,0.9-1.0]
    - For each bin calculates expected (mean predicted prob), actual (empirical win rate), and absolute calibration error
    - Computes Brier score across executed signals
    - Produces a reliability diagram saved to `backtest_results/{symbol}_calibration_curve_{ts}.png` when plotting enabled

    Args:
        buy_probs, sell_probs: array-like of probabilities
        final_signals: array-like with {1,-1,0}
        returns: array-like of realized returns (aligned)
        symbol: optional symbol used for naming output file
        out_dir: optional output directory (defaults to package/backtest_results)
        timestamp: optional timestamp string used in filename
        plot: bool whether to attempt plotting (guarded if matplotlib missing)

    Returns:
        dict with keys: 'brier_score', 'calibration_error_per_bin' (dict with 'buy'/'sell' per-bin info),
                      'calibration_points' and 'plot_path' when available
    """
    import numpy as _np
    from datetime import datetime
    from pathlib import Path

    buy_probs = _np.asarray(buy_probs, dtype=float)
    sell_probs = _np.asarray(sell_probs, dtype=float)
    final_signals = _np.asarray(final_signals, dtype=int)
    returns = _np.asarray(returns, dtype=float)

    executed_mask = final_signals != 0
    # Prepare output directory and timestamp early so even empty results get persisted
    out_base = Path(__file__).resolve().parent
    results_dir = Path(out_dir) if out_dir is not None else out_base / 'backtest_results'
    results_dir = Path(results_dir)
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    ts = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')

    if executed_mask.sum() == 0:
        result_empty = {'brier_score': None, 'calibration_error_per_bin': {}, 'calibration_points': {}, 'plot_path': None}
        # attempt to save an empty calibration JSON so main runs produce the artifact
        try:
            import json as _json
            calib_name = results_dir / 'calibration_analysis.json'
            with open(calib_name, 'w', encoding='utf-8') as _f:
                _json.dump(result_empty, _f, ensure_ascii=False, indent=2)
            try:
                print(f"Saved calibration analysis to: {calib_name}")
            except Exception:
                pass
        except Exception as e:
            try:
                print(f"Failed to write empty calibration JSON: {e}")
            except Exception:
                pass
        return result_empty

    # Brier score: consider only executed signals; for BUY outcome is (return>0), for SELL outcome is (return<0)
    probs_list = []
    outcomes = []
    for p_buy, p_sell, sig, r in zip(buy_probs, sell_probs, final_signals, returns):
        if sig == 1:
            if _np.isnan(p_buy):
                continue
            probs_list.append(float(p_buy))
            outcomes.append(1.0 if (r is not None and not _np.isnan(r) and r > 0.0) else 0.0)
        elif sig == -1:
            if _np.isnan(p_sell):
                continue
            probs_list.append(float(p_sell))
            outcomes.append(1.0 if (r is not None and not _np.isnan(r) and r < 0.0) else 0.0)

    probs_arr = _np.asarray(probs_list, dtype=float)
    outcomes_arr = _np.asarray(outcomes, dtype=float)
    if probs_arr.size == 0:
        brier = None
    else:
        brier = float(_np.mean((probs_arr - outcomes_arr) ** 2))

    # Define bins for calibration (as requested: 0.3-0.4, 0.4-0.5, ..., 0.9-1.0)
    bin_edges = _np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    bin_labels = [f"{bin_edges[i]:.1f}-{bin_edges[i+1]:.1f}" for i in range(len(bin_edges)-1)]

    def _compute_bins(side_mask, probs_side, outcome_mask):
        results = {}
        if probs_side.size == 0:
            for lab in bin_labels:
                results[lab] = {'expected': None, 'actual': None, 'error': None, 'count': 0, 'ci_lower': None, 'ci_upper': None}
            return results

        inds = _np.digitize(probs_side, bins=bin_edges, right=False) - 1
        inds = _np.clip(inds, 0, len(bin_labels)-1)

        z = 1.96
        for i, lab in enumerate(bin_labels):
            mask = inds == i
            count = int(mask.sum())
            if count == 0:
                results[lab] = {'expected': None, 'actual': None, 'error': None, 'count': 0, 'ci_lower': None, 'ci_upper': None}
                continue
            expected = float(_np.nanmean(probs_side[mask]))
            # actual: fraction of positive outcomes (for BUY: r>0, for SELL: r<0)
            actual = float(_np.nanmean(outcome_mask[mask]))
            error = float(_np.abs(expected - actual))

            # Wilson score interval for proportion
            phat = actual
            n = count
            denom = 1 + (z**2)/n
            center = (phat + (z**2)/(2*n)) / denom
            half = (z * _np.sqrt((phat*(1-phat))/n + (z**2)/(4*n*n))) / denom
            ci_low = max(0.0, center - half)
            ci_high = min(1.0, center + half)

            results[lab] = {'expected': expected, 'actual': actual, 'error': error, 'count': count, 'ci_lower': ci_low, 'ci_upper': ci_high}
        return results

    # BUY calibration
    buy_mask = final_signals == 1
    buy_probs_side = buy_probs[buy_mask]
    buy_returns_outcome = _np.array([(1.0 if (r is not None and not _np.isnan(r) and r > 0.0) else 0.0) for r in returns[buy_mask]])
    buy_bins = _compute_bins(buy_mask, buy_probs_side, buy_returns_outcome)

    # SELL calibration (positive outcome defined as return < 0)
    sell_mask = final_signals == -1
    sell_probs_side = sell_probs[sell_mask]
    sell_returns_outcome = _np.array([(1.0 if (r is not None and not _np.isnan(r) and r < 0.0) else 0.0) for r in returns[sell_mask]])
    sell_bins = _compute_bins(sell_mask, sell_probs_side, sell_returns_outcome)

    # Prepare calibration points for plotting (use bin centers where counts>0)
    calibration_points = {'buy': [], 'sell': []}
    for lab in bin_labels:
        b = buy_bins.get(lab)
        if b and b['count'] > 0:
            # predicted x: expected
            calibration_points['buy'].append((b['expected'], b['actual'], b['ci_lower'], b['ci_upper'], b['count']))
        s = sell_bins.get(lab)
        if s and s['count'] > 0:
            calibration_points['sell'].append((s['expected'], s['actual'], s['ci_lower'], s['ci_upper'], s['count']))

    # Prepare output directory and timestamp
    out_base = Path(__file__).resolve().parent
    results_dir = Path(out_dir) if out_dir is not None else out_base / 'backtest_results'
    try:
        results_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        # best-effort: ignore if cannot create
        pass
    ts = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')

    # Plot reliability diagram if requested
    plot_path = None
    if plot:
        try:
            # lazy import matplotlib
            import matplotlib.pyplot as _plt

            name = 'calibration_curve.png'
            plot_path = results_dir / name

            fig, ax = _plt.subplots(figsize=(7, 6))
            # perfect calibration line
            ax.plot([0, 1], [0, 1], color='gray', linestyle='--', label='Perfect calibration')

            # plot buy points
            if calibration_points['buy']:
                xs = [_p for (_p, _a, _l, _u, _c) in calibration_points['buy']]
                ys = [_a for (_p, _a, _l, _u, _c) in calibration_points['buy']]
                ylow = [_l for (_p, _a, _l, _u, _c) in calibration_points['buy']]
                yhigh = [_u for (_p, _a, _l, _u, _c) in calibration_points['buy']]
                ax.errorbar(xs, ys, yerr=[_np.array(ys)-_np.array(ylow), _np.array(yhigh)-_np.array(ys)], fmt='o', color='green', label='BUY', capsize=3)

            # plot sell points
            if calibration_points['sell']:
                xs = [_p for (_p, _a, _l, _u, _c) in calibration_points['sell']]
                ys = [_a for (_p, _a, _l, _u, _c) in calibration_points['sell']]
                ylow = [_l for (_p, _a, _l, _u, _c) in calibration_points['sell']]
                yhigh = [_u for (_p, _a, _l, _u, _c) in calibration_points['sell']]
                ax.errorbar(xs, ys, yerr=[_np.array(ys)-_np.array(ylow), _np.array(yhigh)-_np.array(ys)], fmt='s', color='red', label='SELL', capsize=3)

            ax.set_xlim(0.25, 1.01)
            ax.set_ylim(-0.05, 1.05)
            ax.set_xlabel('Predicted probability')
            ax.set_ylabel('Observed frequency')
            ax.set_title(f'Calibration Curve ({symbol or "unknown"})')
            ax.legend()
            fig.tight_layout()
            fig.savefig(plot_path)
            _plt.close(fig)
        except Exception as e:
            # Surface plotting error for visibility
            try:
                print(f"Calibration plot generation failed: {e}")
            except Exception:
                pass
            plot_path = None

    # Build result dict
    result = {
        'brier_score': brier,
        'calibration_error_per_bin': {'buy': buy_bins, 'sell': sell_bins},
        'calibration_points': calibration_points,
        'plot_path': str(plot_path) if plot_path is not None else None
    }

    # Attempt to save a JSON copy of the calibration results for downstream inspection
    try:
        import json as _json
        calib_name = results_dir / 'calibration_analysis.json'
        with open(calib_name, 'w', encoding='utf-8') as _f:
            _json.dump(result, _f, ensure_ascii=False, indent=2)
        try:
            print(f"Saved calibration analysis to: {calib_name}")
        except Exception:
            pass
    except Exception as e:
        try:
            print(f"Failed to write calibration JSON: {e}")
        except Exception:
            pass

    return result


def analyze_adaptive_exits(trade_log, profit_targets_hit_history, symbol: str | None = None, out_dir=None):
    """Analyze adaptive ATR-based exits and produce a text report.

    Args:
        trade_log: pandas.DataFrame or list of per-step trade log dicts (as produced by compute_hybrid_positions)
        profit_targets_hit_history: list of per-trade dicts describing completed trades. Each dict should
            preferably contain keys: 'entry_price','exit_price','targets_hit' (list of 4 bools),
            'days_held', 'time_degraded' (bool), 'resistance_exit' (bool), 'pnl_pct' (decimal),
            and optionally 'price_path' (list of daily prices) and 'target_prices' (list of target prices).
        out_dir: Output directory for saving the report (optional)

    Returns:
        report_path: Path to the saved text report

    Notes:
        This function is defensive: if detailed fields are missing it will attempt a best-effort
        reconstruction from `trade_log` but will not raise errors; instead it writes a report noting
        missing pieces.
    """
    from datetime import datetime
    if out_dir is None:
        out_dir = Path(__file__).resolve().parent / 'backtest_results'
    else:
        out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Normalize trade_log to DataFrame if possible
    if isinstance(trade_log, list):
        try:
            trade_log_df = pd.DataFrame(trade_log)
        except Exception:
            trade_log_df = None
    elif isinstance(trade_log, pd.DataFrame):
        trade_log_df = trade_log.copy()
    else:
        trade_log_df = None

    # Ensure profit_targets_hit_history is a list
    trades = profit_targets_hit_history or []

    # Attempt reconstruction if no detailed history provided
    if (not trades) and trade_log_df is not None and not trade_log_df.empty:
        # FIFO lot-based reconstruction: create lots for incremental BUYs (shares increases)
        # and consume lots on decreases (partial/full exits). This handles multiple adds/exits.
        trades = []
        lots: list[dict] = []  # each lot: {'qty', 'entry_price','entry_index','entry_date'}
        prev_shares = 0.0

        for idx, row in trade_log_df.iterrows():
            # Try multiple possible column names for shares
            shares = None
            for k in ('shares', 'current_shares', 'position_shares'):
                if k in row.index:
                    shares = row.get(k)
                    break
            try:
                shares = float(shares) if shares is not None and not pd.isna(shares) else 0.0
            except Exception:
                shares = 0.0

            price = row.get('price') if 'price' in row.index else None
            date = row.get('date') if 'date' in row.index else None

            delta = shares - (prev_shares if prev_shares is not None else 0.0)

            # New BUY / add-to-position -> create a new lot for the incremental quantity
            if delta > 0:
                lot = {
                    'qty': float(delta),
                    'entry_price': float(price) if price is not None and not pd.isna(price) else None,
                    'entry_index': int(idx),
                    'entry_date': date
                }
                lots.append(lot)

            # SELL or reduction -> consume lots FIFO and emit exit rows per consumed lot portion
            elif delta < 0:
                qty_to_close = float(-delta)
                exit_price = float(price) if price is not None and not pd.isna(price) else None
                while qty_to_close > 0 and lots:
                    lot = lots[0]
                    take = min(lot['qty'], qty_to_close)
                    days_held = int(idx - lot.get('entry_index', idx)) if lot.get('entry_index') is not None else None
                    pnl = None
                    if lot.get('entry_price') is not None and exit_price is not None and lot.get('entry_price') != 0:
                        pnl = (exit_price / lot.get('entry_price') - 1.0)

                    exit_row = {
                        'entry_price': lot.get('entry_price'),
                        'exit_price': exit_price,
                        'entry_index': lot.get('entry_index'),
                        'exit_index': int(idx),
                        'days_held': days_held,
                        'targets_hit': [False, False, False, False],
                        'time_degraded': False,
                        'resistance_exit': False,
                        'pnl_pct': pnl,
                        'price_path': [lot.get('entry_price'), exit_price] if lot.get('entry_price') is not None and exit_price is not None else None
                    }
                    trades.append(exit_row)

                    # decrement lot quantity
                    lot['qty'] = float(lot['qty'] - take)
                    qty_to_close = float(qty_to_close - take)
                    if lot['qty'] <= 0:
                        lots.pop(0)

            prev_shares = shares

        # Any remaining open lots at the end of the log -> emit open_at_end rows using final price
        if lots:
            try:
                last_idx = int(trade_log_df.index[-1]) if len(trade_log_df.index) > 0 else 0
            except Exception:
                last_idx = 0
            last_row = trade_log_df.iloc[-1]
            last_price = last_row.get('price') if 'price' in last_row.index else None
            exit_price = float(last_price) if last_price is not None and not pd.isna(last_price) else None
            for lot in lots:
                days_held = int(last_idx - lot.get('entry_index', last_idx)) if lot.get('entry_index') is not None else None
                pnl = None
                if lot.get('entry_price') is not None and exit_price is not None and lot.get('entry_price') != 0:
                    pnl = (exit_price / lot.get('entry_price') - 1.0)

                exit_row = {
                    'entry_price': lot.get('entry_price'),
                    'exit_price': exit_price,
                    'entry_index': lot.get('entry_index'),
                    'exit_index': last_idx,
                    'days_held': days_held,
                    'targets_hit': [False, False, False, False],
                    'time_degraded': False,
                    'resistance_exit': False,
                    'pnl_pct': pnl,
                    'price_path': [lot.get('entry_price'), exit_price] if lot.get('entry_price') is not None and exit_price is not None else None,
                    'open_at_end': True
                }
                trades.append(exit_row)

        # Fallback: when shares column is missing, do an action-based FIFO lot reconstruction
        if not trades:
            lots_action: list[dict] = []
            for idx, row in trade_log_df.iterrows():
                action = (row.get('action') or '').upper() if 'action' in row.index else ''
                price = row.get('price')
                date = row.get('date')

                if action == 'BUY':
                    lots_action.append({'entry_price': float(price) if pd.notna(price) else None, 'entry_index': int(idx), 'entry_date': date})
                elif action == 'SELL':
                    # consume FIFO lot
                    if lots_action:
                        lot = lots_action.pop(0)
                        exit_price = float(price) if pd.notna(price) else None
                        days_held = int(idx - lot.get('entry_index', idx)) if lot.get('entry_index') is not None else None
                        pnl = None
                        if lot.get('entry_price') is not None and exit_price is not None and lot.get('entry_price') != 0:
                            pnl = (exit_price / lot.get('entry_price') - 1.0)
                        exit_row = {
                            'entry_price': lot.get('entry_price'),
                            'exit_price': exit_price,
                            'entry_index': lot.get('entry_index'),
                            'exit_index': int(idx),
                            'days_held': days_held,
                            'targets_hit': [False, False, False, False],
                            'time_degraded': False,
                            'resistance_exit': False,
                            'pnl_pct': pnl,
                            'price_path': [lot.get('entry_price'), exit_price] if lot.get('entry_price') is not None and exit_price is not None else None
                        }
                        trades.append(exit_row)

            # Any remaining BUY lots are open at end
            if lots_action:
                try:
                    last_idx = int(trade_log_df.index[-1]) if len(trade_log_df.index) > 0 else 0
                except Exception:
                    last_idx = 0
                last_price = trade_log_df.iloc[-1].get('price') if 'price' in trade_log_df.columns else None
                exit_price = float(last_price) if last_price is not None and not pd.isna(last_price) else None
                for lot in lots_action:
                    days_held = int(last_idx - lot.get('entry_index', last_idx)) if lot.get('entry_index') is not None else None
                    pnl = None
                    if lot.get('entry_price') is not None and exit_price is not None and lot.get('entry_price') != 0:
                        pnl = (exit_price / lot.get('entry_price') - 1.0)
                    exit_row = {
                        'entry_price': lot.get('entry_price'),
                        'exit_price': exit_price,
                        'entry_index': lot.get('entry_index'),
                        'exit_index': last_idx,
                        'days_held': days_held,
                        'targets_hit': [False, False, False, False],
                        'time_degraded': False,
                        'resistance_exit': False,
                        'pnl_pct': pnl,
                        'price_path': [lot.get('entry_price'), exit_price] if lot.get('entry_price') is not None and exit_price is not None else None,
                        'open_at_end': True
                    }
                    trades.append(exit_row)

    # Prepare aggregates
    total_trades = len(trades)
    target_counts = [0, 0, 0, 0]
    exit_level_returns = {1: [], 2: [], 3: [], 4: []}
    days_to_target = {1: [], 2: [], 3: [], 4: []}
    resistance_returns = {'resisted': [], 'not_resisted': []}
    missing_price_paths = 0

    # Expected mapping of ATR multipliers to target labels (informational)
    target_multipliers = [2.0, 3.5, 5.0, 8.0]

    for tr in trades:
        # Normalize fields
        entry = tr.get('entry_price')
        exitp = tr.get('exit_price')
        pnl = tr.get('pnl_pct')
        days = tr.get('days_held')
        tdegraded = bool(tr.get('time_degraded')) if tr.get('time_degraded') is not None else False
        tres = bool(tr.get('resistance_exit')) if tr.get('resistance_exit') is not None else False

        targets = tr.get('targets_hit') if isinstance(tr.get('targets_hit'), (list, tuple)) else [False, False, False, False]

        # Determine exit level: highest target hit index (1-based), else 0
        exit_level = 0
        for idx_t in range(3, -1, -1):
            if idx_t < len(targets) and targets[idx_t]:
                exit_level = idx_t + 1
                break

        if exit_level > 0:
            target_counts[exit_level - 1] += 1
            if pnl is not None:
                exit_level_returns[exit_level].append(float(pnl))

        # Resistance returns
        if tres:
            if pnl is not None:
                resistance_returns['resisted'].append(float(pnl))
        else:
            if pnl is not None:
                resistance_returns['not_resisted'].append(float(pnl))

        # Time to target analysis requires price_path and target_prices
        price_path = tr.get('price_path')
        target_prices = tr.get('target_prices')
        if price_path and target_prices and isinstance(price_path, (list, tuple)) and isinstance(target_prices, (list, tuple)):
            # For each target, find first index where price >= target_price
            for t_idx, t_price in enumerate(target_prices):
                found_idx = None
                for day_idx, p in enumerate(price_path):
                    try:
                        if p is not None and t_price is not None and p >= t_price:
                            found_idx = day_idx
                            break
                    except Exception:
                        continue
                if found_idx is not None:
                    days_to_target[t_idx + 1].append(found_idx)
        else:
            missing_price_paths += 1

    # Compute averages
    avg_returns_per_level = {}
    for level in [1, 2, 3, 4]:
        vals = exit_level_returns[level]
        avg_returns_per_level[level] = (float(np.mean(vals)) if vals else None)

    avg_days_to_target = {lvl: (float(np.mean(days_to_target[lvl])) if days_to_target[lvl] else None) for lvl in [1, 2, 3, 4]}

    # Resistance effectiveness
    resisted = resistance_returns['resisted']
    not_resisted = resistance_returns['not_resisted']
    resist_effect = None
    if resisted and not_resisted:
        resist_effect = float(np.mean(resisted)) - float(np.mean(not_resisted))

    # Alternative exits simulation - only possible when price_path present
    alt_summary = {'fixed_5pct': [], 'hold_t4': [], 'exit_first_target': [], 'skipped_due_to_missing_price_path': 0}
    for tr in trades:
        price_path = tr.get('price_path')
        entry = tr.get('entry_price')
        exitp = tr.get('exit_price')
        if not price_path or entry is None:
            alt_summary['skipped_due_to_missing_price_path'] += 1
            continue

        # Fixed 5% target
        fixed_target = entry * 1.05
        fixed_ret = None
        for p in price_path:
            if p is not None and p >= fixed_target:
                fixed_ret = (p / entry - 1.0)
                break
        if fixed_ret is None:
            # If never hit, use realized exit
            fixed_ret = (exitp / entry - 1.0) if exitp and entry else None
        alt_summary['fixed_5pct'].append(fixed_ret)

        # Hold to T4: need target_prices or assume multiplier of entry via target_multipliers
        t_prices = tr.get('target_prices')
        if t_prices and len(t_prices) >= 4:
            t4_price = t_prices[3]
        else:
            # fallback: approximate using multipliers if ATR-based targets unknown
            t4_price = None
        hold_t4_ret = None
        if t4_price is not None:
            for p in price_path:
                if p is not None and p >= t4_price:
                    hold_t4_ret = (p / entry - 1.0)
                    break
            if hold_t4_ret is None:
                # never hit T4, use realized exit
                hold_t4_ret = (exitp / entry - 1.0) if exitp and entry else None
        else:
            hold_t4_ret = None
        alt_summary['hold_t4'].append(hold_t4_ret)

        # Exit at first target: find earliest target price hit among target_prices
        first_target_ret = None
        if t_prices:
            first_hit = None
            for p in price_path:
                for t_price in t_prices:
                    if p is not None and t_price is not None and p >= t_price:
                        first_hit = p
                        break
                if first_hit is not None:
                    break
            if first_hit is not None:
                first_target_ret = (first_hit / entry - 1.0)
            else:
                first_target_ret = (exitp / entry - 1.0) if exitp and entry else None
        else:
            first_target_ret = None
        alt_summary['exit_first_target'].append(first_target_ret)

    # Build textual report
    # Prefer explicit symbol parameter when provided; otherwise try to infer from trade_log
    if not symbol:
        symbol = None
        if trade_log_df is not None and 'symbol' in trade_log_df.columns:
            try:
                symbol = str(trade_log_df['symbol'].iloc[0])
            except Exception:
                symbol = None

    report_name = 'exit_analysis.txt'
    report_path = out_dir / report_name

    # Generate timestamp for report
    from datetime import datetime
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"Exit Analysis Report - {symbol or 'unknown'}\nGenerated: {ts}\n\n")
        f.write(f"Total trades analyzed: {total_trades}\n")
        
        # Add diagnostic message if no trades were found
        if total_trades == 0:
            f.write("\n⚠️ NO COMPLETED TRADES FOUND\n")
            f.write("This typically means:\n")
            f.write("  1. Only BUY signals fired (no SELL signals to close positions)\n")
            f.write("  2. SELL threshold may be too high relative to SELL classifier output\n")
            f.write("  3. Check SELL probability distribution - may need threshold adjustment\n")
            f.write("\nRecommendation: Review classifier thresholds and SELL probability stats.\n\n")
        f.write("\nProfit targets hit counts:\n")
        for i, cnt in enumerate(target_counts, start=1):
            f.write(f"  T{i} (mult {target_multipliers[i-1]}xATR): {cnt}\n")

        f.write("\nAverage return when exiting at target level:\n")
        for lvl in [1, 2, 3, 4]:
            val = avg_returns_per_level.get(lvl)
            f.write(f"  T{lvl}: {val*100:.2f}%\n" if val is not None else f"  T{lvl}: N/A\n")

        f.write("\nTime-to-target averages (days):\n")
        for lvl in [1, 2, 3, 4]:
            v = avg_days_to_target.get(lvl)
            f.write(f"  T{lvl}: {v:.2f}\n" if v is not None else f"  T{lvl}: N/A\n")

        f.write("\nResistance-aware exits:\n")
        f.write(f"  Trades using resistance-aware early exit: {len(resisted) if resisted else 0}\n")
        if resist_effect is not None:
            f.write(f"  Avg P&L difference (resisted - not_resisted): {resist_effect*100:.2f}%\n")
        else:
            f.write("  Insufficient data to evaluate resistance effectiveness.\n")

        f.write("\nAlternative exit strategy summary (only computed where price_path available):\n")
        skipped = alt_summary.get('skipped_due_to_missing_price_path', 0)
        f.write(f"  Trades skipped due to missing price_path: {skipped}\n")
        def _mean_or_na(lst):
            lst_clean = [x for x in lst if x is not None]
            return float(np.mean(lst_clean)) if lst_clean else None

        for k in ['fixed_5pct', 'hold_t4', 'exit_first_target']:
            m = _mean_or_na(alt_summary.get(k, []))
            f.write(f"  {k}: avg return = {m*100:.2f}%\n" if m is not None else f"  {k}: N/A (insufficient data)\n")

        f.write("\nNotes:\n")
        f.write(" - This analysis requires per-trade 'price_path' and/or 'target_prices' for precise time-to-target and alternative-exit simulation.\n")
        f.write(" - If profit_targets_hit_history is not provided, the function attempts a best-effort reconstruction from trade_log but detailed flags (time_degraded, resistance) may be unavailable.\n")

    print(f"Saved exit analysis report to: {report_path}")
    return report_path


def analyze_regime_performance(dates, prices, returns, positions, regimes, symbol: str | None = None, window: int = 5):
    """Analyze performance broken down by detected market regimes.

    Args:
        dates: array-like of dates (aligned to returns/positions)
        prices: array-like of prices
        returns: array-like of per-period returns (decimal)
        positions: array-like of strategy positions (fractional exposure)
        regimes: array-like of regime labels (-1, 0, 1)
        symbol: optional symbol string for reporting
        window: integer days used for transition before/after analysis

    Returns:
        dict containing `regime_metrics` and `transition_analysis`.
    """
    # Normalize to numpy
    dates = np.asarray(dates)
    prices = np.asarray(prices)
    returns = np.asarray(returns, dtype=float)
    positions = np.asarray(positions, dtype=float)
    regimes = np.asarray(regimes)

    regime_map = {1: 'BULLISH', 0: 'SIDEWAYS', -1: 'BEARISH'}
    per_day_strategy_returns = positions * returns

    regime_metrics = {}

    # Simple trade reconstruction (entry when pos>0 and prev<=0, exit when pos<=0 and prev>0)
    prev_pos = 0.0
    entries = []
    exits = []
    for i, pos in enumerate(positions):
        if pos > 0 and (i == 0 or prev_pos <= 0):
            entries.append(i)
        if pos <= 0 and prev_pos > 0:
            exits.append(i)
        prev_pos = pos
    # Pair entries->exits (ignore unmatched trailing entry)
    paired_trades = []
    for ei, entry_idx in enumerate(entries):
        exit_idx = exits[ei] if ei < len(exits) else None
        paired_trades.append((entry_idx, exit_idx))

    for val, label in regime_map.items():
        idx = np.where(regimes == val)[0]
        if idx.size == 0:
            regime_metrics[label] = None
            continue

        # Aggregate daily metrics
        seg_ret = per_day_strategy_returns[idx]
        total_return = float(np.prod(1.0 + seg_ret) - 1.0) if seg_ret.size > 0 else None
        avg_daily = float(np.nanmean(seg_ret)) if seg_ret.size > 0 else None
        std_daily = float(np.nanstd(seg_ret)) if seg_ret.size > 0 else None
        sharpe = (avg_daily / std_daily) * np.sqrt(252) if std_daily and std_daily > 0 else None

        # Max drawdown computed on equity built from seg_ret
        try:
            equity = np.cumprod(1.0 + seg_ret)
            roll_max = np.maximum.accumulate(equity)
            drawdowns = (equity - roll_max) / roll_max
            max_dd = float(np.min(drawdowns)) if drawdowns.size > 0 else None
        except Exception:
            max_dd = None

        # Trade-level metrics: select trades whose entry index falls inside regime segment
        trades_in_regime = []
        for entry_idx, exit_idx in paired_trades:
            if entry_idx is None:
                continue
            if entry_idx in idx:
                # compute trade return from entry to exit (or to end)
                start = entry_idx
                end = exit_idx if exit_idx is not None else (len(returns) - 1)
                try:
                    tr_ret = float(np.prod(1.0 + returns[start:end]) - 1.0) if end > start else 0.0
                except Exception:
                    tr_ret = None
                trades_in_regime.append(tr_ret)

        num_trades = len(trades_in_regime)
        avg_trade_return = float(np.nanmean([t for t in trades_in_regime if t is not None])) if trades_in_regime else None
        win_rate = None
        if trades_in_regime:
            wins = [1 for t in trades_in_regime if t is not None and t > 0]
            win_rate = float(len(wins) / len(trades_in_regime))

        avg_position_size = float(np.nanmean(positions[idx])) if idx.size > 0 else None
        # Kelly fraction distribution approximated from position sizes
        kelly_vals = positions[idx]
        kelly_stats = None
        if kelly_vals.size > 0:
            kelly_stats = {
                'count': int(kelly_vals.size),
                'mean': float(np.nanmean(kelly_vals)),
                'median': float(np.nanmedian(kelly_vals)),
                'std': float(np.nanstd(kelly_vals)),
                'p10': float(np.nanpercentile(kelly_vals, 10)),
                'p90': float(np.nanpercentile(kelly_vals, 90))
            }

        regime_metrics[label] = {
            'total_return': total_return,
            'sharpe': sharpe,
            'max_drawdown': max_dd,
            'win_rate': win_rate,
            'avg_trade_return': avg_trade_return,
            'num_trades': num_trades,
            'avg_position_size': avg_position_size,
            'kelly_stats': kelly_stats
        }

    # Transition analysis: examine performance before/after regime changes
    transitions = []
    for t in range(1, len(regimes)):
        if regimes[t] != regimes[t - 1]:
            frm = regime_map.get(int(regimes[t - 1]), str(regimes[t - 1]))
            to = regime_map.get(int(regimes[t]), str(regimes[t]))
            start_before = max(0, t - window)
            end_before = t
            start_after = t
            end_after = min(len(returns), t + window)
            before_ret = float(np.prod(1.0 + per_day_strategy_returns[start_before:end_before]) - 1.0) if end_before > start_before else None
            after_ret = float(np.prod(1.0 + per_day_strategy_returns[start_after:end_after]) - 1.0) if end_after > start_after else None

            # detection lag: days until positions change materially after regime change
            lag = None
            threshold = 0.05  # 5% absolute position change threshold
            for dt in range(0, min(window, len(positions) - t)):
                if abs(positions[t + dt] - positions[t - 1]) >= threshold:
                    lag = int(dt)
                    break

            transitions.append({
                'index': int(t),
                'from': frm,
                'to': to,
                'before_return': before_ret,
                'after_return': after_ret,
                'detection_lag_days': lag
            })

    # Aggregate transition stats
    transition_summary = {}
    if transitions:
        df_trans = transitions
        # group by from->to
        by_type = {}
        for tr in df_trans:
            k = f"{tr['from']}->{tr['to']}"
            by_type.setdefault(k, []).append(tr)
        for k, vals in by_type.items():
            before_vals = [v['before_return'] for v in vals if v['before_return'] is not None]
            after_vals = [v['after_return'] for v in vals if v['after_return'] is not None]
            lags = [v['detection_lag_days'] for v in vals if v['detection_lag_days'] is not None]
            transition_summary[k] = {
                'count': len(vals),
                'avg_before_return': float(np.mean(before_vals)) if before_vals else None,
                'avg_after_return': float(np.mean(after_vals)) if after_vals else None,
                'avg_detection_lag_days': float(np.mean(lags)) if lags else None
            }

    result = {
        'symbol': symbol,
        'regime_metrics': regime_metrics,
        'transitions': transitions,
        'transition_summary': transition_summary
    }
    return result


def compute_hybrid_positions(
    final_signals,
    buy_probs,
    sell_probs,
    prices,
    regressor_positions=None,
    quantile_preds=None,
    tft_forecasts=None,
    regimes=None,
    atr_percent=None,
    vol_ratio_5_20=None,
    verbose: bool = False,
    dates=None,
    confidence_scorer: object | None = None,
    model_availability: dict | None = None,
    regressor_preds=None,
    gbm_preds=None,
    gbm_weight: float = 0.2,
    out_dir=None,
):
    """Generate hybrid positions and an enriched per-step trade log.

    This is a best-effort replacement that:
      - Computes a unified confidence per step using `confidence_scorer`.
      - Builds a simple position sizing: `position = base * unified_confidence` where
        `base` prefers `regressor_positions` when available, otherwise uses `signal`.
      - Optionally fuses GBM predictions with regressor predictions.
      - Returns `(positions_array, trade_log_df)` where `trade_log_df` contains
        fields used by downstream analysis (including `unified_confidence` and
        `position_shares` so exit reconstruction can operate).

    The implementation is defensive: it tolerates missing inputs and tries to
    align lengths to the shortest provided array.
    
    Args:
        gbm_preds: Optional array of GBM predicted returns.
        gbm_weight: Weight for GBM in fusion with regressor (default 0.2).
    """
    # Convert to numpy where appropriate
    try:
        fs = np.asarray(final_signals, dtype=int)
    except Exception:
        fs = np.array(final_signals)

    n = int(len(fs))

    def _val(arr, i, dtype=float):
        try:
            if arr is None:
                return None
            return dtype(arr[i])
        except Exception:
            try:
                return dtype(arr[i])
            except Exception:
                return None

    positions = np.zeros(n, dtype=float)
    rows = []

    avail = model_availability or {}
    scorer = confidence_scorer

    for i in range(n):
        sig = int(fs[i]) if i < len(fs) else 0
        buyp = _val(buy_probs, i, float)
        sellp = _val(sell_probs, i, float)
        price = _val(prices, i, float)
        date = None
        if dates is not None and i < len(dates):
            date = dates[i]

        regpos = None
        if regressor_positions is not None and i < len(regressor_positions):
            try:
                regpos = float(regressor_positions[i])
            except Exception:
                regpos = None

        regpred = _val(regressor_preds, i, float)
        
        # GBM prediction fusion (if available)
        gbm_pred = _val(gbm_preds, i, float)
        fused_pred = regpred
        if gbm_pred is not None and gbm_weight > 0:
            if regpred is not None:
                # Fuse regressor and GBM predictions
                fused_pred = (1.0 - gbm_weight) * regpred + gbm_weight * gbm_pred
            else:
                # Use GBM alone
                fused_pred = gbm_pred

        # quantile / tft per-step extraction (best-effort)
        q_preds = None
        if quantile_preds is not None:
            try:
                # if quantile_preds is dict of arrays, extract per-index
                if isinstance(quantile_preds, dict):
                    q_preds = {k: (quantile_preds.get(k)[i] if quantile_preds.get(k) is not None and len(quantile_preds.get(k))>i else None) for k in ('q10','q50','q90')}
                elif isinstance(quantile_preds, (list, tuple)) and len(quantile_preds) > i:
                    q_preds = quantile_preds[i]
            except Exception:
                q_preds = None

        tft_fore = None
        if tft_forecasts is not None and i < len(tft_forecasts):
            try:
                tft_fore = tft_forecasts[i]
            except Exception:
                tft_fore = None

        regime = None
        if regimes is not None and i < len(regimes):
            regime = regimes[i]

        hist_vol = None
        if atr_percent is not None and i < len(atr_percent):
            try:
                hist_vol = float(atr_percent[i])
            except Exception:
                hist_vol = None

        # Compute unified confidence via scorer if provided
        unified = None
        comp_conf = None
        attribution = None
        regime_mult = None
        if scorer is not None:
            try:
                conf = scorer.compute_unified_confidence(
                    signal=sig,
                    buy_prob=buyp,
                    sell_prob=sellp,
                    regressor_pred=fused_pred,  # Use fused GBM+regressor prediction
                    historical_vol=hist_vol,
                    quantile_preds=q_preds,
                    tft_forecasts=tft_fore,
                    regime=regime,
                    model_availability=avail,
                )
                unified = conf.get('unified_confidence')
                comp_conf = conf.get('component_confidences')
                attribution = conf.get('attribution')
                regime_mult = conf.get('regime_multiplier')
            except Exception:
                unified = None

        # Fallback unified confidence if scorer unavailable
        if unified is None:
            if sig == 1 and buyp is not None:
                unified = float(buyp)
            elif sig == -1 and sellp is not None:
                unified = float(sellp)
            else:
                unified = 0.0

        # P1.4: Kelly Criterion Position Sizing
        # Calculate Kelly fraction based on classifier confidence
        win_prob = 0.5  # Default neutral
        if sig == 1 and buyp is not None:
            win_prob = max(buyp, 0.01)  # Use buy probability as win prob
        elif sig == -1 and sellp is not None:
            win_prob = max(sellp, 0.01)  # Use sell probability as win prob
        
        # Win/loss ratio: use market regime as proxy for expected asymmetry
        # In bullish regimes, wins tend to be larger; in bearish, losses are larger
        win_loss_ratio = DEFAULT_WIN_LOSS_RATIO  # Default 1.5
        if regime is not None:
            if regime == 1:  # Bullish
                win_loss_ratio = 2.0  # Expect larger wins
            elif regime == -1:  # Bearish
                win_loss_ratio = 1.2  # Smaller advantage
            else:  # Sideways
                win_loss_ratio = 1.0  # Equal wins/losses
        
        # Calculate Kelly fraction
        kelly_frac = calculate_kelly_fraction(win_prob, win_loss_ratio)
        
        # Apply regime multiplier to Kelly if available
        if regime_mult is not None:
            kelly_frac = kelly_frac * regime_mult

        # Base position: use signal direction for position sign
        # Regressor veto removed - regressor outputs are currently ~0.00089 (effectively constant)
        # Previously used regressor_positions which blocked all SELL signals when regressor was always positive
        base = float(sig)  # +1 for BUY, -1 for SELL, 0 for HOLD

        # P1.4: Apply Kelly-scaled position sizing
        # Position = direction * unified_confidence * kelly_fraction
        pos = float(base) * float(unified) * float(kelly_frac)
        # Clip to [-1.0, 1.0]
        pos = float(np.clip(pos, -1.0, 1.0))
        # Treat numerical underflow / tiny exposures as zero for deterministic aggregation
        if abs(pos) < 1e-6:
            pos = 0.0
        # Mask positions by discrete execution signal: when HOLD (sig==0) ensure positions are zero
        if sig == 0:
            pos = 0.0
        positions[i] = pos

        # Build row for trade log
        tier = None
        try:
            if scorer is not None:
                tier = scorer.get_confidence_tier(unified)
        except Exception:
            tier = None

        # Mask position_shares with discrete signal: zero when HOLD (signal==0).
        masked_shares = float(pos) if sig != 0 else 0.0
        row = {
            'index': int(i),
            'date': date,
            'price': price,
            'action': 'BUY' if sig > 0 else ('SELL' if sig < 0 else 'HOLD'),
            'buy_prob': buyp,
            'sell_prob': sellp,
            'regressor_pred': regpred,
            'gbm_pred': gbm_pred,
            'fused_pred': fused_pred,
            'unified_confidence': float(unified) if unified is not None else None,
            'confidence_tier': tier,
            'confidence_attribution': attribution,
            'component_confidences': comp_conf,
            # P1.4: Add Kelly criterion fields to trade log
            'kelly_fraction': float(kelly_frac),
            'win_prob': float(win_prob),
            'win_loss_ratio': float(win_loss_ratio),
            'regime': regime,
            # expose a 'position_shares' column (fractional exposure) for exit reconstruction
            # Mask by discrete execution signal so aggregation detects starts/exits
            'position_shares': float(masked_shares),
        }
        rows.append(row)

    trade_log_df = pd.DataFrame(rows)
    return positions, trade_log_df


def compress_daily_trade_log_to_trades(trade_log_df: pd.DataFrame):
    """Compress a per-day trade log into a list of per-trade dicts (entry->exit).

    The returned list contains one dict per closed or open lot with keys similar
    to what `analyze_adaptive_exits` expects: 'entry_price','exit_price','targets_hit',
    'days_held','time_degraded','resistance_exit','pnl_pct','price_path','target_prices'.

    This reconstructs lots using FIFO based on the `position_shares` column when
    available, else falls back to action-based BUY/SELL pairing.
    """
    trades = []
    if trade_log_df is None or trade_log_df.empty:
        return trades

    # Work on a copy with an integer index for positions/days
    df = trade_log_df.reset_index(drop=True).copy()

    # Prefer 'position_shares' column for lot reconstruction
    if 'position_shares' in df.columns:
        lots = []
        prev_shares = 0.0
        for idx, row in df.iterrows():
            try:
                shares = float(row.get('position_shares', 0.0)) if pd.notna(row.get('position_shares', 0.0)) else 0.0
            except Exception:
                shares = 0.0
            # Normalize tiny fractional exposures to zero to avoid open/close flapping
            if abs(shares) < 1e-6:
                shares = 0.0
            price = row.get('price') if 'price' in row.index else None
            date = row.get('date') if 'date' in row.index else None
            delta = shares - (prev_shares if prev_shares is not None else 0.0)
            if delta > 0:
                # New lot
                lots.append({'qty': float(delta), 'entry_price': float(price) if price is not None and not pd.isna(price) else None, 'entry_index': int(idx), 'entry_date': date})
            elif delta < 0:
                qty_to_close = float(-delta)
                exit_price = float(price) if price is not None and not pd.isna(price) else None
                while qty_to_close > 0 and lots:
                    lot = lots[0]
                    take = min(lot['qty'], qty_to_close)
                    days_held = int(idx - lot.get('entry_index', idx)) if lot.get('entry_index') is not None else None
                    pnl = None
                    if lot.get('entry_price') is not None and exit_price is not None and lot.get('entry_price') != 0:
                        pnl = (exit_price / lot.get('entry_price') - 1.0)
                    exit_row = {
                        'entry_price': lot.get('entry_price'),
                        'exit_price': exit_price,
                        'entry_index': lot.get('entry_index'),
                        'exit_index': int(idx),
                        'days_held': days_held,
                        'targets_hit': [False, False, False, False],
                        'time_degraded': False,
                        'resistance_exit': False,
                        'pnl_pct': pnl,
                        'price_path': [lot.get('entry_price'), exit_price] if lot.get('entry_price') is not None and exit_price is not None else None
                    }
                    trades.append(exit_row)
                    lot['qty'] = float(lot['qty'] - take)
                    qty_to_close = float(qty_to_close - take)
                    if lot['qty'] <= 0:
                        lots.pop(0)
            prev_shares = shares

        # Any remaining lots open at end
        if lots:
            try:
                last_idx = int(df.index[-1])
            except Exception:
                last_idx = len(df) - 1
            last_price = df.iloc[-1].get('price') if 'price' in df.columns else None
            exit_price = float(last_price) if last_price is not None and not pd.isna(last_price) else None
            for lot in lots:
                days_held = int(last_idx - lot.get('entry_index', last_idx)) if lot.get('entry_index') is not None else None
                pnl = None
                if lot.get('entry_price') is not None and exit_price is not None and lot.get('entry_price') != 0:
                    pnl = (exit_price / lot.get('entry_price') - 1.0)
                exit_row = {
                    'entry_price': lot.get('entry_price'),
                    'exit_price': exit_price,
                    'entry_index': lot.get('entry_index'),
                    'exit_index': last_idx,
                    'days_held': days_held,
                    'targets_hit': [False, False, False, False],
                    'time_degraded': False,
                    'resistance_exit': False,
                    'pnl_pct': pnl,
                    'price_path': [lot.get('entry_price'), exit_price] if lot.get('entry_price') is not None and exit_price is not None else None,
                    'open_at_end': True
                }
                trades.append(exit_row)
        return trades

    # Fallback: action-based BUY/SELL pairing
    lots_action = []
    for idx, row in df.iterrows():
        action = (row.get('action') or '').upper() if 'action' in row.index else ''
        price = row.get('price')
        date = row.get('date')
        if action == 'BUY':
            lots_action.append({'entry_price': float(price) if pd.notna(price) else None, 'entry_index': int(idx), 'entry_date': date})
        elif action == 'SELL':
            if lots_action:
                lot = lots_action.pop(0)
                exit_price = float(price) if pd.notna(price) else None
                days_held = int(idx - lot.get('entry_index', idx)) if lot.get('entry_index') is not None else None
                pnl = None
                if lot.get('entry_price') is not None and exit_price is not None and lot.get('entry_price') != 0:
                    pnl = (exit_price / lot.get('entry_price') - 1.0)
                exit_row = {
                    'entry_price': lot.get('entry_price'),
                    'exit_price': exit_price,
                    'entry_index': lot.get('entry_index'),
                    'exit_index': int(idx),
                    'days_held': days_held,
                    'targets_hit': [False, False, False, False],
                    'time_degraded': False,
                    'resistance_exit': False,
                    'pnl_pct': pnl,
                    'price_path': [lot.get('entry_price'), exit_price] if lot.get('entry_price') is not None and exit_price is not None else None
                }
                trades.append(exit_row)

    # Any remaining BUY lots open
    if lots_action:
        try:
            last_idx = int(df.index[-1])
        except Exception:
            last_idx = len(df) - 1
        last_price = df.iloc[-1].get('price') if 'price' in df.columns else None
        exit_price = float(last_price) if last_price is not None and not pd.isna(last_price) else None
        for lot in lots_action:
            days_held = int(last_idx - lot.get('entry_index', last_idx)) if lot.get('entry_index') is not None else None
            pnl = None
            if lot.get('entry_price') is not None and exit_price is not None and lot.get('entry_price') != 0:
                pnl = (exit_price / lot.get('entry_price') - 1.0)
            exit_row = {
                'entry_price': lot.get('entry_price'),
                'exit_price': exit_price,
                'entry_index': lot.get('entry_index'),
                'exit_index': last_idx,
                'days_held': days_held,
                'targets_hit': [False, False, False, False],
                'time_degraded': False,
                'resistance_exit': False,
                'pnl_pct': pnl,
                'price_path': [lot.get('entry_price'), exit_price] if lot.get('entry_price') is not None and exit_price is not None else None,
                'open_at_end': True
            }
            trades.append(exit_row)

    return trades


def aggregate_daily_positions_to_trades(trade_log_df: pd.DataFrame):
    """Aggregate a daily position log into per-trade records.

    Logic:
      - Trade start: day position moves from 0 (or <=0) to >0
      - Trade end: day position moves from >0 to 0 (or <=0)
      - For each completed trade, emit one dict with entry/exit price, dates, indices and realized return.

    Expects `position_shares` column (fractional exposure). Falls back to 'position' or 'positions'.
    """
    trades = []
    if trade_log_df is None:
        return trades

    df = trade_log_df.reset_index(drop=True).copy()

    # Prefer action-based detection when action column exists.
    # Option B: Start trade on a BUY day; end on the first subsequent SELL OR the first subsequent HOLD day.
    if 'action' in df.columns:
        lots_action = []
        for idx, row in df.iterrows():
            action = (row.get('action') or '').upper()
            price = row.get('price')
            date = row.get('date')

            # Each BUY creates a new lot (one trade entry)
            if action == 'BUY':
                lots_action.append({'entry_price': float(price) if pd.notna(price) else None, 'entry_index': int(idx), 'entry_date': date})

            # On SELL or HOLD, close all open lots (FIFO) at this price/date
            elif action == 'SELL' or action == 'HOLD':
                if lots_action:
                    exit_price = float(price) if pd.notna(price) else None
                    for lot in lots_action:
                        days_held = int(idx - lot.get('entry_index', idx)) if lot.get('entry_index') is not None else None
                        pnl = None
                        if lot.get('entry_price') is not None and exit_price is not None and lot.get('entry_price') != 0:
                            pnl = (exit_price / lot.get('entry_price') - 1.0)
                        exit_row = {
                            'entry_price': lot.get('entry_price'),
                            'exit_price': exit_price,
                            'entry_index': lot.get('entry_index'),
                            'exit_index': int(idx),
                            'entry_date': lot.get('entry_date'),
                            'exit_date': date,
                            'days_held': days_held,
                            'targets_hit': [False, False, False, False],
                            'time_degraded': False,
                            'resistance_exit': False,
                            'pnl_pct': pnl,
                            'price_path': None
                        }
                        trades.append(exit_row)
                    # Clear lots after closing
                    lots_action = []

        # Any remaining BUY lots open at end -> mark open_at_end and close at final price
        if lots_action:
            try:
                last_idx = int(df.index[-1])
            except Exception:
                last_idx = len(df) - 1
            last_price = df.iloc[-1].get('price') if 'price' in df.columns else None
            exit_price = float(last_price) if last_price is not None and not pd.isna(last_price) else None
            for lot in lots_action:
                days_held = int(last_idx - lot.get('entry_index', last_idx)) if lot.get('entry_index') is not None else None
                pnl = None
                if lot.get('entry_price') is not None and exit_price is not None and lot.get('entry_price') != 0:
                    pnl = (exit_price / lot.get('entry_price') - 1.0)
                exit_row = {
                    'entry_price': lot.get('entry_price'),
                    'exit_price': exit_price,
                    'entry_index': lot.get('entry_index'),
                    'exit_index': last_idx,
                    'entry_date': lot.get('entry_date'),
                    'exit_date': df.iloc[-1].get('date') if 'date' in df.columns else None,
                    'days_held': days_held,
                    'targets_hit': [False, False, False, False],
                    'time_degraded': False,
                    'resistance_exit': False,
                    'pnl_pct': pnl,
                    'price_path': None,
                    'open_at_end': True
                }
                trades.append(exit_row)
        return trades

    # Use position column for start/end detection
    prev_pos = 0.0
    open_trade = None
    prices = df['price'].tolist() if 'price' in df.columns else [None] * len(df)
    dates = df['date'].tolist() if 'date' in df.columns else [None] * len(df)

    for idx, row in df.iterrows():
        try:
            pos = float(row.get(pos_col, 0.0)) if not pd.isna(row.get(pos_col, 0.0)) else 0.0
        except Exception:
            pos = 0.0
        # Treat tiny exposures as zero for robust start/end detection
        if abs(pos) < 1e-6:
            pos = 0.0
        price = row.get('price') if 'price' in row.index else None
        date = row.get('date') if 'date' in row.index else None

        # Start
        if pos > 0 and (prev_pos <= 0):
            open_trade = {'entry_index': int(idx), 'entry_price': float(price) if price is not None and not pd.isna(price) else None, 'entry_date': date}

        # End
        if prev_pos > 0 and pos <= 0 and open_trade is not None:
            exit_price = float(price) if price is not None and not pd.isna(price) else None
            days_held = int(idx - open_trade.get('entry_index', idx)) if open_trade.get('entry_index') is not None else None
            pnl = None
            if open_trade.get('entry_price') is not None and exit_price is not None and open_trade.get('entry_price') != 0:
                pnl = (exit_price / open_trade.get('entry_price') - 1.0)
            price_path = None
            try:
                price_path = [float(p) for p in prices[open_trade.get('entry_index', idx): idx + 1] if p is not None]
            except Exception:
                price_path = None
            trades.append({
                'entry_price': open_trade.get('entry_price'),
                'exit_price': exit_price,
                'entry_index': open_trade.get('entry_index'),
                'exit_index': int(idx),
                'entry_date': open_trade.get('entry_date'),
                'exit_date': date,
                'days_held': days_held,
                'targets_hit': [False, False, False, False],
                'time_degraded': False,
                'resistance_exit': False,
                'pnl_pct': pnl,
                'price_path': price_path
            })
            open_trade = None

        prev_pos = pos

    # If trade still open at end
    if open_trade is not None:
        last_idx = int(df.index[-1])
        last_price = df.iloc[-1].get('price') if 'price' in df.columns else None
        exit_price = float(last_price) if last_price is not None and not pd.isna(last_price) else None
        days_held = int(last_idx - open_trade.get('entry_index', last_idx)) if open_trade.get('entry_index') is not None else None
        pnl = None
        if open_trade.get('entry_price') is not None and exit_price is not None and open_trade.get('entry_price') != 0:
            pnl = (exit_price / open_trade.get('entry_price') - 1.0)
        price_path = None
        try:
            price_path = [float(p) for p in prices[open_trade.get('entry_index', last_idx): last_idx + 1] if p is not None]
        except Exception:
            price_path = None
        trades.append({
            'entry_price': open_trade.get('entry_price'),
            'exit_price': exit_price,
            'entry_index': open_trade.get('entry_index'),
            'exit_index': last_idx,
            'entry_date': open_trade.get('entry_date'),
            'exit_date': df.iloc[-1].get('date') if 'date' in df.columns else None,
            'days_held': days_held,
            'targets_hit': [False, False, False, False],
            'time_degraded': False,
            'resistance_exit': False,
            'pnl_pct': pnl,
            'price_path': price_path,
            'open_at_end': True
        })

    return trades


# -------------------------
# Main
# -------------------------
def main(
    symbol='AAPL',
    seq_len=60,
    backtest_days=60,
    buy_conf_floor=0.33,
    sell_conf_floor=0.5,
    max_short_exposure=0.5,
    reg_strategy='confidence_scaling',
    reg_threshold=0.01,
    reg_scale=15.0,
    fusion_mode='stacking',  # December 2025: Default changed from 'classifier' to 'stacking'
    gbm_weight_override: float | None = None,
    skip_regressor_backtest=False,
    forward_sim=False,
    forward_days=None,
    threshold_file: str | None = None,
    use_cache: bool = True,
    force_refresh: bool = False,
    use_tft: bool = False,
    use_quantile: bool = False,
    get_data: bool = False,
    auto_model: bool = False,
    disable_fallback: bool = False,  # P0 FIX: Allow A/B testing with fallback disabled
    margin_requirement: float = 0.5, # Phase 5: Margin requirement
    borrow_rate: float = 0.02,        # Phase 5: Annual borrow rate
):
    """
    Inference + backtesting with optional data caching.

    Args:
        use_cache: Whether to use cached data (default True)
        force_refresh: Force data refetch even if cache exists
        use_tft: Whether to run TFT backtest alongside standard backtest
        use_quantile: Whether to enable Quantile Regressor for uncertainty-aware fusion
        gbm_weight_override: Override GBM weight (0.0-1.0) regardless of fusion_mode
        auto_model: Automatically select best fusion mode based on model health validation
        disable_fallback: Disable LSTM collapse fallback for A/B testing (default False)
    """
    # ========================================
    # CACHE PRE-VALIDATION (FOR TFT)
    # ========================================
    # Inserted validation to ensure TFT backtests only run when cache is present
    # and to create cache proactively if missing. This keeps TFT backtests
    # deterministic and avoids the 15-minute refetch during heavy backtests.
    if use_tft:
        print("\n" + "="*60)
        print("  TFT CACHE PRE-VALIDATION")
        print("="*60)
        
        # Use module-level DataCacheManager import (already imported at top)
        cache_manager = DataCacheManager()
        cache_info = cache_manager.get_cache_info(symbol)

        # `DataCacheManager.get_cache_info` returns 'is_valid' not 'valid'
        if cache_info is None or not cache_info.get('is_valid', False):
            print(f"❌ No valid cache found for {symbol}")
            print(f"   TFT backtest requires cached data to avoid 15-min refetch")
            print(f"\n   Creating cache now (this will take ~15 minutes)...")
            
            # Create cache once
            try:
                _, _, _, _ = cache_manager.get_or_fetch_data(
                    symbol=symbol,
                    include_sentiment=True,
                    force_refresh=True
                )
                print(f"✓ Cache created for {symbol}")
            except Exception as e:
                print(f"❌ Cache creation failed: {e}")
                print(f"   TFT backtest will be skipped")
                use_tft = False
        else:
            print(f"✓ Valid cache found for {symbol}")
            print(f"   Age: {cache_info.get('age_hours', 0):.1f} hours")
            print(f"   Rows: {cache_info.get('prepared_shape', (0,))[0]}")
            print(f"   Features: {cache_info.get('feature_count', 0)}")

    save_dir = Path('saved_models')
    # Track which models successfully loaded
    model_status = {
        'stacking': False,  # December 2025: Stacking is now primary ensemble method
        'regressor': False,
        'classifiers': False,  # Deprecated - never used
        'quantile': False,
        'tft': False,
        'gbm': False
    }

    # Check for stacking model availability (December 2025)
    stacking_predictor = None
    if fusion_mode == 'stacking':
        if STACKING_AVAILABLE and load_stacking_predictor is not None:
            stacking_predictor = load_stacking_predictor(symbol)
            if stacking_predictor is not None and stacking_predictor.meta_learner is not None:
                model_status['stacking'] = True
                print(f"✓ Loaded stacking predictor for {symbol}")
                print(f"  Base models: {list(stacking_predictor.base_models.keys())}")
                print(f"  Reliability scores: {stacking_predictor.model_reliability}")
            else:
                raise RuntimeError(
                    f"Stacking ensemble not found for {symbol}. Train it first:\n"
                    f"  python train_all.py --symbol {symbol}\n"
                    f"Or use a fallback mode: --fusion-mode gbm_only"
                )
        else:
            raise RuntimeError(
                "StackingPredictor not available. Check inference/stacking_predictor.py import."
            )
    # Guard to ensure TFT backtest only runs once even if code reaches multiple
    # integration points below. This avoids duplicate heavy work and double-logs.
    tft_attempted = False
    
    # CREATE TIMESTAMPED FOLDER EARLY - Used for all backtest outputs
    import time
    ts = time.strftime('%Y%m%d_%H%M%S')
    base_dir = Path(__file__).resolve().parent
    backtest_folder = base_dir / 'backtest_results' / f"{symbol}_{ts}"
    backtest_folder.mkdir(parents=True, exist_ok=True)
    results_dir = backtest_folder  # Use this for all outputs throughout the function
    
    # Look for scalers/features - try new organized structure first, then legacy paths
    feature_scaler = None
    feature_cols = None
    target_scaler = None
    target_metadata = {}
    classifier_metadata = {}
    
    # Try new organized path structure if ModelPaths available
    if ModelPaths is not None:
        paths = ModelPaths(symbol)
        
        # Feature scaler and columns (prefer regressor, fall back to classifiers)
        if paths.regressor.feature_scaler.exists():
            feature_scaler = safe_load_pickle(paths.regressor.feature_scaler)
        elif paths.classifiers.feature_scaler.exists():
            feature_scaler = safe_load_pickle(paths.classifiers.feature_scaler)
        if paths.feature_columns.exists():
            feature_cols = safe_load_pickle(paths.feature_columns)
        
        # Target scaler and metadata from regressor
        if paths.regressor.target_scaler.exists():
            target_scaler = safe_load_pickle(paths.regressor.target_scaler)
        if paths.regressor.metadata.exists():
            target_metadata = safe_load_pickle(paths.regressor.metadata)
        
        # Classifier metadata
        if paths.classifiers.metadata.exists():
            classifier_metadata = safe_load_pickle(paths.classifiers.metadata)
        
        if feature_scaler is not None and feature_cols is not None:
            print(f"✓ Loaded artifacts from new organized path structure: {paths.symbol_dir}")
    
    # Fall back to legacy paths if not found
    if feature_scaler is None or feature_cols is None:
        candidate_feature_scaler_files = [
            save_dir / f'{symbol}_1d_regressor_final_feature_scaler.pkl',
            save_dir / f'{symbol}_binary_feature_scaler.pkl',
            save_dir / f'{symbol}_paper_regressor_5d_scaler.pkl'  # fallback name
        ]
        candidate_feature_list_files = [
            save_dir / f'{symbol}_1d_regressor_final_features.pkl',
            save_dir / f'{symbol}_binary_features.pkl',
            save_dir / f'{symbol}_paper_regressor_5d_features.pkl'
        ]

        for p in candidate_feature_scaler_files:
            if p.exists():
                feature_scaler = safe_load_pickle(p)
                break
        for p in candidate_feature_list_files:
            if p.exists():
                feature_cols = safe_load_pickle(p)
                break
    
    # For GBM-only mode, try to load from GBM model artifacts as fallback
    GBM_ONLY_MODES = ('gbm_only',)
    if (feature_scaler is None or feature_cols is None) and fusion_mode in GBM_ONLY_MODES:
        gbm_feature_cols_path = save_dir / symbol / 'gbm' / 'feature_columns.pkl'
        gbm_xgb_scaler_path = save_dir / symbol / 'gbm' / 'xgb_scaler.joblib'
        
        if gbm_feature_cols_path.exists():
            feature_cols = safe_load_pickle(gbm_feature_cols_path)
            print(f"✓ Loaded feature columns from GBM artifacts: {gbm_feature_cols_path}")
        
        if feature_scaler is None and gbm_xgb_scaler_path.exists():
            # For GBM-only, we'll use identity scaler since GBM models have their own scalers
            feature_scaler = 'gbm_internal'  # Marker to indicate GBM will handle scaling
            print(f"✓ GBM-only mode: Using internal GBM scalers")
    
    if feature_scaler is None or feature_cols is None:
        raise FileNotFoundError("Feature scaler or features not found in saved_models/. "
                                "Make sure training scripts saved artifacts.")

    # Load target scaler if not loaded from new structure
    if target_scaler is None:
        target_scaler_file = save_dir / f'{symbol}_1d_regressor_final_target_scaler.pkl'
        if not target_scaler_file.exists():
            if fusion_mode not in GBM_ONLY_MODES:
                raise FileNotFoundError(f"Target scaler not found: {target_scaler_file}")
            else:
                # GBM-only mode doesn't need target scaler - GBM predicts raw returns
                print(f"✓ GBM-only mode: Target scaler not required (GBM predicts raw returns)")
                target_scaler = None
        else:
            target_scaler = safe_load_pickle(target_scaler_file)
    
    # Load metadata if not loaded from new structure
    if not target_metadata:
        target_metadata_file = save_dir / f'{symbol}_1d_regressor_final_metadata.pkl'
        target_metadata = safe_load_pickle(target_metadata_file) if target_metadata_file.exists() else {}
    
    if not classifier_metadata:
        classifier_metadata_file = save_dir / f'{symbol}_binary_classifiers_final_metadata.pkl'
        classifier_metadata = safe_load_pickle(classifier_metadata_file) if classifier_metadata_file.exists() else {}
    
    target_transform = target_metadata.get('target_transform', 'raw')

    # ===================================================================
    # VALIDATION: Check all required model files exist
    # ===================================================================
    if fusion_mode not in GBM_ONLY_MODES:
        try:
            validate_model_files(symbol, save_dir)
        except FileNotFoundError as e:
            print(f"\n❌ ERROR: Model files not found")
            print(f"{e}")
            print(f"\n💡 Solution: Train models first using:")
            print(f"   python training/train_1d_regressor_final.py {symbol}")
            print(f"   python training/train_binary_classifiers_final.py {symbol}")
            sys.exit(1)
    else:
        # For GBM-only mode, validate GBM models exist instead
        gbm_dir = save_dir / symbol / 'gbm'
        xgb_model_path = gbm_dir / 'xgb_reg.joblib'
        if not xgb_model_path.exists():
            print(f"\n❌ ERROR: GBM model not found")
            print(f"   Missing: {xgb_model_path}")
            print(f"\n💡 Solution: Train GBM models first using:")
            print(f"   python training/train_gbm_baseline.py {symbol}")
            sys.exit(1)
        print(f"✓ GBM-only mode: Found XGBoost model at {xgb_model_path}")

    # Get sequence lengths from metadata (regressor vs classifier may differ)
    regressor_seq_len = target_metadata.get('sequence_length', 90)  # Default to 90
    classifier_seq_len = classifier_metadata.get('sequence_length', 60)  # Default to 60
    
    # Use regressor sequence length for main data preparation
    seq_len = regressor_seq_len
    print(f"Using sequence lengths: Regressor={regressor_seq_len}, Classifier={classifier_seq_len}")

    # load weights (skip check for GBM-only mode)
    regressor_weights = save_dir / f'{symbol}_1d_regressor_final.weights.h5'
    if fusion_mode not in GBM_ONLY_MODES and not regressor_weights.exists():
        raise FileNotFoundError("Regressor weight file not found. Run training first.")

    # load data and prepare (use cache when requested)
    if use_cache:
        cache_manager = DataCacheManager()
        raw_df, engineered_df, prepared_df, cached_feature_cols = cache_manager.get_or_fetch_data(
            symbol=symbol,
            include_sentiment=True,
            force_refresh=force_refresh
        )
        df = engineered_df
        print(f"✓ Loaded engineered features for {symbol} from cache ({len(df)} rows)")
    else:
        # When not using cache, force a fetch-and-cache so the result is saved for future runs.
        # Use the module-level `DataCacheManager` import (avoid local import which would
        # shadow the name and make it a local variable for the whole function).
        cache_manager = DataCacheManager()
        raw_df, engineered_df, prepared_df, feature_cols = cache_manager.get_or_fetch_data(
            symbol=symbol,
            include_sentiment=True,
            force_refresh=True
        )
        df = engineered_df
        print(f"✓ Fetched and cached engineered features for {symbol} ({len(df)} rows)")
    
    df_clean, feature_cols_from_prep = prepare_training_data(df, horizons=[1])
    # Use saved feature_cols if provided, else fallback
    used_feature_cols = feature_cols if feature_cols is not None else feature_cols_from_prep
    
    # ===================================================================
    # VALIDATION: Check feature counts match expectations
    # ===================================================================
    expected_n_features = target_metadata.get('n_features') if isinstance(target_metadata, dict) else None
    if expected_n_features is not None and expected_n_features != len(used_feature_cols):
        print(f"\n❌ ERROR: Feature count mismatch with model metadata")
        print(f"   Model expects {expected_n_features} features, but current feature set has {len(used_feature_cols)}")
        print(f"   Feature columns used: {len(used_feature_cols)}")
        print(f"   Model architecture: {target_metadata.get('architecture', {})}")
        print(f"\n💡 Solutions:")
        print(f"   1. Retrain the model with the current feature set")
        print(f"   2. Recompute features to match saved model (check sentiment flags and feature list)")
        print(f"   3. Edit saved_models/{symbol}_1d_regressor_final_features.pkl if you know what you are doing")
        sys.exit(1)
    try:
        df_clean = validate_feature_counts(df_clean, used_feature_cols, symbol)
    except ValueError as e:
        print(f"\n❌ ERROR: Feature validation failed")
        print(f"{e}")
        print(f"\n💡 Check that:")
        print(f"   1. Symbol parameter passed to engineer_features()")
        print(f"   2. Sentiment features are enabled if model expects 118 features")
        print(f"   3. Feature scaler matches model training")
        print(f"   4. Feature engineering completed successfully")
        # Attempt to auto-fill missing features with neutral values (safe fallback)
        missing_cols = set(used_feature_cols) - set(df_clean.columns)
        if missing_cols:
            print(f"\n⚠️ Auto-filling missing features: {sorted(missing_cols)} with neutral values")
            for col in missing_cols:
                df_clean[col] = 0.0
            # Re-run validation; if still fails, raise
            try:
                df_clean = validate_feature_counts(df_clean, used_feature_cols, symbol)
            except ValueError as e2:
                print(f"\n❌ ERROR: Feature validation still failed after auto-fill: {e2}")
                sys.exit(1)
        else:
            sys.exit(1)
    
    X = df_clean[used_feature_cols].values
    y = df_clean['target_1d'].values
    price_col = 'Close' if 'Close' in df_clean.columns else 'close'
    price_values = df_clean[price_col].values
    date_values = df_clean.index.to_numpy() if hasattr(df_clean.index, 'to_numpy') else np.arange(len(df_clean))
    
    # Extract volatility features for position sizing
    atr_percent_values = df_clean['atr_percent'].values if 'atr_percent' in df_clean.columns else None
    vol_ratio_5_20_values = df_clean['vol_ratio_5_20'].values if 'vol_ratio_5_20' in df_clean.columns else None
    
    # Feature scaling (skip for GBM-only mode - GBM handles its own scaling)
    if feature_scaler == 'gbm_internal':
        # GBM-only mode: use unscaled features (GBM models have internal scalers)
        X_scaled = X.copy()
        print(f"✓ GBM-only mode: Using unscaled features (GBM has internal scalers)")
    else:
        X_scaled = feature_scaler.transform(X)
    
    # ===================================================================
    # CRITICAL: Separate sequence creation for regressor vs classifier
    # - Regressor uses 90-day sequences (self.seq_len_regressor = 90)
    # - Classifiers use 60-day sequences (self.seq_len_classifier = 60)
    # - Need to align sequences by trimming classifier sequences from start
    # ===================================================================
    
    # Create sequences for regressor (90-day)
    X_seq_reg = create_sequences(X_scaled, regressor_seq_len)
    y_aligned_reg = y[regressor_seq_len - 1:]
    prices_aligned_reg = price_values[regressor_seq_len - 1:]
    dates_aligned_reg = date_values[regressor_seq_len - 1:]
    
    # Create sequences for classifiers (60-day)
    X_seq_clf = create_sequences(X_scaled, classifier_seq_len)
    
    # ===================================================================
    # VALIDATION: Check sequence shapes
    # ===================================================================
    try:
        print(f"\n=== Sequence Shape Validation ===")
        validate_sequences(X_seq_reg, f"Regressor sequences", expected_features=len(used_feature_cols))
        validate_sequences(X_seq_clf, f"Classifier sequences", expected_features=len(used_feature_cols))
    except ValueError as e:
        print(f"\n❌ ERROR: Sequence validation failed")
        print(f"{e}")
        print(f"\n💡 This usually means:")
        print(f"   1. Insufficient data (need at least {max(regressor_seq_len, classifier_seq_len)} days)")
        print(f"   2. Feature count mismatch between training and inference")
        print(f"   3. Sequence creation logic error")
        sys.exit(1)
    
    # Calculate alignment offset
    offset = regressor_seq_len - classifier_seq_len  # Should be 90 - 60 = 30
    
    if offset > 0:
        # Classifier has more sequences (because it uses shorter window)
        # Trim first `offset` sequences from classifier to align with regressor dates
        X_seq_clf = X_seq_clf[offset:]
        print(f"\n=== Sequence Alignment ===\nRegressor: {len(X_seq_reg)} sequences ({regressor_seq_len}-day)")
        print(f"Classifier: {len(X_seq_clf)} sequences ({classifier_seq_len}-day, trimmed {offset} to align)")
    elif offset < 0:
        # Regressor has more sequences (shouldn't happen with standard architecture)
        # Trim regressor sequences from start
        offset_abs = abs(offset)
        X_seq_reg = X_seq_reg[offset_abs:]
        y_aligned_reg = y_aligned_reg[offset_abs:]
        prices_aligned_reg = prices_aligned_reg[offset_abs:]
        dates_aligned_reg = dates_aligned_reg[offset_abs:]
        print(f"\n=== Sequence Alignment (UNUSUAL) ===\nTrimmed {offset_abs} regressor sequences to align with classifier")
    else:
        # Same sequence length, no alignment needed
        print(f"\n=== Sequence Alignment ===\nBoth use {regressor_seq_len}-day sequences, no offset needed")
    
    # Verify alignment
    assert len(X_seq_reg) == len(X_seq_clf), f"Sequence alignment failed: reg={len(X_seq_reg)}, clf={len(X_seq_clf)}"
    
    # Use regressor sequences for main data flow (all downstream arrays aligned with 90-day sequences)
    X_seq = X_seq_reg
    y_aligned = y_aligned_reg
    prices_aligned = prices_aligned_reg
    dates_aligned = dates_aligned_reg
    
    # ===================================================================
    # CRITICAL: Align ALL supporting arrays with regressor sequences
    # - All arrays must be aligned to regressor_seq_len - 1 offset
    # - This ensures y, prices, dates, and volatility features match X_seq_reg
    # ===================================================================
    
    # Align volatility features with regressor sequences (90-day alignment)
    # These features are used for Kelly Criterion position sizing
    if atr_percent_values is not None:
        atr_percent_aligned = atr_percent_values[regressor_seq_len - 1:]
    else:
        atr_percent_aligned = None
    
    if vol_ratio_5_20_values is not None:
        vol_ratio_5_20_aligned = vol_ratio_5_20_values[regressor_seq_len - 1:]
    else:
        vol_ratio_5_20_aligned = None

    # ===================================================================
    # TRAIN/TEST SPLIT (80/20) - After sequence creation and alignment
    # - Split is calculated from regressor sequences (X_seq_reg)
    # - All arrays (reg, clf, y, prices, dates, volatility) use same split point
    # ===================================================================
    split = int(len(X_seq) * 0.8)
    X_test = X_seq[split:]
    X_test_clf = X_seq_clf[split:]
    y_test = y_aligned[split:]
    test_prices = prices_aligned[split:]
    test_dates = dates_aligned[split:]
    test_atr_percent = atr_percent_aligned[split:] if atr_percent_aligned is not None else None
    test_vol_ratio = vol_ratio_5_20_aligned[split:] if vol_ratio_5_20_aligned is not None else None
    
    # Verify all test arrays are aligned
    assert len(X_test) == len(X_test_clf) == len(y_test) == len(test_prices) == len(test_dates), \
        f"Test set length mismatch after split: X_test={len(X_test)}, X_test_clf={len(X_test_clf)}, " \
        f"y_test={len(y_test)}, test_prices={len(test_prices)}, test_dates={len(test_dates)}"
    
    # ===================================================================
    # BACKTEST WINDOW TRIMMING (if specified)
    # - Trim from END of arrays to get most recent N days
    # - Maintain alignment across all arrays
    # ===================================================================
    if backtest_days is not None:
        tail = min(int(backtest_days), len(X_test))
        if tail > 0 and tail < len(X_test):
            X_test = X_test[-tail:]
            X_test_clf = X_test_clf[-tail:]
            y_test = y_test[-tail:]
            test_prices = test_prices[-tail:]
            test_dates = test_dates[-tail:]
            if test_atr_percent is not None:
                test_atr_percent = test_atr_percent[-tail:]
            if test_vol_ratio is not None:
                test_vol_ratio = test_vol_ratio[-tail:]
            
            # Verify alignment after trimming
            assert len(X_test) == len(X_test_clf) == len(y_test) == len(test_prices) == len(test_dates), \
                f"Test set length mismatch after backtest trim: X_test={len(X_test)}, X_test_clf={len(X_test_clf)}, " \
                f"y_test={len(y_test)}, test_prices={len(test_prices)}, test_dates={len(test_dates)}"

    window_note = f" (last {len(X_test)} days)" if backtest_days else ""
    print(f"Loaded {len(X_seq)} sequences, test size: {len(X_test)}{window_note}")

    # -------------------------
    # Build models and load weights
    # -------------------------
    print("Rebuilding models (must match training architecture)...")
    
    # Validate feature count before model creation
    # Use expected_n_features from metadata (supports feature selection) instead of hardcoded EXPECTED_FEATURE_COUNT
    model_expected_features = expected_n_features if expected_n_features is not None else len(used_feature_cols)
    if len(used_feature_cols) != model_expected_features:
        print(f"\n❌ ERROR: Assertion failed (alignment issue)")
        print(f"   Feature count mismatch! Model expects {model_expected_features}, got {len(used_feature_cols)}")
        print(f"\n💡 This indicates a sequence alignment bug.")
        print(f"   Please report this error with the full stack trace.")
        sys.exit(1)
    
    print(f"   [✓] {symbol}: Using {len(used_feature_cols)} features")

    # Build multitask regressor (EXACT match to training - 3 output heads)
    arch_reg = target_metadata.get('architecture', {}) if isinstance(target_metadata, dict) else {}
    reg_model = None
    regressor_loaded = False
    
    # Skip regressor loading for GBM-only mode
    if fusion_mode in GBM_ONLY_MODES:
        print(f"\n📊 GBM-only mode: Skipping LSTM regressor loading")
        regressor_loaded = False
        reg_model = None
    else:
        # First, try to load from SavedModel directory (new format)
        # ModelPaths is already imported at the top of the file
        model_paths_reg = ModelPaths(symbol)
        saved_model_path = model_paths_reg.regressor.model
        
        if saved_model_path.exists():
            try:
                # Try Keras load first
                loaded_model = keras.models.load_model(str(saved_model_path))
                # Verify it's a proper Keras model with predict method
                if hasattr(loaded_model, 'predict') and callable(getattr(loaded_model, 'predict')):
                    reg_model = loaded_model
                    regressor_loaded = True
                    print(f"✓ Regressor loaded from Keras SavedModel: {saved_model_path}")
                else:
                    raise ValueError("Loaded object is not a valid Keras model (no predict method)")
            except Exception as e:
                print(f"Keras load failed ({e}), trying TF SavedModel...")
                try:
                    # Fall back to TF SavedModel format (from model.export())
                    saved_model_obj = tf.saved_model.load(str(saved_model_path))
                    
                    # Wrap in a callable class with predict method
                    class SavedModelWrapper:
                        def __init__(self, model):
                            self._model = model
                            self._infer = model.signatures.get('serving_default')
                            if self._infer is None:
                                # Try other signature names
                                sig_keys = list(model.signatures.keys())
                                if sig_keys:
                                    self._infer = model.signatures[sig_keys[0]]
                        
                        def predict(self, x, verbose=0):
                            if self._infer is None:
                                raise ValueError("No serving signature found in SavedModel")
                            result = self._infer(tf.constant(x, dtype=tf.float32))
                            # Return as list of arrays (multitask format: [magnitude, sign, volatility])
                            outputs = [v.numpy() for v in result.values()]
                            # If single output, wrap in list
                            if len(outputs) == 1:
                                return outputs
                            return outputs
                    
                    reg_model = SavedModelWrapper(saved_model_obj)
                    regressor_loaded = True
                    print(f"✓ Regressor loaded from TF SavedModel: {saved_model_path}")
                except Exception as e2:
                    print(f"❌ Failed to load SavedModel: {e2}")
        
        # Fall back to rebuild + load weights approach
        if not regressor_loaded:
            try:
                reg_model = create_multitask_regressor(
                    sequence_length=seq_len,
                    n_features=len(used_feature_cols),
                    name='multitask_regressor',
                    arch=arch_reg
                )
                print(f"Reconstructed regressor architecture: {arch_reg}")
                
                # P0 FIX: Check both new organized path and legacy path for weights
                # New structure: saved_models/{SYMBOL}/regressor/regressor.weights.h5
                new_weights_path = model_paths_reg.regressor.weights
                # Legacy structure: saved_models/{symbol}_1d_regressor_final.weights.h5
                legacy_weights_path = regressor_weights
                
                weights_to_load = None
                if new_weights_path.exists():
                    weights_to_load = new_weights_path
                    print(f"Found weights at new organized path: {new_weights_path}")
                elif legacy_weights_path.exists():
                    weights_to_load = legacy_weights_path
                    print(f"Found weights at legacy path: {legacy_weights_path}")
                else:
                    print(f"❌ No weights found at either:")
                    print(f"   New path: {new_weights_path}")
                    print(f"   Legacy path: {legacy_weights_path}")
                
                if weights_to_load:
                    try:
                        reg_model.load_weights(str(weights_to_load))
                        regressor_loaded = True
                        print(f"✓ Regressor weights loaded from {weights_to_load}")
                    except Exception as e:
                        print(f"❌ Failed to load regressor weights from {weights_to_load}: {e}")
                        import traceback
                        traceback.print_exc()
                        print("Regressor will be set to None and inference will continue where possible.")
                        reg_model = None
                else:
                    print("Regressor will be set to None - no weights available.")
                    reg_model = None
            except Exception as e:
                print(f"❌ Failed to reconstruct or load regressor model: {e}")
                import traceback
                traceback.print_exc()
                reg_model = None
                regressor_loaded = False
    # Update model status
    model_status['regressor'] = bool(regressor_loaded)

    ensemble_meta = classifier_metadata.get('ensemble', {}) if classifier_metadata else {}
    member_suffixes = [m.get('suffix', '') for m in ensemble_meta.get('members', []) if isinstance(m, dict)]
    if not member_suffixes:
        member_suffixes = ['']

    def predict_classifier_probs(model_label: str):
        weights_prefix = f'is_{model_label}_classifier_final'
        probs_stack = []
        loaded = 0
        missing_paths = []
        for suffix in member_suffixes:
            suffix_str = suffix or ''
            weight_path = save_dir / f'{symbol}_{weights_prefix}{suffix_str}.weights.h5'
            if not weight_path.exists():
                continue
            arch_clf = classifier_metadata.get('architecture', {}) if isinstance(classifier_metadata, dict) else {}
            if not arch_clf:
                # Fallback to regressor architecture if classifier arch is missing
                arch_clf = target_metadata.get('architecture', {}) if isinstance(target_metadata, dict) else {}
            
            # Create model with specific name to avoid collisions
            model_name = f'{model_label}_{suffix_str or "base"}'
            model = create_binary_classifier(
                sequence_length=classifier_seq_len,
                n_features=len(used_feature_cols),
                name=model_name,
                arch=arch_clf
            )
            
            try:
                # Dummy call to build model
                _ = model(tf.random.normal((1, classifier_seq_len, len(used_feature_cols))))
                model.load_weights(str(weight_path))
                # Predict
                probs_stack.append(model.predict(X_test_clf, verbose=0).flatten())
                loaded += 1
            except Exception as e:
                print(f"❌ Failed to load/predict with classifier member {weight_path}: {e}")
                import traceback
                traceback.print_exc()
                # Continue to next ensemble member without crashing
                continue
            
        if not probs_stack:
            # No working ensemble member found -> return None to indicate classifiers unavailable
            print(f"❌ No working weights found for classifier ensemble ({model_label}). Ensemble unavailable.")
            return None
        
        avg_probs = np.mean(probs_stack, axis=0)
        print(f"Loaded {loaded} {model_label.upper()} classifier(s) for ensemble averaging")
        return avg_probs

    # Determine classifier thresholds
    # Adjusted to match classifier output distribution (mean ~0.464)
    # Previous defaults (0.30 BUY, 0.45 SELL) were incompatible with model outputs (0.46-0.51)
    try:
        if threshold_file:
            buy_threshold, sell_threshold = load_optimal_thresholds(threshold_file)
            source_str = f"Custom ({threshold_file})"
        else:
            thresholds_meta = classifier_metadata.get('thresholds', {})
            buy_threshold = float(thresholds_meta.get('buy_optimal', 0.42))
            # Use trained optimal threshold directly - no floor!
            # Previous code floored at 0.48 which blocked SELL signals when SELL probs < 0.48
            sell_threshold = float(thresholds_meta.get('sell_optimal', 0.40))
            source_str = "Default (metadata)" if 'thresholds' in classifier_metadata else 'Default'
    except Exception as e:
        print(f"⚠️ Threshold error: {e}; using defaults")
        buy_threshold, sell_threshold = 0.42, 0.48
        source_str = 'Default (fallback)'

    print(f"Classifier thresholds -> BUY: {buy_threshold:.3f}, SELL: {sell_threshold:.3f}")

    # -------------------------
    # Predictions
    # -------------------------
    print("Predicting on test set...")
    
    # Multitask Regressor (skip for GBM-only mode)
    if reg_model is not None:
        multitask_preds = reg_model.predict(X_test, verbose=0)
        y_pred_scaled = multitask_preds[0].flatten()
        
        # ===================================================================
        # PREDICTION INVERSE TRANSFORM WITH DIAGNOSTICS
        # ===================================================================
        print(f"\n=== Prediction Inverse Transform ===")
        print(f"Predictions (scaled) - range: [{y_pred_scaled.min():.6f}, {y_pred_scaled.max():.6f}]")
        print(f"Predictions (scaled) - mean: {y_pred_scaled.mean():.6f}, std: {y_pred_scaled.std():.6f}")
        
        # Apply inverse transform
        if target_scaler is not None:
            y_pred_transformed = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
            if target_transform == 'log1p':
                y_pred_orig = np.expm1(y_pred_transformed)
            else:
                y_pred_orig = y_pred_transformed
        else:
            y_pred_orig = y_pred_scaled  # No scaling needed
        
        # Report unscaled predictions
        print(f"Predictions (unscaled) - range: [{y_pred_orig.min():.6f}, {y_pred_orig.max():.6f}]")
        print(f"Predictions (unscaled) - mean: {y_pred_orig.mean():.6f}, std: {y_pred_orig.std():.6f}")
        
        # Log target scaler parameters for debugging
        if target_scaler is not None:
            try:
                print(f"Target scaler center: {target_scaler.center_[0]:.6f}")
                print(f"Target scaler scale: {target_scaler.scale_[0]:.6f}")
            except Exception:
                print(f"Target scaler: (parameters not accessible)")
        
        # Validate prediction variance
        validation = validate_prediction_variance(
            y_pred_orig, 
            min_std=0.001,  # Minimum std of 0.1%
            min_negative_pct=0.20,  # At least 20% negative
            min_positive_pct=0.20   # At least 20% positive
        )
        
        if not validation['passed']:
            for issue in validation['issues']:
                print(f"❌ {issue}")
            # Log but don't crash - allow backtest to continue for debugging
            print(f"[WARN] Predictions may have collapsed variance - check model and scaler")
        else:
            print(f"[OK] Predictions have healthy variance (std={validation['std']:.4f})")
            print(f"[OK] Distribution: {validation['pct_negative']:.1%} negative, {validation['pct_positive']:.1%} positive")
    else:
        # GBM-only mode: Initialize dummy predictions that will be replaced by GBM
        print(f"\n📊 GBM-only mode: LSTM regressor skipped - using placeholder predictions")
        y_pred_orig = np.zeros(len(X_test))  # Will be replaced by GBM predictions
        y_pred_scaled = y_pred_orig.copy()
    
    # ===================================================================
    # AUTO-MODEL VALIDATION (if enabled)
    # ===================================================================
    if auto_model:
        print("\n" + "="*60)
        print("         AUTO-MODEL VALIDATION")
        print("="*60)
        
        try:
            from inference.model_validator import ModelValidator, ValidationStatus, get_recommended_fallback
            
            validator = ModelValidator()
            lstm_validation = validator.validate_predictions(y_pred_orig, y_test, "lstm_regressor")
            
            print(f"\nLSTM Regressor Validation:")
            print(f"  Status: {lstm_validation.status.value}")
            print(f"  Health Score: {lstm_validation.health_score:.2f}")
            print(f"  Checks: {lstm_validation.checks}")
            
            if lstm_validation.warnings:
                print(f"  Warnings: {lstm_validation.warnings}")
            if lstm_validation.failures:
                print(f"  Failures: {lstm_validation.failures}")
            
            # Auto-adjust fusion mode if LSTM is unhealthy
            if lstm_validation.status in (ValidationStatus.COLLAPSED, ValidationStatus.FAILED):
                recommended_mode = get_recommended_fallback(lstm_validation)
                print(f"\n⚠️ LSTM regressor is {lstm_validation.status.value}")
                print(f"   Original fusion_mode: {fusion_mode}")
                print(f"   Auto-switching to: {recommended_mode}")
                fusion_mode = recommended_mode
                
                # Also show recommendations
                if lstm_validation.recommendations:
                    print(f"\n   Recommendations:")
                    for rec in lstm_validation.recommendations:
                        print(f"     - {rec}")
            elif lstm_validation.status == ValidationStatus.WARNING:
                recommended_mode = get_recommended_fallback(lstm_validation)
                print(f"\n⚠️ LSTM regressor has warnings")
                if fusion_mode not in ('gbm_heavy', 'balanced', 'lstm_heavy', 'gbm_only'):
                    print(f"   Consider using: {recommended_mode}")
            else:
                print(f"\n✅ LSTM regressor is healthy, using requested fusion_mode: {fusion_mode}")
                
        except ImportError as e:
            print(f"⚠️ Auto-model validation unavailable: {e}")
        except Exception as e:
            print(f"⚠️ Auto-model validation failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("="*60)
    
    # Generate prediction distribution plot - save to timestamped backtest folder
    try:
        plot_path = plot_prediction_distribution(
            predictions=y_pred_orig,
            actuals=y_test,
            symbol=symbol,
            output_dir=str(results_dir)  # Use timestamped backtest folder
        )
    except Exception as e:
        print(f"[WARN] Could not generate prediction distribution plot: {e}")

    # Evaluate Regressor
    mse = np.mean((y_test - y_pred_orig)**2)
    mae = np.mean(np.abs(y_test - y_pred_orig))
    dir_acc = np.mean(np.sign(y_test) == np.sign(y_pred_orig))
    reg_eval = {'mse': mse, 'mae': mae, 'directional_accuracy': dir_acc}
    
    print("\n--- Regressor Performance (1-day) ---")
    print(f"MAE: {mae:.6f} | Directional Accuracy: {dir_acc:.2%}")

    # Regressor Positions
    regressor_positions = compute_regressor_positions(
        y_pred_orig,
        strategy=reg_strategy,
        threshold=reg_threshold,
        scale=reg_scale,
        max_short=max_short_exposure
    )

    # =================================================================
    # GBM PREDICTIONS (optional parallel model)
    # =================================================================
    gbm_preds = None
    gbm_weight = 0.0
    
    # P0.1: TEMPORARILY DISABLE GBM FUSION
    # Root cause: LightGBM collapsed to zero variance predictions (μ=0.0000, σ=0.0000)
    # Impact: gbm_only mode produces -2.56% return (Sharpe: -1.48) vs regressor_only +10.69% (Sharpe: 3.20)
    # GBM fusion linearly degrades performance: regressor_only > lstm_heavy > balanced > gbm_heavy > gbm_only
    # P0.2 FIX APPLIED: LightGBM training with huber_delta=0.01 and no early stopping
    GBM_FUSION_DISABLED = False  # P0.2 fix applied - GBM models now have healthy variance
    
    GBM_AFFECTED_MODES = ('gbm_only', 'gbm_heavy', 'balanced', 'lstm_heavy')
    original_fusion_mode = fusion_mode  # Save for logging
    
    if GBM_FUSION_DISABLED and fusion_mode in GBM_AFFECTED_MODES:
        logger.warning(f"⚠️  GBM fusion disabled (P0.1 fix). Requested mode '{fusion_mode}' falling back to 'regressor_only'.")
        logger.warning("   Reason: GBM models collapsed to zero variance, degrading performance by up to -13%.")
        logger.warning("   Fix: Run P0.2 (train_gbm_baseline.py with huber_delta=0.01, no early stopping)")
        print(f"\n⚠️  GBM FUSION DISABLED (P0.1)")
        print(f"   Requested: {fusion_mode} → Fallback: regressor_only")
        print(f"   Reason: GBM predictions collapsed (σ≈0), causing -13% return degradation")
        fusion_mode = 'regressor_only'
    
    # Determine GBM weight based on fusion mode
    # gbm_weight_override takes priority if provided
    FUSION_MODE_GBM_WEIGHTS = {
        'gbm_only': 1.0,       # 100% GBM, 0% LSTM regressor
        'gbm_heavy': 0.7,      # 70% GBM, 30% LSTM regressor  
        'balanced': 0.5,       # 50% GBM, 50% LSTM regressor
        'lstm_heavy': 0.3,     # 30% GBM, 70% LSTM regressor
        'classifier': 0.2,     # Default GBM weight for classifier mode
        'weighted': 0.2,       # Default GBM weight
        'hybrid': 0.2,         # Default GBM weight
        'regressor': 0.2,      # Default GBM weight
        'regressor_only': 0.0, # No GBM weight for regressor_only (P0.1 fix)
    }
    
    try:
        from inference.load_gbm_models import load_gbm_models, predict_with_gbm
        gbm_bundle, gbm_meta = load_gbm_models(symbol)
        
        if gbm_bundle is not None and gbm_bundle.any_loaded():
            print(f"\n📈 GBM models available: {gbm_bundle.get_available_models()}")
            
            # Get features for GBM (use prepared_df which has aligned features)
            gbm_feature_cols = gbm_bundle.feature_columns
            
            # We need to align the test data with GBM features
            # X_test_clf has the same length as test data, but we need unscaled features
            # Get from engineered_df using the test date range
            if 'prepared_df' in locals() and prepared_df is not None:
                # Use prepared_df for test period
                test_features = prepared_df[gbm_feature_cols].iloc[-len(X_test):].values
                gbm_preds_dict = predict_with_gbm(gbm_bundle, test_features, return_both=True)
                
                # P0.3-FIX: Detect GBM model collapse and use only healthy models
                # Models are "collapsed" if:
                # 1. std < 0.0005 (predictions too uniform)
                # 2. pos_pct > 90% (predictions too biased - should be ~50% for random)
                # UPDATED: LightGBM significantly outperforms XGBoost on AAPL:
                # LightGBM: IC=0.652, Dir Acc=65.8%, Variance=0.00569 (healthy)
                # XGBoost:  IC=0.401, Dir Acc=55.1%, Variance=0.00209 (collapsed), 87% positive bias
                # Prefer LightGBM when available.
                MIN_GBM_STD = 0.0005  # Healthy models should have std > 0.05%
                MAX_POSITIVE_PCT = 90.0  # Healthy models should have <90% positive
                healthy_preds = {}
                for model_name, preds in gbm_preds_dict.items():
                    model_std = np.std(preds)
                    model_positive_pct = np.sum(preds > 0) / len(preds) * 100
                    
                    # P0.3b: Stricter criteria - require both std AND balanced predictions
                    is_healthy = model_std >= MIN_GBM_STD and model_positive_pct < MAX_POSITIVE_PCT
                    
                    if is_healthy:
                        healthy_preds[model_name] = preds
                        print(f"   ✓ {model_name}: HEALTHY (std={model_std:.6f}, {model_positive_pct:.1f}% positive)")
                    else:
                        # Model is biased but may still have relative signal value after demeaning
                        reason = []
                        if model_std < MIN_GBM_STD:
                            reason.append(f"low_std={model_std:.6f}")
                        if model_positive_pct >= MAX_POSITIVE_PCT:
                            reason.append(f"biased={model_positive_pct:.1f}%_positive")
                        print(f"   ⚠️ {model_name}: BIASED ({', '.join(reason)}) - using with demeaning")
                        # Still use biased models - demeaning will correct the bias
                        healthy_preds[model_name] = preds
                
                # Use only healthy GBM predictions (or all if none are truly healthy)
                if healthy_preds:
                    if len(healthy_preds) > 1:
                        # P0.3c: Prefer LightGBM over XGBoost (better IC: 0.652 vs 0.401, direction accuracy: 65.8% vs 55.1%)
                        if 'lgb' in healthy_preds:
                            gbm_preds = healthy_preds['lgb']
                            print(f"   ✓ Using LightGBM only (preferred over XGBoost)")
                        elif 'xgb' in healthy_preds:
                            gbm_preds = healthy_preds['xgb']
                            print(f"   ✓ Using XGBoost only (LightGBM not available)")
                        else:
                            gbm_preds = list(healthy_preds.values())[0]
                    else:
                        gbm_preds = list(healthy_preds.values())[0]
                else:
                    # All GBM models collapsed - use XGBoost as fallback (demeaning will help)
                    print(f"   ⚠️ ALL GBM MODELS COLLAPSED - using XGBoost as fallback")
                    gbm_preds = gbm_preds_dict.get('xgb', list(gbm_preds_dict.values())[0])
                
                # Set GBM weight based on: override > fusion_mode > default
                if gbm_weight_override is not None:
                    gbm_weight = gbm_weight_override
                    print(f"   Using GBM weight override: {gbm_weight:.0%}")
                else:
                    gbm_weight = FUSION_MODE_GBM_WEIGHTS.get(fusion_mode, 0.2)
                
                print(f"   GBM predictions: range=[{gbm_preds.min():.6f}, {gbm_preds.max():.6f}]")
                print(f"   GBM predictions: mean={gbm_preds.mean():.6f}, std={gbm_preds.std():.6f}")
                print(f"   GBM weight in fusion: {gbm_weight:.0%} (fusion_mode={fusion_mode})")
                
                # For GBM-only mode, replace y_pred_orig entirely with GBM predictions
                if fusion_mode in GBM_ONLY_MODES:
                    y_pred_orig = gbm_preds.copy()
                    regressor_positions = compute_regressor_positions(
                        y_pred_orig,
                        strategy=reg_strategy,
                        threshold=reg_threshold,
                        scale=reg_scale,
                        max_short=max_short_exposure
                    )
                    print(f"   ✓ GBM-only mode: Using GBM predictions as y_pred_orig")
                
                # For GBM fusion modes (not pure gbm_only), fuse GBM with LSTM using weight
                elif fusion_mode in ('gbm_heavy', 'balanced', 'lstm_heavy') and gbm_weight > 0:
                    # P0.3-ENHANCED: Detect LSTM collapse and auto-fallback to GBM-only
                    lstm_pred_orig = y_pred_orig.copy()
                    lstm_std = np.std(lstm_pred_orig)
                    lstm_positive_pct = np.sum(lstm_pred_orig > 0) / len(lstm_pred_orig) * 100
                    gbm_std = np.std(gbm_preds)
                    
                    # P0.3d REVISED: LSTM collapse criteria - focus on VARIANCE not bias
                    # Stock markets have inherent positive drift (6-10% annually), so 
                    # models predicting 60-90% positive is NORMAL for a bull market.
                    # We only consider LSTM "collapsed" if:
                    # 1. std < 0.002 (predictions too uniform = variance collapse)
                    # 2. OR (std < 0.005 AND positive_pct > 98%) - near-zero variance with extreme bias
                    # Note: A healthy model should have std > 0.003 for meaningful signal
                    MIN_LSTM_STD = 0.002  # Lowered from 0.003 - allow more variance flexibility
                    MIN_LSTM_STD_STRICT = 0.005  # For combined check with bias
                    MAX_LSTM_POSITIVE_PCT = 98.0  # Raised from 80% - allow natural market bias
                    
                    # REVISED LOGIC: Only collapse on extreme variance issues
                    LSTM_COLLAPSED = (lstm_std < MIN_LSTM_STD) or \
                                   (lstm_std < MIN_LSTM_STD_STRICT and lstm_positive_pct > MAX_LSTM_POSITIVE_PCT)
                    
                    # P0 FIX: Allow A/B testing with disable_fallback flag
                    if LSTM_COLLAPSED and not disable_fallback:
                        # LSTM collapsed or too biased - USE GBM ONLY (ignore LSTM completely)
                        print(f"   ⚠️ LSTM BIASED: std={lstm_std:.6f}, {lstm_positive_pct:.1f}% positive (>{MAX_LSTM_POSITIVE_PCT}%)")
                        print(f"   ⚠️ {fusion_mode} → Falling back to GBM-only (100% GBM weight)")
                        y_pred_orig = gbm_preds.copy()  # Use pure GBM
                        gbm_weight = 1.0
                    elif LSTM_COLLAPSED and disable_fallback:
                        # Fallback disabled - use raw fusion despite collapse
                        print(f"   ⚠️ LSTM BIASED: std={lstm_std:.6f}, {lstm_positive_pct:.1f}% positive")
                        print(f"   ⚠️ --disable-fallback: Using raw fusion anyway (A/B test mode)")
                        y_pred_orig = (1.0 - gbm_weight) * lstm_pred_orig + gbm_weight * gbm_preds
                    else:
                        # LSTM healthy - perform normal fusion
                        y_pred_orig = (1.0 - gbm_weight) * lstm_pred_orig + gbm_weight * gbm_preds
                        print(f"   ✓ LSTM HEALTHY: std={lstm_std:.6f}, {lstm_positive_pct:.1f}% positive")
                    
                    regressor_positions = compute_regressor_positions(
                        y_pred_orig,
                        strategy=reg_strategy,
                        threshold=reg_threshold,
                        scale=reg_scale,
                        max_short=max_short_exposure
                    )
                    print(f"   ✓ {fusion_mode} mode: LSTM ({(1-gbm_weight):.0%}) + GBM ({gbm_weight:.0%})")
                    print(f"     Final pred: mean={np.mean(y_pred_orig):.6f}, std={np.std(y_pred_orig):.6f}")
                
                # Add to model status
                model_status['gbm'] = True
    except Exception as e:
        print(f"⚠️ GBM loading/prediction failed: {e}")
        gbm_preds = None
        gbm_weight = 0.0
        model_status['gbm'] = False

    # =================================================================
    # CLASSIFIER LOADING (skipped for regressor-only and GBM fusion modes)
    # =================================================================
    # Binary classifiers have been DEPRECATED and REMOVED (December 2025)
    # All fusion modes are now regressor-only modes - classifiers are never loaded
    REGRESSOR_ONLY_MODES = ('stacking', 'regressor_only', 'regressor', 'gbm_only', 'gbm_heavy', 'balanced', 'lstm_heavy')
    classifier_needed = False  # Classifiers deprecated - never load them
    
    
    # P0.3-FIX: Calculate market regime globally for use in fusion logic
    # This enables aggressive short selling in bear markets for all modes
    regimes = detect_market_regime(test_prices)
    
    if classifier_needed:
        # Classifiers
        buy_probs = predict_classifier_probs('buy')
        sell_probs = predict_classifier_probs('sell')

        classifier_available = True
        # If either buy or sell ensemble failed to load, treat classifiers as unavailable
        if buy_probs is None or sell_probs is None:
            classifier_available = False
            print("⚠️ One or more classifier ensembles failed to load. Classifier-based signals will be disabled.")
            # Provide neutral arrays so downstream code can run without branching
            buy_probs = np.zeros(len(y_test), dtype=float)
            sell_probs = np.zeros(len(y_test), dtype=float)

        # Update model status
        model_status['classifiers'] = bool(classifier_available)

        # Apply thresholds
        buy_pred_initial = (buy_probs >= buy_threshold).astype(int)
        sell_pred_initial = (sell_probs >= sell_threshold).astype(int)
        
        # Log threshold application results
        buy_signals = buy_pred_initial
        sell_signals = sell_pred_initial
        print(f"Threshold application: BUY fired {buy_signals.sum()}, SELL fired {sell_signals.sum()} out of {len(buy_signals)} days")

        buy_pred = buy_pred_initial.copy()
        sell_pred = sell_pred_initial.copy()
        
        # Regime Filtering - Softened version
        # Only filter out weak signals (low probability) against the regime
        # Strong signals are allowed through even against the trend
        # regimes calculated above
        regime_filter_threshold = 0.55  # Only filter if prob < 55%
        
        for i in range(len(buy_pred)):
            if regimes[i] == 1 and sell_pred[i] == 1:  # Bullish regime
                # Only filter weak SELL signals
                if sell_probs[i] < regime_filter_threshold:
                    sell_pred[i] = 0
            elif regimes[i] == -1 and buy_pred[i] == 1:  # Bearish regime
                # Only filter weak BUY signals
                if buy_probs[i] < regime_filter_threshold:
                    buy_pred[i] = 0
    else:
        # regressor_only or regressor mode - skip classifier loading entirely
        print(f"\n=== REGRESSOR-ONLY MODE ===")
        print(f"Skipping classifier loading (fusion_mode={fusion_mode})")
        classifier_available = False
        model_status['classifiers'] = False
        buy_probs = np.zeros(len(y_test), dtype=float)
        sell_probs = np.zeros(len(y_test), dtype=float)
        buy_pred_initial = np.zeros(len(y_test), dtype=int)
        sell_pred_initial = np.zeros(len(y_test), dtype=int)
        buy_pred = buy_pred_initial.copy()
        sell_pred = sell_pred_initial.copy()
        # regimes calculated above

    # Fusion with directional comparison (only for classifier modes)
    if classifier_needed:
        # Given high correlation (0.79) between BUY/SELL classifiers,
        # use direct probability comparison instead of threshold-based margins
        final_signals = np.zeros(len(buy_pred), dtype=int)
        buy_yes = buy_pred == 1
        sell_yes = sell_pred == 1
        
        # Clear wins: one fires, other doesn't
        final_signals[buy_yes & ~sell_yes] = 1
        final_signals[sell_yes & ~buy_yes] = -1
        
        # Conflict resolution: both fire - use direct probability comparison
        # This works better when classifiers are correlated
        both_fire = buy_yes & sell_yes
        if np.any(both_fire):
            # Normalize probabilities to z-scores for fair comparison
            buy_mean, buy_std = np.mean(buy_probs), np.std(buy_probs)
            sell_mean, sell_std = np.mean(sell_probs), np.std(sell_probs)
            
            # Z-score: how many standard deviations above mean
            buy_z = (buy_probs - buy_mean) / max(buy_std, 0.001)
            sell_z = (sell_probs - sell_mean) / max(sell_std, 0.001)
            
            # Decision: pick higher z-score (more unusual for that classifier)
            # Small margin to avoid noise
            z_margin = 0.3  # About 1/3 of a standard deviation
            buy_wins = buy_z > sell_z + z_margin
            sell_wins = sell_z > buy_z + z_margin
            
            conflict_indices = np.where(both_fire)[0]
            for idx in conflict_indices:
                if buy_wins[idx]:
                    final_signals[idx] = 1
                elif sell_wins[idx]:
                    final_signals[idx] = -1
                # else: stays 0 (HOLD) - z-scores too close
            
            print(f"Conflict resolution (z-score method): {np.sum(both_fire)} days with both signals")
            print(f"  BUY z-scores: mean={np.mean(buy_z[both_fire]):.2f}, range=[{np.min(buy_z[both_fire]):.2f}, {np.max(buy_z[both_fire]):.2f}]")
            print(f"  SELL z-scores: mean={np.mean(sell_z[both_fire]):.2f}, range=[{np.min(sell_z[both_fire]):.2f}, {np.max(sell_z[both_fire]):.2f}]")
            print(f"  BUY won: {np.sum(buy_wins[both_fire])}, SELL won: {np.sum(sell_wins[both_fire])}, HOLD (tie): {np.sum(both_fire) - np.sum(buy_wins[both_fire]) - np.sum(sell_wins[both_fire])}")
        
        buy_count = (final_signals == 1).sum()
        sell_count = (final_signals == -1).sum()
        hold_count = (final_signals == 0).sum()

        # =================================================================
        # SIGNAL DIAGNOSTICS - Visibility into signal generation (classifier modes)
        # =================================================================
        print("\n" + "=" * 50)
        print("           SIGNAL DIAGNOSTICS")
        print("=" * 50)
        
        # Classifier output statistics
        print("Classifier Outputs:")
        print(f"  BUY:  mean={np.mean(buy_probs):.4f}, std={np.std(buy_probs):.4f}, range=[{np.min(buy_probs):.4f}, {np.max(buy_probs):.4f}]")
        print(f"  SELL: mean={np.mean(sell_probs):.4f}, std={np.std(sell_probs):.4f}, range=[{np.min(sell_probs):.4f}, {np.max(sell_probs):.4f}]")
        
        # Regressor output statistics
        print(f"Regressor: mean={np.mean(y_pred_orig):.6f}, std={np.std(y_pred_orig):.8f}, range=[{np.min(y_pred_orig):.6f}, {np.max(y_pred_orig):.6f}]")
        
        # Signal counts
        print("Signals Generated:")
        print(f"  BUY: {buy_count}, SELL: {sell_count}, HOLD: {hold_count}")
        
        # Thresholds used
        print(f"Thresholds: BUY={buy_threshold:.3f}, SELL={sell_threshold:.3f} (source: {source_str})")
        
        # Pre-filter vs post-filter comparison
        buy_pre_filter = buy_pred_initial.sum()
        sell_pre_filter = sell_pred_initial.sum()
        print(f"Pre-regime filter: BUY={buy_pre_filter}, SELL={sell_pre_filter}")
        print(f"Post-regime filter: BUY={buy_count}, SELL={sell_count}")
        
        print("=" * 50 + "\n")
    else:
        # P0.1 FIX: Regressor-only mode - generate signals from predictions
        # Previously this was hardcoded to zeros, causing all signals to appear as 0
        # even though positions were being computed and trades executed
        if 'y_pred_orig' in locals() and y_pred_orig is not None:
            # Use adaptive threshold based on prediction statistics
            # For GBM models with small magnitude predictions (~0.001), 
            # fixed 1% threshold would miss all signals
            pred_std = np.std(y_pred_orig)
            adaptive_threshold = max(0.5 * pred_std, 0.0001)  # At least 0.01%, or half std
            
            final_signals = compute_signals_from_predictions(
                predictions=y_pred_orig,
                buy_threshold=adaptive_threshold,
                sell_threshold=adaptive_threshold,
                confidence_scores=None,  # Use all predictions
                min_confidence=0.0
            )
            print(f"\n✓ P0.1 FIX: Generated signals from predictions")
            print(f"  Prediction std: {pred_std:.6f}")
            print(f"  Adaptive threshold: {adaptive_threshold:.6f} (vs fixed {reg_threshold:.4f})")
            print(f"  Signals: BUY={np.sum(final_signals==1)}, SELL={np.sum(final_signals==-1)}, HOLD={np.sum(final_signals==0)}")
        else:
            # Fallback if no predictions available
            final_signals = np.zeros(len(y_test), dtype=int)
            print(f"\n⚠️ No predictions available, using zero signals")
        
        buy_count = int((final_signals == 1).sum())
        sell_count = int((final_signals == -1).sum())
        hold_count = int((final_signals == 0).sum())

    # Calculate Positions (only needed for classifier modes)
    if classifier_needed:
        classifier_positions = compute_classifier_positions(
            final_signals, buy_probs, sell_probs, max_short=max_short_exposure
        )
        
        # Position statistics (after classifier_positions are computed)
        buy_mask = final_signals == 1
        sell_mask = final_signals == -1
        if buy_mask.any():
            mean_buy_position = np.mean(classifier_positions[buy_mask])
            print(f"Position stats: Mean BUY position = {mean_buy_position:.4f}")
        else:
            print("Position stats: No BUY signals fired")
        if sell_mask.any():
            mean_sell_position = np.mean(classifier_positions[sell_mask])
            print(f"Position stats: Mean SELL position = {mean_sell_position:.4f}")
        else:
            print("Position stats: No SELL signals fired")
    else:
        # Regressor-only: no classifier positions needed
        classifier_positions = np.zeros(len(y_test), dtype=float)

    # Determine boolean flags for availability (used for graceful degradation)
    regressor_available = bool(model_status.get('regressor', False))
    classifiers_available = bool(model_status.get('classifiers', False))
    quantile_available = bool(model_status.get('quantile', False))
    tft_available = bool(model_status.get('tft', False))
    gbm_available = bool(model_status.get('gbm', False))

    available = {
        'regressor': regressor_available,
        'classifiers': classifiers_available,
        'quantile': quantile_available,
        'tft': tft_available,
        'gbm': gbm_available
    }

    # Initialize unified confidence scorer and model availability map
    confidence_scorer = ConfidenceScorer(model_weights={
        'classifier': 0.35,
        'regressor': 0.30,
        'quantile': 0.20,
        'tft': 0.15
    })
    model_availability = {
        'classifier': classifiers_available,
        'regressor': regressor_available,
        'quantile': quantile_available,
        'tft': tft_available
    }

    # Choose the best fusion mode based on requested mode and available models
    best_mode = select_best_available_fusion_mode(fusion_mode, available)
    if best_mode is None:
        print("❌ No suitable models available for fusion. Need at least the regressor or classifiers to proceed.")
        # Attach availability to model_status and return minimal result for debugging
        model_status.update(available)
        return {
            'error': 'no_models_available',
            'model_status': model_status
        }

    if best_mode != fusion_mode:
        print(f"⚠️ Requested fusion mode '{fusion_mode}' not fully supported by loaded models; falling back to '{best_mode}'")
    fusion_mode = best_mode

    # Log which models are available for this run
    print("Model availability:")
    for k, v in available.items():
        print(f"  {k}: {'✅' if v else '❌'}")

    # Define regressor-like modes that don't use classifiers
    REGRESSOR_LIKE_MODES = ('regressor_only', 'gbm_only', 'gbm_heavy', 'balanced', 'lstm_heavy')

    # Get fused positions (with confidence for regressor_only and GBM modes)
    regressor_confidence = None
    if fusion_mode in REGRESSOR_LIKE_MODES:
        fused_positions, regressor_confidence = fuse_positions(
            fusion_mode,
            classifier_positions=classifier_positions,
            regressor_positions=regressor_positions,
            final_signals=final_signals,
            regime=regimes,
            max_short=max_short_exposure,
            regressor_preds=(y_pred_orig if 'y_pred_orig' in locals() else None),
            atr_percent=test_atr_percent,
            return_confidence=True
        )
    else:
        fused_positions = fuse_positions(
            fusion_mode,
            classifier_positions=classifier_positions,
            regressor_positions=regressor_positions,
            final_signals=final_signals,
            max_short=max_short_exposure,
            regressor_preds=(y_pred_orig if 'y_pred_orig' in locals() else None),
            atr_percent=test_atr_percent
        )

    # Print signal diagnostics for regressor_only and GBM modes
    if fusion_mode in REGRESSOR_LIKE_MODES:
        print(f"\n=== SIGNAL DIAGNOSTICS ({fusion_mode.upper()} MODE) ===")
        print(f"Fusion mode: {fusion_mode} (classifiers ignored)")
        if 'y_pred_orig' in locals():
            print(f"Predictions: mean={np.mean(y_pred_orig):.5f}, std={np.std(y_pred_orig):.5f}")
        print(f"Positions: mean={np.mean(fused_positions):.3f} ({np.mean(fused_positions)*100:.1f}% avg exposure), std={np.std(fused_positions):.3f}")
        long_count = np.sum(fused_positions > 0)
        short_count = np.sum(fused_positions < 0)
        neutral_count = np.sum(fused_positions == 0)
        print(f"Signals Generated:")
        print(f"  Long positions (>0): {long_count}")
        print(f"  Short positions (<0): {short_count}")
        print(f"  Neutral/HOLD: {neutral_count}")
        # Confidence filtering diagnostics
        if regressor_confidence is not None:
            low_conf_count = np.sum(regressor_confidence < 0.3)
            high_conf_count = np.sum(regressor_confidence >= 0.3)
            print(f"Confidence Filtering (MIN_CONFIDENCE=0.3):")
            print(f"  High confidence (>=0.3): {high_conf_count}")
            print(f"  Filtered out (<0.3): {low_conf_count}")
            print(f"  Avg confidence: {np.mean(regressor_confidence):.3f}")

    # For regressor_only and GBM modes, use fused_positions directly (skip hybrid which uses classifiers)
    if fusion_mode in REGRESSOR_LIKE_MODES:
        strategy_positions = fused_positions
        # Update signal counts from actual positions (these were initialized to 0 since no classifiers)
        buy_count = int(np.sum(fused_positions > 0))
        sell_count = int(np.sum(fused_positions < 0))
        hold_count = int(np.sum(fused_positions == 0))
        # Build a minimal trade_log for regressor_only/GBM mode
        try:
            rows = []
            for i in range(len(test_prices)):
                regp = float(y_pred_orig[i]) if ('y_pred_orig' in locals() and i < len(y_pred_orig)) else None
                pos = float(fused_positions[i]) if i < len(fused_positions) else 0.0
                price = float(test_prices[i]) if i < len(test_prices) else None
                date = test_dates[i] if i < len(test_dates) else i
                conf = float(regressor_confidence[i]) if (regressor_confidence is not None and i < len(regressor_confidence)) else None
                # Determine action from position (regressor_only/GBM has no classifier signals)
                action = 'BUY' if pos > 0 else ('SELL' if pos < 0 else 'HOLD')
                rows.append({
                    'index': i,
                    'date': date,
                    'price': price,
                    'action': action,
                    'buy_prob': None,  # No classifier probs in regressor_only/GBM mode
                    'sell_prob': None,
                    'regressor_pred': regp,
                    'unified_confidence': conf,  # Use actual confidence score
                    'confidence_tier': 'high' if conf is not None and conf >= 0.3 else 'low',
                    'confidence_attribution': 'regressor_only',
                    'position_shares': pos,
                    'reasoning': f'regressor_pred={regp:.5f}, conf={conf:.3f}' if regp is not None and conf is not None else None
                })
            out_trade_log = pd.DataFrame(rows)
            print("✓ Created trade_log for regressor_only mode")

        except Exception as e:
            print(f"⚠️ Failed to create trade_log for regressor_only: {e}")
            out_trade_log = None
    else:
        # Prefer the full hybrid position generator which returns diagnostic trade log
        try:
            positions_from_hybrid, trade_log_df = compute_hybrid_positions(
                final_signals,
                buy_probs,
                sell_probs,
                test_prices,
                regressor_positions=regressor_positions,
                quantile_preds=None,
                tft_forecasts=None,
                regimes=regimes,
                atr_percent=test_atr_percent,
                vol_ratio_5_20=test_vol_ratio,
                verbose=False,
                dates=test_dates,
                confidence_scorer=confidence_scorer,
                model_availability=model_availability,
                regressor_preds=(y_pred_orig if 'y_pred_orig' in locals() else None),
                gbm_preds=gbm_preds,
                gbm_weight=gbm_weight,
                out_dir=results_dir
            )
            strategy_positions = positions_from_hybrid
            # attach trade log to output for later inspection
            out_trade_log = trade_log_df
        except Exception as e:
            print(f"⚠️ compute_hybrid_positions failed, falling back to fused positions: {e}")
            import traceback
            traceback.print_exc()
            strategy_positions = fused_positions
            # Best-effort: build a minimal trade_log_df so downstream analysis and CSV/JSON are generated
            try:
                rows = []
                for i in range(len(test_prices)):
                    sig = int(final_signals[i]) if i < len(final_signals) else 0
                    action = 'BUY' if sig > 0 else ('SELL' if sig < 0 else 'HOLD')
                    buyp = float(buy_probs[i]) if i < len(buy_probs) else float('nan')
                    sellp = float(sell_probs[i]) if i < len(sell_probs) else float('nan')
                    regp = float(y_pred_orig[i]) if ('y_pred_orig' in locals() and i < len(y_pred_orig)) else None
                    price = float(test_prices[i]) if i < len(test_prices) else None
                    date = test_dates[i] if i < len(test_dates) else i
                    # fallback unified confidence = buy or sell prob (signal-aware)
                    uni = buyp if sig > 0 else (sellp if sig < 0 else float('nan'))
                    rows.append({
                        'index': i,
                        'date': date,
                        'price': price,
                        'action': action,
                        'buy_prob': buyp,
                        'sell_prob': sellp,
                        'regressor_pred': regp,
                        'unified_confidence': uni,
                        'confidence_tier': None,
                        'confidence_attribution': None,
                        'reasoning': None
                    })
                out_trade_log = pd.DataFrame(rows)
                print("✓ Created fallback trade_log for analysis (compute_hybrid_positions failed)")
                # Run confidence analysis and save CSV/JSON here to ensure artifacts exist
                try:
                    # Build arrays for analysis
                    try:
                        returns_arr = np.concatenate([[np.nan], np.diff(test_prices) / test_prices[:-1]]) if len(test_prices) > 1 else np.array([np.nan] * len(test_prices))
                    except Exception:
                        returns_arr = np.array([np.nan] * len(test_prices))

                    buy_for_analysis = out_trade_log['unified_confidence'].where(~out_trade_log['unified_confidence'].isna(), out_trade_log['buy_prob']).to_numpy(dtype=float)
                    sell_for_analysis = out_trade_log['unified_confidence'].where(~out_trade_log['unified_confidence'].isna(), out_trade_log['sell_prob']).to_numpy(dtype=float)
                    signals_arr_local = np.asarray(final_signals, dtype=int)
                    analysis = analyze_signal_confidence(buy_for_analysis, sell_for_analysis, strategy_positions, returns_arr, signals_arr_local)

                    out_json = results_dir / 'confidence_analysis.json'
                    with open(out_json, 'w', encoding='utf-8') as f:
                        json.dump(analysis, f, ensure_ascii=False, indent=2)
                    print(f"Saved fallback confidence analysis to: {out_json}")

                    # save trade log csv
                    # compute confidence column
                    def _row_conf(r):
                        try:
                            if not pd.isna(r.get('unified_confidence')):
                                return float(r.get('unified_confidence'))
                        except Exception:
                            pass
                        if r.get('action') == 'BUY':
                            return float(r.get('buy_prob')) if r.get('buy_prob') is not None else float('nan')
                        if r.get('action') == 'SELL':
                            return float(r.get('sell_prob')) if r.get('sell_prob') is not None else float('nan')
                        return float('nan')

                    out_trade_log['confidence'] = out_trade_log.apply(_row_conf, axis=1)
                    cbuckets = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
                    clabels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
                    conf_vals = out_trade_log['confidence'].to_numpy(dtype=float)
                    inds = np.digitize(conf_vals, bins=cbuckets, right=False) - 1
                    inds = np.clip(inds, 0, len(clabels) - 1)
                    out_trade_log['confidence_bucket'] = [clabels[i] if not np.isnan(v) else None for i, v in zip(inds, conf_vals)]
                    
                    out_csv = results_dir / 'confidence_trade_log.csv'
                    out_trade_log.to_csv(out_csv, index=False)
                    print(f"Saved fallback trade log CSV to: {out_csv}")
                except Exception as ae:
                    print(f"Failed to run fallback confidence analysis: {ae}")
            except Exception:
                out_trade_log = None

    # If we have a proper trade_log from hybrid path, run confidence analysis and save artifacts
    try:
        if 'out_trade_log' in locals() and isinstance(out_trade_log, pd.DataFrame) and not out_trade_log.empty:
            # Ensure unified_confidence present
            try:
                conf_col = out_trade_log['unified_confidence']
            except Exception:
                conf_col = None

            # Build arrays for analysis
            try:
                returns_arr = np.concatenate([[np.nan], np.diff(test_prices) / test_prices[:-1]]) if len(test_prices) > 1 else np.array([np.nan] * len(test_prices))
            except Exception:
                returns_arr = np.array([np.nan] * len(test_prices))

            buy_for_analysis = out_trade_log['unified_confidence'].where(~out_trade_log['unified_confidence'].isna(), out_trade_log.get('buy_prob')) if 'unified_confidence' in out_trade_log.columns else out_trade_log.get('buy_prob')
            sell_for_analysis = out_trade_log['unified_confidence'].where(~out_trade_log['unified_confidence'].isna(), out_trade_log.get('sell_prob')) if 'unified_confidence' in out_trade_log.columns else out_trade_log.get('sell_prob')
            buy_arr = np.asarray(buy_for_analysis.fillna(np.nan) if isinstance(buy_for_analysis, pd.Series) else buy_for_analysis, dtype=float)
            sell_arr = np.asarray(sell_for_analysis.fillna(np.nan) if isinstance(sell_for_analysis, pd.Series) else sell_for_analysis, dtype=float)
            signals_arr_local = np.asarray(final_signals, dtype=int)

            analysis = analyze_signal_confidence(buy_arr, sell_arr, strategy_positions, returns_arr, signals_arr_local)

            out_json = results_dir / 'confidence_analysis.json'
            with open(out_json, 'w', encoding='utf-8') as f:
                json.dump(analysis, f, ensure_ascii=False, indent=2)
            print(f"Saved confidence analysis to: {out_json}")

            # Save enriched trade log CSV with confidence bucket
            try:
                # compute confidence column
                def _row_conf(r):
                    try:
                        if not pd.isna(r.get('unified_confidence')):
                            return float(r.get('unified_confidence'))
                    except Exception:
                        pass
                    if r.get('action') == 'BUY':
                        return float(r.get('buy_prob')) if r.get('buy_prob') is not None else float('nan')
                    if r.get('action') == 'SELL':
                        return float(r.get('sell_prob')) if r.get('sell_prob') is not None else float('nan')
                    return float('nan')

                out_trade_log['confidence'] = out_trade_log.apply(_row_conf, axis=1)
                cbuckets = [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]
                clabels = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
                conf_vals = out_trade_log['confidence'].to_numpy(dtype=float)
                inds = np.digitize(conf_vals, bins=cbuckets, right=False) - 1
                inds = np.clip(inds, 0, len(clabels) - 1)
                out_trade_log['confidence_bucket'] = [clabels[i] if not np.isnan(v) else None for i, v in zip(inds, conf_vals)]
                out_csv = results_dir / 'confidence_trade_log.csv'
                out_trade_log.to_csv(out_csv, index=False)
                print(f"Saved enriched trade log CSV to: {out_csv}")
            except Exception as _e:
                print(f"Failed to save enriched trade log CSV: {_e}")
    except Exception as _ci:
        print(f"⚠️ Confidence analysis (hybrid) failed: {_ci}")

    strat_ret = strategy_positions * y_test
    
    # Metrics
    cum_return = np.prod(1 + strat_ret) - 1
    avg_daily = np.mean(strat_ret)
    std_daily = np.std(strat_ret)
    sharpe = (avg_daily / std_daily) * np.sqrt(252) if std_daily > 0 else 0
    
    print("\n=== BACKTEST RESULTS ===")
    print(f"Cumulative Return: {cum_return:.2%}")
    print(f"Sharpe: {sharpe:.3f}")
    print(f"Signals: BUY={buy_count}, SELL={sell_count}")

    # Advanced Backtester
    try:
        # Phase 5: Passing margin and borrow rate
        backtester = AdvancedBacktester(
            initial_capital=10_000,
            margin_requirement=margin_requirement,
            borrow_rate=borrow_rate
        )
        bt_results = backtester.backtest_with_positions(
            dates=test_dates,
            prices=test_prices,
            returns=y_test,
            positions=strategy_positions,
            max_long=1.0,
            max_short=max_short_exposure
        )
        backtester.print_metrics_table()
        # Backtest completed; benchmark metrics are available in `bt_results.metrics` and
        # advanced buy-hold equity curve is in `bt_results.buy_hold_equity`.
        # Produce an advanced plot that includes the buy-and-hold curve (best-effort)
        try:
            adv_png = results_dir / (report_name + '_advanced.png') if 'results_dir' in locals() and 'report_name' in locals() else None
            backtester.plot_results(save_path=adv_png, title=f"Advanced Backtest: {symbol}")
            if adv_png is not None:
                print(f"✓ Saved advanced backtest plot: {adv_png}")
        except Exception as e:
            print(f"⚠️ Failed to produce advanced backtest plot: {e}")
    except Exception as e:
        print(f"⚠️ Advanced backtester failed: {e}")

    # -------------------------
    # Quantile Regressor Integration (Uncertainty-Aware Fusion)
    # -------------------------
    if use_quantile:
        print("\n" + "="*60)
        print("  QUANTILE REGRESSOR INTEGRATION")
        print("="*60)
        quantile_loaded = False
        # Initialize defaults so variables exist even if loading fails
        q_meta = {
            'sequence_length': 60,
            'n_features': 147,
            'scaling_method': 'robust'
        }
        q_feat_scaler = None
        q_target_scaler = None
        q_model = None
        try:
            # Load artifacts
            q_model_dir = save_dir / f'{symbol}_quantile_regressor_model'
            q_weights = save_dir / f'{symbol}_quantile_regressor.weights.h5'
            q_meta_path = save_dir / f'{symbol}_quantile_metadata.pkl'
            q_feat_scaler_path = save_dir / f'{symbol}_quantile_feature_scaler.pkl'
            q_target_scaler_path = save_dir / f'{symbol}_quantile_target_scaler.pkl'

            # Try to load saved metadata/scalers; update defaults rather than overwrite
            if q_meta_path.exists():
                try:
                    loaded_meta = safe_load_pickle(q_meta_path)
                    if isinstance(loaded_meta, dict):
                        q_meta.update(loaded_meta)
                except Exception as e:
                    print(f"⚠️ Failed to load quantile metadata: {e}")

            if q_feat_scaler_path.exists():
                try:
                    q_feat_scaler = safe_load_pickle(q_feat_scaler_path)
                except Exception as e:
                    print(f"⚠️ Failed to load quantile feature scaler: {e}")

            if q_target_scaler_path.exists():
                try:
                    q_target_scaler = safe_load_pickle(q_target_scaler_path)
                except Exception as e:
                    print(f"⚠️ Failed to load quantile target scaler: {e}")

            # Import QuantileRegressor and prefer the standardized load() API
            try:
                from models.quantile_regressor import QuantileRegressor
            except ImportError:
                import sys
                repo_root = Path(__file__).parent
                if str(repo_root) not in sys.path:
                    sys.path.insert(0, str(repo_root))
                from models.quantile_regressor import QuantileRegressor

            q_model = None
            # Prefer SavedModel dir (full model export) first
            if q_model_dir.exists():
                try:
                    q_model = keras.models.load_model(str(q_model_dir))
                    print("✓ Loaded Quantile Regressor from model directory")
                except Exception as e:
                    print(f"⚠️ Failed to load full model: {e}. Will try standardized QuantileRegressor.load() or weights/backups.")

            # Try the new standardized loader provided by the class
            if q_model is None:
                try:
                    q_model = QuantileRegressor.load(symbol, save_dir=save_dir)
                    print("✓ Loaded QuantileRegressor via QuantileRegressor.load()")
                except Exception as le:
                    print(f"⚠️ QuantileRegressor.load() failed: {le}. Falling back to manual reconstruction and weight loading.")
                    # Fall back to manual reconstruction and existing heuristics
                    q_model = QuantileRegressor(
                        n_features=q_meta.get('n_features', 147),
                        sequence_length=q_meta.get('sequence_length', 60),
                        d_model=q_meta.get('d_model', 128),
                        n_transformer_blocks=q_meta.get('n_transformer_blocks', 4),
                        n_heads=q_meta.get('n_heads', 4),
                        dropout=q_meta.get('dropout', 0.2)
                    )
                    # build
                    _ = q_model(tf.zeros((1, q_meta.get('sequence_length', 60), q_meta.get('n_features', 147))))

                loaded_from_backup = False
                backup_pickle = save_dir / f'{symbol}_quantile_regressor.weights.pkl'
                if backup_pickle.exists():
                    try:
                        import pickle as _pickle
                        with open(backup_pickle, 'rb') as bf:
                            data = _pickle.load(bf)
                        vals = data.get('values')
                        if vals is not None:
                            q_model.set_weights(vals)
                            print(f"✓ Loaded weights from backup pickle: {backup_pickle}")
                            loaded_from_backup = True
                    except Exception as e:
                        print(f"⚠️ Failed to apply backup pickle weights: {e}")

                if not loaded_from_backup and q_weights.exists():
                    try:
                        q_model.load_weights(str(q_weights), by_name=True, skip_mismatch=False)
                        print("✓ Loaded Quantile Regressor weights (by_name=True)")
                    except Exception as e:
                        print(f"⚠️ Failed to load Quantile weights: {e}")
                        # diagnostics
                        try:
                            import h5py
                            with h5py.File(str(q_weights), 'r') as hf:
                                h5_keys = []
                                def _collect(name, obj):
                                    h5_keys.append(name)
                                hf.visititems(_collect)
                        except Exception:
                            h5_keys = None

                        try:
                            model_vars = [v.name for v in q_model.weights]
                        except Exception:
                            model_vars = None

                        print('--- Weight load diagnostics ---')
                        print(f'TensorFlow version: {tf.__version__}')
                        if h5_keys is not None:
                            print(f'Weights file contains ~{len(h5_keys)} HDF5 keys (sample): {h5_keys[:10]}')
                        if model_vars is not None:
                            print(f'Model exposes {len(model_vars)} named weight entries (sample): {model_vars[:10]}')
                        print('Proceeding with whatever weights were loaded (may be random)')

            # Predict
            q_seq_len = q_meta.get('sequence_length', 60)
            X_q = df_clean[used_feature_cols].values
            X_q_scaled = q_feat_scaler.transform(X_q)
            X_seq_q = create_sequences(X_q_scaled, q_seq_len)

            print(f"Predicting quantiles on {len(X_seq_q)} sequences...")
            q_preds_raw = q_model.predict(X_seq_q, verbose=0)

            if isinstance(q_preds_raw, dict):
                q10 = q_preds_raw.get('q10')
                q50 = q_preds_raw.get('q50')
                q90 = q_preds_raw.get('q90')
                if hasattr(q10, 'numpy'): q10 = q10.numpy()
                if hasattr(q50, 'numpy'): q50 = q50.numpy()
                if hasattr(q90, 'numpy'): q90 = q90.numpy()
                q_preds_scaled = np.hstack([q10, q50, q90])
            else:
                q_preds_scaled = q_preds_raw

            q10_pred = q_target_scaler.inverse_transform(q_preds_scaled[:, 0].reshape(-1, 1)).flatten()
            q50_pred = q_target_scaler.inverse_transform(q_preds_scaled[:, 1].reshape(-1, 1)).flatten()
            q90_pred = q_target_scaler.inverse_transform(q_preds_scaled[:, 2].reshape(-1, 1)).flatten()

            test_len = len(y_test)
            if len(q50_pred) < test_len:
                pad = test_len - len(q50_pred)
                q10_pred = np.pad(q10_pred, (pad, 0), mode='edge')
                q50_pred = np.pad(q50_pred, (pad, 0), mode='edge')
                q90_pred = np.pad(q90_pred, (pad, 0), mode='edge')

            q10_test = q10_pred[-test_len:]
            q50_test = q50_pred[-test_len:]
            q90_test = q90_pred[-test_len:]

            mae_q50 = np.mean(np.abs(y_test - q50_test))
            print(f"Quantile MAE (median): {mae_q50:.6f}")
            model_status['quantile'] = True

            uncertainty = q90_test - q10_test
            u_min, u_max = np.min(uncertainty), np.max(uncertainty)
            if u_max > u_min:
                uncertainty_norm = (uncertainty - u_min) / (u_max - u_min)
            else:
                uncertainty_norm = np.zeros_like(uncertainty)

            # Adjust positions conservatively
            modified_positions = strategy_positions.copy()
            bearish_q50 = q50_test < 0
            bullish_strategy = modified_positions > 0
            conflict_mask = bearish_q50 & bullish_strategy
            modified_positions[conflict_mask] = 0
            penalty_factor = 1.0 - (uncertainty_norm * 0.5)
            modified_positions = modified_positions * penalty_factor
            strategy_positions = modified_positions

            strat_ret = strategy_positions * y_test
            cum_return_q = np.prod(1 + strat_ret) - 1
            sharpe_q = (np.mean(strat_ret) / np.std(strat_ret)) * np.sqrt(252) if np.std(strat_ret) > 0 else 0
            print(f"--- Uncertainty-Aware Results ---")
            print(f"Cumulative Return: {cum_return_q:.2%}")
            print(f"Sharpe: {sharpe_q:.3f}")

        except Exception as e:
            print(f"❌ Quantile Integration failed: {e}")
            import traceback
            # Truncate long tracebacks in console; save full traceback to backtest_results
            tb = traceback.format_exc()
            if len(tb) > 1000:
                try:
                    base_dir = Path(__file__).resolve().parent
                    results_dir = base_dir / 'backtest_results'
                    results_dir.mkdir(parents=True, exist_ok=True)
                    import time
                    ts = time.strftime('%Y%m%d_%H%M%S')
                    err_path = results_dir / f"{symbol}_quantile_error_{ts}.log"
                    with open(err_path, 'w', encoding='utf-8') as ef:
                        ef.write(tb)
                    print(f"Full traceback saved to: {err_path}")
                except Exception:
                    print(tb[:1000])
            else:
                print(tb)
            model_status['quantile'] = False

    # ========================================
    # TFT INTEGRATION (CACHE-ONLY MODE)
    # ========================================
    if use_tft and not tft_attempted:
        tft_attempted = True
        
        print("\n" + "="*60)
        print("  RUNNING TFT BACKTEST (EXPERIMENTAL)")
        print("="*60)
        
        try:
            from inference.predict_tft import load_tft_model, prepare_inference_data, generate_forecast
            
            # ========================================
            # STEP 1: FIND CHECKPOINT
            # ========================================
            checkpoint_path = None
            
            # Try new organized path structure first
            if ModelPaths is not None:
                paths = ModelPaths(symbol)
                if paths.tft.checkpoint.exists():
                    checkpoint_path = paths.tft.checkpoint
                    print(f"Found TFT checkpoint (new structure): {checkpoint_path}")
            
            # Fall back to legacy paths
            if checkpoint_path is None:
                checkpoint_candidates = [
                    Path('saved_models') / 'tft' / symbol / 'best_model.ckpt',
                    Path('saved_models') / 'tft' / symbol.upper() / 'best_model.ckpt',
                    Path('saved_models') / f'{symbol}_tft.ckpt',
                ]
                
                # Add backup checkpoints (e.g., backup_epoch_20.ckpt)
                tft_dir = Path('saved_models') / 'tft' / symbol
                if tft_dir.exists():
                    checkpoint_candidates.extend(sorted(tft_dir.glob('*.ckpt'), reverse=True))
                
                for candidate in checkpoint_candidates:
                    if candidate.exists():
                        checkpoint_path = candidate
                        print(f"Found TFT checkpoint (legacy): {checkpoint_path}")
                        break
            
            if not checkpoint_path:
                all_tried = []
                if ModelPaths is not None:
                    all_tried.append(str(ModelPaths(symbol).tft.checkpoint))
                all_tried.extend([
                    f"saved_models/tft/{symbol}/best_model.ckpt",
                    f"saved_models/{symbol}_tft.ckpt"
                ])
                print(f"⚠️  No TFT checkpoint found for {symbol}")
                print(f"   Searched: {all_tried}")
                print(f"   Train TFT first: python training/train_tft.py --symbols {symbol}")
                model_status['tft'] = False
                raise FileNotFoundError("TFT checkpoint missing")
            
            # ========================================
            # STEP 2: LOAD MODEL (WITH EXPLICIT SYMBOL)
            # ========================================
            print(f"Loading TFT model for {symbol}...")
            model, dataset_params = load_tft_model(
                checkpoint_path=str(checkpoint_path),
                symbol=symbol,  # ← EXPLICIT SYMBOL (prevents fake symbol inference)
                map_location='cpu'
            )
            
            if model is None:
                print(f"❌ TFT model load failed for {symbol}")
                model_status['tft'] = False
                raise RuntimeError("TFT model load returned None")
            
            # ========================================
            # STEP 3: PREPARE DATA (CACHE-ONLY MODE)
            # ========================================
            print(f"Preparing TFT inference data for {symbol}...")
            
            try:
                df_tft = prepare_inference_data(
                    symbol=symbol,
                    use_cache_only=True,      # ← CRITICAL: No network fetch
                    include_sentiment=True     # ← Must match training
                )
                
                print(f"✓ TFT data ready: {len(df_tft)} rows, {len(df_tft.columns)} columns")
                
            except FileNotFoundError as e:
                print(f"\n{e}\n")
                print(f"TFT backtest requires cached data. Skipping TFT.")
                model_status['tft'] = False
                raise
            
            # ========================================
            # STEP 4: GENERATE FORECAST
            # ========================================
            horizon = 5
            print(f"Generating {horizon}-step forecast...")
            
            forecast = generate_forecast(model, df_tft, horizon=horizon)
            
            # ========================================
            # STEP 5: BACKTEST (SIMPLE VERSION)
            # ========================================
            # Extract q50 (median) predictions
            if 'predictions' in forecast and forecast['predictions']:
                q50_values = [
                    forecast['predictions'][h].get('q50', 0.0)
                    for h in forecast.get('horizons', [1])
                ]
                
                if q50_values:
                    print(f"✓ TFT forecast generated: {len(q50_values)} horizons")
                    print(f"   Median predictions: {[f'{v:.4f}' for v in q50_values]}")
                    model_status['tft'] = True
                else:
                    print(f"⚠️  TFT forecast empty")
                    model_status['tft'] = False
            else:
                print(f"⚠️  TFT forecast malformed")
                model_status['tft'] = False
        
        except FileNotFoundError:
            # Already logged above
            pass
        except Exception as e:
            print(f"❌ TFT backtest failed: {type(e).__name__}: {e}")
            
            # Save full traceback to file
            import traceback
            import time
            
            results_dir = Path('backtest_results')
            results_dir.mkdir(parents=True, exist_ok=True)
            
            error_log = results_dir / f"{symbol}_tft_error_{time.strftime('%Y%m%d_%H%M%S')}.log"
            with open(error_log, 'w') as f:
                f.write(traceback.format_exc())
            
            print(f"Full traceback saved to: {error_log}")
            model_status['tft'] = False

    # TFT Integration (final guard to avoid duplicate runs)
    if use_tft and not tft_attempted:
        print("\n" + "="*60)
        print("  RUNNING TFT BACKTEST (EXPERIMENTAL) - final guard")
        print("="*60)
        try:
            # Prefer running the standardized helper which already uses cache-only
            tft_results = run_backtest_with_tft(
                symbol=symbol,
                use_tft=True,
                horizon=5,
                reg_threshold=reg_threshold,
                reg_scale=reg_scale,
                max_short=max_short_exposure
            )
            if tft_results:
                model_status['tft'] = True
        except Exception as e:
            # Print concise error to console but persist full traceback to file
            print(f"❌ TFT Backtest failed: {e}")
            import traceback, time
            tb = traceback.format_exc()
            try:
                base_dir = Path(__file__).resolve().parent
                results_dir = base_dir / 'backtest_results'
                results_dir.mkdir(parents=True, exist_ok=True)
                ts = time.strftime('%Y%m%d_%H%M%S')
                err_path = results_dir / f"{symbol}_tft_error_{ts}.log"
                with open(err_path, 'w', encoding='utf-8') as ef:
                    ef.write(tb)
                print(f"Full traceback saved to: {err_path}")
            except Exception:
                # If logging fails, still print truncated traceback
                print(tb[:2000])
            # Ensure we don't treat TFT as available
            model_status['tft'] = False

    # Final model-status summary and validation
    try:
        print("\n=== Model Load Summary ===")
        for k, v in model_status.items():
            status = '✅ Loaded' if v else '❌ Unavailable'
            print(f"  {k}: {status}")

        # Minimum requirement: at least regressor OR classifiers OR GBM must be available
        if not model_status.get('regressor') and not model_status.get('classifiers') and not model_status.get('gbm'):
            raise RuntimeError("Minimum models unavailable: need at least the regressor, classifiers, or GBM to run backtest.")
    except Exception as e:
        print(f"\n❌ Model availability validation failed: {e}")
        import traceback
        traceback.print_exc()
        # Fail-fast to avoid misleading backtest results
        import sys as sys_module
        sys_module.exit(1)

    # Prepare return object
    # === Ensure calibration JSON is produced (robust final call) ===
    try:
        try:
            import time as _time
            ts_cal = _time.strftime('%Y%m%d_%H%M%S')
        except Exception:
            from datetime import datetime as _dt
            ts_cal = _dt.now().strftime('%Y%m%d_%H%M%S')

        # Use the results_dir created earlier if available, otherwise fall back to root
        if 'results_dir' not in locals():
            base_dir = Path(__file__).resolve().parent
            results_dir = base_dir / 'backtest_results'
        try:
            results_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Build a reasonable returns array aligned to test set
        try:
            returns_arr_final = np.asarray(y_test, dtype=float)
        except Exception:
            try:
                returns_arr_final = np.concatenate([[np.nan], np.diff(test_prices) / test_prices[:-1]]) if len(test_prices) > 1 else np.array([np.nan] * len(test_prices))
            except Exception:
                returns_arr_final = np.array([np.nan])

        # Call the calibration analysis in a best-effort manner and persist results
        try:
            # Determine whether matplotlib is available
            try:
                import matplotlib
                do_plot = True
            except Exception:
                do_plot = False

            calib_res = analyze_confidence_calibration(buy_probs, sell_probs, final_signals, returns_arr_final, symbol=symbol, out_dir=str(results_dir), timestamp=ts_cal, plot=do_plot)
            try:
                print(f"Saved calibration analysis to: {calib_res.get('plot_path') or (results_dir / ('calibration_analysis_' + ts_cal + '.json'))}")
            except Exception:
                pass
        except Exception as _e:
            try:
                print(f"Final calibration analysis failed: {_e}")
            except Exception:
                pass
    except Exception:
        pass

    out = {
        'cum_return': float(cum_return),
        'avg_daily': float(avg_daily),
        'std_daily': float(std_daily) if not np.isnan(std_daily) else None,
        'sharpe': float(sharpe) if not np.isnan(sharpe) else None,
        'dir_acc': float(dir_acc),
        'buy_count': int(buy_count),
        'sell_count': int(sell_count),
        'regressor_metrics': reg_eval,
        'fusion_mode': fusion_mode,
        'regressor_strategy': reg_strategy,
        'model_status': model_status
    }
    # If advanced backtester ran earlier, attach benchmark outputs
    try:
        if 'bt_results' in locals() and bt_results is not None:
            # Align buy-hold equity to the same indexing as `dates`/`returns` (drop initial capital at index 0)
            try:
                out['buy_hold_equity'] = np.asarray(bt_results.buy_hold_equity[1:])
            except Exception:
                out['buy_hold_equity'] = np.asarray(bt_results.buy_hold_equity)
            # Strategy equity (aligned to dates) - include for direct comparison/analysis
            try:
                out['strategy_equity'] = np.asarray(bt_results.equity_curve[1:])
            except Exception:
                # Fallback: compute from positions and returns if not available
                try:
                    out['strategy_equity'] = np.asarray(np.cumprod(1 + (np.asarray(strategy_positions) * np.asarray(y_test))))
                except Exception:
                    out['strategy_equity'] = None
            out['buy_hold_return'] = float(bt_results.metrics.get('buy_hold_cum_return', 0.0))
            out['buy_hold_sharpe'] = float(bt_results.metrics.get('buy_hold_sharpe', 0.0))
            out['alpha'] = float(bt_results.metrics.get('alpha', 0.0))
            out['beta'] = float(bt_results.metrics.get('beta', 0.0))
            out['information_ratio'] = float(bt_results.metrics.get('information_ratio', 0.0))
            out['win_rate_vs_buy_hold_down_days'] = float(bt_results.metrics.get('win_rate_vs_buy_hold_down_days', 0.0))
            out['backtester_metrics'] = bt_results.metrics
    except Exception:
        # Non-critical - skip attaching benchmark info if anything fails
        pass
    # Attach trade log if available
    try:
        if 'out_trade_log' in locals() and out_trade_log is not None:
            out['trade_log'] = out_trade_log
    except Exception:
        pass

    # Save detailed report and plots (best-effort, optional)
    try:
        # Note: results_dir was already created at the start of main()
        # Ensure we have the folder path (should already exist from early setup)
        
        report_name = "backtest"  # Simplified name (folder contains symbol+timestamp)

        # Dump pickle with arrays for later inspection
        report_path = results_dir / (report_name + '.pkl')
        with open(report_path, 'wb') as f:
            pickle.dump({
                'dates': np.asarray(test_dates),
                'prices': np.asarray(test_prices),
                'returns': np.asarray(y_test),
                'positions': np.asarray(strategy_positions),
                'signals': np.asarray(final_signals),
                'cum_return': float(cum_return),
                'metrics': {
                    'sharpe': float(sharpe),
                    'mae': float(mae),
                    'mse': float(mse)
                }
                ,
                'trade_log': out.get('trade_log') if 'trade_log' in out else None,
                'backtester_metrics': out.get('backtester_metrics') if 'backtester_metrics' in out else None,
                'buy_hold_equity': np.asarray(out.get('buy_hold_equity')) if 'buy_hold_equity' in out else None,
                'equity': np.asarray(out.get('strategy_equity')) if 'strategy_equity' in out else None
            }, f)

        # Create plots if matplotlib is available
        png_path = None
        if plt is not None:
            try:
                dates_plot = np.arange(len(test_dates))
                equity = np.cumprod(1 + (np.asarray(strategy_positions) * np.asarray(y_test)))

                fig, ax1 = plt.subplots(figsize=(10, 5))
                ax1.plot(dates_plot, equity, label='Equity (strategy)', color='tab:blue')
                # (No buy-hold plotted here — keep default PNG focused on strategy + buy/sell markers)
                ax1.set_ylabel('Equity (cumulative)', color='tab:blue')
                ax1.set_xlabel('Test index (most recent last)')
                ax1.grid(True, alpha=0.3)

                ax2 = ax1.twinx()
                # align test_prices length with equity if necessary
                price_series = np.asarray(test_prices)
                if len(price_series) != len(equity):
                    price_series = price_series[-len(equity):]
                ax2.plot(dates_plot, price_series, label='Price', color='tab:gray', alpha=0.6)
                ax2.set_ylabel('Price', color='tab:gray')

                # Position overlay as step-filled area
                pos = np.asarray(strategy_positions)
                if len(pos) != len(equity):
                    pos = pos[-len(equity):]
                ax1.fill_between(dates_plot, 0, pos, color='tab:green', alpha=0.15, step='mid')

                # Annotate BUY/SELL signals on price axis using arrows/scatter
                # For regressor_only and GBM modes, use positions since final_signals are all 0
                try:
                    sig = np.asarray(final_signals)
                    if len(sig) != len(equity):
                        sig = sig[-len(equity):]

                    # Define regressor-like modes that don't use classifier signals
                    _REGRESSOR_LIKE_MODES = ('regressor_only', 'gbm_only', 'gbm_heavy', 'balanced', 'lstm_heavy')
                    
                    # Check if we're in regressor_only/GBM mode (all signals are 0)
                    # In that case, derive signals from position signs
                    if fusion_mode in _REGRESSOR_LIKE_MODES or np.all(sig == 0):
                        # Use position signs as signals for plotting
                        pos_for_signals = np.asarray(strategy_positions)
                        if len(pos_for_signals) != len(equity):
                            pos_for_signals = pos_for_signals[-len(equity):]
                        # Lower threshold to 0.001 (0.1%) to show even small positions
                        buy_idx = np.where(pos_for_signals > 0.001)[0]  # Long positions > 0.1%
                        sell_idx = np.where(pos_for_signals < -0.001)[0]  # Short positions < -0.1%
                    else:
                        buy_idx = np.where(sig == 1)[0]
                        sell_idx = np.where(sig == -1)[0]

                    if buy_idx.size > 0:
                        buy_prices = price_series[buy_idx]
                        ax2.scatter(buy_idx, buy_prices, marker='^', color='green', s=120, label='LONG', zorder=5, edgecolors='darkgreen', linewidths=1)
                    if sell_idx.size > 0:
                        sell_prices = price_series[sell_idx]
                        ax2.scatter(sell_idx, sell_prices, marker='v', color='red', s=120, label='SHORT', zorder=5, edgecolors='darkred', linewidths=1)

                    # Legends on both axes
                    h1, l1 = ax1.get_legend_handles_labels()
                    h2, l2 = ax2.get_legend_handles_labels()
                    # Merge legends, avoid duplicates
                    handles = h1 + [h for h in h2 if h not in h1]
                    labels = l1 + [lab for i, lab in enumerate(l2) if h2[i] not in h1]
                    if handles:
                        ax1.legend(handles, labels, loc='upper left')
                except Exception:
                    # Non-critical - don't fail plotting if signals misaligned
                    pass

                ax1.set_title(f"Backtest: {symbol} | CumReturn {cum_return:.2%} | Sharpe {sharpe:.3f}")
                fig.tight_layout()
                png_path = results_dir / (report_name + '.png')
                fig.savefig(png_path)
                plt.close(fig)
                # Also save a dedicated equity comparison plot (strategy vs buy-and-hold)
                try:
                    eq_fig, eq_ax = plt.subplots(figsize=(10, 5))
                    # Strategy equity (already computed as `equity`) and buy-hold
                    # Normalize to dollar equity for better visibility using HYBRID_REFERENCE_EQUITY
                    try:
                        start_cap = float(HYBRID_REFERENCE_EQUITY)
                    except Exception:
                        start_cap = 10000.0

                    strat_series = np.asarray(equity, dtype=float)
                    bh_returns = np.asarray(y_test, dtype=float)
                    buy_hold_series = np.cumprod(1 + bh_returns)
                    # Align lengths if necessary
                    if len(buy_hold_series) != len(strat_series):
                        buy_hold_series = buy_hold_series[-len(strat_series):]
                    # Convert to dollar equity
                    strat_dollars = start_cap * strat_series
                    bh_dollars = start_cap * buy_hold_series

                    eq_ax.plot(dates_plot, strat_dollars, label='Equity (strategy)', color='tab:blue')
                    eq_ax.plot(dates_plot, bh_dollars, label='Buy & Hold', color='tab:green', linestyle='--', alpha=0.9)
                    eq_ax.set_ylabel('Equity ($)')
                    eq_ax.set_xlabel('Test index (most recent last)')
                    eq_ax.grid(True, alpha=0.3)

                    # Tighter y-limits so small changes are visible
                    try:
                        ymin = float(np.nanmin(np.concatenate([strat_dollars, bh_dollars])))
                        ymax = float(np.nanmax(np.concatenate([strat_dollars, bh_dollars])))
                        if np.isfinite(ymin) and np.isfinite(ymax) and ymax > ymin:
                            pad = max((ymax - ymin) * 0.05, 1.0)
                            eq_ax.set_ylim(ymin - pad, ymax + pad)
                    except Exception:
                        pass

                    # Annotate outperformance percentage (strategy vs buy-hold)
                    try:
                        strat_final = float(strat_dollars[-1]) if strat_dollars.size else start_cap
                        bh_final = float(bh_dollars[-1]) if bh_dollars.size else start_cap
                        outperf_pct = (strat_final - bh_final) / max(bh_final, 1e-9) * 100.0
                        eq_ax.annotate(f"Outperformance: {outperf_pct:+.2f}%", xy=(0.99, 0.95), xycoords='axes fraction',
                                       ha='right', va='top', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.2))
                    except Exception:
                        pass

                    eq_ax.legend(loc='upper left')
                    equity_png = results_dir / 'equity_curve.png'
                    eq_fig.tight_layout()
                    eq_fig.savefig(equity_png)
                    plt.close(eq_fig)
                    # Also write a CSV with the two series for easy verification
                    try:
                        import csv
                        csv_path = results_dir / 'equity_comparison.csv'
                        dates_out = np.asarray(test_dates)
                        # Align lengths
                        min_len = min(len(dates_out), len(strat_dollars), len(bh_dollars))
                        with open(csv_path, 'w', newline='') as cf:
                            writer = csv.writer(cf)
                            writer.writerow(['date_index', 'date', 'strategy_equity_usd', 'buy_hold_equity_usd'])
                            for k in range(min_len):
                                d = dates_out[-min_len:][k] if len(dates_out) >= min_len else k
                                writer.writerow([k, d, float(strat_dollars[-min_len:][k]), float(bh_dollars[-min_len:][k])])
                    except Exception:
                        pass
                except Exception:
                    # Non-critical: continue if dedicated equity plot fails
                    pass
            except Exception as e:
                print(f"⚠️ Failed to produce plots: {e}")
        else:
            print("⚠️ matplotlib not available; skipping plot generation")

        print(f"✓ Saved backtest report: {report_path}")
        if png_path is not None:
            print(f"✓ Saved backtest plot: {png_path}")
        # Run adaptive exit analysis automatically if trade log is available
        try:
            trade_log_for_analysis = None
            if 'out_trade_log' in locals() and out_trade_log is not None:
                trade_log_for_analysis = out_trade_log
            elif 'out' in locals() and isinstance(out, dict) and out.get('trade_log') is not None:
                trade_log_for_analysis = out.get('trade_log')

            if trade_log_for_analysis is not None:
                try:
                    # Aggregate daily positions into per-trade records using start when pos>0 and end when pos==0
                    trades_list = aggregate_daily_positions_to_trades(trade_log_for_analysis) if isinstance(trade_log_for_analysis, pd.DataFrame) else trade_log_for_analysis
                    report_path_exits = analyze_adaptive_exits(trades_list, trades_list, symbol=symbol, out_dir=results_dir)
                    print(f"✓ Saved exit analysis report: {report_path_exits}")
                except Exception as _e:
                    print(f"⚠️ Exit analysis failed: {_e}")
            else:
                # Create a small marker file so it's obvious the exit analysis was skipped
                try:
                    import time
                    ts_skip = time.strftime('%Y%m%d_%H%M%S')
                    skip_name = f"{symbol or 'unknown'}_exit_analysis_{ts_skip}_SKIPPED.txt"
                    skip_path = results_dir / skip_name
                    with open(skip_path, 'w', encoding='utf-8') as sf:
                        sf.write(f"Exit analysis skipped: no trade_log available for symbol={symbol or 'unknown'}\n")
                        sf.write("Run the script with a working trade log or ensure compute_hybrid_positions returns a non-empty trade log.\n")
                    print(f"Exit analysis skipped - marker written to: {skip_path}")
                except Exception:
                    pass
        except Exception:
            pass

        # Regime performance analysis (requires dates/prices/returns/positions/regimes)
        try:
            if 'test_dates' in locals() and 'test_prices' in locals() and 'y_test' in locals() and 'strategy_positions' in locals():
                # Regimes optional - use if available, otherwise None
                regimes_for_analysis = regimes if 'regimes' in locals() else None
                regime_analysis = analyze_regime_performance(test_dates, test_prices, y_test, strategy_positions, regimes_for_analysis, symbol=symbol)
                ts_reg = time.strftime('%Y%m%d_%H%M%S')
                regime_json = results_dir / 'regime_analysis.json'
                with open(regime_json, 'w', encoding='utf-8') as rf:
                    json.dump(regime_analysis, rf, indent=2, ensure_ascii=False)
                print(f"✓ Saved regime analysis: {regime_json}")
                try:
                    # Produce drawdown analysis CSV/PNG using strategy equity
                    try:
                        equity_final = np.cumprod(1 + (np.asarray(strategy_positions) * np.asarray(y_test)))
                    except Exception:
                        equity_final = None
                    if equity_final is not None:
                        try:
                            dd_ts = time.strftime('%Y%m%d_%H%M%S')
                        except Exception:
                            from datetime import datetime as _dt
                            dd_ts = _dt.now().strftime('%Y%m%d_%H%M%S')
                        try:
                            # Compute per-day Kelly fractions (best-effort using final trade history)
                            try:
                                win_loss_ratio_final = calculate_win_loss_ratio_from_history(trade_history) if 'trade_history' in locals() else DEFAULT_WIN_LOSS_RATIO
                            except Exception:
                                win_loss_ratio_final = DEFAULT_WIN_LOSS_RATIO

                            kelly_fractions = None
                            try:
                                kf_list = []
                                for i_idx in range(len(equity_final)):
                                    try:
                                        # Prefer final_signals if available, otherwise infer from positions
                                        if 'final_signals' in locals() and final_signals is not None and i_idx < len(final_signals):
                                            sig_i = int(final_signals[i_idx])
                                        else:
                                            sig_i = 1 if (isinstance(strategy_positions, (list, np.ndarray)) and i_idx < len(strategy_positions) and strategy_positions[i_idx] > 0) else (-1 if (isinstance(strategy_positions, (list, np.ndarray)) and i_idx < len(strategy_positions) and strategy_positions[i_idx] < 0) else 0)

                                        if sig_i > 0:
                                            win_prob_i = float(buy_probs[i_idx]) if ('buy_probs' in locals() and i_idx < len(buy_probs)) else 0.5
                                        elif sig_i < 0:
                                            win_prob_i = float(sell_probs[i_idx]) if ('sell_probs' in locals() and i_idx < len(sell_probs)) else 0.5
                                        else:
                                            win_prob_i = 0.5

                                        win_prob_i = np.clip(win_prob_i, 0.01, 0.99)
                                        kf = calculate_kelly_fraction(win_prob_i, win_loss_ratio_final)
                                        kf_list.append(float(kf))
                                    except Exception:
                                        kf_list.append(np.nan)
                                kelly_fractions = np.asarray(kf_list, dtype=float)
                            except Exception:
                                kelly_fractions = None

                            # Compute buy-hold equity (aligned to strategy equity) for buy-hold drawdown comparisons
                            try:
                                bh = np.cumprod(1 + np.asarray(y_test, dtype=float))
                                if len(bh) != len(equity_final):
                                    bh = bh[-len(equity_final):]
                                buy_hold_equity = bh
                            except Exception:
                                buy_hold_equity = None

                            dd_res = analyze_drawdowns(
                                equity_final,
                                dates=test_dates,
                                positions=strategy_positions,
                                kelly_fractions=kelly_fractions,
                                buy_hold_equity=buy_hold_equity,
                                symbol=symbol,
                                out_dir=str(results_dir),
                                timestamp=dd_ts,
                                save_csv=True,
                                plot=('plt' in locals() and plt is not None)
                            )
                            if dd_res.get('csv_path'):
                                print(f"✓ Saved drawdown CSV: {dd_res.get('csv_path')}")
                            if dd_res.get('plot_path'):
                                print(f"✓ Saved drawdown plot: {dd_res.get('plot_path')}")
                        except Exception as _e:
                            print(f"⚠️ Drawdown analysis failed: {_e}")
                        
                        # ===================================================================
                        # POSITION SIZING HEATMAP VISUALIZATION
                        # ===================================================================
                        try:
                            from evaluation.position_heatmap import integrate_with_backtest
                            
                            # Always build a fresh trade log DataFrame for heatmap from available data
                            try:
                                heatmap_trade_log = pd.DataFrame({
                                    'date': test_dates,
                                    'price': test_prices,
                                    'action': ['BUY' if s == 1 else ('SELL' if s == -1 else 'HOLD') for s in final_signals],
                                    'portfolio_pct': strategy_positions,
                                    'unified_confidence': unified_confidence if 'unified_confidence' in locals() else None,
                                    'buy_prob': buy_probs if 'buy_probs' in locals() else None,
                                    'sell_prob': sell_probs if 'sell_probs' in locals() else None,
                                    'regime': regimes if 'regimes' in locals() else None,
                                    'atr': test_atr_percent if test_atr_percent is not None else None,
                                })
                                
                                heatmap_result = integrate_with_backtest(
                                    trade_log_df=heatmap_trade_log,
                                    equity_curve=equity_final,
                                    symbol=symbol,
                                    timestamp=ts,
                                    out_dir=str(results_dir)
                                )
                                
                                print(f"\n✓ Saved position sizing heatmap: {heatmap_result['plot_path']}")
                                
                                # Print summary metrics
                                summary = heatmap_result['data_summary']
                                print(f"  Total days: {summary['total_days']}")
                                print(f"  Signals: BUY={summary['buy_signals']}, SELL={summary['sell_signals']}, HOLD={summary['hold_signals']}")
                                print(f"  Avg BUY position: {summary['avg_buy_position_pct']:.1f}%")
                                print(f"  Position range: {summary['min_position_pct']:.1f}% to {summary['max_position_pct']:.1f}%")
                            except Exception as _inner_err:
                                print(f"⚠️ Position heatmap error: {_inner_err}")
                                import traceback
                                traceback.print_exc()
                        except Exception as _hm_err:
                            print(f"⚠️ Position heatmap import failed: {_hm_err}")
                            import traceback
                            traceback.print_exc()
                        
                        # ===================================================================
                        # COMPREHENSIVE BACKTEST DASHBOARD (2x3 SUBPLOTS)
                        # ===================================================================
                        try:
                            from evaluation.backtest_dashboard import integrate_with_backtest as create_dashboard
                            
                            # Calculate max position limits (drawdown-adjusted)
                            try:
                                running_max_eq = np.maximum.accumulate(equity_final)
                                current_dd = (equity_final - running_max_eq) / running_max_eq
                                # Reduce max position as drawdown increases
                                max_pos_limits = np.clip(1.0 + current_dd, 0.3, 1.0)  # 30% min, 100% max
                            except Exception:
                                max_pos_limits = np.ones(len(equity_final))
                            
                            dashboard_result = create_dashboard(
                                dates=test_dates,
                                prices=test_prices,
                                returns=y_test,
                                strategy_positions=strategy_positions,
                                equity_curve=equity_final,
                                buy_hold_equity=buy_hold_equity if 'buy_hold_equity' in locals() else np.cumprod(1 + y_test),
                                final_signals=final_signals,
                                buy_probs=buy_probs if 'buy_probs' in locals() else None,
                                sell_probs=sell_probs if 'sell_probs' in locals() else None,
                                regimes=regimes if 'regimes' in locals() else None,
                                kelly_fractions=kelly_fractions if 'kelly_fractions' in locals() else None,
                                max_position_limits=max_pos_limits,
                                symbol=symbol,
                                timestamp=ts,
                                out_dir=str(results_dir)
                            )
                            
                            print(f"\n✓ Saved comprehensive dashboard: {dashboard_result['plot_path']}")
                            
                            # Print summary
                            summ = dashboard_result['summary']
                            print(f"\n=== Dashboard Summary ===")
                            print(f"Strategy Return: {summ['total_return_strategy_pct']:.2f}%")
                            print(f"Buy & Hold Return: {summ['total_return_buyhold_pct']:.2f}%")
                            print(f"Max Drawdown: {summ['max_drawdown_pct']:.2f}%")
                            print(f"Sharpe Ratio: {summ['sharpe_ratio']:.2f}")
                            print(f"Win Rate: {summ['win_rate_pct']:.1f}%")
                            print(f"Total Trades: {summ['total_trades']} (BUY={summ['buy_trades']}, SELL={summ['sell_trades']})")
                            print(f"Avg Kelly Fraction: {summ['avg_kelly_fraction']:.3f}")
                        except Exception as _dash_err:
                            print(f"⚠️ Dashboard generation failed: {_dash_err}")
                            import traceback
                            traceback.print_exc()
                except Exception:
                    pass
        except Exception as _re:
            print(f"⚠️ Regime analysis failed: {_re}")
        # Focused validation: compare fused_positions vs hybrid positions (if both exist)
        try:
            try:
                ts_cmp = time.strftime('%Y%m%d_%H%M%S')
            except Exception:
                import datetime as _dt
                ts_cmp = _dt.datetime.now().strftime('%Y%m%d_%H%M%S')

            cmp_path = results_dir / 'confidence_sizing_comparison.json'
            fused_metrics = None
            hybrid_metrics = None
            try:
                fused_ret = np.prod(1 + (np.asarray(fused_positions) * np.asarray(y_test))) - 1
                fused_sharpe = (np.mean(np.asarray(fused_positions) * np.asarray(y_test)) / np.std(np.asarray(fused_positions) * np.asarray(y_test))) * np.sqrt(252) if np.std(np.asarray(fused_positions) * np.asarray(y_test)) > 0 else None
                fused_metrics = {'cum_return': float(fused_ret), 'sharpe': float(fused_sharpe) if fused_sharpe is not None else None}
            except Exception:
                fused_metrics = None
            try:
                if 'positions' in locals() and positions is not None:
                    hybrid_pos = np.asarray(positions)
                else:
                    hybrid_pos = np.asarray(strategy_positions)
                hybrid_ret = np.prod(1 + (hybrid_pos * np.asarray(y_test))) - 1
                hybrid_sharpe = (np.mean(hybrid_pos * np.asarray(y_test)) / np.std(hybrid_pos * np.asarray(y_test))) * np.sqrt(252) if np.std(hybrid_pos * np.asarray(y_test)) > 0 else None
                hybrid_metrics = {'cum_return': float(hybrid_ret), 'sharpe': float(hybrid_sharpe) if hybrid_sharpe is not None else None}
            except Exception:
                hybrid_metrics = None

            cmp_obj = {'symbol': symbol, 'fused_metrics': fused_metrics, 'hybrid_metrics': hybrid_metrics}
            with open(cmp_path, 'w', encoding='utf-8') as cf:
                json.dump(cmp_obj, cf, indent=2, ensure_ascii=False)
            print(f"✓ Saved sizing comparison: {cmp_path}")
            # Print a concise comparison to console
            try:
                print('\n=== Focused sizing comparison ===')
                print(f"Fused: cum_return={fused_metrics.get('cum_return') if fused_metrics else 'N/A'}, sharpe={fused_metrics.get('sharpe') if fused_metrics else 'N/A'}")
                print(f"Hybrid: cum_return={hybrid_metrics.get('cum_return') if hybrid_metrics else 'N/A'}, sharpe={hybrid_metrics.get('sharpe') if hybrid_metrics else 'N/A'}")
            except Exception:
                pass
        except Exception as _e:
            print(f"⚠️ Focused validation failed: {_e}")
        # Optionally analyze the saved backtest pickle using the analyzer
        if get_data:
            try:
                from analyze_backtest_pickle import analyze as _analyze_pickle
                try:
                    print('\n>> Running built-in backtest analyzer on the saved .pkl file...')
                    _analyze_pickle(report_path)
                    print('>> Analyzer finished')
                except SystemExit:
                    # analyzer may call SystemExit on completion; ignore
                    pass
                except Exception as _e:
                    print(f"⚠️ Analyzer failed on {report_path}: {_e}")
            except Exception:
                # If analyzer not available for any reason, do not fail the backtest
                pass
    except Exception as e:
        print(f"⚠️ Failed to save backtest report: {e}")

    return out


def analyze_drawdowns(equity_curve, dates=None, positions=None, kelly_fractions=None, buy_hold_equity=None, symbol: str | None = None, out_dir: str | None = None, timestamp: str | None = None, save_csv: bool = True, plot: bool = True):
    """Detailed drawdown analysis and visualization.

    Computes drawdown periods (peak -> trough -> recovery), summary statistics,
    optional position/kelly comparisons, writes CSV and an optional plot.
    """
    from pathlib import Path
    import numpy as _np

    eq = _np.asarray(equity_curve, dtype=float)
    n = eq.size
    if n == 0:
        return {'drawdowns': [], 'stats': {}, 'csv_path': None, 'plot_path': None}

    # normalize dates
    dates_arr = None
    if dates is not None:
        try:
            import pandas as _pd
            dates_arr = _pd.to_datetime(dates)
        except Exception:
            try:
                dates_arr = _np.asarray(dates)
            except Exception:
                dates_arr = None

    running_max = _np.maximum.accumulate(eq)

    # drawdown values (negative or zero)
    dd = (eq - running_max) / running_max

    peak_inds = _np.where(eq == running_max)[0]
    drawdowns = []

    for i in range(len(peak_inds)):
        start = int(peak_inds[i])
        end = int(peak_inds[i + 1]) if i + 1 < len(peak_inds) else n - 1
        segment = eq[start:end + 1]
        if segment.size == 0:
            continue
        trough_rel = int(_np.argmin(segment))
        trough = start + trough_rel
        if trough == start:
            continue

        depth = float((eq[start] - eq[trough]) / eq[start]) if eq[start] != 0 else 0.0
        duration = int(trough - start)

        recovery = None
        for j in range(trough + 1, n):
            if eq[j] >= eq[start] - 1e-12:
                recovery = j
                break

        recovery_days = int(recovery - trough) if recovery is not None else None

        draw = {
            'peak_index': int(start),
            'peak_date': (str(dates_arr[start]) if dates_arr is not None else None),
            'trough_index': int(trough),
            'trough_date': (str(dates_arr[trough]) if dates_arr is not None else None),
            'recovery_index': int(recovery) if recovery is not None else None,
            'recovery_date': (str(dates_arr[recovery]) if (dates_arr is not None and recovery is not None) else None),
            'depth_pct': float(depth * 100.0),
            'duration_days': int(duration),
            'recovery_days': int(recovery_days) if recovery_days is not None else None
        }

        # buy-hold comparison for the same interval
        try:
            if buy_hold_equity is not None:
                bh = _np.asarray(buy_hold_equity, dtype=float)
                # align lengths
                if bh.size == eq.size:
                    bh_start = bh[start]
                    bh_trough = bh[trough]
                    bh_depth = float((bh_start - bh_trough) / bh_start) if bh_start != 0 else 0.0
                    # buy-hold recovery detection
                    bh_recovery = None
                    for jj in range(trough + 1, bh.size):
                        if bh[jj] >= bh_start - 1e-12:
                            bh_recovery = jj
                            break
                    bh_recovery_days = int(bh_recovery - trough) if bh_recovery is not None else None
                    draw['buy_hold_depth_pct'] = float(bh_depth * 100.0)
                    draw['buy_hold_duration_days'] = int(duration)
                    draw['buy_hold_recovery_days'] = int(bh_recovery_days) if bh_recovery_days is not None else None
                else:
                    draw['buy_hold_depth_pct'] = None
                    draw['buy_hold_duration_days'] = None
                    draw['buy_hold_recovery_days'] = None
            else:
                draw['buy_hold_depth_pct'] = None
                draw['buy_hold_duration_days'] = None
                draw['buy_hold_recovery_days'] = None
        except Exception:
            draw['buy_hold_depth_pct'] = None
            draw['buy_hold_duration_days'] = None
            draw['buy_hold_recovery_days'] = None

        # position sizing metrics
        try:
            if positions is not None:
                pos = _np.asarray(positions, dtype=float)
                seg_pos = pos[start:trough + 1]
                outside_mask = _np.ones_like(pos, dtype=bool)
                outside_mask[start:trough + 1] = False
                outside_pos = pos[outside_mask]
                draw['mean_pos_during_drawdown'] = float(_np.nanmean(_np.abs(seg_pos))) if seg_pos.size > 0 else None
                draw['mean_pos_outside_drawdown'] = float(_np.nanmean(_np.abs(outside_pos))) if outside_pos.size > 0 else None
            else:
                draw['mean_pos_during_drawdown'] = None
                draw['mean_pos_outside_drawdown'] = None
        except Exception:
            draw['mean_pos_during_drawdown'] = None
            draw['mean_pos_outside_drawdown'] = None

        # kelly comparison
        try:
            if kelly_fractions is not None:
                kf = _np.asarray(kelly_fractions, dtype=float)
                seg_k = kf[start:trough + 1]
                outside_k = _np.concatenate([kf[:start], kf[trough + 1:]]) if kf.size > 0 else _np.array([])
                draw['mean_kelly_during'] = float(_np.nanmean(seg_k)) if seg_k.size > 0 else None
                draw['mean_kelly_outside'] = float(_np.nanmean(outside_k)) if outside_k.size > 0 else None
            else:
                draw['mean_kelly_during'] = None
                draw['mean_kelly_outside'] = None
        except Exception:
            draw['mean_kelly_during'] = None
            draw['mean_kelly_outside'] = None

        drawdowns.append(draw)

    depths = _np.array([d['depth_pct'] for d in drawdowns if d.get('depth_pct') is not None], dtype=float)
    recovery_times = _np.array([d['recovery_days'] for d in drawdowns if d.get('recovery_days') is not None], dtype=float)
    durations = _np.array([d['duration_days'] for d in drawdowns if d.get('duration_days') is not None], dtype=float)

    stats = {
        'num_drawdowns': int(len(drawdowns)),
        'avg_drawdown_pct': float(_np.nanmean(depths)) if depths.size > 0 else None,
        'median_drawdown_pct': float(_np.nanmedian(depths)) if depths.size > 0 else None,
        'num_gt_5pct': int(_np.sum(depths > 5.0)) if depths.size > 0 else 0,
        'num_gt_10pct': int(_np.sum(depths > 10.0)) if depths.size > 0 else 0,
        'num_gt_15pct': int(_np.sum(depths > 15.0)) if depths.size > 0 else 0,
        'longest_peak_to_trough_days': int(_np.nanmax(durations)) if durations.size > 0 else None,
        'avg_recovery_days': float(_np.nanmean(recovery_times)) if recovery_times.size > 0 else None
    }

    csv_path = None
    plot_path = None
    try:
        base_dir = Path(__file__).resolve().parent
        results_dir = Path(out_dir) if out_dir is not None else base_dir / 'backtest_results'
        results_dir.mkdir(parents=True, exist_ok=True)
        ts = timestamp or _np.datetime_as_string(_np.datetime64('now'), unit='s')
        ts = str(ts).replace(' ', '_').replace(':', '').replace('.', '')

        if save_csv:
            import csv as _csv
            csv_path = results_dir / 'drawdowns.csv'
            with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
                writer = _csv.writer(cf)
                header = ['peak_index', 'peak_date', 'trough_index', 'trough_date', 'recovery_index', 'recovery_date', 'depth_pct', 'duration_days', 'recovery_days', 'mean_pos_during_drawdown', 'mean_pos_outside_drawdown', 'mean_kelly_during', 'mean_kelly_outside', 'buy_hold_depth_pct', 'buy_hold_duration_days', 'buy_hold_recovery_days']
                writer.writerow(header)
                for d in drawdowns:
                    writer.writerow([d.get(h) for h in ['peak_index', 'peak_date', 'trough_index', 'trough_date', 'recovery_index', 'recovery_date', 'depth_pct', 'duration_days', 'recovery_days', 'mean_pos_during_drawdown', 'mean_pos_outside_drawdown', 'mean_kelly_during', 'mean_kelly_outside', 'buy_hold_depth_pct', 'buy_hold_duration_days', 'buy_hold_recovery_days']])
    except Exception:
        csv_path = None

    if plot:
        try:
            import matplotlib.pyplot as _plt
            base_dir = Path(__file__).resolve().parent
            results_dir = Path(out_dir) if out_dir is not None else base_dir / 'backtest_results'
            results_dir.mkdir(parents=True, exist_ok=True)
            plot_path = results_dir / 'drawdowns.png'

            fig, axes = _plt.subplots(3, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 1, 1]})

            ax = axes[0]
            x = _np.arange(n)
            x_dates = dates_arr if dates_arr is not None else x

            ax.plot(x_dates, eq, label='Equity', color='tab:blue')
            ax.set_title(f"Equity Curve and Drawdowns ({symbol or 'unknown'})")
            ax.set_ylabel('Equity')

            for d in drawdowns:
                try:
                    s = d['peak_index']
                    r = d['recovery_index'] if d.get('recovery_index') is not None else n - 1
                    ax.axvspan(x_dates[s], x_dates[r], color='red', alpha=0.08)
                except Exception:
                    continue

            ax2 = axes[1]
            underwater = running_max - eq
            ax2.fill_between(x_dates, 0, underwater, color='tab:purple', alpha=0.4)
            ax2.set_ylabel('Drawdown (abs)')
            ax2.set_title('Underwater (distance from peak)')

            ax3 = axes[2]
            recs = [d['recovery_days'] for d in drawdowns if d.get('recovery_days') is not None]
            if len(recs) > 0:
                ax3.hist(recs, bins=20, color='tab:green', alpha=0.7)
            ax3.set_xlabel('Recovery days')
            ax3.set_title('Recovery Time Distribution')

            fig.tight_layout()
            fig.savefig(plot_path)
            _plt.close(fig)
        except Exception:
            plot_path = None

    return {'drawdowns': drawdowns, 'stats': stats, 'csv_path': str(csv_path) if csv_path is not None else None, 'plot_path': str(plot_path) if plot_path is not None else None}

if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(
        description='Run backtesting with various fusion modes (regressor_only, classifier, hybrid, etc.)'
    )
    
    # Core arguments
    p.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol (default: AAPL)')
    p.add_argument('--backtest-days', type=int, default=60,
                   help='Number of most recent test days to evaluate (default 60, 0 = full set)')
    p.add_argument('--fusion-mode', type=str, default='stacking',
                   choices=['stacking', 'regressor', 'regressor_only',
                            'gbm_only', 'gbm_heavy', 'balanced', 'lstm_heavy'],
                   help='Model fusion strategy: stacking (XGBoost meta-learner ensemble - RECOMMENDED), '
                        'regressor_only (pure LSTM regressor), regressor (pure regressor positions), '
                        'gbm_only (GBM models only, no LSTM), gbm_heavy (70%% GBM + 30%% LSTM), '
                        'balanced (50%% GBM + 50%% LSTM), lstm_heavy (70%% LSTM + 30%% GBM)')
    
    # Phase 5: Margin and borrowing costs
    p.add_argument('--margin-requirement', type=float, default=0.5,
                   help='Margin requirement for $1 of position (default 0.5 = 2x leverage)')
    p.add_argument('--borrow-rate', type=float, default=0.02,
                   help='Annual interest rate charged for shorting (default 0.02 = 2%%)')
    
    # GBM weight override
    p.add_argument('--gbm-weight', type=float, default=None,
                   help='Override GBM weight in fusion (0.0-1.0). Overrides fusion-mode defaults.')
    
    # Position sizing
    p.add_argument('--max-short', type=float, default=0.5,
                   help='Maximum short exposure as fraction of capital (default 0.5 = 50%%)')
    p.add_argument('--reg-strategy', type=str, default='confidence_scaling',
                   choices=['simple_threshold', 'confidence_scaling'],
                   help='Position sizing strategy for regressor modes')
    p.add_argument('--reg-threshold', type=float, default=0.01,
                   help='Threshold for simple regressor strategy (default 1%%)')
    p.add_argument('--reg-scale', type=float, default=15.0,
                   help='Scale factor for confidence-based regressor strategy')
    
    # Classifier thresholds (only used when classifiers are enabled)
    p.add_argument('--buy-conf-floor', type=float, default=0.33,
                   help='Minimum BUY probability required (default 0.33, only for classifier modes)')
    p.add_argument('--sell-conf-floor', type=float, default=0.5,
                   help='Minimum SELL probability required (default 0.50, only for classifier modes)')
    p.add_argument('--threshold-file', type=str, default=None,
                   help='Path to JSON file with custom optimal thresholds')
    
    # Optional model integrations (disabled by default)
    p.add_argument('--tft', action='store_true', default=False,
                   help='Enable TFT (Temporal Fusion Transformer) backtest alongside main backtest')
    p.add_argument('--quantile', action='store_true', default=False,
                   help='Enable Quantile Regressor for uncertainty-aware position sizing')
    
    # Auto-model selection (validates models and falls back if needed)
    p.add_argument('--auto-model', action='store_true', default=False,
                   help='Automatically select best fusion mode based on model health validation. '
                        'Falls back to GBM modes if LSTM regressor is collapsed.')
    
    # Data caching
    p.add_argument('--no-cache', action='store_true', default=False,
                   help='Disable cache and fetch/engineer data fresh')
    p.add_argument('--force-refresh', action='store_true', default=False,
                   help='Force data refetch even if cache exists')
    
    # Forward simulation
    p.add_argument('--forward-sim', action='store_true',
                   help='Enable forward-looking simulation using predicted returns')
    p.add_argument('--forward-days', type=int, default=None,
                   help='Number of predicted days to project (defaults to backtest window)')
    
    # Misc
    p.add_argument('--skip-regressor-backtest', action='store_true',
                   help='Skip standalone regressor backtest/CSV export')
    p.add_argument('--skip-analysis', action='store_true', default=False,
                   help='Skip running the built-in analyzer on backtest results')
    
    # P0 FIX: Add disable-fallback flag for A/B testing
    p.add_argument('--disable-fallback', action='store_true', default=False,
                   help='Disable LSTM collapse fallback (use raw fusion even if LSTM is biased)')
    
    args = p.parse_args()
    bt_days = args.backtest_days if args.backtest_days and args.backtest_days > 0 else None

    # ===================================================================
    # GRACEFUL ERROR HANDLING - Catch common failures with helpful messages
    # ===================================================================
    try:
        main(
            args.symbol,
            backtest_days=bt_days,
            buy_conf_floor=args.buy_conf_floor,
            sell_conf_floor=args.sell_conf_floor,
            max_short_exposure=args.max_short,
            reg_strategy=args.reg_strategy,
            reg_threshold=args.reg_threshold,
            reg_scale=args.reg_scale,
            fusion_mode=args.fusion_mode,
            gbm_weight_override=args.gbm_weight,
            skip_regressor_backtest=args.skip_regressor_backtest,
            forward_sim=args.forward_sim,
            forward_days=args.forward_days,
            threshold_file=args.threshold_file,
            use_cache=not args.no_cache,
            force_refresh=args.force_refresh,
            use_tft=args.tft,
            use_quantile=args.quantile,
            get_data=not args.skip_analysis,
            auto_model=args.auto_model,
            disable_fallback=args.disable_fallback,  # P0 FIX: Pass fallback flag
            margin_requirement=args.margin_requirement,
            borrow_rate=args.borrow_rate,
        )
    except FileNotFoundError as e:
        print(f"\n❌ ERROR: Required files not found")
        print(f"   {e}")
        print(f"\n💡 Solution: Train models first using:")
        print(f"   python training/train_1d_regressor_final.py {args.symbol}")
        print(f"   python training/train_binary_classifiers_final.py {args.symbol}")
        sys.exit(1)
    except ValueError as e:
        print(f"\n❌ ERROR: Data validation failed")
        print(f"   {e}")
        print(f"\n💡 Common causes:")
        print(f"   1. Symbol parameter not passed to engineer_features()")
        print(f"   2. Sentiment features unavailable but model expects them")
        print(f"   3. Feature count mismatch between training and inference")
        print(f"   4. Insufficient historical data for {args.symbol}")
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

