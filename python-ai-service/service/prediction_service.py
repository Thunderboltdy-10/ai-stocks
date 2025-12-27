"""Prediction + backtest orchestration utilities for the FastAPI service."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from pandas.tseries.offsets import BDay

from data.data_fetcher import fetch_stock_data
from data.feature_engineer import engineer_features
from data.target_engineering import prepare_training_data
from evaluation.advanced_backtester import AdvancedBacktester
from models.lstm_transformer_paper import LSTMTransformerPaper
from utils.losses import get_custom_objects


SAVED_MODELS_DIR = Path(__file__).resolve().parent.parent / "saved_models"
DEFAULT_SEQUENCE_LENGTH = 60
DEFAULT_LOOKBACK = 180
DEFAULT_MAX_SHORT = 0.5
FORECAST_MIN_HISTORY = 21  # ~1 month of sessions
DEFAULT_TRADE_SHARE_FLOOR = 0.5
HYBRID_SCALE_FACTOR = 4.0
HYBRID_BASE_FRACTION = 0.15
HYBRID_MAX_POSITION = 0.40  # Reduced from 75% to 40% to force more frequent rebalancing
HYBRID_MIN_SHARES_THRESHOLD = 1.0
HYBRID_REFERENCE_EQUITY = 10_000.0
HYBRID_POSITION_EPS = 1e-6
HYBRID_MIN_CONFIDENCE = 0.03

# Volatility-based position sizing configuration
ATR_PERIOD = 14  # 14-day Average True Range for volatility measure
BASELINE_VOLATILITY = 0.02  # 2% baseline ATR (typical for moderate volatility)
VOLATILITY_SCALE_MIN = 0.5  # Minimum scale factor during high volatility
VOLATILITY_SCALE_MAX = 2.0  # Maximum scale factor during low volatility

# Scale-out exit strategy configuration
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

# Profit-target order system configuration
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


@dataclass
class ArtifactBundle:
	symbol: str
	sequence_length: int
	feature_columns: List[str]
	feature_scaler: object
	target_scaler: object
	target_metadata: Dict[str, object]
	classifier_metadata: Dict[str, object]
	regressor_weights: Path
	buy_classifier_weights: List[Path]
	sell_classifier_weights: List[Path]


def calculate_atr(prices: np.ndarray, period: int = ATR_PERIOD) -> np.ndarray:
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


def calculate_volatility_adjusted_scale(current_atr: float) -> float:
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


def calculate_drawdown_adjustments(current_equity: float, high_water_mark: float) -> dict:
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


def _load_pickle(path: Path) -> object | None:
	if not path.exists():
		return None
	try:
		import pickle

		with path.open("rb") as fh:
			return pickle.load(fh)
	except Exception:
		return None


def _require_file(path: Path, description: str) -> Path:
	if not path.exists():
		raise FileNotFoundError(f"Missing {description}: {path}")
	return path


def _find_first_existing(candidates: Sequence[Path]) -> Path:
	for path in candidates:
		if path.exists():
			return path
	joined = "\n".join(str(p) for p in candidates)
	raise FileNotFoundError(f"None of the candidate files exist:\n{joined}")


def load_artifacts(symbol: str) -> ArtifactBundle:
	symbol = symbol.upper()
	saved_dir = SAVED_MODELS_DIR
	saved_dir.mkdir(parents=True, exist_ok=True)

	feature_scaler_path = _find_first_existing(
		[
			saved_dir / f"{symbol}_1d_regressor_final_feature_scaler.pkl",
			saved_dir / f"{symbol}_binary_feature_scaler.pkl",
		]
	)
	feature_scaler = _load_pickle(feature_scaler_path)
	feature_cols_path = _find_first_existing(
		[
			saved_dir / f"{symbol}_1d_regressor_final_features.pkl",
			saved_dir / f"{symbol}_binary_features.pkl",
		]
	)
	feature_columns = list(_load_pickle(feature_cols_path) or [])
	if not feature_columns:
		raise ValueError(f"Feature column list is empty for {symbol}")

	target_scaler_path = _require_file(
		saved_dir / f"{symbol}_1d_regressor_final_target_scaler.pkl",
		"target scaler",
	)
	target_scaler = _load_pickle(target_scaler_path)

	target_meta_path = _require_file(
		saved_dir / f"{symbol}_1d_regressor_final_metadata.pkl",
		"regressor metadata",
	)
	target_metadata = dict(_load_pickle(target_meta_path) or {})
	sequence_length = int(target_metadata.get("sequence_length", DEFAULT_SEQUENCE_LENGTH))

	# Verify feature count matches model metadata where possible
	expected_n_features = int(target_metadata.get('n_features')) if 'n_features' in target_metadata else None
	if expected_n_features is not None and expected_n_features != len(feature_columns):
		raise ValueError(
			f"Feature count mismatch: Model expects {expected_n_features} features but current feature set has {len(feature_columns)}" 
		)

	classifier_metadata_path = saved_dir / f"{symbol}_binary_classifiers_final_metadata.pkl"
	classifier_metadata = dict(_load_pickle(classifier_metadata_path) or {})

	regressor_weights = _require_file(
		saved_dir / f"{symbol}_1d_regressor_final.weights.h5",
		"regressor weights",
	)
	buy_classifier_weights = sorted(saved_dir.glob(f"{symbol}_is_buy_classifier_final*.weights.h5"))
	sell_classifier_weights = sorted(saved_dir.glob(f"{symbol}_is_sell_classifier_final*.weights.h5"))
	if not buy_classifier_weights:
		raise FileNotFoundError(f"No buy classifier weights found for {symbol}")
	if not sell_classifier_weights:
		raise FileNotFoundError(f"No sell classifier weights found for {symbol}")

	return ArtifactBundle(
		symbol=symbol,
		sequence_length=sequence_length,
		feature_columns=feature_columns,
		feature_scaler=feature_scaler,
		target_scaler=target_scaler,
		target_metadata=target_metadata,
		classifier_metadata=classifier_metadata,
		regressor_weights=regressor_weights,
		buy_classifier_weights=buy_classifier_weights,
		sell_classifier_weights=sell_classifier_weights,
	)


def create_sequences(features: np.ndarray, sequence_length: int) -> np.ndarray:
	if features.shape[0] < sequence_length:
		raise ValueError("Not enough feature rows to create sequences")
	return np.array([features[i : i + sequence_length] for i in range(features.shape[0] - sequence_length + 1)])


def create_binary_classifier(sequence_length: int, n_features: int, name: str, arch: dict | None = None) -> keras.Model:
	arch = arch or {}
	base = LSTMTransformerPaper(
		sequence_length=sequence_length,
		n_features=n_features,
		lstm_units=int(arch.get('lstm_units', 64)),
		d_model=int(arch.get('d_model', 128)),
		num_heads=int(arch.get('num_heads', 4)),
		num_blocks=int(arch.get('num_blocks', 6)),
		ff_dim=int(arch.get('ff_dim', 256)),
		dropout=float(arch.get('dropout', 0.35)),
	)
	dummy = tf.random.normal((1, sequence_length, n_features))
	_ = base(dummy)
	print(f"Built classifier base with architecture: lstm_units={int(arch.get('lstm_units', 64))}, d_model={int(arch.get('d_model', 128))}," \
		f" num_blocks={int(arch.get('num_blocks', 6))}, ff_dim={int(arch.get('ff_dim', 256))}")

	inputs = keras.Input(shape=(sequence_length, n_features))
	x = base.lstm_layer(inputs)
	x = base.projection(x)
	x = x + base.pos_encoding[:, :sequence_length, :]
	for block in base.transformer_blocks:
		attn = block["attention"](x, x)
		attn = block["dropout1"](attn)
		x = block["norm1"](x + attn)
		ffn = block["ffn2"](block["ffn1"](x))
		ffn = block["dropout2"](ffn)
		x = block["norm2"](x + ffn)
	x = base.global_pool(x)
	x = base.dropout_out(x)
	output = keras.layers.Dense(1, activation="sigmoid", name=f"{name}_output")(x)
	model = keras.Model(inputs=inputs, outputs=output, name=name)
	model.trainable = False
	return model


def _build_regressor(bundle: ArtifactBundle) -> keras.Model:
	key = (bundle.symbol, len(bundle.feature_columns), bundle.sequence_length)
	model = _REGRESSOR_CACHE.get(key)
	if model is None:
		# Try loading complete keras model first (preferred)
		keras_model_path = bundle.regressor_weights.parent / f"{bundle.symbol}_1d_regressor_final_model.keras"
		if keras_model_path.exists():
			print(f"Loading complete model from {keras_model_path}")
			# CRITICAL: Pass custom_objects to properly deserialize LSTMTransformerPaper class
			custom_objects = get_custom_objects()
			model = keras.models.load_model(str(keras_model_path), custom_objects=custom_objects)
			print(f"Successfully loaded regressor model (type: {type(model).__name__})")
		else:
			# Fall back to building architecture and loading .h5 weights
			arch = bundle.target_metadata.get('architecture', {}) if hasattr(bundle, 'target_metadata') else {}
			model = LSTMTransformerPaper(
				sequence_length=bundle.sequence_length,
				n_features=len(bundle.feature_columns),
				lstm_units=int(arch.get('lstm_units', 64)),
				d_model=int(arch.get('d_model', 128)),
				num_heads=int(arch.get('num_heads', 4)),
				num_blocks=int(arch.get('num_blocks', 6)),
				ff_dim=int(arch.get('ff_dim', 256)),
				dropout=float(arch.get('dropout', 0.3)),
			)
			dummy = tf.random.normal((1, bundle.sequence_length, len(bundle.feature_columns)))
			_ = model(dummy)
			print(f"Built regressor with architecture: lstm_units={int(arch.get('lstm_units', 64))}, d_model={int(arch.get('d_model', 128))}," \
				f" num_blocks={int(arch.get('num_blocks', 6))}, ff_dim={int(arch.get('ff_dim', 256))}")
			model.load_weights(str(bundle.regressor_weights))
			print(f"Loaded weights from {bundle.regressor_weights}")

		model.trainable = False
		_REGRESSOR_CACHE[key] = model
	return model


def _build_classifier(weight_path: Path, bundle: ArtifactBundle, label: str) -> keras.Model:
	key = (bundle.symbol, weight_path.name)
	model = _CLASSIFIER_CACHE.get(key)
	if model is None:
		arch = bundle.classifier_metadata.get('architecture', {}) if hasattr(bundle, 'classifier_metadata') else {}
		if not arch:
			# Fallback to regressor architecture if classifier arch is missing
			arch = bundle.target_metadata.get('architecture', {}) if hasattr(bundle, 'target_metadata') else {}
		classifier = create_binary_classifier(
			bundle.sequence_length,
			len(bundle.feature_columns),
			name=f"{label}_{weight_path.stem}",
			arch=arch,
		)
		classifier.load_weights(str(weight_path))
		classifier.trainable = False
		model = classifier
		_CLASSIFIER_CACHE[key] = model
	return model


_REGRESSOR_CACHE: Dict[tuple[str, int, int], keras.Model] = {}
_CLASSIFIER_CACHE: Dict[tuple[str, str], keras.Model] = {}


# Data preparation --------------------------------------------------------

def _prepare_sequence_dataset(bundle: ArtifactBundle, min_rows: int) -> Dict[str, np.ndarray | List[str]]:
	# Use DataCacheManager to fetch and persist engineered/prepared artifacts
	from data.cache_manager import DataCacheManager
	cache_manager = DataCacheManager()
	raw_df, engineered_df, prepared_df, feature_cols = cache_manager.get_or_fetch_data(
		symbol=bundle.symbol,
		include_sentiment=True,
		force_refresh=False
	)
	df = engineered_df
	df_clean, _ = prepare_training_data(df, horizons=[1])

	feature_matrix = df_clean[bundle.feature_columns].values
	feature_scaled = bundle.feature_scaler.transform(feature_matrix)
	sequences = create_sequences(feature_scaled, bundle.sequence_length)

	target_values = df_clean["target_1d"].values[bundle.sequence_length - 1 :]
	close_prices = df_clean["Close"].values[bundle.sequence_length - 1 :]
	open_prices = df_clean["Open"].values[bundle.sequence_length - 1 :]
	high_prices = df_clean["High"].values[bundle.sequence_length - 1 :]
	low_prices = df_clean["Low"].values[bundle.sequence_length - 1 :]
	volumes = df_clean["Volume"].values[bundle.sequence_length - 1 :]
	dates = df_clean.index[bundle.sequence_length - 1 :]

	if sequences.shape[0] != target_values.shape[0]:
		raise ValueError("Sequence/target length mismatch after alignment")

	window = min(sequences.shape[0], max(min_rows, bundle.sequence_length))
	start_idx = sequences.shape[0] - window

	def _slice(arr: np.ndarray) -> np.ndarray:
		return arr[start_idx:]

	sliced_sequences = sequences[start_idx:]
	return {
		"X": sliced_sequences,
		"actual_returns": _slice(target_values),
		"prices": _slice(close_prices),
		"opens": _slice(open_prices),
		"highs": _slice(high_prices),
		"lows": _slice(low_prices),
		"volumes": _slice(volumes),
		"dates": [pd.to_datetime(d).strftime("%Y-%m-%d") for d in dates[start_idx:]],
	}


def _apply_smoothing(values: np.ndarray, method: str) -> np.ndarray:
	if method == "ema":
		series = pd.Series(values)
		return series.ewm(span=5, adjust=False).mean().to_numpy()
	if method == "moving-average":
		series = pd.Series(values)
		return series.rolling(window=5, min_periods=1).mean().to_numpy()
	return values


def _next_business_days(start_date: str, horizon: int) -> List[str]:

	if horizon <= 0:
		return []
	start_ts = pd.Timestamp(start_date)
	future_index = pd.bdate_range(start=start_ts + BDay(1), periods=horizon)
	return [ts.strftime("%Y-%m-%d") for ts in future_index]


def _compute_smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	denom = (np.abs(y_true) + np.abs(y_pred)) / 2.0
	denom = np.where(denom == 0, 1e-9, denom)
	return float(np.mean(np.abs(y_true - y_pred) / denom) * 100.0)


def _compute_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
	if y_true.size <= 1 or y_pred.size <= 1:
		return 0.0
	corr = np.corrcoef(y_true, y_pred)[0, 1]
	return float(corr) if np.isfinite(corr) else 0.0


def compute_regressor_positions(
	predicted_returns: np.ndarray,
	*,
	threshold: float,
	scale: float,
	max_short: float,
	strategy: str = "confidence_scaling",
) -> np.ndarray:
	predicted_returns = np.asarray(predicted_returns, dtype=float)
	if strategy == "simple_threshold":
		positions = np.zeros_like(predicted_returns)
		positions[predicted_returns > threshold] = 1.0
		positions[predicted_returns < -threshold] = -max_short
		return positions
	if strategy == "confidence_scaling":
		return np.clip(predicted_returns * scale, -max_short, 1.0)
	raise ValueError(f"Unsupported regressor strategy: {strategy}")


def compute_classifier_positions(
	signals: np.ndarray,
	buy_probs: np.ndarray,
	sell_probs: np.ndarray,
	*,
	max_short: float,
) -> np.ndarray:
	"""
	Compute position sizes from classifier probabilities.
	
	Classifier-only mode: Uses classifier probabilities directly for position sizing,
	ignoring regressor entirely. Position sizes are clipped to meaningful ranges:
	- BUY positions: 0.3 to 1.0 (minimum 30% position when signal fires)
	- SELL positions: 0.1 to 0.5 (with 0.5x multiplier for risk management)
	
	Args:
		signals: Array of signals (1=BUY, -1=SELL, 0=HOLD)
		buy_probs: BUY classifier probabilities
		sell_probs: SELL classifier probabilities
		max_short: Maximum short exposure (typically 0.5)
	
	Returns:
		Array of position sizes
	"""
	signals = np.asarray(signals, dtype=float)
	positions = np.zeros_like(signals)
	
	buy_mask = signals == 1
	sell_mask = signals == -1
	
	# BUY positions: use buy probability as position size, clipped to [0.3, 1.0]
	if np.any(buy_mask):
		positions[buy_mask] = np.clip(buy_probs[buy_mask], 0.3, 1.0)
	
	# SELL positions: use sell probability with 0.5x multiplier, clipped to [0.1, 0.5]
	if np.any(sell_mask):
		positions[sell_mask] = -np.clip(sell_probs[sell_mask] * 0.5, 0.1, max_short)
	
	return positions


def fuse_positions(
	mode: str,
	classifier_positions: np.ndarray,
	regressor_positions: np.ndarray,
	signals: np.ndarray,
	*,
	max_short: float,
) -> np.ndarray:
	"""
	Fuse classifier and regressor positions based on fusion mode.
	
	Fusion Modes:
	- 'classifier': Ignores regressor entirely, uses classifier probabilities for position sizing.
	               Returns classifier_positions directly with no regressor reference.
	- 'regressor': Uses only regressor predictions for position sizing.
	- 'hybrid': Uses classifier when signals fire, regressor positions for HOLD signals.
	- 'weighted': Amplifies classifier positions by regressor magnitude.
	
	Args:
		mode: Fusion mode ('classifier', 'regressor', 'hybrid', 'weighted')
		classifier_positions: Position sizes from classifier
		regressor_positions: Position sizes from regressor
		signals: Array of signals (1=BUY, -1=SELL, 0=HOLD)
		max_short: Maximum short exposure
	
	Returns:
		Fused position sizes array
	"""
	mode = (mode or "classifier").lower()
	if mode == "classifier":
		# Classifier-only mode: use classifier positions directly, ignore regressor completely
		return classifier_positions
	if mode == "regressor":
		return regressor_positions
	if mode == "hybrid":
		combined = classifier_positions.copy()
		mask = signals == 0
		combined[mask] = regressor_positions[mask]
		return np.clip(combined, -max_short, 1.0)
	if mode == "weighted":
		weight = 1.0 + np.clip(np.abs(regressor_positions), 0.0, 1.0)
		combined = classifier_positions * weight
		fallback = signals == 0
		combined[fallback] = regressor_positions[fallback]
		return np.clip(combined, -max_short, 1.0)
	return classifier_positions


def simulate_price_path(start_price: float, predicted_returns: np.ndarray) -> np.ndarray:
	prices = []
	current = float(start_price)
	for delta in predicted_returns:
		current = current * (1.0 + float(delta))
		prices.append(current)
	return np.asarray(prices, dtype=float)


def _build_backtest_metrics(
	actual_returns: np.ndarray,
	predicted_returns: np.ndarray,
	daily_returns: np.ndarray,
	trade_log: List[Dict[str, float]],
) -> Dict[str, float]:
	if daily_returns.size == 0:
		return {
			"cumulativeReturn": 0.0,
			"sharpeRatio": 0.0,
			"maxDrawdown": 0.0,
			"winRate": 0.0,
			"averageTradeProfit": 0.0,
			"totalTrades": 0,
			"directionalAccuracy": 0.0,
			"correlation": 0.0,
			"smape": 0.0,
			"rmse": 0.0,
		}

	cum_return = float(np.prod(1.0 + daily_returns) - 1.0)
	avg_daily = float(np.mean(daily_returns))
	std_daily = float(np.std(daily_returns))
	sharpe = (avg_daily / std_daily * np.sqrt(252)) if std_daily > 0 else 0.0
	equity = np.cumprod(1.0 + daily_returns)
	peak = np.maximum.accumulate(equity)
	drawdown = (equity - peak) / peak
	max_drawdown = float(drawdown.min()) if drawdown.size else 0.0
	win_rate = float(np.mean(daily_returns > 0))
	avg_trade_profit = float(np.mean([entry["pnl"] for entry in trade_log])) if trade_log else 0.0
	total_trades = len(trade_log)
	directional_accuracy = float(np.mean(np.sign(actual_returns) == np.sign(predicted_returns))) if actual_returns.size else 0.0
	correlation = _compute_correlation(actual_returns, predicted_returns)
	smape = _compute_smape(actual_returns, predicted_returns) if actual_returns.size else 0.0
	rmse = float(np.sqrt(np.mean((actual_returns - predicted_returns) ** 2))) if actual_returns.size else 0.0

	return {
		"cumulativeReturn": cum_return,
		"sharpeRatio": sharpe,
		"maxDrawdown": max_drawdown,
		"winRate": win_rate,
		"averageTradeProfit": avg_trade_profit,
		"totalTrades": total_trades,
		"directionalAccuracy": directional_accuracy,
		"correlation": correlation,
		"smape": smape,
		"rmse": rmse,
	}


def _render_backtest_csv(
	dates: Sequence[str],
	prices: np.ndarray,
	actual_returns: np.ndarray,
	predicted_returns: np.ndarray,
	positions: np.ndarray,
) -> str:
	frame = pd.DataFrame(
		{
			"date": list(dates),
			"price": prices,
			"actual_return": actual_returns,
			"predicted_return": predicted_returns,
			"position": positions,
		}
	)
	return frame.to_csv(index=False)


# Public API ---------------------------------------------------------------

def generate_prediction(payload: Dict) -> Dict:
	symbol = str(payload.get("symbol", "AAPL")).upper()
	days_on_chart = int(payload.get("daysOnChart", 120))
	smoothing = str(payload.get("smoothing", "none"))
	horizon = int(payload.get("horizon", 5))
	confidence_floors = payload.get("confidenceFloors") or {}
	fusion = payload.get("fusion") or {}
	trade_share_floor = float(payload.get("tradeShareFloor", DEFAULT_TRADE_SHARE_FLOOR))
	history_window = max(days_on_chart, FORECAST_MIN_HISTORY)

	buy_floor = float(confidence_floors.get("buy", 0.3))
	sell_floor = float(confidence_floors.get("sell", 0.45))
	fusion_mode = fusion.get("mode", "classifier")
	reg_scale = float(fusion.get("regressorScale", 15.0))
	reg_threshold = float(fusion.get("buyThreshold", 0.3)) * 0.25

	bundle = load_artifacts(symbol)
	dataset = _prepare_sequence_dataset(
		bundle,
		min_rows=max(history_window + horizon, DEFAULT_LOOKBACK),
	)

	X_seq = dataset["X"]
	actual_returns = np.asarray(dataset["actual_returns"], dtype=float)
	prices = np.asarray(dataset["prices"], dtype=float)
	dates = dataset["dates"]

	reg_model = _build_regressor(bundle)
	y_pred_scaled = reg_model.predict(X_seq, verbose=0).flatten()
	y_pred_transformed = bundle.target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
	if bundle.target_metadata.get("target_transform") == "log1p":
		predicted_returns = np.expm1(y_pred_transformed)
	else:
		predicted_returns = y_pred_transformed

	buy_probs_list = []
	for weight_path in bundle.buy_classifier_weights:
		buy_model = _build_classifier(weight_path, bundle, "buy")
		buy_probs_list.append(buy_model.predict(X_seq, verbose=0).flatten())
	sell_probs_list = []
	for weight_path in bundle.sell_classifier_weights:
		sell_model = _build_classifier(weight_path, bundle, "sell")
		sell_probs_list.append(sell_model.predict(X_seq, verbose=0).flatten())

	buy_probs = np.mean(buy_probs_list, axis=0)
	sell_probs = np.mean(sell_probs_list, axis=0)
	hold_probs = np.clip(1.0 - buy_probs - sell_probs, 0.0, 1.0)

	buy_signals = (buy_probs >= buy_floor).astype(int)
	sell_signals = (sell_probs >= sell_floor).astype(int)
	combined_signals = buy_signals - sell_signals
	conflict = (buy_signals == 1) & (sell_signals == 1)
	combined_signals[conflict] = 0

	regressor_positions = compute_regressor_positions(
		predicted_returns,
		threshold=reg_threshold,
		scale=reg_scale,
		max_short=DEFAULT_MAX_SHORT,
		strategy="confidence_scaling",
	)
	classifier_positions = compute_classifier_positions(
		combined_signals,
		buy_probs,
		sell_probs,
		max_short=DEFAULT_MAX_SHORT,
	)
	fused_positions = fuse_positions(
		fusion_mode,
		classifier_positions,
		regressor_positions,
		combined_signals,
		max_short=DEFAULT_MAX_SHORT,
	)

	predicted_prices = prices * (1.0 + predicted_returns)
	predicted_prices = _apply_smoothing(predicted_prices, smoothing)
	predicted_returns_smoothed = _apply_smoothing(predicted_returns, smoothing)
	
	# Calculate ATR for volatility-based position sizing
	atr_values = calculate_atr(prices)

	classifier_probabilities = [
		{
			"buy": float(buy_probs[i]),
			"sell": float(sell_probs[i]),
			"hold": float(hold_probs[i]),
		}
		for i in range(len(buy_probs))
	]

	forecast_horizon = max(horizon, 0)
	if forecast_horizon and forecast_horizon > len(predicted_returns):
		forecast_horizon = len(predicted_returns)
	forecast_returns = (
		predicted_returns[-forecast_horizon:]
		if forecast_horizon > 0
		else np.asarray([], dtype=float)
	)
	forecast_dates = _next_business_days(dates[-1], forecast_horizon) if forecast_horizon else []
	forecast_prices = (
		simulate_price_path(prices[-1], forecast_returns)
		if forecast_horizon > 0
		else np.asarray([], dtype=float)
	)
	forecast_positions = (
		fused_positions[-forecast_horizon:]
		if forecast_horizon > 0
		else np.asarray([], dtype=float)
	)
	future_start_idx = len(prices) - forecast_horizon if forecast_horizon else len(prices)
	trade_markers: List[Dict[str, object]] = []
	current_shares = 0.0
	available_cash = HYBRID_REFERENCE_EQUITY
	high_water_mark = HYBRID_REFERENCE_EQUITY  # Initialize high-water mark for drawdown tracking

	# Position tracking for scale-out strategy
	entry_price = 0.0
	initial_shares = 0.0
	profit_targets_hit = [False] * len(PROFIT_TARGETS)
	current_stop_loss = INITIAL_STOP_LOSS
	highest_price_since_entry = 0.0
	
	# Profit-target order queue
	trade_orders_queue: List[Dict[str, object]] = []
	
	# Tolerance band rebalancing state
	position_state = {
		'target_weight': TARGET_POSITION_WEIGHT,
		'current_weight': 0.0,
		'upper_band': UPPER_REBALANCE_BAND,
		'lower_band': LOWER_REBALANCE_BAND
	}
	
	for idx, signal in enumerate(combined_signals):
		segment = "forecast" if idx >= future_start_idx else "history"
		price_reference = float(
			forecast_prices[idx - future_start_idx]
			if (segment == "forecast" and forecast_prices.size and idx - future_start_idx < forecast_prices.size)
			else prices[idx]
		)
		
		# Check and execute pending profit-target orders
		if trade_orders_queue and current_shares > 0 and price_reference > 0:
			for order in trade_orders_queue:
				if not order.get('executed', False) and price_reference >= order['trigger_price']:
					# Execute profit-target order
					shares_to_sell = min(order['shares_to_sell'], current_shares)
					if shares_to_sell >= HYBRID_MIN_SHARES_THRESHOLD:
						available_cash += shares_to_sell * price_reference
						current_shares -= shares_to_sell
						order['executed'] = True
						
						trade_markers.append({
							"date": segment == "forecast" and forecast_dates and (idx - future_start_idx) < len(forecast_dates)
								and forecast_dates[idx - future_start_idx]
								or dates[idx],
							"price": price_reference,
							"type": "sell",
							"shares": shares_to_sell,
							"confidence": 1.0,
							"segment": segment,
							"scope": "prediction",
							"explanation": f"PROFIT-TARGET ORDER at +{order['target_level']*100:.0f}% (25% position)",
						})
						
						# If position reduced below threshold, close entirely and cancel orders
						if current_shares < HYBRID_MIN_SHARES_THRESHOLD:
							available_cash += current_shares * price_reference
							current_shares = 0.0
							entry_price = 0.0
							initial_shares = 0.0
							trade_orders_queue = []
							profit_targets_hit = [False] * len(PROFIT_TARGETS)
							current_stop_loss = INITIAL_STOP_LOSS
							highest_price_since_entry = 0.0
							break
		
		confidence_delta = float(buy_probs[idx] - sell_probs[idx])
		effective_signal = signal if abs(fused_positions[idx]) >= HYBRID_POSITION_EPS else 0
		target_fraction = 0.0
		shares_to_trade = 0.0
		action_signal = 0
		block_reason: str | None = None
		equity = max(available_cash + current_shares * price_reference, 1e-6)
		
		# Update high-water mark and calculate drawdown adjustments
		high_water_mark = max(high_water_mark, equity)
		drawdown_adj = calculate_drawdown_adjustments(equity, high_water_mark)
		drawdown_max_position = drawdown_adj['max_position']
		drawdown_scale_factor = drawdown_adj['scale_factor']
		drawdown_multiplier = drawdown_adj['multiplier']
		
		# Apply volatility-based position sizing adjustment
		current_atr = float(atr_values[idx]) if idx < len(atr_values) else BASELINE_VOLATILITY
		volatility_adjusted_scale = calculate_volatility_adjusted_scale(current_atr)
		
		# Apply drawdown-based scale factor reduction
		effective_scale_factor = min(volatility_adjusted_scale, drawdown_scale_factor)
		
		# Apply concentration-based position limits
		current_position_pct = (current_shares * price_reference) / equity if equity > 0 and price_reference > 0 else 0.0
		concentration_factor = max(0.5, 1.0 - (current_position_pct / drawdown_max_position))  # Use drawdown-adjusted max
		
		# Update current position weight
		current_position_value = current_shares * price_reference
		position_state['current_weight'] = current_position_value / equity if equity > 0 else 0.0
		
		# Tolerance band rebalancing check (before signal processing)
		if current_shares > 0 and price_reference > 0:
			# Check if position exceeds upper band - trigger partial sell
			if position_state['current_weight'] > position_state['upper_band']:
				excess_weight = position_state['current_weight'] - position_state['target_weight']
				excess_value = excess_weight * equity
				shares_to_rebalance = excess_value / price_reference
				
				if shares_to_rebalance >= HYBRID_MIN_SHARES_THRESHOLD:
					# Execute rebalancing sell
					available_cash += shares_to_rebalance * price_reference
					current_shares -= shares_to_rebalance
					
					trade_markers.append({
						"date": segment == "forecast" and forecast_dates and (idx - future_start_idx) < len(forecast_dates)
							and forecast_dates[idx - future_start_idx]
							or dates[idx],
						"price": price_reference,
						"type": "sell",
						"shares": shares_to_rebalance,
						"confidence": 1.0,
						"segment": segment,
						"scope": "prediction",
						"explanation": f"REBALANCE: Position {position_state['current_weight']*100:.1f}% > {position_state['upper_band']*100:.0f}% upper band",
					})
					
					# Update position weight after rebalance
					current_position_value = current_shares * price_reference
					position_state['current_weight'] = current_position_value / equity if equity > 0 else 0.0
			
			# Check if position falls below lower band - trigger buy if BUY signal active
			elif position_state['current_weight'] < position_state['lower_band'] and effective_signal > 0:
				deficit_weight = position_state['target_weight'] - position_state['current_weight']
				deficit_value = deficit_weight * equity
				shares_to_add = deficit_value / price_reference
				max_affordable = available_cash / price_reference
				shares_to_add = min(shares_to_add, max_affordable)
				
				if shares_to_add >= HYBRID_MIN_SHARES_THRESHOLD:
					# Execute rebalancing buy
					available_cash -= shares_to_add * price_reference
					current_shares += shares_to_add
					available_cash = max(available_cash, 0.0)
					
					# Update average entry price
					if entry_price > 0:
						total_cost = (initial_shares * entry_price) + (shares_to_add * price_reference)
						new_total_shares = initial_shares + shares_to_add
						entry_price = total_cost / new_total_shares if new_total_shares > 0 else price_reference
						initial_shares = new_total_shares
					
					trade_markers.append({
						"date": segment == "forecast" and forecast_dates and (idx - future_start_idx) < len(forecast_dates)
							and forecast_dates[idx - future_start_idx]
							or dates[idx],
						"price": price_reference,
						"type": "buy",
						"shares": shares_to_add,
						"confidence": 1.0,
						"segment": segment,
						"scope": "prediction",
						"explanation": f"REBALANCE: Position {position_state['current_weight']*100:.1f}% < {position_state['lower_band']*100:.0f}% lower band",
					})
					
					# Update position weight after rebalance
					current_position_value = current_shares * price_reference
					position_state['current_weight'] = current_position_value / equity if equity > 0 else 0.0
		
		# Update highest price tracking for trailing stop
		if current_shares > 0 and price_reference > highest_price_since_entry:
			highest_price_since_entry = price_reference
		
		# Check for stop-loss exit
		if current_shares > 0 and entry_price > 0 and price_reference > 0:
			unrealized_pnl_pct = (price_reference - entry_price) / entry_price
			
			# Update trailing stop based on profit levels
			for profit_level, stop_level in TRAILING_STOP_LEVELS:
				if unrealized_pnl_pct >= profit_level:
					current_stop_loss = stop_level
					break
			
			# Check if stop-loss triggered
			stop_price = highest_price_since_entry * (1 - current_stop_loss)
			if price_reference <= stop_price:
				# Stop-loss hit - exit entire position and cancel orders
				shares_delta = -current_shares
				available_cash += current_shares * price_reference
				current_shares = 0.0
				entry_price = 0.0
				initial_shares = 0.0
				trade_orders_queue = []  # Cancel all pending orders
				profit_targets_hit = [False] * len(PROFIT_TARGETS)
				current_stop_loss = INITIAL_STOP_LOSS
				highest_price_since_entry = 0.0
				
				trade_markers.append({
					"date": segment == "forecast" and forecast_dates and (idx - future_start_idx) < len(forecast_dates)
						and forecast_dates[idx - future_start_idx]
						or dates[idx],
					"price": price_reference,
					"type": "sell",
					"shares": abs(shares_delta),
					"confidence": 1.0,
					"segment": segment,
					"scope": "prediction",
					"explanation": f"STOP-LOSS triggered at {current_stop_loss*100:.2f}% (P&L: {unrealized_pnl_pct*100:.1f}%)",
				})
				continue
			
			# Check profit targets for scale-out
			for target_idx, (target_pct, scale_out_pct) in enumerate(zip(PROFIT_TARGETS, SCALE_OUT_PERCENTAGES)):
				if not profit_targets_hit[target_idx] and unrealized_pnl_pct >= target_pct:
					# Scale out at this profit target
					shares_to_sell = initial_shares * scale_out_pct
					shares_to_sell = min(shares_to_sell, current_shares)
					
					if shares_to_sell >= HYBRID_MIN_SHARES_THRESHOLD:
						available_cash += shares_to_sell * price_reference
						current_shares -= shares_to_sell
						profit_targets_hit[target_idx] = True
						
						trade_markers.append({
							"date": segment == "forecast" and forecast_dates and (idx - future_start_idx) < len(forecast_dates)
								and forecast_dates[idx - future_start_idx]
								or dates[idx],
							"price": price_reference,
							"type": "sell",
							"shares": shares_to_sell,
							"confidence": 1.0,
							"segment": segment,
							"scope": "prediction",
							"explanation": f"SCALE-OUT at +{target_pct*100:.0f}% profit ({scale_out_pct*100:.0f}% position)",
						})
						
						# If position reduced below threshold, close entirely
						if current_shares < HYBRID_MIN_SHARES_THRESHOLD:
							available_cash += current_shares * price_reference
							current_shares = 0.0
							entry_price = 0.0
							initial_shares = 0.0
							profit_targets_hit = [False] * len(PROFIT_TARGETS)
							current_stop_loss = INITIAL_STOP_LOSS
							highest_price_since_entry = 0.0
							break
		
		if effective_signal != 0 and price_reference > 0.0:
			direction = 1 if effective_signal > 0 else -1
			confidence_strength = max(abs(confidence_delta), HYBRID_MIN_CONFIDENCE)
			if direction > 0:
				raw_fraction = HYBRID_BASE_FRACTION + confidence_strength * effective_scale_factor * concentration_factor * drawdown_multiplier
				target_fraction = float(np.clip(raw_fraction, HYBRID_BASE_FRACTION, drawdown_max_position))  # Use drawdown-adjusted max
			else:
				target_fraction = 0.0
			target_position_value = target_fraction * equity
			target_shares = target_position_value / price_reference
			shares_delta = target_shares - current_shares
			if shares_delta > 0:
				max_affordable = available_cash / price_reference
				if max_affordable <= 0:
					shares_delta = 0.0
					block_reason = "Insufficient cash"
				else:
					shares_delta = min(shares_delta, max_affordable)
					if shares_delta == 0.0:
						block_reason = "Insufficient cash"
			elif shares_delta < 0:
				sellable = current_shares
				if sellable <= 0:
					shares_delta = 0.0
					block_reason = "No holdings"
				else:
					shares_delta = -min(-shares_delta, sellable)
					if shares_delta == 0.0:
						block_reason = "No holdings"
			if abs(shares_delta) < HYBRID_MIN_SHARES_THRESHOLD:
				shares_delta = 0.0
			else:
				cost = shares_delta * price_reference
				available_cash -= cost
				
				# Track entry for scale-out strategy
				if shares_delta > 0:  # BUY
					if current_shares == 0:
						# Fresh entry - create profit-target orders
						entry_price = price_reference
						initial_shares = shares_delta
						profit_targets_hit = [False] * len(PROFIT_TARGETS)
						current_stop_loss = INITIAL_STOP_LOSS
						highest_price_since_entry = price_reference
						
						# Create pending profit-target orders
						trade_orders_queue = []
						for level_pct, size_pct in zip(PROFIT_TARGET_LEVELS, PROFIT_TARGET_SIZES):
							trigger_price = entry_price * (1.0 + level_pct)
							shares_to_sell = shares_delta * size_pct
							trade_orders_queue.append({
								'date_idx': idx,
								'trigger_price': trigger_price,
								'shares_to_sell': shares_to_sell,
								'target_level': level_pct,
								'executed': False
							})
					else:
						# Adding to position - update average entry and recreate orders
						total_cost = (current_shares * entry_price) + (shares_delta * price_reference)
						new_total_shares = current_shares + shares_delta
						entry_price = total_cost / new_total_shares if new_total_shares > 0 else price_reference
						initial_shares = new_total_shares
						profit_targets_hit = [False] * len(PROFIT_TARGETS)
						highest_price_since_entry = max(highest_price_since_entry, price_reference)
						
						# Recreate profit-target orders with new average entry
						trade_orders_queue = []
						for level_pct, size_pct in zip(PROFIT_TARGET_LEVELS, PROFIT_TARGET_SIZES):
							trigger_price = entry_price * (1.0 + level_pct)
							shares_to_sell = new_total_shares * size_pct
							trade_orders_queue.append({
								'date_idx': idx,
								'trigger_price': trigger_price,
								'shares_to_sell': shares_to_sell,
								'target_level': level_pct,
								'executed': False
							})
				else:  # SELL
					# Full exit on SELL signal - cancel all orders
					if abs(shares_delta) >= abs(current_shares) * 0.9:  # Nearly full exit
						entry_price = 0.0
						initial_shares = 0.0
						trade_orders_queue = []  # Cancel all pending orders
						profit_targets_hit = [False] * len(PROFIT_TARGETS)
						current_stop_loss = INITIAL_STOP_LOSS
						highest_price_since_entry = 0.0
				
				current_shares += shares_delta
				action_signal = 1 if shares_delta > 0 else -1
				available_cash = max(available_cash, 0.0)
			shares_to_trade = shares_delta
		shares = float(abs(shares_to_trade))
		if action_signal == 0 and shares < trade_share_floor:
			continue
		if shares < trade_share_floor or action_signal == 0:
			marker_type = "hold"
		else:
			marker_type = "buy" if action_signal == 1 else "sell"
		confidence = float(
			buy_probs[idx]
			if action_signal == 1
			else (
				sell_probs[idx]
				if action_signal == -1
				else max(hold_probs[idx], 1.0 - buy_probs[idx] - sell_probs[idx])
			)
		)
		trade_markers.append(
			{
				"date": segment == "forecast" and forecast_dates and (idx - future_start_idx) < len(forecast_dates)
					and forecast_dates[idx - future_start_idx]
					or dates[idx],
				"price": price_reference,
				"type": marker_type,
				"shares": shares,
				"confidence": confidence,
				"segment": segment,
				"scope": "prediction",
				"explanation": (
					block_reason
					if marker_type == "hold" and block_reason
					else f"{marker_type.upper()} signal"
				),
			}
		)

	overlays = [
		{
			"type": "predicted-path",
			"points": [
				{"date": date, "value": float(value)}
				for date, value in zip(dates, predicted_prices)
			],
		}
	]

	candles = [
		{
			"date": dates[i],
			"open": float(dataset["opens"][i]),
			"high": float(dataset["highs"][i]),
			"low": float(dataset["lows"][i]),
			"close": float(prices[i]),
			"volume": float(dataset["volumes"][i]),
		}
		for i in range(len(dates))
	]

	metadata = {
		"modelId": payload.get("modelId") or f"{symbol.lower()}-regressor",
		"fusionMode": fusion_mode,
		"buyThreshold": buy_floor,
		"sellThreshold": sell_floor,
		"horizon": horizon,
		"tradeShareFloor": trade_share_floor,
	}

	return {
		"symbol": symbol,
		"dates": dates,
		"prices": prices.tolist(),
		"actualReturns": actual_returns.tolist(),
		"predictedPrices": predicted_prices.tolist(),
		"predictedReturns": predicted_returns_smoothed.tolist(),
		"fusedPositions": fused_positions.tolist(),
		"classifierProbabilities": classifier_probabilities,
		"tradeMarkers": trade_markers,
		"overlays": overlays,
		"candles": candles,
		"forecast": {
			"dates": forecast_dates,
			"prices": forecast_prices.tolist() if forecast_prices.size else [],
			"returns": forecast_returns.tolist() if forecast_returns.size else [],
			"positions": forecast_positions.tolist() if forecast_positions.size else [],
		},
		"metadata": metadata,
	}


def run_backtest(prediction: Dict, params: Dict) -> Dict:
	if not prediction:
		raise ValueError("Prediction payload is required for backtest")

	dates: List[str] = list(prediction.get("dates") or [])
	prices = np.asarray(prediction.get("prices") or [], dtype=float)
	actual_returns = np.asarray(prediction.get("actualReturns") or [], dtype=float)
	positions = np.asarray(prediction.get("fusedPositions") or [], dtype=float)
	predicted_returns = np.asarray(prediction.get("predictedReturns") or [], dtype=float)
	window = int(params.get("backtestWindow") or len(actual_returns))

	if len(dates) == 0 or prices.size == 0 or actual_returns.size == 0:
		raise ValueError("Prediction result missing series required for backtest")

	window = min(window, len(dates))
	if window <= 5:
		raise ValueError("Not enough datapoints for backtest window")

	def _slice(series: np.ndarray) -> np.ndarray:
		return series[-window:]

	dates = dates[-window:]
	prices = _slice(prices)
	actual_returns = _slice(actual_returns)
	positions = _slice(positions)
	predicted_returns = _slice(predicted_returns)

	initial_capital = float(params.get("initialCapital", 10_000))
	trade_share_floor = float(params.get("tradeShareFloor", DEFAULT_TRADE_SHARE_FLOOR))
	backtester = AdvancedBacktester(initial_capital=initial_capital)
	results = backtester.backtest_with_positions(
		dates=dates,
		prices=prices,
		returns=actual_returns,
		positions=positions,
		max_long=float(params.get("maxLong", 1.0)),
		max_short=float(params.get("maxShort", DEFAULT_MAX_SHORT)),
	)

	equity = results.equity_curve[1:]
	peak = np.maximum.accumulate(equity)
	drawdown = np.where(peak != 0, (equity - peak) / peak, 0.0)
	equity_curve = [
		{
			"date": date,
			"equity": float(equity[i]),
			"drawdown": float(drawdown[i]),
		}
		for i, date in enumerate(dates)
	]
	price_series = [
		{"date": date, "price": float(price)}
		for date, price in zip(dates, prices)
	]

	trade_log = []
	cumulative_pnl = 0.0
	for entry in results.trade_log:
		idx = int(entry["index"])
		pnl = float(results.daily_returns[idx] * results.equity_curve[idx])
		cumulative_pnl += pnl
		trade_log.append(
			{
				"id": f"trade-{idx}",
				"date": str(entry["date"]),
				"action": entry["action"].upper(),
				"price": float(entry["price"]),
				"shares": float(entry["shares"]),
				"position": float(entry["position"]),
				"pnl": pnl,
				"cumulativePnl": cumulative_pnl,
			}
		)

	metrics = _build_backtest_metrics(
		actual_returns,
		predicted_returns,
		results.daily_returns,
		trade_log,
	)

	buy_hold_curve: List[Dict[str, float]] = []
	if prices.size:
		units = initial_capital / prices[0] if prices[0] != 0 else 0.0
		buy_hold_equity = np.asarray(prices) * units
		buy_hold_curve = [
			{"date": date, "equity": float(buy_hold_equity[i])}
			for i, date in enumerate(dates)
		]

	annotation_markers: List[Dict[str, object]] = []
	for log_entry in trade_log:
		shares = float(abs(log_entry["shares"]))
		if shares < trade_share_floor:
			continue
		marker_type = "sell" if log_entry["action"] == "SELL" else "buy"
		annotation_markers.append(
			{
				"date": log_entry["date"],
				"price": log_entry["price"],
				"type": marker_type,
				"shares": shares,
				"confidence": 1.0,
				"segment": "history",
				"scope": "backtest",
			}
		)

	return {
		"equityCurve": equity_curve,
		"priceSeries": price_series,
		"tradeLog": trade_log,
		"metrics": metrics,
		"annotations": annotation_markers,
		"buyHoldEquity": buy_hold_curve,
		"csv": _render_backtest_csv(dates, prices, actual_returns, predicted_returns, positions),
	}
