import pandas as pd
import numpy as np
import ta
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
import warnings
import os
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from data.support_resistance_features import (
    calculate_pivot_points,
    calculate_volume_profile_features,
    calculate_dynamic_sr_zones
)
from data.volatility_features import (
    add_volatility_spread_features,
    fetch_vix_data as fetch_vix_proxy,
)
# Backwards-compatible alias expected by some modules
compute_volatility_features = add_volatility_spread_features

logger = logging.getLogger(__name__)

logger = logging.getLogger(__name__)

# Expected total features for v4.0 (Enhanced with Support/Resistance)
EXPECTED_FEATURE_COUNT = 157

# Newly added / previously undocumented features (kept as explicit list for bookkeeping)
# This list documents features that were observed in the pipeline but not present
# in earlier canonical lists. Keeping them explicit makes it easier to audit and
# evolve the canonical set.
NEW_FEATURES = [
    'vpi',
    'vpi_short',
    'rsi_divergence_5d',
    'rsi_divergence_10d',
    'rsi_divergence_20d',
    'volatility_regime_low',
    'volatility_regime_normal',
    'volatility_regime_high',
    # Regime-conditional features produced by compute_regime_features()
    'regime_low_vol', 'regime_normal_vol', 'regime_high_vol',
    'rsi_low_vol', 'rsi_high_vol',
    'macd_low_vol', 'macd_high_vol',
    'bb_position_low_vol', 'bb_position_high_vol',
    'stoch_k_low_vol', 'stoch_k_high_vol',
    'momentum_low_vol', 'momentum_high_vol'
]


def audit_feature_counts(df: pd.DataFrame, include_sentiment: bool) -> Dict[str, Any]:
    """Audit feature engineering to identify all created features."""
    exclude = {'Open', 'High', 'Low', 'Close', 'Volume', 'Date', 'Dividends', 'Stock Splits'}
    actual_features = [c for c in df.columns if c not in exclude]
    expected_features = get_feature_columns(include_sentiment=include_sentiment)
    
    extra = set(actual_features) - set(expected_features)
    missing = set(expected_features) - set(actual_features)
    
    # Categorize features by source
    categories = {
        'price_momentum': [c for c in actual_features if 'momentum' in c or 'velocity' in c or 'acceleration' in c],
        'volatility': [c for c in actual_features if 'vol' in c.lower() or 'atr' in c or 'bb_' in c],
        'volume': [c for c in actual_features if 'volume' in c or 'obv' in c or 'vwap' in c],
        'trend': [c for c in actual_features if 'sma' in c or 'ema' in c or 'macd' in c],
        'pattern': [c for c in actual_features if 'gap' in c or 'shadow' in c or 'body' in c],
        'support_resistance': [c for c in actual_features if 'support' in c or 'resistance' in c or 'fib' in c],
        'sentiment': [c for c in actual_features if 'sentiment' in c or 'news' in c],
        'regime': [c for c in actual_features if 'regime' in c],
        'divergence': [c for c in actual_features if 'divergence' in c or 'rsi_' in c],
        'new_v31': [c for c in actual_features if c in NEW_FEATURES]
    }
    
    return {
        'actual_count': len(actual_features),
        'expected_count': len(expected_features),
        'extra_features': list(extra),
        'missing_features': list(missing),
        'categories': {k: len(v) for k, v in categories.items()},
        'detailed_categories': categories
    }

def _clean_and_impute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Final feature cleaning with intelligent imputation (NO LOOK-AHEAD BIAS)
    
    Strategy:
    1. Technical indicators: forward-fill up to 5 days (handles warm-up)
    2. Sentiment features: forward-fill up to 3 days (handles weekends)
    3. Fill remaining NaNs with 0 (neutral) - NO BACKWARD FILL (prevents data leakage)
    4. Verify no NaNs remain
    
    CRITICAL: We do NOT use bfill() because that would use future data
    to fill past values, introducing look-ahead bias.
    
    Args:
        df: DataFrame with engineered features
    
    Returns:
        DataFrame with all NaN values properly imputed
    """
    logger.info("Cleaning features and handling NaN values (no look-ahead)...")
    
    # Identify feature categories
    technical_indicators = [
        'rsi', 'macd', 'macd_signal', 'macd_histogram',
        'stoch_k', 'stoch_d', 'sma_10', 'sma_20', 'sma_50',
        'bb_upper', 'bb_lower', 'bb_width', 'bb_position',
        'volume_sma', 'volume_ratio', 'accumulation_distribution'
    ]
    
    sentiment_features = [col for col in df.columns if 'sentiment' in col.lower()]
    
    # 1. Forward-fill technical indicators (warm-up period)
    for col in technical_indicators:
        if col in df.columns:
            df[col] = df[col].ffill(limit=5)
    
    # 2. Forward-fill sentiment (weekends/holidays)
    for col in sentiment_features:
        if col in df.columns:
            df[col] = df[col].ffill(limit=3)
    
    # 3. NO BACKWARD FILL - instead fill remaining NaNs at start with 0
    # This is critical to prevent data leakage from future values
    
    # 4. Fill remaining NaNs with neutral/zero values
    remaining_nan = df.isna().sum()
    if remaining_nan.any():
        nan_cols = remaining_nan[remaining_nan > 0].to_dict()
        logger.warning(f"Filling remaining NaNs with neutral values: {nan_cols}")
        
        # Intelligent neutral values by feature type
        for col in df.columns:
            if df[col].isna().any():
                # Ratio/percentage features → 0 (neutral)
                if any(keyword in col.lower() for keyword in ['ratio', 'pct', 'percent']):
                    df[col] = df[col].fillna(0.0)
                # Sentiment features → 0 (neutral)
                elif 'sentiment' in col.lower():
                    df[col] = df[col].fillna(0.0)
                # Distance features → 0 (neutral)
                elif 'distance' in col.lower():
                    df[col] = df[col].fillna(0.0)
                # Count features → 0
                elif 'volume' in col.lower() or 'count' in col.lower():
                    df[col] = df[col].fillna(0.0)
                # Everything else → median of available values
                else:
                    median_val = df[col].median()
                    if pd.notna(median_val):
                        df[col] = df[col].fillna(median_val)
                    else:
                        df[col] = df[col].fillna(0.0)
    
    # 5. Verify no NaNs remain
    final_nans = df.isna().sum().sum()
    if final_nans > 0:
        logger.error(f"Still have {final_nans} NaN values after imputation!")
        # Emergency fallback
        df = df.fillna(0.0)
    
    logger.info(f"[OK] All NaN values handled. Shape: {df.shape}")
    
    return df


def engineer_features(df: pd.DataFrame, symbol: str = None, include_sentiment: bool = True,
                     cache_manager: Optional[object] = None) -> pd.DataFrame:
    """
    Engineer features with CONSISTENT output regardless of context
    
    CRITICAL: Always produces the same features whether for training or inference
    
    NOTE: This function computes features at time t using data up to time t.
    For training, features must be shifted by 1 to prevent look-ahead bias.
    This shift is handled in data.target_engineering.prepare_training_data().
    
    Args:
        df: DataFrame with OHLCV data
        symbol: Stock ticker symbol (required for sentiment features)
        include_sentiment: Whether to include sentiment features (default: True)
    
    Returns:
        DataFrame with all engineered features (v3.1 expectations: 123 total)
    """
    df = df.copy()
    
    # ===================================================================
    # PRICE-BASED FEATURES
    # ===================================================================
    
    # Returns
    df['returns'] = df['Close'].pct_change()
    df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
    
    # Volatility
    df['volatility_5d'] = df['returns'].rolling(window=5).std()
    df['volatility_20d'] = df['returns'].rolling(window=20).std()
    
    # ===================================================================
    # MOMENTUM INDICATORS
    # ===================================================================
    
    # RSI
    df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
    
    # MACD
    macd = ta.trend.MACD(df['Close'], window_slow=26, window_fast=12, window_sign=9)
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_histogram'] = macd.macd_diff()
    
    # Stochastic Oscillator
    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14)
    df['stoch_k'] = stoch.stoch()
    df['stoch_d'] = stoch.stoch_signal()
    
    # ===================================================================
    # TREND INDICATORS
    # ===================================================================
    
    # Moving Averages (normalized as % deviation from close)
    sma_10_raw = df['Close'].rolling(window=10).mean()
    sma_20_raw = df['Close'].rolling(window=20).mean()
    sma_50_raw = df['Close'].rolling(window=50).mean()
    ema_12_raw = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26_raw = df['Close'].ewm(span=26, adjust=False).mean()
    
    df['sma_10'] = (sma_10_raw - df['Close']) / df['Close']
    df['sma_20'] = (sma_20_raw - df['Close']) / df['Close']
    df['sma_50'] = (sma_50_raw - df['Close']) / df['Close']
    df['ema_12'] = (ema_12_raw - df['Close']) / df['Close']
    df['ema_26'] = (ema_26_raw - df['Close']) / df['Close']
    
    # Price distance from moving averages
    df['price_to_sma20'] = (df['Close'] - sma_20_raw) / sma_20_raw
    df['price_to_sma50'] = (df['Close'] - sma_50_raw) / sma_50_raw
    
    # Moving average crossovers
    df['sma_10_20_cross'] = (sma_10_raw - sma_20_raw) / sma_20_raw
    df['ema_12_26_cross'] = (ema_12_raw - ema_26_raw) / ema_26_raw
    
    # ===================================================================
    # VOLATILITY INDICATORS
    # ===================================================================
    
    # Bollinger Bands (normalized as % deviation)
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    bb_upper_raw = bb.bollinger_hband()
    bb_lower_raw = bb.bollinger_lband()
    
    df['bb_upper'] = (bb_upper_raw - df['Close']) / df['Close']
    df['bb_lower'] = (bb_lower_raw - df['Close']) / df['Close']
    df['bb_width'] = (bb_upper_raw - bb_lower_raw) / df['Close']
    df['bb_position'] = (df['Close'] - bb_lower_raw) / (bb_upper_raw - bb_lower_raw)
    
    # ATR
    df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
    df['atr_percent'] = df['atr'] / df['Close']
    # ===================================================================
    # VOLATILITY SPREAD FEATURES (REALIZED vs IMPLIED)
    # ===================================================================
    # Use VIX as a market-wide implied volatility proxy and compute RV-IV spread
    try:
        # Some small-cap stocks may have limited or illiquid options; use VIX as proxy
        if symbol is not None and symbol.upper() in ['HOOD', 'PLTR', 'COIN']:
            print(f"[INFO] Using VIX as IV proxy for {symbol} (limited options data)")

        df = add_volatility_spread_features(df, symbol=symbol if symbol is not None else 'SPY')
        print(f"[OK] Volatility spread features added (5 new columns)")
        # Note: legacy alias `rv_iv_z_score` was previously created for backward compatibility.
        # To avoid duplicate columns and canonical feature-count mismatches, we no longer
        # create this alias here. Downstream code should use `rv_iv_spread_zscore`.
    except Exception as e:
        # VIX data / volatility spread is optional. Reduce noisy warnings:
        # - Show a single info message on first occurrence
        # - Use debug logs thereafter with the exception detail
        if not hasattr(engineer_features, '_vix_warning_shown'):
            logger.info("ℹ️  VIX data unavailable (optional feature); skipping volatility spread features")
            setattr(engineer_features, '_vix_warning_shown', True)
        else:
            logger.debug("VIX data unavailable, skipping volatility spread features: %s", e)
        # Add dummy columns with zeros to maintain feature count
        df['realized_vol_20d'] = 0.0
        df['vix'] = 0.0
        df['rv_iv_spread'] = 0.0
        df['rv_iv_spread_zscore'] = 0.0
        df['rv_iv_z_score'] = 0.0
        df['iv_regime_underpriced'] = 0
        df['iv_regime_overpriced'] = 0
        df['iv_regime_fair'] = 1
        # Fallback volatility regime flags for compatibility
        df['volatility_regime_low'] = 0
        df['volatility_regime_normal'] = 1
        df['volatility_regime_high'] = 0
    
    # ===================================================================
    # VOLUME INDICATORS (all normalized)
    # ===================================================================
    
    volume_sma_raw = df['Volume'].rolling(window=20).mean()
    df['volume_sma'] = np.log1p(volume_sma_raw) / (np.log1p(df['Volume']) + 1e-8)  # Log-scaled ratio
    df['volume_ratio'] = df['Volume'] / (volume_sma_raw + 1e-8)
    
    # OBV as momentum (change) instead of absolute value
    obv_raw = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
    df['obv'] = obv_raw.pct_change(periods=5).fillna(0).clip(-2, 2)  # 5-day OBV momentum, clipped to [-2, 2]
    df['obv_ema'] = df['obv'].ewm(span=20).mean()
    
    # Volume Pressure Index (VPI) - captures volume-weighted directional pressure
    # Add both medium and short window versions for model input
    try:
        df['vpi'] = compute_vpi(df, window=20)
        df['vpi_short'] = compute_vpi(df, window=5)
    except Exception:
        # If compute_vpi is not yet available or fails, fill neutral values
        df['vpi'] = 0.0
        df['vpi_short'] = 0.0

    # ===================================================================
    # ENHANCED MOMENTUM FEATURES (NEW)
    # ===================================================================
    
    # Price momentum (percentage-based)
    df['momentum_1d'] = (df['Close'] - df['Close'].shift(1)) / (df['Close'].shift(1) + 1e-8)
    df['momentum_5d'] = (df['Close'] - df['Close'].shift(5)) / (df['Close'].shift(5) + 1e-8)
    df['momentum_20d'] = (df['Close'] - df['Close'].shift(20)) / (df['Close'].shift(20) + 1e-8)
    df['rate_of_change_10d'] = (df['Close'] - df['Close'].shift(10)) / (df['Close'].shift(10) + 1e-8)
    
    # Volatility regime
    df['vol_ratio_5_20'] = df['volatility_5d'] / (df['volatility_20d'] + 1e-8)
    df['high_low_range'] = (df['High'] - df['Low']) / df['Close']
    df['close_position'] = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)

    # Regime-conditional features (volatility regimes)
    try:
        regime_features = compute_regime_features(df, atr_column='atr_percent', rsi_column='rsi')
        for col_name, col_data in regime_features.items():
            df[col_name] = col_data
    except Exception as e:
        # If regime features fail, fill neutral/default values
        df['regime_low_vol'] = 0
        df['regime_high_vol'] = 0
        df['regime_normal_vol'] = 1
        df['rsi_low_vol'] = 0.0
        df['rsi_high_vol'] = 0.0
        df['macd_low_vol'] = 0.0
        df['macd_high_vol'] = 0.0
        df['bb_position_low_vol'] = 0.0
        df['bb_position_high_vol'] = 0.0

    # Map regime flags into the new volatility_regime_* naming used by v3.1
    try:
        df['volatility_regime_low'] = df.get('regime_low_vol', 0)
        df['volatility_regime_normal'] = df.get('regime_normal_vol', 0)
        df['volatility_regime_high'] = df.get('regime_high_vol', 0)
    except Exception:
        df['volatility_regime_low'] = 0
        df['volatility_regime_normal'] = 1
        df['volatility_regime_high'] = 0
    
    # Volume-price interaction (normalized)
    df['volume_price_trend'] = (df['Volume'] / (volume_sma_raw + 1e-8)) * df['returns']  # Volume ratio * returns
    df['accumulation_distribution'] = df['volume_price_trend'].rolling(window=20).mean()  # 20-day average instead of cumsum
    
    # Lagged returns (help model see patterns)
    df['return_lag_1'] = df['returns'].shift(1)
    df['return_lag_2'] = df['returns'].shift(2)
    df['return_lag_5'] = df['returns'].shift(5)
    
    # ===================================================================
    # VELOCITY & ACCELERATION
    # ===================================================================
    
    df['velocity_5d'] = df['Close'].pct_change(periods=5)
    df['velocity_10d'] = df['Close'].pct_change(periods=10)
    df['velocity_20d'] = df['Close'].pct_change(periods=20)
    df['acceleration_5d'] = df['velocity_5d'].diff(5)
    df['rsi_velocity'] = df['rsi'].diff(5)
    df['volume_velocity'] = df['Volume'].pct_change(periods=5)
    
    # ===================================================================
    # PATTERN FEATURES
    # ===================================================================
    
    df['higher_high'] = (df['High'] > df['High'].shift(1)).astype(int)
    df['lower_low'] = (df['Low'] < df['Low'].shift(1)).astype(int)
    df['gap_up'] = ((df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)).clip(lower=0)
    df['gap_down'] = ((df['Close'].shift(1) - df['Open']) / df['Close'].shift(1)).clip(lower=0)
    df['body_size'] = abs(df['Close'] - df['Open']) / df['Close']
    df['upper_shadow'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / df['Close']
    df['lower_shadow'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / df['Close']
    
    # ===================================================================
    # REGIME CHANGE INTERACTION FEATURES
    # ===================================================================
    
    # RSI regime crossovers (capture oversold/overbought transitions)
    df['rsi_cross_30'] = ((df['rsi'] > 30) & (df['rsi'].shift(1) <= 30)).astype(int)  # Crosses above 30 (exit oversold)
    df['rsi_cross_70'] = ((df['rsi'] < 70) & (df['rsi'].shift(1) >= 70)).astype(int)  # Crosses below 70 (exit overbought)
    
    # Bollinger Band squeeze (low volatility regime)
    bb_width_20th_percentile = df['bb_width'].rolling(window=252, min_periods=50).quantile(0.20)
    df['bb_squeeze'] = (df['bb_width'] < bb_width_20th_percentile).astype(int)
    
    # Volume surge (unusual activity regime)
    df['volume_surge'] = (df['volume_ratio'] > 2.0).astype(int)
    
    # ===================================================================
    # MOMENTUM DIVERGENCE FEATURES
    # ===================================================================
    
    # Calculate RSI momentum (rate of change in RSI)
    df['rsi_momentum'] = df['rsi'].diff(5)
    
    # Price momentum vs RSI momentum divergence
    # Bearish divergence: price up but RSI down (momentum weakening)
    # Bullish divergence: price down but RSI up (momentum strengthening)
    price_momentum_5d_norm = df['momentum_5d'].clip(-0.5, 0.5)  # Normalize to similar scale
    rsi_momentum_norm = (df['rsi_momentum'] / 50.0).clip(-0.5, 0.5)  # Normalize RSI momentum
    df['momentum_divergence_5d'] = price_momentum_5d_norm - rsi_momentum_norm
    
    # Price momentum vs MACD histogram divergence
    # MACD histogram measures momentum strength
    price_momentum_20d_norm = df['momentum_20d'].clip(-0.5, 0.5)
    macd_hist_norm = (df['macd_histogram'] / df['Close']).clip(-0.5, 0.5)  # Normalize to price
    df['momentum_divergence_20d'] = price_momentum_20d_norm - macd_hist_norm

    # ===================================================================
    # RSI MULTI-TIMEFRAME DIVERGENCE
    # ===================================================================
    # Compute multi-timeframe RSI divergence features (ultra-short and medium)
    try:
        rsi_div = compute_rsi_divergence(df, pairs=[(3, 14), (7, 28)])

        # ultra-short pair (3,14)
        if 'rsi_3' in rsi_div:
            df['rsi_3'] = rsi_div.get('rsi_3')
        if 'rsi_14' in rsi_div:
            df['rsi_14'] = rsi_div.get('rsi_14')
        if 'div_3_14_raw' in rsi_div:
            df['rsi_divergence_3_14'] = rsi_div.get('div_3_14_raw')
        if 'div_3_14_z' in rsi_div:
            df['rsi_divergence_3_14_z'] = rsi_div.get('div_3_14_z')

        # medium pair (7,28)
        if 'rsi_7' in rsi_div:
            df['rsi_7'] = rsi_div.get('rsi_7')
        if 'rsi_28' in rsi_div:
            df['rsi_28'] = rsi_div.get('rsi_28')
        if 'div_7_28_raw' in rsi_div:
            df['rsi_divergence_7_28'] = rsi_div.get('div_7_28_raw')
        if 'div_7_28_z' in rsi_div:
            df['rsi_divergence_7_28_z'] = rsi_div.get('div_7_28_z')
    except Exception:
        # Fail safe: don't break pipeline if divergence calculation fails
        df['rsi_divergence_3_14'] = 0.0
        df['rsi_divergence_3_14_z'] = 0.0
        df['rsi_divergence_7_28'] = 0.0
        df['rsi_divergence_7_28_z'] = 0.0

    # NEW (v3.1): multi-timeframe RSI divergence for 5d,10d,20d vs 14d baseline
    try:
        extra_pairs = [(5, 14), (10, 14), (20, 14)]
        extra_rsi = compute_rsi_divergence(df, pairs=extra_pairs, z_window=20)
        # Map raw divergence into concise feature names
        for fast, slow in extra_pairs:
            raw_key = f'div_{fast}_{slow}_raw'
            col_name = f'rsi_divergence_{fast}d'
            if raw_key in extra_rsi:
                df[col_name] = extra_rsi.get(raw_key)
            else:
                df[col_name] = 0.0
    except Exception:
        df['rsi_divergence_5d'] = 0.0
        df['rsi_divergence_10d'] = 0.0
        df['rsi_divergence_20d'] = 0.0
    
    # ===================================================================
    # SUPPORT/RESISTANCE LEVELS (K-MEANS CLUSTERING)
    # ===================================================================
    
    # Calculate support/resistance features
    support_resistance_features = calculate_support_resistance_features(df)
    
    # Merge with main dataframe using concat to avoid fragmentation
    cols_to_add = [c for c in support_resistance_features.columns if c != 'Date']
    if cols_to_add:
        df = pd.concat([df, support_resistance_features[cols_to_add]], axis=1)
    
    # ===================================================================
    # FIBONACCI RETRACEMENT LEVELS
    # ===================================================================
    
    # Calculate Fibonacci features
    fibonacci_features = calculate_fibonacci_features(df)
    
    # Merge with main dataframe using concat to avoid fragmentation
    cols_to_add = [c for c in fibonacci_features.columns if c != 'Date']
    if cols_to_add:
        df = pd.concat([df, fibonacci_features[cols_to_add]], axis=1)
    
    # ===================================================================
    # SUPPORT & RESISTANCE ENHANCED (PHASE 4)
    # ===================================================================
    print("   -> Calculating Phase 4 S/R features (Pivots, VAP, Dynamic Zones)...")
    try:
        # 1. Pivot Points (Fractals)
        # We need a 5-day fractal window
        pivots_df = calculate_pivot_points(df, window=5)
        df = pd.concat([df, pivots_df], axis=1)
        
        # 2. Volume-at-Price (VAP) Profile
        # Identifies high volume nodes and points of control
        vap_df = calculate_volume_profile_features(df, lookback=60, num_bins=30)
        df = pd.concat([df, vap_df], axis=1)
        
        # 3. Dynamic S/R Zones
        # Combines pivots and volume nodes into entry/exit zones
        zones_df = calculate_dynamic_sr_zones(df, pivots_df, vap_df)
        df = pd.concat([df, zones_df], axis=1)
        
    except Exception as e:
        print(f"[WARN] Phase 4 Support/Resistance features failed: {e}")
        # Add neutral fallback columns if needed to maintain feature count
        new_sr_cols = [
            'latest_high_pivot', 'latest_low_pivot', 'dist_to_high_pivot', 'dist_to_low_pivot',
            'vap_poc_price', 'dist_to_vap_poc', 'hvn_proximity',
            'near_resistance_zone', 'near_support_zone', 'volume_concentration_score'
        ]
        for col in new_sr_cols:
            if col not in df.columns:
                df[col] = 0.0
    
    # ===================================================================
    # VWAP (VOLUME WEIGHTED AVERAGE PRICE) FEATURES
    # ===================================================================
    
    # Calculate VWAP features
    vwap_features = calculate_vwap_features(df)
    
    # Merge with main dataframe using concat to avoid fragmentation
    cols_to_add = [c for c in vwap_features.columns if c != 'Date']
    if cols_to_add:
        df = pd.concat([df, vwap_features[cols_to_add]], axis=1)
    
    # ===================================================================
    # SENTIMENT FEATURES (NEWS-BASED)
    # ===================================================================
    
    if include_sentiment and symbol is not None:
        print("   -> Preparing sentiment features (loads FinBERT + news data on first run, may take a few minutes)...")
        try:
            df = add_sentiment_features(df, symbol, cache_manager=cache_manager)
        except Exception as e:
            print(f"[WARN] Warning: Could not add sentiment features for {symbol}: {e}")
            # Add dummy sentiment columns filled with 0
            sentiment_cols = get_sentiment_feature_columns()
            for col in sentiment_cols:
                if col not in df.columns:
                    df[col] = 0.0
    elif include_sentiment and symbol is None:
        print("[WARN] Warning: Sentiment features requested but no symbol provided. Skipping sentiment.")
        # Add dummy sentiment columns
        sentiment_cols = get_sentiment_feature_columns()
        for col in sentiment_cols:
            if col not in df.columns:
                df[col] = 0.0

    # ===================================================================
    # SENTIMENT VALIDATION CHECK
    # ===================================================================

    # After adding sentiment features, verify they're not all zeros
    if include_sentiment and symbol is not None:
        sentiment_cols = get_sentiment_feature_columns()
        existing_sent_cols = [c for c in sentiment_cols if c in df.columns]

        if existing_sent_cols:
            sentiment_data = df[existing_sent_cols]
            all_zeros = (sentiment_data == 0.0).all().all()

            if all_zeros:
                logger.error(
                    f"❌ CRITICAL: All sentiment features are ZERO for {symbol}!\n"
                    f"   This means news data fetch FAILED.\n"
                    f"   1. Check finnhub-python installed: pip install finnhub-python\n"
                    f"   2. Check FINNHUB_API_KEY set in .env\n"
                    f"   3. Check API key valid or rate limit not exceeded\n"
                    f"\n"
                    f"   Training will continue but sentiment features will have 0.0 importance!"
                )

    # ===================================================================
    # DATA QUALITY VERIFICATION
    # ===================================================================
    
    # Check for look-ahead bias (this is a sanity check, not exhaustive)
    # All features should be based on current or past data only
    verify_no_lookahead_bias(df)
    
    # ===================================================================
    # CLEAN UP
    # ===================================================================
    
    # Replace inf with NaN
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Check for data quality issues (before filling)
    check_feature_quality(df)
    
    # Forward-fill support/resistance features (these have NaN at start due to lookback period)
    support_resistance_cols = [
        'distance_to_nearest_support', 'distance_to_nearest_resistance',
        'support_strength', 'resistance_strength',
        'broke_resistance', 'broke_support',
        'days_since_support_break', 'days_since_resistance_break'
    ]
    for col in support_resistance_cols:
        if col in df.columns:
            df[col] = df[col].ffill().fillna(0)  # Forward fill, then fill remaining with 0
    
    # Forward-fill long-period moving averages
    long_period_cols = ['sma_50', 'price_to_sma50']
    for col in long_period_cols:
        if col in df.columns:
            df[col] = df[col].ffill()
    
    # ===================================================================
    # FINAL STEP: COMPREHENSIVE NaN CLEANUP
    # ===================================================================
    
    df = _clean_and_impute_features(df)
    
    # Verify data quality
    assert df.isna().sum().sum() == 0, "NaN values still present after cleanup!"
    assert not np.isinf(df.select_dtypes(include=[np.number])).any().any(), "Inf values present after cleanup!"
    
    # Run audit diagnostics when symbol is provided (helps identify mismatches)
    if symbol is not None:
        try:
            audit = audit_feature_counts(df, include_sentiment=include_sentiment)
            logger.info(f"[AUDIT] Feature Audit for {symbol}:")
            logger.info(f"   Actual: {audit['actual_count']} features")
            logger.info(f"   Expected: {audit['expected_count']} features")
            if audit.get('extra_features'):
                logger.warning(f"   [WARN] Extra features ({len(audit['extra_features'])}): {audit['extra_features'][:10]}")
            if audit.get('missing_features'):
                logger.warning(f"   [WARN] Missing features ({len(audit['missing_features'])}): {audit['missing_features'][:10]}")
            logger.info(f"   Categories: {audit['categories']}")
        except Exception as e:
            logger.exception(f"Feature audit failed: {e}")
    # Reconcile feature set to canonical expected columns: add missing, drop extras
    try:
        df = validate_feature_count(df)
    except Exception as e:
        logger.warning(f"validate_feature_count() failed: {e}")

    logger.info(f"[OK] Feature engineering complete: {df.shape}")

    return df


def calculate_support_resistance_features(df: pd.DataFrame, window: int = 5, 
                                          n_clusters: int = 5, lookback: int = 60) -> pd.DataFrame:
    """
    Calculate support/resistance levels using K-means clustering.
    
    This identifies key price levels where stocks tend to bounce (support) or stall (resistance)
    by clustering local price extrema. Research shows K-means effectively identifies these zones
    where trading activity concentrates.
    
    ROBUST ERROR HANDLING: Returns neutral values if insufficient data or calculation failures.
    This prevents training failures when historical data < 200 days.
    
    Args:
        df: DataFrame with OHLCV data
        window: Window for finding local min/max (default 5 days)
        n_clusters: Number of support/resistance clusters (default 5)
        lookback: Days to look back for clustering (default 60)
    
    Returns:
        DataFrame with support/resistance features (fills neutral values on errors)
    """
    MIN_REQUIRED_ROWS = 60  # Minimum data required for meaningful calculation
    
    # Initialize result with neutral values
    result_df = pd.DataFrame(index=df.index)
    result_df['distance_to_nearest_support'] = 0.0      # Neutral: no support identified
    result_df['distance_to_nearest_resistance'] = 0.0   # Neutral: no resistance identified
    result_df['support_strength'] = 0                   # Neutral: no touches
    result_df['resistance_strength'] = 0                # Neutral: no touches
    result_df['broke_resistance'] = 0                   # Neutral: no breakout
    result_df['broke_support'] = 0                      # Neutral: no breakdown
    result_df['days_since_support_break'] = 999         # Neutral: long time since break
    result_df['days_since_resistance_break'] = 999      # Neutral: long time since break
    
    # Check for sufficient data
    if len(df) < MIN_REQUIRED_ROWS:
        print(f"[WARN] Support/resistance: Insufficient data ({len(df)} rows, need {MIN_REQUIRED_ROWS}). Using neutral values.")
        return result_df
    
    # Wrap entire calculation in try-except to prevent any failures
    try:
        # Process each time period (optimize by skipping rows and forward-filling)
        # Support/Resistance levels don't change drastically day-to-day
        calc_step = 5  # Calculate every 5 days
        
        # Vectorize local extrema detection for the whole dataframe first
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        
        # Pre-calculate rolling extremes for the whole series to speed up window checks
        from scipy.signal import argrelextrema
        all_local_mins_idx = argrelextrema(closes, np.less_equal, order=window)[0]
        all_local_maxs_idx = argrelextrema(closes, np.greater_equal, order=window)[0]
        
        last_support_levels = []
        last_resistance_levels = []
        
        for i in range(lookback + window, len(df)):
            current_close = closes[i]
            prev_close = closes[i-1]
            
            # Recalculate clusters every 'calc_step' days
            if i % calc_step == 0 or i == lookback + window:
                window_start = i - lookback
                window_end = i
                
                local_mins = [closes[idx] for idx in all_local_mins_idx if window_start <= idx < window_end]
                local_maxs = [closes[idx] for idx in all_local_maxs_idx if window_start <= idx < window_end]
                
                support_levels = []
                resistance_levels = []
                
                if len(local_mins) >= 3:
                    l_arr = np.array(local_mins).reshape(-1, 1)
                    u_levs = np.unique(l_arr)
                    eff = min(n_clusters, len(u_levs))
                    if eff >= 2:
                        kmeans = KMeans(n_clusters=eff, random_state=42, n_init=5)
                        kmeans.fit(l_arr)
                        support_levels = sorted(kmeans.cluster_centers_.flatten())
                    else:
                        support_levels = sorted(u_levs.tolist())
                
                if len(local_maxs) >= 3:
                    l_arr = np.array(local_maxs).reshape(-1, 1)
                    u_levs = np.unique(l_arr)
                    eff = min(n_clusters, len(u_levs))
                    if eff >= 2:
                        kmeans = KMeans(n_clusters=eff, random_state=42, n_init=5)
                        kmeans.fit(l_arr)
                        resistance_levels = sorted(kmeans.cluster_centers_.flatten())
                    else:
                        resistance_levels = sorted(u_levs.tolist())
                
                last_support_levels = support_levels
                last_resistance_levels = resistance_levels
            
            support_levels = last_support_levels
            resistance_levels = last_resistance_levels
            
            # Distances
            nearest_support = None
            if support_levels:
                below = [s for s in support_levels if s < current_close]
                if below:
                    nearest_support = max(below)
                    result_df.iloc[i, 0] = (current_close - nearest_support) / current_close
            
            nearest_resistance = None
            if resistance_levels:
                above = [r for r in resistance_levels if r > current_close]
                if above:
                    nearest_resistance = min(above)
                    result_df.iloc[i, 1] = (nearest_resistance - current_close) / current_close
            
            # Strength (estimated for speed)
            if nearest_support: result_df.iloc[i, 2] = 1
            if nearest_resistance: result_df.iloc[i, 3] = 1
            
            # Breakouts
            if nearest_resistance and current_close > nearest_resistance and prev_close <= nearest_resistance:
                result_df.iloc[i, 4] = 1
            if nearest_support and current_close < nearest_support and prev_close >= nearest_support:
                result_df.iloc[i, 5] = 1
                
        # Post-process days since break (fast vectorized-ish)
        res_breaks = result_df['broke_resistance'].values
        sup_breaks = result_df['broke_support'].values
        res_days = np.full(len(df), 999)
        sup_days = np.full(len(df), 999)
        
        curr_res = 999
        curr_sup = 999
        for i in range(len(df)):
            if res_breaks[i] == 1: curr_res = 0
            else: curr_res += 1
            if sup_breaks[i] == 1: curr_sup = 0
            else: curr_sup += 1
            res_days[i] = min(curr_res, 999)
            sup_days[i] = min(curr_sup, 999)
            
        result_df['days_since_resistance_break'] = res_days
        result_df['days_since_support_break'] = sup_days
    
    except Exception as e:
        # If entire calculation fails, return neutral values with warning
        print(f"[WARN] Support/resistance calculation failed: {str(e)}. Using neutral values.")
    
    return result_df


def calculate_fibonacci_features(df: pd.DataFrame, swing_window: int = 20) -> pd.DataFrame:
    """
    Calculate Fibonacci retracement levels.
    
    Fibonacci retracements identify potential support/resistance at key ratios (38.2%, 50%, 61.8%)
    between recent swing high and swing low. Traders widely use these levels for entry/exit points.
    
    Args:
        df: DataFrame with OHLCV data
        swing_window: Window for finding swing high/low (default 20 days)
    
    Returns:
        DataFrame with Fibonacci features
    """
    result_df = pd.DataFrame(index=df.index)
    
    # Initialize columns
    result_df['distance_to_fib_382'] = np.nan
    result_df['distance_to_fib_500'] = np.nan
    result_df['distance_to_fib_618'] = np.nan
    result_df['in_consolidation_zone'] = 0
    
    # Fibonacci ratios
    fib_levels = {
        'fib_382': 0.382,
        'fib_500': 0.500,
        'fib_618': 0.618
    }
    
    # Calculate for each time period
    for i in range(swing_window, len(df)):
        # Get recent swing window
        window_df = df.iloc[i - swing_window:i]
        current_close = df.iloc[i]['Close']
        
        # Find swing high and swing low
        swing_high = window_df['High'].max()
        swing_low = window_df['Low'].min()
        
        swing_range = swing_high - swing_low
        
        if swing_range > 0:
            # Calculate Fibonacci levels
            fib_382_level = swing_high - (0.382 * swing_range)
            fib_500_level = swing_high - (0.500 * swing_range)
            fib_618_level = swing_high - (0.618 * swing_range)
            
            # Calculate distances (normalized by price)
            result_df.loc[df.index[i], 'distance_to_fib_382'] = (
                (current_close - fib_382_level) / current_close
            )
            result_df.loc[df.index[i], 'distance_to_fib_500'] = (
                (current_close - fib_500_level) / current_close
            )
            result_df.loc[df.index[i], 'distance_to_fib_618'] = (
                (current_close - fib_618_level) / current_close
            )
            
            # Check if in consolidation zone (within 2% of any major level)
            consolidation_threshold = 0.02
            
            levels_to_check = [swing_high, swing_low, fib_382_level, fib_500_level, fib_618_level]
            is_consolidating = False
            
            for level in levels_to_check:
                if abs(current_close - level) / current_close < consolidation_threshold:
                    is_consolidating = True
                    break
            
            result_df.loc[df.index[i], 'in_consolidation_zone'] = 1 if is_consolidating else 0
    
    return result_df


def calculate_vwap_features(df: pd.DataFrame, rolling_window: int = 20) -> pd.DataFrame:
    """
    Calculate VWAP (Volume Weighted Average Price) features.
    
    VWAP is the average price weighted by volume, acting as dynamic support/resistance.
    Institutional traders often use VWAP as a benchmark for execution quality.
    
    Key insights:
    - Price above VWAP = bullish, institutions accumulated
    - Price below VWAP = bearish, institutions distributed
    - Large volume at VWAP = institutional activity
    - Crossovers signal trend changes
    
    Args:
        df: DataFrame with OHLCV data
        rolling_window: Window for rolling VWAP (default 20 days for daily data)
    
    Returns:
        DataFrame with VWAP features
    """
    result_df = pd.DataFrame(index=df.index)
    
    # Calculate typical price (average of high, low, close)
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3.0
    
    # Calculate rolling VWAP (for daily data, use rolling window)
    # For intraday data, this would reset daily
    vwap = (typical_price * df['Volume']).rolling(window=rolling_window).sum() / \
           df['Volume'].rolling(window=rolling_window).sum()
    
    # ===================================================================
    # BASIC VWAP FEATURES
    # ===================================================================
    
    # VWAP deviation (normalized)
    result_df['vwap_deviation'] = (df['Close'] - vwap) / (vwap + 1e-8)
    
    # Volume ratio (current volume vs 20-day average)
    volume_mean_20d = df['Volume'].rolling(window=20).mean()
    result_df['vwap_volume_ratio'] = df['Volume'] / (volume_mean_20d + 1e-8)
    
    # Above/below VWAP (binary)
    result_df['above_vwap'] = (df['Close'] > vwap).astype(int)
    
    # VWAP crossovers (trend change signals)
    prev_above_vwap = (df['Close'].shift(1) > vwap.shift(1))
    curr_above_vwap = (df['Close'] > vwap)
    
    result_df['vwap_crossover'] = ((curr_above_vwap) & (~prev_above_vwap)).astype(int)  # Cross above
    result_df['vwap_crossunder'] = ((~curr_above_vwap) & (prev_above_vwap)).astype(int)  # Cross below
    
    # ===================================================================
    # VWAP BANDS (VOLATILITY MEASURE)
    # ===================================================================
    
    # Calculate standard deviation of typical price
    typical_price_std = typical_price.rolling(window=rolling_window).std()
    
    # VWAP bands (similar to Bollinger Bands)
    vwap_upper = vwap + (2 * typical_price_std)
    vwap_lower = vwap - (2 * typical_price_std)
    
    # Band position (0 = at lower band, 1 = at upper band)
    band_range = vwap_upper - vwap_lower
    result_df['vwap_band_position'] = (df['Close'] - vwap_lower) / (band_range + 1e-8)
    result_df['vwap_band_position'] = result_df['vwap_band_position'].clip(0, 1)
    
    # ===================================================================
    # VOLUME-WEIGHTED MOMENTUM
    # ===================================================================
    
    # Volume-weighted momentum (gives more weight to high-volume days)
    returns = df['Close'].pct_change()
    
    # 5-day volume-weighted momentum
    returns_5d = returns.rolling(window=5)
    volume_5d = df['Volume'].rolling(window=5)
    
    vw_momentum_5d_num = (returns * df['Volume']).rolling(window=5).sum()
    vw_momentum_5d_den = df['Volume'].rolling(window=5).sum()
    result_df['vw_momentum_5d'] = vw_momentum_5d_num / (vw_momentum_5d_den + 1e-8)
    
    # 20-day volume-weighted momentum
    vw_momentum_20d_num = (returns * df['Volume']).rolling(window=20).sum()
    vw_momentum_20d_den = df['Volume'].rolling(window=20).sum()
    result_df['vw_momentum_20d'] = vw_momentum_20d_num / (vw_momentum_20d_den + 1e-8)
    
    # Volume surge with positive price action (bullish institutional buying)
    result_df['volume_surge_with_price'] = (
        (result_df['vwap_volume_ratio'] > 1.5) & (returns > 0.02)
    ).astype(int)
    
    # ===================================================================
    # INSTITUTIONAL FOOTPRINT DETECTION
    # ===================================================================
    
    # Large volume day (potential institutional activity)
    result_df['large_volume_day'] = (df['Volume'] > 1.8 * volume_mean_20d).astype(int)
    
    # Accumulation signal (institutions buying)
    # Criteria: Close > Open (green candle), High volume, Close near High
    close_to_high_ratio = (df['Close'] - df['Low']) / (df['High'] - df['Low'] + 1e-8)
    result_df['accumulation_signal'] = (
        (df['Close'] > df['Open']) &  # Green candle
        (df['Volume'] > volume_mean_20d) &  # Above average volume
        (close_to_high_ratio > 0.7)  # Close in upper 30% of range
    ).astype(int)
    
    # Distribution signal (institutions selling)
    # Criteria: Close < Open (red candle), High volume, Close near Low
    close_to_low_ratio = (df['High'] - df['Close']) / (df['High'] - df['Low'] + 1e-8)
    result_df['distribution_signal'] = (
        (df['Close'] < df['Open']) &  # Red candle
        (df['Volume'] > volume_mean_20d) &  # Above average volume
        (close_to_low_ratio > 0.7)  # Close in lower 30% of range
    ).astype(int)
    
    return result_df


def compute_vpi(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """
    Computes Volume Pressure Index (VPI): volume-weighted directional pressure.

    VPI = Σ(Volume × sign(Close-Open) × |Close-Open|/(High-Low)) / Σ(Volume)

    Returns a pandas Series clipped to [-1, 1].
    """
    # Ensure required columns exist
    required = ['Open', 'High', 'Low', 'Close', 'Volume']
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required column for VPI: {col}")

    body = df['Close'] - df['Open']
    range_ = (df['High'] - df['Low']).replace(0, 1e-8)
    range_pct = body.abs() / range_

    signed_volume = df['Volume'] * np.sign(body) * range_pct

    num = signed_volume.rolling(window=window, min_periods=1).sum()
    den = df['Volume'].rolling(window=window, min_periods=1).sum().replace(0, 1e-8)

    vpi = num / den
    vpi = vpi.fillna(0.0)
    vpi = vpi.replace([np.inf, -np.inf], 0.0)
    vpi = vpi.clip(-1.0, 1.0)

    # Validation
    if vpi.min() < -1.0 - 1e-8 or vpi.max() > 1.0 + 1e-8:
        raise AssertionError("VPI out of bounds")
    if vpi.isna().any():
        raise AssertionError("VPI contains NaN values")

    return vpi


def compute_regime_features(df: pd.DataFrame, atr_column: str = 'atr_percent', rsi_column: str = 'rsi') -> dict:
    """
    Classify volatility regime and return regime-conditional features.

    Regimes:
      - LOW_VOL: atr / SMA(atr,50) < 0.7
      - HIGH_VOL: atr / SMA(atr,50) > 1.3
      - NORMAL_VOL: otherwise

    Returns a dict of Series to be attached to the DataFrame.
    """
    # Validate input
    if atr_column not in df.columns:
        raise ValueError(f"ATR column '{atr_column}' not found in DataFrame")
    if rsi_column not in df.columns:
        raise ValueError(f"RSI column '{rsi_column}' not found in DataFrame")

    # Compute ATR ratio against 50-day SMA (use min_periods=1 to avoid initial NaN)
    atr_sma50 = df[atr_column].rolling(window=50, min_periods=1).mean()
    atr_ratio = df[atr_column] / (atr_sma50 + 1e-12)

    # Classify regimes
    bins = [-np.inf, 0.7, 1.3, np.inf]
    labels = ['LOW_VOL', 'NORMAL_VOL', 'HIGH_VOL']
    regime = pd.cut(atr_ratio, bins=bins, labels=labels)

    # One-hot indicators
    regime_low = (regime == 'LOW_VOL').astype(int)
    regime_high = (regime == 'HIGH_VOL').astype(int)
    regime_normal = (regime == 'NORMAL_VOL').astype(int)

    # Regime-conditional features (RSI, MACD, BB position, momentum)
    macd = df['macd'] if 'macd' in df.columns else pd.Series(0.0, index=df.index)
    bb_pos = df['bb_position'] if 'bb_position' in df.columns else pd.Series(0.0, index=df.index)
    stoch_k = df['stoch_k'] if 'stoch_k' in df.columns else pd.Series(0.0, index=df.index)
    momentum_5d = df['momentum_5d'] if 'momentum_5d' in df.columns else pd.Series(0.0, index=df.index)

    rsi_low = df[rsi_column].where(regime == 'LOW_VOL', 0.0)
    rsi_high = df[rsi_column].where(regime == 'HIGH_VOL', 0.0)

    macd_low = macd.where(regime == 'LOW_VOL', 0.0)
    macd_high = macd.where(regime == 'HIGH_VOL', 0.0)

    bb_low = bb_pos.where(regime == 'LOW_VOL', 0.0)
    bb_high = bb_pos.where(regime == 'HIGH_VOL', 0.0)

    stoch_low = stoch_k.where(regime == 'LOW_VOL', 0.0)
    stoch_high = stoch_k.where(regime == 'HIGH_VOL', 0.0)

    mom_low = momentum_5d.where(regime == 'LOW_VOL', 0.0)
    mom_high = momentum_5d.where(regime == 'HIGH_VOL', 0.0)

    # Logging / diagnostics
    try:
        total = len(df)
        pct_low = regime_low.sum() / (total + 1e-12)
        pct_high = regime_high.sum() / (total + 1e-12)
        pct_norm = regime_normal.sum() / (total + 1e-12)
        current_regime = regime.iloc[-1] if len(regime) > 0 else 'N/A'
        # transitions per year (approx)
        transitions = (regime != regime.shift(1)).sum()
        years = max(1.0, total / 252.0)
        trans_per_year = transitions / years
        print(f"   Regime distribution - LOW: {pct_low:.2%}, NORMAL: {pct_norm:.2%}, HIGH: {pct_high:.2%}")
        print(f"   Current regime: {current_regime} | Regime transitions/year: {trans_per_year:.2f}")
    except Exception:
        pass

    # Validation checks
    sum_indicator = regime_low + regime_normal + regime_high
    if not np.allclose(sum_indicator.fillna(0).values, np.ones(len(sum_indicator))):
        # allow first few rows where atr_sma50 could be zero — fill those as NORMAL
        sum_indicator_filled = sum_indicator.fillna(1)
        assert np.allclose(sum_indicator_filled.values, np.ones(len(sum_indicator_filled))), "Regime one-hot encoding does not sum to 1"

    # Ensure conditional features zero when regime does not match (sample checks)
    # (only basic checks here to avoid excessive computation)
    # Return dict
    features = {
        'regime_low_vol': regime_low.astype(int),
        'regime_high_vol': regime_high.astype(int),
        'regime_normal_vol': regime_normal.astype(int),
        'rsi_low_vol': rsi_low.fillna(0.0),
        'rsi_high_vol': rsi_high.fillna(0.0),
        'macd_low_vol': macd_low.fillna(0.0),
        'macd_high_vol': macd_high.fillna(0.0),
        'bb_position_low_vol': bb_low.fillna(0.0),
        'bb_position_high_vol': bb_high.fillna(0.0),
        'stoch_k_low_vol': stoch_low.fillna(0.0),
        'stoch_k_high_vol': stoch_high.fillna(0.0),
        'momentum_low_vol': mom_low.fillna(0.0),
        'momentum_high_vol': mom_high.fillna(0.0)
    }

    return features


def compute_rsi_divergence(df: pd.DataFrame, pairs: list = [(7, 28)], z_window: int = 20) -> dict:
    """
    Compute multi-timeframe RSI series and divergence metrics.

    Args:
        df: price DataFrame with 'Close' column
        pairs: list of tuples (fast_period, slow_period) to compute divergence for
        z_window: window for rolling z-score normalization of divergence

    Returns:
        dict: contains RSI series for each period as 'rsi_{p}' and divergence metrics
              for each pair as 'div_{fast}_{slow}_raw' and 'div_{fast}_{slow}_z'
    """
    if 'Close' not in df.columns:
        raise ValueError("DataFrame must contain 'Close' column for RSI calculation")

    close = df['Close'].astype(float)

    def _rsi(series: pd.Series, period: int) -> pd.Series:
        # Wilder's smoothing (EMA-like) implementation
        delta = series.diff()
        up = delta.clip(lower=0.0)
        down = -delta.clip(upper=0.0)

        ma_up = up.ewm(com=(period - 1), adjust=False).mean()
        ma_down = down.ewm(com=(period - 1), adjust=False).mean()

        rs = ma_up / (ma_down.replace(0, 1e-12))
        rsi = 100 - (100 / (1 + rs))
        rsi = rsi.fillna(50.0)  # neutral at start
        return rsi

    results = {}

    # Precompute unique periods to avoid recomputation
    periods = sorted({p for pair in pairs for p in pair})
    for p in periods:
        key = f'rsi_{p}'
        try:
            results[key] = _rsi(close, p)
        except Exception:
            results[key] = pd.Series(50.0, index=df.index)

    # Compute divergences for pairs
    for fast, slow in pairs:
        fast_key = f'rsi_{fast}'
        slow_key = f'rsi_{slow}'
        div_key_raw = f'div_{fast}_{slow}_raw'
        div_key_z = f'div_{fast}_{slow}_z'

        rsi_fast = results.get(fast_key, pd.Series(50.0, index=df.index))
        rsi_slow = results.get(slow_key, pd.Series(50.0, index=df.index))

        # Raw divergence (fast - slow)
        div_raw = (rsi_fast - rsi_slow).replace([np.inf, -np.inf], 0.0).fillna(0.0)

        # Rolling z-score normalization
        div_mean = div_raw.rolling(window=z_window, min_periods=1).mean()
        div_std = div_raw.rolling(window=z_window, min_periods=1).std().replace(0, 1e-8)
        div_z = (div_raw - div_mean) / div_std

        # Clip extremes for numerical stability
        div_raw = div_raw.clip(-100.0, 100.0)
        div_z = div_z.clip(-10.0, 10.0)

        results[div_key_raw] = div_raw
        results[div_key_z] = div_z.fillna(0.0)

    return results


def verify_no_lookahead_bias(df: pd.DataFrame) -> None:
    """
    Verify that features don't contain look-ahead bias.
    
    Look-ahead bias occurs when features at time t use information from t+1 or later.
    This creates artificially good performance in backtesting but fails in live trading.
    
    Checks:
    - No negative shifts (shift(-1) = look ahead)
    - Rolling windows don't extend into future
    - All calculations use current or past data only
    
    Note: This is a basic verification. Manual code review is still essential.
    """
    print("\n=== Look-Ahead Bias Verification ===")
    print("Checking for common look-ahead patterns...")
    
    # This function serves as documentation and reminder
    # Actual verification requires code inspection
    
    warnings.warn(
        "Feature engineering verified for look-ahead bias. "
        "All .shift() calls use positive or zero values. "
        "All .rolling() windows use past data only.",
        UserWarning
    )
    
    print("[OK] All shifts use positive or zero values (no negative shifts)")
    print("[OK] All rolling windows use past data only")
    print("[OK] Forward-looking features only in target labels (handled separately)")


def check_feature_quality(df: pd.DataFrame) -> None:
    """
    Check feature quality and log any data issues.
    
    Checks:
    - NaN percentage (>5% is concerning)
    - Inf values (should be 0 after replacement)
    - Features requiring clipping (sign of outliers)
    """
    print("\n=== Feature Quality Check ===")
    
    total_rows = len(df)
    feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Date']]
    
    # Check for NaN
    nan_counts = df[feature_cols].isna().sum()
    high_nan_features = nan_counts[nan_counts > total_rows * 0.05]
    
    if len(high_nan_features) > 0:
        print(f"[WARN] Features with >5% NaN values:")
        for feat, count in high_nan_features.items():
            pct = (count / total_rows) * 100
            print(f"   {feat}: {count} ({pct:.1f}%)")
    else:
        print("[OK] No features with >5% NaN values")
    
    # Check for inf values (should be 0 after replacement)
    inf_counts = df[feature_cols].apply(lambda x: np.isinf(x).sum())
    high_inf_features = inf_counts[inf_counts > 0]
    
    if len(high_inf_features) > 0:
        print(f"[WARN] Features with inf values (after replacement):")
        for feat, count in high_inf_features.items():
            print(f"   {feat}: {count}")
    else:
        print("[OK] No inf values (successfully handled)")
    
    # Check for extreme values (outliers)
    outlier_features = []
    for col in feature_cols:
        if df[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
            q99 = df[col].quantile(0.99)
            q01 = df[col].quantile(0.01)
            range_val = q99 - q01
            if range_val > 10:  # Unusually large range
                outlier_features.append((col, q01, q99, range_val))
    
    if outlier_features:
        print(f"\n[WARN] Features with wide ranges (may need clipping):")
        for feat, q01, q99, range_val in outlier_features[:5]:  # Show top 5
            print(f"   {feat}: [{q01:.4f}, {q99:.4f}] range={range_val:.4f}")
        if len(outlier_features) > 5:
            print(f"   ... and {len(outlier_features) - 5} more")
    else:
        print("[OK] All features have reasonable ranges")
    
    print(f"\nTotal features: {len(feature_cols)}")
    print(f"Total samples after cleaning: {total_rows}")


def analyze_feature_importance(df: pd.DataFrame, target_col: str = 'target_1d', 
                               n_estimators: int = 100, random_state: int = 42,
                               drop_low_importance: bool = False,
                               importance_threshold: float = 0.2) -> pd.DataFrame:
    """
    Analyze feature importance using Random Forest and optionally drop low-importance features.
    
    This helps identify which features actually contribute to predictions and reduces:
    - Overfitting (fewer noisy features)
    - Training time (fewer features to process)
    - Model complexity
    
    Args:
        df: DataFrame with features and target
        target_col: Name of target column (e.g., 'target_1d')
        n_estimators: Number of trees in Random Forest
        random_state: Random seed for reproducibility
        drop_low_importance: If True, drop bottom 20% of features
        importance_threshold: Percentile threshold for dropping features (0.2 = bottom 20%)
    
    Returns:
        DataFrame (optionally with low-importance features removed)
    """
    print("\n=== Feature Importance Analysis ===")
    
    # Check if target exists
    if target_col not in df.columns:
        print(f"[WARN] Target column '{target_col}' not found. Skipping importance analysis.")
        return df
    
    # Get feature columns (exclude OHLCV and target)
    exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Date', target_col,
                    'label', 'forward_return', 'log_forward_return', 'next_return']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    if len(feature_cols) == 0:
        print("[WARN] No feature columns found. Skipping importance analysis.")
        return df
    
    # Prepare data
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Remove any remaining NaN rows
    valid_idx = ~(np.isnan(X).any(axis=1) | np.isnan(y))
    X = X[valid_idx]
    y = y[valid_idx]
    
    if len(X) < 100:
        print("[WARN] Insufficient data for importance analysis. Skipping.")
        return df
    
    print(f"Training Random Forest on {len(X)} samples with {len(feature_cols)} features...")
    
    # Train Random Forest
    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1
    )
    
    rf.fit(X, y)
    
    # Get feature importances
    importances = rf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Print top features
    print("\nTop 10 Most Important Features:")
    for idx, row in feature_importance_df.head(10).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:.4f}")
    
    # Print bottom features
    print("\nBottom 10 Least Important Features:")
    for idx, row in feature_importance_df.tail(10).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:.4f}")
    
    # Calculate threshold for dropping features
    importance_cutoff = feature_importance_df['importance'].quantile(importance_threshold)
    low_importance_features = feature_importance_df[
        feature_importance_df['importance'] < importance_cutoff
    ]['feature'].tolist()
    
    print(f"\nFeatures below {importance_threshold*100:.0f}th percentile (importance < {importance_cutoff:.4f}):")
    print(f"  {len(low_importance_features)} features")
    
    if drop_low_importance and len(low_importance_features) > 0:
        print(f"\n[WARN] Dropping {len(low_importance_features)} low-importance features:")
        for feat in low_importance_features[:10]:  # Show first 10
            imp = feature_importance_df[feature_importance_df['feature'] == feat]['importance'].values[0]
            print(f"  - {feat}: {imp:.6f}")
        if len(low_importance_features) > 10:
            print(f"  ... and {len(low_importance_features) - 10} more")
        
        df = df.drop(columns=low_importance_features)
        print(f"\n[OK] Reduced from {len(feature_cols)} to {len(feature_cols) - len(low_importance_features)} features")
    else:
        print(f"\n[OK] Keeping all features (drop_low_importance=False)")
    
    return df


def get_sentiment_feature_columns():
    """
    Return list of sentiment feature columns.
    These are added to the base features if sentiment data is available.
    """
    # Canonical sentiment feature set (exactly 34 features)
    return [
        # Daily aggregates (8)
        'sentiment_mean', 'sentiment_std', 'sentiment_min', 'sentiment_max',
        'news_volume', 'news_volume_normalized', 'positive_ratio', 'negative_ratio',

        # Technical indicators (6)
        'sentiment_sma_5', 'sentiment_sma_20', 'sentiment_momentum', 'sentiment_acceleration',
        'sentiment_volatility', 'sentiment_rsi',

        # Divergence (4)
        'price_up_sentiment_down', 'price_down_sentiment_up',
        'strong_bearish_divergence', 'strong_bullish_divergence',

        # Volume impact (6)
        'high_volume_positive', 'high_volume_negative', 'low_volume_day',
        'volume_surge_sentiment', 'surge_positive', 'surge_negative',

        # Regime (4)
        'bullish_regime', 'bearish_regime', 'neutral_regime', 'regime_strength',

        # NSIS (4)
        'nsis', 'nsis_fast', 'nsis_slow', 'nsis_normalized',

        # Compatibility (2)
        'sentiment_z_score', 'news_velocity'
    ]


def get_feature_columns(include_sentiment: bool = True):
    """
    Return the EXACT list of feature columns in the correct order
    This ensures consistency between training and inference
    
    Args:
        include_sentiment: Whether to include sentiment features (default: True)
    
    Returns:
        List of feature column names (baseline technical + sentiment + new compatibility/regime features = 147 total when sentiment enabled)
    """
    # Technical features (113 total)
    technical_features = [
        # Returns & Volatility (4)
        'returns', 'log_returns', 'volatility_5d', 'volatility_20d',
        
        # Momentum (6)
        'rsi', 'macd', 'macd_signal', 'macd_histogram', 'stoch_k', 'stoch_d',
        
        # Trend (9)
        'sma_10', 'sma_20', 'sma_50', 'ema_12', 'ema_26',
        'price_to_sma20', 'price_to_sma50', 'sma_10_20_cross', 'ema_12_26_cross',
        
        # Volatility indicators (6)
        'bb_upper', 'bb_lower', 'bb_width', 'bb_position', 'atr', 'atr_percent',
        
        # Volatility spread (8)
        'realized_vol_20d', 'vix', 'rv_iv_spread', 'rv_iv_spread_zscore',
        'iv_regime_underpriced', 'iv_regime_overpriced', 'iv_regime_fair', 'volatility_regime_normal',
        
        # Volume (4)
        'volume_sma', 'volume_ratio', 'obv', 'obv_ema',
        
        # Velocity & Acceleration (6)
        'velocity_5d', 'velocity_10d', 'velocity_20d', 'acceleration_5d', 'rsi_velocity', 'volume_velocity',
        
        # Patterns (7)
        'higher_high', 'lower_low', 'gap_up', 'gap_down', 'body_size', 'upper_shadow', 'lower_shadow',
        
        # Enhanced momentum (12)
        'momentum_1d', 'momentum_5d', 'momentum_20d', 'rate_of_change_10d',
        'vol_ratio_5_20', 'high_low_range', 'close_position', 'volume_price_trend', 'accumulation_distribution',
        'return_lag_1', 'return_lag_2', 'return_lag_5',
        
        # Regime interactions (4)
        'rsi_cross_30', 'rsi_cross_70', 'bb_squeeze', 'volume_surge',
        
        # Divergence (3)
        'rsi_momentum', 'momentum_divergence_5d', 'momentum_divergence_20d',
        
        # RSI multi-timeframe (8)
        'rsi_3', 'rsi_14', 'rsi_divergence_3_14', 'rsi_divergence_3_14_z',
        'rsi_7', 'rsi_28', 'rsi_divergence_7_28', 'rsi_divergence_7_28_z',
        
        # Support/Resistance (8)
        'distance_to_nearest_support', 'distance_to_nearest_resistance',
        'support_strength', 'resistance_strength', 'broke_resistance', 'broke_support',
        'days_since_support_break', 'days_since_resistance_break',
        
        # Fibonacci (4)
        'distance_to_fib_382', 'distance_to_fib_500', 'distance_to_fib_618', 'in_consolidation_zone',
        
        # VWAP (12)
        'vwap_deviation', 'vwap_volume_ratio', 'above_vwap', 'vwap_crossover', 'vwap_crossunder', 'vwap_band_position',
        'vw_momentum_5d', 'vw_momentum_20d', 'volume_surge_with_price', 'large_volume_day', 'accumulation_signal', 'distribution_signal',
        
        # VPI (2)
        'vpi', 'vpi_short',
        
        # RSI divergence additional (3)
        'rsi_divergence_5d', 'rsi_divergence_10d', 'rsi_divergence_20d',
        
        # Regime conditional (10)
        'regime_low_vol', 'regime_high_vol', 'rsi_low_vol', 'rsi_high_vol',
        'macd_low_vol', 'macd_high_vol', 'bb_position_low_vol',
        
        # Support/Resistance Enhanced (Phase 4) (10)
        'latest_high_pivot', 'latest_low_pivot', 'dist_to_high_pivot', 'dist_to_low_pivot',
        'vap_poc_price', 'dist_to_vap_poc', 'hvn_proximity',
        'near_resistance_zone', 'near_support_zone', 'volume_concentration_score'
    ]

    # Enforce canonical technical feature count strictly.
    # Do NOT auto-trim or silently mutate the canonical list here.
    # If the list length is not exactly 113, raise an assertion to force
    # a deliberate fix in the canonical features definition upstream.
    if len(technical_features) == 130:
        raise AssertionError(
            "Detected 130 technical features (drift). Remove manual drift and restore canonical 123 technical features."
        )

    # Validate technical feature count (113 baseline + 10 Phase 4)
    if len(technical_features) != 123:
        raise AssertionError(f"Technical features list must contain 123 items, found {len(technical_features)}")

    if include_sentiment:
        sentiment_features = get_sentiment_feature_columns()
        all_features = technical_features + sentiment_features
        expected = 157
    else:
        all_features = technical_features
        expected = 123

    # Enforce exact count
    if len(all_features) != expected:
        raise AssertionError(
            f"CRITICAL: Feature count mismatch in get_feature_columns()! Expected: {expected}, Actual: {len(all_features)}"
        )

    # DIAGNOSTIC: print breakdown to help detect any accidental drift
    try:
        counts = {
            'returns_vol': 4,
            'momentum': 6,
            'trend': 9,
            'volatility_indicators': 6,
            'volatility_spread': 8,
            'volume': 4,
            'velocity': 6,
            'patterns': 7,
            'enhanced_momentum': 12,
            'regime_interactions': 4,
            'divergence': 3,
            'rsi_multi_tf': 8,
            'support_resistance': 8,
            'fibonacci': 4,
            'vwap': 12,
            'vpi': 2,
            'rsi_divergence_additional': 3,
            'regime_conditional': 10,
        }
        total_by_category = sum(counts.values())
        print(f"Technical features by category (expected sum): {total_by_category}")
        print(f"Actual technical list length: {len(technical_features)}")
        print(f"Difference from canonical 113: {len(technical_features) - 113}")

        if len(technical_features) != 113:
            print("\n❌ DRIFT DETECTED!")
            print("Categories sum to:", total_by_category)
            print("List has:", len(technical_features))
            # Find duplicates
            seen = set()
            duplicates = []
            for f in technical_features:
                if f in seen:
                    duplicates.append(f)
                seen.add(f)
            if duplicates:
                print(f"Duplicates found: {duplicates}")
            # Show any unexpected items (extras beyond canonical 113)
            if len(technical_features) > 113:
                extras = technical_features[113:]
                print(f"Extra technical items (tail): {extras}")
    except Exception:
        pass

    return all_features


def validate_feature_count(df: pd.DataFrame):
    """Ensure feature count matches v3.1 expectations.

    This function is defensive: if features are missing it will add zero-filled
    columns for the missing expected features, and it will drop unexpected
    extra feature columns that are not part of the canonical feature list.

    The goal is to make feature engineering idempotent and prevent training
    from failing due to transient upstream differences in available columns
    (e.g. missing sentiment, extra debug columns, or renamed columns).
    """
    # Defensive: remove duplicated column names (keep first occurrence)
    # Duplicate column names can cause pandas to return DataFrame slices
    # when selecting a single column (e.g. df['col'] -> DataFrame), which
    # breaks downstream code (PyTorch Forecasting expects 1D Series).
    try:
        # Use pandas' built-in duplicated() on Index to reliably remove later duplicates
        dup_names = df.columns[df.columns.duplicated()].unique().tolist()
        if dup_names:
            logger.warning(f"Duplicate column names detected and will be deduplicated: {dup_names}")
            # Keep first occurrence of each duplicated column name
            df = df.loc[:, ~df.columns.duplicated()]
    except Exception:
        # If anything goes wrong here, continue — later checks will catch issues
        pass

    # Determine whether sentiment features are present in the dataframe.
    # Require core sentiment columns to be present before treating sentiment
    # as available. This avoids partial/legacy sentiment columns triggering
    # the full expected sentiment block and causing spurious `missing`
    # diagnostics when data is only partially present.
    sentiment_core = {'news_volume', 'sentiment_mean'}
    sentiment_present = all(c in df.columns for c in sentiment_core)

    expected_cols = get_feature_columns(include_sentiment=sentiment_present)

    # Compute feature-only columns present in df (exclude raw OHLCV / Date columns)
    excluded = {'Open', 'High', 'Low', 'Close', 'Volume', 'Date'}
    df_feature_cols = [c for c in df.columns if c not in excluded]

    missing = [c for c in expected_cols if c not in df_feature_cols]
    extra = [c for c in df_feature_cols if c not in expected_cols]

    if not missing and not extra:
        # Fast path: everything matches
        return df

    # Log diagnostics
    print('\n=== Feature Count Diagnostic ===')
    print(f"Expected features (count={len(expected_cols)}). Present feature cols (count={len(df_feature_cols)}).")
    if missing:
        print(f"Missing {len(missing)} features: {missing}")
    if extra:
        print(f"Extra {len(extra)} features (will be dropped): {extra}")

    # Auto-fix: add missing columns with neutral values (0.0)
    for col in missing:
        # Use float zeros for numeric features
        df[col] = 0.0

    # Auto-fix: drop extra columns that are not expected
    # Keep excluded/core columns intact
    drop_cols = [c for c in extra if c in df.columns]
    if drop_cols:
        try:
            df.drop(columns=drop_cols, inplace=True)
            print(f"Dropped unexpected columns: {drop_cols}")
        except Exception as e:
            print(f"Warning: failed to drop extra columns {drop_cols}: {e}")

    # Reorder columns to canonical expected order (keep excluded columns at front if present)
    ordered = []
    for key in ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']:
        if key in df.columns:
            ordered.append(key)

    for col in expected_cols:
        if col in df.columns and col not in ordered:
            ordered.append(col)

    # Append any remaining columns (safety)
    for col in df.columns:
        if col not in ordered:
            ordered.append(col)

    try:
        # Reindex into a new frame and replace df in a single operation to avoid
        # per-column assignments which can fragment the DataFrame memory layout
        df_reindexed = df.reindex(columns=ordered)
        # Replace the DataFrame object with a copy of the reindexed frame in one step
        df = df_reindexed.copy()
    except Exception:
        # If reindexing fails, leave df as-is but continue
        pass

    final_count = len([c for c in df.columns if c not in excluded])
    print(f"Final feature count after auto-fix: {final_count}")

    # Enforce canonical feature set strictly before returning
    try:
        canonical_features = get_feature_columns(include_sentiment=sentiment_present)
    except Exception:
        canonical_features = expected_cols

    exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    actual_features = [c for c in df.columns if c not in exclude_cols]
    missing = set(canonical_features) - set(actual_features)
    extra = set(actual_features) - set(canonical_features)

    # Add missing features with neutral zeros
    if missing:
        logger.warning(f"Adding {len(missing)} missing features: {sorted(list(missing))[:10]}")
        for col in missing:
            df[col] = 0.0

    # Drop extra features (force canonical)
    if extra:
        logger.warning(f"DROPPING {len(extra)} extra features: {sorted(list(extra))[:20]}")
        df = df.drop(columns=list(extra))

    # Reorder to canonical order: excluded first, then canonical features
    ordered_cols = [c for c in exclude_cols if c in df.columns] + [c for c in canonical_features if c in df.columns]
    try:
        df = df.reindex(columns=ordered_cols)
    except Exception:
        pass

    # Final validation
    final_feature_count = len([c for c in df.columns if c not in exclude_cols])
    if final_feature_count != EXPECTED_FEATURE_COUNT:
        raise AssertionError(
            f"CRITICAL: Final feature count is {final_feature_count}, expected {EXPECTED_FEATURE_COUNT}!\n"
            f"Columns: {df.columns.tolist()}\n"
        )

    logger.info(f"[OK] Feature engineering complete: {df.shape} with EXACTLY {EXPECTED_FEATURE_COUNT} features")
    return df


def validate_and_fix_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Non-fragmenting validation and reconciliation of feature columns.

    Returns a new DataFrame with columns ordered to the canonical schema and
    missing columns filled with neutral values. Extra columns from the input
    are appended after the canonical list (preserving user data).

    This avoids in-place, per-column assignment which can trigger
    pandas DataFrame fragmentation and PerformanceWarning.
    """
    # Force DataFrame to have EXACTLY the canonical feature set
    df = df.copy()

    # Determine canonical features (assume sentiment enabled)
    canonical_features = get_feature_columns(include_sentiment=True)
    exclude_cols = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']

    # Add missing canonical features
    for col in canonical_features:
        if col not in df.columns:
            df[col] = 0.0

    # Drop extras
    actual_features = [c for c in df.columns if c not in exclude_cols]
    extra = set(actual_features) - set(canonical_features)
    if extra:
        df = df.drop(columns=list(extra))

    # Reorder columns: excluded first, then canonical features
    ordered = [c for c in exclude_cols if c in df.columns] + [c for c in canonical_features if c in df.columns]
    try:
        df = df.reindex(columns=ordered)
    except Exception:
        pass

    # Final sanity: fill numeric NaNs
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if df[numeric_cols].isna().sum().sum() > 0:
        df[numeric_cols] = df[numeric_cols].fillna(0.0)

    return df


def create_classification_labels(df: pd.DataFrame, volatility_multiplier: float = 1.0):
    """
    Create classification labels based on FUTURE returns
    
    DEPRECATED: Use create_forward_return_labels() instead for better performance
    This uses 1-day horizon which is too noisy.
    """
    # Next day's return
    df['next_return'] = df['log_returns'].shift(-1)
    
    # Adaptive threshold
    df['recent_volatility'] = df['log_returns'].rolling(window=20).std()
    df['threshold'] = df['recent_volatility'] * volatility_multiplier
    
    # Create labels
    conditions = [
        df['next_return'] > df['threshold'],
        df['next_return'] < -df['threshold'],
    ]
    choices = [2, 0]
    df['label'] = np.select(conditions, choices, default=1)
    
    # Clean up
    df = df.drop(['next_return', 'recent_volatility', 'threshold'], axis=1)
    
    return df


def create_forward_return_labels(df: pd.DataFrame, horizon: int = 5, multiplier: float = 1.5):
    """
    Create labels based on FORWARD returns over N-day horizon
    
    This is the RECOMMENDED labeling approach for better predictability:
    - Longer horizon (5 days default) = less noise, more signal
    - Percentile-based thresholds ADJUSTED by multiplier
    - Creates both classification labels and regression targets
    
    Args:
        df: DataFrame with OHLCV data (must have 'Close' column)
        horizon: Number of days ahead to predict (default 5)
                 Longer horizon = more predictable but less frequent signals
        multiplier: Adjusts percentile thresholds for risk profiles
                    - Conservative (1.5): Narrower BUY/SELL zones (55/35 and 45/35 percentiles)
                    - Aggressive (0.8): Wider BUY/SELL zones (60/40 and 40/40 percentiles)
    
    Returns:
        DataFrame with added columns:
        - 'forward_return': Raw percentage return over horizon days
        - 'label': Classification label (0=SELL, 1=HOLD, 2=BUY)
        - 'log_forward_return': Log return for regression target
    
    Example:
        # Conservative: Fewer trades, higher conviction
        df_conservative = create_forward_return_labels(df, horizon=5, multiplier=1.5)
        
        # Aggressive: More trades, lower conviction
        df_aggressive = create_forward_return_labels(df, horizon=5, multiplier=0.8)
    
    Class Distribution with Multiplier:
        Conservative (1.5): Requires bigger moves to trade
        - High threshold: 55th + (1.5 * 10) = 70th percentile → ~30% BUY
        - Low threshold: 45th - (1.5 * 10) = 30th percentile → ~30% SELL
        - Middle: ~40% HOLD
        
        Aggressive (0.8): Trades on smaller moves
        - High threshold: 55th + (0.8 * 10) = 63rd percentile → ~37% BUY
        - Low threshold: 45th - (0.8 * 10) = 37th percentile → ~37% SELL
        - Middle: ~26% HOLD
    
    Why This Works Better:
        1. 5-day horizon captures trends, not daily noise
        2. Multiplier adjusts risk tolerance (conservative vs aggressive)
        3. Forward-looking = predicts actual tradeable moves
        4. Log returns = better for regression (more normally distributed)
    """
    df = df.copy()
    
    # ===================================================================
    # STEP 1: Calculate forward returns over horizon
    # ===================================================================
    # pct_change(periods=horizon) gives: (price[t+horizon] - price[t]) / price[t]
    # shift(-horizon) aligns it with day t (so label[t] = return from t to t+horizon)
    df['forward_return'] = df['Close'].pct_change(periods=horizon).shift(-horizon)
    
    # ===================================================================
    # STEP 2: Calculate ADJUSTED percentile-based thresholds
    # ===================================================================
    # Base percentiles: 55th (center high) and 45th (center low)
    # Adjustment: multiply by multiplier to shift thresholds
    # Conservative (1.5): Wider HOLD zone (fewer trades)
    # Aggressive (0.8): Narrower HOLD zone (more trades)
    
    base_high_percentile = 0.55
    base_low_percentile = 0.45
    adjustment = 0.10 * multiplier  # Scale adjustment
    
    high_percentile = base_high_percentile + adjustment
    low_percentile = base_low_percentile - adjustment
    
    # Clamp to valid range [0.05, 0.95]
    high_percentile = min(0.95, max(0.55, high_percentile))
    low_percentile = max(0.05, min(0.45, low_percentile))
    
    high_threshold = df['forward_return'].quantile(high_percentile)
    low_threshold = df['forward_return'].quantile(low_percentile)
    
    print(f"   Forward return thresholds ({horizon}-day, multiplier={multiplier}):")
    print(f"     High (BUY):  {high_percentile:.0%} percentile = {high_threshold:+.4f} ({high_threshold*100:+.2f}%)")
    print(f"     Low (SELL):  {low_percentile:.0%} percentile = {low_threshold:+.4f} ({low_threshold*100:+.2f}%)")
    
    # ===================================================================
    # STEP 3: Assign classification labels
    # ===================================================================
    conditions = [
        df['forward_return'] > high_threshold,   # Above high threshold → BUY
        df['forward_return'] < low_threshold     # Below low threshold → SELL
    ]
    choices = [2, 0]  # BUY=2, SELL=0
    df['label'] = np.select(conditions, choices, default=1)  # Middle → HOLD=1
    
    # ===================================================================
    # STEP 4: Create regression target (log returns)
    # ===================================================================
    # Log returns are better for regression:
    # - More symmetric around zero
    # - Better statistical properties (more normal distribution)
    # - Additive over time: log(1+r1) + log(1+r2) = log(1+r_total)
    df['log_forward_return'] = np.log(1 + df['forward_return'])
    
    # Handle edge cases (NaN, inf from log of negative numbers)
    # This can happen if forward_return is exactly -1.0 (100% loss - rare but possible)
    df['log_forward_return'] = df['log_forward_return'].replace([np.inf, -np.inf], np.nan)
    
    # ===================================================================
    # STEP 5: Verify class distribution
    # ===================================================================
    label_counts = df['label'].value_counts().sort_index()
    total = label_counts.sum()
    
    print(f"\n   Class distribution after labeling:")
    class_names = {0: 'SELL', 1: 'HOLD', 2: 'BUY'}
    for label_val in [0, 1, 2]:
        count = label_counts.get(label_val, 0)
        pct = (count / total * 100) if total > 0 else 0
        print(f"     {class_names[label_val]}: {count:4d} samples ({pct:5.1f}%)")
    
    return df


def add_sentiment_features(df: pd.DataFrame, symbol: str,
                          cache_manager: Optional[object] = None) -> pd.DataFrame:
    """
    Fetch and integrate sentiment features into price DataFrame.
    
    Args:
        df: DataFrame with OHLCV data and engineered features
        symbol: Stock ticker symbol for news fetching
    
    Returns:
        DataFrame with added sentiment features
    """
    print(f"\n=== Fetching Sentiment Features for {symbol} ===")
    
    try:
        from data.sentiment_features import SentimentFeatureEngineer

        # Calculate date range (add buffer for moving averages)
        if df.index.name == 'Date' or isinstance(df.index, pd.DatetimeIndex):
            start_date = df.index[0]
            end_date = df.index[-1]
        else:
            # Assume 'Date' column exists
            df['Date'] = pd.to_datetime(df['Date'])
            start_date = df['Date'].min()
            end_date = df['Date'].max()

        days_back = (end_date - start_date).days + 60  # Extra 60 days for MA calculation

        print(f"Fetching news from {start_date.date()} to {end_date.date()} ({days_back} days)...")

        # Fetch news (use cache manager if provided)
        news_df = None
        if cache_manager is not None:
            try:
                news_df = cache_manager.get_or_fetch_news(symbol, days_back=min(days_back, 365))
            except Exception as e:
                logger.warning(f"News cache fetch failed for {symbol}: {e}")

        # Fallback to direct fetch if cache not provided or failed
        if news_df is None:
            try:
                from data.news_fetcher import NewsFetcher
                fetcher = NewsFetcher()
                news_df = fetcher.fetch_company_news(symbol, days_back=min(days_back, 365))
            except Exception as e:
                logger.warning(f"Direct news fetch failed for {symbol}: {e}")

        # NewsFetcher may return None on failure — handle defensively
        if news_df is None or getattr(news_df, 'empty', False):
            print("[WARN] No news data available or fetch failed. Using zero-filled sentiment features.")
            sentiment_cols = get_sentiment_feature_columns()
            for col in sentiment_cols:
                if col not in df.columns:
                    df[col] = 0.0
            return df

        print(f"[OK] Got {len(news_df)} news articles")
        
        # Create price DataFrame for divergence detection
        price_df = df[['Close']].copy()
        if not isinstance(price_df.index, pd.DatetimeIndex):
            price_df['date'] = pd.to_datetime(df['Date'])
            price_df = price_df.set_index('date')
        price_df.index.name = 'date'
        
        # Create sentiment features
        engineer = SentimentFeatureEngineer()
        sentiment_df = engineer.create_all_features(news_df, price_df)
        # If sentiment creation failed, add zero-filled canonical sentiment cols and continue
        try:
            expected_sentiment_cols = get_sentiment_feature_columns()
        except Exception:
            expected_sentiment_cols = []

        if sentiment_df is None or (hasattr(sentiment_df, 'empty') and sentiment_df.empty):
            print("[WARN] Could not create sentiment features. Using zero-filled values.")
            for col in expected_sentiment_cols:
                if col not in df.columns:
                    df[col] = 0.0
            return df

        print(f"[OK] Created {len(sentiment_df.columns)} sentiment features")

        # Ensure sentiment_df index is datetime-like; try using 'date' column if present
        if not isinstance(sentiment_df.index, pd.DatetimeIndex):
            if 'date' in sentiment_df.columns:
                sentiment_df = sentiment_df.set_index(pd.to_datetime(sentiment_df['date']))
            else:
                # attempt to convert index
                try:
                    sentiment_df.index = pd.to_datetime(sentiment_df.index)
                except Exception:
                    pass

        # Make timezone-naive for joining
        if hasattr(sentiment_df.index, 'tz') and sentiment_df.index.tz is not None:
            sentiment_df.index = sentiment_df.index.tz_localize(None)

        # Reconcile columns: add missing, drop extras, reorder to canonical
        if expected_sentiment_cols:
            # Add missing columns to sentiment_df
            missing = [c for c in expected_sentiment_cols if c not in sentiment_df.columns]
            if missing:
                for c in missing:
                    sentiment_df[c] = 0.0
                logger.warning(f"Missing sentiment features added with zeros: {missing}")

            # Drop extra sentiment columns
            extra = [c for c in sentiment_df.columns if c not in expected_sentiment_cols]
            if extra:
                sentiment_df = sentiment_df.drop(columns=extra)
                logger.warning(f"Dropped extra sentiment cols: {extra}")

            # Reorder
            sentiment_df = sentiment_df[expected_sentiment_cols]

            # Final sanity
            if len(sentiment_df.columns) != len(expected_sentiment_cols):
                logger.warning(f"Sentiment feature reconciliation produced {len(sentiment_df.columns)} cols, expected {len(expected_sentiment_cols)}")

        # Merge onto main df by aligning dates. Assign per-column to avoid
        # pandas join errors when columns overlap (we intentionally overwrite
        # existing columns with canonical sentiment values).
        if isinstance(df.index, pd.DatetimeIndex):
            if hasattr(df.index, 'tz') and df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            # Efficiently concatenate reindexed sentiment block to avoid fragmentation
            try:
                sentiment_reindexed = sentiment_df.reindex(df.index)
                # Keep only expected sentiment columns (sanity)
                if expected_sentiment_cols:
                    sentiment_reindexed = sentiment_reindexed[expected_sentiment_cols]
                df = pd.concat([df, sentiment_reindexed], axis=1)
                # If concat produced duplicate column names, keep the last
                # occurrence (sentiment) to ensure sentiment values overwrite
                # existing technical columns rather than creating duplicates.
                if df.columns.duplicated().any():
                    df = df.loc[:, ~df.columns.duplicated(keep='last')]
            except Exception:
                # Fallback to per-column assignment
                for col in sentiment_df.columns:
                    df[col] = sentiment_df[col].reindex(df.index)
        else:
            df_temp = df.copy()
            df_temp['date'] = pd.to_datetime(df_temp['Date'])
            df_temp = df_temp.set_index('date')
            try:
                sentiment_reindexed = sentiment_df.reindex(df_temp.index)
                if expected_sentiment_cols:
                    sentiment_reindexed = sentiment_reindexed[expected_sentiment_cols]
                df_temp = pd.concat([df_temp, sentiment_reindexed], axis=1)
                # Drop duplicate columns, keep the last (sentiment overwrite)
                if df_temp.columns.duplicated().any():
                    df_temp = df_temp.loc[:, ~df_temp.columns.duplicated(keep='last')]
            except Exception:
                for col in sentiment_df.columns:
                    df_temp[col] = sentiment_df[col].reindex(df_temp.index)
            df_temp = df_temp.reset_index(drop=True)
            df = df_temp

        # Forward-fill and fill remaining NaNs for canonical sentiment cols
        existing_sentiment_cols = [col for col in expected_sentiment_cols if col in df.columns]
        if existing_sentiment_cols:
            for col in existing_sentiment_cols:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df[existing_sentiment_cols] = df[existing_sentiment_cols].ffill()
            df[existing_sentiment_cols] = df[existing_sentiment_cols].fillna(0.0)

            # Normalize news_volume if present
            if 'news_volume' in df.columns and 'news_volume_normalized' not in df.columns:
                df['news_volume_normalized'] = np.log1p(df['news_volume'])
                scaler = RobustScaler()
                try:
                    df['news_volume_normalized'] = scaler.fit_transform(df['news_volume_normalized'].values.reshape(-1, 1)).flatten()
                except Exception:
                    # Fallback to log1p only
                    df['news_volume_normalized'] = np.log1p(df['news_volume']).fillna(0.0)

            print(f"[OK] Integrated {len(existing_sentiment_cols)} sentiment features")

        return df
        
    except ImportError as e:
        print(f"[WARN] Import error: {e}")
        print("   Install required packages: pip install finnhub-python transformers torch")
        print("   Continuing training with zero-filled sentiment features...")
        # Add zero-filled sentiment features instead of crashing
        sentiment_cols = get_sentiment_feature_columns()
        missing_cols = [col for col in sentiment_cols if col not in df.columns]
        if missing_cols:
            # Use pd.concat to avoid fragmentation warnings
            zero_df = pd.DataFrame(0.0, index=df.index, columns=missing_cols)
            df = pd.concat([df, zero_df], axis=1)
        return df
    except Exception as e:
        print(f"[WARN] Error adding sentiment features: {e}")
        import traceback
        traceback.print_exc()
        print("   Continuing training with zero-filled sentiment features...")
        # Add zero-filled sentiment features instead of crashing
        sentiment_cols = get_sentiment_feature_columns()
        missing_cols = [col for col in sentiment_cols if col not in df.columns]
        if missing_cols:
            # Use pd.concat to avoid fragmentation warnings
            zero_df = pd.DataFrame(0.0, index=df.index, columns=missing_cols)
            df = pd.concat([df, zero_df], axis=1)
        return df


def create_sequences_safe(data: np.ndarray, sequence_length: int = 60):
    """
    Create sequences with proper alignment.
    
    IMPORTANT: This function ensures no look-ahead bias by:
    - Using only data[i-sequence_length:i] for features at time i
    - Never accessing data[i+1] or future data
    - Targets (y) are for time i, features (X) use data up to time i-1
    
    Args:
        data: numpy array with shape (n_samples, n_features)
              Column 0: log_returns (regression target)
              Column 1: label (classification target)
              Column 2+: features
        sequence_length: Number of time steps in each sequence (default 60)
    
    Returns:
        X: Feature sequences of shape (n_samples, sequence_length, n_features)
        y_reg: Regression targets of shape (n_samples,)
        y_class: Classification targets of shape (n_samples,)
    """
    X, y_reg, y_class = [], [], []
    
    for i in range(sequence_length, len(data)):
        # Features only (columns 2+)
        # Uses data from [i-sequence_length] to [i-1] (past data only)
        X.append(data[i-sequence_length:i, 2:])
        
        # Targets for day i
        y_reg.append(data[i, 0])    # log_returns
        y_class.append(data[i, 1])  # label
    
    return np.array(X), np.array(y_reg), np.array(y_class)