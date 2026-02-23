"""Deterministic feature engineering for the GBM-first pipeline.

This module intentionally keeps a compact, stable feature set with no external
news/sentiment dependencies by default. The goal is reproducible training and
low operational risk.
"""

from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

BASE_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

TECHNICAL_FEATURE_COLUMNS: List[str] = [
    # Returns & volatility (10)
    "ret_1d",
    "log_ret_1d",
    "vol_5d",
    "vol_10d",
    "vol_20d",
    "vol_ratio_5_20",
    "atr_14",
    "atr_pct_14",
    "garman_klass_10",
    "parkinson_10",
    # Momentum (11)
    "rsi_7",
    "rsi_14",
    "rsi_21",
    "stoch_k_14",
    "stoch_d_3",
    "williams_r_14",
    "cci_20",
    "roc_5",
    "roc_10",
    "roc_20",
    "tsi_25_13",
    # Trend (10)
    "sma_10_dist",
    "sma_20_dist",
    "sma_50_dist",
    "ema_12_dist",
    "ema_26_dist",
    "macd",
    "macd_signal",
    "macd_hist",
    "adx_14",
    "trend_strength",
    # Bands (6)
    "bb_upper_dist",
    "bb_lower_dist",
    "bb_width",
    "bb_pct_b",
    "kc_upper_dist",
    "kc_lower_dist",
    # Volume (7)
    "vol_sma_20_ratio",
    "obv",
    "obv_slope_5",
    "vwap_dist",
    "mfi_14",
    "vpt",
    "ad_line",
    # Multi-timeframe dynamics (8)
    "velocity_5",
    "velocity_10",
    "velocity_20",
    "accel_5",
    "rsi_velocity_5",
    "volume_velocity_5",
    "momentum_10_20",
    "price_accel_10",
    # Pattern (6)
    "higher_high",
    "lower_low",
    "gap_up",
    "gap_down",
    "body_size",
    "inside_bar",
    # Regime (6)
    "regime_low_vol",
    "regime_mid_vol",
    "regime_high_vol",
    "trend_dir",
    "mean_reversion_z",
    "trend_persistence",
    # Microstructure (4)
    "hl_range",
    "close_location",
    "intraday_vol",
    "overnight_ret",
    # Cross-asset proxies (4)
    "vix_proxy_level",
    "vix_proxy_change",
    "rv_iv_spread",
    "rv_iv_spread_zscore",
]

SENTIMENT_FEATURE_COLUMNS: List[str] = []

EXPECTED_FEATURE_COUNT = len(TECHNICAL_FEATURE_COLUMNS)


def _safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-9) -> pd.Series:
    return a / (b.abs() + eps)


def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = _safe_div(avg_gain, avg_loss)
    return 100.0 - (100.0 / (1.0 + rs))


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr_a = high - low
    tr_b = (high - prev_close).abs()
    tr_c = (low - prev_close).abs()
    return pd.concat([tr_a, tr_b, tr_c], axis=1).max(axis=1)


def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = _true_range(high, low, close)
    atr = tr.rolling(period, min_periods=period).mean()

    plus_di = 100.0 * _safe_div(plus_dm.rolling(period, min_periods=period).sum(), atr)
    minus_di = 100.0 * _safe_div(minus_dm.rolling(period, min_periods=period).sum(), atr)
    dx = 100.0 * _safe_div((plus_di - minus_di).abs(), plus_di + minus_di)
    return dx.rolling(period, min_periods=period).mean()


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> Tuple[pd.Series, pd.Series]:
    lowest = low.rolling(period, min_periods=period).min()
    highest = high.rolling(period, min_periods=period).max()
    k = 100.0 * _safe_div(close - lowest, highest - lowest)
    d = k.rolling(3, min_periods=3).mean()
    return k, d


def _cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    typical = (high + low + close) / 3.0
    sma = typical.rolling(period, min_periods=period).mean()
    mad = (typical - sma).abs().rolling(period, min_periods=period).mean()
    return _safe_div(typical - sma, 0.015 * mad)


def _williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    highest = high.rolling(period, min_periods=period).max()
    lowest = low.rolling(period, min_periods=period).min()
    return -100.0 * _safe_div(highest - close, highest - lowest)


def _mfi(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14) -> pd.Series:
    tp = (high + low + close) / 3.0
    raw_mf = tp * volume
    direction = np.sign(tp.diff()).fillna(0.0)
    pos_flow = raw_mf.where(direction > 0, 0.0).rolling(period, min_periods=period).sum()
    neg_flow = raw_mf.where(direction < 0, 0.0).rolling(period, min_periods=period).sum()
    money_ratio = _safe_div(pos_flow, neg_flow)
    return 100.0 - (100.0 / (1.0 + money_ratio))


def _tsi(close: pd.Series, long: int = 25, short: int = 13) -> pd.Series:
    mom = close.diff()
    ema1 = _ema(mom, long)
    ema2 = _ema(ema1, short)
    abs_ema1 = _ema(mom.abs(), long)
    abs_ema2 = _ema(abs_ema1, short)
    return 100.0 * _safe_div(ema2, abs_ema2)


def _obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    direction = np.sign(close.diff()).fillna(0.0)
    return (direction * volume).cumsum()


def _ad_line(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    mfm = _safe_div((close - low) - (high - close), high - low)
    mfv = mfm * volume
    return mfv.cumsum()


def get_sentiment_feature_columns() -> List[str]:
    """Sentiment features are intentionally disabled in the rewritten pipeline."""
    return SENTIMENT_FEATURE_COLUMNS.copy()


def get_feature_columns(include_sentiment: bool = True) -> List[str]:
    cols = TECHNICAL_FEATURE_COLUMNS.copy()
    if include_sentiment and SENTIMENT_FEATURE_COLUMNS:
        cols.extend(SENTIMENT_FEATURE_COLUMNS)
    return cols


def _assert_required_columns(df: pd.DataFrame) -> None:
    missing = [c for c in BASE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required OHLCV columns: {missing}")


def engineer_features(
    df: pd.DataFrame,
    symbol: str | None = None,
    include_sentiment: bool = False,
    cache_manager: object | None = None,
) -> pd.DataFrame:
    """Create deterministic technical features.

    The function does not fetch external news/sentiment and is safe for both
    training and inference paths.
    """
    _assert_required_columns(df)

    work = df.copy()
    for col in BASE_COLUMNS:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    open_ = work["Open"]
    high = work["High"]
    low = work["Low"]
    close = work["Close"]
    volume = work["Volume"].clip(lower=0.0)

    ret_1d = close.pct_change()
    log_ret_1d = np.log(_safe_div(close, close.shift(1)))

    tr = _true_range(high, low, close)
    atr_14 = tr.rolling(14, min_periods=14).mean()

    gk_daily = np.sqrt(
        (0.5 * (np.log(_safe_div(high, low)) ** 2))
        - ((2.0 * np.log(2.0) - 1.0) * (np.log(_safe_div(close, open_)) ** 2))
    )
    parkinson_daily = np.sqrt((1.0 / (4.0 * np.log(2.0))) * (np.log(_safe_div(high, low)) ** 2))

    ema_12 = _ema(close, 12)
    ema_26 = _ema(close, 26)
    macd = ema_12 - ema_26
    macd_signal = _ema(macd, 9)
    macd_hist = macd - macd_signal

    stoch_k, stoch_d = _stochastic(high, low, close, 14)

    sma_10 = close.rolling(10, min_periods=10).mean()
    sma_20 = close.rolling(20, min_periods=20).mean()
    sma_50 = close.rolling(50, min_periods=50).mean()

    bb_mid = sma_20
    bb_std = close.rolling(20, min_periods=20).std()
    bb_upper = bb_mid + (2.0 * bb_std)
    bb_lower = bb_mid - (2.0 * bb_std)

    atr_20 = tr.rolling(20, min_periods=20).mean()
    kc_mid = _ema(close, 20)
    kc_upper = kc_mid + (2.0 * atr_20)
    kc_lower = kc_mid - (2.0 * atr_20)

    obv = _obv(close, volume)
    vpt = (ret_1d.fillna(0.0) * volume).cumsum()
    ad_line = _ad_line(high, low, close, volume)

    typical_price = (high + low + close) / 3.0
    vwap = _safe_div((typical_price * volume).cumsum(), volume.cumsum())

    vol_20 = ret_1d.rolling(20, min_periods=20).std()
    vol_60 = ret_1d.rolling(60, min_periods=60).std()

    vol_q_low = vol_20.rolling(252, min_periods=80).quantile(0.33)
    vol_q_high = vol_20.rolling(252, min_periods=80).quantile(0.67)

    regime_low = (vol_20 < vol_q_low).astype(float)
    regime_high = (vol_20 > vol_q_high).astype(float)
    regime_mid = (1.0 - regime_low - regime_high).clip(lower=0.0)

    trend_dir = np.sign(_ema(close, 20) - _ema(close, 50))
    trend_persistence = np.sign(ret_1d).rolling(10, min_periods=5).mean()

    rv_iv_spread = vol_20 - vol_60
    rv_iv_spread_zscore = _safe_div(
        rv_iv_spread - rv_iv_spread.rolling(120, min_periods=40).mean(),
        rv_iv_spread.rolling(120, min_periods=40).std(),
    )

    feat = pd.DataFrame(index=work.index)
    feat["ret_1d"] = ret_1d
    feat["log_ret_1d"] = log_ret_1d
    feat["vol_5d"] = ret_1d.rolling(5, min_periods=5).std()
    feat["vol_10d"] = ret_1d.rolling(10, min_periods=10).std()
    feat["vol_20d"] = vol_20
    feat["vol_ratio_5_20"] = _safe_div(feat["vol_5d"], feat["vol_20d"])
    feat["atr_14"] = atr_14
    feat["atr_pct_14"] = _safe_div(atr_14, close)
    feat["garman_klass_10"] = gk_daily.rolling(10, min_periods=10).mean()
    feat["parkinson_10"] = parkinson_daily.rolling(10, min_periods=10).mean()

    feat["rsi_7"] = _rsi(close, 7)
    feat["rsi_14"] = _rsi(close, 14)
    feat["rsi_21"] = _rsi(close, 21)
    feat["stoch_k_14"] = stoch_k
    feat["stoch_d_3"] = stoch_d
    feat["williams_r_14"] = _williams_r(high, low, close, 14)
    feat["cci_20"] = _cci(high, low, close, 20)
    feat["roc_5"] = close.pct_change(5)
    feat["roc_10"] = close.pct_change(10)
    feat["roc_20"] = close.pct_change(20)
    feat["tsi_25_13"] = _tsi(close, 25, 13)

    feat["sma_10_dist"] = _safe_div(close - sma_10, sma_10)
    feat["sma_20_dist"] = _safe_div(close - sma_20, sma_20)
    feat["sma_50_dist"] = _safe_div(close - sma_50, sma_50)
    feat["ema_12_dist"] = _safe_div(close - ema_12, ema_12)
    feat["ema_26_dist"] = _safe_div(close - ema_26, ema_26)
    feat["macd"] = macd
    feat["macd_signal"] = macd_signal
    feat["macd_hist"] = macd_hist
    feat["adx_14"] = _adx(high, low, close, 14)
    feat["trend_strength"] = _safe_div((ema_12 - ema_26).abs(), close)

    feat["bb_upper_dist"] = _safe_div(close - bb_upper, bb_upper)
    feat["bb_lower_dist"] = _safe_div(close - bb_lower, bb_lower)
    feat["bb_width"] = _safe_div(bb_upper - bb_lower, bb_mid)
    feat["bb_pct_b"] = _safe_div(close - bb_lower, bb_upper - bb_lower)
    feat["kc_upper_dist"] = _safe_div(close - kc_upper, kc_upper)
    feat["kc_lower_dist"] = _safe_div(close - kc_lower, kc_lower)

    feat["vol_sma_20_ratio"] = _safe_div(volume, volume.rolling(20, min_periods=20).mean())
    feat["obv"] = obv
    feat["obv_slope_5"] = _safe_div(obv.diff(5), volume.rolling(20, min_periods=20).mean())
    feat["vwap_dist"] = _safe_div(close - vwap, vwap)
    feat["mfi_14"] = _mfi(high, low, close, volume, 14)
    feat["vpt"] = vpt
    feat["ad_line"] = ad_line

    feat["velocity_5"] = close.pct_change(5)
    feat["velocity_10"] = close.pct_change(10)
    feat["velocity_20"] = close.pct_change(20)
    feat["accel_5"] = feat["velocity_5"].diff(5)
    feat["rsi_velocity_5"] = feat["rsi_14"].diff(5)
    feat["volume_velocity_5"] = volume.pct_change(5)
    feat["momentum_10_20"] = feat["velocity_10"] - feat["velocity_20"]
    feat["price_accel_10"] = _safe_div(close.diff(10).diff(10), close.shift(20))

    feat["higher_high"] = (high > high.shift(1)).astype(float)
    feat["lower_low"] = (low < low.shift(1)).astype(float)
    feat["gap_up"] = (open_ > high.shift(1)).astype(float)
    feat["gap_down"] = (open_ < low.shift(1)).astype(float)
    feat["body_size"] = _safe_div((close - open_).abs(), open_)
    feat["inside_bar"] = ((high < high.shift(1)) & (low > low.shift(1))).astype(float)

    feat["regime_low_vol"] = regime_low
    feat["regime_mid_vol"] = regime_mid
    feat["regime_high_vol"] = regime_high
    feat["trend_dir"] = trend_dir
    feat["mean_reversion_z"] = _safe_div(close - close.rolling(20, min_periods=20).mean(), close.rolling(20, min_periods=20).std())
    feat["trend_persistence"] = trend_persistence

    feat["hl_range"] = _safe_div(high - low, close)
    feat["close_location"] = _safe_div(close - low, high - low)
    feat["intraday_vol"] = _safe_div((close - open_).abs(), open_)
    feat["overnight_ret"] = _safe_div(open_ - close.shift(1), close.shift(1))

    feat["vix_proxy_level"] = vol_20 * np.sqrt(252.0) * 100.0
    feat["vix_proxy_change"] = feat["vix_proxy_level"].pct_change(5)
    feat["rv_iv_spread"] = rv_iv_spread
    feat["rv_iv_spread_zscore"] = rv_iv_spread_zscore

    # Ensure exact canonical features.
    for col in TECHNICAL_FEATURE_COLUMNS:
        if col not in feat.columns:
            feat[col] = 0.0

    feat = feat[TECHNICAL_FEATURE_COLUMNS]
    feat = feat.replace([np.inf, -np.inf], np.nan)
    feat = feat.ffill(limit=5)
    feat = feat.fillna(0.0)
    feat = feat.clip(lower=-20.0, upper=20.0)

    if include_sentiment and SENTIMENT_FEATURE_COLUMNS:
        for col in SENTIMENT_FEATURE_COLUMNS:
            feat[col] = 0.0

    out = pd.concat([work[BASE_COLUMNS], feat], axis=1)
    validate_feature_count(out)

    logger.info(
        "Engineered features for %s: rows=%d, features=%d",
        symbol or "UNKNOWN",
        len(out),
        len(TECHNICAL_FEATURE_COLUMNS),
    )
    return out


def validate_feature_count(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c not in BASE_COLUMNS]
    expected = get_feature_columns(include_sentiment=False)

    missing = [c for c in expected if c not in feature_cols]
    if missing:
        raise AssertionError(f"Missing canonical features: {missing}")

    extra = [c for c in feature_cols if c not in expected]
    if extra:
        logger.warning("Dropping %d non-canonical feature columns", len(extra))
        df = df.drop(columns=extra)

    if len([c for c in df.columns if c not in BASE_COLUMNS]) != EXPECTED_FEATURE_COUNT:
        raise AssertionError(
            f"Feature count mismatch. expected={EXPECTED_FEATURE_COUNT}, "
            f"actual={len([c for c in df.columns if c not in BASE_COLUMNS])}"
        )
    return df


def validate_and_fix_features(df: pd.DataFrame) -> pd.DataFrame:
    fixed = df.copy()
    expected = get_feature_columns(include_sentiment=False)
    for col in expected:
        if col not in fixed.columns:
            fixed[col] = 0.0
    keep_cols = [c for c in BASE_COLUMNS if c in fixed.columns] + expected
    fixed = fixed.reindex(columns=keep_cols)
    return validate_feature_count(fixed)


def create_sequences_safe(data: np.ndarray, sequence_length: int = 60) -> np.ndarray:
    if data.ndim != 2:
        raise ValueError("Expected 2D array for sequence creation")
    if len(data) < sequence_length:
        return np.empty((0, sequence_length, data.shape[1]), dtype=np.float32)
    return np.stack([data[i : i + sequence_length] for i in range(len(data) - sequence_length + 1)]).astype(np.float32)
