"""
Volatility spread features: realized vs implied (RV - IV) using VIX as market proxy.

Functions:
- fetch_vix_data(start_date, end_date) -> pd.DataFrame
- compute_realized_volatility(df, price_col='Close', window=20, annualize=True) -> pd.Series
- compute_rv_iv_spread(df, realized_vol_col='realized_vol_20d', implied_vol_col='vix', zscore_window=20) -> dict
- add_volatility_spread_features(df_stock, symbol='AAPL') -> pd.DataFrame

Interpretation guide:
- Spread > 0.2: Realized vol exceeds implied vol -> market underpricing risk
- Spread < -0.2: Implied vol exceeds realized vol -> market overpricing risk
- Spread ≈ 0: Fair pricing

Cross-sectional notes: compare stock spread to SPY spread to identify idiosyncratic vs systematic risk.

Edge cases handled:
- Forward-fill VIX for weekends/holidays
- Treat VIX values > 1 as percentage (divide by 100)
- Avoid division-by-zero with small epsilon
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime


def _to_date_str(d):
    if isinstance(d, str):
        return d
    if isinstance(d, datetime):
        return d.strftime("%Y-%m-%d")
    # If pandas Timestamp
    try:
        return pd.to_datetime(d).strftime("%Y-%m-%d")
    except Exception:
        return str(d)


def fetch_vix_data(start_date, end_date) -> pd.DataFrame:
    """
    Fetch VIX (CBOE Volatility Index) daily close using yfinance.

    Returns a DataFrame indexed by date with a single column `vix` containing
    implied volatility in decimal form (e.g., 0.20 for 20%).
    """
    start_str = _to_date_str(start_date)
    end_str = _to_date_str(end_date)

    vix_ticker = "^VIX"
    try:
        # Explicitly set auto_adjust=False to preserve explicit Close/Adj Close columns
        raw = yf.download(vix_ticker, start=start_str, end=end_str, progress=False, auto_adjust=False)
    except Exception as e:
        raise RuntimeError(f"Failed to download VIX data: {e}")

    # yfinance may sometimes return a Series, an empty object, or a DataFrame with
    # multi-index columns. Normalize to a DataFrame and make column-name checks
    # robust to avoid ambiguous truth-value checks (e.g., `if raw:` on a Series).
    if isinstance(raw, pd.Series):
        # single-row/series -> convert to single-row DataFrame
        raw = raw.to_frame().T

    if raw is None or (hasattr(raw, 'empty') and raw.empty):
        raise RuntimeError("No VIX data returned by yfinance")

    # Flatten multi-index columns if present
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = ["_".join([str(c) for c in col]).strip() for col in raw.columns]

    # Normalize column names to find Close/Adj Close variants. If those aren't
    # present, fall back to the first numeric column returned by yfinance.
    cols_lower = {c.lower(): c for c in raw.columns}
    close_col = None
    if 'close' in cols_lower:
        close_col = cols_lower['close']
    elif 'adj close' in cols_lower:
        close_col = cols_lower['adj close']
    else:
        # fallback: find first column containing 'close' (case-insensitive)
        for c in raw.columns:
            if 'close' in str(c).lower():
                close_col = c
                break

    # If still not found, pick the first numeric column as a last-resort fallback
    if close_col is None:
        numeric_cols = raw.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            close_col = numeric_cols[0]

    if close_col is None:
        # Provide helpful debug info for downstream diagnosis
        col_info = ", ".join([f"{c}({str(raw[c].dtype)})" for c in raw.columns])
        raise RuntimeError(f"Unexpected VIX data format (no 'Close' column). Columns returned: {col_info}")

    # Use the detected close column as VIX value
    df = raw[[close_col]].copy()
    df.index = pd.to_datetime(df.index)
    df.index = df.index.tz_localize(None)
    df.rename(columns={close_col: 'vix_raw'}, inplace=True)

    # Defensive: if values look like percentages (e.g., 20), convert to decimals
    median_val = df['vix_raw'].median(skipna=True)
    if pd.notna(median_val) and median_val > 2.0:
        df['vix'] = df['vix_raw'] / 100.0
    else:
        df['vix'] = df['vix_raw']

    # Forward-fill to cover non-trading days later when merged
    df['vix'] = df['vix'].ffill()

    # Keep only the vix column
    df = df[['vix']]

    return df


def compute_realized_volatility(df: pd.DataFrame, price_col: str = 'Close', window: int = 20, annualize: bool = True) -> pd.Series:
    """
    Compute realized volatility as rolling std of log returns.

    Returns a Series with values in decimal form (e.g., 0.25 for 25%).
    """
    if price_col not in df.columns:
        raise ValueError(f"Price column '{price_col}' not found in DataFrame")

    price = df[price_col].astype(float)
    # Log returns
    log_ret = np.log(price).diff()

    rv = log_ret.rolling(window=window, min_periods=1).std()

    if annualize:
        rv = rv * np.sqrt(252.0)

    # Replace inf/nans gracefully; initial periods with too few data points will be NaN
    rv = rv.replace([np.inf, -np.inf], np.nan)

    return rv


def compute_rv_iv_spread(df: pd.DataFrame, realized_vol_col: str = 'realized_vol_20d', implied_vol_col: str = 'vix', zscore_window: int = 20) -> dict:
    """
    Compute RV-IV spread and regime classification.

    Spread = (RV - IV) / (IV)

    Returns dict of Series:
      - 'rv_iv_spread'
      - 'rv_iv_spread_zscore'
      - 'iv_regime_underpriced'
      - 'iv_regime_overpriced'
      - 'iv_regime_fair'
    """
    if realized_vol_col not in df.columns:
        raise ValueError(f"Realized vol column '{realized_vol_col}' not found")
    if implied_vol_col not in df.columns:
        raise ValueError(f"Implied vol column '{implied_vol_col}' not found")

    rv = df[realized_vol_col].astype(float)
    iv = df[implied_vol_col].astype(float)

    # Defensive: if IV values look like percent integers (e.g., 20), convert to decimal
    median_iv = iv.median(skipna=True)
    if pd.notna(median_iv) and median_iv > 2.0:
        iv = iv / 100.0

    eps = 1e-8
    denom = iv.replace(0.0, eps)

    spread = (rv - iv) / denom

    # Rolling z-score normalization of spread
    spread_mean = spread.rolling(window=zscore_window, min_periods=1).mean()
    spread_std = spread.rolling(window=zscore_window, min_periods=1).std().replace(0.0, 1e-8)
    spread_z = (spread - spread_mean) / spread_std

    # Regime classification thresholds (20% as requested)
    underpriced = (spread > 0.20).astype(int)
    overpriced = (spread < -0.20).astype(int)
    fair = ((spread >= -0.20) & (spread <= 0.20)).astype(int)

    return {
        'rv_iv_spread': spread,
        'rv_iv_spread_zscore': spread_z.fillna(0.0),
        'iv_regime_underpriced': underpriced,
        'iv_regime_overpriced': overpriced,
        'iv_regime_fair': fair
    }


def add_volatility_spread_features(df_stock: pd.DataFrame, symbol: str = 'AAPL') -> pd.DataFrame:
    """
    Orchestrator that fetches VIX, computes RV, then RV-IV spread and attaches columns to stock df.

    Returns augmented DataFrame with new columns:
      - realized_vol_20d
      - vix
      - rv_iv_spread
      - rv_iv_spread_zscore
      - iv_regime_underpriced/overpriced/fair
    """
    print(f"[INFO] Adding volatility spread features for {symbol}...")

    # Get date range from stock data
    if isinstance(df_stock.index, pd.DatetimeIndex):
        start_date = df_stock.index.min()
        end_date = df_stock.index.max()
    else:
        # Expect 'Date' column
        start_date = df_stock['Date'].min()
        end_date = df_stock['Date'].max()

    # Step 1: Fetch VIX
    try:
        vix_df = fetch_vix_data(start_date, end_date)
        print(f"✓ Fetched VIX data: {len(vix_df)} days")
    except Exception as e:
        print(f"[WARN] Could not fetch VIX data: {e}")
        print("[WARN] Skipping volatility spread features")
        return df_stock

    # Step 2: Merge VIX with stock data (align on Date index)
    # Ensure both indices are datetime and timezone-naive
    df_work = df_stock.copy()
    if not isinstance(df_work.index, pd.DatetimeIndex):
        df_work = df_work.set_index(pd.to_datetime(df_work['Date']))
    df_work.index = pd.to_datetime(df_work.index).tz_localize(None)

    vix_df.index = pd.to_datetime(vix_df.index).tz_localize(None)

    merged = df_work.merge(vix_df, left_index=True, right_index=True, how='left')
    merged['vix'] = merged['vix'].ffill()

    # Step 3: Compute realized volatility (20-day)
    merged['realized_vol_20d'] = compute_realized_volatility(merged, price_col='Close', window=20, annualize=True)

    # Step 4: Compute spread features
    spreads = compute_rv_iv_spread(merged, 'realized_vol_20d', 'vix')
    for k, v in spreads.items():
        merged[k] = v

    # Final: ensure types and return with original indexing/columns shape
    # If original df_stock had a Date column and non-datetime index, try to restore
    try:
        # If original had non-datetime index, restore to that format
        if 'Date' in df_stock.columns and not isinstance(df_stock.index, pd.DatetimeIndex):
            merged = merged.reset_index(drop=False)
            # rename index column back to Date
            merged.rename(columns={'index': 'Date'}, inplace=True)
            # Reorder columns to put Date first
            cols = merged.columns.tolist()
            if 'Date' in cols:
                cols = ['Date'] + [c for c in cols if c != 'Date']
                merged = merged[cols]
    except Exception:
        pass

    print(f"✓ Added volatility spread features (latest spread: {merged['rv_iv_spread'].iloc[-1]:.2%})")

    return merged
