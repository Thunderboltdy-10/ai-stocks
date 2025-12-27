import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_pivot_points(df: pd.DataFrame, window: int = 5) -> pd.DataFrame:
    """
    Identify Williams Fractals (pivots) with no look-ahead bias.
    
    A fractal is confirmed after 'window' samples. 
    At index i, we check if index (i - window) was a local extreme.
    
    Args:
        df: DataFrame with OHLC data
        window: Window size for fractal detection (default 5)
        
    Returns:
        DataFrame with latest confirmed high/low pivots and distance to them.
    """
    res = pd.DataFrame(index=df.index)
    
    # Initialize latest pivots
    latest_high_pivot = np.nan
    latest_low_pivot = np.nan
    
    highs = df['High'].values
    lows = df['Low'].values
    
    high_pivots = np.full(len(df), np.nan)
    low_pivots = np.full(len(df), np.nan)
    
    # Shift window to identify confirmed peaks at current index i
    # i-window is the candidate. 
    # We check its neighbors in [i-2*window, i]
    side = window // 2
    
    for i in range(window, len(df)):
        # Check for High Fractal at (i - side)
        idx = i - side
        is_high_pivot = True
        for k in range(1, side + 1):
            if highs[idx] < highs[idx-k] or highs[idx] < highs[idx+k]:
                is_high_pivot = False
                break
        
        if is_high_pivot:
            latest_high_pivot = highs[idx]
            
        # Check for Low Fractal at (i - side)
        is_low_pivot = True
        for k in range(1, side + 1):
            if lows[idx] > lows[idx-k] or lows[idx] > lows[idx+k]:
                is_low_pivot = False
                break
        
        if is_low_pivot:
            latest_low_pivot = lows[idx]
            
        high_pivots[i] = latest_high_pivot
        low_pivots[i] = latest_low_pivot
        
    res['latest_high_pivot'] = high_pivots
    res['latest_low_pivot'] = low_pivots
    
    # Normalized distance to pivots
    close = df['Close'].values
    res['dist_to_high_pivot'] = (res['latest_high_pivot'] - close) / close
    res['dist_to_low_pivot'] = (close - res['latest_low_pivot']) / close
    
    # Fill NaNs at start
    res = res.ffill().fillna(0)
    
    return res

def calculate_volume_profile_features(df: pd.DataFrame, lookback: int = 60, num_bins: int = 30) -> pd.DataFrame:
    """
    Calculate Volume-at-Price (VAP) features.
    
    Identifies high volume nodes (HVN) over a rolling window.
    Features: distance to Point of Control (POC) and HVN density.
    
    Args:
        df: DataFrame with OHLCV data
        lookback: Rolling window for profile calculation
        num_bins: Number of price bins
        
    Returns:
        DataFrame with VAP features
    """
    res = pd.DataFrame(index=df.index)
    poc_price = np.full(len(df), np.nan)
    hvn_proximity = np.zeros(len(df))
    
    # This is compute intensive if done for every row. 
    # We optimize by using a larger step or a faster algorithm.
    # For now, let's do a sliding window.
    
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    volumes = df['Volume'].values
    avg_prices = (highs + lows + closes) / 3
    
    # Pre-calculate bin edges for each window (assuming fixed num_bins)
    # Actually, min/max changes per window, so we still need some loop.
    # But we can speed up the histogram part.
    
    for i in range(lookback, len(df)):
        # Use numpy slicing instead of iloc
        h_win = highs[i-lookback:i]
        l_win = lows[i-lookback:i]
        ap_win = avg_prices[i-lookback:i]
        v_win = volumes[i-lookback:i]
        
        price_min = l_win.min()
        price_max = h_win.max()
        
        if price_max == price_min:
            continue
            
        bins = np.linspace(price_min, price_max, num_bins + 1)
        
        # Aggregate volume by bin
        hist, _ = np.histogram(ap_win, bins=bins, weights=v_win)
        
        # Point of Control (POC) - bin with highest volume
        max_bin_idx = np.argmax(hist)
        poc = (bins[max_bin_idx] + bins[max_bin_idx + 1]) / 2
        poc_price[i] = poc
        
        # Proximity to any High Volume Node (above average volume bin)
        avg_bin_vol = np.mean(hist)
        threshold = avg_bin_vol * 1.5
        
        mask = hist > threshold
        if np.any(mask):
            high_vol_bins = bins[:-1][mask]
            bin_centers = high_vol_bins + (bins[1]-bins[0])/2
            
            curr_close = closes[i]
            # Find nearest high vol bin center
            dist_idx = np.argmin(np.abs(bin_centers - curr_close))
            nearest_hvn = bin_centers[dist_idx]
            hvn_proximity[i] = (nearest_hvn - curr_close) / curr_close

    res['vap_poc_price'] = poc_price
    res['dist_to_vap_poc'] = (res['vap_poc_price'] - df['Close']) / df['Close']
    res['hvn_proximity'] = hvn_proximity
    
    res = res.ffill().fillna(0)
    return res

def calculate_dynamic_sr_zones(df: pd.DataFrame, pivots_df: pd.DataFrame, vap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine pivots and volume nodes into dynamic S/R zones.
    
    Args:
        df: Original OHLCV
        pivots_df: Output from calculate_pivot_points
        vap_df: Output from calculate_volume_profile_features
        
    Returns:
        DataFrame with zone features
    """
    res = pd.DataFrame(index=df.index)
    
    # Zone proximity: price within 1% of a major level
    ZONE_THRESHOLD = 0.015
    
    # Resistance zone (pivots and POC above)
    res['near_resistance_zone'] = (
        (pivots_df['dist_to_high_pivot'].abs() < ZONE_THRESHOLD) | 
        (vap_df['dist_to_vap_poc'].abs() < ZONE_THRESHOLD)
    ).astype(int)
    
    # Support zone (pivots and POC below)
    res['near_support_zone'] = (
        (pivots_df['dist_to_low_pivot'].abs() < ZONE_THRESHOLD) | 
        (vap_df['dist_to_vap_poc'].abs() < ZONE_THRESHOLD)
    ).astype(int)
    
    # Volume weighted relative position in range
    # 1.0 = near POC, 0.0 = far from volume concentration
    res['volume_concentration_score'] = 1.0 / (1.0 + vap_df['dist_to_vap_poc'].abs() * 100)
    
    return res
