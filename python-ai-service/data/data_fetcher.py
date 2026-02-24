import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import logging

# Setup logging
logger = logging.getLogger(__name__)

def fetch_stock_data(
    symbol: str, 
    period: str = "max",
    min_required_days: int = 500,
    interval: str = "1d",
    start: datetime | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    """
    Fetch historical stock data with minimum data requirement.
    
    For training with 90-day sequences, we need:
    - 90 days for sequence window
    - 50 days for indicators (SMA_50, MACD, etc.)
    - 365 days for meaningful training/validation split
    - Total minimum: 500 days recommended
    
    Args:
        symbol: Stock ticker (e.g., 'AAPL')
        period: Time period ('1y', '2y', '5y', 'max')
        min_required_days: Minimum days of data required for training
    
    Returns:
        DataFrame with OHLCV data
        
    Raises:
        ValueError: If insufficient data available
    """
    from datetime import datetime, timedelta
    import time
    
    logger.info(f"Fetching data for {symbol} (period={period}, interval={interval})")
    
    # Fetch data with retries and fallback periods
    df = pd.DataFrame()
    max_retries = 3
    
    for retry in range(max_retries):
        # Create fresh ticker object each retry
        ticker = yf.Ticker(symbol)
        
        # Try period-based first unless explicit dates are provided.
        if start is not None or end is not None:
            df = ticker.history(start=start, end=end, interval=interval)
        else:
            df = ticker.history(period=period, interval=interval)
        
        if not df.empty:
            break
            
        # If period fails, try date-based fallback
        logger.warning(f"period='{period}' failed for {symbol} (retry {retry+1}/{max_retries}), trying date-based fallback...")
        
        # Use explicit start/end dates (more reliable)
        end_date = datetime.now()
        
        # Map period to years
        period_years = {'max': 20, '10y': 10, '5y': 5, '2y': 2, '1y': 1, '730d': 2, '365d': 1, '180d': 1}
        years = period_years.get(period, 5)
        
        for try_years in [years, 10, 5, 2]:
            start_date = end_date - timedelta(days=365 * try_years)
            logger.info(f"Trying date range: {start_date.date()} to {end_date.date()}...")
            df = ticker.history(start=start_date, end=end_date, interval=interval)
            if not df.empty:
                logger.info(f"Successfully fetched {len(df)} days with {try_years}y date range")
                break
        
        if not df.empty:
            break
            
        # Wait before retry
        if retry < max_retries - 1:
            time.sleep(2)
    
    if df.empty:
        raise ValueError(f"No data retrieved for {symbol} after {max_retries} retries")
    
    # Clean data
    df = df.dropna()
    df.index = pd.to_datetime(df.index)
    
    # Check if we have enough data
    if len(df) < min_required_days:
        logger.warning(
            f"Insufficient data: {len(df)} days. "
            f"Minimum recommended: {min_required_days} days. "
            f"Training may not work properly."
        )
        
        # Try fetching longer period if not already 'max'
        if period != 'max':
            fallback_period = "max" if interval in {"1d", "1wk", "1mo"} else "730d"
            logger.info("Attempting fallback period '%s'...", fallback_period)
            df = ticker.history(period=fallback_period, interval=interval)
            df = df.dropna()
            df.index = pd.to_datetime(df.index)
            
            if len(df) < min_required_days:
                raise ValueError(
                    f"Insufficient historical data for {symbol}. "
                    f"Got {len(df)} days, need at least {min_required_days}. "
                    f"This stock may be too new for this strategy."
                )
    
    logger.info(f"✓ Fetched {len(df)} days of data for {symbol}")
    
    return df


def validate_training_data_sufficiency(
    df: pd.DataFrame, 
    sequence_length: int = 90
) -> dict:
    """
    Check if we have enough data for training.
    
    Calculates expected sample counts after feature engineering,
    sequence creation, and train/test splitting.
    
    Args:
        df: Raw stock DataFrame
        sequence_length: Length of input sequences (default 90 for regressor)
    
    Returns:
        Dict with sufficiency metrics and recommendations
    """
    total_rows = len(df)
    
    # Calculate how many samples we can create
    # Need: sequence_length days + indicators warm-up (50 days for SMA_50)
    min_required = sequence_length + 50
    
    if total_rows < min_required:
        return {
            'sufficient': False,
            'total_rows': total_rows,
            'min_required': min_required,
            'shortfall': min_required - total_rows,
            'message': f"Need {min_required - total_rows} more days of data"
        }
    
    # After feature engineering, we lose rows to NaN
    # Typically lose ~50 rows to indicator warm-up (SMA_50, MACD, etc.)
    usable_rows = total_rows - 50
    
    # After creating sequences (each sequence needs sequence_length + 1 rows for target)
    num_sequences = max(0, usable_rows - sequence_length)
    
    # After train/test split (80/20)
    train_samples = int(num_sequences * 0.8)
    val_samples = num_sequences - train_samples
    
    # Minimum 100 training samples for reliable model
    sufficient = train_samples >= 100
    
    return {
        'sufficient': sufficient,
        'total_rows': total_rows,
        'usable_rows': usable_rows,
        'num_sequences': num_sequences,
        'train_samples': train_samples,
        'val_samples': val_samples,
        'recommendation': 'PASS' if sufficient else 
                        f'WARNING: Only {train_samples} training samples. Recommend ≥100.'
    }

def get_realtime_price(symbol: str) -> dict:
    """Get current price and basic info"""
    ticker = yf.Ticker(symbol)
    info = ticker.info
    
    return {
        "symbol": symbol,
        "current_price": info.get("currentPrice"),
        "previous_close": info.get("previousClose"),
        "open": info.get("open"),
        "day_high": info.get("dayHigh"),
        "day_low": info.get("dayLow"),
        "volume": info.get("volume")
    }
