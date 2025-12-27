"""
Target engineering for forward returns with proper masking.
No data leakage - uses only past information.
"""
import numpy as np
import pandas as pd
from typing import Tuple, List
from pathlib import Path
from sklearn.preprocessing import RobustScaler, StandardScaler
import matplotlib.pyplot as plt
from scipy import stats


def create_forward_returns(
    df: pd.DataFrame,
    horizons: List[int] = [3, 5, 10],
    price_col: str = 'Close'
) -> pd.DataFrame:
    """
    Create forward log-returns for multiple horizons.
    
    Args:
        df: DataFrame with OHLCV data (indexed by date)
        horizons: List of forward-looking days [3, 5, 10]
        price_col: Column name for price
    
    Returns:
        DataFrame with columns: target_3d, target_5d, target_10d
        NaN for rows where future data unavailable
    """
    df = df.copy()
    
    for h in horizons:
        # Forward return: log(price_t+h / price_t)
        # .shift(-h) moves future price to current row
        future_price = df[price_col].shift(-h)
        current_price = df[price_col]
        
        # Compute log return
        log_return = np.log(future_price / current_price)
        
        # Store with explicit NaN for unavailable future
        df[f'target_{h}d'] = log_return
    
    # Sanity check: last H rows should be NaN
    for h in horizons:
        assert df[f'target_{h}d'].iloc[-h:].isna().all(), \
            f"Last {h} rows of target_{h}d must be NaN"
    
    return df


def calculate_smart_target(
    df: pd.DataFrame,
    horizon: int = 1,
    vol_window: int = 20,
    price_col: str = 'Close'
) -> pd.DataFrame:
    """
    Calculate high-precision 'Smart Target' (Volatility-Normalized Returns).
    
    Based on Kelly-aware position sizing, this target focuses on Sharpe-like 
    efficiency rather than raw magnitude. It penalizes returns during high-volatility 
    regimes to encourage more stable model learning.
    
    Formula: target_smart = log_return_1d / rolling_volatility_20d
    
    Args:
        df: DataFrame with OHLCV data
        horizon: Prediction horizon (default 1d)
        vol_window: Window for rolling volatility calculation (default 20d)
        price_col: Column name for price
        
    Returns:
        DataFrame with 'target_smart' column
    """
    df = df.copy()
    
    # 1. Calculate raw forward log returns
    future_price = df[price_col].shift(-horizon)
    current_price = df[price_col]
    log_return = np.log(future_price / current_price)
    
    # 2. Calculate historical rolling volatility (standard deviation of daily log returns)
    daily_returns = np.log(df[price_col] / df[price_col].shift(1))
    rolling_vol = daily_returns.rolling(window=vol_window).std()
    
    # 3. Handle zero/NaN volatility (use baseline or forward fill)
    rolling_vol = rolling_vol.replace(0, np.nan).fillna(method='ffill')
    
    # 4. Create Smart Target: Volatility-Normalized Return
    # This creates a target with more consistent variance across different regimes
    df['target_smart'] = log_return / rolling_vol
    
    # 5. Clip extreme outliers in the normalized target to prevent gradient explosion
    # Normalized returns follow a distribution closer to normal but still have tails
    df['target_smart'] = np.clip(df['target_smart'], -5.0, 5.0)
    
    return df


def prevent_lookahead_bias(
    df: pd.DataFrame,
    feature_cols: List[str] = None
) -> pd.DataFrame:
    """
    Shift all features by 1 to ensure they use only past information.
    
    Critical: At time t, we can only use data up to t-1 (previous close).
    Features computed at t (using t's OHLCV) would leak information.
    
    Args:
        df: DataFrame with features
        feature_cols: List of feature column names to shift. If None, auto-detect.
    
    Returns:
        DataFrame with features shifted forward by 1
    """
    df = df.copy()
    
    # Identify feature columns if not provided
    if feature_cols is None:
        exclude_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 
                        'Stock Splits', 'Date', 'forward_return', 'label', 'log_forward_return']
        exclude_cols += [c for c in df.columns if c.startswith('target_')]
        feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Shift features forward by 1 (use yesterday's features for today's prediction)
    for col in feature_cols:
        if col in df.columns:
            df[col] = df[col].shift(1)
    
    # First row now has NaN features (no past data) - this is correct
    if len(feature_cols) > 0 and len(df) > 0:
        first_row_features = df[feature_cols].iloc[0]
        assert first_row_features.isna().all(), \
            f"First row features must be NaN (no past data available). Got: {first_row_features.head()}"
    
    return df


def prepare_training_data(
    df: pd.DataFrame,
    horizons: List[int] = [3, 5, 10],
    use_smart_target: bool = False
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Complete pipeline: create targets, prevent leakage, drop NaN.
    
    Args:
        df: DataFrame with OHLCV data
        horizons: List of horizons for standard returns
        use_smart_target: Whether to also calculate and scale the Smart Target
        
    Returns:
        (clean_df, feature_columns)
    """
    # Create forward targets
    df = create_forward_returns(df, horizons)
    
    if use_smart_target:
        df = calculate_smart_target(df, horizon=1)
        print("   [OK] Smart Target (target_smart) integrated")
    
    # Shift features to prevent leakage
    df = prevent_lookahead_bias(df)
    
    # Drop rows with NaN (first row from shift, last H rows from targets)
    df_clean = df.dropna()
    
    # Identify feature columns
    exclude = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends',
               'Stock Splits', 'Date', 'forward_return', 'label', 'log_forward_return']
    exclude += [c for c in df.columns if c.startswith('target_')]
    feature_cols = [c for c in df_clean.columns if c not in exclude]
    
    print(f"✅ Training data prepared:")
    print(f"   Total rows: {len(df_clean)}")
    print(f"   Features: {len(feature_cols)}")
    print(f"   Targets: {[f'target_{h}d' for h in horizons] + (['target_smart'] if use_smart_target else [])}")
    print(f"   Lost rows (NaN): {len(df) - len(df_clean)}")

    # ------------------------------------------------------------------
    # Enhanced Target scaling diagnostics (StandardScaler)
    # Operate on primary target: target_smart if available, else target_1d
    target_col = 'target_smart' if use_smart_target and 'target_smart' in df_clean.columns else 'target_1d'
    if target_col in df_clean.columns:
        y_raw = df_clean[target_col].values.astype(float)

        # Check for non-finite values before processing
        if not np.all(np.isfinite(y_raw)):
            raise ValueError(f"Target contains non-finite values: {np.sum(~np.isfinite(y_raw))} entries")

        # Statistics BEFORE clipping
        print("\n=== Target Statistics (BEFORE clipping) ===")
        print(f"Range: [{y_raw.min():.4f}, {y_raw.max():.4f}]")
        print(f"Mean: {y_raw.mean():.6f}, Std: {y_raw.std():.6f}")
        print(f"Percentiles: 1%={np.percentile(y_raw, 1):.4f}, 99%={np.percentile(y_raw, 99):.4f}")

        # P1.3: Winsorize extreme tails (2% each side) before clipping
        from scipy.stats.mstats import winsorize
        y_winsorized = np.array(winsorize(y_raw, limits=(0.02, 0.02)))
        winsor_count = np.sum(y_raw != y_winsorized)
        print(f"P1.3: Winsorized {winsor_count} values ({100*winsor_count/len(y_raw):.2f}%)")
        
        # P1.3: Symmetric clipping at ±20%
        y_clipped = np.clip(y_winsorized, -0.20, 0.20)
        clipped_count = np.sum(np.abs(y_winsorized) > 0.20)
        print(f"Clipped {clipped_count} values ({100*clipped_count/len(y_raw):.2f}%)")

        # Use StandardScaler for targets to provide mean=0, std=1 (better gradient flow)
        scaler = StandardScaler()
        y_scaled = scaler.fit_transform(y_clipped.reshape(-1, 1)).flatten()

        # Validation checks
        if not np.all(np.isfinite(y_scaled)):
            raise ValueError(f"Scaled target contains non-finite values: {np.sum(~np.isfinite(y_scaled))}")

        # Attach diagnostic columns efficiently (avoid fragmentation)
        new_cols = pd.DataFrame({
            f'{target_col}_clipped': y_clipped,
            f'{target_col}_scaled': y_scaled
        }, index=df_clean.index)
        df_clean = pd.concat([df_clean, new_cols], axis=1)

        # Save scaler for inverse transform downstream
        df_clean.attrs['target_scaler'] = scaler

        # Log scaling statistics
        print("\\n=== Target Statistics (AFTER scaling) ===")
        print(f"Mean: {y_scaled.mean():.6f} (should be ~0)")
        print(f"Std: {y_scaled.std():.6f} (should be ~1)")
        print(f"Range: [{y_scaled.min():.2f}, {y_scaled.max():.2f}]")

        # Save diagnostic plot (best-effort)
        try:
            plot_path = Path('data') / 'target_engineering_comparison.png'
            plot_target_engineering_diagnostics(y_raw, y_clipped, y_scaled, str(plot_path))
            print(f"Saved target engineering diagnostics to: {plot_path}")
        except Exception as e:
            print(f"⚠️  Failed to save target engineering diagnostics: {e}")

    return df_clean, feature_cols


def plot_target_engineering_diagnostics(y_raw, y_clipped, y_scaled, save_path):
    """Create 4-panel diagnostic plot for target engineering.

    Panels:
    1. Original returns histogram
    2. Clipped returns histogram
    3. Scaled returns histogram
    4. Q-Q plot of scaled returns vs normal distribution
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    ax0 = axes[0, 0]
    ax1 = axes[0, 1]
    ax2 = axes[1, 0]
    ax3 = axes[1, 1]

    ax0.hist(y_raw, bins=80, color='C0', alpha=0.8)
    ax0.set_title('Original returns')
    ax0.set_xlabel('Return')

    ax1.hist(y_clipped, bins=80, color='C1', alpha=0.8)
    ax1.set_title('Clipped returns (±15%)')
    ax1.set_xlabel('Return')

    ax2.hist(y_scaled, bins=80, color='C2', alpha=0.8)
    ax2.set_title('Scaled returns (RobustScaler)')
    ax2.set_xlabel('Scaled value')

    # Q-Q plot
    stats.probplot(y_scaled, dist='norm', plot=ax3)
    ax3.set_title('Q-Q plot: scaled vs normal')

    plt.tight_layout()
    fig.savefig(str(save_path))
    plt.close(fig)
