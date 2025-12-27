"""
GBM-Only Backtest Script

Runs a backtest using only GBM models (XGBoost and LightGBM) 
without the LSTM/Transformer regressor to verify GBM performance in isolation.

Produces the same output files as the standard backtest for consistency.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import joblib
from datetime import datetime
import json
import pickle

# Optional plotting
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

from data.data_fetcher import fetch_stock_data
from data.feature_engineer import engineer_features, get_feature_columns
from inference.load_gbm_models import load_gbm_models, predict_with_gbm, predict_with_ensemble


def run_gbm_only_backtest(
    symbol: str = 'AAPL',
    lookback_days: int = 120,
    use_ensemble: bool = True,
    buy_threshold: float = 0.002,  # 0.2% predicted return to trigger BUY
    sell_threshold: float = -0.002,  # -0.2% predicted return to trigger SELL
):
    """
    Run backtest using only GBM models.
    
    Args:
        symbol: Stock symbol
        lookback_days: Number of days to backtest
        use_ensemble: Whether to ensemble XGB and LGB predictions
        buy_threshold: Minimum predicted return to BUY
        sell_threshold: Maximum predicted return to SELL
    """
    print(f"\n{'='*70}")
    print(f"  GBM-ONLY BACKTEST: {symbol}")
    print(f"  Lookback: {lookback_days} days")
    print(f"  Ensemble: {use_ensemble}")
    print(f"  Buy threshold: {buy_threshold*100:.2f}%")
    print(f"  Sell threshold: {sell_threshold*100:.2f}%")
    print(f"{'='*70}")
    
    # Load GBM models
    print("\n[1] Loading GBM models...")
    gbm_bundle, load_metadata = load_gbm_models(symbol)
    
    if gbm_bundle is None:
        print(f"[ERROR] Failed to load GBM models: {load_metadata.get('errors', [])}")
        return None
    
    print(f"   ✓ XGB model loaded: {gbm_bundle.has_xgb()}")
    print(f"   ✓ LGB model loaded: {gbm_bundle.has_lgb()}")
    print(f"   ✓ Feature columns: {len(gbm_bundle.feature_columns)}")
    
    # Fetch data
    print("\n[2] Fetching data...")
    df = fetch_stock_data(symbol, period='10y')
    
    if df is None or len(df) == 0:
        print("[ERROR] Failed to fetch data")
        return None
    
    print(f"   ✓ Fetched {len(df)} rows")
    
    # Keep a copy of OHLCV data before feature engineering
    # Note: yfinance uses capitalized column names
    price_cols = [c for c in ['Close', 'close', 'Open', 'open', 'High', 'high', 'Low', 'low', 'Volume', 'volume'] if c in df.columns]
    df_prices = df[price_cols].copy()
    
    # Normalize column names to lowercase
    df_prices.columns = [c.lower() for c in df_prices.columns]
    
    # Remove timezone info to allow join
    if hasattr(df_prices.index, 'tz') and df_prices.index.tz is not None:
        df_prices.index = df_prices.index.tz_localize(None)
    
    # Engineer features
    print("\n[3] Engineering features...")
    df_engineered = engineer_features(df, symbol=symbol)
    
    if df_engineered is None:
        print("[ERROR] Failed to engineer features")
        return None
    
    print(f"   ✓ Engineered features: {len(df_engineered)} rows")
    
    # Get feature columns
    feature_cols = gbm_bundle.feature_columns
    
    # Re-attach price data to engineered features (using index alignment)
    df_engineered = df_engineered.join(df_prices, how='left')
    
    # Prepare backtest window
    df_backtest = df_engineered.tail(lookback_days + 10).copy()  # Extra buffer
    
    # Calculate actual returns
    df_backtest['actual_return'] = df_backtest['close'].pct_change().shift(-1)
    df_backtest = df_backtest.dropna(subset=['actual_return'])
    df_backtest = df_backtest.tail(lookback_days)
    
    print(f"\n[4] Running backtest on {len(df_backtest)} days...")
    
    # Track results
    results = []
    position = 0  # 0 = flat, 1 = long, -1 = short
    cash = 100000
    shares = 0
    portfolio_values = []
    trade_log = []
    
    for i, (idx, row) in enumerate(df_backtest.iterrows()):
        # Get features for this day
        try:
            X = row[feature_cols].values.reshape(1, -1)
            
            # Handle missing values
            X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Get predictions
            if use_ensemble:
                # Use both XGB and LGB with equal weights
                ensemble_result = predict_with_ensemble(
                    gbm_bundle=gbm_bundle, 
                    regressor_pred=None,  # No LSTM regressor 
                    features=X, 
                    fusion_mode='gbm_only',
                    weights={'xgb': 0.4, 'lgb': 0.6}
                )
                pred = ensemble_result['fused_prediction']
            else:
                pred = predict_with_gbm(gbm_bundle, X, model='lgb')
            
            pred_return = float(pred[0]) if hasattr(pred, '__len__') else float(pred)
            actual_return = row['actual_return']
            price = row['close']
            
            # Generate signal
            if pred_return > buy_threshold:
                signal = 'BUY'
            elif pred_return < sell_threshold:
                signal = 'SELL'
            else:
                signal = 'HOLD'
            
            # Execute trades
            if signal == 'BUY' and position <= 0:
                if position == -1:
                    # Close short
                    cash += shares * price
                    shares = 0
                # Go long
                shares = cash // price
                cash -= shares * price
                position = 1
                trade_log.append({
                    'date': idx,
                    'action': 'BUY',
                    'price': price,
                    'shares': shares,
                    'pred_return': pred_return,
                    'actual_return': actual_return
                })
            elif signal == 'SELL' and position >= 0:
                if position == 1:
                    # Close long
                    cash += shares * price
                    shares = 0
                # Go short (if allowed)
                # For simplicity, just go to cash
                position = 0
                trade_log.append({
                    'date': idx,
                    'action': 'SELL',
                    'price': price,
                    'shares': shares,
                    'pred_return': pred_return,
                    'actual_return': actual_return
                })
            
            # Calculate portfolio value
            portfolio_value = cash + shares * price
            portfolio_values.append(portfolio_value)
            
            results.append({
                'date': idx,
                'price': price,
                'pred_return': pred_return,
                'actual_return': actual_return,
                'signal': signal,
                'position': position,
                'portfolio_value': portfolio_value,
                'correct_direction': (pred_return > 0 and actual_return > 0) or 
                                   (pred_return < 0 and actual_return < 0)
            })
            
        except Exception as e:
            print(f"   [WARN] Error on day {i}: {e}")
            continue
    
    # Calculate metrics
    results_df = pd.DataFrame(results)
    
    if len(results_df) == 0:
        print("[ERROR] No results generated")
        return None
    
    # Directional accuracy
    dir_accuracy = results_df['correct_direction'].mean()
    
    # Returns and equity curve
    results_df['daily_return'] = results_df['portfolio_value'].pct_change().fillna(0)
    results_df['cumulative_return'] = (1 + results_df['daily_return']).cumprod() - 1
    
    total_return = (portfolio_values[-1] - 100000) / 100000
    buy_hold_return = (df_backtest['close'].iloc[-1] / df_backtest['close'].iloc[0]) - 1
    
    # Calculate buy & hold equity curve for comparison
    initial_shares_bh = 100000 / df_backtest['close'].iloc[0]
    buy_hold_equity = initial_shares_bh * df_backtest['close'].values
    
    # Calculate Sharpe ratio
    daily_returns = results_df['daily_return'].values
    if np.std(daily_returns) > 0:
        sharpe = np.sqrt(252) * np.mean(daily_returns) / np.std(daily_returns)
    else:
        sharpe = 0.0
    
    # Calculate max drawdown
    equity_series = np.array(portfolio_values)
    running_max = np.maximum.accumulate(equity_series)
    drawdown = (equity_series - running_max) / running_max
    max_drawdown = np.min(drawdown)
    
    # Prediction stats
    pred_positive = (results_df['pred_return'] > 0).mean()
    actual_positive = (results_df['actual_return'] > 0).mean()
    
    # Signal distribution
    signal_counts = results_df['signal'].value_counts()
    
    # Win rate
    winning_trades = results_df[(results_df['signal'] != 'HOLD') & (results_df['correct_direction'] == True)]
    total_signal_days = len(results_df[results_df['signal'] != 'HOLD'])
    win_rate = len(winning_trades) / total_signal_days if total_signal_days > 0 else 0
    
    print(f"\n{'='*70}")
    print(f"  BACKTEST RESULTS")
    print(f"{'='*70}")
    print(f"\n  Directional Accuracy: {dir_accuracy*100:.2f}%")
    print(f"  Strategy Return: {total_return*100:.2f}%")
    print(f"  Buy & Hold Return: {buy_hold_return*100:.2f}%")
    print(f"  Alpha: {(total_return - buy_hold_return)*100:.2f}%")
    print(f"  Sharpe Ratio: {sharpe:.3f}")
    print(f"  Max Drawdown: {max_drawdown*100:.2f}%")
    print(f"  Win Rate: {win_rate*100:.1f}%")
    print(f"\n  Prediction Stats:")
    print(f"    Positive predictions: {pred_positive*100:.1f}%")
    print(f"    Actual positive days: {actual_positive*100:.1f}%")
    print(f"    Pred range: [{results_df['pred_return'].min()*100:.3f}%, {results_df['pred_return'].max()*100:.3f}%]")
    print(f"\n  Signal Distribution:")
    for signal, count in signal_counts.items():
        print(f"    {signal}: {count} ({count/len(results_df)*100:.1f}%)")
    print(f"\n  Trades Executed: {len(trade_log)}")
    
    # ========== SAVE ALL FILES (matching standard backtest) ==========
    output_dir = Path('backtest_results') / f"{symbol}_gbm_only_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. daily_results.csv (enhanced)
    results_df.to_csv(output_dir / 'daily_results.csv', index=False)
    
    # 2. trade_log.csv
    trade_log_df = pd.DataFrame(trade_log)
    trade_log_df.to_csv(output_dir / 'trade_log.csv', index=False)
    
    # 3. backtest_trades.csv (alias for compatibility)
    trade_log_df.to_csv(output_dir / 'backtest_trades.csv', index=False)
    
    # 4. backtest_daily_positions.csv
    positions_df = results_df[['date', 'price', 'position', 'portfolio_value', 'daily_return']].copy()
    positions_df.to_csv(output_dir / 'backtest_daily_positions.csv', index=False)
    
    # 5. backtest_signals.csv
    signals_df = results_df[['date', 'signal', 'pred_return', 'actual_return', 'correct_direction']].copy()
    signals_df.to_csv(output_dir / 'backtest_signals.csv', index=False)
    
    # 6. summary.json (enhanced)
    summary = {
        'symbol': symbol,
        'model_type': 'gbm_only',
        'lookback_days': lookback_days,
        'use_ensemble': use_ensemble,
        'buy_threshold': buy_threshold,
        'sell_threshold': sell_threshold,
        'directional_accuracy': float(dir_accuracy),
        'strategy_return': float(total_return),
        'buy_hold_return': float(buy_hold_return),
        'alpha': float(total_return - buy_hold_return),
        'sharpe_ratio': float(sharpe),
        'max_drawdown': float(max_drawdown),
        'win_rate': float(win_rate),
        'pred_positive_pct': float(pred_positive),
        'actual_positive_pct': float(actual_positive),
        'pred_std': float(results_df['pred_return'].std()),
        'pred_mean': float(results_df['pred_return'].mean()),
        'signal_counts': {str(k): int(v) for k, v in signal_counts.items()},
        'n_trades': len(trade_log),
        'run_date': datetime.now().isoformat()
    }
    with open(output_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    # 7. backtest.pkl (for analyzer compatibility)
    backtest_data = {
        'dates': results_df['date'].values,
        'prices': results_df['price'].values,
        'positions': results_df['position'].values,
        'signals': results_df['signal'].values,
        'equity': np.array(portfolio_values),
        'buy_hold_equity': buy_hold_equity,
        'returns': daily_returns,
        'predictions': results_df['pred_return'].values,
        'actuals': results_df['actual_return'].values,
        'cum_return': total_return,
        'metrics': {
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'directional_accuracy': dir_accuracy
        },
        'backtester_metrics': {
            'cum_return': f"{total_return*100:.2f}%",
            'sharpe': sharpe,
            'max_drawdown': f"{max_drawdown*100:.2f}%",
            'buy_hold_cum_return': buy_hold_return,
            'alpha': total_return - buy_hold_return
        },
        'trade_log': trade_log_df
    }
    with open(output_dir / 'backtest.pkl', 'wb') as f:
        pickle.dump(backtest_data, f)
    
    # 8. equity_comparison.csv
    equity_df = pd.DataFrame({
        'date': results_df['date'].values,
        'strategy_equity': portfolio_values,
        'buy_hold_equity': buy_hold_equity,
        'strategy_return': results_df['cumulative_return'].values,
        'buy_hold_return': (buy_hold_equity / buy_hold_equity[0]) - 1
    })
    equity_df.to_csv(output_dir / 'equity_comparison.csv', index=False)
    
    # 9. drawdowns.csv
    drawdown_df = pd.DataFrame({
        'date': results_df['date'].values,
        'equity': portfolio_values,
        'drawdown': drawdown,
        'running_max': running_max
    })
    drawdown_df.to_csv(output_dir / 'drawdowns.csv', index=False)
    
    # 10. confidence_analysis.json (simplified for GBM)
    confidence_analysis = {
        'model_type': 'gbm_ensemble',
        'avg_prediction_magnitude': float(np.mean(np.abs(results_df['pred_return']))),
        'prediction_std': float(results_df['pred_return'].std()),
        'directional_accuracy': float(dir_accuracy)
    }
    with open(output_dir / 'confidence_analysis.json', 'w') as f:
        json.dump(confidence_analysis, f, indent=2)
    
    # 11. calibration_analysis.json
    calibration = {
        'predicted_positive_pct': float(pred_positive),
        'actual_positive_pct': float(actual_positive),
        'calibration_error': float(abs(pred_positive - actual_positive))
    }
    with open(output_dir / 'calibration_analysis.json', 'w') as f:
        json.dump(calibration, f, indent=2)
    
    # ========== PLOTS (if matplotlib available) ==========
    if HAS_MATPLOTLIB:
        try:
            # 12. equity_curve.png
            fig, ax = plt.subplots(figsize=(12, 6))
            dates = pd.to_datetime(results_df['date'])
            ax.plot(dates, portfolio_values, label='GBM Strategy', linewidth=2)
            ax.plot(dates, buy_hold_equity, label='Buy & Hold', linewidth=2, alpha=0.7)
            ax.set_xlabel('Date')
            ax.set_ylabel('Portfolio Value ($)')
            ax.set_title(f'{symbol} - GBM-Only Backtest Equity Curve')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'equity_curve.png', dpi=150)
            plt.close()
            
            # 13. backtest.png (same as equity curve)
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(dates, portfolio_values, label='GBM Strategy', linewidth=2, color='blue')
            ax.plot(dates, buy_hold_equity, label='Buy & Hold', linewidth=2, color='orange', alpha=0.7)
            ax.fill_between(dates, portfolio_values, buy_hold_equity, alpha=0.2, 
                           where=(np.array(portfolio_values) > buy_hold_equity), color='green')
            ax.fill_between(dates, portfolio_values, buy_hold_equity, alpha=0.2,
                           where=(np.array(portfolio_values) <= buy_hold_equity), color='red')
            ax.set_xlabel('Date')
            ax.set_ylabel('Portfolio Value ($)')
            ax.set_title(f'{symbol} - GBM-Only Backtest ({total_return*100:.1f}% vs B&H {buy_hold_return*100:.1f}%)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(output_dir / 'backtest.png', dpi=150)
            plt.close()
            
            # 14. prediction_distribution.png
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Histogram
            axes[0].hist(results_df['pred_return']*100, bins=30, alpha=0.7, label='Predicted', color='blue')
            axes[0].hist(results_df['actual_return']*100, bins=30, alpha=0.7, label='Actual', color='orange')
            axes[0].set_xlabel('Return (%)')
            axes[0].set_ylabel('Frequency')
            axes[0].set_title('Prediction vs Actual Distribution')
            axes[0].legend()
            axes[0].axvline(0, color='black', linestyle='--', alpha=0.5)
            
            # Scatter
            axes[1].scatter(results_df['pred_return']*100, results_df['actual_return']*100, alpha=0.5)
            axes[1].set_xlabel('Predicted Return (%)')
            axes[1].set_ylabel('Actual Return (%)')
            axes[1].set_title(f'Predicted vs Actual (Dir Acc: {dir_accuracy*100:.1f}%)')
            # Add diagonal line
            lims = [min(axes[1].get_xlim()[0], axes[1].get_ylim()[0]),
                    max(axes[1].get_xlim()[1], axes[1].get_ylim()[1])]
            axes[1].plot(lims, lims, 'r--', alpha=0.5, label='Perfect')
            axes[1].axhline(0, color='gray', linestyle=':', alpha=0.5)
            axes[1].axvline(0, color='gray', linestyle=':', alpha=0.5)
            
            plt.tight_layout()
            plt.savefig(output_dir / f'{symbol}_prediction_distribution.png', dpi=150)
            plt.close()
            
            # 15. backtest_positions_hist.png
            fig, ax = plt.subplots(figsize=(8, 5))
            position_counts = results_df['position'].value_counts()
            ax.bar(position_counts.index.astype(str), position_counts.values)
            ax.set_xlabel('Position')
            ax.set_ylabel('Days')
            ax.set_title('Position Distribution')
            plt.tight_layout()
            plt.savefig(output_dir / 'backtest_positions_hist.png', dpi=150)
            plt.close()
            
            # 16. dashboard.png (comprehensive)
            fig = plt.figure(figsize=(16, 12))
            
            # Equity curve
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.plot(dates, portfolio_values, label='GBM Strategy', linewidth=2)
            ax1.plot(dates, buy_hold_equity, label='Buy & Hold', linewidth=2, alpha=0.7)
            ax1.set_title(f'{symbol} Equity Curve')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Drawdown
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.fill_between(dates, drawdown*100, 0, color='red', alpha=0.5)
            ax2.set_title(f'Drawdown (Max: {max_drawdown*100:.1f}%)')
            ax2.set_ylabel('Drawdown (%)')
            ax2.grid(True, alpha=0.3)
            
            # Prediction scatter
            ax3 = fig.add_subplot(2, 2, 3)
            colors = ['green' if c else 'red' for c in results_df['correct_direction']]
            ax3.scatter(results_df['pred_return']*100, results_df['actual_return']*100, c=colors, alpha=0.5)
            ax3.set_xlabel('Predicted (%)')
            ax3.set_ylabel('Actual (%)')
            ax3.set_title(f'Predictions (Dir Acc: {dir_accuracy*100:.1f}%)')
            ax3.axhline(0, color='gray', linestyle=':', alpha=0.5)
            ax3.axvline(0, color='gray', linestyle=':', alpha=0.5)
            
            # Stats table
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.axis('off')
            stats_text = f"""
            GBM-ONLY BACKTEST SUMMARY
            ========================
            
            Symbol: {symbol}
            Period: {lookback_days} days
            
            RETURNS
            -------
            Strategy Return: {total_return*100:.2f}%
            Buy & Hold Return: {buy_hold_return*100:.2f}%
            Alpha: {(total_return - buy_hold_return)*100:.2f}%
            
            RISK METRICS
            ------------
            Sharpe Ratio: {sharpe:.3f}
            Max Drawdown: {max_drawdown*100:.2f}%
            
            ACCURACY
            --------
            Directional Accuracy: {dir_accuracy*100:.1f}%
            Win Rate: {win_rate*100:.1f}%
            
            SIGNALS
            -------
            BUY: {signal_counts.get('BUY', 0)}
            SELL: {signal_counts.get('SELL', 0)}
            HOLD: {signal_counts.get('HOLD', 0)}
            """
            ax4.text(0.1, 0.9, stats_text, transform=ax4.transAxes, fontsize=10,
                    verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'dashboard.png', dpi=150)
            plt.close()
            
            print(f"   ✓ Generated all plots")
            
        except Exception as e:
            print(f"   [WARN] Could not generate plots: {e}")
    else:
        print("   [INFO] matplotlib not available, skipping plots")
    
    print(f"\n  ✓ All files saved to: {output_dir}")
    print(f"    - daily_results.csv")
    print(f"    - trade_log.csv")
    print(f"    - backtest_trades.csv")
    print(f"    - backtest_daily_positions.csv")
    print(f"    - backtest_signals.csv")
    print(f"    - summary.json")
    print(f"    - backtest.pkl")
    print(f"    - equity_comparison.csv")
    print(f"    - drawdowns.csv")
    print(f"    - confidence_analysis.json")
    print(f"    - calibration_analysis.json")
    if HAS_MATPLOTLIB:
        print(f"    - equity_curve.png")
        print(f"    - backtest.png")
        print(f"    - {symbol}_prediction_distribution.png")
        print(f"    - backtest_positions_hist.png")
        print(f"    - dashboard.png")
    
    return summary


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='GBM-Only Backtest')
    parser.add_argument('symbol', type=str, default='AAPL', nargs='?', help='Stock symbol')
    parser.add_argument('--days', type=int, default=120, help='Lookback days')
    parser.add_argument('--no-ensemble', action='store_true', help='Use only LGB (no ensemble)')
    parser.add_argument('--buy-threshold', type=float, default=0.002, help='Buy threshold')
    parser.add_argument('--sell-threshold', type=float, default=-0.002, help='Sell threshold')
    
    args = parser.parse_args()
    
    run_gbm_only_backtest(
        symbol=args.symbol,
        lookback_days=args.days,
        use_ensemble=not args.no_ensemble,
        buy_threshold=args.buy_threshold,
        sell_threshold=args.sell_threshold
    )
