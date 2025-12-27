"""
GBM Diagnostics and Explainability Module

Provides comprehensive diagnostics for gradient boosting models including:
- Prediction distribution analysis
- Calibration curves (sign prediction)
- SHAP value analysis
- Feature importance rankings

Author: AI-Stocks GBM Integration
Date: December 2025
"""

import sys
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import SHAP
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available. Install with: pip install shap")


# ============================================================================
# PREDICTION DISTRIBUTION ANALYSIS
# ============================================================================

def analyze_prediction_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    symbol: str,
    output_dir: Path = None,
    show_plot: bool = False,
) -> Dict[str, Any]:
    """
    Analyze and visualize prediction distribution vs actuals.
    
    Produces:
    1. Histogram comparison (predicted vs actual)
    2. Q-Q plot for normality
    3. Scatter plot with regression line
    
    Args:
        y_true: Actual returns
        y_pred: Predicted returns
        model_name: Model identifier (e.g., 'xgb', 'lgb')
        symbol: Stock ticker symbol
        output_dir: Directory to save plots
        show_plot: Whether to display plots interactively
    
    Returns:
        Dict with distribution statistics and validation results
    """
    output_dir = output_dir or Path('plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Calculate statistics
    pred_mean = float(np.mean(y_pred))
    pred_std = float(np.std(y_pred))
    pred_min = float(np.min(y_pred))
    pred_max = float(np.max(y_pred))
    
    actual_mean = float(np.mean(y_true))
    actual_std = float(np.std(y_true))
    
    pct_positive_pred = float((y_pred > 0).mean())
    pct_positive_actual = float((y_true > 0).mean())
    
    # Direction accuracy
    dir_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    
    # Correlation
    ic = float(np.corrcoef(y_true, y_pred)[0, 1])
    
    # Variance validation
    variance_valid = pred_std >= 0.001
    distribution_valid = pct_positive_pred >= 0.30 and (1 - pct_positive_pred) >= 0.30
    
    stats = {
        'model': model_name,
        'symbol': symbol,
        'predicted': {
            'mean': pred_mean,
            'std': pred_std,
            'min': pred_min,
            'max': pred_max,
            'pct_positive': pct_positive_pred,
        },
        'actual': {
            'mean': actual_mean,
            'std': actual_std,
            'pct_positive': pct_positive_actual,
        },
        'metrics': {
            'direction_accuracy': dir_acc,
            'information_coefficient': ic,
            'r2': float(r2_score(y_true, y_pred)),
            'mse': float(mean_squared_error(y_true, y_pred)),
            'mae': float(mean_absolute_error(y_true, y_pred)),
        },
        'validation': {
            'variance_valid': variance_valid,
            'distribution_valid': distribution_valid,
            'overall_valid': variance_valid and distribution_valid,
        }
    }
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 1. Histogram comparison
    ax1 = axes[0, 0]
    bins = 50
    ax1.hist(y_true, bins=bins, alpha=0.5, label='Actual', color='blue', density=True)
    ax1.hist(y_pred, bins=bins, alpha=0.5, label='Predicted', color='orange', density=True)
    ax1.axvline(actual_mean, color='blue', linestyle='--', label=f'Actual Mean: {actual_mean:.4f}')
    ax1.axvline(pred_mean, color='orange', linestyle='--', label=f'Pred Mean: {pred_mean:.4f}')
    ax1.set_xlabel('Return')
    ax1.set_ylabel('Density')
    ax1.set_title(f'{model_name.upper()} - Prediction vs Actual Distribution')
    ax1.legend()
    
    # 2. Q-Q Plot for predictions
    ax2 = axes[0, 1]
    from scipy import stats as scipy_stats
    scipy_stats.probplot(y_pred, dist="norm", plot=ax2)
    ax2.set_title(f'{model_name.upper()} - Q-Q Plot (Predictions)')
    
    # 3. Scatter plot: Predicted vs Actual
    ax3 = axes[1, 0]
    ax3.scatter(y_true, y_pred, alpha=0.3, s=10)
    
    # Add regression line
    z = np.polyfit(y_true, y_pred, 1)
    p = np.poly1d(z)
    x_line = np.linspace(y_true.min(), y_true.max(), 100)
    ax3.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'Fit: y={z[0]:.3f}x + {z[1]:.4f}')
    
    # Add perfect prediction line
    ax3.plot(x_line, x_line, 'k--', linewidth=1, alpha=0.5, label='Perfect Prediction')
    
    ax3.set_xlabel('Actual Return')
    ax3.set_ylabel('Predicted Return')
    ax3.set_title(f'{model_name.upper()} - Predicted vs Actual (R²={stats["metrics"]["r2"]:.4f})')
    ax3.legend()
    
    # 4. Residual plot
    ax4 = axes[1, 1]
    residuals = y_true - y_pred
    ax4.scatter(y_pred, residuals, alpha=0.3, s=10)
    ax4.axhline(0, color='red', linestyle='--')
    ax4.set_xlabel('Predicted Return')
    ax4.set_ylabel('Residual (Actual - Predicted)')
    ax4.set_title(f'{model_name.upper()} - Residual Plot')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f'gbm_distribution_{model_name}_{symbol}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved distribution plot to {plot_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    stats['plot_path'] = str(plot_path)
    
    return stats


# ============================================================================
# CALIBRATION ANALYSIS
# ============================================================================

def analyze_sign_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    symbol: str,
    n_bins: int = 10,
    output_dir: Path = None,
    show_plot: bool = False,
) -> Dict[str, Any]:
    """
    Analyze calibration of sign predictions.
    
    Bins predictions by magnitude and computes actual win rate per bin.
    A well-calibrated model should have higher win rates in higher confidence bins.
    
    Args:
        y_true: Actual returns
        y_pred: Predicted returns
        model_name: Model identifier
        symbol: Stock ticker
        n_bins: Number of bins for calibration
        output_dir: Directory to save plots
        show_plot: Whether to display plots
    
    Returns:
        Dict with calibration metrics per bin
    """
    output_dir = output_dir or Path('plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to numpy
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Compute prediction magnitude (confidence proxy)
    pred_magnitude = np.abs(y_pred)
    
    # Create bins based on prediction magnitude
    bins = np.percentile(pred_magnitude, np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)  # Remove duplicates
    
    calibration_data = []
    
    for i in range(len(bins) - 1):
        bin_mask = (pred_magnitude >= bins[i]) & (pred_magnitude < bins[i + 1])
        if i == len(bins) - 2:  # Include right edge in last bin
            bin_mask = (pred_magnitude >= bins[i]) & (pred_magnitude <= bins[i + 1])
        
        if bin_mask.sum() > 0:
            # Check if sign prediction was correct
            correct = np.sign(y_true[bin_mask]) == np.sign(y_pred[bin_mask])
            
            bin_info = {
                'bin_idx': i,
                'bin_low': float(bins[i]),
                'bin_high': float(bins[i + 1]),
                'count': int(bin_mask.sum()),
                'win_rate': float(correct.mean()),
                'avg_magnitude': float(pred_magnitude[bin_mask].mean()),
                'avg_actual_return': float(y_true[bin_mask].mean()),
                'avg_predicted_return': float(y_pred[bin_mask].mean()),
            }
            calibration_data.append(bin_info)
    
    calibration_df = pd.DataFrame(calibration_data)
    
    # Compute overall calibration metrics
    overall_dir_acc = float(np.mean(np.sign(y_true) == np.sign(y_pred)))
    
    # Check monotonicity: higher confidence should have higher win rate
    win_rates = calibration_df['win_rate'].values
    is_monotonic = np.all(np.diff(win_rates) >= -0.05)  # Allow 5% deviation
    
    result = {
        'model': model_name,
        'symbol': symbol,
        'n_bins': len(calibration_data),
        'calibration_bins': calibration_data,
        'overall_direction_accuracy': overall_dir_acc,
        'is_monotonic': bool(is_monotonic),
        'highest_bin_win_rate': float(win_rates[-1]) if len(win_rates) > 0 else 0.0,
        'lowest_bin_win_rate': float(win_rates[0]) if len(win_rates) > 0 else 0.0,
    }
    
    # Create calibration plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Win rate by confidence bin
    ax1 = axes[0]
    bin_centers = [(b['bin_low'] + b['bin_high']) / 2 for b in calibration_data]
    win_rates = [b['win_rate'] for b in calibration_data]
    counts = [b['count'] for b in calibration_data]
    
    bars = ax1.bar(range(len(calibration_data)), win_rates, color='steelblue', alpha=0.7)
    ax1.axhline(0.5, color='red', linestyle='--', label='Random (50%)')
    ax1.axhline(overall_dir_acc, color='green', linestyle='--', label=f'Overall ({overall_dir_acc:.1%})')
    
    ax1.set_xlabel('Prediction Magnitude Bin (Low → High)')
    ax1.set_ylabel('Win Rate (Correct Sign)')
    ax1.set_title(f'{model_name.upper()} - Sign Prediction Calibration')
    ax1.set_ylim(0, 1)
    ax1.legend()
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'n={count}', ha='center', va='bottom', fontsize=8)
    
    # 2. Reliability diagram
    ax2 = axes[1]
    avg_magnitudes = [b['avg_magnitude'] for b in calibration_data]
    
    ax2.scatter(avg_magnitudes, win_rates, s=100, c='steelblue', edgecolors='black')
    ax2.plot([0, max(avg_magnitudes)], [0.5, 0.5 + max(avg_magnitudes)*2], 
             'k--', alpha=0.5, label='Ideal Calibration')
    
    ax2.set_xlabel('Average Prediction Magnitude (Confidence Proxy)')
    ax2.set_ylabel('Actual Win Rate')
    ax2.set_title(f'{model_name.upper()} - Reliability Diagram')
    ax2.set_ylim(0.3, 0.8)
    ax2.legend()
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f'gbm_calibration_{model_name}_{symbol}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved calibration plot to {plot_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    result['plot_path'] = str(plot_path)
    
    return result


# ============================================================================
# SHAP ANALYSIS
# ============================================================================

def compute_shap_values(
    model,
    X: np.ndarray,
    feature_names: List[str],
    model_name: str,
    symbol: str,
    max_samples: int = 1000,
    output_dir: Path = None,
    show_plot: bool = False,
) -> Dict[str, Any]:
    """
    Compute SHAP values for model interpretability.
    
    Args:
        model: Trained GBM model
        X: Feature matrix (scaled)
        feature_names: List of feature names
        model_name: Model identifier
        symbol: Stock ticker
        max_samples: Maximum samples for SHAP calculation
        output_dir: Directory to save plots
        show_plot: Whether to display plots
    
    Returns:
        Dict with SHAP values and feature importance rankings
    """
    if not SHAP_AVAILABLE:
        logger.warning("SHAP not available, skipping SHAP analysis")
        return {'status': 'skipped', 'reason': 'shap_not_available'}
    
    output_dir = output_dir or Path('plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Computing SHAP values for {model_name.upper()}...")
    
    # Subsample if too large
    if len(X) > max_samples:
        indices = np.random.choice(len(X), max_samples, replace=False)
        X_sample = X[indices]
    else:
        X_sample = X
    
    # Create explainer
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
    except Exception as e:
        logger.error(f"SHAP computation failed: {e}")
        return {'status': 'failed', 'error': str(e)}
    
    # Compute mean absolute SHAP values per feature
    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    
    # Create importance ranking
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_abs_shap,
    }).sort_values('mean_abs_shap', ascending=False)
    
    # Save importance CSV
    csv_path = output_dir.parent / 'logs' / f'feature_importances_{model_name}_{symbol}.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    importance_df.to_csv(csv_path, index=False)
    logger.info(f"Saved feature importances to {csv_path}")
    
    # Create SHAP summary plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    # 1. Summary plot (bar)
    plt.sca(axes[0])
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                      plot_type='bar', show=False, max_display=20)
    axes[0].set_title(f'{model_name.upper()} - SHAP Feature Importance')
    
    # 2. Summary plot (beeswarm)
    plt.sca(axes[1])
    shap.summary_plot(shap_values, X_sample, feature_names=feature_names, 
                      show=False, max_display=20)
    axes[1].set_title(f'{model_name.upper()} - SHAP Summary')
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f'shap_summary_{model_name}_{symbol}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved SHAP summary to {plot_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    # Create dependence plots for top 3 features
    top_features = importance_df.head(3)['feature'].tolist()
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for i, feat in enumerate(top_features):
        feat_idx = feature_names.index(feat)
        plt.sca(axes[i])
        shap.dependence_plot(feat_idx, shap_values, X_sample, 
                            feature_names=feature_names, show=False)
        axes[i].set_title(f'{feat}')
    
    plt.tight_layout()
    
    dep_plot_path = output_dir / f'shap_dependence_top3_{model_name}_{symbol}.png'
    plt.savefig(dep_plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"Saved SHAP dependence plots to {dep_plot_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    result = {
        'status': 'success',
        'model': model_name,
        'symbol': symbol,
        'n_samples': len(X_sample),
        'top_features': importance_df.head(20).to_dict('records'),
        'importance_csv_path': str(csv_path),
        'summary_plot_path': str(plot_path),
        'dependence_plot_path': str(dep_plot_path),
    }
    
    return result


# ============================================================================
# MAIN DIAGNOSTICS FUNCTION
# ============================================================================

def run_gbm_diagnostics(
    symbol: str,
    model_dir: Path = None,
    output_dir: Path = None,
    show_plots: bool = False,
) -> Dict[str, Any]:
    """
    Run comprehensive diagnostics on trained GBM models.
    
    Args:
        symbol: Stock ticker symbol
        model_dir: Directory containing trained models
        output_dir: Directory to save diagnostic outputs
        show_plots: Whether to display plots interactively
    
    Returns:
        Dict with all diagnostic results
    """
    model_dir = model_dir or Path(f'saved_models/{symbol}/gbm')
    output_dir = output_dir or Path('plots')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    logger.info(f"Running GBM diagnostics for {symbol}...")
    
    results = {
        'symbol': symbol,
        'models': {}
    }
    
    # Load feature columns
    feature_cols_path = model_dir / 'feature_columns.pkl'
    if not feature_cols_path.exists():
        # Try parent directory
        feature_cols_path = model_dir.parent / 'feature_columns.pkl'
    
    if feature_cols_path.exists():
        with open(feature_cols_path, 'rb') as f:
            feature_cols = pickle.load(f)
        logger.info(f"Loaded {len(feature_cols)} feature columns")
    else:
        logger.error(f"Feature columns not found at {feature_cols_path}")
        return {'status': 'failed', 'error': 'feature_columns_not_found'}
    
    # Load OOF predictions and run diagnostics for each model
    for model_name in ['xgb', 'lgb']:
        model_path = model_dir / f'{model_name}_reg.joblib'
        scaler_path = model_dir / f'{model_name}_scaler.joblib'
        oof_path = logs_dir / f'gbm_oof_preds_{model_name}_{symbol}.csv'
        
        if not model_path.exists():
            logger.warning(f"{model_name.upper()} model not found at {model_path}")
            continue
        
        logger.info(f"\n=== {model_name.upper()} Diagnostics ===")
        
        model_results = {}
        
        # Load OOF predictions
        if oof_path.exists():
            oof_df = pd.read_csv(oof_path)
            # Filter out NaN predictions (from folds not used)
            valid_mask = ~oof_df['predicted'].isna()
            y_true = oof_df.loc[valid_mask, 'actual'].values
            y_pred = oof_df.loc[valid_mask, 'predicted'].values
            
            logger.info(f"Loaded {len(y_true)} OOF predictions")
            
            # 1. Distribution analysis
            dist_results = analyze_prediction_distribution(
                y_true, y_pred, model_name, symbol, output_dir, show_plots
            )
            model_results['distribution'] = dist_results
            
            # 2. Calibration analysis
            calib_results = analyze_sign_calibration(
                y_true, y_pred, model_name, symbol, output_dir=output_dir, show_plot=show_plots
            )
            model_results['calibration'] = calib_results
        else:
            logger.warning(f"OOF predictions not found at {oof_path}")
        
        # 3. SHAP analysis (requires loading model and data)
        if SHAP_AVAILABLE and model_path.exists() and scaler_path.exists():
            try:
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)
                
                # Load data for SHAP
                from data.cache_manager import DataCacheManager
                cache_manager = DataCacheManager()
                _, _, prepared_df, _ = cache_manager.get_or_fetch_data(symbol, include_sentiment=True)
                
                if prepared_df is not None:
                    X = prepared_df[feature_cols].values
                    X_scaled = scaler.transform(X)
                    
                    shap_results = compute_shap_values(
                        model, X_scaled, feature_cols, model_name, symbol,
                        output_dir=output_dir, show_plot=show_plots
                    )
                    model_results['shap'] = shap_results
            except Exception as e:
                logger.error(f"SHAP analysis failed: {e}")
                model_results['shap'] = {'status': 'failed', 'error': str(e)}
        
        results['models'][model_name] = model_results
    
    # Save combined results
    results_path = logs_dir / f'gbm_diagnostics_{symbol}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Saved diagnostics results to {results_path}")
    
    return results


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run GBM diagnostics')
    parser.add_argument('symbol', type=str, help='Stock ticker symbol')
    parser.add_argument('--show-plots', action='store_true', help='Display plots interactively')
    
    args = parser.parse_args()
    
    results = run_gbm_diagnostics(
        symbol=args.symbol.upper(),
        show_plots=args.show_plots
    )
    
    print("\n" + "="*60)
    print("DIAGNOSTICS SUMMARY")
    print("="*60)
    
    for model_name, model_results in results.get('models', {}).items():
        print(f"\n{model_name.upper()}:")
        
        if 'distribution' in model_results:
            dist = model_results['distribution']
            metrics = dist.get('metrics', {})
            print(f"  R²: {metrics.get('r2', 0):.4f}")
            print(f"  Direction Acc: {metrics.get('direction_accuracy', 0):.4f}")
            print(f"  IC: {metrics.get('information_coefficient', 0):.4f}")
            
            valid = dist.get('validation', {})
            status = "✓" if valid.get('overall_valid', False) else "✗"
            print(f"  Validation: {status}")
        
        if 'calibration' in model_results:
            calib = model_results['calibration']
            print(f"  Calibration Monotonic: {'✓' if calib.get('is_monotonic') else '✗'}")
            print(f"  Highest Bin Win Rate: {calib.get('highest_bin_win_rate', 0):.4f}")
