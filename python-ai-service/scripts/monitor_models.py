#!/usr/bin/env python3
"""
Daily Model Monitoring and Degradation Detection

Runs predictions for all trained symbols and compares with actual market data:
- Calculates Information Coefficient (IC) for prediction quality
- Measures directional accuracy against actual returns
- Detects performance degradation
- Sends alerts if metrics fall below thresholds
- Logs results to monitoring_logs/ for trend analysis

Run daily after market close via cron job.
"""

import sys
from pathlib import Path
import json
import pickle
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# Setup paths
CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Setup logging
log_dir = PROJECT_ROOT / 'monitoring_logs'
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"monitoring_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

from data.data_fetcher import fetch_stock_data
from data.feature_engineer import engineer_features
from utils.model_paths import ModelPaths, get_saved_models_root


# ===================================================================
# MONITORING THRESHOLDS
# ===================================================================

MONITORING_THRESHOLDS = {
    'ic_min_warning': 0.30,          # IC warning threshold
    'ic_min_critical': 0.20,         # IC critical threshold
    'accuracy_min_warning': 0.52,    # Direction accuracy warning
    'accuracy_min_critical': 0.50,   # Direction accuracy critical
    'sharpe_min_warning': 0.5,       # Sharpe ratio warning
    'sharpe_min_critical': 0.0,      # Sharpe ratio critical (can be negative)
    'calmar_min_warning': 0.3,       # Calmar ratio warning
    'sample_size': 90,               # Minimum samples for metric calculation
    'lookback_days': 500,            # Days of history to use
}

# Alert severity levels
ALERT_LEVELS = {
    'INFO': 0,
    'WARNING': 1,
    'CRITICAL': 2
}


# ===================================================================
# MONITORING FUNCTIONS
# ===================================================================

def find_trained_symbols() -> List[str]:
    """Discover all trained symbols from saved_models directory."""
    saved_models_root = get_saved_models_root()
    if not saved_models_root.exists():
        logger.warning(f"saved_models directory not found: {saved_models_root}")
        return []

    symbols = []
    for item in saved_models_root.iterdir():
        if item.is_dir() and item.name.isupper():
            has_models = any(
                (item / subdir).exists()
                for subdir in ['regressor', 'classifiers', 'gbm', 'quantile']
            )
            if has_models:
                symbols.append(item.name)

    return sorted(symbols)


def load_feature_columns(symbol: str) -> Optional[List[str]]:
    """Load canonical feature columns for symbol."""
    paths = ModelPaths(symbol)

    if paths.feature_columns.exists():
        try:
            return pickle.load(open(paths.feature_columns, 'rb'))
        except Exception as e:
            logger.error(f"Failed to load feature columns for {symbol}: {e}")
            return None

    return None


def fetch_symbol_data(symbol: str, days: int = 500) -> Optional[pd.DataFrame]:
    """Fetch stock data for symbol."""
    try:
        df = fetch_stock_data(symbol, period='max')
        if len(df) < days:
            logger.warning(f"{symbol}: Insufficient data ({len(df)} days < {days} days)")
            return None
        return df.tail(days)
    except Exception as e:
        logger.error(f"Failed to fetch data for {symbol}: {e}")
        return None


def load_model(symbol: str):
    """Load trained model."""
    try:
        from tensorflow.keras.models import load_model as keras_load
        paths = ModelPaths(symbol)

        if paths.regressor.model.exists():
            return keras_load(str(paths.regressor.model))

        logger.error(f"Model file not found for {symbol}")
        return None
    except Exception as e:
        logger.error(f"Failed to load model for {symbol}: {e}")
        return None


def load_scalers(symbol: str) -> Tuple[Optional, Optional]:
    """Load feature and target scalers."""
    try:
        paths = ModelPaths(symbol)

        feature_scaler = None
        if paths.regressor.feature_scaler.exists():
            feature_scaler = pickle.load(open(paths.regressor.feature_scaler, 'rb'))

        target_scaler = None
        if paths.regressor.target_scaler_robust.exists():
            target_scaler = pickle.load(open(paths.regressor.target_scaler_robust, 'rb'))
        elif paths.regressor.target_scaler.exists():
            target_scaler = pickle.load(open(paths.regressor.target_scaler, 'rb'))

        return feature_scaler, target_scaler
    except Exception as e:
        logger.error(f"Failed to load scalers for {symbol}: {e}")
        return None, None


def engineer_features_safe(df: pd.DataFrame, feature_cols: List[str]) -> Optional[pd.DataFrame]:
    """Engineer features with error handling."""
    try:
        df_features = engineer_features(df, include_sentiment=True)

        if len(df_features) < len(feature_cols):
            logger.warning(f"Feature engineering produced {len(df_features)} features, expected {len(feature_cols)}")
            return None

        missing_cols = set(feature_cols) - set(df_features.columns)
        if missing_cols:
            logger.warning(f"Missing features: {missing_cols}")
            return None

        return df_features[feature_cols]
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        return None


def run_predictions(model, X: np.ndarray, sequence_length: int = 90) -> Optional[np.ndarray]:
    """Run predictions with error handling."""
    try:
        if len(X) < sequence_length:
            logger.warning(f"Insufficient data for sequences. Have {len(X)}, need {sequence_length}")
            return None

        X_seq = np.array([X[i:i+sequence_length] for i in range(len(X) - sequence_length)])

        if X_seq.shape[0] == 0:
            logger.warning("No sequences created for prediction")
            return None

        predictions = model.predict(X_seq, verbose=0)
        return predictions.flatten()
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return None


def calculate_returns(df: pd.DataFrame) -> np.ndarray:
    """Calculate daily log returns from price data."""
    prices = df['Close'].values
    log_returns = np.diff(np.log(prices))
    return log_returns


def calculate_ic(predictions: np.ndarray, returns: np.ndarray) -> Tuple[float, bool]:
    """
    Calculate Information Coefficient (Spearman correlation).

    Measures how well predictions rank with actual returns.
    Range: [-1, 1]
    - 0.5+: Excellent prediction
    - 0.3-0.5: Good prediction
    - 0.2-0.3: Acceptable
    - <0.2: Weak prediction
    - Negative: Reverse prediction
    """
    if len(predictions) < MONITORING_THRESHOLDS['sample_size']:
        return 0.0, False

    try:
        # Remove NaN and infinite values
        mask = np.isfinite(predictions) & np.isfinite(returns)
        if np.sum(mask) < MONITORING_THRESHOLDS['sample_size']:
            return 0.0, False

        pred_valid = predictions[mask]
        returns_valid = returns[mask]

        # Spearman correlation as IC
        from scipy.stats import spearmanr
        ic, p_value = spearmanr(pred_valid, returns_valid)
        return float(ic), p_value < 0.05
    except Exception as e:
        logger.error(f"Failed to calculate IC: {e}")
        return 0.0, False


def calculate_directional_accuracy(predictions: np.ndarray, returns: np.ndarray) -> float:
    """
    Calculate directional accuracy (does sign match?).

    Measures if prediction direction matches actual return direction.
    Range: [0, 1] or 0.5 for random
    - 0.55+: Good
    - 0.52-0.55: Acceptable
    - 0.50: Random (break-even)
    - <0.50: Worse than random
    """
    if len(predictions) < MONITORING_THRESHOLDS['sample_size']:
        return 0.5

    try:
        mask = np.isfinite(predictions) & np.isfinite(returns)
        if np.sum(mask) < MONITORING_THRESHOLDS['sample_size']:
            return 0.5

        pred_valid = predictions[mask]
        returns_valid = returns[mask]

        # Count correct direction predictions
        correct = np.sum(np.sign(pred_valid) == np.sign(returns_valid))
        accuracy = correct / len(pred_valid)
        return float(accuracy)
    except Exception as e:
        logger.error(f"Failed to calculate directional accuracy: {e}")
        return 0.5


def calculate_sharpe_ratio(predictions: np.ndarray, returns: np.ndarray, risk_free_rate: float = 0.02) -> float:
    """
    Calculate Sharpe ratio for strategy using predictions as positions.

    Annualized: assumes 252 trading days per year.
    """
    if len(predictions) < MONITORING_THRESHOLDS['sample_size']:
        return 0.0

    try:
        mask = np.isfinite(predictions) & np.isfinite(returns)
        if np.sum(mask) < MONITORING_THRESHOLDS['sample_size']:
            return 0.0

        pred_valid = predictions[mask]
        returns_valid = returns[mask]

        # Normalize predictions to [-1, 1] for position sizing
        pred_min, pred_max = np.min(pred_valid), np.max(pred_valid)
        if pred_max - pred_min == 0:
            positions = np.zeros_like(pred_valid)
        else:
            positions = 2 * (pred_valid - pred_min) / (pred_max - pred_min) - 1

        # Strategy returns
        strategy_returns = positions * returns_valid

        # Daily Sharpe
        mean_ret = np.mean(strategy_returns)
        std_ret = np.std(strategy_returns)

        if std_ret == 0:
            return 0.0

        daily_sharpe = mean_ret / std_ret
        annual_sharpe = daily_sharpe * np.sqrt(252)

        # Adjust for risk-free rate
        daily_rf = risk_free_rate / 252
        annual_sharpe -= (daily_rf * np.sqrt(252) / std_ret)

        return float(annual_sharpe)
    except Exception as e:
        logger.error(f"Failed to calculate Sharpe ratio: {e}")
        return 0.0


def run_daily_monitoring(symbol: str) -> Dict:
    """Run daily monitoring for a single symbol."""
    logger.info(f"Monitoring {symbol}...")

    monitoring = {
        'symbol': symbol,
        'timestamp': datetime.now().isoformat(),
        'status': 'OK',
        'metrics': {},
        'alerts': []
    }

    # Load feature columns
    feature_cols = load_feature_columns(symbol)
    if feature_cols is None:
        monitoring['status'] = 'CRITICAL'
        monitoring['alerts'].append({
            'level': 'CRITICAL',
            'message': 'Failed to load feature columns'
        })
        return monitoring

    # Fetch data
    df = fetch_symbol_data(symbol, days=MONITORING_THRESHOLDS['lookback_days'])
    if df is None or len(df) < MONITORING_THRESHOLDS['sample_size']:
        monitoring['status'] = 'CRITICAL'
        monitoring['alerts'].append({
            'level': 'CRITICAL',
            'message': f'Insufficient data ({len(df) if df is not None else 0} days)'
        })
        return monitoring

    # Engineer features
    df_features = engineer_features_safe(df, feature_cols)
    if df_features is None:
        monitoring['status'] = 'CRITICAL'
        monitoring['alerts'].append({
            'level': 'CRITICAL',
            'message': 'Feature engineering failed'
        })
        return monitoring

    # Load model and scalers
    model = load_model(symbol)
    if model is None:
        monitoring['status'] = 'CRITICAL'
        monitoring['alerts'].append({
            'level': 'CRITICAL',
            'message': 'Model load failed'
        })
        return monitoring

    feature_scaler, target_scaler = load_scalers(symbol)
    if feature_scaler is None or target_scaler is None:
        monitoring['status'] = 'CRITICAL'
        monitoring['alerts'].append({
            'level': 'CRITICAL',
            'message': 'Scaler load failed'
        })
        return monitoring

    # Run predictions
    try:
        X_scaled = feature_scaler.transform(df_features.values)
    except Exception as e:
        monitoring['status'] = 'CRITICAL'
        monitoring['alerts'].append({
            'level': 'CRITICAL',
            'message': f'Feature scaling failed: {e}'
        })
        return monitoring

    predictions = run_predictions(model, X_scaled)
    if predictions is None:
        monitoring['status'] = 'CRITICAL'
        monitoring['alerts'].append({
            'level': 'CRITICAL',
            'message': 'Prediction failed'
        })
        return monitoring

    # Inverse scale predictions
    try:
        predictions_rescaled = target_scaler.inverse_transform(
            predictions.reshape(-1, 1)
        ).flatten()
    except Exception as e:
        logger.warning(f"Failed to inverse scale predictions: {e}")
        predictions_rescaled = predictions

    # Calculate returns (align with predictions)
    returns = calculate_returns(df)
    # Align: predictions start after sequence_length, returns starts after 1st day
    sequence_length = 90
    if len(returns) > len(predictions):
        returns = returns[-(len(predictions)):]
    elif len(predictions) > len(returns):
        predictions_rescaled = predictions_rescaled[:len(returns)]

    # Calculate metrics
    ic, ic_significant = calculate_ic(predictions_rescaled, returns)
    direction_accuracy = calculate_directional_accuracy(predictions_rescaled, returns)
    sharpe_ratio = calculate_sharpe_ratio(predictions_rescaled, returns)

    monitoring['metrics'] = {
        'ic': float(ic),
        'ic_significant': ic_significant,
        'direction_accuracy': float(direction_accuracy),
        'sharpe_ratio': float(sharpe_ratio),
        'prediction_count': len(predictions),
        'return_count': len(returns)
    }

    # Check thresholds and generate alerts
    if ic < MONITORING_THRESHOLDS['ic_min_critical']:
        monitoring['status'] = 'CRITICAL'
        monitoring['alerts'].append({
            'level': 'CRITICAL',
            'metric': 'ic',
            'value': ic,
            'threshold': MONITORING_THRESHOLDS['ic_min_critical'],
            'message': f'IC below critical threshold ({ic:.3f} < {MONITORING_THRESHOLDS["ic_min_critical"]})'
        })
    elif ic < MONITORING_THRESHOLDS['ic_min_warning']:
        monitoring['status'] = 'WARNING'
        monitoring['alerts'].append({
            'level': 'WARNING',
            'metric': 'ic',
            'value': ic,
            'threshold': MONITORING_THRESHOLDS['ic_min_warning'],
            'message': f'IC below warning threshold ({ic:.3f} < {MONITORING_THRESHOLDS["ic_min_warning"]})'
        })

    if direction_accuracy < MONITORING_THRESHOLDS['accuracy_min_critical']:
        monitoring['status'] = 'CRITICAL'
        monitoring['alerts'].append({
            'level': 'CRITICAL',
            'metric': 'direction_accuracy',
            'value': direction_accuracy,
            'threshold': MONITORING_THRESHOLDS['accuracy_min_critical'],
            'message': f'Direction accuracy below critical ({direction_accuracy:.1%} < {MONITORING_THRESHOLDS["accuracy_min_critical"]:.1%})'
        })
    elif direction_accuracy < MONITORING_THRESHOLDS['accuracy_min_warning']:
        monitoring['status'] = 'WARNING'
        monitoring['alerts'].append({
            'level': 'WARNING',
            'metric': 'direction_accuracy',
            'value': direction_accuracy,
            'threshold': MONITORING_THRESHOLDS['accuracy_min_warning'],
            'message': f'Direction accuracy below warning ({direction_accuracy:.1%} < {MONITORING_THRESHOLDS["accuracy_min_warning"]:.1%})'
        })

    logger.info(f"{symbol}: IC={ic:.3f}, Dir.Acc={direction_accuracy:.1%}, Sharpe={sharpe_ratio:.3f}")
    return monitoring


def generate_monitoring_report(results: List[Dict]) -> str:
    """Generate human-readable monitoring report."""
    report = []
    report.append("=" * 100)
    report.append(f"AI-STOCKS DAILY MONITORING REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 100)
    report.append("")

    # Summary statistics
    statuses = [r['status'] for r in results]
    critical_count = statuses.count('CRITICAL')
    warning_count = statuses.count('WARNING')
    ok_count = statuses.count('OK')

    report.append("SUMMARY")
    report.append("-" * 100)
    report.append(f"Total Models: {len(results)}")
    report.append(f"  OK:       {ok_count}")
    report.append(f"  WARNING:  {warning_count}")
    report.append(f"  CRITICAL: {critical_count}")
    report.append("")

    # Metrics summary
    if results:
        ics = [r['metrics'].get('ic', 0) for r in results if r['metrics']]
        accs = [r['metrics'].get('direction_accuracy', 0.5) for r in results if r['metrics']]
        sharpes = [r['metrics'].get('sharpe_ratio', 0) for r in results if r['metrics']]

        report.append("AGGREGATE METRICS")
        report.append("-" * 100)
        if ics:
            report.append(f"IC (Spearman Correlation)")
            report.append(f"  Mean:   {np.mean(ics):.3f}")
            report.append(f"  Median: {np.median(ics):.3f}")
            report.append(f"  Min:    {np.min(ics):.3f}")
            report.append(f"  Max:    {np.max(ics):.3f}")
        if accs:
            report.append(f"Direction Accuracy")
            report.append(f"  Mean:   {np.mean(accs):.1%}")
            report.append(f"  Median: {np.median(accs):.1%}")
            report.append(f"  Min:    {np.min(accs):.1%}")
            report.append(f"  Max:    {np.max(accs):.1%}")
        if sharpes:
            report.append(f"Sharpe Ratio (Annualized)")
            report.append(f"  Mean:   {np.mean(sharpes):.3f}")
            report.append(f"  Median: {np.median(sharpes):.3f}")
            report.append(f"  Min:    {np.min(sharpes):.3f}")
            report.append(f"  Max:    {np.max(sharpes):.3f}")
        report.append("")

    # Detailed results
    report.append("DETAILED RESULTS")
    report.append("-" * 100)
    report.append(f"{'Symbol':<10} {'Status':<10} {'IC':<10} {'Dir.Acc':<10} {'Sharpe':<10} {'Notes'}")
    report.append("-" * 100)

    for result in sorted(results, key=lambda x: (x['status'], x['symbol']), reverse=True):
        symbol = result['symbol']
        status = result['status']
        metrics = result['metrics']

        ic = metrics.get('ic', 'N/A')
        dir_acc = metrics.get('direction_accuracy', 'N/A')
        sharpe = metrics.get('sharpe_ratio', 'N/A')

        ic_str = f"{ic:.3f}" if isinstance(ic, float) else ic
        dir_acc_str = f"{dir_acc:.1%}" if isinstance(dir_acc, float) else dir_acc
        sharpe_str = f"{sharpe:.3f}" if isinstance(sharpe, float) else sharpe

        notes = []
        if result['alerts']:
            alert_levels = [a.get('level', 'INFO') for a in result['alerts']]
            if 'CRITICAL' in alert_levels:
                notes.append("CRITICAL ALERTS")
            elif 'WARNING' in alert_levels:
                notes.append("HAS WARNINGS")

        notes_str = ", ".join(notes) if notes else ""

        report.append(f"{symbol:<10} {status:<10} {ic_str:<10} {dir_acc_str:<10} {sharpe_str:<10} {notes_str}")

    # Alerts section
    all_alerts = []
    for result in results:
        if result['alerts']:
            for alert in result['alerts']:
                alert['symbol'] = result['symbol']
                all_alerts.append(alert)

    if all_alerts:
        report.append("\n" + "=" * 100)
        report.append("ALERTS")
        report.append("=" * 100)
        for alert in sorted(all_alerts, key=lambda x: (ALERT_LEVELS.get(x['level'], 0), x['symbol']), reverse=True):
            level = alert['level']
            symbol = alert['symbol']
            message = alert.get('message', '')
            report.append(f"[{level}] {symbol}: {message}")

    report.append("\n" + "=" * 100)
    return "\n".join(report)


def main():
    """Main monitoring routine."""
    logger.info("Starting daily model monitoring...")

    symbols = find_trained_symbols()
    if not symbols:
        logger.warning("No trained symbols found. Exiting.")
        return

    logger.info(f"Found {len(symbols)} trained symbols: {', '.join(symbols)}")

    results = []
    for symbol in symbols:
        result = run_daily_monitoring(symbol)
        results.append(result)

    # Generate report
    report = generate_monitoring_report(results)
    print(report)

    # Save report
    report_file = log_dir / f"monitoring_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {report_file}")

    # Save JSON results
    json_file = log_dir / f"monitoring_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(json_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {json_file}")

    # Return exit code based on worst status
    worst_status = max(r['status'] for r in results)
    exit_code = {
        'OK': 0,
        'WARNING': 1,
        'CRITICAL': 2
    }.get(worst_status, 1)

    logger.info(f"Monitoring complete. Status: {worst_status}")
    return exit_code


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
