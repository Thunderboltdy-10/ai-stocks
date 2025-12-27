#!/usr/bin/env python3
"""
Health Check System for AI-Stocks Models

Validates model health across all trained symbols:
- Loads models and checks file integrity
- Runs predictions on recent data
- Detects variance collapse, NaN values, and prediction bias
- Reports health status: OK, WARNING, CRITICAL

Run daily to monitor model degradation.
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
log_file = log_dir / f"health_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
# HEALTH CHECK THRESHOLDS
# ===================================================================

HEALTH_THRESHOLDS = {
    'variance_min': 0.005,           # Variance collapse if std < 0.005
    'nan_max': 0,                    # No NaN values allowed
    'inf_max': 0,                    # No infinite values allowed
    'bias_max': 0.80,                # Positive/negative ratio max
    'ic_min_warning': 0.30,          # IC warning threshold
    'ic_min_critical': 0.20,         # IC critical threshold
    'accuracy_min_warning': 0.52,    # Direction accuracy warning
    'accuracy_min_critical': 0.50,   # Direction accuracy critical
    'sample_size': 250,              # Minimum recent samples for validation
}


# ===================================================================
# HEALTH CHECK FUNCTIONS
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
            # Check if symbol has any model subdirectories
            has_models = any(
                (item / subdir).exists()
                for subdir in ['regressor', 'classifiers', 'gbm', 'quantile']
            )
            if has_models:
                symbols.append(item.name)

    return sorted(symbols)


def check_model_files(symbol: str) -> Tuple[bool, List[str]]:
    """Check if all required model files exist."""
    paths = ModelPaths(symbol)
    issues = []

    # Check regressor
    if not paths.regressor.model.exists():
        issues.append(f"Regressor model missing: {paths.regressor.model}")
    if not paths.regressor.target_scaler.exists() and not paths.regressor.target_scaler_robust.exists():
        issues.append(f"Target scaler missing for regressor")
    if not paths.regressor.feature_scaler.exists():
        issues.append(f"Feature scaler missing for regressor")

    # Check feature columns file
    if not paths.feature_columns.exists():
        issues.append(f"Feature columns file missing: {paths.feature_columns}")

    return len(issues) == 0, issues


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


def fetch_recent_data(symbol: str, days: int = 500) -> Optional[pd.DataFrame]:
    """Fetch recent stock data for health check."""
    try:
        df = fetch_stock_data(symbol, period='max')
        if len(df) < HEALTH_THRESHOLDS['sample_size']:
            logger.warning(f"{symbol}: Insufficient data ({len(df)} days). Need {HEALTH_THRESHOLDS['sample_size']}")
            return None
        return df.tail(days)
    except Exception as e:
        logger.error(f"Failed to fetch data for {symbol}: {e}")
        return None


def engineer_features_safe(df: pd.DataFrame, feature_cols: List[str]) -> Optional[pd.DataFrame]:
    """Engineer features with error handling."""
    try:
        df_features = engineer_features(df, include_sentiment=True)

        # Check feature count
        if len(df_features) < len(feature_cols):
            logger.warning(f"Feature engineering produced {len(df_features)} features, expected {len(feature_cols)}")
            return None

        # Select only required columns
        missing_cols = set(feature_cols) - set(df_features.columns)
        if missing_cols:
            logger.warning(f"Missing features: {missing_cols}")
            return None

        return df_features[feature_cols]
    except Exception as e:
        logger.error(f"Feature engineering failed: {e}")
        return None


def load_model(symbol: str):
    """Load trained model with error handling."""
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

        # Load feature scaler
        feature_scaler = None
        if paths.regressor.feature_scaler.exists():
            feature_scaler = pickle.load(open(paths.regressor.feature_scaler, 'rb'))

        # Load target scaler (prefer robust)
        target_scaler = None
        if paths.regressor.target_scaler_robust.exists():
            target_scaler = pickle.load(open(paths.regressor.target_scaler_robust, 'rb'))
        elif paths.regressor.target_scaler.exists():
            target_scaler = pickle.load(open(paths.regressor.target_scaler, 'rb'))

        return feature_scaler, target_scaler
    except Exception as e:
        logger.error(f"Failed to load scalers for {symbol}: {e}")
        return None, None


def run_predictions(model, X: np.ndarray, sequence_length: int = 90) -> Optional[np.ndarray]:
    """Run predictions with error handling."""
    try:
        # Create sequences for prediction
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


def calculate_prediction_metrics(predictions: np.ndarray) -> Dict:
    """Calculate key metrics for predictions."""
    metrics = {
        'count': len(predictions),
        'mean': float(np.mean(predictions)),
        'std': float(np.std(predictions)),
        'min': float(np.min(predictions)),
        'max': float(np.max(predictions)),
        'nan_count': int(np.sum(np.isnan(predictions))),
        'inf_count': int(np.sum(np.isinf(predictions))),
    }

    # Calculate positive/negative ratio
    positive = np.sum(predictions > 0)
    negative = np.sum(predictions <= 0)
    metrics['positive_ratio'] = float(positive / len(predictions)) if len(predictions) > 0 else 0.0
    metrics['negative_ratio'] = float(negative / len(predictions)) if len(predictions) > 0 else 0.0

    return metrics


def check_variance_collapse(metrics: Dict) -> Tuple[str, str]:
    """Check if model has variance collapse."""
    std = metrics['std']
    threshold = HEALTH_THRESHOLDS['variance_min']

    if std < threshold:
        return 'CRITICAL', f"Variance collapse detected (std={std:.6f} < {threshold})"
    elif std < threshold * 2:
        return 'WARNING', f"Low variance (std={std:.6f}, threshold={threshold})"

    return 'OK', f"Variance healthy (std={std:.6f})"


def check_nan_values(metrics: Dict) -> Tuple[str, str]:
    """Check for NaN or infinite values."""
    nan_count = metrics['nan_count']
    inf_count = metrics['inf_count']

    if nan_count > 0 or inf_count > 0:
        return 'CRITICAL', f"Found {nan_count} NaN and {inf_count} infinite values"

    return 'OK', "No NaN or infinite values"


def check_prediction_bias(metrics: Dict) -> Tuple[str, str]:
    """Check for positive/negative prediction bias."""
    positive_ratio = metrics['positive_ratio']
    negative_ratio = metrics['negative_ratio']

    bias_threshold = HEALTH_THRESHOLDS['bias_max']
    max_ratio = max(positive_ratio, negative_ratio)

    if max_ratio > bias_threshold:
        direction = "positive" if positive_ratio > negative_ratio else "negative"
        return 'WARNING', f"Prediction bias detected ({direction}: {max_ratio:.1%} > {bias_threshold:.1%})"
    elif max_ratio > bias_threshold * 0.95:
        return 'WARNING', f"Approaching bias threshold ({max_ratio:.1%})"

    return 'OK', f"Balanced predictions (+{metrics['positive_ratio']:.1%}, -{metrics['negative_ratio']:.1%})"


def check_model_health(symbol: str) -> Dict:
    """Comprehensive health check for a single symbol."""
    logger.info(f"Checking health for {symbol}...")

    health = {
        'symbol': symbol,
        'timestamp': datetime.now().isoformat(),
        'status': 'OK',
        'checks': {},
        'metrics': {}
    }

    # Check 1: Model files exist
    files_ok, file_issues = check_model_files(symbol)
    if not files_ok:
        health['status'] = 'CRITICAL'
        health['checks']['model_files'] = {
            'status': 'CRITICAL',
            'issues': file_issues
        }
        logger.error(f"{symbol}: Model files check FAILED")
        return health

    health['checks']['model_files'] = {'status': 'OK'}

    # Check 2: Feature columns can be loaded
    feature_cols = load_feature_columns(symbol)
    if feature_cols is None:
        health['status'] = 'CRITICAL'
        health['checks']['feature_columns'] = {
            'status': 'CRITICAL',
            'message': 'Failed to load feature columns'
        }
        logger.error(f"{symbol}: Feature columns load FAILED")
        return health

    health['checks']['feature_columns'] = {
        'status': 'OK',
        'feature_count': len(feature_cols)
    }

    # Check 3: Fetch recent data
    df = fetch_recent_data(symbol)
    if df is None or len(df) < HEALTH_THRESHOLDS['sample_size']:
        health['status'] = 'CRITICAL'
        health['checks']['data_fetch'] = {
            'status': 'CRITICAL',
            'message': f'Insufficient data (need {HEALTH_THRESHOLDS["sample_size"]} days)'
        }
        logger.error(f"{symbol}: Data fetch FAILED")
        return health

    health['checks']['data_fetch'] = {
        'status': 'OK',
        'data_points': len(df)
    }

    # Check 4: Engineer features
    df_features = engineer_features_safe(df, feature_cols)
    if df_features is None:
        health['status'] = 'CRITICAL'
        health['checks']['feature_engineering'] = {
            'status': 'CRITICAL',
            'message': 'Feature engineering failed'
        }
        logger.error(f"{symbol}: Feature engineering FAILED")
        return health

    health['checks']['feature_engineering'] = {
        'status': 'OK',
        'shape': df_features.shape
    }

    # Check 5: Load model and scalers
    model = load_model(symbol)
    if model is None:
        health['status'] = 'CRITICAL'
        health['checks']['model_load'] = {
            'status': 'CRITICAL',
            'message': 'Failed to load model'
        }
        logger.error(f"{symbol}: Model load FAILED")
        return health

    feature_scaler, target_scaler = load_scalers(symbol)
    if feature_scaler is None or target_scaler is None:
        health['status'] = 'CRITICAL'
        health['checks']['scaler_load'] = {
            'status': 'CRITICAL',
            'message': 'Failed to load scalers'
        }
        logger.error(f"{symbol}: Scaler load FAILED")
        return health

    health['checks']['model_load'] = {'status': 'OK'}
    health['checks']['scaler_load'] = {'status': 'OK'}

    # Check 6: Scale features and run predictions
    try:
        X_scaled = feature_scaler.transform(df_features.values)
    except Exception as e:
        health['status'] = 'CRITICAL'
        health['checks']['feature_scaling'] = {
            'status': 'CRITICAL',
            'message': f'Feature scaling failed: {e}'
        }
        logger.error(f"{symbol}: Feature scaling FAILED")
        return health

    health['checks']['feature_scaling'] = {'status': 'OK'}

    predictions = run_predictions(model, X_scaled)
    if predictions is None:
        health['status'] = 'CRITICAL'
        health['checks']['predictions'] = {
            'status': 'CRITICAL',
            'message': 'Prediction failed'
        }
        logger.error(f"{symbol}: Prediction FAILED")
        return health

    # Check 7: Analyze prediction metrics
    metrics = calculate_prediction_metrics(predictions)
    health['metrics'] = metrics

    # Variance check
    var_status, var_msg = check_variance_collapse(metrics)
    health['checks']['variance'] = {'status': var_status, 'message': var_msg}
    if var_status != 'OK':
        health['status'] = var_status if health['status'] == 'OK' else health['status']

    # NaN check
    nan_status, nan_msg = check_nan_values(metrics)
    health['checks']['nan_values'] = {'status': nan_status, 'message': nan_msg}
    if nan_status == 'CRITICAL':
        health['status'] = 'CRITICAL'

    # Bias check
    bias_status, bias_msg = check_prediction_bias(metrics)
    health['checks']['prediction_bias'] = {'status': bias_status, 'message': bias_msg}
    if bias_status != 'OK':
        health['status'] = bias_status if health['status'] == 'OK' else health['status']

    logger.info(f"{symbol}: Health check {health['status']}")
    return health


def generate_health_report(results: List[Dict]) -> str:
    """Generate human-readable health report."""
    report = []
    report.append("=" * 80)
    report.append(f"AI-STOCKS MODEL HEALTH CHECK REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    report.append("")

    # Summary
    statuses = [r['status'] for r in results]
    critical_count = statuses.count('CRITICAL')
    warning_count = statuses.count('WARNING')
    ok_count = statuses.count('OK')

    report.append("SUMMARY")
    report.append("-" * 80)
    report.append(f"Total Models: {len(results)}")
    report.append(f"  OK:       {ok_count}")
    report.append(f"  WARNING:  {warning_count}")
    report.append(f"  CRITICAL: {critical_count}")
    report.append("")

    # Detailed results
    report.append("DETAILED RESULTS")
    report.append("-" * 80)

    for result in sorted(results, key=lambda x: x['status']):
        symbol = result['symbol']
        status = result['status']
        status_icon = {
            'OK': '✓',
            'WARNING': '⚠',
            'CRITICAL': '✗'
        }.get(status, '?')

        report.append(f"\n[{status_icon}] {symbol:8} - {status}")

        for check_name, check_result in result['checks'].items():
            check_status = check_result.get('status', 'UNKNOWN')
            if 'message' in check_result:
                report.append(f"    {check_name:20} {check_status:10} {check_result['message']}")
            elif 'issues' in check_result:
                report.append(f"    {check_name:20} {check_status:10}")
                for issue in check_result['issues']:
                    report.append(f"      - {issue}")
            else:
                report.append(f"    {check_name:20} {check_status:10}")

        if result['metrics']:
            metrics = result['metrics']
            report.append(f"    Prediction Metrics:")
            report.append(f"      std:    {metrics['std']:.6f}")
            report.append(f"      mean:   {metrics['mean']:.6f}")
            report.append(f"      +ratio: {metrics['positive_ratio']:.1%}")
            report.append(f"      -ratio: {metrics['negative_ratio']:.1%}")

    report.append("\n" + "=" * 80)
    return "\n".join(report)


def main():
    """Main health check routine."""
    logger.info("Starting AI-Stocks health check...")

    symbols = find_trained_symbols()
    if not symbols:
        logger.warning("No trained symbols found. Exiting.")
        return

    logger.info(f"Found {len(symbols)} trained symbols: {', '.join(symbols)}")

    results = []
    for symbol in symbols:
        result = check_model_health(symbol)
        results.append(result)

    # Generate report
    report = generate_health_report(results)
    print(report)

    # Save report
    report_file = log_dir / f"health_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    logger.info(f"Report saved to {report_file}")

    # Save JSON results
    json_file = log_dir / f"health_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
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

    logger.info(f"Health check complete. Status: {worst_status}")
    return exit_code


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
