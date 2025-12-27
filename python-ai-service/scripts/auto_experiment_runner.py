#!/usr/bin/env python3
"""
Automated ML Pipeline Experiment Runner

This script runs a comprehensive pipeline audit for AI-Stocks:
- Clears/refreshes cache per symbol
- Trains regressor + GBM models
- Validates model artifacts
- Runs backtests for 5 fusion modes
- Generates diagnostics and metrics
- Produces root cause analysis

Run with: python scripts/auto_experiment_runner.py --symbols AAPL,MSFT --timestamp 20251215_185847
"""

import argparse
import json
import logging
import os
import pickle
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Canonical symbol list
CANONICAL_SYMBOLS = ['AAPL', 'MSFT', 'AMZN', 'NVDA', 'TSLA', 'HOOD', 'XOM', 'JPM']
FUSION_MODES = ['regressor_only', 'gbm_only', 'gbm_heavy', 'balanced', 'lstm_heavy']

# Set seeds for reproducibility
SEED = 42


def set_global_seeds(seed: int = SEED):
    """Set seeds for reproducibility across all libraries."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
    except ImportError:
        pass
    
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    
    # Set LightGBM/XGBoost seeds via environment
    os.environ['PYTHONHASHSEED'] = str(seed)


def run_command(cmd: List[str], cwd: str = None, timeout: int = 3600) -> Tuple[int, str, str]:
    """Run a subprocess command and capture output."""
    logger.info(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            cwd=cwd or str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out after {timeout}s")
        return -1, "", "Timeout"
    except Exception as e:
        logger.error(f"Command failed: {e}")
        return -1, "", str(e)


def get_gpu_info() -> Dict[str, Any]:
    """Get GPU utilization info."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(',')
            if len(parts) >= 6:
                return {
                    'name': parts[0].strip(),
                    'memory_total_mb': int(parts[1].strip()),
                    'memory_used_mb': int(parts[2].strip()),
                    'memory_free_mb': int(parts[3].strip()),
                    'utilization_pct': int(parts[4].strip()),
                    'temperature_c': int(parts[5].strip()),
                }
    except Exception as e:
        logger.warning(f"Failed to get GPU info: {e}")
    return {}


def clear_cache_for_symbol(symbol: str) -> bool:
    """Clear cache for a specific symbol."""
    try:
        from data.cache_manager import DataCacheManager
        cache_manager = DataCacheManager()
        cache_manager.clear_cache(symbol)
        logger.info(f"Cleared cache for {symbol}")
        return True
    except Exception as e:
        logger.error(f"Failed to clear cache for {symbol}: {e}")
        return False


def validate_model_artifacts(symbol: str) -> Dict[str, Any]:
    """Validate model artifacts for a symbol."""
    results = {
        'symbol': symbol,
        'timestamp': datetime.now().isoformat(),
        'issues': [],
        'warnings': [],
        'artifacts': {}
    }
    
    saved_models_dir = PROJECT_ROOT / 'saved_models'
    symbol_dir = saved_models_dir / symbol
    
    # Check for required artifacts
    required_artifacts = {
        'feature_columns': symbol_dir / 'feature_columns.pkl',
        'target_metadata': symbol_dir / 'target_metadata.pkl',
        'regressor_weights': None,  # Multiple possible locations
        'feature_scaler': None,     # Multiple possible locations
    }
    
    # Check symbol directory exists
    if not symbol_dir.exists():
        # Check for legacy flat structure
        legacy_weights = saved_models_dir / f'{symbol}_1d_regressor_final.weights.h5'
        if legacy_weights.exists():
            results['artifacts']['structure'] = 'legacy_flat'
            required_artifacts['regressor_weights'] = legacy_weights
            required_artifacts['feature_scaler'] = saved_models_dir / f'{symbol}_1d_regressor_final_feature_scaler.pkl'
        else:
            results['issues'].append(f"No model directory found for {symbol}")
            return results
    else:
        results['artifacts']['structure'] = 'organized'
        
        # Check new organized structure
        regressor_dir = symbol_dir / 'regressor'
        if regressor_dir.exists():
            required_artifacts['regressor_weights'] = regressor_dir / 'weights.h5'
            required_artifacts['feature_scaler'] = regressor_dir / 'feature_scaler.pkl'
        else:
            # Check for direct files in symbol_dir
            required_artifacts['regressor_weights'] = symbol_dir / 'regressor_weights.h5'
            required_artifacts['feature_scaler'] = symbol_dir / 'feature_scaler.pkl'
    
    # Validate each artifact
    for name, path in required_artifacts.items():
        if path is None:
            continue
        if path.exists():
            results['artifacts'][name] = {
                'path': str(path),
                'exists': True,
                'size_bytes': path.stat().st_size
            }
            
            # Try to load and validate
            if name == 'feature_columns':
                try:
                    with open(path, 'rb') as f:
                        feature_cols = pickle.load(f)
                    results['artifacts'][name]['count'] = len(feature_cols)
                    
                    # Check against expected count
                    from data.feature_engineer import EXPECTED_FEATURE_COUNT
                    if len(feature_cols) != EXPECTED_FEATURE_COUNT:
                        results['warnings'].append(
                            f"Feature count mismatch: {len(feature_cols)} vs expected {EXPECTED_FEATURE_COUNT}"
                        )
                except Exception as e:
                    results['issues'].append(f"Failed to load {name}: {e}")
            
            elif name == 'target_metadata':
                try:
                    with open(path, 'rb') as f:
                        metadata = pickle.load(f)
                    results['artifacts'][name]['keys'] = list(metadata.keys()) if isinstance(metadata, dict) else []
                except Exception as e:
                    results['issues'].append(f"Failed to load {name}: {e}")
        else:
            results['artifacts'][name] = {'path': str(path), 'exists': False}
            results['issues'].append(f"Missing artifact: {name} at {path}")
    
    # Check GBM models
    gbm_dir = symbol_dir / 'gbm'
    if gbm_dir.exists():
        results['artifacts']['gbm'] = {
            'exists': True,
            'lgb': (gbm_dir / 'lgb_reg.joblib').exists(),
            'xgb': (gbm_dir / 'xgb_reg.joblib').exists(),
        }
    else:
        results['artifacts']['gbm'] = {'exists': False}
        results['warnings'].append("No GBM models found")
    
    return results


def validate_feature_alignment(symbol: str) -> Dict[str, Any]:
    """Validate feature alignment between saved artifacts and current feature engineering."""
    results = {
        'symbol': symbol,
        'aligned': False,
        'issues': [],
        'details': {}
    }
    
    try:
        from data.feature_engineer import engineer_features, get_feature_columns, EXPECTED_FEATURE_COUNT
        from data.data_fetcher import fetch_stock_data
        
        # Load saved feature columns
        saved_models_dir = PROJECT_ROOT / 'saved_models'
        symbol_dir = saved_models_dir / symbol
        
        feature_cols_path = symbol_dir / 'feature_columns.pkl'
        if not feature_cols_path.exists():
            # Try legacy path
            feature_cols_path = saved_models_dir / f'{symbol}_1d_regressor_final_features.pkl'
        
        if not feature_cols_path.exists():
            results['issues'].append("No saved feature_columns.pkl found")
            return results
        
        with open(feature_cols_path, 'rb') as f:
            saved_features = pickle.load(f)
        
        results['details']['saved_feature_count'] = len(saved_features)
        
        # Get current canonical features
        current_features = get_feature_columns(include_sentiment=True)
        results['details']['current_feature_count'] = len(current_features)
        results['details']['expected_feature_count'] = EXPECTED_FEATURE_COUNT
        
        # Check alignment
        saved_set = set(saved_features)
        current_set = set(current_features)
        
        missing_in_current = saved_set - current_set
        extra_in_current = current_set - saved_set
        
        if missing_in_current:
            results['issues'].append(f"Saved features not in current: {list(missing_in_current)[:5]}...")
        if extra_in_current:
            results['issues'].append(f"Current features not in saved: {list(extra_in_current)[:5]}...")
        
        if not missing_in_current and not extra_in_current:
            # Check order
            if list(saved_features) == list(current_features):
                results['aligned'] = True
            else:
                results['issues'].append("Feature order mismatch")
        
        results['details']['missing_in_current'] = len(missing_in_current)
        results['details']['extra_in_current'] = len(extra_in_current)
        
    except Exception as e:
        results['issues'].append(f"Validation failed: {traceback.format_exc()}")
    
    return results


def run_training_regressor(symbol: str, output_dir: Path, epochs: int = 300) -> Dict[str, Any]:
    """Run regressor training for a symbol."""
    results = {
        'symbol': symbol,
        'model': 'regressor',
        'success': False,
        'output_dir': str(output_dir),
        'metrics': {},
        'errors': []
    }
    
    gpu_before = get_gpu_info()
    results['gpu_before'] = gpu_before
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / 'training' / 'train_1d_regressor_final.py'),
        '--symbol', symbol,
        '--epochs', str(epochs),
        '--batch_size', '64',
        '--seed', str(SEED),
    ]
    
    start_time = datetime.now()
    returncode, stdout, stderr = run_command(cmd, timeout=7200)  # 2 hour timeout
    end_time = datetime.now()
    
    results['duration_seconds'] = (end_time - start_time).total_seconds()
    results['gpu_after'] = get_gpu_info()
    
    # Save logs
    log_file = output_dir / 'regressor_training.log'
    with open(log_file, 'w') as f:
        f.write(f"=== STDOUT ===\n{stdout}\n\n=== STDERR ===\n{stderr}")
    results['log_file'] = str(log_file)
    
    if returncode == 0:
        results['success'] = True
        # Try to parse metrics from output
        try:
            for line in stdout.split('\n'):
                if 'R²' in line or 'MAE' in line or 'RMSE' in line:
                    results['metrics']['training_output'] = line.strip()
        except:
            pass
    else:
        results['errors'].append(f"Training failed with code {returncode}")
        results['errors'].append(stderr[-2000:] if stderr else "No stderr")
    
    return results


def run_training_gbm(symbol: str, output_dir: Path, num_boost_round: int = 5000) -> Dict[str, Any]:
    """Run GBM training for a symbol."""
    results = {
        'symbol': symbol,
        'model': 'gbm',
        'success': False,
        'output_dir': str(output_dir),
        'metrics': {},
        'errors': []
    }
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / 'training' / 'train_gbm_baseline.py'),
        '--symbol', symbol,
        '--use_cache',
    ]
    
    start_time = datetime.now()
    returncode, stdout, stderr = run_command(cmd, timeout=3600)  # 1 hour timeout
    end_time = datetime.now()
    
    results['duration_seconds'] = (end_time - start_time).total_seconds()
    
    # Save logs
    log_file = output_dir / 'gbm_training.log'
    with open(log_file, 'w') as f:
        f.write(f"=== STDOUT ===\n{stdout}\n\n=== STDERR ===\n{stderr}")
    results['log_file'] = str(log_file)
    
    if returncode == 0:
        results['success'] = True
        # Try to parse metrics from output
        try:
            for line in stdout.split('\n'):
                if 'R²' in line or 'MAE' in line or 'RMSE' in line or 'IC' in line:
                    if 'metrics' not in results:
                        results['metrics'] = {}
                    results['metrics']['training_output'] = line.strip()
        except:
            pass
    else:
        results['errors'].append(f"Training failed with code {returncode}")
        results['errors'].append(stderr[-2000:] if stderr else "No stderr")
    
    return results


def run_backtest(symbol: str, fusion_mode: str, output_dir: Path) -> Dict[str, Any]:
    """Run backtest for a symbol with specific fusion mode."""
    results = {
        'symbol': symbol,
        'fusion_mode': fusion_mode,
        'success': False,
        'output_dir': str(output_dir),
        'metrics': {},
        'errors': []
    }
    
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / 'inference_and_backtest.py'),
        '--symbol', symbol,
        '--fusion_mode', fusion_mode,
        '--use_cache',
    ]
    
    start_time = datetime.now()
    returncode, stdout, stderr = run_command(cmd, timeout=1800)  # 30 min timeout
    end_time = datetime.now()
    
    results['duration_seconds'] = (end_time - start_time).total_seconds()
    
    # Save logs
    log_file = output_dir / f'{fusion_mode}_backtest.log'
    with open(log_file, 'w') as f:
        f.write(f"=== STDOUT ===\n{stdout}\n\n=== STDERR ===\n{stderr}")
    results['log_file'] = str(log_file)
    
    if returncode == 0:
        results['success'] = True
        
        # Try to parse metrics from output
        try:
            for line in stdout.split('\n'):
                line = line.strip()
                if 'Sharpe' in line:
                    results['metrics']['sharpe_line'] = line
                if 'cum_return' in line.lower() or 'cumulative return' in line.lower():
                    results['metrics']['return_line'] = line
                if 'max_drawdown' in line.lower():
                    results['metrics']['drawdown_line'] = line
        except:
            pass
        
        # Try to find and load the backtest pickle
        backtest_dir = PROJECT_ROOT / 'backtest_results'
        if backtest_dir.exists():
            pkl_files = list(backtest_dir.glob(f'{symbol}_backtest_*.pkl'))
            if pkl_files:
                latest_pkl = max(pkl_files, key=lambda p: p.stat().st_mtime)
                try:
                    with open(latest_pkl, 'rb') as f:
                        bt_data = pickle.load(f)
                    if isinstance(bt_data, dict) and 'metrics' in bt_data:
                        results['metrics'].update(bt_data['metrics'])
                    results['backtest_pickle'] = str(latest_pkl)
                except Exception as e:
                    results['warnings'] = [f"Failed to load backtest pickle: {e}"]
    else:
        results['errors'].append(f"Backtest failed with code {returncode}")
        # Include relevant error messages
        for line in stderr.split('\n'):
            if 'Error' in line or 'Exception' in line or 'Traceback' in line:
                results['errors'].append(line.strip())
    
    return results


def compute_diagnostics(symbol: str, results_dir: Path) -> Dict[str, Any]:
    """Compute comprehensive diagnostics for a symbol."""
    diagnostics = {
        'symbol': symbol,
        'timestamp': datetime.now().isoformat(),
        'checks': {}
    }
    
    try:
        # 1. Validate artifacts
        diagnostics['checks']['artifacts'] = validate_model_artifacts(symbol)
        
        # 2. Validate feature alignment
        diagnostics['checks']['feature_alignment'] = validate_feature_alignment(symbol)
        
        # 3. Load and analyze predictions if available
        backtest_dir = PROJECT_ROOT / 'backtest_results'
        if backtest_dir.exists():
            pkl_files = list(backtest_dir.glob(f'{symbol}_backtest_*.pkl'))
            if pkl_files:
                latest_pkl = max(pkl_files, key=lambda p: p.stat().st_mtime)
                try:
                    with open(latest_pkl, 'rb') as f:
                        bt_data = pickle.load(f)
                    
                    if isinstance(bt_data, dict):
                        # Check predictions
                        if 'positions' in bt_data:
                            positions = np.asarray(bt_data['positions'])
                            diagnostics['checks']['positions'] = {
                                'mean': float(np.mean(positions)),
                                'std': float(np.std(positions)),
                                'min': float(np.min(positions)),
                                'max': float(np.max(positions)),
                                'pct_zero': float(np.mean(positions == 0)),
                                'pct_positive': float(np.mean(positions > 0)),
                                'pct_negative': float(np.mean(positions < 0)),
                            }
                        
                        if 'metrics' in bt_data:
                            diagnostics['checks']['backtest_metrics'] = bt_data['metrics']
                        
                        # Check for NaN/Inf
                        nan_checks = {}
                        for key in ['positions', 'equity', 'returns']:
                            if key in bt_data:
                                arr = np.asarray(bt_data[key])
                                nan_checks[key] = {
                                    'has_nan': bool(np.any(np.isnan(arr))),
                                    'has_inf': bool(np.any(np.isinf(arr))),
                                }
                        diagnostics['checks']['nan_inf'] = nan_checks
                        
                except Exception as e:
                    diagnostics['checks']['backtest_load_error'] = str(e)
        
        # 4. Check cache metadata
        cache_dir = PROJECT_ROOT / 'cache' / symbol
        if cache_dir.exists():
            metadata_path = cache_dir / 'metadata.json'
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    cache_meta = json.load(f)
                diagnostics['checks']['cache_metadata'] = cache_meta
        
    except Exception as e:
        diagnostics['error'] = traceback.format_exc()
    
    # Save diagnostics
    diag_file = results_dir / f'diagnostics_{symbol}.json'
    with open(diag_file, 'w') as f:
        json.dump(diagnostics, f, indent=2, default=str)
    
    return diagnostics


def run_symbol_pipeline(
    symbol: str,
    results_dir: Path,
    skip_training: bool = False,
    training_epochs: int = 300
) -> Dict[str, Any]:
    """Run complete pipeline for a single symbol."""
    
    symbol_results = {
        'symbol': symbol,
        'start_time': datetime.now().isoformat(),
        'training': {},
        'backtests': {},
        'diagnostics': {},
        'errors': []
    }
    
    # Create symbol results directory
    symbol_dir = results_dir / 'per_symbol' / symbol
    symbol_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"PROCESSING SYMBOL: {symbol}")
    logger.info(f"{'='*60}")
    
    # Step 1: Clear cache (optional - for fresh start)
    # clear_cache_for_symbol(symbol)
    
    # Step 2: Validate existing artifacts
    logger.info(f"Validating existing artifacts for {symbol}...")
    artifact_validation = validate_model_artifacts(symbol)
    symbol_results['artifact_validation'] = artifact_validation
    
    # Step 3: Training (if not skipping)
    if not skip_training:
        # Train Regressor
        logger.info(f"Training regressor for {symbol}...")
        regressor_results = run_training_regressor(symbol, symbol_dir, epochs=training_epochs)
        symbol_results['training']['regressor'] = regressor_results
        
        # Train GBM
        logger.info(f"Training GBM for {symbol}...")
        gbm_results = run_training_gbm(symbol, symbol_dir)
        symbol_results['training']['gbm'] = gbm_results
    else:
        logger.info(f"Skipping training for {symbol} (--skip_training flag)")
    
    # Step 4: Run backtests for all fusion modes
    for fusion_mode in FUSION_MODES:
        logger.info(f"Running backtest for {symbol} with {fusion_mode}...")
        try:
            backtest_results = run_backtest(symbol, fusion_mode, symbol_dir)
            symbol_results['backtests'][fusion_mode] = backtest_results
        except Exception as e:
            logger.error(f"Backtest failed for {symbol}/{fusion_mode}: {e}")
            symbol_results['backtests'][fusion_mode] = {
                'success': False,
                'error': str(e)
            }
    
    # Step 5: Compute diagnostics
    logger.info(f"Computing diagnostics for {symbol}...")
    diagnostics = compute_diagnostics(symbol, symbol_dir)
    symbol_results['diagnostics'] = diagnostics
    
    symbol_results['end_time'] = datetime.now().isoformat()
    
    # Save symbol results
    results_file = symbol_dir / 'pipeline_results.json'
    with open(results_file, 'w') as f:
        json.dump(symbol_results, f, indent=2, default=str)
    
    return symbol_results


def generate_summary_table(all_results: Dict[str, Any], results_dir: Path) -> pd.DataFrame:
    """Generate summary table comparing fusion modes across symbols."""
    
    rows = []
    for symbol, symbol_data in all_results.get('symbols', {}).items():
        for fusion_mode, bt_data in symbol_data.get('backtests', {}).items():
            if bt_data.get('success', False):
                metrics = bt_data.get('metrics', {})
                row = {
                    'symbol': symbol,
                    'fusion_mode': fusion_mode,
                    'sharpe': metrics.get('sharpe', np.nan),
                    'cum_return': metrics.get('cum_return', np.nan),
                    'max_drawdown': metrics.get('max_drawdown', np.nan),
                    'buy_hold_sharpe': metrics.get('buy_hold_sharpe', np.nan),
                    'alpha': metrics.get('alpha', np.nan),
                    'total_costs': metrics.get('total_transaction_costs', np.nan),
                }
                rows.append(row)
    
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(results_dir / 'summary_table.csv', index=False)
        return df
    return pd.DataFrame()


def generate_root_cause_report(all_results: Dict[str, Any], results_dir: Path) -> str:
    """Generate root cause analysis report."""
    
    report = []
    report.append("# Root Cause Analysis Report")
    report.append(f"\nGenerated: {datetime.now().isoformat()}")
    report.append(f"\nSymbols analyzed: {list(all_results.get('symbols', {}).keys())}")
    
    # Analyze issues
    report.append("\n## Issues Detected")
    
    issues = []
    for symbol, symbol_data in all_results.get('symbols', {}).items():
        # Check artifact issues
        artifacts = symbol_data.get('artifact_validation', {})
        if artifacts.get('issues'):
            for issue in artifacts['issues']:
                issues.append(f"- [{symbol}] Artifact: {issue}")
        
        # Check backtest failures
        for fusion_mode, bt_data in symbol_data.get('backtests', {}).items():
            if not bt_data.get('success', False):
                errors = bt_data.get('errors', ['Unknown error'])
                issues.append(f"- [{symbol}/{fusion_mode}] Backtest: {errors[0]}")
    
    if issues:
        report.extend(issues)
    else:
        report.append("- No critical issues detected")
    
    # Recommendations
    report.append("\n## Recommendations")
    report.append("""
### P0 - Critical Fixes
1. Ensure all symbols have `feature_columns.pkl` matching current feature engineering
2. Verify target alignment (no look-ahead bias)
3. Check backtester transaction cost application

### P1 - High Priority
1. Retrain models with higher epochs if R² is poor
2. Investigate fusion modes with negative Sharpe ratios
3. Validate GBM feature importance alignment

### P2 - Medium Priority  
1. Add more robust error handling to inference pipeline
2. Implement automated feature alignment validation
3. Add rolling R² diagnostics to training output
""")
    
    report_text = '\n'.join(report)
    
    with open(results_dir / 'root_cause_report.md', 'w') as f:
        f.write(report_text)
    
    return report_text


def main():
    parser = argparse.ArgumentParser(description='Automated ML Pipeline Experiment Runner')
    parser.add_argument('--symbols', type=str, default=','.join(CANONICAL_SYMBOLS),
                        help='Comma-separated list of symbols to process')
    parser.add_argument('--timestamp', type=str, default=datetime.now().strftime('%Y%m%d_%H%M%S'),
                        help='Timestamp for results directory')
    parser.add_argument('--skip_training', action='store_true',
                        help='Skip training and only run backtests')
    parser.add_argument('--training_epochs', type=int, default=300,
                        help='Number of training epochs for regressor')
    parser.add_argument('--max_symbols', type=int, default=8,
                        help='Maximum number of symbols to process')
    
    args = parser.parse_args()
    
    # Set seeds
    set_global_seeds(SEED)
    
    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(',')][:args.max_symbols]
    
    # Create results directory
    results_dir = PROJECT_ROOT / 'results' / args.timestamp
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Processing symbols: {symbols}")
    logger.info(f"Skip training: {args.skip_training}")
    
    # Initialize results
    all_results = {
        'timestamp': args.timestamp,
        'start_time': datetime.now().isoformat(),
        'config': {
            'symbols': symbols,
            'skip_training': args.skip_training,
            'training_epochs': args.training_epochs,
            'seed': SEED,
            'fusion_modes': FUSION_MODES,
        },
        'gpu_info': get_gpu_info(),
        'symbols': {}
    }
    
    # Process each symbol
    for symbol in symbols:
        try:
            symbol_results = run_symbol_pipeline(
                symbol=symbol,
                results_dir=results_dir,
                skip_training=args.skip_training,
                training_epochs=args.training_epochs
            )
            all_results['symbols'][symbol] = symbol_results
        except Exception as e:
            logger.error(f"Failed to process {symbol}: {traceback.format_exc()}")
            all_results['symbols'][symbol] = {
                'error': str(e),
                'traceback': traceback.format_exc()
            }
    
    all_results['end_time'] = datetime.now().isoformat()
    
    # Generate summary table
    logger.info("Generating summary table...")
    summary_df = generate_summary_table(all_results, results_dir)
    if not summary_df.empty:
        logger.info(f"\nSummary Table:\n{summary_df.to_string()}")
    
    # Generate root cause report
    logger.info("Generating root cause report...")
    generate_root_cause_report(all_results, results_dir)
    
    # Save master results
    master_results_file = results_dir / 'auto_run_report.json'
    with open(master_results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"EXPERIMENT COMPLETE")
    logger.info(f"Results saved to: {results_dir}")
    logger.info(f"{'='*60}")
    
    return all_results


if __name__ == '__main__':
    main()
