#!/usr/bin/env python3
"""
Complete Training & Backtesting Pipeline
Trains models, validates, backtests, and logs all errors
"""

import os
import sys
import json
import logging
import argparse
import subprocess
from pathlib import Path
from datetime import datetime
import traceback
from typing import Dict, List, Tuple

# Setup paths
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd

# Configure logging
LOG_DIR = Path(__file__).parent / 'logs'
LOG_DIR.mkdir(exist_ok=True)

TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE = LOG_DIR / f'pipeline_{TIMESTAMP}.log'

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'symbols': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA'],
    'epochs': 100,
    'batch_size': 512,
    'validation_split': 0.15,
    'test_split': 0.15,
    'output_dir': 'saved_models',
    'backtest_output_dir': 'backtest_results',
    'validation_output_dir': 'validation_results',
}

# ============================================================================
# PHASE 1: DATA PREPARATION
# ============================================================================

def prepare_data(symbol: str) -> Tuple[bool, str]:
    """Prepare and cache data for a symbol."""
    try:
        logger.info(f"{'='*70}")
        logger.info(f"PHASE 1: DATA PREPARATION - {symbol}")
        logger.info(f"{'='*70}")
        
        from data.data_fetcher import fetch_stock_data
        from data.feature_engineer import engineer_features
        import pickle
        
        # Fetch raw data
        logger.info(f"Fetching raw data for {symbol}...")
        df_raw = fetch_stock_data(symbol)
        logger.info(f"✓ Fetched {len(df_raw)} rows of OHLCV data")
        
        # Engineer features
        logger.info(f"Engineering features for {symbol}...")
        df_features = engineer_features(df_raw, symbol=symbol, include_sentiment=False)
        logger.info(f"✓ Engineered {len(df_features.columns)} features")
        
        # Save feature columns
        cache_dir = Path(__file__).parent / 'cache' / symbol
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        feature_cols = list(df_features.columns)
        with open(cache_dir / 'feature_columns.pkl', 'wb') as f:
            pickle.dump(feature_cols, f)
        logger.info(f"✓ Saved {len(feature_cols)} feature columns to cache")
        
        # Validate data
        n_nan = df_features.isna().sum().sum()
        if n_nan > 0:
            logger.warning(f"⚠️ Found {n_nan} NaN values in engineered features")
        
        logger.info(f"✓ Data preparation completed for {symbol}")
        return True, "OK"
        
    except Exception as e:
        error_msg = f"Data preparation failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return False, error_msg

# ============================================================================
# PHASE 2: MODEL TRAINING
# ============================================================================

def train_regressor(symbol: str) -> Tuple[bool, str]:
    """Train 1D regressor model."""
    try:
        logger.info(f"\nPHASE 2A: TRAIN REGRESSOR - {symbol}")
        logger.info("-" * 70)
        
        import subprocess
        
        # Run training as subprocess to avoid sys.argv issues
        result = subprocess.run(
            [
                'python', 'training/train_1d_regressor_final.py',
                symbol,
                '--epochs', str(CONFIG['epochs']),
                '--batch-size', str(CONFIG['batch_size']),
                '--use-anti-collapse-loss',
            ],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        
        logger.info(result.stdout[-500:] if result.stdout else "No output")
        if result.stderr:
            logger.warning("Stderr: " + result.stderr[-500:])
        
        if result.returncode != 0:
            raise RuntimeError(f"Training failed with code {result.returncode}")
        
        # Verify model saved
        model_path = Path(CONFIG['output_dir']) / f'{symbol}_regressor.pkl'
        if model_path.exists():
            size_mb = model_path.stat().st_size / 1024 / 1024
            logger.info(f"✓ Regressor trained and saved ({size_mb:.2f} MB)")
            return True, "OK"
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
            
    except Exception as e:
        error_msg = f"Regressor training failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return False, error_msg

def train_classifiers(symbol: str) -> Tuple[bool, str]:
    """Train BUY/SELL classifiers."""
    try:
        logger.info(f"\nPHASE 2B: TRAIN CLASSIFIERS - {symbol}")
        logger.info("-" * 70)
        
        import subprocess
        
        # Run training as subprocess
        result = subprocess.run(
            [
                'python', 'training/train_binary_classifiers_final.py',
                symbol,
                '--epochs', str(CONFIG['epochs']),
                '--batch-size', str(CONFIG['batch_size']),
            ],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        
        logger.info(result.stdout[-500:] if result.stdout else "No output")
        if result.stderr:
            logger.warning("Stderr: " + result.stderr[-500:])
        
        if result.returncode != 0:
            raise RuntimeError(f"Training failed with code {result.returncode}")
        
        # Verify models saved
        buy_model = Path(CONFIG['output_dir']) / f'{symbol}_buy_classifier.pkl'
        sell_model = Path(CONFIG['output_dir']) / f'{symbol}_sell_classifier.pkl'
        
        if buy_model.exists() and sell_model.exists():
            logger.info(f"✓ Classifiers trained and saved")
            return True, "OK"
        else:
            raise FileNotFoundError("Classifier models not found")
            
    except Exception as e:
        error_msg = f"Classifiers training failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return False, error_msg

def train_quantile(symbol: str) -> Tuple[bool, str]:
    """Train quantile regressor."""
    try:
        logger.info(f"\nPHASE 2C: TRAIN QUANTILE REGRESSOR - {symbol}")
        logger.info("-" * 70)
        
        import subprocess
        
        # Run training as subprocess
        result = subprocess.run(
            [
                'python', 'training/train_quantile_regressor.py',
                symbol,
                '--epochs', str(CONFIG['epochs']),
                '--batch-size', str(CONFIG['batch_size']),
            ],
            cwd=Path(__file__).parent,
            capture_output=True,
            text=True,
            timeout=3600,
        )
        
        logger.info(result.stdout[-500:] if result.stdout else "No output")
        if result.stderr:
            logger.warning("Stderr: " + result.stderr[-500:])
        
        if result.returncode != 0:
            raise RuntimeError(f"Training failed with code {result.returncode}")
        
        # Verify model saved
        model_path = Path(CONFIG['output_dir']) / f'{symbol}_quantile_regressor.pkl'
        if model_path.exists():
            logger.info(f"✓ Quantile regressor trained and saved")
            return True, "OK"
        else:
            raise FileNotFoundError(f"Model not found at {model_path}")
            
    except Exception as e:
        error_msg = f"Quantile training failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return False, error_msg

# ============================================================================
# PHASE 3: MODEL VALIDATION
# ============================================================================

def validate_model(symbol: str) -> Tuple[bool, str, Dict]:
    """Validate trained model."""
    try:
        logger.info(f"\nPHASE 3: VALIDATION - {symbol}")
        logger.info("-" * 70)
        
        from scripts.validate_pipeline import run_validation
        
        # Create output dir
        val_dir = Path(CONFIG['validation_output_dir'])
        val_dir.mkdir(parents=True, exist_ok=True)
        
        # Run validation
        results = run_validation(
            symbol=symbol,
            epochs=0,
            batch_size=CONFIG['batch_size'],
            load_model_path=None,
            save_model_path=None,
            output_dir=str(val_dir),
        )
        
        # Log key metrics
        if results and isinstance(results, dict):
            metrics = results.get('test_metrics', {})
            logger.info(f"  Directional Accuracy: {metrics.get('directional_accuracy', 0):.2%}")
            logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
            logger.info(f"  Annualized Return: {metrics.get('annualized_return', 0):.2%}")
            
            # Check if validation passed
            if metrics.get('directional_accuracy', 0) > 0.50:
                logger.info(f"✓ Validation passed for {symbol}")
                return True, "OK", metrics
            else:
                logger.warning(f"⚠️ Directional accuracy < 50% for {symbol}")
                return True, "WARN", metrics
        else:
            return True, "OK", {}
            
    except Exception as e:
        error_msg = f"Validation failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return False, error_msg, {}

# ============================================================================
# PHASE 4: BACKTESTING
# ============================================================================

def run_backtest(symbol: str) -> Tuple[bool, str, Dict]:
    """Run backtest for trained model."""
    try:
        logger.info(f"\nPHASE 4: BACKTESTING - {symbol}")
        logger.info("-" * 70)
        
        from inference_and_backtest import backtest_symbol
        
        # Create output dir
        bt_dir = Path(CONFIG['backtest_output_dir'])
        bt_dir.mkdir(parents=True, exist_ok=True)
        
        # Run backtest
        logger.info(f"Running backtest for {symbol}...")
        backtest_results = backtest_symbol(
            symbol=symbol,
            start_date='2024-01-01',
            end_date=datetime.now().strftime('%Y-%m-%d'),
            fusion_mode='risk_aware',
            output_dir=str(bt_dir),
        )
        
        if backtest_results:
            metrics = backtest_results.get('metrics', {})
            logger.info(f"  Total Return: {metrics.get('total_return', 0):.2%}")
            logger.info(f"  Annualized Return: {metrics.get('annualized_return', 0):.2%}")
            logger.info(f"  Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.4f}")
            logger.info(f"  Max Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            logger.info(f"  Win Rate: {metrics.get('win_rate', 0):.2%}")
            
            logger.info(f"✓ Backtest completed for {symbol}")
            return True, "OK", metrics
        else:
            raise ValueError("No backtest results returned")
            
    except Exception as e:
        error_msg = f"Backtest failed: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return False, error_msg, {}

# ============================================================================
# MAIN ORCHESTRATION
# ============================================================================

def run_pipeline(symbols: List[str]) -> Dict:
    """Run complete pipeline for multiple symbols."""
    
    logger.info("=" * 70)
    logger.info("STARTING COMPLETE TRAINING & BACKTESTING PIPELINE")
    logger.info(f"Symbols: {', '.join(symbols)}")
    logger.info(f"Epochs: {CONFIG['epochs']}")
    logger.info(f"Batch Size: {CONFIG['batch_size']}")
    logger.info("=" * 70)
    
    results = {}
    
    for symbol in symbols:
        logger.info("\n\n")
        logger.info(f"{'#' * 70}")
        logger.info(f"# PROCESSING: {symbol}")
        logger.info(f"{'#' * 70}")
        
        symbol_results = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'phases': {},
        }
        
        # Phase 1: Data Preparation
        success, msg = prepare_data(symbol)
        symbol_results['phases']['data_preparation'] = {
            'success': success,
            'message': msg,
        }
        if not success:
            logger.error(f"✗ Skipping {symbol} - data preparation failed")
            results[symbol] = symbol_results
            continue
        
        # Phase 2: Model Training
        success, msg = train_regressor(symbol)
        symbol_results['phases']['train_regressor'] = {
            'success': success,
            'message': msg,
        }
        if not success:
            logger.error(f"⚠️ Regressor training failed for {symbol}")
        
        success, msg = train_classifiers(symbol)
        symbol_results['phases']['train_classifiers'] = {
            'success': success,
            'message': msg,
        }
        if not success:
            logger.error(f"⚠️ Classifiers training failed for {symbol}")
        
        success, msg = train_quantile(symbol)
        symbol_results['phases']['train_quantile'] = {
            'success': success,
            'message': msg,
        }
        if not success:
            logger.error(f"⚠️ Quantile training failed for {symbol}")
        
        # Phase 3: Validation
        success, msg, metrics = validate_model(symbol)
        symbol_results['phases']['validation'] = {
            'success': success,
            'message': msg,
            'metrics': metrics,
        }
        
        # Phase 4: Backtesting
        success, msg, metrics = run_backtest(symbol)
        symbol_results['phases']['backtesting'] = {
            'success': success,
            'message': msg,
            'metrics': metrics,
        }
        
        results[symbol] = symbol_results
        
        logger.info(f"\n✓ Completed {symbol}\n")
    
    # Save summary
    summary_file = LOG_DIR / f'pipeline_summary_{TIMESTAMP}.json'
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"\n\nPipeline summary saved to: {summary_file}")
    
    return results

# ============================================================================
# MAIN
# ============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Complete Training & Backtesting Pipeline')
    parser.add_argument('--symbols', type=str, default='AAPL,MSFT,GOOGL',
                       help='Comma-separated symbols to train')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=512)
    
    args = parser.parse_args()
    
    # Update config
    CONFIG['symbols'] = [s.strip() for s in args.symbols.split(',')]
    CONFIG['epochs'] = args.epochs
    CONFIG['batch_size'] = args.batch_size
    
    # Run pipeline
    try:
        results = run_pipeline(CONFIG['symbols'])
        
        # Print summary
        logger.info("\n\n" + "=" * 70)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 70)
        
        for symbol, result in results.items():
            logger.info(f"\n{symbol}:")
            for phase, phase_result in result['phases'].items():
                status = "✓" if phase_result['success'] else "✗"
                logger.info(f"  {status} {phase}: {phase_result['message']}")
        
        logger.info("\n" + "=" * 70)
        logger.info(f"Pipeline completed. Log: {LOG_FILE}")
        logger.info("=" * 70)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        logger.error(traceback.format_exc())
        sys.exit(1)
