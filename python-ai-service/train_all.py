#!/usr/bin/env python3
"""
Unified Training Pipeline for AI-Stocks

This script trains the complete stacking ensemble in one command:
1. LSTM+Transformer Regressor
2. xLSTM-TS Model
3. GBM Models (XGBoost + LightGBM)
4. Stacking Meta-Learner (XGBoost)

Usage:
    python train_all.py --symbol AAPL [--epochs 50] [--batch-size 512]

The stacking ensemble is the RECOMMENDED inference method (default in inference_and_backtest.py).
Binary classifiers have been deprecated as of December 2025.

Author: AI-Stocks
Date: December 2025
"""

import argparse
import sys
import subprocess
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent


def run_training_script(
    script_name: str,
    args: list,
    description: str,
    required: bool = True
) -> bool:
    """Run a training script and capture output.

    Args:
        script_name: Name of the script in training/ directory
        args: Command line arguments to pass
        description: Human-readable description for logging
        required: If True, raise error on failure

    Returns:
        True if successful, False otherwise
    """
    script_path = PROJECT_ROOT / 'training' / script_name
    if not script_path.exists():
        logger.error(f"Script not found: {script_path}")
        if required:
            raise FileNotFoundError(f"Required script not found: {script_name}")
        return False

    cmd = [sys.executable, str(script_path)] + args

    print("\n" + "=" * 70)
    print(f"  {description}")
    print("=" * 70)
    print(f"Command: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            check=True
        )
        print(f"\n[OK] {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"{description} failed with exit code {e.returncode}")
        if required:
            raise
        return False
    except Exception as e:
        logger.error(f"{description} failed: {e}")
        if required:
            raise
        return False


def check_model_exists(symbol: str, model_type: str) -> bool:
    """Check if a trained model exists for a symbol."""
    model_dir = PROJECT_ROOT / 'saved_models' / symbol

    if model_type == 'regressor':
        return (model_dir / 'regressor' / 'model.keras').exists()
    elif model_type == 'xlstm':
        return (model_dir / 'xlstm' / 'model.keras').exists()
    elif model_type == 'gbm':
        return (model_dir / 'gbm' / 'xgboost_model.pkl').exists()
    elif model_type == 'stacking':
        return (model_dir / 'stacking' / 'meta_learner.pkl').exists()
    return False


def train_all(
    symbol: str,
    epochs: int = 50,
    batch_size: int = 512,
    skip_wfe: bool = False,
    force: bool = False,
    skip_xlstm: bool = False,
) -> Dict:
    """Train all models for a symbol.

    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        epochs: Number of epochs for neural network training
        batch_size: Batch size for training
        skip_wfe: Skip walk-forward validation (not recommended)
        force: Overwrite existing models
        skip_xlstm: Skip xLSTM training (if it's failing)

    Returns:
        Dictionary with training results
    """
    results = {
        'symbol': symbol,
        'timestamp': datetime.now().isoformat(),
        'models': {}
    }

    print("\n" + "=" * 70)
    print(f"  UNIFIED TRAINING PIPELINE - {symbol}")
    print("=" * 70)
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Skip WFE: {skip_wfe}")
    print(f"Force overwrite: {force}")

    # Step 1: Train LSTM+Transformer Regressor
    if force or not check_model_exists(symbol, 'regressor'):
        args = [
            symbol,
            '--epochs', str(epochs),
            '--batch-size', str(batch_size),
        ]
        if skip_wfe:
            args.append('--skip-wfe')

        success = run_training_script(
            'train_1d_regressor_final.py',
            args,
            f"Step 1/4: LSTM+Transformer Regressor ({symbol})",
            required=True
        )
        results['models']['regressor'] = 'trained' if success else 'failed'
    else:
        print(f"\n[SKIP] LSTM Regressor already exists for {symbol}. Use --force to retrain.")
        results['models']['regressor'] = 'skipped'

    # Step 2: Train xLSTM-TS Model
    if not skip_xlstm:
        if force or not check_model_exists(symbol, 'xlstm'):
            args = [
                '--symbol', symbol,
                '--epochs', str(min(epochs, 30)),  # xLSTM trains faster
                '--batch-size', str(batch_size),
            ]
            if skip_wfe:
                args.append('--skip-wfe')

            success = run_training_script(
                'train_xlstm_ts.py',
                args,
                f"Step 2/4: xLSTM-TS Model ({symbol})",
                required=False  # xLSTM is optional
            )
            results['models']['xlstm'] = 'trained' if success else 'failed'
        else:
            print(f"\n[SKIP] xLSTM already exists for {symbol}. Use --force to retrain.")
            results['models']['xlstm'] = 'skipped'
    else:
        print(f"\n[SKIP] xLSTM training skipped by user request.")
        results['models']['xlstm'] = 'skipped'

    # Step 3: Train GBM Models (XGBoost + LightGBM)
    if force or not check_model_exists(symbol, 'gbm'):
        args = [
            symbol,
            '--overwrite',
        ]

        success = run_training_script(
            'train_gbm_baseline.py',
            args,
            f"Step 3/4: GBM Models ({symbol})",
            required=True
        )
        results['models']['gbm'] = 'trained' if success else 'failed'
    else:
        print(f"\n[SKIP] GBM models already exist for {symbol}. Use --force to retrain.")
        results['models']['gbm'] = 'skipped'

    # Step 4: Train Stacking Meta-Learner
    if force or not check_model_exists(symbol, 'stacking'):
        args = [
            '--symbol', symbol,
        ]

        success = run_training_script(
            'train_stacking_ensemble.py',
            args,
            f"Step 4/4: Stacking Meta-Learner ({symbol})",
            required=True
        )
        results['models']['stacking'] = 'trained' if success else 'failed'
    else:
        print(f"\n[SKIP] Stacking ensemble already exists for {symbol}. Use --force to retrain.")
        results['models']['stacking'] = 'skipped'

    # Summary
    print("\n" + "=" * 70)
    print(f"  TRAINING COMPLETE - {symbol}")
    print("=" * 70)

    for model, status in results['models'].items():
        icon = "✓" if status in ('trained', 'skipped') else "✗"
        print(f"  {icon} {model}: {status}")

    print("\n" + "-" * 70)
    print("Next steps:")
    print(f"  1. Run backtest: python inference_and_backtest.py --symbol {symbol}")
    print(f"  2. (Default mode is 'stacking' - uses trained meta-learner)")
    print("-" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Unified Training Pipeline for AI-Stocks',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train all models for AAPL
    python train_all.py --symbol AAPL

    # Train with more epochs
    python train_all.py --symbol AAPL --epochs 100

    # Force retrain all models
    python train_all.py --symbol AAPL --force

    # Skip walk-forward validation (faster, not recommended)
    python train_all.py --symbol AAPL --skip-wfe

After training, run inference:
    python inference_and_backtest.py --symbol AAPL
        """
    )

    parser.add_argument(
        '--symbol', '-s',
        type=str,
        required=True,
        help='Stock symbol (e.g., AAPL, MSFT, TSLA)'
    )
    parser.add_argument(
        '--epochs', '-e',
        type=int,
        default=50,
        help='Number of epochs for neural network training (default: 50)'
    )
    parser.add_argument(
        '--batch-size', '-b',
        type=int,
        default=512,
        help='Batch size for training (default: 512, use GPU)'
    )
    parser.add_argument(
        '--skip-wfe',
        action='store_true',
        help='Skip walk-forward validation (not recommended)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force overwrite existing models'
    )
    parser.add_argument(
        '--skip-xlstm',
        action='store_true',
        help='Skip xLSTM training (if it keeps failing)'
    )

    args = parser.parse_args()

    try:
        results = train_all(
            symbol=args.symbol.upper(),
            epochs=args.epochs,
            batch_size=args.batch_size,
            skip_wfe=args.skip_wfe,
            force=args.force,
            skip_xlstm=args.skip_xlstm,
        )

        # Check if any required models failed
        failed = [m for m, s in results['models'].items() if s == 'failed' and m != 'xlstm']
        if failed:
            print(f"\n[ERROR] Training failed for: {', '.join(failed)}")
            sys.exit(1)

        print(f"\n[SUCCESS] All models trained for {args.symbol.upper()}")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Training pipeline failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
