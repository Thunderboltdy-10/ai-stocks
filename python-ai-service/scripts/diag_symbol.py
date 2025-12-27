#!/usr/bin/env python3
"""
Symbol Diagnostic Script (diag_symbol.py)

Produces comprehensive diagnostics for a given symbol:
- Model artifacts validation
- Feature alignment check
- Cache metadata inspection
- Sample inference test
- Backtester sanity checks

Usage: python scripts/diag_symbol.py --symbol AAPL
"""

import argparse
import json
import pickle
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_model_artifacts(symbol: str) -> Dict[str, Any]:
    """Check all model artifacts for a symbol."""
    results = {
        'symbol': symbol,
        'artifacts': {},
        'issues': [],
        'warnings': []
    }
    
    saved_models = PROJECT_ROOT / 'saved_models'
    symbol_dir = saved_models / symbol
    
    # Check symbol directory
    if not symbol_dir.exists():
        # Check legacy structure
        legacy_weights = saved_models / f'{symbol}_1d_regressor_final.weights.h5'
        if legacy_weights.exists():
            results['artifacts']['structure'] = 'legacy'
            results['warnings'].append("Using legacy flat file structure")
        else:
            results['issues'].append(f"No model directory found: {symbol_dir}")
            return results
    else:
        results['artifacts']['structure'] = 'organized'
    
    # Check each model type
    model_checks = {
        'regressor': {
            'weights': symbol_dir / 'regressor' / 'regressor.weights.h5',
            'scaler': symbol_dir / 'regressor' / 'feature_scaler.pkl',
            'features': symbol_dir / 'regressor' / 'features.pkl',
            'metadata': symbol_dir / 'regressor' / 'metadata.pkl',
        },
        'gbm': {
            'lgb': symbol_dir / 'gbm' / 'lgb_reg.joblib',
            'xgb': symbol_dir / 'gbm' / 'xgb_reg.joblib',
            'scaler': symbol_dir / 'gbm' / 'lgb_scaler.joblib',
            'features': symbol_dir / 'gbm' / 'feature_columns.pkl',
            'metadata': symbol_dir / 'gbm' / 'training_metadata.json',
        },
        'classifiers': {
            'buy_weights': symbol_dir / 'classifiers' / 'buy.weights.h5',
            'sell_weights': symbol_dir / 'classifiers' / 'sell.weights.h5',
        },
        'shared': {
            'feature_columns': symbol_dir / 'feature_columns.pkl',
            'target_metadata': symbol_dir / 'target_metadata.pkl',
        }
    }
    
    for model_type, paths in model_checks.items():
        results['artifacts'][model_type] = {}
        for name, path in paths.items():
            exists = path.exists()
            results['artifacts'][model_type][name] = {
                'path': str(path),
                'exists': exists,
            }
            if exists:
                results['artifacts'][model_type][name]['size_bytes'] = path.stat().st_size
            else:
                if model_type in ['regressor', 'gbm', 'shared']:
                    results['issues'].append(f"Missing {model_type}/{name}: {path}")
    
    return results


def check_feature_alignment(symbol: str) -> Dict[str, Any]:
    """Check feature alignment between saved artifacts and current code."""
    results = {
        'symbol': symbol,
        'aligned': False,
        'details': {},
        'issues': []
    }
    
    try:
        from data.feature_engineer import get_feature_columns, EXPECTED_FEATURE_COUNT
        
        saved_models = PROJECT_ROOT / 'saved_models'
        symbol_dir = saved_models / symbol
        
        # Load saved feature columns
        feature_cols_path = symbol_dir / 'feature_columns.pkl'
        if not feature_cols_path.exists():
            feature_cols_path = symbol_dir / 'regressor' / 'features.pkl'
        
        if not feature_cols_path.exists():
            results['issues'].append("No feature_columns.pkl found")
            return results
        
        with open(feature_cols_path, 'rb') as f:
            saved_features = pickle.load(f)
        
        # Get canonical features
        canonical_features = get_feature_columns(include_sentiment=True)
        
        results['details'] = {
            'saved_count': len(saved_features),
            'canonical_count': len(canonical_features),
            'expected_count': EXPECTED_FEATURE_COUNT,
        }
        
        # Check sets
        saved_set = set(saved_features)
        canonical_set = set(canonical_features)
        
        missing_in_saved = canonical_set - saved_set
        extra_in_saved = saved_set - canonical_set
        
        if missing_in_saved:
            results['details']['missing_in_saved'] = list(missing_in_saved)[:10]
            results['issues'].append(f"Model missing {len(missing_in_saved)} canonical features")
        
        if extra_in_saved:
            results['details']['extra_in_saved'] = list(extra_in_saved)[:10]
            results['issues'].append(f"Model has {len(extra_in_saved)} extra features")
        
        if not missing_in_saved and not extra_in_saved:
            if list(saved_features) == list(canonical_features):
                results['aligned'] = True
            else:
                results['issues'].append("Feature sets match but order differs")
        
    except Exception as e:
        results['issues'].append(f"Error: {e}")
    
    return results


def check_cache_metadata(symbol: str) -> Dict[str, Any]:
    """Check cache metadata for a symbol."""
    results = {
        'symbol': symbol,
        'cached': False,
        'details': {},
        'issues': []
    }
    
    cache_dir = PROJECT_ROOT / 'cache' / symbol
    if not cache_dir.exists():
        results['issues'].append("No cache directory found")
        return results
    
    results['cached'] = True
    
    # Check metadata.json
    metadata_path = cache_dir / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        results['details']['metadata'] = metadata
    else:
        results['issues'].append("No metadata.json in cache")
    
    # Check cached files
    expected_files = ['raw_data.pkl', 'engineered_features.pkl', 'prepared_training.pkl', 'feature_columns.pkl']
    for fname in expected_files:
        fpath = cache_dir / fname
        if fpath.exists():
            results['details'][fname] = {
                'exists': True,
                'size_bytes': fpath.stat().st_size
            }
        else:
            results['issues'].append(f"Missing cache file: {fname}")
    
    return results


def run_sample_inference(symbol: str) -> Dict[str, Any]:
    """Run a quick sample inference to check the pipeline."""
    results = {
        'symbol': symbol,
        'success': False,
        'details': {},
        'issues': []
    }
    
    try:
        from data.cache_manager import DataCacheManager
        from data.target_engineering import prepare_training_data
        
        cache_manager = DataCacheManager()
        raw_df, engineered_df, prepared_df, feature_cols = cache_manager.get_or_fetch_data(
            symbol=symbol,
            include_sentiment=True,
            force_refresh=False
        )
        
        results['details']['data_rows'] = len(engineered_df)
        results['details']['feature_count'] = len(feature_cols)
        
        # Check for NaN/Inf
        nan_count = engineered_df[feature_cols].isna().sum().sum()
        inf_count = np.isinf(engineered_df[feature_cols].select_dtypes(include=[np.number])).sum().sum()
        
        results['details']['nan_count'] = int(nan_count)
        results['details']['inf_count'] = int(inf_count)
        
        if nan_count > 0:
            results['warnings'] = [f"Found {nan_count} NaN values in features"]
        if inf_count > 0:
            results['issues'].append(f"Found {inf_count} Inf values in features")
        
        # Feature statistics
        numeric_cols = engineered_df[feature_cols].select_dtypes(include=[np.number]).columns
        results['details']['feature_stats'] = {
            'mean_of_means': float(engineered_df[numeric_cols].mean().mean()),
            'mean_of_stds': float(engineered_df[numeric_cols].std().mean()),
            'min_min': float(engineered_df[numeric_cols].min().min()),
            'max_max': float(engineered_df[numeric_cols].max().max()),
        }
        
        results['success'] = True
        
    except Exception as e:
        results['issues'].append(f"Inference test failed: {e}")
        import traceback
        results['traceback'] = traceback.format_exc()
    
    return results


def check_backtester_config() -> Dict[str, Any]:
    """Check backtester configuration for realism issues."""
    results = {
        'checks': {},
        'issues': []
    }
    
    try:
        from evaluation.advanced_backtester import AdvancedBacktester
        
        bt = AdvancedBacktester()
        
        results['checks'] = {
            'initial_capital': bt.initial_capital,
            'commission_pct': bt.commission_pct,
            'slippage_pct': bt.slippage_pct,
            'min_commission': bt.min_commission,
        }
        
        # Check for realism
        if bt.commission_pct == 0:
            results['issues'].append("Commission is 0 - unrealistic")
        if bt.slippage_pct == 0:
            results['issues'].append("Slippage is 0 - unrealistic")
        
        # Check for reasonable defaults
        if bt.commission_pct > 0.01:
            results['issues'].append(f"Commission seems high: {bt.commission_pct*100:.2f}%")
        
    except Exception as e:
        results['issues'].append(f"Error checking backtester: {e}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Symbol Diagnostic Script')
    parser.add_argument('--symbol', type=str, default='AAPL', help='Stock symbol to diagnose')
    parser.add_argument('--output', type=str, default=None, help='Output file path (JSON)')
    
    args = parser.parse_args()
    symbol = args.symbol.upper()
    
    print(f"\n{'='*60}")
    print(f"DIAGNOSTIC REPORT: {symbol}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")
    
    results = {
        'symbol': symbol,
        'timestamp': datetime.now().isoformat(),
        'diagnostics': {}
    }
    
    # 1. Model artifacts
    print("1. Checking model artifacts...")
    artifacts = check_model_artifacts(symbol)
    results['diagnostics']['artifacts'] = artifacts
    if artifacts['issues']:
        for issue in artifacts['issues']:
            print(f"   ❌ {issue}")
    else:
        print("   ✅ All model artifacts present")
    
    # 2. Feature alignment
    print("\n2. Checking feature alignment...")
    alignment = check_feature_alignment(symbol)
    results['diagnostics']['feature_alignment'] = alignment
    if alignment['aligned']:
        print("   ✅ Features aligned")
    else:
        for issue in alignment['issues']:
            print(f"   ❌ {issue}")
    print(f"   Details: {alignment['details']}")
    
    # 3. Cache metadata
    print("\n3. Checking cache...")
    cache = check_cache_metadata(symbol)
    results['diagnostics']['cache'] = cache
    if cache['cached']:
        print(f"   ✅ Cache present")
    else:
        for issue in cache['issues']:
            print(f"   ❌ {issue}")
    
    # 4. Sample inference
    print("\n4. Running sample inference...")
    inference = run_sample_inference(symbol)
    results['diagnostics']['inference'] = inference
    if inference['success']:
        print(f"   ✅ Inference test passed")
        print(f"   Data rows: {inference['details'].get('data_rows', 'N/A')}")
        print(f"   Features: {inference['details'].get('feature_count', 'N/A')}")
    else:
        for issue in inference['issues']:
            print(f"   ❌ {issue}")
    
    # 5. Backtester config
    print("\n5. Checking backtester configuration...")
    backtester = check_backtester_config()
    results['diagnostics']['backtester'] = backtester
    if not backtester['issues']:
        print("   ✅ Backtester configured realistically")
        print(f"   Commission: {backtester['checks'].get('commission_pct', 0)*100:.2f}%")
        print(f"   Slippage: {backtester['checks'].get('slippage_pct', 0)*100:.2f}%")
    else:
        for issue in backtester['issues']:
            print(f"   ⚠️ {issue}")
    
    # Summary
    all_issues = []
    for diag_name, diag_data in results['diagnostics'].items():
        if isinstance(diag_data, dict) and 'issues' in diag_data:
            all_issues.extend(diag_data['issues'])
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {len(all_issues)} issue(s) found")
    print(f"{'='*60}")
    
    if all_issues:
        for i, issue in enumerate(all_issues, 1):
            print(f"{i}. {issue}")
    else:
        print("All checks passed!")
    
    # Save output
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nSaved diagnostics to: {output_path}")
    
    return results


if __name__ == '__main__':
    main()
