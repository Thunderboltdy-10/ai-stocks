#!/usr/bin/env python3
"""
Analyze GBM Training Metadata for All Symbols
"""
import json
from pathlib import Path

SYMBOLS = ['AAPL', 'MSFT', 'NVDA', 'GOOGL', 'TSLA', 'META', 'SPY', 'IWM', 'ASML', 'KO']

def analyze_all_gbm():
    results = {}

    for symbol in SYMBOLS:
        meta_path = Path(f'saved_models/{symbol}/gbm/training_metadata.json')

        if not meta_path.exists():
            results[symbol] = {'status': 'MISSING', 'error': 'metadata_not_found'}
            continue

        try:
            with open(meta_path) as f:
                data = json.load(f)

            # Extract key metrics
            xgb_cv = data['models']['xgb']['cv_results']['aggregate_metrics']
            lgb_cv = data['models']['lgb']['cv_results']['aggregate_metrics']

            xgb_final = data['models']['xgb']['final_metadata']
            lgb_final = data['models']['lgb']['final_metadata']

            results[symbol] = {
                'status': 'OK',
                'xgb': {
                    'dir_acc': xgb_cv['dir_acc_mean'],
                    'ic': xgb_cv['ic_mean'],
                    'r2': xgb_cv['r2_mean'],
                    'mae': xgb_cv['mae_mean'],
                    'final_ic': xgb_final.get('ic', 0),
                    'final_dir_acc': xgb_final.get('direction_acc', 0),
                    'pred_std': xgb_final['prediction_stats']['std'],
                },
                'lgb': {
                    'dir_acc': lgb_cv['dir_acc_mean'],
                    'ic': lgb_cv['ic_mean'],
                    'r2': lgb_cv['r2_mean'],
                    'mae': lgb_cv['mae_mean'],
                    'final_ic': lgb_final.get('ic', 0),
                    'final_dir_acc': lgb_final.get('direction_acc', 0),
                    'pred_std': lgb_final['prediction_stats']['std'],
                }
            }

        except Exception as e:
            results[symbol] = {'status': 'ERROR', 'error': str(e)}

    # Print summary
    print("="*100)
    print("GBM TRAINING METADATA ANALYSIS")
    print("="*100)
    print(f"\n{'Symbol':<8} {'Model':<8} {'Dir Acc':<10} {'IC':<10} {'R²':<10} {'MAE':<10} {'PredStd':<12} {'Status':<8}")
    print("-"*100)

    for symbol, data in results.items():
        if data['status'] == 'OK':
            # XGBoost
            xgb = data['xgb']
            status = '✓' if xgb['ic'] > 0 and xgb['dir_acc'] > 0.50 else '⚠️'
            print(f"{symbol:<8} {'XGB':<8} {xgb['dir_acc']:<10.3f} {xgb['ic']:<10.3f} {xgb['r2']:<10.3f} {xgb['mae']:<10.5f} {xgb['pred_std']:<12.6f} {status:<8}")

            # LightGBM
            lgb = data['lgb']
            status = '✓' if lgb['ic'] > 0 and lgb['dir_acc'] > 0.50 else '⚠️'
            print(f"{symbol:<8} {'LGB':<8} {lgb['dir_acc']:<10.3f} {lgb['ic']:<10.3f} {lgb['r2']:<10.3f} {lgb['mae']:<10.5f} {lgb['pred_std']:<12.6f} {status:<8}")
        else:
            print(f"{symbol:<8} {'N/A':<8} {'-':<10} {'-':<10} {'-':<10} {'-':<10} {'-':<12} {data['status']:<8}")

    print("-"*100)

    # Aggregate statistics
    ok_symbols = [s for s, d in results.items() if d['status'] == 'OK']
    print(f"\nSymbols with GBM models: {len(ok_symbols)}/10")

    if ok_symbols:
        avg_xgb_dir = sum(results[s]['xgb']['dir_acc'] for s in ok_symbols) / len(ok_symbols)
        avg_xgb_ic = sum(results[s]['xgb']['ic'] for s in ok_symbols) / len(ok_symbols)
        avg_lgb_dir = sum(results[s]['lgb']['dir_acc'] for s in ok_symbols) / len(ok_symbols)
        avg_lgb_ic = sum(results[s]['lgb']['ic'] for s in ok_symbols) / len(ok_symbols)

        print(f"\nAverage Performance:")
        print(f"  XGBoost: Dir={avg_xgb_dir:.3f}, IC={avg_xgb_ic:.3f}")
        print(f"  LightGBM: Dir={avg_lgb_dir:.3f}, IC={avg_lgb_ic:.3f}")

        # Check for issues
        variance_collapse = [s for s in ok_symbols if results[s]['lgb']['pred_std'] < 0.001]
        negative_ic = [s for s in ok_symbols if results[s]['xgb']['ic'] < 0 or results[s]['lgb']['ic'] < 0]
        low_dir_acc = [s for s in ok_symbols if results[s]['xgb']['dir_acc'] < 0.50 or results[s]['lgb']['dir_acc'] < 0.50]

        print(f"\nIssues Detected:")
        print(f"  Variance collapse (LGB std < 0.001): {len(variance_collapse)} symbols")
        if variance_collapse:
            print(f"    {', '.join(variance_collapse)}")
        print(f"  Negative IC: {len(negative_ic)} symbols")
        if negative_ic:
            print(f"    {', '.join(negative_ic)}")
        print(f"  Low directional accuracy (<50%): {len(low_dir_acc)} symbols")
        if low_dir_acc:
            print(f"    {', '.join(low_dir_acc)}")

    return results

if __name__ == '__main__':
    results = analyze_all_gbm()

    # Save to JSON
    output_path = Path('f4_validation_results/gbm_metadata_analysis.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Results saved to: {output_path}")
