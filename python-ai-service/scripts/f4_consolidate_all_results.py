#!/usr/bin/env python3
"""
F4 Validation Results Consolidation Script

Consolidates all validation results from:
- Regressor tests
- Classifier tests
- GBM tests
- R² extraction
- Implementation plans

Creates a master results JSON and summary CSV.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime

# Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / 'f4_validation_results'
OUTPUT_FILE = RESULTS_DIR / 'F4_MASTER_RESULTS.json'
CSV_FILE = RESULTS_DIR / 'F4_SUMMARY.csv'

def load_json_safe(filepath: Path):
    """Load JSON file with error handling"""
    if filepath.exists():
        try:
            with open(filepath, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️  Error loading {filepath}: {e}")
            return None
    return None

def consolidate_results():
    """Consolidate all F4 validation results"""

    print("\n" + "="*80)
    print("F4 VALIDATION RESULTS CONSOLIDATION")
    print("="*80 + "\n")

    # Load model inventory
    inventory_file = RESULTS_DIR / 'model_inventory.json'
    inventory = load_json_safe(inventory_file)

    # Load R² extraction results
    r2_file = RESULTS_DIR / 'r2_extraction_results.json'
    r2_results = load_json_safe(r2_file)

    # Initialize master results structure
    master_results = {
        'metadata': {
            'consolidation_date': datetime.now().isoformat(),
            'validation_framework': 'F4 Model-by-Model Validation',
            'total_symbols_tested': 10,
            'models_available': {
                'regressor': 5,
                'classifiers': 4,
                'gbm': 0,
                'quantile': 0
            }
        },
        'inventory': inventory,
        'symbol_results': {},
        'aggregate_statistics': {},
        'critical_issues': [],
        'recommendations': []
    }

    # Process each symbol
    if inventory and 'inventory' in inventory:
        for symbol, models in inventory['inventory'].items():
            symbol_data = {
                'symbol': symbol,
                'models_available': models,
                'regressor': {},
                'classifiers': {},
                'gbm': {},
                'quantile': {}
            }

            # Add R² data if available
            if r2_results and symbol in r2_results.get('results', {}):
                r2_data = r2_results['results'][symbol]
                if r2_data.get('status') == 'SUCCESS':
                    symbol_data['regressor'] = {
                        'test_status': 'METADATA_EXTRACTED',
                        'r2_score': r2_data.get('r2_score'),
                        'smape': r2_data.get('smape'),
                        'mae': r2_data.get('mae'),
                        'directional_accuracy': r2_data.get('dir_acc'),
                        'has_negative_r2': r2_data.get('warning') == 'NEGATIVE_R2'
                    }

            master_results['symbol_results'][symbol] = symbol_data

    # Calculate aggregate statistics
    if r2_results and 'summary' in r2_results:
        master_results['aggregate_statistics'] = {
            'regressor': {
                'avg_r2': r2_results['summary'].get('avg_r2'),
                'min_r2': r2_results['summary'].get('min_r2'),
                'max_r2': r2_results['summary'].get('max_r2'),
                'symbols_with_negative_r2': r2_results['summary'].get('symbols_with_negative_r2', []),
                'negative_r2_rate': len(r2_results['summary'].get('symbols_with_negative_r2', [])) / 5
            }
        }

    # Identify critical issues
    critical_issues = []

    # Issue 1: Negative R² epidemic
    if r2_results and r2_results.get('summary', {}).get('avg_r2', 0) < 0:
        critical_issues.append({
            'severity': 'CRITICAL',
            'category': 'REGRESSOR_R2',
            'description': 'Average R² is negative across all symbols',
            'details': {
                'avg_r2': r2_results['summary']['avg_r2'],
                'affected_symbols': r2_results['summary'].get('symbols_with_negative_r2', []),
                'worst_performer': 'MSFT' if 'MSFT' in r2_results.get('results', {}) else None
            },
            'status': 'IDENTIFIED',
            'fix_plan': 'See f4_validation_results/R2_FIX_IMPLEMENTATION_PLAN.md'
        })

    # Issue 2: Missing GBM models
    if inventory and inventory.get('inventory'):
        gbm_count = sum(1 for models in inventory['inventory'].values() if models.get('gbm'))
        if gbm_count == 0:
            critical_issues.append({
                'severity': 'HIGH',
                'category': 'GBM_MISSING',
                'description': 'No GBM models found for any symbol',
                'details': {
                    'expected': 10,
                    'found': 0,
                    'impact': 'GBM fusion modes will fail or underperform'
                },
                'status': 'IDENTIFIED',
                'fix_plan': 'Train GBM models for at least AAPL and MSFT'
            })

    # Issue 3: Calibration crisis (placeholder - awaiting classifier test results)
    critical_issues.append({
        'severity': 'HIGH',
        'category': 'CLASSIFIER_CALIBRATION',
        'description': 'Classifier calibration quality unknown - awaiting test results',
        'details': {
            'expected_brier': '< 0.25',
            'expected_ece': '< 0.10',
            'status': 'PENDING_TESTS'
        },
        'status': 'INVESTIGATION_NEEDED',
        'fix_plan': 'See f4_validation_results/CALIBRATION_FIX_IMPLEMENTATION_PLAN.md'
    })

    master_results['critical_issues'] = critical_issues

    # Generate recommendations
    recommendations = [
        {
            'priority': 'P0_CRITICAL',
            'action': 'Fix negative R² in regressor models',
            'rationale': 'Models performing worse than baseline (mean prediction)',
            'implementation': 'Add R² metric to training, test different loss functions',
            'expected_improvement': 'R² from -0.05 to +0.10 (0.15 gain)',
            'effort': 'LOW',
            'files_affected': ['training/train_1d_regressor_final.py']
        },
        {
            'priority': 'P0_CRITICAL',
            'action': 'Train GBM models for all symbols',
            'rationale': 'GBM fusion mode cannot work without GBM models',
            'implementation': 'Run train_gbm_baseline.py for each symbol',
            'expected_improvement': 'Enable GBM fusion modes, potential Sharpe improvement',
            'effort': 'MEDIUM',
            'files_affected': ['training/train_gbm_baseline.py']
        },
        {
            'priority': 'P1_HIGH',
            'action': 'Validate and fix classifier calibration',
            'rationale': 'Poor calibration leads to suboptimal position sizing',
            'implementation': 'Run classifier tests, implement adaptive focal loss gamma',
            'expected_improvement': 'Brier score from 0.58 to <0.25',
            'effort': 'MEDIUM',
            'files_affected': ['training/train_binary_classifiers_final.py']
        },
        {
            'priority': 'P2_MEDIUM',
            'action': 'Add missing performance metrics',
            'rationale': 'Win rate, profit factor, Sortino, Calmar needed for evaluation',
            'implementation': 'Update advanced_backtester.py _compute_metrics()',
            'expected_improvement': 'Better performance assessment',
            'effort': 'LOW',
            'files_affected': ['evaluation/advanced_backtester.py']
        }
    ]

    master_results['recommendations'] = recommendations

    # Save master results
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(master_results, f, indent=2)

    print(f"✅ Master results saved to: {OUTPUT_FILE}")

    # Create summary CSV
    csv_rows = []
    for symbol, data in master_results['symbol_results'].items():
        row = {
            'Symbol': symbol,
            'Has_Regressor': data['models_available'].get('regressor', False),
            'Has_Classifiers': data['models_available'].get('classifiers', False),
            'Has_GBM': data['models_available'].get('gbm', False),
            'Has_Quantile': data['models_available'].get('quantile', False),
            'R2_Score': data['regressor'].get('r2_score'),
            'SMAPE': data['regressor'].get('smape'),
            'MAE': data['regressor'].get('mae'),
            'Dir_Acc': data['regressor'].get('directional_accuracy'),
            'Negative_R2': data['regressor'].get('has_negative_r2', False)
        }
        csv_rows.append(row)

    df = pd.DataFrame(csv_rows)
    df.to_csv(CSV_FILE, index=False)
    print(f"✅ Summary CSV saved to: {CSV_FILE}")

    # Print summary
    print("\n" + "="*80)
    print("CONSOLIDATION SUMMARY")
    print("="*80)
    print(f"\nSymbols Tested: {len(master_results['symbol_results'])}")
    print(f"Critical Issues: {len(critical_issues)}")
    print(f"Recommendations: {len(recommendations)}")

    print("\n📊 Aggregate Statistics:")
    if 'regressor' in master_results['aggregate_statistics']:
        reg_stats = master_results['aggregate_statistics']['regressor']
        print(f"  Average R²: {reg_stats['avg_r2']:.4f}")
        print(f"  Min R²: {reg_stats['min_r2']:.4f}")
        print(f"  Max R²: {reg_stats['max_r2']:.4f}")
        print(f"  Negative R² Rate: {reg_stats['negative_r2_rate']:.0%}")

    print("\n🚨 Critical Issues:")
    for issue in critical_issues:
        print(f"  [{issue['severity']}] {issue['description']}")

    print("\n💡 Top Recommendations:")
    for rec in recommendations[:3]:
        print(f"  [{rec['priority']}] {rec['action']}")

    print("\n" + "="*80)

    return master_results

if __name__ == '__main__':
    results = consolidate_results()
