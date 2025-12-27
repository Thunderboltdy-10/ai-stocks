#!/usr/bin/env python3
"""
Extract R² scores from regressor metadata files
"""
import pickle
import json
from pathlib import Path

SYMBOLS = ['AAPL', 'ASML', 'IWM', 'KO', 'MSFT']
SAVED_MODELS_DIR = Path('/home/thunderboltdy/ai-stocks/python-ai-service/saved_models')

results = {}

for symbol in SYMBOLS:
    metadata_file = SAVED_MODELS_DIR / f"{symbol}_1d_regressor_final_metadata.pkl"

    if not metadata_file.exists():
        print(f"❌ {symbol}: Metadata file not found")
        results[symbol] = {
            'status': 'FAILED',
            'error': 'Metadata file not found'
        }
        continue

    try:
        with open(metadata_file, 'rb') as f:
            metadata = pickle.load(f)

        r2_score = metadata.get('val_r2', None)
        smape = metadata.get('val_smape', None)
        mae = metadata.get('val_mae', None)
        dir_acc = metadata.get('val_dir_acc', None)

        print(f"\n{symbol}:")
        print(f"  R² Score: {r2_score:.4f}" if r2_score is not None else "  R² Score: N/A")
        print(f"  SMAPE: {smape:.2f}%" if smape is not None else "  SMAPE: N/A")
        print(f"  MAE: {mae:.6f}" if mae is not None else "  MAE: N/A")
        print(f"  Dir Acc: {dir_acc:.1%}" if dir_acc is not None else "  Dir Acc: N/A")

        results[symbol] = {
            'status': 'SUCCESS',
            'r2_score': float(r2_score) if r2_score is not None else None,
            'smape': float(smape) if smape is not None else None,
            'mae': float(mae) if mae is not None else None,
            'dir_acc': float(dir_acc) if dir_acc is not None else None,
            'metadata': {
                'training_date': metadata.get('training_timestamp', 'N/A'),
                'sequence_length': metadata.get('model_architecture', {}).get('sequence_length', 'N/A'),
                'units': metadata.get('model_architecture', {}).get('units', 'N/A')
            }
        }

        if r2_score is not None and r2_score < 0:
            print(f"  ⚠️  WARNING: Negative R² score!")
            results[symbol]['warning'] = 'NEGATIVE_R2'

    except Exception as e:
        print(f"❌ {symbol}: Error loading metadata - {str(e)}")
        results[symbol] = {
            'status': 'FAILED',
            'error': str(e)
        }

# Summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)

r2_scores = [results[s]['r2_score'] for s in SYMBOLS if results[s].get('r2_score') is not None]
negative_r2 = [s for s in SYMBOLS if results[s].get('r2_score') is not None and results[s]['r2_score'] < 0]

if r2_scores:
    print(f"Average R²: {sum(r2_scores)/len(r2_scores):.4f}")
    print(f"Min R²: {min(r2_scores):.4f}")
    print(f"Max R²: {max(r2_scores):.4f}")

if negative_r2:
    print(f"\n⚠️  Symbols with NEGATIVE R²: {', '.join(negative_r2)}")
else:
    print(f"\n✅ No symbols with negative R²")

# Save results
output_file = Path('/home/thunderboltdy/ai-stocks/python-ai-service/f4_validation_results/r2_extraction_results.json')
with open(output_file, 'w') as f:
    json.dump({
        'symbols_tested': SYMBOLS,
        'results': results,
        'summary': {
            'avg_r2': sum(r2_scores)/len(r2_scores) if r2_scores else None,
            'min_r2': min(r2_scores) if r2_scores else None,
            'max_r2': max(r2_scores) if r2_scores else None,
            'symbols_with_negative_r2': negative_r2
        }
    }, f, indent=2)

print(f"\n✅ Results saved to: {output_file}")
