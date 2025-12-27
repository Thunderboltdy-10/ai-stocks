"""
F4 Model Discovery Script

Scans the saved_models/ directory to identify which symbols have which model types available.
Checks both new organized structure (saved_models/{SYMBOL}/regressor/) and
legacy flat structure (saved_models/{SYMBOL}_*_model.keras).

Usage:
    python scripts/f4_discover_models.py --output f4_validation_results/model_inventory.json

Output: JSON file with availability matrix for all symbols × model types
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

def discover_models(saved_models_dir: Path) -> dict:
    """
    Scan saved_models directory and categorize available models.

    Returns:
        dict: {symbol: {regressor: bool, classifiers: bool, gbm: bool, quantile: bool}}
    """
    inventory = {}

    if not saved_models_dir.exists():
        print(f"ERROR: saved_models directory not found: {saved_models_dir}")
        return inventory

    # Track symbol directories found
    symbol_dirs = []

    # Scan for symbol directories (new structure)
    for item in saved_models_dir.iterdir():
        if item.is_dir() and not item.name.startswith('.'):
            # Check if this looks like a symbol directory (not a model artifact)
            if not item.name.endswith('_logs') and len(item.name) <= 10:
                symbol_dirs.append(item)

    print(f"Found {len(symbol_dirs)} potential symbol directories")

    # Check each symbol directory
    for symbol_dir in symbol_dirs:
        symbol = symbol_dir.name

        # Initialize availability
        has_regressor = False
        has_classifiers = False
        has_gbm = False
        has_quantile = False

        # Check new organized structure
        regressor_dir = symbol_dir / 'regressor'
        if regressor_dir.exists():
            # Check for model.keras or model directory
            has_regressor = (
                (regressor_dir / 'model.keras').exists() or
                (regressor_dir / 'model').exists() or
                (regressor_dir / 'regressor.weights.h5').exists()
            )

        classifiers_dir = symbol_dir / 'classifiers'
        if classifiers_dir.exists():
            # Check for both BUY and SELL models
            has_buy = (
                (classifiers_dir / 'buy_model.keras').exists() or
                (classifiers_dir / 'buy.weights.h5').exists() or
                (classifiers_dir / 'buy_calibrated.pkl').exists()
            )
            has_sell = (
                (classifiers_dir / 'sell_model.keras').exists() or
                (classifiers_dir / 'sell.weights.h5').exists() or
                (classifiers_dir / 'sell_calibrated.pkl').exists()
            )
            has_classifiers = has_buy and has_sell

        gbm_dir = symbol_dir / 'gbm'
        if gbm_dir.exists():
            # Check for XGBoost or LightGBM models
            has_gbm = (
                (gbm_dir / 'xgboost_model.pkl').exists() or
                (gbm_dir / 'lightgbm_model.pkl').exists() or
                (gbm_dir / 'gbm_model.pkl').exists()
            )

        quantile_dir = symbol_dir / 'quantile'
        if quantile_dir.exists():
            has_quantile = (
                (quantile_dir / 'quantile.weights.h5').exists() or
                (quantile_dir / 'model.keras').exists()
            )

        # Also check legacy flat structure
        if not has_regressor:
            legacy_regressor_paths = [
                saved_models_dir / f'{symbol}_1d_regressor_final_model',
                saved_models_dir / f'{symbol}_1d_regressor_final_model.keras',
                saved_models_dir / f'{symbol}_1d_regressor_final.weights.h5',
            ]
            has_regressor = any(p.exists() for p in legacy_regressor_paths)

        if not has_classifiers:
            legacy_buy_paths = [
                saved_models_dir / f'{symbol}_buy_classifier_model',
                saved_models_dir / f'{symbol}_buy_classifier.keras',
                saved_models_dir / f'{symbol}_buy_calibrated.pkl',
            ]
            legacy_sell_paths = [
                saved_models_dir / f'{symbol}_sell_classifier_model',
                saved_models_dir / f'{symbol}_sell_classifier.keras',
                saved_models_dir / f'{symbol}_sell_calibrated.pkl',
            ]
            has_buy_legacy = any(p.exists() for p in legacy_buy_paths)
            has_sell_legacy = any(p.exists() for p in legacy_sell_paths)
            has_classifiers = has_buy_legacy and has_sell_legacy

        if not has_gbm:
            legacy_gbm_paths = [
                saved_models_dir / f'{symbol}_xgboost_model.pkl',
                saved_models_dir / f'{symbol}_lightgbm_model.pkl',
                saved_models_dir / f'{symbol}_gbm_model.pkl',
            ]
            has_gbm = any(p.exists() for p in legacy_gbm_paths)

        if not has_quantile:
            legacy_quantile_paths = [
                saved_models_dir / f'{symbol}_quantile_regressor.weights.h5',
                saved_models_dir / f'{symbol}_quantile_model.keras',
            ]
            has_quantile = any(p.exists() for p in legacy_quantile_paths)

        # Store in inventory
        inventory[symbol] = {
            'regressor': has_regressor,
            'classifiers': has_classifiers,
            'gbm': has_gbm,
            'quantile': has_quantile
        }

        # Print summary for this symbol
        models_available = sum([has_regressor, has_classifiers, has_gbm, has_quantile])
        model_types = []
        if has_regressor:
            model_types.append('regressor')
        if has_classifiers:
            model_types.append('classifiers')
        if has_gbm:
            model_types.append('gbm')
        if has_quantile:
            model_types.append('quantile')

        print(f"  {symbol}: {models_available}/4 models ({', '.join(model_types) if model_types else 'none'})")

    return inventory


def print_summary(inventory: dict):
    """Print summary statistics of the inventory."""
    if not inventory:
        print("\nNo models found in inventory.")
        return

    total_symbols = len(inventory)

    # Count model types
    regressor_count = sum(1 for m in inventory.values() if m['regressor'])
    classifier_count = sum(1 for m in inventory.values() if m['classifiers'])
    gbm_count = sum(1 for m in inventory.values() if m['gbm'])
    quantile_count = sum(1 for m in inventory.values() if m['quantile'])

    # Count symbols with complete model sets
    complete_count = sum(
        1 for m in inventory.values()
        if m['regressor'] and m['classifiers'] and m['gbm']
    )

    print("\n" + "=" * 80)
    print("MODEL INVENTORY SUMMARY")
    print("=" * 80)
    print(f"\nTotal Symbols: {total_symbols}")
    print(f"\nModel Availability:")
    print(f"  Regressor:   {regressor_count}/{total_symbols} symbols ({regressor_count/total_symbols*100:.0f}%)")
    print(f"  Classifiers: {classifier_count}/{total_symbols} symbols ({classifier_count/total_symbols*100:.0f}%)")
    print(f"  GBM:         {gbm_count}/{total_symbols} symbols ({gbm_count/total_symbols*100:.0f}%)")
    print(f"  Quantile:    {quantile_count}/{total_symbols} symbols ({quantile_count/total_symbols*100:.0f}%)")
    print(f"\nComplete Sets (Regressor + Classifiers + GBM): {complete_count}/{total_symbols} symbols")

    # List symbols by model availability
    complete_symbols = [s for s, m in inventory.items() if m['regressor'] and m['classifiers'] and m['gbm']]
    partial_symbols = [s for s, m in inventory.items() if not (m['regressor'] and m['classifiers'] and m['gbm']) and any(m.values())]
    empty_symbols = [s for s, m in inventory.items() if not any(m.values())]

    if complete_symbols:
        print(f"\nComplete Model Sets ({len(complete_symbols)}):")
        print(f"  {', '.join(sorted(complete_symbols))}")

    if partial_symbols:
        print(f"\nPartial Model Sets ({len(partial_symbols)}):")
        print(f"  {', '.join(sorted(partial_symbols))}")

    if empty_symbols:
        print(f"\nNo Models ({len(empty_symbols)}):")
        print(f"  {', '.join(sorted(empty_symbols))}")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Discover available models in saved_models directory'
    )
    parser.add_argument(
        '--saved-models-dir',
        type=Path,
        default=PROJECT_ROOT / 'saved_models',
        help='Path to saved_models directory (default: python-ai-service/saved_models)'
    )
    parser.add_argument(
        '--output',
        type=Path,
        required=True,
        help='Output JSON file path for inventory'
    )

    args = parser.parse_args()

    print(f"\nScanning models in: {args.saved_models_dir}")
    print("-" * 80)

    # Discover models
    inventory = discover_models(args.saved_models_dir)

    # Print summary
    print_summary(inventory)

    # Save to JSON
    output_data = {
        'metadata': {
            'scan_date': datetime.now().isoformat(),
            'saved_models_dir': str(args.saved_models_dir),
            'total_symbols': len(inventory)
        },
        'inventory': inventory
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✅ Inventory saved to: {args.output}")
    print(f"   Total symbols discovered: {len(inventory)}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
