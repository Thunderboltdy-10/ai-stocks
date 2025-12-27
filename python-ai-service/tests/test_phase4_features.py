
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(os.getcwd())

from data.data_fetcher import fetch_stock_data
from data.feature_engineer import engineer_features, get_feature_columns

def test_features():
    symbol = 'AAPL'
    print(f"Testing feature engineering for {symbol}...")
    
    # Fetch some data
    df = fetch_stock_data(symbol, period='2y')
    df = df.tail(500)
    print(f"Using {len(df)} rows for testing.")
    
    # Run engineering
    try:
        engineered_df = engineer_features(df, symbol=symbol, include_sentiment=True)
        print("\n✅ engineer_features completed successfully!")
        
        # Check columns
        expected_cols = get_feature_columns(include_sentiment=True)
        actual_cols = [c for c in engineered_df.columns if c in expected_cols]
        
        print(f"Expected technical + sentiment columns: {len(expected_cols)}")
        print(f"Actual columns in engineered_df: {len(engineered_df.columns)}")
        print(f"Successfully matched expected columns: {len(actual_cols)}")
        
        # Check specifically for Phase 4 features
        phase4_cols = [
            'latest_high_pivot', 'latest_low_pivot', 'dist_to_high_pivot', 'dist_to_low_pivot',
            'vap_poc_price', 'dist_to_vap_poc', 'hvn_proximity',
            'near_resistance_zone', 'near_support_zone', 'volume_concentration_score'
        ]
        
        found_p4 = [c for c in phase4_cols if c in engineered_df.columns]
        print(f"\nPhase 4 features found ({len(found_p4)}/10):")
        for c in phase4_cols:
            status = "✅" if c in found_p4 else "❌"
            val = engineered_df[c].iloc[-1] if c in found_p4 else "N/A"
            print(f"  {status} {c}: {val}")
            
        if len(found_p4) == 10:
            print("\n🎉 PHASE 4 FEATURE VERIFICATION SUCCESSFUL!")
        else:
            print("\n❌ PHASE 4 FEATURE VERIFICATION FAILED!")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n❌ Feature engineering failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_features()
