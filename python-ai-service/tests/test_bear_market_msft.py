
import os
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import time

# Add project root to path
sys.path.append(os.getcwd())

from inference_and_backtest import main as backtest_main

# Target: MSFT bear market Jan-Jun 2022
# To achieve this with the existing backtest_main, we would need to modify it or 
# pass a very large backtest_days and then slice the results.
# However, backtest_main splits 80/20. 
# If we want to test a specific historical period, we should really modify the script.
# But for now, let's try to run a long backtest and then filter the trade log.

print("Running MSFT backtest for a long period to capture 2022...")
try:
    # 1000 days should cover back to 2022 (approx 252 days/year)
    # 4 years * 252 = 1008 days
    results = backtest_main(
        symbol='MSFT',
        backtest_days=1200, 
        fusion_mode='balanced', # Test LSTM + GBM
        use_cache=True
    )
    
    # After running, we can look at the saved artifacts in backtest_results/
    print("\nBacktest completed. Check the latest folder in backtest_results/ for MSFT.")
    
except Exception as e:
    print(f"Error during backtest: {e}")
    import traceback
    traceback.print_exc()
