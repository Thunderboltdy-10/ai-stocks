#!/usr/bin/env python3
"""
Autonomous Execution Loop for AI-Stocks (Phase 8)
Monitors model performance and triggers automated retraining.
"""

import os
import sys
import time
import json
import logging
import subprocess
from pathlib import Path
from datetime import datetime

# Configuration
THRESHOLD_SHARPE = 1.0
THRESHOLD_ACCURACY = 0.52
SYMBOLS = ["IWM", "ASML", "KO"]
CHECK_INTERVAL_HOURS = 24

# Paths
BASE_DIR = Path(__file__).parent.absolute()
PYTHON_EXEC = "/home/thunderboltdy/miniconda3/envs/ai-stocks/bin/python"
ORCHESTRATOR = BASE_DIR / "orchestrate_training.py"
BACKTESTER = BASE_DIR / "test_diverse_universe.py"
LOG_FILE = BASE_DIR / "autonomous_loop.log"

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("AutonomousLoop")

def run_performance_check():
    """Run backtests and return results."""
    logger.info("Starting performance check (Diverse Universe Backtest)...")
    cmd = [PYTHON_EXEC, str(BACKTESTER), "--days", "60"]
    
    try:
        # Run backtest script
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        
        # Find the latest report
        report_dir = BASE_DIR / "diverse_validation_results"
        reports = sorted(report_dir.glob("diverse_report_*.json"), key=os.path.getmtime)
        if not reports:
            logger.error("No reports found after backtest run.")
            return None
        
        latest_report = reports[-1]
        with open(latest_report, 'r') as f:
            return json.load(f)
            
    except Exception as e:
        logger.error(f"Error during performance check: {e}")
        return None

def analyze_and_retrain(report):
    """Check if any symbol needs retraining."""
    if not report:
        return
        
    for symbol, metrics in report.get("symbols", {}).items():
        sharpe = metrics.get("sharpe", 0)
        # Handle cases where dir_acc might be a string with %
        acc_str = metrics.get("dir_acc", "0%").replace("%", "")
        try:
            accuracy = float(acc_str) / 100.0
        except:
            accuracy = 0
            
        logger.info(f"Symbol {symbol}: Sharpe={sharpe:.2f}, Accuracy={accuracy:.2%}")
        
        needs_retrain = False
        reason = ""
        
        if sharpe < THRESHOLD_SHARPE:
            needs_retrain = True
            reason = f"Sharpe {sharpe:.2f} < {THRESHOLD_SHARPE}"
        elif accuracy < THRESHOLD_ACCURACY:
            needs_retrain = True
            reason = f"Accuracy {accuracy:.2%} < {THRESHOLD_ACCURACY}"
            
        if needs_retrain:
            logger.warning(f"RETRAIN TRIGGERED for {symbol}. Reason: {reason}")
            trigger_retraining(symbol)
        else:
            logger.info(f"Symbol {symbol} performance is healthy.")

def trigger_retraining(symbol):
    """Launch the orchestrator for a specific symbol."""
    logger.info(f"Launching Model Orchestrator for {symbol}...")
    # Add LD_LIBRARY_PATH for GPU
    env = os.environ.copy()
    try:
        # Find nvidia libs
        import site
        site_packages = site.getsitepackages()[0]
        nvidia_path = Path(site_packages) / "nvidia"
        if nvidia_path.exists():
            lib_dirs = [str(d / "lib") for d in nvidia_path.iterdir() if (d / "lib").exists()]
            if lib_dirs:
                current_ld = env.get("LD_LIBRARY_PATH", "")
                env["LD_LIBRARY_PATH"] = ":".join(lib_dirs) + (":" + current_ld if current_ld else "")
    except:
        pass

    cmd = [PYTHON_EXEC, str(ORCHESTRATOR), symbol]
    try:
        # Run asynchronously to not block the loop (could take hours)
        subprocess.Popen(cmd, env=env)
        logger.info(f"Orchestrator started for {symbol} in background.")
    except Exception as e:
        logger.error(f"Failed to start orchestrator for {symbol}: {e}")

def main_loop():
    logger.info("=== AI-Stocks Autonomous Loop Started ===")
    while True:
        report = run_performance_check()
        analyze_and_retrain(report)
        
        logger.info(f"Sleeping for {CHECK_INTERVAL_HOURS} hours...")
        time.sleep(CHECK_INTERVAL_HOURS * 3600)

if __name__ == "__main__":
    if "--run-once" in sys.argv:
        report = run_performance_check()
        analyze_and_retrain(report)
    else:
        main_loop()
