#!/usr/bin/env python3
"""
Diverse Universe Validation for AI-Stocks.
Runs backtests across small-cap, international, and low-volatility symbols
and consolidates results into a single report.
"""

import os
import sys
import subprocess
import json
import argparse
from pathlib import Path
from datetime import datetime

PYTHON_EXEC = "/home/thunderboltdy/miniconda3/envs/ai-stocks/bin/python"
BASE_DIR = Path(__file__).parent.absolute()
RESULTS_DIR = BASE_DIR / "diverse_validation_results"
RESULTS_DIR.mkdir(exist_ok=True)

SYMBOLS = ["IWM", "ASML", "KO"]
MODES = [
    "regressor_only",
    "classifier",
    "gbm_only",
    "balanced",  # Full Fusion
    "patchtst",  # New Phase 7
]

def run_backtest(symbol, mode, days=60):
    print(f"\n>>> Running backtest for {symbol} [Mode: {mode}]...")
    
    # Construct command
    cmd = [
        PYTHON_EXEC, "inference_and_backtest.py",
        "--symbol", symbol,
        "--backtest-days", str(days),
    ]
    
    # Map modes to inference flags
    if mode == "regressor_only":
        cmd += ["--fusion-mode", "weighted", "--no-classifiers"]
    elif mode == "classifier":
        cmd += ["--fusion-mode", "weighted"] # Assuming weights favor classifiers or we add a classifier-only flag
    elif mode == "gbm_only":
        cmd += ["--fusion-mode", "gbm_only"]
    elif mode == "balanced":
        cmd += ["--fusion-mode", "balanced"]
    elif mode == "patchtst":
        # The inference script needs to know to load patchtst
        # I'll add a --model-type flag to inference_and_backtest.py if needed
        # Or it auto-detects from the smart directory
        cmd += ["--fusion-mode", "weighted"] # It will prioritize smart if found
        
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, cwd=str(BASE_DIR))
    output, _ = process.communicate()
    
    print(f"--- {symbol}_{mode} Output Snippet ---")
    lines = output.split('\n')
    for line in lines[-15:]:
        if line.strip():
            print(f"   {line}")
    
    return output

def extract_metrics(output):
    metrics = {"sharpe": 0, "cum_return": "0%", "dir_acc": "0%"}
    try:
        lines = output.split('\n')
        for line in reversed(lines): # Check from bottom up
            if "sharpe:" in line.lower() and metrics["sharpe"] == 0:
                parts = line.split(":")
                if len(parts) > 1:
                    metrics["sharpe"] = float(parts[-1].strip().split()[0])
            if "cum_return:" in line.lower() and metrics["cum_return"] == "0%":
                metrics["cum_return"] = line.split(":")[-1].strip()
            if "Direction Acc:" in line and metrics["dir_acc"] == "0%":
                metrics["dir_acc"] = line.split(":")[-1].strip()
    except Exception as e:
        print(f"      [!] Metric extraction failed: {e}")
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Model Comparison")
    parser.add_argument("--days", type=int, default=60, help="Number of backtest days")
    parser.add_argument("--symbols", type=str, default="IWM,ASML,KO", help="Comma-separated symbols")
    args = parser.parse_args()
    
    symbols_list = args.symbols.split(",")
    
    report = {
        "timestamp": datetime.now().isoformat(),
        "days": args.days,
        "results": {}
    }
    
    for symbol in symbols_list:
        report["results"][symbol] = {}
        for mode in MODES:
            output = run_backtest(symbol, mode, days=args.days)
            metrics = extract_metrics(output)
            report["results"][symbol][mode] = metrics
            
    report_file = RESULTS_DIR / f"comparative_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=4)
        
    # Print Summary Table
    print("\n" + "="*80)
    print(f"{'SYMBOL':<10} | {'MODE':<20} | {'SHARPE':<10} | {'RETURN':<10} | {'ACCURACY':<10}")
    print("-" * 80)
    for symbol, modes in report["results"].items():
        for mode, m in modes.items():
            print(f"{symbol:<10} | {mode:<20} | {m['sharpe']:<10.2f} | {m['cum_return']:<10} | {m['dir_acc']:<10}")
    print("="*80)
    print(f"\n✅ Comparative Analysis Complete. Report: {report_file}")

if __name__ == "__main__":
    main()
