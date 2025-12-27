#!/bin/bash
# Script to monitor AAPL regressor training progress
# Usage: ./monitor_training.sh

echo "====================================================================="
echo "AAPL Regressor Training Monitor"
echo "====================================================================="
echo "Current time: $(date)"
echo ""

# Find latest log file
LATEST_LOG=$(ls -t /home/thunderboltdy/ai-stocks/python-ai-service/training_logs/AAPL_regressor_*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    echo "ERROR: No training log found!"
    exit 1
fi

echo "Latest log: $LATEST_LOG"
echo "Log size: $(wc -l < "$LATEST_LOG") lines"
echo ""

# Check if process is running
if ps aux | grep -q "[t]rain_1d_regressor_final.py AAPL"; then
    echo "STATUS: Training process is RUNNING"
    PID=$(ps aux | grep "[t]rain_1d_regressor_final.py AAPL" | awk '{print $2}')
    echo "PID: $PID"
    echo ""
else
    echo "STATUS: No training process found"
    echo ""
fi

echo "====================================================================="
echo "Recent log entries (last 30 lines):"
echo "====================================================================="
tail -30 "$LATEST_LOG"

echo ""
echo "====================================================================="
echo "Epoch Progress (if any):"
echo "====================================================================="
grep -E "Epoch [0-9]+/|R²|loss:|dir_acc" "$LATEST_LOG" | tail -20

echo ""
echo "====================================================================="
echo "Errors/Warnings:"
echo "====================================================================="
grep -iE "error|exception|nan|inf|failed" "$LATEST_LOG" | tail -10 || echo "No errors found"

echo ""
echo "====================================================================="
echo "End of monitor report"
echo "====================================================================="
