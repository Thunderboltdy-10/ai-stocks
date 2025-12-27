#!/bin/bash
# run_smart_training_sequential.sh
# Ensures GPU training runs one-by-one to prevent system freezes.

# Set up LD_LIBRARY_PATH for the entire script
echo "Configuring NVIDIA library paths..."
EXPORT_LD="export LD_LIBRARY_PATH=$(find /home/thunderboltdy/miniconda3/envs/ai-stocks/lib/python3.10/site-packages/nvidia -name "lib" -type d | tr '\n' ':')"
eval $EXPORT_LD

PYTHON_BIN="/home/thunderboltdy/miniconda3/envs/ai-stocks/bin/python"
SCRIPT="training/train_smart_regressor.py"

echo "Starting sequential training pipeline..."

echo "[1/4] Training IWM PatchTST..."
$PYTHON_BIN -u $SCRIPT IWM --model-type patchtst --epochs 30

echo "[2/4] Training IWM LSTM-Transformer..."
$PYTHON_BIN -u $SCRIPT IWM --model-type lstm_transformer --epochs 30

echo "[3/4] Training ASML PatchTST..."
$PYTHON_BIN -u $SCRIPT ASML --model-type patchtst --epochs 30

echo "[4/4] Training KO PatchTST..."
$PYTHON_BIN -u $SCRIPT KO --model-type patchtst --epochs 30

echo "ALL training tasks complete."
