#!/bin/bash
# =============================================================================
# RETRAIN COLLAPSED MODELS
# =============================================================================
# This script retrains models that have experienced variance collapse.
# Uses the new AntiCollapseDirectionalLoss to prevent constant predictions.
#
# Usage:
#   ./scripts/retrain_collapsed_models.sh           # Retrain all collapsed
#   ./scripts/retrain_collapsed_models.sh NVDA      # Retrain specific symbol
#   ./scripts/retrain_collapsed_models.sh --all     # Force retrain ALL symbols
#
# Date: December 16, 2025
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
EPOCHS=100
BATCH_SIZE=512
SEQUENCE_LENGTH=90

# Known collapsed symbols (from analysis)
COLLAPSED_SYMBOLS=("NVDA" "SPY")

# All symbols to potentially retrain
ALL_SYMBOLS=("AAPL" "MSFT" "NVDA" "SPY" "TSLA")

echo ""
echo "=============================================="
echo "  ANTI-COLLAPSE MODEL RETRAINING"
echo "=============================================="
echo ""
echo "Using AntiCollapseDirectionalLoss to prevent variance collapse"
echo ""

cd "$PROJECT_DIR"

# Function to retrain a single symbol
retrain_symbol() {
    local symbol=$1
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BLUE}  Retraining: ${symbol}${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
    
    # Backup old model if exists
    if [ -d "saved_models/${symbol}" ]; then
        backup_dir="saved_models/${symbol}_backup_$(date +%Y%m%d_%H%M%S)"
        echo -e "${YELLOW}Backing up existing model to ${backup_dir}${NC}"
        cp -r "saved_models/${symbol}" "$backup_dir"
    fi
    
    # Train regressor with anti-collapse loss
    echo -e "${GREEN}Training 1-day regressor with anti-collapse loss...${NC}"
    python training/train_1d_regressor_final.py "$symbol" \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --sequence-length "$SEQUENCE_LENGTH" \
        --use-anti-collapse-loss \
        --variance-regularization 0.5 \
        --seed 42
    
    # Check if training succeeded
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ Regressor training complete for ${symbol}${NC}"
    else
        echo -e "${RED}✗ Regressor training FAILED for ${symbol}${NC}"
        return 1
    fi
    
    # Train GBM baseline
    echo ""
    echo -e "${GREEN}Training GBM baseline...${NC}"
    python training/train_gbm_baseline.py "$symbol"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}✓ GBM training complete for ${symbol}${NC}"
    else
        echo -e "${YELLOW}⚠ GBM training had issues for ${symbol}${NC}"
    fi
    
    echo ""
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}  Completed: ${symbol}${NC}"
    echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo ""
}

# Parse arguments
if [ "$1" == "--all" ]; then
    echo "Retraining ALL symbols..."
    SYMBOLS_TO_TRAIN=("${ALL_SYMBOLS[@]}")
elif [ -n "$1" ]; then
    echo "Retraining specific symbol: $1"
    SYMBOLS_TO_TRAIN=("$1")
else
    echo "Retraining collapsed symbols only: ${COLLAPSED_SYMBOLS[*]}"
    SYMBOLS_TO_TRAIN=("${COLLAPSED_SYMBOLS[@]}")
fi

echo ""

# Retrain each symbol
FAILED_SYMBOLS=()
SUCCESS_SYMBOLS=()

for symbol in "${SYMBOLS_TO_TRAIN[@]}"; do
    if retrain_symbol "$symbol"; then
        SUCCESS_SYMBOLS+=("$symbol")
    else
        FAILED_SYMBOLS+=("$symbol")
    fi
done

# Summary
echo ""
echo "=============================================="
echo "  RETRAINING SUMMARY"
echo "=============================================="
echo ""

if [ ${#SUCCESS_SYMBOLS[@]} -gt 0 ]; then
    echo -e "${GREEN}✓ Successfully retrained: ${SUCCESS_SYMBOLS[*]}${NC}"
fi

if [ ${#FAILED_SYMBOLS[@]} -gt 0 ]; then
    echo -e "${RED}✗ Failed to retrain: ${FAILED_SYMBOLS[*]}${NC}"
    exit 1
fi

echo ""
echo "Next steps:"
echo "  1. Run inference: python inference_and_backtest.py <SYMBOL>"
echo "  2. Check prediction variance in backtest results"
echo "  3. Verify sharpe ratio > 1.0"
echo ""
echo "=============================================="
