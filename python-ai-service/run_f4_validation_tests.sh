#!/bin/bash

# F4 Validation Test Runner
# This script runs comprehensive validation tests on all models

set -e

# Setup
cd /home/thunderboltdy/ai-stocks/python-ai-service
source ~/miniconda3/etc/profile.d/conda.sh
conda activate ai-stocks

# Create results directory
mkdir -p f4_validation_results

echo "========================================="
echo "Starting F4 Validation Tests"
echo "========================================="
echo ""

# Regressor Tests (5 symbols)
echo "=== Running Regressor Tests ==="
for symbol in AAPL ASML IWM KO MSFT; do
    echo "Testing regressor for $symbol..."
    TEST_SYMBOL=$symbol python tests/test_regressor_standalone.py > f4_validation_results/test_regressor_${symbol}.log 2>&1
    if [ $? -eq 0 ]; then
        echo "  ✓ $symbol regressor test completed"
    else
        echo "  ✗ $symbol regressor test failed"
    fi
done

echo ""
echo "=== Running Classifier Tests ==="
# Classifier Tests (4 symbols)
for symbol in AAPL ASML IWM KO; do
    echo "Testing classifiers for $symbol..."
    TEST_SYMBOL=$symbol python tests/test_classifier_standalone.py > f4_validation_results/test_classifier_${symbol}.log 2>&1
    if [ $? -eq 0 ]; then
        echo "  ✓ $symbol classifier test completed"
    else
        echo "  ✗ $symbol classifier test failed"
    fi
done

echo ""
echo "========================================="
echo "All Tests Completed"
echo "========================================="
echo "Results saved to: f4_validation_results/"
echo ""
echo "Next step: Run the parser to consolidate results"
echo "  python parse_f4_validation_results.py"
