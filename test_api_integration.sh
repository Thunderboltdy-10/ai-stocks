#!/bin/bash

# API Integration Test Script
# Tests Next.js API endpoints and Python inference backend

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}AI-Stocks API Integration Test Suite${NC}"
echo -e "${BLUE}========================================${NC}\n"

# Configuration
NEXT_JS_URL="http://localhost:3000"
PYTHON_API_URL="http://localhost:8000"
TEST_SYMBOL="AAPL"

# Helper function for API calls
test_endpoint() {
    local method=$1
    local url=$2
    local data=$3
    local description=$4

    echo -e "${YELLOW}Testing:${NC} $description"
    echo -e "  ${BLUE}$method $url${NC}"

    if [ -z "$data" ]; then
        response=$(curl -s -X "$method" "$url" \
            -H "Content-Type: application/json" \
            -w "\n%{http_code}")
    else
        echo -e "  ${BLUE}Data:${NC} $data"
        response=$(curl -s -X "$method" "$url" \
            -H "Content-Type: application/json" \
            -d "$data" \
            -w "\n%{http_code}")
    fi

    # Extract status code and body
    http_code=$(echo "$response" | tail -n 1)
    body=$(echo "$response" | sed '$d')

    if [ "$http_code" -eq 200 ] || [ "$http_code" -eq 201 ]; then
        echo -e "  ${GREEN}✓ Success (HTTP $http_code)${NC}"
        echo -e "  ${BLUE}Response:${NC}"
        echo "$body" | jq '.' 2>/dev/null || echo "$body"
    else
        echo -e "  ${RED}✗ Failed (HTTP $http_code)${NC}"
        echo -e "  ${BLUE}Response:${NC}"
        echo "$body" | jq '.' 2>/dev/null || echo "$body"
    fi
    echo ""
}

# Check if servers are running
echo -e "${YELLOW}Checking server availability...${NC}\n"

if ! curl -s "$NEXT_JS_URL" > /dev/null 2>&1; then
    echo -e "${RED}✗ Next.js server not running at $NEXT_JS_URL${NC}"
    echo -e "  ${YELLOW}Start with: npm run dev${NC}\n"
fi

if ! curl -s "$PYTHON_API_URL" > /dev/null 2>&1; then
    echo -e "${RED}✗ Python API server not running at $PYTHON_API_URL${NC}"
    echo -e "  ${YELLOW}Start with: cd python-ai-service && python app.py${NC}\n"
fi

echo ""

# Test 1: Python API Health Check
echo -e "${BLUE}[TEST 1]${NC} Python API Health Check"
test_endpoint "GET" "$PYTHON_API_URL/api/health" "" "Health check"

# Test 2: List Available Models
echo -e "${BLUE}[TEST 2]${NC} List Available Models"
test_endpoint "GET" "$PYTHON_API_URL/api/models" "" "List models from Python API"

# Test 3: Get Specific Model
echo -e "${BLUE}[TEST 3]${NC} Get Specific Model"
test_endpoint "GET" "$PYTHON_API_URL/api/models/$TEST_SYMBOL" "" "Get model metadata for $TEST_SYMBOL"

# Test 4: Python Direct Prediction (Regressor Only)
echo -e "${BLUE}[TEST 4]${NC} Python Direct Prediction (Regressor + Conservative Risk)"
test_endpoint "POST" "$PYTHON_API_URL/api/predict" \
    "{\"symbol\": \"$TEST_SYMBOL\", \"horizon\": 5, \"daysOnChart\": 120}" \
    "Direct Python prediction endpoint"

# Test 5: Python Ensemble Prediction
echo -e "${BLUE}[TEST 5]${NC} Python Ensemble Prediction (Hybrid Fusion)"
test_endpoint "POST" "$PYTHON_API_URL/api/predict" \
    "{\"symbol\": \"$TEST_SYMBOL\", \"horizon\": 5, \"daysOnChart\": 120, \"fusion\": {\"mode\": \"weighted\"}}" \
    "Python ensemble prediction with weighted fusion"

# Test 6: Next.js Prediction Proxy (if Next.js is running)
if curl -s "$NEXT_JS_URL" > /dev/null 2>&1; then
    echo -e "${BLUE}[TEST 6]${NC} Next.js Prediction Proxy Endpoint"
    test_endpoint "POST" "$NEXT_JS_URL/api/predict" \
        "{\"symbol\": \"$TEST_SYMBOL\"}" \
        "Next.js proxy to Python predict endpoint"

    # Test 7: Next.js Ensemble Proxy (if Next.js is running)
    echo -e "${BLUE}[TEST 7]${NC} Next.js Ensemble Proxy Endpoint"
    test_endpoint "POST" "$NEXT_JS_URL/api/predict-ensemble" \
        "{\"symbol\": \"$TEST_SYMBOL\", \"riskProfile\": \"conservative\"}" \
        "Next.js proxy to Python ensemble endpoint"
else
    echo -e "${YELLOW}[SKIPPED]${NC} Next.js tests (server not running)"
fi

# Test 8: Backtest Endpoint
echo -e "${BLUE}[TEST 8]${NC} Backtest Endpoint"
test_endpoint "POST" "$PYTHON_API_URL/api/backtest" \
    "{\"prediction\": {\"symbol\": \"$TEST_SYMBOL\", \"prediction\": 0.015, \"confidence\": 0.65}, \"params\": {\"backtestWindow\": 60, \"initialCapital\": 10000}}" \
    "Backtest with sample prediction"

# Test 9: Historical Data Endpoint
echo -e "${BLUE}[TEST 9]${NC} Historical Data Endpoint"
test_endpoint "GET" "$PYTHON_API_URL/api/historical/$TEST_SYMBOL?days=30" "" \
    "Get historical OHLCV data for $TEST_SYMBOL"

# Test 10: Real-time Price Endpoint
echo -e "${BLUE}[TEST 10]${NC} Real-time Price Endpoint"
test_endpoint "GET" "$PYTHON_API_URL/api/realtime/$TEST_SYMBOL" "" \
    "Get real-time price for $TEST_SYMBOL"

echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Test suite complete!${NC}"
echo -e "${BLUE}========================================${NC}\n"

echo -e "${YELLOW}Summary:${NC}"
echo -e "  ${BLUE}Next.js API${NC}: http://localhost:3000/api/"
echo -e "  ${BLUE}Python API${NC}: http://localhost:8000/api/"
echo -e "  ${BLUE}Test Symbol${NC}: $TEST_SYMBOL\n"

echo -e "${YELLOW}Next Steps:${NC}"
echo -e "  1. Run: npm run dev                     # Start Next.js frontend"
echo -e "  2. Run: cd python-ai-service && python app.py  # Start Python backend"
echo -e "  3. Run: ./test_api_integration.sh      # Run this test suite\n"
