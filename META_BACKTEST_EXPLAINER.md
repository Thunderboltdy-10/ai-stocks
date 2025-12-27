# META Backtest Deep Dive: How It All Works

> **Real data from**: META GBM-Only Backtest (Sep 26 - Dec 19, 2025)  
> **Result**: +23.04% return vs Buy-Hold -12.56% | Sharpe Ratio: 6.0

---

## 📊 The Position System Explained

### This is NOT Buy/Sell Order-Based Trading

The backtest uses a **Portfolio Rebalancing Model**, not individual buy/sell orders. Each day, the model outputs a **target position as a fraction of current equity**.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         POSITION VALUE MEANING                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  +1.0  │████████████████████████████████████│  100% LONG (all-in bullish)  │
│  +0.5  │██████████████████                  │   50% LONG (half position)   │
│   0.0  │                                    │   FLAT (100% cash)           │
│  -0.5  │                  ██████████████████│   50% SHORT (bearish bet)    │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Insight: The Model Calculates TRANSITIONS Automatically

When the position changes from one day to the next, the backtester calculates what trades are needed:

```
                    AUTOMATIC POSITION TRANSITIONS
                    ════════════════════════════════

Day 1:  Position = +0.5                    Day 2:  Position = -0.5
        ┌─────────┐                                ┌─────────┐
        │ 50%     │                                │         │
        │ LONG    │          TRANSITION            │ 50%     │
        │         │       ═══════════════▶         │ SHORT   │
        │         │       Sell all longs           │         │
        │         │       + Open short             │         │
        └─────────┘                                └─────────┘

    What actually happens:
    1. SELL 100% of long position (50% → 0%)
    2. SELL SHORT an additional 50% (0% → -50%)
    = Total delta: -1.0 (position changed by -100%)
```

---

## 🎯 META Backtest: Real Example Walkthrough

### Day-by-Day Analysis of Critical Period (Oct 28-31, 2025)

This is where the model made its biggest winning trade - correctly shorting META before the crash.

```
════════════════════════════════════════════════════════════════════════════════
                        THE META CRASH: Oct 28-31, 2025
════════════════════════════════════════════════════════════════════════════════

     Price                                                    
     $751 ┤ ●────●                                              
          │      \                                              
     $700 ┤       \                                             
          │        \                                            
     $666 ┤         ●                                          
          │          \                                          
     $648 ┤           ●                                         
          │                                                     
          └────┬────┬────┬────┬────                            
             Oct27 Oct28 Oct29 Oct30 Oct31                      

    Date      Price    Position   Model Action     Daily P&L
    ─────────────────────────────────────────────────────────
    Oct 27    $750.21   +5.8%     Slight long      (waiting)
    Oct 28    $750.83   -50%      GO FULL SHORT    ◀── Model flips!
    Oct 29    $751.06   -50%      HOLD SHORT       Price stable
    Oct 30    $665.93   -50%      HOLD SHORT       +$42.57 gain! ★
    Oct 31    $647.82   -35%      REDUCE SHORT     +$9.07 more
    ─────────────────────────────────────────────────────────

    RESULT: Model was SHORT when META crashed 14%
            Strategy gained while buy-hold lost ~$1,000
```

### Actual Trade Log Data (from confidence_trade_log.csv):

| Index | Date       | Price   | Position | Confidence | Reasoning                        |
|-------|------------|---------|----------|------------|----------------------------------|
| 21    | Oct 27     | $750.21 | +5.8%    | 0.809      | regressor_pred=-0.00645          |
| 22    | Oct 28     | $750.83 | **-50%** | 1.000      | regressor_pred=-0.01389 ⚠️       |
| 23    | Oct 29     | $751.06 | **-50%** | 1.000      | regressor_pred=-0.01465 ⚠️       |
| 24    | Oct 30     | $665.93 | **-50%** | 1.000      | regressor_pred=-0.01807 (crash!) |
| 25    | Oct 31     | $647.82 | **-35%** | 1.000      | regressor_pred=-0.01505          |

**Key observation**: The GBM model predicted -1.39% to -1.81% returns with 100% confidence **before** the crash. It went max short and held through the drop.

---

## 💰 Position Sizing: How Equity Limits Work

### The Budget Is ALWAYS Respected

```python
# From advanced_backtester.py line 164
position = float(np.clip(positions[i], -max_short, max_long))
#                                       ↑          ↑
#                                    -0.5       +1.0
```

Position is **clipped** to the allowed range:
- **Max Long**: 100% of equity (never more than you have)
- **Max Short**: 50% of equity (configurable, conservative default)

### Real Calculation Example

```
═══════════════════════════════════════════════════════════════════════════
                    POSITION SIZING MATH: OCT 6, 2025
═══════════════════════════════════════════════════════════════════════════

Inputs:
  • Current Equity: $10,058.15
  • Target Position: +1.0 (100% long)
  • Previous Position: +0.09% (nearly flat)
  • META Price: $715.08

Calculation:
  1. Position delta needed:
     Δ = 1.0 - 0.0009 = 0.9991 (~100% change)

  2. Dollar amount to invest:
     Trade Value = |Δ| × Equity = 0.9991 × $10,058.15 = $10,049.10

  3. Shares to buy:
     Shares = $10,049.10 ÷ $715.08 = 14.05 shares

  4. Transaction costs:
     Commission = max($10,049 × 0.001, $1.00) = $10.05
     Slippage   = $10,049 × 0.0005 = $5.02
     Total Cost = $15.07

Result:
  ✓ Position uses exactly 100% of equity
  ✓ No leverage or borrowing needed
  ✓ Budget constraint satisfied
═══════════════════════════════════════════════════════════════════════════
```

### Position Sizing Scales With Equity

```
Day 0:   Equity = $10,000  →  Max Long = $10,000
Day 30:  Equity = $11,500  →  Max Long = $11,500  (grew with profits!)
Day 60:  Equity = $12,304  →  Max Long = $12,304

     ┌───────────────────────────────────────────────────────┐
     │  Position limits are FRACTIONS, not fixed amounts!   │
     │  As equity grows, max position dollar size grows too │
     └───────────────────────────────────────────────────────┘
```

---

## 📈 What the Charts Mean

### Equity Curve (equity_curve.png)

```
    $12,500 ┤                                          ●───●
            │                                     ●────┘
    $12,000 ┤                                ●────┘
            │                           ●────┘
    $11,500 ┤                      ●────┘
            │            Strategy ─┼──────────────────────▶
    $11,000 ┤       ●─────────────┘
            │  ●────┘
    $10,500 ┤ ●
            │●─────────────────────────────────────────────
    $10,000 ┼
            │●
     $9,500 ┤ ●────●                    Buy & Hold
            │       \              ─ ─ ─ ─ ─ ─ ─ ─ ─▶
     $9,000 ┤        \──────●
            │               \────●────●────●────●────●
     $8,500 ┤
            └─┬────┬────┬────┬────┬────┬────┬────┬────┬──
            Sep  Oct        Nov              Dec
```

**Blue line (Strategy)**: Your model's equity over time  
**Orange line (Buy-Hold)**: If you just bought META on day 1 and held

### Position Heatmap (position_heatmap.png)

```
             POSITION INTENSITY OVER TIME
             ─────────────────────────────
    +100% █ │    █   █       ██      ███    │ Full Long
     +50%   │██      █        █   ██         │
       0%   │  █ █    █  █         █  █     │ Flat
     -50%   │    ██████████        ████████ │ Max Short
            └────┬────┬────┬────┬────┬────┬─
                Sep  Oct  Oct  Nov  Nov  Dec

    Green = Long positions (bullish bets)
    Red = Short positions (bearish bets)
    White = Flat/cash
```

---

## 📋 Understanding the Metrics

### From META Backtest Results:

| Metric | Value | What It Means |
|--------|-------|---------------|
| **Cumulative Return** | +23.04% | Total profit over period |
| **Buy-Hold Return** | -12.56% | What you'd get just holding |
| **Sharpe Ratio** | 6.0 | Risk-adjusted return (>1 is good, >3 is excellent) |
| **Max Drawdown** | -0.71% | Worst peak-to-trough loss |
| **Turnover** | ~0.27 | On average, 27% of portfolio changes daily |
| **Total Trades** | 34 | Number of position changes |

### Confidence Distribution:

```
    Confidence Tier Distribution (34 trades)
    ─────────────────────────────────────────
    Very High (90-100%)  ████████████████████ 17 trades (50%)
    High (70-90%)        ████                  2 trades
    Medium (50-70%)      ██████████            5 trades
    Low (30-50%)         ██████████            5 trades
    Very Low (0-30%)     ██████████            5 trades
```

**17 out of 34 trades had "Very High" confidence** - the model was selective and confident.

---

## ⚙️ Cost Model Breakdown

### Transaction Costs Applied:

```
╔═══════════════════════════════════════════════════════════════════╗
║                     COST STRUCTURE                                ║
╠═══════════════════════════════════════════════════════════════════╣
║  Commission Rate:     0.1% per trade (or $1.00 minimum)           ║
║  Slippage:            0.05% per trade                             ║
║  Borrowing Cost:      2.0% annual for short positions             ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  Example: $10,000 trade                                           ║
║  ─────────────────────                                            ║
║  Commission:  $10,000 × 0.001 = $10.00                           ║
║  Slippage:    $10,000 × 0.0005 = $5.00                           ║
║  Total:       $15.00 (0.15% of trade)                             ║
║                                                                   ║
║  Short Position Daily Cost:                                       ║
║  ─────────────────────────                                        ║
║  Position: $5,000 short (50% of $10,000)                         ║
║  Daily borrow: $5,000 × (2.0% / 252) = $0.40/day                 ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
```

---

## 🔄 Trade Flow Diagram

### Complete Lifecycle of a Position Change:

```
     ┌──────────────────────────────────────────────────────────────────┐
     │                    DAILY BACKTEST LOOP                          │
     └──────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
     ┌──────────────────────────────────────────────────────────────────┐
     │  1. GET MODEL PREDICTION                                         │
     │     • GBM predicts next-day return (e.g., -1.4%)                │
     │     • Calculate confidence score (0-1)                          │
     │     • Map to position: negative pred → SHORT, positive → LONG   │
     └──────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
     ┌──────────────────────────────────────────────────────────────────┐
     │  2. CLIP POSITION TO LIMITS                                      │
     │     • Raw position might be -0.8 (80% short)                    │
     │     • Clipped to max_short: min(-0.8, -0.5) = -0.5              │
     │     • Final position: -50% short                                 │
     └──────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
     ┌──────────────────────────────────────────────────────────────────┐
     │  3. CALCULATE DELTA (position change needed)                     │
     │     • Previous position: +0.3 (30% long)                        │
     │     • New position: -0.5 (50% short)                            │
     │     • Delta: -0.5 - 0.3 = -0.8 (need to sell 80% of equity)     │
     └──────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
     ┌──────────────────────────────────────────────────────────────────┐
     │  4. EXECUTE TRADE & APPLY COSTS                                  │
     │     • Trade value: 80% × $10,500 = $8,400                       │
     │     • Commission: $8.40                                          │
     │     • Slippage: $4.20                                            │
     │     • Log trade to trade_log                                     │
     └──────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
     ┌──────────────────────────────────────────────────────────────────┐
     │  5. CALCULATE DAILY RETURN                                       │
     │     daily_return = (position × asset_return) - cost_drag        │
     │                  = (-0.5 × -3%) - 0.12%                          │
     │                  = +1.5% - 0.12%                                 │
     │                  = +1.38%                                        │
     │                    ↑                                             │
     │        Short position PROFITS when stock drops!                  │
     └──────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
     ┌──────────────────────────────────────────────────────────────────┐
     │  6. UPDATE EQUITY                                                │
     │     new_equity = old_equity × (1 + daily_return)                │
     │                = $10,500 × 1.0138                                │
     │                = $10,645                                         │
     └──────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                            [Next day...]
```

---

## 📁 Output Files Explained

| File | Purpose |
|------|---------|
| `backtest.png` | Main chart with equity curves, positions, drawdowns |
| `equity_curve.png` | Strategy vs Buy-Hold comparison |
| `position_heatmap.png` | Visual of position over time |
| `dashboard.png` | Combined multi-panel view |
| `confidence_trade_log.csv` | Every trade with confidence, reasoning |
| `equity_comparison.csv` | Daily equity values for both strategies |
| `drawdowns.csv` | All drawdown periods with recovery times |
| `regime_analysis.json` | Performance by market regime (bull/bear/sideways) |
| `calibration_analysis.json` | How well confidence matches actual outcomes |
| `confidence_analysis.json` | Win rates by confidence tier |
| `exit_analysis.txt` | Profit target hit analysis |
| `backtest.pkl` | Serialized results for programmatic access |

---

## 🎮 Commands Reference

### Run GBM-Only Backtest:

```bash
cd python-ai-service

# Basic backtest (last 60 days)
python inference_and_backtest.py --symbol META --fusion-mode gbm_only

# Extended backtest (2 years)
python inference_and_backtest.py --symbol META --fusion-mode gbm_only \
    --backtest-days 504

# Custom date range
python inference_and_backtest.py --symbol META --fusion-mode gbm_only \
    --start_date 2023-01-01 --end_date 2024-12-31

# Adjust position limits
python inference_and_backtest.py --symbol META --fusion-mode gbm_only \
    --max-long 0.8 --max-short 0.3  # More conservative
```

### Get Future Predictions (NOT Backtest):

```bash
cd python-ai-service

# Single prediction for tomorrow
python inference/predict_ensemble.py META

# Forward simulation (predict next N days)
python inference_and_backtest.py --symbol META --fusion-mode gbm_only \
    --forward-sim --forward-days 5
```

### View Results:

```bash
# Open the dashboard
xdg-open backtest_results/META_*/dashboard.png

# Check metrics
cat backtest_results/META_*/regime_analysis.json | python -m json.tool

# Export trade log to spreadsheet
cp backtest_results/META_*/confidence_trade_log.csv ~/trades.csv
```

---

## ⚠️ Important Caveats

### Why These Results May Not Persist:

1. **Short test period** (60 days) - statistically insufficient
2. **One big winning trade** (META crash) dominates returns
3. **Sharpe of 6.0** is unrealistic long-term - expect 1.0-2.0
4. **No regime diversity** - mostly sideways/bearish market

### What The Model Does NOT Account For:

- ❌ Liquidity constraints (can you actually execute this volume?)
- ❌ Short borrowing availability (shares may not be available)
- ❌ Intraday price movement during execution
- ❌ Market impact of your trades

### Recommended Validation:

```bash
# Run 2-year backtest to test regime robustness
python inference_and_backtest.py --symbol META --fusion-mode gbm_only \
    --start_date 2022-01-01 --end_date 2024-12-31

# Test on different symbols
python inference_and_backtest.py --symbol AAPL --fusion-mode gbm_only
python inference_and_backtest.py --symbol GOOGL --fusion-mode gbm_only

# Check rolling Sharpe over time (should stay >1.0)
```

---

## 📊 Summary: How META Made +23%

```
                    THE WINNING FORMULA
    ═══════════════════════════════════════════════════════

    1. CORRECT MARKET DIRECTION
       • GBM predicted META would fall
       • Went short with high confidence
       • Held through the Oct 29-30 crash

    2. PROPER POSITION SIZING
       • Max 50% short (not over-leveraged)
       • Scaled with equity growth
       • Budget constraints always respected

    3. SELECTIVE HIGH-CONFIDENCE TRADES
       • 50% of trades had "Very High" confidence
       • Model stayed flat during uncertain periods
       • Avoided false signals

    4. COST-AWARE EXECUTION
       • Transaction costs modeled realistically
       • Short borrow costs included
       • Net returns after all fees

    ═══════════════════════════════════════════════════════
           RESULT: +23.04% vs Buy-Hold -12.56%
                   Sharpe Ratio: 6.0
    ═══════════════════════════════════════════════════════
```

---

*Generated from META backtest results: 2025-09-26 to 2025-12-19*
