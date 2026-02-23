"""Inventory-aware long-only execution backtester.

This simulator enforces practical execution barriers:
- No short inventory by default.
- You cannot sell more shares than currently held.
- Transaction costs and margin interest are charged explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np


@dataclass
class ExecutionBacktestResults:
    dates: np.ndarray
    prices: np.ndarray
    target_positions: np.ndarray
    effective_positions: np.ndarray
    equity_curve: np.ndarray
    daily_returns: np.ndarray
    metrics: Dict[str, float]
    trade_log: List[Dict[str, object]]
    buy_hold_equity: np.ndarray
    total_costs: float
    total_commission: float
    total_slippage: float
    total_margin_interest: float


class LongOnlyExecutionBacktester:
    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_pct: float = 0.0005,
        slippage_pct: float = 0.0003,
        margin_interest_annual: float = 0.06,
        min_trade_notional: float = 10.0,
        min_position_change: float = 0.0,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.commission_pct = float(max(0.0, commission_pct))
        self.slippage_pct = float(max(0.0, slippage_pct))
        self.margin_interest_annual = float(max(0.0, margin_interest_annual))
        self.min_trade_notional = float(max(0.0, min_trade_notional))
        self.min_position_change = float(max(0.0, min_position_change))

    @staticmethod
    def _safe_sharpe(daily_returns: np.ndarray) -> float:
        std = float(np.std(daily_returns))
        if std <= 1e-12:
            return 0.0
        return float(np.sqrt(252.0) * np.mean(daily_returns) / std)

    @staticmethod
    def _max_drawdown(equity_curve: np.ndarray) -> float:
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = np.where(running_max > 0, equity_curve / running_max - 1.0, 0.0)
        return float(np.min(drawdown))

    def backtest(
        self,
        *,
        dates: Iterable,
        prices: np.ndarray,
        target_positions: np.ndarray,
        max_long: float = 1.6,
        apply_position_lag: bool = True,
    ) -> ExecutionBacktestResults:
        dates_arr = np.asarray(list(dates))
        prices_arr = np.asarray(prices, dtype=float).reshape(-1)
        targets = np.asarray(target_positions, dtype=float).reshape(-1)

        if len(prices_arr) == 0:
            raise ValueError("prices cannot be empty")
        if len(prices_arr) != len(targets):
            raise ValueError("prices and target_positions must have same length")

        if apply_position_lag:
            lagged = np.zeros_like(targets)
            lagged[1:] = targets[:-1]
            targets = lagged

        targets = np.clip(targets, 0.0, max_long)

        n = len(prices_arr)
        equity_curve = np.zeros(n, dtype=float)
        buy_hold_equity = np.zeros(n, dtype=float)
        daily_returns = np.zeros(n, dtype=float)
        effective_positions = np.zeros(n, dtype=float)

        equity_curve[0] = self.initial_capital
        buy_hold_equity[0] = self.initial_capital

        cash = self.initial_capital
        shares = 0.0
        trade_log: List[Dict[str, object]] = []

        total_commission = 0.0
        total_slippage = 0.0
        total_margin_interest = 0.0
        trade_count = 0

        # Simulate close-to-close trading: rebalance at close t, PnL realized at close t+1.
        for i in range(n - 1):
            px = float(prices_arr[i])
            next_px = float(prices_arr[i + 1])

            equity_before_trade = cash + shares * px
            equity_before_trade = max(equity_before_trade, 1e-6)

            target_weight = float(np.clip(targets[i], 0.0, max_long))
            target_value = target_weight * equity_before_trade
            current_value = shares * px
            delta_value = target_value - current_value
            current_weight = current_value / max(equity_before_trade, 1e-9)

            executed_shares = 0.0
            commission = 0.0
            slippage_cost = 0.0
            blocked_reason = None
            action = None

            if (
                abs(delta_value) >= self.min_trade_notional
                and px > 0
                and abs(target_weight - current_weight) >= self.min_position_change
            ):
                if delta_value > 0:
                    # Buy side (can borrow via negative cash up to max_long target).
                    exec_px = px * (1.0 + self.slippage_pct)
                    buy_shares = delta_value / max(exec_px, 1e-12)
                    trade_notional = buy_shares * exec_px
                    commission = trade_notional * self.commission_pct
                    slippage_cost = buy_shares * (exec_px - px)
                    cash -= trade_notional + commission
                    shares += buy_shares
                    executed_shares = buy_shares
                    action = "BUY"
                else:
                    # Sell side with strict inventory barrier.
                    exec_px = px * (1.0 - self.slippage_pct)
                    requested_sell = abs(delta_value) / max(exec_px, 1e-12)
                    sell_shares = min(requested_sell, shares)
                    if sell_shares + 1e-12 < requested_sell:
                        blocked_reason = "no_inventory"
                    trade_notional = sell_shares * exec_px
                    commission = trade_notional * self.commission_pct
                    slippage_cost = sell_shares * (px - exec_px)
                    cash += trade_notional - commission
                    shares -= sell_shares
                    executed_shares = -sell_shares
                    action = "SELL"

                if abs(executed_shares) > 1e-12:
                    trade_count += 1
                    total_commission += commission
                    total_slippage += slippage_cost
                    trade_log.append(
                        {
                            "index": i,
                            "date": dates_arr[i],
                            "action": action,
                            "price": px,
                            "executed_price": exec_px,
                            "shares": float(abs(executed_shares)),
                            "signed_shares": float(executed_shares),
                            "commission": float(commission),
                            "slippage": float(slippage_cost),
                            "target_weight": float(target_weight),
                            "blocked_reason": blocked_reason,
                            "cash_after": float(cash),
                            "shares_after": float(shares),
                        }
                    )

            # Margin interest on borrowed cash.
            if cash < 0:
                interest = (-cash) * (self.margin_interest_annual / 252.0)
                cash -= interest
                total_margin_interest += interest

            equity_next = cash + shares * next_px
            equity_curve[i + 1] = equity_next
            daily_returns[i + 1] = equity_next / max(equity_curve[i], 1e-9) - 1.0
            effective_positions[i] = (shares * px) / max(equity_before_trade, 1e-9)

            # 1x buy-and-hold benchmark.
            buy_hold_equity[i + 1] = buy_hold_equity[i] * (next_px / max(px, 1e-12))

        if n >= 1:
            final_equity = cash + shares * float(prices_arr[-1])
            equity_curve[-1] = final_equity
            if n > 1:
                effective_positions[-1] = (shares * float(prices_arr[-1])) / max(final_equity, 1e-9)

        turnover = float(np.mean(np.abs(np.diff(effective_positions, prepend=0.0)))) if n > 0 else 0.0
        total_costs = total_commission + total_slippage + total_margin_interest

        strategy_cum = float(equity_curve[-1] / self.initial_capital - 1.0)
        buy_hold_cum = float(buy_hold_equity[-1] / self.initial_capital - 1.0)

        metrics = {
            "cum_return": strategy_cum,
            "sharpe": self._safe_sharpe(daily_returns[1:]),
            "max_drawdown": self._max_drawdown(equity_curve),
            "win_rate": float(np.mean(daily_returns[1:] > 0.0)) if n > 1 else 0.0,
            "buy_hold_cum_return": buy_hold_cum,
            "alpha": strategy_cum - buy_hold_cum,
            "turnover": turnover,
            "trade_count": float(trade_count),
            "total_transaction_costs": float(total_costs),
            "total_commission": float(total_commission),
            "total_slippage": float(total_slippage),
            "total_margin_interest": float(total_margin_interest),
        }

        return ExecutionBacktestResults(
            dates=dates_arr,
            prices=prices_arr,
            target_positions=targets,
            effective_positions=effective_positions,
            equity_curve=equity_curve,
            daily_returns=daily_returns,
            metrics=metrics,
            trade_log=trade_log,
            buy_hold_equity=buy_hold_equity,
            total_costs=float(total_costs),
            total_commission=float(total_commission),
            total_slippage=float(total_slippage),
            total_margin_interest=float(total_margin_interest),
        )
