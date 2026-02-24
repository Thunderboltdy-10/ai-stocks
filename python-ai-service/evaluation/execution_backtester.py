"""Inventory-aware execution backtester with optional shorting.

The class name is kept for compatibility with existing imports, but the
simulator now supports both long-only and long/short execution depending on
`max_short`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


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
    total_borrow_fee: float


class LongOnlyExecutionBacktester:
    def __init__(
        self,
        initial_capital: float = 100_000.0,
        commission_pct: float = 0.0005,
        slippage_pct: float = 0.0003,
        margin_interest_annual: float = 0.06,
        short_borrow_annual: float = 0.03,
        min_trade_notional: float = 10.0,
        min_position_change: float = 0.0,
    ) -> None:
        self.initial_capital = float(initial_capital)
        self.commission_pct = float(max(0.0, commission_pct))
        self.slippage_pct = float(max(0.0, slippage_pct))
        self.margin_interest_annual = float(max(0.0, margin_interest_annual))
        self.short_borrow_annual = float(max(0.0, short_borrow_annual))
        self.min_trade_notional = float(max(0.0, min_trade_notional))
        self.min_position_change = float(max(0.0, min_position_change))

    @staticmethod
    def _safe_sharpe(daily_returns: np.ndarray, annualization_factor: float = 252.0) -> float:
        std = float(np.std(daily_returns))
        if std <= 1e-12:
            return 0.0
        return float(np.sqrt(max(1.0, annualization_factor)) * np.mean(daily_returns) / std)

    @staticmethod
    def _safe_sortino(daily_returns: np.ndarray, annualization_factor: float = 252.0) -> float:
        downside = np.asarray(daily_returns, dtype=float)
        downside = downside[downside < 0.0]
        if len(downside) < 2:
            return 0.0
        dstd = float(np.std(downside))
        if dstd <= 1e-12:
            return 0.0
        return float(np.sqrt(max(1.0, annualization_factor)) * np.mean(daily_returns) / dstd)

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
        max_short: float = 0.0,
        apply_position_lag: bool = True,
        annualization_factor: float = 252.0,
        flat_at_day_end: bool = False,
        day_end_flatten_fraction: float = 1.0,
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

        targets = np.clip(targets, -max_short, max_long)
        annualization_factor = float(max(1.0, annualization_factor))
        day_end_flatten_fraction = float(np.clip(day_end_flatten_fraction, 0.0, 1.0))

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
        total_borrow_fee = 0.0
        trade_count = 0

        # Simulate close-to-close trading: rebalance at close t, PnL realized at close t+1.
        for i in range(n - 1):
            px = float(prices_arr[i])
            next_px = float(prices_arr[i + 1])
            day_change = False
            if flat_at_day_end:
                day_now = pd.to_datetime(dates_arr[i]).date()
                day_next = pd.to_datetime(dates_arr[i + 1]).date()
                day_change = day_now != day_next

            equity_before_trade = cash + shares * px
            equity_before_trade = max(equity_before_trade, 1e-6)

            target_weight = float(np.clip(targets[i], -max_short, max_long))
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
                    # Buy side: can buy from flat/long or cover existing short first.
                    exec_px = px * (1.0 + self.slippage_pct)
                    requested_shares = delta_value / max(exec_px, 1e-12)
                    cover_shares = min(requested_shares, max(0.0, -shares))
                    buy_shares = max(0.0, requested_shares - cover_shares)
                    traded_shares = cover_shares + buy_shares
                    trade_notional = traded_shares * exec_px
                    commission = trade_notional * self.commission_pct
                    slippage_cost = traded_shares * (exec_px - px)
                    cash -= trade_notional + commission
                    shares += traded_shares
                    executed_shares = traded_shares

                    if cover_shares > 1e-12 and buy_shares > 1e-12:
                        action = "COVER_BUY"
                    elif cover_shares > 1e-12:
                        action = "COVER"
                    else:
                        action = "BUY"
                else:
                    # Sell side: can sell existing long and optionally open/increase short.
                    exec_px = px * (1.0 - self.slippage_pct)
                    requested_sell = abs(delta_value) / max(exec_px, 1e-12)
                    long_inventory = max(0.0, shares)
                    sell_shares = min(requested_sell, long_inventory)
                    short_shares = max(0.0, requested_sell - sell_shares)

                    if short_shares > 1e-12:
                        if max_short <= 0.0:
                            blocked_reason = "no_inventory"
                            short_shares = 0.0
                        else:
                            max_short_shares = (max_short * equity_before_trade) / max(exec_px, 1e-12)
                            current_short_shares = max(0.0, -shares)
                            available_short = max(0.0, max_short_shares - current_short_shares)
                            if short_shares > available_short + 1e-12:
                                short_shares = available_short
                                blocked_reason = "short_cap"

                    traded_shares = sell_shares + short_shares
                    trade_notional = traded_shares * exec_px
                    commission = trade_notional * self.commission_pct
                    slippage_cost = traded_shares * (px - exec_px)
                    cash += trade_notional - commission
                    shares -= traded_shares
                    executed_shares = -traded_shares

                    if short_shares > 1e-12 and sell_shares > 1e-12:
                        action = "SELL_SHORT"
                    elif short_shares > 1e-12:
                        action = "SHORT"
                    else:
                        action = "SELL"

                if abs(executed_shares) > 1e-12:
                    trade_count += 1
                    total_commission += commission
                    total_slippage += slippage_cost
                    equity_after_trade = cash + shares * px
                    effective_weight_after = (shares * px) / max(equity_after_trade, 1e-9)
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
                            "effective_weight_after": float(effective_weight_after),
                            "blocked_reason": blocked_reason,
                            "cash_after": float(cash),
                            "shares_after": float(shares),
                        }
                    )

            if flat_at_day_end:
                if day_change and abs(shares) > 1e-12:
                    flatten_frac = day_end_flatten_fraction
                    if flatten_frac <= 1e-9:
                        flatten_frac = 0.0

                    if shares > 0:
                        shares_before = shares
                        target_shares_after = shares_before * (1.0 - flatten_frac)
                        traded_shares = max(0.0, shares_before - target_shares_after)
                        if traded_shares <= 1e-12:
                            traded_shares = 0.0
                        shares = target_shares_after
                        exec_px = px * (1.0 - self.slippage_pct)
                        trade_notional = traded_shares * exec_px
                        commission = trade_notional * self.commission_pct
                        slippage_cost = traded_shares * (px - exec_px)
                        cash += trade_notional - commission
                        action = "EOD_FLAT_SELL" if flatten_frac >= 0.999 else "EOD_REDUCE_SELL"
                        signed = -traded_shares
                    else:
                        shares_before = shares
                        target_shares_after = shares_before * (1.0 - flatten_frac)
                        traded_shares = max(0.0, abs(shares_before - target_shares_after))
                        if traded_shares <= 1e-12:
                            traded_shares = 0.0
                        shares = target_shares_after
                        exec_px = px * (1.0 + self.slippage_pct)
                        trade_notional = traded_shares * exec_px
                        commission = trade_notional * self.commission_pct
                        slippage_cost = traded_shares * (exec_px - px)
                        cash -= trade_notional + commission
                        action = "EOD_FLAT_COVER" if flatten_frac >= 0.999 else "EOD_REDUCE_COVER"
                        signed = traded_shares

                    if traded_shares > 1e-12:
                        trade_count += 1
                        total_commission += commission
                        total_slippage += slippage_cost
                        equity_after_trade = cash + shares * px
                        effective_weight_after = (shares * px) / max(equity_after_trade, 1e-9)
                        trade_log.append(
                            {
                                "index": i,
                                "date": dates_arr[i],
                                "action": action,
                                "price": px,
                                "executed_price": exec_px,
                                "shares": float(abs(traded_shares)),
                                "signed_shares": float(signed),
                                "commission": float(commission),
                                "slippage": float(slippage_cost),
                                "target_weight": float(targets[i] * (1.0 - flatten_frac)),
                                "effective_weight_after": float(effective_weight_after),
                                "blocked_reason": None,
                                "cash_after": float(cash),
                                "shares_after": float(shares),
                            }
                        )

            # Margin interest on borrowed cash.
            if cash < 0:
                interest = (-cash) * (self.margin_interest_annual / annualization_factor)
                cash -= interest
                total_margin_interest += interest
            if shares < 0:
                borrow_fee = abs(shares) * px * (self.short_borrow_annual / annualization_factor)
                cash -= borrow_fee
                total_borrow_fee += borrow_fee

            equity_next = cash + shares * next_px
            equity_curve[i + 1] = equity_next
            daily_returns[i + 1] = equity_next / max(equity_curve[i], 1e-9) - 1.0
            effective_positions[i] = (shares * px) / max(equity_before_trade, 1e-9)

            # 1x buy-and-hold benchmark.
            if flat_at_day_end and day_change:
                # Day-trading benchmark: flatten the same overnight fraction as strategy.
                overnight_exposure = float(np.clip(1.0 - day_end_flatten_fraction, 0.0, 1.0))
                if overnight_exposure <= 1e-9:
                    buy_hold_equity[i + 1] = buy_hold_equity[i]
                else:
                    buy_hold_equity[i + 1] = buy_hold_equity[i] * (
                        1.0 + overnight_exposure * (next_px / max(px, 1e-12) - 1.0)
                    )
            else:
                buy_hold_equity[i + 1] = buy_hold_equity[i] * (next_px / max(px, 1e-12))

        if n >= 1:
            final_equity = cash + shares * float(prices_arr[-1])
            equity_curve[-1] = final_equity
            if n > 1:
                effective_positions[-1] = (shares * float(prices_arr[-1])) / max(final_equity, 1e-9)

        turnover = float(np.mean(np.abs(np.diff(effective_positions, prepend=0.0)))) if n > 0 else 0.0
        total_costs = total_commission + total_slippage + total_margin_interest + total_borrow_fee

        strategy_cum = float(equity_curve[-1] / self.initial_capital - 1.0)
        buy_hold_cum = float(buy_hold_equity[-1] / self.initial_capital - 1.0)
        ann = max(1.0, annualization_factor)
        periods = max(1.0, float(n - 1))
        ann_return = float((1.0 + strategy_cum) ** (ann / periods) - 1.0) if strategy_cum > -1.0 else -1.0
        mdd = self._max_drawdown(equity_curve)
        calmar = float(ann_return / abs(mdd)) if abs(mdd) > 1e-9 else 0.0
        ret_series = np.asarray(daily_returns[1:], dtype=float)
        gains = ret_series[ret_series > 0.0]
        losses = ret_series[ret_series < 0.0]
        gross_profit = float(np.sum(gains)) if len(gains) else 0.0
        gross_loss = float(-np.sum(losses)) if len(losses) else 0.0
        profit_factor = float(gross_profit / gross_loss) if gross_loss > 1e-12 else 0.0

        metrics = {
            "cum_return": strategy_cum,
            "sharpe": self._safe_sharpe(daily_returns[1:], annualization_factor=annualization_factor),
            "sortino": self._safe_sortino(daily_returns[1:], annualization_factor=annualization_factor),
            "calmar": calmar,
            "max_drawdown": mdd,
            "win_rate": float(np.mean(daily_returns[1:] > 0.0)) if n > 1 else 0.0,
            "buy_hold_cum_return": buy_hold_cum,
            "alpha": strategy_cum - buy_hold_cum,
            "annualized_return": ann_return,
            "profit_factor": profit_factor,
            "turnover": turnover,
            "trade_count": float(trade_count),
            "total_transaction_costs": float(total_costs),
            "total_commission": float(total_commission),
            "total_slippage": float(total_slippage),
            "total_margin_interest": float(total_margin_interest),
            "total_borrow_fee": float(total_borrow_fee),
            "day_end_flatten_fraction": float(day_end_flatten_fraction if flat_at_day_end else 0.0),
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
            total_borrow_fee=float(total_borrow_fee),
        )
