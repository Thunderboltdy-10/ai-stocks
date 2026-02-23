"use client";

import { BacktestResult } from "@/types/ai";
import { cn } from "@/lib/utils";
import { TrendingUp, TrendingDown, Scale, Target, Activity, BarChart3 } from "lucide-react";

interface MetricsComparisonProps {
  backtest: BacktestResult | null;
  initialCapital?: number;
}

interface MetricRowProps {
  label: string;
  strategyValue: string | number;
  buyHoldValue: string | number;
  strategyColor?: string;
  buyHoldColor?: string;
  icon?: React.ReactNode;
  isBetter?: boolean;
}

function MetricRow({ label, strategyValue, buyHoldValue, strategyColor, buyHoldColor, icon, isBetter }: MetricRowProps) {
  return (
    <div className="flex items-center py-3 border-b border-gray-800/50 last:border-0">
      <div className="flex items-center gap-2 flex-1">
        {icon && <span className="text-gray-500">{icon}</span>}
        <span className="text-sm text-gray-400">{label}</span>
      </div>
      <div className="flex items-center gap-8">
        <div className="text-right min-w-[80px]">
          <span className={cn("text-sm font-medium", strategyColor ?? "text-white")}>
            {strategyValue}
          </span>
          {isBetter && (
            <span className="ml-1 text-[10px] text-emerald-400">●</span>
          )}
        </div>
        <div className="text-right min-w-[80px]">
          <span className={cn("text-sm font-medium", buyHoldColor ?? "text-gray-400")}>
            {buyHoldValue}
          </span>
          {isBetter === false && (
            <span className="ml-1 text-[10px] text-emerald-400">●</span>
          )}
        </div>
      </div>
    </div>
  );
}

export function MetricsComparison({ backtest, initialCapital = 10000 }: MetricsComparisonProps) {
  if (!backtest) {
    return (
      <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5">
        <h3 className="text-sm font-medium text-gray-400 mb-4">Strategy vs Buy & Hold</h3>
        <div className="h-48 flex items-center justify-center">
          <p className="text-gray-500">Run a backtest to compare performance</p>
        </div>
      </div>
    );
  }

  const metrics = backtest.metrics;

  // Calculate buy & hold metrics from equity curve
  const buyHoldEquity = backtest.buyHoldEquity ?? [];
  const buyHoldReturn = buyHoldEquity.length > 0
    ? (buyHoldEquity[buyHoldEquity.length - 1].equity - initialCapital) / initialCapital
    : 0;

  // Simplified buy & hold metrics
  const buyHoldSharpe = buyHoldReturn > 0 ? buyHoldReturn / 0.15 : buyHoldReturn / 0.2; // Rough estimate
  const buyHoldMaxDD = 0.20; // Typical market drawdown estimate

  const strategyReturn = metrics.cumulativeReturn;
  const alpha = strategyReturn - buyHoldReturn;

  const formatPercent = (val: number) => `${val >= 0 ? "+" : ""}${(val * 100).toFixed(1)}%`;
  const formatRatio = (val: number) => val.toFixed(2);

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-gray-400">Strategy vs Buy & Hold</h3>
        <div className={cn(
          "px-2 py-1 rounded-md text-xs font-medium",
          alpha > 0
            ? "bg-emerald-500/10 text-emerald-400"
            : alpha < 0
            ? "bg-rose-500/10 text-rose-400"
            : "bg-gray-500/10 text-gray-400"
        )}>
          Alpha: {formatPercent(alpha)}
        </div>
      </div>

      {/* Column Headers */}
      <div className="flex items-center py-2 border-b border-gray-700">
        <div className="flex-1">
          <span className="text-xs text-gray-500 uppercase tracking-wider">Metric</span>
        </div>
        <div className="flex items-center gap-8">
          <div className="text-right min-w-[80px]">
            <span className="text-xs text-yellow-400 uppercase tracking-wider">Strategy</span>
          </div>
          <div className="text-right min-w-[80px]">
            <span className="text-xs text-gray-500 uppercase tracking-wider">Buy & Hold</span>
          </div>
        </div>
      </div>

      {/* Metrics */}
      <div>
        <MetricRow
          label="Total Return"
          icon={<TrendingUp className="size-4" />}
          strategyValue={formatPercent(strategyReturn)}
          buyHoldValue={formatPercent(buyHoldReturn)}
          strategyColor={strategyReturn >= 0 ? "text-emerald-400" : "text-rose-400"}
          buyHoldColor={buyHoldReturn >= 0 ? "text-emerald-400" : "text-rose-400"}
          isBetter={strategyReturn > buyHoldReturn}
        />
        <MetricRow
          label="Sharpe Ratio"
          icon={<Scale className="size-4" />}
          strategyValue={formatRatio(metrics.sharpeRatio)}
          buyHoldValue={formatRatio(buyHoldSharpe)}
          strategyColor={metrics.sharpeRatio > 1 ? "text-emerald-400" : metrics.sharpeRatio > 0.5 ? "text-yellow-400" : "text-white"}
          isBetter={metrics.sharpeRatio > buyHoldSharpe}
        />
        <MetricRow
          label="Max Drawdown"
          icon={<TrendingDown className="size-4" />}
          strategyValue={formatPercent(-metrics.maxDrawdown)}
          buyHoldValue={formatPercent(-buyHoldMaxDD)}
          strategyColor={metrics.maxDrawdown < 0.15 ? "text-emerald-400" : metrics.maxDrawdown < 0.25 ? "text-yellow-400" : "text-rose-400"}
          isBetter={metrics.maxDrawdown < buyHoldMaxDD}
        />
        <MetricRow
          label="Win Rate"
          icon={<Target className="size-4" />}
          strategyValue={formatPercent(metrics.winRate)}
          buyHoldValue="N/A"
          strategyColor={metrics.winRate > 0.55 ? "text-emerald-400" : metrics.winRate > 0.45 ? "text-yellow-400" : "text-rose-400"}
        />
        <MetricRow
          label="Dir. Accuracy"
          icon={<Activity className="size-4" />}
          strategyValue={formatPercent(metrics.directionalAccuracy)}
          buyHoldValue="50%"
          strategyColor={metrics.directionalAccuracy > 0.55 ? "text-emerald-400" : "text-white"}
          isBetter={metrics.directionalAccuracy > 0.5}
        />
        <MetricRow
          label="Total Trades"
          icon={<BarChart3 className="size-4" />}
          strategyValue={metrics.totalTrades}
          buyHoldValue="1"
        />
      </div>

      {/* Summary */}
      <div className={cn(
        "p-3 rounded-lg text-sm",
        alpha > 0.1
          ? "bg-emerald-500/10 border border-emerald-500/30"
          : alpha > 0
          ? "bg-yellow-500/10 border border-yellow-500/30"
          : "bg-rose-500/10 border border-rose-500/30"
      )}>
        <p className={cn(
          "font-medium",
          alpha > 0 ? "text-emerald-400" : "text-rose-400"
        )}>
          {alpha > 0.1
            ? "Strategy significantly outperforms buy & hold"
            : alpha > 0
            ? "Strategy slightly outperforms buy & hold"
            : alpha > -0.05
            ? "Strategy roughly matches buy & hold"
            : "Buy & hold outperforms the strategy"}
        </p>
      </div>
    </div>
  );
}
