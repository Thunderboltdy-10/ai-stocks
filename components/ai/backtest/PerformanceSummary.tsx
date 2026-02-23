"use client";

import { BacktestResult } from "@/types/ai";
import { cn } from "@/lib/utils";
import {
  TrendingUp,
  TrendingDown,
  Target,
  Activity,
  BarChart3,
  Scale,
  Percent,
  DollarSign,
} from "lucide-react";

interface PerformanceSummaryProps {
  backtest: BacktestResult | null;
  initialCapital?: number;
}

interface MetricCardProps {
  label: string;
  value: string | number;
  icon: React.ReactNode;
  trend?: "up" | "down" | "neutral";
  description?: string;
}

function MetricCard({ label, value, icon, trend, description }: MetricCardProps) {
  return (
    <div className="rounded-lg bg-gray-800/50 p-3 space-y-1">
      <div className="flex items-center gap-2">
        <span className="text-gray-500">{icon}</span>
        <span className="text-xs text-gray-500">{label}</span>
      </div>
      <p className={cn(
        "text-xl font-bold",
        trend === "up" && "text-emerald-400",
        trend === "down" && "text-rose-400",
        trend === "neutral" && "text-white"
      )}>
        {value}
      </p>
      {description && (
        <p className="text-[10px] text-gray-500">{description}</p>
      )}
    </div>
  );
}

export function PerformanceSummary({ backtest, initialCapital = 10000 }: PerformanceSummaryProps) {
  if (!backtest) {
    return (
      <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5">
        <div className="flex items-center gap-2 mb-4">
          <BarChart3 className="size-5 text-gray-500" />
          <h3 className="text-sm font-medium text-gray-400">Performance Summary</h3>
        </div>
        <div className="h-32 flex items-center justify-center">
          <p className="text-gray-500">Run a backtest to see performance metrics</p>
        </div>
      </div>
    );
  }

  const metrics = backtest.metrics;
  const finalEquity = backtest.equityCurve?.[backtest.equityCurve.length - 1]?.equity ?? initialCapital;
  const totalPnL = finalEquity - initialCapital;
  const returnPct = (metrics.cumulativeReturn * 100).toFixed(1);

  const formatPercent = (val: number) => `${val >= 0 ? "+" : ""}${(val * 100).toFixed(1)}%`;
  const formatRatio = (val: number) => val.toFixed(2);
  const formatDollar = (val: number) => `$${val.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <BarChart3 className="size-5 text-gray-500" />
          <h3 className="text-sm font-medium text-gray-400">Performance Summary</h3>
        </div>
        <div className={cn(
          "px-3 py-1.5 rounded-lg font-bold",
          totalPnL >= 0 ? "bg-emerald-500/10 text-emerald-400" : "bg-rose-500/10 text-rose-400"
        )}>
          {totalPnL >= 0 ? "+" : ""}{returnPct}%
        </div>
      </div>

      {/* Hero Stats */}
      <div className="grid grid-cols-2 gap-3">
        <div className={cn(
          "rounded-lg p-4 text-center",
          totalPnL >= 0 ? "bg-emerald-500/10 border border-emerald-500/30" : "bg-rose-500/10 border border-rose-500/30"
        )}>
          <p className="text-xs text-gray-400 mb-1">Total P&L</p>
          <p className={cn(
            "text-2xl font-bold",
            totalPnL >= 0 ? "text-emerald-400" : "text-rose-400"
          )}>
            {formatDollar(totalPnL)}
          </p>
        </div>
        <div className="rounded-lg p-4 text-center bg-gray-800/50 border border-gray-700">
          <p className="text-xs text-gray-400 mb-1">Final Equity</p>
          <p className="text-2xl font-bold text-white">
            {formatDollar(finalEquity)}
          </p>
        </div>
      </div>

      {/* Metric Grid */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <MetricCard
          label="Sharpe Ratio"
          value={formatRatio(metrics.sharpeRatio)}
          icon={<Scale className="size-4" />}
          trend={metrics.sharpeRatio > 1 ? "up" : metrics.sharpeRatio > 0 ? "neutral" : "down"}
          description="Risk-adjusted return"
        />
        <MetricCard
          label="Win Rate"
          value={formatPercent(metrics.winRate)}
          icon={<Target className="size-4" />}
          trend={metrics.winRate > 0.55 ? "up" : metrics.winRate > 0.45 ? "neutral" : "down"}
          description="Winning trades"
        />
        <MetricCard
          label="Max Drawdown"
          value={formatPercent(-metrics.maxDrawdown)}
          icon={<TrendingDown className="size-4" />}
          trend={metrics.maxDrawdown < 0.15 ? "up" : metrics.maxDrawdown < 0.25 ? "neutral" : "down"}
          description="Largest peak-to-trough"
        />
        <MetricCard
          label="Dir. Accuracy"
          value={formatPercent(metrics.directionalAccuracy)}
          icon={<Activity className="size-4" />}
          trend={metrics.directionalAccuracy > 0.55 ? "up" : "neutral"}
          description="Correct direction"
        />
      </div>

      {/* Additional Stats */}
      <div className="grid grid-cols-3 gap-3 pt-2 border-t border-gray-800">
        <div className="text-center">
          <p className="text-xs text-gray-500">Total Trades</p>
          <p className="text-lg font-semibold text-white">{metrics.totalTrades}</p>
        </div>
        <div className="text-center">
          <p className="text-xs text-gray-500">Avg Trade</p>
          <p className={cn(
            "text-lg font-semibold",
            metrics.cumulativeReturn / Math.max(1, metrics.totalTrades) >= 0 ? "text-emerald-400" : "text-rose-400"
          )}>
            {formatPercent(metrics.cumulativeReturn / Math.max(1, metrics.totalTrades))}
          </p>
        </div>
        <div className="text-center">
          <p className="text-xs text-gray-500">Sortino Ratio</p>
          <p className="text-lg font-semibold text-white">
            {metrics.sortinoRatio?.toFixed(2) ?? "N/A"}
          </p>
        </div>
      </div>
    </div>
  );
}
