"use client";

import { useMemo } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { BacktestResult } from "@/types/ai";
import { TrendingDown, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";

interface DrawdownAnalysisProps {
  backtest: BacktestResult | null;
}

export function DrawdownAnalysis({ backtest }: DrawdownAnalysisProps) {
  const chartData = useMemo(() => {
    if (!backtest?.equityCurve) return [];
    return backtest.equityCurve.map((point) => ({
      date: point.date,
      drawdown: -Math.abs(point.drawdown) * 100, // Negative for visual
      drawdownValue: point.drawdown * 100,
    }));
  }, [backtest]);

  const maxDrawdown = backtest?.metrics?.maxDrawdown ?? 0;
  const maxDrawdownPct = (maxDrawdown * 100).toFixed(1);

  // Find drawdown periods
  const drawdownPeriods = useMemo(() => {
    if (!chartData.length) return [];

    const periods: { start: string; end: string; depth: number; duration: number }[] = [];
    let inDrawdown = false;
    let currentStart = "";
    let currentMax = 0;
    let startIdx = 0;

    chartData.forEach((point, idx) => {
      if (Math.abs(point.drawdownValue) > 1 && !inDrawdown) {
        inDrawdown = true;
        currentStart = point.date;
        startIdx = idx;
        currentMax = Math.abs(point.drawdownValue);
      } else if (inDrawdown) {
        currentMax = Math.max(currentMax, Math.abs(point.drawdownValue));
        if (Math.abs(point.drawdownValue) < 0.5) {
          periods.push({
            start: currentStart,
            end: point.date,
            depth: currentMax,
            duration: idx - startIdx,
          });
          inDrawdown = false;
          currentMax = 0;
        }
      }
    });

    return periods.sort((a, b) => b.depth - a.depth).slice(0, 3);
  }, [chartData]);

  if (!backtest) {
    return (
      <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5">
        <div className="flex items-center gap-2 mb-4">
          <TrendingDown className="size-5 text-gray-500" />
          <h3 className="text-sm font-medium text-gray-400">Drawdown Analysis</h3>
        </div>
        <div className="h-48 flex items-center justify-center">
          <p className="text-gray-500">Run a backtest to see drawdown analysis</p>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5 space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <TrendingDown className="size-5 text-rose-400" />
          <h3 className="text-sm font-medium text-gray-400">Drawdown Analysis</h3>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-gray-500">Max Drawdown:</span>
          <span className={cn(
            "text-lg font-bold",
            maxDrawdown < 0.15 ? "text-emerald-400" : maxDrawdown < 0.25 ? "text-yellow-400" : "text-rose-400"
          )}>
            {maxDrawdownPct}%
          </span>
        </div>
      </div>

      {/* Chart */}
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 5, right: 5, left: 5, bottom: 5 }}>
            <defs>
              <linearGradient id="drawdownGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#EF4444" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#EF4444" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" vertical={false} />
            <XAxis
              dataKey="date"
              stroke="#6B7280"
              fontSize={10}
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) => new Date(value).toLocaleDateString("en-US", { month: "short", day: "numeric" })}
            />
            <YAxis
              stroke="#6B7280"
              fontSize={10}
              tickLine={false}
              axisLine={false}
              tickFormatter={(value) => `${value}%`}
              domain={["dataMin - 5", 0]}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#1F2937",
                border: "1px solid #374151",
                borderRadius: "8px",
                fontSize: "12px",
              }}
              formatter={(value: number) => [`${Math.abs(value).toFixed(2)}%`, "Drawdown"]}
              labelFormatter={(label) => new Date(label).toLocaleDateString()}
            />
            <ReferenceLine y={0} stroke="#374151" strokeDasharray="3 3" />
            <Area
              type="monotone"
              dataKey="drawdown"
              stroke="#EF4444"
              fill="url(#drawdownGradient)"
              strokeWidth={2}
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Major Drawdown Periods */}
      {drawdownPeriods.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <AlertTriangle className="size-4 text-yellow-500" />
            <span className="text-xs text-gray-500 uppercase tracking-wider">Major Drawdown Periods</span>
          </div>
          <div className="space-y-2">
            {drawdownPeriods.map((period, idx) => (
              <div
                key={idx}
                className="flex items-center justify-between p-2 rounded-lg bg-gray-800/50 text-sm"
              >
                <div className="flex items-center gap-2">
                  <span className="text-gray-500">#{idx + 1}</span>
                  <span className="text-gray-300">
                    {new Date(period.start).toLocaleDateString()} → {new Date(period.end).toLocaleDateString()}
                  </span>
                </div>
                <div className="flex items-center gap-4">
                  <span className="text-rose-400 font-medium">-{period.depth.toFixed(1)}%</span>
                  <span className="text-gray-500">{period.duration} days</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
