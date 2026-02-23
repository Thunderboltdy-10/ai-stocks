"use client";

import { useMemo } from "react";
import { ModelMeta } from "@/types/ai";
import { cn } from "@/lib/utils";
import {
  Radar,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { X } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ModelComparisonViewProps {
  models: ModelMeta[];
  onClose: () => void;
}

const COLORS = ["#FBBF24", "#10B981", "#8B5CF6"];

export function ModelComparisonView({ models, onClose }: ModelComparisonViewProps) {
  // Normalize metrics for radar chart (0-100 scale)
  const radarData = useMemo(() => {
    const metrics = [
      {
        metric: "Sharpe",
        fullMark: 100,
        ...models.reduce((acc, model, idx) => ({
          ...acc,
          [model.symbol]: Math.min((model.metrics?.sharpeRatio ?? 0) * 40, 100), // 2.5 Sharpe = 100
        }), {}),
      },
      {
        metric: "Win Rate",
        fullMark: 100,
        ...models.reduce((acc, model, idx) => ({
          ...acc,
          [model.symbol]: (model.metrics?.winRate ?? 0) * 100,
        }), {}),
      },
      {
        metric: "Low DD",
        fullMark: 100,
        ...models.reduce((acc, model, idx) => ({
          ...acc,
          [model.symbol]: Math.max(100 - (model.metrics?.maxDrawdown ?? 0) * 200, 0), // 0% DD = 100, 50% DD = 0
        }), {}),
      },
      {
        metric: "Dir. Acc.",
        fullMark: 100,
        ...models.reduce((acc, model, idx) => ({
          ...acc,
          [model.symbol]: (model.metrics?.directionalAccuracy ?? 0) * 100,
        }), {}),
      },
      {
        metric: "Return",
        fullMark: 100,
        ...models.reduce((acc, model, idx) => ({
          ...acc,
          [model.symbol]: Math.min(Math.max((model.metrics?.cumulativeReturn ?? 0) * 50 + 50, 0), 100), // -100% to +100% mapped to 0-100
        }), {}),
      },
    ];
    return metrics;
  }, [models]);

  if (models.length === 0) {
    return null;
  }

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Model Comparison</h3>
        <Button variant="ghost" size="icon-sm" onClick={onClose} className="text-gray-400 hover:text-white">
          <X className="size-5" />
        </Button>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 mb-4">
        {models.map((model, idx) => (
          <div key={model.id} className="flex items-center gap-2">
            <div
              className="size-3 rounded-full"
              style={{ backgroundColor: COLORS[idx] }}
            />
            <span className="text-sm text-gray-400">{model.symbol}</span>
          </div>
        ))}
      </div>

      {/* Radar Chart */}
      <div className="h-64 mb-6">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={radarData}>
            <PolarGrid stroke="#374151" />
            <PolarAngleAxis dataKey="metric" tick={{ fill: "#9CA3AF", fontSize: 12 }} />
            <PolarRadiusAxis angle={90} domain={[0, 100]} tick={{ fill: "#6B7280", fontSize: 10 }} />
            {models.map((model, idx) => (
              <Radar
                key={model.id}
                name={model.symbol}
                dataKey={model.symbol}
                stroke={COLORS[idx]}
                fill={COLORS[idx]}
                fillOpacity={0.2}
                strokeWidth={2}
              />
            ))}
          </RadarChart>
        </ResponsiveContainer>
      </div>

      {/* Metrics Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-800">
              <th className="text-left py-2 px-3 text-gray-500 font-medium">Metric</th>
              {models.map((model, idx) => (
                <th key={model.id} className="text-left py-2 px-3 font-medium" style={{ color: COLORS[idx] }}>
                  {model.symbol}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            <tr className="border-b border-gray-800/50">
              <td className="py-2 px-3 text-gray-400">Sharpe Ratio</td>
              {models.map((model) => (
                <td key={model.id} className="py-2 px-3 text-white font-medium">
                  {(model.metrics?.sharpeRatio ?? 0).toFixed(2)}
                </td>
              ))}
            </tr>
            <tr className="border-b border-gray-800/50">
              <td className="py-2 px-3 text-gray-400">Win Rate</td>
              {models.map((model) => (
                <td key={model.id} className="py-2 px-3 text-white font-medium">
                  {((model.metrics?.winRate ?? 0) * 100).toFixed(0)}%
                </td>
              ))}
            </tr>
            <tr className="border-b border-gray-800/50">
              <td className="py-2 px-3 text-gray-400">Max Drawdown</td>
              {models.map((model) => (
                <td key={model.id} className="py-2 px-3 text-white font-medium">
                  {((model.metrics?.maxDrawdown ?? 0) * 100).toFixed(1)}%
                </td>
              ))}
            </tr>
            <tr className="border-b border-gray-800/50">
              <td className="py-2 px-3 text-gray-400">Dir. Accuracy</td>
              {models.map((model) => (
                <td key={model.id} className="py-2 px-3 text-white font-medium">
                  {((model.metrics?.directionalAccuracy ?? 0) * 100).toFixed(1)}%
                </td>
              ))}
            </tr>
            <tr className="border-b border-gray-800/50">
              <td className="py-2 px-3 text-gray-400">Cumulative Return</td>
              {models.map((model) => {
                const ret = model.metrics?.cumulativeReturn ?? 0;
                return (
                  <td key={model.id} className={cn("py-2 px-3 font-medium", ret >= 0 ? "text-emerald-400" : "text-rose-400")}>
                    {ret >= 0 ? "+" : ""}{(ret * 100).toFixed(1)}%
                  </td>
                );
              })}
            </tr>
            <tr className="border-b border-gray-800/50">
              <td className="py-2 px-3 text-gray-400">Total Trades</td>
              {models.map((model) => (
                <td key={model.id} className="py-2 px-3 text-white font-medium">
                  {model.metrics?.totalTrades ?? 0}
                </td>
              ))}
            </tr>
            <tr>
              <td className="py-2 px-3 text-gray-400">Sequence Length</td>
              {models.map((model) => (
                <td key={model.id} className="py-2 px-3 text-white font-medium">
                  {model.sequenceLength} days
                </td>
              ))}
            </tr>
          </tbody>
        </table>
      </div>
    </div>
  );
}
