"use client";

import { useMemo } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";
import { EpochUpdate } from "./TrainingProgressPanel";

interface TrainingMetricsChartProps {
  epochs: EpochUpdate[];
}

export function TrainingMetricsChart({ epochs }: TrainingMetricsChartProps) {
  const chartData = useMemo(() => {
    return epochs.map((epoch) => ({
      epoch: epoch.epoch,
      trainLoss: epoch.trainLoss,
      valLoss: epoch.valLoss,
      directionalAccuracy: epoch.directionalAccuracy ? epoch.directionalAccuracy * 100 : null,
    }));
  }, [epochs]);

  if (epochs.length === 0) {
    return (
      <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5">
        <h3 className="text-sm font-medium text-gray-400 mb-4">Training Curves</h3>
        <div className="h-64 flex items-center justify-center">
          <p className="text-gray-500">No data yet</p>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5">
      <h3 className="text-sm font-medium text-gray-400 mb-4">Training Curves</h3>
      <div className="h-64">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
            <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
            <XAxis
              dataKey="epoch"
              stroke="#9CA3AF"
              fontSize={12}
              tickLine={false}
              axisLine={{ stroke: "#374151" }}
            />
            <YAxis
              yAxisId="left"
              stroke="#9CA3AF"
              fontSize={12}
              tickLine={false}
              axisLine={{ stroke: "#374151" }}
              tickFormatter={(value) => value.toFixed(3)}
            />
            <YAxis
              yAxisId="right"
              orientation="right"
              stroke="#9CA3AF"
              fontSize={12}
              tickLine={false}
              axisLine={{ stroke: "#374151" }}
              domain={[0, 100]}
              tickFormatter={(value) => `${value}%`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "#1F2937",
                border: "1px solid #374151",
                borderRadius: "8px",
                fontSize: "12px",
              }}
              labelStyle={{ color: "#9CA3AF" }}
            />
            <Legend
              wrapperStyle={{ fontSize: "12px" }}
              iconType="line"
            />
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="trainLoss"
              stroke="#FBBF24"
              strokeWidth={2}
              dot={false}
              name="Train Loss"
            />
            <Line
              yAxisId="left"
              type="monotone"
              dataKey="valLoss"
              stroke="#10B981"
              strokeWidth={2}
              dot={false}
              name="Val Loss"
            />
            {epochs.some((e) => e.directionalAccuracy) && (
              <Line
                yAxisId="right"
                type="monotone"
                dataKey="directionalAccuracy"
                stroke="#8B5CF6"
                strokeWidth={2}
                dot={false}
                name="Dir. Accuracy"
                strokeDasharray="5 5"
              />
            )}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}
