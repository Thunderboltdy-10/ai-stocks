"use client";

import { Area, Bar, ComposedChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { BacktestResult } from "@/types/ai";
import { formatCurrency, formatPercent } from "@/utils/formatters";

interface BacktestSummaryChartProps {
  backtest?: BacktestResult;
}

export function BacktestSummaryChart({ backtest }: BacktestSummaryChartProps) {
  const data = backtest?.equityCurve.map((point, index) => ({
    date: point.date,
    equity: point.equity,
    drawdown: point.drawdown,
    price: backtest.priceSeries[index]?.price ?? 0,
  }));

  return (
    <div className="h-72 w-full rounded-xl border border-gray-800 bg-gray-900/60 p-4" aria-label="Backtest summary chart">
      {data?.length ? (
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 10, right: 40, bottom: 0, left: 0 }}>
            <XAxis dataKey="date" tickFormatter={(value) => new Date(value).toLocaleDateString()} stroke="#6b7280" />
            <YAxis yAxisId="left" tickFormatter={(value) => formatCurrency(value).replace("$", "")} stroke="#6b7280" />
            <YAxis yAxisId="right" orientation="right" tickFormatter={(value) => formatPercent(value)} stroke="#6b7280" />
            <Tooltip
              contentStyle={{ background: "#111827", border: "1px solid #1f2937" }}
              formatter={(value, name) => {
                if (name === "equity") return [formatCurrency(Number(value)), "Equity"];
                if (name === "drawdown") return [formatPercent(Number(value)), "Drawdown"];
                return [formatCurrency(Number(value)), "Price"];
              }}
            />
            <Area yAxisId="left" dataKey="price" stroke="#38bdf8" fill="rgba(56,189,248,0.15)" name="Price" />
            <Bar yAxisId="right" dataKey="drawdown" fill="rgba(248,113,113,0.4)" name="Drawdown" />
            <Area yAxisId="left" dataKey="equity" stroke="#facc15" fill="rgba(250,204,21,0.15)" name="Equity" />
            <ReferenceLine yAxisId="right" y={0} stroke="#1f2937" strokeDasharray="3 3" />
          </ComposedChart>
        </ResponsiveContainer>
      ) : (
        <p className="text-center text-sm text-gray-500">Backtest results will appear here.</p>
      )}
    </div>
  );
}
