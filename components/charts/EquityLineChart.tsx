"use client";

import {
  Area,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  ReferenceDot,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { TradeMarker, BacktestResult } from "@/types/ai";
import { formatCurrency, formatPercent } from "@/utils/formatters";

interface EquityLineChartProps {
  backtest?: BacktestResult;
  tradeMarkers?: TradeMarker[];
  showStrategyEquity?: boolean;
  showBuyHoldEquity?: boolean;
}

interface TooltipEntry {
  value?: number;
  name?: string;
  dataKey?: string | number;
}

export function EquityLineChart({ backtest, tradeMarkers, showStrategyEquity = true, showBuyHoldEquity = true }: EquityLineChartProps) {
  const buyHoldMap = new Map(backtest?.buyHoldEquity?.map((point) => [point.date, point.equity]) ?? []);
  const data = backtest?.equityCurve.map((point, index) => ({
    date: point.date,
    strategyEquity: point.equity,
    buyHoldEquity: buyHoldMap.get(point.date) ?? backtest.buyHoldEquity?.[index]?.equity ?? null,
    drawdown: point.drawdown,
    price: backtest.priceSeries[index]?.price ?? null,
  }));

  const CustomTooltip = ({ active, payload, label }: { active?: boolean; payload?: TooltipEntry[]; label?: string | number }) => {
    if (!active || !payload?.length) return null;
    const entries = payload.filter((item) => item.value !== null && item.value !== undefined);
    const labelDate = label !== undefined ? new Date(label) : null;
    return (
      <div className="rounded-lg border border-gray-800 bg-gray-950/90 px-3 py-2 text-xs text-gray-200">
        <p className="text-[11px] text-gray-500">{labelDate ? labelDate.toLocaleString() : "—"}</p>
        {entries.map((entry) => {
          if (entry.dataKey === "drawdown") {
            return (
              <p key={entry.dataKey} className="text-rose-300">
                Drawdown · {formatPercent(Number(entry.value ?? 0))}
              </p>
            );
          }
          const labelText = entry.name ?? entry.dataKey;
          return (
            <p key={entry.dataKey} className="text-gray-200">
              {labelText} · {formatCurrency(Number(entry.value ?? 0))}
            </p>
          );
        })}
      </div>
    );
  };

  return (
    <div className="h-80 w-full rounded-xl border border-gray-800 bg-gray-900/60 p-4" aria-label="Equity curve chart">
      {data?.length ? (
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={data} margin={{ top: 10, right: 30, left: 10, bottom: 0 }}>
            <CartesianGrid stroke="rgba(255,255,255,0.05)" />
            <XAxis dataKey="date" tickFormatter={(value) => new Date(value).toLocaleDateString()} stroke="#6b7280" />
            <YAxis yAxisId="left" tickFormatter={(value) => formatCurrency(value).replace("$", "") } stroke="#6b7280" />
            <YAxis yAxisId="right" orientation="right" tickFormatter={(value) => formatPercent(value)} stroke="#6b7280" />
            <Tooltip content={<CustomTooltip />} />
            <Legend wrapperStyle={{ color: "#d1d5db" }} />
            <Area yAxisId="right" type="monotone" dataKey="drawdown" stroke="#f87171" fill="rgba(248,113,113,0.15)" name="Drawdown" />
            {showStrategyEquity && (
              <Line yAxisId="left" type="monotone" dataKey="strategyEquity" stroke="#facc15" strokeWidth={2} dot={false} name="Strategy" />
            )}
            {showBuyHoldEquity && (
              <Line yAxisId="left" type="monotone" dataKey="buyHoldEquity" stroke="#60a5fa" strokeWidth={2} dot={false} name="Buy & Hold" strokeDasharray="4 3" />
            )}
            <Line yAxisId="left" type="monotone" dataKey="price" stroke="#94a3b8" strokeWidth={1} dot={false} name="Spot Price" strokeDasharray="2 2" />
            {tradeMarkers?.map((marker) => {
              const targetPoint = data?.find((point) => point.date === marker.date);
              if (!targetPoint) return null;
              return (
                <ReferenceDot
                  key={`${marker.date}-${marker.type}-${marker.scope ?? "backtest"}`}
                  x={marker.date}
                  yAxisId="left"
                  y={targetPoint.strategyEquity}
                  fill={marker.type === "buy" ? "#22c55e" : marker.type === "sell" ? "#ef4444" : "#94a3b8"}
                  label={{
                    position: "top",
                    value: marker.type.toUpperCase(),
                    fill: marker.type === "buy" ? "#22c55e" : marker.type === "sell" ? "#ef4444" : "#94a3b8",
                  }}
                  r={6}
                />
              );
            })}
          </ComposedChart>
        </ResponsiveContainer>
      ) : (
        <p className="text-center text-sm text-gray-500">Run a backtest to unlock the equity chart.</p>
      )}
    </div>
  );
}
