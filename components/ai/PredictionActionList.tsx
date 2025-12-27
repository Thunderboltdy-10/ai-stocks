"use client";

import { useMemo } from "react";
import { PredictionResult, TradeMarker } from "@/types/ai";
import { formatCurrency, formatPercent } from "@/utils/formatters";

interface PredictionActionListProps {
  forecast?: PredictionResult["forecast"];
  tradeMarkers?: TradeMarker[];
  lastPrice?: number;
  tradeShareFloor?: number;
}

interface ActionRow {
  date: string;
  action: "BUY" | "SELL";
  price: number;
  delta: number;
  shares: number;
  confidence: number;
}

export function PredictionActionList({ forecast, tradeMarkers, lastPrice, tradeShareFloor }: PredictionActionListProps) {
  const rows = useMemo<ActionRow[]>(() => {
    if (!forecast || !tradeMarkers?.length) return [];
    const actionable = tradeMarkers.filter((marker) => marker.segment === "forecast" && marker.type !== "hold");
    if (!actionable.length) return [];

    const detailMap = new Map<string, { price: number; delta: number }>();
    forecast.dates.forEach((date, index) => {
      const price = forecast.prices[index] ?? lastPrice ?? 0;
      const prevPrice = index === 0 ? lastPrice ?? price : forecast.prices[index - 1] ?? price;
      const delta = prevPrice ? (price - prevPrice) / prevPrice : 0;
      const expectedReturn = forecast.returns[index] ?? delta;
      detailMap.set(date, { price, delta: expectedReturn });
    });

    return actionable
      .map((marker) => {
        const details = detailMap.get(marker.date);
        return {
          date: marker.date,
          action: marker.type.toUpperCase() as ActionRow["action"],
          price: details?.price ?? marker.price ?? lastPrice ?? 0,
          delta: details?.delta ?? 0,
          shares: marker.shares,
          confidence: marker.confidence,
        };
      })
      .sort((a, b) => new Date(a.date).getTime() - new Date(b.date).getTime());
  }, [forecast, tradeMarkers, lastPrice]);

  return (
    <div className="rounded-2xl border border-gray-900 bg-gray-900/60 p-4">
      <div className="flex items-center justify-between gap-4 text-xs text-gray-400">
        <div>
          <p className="text-sm font-semibold text-gray-100">Upcoming Signals</p>
          <p className="text-[11px] text-gray-500">Model instructions derived from the forecast horizon.</p>
        </div>
        {typeof tradeShareFloor === "number" && (
          <span className="text-[11px] text-gray-500">Min size · {tradeShareFloor.toFixed(0)}%</span>
        )}
      </div>
      {rows.length ? (
        <div className="mt-3 overflow-x-auto">
          <table className="min-w-full text-xs text-gray-300">
            <thead>
              <tr className="text-[11px] uppercase tracking-wide text-gray-500">
                <th className="py-1 pr-3 text-left font-medium">Date</th>
                <th className="py-1 pr-3 text-left font-medium">Action</th>
                <th className="py-1 pr-3 text-left font-medium">Price</th>
                <th className="py-1 pr-3 text-left font-medium">Δ vs prior</th>
                <th className="py-1 pr-3 text-left font-medium">Shares</th>
                <th className="py-1 text-left font-medium">Confidence</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((row) => (
                <tr key={`${row.date}-${row.action}`} className="border-t border-gray-800">
                  <td className="py-1.5 pr-3 text-gray-400">{new Date(row.date).toLocaleString()}</td>
                  <td className={`py-1.5 pr-3 font-semibold ${row.action === "BUY" ? "text-emerald-400" : "text-rose-400"}`}>
                    {row.action}
                  </td>
                  <td className="py-1.5 pr-3">{formatCurrency(row.price)}</td>
                  <td className={`py-1.5 pr-3 ${row.delta >= 0 ? "text-emerald-300" : "text-rose-300"}`}>
                    {formatPercent(row.delta)}
                  </td>
                  <td className="py-1.5 pr-3">{row.shares.toFixed(1)}</td>
                  <td className="py-1.5">{formatPercent(row.confidence)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="mt-3 text-xs text-gray-500">No discrete buy/sell instructions for this forecast window yet.</p>
      )}
    </div>
  );
}
