"use client";

import { useMemo, useState } from "react";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { PredictionResult, ScenarioBuilderState } from "@/types/ai";
import { synthesizePricePath } from "@/utils/metrics";
import { formatCurrency, formatPercent } from "@/utils/formatters";

interface ScenarioBuilderProps {
  prediction?: PredictionResult | null;
}

export function ScenarioBuilder({ prediction }: ScenarioBuilderProps) {
  const [state, setState] = useState<ScenarioBuilderState>({ buyFloor: 0.35, sellFloor: 0.45 });

  const scenarioPoints = useMemo(() => {
    if (!prediction) return [];
    const positions = prediction.classifierProbabilities.map((prob, index) => {
      if (prob.buy >= state.buyFloor) return Math.min(1, prediction.fusedPositions[index] ?? 1);
      if (prob.sell >= state.sellFloor) return Math.max(-0.5, prediction.fusedPositions[index] ?? -0.5);
      return prediction.fusedPositions[index] ?? 0;
    });
    return synthesizePricePath(prediction.predictedPrices[0] ?? prediction.prices.at(-1) ?? 0, prediction.predictedReturns, positions);
  }, [prediction, state]);

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-200">Scenario Builder</h3>
        <p className="text-xs text-gray-500">Adjust confidence floors</p>
      </div>
      <div className="mt-3 grid grid-cols-2 gap-3 text-xs text-gray-400">
        <label className="flex flex-col gap-1">
          <span>Buy floor ({formatPercent(state.buyFloor)})</span>
          <input
            type="range"
            min={0.1}
            max={0.8}
            step={0.01}
            value={state.buyFloor}
            onChange={(event) => setState((prev) => ({ ...prev, buyFloor: Number(event.target.value) }))}
            aria-label="Buy confidence floor"
          />
        </label>
        <label className="flex flex-col gap-1">
          <span>Sell floor ({formatPercent(state.sellFloor)})</span>
          <input
            type="range"
            min={0.1}
            max={0.9}
            step={0.01}
            value={state.sellFloor}
            onChange={(event) => setState((prev) => ({ ...prev, sellFloor: Number(event.target.value) }))}
            aria-label="Sell confidence floor"
          />
        </label>
      </div>
      {scenarioPoints.length ? (
        <div className="mt-4">
          <ResponsiveContainer width="100%" height={180}>
            <LineChart data={scenarioPoints} margin={{ top: 10, right: 10, left: -20 }}>
              <XAxis dataKey="date" hide />
              <YAxis domain={["auto", "auto"]} hide />
              <Tooltip formatter={(value) => formatCurrency(Number(value))} labelFormatter={(value) => `Step ${value}`} />
              <Line type="monotone" dataKey="simulatedPrice" stroke="#34d399" dot={false} />
            </LineChart>
          </ResponsiveContainer>
          <div className="mt-3 grid grid-cols-3 gap-2 text-xs text-gray-400">
            <div>
              <p className="text-[10px] uppercase">Projected Δ</p>
              <p className="text-lg text-gray-100">
                {formatPercent(
                  scenarioPoints.length ? (scenarioPoints.at(-1)!.simulatedPrice - scenarioPoints[0].simulatedPrice) / scenarioPoints[0].simulatedPrice : 0
                )}
              </p>
            </div>
            <div>
              <p className="text-[10px] uppercase">Max Position</p>
              <p className="text-lg text-gray-100">{Math.max(...scenarioPoints.map((point) => point.position)).toFixed(2)}</p>
            </div>
            <div>
              <p className="text-[10px] uppercase">Min Position</p>
              <p className="text-lg text-gray-100">{Math.min(...scenarioPoints.map((point) => point.position)).toFixed(2)}</p>
            </div>
          </div>
        </div>
      ) : (
        <p className="mt-4 text-xs text-gray-500">Run a prediction to start simulating scenarios.</p>
      )}
    </div>
  );
}
