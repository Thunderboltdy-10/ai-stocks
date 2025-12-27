"use client";

import { ModelMeta } from "@/types/ai";
import { formatPercent } from "@/utils/formatters";

interface ModelComparisonPanelProps {
  models?: ModelMeta[];
  selected: string[];
  onToggle: (modelId: string) => void;
}

export function ModelComparisonPanel({ models, selected, onToggle }: ModelComparisonPanelProps) {
  if (!models?.length) {
    return null;
  }

  const activeModels = models.filter((model) => selected.includes(model.id));

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-200">Model Comparison</h3>
        <p className="text-xs text-gray-500">Select up to 3 models</p>
      </div>
      <div className="mt-3 grid gap-2">
        {models.map((model) => (
          <label key={model.id} className="flex items-center justify-between rounded border border-gray-800 px-3 py-2 text-xs text-gray-300">
            <span>{model.symbol} · {model.fusionModeDefault}</span>
            <input
              type="checkbox"
              className="accent-yellow-500"
              checked={selected.includes(model.id)}
              onChange={() => onToggle(model.id)}
            />
          </label>
        ))}
      </div>
      {activeModels.length ? (
        <div className="mt-4 overflow-x-auto">
          <table className="w-full text-left text-xs text-gray-300">
            <thead>
              <tr className="text-gray-500">
                <th className="py-2">Metric</th>
                {activeModels.map((model) => (
                  <th key={model.id} className="py-2 text-right">{model.symbol}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {["sharpeRatio", "winRate", "maxDrawdown", "directionalAccuracy"].map((metric) => (
                <tr key={metric}>
                  <td className="py-1 capitalize text-gray-400">{metric.replace(/([A-Z])/g, " $1")}</td>
                  {activeModels.map((model) => (
                    <td key={`${model.id}-${metric}`} className="py-1 text-right">
                      {typeof model.metrics[metric as keyof typeof model.metrics] === "number"
                        ? metric === "maxDrawdown"
                          ? formatPercent(model.metrics[metric as keyof typeof model.metrics] as number)
                          : (model.metrics[metric as keyof typeof model.metrics] as number).toFixed(2)
                        : "–"}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : (
        <p className="mt-3 text-xs text-gray-500">Pick models to compare metrics side-by-side.</p>
      )}
    </div>
  );
}
