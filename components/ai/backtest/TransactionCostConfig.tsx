"use client";

import { BacktestParams } from "@/types/ai";
import { cn } from "@/lib/utils";
import { Receipt, Percent, Clock } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";

interface TransactionCostConfigProps {
  params: BacktestParams;
  onChange: (params: Partial<BacktestParams>) => void;
  disabled?: boolean;
}

const PRESETS = [
  { label: "Free", commission: 0, slippage: 0.05 },
  { label: "Discount", commission: 0.5, slippage: 0.1 },
  { label: "Standard", commission: 1, slippage: 0.2 },
  { label: "Premium", commission: 2.5, slippage: 0.3 },
];

export function TransactionCostConfig({ params, onChange, disabled }: TransactionCostConfigProps) {
  const applyPreset = (preset: typeof PRESETS[number]) => {
    onChange({
      commission: preset.commission,
      slippage: preset.slippage,
    });
  };

  const isPresetActive = (preset: typeof PRESETS[number]) =>
    params.commission === preset.commission && params.slippage === preset.slippage;

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5 space-y-4">
      <div className="flex items-center gap-2">
        <Receipt className="size-5 text-gray-500" />
        <h3 className="text-sm font-medium text-gray-400">Transaction Costs</h3>
      </div>

      {/* Presets */}
      <div className="space-y-2">
        <p className="text-xs text-gray-500">Quick Presets</p>
        <div className="grid grid-cols-4 gap-2">
          {PRESETS.map((preset) => (
            <button
              key={preset.label}
              type="button"
              onClick={() => applyPreset(preset)}
              disabled={disabled}
              className={cn(
                "px-3 py-2 rounded-lg border text-xs font-medium transition-all",
                "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-yellow-500/50",
                isPresetActive(preset)
                  ? "border-yellow-500/50 bg-yellow-500/10 text-yellow-200"
                  : "border-gray-700 text-gray-400 hover:border-gray-600 hover:text-gray-300",
                disabled && "opacity-50 cursor-not-allowed"
              )}
            >
              {preset.label}
            </button>
          ))}
        </div>
      </div>

      {/* Custom Values */}
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Receipt className="size-4 text-gray-500" />
            <Label className="text-xs text-gray-400">Commission ($)</Label>
          </div>
          <Input
            type="number"
            value={params.commission}
            onChange={(e) => onChange({ commission: Number(e.target.value) })}
            disabled={disabled}
            min={0}
            max={10}
            step={0.1}
            className="bg-gray-950 border-gray-700 text-white"
          />
          <p className="text-[10px] text-gray-500">Per trade flat fee</p>
        </div>
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Percent className="size-4 text-gray-500" />
            <Label className="text-xs text-gray-400">Slippage (%)</Label>
          </div>
          <Input
            type="number"
            value={params.slippage}
            onChange={(e) => onChange({ slippage: Number(e.target.value) })}
            disabled={disabled}
            min={0}
            max={2}
            step={0.05}
            className="bg-gray-950 border-gray-700 text-white"
          />
          <p className="text-[10px] text-gray-500">Market impact estimate</p>
        </div>
      </div>

      {/* Cost Summary */}
      <div className="p-3 rounded-lg bg-gray-800/50 space-y-2">
        <p className="text-xs text-gray-500">Estimated Round-Trip Cost</p>
        <div className="flex items-baseline gap-2">
          <span className="text-2xl font-bold text-white">
            ${((params.commission * 2) + (params.initialCapital * params.slippage / 100 * 2)).toFixed(2)}
          </span>
          <span className="text-sm text-gray-400">
            on ${params.initialCapital.toLocaleString()} position
          </span>
        </div>
        <p className="text-xs text-gray-500">
          ({((params.commission * 2 + params.initialCapital * params.slippage / 100 * 2) / params.initialCapital * 100).toFixed(2)}% of capital)
        </p>
      </div>
    </div>
  );
}
