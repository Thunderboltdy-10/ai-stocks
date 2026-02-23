"use client";

import { BacktestParams } from "@/types/ai";
import { cn } from "@/lib/utils";
import { Settings2, DollarSign, TrendingUp, TrendingDown } from "lucide-react";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Switch } from "@/components/ui/switch";

interface BacktestConfigPanelProps {
  params: BacktestParams;
  onChange: (params: Partial<BacktestParams>) => void;
  disabled?: boolean;
}

export function BacktestConfigPanel({ params, onChange, disabled }: BacktestConfigPanelProps) {
  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5 space-y-5">
      <div className="flex items-center gap-2">
        <Settings2 className="size-5 text-gray-500" />
        <h3 className="text-sm font-medium text-gray-400">Backtest Configuration</h3>
      </div>

      {/* Capital & Window */}
      <div className="grid grid-cols-2 gap-4">
        <div className="space-y-2">
          <Label className="text-xs text-gray-500">Initial Capital</Label>
          <div className="relative">
            <DollarSign className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-gray-500" />
            <Input
              type="number"
              value={params.initialCapital}
              onChange={(e) => onChange({ initialCapital: Number(e.target.value) })}
              disabled={disabled}
              className="pl-9 bg-gray-950 border-gray-700 text-white"
            />
          </div>
        </div>
        <div className="space-y-2">
          <Label className="text-xs text-gray-500">Backtest Window (days)</Label>
          <Input
            type="number"
            value={params.backtestWindow}
            onChange={(e) => onChange({ backtestWindow: Number(e.target.value) })}
            disabled={disabled}
            min={10}
            max={365}
            className="bg-gray-950 border-gray-700 text-white"
          />
        </div>
      </div>

      {/* Position Limits */}
      <div className="space-y-3">
        <p className="text-xs text-gray-500 uppercase tracking-wider">Position Limits</p>
        <div className="grid grid-cols-2 gap-4">
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <TrendingUp className="size-4 text-emerald-400" />
              <Label className="text-xs text-gray-400">Max Long</Label>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="range"
                min={0}
                max={2}
                step={0.1}
                value={params.maxLong}
                onChange={(e) => onChange({ maxLong: Number(e.target.value) })}
                disabled={disabled}
                className="flex-1 accent-emerald-500"
              />
              <span className="text-sm font-mono text-emerald-400 w-12 text-right">
                {(params.maxLong * 100).toFixed(0)}%
              </span>
            </div>
          </div>
          <div className="space-y-2">
            <div className="flex items-center gap-2">
              <TrendingDown className="size-4 text-rose-400" />
              <Label className="text-xs text-gray-400">Max Short</Label>
            </div>
            <div className="flex items-center gap-2">
              <input
                type="range"
                min={0}
                max={1}
                step={0.1}
                value={params.maxShort}
                onChange={(e) => onChange({ maxShort: Number(e.target.value) })}
                disabled={disabled}
                className="flex-1 accent-rose-500"
              />
              <span className="text-sm font-mono text-rose-400 w-12 text-right">
                {(params.maxShort * 100).toFixed(0)}%
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Forward Simulation Toggle */}
      <div className="flex items-center justify-between p-3 rounded-lg bg-gray-800/50">
        <div>
          <p className="text-sm text-gray-300">Forward Simulation</p>
          <p className="text-xs text-gray-500">Include forecast period in backtest</p>
        </div>
        <Switch
          checked={params.enableForwardSim}
          onCheckedChange={(checked) => onChange({ enableForwardSim: checked })}
          disabled={disabled}
        />
      </div>
    </div>
  );
}
