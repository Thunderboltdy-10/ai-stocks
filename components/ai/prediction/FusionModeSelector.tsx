"use client";

import { cn } from "@/lib/utils";
import { FusionMode } from "@/types/ai";
import { Layers, Scale, Cpu, Brain, Binary, Boxes } from "lucide-react";

interface FusionModeOption {
  id: FusionMode;
  label: string;
  description: string;
  icon: React.ReactNode;
  weights: string;
}

const FUSION_MODES: FusionModeOption[] = [
  {
    id: "weighted",
    label: "Weighted",
    description: "Dynamic weights based on recent performance",
    icon: <Scale className="size-5" />,
    weights: "Adaptive",
  },
  {
    id: "classifier",
    label: "Classifier",
    description: "Binary signals gate regressor predictions",
    icon: <Binary className="size-5" />,
    weights: "High conviction",
  },
  {
    id: "hybrid",
    label: "Hybrid",
    description: "Balanced contribution from all models",
    icon: <Layers className="size-5" />,
    weights: "25% each",
  },
  {
    id: "regressor",
    label: "Regressor",
    description: "Pure LSTM+Transformer predictions",
    icon: <Brain className="size-5" />,
    weights: "100% DL",
  },
];

interface FusionModeSelectorProps {
  selected: FusionMode;
  onChange: (mode: FusionMode) => void;
  disabled?: boolean;
}

export function FusionModeSelector({ selected, onChange, disabled }: FusionModeSelectorProps) {
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <Boxes className="size-4 text-gray-500" />
        <span className="text-sm text-gray-400">Fusion Mode</span>
      </div>
      <div className="grid grid-cols-2 gap-2">
        {FUSION_MODES.map((mode) => (
          <button
            key={mode.id}
            type="button"
            disabled={disabled}
            onClick={() => onChange(mode.id)}
            className={cn(
              "flex flex-col items-start gap-2 p-3 rounded-lg border text-left transition-all",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-yellow-500/50",
              selected === mode.id
                ? "border-yellow-500/50 bg-yellow-500/10"
                : "border-gray-700 hover:border-gray-600",
              disabled && "opacity-50 cursor-not-allowed"
            )}
          >
            <div className="flex items-center gap-2">
              <span className={cn(
                selected === mode.id ? "text-yellow-400" : "text-gray-500"
              )}>
                {mode.icon}
              </span>
              <span className={cn(
                "font-medium",
                selected === mode.id ? "text-yellow-200" : "text-gray-300"
              )}>
                {mode.label}
              </span>
            </div>
            <p className="text-xs text-gray-500 line-clamp-2">{mode.description}</p>
            <span className={cn(
              "text-[10px] px-1.5 py-0.5 rounded",
              selected === mode.id
                ? "bg-yellow-500/20 text-yellow-300"
                : "bg-gray-800 text-gray-500"
            )}>
              {mode.weights}
            </span>
          </button>
        ))}
      </div>
    </div>
  );
}
