"use client";

import { cn } from "@/lib/utils";
import { Clock } from "lucide-react";

const HORIZONS = [
  { days: 1, label: "1D", description: "Next trading day" },
  { days: 3, label: "3D", description: "3 trading days" },
  { days: 5, label: "5D", description: "1 week" },
  { days: 7, label: "7D", description: "7 trading days" },
  { days: 10, label: "10D", description: "2 weeks" },
  { days: 14, label: "14D", description: "~3 weeks" },
  { days: 21, label: "21D", description: "1 month" },
];

interface HorizonSelectorProps {
  selected: number;
  onChange: (horizon: number) => void;
  disabled?: boolean;
}

export function HorizonSelector({ selected, onChange, disabled }: HorizonSelectorProps) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-2">
        <Clock className="size-4 text-gray-500" />
        <span className="text-sm text-gray-400">Forecast Horizon</span>
      </div>
      <div className="flex flex-wrap gap-2">
        {HORIZONS.map((h) => (
          <button
            key={h.days}
            type="button"
            disabled={disabled}
            onClick={() => onChange(h.days)}
            title={h.description}
            className={cn(
              "relative px-3 py-2 rounded-lg border text-sm font-medium transition-all",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-yellow-500/50",
              selected === h.days
                ? "border-yellow-500/50 bg-yellow-500/10 text-yellow-200"
                : "border-gray-700 text-gray-400 hover:border-gray-600 hover:text-gray-300",
              disabled && "opacity-50 cursor-not-allowed"
            )}
          >
            {h.label}
            {selected === h.days && (
              <span className="absolute -bottom-1 left-1/2 -translate-x-1/2 w-1 h-1 rounded-full bg-yellow-400" />
            )}
          </button>
        ))}
      </div>
      <p className="text-xs text-gray-500">
        {HORIZONS.find((h) => h.days === selected)?.description ?? "Custom horizon"}
      </p>
    </div>
  );
}
