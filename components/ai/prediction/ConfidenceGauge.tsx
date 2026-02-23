"use client";

import { cn } from "@/lib/utils";

interface ConfidenceGaugeProps {
  value: number; // 0-1
  label?: string;
  size?: "sm" | "md" | "lg";
  showValue?: boolean;
  signal?: "buy" | "sell" | "hold";
}

const SIZE_CONFIG = {
  sm: { width: 60, strokeWidth: 6, fontSize: "text-sm", labelSize: "text-[8px]" },
  md: { width: 80, strokeWidth: 8, fontSize: "text-lg", labelSize: "text-[10px]" },
  lg: { width: 120, strokeWidth: 10, fontSize: "text-2xl", labelSize: "text-xs" },
};

const SIGNAL_COLORS = {
  buy: { stroke: "#10B981", text: "text-emerald-400" },
  sell: { stroke: "#EF4444", text: "text-rose-400" },
  hold: { stroke: "#6B7280", text: "text-gray-400" },
};

export function ConfidenceGauge({
  value,
  label = "Confidence",
  size = "md",
  showValue = true,
  signal = "hold",
}: ConfidenceGaugeProps) {
  const config = SIZE_CONFIG[size];
  const colors = SIGNAL_COLORS[signal];
  const normalizedValue = Math.max(0, Math.min(1, value));
  const percentage = Math.round(normalizedValue * 100);

  const radius = (config.width - config.strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - normalizedValue * circumference;

  // Determine color based on value if no signal
  const getValueColor = () => {
    if (signal !== "hold") return colors;
    if (normalizedValue >= 0.7) return { stroke: "#10B981", text: "text-emerald-400" };
    if (normalizedValue >= 0.5) return { stroke: "#FBBF24", text: "text-yellow-400" };
    if (normalizedValue >= 0.3) return { stroke: "#F97316", text: "text-orange-400" };
    return { stroke: "#EF4444", text: "text-rose-400" };
  };

  const valueColors = getValueColor();

  return (
    <div className="flex flex-col items-center">
      <div className="relative" style={{ width: config.width, height: config.width }}>
        <svg
          className="w-full h-full -rotate-90"
          viewBox={`0 0 ${config.width} ${config.width}`}
        >
          {/* Background circle */}
          <circle
            cx={config.width / 2}
            cy={config.width / 2}
            r={radius}
            fill="none"
            stroke="currentColor"
            strokeWidth={config.strokeWidth}
            className="text-gray-800"
          />
          {/* Progress circle */}
          <circle
            cx={config.width / 2}
            cy={config.width / 2}
            r={radius}
            fill="none"
            stroke={valueColors.stroke}
            strokeWidth={config.strokeWidth}
            strokeLinecap="round"
            strokeDasharray={circumference}
            strokeDashoffset={strokeDashoffset}
            className="transition-all duration-500 ease-out"
          />
        </svg>
        {showValue && (
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className={cn("font-bold", config.fontSize, valueColors.text)}>
              {percentage}%
            </span>
          </div>
        )}
      </div>
      {label && (
        <span className={cn("mt-1 text-gray-500 uppercase tracking-wider", config.labelSize)}>
          {label}
        </span>
      )}
    </div>
  );
}
