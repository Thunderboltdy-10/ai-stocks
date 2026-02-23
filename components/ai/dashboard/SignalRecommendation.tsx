"use client";

import { cn } from "@/lib/utils";
import { TrendingUp, TrendingDown, Minus, ArrowRight } from "lucide-react";
import { Button } from "@/components/ui/button";

type Signal = "BUY" | "SELL" | "HOLD";

interface SignalRecommendationProps {
  signal: Signal | null;
  confidence: number;
  symbol: string | null;
  currentPrice: number | null;
  targetPrice: number | null;
  expectedReturn: number | null;
  horizon: number;
  onViewDetails?: () => void;
}

const SIGNAL_CONFIG: Record<Signal, { icon: React.ReactNode; color: string; bgColor: string; borderColor: string; label: string }> = {
  BUY: {
    icon: <TrendingUp className="size-8" />,
    color: "text-emerald-400",
    bgColor: "bg-emerald-500/10",
    borderColor: "border-emerald-500/50",
    label: "BUY",
  },
  SELL: {
    icon: <TrendingDown className="size-8" />,
    color: "text-rose-400",
    bgColor: "bg-rose-500/10",
    borderColor: "border-rose-500/50",
    label: "SELL",
  },
  HOLD: {
    icon: <Minus className="size-8" />,
    color: "text-gray-400",
    bgColor: "bg-gray-500/10",
    borderColor: "border-gray-500/50",
    label: "HOLD",
  },
};

function ConfidenceGauge({ value, signal }: { value: number; signal: Signal }) {
  const config = SIGNAL_CONFIG[signal];
  const percentage = Math.round(value * 100);
  const circumference = 2 * Math.PI * 45;
  const strokeDashoffset = circumference - (value * circumference);

  return (
    <div className="relative w-28 h-28">
      <svg className="w-full h-full -rotate-90" viewBox="0 0 100 100">
        {/* Background circle */}
        <circle
          cx="50"
          cy="50"
          r="45"
          fill="none"
          stroke="currentColor"
          strokeWidth="8"
          className="text-gray-800"
        />
        {/* Progress circle */}
        <circle
          cx="50"
          cy="50"
          r="45"
          fill="none"
          stroke="currentColor"
          strokeWidth="8"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          className={cn(
            "transition-all duration-500",
            signal === "BUY" ? "text-emerald-500" :
            signal === "SELL" ? "text-rose-500" : "text-gray-500"
          )}
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className={cn("text-2xl font-bold", config.color)}>{percentage}%</span>
        <span className="text-[10px] text-gray-500 uppercase tracking-wider">Confidence</span>
      </div>
    </div>
  );
}

export function SignalRecommendation({
  signal,
  confidence,
  symbol,
  currentPrice,
  targetPrice,
  expectedReturn,
  horizon,
  onViewDetails,
}: SignalRecommendationProps) {
  if (!signal || !symbol) {
    return (
      <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-6">
        <div className="flex flex-col items-center justify-center py-8 text-center">
          <div className="size-16 rounded-full bg-gray-800/50 flex items-center justify-center mb-4">
            <TrendingUp className="size-8 text-gray-600" />
          </div>
          <p className="text-gray-400 text-sm">Run a prediction to see the recommendation</p>
        </div>
      </div>
    );
  }

  const config = SIGNAL_CONFIG[signal];
  const formatPrice = (price: number) => `$${price.toFixed(2)}`;
  const formatReturn = (ret: number) => `${ret >= 0 ? "+" : ""}${(ret * 100).toFixed(2)}%`;

  return (
    <div className={cn(
      "rounded-xl border p-6",
      config.borderColor,
      config.bgColor
    )}>
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <div className="flex items-center gap-3 mb-4">
            <div className={cn("size-12 rounded-xl flex items-center justify-center", config.bgColor, config.color)}>
              {config.icon}
            </div>
            <div>
              <h3 className={cn("text-3xl font-bold", config.color)}>{config.label}</h3>
              <p className="text-sm text-gray-400">{symbol} · {horizon}-day horizon</p>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4 mb-4">
            <div>
              <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">Current Price</p>
              <p className="text-xl font-semibold text-white">{currentPrice ? formatPrice(currentPrice) : "—"}</p>
            </div>
            <div>
              <p className="text-xs text-gray-500 uppercase tracking-wider mb-1">Target Price</p>
              <p className={cn("text-xl font-semibold", config.color)}>
                {targetPrice ? formatPrice(targetPrice) : "—"}
              </p>
            </div>
          </div>

          {expectedReturn !== null && (
            <div className="flex items-center gap-2 mb-4">
              <span className="text-sm text-gray-400">Expected Return:</span>
              <span className={cn("text-lg font-bold", expectedReturn >= 0 ? "text-emerald-400" : "text-rose-400")}>
                {formatReturn(expectedReturn)}
              </span>
            </div>
          )}

          {onViewDetails && (
            <Button
              variant="outline"
              onClick={onViewDetails}
              className="border-gray-700 text-gray-300 hover:text-white hover:border-gray-500"
            >
              View Details
              <ArrowRight className="size-4 ml-2" />
            </Button>
          )}
        </div>

        <ConfidenceGauge value={confidence} signal={signal} />
      </div>
    </div>
  );
}
