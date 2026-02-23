"use client";

import { cn } from "@/lib/utils";
import { Clock, TrendingUp, TrendingDown, Minus, ChevronRight } from "lucide-react";
import { formatDistanceToNow } from "date-fns";

type Signal = "BUY" | "SELL" | "HOLD";

interface RecentPrediction {
  id: string;
  symbol: string;
  signal: Signal;
  confidence: number;
  expectedReturn: number;
  timestamp: string;
}

interface RecentPredictionsProps {
  predictions: RecentPrediction[];
  onSelect: (prediction: RecentPrediction) => void;
  maxItems?: number;
}

const SIGNAL_STYLES: Record<Signal, { icon: React.ReactNode; color: string; bg: string }> = {
  BUY: {
    icon: <TrendingUp className="size-4" />,
    color: "text-emerald-400",
    bg: "bg-emerald-500/10",
  },
  SELL: {
    icon: <TrendingDown className="size-4" />,
    color: "text-rose-400",
    bg: "bg-rose-500/10",
  },
  HOLD: {
    icon: <Minus className="size-4" />,
    color: "text-gray-400",
    bg: "bg-gray-500/10",
  },
};

export function RecentPredictions({ predictions, onSelect, maxItems = 5 }: RecentPredictionsProps) {
  const displayPredictions = predictions.slice(0, maxItems);

  if (displayPredictions.length === 0) {
    return (
      <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5">
        <div className="flex items-center gap-2 mb-4">
          <Clock className="size-4 text-gray-500" />
          <h3 className="text-sm font-medium text-gray-400">Recent Predictions</h3>
        </div>
        <div className="flex flex-col items-center justify-center py-8 text-center">
          <Clock className="size-8 text-gray-700 mb-2" />
          <p className="text-sm text-gray-500">No recent predictions</p>
          <p className="text-xs text-gray-600 mt-1">Your prediction history will appear here</p>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5">
      <div className="flex items-center gap-2 mb-4">
        <Clock className="size-4 text-gray-500" />
        <h3 className="text-sm font-medium text-gray-400">Recent Predictions</h3>
        <span className="text-xs text-gray-600 ml-auto">{predictions.length} total</span>
      </div>

      <div className="space-y-2">
        {displayPredictions.map((prediction) => {
          const style = SIGNAL_STYLES[prediction.signal];
          const returnColor = prediction.expectedReturn >= 0 ? "text-emerald-400" : "text-rose-400";

          return (
            <button
              key={prediction.id}
              onClick={() => onSelect(prediction)}
              className="w-full flex items-center gap-3 p-3 rounded-lg border border-gray-800 bg-gray-900/40 hover:bg-gray-800/50 hover:border-gray-700 transition-all group"
            >
              <div className={cn("size-10 rounded-lg flex items-center justify-center", style.bg, style.color)}>
                {style.icon}
              </div>

              <div className="flex-1 text-left">
                <div className="flex items-center gap-2">
                  <span className="font-semibold text-white">{prediction.symbol}</span>
                  <span className={cn("text-xs font-medium px-1.5 py-0.5 rounded", style.bg, style.color)}>
                    {prediction.signal}
                  </span>
                </div>
                <div className="flex items-center gap-2 mt-0.5">
                  <span className={cn("text-sm font-medium", returnColor)}>
                    {prediction.expectedReturn >= 0 ? "+" : ""}
                    {(prediction.expectedReturn * 100).toFixed(2)}%
                  </span>
                  <span className="text-xs text-gray-500">
                    {formatDistanceToNow(new Date(prediction.timestamp), { addSuffix: true })}
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-2">
                <div className="text-right">
                  <p className="text-xs text-gray-500">Confidence</p>
                  <p className="text-sm font-medium text-gray-300">
                    {Math.round(prediction.confidence * 100)}%
                  </p>
                </div>
                <ChevronRight className="size-4 text-gray-600 group-hover:text-gray-400 transition" />
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}
