"use client";

import { cn } from "@/lib/utils";
import { FusionMode, TradeMarker } from "@/types/ai";
import { Info, TrendingUp, TrendingDown, Minus, Brain, Binary, Scale } from "lucide-react";

interface SignalExplanationProps {
  marker: TradeMarker | null;
  fusionMode: FusionMode;
  regressorReturn?: number;
  classifierBuyProb?: number;
  classifierSellProb?: number;
}

export function SignalExplanation({
  marker,
  fusionMode,
  regressorReturn,
  classifierBuyProb,
  classifierSellProb,
}: SignalExplanationProps) {
  if (!marker) {
    return (
      <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-4">
        <div className="flex items-center gap-2 text-gray-500">
          <Info className="size-4" />
          <span className="text-sm">Select a trade marker to see explanation</span>
        </div>
      </div>
    );
  }

  const signalConfig = {
    buy: {
      icon: <TrendingUp className="size-5" />,
      color: "text-emerald-400",
      bgColor: "bg-emerald-500/10",
      borderColor: "border-emerald-500/30",
      label: "BUY Signal",
    },
    sell: {
      icon: <TrendingDown className="size-5" />,
      color: "text-rose-400",
      bgColor: "bg-rose-500/10",
      borderColor: "border-rose-500/30",
      label: "SELL Signal",
    },
    hold: {
      icon: <Minus className="size-5" />,
      color: "text-gray-400",
      bgColor: "bg-gray-500/10",
      borderColor: "border-gray-500/30",
      label: "HOLD Signal",
    },
  };

  const config = signalConfig[marker.type];

  const fusionExplanation = {
    weighted: "Dynamic weighting based on recent model performance",
    classifier: "Binary classifier gates regressor predictions",
    hybrid: "Equal contribution from LSTM, GBM, and classifiers",
    regressor: "Pure deep learning prediction (LSTM+Transformer)",
  };

  return (
    <div className={cn(
      "rounded-lg border p-4 space-y-4",
      config.borderColor,
      config.bgColor
    )}>
      {/* Signal Header */}
      <div className="flex items-center gap-3">
        <div className={cn("size-10 rounded-lg flex items-center justify-center", config.bgColor, config.color)}>
          {config.icon}
        </div>
        <div>
          <h4 className={cn("font-semibold", config.color)}>{config.label}</h4>
          <p className="text-sm text-gray-400">
            {new Date(marker.date).toLocaleDateString()} @ ${marker.price.toFixed(2)}
          </p>
        </div>
        <div className="ml-auto text-right">
          <p className="text-xs text-gray-500">Confidence</p>
          <p className={cn("text-lg font-bold", config.color)}>
            {(marker.confidence * 100).toFixed(0)}%
          </p>
        </div>
      </div>

      {/* Fusion Mode */}
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <Scale className="size-4 text-gray-500" />
          <span className="text-xs text-gray-500 uppercase tracking-wider">Fusion Mode</span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-sm font-medium text-white capitalize">{fusionMode}</span>
          <span className="text-xs text-gray-500">- {fusionExplanation[fusionMode]}</span>
        </div>
      </div>

      {/* Model Contributions */}
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <Brain className="size-4 text-gray-500" />
          <span className="text-xs text-gray-500 uppercase tracking-wider">Model Contributions</span>
        </div>
        <div className="grid grid-cols-2 gap-3">
          {regressorReturn !== undefined && (
            <div className="rounded-lg bg-gray-800/50 p-3">
              <p className="text-xs text-gray-500 mb-1">Regressor Return</p>
              <p className={cn(
                "text-lg font-semibold",
                regressorReturn >= 0 ? "text-emerald-400" : "text-rose-400"
              )}>
                {regressorReturn >= 0 ? "+" : ""}{(regressorReturn * 100).toFixed(2)}%
              </p>
            </div>
          )}
          {classifierBuyProb !== undefined && (
            <div className="rounded-lg bg-gray-800/50 p-3">
              <p className="text-xs text-gray-500 mb-1">Buy Probability</p>
              <p className="text-lg font-semibold text-emerald-400">
                {(classifierBuyProb * 100).toFixed(0)}%
              </p>
            </div>
          )}
          {classifierSellProb !== undefined && (
            <div className="rounded-lg bg-gray-800/50 p-3">
              <p className="text-xs text-gray-500 mb-1">Sell Probability</p>
              <p className="text-lg font-semibold text-rose-400">
                {(classifierSellProb * 100).toFixed(0)}%
              </p>
            </div>
          )}
        </div>
      </div>

      {/* Explanation Text */}
      {marker.explanation && (
        <div className="space-y-2">
          <div className="flex items-center gap-2">
            <Info className="size-4 text-gray-500" />
            <span className="text-xs text-gray-500 uppercase tracking-wider">Explanation</span>
          </div>
          <p className="text-sm text-gray-300">{marker.explanation}</p>
        </div>
      )}

      {/* Position Info */}
      {marker.shares > 0 && (
        <div className="pt-2 border-t border-gray-800">
          <p className="text-xs text-gray-500">
            Suggested position: <span className="text-white font-medium">{marker.shares} shares</span>
            {" "}({marker.segment === "forecast" ? "forecast" : "historical"})
          </p>
        </div>
      )}
    </div>
  );
}
