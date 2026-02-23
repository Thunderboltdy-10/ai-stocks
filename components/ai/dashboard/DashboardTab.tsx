"use client";

import { useCallback, useEffect, useState } from "react";
import { useMutation } from "@tanstack/react-query";
import { toast } from "sonner";
import { QuickPredictionCard } from "./QuickPredictionCard";
import { SignalRecommendation } from "./SignalRecommendation";
import { QuickMetrics } from "./QuickMetrics";
import { RecentPredictions } from "./RecentPredictions";
import { runPrediction } from "@/lib/api-client";
import { FusionSettings, PredictionParams, PredictionResult } from "@/types/ai";

type RiskProfile = "conservative" | "balanced" | "aggressive";
type Signal = "BUY" | "SELL" | "HOLD";

interface RecentPrediction {
  id: string;
  symbol: string;
  signal: Signal;
  confidence: number;
  expectedReturn: number;
  timestamp: string;
}

const RISK_PROFILE_CONFIG: Record<RiskProfile, { fusion: Partial<FusionSettings>; confidenceFloors: { buy: number; sell: number } }> = {
  conservative: {
    fusion: { mode: "classifier", buyThreshold: 0.35, sellThreshold: 0.5, regressorScale: 8 },
    confidenceFloors: { buy: 0.35, sell: 0.5 },
  },
  balanced: {
    fusion: { mode: "weighted", buyThreshold: 0.3, sellThreshold: 0.45, regressorScale: 15 },
    confidenceFloors: { buy: 0.3, sell: 0.45 },
  },
  aggressive: {
    fusion: { mode: "regressor", buyThreshold: 0.2, sellThreshold: 0.35, regressorScale: 20 },
    confidenceFloors: { buy: 0.2, sell: 0.35 },
  },
};

const LOCAL_STORAGE_KEY = "ai-dashboard-recent-predictions";
const LOCAL_STORAGE_SYMBOLS_KEY = "ai-dashboard-recent-symbols";

interface DashboardTabProps {
  onNavigateToPrediction?: () => void;
}

export function DashboardTab({ onNavigateToPrediction }: DashboardTabProps) {
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [recentPredictions, setRecentPredictions] = useState<RecentPrediction[]>([]);
  const [recentSymbols, setRecentSymbols] = useState<string[]>([]);

  // Load from localStorage on mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(LOCAL_STORAGE_KEY);
      if (stored) {
        setRecentPredictions(JSON.parse(stored));
      }
      const storedSymbols = localStorage.getItem(LOCAL_STORAGE_SYMBOLS_KEY);
      if (storedSymbols) {
        setRecentSymbols(JSON.parse(storedSymbols));
      }
    } catch {
      // Ignore localStorage errors
    }
  }, []);

  // Save to localStorage when predictions change
  useEffect(() => {
    try {
      localStorage.setItem(LOCAL_STORAGE_KEY, JSON.stringify(recentPredictions.slice(0, 20)));
    } catch {
      // Ignore localStorage errors
    }
  }, [recentPredictions]);

  useEffect(() => {
    try {
      localStorage.setItem(LOCAL_STORAGE_SYMBOLS_KEY, JSON.stringify(recentSymbols.slice(0, 10)));
    } catch {
      // Ignore localStorage errors
    }
  }, [recentSymbols]);

  const predictMutation = useMutation({
    mutationFn: async ({ symbol, riskProfile }: { symbol: string; riskProfile: RiskProfile }) => {
      const config = RISK_PROFILE_CONFIG[riskProfile];
      const params: PredictionParams & { fusion: FusionSettings } = {
        symbol,
        horizon: 5,
        daysOnChart: 60,
        smoothing: "none",
        confidenceFloors: config.confidenceFloors,
        tradeShareFloor: 15,
        fusion: {
          mode: config.fusion.mode ?? "weighted",
          regressorScale: config.fusion.regressorScale ?? 15,
          buyThreshold: config.fusion.buyThreshold ?? 0.3,
          sellThreshold: config.fusion.sellThreshold ?? 0.45,
          regimeFilters: { bull: true, bear: true },
        },
      };
      return runPrediction(params);
    },
    onSuccess: (result, variables) => {
      setPrediction(result);
      toast.success(`Prediction ready for ${result.symbol}`);

      // Determine signal
      const lastPosition = result.fusedPositions.at(-1) ?? 0;
      const signal: Signal = lastPosition > 0.2 ? "BUY" : lastPosition < -0.2 ? "SELL" : "HOLD";

      // Calculate expected return from forecast
      const lastPrice = result.prices.at(-1) ?? 0;
      const forecastPrice = result.forecast?.prices.at(-1) ?? lastPrice;
      const expectedReturn = lastPrice > 0 ? (forecastPrice - lastPrice) / lastPrice : 0;

      // Get confidence from classifiers
      const lastClassifier = result.classifierProbabilities.at(-1);
      const confidence = lastClassifier
        ? Math.max(lastClassifier.buy, lastClassifier.sell, lastClassifier.hold)
        : 0.5;

      // Add to recent predictions
      const newPrediction: RecentPrediction = {
        id: `${result.symbol}-${Date.now()}`,
        symbol: result.symbol,
        signal,
        confidence,
        expectedReturn,
        timestamp: new Date().toISOString(),
      };
      setRecentPredictions((prev) => [newPrediction, ...prev.filter((p) => p.symbol !== result.symbol)]);

      // Update recent symbols
      setRecentSymbols((prev) => [variables.symbol, ...prev.filter((s) => s !== variables.symbol)]);
    },
    onError: (error: Error) => {
      toast.error(`Prediction failed: ${error.message}`);
    },
  });

  const handlePredict = useCallback(async (symbol: string, riskProfile: RiskProfile) => {
    await predictMutation.mutateAsync({ symbol, riskProfile });
  }, [predictMutation]);

  const handleSelectRecent = useCallback((recent: RecentPrediction) => {
    // Re-run prediction for the selected symbol
    handlePredict(recent.symbol, "balanced");
  }, [handlePredict]);

  // Derive display values from prediction
  const signal: Signal | null = prediction
    ? (prediction.fusedPositions.at(-1) ?? 0) > 0.2
      ? "BUY"
      : (prediction.fusedPositions.at(-1) ?? 0) < -0.2
      ? "SELL"
      : "HOLD"
    : null;

  const lastPrice = prediction?.prices.at(-1) ?? null;
  const forecastPrice = prediction?.forecast?.prices.at(-1) ?? null;
  const expectedReturn = lastPrice && forecastPrice
    ? (forecastPrice - lastPrice) / lastPrice
    : null;

  const lastClassifier = prediction?.classifierProbabilities.at(-1);
  const confidence = lastClassifier
    ? Math.max(lastClassifier.buy, lastClassifier.sell, lastClassifier.hold)
    : null;

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr,400px]">
      <div className="space-y-6">
        <QuickPredictionCard
          onPredict={handlePredict}
          isLoading={predictMutation.isPending}
          recentSymbols={recentSymbols}
        />

        <SignalRecommendation
          signal={signal}
          confidence={confidence ?? 0}
          symbol={prediction?.symbol ?? null}
          currentPrice={lastPrice}
          targetPrice={forecastPrice}
          expectedReturn={expectedReturn}
          horizon={prediction?.metadata.horizon ?? 5}
          onViewDetails={onNavigateToPrediction}
        />

        <QuickMetrics
          expectedReturn={expectedReturn}
          confidence={confidence}
          horizon={prediction?.metadata.horizon ?? null}
          fusionMode={prediction?.metadata.fusionMode ?? null}
          sharpeRatio={null}
          directionalAccuracy={null}
        />
      </div>

      <div>
        <RecentPredictions
          predictions={recentPredictions}
          onSelect={handleSelectRecent}
        />
      </div>
    </div>
  );
}
