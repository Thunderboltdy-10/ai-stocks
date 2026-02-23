"use client";

import { useState, useMemo, useCallback, useRef } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { toast } from "sonner";
import {
  FusionSettings,
  PredictionParams,
  PredictionResult,
  ModelMeta,
  FusionMode,
  TradeMarker,
} from "@/types/ai";
import { listModels, loadModel, runPrediction } from "@/lib/api-client";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Brain, Loader2, TrendingUp, TrendingDown, Minus } from "lucide-react";

// Prediction Components
import { HorizonSelector } from "./HorizonSelector";
import { FusionModeSelector } from "./FusionModeSelector";
import { ConfidenceGauge } from "./ConfidenceGauge";
import { SignalExplanation } from "./SignalExplanation";

// Shared Components
import { SymbolAutocomplete } from "../shared/SymbolAutocomplete";
import { RiskProfileSelector, RiskProfile } from "../shared/RiskProfileSelector";
import { LoadingOverlay } from "../shared/LoadingOverlay";
import { ExportMenu } from "../shared/ExportMenu";

// Chart
import InteractiveCandlestick, {
  CandlestickChartHandle,
  CandlestickPoint,
  ChartMarker,
} from "@/components/charts/InteractiveCandlestick";

const DEFAULT_PREDICTION: PredictionParams = {
  symbol: "AAPL",
  horizon: 5,
  daysOnChart: 120,
  smoothing: "none",
  confidenceFloors: { buy: 0.3, sell: 0.45 },
  tradeShareFloor: 15,
};

const DEFAULT_FUSION: FusionSettings = {
  mode: "weighted",
  regressorScale: 15,
  buyThreshold: 0.3,
  sellThreshold: 0.45,
  regimeFilters: { bull: true, bear: true },
};

const RISK_PRESETS: Record<RiskProfile, { fusion: Partial<FusionSettings>; confidenceFloors: { buy: number; sell: number } }> = {
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

interface PredictionTabProps {
  onPredictionReady?: (prediction: PredictionResult) => void;
}

export function PredictionTab({ onPredictionReady }: PredictionTabProps) {
  const candlestickRef = useRef<CandlestickChartHandle>(null);

  // State
  const [predictionParams, setPredictionParams] = useState<PredictionParams>(DEFAULT_PREDICTION);
  const [fusionSettings, setFusionSettings] = useState<FusionSettings>(DEFAULT_FUSION);
  const [riskProfile, setRiskProfile] = useState<RiskProfile>("balanced");
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [selectedMarker, setSelectedMarker] = useState<TradeMarker | null>(null);
  const [recentSymbols, setRecentSymbols] = useState<string[]>([]);

  // Queries
  const modelsQuery = useQuery({
    queryKey: ["models"],
    queryFn: listModels,
    refetchInterval: 60_000,
  });

  // Find model for current symbol
  const selectedModel = useMemo(() => {
    return modelsQuery.data?.find(m => m.symbol === predictionParams.symbol);
  }, [modelsQuery.data, predictionParams.symbol]);

  // Mutation
  const predictMutation = useMutation({
    mutationFn: runPrediction,
    onSuccess: (result) => {
      setPrediction(result);
      onPredictionReady?.(result);

      // Add to recent symbols
      setRecentSymbols(prev => {
        const filtered = prev.filter(s => s !== result.symbol);
        return [result.symbol, ...filtered].slice(0, 5);
      });

      toast.success(`Prediction ready for ${result.symbol}`);
    },
    onError: (error: Error) => {
      toast.error(error.message);
    },
  });

  // Handlers
  const handleSymbolChange = useCallback((symbol: string) => {
    setPredictionParams(prev => ({ ...prev, symbol }));
  }, []);

  const handleHorizonChange = useCallback((horizon: number) => {
    setPredictionParams(prev => ({ ...prev, horizon }));
  }, []);

  const handleFusionModeChange = useCallback((mode: FusionMode) => {
    setFusionSettings(prev => ({ ...prev, mode }));
  }, []);

  const handleRiskProfileChange = useCallback((profile: RiskProfile) => {
    setRiskProfile(profile);
    const preset = RISK_PRESETS[profile];
    setFusionSettings(prev => ({ ...prev, ...preset.fusion }));
    setPredictionParams(prev => ({ ...prev, confidenceFloors: preset.confidenceFloors }));
  }, []);

  const handlePredict = useCallback(async () => {
    if (!predictionParams.symbol) {
      toast.error("Enter a symbol");
      return;
    }

    await predictMutation.mutateAsync({
      ...predictionParams,
      modelId: selectedModel?.id,
      fusion: fusionSettings,
    });
  }, [predictionParams, selectedModel, fusionSettings, predictMutation]);

  // Chart data
  const candlestickData: CandlestickPoint[] = useMemo(() => {
    if (!prediction) return [];
    const baseSeries = prediction.candles
      ? [...prediction.candles]
      : prediction.dates.map((date, index) => {
          const close = prediction.prices[index] ?? prediction.predictedPrices[index];
          const prev = prediction.prices[index - 1] ?? close;
          return { date, open: prev, high: Math.max(close, prev), low: Math.min(close, prev), close };
        });
    const limit = Math.max(1, predictionParams.daysOnChart ?? baseSeries.length);
    return baseSeries.slice(-limit);
  }, [prediction, predictionParams.daysOnChart]);

  const predictedSeries = useMemo(() => {
    if (!prediction) return [];
    const history = prediction.dates.map((date, index) => ({
      date,
      price: prediction.predictedPrices[index] ?? prediction.prices[index],
      segment: "history" as const,
    }));
    const limit = Math.max(1, predictionParams.daysOnChart ?? history.length);
    const trimmedHistory = history.slice(-limit);

    const forecast = prediction.forecast;
    const future = forecast
      ? forecast.dates.map((date, index) => ({
          date,
          price: forecast.prices[index] ?? forecast.prices[index - 1] ?? trimmedHistory.at(-1)?.price ?? 0,
          segment: "forecast" as const,
        }))
      : [];
    return [...trimmedHistory, ...future];
  }, [prediction, predictionParams.daysOnChart]);

  const chartMarkers: ChartMarker[] = useMemo(() => {
    if (!prediction?.tradeMarkers) return [];
    return prediction.tradeMarkers.map(marker => ({
      ...marker,
      segment: marker.segment ?? "history",
      scope: marker.scope ?? "prediction",
    }));
  }, [prediction?.tradeMarkers]);

  // Signal determination
  const currentSignal = useMemo(() => {
    if (!prediction?.tradeMarkers?.length) return { type: "hold" as const, confidence: 0 };
    const latestMarker = prediction.tradeMarkers[prediction.tradeMarkers.length - 1];
    return {
      type: latestMarker.type as "buy" | "sell" | "hold",
      confidence: latestMarker.confidence,
    };
  }, [prediction?.tradeMarkers]);

  const handleExportJSON = useCallback(() => {
    if (!prediction) return;
    const blob = new Blob([JSON.stringify(prediction, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `prediction-${prediction.symbol}-${new Date().toISOString().split("T")[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [prediction]);

  const handleExportPNG = useCallback(() => {
    candlestickRef.current?.exportImage();
  }, []);

  return (
    <div className="grid gap-6 xl:grid-cols-[360px,1fr,300px]">
      {/* Left Panel - Configuration */}
      <aside className="space-y-4">
        {/* Symbol Input */}
        <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-4 space-y-4">
          <h3 className="text-sm font-medium text-gray-400">Symbol</h3>
          <SymbolAutocomplete
            value={predictionParams.symbol}
            onChange={handleSymbolChange}
            recentSymbols={recentSymbols}
            disabled={predictMutation.isPending}
          />
          {selectedModel && (
            <div className="flex items-center gap-2 p-2 rounded-lg bg-emerald-500/10 border border-emerald-500/30">
              <Brain className="size-4 text-emerald-400" />
              <span className="text-xs text-emerald-400">Model available</span>
            </div>
          )}
        </div>

        {/* Risk Profile */}
        <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-4 space-y-4">
          <h3 className="text-sm font-medium text-gray-400">Risk Profile</h3>
          <RiskProfileSelector
            selected={riskProfile}
            onChange={handleRiskProfileChange}
            disabled={predictMutation.isPending}
            variant="buttons"
          />
        </div>

        {/* Horizon Selector */}
        <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-4">
          <HorizonSelector
            selected={predictionParams.horizon}
            onChange={handleHorizonChange}
            disabled={predictMutation.isPending}
          />
        </div>

        {/* Fusion Mode */}
        <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-4">
          <FusionModeSelector
            selected={fusionSettings.mode}
            onChange={handleFusionModeChange}
            disabled={predictMutation.isPending}
          />
        </div>

        {/* Run Button */}
        <Button
          onClick={handlePredict}
          disabled={predictMutation.isPending || !predictionParams.symbol}
          className="w-full h-12 text-lg font-semibold bg-yellow-500 hover:bg-yellow-600 text-gray-900"
        >
          {predictMutation.isPending ? (
            <>
              <Loader2 className="size-5 mr-2 animate-spin" />
              Running...
            </>
          ) : (
            <>
              <Brain className="size-5 mr-2" />
              Run Prediction
            </>
          )}
        </Button>
      </aside>

      {/* Center - Chart */}
      <main className="space-y-4">
        {/* Header */}
        <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <div>
                <p className="text-xs text-gray-500">Symbol</p>
                <p className="text-2xl font-bold text-white">
                  {prediction?.symbol ?? predictionParams.symbol}
                </p>
              </div>
              {prediction && (
                <>
                  <div>
                    <p className="text-xs text-gray-500">Last Price</p>
                    <p className="text-xl font-semibold text-white">
                      ${prediction.prices.at(-1)?.toFixed(2)}
                    </p>
                  </div>
                  <div>
                    <p className="text-xs text-gray-500">Horizon</p>
                    <p className="text-xl font-semibold text-white">
                      {predictionParams.horizon}D
                    </p>
                  </div>
                </>
              )}
            </div>
            <ExportMenu
              onExportJSON={prediction ? handleExportJSON : undefined}
              onExportPNG={prediction ? handleExportPNG : undefined}
              disabled={!prediction}
            />
          </div>
        </div>

        {/* Chart */}
        <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-4 relative min-h-[400px]">
          <LoadingOverlay
            isLoading={predictMutation.isPending}
            message="Running prediction..."
            submessage="Analyzing market data"
          />
          {prediction ? (
            <InteractiveCandlestick
              ref={candlestickRef}
              historicalSeries={candlestickData}
              tradeMarkers={chartMarkers}
              predictedSeries={predictedSeries}
              overlays={prediction.overlays}
              forecastChanges={prediction.forecast?.dates.map((date, i) => ({
                date,
                value: prediction.forecast?.returns[i] ?? 0,
              }))}
              renderMode="candlestick"
              mode="prediction"
              showBuyMarkers={true}
              showSellMarkers={true}
              segmentFilters={{ history: true, forecast: true }}
              scopeFilters={{ prediction: true, backtest: false }}
              ariaLabel="Prediction chart"
              onMarkerClick={(marker) => setSelectedMarker(marker as TradeMarker)}
            />
          ) : (
            <div className="h-[400px] flex items-center justify-center">
              <div className="text-center space-y-3">
                <Brain className="size-16 text-gray-700 mx-auto" />
                <p className="text-gray-500">Enter a symbol and run prediction to see results</p>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Right Panel - Results */}
      <aside className="space-y-4">
        {/* Signal Card */}
        <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5">
          <h3 className="text-sm font-medium text-gray-400 mb-4">Current Signal</h3>
          {prediction ? (
            <div className="flex flex-col items-center gap-4">
              <div className={cn(
                "size-20 rounded-full flex items-center justify-center",
                currentSignal.type === "buy" && "bg-emerald-500/20",
                currentSignal.type === "sell" && "bg-rose-500/20",
                currentSignal.type === "hold" && "bg-gray-500/20"
              )}>
                {currentSignal.type === "buy" && <TrendingUp className="size-10 text-emerald-400" />}
                {currentSignal.type === "sell" && <TrendingDown className="size-10 text-rose-400" />}
                {currentSignal.type === "hold" && <Minus className="size-10 text-gray-400" />}
              </div>
              <p className={cn(
                "text-2xl font-bold uppercase",
                currentSignal.type === "buy" && "text-emerald-400",
                currentSignal.type === "sell" && "text-rose-400",
                currentSignal.type === "hold" && "text-gray-400"
              )}>
                {currentSignal.type}
              </p>
              <ConfidenceGauge
                value={currentSignal.confidence}
                signal={currentSignal.type}
                size="lg"
              />
            </div>
          ) : (
            <div className="h-40 flex items-center justify-center">
              <p className="text-gray-500 text-sm">Run prediction to see signal</p>
            </div>
          )}
        </div>

        {/* Signal Explanation */}
        <SignalExplanation
          marker={selectedMarker}
          fusionMode={fusionSettings.mode}
          regressorReturn={prediction?.forecast?.returns.at(-1)}
        />

        {/* Forecast Summary */}
        {prediction?.forecast && (
          <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-4 space-y-3">
            <h3 className="text-sm font-medium text-gray-400">Forecast Summary</h3>
            <div className="grid grid-cols-2 gap-3">
              <div className="rounded-lg bg-gray-800/50 p-3">
                <p className="text-xs text-gray-500">Expected Return</p>
                <p className={cn(
                  "text-lg font-bold",
                  (prediction.forecast.returns.at(-1) ?? 0) >= 0 ? "text-emerald-400" : "text-rose-400"
                )}>
                  {((prediction.forecast.returns.at(-1) ?? 0) * 100).toFixed(2)}%
                </p>
              </div>
              <div className="rounded-lg bg-gray-800/50 p-3">
                <p className="text-xs text-gray-500">Target Price</p>
                <p className="text-lg font-bold text-white">
                  ${prediction.forecast.prices.at(-1)?.toFixed(2)}
                </p>
              </div>
            </div>
          </div>
        )}
      </aside>
    </div>
  );
}
