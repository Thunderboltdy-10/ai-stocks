"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { toast } from "sonner";
import { BacktestParams, BacktestResult, FusionSettings, JobEvent, ModelMeta, PredictionParams, PredictionResult } from "@/types/ai";
import { downloadModelArtifacts, listModels, loadModel, runBacktest, runPrediction } from "@/lib/api-client";
import { ModelControlPanel } from "@/components/ai/ModelControlPanel";
import InteractiveCandlestick, { CandlestickChartHandle, CandlestickPoint, ChartMarker } from "@/components/charts/InteractiveCandlestick";
import { EquityLineChart } from "@/components/charts/EquityLineChart";
import { BacktestSummaryChart } from "@/components/charts/BacktestSummaryChart";
import { BacktestResults } from "@/components/ai/BacktestResults";
import { PredictionActionList } from "../../../components/ai/PredictionActionList";
import { StreamingLog } from "@/components/ai/StreamingLog";
import { DiagnosticsPanel } from "@/components/ai/DiagnosticsPanel";
import { ScenarioBuilder } from "@/components/ai/ScenarioBuilder";
import { ModelComparisonPanel } from "@/components/ai/ModelComparisonPanel";
import { useKeyboardShortcuts } from "@/hooks/useKeyboardShortcuts";
import { useShortcuts } from "@/hooks/useShortcuts";
import { Button } from "@/components/ui/button";
import { formatCurrency, downloadJson } from "@/utils/formatters";
import { cn } from "@/lib/utils";
import { applySignalThreshold } from "@/lib/signals";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";

const SIGNAL_CONFIDENCE_FLOOR = 0.25;

const GLOSSARY_ITEMS = [
  { term: "Candle", description: "Each bar shows where price opened, closed, and the intraday battle between buyers and sellers." },
  { term: "Forecast", description: "A forward-looking price path so you can see the AI's best guess for the next few sessions." },
  { term: "Confidence", description: "A 0-1 score showing how sure the model is about a buy/sell call. Low confidence reverts to HOLD." },
  { term: "Backtest", description: "A replay of the strategy on historical data to estimate edge, drawdown, and risk before deploying." },
  { term: "Ensemble", description: "Multiple models voting together to smooth out noise and avoid over-relying on a single brain." },
];

const DEFAULT_PREDICTION: PredictionParams = {
  symbol: "AAPL",
  horizon: 10,
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

const DEFAULT_BACKTEST: BacktestParams = {
  backtestWindow: 60,
  initialCapital: 10_000,
  maxLong: 1,
  maxShort: 0.5,
  commission: 0.5,
  slippage: 0.2,
  enableForwardSim: false,
  shortCap: 0.5,
};

const QUICK_PRESETS = {
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
} as const;

export default function AiWorkbenchPage() {
  const candlestickRef = useRef<CandlestickChartHandle>(null);
  const [selectedModelId, setSelectedModelId] = useState<string | undefined>();
  const [comparisonSelection, setComparisonSelection] = useState<string[]>([]);
  const [predictionParams, setPredictionParams] = useState<PredictionParams>(DEFAULT_PREDICTION);
  const [fusionSettings, setFusionSettings] = useState<FusionSettings>(DEFAULT_FUSION);
  const [backtestParams, setBacktestParams] = useState<BacktestParams>(DEFAULT_BACKTEST);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [backtest, setBacktest] = useState<BacktestResult | null>(null);
  const [runEvents, setRunEvents] = useState<JobEvent[]>([]);
  const [renderMode, setRenderMode] = useState<"candlestick" | "line">("candlestick");
  const [viewMode, setViewMode] = useState<"history" | "prediction" | "backtest">("history");
  const [glossaryOpen, setGlossaryOpen] = useState(false);
  const [lastBacktestKey, setLastBacktestKey] = useState<string | null>(null);
  const [markerFilters, setMarkerFilters] = useState<{ buy: boolean; sell: boolean }>({ buy: true, sell: true });
  const [segmentFilters, setSegmentFilters] = useState<{ history: boolean; forecast: boolean }>({ history: true, forecast: true });
  const [scopeFilters, setScopeFilters] = useState<{ prediction: boolean; backtest: boolean }>({ prediction: true, backtest: false });
  const [equitySeriesVisibility, setEquitySeriesVisibility] = useState<{ strategy: boolean; buyHold: boolean }>({ strategy: true, buyHold: true });
  const toggleButtonClass = (active: boolean) =>
    cn(
      "rounded-md border px-2.5 py-1 text-xs font-semibold transition",
      active ? "border-sky-500/80 bg-sky-500/10 text-sky-100" : "border-gray-800 text-gray-400 hover:border-gray-600"
    );

  const modelsQuery = useQuery({ queryKey: ["models"], queryFn: listModels, refetchInterval: 60_000 });
  const selectedModelQuery = useQuery<ModelMeta | undefined>({
    queryKey: ["models", selectedModelId],
    queryFn: () => (selectedModelId ? loadModel(selectedModelId) : Promise.resolve(undefined)),
    enabled: Boolean(selectedModelId),
  });

  const addRunEvent = useCallback((message: string, status: JobEvent["status"] = "running") => {
    setRunEvents((prev) => [...prev, { timestamp: new Date().toISOString(), message, status }]);
  }, []);

  const predictMutation = useMutation({
    mutationFn: runPrediction,
    onSuccess: (result: PredictionResult) => {
      setPrediction(result);
      setBacktest(null);
      toast.success(`Prediction ready for ${result.symbol}`);
    },
    onError: (error: Error) => toast.error(error.message),
  });

  const backtestMutation = useMutation({
    mutationFn: runBacktest,
    onSuccess: (result: BacktestResult) => {
      setBacktest(result);
      toast.success("Backtest complete");
    },
    onError: (error: Error) => toast.error(error.message),
  });

  useKeyboardShortcuts([
    { key: "i", handler: () => handlePredict() },
    { key: "f", handler: () => handleBacktest() },
  ]);


  const handlePredict = useCallback(async () => {
    if (viewMode === "backtest") {
      toast.info("Switch to History or Prediction mode before running a new inference");
      return;
    }
    if (!predictionParams.symbol) {
      toast.error("Add a symbol to run inference");
      return;
    }
    addRunEvent("Prediction requested", "running");
    await predictMutation.mutateAsync({ ...predictionParams, modelId: selectedModelId, fusion: fusionSettings });
    addRunEvent("Prediction finished", "completed");
  }, [viewMode, predictionParams, addRunEvent, predictMutation, selectedModelId, fusionSettings]);

  const handleBacktest = useCallback(async () => {
    if (!prediction) {
      toast.error("Generate a prediction first");
      return;
    }
    addRunEvent("Backtest started", "running");
    await backtestMutation.mutateAsync({ prediction, params: backtestParams });
    addRunEvent("Backtest finished", "completed");
  }, [prediction, addRunEvent, backtestMutation, backtestParams]);

  useShortcuts([
    { combo: "ctrl+h", handler: () => setViewMode("history") },
    { combo: "ctrl+p", handler: () => void handlePredict() },
    { combo: "ctrl+b", handler: () => setViewMode("backtest") },
  ]);

  useEffect(() => {
    if (viewMode !== "backtest" || !prediction) return;
    const key = `${prediction.symbol}-${prediction.dates.at(-1) ?? ""}-${prediction.predictedPrices.length}`;
    if (key === lastBacktestKey || backtestMutation.isPending) return;
    setLastBacktestKey(key);
    void handleBacktest();
  }, [viewMode, prediction, lastBacktestKey, backtestMutation.isPending, handleBacktest]);

  const applyPreset = (preset: keyof typeof QUICK_PRESETS) => {
    const config = QUICK_PRESETS[preset];
    setFusionSettings((prev) => ({ ...prev, ...config.fusion }));
    setPredictionParams((prev) => ({ ...prev, confidenceFloors: config.confidenceFloors }));
    toast.success(`${preset} preset applied`);
  };

  const downloadArtifacts = async () => {
    if (!selectedModelId) {
      toast.error("Select a model to download");
      return;
    }
    const blob = await downloadModelArtifacts(selectedModelId);
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `${selectedModelId}.zip`;
    link.click();
    URL.revokeObjectURL(link.href);
  };

  const candlestickData: CandlestickPoint[] = useMemo(() => {
    if (!prediction) return [];
    const baseSeries = prediction.candles
      ? [...prediction.candles]
      : prediction.dates.map((date, index) => {
          const close = prediction.prices[index] ?? prediction.predictedPrices[index];
          const prev = prediction.prices[index - 1] ?? close;
          const high = Math.max(close, prev);
          const low = Math.min(close, prev);
          return { date, open: prev, high, low, close };
        });
    const limit = Math.max(1, predictionParams.daysOnChart ?? baseSeries.length);
    return baseSeries.slice(-limit);
  }, [prediction, predictionParams.daysOnChart]);

  const lastObservedPrice = candlestickData.at(-1)?.close ?? prediction?.prices.at(-1);

  const predictedSeries = useMemo(() => {
    if (!prediction) return [];
    const history = prediction.dates.map((date, index) => ({
      date,
      price: prediction.predictedPrices[index] ?? prediction.prices[index],
      segment: "history" as const,
    }));
    const historyLimit = Math.max(1, predictionParams.daysOnChart ?? history.length);
    const trimmedHistory = history.slice(-historyLimit);
    const forecast = prediction.forecast;
    const future = forecast
      ? forecast.dates.map((date, index) => ({
          date,
          price: forecast.prices[index] ?? forecast.prices[index - 1] ?? trimmedHistory.at(-1)?.price ?? lastObservedPrice ?? 0,
          segment: "forecast" as const,
        }))
      : [];
    return [...trimmedHistory, ...future];
  }, [prediction, predictionParams.daysOnChart, lastObservedPrice]);

  const forecastChanges = useMemo(() => {
    if (!prediction?.forecast || !segmentFilters.forecast) return [];
    return prediction.forecast.dates.map((date, index) => ({
      date,
      value: prediction.forecast?.returns[index] ?? 0,
    }));
  }, [prediction?.forecast, segmentFilters.forecast]);

  const markerWindow = useMemo(() => {
    const toTimestamp = (value?: string) => {
      if (!value) return Number.NaN;
      const ts = new Date(value).getTime();
      return Number.isNaN(ts) ? Number.NaN : ts;
    };

    const historyStart = toTimestamp(candlestickData[0]?.date);
    const historyEnd = toTimestamp(candlestickData.at(-1)?.date);
    const forecastEnd = toTimestamp(prediction?.forecast?.dates.at(-1));
    const endCandidates = [historyEnd, forecastEnd].filter((value) => Number.isFinite(value)) as number[];
    const end = endCandidates.length ? Math.max(...endCandidates) : historyStart;

    if (!Number.isFinite(historyStart) || !Number.isFinite(end)) return null;
    return { start: historyStart, end };
  }, [candlestickData, prediction?.forecast?.dates]);

  const effectiveTradeShareFloor = prediction?.metadata?.tradeShareFloor ?? predictionParams.tradeShareFloor ?? 0;

  const classificationOptions = useMemo(
    () => ({ shareThreshold: effectiveTradeShareFloor, confidenceFloor: SIGNAL_CONFIDENCE_FLOOR }),
    [effectiveTradeShareFloor]
  );

  const allMarkers = useMemo(() => {
    const normalized: ChartMarker[] = [];
    const forecastDates = new Set(prediction?.forecast?.dates ?? []);
    if (prediction?.tradeMarkers?.length) {
      normalized.push(
        ...prediction.tradeMarkers.map((marker) =>
          applySignalThreshold(
            {
              ...marker,
              segment: marker.segment ?? (forecastDates.has(marker.date) ? "forecast" : "history"),
              scope: marker.scope ?? "prediction",
            },
            classificationOptions
          ) as ChartMarker
        )
      );
    }
    if (backtest?.annotations?.length) {
      normalized.push(
        ...backtest.annotations.map((marker) =>
          applySignalThreshold(
            {
              ...marker,
              segment: marker.segment ?? "history",
              scope: marker.scope ?? "backtest",
            },
            classificationOptions
          ) as ChartMarker
        )
      );
    }
    if (!markerWindow) {
      return normalized;
    }
    return normalized.filter((marker) => {
      const ts = new Date(marker.date).getTime();
      if (Number.isNaN(ts)) return false;
      return ts >= markerWindow.start && ts <= markerWindow.end;
    });
  }, [prediction?.tradeMarkers, prediction?.forecast?.dates, backtest?.annotations, markerWindow, classificationOptions]);

  const filteredMarkers: ChartMarker[] = useMemo(() => {
    return allMarkers.filter((marker) => {
      const segment = marker.segment ?? "history";
      const scope = marker.scope ?? "prediction";
      const displayType = ((marker as ChartMarker).displayType ?? marker.type) as "buy" | "sell" | "hold";
      if (displayType !== "hold" && !markerFilters[displayType]) {
        return false;
      }
      return Boolean(segmentFilters[segment]) && Boolean(scopeFilters[scope]);
    });
  }, [allMarkers, markerFilters, segmentFilters, scopeFilters]);

  const backtestMarkers = useMemo(() => filteredMarkers.filter((marker) => (marker.scope ?? "prediction") === "backtest"), [filteredMarkers]);

  const chartMarkers = useMemo(() => {
    if (viewMode === "history") {
      return filteredMarkers.filter(
        (marker) => (marker.scope ?? "prediction") === "prediction" && (marker.segment ?? "history") === "history"
      );
    }
    if (viewMode === "prediction") {
      return filteredMarkers.filter((marker) => (marker.scope ?? "prediction") === "prediction");
    }
    return filteredMarkers;
  }, [filteredMarkers, viewMode]);

  const toggleMarkerVisibility = (type: keyof typeof markerFilters) => {
    setMarkerFilters((prev) => ({ ...prev, [type]: !prev[type] }));
  };

  const toggleSegmentVisibility = (segment: keyof typeof segmentFilters) => {
    setSegmentFilters((prev) => ({ ...prev, [segment]: !prev[segment] }));
  };

  const toggleScopeVisibility = (scope: keyof typeof scopeFilters) => {
    setScopeFilters((prev) => ({ ...prev, [scope]: !prev[scope] }));
  };

  const toggleEquitySeries = (series: keyof typeof equitySeriesVisibility) => {
    setEquitySeriesVisibility((prev) => ({ ...prev, [series]: !prev[series] }));
  };

  const onToggleComparison = (modelId: string) => {
    setComparisonSelection((prev) => {
      if (prev.includes(modelId)) {
        return prev.filter((id) => id !== modelId);
      }
      if (prev.length >= 3) {
        return [...prev.slice(1), modelId];
      }
      return [...prev, modelId];
    });
  };

  const selectedModel = selectedModelQuery.data;
  const modeOptions: Array<{ value: typeof viewMode; label: string; helper: string }> = [
    { value: "history", label: "History", helper: "Show only realized candles" },
    { value: "prediction", label: "Prediction", helper: "Overlay the AI forecast" },
    { value: "backtest", label: "Backtest", helper: "Show simulated equity & trades" },
  ];
  const overlaysForChart = viewMode === "history" ? undefined : prediction?.overlays;
  const forecastChangesForChart = viewMode === "history" ? undefined : forecastChanges;
  const predictedSeriesForChart = viewMode === "history" ? [] : predictedSeries;
  const showPredictionInsights = viewMode !== "history";
  const showBacktestPanels = viewMode === "backtest";
  const isPredictionDisabled = predictMutation.isPending || viewMode === "backtest";

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <div className="mx-auto max-w-[1600px] space-y-6 px-4 py-6">
        <header className="space-y-2">
          <p className="text-xs uppercase tracking-wide text-gray-500">AI-Stocks v1 · unified control surface</p>
          <h1 className="text-3xl font-semibold text-white">AI Lifecycle Console</h1>
          <p className="text-sm text-gray-400">
            Orchestrate inference, backtest fusion strategies, and export results from one console. Training jobs now run exclusively via the CLI pipeline.
          </p>
        </header>

        <div className="grid gap-6 xl:grid-cols-[360px,1fr,360px]">
          <aside className="space-y-4">
            <ModelControlPanel
              models={modelsQuery.data}
              modelsLoading={modelsQuery.isRefetching}
              selectedModelId={selectedModelId}
              onSelectModel={(id) => {
                setSelectedModelId(id);
                setPredictionParams((prev) => ({ ...prev, modelId: id }));
              }}
              predictionParams={predictionParams}
              onChangePredictionParams={(partial) => setPredictionParams((prev) => ({ ...prev, ...partial, confidenceFloors: partial.confidenceFloors ?? prev.confidenceFloors }))}
              fusionSettings={fusionSettings}
              onChangeFusionSettings={(partial) => setFusionSettings((prev) => ({ ...prev, ...partial, regimeFilters: partial.regimeFilters ?? prev.regimeFilters }))}
              backtestParams={backtestParams}
              onChangeBacktestParams={(partial) => setBacktestParams((prev) => ({ ...prev, ...partial }))}
              onApplyPreset={applyPreset}
              onPredict={handlePredict}
              onBacktest={handleBacktest}
              disablePredict={predictMutation.isPending}
              disableBacktest={backtestMutation.isPending}
            />

            <ModelComparisonPanel models={modelsQuery.data} selected={comparisonSelection} onToggle={onToggleComparison} />

            <Button variant="outline" className="w-full border-gray-700 text-gray-200" onClick={downloadArtifacts}>
              Download Model Artifacts
            </Button>
          </aside>

          <main className="space-y-4">
            <section className="rounded-2xl border border-gray-900 bg-gray-900/60 p-4">
              <div className="flex flex-wrap items-center gap-4">
                <div>
                  <p className="text-xs text-gray-500">Loaded Model</p>
                  <p className="text-lg font-semibold text-white">{selectedModel?.symbol ?? "—"}</p>
                  {selectedModel && <p className="text-xs text-gray-500">{new Date(selectedModel.createdAt).toLocaleString()}</p>}
                </div>
                <div>
                  <p className="text-xs text-gray-500">Latest Prediction</p>
                  <p className="text-lg font-semibold text-white">{prediction?.symbol ?? "Waiting"}</p>
                </div>
                {prediction && (
                  <div>
                    <p className="text-xs text-gray-500">Last price</p>
                    <p className="text-lg font-semibold text-emerald-400">{formatCurrency(prediction.prices.at(-1) ?? 0)}</p>
                  </div>
                )}
                <div className="ml-auto flex gap-2">
                  <Button onClick={handlePredict} disabled={isPredictionDisabled}>
                    Run Inference (⌘I)
                  </Button>
                  <Button variant="secondary" onClick={handleBacktest} disabled={backtestMutation.isPending}>
                    Backtest (⌘F)
                  </Button>
                </div>
              </div>
            </section>

            <section className="space-y-4" aria-label="Primary charts">
              <div className="flex flex-wrap gap-3 rounded-2xl border border-gray-900 bg-gray-900/60 p-3 text-xs text-gray-400">
                <div>
                  <p className="text-[11px] uppercase tracking-wide text-gray-500">Price View</p>
                  <div className="mt-1 flex gap-1">
                    <button type="button" className={toggleButtonClass(renderMode === "candlestick")} onClick={() => setRenderMode("candlestick")}>
                      Candles
                    </button>
                    <button type="button" className={toggleButtonClass(renderMode === "line")} onClick={() => setRenderMode("line")}>
                      Line
                    </button>
                  </div>
                </div>
                <div>
                  <p className="text-[11px] uppercase tracking-wide text-gray-500">Mode</p>
                  <div className="mt-1 flex gap-1">
                    {modeOptions.map((option) => (
                      <button
                        key={option.value}
                        type="button"
                        className={toggleButtonClass(viewMode === option.value)}
                        onClick={() => setViewMode(option.value)}
                        title={option.helper}
                        aria-label={option.helper}
                      >
                        {option.label}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <p className="text-[11px] uppercase tracking-wide text-gray-500">Markers</p>
                  <div className="mt-1 flex gap-1">
                    {(["buy", "sell"] as const).map((type) => (
                      <button key={type} type="button" className={toggleButtonClass(markerFilters[type])} onClick={() => toggleMarkerVisibility(type)}>
                        {type.toUpperCase()}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <p className="text-[11px] uppercase tracking-wide text-gray-500">Segments</p>
                  <div className="mt-1 flex gap-1">
                    {(["history", "forecast"] as Array<keyof typeof segmentFilters>).map((segment) => (
                      <button key={segment} type="button" className={toggleButtonClass(segmentFilters[segment])} onClick={() => toggleSegmentVisibility(segment)}>
                        {segment === "history" ? "History" : "Forecast"}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <p className="text-[11px] uppercase tracking-wide text-gray-500">Sources</p>
                  <div className="mt-1 flex gap-1">
                    {(["prediction", "backtest"] as Array<keyof typeof scopeFilters>).map((scope) => (
                      <button key={scope} type="button" className={toggleButtonClass(scopeFilters[scope])} onClick={() => toggleScopeVisibility(scope)}>
                        {scope === "prediction" ? "Prediction" : "Backtest"}
                      </button>
                    ))}
                  </div>
                </div>
                <div>
                  <p className="text-[11px] uppercase tracking-wide text-gray-500">Equity Series</p>
                  <div className="mt-1 flex gap-1">
                    <button type="button" className={toggleButtonClass(equitySeriesVisibility.strategy)} onClick={() => toggleEquitySeries("strategy")}>
                      Strategy
                    </button>
                    <button type="button" className={toggleButtonClass(equitySeriesVisibility.buyHold)} onClick={() => toggleEquitySeries("buyHold")}>
                      Buy & Hold
                    </button>
                  </div>
                </div>
                <div className="ml-auto">
                  <Dialog open={glossaryOpen} onOpenChange={setGlossaryOpen}>
                    <DialogTrigger asChild>
                      <button type="button" className="rounded-md border border-gray-800 px-3 py-1 text-[11px] font-semibold text-gray-200 hover:border-gray-600">
                        Glossary
                      </button>
                    </DialogTrigger>
                    <DialogContent className="max-w-md border-gray-800 bg-gray-950 text-gray-100">
                      <DialogHeader>
                        <DialogTitle>Quick Glossary</DialogTitle>
                        <DialogDescription>Short definitions so non-quants can follow along.</DialogDescription>
                      </DialogHeader>
                      <div className="space-y-3 text-sm">
                        {GLOSSARY_ITEMS.map((item) => (
                          <div key={item.term}>
                            <p className="text-xs font-semibold uppercase tracking-wide text-gray-400">{item.term}</p>
                            <p className="text-gray-300">{item.description}</p>
                          </div>
                        ))}
                      </div>
                    </DialogContent>
                  </Dialog>
                </div>
              </div>
              <InteractiveCandlestick
                ref={candlestickRef}
                historicalSeries={candlestickData}
                overlays={overlaysForChart}
                tradeMarkers={chartMarkers}
                predictedSeries={predictedSeriesForChart}
                forecastChanges={forecastChangesForChart}
                renderMode={renderMode}
                mode={viewMode}
                showBuyMarkers={markerFilters.buy}
                showSellMarkers={markerFilters.sell}
                segmentFilters={segmentFilters}
                scopeFilters={scopeFilters}
                ariaLabel="Price action chart"
              />
              {showPredictionInsights && (
                <PredictionActionList
                  forecast={prediction?.forecast}
                  tradeMarkers={prediction?.tradeMarkers}
                  lastPrice={lastObservedPrice}
                  tradeShareFloor={effectiveTradeShareFloor}
                />
              )}
              {showBacktestPanels && (
                <>
                  <EquityLineChart
                    backtest={backtest ?? undefined}
                    tradeMarkers={backtestMarkers}
                    showStrategyEquity={equitySeriesVisibility.strategy}
                    showBuyHoldEquity={equitySeriesVisibility.buyHold}
                  />
                  <BacktestSummaryChart backtest={backtest ?? undefined} />
                </>
              )}
            </section>
          </main>

          <aside className="space-y-4">
            <DiagnosticsPanel prediction={prediction} backtest={backtest} />
            <ScenarioBuilder prediction={prediction} />
            <BacktestResults backtest={backtest} prediction={prediction} onExportChart={() => candlestickRef.current?.exportImage()} />
            <StreamingLog title="Run Activity" events={runEvents} />
          </aside>
        </div>

        {prediction?.metadata && (
          <div className="rounded-2xl border border-gray-900 bg-gray-900/60 p-4 text-sm text-gray-300">
            <p className="text-xs uppercase tracking-wide text-gray-500">Metadata</p>
            <div className="flex flex-wrap gap-4">
              <span>Fusion mode: {prediction.metadata.fusionMode}</span>
              <span>Buy threshold: {prediction.metadata.buyThreshold.toFixed(2)}</span>
              <span>Sell threshold: {prediction.metadata.sellThreshold.toFixed(2)}</span>
              <Button variant="ghost" size="sm" onClick={() => downloadJson("prediction-metadata.json", prediction.metadata)}>
                Download JSON
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

