"use client";

import { useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { toast } from "sonner";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  BacktestParams,
  BacktestResult,
  FusionSettings,
  ModelMeta,
  PredictionParams,
  PredictionResult,
} from "@/types/ai";
import { listModels, runBacktest, runPrediction } from "@/lib/api-client";
import InteractiveCandlestick, {
  CandlestickPoint,
  ChartMarker,
} from "@/components/charts/InteractiveCandlestick";
import { EquityLineChart } from "@/components/charts/EquityLineChart";
import { Button } from "@/components/ui/button";
import { formatCurrency, formatPercent } from "@/utils/formatters";
import { cn } from "@/lib/utils";

const DEFAULT_PREDICTION: PredictionParams = {
  symbol: "AAPL",
  horizon: 10,
  daysOnChart: 180,
  smoothing: "none",
  confidenceFloors: { buy: 0.3, sell: 0.45 },
  tradeShareFloor: 10,
};

const DEFAULT_FUSION: FusionSettings = {
  mode: "weighted",
  regressorScale: 15,
  buyThreshold: 0.3,
  sellThreshold: 0.45,
  regimeFilters: { bull: true, bear: true },
};

const DEFAULT_BACKTEST: BacktestParams = {
  backtestWindow: 120,
  initialCapital: 10_000,
  maxLong: 1.6,
  maxShort: 0,
  commission: 0.5,
  slippage: 0.2,
  enableForwardSim: true,
  shortCap: 0,
};

function MetricCard({
  label,
  value,
  tone = "neutral",
}: {
  label: string;
  value: string;
  tone?: "neutral" | "good" | "bad";
}) {
  return (
    <div className="rounded-xl border border-gray-700/70 bg-gray-900/70 px-4 py-3">
      <p className="text-[11px] uppercase tracking-[0.16em] text-gray-500">{label}</p>
      <p
        className={cn(
          "mt-1 text-xl font-semibold",
          tone === "good" && "text-teal-300",
          tone === "bad" && "text-red-300",
          tone === "neutral" && "text-gray-100"
        )}
      >
        {value}
      </p>
    </div>
  );
}

function ForwardSimulationPanel({ backtest }: { backtest: BacktestResult | null }) {
  const sim = backtest?.forwardSimulation;
  if (!sim) {
    return (
      <div className="rounded-2xl border border-gray-800 bg-gray-900/60 p-5 text-sm text-gray-400">
        Forward simulation appears here after running backtest with `Enable Forward Sim` on.
      </div>
    );
  }

  const data = sim.dates.map((date, i) => ({
    date,
    price: sim.prices[i],
    equity: sim.equityCurve[i],
  }));

  return (
    <div className="space-y-4 rounded-2xl border border-gray-800 bg-gray-900/60 p-5">
      <div className="flex flex-wrap gap-3">
        <MetricCard label="Forward Sharpe" value={sim.sharpe.toFixed(2)} tone={sim.sharpe >= 0 ? "good" : "bad"} />
        <MetricCard label="Forward Max DD" value={formatPercent(sim.maxDrawdown)} tone={sim.maxDrawdown < -0.2 ? "bad" : "neutral"} />
        <MetricCard label="Forward Trades" value={String(sim.trades)} />
      </div>
      <div className="h-64 w-full rounded-xl border border-gray-800 bg-gray-950/70 p-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
            <CartesianGrid stroke="rgba(255,255,255,0.06)" />
            <XAxis dataKey="date" tick={{ fill: "#9ca3af", fontSize: 11 }} tickFormatter={(v) => new Date(v).toLocaleDateString()} />
            <YAxis yAxisId="left" tick={{ fill: "#9ca3af", fontSize: 11 }} />
            <YAxis yAxisId="right" orientation="right" tick={{ fill: "#9ca3af", fontSize: 11 }} />
            <Tooltip
              contentStyle={{ background: "#0b0d10", border: "1px solid #20242c" }}
              formatter={(v: number, key: string) => [key === "price" ? formatCurrency(v) : formatCurrency(v), key]}
            />
            <Line yAxisId="left" type="monotone" dataKey="price" stroke="#38bdf8" strokeWidth={2} dot={false} />
            <Line yAxisId="right" type="monotone" dataKey="equity" stroke="#facc15" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export default function AiPage() {
  const [predictionParams, setPredictionParams] = useState<PredictionParams>(DEFAULT_PREDICTION);
  const [fusionSettings, setFusionSettings] = useState<FusionSettings>(DEFAULT_FUSION);
  const [backtestParams, setBacktestParams] = useState<BacktestParams>(DEFAULT_BACKTEST);
  const [selectedModelId, setSelectedModelId] = useState<string>("");

  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [backtest, setBacktest] = useState<BacktestResult | null>(null);

  const modelsQuery = useQuery({
    queryKey: ["models"],
    queryFn: listModels,
    refetchInterval: 60_000,
  });

  const predictMutation = useMutation({
    mutationFn: runPrediction,
    onSuccess: (result) => {
      setPrediction(result);
      setBacktest(null);
      toast.success(`Prediction ready for ${result.symbol}`);
    },
    onError: (err: Error) => toast.error(err.message),
  });

  const backtestMutation = useMutation({
    mutationFn: runBacktest,
    onSuccess: (result) => {
      setBacktest(result);
      toast.success("Backtest complete");
    },
    onError: (err: Error) => toast.error(err.message),
  });

  const selectedModel = useMemo(() => {
    return modelsQuery.data?.find((m) => m.id === selectedModelId);
  }, [modelsQuery.data, selectedModelId]);

  const candles: CandlestickPoint[] = useMemo(() => {
    if (!prediction) return [];
    if (prediction.candles?.length) {
      return prediction.candles.map((c) => ({
        date: c.date,
        open: c.open,
        high: c.high,
        low: c.low,
        close: c.close,
        volume: c.volume,
      }));
    }
    return prediction.dates.map((date, i) => {
      const close = prediction.prices[i] ?? 0;
      const prev = prediction.prices[i - 1] ?? close;
      return {
        date,
        open: prev,
        high: Math.max(prev, close),
        low: Math.min(prev, close),
        close,
      };
    });
  }, [prediction]);

  const predictedSeries = useMemo(() => {
    if (!prediction) return [];
    const history = prediction.dates.map((date, i) => ({
      date,
      price: prediction.predictedPrices[i] ?? prediction.prices[i] ?? 0,
      segment: "history" as const,
    }));

    const future = prediction.forecast
      ? prediction.forecast.dates.map((date, i) => ({
          date,
          price: prediction.forecast?.prices[i] ?? history.at(-1)?.price ?? 0,
          segment: "forecast" as const,
        }))
      : [];

    return [...history, ...future];
  }, [prediction]);

  const chartMarkers: ChartMarker[] = useMemo(() => {
    const p = prediction?.tradeMarkers ?? [];
    const b = backtest?.annotations ?? [];
    return [...p, ...b] as ChartMarker[];
  }, [prediction?.tradeMarkers, backtest?.annotations]);

  const handlePredict = async () => {
    if (!predictionParams.symbol.trim()) {
      toast.error("Please enter a symbol");
      return;
    }

    await predictMutation.mutateAsync({
      ...predictionParams,
      modelId: selectedModelId || undefined,
      fusion: fusionSettings,
    });
  };

  const handleBacktest = async () => {
    if (!prediction) {
      toast.error("Run prediction first");
      return;
    }

    await backtestMutation.mutateAsync({
      prediction,
      params: backtestParams,
    });
  };

  const metrics = backtest?.metrics;

  return (
    <div className="relative overflow-hidden">
      <div className="pointer-events-none absolute -left-24 -top-24 h-64 w-64 rounded-full bg-yellow-500/20 blur-3xl" />
      <div className="pointer-events-none absolute right-0 top-20 h-72 w-72 rounded-full bg-teal-400/15 blur-3xl" />

      <section className="mb-6 rounded-3xl border border-gray-700/70 bg-gradient-to-br from-gray-900 via-gray-900/95 to-gray-950 p-6 shadow-[0_10px_60px_rgba(0,0,0,0.45)]">
        <p className="text-xs uppercase tracking-[0.24em] text-yellow-400">Ralph Loop Console</p>
        <h1 className="mt-2 font-mono text-3xl font-semibold text-gray-100">AI Trade Lab</h1>
        <p className="mt-2 max-w-3xl text-sm text-gray-300">
          End-to-end visibility for prediction, inventory-aware backtesting, and forward simulation. This page uses the live backend contract so you can inspect trades, equity, and execution behavior directly.
        </p>
      </section>

      <div className="grid gap-6 xl:grid-cols-[350px,1fr]">
        <aside className="space-y-4 rounded-2xl border border-gray-800 bg-gray-900/70 p-4">
          <div className="space-y-2">
            <p className="text-xs uppercase tracking-[0.18em] text-gray-500">Symbol + Horizon</p>
            <input
              className="w-full rounded-lg border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-gray-100 outline-none focus:border-yellow-400"
              value={predictionParams.symbol}
              onChange={(e) => setPredictionParams((p) => ({ ...p, symbol: e.target.value.toUpperCase() }))}
              placeholder="AAPL"
            />
            <div className="grid grid-cols-2 gap-2">
              <input
                className="rounded-lg border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-gray-100 outline-none focus:border-yellow-400"
                type="number"
                value={predictionParams.horizon}
                min={1}
                onChange={(e) => setPredictionParams((p) => ({ ...p, horizon: Number(e.target.value) || 1 }))}
                placeholder="Horizon"
              />
              <input
                className="rounded-lg border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-gray-100 outline-none focus:border-yellow-400"
                type="number"
                value={predictionParams.daysOnChart}
                min={30}
                onChange={(e) => setPredictionParams((p) => ({ ...p, daysOnChart: Number(e.target.value) || 120 }))}
                placeholder="Chart Days"
              />
            </div>
          </div>

          <div className="space-y-2">
            <p className="text-xs uppercase tracking-[0.18em] text-gray-500">Model + Fusion</p>
            <select
              className="w-full rounded-lg border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-gray-100 outline-none focus:border-yellow-400"
              value={selectedModelId}
              onChange={(e) => setSelectedModelId(e.target.value)}
            >
              <option value="">Latest by Symbol</option>
              {(modelsQuery.data ?? []).map((m: ModelMeta) => (
                <option key={m.id} value={m.id}>{`${m.symbol} - ${new Date(m.createdAt).toLocaleDateString()}`}</option>
              ))}
            </select>
            <div className="grid grid-cols-2 gap-2">
              <input
                className="rounded-lg border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-gray-100 outline-none focus:border-yellow-400"
                type="number"
                step="0.01"
                value={fusionSettings.buyThreshold}
                onChange={(e) => setFusionSettings((f) => ({ ...f, buyThreshold: Number(e.target.value) || 0.3 }))}
                placeholder="Buy Thr"
              />
              <input
                className="rounded-lg border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-gray-100 outline-none focus:border-yellow-400"
                type="number"
                step="0.01"
                value={fusionSettings.sellThreshold}
                onChange={(e) => setFusionSettings((f) => ({ ...f, sellThreshold: Number(e.target.value) || 0.45 }))}
                placeholder="Sell Thr"
              />
            </div>
          </div>

          <div className="space-y-2">
            <p className="text-xs uppercase tracking-[0.18em] text-gray-500">Backtest Engine</p>
            <div className="grid grid-cols-2 gap-2">
              <input
                className="rounded-lg border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-gray-100 outline-none focus:border-yellow-400"
                type="number"
                value={backtestParams.initialCapital}
                onChange={(e) => setBacktestParams((b) => ({ ...b, initialCapital: Number(e.target.value) || 10_000 }))}
                placeholder="Capital"
              />
              <input
                className="rounded-lg border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-gray-100 outline-none focus:border-yellow-400"
                type="number"
                step="0.1"
                value={backtestParams.maxLong}
                onChange={(e) => setBacktestParams((b) => ({ ...b, maxLong: Number(e.target.value) || 1.6 }))}
                placeholder="Max Long"
              />
            </div>
            <label className="flex items-center gap-2 text-sm text-gray-300">
              <input
                type="checkbox"
                checked={backtestParams.enableForwardSim}
                onChange={(e) => setBacktestParams((b) => ({ ...b, enableForwardSim: e.target.checked }))}
              />
              Enable Forward Simulation
            </label>
          </div>

          <div className="space-y-2 pt-2">
            <Button className="w-full yellow-btn" onClick={handlePredict} disabled={predictMutation.isPending}>
              {predictMutation.isPending ? "Running Prediction..." : "Run Prediction"}
            </Button>
            <Button
              className="w-full border border-gray-600 bg-gray-950 text-gray-100 hover:bg-gray-900"
              onClick={handleBacktest}
              disabled={backtestMutation.isPending || !prediction}
            >
              {backtestMutation.isPending ? "Running Backtest..." : "Run Backtest + Forward Sim"}
            </Button>
          </div>

          <div className="rounded-xl border border-gray-800 bg-gray-950/70 p-3 text-xs text-gray-400">
            <p className="font-semibold text-gray-200">Execution constraints</p>
            <p className="mt-1">Long-only inventory backtester blocks impossible sells and includes commission/slippage in PnL.</p>
          </div>
        </aside>

        <main className="space-y-6">
          <section className="rounded-2xl border border-gray-800 bg-gray-900/60 p-4">
            <div className="mb-3 flex flex-wrap items-end justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.18em] text-gray-500">Loaded</p>
                <p className="font-mono text-lg text-gray-100">
                  {prediction?.symbol ?? predictionParams.symbol} {selectedModel ? `• ${selectedModel.id}` : ""}
                </p>
              </div>
              <div className="text-sm text-gray-300">
                Quality gate: {String(prediction?.metadata?.modelQualityGatePassed ?? false)}
              </div>
            </div>

            <InteractiveCandlestick
              historicalSeries={candles}
              predictedSeries={predictedSeries}
              tradeMarkers={chartMarkers}
              mode="backtest"
              renderMode="candlestick"
              segmentFilters={{ history: true, forecast: true }}
              scopeFilters={{ prediction: true, backtest: true }}
              ariaLabel="Prediction and execution chart"
            />
          </section>

          <section className="grid gap-3 md:grid-cols-3">
            <MetricCard
              label="Cumulative Return"
              value={metrics ? formatPercent(metrics.cumulativeReturn) : "-"}
              tone={metrics && metrics.cumulativeReturn >= 0 ? "good" : "bad"}
            />
            <MetricCard
              label="Sharpe"
              value={metrics ? metrics.sharpeRatio.toFixed(2) : "-"}
              tone={metrics && metrics.sharpeRatio >= 0 ? "good" : "bad"}
            />
            <MetricCard
              label="Total Trades"
              value={metrics ? String(metrics.totalTrades) : "-"}
            />
          </section>

          <EquityLineChart backtest={backtest ?? undefined} tradeMarkers={backtest?.annotations} />

          <ForwardSimulationPanel backtest={backtest} />

          <section className="rounded-2xl border border-gray-800 bg-gray-900/60 p-4">
            <p className="mb-3 text-xs uppercase tracking-[0.18em] text-gray-500">Trade Log</p>
            {!backtest?.tradeLog?.length ? (
              <p className="text-sm text-gray-400">Run backtest to populate trade details.</p>
            ) : (
              <div className="max-h-80 overflow-auto">
                <table className="w-full text-left text-xs text-gray-300">
                  <thead className="sticky top-0 bg-gray-900">
                    <tr>
                      <th className="px-2 py-2">Date</th>
                      <th className="px-2 py-2">Action</th>
                      <th className="px-2 py-2">Price</th>
                      <th className="px-2 py-2">Shares</th>
                      <th className="px-2 py-2">Position</th>
                      <th className="px-2 py-2">PnL</th>
                      <th className="px-2 py-2">Notes</th>
                    </tr>
                  </thead>
                  <tbody>
                    {backtest.tradeLog.map((trade) => (
                      <tr key={trade.id} className="border-t border-gray-800">
                        <td className="px-2 py-2">{trade.date}</td>
                        <td className={cn("px-2 py-2 font-semibold", trade.action === "BUY" ? "text-teal-300" : "text-red-300")}>{trade.action}</td>
                        <td className="px-2 py-2">{formatCurrency(trade.price)}</td>
                        <td className="px-2 py-2">{trade.shares.toFixed(2)}</td>
                        <td className="px-2 py-2">{trade.position.toFixed(2)}</td>
                        <td className="px-2 py-2">{formatCurrency(trade.pnl)}</td>
                        <td className="px-2 py-2 text-gray-400">{trade.explanation?.notes ?? "-"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </section>
        </main>
      </div>
    </div>
  );
}
