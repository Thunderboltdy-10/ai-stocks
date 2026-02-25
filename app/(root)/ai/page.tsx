"use client";

import { useMemo, useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { toast } from "sonner";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Cell,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import {
  BacktestDiagnostics,
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
  modelVariant: "auto",
  dataInterval: "1d",
  dataPeriod: "10y",
  maxLong: 1.6,
  maxShort: 0.25,
  horizon: 10,
  daysOnChart: 180,
  smoothing: "none",
  confidenceFloors: { buy: 0.3, sell: 0.45 },
  tradeShareFloor: 10,
};

const DEFAULT_FUSION: FusionSettings = {
  mode: "gbm_only",
  regressorScale: 15,
  buyThreshold: 0.3,
  sellThreshold: 0.45,
  regimeFilters: { bull: true, bear: true },
};

const DEFAULT_BACKTEST: BacktestParams = {
  backtestWindow: 120,
  initialCapital: 10_000,
  maxLong: 1.6,
  maxShort: 0.25,
  dataInterval: "1d",
  annualizationFactor: undefined,
  flatAtDayEnd: undefined,
  minPositionChange: 0.0,
  commission: 0.5,
  slippage: 0.2,
  enableForwardSim: true,
  shortCap: 0.25,
};

function toDateTick(value: string): string {
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

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
    <div className="rounded-xl border border-zinc-700/70 bg-zinc-900/80 px-4 py-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.02)]">
      <p className="text-[10px] uppercase tracking-[0.18em] text-zinc-500">{label}</p>
      <p
        className={cn(
          "mt-1 text-xl font-semibold",
          tone === "good" && "text-emerald-300",
          tone === "bad" && "text-rose-300",
          tone === "neutral" && "text-zinc-100"
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
      <div className="rounded-2xl border border-zinc-800 bg-zinc-950/70 p-5 text-sm text-zinc-400">
        Forward simulation appears here after backtest.
      </div>
    );
  }

  const data = sim.dates.map((date, i) => ({
    date,
    price: sim.prices[i],
    equity: sim.equityCurve[i],
  }));

  return (
    <div className="space-y-4 rounded-2xl border border-zinc-800 bg-zinc-950/70 p-5">
      <div className="grid gap-3 md:grid-cols-5">
        <MetricCard label="Forward Sharpe" value={sim.sharpe.toFixed(2)} tone={sim.sharpe >= 0 ? "good" : "bad"} />
        <MetricCard label="Forward Max DD" value={formatPercent(sim.maxDrawdown)} tone={sim.maxDrawdown < -0.2 ? "bad" : "neutral"} />
        <MetricCard label="Forward Trades" value={String(sim.trades)} />
        <MetricCard label="Forward Costs" value={formatCurrency(sim.totalCosts ?? 0)} tone={(sim.totalCosts ?? 0) > 0 ? "bad" : "neutral"} />
        <MetricCard label="Borrow Fee" value={formatCurrency(sim.borrowFee ?? 0)} tone={(sim.borrowFee ?? 0) > 0 ? "bad" : "neutral"} />
      </div>
      <div className="h-64 w-full rounded-xl border border-zinc-800 bg-black/40 p-2">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data}>
            <CartesianGrid stroke="rgba(255,255,255,0.06)" />
            <XAxis dataKey="date" tick={{ fill: "#9ca3af", fontSize: 11 }} tickFormatter={toDateTick} />
            <YAxis yAxisId="left" tick={{ fill: "#9ca3af", fontSize: 11 }} />
            <YAxis yAxisId="right" orientation="right" tick={{ fill: "#9ca3af", fontSize: 11 }} />
            <Tooltip
              contentStyle={{ background: "#0b0b0b", border: "1px solid #262626" }}
              formatter={(v: number) => formatCurrency(v)}
              labelFormatter={(v) => String(v)}
            />
            <Line yAxisId="left" type="monotone" dataKey="price" stroke="#38bdf8" strokeWidth={2} dot={false} />
            <Line yAxisId="right" type="monotone" dataKey="equity" stroke="#eab308" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

function DiagnosticsPanel({ diagnostics, intraday }: { diagnostics?: BacktestDiagnostics; intraday: boolean }) {
  if (!diagnostics) {
    return (
      <div className="rounded-2xl border border-zinc-800 bg-zinc-950/70 p-5 text-sm text-zinc-400">
        Diagnostics will populate after backtest.
      </div>
    );
  }

  const monthly = diagnostics.monthly.slice(-18);
  const actionData = diagnostics.actionBreakdown;
  const hourly = diagnostics.hourly;
  const rolling = diagnostics.rolling;

  return (
    <section className="space-y-4 rounded-2xl border border-zinc-800 bg-zinc-950/70 p-5">
      <div className="flex items-center justify-between">
        <p className="text-xs uppercase tracking-[0.18em] text-zinc-500">Diagnostics</p>
        <p className="text-xs text-zinc-400">Bank-grade risk/alpha telemetry</p>
      </div>

      <div className="grid gap-3 md:grid-cols-6">
        <MetricCard label="Exposure Mean" value={diagnostics.risk.exposureMean.toFixed(2)} />
        <MetricCard label="Exposure Std" value={diagnostics.risk.exposureStd.toFixed(2)} />
        <MetricCard label="Tail Loss P95" value={formatPercent(diagnostics.risk.tailLossP95)} tone="bad" />
        <MetricCard label="CVaR 95" value={formatPercent(diagnostics.risk.cvar95)} tone="bad" />
        <MetricCard label="Best Bar" value={formatPercent(diagnostics.risk.bestBar)} tone="good" />
        <MetricCard label="Worst Bar" value={formatPercent(diagnostics.risk.worstBar)} tone="bad" />
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <div className="h-72 rounded-xl border border-zinc-800 bg-black/40 p-2">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={rolling}>
              <CartesianGrid stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="date" tick={{ fill: "#9ca3af", fontSize: 11 }} tickFormatter={toDateTick} />
              <YAxis yAxisId="left" tick={{ fill: "#9ca3af", fontSize: 11 }} />
              <YAxis yAxisId="right" orientation="right" tick={{ fill: "#9ca3af", fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "#0b0b0b", border: "1px solid #262626" }} />
              <Line yAxisId="left" type="monotone" dataKey="rollingSharpe" stroke="#22d3ee" dot={false} strokeWidth={1.8} name="Rolling Sharpe" />
              <Line yAxisId="left" type="monotone" dataKey="rollingAlpha" stroke="#f59e0b" dot={false} strokeWidth={1.8} name="Rolling Alpha" />
              <Line yAxisId="right" type="monotone" dataKey="drawdown" stroke="#f43f5e" dot={false} strokeWidth={1.4} name="Drawdown" />
            </LineChart>
          </ResponsiveContainer>
        </div>

        <div className="h-72 rounded-xl border border-zinc-800 bg-black/40 p-2">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={rolling}>
              <CartesianGrid stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="date" tick={{ fill: "#9ca3af", fontSize: 11 }} tickFormatter={toDateTick} />
              <YAxis tick={{ fill: "#9ca3af", fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "#0b0b0b", border: "1px solid #262626" }} formatter={(v: number) => formatCurrency(v)} />
              <Area type="monotone" dataKey="equity" stroke="#facc15" fill="rgba(250, 204, 21, 0.2)" name="Strategy Equity" />
              <Area type="monotone" dataKey="buyHoldEquity" stroke="#60a5fa" fill="rgba(96, 165, 250, 0.16)" name="Buy & Hold Equity" />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <div className="h-72 rounded-xl border border-zinc-800 bg-black/40 p-2">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={monthly}>
              <CartesianGrid stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="period" tick={{ fill: "#9ca3af", fontSize: 10 }} />
              <YAxis tick={{ fill: "#9ca3af", fontSize: 11 }} tickFormatter={(v) => `${(Number(v) * 100).toFixed(0)}%`} />
              <Tooltip contentStyle={{ background: "#0b0b0b", border: "1px solid #262626" }} formatter={(v: number) => formatPercent(v)} />
              <Bar dataKey="alpha" name="Monthly Alpha">
                {monthly.map((r) => (
                  <Cell key={`${r.period}-${r.alpha}`} fill={r.alpha >= 0 ? "#10b981" : "#ef4444"} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="h-72 rounded-xl border border-zinc-800 bg-black/40 p-2">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={hourly}>
              <CartesianGrid stroke="rgba(255,255,255,0.06)" />
              <XAxis dataKey="hour" tick={{ fill: "#9ca3af", fontSize: 10 }} />
              <YAxis yAxisId="left" tick={{ fill: "#9ca3af", fontSize: 11 }} tickFormatter={(v) => `${(Number(v) * 100).toFixed(2)}%`} />
              <YAxis yAxisId="right" orientation="right" tick={{ fill: "#9ca3af", fontSize: 11 }} />
              <Tooltip contentStyle={{ background: "#0b0b0b", border: "1px solid #262626" }} />
              <Bar yAxisId="left" dataKey="avgStrategyReturn" name="Avg Return by Hour" fill="#22d3ee" />
              <Bar yAxisId="right" dataKey="tradeCount" name="Trade Count" fill="#f59e0b" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="rounded-xl border border-zinc-800 bg-black/30 p-3">
        <p className="mb-2 text-[11px] uppercase tracking-[0.18em] text-zinc-500">
          Action Quality {intraday ? "(Intraday Execution)" : ""}
        </p>
        <div className="overflow-auto">
          <table className="w-full text-left text-xs text-zinc-300">
            <thead className="sticky top-0 bg-zinc-900">
              <tr>
                <th className="px-2 py-2">Action</th>
                <th className="px-2 py-2">Count</th>
                <th className="px-2 py-2">Win Rate</th>
                <th className="px-2 py-2">Avg PnL</th>
                <th className="px-2 py-2">Total PnL</th>
              </tr>
            </thead>
            <tbody>
              {actionData.map((row) => (
                <tr key={row.action} className="border-t border-zinc-800">
                  <td className="px-2 py-2 font-semibold text-zinc-100">{row.action}</td>
                  <td className="px-2 py-2">{row.count}</td>
                  <td className={cn("px-2 py-2", row.winRate >= 0.5 ? "text-emerald-300" : "text-rose-300")}>{formatPercent(row.winRate)}</td>
                  <td className={cn("px-2 py-2", row.avgPnl >= 0 ? "text-emerald-300" : "text-rose-300")}>{formatCurrency(row.avgPnl)}</td>
                  <td className={cn("px-2 py-2", row.totalPnl >= 0 ? "text-emerald-300" : "text-rose-300")}>{formatCurrency(row.totalPnl)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
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
      return { date, open: prev, high: Math.max(prev, close), low: Math.min(prev, close), close };
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
    const f = backtest?.forwardSimulation?.markers ?? [];
    return [...p, ...b, ...f] as ChartMarker[];
  }, [prediction?.tradeMarkers, backtest?.annotations, backtest?.forwardSimulation?.markers]);

  const applyPreset = (preset: "daily" | "intraday") => {
    if (preset === "intraday") {
      setPredictionParams((p) => ({
        ...p,
        dataInterval: "1h",
        dataPeriod: "730d",
        modelVariant: "auto_intraday",
        maxLong: 1.4,
        maxShort: 0.2,
        horizon: 12,
        daysOnChart: 240,
      }));
      setBacktestParams((b) => ({
        ...b,
        dataInterval: "1h",
        maxLong: 1.4,
        maxShort: 0.2,
        commission: 0.5,
        slippage: 0.2,
        minPositionChange: 0.05,
        flatAtDayEnd: true,
      }));
      return;
    }
    setPredictionParams((p) => ({
      ...p,
      dataInterval: "1d",
      dataPeriod: "10y",
      modelVariant: "auto",
      maxLong: 1.6,
      maxShort: 0.25,
      horizon: 10,
      daysOnChart: 180,
    }));
    setBacktestParams((b) => ({
      ...b,
      dataInterval: "1d",
      maxLong: 1.6,
      maxShort: 0.25,
      commission: 0.5,
      slippage: 0.2,
      minPositionChange: 0.0,
      flatAtDayEnd: false,
    }));
  };

  const handlePredict = async () => {
    if (!predictionParams.symbol.trim()) {
      toast.error("Please enter a symbol");
      return;
    }
    await predictMutation.mutateAsync({
      ...predictionParams,
      maxLong: backtestParams.maxLong,
      maxShort: backtestParams.maxShort,
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
      params: {
        ...backtestParams,
        dataInterval: prediction.metadata?.dataInterval ?? predictionParams.dataInterval ?? "1d",
        annualizationFactor: backtestParams.annualizationFactor ?? prediction.metadata?.annualizationFactor,
        flatAtDayEnd: backtestParams.flatAtDayEnd ?? prediction.metadata?.flatAtDayEnd,
      },
    });
  };

  const metrics = backtest?.metrics;
  const intraday = Boolean(prediction?.metadata?.isIntraday || predictionParams.dataInterval !== "1d");
  const actionSummary = useMemo(() => {
    const rows = backtest?.tradeLog ?? [];
    const out = {
      total: rows.length,
      longBuy: 0,
      longSell: 0,
      shortSell: 0,
      shortCover: 0,
    };
    for (const trade of rows) {
      const action = String(trade.action ?? "").toUpperCase();
      if (action === "BUY") out.longBuy += 1;
      else if (action === "SELL" || action === "EOD_FLAT_SELL" || action === "EOD_REDUCE_SELL") out.longSell += 1;
      else if (action === "SHORT" || action === "SELL_SHORT") out.shortSell += 1;
      else if (action === "COVER" || action === "COVER_BUY" || action === "EOD_FLAT_COVER" || action === "EOD_REDUCE_COVER") out.shortCover += 1;
    }
    return out;
  }, [backtest?.tradeLog]);

  return (
    <div className="relative overflow-hidden bg-gradient-to-b from-zinc-950 to-black">
      <div className="pointer-events-none absolute -left-36 -top-36 h-80 w-80 rounded-full bg-cyan-500/10 blur-3xl" />
      <div className="pointer-events-none absolute right-0 top-24 h-96 w-96 rounded-full bg-amber-500/10 blur-3xl" />

      <section className="mb-6 rounded-3xl border border-zinc-700/70 bg-[radial-gradient(1200px_350px_at_20%_0%,rgba(34,211,238,0.12),transparent_60%),radial-gradient(900px_320px_at_90%_0%,rgba(250,204,21,0.08),transparent_60%),linear-gradient(180deg,rgba(24,24,27,0.95),rgba(9,9,11,0.95))] p-6 shadow-[0_18px_70px_rgba(0,0,0,0.55)]">
        <p className="text-xs uppercase tracking-[0.24em] text-amber-400">Ralph Loop Command Deck</p>
        <h1 className="mt-2 font-mono text-3xl font-semibold text-zinc-100">AI Strategy Cockpit</h1>
        <p className="mt-2 max-w-4xl text-sm text-zinc-300">
          Full control over model variant, interval, execution constraints, and risk diagnostics. Train, backtest, forward-simulate, and inspect
          alpha decay or execution drag from one surface.
        </p>
      </section>

      <div className="grid gap-6 xl:grid-cols-[390px,1fr]">
        <aside className="space-y-4 rounded-2xl border border-zinc-800 bg-zinc-950/75 p-4">
          <div className="grid grid-cols-2 gap-2">
            <button
              type="button"
              onClick={() => applyPreset("daily")}
              className="rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-xs font-semibold text-zinc-200 transition hover:border-cyan-400"
            >
              Daily Preset
            </button>
            <button
              type="button"
              onClick={() => applyPreset("intraday")}
              className="rounded-lg border border-zinc-700 bg-zinc-900 px-3 py-2 text-xs font-semibold text-zinc-200 transition hover:border-amber-400"
            >
              Intraday Preset
            </button>
          </div>

          <div className="space-y-2 rounded-xl border border-zinc-800 bg-zinc-900/60 p-3">
            <p className="text-xs uppercase tracking-[0.18em] text-zinc-500">Symbol + Data</p>
            <input
              className="w-full rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-cyan-400"
              value={predictionParams.symbol}
              onChange={(e) => setPredictionParams((p) => ({ ...p, symbol: e.target.value.toUpperCase() }))}
              placeholder="AAPL"
            />
            <div className="grid grid-cols-2 gap-2">
              <input
                className="rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-cyan-400"
                type="number"
                value={predictionParams.horizon}
                min={1}
                onChange={(e) => setPredictionParams((p) => ({ ...p, horizon: Number(e.target.value) || 1 }))}
                placeholder="Horizon"
              />
              <input
                className="rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-cyan-400"
                type="number"
                value={predictionParams.daysOnChart}
                min={30}
                onChange={(e) => setPredictionParams((p) => ({ ...p, daysOnChart: Number(e.target.value) || 120 }))}
                placeholder="Chart Bars"
              />
            </div>
            <div className="grid grid-cols-3 gap-2">
              <select
                className="rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-xs text-zinc-100 outline-none focus:border-cyan-400"
                value={predictionParams.dataInterval ?? "1d"}
                onChange={(e) => {
                  const v = e.target.value;
                  setPredictionParams((p) => ({
                    ...p,
                    dataInterval: v,
                    dataPeriod: v === "1d" ? "10y" : "730d",
                  }));
                  setBacktestParams((b) => ({ ...b, dataInterval: v }));
                }}
              >
                <option value="1d">1d</option>
                <option value="1h">1h</option>
                <option value="30m">30m</option>
                <option value="15m">15m</option>
              </select>
              <select
                className="rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-xs text-zinc-100 outline-none focus:border-cyan-400"
                value={predictionParams.dataPeriod ?? "10y"}
                onChange={(e) => setPredictionParams((p) => ({ ...p, dataPeriod: e.target.value }))}
              >
                <option value="10y">10y</option>
                <option value="5y">5y</option>
                <option value="730d">730d</option>
                <option value="365d">365d</option>
                <option value="180d">180d</option>
                <option value="60d">60d</option>
              </select>
              <input
                className="rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-xs text-zinc-100 outline-none focus:border-cyan-400"
                value={predictionParams.modelVariant ?? "auto"}
                onChange={(e) => setPredictionParams((p) => ({ ...p, modelVariant: e.target.value || "auto" }))}
                placeholder="Variant"
              />
            </div>
          </div>

          <div className="space-y-2 rounded-xl border border-zinc-800 bg-zinc-900/60 p-3">
            <p className="text-xs uppercase tracking-[0.18em] text-zinc-500">Model + Fusion</p>
            <select
              className="w-full rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-cyan-400"
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
                className="rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-cyan-400"
                type="number"
                step="0.01"
                value={fusionSettings.buyThreshold}
                onChange={(e) => setFusionSettings((f) => ({ ...f, buyThreshold: Number(e.target.value) || 0.3 }))}
                placeholder="Buy Thr"
              />
              <input
                className="rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-cyan-400"
                type="number"
                step="0.01"
                value={fusionSettings.sellThreshold}
                onChange={(e) => setFusionSettings((f) => ({ ...f, sellThreshold: Number(e.target.value) || 0.45 }))}
                placeholder="Sell Thr"
              />
            </div>
          </div>

          <div className="space-y-2 rounded-xl border border-zinc-800 bg-zinc-900/60 p-3">
            <p className="text-xs uppercase tracking-[0.18em] text-zinc-500">Execution + Risk Controls</p>
            <div className="grid grid-cols-2 gap-2">
              <input
                className="rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-cyan-400"
                type="number"
                value={backtestParams.initialCapital}
                onChange={(e) => setBacktestParams((b) => ({ ...b, initialCapital: Number(e.target.value) || 10_000 }))}
                placeholder="Capital"
              />
              <input
                className="rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-cyan-400"
                type="number"
                step="0.1"
                value={backtestParams.maxLong}
                onChange={(e) => setBacktestParams((b) => ({ ...b, maxLong: Number(e.target.value) || 1.6 }))}
                placeholder="Max Long"
              />
              <input
                className="rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-cyan-400"
                type="number"
                step="0.05"
                value={backtestParams.maxShort}
                onChange={(e) => setBacktestParams((b) => ({ ...b, maxShort: Number(e.target.value) || 0.25 }))}
                placeholder="Max Short"
              />
              <input
                className="rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-cyan-400"
                type="number"
                step="0.1"
                value={backtestParams.commission}
                onChange={(e) => setBacktestParams((b) => ({ ...b, commission: Number(e.target.value) || 0.5 }))}
                placeholder="Commission (bps)"
              />
              <input
                className="rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-cyan-400"
                type="number"
                step="0.1"
                value={backtestParams.slippage}
                onChange={(e) => setBacktestParams((b) => ({ ...b, slippage: Number(e.target.value) || 0.2 }))}
                placeholder="Slippage (bps)"
              />
              <input
                className="rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-cyan-400"
                type="number"
                step="0.01"
                value={backtestParams.minPositionChange ?? 0}
                onChange={(e) => setBacktestParams((b) => ({ ...b, minPositionChange: Number(e.target.value) || 0 }))}
                placeholder="Min Pos Delta"
              />
              <input
                className="col-span-2 rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-cyan-400"
                type="number"
                step="1"
                value={backtestParams.annualizationFactor ?? ""}
                onChange={(e) => {
                  const v = e.target.value.trim();
                  setBacktestParams((b) => ({ ...b, annualizationFactor: v === "" ? undefined : Number(v) }));
                }}
                placeholder="Annualization Override (optional)"
              />
            </div>
            <div className="grid grid-cols-2 gap-2 pt-1 text-sm text-zinc-300">
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={backtestParams.enableForwardSim}
                  onChange={(e) => setBacktestParams((b) => ({ ...b, enableForwardSim: e.target.checked }))}
                />
                Forward Sim
              </label>
              <label className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={Boolean(backtestParams.flatAtDayEnd ?? prediction?.metadata?.flatAtDayEnd)}
                  onChange={(e) => setBacktestParams((b) => ({ ...b, flatAtDayEnd: e.target.checked }))}
                />
                Flat At Day End
              </label>
            </div>
          </div>

          <div className="space-y-2 pt-2">
            <Button className="w-full yellow-btn" onClick={handlePredict} disabled={predictMutation.isPending}>
              {predictMutation.isPending ? "Running Prediction..." : "Run Prediction"}
            </Button>
            <Button
              className="w-full border border-zinc-700 bg-zinc-900 text-zinc-100 hover:bg-zinc-800"
              onClick={handleBacktest}
              disabled={backtestMutation.isPending || !prediction}
            >
              {backtestMutation.isPending ? "Running Backtest..." : "Run Backtest + Forward Sim"}
            </Button>
          </div>

          <div className="rounded-xl border border-zinc-800 bg-black/35 p-3 text-xs text-zinc-400">
            <p className="font-semibold text-zinc-200">Live Execution Contract</p>
            <p className="mt-1">Inventory-aware engine: BUY/SELL/SHORT/COVER with borrow, slippage, no impossible sells, and optional EOD flattening.</p>
            <p className="mt-2 text-[11px] text-zinc-500">
              Python PNG batch: run `python run_backtest.py --symbol AAPL --model-variant auto_intraday --data-interval 1h`.
            </p>
          </div>
        </aside>

        <main className="space-y-6">
          <section className="rounded-2xl border border-zinc-800 bg-zinc-950/70 p-4">
            <div className="mb-3 flex flex-wrap items-end justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.18em] text-zinc-500">Loaded</p>
                <p className="font-mono text-lg text-zinc-100">
                  {prediction?.symbol ?? predictionParams.symbol} {selectedModel ? `• ${selectedModel.id}` : ""}
                </p>
              </div>
              <div className="text-sm text-zinc-300">Quality gate: {String(prediction?.metadata?.modelQualityGatePassed ?? false)}</div>
              <div className="text-xs text-zinc-400">
                Interval: {prediction?.metadata?.dataInterval ?? predictionParams.dataInterval ?? "1d"} | Variant:{" "}
                {prediction?.metadata?.modelVariant ?? predictionParams.modelVariant ?? "auto"}
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
            <div className="mt-3 grid gap-2 rounded-lg border border-zinc-800 bg-black/25 p-3 text-xs text-zinc-300 md:grid-cols-6">
              <div className="font-semibold text-zinc-100">Trade Markers</div>
              <div><span className="font-mono text-emerald-300">BY</span> = Long Buy</div>
              <div><span className="font-mono text-red-300">SL</span> = Long Sell</div>
              <div><span className="font-mono text-orange-300">SH</span> = Short Entry</div>
              <div><span className="font-mono text-sky-300">CV</span> = Short Cover</div>
              <div className="text-zinc-400">Total: {actionSummary.total}</div>
            </div>
          </section>

          <section className="grid gap-3 md:grid-cols-8">
            <MetricCard label="Cumulative Return" value={metrics ? formatPercent(metrics.cumulativeReturn) : "-"} tone={metrics && metrics.cumulativeReturn >= 0 ? "good" : "bad"} />
            <MetricCard label="Sharpe" value={metrics ? metrics.sharpeRatio.toFixed(2) : "-"} tone={metrics && metrics.sharpeRatio >= 0 ? "good" : "bad"} />
            <MetricCard label="Max DD" value={metrics ? formatPercent(metrics.maxDrawdown) : "-"} tone={metrics && metrics.maxDrawdown < -0.2 ? "bad" : "neutral"} />
            <MetricCard label="Win Rate" value={metrics ? formatPercent(metrics.winRate) : "-"} tone={metrics && metrics.winRate >= 0.5 ? "good" : "bad"} />
            <MetricCard label="Total Trades" value={metrics ? String(metrics.totalTrades) : "-"} />
            <MetricCard label="Directional Acc" value={metrics ? formatPercent(metrics.directionalAccuracy) : "-"} />
            <MetricCard label="Tx Costs" value={metrics ? formatCurrency(metrics.transactionCosts ?? 0) : "-"} tone={metrics && (metrics.transactionCosts ?? 0) > 0 ? "bad" : "neutral"} />
            <MetricCard label="Borrow Fee" value={metrics ? formatCurrency(metrics.borrowFee ?? 0) : "-"} tone={metrics && (metrics.borrowFee ?? 0) > 0 ? "bad" : "neutral"} />
          </section>

          <section className="rounded-2xl border border-zinc-800 bg-zinc-950/70 p-4">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <p className="text-xs uppercase tracking-[0.18em] text-zinc-500">Portfolio vs Stock</p>
              <p className="text-xs text-zinc-400">
                Portfolio Equity (yellow) vs Stock Buy&Hold (blue) with trade dots on portfolio curve
              </p>
            </div>
            <EquityLineChart backtest={backtest ?? undefined} tradeMarkers={backtest?.annotations} />
          </section>

          <DiagnosticsPanel diagnostics={backtest?.diagnostics} intraday={intraday} />

          <ForwardSimulationPanel backtest={backtest} />

          <section className="rounded-2xl border border-zinc-800 bg-zinc-950/70 p-4">
            <p className="mb-3 text-xs uppercase tracking-[0.18em] text-zinc-500">Trade Log</p>
            {!backtest?.tradeLog?.length ? (
              <p className="text-sm text-zinc-400">Run backtest to populate trade details.</p>
            ) : (
              <div className="max-h-96 overflow-auto">
                <table className="w-full text-left text-xs text-zinc-300">
                  <thead className="sticky top-0 bg-zinc-900">
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
                      <tr key={trade.id} className="border-t border-zinc-800">
                        <td className="px-2 py-2">{trade.date}</td>
                        <td className={cn("px-2 py-2 font-semibold", ["BUY", "COVER", "COVER_BUY"].includes(trade.action) ? "text-emerald-300" : "text-rose-300")}>
                          {trade.action}
                        </td>
                        <td className="px-2 py-2">{formatCurrency(trade.price)}</td>
                        <td className="px-2 py-2">{trade.shares.toFixed(2)}</td>
                        <td className="px-2 py-2">{trade.position.toFixed(2)}</td>
                        <td className={cn("px-2 py-2", trade.pnl >= 0 ? "text-emerald-300" : "text-rose-300")}>{formatCurrency(trade.pnl)}</td>
                        <td className="px-2 py-2 text-zinc-400">{trade.explanation?.notes ?? "-"}</td>
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
