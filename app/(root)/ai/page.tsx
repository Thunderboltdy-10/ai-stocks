"use client";

import { useEffect, useMemo, useRef, useState } from "react";
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
  BenchmarkRun,
  FusionSettings,
  JobEvent,
  ModelMeta,
  PredictionParams,
  PredictionResult,
  ResearchRunSummary,
  TrainingJob,
} from "@/types/ai";
import {
  cancelTrainingJob,
  listBenchmarkRuns,
  listModels,
  listResearchRuns,
  runBacktest,
  runPrediction,
  startTraining,
  subscribeToTrainingEvents,
} from "@/lib/api-client";
import { BenchmarkLabPanel } from "@/components/ai/BenchmarkLabPanel";
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
  forwardWindow: 20,
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

function StatusPill({
  label,
  tone = "neutral",
}: {
  label: string;
  tone?: "neutral" | "good" | "bad" | "warn";
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full border px-3 py-1 text-[11px] font-semibold uppercase tracking-[0.16em]",
        tone === "good" && "border-emerald-500/40 bg-emerald-500/10 text-emerald-200",
        tone === "bad" && "border-rose-500/40 bg-rose-500/10 text-rose-200",
        tone === "warn" && "border-amber-500/40 bg-amber-500/10 text-amber-200",
        tone === "neutral" && "border-zinc-700 bg-zinc-900/80 text-zinc-300"
      )}
    >
      {label}
    </span>
  );
}

function SectionNav() {
  const items = [
    { href: "#controls", label: "Controls" },
    { href: "#chart", label: "Signal Deck" },
    { href: "#equity", label: "Equity" },
    { href: "#diagnostics", label: "Diagnostics" },
    { href: "#trades", label: "Trades" },
  ];

  return (
    <nav className="sticky top-4 z-20 flex flex-wrap gap-2 rounded-2xl border border-zinc-800/80 bg-zinc-950/85 p-2 backdrop-blur">
      {items.map((item) => (
        <a
          key={item.href}
          href={item.href}
          className="rounded-full border border-zinc-800 bg-black/30 px-3 py-1.5 text-xs font-medium text-zinc-300 transition hover:border-cyan-400 hover:text-white"
        >
          {item.label}
        </a>
      ))}
    </nav>
  );
}

function ForwardSimulationPanel({ backtest }: { backtest: BacktestResult | null }) {
  const sim = backtest?.forwardSimulation;
  if (!sim) {
    return (
      <div className="rounded-2xl border border-zinc-800 bg-zinc-950/70 p-5 text-sm text-zinc-400">
        Realized holdout simulation appears here after backtest.
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
      <div className="flex flex-wrap items-center justify-between gap-3">
        <p className="text-xs uppercase tracking-[0.18em] text-zinc-500">Realized Forward Holdout</p>
        <p className="text-xs text-zinc-400">
          {sim.windowStart && sim.windowEnd ? `${sim.windowStart} -> ${sim.windowEnd}` : sim.source ?? "historical_holdout"}
        </p>
      </div>
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

function ResearchWorkflowPanel({
  currentJob,
  logs,
  runs,
  isStarting,
  onStartSingle,
  onStartDaily,
  onStartIntraday,
  onStartFull,
  onCancel,
}: {
  currentJob: TrainingJob | null;
  logs: string[];
  runs?: ResearchRunSummary[];
  isStarting: boolean;
  onStartSingle: () => void;
  onStartDaily: () => void;
  onStartIntraday: () => void;
  onStartFull: () => void;
  onCancel: () => void;
}) {
  const running = currentJob?.status === "queued" || currentJob?.status === "running";
  const gateMetric = (
    payload: Record<string, unknown> | undefined,
    key: "mean_alpha" | "median_alpha"
  ): number | null => {
    const metrics = payload?.holdout_metrics;
    if (!metrics || typeof metrics !== "object") return null;
    const raw = (metrics as Record<string, unknown>)[key];
    return typeof raw === "number" && Number.isFinite(raw) ? raw : null;
  };
  const gateGap = (payload: Record<string, unknown> | undefined): number | null => {
    const raw = payload?.generalization_gap;
    return typeof raw === "number" && Number.isFinite(raw) ? raw : null;
  };

  return (
    <section className="rounded-2xl border border-zinc-800 bg-[linear-gradient(180deg,rgba(20,20,24,0.96),rgba(10,10,12,0.96))] p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.18em] text-zinc-500">Research Ops</p>
          <h2 className="mt-1 text-lg font-semibold text-zinc-100">Train the safe path first</h2>
          <p className="mt-1 text-sm text-zinc-400">
            Run diversified daily and intraday sweeps, rebuild auto-routing registries, then enforce holdout gates before trusting the overlay.
          </p>
        </div>
        <StatusPill
          label={running ? currentJob?.workflow?.replaceAll("_", " ") ?? "running" : "idle"}
          tone={running ? "warn" : "neutral"}
        />
      </div>

      <div className="mt-4 grid gap-3 md:grid-cols-4">
        <button
          type="button"
          onClick={onStartSingle}
          disabled={isStarting || running}
          className="rounded-xl border border-zinc-700 bg-black/25 px-4 py-3 text-left text-sm text-zinc-100 transition hover:border-cyan-400 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <p className="font-semibold">Refresh Current Symbol</p>
          <p className="mt-1 text-xs text-zinc-400">Retrain the active symbol/timeframe with the stricter GBM path.</p>
        </button>
        <button
          type="button"
          onClick={onStartDaily}
          disabled={isStarting || running}
          className="rounded-xl border border-zinc-700 bg-black/25 px-4 py-3 text-left text-sm text-zinc-100 transition hover:border-cyan-400 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <p className="font-semibold">Daily 15-Symbol Gate</p>
          <p className="mt-1 text-xs text-zinc-400">Train daily variants, rebuild registries, then score core vs holdout alpha.</p>
        </button>
        <button
          type="button"
          onClick={onStartIntraday}
          disabled={isStarting || running}
          className="rounded-xl border border-zinc-700 bg-black/25 px-4 py-3 text-left text-sm text-zinc-100 transition hover:border-amber-400 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <p className="font-semibold">Intraday 10-Symbol Gate</p>
          <p className="mt-1 text-xs text-zinc-400">Retrain hourly variants across the intraday basket and re-score short-window stability.</p>
        </button>
        <button
          type="button"
          onClick={onStartFull}
          disabled={isStarting || running}
          className="rounded-xl border border-zinc-700 bg-black/25 px-4 py-3 text-left text-sm text-zinc-100 transition hover:border-emerald-400 disabled:cursor-not-allowed disabled:opacity-50"
        >
          <p className="font-semibold">Full Research Sweep</p>
          <p className="mt-1 text-xs text-zinc-400">Run both workflows end-to-end and persist a single machine-readable summary.</p>
        </button>
      </div>

      <div className="mt-4 grid gap-3 lg:grid-cols-[1.2fr,0.8fr]">
        <div className="rounded-xl border border-zinc-800 bg-black/20 p-3">
          <div className="flex items-center justify-between gap-3">
            <div>
              <p className="text-[11px] uppercase tracking-[0.18em] text-zinc-500">Active Job</p>
              <p className="mt-1 text-sm text-zinc-100">
                {currentJob ? `${currentJob.workflow ?? "single"} • ${currentJob.status}` : "No job running"}
              </p>
            </div>
            {running ? (
              <button
                type="button"
                onClick={onCancel}
                className="rounded-lg border border-rose-500/40 bg-rose-500/10 px-3 py-2 text-xs font-semibold text-rose-200 transition hover:border-rose-400"
              >
                Cancel Job
              </button>
            ) : null}
          </div>

          <div className="mt-3 h-2 overflow-hidden rounded-full bg-zinc-800">
            <div
              className="h-full rounded-full bg-gradient-to-r from-cyan-400 via-amber-300 to-emerald-400 transition-all"
              style={{ width: `${Math.round((currentJob?.progress ?? 0) * 100)}%` }}
            />
          </div>

          <div className="mt-3 grid gap-3 md:grid-cols-3">
            <MetricCard label="Progress" value={`${Math.round((currentJob?.progress ?? 0) * 100)}%`} />
            <MetricCard label="Current Step" value={currentJob?.currentStep ?? "-"} />
            <MetricCard label="Job ID" value={currentJob?.id ?? "-"} />
          </div>

          <div className="mt-3 rounded-xl border border-zinc-800 bg-zinc-950/80 p-3">
            <p className="text-[11px] uppercase tracking-[0.18em] text-zinc-500">Event Log</p>
            <div className="mt-2 max-h-44 space-y-1 overflow-auto font-mono text-[11px] text-zinc-300">
              {logs.length ? logs.map((line, index) => <p key={`${index}-${line}`}>{line}</p>) : <p className="text-zinc-500">No events yet.</p>}
            </div>
          </div>
        </div>

        <div className="rounded-xl border border-zinc-800 bg-black/20 p-3">
          <p className="text-[11px] uppercase tracking-[0.18em] text-zinc-500">Latest Workflow Runs</p>
          <div className="mt-3 space-y-3">
            {(runs ?? []).length ? (
              runs?.map((run) => (
                <div key={run.id} className="rounded-xl border border-zinc-800 bg-zinc-950/80 p-3">
                  <div className="flex items-center justify-between gap-2">
                    <p className="text-sm font-semibold text-zinc-100">{run.workflow.replaceAll("_", " ")}</p>
                    <StatusPill label={run.status} tone={run.status === "completed" ? "good" : run.status === "failed" ? "bad" : "warn"} />
                  </div>
                  <p className="mt-2 text-xs text-zinc-400">{new Date(run.startedAt).toLocaleString()}</p>
                  <div className="mt-3 flex flex-wrap gap-2">
                    <StatusPill label={`Daily ${run.dailyGatePassed ? "pass" : "review"}`} tone={run.dailyGatePassed ? "good" : "warn"} />
                    <StatusPill label={`Intraday ${run.intradayGatePassed ? "pass" : "review"}`} tone={run.intradayGatePassed ? "good" : "warn"} />
                  </div>
                  <div className="mt-3 grid gap-2 text-[11px] text-zinc-400">
                    <p>
                      Daily holdout alpha{" "}
                      <span className="font-semibold text-zinc-200">
                        {gateMetric(run.daily, "mean_alpha") === null ? "-" : formatPercent(gateMetric(run.daily, "mean_alpha") ?? 0)}
                      </span>
                      {" · "}
                      gap{" "}
                      <span className="font-semibold text-zinc-200">
                        {gateGap(run.daily) === null ? "-" : formatPercent(gateGap(run.daily) ?? 0)}
                      </span>
                    </p>
                    <p>
                      Intraday holdout alpha{" "}
                      <span className="font-semibold text-zinc-200">
                        {gateMetric(run.intraday, "mean_alpha") === null ? "-" : formatPercent(gateMetric(run.intraday, "mean_alpha") ?? 0)}
                      </span>
                      {" · "}
                      gap{" "}
                      <span className="font-semibold text-zinc-200">
                        {gateGap(run.intraday) === null ? "-" : formatPercent(gateGap(run.intraday) ?? 0)}
                      </span>
                    </p>
                  </div>
                </div>
              ))
            ) : (
              <div className="rounded-xl border border-zinc-800 bg-zinc-950/80 p-3 text-sm text-zinc-400">
                No research workflow summaries yet.
              </div>
            )}
          </div>
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
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [backtest, setBacktest] = useState<BacktestResult | null>(null);
  const [trainingJob, setTrainingJob] = useState<TrainingJob | null>(null);
  const [trainingLogs, setTrainingLogs] = useState<string[]>([]);
  const [startingWorkflow, setStartingWorkflow] = useState(false);
  const trainingStreamRef = useRef<EventSource | null>(null);

  useEffect(() => {
    return () => trainingStreamRef.current?.close();
  }, []);

  const modelsQuery = useQuery({
    queryKey: ["models"],
    queryFn: listModels,
    refetchInterval: 60_000,
  });

  const researchRunsQuery = useQuery<ResearchRunSummary[]>({
    queryKey: ["research-runs"],
    queryFn: () => listResearchRuns(4),
    refetchInterval: 60_000,
  });

  const benchmarkRunsQuery = useQuery<BenchmarkRun[]>({
    queryKey: ["benchmark-runs"],
    queryFn: () => listBenchmarkRuns(3),
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

  const appendTrainingLog = (line: string) => {
    setTrainingLogs((prev) => [...prev, line].slice(-160));
  };

  const beginTrainingStream = (jobId: string, initialJob: Partial<TrainingJob>) => {
    trainingStreamRef.current?.close();
    const stream = subscribeToTrainingEvents(jobId);
    trainingStreamRef.current = stream;
    setTrainingJob({
      id: jobId,
      symbol: initialJob.symbol ?? predictionParams.symbol,
      status: "queued",
      progress: 0,
      currentEpoch: 0,
      totalEpochs: 100,
      startedAt: new Date().toISOString(),
      workflow: initialJob.workflow,
      currentStep: "",
    });

    stream.onmessage = (event) => {
      let payload: JobEvent | null = null;
      try {
        payload = JSON.parse(event.data) as JobEvent;
      } catch {
        return;
      }
      if (!payload || payload.type === "heartbeat") return;

      if (payload.message) {
        appendTrainingLog(payload.message);
      }
      if (payload.type === "log" && payload.message) {
        appendTrainingLog(payload.message);
      }
      if (payload.type === "stage") {
        appendTrainingLog(`[${payload.step ?? "stage"}] ${payload.message ?? "running"}`);
      }

      setTrainingJob((prev) => {
        if (!prev) return prev;
        if (payload.type === "completed") {
          return {
            ...prev,
            status: "completed",
            progress: 1,
            completedAt: new Date().toISOString(),
          };
        }
        if (payload.type === "failed") {
          return {
            ...prev,
            status: "failed",
            error: payload.error ?? payload.message,
            completedAt: new Date().toISOString(),
          };
        }
        return {
          ...prev,
          status: prev.status === "queued" ? "running" : prev.status,
          progress: payload.progress ?? prev.progress,
          currentStep: payload.step ?? prev.currentStep,
        };
      });

      if (payload.type === "completed") {
        toast.success("Research workflow finished");
        trainingStreamRef.current?.close();
        researchRunsQuery.refetch();
        modelsQuery.refetch();
      } else if (payload.type === "failed") {
        toast.error(payload.error ?? "Training failed");
        trainingStreamRef.current?.close();
        researchRunsQuery.refetch();
      }
    };

    stream.onerror = () => {
      trainingStreamRef.current?.close();
    };
  };

  const startWorkflow = async (workflow: "single" | "daily_research" | "intraday_research" | "full_research") => {
    setStartingWorkflow(true);
    setTrainingLogs([]);
    try {
      const currentInterval = predictionParams.dataInterval ?? "1d";
      const currentPeriod = predictionParams.dataPeriod ?? "10y";
      const singleIntraday = currentInterval !== "1d";
      const featureProfiles =
        workflow === "intraday_research"
          ? ["compact", "trend"]
          : workflow === "single" && singleIntraday
            ? ["compact", "trend"]
            : ["full", "compact", "trend"];
      const targetHorizons =
        workflow === "intraday_research"
          ? [1, 3]
          : workflow === "single" && singleIntraday
            ? [1]
            : [1, 3];
      const symbolSet =
        workflow === "intraday_research"
          ? "intraday10"
          : workflow === "daily_research"
            ? "daily15"
            : singleIntraday
              ? "intraday10"
              : "daily15";
      const response = await startTraining({
        symbol: predictionParams.symbol,
        epochs: 50,
        batchSize: 512,
        loss: "balanced",
        sequenceLength: 90,
        featureToggles: {},
        ensembleSize: 1,
        baseSeed: 42,
        modelType: "gbm",
        workflow,
        nTrials: workflow === "single" ? 12 : 8,
        maxFeatures: 50,
        targetHorizons,
        featureProfiles,
        featureSelectionModes: ["shap_diverse", "shap_ranked"],
        symbolSet,
        dailySymbolSet: "daily15",
        intradaySymbolSet: "intraday10",
        dataInterval: currentInterval,
        dataPeriod: currentPeriod,
        dailyInterval: "1d",
        intradayInterval: "1h",
        dailyPeriod: "max",
        intradayPeriod: "730d",
        useLgb: false,
        overwrite: true,
      });
      appendTrainingLog(`Job ${response.jobId} created for ${workflow}`);
      beginTrainingStream(response.jobId, {
        symbol: predictionParams.symbol,
        workflow,
      });
      toast.success(`Started ${workflow.replaceAll("_", " ")}`);
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to start training");
    } finally {
      setStartingWorkflow(false);
    }
  };

  const handleCancelTraining = async () => {
    if (!trainingJob) return;
    try {
      await cancelTrainingJob(trainingJob.id);
      trainingStreamRef.current?.close();
      setTrainingJob((prev) => (prev ? { ...prev, status: "cancelled", completedAt: new Date().toISOString() } : null));
      appendTrainingLog("Job cancelled by user");
      toast.info("Training cancelled");
    } catch (error) {
      toast.error(error instanceof Error ? error.message : "Failed to cancel training");
    }
  };

  const selectedModel = useMemo(() => {
    return modelsQuery.data?.find((m) => m.id === selectedModelId);
  }, [modelsQuery.data, selectedModelId]);

  const activeQualityGate = prediction?.metadata?.modelQualityGatePassed ?? selectedModel?.qualityGatePassed ?? false;
  const activeQualityScore = prediction?.metadata?.modelQualityScore ?? selectedModel?.qualityScore ?? null;
  const activeQualityReasons = prediction?.metadata?.modelQualityReasons ?? selectedModel?.qualityReasons ?? [];
  const activeHoldoutSource = prediction?.metadata?.holdoutMetricSource ?? selectedModel?.holdoutMetricSource;
  const activeDispersionRatio = prediction?.metadata?.holdoutPredTargetStdRatio ?? selectedModel?.holdoutPredTargetStdRatio ?? null;
  const activeFeatureProfile = prediction?.metadata?.featureProfile ?? selectedModel?.featureProfile ?? null;
  const activeSelectionMode = prediction?.metadata?.featureSelectionMode ?? selectedModel?.featureSelectionMode ?? null;
  const activeTargetHorizon = prediction?.metadata?.targetHorizonDays ?? selectedModel?.targetHorizonDays ?? null;
  const activeVariant = prediction?.metadata?.modelVariant ?? selectedModel?.modelVariant ?? predictionParams.modelVariant ?? "auto";
  const backendHealthy = modelsQuery.isSuccess && (modelsQuery.data?.length ?? 0) > 0;
  const activeHealthTone: "good" | "warn" | "bad" =
    activeQualityGate ? "good" : activeQualityReasons.length ? "warn" : "bad";
  const activeHealthLabel = activeQualityGate
    ? "ML Gate Active"
    : activeQualityReasons.length
      ? "Regime Fallback"
      : "Awaiting Model";

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
    return [...p, ...b, ...f].sort(
      (a, b) => new Date(a.date).getTime() - new Date(b.date).getTime()
    ) as ChartMarker[];
  }, [prediction?.tradeMarkers, backtest?.annotations, backtest?.forwardSimulation?.markers]);

  const forecastBandSummary = useMemo(() => {
    const forecast = prediction?.forecast;
    if (!forecast?.prices?.length) return null;
    const lastIdx = forecast.prices.length - 1;
    const base = forecast.prices[lastIdx] ?? 0;
    const lower = forecast.lowerBand?.[lastIdx];
    const upper = forecast.upperBand?.[lastIdx];
    return {
      base,
      lower: lower ?? base,
      upper: upper ?? base,
    };
  }, [prediction?.forecast]);

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
        forwardWindow: 24,
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
      forwardWindow: 20,
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
        <div className="flex flex-wrap items-start justify-between gap-4">
          <div>
            <p className="text-xs uppercase tracking-[0.24em] text-amber-400">Ralph Loop Command Deck</p>
            <h1 className="mt-2 font-mono text-3xl font-semibold text-zinc-100">AI Strategy Cockpit</h1>
            <p className="mt-2 max-w-4xl text-sm text-zinc-300">
              Run the full loop from model health to forecast, backtest, and execution diagnostics without digging through scripts.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <StatusPill label={activeHealthLabel} tone={activeHealthTone} />
            <StatusPill label={intraday ? "Intraday Profile" : "Daily Profile"} tone="neutral" />
            <StatusPill label={backendHealthy ? "Backend Live" : "Backend Needs Attention"} tone={backendHealthy ? "good" : "bad"} />
          </div>
        </div>
        <div className="mt-5 grid gap-3 md:grid-cols-6">
          <MetricCard label="Selected Symbol" value={prediction?.symbol ?? predictionParams.symbol} />
          <MetricCard label="Active Variant" value={activeVariant} />
          <MetricCard label="Quality Score" value={activeQualityScore === null ? "-" : activeQualityScore.toFixed(2)} tone={activeQualityGate ? "good" : "bad"} />
          <MetricCard label="Holdout Source" value={activeHoldoutSource ?? "-"} />
          <MetricCard label="Dispersion Ratio" value={activeDispersionRatio === null ? "-" : activeDispersionRatio.toFixed(2)} tone={activeDispersionRatio !== null && activeDispersionRatio < 0.18 ? "bad" : "neutral"} />
          <MetricCard label="Feature Profile" value={activeFeatureProfile ?? "-"} />
        </div>
      </section>

      <div className="mb-6">
        <ResearchWorkflowPanel
          currentJob={trainingJob}
          logs={trainingLogs}
          runs={researchRunsQuery.data}
          isStarting={startingWorkflow}
          onStartSingle={() => startWorkflow("single")}
          onStartDaily={() => startWorkflow("daily_research")}
          onStartIntraday={() => startWorkflow("intraday_research")}
          onStartFull={() => startWorkflow("full_research")}
          onCancel={handleCancelTraining}
        />
      </div>

      <div className="mb-6">
        <BenchmarkLabPanel runs={benchmarkRunsQuery.data} />
      </div>

      <SectionNav />

      <div className="grid gap-6 xl:grid-cols-[390px,1fr]">
        <aside id="controls" className="space-y-4 rounded-2xl border border-zinc-800 bg-zinc-950/75 p-4">
          <div className="rounded-2xl border border-zinc-800 bg-[linear-gradient(180deg,rgba(20,20,24,0.96),rgba(10,10,12,0.96))] p-4">
            <div className="flex items-start justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.18em] text-zinc-500">Mission State</p>
                <p className="mt-1 text-lg font-semibold text-zinc-100">
                  {prediction?.symbol ?? predictionParams.symbol} {selectedModel ? `• ${selectedModel.modelVariant ?? selectedModel.id}` : ""}
                </p>
              </div>
              <StatusPill label={activeHealthLabel} tone={activeHealthTone} />
            </div>
            <p className="mt-3 text-sm text-zinc-300">
              {activeQualityReasons.length
                ? `Fallback is in effect because ${activeQualityReasons.join(", ")}.`
                : "Current configuration is cleared for the ML overlay path."}
            </p>
            <div className="mt-3 flex flex-wrap gap-2 text-[11px] uppercase tracking-[0.14em] text-zinc-400">
              <span>{`Profile ${activeFeatureProfile ?? "-"}`}</span>
              <span>{`Selection ${activeSelectionMode ?? "-"}`}</span>
              <span>{`Target ${activeTargetHorizon ?? 1}d`}</span>
            </div>
            <div className="mt-4 flex gap-2">
              <button
                type="button"
                onClick={() => setShowAdvanced(false)}
                className={cn(
                  "rounded-full border px-3 py-1.5 text-xs font-semibold transition",
                  !showAdvanced ? "border-cyan-400 bg-cyan-500/10 text-cyan-100" : "border-zinc-700 bg-zinc-900 text-zinc-300"
                )}
              >
                Focus Mode
              </button>
              <button
                type="button"
                onClick={() => setShowAdvanced(true)}
                className={cn(
                  "rounded-full border px-3 py-1.5 text-xs font-semibold transition",
                  showAdvanced ? "border-amber-400 bg-amber-500/10 text-amber-100" : "border-zinc-700 bg-zinc-900 text-zinc-300"
                )}
              >
                Advanced Controls
              </button>
            </div>
            {modelsQuery.isError ? (
              <p className="mt-3 text-xs text-rose-300">Model registry could not load. Verify the Python API is running and reachable.</p>
            ) : null}
          </div>

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

          <div className="space-y-3 rounded-xl border border-zinc-800 bg-zinc-900/60 p-3">
            <p className="text-xs uppercase tracking-[0.18em] text-zinc-500">Symbol + Data</p>
            
            <div>
              <label className="text-xs text-zinc-400">Stock Symbol</label>
              <input
                className="mt-1 w-full rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-cyan-400"
                value={predictionParams.symbol}
                onChange={(e) => setPredictionParams((p) => ({ ...p, symbol: e.target.value.toUpperCase() }))}
                placeholder="AAPL"
              />
              <p className="mt-1 text-[10px] text-zinc-500">Ticker symbol (e.g., AAPL, TSLA, MSFT)</p>
            </div>

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-zinc-400">Forecast Horizon</label>
                <input
                  className="mt-1 w-full rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-cyan-400"
                  type="number"
                  value={predictionParams.horizon}
                  min={1}
                  onChange={(e) => setPredictionParams((p) => ({ ...p, horizon: Number(e.target.value) || 1 }))}
                  placeholder="Horizon"
                />
                <p className="mt-1 text-[10px] text-zinc-500">Days to predict ahead (1-30)</p>
              </div>
              <div>
                <label className="text-xs text-zinc-400">Chart Bars</label>
                <input
                  className="mt-1 w-full rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-cyan-400"
                  type="number"
                  value={predictionParams.daysOnChart}
                  min={30}
                  onChange={(e) => setPredictionParams((p) => ({ ...p, daysOnChart: Number(e.target.value) || 120 }))}
                  placeholder="Chart Bars"
                />
                <p className="mt-1 text-[10px] text-zinc-500">Historical bars to display</p>
              </div>
            </div>

            <div className={cn("grid gap-3", showAdvanced ? "grid-cols-3" : "grid-cols-2")}>
              <div>
                <label className="text-xs text-zinc-400">Data Interval</label>
                <select
                  className="mt-1 w-full rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-xs text-zinc-100 outline-none focus:border-cyan-400"
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
                  <option value="1d">1 day</option>
                  <option value="1h">1 hour</option>
                  <option value="30m">30 min</option>
                  <option value="15m">15 min</option>
                </select>
                <p className="mt-1 text-[10px] text-zinc-500">Candle frequency</p>
              </div>
              <div>
                <label className="text-xs text-zinc-400">Data Period</label>
                <select
                  className="mt-1 w-full rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-xs text-zinc-100 outline-none focus:border-cyan-400"
                  value={predictionParams.dataPeriod ?? "10y"}
                  onChange={(e) => setPredictionParams((p) => ({ ...p, dataPeriod: e.target.value }))}
                >
                  <option value="10y">10 years</option>
                  <option value="5y">5 years</option>
                  <option value="730d">2 years</option>
                  <option value="365d">1 year</option>
                  <option value="180d">6 months</option>
                  <option value="60d">2 months</option>
                </select>
                <p className="mt-1 text-[10px] text-zinc-500">Historical lookback</p>
              </div>
              {showAdvanced ? (
              <div>
                <label className="text-xs text-zinc-400">Model Variant</label>
                <input
                  className="mt-1 w-full rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-xs text-zinc-100 outline-none focus:border-cyan-400"
                  value={predictionParams.modelVariant ?? "auto"}
                  onChange={(e) => setPredictionParams((p) => ({ ...p, modelVariant: e.target.value || "auto" }))}
                  placeholder="auto"
                />
                <p className="mt-1 text-[10px] text-zinc-500">auto, gbm, intraday_1h_v5</p>
              </div>
              ) : null}
            </div>
          </div>

          <div className="space-y-3 rounded-xl border border-zinc-800 bg-zinc-900/60 p-3">
            <p className="text-xs uppercase tracking-[0.18em] text-zinc-500">Model + Fusion</p>

            {showAdvanced ? (
            <div>
              <label className="text-xs text-zinc-400">Select Model</label>
              <select
                className="mt-1 w-full rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-cyan-400"
                value={selectedModelId}
                onChange={(e) => setSelectedModelId(e.target.value)}
              >
                <option value="">Latest by Symbol (auto)</option>
                {(modelsQuery.data ?? []).map((m: ModelMeta) => (
                  <option key={m.id} value={m.id}>{`${m.symbol} - ${m.modelVariant ?? m.id}`}</option>
                ))}
              </select>
              <p className="mt-1 text-[10px] text-zinc-500">Leave empty for latest trained model</p>
            </div>
            ) : (
              <div className="rounded-lg border border-zinc-800 bg-black/20 px-3 py-3 text-xs text-zinc-300">
                Auto-routing will choose the strongest available variant for the selected symbol and timeframe.
              </div>
            )}

            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-zinc-400">Buy Threshold</label>
                <input
                  className="mt-1 w-full rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-cyan-400"
                  type="number"
                  step="0.01"
                  value={fusionSettings.buyThreshold}
                  onChange={(e) => setFusionSettings((f) => ({ ...f, buyThreshold: Number(e.target.value) || 0.3 }))}
                  placeholder="Buy Thr"
                />
                <p className="mt-1 text-[10px] text-zinc-500">Min confidence to BUY (0-1). Higher = fewer trades, lower = more trades</p>
              </div>
              <div>
                <label className="text-xs text-zinc-400">Sell Threshold</label>
                <input
                  className="mt-1 w-full rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-cyan-400"
                  type="number"
                  step="0.01"
                  value={fusionSettings.sellThreshold}
                  onChange={(e) => setFusionSettings((f) => ({ ...f, sellThreshold: Number(e.target.value) || 0.45 }))}
                  placeholder="Sell Thr"
                />
                <p className="mt-1 text-[10px] text-zinc-500">Min confidence to SELL (0-1). Should be higher than buy threshold</p>
              </div>
            </div>
          </div>

          <div className="space-y-3 rounded-xl border border-zinc-800 bg-zinc-900/60 p-3">
            <p className="text-xs uppercase tracking-[0.18em] text-zinc-500">Execution + Risk Controls</p>
            
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="text-xs text-zinc-400">Backtest Window</label>
                <input
                  className="mt-1 w-full rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-cyan-400"
                  type="number"
                  min={10}
                  step="5"
                  value={backtestParams.backtestWindow}
                  onChange={(e) => setBacktestParams((b) => ({ ...b, backtestWindow: Number(e.target.value) || 120 }))}
                  placeholder="Backtest Window"
                />
                <p className="mt-1 text-[10px] text-zinc-500">Recent realized bars used for the main backtest.</p>
              </div>
              <div>
                <label className="text-xs text-zinc-400">Forward Holdout</label>
                <input
                  className="mt-1 w-full rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-cyan-400"
                  type="number"
                  min={1}
                  step="1"
                  value={backtestParams.forwardWindow}
                  onChange={(e) => setBacktestParams((b) => ({ ...b, forwardWindow: Number(e.target.value) || 20 }))}
                  placeholder="Forward Holdout"
                />
                <p className="mt-1 text-[10px] text-zinc-500">Reserved recent bars for the realized forward simulation.</p>
              </div>
              <div>
                <label className="text-xs text-zinc-400">Initial Capital</label>
                <input
                  className="mt-1 w-full rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-cyan-400"
                  type="number"
                  value={backtestParams.initialCapital}
                  onChange={(e) => setBacktestParams((b) => ({ ...b, initialCapital: Number(e.target.value) || 10_000 }))}
                  placeholder="Capital"
                />
                <p className="mt-1 text-[10px] text-zinc-500">Starting cash for backtest ($)</p>
              </div>
              <div>
                <label className="text-xs text-zinc-400">Max Long Position</label>
                <input
                  className="mt-1 w-full rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-cyan-400"
                  type="number"
                  step="0.1"
                  value={backtestParams.maxLong}
                  onChange={(e) => setBacktestParams((b) => ({ ...b, maxLong: Number(e.target.value) || 1.6 }))}
                  placeholder="Max Long"
                />
                <p className="mt-1 text-[10px] text-zinc-500">Max long as % of capital (1.0=100%). Higher = more bullish, more risk</p>
              </div>
              <div>
                <label className="text-xs text-zinc-400">Max Short Position</label>
                <input
                  className="mt-1 w-full rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-cyan-400"
                  type="number"
                  step="0.05"
                  value={backtestParams.maxShort}
                  onChange={(e) => setBacktestParams((b) => ({ ...b, maxShort: Number(e.target.value) || 0.25 }))}
                  placeholder="Max Short"
                />
                <p className="mt-1 text-[10px] text-zinc-500">Max short as % of capital (0.25=25%). Higher = more bearish, more risk</p>
              </div>
              <div>
                <label className="text-xs text-zinc-400">Commission (bps)</label>
                <input
                  className="mt-1 w-full rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-cyan-400"
                  type="number"
                  step="0.1"
                  value={backtestParams.commission}
                  onChange={(e) => setBacktestParams((b) => ({ ...b, commission: Number(e.target.value) || 0.5 }))}
                  placeholder="Commission (bps)"
                />
                <p className="mt-1 text-[10px] text-zinc-500">Trading fee per trade. 0.5 bps = 0.005%. Higher = more costs, less profit</p>
              </div>
              <div>
                <label className="text-xs text-zinc-400">Slippage (bps)</label>
                <input
                  className="mt-1 w-full rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-cyan-400"
                  type="number"
                  step="0.1"
                  value={backtestParams.slippage}
                  onChange={(e) => setBacktestParams((b) => ({ ...b, slippage: Number(e.target.value) || 0.2 }))}
                  placeholder="Slippage (bps)"
                />
                <p className="mt-1 text-[10px] text-zinc-500">Price impact per trade. 0.2 bps = 0.002%. Simulates market impact</p>
              </div>
              {showAdvanced ? (
              <div>
                <label className="text-xs text-zinc-400">Min Position Change</label>
                <input
                  className="mt-1 w-full rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-cyan-400"
                  type="number"
                  step="0.01"
                  value={backtestParams.minPositionChange ?? 0}
                  onChange={(e) => setBacktestParams((b) => ({ ...b, minPositionChange: Number(e.target.value) || 0 }))}
                  placeholder="Min Pos Delta"
                />
                <p className="mt-1 text-[10px] text-zinc-500">Min position change to execute trade. Higher = fewer trades, less overtrading</p>
              </div>
              ) : null}
            </div>
            
            {showAdvanced ? (
            <div>
              <label className="text-xs text-zinc-400">Annualization Factor (optional)</label>
              <input
                className="mt-1 w-full rounded-lg border border-zinc-700 bg-black/50 px-3 py-2 text-sm text-zinc-100 outline-none focus:border-cyan-400"
                type="number"
                step="1"
                value={backtestParams.annualizationFactor ?? ""}
                onChange={(e) => {
                  const v = e.target.value.trim();
                  setBacktestParams((b) => ({ ...b, annualizationFactor: v === "" ? undefined : Number(v) }));
                }}
                placeholder="Leave empty for auto"
              />
              <p className="mt-1 text-[10px] text-zinc-500">For daily: 252. For hourly: ~6500. Auto-calculated if empty. Used for Sharpe ratio</p>
            </div>
            ) : null}
            
            <div className="grid grid-cols-2 gap-3 pt-2 text-sm text-zinc-300">
              <label className="flex items-start gap-2">
                <input
                  type="checkbox"
                  checked={backtestParams.enableForwardSim}
                  onChange={(e) => setBacktestParams((b) => ({ ...b, enableForwardSim: e.target.checked }))}
                  className="mt-1"
                />
                <span>
                  <span className="font-medium text-zinc-200">Forward Simulation</span>
                  <br/><span className="text-[10px] text-zinc-500">Test predictions on unseen future data. More realistic but shorter evaluation period.</span>
                </span>
              </label>
              <label className="flex items-start gap-2">
                <input
                  type="checkbox"
                  checked={Boolean(backtestParams.flatAtDayEnd ?? prediction?.metadata?.flatAtDayEnd)}
                  onChange={(e) => setBacktestParams((b) => ({ ...b, flatAtDayEnd: e.target.checked }))}
                  className="mt-1"
                />
                <span>
                  <span className="font-medium text-zinc-200">Flat At Day End</span>
                  <br/><span className="text-[10px] text-zinc-500">Close all positions daily. For intraday only - reduces overnight risk.</span>
                </span>
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
          {!prediction ? (
            <section className="rounded-3xl border border-zinc-800 bg-[linear-gradient(180deg,rgba(18,18,22,0.95),rgba(8,8,10,0.95))] p-8">
              <p className="text-xs uppercase tracking-[0.18em] text-zinc-500">Start Here</p>
              <h2 className="mt-2 text-2xl font-semibold text-zinc-100">Pick a symbol, choose a preset, then run prediction.</h2>
              <div className="mt-6 grid gap-3 md:grid-cols-3">
                <div className="rounded-2xl border border-zinc-800 bg-black/25 p-4">
                  <p className="text-sm font-semibold text-zinc-100">1. Select the profile</p>
                  <p className="mt-2 text-sm text-zinc-400">Use the daily or intraday preset first. It sets interval, period, leverage, and trading assumptions together.</p>
                </div>
                <div className="rounded-2xl border border-zinc-800 bg-black/25 p-4">
                  <p className="text-sm font-semibold text-zinc-100">2. Check model health</p>
                  <p className="mt-2 text-sm text-zinc-400">The left rail shows whether the selected setup can use the ML overlay or is falling back to the regime core.</p>
                </div>
                <div className="rounded-2xl border border-zinc-800 bg-black/25 p-4">
                  <p className="text-sm font-semibold text-zinc-100">3. Run the loop</p>
                  <p className="mt-2 text-sm text-zinc-400">Prediction builds the signal deck. Backtest then fills equity, diagnostics, and the execution ledger.</p>
                </div>
              </div>
            </section>
          ) : null}

          <section id="chart" className="rounded-2xl border border-zinc-800 bg-zinc-950/70 p-4">
            <div className="mb-3 flex flex-wrap items-end justify-between gap-3">
              <div>
                <p className="text-xs uppercase tracking-[0.18em] text-zinc-500">Loaded</p>
                <p className="font-mono text-lg text-zinc-100">
                  {prediction?.symbol ?? predictionParams.symbol} {selectedModel ? `• ${selectedModel.modelVariant ?? selectedModel.id}` : ""}
                </p>
              </div>
              <div className="flex flex-wrap gap-2">
                <StatusPill label={activeQualityGate ? "ML Overlay Enabled" : "Regime Core Active"} tone={activeQualityGate ? "good" : "warn"} />
                <StatusPill label={`Source ${activeHoldoutSource ?? "-"}`} tone="neutral" />
              </div>
            </div>

            <div className="mb-4 grid gap-3 rounded-2xl border border-zinc-800 bg-black/20 p-3 md:grid-cols-4">
              <MetricCard label="Interval" value={prediction?.metadata?.dataInterval ?? predictionParams.dataInterval ?? "1d"} />
              <MetricCard label="Variant" value={prediction?.metadata?.modelVariant ?? predictionParams.modelVariant ?? "auto"} />
              <MetricCard label="Gate Score" value={activeQualityScore === null ? "-" : activeQualityScore.toFixed(2)} tone={activeQualityGate ? "good" : "bad"} />
              <MetricCard label="Why" value={activeQualityReasons[0] ?? "healthy"} tone={activeQualityReasons.length ? "bad" : "good"} />
            </div>
            {forecastBandSummary ? (
              <div className="mb-4 grid gap-3 rounded-2xl border border-zinc-800 bg-black/20 p-3 md:grid-cols-3">
                <MetricCard label="Forecast Mid" value={formatCurrency(forecastBandSummary.base)} />
                <MetricCard label="Forecast Floor" value={formatCurrency(forecastBandSummary.lower)} tone="bad" />
                <MetricCard label="Forecast Ceiling" value={formatCurrency(forecastBandSummary.upper)} tone="good" />
              </div>
            ) : null}

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

          <section id="equity" className="rounded-2xl border border-zinc-800 bg-zinc-950/70 p-4">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
              <p className="text-xs uppercase tracking-[0.18em] text-zinc-500">Portfolio vs Stock</p>
              <p className="text-xs text-zinc-400">
                Portfolio Equity (yellow) vs Stock Buy&Hold (blue) with trade dots on portfolio curve
              </p>
            </div>
            <EquityLineChart backtest={backtest ?? undefined} tradeMarkers={backtest?.annotations} />
          </section>

          <div id="diagnostics">
            <DiagnosticsPanel diagnostics={backtest?.diagnostics} intraday={intraday} />
          </div>

          <ForwardSimulationPanel backtest={backtest} />

          <section id="trades" className="rounded-2xl border border-zinc-800 bg-zinc-950/70 p-4">
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
