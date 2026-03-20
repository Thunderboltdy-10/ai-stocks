"use client";

import { BenchmarkRun, BenchmarkModeSummary } from "@/types/ai";
import { formatPercent } from "@/utils/formatters";
import { cn } from "@/lib/utils";

function MetricTile({
  label,
  value,
  tone = "neutral",
}: {
  label: string;
  value: string;
  tone?: "neutral" | "good" | "bad";
}) {
  return (
    <div className="rounded-2xl border border-zinc-800 bg-black/30 px-4 py-3">
      <p className="text-[10px] uppercase tracking-[0.16em] text-zinc-500">{label}</p>
      <p
        className={cn(
          "mt-1 text-lg font-semibold",
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

function ModeCard({ summary }: { summary: BenchmarkModeSummary }) {
  const alphaTone = summary.aggregate.medianAlpha >= 0 ? "good" : "bad";
  const sharpeTone = summary.aggregate.meanSharpe >= 0 ? "good" : "bad";

  return (
    <div className="rounded-[28px] border border-zinc-800 bg-[linear-gradient(180deg,rgba(17,24,39,0.95),rgba(9,9,11,0.95))] p-5 shadow-[0_20px_60px_rgba(0,0,0,0.35)]">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <p className="text-xs uppercase tracking-[0.24em] text-cyan-300">{summary.mode} benchmark</p>
          <h3 className="mt-1 text-2xl font-semibold text-zinc-100">{summary.symbolSet}</h3>
          <p className="mt-1 text-sm text-zinc-400">
            {summary.symbolsTested} symbols, {summary.variantsEvaluated} variants, {summary.windowsEvaluated} windows
          </p>
        </div>
        <div className="rounded-full border border-zinc-700 bg-zinc-950/80 px-3 py-1 text-xs uppercase tracking-[0.16em] text-zinc-300">
          pass rate {formatPercent(summary.aggregate.qualityPassRate)}
        </div>
      </div>

      <div className="mt-4 grid gap-3 md:grid-cols-5">
        <MetricTile label="Median Alpha" value={formatPercent(summary.aggregate.medianAlpha)} tone={alphaTone} />
        <MetricTile label="Mean Sharpe" value={summary.aggregate.meanSharpe.toFixed(2)} tone={sharpeTone} />
        <MetricTile label="Alpha Hit Rate" value={formatPercent(summary.aggregate.positiveAlphaRate)} tone={summary.aggregate.positiveAlphaRate >= 0.5 ? "good" : "bad"} />
        <MetricTile label="Mean Return" value={formatPercent(summary.aggregate.meanStrategyReturn)} tone={summary.aggregate.meanStrategyReturn >= 0 ? "good" : "bad"} />
        <MetricTile label="ML Lift vs Regime" value={formatPercent(summary.aggregate.meanMlLiftVsRegime)} tone={summary.aggregate.meanMlLiftVsRegime >= 0 ? "good" : "bad"} />
      </div>

      <div className="mt-5 grid gap-4 xl:grid-cols-3">
        <div className="rounded-2xl border border-zinc-800 bg-black/20 p-4">
          <p className="text-[11px] uppercase tracking-[0.18em] text-zinc-500">Leaders</p>
          <div className="mt-3 space-y-2">
            {summary.leaders.length ? (
              summary.leaders.map((row) => (
                <div key={`${summary.mode}-lead-${row.symbol}`} className="flex items-center justify-between rounded-xl border border-zinc-800/80 bg-zinc-950/60 px-3 py-2">
                  <div>
                    <p className="font-semibold text-zinc-100">{row.symbol}</p>
                    <p className="text-xs text-zinc-500">{row.variant}</p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm font-semibold text-emerald-300">{formatPercent(row.mean_alpha)}</p>
                    <p className="text-xs text-zinc-500">Sharpe {row.mean_sharpe.toFixed(2)}</p>
                  </div>
                </div>
              ))
            ) : (
              <p className="text-sm text-zinc-500">No benchmark rows recorded yet.</p>
            )}
          </div>
        </div>

        <div className="rounded-2xl border border-zinc-800 bg-black/20 p-4">
          <p className="text-[11px] uppercase tracking-[0.18em] text-zinc-500">Stress Points</p>
          <div className="mt-3 space-y-2">
            {summary.laggards.length ? (
              summary.laggards.map((row) => (
                <div key={`${summary.mode}-lag-${row.symbol}`} className="flex items-center justify-between rounded-xl border border-zinc-800/80 bg-zinc-950/60 px-3 py-2">
                  <div>
                    <p className="font-semibold text-zinc-100">{row.symbol}</p>
                    <p className="text-xs text-zinc-500">{row.variant}</p>
                  </div>
                  <div className="text-right">
                    <p className={cn("text-sm font-semibold", row.mean_alpha >= 0 ? "text-emerald-300" : "text-rose-300")}>
                      {formatPercent(row.mean_alpha)}
                    </p>
                    <p className="text-xs text-zinc-500">{formatPercent(row.positive_alpha_rate)} windows positive</p>
                  </div>
                </div>
              ))
            ) : (
              <p className="text-sm text-zinc-500">No stress cases recorded yet.</p>
            )}
          </div>
        </div>

        <div className="rounded-2xl border border-zinc-800 bg-black/20 p-4">
          <p className="text-[11px] uppercase tracking-[0.18em] text-zinc-500">Top Quality Variants</p>
          <div className="mt-3 space-y-2">
            {summary.qualityLeaders.length ? (
              summary.qualityLeaders.map((row) => (
                <div key={`${summary.mode}-quality-${row.symbol}-${row.variant}`} className="rounded-xl border border-zinc-800/80 bg-zinc-950/60 px-3 py-2">
                  <div className="flex items-center justify-between gap-2">
                    <p className="font-semibold text-zinc-100">{row.symbol}</p>
                    <p className="text-sm font-semibold text-cyan-300">{(row.quality_score ?? 0).toFixed(2)}</p>
                  </div>
                  <p className="text-xs text-zinc-500">
                    {row.variant} • h{row.target_horizon_days ?? 1}
                  </p>
                </div>
              ))
            ) : (
              <p className="text-sm text-zinc-500">Training metadata has not been benchmarked yet.</p>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export function BenchmarkLabPanel({ runs }: { runs: BenchmarkRun[] | undefined }) {
  const latest = runs?.[0];
  const daily = latest?.modes?.daily;
  const intraday = latest?.modes?.intraday;

  return (
    <section className="rounded-[32px] border border-zinc-800 bg-[radial-gradient(800px_320px_at_0%_0%,rgba(34,211,238,0.08),transparent_60%),radial-gradient(700px_280px_at_100%_0%,rgba(245,158,11,0.08),transparent_55%),linear-gradient(180deg,rgba(12,14,18,0.97),rgba(5,6,8,0.97))] p-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <p className="text-xs uppercase tracking-[0.24em] text-amber-300">Benchmark Lab</p>
          <h2 className="mt-2 text-3xl font-semibold text-zinc-100">Cross-Symbol Evidence</h2>
          <p className="mt-2 max-w-3xl text-sm text-zinc-400">
            Daily and intraday research runs are summarized here so model selection is anchored to diversified, out-of-sample evidence instead of a single lucky chart.
          </p>
        </div>
        <div className="rounded-2xl border border-zinc-800 bg-zinc-950/70 px-4 py-3 text-sm text-zinc-300">
          <p className="text-[10px] uppercase tracking-[0.16em] text-zinc-500">Latest Run</p>
          <p className="mt-1 font-mono">{latest?.id ?? "none"}</p>
          <p className="text-xs text-zinc-500">
            {latest?.createdAt ? new Date(latest.createdAt).toLocaleString() : "Run `python python-ai-service/scripts/run_benchmark_lab.py --modes daily,intraday`"}
          </p>
        </div>
      </div>

      <div className="mt-5 space-y-5">
        {daily ? <ModeCard summary={daily} /> : null}
        {intraday ? <ModeCard summary={intraday} /> : null}
        {!daily && !intraday ? (
          <div className="rounded-2xl border border-dashed border-zinc-700 bg-black/20 p-6 text-sm text-zinc-400">
            No benchmark runs found yet. The benchmark lab writes artifacts under `python-ai-service/benchmark_runs/` and the latest summary appears here automatically.
          </div>
        ) : null}
      </div>
    </section>
  );
}
