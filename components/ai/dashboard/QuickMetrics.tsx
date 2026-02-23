"use client";

import { cn } from "@/lib/utils";
import {
  TrendingUp,
  Target,
  Clock,
  Layers,
  Activity,
  BarChart3,
} from "lucide-react";

interface MetricCardProps {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  subValue?: string;
  color?: string;
}

function MetricCard({ icon, label, value, subValue, color = "text-white" }: MetricCardProps) {
  return (
    <div className="rounded-lg border border-gray-800 bg-gray-900/40 p-4">
      <div className="flex items-center gap-2 mb-2">
        <span className="text-gray-500">{icon}</span>
        <span className="text-xs text-gray-500 uppercase tracking-wider">{label}</span>
      </div>
      <p className={cn("text-xl font-semibold", color)}>{value}</p>
      {subValue && <p className="text-xs text-gray-500 mt-1">{subValue}</p>}
    </div>
  );
}

interface QuickMetricsProps {
  expectedReturn: number | null;
  confidence: number | null;
  horizon: number | null;
  fusionMode: string | null;
  sharpeRatio: number | null;
  directionalAccuracy: number | null;
}

export function QuickMetrics({
  expectedReturn,
  confidence,
  horizon,
  fusionMode,
  sharpeRatio,
  directionalAccuracy,
}: QuickMetricsProps) {
  const formatReturn = (ret: number) => `${ret >= 0 ? "+" : ""}${(ret * 100).toFixed(2)}%`;
  const formatPercent = (val: number) => `${(val * 100).toFixed(0)}%`;

  const hasData = expectedReturn !== null || confidence !== null;

  if (!hasData) {
    return (
      <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5">
        <h3 className="text-sm font-medium text-gray-400 mb-3">Quick Metrics</h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
          {[...Array(6)].map((_, i) => (
            <div key={i} className="rounded-lg border border-gray-800 bg-gray-900/40 p-4">
              <div className="h-4 w-16 bg-gray-800 rounded animate-pulse mb-2" />
              <div className="h-6 w-12 bg-gray-800 rounded animate-pulse" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5">
      <h3 className="text-sm font-medium text-gray-400 mb-3">Quick Metrics</h3>
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
        <MetricCard
          icon={<TrendingUp className="size-4" />}
          label="Expected Return"
          value={expectedReturn !== null ? formatReturn(expectedReturn) : "—"}
          color={expectedReturn !== null ? (expectedReturn >= 0 ? "text-emerald-400" : "text-rose-400") : "text-gray-400"}
        />
        <MetricCard
          icon={<Target className="size-4" />}
          label="Confidence"
          value={confidence !== null ? formatPercent(confidence) : "—"}
          color={confidence !== null && confidence >= 0.6 ? "text-emerald-400" : "text-yellow-400"}
        />
        <MetricCard
          icon={<Clock className="size-4" />}
          label="Horizon"
          value={horizon !== null ? `${horizon} days` : "—"}
        />
        <MetricCard
          icon={<Layers className="size-4" />}
          label="Fusion Mode"
          value={fusionMode ?? "—"}
          subValue="Model ensemble strategy"
        />
        <MetricCard
          icon={<Activity className="size-4" />}
          label="Sharpe Ratio"
          value={sharpeRatio !== null ? sharpeRatio.toFixed(2) : "—"}
          color={sharpeRatio !== null && sharpeRatio > 1 ? "text-emerald-400" : "text-white"}
        />
        <MetricCard
          icon={<BarChart3 className="size-4" />}
          label="Dir. Accuracy"
          value={directionalAccuracy !== null ? formatPercent(directionalAccuracy) : "—"}
          color={directionalAccuracy !== null && directionalAccuracy > 0.55 ? "text-emerald-400" : "text-white"}
        />
      </div>
    </div>
  );
}
