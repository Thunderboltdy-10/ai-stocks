"use client";

import { ModelMeta } from "@/types/ai";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { X, Download, TrendingUp, Target, Activity, Clock, Layers, Database } from "lucide-react";
import { format } from "date-fns";

interface ModelDetailCardProps {
  model: ModelMeta | null;
  onClose: () => void;
  onDownload: () => void;
}

interface MetricItemProps {
  icon: React.ReactNode;
  label: string;
  value: string | number;
  color?: string;
}

function MetricItem({ icon, label, value, color = "text-white" }: MetricItemProps) {
  return (
    <div className="flex items-center gap-3 p-3 rounded-lg bg-gray-800/50">
      <span className="text-gray-500">{icon}</span>
      <div>
        <p className="text-xs text-gray-500">{label}</p>
        <p className={cn("text-lg font-semibold", color)}>{value}</p>
      </div>
    </div>
  );
}

export function ModelDetailCard({ model, onClose, onDownload }: ModelDetailCardProps) {
  if (!model) {
    return null;
  }

  const sharpe = model.metrics?.sharpeRatio ?? 0;
  const winRate = model.metrics?.winRate ?? 0;
  const maxDD = model.metrics?.maxDrawdown ?? 0;
  const dirAcc = model.metrics?.directionalAccuracy ?? 0;
  const cumReturn = model.metrics?.cumulativeReturn ?? 0;
  const trades = model.metrics?.totalTrades ?? 0;

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5">
      {/* Header */}
      <div className="flex items-start justify-between mb-4">
        <div>
          <h3 className="text-xl font-semibold text-white">{model.symbol}</h3>
          <p className="text-sm text-gray-500">
            Created {format(new Date(model.createdAt), "MMM d, yyyy 'at' h:mm a")}
          </p>
        </div>
        <Button variant="ghost" size="icon-sm" onClick={onClose} className="text-gray-400 hover:text-white">
          <X className="size-5" />
        </Button>
      </div>

      {/* Performance Metrics */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 mb-4">
        <MetricItem
          icon={<TrendingUp className="size-4" />}
          label="Sharpe Ratio"
          value={sharpe.toFixed(2)}
          color={sharpe > 1 ? "text-emerald-400" : sharpe > 0.5 ? "text-yellow-400" : "text-gray-400"}
        />
        <MetricItem
          icon={<Target className="size-4" />}
          label="Win Rate"
          value={`${(winRate * 100).toFixed(0)}%`}
          color={winRate > 0.55 ? "text-emerald-400" : winRate > 0.45 ? "text-yellow-400" : "text-gray-400"}
        />
        <MetricItem
          icon={<Activity className="size-4" />}
          label="Max Drawdown"
          value={`${(maxDD * 100).toFixed(1)}%`}
          color={maxDD < 0.15 ? "text-emerald-400" : maxDD < 0.25 ? "text-yellow-400" : "text-rose-400"}
        />
        <MetricItem
          icon={<TrendingUp className="size-4" />}
          label="Cumulative Return"
          value={`${cumReturn >= 0 ? "+" : ""}${(cumReturn * 100).toFixed(1)}%`}
          color={cumReturn > 0 ? "text-emerald-400" : "text-rose-400"}
        />
        <MetricItem
          icon={<Target className="size-4" />}
          label="Dir. Accuracy"
          value={`${(dirAcc * 100).toFixed(1)}%`}
          color={dirAcc > 0.55 ? "text-emerald-400" : dirAcc > 0.5 ? "text-yellow-400" : "text-gray-400"}
        />
        <MetricItem
          icon={<Activity className="size-4" />}
          label="Total Trades"
          value={trades}
        />
      </div>

      {/* Model Configuration */}
      <div className="border-t border-gray-800 pt-4 mb-4">
        <h4 className="text-sm font-medium text-gray-400 mb-3">Configuration</h4>
        <div className="grid grid-cols-2 gap-3 text-sm">
          <div className="flex items-center gap-2">
            <Clock className="size-4 text-gray-500" />
            <span className="text-gray-400">Sequence Length:</span>
            <span className="text-white font-medium">{model.sequenceLength} days</span>
          </div>
          <div className="flex items-center gap-2">
            <Layers className="size-4 text-gray-500" />
            <span className="text-gray-400">Ensemble Size:</span>
            <span className="text-white font-medium">{model.ensembleSize}</span>
          </div>
          <div className="flex items-center gap-2">
            <Database className="size-4 text-gray-500" />
            <span className="text-gray-400">Fusion Mode:</span>
            <span className="text-white font-medium capitalize">{model.fusionModeDefault}</span>
          </div>
        </div>
      </div>

      {/* Notes */}
      {model.notes && (
        <div className="border-t border-gray-800 pt-4 mb-4">
          <h4 className="text-sm font-medium text-gray-400 mb-2">Notes</h4>
          <p className="text-sm text-gray-300">{model.notes}</p>
        </div>
      )}

      {/* Actions */}
      <div className="flex gap-2">
        <Button
          onClick={onDownload}
          className="flex-1 bg-gradient-to-b from-yellow-400 to-yellow-500 text-gray-900 font-medium hover:from-yellow-300 hover:to-yellow-400"
        >
          <Download className="size-4 mr-2" />
          Download Artifacts
        </Button>
      </div>
    </div>
  );
}
