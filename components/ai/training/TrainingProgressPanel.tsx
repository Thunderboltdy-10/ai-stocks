"use client";

import { useEffect, useState, useRef } from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Loader2, CheckCircle2, XCircle, Square, Clock, Cpu, Activity } from "lucide-react";

interface EpochUpdate {
  epoch: number;
  totalEpochs: number;
  trainLoss: number;
  valLoss: number;
  learningRate: number;
  timestamp: string;
  directionalAccuracy?: number;
}

interface TrainingJob {
  id: string;
  symbol: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  progress: number;
  currentEpoch: number;
  totalEpochs: number;
  startedAt: string;
  completedAt?: string;
  error?: string;
  epochs: EpochUpdate[];
}

interface TrainingProgressPanelProps {
  job: TrainingJob | null;
  onCancel?: () => void;
  onComplete?: () => void;
}

const STATUS_CONFIG = {
  queued: { icon: <Clock className="size-5" />, color: "text-gray-400", label: "Queued" },
  running: { icon: <Loader2 className="size-5 animate-spin" />, color: "text-yellow-400", label: "Training" },
  completed: { icon: <CheckCircle2 className="size-5" />, color: "text-emerald-400", label: "Completed" },
  failed: { icon: <XCircle className="size-5" />, color: "text-rose-400", label: "Failed" },
  cancelled: { icon: <Square className="size-5" />, color: "text-gray-400", label: "Cancelled" },
};

export function TrainingProgressPanel({ job, onCancel, onComplete }: TrainingProgressPanelProps) {
  const logRef = useRef<HTMLDivElement>(null);

  // Auto-scroll log to bottom
  useEffect(() => {
    if (logRef.current) {
      logRef.current.scrollTop = logRef.current.scrollHeight;
    }
  }, [job?.epochs]);

  // Notify on completion
  useEffect(() => {
    if (job?.status === "completed" && onComplete) {
      onComplete();
    }
  }, [job?.status, onComplete]);

  if (!job) {
    return (
      <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5">
        <div className="flex flex-col items-center justify-center py-12 text-center">
          <Cpu className="size-12 text-gray-700 mb-4" />
          <p className="text-gray-400">No training job active</p>
          <p className="text-sm text-gray-500 mt-1">Configure and start a training job to see progress here</p>
        </div>
      </div>
    );
  }

  const statusConfig = STATUS_CONFIG[job.status];
  const progressPercent = Math.round(job.progress * 100);
  const lastEpoch = job.epochs.at(-1);

  const formatDuration = (startedAt: string, completedAt?: string) => {
    const start = new Date(startedAt).getTime();
    const end = completedAt ? new Date(completedAt).getTime() : Date.now();
    const seconds = Math.floor((end - start) / 1000);
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    if (hours > 0) return `${hours}h ${minutes % 60}m`;
    if (minutes > 0) return `${minutes}m ${seconds % 60}s`;
    return `${seconds}s`;
  };

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className={cn("size-10 rounded-lg flex items-center justify-center bg-gray-800", statusConfig.color)}>
            {statusConfig.icon}
          </div>
          <div>
            <h3 className="text-lg font-semibold text-white">{job.symbol} Training</h3>
            <p className={cn("text-sm", statusConfig.color)}>{statusConfig.label}</p>
          </div>
        </div>

        {job.status === "running" && onCancel && (
          <Button variant="outline" size="sm" onClick={onCancel} className="border-gray-700 text-gray-300">
            <Square className="size-4 mr-2" />
            Cancel
          </Button>
        )}
      </div>

      {/* Progress Bar */}
      <div className="space-y-2">
        <div className="flex items-center justify-between text-sm">
          <span className="text-gray-400">
            Epoch {job.currentEpoch} / {job.totalEpochs}
          </span>
          <span className="text-gray-400">{progressPercent}%</span>
        </div>
        <div className="h-2 rounded-full bg-gray-800 overflow-hidden">
          <div
            className={cn(
              "h-full rounded-full transition-all duration-300",
              job.status === "completed"
                ? "bg-emerald-500"
                : job.status === "failed"
                ? "bg-rose-500"
                : "bg-yellow-500"
            )}
            style={{ width: `${progressPercent}%` }}
          />
        </div>
      </div>

      {/* Metrics */}
      {lastEpoch && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="rounded-lg bg-gray-800/50 p-3">
            <p className="text-xs text-gray-500 mb-1">Train Loss</p>
            <p className="text-lg font-semibold text-white">{lastEpoch.trainLoss.toFixed(4)}</p>
          </div>
          <div className="rounded-lg bg-gray-800/50 p-3">
            <p className="text-xs text-gray-500 mb-1">Val Loss</p>
            <p className="text-lg font-semibold text-white">{lastEpoch.valLoss.toFixed(4)}</p>
          </div>
          <div className="rounded-lg bg-gray-800/50 p-3">
            <p className="text-xs text-gray-500 mb-1">Learning Rate</p>
            <p className="text-lg font-semibold text-white">{lastEpoch.learningRate.toExponential(2)}</p>
          </div>
          <div className="rounded-lg bg-gray-800/50 p-3">
            <p className="text-xs text-gray-500 mb-1">Duration</p>
            <p className="text-lg font-semibold text-white">{formatDuration(job.startedAt, job.completedAt)}</p>
          </div>
        </div>
      )}

      {/* Epoch Log */}
      <div className="space-y-2">
        <div className="flex items-center gap-2">
          <Activity className="size-4 text-gray-500" />
          <span className="text-sm text-gray-400">Training Log</span>
        </div>
        <div
          ref={logRef}
          className="h-40 overflow-y-auto rounded-lg bg-gray-950 border border-gray-800 p-3 font-mono text-xs"
        >
          {job.epochs.length === 0 ? (
            <p className="text-gray-500">Waiting for first epoch...</p>
          ) : (
            job.epochs.map((epoch, idx) => (
              <div key={idx} className="py-1 border-b border-gray-800/50 last:border-0">
                <span className="text-gray-500">[{new Date(epoch.timestamp).toLocaleTimeString()}]</span>{" "}
                <span className="text-yellow-400">Epoch {epoch.epoch}/{epoch.totalEpochs}</span>{" "}
                <span className="text-gray-400">train_loss:</span>{" "}
                <span className="text-white">{epoch.trainLoss.toFixed(4)}</span>{" "}
                <span className="text-gray-400">val_loss:</span>{" "}
                <span className={cn(epoch.valLoss < epoch.trainLoss * 1.1 ? "text-emerald-400" : "text-rose-400")}>
                  {epoch.valLoss.toFixed(4)}
                </span>
                {epoch.directionalAccuracy && (
                  <>
                    {" "}
                    <span className="text-gray-400">DA:</span>{" "}
                    <span className="text-white">{(epoch.directionalAccuracy * 100).toFixed(1)}%</span>
                  </>
                )}
              </div>
            ))
          )}
        </div>
      </div>

      {/* Error Message */}
      {job.status === "failed" && job.error && (
        <div className="rounded-lg bg-rose-500/10 border border-rose-500/50 p-4">
          <div className="flex items-center gap-2 text-rose-400 mb-2">
            <XCircle className="size-4" />
            <span className="font-medium">Training Failed</span>
          </div>
          <p className="text-sm text-rose-300">{job.error}</p>
        </div>
      )}
    </div>
  );
}

export type { TrainingJob, EpochUpdate };
