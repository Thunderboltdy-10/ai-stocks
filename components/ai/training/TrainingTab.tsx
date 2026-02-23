"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { toast } from "sonner";
import { TrainingConfigForm, TrainingConfig } from "./TrainingConfigForm";
import { TrainingProgressPanel, TrainingJob, EpochUpdate } from "./TrainingProgressPanel";
import { TrainingMetricsChart } from "./TrainingMetricsChart";
import { startTraining, cancelTrainingJob, subscribeToTrainingEvents } from "@/lib/api-client";

interface TrainingTabProps {
  onTrainingComplete?: () => void;
}

export function TrainingTab({ onTrainingComplete }: TrainingTabProps) {
  const [currentJob, setCurrentJob] = useState<TrainingJob | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const eventSourceRef = useRef<EventSource | null>(null);

  // Cleanup EventSource on unmount
  useEffect(() => {
    return () => {
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
    };
  }, []);

  const handleStartTraining = useCallback(async (config: TrainingConfig) => {
    setIsSubmitting(true);
    try {
      const response = await startTraining({
        symbol: config.symbol,
        epochs: config.epochs,
        batchSize: config.batchSize,
        loss: config.loss,
        sequenceLength: config.sequenceLength,
        featureToggles: config.featureToggles,
        ensembleSize: 1,
        baseSeed: 42,
        modelType: config.modelType,
        dropout: config.dropout,
        learningRate: config.learningRate,
      });

      // Initialize job state
      const job: TrainingJob = {
        id: response.jobId,
        symbol: config.symbol,
        status: "queued",
        progress: 0,
        currentEpoch: 0,
        totalEpochs: config.epochs,
        startedAt: new Date().toISOString(),
        epochs: [],
      };
      setCurrentJob(job);

      // Subscribe to SSE events
      const eventSource = subscribeToTrainingEvents(response.jobId);
      eventSourceRef.current = eventSource;

      eventSource.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);

          setCurrentJob((prev) => {
            if (!prev) return prev;

            if (data.type === "epoch") {
              const epochUpdate: EpochUpdate = {
                epoch: data.epoch,
                totalEpochs: data.totalEpochs,
                trainLoss: data.trainLoss,
                valLoss: data.valLoss,
                learningRate: data.learningRate,
                timestamp: data.timestamp || new Date().toISOString(),
                directionalAccuracy: data.directionalAccuracy,
              };

              return {
                ...prev,
                status: "running",
                progress: data.epoch / data.totalEpochs,
                currentEpoch: data.epoch,
                epochs: [...prev.epochs, epochUpdate],
              };
            }

            if (data.type === "completed") {
              eventSource.close();
              return {
                ...prev,
                status: "completed",
                progress: 1,
                completedAt: new Date().toISOString(),
              };
            }

            if (data.type === "failed") {
              eventSource.close();
              return {
                ...prev,
                status: "failed",
                error: data.error,
              };
            }

            return prev;
          });
        } catch (err) {
          console.error("Failed to parse SSE event:", err);
        }
      };

      eventSource.onerror = () => {
        eventSource.close();
        setCurrentJob((prev) => {
          if (!prev || prev.status === "completed") return prev;
          return {
            ...prev,
            status: "failed",
            error: "Connection to training service lost",
          };
        });
      };

      toast.success(`Training started for ${config.symbol}`);
    } catch (error) {
      toast.error(`Failed to start training: ${error instanceof Error ? error.message : "Unknown error"}`);
    } finally {
      setIsSubmitting(false);
    }
  }, []);

  const handleCancelTraining = useCallback(async () => {
    if (!currentJob) return;

    try {
      await cancelTrainingJob(currentJob.id);
      if (eventSourceRef.current) {
        eventSourceRef.current.close();
      }
      setCurrentJob((prev) => prev ? { ...prev, status: "cancelled" } : null);
      toast.info("Training cancelled");
    } catch (error) {
      toast.error(`Failed to cancel training: ${error instanceof Error ? error.message : "Unknown error"}`);
    }
  }, [currentJob]);

  const handleTrainingComplete = useCallback(() => {
    toast.success("Training completed successfully!");
    if (onTrainingComplete) {
      onTrainingComplete();
    }
  }, [onTrainingComplete]);

  const isTraining = currentJob?.status === "running" || currentJob?.status === "queued";

  return (
    <div className="grid gap-6 lg:grid-cols-[1fr,1fr]">
      {/* Left Column: Configuration */}
      <div>
        <TrainingConfigForm
          onSubmit={handleStartTraining}
          isSubmitting={isSubmitting}
          disabled={isTraining}
        />
      </div>

      {/* Right Column: Progress and Metrics */}
      <div className="space-y-6">
        <TrainingProgressPanel
          job={currentJob}
          onCancel={handleCancelTraining}
          onComplete={handleTrainingComplete}
        />

        {currentJob && currentJob.epochs.length > 0 && (
          <TrainingMetricsChart epochs={currentJob.epochs} />
        )}
      </div>
    </div>
  );
}
