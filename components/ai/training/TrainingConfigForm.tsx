"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import { GraduationCap, Cpu, Loader2, AlertCircle, Info } from "lucide-react";

type ModelType = "lstm_transformer" | "gbm" | "stacking";
type LossFunction = "huber" | "mae" | "balanced" | "quantile";

interface TrainingConfig {
  symbol: string;
  modelType: ModelType;
  epochs: number;
  batchSize: number;
  sequenceLength: number;
  loss: LossFunction;
  dropout: number;
  learningRate: number;
  featureToggles: {
    technical: boolean;
    sentiment: boolean;
    regime: boolean;
    support_resistance: boolean;
  };
}

interface TrainingConfigFormProps {
  onSubmit: (config: TrainingConfig) => void;
  isSubmitting: boolean;
  disabled?: boolean;
}

const MODEL_TYPES: { id: ModelType; label: string; description: string }[] = [
  { id: "lstm_transformer", label: "LSTM + Transformer", description: "Deep learning hybrid model (recommended)" },
  { id: "gbm", label: "Gradient Boosting", description: "XGBoost/LightGBM ensemble" },
  { id: "stacking", label: "Stacking Ensemble", description: "Meta-learner combining all models" },
];

const LOSS_FUNCTIONS: { id: LossFunction; label: string; description: string }[] = [
  { id: "huber", label: "Huber", description: "Robust to outliers" },
  { id: "mae", label: "MAE", description: "Mean absolute error" },
  { id: "balanced", label: "Balanced", description: "Directional + magnitude" },
  { id: "quantile", label: "Quantile", description: "Uncertainty estimation" },
];

const DEFAULT_CONFIG: TrainingConfig = {
  symbol: "AAPL",
  modelType: "lstm_transformer",
  epochs: 50,
  batchSize: 512,
  sequenceLength: 90,
  loss: "balanced",
  dropout: 0.3,
  learningRate: 0.001,
  featureToggles: {
    technical: true,
    sentiment: true,
    regime: true,
    support_resistance: true,
  },
};

export function TrainingConfigForm({ onSubmit, isSubmitting, disabled }: TrainingConfigFormProps) {
  const [config, setConfig] = useState<TrainingConfig>(DEFAULT_CONFIG);
  const [showAdvanced, setShowAdvanced] = useState(false);

  const updateConfig = <K extends keyof TrainingConfig>(key: K, value: TrainingConfig[K]) => {
    setConfig((prev) => ({ ...prev, [key]: value }));
  };

  const toggleFeature = (feature: keyof TrainingConfig["featureToggles"]) => {
    setConfig((prev) => ({
      ...prev,
      featureToggles: {
        ...prev.featureToggles,
        [feature]: !prev.featureToggles[feature],
      },
    }));
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!config.symbol.trim()) return;
    onSubmit(config);
  };

  return (
    <form onSubmit={handleSubmit} className="space-y-6">
      {/* Basic Settings */}
      <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5">
        <div className="flex items-center gap-2 mb-4">
          <GraduationCap className="size-5 text-yellow-400" />
          <h3 className="text-lg font-semibold text-white">Training Configuration</h3>
        </div>

        <div className="grid gap-4 sm:grid-cols-2">
          <div className="space-y-2">
            <Label htmlFor="symbol" className="text-gray-400 text-sm">Stock Symbol</Label>
            <Input
              id="symbol"
              type="text"
              value={config.symbol}
              onChange={(e) => updateConfig("symbol", e.target.value.toUpperCase())}
              placeholder="AAPL"
              className="h-11 bg-gray-950 border-gray-700 text-white font-mono"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="epochs" className="text-gray-400 text-sm">Epochs</Label>
            <Input
              id="epochs"
              type="number"
              min={10}
              max={200}
              value={config.epochs}
              onChange={(e) => updateConfig("epochs", parseInt(e.target.value) || 50)}
              className="h-11 bg-gray-950 border-gray-700 text-white"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="batchSize" className="text-gray-400 text-sm">Batch Size</Label>
            <Input
              id="batchSize"
              type="number"
              min={32}
              max={2048}
              step={32}
              value={config.batchSize}
              onChange={(e) => updateConfig("batchSize", parseInt(e.target.value) || 512)}
              className="h-11 bg-gray-950 border-gray-700 text-white"
            />
          </div>

          <div className="space-y-2">
            <Label htmlFor="sequenceLength" className="text-gray-400 text-sm">Sequence Length (days)</Label>
            <Input
              id="sequenceLength"
              type="number"
              min={30}
              max={180}
              value={config.sequenceLength}
              onChange={(e) => updateConfig("sequenceLength", parseInt(e.target.value) || 90)}
              className="h-11 bg-gray-950 border-gray-700 text-white"
            />
          </div>
        </div>
      </div>

      {/* Model Type */}
      <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5">
        <Label className="text-gray-400 text-sm mb-3 block">Model Architecture</Label>
        <div className="grid gap-3 sm:grid-cols-3">
          {MODEL_TYPES.map((type) => (
            <button
              key={type.id}
              type="button"
              onClick={() => updateConfig("modelType", type.id)}
              className={cn(
                "flex flex-col items-start gap-1 rounded-lg border p-4 text-left transition-all",
                config.modelType === type.id
                  ? "border-yellow-500/50 bg-yellow-500/10 text-yellow-200"
                  : "border-gray-700 text-gray-400 hover:border-gray-600"
              )}
            >
              <span className="font-medium">{type.label}</span>
              <span className="text-xs text-gray-500">{type.description}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Loss Function */}
      <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5">
        <Label className="text-gray-400 text-sm mb-3 block">Loss Function</Label>
        <div className="grid gap-2 grid-cols-2 sm:grid-cols-4">
          {LOSS_FUNCTIONS.map((loss) => (
            <button
              key={loss.id}
              type="button"
              onClick={() => updateConfig("loss", loss.id)}
              className={cn(
                "flex flex-col items-center gap-1 rounded-lg border p-3 text-center transition-all",
                config.loss === loss.id
                  ? "border-yellow-500/50 bg-yellow-500/10 text-yellow-200"
                  : "border-gray-700 text-gray-400 hover:border-gray-600"
              )}
            >
              <span className="text-sm font-medium">{loss.label}</span>
              <span className="text-[10px] text-gray-500">{loss.description}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Feature Toggles */}
      <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5">
        <Label className="text-gray-400 text-sm mb-3 block">Feature Categories</Label>
        <div className="grid gap-2 grid-cols-2 sm:grid-cols-4">
          {(Object.keys(config.featureToggles) as Array<keyof TrainingConfig["featureToggles"]>).map((feature) => (
            <button
              key={feature}
              type="button"
              onClick={() => toggleFeature(feature)}
              className={cn(
                "rounded-lg border px-4 py-3 text-sm font-medium transition-all capitalize",
                config.featureToggles[feature]
                  ? "border-emerald-500/50 bg-emerald-500/10 text-emerald-400"
                  : "border-gray-700 text-gray-500 hover:border-gray-600"
              )}
            >
              {feature.replace("_", " ")}
            </button>
          ))}
        </div>
        <p className="text-xs text-gray-500 mt-2 flex items-center gap-1">
          <Info className="size-3" />
          {Object.values(config.featureToggles).filter(Boolean).length * 40}+ features enabled
        </p>
      </div>

      {/* Advanced Settings */}
      <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5">
        <button
          type="button"
          onClick={() => setShowAdvanced(!showAdvanced)}
          className="flex items-center gap-2 text-sm text-gray-400 hover:text-gray-200 transition"
        >
          <span>{showAdvanced ? "▼" : "▶"}</span>
          <span>Advanced Settings</span>
        </button>

        {showAdvanced && (
          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            <div className="space-y-2">
              <Label htmlFor="dropout" className="text-gray-400 text-sm">Dropout Rate</Label>
              <Input
                id="dropout"
                type="number"
                min={0.1}
                max={0.5}
                step={0.05}
                value={config.dropout}
                onChange={(e) => updateConfig("dropout", parseFloat(e.target.value) || 0.3)}
                className="h-11 bg-gray-950 border-gray-700 text-white"
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="learningRate" className="text-gray-400 text-sm">Learning Rate</Label>
              <Input
                id="learningRate"
                type="number"
                min={0.0001}
                max={0.01}
                step={0.0001}
                value={config.learningRate}
                onChange={(e) => updateConfig("learningRate", parseFloat(e.target.value) || 0.001)}
                className="h-11 bg-gray-950 border-gray-700 text-white"
              />
            </div>
          </div>
        )}
      </div>

      {/* Submit Button */}
      <div className="flex items-center gap-4">
        <Button
          type="submit"
          disabled={!config.symbol.trim() || isSubmitting || disabled}
          className="flex-1 h-12 bg-gradient-to-b from-yellow-400 to-yellow-500 text-gray-900 font-semibold text-base hover:from-yellow-300 hover:to-yellow-400 disabled:opacity-50"
        >
          {isSubmitting ? (
            <>
              <Loader2 className="size-5 animate-spin mr-2" />
              Starting Training...
            </>
          ) : (
            <>
              <Cpu className="size-5 mr-2" />
              Start Training
            </>
          )}
        </Button>
      </div>

      {disabled && (
        <div className="flex items-center gap-2 text-yellow-400 text-sm">
          <AlertCircle className="size-4" />
          <span>A training job is already in progress</span>
        </div>
      )}
    </form>
  );
}

export type { TrainingConfig };
