"use client";

import { useMemo } from "react";
import { ModelMeta, PredictionParams, FusionSettings, BacktestParams, FusionMode } from "@/types/ai";
import { formatDateLabel } from "@/utils/formatters";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { cn } from "@/lib/utils";

interface ModelControlPanelProps {
  models: ModelMeta[] | undefined;
  modelsLoading: boolean;
  selectedModelId?: string;
  onSelectModel: (modelId: string) => void;
  predictionParams: PredictionParams;
  onChangePredictionParams: (partial: Partial<PredictionParams>) => void;
  fusionSettings: FusionSettings;
  onChangeFusionSettings: (partial: Partial<FusionSettings>) => void;
  backtestParams: BacktestParams;
  onChangeBacktestParams: (partial: Partial<BacktestParams>) => void;
  onApplyPreset: (preset: "conservative" | "balanced" | "aggressive") => void;
  onPredict: () => void;
  onBacktest: () => void;
  disablePredict?: boolean;
  disableBacktest?: boolean;
}

const PRESET_LABELS: Record<"conservative" | "balanced" | "aggressive", { title: string; description: string }> = {
  conservative: { title: "Conservative", description: "Classifier fusion, higher buy floor" },
  balanced: { title: "Balanced", description: "Weighted fusion, neutral thresholds" },
  aggressive: { title: "Aggressive", description: "Regressor fusion, relaxed thresholds" },
};

export function ModelControlPanel({
  models,
  modelsLoading,
  selectedModelId,
  onSelectModel,
  predictionParams,
  onChangePredictionParams,
  fusionSettings,
  onChangeFusionSettings,
  backtestParams,
  onChangeBacktestParams,
  onApplyPreset,
  onPredict,
  onBacktest,
  disablePredict,
  disableBacktest,
}: ModelControlPanelProps) {
  const selectedModel = useMemo(() => models?.find((model) => model.id === selectedModelId), [models, selectedModelId]);

  const renderModelRegistry = () => (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-4">
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-sm font-semibold text-gray-200">Model Registry</h3>
          <p className="text-xs text-gray-500">Select a trained model to load metadata and defaults.</p>
        </div>
        {modelsLoading && <span className="text-xs text-yellow-500">Refreshing…</span>}
      </div>
      <div className="mt-3 space-y-2 max-h-48 overflow-y-auto pr-1">
        {models?.map((model) => (
          <button
            key={model.id}
            onClick={() => onSelectModel(model.id)}
            className={cn(
              "w-full rounded-lg border px-3 py-2 text-left transition",
              selectedModelId === model.id
                ? "border-yellow-500 bg-yellow-500/10 text-yellow-200"
                : "border-gray-800 bg-gray-900 text-gray-300 hover:border-gray-700"
            )}
            aria-pressed={selectedModelId === model.id ? "true" : "false"}
          >
            <p className="text-sm font-semibold">{model.symbol}</p>
            <p className="text-xs text-gray-400">{formatDateLabel(model.createdAt)} · {model.fusionModeDefault}</p>
            <p className="text-[11px] text-gray-500">Sharpe {model.metrics.sharpeRatio.toFixed(2)} · Win {model.metrics.winRate.toFixed(2)}%</p>
          </button>
        ))}
        {!models?.length && <p className="text-xs text-gray-500">No models registered yet.</p>}
      </div>
      {selectedModel && (
        <div className="mt-4 rounded-lg bg-gray-950/40 p-3 text-xs text-gray-400">
          <p>Sequence length: {selectedModel.sequenceLength} · Ensemble: {selectedModel.ensembleSize}</p>
          <p>Directional accuracy: {(selectedModel.metrics.directionalAccuracy * 100).toFixed(1)}%</p>
          {selectedModel.notes && <p className="mt-1 text-gray-500">{selectedModel.notes}</p>}
        </div>
      )}
    </div>
  );

  const renderQuickPresets = () => (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-4">
      <h3 className="text-sm font-semibold text-gray-200">Quick Run Presets</h3>
      <div className="mt-3 grid grid-cols-1 gap-2">
        {Object.entries(PRESET_LABELS).map(([key, value]) => (
          <Button key={key} variant="outline" className="justify-between border-gray-700 text-left text-gray-300 hover:text-white" onClick={() => onApplyPreset(key as keyof typeof PRESET_LABELS)}>
            <span>
              <span className="block text-sm font-semibold">{value.title}</span>
              <span className="text-xs text-gray-500">{value.description}</span>
            </span>
            <span className="text-xs text-gray-500">⌘{key.charAt(0).toUpperCase()}</span>
          </Button>
        ))}
      </div>
    </div>
  );

  const renderPredictionControls = () => (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-200">Prediction & Fusion</h3>
        <span className="text-xs text-gray-500">Shortcut: I</span>
      </div>
      <div className="mt-3 space-y-3 text-xs text-gray-400">
        <div>
          <Label>Symbol</Label>
          <Input
            value={predictionParams.symbol}
            onChange={(event) => onChangePredictionParams({ symbol: event.target.value.toUpperCase() })}
            placeholder="AAPL"
            aria-label="Prediction symbol"
          />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <Label>Horizon (days)</Label>
            <Input type="number" value={predictionParams.horizon} min={1} onChange={(event) => onChangePredictionParams({ horizon: Number(event.target.value) })} />
          </div>
          <div>
            <Label>Chart Days</Label>
            <Input type="number" value={predictionParams.daysOnChart} min={10} onChange={(event) => onChangePredictionParams({ daysOnChart: Number(event.target.value) })} />
          </div>
        </div>
        <div>
          <Label>Smoothing</Label>
          <Select value={predictionParams.smoothing} onValueChange={(value) => onChangePredictionParams({ smoothing: value as PredictionParams["smoothing"] })}>
            <SelectTrigger className="h-9 border-gray-700 bg-gray-900 text-gray-300">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              <SelectItem value="none">None</SelectItem>
              <SelectItem value="ema">EMA</SelectItem>
              <SelectItem value="moving-average">Moving Average</SelectItem>
            </SelectContent>
          </Select>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <Label>Buy Confidence Floor</Label>
            <Input type="number" value={predictionParams.confidenceFloors.buy} min={0} max={1} step={0.01} onChange={(event) => onChangePredictionParams({
              confidenceFloors: { ...predictionParams.confidenceFloors, buy: Number(event.target.value) },
            })} />
          </div>
          <div>
            <Label>Sell Confidence Floor</Label>
            <Input type="number" value={predictionParams.confidenceFloors.sell} min={0} max={1} step={0.01} onChange={(event) => onChangePredictionParams({
              confidenceFloors: { ...predictionParams.confidenceFloors, sell: Number(event.target.value) },
            })} />
          </div>
        </div>
        <div>
          <Label>Trade Share Floor (%)</Label>
          <Input
            type="number"
            min={0}
            max={100}
            step={1}
            value={predictionParams.tradeShareFloor ?? 0}
            onChange={(event) => onChangePredictionParams({ tradeShareFloor: Number(event.target.value) })}
          />
          <p className="mt-1 text-[11px] text-gray-500">Ignore trades smaller than this percent of the baseline size.</p>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <Label>Fusion Mode</Label>
            <Select value={fusionSettings.mode} onValueChange={(value) => onChangeFusionSettings({ mode: value as FusionMode })}>
              <SelectTrigger className="h-9 border-gray-700 bg-gray-900 text-gray-300">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="classifier">Classifier</SelectItem>
                <SelectItem value="weighted">Weighted</SelectItem>
                <SelectItem value="hybrid">Hybrid</SelectItem>
                <SelectItem value="regressor">Regressor</SelectItem>
              </SelectContent>
            </Select>
          </div>
          <div>
            <Label>Regressor Scale</Label>
            <Input type="number" value={fusionSettings.regressorScale} step={0.5} onChange={(event) => onChangeFusionSettings({ regressorScale: Number(event.target.value) })} />
          </div>
          <div>
            <Label>Buy Threshold</Label>
            <Input type="number" value={fusionSettings.buyThreshold} step={0.01} onChange={(event) => onChangeFusionSettings({ buyThreshold: Number(event.target.value) })} />
          </div>
          <div>
            <Label>Sell Threshold</Label>
            <Input type="number" value={fusionSettings.sellThreshold} step={0.01} onChange={(event) => onChangeFusionSettings({ sellThreshold: Number(event.target.value) })} />
          </div>
        </div>
        <div className="flex items-center justify-between text-xs text-gray-400">
          <label className="flex items-center gap-2">
            <input type="checkbox" className="accent-yellow-500" checked={fusionSettings.regimeFilters.bull} onChange={(event) => onChangeFusionSettings({
              regimeFilters: { ...fusionSettings.regimeFilters, bull: event.target.checked },
            })} />
            Bull filter
          </label>
          <label className="flex items-center gap-2">
            <input type="checkbox" className="accent-yellow-500" checked={fusionSettings.regimeFilters.bear} onChange={(event) => onChangeFusionSettings({
              regimeFilters: { ...fusionSettings.regimeFilters, bear: event.target.checked },
            })} />
            Bear filter
          </label>
        </div>
        <Button disabled={disablePredict} onClick={onPredict} className="w-full bg-emerald-500 text-gray-900 hover:bg-emerald-400">
          Run Prediction
        </Button>
      </div>
    </div>
  );

  const renderBacktestControls = () => (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-200">Backtest & Forward Sim</h3>
        <span className="text-xs text-gray-500">Shortcut: F</span>
      </div>
      <div className="mt-3 space-y-3 text-xs text-gray-400">
        <div className="grid grid-cols-2 gap-3">
          <div>
            <Label>Backtest Window</Label>
            <Input type="number" value={backtestParams.backtestWindow} min={10} onChange={(event) => onChangeBacktestParams({ backtestWindow: Number(event.target.value) })} />
          </div>
          <div>
            <Label>Initial Capital</Label>
            <Input type="number" value={backtestParams.initialCapital} onChange={(event) => onChangeBacktestParams({ initialCapital: Number(event.target.value) })} />
          </div>
          <div>
            <Label>Max Long</Label>
            <Input type="number" value={backtestParams.maxLong} step={0.1} onChange={(event) => onChangeBacktestParams({ maxLong: Number(event.target.value) })} />
          </div>
          <div>
            <Label>Max Short</Label>
            <Input type="number" value={backtestParams.maxShort} step={0.1} onChange={(event) => onChangeBacktestParams({ maxShort: Number(event.target.value) })} />
          </div>
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <Label>Commission (bps)</Label>
            <Input type="number" value={backtestParams.commission} step={0.1} onChange={(event) => onChangeBacktestParams({ commission: Number(event.target.value) })} />
          </div>
          <div>
            <Label>Slippage (bps)</Label>
            <Input type="number" value={backtestParams.slippage} step={0.1} onChange={(event) => onChangeBacktestParams({ slippage: Number(event.target.value) })} />
          </div>
        </div>
        <label className="flex items-center gap-2 text-xs">
          <input
            type="checkbox"
            className="accent-yellow-500"
            checked={backtestParams.enableForwardSim}
            onChange={(event) => onChangeBacktestParams({ enableForwardSim: event.target.checked })}
          />
          Enable forward simulation preview
        </label>
        <Button disabled={disableBacktest} onClick={onBacktest} className="w-full bg-blue-500 text-gray-900 hover:bg-blue-400">
          Run Backtest
        </Button>
      </div>
    </div>
  );

  return (
    <div className="space-y-4" aria-label="Model control panel">
      {renderModelRegistry()}
      {renderQuickPresets()}
      {renderPredictionControls()}
      {renderBacktestControls()}
    </div>
  );
}
