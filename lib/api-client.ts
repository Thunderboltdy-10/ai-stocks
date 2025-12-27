import {
  BacktestParams,
  BacktestResult,
  FusionSettings,
  ModelMeta,
  PredictionParams,
  PredictionResult,
  TrainingJobResponse,
  TrainingParams,
} from "@/types/ai";

const API_BASE = process.env.NEXT_PUBLIC_PYTHON_API_URL || "http://localhost:8000";

async function fetchJson<T>(input: RequestInfo, init?: RequestInit): Promise<T> {
  const response = await fetch(input, init);
  if (!response.ok) {
    const detail = await response.json().catch(() => ({}));
    const message = detail?.detail || detail?.error || response.statusText;
    throw new Error(message || "Request failed");
  }
  return response.json();
}

export function listModels(): Promise<ModelMeta[]> {
  return fetchJson<ModelMeta[]>(`${API_BASE}/api/models`, { cache: "no-store" });
}

export function trainModel(params: TrainingParams): Promise<TrainingJobResponse> {
  return fetchJson<TrainingJobResponse>(`${API_BASE}/api/models/train`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
}

export function loadModel(modelId: string): Promise<ModelMeta> {
  return fetchJson<ModelMeta>(`${API_BASE}/api/models/${modelId}`);
}

interface PredictPayload extends PredictionParams {
  fusion: FusionSettings;
}

export function runPrediction(params: PredictPayload): Promise<PredictionResult> {
  return fetchJson<PredictionResult>(`${API_BASE}/api/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(params),
  });
}

export function runBacktest(payload: {
  prediction: PredictionResult;
  params: BacktestParams;
}): Promise<BacktestResult> {
  return fetchJson<BacktestResult>(`${API_BASE}/api/backtest`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export function requestForwardSimulation(payload: {
  prediction: PredictionResult;
  params: BacktestParams;
}): Promise<BacktestResult["forwardSimulation"]> {
  return fetchJson(`${API_BASE}/api/forward_sim`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
}

export async function downloadModelArtifacts(modelId: string): Promise<Blob> {
  const response = await fetch(`${API_BASE}/api/models/${modelId}/download`);
  if (!response.ok) {
    throw new Error("Failed to download artifacts");
  }
  return response.blob();
}

export const apiBase = API_BASE;