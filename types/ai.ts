export interface ModelMeta {
  id: string;
  symbol: string;
  createdAt: string;
  fusionModeDefault: FusionMode;
  metrics: BacktestMetrics;
  scalers?: { feature: string; target: string };
  sequenceLength: number;
  ensembleSize: number;
  lastTrainingJobId?: string;
  notes?: string;
}

export type FusionMode =
  | "gbm_only"
  | "lstm_only"
  | "ensemble"
  | "classifier"
  | "weighted"
  | "hybrid"
  | "regressor";

export type ModelType = "gbm" | "lstm" | "ensemble" | "lstm_transformer" | "stacking";
export type LossFunction = "huber" | "mae" | "balanced" | "quantile";

export interface TrainingParams {
  symbol: string;
  epochs: number;
  batchSize: number;
  loss: LossFunction;
  sequenceLength: number;
  featureToggles: Record<string, boolean>;
  ensembleSize: number;
  baseSeed: number;
  overwriteExisting?: boolean;
  modelType?: ModelType;
  dropout?: number;
  learningRate?: number;
}

export interface TrainingJob {
  id: string;
  symbol: string;
  status: "queued" | "running" | "completed" | "failed" | "cancelled";
  progress: number;
  currentEpoch: number;
  totalEpochs: number;
  startedAt: string;
  completedAt?: string;
  error?: string;
  metrics?: TrainingMetrics;
}

export interface TrainingMetrics {
  trainLoss: number;
  valLoss: number;
  directionalAccuracy: number;
  varianceScore: number;
}

export interface EpochUpdate {
  epoch: number;
  totalEpochs: number;
  trainLoss: number;
  valLoss: number;
  learningRate: number;
  timestamp: string;
  directionalAccuracy?: number;
}

export interface TrainingJobResponse {
  jobId: string;
  modelId?: string;
}

export interface JobEvent {
  timestamp: string;
  message: string;
  progress?: number;
  status?: "running" | "completed" | "failed";
  step?: string;
}

export interface PredictionParams {
  symbol: string;
  modelId?: string;
  horizon: number;
  daysOnChart: number;
  smoothing: "none" | "ema" | "moving-average";
  confidenceFloors: {
    buy: number;
    sell: number;
  };
  tradeShareFloor?: number;
}

export interface FusionSettings {
  mode: FusionMode;
  regressorScale: number;
  buyThreshold: number;
  sellThreshold: number;
  regimeFilters: {
    bull: boolean;
    bear: boolean;
  };
}

export interface PredictionResult {
  symbol: string;
  dates: string[];
  prices: number[];
  actualReturns?: number[];
  predictedPrices: number[];
  predictedReturns: number[];
  fusedPositions: number[];
  classifierProbabilities: Array<{ buy: number; sell: number; hold: number }>;
  tradeMarkers: TradeMarker[];
  overlays: OverlaySeries[];
  candles?: Array<{ date: string; open: number; high: number; low: number; close: number; volume?: number }>;
  forecast?: {
    dates: string[];
    prices: number[];
    returns: number[];
    positions: number[];
  };
  metadata: {
    modelId: string;
    fusionMode: FusionMode;
    buyThreshold: number;
    sellThreshold: number;
    horizon: number;
    tradeShareFloor?: number;
  };
}

export interface OverlaySeries {
  type: "bollinger" | "moving-average" | "predicted-path" | "volume";
  points: Array<{ date: string; value: number }>;
  upper?: number[];
  lower?: number[];
}

export interface TradeMarker {
  date: string;
  price: number;
  type: "buy" | "sell" | "hold";
  shares: number;
  confidence: number;
  segment?: "history" | "forecast";
  scope?: "prediction" | "backtest";
  explanation?: string;
}

export interface BacktestParams {
  backtestWindow: number;
  initialCapital: number;
  maxLong: number;
  maxShort: number;
  commission: number;
  slippage: number;
  enableForwardSim: boolean;
  shortCap?: number;
}

export interface BacktestResult {
  equityCurve: Array<{ date: string; equity: number; drawdown: number }>;
  priceSeries: Array<{ date: string; price: number }>;
  tradeLog: TradeRecord[];
  metrics: BacktestMetrics;
  annotations: TradeMarker[];
  buyHoldEquity?: Array<{ date: string; equity: number }>;
  csv?: string;
  forwardSimulation?: ForwardSimResult;
}

export interface ForwardSimResult {
  dates: string[];
  prices: number[];
  equityCurve: number[];
  sharpe: number;
  maxDrawdown: number;
  trades: number;
}

export interface TradeRecord {
  id: string;
  date: string;
  action: "BUY" | "SELL";
  price: number;
  shares: number;
  position: number;
  pnl: number;
  cumulativePnl: number;
  commission?: number;
  slippage?: number;
  explanation?: {
    classifierProb: number;
    regressorReturn: number;
    fusionMode: FusionMode;
    notes?: string;
  };
}

export interface BacktestMetrics {
  cumulativeReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  averageTradeProfit: number;
  totalTrades: number;
  directionalAccuracy: number;
  correlation: number;
  smape?: number;
  rmse?: number;
}

export interface ScenarioPoint {
  date: string;
  predictedReturn: number;
  simulatedPrice: number;
  position: number;
}

export interface ScenarioBuilderState {
  buyFloor: number;
  sellFloor: number;
}
