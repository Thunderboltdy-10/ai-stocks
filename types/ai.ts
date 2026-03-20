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
  modelVariant?: string;
  dataInterval?: string;
  dataPeriod?: string;
  qualityGatePassed?: boolean;
  qualityScore?: number;
  qualityReasons?: string[];
  holdoutMetricSource?: string;
  holdoutPredTargetStdRatio?: number;
  featureProfile?: string;
  featureSelectionMode?: string;
  targetHorizonDays?: number;
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
export type TrainingWorkflow = "single" | "daily_research" | "intraday_research" | "full_research";

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
  workflow?: TrainingWorkflow;
  dropout?: number;
  learningRate?: number;
  nTrials?: number;
  maxFeatures?: number;
  targetHorizons?: number[];
  featureProfiles?: string[];
  featureSelectionModes?: string[];
  symbolSet?: string;
  dailySymbols?: string;
  intradaySymbols?: string;
  dailySymbolSet?: string;
  intradaySymbolSet?: string;
  dailyPeriod?: string;
  intradayPeriod?: string;
  dailyInterval?: string;
  intradayInterval?: string;
  dataPeriod?: string;
  dataInterval?: string;
  useLgb?: boolean;
  overwrite?: boolean;
  allowCpuFallback?: boolean;
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
  modelType?: string;
  workflow?: string;
  currentStep?: string;
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
  timestamp?: string;
  message?: string;
  progress?: number;
  status?: "running" | "completed" | "failed";
  step?: string;
  type?: "started" | "stage" | "log" | "completed" | "failed" | "cancelled" | "heartbeat";
  command?: string;
  error?: string;
  summary_path?: string;
  daily_gate_passed?: boolean;
  intraday_gate_passed?: boolean;
}

export interface ResearchRunSummary {
  id: string;
  workflow: string;
  status: string;
  startedAt: string;
  completedAt?: string;
  dailyGatePassed?: boolean;
  intradayGatePassed?: boolean;
  daily?: Record<string, unknown>;
  intraday?: Record<string, unknown>;
  summaryPath: string;
  runDir: string;
}

export interface PredictionParams {
  symbol: string;
  modelId?: string;
  modelVariant?: string;
  dataInterval?: string;
  dataPeriod?: string;
  maxLong?: number;
  maxShort?: number;
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
    lowerBand?: number[];
    upperBand?: number[];
    positions: number[];
    actions?: Array<{
      date: string;
      action: "BUY" | "SELL" | "SHORT" | "COVER" | "HOLD";
      price: number;
      targetPosition: number;
    }>;
  };
  metadata: {
    modelId: string;
    fusionMode: FusionMode;
    buyThreshold: number;
    sellThreshold: number;
    horizon: number;
    tradeShareFloor?: number;
    modelQualityGatePassed?: boolean;
    modelQualityScore?: number;
    modelQualityReasons?: string[];
    holdoutMetricSource?: string;
    holdoutPredTargetStdRatio?: number;
    featureProfile?: string;
    featureSelectionMode?: string;
    targetHorizonDays?: number;
    executionMode?: string;
    maxLong?: number;
    maxShort?: number;
    mlOverlayWeight?: number;
    directionConfidenceMean?: number;
    modelVariantRequested?: string;
    modelVariant?: string;
    dataInterval?: string;
    dataPeriod?: string;
    isIntraday?: boolean;
    annualizationFactor?: number;
    flatAtDayEnd?: boolean;
    dayEndFlattenFraction?: number;
    recommendedMinPositionChange?: number;
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
  forwardWindow: number;
  initialCapital: number;
  maxLong: number;
  maxShort: number;
  dataInterval?: string;
  annualizationFactor?: number;
  flatAtDayEnd?: boolean;
  dayEndFlattenFraction?: number;
  minPositionChange?: number;
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
  diagnostics?: BacktestDiagnostics;
  annotations: TradeMarker[];
  buyHoldEquity?: Array<{ date: string; equity: number }>;
  csv?: string;
  forwardSimulation?: ForwardSimResult;
  windowStart?: string;
  windowEnd?: string;
}

export interface BacktestDiagnostics {
  rolling: Array<{
    date: string;
    equity: number;
    buyHoldEquity: number;
    drawdown: number;
    rollingSharpe: number;
    rollingAlpha: number;
    position: number;
    strategyReturn: number;
    buyHoldReturn: number;
  }>;
  monthly: Array<{
    period: string;
    strategyReturn: number;
    buyHoldReturn: number;
    alpha: number;
  }>;
  actionBreakdown: Array<{
    action: string;
    count: number;
    winRate: number;
    avgPnl: number;
    totalPnl: number;
  }>;
  hourly: Array<{
    hour: number;
    avgStrategyReturn: number;
    avgPosition: number;
    tradeCount: number;
  }>;
  risk: {
    exposureMean: number;
    exposureStd: number;
    tailLossP95: number;
    cvar95: number;
    bestBar: number;
    worstBar: number;
  };
}

export interface ForwardSimResult {
  dates: string[];
  prices: number[];
  equityCurve: number[];
  source?: "historical_holdout";
  windowStart?: string;
  windowEnd?: string;
  cumulativeReturn?: number;
  sharpe: number;
  maxDrawdown: number;
  trades: number;
  winRate?: number;
  profitFactor?: number;
  actions?: Array<{
    id: string;
    date: string;
    action:
      | "BUY"
      | "SELL"
      | "SHORT"
      | "COVER"
      | "COVER_BUY"
      | "SELL_SHORT"
      | "EOD_FLAT_SELL"
      | "EOD_FLAT_COVER"
      | "EOD_REDUCE_SELL"
      | "EOD_REDUCE_COVER";
    price: number;
    shares: number;
    targetWeight: number;
    effectiveWeightAfter: number;
    notes: string;
  }>;
  markers?: TradeMarker[];
  totalCosts?: number;
  borrowFee?: number;
}

export interface BenchmarkModeSummary {
  mode: "daily" | "intraday";
  symbolSet: string;
  settings: Record<string, unknown>;
  symbolsTested: number;
  variantsEvaluated: number;
  windowsEvaluated: number;
  aggregate: {
    meanAlpha: number;
    medianAlpha: number;
    positiveAlphaRate: number;
    meanSharpe: number;
    medianSharpe: number;
    meanStrategyReturn: number;
    meanBuyHoldReturn: number;
    meanMlLiftVsRegime: number;
    qualityPassRate: number;
  };
  leaders: Array<{
    symbol: string;
    variant: string;
    mean_alpha: number;
    mean_sharpe: number;
    mean_strategy_return: number;
    mean_buy_hold_return: number;
    mean_ml_lift: number;
    positive_alpha_rate: number;
    windows: number;
    quality_score: number;
    quality_pass_rate: number;
  }>;
  laggards: Array<{
    symbol: string;
    variant: string;
    mean_alpha: number;
    mean_sharpe: number;
    mean_strategy_return: number;
    mean_buy_hold_return: number;
    mean_ml_lift: number;
    positive_alpha_rate: number;
    windows: number;
    quality_score: number;
    quality_pass_rate: number;
  }>;
  qualityLeaders: Array<{
    symbol: string;
    variant: string;
    feature_profile?: string;
    feature_selection_mode?: string;
    target_horizon_days?: number;
    quality_gate_passed?: boolean;
    quality_score?: number;
    holdout_net_return?: number;
    holdout_net_sharpe?: number;
    wfe?: number;
  }>;
}

export interface BenchmarkRun {
  id: string;
  createdAt: string;
  settings: Record<string, unknown>;
  modes: Partial<Record<"daily" | "intraday", BenchmarkModeSummary>>;
}

export interface TradeRecord {
  id: string;
  date: string;
  action:
    | "BUY"
    | "SELL"
    | "SHORT"
    | "COVER"
    | "COVER_BUY"
    | "SELL_SHORT"
    | "EOD_FLAT_SELL"
    | "EOD_FLAT_COVER"
    | "EOD_REDUCE_SELL"
    | "EOD_REDUCE_COVER";
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
  cumulativeReturnPct?: number;
  buyHoldReturnPct?: number;
  excessReturnPct?: number;
  sharpeRatio: number;
  sortinoRatio?: number;
  calmarRatio?: number;
  maxDrawdown: number;
  winRate: number;
  averageTradeProfit: number;
  totalTrades: number;
  directionalAccuracy: number;
  correlation: number;
  smape?: number;
  rmse?: number;
  transactionCosts?: number;
  borrowFee?: number;
  profitFactor?: number;
  turnover?: number;
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
