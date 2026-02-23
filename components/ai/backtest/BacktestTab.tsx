"use client";

import { useState, useMemo, useCallback, useRef } from "react";
import { useMutation } from "@tanstack/react-query";
import { toast } from "sonner";
import {
  BacktestParams,
  BacktestResult,
  PredictionResult,
} from "@/types/ai";
import { runBacktest } from "@/lib/api-client";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { PlayCircle, Loader2, AlertCircle } from "lucide-react";

// Backtest Components
import { BacktestConfigPanel } from "./BacktestConfigPanel";
import { TransactionCostConfig } from "./TransactionCostConfig";
import { DrawdownAnalysis } from "./DrawdownAnalysis";
import { TradeAnalyzer } from "./TradeAnalyzer";
import { MetricsComparison } from "./MetricsComparison";
import { PerformanceSummary } from "./PerformanceSummary";

// Shared Components
import { LoadingOverlay } from "../shared/LoadingOverlay";
import { ExportMenu } from "../shared/ExportMenu";

// Charts
import { EquityLineChart } from "@/components/charts/EquityLineChart";
import { BacktestSummaryChart } from "@/components/charts/BacktestSummaryChart";

const DEFAULT_BACKTEST: BacktestParams = {
  backtestWindow: 60,
  initialCapital: 10_000,
  maxLong: 1,
  maxShort: 0.5,
  commission: 0.5,
  slippage: 0.2,
  enableForwardSim: false,
  shortCap: 0.5,
};

interface BacktestTabProps {
  prediction: PredictionResult | null;
  onBacktestComplete?: (backtest: BacktestResult) => void;
}

export function BacktestTab({ prediction, onBacktestComplete }: BacktestTabProps) {
  // State
  const [backtestParams, setBacktestParams] = useState<BacktestParams>(DEFAULT_BACKTEST);
  const [backtest, setBacktest] = useState<BacktestResult | null>(null);
  const [equityVisibility, setEquityVisibility] = useState({
    strategy: true,
    buyHold: true,
  });

  // Mutation
  const backtestMutation = useMutation({
    mutationFn: runBacktest,
    onSuccess: (result) => {
      setBacktest(result);
      onBacktestComplete?.(result);
      toast.success("Backtest complete");
    },
    onError: (error: Error) => {
      toast.error(error.message);
    },
  });

  // Handlers
  const handleParamsChange = useCallback((partial: Partial<BacktestParams>) => {
    setBacktestParams(prev => ({ ...prev, ...partial }));
  }, []);

  const handleRunBacktest = useCallback(async () => {
    if (!prediction) {
      toast.error("Run a prediction first");
      return;
    }
    await backtestMutation.mutateAsync({ prediction, params: backtestParams });
  }, [prediction, backtestParams, backtestMutation]);

  // Trade markers from backtest
  const backtestMarkers = useMemo(() => {
    if (!backtest?.annotations) return [];
    return backtest.annotations.map(marker => ({
      ...marker,
      segment: marker.segment ?? "history",
      scope: marker.scope ?? "backtest",
    }));
  }, [backtest?.annotations]);

  // Export handlers
  const handleExportJSON = useCallback(() => {
    if (!backtest) return;
    const blob = new Blob([JSON.stringify(backtest, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `backtest-${prediction?.symbol ?? "unknown"}-${new Date().toISOString().split("T")[0]}.json`;
    a.click();
    URL.revokeObjectURL(url);
  }, [backtest, prediction?.symbol]);

  const handleExportCSV = useCallback(() => {
    if (!backtest?.tradeLog) return;
    const headers = ["date", "action", "price", "shares", "pnl"];
    const rows = backtest.tradeLog.map(trade => [
      trade.date,
      trade.action,
      trade.price.toFixed(2),
      trade.shares,
      trade.pnl.toFixed(2),
    ]);
    const csv = [headers.join(","), ...rows.map(r => r.join(","))].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `trades-${prediction?.symbol ?? "unknown"}-${new Date().toISOString().split("T")[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  }, [backtest?.tradeLog, prediction?.symbol]);

  return (
    <div className="grid gap-6 xl:grid-cols-[360px,1fr]">
      {/* Left Panel - Configuration */}
      <aside className="space-y-4">
        {/* Prediction Status */}
        <div className={cn(
          "rounded-xl border p-4",
          prediction
            ? "border-emerald-500/30 bg-emerald-500/10"
            : "border-yellow-500/30 bg-yellow-500/10"
        )}>
          <div className="flex items-center gap-3">
            {prediction ? (
              <>
                <div className="size-10 rounded-lg bg-emerald-500/20 flex items-center justify-center">
                  <PlayCircle className="size-5 text-emerald-400" />
                </div>
                <div>
                  <p className="text-sm font-medium text-emerald-400">Ready to backtest</p>
                  <p className="text-xs text-gray-400">
                    Prediction for {prediction.symbol} loaded
                  </p>
                </div>
              </>
            ) : (
              <>
                <div className="size-10 rounded-lg bg-yellow-500/20 flex items-center justify-center">
                  <AlertCircle className="size-5 text-yellow-400" />
                </div>
                <div>
                  <p className="text-sm font-medium text-yellow-400">No prediction loaded</p>
                  <p className="text-xs text-gray-400">
                    Run a prediction first in the Prediction tab
                  </p>
                </div>
              </>
            )}
          </div>
        </div>

        {/* Backtest Config */}
        <BacktestConfigPanel
          params={backtestParams}
          onChange={handleParamsChange}
          disabled={backtestMutation.isPending}
        />

        {/* Transaction Costs */}
        <TransactionCostConfig
          params={backtestParams}
          onChange={handleParamsChange}
          disabled={backtestMutation.isPending}
        />

        {/* Run Button */}
        <Button
          onClick={handleRunBacktest}
          disabled={backtestMutation.isPending || !prediction}
          className="w-full h-12 text-lg font-semibold bg-yellow-500 hover:bg-yellow-600 text-gray-900"
        >
          {backtestMutation.isPending ? (
            <>
              <Loader2 className="size-5 mr-2 animate-spin" />
              Running Backtest...
            </>
          ) : (
            <>
              <PlayCircle className="size-5 mr-2" />
              Run Backtest
            </>
          )}
        </Button>
      </aside>

      {/* Main Content - Results */}
      <main className="space-y-6">
        {/* Header */}
        <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-6">
              <div>
                <p className="text-xs text-gray-500">Symbol</p>
                <p className="text-2xl font-bold text-white">
                  {prediction?.symbol ?? "—"}
                </p>
              </div>
              <div>
                <p className="text-xs text-gray-500">Initial Capital</p>
                <p className="text-xl font-semibold text-white">
                  ${backtestParams.initialCapital.toLocaleString()}
                </p>
              </div>
              <div>
                <p className="text-xs text-gray-500">Window</p>
                <p className="text-xl font-semibold text-white">
                  {backtestParams.backtestWindow} days
                </p>
              </div>
            </div>
            <ExportMenu
              onExportJSON={backtest ? handleExportJSON : undefined}
              onExportCSV={backtest?.tradeLog ? handleExportCSV : undefined}
              disabled={!backtest}
            />
          </div>
        </div>

        {/* Performance Summary */}
        <PerformanceSummary
          backtest={backtest}
          initialCapital={backtestParams.initialCapital}
        />

        {/* Equity Chart */}
        <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-4 relative min-h-[300px]">
          <LoadingOverlay
            isLoading={backtestMutation.isPending}
            message="Running backtest..."
            submessage="Simulating trades"
          />
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-sm font-medium text-gray-400">Equity Curve</h3>
            <div className="flex items-center gap-2">
              <button
                type="button"
                onClick={() => setEquityVisibility(prev => ({ ...prev, strategy: !prev.strategy }))}
                className={cn(
                  "px-2 py-1 text-xs rounded border transition-colors",
                  equityVisibility.strategy
                    ? "border-yellow-500/50 bg-yellow-500/10 text-yellow-200"
                    : "border-gray-700 text-gray-400"
                )}
              >
                Strategy
              </button>
              <button
                type="button"
                onClick={() => setEquityVisibility(prev => ({ ...prev, buyHold: !prev.buyHold }))}
                className={cn(
                  "px-2 py-1 text-xs rounded border transition-colors",
                  equityVisibility.buyHold
                    ? "border-blue-500/50 bg-blue-500/10 text-blue-200"
                    : "border-gray-700 text-gray-400"
                )}
              >
                Buy & Hold
              </button>
            </div>
          </div>
          <EquityLineChart
            backtest={backtest ?? undefined}
            tradeMarkers={backtestMarkers}
            showStrategyEquity={equityVisibility.strategy}
            showBuyHoldEquity={equityVisibility.buyHold}
          />
        </div>

        {/* Analysis Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Metrics Comparison */}
          <MetricsComparison
            backtest={backtest}
            initialCapital={backtestParams.initialCapital}
          />

          {/* Drawdown Analysis */}
          <DrawdownAnalysis backtest={backtest} />
        </div>

        {/* Trade Analyzer */}
        <TradeAnalyzer
          backtest={backtest}
          onExportCSV={backtest?.tradeLog ? handleExportCSV : undefined}
        />

        {/* Summary Chart */}
        {backtest && (
          <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-4">
            <h3 className="text-sm font-medium text-gray-400 mb-4">Performance Breakdown</h3>
            <BacktestSummaryChart backtest={backtest} />
          </div>
        )}
      </main>
    </div>
  );
}
