"use client";

import { useMemo, useState } from "react";
import { BacktestResult, PredictionResult, TradeRecord } from "@/types/ai";
import { formatCurrency, formatPercent, downloadJson, downloadTextFile } from "@/utils/formatters";
import { toCsv } from "@/utils/metrics";
import { Button } from "@/components/ui/button";
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogTrigger } from "@/components/ui/dialog";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Badge } from "@/components/ui/badge";

interface BacktestResultsProps {
  backtest?: BacktestResult | null;
  prediction?: PredictionResult | null;
  onExportChart?: () => void;
}

const METRIC_LABELS: Array<{ key: keyof BacktestResult["metrics"]; label: string; tooltip: string; formatter?: (value: number) => string }> = [
  { key: "cumulativeReturn", label: "Cumulative Return", tooltip: "Net return over the backtest window", formatter: (value) => formatPercent(value) },
  { key: "sharpeRatio", label: "Sharpe", tooltip: "Annualized Sharpe ratio" },
  { key: "maxDrawdown", label: "Max Drawdown", tooltip: "Worst peak-to-trough drop", formatter: (value) => formatPercent(value) },
  { key: "winRate", label: "Win Rate", tooltip: "% of profitable trades", formatter: (value) => formatPercent(value) },
  { key: "averageTradeProfit", label: "Avg Trade", tooltip: "Average PnL per trade", formatter: (value) => formatPercent(value) },
  { key: "totalTrades", label: "Trades", tooltip: "Total executed trades" },
  { key: "directionalAccuracy", label: "Directional Accuracy", tooltip: "% times sign matched", formatter: (value) => formatPercent(value) },
  { key: "correlation", label: "Correlation", tooltip: "Correlation between prediction and move" },
];

export function BacktestResults({ backtest, prediction, onExportChart }: BacktestResultsProps) {
  const [selectedTrade, setSelectedTrade] = useState<TradeRecord | null>(null);

  const csvContent = useMemo(() => {
    if (!backtest?.tradeLog.length) return "";
    return toCsv(
      backtest.tradeLog.map((trade) => ({
        date: trade.date,
        action: trade.action,
        price: trade.price,
        shares: trade.shares,
        position: trade.position,
        pnl: trade.pnl,
        cumulative_pnl: trade.cumulativePnl,
      }))
    );
  }, [backtest?.tradeLog]);

  const handleExportCsv = () => {
    if (!csvContent) return;
    downloadTextFile(`backtest-${prediction?.symbol ?? "model"}.csv`, csvContent);
  };

  const handleExportMetadata = () => {
    if (!prediction) return;
    downloadJson(`model-${prediction.metadata.modelId}.json`, prediction.metadata);
  };

  if (!backtest) {
    return (
      <div className="rounded-xl border border-dashed border-gray-800 bg-gray-900/30 p-6 text-center text-sm text-gray-500">
        Backtest metrics will appear after running a simulation.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        {METRIC_LABELS.map((metric) => (
          <div key={metric.key as string} className="rounded-xl border border-gray-800 bg-gray-900/60 p-4" aria-label={metric.tooltip}>
            <p className="text-xs text-gray-500">{metric.label}</p>
            <p className="text-2xl font-semibold text-gray-100">
              {metric.formatter
                ? metric.formatter(backtest.metrics[metric.key] as number)
                : typeof backtest.metrics[metric.key] === "number"
                ? (backtest.metrics[metric.key] as number).toFixed(2)
                : String(backtest.metrics[metric.key])}
            </p>
          </div>
        ))}
      </div>

      <div className="flex flex-wrap gap-2">
        <Button variant="outline" className="border-gray-700 text-gray-200" onClick={handleExportCsv} disabled={!csvContent}>
          Export Backtest CSV
        </Button>
        <Button variant="outline" className="border-gray-700 text-gray-200" onClick={onExportChart}>
          Export Chart PNG
        </Button>
        <Button variant="outline" className="border-gray-700 text-gray-200" onClick={handleExportMetadata}>
          Download Model Metadata
        </Button>
        {backtest.annotations?.length && (
          <Button variant="outline" className="border-gray-700 text-gray-200" onClick={() => downloadJson("annotations.json", backtest.annotations)}>
            Export Annotations JSON
          </Button>
        )}
      </div>

      <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-gray-200">Trade Log</h3>
          <p className="text-xs text-gray-500">{backtest.tradeLog.length} trades</p>
        </div>
        <div className="mt-3 max-h-72 overflow-y-auto">
          <Table>
            <TableHeader>
              <TableRow className="border-gray-800">
                <TableHead>Date</TableHead>
                <TableHead>Action</TableHead>
                <TableHead>Price</TableHead>
                <TableHead>Shares</TableHead>
                <TableHead>Position</TableHead>
                <TableHead>PnL</TableHead>
                <TableHead>Cumulative</TableHead>
                <TableHead>Explain</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {backtest.tradeLog.map((trade) => (
                <TableRow key={trade.id} className="border-gray-900/40 text-sm text-gray-200">
                  <TableCell>{new Date(trade.date).toLocaleDateString()}</TableCell>
                  <TableCell>
                    <Badge variant={trade.action === "BUY" ? "default" : "destructive"}>{trade.action}</Badge>
                  </TableCell>
                  <TableCell>{formatCurrency(trade.price)}</TableCell>
                  <TableCell>{trade.shares.toFixed(2)}</TableCell>
                  <TableCell>{trade.position.toFixed(2)}</TableCell>
                  <TableCell className={trade.pnl >= 0 ? "text-green-400" : "text-red-400"}>{formatCurrency(trade.pnl)}</TableCell>
                  <TableCell>{formatCurrency(trade.cumulativePnl)}</TableCell>
                  <TableCell>
                    {trade.explanation ? (
                      <Dialog>
                        <DialogTrigger asChild>
                          <Button variant="ghost" size="sm" onClick={() => setSelectedTrade(trade)}>
                            Details
                          </Button>
                        </DialogTrigger>
                        <DialogContent className="bg-gray-900 text-gray-100">
                          <DialogHeader>
                            <DialogTitle>Trade explanation</DialogTitle>
                          </DialogHeader>
                          <div className="space-y-2 text-sm">
                            <p>Classifier probability: {formatPercent(trade.explanation.classifierProb)}</p>
                            <p>Regressor return: {formatPercent(trade.explanation.regressorReturn)}</p>
                            <p>Fusion mode: {trade.explanation.fusionMode}</p>
                            {trade.explanation.notes && <p className="text-gray-400">{trade.explanation.notes}</p>}
                          </div>
                        </DialogContent>
                      </Dialog>
                    ) : (
                      <span className="text-xs text-gray-500">—</span>
                    )}
                  </TableCell>
                </TableRow>
              ))}
            </TableBody>
          </Table>
        </div>
      </div>

      {prediction && (
        <div className="rounded-xl border border-emerald-700/50 bg-emerald-500/10 p-4 text-sm text-emerald-100">
          <p className="text-xs uppercase tracking-wide text-emerald-300">Suggested Action</p>
          {(() => {
            const lastPosition = prediction.fusedPositions[prediction.fusedPositions.length - 1] ?? 0;
            const lastProb = prediction.classifierProbabilities[prediction.classifierProbabilities.length - 1];
            const action = lastPosition > 0.05 ? "Buy" : lastPosition < -0.05 ? "Sell" : "Hold";
            return (
              <div className="mt-2">
                <p className="text-xl font-semibold">{action}</p>
                {lastProb && (
                  <p className="text-xs text-gray-100">
                    Classifier buy={formatPercent(lastProb.buy)} · sell={formatPercent(lastProb.sell)}
                  </p>
                )}
              </div>
            );
          })()}
        </div>
      )}
    </div>
  );
}
