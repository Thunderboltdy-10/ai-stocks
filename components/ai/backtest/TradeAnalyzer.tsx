"use client";

import { useMemo, useState } from "react";
import { BacktestResult, TradeRecord } from "@/types/ai";
import { cn } from "@/lib/utils";
import {
  ArrowUpDown,
  TrendingUp,
  TrendingDown,
  Filter,
  Download,
  ChevronDown,
  ChevronUp,
} from "lucide-react";
import { Button } from "@/components/ui/button";

type SortKey = "date" | "pnl" | "shares" | "price";
type FilterType = "all" | "winners" | "losers";

interface TradeAnalyzerProps {
  backtest: BacktestResult | null;
  onExportCSV?: () => void;
}

export function TradeAnalyzer({ backtest, onExportCSV }: TradeAnalyzerProps) {
  const [sortKey, setSortKey] = useState<SortKey>("date");
  const [sortAsc, setSortAsc] = useState(false);
  const [filter, setFilter] = useState<FilterType>("all");
  const [expandedTrade, setExpandedTrade] = useState<string | null>(null);

  const trades = backtest?.tradeLog ?? [];

  const filteredTrades = useMemo(() => {
    let result = [...trades];

    // Filter
    if (filter === "winners") {
      result = result.filter((t) => t.pnl > 0);
    } else if (filter === "losers") {
      result = result.filter((t) => t.pnl < 0);
    }

    // Sort
    result.sort((a, b) => {
      let aVal: number | string = 0;
      let bVal: number | string = 0;

      switch (sortKey) {
        case "date":
          aVal = new Date(a.date).getTime();
          bVal = new Date(b.date).getTime();
          break;
        case "pnl":
          aVal = a.pnl;
          bVal = b.pnl;
          break;
        case "shares":
          aVal = a.shares;
          bVal = b.shares;
          break;
        case "price":
          aVal = a.price;
          bVal = b.price;
          break;
      }

      return sortAsc ? (aVal > bVal ? 1 : -1) : aVal < bVal ? 1 : -1;
    });

    return result;
  }, [trades, filter, sortKey, sortAsc]);

  const stats = useMemo(() => {
    const winners = trades.filter((t) => t.pnl > 0);
    const losers = trades.filter((t) => t.pnl < 0);
    const totalPnL = trades.reduce((sum, t) => sum + t.pnl, 0);
    const avgWin = winners.length > 0
      ? winners.reduce((sum, t) => sum + t.pnl, 0) / winners.length
      : 0;
    const avgLoss = losers.length > 0
      ? losers.reduce((sum, t) => sum + t.pnl, 0) / losers.length
      : 0;

    return {
      total: trades.length,
      winners: winners.length,
      losers: losers.length,
      totalPnL,
      avgWin,
      avgLoss,
      profitFactor: avgLoss !== 0 ? Math.abs(avgWin / avgLoss) : 0,
    };
  }, [trades]);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(false);
    }
  };

  if (!backtest) {
    return (
      <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5">
        <h3 className="text-sm font-medium text-gray-400 mb-4">Trade Analyzer</h3>
        <div className="h-48 flex items-center justify-center">
          <p className="text-gray-500">Run a backtest to analyze trades</p>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-medium text-gray-400">Trade Analyzer</h3>
        <div className="flex items-center gap-2">
          {/* Filter Buttons */}
          <div className="flex items-center gap-1 bg-gray-800/50 rounded-lg p-1">
            {(["all", "winners", "losers"] as FilterType[]).map((f) => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={cn(
                  "px-2 py-1 text-xs rounded-md transition-colors capitalize",
                  filter === f
                    ? "bg-gray-700 text-white"
                    : "text-gray-400 hover:text-gray-200"
                )}
              >
                {f}
              </button>
            ))}
          </div>
          {onExportCSV && (
            <Button variant="outline" size="sm" onClick={onExportCSV} className="border-gray-700">
              <Download className="size-4 mr-1" />
              CSV
            </Button>
          )}
        </div>
      </div>

      {/* Stats Row */}
      <div className="grid grid-cols-4 gap-3">
        <div className="rounded-lg bg-gray-800/50 p-3 text-center">
          <p className="text-xs text-gray-500 mb-1">Total Trades</p>
          <p className="text-lg font-semibold text-white">{stats.total}</p>
        </div>
        <div className="rounded-lg bg-gray-800/50 p-3 text-center">
          <p className="text-xs text-gray-500 mb-1">Win Rate</p>
          <p className={cn(
            "text-lg font-semibold",
            stats.winners / stats.total > 0.5 ? "text-emerald-400" : "text-rose-400"
          )}>
            {stats.total > 0 ? ((stats.winners / stats.total) * 100).toFixed(0) : 0}%
          </p>
        </div>
        <div className="rounded-lg bg-gray-800/50 p-3 text-center">
          <p className="text-xs text-gray-500 mb-1">Total P&L</p>
          <p className={cn(
            "text-lg font-semibold",
            stats.totalPnL >= 0 ? "text-emerald-400" : "text-rose-400"
          )}>
            ${stats.totalPnL.toFixed(2)}
          </p>
        </div>
        <div className="rounded-lg bg-gray-800/50 p-3 text-center">
          <p className="text-xs text-gray-500 mb-1">Profit Factor</p>
          <p className={cn(
            "text-lg font-semibold",
            stats.profitFactor > 1.5 ? "text-emerald-400" : stats.profitFactor > 1 ? "text-yellow-400" : "text-rose-400"
          )}>
            {stats.profitFactor.toFixed(2)}
          </p>
        </div>
      </div>

      {/* Trade List */}
      <div className="max-h-64 overflow-y-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-gray-900">
            <tr className="border-b border-gray-800">
              <th className="text-left py-2 px-2">
                <button
                  onClick={() => handleSort("date")}
                  className="flex items-center gap-1 text-gray-500 hover:text-gray-300"
                >
                  Date <ArrowUpDown className="size-3" />
                </button>
              </th>
              <th className="text-left py-2 px-2 text-gray-500">Action</th>
              <th className="text-right py-2 px-2">
                <button
                  onClick={() => handleSort("price")}
                  className="flex items-center gap-1 text-gray-500 hover:text-gray-300 ml-auto"
                >
                  Price <ArrowUpDown className="size-3" />
                </button>
              </th>
              <th className="text-right py-2 px-2">
                <button
                  onClick={() => handleSort("shares")}
                  className="flex items-center gap-1 text-gray-500 hover:text-gray-300 ml-auto"
                >
                  Shares <ArrowUpDown className="size-3" />
                </button>
              </th>
              <th className="text-right py-2 px-2">
                <button
                  onClick={() => handleSort("pnl")}
                  className="flex items-center gap-1 text-gray-500 hover:text-gray-300 ml-auto"
                >
                  P&L <ArrowUpDown className="size-3" />
                </button>
              </th>
              <th className="w-8"></th>
            </tr>
          </thead>
          <tbody>
            {filteredTrades.map((trade) => (
              <>
                <tr
                  key={trade.id}
                  className={cn(
                    "border-b border-gray-800/50 hover:bg-gray-800/30 cursor-pointer",
                    expandedTrade === trade.id && "bg-gray-800/30"
                  )}
                  onClick={() => setExpandedTrade(expandedTrade === trade.id ? null : trade.id)}
                >
                  <td className="py-2 px-2 text-gray-300">
                    {new Date(trade.date).toLocaleDateString()}
                  </td>
                  <td className="py-2 px-2">
                    <span className={cn(
                      "flex items-center gap-1",
                      trade.action === "BUY" ? "text-emerald-400" : "text-rose-400"
                    )}>
                      {trade.action === "BUY" ? <TrendingUp className="size-3" /> : <TrendingDown className="size-3" />}
                      {trade.action}
                    </span>
                  </td>
                  <td className="py-2 px-2 text-right text-gray-300">${trade.price.toFixed(2)}</td>
                  <td className="py-2 px-2 text-right text-gray-300">{trade.shares}</td>
                  <td className={cn(
                    "py-2 px-2 text-right font-medium",
                    trade.pnl >= 0 ? "text-emerald-400" : "text-rose-400"
                  )}>
                    {trade.pnl >= 0 ? "+" : ""}{trade.pnl.toFixed(2)}
                  </td>
                  <td className="py-2 px-2">
                    {expandedTrade === trade.id ? (
                      <ChevronUp className="size-4 text-gray-500" />
                    ) : (
                      <ChevronDown className="size-4 text-gray-500" />
                    )}
                  </td>
                </tr>
                {expandedTrade === trade.id && trade.explanation && (
                  <tr className="bg-gray-800/20">
                    <td colSpan={6} className="py-3 px-4">
                      <div className="text-xs space-y-1">
                        <p className="text-gray-400">
                          <span className="text-gray-500">Fusion Mode:</span> {trade.explanation.fusionMode}
                        </p>
                        <p className="text-gray-400">
                          <span className="text-gray-500">Classifier Prob:</span> {(trade.explanation.classifierProb * 100).toFixed(0)}%
                        </p>
                        <p className="text-gray-400">
                          <span className="text-gray-500">Regressor Return:</span> {(trade.explanation.regressorReturn * 100).toFixed(2)}%
                        </p>
                        {trade.explanation.notes && (
                          <p className="text-gray-400">
                            <span className="text-gray-500">Notes:</span> {trade.explanation.notes}
                          </p>
                        )}
                      </div>
                    </td>
                  </tr>
                )}
              </>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
