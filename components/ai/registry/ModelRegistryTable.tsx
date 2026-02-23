"use client";

import { useMemo, useState } from "react";
import { cn } from "@/lib/utils";
import { ModelMeta } from "@/types/ai";
import { Button } from "@/components/ui/button";
import {
  ArrowUpDown,
  Download,
  Trash2,
  Eye,
  CheckSquare,
  Square,
  Database,
} from "lucide-react";
import { formatDistanceToNow } from "date-fns";

type SortKey = "symbol" | "createdAt" | "sharpeRatio" | "winRate" | "maxDrawdown";
type SortOrder = "asc" | "desc";

interface ModelRegistryTableProps {
  models: ModelMeta[];
  selectedIds: string[];
  onSelect: (id: string) => void;
  onView: (model: ModelMeta) => void;
  onDownload: (model: ModelMeta) => void;
  onDelete: (model: ModelMeta) => void;
  isLoading?: boolean;
}

export function ModelRegistryTable({
  models,
  selectedIds,
  onSelect,
  onView,
  onDownload,
  onDelete,
  isLoading,
}: ModelRegistryTableProps) {
  const [sortKey, setSortKey] = useState<SortKey>("createdAt");
  const [sortOrder, setSortOrder] = useState<SortOrder>("desc");

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortOrder((prev) => (prev === "asc" ? "desc" : "asc"));
    } else {
      setSortKey(key);
      setSortOrder("desc");
    }
  };

  const sortedModels = useMemo(() => {
    return [...models].sort((a, b) => {
      let aVal: string | number = "";
      let bVal: string | number = "";

      switch (sortKey) {
        case "symbol":
          aVal = a.symbol;
          bVal = b.symbol;
          break;
        case "createdAt":
          aVal = new Date(a.createdAt).getTime();
          bVal = new Date(b.createdAt).getTime();
          break;
        case "sharpeRatio":
          aVal = a.metrics?.sharpeRatio ?? 0;
          bVal = b.metrics?.sharpeRatio ?? 0;
          break;
        case "winRate":
          aVal = a.metrics?.winRate ?? 0;
          bVal = b.metrics?.winRate ?? 0;
          break;
        case "maxDrawdown":
          aVal = a.metrics?.maxDrawdown ?? 0;
          bVal = b.metrics?.maxDrawdown ?? 0;
          break;
      }

      if (typeof aVal === "string") {
        return sortOrder === "asc" ? aVal.localeCompare(bVal as string) : (bVal as string).localeCompare(aVal);
      }
      return sortOrder === "asc" ? aVal - (bVal as number) : (bVal as number) - aVal;
    });
  }, [models, sortKey, sortOrder]);

  const SortHeader = ({ label, sortKeyName }: { label: string; sortKeyName: SortKey }) => (
    <button
      onClick={() => handleSort(sortKeyName)}
      className={cn(
        "flex items-center gap-1 text-xs font-medium uppercase tracking-wider",
        sortKey === sortKeyName ? "text-yellow-400" : "text-gray-500 hover:text-gray-300"
      )}
    >
      {label}
      <ArrowUpDown className="size-3" />
    </button>
  );

  if (models.length === 0 && !isLoading) {
    return (
      <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-8">
        <div className="flex flex-col items-center justify-center text-center">
          <Database className="size-12 text-gray-700 mb-4" />
          <p className="text-gray-400">No trained models found</p>
          <p className="text-sm text-gray-500 mt-1">Train a model to see it here</p>
        </div>
      </div>
    );
  }

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 overflow-hidden">
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-800 bg-gray-900/80">
              <th className="px-4 py-3 text-left w-10">
                <span className="sr-only">Select</span>
              </th>
              <th className="px-4 py-3 text-left">
                <SortHeader label="Symbol" sortKeyName="symbol" />
              </th>
              <th className="px-4 py-3 text-left">
                <SortHeader label="Created" sortKeyName="createdAt" />
              </th>
              <th className="px-4 py-3 text-left">
                <SortHeader label="Sharpe" sortKeyName="sharpeRatio" />
              </th>
              <th className="px-4 py-3 text-left">
                <SortHeader label="Win Rate" sortKeyName="winRate" />
              </th>
              <th className="px-4 py-3 text-left">
                <SortHeader label="Max DD" sortKeyName="maxDrawdown" />
              </th>
              <th className="px-4 py-3 text-left">
                <span className="text-xs font-medium uppercase tracking-wider text-gray-500">Seq. Len</span>
              </th>
              <th className="px-4 py-3 text-right">
                <span className="text-xs font-medium uppercase tracking-wider text-gray-500">Actions</span>
              </th>
            </tr>
          </thead>
          <tbody>
            {sortedModels.map((model) => {
              const isSelected = selectedIds.includes(model.id);
              const sharpe = model.metrics?.sharpeRatio ?? 0;
              const winRate = model.metrics?.winRate ?? 0;
              const maxDD = model.metrics?.maxDrawdown ?? 0;

              return (
                <tr
                  key={model.id}
                  className={cn(
                    "border-b border-gray-800/50 transition-colors",
                    isSelected ? "bg-yellow-500/5" : "hover:bg-gray-800/30"
                  )}
                >
                  <td className="px-4 py-3">
                    <button
                      onClick={() => onSelect(model.id)}
                      className={cn(
                        "transition-colors",
                        isSelected ? "text-yellow-400" : "text-gray-600 hover:text-gray-400"
                      )}
                    >
                      {isSelected ? <CheckSquare className="size-5" /> : <Square className="size-5" />}
                    </button>
                  </td>
                  <td className="px-4 py-3">
                    <span className="font-semibold text-white">{model.symbol}</span>
                  </td>
                  <td className="px-4 py-3">
                    <span className="text-gray-400 text-sm">
                      {formatDistanceToNow(new Date(model.createdAt), { addSuffix: true })}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <span className={cn(
                      "font-medium",
                      sharpe > 1 ? "text-emerald-400" : sharpe > 0.5 ? "text-yellow-400" : "text-gray-400"
                    )}>
                      {sharpe.toFixed(2)}
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <span className={cn(
                      "font-medium",
                      winRate > 0.55 ? "text-emerald-400" : winRate > 0.45 ? "text-yellow-400" : "text-gray-400"
                    )}>
                      {(winRate * 100).toFixed(0)}%
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <span className={cn(
                      "font-medium",
                      maxDD < 0.15 ? "text-emerald-400" : maxDD < 0.25 ? "text-yellow-400" : "text-rose-400"
                    )}>
                      {(maxDD * 100).toFixed(1)}%
                    </span>
                  </td>
                  <td className="px-4 py-3">
                    <span className="text-gray-400">{model.sequenceLength}</span>
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center justify-end gap-1">
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        onClick={() => onView(model)}
                        className="text-gray-400 hover:text-white"
                      >
                        <Eye className="size-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        onClick={() => onDownload(model)}
                        className="text-gray-400 hover:text-white"
                      >
                        <Download className="size-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="icon-sm"
                        onClick={() => onDelete(model)}
                        className="text-gray-400 hover:text-rose-400"
                      >
                        <Trash2 className="size-4" />
                      </Button>
                    </div>
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}
