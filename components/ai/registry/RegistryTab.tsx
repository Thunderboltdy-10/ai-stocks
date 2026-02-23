"use client";

import { useCallback, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { toast } from "sonner";
import { ModelRegistryTable } from "./ModelRegistryTable";
import { ModelDetailCard } from "./ModelDetailCard";
import { ModelComparisonView } from "./ModelComparisonView";
import { listModels, downloadModelArtifacts, deleteModel } from "@/lib/api-client";
import { ModelMeta } from "@/types/ai";
import { Button } from "@/components/ui/button";
import { GitCompare, RefreshCw } from "lucide-react";

export function RegistryTab() {
  const queryClient = useQueryClient();
  const [selectedIds, setSelectedIds] = useState<string[]>([]);
  const [viewingModel, setViewingModel] = useState<ModelMeta | null>(null);
  const [showComparison, setShowComparison] = useState(false);

  const modelsQuery = useQuery({
    queryKey: ["models"],
    queryFn: listModels,
  });

  const deleteMutation = useMutation({
    mutationFn: deleteModel,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["models"] });
      toast.success("Model deleted successfully");
    },
    onError: (error: Error) => {
      toast.error(`Failed to delete model: ${error.message}`);
    },
  });

  const handleSelect = useCallback((id: string) => {
    setSelectedIds((prev) => {
      if (prev.includes(id)) {
        return prev.filter((i) => i !== id);
      }
      if (prev.length >= 3) {
        return [...prev.slice(1), id];
      }
      return [...prev, id];
    });
  }, []);

  const handleView = useCallback((model: ModelMeta) => {
    setViewingModel(model);
    setShowComparison(false);
  }, []);

  const handleDownload = useCallback(async (model: ModelMeta) => {
    try {
      const blob = await downloadModelArtifacts(model.id);
      const link = document.createElement("a");
      link.href = URL.createObjectURL(blob);
      link.download = `${model.symbol}_artifacts.zip`;
      link.click();
      URL.revokeObjectURL(link.href);
      toast.success(`Downloaded artifacts for ${model.symbol}`);
    } catch (error) {
      toast.error(`Failed to download: ${error instanceof Error ? error.message : "Unknown error"}`);
    }
  }, []);

  const handleDelete = useCallback((model: ModelMeta) => {
    if (confirm(`Are you sure you want to delete the model for ${model.symbol}?`)) {
      deleteMutation.mutate(model.id);
      if (viewingModel?.id === model.id) {
        setViewingModel(null);
      }
      setSelectedIds((prev) => prev.filter((id) => id !== model.id));
    }
  }, [deleteMutation, viewingModel]);

  const handleShowComparison = useCallback(() => {
    if (selectedIds.length < 2) {
      toast.error("Select at least 2 models to compare");
      return;
    }
    setShowComparison(true);
    setViewingModel(null);
  }, [selectedIds]);

  const selectedModels = (modelsQuery.data ?? []).filter((m) => selectedIds.includes(m.id));

  return (
    <div className="space-y-6">
      {/* Header Actions */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-semibold text-white">Model Registry</h2>
          <p className="text-sm text-gray-400">
            {modelsQuery.data?.length ?? 0} models available
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            variant="outline"
            size="sm"
            onClick={() => modelsQuery.refetch()}
            disabled={modelsQuery.isFetching}
            className="border-gray-700 text-gray-300"
          >
            <RefreshCw className={`size-4 mr-2 ${modelsQuery.isFetching ? "animate-spin" : ""}`} />
            Refresh
          </Button>
          <Button
            variant="outline"
            size="sm"
            onClick={handleShowComparison}
            disabled={selectedIds.length < 2}
            className="border-gray-700 text-gray-300"
          >
            <GitCompare className="size-4 mr-2" />
            Compare ({selectedIds.length})
          </Button>
        </div>
      </div>

      {/* Main Content */}
      <div className="grid gap-6 lg:grid-cols-[1fr,400px]">
        {/* Table */}
        <ModelRegistryTable
          models={modelsQuery.data ?? []}
          selectedIds={selectedIds}
          onSelect={handleSelect}
          onView={handleView}
          onDownload={handleDownload}
          onDelete={handleDelete}
          isLoading={modelsQuery.isLoading}
        />

        {/* Side Panel */}
        <div>
          {showComparison ? (
            <ModelComparisonView
              models={selectedModels}
              onClose={() => setShowComparison(false)}
            />
          ) : viewingModel ? (
            <ModelDetailCard
              model={viewingModel}
              onClose={() => setViewingModel(null)}
              onDownload={() => handleDownload(viewingModel)}
            />
          ) : (
            <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-8">
              <div className="flex flex-col items-center justify-center text-center">
                <p className="text-gray-400">Select a model to view details</p>
                <p className="text-sm text-gray-500 mt-1">Or select multiple models to compare</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
