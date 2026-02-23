"use client";

import { cn } from "@/lib/utils";
import {
  LayoutDashboard,
  GraduationCap,
  TrendingUp,
  FlaskConical,
  Database,
  Loader2
} from "lucide-react";

export type AiTab = "dashboard" | "training" | "prediction" | "backtest" | "registry";

interface TabDefinition {
  id: AiTab;
  label: string;
  icon: React.ReactNode;
  shortcut: string;
  description: string;
}

const TABS: TabDefinition[] = [
  {
    id: "dashboard",
    label: "Dashboard",
    icon: <LayoutDashboard className="size-4" />,
    shortcut: "1",
    description: "Quick predictions and recommendations",
  },
  {
    id: "training",
    label: "Training",
    icon: <GraduationCap className="size-4" />,
    shortcut: "2",
    description: "Train new models with GPU acceleration",
  },
  {
    id: "prediction",
    label: "Prediction",
    icon: <TrendingUp className="size-4" />,
    shortcut: "3",
    description: "Advanced prediction interface",
  },
  {
    id: "backtest",
    label: "Backtest",
    icon: <FlaskConical className="size-4" />,
    shortcut: "4",
    description: "Comprehensive backtesting analytics",
  },
  {
    id: "registry",
    label: "Models",
    icon: <Database className="size-4" />,
    shortcut: "5",
    description: "Manage and compare trained models",
  },
];

interface AiTabNavProps {
  activeTab: AiTab;
  onTabChange: (tab: AiTab) => void;
  trainingInProgress?: boolean;
  className?: string;
}

export function AiTabNav({ activeTab, onTabChange, trainingInProgress, className }: AiTabNavProps) {
  return (
    <nav className={cn("flex items-center gap-1 rounded-xl border border-gray-800 bg-gray-900/60 p-1.5", className)}>
      {TABS.map((tab) => {
        const isActive = activeTab === tab.id;
        const showTrainingBadge = tab.id === "training" && trainingInProgress;

        return (
          <button
            key={tab.id}
            onClick={() => onTabChange(tab.id)}
            title={`${tab.description} (⌘${tab.shortcut})`}
            className={cn(
              "relative flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium transition-all",
              "focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-yellow-500/50",
              isActive
                ? "bg-yellow-500/10 text-yellow-200 border border-yellow-500/50"
                : "text-gray-400 hover:text-gray-200 hover:bg-gray-800/50"
            )}
          >
            {tab.icon}
            <span className="hidden sm:inline">{tab.label}</span>
            <span className="hidden lg:inline text-[10px] text-gray-500 ml-1">⌘{tab.shortcut}</span>

            {showTrainingBadge && (
              <span className="absolute -top-1 -right-1 flex items-center justify-center">
                <span className="absolute inline-flex h-3 w-3 animate-ping rounded-full bg-yellow-400 opacity-75" />
                <Loader2 className="relative size-3 animate-spin text-yellow-400" />
              </span>
            )}
          </button>
        );
      })}
    </nav>
  );
}

export { TABS };
