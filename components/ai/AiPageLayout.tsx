"use client";

import { useCallback, useEffect, useState } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import { AiTabNav, AiTab, TABS } from "./AiTabNav";
import { KeyboardShortcutsDialog } from "./shared/KeyboardShortcutsDialog";
import { cn } from "@/lib/utils";
import { Keyboard } from "lucide-react";

interface AiPageLayoutProps {
  children: React.ReactNode;
  activeTab: AiTab;
  onTabChange: (tab: AiTab) => void;
  trainingInProgress?: boolean;
}

export function AiPageLayout({ children, activeTab, onTabChange, trainingInProgress }: AiPageLayoutProps) {
  const [shortcutsOpen, setShortcutsOpen] = useState(false);

  // Keyboard shortcuts for tab navigation
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Skip if user is typing
      if (
        e.target instanceof HTMLInputElement ||
        e.target instanceof HTMLTextAreaElement
      ) {
        return;
      }

      // Handle Cmd/Ctrl + number shortcuts
      if (e.metaKey || e.ctrlKey) {
        const num = parseInt(e.key, 10);
        if (num >= 1 && num <= 5) {
          e.preventDefault();
          const tab = TABS[num - 1];
          if (tab) {
            onTabChange(tab.id);
          }
        }
      }

      // Tab cycling with [ and ]
      if (!e.metaKey && !e.ctrlKey && !e.altKey) {
        if (e.key === "]") {
          e.preventDefault();
          const currentIndex = TABS.findIndex(t => t.id === activeTab);
          const nextIndex = (currentIndex + 1) % TABS.length;
          onTabChange(TABS[nextIndex].id);
        }
        if (e.key === "[") {
          e.preventDefault();
          const currentIndex = TABS.findIndex(t => t.id === activeTab);
          const prevIndex = (currentIndex - 1 + TABS.length) % TABS.length;
          onTabChange(TABS[prevIndex].id);
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [onTabChange, activeTab]);

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <div className="mx-auto max-w-[1600px] space-y-6 px-4 py-6">
        <header className="space-y-4">
          <div className="flex items-start justify-between">
            <div className="space-y-2">
              <p className="text-xs uppercase tracking-wide text-gray-500">
                AI-Stocks v2 · Comprehensive AI Workbench
              </p>
              <h1 className="text-3xl font-semibold text-white">AI Lifecycle Console</h1>
              <p className="text-sm text-gray-400">
                Train models, generate predictions, run backtests, and manage your model registry from one unified interface.
              </p>
            </div>
            <button
              onClick={() => setShortcutsOpen(true)}
              className="flex items-center gap-2 px-3 py-2 rounded-lg border border-gray-700 text-gray-400 hover:text-gray-200 hover:border-gray-600 transition-colors"
              title="Keyboard shortcuts (?)"
            >
              <Keyboard className="size-4" />
              <span className="text-xs hidden sm:inline">Shortcuts</span>
              <kbd className="px-1.5 py-0.5 text-[10px] bg-gray-800 rounded border border-gray-700">?</kbd>
            </button>
          </div>

          <AiTabNav
            activeTab={activeTab}
            onTabChange={onTabChange}
            trainingInProgress={trainingInProgress}
          />
        </header>

        <main>
          {children}
        </main>
      </div>

      <KeyboardShortcutsDialog open={shortcutsOpen} onOpenChange={setShortcutsOpen} />
    </div>
  );
}

// Hook for managing tab state with URL sync
export function useAiTabState(defaultTab: AiTab = "dashboard") {
  const router = useRouter();
  const searchParams = useSearchParams();

  const tabFromUrl = searchParams.get("tab") as AiTab | null;
  const [activeTab, setActiveTab] = useState<AiTab>(
    tabFromUrl && TABS.some(t => t.id === tabFromUrl) ? tabFromUrl : defaultTab
  );

  const handleTabChange = useCallback((tab: AiTab) => {
    setActiveTab(tab);
    // Update URL without full navigation
    const params = new URLSearchParams(searchParams.toString());
    params.set("tab", tab);
    router.push(`?${params.toString()}`, { scroll: false });
  }, [router, searchParams]);

  // Sync with URL changes
  useEffect(() => {
    if (tabFromUrl && TABS.some(t => t.id === tabFromUrl) && tabFromUrl !== activeTab) {
      setActiveTab(tabFromUrl);
    }
  }, [tabFromUrl, activeTab]);

  return { activeTab, setActiveTab: handleTabChange };
}
