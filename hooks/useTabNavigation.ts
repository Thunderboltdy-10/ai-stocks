"use client";

import { useEffect, useCallback } from "react";
import { AiTab } from "@/components/ai/AiTabNav";

const TAB_ORDER: AiTab[] = ["dashboard", "training", "prediction", "backtest", "registry"];

interface UseTabNavigationOptions {
  activeTab: AiTab;
  onTabChange: (tab: AiTab) => void;
  enabled?: boolean;
}

export function useTabNavigation({ activeTab, onTabChange, enabled = true }: UseTabNavigationOptions) {
  const goToTab = useCallback((index: number) => {
    if (index >= 0 && index < TAB_ORDER.length) {
      onTabChange(TAB_ORDER[index]);
    }
  }, [onTabChange]);

  const goToNextTab = useCallback(() => {
    const currentIndex = TAB_ORDER.indexOf(activeTab);
    const nextIndex = (currentIndex + 1) % TAB_ORDER.length;
    onTabChange(TAB_ORDER[nextIndex]);
  }, [activeTab, onTabChange]);

  const goToPrevTab = useCallback(() => {
    const currentIndex = TAB_ORDER.indexOf(activeTab);
    const prevIndex = (currentIndex - 1 + TAB_ORDER.length) % TAB_ORDER.length;
    onTabChange(TAB_ORDER[prevIndex]);
  }, [activeTab, onTabChange]);

  useEffect(() => {
    if (!enabled) return;

    const handleKeyDown = (event: KeyboardEvent) => {
      // Skip if user is typing in an input
      if (
        event.target instanceof HTMLInputElement ||
        event.target instanceof HTMLTextAreaElement ||
        event.defaultPrevented
      ) {
        return;
      }

      // Cmd/Ctrl + number for direct tab navigation
      if (event.metaKey || event.ctrlKey) {
        const num = parseInt(event.key, 10);
        if (num >= 1 && num <= 5) {
          event.preventDefault();
          goToTab(num - 1);
          return;
        }
      }

      // Alt + Arrow for tab cycling
      if (event.altKey) {
        if (event.key === "ArrowRight") {
          event.preventDefault();
          goToNextTab();
          return;
        }
        if (event.key === "ArrowLeft") {
          event.preventDefault();
          goToPrevTab();
          return;
        }
      }

      // ] and [ for tab cycling (without modifiers)
      if (!event.metaKey && !event.ctrlKey && !event.altKey) {
        if (event.key === "]") {
          event.preventDefault();
          goToNextTab();
          return;
        }
        if (event.key === "[") {
          event.preventDefault();
          goToPrevTab();
          return;
        }
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [enabled, goToTab, goToNextTab, goToPrevTab]);

  return {
    goToTab,
    goToNextTab,
    goToPrevTab,
    tabs: TAB_ORDER,
  };
}
