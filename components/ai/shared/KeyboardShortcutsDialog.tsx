"use client";

import { useEffect, useState } from "react";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Keyboard } from "lucide-react";
import { cn } from "@/lib/utils";

interface ShortcutDefinition {
  keys: string[];
  description: string;
  category: string;
}

const SHORTCUTS: ShortcutDefinition[] = [
  // Navigation
  { keys: ["⌘", "1"], description: "Go to Dashboard", category: "Navigation" },
  { keys: ["⌘", "2"], description: "Go to Training", category: "Navigation" },
  { keys: ["⌘", "3"], description: "Go to Prediction", category: "Navigation" },
  { keys: ["⌘", "4"], description: "Go to Backtest", category: "Navigation" },
  { keys: ["⌘", "5"], description: "Go to Model Registry", category: "Navigation" },
  { keys: ["]"], description: "Next tab", category: "Navigation" },
  { keys: ["["], description: "Previous tab", category: "Navigation" },

  // Actions
  { keys: ["⌘", "I"], description: "Run Inference", category: "Actions" },
  { keys: ["⌘", "F"], description: "Run Backtest", category: "Actions" },
  { keys: ["⌘", "K"], description: "Search stocks", category: "Actions" },

  // View Modes
  { keys: ["Ctrl", "H"], description: "History mode", category: "View Modes" },
  { keys: ["Ctrl", "P"], description: "Prediction mode", category: "View Modes" },
  { keys: ["Ctrl", "B"], description: "Backtest mode", category: "View Modes" },

  // General
  { keys: ["?"], description: "Show keyboard shortcuts", category: "General" },
  { keys: ["Esc"], description: "Close dialog / Cancel", category: "General" },
];

interface KeyboardShortcutsDialogProps {
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
}

export function KeyboardShortcutsDialog({ open, onOpenChange }: KeyboardShortcutsDialogProps) {
  const [isOpen, setIsOpen] = useState(open ?? false);

  const handleOpenChange = (newOpen: boolean) => {
    setIsOpen(newOpen);
    onOpenChange?.(newOpen);
  };

  // Listen for ? key to open dialog
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "?" && !e.metaKey && !e.ctrlKey) {
        e.preventDefault();
        handleOpenChange(true);
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // Group shortcuts by category
  const groupedShortcuts = SHORTCUTS.reduce((acc, shortcut) => {
    if (!acc[shortcut.category]) {
      acc[shortcut.category] = [];
    }
    acc[shortcut.category].push(shortcut);
    return acc;
  }, {} as Record<string, ShortcutDefinition[]>);

  return (
    <Dialog open={isOpen} onOpenChange={handleOpenChange}>
      <DialogContent className="max-w-lg border-gray-800 bg-gray-950 text-gray-100">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Keyboard className="size-5 text-yellow-400" />
            Keyboard Shortcuts
          </DialogTitle>
          <DialogDescription>
            Use these shortcuts to navigate and control the AI workbench
          </DialogDescription>
        </DialogHeader>

        <div className="space-y-6 mt-4">
          {Object.entries(groupedShortcuts).map(([category, shortcuts]) => (
            <div key={category}>
              <h4 className="text-xs uppercase tracking-wider text-gray-500 mb-3">
                {category}
              </h4>
              <div className="space-y-2">
                {shortcuts.map((shortcut, idx) => (
                  <div
                    key={idx}
                    className="flex items-center justify-between py-2 px-3 rounded-lg bg-gray-900/50"
                  >
                    <span className="text-sm text-gray-300">{shortcut.description}</span>
                    <div className="flex items-center gap-1">
                      {shortcut.keys.map((key, keyIdx) => (
                        <kbd
                          key={keyIdx}
                          className={cn(
                            "px-2 py-1 text-xs font-mono rounded",
                            "bg-gray-800 border border-gray-700 text-gray-300"
                          )}
                        >
                          {key}
                        </kbd>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>

        <p className="text-xs text-gray-500 mt-4 text-center">
          Press <kbd className="px-1 py-0.5 text-[10px] bg-gray-800 rounded border border-gray-700">?</kbd> anywhere to show this dialog
        </p>
      </DialogContent>
    </Dialog>
  );
}
