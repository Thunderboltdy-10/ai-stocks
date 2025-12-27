"use client";

import { useEffect } from "react";

interface Shortcut {
  key: string;
  handler: () => void;
}

export function useKeyboardShortcuts(shortcuts: Shortcut[]) {
  useEffect(() => {
    const handleKeydown = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement) {
        return;
      }
      const shortcut = shortcuts.find((item) => item.key.toLowerCase() === event.key.toLowerCase());
      if (shortcut) {
        event.preventDefault();
        shortcut.handler();
      }
    };

    window.addEventListener("keydown", handleKeydown);
    return () => window.removeEventListener("keydown", handleKeydown);
  }, [shortcuts]);
}
