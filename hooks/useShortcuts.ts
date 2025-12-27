"use client";

import { useEffect } from "react";

type ShortcutHandler = () => void;

export interface ShortcutDefinition {
  combo: string; // e.g. "ctrl+b" or "alt+i"
  handler: ShortcutHandler;
  description?: string;
}

const normalizeKey = (value: string) => value.trim().toLowerCase();

const matchShortcut = (event: KeyboardEvent, combo: string) => {
  const parts = combo.split("+").map(normalizeKey);
  const key = parts.pop();
  if (!key) return false;

  const needsCtrl = parts.includes("ctrl") || parts.includes("cmd") || parts.includes("meta");
  const needsAlt = parts.includes("alt");
  const needsShift = parts.includes("shift");

  if (needsCtrl !== (event.ctrlKey || event.metaKey)) return false;
  if (needsAlt !== event.altKey) return false;
  if (needsShift !== event.shiftKey) return false;

  return normalizeKey(event.key) === key;
};

export function useShortcuts(shortcuts: ShortcutDefinition[]) {
  useEffect(() => {
    const handleKeydown = (event: KeyboardEvent) => {
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLTextAreaElement || event.defaultPrevented) {
        return;
      }

      const shortcut = shortcuts.find((item) => matchShortcut(event, item.combo));
      if (shortcut) {
        event.preventDefault();
        shortcut.handler();
      }
    };

    window.addEventListener("keydown", handleKeydown);
    return () => window.removeEventListener("keydown", handleKeydown);
  }, [shortcuts]);
}
