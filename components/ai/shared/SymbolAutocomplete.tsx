"use client";

import { useState, useCallback, useRef, useEffect } from "react";
import { Input } from "@/components/ui/input";
import { cn } from "@/lib/utils";
import { Search, TrendingUp, Clock, Loader2 } from "lucide-react";

// Popular symbols for suggestions
const POPULAR_SYMBOLS = [
  { symbol: "AAPL", name: "Apple Inc." },
  { symbol: "MSFT", name: "Microsoft Corporation" },
  { symbol: "GOOGL", name: "Alphabet Inc." },
  { symbol: "AMZN", name: "Amazon.com Inc." },
  { symbol: "TSLA", name: "Tesla Inc." },
  { symbol: "META", name: "Meta Platforms Inc." },
  { symbol: "NVDA", name: "NVIDIA Corporation" },
  { symbol: "JPM", name: "JPMorgan Chase & Co." },
  { symbol: "V", name: "Visa Inc." },
  { symbol: "JNJ", name: "Johnson & Johnson" },
];

interface SymbolAutocompleteProps {
  value: string;
  onChange: (symbol: string) => void;
  recentSymbols?: string[];
  placeholder?: string;
  disabled?: boolean;
  className?: string;
}

export function SymbolAutocomplete({
  value,
  onChange,
  recentSymbols = [],
  placeholder = "Enter symbol (e.g., AAPL)",
  disabled,
  className,
}: SymbolAutocompleteProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [search, setSearch] = useState(value);
  const [isValidating, setIsValidating] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Filter suggestions based on search
  const suggestions = search.length > 0
    ? POPULAR_SYMBOLS.filter(
        (s) =>
          s.symbol.toLowerCase().includes(search.toLowerCase()) ||
          s.name.toLowerCase().includes(search.toLowerCase())
      ).slice(0, 5)
    : POPULAR_SYMBOLS.slice(0, 5);

  const handleInputChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value.toUpperCase();
    setSearch(val);
    onChange(val);
  }, [onChange]);

  const handleSelect = useCallback((symbol: string) => {
    setSearch(symbol);
    onChange(symbol);
    setIsOpen(false);
    inputRef.current?.blur();
  }, [onChange]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (containerRef.current && !containerRef.current.contains(e.target as Node)) {
        setIsOpen(false);
      }
    };

    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  // Sync external value
  useEffect(() => {
    if (value !== search) {
      setSearch(value);
    }
  }, [value]);

  return (
    <div ref={containerRef} className={cn("relative", className)}>
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 size-4 text-gray-500" />
        <Input
          ref={inputRef}
          type="text"
          value={search}
          onChange={handleInputChange}
          onFocus={() => setIsOpen(true)}
          placeholder={placeholder}
          disabled={disabled}
          className={cn(
            "pl-10 pr-10 h-11 bg-gray-950 border-gray-700 text-white font-mono",
            "focus:border-yellow-500/50 focus:ring-yellow-500/20"
          )}
        />
        {isValidating && (
          <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 size-4 text-gray-500 animate-spin" />
        )}
      </div>

      {isOpen && (
        <div className="absolute top-full left-0 right-0 mt-1 z-50 rounded-lg border border-gray-700 bg-gray-900 shadow-xl overflow-hidden">
          {/* Recent Symbols */}
          {recentSymbols.length > 0 && (
            <div className="p-2 border-b border-gray-800">
              <p className="text-xs text-gray-500 flex items-center gap-1 mb-2">
                <Clock className="size-3" /> Recent
              </p>
              <div className="flex flex-wrap gap-1">
                {recentSymbols.slice(0, 5).map((sym) => (
                  <button
                    key={sym}
                    type="button"
                    onClick={() => handleSelect(sym)}
                    className="px-2 py-1 text-xs rounded bg-gray-800 text-gray-300 hover:bg-gray-700 hover:text-white transition"
                  >
                    {sym}
                  </button>
                ))}
              </div>
            </div>
          )}

          {/* Suggestions */}
          <div className="p-2">
            <p className="text-xs text-gray-500 flex items-center gap-1 mb-2">
              <TrendingUp className="size-3" /> {search ? "Matches" : "Popular"}
            </p>
            <div className="space-y-1">
              {suggestions.map((s) => (
                <button
                  key={s.symbol}
                  type="button"
                  onClick={() => handleSelect(s.symbol)}
                  className="w-full flex items-center gap-3 p-2 rounded-lg hover:bg-gray-800 transition text-left"
                >
                  <span className="font-mono font-semibold text-white">{s.symbol}</span>
                  <span className="text-sm text-gray-500 truncate">{s.name}</span>
                </button>
              ))}
              {suggestions.length === 0 && search && (
                <button
                  type="button"
                  onClick={() => handleSelect(search)}
                  className="w-full p-2 rounded-lg hover:bg-gray-800 transition text-left"
                >
                  <span className="font-mono font-semibold text-white">{search}</span>
                  <span className="text-sm text-gray-500 ml-2">Use this symbol</span>
                </button>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
