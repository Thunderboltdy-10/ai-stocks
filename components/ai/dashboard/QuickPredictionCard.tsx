"use client";

import { useState, useCallback } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { cn } from "@/lib/utils";
import { Loader2, TrendingUp, Zap } from "lucide-react";

type RiskProfile = "conservative" | "balanced" | "aggressive";

interface QuickPredictionCardProps {
  onPredict: (symbol: string, riskProfile: RiskProfile) => Promise<void>;
  isLoading: boolean;
  recentSymbols?: string[];
}

const RISK_PROFILES: { id: RiskProfile; label: string; description: string; color: string }[] = [
  {
    id: "conservative",
    label: "Conservative",
    description: "Lower risk, higher confidence threshold",
    color: "text-blue-400 border-blue-500/50 bg-blue-500/10",
  },
  {
    id: "balanced",
    label: "Balanced",
    description: "Moderate risk, balanced thresholds",
    color: "text-yellow-400 border-yellow-500/50 bg-yellow-500/10",
  },
  {
    id: "aggressive",
    label: "Aggressive",
    description: "Higher risk, lower confidence threshold",
    color: "text-orange-400 border-orange-500/50 bg-orange-500/10",
  },
];

export function QuickPredictionCard({ onPredict, isLoading, recentSymbols = [] }: QuickPredictionCardProps) {
  const [symbol, setSymbol] = useState("");
  const [riskProfile, setRiskProfile] = useState<RiskProfile>("balanced");
  const [showRecent, setShowRecent] = useState(false);

  const handleSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    if (!symbol.trim()) return;
    await onPredict(symbol.toUpperCase(), riskProfile);
  }, [symbol, riskProfile, onPredict]);

  const handleSymbolClick = (sym: string) => {
    setSymbol(sym);
    setShowRecent(false);
  };

  return (
    <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5">
      <div className="flex items-center gap-2 mb-4">
        <Zap className="size-5 text-yellow-400" />
        <h2 className="text-lg font-semibold text-white">Quick Prediction</h2>
      </div>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div className="space-y-2">
          <Label htmlFor="symbol" className="text-gray-400 text-sm">
            Stock Symbol
          </Label>
          <div className="relative">
            <Input
              id="symbol"
              type="text"
              placeholder="AAPL, MSFT, TSLA..."
              value={symbol}
              onChange={(e) => setSymbol(e.target.value.toUpperCase())}
              onFocus={() => recentSymbols.length > 0 && setShowRecent(true)}
              onBlur={() => setTimeout(() => setShowRecent(false), 200)}
              className="h-12 bg-gray-950 border-gray-700 text-white text-lg font-mono placeholder:text-gray-500"
            />
            {showRecent && recentSymbols.length > 0 && (
              <div className="absolute top-full left-0 right-0 mt-1 rounded-lg border border-gray-700 bg-gray-900 p-2 z-10">
                <p className="text-xs text-gray-500 mb-2">Recent</p>
                <div className="flex flex-wrap gap-2">
                  {recentSymbols.slice(0, 5).map((sym) => (
                    <button
                      key={sym}
                      type="button"
                      onClick={() => handleSymbolClick(sym)}
                      className="px-3 py-1 rounded-md bg-gray-800 text-gray-300 text-sm hover:bg-gray-700 hover:text-white transition"
                    >
                      {sym}
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="space-y-2">
          <Label className="text-gray-400 text-sm">Risk Profile</Label>
          <div className="grid grid-cols-3 gap-2">
            {RISK_PROFILES.map((profile) => (
              <button
                key={profile.id}
                type="button"
                onClick={() => setRiskProfile(profile.id)}
                className={cn(
                  "flex flex-col items-center gap-1 rounded-lg border p-3 text-center transition-all",
                  riskProfile === profile.id
                    ? profile.color
                    : "border-gray-700 text-gray-400 hover:border-gray-600"
                )}
              >
                <span className="text-sm font-medium">{profile.label}</span>
                <span className="text-[10px] text-gray-500 hidden sm:block">
                  {profile.description}
                </span>
              </button>
            ))}
          </div>
        </div>

        <Button
          type="submit"
          disabled={!symbol.trim() || isLoading}
          className="w-full h-12 bg-gradient-to-b from-yellow-400 to-yellow-500 text-gray-900 font-semibold text-base hover:from-yellow-300 hover:to-yellow-400 disabled:opacity-50"
        >
          {isLoading ? (
            <>
              <Loader2 className="size-5 animate-spin mr-2" />
              Generating Prediction...
            </>
          ) : (
            <>
              <TrendingUp className="size-5 mr-2" />
              Generate Prediction
            </>
          )}
        </Button>
      </form>
    </div>
  );
}
