"use client";

import { cn } from "@/lib/utils";
import { Loader2, Brain } from "lucide-react";

interface LoadingOverlayProps {
  isLoading: boolean;
  message?: string;
  submessage?: string;
  variant?: "default" | "fullscreen" | "inline";
}

export function LoadingOverlay({
  isLoading,
  message = "Loading...",
  submessage,
  variant = "default",
}: LoadingOverlayProps) {
  if (!isLoading) return null;

  if (variant === "inline") {
    return (
      <div className="flex items-center justify-center gap-3 py-8">
        <Loader2 className="size-6 text-yellow-400 animate-spin" />
        <div>
          <p className="text-gray-300 font-medium">{message}</p>
          {submessage && <p className="text-sm text-gray-500">{submessage}</p>}
        </div>
      </div>
    );
  }

  if (variant === "fullscreen") {
    return (
      <div className="fixed inset-0 z-50 flex items-center justify-center bg-gray-950/90 backdrop-blur-sm">
        <div className="flex flex-col items-center gap-4 text-center">
          <div className="relative">
            <div className="absolute inset-0 animate-ping">
              <Brain className="size-16 text-yellow-400/30" />
            </div>
            <Brain className="relative size-16 text-yellow-400" />
          </div>
          <div>
            <p className="text-xl font-semibold text-white">{message}</p>
            {submessage && <p className="text-gray-400 mt-1">{submessage}</p>}
          </div>
          <div className="flex items-center gap-1 mt-2">
            {[...Array(3)].map((_, i) => (
              <div
                key={i}
                className="size-2 rounded-full bg-yellow-400"
                style={{
                  animation: `pulse 1.4s ease-in-out ${i * 0.2}s infinite`,
                }}
              />
            ))}
          </div>
        </div>
        <style jsx>{`
          @keyframes pulse {
            0%, 100% { opacity: 0.3; transform: scale(0.8); }
            50% { opacity: 1; transform: scale(1.2); }
          }
        `}</style>
      </div>
    );
  }

  // Default overlay
  return (
    <div className={cn(
      "absolute inset-0 z-10 flex items-center justify-center",
      "bg-gray-900/80 backdrop-blur-sm rounded-xl"
    )}>
      <div className="flex flex-col items-center gap-3 text-center">
        <Loader2 className="size-10 text-yellow-400 animate-spin" />
        <div>
          <p className="text-gray-200 font-medium">{message}</p>
          {submessage && <p className="text-sm text-gray-500">{submessage}</p>}
        </div>
      </div>
    </div>
  );
}
