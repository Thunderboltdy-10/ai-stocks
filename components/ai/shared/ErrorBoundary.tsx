"use client";

import React from "react";
import { AlertTriangle, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";

interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
  onReset?: () => void;
}

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error("ErrorBoundary caught an error:", error, errorInfo);
  }

  handleReset = () => {
    this.setState({ hasError: false, error: null });
    this.props.onReset?.();
  };

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) {
        return this.props.fallback;
      }

      return (
        <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-6">
          <div className="flex flex-col items-center text-center space-y-4">
            <div className="size-12 rounded-full bg-rose-500/20 flex items-center justify-center">
              <AlertTriangle className="size-6 text-rose-400" />
            </div>
            <div>
              <h3 className="text-lg font-semibold text-rose-400">Something went wrong</h3>
              <p className="text-sm text-gray-400 mt-1">
                {this.state.error?.message || "An unexpected error occurred"}
              </p>
            </div>
            <Button
              variant="outline"
              onClick={this.handleReset}
              className="border-rose-500/50 text-rose-400 hover:bg-rose-500/10"
            >
              <RefreshCw className="size-4 mr-2" />
              Try Again
            </Button>
          </div>
        </div>
      );
    }

    return this.props.children;
  }
}

// Functional wrapper for easier use with hooks
interface ErrorFallbackProps {
  error: Error | null;
  resetError: () => void;
}

export function ErrorFallback({ error, resetError }: ErrorFallbackProps) {
  return (
    <div className="rounded-xl border border-rose-500/30 bg-rose-500/10 p-6">
      <div className="flex flex-col items-center text-center space-y-4">
        <div className="size-12 rounded-full bg-rose-500/20 flex items-center justify-center">
          <AlertTriangle className="size-6 text-rose-400" />
        </div>
        <div>
          <h3 className="text-lg font-semibold text-rose-400">Something went wrong</h3>
          <p className="text-sm text-gray-400 mt-1">
            {error?.message || "An unexpected error occurred"}
          </p>
        </div>
        <Button
          variant="outline"
          onClick={resetError}
          className="border-rose-500/50 text-rose-400 hover:bg-rose-500/10"
        >
          <RefreshCw className="size-4 mr-2" />
          Try Again
        </Button>
      </div>
    </div>
  );
}
