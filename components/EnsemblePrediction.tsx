"use client";

import { useEffect, useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";

interface IndividualPrediction {
  horizon: string;
  recommendation: string;
  confidence: number;
  price: number;
  return_pct: number;
  weight: number;
  class_probs: {
    sell: number;
    hold: number;
    buy: number;
  };
}

interface EnsemblePredictionResult {
  symbol: string;
  current_price: number;
  risk_profile: string;
  ensemble_recommendation: string;
  ensemble_confidence: number;
  ensemble_price: number;
  ensemble_return_pct: number;
  ensemble_class_probs: {
    sell: number;
    hold: number;
    buy: number;
  };
  individual_predictions: IndividualPrediction[];
  agreement_score: number;
  agreement_level: string;
}

interface EnsemblePredictionProps {
  symbol: string;
  riskProfile?: string;
}

const getRecommendationColor = (recommendation: string): string => {
  switch (recommendation?.toUpperCase()) {
    case "BUY":
      return "bg-gray-700 text-gray-100 border border-gray-600";
    case "SELL":
      return "bg-gray-700 text-gray-100 border border-gray-600";
    case "HOLD":
    default:
      return "bg-gray-700 text-gray-200 border border-gray-600";
  }
};

const getAgreementColor = (level: string): string => {
  if (!level) return "bg-gray-700 text-gray-200 border border-gray-600";
  if (level.includes("Strong")) {
    return "bg-gray-700 text-gray-100 border border-gray-600";
  }
  if (level.includes("Moderate")) {
    return "bg-gray-700 text-gray-100 border border-gray-600";
  }
  return "bg-gray-800 text-gray-100 border border-gray-600";
};

const formatRiskProfile = (profile?: string) => {
  if (!profile) return "";
  return profile.charAt(0).toUpperCase() + profile.slice(1);
};

const clamp01 = (value: number) => Math.min(1, Math.max(0, value));

interface SegmentBarProps {
  value: number;
  segments?: number;
  activeClassName?: string;
  inactiveClassName?: string;
}

const SegmentBar = ({
  value,
  segments = 20,
  activeClassName = "bg-gray-400",
  inactiveClassName = "bg-gray-800",
}: SegmentBarProps) => {
  const clampedValue = clamp01(value);
  const activeSegments = Math.round(clampedValue * segments);

  return (
    <div className="flex items-center gap-0.5">
      {Array.from({ length: segments }).map((_, index) => (
        <span
          key={index}
          className={`h-2 flex-1 rounded-sm ${
            index < activeSegments ? activeClassName : inactiveClassName
          }`}
        />
      ))}
    </div>
  );
};

export const EnsemblePrediction: React.FC<EnsemblePredictionProps> = ({
  symbol,
  riskProfile = "conservative",
}) => {
  const [prediction, setPrediction] = useState<EnsemblePredictionResult | null>(
    null,
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fetchPrediction = async () => {
      setLoading(true);
      setError(null);

      try {
        const response = await fetch("/api/predict-ensemble", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ symbol, riskProfile }),
        });

        if (!response.ok) {
          const errorData = await response.json().catch(() => ({}));
          const detail =
            errorData?.detail || errorData?.error || response.statusText;
          throw new Error(detail || "Failed to fetch prediction");
        }

        const data = await response.json();

        if (data.error) {
          throw new Error(data.error);
        }

        setPrediction(data as EnsemblePredictionResult);
      } catch (err) {
        if (err instanceof Error) {
          setError(err.message);
        } else {
          setError("Unknown error");
        }
      } finally {
        setLoading(false);
      }
    };

    if (symbol) {
      fetchPrediction();
    }
  }, [symbol, riskProfile]);

  if (loading) {
    return (
      <Card className="w-full bg-gray-800 border border-gray-700 text-gray-200">
        <CardHeader>
          <CardTitle>Ensemble Prediction</CardTitle>
          <CardDescription className="text-gray-400">
            Loading ensemble models...
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  if (error) {
    return (
      <Card className="w-full bg-gray-800 border border-gray-700 text-gray-200">
        <CardHeader>
          <CardTitle>Ensemble Prediction Error</CardTitle>
          <CardDescription className="text-gray-400">
            {error}
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  if (!prediction) {
    return (
      <Card className="w-full bg-gray-800 border border-gray-700 text-gray-200">
        <CardHeader>
          <CardTitle>Ensemble Prediction</CardTitle>
          <CardDescription className="text-gray-400">
            No data available
          </CardDescription>
        </CardHeader>
      </Card>
    );
  }

  const priceChange = prediction.ensemble_price - prediction.current_price;
  const priceChangeColor =
    priceChange > 0
      ? "text-emerald-300"
      : priceChange < 0
        ? "text-rose-300"
        : "text-gray-300";

  return (
    <div className="w-full space-y-4 text-gray-200">
      <Card className="w-full bg-gray-800 border border-gray-700">
        <CardHeader className="border-b border-gray-700">
          <div className="flex items-center justify-between">
            <div>
              <CardTitle className="text-2xl text-gray-100">
                {symbol} Ensemble Prediction
              </CardTitle>
              <CardDescription className="text-gray-400">
                {formatRiskProfile(prediction.risk_profile)} risk stance
              </CardDescription>
            </div>
            <Badge className={getRecommendationColor(
              prediction.ensemble_recommendation,
            )}>
              {prediction.ensemble_recommendation}
            </Badge>
          </div>
        </CardHeader>

        <CardContent className="space-y-6">
          <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
            <div className="space-y-1">
              <p className="text-sm text-gray-400">Current Price</p>
              <p className="text-2xl font-semibold text-gray-50">
                ${prediction.current_price.toFixed(2)}
              </p>
            </div>
            <div className="space-y-1">
              <p className="text-sm text-gray-400">Target Price</p>
              <p className="text-2xl font-semibold text-gray-50">
                ${prediction.ensemble_price.toFixed(2)}
              </p>
            </div>
            <div className="space-y-1">
              <p className="text-sm text-gray-400">Expected Change</p>
              <p className={`text-2xl font-semibold ${priceChangeColor}`}>
                {priceChange >= 0 ? "+" : ""}
                {priceChange.toFixed(2)}
                <span className="ml-1 text-base text-gray-400">
                  ({prediction.ensemble_return_pct.toFixed(2)}%)
                </span>
              </p>
            </div>
          </div>

          <div className="grid gap-4 border-t border-gray-700 pt-4 md:grid-cols-2 lg:grid-cols-3">
            <div className="space-y-2">
              <p className="text-sm text-gray-400">Ensemble Confidence</p>
              <div className="flex items-center space-x-3">
                <SegmentBar value={prediction.ensemble_confidence} />
                <span className="text-sm font-semibold text-gray-200">
                  {(prediction.ensemble_confidence * 100).toFixed(0)}%
                </span>
              </div>
            </div>
            <div className="space-y-2">
              <p className="text-sm text-gray-400">Model Agreement</p>
              <Badge className={getAgreementColor(prediction.agreement_level)}>
                {prediction.agreement_level} ({
                  (prediction.agreement_score * 100).toFixed(0)
                }%)
              </Badge>
            </div>
            <div className="space-y-2">
              <p className="text-sm font-semibold text-gray-300">
                Class Probabilities
              </p>
              {["buy", "hold", "sell"].map((bucket) => {
                const value =
                  prediction.ensemble_class_probs[
                    bucket as keyof EnsemblePredictionResult["ensemble_class_probs"]
                  ];
                const label = bucket.charAt(0).toUpperCase() + bucket.slice(1);

                return (
                  <div key={bucket} className="space-y-1">
                    <div className="flex items-center justify-between text-sm text-gray-300">
                      <span>{label}</span>
                      <span>{(value * 100).toFixed(1)}%</span>
                    </div>
                    <SegmentBar
                      value={value}
                      activeClassName={
                        bucket === "buy"
                          ? "bg-gray-300"
                          : bucket === "hold"
                            ? "bg-gray-500"
                            : "bg-gray-600"
                      }
                    />
                  </div>
                );
              })}
            </div>
          </div>
        </CardContent>
      </Card>

      <div className="space-y-3">
        <h3 className="text-lg font-semibold text-gray-100">
          Individual Model Predictions
        </h3>
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {prediction.individual_predictions.map((pred) => {
            return (
              <Card
                key={`${pred.horizon}-${pred.recommendation}`}
                className="bg-gray-800 border border-gray-700 text-gray-200"
              >
                <CardHeader className="border-b border-gray-700 pb-3">
                  <div className="flex items-start justify-between">
                    <div>
                      <CardTitle className="text-base text-gray-100">
                        {pred.horizon.toUpperCase()} Horizon
                      </CardTitle>
                      <CardDescription className="text-gray-400">
                        Weight: {(pred.weight * 100).toFixed(0)}%
                      </CardDescription>
                    </div>
                    <Badge className="bg-gray-700 text-gray-100 border border-gray-600">
                      {pred.recommendation}
                    </Badge>
                  </div>
                </CardHeader>
                <CardContent className="space-y-4 pt-4">
                  <div className="space-y-1">
                    <p className="text-sm text-gray-400">Price Target</p>
                    <p className="text-lg font-semibold text-gray-50">
                      ${pred.price.toFixed(2)}
                    </p>
                    <p className="text-sm text-gray-500">
                      {pred.return_pct >= 0 ? "+" : ""}
                      {pred.return_pct.toFixed(2)}%
                    </p>
                  </div>
                  <div className="space-y-2">
                    <p className="text-sm text-gray-400">Confidence</p>
                    <div className="flex items-center space-x-3">
                      <SegmentBar value={pred.confidence} />
                      <span className="text-xs font-semibold text-gray-200">
                        {(clamp01(pred.confidence) * 100).toFixed(0)}%
                      </span>
                    </div>
                  </div>
                </CardContent>
              </Card>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default EnsemblePrediction;
