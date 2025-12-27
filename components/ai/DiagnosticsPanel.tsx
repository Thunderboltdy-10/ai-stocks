"use client";

import { Radar, RadarChart, ResponsiveContainer, PolarGrid, PolarAngleAxis, PolarRadiusAxis, ScatterChart, Scatter, XAxis, YAxis, ZAxis, Tooltip, PieChart, Pie, Cell, BarChart, Bar, CartesianGrid } from "recharts";
import { BacktestResult, PredictionResult } from "@/types/ai";
import { formatPercent } from "@/utils/formatters";

interface DiagnosticsPanelProps {
  prediction?: PredictionResult | null;
  backtest?: BacktestResult | null;
}

const COLORS = ["#22c55e", "#ef4444", "#f97316", "#60a5fa"]; 

export function DiagnosticsPanel({ prediction, backtest }: DiagnosticsPanelProps) {
  const scatterData = prediction?.predictedReturns.map((value, index) => ({
    predicted: value,
    actual: prediction.actualReturns?.[index] ?? 0,
    idx: index,
  }));

  const probabilityHistogram = prediction?.classifierProbabilities.map((prob, index) => ({
    index,
    buy: prob.buy,
    sell: prob.sell,
    hold: prob.hold,
  }));

  const metricRadar = backtest
    ? [
        { metric: "Sharpe", value: backtest.metrics.sharpeRatio },
        { metric: "Win", value: backtest.metrics.winRate / 100 },
        { metric: "Drawdown", value: 1 + backtest.metrics.maxDrawdown },
        { metric: "DA", value: backtest.metrics.directionalAccuracy },
        { metric: "Trades", value: backtest.metrics.totalTrades / 20 },
      ]
    : [];

  const smapeRmse = backtest
    ? [
        { label: "SMAPE", value: backtest.metrics.smape ?? 0 },
        { label: "RMSE", value: backtest.metrics.rmse ?? 0 },
      ]
    : [];

  return (
    <div className="grid gap-4 md:grid-cols-2" aria-label="Diagnostics insights">
      <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-4">
        <h3 className="text-sm font-semibold text-gray-200">Regressor vs Actual</h3>
        {scatterData?.length ? (
          <ResponsiveContainer width="100%" height={220}>
            <ScatterChart margin={{ top: 10, right: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.04)" />
              <XAxis type="number" dataKey="predicted" tickFormatter={(value) => formatPercent(value)} stroke="#6b7280" />
              <YAxis type="number" dataKey="actual" tickFormatter={(value) => formatPercent(value)} stroke="#6b7280" />
              <ZAxis type="number" dataKey="idx" range={[30, 200]} />
              <Tooltip cursor={{ strokeDasharray: "3 3" }} formatter={(value, name) => [formatPercent(Number(value)), name]} contentStyle={{ background: "#111827", border: "1px solid #1f2937" }} />
              <Scatter data={scatterData} fill="#facc15" />
            </ScatterChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-xs text-gray-500">Run predictions to unlock scatter diagnostics.</p>
        )}
      </div>

      <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-4">
        <h3 className="text-sm font-semibold text-gray-200">Classifier Probability Spread</h3>
        {probabilityHistogram?.length ? (
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={probabilityHistogram} margin={{ top: 10 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
              <XAxis dataKey="index" stroke="#6b7280" tick={false} />
              <YAxis stroke="#6b7280" />
              <Tooltip
                formatter={(value, name) => [Number(value).toFixed(2), name.toUpperCase()]}
                wrapperStyle={{ background: "#111827", border: "1px solid #1f2937" }}
              />
              <Bar dataKey="buy" stackId="a" fill="#22c55e" />
              <Bar dataKey="sell" stackId="a" fill="#ef4444" />
              <Bar dataKey="hold" stackId="a" fill="#eab308" />
            </BarChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-xs text-gray-500">Classifier outputs will populate here.</p>
        )}
      </div>

      <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-4">
        <h3 className="text-sm font-semibold text-gray-200">Performance Radar</h3>
        {metricRadar.length ? (
          <ResponsiveContainer width="100%" height={220}>
            <RadarChart data={metricRadar}>
              <PolarGrid stroke="rgba(255,255,255,0.05)" />
              <PolarAngleAxis dataKey="metric" stroke="#6b7280" />
              <PolarRadiusAxis tick={{ fill: "#4b5563" }} angle={30} domain={[0, 1.5]} />
              <Radar dataKey="value" stroke="#34d399" fill="rgba(52,211,153,0.3)" />
            </RadarChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-xs text-gray-500">Run a backtest to see radar metrics.</p>
        )}
      </div>

      <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-4">
        <h3 className="text-sm font-semibold text-gray-200">SMAPE vs RMSE</h3>
        {smapeRmse.length ? (
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={smapeRmse} dataKey="value" innerRadius={40} outerRadius={80} paddingAngle={4}>
                {smapeRmse.map((entry, index) => (
                  <Cell key={`cell-${entry.label}`} fill={COLORS[index % COLORS.length]} />
                ))}
              </Pie>
            </PieChart>
          </ResponsiveContainer>
        ) : (
          <p className="text-xs text-gray-500">SMAPE/RMSE metrics appear after evaluation.</p>
        )}
      </div>
    </div>
  );
}
