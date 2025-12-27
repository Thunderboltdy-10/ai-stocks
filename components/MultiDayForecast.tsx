'use client';
import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Area, AreaChart } from 'recharts';
import { toast } from 'sonner';

interface PredictionDay {
  day: number;
  predicted_price: number;
  log_return: number;
  return_pct: number;
  recommendation: 'BUY' | 'SELL' | 'HOLD';
  class_confidence: number;
  adjusted_confidence: number;
  confidence_factor: number;
}

interface MultiDayData {
  symbol: string;
  risk_profile: string;
  predictions: PredictionDay[];
}

interface MultiDayForecastProps {
  symbol: string;
  riskProfile: 'conservative' | 'aggressive';
  currentPrice: number;
}

export default function MultiDayForecast({ symbol, riskProfile: initialRiskProfile, currentPrice }: MultiDayForecastProps) {
  const [forecastData, setForecastData] = useState<MultiDayData | null>(null);
  const [loading, setLoading] = useState(false);
  const [days, setDays] = useState(5);
  const [riskProfile, setRiskProfile] = useState<'conservative' | 'aggressive'>(initialRiskProfile);

  const runForecast = async () => {
    if (days < 1 || days > 10) {
      toast.error('Days must be between 1 and 10');
      return;
    }

    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/predict-multiday', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: symbol,
          risk_profile: riskProfile,  // Use state variable
          days: days,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Forecast failed');
      }

      const data = await response.json();
      setForecastData(data);
      toast.success(`${days}-day forecast completed!`);
    } catch (error) {
      console.error('Forecast error:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to generate forecast');
    } finally {
      setLoading(false);
    }
  };

  // Prepare chart data
  const chartData = forecastData?.predictions.map((pred) => {
    // Calculate uncertainty band (±2% per day)
    const uncertainty = 2 * pred.day;
    return {
      day: `Day ${pred.day}`,
      Price: pred.predicted_price,
      'Upper Band': pred.predicted_price * (1 + uncertainty / 100),
      'Lower Band': pred.predicted_price * (1 - uncertainty / 100),
      Confidence: pred.adjusted_confidence * 100,
    };
  }) || [];

  // Add current price as day 0
  if (chartData.length > 0) {
    chartData.unshift({
      day: 'Today',
      Price: currentPrice,
      'Upper Band': currentPrice,
      'Lower Band': currentPrice,
      Confidence: 100,
    });
  }

  return (
    <div className="bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-700">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-white mb-2">🔮 Multi-Day Forecast</h2>
        <p className="text-gray-400 text-sm">
          Predict future prices with confidence degradation tracking
        </p>
        <div className="mt-2 p-3 bg-yellow-900 bg-opacity-20 border border-yellow-600 rounded-lg">
          <p className="text-yellow-200 text-xs">
            ⚠️ <strong>Warning:</strong> Predictions become less reliable beyond day 1. 
            Confidence degrades by 15% per day due to uncertainty accumulation.
          </p>
        </div>
      </div>

      {/* Controls */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Risk Profile
          </label>
          <select
            value={riskProfile}
            onChange={(e) => setRiskProfile(e.target.value as 'conservative' | 'aggressive')}
            className="w-full px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-lg focus:border-yellow-500 focus:outline-none"
          >
            <option value="conservative">Conservative</option>
            <option value="aggressive">Aggressive</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Forecast Days (1-10)
          </label>
          <input
            type="number"
            value={days}
            onChange={(e) => setDays(Number(e.target.value))}
            min="1"
            max="10"
            className="w-full px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-lg focus:border-yellow-500 focus:outline-none"
          />
        </div>
        <div className="flex items-end">
          <button
            onClick={runForecast}
            disabled={loading}
            className="w-full px-4 py-2 bg-yellow-500 text-gray-900 font-semibold rounded-lg hover:bg-yellow-400 disabled:bg-gray-600 disabled:text-gray-400 transition-all"
          >
            {loading ? 'Forecasting...' : 'Generate Forecast'}
          </button>
        </div>
      </div>

      {/* Results */}
      {forecastData && (
        <>
          {/* Price Forecast Chart */}
          <div className="bg-gray-900 p-4 rounded-lg mb-6">
            <h3 className="text-sm font-semibold text-gray-300 mb-3">Price Forecast with Uncertainty</h3>
            <ResponsiveContainer width="100%" height={300}>
              <AreaChart data={chartData}>
                <defs>
                  <linearGradient id="colorUncertainty" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.3}/>
                    <stop offset="95%" stopColor="#3B82F6" stopOpacity={0}/>
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="day" 
                  stroke="#9CA3AF"
                  style={{ fontSize: '12px' }}
                />
                <YAxis 
                  stroke="#9CA3AF"
                  style={{ fontSize: '12px' }}
                  tickFormatter={(value) => `$${value.toFixed(0)}`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1F2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    color: '#F9FAFB',
                  }}
                  formatter={(value: number) => `$${value.toFixed(2)}`}
                />
                <Legend 
                  wrapperStyle={{ paddingTop: '20px' }}
                />
                <Area
                  type="monotone"
                  dataKey="Upper Band"
                  stackId="1"
                  stroke="none"
                  fill="url(#colorUncertainty)"
                />
                <Area
                  type="monotone"
                  dataKey="Lower Band"
                  stackId="1"
                  stroke="none"
                  fill="url(#colorUncertainty)"
                />
                <Line
                  type="monotone"
                  dataKey="Price"
                  stroke="#3B82F6"
                  strokeWidth={3}
                  dot={{ fill: '#3B82F6', r: 4 }}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Confidence Degradation Chart */}
          <div className="bg-gray-900 p-4 rounded-lg mb-6">
            <h3 className="text-sm font-semibold text-gray-300 mb-3">Prediction Confidence</h3>
            <ResponsiveContainer width="100%" height={200}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="day" 
                  stroke="#9CA3AF"
                  style={{ fontSize: '12px' }}
                />
                <YAxis 
                  stroke="#9CA3AF"
                  style={{ fontSize: '12px' }}
                  domain={[0, 100]}
                  tickFormatter={(value) => `${value}%`}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1F2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    color: '#F9FAFB',
                  }}
                  formatter={(value: number) => `${value.toFixed(1)}%`}
                />
                <Line
                  type="monotone"
                  dataKey="Confidence"
                  stroke="#10B981"
                  strokeWidth={2}
                  dot={{ fill: '#10B981', r: 4 }}
                />
                <Line
                  type="monotone"
                  dataKey={50}
                  stroke="#EF4444"
                  strokeWidth={1}
                  strokeDasharray="5 5"
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
            <p className="text-xs text-gray-400 mt-2 text-center">
              Red line = Random guess level (50%)
            </p>
          </div>

          {/* Predictions Table */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-3">Daily Predictions</h3>
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-gray-800">
                  <tr>
                    <th className="px-4 py-3 text-left text-gray-300 font-medium">Day</th>
                    <th className="px-4 py-3 text-right text-gray-300 font-medium">Price</th>
                    <th className="px-4 py-3 text-right text-gray-300 font-medium">Change</th>
                    <th className="px-4 py-3 text-center text-gray-300 font-medium">Signal</th>
                    <th className="px-4 py-3 text-right text-gray-300 font-medium">Confidence</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800">
                  {forecastData.predictions.map((pred) => (
                    <tr key={pred.day} className="hover:bg-gray-800 transition-colors">
                      <td className="px-4 py-3 text-gray-300 font-medium">
                        Day +{pred.day}
                      </td>
                      <td className="px-4 py-3 text-right text-white font-semibold">
                        ${pred.predicted_price.toFixed(2)}
                      </td>
                      <td className="px-4 py-3 text-right">
                        <span className={pred.return_pct >= 0 ? 'text-green-400' : 'text-red-400'}>
                          {pred.return_pct >= 0 ? '+' : ''}{pred.return_pct.toFixed(2)}%
                        </span>
                      </td>
                      <td className="px-4 py-3 text-center">
                        <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded ${
                          pred.recommendation === 'BUY' 
                            ? 'bg-green-900 text-green-300' 
                            : pred.recommendation === 'SELL'
                            ? 'bg-red-900 text-red-300'
                            : 'bg-yellow-900 text-yellow-300'
                        }`}>
                          {pred.recommendation}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right">
                        <div className="flex flex-col items-end">
                          <span className="text-gray-300">
                            {(pred.adjusted_confidence * 100).toFixed(1)}%
                          </span>
                          <span className="text-xs text-gray-500">
                            ({(pred.confidence_factor * 100).toFixed(0)}% reliable)
                          </span>
                        </div>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Summary Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-6">
            <div className="p-4 bg-gray-700 rounded-lg border border-gray-600">
              <p className="text-xs text-gray-400 mb-1">Final Price (Day {days})</p>
              <p className="text-xl font-bold text-white">
                ${forecastData.predictions[days - 1].predicted_price.toFixed(2)}
              </p>
            </div>
            <div className="p-4 bg-gray-700 rounded-lg border border-gray-600">
              <p className="text-xs text-gray-400 mb-1">Total Expected Return</p>
              <p className={`text-xl font-bold ${
                forecastData.predictions[days - 1].return_pct >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {forecastData.predictions[days - 1].return_pct >= 0 ? '+' : ''}
                {forecastData.predictions[days - 1].return_pct.toFixed(2)}%
              </p>
            </div>
            <div className="p-4 bg-gray-700 rounded-lg border border-gray-600">
              <p className="text-xs text-gray-400 mb-1">Average Confidence</p>
              <p className="text-xl font-bold text-white">
                {(forecastData.predictions.reduce((sum, p) => sum + p.adjusted_confidence, 0) / days * 100).toFixed(1)}%
              </p>
            </div>
            <div className="p-4 bg-gray-700 rounded-lg border border-gray-600">
              <p className="text-xs text-gray-400 mb-1">Trend</p>
              <p className="text-xl font-bold text-yellow-400">
                {forecastData.predictions.filter(p => p.recommendation === 'BUY').length > days / 2 
                  ? '📈 Bullish' 
                  : forecastData.predictions.filter(p => p.recommendation === 'SELL').length > days / 2
                  ? '📉 Bearish'
                  : '⏸️ Neutral'}
              </p>
            </div>
          </div>
        </>
      )}

      {!forecastData && !loading && (
        <div className="text-center py-12 text-gray-400">
          <p>Click "Generate Forecast" to predict future prices</p>
        </div>
      )}
    </div>
  );
}
