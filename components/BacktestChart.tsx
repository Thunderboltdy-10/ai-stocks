'use client';
import React, { useState } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { toast } from 'sonner';

interface BacktestData {
  symbol: string;
  risk_profile: string;
  initial_capital: number;
  final_value: number;
  total_return: number;
  trades: Array<{
    date: string;
    action: string;
    price: number;
    shares: number;
    cost?: number;
    proceeds?: number;
    profit?: number;
    confidence: number;
  }>;
  daily_values: Array<{
    date: string;
    portfolio_value: number;
    price: number;
  }>;
}

interface BacktestChartProps {
  symbol: string;
  riskProfile: 'conservative' | 'aggressive';
}

export default function BacktestChart({ symbol, riskProfile: initialRiskProfile }: BacktestChartProps) {
  const [backtestData, setBacktestData] = useState<BacktestData | null>(null);
  const [loading, setLoading] = useState(false);
  const [days, setDays] = useState(90);
  const [capital, setCapital] = useState(10000);
  const [riskProfile, setRiskProfile] = useState<'conservative' | 'aggressive'>(initialRiskProfile);

  const runBacktest = async () => {
    setLoading(true);
    try {
      const response = await fetch('http://localhost:8000/api/backtest', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          symbol: symbol,
          risk_profile: riskProfile,
          days: days,
          initial_capital: capital,
        }),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Backtest failed');
      }

      const data = await response.json();
      setBacktestData(data);
      toast.success('Backtest completed successfully!');
    } catch (error) {
      console.error('Backtest error:', error);
      toast.error(error instanceof Error ? error.message : 'Failed to run backtest');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="bg-gray-800 rounded-xl shadow-lg p-6 border border-gray-700 mt-6">
      <div className="mb-6">
        <h2 className="text-2xl font-bold text-white mb-2">📊 Strategy Backtest</h2>
        <p className="text-gray-400 text-sm">
          Test historical performance of AI trading strategy vs buy-and-hold
        </p>
        <p className="text-gray-500 text-xs mt-1">
          Hybrid 3d/5d/10d ensemble with dual-risk models
        </p>
      </div>

      {/* Controls */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Risk Profile
          </label>
          <select
            value={riskProfile}
            onChange={(e) => setRiskProfile(e.target.value as 'conservative' | 'aggressive')}
            className="w-full px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-lg focus:border-yellow-500 focus:outline-none"
            title="Select risk profile"
          >
            <option value="conservative">Conservative</option>
            <option value="aggressive">Aggressive</option>
          </select>
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Test Period (Days)
          </label>
          <input
            type="number"
            value={days}
            onChange={(e) => setDays(Number(e.target.value))}
            min="30"
            max="365"
            className="w-full px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-lg focus:border-yellow-500 focus:outline-none"
            title="Test period in days"
          />
        </div>
        <div>
          <label className="block text-sm font-medium text-gray-300 mb-2">
            Initial Capital ($)
          </label>
          <input
            type="number"
            value={capital}
            onChange={(e) => setCapital(Number(e.target.value))}
            min="1000"
            step="1000"
            className="w-full px-3 py-2 bg-gray-700 text-white border border-gray-600 rounded-lg focus:border-yellow-500 focus:outline-none"
            title="Initial capital amount"
          />
        </div>
        <div className="flex items-end">
          <button
            onClick={runBacktest}
            disabled={loading}
            className="w-full px-4 py-2 bg-yellow-500 text-gray-900 font-semibold rounded-lg hover:bg-yellow-400 disabled:bg-gray-600 disabled:text-gray-400 transition-all"
          >
            {loading ? 'Running...' : 'Run Backtest'}
          </button>
        </div>
      </div>

      {/* Results */}
      {backtestData && (() => {
        const hasDailyValues = Array.isArray(backtestData.daily_values) && backtestData.daily_values.length > 0;
        
        if (!hasDailyValues) {
          console.warn('❌ No daily values received from backtest:', backtestData.daily_values);
          return <div className="text-center py-12 text-red-400">No chart data available</div>;
        }

        const baselinePrice = Number(backtestData.daily_values[0]?.price ?? 0) || 1;

        // Map daily values to chart format with consistent date handling
        const chartData = backtestData.daily_values
          .map((day, index) => {
            try {
              const closingPrice = Number(day.price ?? 0);
              const portfolioValue = Number(day.portfolio_value ?? 0);
              
              // Only include valid data points
              if (!closingPrice || !portfolioValue || portfolioValue <= 0) {
                return null;
              }

              const buyHoldValue = capital * (closingPrice / baselinePrice);

              return {
                date: day.date, // Keep raw date for consistency
                displayDate: new Date(day.date).toLocaleDateString('en-US', { month: 'short', day: 'numeric' }),
                aiValue: Math.round(portfolioValue * 100) / 100,
                buyHoldValue: Math.round(buyHoldValue * 100) / 100,
                index,
              };
            } catch (e) {
              console.warn(`⚠️  Error parsing day ${index}:`, day, e);
              return null;
            }
          })
          .filter((d) => d !== null);

        console.log(`📊 Chart data prepared: ${chartData.length} valid data points out of ${backtestData.daily_values.length}`);
        console.log('📈 First chart point:', chartData[0]);
        console.log('📈 Last chart point:', chartData[chartData.length - 1]);

        if (chartData.length === 0) {
          console.error('❌ No valid chart data after filtering');
          return <div className="text-center py-12 text-red-400">No valid data points to display</div>;
        }

        const profitableTradeCount = backtestData.trades?.filter((t) => t.profit && t.profit > 0).length || 0;
        const totalSellTrades = backtestData.trades?.filter((t) => t.action === 'SELL').length || 0;
        const winRate = totalSellTrades > 0 ? (profitableTradeCount / totalSellTrades) * 100 : 0;

        return (
        <>
          {/* Performance Metrics */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
            <div className="p-4 bg-gray-700 rounded-lg border border-gray-600">
              <p className="text-xs text-gray-400 mb-1">Initial Capital</p>
              <p className="text-2xl font-bold text-white">
                ${backtestData.initial_capital.toLocaleString()}
              </p>
            </div>
            <div className="p-4 bg-gray-700 rounded-lg border border-gray-600">
              <p className="text-xs text-gray-400 mb-1">Final Value</p>
              <p className="text-2xl font-bold text-yellow-400">
                ${backtestData.final_value.toLocaleString()}
              </p>
            </div>
            <div className="p-4 bg-gray-700 rounded-lg border border-gray-600">
              <p className="text-xs text-gray-400 mb-1">Total Return</p>
              <p className={`text-2xl font-bold ${
                backtestData.total_return >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {(backtestData.total_return * 100).toFixed(2)}%
              </p>
            </div>
            <div className="p-4 bg-gray-700 rounded-lg border border-gray-600">
              <p className="text-xs text-gray-400 mb-1">Win Rate</p>
              <p className="text-2xl font-bold text-white">
                {winRate.toFixed(1)}%
              </p>
              <p className="text-xs text-gray-400 mt-1">
                {profitableTradeCount}/{totalSellTrades} trades
              </p>
            </div>
          </div>

          {/* Chart */}
          <div className="bg-gray-900 p-4 rounded-lg mb-6">
            <ResponsiveContainer width="100%" height={500}>
              <LineChart data={chartData} margin={{ top: 5, right: 30, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis 
                  dataKey="displayDate" 
                  stroke="#9CA3AF"
                  style={{ fontSize: '12px' }}
                  tick={{ fontSize: 11 }}
                />
                <YAxis 
                  stroke="#9CA3AF"
                  style={{ fontSize: '12px' }}
                  tickFormatter={(value) => {
                    if (value >= 1000) {
                      return `$${(value / 1000).toFixed(0)}k`;
                    }
                    return `$${value.toFixed(0)}`;
                  }}
                  width={70}
                  type="number"
                  tick={{ fontSize: 12 }}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: '#1F2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    color: '#F9FAFB',
                  }}
                  formatter={(value: number | string, name: string) => {
                    const numericValue = typeof value === 'number' ? value : Number(value);
                    return [`$${(numericValue || 0).toLocaleString()}`, name];
                  }}
                />
                <Legend 
                  wrapperStyle={{ paddingTop: '20px' }}
                  iconType="line"
                />
                <Line
                  type="monotone"
                  dataKey="aiValue"
                  name="AI Strategy"
                  stroke="#00FF00"
                  strokeWidth={4}
                  isAnimationActive={false}
                  dot={false}
                  connectNulls
                />
                <Line
                  type="monotone"
                  dataKey="buyHoldValue"
                  name="Buy & Hold"
                  stroke="#FF8C00"
                  strokeWidth={3}
                  strokeDasharray="8 4"
                  isAnimationActive={false}
                  dot={false}
                  connectNulls
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Trade History */}
          <div>
            <h3 className="text-lg font-semibold text-white mb-3">Recent Trades</h3>
            <div className="bg-gray-900 rounded-lg overflow-hidden">
              <table className="w-full text-sm">
                <thead className="bg-gray-800">
                  <tr>
                    <th className="px-4 py-3 text-left text-gray-300 font-medium">Date</th>
                    <th className="px-4 py-3 text-left text-gray-300 font-medium">Action</th>
                    <th className="px-4 py-3 text-right text-gray-300 font-medium">Price</th>
                    <th className="px-4 py-3 text-right text-gray-300 font-medium">Shares</th>
                    <th className="px-4 py-3 text-right text-gray-300 font-medium">Profit/Loss</th>
                    <th className="px-4 py-3 text-right text-gray-300 font-medium">Confidence</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-800">
                  {backtestData.trades.slice(-10).reverse().map((trade, index) => (
                    <tr key={index} className="hover:bg-gray-800 transition-colors">
                      <td className="px-4 py-3 text-gray-300">
                        {new Date(trade.date).toLocaleDateString()}
                      </td>
                      <td className="px-4 py-3">
                        <span className={`inline-flex px-2 py-1 text-xs font-semibold rounded ${
                          trade.action === 'BUY' 
                            ? 'bg-green-900 text-green-300' 
                            : 'bg-red-900 text-red-300'
                        }`}>
                          {trade.action}
                        </span>
                      </td>
                      <td className="px-4 py-3 text-right text-gray-300">
                        ${trade.price.toFixed(2)}
                      </td>
                      <td className="px-4 py-3 text-right text-gray-300">
                        {trade.shares}
                      </td>
                      <td className="px-4 py-3 text-right">
                        {trade.profit !== undefined && (
                          <span className={trade.profit >= 0 ? 'text-green-400' : 'text-red-400'}>
                            {trade.profit >= 0 ? '+' : ''}${trade.profit.toFixed(2)}
                          </span>
                        )}
                      </td>
                      <td className="px-4 py-3 text-right text-gray-400">
                        {(trade.confidence * 100).toFixed(1)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
        );
      })()}

      {!backtestData && !loading && (
        <div className="text-center py-12 text-gray-400">
          <p>Click &quot;Run Backtest&quot; to test the strategy performance</p>
        </div>
      )}
    </div>
  );
};
