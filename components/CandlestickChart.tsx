'use client';
import React, { useEffect, useRef } from 'react';
import { createChart, IChartApi } from 'lightweight-charts';

interface IndividualPrediction {
  horizon: string | number;
  price: number;
  weight?: number;
  confidence?: number;
  recommendation?: string;
}

interface EnsembleData {
  ensemble_price: number;
  individual_predictions?: IndividualPrediction[];
}

const CandlestickChart: React.FC<CandlestickChartProps & { ensembleData?: EnsembleData }> = ({ symbol, data, prediction, ensembleData }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current || !data || data.length === 0) return;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 500,
      layout: {
        background: { color: '#1f2937' },
        textColor: '#d1d5db',
      },
      grid: {
        vertLines: { color: 'rgba(255, 255, 255, 0.08)' },
        horzLines: { color: 'rgba(255, 255, 255, 0.08)' },
      },
      crosshair: { mode: 1 },
      timeScale: { borderColor: '#4b5563', timeVisible: true },
    });

    chartRef.current = chart;

    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#26a69a',
      downColor: '#ef5350',
      borderVisible: false,
      wickUpColor: '#26a69a',
      wickDownColor: '#ef5350',
    });

    const formattedData = data
      .map(d => {
        let dateStr;
        try {
          const date = new Date(d.date);
          dateStr = date.toISOString().split('T')[0];
        } catch {
          dateStr = d.date;
        }
        return {
          time: dateStr,
          open: Number(d.open),
          high: Number(d.high),
          low: Number(d.low),
          close: Number(d.close),
        };
      })
      .sort((a, b) => a.time.localeCompare(b.time));

    candlestickSeries.setData(formattedData);

    // Helper function to calculate trading days ahead
    const getTradingDaysAhead = (startDate: Date, days: number): string => {
      const date = new Date(startDate);
      let tradingDaysAdded = 0;
      while (tradingDaysAdded < days) {
        date.setDate(date.getDate() + 1);
        if (date.getDay() !== 0 && date.getDay() !== 6) {
          tradingDaysAdded++;
        }
      }
      return date.toISOString().split('T')[0];
    };

    // Add ensemble prediction candles that bridge from the latest close to each horizon
    if (formattedData.length > 0 && (ensembleData || prediction)) {
      const lastCandle = formattedData[formattedData.length - 1];
      const lastDate = new Date(lastCandle.time);
      const individualPredictions = (ensembleData?.individual_predictions || []).filter(
        (pred): pred is IndividualPrediction => pred != null && typeof pred.price === 'number'
      );

      const predictionSeries = chart.addCandlestickSeries({
        upColor: '#FACC15',
        downColor: '#F97316',
        wickUpColor: '#FACC15',
        wickDownColor: '#F97316',
        borderVisible: true,
        priceScaleId: 'right',
      });

      const predictionCandles: { time: string; open: number; high: number; low: number; close: number }[] = [];

      const appendPredictionCandle = (horizonDays: number, predictedPrice: number) => {
        if (!Number.isFinite(horizonDays) || !Number.isFinite(predictedPrice)) {
          return;
        }
        const targetDate = getTradingDaysAhead(lastDate, horizonDays);
        const prevClose = predictionCandles.length > 0 ? predictionCandles[predictionCandles.length - 1].close : lastCandle.close;
        const openPrice = prevClose;
        predictionCandles.push({
          time: targetDate,
          open: openPrice,
          close: predictedPrice,
          high: Math.max(openPrice, predictedPrice),
          low: Math.min(openPrice, predictedPrice),
        });
      };

      if (individualPredictions.length > 0) {
        const sortedPredictions = [...individualPredictions].sort((a, b) => {
          const aHorizon = typeof a.horizon === 'string' ? parseInt(a.horizon, 10) : a.horizon;
          const bHorizon = typeof b.horizon === 'string' ? parseInt(b.horizon, 10) : b.horizon;
          return (aHorizon || 0) - (bHorizon || 0);
        });

        sortedPredictions.forEach((pred) => {
          const horizonDays = typeof pred.horizon === 'string' ? parseInt(pred.horizon, 10) : pred.horizon;
          appendPredictionCandle(horizonDays, Number(pred.price));
        });
      } else {
        const fallbackPrice = prediction?.predicted_price ?? prediction?.predicted_close ?? ensembleData?.ensemble_price;
        if (fallbackPrice !== undefined) {
          appendPredictionCandle(5, Number(fallbackPrice));
        }
      }

      if (predictionCandles.length > 0) {
        predictionSeries.setData(predictionCandles);
      }
    }

    chart.timeScale().fitContent();

    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);

    return () => {
      window.removeEventListener('resize', handleResize);
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null; // ✅ prevents "Object is disposed" errors
      }
    };
  }, [data, prediction, ensembleData]);

  return (
    <div className="bg-gray-800 rounded-lg shadow-lg p-6 border border-gray-700">
      <h3 className="text-xl font-bold text-white mb-4">{symbol} - Candlestick Chart</h3>
      <div ref={chartContainerRef} className="chart-wrapper" />
    </div>
  );
};

export default CandlestickChart;
