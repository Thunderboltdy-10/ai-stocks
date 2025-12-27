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

const LineChartComponent: React.FC<LineChartProps & { ensembleData?: EnsembleData }> = ({
  symbol,
  historicalData,
  prediction,
  ensembleData,
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current || !historicalData || historicalData.length === 0) return;

    const chart = createChart(chartContainerRef.current, {
      width: chartContainerRef.current.clientWidth,
      height: 500,
      layout: {
        background: { color: '#1f2937' },
        textColor: '#d1d5db',
      },
      grid: {
        vertLines: { color: 'rgba(255, 255, 255, 0.08)' },
        horzLines: { color: 'rgba(255, 255, 255, 0)' }, // Hide horizontal lines
      },
      crosshair: { mode: 1 },
      timeScale: { 
        borderColor: '#4b5563', 
        timeVisible: true,
        tickMarkFormatter: (time: any) => {
          const date = new Date(time);
          const month = date.getMonth() + 1;
          const day = date.getDate();
          return `${month}/${day}`;
        },
      },
    });

    chartRef.current = chart;

    // Format historical data
    const formattedData = historicalData
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
          value: Number(d.close), // lightweight-charts expects 'value', not 'close'
        };
      })
      .sort((a, b) => a.time.localeCompare(b.time));

    // Determine trend direction based on historical data
    const prices = formattedData.map(d => d.value);
    const firstFive = prices.slice(0, Math.min(5, prices.length));
    const lastFive = prices.slice(-5);
    const avgFirstFive = firstFive.reduce((a, b) => a + b, 0) / firstFive.length;
    const avgLastFive = lastFive.reduce((a, b) => a + b, 0) / lastFive.length;
    const isUptrend = avgLastFive > avgFirstFive;

    // Add main line series for historical data (colored by trend)
    const lineSeries = chart.addLineSeries({
      color: isUptrend ? '#22c55e' : '#ef5350', // green or red based on trend
      lineWidth: 3,
      crosshairMarkerVisible: true,
      crosshairMarkerRadius: 6,
    });

    lineSeries.setData(formattedData);

    // Helper function to calculate trading days ahead
    const getTradingDaysAhead = (startDate: Date, daysAhead: number): string => {
      const date = new Date(startDate);
      let tradingDaysAdded = 0;
      while (tradingDaysAdded < daysAhead) {
        date.setDate(date.getDate() + 1);
        // Skip weekends
        if (date.getDay() !== 0 && date.getDay() !== 6) {
          tradingDaysAdded++;
        }
      }
      return date.toISOString().split('T')[0];
    };

    // Helper function to interpolate between two points
    const interpolatePoints = (
      startTime: string,
      startValue: number,
      endTime: string,
      endValue: number,
      numInterpolations: number = 5  // Reduced from 20 for cleaner, more continuous look
    ): Array<{ time: string; value: number }> => {
      const points: Array<{ time: string; value: number }> = [
        { time: startTime, value: startValue },
      ];

      const startDate = new Date(startTime);
      const endDate = new Date(endTime);
      const timeDiff = endDate.getTime() - startDate.getTime();
      const valueDiff = endValue - startValue;

      for (let i = 1; i < numInterpolations; i++) {
        const progress = i / numInterpolations;
        const interpolatedDate = new Date(startDate.getTime() + timeDiff * progress);
        const interpolatedValue = startValue + valueDiff * progress;
        const dateStr = interpolatedDate.toISOString().split('T')[0];

        // Avoid duplicate points
        if (dateStr !== points[points.length - 1].time) {
          points.push({ time: dateStr, value: interpolatedValue });
        }
      }

      points.push({ time: endTime, value: endValue });
      return points;
    };

    // Add prediction - ONLY show ensemble for cleaner visualization
    if (ensembleData && formattedData.length > 0) {
      const lastCandle = formattedData[formattedData.length - 1];
      const lastDate = new Date(lastCandle.time);

      console.log('Ensemble Data:', ensembleData);
      console.log('Individual Predictions:', ensembleData.individual_predictions);

      // Add ensemble prediction as gray points at 3, 5, 10 days
      try {
        const ensemblePredictionSeries = chart.addLineSeries({
          color: '#9CA3AF', // Gray-400 for ensemble predictions
          lineWidth: 3,
          lineStyle: 0, // solid line
          crosshairMarkerVisible: true,
          crosshairMarkerRadius: 6,
          priceLineVisible: false,
          lastValueVisible: false,
          pointMarkersVisible: true,
          pointMarkersRadius: 4,
        });

        // Create prediction points at 3, 5, and 10 days using individual model predictions
        const predictionPoints: { time: string; value: number }[] = [];

        const individualPredictions = (ensembleData.individual_predictions || []).filter(
          (pred): pred is IndividualPrediction => pred != null && typeof pred.price === 'number'
        );

        console.log('Processing individual predictions, count:', individualPredictions.length);

        if (individualPredictions.length > 0) {
          const sortedPredictions = [...individualPredictions].sort((a, b) => {
            const aHorizon = typeof a.horizon === 'string' ? parseInt(a.horizon, 10) : a.horizon;
            const bHorizon = typeof b.horizon === 'string' ? parseInt(b.horizon, 10) : b.horizon;
            return (aHorizon || 0) - (bHorizon || 0);
          });

          sortedPredictions.forEach((pred) => {
            const horizonDays = typeof pred.horizon === 'string' ? parseInt(pred.horizon, 10) : pred.horizon;
            if (!Number.isFinite(horizonDays)) {
              return;
            }

            const predDate = getTradingDaysAhead(lastDate, horizonDays);
            console.log(`Adding prediction point: ${horizonDays} days, date: ${predDate}, price: ${pred.price}`);
            predictionPoints.push({
              time: predDate,
              value: Number(pred.price),
            });
          });
        }

        if (predictionPoints.length === 0) {
          // Fall back to single ensemble point at 10 days if no individual predictions were available
          console.log('No individual predictions, using ensemble price');
          const predictionDateStr = getTradingDaysAhead(lastDate, 10);
          predictionPoints.push({
            time: predictionDateStr,
            value: Number(ensembleData.ensemble_price),
          });
        }

        const seriesPoints = [
          { time: lastCandle.time, value: lastCandle.value },
          ...predictionPoints,
        ];

        console.log('Final prediction points:', seriesPoints);
        ensemblePredictionSeries.setData(seriesPoints);

      } catch (err) {
        console.warn('Could not draw ensemble prediction line', err);
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
        chartRef.current = null;
      }
    };
  }, [historicalData, prediction, ensembleData]);

  return (
    <div className="bg-gray-800 rounded-lg shadow-lg p-6 border border-gray-700">
      <h3 className="text-xl font-bold text-white mb-4">{symbol} - Line Chart</h3>
      <div ref={chartContainerRef} className="chart-wrapper" />
    </div>
  );
};

export default LineChartComponent;
