"use client";

import { forwardRef, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import { createChart, CrosshairMode, IChartApi, LineStyle, SeriesMarker, SeriesMarkerShape, Time } from "lightweight-charts";
import { OverlaySeries, TradeMarker } from "@/types/ai";

export interface CandlestickPoint {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume?: number;
}

export interface ChartMarker extends TradeMarker {
  displayType?: "buy" | "sell" | "hold";
}

interface InteractiveCandlestickProps {
  historicalSeries: CandlestickPoint[];
  predictedSeries?: Array<{ date: string; price: number; segment?: "history" | "forecast" }>;
  tradeMarkers?: ChartMarker[];
  overlays?: OverlaySeries[];
  forecastChanges?: Array<{ date: string; value: number }>;
  onRangeChange?: (range: { from: Time; to: Time } | null) => void;
  renderMode?: "candlestick" | "line";
  mode?: "history" | "prediction" | "backtest";
  showBuyMarkers?: boolean;
  showSellMarkers?: boolean;
  segmentFilters?: { history: boolean; forecast: boolean };
  scopeFilters?: { prediction: boolean; backtest: boolean };
  ariaLabel?: string;
}

export interface CandlestickChartHandle {
  exportImage: () => void;
}

const formatTimeLabel = (time: Time): string => {
  if (typeof time === "string") return time;
  if (typeof time === "number") {
    return new Date(time * 1000).toISOString().slice(0, 10);
  }
  const month = `${time.month}`.padStart(2, "0");
  const day = `${time.day}`.padStart(2, "0");
  return `${time.year}-${month}-${day}`;
};

const InteractiveCandlestick = forwardRef<CandlestickChartHandle, InteractiveCandlestickProps>(
  (
    {
      historicalSeries,
      overlays,
      tradeMarkers,
      predictedSeries,
      forecastChanges,
      onRangeChange,
      renderMode = "candlestick",
      mode = "history",
      showBuyMarkers = true,
      showSellMarkers = true,
      segmentFilters = { history: true, forecast: true },
      scopeFilters = { prediction: true, backtest: true },
      ariaLabel,
    },
    ref
  ) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const chartRef = useRef<IChartApi | null>(null);
    const priceSeriesRef = useRef<ReturnType<IChartApi["addCandlestickSeries"]> | ReturnType<IChartApi["addLineSeries"]> | null>(null);
    const [hoverInfo, setHoverInfo] = useState<{ date: string; price?: number; signal?: string } | null>(null);

    useImperativeHandle(ref, () => ({
      exportImage: () => {
        if (!chartRef.current) return;
        const canvas = chartRef.current.takeScreenshot();
        canvas.toBlob((blob) => {
          if (!blob) return;
          const link = document.createElement("a");
          link.href = URL.createObjectURL(blob);
          link.download = "candlestick.png";
          link.click();
          URL.revokeObjectURL(link.href);
        });
      },
    }));

    const filteredMarkers = useMemo(() => {
      if (!tradeMarkers?.length) {
        console.warn("⚠️  No markers received from backend");
        return [];
      }

      const counts = {
        buy: tradeMarkers.filter((m) => (m.displayType ?? m.type) === "buy").length,
        sell: tradeMarkers.filter((m) => (m.displayType ?? m.type) === "sell").length,
        hold: tradeMarkers.filter((m) => (m.displayType ?? m.type) === "hold").length,
      };
      console.log(`📊 Total markers: BUY=${counts.buy}, SELL=${counts.sell}, HOLD=${counts.hold}`);

      const filtered = tradeMarkers.filter((marker) => {
        const markerType = (marker.displayType ?? marker.type) as "buy" | "sell" | "hold";
        if (markerType === "buy" && !showBuyMarkers) return false;
        if (markerType === "sell" && !showSellMarkers) return false;
        if ((marker.segment ?? "history") === "history" && !segmentFilters.history) return false;
        if ((marker.segment ?? "history") === "forecast" && !segmentFilters.forecast) return false;
        if ((marker.scope ?? "prediction") === "prediction" && !scopeFilters.prediction) return false;
        if ((marker.scope ?? "prediction") === "backtest" && !scopeFilters.backtest) return false;
        return true;
      });

      console.log(
        `✅ Filtered markers: ${filtered.length} (BUY=${filtered.filter((m) => (m.displayType ?? m.type) === "buy").length}, SELL=${filtered.filter((m) => (m.displayType ?? m.type) === "sell").length})`
      );

      return filtered;
    }, [tradeMarkers, showBuyMarkers, showSellMarkers, segmentFilters, scopeFilters]);

    useEffect(() => {
      if (tradeMarkers?.length) {
        const sellCount = tradeMarkers.filter((m) => (m.displayType ?? m.type) === "sell").length;
        if (sellCount === 0) {
          console.error("🚨 ZERO SELL MARKERS! Check backend threshold settings.");
          console.log("Backend should have: buy_threshold < sell_threshold");
          console.log("Recommended: buy_threshold=0.25, sell_threshold=0.40");
        }
      }
    }, [tradeMarkers]);

    const getMarkerStyle = (marker: TradeMarker): SeriesMarker<Time> => {
      const markerType = (marker as ChartMarker).displayType ?? marker.type;
      const isBuy = markerType === "buy";
      const isSell = markerType === "sell";
      const isHold = markerType === "hold";
      const isForecast = marker.segment === "forecast";

      const baseStyle = {
        time: marker.date as Time,
        position: isBuy ? "aboveBar" : isSell ? "belowBar" : "inBar",
        shape: (isBuy ? "arrowUp" : isSell ? "arrowDown" : "triangle") as SeriesMarkerShape,
        text: "",
      } as const;

      if (isForecast) {
        return {
          ...baseStyle,
          color: isBuy ? "#10b981" : isSell ? "#ef4444" : "#94a3b8",
          size: isHold ? 0.6 : 0.9,
        };
      }

      return {
        ...baseStyle,
        color: isBuy ? "#059669" : isSell ? "#dc2626" : "#64748b",
        size: isHold ? 0.6 : 0.8,
      };
    };

    useEffect(() => {
      if (!containerRef.current || !historicalSeries.length) return;
      const chart = createChart(containerRef.current, {
        layout: {
          background: { color: "#050505" },
          textColor: "#9CA3AF",
        },
        grid: {
          vertLines: { color: "rgba(255,255,255,0.04)" },
          horzLines: { color: "rgba(255,255,255,0.04)" },
        },
        crosshair: { mode: CrosshairMode.Normal },
        localization: { priceFormatter: (price: number) => `$${price.toFixed(2)}` },
        rightPriceScale: { borderVisible: false },
        timeScale: { borderVisible: false, timeVisible: true },
      });

      chartRef.current = chart;

      const handleCrosshairMove: Parameters<IChartApi["subscribeCrosshairMove"]>[0] = (param) => {
        if (!param.time) {
          setHoverInfo(null);
          return;
        }

        const timeLabel = formatTimeLabel(param.time);
        let priceValue: number | undefined;
        const seriesPrices = (param as { seriesPrices?: Map<unknown, unknown> }).seriesPrices;
        if (priceSeriesRef.current && seriesPrices?.get) {
          const rawPrice = seriesPrices.get(priceSeriesRef.current as unknown);
          if (typeof rawPrice === "number") {
            priceValue = rawPrice;
          } else if (rawPrice && typeof rawPrice === "object" && "close" in rawPrice) {
            priceValue = Number((rawPrice as { close: number }).close);
          }
        }

        const markerAtTime = filteredMarkers.find((marker) => marker.date === timeLabel);
        setHoverInfo(
          markerAtTime || priceValue
            ? {
                date: timeLabel,
                price: priceValue,
                signal: markerAtTime ? ((markerAtTime as ChartMarker).displayType ?? markerAtTime.type) : undefined,
              }
            : null
        );
      };
      chart.subscribeCrosshairMove(handleCrosshairMove);

      const isLineMode = renderMode === "line";
      const priceSeries = isLineMode
        ? chart.addLineSeries({ color: "#f8fafc", lineWidth: 2 })
        : chart.addCandlestickSeries({
            upColor: "#22c55e",
            borderUpColor: "#22c55e",
            wickUpColor: "#22c55e",
            downColor: "#ef4444",
            borderDownColor: "#ef4444",
            wickDownColor: "#ef4444",
          });

      if (isLineMode) {
        priceSeries.setData(historicalSeries.map((point) => ({ time: point.date as Time, value: point.close })));
      } else {
        priceSeries.setData(
          historicalSeries.map((point) => ({
            time: point.date as Time,
            open: point.open,
            high: point.high,
            low: point.low,
            close: point.close,
          }))
        );
      }
      priceSeriesRef.current = priceSeries;

      const predictedHistory = predictedSeries?.filter((point) => point.segment !== "forecast") ?? [];
      if (predictedHistory.length && mode !== "prediction") {
        const predictedLine = chart.addLineSeries({
          color: "#22d3ee",
          lineWidth: 2,
          lineStyle: LineStyle.Solid,
        });
        predictedLine.setData(predictedHistory.map((point) => ({ time: point.date as Time, value: point.price })));
      }

      if (predictedSeries && mode !== "history") {
        const forecastPoints = predictedSeries.filter((point) => point.segment === "forecast");
        if (forecastPoints.length) {
          const lastHistorical = historicalSeries[historicalSeries.length - 1];
          const pathPoints = [
            lastHistorical && {
              time: lastHistorical.date as Time,
              value: lastHistorical.close,
            },
            ...forecastPoints.map((point) => ({ time: point.date as Time, value: point.price })),
          ].filter(Boolean) as Array<{ time: Time; value: number }>;
          const forecastSeries = chart.addLineSeries({
            color: "#3b82f6",
            lineWidth: 2,
            lineStyle: LineStyle.Dashed,
          });
          forecastSeries.setData(pathPoints);
        }
      }

      overlays?.forEach((overlay) => {
        if (overlay.type === "moving-average" || overlay.type === "predicted-path") {
          const series = chart.addLineSeries({
            color: overlay.type === "predicted-path" ? "#facc15" : "#f97316",
            lineWidth: 2,
            lineStyle: overlay.type === "predicted-path" ? LineStyle.Dotted : LineStyle.Solid,
          });
          series.setData(overlay.points.map((point) => ({ time: point.date as Time, value: point.value })));
        }
        if (overlay.type === "bollinger" && overlay.upper && overlay.lower) {
          const upper = chart.addLineSeries({ color: "rgba(148,163,184,0.6)", lineWidth: 1 });
          const lower = chart.addLineSeries({ color: "rgba(148,163,184,0.6)", lineWidth: 1 });
          upper.setData(overlay.points.map((point, idx) => ({ time: point.date as Time, value: overlay.upper![idx] })));
          lower.setData(overlay.points.map((point, idx) => ({ time: point.date as Time, value: overlay.lower![idx] })));
        }
      });

      if (!isLineMode && historicalSeries.some((point) => point.volume)) {
        const volumeSeries = chart.addHistogramSeries({
          priceFormat: { type: "volume" },
          priceScaleId: "",
          color: "rgba(59,130,246,0.4)",
        });
        volumeSeries.priceScale().applyOptions({ scaleMargins: { top: 0.8, bottom: 0 } });
        volumeSeries.setData(
          historicalSeries.map((point) => ({
            time: point.date as Time,
            value: point.volume ?? 0,
            color: point.close >= point.open ? "rgba(34,197,94,0.4)" : "rgba(248,113,113,0.4)",
          }))
        );
      }

      if (forecastChanges?.length) {
        const changeSeries = chart.addHistogramSeries({
          priceScaleId: "predicted-change",
          priceFormat: { type: "percent" },
          base: 0,
        });
        changeSeries.priceScale().applyOptions({ scaleMargins: { top: 0.85, bottom: 0 } });
        changeSeries.setData(
          forecastChanges.map((point) => ({
            time: point.date as Time,
            value: point.value * 100,
            color: point.value >= 0 ? "rgba(34,197,94,0.45)" : "rgba(248,113,113,0.45)",
          }))
        );
      }

      chart.timeScale().fitContent();

      const handleRangeChange = (range: { from: Time; to: Time } | null) => {
        onRangeChange?.(range ?? null);
      };
      chart.timeScale().subscribeVisibleTimeRangeChange(handleRangeChange);

      const resizeObserver = new ResizeObserver((entries) => {
        for (const entry of entries) {
          const { width, height } = entry.contentRect;
          chart.applyOptions({ width, height });
        }
      });
      resizeObserver.observe(containerRef.current);

      return () => {
        chart.timeScale().unsubscribeVisibleTimeRangeChange(handleRangeChange);
        chart.unsubscribeCrosshairMove(handleCrosshairMove);
        resizeObserver.disconnect();
        chart.remove();
        chartRef.current = null;
        priceSeriesRef.current = null;
        setHoverInfo(null);
      };
    }, [historicalSeries, overlays, predictedSeries, forecastChanges, renderMode, onRangeChange, mode, filteredMarkers]);

    useEffect(() => {
      if (!chartRef.current || !priceSeriesRef.current) return;
      const styledMarkers = filteredMarkers.map(getMarkerStyle);
      priceSeriesRef.current.setMarkers(styledMarkers);
    }, [filteredMarkers]);

    return (
      <div className="relative h-[420px] w-full" aria-label={ariaLabel ?? "Trading chart"} role="region">
        <div ref={containerRef} className="absolute inset-0" />
        {hoverInfo && (
          <div className="pointer-events-none absolute left-3 top-3 rounded-md bg-black/70 px-3 py-2 text-xs text-gray-100 shadow-lg">
            <p className="font-semibold">{hoverInfo.date}</p>
            {typeof hoverInfo.price === "number" && <p>Price: ${hoverInfo.price.toFixed(2)}</p>}
            {hoverInfo.signal && <p>Signal: {hoverInfo.signal.toUpperCase()}</p>}
          </div>
        )}
        <div className="sr-only" role="status" aria-live="polite" aria-atomic="true">
          {hoverInfo ? `Hovering ${hoverInfo.date} price ${hoverInfo.price ?? "n/a"} signal ${hoverInfo.signal ?? "none"}` : "No point highlighted"}
        </div>
      </div>
    );
  }
);

InteractiveCandlestick.displayName = "InteractiveCandlestick";

export default InteractiveCandlestick;
