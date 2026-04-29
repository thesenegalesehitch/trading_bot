"use client";

import { useEffect, useRef } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi } from 'lightweight-charts';

interface TradingChartProps {
  data: any[];
  markers?: any[];
  fvgZones?: any[];
  symbol?: string;
  enableLive?: boolean;
}

export function TradingChart({ data, markers, fvgZones, symbol, enableLive = false }: TradingChartProps) {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const wsRef = useRef<WebSocket | null>(null);

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#d1d5db',
      },
      grid: {
        vertLines: { color: 'rgba(75, 85, 99, 0.1)' },
        horzLines: { color: 'rgba(75, 85, 99, 0.1)' },
      },
      width: chartContainerRef.current.clientWidth,
      height: 500,
      timeScale: {
        timeVisible: true,
        secondsVisible: false,
      },
    });

    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#10b981',
      downColor: '#ef4444',
      borderVisible: false,
      wickUpColor: '#10b981',
      wickDownColor: '#ef4444',
    });

    candlestickSeries.setData(data);
    
    // Add Markers (Buy/Sell)
    if (markers && markers.length > 0) {
      candlestickSeries.setMarkers(markers);
    }

    // Add Price Lines (FVG/OB Levels)
    if (fvgZones && fvgZones.length > 0) {
      fvgZones.forEach((zone: any) => {
        candlestickSeries.createPriceLine({
          price: zone.price,
          color: zone.color || '#3b82f6',
          lineWidth: 2,
          lineStyle: 2, // Dashed
          axisLabelVisible: true,
          title: zone.label,
        });
      });
    }

    // Handle Resize
    const handleResize = () => {
      if (chartContainerRef.current) {
        chart.applyOptions({ width: chartContainerRef.current.clientWidth });
      }
    };
    window.addEventListener('resize', handleResize);

    chartRef.current = chart;
    seriesRef.current = candlestickSeries;

    return () => {
      window.removeEventListener('resize', handleResize);
      chart.remove();
    };
  }, [data, markers, fvgZones]);

  // Live WebSocket Connection
  useEffect(() => {
    if (!enableLive || !symbol || !seriesRef.current) return;

    // We get the last candle time to update it instead of creating a new one unless necessary
    const connectWs = () => {
      const wsUrl = `ws://localhost:8000/api/v1/market/stream/${symbol}`;
      const ws = new WebSocket(wsUrl);

      ws.onmessage = (event) => {
        try {
          const streamData = JSON.parse(event.data);
          const currentData = seriesRef.current?.data();
          if (!currentData || currentData.length === 0) return;

          // Replace the close price of the last candle
          const lastCandle = currentData[currentData.length - 1] as any;
          const updatedCandle = {
            ...lastCandle,
            close: streamData.price,
            high: Math.max(lastCandle.high, streamData.price),
            low: Math.min(lastCandle.low, streamData.price),
          };

          seriesRef.current?.update(updatedCandle);
        } catch (e) {
          console.error("WS Parse error", e);
        }
      };

      wsRef.current = ws;
    };

    connectWs();

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [symbol, enableLive]);

  return <div ref={chartContainerRef} className="w-full h-[500px]" />;
}
