"use client";

import { useEffect, useRef } from "react";
import { JobEvent } from "@/types/ai";

interface StreamingLogProps {
  title: string;
  events: JobEvent[];
  status?: JobEvent["status"];
}

export function StreamingLog({ title, events, status }: StreamingLogProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const node = containerRef.current;
    if (node) {
      node.scrollTop = node.scrollHeight;
    }
  }, [events]);

  return (
    <div className="flex h-full flex-col rounded-xl border border-gray-800 bg-gray-900/60">
      <div className="flex items-center justify-between border-b border-gray-800 px-4 py-2 text-xs text-gray-400">
        <span>{title}</span>
        <span className="text-[11px] uppercase tracking-wide text-gray-500">{status ?? "running"}</span>
      </div>
      <div ref={containerRef} className="flex-1 space-y-1 overflow-y-auto px-4 py-3 text-[11px] text-gray-300">
        {events.length === 0 && <p className="text-gray-600">Awaiting events…</p>}
        {events.map((event, index) => (
          <div key={`${event.timestamp}-${index}`} className="rounded bg-gray-950/60 px-2 py-1">
            <span className="text-[10px] text-gray-500">{new Date(event.timestamp).toLocaleTimeString()}</span>
            <p>{event.message}</p>
            {typeof event.progress === "number" && (
              <progress className="progress-element mt-1" value={event.progress} max={100} aria-label="Job progress" />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
