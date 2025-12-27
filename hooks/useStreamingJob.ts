"use client";

import { useEffect, useRef, useState } from "react";
import { apiBase } from "@/lib/api-client";
import { JobEvent } from "@/types/ai";

interface StreamingJobHook {
  events: JobEvent[];
  progress: number;
  status: JobEvent["status"];
}

export function useStreamingJob(jobId?: string, enabled = Boolean(jobId)): StreamingJobHook {
  const [events, setEvents] = useState<JobEvent[]>([]);
  const [progress, setProgress] = useState(0);
  const [status, setStatus] = useState<JobEvent["status"]>("running");
  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => {
    if (!jobId || !enabled) return () => undefined;

    const source = new EventSource(`${apiBase}/api/jobs/${jobId}/events`);
    eventSourceRef.current = source;

    source.onmessage = (event) => {
      try {
        const data: JobEvent = JSON.parse(event.data);
        setEvents((prev) => [...prev, data]);
        if (typeof data.progress === "number") {
          setProgress(data.progress);
        }
        if (data.status) {
          setStatus(data.status);
          if (data.status !== "running") {
            source.close();
          }
        }
      } catch (error) {
        console.warn("Malformed job event", error);
      }
    };

    source.onerror = () => {
      setStatus("failed");
      source.close();
    };

    return () => {
      source.close();
      eventSourceRef.current = null;
      setEvents([]);
      setProgress(0);
      setStatus("running");
    };
  }, [jobId, enabled]);

  return { events, progress, status };
}
