import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router";

import { api, type Job, type MetricPoint, type Overview, type SystemStatus } from "./api";
import { type ToastInput, useToast } from "./ui";

export function useProjectId(): number {
  return Number(useParams().projectId);
}

export function useSystem() {
  return useQuery({ queryKey: ["system"], queryFn: () => api.get<SystemStatus>("/api/system"), refetchInterval: 5000 });
}

export function useOverview(projectId: number) {
  return useQuery({
    queryKey: ["overview", projectId],
    queryFn: () => api.get<Overview>(`/api/projects/${projectId}`),
    enabled: Number.isFinite(projectId),
  });
}

/** Subscribe to a server-sent event stream; reconnects automatically (EventSource default). */
function useEventSource(url: string | null, onEvent: (type: string, data: Record<string, unknown>) => void, types: string[]) {
  const handler = useRef(onEvent);
  handler.current = onEvent;
  useEffect(() => {
    if (!url) return;
    const es = new EventSource(url);
    const listeners = types.map((t) => {
      const fn = (e: MessageEvent) => handler.current(t, JSON.parse(e.data));
      es.addEventListener(t, fn);
      return [t, fn] as const;
    });
    return () => {
      listeners.forEach(([t, fn]) => es.removeEventListener(t, fn));
      es.close();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url, types.join()]);
}

/** Keep job lists and dependent views fresh as any job changes status. */
export function useJobsFeed() {
  const qc = useQueryClient();
  const toast = useToast();
  useEventSource(
    "/api/stream/jobs",
    (_t, ev) => {
      qc.invalidateQueries({ queryKey: ["jobs"] });
      qc.invalidateQueries({ queryKey: ["system"] });
      const status = ev.status as string;
      if (status === "succeeded" || status === "failed") {
        toast(jobToast(ev as unknown as Job));
        // A finished job can create datasets, versions, checkpoints, examples or proposals.
        for (const k of ["overview", "datasets", "exports", "examples", "proposals", "models-local", "studio"]) {
          qc.invalidateQueries({ queryKey: [k] });
        }
      }
    },
    ["status"],
  );
}

/** Live agent activity and proposal changes for a project. */
export function useProjectFeed(projectId: number) {
  const qc = useQueryClient();
  useEventSource(
    Number.isFinite(projectId) ? `/api/projects/${projectId}/stream` : null,
    (type) => {
      if (type === "agent_event") qc.invalidateQueries({ queryKey: ["agent-events", projectId] });
      if (type === "proposal") {
        qc.invalidateQueries({ queryKey: ["proposals", projectId] });
        qc.invalidateQueries({ queryKey: ["overview", projectId] });
      }
    },
    ["agent_event", "proposal"],
  );
}

export interface LiveJob {
  job: Job | undefined;
  metrics: MetricPoint[];
  log: string[];
  progress: { current: number; total: number } | null;
}

/** A job's detail plus live logs/metrics streamed as they arrive. */
export function useLiveJob(jobId: number | null): LiveJob {
  const qc = useQueryClient();
  const detail = useQuery({
    queryKey: ["job", jobId],
    queryFn: () => api.get<{ job: Job; metrics: MetricPoint[]; log: string }>(`/api/jobs/${jobId}?log_lines=400`),
    enabled: jobId != null,
  });
  const [live, setLive] = useState<{ metrics: MetricPoint[]; log: string[]; progress: LiveJob["progress"] }>({
    metrics: [],
    log: [],
    progress: null,
  });

  // Reset when switching jobs, then seed from the snapshot.
  useEffect(() => {
    setLive({ metrics: [], log: [], progress: null });
  }, [jobId]);

  useEventSource(
    jobId != null ? `/api/jobs/${jobId}/stream` : null,
    (type, ev) => {
      if (type === "log") setLive((s) => ({ ...s, log: [...s.log, ev.line as string].slice(-1000) }));
      if (type === "metric") setLive((s) => ({ ...s, metrics: [...s.metrics, ev as unknown as MetricPoint] }));
      if (type === "progress") setLive((s) => ({ ...s, progress: { current: ev.current as number, total: ev.total as number } }));
      if (type === "status") {
        // The refetched snapshot carries the full log tail; drop the live copy to avoid duplicates.
        setLive((s) => ({ ...s, log: [] }));
        qc.invalidateQueries({ queryKey: ["job", jobId] });
      }
    },
    ["log", "metric", "progress", "status"],
  );

  const snapshot = detail.data;
  const seenIters = new Set((snapshot?.metrics ?? []).map((m) => `${m.split}:${m.iteration}`));
  const metrics = [...(snapshot?.metrics ?? []), ...live.metrics.filter((m) => !seenIters.has(`${m.split}:${m.iteration}`))];
  const baseLog = snapshot?.log ? snapshot.log.split("\n") : [];
  return {
    job: snapshot?.job,
    metrics,
    log: live.log.length ? [...baseLog, ...live.log].slice(-1000) : baseLog,
    progress: live.progress ?? snapshot?.job.result.progress ?? null,
  };
}

/** Poll a job until it finishes; resolves with the final job. */
export async function waitForJob(jobId: number, onTick?: (j: Job) => void): Promise<Job> {
  for (;;) {
    const { job } = await api.get<{ job: Job }>(`/api/jobs/${jobId}?log_lines=0`);
    onTick?.(job);
    if (["succeeded", "failed", "cancelled"].includes(job.status)) return job;
    await new Promise((r) => setTimeout(r, 1000));
  }
}

const JOB_LABEL: Record<string, string> = {
  download: "Model downloaded",
  import_dataset: "Dataset imported",
  prepare_dataset: "Data prepared",
  sft: "Fine-tuning finished",
  dpo: "Preference round finished",
  fuse: "Adapter fused",
  export: "Model exported",
  agent_scout: "DataScout finished",
  agent_prep: "DataPrep suggested a mapping",
  agent_observer: "Observer finished",
  synthesize: "Synthetic examples ready",
};

const JOB_NAME: Record<string, string> = {
  download: "Download",
  import_dataset: "Import",
  prepare_dataset: "Data preparation",
  sft: "Fine-tuning",
  dpo: "Preference round",
  fuse: "Fuse",
  export: "Export",
  agent_scout: "DataScout",
  agent_prep: "DataPrep",
  agent_observer: "Observer",
  synthesize: "Synthesis",
};

const JOB_PAGE: Record<string, string> = {
  download: "data",
  import_dataset: "data",
  prepare_dataset: "data",
  agent_prep: "data",
  agent_scout: "data",
  sft: "train",
  dpo: "train",
  export: "export",
  agent_observer: "agents",
  synthesize: "feedback",
};

/** Turn a finished job into a one-glance summary with a link to its result. */
export function jobToast(job: Job): ToastInput {
  const base = job.project_id != null ? `/p/${job.project_id}` : "";
  const r = job.result as Record<string, unknown>;
  const page = JOB_PAGE[job.kind];
  const to = !base || !page ? undefined : ["sft", "dpo"].includes(job.kind) ? `${base}/train/${job.id}` : `${base}/${page}`;
  if (job.status === "failed") {
    return {
      tone: "bad",
      title: `${JOB_NAME[job.kind] ?? job.kind} failed`,
      body: job.error.slice(0, 180),
      action: to ? { label: "Open", to } : undefined,
    };
  }
  let body = "";
  let label = "Open";
  switch (job.kind) {
    case "import_dataset":
      body = `${Number(r.rows).toLocaleString()} rows. Next: map the columns.`;
      label = "Map it";
      break;
    case "prepare_dataset":
      body =
        r.n_train != null
          ? `${Number(r.n_train).toLocaleString()} training examples (kept ${Number(r.kept).toLocaleString()} of ${Number(r.input_rows).toLocaleString()}).`
          : "Ready to train on.";
      label = "View";
      break;
    case "sft":
    case "dpo": {
      const m = (r.metrics ?? {}) as Record<string, number>;
      const warn = (r.warnings as { code: string }[] | undefined)?.length;
      body = `val loss ${m.val_loss ?? "?"} · peak ${m.peak_mem_gb ?? "?"} GB${warn ? ` · ${warn} warning${warn > 1 ? "s" : ""}` : ""}`;
      label = "See the run";
      break;
    }
    case "export":
      body = `${r.size_gb} GB · runs on ${r.min_ram_gb} GB+ Macs`;
      break;
    case "synthesize":
      body = `${r.saved} examples waiting for your review.`;
      label = "Review";
      break;
    case "agent_prep":
      label = "See it";
      break;
  }
  return { tone: "good", title: JOB_LABEL[job.kind] ?? `${job.kind} finished`, body, action: to ? { label, to } : undefined };
}
