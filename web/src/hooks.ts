import { useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { useParams } from "react-router";

import { api, isActive, type Job, type MetricPoint, type Overview, postStream, type Snapshot, type SystemStatus } from "./api";
import { type ToastInput, useToast } from "./ui";

export function useProjectId(): number {
  return Number(useParams().projectId);
}

// Pushed updates (the jobs feed) cover every change of state; polling only refreshes live numbers
// while something runs, plus a slow safety net when idle.
export const IDLE_POLL_MS = 60_000;

export function useSystem() {
  return useQuery({
    queryKey: ["system"],
    queryFn: () => api.get<SystemStatus>("/api/system"),
    refetchInterval: (q) => {
      const running = q.state.data?.worker.running;
      return running && Object.values(running).some((j) => j != null) ? 5000 : IDLE_POLL_MS;
    },
  });
}

/** Everything the Studio canvas shows. Canvas and job events push every change; polling only catches
 * up while work is in flight. */
export function useStudio(projectId: number) {
  return useQuery({
    queryKey: ["studio", projectId],
    queryFn: () => api.get<Snapshot>(`/api/projects/${projectId}/studio`),
    refetchInterval: (q) => {
      const d = q.state.data;
      return d && (d.tuner_busy || d.jobs.some(isActive)) ? 15000 : IDLE_POLL_MS;
    },
  });
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
    (type, ev) => {
      if (type === "tuner") {
        // A Tuner started or stopped thinking somewhere: only the sessions list shows that.
        qc.invalidateQueries({ queryKey: ["sessions"] });
        return;
      }
      qc.invalidateQueries({ queryKey: ["jobs"] });
      qc.invalidateQueries({ queryKey: ["system"] });
      qc.invalidateQueries({ queryKey: ["sessions"] });
      const status = ev.status as string;
      if (status === "succeeded" || status === "failed") {
        toast(jobToast(ev as unknown as Job));
        // A finished job can create datasets, versions, checkpoints, examples or proposals.
        for (const k of ["overview", "datasets", "exports", "examples", "proposals", "studio", "sessions"]) {
          qc.invalidateQueries({ queryKey: [k] });
        }
      }
    },
    ["status", "tuner"],
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

interface LiveJob {
  job: Job | undefined;
  metrics: MetricPoint[];
  log: string[];
  progress: { current: number; total: number } | null;
}

/** Where a training run is: steps done, percent, minutes left, the latest values and any warnings. */
export function runProgress({ job, metrics, progress }: LiveJob) {
  const train = metrics.filter((m) => m.split === "train");
  const last = train.at(-1)?.values;
  const lastVal = metrics.filter((m) => m.split === "val").at(-1)?.values;
  const total = progress?.total || (job?.result.total_iters as number) || 0;
  const current = progress?.current ?? train.at(-1)?.iteration ?? 0;
  const pct = job?.status === "succeeded" ? 100 : total ? Math.min(100, (current / total) * 100) : 0;
  const minutesLeft = last?.it_per_sec && total > current ? Math.ceil((total - current) / last.it_per_sec / 60) : null;
  const warnings = (job?.result.warnings as { code: string; message: string }[] | undefined) ?? [];
  return { current, total, pct, minutesLeft, last, lastVal, warnings };
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
async function waitForJob(jobId: number, onTick?: (j: Job) => void): Promise<Job> {
  for (;;) {
    const { job } = await api.get<{ job: Job }>(`/api/jobs/${jobId}?log_lines=0`);
    onTick?.(job);
    if (["succeeded", "failed", "cancelled"].includes(job.status)) return job;
    await new Promise((r) => setTimeout(r, 1000));
  }
}

/** Follow a job to the end, reporting its status as it goes; throws unless it succeeded. */
export async function followJob(jobId: number, what: string, onStatus: (status: string) => void): Promise<Job> {
  const job = await waitForJob(jobId, (j) => onStatus(j.status));
  if (job.status !== "succeeded") throw new Error(job.error || `${what} ${job.status}`);
  return job;
}

const JOB_LABEL: Record<string, string> = {
  download: "Model downloaded",
  import_dataset: "Dataset imported",
  prepare_dataset: "Data prepared",
  sft: "Fine-tuning finished",
  dpo: "Preference round finished",
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
function jobToast(job: Job): ToastInput {
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

// ── chatting with a local model ─────────────────────────────────────────────

export interface ChatMsg {
  role: "user" | "assistant";
  content: string;
  stats?: { tokens_per_sec: number; generation_tokens: number; seconds: number; finish_reason: string | null };
}

/** A streamed conversation with one of the project's models (POST /generate). `bottom` goes on an
 * element after the last message, to keep it in view as tokens arrive. */
export function useChatStream(projectId: number) {
  const [messages, setMessages] = useState<ChatMsg[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);
  // Block body: scrollIntoView() returns a Promise in current Chrome, and an effect must not return one.
  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [messages]);

  /** `body` builds the request from the conversation so far (the new message included). */
  async function send(text: string, body: (history: { role: string; content: string }[]) => object) {
    const t = text.trim();
    if (!t || busy) return;
    const history: ChatMsg[] = [...messages, { role: "user", content: t }];
    setMessages([...history, { role: "assistant", content: "" }]);
    setBusy(true);
    setError(null);
    abort.current = new AbortController();
    const patchLast = (fn: (m: ChatMsg) => ChatMsg) =>
      setMessages((m) => [...m.slice(0, -1), fn(m[m.length - 1])]);
    try {
      await postStream(
        `/api/projects/${projectId}/generate`,
        body(history.map(({ role, content }) => ({ role, content }))),
        (ev) => {
          if (ev.type === "token") patchLast((m) => ({ ...m, content: m.content + String(ev.text) }));
          else if (ev.type === "done") patchLast((m) => ({ ...m, stats: ev as unknown as ChatMsg["stats"] }));
          else if (ev.type === "error") setError(new Error(String(ev.message)));
        },
        abort.current.signal,
      );
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError(e);
    } finally {
      setBusy(false);
    }
  }

  const reset = () => {
    abort.current?.abort();
    setMessages([]);
    setError(null);
  };
  return { messages, busy, error, bottom, send, stop: () => abort.current?.abort(), reset };
}
