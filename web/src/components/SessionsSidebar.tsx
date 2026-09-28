import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link, NavLink, useMatch } from "react-router";

import { api } from "../api";
import ThemeToggle from "./ThemeToggle";
import UsageMeter from "./UsageMeter";
import { IDLE_POLL_MS, invalidate } from "../hooks";
import { Badge, cx, Spinner } from "../ui";

interface SessionJob {
  job_id: number;
  project_id: number;
  project: string;
  kind: string;
  label: string;
  status: string;
  progress?: { current: number; total: number; percent: number | null; minutes_left?: number };
  queue_position?: number;
}

interface SessionItem {
  project_id: number;
  name: string;
  goal: string;
  stage: string;
  autopilot: boolean;
  completed: boolean;
  state: "running" | "queued" | "thinking" | "waiting" | "done" | "paused" | "idle";
  awaiting?: string | null;
  models: number; // exported models still on disk
  job?: SessionJob;
}

interface Sessions {
  capacity: {
    chip: string;
    memory_gb: number;
    ml_budget_gb: number;
    gpu_slots: number;
    explanation: string;
    gpu_running: SessionJob | null;
    gpu_queue: SessionJob[];
  };
  sessions: SessionItem[];
}

export function useSessions() {
  return useQuery({
    queryKey: ["sessions"],
    queryFn: () => api.get<Sessions>("/api/sessions"),
    // Fast only while a job runs (progress and minutes left); state changes arrive by push.
    refetchInterval: (q) => (q.state.data?.sessions.some((s) => s.job) ? 4000 : IDLE_POLL_MS),
  });
}

const STATE: Record<SessionItem["state"], { tone: "info" | "accent" | "good" | "warn" | "neutral"; label: string }> = {
  running: { tone: "info", label: "running" },
  queued: { tone: "warn", label: "queued" },
  thinking: { tone: "accent", label: "thinking" },
  waiting: { tone: "warn", label: "needs you" },
  done: { tone: "good", label: "done" },
  paused: { tone: "neutral", label: "paused" },
  idle: { tone: "neutral", label: "idle" },
};

export function sessionDetail(s: SessionItem): string {
  const j = s.job;
  if (j?.progress?.percent != null) {
    return `${j.label} ${j.progress.percent}%${j.progress.minutes_left ? ` · ~${j.progress.minutes_left} min` : ""}`;
  }
  if (j?.queue_position) return `waiting for the GPU (#${j.queue_position})`;
  if (j) return j.label;
  if (s.state === "waiting") return `waiting for your go-ahead: ${s.awaiting ?? "next run"}`;
  if (s.state === "done") return "model exported";
  const model = s.models ? "has a model · " : "";
  if (s.state === "paused") return `${model}paused at ${s.stage}`;
  return `${model}${s.stage}`;
}

/** Every project and what this Mac is doing. Shown on Home and in the Studio, so leaving a
 * project never hides its running work. */
export default function SessionsSidebar() {
  const { data } = useSessions();
  // Both hooks run on every render (hooks must not be called conditionally).
  const nested = useMatch("/p/:projectId/*");
  const studio = useMatch("/p/:projectId");
  const current = Number((nested ?? studio)?.params.projectId);
  const [collapsed, setCollapsed] = useState(() => {
    try {
      return localStorage.getItem("slm.sidebar") === "collapsed";
    } catch {
      return false;
    }
  });
  const toggle = () => {
    setCollapsed(!collapsed);
    try {
      localStorage.setItem("slm.sidebar", collapsed ? "open" : "collapsed");
    } catch {
      /* private mode: just don't remember */
    }
  };

  // Below lg only the rail fits; from lg the full sidebar shows unless the user collapsed it.
  return (
    <>
      <aside className={cx("flex w-12 shrink-0 flex-col items-center gap-2 border-r border-line bg-panel py-3", !collapsed && "lg:hidden")}>
        <Link to="/" className="grid size-7 place-items-center rounded-lg bg-accent text-sm font-bold text-white" title="Home">
          ▲
        </Link>
        <button onClick={toggle} className="text-faint hover:text-fg" title="Show projects">
          »
        </button>
        <ThemeToggle />
        {data?.sessions.map((s) => (
          <NavLink
            key={s.project_id}
            to={`/p/${s.project_id}`}
            title={`${s.name}: ${sessionDetail(s)}`}
            className={cx("grid size-7 place-items-center rounded-md text-[11px] font-semibold", s.project_id === current ? "bg-accent-soft text-accent" : "text-muted hover:bg-panel-2")}
          >
            {s.state === "running" ? <span className="size-2 animate-pulse rounded-full bg-info" /> : s.name.slice(0, 1).toUpperCase()}
          </NavLink>
        ))}
      </aside>
      <aside className={cx("hidden w-64 shrink-0 flex-col border-r border-line bg-panel", !collapsed && "lg:flex")}>
      <div className="flex items-center gap-2 px-4 pt-4 pb-3">
        <Link to="/" className="flex items-center gap-2">
          <span className="grid size-7 place-items-center rounded-lg bg-accent text-sm font-bold text-white">▲</span>
          <span className="font-semibold tracking-tight">SLM Forge</span>
        </Link>
        <ThemeToggle className="ml-auto" />
        <button onClick={toggle} className="text-faint hover:text-fg" title="Collapse">
          «
        </button>
      </div>

      <div className="px-3">
        <Link to="/" className="flex items-center justify-center gap-1.5 rounded-lg border border-accent/50 bg-accent-soft px-3 py-1.5 text-[13px] font-medium text-accent hover:brightness-110">
          + New model
        </Link>
      </div>

      <div className="mt-4 min-h-0 flex-1 overflow-y-auto px-2">
        <div className="px-2 pb-1.5 text-[11px] font-medium uppercase tracking-wide text-faint">Projects</div>
        {!data && <Spinner className="m-3" />}
        {data?.sessions.length === 0 && <p className="px-2 text-xs text-faint">No projects yet.</p>}
        <ul className="space-y-0.5">
          {data?.sessions.map((s) => (
            <SessionRow key={s.project_id} s={s} active={s.project_id === current} />
          ))}
        </ul>
      </div>

      {data && <MachinePanel c={data.capacity} />}
      </aside>
    </>
  );
}

function SessionRow({ s, active }: { s: SessionItem; active: boolean }) {
  const qc = useQueryClient();
  const [menu, setMenu] = useState(false);
  const act = useMutation({
    mutationFn: (action: "pause" | "stop" | "resume") =>
      action === "resume"
        ? api.post(`/api/sessions/${s.project_id}/resume`)
        : api.post(`/api/sessions/${s.project_id}/stop`, { cancel_jobs: action === "stop" }),
    onSuccess: () => {
      setMenu(false);
      invalidate(qc, "sessions", "studio", "jobs");
    },
  });
  const st = STATE[s.state];
  const pct = s.job?.progress?.percent;
  return (
    <li className="group relative">
      <NavLink
        to={`/p/${s.project_id}`}
        className={cx("block rounded-lg px-2 py-2 pr-7", active ? "bg-accent-soft" : "hover:bg-panel-2")}
      >
        <div className="flex items-center gap-1.5">
          <span className={cx("truncate text-[13px]", active ? "font-medium text-fg" : "text-fg")}>{s.name}</span>
          <Badge tone={st.tone} className="ml-auto shrink-0">
            {(s.state === "running" || s.state === "thinking") && <span className="size-1.5 animate-pulse rounded-full bg-current" />}
            {st.label}
          </Badge>
        </div>
        <div className="mt-0.5 truncate text-[11px] text-faint">{sessionDetail(s)}</div>
        {pct != null && (
          <div className="mt-1 h-1 overflow-hidden rounded-full bg-panel-2">
            <div className="h-full bg-info transition-all" style={{ width: `${pct}%` }} />
          </div>
        )}
      </NavLink>
      <button
        onClick={() => setMenu(!menu)}
        className="absolute top-2 right-1 rounded px-1 text-faint opacity-0 group-hover:opacity-100 hover:text-fg"
        title="Actions"
      >
        ⋯
      </button>
      {menu && (
        <div className="absolute top-8 right-1 z-20 w-44 overflow-hidden rounded-lg border border-line bg-panel shadow-lg shadow-black/30">
          {s.autopilot && !s.completed ? (
            <MenuItem onClick={() => act.mutate("pause")} label="Pause autopilot" hint="a running job finishes" />
          ) : (
            <MenuItem onClick={() => act.mutate("resume")} label="Resume" hint="the Tuner carries on" />
          )}
          {s.job && <MenuItem onClick={() => act.mutate("stop")} label="Stop now" hint="cancels the running job" danger />}
          <Link to={`/p/${s.project_id}/overview`} className="block px-3 py-2 text-xs hover:bg-panel-2">
            Advanced controls
            <span className="block text-[11px] text-faint">the full manual screens</span>
          </Link>
        </div>
      )}
    </li>
  );
}

function MenuItem({ onClick, label, hint, danger }: { onClick: () => void; label: string; hint: string; danger?: boolean }) {
  return (
    <button onClick={onClick} className={cx("block w-full px-3 py-2 text-left text-xs hover:bg-panel-2", danger && "text-bad")}>
      {label}
      <span className="block text-[11px] text-faint">{hint}</span>
    </button>
  );
}

function MachinePanel({ c }: { c: Sessions["capacity"] }) {
  const [why, setWhy] = useState(false);
  const run = c.gpu_running;
  return (
    <div className="border-t border-line p-3 text-xs">
      <div className="flex items-center gap-1.5 font-medium">
        This Mac
        <span className="font-normal text-faint">
          · {c.memory_gb.toFixed(0)} GB · {c.gpu_slots} training at a time
        </span>
      </div>
      <UsageMeter className="mt-2" side="up" align="left" />
      <Link to="/storage" className="mt-2 block text-[12px] text-muted hover:text-fg">
        Storage & cleanup →
      </Link>
      {run ? (
        <div className="mt-1.5">
          <span className="text-info">● </span>
          <span className="text-fg">{run.project}</span>
          <span className="text-muted">
            : {run.label}
            {run.progress?.percent != null ? ` ${run.progress.percent}%` : ""}
            {run.progress?.minutes_left ? ` · ~${run.progress.minutes_left} min left` : ""}
          </span>
        </div>
      ) : (
        <div className="mt-1.5 text-muted">GPU free</div>
      )}
      {c.gpu_queue.length > 0 && (
        <div className="mt-1 text-muted">
          Waiting: {c.gpu_queue.map((j) => j.project).join(", ")}
        </div>
      )}
      <button onClick={() => setWhy(!why)} className="mt-1.5 text-[11px] text-faint hover:text-fg">
        {why ? "hide" : "why one at a time?"}
      </button>
      {why && <p className="mt-1 text-[11px] leading-relaxed text-faint">{c.explanation}</p>}
    </div>
  );
}
