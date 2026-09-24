import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router";

import { api, type Checkpoint, type DatasetVersion, type Dataset, fmt, type Job, type MemoryEstimate, type Project } from "../api";
import { MetricChart } from "../components/Charts";
import JobLog from "../components/JobLog";
import Markdown from "../components/Markdown";
import { useLiveJob, useSystem } from "../hooks";
import { Badge, Button, cx, MemoryBar, Spinner, StatusBadge, TextArea } from "../ui";

// ── types ────────────────────────────────────────────────────────────────────

interface TunerMessage {
  id: number;
  role: "user" | "assistant" | "event" | "tool";
  content: string;
  meta: {
    name?: string;
    args?: Record<string, unknown>;
    status?: string;
    output?: string;
    error?: boolean;
    kickoff?: boolean;
    autopilot?: boolean;
    autopilot_paused?: boolean;
  } & Record<string, unknown>;
  created_at: string;
}

interface Comparison {
  id: string;
  prompt: string;
  a: string;
  b: string;
  status: "pending" | "judged";
  choice?: string;
  judge?: "ai";
  critique?: string;
  ideal?: string;
}

interface Sample {
  prompt: string;
  target: string;
  text: string;
  tokens_per_sec?: number;
}

interface Snapshot {
  project: Project;
  stage: Stage;
  note: string;
  tuner_busy: boolean;
  autopilot: boolean;
  completed: boolean;
  hardware: { chip: string; total_memory_gb: number; budget_gb: number };
  model: { repo_id: string; downloaded: boolean; params?: number; bits?: number; size_gb?: number; layers?: number; inference?: MemoryEstimate } | null;
  datasets: Dataset[];
  versions: DatasetVersion[];
  jobs: Job[];
  checkpoints: (Checkpoint & { warnings: { code: string; message: string }[] })[];
  samples: Sample[];
  comparisons: Comparison[];
  feedback: { judgements: number; pairs_ready: number };
  exports: { path: string; size_gb: number; min_ram_gb: number }[];
}

const STAGES = ["goal", "model", "data", "train", "evaluate", "refine", "export"] as const;
type Stage = (typeof STAGES)[number];
const STAGE_LABEL: Record<Stage, string> = {
  goal: "Goal",
  model: "Model",
  data: "Data",
  train: "Train",
  evaluate: "Evaluate",
  refine: "Refine",
  export: "Export",
};

const TOOL_LABEL: Record<string, string> = {
  update_project: "Saving the project brief",
  find_base_models: "Searching for base models",
  choose_base_model: "Choosing the base model",
  search_datasets: "Searching datasets",
  preview_dataset: "Previewing a dataset",
  import_dataset: "Importing data",
  inspect_dataset: "Inspecting the data",
  prepare_dataset: "Cleaning and preparing the data",
  plan_training: "Planning the training run",
  start_training: "Starting training",
  training_progress: "Checking on training",
  cancel_job: "Stopping a job",
  try_model: "Testing the model",
  ask_user_to_compare: "Preparing answers for you to compare",
  feedback_summary: "Reading your feedback",
  generate_synthetic_examples: "Writing training examples",
  review_synthetic_examples: "Reviewing examples",
  build_dataset_from_examples: "Building a dataset",
  export_model: "Exporting the model",
  ai_review_answers: "Having GPT-6 review the model's answers",
  finish_project: "Wrapping up",
};
// Bookkeeping calls show in the console, not the chat.
const QUIET_TOOLS = new Set(["get_status", "set_stage"]);

// ── page ─────────────────────────────────────────────────────────────────────

export default function Studio() {
  const projectId = Number(useParams().projectId);
  const qc = useQueryClient();
  const snapshot = useQuery({
    queryKey: ["studio", projectId],
    queryFn: () => api.get<Snapshot>(`/api/projects/${projectId}/studio`),
    refetchInterval: 15000,
  });
  const messages = useQuery({
    queryKey: ["tuner-messages", projectId],
    queryFn: () => api.get<TunerMessage[]>(`/api/projects/${projectId}/tuner/messages`),
  });
  const [streaming, setStreaming] = useState("");
  const [busy, setBusy] = useState(false);

  // Live stream from the Tuner: tokens, tool activity, new messages, canvas changes.
  useEffect(() => {
    const es = new EventSource(`/api/projects/${projectId}/tuner/stream`);
    const on = (type: string, fn: (d: Record<string, unknown>) => void) =>
      es.addEventListener(type, (e) => fn(JSON.parse((e as MessageEvent).data)));
    on("turn_start", () => setBusy(true));
    on("turn_end", () => {
      setBusy(false);
      setStreaming("");
    });
    on("delta", (d) => setStreaming((s) => s + String(d.text)));
    on("message", (d) => {
      const m = d.message as TunerMessage;
      if (m.role === "assistant") setStreaming("");
      qc.setQueryData<TunerMessage[]>(["tuner-messages", projectId], (old = []) => {
        const i = old.findIndex((x) => x.id === m.id);
        if (i === -1) return [...old, m];
        const copy = [...old];
        copy[i] = m;
        return copy;
      });
    });
    on("canvas", () => qc.invalidateQueries({ queryKey: ["studio", projectId] }));
    return () => es.close();
  }, [projectId, qc]);

  // A new project's conversation starts on its own.
  const started = useRef(false);
  useEffect(() => {
    if (messages.data && messages.data.length === 0 && !started.current) {
      started.current = true;
      api.post(`/api/projects/${projectId}/tuner/start`);
    }
  }, [messages.data, projectId]);

  useEffect(() => {
    if (snapshot.data) setBusy((b) => b || snapshot.data!.tuner_busy);
  }, [snapshot.data]);

  return (
    <div className="flex h-full flex-col">
      <TopBar snapshot={snapshot.data} />
      <div className="grid min-h-0 flex-1 grid-cols-1 lg:grid-cols-2">
        <Chat projectId={projectId} messages={messages.data ?? []} streaming={streaming} busy={busy} />
        {snapshot.data ? <Canvas snapshot={snapshot.data} messages={messages.data ?? []} /> : <div className="grid place-items-center"><Spinner /></div>}
      </div>
    </div>
  );
}

function TopBar({ snapshot }: { snapshot?: Snapshot }) {
  const { data: sys } = useSystem();
  const qc = useQueryClient();
  const gpu = sys?.worker.running.gpu;
  const toggle = useMutation({
    mutationFn: (on: boolean) => api.post(`/api/projects/${snapshot!.project.id}/studio/autopilot`, { on }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["studio", snapshot!.project.id] }),
  });
  return (
    <header className="flex h-12 shrink-0 items-center gap-3 border-b border-line px-4">
      <Link to="/" className="grid size-7 place-items-center rounded-lg bg-accent text-sm font-bold text-white" title="All projects">
        ▲
      </Link>
      <div className="min-w-0">
        <div className="truncate text-[13px] font-semibold">{snapshot?.project.name ?? "…"}</div>
        <div className="truncate text-[11px] text-faint">{snapshot?.project.goal}</div>
      </div>
      <div className="ml-auto flex items-center gap-3 text-xs text-muted">
        {sys && (
          <span className="hidden md:inline">
            {sys.hardware.chip} · <span className="num">{sys.hardware.budget_gb.toFixed(1)} GB</span> for ML
          </span>
        )}
        {gpu ? (
          <Badge tone="info">
            <span className="size-1.5 animate-pulse rounded-full bg-current" /> GPU: job {gpu}
          </Badge>
        ) : (
          <Badge>GPU idle</Badge>
        )}
        {sys && !sys.agents.key_configured && <Badge tone="warn">set {sys.agents.key_env} in .env</Badge>}
        {snapshot && (
          <button
            onClick={() => toggle.mutate(!snapshot.autopilot)}
            title={snapshot.autopilot ? "The Tuner builds the model on its own. Click to pause." : "Paused. Click to let the Tuner carry on."}
            className={cx(
              "flex items-center gap-1.5 rounded-full border px-2.5 py-1 font-medium transition",
              snapshot.autopilot ? "border-accent/50 bg-accent-soft text-accent" : "border-line text-muted hover:text-fg",
            )}
          >
            <span className={cx("size-1.5 rounded-full", snapshot.autopilot ? "animate-pulse bg-accent" : "bg-faint")} />
            Autopilot {snapshot.completed ? "done" : snapshot.autopilot ? "on" : "paused"}
          </button>
        )}
        {snapshot && (
          <Link to={`/p/${snapshot.project.id}/overview`} className="rounded-md px-2 py-1 hover:bg-panel-2 hover:text-fg">
            Advanced
          </Link>
        )}
      </div>
    </header>
  );
}

// ── left: chat ───────────────────────────────────────────────────────────────

function Chat({ projectId, messages, streaming, busy }: { projectId: number; messages: TunerMessage[]; streaming: string; busy: boolean }) {
  const [input, setInput] = useState("");
  const { data: sys } = useSystem();
  const send = useMutation({
    mutationFn: (text: string) => api.post(`/api/projects/${projectId}/tuner/message`, { text }),
  });
  const bottom = useRef<HTMLDivElement>(null);
  // Block body: scrollIntoView() now returns a Promise in Chrome, and an effect must not return one.
  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [messages.length, streaming]);

  const submit = (text: string) => {
    const t = text.trim();
    if (!t) return;
    setInput("");
    send.mutate(t);
  };
  const visible = messages.filter(
    (m) => !(m.role === "tool" && QUIET_TOOLS.has(m.meta.name ?? "")) && !m.meta.kickoff && !m.meta.autopilot,
  );
  const lastAssistant = [...messages].reverse().find((m) => m.role === "assistant");
  const wantsGo = !busy && !!lastAssistant && /[“"']go[”"']/i.test(lastAssistant.content) && messages.at(-1)?.id === lastAssistant.id;

  return (
    <section className="flex min-h-0 flex-col border-r border-line">
      <div className="flex items-center gap-2 border-b border-line px-5 py-2.5">
        <span className="grid size-7 place-items-center rounded-full bg-accent-soft text-sm">✦</span>
        <div>
          <div className="text-[13px] font-semibold">Tuner</div>
          <div className="text-[11px] text-faint">
            {busy ? (
              <span className="text-accent">working…</span>
            ) : (
              <>your guide · {sys?.agents.model ?? "…"}</>
            )}
          </div>
        </div>
      </div>

      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto px-5 py-4">
        {groupTools(visible).map((item) =>
          Array.isArray(item) ? <ToolGroup key={item[0].id} steps={item} /> : <ChatItem key={item.id} m={item} />,
        )}
        {streaming && (
          <div className="max-w-[88%] rounded-2xl rounded-tl-sm border border-line bg-panel px-4 py-3 text-[14px] leading-relaxed">
            <Markdown text={streaming} />
            <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-accent align-middle" />
          </div>
        )}
        {busy && !streaming && (
          <div className="flex items-center gap-2 text-xs text-faint">
            <Spinner /> thinking…
          </div>
        )}
        <div ref={bottom} />
      </div>

      <div className="border-t border-line p-4">
        {wantsGo && (
          <div className="mb-2 flex gap-2">
            <Button size="sm" variant="primary" onClick={() => submit("go")}>
              Go
            </Button>
            <Button size="sm" onClick={() => submit("Explain that a bit more first.")}>
              Explain more
            </Button>
          </div>
        )}
        <form
          className="flex gap-2"
          onSubmit={(e) => {
            e.preventDefault();
            submit(input);
          }}
        >
          <TextArea
            rows={2}
            className="flex-1"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                submit(input);
              }
            }}
            placeholder={busy ? "The Tuner is building your model. Type anything to steer it." : "Type to steer the Tuner (optional)…"}
          />
          <Button type="submit" variant="primary" disabled={!input.trim()}>
            Send
          </Button>
        </form>
      </div>
    </section>
  );
}

/** Consecutive tool calls collapse into one group, so the conversation stays readable. */
function groupTools(messages: TunerMessage[]): (TunerMessage | TunerMessage[])[] {
  const out: (TunerMessage | TunerMessage[])[] = [];
  for (const m of messages) {
    const last = out.at(-1);
    if (m.role === "tool" && Array.isArray(last)) last.push(m);
    else out.push(m.role === "tool" ? [m] : m);
  }
  return out;
}

function ToolGroup({ steps }: { steps: TunerMessage[] }) {
  const [open, setOpen] = useState(false);
  const running = steps.find((m) => m.meta.status === "running");
  if (steps.length === 1) return <ChatItem m={steps[0]} />;
  const counts = new Map<string, number>();
  for (const m of steps) {
    const label = TOOL_LABEL[m.meta.name ?? ""] ?? m.meta.name ?? "step";
    counts.set(label, (counts.get(label) ?? 0) + 1);
  }
  const summary = [...counts.entries()].map(([l, n]) => (n > 1 ? `${l.toLowerCase()} ×${n}` : l.toLowerCase())).join(" · ");
  return (
    <div className="pl-1 text-xs text-muted">
      <button onClick={() => setOpen(!open)} className="flex w-full items-start gap-2 text-left hover:text-fg">
        {running ? <Spinner className="mt-0.5 size-3" /> : <span className="text-good">✓</span>}
        <span className="min-w-0 flex-1">
          {running ? `${TOOL_LABEL[running.meta.name ?? ""] ?? running.meta.name}…` : `${steps.length} steps`}
          <span className="text-faint"> · {summary}</span>
        </span>
        <span className={cx("text-faint transition", open && "rotate-90")}>›</span>
      </button>
      {open && (
        <div className="mt-1.5 space-y-1 border-l border-line pl-3">
          {steps.map((m) => (
            <ChatItem key={m.id} m={m} />
          ))}
        </div>
      )}
    </div>
  );
}

function ChatItem({ m }: { m: TunerMessage }) {
  if (m.role === "user")
    return (
      <div className="flex justify-end">
        <div className="max-w-[80%] whitespace-pre-wrap rounded-2xl rounded-tr-sm bg-accent px-4 py-2.5 text-[14px] text-white">{m.content}</div>
      </div>
    );
  if (m.role === "assistant")
    return (
      <div className="max-w-[88%] rounded-2xl rounded-tl-sm border border-line bg-panel px-4 py-3 text-[14px] leading-relaxed">
        <Markdown text={m.content} />
      </div>
    );
  if (m.role === "tool") {
    const running = m.meta.status === "running";
    const failed = !running && /^(Error|An error occurred)/i.test(m.meta.output ?? "");
    return (
      <div className="flex items-center gap-2 pl-1 text-xs text-muted">
        {running ? <Spinner className="size-3" /> : <span className={failed ? "text-bad" : "text-good"}>{failed ? "✗" : "✓"}</span>}
        {TOOL_LABEL[m.meta.name ?? ""] ?? m.meta.name}
      </div>
    );
  }
  // event
  return <div className="text-center text-[11px] text-faint">{eventLabel(m)}</div>;
}

function eventLabel(m: TunerMessage): string {
  if (m.meta.error || m.meta.autopilot_paused) return m.content;
  const job = m.content.match(/^\[Job update\] (\w+) job (\d+) (\w+)/);
  if (job) {
    const kind = { download: "Download", sft: "Training run", dpo: "Preference round", export: "Export" }[job[1]] ?? job[1];
    return `— ${kind} ${job[3]} —`;
  }
  if (m.content.startsWith("[Feedback]")) return "— You finished judging the answers —";
  return m.content;
}

// ── right: canvas ────────────────────────────────────────────────────────────

function stageDone(s: Snapshot): Record<Stage, boolean> {
  const sft = s.checkpoints.some((c) => c.kind === "sft");
  return {
    goal: !!s.project.goal && s.stage !== "goal",
    model: !!s.model?.downloaded,
    data: s.versions.length > 0,
    train: sft,
    evaluate: sft && (s.samples.some((x) => x.target !== "base") || s.feedback.judgements > 0),
    refine: s.checkpoints.some((c) => c.kind === "dpo"),
    export: s.exports.length > 0,
  };
}

function Canvas({ snapshot: s, messages }: { snapshot: Snapshot; messages: TunerMessage[] }) {
  const done = stageDone(s);
  const reached = STAGES.slice(0, STAGES.indexOf(s.stage) + 1);
  // Show every stage that has content or has been reached.
  const visible = STAGES.filter((st) => reached.includes(st) || done[st]);
  const scroller = useRef<HTMLDivElement>(null);
  const activeJob = s.jobs.find((j) => j.status === "running" || j.status === "queued");
  // Keep the current stage in view: when the stage changes, and when a job starts (cards above
  // may have grown since). Delayed a beat so freshly rendered cards have their final height.
  useEffect(() => {
    const t = setTimeout(() => {
      document.getElementById(`stage-${s.stage}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 300);
    return () => clearTimeout(t);
  }, [s.stage, activeJob?.id]);
  return (
    <section className="flex min-h-0 flex-col bg-bg">
      <div className="border-b border-line px-5 py-3">
        <ol className="flex items-center gap-1">
          {STAGES.map((st, i) => (
            <li key={st} className="flex flex-1 items-center gap-1">
              <button
                onClick={() => document.getElementById(`stage-${st}`)?.scrollIntoView({ behavior: "smooth", block: "start" })}
                className={cx(
                  "flex items-center gap-1.5 rounded-full px-2 py-1 text-[11px] font-medium transition",
                  st === s.stage ? "bg-accent text-white" : done[st] ? "text-good" : "text-faint",
                )}
              >
                <span className={cx("grid size-4 place-items-center rounded-full text-[9px]", st === s.stage ? "bg-white/25" : done[st] ? "bg-good-soft" : "bg-panel-2")}>
                  {done[st] && st !== s.stage ? "✓" : i + 1}
                </span>
                {STAGE_LABEL[st]}
              </button>
              {i < STAGES.length - 1 && <span className="h-px flex-1 bg-line" />}
            </li>
          ))}
        </ol>
        {s.note && <p className="mt-2 text-xs text-muted">{s.note}</p>}
      </div>

      <div ref={scroller} className="min-h-0 flex-1 space-y-4 overflow-y-auto p-5">
        {visible.map((st) => (
          <div key={st} id={`stage-${st}`} className="scroll-mt-4">
            <StageCard stage={st} active={st === s.stage} done={done[st]}>
              {st === "goal" && <GoalView s={s} />}
              {st === "model" && <ModelView s={s} />}
              {st === "data" && <DataView s={s} />}
              {st === "train" && <TrainView s={s} />}
              {st === "evaluate" && <EvaluateView s={s} />}
              {st === "refine" && <RefineView s={s} />}
              {st === "export" && <ExportView s={s} />}
            </StageCard>
          </div>
        ))}
      </div>

      {s.completed && (
        <div className="border-t border-good/40 bg-good-soft px-5 py-2.5 text-[13px] text-good">
          ✓ Done. {s.note}
        </div>
      )}
      <Console messages={messages} activeJob={activeJob} />
    </section>
  );
}

function StageCard({ stage, active, done, children }: { stage: Stage; active: boolean; done: boolean; children: React.ReactNode }) {
  return (
    <div className={cx("rounded-xl border bg-panel transition", active ? "border-accent/60 shadow-[0_0_0_3px_var(--accent-soft)]" : "border-line")}>
      <div className="flex items-center gap-2 border-b border-line px-4 py-2.5">
        <span className="text-[13px] font-semibold">{STAGE_LABEL[stage]}</span>
        {active && <Badge tone="accent">now</Badge>}
        {done && !active && <Badge tone="good">done</Badge>}
      </div>
      <div className="p-4">{children}</div>
    </div>
  );
}

function Empty({ children }: { children: React.ReactNode }) {
  return <p className="text-xs text-faint">{children}</p>;
}

function GoalView({ s }: { s: Snapshot }) {
  return (
    <dl className="space-y-2 text-[13px]">
      <div>
        <dt className="text-[11px] uppercase tracking-wide text-faint">Model purpose</dt>
        <dd>{s.project.goal || "The Tuner will fill this in as you talk."}</dd>
      </div>
      {s.project.system_prompt && (
        <div>
          <dt className="text-[11px] uppercase tracking-wide text-faint">System prompt</dt>
          <dd className="font-mono text-xs text-muted">{s.project.system_prompt}</dd>
        </div>
      )}
    </dl>
  );
}

function ModelView({ s }: { s: Snapshot }) {
  const download = s.jobs.find((j) => j.kind === "download" && (j.status === "running" || j.status === "queued"));
  if (!s.model) return <Empty>The Tuner is choosing a base model that fits this Mac.</Empty>;
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <a className="font-mono text-[13px] hover:text-accent" href={`https://huggingface.co/${s.model.repo_id}`} target="_blank" rel="noreferrer">
          {s.model.repo_id}
        </a>
        {s.model.downloaded ? <Badge tone="good">on this Mac</Badge> : download ? <Badge tone="info">downloading…</Badge> : <Badge>not downloaded</Badge>}
      </div>
      {s.model.params != null && (
        <div className="grid grid-cols-4 gap-2 text-xs">
          <Fact label="Parameters" value={fmt.params(s.model.params)} />
          <Fact label="Precision" value={`${s.model.bits}-bit`} />
          <Fact label="On disk" value={fmt.gb(s.model.size_gb)} />
          <Fact label="Layers" value={String(s.model.layers)} />
        </div>
      )}
      {s.model.inference && (
        <div>
          <div className="mb-1 text-[11px] text-faint">Memory to run it (4k context) vs. this Mac's budget</div>
          <MemoryBar est={s.model.inference} />
        </div>
      )}
      {download && <Progress label="Downloading" />}
    </div>
  );
}

function DataView({ s }: { s: Snapshot }) {
  const prepping = s.jobs.find((j) => ["import_dataset", "prepare_dataset", "synthesize"].includes(j.kind) && (j.status === "running" || j.status === "queued"));
  if (!s.datasets.length && !s.versions.length && !prepping) return <Empty>The Tuner is looking for training data that matches the goal.</Empty>;
  return (
    <div className="space-y-3">
      {s.datasets.map((d) => (
        <div key={d.id} className="flex items-center gap-2 text-[13px]">
          <span className="text-faint">⬇</span>
          <span className="font-mono text-xs">{d.source_ref || d.name}</span>
          <span className="text-xs text-muted">{fmt.compact(d.n_rows)} rows</span>
          {d.license && <Badge tone={/mit|apache|cc-by|cc0/i.test(d.license) ? "good" : "neutral"}>{d.license}</Badge>}
        </div>
      ))}
      {prepping && <Progress label={{ import_dataset: "Importing rows", prepare_dataset: "Cleaning and tokenising", synthesize: "Writing examples with GPT-6" }[prepping.kind] ?? "Working"} />}
      {s.versions.map((v) => {
        const dropped = Object.entries(v.cleaning_report?.dropped ?? {});
        return (
          <div key={v.id} className="rounded-lg border border-line bg-panel-2/40 p-3">
            <div className="flex items-center gap-2 text-[13px]">
              <span className="font-medium">Training set v{v.id}</span>
              <Badge tone={v.kind === "dpo" ? "accent" : "info"}>{v.kind === "dpo" ? "preferences" : "examples"}</Badge>
            </div>
            <div className="mt-2 grid grid-cols-4 gap-2 text-xs">
              <Fact label="Train" value={v.n_train.toLocaleString()} />
              <Fact label="Validation" value={v.n_valid.toLocaleString()} />
              <Fact label="Typical length" value={v.token_stats?.p50 ? `${v.token_stats.p50} tok` : "–"} />
              <Fact label="Long (p95)" value={v.token_stats?.p95 ? `${v.token_stats.p95} tok` : "–"} />
            </div>
            {dropped.length > 0 && (
              <p className="mt-2 text-[11px] text-faint">
                Cleaned out: {dropped.map(([k, n]) => `${n} ${k.replace(/_/g, " ")}`).join(", ")}
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}

function TrainView({ s }: { s: Snapshot }) {
  const runs = s.jobs.filter((j) => j.kind === "sft" || j.kind === "dpo");
  const current = runs[0];
  if (!current) return <Empty>Training starts once the data is ready.</Empty>;
  return (
    <div className="space-y-4">
      <LiveRun jobId={current.id} />
      {runs.length > 1 && (
        <div className="space-y-1">
          <div className="text-[11px] uppercase tracking-wide text-faint">Earlier runs</div>
          {runs.slice(1, 6).map((j) => (
            <div key={j.id} className="flex items-center gap-2 text-xs text-muted">
              <Badge tone={j.kind === "dpo" ? "accent" : "info"}>{j.kind.toUpperCase()}</Badge>
              run {j.id}
              <span className="num">val loss {String((j.result.metrics as Record<string, number> | undefined)?.val_loss ?? "–")}</span>
              <StatusBadge status={j.status} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function LiveRun({ jobId }: { jobId: number }) {
  const { job, metrics, progress } = useLiveJob(jobId);
  if (!job) return null;
  const train = metrics.filter((m) => m.split === "train");
  const last = train.at(-1)?.values;
  const total = progress?.total || (job.result.total_iters as number) || 0;
  const cur = progress?.current ?? train.at(-1)?.iteration ?? 0;
  const pct = job.status === "succeeded" ? 100 : total ? Math.min(100, (cur / total) * 100) : 0;
  const eta = last?.it_per_sec && total > cur ? Math.ceil((total - cur) / last.it_per_sec / 60) : null;
  const warnings = (job.result.warnings as { code: string; message: string }[] | undefined) ?? [];
  const isDpo = job.kind === "dpo";
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 text-[13px]">
        <span className="font-medium">{isDpo ? "Preference round" : "Fine-tuning"} · run {job.id}</span>
        <StatusBadge status={job.status} />
        <span className="ml-auto num text-xs text-muted">
          {cur}/{total || "?"} steps{eta != null && job.status === "running" ? ` · ~${eta} min left` : ""}
        </span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-panel-2">
        <div className={cx("h-full transition-all", job.status === "failed" ? "bg-bad" : "bg-accent")} style={{ width: `${pct}%` }} />
      </div>
      <div className="grid grid-cols-4 gap-2 text-xs">
        <Fact label="Train loss" value={fmt.num(last?.loss)} />
        <Fact label="Val loss" value={fmt.num(metrics.filter((m) => m.split === "val").at(-1)?.values.loss)} />
        {isDpo ? <Fact label="Prefers chosen" value={last?.accuracy != null ? `${Math.round(last.accuracy * 100)}%` : "–"} /> : <Fact label="Tokens/sec" value={last ? String(Math.round(last.tokens_per_sec)) : "–"} />}
        <Fact label="Peak memory" value={fmt.gb(last?.peak_mem_gb)} />
      </div>
      <MetricChart
        height={180}
        metrics={metrics}
        series={[
          { key: "loss", label: "train", split: "train", color: "var(--chart-1)" },
          { key: "loss", label: "validation", split: "val", color: "var(--chart-2)", dashed: true },
        ]}
      />
      {warnings.map((w) => (
        <p key={w.code} className="rounded-lg border border-warn/40 bg-warn-soft px-3 py-2 text-xs text-warn">
          <strong>{w.code.replace("_", " ")}:</strong> {w.message}
        </p>
      ))}
      {job.status === "failed" && <p className="rounded-lg bg-bad-soft px-3 py-2 text-xs text-bad">{job.error}</p>}
    </div>
  );
}

function EvaluateView({ s }: { s: Snapshot }) {
  const pending = s.comparisons.filter((c) => c.status === "pending");
  const groups = useMemo(() => {
    const byPrompt = new Map<string, Sample[]>();
    for (const x of s.samples) byPrompt.set(x.prompt, [...(byPrompt.get(x.prompt) ?? []), x]);
    return [...byPrompt.entries()].slice(0, 6);
  }, [s.samples]);
  if (!groups.length && !s.comparisons.length) return <Empty>The Tuner will test the model on your example questions here.</Empty>;
  return (
    <div className="space-y-4">
      {pending.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Badge tone="accent">your turn</Badge>
            <span className="text-[13px]">Pick the better answer ({pending.length} left). This teaches the model your taste.</span>
          </div>
          <CompareTask key={pending[0].id} c={pending[0]} projectId={s.project.id} />
        </div>
      )}
      {groups.map(([prompt, answers]) => (
        <div key={prompt} className="rounded-lg border border-line p-3">
          <p className="text-[13px] font-medium">{prompt}</p>
          <div className={cx("mt-2 grid gap-2", answers.length > 1 && "md:grid-cols-2")}>
            {answers.slice(0, 2).map((a, i) => (
              <div key={i} className="rounded-md bg-panel-2/50 p-2">
                <Badge tone={a.target === "base" ? "neutral" : "good"}>{a.target === "base" ? "before training" : "trained"}</Badge>
                <p className="mt-1 whitespace-pre-wrap text-xs leading-relaxed text-muted">{a.text}</p>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

function CompareTask({ c, projectId }: { c: Comparison; projectId: number }) {
  const qc = useQueryClient();
  const [choice, setChoice] = useState<string | null>(null);
  const [rewrite, setRewrite] = useState("");
  const [critique, setCritique] = useState("");
  const judge = useMutation({
    mutationFn: () => api.post(`/api/projects/${projectId}/studio/judge`, { comparison_id: c.id, choice, edited_answer: rewrite, critique }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["studio", projectId] }),
  });
  const identical = c.a.trim() === c.b.trim();
  return (
    <div className="rounded-xl border border-accent/40 bg-accent-soft/30 p-3">
      <p className="text-[13px] font-medium">{c.prompt}</p>
      <div className="mt-2 grid gap-2 md:grid-cols-2">
        {(["a", "b"] as const).map((side) => (
          <button
            key={side}
            onClick={() => (setChoice(side), setRewrite(c[side]))}
            className={cx("rounded-lg border p-2 text-left text-xs leading-relaxed transition", choice === side ? "border-good bg-good-soft" : "border-line bg-panel hover:border-line-strong")}
          >
            <span className="font-semibold uppercase">{side}</span>
            <p className="mt-1 whitespace-pre-wrap">{c[side]}</p>
          </button>
        ))}
      </div>
      {identical && <p className="mt-2 text-[11px] text-warn">These are identical. Rewrite a better answer below, or skip with "About the same".</p>}
      <div className="mt-2 flex flex-wrap gap-2">
        <Button size="sm" variant={choice === "tie" ? "primary" : "secondary"} onClick={() => setChoice("tie")}>
          About the same
        </Button>
        <Button size="sm" variant={choice === "both_bad" ? "danger" : "secondary"} onClick={() => (setChoice("both_bad"), setRewrite(""))}>
          Both are bad
        </Button>
      </div>
      {choice && (
        <div className="mt-2 space-y-2">
          <TextArea rows={3} value={rewrite} onChange={(e) => setRewrite(e.target.value)} placeholder="Optional: write the ideal answer (the strongest signal you can give)" />
          <TextArea rows={1} value={critique} onChange={(e) => setCritique(e.target.value)} placeholder="Optional: what was wrong?" />
          <Button
            size="sm"
            variant="primary"
            loading={judge.isPending}
            disabled={choice === "both_bad" && !rewrite.trim() && !critique.trim()}
            onClick={() => judge.mutate()}
          >
            Save and next
          </Button>
          {judge.error && <p className="text-xs text-bad">{String((judge.error as Error).message)}</p>}
        </div>
      )}
    </div>
  );
}

const VERDICT: Record<string, { label: string; tone: "good" | "warn" | "bad" | "neutral" }> = {
  a: { label: "A better", tone: "good" },
  b: { label: "B better", tone: "good" },
  tie: { label: "about the same", tone: "neutral" },
  both_bad: { label: "both wrong", tone: "bad" },
};

function RefineView({ s }: { s: Snapshot }) {
  const dpo = s.checkpoints.filter((c) => c.kind === "dpo");
  const reviews = s.comparisons.filter((c) => c.judge === "ai").slice(-8).reverse();
  return (
    <div className="space-y-3">
      {reviews.length > 0 && (
        <div className="space-y-2">
          <div className="text-[11px] uppercase tracking-wide text-faint">Reviewed by GPT-6 · each verdict becomes training signal</div>
          {reviews.map((c) => (
            <details key={c.id} className="group rounded-lg border border-line p-2.5">
              <summary className="flex cursor-pointer list-none items-center gap-2 text-xs">
                <Badge tone={VERDICT[c.choice ?? "tie"]?.tone ?? "neutral"}>{VERDICT[c.choice ?? "tie"]?.label ?? c.choice}</Badge>
                <span className="min-w-0 flex-1 truncate text-fg">{c.prompt}</span>
                <span className="text-faint transition group-open:rotate-90">›</span>
              </summary>
              <div className="mt-2 space-y-1.5 text-xs leading-relaxed">
                {c.critique && <p className="text-muted">{c.critique}</p>}
                {c.ideal && (
                  <p className="rounded-md bg-good-soft px-2 py-1.5 text-fg">
                    <span className="font-medium text-good">Ideal answer: </span>
                    {c.ideal}
                  </p>
                )}
              </div>
            </details>
          ))}
        </div>
      )}
      <div className="grid grid-cols-3 gap-2 text-xs">
        <Fact label="Your judgements" value={String(s.feedback.judgements)} />
        <Fact label="Preference pairs ready" value={String(s.feedback.pairs_ready)} />
        <Fact label="Preference rounds" value={String(dpo.length)} />
      </div>
      {dpo.map((c) => (
        <p key={c.id} className="text-xs text-muted">
          Round {c.job_id}: prefers your picks {c.metrics.reward_accuracy != null ? `${Math.round(c.metrics.reward_accuracy * 100)}%` : "–"} of the time
        </p>
      ))}
    </div>
  );
}

function ExportView({ s }: { s: Snapshot }) {
  if (!s.exports.length) return <Empty>When you're happy, the Tuner packages the model here.</Empty>;
  return (
    <div className="space-y-3">
      {s.exports.map((e) => (
        <div key={e.path} className="space-y-1.5">
          <div className="flex flex-wrap items-center gap-2 text-[13px]">
            <span className="font-medium">{e.path.split("/").pop()}</span>
            <Badge>{fmt.gb(e.size_gb)}</Badge>
            <Badge tone="good">runs on {e.min_ram_gb} GB+ Macs</Badge>
          </div>
          <pre className="overflow-x-auto rounded-md bg-bg px-3 py-2 font-mono text-[11.5px] text-muted">mlx_lm.generate --model {e.path} --prompt "Hello"</pre>
        </div>
      ))}
    </div>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md bg-panel-2/60 px-2 py-1.5">
      <div className="text-[10px] uppercase tracking-wide text-faint">{label}</div>
      <div className="num text-[13px]">{value}</div>
    </div>
  );
}

function Progress({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-2 text-xs text-info">
      <Spinner className="size-3" /> {label}…
      <div className="h-1 flex-1 overflow-hidden rounded-full bg-panel-2">
        <div className="h-full w-1/3 animate-[pulse_1.2s_ease-in-out_infinite] rounded-full bg-info" />
      </div>
    </div>
  );
}

// ── console ──────────────────────────────────────────────────────────────────

function Console({ messages, activeJob }: { messages: TunerMessage[]; activeJob?: Job }) {
  const [tab, setTab] = useState<"agent" | "job">("agent");
  const [open, setOpen] = useState(true);
  const live = useLiveJob(activeJob?.id ?? null);
  useEffect(() => {
    if (activeJob) setTab("job");
  }, [activeJob?.id]); // eslint-disable-line react-hooks/exhaustive-deps

  const agentLines = messages
    .filter((m) => m.role === "tool")
    .slice(-80)
    .flatMap((m) => {
      const args = Object.entries(m.meta.args ?? {})
        .map(([k, v]) => `${k}=${typeof v === "string" ? JSON.stringify(v.length > 40 ? v.slice(0, 40) + "…" : v) : JSON.stringify(v)}`)
        .join(" ");
      const head = `› ${m.meta.name} ${args}`;
      if (m.meta.status === "running") return [head, "  …"];
      return [head, `  ↳ ${(m.meta.output ?? "").replace(/\s+/g, " ").slice(0, 160)}`];
    });

  return (
    <div className="border-t border-line bg-panel">
      <div className="flex items-center gap-1 px-3 py-1.5">
        {(["agent", "job"] as const).map((t) => (
          <button
            key={t}
            onClick={() => (setTab(t), setOpen(true))}
            className={cx("rounded-md px-2 py-0.5 text-[11px] font-medium", tab === t && open ? "bg-panel-2 text-fg" : "text-faint hover:text-fg")}
          >
            {t === "agent" ? "Agent activity" : activeJob ? `Job ${activeJob.id} output` : "Job output"}
          </button>
        ))}
        {activeJob && <span className="size-1.5 animate-pulse rounded-full bg-info" />}
        <button className="ml-auto text-[11px] text-faint hover:text-fg" onClick={() => setOpen(!open)}>
          {open ? "hide" : "show"}
        </button>
      </div>
      {open && (
        <div className="px-3 pb-3">
          <JobLog height={150} lines={tab === "agent" ? agentLines : live.log.length ? live.log : ["No job running."]} />
        </div>
      )}
    </div>
  );
}
