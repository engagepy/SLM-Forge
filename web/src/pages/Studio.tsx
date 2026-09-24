import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Fragment, useEffect, useRef, useState } from "react";
import { useParams } from "react-router";

import {
  api,
  type Comparison,
  type Evaluation,
  fmt,
  isActive,
  type Job,
  JOB_KIND,
  type PendingAction,
  runCommand,
  type Sample,
  type Snapshot,
  type Stage,
  STAGES,
  type TunerMessage,
} from "../api";
import { MetricChart } from "../components/Charts";
import ChatComposer from "../components/ChatComposer";
import JobLog from "../components/JobLog";
import Markdown from "../components/Markdown";
import MetricsBar from "../components/MetricsBar";
import { invalidate, runProgress, useLiveJob, useStudio, useSystem, useTunerStream } from "../hooks";
import { Badge, Bubble, Button, CodeBlock, cx, ErrorNote, LinkButton, MemoryBar, ProgressBar, SectionLabel, Spinner, StatusBadge, TextArea } from "../ui";

const STAGE_LABEL: Record<Stage, string> = {
  goal: "Goal",
  model: "Model",
  data: "Data",
  train: "Train",
  evaluate: "Evaluate",
  refine: "Refine",
  export: "Export",
};

const STAGE_VIEW: Record<Stage, React.FC<{ s: Snapshot }>> = {
  goal: GoalView,
  model: ModelView,
  data: DataView,
  train: TrainView,
  evaluate: EvaluateView,
  refine: RefineView,
  export: ExportView,
};

const TOOL_LABEL: Record<string, string> = {
  update_project: "Saving the project brief",
  find_base_models: "Searching for base models",
  choose_base_model: "Proposing the base model",
  search_datasets: "Searching datasets",
  preview_dataset: "Previewing a dataset",
  scout_datasets: "Sending DataScout to find datasets",
  plan_preparation: "Asking DataPrep for a cleaning plan",
  import_dataset: "Importing data",
  sample_rows: "reading sample rows",
  check_mapping: "checking a mapping",
  dataset_card: "reading a dataset card",
  inspect_dataset: "Inspecting the data",
  prepare_dataset: "Cleaning and preparing the data",
  plan_training: "Planning the training run",
  start_training: "Proposing a training run",
  training_progress: "Checking on training",
  cancel_job: "Stopping a job",
  try_model: "Testing the model",
  evaluate_model: "Scoring the model on the test questions",
  serve_checkpoint: "Rolling back to an earlier checkpoint",
  ask_user_to_compare: "Preparing answers for you to compare",
  feedback_summary: "Reading your feedback",
  generate_synthetic_examples: "Writing training examples",
  review_synthetic_examples: "Reviewing examples",
  build_dataset_from_examples: "Building a dataset",
  export_model: "Proposing the export",
  ai_review_answers: "Having GPT-6 review the model's answers",
  finish_project: "Wrapping up",
};
// Bookkeeping calls show in the console, not the chat.
const QUIET_TOOLS = new Set(["get_status", "set_stage"]);

// ── page ─────────────────────────────────────────────────────────────────────

export default function Studio() {
  const projectId = Number(useParams().projectId);
  const snapshot = useStudio(projectId);
  const messages = useQuery({
    queryKey: ["tuner-messages", projectId],
    queryFn: () => api.get<TunerMessage[]>(`/api/projects/${projectId}/tuner/messages`),
  });
  const { streaming, busy, setBusy } = useTunerStream(projectId);

  // Opening a project never starts anything: the Tuner only begins from an explicit action
  // (creating the project on Home, or the Start button below).
  useEffect(() => {
    if (snapshot.data) setBusy((b) => b || snapshot.data!.tuner_busy);
  }, [snapshot.data, setBusy]);

  return (
    <div className="flex h-full flex-col">
      <TopBar snapshot={snapshot.data} />
      {/* Side by side from lg; stacked on a phone, each half scrolling on its own. */}
      <div className="grid min-h-0 flex-1 grid-cols-1 grid-rows-[minmax(0,1fr)_minmax(0,1fr)] lg:grid-cols-2 lg:grid-rows-1">
        <Chat
          projectId={projectId}
          messages={messages.data ?? []}
          loaded={messages.isSuccess}
          streaming={streaming}
          busy={busy}
          pending={snapshot.data?.pending_action ?? null}
        />
        {snapshot.data ? <Canvas snapshot={snapshot.data} messages={messages.data ?? []} /> : <div className="grid place-items-center"><Spinner /></div>}
      </div>
    </div>
  );
}

function TopBar({ snapshot }: { snapshot?: Snapshot }) {
  return (
    <>
      <MetricsBar snapshot={snapshot} />
      <header className="flex h-11 shrink-0 items-center gap-3 border-b border-line px-4">
        <div className="min-w-0">
          <div className="truncate text-[13px] font-semibold">{snapshot?.project.name ?? "…"}</div>
          <div className="truncate text-[11px] text-faint" title={snapshot?.project.goal}>
            {snapshot?.project.goal}
          </div>
        </div>
      </header>
    </>
  );
}

// ── left: chat ───────────────────────────────────────────────────────────────

function Chat({
  projectId,
  messages,
  loaded,
  streaming,
  busy,
  pending,
}: {
  projectId: number;
  messages: TunerMessage[];
  loaded: boolean;
  streaming: string;
  busy: boolean;
  pending: PendingAction | null;
}) {
  const { data: sys } = useSystem();
  const send = useMutation({
    // pending_id: a plain "yes" confirms only the card the user was looking at.
    mutationFn: (text: string) => api.post(`/api/projects/${projectId}/tuner/message`, { text, pending_id: pending?.id ?? null }),
  });
  const bottom = useRef<HTMLDivElement>(null);
  // Block body: scrollIntoView() now returns a Promise in Chrome, and an effect must not return one.
  useEffect(() => {
    bottom.current?.scrollIntoView({ block: "end" });
  }, [messages.length, streaming]);
  const visible = messages.filter(
    (m) => !(m.role === "tool" && QUIET_TOOLS.has(m.meta.name ?? "")) && !m.meta.kickoff && !m.meta.autopilot,
  );

  return (
    <section className="flex min-h-0 min-w-0 flex-col border-b border-line lg:border-r lg:border-b-0">
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
        {loaded && messages.length === 0 && !busy && <StartTuner projectId={projectId} />}
        {groupTools(visible).map((item) =>
          Array.isArray(item) ? <ToolGroup key={item[0].id} steps={item} /> : <ChatItem key={item.id} m={item} />,
        )}
        {streaming && (
          <Bubble role="assistant">
            <Markdown text={streaming} />
            <span className="ml-0.5 inline-block h-4 w-1.5 animate-pulse bg-accent align-middle" />
          </Bubble>
        )}
        {busy && !streaming && (
          <div className="flex items-center gap-2 text-xs text-faint">
            <Spinner /> thinking…
          </div>
        )}
        <div ref={bottom} />
      </div>

      {(pending || send.error) && (
        <div className="border-t border-line px-4 pt-4">
          {pending && <ConfirmCard projectId={projectId} action={pending} />}
          <ErrorNote error={send.error} />
        </div>
      )}
      <ChatComposer
        busy={busy}
        sending={send.isPending}
        onSend={(text) => send.mutateAsync(text)}
        placeholder={
          pending
            ? "Say yes, or tell the Tuner what to change…"
            : busy
              ? "The Tuner is building your model. Type anything to steer it."
              : "Type to steer the Tuner (optional)…"
        }
      />
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
    const label = (m.meta.agent ? `${m.meta.agent}: ` : "") + (TOOL_LABEL[m.meta.name ?? ""] ?? m.meta.name ?? "step");
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

function StartTuner({ projectId }: { projectId: number }) {
  const qc = useQueryClient();
  const start = useMutation({
    mutationFn: () => api.post(`/api/projects/${projectId}/tuner/start`),
    onSuccess: () => invalidate(qc, "studio", "sessions"),
  });
  return (
    <div className="rounded-xl border border-line bg-panel p-4 text-[13px] text-muted">
      <p>
        The Tuner hasn't started on this project. Once started, it prepares everything itself and asks you before each run
        (downloading a model, training, exporting).
      </p>
      <ErrorNote error={start.error} />
      <Button className="mt-3" size="sm" variant="primary" loading={start.isPending} onClick={() => start.mutate()}>
        Start the Tuner
      </Button>
    </div>
  );
}

const ACTION_ICON: Record<PendingAction["kind"], string> = { model: "◆", sft: "▲", dpo: "▲", export: "⬇", synthesize: "✎", review: "⚖", import: "⇣", evaluate: "★", scout: "⌕", prep: "✂" };

/** The Tuner's proposed run. It only starts from here (or a plain "yes" in the chat). */
function ConfirmCard({ projectId, action }: { projectId: number; action: PendingAction }) {
  const qc = useQueryClient();
  const [why, setWhy] = useState("");
  const [asking, setAsking] = useState(false);
  const decide = useMutation({
    mutationFn: (go: boolean) =>
      api.post(`/api/projects/${projectId}/studio/${go ? "confirm" : "decline"}`, { action_id: action.id, reason: why }),
    onSuccess: () => {
      setAsking(false);
      setWhy("");
    },
    onSettled: () => invalidate(qc, "studio", "sessions"),
  });
  const facts = actionFacts(action);
  return (
    <div className="mb-3 rounded-xl border border-accent/50 bg-accent-soft/40 p-3.5" role="group" aria-label="Waiting for your go-ahead">
      <div className="flex items-start gap-2.5">
        <span className="mt-0.5 grid size-6 shrink-0 place-items-center rounded-full bg-accent text-[11px] text-white">{ACTION_ICON[action.kind]}</span>
        <div className="min-w-0 flex-1">
          <div className="text-[11px] font-medium uppercase tracking-wide text-accent">Waiting for your go-ahead</div>
          <div className="text-[14px] font-semibold">{action.title}</div>
          {action.reason && <p className="mt-0.5 text-[13px] text-muted">{action.reason}</p>}
          {facts.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {facts.map((f) => (
                <Badge key={f}>{f}</Badge>
              ))}
            </div>
          )}
        </div>
      </div>
      {asking && (
        <TextArea
          rows={2}
          className="mt-3 w-full"
          autoFocus
          value={why}
          onChange={(e) => setWhy(e.target.value)}
          placeholder="Optional: what would you rather do? (e.g. “use fewer examples”)"
        />
      )}
      <ErrorNote error={decide.error} />
      <div className="mt-3 flex gap-2">
        {asking ? (
          <>
            <Button size="sm" loading={decide.isPending} onClick={() => decide.mutate(false)}>
              Send “not now”
            </Button>
            <Button size="sm" variant="ghost" onClick={() => setAsking(false)}>
              Back
            </Button>
          </>
        ) : (
          <>
            <Button size="sm" variant="primary" loading={decide.isPending} onClick={() => decide.mutate(true)}>
              Go ahead
            </Button>
            <Button size="sm" onClick={() => setAsking(true)}>
              Not now
            </Button>
          </>
        )}
      </div>
    </div>
  );
}

function actionFacts(a: PendingAction): string[] {
  const d = a.details as Record<string, number | string | boolean | null | undefined>;
  const out: string[] = [];
  if (a.kind === "model") {
    if (d.params_b) out.push(`${d.params_b}B parameters`);
    out.push(d.on_this_mac ? "already on this Mac" : "needs a download");
  } else if (a.kind === "sft" || a.kind === "dpo") {
    if (d.train_examples) out.push(`${d.train_examples} examples`);
    if (d.iterations) out.push(`${d.iterations} steps`);
    if (d.epochs) out.push(`${Number(d.epochs).toFixed(1)} epochs`);
    if (d.memory_gb) out.push(`${d.memory_gb} GB of ${d.budget_gb} GB`);
    if (d.would_queue_behind) out.push("will queue behind another run");
  } else if (a.kind === "export") {
    out.push(d.quantize_bits ? `${d.quantize_bits}-bit` : "no extra quantizing");
  } else if (a.kind === "synthesize") {
    out.push(`${d.count} ${d.kind === "preference" ? "preference pairs" : "examples"}`, "uses the OpenAI API");
  } else if (a.kind === "review") {
    out.push(`${d.prompts} questions`, "runs the model, then uses the OpenAI API");
  } else if (a.kind === "evaluate") {
    out.push(`${d.questions} test questions`, String(d.target), "runs the model, then uses the OpenAI API");
  } else if (a.kind === "scout" || a.kind === "prep") {
    out.push("a specialist agent · a handful of API calls");
  } else if (a.kind === "import") {
    out.push(`up to ${Number(d.max_rows).toLocaleString()} rows`, "downloads from Hugging Face");
  }
  return out;
}

function ChatItem({ m }: { m: TunerMessage }) {
  if (m.role === "user") return <Bubble role="user">{m.content}</Bubble>;
  if (m.role === "assistant")
    return (
      <Bubble role="assistant">
        <Markdown text={m.content} />
      </Bubble>
    );
  if (m.role === "tool") {
    const running = m.meta.status === "running";
    const failed = !running && /^(Error|An error occurred)/i.test(m.meta.output ?? "");
    return (
      <div className="flex items-center gap-2 pl-1 text-xs text-muted">
        {running ? <Spinner className="size-3" /> : <span className={failed ? "text-bad" : "text-good"}>{failed ? "✗" : "✓"}</span>}
        {m.meta.agent && <span className="rounded bg-panel-2 px-1 text-[10px] font-medium text-faint">{m.meta.agent}</span>}
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
    return `— ${JOB_KIND[job[1]]?.name ?? job[1]} ${job[3]} —`;
  }
  if (m.content.startsWith("[Feedback]")) return "— You finished judging the answers —";
  if (m.meta.confirmed) return m.meta.error ? m.content : "— You said go ahead —";
  if (m.meta.declined) return "— You said not now —";
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
    evaluate: sft && (s.evals.some((e) => e.checkpoint_id != null) || s.samples.some((x) => x.target !== "base") || s.feedback.judgements > 0),
    refine: s.checkpoints.some((c) => c.kind === "dpo"),
    export: s.exports.length > 0,
  };
}

function Canvas({ snapshot: s, messages }: { snapshot: Snapshot; messages: TunerMessage[] }) {
  const done = stageDone(s);
  const reached = STAGES.slice(0, STAGES.indexOf(s.stage) + 1);
  // Show every stage that has content or has been reached.
  const visible = STAGES.filter((st) => reached.includes(st) || done[st]);
  const activeJob = s.jobs.find(isActive);
  // Keep the current stage in view: when the stage changes, and when a job starts (cards above
  // may have grown since). Delayed a beat so freshly rendered cards have their final height.
  useEffect(() => {
    const t = setTimeout(() => {
      document.getElementById(`stage-${s.stage}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
    }, 300);
    return () => clearTimeout(t);
  }, [s.stage, activeJob?.id]);
  return (
    <section className="flex min-h-0 min-w-0 flex-col bg-bg">
      <div className="border-b border-line px-5 py-3">
        <ol className="flex items-center gap-1 overflow-x-auto">
          {STAGES.map((st, i) => (
            <li key={st} className="flex flex-1 items-center gap-1">
              <button
                onClick={() => document.getElementById(`stage-${st}`)?.scrollIntoView({ behavior: "smooth", block: "start" })}
                className={cx(
                  "flex shrink-0 items-center gap-1.5 rounded-full px-2 py-1 text-[11px] font-medium whitespace-nowrap transition",
                  // A finished stage shows its tick even when it's the current one (e.g. Export at the end).
                  st === s.stage && !done[st] ? "bg-accent text-white" : done[st] ? "text-good" : "text-faint",
                  st === s.stage && done[st] && "bg-good-soft",
                )}
              >
                <span
                  className={cx(
                    "grid size-4 place-items-center rounded-full text-[9px]",
                    done[st] ? "bg-good text-white" : st === s.stage ? "bg-white/25" : "bg-panel-2",
                  )}
                >
                  {done[st] ? "✓" : i + 1}
                </span>
                {STAGE_LABEL[st]}
              </button>
              {i < STAGES.length - 1 && <span className="h-px flex-1 bg-line" />}
            </li>
          ))}
        </ol>
        {s.note && <p className="mt-2 text-xs text-muted">{s.note}</p>}
      </div>

      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-5">
        {visible.map((st) => {
          const View = STAGE_VIEW[st];
          return (
            <div key={st} id={`stage-${st}`} className="scroll-mt-4">
              <StageCard stage={st} active={st === s.stage} done={done[st]}>
                <View s={s} />
              </StageCard>
            </div>
          );
        })}
      </div>

      {s.completed && <DoneBanner s={s} />}
      <Console messages={messages} activeJob={activeJob} />
    </section>
  );
}

/** Finished, but not closed: try the model, or ask the Tuner to make it better. */
function DoneBanner({ s }: { s: Snapshot }) {
  const improve = useMutation({
    mutationFn: () =>
      api.post(`/api/projects/${s.project.id}/tuner/message`, { text: "I'd like to keep improving this model. What would you suggest?" }),
  });
  return (
    <div className="flex items-center gap-3 border-t border-good/40 bg-good-soft px-5 py-2.5 text-[13px] text-good">
      <span className="min-w-0 flex-1">✓ Done. {s.note}</span>
      <Button size="sm" variant="ghost" loading={improve.isPending} disabled={improve.isSuccess} onClick={() => improve.mutate()}>
        Keep improving
      </Button>
      {!!s.exports.length && (
        <LinkButton to={`/p/${s.project.id}/try`} variant="good" size="sm" className="shrink-0">
          Try your model →
        </LinkButton>
      )}
    </div>
  );
}

function StageCard({ stage, active, done, children }: { stage: Stage; active: boolean; done: boolean; children: React.ReactNode }) {
  return (
    <div className={cx("rounded-xl border bg-panel transition", active ? "border-accent/60 shadow-[0_0_0_3px_var(--accent-soft)]" : "border-line")}>
      <div className="flex items-center gap-2 border-b border-line px-4 py-2.5">
        <span className="text-[13px] font-semibold">{STAGE_LABEL[stage]}</span>
        {active && !done && <Badge tone="accent">now</Badge>}
        {done && <Badge tone="good">done</Badge>}
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
        <dt><SectionLabel>Model purpose</SectionLabel></dt>
        <dd>{s.project.goal || "The Tuner will fill this in as you talk."}</dd>
      </div>
      {s.project.system_prompt && (
        <div>
          <dt><SectionLabel>System prompt</SectionLabel></dt>
          <dd className="font-mono text-xs text-muted">{s.project.system_prompt}</dd>
        </div>
      )}
      {Object.keys(s.project.plan ?? {}).length > 0 && (
        <div>
          <dt><SectionLabel>The Tuner's plan (say so in the chat to change it)</SectionLabel></dt>
          <dd>
            <dl className="mt-1 grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-0.5 text-xs">
              {Object.entries(s.project.plan).map(([k, v]) => (
                <Fragment key={k}>
                  <dt className="text-faint">{k.replace(/_/g, " ")}</dt>
                  <dd className="text-muted">{v}</dd>
                </Fragment>
              ))}
            </dl>
          </dd>
        </div>
      )}
      {s.project.test_questions?.length > 0 && (
        <div>
          <dt>
            <SectionLabel>
              Test set · {s.project.test_questions.length} cases, {s.project.test_questions.filter((q) => q.expected).length} with an expected output
            </SectionLabel>
          </dt>
          <dd>
            <ol className="mt-1 list-decimal space-y-0.5 pl-5 text-xs text-muted">
              {s.project.test_questions.map((q) => (
                <li key={q.input} className={cx(q.kind === "should-not" && "italic")} title={q.expected ? `Expected: ${q.expected}` : undefined}>
                  {q.input}
                  {q.kind === "should-not" && <span className="ml-1 text-faint">(should return nothing)</span>}
                </li>
              ))}
            </ol>
          </dd>
        </div>
      )}
    </dl>
  );
}

function ModelView({ s }: { s: Snapshot }) {
  const download = s.jobs.find((j) => j.kind === "download" && isActive(j));
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
        <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
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
  const prepping = s.jobs.find((j) => ["import_dataset", "prepare_dataset", "synthesize"].includes(j.kind) && isActive(j));
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
      {prepping && <Progress label={JOB_KIND[prepping.kind]?.doing ?? "Working"} />}
      {s.versions.map((v) => {
        const dropped = Object.entries(v.cleaning_report?.dropped ?? {});
        return (
          <div key={v.id} className="rounded-lg border border-line bg-panel-2/40 p-3">
            <div className="flex items-center gap-2 text-[13px]">
              <span className="font-medium">Training set v{v.id}</span>
              <Badge tone={v.kind === "dpo" ? "accent" : "info"}>{v.kind === "dpo" ? "preferences" : "examples"}</Badge>
            </div>
            <div className="mt-2 grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
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
          <SectionLabel>Earlier runs</SectionLabel>
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
  const live = useLiveJob(jobId);
  const { job, metrics } = live;
  if (!job) return null;
  const { current: cur, total, pct, minutesLeft: eta, last, lastVal, warnings } = runProgress(live);
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
      <ProgressBar pct={pct} failed={job.status === "failed"} />
      <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
        <Fact label="Train loss" value={fmt.num(last?.loss)} />
        <Fact label="Val loss" value={fmt.num(lastVal?.loss)} />
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
  const byPrompt = new Map<string, Sample[]>();
  for (const x of s.samples) byPrompt.set(x.prompt, [...(byPrompt.get(x.prompt) ?? []), x]);
  const groups = [...byPrompt.entries()].slice(0, 6);
  if (!groups.length && !s.comparisons.length && !s.evals.length) return <Empty>The Tuner will score the model on your test questions here.</Empty>;
  return (
    <div className="space-y-4">
      {s.evals.length > 0 && <Scores evals={s.evals} />}
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

/** Scores per checkpoint, latest first: the before/after as numbers. */
function Scores({ evals }: { evals: Evaluation[] }) {
  const [open, setOpen] = useState<string | null>(null);
  const best = Math.max(...evals.map((e) => e.mean));
  const label = (e: Evaluation) => (e.checkpoint_id == null ? "before training" : `checkpoint ${e.checkpoint_id}`);
  return (
    <div className="space-y-1.5">
      {[...evals].reverse().map((e) => (
        <div key={e.id} className="rounded-lg border border-line">
          <button onClick={() => setOpen(open === e.id ? null : e.id)} className="flex w-full items-center gap-3 px-3 py-2 text-left">
            <span className={cx("num w-12 text-[15px] font-semibold", e.mean === best ? "text-good" : "text-fg")}>{e.mean.toFixed(1)}</span>
            <span className="text-[11px] text-faint">/ 10{e.exact_rate != null && ` · ${Math.round(e.exact_rate * 100)}% exact`}</span>
            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-panel-2">
              <div className={cx("h-full", e.mean === best ? "bg-good" : "bg-accent")} style={{ width: `${e.mean * 10}%` }} />
            </div>
            <Badge tone={e.checkpoint_id == null ? "neutral" : e.mean === best ? "good" : "info"}>{label(e)}</Badge>
            <span className={cx("text-faint transition", open === e.id && "rotate-90")}>›</span>
          </button>
          {open === e.id && (
            <ul className="space-y-2 border-t border-line px-3 py-2">
              {e.items.map((it) => (
                <li key={it.prompt} className="text-xs">
                  <div className="flex items-start gap-2">
                    <span className={cx("num w-5 shrink-0 font-semibold", it.score >= 7 ? "text-good" : it.score >= 4 ? "text-warn" : "text-bad")}>{it.score}</span>
                    <div className="min-w-0">
                      <p className="font-medium">{it.prompt}</p>
                      <p className="whitespace-pre-wrap text-muted">{it.answer}</p>
                      <p className="mt-0.5 text-faint">{it.reason}</p>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          )}
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
          <SectionLabel>Reviewed by GPT-6 · each verdict becomes training signal</SectionLabel>
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
  if (!s.exports.length) return <Empty>When you're happy, the Tuner packages the model here (it asks you first).</Empty>;
  return (
    <div className="space-y-3">
      {s.exports.map((e) => (
        <div key={e.path} className="space-y-1.5">
          <div className="flex flex-wrap items-center gap-2 text-[13px]">
            <span className="font-medium">{e.path.split("/").pop()}</span>
            <Badge>{fmt.gb(e.size_gb)}</Badge>
            <Badge tone="good">runs on {e.min_ram_gb} GB+ Macs</Badge>
            <LinkButton to={`/p/${s.project.id}/try?export=${e.job_id}`} variant="primary" size="sm" className="ml-auto">
              ▶ Try it
            </LinkButton>
          </div>
          <CodeBlock text={runCommand(e, s.project.system_prompt)} />
          {!e.system_prompt_built_in && s.project.system_prompt && (
            <p className="text-[11px] text-faint">This export needs the system prompt passed in; newer exports build it in.</p>
          )}
        </div>
      ))}
    </div>
  );
}

function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md bg-panel-2/60 px-2 py-1.5">
      <SectionLabel className="text-[10px]">{label}</SectionLabel>
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
  // Training runs: how far along, as done/total iterations, right on the tab.
  const training = activeJob && (activeJob.kind === "sft" || activeJob.kind === "dpo") ? runProgress(live) : null;
  const steps = training && training.total ? `${training.current}/${training.total}` : null;

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
            {t === "job" && steps && <span className="num ml-1.5 text-faint">{steps} iterations</span>}
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
