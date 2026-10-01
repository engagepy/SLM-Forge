// The left half: the conversation with the Tuner, its tool activity, and the composer.
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { api, JOB_KIND, type PendingAction, type TunerMessage } from "../../api";
import ChatComposer from "../../components/ChatComposer";
import Markdown from "../../components/Markdown";
import { invalidate, useSystem } from "../../hooks";
import { Bubble, Button, cx, ErrorNote, Spinner } from "../../ui";
import { ConfirmCard } from "./ConfirmCard";

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

// ── left: chat ───────────────────────────────────────────────────────────────

export function Chat({
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
              <>
                your guide · {sys?.agents.tuner.model ?? "…"}
                {sys?.agents.tuner.experimental && (
                  <span className="ml-1.5 rounded bg-warn-soft px-1 py-px text-warn" title="A local Ollama model runs the Tuner, but rarely drives its long, tool-heavy loop well">
                    experimental
                  </span>
                )}
              </>
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
