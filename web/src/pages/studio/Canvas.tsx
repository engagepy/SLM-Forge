// The right half: the stage stepper, one card per stage, the done banner and the console.
import { useMutation } from "@tanstack/react-query";
import { useEffect } from "react";
import { api, isActive, type Snapshot, type Stage, STAGES, type TunerMessage } from "../../api";
import { Badge, Button, cx, LinkButton } from "../../ui";
import { Console } from "./Console";
import { EvaluateView } from "./Evaluate";
import { STAGE_LABEL } from "./bits";
import { DataView, ExportView, GoalView, ModelView, RefineView, TrainView } from "./stages";

const STAGE_VIEW: Record<Stage, React.FC<{ s: Snapshot }>> = {
  goal: GoalView,
  model: ModelView,
  data: DataView,
  train: TrainView,
  evaluate: EvaluateView,
  refine: RefineView,
  export: ExportView,
};

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

export function Canvas({ snapshot: s, messages }: { snapshot: Snapshot; messages: TunerMessage[] }) {
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
