// The pipeline stepper the Studio canvas and the Advanced screens share, so both always agree on
// the stages, their order and which ones are done (a green tick).
import { Link } from "react-router";

import { type Snapshot, type Stage, STAGES } from "../api";
import { STAGE_LABEL } from "../pages/studio/bits";
import { cx } from "../ui";

/** The one definition of "this stage is finished", from the Studio snapshot. */
export function stageDone(s: Snapshot): Record<Stage, boolean> {
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

/** Refine (DPO) is optional: a project that finished without it skipped it; it isn't pending. */
export const stageSkipped = (st: Stage, done: Record<Stage, boolean>, s: Snapshot) => st === "refine" && !done.refine && done.export && s.completed;

/** Where each stage lives in the Advanced screens. */
export const ADVANCED_PAGE: Record<Stage, string> = {
  goal: "overview",
  data: "data",
  model: "model",
  train: "train",
  evaluate: "evaluate",
  refine: "feedback",
  export: "export",
};

export function StageStepper({
  s,
  current,
  onPick,
  linkTo,
}: {
  s: Snapshot;
  /** The highlighted stage: where the Tuner is (Studio) or the page being viewed (Advanced). */
  current: Stage | null;
  onPick?: (st: Stage) => void;
  linkTo?: (st: Stage) => string;
}) {
  const done = stageDone(s);
  return (
    <ol className="flex items-center gap-1 overflow-x-auto">
      {STAGES.map((st, i) => {
        const skipped = stageSkipped(st, done, s);
        const here = st === current;
        const pill = cx(
          "flex shrink-0 items-center gap-1.5 rounded-full px-2 py-1 text-[11px] font-medium whitespace-nowrap transition",
          // A finished stage keeps its tick even when it's the current one (e.g. Export at the end).
          here && !done[st] ? "bg-accent text-white" : done[st] ? "text-good" : "text-faint hover:text-fg",
          here && done[st] && "bg-good-soft",
        );
        const inner = (
          <>
            <span
              className={cx(
                "grid size-4 place-items-center rounded-full text-[9px]",
                done[st] ? "bg-good text-white" : here ? "bg-white/25" : "bg-panel-2",
              )}
            >
              {done[st] ? "✓" : skipped ? "–" : i + 1}
            </span>
            {STAGE_LABEL[st]}
          </>
        );
        const title = done[st] ? `${STAGE_LABEL[st]}: done` : skipped ? `${STAGE_LABEL[st]}: skipped (optional)` : STAGE_LABEL[st];
        return (
          <li key={st} className="flex flex-1 items-center gap-1">
            {linkTo ? (
              <Link to={linkTo(st)} className={pill} title={title} aria-current={here ? "page" : undefined}>
                {inner}
              </Link>
            ) : (
              <button onClick={() => onPick?.(st)} className={pill} title={title}>
                {inner}
              </button>
            )}
            {i < STAGES.length - 1 && <span className={cx("h-px flex-1", done[st] ? "bg-good/50" : "bg-line")} />}
          </li>
        );
      })}
    </ol>
  );
}
