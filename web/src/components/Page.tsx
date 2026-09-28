// The frame every full page shares: one width, one heading, and the way on always at the top right.
import { type ReactNode } from "react";

import { type Stage, STAGES } from "../api";
import { useProjectId, useStudio } from "../hooks";
import { STAGE_LABEL } from "../pages/studio/bits";
import { LinkButton } from "../ui";
import { ADVANCED_PAGE, stageDone, stageSkipped } from "./StageStepper";

/** The page container: the same width and spacing on every Advanced page, Home and Storage. */
export const PAGE = "mx-auto w-full max-w-6xl space-y-5 p-6";

/** A page's title, one line on what it's for, and its actions (the next step last, top right). */
export function PageHeader({ title, children, actions }: { title: ReactNode; children?: ReactNode; actions?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-x-4 gap-y-2">
      <div className="min-w-0 flex-1">
        <h1 className="text-xl font-semibold tracking-tight">{title}</h1>
        {children && <p className="mt-1 max-w-3xl text-[13px] leading-relaxed text-muted">{children}</p>}
      </div>
      {actions && <div className="flex shrink-0 flex-wrap items-center gap-2 whitespace-nowrap">{actions}</div>}
    </div>
  );
}

/** The way on from a stage's page, in the same place on every one: the following stage (primary
 * once this one is done), and on Export, Try it. */
export function NextStage({ stage }: { stage: Stage }) {
  const projectId = useProjectId();
  const { data: s } = useStudio(projectId);
  if (!s) return null;
  const done = stageDone(s);
  const next = STAGES[STAGES.indexOf(stage) + 1];
  if (!next)
    return s.exports.length ? (
      <LinkButton to={`/p/${projectId}/try`} variant="good">
        ▶ Try it
      </LinkButton>
    ) : null;
  const finished = done[stage] || stageSkipped(stage, done, s);
  return (
    <LinkButton to={`/p/${projectId}/${ADVANCED_PAGE[next]}`} variant={finished ? "primary" : "secondary"}>
      Next: {STAGE_LABEL[next]} →
    </LinkButton>
  );
}
