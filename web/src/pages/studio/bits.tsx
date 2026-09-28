// Small pieces the canvas cards share.
import { type ExportInfo, type Stage } from "../../api";
import { Badge, SectionLabel, Spinner } from "../../ui";

export const STAGE_LABEL: Record<Stage, string> = {
  goal: "Goal",
  model: "Model",
  data: "Data",
  train: "Train",
  evaluate: "Evaluate",
  refine: "Refine",
  export: "Export",
};

export function Empty({ children }: { children: React.ReactNode }) {
  return <p className="text-xs text-faint">{children}</p>;
}

export function Fact({ label, value }: { label: string; value: string }) {
  return (
    <div className="rounded-md bg-panel-2/60 px-2 py-1.5">
      <SectionLabel className="text-[10px]">{label}</SectionLabel>
      <div className="num text-[13px]">{value}</div>
    </div>
  );
}

export function Progress({ label }: { label: string }) {
  return (
    <div className="flex items-center gap-2 text-xs text-info">
      <Spinner className="size-3" /> {label}…
      <div className="h-1 flex-1 overflow-hidden rounded-full bg-panel-2">
        <div className="h-full w-1/3 animate-[pulse_1.2s_ease-in-out_infinite] rounded-full bg-info" />
      </div>
    </div>
  );
}

/** On a model kept through a project reset: why its stages aren't ticked, and what it was built from. */
export function BeforeResetBadge({ e }: { e: Pick<ExportInfo, "trained_before_reset"> }) {
  const runs = e.trained_before_reset;
  if (!runs?.length) return null;
  const built = runs
    .map((r) => `${r.kind.toUpperCase()} job ${r.job_id}${r.metrics?.val_loss != null ? ` (val loss ${r.metrics.val_loss})` : ""}`)
    .join(", then ");
  return (
    <span title={`Built from ${built}. The project was reset after, so those runs are no longer in its history and its stages aren't ticked for this model.`}>
      <Badge>trained before a reset</Badge>
    </span>
  );
}
