// Small pieces the canvas cards share.
import { type Stage } from "../../api";
import { SectionLabel, Spinner } from "../../ui";

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
