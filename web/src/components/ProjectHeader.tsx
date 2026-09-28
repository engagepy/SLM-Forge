// The project's name and goal under the metrics bar: the same line on the Studio and Advanced.
import { type Snapshot } from "../api";

export default function ProjectHeader({ snapshot, children }: { snapshot?: Snapshot; children?: React.ReactNode }) {
  return (
    <header className="flex h-11 shrink-0 items-center gap-3 border-b border-line px-4">
      <div className="min-w-0 flex-1">
        <div className="truncate text-[13px] font-semibold">{snapshot?.project.name ?? "…"}</div>
        <div className="truncate text-[11px] text-faint" title={snapshot?.project.goal}>
          {snapshot?.project.goal}
        </div>
      </div>
      {children}
    </header>
  );
}
