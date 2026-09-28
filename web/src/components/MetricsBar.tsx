import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router";

import { api, fmt, type Snapshot, type SystemStatus } from "../api";
import { useSystem } from "../hooks";
import { ADVANCED_PAGE } from "./StageStepper";
import { Badge, cx, Popover } from "../ui";
import UsageMeter from "./UsageMeter";

const size = fmt.size;

const PART_LABEL: Record<string, string> = {
  models: "downloaded models",
  datasets: "datasets",
  runs: "training runs",
  exports: "exported models",
  uploads: "uploads",
  database: "databases",
};

/** What the app occupies on this Mac's disk, with a breakdown on click. */
function DiskMeter({ disk }: { disk: SystemStatus["disk"] }) {
  const [open, setOpen] = useState(false);
  const parts = Object.entries(disk.parts_gb).sort((a, b) => b[1] - a[1]);
  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        aria-expanded={open}
        title="Disk used by SLM Forge on this Mac"
        className="flex items-center gap-1.5 rounded-full border border-line px-2.5 py-1 text-xs text-muted hover:text-fg"
      >
        <span className="text-faint">Disk</span>
        <span className="num font-medium text-fg">{size(disk.total_gb)}</span>
        <span className="num text-faint">· {fmt.gb(disk.disk_free_gb)} free</span>
        {disk.tidy_suggested && <span className="size-1.5 rounded-full bg-warn" title={`${size(disk.reclaimable_gb)} can be cleared`} />}
      </button>
      <Popover open={open} onClose={() => setOpen(false)} align="left">
          <div className="flex items-center justify-between">
            <span className="font-semibold">Used by SLM Forge</span>
            <span className="num text-[15px] font-semibold">{size(disk.total_gb)}</span>
          </div>
          <ul className="mt-2 space-y-1 border-t border-line pt-2">
            {parts.map(([k, v]) => (
              <li key={k} className="flex items-center gap-2">
                <span className="w-32 text-muted">{PART_LABEL[k] ?? k}</span>
                <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-panel-2">
                  <div className="h-full bg-accent" style={{ width: `${disk.total_gb ? Math.max(2, (v / disk.total_gb) * 100) : 0}%` }} />
                </div>
                <span className="num w-16 text-right">{size(v)}</span>
              </li>
            ))}
          </ul>
          <p className="mt-2 border-t border-line pt-2 text-[11px] leading-relaxed break-all whitespace-normal text-faint">
            {fmt.gb(disk.disk_free_gb)} free of {fmt.gb(disk.disk_total_gb)} on this disk. Workspace: {disk.workspace}
          </p>
          {disk.reclaimable_gb >= 0.05 && (
            <p className={cx("mt-2 text-[11px] leading-relaxed", disk.tidy_suggested ? "text-warn" : "text-faint")}>
              {size(disk.reclaimable_gb)} of intermediate run files can be cleared (fused copies earlier runs left behind; adapters,
              exports and rollback stay).
            </p>
          )}
          <Link to="/storage" className="mt-2 block text-[12px] font-medium text-accent hover:underline" onClick={() => setOpen(false)}>
            Manage storage →
          </Link>
      </Popover>
    </div>
  );
}

export type View = "studio" | "advanced" | "try";

/** Every screen's top strip. On a project it also carries the Autopilot pill and the view switch,
 * always at the far right; on Home and Storage it shows the machine only. */
export default function MetricsBar({ snapshot, view }: { snapshot?: Snapshot; view?: View }) {
  const { data: sys } = useSystem();
  const qc = useQueryClient();
  const gpu = sys?.worker.running.gpu;
  const toggle = useMutation({
    mutationFn: (on: boolean) => api.post(`/api/projects/${snapshot!.project.id}/studio/autopilot`, { on }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["studio", snapshot!.project.id] }),
  });
  return (
    <div className="flex min-h-10 shrink-0 flex-wrap items-center gap-x-2.5 gap-y-1 border-b border-line bg-panel px-4 py-1 text-xs whitespace-nowrap text-muted">
      {sys && (
        <span className="hidden lg:inline">
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
      {sys?.disk && <DiskMeter disk={sys.disk} />}
      <UsageMeter />
      {sys && !sys.agents.tuner.ready && <Badge tone="warn">Tuner needs {sys.agents.tuner.key_env} in .env</Badge>}
      {sys && sys.agents.tuner.ready && !sys.agents.key_configured && (
        <Badge tone="warn">set {sys.agents.key_env} in .env</Badge>
      )}
      <span className="ml-auto" />
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
      {view && (snapshot ? <ViewSwitch s={snapshot} view={view} /> : <span className="h-7 w-52" aria-hidden />)}
    </div>
  );
}

/** Studio | Advanced | Try it: three views of one project, one control, in the same place on every
 * project screen. Try it stays in place, greyed out, until there is a finished model. */
function ViewSwitch({ s, view }: { s: Snapshot; view: View }) {
  const id = s.project.id;
  const tab = (on: boolean) => cx("rounded px-2 py-0.5 font-medium transition", on ? "bg-panel-2 text-fg" : "text-faint hover:text-fg");
  const here = (v: View) => (view === v ? "page" : undefined);
  return (
    <nav className="flex rounded-md border border-line p-0.5" aria-label="View">
      <Link to={`/p/${id}`} className={tab(view === "studio")} aria-current={here("studio")}>
        Studio
      </Link>
      <Link
        to={`/p/${id}/${ADVANCED_PAGE[s.stage] ?? "overview"}`}
        className={tab(view === "advanced")}
        aria-current={here("advanced")}
        title="Every setting and number, for ML experts"
      >
        Advanced
      </Link>
      {s.exports.length ? (
        <Link
          to={`/p/${id}/try`}
          className={cx("rounded px-2 py-0.5 font-medium transition", view === "try" ? "bg-good-soft text-good" : "text-good/80 hover:text-good")}
          aria-current={here("try")}
          title="Chat with your finished model"
        >
          ▶ Try it
        </Link>
      ) : (
        <span className="cursor-not-allowed rounded px-2 py-0.5 font-medium text-faint/50" title="No finished model yet: the Tuner exports one at the end" aria-disabled="true">
          ▶ Try it
        </span>
      )}
    </nav>
  );
}
