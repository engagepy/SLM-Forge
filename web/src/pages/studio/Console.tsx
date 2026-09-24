// The console under the canvas: agent activity and the running job's output.
import { useEffect, useState } from "react";
import { type Job, type TunerMessage } from "../../api";
import JobLog from "../../components/JobLog";
import { runProgress, useLiveJob } from "../../hooks";
import { cx } from "../../ui";

// ── console ──────────────────────────────────────────────────────────────────

export function Console({ messages, activeJob }: { messages: TunerMessage[]; activeJob?: Job }) {
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
