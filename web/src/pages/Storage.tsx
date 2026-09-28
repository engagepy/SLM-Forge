import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router";

import MetricsBar from "../components/MetricsBar";
import { PAGE, PageHeader } from "../components/Page";
import { api, fmt } from "../api";
import { Badge, Button, cx, ErrorNote, useToast } from "../ui";

interface Inventory {
  models: { repo_id: string; size_gb: number; origin: "downloaded" | "cache"; used_by: { id: number; name: string }[]; downloaded_at: string }[];
  projects: {
    id: number;
    name: string;
    goal: string;
    base_model: string | null;
    parts_gb: Record<string, number>;
    total_gb: number;
    exports: { job_id: number; name: string; path: string; size_gb: number }[];
  }[];
  footprint: {
    total_gb: number;
    disk_free_gb: number;
    disk_total_gb: number;
    parts_gb: Record<string, number>;
    reclaimable_gb: number;
    tidy_threshold_gb: number;
    tidy_suggested: boolean;
  };
}

const size = fmt.size;

/** Everything the app keeps on this Mac, and the buttons that give the space back. */
export default function StoragePage() {
  const qc = useQueryClient();
  const toast = useToast();
  const inv = useQuery({ queryKey: ["storage"], queryFn: () => api.get<Inventory>("/api/storage") });
  const [error, setError] = useState<unknown>(null);
  const done = (what: string) => (r: { freed_gb: number }) => {
    toast({ tone: "good", title: `${what} deleted`, body: `Freed ${size(r.freed_gb)}.` });
    for (const k of ["storage", "system", "sessions", "projects", "studio", "exports"]) qc.invalidateQueries({ queryKey: [k] });
  };
  const removeModel = useMutation({
    mutationFn: (repo: string) => api.delete<{ freed_gb: number }>(`/api/models/${repo}`),
    onSuccess: done("Model"),
    onError: setError,
  });
  const deleteExport = useMutation({
    mutationFn: (x: { pid: number; job: number }) => api.delete<{ freed_gb: number }>(`/api/projects/${x.pid}/exports/${x.job}`),
    onSuccess: done("Exported model"),
    onError: setError,
  });
  const resetProject = useMutation({
    mutationFn: (x: { pid: number; keep: number[] }) =>
      api.post<{ freed_gb: number }>(`/api/projects/${x.pid}/reset`, { keep_export_job_ids: x.keep }),
    onSuccess: done("Project history"),
    onError: setError,
  });
  const deleteProject = useMutation({
    mutationFn: (x: { pid: number; keep: boolean }) => api.delete<{ freed_gb: number }>(`/api/projects/${x.pid}?keep_exports=${x.keep}`),
    onSuccess: done("Project"),
    onError: setError,
  });
  const tidy = useMutation({
    mutationFn: () => api.post<{ freed_gb: number; deleted: number }>("/api/storage/tidy"),
    onSuccess: done("Intermediate run files"),
    onError: setError,
  });
  const busy = removeModel.isPending || deleteExport.isPending || deleteProject.isPending || resetProject.isPending || tidy.isPending;
  const d = inv.data;

  return (
    <div className="flex h-full flex-col">
      <MetricsBar />
      <div className="min-h-0 flex-1 overflow-y-auto">
      <div className={PAGE}>
        <div>
          <PageHeader title="Storage">
            What SLM Forge keeps on this Mac, and how to give the space back. Deleting is immediate and can't be undone.
          </PageHeader>
          {d && (
            <p className="mt-2 text-[13px] text-muted">
              <span className="num font-medium text-fg">{size(d.footprint.total_gb)}</span> used by the app ·{" "}
              <span className="num">{fmt.gb(d.footprint.disk_free_gb)}</span> free on this disk
            </p>
          )}
        </div>
        <ErrorNote error={error} />

        {d && d.footprint.reclaimable_gb >= 0.05 && (
          <section className={cx("rounded-xl border p-4", d.footprint.tidy_suggested ? "border-warn/50 bg-warn-soft/40" : "border-line bg-panel")}>
            <div className="flex items-center gap-3">
              <div className="min-w-0 flex-1">
                <div className="text-[13px] font-medium">
                  {size(d.footprint.reclaimable_gb)} of intermediate run files can be cleared
                  {d.footprint.tidy_suggested && ` · the app is over ${d.footprint.tidy_threshold_gb} GB`}
                </div>
                <p className="mt-1 text-xs leading-relaxed text-muted">
                  Each training run writes a fused copy of the model so the next run can start from it, and failed or cancelled runs leave
                  their folders. Clearing removes those copies only. Every run's adapter, metrics and evaluation stay, exports stay, and any
                  checkpoint can still be served or built on (it is re-fused from its adapter when needed).
                </p>
              </div>
              <Button variant="primary" size="sm" loading={tidy.isPending} disabled={busy} onClick={() => tidy.mutate()}>
                Clear {size(d.footprint.reclaimable_gb)}
              </Button>
            </div>
          </section>
        )}

        <section>
          <h2 className="mb-2 text-[11px] font-medium uppercase tracking-wide text-faint">Base models downloaded from Hugging Face</h2>
          <ul className="space-y-2">
            {d?.models.map((m) => (
              <li key={m.repo_id} className="flex items-center gap-3 rounded-xl border border-line bg-panel p-3">
                <div className="min-w-0 flex-1">
                  <div className="truncate font-mono text-[13px]">{m.repo_id}</div>
                  <div className="mt-0.5 text-xs text-muted">
                    {m.origin === "downloaded" ? "downloaded by SLM Forge" : "was already on this Mac"} ·{" "}
                    {m.used_by.length ? `used by ${m.used_by.map((p) => p.name).join(", ")}` : "no project uses it"}
                  </div>
                </div>
                <span className="num text-[13px]">{size(m.size_gb)}</span>
                <Button
                  size="sm"
                  variant="danger"
                  disabled={busy || m.used_by.length > 0}
                  title={m.used_by.length ? "Delete the projects that use it first" : "Delete the files from the Hugging Face cache"}
                  onClick={() => confirm(`Delete ${m.repo_id} (${size(m.size_gb)}) from this Mac?`) && removeModel.mutate(m.repo_id)}
                >
                  Remove
                </Button>
              </li>
            ))}
            {d && !d.models.length && <li className="text-xs text-faint">No base models downloaded.</li>}
          </ul>
        </section>

        <section>
          <h2 className="mb-2 text-[11px] font-medium uppercase tracking-wide text-faint">Projects</h2>
          <ul className="space-y-2">
            {d?.projects.map((p) => (
              <li key={p.id} className="rounded-xl border border-line bg-panel p-3">
                <div className="flex items-center gap-3">
                  <div className="min-w-0 flex-1">
                    <Link to={`/p/${p.id}`} className="block truncate text-[13px] font-medium hover:underline">
                      {p.name}
                    </Link>
                    <div className="truncate text-xs text-muted">{p.goal}</div>
                    <div className="mt-1 flex flex-wrap gap-1.5 text-[11px] text-faint">
                      {(["runs", "datasets", "uploads", "exports"] as const).map((k) => (
                        <Badge key={k}>
                          {k} {size(p.parts_gb[k] ?? 0)}
                        </Badge>
                      ))}
                      {p.base_model && <Badge tone="info">base {p.base_model.split("/").pop()}</Badge>}
                    </div>
                  </div>
                  <span className="num text-[13px]">{size(p.total_gb)}</span>
                  <div className="flex flex-col gap-1">
                    <Button
                      size="sm"
                      disabled={busy}
                      title="Delete its runs, data, chat and memory; keep the project, its brief and its exported models"
                      onClick={() =>
                        confirm(`Reset "${p.name}"? Its runs, data, chat and memory are deleted; the project, its brief and its exported models stay.`) &&
                        resetProject.mutate({ pid: p.id, keep: p.exports.map((e) => e.job_id) })
                      }
                    >
                      Reset, keep exports
                    </Button>
                    <Button
                      size="sm"
                      variant="danger"
                      disabled={busy}
                      onClick={() => {
                        const keep = p.exports.length > 0 && confirm(`Keep the ${p.exports.length} exported model(s) of "${p.name}"? OK keeps them, Cancel deletes them too.`);
                        if (confirm(`Delete the project "${p.name}", its runs, data and chat${keep ? "" : p.exports.length ? " and its exported models" : ""}?`))
                          deleteProject.mutate({ pid: p.id, keep });
                      }}
                    >
                      Delete project
                    </Button>
                  </div>
                </div>
                {p.exports.length > 0 && (
                  <ul className="mt-2 space-y-1 border-t border-line pt-2">
                    {p.exports.map((e) => (
                      <li key={e.job_id} className={cx("flex items-center gap-3 text-xs")}>
                        <span className="text-faint">exported model</span>
                        <span className="min-w-0 flex-1 truncate font-mono">{e.name}</span>
                        <span className="num">{size(e.size_gb)}</span>
                        <Link to={`/p/${p.id}/try?export=${e.job_id}`} className="font-medium text-good hover:underline">
                          ▶ Try it
                        </Link>
                        <button
                          className="text-bad hover:underline disabled:opacity-50"
                          disabled={busy}
                          onClick={() => confirm(`Delete the exported model "${e.name}" (${size(e.size_gb)})?`) && deleteExport.mutate({ pid: p.id, job: e.job_id })}
                        >
                          delete
                        </button>
                      </li>
                    ))}
                  </ul>
                )}
              </li>
            ))}
            {d && !d.projects.length && <li className="text-xs text-faint">No projects.</li>}
          </ul>
        </section>
      </div>
      </div>
    </div>
  );
}
