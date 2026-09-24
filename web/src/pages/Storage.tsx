import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router";

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
  footprint: { total_gb: number; disk_free_gb: number; disk_total_gb: number; parts_gb: Record<string, number> };
}

const size = (gb: number) => (gb < 1 ? `${Math.round(gb * 1024)} MB` : fmt.gb(gb));

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
  const deleteProject = useMutation({
    mutationFn: (x: { pid: number; keep: boolean }) => api.delete<{ freed_gb: number }>(`/api/projects/${x.pid}?keep_exports=${x.keep}`),
    onSuccess: done("Project"),
    onError: setError,
  });
  const busy = removeModel.isPending || deleteExport.isPending || deleteProject.isPending;
  const d = inv.data;

  return (
    <div className="h-full overflow-y-auto">
      <div className="mx-auto max-w-3xl space-y-8 px-6 py-10">
        <div>
          <h1 className="text-2xl font-semibold tracking-tight">Storage</h1>
          <p className="mt-2 text-[14px] leading-relaxed text-muted">
            What SLM Forge keeps on this Mac, and how to give the space back. Deleting is immediate and can't be undone.
          </p>
          {d && (
            <p className="mt-2 text-[13px] text-muted">
              <span className="num font-medium text-fg">{size(d.footprint.total_gb)}</span> used by the app ·{" "}
              <span className="num">{fmt.gb(d.footprint.disk_free_gb)}</span> free on this disk
            </p>
          )}
        </div>
        <ErrorNote error={error} />

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
                    <Link to={`/p/${p.id}`} className="truncate text-[13px] font-medium hover:underline">
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
                        <Link to={`/p/${p.id}/try?export=${e.job_id}`} className="text-accent hover:underline">
                          try
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
  );
}
