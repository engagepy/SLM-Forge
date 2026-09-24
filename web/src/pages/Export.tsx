import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router";

import { api, DEFAULT_SAMPLING, fmt } from "../api";
import { useOverview, useProjectId, waitForJob } from "../hooks";
import { Badge, Button, Card, Empty, ErrorNote, Field, Input, Mono, Select } from "../ui";

interface ExportRow {
  job_id: number;
  path: string;
  size_gb: number;
  min_ram_gb: number;
  created_at: string;
}

export default function ExportPage() {
  const projectId = useProjectId();
  const qc = useQueryClient();
  const { data: ov } = useOverview(projectId);
  const exports = useQuery({ queryKey: ["exports", projectId], queryFn: () => api.get<ExportRow[]>(`/api/projects/${projectId}/exports`) });
  const [name, setName] = useState("");
  const [bits, setBits] = useState("none");
  const [status, setStatus] = useState<string | null>(null);

  const run = useMutation({
    mutationFn: async () => {
      const { job_id } = await api.post<{ job_id: number }>(`/api/projects/${projectId}/export`, {
        name,
        quantize_bits: bits === "none" ? null : Number(bits),
        sampling: { temperature: DEFAULT_SAMPLING.temperature, top_p: DEFAULT_SAMPLING.top_p, repetition_penalty: DEFAULT_SAMPLING.repetition_penalty },
      });
      const job = await waitForJob(job_id, (j) => setStatus(j.status));
      if (job.status !== "succeeded") throw new Error(job.error || `Export ${job.status}`);
    },
    onSettled: () => {
      setStatus(null);
      qc.invalidateQueries({ queryKey: ["exports", projectId] });
    },
  });

  const activate = useMutation({
    mutationFn: (id: number) => api.post(`/api/projects/${projectId}/checkpoints/${id}/activate`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["overview", projectId] }),
  });

  if (!ov) return null;
  const { project, checkpoints } = ov;
  const baseQuantized = /(\d)bit/i.test(project.base_model ?? "");

  return (
    <div className="mx-auto max-w-5xl space-y-5 p-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Export</h1>
        <p className="mt-1 text-[13px] text-muted">
          Fuse the adapters into one standalone MLX model with a model card and a minimum-RAM rating. It runs with{" "}
          <Mono>mlx_lm.generate</Mono> on this Mac and any Apple Silicon Mac with at least as much memory.
        </p>
      </div>

      <Card title="Checkpoints" subtitle="Roll back by serving an earlier checkpoint; exports use whatever is being served" pad={false}>
        {checkpoints.length ? (
          <ul className="divide-y divide-line">
            {checkpoints.map((c) => {
              const serving = project.current_adapter_path === c.adapter_path || (!!c.fused_path && project.current_model_path === c.fused_path);
              return (
                <li key={c.id} className="flex items-center gap-3 px-4 py-2.5 text-[13px]">
                  <Badge tone={c.kind === "dpo" ? "accent" : "info"}>{c.kind.toUpperCase()}</Badge>
                  <Link to={`/p/${projectId}/train/${c.job_id}`} className="hover:text-accent">
                    job {c.job_id}
                  </Link>
                  <span className="num flex-1 truncate text-xs text-muted">
                    {Object.entries(c.metrics)
                      .map(([k, v]) => `${k.replace(/_/g, " ")} ${v}`)
                      .join(" · ")}
                  </span>
                  <span className="text-[11px] text-faint">{fmt.ago(c.created_at)}</span>
                  {serving ? (
                    <Badge tone="good">serving</Badge>
                  ) : (
                    <Button size="sm" variant="ghost" loading={activate.isPending} onClick={() => activate.mutate(c.id)}>
                      Serve this
                    </Button>
                  )}
                </li>
              );
            })}
          </ul>
        ) : (
          <div className="p-4">
            <Empty title="Nothing trained yet">You can still export the base model, but you probably want to train first.</Empty>
          </div>
        )}
      </Card>

      <Card title="New export">
        <div className="grid gap-3 md:grid-cols-3">
          <Field label="Name" hint="folder name">
            <Input value={name} onChange={(e) => setName(e.target.value)} placeholder={`${project.name}-v${checkpoints.length}`} />
          </Field>
          <Field label="Quantize" hint={baseQuantized ? "base is already quantized" : "smaller and faster"}>
            <Select
              value={bits}
              onChange={setBits}
              options={[
                { value: "none", label: baseQuantized ? "Keep as trained" : "No (16-bit)" },
                { value: "8", label: "8-bit" },
                { value: "6", label: "6-bit" },
                { value: "4", label: "4-bit" },
              ]}
            />
          </Field>
          <div className="flex items-end">
            <Button variant="primary" className="w-full" loading={run.isPending} onClick={() => run.mutate()}>
              {run.isPending ? `Exporting (${status ?? "queued"})…` : "Fuse and export"}
            </Button>
          </div>
        </div>
        <ErrorNote error={run.error} />
      </Card>

      <Card title="Exports" pad={false}>
        {exports.data?.length ? (
          <ul className="divide-y divide-line">
            {exports.data.map((e) => (
              <li key={e.job_id} className="space-y-1.5 px-4 py-3">
                <div className="flex items-center gap-2 text-[13px]">
                  <span className="font-medium">{e.path.split("/").pop()}</span>
                  <Badge>{fmt.gb(e.size_gb)}</Badge>
                  <Badge tone="good">runs on {e.min_ram_gb} GB+ Macs</Badge>
                  <span className="ml-auto text-[11px] text-faint">{fmt.ago(e.created_at)}</span>
                </div>
                <div className="font-mono text-[11px] text-muted">{e.path}</div>
                <pre className="rounded-md bg-bg px-3 py-2 font-mono text-[11.5px] text-muted">
                  mlx_lm.generate --model {e.path} --prompt "Hello"
                </pre>
              </li>
            ))}
          </ul>
        ) : (
          <p className="p-4 text-xs text-muted">No exports yet.</p>
        )}
      </Card>
    </div>
  );
}
