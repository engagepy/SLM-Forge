import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router";

import { api, fmt, type Dataset, type DatasetVersion, type Mapping, type Proposal } from "../api";
import ProposalCard from "../components/ProposalCard";
import { useProjectId, waitForJob } from "../hooks";
import { Badge, Button, Card, Collapsible, cx, Empty, ErrorNote, Field, Input, NumberField, Select, Spinner, Toggle } from "../ui";

export default function DataPage() {
  const projectId = useProjectId();
  const data = useQuery({
    queryKey: ["datasets", projectId],
    queryFn: () => api.get<{ datasets: Dataset[]; versions: DatasetVersion[] }>(`/api/projects/${projectId}/datasets`),
  });
  const [selected, setSelected] = useState<number | null>(null);
  const datasets = data.data?.datasets ?? [];
  const current = datasets.find((d) => d.id === selected) ?? datasets[0];

  return (
    <div className="mx-auto max-w-6xl space-y-5 p-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Training data</h1>
        <p className="mt-1 text-[13px] text-muted">
          Let the scout find public datasets, browse the Hub yourself, or upload your own. Then map columns to training records,
          clean them and split them.
        </p>
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <ScoutPanel projectId={projectId} />
        <div className="space-y-5">
          <UploadPanel projectId={projectId} onUploaded={(id) => setSelected(id)} />
          <HubSearch projectId={projectId} />
        </div>
      </div>

      <Card title="Raw datasets" subtitle="Pick one to map and prepare" pad={false}>
        {datasets.length ? (
          <div className="grid lg:grid-cols-[260px_1fr]">
            <ul className="border-b border-line lg:border-r lg:border-b-0">
              {datasets.map((d) => (
                <li key={d.id}>
                  <button
                    onClick={() => setSelected(d.id)}
                    className={cx("w-full px-4 py-2.5 text-left hover:bg-panel-2", current?.id === d.id && "bg-accent-soft")}
                  >
                    <div className="truncate text-[13px] font-medium">{d.name}</div>
                    <div className="text-[11px] text-faint">
                      {d.source} · {fmt.compact(d.n_rows)} rows{d.license && ` · ${d.license}`}
                    </div>
                  </button>
                </li>
              ))}
            </ul>
            {current && <MappingEditor key={current.id} dataset={current} projectId={projectId} />}
          </div>
        ) : (
          <div className="p-4">
            <Empty title="No datasets yet">Run the scout, import from the Hub, or upload a file above.</Empty>
          </div>
        )}
      </Card>

      <Versions versions={data.data?.versions ?? []} projectId={projectId} />
    </div>
  );
}

function ScoutPanel({ projectId }: { projectId: number }) {
  const [request, setRequest] = useState("");
  const [jobId, setJobId] = useState<number | null>(null);
  const [jobStatus, setJobStatus] = useState<string | null>(null);
  const proposals = useQuery({
    queryKey: ["proposals", projectId],
    queryFn: () => api.get<Proposal[]>(`/api/projects/${projectId}/proposals`),
  });
  const run = useMutation({
    mutationFn: async () => {
      const { job_id } = await api.post<{ job_id: number }>(`/api/projects/${projectId}/agents/scout`, { request });
      setJobId(job_id);
      const job = await waitForJob(job_id, (j) => setJobStatus(j.status));
      if (job.status !== "succeeded") throw new Error(job.error || `Scout ${job.status}`);
      return job;
    },
  });
  const dataProposals = (proposals.data ?? []).filter(
    (p) => ["import_dataset", "acquire_manually", "prepare_dataset"].includes(p.action) && ["pending", "approved"].includes(p.status),
  );

  return (
    <Card
      title="DataScout agent"
      subtitle="Searches the Hub, reads dataset cards, previews rows, then proposes imports for you to approve."
      actions={
        <Link to={`/p/${projectId}/agents`} className="text-xs text-info hover:underline">
          activity →
        </Link>
      }
    >
      <div className="flex gap-2">
        <Input value={request} onChange={(e) => setRequest(e.target.value)} placeholder="Optional: “prefer recipe Q&A, English only”" />
        <Button variant="primary" loading={run.isPending} onClick={() => run.mutate()}>
          Scout
        </Button>
      </div>
      {run.isPending && (
        <p className="mt-2 flex items-center gap-2 text-xs text-muted">
          <Spinner /> Job {jobId} {jobStatus ?? "queued"}… this takes a minute or two.
        </p>
      )}
      {run.data && <p className="mt-2 text-xs text-good">{String(run.data.result.summary ?? "Done.")}</p>}
      <ErrorNote error={run.error} />
      <div className="mt-4 space-y-3">
        {dataProposals.map((p) => (
          <ProposalCard key={p.id} p={p} projectId={projectId} />
        ))}
        {!dataProposals.length && !run.isPending && <p className="text-xs text-faint">No open data proposals.</p>}
      </div>
    </Card>
  );
}

function UploadPanel({ projectId, onUploaded }: { projectId: number; onUploaded: (id: number) => void }) {
  const qc = useQueryClient();
  const ref = useRef<HTMLInputElement>(null);
  const [drag, setDrag] = useState(false);
  const upload = useMutation({
    mutationFn: (file: File) => {
      const form = new FormData();
      form.append("file", file);
      return api.upload<Dataset>(`/api/projects/${projectId}/datasets/upload`, form);
    },
    onSuccess: (d) => {
      qc.invalidateQueries({ queryKey: ["datasets", projectId] });
      onUploaded(d.id);
    },
  });
  return (
    <Card title="Upload your own">
      <button
        type="button"
        onClick={() => ref.current?.click()}
        onDragOver={(e) => (e.preventDefault(), setDrag(true))}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          if (e.dataTransfer.files[0]) upload.mutate(e.dataTransfer.files[0]);
        }}
        className={cx(
          "flex w-full flex-col items-center rounded-lg border border-dashed px-4 py-6 text-center transition",
          drag ? "border-accent bg-accent-soft" : "border-line-strong hover:border-accent",
        )}
      >
        {upload.isPending ? <Spinner /> : <span className="text-[13px]">Drop a file or click to choose</span>}
        <span className="mt-1 text-[11px] text-faint">.jsonl · .json · .csv · .parquet · .txt/.md (blank-line separated passages)</span>
      </button>
      <input ref={ref} type="file" className="hidden" accept=".jsonl,.json,.csv,.parquet,.txt,.md" onChange={(e) => e.target.files?.[0] && upload.mutate(e.target.files[0])} />
      <ErrorNote error={upload.error} />
    </Card>
  );
}

interface HubRow {
  id: string;
  downloads: number;
  license: string;
  gated: boolean;
  size: string;
  tasks: string[];
  description: string;
}

function HubSearch({ projectId }: { projectId: number }) {
  const qc = useQueryClient();
  const [q, setQ] = useState("");
  const [submitted, setSubmitted] = useState("");
  const [preview, setPreview] = useState<string | null>(null);
  const [maxRows, setMaxRows] = useState(3000);
  const results = useQuery({
    queryKey: ["hub-datasets", submitted],
    queryFn: () => api.get<HubRow[]>(`/api/datasets/search?q=${encodeURIComponent(submitted)}`),
    enabled: !!submitted,
  });
  const prev = useQuery({
    queryKey: ["hub-preview", preview],
    queryFn: () => api.get<{ columns: string[]; rows: Record<string, unknown>[]; split: string; config: string }>(
      `/api/datasets/preview?repo_id=${encodeURIComponent(preview!)}`,
    ),
    enabled: !!preview,
  });
  const [importStatus, setImportStatus] = useState<string | null>(null);
  const importDs = useMutation({
    mutationFn: async (row: HubRow) => {
      const { job_id } = await api.post<{ job_id: number }>(`/api/projects/${projectId}/datasets/import`, {
        repo_id: row.id,
        config: prev.data?.config,
        split: prev.data?.split ?? "train",
        max_rows: maxRows,
        license: row.license,
      });
      const job = await waitForJob(job_id, (j) => setImportStatus(j.status));
      if (job.status !== "succeeded") throw new Error(job.error || `Import ${job.status}`);
    },
    onSettled: () => {
      setImportStatus(null);
      qc.invalidateQueries({ queryKey: ["datasets", projectId] });
    },
  });

  return (
    <Card title="Browse Hugging Face datasets">
      <form className="flex gap-2" onSubmit={(e) => (e.preventDefault(), setSubmitted(q))}>
        <Input value={q} onChange={(e) => setQ(e.target.value)} placeholder="e.g. cooking instructions" />
        <Button type="submit">Search</Button>
      </form>
      {results.isFetching && <Spinner className="mt-3" />}
      <ErrorNote error={results.error} />
      <ul className="mt-3 max-h-80 divide-y divide-line overflow-y-auto">
        {results.data?.map((r) => (
          <li key={r.id} className="py-2">
            <div className="flex items-center gap-2">
              <button className="min-w-0 flex-1 truncate text-left font-mono text-[12px] hover:text-accent" onClick={() => setPreview(preview === r.id ? null : r.id)}>
                {r.id}
              </button>
              <Badge tone={/mit|apache|cc-by-4|cc0|bsd/i.test(r.license) ? "good" : r.license === "unknown" ? "warn" : "neutral"}>{r.license}</Badge>
              {r.gated && <Badge tone="warn">gated</Badge>}
            </div>
            <div className="text-[11px] text-faint">
              {fmt.compact(r.downloads)} downloads {r.size && `· ${r.size}`}
            </div>
            {preview === r.id && (
              <div className="mt-2 space-y-2">
                {prev.isLoading && <Spinner />}
                <ErrorNote error={prev.error} />
                {prev.data && (
                  <>
                    <div className="text-[11px] text-muted">
                      columns: <span className="font-mono">{prev.data.columns.join(", ")}</span>
                    </div>
                    <pre className="max-h-40 overflow-auto rounded-md bg-bg p-2 font-mono text-[11px] text-muted">
                      {JSON.stringify(prev.data.rows.slice(0, 2), null, 2)}
                    </pre>
                    <div className="flex items-end gap-2">
                      <div className="w-32">
                        <NumberField label="Rows to import" value={maxRows} onChange={(v) => v && setMaxRows(v)} step={500} min={10} />
                      </div>
                      <Button size="sm" variant="primary" loading={importDs.isPending} onClick={() => importDs.mutate(r)}>
                        {importDs.isPending ? `Importing (${importStatus ?? "queued"})` : "Import"}
                      </Button>
                    </div>
                    <ErrorNote error={importDs.error} />
                  </>
                )}
              </div>
            )}
          </li>
        ))}
      </ul>
    </Card>
  );
}

const FORMAT_FIELDS: Record<string, { key: string; label: string; required?: boolean }[]> = {
  chat: [{ key: "messages", label: "Messages column", required: true }],
  instruction: [
    { key: "prompt", label: "Prompt", required: true },
    { key: "input", label: "Extra input (appended to prompt)" },
    { key: "response", label: "Response", required: true },
  ],
  text: [{ key: "text", label: "Text column", required: true }],
  preference: [
    { key: "prompt", label: "Prompt", required: true },
    { key: "chosen", label: "Chosen (better)", required: true },
    { key: "rejected", label: "Rejected (worse)", required: true },
  ],
};

function MappingEditor({ dataset, projectId }: { dataset: Dataset; projectId: number }) {
  const qc = useQueryClient();
  const initial = dataset.suggested_mapping.format === "unknown" ? ({ format: "instruction" } as Mapping) : dataset.suggested_mapping;
  const [mapping, setMapping] = useState<Mapping>(initial);
  const [system, setSystem] = useState("");
  const [rules, setRules] = useState({ min_chars: 2, max_chars: 20000, dedupe: true });
  const [maxSeq, setMaxSeq] = useState(1024);
  const [includeFeedback, setIncludeFeedback] = useState(false);
  const full: Mapping = system ? ({ ...mapping, system: `=${system}` } as Mapping) : mapping;

  const preview = useMutation({
    mutationFn: () =>
      api.post<{ records: unknown[]; failed: number; sampled: number; errors: string[] }>(
        `/api/projects/${projectId}/datasets/${dataset.id}/mapping-preview`,
        { mapping: full },
      ),
  });
  // Re-preview whenever the mapping changes.
  const key = JSON.stringify(full);
  useEffect(() => {
    preview.mutate();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  const [prepStatus, setPrepStatus] = useState<string | null>(null);
  const prepare = useMutation({
    mutationFn: async () => {
      const { job_id } = await api.post<{ job_id: number }>(`/api/projects/${projectId}/datasets/${dataset.id}/prepare`, {
        mapping: full,
        rules,
        max_seq_length: maxSeq,
        include_feedback: includeFeedback,
      });
      const job = await waitForJob(job_id, (j) => setPrepStatus(j.status));
      if (job.status !== "succeeded") throw new Error(job.error || `Prepare ${job.status}`);
    },
    onSettled: () => {
      setPrepStatus(null);
      qc.invalidateQueries({ queryKey: ["datasets", projectId] });
      qc.invalidateQueries({ queryKey: ["overview", projectId] });
    },
  });
  const askPrep = useMutation({
    mutationFn: () => api.post(`/api/projects/${projectId}/agents/prep`, { dataset_id: dataset.id }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["proposals", projectId] }),
  });

  const fields = FORMAT_FIELDS[mapping.format] ?? [];
  const cols = ["", ...dataset.columns];
  const ok = preview.data && preview.data.failed < preview.data.sampled;

  return (
    <div className="space-y-4 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[13px] font-medium">{dataset.name}</span>
        <span className="font-mono text-[11px] text-faint">{dataset.source_ref}</span>
        <Button size="sm" variant="ghost" className="ml-auto" loading={askPrep.isPending} onClick={() => askPrep.mutate()}>
          Ask DataPrep agent
        </Button>
      </div>
      {askPrep.isSuccess && <p className="text-xs text-muted">DataPrep is working; its proposal will appear under Agents and on this page.</p>}

      <div className="grid gap-3 md:grid-cols-4">
        <Field label="Record format">
          <Select
            value={mapping.format}
            onChange={(f) => setMapping({ format: f } as Mapping)}
            options={[
              { value: "instruction", label: "Instruction → response" },
              { value: "chat", label: "Chat messages" },
              { value: "text", label: "Raw text" },
              { value: "preference", label: "Preference pairs (DPO)" },
            ]}
          />
        </Field>
        {fields.map((f) => (
          <Field key={f.key} label={f.label + (f.required ? "" : "")}>
            <Select value={mapping[f.key] ?? ""} onChange={(v) => setMapping({ ...mapping, [f.key]: v } as Mapping)} options={cols.map((c) => ({ value: c, label: c || "(none)" }))} />
          </Field>
        ))}
      </div>
      {mapping.format !== "text" && (
        <Field label="Constant system prompt" hint="optional">
          <Input value={system} onChange={(e) => setSystem(e.target.value)} placeholder="Leave blank to use none" />
        </Field>
      )}

      <div>
        <div className="mb-1 flex items-center gap-2 text-xs text-muted">
          Preview
          {preview.data && (
            <Badge tone={preview.data.failed ? (ok ? "warn" : "bad") : "good"}>
              {preview.data.sampled - preview.data.failed}/{preview.data.sampled} rows map
            </Badge>
          )}
        </div>
        <pre className="max-h-64 overflow-auto rounded-lg border border-line bg-bg p-3 font-mono text-[11px] leading-relaxed text-muted">
          {preview.data ? JSON.stringify(preview.data.records.slice(0, 2), null, 2) : "…"}
        </pre>
        {!!preview.data?.errors.length && <p className="mt-1 text-[11px] text-bad">{preview.data.errors[0]}</p>}
      </div>

      <Collapsible title="Cleaning and splitting">
        <div className="grid gap-3 md:grid-cols-4">
          <NumberField label="Min answer length" hint="chars" value={rules.min_chars} onChange={(v) => v != null && setRules({ ...rules, min_chars: v })} min={0} />
          <NumberField label="Max record length" hint="chars" value={rules.max_chars} onChange={(v) => v && setRules({ ...rules, max_chars: v })} step={1000} />
          <NumberField label="Max sequence length" hint="for token stats" value={maxSeq} onChange={(v) => v && setMaxSeq(v)} step={128} min={64} />
          <div className="space-y-2 pt-5">
            <Toggle label="Remove duplicates" checked={rules.dedupe} onChange={(v) => setRules({ ...rules, dedupe: v })} />
            <Toggle label="Add approved feedback examples" checked={includeFeedback} onChange={setIncludeFeedback} />
          </div>
        </div>
      </Collapsible>

      <ErrorNote error={prepare.error} />
      <Button variant="primary" disabled={!ok} loading={prepare.isPending} onClick={() => prepare.mutate()}>
        {prepare.isPending ? `Preparing (${prepStatus ?? "queued"})…` : "Clean, split and tokenise"}
      </Button>
    </div>
  );
}

function Versions({ versions, projectId }: { versions: DatasetVersion[]; projectId: number }) {
  const [open, setOpen] = useState<number | null>(null);
  const sample = useQuery({
    queryKey: ["version-sample", open],
    queryFn: () => api.get<{ records: unknown[] }>(`/api/projects/${projectId}/versions/${open}/sample?limit=5`),
    enabled: open != null,
  });
  if (!versions.length) return null;
  return (
    <Card title="Prepared versions" subtitle="Immutable snapshots; training runs reference these" pad={false}>
      <table className="w-full text-[13px]">
        <thead>
          <tr className="border-b border-line text-left text-[11px] uppercase tracking-wide text-faint">
            <th className="px-4 py-2 font-medium">Version</th>
            <th className="px-2 py-2 font-medium">Kind</th>
            <th className="px-2 py-2 text-right font-medium">Train / valid / test</th>
            <th className="px-2 py-2 text-right font-medium">Tokens p50 / p95 / max</th>
            <th className="px-2 py-2 font-medium">Cleaning</th>
            <th className="px-4 py-2" />
          </tr>
        </thead>
        <tbody>
          {versions.map((v) => {
            const dropped = Object.entries(v.cleaning_report.dropped ?? {});
            return (
              <tr key={v.id} className="border-b border-line align-top last:border-0">
                <td className="px-4 py-2">
                  <div className="font-mono text-[12px]">v{v.id}</div>
                  <div className="text-[11px] text-faint">{fmt.ago(v.created_at)}</div>
                </td>
                <td className="px-2 py-2">
                  <Badge tone={v.kind === "dpo" ? "accent" : "info"}>{v.kind.toUpperCase()}</Badge>
                </td>
                <td className="num px-2 py-2 text-right">
                  {v.n_train} / {v.n_valid} / {v.n_test}
                </td>
                <td className="num px-2 py-2 text-right text-muted">
                  {v.token_stats.p50 ?? "–"} / {v.token_stats.p95 ?? "–"} / {v.token_stats.max ?? "–"}
                  {!!v.token_stats.over_max_seq_length && <div className="text-[11px] text-warn">{v.token_stats.over_max_seq_length} over max length</div>}
                </td>
                <td className="px-2 py-2 text-[11px] text-muted">
                  kept {v.cleaning_report.kept ?? "?"}/{v.cleaning_report.input_rows ?? "?"}
                  {dropped.length > 0 && <div className="text-faint">{dropped.map(([k, n]) => `${k.replace(/_/g, " ")} ${n}`).join(", ")}</div>}
                </td>
                <td className="px-4 py-2 text-right">
                  <Button size="sm" variant="ghost" onClick={() => setOpen(open === v.id ? null : v.id)}>
                    {open === v.id ? "Hide" : "Sample"}
                  </Button>
                  {v.kind === "sft" && (
                    <Link to={`/p/${projectId}/train?version=${v.id}`}>
                      <Button size="sm">Train →</Button>
                    </Link>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {open != null && (
        <pre className="max-h-72 overflow-auto border-t border-line bg-bg p-4 font-mono text-[11px] text-muted">
          {sample.data ? JSON.stringify(sample.data.records, null, 2) : "Loading…"}
        </pre>
      )}
    </Card>
  );
}
