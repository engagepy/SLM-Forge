import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router";

import { api, fmt, type Dataset, type DatasetVersion, type Mapping, type Proposal } from "../api";
import ProposalCard from "../components/ProposalCard";
import { useProjectId, followJob } from "../hooks";
import { Badge, Button, Card, Collapsible, cx, Empty, ErrorNote, Field, Input, LinkButton, NumberField, Select, Spinner, Toggle, useSpotlight } from "../ui";

export default function DataPage() {
  const projectId = useProjectId();
  const data = useQuery({
    queryKey: ["datasets", projectId],
    queryFn: () => api.get<{ datasets: Dataset[]; versions: DatasetVersion[] }>(`/api/projects/${projectId}/datasets`),
  });
  const [selected, setSelected] = useState<number | null>(null);
  const datasets = data.data?.datasets ?? [];
  const versions = data.data?.versions ?? [];
  const current = datasets.find((d) => d.id === selected) ?? datasets[0];
  const [spot, spotlight] = useSpotlight();

  // Whatever produced it (your click, an approved proposal, an agent), bring new work into view.
  useArrivals(datasets, data.isSuccess, (d) => {
    setSelected(d.id);
    spotlight("dataset-editor");
  });
  useArrivals(versions, data.isSuccess, (v) => spotlight(`version-${v.id}`));
  const latest = versions.find((v) => v.kind === "sft");

  return (
    <div className="mx-auto max-w-6xl space-y-5 p-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Data</h1>
        <p className="mt-1 text-[13px] text-muted">
          Training data: let the scout find public datasets, browse the Hub yourself, or upload your own. Then map columns to training records,
          clean them and split them.
        </p>
      </div>

      {latest && (
        <div className="flex flex-wrap items-center gap-3 rounded-xl border border-good/40 bg-good-soft px-4 py-3">
          <span className="grid size-6 place-items-center rounded-full bg-good text-xs font-bold text-white">✓</span>
          <div className="min-w-0 flex-1 text-[13px]">
            <span className="font-medium">Data ready: v{latest.id}</span>
            <span className="text-muted">
              {" "}
              · {latest.n_train.toLocaleString()} training examples · p95 {latest.token_stats.p95 ?? "?"} tokens
            </span>
          </div>
          <Button size="sm" variant="ghost" onClick={() => spotlight(`version-${latest.id}`)}>
            View
          </Button>
          <LinkButton to={`/p/${projectId}/train?version=${latest.id}`} size="sm" variant="primary">Next: train on it →</LinkButton>
        </div>
      )}

      <div className="grid gap-5 lg:grid-cols-2">
        <ScoutPanel projectId={projectId} />
        <div className="space-y-5">
          <UploadPanel projectId={projectId} onUploaded={(id) => setSelected(id)} />
          <HubSearch projectId={projectId} />
        </div>
      </div>

      <Card title="Raw datasets" subtitle="Pick one to map and prepare" pad={false}>
        {datasets.length ? (
          <div className="grid lg:grid-cols-[260px_minmax(0,1fr)]">
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
            <div id="dataset-editor" className={cx("min-w-0 scroll-mt-4 transition-shadow", spot === "dataset-editor" && "ring-2 ring-accent ring-inset")}>
              {current && <MappingEditor key={current.id} dataset={current} projectId={projectId} onPrepared={(id) => spotlight(`version-${id}`)} />}
            </div>
          </div>
        ) : (
          <div className="p-4">
            <Empty title="No datasets yet">Run the scout, import from the Hub, or upload a file above.</Empty>
          </div>
        )}
      </Card>

      <Versions versions={versions} projectId={projectId} spot={spot} />
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
      const job = await followJob(job_id, "Scout", setJobStatus);
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
      await followJob(job_id, "Import", setImportStatus);
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
                    <pre className="max-h-40 overflow-auto rounded-md bg-bg p-2 whitespace-pre-wrap break-words font-mono text-[11px] text-muted">
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

function MappingEditor({ dataset, projectId, onPrepared }: { dataset: Dataset; projectId: number; onPrepared: (versionId: number) => void }) {
  const qc = useQueryClient();
  const initial = dataset.suggested_mapping.format === "unknown" ? ({ format: "instruction" } as Mapping) : dataset.suggested_mapping;
  const [mapping, setMapping] = useState<Mapping>(initial);
  const [system, setSystem] = useState("");
  const [rules, setRules] = useState({ min_chars: 2, max_chars: 20000, dedupe: true, long_examples: "auto" });
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
      const job = await followJob(job_id, "Prepare", setPrepStatus);
      return job.result as { dataset_version_id: number; n_train?: number; kept?: number; input_rows?: number };
    },
    onSettled: () => {
      setPrepStatus(null);
      qc.invalidateQueries({ queryKey: ["datasets", projectId] });
      qc.invalidateQueries({ queryKey: ["overview", projectId] });
    },
    onSuccess: (r) => onPrepared(r.dataset_version_id),
  });
  const [applied, setApplied] = useState(false);
  const askPrep = useMutation({
    mutationFn: () => api.post(`/api/projects/${projectId}/agents/prep`, { dataset_id: dataset.id }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["proposals", projectId] }),
  });
  // DataPrep's latest suggestion for this dataset (the project feed keeps this fresh).
  const proposals = useQuery({
    queryKey: ["proposals", projectId],
    queryFn: () => api.get<Proposal[]>(`/api/projects/${projectId}/proposals`),
  });
  const suggestion = proposals.data?.find(
    (p) => p.action === "prepare_dataset" && p.status === "pending" && (p.payload as { dataset_id?: number }).dataset_id === dataset.id,
  );
  const applySuggestion = (p: Proposal) => {
    const { system: sys, ...rest } = (p.payload as { mapping: Mapping }).mapping;
    setMapping(rest as Mapping);
    setSystem(sys?.startsWith("=") ? sys.slice(1) : "");
    const r = (p.payload as { rules?: Partial<typeof rules> }).rules ?? {};
    setRules({ ...rules, ...r });
    setApplied(true);
    requestAnimationFrame(() => document.getElementById(`preview-${dataset.id}`)?.scrollIntoView({ behavior: "smooth", block: "center" }));
  };

  const fields = FORMAT_FIELDS[mapping.format] ?? [];
  const cols = ["", ...dataset.columns];
  const missing = fields.filter((f) => f.required && !mapping[f.key]).map((f) => f.label.toLowerCase());
  const ok = !missing.length && preview.data && preview.data.failed < preview.data.sampled;

  return (
    <div className="space-y-4 p-4">
      <div className="flex flex-wrap items-center gap-2">
        <span className="text-[13px] font-medium">{dataset.name}</span>
        <span className="font-mono text-[11px] text-faint">{dataset.source_ref}</span>
        <Button size="sm" variant="ghost" className="ml-auto" loading={askPrep.isPending} onClick={() => askPrep.mutate()}>
          Ask DataPrep agent
        </Button>
      </div>
      {askPrep.isSuccess && !suggestion && (
        <p className="flex items-center gap-2 text-xs text-muted">
          <Spinner /> DataPrep is reading the columns and sample rows…
        </p>
      )}
      {suggestion && (
        <div className="rounded-lg border border-accent/40 bg-accent-soft/50 p-3 text-[13px]">
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone="accent">DataPrep suggests</Badge>
            <span className="font-mono text-xs">
              {Object.entries((suggestion.payload as { mapping: Record<string, string> }).mapping)
                .map(([k, v]) => `${k}: ${v}`)
                .join(" · ")}
            </span>
            <Button size="sm" variant={applied ? "good" : "primary"} className="ml-auto" onClick={() => applySuggestion(suggestion)}>
              {applied ? "✓ Applied" : "Use this mapping"}
            </Button>
          </div>
          {suggestion.rationale && <p className="mt-1.5 text-xs leading-relaxed text-muted">{suggestion.rationale}</p>}
        </div>
      )}

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
        <Field label="Constant system prompt" hint="optional · {column} inserts a value">
          <Input value={system} onChange={(e) => setSystem(e.target.value)} placeholder="Leave blank to use none" />
        </Field>
      )}

      <div id={`preview-${dataset.id}`} className="scroll-mt-4">
        <div className="mb-1 flex items-center gap-2 text-xs text-muted">
          Preview
          {preview.data && !missing.length && (
            <Badge tone={preview.data.failed ? (ok ? "warn" : "bad") : "good"}>
              {preview.data.sampled - preview.data.failed}/{preview.data.sampled} rows map
            </Badge>
          )}
        </div>
        {missing.length ? (
          <div className="rounded-lg border border-warn/40 bg-warn-soft px-3 py-2.5 text-xs text-warn">
            Choose a column for {missing.join(" and ")}
            {dataset.columns.length > 0 && (
              <span className="text-muted">
                {" "}
                (available: <span className="font-mono">{dataset.columns.join(", ")}</span>)
              </span>
            )}
            , or ask DataPrep.
          </div>
        ) : (
          <pre className="max-h-64 overflow-auto rounded-lg border border-line bg-bg p-3 whitespace-pre-wrap break-words font-mono text-[11px] leading-relaxed text-muted">
            {preview.data ? (preview.data.records.length ? JSON.stringify(preview.data.records.slice(0, 2), null, 2) : "No rows mapped.") : "…"}
          </pre>
        )}
        {!missing.length && !!preview.data?.errors.length && <p className="mt-1 text-[11px] text-bad">{preview.data.errors[0]}</p>}
      </div>

      <Collapsible title="Cleaning and splitting">
        <div className="grid gap-3 md:grid-cols-4">
          <NumberField label="Min answer length" hint="chars" value={rules.min_chars} onChange={(v) => v != null && setRules({ ...rules, min_chars: v })} min={0} />
          <NumberField label="Max record length" hint="chars" value={rules.max_chars} onChange={(v) => v && setRules({ ...rules, max_chars: v })} step={1000} />
          <NumberField label="Max sequence length" hint="tokens" value={maxSeq} onChange={(v) => v && setMaxSeq(v)} step={128} min={64} />
          <Field label="Longer examples" hint="than max length">
            <Select
              value={rules.long_examples}
              onChange={(v) => setRules({ ...rules, long_examples: v })}
              options={[
                { value: "auto", label: "Auto: drop Q&A, split text" },
                { value: "drop", label: "Drop them" },
                { value: "split", label: "Split raw text into windows" },
                { value: "keep", label: "Keep (trainer cuts endings)" },
              ]}
            />
          </Field>
          <div className="space-y-2 pt-5">
            <Toggle label="Remove duplicates" checked={rules.dedupe} onChange={(v) => setRules({ ...rules, dedupe: v })} />
            <Toggle label="Add approved feedback examples" checked={includeFeedback} onChange={setIncludeFeedback} />
          </div>
        </div>
      </Collapsible>

      <ErrorNote error={prepare.error} />
      <div className="flex flex-wrap items-center gap-3">
        <Button variant="primary" disabled={!ok} loading={prepare.isPending} onClick={() => prepare.mutate()}>
          {prepare.isPending ? `Preparing (${prepStatus ?? "queued"})…` : "Clean, split and tokenise"}
        </Button>
        {prepare.isPending && <span className="text-xs text-muted">Tokenising every record; large datasets take a minute.</span>}
        {prepare.data && !prepare.isPending && (
          <>
            <span className="text-[13px] text-good">
              ✓ Prepared v{prepare.data.dataset_version_id}
              {prepare.data.n_train != null && `: ${prepare.data.n_train.toLocaleString()} training examples`}
            </span>
            <LinkButton to={`/p/${projectId}/train?version=${prepare.data.dataset_version_id}`} size="sm">Train on it →</LinkButton>
          </>
        )}
      </div>
    </div>
  );
}

function Versions({ versions, projectId, spot }: { versions: DatasetVersion[]; projectId: number; spot: string | null }) {
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
              <tr
                key={v.id}
                id={`version-${v.id}`}
                className={cx("border-b border-line align-top transition-colors duration-700 last:border-0", spot === `version-${v.id}` && "bg-good-soft")}
              >
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
                    <LinkButton to={`/p/${projectId}/train?version=${v.id}`} size="sm">Train →</LinkButton>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
      {open != null && (
        <pre className="max-h-72 overflow-auto border-t border-line bg-bg p-4 whitespace-pre-wrap break-words font-mono text-[11px] text-muted">
          {sample.data ? JSON.stringify(sample.data.records, null, 2) : "Loading…"}
        </pre>
      )}
    </Card>
  );
}

/** Call `onNew` for items that appear after the first successful load (not for what was
 * already there). `loaded` distinguishes "still loading" from "loaded, and empty". */
function useArrivals<T extends { id: number }>(items: T[], loaded: boolean, onNew: (item: T) => void) {
  const seen = useRef<Set<number> | null>(null);
  const cb = useRef(onNew);
  cb.current = onNew;
  useEffect(() => {
    if (seen.current === null) {
      if (loaded) seen.current = new Set(items.map((i) => i.id));
      return;
    }
    for (const item of items) {
      if (!seen.current.has(item.id)) {
        seen.current.add(item.id);
        cb.current(item);
      }
    }
  }, [items, loaded]);
}
