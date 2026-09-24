import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useRef, useState } from "react";
import { Link } from "react-router";

import { api, fmt, type Proposal } from "../api";
import { Badge, Button, Collapsible, ErrorNote, Mono, StatusBadge, TextArea } from "../ui";

const ACTION_LABEL: Record<string, string> = {
  import_dataset: "Import dataset",
  acquire_manually: "Needs you",
  prepare_dataset: "Prepare data",
  run_sft: "Fine-tune",
  run_dpo: "Preference round",
  generate_synthetic: "Synthetic data",
};

/** An agent's proposed action: the human approval gate. */
export default function ProposalCard({ p, projectId }: { p: Proposal; projectId: number }) {
  const qc = useQueryClient();
  const [payload, setPayload] = useState(() => JSON.stringify(stripPreview(p.payload), null, 2));
  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["proposals", projectId] });
    qc.invalidateQueries({ queryKey: ["overview", projectId] });
  };
  const approve = useMutation({
    mutationFn: () => {
      let overrides = {};
      try {
        overrides = JSON.parse(payload);
      } catch {
        throw new Error("Payload isn't valid JSON");
      }
      return api.post(`/api/proposals/${p.id}/approve`, { overrides });
    },
    onSuccess: refresh,
  });
  const reject = useMutation({ mutationFn: () => api.post(`/api/proposals/${p.id}/reject`), onSuccess: refresh });
  const fileRef = useRef<HTMLInputElement>(null);
  const upload = useMutation({
    mutationFn: (file: File) => {
      const form = new FormData();
      form.append("file", file);
      form.append("proposal_id", String(p.id));
      return api.upload(`/api/projects/${projectId}/datasets/upload`, form);
    },
    onSuccess: () => {
      refresh();
      qc.invalidateQueries({ queryKey: ["datasets", projectId] });
    },
  });

  const pl = p.payload as Record<string, unknown>;
  const pending = p.status === "pending";
  const awaitingUpload = p.action === "acquire_manually" && (p.status === "pending" || p.status === "approved");
  const jobId = (p.result as { job_id?: number }).job_id;

  return (
    <div className="rounded-xl border border-line bg-panel p-4">
      <div className="flex items-start gap-3">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone={p.action === "acquire_manually" ? "warn" : "accent"}>{ACTION_LABEL[p.action] ?? p.action}</Badge>
            <span className="text-[13px] font-medium">{p.title}</span>
          </div>
          <div className="mt-1 text-[11px] text-faint">
            {p.agent} · {fmt.ago(p.created_at)}
          </div>
        </div>
        <StatusBadge status={p.status} />
      </div>

      {p.rationale && <p className="mt-3 text-[13px] leading-relaxed text-muted">{p.rationale}</p>}

      {p.action === "import_dataset" && (
        <div className="mt-3 flex flex-wrap gap-x-4 gap-y-1 text-xs text-muted">
          <a className="text-info hover:underline" href={`https://huggingface.co/datasets/${pl.repo_id}`} target="_blank" rel="noreferrer">
            {String(pl.repo_id)} ↗
          </a>
          <span>licence: {String(pl.license || "unknown")}</span>
          <span>up to {String(pl.max_rows)} rows</span>
          <span>
            as <Mono>{String((pl.mapping as { format?: string })?.format)}</Mono>
          </span>
        </div>
      )}

      {p.action === "acquire_manually" && (
        <div className="mt-3 space-y-2 rounded-lg border border-warn/30 bg-warn-soft/40 p-3 text-[13px]">
          {!!pl.source_url && (
            <a className="text-info hover:underline" href={String(pl.source_url)} target="_blank" rel="noreferrer">
              {String(pl.source_url)} ↗
            </a>
          )}
          <p className="whitespace-pre-wrap leading-relaxed">{String(pl.instructions)}</p>
          {!!pl.expected_format && <p className="text-xs text-muted">Expected file: {String(pl.expected_format)}</p>}
        </div>
      )}

      {Array.isArray(pl.preview) && pl.preview.length > 0 && (
        <Collapsible title={`Preview ${pl.preview.length} mapped records`}>
          <pre className="max-h-56 overflow-auto font-mono text-[11px] text-muted">{JSON.stringify(pl.preview, null, 2)}</pre>
        </Collapsible>
      )}

      {pending && p.action !== "acquire_manually" && (
        <div className="mt-3">
          <Collapsible title="Edit before approving">
            <TextArea className="font-mono text-[11.5px]" rows={8} value={payload} onChange={(e) => setPayload(e.target.value)} />
          </Collapsible>
        </div>
      )}

      <ErrorNote error={approve.error ?? reject.error ?? upload.error} />
      {p.status === "failed" && !!(p.result as { error?: string }).error && <ErrorNote error={(p.result as { error: string }).error} />}

      <div className="mt-3 flex flex-wrap items-center gap-2">
        {pending && p.action !== "acquire_manually" && (
          <Button variant="primary" size="sm" loading={approve.isPending} onClick={() => approve.mutate()}>
            Approve & run
          </Button>
        )}
        {awaitingUpload && (
          <>
            <input
              ref={fileRef}
              type="file"
              className="hidden"
              accept=".jsonl,.json,.csv,.parquet,.txt,.md"
              onChange={(e) => e.target.files?.[0] && upload.mutate(e.target.files[0])}
            />
            <Button variant="primary" size="sm" loading={upload.isPending} onClick={() => fileRef.current?.click()}>
              Upload the file
            </Button>
          </>
        )}
        {pending && (
          <Button variant="ghost" size="sm" loading={reject.isPending} onClick={() => reject.mutate()}>
            Dismiss
          </Button>
        )}
        {jobId != null && (
          <Link
            className="text-xs text-info hover:underline"
            to={["run_sft", "run_dpo"].includes(p.action) ? `/p/${projectId}/train/${jobId}` : `/p/${projectId}/agents`}
          >
            job {jobId} →
          </Link>
        )}
      </div>
    </div>
  );
}

function stripPreview(payload: Record<string, unknown>) {
  const { preview: _preview, ...rest } = payload;
  return rest;
}
