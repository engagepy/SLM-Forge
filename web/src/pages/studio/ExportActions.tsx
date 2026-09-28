// After the export: GGUF files for llama.cpp / Ollama / LM Studio, and publishing to Hugging Face.
// Clicking a button here is the user's go-ahead; the Tuner reaches the same jobs only through a card.
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api, type ExportInfo, fmt, type HuggingFaceAccount, type Snapshot } from "../../api";
import { invalidate } from "../../hooks";
import { Badge, Button, CodeBlock, cx, ErrorNote, Input, Spinner } from "../../ui";

const repoName = (path: string) =>
  (path.split("/").pop() ?? "model").replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^[^A-Za-z0-9]+/, "").slice(0, 96) || "model";

export function ExportActions({ s, e }: { s: Snapshot; e: ExportInfo }) {
  const qc = useQueryClient();
  const account = useQuery({ queryKey: ["huggingface"], queryFn: () => api.get<HuggingFaceAccount>("/api/huggingface") });
  const [ggufJob, setGgufJob] = useState<number | null>(null);
  const [uploadJob, setUploadJob] = useState<number | null>(null);
  const [publishing, setPublishing] = useState(false);
  const [name, setName] = useState(() => repoName(e.path));
  const [isPrivate, setPrivate] = useState(false);

  const status = (id: number | null) => (id == null ? null : s.jobs.find((j) => j.id === id));
  const gJob = status(ggufJob);
  const uJob = status(uploadJob);
  const running = (j: ReturnType<typeof status>) => !!j && (j.status === "queued" || j.status === "running");

  const makeGguf = useMutation({
    mutationFn: () => api.post<{ job_id: number }>(`/api/projects/${s.project.id}/exports/${e.job_id}/gguf`, {}),
    onSuccess: (r) => {
      setGgufJob(r.job_id);
      invalidate(qc, "studio", "huggingface");
    },
  });
  const publish = useMutation({
    mutationFn: () =>
      api.post<{ job_id: number; repo_id: string }>(`/api/projects/${s.project.id}/exports/${e.job_id}/publish`, {
        repo_name: name.trim(),
        private: isPrivate,
      }),
    onSuccess: (r) => {
      setUploadJob(r.job_id);
      setPublishing(false);
      invalidate(qc, "studio");
    },
  });

  const acct = account.data;
  const lic = e.base_license;
  const files = e.gguf ?? [];
  return (
    <div className="space-y-2 rounded-lg border border-line bg-panel-2/40 p-3">
      <div className="flex flex-wrap items-center gap-2">
        <Button size="sm" loading={makeGguf.isPending || running(gJob)} onClick={() => makeGguf.mutate()}>
          {files.length ? "Remake GGUF" : "Make GGUF"}
        </Button>
        {e.huggingface ? (
          <a className="text-[12px] font-medium text-accent hover:underline" href={e.huggingface.url} target="_blank" rel="noreferrer">
            {e.huggingface.repo_id} ↗
          </a>
        ) : (
          <Button size="sm" variant="primary" disabled={running(uJob)} onClick={() => setPublishing(!publishing)}>
            {running(uJob) ? <Spinner /> : null} Upload to Hugging Face
          </Button>
        )}
        {e.huggingface && <Badge tone={e.huggingface.private ? "neutral" : "good"}>{e.huggingface.private ? "private" : "public"}</Badge>}
        <span className="ml-auto text-[11px] text-faint">GGUF runs in llama.cpp, Ollama and LM Studio</span>
      </div>

      {running(gJob) && <p className="text-[12px] text-info">Converting to GGUF… the first time also installs llama.cpp's converter (~300 MB).</p>}
      {gJob?.status === "failed" && <p className="text-[12px] text-bad">{gJob.error}</p>}
      {acct && !acct.quantizer_available && !files.some((f) => f.quant === "Q4_K_M") && (
        <p className="text-[11px] text-faint">Q4_K_M needs llama.cpp's quantizer: `brew install llama.cpp`. Without it you get Q8_0.</p>
      )}
      {files.length > 0 && (
        <div className="space-y-1.5">
          <div className="flex flex-wrap gap-1.5">
            {files.map((f) => (
              <Badge key={f.name} tone="info">
                {f.quant} · {fmt.gb(f.size_gb)}
              </Badge>
            ))}
          </div>
          <CodeBlock
            text={
              e.huggingface
                ? `ollama run hf.co/${e.huggingface.repo_id}:${files[0].quant}`
                : `llama-cli -m "${e.path}/${files[0].name}" -p "Hello"`
            }
          />
        </div>
      )}

      {publishing && !e.huggingface && (
        <div className="space-y-2 border-t border-line pt-2">
          {!acct ? (
            <Spinner />
          ) : !acct.logged_in || !acct.can_write ? (
            <p className="text-[12px] text-warn">
              {acct.logged_in ? "Your Hugging Face token is read-only." : "Not logged in to Hugging Face."} Add{" "}
              <code>HF_TOKEN=hf_…</code> (a token that can write, from huggingface.co/settings/tokens) to <code>.env</code> and
              restart SLM Forge, or run <code>uv run hf auth login</code>.
            </p>
          ) : (
            <>
              <div className="flex flex-wrap items-center gap-2 text-[12px]">
                <span className="text-muted">huggingface.co/{acct.user}/</span>
                <Input className="w-64 py-1 text-[12px]" value={name} onChange={(ev) => setName(ev.target.value)} aria-label="Repository name" />
                <div className="flex rounded-md border border-line p-0.5" role="radiogroup" aria-label="Visibility">
                  {(["public", "private"] as const).map((v) => (
                    <button
                      key={v}
                      role="radio"
                      aria-checked={(v === "private") === isPrivate}
                      onClick={() => setPrivate(v === "private")}
                      className={cx(
                        "rounded px-2 py-0.5 text-[12px] font-medium",
                        (v === "private") === isPrivate ? "bg-panel-2 text-fg" : "text-faint hover:text-fg",
                      )}
                    >
                      {v === "public" ? "Public" : "Private"}
                    </button>
                  ))}
                </div>
              </div>
              <p className="text-[11px] leading-relaxed text-muted">
                Uploads the model, {files.length ? "its GGUF files, " : ""}the model card and the licence files.{" "}
                {lic ? (
                  <>
                    Licence: <strong>{lic.licence}</strong>. {lic.conditions}
                  </>
                ) : (
                  "The base model's licence is in the model card."
                )}
              </p>
              {lic?.commercial_ok === false && (
                <p className="text-[11px] text-warn">This base model's licence doesn't allow commercial use: say so wherever you share it.</p>
              )}
              <ErrorNote error={publish.error} />
              <div className="flex gap-2">
                <Button size="sm" variant="primary" loading={publish.isPending} disabled={!name.trim()} onClick={() => publish.mutate()}>
                  Publish {isPrivate ? "privately" : "publicly"}
                </Button>
                <Button size="sm" variant="ghost" onClick={() => setPublishing(false)}>
                  Cancel
                </Button>
              </div>
            </>
          )}
        </div>
      )}
      {running(uJob) && <p className="text-[12px] text-info">Uploading to Hugging Face…</p>}
      {uJob?.status === "failed" && <p className="text-[12px] text-bad">{uJob.error}</p>}
      <ErrorNote error={makeGguf.error} />
    </div>
  );
}
