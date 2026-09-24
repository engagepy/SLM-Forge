import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api, fmt, type Job, type MemoryEstimate, type ModelCandidate, type Preset } from "../api";
import { useOverview, useProjectId, waitForJob } from "../hooks";
import { Badge, Button, Card, cx, Empty, ErrorNote, Input, MemoryBar, Spinner, Toggle } from "../ui";

const FIT = {
  fits: { tone: "good", label: "fits" },
  tight: { tone: "warn", label: "tight" },
  too_big: { tone: "bad", label: "too big" },
  unknown: { tone: "neutral", label: "?" },
} as const;

const SUGGESTIONS = ["Qwen3", "Qwen2.5 Instruct", "Llama-3.2", "gemma-3", "SmolLM", "Phi-4-mini"];

interface Inspection {
  repo_id: string;
  params: number;
  bits: number;
  layers: number;
  hidden_size: number;
  vocab_size: number;
  inference: MemoryEstimate;
  presets: Record<"safe" | "balanced" | "quality", Preset>;
  dpo_safe: MemoryEstimate;
}

export default function ModelPage() {
  const projectId = useProjectId();
  const { data: ov } = useOverview(projectId);
  const [q, setQ] = useState("Qwen2.5 Instruct");
  const [submitted, setSubmitted] = useState(q);
  const [includeBig, setIncludeBig] = useState(false);
  const [selected, setSelected] = useState<string | null>(null);

  const results = useQuery({
    queryKey: ["model-search", submitted, includeBig],
    queryFn: () =>
      api.get<ModelCandidate[]>(`/api/models/search?q=${encodeURIComponent(submitted)}&include_too_big=${includeBig}`),
  });

  const current = ov?.project.base_model;
  return (
    <div className="mx-auto max-w-6xl space-y-5 p-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Base model</h1>
        <p className="mt-1 text-[13px] text-muted">
          Models from <span className="font-mono">mlx-community</span> are already converted for Apple Silicon. Sizes are checked
          against this Mac's memory. Smaller models train faster and are easier to run anywhere; start with 0.5–3B.
        </p>
      </div>

      {current && (
        <div className="flex items-center gap-2 rounded-lg border border-line bg-panel px-4 py-3 text-[13px]">
          Current base model: <span className="font-mono">{current}</span>
          {ov?.base_model_downloaded ? <Badge tone="good">downloaded</Badge> : <Badge tone="warn">not downloaded</Badge>}
          {!!ov?.checkpoints.length && <span className="text-xs text-faint">(locked: training has started)</span>}
        </div>
      )}

      <form
        className="flex flex-wrap items-center gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          setSubmitted(q);
        }}
      >
        <Input className="max-w-sm" value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search models" />
        <Button type="submit">Search</Button>
        <div className="flex gap-1">
          {SUGGESTIONS.map((s) => (
            <button
              key={s}
              type="button"
              className="rounded-md px-2 py-1 text-xs text-muted hover:bg-panel-2 hover:text-fg"
              onClick={() => (setQ(s), setSubmitted(s))}
            >
              {s}
            </button>
          ))}
        </div>
        <div className="ml-auto">
          <Toggle label="Show models that won't fit" checked={includeBig} onChange={setIncludeBig} />
        </div>
      </form>

      <div className="grid gap-5 lg:grid-cols-5">
        <Card pad={false} className="lg:col-span-3">
          {results.isLoading ? (
            <div className="flex items-center gap-2 p-6 text-xs text-muted">
              <Spinner /> Searching Hugging Face…
            </div>
          ) : results.error ? (
            <div className="p-4">
              <ErrorNote error={results.error} />
            </div>
          ) : !results.data?.length ? (
            <div className="p-4">
              <Empty title="No matching models">Try a broader search, or show models that won't fit.</Empty>
            </div>
          ) : (
            <table className="w-full text-[13px]">
              <thead>
                <tr className="border-b border-line text-left text-[11px] uppercase tracking-wide text-faint">
                  <th className="px-4 py-2 font-medium">Model</th>
                  <th className="px-2 py-2 text-right font-medium">Params</th>
                  <th className="px-2 py-2 text-right font-medium">Bits</th>
                  <th className="px-2 py-2 text-right font-medium">Train ≈</th>
                  <th className="px-4 py-2 font-medium">Fit</th>
                </tr>
              </thead>
              <tbody>
                {results.data.map((m) => (
                  <tr
                    key={m.id}
                    onClick={() => setSelected(m.id)}
                    className={cx("cursor-pointer border-b border-line last:border-0 hover:bg-panel-2", selected === m.id && "bg-accent-soft")}
                  >
                    <td className="px-4 py-2">
                      <div className="font-mono text-[12px]">{m.id.replace("mlx-community/", "")}</div>
                      <div className="text-[11px] text-faint">
                        {fmt.compact(m.downloads)} downloads{m.gated && " · gated"}
                        {m.id === current && " · current"}
                      </div>
                    </td>
                    <td className="num px-2 py-2 text-right">{fmt.params(m.params)}</td>
                    <td className="num px-2 py-2 text-right text-muted">{m.bits}</td>
                    <td className="num px-2 py-2 text-right text-muted">{fmt.gb(m.train_estimate_gb)}</td>
                    <td className="px-4 py-2">
                      <Badge tone={FIT[m.fit].tone}>{FIT[m.fit].label}</Badge>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </Card>

        <div className="lg:col-span-2">
          {selected ? (
            <ModelDetail repoId={selected} projectId={projectId} locked={!!ov?.checkpoints.length && selected !== current} />
          ) : (
            <Empty title="Select a model">See the exact memory it needs for inference and for each training preset.</Empty>
          )}
        </div>
      </div>
    </div>
  );
}

function ModelDetail({ repoId, projectId, locked }: { repoId: string; projectId: number; locked: boolean }) {
  const qc = useQueryClient();
  const info = useQuery({
    queryKey: ["inspect", repoId],
    queryFn: () => api.get<Inspection>(`/api/models/inspect?repo_id=${encodeURIComponent(repoId)}`),
  });
  const [status, setStatus] = useState<string | null>(null);
  const choose = useMutation({
    mutationFn: async () => {
      await api.patch(`/api/projects/${projectId}`, { base_model: repoId });
      const { job_id } = await api.post<{ job_id: number }>("/api/models/download", { repo_id: repoId, project_id: projectId });
      const job: Job = await waitForJob(job_id, (j) => setStatus(j.status));
      if (job.status !== "succeeded") throw new Error(job.error || `Download ${job.status}`);
      return job;
    },
    onSettled: () => {
      setStatus(null);
      qc.invalidateQueries({ queryKey: ["overview", projectId] });
      qc.invalidateQueries({ queryKey: ["projects"] });
    },
  });

  return (
    <Card
      title={<span className="font-mono">{repoId}</span>}
      subtitle={
        <a className="text-info hover:underline" href={`https://huggingface.co/${repoId}`} target="_blank" rel="noreferrer">
          model card ↗
        </a>
      }
    >
      {info.isLoading && (
        <div className="flex items-center gap-2 text-xs text-muted">
          <Spinner /> Reading config.json…
        </div>
      )}
      <ErrorNote error={info.error} />
      {info.data && (
        <div className="space-y-4">
          <div className="grid grid-cols-3 gap-2 text-xs">
            <div>
              <div className="text-faint">Parameters</div>
              <div className="num text-[13px]">{fmt.params(info.data.params)}</div>
            </div>
            <div>
              <div className="text-faint">Layers × width</div>
              <div className="num text-[13px]">
                {info.data.layers} × {info.data.hidden_size}
              </div>
            </div>
            <div>
              <div className="text-faint">Vocabulary</div>
              <div className="num text-[13px]">{fmt.compact(info.data.vocab_size)}</div>
            </div>
          </div>

          <div>
            <div className="mb-1.5 text-xs font-medium text-muted">Chat (4k context)</div>
            <MemoryBar est={info.data.inference} />
          </div>
          {(["safe", "balanced", "quality"] as const).map((name) => (
            <div key={name}>
              <div className="mb-1.5 flex items-baseline justify-between text-xs">
                <span className="font-medium capitalize text-muted">{name} training</span>
                <span className="num text-faint">
                  rank {info.data!.presets[name].config.lora_rank} · bs {info.data!.presets[name].config.batch_size} · seq{" "}
                  {info.data!.presets[name].config.max_seq_length}
                </span>
              </div>
              <MemoryBar est={info.data.presets[name].estimate} />
            </div>
          ))}
          <p className="text-[11px] leading-snug text-faint">
            Training estimates assume every sequence is max length. Once your data is prepared they're recomputed from its real
            token lengths, and are usually much lower.
          </p>

          <ErrorNote error={choose.error} />
          <Button
            variant="primary"
            className="w-full"
            disabled={locked}
            loading={choose.isPending}
            onClick={() => choose.mutate()}
          >
            {choose.isPending ? `Downloading… (${status ?? "queued"})` : locked ? "Base model locked after training" : "Use this model"}
          </Button>
        </div>
      )}
    </Card>
  );
}
