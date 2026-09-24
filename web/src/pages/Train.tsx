import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Link, useNavigate, useParams, useSearchParams } from "react-router";

import { api, fmt, type DatasetVersion, type Job, type MemoryEstimate, type Preset, type TrainConfig } from "../api";
import { MetricChart } from "../components/Charts";
import JobLog from "../components/JobLog";
import TrainControls from "../components/TrainControls";
import { useLiveJob, useOverview, useProjectId } from "../hooks";
import { Badge, Button, Card, cx, Empty, ErrorNote, Field, MemoryBar, Select, Stat, StatusBadge } from "../ui";

export default function TrainPage() {
  const projectId = useProjectId();
  const { jobId } = useParams();
  const runs = useQuery({
    queryKey: ["jobs", projectId, "train"],
    queryFn: () => api.get<Job[]>(`/api/jobs?project_id=${projectId}&kind=sft,dpo&limit=50`),
  });
  return (
    <div className="grid h-full lg:grid-cols-[260px_1fr]">
      <aside className="border-b border-line lg:border-r lg:border-b-0">
        <div className="flex items-center justify-between px-4 py-3">
          <span className="text-[13px] font-semibold">Runs</span>
          <Link to={`/p/${projectId}/train`}>
            <Button size="sm" variant={jobId ? "secondary" : "primary"}>
              New run
            </Button>
          </Link>
        </div>
        <ul>
          {runs.data?.map((j) => (
            <li key={j.id}>
              <Link
                to={`/p/${projectId}/train/${j.id}`}
                className={cx("flex items-center gap-2 px-4 py-2 hover:bg-panel-2", String(j.id) === jobId && "bg-accent-soft")}
              >
                <Badge tone={j.kind === "dpo" ? "accent" : "info"}>{j.kind.toUpperCase()}</Badge>
                <span className="flex-1 text-xs">
                  job {j.id}
                  <span className="block text-[11px] text-faint">{fmt.ago(j.created_at)}</span>
                </span>
                <StatusBadge status={j.status} />
              </Link>
            </li>
          ))}
          {!runs.data?.length && <li className="px-4 py-2 text-xs text-faint">No runs yet.</li>}
        </ul>
      </aside>
      <div className="min-w-0 p-6">{jobId ? <RunView key={jobId} jobId={Number(jobId)} projectId={projectId} /> : <Launcher projectId={projectId} />}</div>
    </div>
  );
}

function Launcher({ projectId }: { projectId: number }) {
  const navigate = useNavigate();
  const qc = useQueryClient();
  const [search] = useSearchParams();
  const { data: ov } = useOverview(projectId);
  const [mode, setMode] = useState<"sft" | "dpo">("sft");
  const datasets = useQuery({
    queryKey: ["datasets", projectId],
    queryFn: () => api.get<{ versions: DatasetVersion[] }>(`/api/projects/${projectId}/datasets`),
  });
  const versions = (datasets.data?.versions ?? []).filter((v) => v.kind === mode);
  const [versionId, setVersionId] = useState<string>(search.get("version") ?? "");
  // DPO defaults to the feedback pairs collected in the app ("" = build from feedback).
  const effectiveVersion = mode === "sft" ? versionId || String(versions[0]?.id ?? "") : versionId;

  const presets = useQuery({
    queryKey: ["presets", projectId, mode, effectiveVersion],
    queryFn: () =>
      api.get<{ presets: Record<string, Preset>; budget_gb: number; typical_len: number | null }>(
        `/api/projects/${projectId}/train/presets?mode=${mode}${effectiveVersion ? `&dataset_version_id=${effectiveVersion}` : ""}`,
      ),
    enabled: !!ov?.base_model_downloaded,
  });
  const [presetName, setPresetName] = useState("safe");
  const [startFrom, setStartFrom] = useState<"current" | "base">("current");
  const [cfg, setCfg] = useState<TrainConfig | null>(null);
  useEffect(() => {
    if (presets.data) setCfg(presets.data.presets[presetName].config);
  }, [presets.data, presetName]);

  // Live estimate as settings change (debounced).
  const [estimate, setEstimate] = useState<{ estimate: MemoryEstimate; iters?: number; epochs?: number } | null>(null);
  useEffect(() => {
    if (!cfg) return;
    const t = setTimeout(() => {
      api
        .post<{ estimate: MemoryEstimate; iters?: number; epochs?: number }>(`/api/projects/${projectId}/train/estimate`, {
          train: cfg,
          dataset_version_id: effectiveVersion ? Number(effectiveVersion) : null,
        })
        .then(setEstimate)
        .catch(() => setEstimate(null));
    }, 250);
    return () => clearTimeout(t);
  }, [cfg, effectiveVersion, projectId]);

  const launch = useMutation({
    mutationFn: (force: boolean) =>
      api.post<{ job_id: number }>(`/api/projects/${projectId}/train/${mode}`, {
        train: cfg,
        dataset_version_id: effectiveVersion ? Number(effectiveVersion) : null,
        start_from: mode === "sft" ? startFrom : "current",
        force,
      }),
    onSuccess: ({ job_id }) => {
      qc.invalidateQueries({ queryKey: ["jobs"] });
      navigate(`/p/${projectId}/train/${job_id}`);
    },
  });

  if (ov && !ov.base_model_downloaded) {
    return (
      <Empty title="Choose a base model first" action={<Link to={`/p/${projectId}/model`}><Button variant="primary">Pick a model →</Button></Link>}>
        Training needs a downloaded base model.
      </Empty>
    );
  }

  const pairsReady = ov?.counts.pairs_ready ?? 0;
  const canLaunch = cfg && (mode === "sft" ? !!effectiveVersion : !!effectiveVersion || pairsReady >= 3);

  return (
    <div className="mx-auto max-w-4xl space-y-5">
      <div className="flex items-center justify-between">
        <h1 className="text-xl font-semibold tracking-tight">New training run</h1>
        <div className="flex rounded-lg border border-line p-0.5">
          {(["sft", "dpo"] as const).map((m) => (
            <button
              key={m}
              onClick={() => (setMode(m), setVersionId(""))}
              className={cx("rounded-md px-3 py-1 text-[13px]", mode === m ? "bg-accent text-white" : "text-muted hover:text-fg")}
            >
              {m === "sft" ? "Fine-tune (SFT)" : "Preference (DPO)"}
            </button>
          ))}
        </div>
      </div>
      <p className="-mt-2 text-[13px] text-muted">
        {mode === "sft"
          ? "Supervised fine-tuning teaches the model to produce the answers in your dataset."
          : "DPO nudges the model toward answers people preferred and away from the ones they rejected. It uses your feedback, so run it after a few rounds of comparisons."}{" "}
        By default training continues from whatever the project is serving now.
      </p>

      <Card title="Data">
        <Field label={mode === "sft" ? "Dataset version" : "Preference data"}>
          <Select
            value={effectiveVersion}
            onChange={setVersionId}
            options={[
              ...(mode === "dpo" ? [{ value: "", label: `Feedback pairs collected in this app (${pairsReady} ready)` }] : []),
              ...versions.map((v) => ({ value: String(v.id), label: `v${v.id} · ${v.n_train} train / ${v.n_valid} valid · p95 ${v.token_stats.p95 ?? "?"} tokens` })),
            ]}
          />
        </Field>
        {mode === "sft" && !!ov?.checkpoints.length && (
          <div className="mt-3">
            <Field label="Start from">
              <Select
                value={startFrom}
                onChange={setStartFrom}
                options={[
                  { value: "current", label: "The model being served now (continue training)" },
                  { value: "base", label: "The untrained base model (fresh run, e.g. to compare settings)" },
                ]}
              />
            </Field>
          </div>
        )}
        {mode === "sft" && !versions.length && (
          <p className="mt-2 text-xs text-warn">
            No prepared SFT data yet. <Link className="underline" to={`/p/${projectId}/data`}>Prepare a dataset →</Link>
          </p>
        )}
        {mode === "dpo" && !effectiveVersion && pairsReady < 3 && (
          <p className="mt-2 text-xs text-warn">
            Need at least 3 preference pairs. <Link className="underline" to={`/p/${projectId}/feedback`}>Give feedback →</Link>
          </p>
        )}
      </Card>

      {presets.data && (
        <div className="grid grid-cols-3 gap-3">
          {(["safe", "balanced", "quality"] as const).map((name) => {
            const p = presets.data!.presets[name];
            return (
              <button
                key={name}
                onClick={() => setPresetName(name)}
                className={cx(
                  "rounded-xl border p-3 text-left transition",
                  presetName === name ? "border-accent bg-accent-soft" : "border-line bg-panel hover:border-line-strong",
                )}
              >
                <div className="flex items-center justify-between">
                  <span className="text-[13px] font-semibold capitalize">{name}</span>
                  <span className={cx("num text-xs", p.estimate.fits ? "text-good" : "text-bad")}>{fmt.gb(p.estimate.total_gb)}</span>
                </div>
                <div className="num mt-1 text-[11px] text-muted">
                  rank {p.config.lora_rank} · {p.config.num_layers === -1 ? "all" : p.config.num_layers} layers · bs {p.config.batch_size}
                  {p.config.grad_accumulation_steps > 1 && `×${p.config.grad_accumulation_steps}`}
                </div>
              </button>
            );
          })}
        </div>
      )}

      {cfg && (
        <Card title="Hyperparameters" subtitle={`Starting from the ${presetName} preset`}>
          <TrainControls value={cfg} onChange={setCfg} />
        </Card>
      )}

      {estimate && (
        <Card
          title="Memory estimate"
          subtitle={presets.data?.typical_len ? `From the data's p95 length (${presets.data.typical_len} tokens)` : "Assuming full-length sequences"}
          actions={
            estimate.iters != null && (
              <span className="num text-xs text-muted">
                {estimate.iters} iterations ≈ {estimate.epochs} epochs
              </span>
            )
          }
        >
          <MemoryBar est={estimate.estimate} />
        </Card>
      )}

      <ErrorNote error={launch.error} />
      <div className="flex gap-2">
        <Button variant="primary" disabled={!canLaunch} loading={launch.isPending} onClick={() => launch.mutate(false)}>
          Start {mode.toUpperCase()} run
        </Button>
        {launch.error && String((launch.error as Error).message).includes("exceeds") && (
          <Button variant="danger" onClick={() => launch.mutate(true)}>
            Launch anyway
          </Button>
        )}
      </div>
    </div>
  );
}

function RunView({ jobId, projectId }: { jobId: number; projectId: number }) {
  const { job, metrics, log, progress } = useLiveJob(jobId);
  const qc = useQueryClient();
  const cancel = useMutation({
    mutationFn: () => api.post(`/api/jobs/${jobId}/cancel`),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["job", jobId] }),
  });
  if (!job) return null;

  const train = metrics.filter((m) => m.split === "train");
  const val = metrics.filter((m) => m.split === "val");
  const last = train.at(-1)?.values;
  const lastVal = val.at(-1)?.values;
  const isDpo = job.kind === "dpo";
  const total = progress?.total || (job.result.total_iters as number) || 0;
  const current = progress?.current ?? train.at(-1)?.iteration ?? 0;
  const pct = total ? Math.min(100, (current / total) * 100) : 0;
  const itPerSec = last?.it_per_sec;
  const eta = itPerSec && total > current ? (total - current) / itPerSec : null;
  const cfg = (job.config.train ?? {}) as Partial<TrainConfig>;
  const active = job.status === "running" || job.status === "queued";

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center gap-3">
        <h1 className="text-xl font-semibold tracking-tight">
          {isDpo ? "DPO" : "SFT"} run · job {job.id}
        </h1>
        <StatusBadge status={job.status} />
        <span className="text-xs text-faint">{fmt.duration(job.started_at, job.finished_at)}</span>
        {active && (
          <Button size="sm" variant="danger" className="ml-auto" loading={cancel.isPending} onClick={() => cancel.mutate()}>
            Cancel
          </Button>
        )}
        {job.status === "succeeded" && (
          <Link to={`/p/${projectId}/playground`} className="ml-auto">
            <Button size="sm" variant="primary">
              Try it in the playground →
            </Button>
          </Link>
        )}
      </div>

      {job.status === "failed" && <ErrorNote error={job.error} />}
      {((job.result.warnings as { code: string; message: string }[] | undefined) ?? []).map((w) => (
        <div key={w.code} className="rounded-lg border border-warn/40 bg-warn-soft px-3 py-2 text-[13px] text-warn">
          <span className="font-semibold">{w.code.replace("_", " ")}:</span> {w.message}
        </div>
      ))}

      <div>
        <div className="mb-1 flex justify-between text-xs text-muted">
          <span className="num">
            iteration {current} / {total || "?"}
            {job.result.epochs != null && ` · ${String(job.result.epochs)} epochs`}
          </span>
          {eta != null && active && <span className="num">~{Math.ceil(eta / 60)} min left</span>}
        </div>
        <div className="h-1.5 overflow-hidden rounded-full bg-panel-2">
          <div className={cx("h-full transition-all", job.status === "failed" ? "bg-bad" : "bg-accent")} style={{ width: `${job.status === "succeeded" ? 100 : pct}%` }} />
        </div>
      </div>

      <div className="grid grid-cols-2 gap-3 md:grid-cols-5">
        <Stat label="Train loss" value={fmt.num(last?.loss)} />
        <Stat label="Val loss" value={fmt.num(lastVal?.loss)} sub={val.length > 1 ? `from ${fmt.num(val[0].values.loss)}` : undefined} />
        {isDpo ? (
          <Stat label="Reward accuracy" value={last?.accuracy != null ? `${Math.round(last.accuracy * 100)}%` : "–"} sub={`margin ${fmt.num(last?.margin, 2)}`} />
        ) : (
          <Stat label="Tokens / sec" value={last ? Math.round(last.tokens_per_sec) : "–"} />
        )}
        <Stat label="Peak memory" value={fmt.gb(last?.peak_mem_gb)} tone={last && last.peak_mem_gb > 10 ? "warn" : undefined} />
        <Stat label="Learning rate" value={last?.learning_rate != null ? last.learning_rate.toExponential(1) : "–"} />
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <Card title="Loss" subtitle="Validation diverging upward from training loss means overfitting">
          <MetricChart
            metrics={metrics}
            series={[
              { key: "loss", label: "train", split: "train", color: "var(--chart-1)" },
              { key: "loss", label: "validation", split: "val", color: "var(--chart-2)", dashed: true },
            ]}
          />
        </Card>
        {isDpo ? (
          <Card title="Rewards" subtitle="Chosen should rise above rejected">
            <MetricChart
              metrics={metrics}
              format={(v) => v.toFixed(2)}
              series={[
                { key: "chosen_reward", label: "chosen", split: "train", color: "var(--chart-3)" },
                { key: "rejected_reward", label: "rejected", split: "train", color: "var(--chart-1)" },
                { key: "margin", label: "margin", split: "train", color: "var(--chart-4)", dashed: true },
              ]}
            />
          </Card>
        ) : (
          <Card title="Learning rate">
            <MetricChart
              metrics={metrics}
              format={(v) => v.toExponential(1)}
              series={[{ key: "learning_rate", label: "learning rate", split: "train", color: "var(--chart-4)" }]}
            />
          </Card>
        )}
      </div>

      <Card title="Log" pad={false}>
        <div className="p-3">
          <JobLog lines={log} />
        </div>
      </Card>

      <Card title="Configuration">
        <div className="num grid grid-cols-2 gap-x-6 gap-y-1 text-xs md:grid-cols-4">
          {Object.entries(cfg)
            .filter(([, v]) => v !== null && typeof v !== "object")
            .map(([k, v]) => (
              <div key={k} className="flex justify-between gap-2 border-b border-line/60 py-1">
                <span className="text-faint">{k.replace(/_/g, " ")}</span>
                <span className="font-mono">{String(v)}</span>
              </div>
            ))}
        </div>
      </Card>
    </div>
  );
}
