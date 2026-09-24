import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router";

import { api, fmt, isServing, type Job, type Project } from "../api";
import { useOverview, useProjectId } from "../hooks";
import { Badge, Button, Card, cx, Field, LinkButton, StatusBadge, TextArea } from "../ui";

export default function OverviewPage() {
  const projectId = useProjectId();
  const { data } = useOverview(projectId);
  const jobs = useQuery({
    queryKey: ["jobs", projectId],
    queryFn: () => api.get<Job[]>(`/api/jobs?project_id=${projectId}&limit=12`),
  });
  if (!data) return null;
  const { project, counts, checkpoints } = data;

  const steps = [
    { label: "Choose and download a base model", done: data.base_model_downloaded, to: "model", detail: project.base_model },
    { label: "Get training data", done: counts.dataset_versions > 0, to: "data", detail: `${counts.datasets} datasets · ${counts.dataset_versions} prepared` },
    { label: "Fine-tune (SFT)", done: checkpoints.some((c) => c.kind === "sft"), to: "train", detail: `${checkpoints.filter((c) => c.kind === "sft").length} runs` },
    { label: "Compare answers and give feedback", done: counts.feedback >= 8, to: "feedback", detail: `${counts.feedback} judgements · ${counts.pairs_ready} pairs ready` },
    { label: "Preference-tune (DPO) on your feedback", done: checkpoints.some((c) => c.kind === "dpo"), to: "train", detail: `${checkpoints.filter((c) => c.kind === "dpo").length} rounds` },
    { label: "Export the model", done: false, to: "export", detail: "fused, quantized, with a model card" },
  ];
  const next = steps.find((s) => !s.done);

  return (
    <div className="mx-auto max-w-5xl space-y-5 p-6">
      <div className="flex items-start justify-between gap-4">
        <div>
          <h1 className="text-xl font-semibold tracking-tight">{project.name}</h1>
          <p className="mt-1 max-w-2xl text-[13px] text-muted">{project.goal}</p>
        </div>
        {next && (
          <LinkButton to={`/p/${projectId}/${next.to}`} variant="primary">Next: {next.label.split(" (")[0].toLowerCase()} →</LinkButton>
        )}
      </div>

      {(counts.pending_proposals > 0 || counts.awaiting_review > 0) && (
        <div className="flex flex-wrap gap-3">
          {counts.pending_proposals > 0 && (
            <Link to={`/p/${projectId}/agents`} className="rounded-lg border border-accent/40 bg-accent-soft px-3 py-2 text-[13px] text-accent">
              {counts.pending_proposals} agent proposal{counts.pending_proposals > 1 ? "s" : ""} waiting for approval →
            </Link>
          )}
          {counts.awaiting_review > 0 && (
            <Link to={`/p/${projectId}/feedback`} className="rounded-lg border border-warn/40 bg-warn-soft px-3 py-2 text-[13px] text-warn">
              {counts.awaiting_review} synthetic examples to review →
            </Link>
          )}
        </div>
      )}

      <div className="grid gap-5 md:grid-cols-5">
        <Card title="Pipeline" className="md:col-span-3" pad={false}>
          <ol className="divide-y divide-line">
            {steps.map((s, i) => (
              <li key={i}>
                <Link to={`/p/${projectId}/${s.to}`} className="flex items-center gap-3 px-4 py-3 hover:bg-panel-2">
                  <span
                    className={cx(
                      "grid size-6 shrink-0 place-items-center rounded-full text-[11px] font-semibold",
                      s.done ? "bg-good-soft text-good" : s === next ? "bg-accent text-white" : "bg-panel-2 text-faint",
                    )}
                  >
                    {s.done ? "✓" : i + 1}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block text-[13px]">{s.label}</span>
                    {s.detail && <span className="block truncate text-xs text-faint">{s.detail}</span>}
                  </span>
                </Link>
              </li>
            ))}
          </ol>
        </Card>

        <div className="space-y-5 md:col-span-2">
          <Card title="Training history" subtitle="Runs build on each other unless marked “from base”" pad={false}>
            {checkpoints.length ? (
              <ol className="space-y-0 p-4">
                <li className="flex gap-2 text-xs">
                  <span className="mt-1 size-2 rounded-full bg-faint" />
                  <span className="font-mono text-muted">{project.base_model}</span>
                </li>
                {checkpoints.map((c, i) => {
                  const serving = isServing(project, c);
                  // No parent after the first checkpoint means a fresh run branched from the base model.
                  const fromBase = i > 0 && c.parent_id == null;
                  return (
                    <li key={c.id} className="flex gap-2 border-l border-line pl-3 ml-0.75 pt-3 text-xs">
                      <span className="min-w-0 flex-1">
                        <span className="flex items-center gap-1.5">
                          <Badge tone={c.kind === "dpo" ? "accent" : "info"}>{c.kind.toUpperCase()}</Badge>
                          <Link className="text-muted hover:text-fg" to={`/p/${projectId}/train/${c.job_id}`}>
                            job {c.job_id}
                          </Link>
                          {fromBase && <Badge>from base</Badge>}
                          {serving && <Badge tone="good">serving</Badge>}
                        </span>
                        <span className="num mt-1 block text-faint">
                          {Object.entries(c.metrics)
                            .filter(([k]) => ["val_loss", "reward_accuracy", "peak_mem_gb"].includes(k))
                            .map(([k, v]) => `${k.replace(/_/g, " ")} ${v}`)
                            .join(" · ")}
                        </span>
                      </span>
                    </li>
                  );
                })}
              </ol>
            ) : (
              <p className="p-4 text-xs text-muted">No training runs yet.</p>
            )}
          </Card>

          <Card title="Recent jobs" pad={false}>
            <ul className="divide-y divide-line">
              {jobs.data?.map((j) => (
                <li key={j.id} className="flex items-center gap-2 px-4 py-2 text-xs">
                  <span className="num w-8 text-faint">#{j.id}</span>
                  <span className="flex-1">{j.kind.replace("_", " ")}</span>
                  <span className="text-faint">{fmt.ago(j.created_at)}</span>
                  <StatusBadge status={j.status} />
                </li>
              ))}
              {!jobs.data?.length && <li className="px-4 py-3 text-xs text-muted">Nothing has run yet.</li>}
            </ul>
          </Card>
        </div>
      </div>

      <ProjectSettings project={project} />
    </div>
  );
}

function ProjectSettings({ project }: { project: Project }) {
  const qc = useQueryClient();
  const [goal, setGoal] = useState(project.goal);
  const [system, setSystem] = useState(project.system_prompt);
  const save = useMutation({
    mutationFn: () => api.patch(`/api/projects/${project.id}`, { goal, system_prompt: system }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["overview", project.id] }),
  });
  const dirty = goal !== project.goal || system !== project.system_prompt;
  return (
    <Card title="Goal and system prompt" subtitle="Agents use the goal; the system prompt goes into training data and chat.">
      <div className="grid gap-3 md:grid-cols-2">
        <Field label="Goal">
          <TextArea rows={3} value={goal} onChange={(e) => setGoal(e.target.value)} />
        </Field>
        <Field label="System prompt">
          <TextArea rows={3} value={system} onChange={(e) => setSystem(e.target.value)} />
        </Field>
      </div>
      <Button className="mt-3" size="sm" disabled={!dirty} loading={save.isPending} onClick={() => save.mutate()}>
        Save
      </Button>
    </Card>
  );
}
