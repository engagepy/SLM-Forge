import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router";

import { api, DPO_MIN_PAIRS, fmt, isServing, type Job, type Project, type Stage, STAGES } from "../api";
import { NextStage, PAGE, PageHeader } from "../components/Page";
import { ADVANCED_PAGE, stageDone, stageSkipped } from "../components/StageStepper";
import { useOverview, useProjectId, useStudio } from "../hooks";
import { STAGE_LABEL } from "./studio/bits";
import { Badge, Button, Card, cx, EmptyNote, Field, StatusBadge, TextArea } from "../ui";

export default function OverviewPage() {
  const projectId = useProjectId();
  const { data } = useOverview(projectId);
  const { data: snap } = useStudio(projectId);
  const jobs = useQuery({
    queryKey: ["jobs", projectId],
    queryFn: () => api.get<Job[]>(`/api/jobs?project_id=${projectId}&limit=12`),
  });
  if (!data || !snap) return null;
  const { project, counts, checkpoints } = data;

  // The Studio's stages, in its order, with its definition of done.
  const done = stageDone(snap);
  const sft = checkpoints.filter((c) => c.kind === "sft").length;
  const dpo = checkpoints.filter((c) => c.kind === "dpo").length;
  const best = snap.evals.length ? Math.max(...snap.evals.map((e) => e.mean)) : null;
  const tests = project.test_questions ?? [];
  const DETAIL: Record<Stage, string> = {
    goal: tests.length ? `${tests.length} test cases · system prompt ${project.system_prompt ? "set" : "not set"}` : "goal, system prompt and test set",
    data: `${counts.datasets} datasets · ${counts.dataset_versions} prepared`,
    model: project.base_model || "not chosen yet",
    train: `${sft} SFT run${sft === 1 ? "" : "s"}`,
    evaluate: best != null ? `${snap.evals.length} evaluations · best ${best.toFixed(1)} / 10` : `${counts.feedback} judgements`,
    refine: stageSkipped("refine", done, snap) ? "skipped (optional)" : `${dpo} DPO rounds · ${counts.pairs_ready} pairs ready (${DPO_MIN_PAIRS}+ to be worth it)`,
    export: snap.exports.length ? `${snap.exports.length} export${snap.exports.length === 1 ? "" : "s"}` : "fused, quantized, with a model card",
  };
  // "Now" is where the Tuner is, as in the Studio; finished projects have nothing left.
  const next = snap.completed ? undefined : snap.stage;

  return (
    <div className={PAGE}>
      <PageHeader title="Goal" actions={<NextStage stage="goal" />}>
        The goal, the system prompt and the pipeline: the same stages and ticks as the Studio, each opening its expert screen.
      </PageHeader>

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
            {STAGES.map((st, i) => {
              const skipped = stageSkipped(st, done, snap);
              return (
                <li key={st}>
                  <Link to={`/p/${projectId}/${ADVANCED_PAGE[st]}`} className="flex items-center gap-3 px-4 py-3 hover:bg-panel-2">
                    <span
                      className={cx(
                        "grid size-6 shrink-0 place-items-center rounded-full text-[11px] font-semibold",
                        done[st] ? "bg-good text-white" : st === next ? "bg-accent text-white" : "bg-panel-2 text-faint",
                      )}
                    >
                      {done[st] ? "✓" : skipped ? "–" : i + 1}
                    </span>
                    <span className="min-w-0 flex-1">
                      <span className={cx("block text-[13px]", done[st] && "text-good")}>{STAGE_LABEL[st]}</span>
                      <span className="block truncate text-xs text-faint">{DETAIL[st]}</span>
                    </span>
                    {done[st] && <Badge tone="good">✓ done</Badge>}
                    {st === next && !done[st] && <Badge tone="accent">now</Badge>}
                  </Link>
                </li>
              );
            })}
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
                            run {c.job_id}
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
              <EmptyNote className="p-4">No training runs yet.</EmptyNote>
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
              {!jobs.data?.length && <li className="px-4 py-3"><EmptyNote>Nothing has run yet.</EmptyNote></li>}
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
