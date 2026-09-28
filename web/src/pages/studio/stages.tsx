// The stage cards: goal, model, data, train, refine, export.
import { Fragment } from "react";
import { fmt, isActive, JOB_KIND, runCommand, runLabel, type Snapshot } from "../../api";
import { MetricChart } from "../../components/Charts";
import { runProgress, useLiveJob } from "../../hooks";
import { Badge, CodeBlock, cx, LinkButton, MemoryBar, ProgressBar, SectionLabel, StatusBadge } from "../../ui";
import { Empty, Fact, Progress, BeforeResetBadge } from "./bits";
import { ExportActions } from "./ExportActions";

export function GoalView({ s }: { s: Snapshot }) {
  return (
    <dl className="space-y-2 text-[13px]">
      <div>
        <dt><SectionLabel>Model purpose</SectionLabel></dt>
        <dd>{s.project.goal || "The Tuner will fill this in as you talk."}</dd>
      </div>
      {s.project.system_prompt && (
        <div>
          <dt><SectionLabel>System prompt</SectionLabel></dt>
          <dd className="font-mono text-xs text-muted">{s.project.system_prompt}</dd>
        </div>
      )}
      {Object.keys(s.project.plan ?? {}).length > 0 && (
        <div>
          <dt><SectionLabel>The Tuner's plan (say so in the chat to change it)</SectionLabel></dt>
          <dd>
            <dl className="mt-1 grid grid-cols-[auto_minmax(0,1fr)] gap-x-3 gap-y-0.5 text-xs">
              {Object.entries(s.project.plan).map(([k, v]) => (
                <Fragment key={k}>
                  <dt className="text-faint">{k.replace(/_/g, " ")}</dt>
                  <dd className="text-muted">{v}</dd>
                </Fragment>
              ))}
            </dl>
          </dd>
        </div>
      )}
      {s.project.test_questions?.length > 0 && (
        <div>
          <dt>
            <SectionLabel>
              Test set · {s.project.test_questions.length} cases, {s.project.test_questions.filter((q) => q.expected).length} with an expected output
            </SectionLabel>
          </dt>
          <dd>
            <ol className="mt-1 list-decimal space-y-0.5 pl-5 text-xs text-muted">
              {s.project.test_questions.map((q) => (
                <li key={q.input} className={cx(q.kind === "should-not" && "italic")} title={q.expected ? `Expected: ${q.expected}` : undefined}>
                  {q.input}
                  {q.kind === "should-not" && <span className="ml-1 text-faint">(should return nothing)</span>}
                </li>
              ))}
            </ol>
          </dd>
        </div>
      )}
    </dl>
  );
}

export function ModelView({ s }: { s: Snapshot }) {
  const download = s.jobs.find((j) => j.kind === "download" && isActive(j));
  if (!s.model) return <Empty>The Tuner is choosing a base model that fits this Mac.</Empty>;
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-2">
        <a className="font-mono text-[13px] hover:text-accent" href={`https://huggingface.co/${s.model.repo_id}`} target="_blank" rel="noreferrer">
          {s.model.repo_id}
        </a>
        {s.model.downloaded ? <Badge tone="good">on this Mac</Badge> : download ? <Badge tone="info">downloading…</Badge> : <Badge>not downloaded</Badge>}
      </div>
      {s.model.params != null && (
        <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
          <Fact label="Parameters" value={fmt.params(s.model.params)} />
          <Fact label="Precision" value={`${s.model.bits}-bit`} />
          <Fact label="On disk" value={fmt.gb(s.model.size_gb)} />
          <Fact label="Layers" value={String(s.model.layers)} />
        </div>
      )}
      {s.model.inference && (
        <div>
          <div className="mb-1 text-[11px] text-faint">Memory to run it (4k context) vs. this Mac's budget</div>
          <MemoryBar est={s.model.inference} />
        </div>
      )}
      {download && <Progress label="Downloading" />}
    </div>
  );
}

export function DataView({ s }: { s: Snapshot }) {
  const prepping = s.jobs.find((j) => ["import_dataset", "prepare_dataset", "synthesize"].includes(j.kind) && isActive(j));
  if (!s.datasets.length && !s.versions.length && !prepping) return <Empty>The Tuner is looking for training data that matches the goal.</Empty>;
  return (
    <div className="space-y-3">
      {s.datasets.map((d) => (
        <div key={d.id} className="flex items-center gap-2 text-[13px]">
          <span className="text-faint">⬇</span>
          <span className="font-mono text-xs">{d.source_ref || d.name}</span>
          <span className="text-xs text-muted">{fmt.compact(d.n_rows)} rows</span>
          {d.license && <Badge tone={/mit|apache|cc-by|cc0/i.test(d.license) ? "good" : "neutral"}>{d.license}</Badge>}
        </div>
      ))}
      {prepping && <Progress label={JOB_KIND[prepping.kind]?.doing ?? "Working"} />}
      {s.versions.map((v) => {
        const dropped = Object.entries(v.cleaning_report?.dropped ?? {});
        return (
          <div key={v.id} className="rounded-lg border border-line bg-panel-2/40 p-3">
            <div className="flex items-center gap-2 text-[13px]">
              <span className="font-medium">Training set v{v.id}</span>
              <Badge tone={v.kind === "dpo" ? "accent" : "info"}>{v.kind === "dpo" ? "preferences" : "examples"}</Badge>
            </div>
            <div className="mt-2 grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
              <Fact label="Train" value={v.n_train.toLocaleString()} />
              <Fact label="Validation" value={v.n_valid.toLocaleString()} />
              <Fact label="Typical length" value={v.token_stats?.p50 ? `${v.token_stats.p50} tok` : "–"} />
              <Fact label="Long (p95)" value={v.token_stats?.p95 ? `${v.token_stats.p95} tok` : "–"} />
            </div>
            {dropped.length > 0 && (
              <p className="mt-2 text-[11px] text-faint">
                Cleaned out: {dropped.map(([k, n]) => `${n} ${k.replace(/_/g, " ")}`).join(", ")}
              </p>
            )}
          </div>
        );
      })}
    </div>
  );
}

export function TrainView({ s }: { s: Snapshot }) {
  const runs = s.jobs.filter((j) => j.kind === "sft" || j.kind === "dpo");
  const current = runs[0];
  if (!current) return <Empty>Training starts once the data is ready.</Empty>;
  return (
    <div className="space-y-4">
      <LiveRun jobId={current.id} />
      {runs.length > 1 && (
        <div className="space-y-1">
          <SectionLabel>Earlier runs</SectionLabel>
          {runs.slice(1, 6).map((j) => (
            <div key={j.id} className="flex items-center gap-2 text-xs text-muted">
              <Badge tone={j.kind === "dpo" ? "accent" : "info"}>{j.kind.toUpperCase()}</Badge>
              run {j.id}
              <span className="num">val loss {String((j.result.metrics as Record<string, number> | undefined)?.val_loss ?? "–")}</span>
              <StatusBadge status={j.status} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function LiveRun({ jobId }: { jobId: number }) {
  const live = useLiveJob(jobId);
  const { job, metrics } = live;
  if (!job) return null;
  const { current: cur, total, pct, minutesLeft: eta, last, lastVal, warnings } = runProgress(live);
  const isDpo = job.kind === "dpo";
  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2 text-[13px]">
        <span className="font-medium">{isDpo ? "Preference round" : "Fine-tuning"} · run {job.id}</span>
        <StatusBadge status={job.status} />
        <span className="ml-auto num text-xs text-muted">
          {cur}/{total || "?"} steps{eta != null && job.status === "running" ? ` · ~${eta} min left` : ""}
        </span>
      </div>
      <ProgressBar pct={pct} failed={job.status === "failed"} />
      <div className="grid grid-cols-2 gap-2 text-xs sm:grid-cols-4">
        <Fact label="Train loss" value={fmt.num(last?.loss)} />
        <Fact label="Val loss" value={fmt.num(lastVal?.loss)} />
        {isDpo ? <Fact label="Prefers your picks" value={last?.accuracy != null ? `${Math.round(last.accuracy * 100)}%` : "–"} /> : <Fact label="Tokens/sec" value={last ? String(Math.round(last.tokens_per_sec)) : "–"} />}
        <Fact label="Peak memory" value={fmt.gb(last?.peak_mem_gb)} />
      </div>
      <MetricChart
        height={180}
        metrics={metrics}
        series={[
          { key: "loss", label: "train", split: "train", color: "var(--chart-1)" },
          { key: "loss", label: "validation", split: "val", color: "var(--chart-2)", dashed: true },
        ]}
      />
      {warnings.map((w) => (
        <p key={w.code} className="rounded-lg border border-warn/40 bg-warn-soft px-3 py-2 text-xs text-warn">
          <strong>{w.code.replace("_", " ")}:</strong> {w.message}
        </p>
      ))}
      {job.status === "failed" && <p className="rounded-lg bg-bad-soft px-3 py-2 text-xs text-bad">{job.error}</p>}
    </div>
  );
}

const VERDICT: Record<string, { label: string; tone: "good" | "warn" | "bad" | "neutral" }> = {
  a: { label: "A better", tone: "good" },
  b: { label: "B better", tone: "good" },
  tie: { label: "about the same", tone: "neutral" },
  both_bad: { label: "both wrong", tone: "bad" },
};

export function RefineView({ s }: { s: Snapshot }) {
  const dpo = s.checkpoints.filter((c) => c.kind === "dpo");
  const reviews = s.comparisons.filter((c) => c.judge === "ai").slice(-8).reverse();
  return (
    <div className="space-y-3">
      {reviews.length > 0 && (
        <div className="space-y-2">
          <SectionLabel>Reviewed by GPT-6 · each verdict becomes training signal</SectionLabel>
          {reviews.map((c) => (
            <details key={c.id} className="group rounded-lg border border-line p-2.5">
              <summary className="flex cursor-pointer list-none items-center gap-2 text-xs">
                <Badge tone={VERDICT[c.choice ?? "tie"]?.tone ?? "neutral"}>{VERDICT[c.choice ?? "tie"]?.label ?? c.choice}</Badge>
                <span className="min-w-0 flex-1 truncate text-fg">{c.prompt}</span>
                <span className="text-faint transition group-open:rotate-90">›</span>
              </summary>
              <div className="mt-2 space-y-1.5 text-xs leading-relaxed">
                {c.critique && <p className="text-muted">{c.critique}</p>}
                {c.ideal && (
                  <p className="rounded-md bg-good-soft px-2 py-1.5 text-fg">
                    <span className="font-medium text-good">Ideal answer: </span>
                    {c.ideal}
                  </p>
                )}
              </div>
            </details>
          ))}
        </div>
      )}
      <div className="grid grid-cols-3 gap-2 text-xs">
        <Fact label="Your judgements" value={String(s.feedback.judgements)} />
        <Fact label="Preference pairs ready" value={String(s.feedback.pairs_ready)} />
        <Fact label="Preference rounds" value={String(dpo.length)} />
      </div>
      {dpo.map((c) => (
        <p key={c.id} className="text-xs text-muted">
          {runLabel("dpo", c.job_id)}: prefers your picks {c.metrics.reward_accuracy != null ? `${Math.round(c.metrics.reward_accuracy * 100)}%` : "–"} of the time
        </p>
      ))}
    </div>
  );
}

export function ExportView({ s }: { s: Snapshot }) {
  if (!s.exports.length) return <Empty>When you're happy, the Tuner packages the model here (it asks you first).</Empty>;
  return (
    <div className="space-y-3">
      {s.exports.map((e) => (
        <div key={e.path} className="space-y-1.5">
          <div className="flex flex-wrap items-center gap-2 text-[13px]">
            <span className="font-medium">{e.path.split("/").pop()}</span>
            <Badge>{fmt.gb(e.size_gb)}</Badge>
            <Badge tone="good">runs on {e.min_ram_gb} GB+ Macs</Badge>
            <BeforeResetBadge e={e} />
            <LinkButton to={`/p/${s.project.id}/try?export=${e.job_id}`} variant="good" size="sm" className="ml-auto">
              ▶ Try it
            </LinkButton>
          </div>
          <CodeBlock text={runCommand(e, s.project.system_prompt)} />
          {!e.system_prompt_built_in && s.project.system_prompt && (
            <p className="text-[11px] text-faint">This export needs the system prompt passed in; newer exports build it in.</p>
          )}
          <ExportActions s={s} e={e} />
        </div>
      ))}
    </div>
  );
}
