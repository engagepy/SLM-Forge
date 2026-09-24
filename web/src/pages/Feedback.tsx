import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { Link } from "react-router";

import { api, DEFAULT_SAMPLING, postStream, type PreferencePair, type SamplingParams, type SftExample } from "../api";
import SamplingControls from "../components/SamplingControls";
import { useOverview, useProjectId, waitForJob } from "../hooks";
import { Badge, Button, Card, Collapsible, cx, Empty, ErrorNote, Field, Spinner, Stat, TextArea } from "../ui";

type Choice = "a" | "b" | "tie" | "both_bad";

const SIDE_DEFAULTS: Record<"a" | "b", SamplingParams> = {
  a: { ...DEFAULT_SAMPLING, temperature: 0.7 },
  b: { ...DEFAULT_SAMPLING, temperature: 1.0 },
};

export default function FeedbackPage() {
  const projectId = useProjectId();
  const { data: ov } = useOverview(projectId);
  return (
    <div className="mx-auto max-w-6xl space-y-5 p-6">
      <div>
        <h1 className="text-xl font-semibold tracking-tight">Human feedback</h1>
        <p className="mt-1 max-w-3xl text-[13px] text-muted">
          Ask something, compare two answers and pick the better one. Each pick becomes a preference pair for DPO. Rewriting an
          answer also creates a supervised example. The Observer reads your critiques and decides what to generate next.
        </p>
      </div>
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_300px]">
        <Compare projectId={projectId} systemPrompt={ov?.project.system_prompt ?? ""} />
        <SidePanel projectId={projectId} />
      </div>
      <ReviewQueue projectId={projectId} />
    </div>
  );
}

function Compare({ projectId, systemPrompt }: { projectId: number; systemPrompt: string }) {
  const qc = useQueryClient();
  const [prompt, setPrompt] = useState("");
  const [params, setParams] = useState(SIDE_DEFAULTS);
  const [cands, setCands] = useState<{ a: string; b: string }>({ a: "", b: "" });
  const [phase, setPhase] = useState<"idle" | "generating" | "judging">("idle");
  const [choice, setChoice] = useState<Choice | null>(null);
  const [edited, setEdited] = useState("");
  const [critique, setCritique] = useState("");
  const [error, setError] = useState<unknown>(null);
  const [lastResult, setLastResult] = useState<string | null>(null);

  async function generate() {
    if (!prompt.trim()) return;
    setCands({ a: "", b: "" });
    setChoice(null);
    setEdited("");
    setCritique("");
    setError(null);
    setPhase("generating");
    // Fresh random seeds each round unless the user pinned one.
    const withSeed = (p: SamplingParams) => ({ ...p, seed: p.seed ?? Math.floor(Math.random() * 1e9) });
    try {
      await postStream(
        `/api/projects/${projectId}/compare`,
        { prompt, params_a: withSeed(params.a), params_b: withSeed(params.b) },
        (ev) => {
          if (ev.type === "token") {
            const side = ev.candidate as "a" | "b";
            setCands((c) => ({ ...c, [side]: c[side] + (ev.text as string) }));
          } else if (ev.type === "error") setError(new Error(String(ev.message)));
        },
      );
      setPhase("judging");
    } catch (e) {
      setError(e);
      setPhase("idle");
    }
  }

  const submit = useMutation({
    mutationFn: () =>
      api.post<{ preference_pairs: number; sft_examples: number }>(`/api/projects/${projectId}/feedback`, {
        prompt,
        system: systemPrompt,
        candidate_a: cands.a,
        candidate_b: cands.b,
        params_a: params.a,
        params_b: params.b,
        choice,
        edited_answer: edited.trim() === (choice === "a" ? cands.a : choice === "b" ? cands.b : "").trim() ? "" : edited,
        critique,
      }),
    onSuccess: (r) => {
      setLastResult(`Saved: ${r.preference_pairs} preference pair${r.preference_pairs === 1 ? "" : "s"}${r.sft_examples ? `, ${r.sft_examples} SFT example` : ""}.`);
      setPhase("idle");
      setPrompt("");
      setCands({ a: "", b: "" });
      setChoice(null);
      qc.invalidateQueries({ queryKey: ["overview", projectId] });
    },
  });

  const pick = (c: Choice) => {
    setChoice(c);
    if (c === "a" || c === "b") setEdited(cands[c]);
    else if (!edited) setEdited("");
  };
  const needsText = choice === "both_bad" && !edited.trim() && !critique.trim();

  return (
    <Card title="Compare" pad={false}>
      <div className="space-y-3 p-4">
        <div className="flex gap-2">
          <TextArea
            rows={2}
            className="flex-1"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && (e.metaKey || e.ctrlKey) && generate()}
            placeholder="A prompt your model should handle well…"
          />
          <Button variant="primary" loading={phase === "generating"} onClick={generate} disabled={!prompt.trim()}>
            Generate A / B
          </Button>
        </div>
        <Collapsible title="Sampling for each side">
          <div className="grid gap-5 md:grid-cols-2">
            {(["a", "b"] as const).map((s) => (
              <div key={s}>
                <div className="mb-2 text-xs font-semibold uppercase text-muted">Side {s}</div>
                <SamplingControls compact value={params[s]} onChange={(v) => setParams({ ...params, [s]: v })} />
              </div>
            ))}
          </div>
        </Collapsible>
        {lastResult && phase === "idle" && <p className="text-xs text-good">{lastResult}</p>}
        <ErrorNote error={error} />
      </div>

      {phase !== "idle" && (
        <div className="border-t border-line p-4">
          <div className="grid gap-3 md:grid-cols-2">
            {(["a", "b"] as const).map((s) => (
              <button
                key={s}
                disabled={phase !== "judging"}
                onClick={() => pick(s)}
                className={cx(
                  "rounded-xl border p-3 text-left transition",
                  choice === s ? "border-good bg-good-soft" : choice && choice !== "tie" ? "border-line opacity-60" : "border-line hover:border-line-strong",
                )}
              >
                <div className="mb-1.5 flex items-center gap-2 text-xs">
                  <span className="font-semibold uppercase">{s}</span>
                  <span className="num text-faint">t={params[s].temperature}</span>
                  {choice === s && <Badge tone="good">preferred</Badge>}
                </div>
                <div className="whitespace-pre-wrap text-[13px] leading-relaxed">{cands[s] || <Spinner />}</div>
              </button>
            ))}
          </div>

          {phase === "judging" && cands.a.trim() === cands.b.trim() && (
            <p className="mt-3 rounded-lg border border-warn/40 bg-warn-soft px-3 py-2 text-xs text-warn">
              Both answers are identical, so picking one teaches nothing. Raise side B's temperature or regenerate. You can
              still write a better answer below.
            </p>
          )}
          {phase === "judging" && (
            <div className="mt-4 space-y-3">
              <div className="flex flex-wrap gap-2">
                <Button variant={choice === "a" ? "good" : "secondary"} onClick={() => pick("a")}>
                  A is better
                </Button>
                <Button variant={choice === "b" ? "good" : "secondary"} onClick={() => pick("b")}>
                  B is better
                </Button>
                <Button variant={choice === "tie" ? "primary" : "secondary"} onClick={() => pick("tie")}>
                  About the same
                </Button>
                <Button variant={choice === "both_bad" ? "danger" : "secondary"} onClick={() => pick("both_bad")}>
                  Both are bad
                </Button>
              </div>
              {choice && (
                <>
                  <Field
                    label={choice === "both_bad" ? "Write the answer it should have given" : "Improve the answer (optional)"}
                    hint="Edits become supervised examples, the strongest signal you can give"
                  >
                    <TextArea rows={4} value={edited} onChange={(e) => setEdited(e.target.value)} />
                  </Field>
                  <Field label="Critique" hint="What was wrong? The Observer reads these">
                    <TextArea rows={2} value={critique} onChange={(e) => setCritique(e.target.value)} placeholder="Too long; missed the temperature; wrong tone…" />
                  </Field>
                  <ErrorNote error={submit.error} />
                  <Button variant="primary" disabled={needsText} loading={submit.isPending} onClick={() => submit.mutate()}>
                    Save feedback
                  </Button>
                  {needsText && <span className="ml-2 text-xs text-faint">Write a better answer or a critique.</span>}
                </>
              )}
            </div>
          )}
        </div>
      )}
    </Card>
  );
}

function SidePanel({ projectId }: { projectId: number }) {
  const { data: ov } = useOverview(projectId);
  const qc = useQueryClient();
  const [status, setStatus] = useState<string | null>(null);
  const observe = useMutation({
    mutationFn: async () => {
      const { job_id } = await api.post<{ job_id: number }>(`/api/projects/${projectId}/agents/observe`, { use_llm: true });
      const job = await waitForJob(job_id, (j) => setStatus(j.status));
      if (job.status !== "succeeded") throw new Error(job.error || `Observer ${job.status}`);
      return job.result as { assessment?: string; notes?: string[]; rule_proposals?: string[]; observed?: number };
    },
    onSettled: () => {
      setStatus(null);
      qc.invalidateQueries({ queryKey: ["overview", projectId] });
      qc.invalidateQueries({ queryKey: ["proposals", projectId] });
    },
  });
  const c = ov?.counts;
  return (
    <div className="space-y-4">
      <div className="grid grid-cols-2 gap-2">
        <Stat label="Judgements" value={c?.feedback ?? 0} />
        <Stat label="Pairs ready" value={c?.pairs_ready ?? 0} tone={(c?.pairs_ready ?? 0) >= 8 ? "good" : undefined} sub="for DPO" />
        <Stat label="SFT examples" value={c?.sft_ready ?? 0} sub="ready" />
        <Stat label="To review" value={c?.awaiting_review ?? 0} tone={c?.awaiting_review ? "warn" : undefined} />
      </div>
      <Card title="Observer" subtitle="Reads new feedback and proposes the next step">
        <Button className="w-full" variant="primary" loading={observe.isPending} onClick={() => observe.mutate()}>
          {observe.isPending ? `Observing (${status ?? "queued"})…` : "Analyse feedback"}
        </Button>
        <ErrorNote error={observe.error} />
        {observe.data && (
          <div className="mt-3 space-y-2 text-[13px]">
            {observe.data.assessment && <p className="leading-relaxed">{observe.data.assessment}</p>}
            {!!observe.data.notes?.length && (
              <ul className="list-disc space-y-1 pl-4 text-xs text-muted">
                {observe.data.notes.map((n, i) => (
                  <li key={i}>{n}</li>
                ))}
              </ul>
            )}
            <Link to={`/p/${projectId}/agents`} className="block text-xs text-info hover:underline">
              See proposals →
            </Link>
          </div>
        )}
      </Card>
      {(c?.pairs_ready ?? 0) >= 3 && (
        <Link to={`/p/${projectId}/train`}>
          <Button className="w-full">Run a DPO round now →</Button>
        </Link>
      )}
    </div>
  );
}

function ReviewQueue({ projectId }: { projectId: number }) {
  const qc = useQueryClient();
  const q = useQuery({
    queryKey: ["examples", projectId],
    queryFn: () => api.get<{ pairs: PreferencePair[]; sft: SftExample[] }>(`/api/projects/${projectId}/examples?status=pending`),
  });
  const review = useMutation({
    mutationFn: (body: { pair_ids?: number[]; sft_ids?: number[]; approve: boolean }) => api.post(`/api/projects/${projectId}/examples/review`, body),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["examples", projectId] });
      qc.invalidateQueries({ queryKey: ["overview", projectId] });
    },
  });
  const pairs = q.data?.pairs ?? [];
  const sft = q.data?.sft ?? [];
  if (!pairs.length && !sft.length) {
    return (
      <Card title="Synthetic examples to review">
        <Empty title="Nothing to review">When the Synth agent generates examples from your feedback, they wait here for approval before training.</Empty>
      </Card>
    );
  }
  return (
    <Card
      title="Synthetic examples to review"
      subtitle="Nothing generated reaches training until you approve it"
      actions={
        <>
          <Button size="sm" variant="good" loading={review.isPending} onClick={() => review.mutate({ pair_ids: pairs.map((p) => p.id), sft_ids: sft.map((s) => s.id), approve: true })}>
            Approve all ({pairs.length + sft.length})
          </Button>
          <Button size="sm" variant="ghost" onClick={() => review.mutate({ pair_ids: pairs.map((p) => p.id), sft_ids: sft.map((s) => s.id), approve: false })}>
            Discard all
          </Button>
        </>
      }
      pad={false}
    >
      <ul className="divide-y divide-line">
        {pairs.map((p) => (
          <li key={`p${p.id}`} className="grid gap-3 p-4 md:grid-cols-[minmax(0,1fr)_minmax(0,1fr)_minmax(0,1fr)_auto]">
            <div className="text-[13px]">
              <Badge tone="accent">pair</Badge>
              <p className="mt-1 whitespace-pre-wrap">{p.prompt}</p>
            </div>
            <div className="rounded-lg bg-good-soft p-2 text-xs whitespace-pre-wrap">{p.chosen}</div>
            <div className="rounded-lg bg-bad-soft p-2 text-xs whitespace-pre-wrap">{p.rejected}</div>
            <ReviewButtons onApprove={() => review.mutate({ pair_ids: [p.id], approve: true })} onReject={() => review.mutate({ pair_ids: [p.id], approve: false })} />
          </li>
        ))}
        {sft.map((s) => (
          <li key={`s${s.id}`} className="grid gap-3 p-4 md:grid-cols-[minmax(0,1fr)_minmax(0,2fr)_auto]">
            <div className="text-[13px]">
              <Badge tone="info">SFT</Badge>
              <p className="mt-1 whitespace-pre-wrap">{s.messages.find((m) => m.role === "user")?.content}</p>
            </div>
            <div className="rounded-lg bg-panel-2 p-2 text-xs whitespace-pre-wrap">{s.messages.find((m) => m.role === "assistant")?.content}</div>
            <ReviewButtons onApprove={() => review.mutate({ sft_ids: [s.id], approve: true })} onReject={() => review.mutate({ sft_ids: [s.id], approve: false })} />
          </li>
        ))}
      </ul>
    </Card>
  );
}

function ReviewButtons({ onApprove, onReject }: { onApprove: () => void; onReject: () => void }) {
  return (
    <div className="flex gap-1 md:flex-col">
      <Button size="sm" variant="good" onClick={onApprove}>
        Keep
      </Button>
      <Button size="sm" variant="ghost" onClick={onReject}>
        Drop
      </Button>
    </div>
  );
}
