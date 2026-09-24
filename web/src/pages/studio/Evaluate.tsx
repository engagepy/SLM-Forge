// The evaluate stage: scores per checkpoint and the user's own A/B judgements.
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api, type Comparison, type Evaluation, type Sample, type Snapshot } from "../../api";
import { Badge, Button, cx, TextArea } from "../../ui";
import { Empty } from "./bits";

export function EvaluateView({ s }: { s: Snapshot }) {
  const pending = s.comparisons.filter((c) => c.status === "pending");
  const byPrompt = new Map<string, Sample[]>();
  for (const x of s.samples) byPrompt.set(x.prompt, [...(byPrompt.get(x.prompt) ?? []), x]);
  const groups = [...byPrompt.entries()].slice(0, 6);
  if (!groups.length && !s.comparisons.length && !s.evals.length) return <Empty>The Tuner will score the model on your test questions here.</Empty>;
  return (
    <div className="space-y-4">
      {s.evals.length > 0 && <Scores evals={s.evals} />}
      {pending.length > 0 && (
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <Badge tone="accent">your turn</Badge>
            <span className="text-[13px]">Pick the better answer ({pending.length} left). This teaches the model your taste.</span>
          </div>
          <CompareTask key={pending[0].id} c={pending[0]} projectId={s.project.id} />
        </div>
      )}
      {groups.map(([prompt, answers]) => (
        <div key={prompt} className="rounded-lg border border-line p-3">
          <p className="text-[13px] font-medium">{prompt}</p>
          <div className={cx("mt-2 grid gap-2", answers.length > 1 && "md:grid-cols-2")}>
            {answers.slice(0, 2).map((a, i) => (
              <div key={i} className="rounded-md bg-panel-2/50 p-2">
                <Badge tone={a.target === "base" ? "neutral" : "good"}>{a.target === "base" ? "before training" : "trained"}</Badge>
                <p className="mt-1 whitespace-pre-wrap text-xs leading-relaxed text-muted">{a.text}</p>
              </div>
            ))}
          </div>
        </div>
      ))}
    </div>
  );
}

/** Scores per checkpoint, latest first: the before/after as numbers. */
function Scores({ evals }: { evals: Evaluation[] }) {
  const [open, setOpen] = useState<string | null>(null);
  const best = Math.max(...evals.map((e) => e.mean));
  const label = (e: Evaluation) => (e.checkpoint_id == null ? "before training" : `checkpoint ${e.checkpoint_id}`);
  return (
    <div className="space-y-1.5">
      {[...evals].reverse().map((e) => (
        <div key={e.id} className="rounded-lg border border-line">
          <button onClick={() => setOpen(open === e.id ? null : e.id)} className="flex w-full items-center gap-3 px-3 py-2 text-left">
            <span className={cx("num w-12 text-[15px] font-semibold", e.mean === best ? "text-good" : "text-fg")}>{e.mean.toFixed(1)}</span>
            <span className="text-[11px] text-faint">/ 10{e.exact_rate != null && ` · ${Math.round(e.exact_rate * 100)}% exact`}</span>
            <div className="h-1.5 flex-1 overflow-hidden rounded-full bg-panel-2">
              <div className={cx("h-full", e.mean === best ? "bg-good" : "bg-accent")} style={{ width: `${e.mean * 10}%` }} />
            </div>
            <Badge tone={e.checkpoint_id == null ? "neutral" : e.mean === best ? "good" : "info"}>{label(e)}</Badge>
            <span className={cx("text-faint transition", open === e.id && "rotate-90")}>›</span>
          </button>
          {open === e.id && (
            <ul className="space-y-2 border-t border-line px-3 py-2">
              {e.items.map((it) => (
                <li key={it.prompt} className="text-xs">
                  <div className="flex items-start gap-2">
                    <span className={cx("num w-5 shrink-0 font-semibold", it.score >= 7 ? "text-good" : it.score >= 4 ? "text-warn" : "text-bad")}>{it.score}</span>
                    <div className="min-w-0">
                      <p className="font-medium">{it.prompt}</p>
                      <p className="whitespace-pre-wrap text-muted">{it.answer}</p>
                      <p className="mt-0.5 text-faint">{it.reason}</p>
                    </div>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      ))}
    </div>
  );
}

function CompareTask({ c, projectId }: { c: Comparison; projectId: number }) {
  const qc = useQueryClient();
  const [choice, setChoice] = useState<string | null>(null);
  const [rewrite, setRewrite] = useState("");
  const [critique, setCritique] = useState("");
  const judge = useMutation({
    mutationFn: () => api.post(`/api/projects/${projectId}/studio/judge`, { comparison_id: c.id, choice, edited_answer: rewrite, critique }),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["studio", projectId] }),
  });
  const identical = c.a.trim() === c.b.trim();
  return (
    <div className="rounded-xl border border-accent/40 bg-accent-soft/30 p-3">
      <p className="text-[13px] font-medium">{c.prompt}</p>
      <div className="mt-2 grid gap-2 md:grid-cols-2">
        {(["a", "b"] as const).map((side) => (
          <button
            key={side}
            onClick={() => (setChoice(side), setRewrite(c[side]))}
            className={cx("rounded-lg border p-2 text-left text-xs leading-relaxed transition", choice === side ? "border-good bg-good-soft" : "border-line bg-panel hover:border-line-strong")}
          >
            <span className="font-semibold uppercase">{side}</span>
            <p className="mt-1 whitespace-pre-wrap">{c[side]}</p>
          </button>
        ))}
      </div>
      {identical && <p className="mt-2 text-[11px] text-warn">These are identical. Rewrite a better answer below, or skip with "About the same".</p>}
      <div className="mt-2 flex flex-wrap gap-2">
        <Button size="sm" variant={choice === "tie" ? "primary" : "secondary"} onClick={() => setChoice("tie")}>
          About the same
        </Button>
        <Button size="sm" variant={choice === "both_bad" ? "danger" : "secondary"} onClick={() => (setChoice("both_bad"), setRewrite(""))}>
          Both are bad
        </Button>
      </div>
      {choice && (
        <div className="mt-2 space-y-2">
          <TextArea rows={3} value={rewrite} onChange={(e) => setRewrite(e.target.value)} placeholder="Optional: write the ideal answer (the strongest signal you can give)" />
          <TextArea rows={1} value={critique} onChange={(e) => setCritique(e.target.value)} placeholder="Optional: what was wrong?" />
          <Button
            size="sm"
            variant="primary"
            loading={judge.isPending}
            disabled={choice === "both_bad" && !rewrite.trim() && !critique.trim()}
            onClick={() => judge.mutate()}
          >
            Save and next
          </Button>
          {judge.error && <p className="text-xs text-bad">{String((judge.error as Error).message)}</p>}
        </div>
      )}
    </div>
  );
}
