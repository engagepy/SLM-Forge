// The card for a proposed run or spend: the only way work starts (or a plain "yes" in the chat).
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";
import { api, type PendingAction } from "../../api";
import { invalidate } from "../../hooks";
import { Badge, Button, ErrorNote, TextArea } from "../../ui";

const ACTION_ICON: Record<PendingAction["kind"], string> = { model: "◆", sft: "▲", dpo: "▲", export: "⬇", synthesize: "✎", review: "⚖", import: "⇣", evaluate: "★", scout: "⌕", prep: "✂" };

/** The Tuner's proposed run. It only starts from here (or a plain "yes" in the chat). */
export function ConfirmCard({ projectId, action }: { projectId: number; action: PendingAction }) {
  const qc = useQueryClient();
  const [why, setWhy] = useState("");
  const [asking, setAsking] = useState(false);
  const decide = useMutation({
    mutationFn: (go: boolean) =>
      api.post(`/api/projects/${projectId}/studio/${go ? "confirm" : "decline"}`, { action_id: action.id, reason: why }),
    onSuccess: () => {
      setAsking(false);
      setWhy("");
    },
    onSettled: () => invalidate(qc, "studio", "sessions"),
  });
  const facts = actionFacts(action);
  return (
    <div className="mb-3 rounded-xl border border-accent/50 bg-accent-soft/40 p-3.5" role="group" aria-label="Waiting for your go-ahead">
      <div className="flex items-start gap-2.5">
        <span className="mt-0.5 grid size-6 shrink-0 place-items-center rounded-full bg-accent text-[11px] text-white">{ACTION_ICON[action.kind]}</span>
        <div className="min-w-0 flex-1">
          <div className="text-[11px] font-medium uppercase tracking-wide text-accent">Waiting for your go-ahead</div>
          <div className="text-[14px] font-semibold">{action.title}</div>
          {action.reason && <p className="mt-0.5 text-[13px] text-muted">{action.reason}</p>}
          {facts.length > 0 && (
            <div className="mt-2 flex flex-wrap gap-1.5">
              {facts.map((f) => (
                <Badge key={f}>{f}</Badge>
              ))}
            </div>
          )}
        </div>
      </div>
      {asking && (
        <TextArea
          rows={2}
          className="mt-3 w-full"
          autoFocus
          value={why}
          onChange={(e) => setWhy(e.target.value)}
          placeholder="Optional: what would you rather do? (e.g. “use fewer examples”)"
        />
      )}
      <ErrorNote error={decide.error} />
      <div className="mt-3 flex gap-2">
        {asking ? (
          <>
            <Button size="sm" loading={decide.isPending} onClick={() => decide.mutate(false)}>
              Send “not now”
            </Button>
            <Button size="sm" variant="ghost" onClick={() => setAsking(false)}>
              Back
            </Button>
          </>
        ) : (
          <>
            <Button size="sm" variant="primary" loading={decide.isPending} onClick={() => decide.mutate(true)}>
              Go ahead
            </Button>
            <Button size="sm" onClick={() => setAsking(true)}>
              Not now
            </Button>
          </>
        )}
      </div>
    </div>
  );
}

function actionFacts(a: PendingAction): string[] {
  const d = a.details as Record<string, number | string | boolean | null | undefined>;
  const out: string[] = [];
  if (a.kind === "model") {
    if (d.params_b) out.push(`${d.params_b}B parameters`);
    out.push(d.on_this_mac ? "already on this Mac" : "needs a download");
  } else if (a.kind === "sft" || a.kind === "dpo") {
    if (d.train_examples) out.push(`${d.train_examples} examples`);
    if (d.iterations) out.push(`${d.iterations} steps`);
    if (d.epochs) out.push(`${Number(d.epochs).toFixed(1)} epochs`);
    if (d.memory_gb) out.push(`${d.memory_gb} GB of ${d.budget_gb} GB`);
    if (d.would_queue_behind) out.push("will queue behind another run");
  } else if (a.kind === "export") {
    out.push(d.quantize_bits ? `${d.quantize_bits}-bit` : "no extra quantizing");
  } else if (a.kind === "synthesize") {
    out.push(`${d.count} ${d.kind === "preference" ? "preference pairs" : "examples"}`, "uses the OpenAI API");
  } else if (a.kind === "review") {
    out.push(`${d.prompts} questions`, "runs the model, then uses the OpenAI API");
  } else if (a.kind === "evaluate") {
    out.push(`${d.questions} test questions`, String(d.target), "runs the model, then uses the OpenAI API");
  } else if (a.kind === "scout" || a.kind === "prep") {
    out.push("a specialist agent · a handful of API calls");
  } else if (a.kind === "import") {
    out.push(`up to ${Number(d.max_rows).toLocaleString()} rows`, "downloads from Hugging Face");
  }
  return out;
}
