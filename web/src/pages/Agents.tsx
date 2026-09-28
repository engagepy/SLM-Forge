import { useMutation, useQuery } from "@tanstack/react-query";
import { useState } from "react";

import { api, fmt, type AgentEvent, type Proposal } from "../api";
import { PAGE, PageHeader } from "../components/Page";
import ProposalCard from "../components/ProposalCard";
import { useProjectId, useSystem } from "../hooks";
import { Badge, Button, Card, cx, EmptyNote, ErrorNote, Field, Input, NumberField, Select, StatusBadge } from "../ui";

const AGENT_TONE = { scout: "info", prep: "neutral", observer: "accent", synth: "good" } as const;

export default function AgentsPage() {
  const projectId = useProjectId();
  const system = useSystem();
  const proposals = useQuery({
    queryKey: ["proposals", projectId],
    queryFn: () => api.get<Proposal[]>(`/api/projects/${projectId}/proposals`),
  });
  const events = useQuery({
    queryKey: ["agent-events", projectId],
    queryFn: () => api.get<AgentEvent[]>(`/api/projects/${projectId}/agents/events?limit=300`),
  });
  const pending = (proposals.data ?? []).filter((p) => p.status === "pending" || (p.action === "acquire_manually" && p.status === "approved"));
  const history = (proposals.data ?? []).filter((p) => !pending.includes(p));
  const noKey = system.data && !system.data.agents.key_configured;

  return (
    <div className={PAGE}>
      <PageHeader title="Agents">
        Agents research, prepare data, watch feedback and write synthetic examples. They only propose: anything that
        downloads, trains or adds data waits here for your approval.
      </PageHeader>
      {noKey && (
        <div className="rounded-lg border border-warn/40 bg-warn-soft px-4 py-3 text-[13px] text-warn">
          No API key for the {system.data?.agents.provider} agent provider. Add{" "}
          <span className="font-mono">{system.data?.agents.key_env}=…</span> to <span className="font-mono">.env</span> in the project root and
          restart the server, or set <span className="font-mono">SLM_AGENT_PROVIDER=ollama</span> to use a local model.
        </div>
      )}

      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_380px]">
        <div className="space-y-4">
          <Card title={`Waiting for you (${pending.length})`} subtitle="Nothing downloads, trains or adds data until you approve it">
            {pending.length ? (
              <div className="space-y-4">
                {pending.map((p) => (
                  <ProposalCard key={p.id} p={p} projectId={projectId} />
                ))}
              </div>
            ) : (
              <EmptyNote>No pending proposals. Run an agent, or give feedback on the Refine page and ask the Observer.</EmptyNote>
            )}
          </Card>
          {history.length > 0 && (
            <Card title="History" pad={false}>
              <ul className="divide-y divide-line">
                {history.map((p) => (
                  <li key={p.id} className="flex items-center gap-3 px-4 py-2.5 text-[13px]">
                    <Badge tone={AGENT_TONE[p.agent as keyof typeof AGENT_TONE] ?? "neutral"}>{p.agent}</Badge>
                    <span className="min-w-0 flex-1 truncate">{p.title}</span>
                    <span className="text-[11px] text-faint">{fmt.ago(p.created_at)}</span>
                    <StatusBadge status={p.status} />
                  </li>
                ))}
              </ul>
            </Card>
          )}
        </div>

        <div className="space-y-5">
          <RunAgents projectId={projectId} />
          <Card title="Activity" subtitle="Live" pad={false}>
            <ol className="max-h-[560px] space-y-0 overflow-y-auto p-3">
              {(events.data ?? []).slice().reverse().map((e) => (
                <EventRow key={e.id} e={e} />
              ))}
              {!events.data?.length && <li className="p-2"><EmptyNote>No agent activity yet.</EmptyNote></li>}
            </ol>
          </Card>
        </div>
      </div>
    </div>
  );
}

function EventRow({ e }: { e: AgentEvent }) {
  const [open, setOpen] = useState(false);
  const c = e.content as Record<string, unknown>;
  let text = "";
  if (e.kind === "message") text = String(c.text ?? "");
  else if (e.kind === "tool_call") text = `${c.tool}(${summarise(c.input)})`;
  else if (e.kind === "tool_result") text = `${c.is_error ? "✗" : "↳"} ${String(c.output ?? "").slice(0, 140)}`;
  else if (e.kind === "proposal") text = `Proposed: ${c.title}`;
  else text = String(c.text ?? JSON.stringify(c));
  const expandable = e.kind === "tool_result" || e.kind === "tool_call";
  return (
    <li className="border-b border-line/60 py-2 last:border-0">
      <button className="w-full text-left" onClick={() => expandable && setOpen(!open)}>
        <div className="flex items-center gap-1.5 text-[11px]">
          <Badge tone={AGENT_TONE[e.agent as keyof typeof AGENT_TONE] ?? "neutral"}>{e.agent}</Badge>
          <span className={cx("text-faint", e.kind === "error" && "text-bad")}>{e.kind.replace("_", " ")}</span>
          <span className="ml-auto text-faint">{fmt.ago(e.created_at)}</span>
        </div>
        <p
          className={cx(
            "mt-1 text-xs leading-relaxed",
            e.kind === "message" ? "text-fg" : "font-mono text-muted",
            e.kind === "error" && "text-bad",
            !open && "line-clamp-3",
          )}
        >
          {text}
        </p>
      </button>
      {open && (
        <pre className="mt-1 max-h-64 overflow-auto rounded-md bg-bg p-2 font-mono text-[11px] text-muted">
          {JSON.stringify(e.kind === "tool_result" ? tryParse(String(c.output)) : c.input, null, 2)}
        </pre>
      )}
    </li>
  );
}

function RunAgents({ projectId }: { projectId: number }) {
  const [request, setRequest] = useState("");
  const [synth, setSynth] = useState<{ kind: "sft" | "preference"; count: number; focus: string }>({ kind: "sft", count: 20, focus: "" });
  const run = useMutation({
    mutationFn: ({ path, body }: { path: string; body: unknown }) => api.post<{ job_id: number }>(`/api/projects/${projectId}/agents/${path}`, body),
  });
  return (
    <Card title="Run an agent">
      <div className="space-y-4">
        <div className="space-y-2">
          <div className="text-xs font-medium">DataScout</div>
          <div className="flex gap-2">
            <Input value={request} onChange={(e) => setRequest(e.target.value)} placeholder="optional guidance" />
            <Button onClick={() => run.mutate({ path: "scout", body: { request } })}>Run</Button>
          </div>
        </div>
        <div className="flex items-center justify-between gap-2 border-t border-line pt-3">
          <div>
            <div className="text-xs font-medium">Observer</div>
            <div className="text-[11px] text-faint">Analyse new feedback and propose next steps</div>
          </div>
          <Button onClick={() => run.mutate({ path: "observe", body: { use_llm: true } })}>Run</Button>
        </div>
        <div className="space-y-2 border-t border-line pt-3">
          <div className="text-xs font-medium">Synth</div>
          <div className="grid grid-cols-2 gap-2">
            <Field label="Kind">
              <Select
                value={synth.kind}
                onChange={(k) => setSynth({ ...synth, kind: k })}
                options={[
                  { value: "sft", label: "SFT examples" },
                  { value: "preference", label: "Preference pairs" },
                ]}
              />
            </Field>
            <NumberField label="Count" value={synth.count} onChange={(v) => v && setSynth({ ...synth, count: v })} min={1} max={200} />
          </div>
          <Input value={synth.focus} onChange={(e) => setSynth({ ...synth, focus: e.target.value })} placeholder="Focus, e.g. “short answers with exact temperatures”" />
          <Button className="w-full" onClick={() => run.mutate({ path: "synthesize", body: synth })}>
            Generate
          </Button>
        </div>
        {run.data && <p className="text-xs text-good">Started job {run.data.job_id}. Progress appears in Activity.</p>}
        <ErrorNote error={run.error} />
      </div>
    </Card>
  );
}

function summarise(input: unknown) {
  if (!input || typeof input !== "object") return "";
  return Object.entries(input as Record<string, unknown>)
    .map(([k, v]) => `${k}=${typeof v === "string" ? JSON.stringify(v.length > 40 ? v.slice(0, 40) + "…" : v) : typeof v === "object" ? "{…}" : v}`)
    .join(", ");
}

function tryParse(s: string) {
  try {
    return JSON.parse(s);
  } catch {
    return s;
  }
}
