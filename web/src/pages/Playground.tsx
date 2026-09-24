import { useEffect, useRef, useState } from "react";

import { DEFAULT_SAMPLING, postStream, type SamplingParams } from "../api";
import SamplingControls from "../components/SamplingControls";
import { useOverview, useProjectId } from "../hooks";
import { Button, Card, cx, ErrorNote, Field, Select, TextArea } from "../ui";

interface Msg {
  role: "user" | "assistant";
  content: string;
  stats?: { tokens_per_sec: number; generation_tokens: number; seconds: number; finish_reason: string | null };
}

export default function Playground() {
  const projectId = useProjectId();
  const { data: ov } = useOverview(projectId);
  const [params, setParams] = useState<SamplingParams>(DEFAULT_SAMPLING);
  const [target, setTarget] = useState("current");
  const [system, setSystem] = useState<string | null>(null);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<unknown>(null);
  const abort = useRef<AbortController | null>(null);
  const bottom = useRef<HTMLDivElement>(null);
  useEffect(() => bottom.current?.scrollIntoView({ block: "end" }), [messages]);

  const effectiveSystem = system ?? ov?.project.system_prompt ?? "";

  async function send() {
    const text = input.trim();
    if (!text || busy) return;
    const history: Msg[] = [...messages, { role: "user", content: text }];
    setMessages([...history, { role: "assistant", content: "" }]);
    setInput("");
    setBusy(true);
    setError(null);
    abort.current = new AbortController();
    const payload: { role: string; content: string }[] = history.map(({ role, content }) => ({ role, content }));
    if (effectiveSystem) payload.unshift({ role: "system", content: effectiveSystem });
    try {
      await postStream(
        `/api/projects/${projectId}/generate`,
        { messages: payload, params, target },
        (ev) => {
          if (ev.type === "token") {
            setMessages((m) => {
              const copy = [...m];
              copy[copy.length - 1] = { ...copy[copy.length - 1], content: copy[copy.length - 1].content + (ev.text as string) };
              return copy;
            });
          } else if (ev.type === "done") {
            setMessages((m) => {
              const copy = [...m];
              copy[copy.length - 1] = { ...copy[copy.length - 1], stats: ev as unknown as Msg["stats"] };
              return copy;
            });
          } else if (ev.type === "error") {
            setError(new Error(String(ev.message)));
          }
        },
        abort.current.signal,
      );
    } catch (e) {
      if ((e as Error).name !== "AbortError") setError(e);
    } finally {
      setBusy(false);
    }
  }

  const targets = [
    { value: "current", label: ov?.project.current_adapter_path || ov?.checkpoints.length ? "Current (latest trained)" : "Current (base model)" },
    { value: "base", label: "Base model (untrained)" },
    ...(ov?.checkpoints ?? []).map((c) => ({ value: `checkpoint:${c.id}`, label: `${c.kind.toUpperCase()} · job ${c.job_id}` })),
  ];

  return (
    <div className="grid h-full lg:grid-cols-[1fr_320px]">
      <div className="flex min-h-0 flex-col">
        <div className="flex-1 space-y-4 overflow-y-auto p-6">
          {!messages.length && (
            <div className="mx-auto mt-16 max-w-md text-center">
              <h1 className="text-lg font-semibold">Playground</h1>
              <p className="mt-1 text-[13px] text-muted">
                Chat with any version of your model. Switch between the base model and checkpoints to see what training changed.
              </p>
            </div>
          )}
          {messages.map((m, i) => (
            <div key={i} className={cx("flex", m.role === "user" ? "justify-end" : "justify-start")}>
              <div
                className={cx(
                  "max-w-[80%] rounded-2xl px-4 py-2.5 text-[13.5px] leading-relaxed",
                  m.role === "user" ? "bg-accent text-white" : "border border-line bg-panel",
                )}
              >
                <div className="whitespace-pre-wrap">{m.content || (busy && i === messages.length - 1 ? "…" : "")}</div>
                {m.stats && (
                  <div className="num mt-1.5 text-[11px] text-faint">
                    {m.stats.generation_tokens} tokens · {m.stats.tokens_per_sec} tok/s · {m.stats.seconds}s
                    {m.stats.finish_reason === "length" && " · hit max tokens"}
                  </div>
                )}
              </div>
            </div>
          ))}
          <ErrorNote error={error} />
          <div ref={bottom} />
        </div>
        <form
          className="flex gap-2 border-t border-line p-4"
          onSubmit={(e) => {
            e.preventDefault();
            send();
          }}
        >
          <TextArea
            rows={2}
            className="flex-1"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                send();
              }
            }}
            placeholder="Message the model (Enter to send, Shift+Enter for a new line)"
          />
          <div className="flex flex-col gap-1.5">
            {busy ? (
              <Button type="button" variant="danger" onClick={() => abort.current?.abort()}>
                Stop
              </Button>
            ) : (
              <Button type="submit" variant="primary">
                Send
              </Button>
            )}
            <Button type="button" variant="ghost" size="sm" onClick={() => setMessages([])} disabled={busy}>
              Clear
            </Button>
          </div>
        </form>
      </div>

      <aside className="space-y-4 overflow-y-auto border-l border-line p-4">
        <Field label="Model">
          <Select value={target} onChange={setTarget} options={targets} />
        </Field>
        <Field label="System prompt" hint={system === null ? "project default" : "overridden"}>
          <TextArea rows={3} value={effectiveSystem} onChange={(e) => setSystem(e.target.value)} />
        </Field>
        <Card title="Sampling">
          <SamplingControls value={params} onChange={setParams} />
          <Button size="sm" variant="ghost" className="mt-3" onClick={() => setParams(DEFAULT_SAMPLING)}>
            Reset to defaults
          </Button>
        </Card>
      </aside>
    </div>
  );
}
