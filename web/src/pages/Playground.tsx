import { useState } from "react";

import { DEFAULT_SAMPLING, type SamplingParams } from "../api";
import ChatComposer from "../components/ChatComposer";
import SamplingControls from "../components/SamplingControls";
import { useChatStream, useOverview, useProjectId } from "../hooks";
import { Button, Card, cx, ErrorNote, Field, Select, TextArea } from "../ui";

export default function Playground() {
  const projectId = useProjectId();
  const { data: ov } = useOverview(projectId);
  const [params, setParams] = useState<SamplingParams>(DEFAULT_SAMPLING);
  const [target, setTarget] = useState("current");
  const [system, setSystem] = useState<string | null>(null);
  const chat = useChatStream(projectId);
  const { messages, busy } = chat;

  const effectiveSystem = system ?? ov?.project.system_prompt ?? "";

  const send = (text: string) =>
    chat.send(text, (history) => ({
      messages: effectiveSystem ? [{ role: "system", content: effectiveSystem }, ...history] : history,
      params,
      target,
    }));

  const targets = [
    { value: "current", label: ov?.project.current_adapter_path || ov?.checkpoints.length ? "Current (latest trained)" : "Current (base model)" },
    { value: "base", label: "Base model (untrained)" },
    ...(ov?.checkpoints ?? []).map((c) => ({ value: `checkpoint:${c.id}`, label: `${c.kind.toUpperCase()} · job ${c.job_id}` })),
  ];

  return (
    <div className="grid h-full lg:grid-cols-[minmax(0,1fr)_320px]">
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
          <ErrorNote error={chat.error} />
          <div ref={chat.bottom} />
        </div>
        <ChatComposer
          busy={busy}
          onSend={send}
          onStop={chat.stop}
          onClear={chat.reset}
          placeholder="Message the model (Enter to send, Shift+Enter for a new line)"
        />
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
