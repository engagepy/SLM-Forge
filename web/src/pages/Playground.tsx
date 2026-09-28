import { useState } from "react";

import { DEFAULT_SAMPLING, runLabel, type SamplingParams } from "../api";
import ChatComposer from "../components/ChatComposer";
import ChatMessage from "../components/ChatMessage";
import SamplingControls from "../components/SamplingControls";
import { useChatStream, useOverview, useProjectId } from "../hooks";
import { Button, Card, ErrorNote, Field, Select, TextArea } from "../ui";

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
    { value: "base", label: "Before training (base model)" },
    ...(ov?.checkpoints ?? []).map((c) => ({ value: `checkpoint:${c.id}`, label: runLabel(c.kind, c.job_id) })),
  ];

  return (
    <div className="grid h-full lg:grid-cols-[minmax(0,1fr)_320px]">
      <div className="flex min-h-0 flex-col">
        <div className="min-h-0 flex-1 space-y-3 overflow-y-auto px-5 py-5">
          {!messages.length && (
            <div className="mx-auto mt-10 max-w-md text-center">
              <div className="mx-auto grid size-11 place-items-center rounded-full bg-accent-soft text-lg text-accent">✦</div>
              <h1 className="mt-3 text-lg font-semibold">Playground</h1>
              <p className="mt-1 text-[13px] text-muted">
                Chat with any version of your model. Switch between the base model and checkpoints to see what training changed.
              </p>
            </div>
          )}
          {messages.map((m, i) => (
            <ChatMessage key={i} m={m} pending={busy && i === messages.length - 1} />
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

      <aside className="space-y-5 overflow-y-auto border-l border-line p-4 text-[13px]">
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
