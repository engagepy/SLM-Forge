import { useQuery } from "@tanstack/react-query";
import { useEffect, useState } from "react";
import { Link, useParams, useSearchParams } from "react-router";

import { api, DEFAULT_SAMPLING, type ExportRow, fmt } from "../api";
import ChatComposer from "../components/ChatComposer";
import Markdown from "../components/Markdown";
import { useChatStream, useStudio } from "../hooks";
import { Badge, CodeBlock, cx, ErrorNote, Select, Spinner } from "../ui";

const CREATIVITY = [
  { value: "0.3", label: "Focused" },
  { value: "0.7", label: "Balanced" },
  { value: "1.0", label: "Playful" },
] as const;

/** Chat with the finished, exported model: the files on disk, exactly as they'd run elsewhere. */
export default function TryModel() {
  const projectId = Number(useParams().projectId);
  const [search, setSearch] = useSearchParams();
  const exports = useQuery({
    queryKey: ["exports", projectId],
    queryFn: () => api.get<ExportRow[]>(`/api/projects/${projectId}/exports`),
  });
  const project = useStudio(projectId);
  const chosen = exports.data?.find((e) => String(e.job_id) === search.get("export")) ?? exports.data?.[0];
  const [creativity, setCreativity] = useState<(typeof CREATIVITY)[number]["value"]>("0.7");
  const chat = useChatStream(projectId);
  const { messages, busy } = chat;
  // A different model means a fresh conversation.
  const reset = chat.reset;
  useEffect(() => {
    reset();
  }, [chosen?.job_id]); // eslint-disable-line react-hooks/exhaustive-deps

  // The Tuner's own test questions make good first things to ask.
  const starters = [...new Set((project.data?.samples ?? []).filter((x) => x.target !== "base").map((x) => x.prompt))].slice(0, 4);

  const send = (text: string) =>
    chosen &&
    chat.send(text, (history) => ({
      messages: history,
      params: { ...DEFAULT_SAMPLING, temperature: Number(creativity), max_tokens: 400 },
      target: `export:${chosen.job_id}`,
    }));

  if (exports.isLoading) return <Spinner className="m-6" />;

  return (
    <div className="flex h-full flex-col">
      <header className="flex h-12 shrink-0 items-center gap-3 border-b border-line px-4">
        <Link to={`/p/${projectId}`} className="shrink-0 whitespace-nowrap rounded-md px-2 py-1 text-xs text-muted hover:bg-panel-2 hover:text-fg">
          ← Studio
        </Link>
        <div className="min-w-0">
          <div className="truncate text-[13px] font-semibold">{chosen ? chosen.name : project.data?.project.name ?? "…"}</div>
          <div className="truncate text-[11px] text-faint">{project.data?.project.goal}</div>
        </div>
        <div className="ml-auto flex shrink-0 items-center gap-2 whitespace-nowrap text-xs">
          {chosen && <Badge>{fmt.gb(chosen.size_gb)}</Badge>}
          {chosen && <Badge tone="good">runs on {chosen.min_ram_gb} GB+ Macs</Badge>}
          {exports.data && exports.data.length > 1 && (
            <Select
              className="w-auto py-1 text-xs"
              value={String(chosen?.job_id)}
              onChange={(v) => setSearch({ export: v })}
              options={exports.data.map((e) => ({ value: String(e.job_id), label: e.name }))}
            />
          )}
        </div>
      </header>

      {!chosen ? (
        <div className="mx-auto mt-24 max-w-sm text-center text-[13px] text-muted">
          <h1 className="text-lg font-semibold text-fg">No finished model yet</h1>
          <p className="mt-1">Once the Tuner exports your model (it asks you first), you can chat with it here.</p>
          <Link to={`/p/${projectId}`} className="mt-4 inline-block text-accent hover:underline">
            Back to the Studio
          </Link>
        </div>
      ) : (
        <div className="grid min-h-0 flex-1 lg:grid-cols-[minmax(0,1fr)_300px]">
          <div className="flex min-h-0 flex-col">
            <div className="min-h-0 flex-1 space-y-3 overflow-y-auto px-5 py-5">
              {!chosen.on_disk && (
                <p className="rounded-lg border border-warn/40 bg-warn-soft px-3 py-2 text-[13px] text-warn">
                  This model is no longer at {chosen.path}. It may have been moved or deleted.
                </p>
              )}
              {!messages.length && (
                <div className="mx-auto mt-10 max-w-md text-center">
                  <div className="mx-auto grid size-11 place-items-center rounded-full bg-accent-soft text-lg text-accent">✦</div>
                  <h1 className="mt-3 text-lg font-semibold">Say hello to {chosen.name}</h1>
                  <p className="mt-1 text-[13px] text-muted">
                    This is your finished model, running from its folder on this Mac. Nothing leaves your computer.
                  </p>
                  {starters.length > 0 && (
                    <div className="mt-5 flex flex-wrap justify-center gap-2">
                      {starters.map((q) => (
                        <button
                          key={q}
                          onClick={() => send(q)}
                          className="rounded-full border border-line bg-panel px-3 py-1.5 text-left text-[12.5px] text-muted hover:border-accent/50 hover:text-fg"
                        >
                          {q}
                        </button>
                      ))}
                    </div>
                  )}
                </div>
              )}
              {messages.map((m, i) => (
                <div key={i} className={cx("flex", m.role === "user" ? "justify-end" : "justify-start")}>
                  <div
                    className={cx(
                      "max-w-[85%] rounded-2xl px-4 py-2.5 text-[14px] leading-relaxed",
                      m.role === "user" ? "rounded-tr-sm bg-accent text-white whitespace-pre-wrap" : "rounded-tl-sm border border-line bg-panel",
                    )}
                  >
                    {m.role === "user" ? m.content : m.content ? <Markdown text={m.content} /> : busy && i === messages.length - 1 ? <Spinner className="size-3" /> : null}
                    {m.stats && (
                      <div className="num mt-1.5 text-[11px] text-faint">
                        {m.stats.tokens_per_sec} tokens/s{m.stats.finish_reason === "length" && " · stopped at the length limit"}
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
              clearLabel="New chat"
              canClear={messages.length > 0}
              disabled={!chosen.on_disk}
              placeholder={`Message ${chosen.name}…`}
              autoFocus
            />
          </div>

          <aside className="space-y-5 overflow-y-auto border-l border-line p-4 text-[13px]">
            <div>
              <div className="mb-1.5 text-[11px] font-medium uppercase tracking-wide text-faint">Answers</div>
              <div className="flex rounded-lg border border-line p-0.5">
                {CREATIVITY.map((c) => (
                  <button
                    key={c.value}
                    onClick={() => setCreativity(c.value)}
                    className={cx(
                      "flex-1 rounded-md px-2 py-1 text-xs font-medium transition",
                      creativity === c.value ? "bg-accent text-white" : "text-muted hover:text-fg",
                    )}
                  >
                    {c.label}
                  </button>
                ))}
              </div>
            </div>
            {project.data?.project.system_prompt && (
              <div>
                <div className="mb-1 text-[11px] font-medium uppercase tracking-wide text-faint">Built-in instructions</div>
                <p className="text-xs text-muted">{project.data.project.system_prompt}</p>
              </div>
            )}
            <div>
              <div className="mb-1 text-[11px] font-medium uppercase tracking-wide text-faint">Use it outside SLM Forge</div>
              <p className="text-xs text-muted">The model is a folder you can copy to any Apple Silicon Mac with {chosen.min_ram_gb} GB or more.</p>
              <CodeBlock className="mt-2" label="Folder" text={chosen.path} />
              <CodeBlock className="mt-2" label="Chat in Terminal" text={`pip install mlx-lm && mlx_lm.chat --model "${chosen.path}"`} />
            </div>
          </aside>
        </div>
      )}
    </div>
  );
}
