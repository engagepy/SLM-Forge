// One chat bubble for every place you talk to a local model (Try it and the Playground): the
// answer rendered as Markdown, and the same one-line stats under it.
import { type ChatMsg as Message } from "../hooks";
import { Bubble, Spinner } from "../ui";
import Markdown from "./Markdown";

// A fine-tuned Qwen3 writes an empty <think></think> before its answer (its template trains one
// into every reply); hide it, as engine.strip_think does for scoring.
const EMPTY_THINK = /^\s*<think>\s*<\/think>\s*/;

export default function ChatMessage({ m, pending }: { m: Message; pending: boolean }) {
  const content = m.role === "user" ? m.content : m.content.replace(EMPTY_THINK, "");
  return (
    <Bubble role={m.role === "user" ? "user" : "assistant"}>
      {m.role === "user" ? content : content ? <Markdown text={content} /> : pending ? <Spinner className="size-3" /> : null}
      {m.stats && (
        <div className="num mt-1.5 text-[11px] text-faint">
          {m.stats.generation_tokens} tokens · {m.stats.tokens_per_sec} tokens/s · {m.stats.seconds}s
          {m.stats.finish_reason === "length" && " · stopped at the length limit"}
        </div>
      )}
    </Bubble>
  );
}
