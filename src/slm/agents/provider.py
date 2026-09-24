"""LLM providers for the agents: Claude (default), Ollama (offline fallback), Fake (tests).

The tool loop lives here rather than in the SDK's tool runner so the same loop, event
logging and error handling drive every provider.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from slm.config import get_settings


@dataclass
class Tool:
    name: str
    description: str
    input_schema: dict
    fn: Callable[[dict], Any]


@dataclass
class AgentResult:
    text: str
    steps: int
    tool_calls: list[dict] = field(default_factory=list)


EventFn = Callable[[str, dict], None]  # (kind, content)


class ProviderError(RuntimeError):
    pass


class LLMProvider(Protocol):
    name: str

    def run(
        self,
        system: str,
        user: str,
        tools: list[Tool],
        *,
        on_event: EventFn,
        max_steps: int = 12,
    ) -> AgentResult: ...

    def json(self, system: str, user: str, schema: dict) -> dict: ...


def _call_tool(tool: Tool | None, name: str, args: dict, on_event: EventFn) -> tuple[str, bool]:
    on_event("tool_call", {"tool": name, "input": args})
    if tool is None:
        out, is_error = f"Unknown tool {name!r}", True
    else:
        try:
            result = tool.fn(args)
            out, is_error = result if isinstance(result, str) else json.dumps(result, default=str), False
        except Exception as e:  # report to the model so it can adjust
            out, is_error = f"{e.__class__.__name__}: {e}", True
    on_event("tool_result", {"tool": name, "output": out[:4000], "is_error": is_error})
    return out[:30000], is_error


def strict_schema(schema: dict) -> dict:
    """Structured outputs need additionalProperties: false and every property required."""
    s = dict(schema)
    if s.get("type") == "object":
        props = {k: strict_schema(v) for k, v in s.get("properties", {}).items()}
        s["properties"] = props
        s["required"] = list(props)
        s["additionalProperties"] = False
    elif s.get("type") == "array" and "items" in s:
        s["items"] = strict_schema(s["items"])
    return s


# ── Claude ──────────────────────────────────────────────────────────────────


class ClaudeProvider:
    name = "claude"

    def __init__(self, model: str | None = None, effort: str = "high") -> None:
        import anthropic

        settings = get_settings()
        self.model = model or settings.claude_model
        self.effort = effort
        # api_key=None lets the SDK resolve ANTHROPIC_API_KEY / `ant auth login` profiles.
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    def _create(self, **kwargs):
        import anthropic

        try:
            resp = self.client.beta.messages.create(
                model=self.model,
                max_tokens=16000,
                thinking={"type": "adaptive"},
                betas=["server-side-fallback-2026-07-01"],
                # Re-run declined requests on Anthropic's recommended fallback model.
                extra_body={"fallbacks": "default"},
                **kwargs,
            )
        except anthropic.AuthenticationError as e:
            raise ProviderError(
                "Claude authentication failed. Set ANTHROPIC_API_KEY in .env or run `ant auth login`."
            ) from e
        except anthropic.RateLimitError as e:
            raise ProviderError("Claude rate limit hit; try again shortly.") from e
        except anthropic.APIConnectionError as e:
            raise ProviderError("Could not reach the Claude API (network).") from e
        except anthropic.APIStatusError as e:
            raise ProviderError(f"Claude API error {e.status_code}: {e.message}") from e
        if resp.stop_reason == "refusal":
            raise ProviderError("Claude declined this request.")
        return resp

    def _system(self, system: str) -> list[dict]:
        # Stable across a run's turns, so cache it.
        return [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]

    def run(self, system, user, tools, *, on_event, max_steps=12) -> AgentResult:
        by_name = {t.name: t for t in tools}
        tool_defs = [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in tools]
        messages: list[dict] = [{"role": "user", "content": user}]
        calls: list[dict] = []
        for step in range(1, max_steps + 1):
            resp = self._create(
                system=self._system(system),
                tools=tool_defs,
                messages=messages,
                output_config={"effort": self.effort},
            )
            text = "".join(b.text for b in resp.content if b.type == "text")
            if text.strip():
                on_event("message", {"text": text})
            uses = [b for b in resp.content if b.type == "tool_use"]
            if resp.stop_reason != "tool_use" or not uses:
                if resp.stop_reason == "max_tokens":
                    on_event("error", {"text": "Response hit max_tokens"})
                return AgentResult(text=text, steps=step, tool_calls=calls)
            messages.append({"role": "assistant", "content": resp.content})
            results = []
            for b in uses:  # all results go back in one user message
                out, is_error = _call_tool(by_name.get(b.name), b.name, dict(b.input), on_event)
                calls.append({"tool": b.name, "input": dict(b.input), "is_error": is_error})
                results.append({"type": "tool_result", "tool_use_id": b.id, "content": out, "is_error": is_error})
            messages.append({"role": "user", "content": results})
        on_event("error", {"text": f"Stopped after {max_steps} steps"})
        return AgentResult(text="", steps=max_steps, tool_calls=calls)

    def json(self, system, user, schema) -> dict:
        resp = self._create(
            system=self._system(system),
            messages=[{"role": "user", "content": user}],
            output_config={
                "effort": self.effort,
                "format": {"type": "json_schema", "schema": strict_schema(schema)},
            },
        )
        text = next((b.text for b in resp.content if b.type == "text"), "")
        return json.loads(text)


# ── Ollama ──────────────────────────────────────────────────────────────────


class OllamaProvider:
    name = "ollama"

    def __init__(self, model: str | None = None) -> None:
        settings = get_settings()
        self.model = model or settings.ollama_model
        self.url = settings.ollama_url.rstrip("/")

    def _chat(self, payload: dict) -> dict:
        import httpx

        try:
            r = httpx.post(f"{self.url}/api/chat", json={"model": self.model, "stream": False, **payload}, timeout=600)
            r.raise_for_status()
        except httpx.HTTPError as e:
            raise ProviderError(f"Ollama request failed ({self.url}, model {self.model}): {e}") from e
        return r.json()["message"]

    def run(self, system, user, tools, *, on_event, max_steps=12) -> AgentResult:
        by_name = {t.name: t for t in tools}
        tool_defs = [
            {
                "type": "function",
                "function": {"name": t.name, "description": t.description, "parameters": t.input_schema},
            }
            for t in tools
        ]
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        calls: list[dict] = []
        for step in range(1, max_steps + 1):
            msg = self._chat({"messages": messages, "tools": tool_defs})
            text = msg.get("content", "")
            if text.strip():
                on_event("message", {"text": text})
            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                return AgentResult(text=text, steps=step, tool_calls=calls)
            messages.append(msg)
            for tc in tool_calls:
                fn = tc["function"]
                args = fn.get("arguments") or {}
                if isinstance(args, str):
                    args = json.loads(args or "{}")
                out, is_error = _call_tool(by_name.get(fn["name"]), fn["name"], args, on_event)
                calls.append({"tool": fn["name"], "input": args, "is_error": is_error})
                messages.append({"role": "tool", "content": out, "tool_name": fn["name"]})
        on_event("error", {"text": f"Stopped after {max_steps} steps"})
        return AgentResult(text="", steps=max_steps, tool_calls=calls)

    def json(self, system, user, schema) -> dict:
        msg = self._chat(
            {
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "format": strict_schema(schema),
                "options": {"temperature": 0.4},
            }
        )
        return json.loads(msg["content"])


# ── Fake (tests) ────────────────────────────────────────────────────────────


class FakeProvider:
    """Scripted provider. `script` items: {"tool": name, "input": {...}} or {"text": "..."};
    `json_responses` are returned in order from json()."""

    name = "fake"

    def __init__(self, script: list[dict] | None = None, json_responses: list[dict] | None = None) -> None:
        self.script = list(script or [])
        self.json_responses = list(json_responses or [])
        self.json_calls: list[tuple[str, str]] = []

    def run(self, system, user, tools, *, on_event, max_steps=12) -> AgentResult:
        by_name = {t.name: t for t in tools}
        calls = []
        text = ""
        for item in self.script[:max_steps]:
            if "tool" in item:
                _, is_error = _call_tool(by_name.get(item["tool"]), item["tool"], item.get("input", {}), on_event)
                calls.append({"tool": item["tool"], "input": item.get("input", {}), "is_error": is_error})
            else:
                text = item["text"]
                on_event("message", {"text": text})
        return AgentResult(text=text, steps=len(self.script), tool_calls=calls)

    def json(self, system, user, schema) -> dict:
        self.json_calls.append((system, user))
        if not self.json_responses:
            raise ProviderError("FakeProvider has no scripted json response")
        return self.json_responses.pop(0)


_override: LLMProvider | None = None


def set_provider(p: LLMProvider | None) -> None:
    """Tests inject a FakeProvider here."""
    global _override
    _override = p


def get_provider(name: str | None = None) -> LLMProvider:
    if _override is not None:
        return _override
    name = name or get_settings().agent_provider
    if name == "ollama":
        return OllamaProvider()
    return ClaudeProvider()
