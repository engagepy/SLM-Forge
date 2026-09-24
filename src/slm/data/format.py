"""Map raw dataset rows onto the record shapes the trainers expect.

SFT (mlx_lm.lora):       {"messages": [{"role": ..., "content": ...}, ...]}  or  {"text": ...}
Preference (mlx_lm_lora): {"prompt": ..., "chosen": ..., "rejected": ..., "system"?: ...}

A mapping is a small dict describing which raw columns feed which fields. Each value is a
column name, or `=text` for a constant:

    {"format": "chat", "messages": "messages"}
    {"format": "instruction", "prompt": "instruction", "input": "input", "response": "output",
     "system": "=You are a helpful assistant."}
    {"format": "text", "text": "text"}
    {"format": "preference", "prompt": "prompt", "chosen": "chosen", "rejected": "rejected"}
"""

from typing import Any

ROLE_ALIASES = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "model": "assistant",
    "system": "system",
}

MappingError = ValueError


def _normalise_turns(turns: Any) -> list[dict]:
    """Accept OpenAI-style {role, content} or ShareGPT-style {from, value} turns."""
    if not isinstance(turns, list):
        raise MappingError("messages column is not a list")
    out = []
    for t in turns:
        if not isinstance(t, dict):
            raise MappingError("message turn is not an object")
        role = t.get("role", t.get("from"))
        content = t.get("content", t.get("value"))
        role = ROLE_ALIASES.get(str(role).lower())
        if role is None or content is None:
            raise MappingError(f"unrecognised turn: {list(t.keys())}")
        out.append({"role": role, "content": str(content)})
    return out


def _resolve(row: dict, spec: str | None) -> str:
    """A spec is a column name, or `=literal text` for a constant."""
    if not spec:
        return ""
    if spec.startswith("="):
        return spec[1:]
    value = row.get(spec)
    return "" if value is None else str(value)


def _last_assistant(value: Any) -> str:
    """Preference columns are sometimes whole conversations; keep the final assistant turn."""
    if isinstance(value, list):
        turns = _normalise_turns(value)
        for t in reversed(turns):
            if t["role"] == "assistant":
                return t["content"]
        raise MappingError("no assistant turn in preference conversation")
    return "" if value is None else str(value)


def map_row(row: dict, mapping: dict) -> dict:
    fmt = mapping.get("format")
    system = _resolve(row, mapping.get("system"))

    if fmt == "chat":
        turns = _normalise_turns(row.get(mapping.get("messages", "messages")))
        if system and not any(t["role"] == "system" for t in turns):
            turns = [{"role": "system", "content": system}, *turns]
        return {"messages": turns}

    if fmt == "instruction":
        prompt = _resolve(row, mapping.get("prompt"))
        extra = _resolve(row, mapping.get("input"))
        if extra:
            prompt = f"{prompt}\n\n{extra}"
        turns = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": _resolve(row, mapping.get("response"))},
        ]
        if system:
            turns.insert(0, {"role": "system", "content": system})
        return {"messages": turns}

    if fmt == "text":
        return {"text": _resolve(row, mapping.get("text", "text"))}

    if fmt == "preference":
        prompt_value = row.get(mapping.get("prompt", "prompt"))
        if isinstance(prompt_value, list):  # conversation-style prompt: last user turn
            turns = _normalise_turns(prompt_value)
            prompt = next((t["content"] for t in reversed(turns) if t["role"] == "user"), "")
        else:
            prompt = _resolve(row, mapping.get("prompt", "prompt"))
        rec = {
            "prompt": prompt,
            "chosen": _last_assistant(row.get(mapping.get("chosen", "chosen"))),
            "rejected": _last_assistant(row.get(mapping.get("rejected", "rejected"))),
        }
        if system:
            rec["system"] = system
        return rec

    raise MappingError(f"unknown format: {fmt!r}")


def guess_mapping(columns: list[str]) -> dict:
    """Heuristic first guess; the DataPrep agent or the user can refine it."""
    cols = {c.lower(): c for c in columns}

    def pick(*names: str) -> str | None:
        return next((cols[n] for n in names if n in cols), None)

    if (c := pick("chosen")) and (r := pick("rejected")):
        prompt = pick("prompt", "question", "instruction") or "prompt"
        return {"format": "preference", "prompt": prompt, "chosen": c, "rejected": r}
    if m := pick("messages", "conversations", "conversation", "dialog", "chat"):
        return {"format": "chat", "messages": m}
    prompt = pick("instruction", "prompt", "question", "query", "input", "problem")
    response = pick("output", "response", "answer", "completion", "solution", "target")
    if prompt and response:
        mapping = {"format": "instruction", "prompt": prompt, "response": response}
        # A separate context column (Alpaca's "input") is appended to the prompt.
        if (extra := pick("input", "context")) and extra != prompt:
            mapping["input"] = extra
        if s := pick("system", "system_prompt"):
            mapping["system"] = s
        return mapping
    if t := pick("text", "content", "document"):
        return {"format": "text", "text": t}
    return {"format": "unknown"}


def record_kind(mapping: dict) -> str:
    return "dpo" if mapping.get("format") == "preference" else "sft"
