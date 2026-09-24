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

import json
import re
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


def stringify(value: Any) -> str:
    """Render a cell as training text. Lists (ingredients, steps…) become one item per line."""
    if value is None:
        return ""
    if isinstance(value, list):
        if all(not isinstance(v, (dict, list)) for v in value):
            return "\n".join(f"- {str(v).strip()}" for v in value if str(v).strip())
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


_PLACEHOLDER = re.compile(r"\{(\w+)\}")


def _resolve(row: dict, spec: str | None, field: str = "", required: bool = False) -> str:
    """A spec is a column name, or `=text` for a constant. A constant can reference columns
    as {column}, e.g. `=How do I make {title}?`; other braces are left alone."""
    if not spec:
        if required:
            raise MappingError(f"choose a column for '{field}'")
        return ""
    if spec.startswith("="):
        return _PLACEHOLDER.sub(lambda m: stringify(row[m[1]]) if m[1] in row else m[0], spec[1:])
    if spec not in row:
        raise MappingError(f"column '{spec}' is not in the data")
    return stringify(row[spec])


def _last_assistant(value: Any) -> str:
    """Preference columns are sometimes whole conversations; keep the final assistant turn."""
    if isinstance(value, list):
        turns = _normalise_turns(value)
        for t in reversed(turns):
            if t["role"] == "assistant":
                return t["content"]
        raise MappingError("no assistant turn in preference conversation")
    return stringify(value)


def map_row(row: dict, mapping: dict) -> dict:
    fmt = mapping.get("format")
    system = _resolve(row, mapping.get("system"), "system")

    if fmt == "chat":
        column = mapping.get("messages") or "messages"
        if column not in row:
            raise MappingError(f"column '{column}' is not in the data")
        turns = _normalise_turns(row[column])
        if system and not any(t["role"] == "system" for t in turns):
            turns = [{"role": "system", "content": system}, *turns]
        return {"messages": turns}

    if fmt == "instruction":
        prompt = _resolve(row, mapping.get("prompt"), "prompt", required=True)
        extra = _resolve(row, mapping.get("input"), "input")
        if extra:
            prompt = f"{prompt}\n\n{extra}"
        turns = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": _resolve(row, mapping.get("response"), "response", required=True)},
        ]
        if system:
            turns.insert(0, {"role": "system", "content": system})
        return {"messages": turns}

    if fmt == "text":
        return {"text": _resolve(row, mapping.get("text"), "text", required=True)}

    if fmt == "preference":
        prompt_spec = mapping.get("prompt")
        if prompt_spec and not prompt_spec.startswith("=") and isinstance(row.get(prompt_spec), list):
            turns = _normalise_turns(row[prompt_spec])  # conversation-style prompt: last user turn
            prompt = next((t["content"] for t in reversed(turns) if t["role"] == "user"), "")
        else:
            prompt = _resolve(row, prompt_spec, "prompt", required=True)
        for field in ("chosen", "rejected"):
            if not mapping.get(field):
                raise MappingError(f"choose a column for '{field}'")
            if mapping[field] not in row:
                raise MappingError(f"column '{mapping[field]}' is not in the data")
        rec = {
            "prompt": prompt,
            "chosen": _last_assistant(row[mapping["chosen"]]),
            "rejected": _last_assistant(row[mapping["rejected"]]),
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
    prompt = pick("instruction", "prompt", "question", "query", "input", "problem", "title", "name", "topic", "subject")
    response = pick(
        "output", "response", "answer", "completion", "solution", "target",
        "method", "instructions", "steps", "directions", "recipe", "body", "summary", "description",
    )  # fmt: skip
    if prompt and response:
        mapping = {"format": "instruction", "prompt": prompt, "response": response}
        # A separate context column (Alpaca's "input") is appended to the prompt.
        if (extra := pick("input", "context", "ingredients", "details", "background")) and extra not in (
            prompt,
            response,
        ):
            mapping["input"] = extra
        if s := pick("system", "system_prompt"):
            mapping["system"] = s
        return mapping
    if t := pick("text", "content", "document"):
        return {"format": "text", "text": t}
    return {"format": "unknown"}


def record_kind(mapping: dict) -> str:
    return "dpo" if mapping.get("format") == "preference" else "sft"
