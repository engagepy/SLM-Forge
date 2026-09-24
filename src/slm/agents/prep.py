"""DataPrep: proposes how a raw dataset maps onto training records and how to clean it."""

import json
from pathlib import Path

from sqlmodel import Session

from slm.agents.base import create_proposal, event_logger, project_context
from slm.agents.provider import get_provider
from slm.data import format as fmt
from slm.data.pipeline import preview_mapping
from slm.data.scout_tools import read_raw
from slm.db import Dataset, Proposal, engine

AGENT = "prep"

SYSTEM = """You are DataPrep. Given a raw dataset's columns and sample rows, decide how to turn it
into fine-tuning records for a small language model, and which cleaning thresholds to use.

Formats and their mapping fields (values are column names, or "=text" for a constant; use "" for
fields that don't apply):
- chat: messages (a column holding a list of {role, content} or {from, value} turns)
- instruction: prompt, response, and optionally input (extra context appended to the prompt) and system
- text: text (raw continuation text; for style/domain adaptation only)
- preference: prompt, chosen, rejected (for DPO)

Pick the format that preserves the most useful signal. Add a constant system prompt ("=...") only
if it clearly helps the project's goal. Set min_chars to drop trivial responses and max_chars to
drop outliers that waste memory. Explain your choice in one or two sentences."""

SCHEMA = {
    "type": "object",
    "properties": {
        "format": {"type": "string", "enum": ["chat", "instruction", "text", "preference"]},
        "messages": {"type": "string"},
        "prompt": {"type": "string"},
        "response": {"type": "string"},
        "input": {"type": "string"},
        "system": {"type": "string"},
        "text": {"type": "string"},
        "chosen": {"type": "string"},
        "rejected": {"type": "string"},
        "min_chars": {"type": "integer"},
        "max_chars": {"type": "integer"},
        "rationale": {"type": "string"},
    },
}

MAPPING_KEYS = ("messages", "prompt", "response", "input", "system", "text", "chosen", "rejected")


def _clip(row: dict, n: int = 400) -> dict:
    return {k: (v[:n] + "…" if isinstance(v, str) and len(v) > n else v) for k, v in row.items()}


def run(dataset_id: int, *, use_llm: bool = True) -> Proposal:
    with Session(engine()) as s:
        ds = s.get(Dataset, dataset_id)
        if ds is None:
            raise ValueError(f"dataset {dataset_id} not found")
    on_event = event_logger(ds.project_id, AGENT)
    rows = read_raw(Path(ds.raw_path), limit=5)
    heuristic = fmt.guess_mapping(ds.columns)
    mapping, rules, rationale = heuristic, {}, "Heuristic mapping from column names."

    if use_llm:
        ctx = project_context(ds.project_id)
        user = (
            f"Project goal: {ctx['goal']}\nDataset: {ds.name} ({ds.source}: {ds.source_ref})\n"
            f"Columns: {ds.columns}\nHeuristic guess: {json.dumps(heuristic)}\n\n"
            "Sample rows:\n" + "\n".join(json.dumps(_clip(r), ensure_ascii=False, default=str) for r in rows)
        )
        try:
            out = get_provider().json(SYSTEM, user, SCHEMA)
            candidate = {"format": out["format"]} | {k: out[k] for k in MAPPING_KEYS if out.get(k)}
            check = preview_mapping(ds, candidate)
            if check["failed"] > check["sampled"] // 2:
                on_event(
                    "error", {"text": f"LLM mapping failed on sample rows ({check['errors'][:2]}); using heuristic"}
                )
            else:
                mapping, rationale = candidate, out.get("rationale", "")
                rules = {k: out[k] for k in ("min_chars", "max_chars") if isinstance(out.get(k), int) and out[k] > 0}
        except Exception as e:  # provider down: the heuristic still gives the user something
            on_event("error", {"text": f"LLM unavailable ({e}); using heuristic mapping"})

    preview = preview_mapping(ds, mapping)
    on_event("message", {"text": f"Mapping for {ds.name}: {json.dumps(mapping)}. {rationale}"})
    return create_proposal(
        ds.project_id,
        AGENT,
        "prepare_dataset",
        f"Prepare {ds.name} as {mapping.get('format')} data",
        rationale,
        {"dataset_id": ds.id, "mapping": mapping, "rules": rules, "preview": preview["records"]},
    )
