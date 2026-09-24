"""Specialists the Tuner delegates to, as agents-as-tools on the OpenAI Agents SDK.

The Tuner stays in charge of the conversation; a specialist does one noisy, parallelisable job and
hands back a compact, structured result, so the Tuner's own context holds a table rather than
seventeen search dumps. Each specialist's tool calls are recorded in the transcript under its name,
so the console shows what it did.

- DataScout: finds public datasets for a brief, previewing candidates in parallel, and returns a
  ranked shortlist with licence, columns, a suggested mapping and a fit score.
- DataPrep: inspects one imported dataset and returns a mapping and cleaning plan, checked against
  real rows before it answers.
"""

import asyncio
import functools
from dataclasses import dataclass, field

from agents import Agent, AgentOutputSchema, ModelSettings, RunContextWrapper, Runner, function_tool
from pydantic import BaseModel

from slm.config import get_settings
from slm.data import format as fmt
from slm.data import scout_tools

model_override = None  # tests inject a scripted model


# ── what they return ────────────────────────────────────────────────────────


class DatasetCandidate(BaseModel):
    repo_id: str
    config: str | None = None
    split: str = "train"
    licence: str = ""
    commercial_ok: bool | None = None
    rows: int | None = None
    columns: list[str] = []
    suggested_mapping: dict[str, str] = {}
    answer_length: str = ""  # e.g. "one sentence", "paragraphs"
    fit_score: int = 0  # 0–10 for this brief
    why: str = ""
    caveats: str = ""


class ScoutReport(BaseModel):
    candidates: list[DatasetCandidate] = []
    best: str | None = None
    summary: str = ""


class PrepPlan(BaseModel):
    mapping: dict[str, str] = {}
    min_chars: int = 2
    max_chars: int | None = None
    max_seq_length: int = 512
    expected_kept_fraction: float = 1.0
    notes: str = ""


@dataclass
class SpecialistContext:
    project_id: int
    agent: str
    calls: list[dict] = field(default_factory=list)


Ctx = RunContextWrapper[SpecialistContext]


def _record(ctx: Ctx, name: str, args: dict, output) -> None:
    """One transcript row per specialist tool call, so the console shows the delegate's work."""
    from slm.tuner.session import save_message

    ctx.context.calls.append({"tool": name, "args": args})
    save_message(
        ctx.context.project_id,
        "tool",
        name,
        {"name": name, "agent": ctx.context.agent, "args": args, "status": "done", "output": str(output)[:1500]},
    )


def _tool(fn):
    """A specialist tool: runs in a thread (Hub calls block), records itself, non-strict schema."""

    @functools.wraps(fn)
    async def run(ctx: Ctx, *args, **kwargs):
        out = await asyncio.to_thread(fn, ctx, *args, **kwargs)
        _record(ctx, fn.__name__, kwargs, out)
        return out

    return function_tool(run, strict_mode=False)


def _clip(v, n: int = 300):
    if isinstance(v, str):
        return v if len(v) <= n else v[:n] + "…"
    if isinstance(v, list):
        return [_clip(x, n) for x in v[:12]]
    if isinstance(v, dict):
        return {k: _clip(x, n) for k, x in v.items()}
    return v


# ── DataScout ───────────────────────────────────────────────────────────────


@_tool
def search_datasets(ctx: Ctx, query: str) -> list[dict]:
    """Search Hugging Face datasets: id, licence, size, tasks, description. Try several distinct
    queries (task words, domain words, formats): the Hub matches names only."""
    return _clip(scout_tools.search_datasets(query, limit=8), 200)


@_tool
def dataset_card(ctx: Ctx, repo_id: str) -> str:
    """The dataset's README: provenance, licence, intended use, known issues."""
    return _clip(scout_tools.dataset_card(repo_id, max_chars=3000), 3000)


@_tool
def preview_dataset(ctx: Ctx, repo_id: str, config: str | None = None, split: str | None = None) -> dict:
    """First rows, columns and a suggested mapping. Preview every candidate you rank; several at
    once is fine."""
    data = scout_tools.preview_rows(repo_id, config, split, n=3)
    data["suggested_mapping"] = fmt.guess_mapping(data["columns"])
    return _clip(data, 350)


SCOUT_INSTRUCTIONS = """You are DataScout. Find the best public datasets on the Hugging Face Hub for a
brief and return a ranked shortlist. Work like this:
1. Run 3–5 searches with different angles (task words, domain words, formats). Don't stop at one.
2. Preview the promising ones (you may preview several at once) and read the card of any you'd rank
   in the top three, for licence and provenance.
3. Rank by fit to the brief: on-goal content, an answer form that maps onto the target format (or
   can with a template), answer length near the target, licence (prefer permissive; flag
   non-commercial), size (a few thousand good rows beat a million noisy ones), and whether it is
   clean. Score 0–10 and say why in one line, with caveats.
4. Return 2–4 candidates, best first, each with its columns, suggested mapping, rows if known,
   licence and commercial_ok. best is the repo_id you'd import, or null if nothing fits well
   (say so in summary; the caller will write examples instead). Never invent a dataset id: only
   ones a search returned."""

PREP_INSTRUCTIONS = """You are DataPrep. Given one imported dataset's columns and sample rows, and the
project's goal, output format and target answer length, decide how to turn rows into training
records and how to clean them:
- mapping: format chat | instruction | text | preference, with the column for each field. A value
  may be a constant written as "=text" and may include {column} placeholders, e.g. the prompt
  "=How do I make {title}?" when the prompt column isn't phrased like a user would.
- min_chars / max_chars on the answer to drop trivial and outsized rows; max_seq_length to cover
  the typical row (512 for short answers, 1024 for paragraphs).
- Always call check_mapping before answering, and fix the mapping if rows fail.
- expected_kept_fraction: your estimate after cleaning; notes: one or two sentences of reasoning."""


@_tool
def sample_rows(ctx: Ctx, dataset_id: int, n: int = 6) -> dict:
    """Columns and sample rows of an imported dataset in this project."""
    from pathlib import Path

    from sqlmodel import Session

    from slm.db import Dataset, engine

    with Session(engine()) as s:
        ds = s.get(Dataset, dataset_id)
        if ds is None or ds.project_id != ctx.context.project_id:
            raise ValueError("no such dataset in this project")
        columns, raw_path, rows_total = ds.columns, ds.raw_path, ds.n_rows
    rows = scout_tools.read_raw(Path(raw_path), limit=max(1, min(n, 12)))
    return _clip(
        {"columns": columns, "rows_total": rows_total, "sample": rows, "guess": fmt.guess_mapping(columns)}, 400
    )


@_tool
def check_mapping(ctx: Ctx, dataset_id: int, mapping: dict) -> dict:
    """Apply a mapping to sample rows: how many map, the errors, and one resulting record."""
    from sqlmodel import Session

    from slm.data.pipeline import preview_mapping
    from slm.db import Dataset, engine

    with Session(engine()) as s:
        ds = s.get(Dataset, dataset_id)
        if ds is None or ds.project_id != ctx.context.project_id:
            raise ValueError("no such dataset in this project")
    check = preview_mapping(ds, mapping, n=5)
    return _clip(
        {
            "sampled": check["sampled"],
            "failed": check["failed"],
            "errors": check["errors"][:3],
            "record": check["records"][:1],
        },
        400,
    )


def build_scout(model=None) -> Agent:
    return Agent(
        name="DataScout",
        instructions=SCOUT_INSTRUCTIONS,
        model=model or model_override or get_settings().openai_model,
        tools=[search_datasets, dataset_card, preview_dataset],
        output_type=AgentOutputSchema(ScoutReport, strict_json_schema=False),
        model_settings=ModelSettings(parallel_tool_calls=True),  # previews run side by side
    )


def build_prep(model=None) -> Agent:
    return Agent(
        name="DataPrep",
        instructions=PREP_INSTRUCTIONS,
        model=model or model_override or get_settings().openai_model,
        tools=[sample_rows, check_mapping],
        output_type=AgentOutputSchema(PrepPlan, strict_json_schema=False),
        model_settings=ModelSettings(parallel_tool_calls=False),
    )


async def _run(agent: Agent, brief: str, ctx: SpecialistContext, max_turns: int) -> tuple[object, list[dict]]:
    result = await Runner.run(agent, brief, context=ctx, max_turns=max_turns)
    return result.final_output, ctx.calls


def scout(project_id: int, brief: str) -> tuple[ScoutReport, list[dict]]:
    """Run DataScout on the Tuner's loop (it shares the SDK client) and return its report."""
    from slm.tuner.session import tuner

    ctx = SpecialistContext(project_id=project_id, agent="DataScout")
    out, calls = tuner.run_coroutine(_run(build_scout(), brief, ctx, max_turns=18))
    return out, calls


def prep(project_id: int, brief: str) -> tuple[PrepPlan, list[dict]]:
    from slm.tuner.session import tuner

    ctx = SpecialistContext(project_id=project_id, agent="DataPrep")
    out, calls = tuner.run_coroutine(_run(build_prep(), brief, ctx, max_turns=10))
    return out, calls
